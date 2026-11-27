# How the curve, pseudo-target and losses work in this repo

Reference for any change to the tone curve, the pseudo-target, the colour basis, the losses or the optimisers. It describes what the **code** does, which differs from the paper in places (see `repo_vs_paper.md`). The long version with diagrams and a worked example is sections 6.1 to 6.7 of `Luminance_GS_Research_Project_Explained.md`.

Paths are relative to `Luminance-GS/examples/`. Line numbers are for `simple_trainer_ours.py` on `main` with the M2 part C changes (2026-11-13) and will drift, so search by function name first.

## One training step

Batch size 1: one training view k per step.

1. **Render twice** (`Runner.rasterize_splats`, about line 487; `gsplat/rendering_double.py:rasterization_dual`). The same Gaussians are rasterised with two colour sets:
   - base SH colours `c` give the **enhanced** render `colors_enh` (used at test time);
   - `c_low = c * adjust_k + adjust_b` (per Gaussian, per channel; starts at k = 1, b = 0) gives the **dark** render `colors_low`.
2. **Make the pseudo-target** with `Runner.curve_forward(pixels, camtoworlds, image_ids)` (about line 541). It returns `pixels_enh`, `curve_adj` (the view curve L_k), `pesdo_curve` (the prior P_k) and the raw curve pieces. Training and `diagnostics.py` both call it, so they cannot drift apart. With `--curve_model rung2` it hands over to `Runner.rung2_forward`, which returns the same keys (see "Rung 2 as built").
3. **Losses** (about lines 740 to 758):

   ```
   loss = loss_regress_low + 0.5 * loss_regress_enh + loss_co + 10 * hist_loss
   ```

At test time only the Gaussians' enhanced colours are rendered. The curve, the networks, the colour basis and the dark photos are not used. **The curve's only job is to make training targets**, and whatever brightness and colour the targets have gets baked into the Gaussians.

## Inside `curve_forward`

```
dark photo I_k ──► CrossAttention (bias net, 143,743 params) ──► L_b,k  (255 values)
pose (4x4)     ──┘
global curve L_g (255 learnable values) ──► L_k = clamp(L_g + L_b,k, 0, 1)      = curve_adj

I_k ──► pixel_project (rotate RGB into the view's basis n1, n2, n3; min-max encode each axis to [0, 1])
    ──► LUT_mapping on each of the 3 channels: out = L_k[ int(254 * t) ]
    ──► pixel_project_back (decode, rotate back to RGB) ──► pixels_enh

I_k, pose ──► CrossAttention_Curve (shape net, 111,235 params) ──► (g, a, b)
gamma = 1 + 0.1 g,  alpha = 0.5 + 0.002 a,  beta = 1 + 0.002 b
P_k = s_curve(gamma_curve(linspace(0, 1, 255), gamma), alpha, beta)              = pesdo_curve
```

- **Global curve** `self.curve`: 255 values, initialised to the identity `linspace(0, 1, 255)`.
- **Bias network** `utils.py:CrossAttention`: conv 3→64→128, pooled to 16 image tokens (keys and values); the flattened camera-to-world matrix is the query; output Linear 128→255. Xavier init with gain 0.01, so L_k starts near the identity.
- **View curve:** `curve_adj = clamp(curve + bias, 0, 1)`. **Nothing forces it to increase**; the clamp is the only constraint.
- **Colour basis** `self.axis1_para[k]` (3 values) and `self.axis2_para[k]` (2 values), both initialised to zero, so the basis starts as plain R, G, B. `tools.py:pixel_project` builds an orthonormal basis from them: n1 = (1,0,0) + a1; n2 is constructed perpendicular to n1 from (1,0) + a2; n3 = n1 × n2; all normalised. That is 5 numbers but only 3 real degrees of freedom (a rotation). Each projected channel is min-max encoded using the range that axis spans over the RGB cube.
- **Lookup** `tools.py:LUT_mapping`: `idx = int(254 * clamp(t, 0, 1))` (truncation, no interpolation); the **same** L_k is used for all three projected channels.
  - Gradient reaches the curve entries (summed over the pixels that land in each bin) but **never the pixel values**, because the index is an integer.
  - Pixels use only 14 to 25% of the bins (Rung 0), so most of the curve gets no image gradient and is shaped only by the curve losses.
- **Back to RGB** `tools.py:pixel_project_back`: decode with each axis's max/min, recombine along n1, n2, n3. **This decode is the colour basis's only gradient path**, and it reaches a loss only through `pixels_enh`.
- **Shape prior** `losses.py:gamma_curve` and `s_curve`. `gamma_curve` clamps its input at 1e-3, so with gamma ≈ 0.1 it maps black to 0.30 to 0.64 (the lifted black level). The ×0.002 scaling means alpha and beta barely move (Rung 0: 0.506 to 0.510 and 1.004 to 1.010), so in practice P_k is a one-parameter gamma curve per view.

## The losses

| Term | Formula in code | Trains |
|---|---|---|
| `loss_regress_low` | 0.8·L1 + 0.2·(1 − SSIM) between `colors_low` and the dark photo (with `--dark_branch` other than `shared`: between 2^{−s_k}·`colors_low` and the photo; see "The dark branch") | geometry, SH colours, `adjust_k/b` |
| `loss_regress_enh` (×0.5) | same form, between `colors_enh` and `pixels_enh` | geometry, SH colours; **when attached** also the curve, bias net and colour basis |
| `loss_co` = `losses.py:L_spa` | channel-mean, 4×4 avg-pool, 4-neighbour differences: mean of (c·Δ dark − Δ enhanced)², c = `constrast_level` / mean(dark) = 0.5 / mean(dark) | geometry, SH colours |
| `hist_loss` (×10) = `losses.py:HistogramPriorLoss` | w·mean((L_k − CDF_k)²) + 0.1·mean((L_k[i+1] − L_k[i])²) + 0.5·[mean((P_k − L_k)²) + 0.01·mean((P_k − CDF_k)²)] | global curve, bias net, shape net |

- `CDF_k` is the histogram-equalisation curve of the dark photo: channel mean of a ¼-size image, 255 bins.
- w = 1 for steps < 3000, then 0.5 for low light (0.1 only for `over_exp`).
- Because of the ×10, the histogram terms dominate: about 5× all image terms combined at the end of training (Rung 0, buu). The curve is shaped mainly by the priors.

## Who gets gradient from what

| Parameters | L_low | L_enh | L_spa | L_hist |
|---|---|---|---|---|
| Gaussian geometry | yes | yes | yes | no |
| SH colours (= enhanced colours) | yes (through c_low) | yes | yes | no |
| `adjust_k`, `adjust_b` | yes | no | no | no |
| Global curve, bias net | no | attached only | no | yes |
| Shape net (gamma, alpha, beta) | no | no | no | yes |
| **Colour basis `axis1/axis2`** | no | **attached only; its only source** | no | no |

The flags that cut these paths (all default off, so the default is the official code):

| Flag | Where | Effect on gradient from L_enh |
|---|---|---|
| `--detach_pseudo_target` | `train()`, `pixels_enh.detach()` | cuts the curve **and** the colour basis (which then never moves from RGB, because Adam skips parameters with no gradient) |
| `--freeze_color_basis` | `curve_forward`, `axis.detach()` | cuts the colour basis only; an end-of-training assertion checks it stayed exactly zero |
| `--detach_curve_in_target` | `curve_forward`, `curve_lut = curve_adj.detach()` | cuts the curve only; `hist_loss` still gets the attached `curve_adj` |
| `--grad_check` | `Runner.grad_check` | prints, at the first step, which of curve / basis get gradient from L_enh |

This is the Rung 0.5 2×2: `baseline`, `frozenbasis`, `detachcurve` and `detach` are the four cells.

## Optimisers and schedule

| Group | Parameters | Adam lr | Weight decay | Decays to 1% over training |
|---|---|---|---|---|
| Gaussians | means 1.6e-4 × scene scale, scales 5e-3, quats 1e-3, opacities 5e-2, sh0 2.5e-3, shN 2.5e-3/20, adjust_k/b 2.5e-3 | as listed | 0 | means only |
| `curve_optimizers` | global curve | 1e-3 | 1e-4 | yes |
| `adjust_optimizers` | bias net and shape net | 1e-5 | 1e-5 | yes |
| `sat_optimizers` | colour basis, all views | 2e-4 | 1e-4 | yes |

Over 10k steps: the SH degree rises every 1000 steps up to 3; densification runs every 100 steps from step 500 to 8000, with opacity resets at 3000 and 6000; the histogram weight w drops at step 3000; gamma falls from 1 to about 0.1 between steps 1000 and 3000, then stays flat.

## The dark branch (M2 part A, 2026-11-11)

Each Gaussian has one dark colour, `c·adjust_k + adjust_b`, for every view, so `loss_regress_low` asks one dark render to match every dark photo. That works on LOM. On the varying-exposure set (photos up to 2 stops apart), the trained fit drops by 3.6 to 11.9 dB per scene (M1).

`--dark_branch` (default `shared`, the official code) multiplies view k's rendered dark image by 2^{−s_k}. This happens in `train()`, right after the render split and before `random_bkgd` and `loss_regress_low`. Compositing is linear and the background is black, so this equals scaling every Gaussian's dark colour for that view.

| `--dark_branch` | s_k | Code |
|---|---|---|
| `shared` (default) | none | the official code; nothing is computed |
| `measured` | log2(m̄/m_k), fixed. m_k is the training view's mean as loaded (undistorted, /255), m̄ the geometric mean | `Runner.measure_exposures`; also written to `<result_dir>/exposure_init.json` |
| `model` | Rung 2's s_k, `.detach()` | needs `--curve_model rung2` (built in part C, 2026-11-13; variant `rung2exp`) |
| `gains` | −(log2_ratio − mean) from `<data_dir>/exposure_gains.json`, the set's true gains | `Runner.load_gains`; diagnosis only |

- **One function for training and diagnostics.** `Runner.dark_scale(image_ids)` returns the factor (or `None`) and is shared by `train()` and `diagnostics.py`, so the two cannot drift apart.
- **No gradient through the factor.** `loss_regress_low` still trains geometry, SH colours and `adjust_k/b`, but each view's gradient is multiplied by 2^{−s_k}. No exposure learns from it.
- **exp(−ln2·s), not `torch.exp2`.** torch 1.13 compiles `exp2`'s CUDA kernel at run time (NVRTC), which fails in this environment (`nvrtc-builtins64_117.dll` missing).
- **No random numbers.** `measure_exposures` reads `trainset[i]` (no DataLoader), so the view order is unchanged.
- **Diagnostics:** `pseudo_targets.csv` gets two new last columns.
  - `s_dark`: 0 for `shared`.
  - `fit_psnr_lowrender_exp_vs_input`: the dark render as its loss sees it; equal to `fit_psnr_lowrender_vs_input` for `shared`.
- **Re-scoring:** `rescore_runs.py` passes `--dark_branch` on from each run's `cfg.json` (`MODEL_FLAGS`). `diagnostics._load_run_state` asserts that the re-score uses the run's value (`MODEL_DEFAULTS`).
- **Variant:** `detachexp` in `run_rung0.py` = `--detach_pseudo_target --dark_branch measured`.
- **Built-in check** (smoke test, buu, 300 steps): `detach` and `detachexp` with the same seed produced identical pseudo-targets. Nothing in the curve path sees the dark render.

## What Rung 2 changes (planned; built 2026-11-13)

Rung 2 replaces L_g, the bias net, the shape net and the colour basis with `L_k(x) = F(min(1, 2^{s_k} x))`:

- one shared monotone curve F: K = 32 softplus increments, knots dense near black, F(0) = 0, F(1) = 1, linear interpolation so gradient can reach `s_k`;
- one exposure `s_k` per view, a free scalar with mean fixed at 0, so it is relative exposure and F carries the overall brightening;
- a fixed RGB basis: F is applied to R, G and B directly (Rung 0.5 showed the learned basis costs nothing to remove once the curve is detached).

L_hist becomes L_curve: the MSE between F(min(1, 2^{s_k} x_b)) and CDF_k on the 255-bin grid, plus a smoothness term on F. The rendering, the Gaussians and L_low, L_enh and L_spa stay as they are. Full design: section 8.6 of the explainer. Decisions and ablation order: `Luminance-GS/Implementation_guides/Rung0.5_interpretation.md`. Flag names are decided in the Rung 2 plan.

*Update (2026-11-11):*
- **Flag names:** the Rung 2 flag names are now set in `Luminance-GS/Implementation_guides/M2_rung2_plan.md` section 2.2, and go here once the code exists.
- **L_curve** keeps the official smoothness term, 0.1·mean of the squared first differences of the composed curve. It is L_hist minus its shape-prior term. *(Superseded 2026-11-13: the shape-prior term is kept, with one global γ on F; see "Rung 2 as built".)*
- **L_low is no longer fixed.** On the varying-exposure set it can be exposure-aware ("The dark branch" above; `rung2exp` uses Rung 2's own s_k with stop-gradient).
- **Target alignment (decided by Bart, 2026-11-11):** the CDF target is shifted by one bin, T_k = [0, CDF_k[0], …, CDF_k[253]], in the loss and the `cdf` initialisation. Rung 2 then reads the same bin as the official truncating lookup for every level ≥ 1, and level 0 stays black (plan section 2.2).

### Rung 2 as built (M2 part C, 2026-11-13)

`--curve_model rung2` (default `official`). The `--rung2_*` flags act only then.

| Flag | Default | Meaning |
|---|---|---|
| `--rung2_knots` | 32 | K, the segments of F |
| `--rung2_knot_spacing` | `quadratic` | y_j = (j/K)²; also `uniform`, `usedrange` (the step 6 rule kept quadratic) |
| `--rung2_F_init` | `cdf` | the views' shifted HE-CDFs aligned by their initial s_k and averaged; also `identity` |
| `--rung2_s_init` | `measured` | log2(m̄/m_k), the same numbers as `--dark_branch measured`; also `zero` |
| `--rung2_attach` | `none` | gradient from L_enh: `none`, `s` (H3; variant `rung2att`), `all` |
| `--rung2_prior` | `F` | the shape-prior term (below): `F`, `composed` (the official form), `none` (the first draft; M5 ablation) |
| `--rung2_smooth_lambda` | 0.1 | first-difference smoothness of the composed curve |
| `--rung2_lr_F`, `--rung2_lr_s` | 1e-2, 1e-3 | Adam on (θ, ρ) and on s; no weight decay; ExponentialLR to 1% |
| `--rung2_color` | `rgb` | E5 (2026-11-25): how L_k maps a pixel. `rgb`: on R, G and B separately. `luminance`: on the channel-mean grey Y = (R+G+B)/3, out_c = clamp(x_c · L_k(Y) / max(Y, 1e-6), 0, 1) (variant `rung2idlum`, with the identity init). Refused without `--curve_model rung2` |

- **Curve loss (decided by Bart, 2026-11-13):** L_curve = w(t)·mse(L_k, T_k) + 0.1·mean(L_k[b+1] − L_k[b])² + 0.5·[mse(p, F) + 0.01·mse(p, T_k)], with p = clamp(x_b, 1e-3, 1)^γ (`losses.gamma_curve`; the clamp only touches b = 0) and one learned global γ = exp(ρ), starting at 1. The total loss uses 10·L_curve, as the official uses 10·L_hist.
  - This is the official L_hist with the shape network replaced by one γ, and the prior moved from each view's curve to the shared F. So the prior never trains s_k.
  - Why: from step 3000 the official loss puts the curve halfway between the HE-CDF and the gamma prior, and `detach`'s curves sit there. Without the term, Rung 2 would fit a different target (research log, 2026-11-11 part B and 2026-11-13).
  - Learning rate 1e-2 for F and γ: plan step 6's rule, applied with the prior on (log, 2026-11-13).
- **Code:**
  - `Runner.build_rung2`: the init. It reads `trainset[i]` (no random numbers) and prints the parameter count, K + N + 1.
  - `Runner.rung2_forward`: `curve_forward`'s Rung 2 branch, with the same keys. `curve_adj` = the composed curve, `pixels_enh` = `map_pixels` under `--rung2_attach` and `--rung2_color`, `pesdo_curve` = p, `gamma` = γ, `projected` = (R, G, B), or (Y, Y, Y) with `--rung2_color luminance`; official-only keys are zeros or NaN.
  - `--rung2_color luminance` (E5) changes only the pseudo-target: Y is the grey the HE-CDF target is built on (`losses.py` line 56), so F, s and γ train exactly as with `rgb` (the curve loss never sees the pseudo-target in the detached default). `ExposureCurve.clipped_share` gives the share of channel values the pseudo-target clips. Design and readings: `E5_luminance_plan.md`.
  - `train()`: target = `hist_target` of the step's HE-CDF, then `ExposureCurve.loss`; the gauge projection after the optimiser step; end-of-run asserts on mean s and a monotone F.
  - `grad_check_rung2`: gradient norms from L_enh and L_low into θ, s and ρ.
- **Not built in Rung 2 mode:** `self.curve`, the bias and shape networks, `pesdo_curve`, the colour basis and their optimisers. Refused with `--detach_pseudo_target`, `--detach_curve_in_target`, `--freeze_color_basis`, `--patch_size`, or a batch size other than 1.
- **Diagnostics:**
  - `curves_step*.npz` keeps every key (official-only ones zeros or NaN; `curve_global` = F on the grid) and adds `s`, `s_init`, `F_knots` and `F_heights`.
  - New files: `diagnostics/rung2_state.json` (view names, s_init, s_final, knots, heights, γ, flags, `n_params`) and `diagnostics/exposures_by_step.csv` (step, view, s; step 0 is s_init).
  - `curve_state.pt` holds the module's state, and `_load_run_state` restores it, knots included.
- **Re-scoring:** `rescore_runs.MODEL_FLAGS` passes `curve_model` and every `rung2_*` flag on from `cfg.json`. `diagnostics.MODEL_DEFAULTS` mirrors the `Config` defaults, and the trainer asserts this at start-up.
- **Checks (2026-11-13, research log):**
  - 14 unit tests pass.
  - Re-scores of an official and a `detach` run with the new code are bit-identical to their old outputs.
  - `rung2` and `rung2exp` with the same seed give bit-identical F, s and pseudo-targets.
  - Gradient flow is as designed.

**Default mode: detached** (chosen by Rung 0.5, where cutting L_enh from the curve explained 85 to 100% of the detach gain).

| Parameters | L_curve | L_enh, default | L_enh, attached-`s_k` ablation |
|---|---|---|---|
| F knot heights | yes | no | no |
| `s_k` | yes | no | yes |

- **In the default, nothing rendered reaches F or `s_k`.** The pseudo-targets depend only on the dark photos and the training step, so F and `s_k` could even be fitted before training. 3D consistency has to come from the shared F.
- **Attached-`s_k` ablation:** evaluate F with detached knot heights but a differentiable input y = 2^{s_k} x, so L_enh reaches `s_k` only (∂F/∂s_k = F'(y)·y·ln 2). One number per view cannot recolour part of an image, so the hue flip should not return.
- **Colour-preserving ablation:** apply F to the luminance Y and scale RGB, `out = rgb · F(2^{s_k} Y) / max(Y, ε)`, clipped to [0, 1]. This keeps hue and saturation, apart from clipping in highlights.
- **Brightness:** with mean `s_k` = 0, global brightness is set by F's fit to the HE-CDFs and, for the render, by L_spa. The exposure model does not remove the offset from the ground truth's brightness, so brightness-matched PSNR stays the primary metric.
- **Noise:** with F detached, its dark-end slope is the only thing that sets how much each photo's noise is amplified. Check that the smoothness weight keeps F off the HE-CDF's steep jump near black.

## Rung 3 as built (M6 step 2, 2026-11-26)

`--rung3_zones 2` (default 1 = Rung 2, and then no Rung 3 code runs), with `--curve_model rung2 --rung2_F_init identity`; variant `rung3`. Design and readings: `Luminance-GS/Implementation_guides/M6_rung3_plan.md`, pre-registered on 2026-11-26.

| Flag | Default | Meaning |
|---|---|---|
| `--rung3_zones` | 1 | 1: Rung 2. 2: Rung 3, two zone curves. Any other value is refused |
| `--rung3_blur` | 16 | block size in pixels of the guide's area average |

- **Model** (`rung3.py`, `ZoneExposureCurve`): L_k(x, p) = w_sh(p)·F_sh(min(1, 2^{s_k} x)) + w_lit(p)·F_lit(min(1, 2^{s_k} x)), on R, G and B.
  - Two Rung 2 curves on the same knots (θ is [2, K], identity init), one s_k per view (mean 0), one shared γ.
  - 2K + N + 1 parameters: 87 to 108 per scene.
- **Weights, fixed per view at s_init (decided 2026-11-26):** `rung3.build_zones`, run on the CPU by `Runner.build_rung3`.
  - The guide is g = log2(max(G, 1/255)), with G the bilinear upsampling of the adaptive average pool, to (H // 16, W // 16), of min(1, 2^{s_init,k} Y). Y is the channel-mean grey.
  - w_lit = clamp((g − c_1)/(c_2 − c_1), 0, 1) and w_sh = 1 − w_lit. c_1 and c_2 are the 25th and 75th percentiles of g over every training pixel (numpy float64).
  - The weights are buffers with no gradient. The learned s_k still enter the curves.
- **Zone targets and shares**, computed once per view: weighted histograms on `compute_histogram_equalization`'s grey and 1/4 grid, with histc's float32 bins and float64 sums. T_k,m = `hist_target`(CDF_k,m), and π_k,m is the zone's share of the pixels.
  - With one zone this reproduces Rung 2's target exactly.
- **Curve loss:** Σ_m π_k,m·`rung2.curve_loss`(L_k,m, T_k,m, prior x^γ on F_m). The rest is as in Rung 2: ×10, Adam at 1e-2 on (θ, ρ) and 1e-3 on s, the gauge, detached.
- **Code:**
  - `Runner.build_rung3` draws no random numbers, so the view order at a given seed is `rung2id`'s. It prints the centres and the shares.
  - `Runner.rung3_forward` returns `rung2_forward`'s keys plus `zone_curves`, `zone_targets` and `zone_shares`. Its `curve_adj` is Σ_m π_m·L_k,m, and `pixels_enh` is the detached mix.
  - `train()` has a zone-loss branch. At the end of a run it prints both curves at levels 1–64.
  - Method names differ from `ExposureCurve`'s (`F_zones`, `zone_heights`, `composed_curves`), so a Rung 2 code path fails instead of misreading.
- **Refused with `--rung3_zones 2`:**
  - `--curve_model` other than `rung2`, and `--rung2_F_init cdf`;
  - `--rung2_attach` other than `none`, `--rung2_color` other than `rgb`, `--rung2_prior` other than `F`;
  - `--rung2_s_init zero`, `--rung2_knot_spacing usedrange`, `--dark_branch` other than `shared`;
  - c_2 − c_1 below 0.05 stops.
- **Diagnostics: per-zone arrays under new keys only.**
  - npz: `curve_global_zones`, `F_heights_zones`, `zone_centres`, `zone_curves`, `zone_targets` and `zone_shares`. No `curve_global` or `F_heights`.
  - `rung2_state.json`: `F_grid_zones`, `heights_zones`, `zones`, `blur`, `centres` and `zone_shares`. No `F_grid` or `heights`.
  - `curve_state.pt` adds `zones` and the zone buffers, but not the weight maps.
  - `_load_run_state` checks the recomputed centres, targets and shares within 1e-6.
  - Each example view also gets `example_*_wlit.png`: input | w_lit | pseudo-target.
- **Re-scoring:** `MODEL_FLAGS` and `MODEL_DEFAULTS` gain `rung3_zones` and `rung3_blur`.
- **Checks (2026-11-26, research log "M6 step 2"):** `test_rung3.py` 9/9 and its `--cuda` check; `test_rung2.py` 16/16; the refusals; a 300-step check run with a re-score.
