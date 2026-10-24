# How the curve, pseudo-target and losses work in this repo

Reference for any change to the tone curve, the pseudo-target, the colour basis, the losses or the optimisers. It describes what the **code** does, which differs from the paper in places (see `repo_vs_paper.md`). The long version with diagrams and a worked example is sections 6.1 to 6.7 of `Luminance_GS_Research_Project_Explained.md`.

Paths are relative to `Luminance-GS/examples/`. Line numbers are for `simple_trainer_ours.py` on `rung05-diagnostics` (2026-10-23) and will drift, so search by function name first.

## One training step

Batch size 1: one training view k per step.

1. **Render twice** (`Runner.rasterize_splats`, about line 429; `gsplat/rendering_double.py:rasterization_dual`). The same Gaussians are rasterised with two colour sets:
   - base SH colours `c` give the **enhanced** render `colors_enh` (used at test time);
   - `c_low = c * adjust_k + adjust_b` (per Gaussian, per channel; starts at k = 1, b = 0) gives the **dark** render `colors_low`.
2. **Make the pseudo-target** with `Runner.curve_forward(pixels, camtoworlds, image_ids)` (about line 483). It returns `pixels_enh`, `curve_adj` (the view curve L_k), `pesdo_curve` (the prior P_k) and the raw curve pieces. Training and `diagnostics.py` both call it, so they cannot drift apart.
3. **Losses** (about lines 662 to 679):

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
| `loss_regress_low` | 0.8·L1 + 0.2·(1 − SSIM) between `colors_low` and the dark photo | geometry, SH colours, `adjust_k/b` |
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

## What Rung 2 changes (planned)

Rung 2 replaces L_g, the bias net, the shape net and the colour basis with `L_k(x) = F(min(1, 2^{s_k} x))`:

- one shared monotone curve F: K = 32 softplus increments, knots dense near black, F(0) = 0, F(1) = 1, linear interpolation so gradient can reach `s_k`;
- one exposure `s_k` per view, a free scalar with mean fixed at 0, so it is relative exposure and F carries the overall brightening;
- a fixed RGB basis: F is applied to R, G and B directly (Rung 0.5 showed the learned basis costs nothing to remove once the curve is detached).

L_hist becomes L_curve: the MSE between F(min(1, 2^{s_k} x_b)) and CDF_k on the 255-bin grid, plus a smoothness term on F. The rendering, the Gaussians and L_low, L_enh and L_spa stay as they are. Full design: section 8.6 of the explainer. Decisions and ablation order: `Luminance-GS/Implementation_guides/Rung0.5_interpretation.md`. Flag names are decided in the Rung 2 plan.

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
