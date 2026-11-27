# M6 plan: Rung 3, local monotone curves on the identity base

Written on the evening of 2026-11-25 in the project thread, for Claude Code on Bart's PC, after Bart's decisions of 2026-11-25 (log: "Decisions after M5") and E5's result (log: "E5 results"; `E5_results.md`). It continues `M5_plan.md` section 8 and `E5_luminance_plan.md`.

**Status: design settled and readings pre-registered (2026-11-26; log: "M6 step 1").**
- Step 0 checked this plan against the code; the corrections are applied below (log: "M6 step 0").
- Bart decided D5 and D6 at step 0: harm signal 3 is judged on the decision scenes at +20% (section 4.1), and the zone weights are fixed at each view's initial exposure s_init (section 2.1).
- On 2026-11-26 Bart also chose D1–D4 as recommended and approved section 4 as written.
- Step 2 done 2026-11-26 (log: "M6 step 2"): the code of section 3's table, apart from `rung2_replay.py --mode rung3`, which is the preview of step 3, and `rung2_ablation.py --mode rung3`, which is step 4. Tests 1 and 2 pass.
- Step 3 done 2026-11-27 (log: "M6 step 3, tests 3 and 4" and "M6 step 3, test 5"):
  - the regression is bit-identical on the curve side, and the smokes pass;
  - the preview passes its stop rule;
  - plan 4.3's predicted directions are logged with the sha256 of `results/rung3/preview/numbers.json`, now protected.
- Next: step 4, the batch, launched by Bart with `/run-batch`.

**Labels:** every number is *measured* unless marked *inferred*. Sources are named. Numbers from the quick photo check of section 1.4 are *measured* on the training photos only, with a quick script, and step 3's preview recomputes them with the trainer's loader.

---

## 0. At a glance

| | |
|---|---|
| **Question** | Does a per-pixel convex mix of two shared monotone curves, weighted by exposure-aligned local brightness, improve on Rung 2's single global curve? |
| **What it should fix** | The dark-end compromise of one global curve: today the same F lifts the dark pixels of shadowed regions and the dark pixels of coloured objects in lit regions. Bike's hue error is the measurable target (E4 and E5, section 1.3). |
| **What it cannot fix** | The per-scene saturation mismatch E5 found: it is set by the low-light photo's own saturation relative to the ground truth, which no photo-only rule can know (section 1.3). |
| **Design** | `--rung3_zones 2` (default 1 = Rung 2, code path untouched), with `--curve_model rung2 --rung2_F_init identity`. Variant `rung3`. The zone weights, zone targets and zone shares are fixed per view, from the photo at its initial exposure s_init (decided 2026-11-26). 2K + N + 1 curve parameters (87 to 108 per scene). |
| **Base** | `rung2id`: identity init, 10k steps, the same learning rates (decided 2026-11-25). Every comparison run already exists. |
| **Runs** | `rung3` on LOM and the varying set, seeds 0–2: 30 runs, about 2.3 h (*inferred*). Optional: `over_exp`, 15 runs, exploratory. |
| **Readings** | Section 4: H4 (helps, hurts, mixed, no effect), bike's hue (fixes, fixes at a cost, no fix), and a mechanism check: does validation colour move the way the photo-only preview predicts? |
| **Defaults** | None change before the freeze. Rung 3 is reported as the stretch result whatever its verdict. |

---

## 1. Where we stand, and why Rung 3 now

### 1.1 Measured so far

- **The base, `rung2id`, on LOM** (`results/rung2/summary_headstart_seeds012/tables.md`): pooled bm PSNR 23.67 ± 0.07, against `detach` 23.25. It meets the H1 rule against `detach` on seeds 0–2 (12/12, +0.415 dB, 6.0×, exploratory) and on fresh seeds 3–5 (H1b, 12/12, +0.37 dB, 3.7×, pre-registered).
- **On the varying set** (M5 E1, `results/rung2/summary_rung2id/`): curve-model effect +2.72 dB (6.8×); leak 0.009 against `detach`'s 0.512; s_k slope −1.00.
- **A1:** the identity gain is early stopping of F (gap 0.051 at 10k, 0.001 at 50k, bike). So M6 keeps 10k steps and the same learning rates.
- **E4:** without the shape prior, pooled bm PSNR falls 6.47 dB, F at levels 1–4 drops to about 0, and bike's hue error falls 27.3°. The deep-shadow lift comes from the one-γ prior, and bike's hue error goes with it.
- **E5** (`E5_results.md`): applying F to the channel-mean grey and scaling the channels removes bike's hue error (LOM 33.6° → 9.6°, varying 27.0° → 9.3°). F, s and γ were bit-identical between the arms, so per-channel application causes the hue error. But buu (both sets) and sofa (varying) lose bm PSNR beyond 2× noise, and pooled SSIM falls on both sets. Verdict "fixes at a cost", not adopted.
- **E6:** overexposure fails the H1 rule (3/12). It is reported as a limitation of the one-γ prior.

### 1.2 What Rung 3 is, and how it differs from the first sketch

The explainer's sketch (section 8.8) was: M = 4–8 shared monotone curves, mixed per pixel by weights from a small conv net. A convex mix of increasing curves is increasing, so tone order is kept at every pixel.

This plan keeps the mix and replaces the conv net with **fixed weights from the photo's exposure-aligned local brightness**, with **M = 2**. Why:
- **Nothing would train a conv net in the detached default.** F, s and γ are trained only by the curve loss, which is a histogram of each photo and has no spatial signal.
- **Attaching the weights to the image loss would bring back curve chasing.** Rung 0.5 found that the attached curve, not the colour basis, explains the detach gain and bike's hue flip. H3 (M5 E2) showed that 3D feedback into s_k does nothing; per-pixel weights would give the renders far more to chase (*inferred*).
- **Brightness zones follow the scene, not the frame.** A tile-based mix (CLAHE-style) would put a surface in a different tile in each view, so pseudo-targets would disagree across views (*inferred*). The local brightness of a surface changes much less between views, and the initial exposures s_init already align it (2.1).
- **It stays in the project's framing:** 2D enhancement from the photos' own histograms, with shared curves and one exposure per view, plus one extra curve.
- **Time:** six working days to the freeze.

### 1.3 What Rung 3 should fix, and what it cannot

**The dark-end compromise (the target).**
- One global F has to serve every dark pixel the same way. A dark pixel in a shadowed region is an underexposed surface; a dark pixel in a lit region is more often a dark or saturated object.
- E4 and E5 together say bike's hue error comes from a strongly lifting curve applied per channel (*measured*: E5 isolates per-channel application; E4 ties the error to the lift).
- **Why** (*inferred*): a curve applied per channel multiplies the log-ratios between a pixel's channels by the curve's log-slope, d log F / d log x, at those levels. Where the log-slope is below 1, as at a strongly lifted dark end, channel ratios are compressed, which desaturates and moves the HSV hue. For a power curve x^γ the log-slope is γ, and the base's prior has γ of 0.07 to 0.16 on LOM (*measured*, `rung2id` seeds 0–2: bike 0.074, shrub 0.076, chair 0.109, sofa 0.137, buu 0.165).
- Rung 3 keeps per-channel application but lets the lift depend on the neighbourhood: less lift at the dark end where the neighbourhood is lit, more where it is in shadow.
- **Photo statistic behind the expectation** (section 1.4): 74% of bike's chromatic photo pixels sit in lit neighbourhoods (w_lit > 0.5; the soft share, the mean w_lit over those pixels, is 0.73), and they sit at the dark end (median max-channel level 11 of 255; 13 with the trainer's loader). So a lit-zone curve with a larger log-slope at those levels should reduce bike's hue error (*inferred*). The direction that is logged before the batch comes from the preview's computed pseudo-targets, not from this argument (section 4.3).

**The per-scene saturation mismatch (not a target).**
- E5's saturation ratios on LOM (prediction over GT; 1 is right): per channel (`rung2id`) buu 0.87, chair 0.47, sofa 0.86, bike 0.52, shrub 0.73; luminance (`rung2idlum`) 1.26, 0.76, 1.34, 0.88, 1.07.
- Per-channel application desaturates by a similar factor everywhere: the ratio of the two is 0.59 to 0.69 in every scene (*measured*, from those numbers).
- The scene split comes from the low-light photo's own saturation relative to the GT (luminance column, 0.76 to 1.34), which differs by scene. No rule that sees only the photos can know it (*inferred*).
- So Rung 3 cannot fix the mismatch. It can only move saturation within a scene, through the curve each zone gets where the chromatic pixels sit. That direction is a mechanism check (section 4.3), not a fix.

**The dark end in general.** The identity init's gain is early stopping (A1). Rung 3 does not replace that; it runs on top of it, with the same 10k steps. Its curve loss at 10k is reported next to the base's (decision 2026-11-25), with the caveat that the zone objective is a different objective.

### 1.4 Quick photo check (2026-11-25, project thread; training photos only, no normal-light image read)

A quick script on Bart's PC read the five LOM scenes' `low/` training views (test views excluded by `transforms_test.json`), gained each by its initial exposure s_init = mean log2 m − log2 m_k, and built the guide of section 2 (channel-mean grey, block average over 16 × 16 pixels, bilinear back up, log2). Approximations: PIL's JPEG decoder and box resize instead of the trainer's loader and `interpolate(mode="area")`. Step 3's preview recomputes everything with the trainer's code.

| Scene | Views (train/test), size | Guide 25th / 75th percentile (× 255, gained grey) | Lit share | max \|CDF_shadow − CDF_lit\| | Chromatic photo pixels in lit zones (w_lit > 0.5) |
|---|---|---|---|---|---|
| buu | 22 / 3, 500 × 375 | 16.0 / 31.4 | 0.47 | 0.76 | 56% |
| chair | 43 / 5 | 7.4 / 18.7 | 0.57 | 0.72 | 79% |
| sofa | 29 / 4 | 10.5 / 24.6 | 0.49 | 0.76 | 12% |
| bike | 36 / 4 | 5.4 / 8.5 | 0.52 | 0.56 | 74% |
| shrub | 30 / 5 | 6.1 / 9.4 | 0.52 | 0.36 | 62% |

- "Chromatic" = photo saturation (max − min)/max above 0.3 with the brightest channel at level 4 or more. Bike: 13.8% of its pixels; their brightest channel has median level 11 (quartiles 6 and 23).
- **The zone targets differ strongly.** Bike's one-bin-shifted HE-CDF at level 8 is 228/255 for the shadow zone, 94 for the lit zone and 159 for the whole photo; at level 4, 41, 10 and 25.
- **The block size barely matters:** 1/8 and 1/32 give lit shares within 0.03 of 1/16, and CDF gaps within 0.1 in every scene (bike 0.63, 0.56 and 0.46 at 1/8, 1/16 and 1/32).
- *Inferred:* fitted alone, the shadow zone's target would brighten shadows far more than the lit zone's. That flattens the contrast between shadowed and lit regions, which is risk 1 (section 9). The shared γ prior limits how far the two curves move apart.
- **Step 0 check (2026-11-26; *measured* with the trainer's loader, through `rung2_preview.load_set`; training photos only):**
  - The percentiles reproduce exactly in all five scenes. Lit shares and CDF gaps agree within 0.01–0.02. Bike's zone targets are 230/94/160 at level 8 and 41/9/24 at level 4.
  - The loader's images are 374 × 499 (undistorted and cropped), not 500 × 375.
  - The last column is a hard split (w_lit > 0.5): reproduced as chair 0.80, sofa 0.12, bike 0.75. But 42–57% of those pixels sit inside the ramp (0 < w_lit < 1), where their curve is a real mix. The soft shares (mean w_lit over chromatic pixels) are chair 0.76, sofa 0.19, bike 0.73.

---

## 2. Design: `--rung3_zones {1, 2}` (default 1 reproduces Rung 2 exactly)

### 2.1 The model

```
Y_k(p)  = (R + G + B) / 3                                   the grey the HE-CDF target uses (losses.py line 56)
G_k     = up( area_avg_f( min(1, 2^{s_init,k} Y_k) ) )      f = 16 (--rung3_blur): about 16 x 16 blocks, area down,
                                                            bilinear back to H x W; s_init,k = the view's initial
                                                            exposure (Runner.measure_exposures), not the learned s_k
g_k     = log2( max(G_k, 1/255) )
w_lit   = clamp( (g_k - c_1) / (c_2 - c_1), 0, 1 ),   w_sh = 1 - w_lit
c_1,c_2 = the 25th and 75th percentiles of g over every full-resolution pixel of every training view

L_k(x, p) = w_sh(p) F_sh( min(1, 2^{s_k} x) ) + w_lit(p) F_lit( min(1, 2^{s_k} x) ),   on R, G and B
```

- **F_sh and F_lit** are two Rung 2 curves: K = 32 quadratic knots each, F(0) = 0, F(1) = 1, strictly increasing by construction, both initialised at the identity (as `rung2id`).
- **Shared:** one exposure s_k per view (mean 0, the gauge), and one shape-prior γ for both curves.
- **Monotone per pixel:** for fixed weights, L_k is a convex mix of increasing curves, so it is increasing in each channel. Black maps to 0, and input 1 maps to 1 when s_k ≥ 0. As in Rung 2, a view with s_k < 0 maps 1 to Σ_m w_m F_m(2^{s_k}) < 1.
- **The weights do not depend on the channel,** so a grey pixel stays grey.
- **The weights are fixed at s_init** (decided by Bart 2026-11-26; D6).
  - w_sh and w_lit, the centres, and the zone targets and shares of 2.2 are constants per view. They are computed once in build, on the CPU, from the photos (as `build_rung2` computes its CDFs), and have no gradient path. The learned s_k still enter both curves.
  - **Why:**
    - the learned s_k drift only 0.0014–0.0059 stops RMS from s_init (*measured*, `rung2id`, LOM and the varying set), under 1% of the zone width c_2 − c_1 (0.62–1.34 stops, *measured*);
    - fixed weights keep the curve side deterministic and bit-reproducible, whereas a weighted histogram rebuilt every step on the GPU (`bincount` with weights, `scatter_add_`) uses atomic adds, which are nondeterministic;
    - no new per-step GPU op is needed;
    - the preview can replay precomputed targets, as Rung 2's does.
  - The preview reports how far w_lit at the final s would be from w_lit at s_init (test 5).
- **The guide exactly**, so that training, preview and re-score agree by construction:
  - Adaptive average pooling to (H // f, W // f), then `interpolate(mode="bilinear", size=(H, W), align_corners=False)`. The loader's images are 374 × 499, so the blocks are not exact 16 × 16 squares.
    - As built (2026-11-26), the pooling replaces `interpolate(mode="area", scale_factor=1/f)`. It is the same operation (area mode calls the same pooling, with output size floor(H · 1/f)), but the size comes from integer division instead of a floating-point scale factor.
  - log2 is computed as `torch.log(x) / ln 2`, not `torch.log2`, for the same NVRTC reason as `exp2`.
  - The centres come from numpy float64 `np.percentile(..., method="linear")`.
  - One function, `rung3.build_zones(images, s_init, blur)`, serves both `Runner.build_rung3` and the preview.
- **Stored:**
  - The centres, the zone targets [N, 2, 255] and the shares [N, 2] are buffers saved in `curve_state.pt`.
  - The per-pixel weight maps are non-persistent buffers (about 32 MB for chair), rebuilt in build.
  - The re-score rebuilds everything from the photos and checks it against the state (2.5).

### 2.2 The curve loss

Per training view k. The zone histograms, targets and shares are computed once in build (CPU, with the weights at s_init) and looked up when view k is drawn:

```
grey4   = mean_c interpolate(pixels, scale_factor=0.25)        exactly as HistogramPriorLoss.compute_histogram_equalization
w4_m    = the same interpolate applied to w_m                  (nearest, so it samples the same pixels)
h_k,m(b)= sum over pixels of w4_m [ bin(grey4) = b ],  bin = min(floor(255 grey4), 254)   (torch.histc's bins)
CDF_k,m = cumsum(h_k,m) / (sum h_k,m + 1e-8);   T_k,m = rung2.hist_target(CDF_k,m)   (the one-bin shift)
pi_k,m  = sum w4_m / number of pixels                           (zone share; pi_k,sh + pi_k,lit = 1)

L_curve,k = sum_m pi_k,m [ w(t) mse(L_k,m, T_k,m) + lambda smooth(L_k,m)
                           + 0.5 ( mse(p_gamma, F_m) + 0.01 mse(p_gamma, T_k,m) ) ]
          with L_k,m = F_m( min(1, 2^{s_k} x_b) ) on the 255-bin grid, p_gamma = clamp(x_b, 1e-3, 1)^gamma
```

- The trainer multiplies it by 10, as for Rung 2.
- **With M = 1** (one zone, π = 1, w ≡ 1), this is `ExposureCurve.loss` term for term. Unit test 1 checks it.
- **Bins and counts:** bin(·) is computed in float32 exactly as `torch.histc` computes it, `(grey4 * 255).floor()` clamped to 254. The weighted counts are summed in float64 on the CPU (`torch.bincount(..., weights=w4.double())`), then cast to float32. With all-ones weights this reproduces `histc`'s counts, so M = 1 equals Rung 2 by construction.
- **Loss parts:** `loss` returns the keys `Diagnostics.log_losses` reads (`diagnostics.py` lines 266–269), as `rung2.curve_loss` does: `cdf`, `pseudo` and `smooth`, each the π-weighted sum of its term, and `w_cdf`. Otherwise every diagnostics run would stop at step 0.
- **The zone weighting π** keeps the loss on Rung 2's scale. Adam normalises step sizes, so a curve whose zone covers half the pixels still learns at the same rate (*inferred*).
- **Why each curve gets its own zone's target, not the global one:** fitted to the global target only where its pixels are, the lit curve's dark end would be set by the prior alone, and the prior is what lifts the deep shadows most (E4). The lit curve would then lift dark pixels *more* than the shadow curve, the opposite of the aim.
- **Why one shared γ:** it keeps the official prior as Rung 2 has it, and it pulls both curves toward the same power curve, which limits the contrast flattening of section 1.4. A γ per zone is more freedom, and it is not tested.

### 2.3 Everything else is Rung 2's

- **Learning rates and schedule:** Adam, lr 1e-2 for both curves' θ and for ρ (γ = exp ρ), 1e-3 for s, ExponentialLR to 1% over 10k steps; gauge projection after each step.
- **The 3DGS side:** the dark branch (shared), L_low, L_enh (weight 0.5), L_spa and the curve-loss weight 10 are unchanged.
- **Gradient flow:** detached, as `rung2id`: the image loss never reaches F_m, s or γ. The weights, zone targets and shares are constants (2.1).
- **Parameters:** 2K + N + 1, so 87, 108, 94, 101 and 95 for buu, chair, sofa, bike and shrub (`rung2id`: 55, 76, 62, 69, 63; the official code about 255k).

### 2.4 Refusals

Where the checks run:
- The flag checks sit with Rung 2's in `Runner.__init__`, so they exit before `exposure_init.json` or any result file is written. Like Rung 2's, they come after the run folders and the TensorBoard writer are created (`simple_trainer_ours.py` lines 317–330 before 351–361).
- The c_2 − c_1 check needs the photos, so it exits in build.
- `run_rung0` records a refused run as failed and clears its folder on a retry.

The refusals:
- `--rung3_zones 2` without `--curve_model rung2`.
- `--rung3_zones 2` with any of these, which are out of scope and so refused rather than left untested:
  - `--rung2_attach` other than `none`;
  - `--rung2_color` other than `rgb`;
  - `--rung2_prior` other than `F`;
  - `--rung2_knot_spacing usedrange`;
  - `--rung2_F_init cdf` (no init is defined for two curves);
  - `--rung2_s_init zero` (the guide would not be exposure-aligned);
  - `--dark_branch` other than `shared`.
- `--rung3_zones` other than 1 or 2. M = 3 is not in this plan.
- c_2 − c_1 below 0.05 stops (the zones would not separate; this can only happen on clipped photos).

### 2.5 Diagnostics, state and re-scoring

- **`curve_forward`'s keys stay the same.**
  - `curve_adj` = the π-weighted mean composed curve, so the existing readers see one effective curve per view. The training step does not use it for the loss in Rung 3 mode.
  - New keys: the per-zone composed curves [M, 255], the zone targets [M, 255] and the zone shares [M].
  - `projected` stays (R, G, B).
- **Per-zone arrays get new keys only** (decided at step 0).
  - Three existing readers assume a [255] curve and would broadcast a [2, 255] array without an error:
    - `rung2_headstart.py` line 124;
    - `rung2_ablation.py` line 326, where `F_grid[1:5]` would return the lit curve's whole row;
    - `rung2_ablation.py` line 541.
  - So in Rung 3 mode the single-curve keys are left out. Every old reader then stops with a KeyError, and none needs editing.
- **`curves_step*.npz`:** add the per-zone curves, targets and shares per view. `curve_global_zones` [M, 255] and `F_heights_zones` [M, K + 1] replace `curve_global` and `F_heights`, which are not written. `F_knots`, `s` and `s_init` stay.
- **`rung2_state.json`:**
  - add `zones`, `blur`, `centres`, `F_grid_zones` [M][255], `heights_zones` [M][K + 1], and the zone shares per view;
  - leave out `F_grid` and `heights`;
  - keep `knots`, `s_init`, `s_final`, `view_names`, `gamma` and `n_params` (= 2K + N + 1), which the `h1` mode and the leak and s_k-slope code read.
- **One sample figure per run:** the re-score writes bike's view 29 (`diagnostics.RESCORE_EXAMPLES`) with its w_lit map as a separate `example_29_wlit.png`. The existing `example_29_in_pseudo_render_gt.png` keeps its four panels, because `rung05_analyze.py` (lines 318–326) reads it as input | pseudo-target | render | GT.
- **Re-score:**
  - `rescore_runs.MODEL_FLAGS` and `diagnostics.MODEL_DEFAULTS` gain `rung3_zones` and `rung3_blur`.
  - The re-score's build recomputes the centres, zone targets and shares from the photos.
  - `_load_run_state` loads the stored ones from `curve_state.pt` and asserts they agree within 1e-6, as for `usedrange` knots. If the state's zone count differs from the flags, it stops with a clear message.

### 2.6 Runner variant

```
"rung3": ["--curve_model", "rung2", "--rung2_F_init", "identity", "--rung3_zones", "2"]
```

Added to `NOT_IN_RUNG0`. On the varying set it is the shared column, like `rung2id` and `rung2idlum`. The blur keeps its default (16).

---

## 3. Code and tests (after Bart's OK; about a day, *inferred*)

| File | Change |
|---|---|
| `rung3.py` (new) | `build_zones(images, s_init, blur)`: the guide, centres, weight maps, zone targets and shares of 2.1 and 2.2, on the CPU; the trainer and the preview both call it. `ZoneExposureCurve`: two curves on Rung 2's knots, shared s and γ, the zone constants as buffers; `composed_curves`, `map_pixels`, `loss` (with the four parts keys), `project_gauge`, `strictly_increasing`, `summary`, `n_params`, `f_params`. It imports rung2's helpers (`interp`, `hist_target`, `curve_loss`, `curve_weight`, `make_knots`, `init_heights`, `theta_from_heights`). **`rung2.py` is not changed.** |
| `simple_trainer_ours.py` | `Config.rung3_zones` (1) and `Config.rung3_blur` (16); the refusals; `build_rung3` (calls `rung3.build_zones`; parameter print); the Rung 3 branch of `rung2_forward`; the zone curve loss in `train()`; grad-check and end-of-run prints and asserts for both curves. `build_rung3` draws no random numbers, like `build_rung2`, so `rung3` and `rung2id` see the views in the same order at a given seed. With `rung3_zones` 1, no Rung 3 code runs. |
| `diagnostics.py` | `MODEL_DEFAULTS`; the snapshot and state fields of 2.5 (new keys only); `_load_run_state` checks the zone count, centres, targets and shares; `example_29_wlit.png` |
| `rescore_runs.py` | `MODEL_FLAGS` |
| `run_rung0.py` | variant `rung3`, `NOT_IN_RUNG0` (its message also names `results/rung3`), a docstring example |
| `test_rung3.py` (new) | unit tests (below), CPU |
| `rung2_replay.py` | `--mode rung3`: the preview (test 5), writing `results/rung3/preview/`. Optional `--mode rung3long` (A2, step 6), writing `results/rung3/replay_long/`. Both go into `OUT`. Its guard uses `protected.py` since step 1, which already covers M5's `replay_long/` and `overexp/preview/`. `results/rung3/preview/` is added to `protected.FINISHED` once the preview is logged (step 3). |
| `rung2_ablation.py` | `EXPECTED_ALL["rung3"]` (`rung3_zones` 2, `rung3_blur` 16). `"rung3_zones": 1` added to `EXPECTED_ALL["rung2id"]`, as E5 added `rung2_color`, so a zone run cannot pass as the base. `rung3_zones: 1` and `rung3_blur: 16` in `CFG_DEFAULTS_ALL` (older `cfg.json` files lack them). `--mode rung3` (section 7). `results/rung3` added to `guard_out`'s roots. |
| `docs/curve_estimation.md`, `CLAUDE.md` | "Rung 3 as built"; the flags, variant, roots and repo map; H4 |

**Tests, all before the batch (rule 3):**

1. **Unit tests** (`test_rung3.py`, CPU). All 16 tests in `test_rung2.py` still pass, and:
   - **M = 1 is Rung 2:** with one zone, `map_pixels`, the composed curve and the loss equal `ExposureCurve`'s bit for bit (`torch.equal`).
   - **Two equal curves with constant weights are Rung 2:** F_sh = F_lit = F and w ≡ ½ give Rung 2's pseudo-target and Rung 2's loss (to 1e-6; E5 measured 1.2e-7 against a 1e-7 target for the same float32 reason).
   - **Weights:** non-negative, sum to 1 (to 1e-6), exactly 0 or 1 outside [c_1, c_2].
   - **Monotone:** for random curves and weights, the output never decreases as a channel's input rises (dense grid). Black maps to 0, and 1 maps to 1 when s_k ≥ 0.
   - **Histogram:** with all-ones weights, the weighted histogram equals `torch.histc`'s counts exactly, and the CDF equals `compute_histogram_equalization`'s, on synthetic levels and on three real training photos. The float32 bin formula of 2.2 is chosen to make this hold by construction; if the bins still disagree at edges, report the count and fix the formula before going on.
   - **Exposure invariance:** scaling a photo by 2^u and its s_init by −u leaves the guide and the weights unchanged (to 1e-6, away from clipping).
   - **Gradients:** `map_pixels` gives no gradient; the loss gives gradient to both θ's, s and ρ; the weights, targets and shares are buffers without gradient.
   - **Build:** a second call of `build_zones` gives the same centres, targets and shares bit for bit, and `loss` returns the parts keys `cdf`, `pseudo`, `smooth` and `w_cdf`.
2. **CUDA op check:** with the weights built on the CPU (2.1), the only new op on the GPU is the convex mix in `map_pixels`: two `interp` look-ups and a weighted sum. Run it once on CUDA and check it equals the CPU result to 1e-6. Also check that the weight maps moved to the GPU equal the CPU ones. Torch 1.13's run-time-compiled kernels fail in this env (`nvrtc-builtins64_117.dll`), so every new tensor op is tried on the GPU first.
3. **Regression** (as in E5): a 300-step `rung2id` smoke on the varying set (buu, seed 0) with the new code must match `results/vexp/smoke_e5/buu_rung2id_seed0` in F, s, heights, γ, `exposures_by_step.csv` and every pseudo-target column (max |Δ| 0). The 3DGS side is not bit-reproducible run to run on this GPU (E5: about 1e-4 dB), so only the curve side is checked bit for bit.
4. **Smoke runs:** 300 steps, buu, seed 0, each re-scored with `--ref_root ../results/rung0`:
   - `rung3` on LOM (`results/rung3/smoke_m6/`) and on the varying set (`results/vexp/smoke_m6/`);
   - `rung3` on `over_exp` (`results/overexp/smoke_m6/`), only if the exploratory arm stays in;
   - a `rung2id` LOM smoke in `results/rung3/smoke_m6/` as the timing partner.

   Check:
   - 87 curve parameters on buu; both curves strictly increasing; |mean s| < 1e-5; zone shares near ½;
   - the state and npz fields of 2.5, with no `F_grid`, `heights`, `curve_global` or `F_heights` written;
   - the re-score's rebuilt centres, targets and shares match the state;
   - minutes per run against the `rung2id` partner;
   - the `over_exp` smoke reads `over_exp/` (mean input about 0.9).
5. **Curve-only preview** (`rung2_replay.py --mode rung3`, CPU, training photos only, no normal-light image read): `rung3`'s and `rung2id`'s curve training replayed for 10k steps per scene, with the trainer's loader, model, optimisers, schedule and gauge.
   - The zone targets are precomputed per view by `rung3.build_zones`, as in the trainer, so a step costs about two Rung 2 steps (*inferred*). Rung 2's replay takes about 12 s per scene for 10k steps (*measured*), so both models over five scenes take about 2–3 min.
   - **Report:**
     - F_sh, F_lit and `rung2id`'s F at levels 1, 2, 4, 8, 16, 32 and 64; γ; zone shares;
     - s movement, and s against the `rung2id` replay;
     - the photo statistics of section 1.4 recomputed, with both the hard and the soft chromatic shares;
     - the mean pseudo-target brightness in each zone for both models (how much the zones flatten the contrast);
     - the share of training views in which the mean pseudo-target of shadow-weighted pixels exceeds that of lit-weighted pixels (the lighting order reversed), for both models;
     - per scene, the maximum over pixels and views of |w_lit at the replay's final s − w_lit at s_init| (how much fixing the weights at s_init changes them, 2.1).
     - The last two are reported only.
   - **The predicted directions for section 4.3** (photo-only): with each model's final 10k curves, the pseudo-targets of every training view, and on the chromatic photo pixels (section 1.4's definition) their mean HSV saturation and their mean circular hue drift from the photo's own hue. Per scene: `rung3` − `rung2id` for both. These are logged before the batch.
   - **Stop rule** (photo-only; ask Bart before the batch if any holds):
     - a curve is not strictly increasing (a bug);
     - in every decision scene, max over the grid of |F_sh − F_lit| is below 0.01 (Rung 3 would collapse to Rung 2, so the batch would test nothing);
     - in any scene, the replay's s differ from the `rung2id` replay's by more than 0.05 stops RMS (the exposure model is disturbed).
   - Nothing in the design changes after the preview. Its numbers go in the log either way.
   - **Protecting the predictions:** the preview's log entry records the sha256 of `results/rung3/preview/numbers.json`, and `--mode rung3` checks it (section 7). After that entry, the folder is added to `protected.FINISHED`, rule 6 and the `ask` list (step 1), so a later re-run cannot replace the pre-registered directions.
6. **`rung2_ablation.py --mode rung3 --selftest`** (the `rung3` arms read from `rung2id`'s runs): every effect exactly 0. The zone-curve tables are skipped in selftest.

---

## 4. Pre-registered readings (logged before any run, once Bart confirms)

General rules, as in M5 and E5:
- pair by seed, seeds 0–2;
- noise = sqrt(mean of the two arms' seed variances); an effect is real above 2× noise;
- pooled = the five-scene mean per seed, checked against pooled without shrub (say so if they differ); shrub never decides;
- report raw and bm PSNR side by side.

**Comparison:** `rung3` − `rung2id`.
- LOM: `rung3` in `results/rung3/`, `rung2id` in `results/rung2/`.
- Varying set: both in `results/vexp/` (shared column).
- The `rung2id` runs are from earlier code versions. The regression check (test 3) makes the comparison valid, as for E5.

### 4.1 H4 (Rung 3)

**Statement:** "A per-pixel convex mix of two shared monotone curves, weighted by exposure-aligned local brightness and each fitted to its zone's histogram, changes quality relative to the single global curve, on LOM or the varying set."

- **Helps:** on either set, pooled bm PSNR higher, or pooled LPIPS lower, beyond 2× noise, with no harm signal.
- **Harm signals:**
  1. any decision scene's bm PSNR below −2× noise, on LOM or the varying set;
     - Bike's LOM bm PSNR seed sd is 0.002 dB (`rung2id`, *measured*). So 2× noise there is about 0.005 dB if `rung3` is as stable, and this signal fires on almost any bike drop on LOM.
     - The rule is kept as in M5, E5 and H1 (bike failed H1 at −0.15 dB, −3.8×). The results report the size of any such drop.
  2. pooled bm PSNR lower, or pooled LPIPS higher, beyond 2× noise, on either set;
  3. **cross-view consistency** (decided by Bart 2026-11-26; D5): on LOM, the mean over buu, chair, sofa and bike of the per-scene relative change in pseudo-target inconsistency is above +20%.
     - **Per-scene change:** (`rung3` seed mean)/(`rung2id` seed mean) − 1 of `rel_l1_pseudo`, the mean over each run's `rescore/consistency.csv` (shared reference depth). Shrub and the five-scene pooled value are reported only.
     - **Why a fixed threshold:** its seed noise is negligible. F, s and the pseudo-targets depend on the seed only through the order in which the DataLoader draws the views, and the reference depth is shared (*measured*, `rung2id`: seed sd at most 4.8e-5 in every scene).
     - **Why the decision scenes and not the pooled value:**
       - Shrub's value is about twice the others', so it makes up about 40% of the pooled mean.
       - E5's luminance mode moved the pooled value by +8.0%, but per scene by +12.7% (buu), +30.0% (chair), +18.7% (sofa), +14.3% (bike) and −6.4% (shrub): +18.9% on this measure (*measured* 2026-11-26).
       - The per-scene change is the same for raw `rel_l1_pseudo` and for M3's ×GT `cons_pseudo` (`rung2_analyze.py` line 248), because the GT term is shared within a scene.
     - **Scale:** per scene, `rung2id` has buu 0.0824, chair 0.0460, sofa 0.0567, bike 0.0692, shrub 0.1699 (pooled 0.0848). On the varying set, `detach`'s pseudo-targets (which follow the gains) reach 0.248 pooled against `rung2id`'s 0.095 (*measured*, each run's `rescore/consistency.csv`).
  4. on the varying set, a pooled leak above 0.1, or a pooled s_k slope outside −0.9 to −1.1 (as H3). The leak pairs each `rung3` varying run with the `rung3` LOM run of the same seed.
- **Verdict:** "helps" if a help criterion is met and no harm signal fires; "hurts" if a harm signal fires and there is no help; "mixed" if both; "no effect" otherwise.
- **Direction not predicted.** The lit-zone curve should help where the lit-zone dark pixels are objects, but the zone targets also flatten the shadow-to-lit contrast (section 1.4; risk 1).

### 4.2 Bike's hue (the design's target)

- **Fixes bike's hue:** bike's hue error on LOM falls by more than 2× noise, it does not rise beyond 2× noise on the varying set, and bike's bm PSNR does not fall beyond 2× noise on either set.
- **Fixes at a cost:** the hue falls as above, but bike's bm PSNR falls beyond 2× noise on either set.
- **No fix:** anything else.
- **Why LOM decides:** `rung2id`'s bike hue sd is 0.47° on LOM and 5.56° on the varying set (*measured*, `E5_results.md`), so 2× noise is about 1° on LOM and about 8–11° on the varying set (*inferred*).
- **Bike's bm PSNR bar is very low on LOM:** about 0.005 dB (`rung2id` seed sd 0.002 dB, *measured*; 0.034 dB on the varying set). In practice, "fixes" needs bike's LOM bm PSNR not to fall at all (4.1, harm 1).
- **For scale:** E5's luminance mode took bike from 33.6° to 9.6° on LOM, at a cost elsewhere. Rung 3 keeps per-channel application, so a smaller change is expected (*inferred*).

### 4.3 Mechanism check: does validation colour move the way the photo-only pseudo-targets predict?

- **The rule is fixed now; the predicted directions are filled in from the preview (test 5) and logged before the batch.** No normal-light image is read to make them.
- **Saturation:** a decision scene gets a prediction if its preview change in mean pseudo-target saturation on chromatic pixels (`rung3` − `rung2id`) is at least 0.02 in size; the predicted direction is that change's sign. The check is the validation saturation ratio, `rung3` − `rung2id`, on LOM.
- **Hue:** bike gets a prediction if its preview hue drift from the photo changes by at least 1°; a smaller drift predicts a smaller hue error. The check is bike's validation hue error on LOM (the same quantity as 4.2).
- **Reading:** "as predicted" if every predicted direction holds beyond 2× noise; "partly" if at least one holds and none is reversed beyond 2×; "not as predicted" if none holds or any is reversed beyond 2×; "no prediction" if no scene reaches the minimum size.
- **Why from the preview and not from the lit shares alone:** what moves colour is the curve's log-slope at each pixel's levels, not the amount of lift. The lit shares of section 1.4 (w_lit > 0.5: chair 79%, bike 74%, sofa 12%; soft shares 0.76, 0.73, 0.19) suggest saturation up in chair and bike, but in sofa the shadow-zone target is steep exactly where its chromatic pixels sit (levels 8 to 16), so its direction is not obvious (*inferred*). The preview computes it.
- **Label:** pre-registered after E5's results were known; the photo statistics were computed on 2026-11-25 from training photos only. The directions predict; they did not choose the design.
- It decides nothing about defaults. It tests whether the curves act through the mechanism the design assumes.

### 4.4 Reported without a decision

- Every metric on both sets, per scene and pooled, with `detach` alongside as context. Also: hue error, angle error, saturation ratio, pseudo-target bm PSNR, dark fit, Gaussians and minutes per run.
- Both curves at levels 1–64, γ and the zone shares.
- s against `rung2id`'s, same seed: max |Δs| and the drift from s_init. At the same seed both arms see the views in the same order, because `build_rung3` draws no random numbers.
- Curve loss at 10k for `rung3` and `rung2id` (different objectives; reported side by side, per the 2026-11-25 decision), from two sources:
  - the mean of the last 10 logged `hist_loss` values (`rung2_headstart.curve_loss`, M3's source);
  - the all-view objective at the late weight, from the preview replay.
- **Exploratory H1 rule,** `rung3` − `detach` on LOM. It changes neither H1 (`cdf`, not supported) nor H1b (supported).
- **Overexposure (if run; exploratory):** `rung3` − `rung2id` on `over_exp`, and the H1 rule against `detach` there. No direction is predicted, and it cannot change H1-over.

### 4.5 What happens after the verdict

- No default changes before the freeze. The report's model stays Rung 2: H1 on `cdf`, H1b on `rung2id`, side by side.
- Rung 3 is reported as the stretch result with its verdict, whatever it is.
- A "helps" verdict would be a finding about local tone, not a new default chosen after seeing results.

---

## 5. What M6 can and cannot show (state it in the results)

**Can show:**
- whether brightness-adaptive local tone, with the paper's histogram prior made zone-wise, changes quality against one global curve, with the same training budget;
- whether it keeps Rung 2's exposure handling (leak, s_k slope) and cross-view consistency;
- whether lifting the dark end less in lit neighbourhoods reduces bike's hue error while keeping per-channel application.

**Cannot show:**
- **Locality and the zone targets change together.** A "helps" or "hurts" cannot be split between "local" and "zone-wise equalisation".
- **One configuration.** M = 2, a 16-pixel block, the 25th/75th-percentile centres and one shared γ are fixed in advance and not tuned. Another M, block size or prior could behave differently.
- **Learned or attached weights** (3D feedback into the mix) are not tested.
- **The per-scene saturation mismatch** (section 1.3).
- **Generality:** five LOM scenes. The varying set has the same scenes and ground truth (same caveat as H2b).
- **Seeds:** the zone weights do not depend on the seed. F and s depend on it only through the order in which views are drawn (*measured*, `rung2id` seeds 0–2: F within 7e-5 on LOM and 2e-4 on the varying set; s within 0.005 stops, the same size as s's drift from s_init). So seeds re-test mainly the 3DGS randomness (same caveat as H1b).

**Detection limits** (`rung2id`'s LOM seed sd, `summary_headstart_seeds012/tables.md`):

| | buu | chair | sofa | bike | shrub | pooled |
|---|---|---|---|---|---|---|
| bm PSNR sd (dB) | 0.197 | 0.218 | 0.400 | 0.002 | 0.010 | 0.071 |
| hue error sd (°) | 0.25 | 0.16 | 0.19 | 0.47 | 0.03 | 0.15 |

So 2× noise on bm PSNR is roughly 0.4–0.6 dB in buu and chair, about 0.8 dB in sofa, and about 0.005 dB in bike (*inferred*; it depends on the `rung3` arm's spread too). Bike's bar is so low that harm signal 1 and 4.2's cost test fire on almost any bike drop on LOM (4.1).

---

## 6. Runs, roots and commands

**Batch:** 30 runs, plus 15 optional. About 2.3 h, or 3.4 h with `over_exp` (*inferred* from measured minutes per run: `rung2id` 4.3 on LOM (`h1b`) and 4.4 on the varying set; `rung2idlum` 4.1 on LOM, 4.3 on the varying set and 4.8 on `over_exp`; plus a few percent for the convex mix, which the smoke measures). Bart launches it with `/run-batch`, with sleep off (M5 lost 1 h 44 min to sleep; use M5's stay-awake helper). From `examples/`, in this order:

```
python run_rung0.py --root ../results/rung3 --variants rung3 --seeds 0 1 2
python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants rung3 --seeds 0 1 2
python run_rung0.py --exp_name over_exp --root ../results/overexp --variants rung3 --seeds 0 1 2      (optional)
```

**Why `results/rung3/` for LOM:** Rung 3 is its own rung, and the report's reader will look for it there. The `rung2id` arm stays in `results/rung2/`, and the analysis reads arms across roots, as M5's `h1` mode does. The varying-set and `over_exp` runs go to the roots that hold every run of those sets (the runner requires `results/overexp/` for `over_exp`).

**Re-score** (about 10 s per run):
```
python rescore_runs.py --root ../results/rung3 --ref_root ../results/rung0
python rescore_runs.py --root ../results/vexp --ref_root ../results/rung0
python rescore_runs.py --root ../results/overexp --ref_root ../results/rung0      (if run)
```

**Smokes** (step 3):
```
python run_rung0.py --root ../results/rung3/smoke_m6 --variants rung2id rung3 --scenes buu --seeds 0 --max_steps 300
python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp/smoke_m6 --variants rung2id rung3 --scenes buu --seeds 0 --max_steps 300
python run_rung0.py --exp_name over_exp --root ../results/overexp/smoke_m6 --variants rung3 --scenes buu --seeds 0 --max_steps 300
```
The `rung2id` smoke in `results/vexp/smoke_m6/` is the regression run (test 3). `smoke*` folders are git-ignored.

---

## 7. Analysis

**`rung2_ablation.py --mode rung3`** → `results/rung3/summary/`:
- **Cells:** per arm and set (`rung3`, `rung2id`, `detach` as context; `over_exp` arms if run).
- **Effects:** `rung3` − `rung2id`, paired by seed, on every metric and extra, plus `rel_l1_pseudo`, with harm 3's per-scene relative changes (the decision-scene mean decides; shrub and the pooled value are reported).
- **Mechanism:** leak, pooled s_k slope, s drift, max |Δs| against `rung2id`.
- **Zone curves:** both curves at levels 1–64 against `rung2id`'s F; γ; zone shares; the chromatic lit share per scene (hard and soft).
- **The readings of section 4,** printed as decisions, as `--mode lum` does. For 4.3 the script reads the predicted directions from the preview's output (`results/rung3/preview/numbers.json`, logged before the batch) and refuses to run if they are missing. It also prints that file's sha256 and stops if it differs from the hash in the preview's log entry, which is written into the script as a constant when the preview is logged.
- **Figures:** the two zone curves and `rung2id`'s F per scene; bike's view 29 (input, w_lit, pseudo-target for both models, GT); the validation saturation change against the preview's predicted change, per scene.

**Exploratory H1 rule** (the tested `h1` mode):
```
python rung2_ablation.py --mode h1 --a rung3 --a_roots ../results/rung3 ../results/rung2 --b detach --b_roots ../results/rung0 --official_roots ../results/rung0 --extra rung2id --title "M6, exploratory" --out ../results/rung3/summary_h1
```
On `over_exp` (if run): `--a_roots ../results/overexp`, `--b_roots ../results/overexp`, `--official_roots ../results/overexp`, `--exp_name over_exp`, `--out ../results/overexp/summary_m6`.

**Write-up:** one log entry per step, as before, and `Implementation_guides/M6_results.md` with a "What the final report says" section.

---

## 8. Steps and schedule

### Step 0: check this plan against the code, then stop (done 2026-11-26)
- Read `CLAUDE.md`, this plan, the last three entries of `docs/RESEARCH_LOG.md`, and `E5_results.md`.
- Check every claim about the code (and section 1.4's quick-check figures, which the preview recomputes): the curve-loss target (`compute_histogram_equalization`), `hist_target`, `rung2_forward`'s keys, the curve-loss branch in `train()`, the diagnostics readers of `curve_adj`, `curve_global` and `rung2_state.json`, `MODEL_FLAGS`/`MODEL_DEFAULTS`, the runner's guards, and `rung2_ablation.py`'s arms, guard and `h1` mode.
- Report errors or gaps, then wait for Bart's OK. Stay on `main`, no commits (rule 1).
- **Done 2026-11-26** (log: "M6 step 0"):
  - eight corrections and seven gaps are applied in this text;
  - Bart chose the s_init weights (2.1; D6) and harm 3 on the decision scenes (4.1; D5);
  - E5 was already committed (HEAD `89b5061`).
- **Before any M6 code:** Bart commits this plan together with the step-1 log entry, so the pre-registration has a git timestamp.

### Step 1: log the decisions and readings (before any code or run) (done 2026-11-26)
- One `/log-experiment` decision entry: the design (section 2, with Bart's choices on section 11) and the readings of section 4 as Bart confirmed them (log: "M6 step 1").
- `CLAUDE.md`: H4 added to the goal section; the Rung 3 row updated.
- Finished outputs protected (Bart chose "all of it", 2026-11-26; log: "Finished M5/E5 outputs protected").
  - **Folders:** M5's and E5's outputs, plus the 2026-11-11 `results/rung2/preview/` and `preview_lrF1e-2/`.
  - **Rule 6 and settings:** added to CLAUDE.md rule 6 and to the `ask` list in `.claude/settings.json`.
  - **Scripts:** added to the new `examples/protected.py`, which `rung2_analyze`, `rung2_ablation`, `rung2_replay`, `rung2_preview` and `rung2_headstart` now use.
  - **Holes closed:** before this, a bare `rung2_preview.py` overwrote `preview_final/`, a bare `rung2_headstart.py` overwrote `summary_headstart*/`, and `rung2_replay.py --mode long` or `--mode overexp` overwrote A1's and E6's outputs.
- After the preview is logged (step 3), `results/rung3/preview/` joins `protected.FINISHED`, rule 6 and the `ask` list, so the pre-registered predictions cannot be overwritten.

### Step 2: code and unit tests (section 3, tests 1–2) (done 2026-11-26)
- Built `rung3.py`, `test_rung3.py` and the trainer, diagnostics, re-score and runner changes; the docs are updated.
- `rung2_replay.py --mode rung3` (the preview) comes with step 3, and `rung2_ablation.py --mode rung3` with step 4.
- Tests 1 and 2 pass; results are in the log.

### Step 3: regression, smokes, preview (tests 3–5); ask Bart before the batch (done 2026-11-27)
Log the preview, its stop rule and the predicted directions of section 4.3 in one entry before the batch starts.
- **Done:** tests 3 and 4 pass, and the preview passes its stop rule.
- **Predictions:**
  - saturation goes up in buu, chair, sofa and bike;
  - bike gets no hue prediction, because its drift change of −0.72° is under the 1° minimum.
- **Hash:** `numbers.json` sha256 `991f45a9…3974f` is in the log, and the folder is protected.
- **Two details:** the preview also ran on the varying set, for the exposure check only. Fixing the weights at s_init changes them by up to 0.06 in buu (log, test 5).

### Step 4: batch (section 6), launched by Bart with `/run-batch`
While it runs (CPU only; do not edit any file the trainer imports): `--mode rung3` and its selftest (test 6).

### Step 5: re-score, analyse, report
Apply the readings as logged; log entries; `M6_results.md`; update `CLAUDE.md`'s rung table and repo map; propose the commits.

### Step 6 (optional): A2, a long replay of the zone curves
`rung2_replay.py --mode rung3long`: replay `rung3`'s curve training to 50k steps, as A1, and report how far the zone curves at 10k are from their 50k end. CPU, about 10 min (*inferred*). A1 took 698 s for 600k Rung 2 steps, about 1.2 ms per step (*measured*); a Rung 3 step costs about two, and the zone targets are fixed per view as in the trainer. Time 1k steps first and ask Bart if the estimate is above 30 min. It says whether Rung 3 is early-stopped like its base; no decision depends on it.

### Schedule (*inferred*)

| When | What |
|---|---|
| Thu 26 Nov | Steps 0 and 1 done: plan checked against the code, D1–D6 settled, section 4 approved and logged, finished outputs protected. Bart commits the plan with the step-1 entry |
| Fri 27 Nov | **Design settled.** Steps 2–3: code, unit tests, CUDA check, regression, smokes, preview. Batch launched in the evening (2.3 h, or 3.4 h with `over_exp`) |
| Sat 28 – Sun 29 | Re-score (about 8 min); `--mode rung3` and its selftest; the readings; log entries (or Mon 30, as Bart's time allows) |
| Mon 30 Nov | `M6_results.md`; message Wei (section 10) |
| Tue 1 – Thu 3 Dec | Slack for a failed run or a fix; A2 if wanted; start M7's final tables and figures |
| Fri 4 – Sat 5 Dec | Final checks; **freeze on Sat 5 Dec** (M7) |
| 6 – 18 Dec | Report |

**If time runs short, drop in this order:**
1. A2 (the long replay);
2. the `over_exp` arm (15 runs);
3. the figures beyond the zone-curve plot and the bike example;
4. the varying-set arm (15 runs). This loses the leak and s_k-slope checks, so H4 is then judged on LOM only, and the results say so;
5. if the LOM arm cannot start by Wed 2 Dec, Rung 3 is reported as designed, unit-tested and smoke-tested but not run, as a limitation.

Seeds are never dropped: three seeds are the minimum for a noise estimate.

---

## 9. Risks

1. **The zone targets flatten the contrast between shadowed and lit regions** (section 1.4: bike's level 8 maps toward 228 in the shadow zone and 94 in the lit zone). That could lower bm PSNR and SSIM. It is a legitimate "hurts" result, not a bug. The shared γ limits it, and the preview reports how much: the zone brightness, and the share of views with the lighting order reversed. Bike's zones are only 0.65 stops apart (*measured*), so it is the scene most exposed.
2. **Cross-view inconsistency.** The neighbourhood of a surface changes near image borders and occlusions, so its weights can differ between views, and the 3DGS then averages two targets (*inferred*). Harm signal 3 measures it.
3. **The weights use s_init, not the learned s_k** (decided 2026-11-26; D6). The learned s_k drift 0.0014–0.0059 stops RMS from s_init (*measured*, `rung2id`, LOM and the varying set). That is under 1% of the zone width (0.62–1.34 stops), so weights that followed s_k would differ little (*inferred*). The preview measures the difference (test 5).
4. **Histogram bins.** A weighted histogram that does not bin exactly like `torch.histc` would break the M = 1 equivalence. The float32 bin formula of 2.2 follows `histc`'s arithmetic, and unit test 1 checks it on real photos.
5. **CUDA ops.** New tensor ops can fail at run time in this env (NVRTC). With the weights built on the CPU, the only new GPU op is the convex mix, which test 2 runs first.
6. **GPU nondeterminism and sleep:** as before. Compare by seed, never single runs, and keep the PC awake.
7. **Forking paths.** One configuration is fixed in advance (M = 2, 16-pixel block, 25th/75th-percentile centres, shared γ, identity init). Nothing is retuned after the preview or the batch; a failure is reported as one.
8. **The analysis script is on the critical path.** If `--mode rung3` slips, the batch still runs and its results wait. No run is read before the script passes its selftest.
9. **The pre-registration record.** E5 is committed (HEAD `89b5061`). This plan and the step-1 entry are committed before any M6 code, and the preview's predictions are hashed and protected (test 5, section 7), so the readings cannot drift after results are seen.
10. **Run time.** Two curve look-ups per pixel and the weighted sum add a little to each run (*inferred*). The block average and the zone histograms are computed once, in build. The smoke measures the cost against `rung2id`'s.

---

## 10. For Wei (the next update; E5's headline goes with it if not yet sent)

- **Rung 3 is running as M6,** on the identity base, with fixed brightness-zone weights instead of a learned conv net, and two curves instead of 4–8. Why (section 1.2): nothing trains a conv net in the detached setting, and attaching it to the image loss would bring back the curve chasing Rung 0.5 measured.
- **What it should fix and what it cannot** (section 1.3): the dark-end compromise of one global curve, measured through bike's hue; not the per-scene saturation mismatch, which E5 traces to the photos' own saturation relative to the ground truth.
- **Readings fixed before any run** (section 4); results expected about Mon 30 Nov.
- **No default changes before the freeze.** The report keeps H1 (`cdf`) and H1b (`rung2id`) side by side, and Rung 3 is the stretch result.

---

## 11. Decisions (all settled by Bart on 2026-11-26: D1–D4 as recommended, D5 and D6 as recorded; log: "M6 step 0", "M6 step 1")

| # | Decision | Recommended | Alternatives | Why |
|---|---|---|---|---|
| D1 | How the per-pixel weights are set | Fixed, from exposure-aligned local brightness (section 2.1) | A small conv net attached to the image loss | Nothing trains a conv net in the detached default; attaching it risks curve chasing (Rung 0.5) |
| D2 | Number of curves | M = 2 | M = 3 | The smallest local model; each extra zone thins the zone histograms and adds K parameters |
| D3 | Shape prior | One γ shared by both curves | One γ per curve | Keeps Rung 2's prior and limits contrast flattening; a γ per curve adds freedom and is untested |
| D4 | The `over_exp` arm | Include it last, exploratory (15 runs) | Leave it out | Cheap (about 1.2 h), and Bart allowed an exploratory check; first to drop after A2 |
| D5 | The consistency threshold (harm signal 3) | **Decided by Bart 2026-11-26:** the mean per-scene relative change over buu, chair, sofa and bike above +20% (shrub and pooled reported only) | The draft's pooled five-scene value 10% above 0.0848; the four-scene mean at 10% | Its seed noise is negligible (≤ 4.8e-5), so the 2× rule cannot apply. Shrub is about 40% of the pooled value: E5's luminance mode was +8% pooled but +18.9% on the four-scene measure |
| D6 | The exposure the guide uses | **Decided by Bart 2026-11-26:** s_init, fixed per view (weights, zone targets and shares built once on the CPU) | The learned s_k at every step (the draft) | s_k drift is under 1% of the zone width; fixed weights keep the curve side deterministic, need no new per-step GPU op and make the preview and A2 fast |
