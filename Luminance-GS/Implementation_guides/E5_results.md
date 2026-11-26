# E5 results: the colour-preserving variant (`rung2idlum`) on the identity base

Written 2026-11-25 by Claude Code for Bart, and as source material for the final report and the next update to Wei.
- **Plan:** `E5_luminance_plan.md`. The readings were pre-registered in `docs/RESEARCH_LOG.md` on 2026-11-25, before any E5 code or run.
- **Labels:** every number is *measured* unless marked *inferred*.
- **Statistics:** pairs are matched by seed. Noise = sqrt(mean of the arms' seed variances). An effect is real above 2× noise (*).
- **Pooled:** the mean over the five scenes per seed, checked against pooled without shrub. Shrub never decides.
- **bm PSNR** = brightness-matched PSNR.

**What E5 changes:** the shared curve F is applied to the channel-mean grey Y = (R+G+B)/3, and each channel is scaled by F(2^{s_k} Y)/Y. Against `rung2id` (F applied to R, G and B separately), only the colour handling changes:
- F, s and γ are trained only by the curve loss on the photos;
- that loss's HE-CDF target is built on this same grey (`losses.py` line 56).

## Verdicts at a glance

| Question | Pre-registered verdict | Headline | Source |
|---|---|---|---|
| Built-in check: F, s, γ identical to `rung2id`? | **Passes** | max \|Δ\| = 0.0 in all 45 pairs | `results/rung2/summary_e5/` |
| Primary: does luminance mode fix bike's hue? | **Fixes at a cost** | bike hue on the varying set 27.0° → 9.3° (−17.7°, −4.4×); but buu loses bm PSNR on both sets and sofa on the varying set | same |
| Adoption into M6's base | **Not adopted: M6 builds on `rung2id`** | verdict is not "fixes", and pooled SSIM is worse on both sets (−0.013, −6.7× and −9.8×) | same |
| H1 rule, `rung2idlum` − `detach`, LOM (exploratory) | 11 of 12 checks meet (buu bm PSNR fails) | pooled +0.54 dB (+3.2×); bike +2.61 dB (+35.7×) | `results/rung2/summary_e5_h1/` |
| Overexposure (exploratory) | **No:** colour handling does not account for part of the gap | pooled −1.56 dB (−19.2×); the gap to `detach` widens from −1.20 to −2.75 dB | `results/rung2/summary_e5/`, `results/overexp/summary_e5/` |

**Scope:**
- **Runs:** 45 training runs (`rung2idlum`: LOM, the varying set and `over_exp`, seeds 0–2). All were `ok` and all were re-scored against `results/rung0/<scene>_baseline_seed0`.
- **Code:** commit `01d43c5`, plus the uncommitted E5 changes (log, 2026-11-25).
- **Comparison runs:** `rung2id` on LOM (M3 head-start), on the varying set (M5 E1) and on `over_exp` (M5 E6). `detach` is the context arm.
- **Tests before the batch:** unit 16/16; curve-side regression identical; smokes; `--selftest` at exactly 0.

---

## 1. The hue: luminance mode removes bike's hue error on both sets

Hue error on saturated pixels (°), mean ± sd over seeds 0–2:

| Scene | LOM `rung2id` | LOM `rung2idlum` | LOM `detach` | varying `rung2id` | varying `rung2idlum` | varying `detach` |
|---|---|---|---|---|---|---|
| buu | 7.35 ± 0.25 | 4.78 ± 0.10 | 8.23 | 9.33 ± 0.33 | 5.75 ± 0.25 | 9.68 |
| chair | 16.68 ± 0.16 | 19.12 ± 0.13 | 18.33 | 12.55 ± 0.83 | 15.00 ± 0.56 | 14.89 |
| sofa | 7.64 ± 0.19 | 7.14 ± 0.53 | 11.21 | 8.87 ± 0.32 | 8.34 ± 0.45 | 8.63 |
| **bike** | **33.58 ± 0.47** | **9.60 ± 0.77** | 31.78 | **27.01 ± 5.56** | **9.34 ± 0.95** | 12.34 |
| shrub | 10.22 ± 0.03 | 6.38 ± 0.02 | 9.16 | 9.15 ± 0.05 | 5.81 ± 0.01 | 8.74 |
| pooled | 15.09 | 9.40 | 15.74 | 13.38 | 8.85 | 10.85 |

**Bike, `rung2idlum` − `rung2id`:**
- −24.0° on LOM (−37.7×) and −17.7° on the varying set (−4.4×).
- On both sets bike ends below `detach`.
- The plan expected at most a partial fix: part of the varying-set gap looked like `detach`'s own drop from 31.8° to 12.3°. The result is larger than that expectation.

**Other scenes:**
- Hue error falls in buu and shrub (real) and in sofa (within noise).
- Hue error rises in chair: +2.4° on LOM and +2.5° on the varying set, both real.

Because F, s and γ are bit-identical between the arms, this is a *measured*, isolated result: **in Rung 2, applying the shared curve per channel causes bike's hue error.**

*Inferred:*
- A lifting curve applied per channel raises a pixel's darkest channel relatively more than its brightest. That compresses the channel ratios, which desaturates.
- It also moves the HSV hue, because hue depends on (mid − min)/(max − min), and a nonlinear curve changes that ratio.
- Luminance mode keeps the ratios, and with them the hue.

## 2. The cost: buu and sofa get worse; pooled image quality does not improve

`rung2idlum` − `rung2id`, paired by seed:

| Scene | Set | bm PSNR | raw PSNR | SSIM | LPIPS | sat ratio (id → lum) | pseudo bm PSNR | clipped share (lum) |
|---|---|---|---|---|---|---|---|---|
| buu | LOM | −2.24 (−9.4×) * | −2.13 * | −0.0108 * | +0.0182 * | 0.87 → 1.26 | −2.81 | 8.3% |
| buu | varying | −1.64 (−5.8×) * | −1.82 * | −0.0097 * | +0.0070 | 0.89 → 1.27 | −3.30 | 8.5% |
| chair | LOM | +1.16 (+5.8×) * | −0.26 | −0.0049 * | −0.0030 * | 0.47 → 0.76 | +0.73 | 1.0% |
| chair | varying | +0.84 (+3.8×) * | −0.47 | −0.0109 * | +0.0082 | 0.49 → 0.78 | +0.21 | 2.2% |
| sofa | LOM | −1.10 (−1.4×) | −1.04 * | −0.0040 | +0.0345 * | 0.86 → 1.34 | −1.66 | 0.01% |
| sofa | varying | −1.91 (−3.1×) * | −1.41 * | −0.0074 * | +0.0283 * | 0.95 → 1.40 | −2.51 | 0.01% |
| bike | LOM | +2.50 (+38.6×) * | +0.97 * | −0.0053 * | −0.0351 * | 0.52 → 0.88 | +0.81 | 2.6% |
| bike | varying | +2.24 (+11.3×) * | +0.77 * | −0.0048 * | −0.0281 * | 0.56 → 0.89 | +0.64 | 2.7% |
| shrub | LOM | +0.28 (+19.0×) * | −1.24 * | −0.0404 * | −0.0117 * | 0.73 → 1.07 | +0.02 | 12.4% |
| shrub | varying | +0.30 (+11.2×) * | −1.21 * | −0.0348 * | −0.0383 * | 0.76 → 1.09 | −0.06 | 12.2% |
| **pooled** | LOM | +0.12 (+0.7×) | −0.74 (−4.8×) * | −0.0131 (−6.7×) * | +0.0006 (+0.2×) | 0.69 → 1.06 | −0.58 | 4.9% |
| **pooled** | varying | −0.03 (−0.3×) | −0.83 (−7.2×) * | −0.0135 (−9.8×) * | −0.0046 (−2.2×) * | 0.73 → 1.09 | −1.00 | 5.1% |
| pooled without shrub | LOM | +0.08 (+0.4×) | −0.61 * | −0.0062 (−2.4×) * | +0.0037 (+1.0×) | 0.68 → 1.06 | −0.73 | 3.0% |
| pooled without shrub | varying | −0.12 (−0.9×) | −0.73 * | −0.0082 (−4.8×) * | +0.0039 (+1.6×) | 0.72 → 1.09 | −1.24 | 3.4% |

**What the numbers show:**
- **The scenes split in two.** Bike and chair gain bm PSNR on both sets. Buu loses on both sets, and sofa loses on the varying set (on LOM, −1.10 dB is within 2× noise).
  - Pooled, bm PSNR does not move. Raw PSNR and SSIM get worse beyond 2× noise, with and without shrub.
  - Pooled LPIPS is flat on LOM and slightly better on the varying set (−2.2×). It is worse in buu (LOM) and in sofa (both sets).
- **The validation follows the pseudo-targets.** In every decision scene, the sign of the pseudo-target bm PSNR change matches the sign of the validation change. The cost is already in the pseudo-targets the 3DGS is trained on, not in the 3DGS fit.
- **The saturation ratio rises by +0.29 to +0.49 in every scene** (prediction over GT; 1 is right).
  - Where `rung2id` was strongly undersaturated (chair 0.47, bike 0.52), luminance mode moves it toward 1 and bm PSNR rises.
  - Where `rung2id` was already close (buu 0.87, sofa 0.86), luminance mode overshoots (1.26, 1.34) and bm PSNR falls.
- **Clipping does not explain the split.** Sofa loses with almost no clipped channel values (0.01%), and shrub gains with the most (12.4%).

**Interpretation (*inferred*):**
- Luminance mode keeps each low-light photo's own chroma. Its outputs' saturation ratio (0.76 to 1.34 on LOM) is roughly the photo's saturation relative to the GT.
- The camera's low-light saturation matches the GT in some scenes and not in others, so it is not a reliable target.
- Per-channel F desaturates every scene instead (0.47 to 0.87). **Neither colour handling is right everywhere.**
- Angle error rises in buu (+1.6°) and sofa (+0.8°, LOM), which fits a colour cast carried over from the dark photo.
- The plan's noise risk (plan section 5: multiplied colour noise) shows in SSIM, which falls in every scene but sofa on LOM. It shows in LPIPS only in buu and sofa.

## 3. Adoption: `rung2id` stays M6's base

The pre-registered rule needs both "fixes" and pooled SSIM and LPIPS not worse beyond 2× noise on either set. Both conditions fail:
- the verdict is "fixes at a cost";
- pooled SSIM is worse on LOM (−6.7×) and on the varying set (−9.8×), and still worse without shrub.

So M6 (Rung 3) builds on `rung2id`, as decided on 2026-11-25 for this case, and E5 is reported as an ablation.

## 4. Against `detach` (context; exploratory)

`rung2idlum` − `detach` on LOM (`results/rung2/summary_e5_h1/`; H1 rule as in M5):

| Scene | bm PSNR | SSIM | LPIPS | H1 checks |
|---|---|---|---|---|
| buu | −1.72 (−6.6×) * | −0.0050 (−1.5×) | −0.0269 (better) | fails bm PSNR |
| chair | +1.57 (+10.3×) * | +0.0117 * | −0.0294 * | meets |
| sofa | −0.23 (−0.3×) | +0.0047 | +0.0071 (+0.6×) | meets |
| bike | +2.61 (+35.7×) * | +0.0192 * | −0.0240 * | meets |
| pooled | +0.54 (+3.2×) * | −0.0038 (−1.7×) | −0.0063 (−0.7×) | 11 of 12 |

For comparison, `rung2id` − `detach` meets 12 of 12 on LOM (pooled +0.41 dB, +6.0×).

On the varying set, `rung2idlum` beats `detach` by +2.65 dB pooled (+5.2×), about as much as `rung2id` does (+2.69 dB, +5.2×). Rung 2's win on the varying set does not depend on the colour handling.

This is exploratory. It changes neither H1 (`cdf` default, not supported) nor H1b (supported).

## 5. Overexposure (exploratory): per-channel F was helping here

`rung2idlum` − `rung2id` on `over_exp` (pre-registered as exploratory; nothing was tuned on it):

| Scene | bm PSNR | LPIPS | hue err (°) | sat ratio: `rung2id` → `rung2idlum` (`detach`) |
|---|---|---|---|---|
| buu | −2.05 (−24.8×) * | +0.1012 * | −3.8 * | 1.52 → 0.26 (1.24) |
| chair | −0.54 (−4.1×) * | +0.0214 * | −4.0 * | 1.55 → 0.54 (1.55) |
| sofa | −1.11 (−5.0×) * | +0.0614 * | −5.6 * | 1.63 → 0.47 (1.60) |
| bike | −1.67 (−10.6×) * | +0.0830 * | −4.6 * | 1.34 → 0.47 (1.34) |
| shrub | −2.41 (−102.1×) * | +0.0881 * | −0.9 * | 0.80 → 0.47 (0.81) |
| pooled | −1.56 (−19.2×) * | +0.0710 * | −3.8 * | 1.37 → 0.44 (1.31) |

- **Reading:** colour handling does **not** account for part of the overexposure gap. Luminance mode makes every scene worse.
  - Without shrub: −1.34 dB (−13.0×).
  - It widens `rung2id`'s pooled gap to `detach` from −1.20 to −2.75 dB, a "fraction closed" of −1.30.
  - Pooled bm PSNR: official 21.69 (M5), `detach` 20.75, `rung2id` 19.55, `rung2idlum` 17.99.
- **H1 rule, `rung2idlum` − `detach` on `over_exp`:** 1 of 12 checks meet (`results/overexp/summary_e5/`).
- *Inferred:*
  - The overexposed photos are washed out, and luminance mode keeps their low saturation (0.44 of the GT).
  - Per-channel darkening re-saturates them, which overshoots (1.37) but lands closer in PSNR terms.
  - So on overexposure the per-channel side effect works in Rung 2's favour, and the overexposure limitation of M5 E6 stays with the one-γ prior, not with the colour handling.

## 6. What the final report says

- **Measured, isolated by construction (identical F, s, γ):** applying Rung 2's shared curve per channel is what causes bike's hue error. Applying it to the grey and scaling the channels removes the error on both sets: 33.6° → 9.6° on LOM, 27.0° → 9.3° on the varying set, below `detach`'s.
- **But the luminance variant is not a better model overall.**
  - It raises every scene's saturation ratio by 0.29 to 0.49.
  - That helps the two undersaturated scenes (chair, bike: +0.8 to +2.5 dB) and hurts the two that were already close (buu −1.6 to −2.2 dB; sofa −1.9 dB on the varying set).
  - Pooled bm PSNR is unchanged, SSIM is worse, and on overexposure it is clearly worse (−1.56 dB).
  - By the pre-registered rule it is reported as an ablation, and Rung 3 builds on `rung2id`.
- **Limitation, stated as such:** neither colour handling recovers the GT saturation in all scenes. Per-channel application desaturates; luminance inherits the camera's low-light saturation.
  - A blend between the two, or a learned chroma gain, was not tested.
  - Choosing one after seeing these results would be post hoc.

## 7. Next

- `M6_rung3_plan.md` (Rung 3: local monotone curves) on `rung2id`, with the same 10k steps and learning rates, for Bart's review. It is still pending.
- Send Wei the E5 headline with the M5 results (draft in the git-ignored `wei_drafts/` on request).
