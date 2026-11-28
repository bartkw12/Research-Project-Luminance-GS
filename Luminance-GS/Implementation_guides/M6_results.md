# M6 results: Rung 3 (two zone curves, `rung3`) on the identity base

Written 2026-11-27 by Claude Code for Bart, and as source material for the final report and the next update to Wei.
- **Plan:** `M6_rung3_plan.md`.
- **Pre-registration:** the readings were pre-registered in `docs/RESEARCH_LOG.md` before any M6 code or run ("M6 step 1", 2026-11-26). The 4.3 predictions were fixed before the batch ("M6 step 3, test 5", 2026-11-27). The analysis checks the predictions file's sha256 (`991f45a9…3974f`) and found it unchanged.
- **Labels:** every number is *measured* unless marked *inferred* or *post hoc*.
- **Statistics:** pairs are matched by seed. Noise = sqrt(mean of the arms' seed variances). An effect is real above 2× noise (*).
- **Pooled:** the mean over the five scenes per seed, checked against pooled without shrub. Shrub never decides.
- **bm PSNR** = brightness-matched PSNR.

**What M6 changes:**
- **The model:** Rung 3 replaces Rung 2's single shared curve F with two shared curves on the same 32 knots, F_sh (shadow) and F_lit (lit). It mixes them per pixel:
  `L_k(x, p) = w_sh(p) F_sh(min(1, 2^{s_k} x)) + w_lit(p) F_lit(min(1, 2^{s_k} x))`, on R, G and B.
- **The weights:**
  - They are fixed per view at s_init, from a 16-pixel block average of the exposure-aligned grey.
  - The ramp runs between the 25th and 75th percentiles of the log-brightness.
- **The loss:** each curve is fitted to its own zone's histogram-equalisation target, weighted by the zone's share of pixels. One γ prior is shared by both curves.
- **What is unchanged from the base (`rung2id`):** identity init, detached curves, free s_k, 10k steps, and the same learning rates.
- **Parameters:** 2K + N + 1 = 87–108 per scene, against `rung2id`'s 55–76.

## Verdicts at a glance

| Question | Pre-registered verdict | Headline | Source |
|---|---|---|---|
| H4 (plan 4.1): does Rung 3 change quality against `rung2id`? | **Hurts** | pooled bm PSNR −3.45 dB on LOM (−45.1×) and −3.64 dB on the varying set (−31.2×); every decision scene worse on both sets | `results/rung3/summary/` |
| Bike's hue (4.2) | **Fixes at a cost** | LOM 33.6° → 23.4° (−10.2°, −7.0×); bike bm PSNR −0.70 dB (LOM) and −0.92 dB (varying) | same |
| Mechanism check (4.3): does validation colour move as the photo-only preview predicted? | **As predicted** | saturation ratio up in all four predicted scenes (buu +0.26, chair +0.18, sofa +0.24, bike +0.02) | same |
| Exploratory: H1 rule, `rung3` − `detach`, LOM | 5 of 12 checks meet; not supported | pooled −3.04 dB (−40.5×) | `results/rung3/summary_h1/` |
| Exploratory: overexposure | `rung3` − `rung2id` +0.56 dB pooled (+7.4×); H1 rule 8 of 12, not supported | still −0.63 dB below `detach` and −1.58 dB below the official code | `results/rung3/summary/`, `results/overexp/summary_m6/` |
| Post hoc: why it hurts | The zone targets flatten the lit/shadow contrast below the ground truth's; one global curve already matches it | lit/shadow brightness ratio, GT 1.33–1.70; `rung2id` 1.28–1.57; `rung3` 1.12–1.27 | section 2 |

**Scope:**
- **Runs:** 45 training runs (`rung3`: LOM, the varying set and `over_exp`, seeds 0–2). All were `ok` and all were re-scored against `results/rung0/<scene>_baseline_seed0`.
- **Code:** commit `419ffc7` (training); the analysis is `rung2_ablation.py --mode rung3` (uncommitted at the time of writing).
- **Comparison runs:** `rung2id` on LOM (M3 head-start, `results/rung2/`), on the varying set (M5 E1) and on `over_exp` (M5 E6). `detach` is the context arm.
- **Tests before the batch:**
  - unit tests 9 of 9, plus the CUDA check;
  - the `rung2id` path is unchanged (curve side bit-identical);
  - smokes on all three sets;
  - the preview passed its stop rule;
  - `--selftest` gives exactly 0.
- **Run time:** 4.0–5.2 min per run (shrub 6.4–8.9), about +0.3 min against `rung2id` (+5.0× on LOM).

---

## 1. Quality: Rung 3 is worse on both sets, by a lot

`rung3` − `rung2id`, paired by seed (`results/rung3/summary/tables.md`):

| Scene | Set | bm PSNR | raw PSNR | SSIM | LPIPS | pseudo bm PSNR |
|---|---|---|---|---|---|---|
| buu | LOM | −4.88 (−22.0×) * | −1.57 * | −0.0318 * | +0.0398 * | −5.69 |
| buu | varying | −4.45 (−13.6×) * | −1.46 * | −0.0347 * | +0.0354 * | −5.96 |
| chair | LOM | −3.61 (−15.6×) * | −1.65 * | −0.0315 * | −0.0054 (−2.1×) * | −3.89 |
| chair | varying | −4.03 (−20.0×) * | −1.99 * | −0.0417 * | +0.0098 (+2.6×) * | −4.31 |
| sofa | LOM | −7.69 (−17.8×) * | −4.01 * | −0.0316 * | +0.0251 * | −8.98 |
| sofa | varying | −8.40 (−14.0×) * | −4.57 * | −0.0491 * | +0.0511 * | −9.02 |
| bike | LOM | −0.70 (−9.0×) * | +0.13 * | −0.0166 * | −0.0183 (−3.1×) * | −1.28 |
| bike | varying | −0.92 (−12.3×) * | −0.11 | −0.0148 * | −0.0066 | −1.35 |
| shrub | LOM | −0.40 (−31.8×) * | −0.09 * | +0.0025 * | +0.0030 | −0.60 |
| shrub | varying | −0.39 (−11.5×) * | −0.07 | +0.0011 | +0.0012 | −0.62 |
| **pooled** | LOM | **−3.45 (−45.1×) *** | −1.44 (−13.0×) * | −0.0218 (−12.8×) * | +0.0088 (+6.7×) * | −4.09 |
| **pooled** | varying | **−3.64 (−31.2×) *** | −1.64 (−15.6×) * | −0.0278 (−10.7×) * | +0.0182 (+5.1×) * | −4.25 |
| pooled without shrub | LOM | −4.22 (−44.3×) * | −1.78 * | −0.0279 * | +0.0103 (+6.1×) * | −4.96 |
| pooled without shrub | varying | −4.45 (−31.5×) * | −2.03 * | −0.0351 * | +0.0224 (+5.4×) * | −5.16 |

Absolute pooled bm PSNR: LOM `rung3` 20.21 ± 0.08 against `rung2id` 23.67 ± 0.07 (`detach` 23.25, official 22.00); varying `rung3` 20.07 ± 0.09 against `rung2id` 23.71 ± 0.14 (`detach` 21.03).

**What the numbers show:**
- **Every scene loses bm PSNR on both sets, beyond 2× noise.** The size follows how far the two curves split (section 2):
  - large in buu, chair and sofa (−3.6 to −8.4 dB; curve split 0.34–0.39);
  - small in bike and shrub (−0.4 to −0.9 dB; split 0.17).
- **SSIM falls in every decision scene.** LPIPS rises in buu and sofa, and falls slightly in chair (LOM) and bike (LOM). Pooled, both get worse on both sets.
- **The validation follows the pseudo-targets.** In every scene and set, the pseudo-target bm PSNR change has the same sign as the validation change, and a similar size. The loss is already in the targets the 3DGS is trained on, not in the 3DGS fit.
- **The training did what the design says.** The curve losses at 10k equal the photo-only preview's replay within 3% (buu 0.00966 against 0.00992; chair 0.00721 against 0.00722). So this is the design's outcome, not a training failure.

**Pre-registered verdict (plan 4.1): hurts.**
- **Help signals:** none. Pooled bm PSNR and LPIPS get worse on both sets, with and without shrub.
- **Harm 1** (any decision scene below −2× noise): fires in all four decision scenes on both sets.
- **Harm 2** (pooled bm PSNR down or LPIPS up): fires for both metrics on both sets.
- **Harm 3** (consistency, below): does not fire.
- **Harm 4** (exposure handling, section 5): does not fire.

## 2. Why: the zone targets flatten the lit/shadow contrast

The two curves on LOM, at grid levels 1, 2, 4, 8, 16, 32 and 64 (×255; mean over seeds):

| Scene | F_sh (shadow) | F_lit (lit) | `rung2id` F | max \|F_sh − F_lit\| |
|---|---|---|---|---|
| buu | 40, 51, 68, 91, **150**, 216, 230 | 42, 52, 63, 72, **85**, 152, 231 | 42, 52, 66, 83, **119**, 188, 230 | 0.386 |
| chair | 59, 78, 113, 162, 213, 230, 238 | 55, 65, 77, 89, 141, 230, 239 | 57, 72, 93, 121, 174, 229, 239 | 0.344 |
| sofa | 43, 55, 72, 111, 199, 224, 234 | 47, 58, 70, 80, 103, 195, 234 | 47, 58, 72, 96, 153, 210, 234 | 0.375 |
| bike | 69, 91, 131, 198, 231, 239, 244 | 63, 79, 103, 155, 221, 237, 244 | 66, 85, 117, 176, 226, 238, 244 | 0.175 |
| shrub | 81, 108, 148, 193, 227, 237, 244 | 67, 84, 110, 151, 209, 236, 244 | 74, 96, 129, 171, 218, 236, 244 | 0.171 |

- γ is unchanged (for example buu 0.1650 against 0.1647).
- The zone shares are close to ½ (shadow 0.44–0.54).
- Figure: `results/rung3/summary/figures/zone_curves_lom.png`.

**How the curves flatten contrast:**
- Each zone's target equalises that zone's own histogram, so each zone is spread over the full output range.
- At mid-dark levels the shadow curve therefore lifts far more than the lit curve. In buu, level 16 maps to 150 on F_sh, 85 on F_lit and 119 on `rung2id`'s F.
- A shadowed surface at level 16 ends up brighter than a lit one at level 16, and close to a lit one at level 32 (152). So the brightness difference between shadowed and lit regions shrinks.

**Post hoc check against the ground truth** (run after the readings; it decides nothing).
- **Quantity:** the lit/shadow brightness ratio with the preview's definition: the zone-weighted mean grey per training view, with the zone weights fixed at s_init. Then the mean over views of each zone, lit over shadow.
- **Sources:**
  - photo and GT (`high/`): scratchpad script `m6_gt_contrast.py`;
  - pseudo-targets: the preview's replay (`results/rung3/preview/numbers.json`, read only).
- **Check:** the script's photo ratios equal the preview's to 3 decimals.

| Scene | photo | **GT** | `rung2id` pseudo-target | `rung3` pseudo-target | `rung3` − `rung2id` bm PSNR (LOM) |
|---|---|---|---|---|---|
| buu | 2.07 | **1.56** | 1.54 | 1.23 | −4.88 |
| chair | 2.32 | **1.70** | 1.57 | 1.27 | −3.61 |
| sofa | 2.27 | **1.59** | 1.56 | 1.24 | −7.69 |
| bike | 1.72 | **1.33** | 1.29 | 1.12 | −0.70 |
| shrub | 1.85 | **1.56** | 1.28 | 1.13 | −0.40 |

- **One global curve already gets this contrast about right.** `rung2id` is within 0.04 of the GT in buu, sofa and bike, and 0.13 low in chair. Shrub, which never decides, is 0.28 low.
- **Rung 3 pushes it well below the GT** in every scene. No GT view has its shadow zone brighter than its lit zone (0 of 22–43 views per scene). No pseudo-target view does either (preview), so the lighting order is kept and only its strength is lost.
- *Inferred:* this explains the size and the pattern of the loss. Brightness matching corrects one global gain, but not a lit/shadow ratio that is 15–25% too flat. The scenes whose curves split most (buu, chair, sofa) lose most.
- This is the plan's risk 1 ("the zone targets flatten the contrast between shadowed and lit regions … a legitimate 'hurts' result, not a bug"). The preview showed the flattening before the batch (lit/shadow 1.12–1.27 against `rung2id`'s 1.28–1.57). It was reported, not a stop criterion, and no GT was read at that point.
- **Limit of the design (plan section 5):** locality and the zone-wise equalisation change together. These results cannot say whether local curves fitted to a different target (one that keeps the global contrast) would help.

## 3. Bike's hue (plan 4.2): fixes at a cost

Hue error on saturated pixels (°), mean ± sd over seeds 0–2:

| Set | `rung2id` | `rung3` | `detach` | `rung3` − `rung2id` |
|---|---|---|---|---|
| LOM | 33.58 ± 0.47 | 23.42 ± 1.99 | 31.78 | −10.16 (−7.0×) * |
| varying | 27.01 ± 5.56 | 23.13 ± 2.47 | 12.34 | −3.88 (−0.9×) |

- **Applying the rule:**
  - the LOM hue error falls beyond 2× noise;
  - it does not rise on the varying set;
  - but bike's bm PSNR falls on both sets (−0.70 dB, −9.0×; −0.92 dB, −12.3×).
  - So the verdict is **fixes at a cost**.
- **For scale:** E5's luminance mode took bike's LOM hue to 9.6° while bike gained +2.50 dB. Rung 3 removes less than half as much hue error (−10.2° against E5's −24.0°), and costs bike bm PSNR instead.
- **Other scenes (LOM):** hue error falls in buu (−1.03°) and shrub (−2.28°), and rises in chair (+1.05°) and sofa (+1.10°), all beyond 2× noise.
- *Inferred:* the lit-zone curve lifts bike's dark levels less, so per-channel application compresses the channel ratios less there. That is the mechanism the design aimed at, but it comes with the flatter contrast of section 2.
- Figure: `results/rung3/summary/figures/bike_29_input_wlit_rung2id_rung3_gt.png` (view 29: input, w_lit, `rung2id` and `rung3` pseudo-targets, GT).

## 4. Mechanism check (plan 4.3): as predicted

| Scene | preview Δ saturation (chromatic pixels; logged before the batch) | prediction | LOM validation saturation ratio, `rung3` − `rung2id` | result |
|---|---|---|---|---|
| buu | +0.102 | up | +0.256 (+62.0×) * (0.866 → 1.122) | holds |
| chair | +0.096 | up | +0.180 (+60.3×) * (0.465 → 0.645) | holds |
| sofa | +0.069 | up | +0.244 (+50.5×) * (0.856 → 1.100) | holds |
| bike | +0.024 | up | +0.022 (+3.1×) * (0.522 → 0.544) | holds |

- Every predicted direction holds beyond 2× noise. Bike had no hue prediction (its preview drift change, −0.72°, was under the 1° minimum).
- **What it means:** the photo-only preview predicted how the curves move validation colour, so the curves act on the 3DGS through the mechanism the design assumes.
  - The design is not wrong in how it works. It is wrong in what its targets ask for (section 2).
  - Saturation moving toward 1 does not rescue bm PSNR here. Chair goes 0.47 → 0.65 (where E5 gained +1.16 dB from a similar move) and still loses −3.61 dB. *Inferred:* the contrast loss outweighs the colour gain.
- Figure: `results/rung3/summary/figures/saturation_predicted_vs_measured.png`.

## 5. Exposure handling and cross-view consistency are kept

**Harm 4 (varying set): does not fire.**

| | pooled leak (limit 0.1) | pooled s_k slope on u_k (range −1.1 to −0.9) |
|---|---|---|
| `rung3` | 0.008 | −1.002 |
| `rung2id` | 0.009 | −1.004 |

Rung 3 keeps Rung 2's exposure handling. The pseudo-targets still do not follow the gains.

**Harm 3 (LOM; D5): does not fire.** The per-scene relative change in pseudo-target inconsistency (`rel_l1_pseudo`, shared reference depth):

| buu | chair | sofa | bike | decision-scene mean (decides; limit +20%) | shrub (reported) | five-scene pooled (reported) |
|---|---|---|---|---|---|---|
| −0.9% | +31.0% | +27.0% | +8.4% | **+16.4%** | +4.8% | +10.1% |

- Chair and sofa individually exceed 20%, but the pre-registered measure is the four-scene mean, and it stays under the limit.
- *Inferred:* the zone boundary sits at different image positions in different views, so the same surface can get different targets (plan risk 2).

**The s_k drift and the fixed weights (D6):**
- Rung 3's learned exposures drift further from s_init than `rung2id`'s, most in buu: 0.030 stops RMS on LOM against 0.0026, and max |s_rung3 − s_rung2id| = 0.059 stops. Elsewhere the drift is 0.0016–0.0099 stops.
- Against buu's zone width (0.97 stops), that is about 3%, above the "under 1%" the plan assumed from `rung2id`'s drift.
- The preview measured what it would change: weights that followed the final s would differ by at most 0.061 in buu. This is reported only; it does not bear on the verdict, which is driven by the targets (section 2).

**Also measured:** about 4% more Gaussians on LOM (+9,826 pooled, +2.2×), and about 0.3 min more per run.

## 6. Against `detach` (context; exploratory)

`rung3` − `detach` on LOM (`results/rung3/summary_h1/`; the H1 rule as in M5):

| Scene | bm PSNR | SSIM | LPIPS | H1 checks |
|---|---|---|---|---|
| buu | −4.36 (−17.6×) * | −0.0260 * | −0.0053 (−0.8×) | fails bm PSNR, SSIM |
| chair | −3.19 (−16.8×) * | −0.0150 * | −0.0318 * | fails bm PSNR, SSIM |
| sofa | −6.82 (−15.8×) * | −0.0229 * | −0.0023 (−0.5×) | fails bm PSNR, SSIM |
| bike | −0.59 (−7.1×) * | +0.0079 * | −0.0072 (−1.1×) | fails bm PSNR |
| pooled | −3.04 (−40.5×) * | −0.0125 * | +0.0019 (+0.2×) | 5 of 12 |

- For comparison, `rung2id` − `detach` meets 12 of 12 (pooled +0.41 dB, +6.0×).
- **On the varying set,** Rung 3 loses Rung 2's win over `detach`:
  - `rung3` − `detach` is −0.95 dB pooled (−1.9×, within noise);
  - `rung2id` − `detach` is +2.69 dB (+5.2×).
  - Per scene, `rung3` is still ahead of `detach` in shrub (+1.28 dB *), level in chair and bike, and behind in buu (−2.75 dB *) and sofa (−3.58 dB *).
- This is exploratory. It changes neither H1 (`cdf` default, not supported) nor H1b (`rung2id`, supported).

## 7. Overexposure (exploratory): Rung 3 helps a little, from a low base

`rung3` − `rung2id` on `over_exp` (pre-registered as exploratory; no direction was predicted; nothing was tuned on it):

| Scene | bm PSNR | SSIM | LPIPS | `rung3` / `rung2id` / `detach` / official bm PSNR |
|---|---|---|---|---|
| buu | −0.40 (−3.8×) * | +0.0001 | +0.0018 | 18.11 / 18.51 / 23.55 / 21.60 |
| chair | +0.17 (+1.2×) | +0.0117 * | +0.0016 | 19.13 / 18.97 / 19.10 / 21.24 |
| sofa | +1.67 (+6.6×) * | +0.0096 * | −0.0035 | 19.53 / 17.85 / 18.37 / 20.33 |
| bike | +1.27 (+7.0×) * | +0.0056 * | +0.0022 | 22.10 / 20.83 / 21.04 / 24.84 |
| shrub | +0.11 (+2.6×) * | −0.0018 | −0.0005 | 21.69 / 21.58 / 21.67 / 20.44 |
| pooled | +0.56 (+7.4×) * | +0.0050 (+6.0×) * | +0.0003 | 20.11 / 19.55 / 20.75 / 21.69 |

- Rung 3 closes about half of `rung2id`'s gap to `detach` (−1.20 → −0.63 dB). It is still below `detach` (−0.63 dB, −6.4×) and the official code (−1.58 dB).
- **H1 rule, `rung3` − `detach` on `over_exp`:** 8 of 12 checks meet; not supported (buu fails all three, sofa fails LPIPS). Here `detach` itself trails the official code (−0.94 dB), so this is a weaker bar (M5 3.6).
- **A wrong inference, corrected:** the smoke entry (2026-11-27, tests 3 and 4) inferred that Rung 3 "should differ little from `rung2id`" on `over_exp`, because buu's zones there are only 0.09 stops apart. That was not a reading. The measured effect is +0.56 dB pooled, though buu, the scene measured in the smoke, does move little and the other way (−0.40 dB).
- *Inferred, untested:* in washed-out photos the zone targets add contrast rather than remove it, which is the direction overexposure needs. Rung 3 does not change H1-over (not supported) or the overexposure limitation of M5 E6.

## 8. What M6 can and cannot show (from plan section 5)

- **Can show:** with the same training budget, this zone-wise local tone model is clearly worse than one global curve on low light. It keeps Rung 2's exposure handling and stays within the pre-registered consistency limit.
- **Cannot show:**
  - **Locality and targets together:** locality and the zone-wise equalisation targets change together. A "hurts" cannot be split between "local curves" and "zone-wise histogram targets". Section 2's post hoc check points at the targets (*inferred*).
  - **One configuration:** M = 2, a 16-pixel block, the 25th/75th-percentile centres and one shared γ, fixed in advance and not tuned. Choosing another one after these results would be post hoc.
  - **Weights:** learned or attached weights (3D feedback into the mix) were not tested.
  - **Seeds:** the zone weights do not depend on the seed. F and s depend on it only through the view order, so seeds re-test mainly the 3DGS randomness.
  - **Data:** five LOM scenes; the varying set has the same scenes and ground truth.

## 9. What the final report says

- **Rung 3 hurts, by the pre-registered rule.**
  - **The test:** a per-pixel mix of two shared monotone curves, weighted by exposure-aligned local brightness and each fitted to its zone's histogram, against the single global curve of Rung 2 (`rung2id`).
  - **Result:** it lowers bm PSNR by 3.45 dB pooled on LOM and 3.64 dB on the varying set, every decision scene beyond 2× noise. SSIM and LPIPS also get worse pooled.
- **Measured, not inferred:** the loss is already in the pseudo-targets, and the curves train exactly as the photo-only replay predicts. The validation colour moves as predicted (saturation up in all four predicted scenes).
- **The likely cause (*post hoc*, measured against the GT):**
  - zone-wise histogram equalisation flattens the brightness ratio between lit and shadowed regions to 1.12–1.27;
  - the ground truth keeps 1.33–1.70, which one global curve already matches within 0.04 in three of four decision scenes.
  - **Framing:** under this prior, one global curve is not the bottleneck for low light. Making the histogram prior local removes contrast the scene needs.
- **Bike's hue:** Rung 3 reduces bike's hue error (33.6° → 23.4° on LOM) at a bm PSNR cost. That supports E5's finding that per-channel application of a strong lift causes the error, but this route is worse than the luminance one.
- **Kept:** exposure handling (leak 0.008, slope −1.00) and the consistency limit.
- **Lost:** Rung 2's +2.7 dB margin over `detach` on the varying set.
- **Overexposure (exploratory):** +0.56 dB over `rung2id`, still below `detach` and the official code.
- **No default changes.** The report's model stays Rung 2: H1 on the `cdf` default (not supported) and H1b on `rung2id` (supported), side by side. Rung 3 is reported as a pre-registered negative result.
- **Limitation, stated as such:** one configuration of the local model was tested, and its locality and targets changed together. A local model with a contrast-preserving target (for example one global target with local slopes) was not tested.

## 10. Next

- Protect the M6 outputs (`results/rung3/summary/`, `results/rung3/summary_h1/`, `results/overexp/summary_m6/`, this file): rule 6, the `ask` list and `protected.FINISHED`.
- Send Wei the M6 headline (draft in the git-ignored `wei_drafts/` on request), with E5's if not yet sent.
- Optional A2 (plan step 6, `rung2_replay.py --mode rung3long`, about 10 min of CPU): would only say whether Rung 3 is early-stopped like its base. After a "hurts", it decides nothing.
- M7: final tables and figures; freeze on Sat 5 Dec.
