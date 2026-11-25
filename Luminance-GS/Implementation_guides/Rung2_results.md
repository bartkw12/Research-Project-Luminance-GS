# Rung 2 results (M3): one shared monotone curve plus one exposure per view

> M5 follow-up (H3, H1b, H2b, no prior, long replay, overexposure): see `M5_results.md` (2026-11-25).

## The model

Rung 2 replaces the official per-view curve machinery with one explicit model:

L_k(x) = F(min(1, 2^{s_k} x))

- **What it replaces:** two cross-attention networks plus a per-view colour basis, about 255k parameters.
- **F:** one shared monotone curve, with K = 32 knots spaced quadratically, F(0) = 0 and F(1) = 1.
- **s_k:** one exposure per training view, with their mean fixed at 0.
- **γ:** one global value that carries the official shape prior on F.
- **Size:** K + N + 1 = 55 to 76 curve parameters per scene.

Defaults (plan `M2_rung2_plan.md` section 2.2; decisions in the research log, 2026-11-11 and 2026-11-13):
- F is detached from the image loss, so it is fitted to the photos' histograms only.
- The colour basis is plain RGB.
- The curve loss targets each view's histogram-equalisation CDF, shifted by one bin.

## What M3 ran

- **LOM:** `rung2` on 5 scenes × seeds 0–2, 15 runs in `results/rung2/`. They are compared with `detach` and the official code ("official") from Rung 0 (`results/rung0/`).
- **Varying-exposure set:** `rung2` and `rung2exp`, 30 runs in `results/vexp/`. With M1's `detach` and M2's `detachexp` they form a 2×2: curve model (`detach` or Rung 2) × dark branch (shared, or exposure-aware = "exp").
- **When:** all 45 runs finished `ok` on 2026-11-14 (17:18–20:48, commit `c304662`). Each was re-scored with the LOM reference depth (`results/rung0/<scene>_baseline_seed0`).
- **Follow-up:** the plan's pre-declared head-start check (`rung2id`, 5 runs, 2026-11-15) is reported under Q1. Its results are in [`summary_headstart/`](../results/rung2/summary_headstart/). The 3-seed confirmation (10 more runs, 2026-11-16) is in [`summary_headstart_seeds012/`](../results/rung2/summary_headstart_seeds012/).

**Where things are:**
- **Outputs:** [`results/rung2/summary/`](../results/rung2/summary/), mainly [`tables.md`](../results/rung2/summary/tables.md) and `figures/`. Regenerate with `python rung2_analyze.py`, about 2 minutes on the CPU.
- **Seed-0 snapshot** from the night of the runs: [`summary_seed0/`](../results/rung2/summary_seed0/).
- **Plan:** [`M2_rung2_plan.md`](M2_rung2_plan.md), sections 5 and 6.
- **Log:** `docs/RESEARCH_LOG.md`, entries 2026-11-14 and 2026-11-15.

**How to read the claims** (as in M1):
- *Measured* means read from the runs; *inferred* marks an interpretation.
- Effects are paired by seed. Noise is the square root of the mean of the two arms' seed variances, with 3 seeds each.
- An effect is real above 2× noise, written (×) with bold or * for real.
- "bm" is brightness-matched PSNR: the render is scaled to the ground truth's mean brightness, then PSNR is taken (`diagnostics.psnr_gt_mean`).
- "Pooled" is the mean over the five scenes, per seed. Shrub is reported but not used to decide (protocol rule 7).

**No setting was tuned on the answer.**
- The learning rate was chosen by the plan's gap-to-optimum rule on the training photos.
- The knot spacing was chosen on training photos.
- The one preview table that uses normal-light images (P5) chose nothing (log, 2026-11-13 and 2026-11-14).
- The H1 and H2 rules were fixed on 2026-11-11, before any Rung 2 training run.

## Short answers

- **Q1. H1 (LOM): not supported by the pre-registered rule.**
  - 11 of the 12 decision checks pass (4 scenes × bm PSNR, SSIM, LPIPS).
  - **The one failure:** bike's bm PSNR is 0.155 dB below `detach`, which is −3.8× noise against a bar of −2×. Bike's noise is only 0.041 dB, so its bar is −0.082 dB.
  - **Pooled, the two models are indistinguishable:** bm PSNR −0.02 dB (−0.3×). Without shrub, LPIPS is better (−0.0095, 4.1×).
  - **Cost:** 55–76 curve parameters instead of about 255k, and 4.6 instead of 7.8 minutes per run.
  - **Head-start check** (plan 5.3), confirmed with seeds 1–2: starting F at the identity instead of the `cdf` init raises bm PSNR by **+0.44 dB pooled (6.2×)** and +0.26 dB on bike (10.9×).
    - With it, Rung 2 meets all 12 H1 checks and beats `detach` by +0.42 dB pooled (6.0×).
    - The identity-init F never converges to the `cdf` one; it stays less lifted at levels 1–2.
    - The training objective prefers the default.
    - So this is an exploratory (post-hoc) result, not a new default, and it does not change the H1 verdict.
- **Q2. Dark branch:** the exposure-aware dark branch restores most of the dark-photo fit lost on the varying set, for both curve models. Rung 2 gains +7.63 dB (50.6×) and `detach` +5.77 dB (17.9×). Its effect on the image metrics is small: +0.28 dB bm PSNR pooled (0.7×), real only within the Rung 2 row (+0.31, 2.7×).
- **Q3. H2 (varying set): supported.**
  - The curve-model main effect is **+2.11 dB bm PSNR (5.3×)**. LPIPS is better (−0.024, 2.5×).
  - It is positive in all four decision scenes and real in three (buu +1.84 at 1.6×).
  - It is real in both dark-branch columns: shared +2.09 (4.0×), exp +2.14 (9.1×).
- **Q4. Mechanism: yes on both counts.**
  - **Leak near 0:** Rung 2's pseudo-targets let 1% of the exposure difference through (leak 0.012), against `detach`'s 51% (0.512).
  - **Compensating s_k:** its s_k follow the true gains (slope −1.00) within 0.018 stops.
- **Q5. Side effects:**
  - **Better:** colour angle error on both sets; much higher cross-view consistency of the pseudo-targets on the varying set (1.69 vs 5.06 × GT); 40% less time per run.
  - **Worse:** hue error on bike (LOM 38.5° vs 31.8°; varying set 35.4° vs 12.3°); shrub SSIM and LPIPS; render consistency on LOM (sofa).

## Q1. H1 in detail: does Rung 2 match `detach` on LOM?

### The rule, as pre-registered

> H1 is supported if, in each of buu, chair, sofa and bike, `rung2` − `detach` in brightness-matched PSNR is not below −2× noise, and SSIM and LPIPS are not worse by more than 2× noise. (Plan 6.5, decided by Bart on 2026-11-11; shrub reported only.)

That is 12 checks, all of which must pass. "Worse" means lower for bm PSNR and SSIM, and higher for LPIPS. Raw PSNR is reported next to bm PSNR but is not part of the rule.

### The 12 checks

`rung2` − `detach`, paired by seed. The bar is 2× noise on the worse side.

| Scene | Metric | rung2 − detach | Noise | Bar | ×noise | Check |
|---|---|---|---|---|---|---|
| buu | bm PSNR | −0.119 dB | 0.226 | −0.451 | −0.5× | meets |
| buu | SSIM | −0.0071 | 0.0036 | −0.0073 | −1.96× | meets (closest pass) |
| buu | LPIPS | **−0.0202** (better) | 0.0086 | +0.0173 | −2.3× | meets |
| chair | bm PSNR | +0.398 dB | 0.201 | −0.402 | +2.0× | meets |
| chair | SSIM | **+0.0065** (better) | 0.0008 | −0.0017 | +7.7× | meets |
| chair | LPIPS | **−0.0164** (better) | 0.0022 | +0.0043 | −7.6× | meets |
| sofa | bm PSNR | −0.387 dB | 0.340 | −0.680 | −1.1× | meets |
| sofa | SSIM | −0.0002 | 0.0035 | −0.0069 | −0.1× | meets |
| sofa | LPIPS | −0.0090 (better) | 0.0071 | +0.0142 | −1.3× | meets |
| **bike** | **bm PSNR** | **−0.155 dB** | **0.041** | **−0.082** | **−3.8×** | **fails** |
| bike | SSIM | −0.0004 | 0.0041 | −0.0081 | −0.1× | meets |
| bike | LPIPS | +0.0076 (worse) | 0.0056 | +0.0112 | +1.35× | meets |

Cell means, mean ± sd over seeds (*measured*, `tables.md` R1a). In the right-hand column, raw PSNR differences are in brackets, with * for real:

| Scene | Official bm | `detach` bm | `rung2` bm | `detach` raw | `rung2` raw (rung2 − detach) |
|---|---|---|---|---|---|
| buu | 26.00 ± 0.33 | 26.37 ± 0.25 | 26.25 ± 0.20 | 21.56 ± 0.52 | 20.02 ± 0.24 (−1.54, 3.8×*) |
| chair | 20.94 ± 0.16 | 23.09 ± 0.12 | 23.49 ± 0.26 | 18.82 ± 0.03 | 19.89 ± 0.33 (+1.07, 4.5×*) |
| sofa | 25.22 ± 0.46 | 27.41 ± 0.40 | 27.03 ± 0.27 | 23.47 ± 0.29 | 23.13 ± 0.40 (−0.34, 1.0×) |
| bike | 19.45 ± 0.13 | 21.31 ± 0.05 | 21.16 ± 0.03 | 16.02 ± 0.10 | 16.35 ± 0.04 (+0.33, 4.2×*) |
| shrub (reported only) | 18.37 ± 0.99 | 18.06 ± 0.50 | 18.22 ± 0.03 | 11.76 ± 0.66 | 11.91 ± 0.03 (+0.15, 0.3×) |
| pooled | 22.00 ± 0.32 | 23.25 ± 0.07 | 23.23 ± 0.07 | 18.33 ± 0.02 | 18.26 ± 0.17 (−0.07, 0.5×) |
| pooled without shrub | 22.90 ± 0.16 | 24.55 ± 0.08 | 24.48 ± 0.08 | 19.97 ± 0.18 | 19.85 ± 0.21 (−0.12, 0.6×) |

Figure: [`fig1_H1_lom.png`](../results/rung2/summary/figures/fig1_H1_lom.png), one dot per seed.

### Bike, the failing check

Validation, per seed (*measured*, `rescore/val_summary.json`):

| Seed | `detach` bm | `rung2` bm | Difference | `detach` raw | `rung2` raw | Difference |
|---|---|---|---|---|---|---|
| 0 | 21.273 | 21.162 | −0.112 | 15.962 | 16.364 | +0.402 |
| 1 | 21.364 | 21.120 | −0.244 | 16.138 | 16.298 | +0.160 |
| 2 | 21.295 | 21.186 | −0.109 | 15.962 | 16.381 | +0.419 |
| mean ± sd | 21.311 ± 0.048 | 21.156 ± 0.034 | **−0.155** | 16.021 ± 0.102 | 16.348 ± 0.044 | **+0.327** |

*Measured:*
- **The deficit is consistent.** All three seeds are negative, and the six runs separate completely: Rung 2's best run (21.186) is below `detach`'s worst (21.273). A Mann-Whitney test gives an exact two-sided p of 0.10, the smallest a 3-against-3 comparison can produce.
- **Bike's bar is by far the tightest.** Its noise is 0.041 dB, against 0.20 to 0.34 dB on buu, chair and sofa, so its bar is −0.082 dB, against −0.40 to −0.68 dB elsewhere. The same −0.155 dB would pass in any of the other three scenes.
- **In size the deficit is small.**
  - It is about 3.6% more squared error.
  - Rung 2 keeps 1.71 of `detach`'s 1.86 dB gain over the official code on bike, about 92% (computed from the R1a means).
- **Raw PSNR goes the other way:** +0.33 dB (4.2×, real). Rung 2's bike renders are slightly closer to the ground truth's mean brightness. Validation render mean ÷ ground-truth mean is 1.239 for Rung 2, 1.257 for `detach` and 1.108 for official. Both detached models render bike too bright.

### How firm the verdict is

**The verdict stands: H1 is not supported.** This section adds context and does not re-test anything.

- **Bike's noise comes from 3 seeds per arm (4 degrees of freedom), so it is itself uncertain.**
  - The 95% interval for the true noise is 0.025 to 0.118 dB, so the true 2× bar could lie anywhere from 0.05 to 0.24 dB.
  - The verdict therefore rests on a small noise estimate (*inferred*).
  - The other evidence points the same way as the verdict: complete separation, the same sign in every seed, and the same gap on the training views (below).
  - Extra seeds are not pre-declared for H1, so none are planned.
- **Other noise definitions,** all against the same 2× bar:

  | Noise definition | Noise (dB) | ×noise | Comment |
  |---|---|---|---|
  | Both arms pooled (pre-registered) | 0.041 | −3.8× | — |
  | `detach`'s sd alone | 0.048 | −3.3× | — |
  | Rung 2's sd alone | 0.034 | −4.6× | — |
  | sd of the paired differences | 0.078 | −2.0× | Spread of a difference of two runs, about √2 times one run's noise by construction, so it is not comparable with the 2× bar. |

- **The checks closest to the edge:**
  - buu SSIM passes at −1.96×, by 0.0002. Had it failed, H1 would have failed on two scenes.
  - bike LPIPS is +1.35× (worse, not real).
  - sofa bm PSNR is −1.1×.
- **Pooled numbers do not rescue H1.**
  - Pooled bm PSNR is −0.02 dB (−0.3×); without shrub it is −0.07 dB (−0.8×).
  - Without shrub, LPIPS is better (−0.0095, 4.1×, real).
  - The rule is per scene by design (protocol rules 5 and 7), so the pooled match is reported as context only.

### What the bike gap is

All values are bike, `rung2` − `detach`, paired by seed (*measured*). Sources: `tables.md` R2, and each run's `diagnostics/pseudo_targets.csv` at step 10k, which covers the training views.

| Quantity | `detach` | `rung2` | Difference |
|---|---|---|---|
| Render bm PSNR against GT, **training views** | 21.34 | 21.18 | **−0.165 (3.8×)** |
| Pseudo-target bm PSNR against GT, training views | 22.504 | 22.505 | +0.002 (0.2×) |
| Render fit to its own pseudo-target (PSNR), training views | 23.58 | 23.65 | +0.07 (1.7×) |
| Pseudo-target consistency across views (× GT; lower = more consistent) | 1.72 | 1.58 | **−0.13 (10.7×)** |
| Render consistency (× GT) | 0.58 | 0.56 | **−0.02 (2.6×)** |
| Hue error on saturated pixels, validation (9% of pixels) | 31.8° | 38.5° | **+6.75° (3.3×)** |
| RGB angle error, validation | 2.48° | 2.54° | **+0.05° (3.9×)** |
| Saturation ratio, validation (1 = GT) | 0.497 | 0.525 | **+0.028 (3.7×)** |
| Dark fit (dark render against dark photo) | 47.72 dB | 47.67 dB | −0.06 (0.5×) |
| Final Gaussians | 237,520 | 231,098 | −6,422 (0.8×) |
| Curve parameters | 255,413 | 69 | — |
| Minutes per run | 8.1 | 4.8 | **−3.3** |

**The dark end** (*measured*, seed-0 curve snapshots at step 10k):
- `detach`'s bike curve maps level 0 to 83–85 of 255 in all 36 views. Rung 2 maps level 0 to 0 and level 1 to about 78.
- Only 0.8% of bike's training pixels sit at level 0, and 2.8% at level 1.

*Inferred:* bike's pseudo-targets are as close to the ground truth as `detach`'s, and its renders fit them as well, yet the renders end 0.165 dB further from the ground truth. That happens on the training views too, so the gap is not about novel views. It lies in how the pseudo-targets' errors combine in 3D, not in their average size.

One reading, untested:
- `detach`'s pseudo-target errors differ from view to view (they are less consistent), so they partly average out in the shared Gaussians.
- Rung 2's errors are more consistent across views, so they survive into the scene.
- The hue error fits this picture: a colour error made the same way in every view becomes part of the 3D model.

### Candidate causes

| Candidate | For | Against | How it is tested |
|---|---|---|---|
| Head start: the `cdf` init gives bright targets from step 0, while `detach`'s curve starts at the identity (plan risk 5) | Pre-declared risk. **Check result:** `rung2id` − `rung2` on bike = +0.26 dB (10.9×, 3 seeds) | The identity init also changes the *final* F, so "head start" really means the init changes the curve | **Done** (below): confirmed over 3 seeds; exploratory |
| Black level: level 0 maps to 0 instead of about 84 | The mappings do differ | Only 0.8% of bike's pixels sit at level 0. Shrub has 14.1% there and is not worse on bm PSNR (+0.16, 0.5×). Bike's pseudo-target bm PSNR is equal. Both inits keep F(0) = 0, so the identity init's gain comes from levels 1–4 (*measured*) | Learnable black level, an M5 fallback only |
| Colour of the shared F | Hue +6.75° (3.3×) and angle +0.05° (3.9×) on bike | Hue is *better* on buu (−0.88°, 3.7×) and chair (−0.40°, 2.1×). With the identity init, bike's hue error equals `detach`'s (33.7° against 33.7°, seed 0), so the hue gap goes with the dark end | Colour-preserving luminance variant (M5 ablation 2) |
| Consistent errors do not average out | Pseudo-targets equal on average and more consistent, fit equal, render worse | Untested and hard to isolate | — |

**Correction.** On 2026-11-14, Claude called the black level "the more likely bike lever" and listed it in the log as a possible cause. The level-0 pixel shares and the shrub comparison above make it the less likely one (*inferred*).

### The head-start check (plan 5.3)

**What ran:** `rung2` with `--rung2_F_init identity` (variant `rung2id`), on 5 LOM scenes with seed 0, on 2026-11-15. All 5 runs finished `ok` (3.7–6.5 min) and were re-scored like M3.

**The reading, fixed in the research log before any result:**
- one seed, so no "real", and each difference sits next to the M3 cells' 3-seed sd;
- the check cannot change the H1 verdict;
- "the head start explains bike" requires `rung2id` − `rung2` above 2× Rung 2's bike sd (0.067 dB) *and* closing more than half of the −0.155 dB gap (0.077 dB).

**Outputs:** [`results/rung2/summary_headstart/`](../results/rung2/summary_headstart/) (`tables.md`), from `python rung2_headstart.py`. Its `--selftest` reads `rung2id` from `rung2` seed 0 and gives exactly 0 for every difference.

Validation, seed 0 (*measured*). M3's 3-seed sd is in brackets.

| Scene | `rung2id` − `rung2`, bm PSNR [rung2 sd] | `rung2id` − `detach`, bm PSNR [detach sd] | `rung2id` − `rung2`, SSIM | `rung2id` − `rung2`, LPIPS | `rung2id` − `rung2`, hue err |
|---|---|---|---|---|---|
| buu | +0.80 [0.20] | +0.38 [0.25] | +0.0120 | −0.0213 | −0.03° |
| chair | −0.01 [0.26] | +0.63 [0.12] | +0.0100 | −0.0114 | −1.35° |
| sofa | +0.49 [0.27] | +0.18 [0.40] | +0.0039 | −0.0029 | −4.30° |
| **bike** | **+0.25 [0.03]** | **+0.14 [0.05]** | +0.0297 | −0.0012 | −3.84° |
| shrub | +0.00 [0.03] | +0.54 [0.50] | −0.0038 | +0.0016 | +0.33° |
| pooled | +0.31 [0.07] | +0.37 [0.07] | +0.0104 [0.0011] | −0.0070 [0.0024] | −1.84° [0.28] |

**By the pre-declared reading, the head start is a likely contributor to the bike gap** (*inferred*, one seed).
- Bike's +0.253 dB is 7.5× Rung 2's bike sd, and it more than closes the gap: on seed 0, `rung2id` is 0.14 dB *above* `detach`.
- The identity init is also better pooled: +0.31 dB bm PSNR, with higher SSIM, lower LPIPS and lower hue error. That includes bike's hue error, which drops to `detach`'s level (33.7° against 33.7°).

**The caveat written down beforehand applies: the init changes where F ends up, not only how fast it gets there** (*measured*, `curves.csv`, `F_trajectory.csv`).
- The identity-init F never comes within 0.01 of the `cdf`-init F in any scene.
- At 10k steps it is still 8–13 levels (of 255) away, mostly at levels 1–2. Bike's gap shrinks from 0.20 at step 1k to 0.065 at 5k and 0.051 at 10k.
- On bike, level 1 maps to 66 instead of 79 and level 2 to 85 instead of 87, while levels 3–4 sit 5–10 levels higher. Both inits keep F(0) = 0, so level 0 is not where they differ.
- The identity-init pseudo-targets are closer to the ground truth on the training views: +0.85 (buu), +0.27 (chair), +1.11 (sofa), +0.33 (bike) and −0.01 dB (shrub) bm PSNR.

**The training objective prefers the default** (*measured*, `loss_log.csv`, last 10 logged steps).
- The identity-init runs end with a slightly *higher* curve loss in all five scenes: +0.5% to +1.6%, bike +1.6%.
- Their CDF-matching term is higher in four of five scenes, and their smoothness term is lower in all five.

*Inferred:*
**Corrected on 2026-11-16.** The 2026-11-15 text here blamed the equalisation target for over-lifting the deepest shadows. The curves show otherwise (*measured*, bike and sofa, seed 0, `curves_step*.npz`, values × 255 at levels 1–4):
- **The equalisation target is near 0 there:** 1, 8, 25, 52 on bike and 0, 0, 1, 3 on sofa.
- **The shape prior x^γ is far above it:** γ is 0.075 on bike and 0.136 on sofa, giving 169–187 and 120–145.
- **Both inits start F near 0 there:** `cdf` 0, 2, 12, 25 on bike; identity 1, 2, 3, 4.
- **F rises during training toward the prior** and ends at a compromise:

  | | Level 1 | Level 2 | Level 3 | Level 4 |
  |---|---|---|---|---|
  | Bike, `cdf` init | 79 | 87 | 96 | 106 |
  | Bike, identity init | 66 | 85 | 101 | 117 |
  | Sofa, `cdf` init | 56 | 64 | 69 | 74 |
  | Sofa, identity init | 47 | 58 | 66 | 72 |

*Inferred:*
- **The lift in the deepest shadows comes from the official shape prior,** which Rung 2 kept (decision D1, 2026-11-13). It does not come from the equalisation target. `detach`'s curves lift those levels too (fig. 6, LOM panel; level 0 to about 84 on bike).
- **The two inits end at different compromises** between the prior and the target. The identity init ends less lifted at level 1, closer to the target, and the ground truth prefers that.
- **Neither curve is at the curve loss's optimum within 10k steps.** The identity init's curve is further from it.
- **The dark end does matter for bike, but at levels 1–2, not at level 0.** This revises the "black level" row below.

**What this does and does not change:**
- **H1 is still not supported.** The rule applies to the pre-registered default (`cdf` init), and the check was declared unable to change the verdict.
- **Rung 2's LOM result depends on how F is initialised,** by about +0.3 dB pooled on one seed. That belongs in the final report.
- **The identity init cannot simply become the default.**
  - The choice of `cdf` followed the plan's rule on training photos only (preview, step 6).
  - The training objective still prefers `cdf`.
  - The identity init wins only on metrics that use the normal-light images.
  - Switching now would choose a setting on the ground truth, which P5 was barred from doing.
  - Any result with the identity init is therefore exploratory (post hoc) and must be labelled so.

### Confirmation with seeds 1 and 2 (2026-11-16, exploratory)

**What ran:** Bart approved `rung2id` with seeds 1 and 2 on LOM: 10 runs, all `ok`, 21:23–22:08, re-scored like M3. The analysis is `python rung2_headstart.py --seeds 0 1 2`, written to [`summary_headstart_seeds012/`](../results/rung2/summary_headstart_seeds012/).

**The reading, fixed in the log before launch:**
- The init effect is confirmed if three things hold: the pooled `rung2id` − `rung2` in bm PSNR is real and positive; LPIPS is not worse beyond 2× noise; and the mean over seeds 1–2 is positive. Seeds 1–2 are the clean replication, since seed 0 chose the question.
- Bike is judged the same way.
- `rung2id` − `detach` also goes through the 12 H1 checks, as exploratory only.

**Testing:**
- Before use, the script's seed-0 mode reproduced the committed `summary_headstart/` byte for byte.
- Its 3-seed self-test gave 0 for every init effect. It also reproduced M3's R1b exactly.

Paired by seed over seeds 0–2, with ×noise and **bold** for real (*measured*):

| Scene | `rung2id` − `rung2`, bm PSNR | Seeds 1–2 only | SSIM | LPIPS | Hue err | `rung2id` − `detach`, bm PSNR |
|---|---|---|---|---|---|---|
| buu | **+0.63 (3.2×)** | +0.55 | **+0.0129 (4.9×)** | **−0.0249 (4.0×)** | −0.01° (0.0×) | **+0.52 (2.3×)** |
| chair | +0.01 (0.1×) | +0.02 | **+0.0100 (9.1×)** | **−0.0099 (5.5×)** | **−1.25° (7.1×)** | **+0.41 (2.4×)** |
| sofa | **+1.26 (3.7×)** | +1.64 | **+0.0089 (3.2×)** | **−0.0184 (2.6×)** | **−3.53° (7.7×)** | **+0.87 (2.2×)** |
| bike | **+0.26 (10.9×)** | +0.26 | **+0.0249 (7.9×)** | +0.0035 (0.7×) | **−4.95° (3.2×)** | **+0.10 (3.1×)** |
| shrub (reported only) | +0.01 (0.6×) | +0.02 | −0.0015 (1.2×) | +0.0034 (1.9×) | **+0.35° (7.4×)**, worse | +0.17 (0.5×) |
| **pooled** | **+0.44 (6.2×)** | +0.50 | **+0.0110 (13.4×)** | **−0.0093 (4.5×)** | **−1.88° (8.4×)** | **+0.42 (6.0×)** |
| pooled without shrub | **+0.54 (6.2×)** | +0.62 | **+0.0142 (15.4×)** | **−0.0124 (5.1×)** | **−2.44° (9.0×)** | **+0.48 (5.7×)** |

**Decisions by the pre-declared reading:**
- **The init effect is confirmed** pooled, pooled without shrub, and on bike. The three seeds agree with seed 0.
- **Exploratory H1 check:** `rung2id` − `detach` **meets all 12 H1 checks.**
  - Closest are bike LPIPS (+0.0111, 1.7×, worse) and buu SSIM (+0.0058, 1.6×, better).
  - Bike's bm PSNR goes from −0.155 dB (−3.8×) with the default to **+0.103 dB (3.1×)**.
  - Raw PSNR, outside the rule: buu −1.05 dB (2.3×, worse), bike +0.19 dB (2.5×, better), pooled +0.16 dB (1.9×).

**Also measured:**
- **F does not depend on the seed.** For both inits, the final F is the same in all three seeds to 0.1 of a level (`curves.csv`). So the init effect is a fixed shift of the curve, not seed noise.
  - The identity-init F stays 8–13 levels from the `cdf` F in every seed. It never comes within 0.01.
  - Because F is seed-independent, the pseudo-targets barely vary across seeds, and the ×noise values of the training-view pseudo-target rows are not informative.
- **The curve loss is higher with the identity init in all 15 pairs,** by +0.3% to +1.6%.
- **Training views:** pseudo-target bm PSNR is higher by +0.27 to +1.11 dB in four scenes (shrub −0.01). Render bm PSNR is higher by +0.12 to +1.13 dB.

*Inferred:*
- **The bike failure belongs to the default curve fit, not to the model.** On bike, the init effect (+0.26 dB) is larger than the H1 gap (−0.155 dB). With the identity init, the same model (one shared F plus one exposure per view, 55–76 parameters) meets every H1 check. What fails H1 is how the default F ends up in the deepest shadows (pulled up by the shape prior), not the shared-curve-plus-exposure model.
- **The gain is an implicit regularisation, not a principled setting.** It comes from F *not* reaching the curve loss's optimum within 10k steps, so it is tied to the init, the learning rate and the step count. Training longer would probably erode it: bike's F gap is still shrinking (0.065 at 5k, 0.051 at 10k).
- **A principled version would change the objective, chosen on training photos.** The natural lever is the shape prior: its weight, or the planned `--rung2_prior none` ablation in M5.
- **Not tested:** whether the identity init changes H2 on the varying set.

**For the final report, the two results side by side:**
- *Pre-registered:* H1 not supported. With the default `cdf` init, Rung 2 is 0.155 dB below `detach` on bike.
- *Exploratory, after a pre-declared check:* with F initialised at the identity, Rung 2 meets all 12 H1 checks and beats `detach` by +0.42 dB bm PSNR pooled (6.0×). The curve loss prefers the default, so the gain comes from a deep-shadow end that the shape prior has pulled up less far.

### What this means (*inferred*)

- **By the rule:** H1 is not supported. Rung 2 does not match `detach` within noise in every decision scene. On bike it is 0.155 dB worse in bm PSNR, a small deficit that is consistent across seeds.
- **In size:** the deficit is small next to what both detached models gain over the official code on bike (1.86 dB). It comes with about 3,700× fewer curve parameters on bike (69 vs 255,413) and 40% less training time.
- **The bike deficit is not a fixed cost of the simple model.** The head-start check, confirmed over three seeds, shows that F's initialisation moves Rung 2's result by more than the H1 gap. With the identity init, Rung 2 meets all 12 H1 checks: bike +0.10 dB, pooled +0.42 dB against `detach`. That setting was chosen after seeing results, so it is exploratory.
- **Wording for the final report:** "Rung 2 matches `detach` pooled and in three of the four decision scenes. On bike it is 0.15 dB worse in brightness-matched PSNR, which fails the pre-registered H1 rule. A pre-declared follow-up shows the result depends on how the curve is initialised. With F started at the identity (3 seeds, exploratory), Rung 2 meets every H1 check and is 0.42 dB better than `detach` pooled. The training loss prefers the default, so the gain comes from deepest shadows that the shape prior has lifted less." The hypothesis is not reworded after the fact.

## Q2. Dark branch: how much of the varying-exposure cost is the dark branch?

The dark-photo fit is the PSNR of the dark render against the dark photo, as its loss sees it, mean over training views (*measured*, R5b and R5b′):

| Cell | Dark fit, varying set | LOM level | Column effect (exp − shared) |
|---|---|---|---|
| `detach` | 38.75 ± 0.45 | 46.24 | — |
| `detachexp` | 44.52 ± 0.05 | 46.24 | **+5.77 (17.9×)** |
| `rung2` | 37.24 ± 0.15 | 46.14 | — |
| `rung2exp` | 44.87 ± 0.15 | 46.14 | **+7.63 (50.6×)** |

*Measured:*
- **The switch removes most of the misfit.** The gain is real in every scene for Rung 2 (+3.9 to +10.3 dB) and in four of five for `detach`; on buu it is +1.23 dB (1.2×).
  - The per-view slopes flatten: Rung 2's paired change for brightened views goes from −14.5 to −5.0 dB per stop.
  - Pooled, 1.3 dB (Rung 2) and 1.7 dB (`detach`) stay below LOM.
- **The image metrics move little.**
  - Dark-branch main effect: bm PSNR +0.28 dB (0.7×), interaction +0.05 dB (0.1×).
  - Within the Rung 2 row it is real: +0.31 dB bm (2.7×), +0.35 raw (3.7×), +0.0083 SSIM (3.5×).
  - Bike in that row goes the other way: −0.08 dB (2.5×).
- **The exp cells grow fewer Gaussians:** −36k for `detach` (6.6×) and −26k for Rung 2 (8.5×), pooled.

*Inferred:* the shared dark branch costs dark-photo fit far more than it costs image quality. So the H2 effect below is not a dark-branch artefact, which matches the absence of an interaction.

## Q3. H2: does Rung 2 beat `detach` on the varying set?

**Yes.** The pre-registered rule is met, pooled and pooled without shrub.

Cells, pooled (*measured*, R3):

| Cell | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| Official | 19.31 ± 0.33 | 16.35 ± 0.59 | 0.7685 | 0.3517 |
| `detach` | 21.03 ± 0.72 | 16.87 ± 1.15 | 0.7748 | 0.3064 |
| `detachexp` | 21.28 ± 0.31 | 17.28 ± 0.25 | 0.7923 | 0.2880 |
| `rung2` | **23.11 ± 0.11** | 18.25 ± 0.10 | 0.7949 | 0.2756 |
| `rung2exp` | **23.42 ± 0.12** | 18.60 ± 0.09 | 0.8031 | 0.2703 |

Curve-model main effect, ½[(rung2 − detach) + (rung2exp − detachexp)], with noise over the four cells (*measured*, R4):

| Scene | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| buu | +1.84 (1.6×) | **−2.19 (4.1×)** | −0.0047 (0.6×) | **−0.0418 (2.6×)** |
| chair | **+3.59 (7.6×)** | **+4.39 (5.9×)** | **+0.0303 (2.4×)** | −0.0481 (1.8×) |
| sofa | **+2.95 (3.5×)** | +0.70 (0.5×) | +0.0077 (1.0×) | **−0.0455 (2.5×)** |
| bike | **+0.56 (2.7×)** | **+1.57 (2.5×)** | +0.0009 (0.1×) | **−0.0402 (4.4×)** |
| shrub (reported only) | **+1.64 (67.7×)** | **+2.27 (21.6×)** | **+0.0431 (41.5×)** | **+0.0543 (46.8×)**, worse |
| **pooled** | **+2.11 (5.3×)** | **+1.35 (2.3×)** | **+0.0155 (2.3×)** | **−0.0242 (2.5×)** |
| pooled without shrub | **+2.23 (4.5×)** | +1.12 (1.5×) | +0.0085 (1.1×) | **−0.0439 (3.6×)** |

*Measured:*
- **Both columns:** shared +2.09 (4.0×), exp +2.14 (9.1×). H2 holds without the dark-branch fix.
- **Buu's raw and bm PSNR disagree:** raw falls by 2.19 dB while bm rises by 1.84 dB. This is the brightness offset M1 described, and bm is the decision metric.
- **Bike, where Rung 2 lost on LOM, it wins here:** +0.81 dB in the shared column (3.2×).

*Inferred:*
- **Rung 2 is nearly insensitive to the exposure spread.** Its pooled bm PSNR is 23.23 on LOM and 23.11 on the varying set, while `detach` drops from 23.25 to 21.03 (unpaired cell means).
- **It is also far more repeatable across seeds:** sd 0.11 against `detach`'s 0.72.

Figure: [`fig2_2x2_interaction.png`](../results/rung2/summary/figures/fig2_2x2_interaction.png).

## Q4. Mechanism: does Rung 2 compensate exposure?

**Yes, on both pre-registered diagnostics** (*measured*, R5a and R5c).

**Pseudo-target leak** is the slope of log2(pseudo-target brightness, varying ÷ LOM) on the applied gain: 0 means fully compensated and 1 means the gain passes straight through.

| | Official | `detach` | `rung2` |
|---|---|---|---|
| Pseudo-targets, pooled | 0.405 | 0.512 | **0.012** |
| Pseudo-targets, per scene | 0.27–0.56 | 0.36–0.65 | −0.003 to 0.029 |
| Enhanced renders at training views, pooled | 0.186 | 0.222 | **0.004** |

The rung2 − `detach` difference is −0.500 (720×). The mechanism criterion (near 0, and below `detach`'s by more than 2× noise) is met.

**Exposures:**
- Rung 2's final s_k follow the applied gains: slope −1.00 pooled, from −0.91 on bike to −1.04 on buu.
- Paired RMSE is 0.018 stops. For comparison:
  - the initialisation: 0.011;
  - the histogram oracle: 0.052;
  - a post-hoc fit of `detach`'s curves: 0.726.
- Training moves s_k only 0.018 stops from the initialisation. As planned, this is a sanity check and not a test, because the initialisation already sits at the quantisation floor.

Figures: [`fig3_leak.png`](../results/rung2/summary/figures/fig3_leak.png), [`fig5_exposure.png`](../results/rung2/summary/figures/fig5_exposure.png).

## Q5. Side effects

*Measured* (R2 for LOM, R6 for the varying set):

- **Colour, LOM, pooled rung2 − detach:**
  - RGB angle error −0.48° (2.7×, better);
  - hue error +1.23° (4.7×, worse), carried by bike's +6.75°. Hue is better on buu and chair.
- **Colour, varying set:**
  - angle error 4.66° against 5.81°;
  - saturation ratio 0.765 against 0.590, closer to 1;
  - hue error 16.20° against 10.85°, carried by bike (35.37° against 12.34°) and sofa (12.54° against 8.63°).
- **Cross-view consistency, varying set:** pseudo-targets 1.69 against 5.06 × GT, renders 0.73 against 1.71.
- **Cross-view consistency, LOM:**
  - pseudo-targets about equal (+0.04, 0.7×);
  - renders slightly *less* consistent (+0.06, 2.4×), mostly sofa (+0.23, 2.9×).
- **Dark end:** LOM dark fit −0.10 dB pooled (2.9×) and −0.16 dB on shrub (3.1×).
  - Shrub has 14.1% of its pixels at level 0. Its bm PSNR is not worse (+0.16), but SSIM is (−0.0076, 2.4×, real) and LPIPS rises (+0.050, 1.3×).
  - On the varying set, shrub's LPIPS is worse by +0.054 (46.8×).
- **Cost:** 4.6 against 7.8 minutes per run on LOM, and 4.7 against 8.1 on the varying set. Final Gaussian counts are similar (LOM +4,188, 1.2×).

*Inferred:*
- **Shrub** is the scene where the black level could matter (many level-0 pixels, worse SSIM and LPIPS).
- **Bike** is the scene where Rung 2's colour differs most.

## What this means for M5

*Inferred:*
- **H3 (`rung2att`, attached s_k):** unchanged as the first ablation. It runs on LOM and on the varying set.
- **F's dark end (new, from the head-start check):** the clearest lead.
  - The identity init gave a less-lifted curve at levels 1–2 and +0.44 dB pooled (6.2×, 3 seeds).
  - It is not yet tested on the varying set (H2).
  - The lift there comes from the shape prior (corrected 2026-11-16), so the first lever is the already-planned no-prior ablation (`--rung2_prior none`) or a lower prior weight. After that come the smoothness weight λ (the step 6 fallback) and K and knot spacing.
  - A principled version would choose them on training photos alone. Any F-init ablation is labelled as suggested by validation results.
- **Colour-preserving luminance variant (ablation 2):** still relevant to bike's hue. Note that the identity init already brought bike's hue error down to `detach`'s level, so part of the hue difference is the dark end too.
- **Learnable black level:** stays a fallback. Both inits keep F(0) = 0, and the identity init's gain comes from levels 1–4, so level 0 looks like the wrong target for bike. Shrub's SSIM and LPIPS remain the better test case.
- **H1 is not re-tested by any of these.** An ablation that closes the bike gap is a finding about that variant. It does not overturn the pre-registered verdict for the default.

## Setup and caveats

- **Runs:** commit `c304662`, 10k steps, seeds 0–2.
  - The official and `detach` runs come from Rung 0 (LOM) and M1 (varying set); `detachexp` comes from M2 part A.
  - Analysis: `rung2_analyze.py` (fingerprint `cfcde97621bd4381`), committed with the results.
- **Integrity checks** (`tables.md`, "Integrity"):
  - re-score against training eval within 2.5e-6 dB;
  - `rung2` and `rung2exp` bit-identical in s, F and pseudo-targets in all 15 pairs;
  - `fit_exposure_form` recovers each run's s_k within 0.007 stops;
  - M1's tables reproduced exactly.
- **Self-test:** `rung2_analyze.py --selftest` replaces the Rung 2 cells with copies of `detach`, and every curve-model effect comes out exactly 0.
- **GPU nondeterminism:** same-seed runs can differ (plan risk 7), so single runs are never compared alone.
- **Wall times:** seed 0's re-scoring overlapped the last LOM runs, so `sofa_rung2_seed2` and `bike_rung2_seed2` may read 0.2–0.3 minutes high.
