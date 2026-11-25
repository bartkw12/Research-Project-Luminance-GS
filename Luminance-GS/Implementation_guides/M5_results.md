# M5 results: H3, the identity init (H1b, H2b), no prior, long replay and overexposure

Written 2026-11-25 by Claude Code for Bart (and as source material for the final report and the next update to Wei). Plan: `M5_plan.md`. Readings pre-registered in `docs/RESEARCH_LOG.md` on 2026-11-24, before any run. Every number below is *measured* unless marked *inferred*. Noise = sqrt(mean of the arms' seed variances), paired by seed; an effect is "real" above 2× noise (*). Pooled = mean over the five scenes per seed; shrub never decides. bm PSNR = brightness-matched PSNR.

## Verdicts at a glance

| ID | Question | Pre-registered verdict | Headline (pooled) | Source |
|---|---|---|---|---|
| E1 · H2b(a) | Does the identity init's gain carry over to the varying-exposure photos? | **Confirmed** (pooled, pooled without shrub, bike) | init effect +0.61 dB bm PSNR (+5.1×), LPIPS −0.0117 (better, −3.1×); bike +0.28 dB (+9.7×) | `results/rung2/summary_h2b/` |
| E1 · H2b(b) | Does H2 hold for the identity variant? | **Supported** | curve-model effect +2.72 dB (+6.8×), LPIPS −0.0360; positive in 4/4 decision scenes | `results/rung2/summary_rung2id/` |
| E1 · H2b(c) | Mechanism unchanged? | **Yes** (no surprise) | leak 0.009 vs `detach` 0.512; s_k slope −1.00 | same |
| E2 · H3 (restated) | Does sending the image loss into s_k change quality? | **No effect** | LOM +0.08 dB (+1.5×); varying −0.01 dB (−0.1×); no harm signal | `results/rung2/summary_h3/` |
| E3 · H1b | Does `rung2id` meet the H1 rule against `detach` on fresh seeds 3–5? | **Supported, 12 of 12** | +0.37 dB (+3.7×); bike +0.14 dB (+2.6×) | `results/h1b/summary/` |
| A1 | Is the identity gain early stopping? | **Early stopping** (all 5 scenes) | F gap 0.051 at 10k → 0.001 at 50k (bike) | `results/rung2/replay_long/` |
| E4 | What does dropping the shape prior cost? | reported only | −6.47 dB bm PSNR (−130×); the dark end collapses to ~0 | `results/rung2/summary_noprior/` |
| E6 · H1-over | Does Rung 2 match `detach` on LOM overexposure? | **Not supported, 3 of 12** | −1.26 dB (−17.6×); buu −5.35 dB; and `detach` itself is 0.94 dB below official | `results/overexp/summary/` |

Scope: 165 training runs (commit `dbdcfdb`), all `ok`, all re-scored against `results/rung0/<scene>_baseline_seed0`. Analysis scripts `rung2_analyze.py` (with `--rung2_variant rung2id`) and the new `rung2_ablation.py`, both tested before use (log 2026-11-25: byte-for-byte reproduction of M3, selftests at exactly 0, exact reproduction of M3's R1b and the head-start H1 table).

---

## E1 · H2b: the identity init on the varying-exposure set

**Runs:** `rung2id` and `rung2idexp`, `results/vexp/`, seeds 0–2 (30 runs), with M3's `rung2`, `rung2exp`, `detach`, `detachexp` and official cells. `rung2id` and `rung2idexp` agree bit for bit in F and s (integrity check, max |Δ| = 0), as `rung2` and `rung2exp` did.

**(a) The init effect carries over: confirmed.** Init main effect ½[(`rung2id` − `rung2`) + (`rung2idexp` − `rung2exp`)], noise over the four cells:

| | bm PSNR | raw PSNR | SSIM | LPIPS | Reading |
|---|---|---|---|---|---|
| buu | +0.66 (+2.2×) * | +0.32 (+0.6×) | +0.0119 (+2.4×) * | −0.0146 (−1.2×) | |
| chair | +0.42 (+1.5×) | +0.35 (+1.5×) | +0.0136 (+7.9×) * | −0.0218 (−6.0×) * | |
| sofa | +1.67 (+3.8×) * | +0.90 (+4.2×) * | +0.0123 (+7.4×) * | −0.0274 (−2.9×) * | |
| bike | +0.28 (+9.7×) * | −0.11 (−1.8×) | +0.0262 (+5.4×) * | +0.0018 (+0.3×) | **confirmed** |
| shrub | −0.01 (−0.6×) | +0.05 (+1.7×) | −0.0012 (−1.0×) | +0.0033 (+2.3×) * | |
| pooled | +0.61 (+5.1×) * | +0.30 (+3.3×) * | +0.0126 (+7.3×) * | −0.0117 (−3.1×) * | **confirmed** |
| pooled without shrub | +0.76 (+5.2×) * | +0.36 (+3.3×) * | +0.0160 (+7.5×) * | −0.0155 (−3.2×) * | **confirmed** |

The init gain on the varying set (+0.61 dB) is larger than on LOM (+0.44 dB, M3 head-start check), and it is positive in every decision scene. Shrub, as on LOM, does not move.

**(b) H2 holds for the identity variant: supported.** The plan 6.5 H2 rule with `rung2id`/`rung2idexp` in place of `rung2`/`rung2exp`:

| Pooled cell (bm PSNR, mean ± sd) | official | `detach` | `detachexp` | `rung2id` | `rung2idexp` |
|---|---|---|---|---|---|
| bm PSNR | 19.31 ± 0.33 | 21.03 ± 0.72 | 21.28 ± 0.31 | 23.71 ± 0.14 | 24.04 ± 0.10 |
| LPIPS | 0.3517 ± 0.0073 | 0.3064 ± 0.0167 | 0.2880 ± 0.0069 | 0.2623 ± 0.0024 | 0.2602 ± 0.0021 |

- Curve-model main effect: **+2.72 dB (+6.8×)** pooled (M3's `cdf` default: +2.11 dB, +5.3×), +2.99 dB (+6.0×) without shrub. LPIPS −0.0360 (−3.9×, better). Positive in buu (+2.50), chair (+4.01), sofa (+4.62) and bike (+0.84).
- Exp-column simple effect +2.75 dB (+12.0×).

**(c) Mechanism: unchanged, as expected.** The init does not touch s_k.
- Pooled paired leak 0.009 against `detach`'s 0.512 (M3 `rung2`: 0.012).
- s_k slope on the gains −1.00 (r −0.98), paired RMSE 0.011 stops, s moves 0.005 stops from its initialisation.
- No leak above 0.1, so no surprise to flag.

**Bike's hue on the varying set (for the E5 decision):** `rung2id` 27.0° ± 5.6, M3's `rung2` 35.4° ± 0.7, `detach` 12.3° ± 1.0. The identity init removes about a third of the gap to `detach`, not all of it, and its seed spread is large.

**Label:** pre-registered on 2026-11-24. The photos are new (gained) inputs, but the scenes and their ground truth are the ones the init was chosen on.

## E2 · H3 (restated): attaching s_k to the image loss

**Runs:** `rung2att` (`--rung2_attach s`, F stays detached), LOM (`results/rung2/`) and the varying set (`results/vexp/`, shared column), seeds 0–2 (30 runs), against `rung2`.

**Verdict: no effect.** No help signal and no harm signal.

| `rung2att` − `rung2` | bm PSNR | raw PSNR | SSIM | LPIPS | bike hue |
|---|---|---|---|---|---|
| LOM pooled | +0.08 (+1.5×) | +0.08 (+0.6×) | +0.0007 (+0.7×) | +0.0004 (+0.2×) | +0.23° (+0.1×) |
| LOM pooled without shrub | +0.09 (+1.4×) | +0.10 (+0.6×) | +0.0005 (+0.4×) | +0.0003 (+0.1×) | |
| varying pooled | −0.01 (−0.1×) | +0.17 (+1.9×) | +0.0004 (+0.1×) | +0.0015 (+0.2×) | +0.82° (+0.6×) |
| varying pooled without shrub | −0.01 (−0.1×) | +0.21 (+2.1×) * | +0.0006 (+0.1×) | +0.0022 (+0.3×) | |

- **Harm checks:** LOM decision scenes' bm PSNR buu +0.06, chair +0.06, sofa +0.24, bike +0.00 (none below −2× noise). Bike hue not worse beyond 2× on either set. Varying pooled leak 0.006 (limit 0.1). Pooled s_k slope −1.003 (range −1.1 to −0.9).
- **What the attachment does do (measured):** the exposures move more. s drift from the initialisation is 0.044 stops on the varying set (`rung2`: 0.018) and 0.049 on LOM (`rung2`: 0.012). The paired s RMSE against the gains rises slightly (0.022 vs 0.018, +2.2×), and the pooled leak halves (0.006 vs 0.012). On LOM, pseudo-target bm PSNR is 0.06 dB lower (−39×) and the validation angle error 0.06° higher (+7.8×): real but tiny, and neither is a pre-registered signal.
- *Inferred:* 3D feedback into the exposures moves them by a few hundredths of a stop without changing quality. The exposures are already where the 3D model wants them, so the detached default loses nothing by not using the renders.

## E3 · H1b: the identity variant on fresh seeds

**Runs:** `detach` and `rung2id`, LOM, seeds 3–5 (30 runs), `results/h1b/`.

**Verdict: supported, 12 of 12 checks meet.**

| `rung2id` − `detach`, seeds 3–5 | bm PSNR | SSIM | LPIPS |
|---|---|---|---|
| buu | +0.62 (+4.7×) * | +0.0046 (+6.7×) * | −0.0380 (−14.6×) * |
| chair | +0.48 (+1.8×) | +0.0154 (+7.9×) * | −0.0221 (−6.9×) * |
| sofa | +0.68 (+1.7×) | +0.0076 (+1.4×) | −0.0144 (−1.2×) |
| bike | +0.14 (+2.6×) * | +0.0270 (+5.3×) * | +0.0062 (+0.9×) |
| pooled | +0.37 (+3.7×) * | +0.0101 (+8.6×) * | −0.0084 (−1.1×) |
| pooled without shrub | +0.48 (+3.0×) * | +0.0136 (+5.7×) * | −0.0171 (−6.0×) * |

- The only check on the worse side at all is bike's LPIPS (+0.9×, allowance 2×).
- **Context only, seeds 0–5** (`results/h1b/summary_seeds012345/`; mixes `detach` from October, Rung 0, with today's; M2's regression check makes that acceptable): 12 of 12, pooled +0.39 dB (+4.3×), bike +0.12 dB (+2.9×).
- **What it can and cannot show:** F does not depend on the seed, so seeds 3–5 re-test only the 3DGS randomness. H1b turns the exploratory 12/12 of seeds 0–2 into a pre-registered verdict. It cannot rule out that the init was chosen to suit these five scenes; E1 is the stronger test of that, and it confirmed the gain on new photos.

## A1 · long curve replay: early stopping

Curve-only replay (no 3DGS) on the LOM training photos, both inits, 50k steps with the learning rate decayed to 1% over the replay. Built-in check: the 10k replay's gap matches the trainer's in every scene (bike 0.0511 vs 0.0510).

| Scene | gap at 1k | gap at 10k | gap at 40k | gap at 50k | gap halves by | Outcome |
|---|---|---|---|---|---|---|
| buu | 0.078 | 0.034 | 0.002 | 0.002 | 4k | early stopping |
| chair | 0.126 | 0.031 | 0.000 | 0.000 | 3k | early stopping |
| sofa | 0.095 | 0.035 | 0.001 | 0.001 | 4k | early stopping |
| bike | 0.177 | 0.051 | 0.001 | 0.001 | 4k | early stopping |
| shrub | 0.106 | 0.031 | 0.002 | 0.002 | 2k | early stopping |

*Inferred:* at 10k steps, the identity-init F is still on its way to the same F the `cdf` init reaches; the curve loss does have one minimum. So the identity variant's gain (E1, E3) comes from stopping F before it fully lifts the dark end. The 3DGS consequence of a 50k-step run is not measured. This matches M3's finding that the curve loss prefers `cdf` in all 15 pairs, and it means the better validation scores come from a curve the training objective has not yet reached: the objective's optimum is not the best curve for the ground truth.

## E4 · no shape prior

`rung2noprior` (`--rung2_prior none`, K + N parameters: one fewer than `rung2`) − `rung2`, LOM, seeds 0–2. Reported only.

| | bm PSNR | raw PSNR | SSIM | LPIPS | bike hue | pseudo-target bm PSNR |
|---|---|---|---|---|---|---|
| buu | −9.49 (−64×) * | −3.80 * | −0.1325 * | +0.0619 * | | −10.77 * |
| chair | −6.91 (−37×) * | −3.05 * | −0.1136 * | +0.0285 * | | −7.07 * |
| sofa | −10.88 (−56×) * | −6.25 * | −0.1522 * | +0.0468 * | | −12.63 * |
| bike | −6.66 (−127×) * | −1.21 * | −0.1967 * | +0.0266 * | −27.3° (−17.8×) * | −8.42 * |
| shrub | +1.59 (+44×) * | +3.04 * | +0.0720 * | −0.1310 * | | +1.31 * |
| pooled | −6.47 (−130×) * | −2.26 * | −0.1046 * | +0.0065 * | | −7.52 * |

**F at grid levels 1–4 (×255), mean over seeds:**

| Scene | `rung2noprior` | `rung2` |
|---|---|---|
| buu | 0.04, 0.12, 1.22, 2.79 | 47.1, 54.0, 60.5, 66.9 |
| chair | 0.02, 0.41, 11.4, 25.5 | 64.6, 72.4, 83.6, 95.8 |
| sofa | 0.04, 0.07, 0.44, 0.96 | 56.2, 63.7, 69.3, 74.2 |
| bike | 0.08, 1.53, 11.0, 23.6 | 78.9, 87.0, 96.4, 106.4 |
| shrub | 6.2, 22.7, 44.6, 68.2 | 81.8, 96.0, 112.2, 128.8 |

- The cost matches the expectation from `preview_final` (pseudo-targets 7.0 to 12.8 dB short of `detach`'s; shrub +1.5): in training, the pseudo-targets fall 7.1 to 12.6 dB short of `rung2`'s, and shrub's gain 1.3 dB.
- Without the prior, F does not lift the deep shadows at all: the histogram-equalisation target alone leaves levels 1–4 near 0. This confirms the 2026-11-16 correction (the deep-shadow lift comes from the one-γ prior, not from the equalisation target) with a direct measurement.
- Bike's hue error falls by 27° without the prior. *Inferred:* bike's hue problem is tied to how strongly the prior lifts the deep shadows (which amplifies colour noise there), consistent with M3's finding that the hue gap "goes with the dark end". The fix it suggests is not "drop the prior": every decision scene loses 7 to 11 dB.

## E6 · H1-over: overexposure

**Runs:** official, `detach` and `rung2` with `--exp_name over_exp`, seeds 0–2 (45 runs), plus optional `rung2id` (15 runs), `results/overexp/`. Every run's `exp_name` checked. Pre-launch preview (2026-11-24) passed its stop rule.

**Verdict: not supported, 3 of 12.** Bike meets; buu, chair and sofa fail on all three metrics.

| `rung2` − `detach` (over_exp) | bm PSNR | SSIM | LPIPS | check |
|---|---|---|---|---|
| buu | −5.35 (−28.1×) * | −0.0584 (−30.3×) * | +0.0218 (+8.3×) * | fails |
| chair | −0.40 (−3.1×) * | −0.0044 (−4.4×) * | +0.0075 (+5.4×) * | fails |
| sofa | −0.35 (−3.3×) * | −0.0134 (−9.4×) * | +0.0106 (+3.9×) * | fails |
| bike | −0.14 (−0.9×) | −0.0011 (−0.4×) | +0.0050 (+0.7×) | meets |
| shrub (reported) | −0.08 (−1.1×) | −0.0007 (−0.4×) | +0.0010 (+1.2×) | |
| pooled | −1.26 (−17.6×) * | −0.0156 (−34.6×) * | +0.0092 (+6.1×) * | |

**The bar itself is weaker here.** `detach` − official on overexposure is −0.94 dB pooled (−9.7×): official is best on chair (21.24 dB), sofa (20.33) and bike (24.84), `detach` on buu (23.55 against 21.60). The detach gain is a low-light result. Pooled bm PSNR: official 21.69, `detach` 20.75, `rung2` 19.48, `rung2id` 19.55.

- **Exploratory:** `rung2id` − `rung2` +0.07 dB (+2.0×, not real); `rung2id` − `detach` −1.20 dB (−16.0×). The init does not rescue overexposure.
- **Where it fails (measured):** on buu, `rung2`'s pseudo-targets are 5.65 dB below `detach`'s, its raw PSNR is higher (+1.68 dB), and its saturation ratio is +0.36 above `detach`'s. The preview had found buu's F ≈ x^12.8, mapping level 127 to about 0.5/255.
- *Inferred:* Rung 2's shape prior is one power curve. On low light it lifts (γ about 0.08–0.16); on overexposure it must darken, and a strong power curve applied per RGB channel both crushes mid-tones and raises saturation (channel ratios are raised to the power γ). The official per-view networks can bend the curve in ways one γ cannot. This is the low-light-specific default the plan's risk 6 anticipated; per the plan, it is reported as such and nothing is tuned on the overexposure validation results.

---

## What the final report says

- **H1 (low light), pre-registered on the `cdf` init: not supported** (11 of 12; bike −0.155 dB at −3.8× noise; pooled −0.02 dB).
- **H1b (identity init, pre-registered 2026-11-24 after H1, fresh seeds 3–5): supported, 12 of 12** (pooled +0.37 dB, +3.7×). Report side by side with H1, never in its place, with the label: the init was chosen after seeing seed-0 results; E1 shows the gain carries over to new (gained) photos; A1 shows it is early stopping of F, not a better optimum of the curve loss.
- **H1-over (overexposure, pre-registered): not supported, 3 of 12**, and the `detach` bar is itself 0.94 dB below the official code there. Rung 2's one-γ shape prior is a low-light default.
- **H2 supported** (M3, `cdf` init: +2.11 dB, 5.3×), **and it holds for the identity variant** (+2.72 dB, 6.8×). The explicit exposure model, not the init, carries H2: leak 0.009–0.012 against `detach`'s 0.512.
- **H3 (restated 2026-11-24): no effect.** Attaching s_k moves the exposures by a few hundredths of a stop and changes no metric beyond noise. The detached default (2D enhancement with a shared curve plus exposure alignment) loses nothing measurable by not using 3D feedback in s_k.
- **The shape prior is essential in low light** (E4: −6.5 dB pooled without it) and **wrong in overexposure** (E6). One sentence for the report: the one global γ that carries the official shape prior is what lifts the deep shadows; it is also what fails when the photos need darkening.
- **Parameters:** `rung2` has K + N + 1 curve parameters (55 to 76 per scene) against the official code's about 255k.

## Open after M5 (for Bart's decisions)

1. **Which Rung 2 version Rung 3 (M6) builds on.** M5 supports: detached s_k (attaching does nothing); the `cdf` init stays the pre-registered default; the identity init is a pre-registered-on-fresh-seeds alternative whose gain is early stopping. Bart's call.
2. **E5 (colour-preserving luminance variant).** Bike's hue gap on the varying set remains (27° against `detach`'s 12°), so E5 stays an M6 candidate (plan section 8). E4 suggests the hue gap is tied to the dark-end lift.
3. **Overexposure.** A fix would need a different shape prior for darkening (not tuned on validation). Out of scope before the freeze unless Bart decides otherwise; report it as a limitation.
4. **For Wei** (plan section 7): H1b and H2b next to the side-by-side H1 story; the H3 restatement and its "no effect" result; the overexposure verdict and the weaker `detach` bar there; the no-prior cost; A1.
