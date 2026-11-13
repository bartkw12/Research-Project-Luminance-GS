# Experiment protocol

How experiments in this project are set up, scored and compared, so results from different rungs can sit in one table. Read this before planning, running or analysing an experiment. Numbers come from `Luminance-GS/Implementation_guides/Rung0_results.md` unless stated.

## Data

- **LOM** (from Aleth-NeRF), `Luminance-GS/data/LOM_full/<scene>/`, five scenes: **buu, chair, sofa, bike, shrub**. Training reads `low/` (dark photos); validation reads `high/` (normal-light photos of held-out views) and is used only for scoring.
- `--exp_name low` for low light (the main setting), `over_exp` for the overexposure subset (needed for H1 later).
  - *Note (2026-11-11):* whether overexposure is part of H1 is pending Wei (13 Nov check-in). It would need official and `detach` baselines on `over_exp` too. Until then, M2 and M3 test H1 on low light only.
- The COLMAP cameras are not pinhole, so images are undistorted and cropped to 374×499. Training-view ground truth used for oracle scoring gets the same undistortion.
- **Varying-exposure LOM** (to build for Rung 2, tests H2): for each training view k, draw u_k uniformly from [−1, 1] and use `clip(2^{u_k} · I_k, 0, 1)` re-quantised to 8 bits. Keep the test views and `high/` unchanged. Save every u_k so learned exposures can be checked: the ideal is s_k ≈ −u_k + constant. Compare the learned s_k with its initialisation log2(m̄ / m_k), which already tracks u_k through each view's mean brightness. A histogram-only s_k follows u_k almost by construction, so the real test is beating the initialisation.
  - **As built (M1, 2026-11-08):**
    - Location: `Luminance-GS/data/LOM_vexp/<scene>/`, built and verified by `examples/build_vexp.py` (git-ignored, rebuildable).
    - Gain model: `clip(round(2^{u_k} · I_k), 0, 255)` on the 8-bit values.
    - Sampling is **stratified**, not independent: N evenly spaced values −1 + (2j+1)/N in random order, so the mean of u_k is exactly 0 and the sd is 0.58 to 0.59. `--sampling iid` gives independent draws.
    - Dataset seed 701, one stream per scene; all training seeds see the same set.
    - Files are JPEG quality 100 with 4:4:4 chroma, under the original names.
    - The record of the set is `Luminance-GS/results/vexp/dataset/gains.json` (u_k and the realised `log2_ratio` per view); `oracle.json` holds the initialisation and histogram-oracle exposures.
    - **Quantisation caveat:** the dark views have means of 7 to 29 levels, so the realised ratio differs from 2^{u_k} by up to 0.06 stops (bike). Darkened views also lose tonal levels that brightened views do not gain, so part of any varying-exposure cost is unrecoverable and not symmetric in u.
  - **Correction (M1):** "beating the initialisation" cannot be judged on this set. The init already gives slope −0.95 to −1.04 and r −0.95 to −0.99. Its paired error (s_k on the varying set minus s_k on LOM, same view) is only 0.004 to 0.019 stops of 8-bit rounding, and its unpaired error is LOM's natural spread, whose split into exposure and scene content is unknown. Report s_k against u_k as a sanity check, and let the PSNR comparison against `detach` and the pseudo-target leak (Metrics) carry H2. Results: `Luminance-GS/Implementation_guides/M1_vexp_results.md`.
- The paper's MipNeRF360-varying set (`--exp_name variance --data_factor 8`) is an optional second test.

## Runs

- **10,000 steps**, seeds **0, 1 and 2** for every configuration. The official code hard-coded seed 42, and its runs differ by up to about 1 dB from GPU nondeterminism alone.
- **Naming:** `Luminance-GS/results/<rung root>/<scene>_<variant>_seed<seed>/`. Rung 0 and 0.5 share `results/rung0/` so one analysis can read every cell. Start a new root for Rung 2 (for example `results/rung2/`).
- **Varying-exposure runs** live in `results/vexp/` (M1: `baseline` and `detach`, seeds 0 to 2).
  - Run: `python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp ...`. The runner refuses a non-default `--data_root` with the default `results/rung0` root.
  - Re-score with the **LOM** reference depth, so consistency matches Rung 0.5: `python rescore_runs.py --root ../results/vexp --ref_root ../results/rung0`.
  - Analysis: `python vexp_analyze.py` writes `results/vexp/summary/`.
- **M2 and M3** (planned 2026-11-11; details in `Luminance-GS/Implementation_guides/M2_rung2_plan.md`, section 2.3):
  - Rung 2 runs on LOM go to a new root, `results/rung2/`. Varying-set runs join M1's in `results/vexp/`, so all four cells of the 2×2 share one root.
  - The 2×2 on the varying set: curve (`detach` or Rung 2) × dark branch (shared or exposure-aware), i.e. `detach`, `detachexp`, `rung2`, `rung2exp`. "exp" in a variant name means the exposure-aware dark branch.
  - Re-score every root other than `results/rung0` with `--ref_root ../results/rung0`. Only `rung0` holds the `<scene>_baseline_seed0` reference runs; without the flag, `rescore_runs.py` skips every run in `results/rung2/`.
- **Variants** are entries in `VARIANTS` in `examples/run_rung0.py` (name → extra flags). `baseline` (no flags) is the official code and stays in every table, labelled "official".
- **Batch runner:** `python run_rung0.py --scenes ... --variants ... --seeds 0 1 2 [--root ...] [--dry_run]`. It is resumable: a run counts as done when `stats/val_step9999.json` exists, and an incomplete run folder is deleted and restarted. Each run logs to `<root>/logs/<run>.log` and adds a row to `<root>/runs.csv` (finish time, ok, return code, minutes).
- **Smoke test first:** about 300 steps per new flag set, for example `--max_steps 300 --eval_steps 300 --save_steps 300 --diag_steps 1 300 --diagnostics`, into a `smoke*` folder (git-ignored).
- **Cost:** one run takes about 8 to 14 minutes (mean minutes per 10k-step run: `baseline` 12.6, `frozenbasis` 13.1, `detachcurve` 11.3, `detach` 7.9; source `Rung0.5_results.md` Q7). One configuration on 5 scenes × 3 seeds takes about 3 hours. Before any batch over an hour, the PC must not sleep: `powercfg /change standby-timeout-ac 0` in a Windows terminal, restored afterwards.

## Per-run outputs

| Path in the run folder | Contents | Commit? |
|---|---|---|
| `cfg.json` | full config, including seed and flags | yes |
| `stats/val_step9999.json` | validation `psnr`, `ssim`, `lpips`, `num_GS` | yes |
| `stats/train_*.json`, other `stats/val_*.json` | intermediate stats | yes |
| `diagnostics/*.csv`, `diagnostics/timing.json` | loss log, pseudo-target scores, consistency | yes |
| `rescore/*.csv`, `rescore/*.json` | Rung 0.5 re-scoring (val metrics, colour metrics, shared-depth consistency) | yes |
| `exposure_init.json` (and a copy in `rescore/`) | M2: per training view, the mean input and s_measured = log2(m̄/m_k). Written whenever `--dark_branch` is not `shared` | yes |
| `ckpts/`, `*.pt`, `*.npz`, `renders/`, `renders_depth/`, `videos/`, `tb/`, per-view PNGs | bulky outputs | **no** (git-ignored) |

Summary folders (`results/rung0/summary/`, `summary_rung05/`) and the hand-written reports in `Luminance-GS/Implementation_guides/` are committed in full. Once a rung's report is written, its summary folder is not edited again; a re-analysis goes into a new folder.

## Metrics

| Metric | Definition | Source |
|---|---|---|
| PSNR, SSIM, LPIPS (raw) | enhanced render of each validation view vs its `high/` image | `stats/val_step9999.json` |
| Brightness-matched PSNR (`psnr_gtmean`) | scale the prediction by mean(GT) / mean(prediction) over all pixels and channels, clamp to [0, 1], then PSNR (`diagnostics.py:psnr_gt_mean`) | Rung 0: computed by `rung0_analyze.py` from the local (git-ignored) renders; from Rung 0.5 on: `rescore/val_summary.json` |
| `ang_err_deg` (and `_p95`) | mean angle between predicted and GT RGB vectors, pixels with \|g\| > 0.05·√3; brightness-invariant | `rescore/` |
| `hue_err_sat_deg` | mean circular HSV hue difference on pixels with GT saturation > 0.3 and value > 0.1 (the "red wheel" measure). Unreliable on desaturated predictions (Rung 0.5 Q5) | `rescore/` |
| `sat_ratio` | mean predicted saturation / mean GT saturation; below 1 means washed out | `rescore/` |
| `chroma_err_ab` (planned for Rung 2) | mean error in CIELAB (a\*, b\*) after brightness matching, reported next to `sat_ratio`; meant to stay stable on pale images, where the hue metric does not | to be added to `rescore/` |
| Pseudo-target quality (oracle) | pseudo-target vs the training view's normal-light image; never used in training | `diagnostics/pseudo_targets.csv`, `rescore/` |
| Cross-view consistency | warp each view to its 2 nearest training cameras with rendered depth (alpha ≥ 0.5, 5% depth tolerance), relative L1 = L1 / mean intensity, for pseudo-targets, renders and GT. Use **one shared reference depth per scene** (`<scene>_baseline_seed0`), not each run's own depth. A diagnostic, not a quality proxy: `detach` has the least consistent pseudo-targets and the best renders | `rescore/consistency.csv` |
| Exposure recovery (varying-exposure set) | per scene, slope and Pearson r of learned s_k against true u_k, next to the same for the initialisation log2(m̄ / m_k). **Slope and r saturate on this set** (the init alone reaches r −0.95 to −0.99), so always add the residual RMSE in stops of s_k + u_k after removing the mean, unpaired and paired (minus the same view's s_k on LOM), next to the init and the histogram oracle. For models without an explicit s_k, fit one shared F plus one s_k per view to the final curves (`build_vexp.fit_exposure_form`) | `vexp_analyze.py` T4a |
| Pseudo-target leak (varying-exposure set) | slope of log2(mean pseudo-target, varying / LOM, same view and seed) against u_k: 0 = exposure fully compensated, 1 = passes straight through. Same for the enhanced render at the training view, and for the dark input as a check (≈ 1) | `vexp_analyze.py` T4b |
| Dark-branch fit | `fit_psnr_lowrender_vs_input`: shared dark render vs the dark input at the training views. On the varying set, report the paired difference against signed u_k, split by sign (not against \|u_k\|) | `rescore/pseudo_targets.csv`, `vexp_analyze.py` T5 |
| Dark-branch fit as trained (M2) | `fit_psnr_lowrender_exp_vs_input`: the dark render as its loss sees it, i.e. ×2^(−s_k) with an exposure-aware `--dark_branch`, and equal to the column above with `shared`. `s_dark` is the s_k used (0 for `shared`). Compare cells with this column | `rescore/pseudo_targets.csv` (last two columns) |
| Curve parameters | count of learnable curve parameters (baseline about 255k + 5N; Rung 2 about 32 + N) | by hand |
| Wall time | minutes per run from `runs.csv`; exclude `sofa_detach_seed1` (PC slept 11 h) | `runs.csv` |

Always report raw and brightness-matched PSNR side by side. The unsupervised method sets global brightness through the histogram prior (pseudo-targets average 0.54 to 0.62 in every scene, while the ground truth ranges 0.36 to 0.54), so raw PSNR mostly measures that offset on bike and shrub. Rung 2 does not change this: its exposures are relative (mean 0), and global brightness still comes from the CDF fit.

## Noise floor and how to compare

Seed std of validation PSNR (baseline, seeds 0 to 2):

| Scene | Raw PSNR std | Brightness-matched std | Smallest detectable raw effect | Smallest detectable brightness-matched effect |
|---|---|---|---|---|
| buu | 0.21 | 0.34 | 0.3 dB | about 0.2 to 0.7 dB |
| chair | 0.37 | 0.16 | 0.6 dB | about 0.2 to 0.7 dB |
| sofa | 0.58 | 0.46 | 0.9 dB | about 0.2 to 0.7 dB |
| bike | 0.45 | 0.13 | 0.7 dB | about 0.2 to 0.7 dB |
| shrub | 1.13 | 0.98 | 1.8 dB | about 1.6 dB |

"Detectable" means the difference exceeds 2 standard errors with 3 seeds per arm, about 1.6× the per-run std.

*Note (2026-11-11):* H1 uses the M2 plan's rule (section 6.5: buu, chair, sofa and bike; brightness-matched PSNR not below −2× noise, SSIM and LPIPS not worse by more than 2× noise, noise as in comparison rule 2). Bart chose it over the explainer's earlier wording (every scene, within the detectable effect above). Overexposure in H1 is pending Wei.

## Reference numbers for Rung 2

Validation means over seeds 0 to 2 (source `Luminance-GS/results/rung0/summary_rung05/tables.md`, tables 1 and 4). Rung 2 is compared against `detach`; `baseline` is the official code.

| Scene | `detach`: raw / brightness-matched PSNR | SSIM | LPIPS | Angle error (°) | `sat_ratio` | Pseudo-target consistency (× GT) | `baseline` (official): raw / brightness-matched / SSIM / LPIPS |
|---|---|---|---|---|---|---|---|
| buu | 21.56 / 26.37 | 0.904 | 0.203 | 4.03 | 0.93 | 1.56 | 20.81 / 26.00 / 0.909 / 0.186 |
| chair | 18.82 / 23.09 | 0.853 | 0.311 | 5.87 | 0.50 | 0.97 | 17.76 / 20.94 / 0.839 / 0.338 |
| sofa | 23.47 / 27.41 | 0.904 | 0.260 | 2.57 | 0.98 | 1.67 | 23.22 / 25.23 / 0.899 / 0.276 |
| bike | 16.02 / 21.31 | 0.764 | 0.291 | 2.48 | 0.50 | 1.72 | 18.42 / 19.45 / 0.752 / 0.409 |
| shrub | 11.76 / 18.06 | 0.597 | 0.288 | 10.77 | 0.63 | 1.19 | 12.13 / 18.37 / 0.616 / 0.263 |
| pooled | 18.33 / 23.25 | 0.804 | 0.270 | 5.15 | 0.71 | | 18.47 / 22.00 / 0.803 / 0.294 |

**Varying-exposure set (H2 bar).** Means over seeds 0 to 2, source `Luminance-GS/results/vexp/summary/tables.md` (T1, T6, T7) and `M1_vexp_results.md`.
- Seed noise is larger here than on LOM: `detach` bm PSNR sd is 0.03 to 1.51 dB (buu 1.51, sofa 1.11), and its raw PSNR sd reaches 2.59 dB on sofa.
- Compute H2's detectable effect from these runs, not from the LOM noise-floor table above.

| Scene | `detach`: raw / brightness-matched PSNR | SSIM | LPIPS | Angle error (°) | `sat_ratio` | Pseudo-target consistency (× GT) | `baseline` (official): raw / brightness-matched / SSIM / LPIPS |
|---|---|---|---|---|---|---|---|
| buu | 22.81 / 24.26 | 0.881 | 0.251 | 3.64 | 0.94 | 4.73 | 22.01 / 22.33 / 0.877 / 0.261 |
| chair | 15.49 / 19.89 | 0.815 | 0.356 | 7.12 | 0.33 | 5.41 | 14.76 / 17.82 / 0.797 / 0.397 |
| sofa | 21.67 / 23.84 | 0.884 | 0.300 | 2.52 | 0.84 | 8.98 | 18.77 / 20.04 / 0.853 / 0.375 |
| bike | 14.55 / 20.42 | 0.745 | 0.329 | 2.54 | 0.44 | 4.61 | 15.48 / 19.24 / 0.742 / 0.403 |
| shrub | 9.83 / 16.71 | 0.550 | 0.296 | 13.27 | 0.40 | 1.60 | 10.74 / 17.13 / 0.573 / 0.323 |
| pooled | 16.87 / 21.03 | 0.775 | 0.306 | 5.82 | 0.59 | | 16.35 / 19.31 / 0.768 / 0.352 |

Comparison rules:

1. Pair runs by seed (variant A seed s vs variant B seed s), and report the mean ± sample std of the paired differences.
2. Express each effect in units of the pooled per-run seed std ("×noise"). Call it real only above about 2×.
3. Report per scene first, then pooled over scenes. Note which scenes carry an effect.
4. For factorial designs (Rung 0.5), report main effects and the interaction, as in `rung05_analyze.py`, and lead with them. Quote a factor's share of a combined gain only as a range over both orders: in Rung 0.5 the basis share ranged from −29% to 42%.
5. Keep the official configuration in every table as "official", next to the `detach` cell, which Rung 0.5 chose as the reference baseline for Rung 2.
6. A negative or null result is still a result. Log it with the same care.
7. Shrub is uninformative so far. Its seed noise is the largest (0.75 to 1 dB brightness-matched), and no factor has had a real effect on it. Report it, say so, and do not let it drive a conclusion.
8. **M2 and M3** (pre-registered in `M2_rung2_plan.md` section 6.5, 2026-11-11; to confirm with Wei on 13 Nov): H2 is read from the 2×2 on the varying set, with the factorial effects of rule 4.
   - **Supported:** the pooled curve-model main effect on brightness-matched PSNR is real and positive, LPIPS is not worse beyond noise, and the effect is positive in most informative scenes.
   - **Supported only with the dark-branch fix:** only `rung2exp` − `detachexp` is real.
   - **Pooled effect between 1× and 2× noise:** run seeds 3 and 4 before concluding.
   - **Mechanism, which supports H2 but does not replace it:** Rung 2's pooled pseudo-target leak is near 0 and below `detach`'s by more than 2× noise.
   - **Report `num_GS` per cell:** the exposure-aware dark branch also re-weights each view's `L_low` gradient, which can change densification.

## Analysis and reporting

- Each rung has an analysis script in `examples/` (`rung0_analyze.py`, `rung05_analyze.py`) that reads run folders and writes CSVs, `tables.md`, `numbers.json` and figures to a summary folder. New scripts import helpers from the old ones instead of copying them.
- Figures: labelled axes with units, one idea per figure, and the same colour for the same variant across all figures.
- The narrative report (`Rung*_results.md` in `Luminance-GS/Implementation_guides/`) is written by hand from the summary outputs. It opens with direct answers to the plan's questions, labels each claim *measured* or *inferred*, and ends with implications for the next rung.
- Every finished experiment also gets an entry in `docs/RESEARCH_LOG.md`.
