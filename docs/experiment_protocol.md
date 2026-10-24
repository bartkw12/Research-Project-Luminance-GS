# Experiment protocol

How experiments in this project are set up, scored and compared, so results from different rungs can sit in one table. Read this before planning, running or analysing an experiment. Numbers come from `Luminance-GS/Implementation_guides/Rung0_results.md` unless stated.

## Data

- **LOM** (from Aleth-NeRF), `Luminance-GS/data/LOM_full/<scene>/`, five scenes: **buu, chair, sofa, bike, shrub**. Training reads `low/` (dark photos); validation reads `high/` (normal-light photos of held-out views) and is used only for scoring.
- `--exp_name low` for low light (the main setting), `over_exp` for the overexposure subset (needed for H1 later).
- The COLMAP cameras are not pinhole, so images are undistorted and cropped to 374×499. Training-view ground truth used for oracle scoring gets the same undistortion.
- **Varying-exposure LOM** (to build for Rung 2, tests H2): for each training view k, draw u_k uniformly from [−1, 1] and use `clip(2^{u_k} · I_k, 0, 1)` re-quantised to 8 bits. Keep the test views and `high/` unchanged. Save every u_k so learned exposures can be checked: the ideal is s_k ≈ −u_k + constant. Compare the learned s_k with its initialisation log2(m̄ / m_k), which already tracks u_k through each view's mean brightness. A histogram-only s_k follows u_k almost by construction, so the real test is beating the initialisation.
- The paper's MipNeRF360-varying set (`--exp_name variance --data_factor 8`) is an optional second test.

## Runs

- **10,000 steps**, seeds **0, 1 and 2** for every configuration. The official code hard-coded seed 42, and its runs differ by up to about 1 dB from GPU nondeterminism alone.
- **Naming:** `Luminance-GS/results/<rung root>/<scene>_<variant>_seed<seed>/`. Rung 0 and 0.5 share `results/rung0/` so one analysis can read every cell. Start a new root for Rung 2 (for example `results/rung2/`).
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
| Exposure recovery (varying-exposure set) | per scene, slope and Pearson r of learned s_k against true u_k, next to the same for the initialisation log2(m̄ / m_k) | Rung 2 analysis script |
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

Comparison rules:

1. Pair runs by seed (variant A seed s vs variant B seed s), and report the mean ± sample std of the paired differences.
2. Express each effect in units of the pooled per-run seed std ("×noise"). Call it real only above about 2×.
3. Report per scene first, then pooled over scenes. Note which scenes carry an effect.
4. For factorial designs (Rung 0.5), report main effects and the interaction, as in `rung05_analyze.py`, and lead with them. Quote a factor's share of a combined gain only as a range over both orders: in Rung 0.5 the basis share ranged from −29% to 42%.
5. Keep the official configuration in every table as "official", next to the `detach` cell, which Rung 0.5 chose as the reference baseline for Rung 2.
6. A negative or null result is still a result. Log it with the same care.
7. Shrub is uninformative so far. Its seed noise is the largest (0.75 to 1 dB brightness-matched), and no factor has had a real effect on it. Report it, say so, and do not let it drive a conclusion.

## Analysis and reporting

- Each rung has an analysis script in `examples/` (`rung0_analyze.py`, `rung05_analyze.py`) that reads run folders and writes CSVs, `tables.md`, `numbers.json` and figures to a summary folder. New scripts import helpers from the old ones instead of copying them.
- Figures: labelled axes with units, one idea per figure, and the same colour for the same variant across all figures.
- The narrative report (`Rung*_results.md` in `Luminance-GS/Implementation_guides/`) is written by hand from the summary outputs. It opens with direct answers to the plan's questions, labels each claim *measured* or *inferred*, and ends with implications for the next rung.
- Every finished experiment also gets an entry in `docs/RESEARCH_LOG.md`.
