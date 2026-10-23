# Research log

An append-only record of what was done in this project, why, and what came out of it. It is the source for the final ECE 701 report, so each entry should still make sense months later without the chat history or the run folders.

**Rules**

- Add new entries at the bottom. Never rewrite or delete an old entry. If one turns out to be wrong, add a new entry that corrects it and names the date of the old one.
- Write one entry per experiment, batch, code change that affects runs, or decision.
- Numbers come from files, and the entry names the file. Mark interpretations as *inferred*. Unknown values are written "pending", never guessed.
- Record enough to reproduce: branch, commit (and whether there were uncommitted changes), command, seeds, steps, and any environment change.
- The `/log-experiment` skill writes entries in this format.

## Entry template

```markdown
### YYYY-MM-DD: <short title>

- **Rung / type:** Rung N · experiment | code | decision | setup
- **Why:** the question or motivation, in one or two sentences.
- **What was done:** the change, runs or analysis.
- **Reproduce:** branch `...`, commit `abc1234` (clean or with uncommitted changes), command(s), scenes, seeds, steps.
- **Results:** key numbers as mean ± std over seeds, each with its source path. Figures by path.
- **Takeaway:** what it means. Mark interpretations *inferred*.
- **Next:** the follow-up or decision this leads to.
```

---

## Entries

Entries up to 2026-10-23 were backfilled from git history, the run folders and the project discussions when this log was created.

### 2026-09-16: Cloned the official Luminance-GS code

- **Rung / type:** setup
- **Why:** the project extends Luminance-GS, so the authors' implementation is the baseline.
- **What was done:** cloned `https://github.com/cuiziteng/Luminance-GS` and set up the environment on Bart's Windows PC.
- **Reproduce:** conda env `luminancegs`, gsplat 1.0.0 (`pip install git+https://github.com/nerfstudio-project/gsplat@v1.0.0`), numpy < 2, one NVIDIA GPU. PyTorch and CUDA versions: pending (record them with `python -c "import torch; print(torch.__version__, torch.version.cuda)"`).
- **Results:** none.
- **Next:** reproduce the paper's LOM low-light numbers.

### 2026-10-17: Reproduced the LOM low-light baseline (buu, chair, sofa)

- **Rung / type:** baseline · experiment
- **Why:** check that the official code runs on this machine and measure run-to-run variation before changing anything.
- **What was done:** 3 runs per scene with the unmodified config, 10k steps, final evaluation at step 9999.
- **Reproduce:** branch `main`, commits `970b1b1` and `5652299`. `python simple_trainer_ours.py --data_dir ../data/LOM_full/<scene> --exp_name low --result_dir ../results/<scene>_low[_runN]`. The seed was hard-coded to 42 in every run.
- **Results** (mean ± sample std, n = 3; source `Luminance-GS/results/reproduction/average_results.md`):

  | Scene | PSNR | SSIM | LPIPS |
  |---|---|---|---|
  | buu | 21.30 ± 0.38 | 0.9116 ± 0.0010 | 0.1809 ± 0.0016 |
  | chair | 17.48 ± 0.05 | 0.8377 ± 0.0014 | 0.3317 ± 0.0047 |
  | sofa | 23.41 ± 0.54 | 0.9061 ± 0.0022 | 0.2569 ± 0.0104 |

  The paper reference for buu (as noted in the README) is 18.09 / 0.877 / 0.193.
- **Takeaway:** the code reproduces and runs on one GPU in about 10 to 14 minutes. With an identical seed, runs still differ by up to about 1 dB, so GPU nondeterminism alone is a large noise source (measured).
- **Next:** choose a research direction.

### 2026-10-17: Chose research direction 1 (a new low-light to normal-light transform)

- **Rung / type:** decision
- **Why:** Wei proposed two directions: (1) a new way to turn low-light renders into normal-light ones, such as an alternative curve estimation, and (2) wider real-world evaluation, especially camera-pose errors in low light. Both do not fit in one semester.
- **What was done:** picked direction 1, because it matches Bart's interest in AI and computer vision, and because NOVA-GS and Naka-GS (2026) already work on poses from dark images. ChatGPT's idea (a compact monotonic curve) was judged a good first experiment but too thin for the whole project. Proposed a ladder: Rung 1 compact monotone curve, Rung 2 one shared curve plus one exposure per view, Rung 3 local monotone curves (stretch). Reading the code also showed the curve is never forced to increase and the seed was hard-coded to 42.
- **Reproduce:** gameplan doc "Luminance-GS Extension: Research Direction and Gameplan" (claude.ai artifact).
- **Results:** none.
- **Next:** measure what the official curve actually learns before building anything (Rung 0).

### 2026-10-20 to 2026-10-21: Rung 0, diagnostics of the official curve

- **Rung / type:** Rung 0 · code and experiment
- **Why:** find out whether the learned curves are non-monotone, how much they differ across views, and what the priors and the pseudo-target gradient actually do, so the method targets a real weakness.
- **What was done:** added the flags `--seed`, `--detach_pseudo_target`, `--diagnostics` and `--diag_steps`, the shared helper `Runner.curve_forward`, `examples/diagnostics.py`, `examples/run_rung0.py` and `examples/rung0_analyze.py`. A run with no new flags reproduces the original code (the helper is bit-identical; training order unchanged with diagnostics on). Then ran 30 runs: 5 LOM scenes × {baseline, detach} × seeds 0, 1, 2, 10k steps.
- **Reproduce:** branch `rung0-diagnostics`, commits `c56ea4f` to `f078d6c`. `python run_rung0.py`, then `python rung0_analyze.py`. Plan: `Luminance-GS/Implementation_guides/Rung0_diagnostics_plan.md`.
- **Results** (full report `Luminance-GS/Implementation_guides/Rung0_results.md`, tables and figures in `Luminance-GS/results/rung0/summary/`):
  - The curves are already monotone, except shrub baseline (about 10 decreasing steps per view, at most 5/255).
  - Cross-view variation is one-dimensional: PC1 explains 99.2 to 99.8%, and its score correlates with each view's mean brightness at |r| = 0.97 to 1.00. The bias network's output is large but almost identical across views.
  - A monotone piecewise-linear curve with K = 32 knots over the used input range fits to 0.70/255. Pixels use only 14 to 25% of the 255 bins.
  - Only gamma in the shape prior moves (to 0.07 to 0.20); alpha and beta stay at their initial values. The gamma prior's 1e-3 input clamp maps black to 0.30 to 0.64.
  - The histogram-prior terms are about 5× all image terms combined at the end of training.
  - Detach minus baseline, brightness-matched PSNR: buu +0.35, chair +2.16, sofa +2.16, bike +1.86, shrub −0.31 dB (5 to 19× seed noise on chair, sofa and bike). Bike LPIPS −0.118. Detached runs are about 40% faster.
  - Seed std of raw validation PSNR (baseline): buu 0.21, chair 0.37, sofa 0.58, bike 0.45, shrub 1.13 dB.
  - Baseline pseudo-targets are 0.69 to 1.30× as inconsistent across views as the real ground-truth photos.
- **Takeaway:** monotonicity alone would change almost nothing. One exposure number per view captures what the per-view network does (*inferred*). The PC slept for about 11 hours during `sofa_detach_seed1`; its results are unaffected but its wall time is excluded.
- **Next:** revise the plan (next entry).

### 2026-10-21: Revised the plan after Rung 0

- **Rung / type:** decision
- **Why:** Rung 0 changed which rungs are worth doing.
- **What was done:**
  - Dropped Rung 1 as a standalone result. Its compact, anchored curve (K = 32, knots dense near black, F(0) = 0) becomes the parameterisation of Rung 2's shared curve.
  - Made Rung 2 the core, and added a varying-exposure version of LOM (a random gain 2^u per training view, u in [−1, 1], ground truth unchanged), because on evenly lit LOM Rung 2 can at best tie.
  - Made Rung 3 optional, since the pseudo-targets are already about as consistent as real photos.
  - Found that the detach result is confounded. The per-view colour basis (`axis1_para`, `axis2_para`) gets gradient only through the pseudo-target, so `--detach_pseudo_target` also froze it at RGB in all 15 detach runs. The bike baseline turns one yellow wheel red; the detached run does not.
  - Noted two measurement fixes: the gamma prior's 1e-3 clamp lifts the black level (*inferred* cause of the grey look on bike and shrub), and the consistency metric should use one shared depth per scene.
- **Reproduce:** "Rung 0 results and revised plan" section of the gameplan doc.
- **Results:** none new.
- **Next:** Rung 0.5, a 2×2 experiment that separates the two effects of detaching.

### 2026-10-23: Wrote the Rung 0.5 plan and the project explainer

- **Rung / type:** setup
- **Why:** give Claude Code a self-contained Rung 0.5 plan, and give Bart one document explaining the paper, the code and the direction for meetings with Wei and Prof. Chen.
- **What was done:** `Rung0.5_diagnostics_plan.md` (merged into `rung0-diagnostics` as PR #2) and `Luminance_GS_Research_Project_Explained.md`. The explainer lists two more code-vs-paper differences: the colour roles are reversed, and the histogram weight drops to 0.5 after step 3000, not 0.1.
- **Reproduce:** `Luminance-GS/Implementation_guides/Rung0.5_diagnostics_plan.md`, `docs/Luminance_GS_Research_Project_Explained.md`.
- **Results:** none.
- **Next:** run Rung 0.5.

### 2026-10-23: Reorganised the repository layout

- **Rung / type:** setup
- **Why:** keep the repository root clean.
- **What was done:** moved the rung plans and `Rung0_results.md` into `Luminance-GS/Implementation_guides/`, and the reproduction artefacts (`average_results.md`, the console log, the smoke-test screenshots) into `Luminance-GS/results/reproduction/`. Fixed the links broken by the move.
- **Reproduce:** branch `rung05-diagnostics`, first two commits after branching from `rung0-diagnostics` (hashes pending: the branch was not yet pushed when this entry was written).
- **Results:** none.
- **Next:** none.

### 2026-10-23: Rung 0.5, untangling the detach result (in progress)

- **Rung / type:** Rung 0.5 · code and experiment
- **Why:** decide whether Rung 2 keeps a per-view colour basis and whether its curve trains attached or detached, and test H3. The 2×2 crosses colour basis (learned or frozen at RGB) with curve (trained by the enhancement loss or not); the existing `baseline` and `detach` runs fill two of the four cells.
- **What was done:**
  - Flags `--freeze_color_basis` (new cell `frozenbasis`), `--detach_curve_in_target` (new cell `detachcurve`) and `--grad_check`.
  - A re-scoring mode (`--rescore_run`, `--ref_depth_run`) that scores finished runs from their checkpoints with one shared reference depth per scene (`<scene>_baseline_seed0`), brightness-invariant colour metrics (`ang_err_deg`, `hue_err_sat_deg`, `sat_ratio`) and a zero-basis counterfactual.
  - `run_rung0.py` variants `frozenbasis` and `detachcurve`, `rescore_runs.py` and `rung05_analyze.py`.
  - Started the 30-run batch: 5 scenes × {frozenbasis, detachcurve} × seeds 0, 1, 2, 10k steps.
- **Reproduce:** branch `rung05-diagnostics` (hashes pending). `python run_rung0.py --variants frozenbasis detachcurve`, then `python rescore_runs.py`, then `python rung05_analyze.py`. Plan: `Luminance-GS/Implementation_guides/Rung0.5_diagnostics_plan.md`.
- **Results:** Phase 0 (confound check) and grad-check results: pending, from the Claude Code session report. Re-scoring check (measured): re-scoring `buu_baseline_seed0` with its own depth reproduced its validation PSNR, 21.0166 dB in both `stats/val_step9999.json` and `results/rung0/smoke05_rescore_self/val_summary.json` (local, git-ignored). Batch progress (measured from `results/rung0/runs.csv`): started 14:41 EDT, 15 of 30 runs finished by 17:27 EDT; mean minutes per run so far are 11.6 for `frozenbasis` (n = 8) and 10.3 for `detachcurve` (n = 7). Validation metrics, factorial effects and colour metrics: pending.
- **Takeaway:** pending.
- **Next:** finish the batch and re-scoring, write `Luminance-GS/Implementation_guides/Rung0.5_results.md`, and add a results entry here.

### 2026-10-23: Set up Claude Code for the project

- **Rung / type:** setup
- **Why:** every Claude Code session should start with the goal, rules and rung status; the log should be kept reliably for the report; and Bart wants to make all commits himself.
- **What was done:** added `CLAUDE.md` (goal, rung status, repo map, how to run, rules, docs index), this log, the reference docs `docs/curve_estimation.md`, `docs/repo_vs_paper.md` and `docs/experiment_protocol.md`, the explainer under `docs/`, the skills `/log-experiment` and `/run-batch`, a Stop hook that reminds Claude to log changes, `ask` rules so `git commit` and `git push` always need Bart's approval, and `.gitignore` rules for bulky outputs of future runs.
- **Reproduce:** PR "Add Claude Code project setup" on GitHub.
- **Results:** none.
- **Next:** log the Rung 0.5 results with `/log-experiment` when the batch finishes.
