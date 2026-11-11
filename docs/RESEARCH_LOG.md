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

### 2026-10-24: Fixed the Claude Code setup

- **Rung / type:** setup
- **Why:** a review of the 2026-10-23 setup (PR #4) found that it was not active and was partly out of date.
- **What was done:**
  - Found that VS Code had been opened on the parent folder, not the repository. Claude Code reads `.claude/settings.json` only from the folder it starts in, so the Stop hook and the git `ask` rules were never loaded, and `CLAUDE.md` loaded only on demand. From now on, open the repository folder itself.
  - `CLAUDE.md`: Rung 0.5 marked done, H3 marked not supported, the Rung 2 reference baseline (`detach`) recorded, and the shell and Python notes corrected (sessions use PowerShell; plain `python` has no torch).
  - Added a git-ignored `CLAUDE.local.md` with the env interpreter path.
  - `.claude/settings.json`:
    - `ask` rules for `git pull`, `restore`, `clean`, `stash` and `branch -D`;
    - `ask` rules for Claude's edits to the finished Rung 0 and 0.5 summaries and reports;
    - the Stop hook switched to exec form, so it no longer depends on which shell runs it.
  - Skills: commands that work in PowerShell; the helper script is now located through `${CLAUDE_SKILL_DIR}`.
  - `docs/experiment_protocol.md`: final Rung 0.5 run times and the chosen reference cell.
  - Deleted the duplicate `Luminance_GS_Research_Project_Explained.md` at the repository root (broken links; the copy in `docs/` is the canonical one).
- **Reproduce:** branch `rung0-diagnostics`, on top of commit `1081d0c`.
- **Results:** none.
- **Next:** reopen VS Code on the repository folder, check that the hook and the rules load, and log the Rung 0.5 results with `/log-experiment`.

### 2026-10-24: Rung 0.5 results: the detach gain comes from the curve, not the colour basis

- **Rung / type:** Rung 0.5 · experiment
- **Why:** finish the 2×2 started in the 2026-10-23 entry "Rung 0.5, untangling the detach result (in progress)". The aim is to separate the two things `--detach_pseudo_target` does (the enhancement loss stops training the curve, and the colour basis stays frozen at RGB), test H3, and pick the reference baseline for Rung 2.
- **What was done:** finished the 30-run batch (`results/rung0/runs.csv` now has 15 runs in each of the four cells). Re-scored all 60 runs from their checkpoints with one shared reference depth per scene, the colour metrics and the zero-basis counterfactual. Wrote `Luminance-GS/Implementation_guides/Rung0.5_results.md`. Checks left pending in the 2026-10-23 entry: the confound is real (every basis parameter is exactly 0 in all 15 detach runs), and the gradient check matched the plan in all four cells.
- **Reproduce:** branch `rung05-diagnostics`, commits `4d1263b` to `ce69255` (runs and summaries in `559815a`), merged into `rung0-diagnostics`. Logged at `1081d0c` with `examples/` clean. From `examples/`: `python run_rung0.py --variants frozenbasis detachcurve` (5 scenes, seeds 0 1 2, 10k steps), then `python rescore_runs.py`, then `python rung05_analyze.py`. `frozenbasis` = `--freeze_color_basis`, `detachcurve` = `--detach_curve_in_target`; `detach` = `--detach_pseudo_target` (the Rung 0 runs).
- **Results** (validation, mean over seeds 0 to 2; each cell is raw PSNR / brightness-matched PSNR / SSIM / LPIPS; source `Luminance-GS/results/rung0/summary_rung05/tables.md`, table 1):

  | Scene | baseline (official) | frozenbasis | detachcurve | detach |
  |---|---|---|---|---|
  | buu | 20.81 / 26.00 / .909 / .186 | 22.36 / 27.11 / .917 / .166 | 20.63 / 26.08 / .903 / .200 | 21.56 / 26.37 / .904 / .203 |
  | chair | 17.76 / 20.94 / .839 / .338 | 18.65 / 21.49 / .845 / .322 | 18.34 / 22.78 / .849 / .313 | 18.82 / 23.09 / .853 / .311 |
  | sofa | 23.22 / 25.23 / .899 / .276 | 23.92 / 25.91 / .907 / .252 | 23.85 / 28.04 / .910 / .243 | 23.47 / 27.41 / .904 / .260 |
  | bike | 18.42 / 19.45 / .752 / .409 | 18.56 / 19.02 / .743 / .409 | 16.02 / 21.31 / .766 / .292 | 16.02 / 21.31 / .764 / .291 |
  | shrub | 12.13 / 18.37 / .616 / .263 | 13.52 / 19.05 / .639 / .247 | 10.71 / 17.42 / .568 / .317 | 11.76 / 18.06 / .597 / .288 |
  | pooled | 18.47 / 22.00 / .803 / .294 | 19.40 / 22.52 / .810 / .279 | 17.91 / 23.13 / .799 / .273 | 18.33 / 23.25 / .804 / .270 |

  - Factorial effects, pooled, paired by seed (table 2a). Brightness-matched PSNR: curve +0.93 dB (5.1× noise), basis +0.32 dB (1.8×, not real), interaction −0.40 dB (2.2×), total detach gain +1.26 dB (6.8×). Raw PSNR: curve −0.82 dB (3.1×), basis +0.68 dB (2.6×).
  - Share of the brightness-matched detach gain explained by detaching the curve first (table 2b): chair 85%, sofa 129%, bike 100%, pooled 90%. Freezing the basis explains 10% (applied last) to 42% (applied first) pooled, and none on bike.
  - Exceptions: on buu, detaching the curve worsens LPIPS by 0.026 (4.5×). Shrub shows no real effect of either factor (seed noise 0.75 dB).
  - Bike hue flip (view 29, seed 0): the wheel is still red with `frozenbasis` and yellow in both detached cells. Pseudo-target hue error by cell: 20.7°, 16.0°, 10.8°, 10.7° (table 3). On validation renders the hue metric ranks the cells the other way, because the detached renders are desaturated (bike saturation ratio 0.50 vs 0.77); the metric is unreliable there.
  - The learned basis rotates only 0.02° to 0.49° on average (table 5), yet resetting it to RGB improves the baseline pseudo-targets in every scene, by +0.16 to +0.93 dB brightness-matched PSNR.
  - Pseudo-target inconsistency with a shared depth, relative L1 over the ground-truth level (table 4): higher with a detached curve in 4 of 5 scenes (baseline → detach: sofa 1.16 → 1.67, bike 1.11 → 1.72); shrub is the exception.
  - Minutes per run (table 6): baseline 12.6, frozenbasis 13.1, detachcurve 11.3, detach 7.9. The re-score reproduces every run's validation PSNR to below 0.001 dB.
- **Takeaway:** H3 is not supported. Freezing the colour basis neither explains most of the detach gain nor removes the bike hue flip. Detaching the curve explains 85 to 100% of the gain wherever the gain is real (measured). *Inferred:* the hue flip comes from the curve chasing the render. Once the curve is detached, a fixed RGB basis costs nothing real (detach − detachcurve ranges from −0.63 dB on sofa, 1.6× noise, to +0.65 dB on shrub). Detaching has costs: worse LPIPS on buu, paler renders, lower raw PSNR, and less view-consistent pseudo-targets.
- **Next:** Rung 2 trains a detached curve with a fixed RGB basis. Its reference baseline is the `detach` cell, with `baseline` kept in every table as "official" and the full 2×2 reported as an ablation. Track saturation and pseudo-target consistency in Rung 2. Write the Rung 2 plan and build the varying-exposure LOM set.

### 2026-10-24: Rung 2 direction after Rung 0.5; H3 replaced

- **Rung / type:** Rung 2 · decision
- **Why:** turn the Rung 0.5 results into a Rung 2 design. Claude desktop drafted `Luminance-GS/Implementation_guides/Rung0.5_interpretation.md`; Claude Code checked it against the summary tables and the code, corrected it in place (its section 5 lists the changes), and Bart adopted it.
- **What was done:**
  - Rung 2 defaults: detached F (K = 32, F(0) = 0), fixed RGB basis, free s_k with mean 0; the bias net, shape net and colour basis are removed. Reference: `detach`; `baseline` stays as "official".
  - Ablations in order: attached s_k (F detached), on LOM and the varying-exposure set; colour-preserving luminance variant; K and knot spacing; fully attached F.
  - **New H3:** attaching only s_k to L_enh recovers the buu LPIPS lost by detaching, without bringing back the hue flips. The old H3 (colour basis) is reported as answered.
  - Framing: in detached mode the curve learns only from `HistogramPriorLoss`, whose inputs are the curve, the dark photo and the prior. No rendered information reaches the pseudo-targets, so the default is close to "2D enhancement, then 3DGS" with one shared curve.
  - Rung 3 demoted: spare time goes to the Rung 2 ablations first.
  - Updated `CLAUDE.md`, `docs/curve_estimation.md`, `docs/experiment_protocol.md` (planned CIELAB chroma metric, exposure-recovery check against the s_k initialisation, `detach` reference numbers, shrub rule), `docs/repo_vs_paper.md` and the explainer (sections 1, 8, 9 and 10; Q2 rewritten).
- **Reproduce:** branch `main`, commit `5ac7b5b` plus the uncommitted doc changes above; `examples/` clean. No runs.
- **Results** (corrections, all measured; sources `Luminance-GS/results/rung0/summary_rung05/tables.md` and `Luminance-GS/results/rung0/summary/tables.md`):
  - Corrects the 2026-10-24 Rung 0.5 entry ("paler renders") and `Rung0.5_results.md` line 141. Detaching does not make renders paler. Pooled `sat_ratio` is 0.617 for baseline and 0.709 for detach. The curve effect raises it on buu (+0.105, 12×), chair (+0.146, 22×) and sofa (+0.277, 13×). Bike's −0.136 is 1.6× noise (table 2a). Every cell is still below the ground truth (detach: chair 0.50, bike 0.50).
  - Corrects `Rung0.5_results.md` line 147: the exposure model will not fix global brightness. With mean s_k = 0 the exposures are relative; brightness comes from the CDF fit and, for the render, `L_spa`.
  - Qualifies Rung 0's "this is the number Rungs 1 and 2 must improve" (pseudo-target consistency). Per-view curve variation is equally small with and without detach (PC1 0.992 to 0.998; RMS 1.42 to 4.47/255 baseline, 1.46 to 4.87/255 detach; Rung 0 Q2 PCA table). A shared F with one s_k is therefore unlikely to change consistency by itself. `detach` has the least consistent targets and the best renders, so consistency becomes a diagnostic.
- **Takeaway:** the Rung 0.5 direction stands. *Inferred:* the detached targets' extra inconsistency comes from the curve's shape, so F's dark-end slope is the lever. The attached-s_k ablation matters most for buu and the varying-exposure set.
- **Next:** write the Rung 2 plan; build the varying-exposure set and run `baseline` and `detach` on it; re-render the bike 2×2 for seeds 1 and 2 before showing Rung 0.5 to Wei.

### 2026-11-06: Research direction approved by Prof. Chen and Wei; report due 18 Dec

- **Rung / type:** Rung 2 · decision
- **Why:** confirm that the Rung 2 direction (one shared monotone curve plus one exposure per view, H1 and H2) meets the ECE 701 requirements before the main experiments start.
- **What was done:** Bart shared the proposal slides with Wei on 2026-10-28 (that hand-off was not logged at the time). On 2026-11-06 at 14:10 he emailed the slides (`ECE701_Research_proposal_Bart_Kowal`) to Prof. Chen, cc Wei, and asked for approval and for the report deadline and format. The email summarised the reproduction (18.47 dB pooled PSNR on LOM vs 18.34 reported), Rung 0 and 0.5 (60 runs), the proposed model and H1/H2, and pointed to slides 2, 6, 7, 8 and 24.
- **Reproduce:** no runs and no code change. Source: the email thread "ECE701 research proposal: request for approval (Luminance-GS)", 2026-11-06, screenshots provided by Bart on 2026-11-08.
- **Results** (replies, quoted from the thread):
  - Wei, 14:30: "I am ok with this direction and share me any results you obtain in your experiments. Please also follow Prof. Chen's other suggestions."
  - Prof. Chen, 14:44: "The proposal looks good to me. Please send over your report in PDF by December 18. The format is flexible, but please include as much detail as possible."
  - Bart acknowledged both replies (14:38 and 14:52).
- **Takeaway:** the direction is approved by both supervisors. The report is a **PDF due 2026-12-18**, format flexible, as detailed as possible. That leaves 13 days after the 5 Dec experiment freeze. *Inferred:* "Prof. Chen's other suggestions" means the deadline and format guidance in Prof. Chen's reply; no other suggestions appear in this thread.
- **Next:** start the varying-exposure set and the H1/H2 runs. Send Wei results as they come in. Plan the report for a detail-heavy PDF.

### 2026-11-08: M1 plan and varying-exposure builder reviewed; init switched to the geometric mean

- **Rung / type:** Rung 2 (M1) · code + setup
- **Why:** check that Bart's M1 plan (`Luminance-GS/Implementation_guides/M1_vexp_plan.md`) and the new dataset builder (`Luminance-GS/examples/build_vexp.py`) work with the repo before step 1. Recipe from the explainer's section 8.7: a gain 2^u per training view, u in [−1, 1].
- **What was done:**
  - Audited both against `datasets/colmap.py`, `simple_trainer_ours.py`, `diagnostics.py`, `run_rung0.py`, `rescore_runs.py`, the LOM data and the 30 Rung 0 baseline/detach run folders. No blocking bug; the layout, the step 2 runner/re-scorer diffs, re-scoring with a LOM reference depth and the time estimate all hold.
  - Ran the builder's logic in memory on all five scenes, with no files written: gains, the JPEG round trip, the trainer's own `Parser`/`Dataset`/undistortion, the HE-CDF and `fit_exposure_form`. This exercised the `--verify` code, which had never run before.
  - `build_vexp.py`: (1) the s_k initialisation now uses the geometric mean, as in the explainer's section 8.6. The arithmetic mean gave the init a mean of +0.11 to +0.13 stops; slope and r are unchanged. (2) Rebuilding or re-verifying some scenes merges them into `gains.*`/`oracle.*`, and refuses before writing anything if the other rows used another seed, range or sampling. (3) The `low` path guard checks every scene. Merge, guards and dry run were tested in a scratch folder (deleted).
  - `.gitignore`: added `Luminance-GS/results/**/rescore/*.png`, and fixed the capital P in the proposal-PDF rule (it matched only through `core.ignorecase`).
  - Plan updated in place (untracked): tolerances, all-scene numbers, the H2 caveat, guards that keep `results/rung0/` safe, T4 residual RMSE, and T5 paired against signed u.
- **Reproduce:** branch `main`, commit `ea3f8ab` plus the uncommitted, untracked `build_vexp.py` and `M1_vexp_plan.md`, and the modified `.gitignore`. Dataset seed 701, stratified sampling. No training runs.
- **Results** (measured in memory with the luminancegs env; full table in `M1_vexp_plan.md` step 1):

| scene | clip frac at 2× | ratio error max (stops) | init slope / r | oracle slope / r | paired init RMSE (stops) |
|---|---|---|---|---|---|
| buu | 0 | 0.020 | −1.04 / −0.952 | −1.08 / −0.968 | 0.006 |
| chair | 0 | 0.046 | −1.04 / −0.991 | −1.09 / −0.995 | 0.013 |
| sofa | 0 | 0.011 | −1.02 / −0.991 | −1.07 / −0.996 | 0.004 |
| bike | 1.8e-6 | 0.060 | −0.95 / −0.990 | −1.04 / −0.989 | 0.019 |
| shrub | 0 | 0.028 | −0.98 / −0.983 | −1.11 / −0.980 | 0.014 |

  Re-encode PSNR minimum 50.4 dB. Natural LOM spread sd 0.20 (buu) and 0.08 to 0.10 stops (the others); u_k sd 0.58 to 0.59. Paired r (minus the same view on LOM) is −0.999 to −1.000 for the init and the oracle.
- **Takeaway:**
  - The plan's 0.02-stop ratio check would have failed on chair, bike and shrub. 8-bit rounding of very dark views (bike mean 7 levels) is not symmetric, so the tolerance is now 0.07.
  - *Inferred:* the initialisation already meets the explainer's H2 exposure threshold (\|r\| ≥ 0.9, slope near −1), so exposure recovery on this set is a sanity check, and the PSNR comparison against `detach` carries H2.
  - Discrepancies recorded: the builder's arithmetic-mean init disagreed with the explainer (fixed in the code). The explainer and the protocol describe independent draws, while the plan uses stratified draws (with an `--sampling iid` switch). Slides 20, 22 and 24 were not checked, because the PC has no PDF reader.
- **Next:** M1 step 1 (`python build_vexp.py --verify`), then the step 2 runner/re-scorer options and the smoke test. Tell Wei about the H2 threshold.

### 2026-11-08: Varying-exposure LOM set built and verified; runner options added; smoke test passed

- **Rung / type:** Rung 2 (M1) · setup + code
- **Why:** M1 steps 1 to 3 of `Luminance-GS/Implementation_guides/M1_vexp_plan.md`. The set is the H2 test set; this entry records it before the 30-run baseline batch.
- **What was done:**
  - Built `Luminance-GS/data/LOM_vexp/` (all 5 scenes). Dataset seed 701, stratified u_k in [−1, 1], JPEG quality 100 with no chroma subsampling. Verified it with the trainer's own `Parser`/`Dataset`; manifest in `Luminance-GS/results/vexp/dataset/` (gains, oracle, previews).
  - Checks: test views in `low/`, all of `high/` and `colmap_sparse/` are byte-identical to `LOM_full` (MD5); every training view changed. Split sizes, image order, cameras and validation images are identical (asserted). The previews brighten left to right.
  - Code: `run_rung0.py --data_root` (default `LOM_full`; it refuses a non-default data root with the default `results/rung0` root), and `rescore_runs.py --ref_root` (default `--root`). Verified: the default dry-run command is identical to before (diffed), the varying-exposure dry run lists 30 runs, and the Rung 0 re-score still reports 60 done.
  - Smoke test: buu, `baseline` and `detach`, seed 0, 300 steps, re-scored against `results/rung0/buu_baseline_seed0` depth.
  - Slides 9, 13, 20, 22 and 24 checked from screenshots Bart provided. They correct the 2026-11-08 review entry, which said the slides were not checked. Slide 20's recipe matches the builder, except that it says "draw u_k uniformly" (the set uses stratified draws).
  - Updated `CLAUDE.md` (Rung 2 in progress, repo map, `--data_root`/`--ref_root`) and the run-batch skill (data-root pre-flight, varying-exposure re-score command) before the batch.
- **Reproduce:** branch `main`, commit `2e714d6` plus uncommitted `run_rung0.py`, `rescore_runs.py`, `CLAUDE.md`, the run-batch skill and plan edits. From `Luminance-GS/examples`: `python build_vexp.py --verify`, then `python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp/smoke --scenes buu --variants baseline detach --seeds 0 --max_steps 300` and `python rescore_runs.py --root ../results/vexp/smoke --ref_root ../results/rung0`.
- **Results** (source `Luminance-GS/results/vexp/dataset/oracle.csv`; ratio error after undistortion):

| scene | n train | ratio error max (stops) | init slope / r | oracle slope / r | oracle explained | natural sd (stops) |
|---|---|---|---|---|---|---|
| buu | 22 | 0.020 | −1.04 / −0.952 | −1.08 / −0.968 | 0.957 | 0.20 |
| chair | 43 | 0.046 | −1.04 / −0.991 | −1.09 / −0.995 | 0.978 | 0.08 |
| sofa | 29 | 0.011 | −1.02 / −0.991 | −1.07 / −0.996 | 0.974 | 0.08 |
| bike | 36 | 0.060 | −0.95 / −0.990 | −1.04 / −0.989 | 0.984 | 0.08 |
| shrub | 30 | 0.028 | −0.98 / −0.983 | −1.11 / −0.980 | 0.983 | 0.10 |

  Build: max clip fraction 1.8e-6 (bike), 0 elsewhere; re-encode PSNR minimum 50.4 dB (console output). Smoke (`results/vexp/smoke/`, git-ignored): both runs `ok` (0.87 and 0.63 min). The trainer's `mean_input` per view over the LOM run's matches 2^{u_k} within 0.0198 stops (r 0.99998), and both re-scores used the `results/rung0` reference depth.
- **Takeaway:** the set matches the plan and slide 20's recipe, and the trainer reads the gained images. Numbers are identical to the in-memory check in the previous entry.
- **Next:** Bart starts the 30-run batch with `/run-batch` (about 5 h 10 min; stop the PC sleeping first), then re-score with `--ref_root ../results/rung0` and write `vexp_analyze.py`. When M1 is done, message Wei (plan step 7.4).

### 2026-11-08: M1 batch: official and `detach` on the varying-exposure set (30 runs), re-scored

- **Rung / type:** Rung 2 (M1) · experiment
- **Why:** M1 steps 4 and 5 (`M1_vexp_plan.md`). Measure what varying exposure costs the official code and the `detach` reference. This sets the bar Rung 2 has to beat for H2.
- **What was done:** 5 scenes × {`baseline`, `detach`} × seeds {0, 1, 2}, 10k steps, on `data/LOM_vexp`. All 30 runs are `ok` and each has `val_step9999.json`, `ckpt_9999.pt` and `curve_state.pt`. Re-scored all 30 against the LOM reference depth `results/rung0/<scene>_baseline_seed0` (checked in every `rescore/val_summary.json`). Flags per `cfg.json`: `--diagnostics`, plus `--detach_pseudo_target` for `detach`.
- **Reproduce:** branch `main`, commit `31bfa65` (`examples/` clean). From `Luminance-GS/examples`: `python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants baseline detach --seeds 0 1 2`, then `python rescore_runs.py --root ../results/vexp --ref_root ../results/rung0`. Wall time: 16:10 to 21:30 (official 11.4 to 15.1 min per run, `detach` 7.4 to 9.6); re-scoring about 3 s per run.
- **Results** on the varying-exposure set (mean of 3 seeds; ×noise = mean difference / pooled seed std; raw PSNR, SSIM and LPIPS from `stats/val_step9999.json`, bm PSNR from `rescore/`):

| scene | official raw / bm PSNR | `detach` raw / bm PSNR | bm cost, vexp − LOM: official / `detach` | `detach` − official: raw / bm / LPIPS | change in `detach` bm gain vs LOM |
|---|---|---|---|---|---|
| buu | 22.01 / 22.33 | 22.81 / 24.26 | −3.67 (6.5×) / −2.11 (2.0×) | +0.80 (1.0×) / +1.93 (1.6×) / −0.009 (0.7×) | +1.56 (2.2×) |
| chair | 14.76 / 17.82 | 15.49 / 19.89 | −3.12 (18.5×) / −3.20 (5.5×) | +0.72 (0.7×) / +2.07 (3.5×) / −0.041 (1.1×) | −0.08 (0.1×) |
| sofa | 18.77 / 20.04 | 21.67 / 23.84 | −5.19 (10.8×) / −3.57 (4.3×) | +2.90 (1.5×) / +3.80 (4.4×) / −0.076 (3.5×) | +1.62 (1.7×) |
| bike | 15.48 / 19.24 | 14.55 / 20.42 | −0.21 (1.5×) / −0.89 (3.5×) | −0.93 (0.9×) / +1.19 (4.3×) / −0.074 (6.4×) | −0.68 (2.7×) |
| shrub | 10.74 / 17.13 | 9.83 / 16.71 | −1.24 (1.5×) / −1.35 (3.8×) | −0.90 (1.8×) / −0.42 (1.0×) / −0.027 (0.6×) | −0.11 (0.2×) |

  Sources: `summarize_runs.py results/vexp` and `results/rung0 --variants baseline detach`; vexp − LOM differences computed from the same files, paired by scene, variant and seed. Other measurements:
  - **Seed noise is larger on the varying-exposure set:** `detach` raw PSNR sd is 2.59 dB on sofa and 1.38 on chair, against 0.29 and 0.03 on LOM.
  - **Raw PSNR on buu rises** by +1.20 (official) and +1.25 (`detach`) on the varying-exposure set, while bm PSNR falls.
  - **Bike hue error:** `detach` has 12.3° ± 1.0 on the varying-exposure set against 31.8° ± 1.9 on LOM. `detach` − official is −4.6° (5.0×) here and +10.3° (6.4×) on LOM.
- **Takeaway:**
  - *Measured:* varying exposure costs both methods 2 to 5 dB of bm PSNR on buu, chair and sofa, and up to about 1.4 dB on bike and shrub. `detach` still beats the official code on bm PSNR on chair, sofa and bike (above 2× noise). Its gain over the official code does not grow consistently under varying exposure: it moves by more than 2× noise only on buu (+1.56) and bike (−0.68), in opposite directions.
  - *Inferred:*
    - The H2 bar is `detach` on this set, with up to about 2 to 3.6 dB of bm PSNR to recover on buu, chair and sofa. Part of that cost is 8-bit quantisation on darkened views, which no model can recover (plan section 2 caveat).
    - The buu raw-PSNR rise is probably a brightness shift, not better quality. Check `mean_render` against `mean_gt` in T1.
    - The bike hue result is unexplained; it belongs in T7.
- **Next:** write and run `examples/vexp_analyze.py` (plan section 5: T1 to T7, per-view T5 against signed u, s_k tracking T4). Then write `M1_vexp_results.md` and message Wei (plan step 7.4). The H2 bar is the `detach` row above.

### 2026-11-08: M1 analysis (`vexp_analyze.py`): the official per-view curves do not compensate exposure

- **Rung / type:** Rung 2 (M1) · analysis + code
- **Why:** M1 step 6 (plan section 5). Turn the 30 varying-exposure runs and their 30 LOM partners into tables T1 to T7 and figures 1 to 4 for `M1_vexp_results.md`.
- **What was done:** wrote `examples/vexp_analyze.py`. It reuses the `rung0_analyze` helpers and `build_vexp.fit_exposure_form`/`slope_r`, and writes only to `results/vexp/summary/` (guarded).
  - It stops unless every run has the right `data_dir`, flags and LOM reference depth.
  - It recomputes the init and the histogram oracle on both sets, because `oracle.json` holds only the varying side.
  - Integrity checks, all passed:
    - re-score = training eval to 2.5e-6 dB;
    - the recomputed init and oracle equal `oracle.json` exactly;
    - `rel_l1_gt` is identical in all 60 runs.
- **Reproduce:** branch `main`, commit `31bfa65` plus the new uncommitted `examples/vexp_analyze.py`. Run `python vexp_analyze.py` from `Luminance-GS/examples` (242 s).
- **Results** (source `Luminance-GS/results/vexp/summary/tables.md`, `numbers.json`, `figures/`; bm = brightness-matched; pooled = 5 scenes):
  - **T1/T3, the H2 bar:**
    - On the varying set, `detach` vs official: pooled bm PSNR 21.03 ± 0.72 vs 19.31 ± 0.33 (+1.72, 3.1×); pooled LPIPS 0.306 vs 0.352 (−0.045, 3.5×).
    - The `detach` − official gap does not change detectably from LOM, pooled (bm +0.46, 0.8×).
  - **T2:**
    - Pooled cost of varying exposure in bm PSNR: official −2.68 (8.2×), `detach` −2.22 (4.3×).
    - Buu's raw-PSNR rise is a brightness shift: official validation renders went from 0.571 to 0.513 mean (GT 0.491; −0.058, 7.9×).
  - **T4, exposure (Q3):**
    - The s_k fitted to the final curves *rise* with u: slope +0.07 to +0.31 (official) and +0.10 to +0.44 (`detach`), r +0.80 to +0.98. Ideal is −1; the init gives −0.95 to −1.04 and the oracle −1.04 to −1.11.
    - Paired RMSE of s + u: 0.61 to 0.84 stops, against 0.004 to 0.019 for the init and 0.038 to 0.082 for the oracle.
    - Model-free check: curve values at fixed input levels correlate +0.94 to +0.97 with u (chair, buu, sofa, seed 0). The fit returns slope −1.000 on synthetic compensating curves.
    - Pseudo-target brightness leak (paired, 0 = compensated, 1 = passes through): official 0.27 to 0.56, `detach` 0.36 to 0.65. Render leak 0.06 to 0.54. Input check 1.00 to 1.02.
    - Cross-view curve spread (×255): 2.3 to 2.6 on LOM, 6.6 to 7.8 on the varying set (pooled).
  - **T5, the dark branch (Q4):**
    - The shared dark render's fit to the input drops 3.6 to 11.9 dB per scene (pooled −7.4 official, −7.5 `detach`).
    - Per view it drops on both sides of u = 0, more steeply for brightened views (pooled slope −13.4 dB/stop for u > 0 vs +5.7 for u < 0, official).
  - **T6:** pseudo-target consistency `rel_l1_pseudo` (pooled) 0.071 → 0.202 (official) and 0.084 → 0.248 (`detach`); GT floor 0.062.
  - **T7:**
    - Bike `detach` hue error 31.8° → 12.3° (12.9×); saturation ratio falls on chair and sofa.
    - Wall time +0.6 min (official) and +0.2 min (`detach`) per run.
- **Takeaway:**
  - *Measured:* the official per-view curves (and `detach`'s) do not compensate exposure. Their per-view part is a near-uniform upward shift that grows with input brightness, so it goes in the opposite direction. The pseudo-targets still half-compensate only because the shared curve is concave.
  - *Inferred:* an explicit s_k initialised at log2(m̄/m_k), which already has slope −1, removes that leak by construction. That is the mechanism H2 relies on. The shared dark branch is a large cost that Rung 2 keeps unchanged (plan 3.3: a 2^{−s_k} dark branch is an M2 design question for Bart).
- **Next:** write `M1_vexp_results.md` (Q1 to Q4), update `experiment_protocol.md`, the explainer's H2 note and `CLAUDE.md`, then message Wei.

### 2026-11-09: M1 done: results report, revised H2 criterion, and the sign of Rung 0's curve variation

- **Rung / type:** Rung 2 (M1) · decision + analysis
- **Why:** M1 step 7. Answer Q1 to Q4 in a report, and carry what M1 changes into the protocol and the H2 criterion.
- **What was done:**
  - Wrote `Luminance-GS/Implementation_guides/M1_vexp_results.md` from `results/vexp/summary/`. Numbers as in the previous entry; it adds the varying-set reference table for Rung 2.
  - Added notes, without rewriting old text, to:
    - `docs/experiment_protocol.md`: the set as built, the `results/vexp` root, the RMSE, leak and dark-branch metrics, the H2 reference table, and a correction to "beat the initialisation";
    - explainer sections 8.7 and 8.11 (H2 note);
    - `CLAUDE.md` (H2 line, Rung 0 and Rung 2 rows, docs index).
  - Ticked the plan's "Done when" list.
  - Two extra read-only checks on the local curve snapshots, below.
- **Reproduce:** branch `main`, commit `9344652` (`examples/` clean). The two checks were inline Python on `diagnostics/curves_step*.npz`, `curve_bias` averaged over bins 0 to 29.
- **Results** (*measured*):
  - **Sign of Rung 0's curve variation.** On LOM, the per-view curve bias correlates +0.97 to +1.00 with the view's log2 mean input brightness in all 30 Rung 0 runs (`baseline` and `detach`, seeds 0 to 2). Rung 0 reported only |r|; the variation *follows* brightness, it does not compensate it.
  - **The direction is set early and only grows.** On the varying set (seed 0, both variants), r(bias, u_k) is +0.94 to +0.97 in every scene from step 1000 to step 10000, and at step 1 in four scenes (shrub −0.95). The spread across views grows from about 0 at step 1 to 0.017 (buu) to 0.16 (shrub `detach`) at step 10k.
  - **Pseudo-target consistency in units of the GT floor:** `detach` goes from 0.97–1.72 on LOM to 1.60–8.98 on the varying set; official from 0.69–1.29 to 2.27–6.23.
- **Decision** (recorded in the protocol and the explainer):
  - H2's \|r\| ≥ 0.9 exposure threshold is met by the initialisation alone, so it becomes a sanity check.
  - H2 is carried by bm PSNR against `detach` on the varying set, using that set's own seed noise (`detach` bm sd up to 1.51 dB), together with Rung 2's pseudo-target leak (T4b).
  - To tell Wei.
- **Takeaway** (*inferred*): Rung 2's s_k starts on the compensating side (slope −1) where the official curves sit on the wrong one. A near-zero pseudo-target leak would be a qualitative difference, not only a parameter saving. Why the bias network goes the wrong way is not identified.
- **Next:** message Wei (H2 bar, revised criterion, stratified draws, slide 20's extra check). Then M2: build Rung 2 behind flags. The open design question from plan section 3.3 is Bart's call: a per-view 2^{−s_k} on the dark branch.

### 2026-11-11: M2 start: exposure-aware dark branch (option A), and a review of the M2 plan

- **Rung / type:** Rung 2 (M2) · decision
- **Why:** M1 found that one shared dark branch cannot fit photos up to 2 stops apart. Its fit to the dark input drops from 46.2 to 38.8 dB (pooled, `detach`), by 3.6 to 11.9 dB per scene (source: `Luminance-GS/Implementation_guides/M1_vexp_results.md`, Q4). Rung 2 changes only the curve, so it would inherit this misfit, and a weak H2 result could not be pinned on either part.
- **Decision** (Bart, 2026-11-09, option A; plan: `Luminance-GS/Implementation_guides/M2_rung2_plan.md`):
  - New flag `--dark_branch {shared, measured, model, gains}`, default `shared` (the official code). Otherwise view k's dark render is multiplied by 2^{−s_k} before `loss_regress_low`. `measured` = log2(m̄/m_k), fixed; `model` = Rung 2's own s_k; `gains` = the set's true gains, for diagnosis only.
  - Stop-gradient: `loss_regress_low` never trains s_k, so H3 stays separate.
  - The 2×2 on the varying set: rows are the curve (`detach` or Rung 2), columns the dark branch (shared or exposure-aware): `detach`, `detachexp`, `rung2`, `rung2exp`. The row effect is H2; the column effect is the dark branch's share of the varying-exposure cost.
  - The switch is off on LOM, so H1 is unchanged. It goes slightly beyond the approved plan: tell Wei on 13 Nov.
- **Review** (Claude Code, 2026-11-10, against the code at `c5fa94c`, read-only plus in-memory checks): part A matches the code. Folded into the plan, each marked "review":
  - fixes: the sign in unit test 9 (the synthetic CDFs must be G(min(1, 2^{s_k} x))); milestone numbers (the deck has M4 = the 13 Nov check-in, so the ablations are M5); the LOM-row wording (the content/exposure split is unknown, as M1 found); the guard order in step 1;
  - added: `num_GS` per cell, because the switch also re-weights each view's `L_low` gradient by 2^{−s_k}; a ratio-based speed estimate (M1's 300-step `detach` smoke took 0.63 min, which scales to 21 min against the real 8.1; source `results/vexp/smoke/runs.csv`); same-seed repeats as the reference for the determinism checks; notes on reusing the analysis helpers; the `--ref_root` rule for every root other than `results/rung0`.
- **Reproduce:** branch `main` (Bart chose to stay on it), commit `c5fa94c`, `examples/` clean. No runs and no code changes in this entry.
- **Open, for Bart:**
  - the dark-end alignment of Rung 2's curve loss, before part C. `detach`'s truncating LUT maps level ℓ to about CDF[ℓ−1]; the plan's interpolated fit maps it to about CDF[ℓ], one bin brighter. A one-bin shift of the target removes that (recommended);
  - the H1 rule: plan 6.5 (buu, chair, sofa, bike; not below −2× noise) against the explainer's 8.11 (every scene, within the detectable effect);
  - whether LOM overexposure is part of H1 (a question for Wei).
- **Takeaway** (*inferred*): the measured s_k is off from the true gains by 0.08 to 0.19 stops on this set (`M1_vexp_plan.md`, step 1 table), and darkened views lost tonal levels to 8-bit rounding. So `detachexp`'s dark fit should climb toward 46.2 dB but not reach it.
- **Next:** doc updates (explainer, `CLAUDE.md`, protocol, curve notes, settings, run-batch skill), then part A: the switch, its checks and a 300-step smoke test. Then ask Bart before the 15-run `detachexp` batch (about 2 h).

### 2026-11-11: M2 part A: `--dark_branch` built; smoke test passed after two fixes

- **Rung / type:** Rung 2 (M2) · code
- **Why:** M2 plan part A (steps 1 to 3): build the exposure-aware dark branch decided in the previous entry, and check it before the 15-run `detachexp` batch.
- **What was done:**
  - `simple_trainer_ours.py`:
    - `Config.dark_branch`, default `shared`;
    - `Runner.measure_exposures`: s_k = log2(m̄/m_k) from `trainset[i]`, written to `exposure_init.json`;
    - `Runner.load_gains` and `Runner.dark_scale`;
    - `train()` multiplies `colors_low` by 2^{−s_k} right after the render split. The `model` guard runs before anything is written.
  - `diagnostics.py`: `s_dark` and `fit_psnr_lowrender_exp_vs_input` are the last two columns of `pseudo_targets.csv`. `_load_run_state` asserts the run's `dark_branch` (`MODEL_DEFAULTS`).
  - `rescore_runs.py`: variant names may contain digits, and `MODEL_FLAGS` are passed on from `cfg.json`. `run_rung0.py`: variant `detachexp` = `--detach_pseudo_target --dark_branch measured`.
  - **Two bugs found by the smoke test, both fixed:**
    - `torch.exp2` on CUDA fails in this env: torch 1.13.1 compiles it at run time (NVRTC), and `nvrtc-builtins64_117.dll` is missing. The factor is now exp(−ln2·s).
    - The flag loop in `rescore_runs.py` reused the progress counter `k`. Only the printout was wrong.
- **Reproduce:** branch `main`, commit `c5fa94c` plus uncommitted changes to the four files above. From `Luminance-GS/examples`:
  - `python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp/smoke_m2 --scenes buu --variants detach detachexp --seeds 0 --max_steps 300`;
  - then `python rescore_runs.py --root ../results/vexp/smoke_m2 --ref_root ../results/rung0`;
  - plus one direct trainer run with `--dark_branch gains` (`buu_detachgains_seed0`).
  - The smoke folders are git-ignored.
- **Results** (*measured*, from the smoke runs' `cfg.json`, `exposure_init.json`, `diagnostics/` and `rescore/`):
  - **Default path unchanged:**
    - `run_rung0.py --dry_run` prints identical commands for the old variants (0 differences);
    - `rescore_runs.py --root ../results/vexp --dry_run` still finds the 30 M1 runs re-scored;
    - the new `detach` run's step-0 loss row equals M1's smoke run to every printed digit, and later rows agree within 2.2e-4 relative.
  - **Exposures:** `s_measured` equals `oracle.json` `s_init` for all 22 buu views (max difference 0.0). `gains` gives −(log2_ratio − mean) within 1.1e-7.
  - **Built-in check:** `mean_pseudo` is identical between `detach` and `detachexp` (max difference 0.0, in `diagnostics/` and `rescore/`). `detach`'s new fit column equals its old one exactly.
  - **Dark fit as trained,** `detachexp` − `detach` (buu, seed 0, 300 steps): +1.89 dB on average, higher on 12 of 22 views. By u:
    - +9.40 dB for u < −0.5 (n = 5);
    - +1.44 for −0.5 ≤ u < 0;
    - −1.80 for 0 ≤ u < 0.5;
    - −0.85 for u ≥ 0.5 (n = 6).
- **Takeaway** (*inferred*):
  - The switch works as designed and leaves the default path and the curve path untouched.
  - At 300 steps there is no densification yet (`num_GS` 1090 in both runs), and the gain is clear only on darkened views. For brightened views the factor (up to 2×) also scales up the immature render's structural error, which absolute-error PSNR penalises.
  - The 10k-step batch decides.
- **Next:** ask Bart before the `detachexp` batch (15 runs, about 2 h). Part B (`rung2.py`, tests, preview) can be written meanwhile, as new files only.
