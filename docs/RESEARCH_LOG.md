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
