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

### 2026-11-11: M2 open items decided; `detachexp` batch launched

- **Rung / type:** Rung 2 (M2) · decision + setup
- **Why:** the three open items of the 2026-11-11 review entry, and plan step 4 (the `detachexp` batch).
- **Decisions** (Bart, 2026-11-11):
  1. **Dark-end alignment: the one-bin shift.** Rung 2's curve loss and `cdf` init use T_k = [0, CDF_k[0], …, CDF_k[253]], so every level ≥ 1 reads the same CDF bin as `detach`'s truncating LUT and level 0 stays black. The step 6 fallback is then the smoothness ablation in M5.
  2. **H1 rule: plan 6.5.** In each of buu, chair, sofa and bike, Rung 2 − `detach` in brightness-matched PSNR is not below −2× noise, and SSIM and LPIPS are not worse by more than 2× noise. Shrub is reported but does not decide. This replaces the explainer's 8.11 wording (every scene, "within the detectable effect").
  3. **Overexposure in H1:** pending Wei (13 Nov). Until then H1 is tested on low light only.
  - Recorded in `M2_rung2_plan.md` (header, 2.2, step 6, 6.5, risk 1, section 9), the explainer (8.6, 8.11, Q8, Q15), `docs/experiment_protocol.md`, `docs/curve_estimation.md` and `CLAUDE.md` (H1 line).
- **Setup:** Bart committed part A as `b8b811a`. With Bart's go-ahead, Claude started the batch in the background at 14:17 (pre-flight: GPU free, 3.9 TB free, dry run 15 runs and 0 done, sleep on AC already 0).
- **Reproduce:** branch `main`, commit `b8b811a`. From `Luminance-GS/examples`: `python -u run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants detachexp --seeds 0 1 2`, console in `results/vexp/batch_console_detachexp.log` (git-ignored).
- **Results:** pending (batch running).
- **Next:**
  - After the batch: re-score with `--ref_root ../results/rung0`, quick look (`summarize_runs.py`, `detach` vs `detachexp`), and a results entry.
  - Part B is written but not yet logged (`rung2.py`, `test_rung2.py`, `rung2_preview.py`; uncommitted). It gets its own entry once the full preview has finished, together with the shape-prior finding from its first run.

### 2026-11-11: M2 part B: Rung 2 module, tests and preview; the dropped shape prior breaks H1's premise

- **Rung / type:** Rung 2 (M2) · code + experiment (CPU only, no 3DGS)
- **Why:** plan steps 5 and 6: build Rung 2 as new files and preview its curve before any GPU run.
- **What was done:** `rung2.py` (`ExposureCurve`, knots, `cdf` init, `curve_loss`, one-bin `hist_target`); `test_rung2.py` (13 tests: the plan's 10 plus target/round trip, `used_range_max`, alignment against `tools.LUT_mapping`); `rung2_preview.py` (replays 10k steps of 10·L_curve per view with the trainer's optimisers, plus a full-batch optimum). After the first run, two EXPLORATORY configs (not in the plan): the official shape-prior term 0.5[mse(p, X) + 0.01 mse(p, T_k)] with one learned global gamma, on the shared F (`priorF`) or on each composed curve (`priorL`).
- **Reproduce:** branch `main`, commit `b8b811a` plus the three new (untracked) files. From `examples/`: `python test_rung2.py`; `python rung2_preview.py` (lr 1e-3, `results/rung2/preview/`, 920 s); `python rung2_preview.py --lr_F 1e-2 --out ../results/rung2/preview_lrF1e-2` (918 s). Seed 0, CPU, 2 threads, during the batch.
- **Results** (*measured*; sources: the two `tables.md`):
  - Tests 13/13. Synthetic recovery of s within 0.0001 stops; shifted target vs official LUT at levels 1-16 max 0.0015 (unshifted 0.098).
  - **Shape prior:** `detach`'s seed-0 curves sit at (HE-CDF + pseudo curve)/2: mean |diff| 0.009-0.020, against 0.034-0.119 from either alone (buu, bike, sofa on LOM; buu varying). The plan's L_curve ("L_hist minus the shape-prior term") therefore fits pure HE, a different target.
  - P5 machinery check: `detach`'s own curves through the preview's code reproduce its re-scored `pseudo_psnr_gtmean` within 3.8e-6 dB (buu LOM, 22 views).
  - Plan default (`quadratic-cdf`), LOM, pseudo-target bm PSNR − `detach`'s (P5, diagnostic only): buu −10.85, chair −7.00, sofa −12.81, bike −8.45, shrub +1.50 dB; P5 rule fires. Same at lr 1e-2. Applied-curve distance to `detach` (P4, training photos only) 32-46 levels.
  - `priorF` at lr 1e-2: γ 0.164/0.108/0.136/0.075/0.077 vs `detach` mean γ 0.195/0.095/0.149/0.065/0.068; distance to `detach` on LOM 5.4-14.3 levels; P5 bm gap LOM −0.07/+0.06/−0.19/+0.00/+0.22, varying −0.50/+2.07/+1.10/+0.60/+1.61. At lr 1e-3 it cannot reach that (gap −5.8 to −11.3 on four scenes).
  - Pseudo-target leak (P3): default +0.007 to +0.098, `priorF` −0.003 to +0.029, `priorL` +0.086 to +0.122; M1 `detach` +0.36 to +0.65.
  - Plan rules: knots stay quadratic (usedrange −8.4%/−9.1%, < 30%); lr gap ≤ 2.64% (1e-3), ≤ 0.33% (1e-2). The identity init never comes within 0.01 of the `cdf` F at either lr.
- **Takeaway:**
  - *Measured:* the H2 mechanism holds in the replay (leak near 0, s_k tracking the gains with slope −0.89 to −1.04).
  - *Inferred:* H1 as planned would compare different targets. One global gamma on F (K + N + 1 parameters) reproduces `detach`'s targets on LOM. F needs lr ≈ 1e-2, because steps on softplus increments move it far less than the official curve's direct steps.
  - The one-bin shift was premised on `detach` ≈ CDF[ℓ−1]. That premise fails at the dark end, where the prior dominates. Level 0 still differs: `detach` maps it to 68-86 levels on chair, bike and shrub, and F(0) = 0 maps it to 0.
- **Next:** Bart decides the curve-loss default (prior on F with one gamma, lr 1e-2; or as planned), and whether to revisit the one-bin shift. Then part C, and a final preview run with the decided default. P5 never chose anything here; it agrees with the training-photo distance.

### 2026-11-11: M2 step 4: `detachexp` on the varying-exposure set (first dark-branch column effect)

- **Rung / type:** Rung 2 (M2) · experiment
- **Why:** plan step 4. Measures the exposure-aware dark branch's effect for the `detach` curve before any Rung 2 run. This is the batch launched in the decisions entry above.
- **What was done:** 15 runs (5 scenes × seeds 0-2), all `ok`, 7.5-9.8 min (mean 8.3; M1 `detach` 8.1), batch 14:17-16:22. Re-scored against `results/rung0/<scene>_baseline_seed0` depth, about 9 s per run. Quick look only: `summarize_runs.py --variants detach detachexp --reference detach`, plus two read-only scratch scripts for dark fit, `num_GS` and pooled means. `cfg.json` and re-score flags were checked for all 15 runs.
- **Reproduce:** branch `main`, commit `b8b811a` (the untracked part B files are not imported). From `Luminance-GS/examples`: `python -u run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants detachexp --seeds 0 1 2`, then `python rescore_runs.py --root ../results/vexp --ref_root ../results/rung0`. Consoles in `results/vexp/batch_console_detachexp*.log` (git-ignored).
- **Results** (*measured*). Each cell is `detachexp` − `detach`, paired by seed, with ×noise = mean Δ / pooled seed sd. Dark fit = mean PSNR of the dark render, as L_low sees it, against the input. Sources: each run's `rescore/val_summary.json`, `rescore/pseudo_targets.csv` and `stats/val_step9999.json`.

  | Scene | bm PSNR | raw PSNR | SSIM | LPIPS | dark fit: shared → exp (LOM `detach`) | num_GS |
  |---|---|---|---|---|---|---|
  | buu | −0.24 (−0.2×) | −1.13 (−2.0×) | +0.0222 (+2.2×) | −0.0351 (−2.1×) | 41.18 → 42.41 (46.25), +1.2× | −31.5% (−4.7×) |
  | chair | +0.47 (+0.8×) | +0.67 (+0.6×) | +0.0213 (+1.2×) | −0.0437 (−1.2×) | 38.02 → 46.74 (49.02), +8.5× | −26.7% (−2.5×) |
  | sofa | +0.53 (+0.5×) | +1.88 (+0.9×) | +0.0130 (+1.2×) | −0.0024 (−0.1×) | 35.57 → 45.44 (47.42), +6.6× | −23.3% (−3.7×) |
  | bike | +0.42 (+1.4×) | +0.69 (+0.8×) | +0.0263 (+1.7×) | −0.0052 (−0.5×) | 41.86 → 47.30 (47.72), +12.4× | −16.5% (−7.5×) |
  | shrub | +0.11 (+3.4×) | −0.07 (−0.5×) | +0.0049 (+3.6×) | −0.0052 (−3.9×) | 37.11 → 40.68 (40.81), +15.3× | −5.0% (−2.4×) |
  | pooled | 21.03 → 21.28, +0.26 (+0.5×) | +0.41 (+0.5×) | +0.0176 (+1.9×) | −0.0183 (−1.4×) | 38.75 → 44.52 (46.24), +17.9× | |

  - Built-in check (plan 2.1, 6.1): same-seed pseudo-targets are identical; the largest |Δ log2 `mean_pseudo`| is 0.0000 in all 15 pairs.
  - Pooled seed sd shrinks: bm PSNR 0.72 → 0.31, raw 1.15 → 0.25, SSIM 0.0126 → 0.0019, LPIPS 0.0167 → 0.0069.
  - Per-view dark-fit gain vs |u_k|: r = 0.18 on buu, 0.53-0.74 on the others.
  - Colour, same table: bike hue error −3.35° (−2.5×). The saturation ratio moves away from 1 on bike (−0.024, −4.4×) and buu (0.940 → 0.847, −2.2×).
- **Takeaway:**
  - *Measured:* the switch does its job. Dark fit is +5.8 dB pooled, real in chair, sofa, bike and shrub. On buu, `detach`'s own dark fit varies 1.5 dB across seeds, so buu's +1.2 dB is not detectable. Pooled, the fit ends 1.7 dB below LOM: buu −3.8, chair −2.3, sofa −2.0, bike −0.4, shrub −0.1.
  - *Measured:* SSIM and LPIPS improve in all five scenes, but only shrub (all three metrics) and buu (SSIM, LPIPS) pass 2×. Pooled, nothing passes 2× (SSIM +1.9×). Brightness-matched PSNR does not move beyond noise except on shrub, +0.11 dB.
  - *Measured:* `detachexp` grows 5-32% fewer Gaussians, real in all five scenes. *Inferred:* with the shared dark branch, extra Gaussians absorb the exposure mismatch between views. The column effect is therefore not only "better fit", as review item 7 predicted.
  - *Inferred:* the smaller seed noise in the exposure-aware column lowers the detectable effect for `rung2exp` − `detachexp`. The bar in that column is 21.28 dB bm pooled.
  - Plan 5.3 `detachgains` ("dark fit more than 2 dB below LOM"): not met pooled (1.7 dB), met on buu and chair individually. It runs only with Bart's OK.
- **Next:** send Wei this column effect (plan step 4). Bart decides the Rung 2 curve-loss default (part B entry above), then part C and M3.

### 2026-11-13: Rung 2 curve loss keeps the shape prior (one global gamma on F); part B choices confirmed

- **Rung / type:** Rung 2 (M2) · decision
- **Why:** the open items of the 2026-11-11 part B entry. Plan step 6's stop rule had fired, and part C waited on these choices.
- **Decisions** (Bart, 2026-11-13):
  1. **Shape prior kept, as one learned global γ on the shared F** (`--rung2_prior F`, the default). L_curve gains the official term 0.5·[mse(p, F) + 0.01·mse(p, T_k)], with p = clamp(x, 1e-3, 1)^γ and γ = exp(ρ) starting at 1. Parameters: K + N + 1.
     - *Reason:* in the official loss the HE weight drops to 0.5 at step 3000, and from then on the best curve sits halfway between the HE-CDF and the gamma curve. `detach`'s curves sit there. Dropping the prior (the plan's 2.2 "L_hist minus the shape-prior term") changed what Rung 2 is trained toward, so H1 would have failed for that reason alone. Bart: a mistake in the plan, and a fix to it rather than tuning on the normal-light images.
     - *Transparency:* the P5 flag, which uses the normal-light images, is what prompted the investigation. The decision rests on the official loss and on training-photo evidence: the decomposition of `detach`'s curves and the P4 applied-curve distance.
     - *Not the official form* (`priorL`, the prior on each view's curve): it leaks (+0.09 to +0.12) and pulls s_k toward 0.
     - Tell Wei: the prior puts back a piece the agreed design had removed.
  2. **Learning rate of F and γ:** by plan step 6's own rule, applied with the prior on: final objective within 5% of the full-batch optimum, trying 1e-3, then 3e-3, then 1e-2. Bart asked that closeness to `detach` stay out of the justification, because it leans toward the answer H1 tests. s stays at 1e-3. Result: next entry.
  3. **Knots: quadratic**, the pre-registered rule (`usedrange` −8.4% / −9.1%, below 30%). The `cdf` init stays (the identity init never converged).
  4. **One-bin shift: kept.** A learnable black level is an M5 ablation only, as a fallback if M3 shows a problem at level 0.
  5. **`detachgains`: not now.**
- **Recorded in:** `M2_rung2_plan.md` (2.2 note, 7.1, 8.4), `CLAUDE.md`, `docs/curve_estimation.md` ("Rung 2 as built"), explainer 8.6 and the 10. TL;DR, `docs/experiment_protocol.md`.
- **Next:** part C and the final preview with the learning-rate ladder (entries below), then ask Bart before M3.

### 2026-11-13: M2 step 6, final preview: the learning-rate rule with the prior on picks 1e-2

- **Rung / type:** Rung 2 (M2) · experiment (CPU only, no 3DGS)
- **Why:** Bart's condition on the learning rate (entry above), and the preview of the decided default.
- **What was done:**
  - `rung2_preview.py` reworked. The default is `quadratic-cdf-priorF`, using the prior inside `ExposureCurve`, the trainer's own code. It is replayed at lr_F 1e-3, 3e-3 and 1e-2 (F and γ; s at 1e-3). Around it: `usedrange`, the identity init, the unshifted target, no prior, and `priorL`.
  - First, a scratch check: the plan's full-batch optimum (3k steps at 1e-2) is not converged with the prior. Bike LOM gives 5.012e-3 against 4.4295e-3 at 30k steps, 13% above. So the optimum is now 10k steps at 3e-2 (within 0.013% of 30k), or a polish of the best replay if that ends lower.
- **Reproduce:** branch `main`, commit `9b9e070` plus uncommitted `rung2.py` and `rung2_preview.py`. From `examples/`: `python rung2_preview.py --threads 4` (1,286 s, seed 0), writing `results/rung2/preview_final/`.
- **Results** (*measured* on the replay; source `results/rung2/preview_final/tables.md`):
  - **Ladder** (largest gap above the optimum over 5 scenes × 2 sets): 1e-3 +201.5%; 3e-3 +7.0% (bike +6.2% LOM, +7.0% varying; the others +0.9 to +3.8%); 1e-2 +0.82%. The rule picks **1e-2**.
  - **Knots:** `usedrange` −0.3%, so quadratic stays. **Init:** the identity init never comes within 0.01, so `cdf` stays.
  - **Default at 1e-2:**
    - γ 0.164/0.108/0.136/0.075/0.077, the same on both sets within 0.001.
    - Leak (P3) −0.003 to +0.029. s_k against u_k: slope −0.91 to −1.04, paired RMSE 0.010–0.037.
    - Applied-curve distance to `detach` on LOM (P4): 5.4–14.3 levels. Unshifted with the prior: 6.9–15.3. No prior: 32–46.
  - **P5 (diagnostic only), bm gap to `detach`:**
    - LOM −0.07/+0.06/−0.19/+0.00/+0.22 dB; varying set −0.50/+2.07/+1.10/+0.60/+1.61 dB.
    - The flag fires on buu and sofa when measured against `detach`'s pseudo-target seed sd (0.02 and 0.00 dB). Against H1's noise (`detach`'s validation bm sd, 0.25 and 0.40 dB) the gaps are 0.3× and 0.5×.
  - **Visual (scratch):** bike view 29 through the final replayed F looks the same as `detach`'s pseudo-target, noise included. Input levels 0/1/2/8/16 map to 0/80/89/193/227.
- **Takeaway:**
  - *Measured:* the plan's own learning-rate rule, applied with the prior on, picks 1e-2. At 1e-3, F and γ cannot reach their optimum in 10k steps; at 3e-3 they fall short on bike.
  - With the prior, the shifted target is still closer to `detach` than the unshifted one in every LOM scene (training photos only).
  - P5 changed nothing.
- **Next:** part C (next entry), then Bart's go for M3.

### 2026-11-13: M2 part C: Rung 2 in the trainer (`--curve_model rung2`); regression and smoke checks passed

- **Rung / type:** Rung 2 (M2) · code + setup
- **Why:** plan steps 7 to 9.
- **What was done:**
  - `rung2.py`: the prior inside `ExposureCurve` (`prior`, `rho`, `gamma`, `prior_curve`, `loss`, `f_params`), and `curve_loss(prior=...)`. New test 14: the prior term equals the official algebra, and with the prior on F, s gets no gradient from it. 14/14 tests pass.
  - Trainer:
    - `Config` gains `curve_model` and the `rung2_*` flags; every default reproduces the official code, and `rung2_lr_F` defaults to 1e-2. The trainer asserts that `MODEL_DEFAULTS` equals the `Config` defaults.
    - New: the guards, `build_rung2`, `rung2_forward`, Rung 2's loss in `train()`, schedulers built over every curve-side optimiser, the gauge projection, end-of-run asserts, and `grad_check_rung2`.
  - `diagnostics.py`: Rung 2 branches (snapshot keys, `rung2_state.json`, `exposures_by_step.csv`, curve-state save and load). `rescore_runs.py`: `MODEL_FLAGS`. `run_rung0.py`: variants `rung2`, `rung2exp` and `rung2att`, refused in `results/rung0`.
- **Reproduce:** branch `main`, commit `9b9e070` plus uncommitted changes to these files. From `examples/`, all into git-ignored smoke folders:
  - `run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp/smoke_m2 --scenes buu --variants rung2 rung2exp rung2att --seeds 0 --max_steps 300`;
  - `run_rung0.py --root ../results/rung2/smoke_m2 --scenes buu bike --variants rung2 --seeds 0 --max_steps 300`;
  - a `detach` smoke into `results/vexp/smoke_m2c`;
  - re-scores with `--ref_root ../results/rung0`;
  - 20-step `--grad_check` runs.
- **Results** (*measured*):
  - **Default path unchanged:**
    - The runner's dry-run commands are identical to HEAD's.
    - Re-scores of `results/vexp/buu_detach_seed0` and `results/rung0/buu_baseline_seed0` with the new code are bit-identical to their `rescore/` outputs (max |diff| 0 in every shared column and key).
    - The 300-step `detach` loss log is identical at step 0 (`loss_regress_low` 0.2573000490665436). Steps 100–200 are within 1e-5, the same gap as between the two older smokes.
  - **Rung 2 smokes (all `ok`):**
    - 55 parameters for buu (32 + 22 + 1) and 69 for bike. |mean s| ≤ 2.2e-8, F strictly increasing, `s_init` equal to `oracle.json` within 2.7e-8.
    - `rung2` and `rung2exp` with the same seed: s, F, γ and `mean_pseudo` are bit-identical.
    - `fit_exposure_form` on `curve_adj` recovers the run's own s_k within 0.0008–0.0034 stops. The re-scored `mean_pseudo` is identical to the run's.
  - **grad_check:**
    - `rung2`: L_enh and L_low send nothing to θ, s or ρ.
    - `rung2att`: s gets 0.239 from L_enh; θ and ρ get nothing.
    - `rung2exp`: s gets nothing from L_low.
    - Official `detach`: the output is unchanged.
  - **Speed:** 300-step smokes take 0.51–0.59 min for `rung2`, against 0.62 for `detach`.
- **Takeaway:**
  - Part C is done, so M2 is complete apart from Bart's commit.
  - *Inferred:* about 6.6–7.7 min per 10k-step Rung 2 run, so M3 (45 runs) takes about 5–6 h.
  - The 300-step bike pseudo-target is mid-trajectory (γ 0.54), so the dark end is judged from the preview, not from the smoke image.
- **Next:** Bart's go for the M3 batch (plan 5.1, with the sleep check). Tell Wei that the prior is back.

### 2026-11-14: P5 flag on the final preview cleared; M3 batch launched

- **Rung / type:** Rung 2 (M2 → M3) · decision + setup
- **Why:** plan step 6 says to stop and ask Bart when P5 flags two or more LOM scenes. The final preview (entry of 2026-11-13) flagged buu and sofa.
- **Decision** (Bart, 2026-11-14): **confirmed, proceed with the default.**
  - The plan's "LOM's brightness-matched seed noise" means `detach`'s *validation* noise, the noise H1 is judged against.
  - Against that bar the pseudo-target gaps are small. Buu: −0.07 dB against a validation bm sd of 0.25 dB, i.e. −0.3×. Sofa: −0.19 dB against 0.40 dB, i.e. −0.5×. Source of the sd: `results/rung0/<scene>_detach_seed{0,1,2}/rescore/val_summary.json`.
  - **Doc vs code (rule 8):** `rung2_preview.py` divides by `detach`'s *pseudo-target* bm PSNR sd instead: 0.02 dB (buu) and 0.00 dB (sofa). The same gaps then read −4.7× and about −700×, which is why the flag fired. `results/rung2/preview_final/` stays as written.
  - P5 chose nothing; the default is unchanged.
- **Setup:** Bart committed parts B and C as `c304662` and approved M3. At his request (option b of the run-batch skill), Claude started the plan's four commands as one background process at 17:18.
  - Pre-flight: tree clean; all 10 data folders complete; GPU free; 3.9 TB free; sleep on AC already 0 (never); dry runs list 10 + 5 + 20 + 10 = 45 runs, none done.
  - The first run printed 55 curve parameters with the decided defaults.
- **Reproduce:** branch `main`, commit `c304662`. From `Luminance-GS/examples`, in order:
  - `python -u run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants rung2 rung2exp --seeds 0`
  - `python -u run_rung0.py --root ../results/rung2 --variants rung2 --seeds 0`
  - the same two with `--seeds 1 2`

  Consoles in `results/{vexp,rung2}/batch_console_m3_{seed0,seeds12}.log` (git-ignored).
- **Results:** pending. *Inferred* estimate: seed 0 done about 19:00–19:15, all 45 runs about 22:15–23:05.
- **Next:** at seed 0, re-score those runs with `--ref_root ../results/rung0`, then run `rung2_analyze.py --seeds 0` (`results/rung2/summary_seed0/`). After all seeds, re-score and run the full analysis.

### 2026-11-14: M3 results: H2 supported; H1 fails on bike by the pre-registered rule

- **Rung / type:** Rung 2 (M3) · experiment + code (analysis)
- **Why:** plan sections 5 and 6: the 2×2 on the varying-exposure set (H2) and Rung 2 on LOM (H1). This is the batch launched in the entry above.
- **What was done:**
  - **Batch:** all 45 runs `ok`, 17:18 to 20:48, 3.7–7.0 min per run (mean 4.67; the estimate of 6.6–7.7 was too high).
  - **Re-score:** every run re-scored against `results/rung0/<scene>_baseline_seed0` depth. The 15 seed-0 runs were re-scored at 20:35–20:38 while the last LOM runs trained, so the wall times of `sofa_rung2_seed2` and `bike_rung2_seed2` may be about 0.2–0.3 min high.
  - **`rung2_analyze.py` (new, CPU only, plan section 6):** tables R1–R6, the 6.5 decision rules, figures 1–6, and a seed-0 mode. Tested three ways:
    - `--selftest`: the Rung 2 cells are copies of `detach`/`detachexp`, and every curve-model effect came out exactly 0.
    - `--smoke`: runs on the 300-step smoke runs.
    - On the real runs it reproduces M1's T1 means, paired leak and dark-fit slopes exactly (difference 0).
  - **Integrity:**
    - `rung2` vs `rung2exp`, same seed: s, F and pseudo-targets bit-identical in all 15 pairs.
    - `fit_exposure_form` on `curve_adj` recovers each run's own s_k within 0.0066 stops.
    - The trainer's s_k match the preview's within 0.0044 stops.
- **Reproduce:** runs at commit `c304662`. From `examples/` (the script is uncommitted): `python rung2_analyze.py` (all seeds, 103 s) and `python rung2_analyze.py --seeds 0`.
- **Results** (*measured*; source `results/rung2/summary/tables.md`; seed-0 snapshot in `summary_seed0/`). Values are rung2 − detach on LOM, paired by seed (×noise, * = real):

  | Scene | bm PSNR | raw PSNR | SSIM | LPIPS | H1 rule |
  |---|---|---|---|---|---|
  | buu | −0.12 (−0.5×) | −1.54 (−3.8×) * | −0.0071 (−2.0×) | −0.0202 (−2.3×) * | meets |
  | chair | +0.40 (+2.0×) | +1.07 (+4.5×) * | +0.0065 (+7.7×) * | −0.0164 (−7.6×) * | meets |
  | sofa | −0.39 (−1.1×) | −0.34 (−1.0×) | −0.0002 (−0.1×) | −0.0090 (−1.3×) | meets |
  | bike | −0.15 (−3.8×) * | +0.33 (+4.2×) * | −0.0004 (−0.1×) | +0.0076 (+1.4×) | **fails: bm PSNR** |
  | shrub (reported only) | +0.16 (+0.5×) | +0.15 (+0.3×) | −0.0076 (−2.4×) * | +0.0498 (+1.3×) | — |
  | pooled | −0.02 (−0.3×) | −0.07 (−0.5×) | −0.0018 (−1.2×) | +0.0023 (+0.3×) | — |

  - **H1 extras (LOM, pooled):**
    - Pseudo-target bm PSNR 23.80 vs 23.80.
    - Hue error +1.23° (+4.7×, worse), mostly bike: 38.5° vs 31.8°, +6.75° (+3.3×). Angle error −0.48° (−2.7×, better).
    - Curve parameters 55–76 vs about 255k; 4.6 vs 7.8 min per run.
  - **H2, varying set, pooled factorial:**
    - Curve-model main effect: bm PSNR **+2.11 dB (+5.3×)**, raw +1.35 (+2.3×), SSIM +0.0155 (+2.3×), LPIPS −0.0242 (−2.5×).
    - Per scene, bm PSNR: buu +1.84 (+1.6×), chair +3.59 (+7.6×), sofa +2.95 (+3.5×), bike +0.56 (+2.7×), shrub +1.64 (+67.7×; LPIPS +0.054 worse).
    - Shared column +2.09 (+4.0×); exposure-aware column +2.14 (+9.1×). Dark-branch main effect +0.28 (+0.7×); interaction +0.05 (+0.1×).
    - Cells, bm PSNR: `detach` 21.03 ± 0.72, `detachexp` 21.28 ± 0.31, `rung2` 23.11 ± 0.11, `rung2exp` 23.42 ± 0.12.
  - **Mechanism:**
    - Paired pseudo-target leak: `rung2` 0.012 vs `detach` 0.512, a difference of −0.500 (−720×). Render leak 0.004 vs 0.222.
    - Rung 2's s_k: slope −1.00, paired RMSE 0.018 stops (init 0.011, oracle 0.052, `detach` post-hoc 0.726).
    - Dark fit: `rung2` 37.24 → `rung2exp` 44.87 dB (+7.63, +50.6×); LOM level 46.14.
- **Takeaway:**
  - **H2: supported** by the pre-registered rule. The main effect is real and positive and LPIPS is not worse. It is positive in all four non-shrub scenes (buu alone at +1.6×) and real in both dark-branch columns. The mechanism criterion also holds: the leak is near 0 and below `detach`'s.
  - **H1: not supported** by the pre-registered rule. Buu, chair and sofa meet it. Bike fails on brightness-matched PSNR by 0.15 dB, which is −3.8× because bike's seed noise is only 0.04 dB. Pooled, the two models agree (−0.02 dB, −0.3×). The rule is reported as written, not reinterpreted.
  - *Inferred:* Rung 2 is nearly insensitive to the exposure variation. Its pooled bm PSNR is 23.23 on LOM and 23.11 on the varying set, while `detach` drops from 23.25 to 21.03 (unpaired cell means from R1a and R3). Rung 2's seed sd on the varying set is also 0.11 against `detach`'s 0.72.
  - *Inferred, untested:* possible bike causes are the dark end (level 0 maps to 0, against `detach`'s 84 levels; risk 1) and the hue shift.
- **Next:**
  - Bart decides on H1 follow-up. Plan 5.3 pre-declares only an identity-init head-start check for a borderline H1 (5 runs; about 25 min at Rung 2's speed). The learnable black level is an M5 fallback.
  - `Rung2_results.md` by hand (Q1–Q5); send Wei the results.
  - M5: H3 (`rung2att`).

### 2026-11-15: H1 reported as not supported; head-start check launched, with its reading fixed before results

- **Rung / type:** Rung 2 (M3) · decision + setup
- **Decision** (Bart, 2026-11-15): options 1 and 2 of the H1 follow-up.
  1. H1 is reported as **not supported**, as the pre-registered rule says.
  2. The plan's pre-declared conditional run (plan 5.3, risk 5) is done.
- **What was done:**
  - **New variant:** `rung2id` in `run_rung0.py` = `--curve_model rung2 --rung2_F_init identity`, added to `NOT_IN_RUNG0`. Existing variants are unchanged.
  - **Smoke run:** `results/rung2/smoke_m2/buu_rung2id_seed0`, 300 steps, 0.5 min. The log prints "F init identity". End checks: |mean s| = 2.7e-09, F strictly increasing, 55 curve parameters.
  - **Launch:** Claude started the 5 runs at 12:43 in the background. Pre-flight: GPU free, sleep on AC 0. Console: `results/rung2/batch_console_rung2id.log` (git-ignored).
- **Reproduce:** branch `main`, commit `3bf7477` plus the uncommitted `run_rung0.py` change. From `examples/`: `python run_rung0.py --root ../results/rung2 --variants rung2id --seeds 0`.
- **Reading, fixed before any result** (plan 5.3 gives none):
  - **Comparisons:** seed 0, `rung2id` − `rung2` and `rung2id` − `detach`, per scene and pooled, in bm PSNR, raw PSNR, SSIM and LPIPS. With one seed there is no "real": each difference sits next to the 3-seed sd of the M3 cells. Also reported: how far the identity-init F ends from the `cdf`-init F, and the s_k.
  - **The check cannot change the H1 verdict:** the rule applies to the pre-registered default (`cdf` init).
  - **If the head start explains bike:** `rung2id` − `rung2` is above 2× `rung2`'s bike sd (0.03 dB) and closes most of the −0.15 dB gap to `detach`. That would make the `cdf` head start a likely contributor (*inferred*, one seed).
  - **If it doesn't:** the difference stays within about 2× that sd.
  - **Caveat:** the curve-only preview never brought the identity init within 0.01 of the `cdf` F. A difference could therefore also come from a different final F, not only from the head start, so the F gap is reported alongside.
- **Also measured while it runs** (existing runs, CPU only; for `Rung2_results.md`):
  - **Bike margin:** bm PSNR −0.155 dB against a bar of −0.082 dB (2 × 0.041). All three seeds are negative (−0.11, −0.24, −0.11). The six runs separate fully (Mann-Whitney exact p = 0.10, the smallest possible with 3 vs 3).
  - **Noise uncertainty:** from 4 degrees of freedom, the 95% interval for the noise is 0.025–0.118 dB.
  - **Near miss:** buu's SSIM passes at −1.96× noise (bar −2×).
  - **Bike, training views:** the pseudo-targets are as good as `detach`'s (bm +0.002 dB). The render fits them as well (+0.07 dB, 1.7×). The render's bm PSNR against GT is still −0.165 dB (−3.8×), the same gap as on validation, so it is not a generalisation effect.
  - **Correction (rule 4):** the entry above names the dark end as a possible bike cause, and Claude's 2026-11-14 message called the black level the more likely lever. The evidence now weakens this (*inferred*):
    - only 0.8% of bike's training pixels sit at level 0, against 14.1% on shrub, where Rung 2 is not worse (+0.16 dB);
    - bike's pseudo-target bm PSNR is equal.

    The 84-level figure is right: `detach`'s bike curve maps level 0 to 83–85 of 255; Rung 2 maps it to 0.
- **Results:** pending.
- **Next:** re-score with `--ref_root ../results/rung0`, compare as fixed above, then write `Rung2_results.md`.

### 2026-11-15: Head-start check: the identity init of F changes Rung 2's LOM result (+0.31 dB pooled, one seed); H1 verdict unchanged

- **Rung / type:** Rung 2 (M3, plan 5.3) · experiment + code (analysis)
- **Why:** the conditional run of the entry above, with the reading fixed there before any result.
- **What was done:**
  - **Runs:** 5 runs `ok`, 12:45–13:08, 3.7–6.5 min each.
  - **Re-score:** done with `--ref_root ../results/rung0`, 7–8 s per run.
  - **New script `rung2_headstart.py`** (CPU only): it checks the inputs first. Each run must be `ok` and re-scored against `rung0/<scene>_baseline_seed0`, and its cfg must equal the `rung2` seed-0 run's except for `rung2_F_init`. It writes `results/rung2/summary_headstart/`. Its `--selftest` reads `rung2id` from `rung2` seed 0, and every difference came out exactly 0.
  - **`Rung2_results.md` written** (Q1–Q5, with the long H1 section).
- **Reproduce:** branch `main`, commit `3bf7477` plus the uncommitted `run_rung0.py` (variant `rung2id`) and the new `rung2_headstart.py`. From `examples/`:
  - `python run_rung0.py --root ../results/rung2 --variants rung2id --seeds 0`
  - `python rescore_runs.py --root ../results/rung2 --ref_root ../results/rung0`
  - `python rung2_headstart.py`
- **Results** (*measured*, seed 0; M3's 3-seed sd in brackets; source `results/rung2/summary_headstart/tables.md`):

  | Scene | rung2id − rung2, bm PSNR | rung2id − detach, bm PSNR | rung2id − rung2, raw PSNR | SSIM | LPIPS |
  |---|---|---|---|---|---|
  | buu | +0.80 [0.20] | +0.38 [0.25] | +0.32 | +0.0120 | −0.0213 |
  | chair | −0.01 [0.26] | +0.63 [0.12] | +0.07 | +0.0100 | −0.0114 |
  | sofa | +0.49 [0.27] | +0.18 [0.40] | +0.11 | +0.0039 | −0.0029 |
  | bike | +0.25 [0.03] | +0.14 [0.05] | −0.13 | +0.0297 | −0.0012 |
  | shrub | +0.00 [0.03] | +0.54 [0.50] | +0.02 | −0.0038 | +0.0016 |
  | pooled | +0.31 [0.07] | +0.37 [0.07] | +0.08 | +0.0104 | −0.0070 |

  - **Hue error:** −1.84° pooled; on bike −3.84°, down to `detach`'s 33.7°.
  - **Pseudo-targets on the training views** move by +0.85, +0.27, +1.11, +0.33 and −0.01 dB bm PSNR (buu, chair, sofa, bike, shrub).
  - **F gap:** the identity-init F never comes within 0.01 of the `cdf`-init F. At 10k steps it is 8–13 levels away, mostly at levels 1–2 (bike: level 1 → 66 against 79), and still closing (bike 0.065 at 5k, 0.051 at 10k).
  - **Training objective:** the curve loss ends 0.5–1.6% *higher* with the identity init in all five scenes.
- **Takeaway:**
  - **By the pre-declared reading:** the head start is a likely contributor to the bike gap (*inferred*, one seed). Bike's +0.253 dB exceeds both 2 × 0.034 and half the 0.155 dB gap.
  - **The init changes where F ends up, not only how fast it gets there.** The identity-init curve stays less lifted at levels 1–2, and the ground truth prefers it.
  - *Inferred:* the equalisation target over-lifts the deepest shadows. An early-stopped curve does better there, as `CLAUDE.md`'s Rung 1 note on the HE-CDF's dark-end jump anticipated.
  - **H1 is still not supported** (pre-registered default).
  - **The identity init is not adopted.** `cdf` was chosen by the plan's training-photo rule, the training objective still prefers it, and the identity init wins only on ground-truth metrics. Any identity-init result is exploratory and labelled post hoc.
  - **Correction of the entry above:** the dark end does matter for bike, but at levels 1–2, not at level 0. Both inits keep F(0) = 0.
- **Next:** Bart decides whether to confirm with seeds 1 and 2 (`rung2id` on LOM, 10 runs, about 45 min; exploratory). Send Wei M3 plus this check. In M5, the dark-end levers that use training photos only (smoothness λ, K and knot spacing) come before any F-init ablation.

### 2026-11-16: Identity-init confirmation (seeds 1 and 2) launched, with its reading fixed before results

- **Rung / type:** Rung 2 (M3 follow-up) · decision + setup. **Exploratory:** the setting was picked after seeing seed-0 validation results.
- **Decision** (Bart, 2026-11-16): confirm the identity init with seeds 1 and 2 on LOM. That is 10 runs, about 45 min.
- **Setup:**
  - `rung2id` with seeds 1 and 2, `results/rung2/`. The flag set is unchanged from the 2026-11-15 smoke run.
  - Pre-flight: HEAD `993636e`, tree clean except `.vscode/`, GPU free, sleep on AC 0, 3.6 TB free. The dry run lists 10 runs.
  - Claude started the batch in the background at 21:23. Console: `results/rung2/batch_console_rung2id_s12.log`.
- **Reproduce:** from `examples/`, `python run_rung0.py --root ../results/rung2 --variants rung2id --seeds 1 2`, then `python rescore_runs.py --root ../results/rung2 --ref_root ../results/rung0`.
- **Reading, fixed before any result:**
  - **Comparisons:** paired by seed, seeds 0–2, `rung2id` − `rung2` (the init effect) and `rung2id` − `detach`. Per scene, pooled and pooled without shrub: bm PSNR, raw PSNR, SSIM, LPIPS and hue error. Noise is sqrt(mean of the two arms' seed variances), and an effect is real above 2× (protocol).
  - **Clean replication:** seed 0 chose the question, so seeds 1 and 2 alone are also reported (mean of 2, next to the 3-seed noise).
  - **Init effect confirmed:** pooled `rung2id` − `rung2` in bm PSNR is real and positive, LPIPS is not worse beyond 2× noise, and the seed 1–2 mean is positive. **Not confirmed:** the pooled effect is within 2× noise. Bike is judged separately by the same test.
  - **Exploratory only:** `rung2id` − `detach` is put through the 12 H1 checks (plan 6.5). It cannot change the H1 verdict, which belongs to the pre-registered default. Any pass is labelled post hoc.
  - **The identity init is still not adopted on this evidence:** the training objective preferred `cdf` (entry above). Whether a training-photo-only lever (smoothness λ, K and knot spacing) gives the same dark end is an M5 question.
  - **Also reported:** the F gap and curve loss per seed.
- **Results:** pending.

### 2026-11-16: Identity init confirmed over 3 seeds (+0.44 dB pooled); exploratory H1 check meets 12 of 12; pre-registered H1 verdict unchanged

- **Rung / type:** Rung 2 (M3 follow-up) · experiment + code (analysis). **Exploratory** (post hoc).
- **Why:** the confirmation launched in the entry above, judged by the reading fixed there.
- **What was done:**
  - **Runs:** 10 runs `ok`, 21:23–22:08, 3.7–6.6 min each. Re-scored with `--ref_root ../results/rung0`, 8–11 s per run.
  - **`rung2_headstart.py`:** gained `--seeds 0 1 2` (paired effects, the pre-declared decisions, the H1 rule applied to `rung2id` − `detach`, curve loss per seed) and `--out` for test outputs under `smoke_m2/`.
  - **Tests, both before use:**
    - the seed-0 mode reproduces the committed `summary_headstart/` byte for byte;
    - the 3-seed `--selftest` gives 0 for every init effect, and its H1 table reproduces M3's R1b exactly (11 of 12, bike −0.155 dB at −3.8×).
- **Reproduce:** branch `main`, commit `993636e` plus the uncommitted changes to `rung2_headstart.py`. From `examples/`:
  - `python run_rung0.py --root ../results/rung2 --variants rung2id --seeds 1 2`
  - `python rescore_runs.py --root ../results/rung2 --ref_root ../results/rung0`
  - `python rung2_headstart.py --seeds 0 1 2`
- **Results** (*measured*; paired by seed over seeds 0–2, ×noise, * = real; source `results/rung2/summary_headstart_seeds012/tables.md`):

  | Scene | rung2id − rung2, bm PSNR | Seeds 1–2 | SSIM | LPIPS | rung2id − detach, bm PSNR |
  |---|---|---|---|---|---|
  | buu | +0.63 (3.2×) * | +0.55 | +0.0129 * | −0.0249 * | +0.52 (2.3×) * |
  | chair | +0.01 (0.1×) | +0.02 | +0.0100 * | −0.0099 * | +0.41 (2.4×) * |
  | sofa | +1.26 (3.7×) * | +1.64 | +0.0089 * | −0.0184 * | +0.87 (2.2×) * |
  | bike | +0.26 (10.9×) * | +0.26 | +0.0249 * | +0.0035 | +0.10 (3.1×) * |
  | shrub | +0.01 (0.6×) | +0.02 | −0.0015 | +0.0034 | +0.17 (0.5×) |
  | pooled | +0.44 (6.2×) * | +0.50 | +0.0110 * | −0.0093 * | +0.42 (6.0×) * |

  - **Hue error:** −1.88° pooled (8.4×), bike −4.95° (3.2×).
  - **F:** identical across seeds to 0.1 of a level for both inits. The identity-init F stays 8–13 levels from the `cdf` F in every seed.
  - **Curve loss:** higher with the identity init in all 15 pairs, by +0.3% to +1.6%.
- **Takeaway:**
  - **By the pre-declared reading:** the init effect is confirmed (pooled, pooled without shrub, bike).
  - **Exploratory:** `rung2id` meets all 12 H1 checks against `detach` (closest: bike LPIPS +1.7×).
  - **The pre-registered H1 verdict stands** (default `cdf` init: not supported).
  - *Inferred:* the bike failure comes from fitting F fully to the equalisation target, not from the shared-curve-plus-exposure model. The gain is an implicit regularisation (an under-fitted dark end, tied to init, learning rate and step count). The identity init is not adopted as the default without Bart's decision. Untested on the varying set (H2).
- **Next:** Bart and Wei decide:
  - how to present this (the report gives both results side by side);
  - whether to test `rung2id` on the varying set (15 runs, about 1.2 h, exploratory);
  - whether M5 looks for a training-photo-only lever that gives the same dark end (smoothness λ, K and knot spacing).

### 2026-11-16: Correction: the deep-shadow lift comes from the shape prior, not the equalisation target

- **Rung / type:** Rung 2 (M3 follow-up) · correction (rule 4). Found while choosing figures for Wei.
- **What was wrong:** two earlier entries (2026-11-15 head-start results, and the 3-seed entry above) and `Rung2_results.md` gave an *inferred* cause. They said the equalisation target over-lifts the deepest shadows and that the identity init fits it less fully. Figure 6 (`results/rung2/summary/figures/fig6_curves.png`, LOM panel) contradicts this. There the views' HE-CDFs are near 0 at the darkest levels, while both `detach`'s curves and Rung 2's F jump to about 0.22 at level 1.
- **Measured** (seed 0, `curves_step*.npz`, × 255 at levels 1–4):
  - **Equalisation target** (`cdf_target`, mean over views): bike 1, 8, 25, 52; sofa 0, 0, 1, 3.
  - **Shape prior x^γ:** bike 169–187 (γ 0.075); sofa 120–145 (γ 0.136).
  - **F at step 1:** `cdf` init 0, 2, 12, 25 and identity 1, 2, 3, 4 (bike), so both start near 0.
  - **F at 10k:** bike `cdf` 79, 87, 96, 106 against identity 66, 85, 101, 117. Sofa `cdf` 56, 64, 69, 74 against identity 47, 58, 66, 72.
- **Corrected reading** (*inferred*):
  - The lift in the deepest shadows comes from the official shape prior, which Rung 2 kept (D1, 2026-11-13). `detach`'s curves have it too.
  - The two inits end at different compromises between prior and target. The identity init ends less lifted at level 1, closer to the target, and the ground truth prefers that.
  - The measured results are unchanged: the init effect, the curve loss preferring `cdf`, and the F gap.
- **Consequence for M5:** the first principled lever is the shape prior. That means the already-planned `--rung2_prior none` ablation, or a lower prior weight. The smoothness weight λ and the knots come after it.
- **Updated:** `Rung2_results.md`, Q1 head-start section and M5 section.

### 2026-11-24: M5 decisions and pre-registered readings (H1b, H2b, H3 restated, E4, A1, H1-over), before any M5 run

- **Rung / type:** Rung 2 (M5) · decision. Plan: `Luminance-GS/Implementation_guides/M5_plan.md` (revised after step 0; commit `fb8f14d`).
- **Why:** M3 left four questions open:
  - H3 was never run;
  - the identity init is untested on the varying set;
  - the identity gain may only be early stopping;
  - bike's hue on the varying set is still poor.
- **Decisions (Bart, 2026-11-24):**
  - option B: a fresh pre-registered H1b test of the identity variant on new seeds;
  - M4 is done (M3 results emailed to Wei);
  - Rung 3 stays, as M6;
  - overexposure joins as E6, on Wei's general go-ahead.
- **Step 0 review** (Claude Code; Bart approved the fixes):
  - seven steps could not run as written, and are fixed in the plan;
  - definitions were added to the readings below.
- **H3 restated** (old text kept in `CLAUDE.md` with a dated note):
  - Why: the default already meets the old criterion. Buu LPIPS on LOM is 0.1824 for `rung2`, against official 0.1862 and `detach` 0.2026 (`results/rung2/summary/tables.md`, R1a). So it cannot tell `rung2att` from `rung2`.
  - New text: "Sending the image loss into the per-view exposures (F stays detached) changes quality on LOM or the varying set."
- **Readings, fixed before any run.** Common rules: pair by seed; noise = sqrt(mean of the arms' seed variances); real above 2×; pooled = five-scene mean per seed, checked against pooled without shrub; shrub not deciding.
  - **H1b (E3; `detach`, `rung2id`; LOM; seeds 3–5):**
    - the H1 rule's 12 checks, with noise from seeds 3–5 only;
    - pre-registered, but the variant was chosen from seed-0 results;
    - it does not change H1;
    - the 6-seed view is context only.
  - **H2b (E1; `rung2id`, `rung2idexp`; varying set):**
    - (a) the init main effect ½[(`rung2id` − `rung2`) + (`rung2idexp` − `rung2exp`)], noise over the four cells. Pooled bm PSNR real and positive with LPIPS not worse = confirmed; within 2× = not confirmed; real and negative = reversed; real and positive with LPIPS worse = mixed. Bike is judged alike;
    - (b) the H2 rule with the identity cells, where 1×–2× reads "inconclusive", with no extra seeds;
    - (c) report the leak and the s_k slope; a pooled leak above 0.1 is flagged.
  - **H3 (E2; `rung2att` − `rung2`; LOM and the varying set's shared column; `cdf` init):**
    - helps: on either set, pooled bm PSNR higher or pooled LPIPS lower beyond 2×;
    - harm signals: a decision scene's bm PSNR below −2× on LOM; bike hue worse beyond 2× on either set; on the varying set, a pooled leak above 0.1 or a pooled s_k slope outside −0.9 to −1.1;
    - verdicts: helps, hurts, mixed or no effect.
  - **E4 (`rung2noprior` − `rung2`, LOM):**
    - report only: every metric, plus F at grid b = 1–4, × 255;
    - expected much worse pseudo-targets: 7.0–12.8 dB below `detach` on the four decision scenes (`results/rung2/preview_final/tables.md`, P5).
  - **A1 (CPU replay, both inits, 50k steps):** gap(t) = max over the grid of |F_id − F_cdf| at the same step. Outcomes:
    - gap(50k) < 0.01 = early stopping;
    - gap(50k) ≥ 0.01 and gap(40k) − gap(50k) < 10% of gap(40k) = two end points;
    - otherwise undecided.
  - **H1-over (E6; `baseline`, `detach`, `rung2` with `--exp_name over_exp`):**
    - the H1 rule for `rung2` − `detach`, with official alongside;
    - reported next to the low-light verdict, never merged with it;
    - stop rule before launch, from a curve-only preview: γ ≤ 1, or F below the identity at fewer than 80% of interior grid points, or a pooled fit residual above 2× the low-light preview's.
- **Takeaway:** no setting is chosen and no default changes in M5. Every variant is labelled pre-registered or exploratory.
- **Next:** step 2 (runner variants and `--exp_name`, the preview script, seven 300-step smoke runs and their re-scores, the `over_exp` preview), then Bart launches batches 1, 2 and E6 with `/run-batch`.

### 2026-11-24: M5 step 2: runner variants and `--exp_name`, `rung2_replay.py`, smoke tests passed; E6 pre-launch preview passes its stop rule

- **Rung / type:** Rung 2 (M5) · code + setup. No result here is used to decide anything except the E6 stop rule logged above.
- **Code:**
  - `run_rung0.py`: variants `rung2idexp` and `rung2noprior` (both refused in `results/rung0`); `--exp_name {low, over_exp}` (default `low`).
    - Guards: `over_exp` only with a root inside `results/overexp/` (subtree test); `low` refused there; a data root without `<scene>/over_exp/` refused.
    - Tests: five refusals, none creating a folder. Dry runs before and after: 0 differing lines over 156 (timestamps masked). The six launch commands list 45, 15, 30, 15, 45 and 15 jobs.
  - `rung2_preview.py`: `exp_name` parameters in `load_set`, `objective` and `replay`, defaulting to `low` (its own outputs unchanged).
  - New `rung2_replay.py`:
    - `--mode overexp` (E6 pre-launch check) and `--mode long` (A1);
    - its output guard refused three test paths, two of them rule-6 folders;
    - code-tested on bike (long, 2k steps) and buu (overexp, 1k steps) in `smoke_m5/` folders.
  - `rung2.curve_weight`: 0.1 for `over_exp` from step 3000, 0.5 for `low`, 1 before; unit-checked on the CPU.
- **Reproduce:** branch `main`, commit `fb8f14d` plus uncommitted changes to `run_rung0.py` and `rung2_preview.py` and the new `rung2_replay.py`. From `examples/`:
  - smoke: `run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp/smoke_m5 --scenes buu --variants rung2id rung2idexp --seeds 0 --max_steps 300`, likewise `--root ../results/rung2/smoke_m5 ... rung2att rung2noprior` and `--exp_name over_exp --root ../results/overexp/smoke_m5 ... baseline detach rung2`; then `rescore_runs.py --root <each> --ref_root ../results/rung0`;
  - preview: `python rung2_replay.py --mode overexp`.
- **Smoke results** (buu, seed 0, 300 steps; all 7 `ok`, 0.5–0.9 min; re-scores 7 s each, reference `rung0/buu_baseline_seed0`; re-score vs training PSNR within 1.3e-6; source: the runs' `diagnostics/` and `rescore/`):
  - `rung2idexp` vs `rung2id` on the varying set: max |Δ s| = max |Δ F heights| = max |Δ log2 mean_pseudo| = 0.
  - `rung2noprior`: 54 curve parameters (K 32 + N 22), prior `none`, γ NaN, F strictly increasing; its re-score works.
  - `over_exp` smokes: `exp_name` `over_exp`; mean input 0.915 against 0.091 for the low-light smoke, so they train on `over_exp/`.
- **E6 pre-launch preview** (`results/overexp/preview/tables.md`; training photos only, default configuration, 10k steps, 61 s):
  - **passes the stop rule** in every scene: γ 12.77 (buu), 4.05 (chair), 5.26 (sofa), 4.44 (bike), 1.94 (shrub); F below the identity at 100% of interior grid points;
  - pooled fit residual 18.44/255 against 45.01 in low light (ratio 0.41);
  - mean CDF[253] at least 0.998 in every scene, so the pinned F(1) = 1 does not clash with the shifted target's top value.
  - *Inferred:* F ends close to the shape prior x^γ in every scene, and the darkening is strong (buu maps level 127 to about 0.5/255). Expect very dark pseudo-targets on buu.
- **A1 cost:** measured 8 s for 6k replay steps, so the full run (5 scenes, 2 × 50k + 2 × 10k steps each) is about 13 min of CPU. That is under the 30-minute threshold, so no ask is needed.
- **Next:** Bart commits, then launches the six commands with `/run-batch` (about 15.3 h, or 16.5 h with E6's optional `rung2id`; sleep off). During the batches: the `rung2_analyze.py` changes, `rung2_ablation.py`, and the A1 run.

### 2026-11-24: M5 batch launched (E1–E4 and E6, 150 runs), readings fixed before results

- **Rung / type:** Rung 2 (M5) · setup. The readings are the 2026-11-24 decision entry above, logged before any run.
- **Pre-flight:**
  - commit `dbdcfdb`, clean tree (`.vscode/` only); GPU free; 3.6 TB free; sleep on AC already 0;
  - an extra smoke run, `rung2id` on `over_exp` (`results/overexp/smoke_m5/`): `ok`, re-scored, 55 curve parameters.
- **Launch:**
  - Claude started the six commands of `M5_plan.md` steps 3, 4 and 4b at 13:09:44, in order, as a detached PowerShell process (option b, Bart's go-ahead);
  - console logs `<root>/batch_console_m5_{e1e2,e2,e3,e4,e6,e6b}.log`.
- **Reproduce:** from `examples/`:
  - `run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants rung2id rung2idexp rung2att --seeds 0 1 2`
  - `run_rung0.py --root ../results/rung2 --variants rung2att --seeds 0 1 2`
  - `run_rung0.py --root ../results/h1b --variants detach rung2id --seeds 3 4 5`
  - `run_rung0.py --root ../results/rung2 --variants rung2noprior --seeds 0 1 2`
  - `run_rung0.py --exp_name over_exp --root ../results/overexp --variants baseline detach rung2 --seeds 0 1 2`
  - `run_rung0.py --exp_name over_exp --root ../results/overexp --variants rung2id --seeds 0 1 2`
- **Expected time** (*inferred* from measured minutes per run: `baseline` 12.6, `detach` 7.9, Rung 2 4.5–4.7): about 16.3 h. E1–E4 should end about 22:00, everything about 05:30 on 25 Nov.
- **Results:** pending.

### 2026-11-25: M5 batch finished (165 runs, all ok), one sleep interruption

- **Rung / type:** Rung 2 (M5) · setup. Completes the 2026-11-24 launch entry above. Correction to it: the six commands hold **165** runs (45 + 15 + 30 + 15 + 45 + 15), not 150; 150 left out E6b.
- **Outcome (measured, `runs.csv` of each root):** 165 of 165 runs `ok`, none failed. Status file: E1+E2 (varying set) done 18:12:45, E2 LOM 19:18:26, E3 22:09:36, E4 23:16:35, E6 05:18:05, E6b 06:34:46 (ALL DONE). Wall time 17 h 25 min from 13:09:44.
- **Sleep interruption:**
  - The PC slept from 15:35:16 to 17:19:22 on 24 Nov (Kernel-Power 42, "System Idle"), although the AC sleep timeout was 0. A USB device woke it.
  - `vexp/chair_rung2id_seed2` was suspended at step 4944. It resumed and finished `ok`, but its recorded 108.2 min is not a compute time. As with Rung 0's `sofa_detach_seed1`, its results are kept and its time is excluded. *Inferred:* the pause does not change training (no wall-clock dependence in the trainer).
  - From 21:27:57, a helper (scratchpad `stay_awake.ps1`, `SetThreadExecutionState` ES_CONTINUOUS | ES_SYSTEM_REQUIRED, no power settings changed; Bart's OK) kept the PC awake while the launcher ran. It stopped by itself at 06:35:11 and cleared the request. There was no further sleep during the batch.
- **Minutes per run** (mean, range; slept run excluded):

  | Root | Variant | Mean min | Range |
  |---|---|---|---|
  | `vexp` | `rung2id` | 4.4 | 3.6–6.3 |
  | `vexp` | `rung2idexp` | 4.3 | 3.5–6.2 |
  | `vexp` | `rung2att` | 4.5 | 3.6–6.5 |
  | `rung2` | `rung2att` | 4.4 | 3.6–6.3 |
  | `rung2` | `rung2noprior` | 4.5 | 3.6–6.5 |
  | `h1b` | `detach` | 7.1 | 6.3–9.1 |
  | `h1b` | `rung2id` | 4.3 | 3.6–6.5 |
  | `overexp` | `baseline` | 11.1 | 9.3–14.2 |
  | `overexp` | `detach` | 7.9 | 6.8–11.2 |
  | `overexp` | `rung2` | 5.1 | 4.0–8.6 |
  | `overexp` | `rung2id` | 5.1 | 4.0–8.6 |

  Shrub is the slow scene on `over_exp` (Rung 2 about 8.5 min).
- **Also seen (measured):** Windows logged "WHEA-Logger event 1, fatal hardware error" at 15:15:53 (twice) and 18:10:47 on 24 Nov, and at 00:00:10 on 25 Nov. There was no crash or reboot (last boot 13 Nov), and every run active at those moments finished `ok`. There were 11 such events in 30 days, including 10 Nov, so this is not new. *Inferred:* recovered device errors, harmless to the runs.
- **Re-scoring:** not done yet (it needs Bart's OK). E1–E4 and E6 runs have training-time metrics only.
- **Next:** re-score all four roots with `rescore_runs.py --ref_root ../results/rung0`, then step 5: `rung2_analyze.py` / `rung2_ablation.py` changes and tests, the A1 replay, the analyses, `M5_results.md` and the update to Wei.

### 2026-11-25: M5 step 3 analysis code: rung2_analyze.py --rung2_variant, new rung2_ablation.py, all tests pass

- **Rung / type:** Rung 2 (M5) · code. No run or reading changed. Bart's OK, 2026-11-25.
- **Why:** M5_plan.md step 3. E1–E4 and E6 need these scripts before step 5's analysis.
- **What was done:**
  - **`rung2_analyze.py`** gained two flags.
    - `--rung2_variant {rung2, rung2id}`: with `rung2id`, the Rung 2 cells are read from `rung2id` and `rung2idexp`, and the tables name them so. Default output `results/rung2/summary_rung2id/`.
    - `--out`, behind a guard that refuses M3's rule-6 folders. So the bare default command, which writes `summary/`, is now refused.
    - In `rung2id` mode, the LOM H1 verdict is labelled exploratory, and H2's "1×–2×: run seeds 3 and 4" outcome reads "inconclusive" (M5 3.2(b)).
    - `EXPECTED` has `rung2id` and `rung2idexp` entries (F init identity).
    - The analysis logic is unchanged: the internal cell names are kept, and only the labels and run sources change.
  - **`rung2_ablation.py` (new)** has one mode per job, each following M5 3.1–3.6:
    - `h3`: help and harm signals, leak, pooled s_k slope, s drift;
    - `noprior`: every metric, plus F at grid levels 1–4;
    - `h1`: the H1 rule, official alongside, arms spanning roots by seed, `exp_name` checked;
    - `init`: the init × dark-branch 2×2, noise over the four cells.

    It stops before any table if a run is missing, not ok, has the wrong flags, data, `exp_name`, seed or steps, or was re-scored against the wrong depth. Wall times leave out `vexp/chair_rung2id_seed2` (the run that slept).
- **Tests (measured; scratch outputs in the git-ignored `results/rung2/smoke_m5/`):**
  - **Guards:** `rung2_analyze.py` refused 6 bad paths and `rung2_ablation.py` refused 4, including `summary/`, `summary_headstart*/` and `preview_final/`. No folder was created.
  - **Default `rung2_analyze.py` into `smoke_m5/analyze_default`:** all 12 CSVs are byte-identical to `results/rung2/summary/`, and `tables.md` and `numbers.json` match once the "Generated" line and `meta` are masked. Its `--selftest` gives all 84 curve-model effects and every R1 difference exactly 0.
  - **`--rung2_variant rung2id`:** the LOM H1 table equals `summary_headstart_seeds012/`'s (36 values, max |d| 0). `rung2id` and `rung2idexp` agree bit for bit in s and F, and every integrity check passes.
  - **`rung2_ablation.py --selftest`:** every effect under test is exactly 0 in all four modes (h3 189, noprior 84, h1 231, init 28 per effect).
  - **`h1` reproductions:** `rung2` − `detach` (`results/rung2`, `results/rung0`) reproduces M3's R1b exactly (140 values, including bike failing on bm PSNR). `rung2id` − `detach` reproduces the head-start H1 table exactly (36 values, 12/12).
- **Reproduce:** commit `dbdcfdb` plus uncommitted `examples/rung2_analyze.py` and the new `examples/rung2_ablation.py`; test commands in the script docstrings.
- **Re-scoring** of the 165 M5 runs started at 10:10 (`rescore_runs.py --ref_root ../results/rung0`, four roots, about 10 s per run); `vexp` done 10:16:55. A1 (`rung2_replay.py --mode long`) started at about 10:20. Results: pending.
- **Next:** step 5. Run `rung2_analyze.py --rung2_variant rung2id` and the four `rung2_ablation.py` jobs, then apply the readings, write `M5_results.md` and update Wei.

### 2026-11-25: M5 A1 (long curve replay): the identity-init gap is early stopping in every scene

- **Rung / type:** Rung 2 (M5) · experiment (CPU replay, no 3DGS). Reading pre-registered on 2026-11-24 (M5_plan.md 3.5).
- **What was done:** `rung2_replay.py --mode long`.
  - Curve-only replay on the LOM low-light training photos, both inits, 50,000 steps. The learning rate decays to 1% over the replay, so this models a 50k-step run, not the 10k run continued.
  - Seed 0, 2 threads, 698 s.
  - gap(t) = max over the 255-bin grid of |F_id(t) − F_cdf(t)|.
- **Reproduce:** commit `dbdcfdb` (`rung2_replay.py` unchanged since), `python rung2_replay.py --mode long`, from `examples/`.
- **Built-in check passes in all five scenes:** the 10k replay's final gap matches the trainer's 10k gap (seeds 0–2) within 0.01. Bike: 0.0511 against 0.0510.
- **Results (measured; source `results/rung2/replay_long/tables.md`):**

  | Scene | Gap at 1k | Gap at 10k | Gap at 40k | Gap at 50k | Step where the gap halves | Outcome |
  |---|---|---|---|---|---|---|
  | buu | 0.078 | 0.034 | 0.002 | 0.002 | 4000 | early stopping |
  | chair | 0.126 | 0.031 | 0.000 | 0.000 | 3000 | early stopping |
  | sofa | 0.095 | 0.035 | 0.001 | 0.001 | 4000 | early stopping |
  | bike | 0.177 | 0.051 | 0.001 | 0.001 | 4000 | early stopping |
  | shrub | 0.106 | 0.031 | 0.002 | 0.002 | 2000 | early stopping |

  At 50k the two inits reach the same final curve objective (to 3 decimals) in every scene.
- **Takeaway:** by the pre-registered reading, every scene's gap(50k) is below 0.01, so the outcome is "early stopping".
  - *Inferred:* the identity init's gain at 10k (H1b, E3) comes from F not yet having converged at the dark end. With enough curve training, both inits reach the same F. This is a curve-only replay, so whether a 50k-step 3DGS run would lose the gain is not measured.
  - *Inferred:* the curve loss is not flat at the dark end; it pulls both inits to one F, but slowly at the learning rate of 1e-2.
- **Next:** report it next to H1b in `M5_results.md` (step 5).

### 2026-11-25: M5 re-scoring done (165 runs, no failures)

- **Rung / type:** Rung 2 (M5) · setup. Completes the re-scoring started in the step 3 code entry above.
- **What was done:** `rescore_runs.py --root ../results/<root> --ref_root ../results/rung0` for `vexp` (45 runs), `rung2` (30), `h1b` (30) and `overexp` (60). Run by Claude with Bart's OK, from 10:10:14 to 10:34:20, about 10 s per run. Console logs: `<root>/rescore_console_m5.log`.
- **Result (measured):** every root ends with "all done", with no failed or missing reference. A dry run afterwards finds 0 runs left to re-score in all four roots (vexp 120/120, rung2 60/60, h1b 30/30, overexp 60/60).
- **Next:** step 5 analyses.

### 2026-11-25: M5 E1 (H2b): the identity init on the varying set; (a) confirmed, (b) supported

- **Rung / type:** Rung 2 (M5) · experiment. Readings pre-registered on 2026-11-24 (M5_plan.md 3.2).
- **Reproduce:** batch at commit `dbdcfdb`, analysis with the step 3 code (uncommitted at the time). From `examples/`:
  - `python rung2_ablation.py --mode init` → `results/rung2/summary_h2b/`
  - `python rung2_analyze.py --rung2_variant rung2id` → `results/rung2/summary_rung2id/`

  `rung2id` and `rung2idexp` on `results/vexp`, seeds 0–2.
- **Results (measured):**
  - **(a) Init main effect, noise over four cells:** pooled +0.61 dB bm PSNR (+5.1×), LPIPS −0.0117 (−3.1×, better); without shrub +0.76 dB (+5.2×); bike +0.28 dB (+9.7×), LPIPS +0.3×. Positive in all four decision scenes (buu +0.66, chair +0.42, sofa +1.67).
  - **(b) H2 rule with the identity cells:** curve-model main effect +2.72 dB (+6.8×) pooled, +2.99 dB (+6.0×) without shrub; LPIPS −0.0360 (−3.9×); positive in 4/4 decision scenes; exp-column simple effect +2.75 dB (+12.0×). For comparison, M3's `cdf` default gave +2.11 dB (+5.3×).
  - **(c) Mechanism:** pooled paired leak 0.009 against `detach`'s 0.512; s_k slope −1.00; s moves 0.005 stops from its initialisation.
  - **Bike hue on the varying set:** `rung2id` 27.0° ± 5.6, M3 `rung2` 35.4°, `detach` 12.3°.
  - Integrity: `rung2id` and `rung2idexp` agree bit for bit in F and s.
- **Takeaway:** by the pre-registered readings, H2b(a) is **confirmed** (pooled, without shrub, and bike) and H2b(b) is **supported**. The init gain carries over to new (gained) photos of the same scenes and is larger there than on LOM (+0.44 dB). *Inferred:* the exposure model carries H2 and the init adds on top. The bike hue gap shrinks by about a third but remains.
- **Next:** `M5_results.md`; the E5 decision (Bart).

### 2026-11-25: M5 E2 (H3, restated): attaching s_k has no effect

- **Rung / type:** Rung 2 (M5) · experiment. Readings pre-registered on 2026-11-24 (M5_plan.md 3.3).
- **Reproduce:** `python rung2_ablation.py --mode h3` → `results/rung2/summary_h3/`. `rung2att` (`--rung2_attach s`) against `rung2`, LOM and the varying set, seeds 0–2.
- **Results (measured; `rung2att` − `rung2`):**

  | Set | bm PSNR | LPIPS |
  |---|---|---|
  | LOM pooled | +0.08 dB (+1.5×) | +0.0004 (+0.2×) |
  | Varying pooled | −0.01 dB (−0.1×) | +0.0015 (+0.2×) |

  - Without shrub the result is the same.
  - LOM decision scenes' bm PSNR: buu +0.06, chair +0.06, sofa +0.24, bike +0.00.
  - Bike hue: LOM +0.23° (+0.1×), varying +0.82° (+0.6×).
  - Pooled leak 0.006 (limit 0.1); pooled s_k slope −1.003 (range −1.1 to −0.9).
  - The exposures move more: s drift 0.044 against 0.018 stops on the varying set, 0.049 against 0.012 on LOM.
  - Small real side effects: LOM pseudo-target bm PSNR −0.06 dB and angle error +0.06°.
- **Takeaway:** no help signal and no harm signal, so the verdict is **no effect**. *Inferred:* 3D feedback into s_k changes the exposures by hundredths of a stop and leaves quality unchanged, so the detached default loses nothing measurable.
- **Next:** keep `--rung2_attach none` as the default (no default changes in M5).

### 2026-11-25: M5 E3 (H1b): the identity variant meets the H1 rule on fresh seeds 3–5

- **Rung / type:** Rung 2 (M5) · experiment. Reading pre-registered on 2026-11-24 (M5_plan.md 3.1).
- **Reproduce:** from `examples/`:
  - `python rung2_ablation.py --mode h1 --a rung2id --a_roots ../results/h1b --b detach --b_roots ../results/h1b --seeds 3 4 5 --out ../results/h1b/summary`
  - Context: the same with `--seeds 0 1 2 3 4 5`, `--a_roots ../results/rung2 ../results/h1b` and `--b_roots ../results/rung0 ../results/h1b`, written to `results/h1b/summary_seeds012345/`.
- **Results (measured; `rung2id` − `detach`, seeds 3–5):**

  | Scene | bm PSNR | SSIM | LPIPS |
  |---|---|---|---|
  | buu | +0.62 (+4.7×) | +0.0046 | −0.0380 |
  | chair | +0.48 (+1.8×) | +0.0154 | −0.0221 |
  | sofa | +0.68 (+1.7×) | +0.0076 | −0.0144 |
  | bike | +0.14 (+2.6×) | +0.0270 | +0.0062 (+0.9×) |
  | pooled | +0.37 (+3.7×) | +0.0101 (+8.6×) | −0.0084 |

  - 12 of 12 checks meet.
  - Context, seeds 0–5: 12 of 12, pooled +0.39 dB (+4.3×). The two `detach` code versions are acceptable to pool because of M2's regression check.
- **Takeaway:** H1b is **supported**. Report it side by side with H1 (not supported, `cdf` init), never in its place. The init was chosen after seed 0; E1 and A1 qualify it.
- **Next:** `M5_results.md`, the "What the final report says" section.

### 2026-11-25: M5 E4 (no shape prior): −6.5 dB pooled; the deep-shadow lift disappears

- **Rung / type:** Rung 2 (M5) · experiment (ablation, reported only; M5_plan.md 3.4).
- **Reproduce:** `python rung2_ablation.py --mode noprior` → `results/rung2/summary_noprior/`. `rung2noprior` (`--rung2_prior none`) against `rung2`, LOM, seeds 0–2.
- **Results (measured; `rung2noprior` − `rung2`):**
  - **bm PSNR:** buu −9.49, chair −6.91, sofa −10.88, bike −6.66, shrub +1.59; pooled −6.47 dB (−130×).
  - **Pseudo-target bm PSNR:** −7.1 to −12.6 dB in the decision scenes. This matches `preview_final`'s expectation of 7 to 13 dB.
  - **F at levels 1–4 (×255):** near 0 without the prior (for example sofa 0.04 to 0.96, buu 0.04 to 2.8), against 47 to 129 with it.
  - **Bike hue on LOM:** −27.3° (−17.8×).
  - One parameter fewer (no γ).
- **Takeaway:** the prior is what lifts the deep shadows. This directly measures the 2026-11-16 correction. *Inferred:* bike's hue gap is tied to that lift, which fits M3's "the hue gap goes with the dark end". Dropping the prior is not a fix.
- **Next:** feeds the E5 / M6 decision.

### 2026-11-25: M5 E6 (H1-over): Rung 2 fails the H1 rule on overexposure, and `detach` trails the official code there

- **Rung / type:** Rung 2 (M5) · experiment. Reading pre-registered on 2026-11-24 (M5_plan.md 3.6).
- **Reproduce:** `python rung2_ablation.py --mode h1 --a rung2 --a_roots ../results/overexp --b detach --b_roots ../results/overexp --official_roots ../results/overexp --exp_name over_exp --extra rung2id --out ../results/overexp/summary`. Every run's `exp_name` is checked to be `over_exp`.
- **Results (measured; `rung2` − `detach`):**

  | Scene | bm PSNR | Checks |
  |---|---|---|
  | buu | −5.35 dB (−28.1×) | fails all three |
  | chair | −0.40 (−3.1×) | fails all three |
  | sofa | −0.35 (−3.3×) | fails all three |
  | bike | −0.14 (−0.9×) | meets |
  | pooled | −1.26 (−17.6×) | |

  - 3 of 12 checks meet.
  - **`detach` − official:** pooled −0.94 dB (−9.7×). Official is best on chair, sofa and bike; `detach` is best on buu.
  - **Pooled bm PSNR:** official 21.69, `detach` 20.75, `rung2` 19.48, `rung2id` 19.55.
  - **`rung2id` − `rung2` (exploratory):** +0.07 dB (+2.0×, not real).
  - **Buu:** `rung2`'s pseudo-targets are 5.65 dB below `detach`'s, with raw PSNR +1.68 dB and saturation ratio +0.36.
- **Takeaway:** H1-over is **not supported**, and matching `detach` is a weaker bar on overexposure, because the detach gain is a low-light result. *Inferred:* the one-γ shape prior is a low-light default. On overexposure it becomes a strong power curve (the preview's buu γ = 12.8) that crushes mid-tones and, applied per channel, raises saturation. Nothing was tuned on overexposure validation (plan risk 6).
- **Next:** report it next to the low-light H1 verdict, as a limitation of the one-γ prior.

### 2026-11-25: Decisions after M5: M6 builds on the identity init; E5 runs before Rung 3; overexposure is reported as a limitation

- **Rung / type:** Rung 2 / Rung 3 · decision. Bart, 2026-11-25, after Claude's review of his reasoning.
- **1. M6 (Rung 3) builds on `rung2id`** (F initialised at the identity).
  - **Reasons:**
    - The official curve also starts at the identity (`curve_estimation.md` lines 39–40: the global curve starts as `linspace(0, 1, 255)`, and the bias network has Xavier gain 0.01). So the choice has a reason that does not depend on M5's results.
    - H1b passed on fresh seeds (12/12, pre-registered), and H2b(a) is confirmed on the varying set.
    - On the `cdf` base, Rung 3 could "win" just by fixing bike's known dark-end deficit.
  - **Caveats:**
    - The init was chosen after seeing seed 0.
    - The varying set has the same scenes and ground truth, so E1 is not independent data.
    - Fresh seeds re-test only the 3DGS randomness.
    - A1 shows the gain is early stopping of F. So M6 keeps 10k steps and the same learning rates, and reports Rung 3's curve loss at 10k next to its base's.
  - **Reporting is unchanged:** H1 (`cdf`, pre-registered) stays the H1 verdict, with H1b side by side.
- **2. E5 runs before Rung 3, on the identity base.**
  - The colour-preserving variant, `rung2idlum` − `rung2id`.
  - Readings are logged before any run; design in `E5_luminance_plan.md`.
  - The comparison runs already exist (`rung2id`: LOM seeds 0–2, E1, E6b), so 45 new runs, about 3.5 h (*inferred* from M5's per-run minutes).
  - If E5 "fixes" by its pre-registered rule, M6 builds on `rung2idlum`.
  - **Cutoff:** if E5 is not launched by Fri 27 Nov, Rung 3 goes ahead on `rung2id` without it.
- **3. Overexposure is reported as a limitation.** The diagnosis is split into measured and inferred (`M5_results.md`, E6). An exploratory overexposure check (15 runs) joins E5's batch; it cannot change H1-over.
- **Correction to Claude's earlier note.** "Whether F is fitted to a luminance histogram or to per-channel values" is not an open question: `losses.py` line 56 computes the HE-CDF on the channel-mean grey. A luminance variant that uses the same grey leaves F's training untouched.
- **Measured, for E5's readings** (validation, mean over seeds 0–2, from `rescore/val_summary.json`):
  - **Bike hue error:**

    | Arm | LOM | Varying set |
    |---|---|---|
    | `detach` | 31.8° | 12.3° |
    | `rung2id` | 33.6° | 27.0° |

    The varying-set gap comes largely from `detach`'s own drop there; the cause is not established.
  - **Overexposure saturation ratio** (prediction / GT; above 1 means oversaturated):

    | Arm | buu | chair | sofa | bike |
    |---|---|---|---|---|
    | official | 0.90 | 1.39 | 1.44 | 1.15 |
    | `detach` | 1.24 | 1.55 | 1.60 | 1.34 |
    | `rung2` | 1.60 | 1.59 | 1.64 | 1.36 |
- **Next:** the E5 plan for Bart's approval before any code, and a draft update to Wei.

### 2026-11-25: E5 plan approved; readings pre-registered before any code or run

- **Rung / type:** Rung 2 (E5) · decision. Bart approved `E5_luminance_plan.md` on 2026-11-25, including its section 4 (readings and adoption rule) as written. No E5 code or run existed at this point.
- **What is tested:** `--rung2_color luminance` (variant `rung2idlum`, on the identity base):
  - Y = (R + G + B)/3, the same grey the HE-CDF target uses (`losses.py` line 56);
  - out_c = clamp(x_c · F(min(1, 2^{s_k} Y)) / max(Y, 1e-6), 0, 1).
  - The default `rgb` reproduces the current code.
  - **Built-in check:** F, s and γ must equal `rung2id`'s at the same seed (max |Δ| ≤ 1e-5).
- **Comparison:** `rung2idlum` − `rung2id`, paired by seed, seeds 0–2, on LOM, the varying set and `over_exp`. The `rung2id` arms already exist (LOM seeds 0–2, E1, E6b).
- **General rules (as in M5):**
  - noise = sqrt(mean of the arms' seed variances); an effect is real above 2× noise;
  - pooled = the five-scene mean, checked against pooled without shrub;
  - shrub never decides.
- **Primary reading:**
  - **"Fixes"** if both hold:
    - bike's hue error on the varying set falls by more than 2× noise;
    - no decision scene (buu, chair, sofa, bike) loses bm PSNR beyond 2× noise on LOM or on the varying set.
  - **"Fixes at a cost"** if the hue falls beyond 2× noise but some decision scene loses bm PSNR beyond 2× noise.
  - **"No fix"** otherwise, including a hue change within 2× noise.
- **Adoption into M6's base:** `rung2idlum` replaces `rung2id` only if the verdict is "fixes" and pooled SSIM and LPIPS are not worse beyond 2× noise on either set. Otherwise M6 builds on `rung2id`, and E5 is reported as an ablation.
- **Reported without a decision:**
  - every metric on both sets, per scene and pooled;
  - the hue error in every scene, the saturation ratio, the angle error and the pseudo-target bm PSNR;
  - the share of clipped channel values;
  - the H1 rule for `rung2idlum` − `detach` on LOM (exploratory; it changes neither H1 nor H1b).
- **Overexposure (exploratory; it cannot change H1-over, and nothing is tuned on it):**
  - "Colour accounts for part of the overexposure gap" if pooled bm PSNR of `rung2idlum` − `rung2id` rises beyond 2× noise.
  - Report the fraction of `rung2id`'s −1.20 dB pooled gap to `detach` that it closes, and the saturation ratios.
  - No direction is predicted.
- **Limits stated before results:**
  - Detection limit: 2× noise on bike's varying-set hue is about 8–11° (*inferred* from `rung2id`'s *measured* sd of 5.6°), against a 14.7° gap to `detach`.
  - `detach`'s own bike hue error falls from 31.8° on LOM to 12.3° on the varying set (*measured*; cause not established), so E5 can remove only the part caused by per-channel application.
  - Luminance scaling keeps the dark photos' colour noise, so LPIPS could get worse even if hue improves (*inferred*). The adoption rule checks this.
- **Reproduce:** branch `main`, commit `01d43c5`; plan `Luminance-GS/Implementation_guides/E5_luminance_plan.md`.
- **Next:**
  - code behind `--rung2_color` (default `rgb`), then the unit, regression and smoke tests of plan section 3;
  - ask Bart before the 45-run batch (about 3.5 h, *inferred*).

### 2026-11-25: E5 code: `--rung2_color` and variant `rung2idlum`; unit, regression and smoke tests pass

- **Rung / type:** Rung 2 (E5) · code. As planned in `E5_luminance_plan.md` section 3.
- **What was done:**
  - `rung2.py`:
    - `map_pixels(..., color="rgb")` with the luminance branch;
    - `clipped_share` (the share of channel values the pseudo-target clips at 1).
  - `simple_trainer_ours.py`:
    - `Config.rung2_color` (default `rgb`);
    - `rung2_forward` passes it on, and `projected` = (Y, Y, Y) in luminance mode;
    - refusal without `--curve_model rung2`;
    - the colour mode in the start-up and grad-check prints.
  - `diagnostics.py`: `MODEL_DEFAULTS["rung2_color"]`, and `"color"` in `rung2_state.json`.
  - `rescore_runs.py`: `MODEL_FLAGS`.
  - `run_rung0.py`: variant `rung2idlum` (also in `NOT_IN_RUNG0`).
  - `test_rung2.py`: tests 15–16.
  - `rung2_ablation.py`:
    - `--mode lum` (→ `results/rung2/summary_e5/`), with `detach` alongside as context, the F/s/γ identity check, the clipped share and the readings of the previous entry;
    - `EXPECTED_ALL` entries for `rung2idlum`, and for `rung2id` with `rung2_color: rgb` (`cfg.json` files without the key count as `rgb`).
  - Docs: `curve_estimation.md` and `CLAUDE.md`.
- **Tests (all *measured*):**
  - **Unit tests: 16/16 pass.**
    - The 14 existing tests are unchanged.
    - `rgb` equals the M3 formula bit for bit (`torch.equal`) under `none`, `s` and `all`.
    - **Grey pixels:** luminance vs `rgb` max |Δ| is 1.2e-7. That is above the plan's 1e-7, from float32 rounding of the channel mean; the test tolerance is 1e-6.
    - **Ratios:** channel ratios are kept to 1.8e-7 (relative), and the output's channel mean equals L_k(Y) to 1.2e-7.
    - **Gradients:** they follow `--rung2_attach`, and d/ds_k matches central differences (rel. 1e-5).
  - **Regression:** a 300-step `rung2id` smoke on the varying set (buu, seed 0) against M5's `results/vexp/smoke_m5/buu_rung2id_seed0`.
    - F, s, heights, γ, `exposures_by_step.csv` and every pseudo-target column are identical (max |Δ| 0).
    - The 3DGS side differs: validation PSNR by 2.0e-4 dB, render columns by up to 2.5e-3 dB, the loss log by 1.8e-5.
    - A repeat of the same new-code run differs from the first by the same order (1.1e-4 dB, 4.3e-3, 1.2e-5).
    - So the 3DGS training is not bit-reproducible run to run on this GPU, and this is not the code change. The curve, the exposures and the pseudo-targets are bit-reproducible.
  - **Smokes:** 300 steps, buu, seed 0: `rung2id` and `rung2idlum` on LOM, the varying set and `over_exp` (`smoke_e5/` under `rung2/`, `vexp/`, `overexp/`).
    - **Runs:** all ok, about 0.5 min each, and all re-scored (7 s each).
    - **The pair, on every set:**
      - F, s and γ are identical (max |Δ| 0.0);
      - the pseudo-targets differ (pseudo bm PSNR by 0.64, 0.57 and 0.32 dB);
      - `cfg.json` and `rung2_state.json` carry the right colour.
    - **`over_exp` reads `over_exp/`:** mean training input 0.915, against 0.091 on LOM.
  - **Ablation selftest:** `rung2_ablation.py --mode lum --selftest` (the luminance arm read from `rung2id`):
    - all 273 `rung2idlum` − `rung2id` effect rows are exactly 0, and the F/s check gives 0;
    - it reproduces the `over_exp` `rung2id` − `detach` gap: −1.20 dB (16.0×);
    - `rung2id`'s clipped share is 0 on LOM and the varying set, and 1.3% pooled on `over_exp` (input saturation).
  - **Refusal:** `--rung2_color luminance` without `--curve_model rung2` exits with the message.
- **Reproduce:**
  - branch `main`, commit `01d43c5`, plus uncommitted changes to the seven `examples/` files above;
  - `python test_rung2.py`;
  - `python run_rung0.py --root ../results/rung2/smoke_e5 --variants rung2id rung2idlum --scenes buu --seeds 0 --max_steps 300`, the same with `--data_root ../data/LOM_vexp --root ../results/vexp/smoke_e5`, and with `--exp_name over_exp --root ../results/overexp/smoke_e5`;
  - then `rescore_runs.py --root <smoke root> --ref_root ../results/rung0`.
- **Next:** Bart's OK for the 45-run batch (about 3.5 h, *inferred* from M5's per-run minutes), with the PC kept awake.

### 2026-11-25: E5 batch: 45 `rung2idlum` runs, all ok and re-scored

- **Rung / type:** Rung 2 (E5) · experiment (batch). Launched on Bart's OK.
- **What was done:**
  - `rung2idlum` on LOM, the varying set and `over_exp`, seeds 0–2, 15 runs each, then re-scoring of the three roots.
  - A stay-awake helper (`SetThreadExecutionState`, ES_CONTINUOUS | ES_SYSTEM_REQUIRED) watched the launcher. No power setting was changed.
- **Reproduce:** branch `main`, commit `01d43c5`, plus uncommitted changes to the seven `examples/` files of the previous entry. From `examples/`:
  - `python run_rung0.py --root ../results/rung2 --variants rung2idlum --seeds 0 1 2`
  - `python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants rung2idlum --seeds 0 1 2`
  - `python run_rung0.py --exp_name over_exp --root ../results/overexp --variants rung2idlum --seeds 0 1 2`
  - then `python rescore_runs.py --root ../results/<root> --ref_root ../results/rung0` for `rung2`, `vexp` and `overexp`.
- **Results (measured, from each root's `runs.csv`):**
  - 45 of 45 `ok` (return code 0), and all 45 re-scored (7–8 s each).
  - **Minutes per run:**
    - LOM 3.4–5.8 (mean 4.1);
    - varying 3.5–6.1 (mean 4.3);
    - `over_exp` 3.7–8.2 (mean 4.8; shrub 8.1–8.2).
  - **Timeline:** training ran 13:53–17:12 and re-scoring finished 17:18, so the batch took 3 h 25 min.
  - No run was above 15 min, so the PC did not sleep. The helper log ends "stopped 17:19:25: launcher 45628 exited; sleep request cleared".
- **Next:** `rung2_ablation.py --mode lum` and the exploratory H1 checks (next entry).

### 2026-11-25: E5 results: luminance fixes bike's hue but costs buu and sofa; not adopted, so M6 builds on `rung2id`

- **Rung / type:** Rung 2 (E5) · experiment. The readings were pre-registered on 2026-11-25, before any code or run ("E5 plan approved").
- **Reproduce:** from `examples/`:
  - `python rung2_ablation.py --mode lum` → `results/rung2/summary_e5/`.
  - `python rung2_ablation.py --mode h1 --a rung2idlum --a_roots ../results/rung2 --b detach --b_roots ../results/rung0 --official_roots ../results/rung0 --extra rung2id --title "E5, exploratory" --out ../results/rung2/summary_e5_h1`
  - The same with `--a_roots`, `--b_roots` and `--official_roots ../results/overexp --exp_name over_exp --out ../results/overexp/summary_e5`.
- **Built-in check:** F, s and γ are identical to `rung2id`'s in all 45 pairs (max |Δ| 0.0). Any difference is the colour handling alone.
- **Results (measured; `rung2idlum` − `rung2id`; ×noise; * = real):**

  | Scene | bm PSNR, LOM | bm PSNR, varying | hue err (°), LOM | hue err (°), varying |
  |---|---|---|---|---|
  | buu | −2.24 (−9.4×) * | −1.64 (−5.8×) * | −2.6 * | −3.6 * |
  | chair | +1.16 (+5.8×) * | +0.84 (+3.8×) * | +2.4 * | +2.5 * |
  | sofa | −1.10 (−1.4×) | −1.91 (−3.1×) * | −0.5 | −0.5 |
  | bike | +2.50 (+38.6×) * | +2.24 (+11.3×) * | 33.6 → 9.6 * | 27.0 → 9.3 (−4.4×) * |
  | pooled | +0.12 (+0.7×) | −0.03 (−0.3×) | −5.7 * | −4.5 * |

  - **Pooled, LOM / varying:**
    - SSIM −0.0131 (−6.7×) * / −0.0135 (−9.8×) *; without shrub still real (−2.4×, −4.8×);
    - LPIPS +0.0006 / −0.0046 (−2.2×, better);
    - raw PSNR −0.74 * / −0.83 *.
  - **Saturation ratio:** rises by +0.29 to +0.49 in every scene. LOM, `rung2id` → `rung2idlum`: chair 0.47 → 0.76, bike 0.52 → 0.88, buu 0.87 → 1.26, sofa 0.86 → 1.34.
  - **Clipped share** (LOM): buu 8.3%, shrub 12.4%, sofa 0.01%.
- **Readings:**
  - **Primary: "fixes at a cost".** Bike's hue falls beyond 2×, but buu (both sets) and sofa (varying) lose bm PSNR beyond 2×.
  - **Adoption: not adopted.** The verdict is not "fixes", and pooled SSIM is worse on both sets. M6 builds on `rung2id`; E5 is reported as an ablation.
  - **H1 rule for `rung2idlum` − `detach` on LOM (exploratory):** 11 of 12 checks meet; buu bm PSNR fails (−1.72, −6.6×). Pooled +0.54 dB (+3.2×).
  - **Overexposure (exploratory):** pooled −1.56 dB (−19.2×; without shrub −1.34), so not "colour accounts for part of the gap". It widens `rung2id`'s −1.20 dB gap to `detach` to −2.75 (fraction closed −1.30). Saturation ratio: `rung2idlum` 0.44, `rung2id` 1.37, `detach` 1.31. H1 rule against `detach`: 1 of 12.
- **Takeaway:**
  - *Measured, and isolated by the built-in check:* applying F per channel causes bike's hue error, since luminance mode removes it, below even `detach`'s 31.8° and 12.3°.
  - *Inferred:* luminance mode keeps the photo's own saturation. That is right only where the input's saturation matches the GT (chair, bike), and too much for buu and sofa. Sofa barely clips, so clipping does not explain sofa's loss. Per-channel F desaturates every scene instead, so neither handling is right everywhere.
  - On overexposure, per-channel F's re-saturation of the washed-out photos was helping.
- **Next:** `E5_results.md`; `M6_rung3_plan.md` on `rung2id`.
