# Luminance-GS extension: ECE 701 research project

Bart's ECE 701 project (MEng Computer Engineering, McMaster), supervised by Prof. Chen with PhD mentor Wei. **Experiments freeze on 5 Dec 2026**; the written report follows.

## Goal

Luminance-GS (Cui, Chu, Harada, CVPR 2025, arXiv 2504.01503) builds a normally lit 3D Gaussian Splatting scene from low-light photos with no normal-light supervision. A learned per-view tone curve brightens each dark photo into a pseudo-target, and the Gaussians learn to render both. Our research question: does that curve machinery (about 255k parameters) reduce to **one shared monotone curve plus one exposure value per view**, `L_k(x) = F(min(1, 2^{s_k} x))`, and does that explicit model match the original with far fewer parameters, and beat it when exposure varies across views?

- **H1:** on LOM, Rung 2 matches the baseline within seed noise.
- **H2:** on a varying-exposure LOM set, Rung 2 wins and the learned `s_k` track the true gains.
- **H3:** a fixed RGB colour basis removes the hue flips and explains most of the "detach" gain. **Not supported** (Rung 0.5): the attached curve, not the colour basis, explains the detach gain and the bike hue flip.

## Rung ladder

| Rung | What | Status |
|---|---|---|
| 0 | Diagnostics of the official curve (30 runs) | **Done** (2026-10-21). Curves already monotone; cross-view variation is one number that tracks brightness; detach gain is confounded with a frozen colour basis. |
| 0.5 | 2×2: colour basis learned/frozen × curve attached/detached, plus shared-depth re-scoring and colour metrics | **Done** (2026-10-23, merged into `rung0-diagnostics`). Detaching the curve explains 85 to 100% of the detach gain where it is real; with a detached curve a fixed RGB basis costs nothing. |
| 1 | Compact monotone curve (K = 32 knots dense near black, F(0) = 0) | Folded into Rung 2 as the parameterisation of F. |
| 2 | Shared monotone curve + one exposure per view, plus a varying-exposure LOM test set | **Core, next.** Detached curve, fixed RGB basis. Reference baseline is the `detach` cell; `baseline` stays in every table as "official". |
| 3 | Local monotone curves (per-pixel mix of shared curves) | Optional; decided at the Nov 13 check-in with Wei. |

Update this table whenever a rung changes status.

## Repo map

- `Luminance-GS/examples/` holds everything we run. Run commands from this folder.
  - `simple_trainer_ours.py`: the trainer. `Config` (all flags), `Runner.curve_forward` (curve and pseudo-target, shared with diagnostics), loss in `train()`, `eval()`.
  - `losses.py`: `HistogramPriorLoss`, `L_spa`, `gamma_curve`, `s_curve`. `tools.py`: `pixel_project`, `LUT_mapping`, `pixel_project_back`. `utils.py`: the two `CrossAttention` networks.
  - `datasets/colmap.py`: LOM loader (train reads `low/`, validation reads `high/`).
  - `diagnostics.py`: Rung 0/0.5 measurements. `run_rung0.py`: resumable batch runner. `rescore_runs.py`: re-score finished runs. `rung0_analyze.py`, `rung05_analyze.py`: tables and figures.
- `Luminance-GS/gsplat/`: gsplat 1.0.0 plus the authors' dual rasteriser. Do not modify.
- `Luminance-GS/data/LOM_full/<scene>/`: dataset, local only (git-ignored).
- `Luminance-GS/results/`: `reproduction/` (seed-42 reproduction), `rung0/` (all Rung 0 and 0.5 runs, `summary/`, `summary_rung05/`).
- `Luminance-GS/Implementation_guides/`: rung plans (`Rung*_diagnostics_plan.md`) and reports (`Rung*_results.md`).
- `docs/`: research log and reference docs (index below).

## Environment and how to run

- Windows, one NVIDIA GPU, conda env `luminancegs` (gsplat 1.0.0, numpy < 2). Open this folder (the repository root) in VS Code or start `claude` here: Claude Code loads `.claude/settings.json` only from the folder it starts in. Scripts Bart runs must be Python with `pathlib` and an `if __name__ == "__main__":` guard.
- The shell tool may be PowerShell or Git Bash. Use forward slashes in paths, and `;` rather than `&&` in PowerShell. Training and analysis scripts need the `luminancegs` env's Python (path in `CLAUDE.local.md`); `run_rung0.py` starts the trainer with its own interpreter (`sys.executable`). The Stop hook and `summarize_runs.py` are stdlib-only and run with any `python`.
- One run: `python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name low --result_dir ../results/<rung>/<scene>_<variant>_seed<seed> --seed 0`. `--exp_name` is `low`, `over_exp` or `variance` (MipNeRF360-varying, needs `--data_factor 8`).
- Batch: `python run_rung0.py --variants baseline detach --seeds 0 1 2 --dry_run`, then without `--dry_run`. It is resumable, logs to `<root>/logs/` and records each run in `<root>/runs.csv`.
- Our flags (every default reproduces the official code): `--seed`, `--detach_pseudo_target`, `--diagnostics`, `--diag_steps`, `--freeze_color_basis`, `--detach_curve_in_target`, `--grad_check`, `--rescore_run`, `--ref_depth_run`.
- Cost: one 10k-step run takes about 10 to 14 minutes (detached runs are faster), so one configuration on 5 scenes × 3 seeds takes about 3 hours.

## Rules

1. **Never run `git commit` or `git push`, and never merge, rebase, reset or switch branches.** Bart commits himself. When a piece of work is done, list the changed files and propose a commit message. Ask before creating a branch. This overrides the older plan files that say "commit".
2. **Ask before any job longer than about 30 minutes**, with a time estimate. Before anything over an hour, remind Bart to stop the PC from sleeping (Rung 0 lost 11 hours to sleep). For batches, follow `.claude/skills/run-batch/SKILL.md` (Bart starts it with `/run-batch`).
3. **Keep the baseline reproducible.** New behaviour goes behind a new `Config` flag whose default reproduces the current code exactly. Smoke-test (about 300 steps) before any batch.
4. **Log as you go.** After every experiment, result, decision, or code change that affects runs, append an entry to `docs/RESEARCH_LOG.md` (use `/log-experiment`). Never edit or delete old entries; add a correcting entry instead.
5. **Report honestly.** Label *measured* vs *inferred*. With 3 seeds, call an effect real only if it exceeds about 2× seed noise. Report raw and brightness-matched PSNR side by side. Never guess a number; write "pending".
6. **Leave finished results alone** (`results/rung0/summary/`, `summary_rung05/`, `Rung0_results.md`, `Rung0.5_results.md`). New outputs go to new folders. `.claude/settings.json` makes Claude's own edits there ask first, but a script can still overwrite them.
7. **Diagnostics must not change training.** Keep `rng_guard()`, `eval_mode()` and `trainset[i]` (no DataLoader), as `diagnostics.py` does.
8. **If a doc disagrees with the code, trust the code** and record the discrepancy in the log.

## Docs index

Read these only when the task needs them. Do not load them all up front.

| File | Read it when |
|---|---|
| `docs/RESEARCH_LOG.md` | Starting a task (read the last 2 or 3 entries) or recording one (append at the end). |
| `docs/curve_estimation.md` | Touching the curve, pseudo-target, colour basis, losses, gradient flow or optimisers. |
| `docs/repo_vs_paper.md` | The paper and the code seem to disagree, or you are writing the method or baseline description. |
| `docs/experiment_protocol.md` | Planning, running or comparing experiments: scenes, seeds, metrics, noise floors, naming, what to commit. |
| `docs/Luminance_GS_Research_Project_Explained.md` | Long explainer for humans: background, Rung 2 design and maths, likely questions from Wei. Use only when the focused docs are not enough. |
| `Luminance-GS/Implementation_guides/Rung0_results.md`, `Rung0.5_results.md` | You need exact Rung 0 or 0.5 numbers. |
| `Luminance-GS/Implementation_guides/Rung*_diagnostics_plan.md` | Historical task plans. |

## Skills

- `/log-experiment [what to log]`: writes a research-log entry from the run outputs.
- `/run-batch [scenes, variants, seeds]`: pre-flight checks, launch and summary for a training batch. Manual only: Claude never starts it on its own.
