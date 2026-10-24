---
name: run-batch
description: Pre-flight checks, launch and summary for a multi-run training batch (run_rung0.py or similar) on Bart's Windows GPU machine. Use only when Bart asks to run a batch.
disable-model-invocation: true
argument-hint: "[scenes, variants, seeds, root; e.g. --variants frozenbasis detachcurve]"
---

# Run a training batch

Batch requested: $ARGUMENTS

Batches take hours on Bart's one GPU, and a mistake costs a night. Go through every step and stop at each **ASK**.

## 1. Pre-flight (read-only, a few minutes)

1. **Code state:** `git branch --show-current`, `git rev-parse --short HEAD`, `git status --porcelain`. If the code the batch runs is uncommitted, tell Bart and suggest he commits it first, so the log can name a commit.
2. **Data:** every requested scene exists under `Luminance-GS/data/LOM_full/<scene>/`, with the input folder named by `--exp_name` (`low/` or `over_exp/`), `high/`, `colmap_sparse/` and `transforms_train.json` / `transforms_test.json`.
3. **GPU free:** `nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv` and `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`. If another training job is running, stop and tell Bart.
4. **Disk:** check there is room for about 1 GB per run (checkpoints and renders are local even though they are git-ignored): `Get-PSDrive I` in PowerShell, `df -h .` in Git Bash.
5. **Flags:** every variant's flags exist in `Config` in `simple_trainer_ours.py`, and their defaults reproduce the official code. Then run the runner with `--dry_run` (from `Luminance-GS/examples`) and check the job list: scenes × variants × seeds, root folder, and which runs it skips as already done.
6. **Smoke test:** for each new flag set, confirm a smoke run of about 300 steps finished in a `smoke*` folder under the root. If none exists, run one now (it takes a few minutes) with the `luminancegs` env's Python (path in `CLAUDE.local.md`; plain `python` has no torch): `--max_steps 300 --eval_steps 300 --save_steps 300 --diag_steps 1 300`, plus `--grad_check` if the flags change gradient flow.
7. **Time estimate:** use the mean minutes per variant from existing `runs.csv` files (or the smoke-test speed, scaled to 10k steps). Total = number of runs × minutes per run, plus re-scoring or analysis if part of the batch. State the expected finish time in local time.

## 2. ASK Bart before launching

Send one message with:

- the job list (count, scenes, variants, seeds, root), commit and estimated finish time;
- a reminder to stop the PC from sleeping, in a normal Command Prompt or PowerShell (not Git Bash, which mangles the `/` options):
  - check the current timeout first: `powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE`
  - disable sleep on AC power: `powercfg /change standby-timeout-ac 0`
  - restore it afterwards, for example `powercfg /change standby-timeout-ac 30`;
- how to launch, recommending option (a):
  - **(a) Bart starts it** in an Anaconda Prompt with the `luminancegs` env active, from `Luminance-GS\examples`. Give the exact command, with output redirected to `<root>\batch_console_<name>.log`. It keeps running if Claude Code is closed.
  - **(b) Claude starts it** as a background process with the same redirection, using the env's Python from `CLAUDE.local.md` (the runner starts each training run with its own interpreter). Warn that closing this Claude Code session can stop it.

Wait for an explicit go-ahead. Do not change power settings yourself.

## 3. While it runs

- Check progress only when Bart asks: the tail of the console log, `<root>/runs.csv`, and the latest `<root>/logs/<run>.log`. Report runs done, failures, and the updated finish estimate.
- The runner is resumable. After a crash or reboot, re-running the same command skips finished runs and restarts the incomplete one.
- Do not edit code the batch imports while it is running.

## 4. When it finishes

1. Check that every run has `stats/val_step9999.json` and `runs.csv` shows `ok` True. List any failures with the last lines of their logs.
2. Run the rung's re-scoring and analysis scripts if the plan calls for them, for example `python rescore_runs.py` and then `python rung05_analyze.py`. Ask first if they take more than about 30 minutes.
3. Summarise, from the repository root, with `python .claude/skills/log-experiment/summarize_runs.py <root> --variants ...`.
4. Write a research-log entry with `/log-experiment`.
5. Remind Bart to restore the sleep setting.
6. Do not commit. List the new files that should be committed (`cfg.json`, `stats/`, `diagnostics/*.csv`, `rescore/*.csv` and `*.json`, summary folders) and propose a commit message.
