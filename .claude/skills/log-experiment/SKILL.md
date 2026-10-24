---
name: log-experiment
description: Append an entry to docs/RESEARCH_LOG.md for an experiment, batch, analysis, code change that affects runs, or research decision, with reproducibility details and numbers read from the run outputs. Use after any of these, or when Bart asks to log something.
argument-hint: "[what to log: run folders, a results root, or a short description]"
---

# Log an experiment

Write one research-log entry for: $ARGUMENTS

If no argument was given, log the most recent piece of work in this session. The log feeds Bart's final report, so accuracy beats completeness: every number must come from a file you read in this session.

## 1. Gather the facts

1. Run these from the repository root:
   - `date +%F`
   - `git branch --show-current`
   - `git rev-parse --short HEAD`
   - `git status --porcelain -- Luminance-GS/examples`

   If code under `examples/` is uncommitted, say so in the Reproduce line ("commit `abc1234` plus uncommitted changes to ...").
2. Read the last two or three entries of `docs/RESEARCH_LOG.md`. Don't duplicate them. If an earlier entry says "in progress" or "pending", write a **new** entry with the results and refer to the earlier one by date. Never edit an old entry.
3. For runs, summarise them with the helper script (stdlib only, read-only):

   ```
   python .claude/skills/log-experiment/summarize_runs.py Luminance-GS/results/<root> [--variants ...] [--scenes ...] [--reference baseline]
   ```

   It reads each run's `stats/val_step*.json` and `rescore/val_summary.json` (if present) and `<root>/runs.csv`. It prints mean ± sample std per scene × variant, differences paired by seed against the reference variant (with ×noise), wall times with sleep outliers excluded, and any incomplete runs.
4. Read the flags that differ from the defaults in one run's `cfg.json` per variant, and the exact command if it is known (`run_rung0.py` builds it from `VARIANTS`).
5. If an analysis summary exists (`summary*/tables.md`, `numbers.json`) or a `Rung*_results.md` report, prefer its numbers and cite it by path.

## 2. Write the entry

Append it at the **end** of `docs/RESEARCH_LOG.md`, in the template format at the top of that file:

```markdown
### YYYY-MM-DD: <short title>

- **Rung / type:** Rung N · experiment | code | decision | setup
- **Why:** ...
- **What was done:** ...
- **Reproduce:** branch `...`, commit `...`, command(s), scenes, seeds, steps.
- **Results:** ... (source: `path`)
- **Takeaway:** ...
- **Next:** ...
```

- **Results:** a compact table (scenes as rows) with raw PSNR, brightness-matched PSNR when available, SSIM and LPIPS, then a source path. Put only headline numbers here; detail belongs in the rung report.
- **Takeaway:** call an effect real only above about 2× seed noise (see `docs/experiment_protocol.md`); otherwise write "no detectable difference". Mark interpretations *inferred*.
- Write "pending" for anything you could not read. Never estimate a number.
- Keep the entry under about 30 lines.

## 3. Follow through

- If this work starts or finishes a rung, update that rung's line in the "Rung ladder" table in `CLAUDE.md`.
- **Do not commit.** Show Bart the new entry, list the files you changed, and propose a commit message such as `Log: Rung 0.5 batch results`.
