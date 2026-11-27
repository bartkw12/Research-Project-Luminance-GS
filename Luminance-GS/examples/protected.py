"""CLAUDE.md rule 6 in code: the finished output folders that no script may write into (added 2026-11-26, M6 step 1).

Stdlib only. The analysis and replay scripts import it (rung2_analyze, rung2_ablation, rung2_replay, rung2_preview,
rung2_headstart); the trainer never does. Keep FINISHED, CLAUDE.md rule 6 and the `ask` list in .claude/settings.json
in step. The older scripts rung0_analyze, rung05_analyze and vexp_analyze do not use it.
"""

from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
FINISHED = [RESULTS / p for p in (
    # Rungs 0 and 0.5, M1
    "rung0/summary", "rung0/summary_rung05", "vexp/summary",
    # M3 (since 2026-11-24)
    "rung2/summary", "rung2/summary_seed0", "rung2/summary_headstart", "rung2/summary_headstart_seeds012",
    "rung2/preview_final",
    # the 2026-11-11 previews behind the shape-prior decision (since 2026-11-26)
    "rung2/preview", "rung2/preview_lrF1e-2",
    # M5 and E5 (since 2026-11-26)
    "rung2/summary_h3", "rung2/summary_noprior", "rung2/summary_h2b", "rung2/summary_rung2id", "rung2/summary_e5",
    "rung2/summary_e5_h1", "rung2/replay_long",
    "h1b/summary", "h1b/summary_seeds012345",
    "overexp/summary", "overexp/summary_e5", "overexp/preview",
    # M6: the preview holding plan 4.3's pre-registered predictions (logged 2026-11-27 with its sha256)
    "rung3/preview",
)]


def is_finished(out: Path) -> bool:
    """True if out is a finished folder or lies inside one."""
    out = Path(out).resolve()
    return any(out == p.resolve() or p.resolve() in out.parents for p in FINISHED)


def refuse_finished(out: Path, tag: str) -> Path:
    """The resolved out, or SystemExit (before anything is written) if it is a finished folder or inside one."""
    out = Path(out).resolve()
    if is_finished(out):
        raise SystemExit(f"[{tag}] refusing to write to {out}: a finished folder (CLAUDE.md rule 6; protected.py)")
    return out


if __name__ == "__main__":
    for p in FINISHED:
        print(f"{'exists ' if p.exists() else 'MISSING'}  {p.relative_to(RESULTS.parent)}")
