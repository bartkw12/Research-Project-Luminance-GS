"""Stop hook: remind Claude to add a research-log entry when experiment code or results changed.

Claude Code runs this when Claude is about to finish a turn. If files under the watched folders
have uncommitted changes and docs/RESEARCH_LOG.md has none, it asks Claude (once) to log the work
or say why no entry is needed. It stays quiet when the stop was already blocked once
(stop_hook_active), and when it already reminded Claude about exactly the same set of changed
paths (remembered in .claude/log_reminder_state, which is git-ignored). Stdlib only.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

WATCHED = ("Luminance-GS/examples/", "Luminance-GS/results/", "Luminance-GS/Implementation_guides/")
LOG = "docs/RESEARCH_LOG.md"
MAX_LISTED = 8


def git(args, cwd):
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def group(path: str) -> str:
    # one entry per results folder, so a batch writing new run folders doesn't count as new changes
    parts = path.rstrip("/").split("/")
    return "/".join(parts[:3]) + "/" if path.startswith("Luminance-GS/results/") else path


def main():
    try:
        event = json.loads(sys.stdin.buffer.read().decode("utf-8-sig"))  # tolerate a BOM from PowerShell pipes
    except ValueError:
        event = {}
    if event.get("stop_hook_active"):
        return

    top = git(["rev-parse", "--show-toplevel"], os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or ".")
    status = top and git(["status", "--porcelain", "--untracked-files=all"], top.strip())
    if not status:
        return
    root = Path(top.strip())
    # porcelain v1: "XY path" or "XY old -> new"
    paths = [line[3:].split(" -> ")[-1].strip('"') for line in status.splitlines()]
    if LOG in paths:
        return
    changed = sorted({group(p) for p in paths if p.startswith(WATCHED)})
    if not changed:
        return

    key = hashlib.sha1("\n".join(changed).encode()).hexdigest()
    state = root / ".claude" / "log_reminder_state"
    try:
        if state.read_text().strip() == key:
            return
    except OSError:
        pass
    try:
        state.write_text(key)
    except OSError:
        pass

    listed = ", ".join(changed[:MAX_LISTED]) + (f" and {len(changed) - MAX_LISTED} more" if len(changed) > MAX_LISTED else "")
    reason = (
        f"Experiment code or results have uncommitted changes ({listed}) and {LOG} has no new entry. "
        "If this work produced a result, a decision, or a change that affects runs, add an entry with "
        "/log-experiment before finishing. If it doesn't need one (a trivial edit, or a batch that is "
        "still running), say so in one line and finish."
    )
    print(json.dumps({"decision": "block", "reason": reason}))


if __name__ == "__main__":
    main()
