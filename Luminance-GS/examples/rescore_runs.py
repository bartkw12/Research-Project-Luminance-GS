"""Rung 0.5 re-scoring batch: score every finished run under results/rung0/ without training.

For each <scene>_<variant>_seed<seed> run with a final checkpoint and diagnostics/curve_state.pt,
runs simple_trainer_ours.py --rescore_run into <run>/rescore/, with <scene>_baseline_seed0 as the
shared reference depth for the consistency warp. Resumable: a run is skipped when
rescore/val_summary.json exists; an incomplete rescore/ folder is deleted and redone.
Console output goes to <root>/logs/rescore_<run>.log.

Run from Luminance-GS/examples:
    python rescore_runs.py                    # all runs
    python rescore_runs.py --dry_run          # list what would run
    python rescore_runs.py --runs bike_baseline_seed0 bike_detach_seed0
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent
RUN_RE = re.compile(r"^(?P<scene>[a-z]+)_(?P<variant>[a-z]+)_seed(?P<seed>\d+)$")
REF_VARIANT, REF_SEED = "baseline", 0


def final_ckpt(run_dir: Path) -> Path:
    cfg = json.loads((run_dir / "cfg.json").read_text())
    return run_dir / "ckpts" / f"ckpt_{cfg['max_steps'] - 1}.pt"


def is_finished(run_dir: Path) -> bool:
    return ((run_dir / "cfg.json").exists() and final_ckpt(run_dir).exists()
            and (run_dir / "diagnostics" / "curve_state.pt").exists())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=EXAMPLES.parent / "results" / "rung0")
    ap.add_argument("--runs", nargs="+", default=None, help="only these run names")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = args.root.resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    runs = sorted(d for d in root.iterdir() if d.is_dir() and RUN_RE.match(d.name) and is_finished(d))
    if args.runs:
        runs = [d for d in runs if d.name in set(args.runs)]
    todo = [d for d in runs if not (d / "rescore" / "val_summary.json").exists()]
    print(f"[rescore_runs] {len(runs)} finished runs, {len(runs) - len(todo)} already re-scored, {len(todo)} to go -> {root}")

    failed = []
    for k, run_dir in enumerate(todo, 1):
        scene = RUN_RE.match(run_dir.name)["scene"]
        ref = root / f"{scene}_{REF_VARIANT}_seed{REF_SEED}"
        if not is_finished(ref):
            print(f"[rescore_runs] ({k}/{len(todo)}) {run_dir.name}: reference run {ref.name} missing, skipped")
            failed.append(run_dir.name)
            continue
        cfg = json.loads((run_dir / "cfg.json").read_text())
        out = run_dir / "rescore"
        cmd = [
            sys.executable, "simple_trainer_ours.py",
            "--data_dir", cfg["data_dir"], "--exp_name", cfg["exp_name"],
            "--data_factor", str(cfg["data_factor"]), "--test_every", str(cfg["test_every"]),
            "--result_dir", str(out), "--rescore_run", str(run_dir), "--ref_depth_run", str(ref),
        ]
        print(f"[rescore_runs] ({k}/{len(todo)}) {datetime.now():%H:%M:%S} {run_dir.name} (ref {ref.name})", flush=True)
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue
        if out.exists():
            shutil.rmtree(out)  # incomplete earlier attempt

        tic = time.time()
        with open(root / "logs" / f"rescore_{run_dir.name}.log", "w", encoding="utf-8", errors="replace") as log:
            ret = subprocess.call(cmd, cwd=EXAMPLES, stdout=log, stderr=subprocess.STDOUT)
        ok = ret == 0 and (out / "val_summary.json").exists()
        if not ok:
            failed.append(run_dir.name)
        print(f"[rescore_runs]     {'done' if ok else 'FAILED (return code %d)' % ret} in {time.time() - tic:.0f} s", flush=True)

    print(f"[rescore_runs] all done{'; failed: ' + ', '.join(failed) if failed else ''}")


if __name__ == "__main__":
    main()
