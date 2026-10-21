"""Rung 0 batch runner: baseline vs detached pseudo-target, several seeds, all LOM scenes.

Resumable: a run is skipped when its final eval file (stats/val_step{max_steps-1}.json)
exists; an incomplete run folder is deleted and the run restarted. Each run's console
output goes to <root>/logs/<run>.log and a status line is appended to <root>/runs.csv.

Run from Luminance-GS/examples:
    python run_rung0.py                       # the full Rung 0 batch (30 runs)
    python run_rung0.py --dry_run             # list what would run
    python run_rung0.py --scenes buu --seeds 0 --max_steps 50 --root ../results/rung0_test
"""

import argparse
import csv
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent
DATA = EXAMPLES.parent / "data" / "LOM_full"

ALL_SCENES = ["buu", "chair", "sofa", "bike", "shrub"]
VARIANTS = {"baseline": [], "detach": ["--detach_pseudo_target"]}


def run_name(scene: str, variant: str, seed: int) -> str:
    return f"{scene}_{variant}_seed{seed}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=ALL_SCENES)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max_steps", type=int, default=10_000)
    ap.add_argument("--root", type=Path, default=EXAMPLES.parent / "results" / "rung0")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    scenes = [s for s in args.scenes if (DATA / s).is_dir()]
    missing = sorted(set(args.scenes) - set(scenes))
    if missing:
        print(f"[run_rung0] scenes not found under {DATA}, skipped: {missing}")

    root = args.root.resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    status_path = root / "runs.csv"
    new_status = not status_path.exists()

    # seed-major order: after each seed block every scene/variant pair has one run
    jobs = [(scene, variant, seed) for seed in args.seeds for scene in scenes for variant in args.variants]
    done_file = f"val_step{args.max_steps - 1:04d}.json"
    todo = [j for j in jobs if not (root / run_name(*j) / "stats" / done_file).exists()]
    print(f"[run_rung0] {len(jobs)} runs, {len(jobs) - len(todo)} already done, {len(todo)} to go -> {root}")

    for k, (scene, variant, seed) in enumerate(todo, 1):
        name = run_name(scene, variant, seed)
        run_dir = root / name
        cmd = [
            sys.executable, "simple_trainer_ours.py",
            "--data_dir", str(DATA / scene), "--exp_name", "low", "--result_dir", str(run_dir),
            "--seed", str(seed), "--max_steps", str(args.max_steps), "--diagnostics",
        ] + VARIANTS[variant]
        if args.max_steps != 10_000:
            cmd += ["--eval_steps", str(args.max_steps), "--save_steps", str(args.max_steps),
                    "--diag_steps", "1", str(args.max_steps // 2), str(args.max_steps)]
        print(f"[run_rung0] ({k}/{len(todo)}) {datetime.now():%H:%M:%S} {name}", flush=True)
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)  # incomplete earlier attempt

        tic = time.time()
        with open(root / "logs" / f"{name}.log", "w", encoding="utf-8", errors="replace") as log:
            ret = subprocess.call(cmd, cwd=EXAMPLES, stdout=log, stderr=subprocess.STDOUT)
        minutes = (time.time() - tic) / 60
        ok = ret == 0 and (run_dir / "stats" / done_file).exists()
        print(f"[run_rung0]     {'done' if ok else 'FAILED (return code %d)' % ret} in {minutes:.1f} min", flush=True)

        with open(status_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_status:
                w.writerow(["finished", "run", "scene", "variant", "seed", "ok", "return_code", "minutes"])
                new_status = False
            w.writerow([f"{datetime.now():%Y-%m-%d %H:%M:%S}", name, scene, variant, seed, ok, ret, f"{minutes:.2f}"])

    print("[run_rung0] all done")


if __name__ == "__main__":
    main()
