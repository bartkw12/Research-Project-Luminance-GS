"""Rung 0 batch runner: baseline vs detached pseudo-target, several seeds, all LOM scenes.

Resumable: a run is skipped when its final eval file (stats/val_step{max_steps-1}.json)
exists; an incomplete run folder is deleted and the run restarted. Each run's console
output goes to <root>/logs/<run>.log and a status line is appended to <root>/runs.csv.

Run from Luminance-GS/examples:
    python run_rung0.py                       # the full Rung 0 batch (30 runs)
    python run_rung0.py --dry_run             # list what would run
    python run_rung0.py --scenes buu --seeds 0 --max_steps 50 --root ../results/rung0_test
    python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants baseline detach
                                              # M1: the same runs on the varying-exposure set
    python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants detachexp
                                              # M2: detach with the exposure-aware dark branch
    python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants rung2 rung2exp
    python run_rung0.py --root ../results/rung2 --variants rung2
                                              # M3: Rung 2 on the varying set and on LOM (never in results/rung0)
    python run_rung0.py --root ../results/rung2 --variants rung2id --seeds 0
                                              # M3 conditional (plan 5.3): Rung 2 with F initialised at the identity
    python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants rung2id rung2idexp rung2att
    python run_rung0.py --root ../results/h1b --variants detach rung2id --seeds 3 4 5
    python run_rung0.py --exp_name over_exp --root ../results/overexp --variants baseline detach rung2
                                              # M5 (Implementation_guides/M5_plan.md): E1/E2, E3 and E6
    python run_rung0.py --root ../results/rung2 --variants rung2idlum
                                              # E5 (E5_luminance_plan.md); also on the varying set and over_exp
    python run_rung0.py --root ../results/rung3 --variants rung3
                                              # M6 (M6_rung3_plan.md): Rung 3 on LOM; the varying set goes to
                                              # results/vexp, over_exp to results/overexp
--exp_name over_exp trains on <scene>/over_exp/ and is accepted only with a root inside results/overexp/ (run names
do not include the exposure type, so another root would skip the runs as already done); --exp_name low is refused
inside results/overexp/.
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
OVEREXP = EXAMPLES.parent / "results" / "overexp"  # M5 E6: the only root for --exp_name over_exp
EXP_NAMES = ["low", "over_exp"]

ALL_SCENES = ["buu", "chair", "sofa", "bike", "shrub"]
VARIANTS = {
    "baseline": [],
    "detach": ["--detach_pseudo_target"],
    # Rung 0.5 factorial: each new cell cuts one of the two gradient paths that "detach" cuts
    "frozenbasis": ["--freeze_color_basis"],
    "detachcurve": ["--detach_curve_in_target"],
    # M2: "exp" = exposure-aware dark branch (view k's dark render scaled by 2^{-s_k} before loss_regress_low)
    "detachexp": ["--detach_pseudo_target", "--dark_branch", "measured"],
    # M2 part C: Rung 2 (one shared monotone curve F + one exposure s_k per view; rung2.py). rung2exp's dark branch
    # uses Rung 2's own s_k (stop-gradient); rung2att sends the enhancement loss into s_k only (H3, run in M5)
    "rung2": ["--curve_model", "rung2"],
    "rung2exp": ["--curve_model", "rung2", "--dark_branch", "model"],
    "rung2att": ["--curve_model", "rung2", "--rung2_attach", "s"],
    # M3 conditional run (M2_rung2_plan.md 5.3, risk 5): head-start check, F starts at the identity as detach's curve does
    "rung2id": ["--curve_model", "rung2", "--rung2_F_init", "identity"],
    # M5 (M5_plan.md section 2): E1, the identity init with the exposure-aware dark branch (varying set); E4, Rung 2
    # without the shape prior (LOM)
    "rung2idexp": ["--curve_model", "rung2", "--rung2_F_init", "identity", "--dark_branch", "model"],
    "rung2noprior": ["--curve_model", "rung2", "--rung2_prior", "none"],
    # E5 (E5_luminance_plan.md): the identity init with F applied to the channel-mean grey, RGB scaled by the ratio
    "rung2idlum": ["--curve_model", "rung2", "--rung2_F_init", "identity", "--rung2_color", "luminance"],
    # M6 (M6_rung3_plan.md, pre-registered 2026-11-26): Rung 3, two shared curves mixed by fixed brightness-zone weights,
    # on the identity base; the guide's blur keeps its default (16)
    "rung3": ["--curve_model", "rung2", "--rung2_F_init", "identity", "--rung3_zones", "2"],
}
# results/rung0 holds Rungs 0 and 0.5 only (CLAUDE.md rule 6): these variants must go to another root
NOT_IN_RUNG0 = {"rung2", "rung2exp", "rung2att", "rung2id", "rung2idexp", "rung2noprior", "rung2idlum", "rung3"}


def run_name(scene: str, variant: str, seed: int) -> str:
    return f"{scene}_{variant}_seed{seed}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", nargs="+", default=ALL_SCENES)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--max_steps", type=int, default=10_000)
    ap.add_argument("--root", type=Path, default=EXAMPLES.parent / "results" / "rung0")
    ap.add_argument("--data_root", type=Path, default=DATA,
                    help="folder holding <scene>/ in the LOM layout (default: data/LOM_full)")
    ap.add_argument("--exp_name", default="low", choices=EXP_NAMES,
                    help="training input folder <scene>/<exp_name>/; over_exp only with a root inside results/overexp/")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    data = args.data_root.resolve()
    scenes = [s for s in args.scenes if (data / s).is_dir()]
    missing = sorted(set(args.scenes) - set(scenes))
    if missing:
        print(f"[run_rung0] scenes not found under {data}, skipped: {missing}")

    root = args.root.resolve()
    # never put runs on another dataset into the protected Rung 0 root
    if data != DATA.resolve() and root == (EXAMPLES.parent / "results" / "rung0").resolve():
        raise SystemExit(f"[run_rung0] --data_root {data} is not LOM_full: pass a --root other than results/rung0")
    if NOT_IN_RUNG0 & set(args.variants) and root == (EXAMPLES.parent / "results" / "rung0").resolve():
        raise SystemExit(f"[run_rung0] {sorted(NOT_IN_RUNG0 & set(args.variants))} do not belong in results/rung0: "
                         f"pass --root ../results/rung2 or ../results/rung3 (LOM), or ../results/vexp (varying set)")
    # M5 E6: run names do not include the exposure type, so over_exp runs get their own root (a subtree test, so
    # results/overexp/smoke*/ counts as inside), and low-light runs stay out of it
    in_overexp = root.is_relative_to(OVEREXP.resolve())
    if args.exp_name == "over_exp" and not in_overexp:
        raise SystemExit(f"[run_rung0] --exp_name over_exp needs a root inside {OVEREXP.resolve()}, got {root}")
    if args.exp_name == "low" and in_overexp:
        raise SystemExit(f"[run_rung0] {root} is for --exp_name over_exp runs only")
    lacking = [s for s in scenes if not (data / s / args.exp_name).is_dir()]
    if lacking:
        raise SystemExit(f"[run_rung0] no {args.exp_name}/ folder under {data} for {lacking}")
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
            "--data_dir", str(data / scene), "--exp_name", args.exp_name, "--result_dir", str(run_dir),
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
