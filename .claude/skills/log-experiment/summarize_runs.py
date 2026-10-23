"""Summarise finished runs for a research-log entry (stdlib only, read-only).

Takes run folders (<scene>_<variant>_seed<seed>/) or root folders that contain them, and prints
markdown tables: mean ± sample std over seeds per scene x variant, and paired differences
against a reference variant, with x-noise = mean difference / pooled per-run seed std.

    python .claude/skills/log-experiment/summarize_runs.py Luminance-GS/results/rung0
    python .claude/skills/log-experiment/summarize_runs.py Luminance-GS/results/rung0 --variants baseline frozenbasis
"""

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

RUN_RE = re.compile(r"^(?P<scene>[a-z0-9]+)_(?P<variant>[a-z0-9]+)_seed(?P<seed>\d+)$")
# (key, label, decimals); rescore keys appear only for runs re-scored in Rung 0.5 or later
METRICS = [
    ("psnr", "PSNR", 2), ("psnr_gtmean", "PSNR bm", 2), ("ssim", "SSIM", 4), ("lpips", "LPIPS", 4),
    ("ang_err_deg", "ang err", 2), ("hue_err_sat_deg", "hue err", 2), ("sat_ratio", "sat ratio", 3),
    ("minutes", "min", 1),
]


def find_runs(paths):
    runs = []
    for p in map(Path, paths):
        dirs = [p] if (p / "cfg.json").exists() else sorted(d for d in p.iterdir() if d.is_dir())
        runs += [d for d in dirs if RUN_RE.match(d.name)]
    return runs


def minutes_by_run(roots):
    out = {}
    for root in roots:
        f = root / "runs.csv"
        if f.exists():
            with open(f, newline="", encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    if row.get("ok") == "True":
                        out[row["run"]] = float(row["minutes"])
    return out


def read_run(run_dir, minutes):
    m = RUN_RE.match(run_dir.name)
    rec = {"run": run_dir.name, "scene": m["scene"], "variant": m["variant"], "seed": int(m["seed"])}
    vals = sorted((run_dir / "stats").glob("val_step*.json"))
    if not vals:
        return rec, False
    rec["step"] = int(vals[-1].stem.replace("val_step", ""))
    rec.update(json.loads(vals[-1].read_text()))
    rescore = run_dir / "rescore" / "val_summary.json"
    if rescore.exists():
        r = json.loads(rescore.read_text())
        rec.update({k: r[k] for k, _, _ in METRICS if k in r and k not in rec})  # keep the training-time eval
    if run_dir.name in minutes:
        rec["minutes"] = minutes[run_dir.name]
    return rec, True


def mean_std(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else float("nan"))


def fmt(x, d):
    return "n/a" if x is None or math.isnan(x) else f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--reference", default="baseline", help="variant the paired differences are taken against")
    args = ap.parse_args()

    run_dirs = find_runs(args.paths)
    minutes = minutes_by_run({d.parent for d in run_dirs})
    groups, missing = defaultdict(list), []
    for d in run_dirs:
        rec, ok = read_run(d, minutes)
        if args.variants and rec["variant"] not in args.variants or args.scenes and rec["scene"] not in args.scenes:
            continue
        (groups[(rec["scene"], rec["variant"])].append(rec) if ok else missing.append(rec["run"]))
    if not groups:
        raise SystemExit("no finished runs found")

    # a run whose wall time is > 3x its variant's median was almost certainly paused (PC asleep)
    slow = []
    for variant in {v for _, v in groups}:
        rs = [r for (_, v), g in groups.items() if v == variant for r in g if "minutes" in r]
        med = statistics.median(r["minutes"] for r in rs) if rs else 0
        for r in rs:
            if r["minutes"] > 3 * med:
                slow.append(f"{r['run']} ({r['minutes']:.0f} min)")
                del r["minutes"]

    metrics = [m for m in METRICS if any(m[0] in r for rs in groups.values() for r in rs)]
    steps = sorted({r["step"] for rs in groups.values() for r in rs})
    print(f"Runs: {sum(map(len, groups.values()))} finished, final eval step(s) {steps}. Mean ± sample std over seeds.\n")
    print("| Scene | Variant | n | " + " | ".join(lbl for _, lbl, _ in metrics) + " |")
    print("|---" * (3 + len(metrics)) + "|")
    for (scene, variant), rs in sorted(groups.items()):
        cells = []
        for key, _, d in metrics:
            xs = [r[key] for r in rs if key in r]
            cells.append("n/a" if not xs else "{} ± {}".format(*(fmt(v, d) for v in mean_std(xs))))
        print(f"| {scene} | {variant} | {len(rs)} | " + " | ".join(cells) + " |")

    others = sorted({v for _, v in groups} - {args.reference})
    if (args.reference in {v for _, v in groups}) and others:
        print(f"\nPaired by seed: variant − {args.reference} (mean ± std of differences, ×noise = mean / pooled seed std)\n")
        paired = [m for m in metrics if m[0] != "minutes"]
        print("| Scene | Variant | n pairs | " + " | ".join(lbl for _, lbl, _ in paired) + " |")
        print("|---" * (3 + len(paired)) + "|")
        for scene in sorted({s for s, _ in groups}):
            ref = {r["seed"]: r for r in groups.get((scene, args.reference), [])}
            for variant in others:
                var = {r["seed"]: r for r in groups.get((scene, variant), [])}
                seeds = sorted(ref.keys() & var.keys())
                if not seeds:
                    continue
                cells = []
                for key, _, d in paired:
                    pairs = [(var[s][key], ref[s][key]) for s in seeds if key in var[s] and key in ref[s]]
                    if len(pairs) < 2:
                        cells.append("n/a")
                        continue
                    mu, sd = mean_std([a - b for a, b in pairs])
                    noise = math.sqrt((statistics.variance([a for a, _ in pairs]) + statistics.variance([b for _, b in pairs])) / 2)
                    cells.append(f"{mu:+.{d}f} ± {fmt(sd, d)} ({mu / noise:+.1f}×)" if noise > 0 else f"{mu:+.{d}f}")
                print(f"| {scene} | {variant} | {len(seeds)} | " + " | ".join(cells) + " |")

    if slow:
        print(f"\nWall time excluded (over 3x the variant median, likely sleep): {', '.join(sorted(slow))}")
    if missing:
        print(f"\nIncomplete runs (no stats/val_step*.json): {', '.join(sorted(missing))}")


if __name__ == "__main__":
    main()
