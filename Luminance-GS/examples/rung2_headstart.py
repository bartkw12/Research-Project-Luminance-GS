"""Head-start check (M2_rung2_plan.md 5.3, risk 5): Rung 2 with F initialised at the identity (variant rung2id) against
the default `cdf` init (rung2) and detach, LOM, seed 0. CPU only. Reads finished runs; writes only
results/rung2/summary_headstart/ (or, with --selftest, the git-ignored results/rung2/smoke_m2/summary_headstart_selftest/).

The reading was fixed in docs/RESEARCH_LOG.md (2026-11-15) before any result:
- one seed, so no "real": each difference is shown next to the 3-seed sd of the M3 cells (rung2, detach);
- the check cannot change the H1 verdict (the rule applies to the pre-registered default, `cdf` init);
- "the head start explains bike": rung2id - rung2 (bike bm PSNR) above 2x rung2's bike sd AND closing most (> half) of
  the 3-seed bike gap rung2 - detach; "it does not": |rung2id - rung2| within 2x that sd; otherwise neither reading holds;
- the F gap (identity-init F against cdf-init F at the end) is reported alongside, because a difference could also come
  from a different final F, not only from the head start.

Run from Luminance-GS/examples:
    python rung2_headstart.py              # the check
    python rung2_headstart.py --selftest   # rung2id read from rung2 seed 0: every rung2id - rung2 difference must be 0
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parent
RESULTS = EXAMPLES.parent / "results"
RUNG0, RUNG2 = RESULTS / "rung0", RESULTS / "rung2"
SCENES = ["buu", "chair", "sofa", "bike", "shrub"]
SEEDS = (0, 1, 2)
METRICS = [("psnr_gtmean", "bm PSNR"), ("psnr", "raw PSNR"), ("ssim", "SSIM"), ("lpips", "LPIPS"),
           ("hue_err_sat_deg", "hue err (°)")]
TRAIN = [("pseudo_psnr_gtmean", "pseudo-target bm PSNR"), ("render_psnr_gtmean", "render bm PSNR"),
         ("fit_psnr_render_vs_pseudo", "render fit to pseudo-target")]
CFG_MAY_DIFFER = {"rung2_F_init", "result_dir"}  # everything else in cfg.json must equal the rung2 seed-0 run's
F_CLOSE = 0.01  # the preview's "within 0.01 of the same F" (plan step 6)


def run_dir(variant, scene, seed, selftest=False):
    if variant == "rung2id":
        return RUNG2 / (f"{scene}_rung2_seed0" if selftest else f"{scene}_rung2id_seed{seed}")
    return (RUNG0 if variant == "detach" else RUNG2) / f"{scene}_{variant}_seed{seed}"


def ok_runs(root):
    rows = list(csv.DictReader(open(root / "runs.csv", newline="")))
    last = {}
    for r in rows:
        last[r["run"]] = r["ok"] == "True"
    return last


def val(d):
    return json.load(open(d / "rescore" / "val_summary.json"))


def train_means(d):
    rows = list(csv.DictReader(open(d / "diagnostics" / "pseudo_targets.csv", newline="")))
    return {k: float(np.mean([float(r[k]) for r in rows])) for k, _ in TRAIN}


def check_inputs(selftest):
    """Stop before any table if a run is missing, not ok, not re-scored against rung0's baseline seed-0 depth, or if its
    cfg differs from the rung2 seed-0 run in anything but the F init (plan 6.1)."""
    problems, ok0, ok2 = [], ok_runs(RUNG0), ok_runs(RUNG2)
    for sc in SCENES:
        for v, seeds in (("rung2id", (0,)), ("rung2", SEEDS), ("detach", SEEDS)):
            for s in seeds:
                d = run_dir(v, sc, s, selftest)
                if not (d / "rescore" / "val_summary.json").exists():
                    problems.append(f"{d.name}: no rescore/val_summary.json")
                    continue
                if not (ok0 if v == "detach" else ok2).get(d.name, False):
                    problems.append(f"{d.name}: no ok row in runs.csv")
                ref = Path(val(d).get("ref_depth_run", ""))
                if ref.name != f"{sc}_baseline_seed0" or ref.parent.name != "rung0":
                    problems.append(f"{d.name}: re-scored against {ref} (needs rung0/{sc}_baseline_seed0)")
        cid = json.load(open(run_dir("rung2id", sc, 0, selftest) / "cfg.json"))
        c2 = json.load(open(run_dir("rung2", sc, 0) / "cfg.json"))
        diff = {k for k in set(cid) | set(c2) if cid.get(k) != c2.get(k)} - CFG_MAY_DIFFER
        if diff:
            problems.append(f"{sc}: rung2id cfg differs from rung2 seed 0 in {sorted(diff)}")
        if not selftest and cid.get("rung2_F_init") != "identity":
            problems.append(f"{sc}: rung2id has rung2_F_init {cid.get('rung2_F_init')}")
        if cid.get("max_steps") != 10_000:
            problems.append(f"{sc}: rung2id max_steps {cid.get('max_steps')}")
    return problems


def sd(x):
    return float(np.std(x, ddof=1))


def curves(selftest):
    """Final F and s of the identity-init run against the cdf-init run (seed 0), plus the identity run's F trajectory
    against the cdf run's final F."""
    out, traj = {}, []
    for sc in SCENES:
        a = json.load(open(run_dir("rung2id", sc, 0, selftest) / "diagnostics" / "rung2_state.json"))
        b = json.load(open(run_dir("rung2", sc, 0) / "diagnostics" / "rung2_state.json"))
        Fa, Fb = np.asarray(a["F_grid"]), np.asarray(b["F_grid"])
        sa, sb = np.asarray(a["s_final"]), np.asarray(b["s_final"])
        out[sc] = {"F_init_id_run": a["F_init"], "max_dF": float(np.max(np.abs(Fa - Fb))),
                   "argmax_dF_level": int(np.argmax(np.abs(Fa - Fb))),
                   "max_dF_levels_1_8": float(np.max(np.abs(Fa[1:9] - Fb[1:9]))),
                   "F_id_levels_1_4_x255": [round(float(v) * 255, 1) for v in Fa[1:5]],
                   "F_cdf_levels_1_4_x255": [round(float(v) * 255, 1) for v in Fb[1:5]],
                   "gamma_id": a["gamma"], "gamma_cdf": b["gamma"],
                   "max_ds": float(np.max(np.abs(sa - sb))), "rms_ds": float(np.sqrt(np.mean((sa - sb) ** 2)))}
        first = None
        for f in sorted((run_dir("rung2id", sc, 0, selftest) / "diagnostics").glob("curves_step*.npz")):
            z = np.load(f)
            gap = float(np.max(np.abs(z["curve_global"] - Fb)))
            traj.append({"scene": sc, "step": int(z["step"]), "max_dF_to_cdf_final": round(gap, 4)})
            if first is None and gap <= F_CLOSE:
                first = int(z["step"])
        out[sc]["first_snapshot_step_within_0.01"] = first
    return out, traj


def fmt(x, k):
    return f"{x:+.4f}" if k in ("ssim", "lpips") else f"{x:+.2f}" if k == "hue_err_sat_deg" else f"{x:+.3f}"


def table(rows):
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="read rung2id from rung2 seed 0; all differences must be 0")
    args = ap.parse_args()
    out = RUNG2 / ("smoke_m2/summary_headstart_selftest" if args.selftest else "summary_headstart")
    assert out.parent in (RUNG2, RUNG2 / "smoke_m2") and out.name.startswith("summary_headstart"), out

    problems = check_inputs(args.selftest)
    if problems:
        raise SystemExit("[headstart] stopped before any table:\n  " + "\n  ".join(problems))

    v = {(var, sc, s): val(run_dir(var, sc, s, args.selftest)) for sc in SCENES
         for var, seeds in (("rung2id", (0,)), ("rung2", SEEDS), ("detach", SEEDS)) for s in seeds}
    groups = {**{sc: [sc] for sc in SCENES}, "pooled": SCENES, "pooled_no_shrub": SCENES[:4]}

    def per_seed(var, k, g, seeds):
        return np.array([np.mean([v[(var, sc, s)][k] for sc in groups[g]]) for s in seeds])

    val_rows, nums = [], {"validation": {}, "train_views": {}}
    for g in groups:
        for k, name in METRICS:
            i0 = per_seed("rung2id", k, g, (0,))[0]
            r = per_seed("rung2", k, g, SEEDS)
            d = per_seed("detach", k, g, SEEDS)
            row = {"scene": g, "metric": name, "rung2id s0": f"{i0:.4f}", "rung2 s0": f"{r[0]:.4f}",
                   "detach s0": f"{d[0]:.4f}",
                   "rung2id − rung2 [rung2 sd]": f"{fmt(i0 - r[0], k)} [{sd(r):.4f}]",
                   "rung2id − detach [detach sd]": f"{fmt(i0 - d[0], k)} [{sd(d):.4f}]",
                   "rung2 − detach, s0 / 3-seed mean": f"{fmt(r[0] - d[0], k)} / {fmt(np.mean(r - d), k)}"}
            val_rows.append(row)
            nums["validation"][f"{g}/{k}"] = {"rung2id_s0": i0, "rung2_s0": r[0], "detach_s0": d[0],
                                             "rung2_sd": sd(r), "detach_sd": sd(d),
                                             "rung2_minus_detach_mean": float(np.mean(r - d))}

    train_rows = []
    for sc in SCENES:
        ti = train_means(run_dir("rung2id", sc, 0, args.selftest))
        t2, td = train_means(run_dir("rung2", sc, 0)), train_means(run_dir("detach", sc, 0))
        for k, name in TRAIN:
            train_rows.append({"scene": sc, "quantity (training views, seed 0)": name, "rung2id": f"{ti[k]:.3f}",
                               "rung2": f"{t2[k]:.3f}", "detach": f"{td[k]:.3f}",
                               "rung2id − rung2": f"{ti[k] - t2[k]:+.3f}", "rung2id − detach": f"{ti[k] - td[k]:+.3f}"})
            nums["train_views"][f"{sc}/{k}"] = {"rung2id": ti[k], "rung2": t2[k], "detach": td[k]}

    cur, traj = curves(args.selftest)
    nums["curves"] = cur
    curve_rows = [{"scene": sc, "max |ΔF| (×255)": f"{c['max_dF'] * 255:.1f} at level {c['argmax_dF_level']}",
                   "max |ΔF| levels 1–8 (×255)": f"{c['max_dF_levels_1_8'] * 255:.1f}",
                   "F levels 1–4, identity init": c["F_id_levels_1_4_x255"],
                   "F levels 1–4, cdf init": c["F_cdf_levels_1_4_x255"],
                   "γ id / cdf": f"{c['gamma_id']:.3f} / {c['gamma_cdf']:.3f}",
                   "max / rms |Δs| (stops)": f"{c['max_ds']:.3f} / {c['rms_ds']:.3f}",
                   f"first snapshot within {F_CLOSE}": c["first_snapshot_step_within_0.01"] or "never"}
                  for sc, c in cur.items()]

    # the reading fixed in the log before results (bike, bm PSNR)
    b = nums["validation"]["bike/psnr_gtmean"]
    dd, sd2, gap = b["rung2id_s0"] - b["rung2_s0"], b["rung2_sd"], -b["rung2_minus_detach_mean"]
    if dd > 2 * sd2 and dd > 0.5 * gap:
        reading = "the head start is a likely contributor to the bike gap (inferred, one seed)"
    elif abs(dd) <= 2 * sd2:
        reading = "the head start does not explain the bike gap (one seed)"
    else:
        reading = "neither pre-declared reading holds (one seed)"
    nums["reading"] = {"bike_rung2id_minus_rung2": dd, "two_sd_rung2": 2 * sd2, "half_gap": 0.5 * gap,
                       "verdict": reading}

    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "validation.csv", val_rows)
    save_csv(out / "train_views.csv", train_rows)
    save_csv(out / "curves.csv", curve_rows)
    save_csv(out / "F_trajectory.csv", traj)
    with open(out / "numbers.json", "w", encoding="utf-8") as f:
        json.dump(nums, f, indent=1, default=float)
    md = [f"# Head-start check (plan 5.3): rung2 with F initialised at the identity, LOM, seed 0"
          f"{' (SELFTEST: rung2id = rung2 seed 0)' if args.selftest else ''}", "",
          "One seed: no ×noise and no \"real\". Differences sit next to the 3-seed sd of the M3 cells. The check cannot "
          "change the H1 verdict. Reading fixed in docs/RESEARCH_LOG.md (2026-11-15) before any result. Every number is "
          "*measured*.", "",
          f"**Reading (bike, bm PSNR):** rung2id − rung2 = {dd:+.3f} dB; 2× rung2's bike sd = {2 * sd2:.3f}; half the "
          f"3-seed gap to detach = {0.5 * gap:.3f} -> **{reading}**.", "",
          "## Validation (rescore/val_summary.json)", "", table(val_rows), "",
          "## Training views (diagnostics/pseudo_targets.csv, mean over views)", "", table(train_rows), "",
          "## Curves: identity-init F and s against the cdf-init run, seed 0 (rung2_state.json; trajectory from "
          "curves_step*.npz against the cdf run's final F)", "", table(curve_rows), ""]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"[headstart] wrote {out}")


if __name__ == "__main__":
    main()
