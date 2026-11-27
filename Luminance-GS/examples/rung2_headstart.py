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

3-seed confirmation (--seeds 0 1 2; exploratory, the identity init was picked after seeing seed 0; reading fixed in the
log on 2026-11-16 before any result): rung2id - rung2 and rung2id - detach paired by seed, noise = sqrt(mean of the two
arms' seed variances), real above 2x. "Init effect confirmed": pooled rung2id - rung2 bm PSNR real and positive, LPIPS
not worse beyond 2x noise, and the seed 1-2 mean positive (seeds 1-2 are the clean replication); bike judged the same
way. rung2id - detach also goes through the 12 H1 checks (plan 6.5), as exploratory only: it cannot change H1.
Writes results/rung2/summary_headstart_seeds012/.

Run from Luminance-GS/examples:
    python rung2_headstart.py                         # the seed-0 check
    python rung2_headstart.py --seeds 0 1 2           # the 3-seed confirmation
    python rung2_headstart.py [--seeds ...] --selftest   # rung2id read from rung2: every rung2id - rung2 difference is 0
Both outputs are finished (CLAUDE.md rule 6, protected.py): since 2026-11-26 the first two commands refuse before
anything is written; --selftest still runs.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from protected import refuse_finished

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
        return RUNG2 / (f"{scene}_rung2_seed{seed}" if selftest else f"{scene}_rung2id_seed{seed}")
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


def check_inputs(selftest, id_seeds=(0,)):
    """Stop before any table if a run is missing, not ok, not re-scored against rung0's baseline seed-0 depth, or if its
    cfg differs from the rung2 run of the same seed in anything but the F init (plan 6.1)."""
    problems, ok0, ok2 = [], ok_runs(RUNG0), ok_runs(RUNG2)
    for sc in SCENES:
        for v, seeds in (("rung2id", id_seeds), ("rung2", SEEDS), ("detach", SEEDS)):
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
        for s in id_seeds:
            cfg_path = run_dir("rung2id", sc, s, selftest) / "cfg.json"
            if not cfg_path.exists():
                continue  # already reported above
            cid = json.load(open(cfg_path))
            c2 = json.load(open(run_dir("rung2", sc, s) / "cfg.json"))
            diff = {k for k in set(cid) | set(c2) if cid.get(k) != c2.get(k)} - CFG_MAY_DIFFER
            if diff:
                problems.append(f"{sc} seed {s}: rung2id cfg differs from rung2 seed {s} in {sorted(diff)}")
            if not selftest and cid.get("rung2_F_init") != "identity":
                problems.append(f"{sc} seed {s}: rung2id has rung2_F_init {cid.get('rung2_F_init')}")
            if cid.get("max_steps") != 10_000:
                problems.append(f"{sc} seed {s}: rung2id max_steps {cid.get('max_steps')}")
    return problems


def sd(x):
    return float(np.std(x, ddof=1))


def curves(selftest, seed=0):
    """Final F and s of the identity-init run against the cdf-init run (same seed), plus the identity run's F trajectory
    against the cdf run's final F."""
    out, traj = {}, []
    for sc in SCENES:
        a = json.load(open(run_dir("rung2id", sc, seed, selftest) / "diagnostics" / "rung2_state.json"))
        b = json.load(open(run_dir("rung2", sc, seed) / "diagnostics" / "rung2_state.json"))
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
        for f in sorted((run_dir("rung2id", sc, seed, selftest) / "diagnostics").glob("curves_step*.npz")):
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


H1_SCENES = ["buu", "chair", "sofa", "bike"]
WORSE = {"psnr_gtmean": -1, "ssim": -1, "lpips": +1}  # the direction in which each H1 metric gets worse (plan 6.5)


def curve_loss(d):
    """Rung 2's curve loss (training photos only): mean of the last 10 logged steps of loss_log.csv."""
    rows = list(csv.DictReader(open(d / "diagnostics" / "loss_log.csv", newline="")))[-10:]
    return float(np.mean([float(r["hist_loss"]) for r in rows]))


def paired(seeds, selftest, out):
    """The 3-seed confirmation (reading fixed in the log on 2026-11-16 before any result)."""
    seeds = tuple(seeds)
    problems = check_inputs(selftest, seeds)
    if problems:
        raise SystemExit("[headstart] stopped before any table:\n  " + "\n  ".join(problems))
    v = {(var, sc, s): val(run_dir(var, sc, s, selftest)) for sc in SCENES for var in ("rung2id", "rung2", "detach")
         for s in seeds}
    groups = {**{sc: [sc] for sc in SCENES}, "pooled": SCENES, "pooled_no_shrub": SCENES[:4]}
    rep = [i for i, s in enumerate(seeds) if s != 0]  # seeds 1-2: the clean replication (seed 0 chose the question)

    def per_seed(var, k, g):
        return np.array([np.mean([v[(var, sc, s)][k] for sc in groups[g]]) for s in seeds])

    def eff(a, b):
        diff, noise = a - b, float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2))
        return float(diff.mean()), noise, (float(diff.mean()) / noise if noise > 0 else float("nan"))

    def show(m, r, k):
        return f"{fmt(m, k)} ({r:+.1f}×){' *' if abs(r) > 2 else ''}"

    val_rows, nums = [], {"seeds": list(seeds), "validation": {}, "decisions": {}, "h1_exploratory": {}, "curves": {}}
    for g in groups:
        for k, name in METRICS:
            i, r, d = per_seed("rung2id", k, g), per_seed("rung2", k, g), per_seed("detach", k, g)
            m1, n1, x1 = eff(i, r)
            m2, n2, x2 = eff(i, d)
            m12 = float(np.mean((i - r)[rep])) if rep else float("nan")
            val_rows.append({"scene": g, "metric": name, "rung2id": f"{i.mean():.4f} ± {sd(i):.4f}",
                             "rung2": f"{r.mean():.4f} ± {sd(r):.4f}", "detach": f"{d.mean():.4f} ± {sd(d):.4f}",
                             "rung2id − rung2": show(m1, x1, k), "rung2id − rung2, seeds 1–2 only": fmt(m12, k),
                             "rung2id − detach": show(m2, x2, k)})
            nums["validation"][f"{g}/{k}"] = {"rung2id": i.tolist(), "rung2": r.tolist(), "detach": d.tolist(),
                                             "id_minus_rung2": [m1, n1, x1], "id_minus_rung2_seeds12": m12,
                                             "id_minus_detach": [m2, n2, x2]}

    dec_lines = []
    for g in ("pooled", "pooled_no_shrub", "bike"):
        e = nums["validation"][f"{g}/psnr_gtmean"]
        m, n, x = e["id_minus_rung2"]
        lx = nums["validation"][f"{g}/lpips"]["id_minus_rung2"][2]
        s12 = e["id_minus_rung2_seeds12"]
        if x > 2 and m > 0 and lx <= 2 and s12 > 0:
            verdict = "init effect confirmed"
        elif abs(x) <= 2:
            verdict = "not confirmed (within 2× noise)"
        else:
            verdict = "neither pre-declared reading holds"
        nums["decisions"][g] = {"bm": [m, n, x], "lpips_x": lx, "seeds12_bm": s12, "verdict": verdict}
        dec_lines.append(f"- **{g}:** rung2id − rung2 bm PSNR {m:+.3f} dB ({x:+.1f}× noise {n:.3f}); LPIPS {lx:+.1f}×; "
                         f"seeds 1–2 mean {s12:+.3f} dB -> **{verdict}**.")

    h1_rows, n_pass = [], 0
    for sc in H1_SCENES:
        for k, sign in WORSE.items():
            m, n, x = eff(per_seed("rung2id", k, sc), per_seed("detach", k, sc))
            ok = not (sign * x > 2)
            n_pass += ok
            h1_rows.append({"scene": sc, "metric": dict(METRICS)[k], "rung2id − detach": show(m, x, k),
                            "bar (2× noise, worse side)": fmt(sign * 2 * n, k), "check": "meets" if ok else "fails"})
            nums["h1_exploratory"][f"{sc}/{k}"] = {"diff": m, "noise": n, "x": x, "meets": bool(ok)}
    nums["h1_exploratory"]["n_pass"] = int(n_pass)

    train_rows = []
    for sc in SCENES:
        t = {var: [train_means(run_dir(var, sc, s, selftest)) for s in seeds] for var in ("rung2id", "rung2", "detach")}
        for k, name in TRAIN:
            i, r, d = (np.array([x[k] for x in t[var]]) for var in ("rung2id", "rung2", "detach"))
            m1, _, x1 = eff(i, r)
            m2, _, x2 = eff(i, d)
            train_rows.append({"scene": sc, "quantity (training views)": name, "rung2id": f"{i.mean():.3f}",
                               "rung2": f"{r.mean():.3f}", "detach": f"{d.mean():.3f}",
                               "rung2id − rung2": show(m1, x1, "x"), "rung2id − detach": show(m2, x2, "x")})

    curve_rows = []
    for s in seeds:
        cur, _ = curves(selftest, s)
        for sc, c in cur.items():
            li, lc = curve_loss(run_dir("rung2id", sc, s, selftest)), curve_loss(run_dir("rung2", sc, s))
            curve_rows.append({"scene": sc, "seed": s,
                               "max |ΔF| (×255)": f"{c['max_dF'] * 255:.1f} at level {c['argmax_dF_level']}",
                               "F levels 1–4, identity / cdf": f"{c['F_id_levels_1_4_x255']} / {c['F_cdf_levels_1_4_x255']}",
                               f"first snapshot within {F_CLOSE}": c["first_snapshot_step_within_0.01"] or "never",
                               "curve loss identity / cdf": f"{li:.5f} / {lc:.5f} ({(li / lc - 1) * 100:+.1f}%)"})
            nums["curves"][f"{sc}/seed{s}"] = {**c, "curve_loss_id": li, "curve_loss_cdf": lc}

    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / "validation.csv", val_rows)
    save_csv(out / "h1_exploratory.csv", h1_rows)
    save_csv(out / "train_views.csv", train_rows)
    save_csv(out / "curves.csv", curve_rows)
    with open(out / "numbers.json", "w", encoding="utf-8") as f:
        json.dump(nums, f, indent=1, default=float)
    md = [f"# Identity-init confirmation: rung2id against rung2 (cdf init) and detach, LOM, seeds "
          f"{', '.join(map(str, seeds))}{' (SELFTEST: rung2id = rung2)' if selftest else ''}", "",
          "**Exploratory:** the identity init was picked after seeing seed-0 validation results. Paired by seed; noise = "
          "sqrt(mean of the two arms' seed variances); (×) and * = real above 2×. Seeds 1–2 are the clean replication. "
          "Reading fixed in docs/RESEARCH_LOG.md (2026-11-16) before any result. Every number is *measured*.", "",
          "## Decisions (pre-declared)", "", *dec_lines, "",
          f"## H1 rule applied to rung2id − detach (exploratory; cannot change H1): {n_pass} of 12 checks meet", "",
          table(h1_rows), "", "## Validation (rescore/val_summary.json)", "", table(val_rows), "",
          "## Training views (diagnostics/pseudo_targets.csv, mean over views)", "", table(train_rows), "",
          "## Curves per seed: identity-init F against the cdf-init run of the same seed; curve loss = mean of the last "
          "10 logged steps (training photos only)", "", table(curve_rows), ""]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"[headstart] wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true", help="read rung2id from rung2 (same seed); differences must be 0")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0], help="0 = the seed-0 check; 0 1 2 = the confirmation")
    ap.add_argument("--out", type=Path, default=None, help="test output folder, only under results/rung2/smoke_m2/")
    args = ap.parse_args()
    tag = "" if args.seeds == [0] else "_seeds" + "".join(map(str, args.seeds))
    out = args.out or RUNG2 / (f"smoke_m2/summary_headstart{tag}_selftest" if args.selftest else f"summary_headstart{tag}")
    out = refuse_finished(out, "headstart")  # CLAUDE.md rule 6: summary_headstart/ and summary_headstart_seeds012/
    assert (out.parent == RUNG2.resolve() and out.name.startswith("summary_headstart")) or \
        out.parent == (RUNG2 / "smoke_m2").resolve(), out
    if args.seeds != [0]:
        return paired(args.seeds, args.selftest, out)

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
