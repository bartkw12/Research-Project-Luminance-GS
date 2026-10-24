"""Rung 0.5 analysis: the 2 x 2 factorial (colour basis learned/frozen x curve attached/detached),
read from every run's rescore/ outputs, plus colour-basis rotations and timings. Writes tables,
figures and key numbers to results/rung0/summary_rung05/.

    python rung05_analyze.py            (from Luminance-GS/examples)

Cells: baseline (official), frozenbasis (--freeze_color_basis), detachcurve
(--detach_curve_in_target), detach (--detach_pseudo_target = both cuts).
Effects are paired by seed; noise = pooled per-run seed std of the four cells.
The narrative report (Luminance-GS/Implementation_guides/Rung0.5_results.md) is written by hand
from these outputs.
"""

import json
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from rung0_analyze import (ROOT, SCENES, SEEDS, SERIES, MUTED, GRID, INK2, TIMING_EXCLUDE,
                           read_csv, write_csv, md_table, mean_std, savefig, run_name)

OUT = ROOT / "summary_rung05"
FIG = OUT / "figures"
CELLS = ["baseline", "frozenbasis", "detachcurve", "detach"]
CELL_COLOR = {"baseline": SERIES[0], "detach": SERIES[1], "frozenbasis": SERIES[2], "detachcurve": SERIES[3]}
CELL_LABEL = {"baseline": "baseline (basis learned, curve attached)",
              "frozenbasis": "frozenbasis (basis frozen, curve attached)",
              "detachcurve": "detachcurve (basis learned, curve detached)",
              "detach": "detach (basis frozen, curve detached)"}
REF_SEED = 0  # <scene>_baseline_seed0 is the shared reference depth
VAL_METRICS = ["psnr", "ssim", "lpips", "psnr_gtmean", "ang_err_deg", "hue_err_sat_deg", "sat_ratio"]
SHARE_METRICS = ["psnr_gtmean", "lpips", "hue_err_sat_deg"]
PSEUDO_KEYS = ["pseudo_psnr_gtmean", "pseudo_ang_err_deg", "pseudo_hue_err_sat_deg", "pseudo_sat_ratio",
               "pseudo0_psnr_gtmean", "pseudo0_ang_err_deg", "pseudo0_hue_err_sat_deg", "pseudo0_vs_pseudo_l1",
               "render_psnr_gtmean", "render_ang_err_deg", "render_hue_err_sat_deg", "render_sat_ratio"]
EXAMPLES = {"bike": "29", "shrub": "26"}
REAL = 2.0  # call an effect real if |mean| > REAL x seed noise


def basis_rotation_deg(axis1, axis2):
    """Rotation angle of each view's colour basis away from RGB, built exactly as tools.pixel_project."""
    a1, a2 = axis1.astype(np.float64), axis2.astype(np.float64)
    n = a1 + np.array([1.0, 0.0, 0.0])
    m = a2 + np.array([1.0, 0.0])
    n2 = np.stack([-(m[:, 0] * n[:, 1] + m[:, 1] * n[:, 2]) / n[:, 0], m[:, 0], m[:, 1]], 1)
    n3 = np.cross(n, n2)
    B = np.stack([v / np.linalg.norm(v, axis=1, keepdims=True) for v in (n, n2, n3)], 2)
    tr = np.trace(B, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def load(scene, cell, seed):
    d = ROOT / run_name(scene, cell, seed)
    rs = d / "rescore"
    if not (rs / "val_summary.json").exists():
        return None
    z = np.load(d / "diagnostics" / "curves_step10000.npz")
    return {
        "dir": d, "val": json.loads((rs / "val_summary.json").read_text()),
        "pseudo": read_csv(rs / "pseudo_targets.csv"),
        "cons": read_csv(rs / "consistency.csv"), "cons_own": read_csv(d / "diagnostics" / "consistency.csv"),
        "own_pseudo": read_csv(d / "diagnostics" / "pseudo_targets.csv"),
        "stats": json.loads((d / "stats" / "val_step9999.json").read_text()),
        "rotation": basis_rotation_deg(z["axis1"], z["axis2"]), "view_names": [str(v) for v in z["view_names"]],
    }


def col_mean(rows, key):
    return float(np.nanmean([float(r[key]) for r in rows]))


def effects(vals):
    """vals[cell] = per-seed array -> factorial effects per seed and the pooled per-run seed std."""
    b, f, c, d = (np.asarray(vals[k], float) for k in CELLS)
    eff = {
        "basis": 0.5 * ((f - b) + (d - c)),
        "curve": 0.5 * ((c - b) + (d - f)),
        "interaction": (d - f) - (c - b),
        "detach_total": d - b,
        "freeze_only": f - b,          # basis frozen first
        "basis_given_detached": d - c,  # basis frozen last
    }
    noise = float(np.sqrt(np.mean([np.var(vals[k], ddof=1) for k in CELLS])))
    return eff, noise


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = {(sc, c, s): r for sc in SCENES for c in CELLS for s in SEEDS if (r := load(sc, c, s)) is not None}
    missing = [run_name(sc, c, s) for sc in SCENES for c in CELLS for s in SEEDS if (sc, c, s) not in runs]
    print(f"loaded {len(runs)} re-scored runs" + (f"; missing {missing}" if missing else ""))
    complete = [sc for sc in SCENES if all((sc, c, s) in runs for c in CELLS for s in SEEDS)]
    groups = complete + (["pooled"] if complete else [])  # factorial tables need all 12 runs of a scene
    status = {r["run"]: r for r in read_csv(ROOT / "runs.csv")}
    summary0 = {(r["scene"], r["variant"], int(r["seed"])): r for r in read_csv(ROOT / "summary" / "val_metrics_per_run.csv")}
    tables, numbers = {}, {"missing_runs": missing}

    # ---- integrity: re-score vs the runs' own Rung 0 outputs
    chk = {"val_psnr_vs_stats": 0.0, "psnr_gtmean_png_vs_rung0": 0.0, "pseudo_psnr_vs_own_diag": 0.0}
    for (sc, c, s), r in runs.items():
        chk["val_psnr_vs_stats"] = max(chk["val_psnr_vs_stats"], abs(r["val"]["psnr"] - r["stats"]["psnr"]))
        if (sc, c, s) in summary0:
            chk["psnr_gtmean_png_vs_rung0"] = max(chk["psnr_gtmean_png_vs_rung0"],
                                                  abs(r["val"]["psnr_gtmean_png"] - float(summary0[(sc, c, s)]["psnr_gtmean"])))
        d = max(abs(float(a["pseudo_psnr"]) - float(b["pseudo_psnr"])) for a, b in zip(r["pseudo"], r["own_pseudo"]))
        chk["pseudo_psnr_vs_own_diag"] = max(chk["pseudo_psnr_vs_own_diag"], d)
    numbers["integrity_max_abs_diff"] = chk
    tables["Integrity: largest |difference| between the re-score and each run's own end-of-training outputs"] = [chk]

    # ---- per-run validation table
    per_run = []
    for (sc, c, s), r in sorted(runs.items(), key=lambda kv: (SCENES.index(kv[0][0]), CELLS.index(kv[0][1]), kv[0][2])):
        name = r["dir"].name
        minutes = float(status[name]["minutes"]) if name in status and name not in TIMING_EXCLUDE else float("nan")
        per_run.append({"scene": sc, "cell": c, "seed": s, **{m: r["val"][m] for m in VAL_METRICS},
                        "psnr_gtmean_png": r["val"]["psnr_gtmean_png"], "mean_render": r["val"]["mean_render"],
                        "mean_gt": r["val"]["mean_gt"], "minutes": minutes})
    write_csv("val_metrics_per_run", per_run, OUT)

    def vals(metric, sc=None):
        """per-seed values of a val metric per cell; sc=None pools (averages) over complete scenes."""
        scenes = [sc] if sc else complete
        return {c: [float(np.mean([runs[(x, c, s)]["val"][metric] for x in scenes])) for s in SEEDS] for c in CELLS}

    # ---- 1. validation metrics per cell
    cell_rows = []
    for sc in groups:
        for c in CELLS:
            row = {"scene": sc, "cell": c}
            for m in VAL_METRICS:
                v = vals(m, None if sc == "pooled" else sc)[c]
                row[f"{m}_mean"], row[f"{m}_std"] = mean_std(v)
            cell_rows.append(row)
    write_csv("val_metrics_per_cell", cell_rows, OUT)
    tables["1. Validation metrics per cell (mean ± sample std over seeds 0-2; pooled = mean over scenes per seed)"] = cell_rows
    numbers["val_per_cell"] = cell_rows

    # ---- 2. factorial effects and shares
    eff_rows, share_rows = [], []
    for sc in groups:
        for m in VAL_METRICS:
            eff, noise = effects(vals(m, None if sc == "pooled" else sc))
            row = {"scene": sc, "metric": m, "noise": noise}
            for k in ("basis", "curve", "interaction", "detach_total"):
                row[f"{k}_mean"] = float(eff[k].mean())
                row[f"{k}_std"] = float(eff[k].std(ddof=1))
                row[f"{k}_x_noise"] = float(eff[k].mean() / noise) if noise > 0 else float("nan")
            eff_rows.append(row)
            if m in SHARE_METRICS:
                tot = eff["detach_total"].mean()
                real = noise > 0 and abs(tot) > REAL * noise
                share_rows.append({
                    "scene": sc, "metric": m, "detach_total": float(tot), "noise": noise,
                    "detach_gain_real": bool(real),
                    "share_freeze_first": float(eff["freeze_only"].mean() / tot) if real else float("nan"),
                    "share_freeze_last": float(eff["basis_given_detached"].mean() / tot) if real else float("nan"),
                    "share_curve_first": float((np.mean(vals(m, None if sc == "pooled" else sc)["detachcurve"]) -
                                                np.mean(vals(m, None if sc == "pooled" else sc)["baseline"])) / tot) if real else float("nan"),
                })
    write_csv("factorial_effects", eff_rows, OUT)
    write_csv("detach_gain_shares", share_rows, OUT)
    tables["2a. Factorial effects, paired by seed (mean, std over seeds, and mean / pooled per-run seed std). "
           "basis = ½[(frozenbasis−baseline)+(detach−detachcurve)]; curve = ½[(detachcurve−baseline)+(detach−frozenbasis)]; "
           "interaction = (detach−frozenbasis)−(detachcurve−baseline)"] = eff_rows
    tables[f"2b. Share of the detach gain explained by freezing the basis, in both orders (only where |detach−baseline| > {REAL:g}× noise). "
           "freeze_first = (frozenbasis−baseline)/(detach−baseline); freeze_last = (detach−detachcurve)/(detach−baseline); "
           "curve_first = (detachcurve−baseline)/(detach−baseline)"] = share_rows
    numbers["effects"] = eff_rows
    numbers["shares"] = share_rows

    # ---- 3. pseudo-target quality per cell (training views, oracle) incl. zero-basis counterfactual
    pt_rows = []
    for sc in SCENES:
        for c in CELLS:
            rs = [runs[(sc, c, s)] for s in SEEDS if (sc, c, s) in runs]
            if not rs:
                continue
            row = {"scene": sc, "cell": c, "n_seeds": len(rs)}
            for k in PSEUDO_KEYS:
                row[k] = float(np.mean([col_mean(r["pseudo"], k) for r in rs]))
            pt_rows.append(row)
    write_csv("pseudo_targets_per_cell", pt_rows, OUT)
    tables["3. Training-view pseudo-targets and renders vs normal-light GT (oracle), mean over views and seeds. "
           "pseudo0 = same curve with the colour basis reset to RGB (counterfactual, no retraining)"] = pt_rows
    numbers["pseudo_per_cell"] = pt_rows

    # ---- 4. consistency: shared reference depth (reference run excluded) vs own depth
    cons_rows = []
    for sc in SCENES:
        gt_level = [col_mean(runs[(sc, c, s)]["cons"], "rel_l1_gt") for c in CELLS for s in SEEDS if (sc, c, s) in runs]
        for c in CELLS:
            seeds = [s for s in SEEDS if (sc, c, s) in runs and not (c == "baseline" and s == REF_SEED)]
            if not seeds:
                continue
            rs = [runs[(sc, c, s)] for s in seeds]
            row = {"scene": sc, "cell": c, "seeds": " ".join(map(str, seeds))}
            for k in ("pseudo", "pseudo0", "render"):
                row[f"shared_rel_l1_{k}"] = float(np.mean([col_mean(r["cons"], f"rel_l1_{k}") for r in rs]))
            row["shared_rel_l1_gt"] = float(np.mean(gt_level))
            row["shared_pseudo_over_gt"] = row["shared_rel_l1_pseudo"] / row["shared_rel_l1_gt"]
            for k in ("pseudo", "render", "gt"):
                row[f"own_rel_l1_{k}"] = float(np.mean([col_mean(r["cons_own"], f"rel_l1_{k}") for r in rs]))
            row["own_pseudo_over_gt"] = row["own_rel_l1_pseudo"] / row["own_rel_l1_gt"]
            cons_rows.append(row)
        numbers.setdefault("gt_level_spread", {})[sc] = float(np.ptp(gt_level)) if gt_level else float("nan")
    write_csv("consistency_per_cell", cons_rows, OUT)
    tables[f"4. Cross-view consistency (relative L1 after depth warp). shared_* = warp with <scene>_baseline_seed{REF_SEED}'s depth, "
           f"the reference run itself excluded; own_* = each run's own depth (Rung 0 method)"] = cons_rows
    numbers["consistency_per_cell"] = cons_rows

    # ---- 5. basis rotation vs colour error (baseline runs, per view)
    rot_rows, scatter = [], defaultdict(list)
    for sc in SCENES:
        for c in ("baseline", "detachcurve"):  # the two cells whose basis is learned
            pts = []
            for s in SEEDS:
                r = runs.get((sc, c, s))
                if r is None:
                    continue
                by_view = {p["view"]: p for p in r["pseudo"]}
                for name, rot in zip(r["view_names"], r["rotation"]):
                    p = by_view[name]
                    pts.append((rot, float(p["pseudo_hue_err_sat_deg"]),
                                float(p["pseudo_hue_err_sat_deg"]) - float(p["pseudo0_hue_err_sat_deg"]),
                                float(p["pseudo0_psnr_gtmean"]) - float(p["pseudo_psnr_gtmean"])))
            if not pts:
                continue
            a = np.array(pts)
            ok = np.isfinite(a).all(1)
            a = a[ok]
            scatter[(sc, c)] = a
            rho_h = spearmanr(a[:, 0], a[:, 1])[0] if len(a) > 2 else float("nan")
            rho_d = spearmanr(a[:, 0], a[:, 2])[0] if len(a) > 2 else float("nan")
            rho_p = spearmanr(a[:, 0], a[:, 3])[0] if len(a) > 2 else float("nan")
            rot_rows.append({"scene": sc, "cell": c, "n_view_seeds": len(a),
                             "rotation_mean_deg": float(a[:, 0].mean()), "rotation_max_deg": float(a[:, 0].max()),
                             "spearman_rot_vs_hue_err": float(rho_h),
                             "zero_basis_hue_gain_mean_deg": float(a[:, 2].mean()), "spearman_rot_vs_hue_gain": float(rho_d),
                             "zero_basis_psnr_gtmean_gain_db": float(a[:, 3].mean()), "spearman_rot_vs_psnr_gain": float(rho_p)})
    write_csv("basis_rotation_vs_colour", rot_rows, OUT)
    tables["5. Colour-basis rotation (degrees from RGB) vs pseudo-target colour error, per view and seed. "
           "hue_gain = hue error removed by resetting the basis (pseudo − pseudo0); psnr_gain = pseudo0 − pseudo (dB)"] = rot_rows
    numbers["rotation"] = rot_rows

    # ---- 6. timing
    time_rows = []
    for c in CELLS:
        mins = [m for m in (float(status[n]["minutes"]) for n in
                            (run_name(sc, c, s) for sc in SCENES for s in SEEDS)
                            if n in status and n not in TIMING_EXCLUDE)]
        per_scene = {sc: float(np.mean([float(status[run_name(sc, c, s)]["minutes"]) for s in SEEDS
                                        if run_name(sc, c, s) in status and run_name(sc, c, s) not in TIMING_EXCLUDE] or [np.nan]))
                     for sc in SCENES}
        time_rows.append({"cell": c, "n_runs": len(mins), "minutes_mean": float(np.mean(mins)) if mins else float("nan"),
                          **{f"{sc}_min": v for sc, v in per_scene.items()}})
    base = time_rows[0]["minutes_mean"]
    for row in time_rows:
        row["relative_to_baseline"] = row["minutes_mean"] / base
    write_csv("timing_per_cell", time_rows, OUT)
    tables[f"6. Wall time per 10k-step run (minutes; {', '.join(sorted(TIMING_EXCLUDE))} excluded)"] = time_rows
    numbers["timing"] = time_rows

    # =========================================================== figures
    # validation metrics per run
    fig, axes = plt.subplots(1, 4, figsize=(19, 3.8))
    for ax, (m, lab) in zip(axes, (("psnr_gtmean", "brightness-matched PSNR (dB) ↑"), ("lpips", "LPIPS ↓"),
                                   ("hue_err_sat_deg", "hue error on coloured pixels (°) ↓"), ("sat_ratio", "saturation ratio (1 = GT)"))):
        for j, sc in enumerate(SCENES):
            for k, c in enumerate(CELLS):
                v = [runs[(sc, c, s)]["val"][m] for s in SEEDS if (sc, c, s) in runs]
                x = j + (k - 1.5) * 0.18
                ax.scatter([x] * len(v), v, color=CELL_COLOR[c], s=22, zorder=3, label=c if j == 0 else None)
                if v:
                    ax.hlines(np.mean(v), x - 0.07, x + 0.07, color=CELL_COLOR[c], linewidth=2)
        if m == "sat_ratio":
            ax.axhline(1.0, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xticks(range(len(SCENES))); ax.set_xticklabels(SCENES); ax.set_ylabel(lab); ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8)
    fig.suptitle("Validation metrics per run, all four cells (dots: seeds 0-2, bar: mean)", x=0.01, ha="left")
    fig.tight_layout(); savefig(fig, "val_metrics_per_cell", FIG)

    # factorial effects with noise bands
    names = {"psnr_gtmean": "brightness-matched PSNR (dB)", "lpips": "LPIPS", "hue_err_sat_deg": "hue error (°)", "sat_ratio": "saturation ratio"}
    fig, axes = plt.subplots(1, len(names), figsize=(19, 3.9))
    parts = [("basis", "freeze basis", SERIES[2]), ("curve", "detach curve", SERIES[3]), ("interaction", "interaction", MUTED)]
    for ax, (m, lab) in zip(axes, names.items()):
        for j, sc in enumerate(groups):
            row = next(r for r in eff_rows if r["scene"] == sc and r["metric"] == m)
            ax.add_patch(plt.Rectangle((j - 0.4, -REAL * row["noise"]), 0.8, 2 * REAL * row["noise"], color=GRID, alpha=0.7, linewidth=0, zorder=1))
            for q, (k, klab, col) in enumerate(parts):
                ax.bar(j + (q - 1) * 0.25, row[f"{k}_mean"], width=0.22, yerr=row[f"{k}_std"], color=col, capsize=2,
                       zorder=3, label=klab if j == 0 else None, error_kw={"elinewidth": 1, "ecolor": INK2})
        ax.axhline(0, color=MUTED, linewidth=1)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups); ax.set_ylabel(f"effect on {lab}"); ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Factorial effects on validation metrics (bar: mean over seeds, whisker: std; grey band: ±{REAL:g}× seed noise)", x=0.01, ha="left")
    fig.tight_layout(); savefig(fig, "factorial_effects", FIG)

    # rotation vs hue error
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (idx, lab) in zip(axes, ((1, "pseudo-target hue error (°)"), (2, "hue error removed by resetting basis (°)"))):
        for sc in SCENES:
            a = scatter.get((sc, "baseline"))
            if a is not None and len(a):
                ax.scatter(a[:, 0], a[:, idx], s=12, alpha=0.7, color=SERIES[SCENES.index(sc)], label=sc)
        ax.set_xlabel("colour-basis rotation from RGB (°)"); ax.set_ylabel(lab)
    axes[0].legend(fontsize=8)
    fig.suptitle("Baseline runs: per-view colour-basis rotation vs pseudo-target colour error (Spearman ρ in table 5)", x=0.01, ha="left")
    fig.tight_layout(); savefig(fig, "rotation_vs_hue_error", FIG)

    # 2 x 2 example grids
    for sc, view in EXAMPLES.items():
        imgs = [(c, runs[(sc, c, 0)]["dir"] / "rescore" / f"example_{view}_in_pseudo_render_gt.png") for c in CELLS if (sc, c, 0) in runs]
        imgs = [(c, p) for c, p in imgs if p.exists()]
        if not imgs:
            continue
        fig, axes = plt.subplots(len(imgs), 1, figsize=(14, 2.9 * len(imgs)))
        for ax, (c, p) in zip(np.atleast_1d(axes), imgs):
            ax.imshow(imageio.imread(p)); ax.set_axis_off()
            ax.set_title(CELL_LABEL[c], loc="left", fontsize=10, color=CELL_COLOR[c])
        fig.suptitle(f"{sc} view {view}, seed 0: input | pseudo-target | enhanced render | normal-light GT", x=0.01, ha="left")
        fig.tight_layout(); savefig(fig, f"examples_{sc}_{view}_per_cell", FIG)
        zb = runs.get((sc, "baseline", 0))
        if zb is not None and (zb["dir"] / "rescore" / f"example_{view}_pseudo_zerobasis_diffx4.png").exists():
            img = imageio.imread(zb["dir"] / "rescore" / f"example_{view}_pseudo_zerobasis_diffx4.png")
            fig, ax = plt.subplots(figsize=(14, 3.2))
            ax.imshow(img); ax.set_axis_off()
            ax.set_title(f"{sc} view {view}, baseline seed 0: pseudo-target | same curve with basis reset to RGB | |difference| × 4", loc="left", fontsize=10)
            fig.tight_layout(); savefig(fig, f"zero_basis_{sc}_{view}", FIG)

    # =========================================================== write
    with open(OUT / "tables.md", "w", encoding="utf-8") as f:
        f.write("# Rung 0.5 tables (generated by examples/rung05_analyze.py)\n\n")
        if missing:
            f.write(f"**Missing runs:** {', '.join(missing)}\n\n")
        for title, rows in tables.items():
            f.write(f"## {title}\n\n{md_table(rows)}\n")
    with open(OUT / "numbers.json", "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
