"""Rung 0 analysis: reads every run under results/rung0/ and writes tables, figures and
key numbers to results/rung0/summary/.

    python rung0_analyze.py            (from Luminance-GS/examples)

Outputs (summary/):
    tables.md            every table below, rendered as markdown
    numbers.json         headline numbers quoted in Rung0_results.md
    *.csv                one CSV per table
    figures/*.png        one idea per figure
The narrative report (Luminance-GS/Implementation_guides/Rung0_results.md) is written by hand
from these outputs.
"""

import contextlib
import csv
import io
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import nnls

EXAMPLES = Path(__file__).resolve().parent
ROOT = EXAMPLES.parent / "results" / "rung0"
SUMMARY = ROOT / "summary"
FIG = SUMMARY / "figures"
DATA = EXAMPLES.parent / "data" / "LOM_full"

SCENES = ["buu", "chair", "sofa", "bike", "shrub"]
VARIANTS = ["baseline", "detach"]
SEEDS = [0, 1, 2]
KS = [4, 8, 16, 32, 64]
DEC_TOL = 1e-4
REP_SCENE, REP_SEED = "buu", 0
TIMING_EXCLUDE = {"sofa_detach_seed1"}  # PC slept during this run; wall time is meaningless

# --- style: reference palette (dataviz skill, light surface), fixed categorical order
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SCENE_COLOR = dict(zip(SCENES, SERIES))
VARIANT_COLOR = {"baseline": SERIES[0], "detach": SERIES[1]}
VARIANT_MARKER = {"baseline": "o", "detach": "s"}
SET_COLOR = {"pseudo": SERIES[0], "render": SERIES[1], "gt": SERIES[2]}
SET_LABEL = {"pseudo": "pseudo-targets", "render": "enhanced renders", "gt": "ground truth (high/)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2.0, "lines.markersize": 6, "legend.frameon": False,
})


def run_name(scene, variant, seed):
    return f"{scene}_{variant}_seed{seed}"


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    if not rows:
        return
    with open(SUMMARY / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        return "nan" if np.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def md_table(rows, nd=3):
    if not rows:
        return "_(empty)_\n"
    keys = list(rows[0].keys())
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    out += ["| " + " | ".join(fmt(r[k], nd) for k in keys) + " |" for r in rows]
    return "\n".join(out) + "\n"


def mean_std(x):
    x = np.asarray(x, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ loading
def load_run(scene, variant, seed):
    d = ROOT / run_name(scene, variant, seed)
    dd = d / "diagnostics"
    curves = {}
    for f in sorted(dd.glob("curves_step*.npz")):
        z = np.load(f)
        curves[int(z["step"])] = {k: z[k] for k in z.files}
    return {
        "name": d.name, "scene": scene, "variant": variant, "seed": seed, "dir": d,
        "val": json.loads((d / "stats" / "val_step9999.json").read_text()),
        "curves": curves,
        "pseudo": read_csv(dd / "pseudo_targets.csv"),
        "consistency": read_csv(dd / "consistency.csv"),
        "loss": read_csv(dd / "loss_log.csv"),
    }


def load_seed42_reference():
    """Per-run seed-42 values from average_results.md (buu, chair, sofa only)."""
    ref = defaultdict(list)
    path = EXAMPLES.parent / "results" / "reproduction" / "average_results.md"
    for line in path.read_text().splitlines():
        m = re.match(r"\|\s*(\w+)\s*\|\s*(\d)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|", line)
        if m:
            ref[m.group(1)].append({"psnr": float(m.group(3)), "ssim": float(m.group(4)), "lpips": float(m.group(5))})
    return ref


def load_val_gt(scene):
    """Undistorted normal-light validation images, in the order eval() renders them."""
    sys.path.insert(0, str(EXAMPLES))
    from datasets.colmap import Dataset, Parser
    with contextlib.redirect_stdout(io.StringIO()):
        parser = Parser(data_dir=str(DATA / scene), exp_name="low", method="", factor=1, normalize=True, test_every=8)
        valset = Dataset(parser, split="val")
        return [valset[i]["image"].numpy() / 255.0 for i in range(len(valset))]


def psnr(a, b):
    return float(10 * np.log10(1.0 / np.mean((a - b) ** 2)))


def psnr_gt_mean(pred, gt):
    return psnr(np.clip(pred * (gt.mean() / max(pred.mean(), 1e-8)), 0, 1), gt)


# ------------------------------------------------------------- curve maths
IDENTITY = np.linspace(0, 1, 255)


def monotonicity(curve, hit=None):
    """Decreasing steps and negative mass, optionally on the subsequence of hit bins."""
    c = curve if hit is None else curve[hit]
    d = np.diff(c)
    return int((d < -DEC_TOL).sum()), float(np.clip(-d, 0, None).sum())


def pl_basis(knots, n=255):
    """Ramp basis for a piecewise-linear curve with the given knot positions (incl. 0 and 1)."""
    x = np.linspace(0, 1, n)
    t0, t1 = np.asarray(knots[:-1]), np.asarray(knots[1:])
    return np.clip((x[:, None] - t0[None, :]) / (t1 - t0)[None, :], 0, 1)  # [n, K]


def uniform_knots(K):
    return np.linspace(0, 1, K + 1)


def used_range_knots(K, top):
    """K segments: K-1 equal segments over the used input range [0, top], one tail segment to 1."""
    return np.concatenate([np.linspace(0, top, K), [1.0]])


def fit_pl(y, knots, monotone):
    """Least-squares piecewise-linear fit with the given knots (len(knots)-1 segments).
    Monotone: non-negative increments (NNLS, free intercept as +1/-1 columns)."""
    B = pl_basis(knots, len(y))
    if monotone:
        A = np.concatenate([np.ones((len(y), 1)), -np.ones((len(y), 1)), B], 1)
        coef, _ = nnls(A, y)
        return A @ coef
    A = np.concatenate([np.ones((len(y), 1)), B], 1)
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    return A @ coef


# ------------------------------------------------------------------ analysis
def main():
    SUMMARY.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    runs = {}
    for sc in SCENES:
        for v in VARIANTS:
            for s in SEEDS:
                if (ROOT / run_name(sc, v, s) / "stats" / "val_step9999.json").exists():
                    runs[(sc, v, s)] = load_run(sc, v, s)
    print(f"loaded {len(runs)} runs")
    status = {r["run"]: r for r in read_csv(ROOT / "runs.csv")}
    ref42 = load_seed42_reference()
    tables, numbers = {}, {}
    steps = sorted(next(iter(runs.values()))["curves"])
    final = steps[-1]

    # ---- validation metrics (+ exposure-aligned PSNR from the saved val renders)
    val_rows = []
    for sc in SCENES:
        gts = load_val_gt(sc)
        for v in VARIANTS:
            for s in SEEDS:
                r = runs[(sc, v, s)]
                renders = [imageio.imread(r["dir"] / "renders" / f"val_{i:04d}_enh.png")[..., :3] / 255.0 for i in range(len(gts))]
                assert renders[0].shape == gts[0].shape, (renders[0].shape, gts[0].shape)
                r["val_psnr_png"] = float(np.mean([psnr(p, g) for p, g in zip(renders, gts)]))
                r["val_psnr_gtmean"] = float(np.mean([psnr_gt_mean(p, g) for p, g in zip(renders, gts)]))
                r["val_mean_render"] = float(np.mean([p.mean() for p in renders]))
                r["val_mean_gt"] = float(np.mean([g.mean() for g in gts]))
                minutes = float(status[r["name"]]["minutes"])
                val_rows.append({
                    "scene": sc, "variant": v, "seed": s,
                    "psnr": r["val"]["psnr"], "ssim": r["val"]["ssim"], "lpips": r["val"]["lpips"],
                    "psnr_gtmean": r["val_psnr_gtmean"], "psnr_from_png": r["val_psnr_png"],
                    "mean_render": r["val_mean_render"], "mean_gt": r["val_mean_gt"],
                    "num_GS": r["val"]["num_GS"],
                    "minutes": float("nan") if r["name"] in TIMING_EXCLUDE else minutes,
                })
    write_csv("val_metrics_per_run", val_rows)
    tables["Validation metrics per run (step 10k)"] = val_rows

    # ---- Q5 seed variance (baseline) and Q4 detach ablation
    seed_rows, abl_rows = [], []
    for sc in SCENES:
        for v in VARIANTS:
            rs = [runs[(sc, v, s)] for s in SEEDS]
            row = {"scene": sc, "variant": v}
            for m in ("psnr", "ssim", "lpips"):
                mu, sd = mean_std([r["val"][m] for r in rs])
                row[f"{m}_mean"], row[f"{m}_std"] = mu, sd
            mu, sd = mean_std([r["val_psnr_gtmean"] for r in rs])
            row["psnr_gtmean_mean"], row["psnr_gtmean_std"] = mu, sd
            row["num_GS_mean"] = float(np.mean([r["val"]["num_GS"] for r in rs]))
            mins = [float(status[r["name"]]["minutes"]) for r in rs if r["name"] not in TIMING_EXCLUDE]
            row["minutes_mean"] = float(np.mean(mins))
            seed_rows.append(row)
        if sc in ref42:
            row = {"scene": sc, "variant": "seed42 (average_results.md)"}
            for m in ("psnr", "ssim", "lpips"):
                row[f"{m}_mean"], row[f"{m}_std"] = mean_std([x[m] for x in ref42[sc]])
            row.update({"psnr_gtmean_mean": float("nan"), "psnr_gtmean_std": float("nan"),
                        "num_GS_mean": float("nan"), "minutes_mean": float("nan")})
            seed_rows.append(row)
        # paired detach - baseline
        arow = {"scene": sc}
        for m in ("psnr", "ssim", "lpips", "psnr_gtmean"):
            get = (lambda r: r["val_psnr_gtmean"]) if m == "psnr_gtmean" else (lambda r, m=m: r["val"][m])
            diffs = np.array([get(runs[(sc, "detach", s)]) - get(runs[(sc, "baseline", s)]) for s in SEEDS])
            base_sd = mean_std([get(runs[(sc, "baseline", s)]) for s in SEEDS])[1]
            det_sd = mean_std([get(runs[(sc, "detach", s)]) for s in SEEDS])[1]
            noise = np.sqrt((base_sd ** 2 + det_sd ** 2) / 2)  # pooled per-run seed std
            arow[f"d_{m}_mean"] = float(diffs.mean())
            arow[f"d_{m}_std"] = float(diffs.std(ddof=1))
            arow[f"d_{m}_over_noise"] = float(diffs.mean() / noise) if noise > 0 else float("nan")
        abl_rows.append(arow)
    write_csv("seed_variance", seed_rows)
    write_csv("detach_ablation", abl_rows)
    tables["Seed variance: mean ± sample std over seeds 0-2 (validation views, step 10k)"] = seed_rows
    tables["Detach ablation: paired differences detach − baseline (n = 3 seeds); over_noise = mean diff / pooled per-run seed std"] = abl_rows
    numbers["seed_psnr_std_baseline"] = {r["scene"]: r["psnr_std"] for r in seed_rows if r["variant"] == "baseline"}
    numbers["seed_psnr_gtmean_std_baseline"] = {r["scene"]: r["psnr_gtmean_std"] for r in seed_rows if r["variant"] == "baseline"}
    numbers["detach_d_psnr"] = {r["scene"]: r["d_psnr_mean"] for r in abl_rows}
    numbers["detach_d_psnr_over_noise"] = {r["scene"]: r["d_psnr_over_noise"] for r in abl_rows}
    numbers["detach_d_lpips"] = {r["scene"]: r["d_lpips_mean"] for r in abl_rows}
    numbers["detach_d_psnr_gtmean"] = {r["scene"]: r["d_psnr_gtmean_mean"] for r in abl_rows}

    # ---- Q1 monotonicity over training; bin coverage; shape prior; spread
    mono_rows, cover_rows, prior_rows, spread_rows = [], [], [], []
    for (sc, v, s), r in runs.items():
        for st in steps:
            c = r["curves"][st]
            V = len(c["view_names"])
            hit = c["bin_hist"] > 0
            m_all = [monotonicity(c["curve_adj"][i]) for i in range(V)]
            m_hit = [monotonicity(c["curve_adj"][i], hit[i]) for i in range(V)]
            pre = c["curve_preclamp"]
            mono_rows.append({
                "scene": sc, "variant": v, "seed": s, "step": st,
                "n_dec_mean": np.mean([a for a, _ in m_all]), "n_dec_max": max(a for a, _ in m_all),
                "neg_mass_mean": np.mean([b for _, b in m_all]), "neg_mass_max": max(b for _, b in m_all),
                "n_dec_hit_mean": np.mean([a for a, _ in m_hit]), "n_dec_hit_max": max(a for a, _ in m_hit),
                "neg_mass_hit_mean": np.mean([b for _, b in m_hit]), "neg_mass_hit_max": max(b for _, b in m_hit),
                "clamp_frac_mean": float(((pre < 0) | (pre > 1)).mean()),
                "clamp_frac_max": float(((pre < 0) | (pre > 1)).mean(1).max()),
                "clamp_lo_frac_mean": float((pre < 0).mean()),
                "clamp_hi_frac_mean": float((pre > 1).mean()),
                "pseudo_curve_n_dec_mean": float(np.mean([monotonicity(p)[0] for p in c["pesdo_curve"]])),
                "views_gamma_neg_frac": float((c["gamma"] < 0).mean()),
            })
            prior_rows.append({"scene": sc, "variant": v, "seed": s, "step": st,
                               **{f"{k}_{f}": float(getattr(np, f)(c[k])) for k in ("gamma", "alpha", "beta") for f in ("mean", "min", "max")}})
        c = r["curves"][final]
        hist = c["bin_hist"].astype(float)
        cum = np.cumsum(hist, 1) / hist.sum(1, keepdims=True)
        cover_rows.append({
            "scene": sc, "variant": v, "seed": s,
            "bins_hit_frac_mean": float((hist > 0).mean()),
            "bins_hit_frac_min": float((hist > 0).mean(1).min()),
            "bin_99pct_mean": float(np.mean(np.argmax(cum >= 0.99, 1))),
            "top_bin_hit_mean": float(np.mean([np.nonzero(h)[0].max() for h in hist])),
            # dark end of the curve: where the darkest input level is mapped
            "curve_adj_bin0": float(c["curve_adj"][:, 0].mean()),
            "pseudo_curve_bin0": float(c["pesdo_curve"][:, 0].mean()),
            "px_frac_bin0": float((hist[:, 0] / hist.sum(1)).mean()),
        })
        ca = c["curve_adj"]
        bias_norm = np.linalg.norm(c["curve_bias"], axis=1)
        glob_norm = np.linalg.norm(c["curve_global"] - IDENTITY)
        spread_rows.append({
            "scene": sc, "variant": v, "seed": s,
            "perbin_std_x255": float(ca.std(0).mean() * 255),
            "perbin_std_hit_x255": float(np.mean([ca[hist[:, b] > 0, b].std() for b in range(255) if (hist[:, b] > 0).sum() > 1]) * 255),
            "bias_over_global_dev": float(bias_norm.mean() / max(glob_norm, 1e-12)),
            "bias_viewmean_rms_x255": float(np.sqrt(np.mean(c["curve_bias"].mean(0) ** 2)) * 255),
            "bias_viewstd_x255": float(c["curve_bias"].std(0).mean() * 255),
            "global_dev_rms_x255": float(np.sqrt(np.mean((c["curve_global"] - IDENTITY) ** 2)) * 255),
        })
    write_csv("monotonicity_per_run_step", mono_rows)
    write_csv("shape_prior_per_run_step", prior_rows)
    write_csv("bin_coverage", cover_rows)
    write_csv("cross_view_spread", spread_rows)

    def agg(rows, keys, by=("scene", "variant"), extra=None):
        groups = defaultdict(list)
        for row in rows:
            if extra is None or extra(row):
                groups[tuple(row[b] for b in by)].append(row)
        out = []
        for g, rs in groups.items():
            o = dict(zip(by, g))
            for k in keys:
                vals = [x[k] for x in rs]
                o[k] = float(np.max(vals)) if k.endswith("_max") else float(np.mean(vals))
            out.append(o)
        return out

    mono_final = agg(mono_rows, ["n_dec_mean", "n_dec_max", "neg_mass_mean", "neg_mass_max", "n_dec_hit_mean",
                                 "n_dec_hit_max", "neg_mass_hit_mean", "neg_mass_hit_max", "clamp_frac_mean", "clamp_frac_max",
                                 "clamp_lo_frac_mean", "clamp_hi_frac_mean"],
                     extra=lambda x: x["step"] == final)
    pseudo_mono = agg(mono_rows, ["pseudo_curve_n_dec_mean", "views_gamma_neg_frac"], extra=lambda x: x["step"] == final)
    tables[f"Pseudo (shape-prior) curve at step {final}: decreasing steps per view, fraction of views with gamma < 0"] = pseudo_mono
    numbers["pseudo_mono"] = pseudo_mono
    tables[f"Q1 Monotonicity of curve_adj at step {final} (mean over views and seeds; _max = worst view). "
           f"neg_mass in curve units (1.0 = full range); _hit = only bins that pixels use"] = mono_final
    numbers["mono_final"] = mono_final
    cover = agg(cover_rows, ["bins_hit_frac_mean", "bins_hit_frac_min", "bin_99pct_mean", "top_bin_hit_mean",
                             "curve_adj_bin0", "pseudo_curve_bin0", "px_frac_bin0"])
    tables["Bin coverage at step 10k: fraction of the 255 LUT bins hit by any pixel; bin index below which 99% of pixels fall; "
           "curve output at bin 0 (black level; CDF target is 0 there)"] = cover
    numbers["coverage"] = cover
    spread = agg(spread_rows, ["perbin_std_x255", "perbin_std_hit_x255", "bias_over_global_dev", "global_dev_rms_x255",
                               "bias_viewmean_rms_x255", "bias_viewstd_x255"])
    tables["Cross-view spread at step 10k (units of 1/255): per-bin std of curve_adj across views; ||bias|| / ||global - identity||"] = spread
    numbers["spread"] = spread
    prior_final = agg(prior_rows, [f"{k}_{f}" for k in ("gamma", "alpha", "beta") for f in ("mean", "min", "max")],
                      extra=lambda x: x["step"] == final)
    for row in prior_final:  # min over runs for _min columns
        rs = [p for p in prior_rows if p["step"] == final and p["scene"] == row["scene"] and p["variant"] == row["variant"]]
        for k in ("gamma", "alpha", "beta"):
            row[f"{k}_min"] = min(p[f"{k}_min"] for p in rs)
    tables["Q3 Shape-prior parameters at step 10k (init: gamma 1, alpha 0.5, beta 1)"] = prior_final
    numbers["prior_final"] = prior_final

    # ---- Q2 PCA and compact monotone fit
    pca_rows, fit_rows = [], []
    for (sc, v, s), r in runs.items():
        c = r["curves"][final]
        X = c["curve_adj"].astype(float)
        Xc = X - X.mean(0)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        sv = S ** 2
        ratio = np.cumsum(sv) / sv.sum()
        pc1 = U[:, 0] * S[0]
        pca_rows.append({"scene": sc, "variant": v, "seed": s, "n_views": len(X),
                         "k95": int(np.argmax(ratio >= 0.95) + 1), "k99": int(np.argmax(ratio >= 0.99) + 1),
                         "pc1_frac": float(ratio[0]),
                         "total_std_x255": float(np.sqrt(sv.sum() / len(X) / X.shape[1]) * 255),
                         # |Pearson r| between the first component score and the view's mean input brightness
                         "abs_r_pc1_brightness": float(abs(np.corrcoef(pc1, c["mean_brightness"])[0, 1]))})
        hit = c["bin_hist"] > 0
        top = (np.nonzero(hit.any(0))[0].max() + 1) / 254.0  # upper end of the input range any view uses
        for K in KS:
            for knots_name, knots in (("uniform", uniform_knots(K)), ("used-range", used_range_knots(K, top))):
                for mono in (True, False):
                    if knots_name == "used-range" and not mono:
                        continue
                    errs, errs_hit, maxes_hit = [], [], []
                    for i in range(len(X)):
                        e = np.abs(fit_pl(X[i], knots, mono) - X[i]) * 255
                        errs.append(e.mean()); errs_hit.append(e[hit[i]].mean()); maxes_hit.append(e[hit[i]].max())
                    fit_rows.append({"scene": sc, "variant": v, "seed": s, "K": K, "knots": knots_name, "monotone": mono,
                                     "mae_x255": float(np.mean(errs)), "mae_hit_x255": float(np.mean(errs_hit)),
                                     "max_hit_x255": float(np.max(maxes_hit))})
    write_csv("pca_per_run", pca_rows)
    write_csv("pl_fit_per_run", fit_rows)
    pca = agg(pca_rows, ["n_views", "k95", "k99", "pc1_frac", "total_std_x255", "abs_r_pc1_brightness"])
    tables["Q2 PCA of final per-view curves (per run, averaged over seeds); total_std = RMS deviation from the mean curve, "
           "1/255 units; abs_r_pc1_brightness = |corr(PC1 score, mean input brightness)| over views"] = pca
    numbers["pca"] = pca
    fit = agg(fit_rows, ["mae_x255", "mae_hit_x255", "max_hit_x255"], by=("variant", "knots", "monotone", "K"))
    fit.sort(key=lambda x: (x["variant"], x["knots"] != "uniform", not x["monotone"], x["K"]))
    tables["Q2 Piecewise-linear fit of final curves with K segments, error in units of 1/255 (all scenes, views, seeds). "
           "mae = all 255 bins; _hit = bins pixels use; used-range = knots spread over the used input range + 1 tail segment"] = fit
    numbers["pl_fit"] = fit
    fit_scene = agg(fit_rows, ["mae_x255", "mae_hit_x255", "max_hit_x255"], by=("scene", "variant", "knots", "K"),
                    extra=lambda x: x["monotone"])
    write_csv("pl_fit_monotone_per_scene", fit_scene)

    # ---- Q6 pseudo-target quality, Q7 consistency
    pt_rows, cons_rows = [], []
    for (sc, v, s), r in runs.items():
        P = {k: np.array([float(x[k]) for x in r["pseudo"]]) for k in r["pseudo"][0] if k != "view"}
        pt_rows.append({"scene": sc, "variant": v, "seed": s, **{k: float(a.mean()) for k, a in P.items()}})
        C = {k: np.array([float(x[k]) for x in r["consistency"]]) for k in r["consistency"][0] if k not in ("view_i", "view_j")}
        cons_rows.append({"scene": sc, "variant": v, "seed": s, **{k: float(np.nanmean(a)) for k, a in C.items()}})
    write_csv("pseudo_targets_per_run", pt_rows)
    write_csv("consistency_per_run", cons_rows)
    pt_keys = ["mean_input", "mean_pseudo", "mean_render", "mean_gt", "pseudo_psnr", "render_psnr", "pseudo_psnr_gtmean",
               "render_psnr_gtmean", "pseudo_ssim", "render_ssim", "fit_psnr_render_vs_pseudo", "pseudo_out_of_range_frac"]
    pt = agg(pt_rows, pt_keys)
    tables["Q6 Pseudo-target quality on training views (oracle, step 10k): PSNR in dB vs normal-light GT; gtmean = exposure-aligned"] = pt
    numbers["pseudo_quality"] = pt
    cons = agg(cons_rows, ["valid_frac", "l1_pseudo", "l1_render", "l1_gt", "rel_l1_pseudo", "rel_l1_render", "rel_l1_gt"])
    for row in cons:
        row["pseudo_over_gt"] = row["rel_l1_pseudo"] / row["rel_l1_gt"]
    tables["Q7 Cross-view consistency (depth warp to 2 nearest training views, masked): rel_l1 = L1 / mean intensity"] = cons
    numbers["consistency"] = cons

    # =========================================================== figures
    def mono_over_steps(metric, title, fname, ylabel):
        fig, axes = plt.subplots(1, len(SCENES), figsize=(16, 3.4), sharey=True)
        for ax, sc in zip(axes, SCENES):
            for v in VARIANTS:
                mean, lo, hi = [], [], []
                for st in steps:
                    rows = [m for m in mono_rows if m["scene"] == sc and m["variant"] == v and m["step"] == st]
                    mean.append(np.mean([m[f"{metric}_mean"] for m in rows]))
                    lo.append(min(m[f"{metric}_mean"] for m in rows)); hi.append(max(m[f"{metric}_max"] for m in rows))
                ax.plot(steps, mean, color=VARIANT_COLOR[v], marker=VARIANT_MARKER[v], label=v)
                ax.fill_between(steps, lo, hi, color=VARIANT_COLOR[v], alpha=0.12, linewidth=0)
            ax.set_title(sc); ax.set_xlabel("training step")
        axes[0].set_ylabel(ylabel); axes[0].legend(loc="upper left")
        fig.suptitle(title, x=0.01, ha="left")
        fig.tight_layout()
        savefig(fig, fname)

    mono_over_steps("neg_mass", "Q1  Non-monotonicity of the per-view curve over training (line: mean over views and seeds; band: range to worst view)",
                    "q1_negative_mass_all_bins", "negative mass\n(curve units)")
    mono_over_steps("neg_mass_hit", "Q1  Same, counting only order inversions between bins that pixels actually use",
                    "q1_negative_mass_used_bins", "negative mass, used bins\n(curve units)")
    mono_over_steps("clamp_frac", "Fraction of curve entries outside [0, 1] before clamping",
                    "q1_clamp_fraction", "fraction clamped")

    # curve shapes, one panel per scene (baseline, REP_SEED)
    fig, axes = plt.subplots(1, len(SCENES), figsize=(16, 3.8), sharey=True)
    for ax, sc in zip(axes, SCENES):
        c = runs[(sc, "baseline", REP_SEED)]["curves"][final]
        x = np.arange(255)
        top = np.max([np.nonzero(h)[0].max() for h in c["bin_hist"]])
        ax.axvspan(0, top, color=GRID, alpha=0.5, linewidth=0)
        for i, y in enumerate(c["curve_adj"]):
            ax.plot(x, y, color=SERIES[0], linewidth=0.8, alpha=0.35, label="per-view curve_adj" if i == 0 else None)
        ax.plot(x, c["curve_global"], color=INK, linewidth=2, label="global curve")
        ax.plot(x, c["cdf_target"].mean(0), color=SERIES[1], linewidth=2, linestyle="--", label="HE-CDF target (view mean)")
        ax.plot(x, c["pesdo_curve"].mean(0), color=SERIES[2], linewidth=2, linestyle=":", label="pseudo curve (view mean)")
        ax.plot(x, IDENTITY, color=MUTED, linewidth=1, linestyle="-.", label="identity")
        ax.set_title(sc); ax.set_xlabel("LUT bin (input level × 254)")
    axes[0].set_ylabel("output value"); axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Learned curves at step {final} (baseline, seed {REP_SEED}); shaded = bins that any pixel uses", x=0.01, ha="left")
    fig.tight_layout(); savefig(fig, "q1_curve_shapes")

    # Q2 fit error vs K
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    styles = [("uniform", True, "-", "uniform knots, monotone"), ("uniform", False, ":", "uniform knots, unconstrained"),
              ("used-range", True, "--", "knots over used range, monotone")]
    for ax, (col, title) in zip(axes, (("mae_x255", "error over all 255 bins"), ("mae_hit_x255", "error over the bins pixels use"))):
        for v in VARIANTS:
            for kn, mono, ls, lab in styles:
                rows = [f for f in fit if f["variant"] == v and f["knots"] == kn and f["monotone"] == mono]
                ax.plot([f["K"] for f in rows], [f[col] for f in rows], color=VARIANT_COLOR[v], linestyle=ls,
                        marker=VARIANT_MARKER[v], label=f"{v}: {lab}")
        ax.axhline(1.0, color=MUTED, linewidth=1)
        ax.text(KS[0], 1.0, "1/255 ", color=INK2, va="bottom", fontsize=9)
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS)
        ax.set_xlabel("number of piecewise-linear segments K"); ax.set_title(title)
    axes[0].set_ylabel("mean abs. error (units of 1/255)"); axes[1].legend(fontsize=8)
    fig.suptitle("Q2  How well a compact piecewise-linear curve reproduces the learned per-view curves (step 10k)", x=0.01, ha="left")
    fig.tight_layout(); savefig(fig, "q2_pl_fit_error_vs_K")

    # bin coverage: mean normalised histogram per scene
    fig, ax = plt.subplots(figsize=(7, 4))
    for sc in SCENES:
        h = runs[(sc, "baseline", REP_SEED)]["curves"][final]["bin_hist"].astype(float)
        h = (h / h.sum(1, keepdims=True)).mean(0)
        ax.plot(np.arange(255), np.where(h > 0, h, np.nan), color=SCENE_COLOR[sc], label=sc)
    ax.set_yscale("log"); ax.set_xlabel("LUT bin"); ax.set_ylabel("fraction of pixel-channels (log)")
    ax.set_title("Which LUT bins receive pixels (baseline, step 10k, mean over views)"); ax.legend()
    fig.tight_layout(); savefig(fig, "bin_usage")

    # Q3 shape prior over training
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, (k, init) in zip(axes, (("gamma", 1.0), ("alpha", 0.5), ("beta", 1.0))):
        for sc in SCENES:
            rows = [[p for p in prior_rows if p["scene"] == sc and p["variant"] == "baseline" and p["step"] == st] for st in steps]
            ax.plot(steps, [np.mean([p[f"{k}_mean"] for p in rs]) for rs in rows], color=SCENE_COLOR[sc], label=sc)
            ax.fill_between(steps, [min(p[f"{k}_min"] for p in rs) for rs in rows], [max(p[f"{k}_max"] for p in rs) for rs in rows],
                            color=SCENE_COLOR[sc], alpha=0.12, linewidth=0)
        ax.axhline(init, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(f"{k} (init {init})"); ax.set_xlabel("training step")
    axes[0].legend(fontsize=9)
    fig.suptitle("Q3  Shape-prior parameters over training (baseline; line: mean, band: range over views and seeds)", x=0.01, ha="left")
    fig.tight_layout(); savefig(fig, "q3_shape_prior")

    # Q4/Q5 per-run validation metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (m, lab) in zip(axes, (("psnr", "PSNR (dB) ↑"), ("psnr_gtmean", "exposure-aligned PSNR (dB) ↑"), ("lpips", "LPIPS ↓"))):
        for j, sc in enumerate(SCENES):
            for k, v in enumerate(VARIANTS):
                vals = [r[m] for r in val_rows if r["scene"] == sc and r["variant"] == v]
                xpos = j + (k - 0.5) * 0.3
                ax.scatter([xpos] * len(vals), vals, color=VARIANT_COLOR[v], marker=VARIANT_MARKER[v], s=30, zorder=3,
                           label=v if j == 0 else None)
                ax.hlines(np.mean(vals), xpos - 0.1, xpos + 0.1, color=VARIANT_COLOR[v], linewidth=2)
            if m != "psnr_gtmean" and sc in ref42:
                ax.scatter([j + 0.45] * len(ref42[sc]), [x[m] for x in ref42[sc]], facecolors="none", edgecolors=MUTED,
                           marker="D", s=28, label="seed 42 (earlier runs)" if j == 0 else None)
        ax.set_xticks(range(len(SCENES))); ax.set_xticklabels(SCENES); ax.set_ylabel(lab); ax.grid(axis="x", visible=False)
    axes[0].legend(fontsize=9)
    fig.suptitle("Q4/Q5  Validation metrics per run (dots: seeds 0-2, bar: mean)", x=0.01, ha="left")
    fig.tight_layout(); savefig(fig, "q4_q5_val_metrics")

    # Q6 pseudo-target quality
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    for ax, (suffix, lab) in zip(axes, (("psnr", "raw PSNR vs GT (dB)"), ("psnr_gtmean", "exposure-aligned PSNR vs GT (dB)"))):
        for j, sc in enumerate(SCENES):
            for k, v in enumerate(VARIANTS):
                for q, (what, mk) in enumerate((("pseudo", "o"), ("render", "^"))):
                    vals = [p[f"{what}_{suffix}"] for p in pt_rows if p["scene"] == sc and p["variant"] == v]
                    xpos = j + (k - 0.5) * 0.4 + (q - 0.5) * 0.15
                    ax.scatter([xpos] * len(vals), vals, color=VARIANT_COLOR[v], marker=mk, s=26, zorder=3,
                               label=f"{v}: {'pseudo-target' if what == 'pseudo' else 'render'}" if j == 0 else None)
        ax.set_xticks(range(len(SCENES))); ax.set_xticklabels(SCENES); ax.set_title(lab); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("dB (training views)"); axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Q6  Pseudo-targets vs the enhanced render they train (oracle: normal-light GT of the training views)", x=0.01, ha="left")
    fig.tight_layout(); savefig(fig, "q6_pseudo_target_quality")

    # Q7 consistency
    fig, ax = plt.subplots(figsize=(9, 4))
    for j, sc in enumerate(SCENES):
        for q, key in enumerate(("pseudo", "render", "gt")):
            for k, v in enumerate(VARIANTS):
                vals = [c[f"rel_l1_{key}"] for c in cons_rows if c["scene"] == sc and c["variant"] == v]
                xpos = j + (q - 1) * 0.25 + (k - 0.5) * 0.09
                ax.scatter([xpos] * len(vals), vals, color=SET_COLOR[key], marker=VARIANT_MARKER[v], s=24, zorder=3,
                           label=f"{SET_LABEL[key]} ({v})" if j == 0 else None)
    ax.set_xticks(range(len(SCENES))); ax.set_xticklabels(SCENES); ax.grid(axis="x", visible=False)
    ax.set_ylabel("relative L1 after depth warp\n(L1 / mean intensity)")
    ax.set_title("Q7  Cross-view consistency (lower = more consistent; GT sets the natural level)")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    fig.tight_layout(); savefig(fig, "q7_consistency")

    # loss components (representative run)
    L = runs[(REP_SCENE, "baseline", REP_SEED)]["loss"]
    st = np.array([int(x["step"]) for x in L])
    g = lambda k: np.array([float(x[k]) for x in L])
    w = g("w_cdf")
    terms = {
        "L_low (regress dark)": g("loss_regress_low"),
        "0.5 · L_enh (fit pseudo-target)": 0.5 * g("loss_regress_enh"),
        "L_spa (contrast)": g("loss_co"),
        "10 · w · CDF prior": 10 * w * g("hist_cdf"),
        "10 · 0.5 · pseudo-curve prior": 10 * 0.5 * g("hist_pseudo"),
        "10 · 0.1 · smoothness": 10 * 0.1 * g("hist_smooth"),
    }
    fig, ax = plt.subplots(figsize=(8, 4.2))
    palette = SERIES + ["#008300"]
    for (lab, y), col in zip(terms.items(), palette):
        k = 5
        ys = np.convolve(y, np.ones(k) / k, mode="same") if len(y) > k else y
        ax.plot(st, ys, color=col, label=lab, linewidth=1.6)
    ax.axvline(3000, color=MUTED, linewidth=1, linestyle="--")
    ax.text(3000, 1.0, " CDF weight 1 → 0.5", color=INK2, fontsize=9, va="bottom")
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("weighted loss term (log)")
    ax.set_title(f"Weighted loss terms over training ({REP_SCENE}, baseline, seed {REP_SEED}; 5-point moving average)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); savefig(fig, "loss_components")
    numbers["loss_terms_final_mean"] = {k: float(np.mean(v[-10:])) for k, v in terms.items()}
    numbers["loss_terms_first_mean"] = {k: float(np.mean(v[:10])) for k, v in terms.items()}

    # qualitative examples copied into the summary
    for src, dst in [
        (ROOT / "bike_baseline_seed0/diagnostics/example_29_in_pseudo_render_gt.png", "qual_bike_baseline.png"),
        (ROOT / "bike_detach_seed0/diagnostics/example_29_in_pseudo_render_gt.png", "qual_bike_detach.png"),
        (ROOT / "shrub_baseline_seed0/diagnostics/example_26_in_pseudo_render_gt.png", "qual_shrub_baseline.png"),
        (ROOT / "buu_baseline_seed0/diagnostics/example_17_in_pseudo_render_gt.png", "qual_buu_baseline.png"),
        (ROOT / "bike_baseline_seed0/diagnostics/warp_check.png", "warp_check_bike_baseline.png"),
    ]:
        if src.exists():
            shutil.copy(src, FIG / dst)

    # =========================================================== write
    with open(SUMMARY / "tables.md", "w", encoding="utf-8") as f:
        f.write("# Rung 0 tables (generated by examples/rung0_analyze.py)\n\n")
        for title, rows in tables.items():
            f.write(f"## {title}\n\n{md_table(rows)}\n")
    with open(SUMMARY / "numbers.json", "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
