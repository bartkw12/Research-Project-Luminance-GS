"""M1 analysis: the official code (`baseline`) and `detach` on the varying-exposure LOM set
(results/vexp/, data/LOM_vexp) against the same runs on LOM (results/rung0/). Writes tables,
figures and key numbers to results/vexp/summary/ and nowhere else.

    python vexp_analyze.py            (from Luminance-GS/examples, luminancegs env; a few minutes)

Spec: Implementation_guides/M1_vexp_plan.md, section 5.
    T1 validation on the varying set          T2 cost of varying exposure (varying - LOM)
    T3 detach - official, varying vs LOM      T4 exposure handling: fitted s_k vs u_k, curve
                                                 spread, pseudo-target/render leak, init and oracle
    T5 training-view quality, paired per view against signed u_k
    T6 cross-view consistency                 T7 colour and wall time
Effects are paired by seed; noise = sqrt(mean of the seed variances of the two arms compared);
an effect is called real above 2x noise (as rung05_analyze.effects). The narrative report
(Implementation_guides/M1_vexp_results.md) is written by hand from these outputs.

Inputs per run, both roots: cfg.json, stats/val_step9999.json, rescore/{val_summary.json,
pseudo_targets.csv, consistency.csv}, diagnostics/curves_step10000.npz (local only), runs.csv.
u_k and the realised log2_ratio from results/vexp/dataset/gains.json; the init and the
histogram oracle are recomputed on both sets exactly as build_vexp.verify_scene does (it stores
only the varying-set side), and the varying-set side is checked against oracle.json.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from rung0_analyze import (SCENES, SEEDS, SERIES, MUTED, INK2, TIMING_EXCLUDE,
                           read_csv, write_csv, md_table, mean_std, savefig, run_name)
from build_vexp import fit_exposure_form, slope_r

EXAMPLES = Path(__file__).resolve().parent
RESULTS = EXAMPLES.parent / "results"
LOM_ROOT = RESULTS / "rung0"
VEXP_ROOT = RESULTS / "vexp"
DATASET = VEXP_ROOT / "dataset"
OUT = VEXP_ROOT / "summary"
FIG = OUT / "figures"
LOM_DATA = EXAMPLES.parent / "data" / "LOM_full"
VEXP_DATA = EXAMPLES.parent / "data" / "LOM_vexp"

VARIANTS = ["baseline", "detach"]
LABEL = {"baseline": "official", "detach": "detach"}
VARIANT_COLOR = {"baseline": SERIES[0], "detach": SERIES[1]}  # same colours as Rung 0.5
VARIANT_MARKER = {"baseline": "o", "detach": "s"}
ORACLE_COLOR = SERIES[2]
ROOTS = {"vexp": VEXP_ROOT, "lom": LOM_ROOT}
REF_VARIANT, REF_SEED = "baseline", 0  # LOM <scene>_baseline_seed0 is the shared reference depth
REAL = 2.0
GROUPS = SCENES + ["pooled", "pooled_no_shrub"]
VAL_METRICS = ["psnr", "psnr_gtmean", "ssim", "lpips"]
COST_METRICS = VAL_METRICS + ["mean_render"]
COLOUR_METRICS = ["sat_ratio", "ang_err_deg", "hue_err_sat_deg"]
PSEUDO_METRICS = ["pseudo_psnr", "pseudo_psnr_gtmean", "fit_psnr_lowrender_vs_input"]
CONS_METRICS = ["rel_l1_pseudo", "rel_l1_render"]
ND = {"psnr": 2, "psnr_gtmean": 2, "ssim": 4, "lpips": 4, "mean_render": 3, "mean_gt": 3,
      "sat_ratio": 3, "ang_err_deg": 2, "hue_err_sat_deg": 2, "pseudo_psnr": 2, "pseudo_psnr_gtmean": 2,
      "fit_psnr_lowrender_vs_input": 2, "rel_l1_pseudo": 4, "rel_l1_render": 4, "rel_l1_gt": 4}


# ------------------------------------------------------------------ output guard
def _inside_out(d: Path) -> Path:
    """rung0_analyze.write_csv/savefig default to the protected results/rung0/summary/: only ever
    hand them a folder under results/vexp/summary/."""
    d = d.resolve()
    if not (d == OUT.resolve() or OUT.resolve() in d.parents):
        raise SystemExit(f"[vexp_analyze] refusing to write outside {OUT}: {d}")
    return d


def save_csv(name, rows):
    write_csv(name, rows, _inside_out(OUT))


def save_fig(fig, name):
    savefig(fig, name, _inside_out(FIG))


# ------------------------------------------------------------------ helpers
def group_scenes(g):
    return {"pooled": SCENES, "pooled_no_shrub": [s for s in SCENES if s != "shrub"]}.get(g, [g])


def seeds_of(fn, g, seeds=SEEDS):
    """Per-seed values of fn(scene, seed); a pooled group averages over its scenes per seed."""
    return [float(np.mean([fn(sc, s) for sc in group_scenes(g)])) for s in seeds]


def paired(a, b):
    """Effect a - b paired by seed, with noise = sqrt(mean of the two arms' seed variances)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    noise = float(np.sqrt(0.5 * (np.var(a, ddof=1) + np.var(b, ddof=1))))
    m = float(d.mean())
    return {"diff": m, "diff_sd": float(d.std(ddof=1)), "noise": noise,
            "x_noise": m / noise if noise > 0 else float("nan"),
            "real": bool(noise > 0 and abs(m) > REAL * noise)}


def pm(m, s, nd):
    return f"{m:.{nd}f} ± {s:.{nd}f}" if np.isfinite(s) else f"{m:.{nd}f}"


def fx(e, nd):
    x = f"{e['x_noise']:+.1f}×" if np.isfinite(e["x_noise"]) else "n/a"
    return f"{e['diff']:+.{nd}f} ({x}){' *' if e['real'] else ''}"


def rmse_c(x):
    """RMS of x around its own mean (the gauge of s_k is free, so only the shape counts)."""
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean((x - x.mean()) ** 2)))


def pc1_score(X):
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, 0] * S[0]


def as_path(p):
    p = Path(p)
    return (p if p.is_absolute() else EXAMPLES / p).resolve()


def last_ok_minutes(path):
    out = {}
    for r in read_csv(path):
        if r["ok"] == "True":
            out[r["run"]] = float(r["minutes"])  # later rows (retries) overwrite earlier ones
    return out


# ------------------------------------------------------------------ loading and checks
def load(root, scene, variant, seed):
    d = root / run_name(scene, variant, seed)
    rs = d / "rescore"
    z = np.load(d / "diagnostics" / "curves_step10000.npz")
    return {
        "dir": d, "cfg": json.loads((d / "cfg.json").read_text()),
        "val": json.loads((rs / "val_summary.json").read_text()),
        "stats": json.loads((d / "stats" / "val_step9999.json").read_text()),
        "pseudo": {r["view"]: r for r in read_csv(rs / "pseudo_targets.csv")},
        "cons": read_csv(rs / "consistency.csv"),
        "view_names": [str(v) for v in z["view_names"]],
        "curve_adj": z["curve_adj"].astype(np.float64), "bin_hist": z["bin_hist"].astype(np.float64),
    }


def check_inputs(runs, vexp_status):
    """Stop before any table if a run used the wrong data, flags or reference depth."""
    bad = []
    for (root_name, sc, v, s), r in runs.items():
        name = f"{ROOTS[root_name].name}/{run_name(sc, v, s)}"
        want = ((VEXP_DATA if root_name == "vexp" else LOM_DATA) / sc).resolve()
        if as_path(r["cfg"]["data_dir"]) != want:
            bad.append(f"{name}: data_dir {r['cfg']['data_dir']} is not {want}")
        ref = (LOM_ROOT / run_name(sc, REF_VARIANT, REF_SEED)).resolve()
        if as_path(r["val"]["ref_depth_run"]) != ref:
            bad.append(f"{name}: re-scored with reference depth {r['val']['ref_depth_run']}, not {ref} "
                       f"(re-score with rescore_runs.py --ref_root ../results/rung0)")
        c = r["cfg"]
        if (c["max_steps"] != 10_000 or bool(c.get("detach_pseudo_target")) != (v == "detach")
                or c.get("freeze_color_basis") or c.get("detach_curve_in_target")):
            bad.append(f"{name}: unexpected flags in cfg.json")
        if root_name == "vexp" and run_name(sc, v, s) not in vexp_status:
            bad.append(f"{name}: no ok row in results/vexp/runs.csv")
    if bad:
        raise SystemExit("[vexp_analyze] input check failed:\n  " + "\n  ".join(bad))


def histogram_oracle(scene):
    """The init log2(m_bar/m_k) (geometric mean) and the histogram oracle on LOM and on the varying set,
    computed exactly as build_vexp.verify_scene does, which keeps only the varying-set side."""
    import torch
    sys.path.insert(0, str(EXAMPLES))
    from datasets.colmap import Dataset, Parser
    from losses import HistogramPriorLoss

    hist_loss = HistogramPriorLoss()
    out = {}
    for key, root in (("lom", LOM_DATA), ("vexp", VEXP_DATA)):
        with contextlib.redirect_stdout(io.StringIO()):
            p = Parser(data_dir=str(root / scene), exp_name="low", method="", factor=1, normalize=True, test_every=8)
            ds = Dataset(p, split="train")
        names, m, cdfs, hists = [], [], [], []
        for i in range(len(ds)):
            img = ds[i]["image"][None] / 255.0  # undistorted exactly as in training
            names.append(p.image_names[int(ds.indices[i])])
            m.append(float(img.mean()))
            cdfs.append(hist_loss.compute_histogram_equalization(img)[0].numpy())
            grey = torch.mean(torch.nn.functional.interpolate(img.permute(0, 3, 1, 2), scale_factor=0.25), dim=1)
            hists.append(torch.histc(grey.flatten(), bins=255, min=0.0, max=1.0).numpy())
        m = np.array(m)
        s_or, _, _ = fit_exposure_form(np.stack(cdfs), np.stack(hists))
        out[key] = {"names": names, "init": np.log2(m).mean() - np.log2(m), "oracle": s_or}
    assert out["lom"]["names"] == out["vexp"]["names"], f"{scene}: view order differs between LOM and LOM_vexp"
    return out


# ------------------------------------------------------------------ exposure statistics
def recovery_stats(s_v, s_l, u, ue):
    """How well per-view exposures s track the applied gains: ideal is s = -u + const.
    Unpaired uses s on the varying set; paired uses s(varying) - s(LOM) for the same view."""
    out = {}
    for tag, x in (("", u), ("_ueff", ue)):
        out[f"slope{tag}"], out[f"r{tag}"] = slope_r(s_v, x)
        out[f"slope_paired{tag}"], out[f"r_paired{tag}"] = slope_r(s_v - s_l, x)
        out[f"rmse{tag}"] = rmse_c(s_v + x)
        out[f"rmse_paired{tag}"] = rmse_c(s_v - s_l + x)
    out["sd_lom"] = float(np.std(s_l, ddof=1))
    return out


def exposure_run(rv, rl, u_by, ueff_by):
    names = rv["view_names"]
    assert names == rl["view_names"], "view order differs between the varying and the LOM run"
    assert set(names) == set(rv["pseudo"]) == set(rl["pseudo"]) == set(u_by), "view names do not join"
    u = np.array([u_by[n] for n in names])
    ue = np.array([ueff_by[n] for n in names])
    s_v, spread_v, resid_v = fit_exposure_form(rv["curve_adj"], rv["bin_hist"])
    s_l, spread_l, resid_l = fit_exposure_form(rl["curve_adj"], rl["bin_hist"])
    rec = recovery_stats(s_v, s_l, u, ue)
    rec["pc1_abs_r"] = float(abs(np.corrcoef(pc1_score(rv["curve_adj"]), u)[0, 1]))
    rec.update(spread_vexp=spread_v, spread_lom=spread_l, resid_vexp=resid_v, resid_lom=resid_l,
               explained_vexp=1 - (resid_v / spread_v) ** 2, explained_lom=1 - (resid_l / spread_l) ** 2)

    def col(r, k):
        return np.array([float(r["pseudo"][n][k]) for n in names])

    views = {"names": names, "u": u, "u_eff": ue, "s_v": s_v, "s_l": s_l}
    for k, key in (("pseudo", "mean_pseudo"), ("render", "mean_render"), ("input", "mean_input")):
        a, b = col(rv, key), col(rl, key)
        rec[f"{k}_leak"] = slope_r(np.log2(a), u)[0]  # 0 = brightness independent of u, 1 = passes through
        rec[f"{k}_leak_paired"] = slope_r(np.log2(a / b), u)[0]
        views[f"log2_{k}_ratio"] = np.log2(a / b)
    for m in PSEUDO_METRICS:
        views[f"{m}_vexp"], views[f"{m}_lom"] = col(rv, m), col(rl, m)
    return rec, views


# ------------------------------------------------------------------ main
def main():
    _inside_out(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    gains = json.loads((DATASET / "gains.json").read_text())
    u_by = {sc: {r["view"]: r["u"] for r in gains["rows"] if r["scene"] == sc} for sc in SCENES}
    ueff_by = {sc: {r["view"]: r["log2_ratio"] for r in gains["rows"] if r["scene"] == sc} for sc in SCENES}
    oracle_json = {r["scene"]: r for r in json.loads((DATASET / "oracle.json").read_text())["rows"]}

    runs, missing = {}, []
    for root_name, root in ROOTS.items():
        for sc in SCENES:
            for v in VARIANTS:
                for s in SEEDS:
                    try:
                        runs[(root_name, sc, v, s)] = load(root, sc, v, s)
                    except FileNotFoundError as e:
                        missing.append(f"{root.name}/{run_name(sc, v, s)} ({Path(e.filename).name})")
    if missing:
        raise SystemExit(f"[vexp_analyze] missing inputs: {missing}")
    vexp_minutes = last_ok_minutes(VEXP_ROOT / "runs.csv")
    lom_minutes = last_ok_minutes(LOM_ROOT / "runs.csv")
    check_inputs(runs, vexp_minutes)
    print(f"[vexp_analyze] loaded {len(runs)} runs; data_dir, flags and reference depth checked", flush=True)

    tables, numbers = {}, {"dataset_meta": gains["meta"]}
    integrity = []

    def check(name, value, tol):
        integrity.append({"check": name, "value": f"{value:.2e}", "tolerance": f"{tol:.0e}", "ok": bool(value <= tol)})

    # ---- integrity: re-score vs training-time eval, rel_l1_gt identical under the shared depth
    for m in ("psnr", "ssim", "lpips"):
        check(f"max |rescore − stats| {m}, all 60 runs",
              max(abs(r["val"][m] - r["stats"][m]) for r in runs.values()), 1e-3 if m == "psnr" else 1e-4)
    gt_diff = 0.0
    for sc in SCENES:
        ref = {(c["view_i"], c["view_j"]): float(c["rel_l1_gt"])
               for c in runs[("lom", sc, REF_VARIANT, REF_SEED)]["cons"]}
        for (root_name, sc2, v, s), r in runs.items():
            if sc2 == sc:
                gt_diff = max(gt_diff, max(abs(float(c["rel_l1_gt"]) - ref[(c["view_i"], c["view_j"])]) for c in r["cons"]))
    check("max |rel_l1_gt − LOM reference run| (same warp for every run)", gt_diff, 1e-5)

    # ---- init and histogram oracle on both sets
    oracle = {}
    for sc in SCENES:
        o = histogram_oracle(sc)
        oj = oracle_json[sc]
        assert o["vexp"]["names"] == oj["view_names"], f"{sc}: oracle.json view order differs"
        check(f"{sc}: u in gains.json vs oracle.json", float(np.abs(np.array([u_by[sc][n] for n in oj["view_names"]]) - oj["u"]).max()), 1e-12)
        check(f"{sc}: recomputed init vs oracle.json s_init (stops)", float(np.abs(o["vexp"]["init"] - oj["s_init"]).max()), 1e-6)
        check(f"{sc}: recomputed oracle vs oracle.json s_oracle (stops)", float(np.abs(o["vexp"]["oracle"] - oj["s_oracle"]).max()), 2e-3)
        oracle[sc] = o
        print(f"[vexp_analyze] oracle {sc} done", flush=True)
    if not all(c["ok"] for c in integrity):
        print(md_table(integrity))
        raise SystemExit("[vexp_analyze] integrity check failed (table above)")
    tables["Integrity checks"] = integrity
    numbers["integrity"] = integrity
    numbers["u_sd"] = {sc: float(np.std(list(u_by[sc].values()), ddof=1)) for sc in SCENES}

    def val(root_name, v, m):
        return lambda sc, s: runs[(root_name, sc, v, s)]["val"][m]

    # ---- T1 validation on the varying set
    t1, t1_md = [], []
    for g in GROUPS:
        for v in VARIANTS:
            row, md = {"scene": g, "variant": LABEL[v]}, {"scene": g, "variant": LABEL[v]}
            for m in VAL_METRICS + ["mean_render", "mean_gt"]:
                mu, sd = mean_std(seeds_of(val("vexp", v, m), g))
                row[f"{m}_mean"], row[f"{m}_sd"] = mu, sd
                md[m] = pm(mu, sd, ND[m])
            t1.append(row)
            t1_md.append(md)
    save_csv("T1_val_varying", t1)
    tables["T1. Validation on the varying-exposure set (mean ± sd over seeds 0-2; pooled = mean over scenes per seed). "
           "psnr_gtmean = brightness-matched PSNR; mean_render / mean_gt = mean brightness of the validation renders / GT"] = t1_md
    numbers["T1"] = t1

    # ---- T2 cost of varying exposure
    t2, t2_md = [], []
    for g in GROUPS:
        for v in VARIANTS:
            md = {"scene": g, "variant": LABEL[v]}
            for m in COST_METRICS:
                a, b = seeds_of(val("vexp", v, m), g), seeds_of(val("lom", v, m), g)
                e = paired(a, b)
                t2.append({"scene": g, "variant": LABEL[v], "metric": m,
                           "lom_mean": float(np.mean(b)), "vexp_mean": float(np.mean(a)), **e})
                md[m] = fx(e, ND[m])
            t2_md.append(md)
    save_csv("T2_cost_varying_minus_lom", t2)
    tables["T2. Cost of varying exposure: varying − LOM, same variant, scene and seed (mean difference, ×noise; "
           "* = |difference| > 2× noise)"] = t2_md
    numbers["T2"] = t2

    # ---- T3 detach - official, varying vs LOM (the H2 bar)
    t3, t3_md = [], []
    for g in GROUPS:
        for m in VAL_METRICS:
            dv = np.subtract(seeds_of(val("vexp", "detach", m), g), seeds_of(val("vexp", "baseline", m), g))
            dl = np.subtract(seeds_of(val("lom", "detach", m), g), seeds_of(val("lom", "baseline", m), g))
            ev = paired(seeds_of(val("vexp", "detach", m), g), seeds_of(val("vexp", "baseline", m), g))
            el = paired(seeds_of(val("lom", "detach", m), g), seeds_of(val("lom", "baseline", m), g))
            eg = paired(dv, dl)
            t3.append({"scene": g, "metric": m, **{f"varying_{k}": x for k, x in ev.items()},
                       **{f"lom_{k}": x for k, x in el.items()}, **{f"change_{k}": x for k, x in eg.items()}})
            t3_md.append({"scene": g, "metric": m, "detach − official, varying": fx(ev, ND[m]),
                          "detach − official, LOM": fx(el, ND[m]), "change (varying − LOM)": fx(eg, ND[m])})
    save_csv("T3_detach_minus_official", t3)
    tables["T3. detach − official on the varying set and on LOM, paired by seed (×noise; * = real). change = the "
           "difference of the two gaps, noise = pooled seed std of the per-seed gaps. The H2 bar is detach on the varying set"] = t3_md
    numbers["T3"] = t3

    # ---- T4 exposure handling (per run, then over seeds)
    expo, expo_views, per_view = {}, {}, []
    for sc in SCENES:
        o = oracle[sc]
        ref = {k: dict(zip(o["vexp"]["names"], o["vexp"][k])) for k in ("init", "oracle")}
        ref_l = {k: dict(zip(o["lom"]["names"], o["lom"][k])) for k in ("init", "oracle")}
        for v in VARIANTS:
            for s in SEEDS:
                rec, views = exposure_run(runs[("vexp", sc, v, s)], runs[("lom", sc, v, s)], u_by[sc], ueff_by[sc])
                expo[(sc, v, s)], expo_views[(sc, v, s)] = rec, views
                for i, n in enumerate(views["names"]):
                    row = {"scene": sc, "variant": LABEL[v], "seed": s, "view": n,
                           "u": float(views["u"][i]), "u_eff": float(views["u_eff"][i]),
                           "s_fit_vexp": float(views["s_v"][i]), "s_fit_lom": float(views["s_l"][i]),
                           "init_vexp": float(ref["init"][n]), "init_lom": float(ref_l["init"][n]),
                           "oracle_vexp": float(ref["oracle"][n]), "oracle_lom": float(ref_l["oracle"][n])}
                    for k in ("pseudo", "render", "input"):
                        row[f"log2_{k}_ratio"] = float(views[f"log2_{k}_ratio"][i])
                    for m in PSEUDO_METRICS:
                        row[f"{m}_vexp"] = float(views[f"{m}_vexp"][i])
                        row[f"{m}_lom"] = float(views[f"{m}_lom"][i])
                        row[f"{m}_diff"] = row[f"{m}_vexp"] - row[f"{m}_lom"]
                    per_view.append(row)
        print(f"[vexp_analyze] exposure fits {sc} done", flush=True)
    save_csv("per_view_paired", per_view)
    save_csv("T4_exposure_per_run", [{"scene": sc, "variant": LABEL[v], "seed": s, **rec} for (sc, v, s), rec in expo.items()])

    refs = {}
    for sc in SCENES:
        o = oracle[sc]
        names = o["vexp"]["names"]
        u = np.array([u_by[sc][n] for n in names])
        ue = np.array([ueff_by[sc][n] for n in names])
        for method in ("init", "oracle"):
            refs[(sc, method)] = recovery_stats(o["vexp"][method], o["lom"][method], u, ue)

    keys_a = ["slope", "r", "rmse", "slope_paired", "r_paired", "rmse_paired", "slope_ueff", "rmse_ueff",
              "rmse_paired_ueff", "sd_lom"]
    keys_b = ["pc1_abs_r", "spread_vexp", "spread_lom", "explained_vexp", "explained_lom", "pseudo_leak",
              "pseudo_leak_paired", "render_leak", "render_leak_paired", "input_leak_paired"]
    t4, t4a_md, t4b_md = [], [], []
    for g in SCENES + ["pooled"]:
        for method in VARIANTS + ["init", "oracle"]:
            row = {"scene": g, "method": LABEL.get(method, method)}
            for k in keys_a + (keys_b if method in VARIANTS else []):
                if method in VARIANTS:
                    mu, sd = mean_std(seeds_of(lambda sc, s, k=k, v=method: expo[(sc, v, s)][k], g))
                else:
                    mu, sd = float(np.mean([refs[(sc, method)][k] for sc in group_scenes(g)])), float("nan")
                row[f"{k}_mean"], row[f"{k}_sd"] = mu, sd
            t4.append(row)
            nd = {"slope": 2, "r": 3, "rmse": 3, "sd": 3}  # by the key's first word
            t4a_md.append({"scene": g, "method": row["method"], **{
                k: pm(row[f"{k}_mean"], row[f"{k}_sd"], nd[k.split("_")[0]]) for k in keys_a}})
            if method in VARIANTS:
                t4b_md.append({"scene": g, "variant": row["method"], **{
                    k: pm(row[f"{k}_mean"], row[f"{k}_sd"], 2 if k.startswith(("spread", "explained")) else 3)
                    for k in keys_b}})
    save_csv("T4_exposure", t4)
    tables["T4a. Exposure recovery against the applied gain u_k (ideal: slope −1, r −1, rmse 0). official/detach = "
           "s_k fitted to the final per-view curves (curve_adj, weights bin_hist) with build_vexp.fit_exposure_form; "
           "init = log2(m̄/m_k), geometric mean; oracle = the same fit on the HE-CDF targets. paired = s(varying) − s(LOM) "
           "for the same view (and seed). rmse = RMS of s + u_k around its mean, in stops. _ueff = with the realised "
           "log2_ratio from gains.csv in place of u_k. sd_lom = spread of s on LOM (natural exposure spread)"] = t4a_md
    tables["T4b. Curve structure and brightness leak. pc1_abs_r = |r(PC1 score of curve_adj, u_k)| on the varying set; "
           "spread / explained = cross-view curve spread (×255) and the share of it the exposure form explains, varying vs LOM; "
           "leak = slope of log2 mean brightness on u_k (0 = fully compensated, 1 = passes straight through) for the "
           "pseudo-targets and the enhanced renders at the training views, unpaired and paired (varying / LOM, same view); "
           "input_leak_paired = the same for the dark inputs (check: should be about 1)"] = t4b_md
    numbers["T4"] = t4

    # ---- T5 training-view quality, paired per view against signed u
    def view_mean(root_name, v, m):
        return lambda sc, s: float(np.nanmean([float(x[m]) for x in runs[(root_name, sc, v, s)]["pseudo"].values()]))

    t5a, t5a_md = [], []
    for g in GROUPS:
        for v in VARIANTS:
            md = {"scene": g, "variant": LABEL[v]}
            for m in PSEUDO_METRICS:
                a, b = seeds_of(view_mean("vexp", v, m), g), seeds_of(view_mean("lom", v, m), g)
                e = paired(a, b)
                t5a.append({"scene": g, "variant": LABEL[v], "metric": m, "lom_mean": float(np.mean(b)),
                            "lom_sd": float(np.std(b, ddof=1)), "vexp_mean": float(np.mean(a)),
                            "vexp_sd": float(np.std(a, ddof=1)), **e})
                md[m] = f"{np.mean(b):.2f} → {np.mean(a):.2f}, {fx(e, 2)}"
            t5a_md.append(md)
    save_csv("T5a_training_views", t5a)
    tables["T5a. Training-view quality, LOM → varying (mean over views, then seeds), and the paired difference (×noise; * = real). "
           "pseudo_psnr(_gtmean) = pseudo-target vs normal-light GT (raw / brightness-matched); "
           "fit_psnr_lowrender_vs_input = shared dark render vs the dark input photo (the dark-branch fit)"] = t5a_md
    numbers["T5a"] = t5a

    t5b = []
    for g in SCENES + ["pooled"]:
        for v in VARIANTS:
            for m in PSEUDO_METRICS:
                for sign in ("u<0", "u>0"):
                    pts = np.array([(r["u"], r[f"{m}_diff"]) for r in per_view
                                    if r["scene"] in group_scenes(g) and r["variant"] == LABEL[v]
                                    and (r["u"] < -1e-9 if sign == "u<0" else r["u"] > 1e-9) and np.isfinite(r[f"{m}_diff"])])
                    x, y = pts[:, 0], pts[:, 1]
                    t5b.append({"scene": g, "variant": LABEL[v], "metric": m, "u_sign": sign, "n_view_seeds": len(x),
                                "mean_diff_db": float(y.mean()), "slope_db_per_stop": float(np.polyfit(x, y, 1)[0]),
                                "spearman": float(spearmanr(x, y)[0])})
    save_csv("T5b_paired_vs_signed_u", t5b)
    tables["T5b. Per view, paired difference varying − LOM (same view and seed, dB) against signed u_k, darkened (u<0) "
           "and brightened (u>0) views separately (views with u = 0 left out). Points pool views × seeds"] = t5b
    numbers["T5b"] = t5b

    # ---- T6 consistency under the shared LOM reference depth
    t6, t6_md = [], []
    for g in GROUPS:
        for v in VARIANTS:
            seeds = [s for s in SEEDS if not (v == REF_VARIANT and s == REF_SEED)]
            md = {"scene": g, "variant": LABEL[v], "seeds": " ".join(map(str, seeds))}
            for k in CONS_METRICS + ["rel_l1_gt"]:
                fns = {rn: (lambda sc, s, rn=rn, k=k, v=v: float(np.mean([float(c[k]) for c in runs[(rn, sc, v, s)]["cons"]])))
                       for rn in ROOTS}
                a, b = seeds_of(fns["vexp"], g, seeds), seeds_of(fns["lom"], g, seeds)
                if k == "rel_l1_gt":
                    md[k] = f"{np.mean(b):.4f}"
                    continue
                e = paired(a, b)
                t6.append({"scene": g, "variant": LABEL[v], "seeds": md["seeds"], "metric": k,
                           "lom_mean": float(np.mean(b)), "vexp_mean": float(np.mean(a)), **e})
                md[k] = f"{np.mean(b):.4f} → {np.mean(a):.4f}, {fx(e, 4)}"
            t6_md.append(md)
    save_csv("T6_consistency", t6)
    tables[f"T6. Cross-view consistency (relative L1 after a depth warp with the LOM <scene>_{REF_VARIANT}_seed{REF_SEED} "
           f"depth, for both sets), LOM → varying, paired difference (×noise; * = real). The reference run is left out of "
           f"both arms, so official uses seeds 1-2. rel_l1_gt = the normal-light GT under the same warp (the floor)"] = t6_md
    numbers["T6"] = t6

    # ---- T7 colour and wall time
    t7, t7_md = [], []
    for g in GROUPS:
        for v in VARIANTS:
            md = {"scene": g, "variant": LABEL[v]}
            for m in COLOUR_METRICS:
                a, b = seeds_of(val("vexp", v, m), g), seeds_of(val("lom", v, m), g)
                e = paired(a, b)
                t7.append({"scene": g, "variant": LABEL[v], "metric": m,
                           "lom_mean": float(np.mean(b)), "vexp_mean": float(np.mean(a)), **e})
                md[m] = f"{np.mean(b):.{ND[m]}f} → {np.mean(a):.{ND[m]}f}, {fx(e, ND[m])}"
            t7_md.append(md)
    save_csv("T7_colour", t7)
    tables["T7a. Colour on validation, LOM → varying, paired difference (×noise; * = real). sat_ratio = render / GT "
           "saturation (1 = GT); ang_err = mean RGB angle error (°); hue_err_sat = hue error on coloured pixels (°)"] = t7_md
    numbers["T7"] = t7

    timing = []
    for v in VARIANTS:
        row = {"variant": LABEL[v]}
        for sc in SCENES:
            names = [run_name(sc, v, s) for s in SEEDS]
            row[f"{sc}_lom"] = float(np.mean([lom_minutes[n] for n in names if n in lom_minutes and n not in TIMING_EXCLUDE]))
            row[f"{sc}_vexp"] = float(np.mean([vexp_minutes[n] for n in names]))
        allv = [vexp_minutes[run_name(sc, v, s)] for sc in SCENES for s in SEEDS]
        alll = [lom_minutes[run_name(sc, v, s)] for sc in SCENES for s in SEEDS
                if run_name(sc, v, s) in lom_minutes and run_name(sc, v, s) not in TIMING_EXCLUDE]
        row["all_lom"], row["all_vexp"] = float(np.mean(alll)), float(np.mean(allv))
        timing.append(row)
    save_csv("T7_timing", timing)
    tables[f"T7b. Wall time per 10k-step run (minutes, last ok row of runs.csv; LOM excludes {', '.join(sorted(TIMING_EXCLUDE))})"] = timing
    numbers["timing"] = timing

    # headline numbers
    numbers["headline"] = {
        "T2_pooled": [r for r in t2 if r["scene"] in ("pooled", "pooled_no_shrub")],
        "T3_pooled": [r for r in t3 if r["scene"] in ("pooled", "pooled_no_shrub")],
        "T4_per_scene": [{k: r[k] for k in ("scene", "method", "slope_mean", "slope_sd", "r_mean", "rmse_mean", "rmse_sd",
                                             "slope_paired_mean", "rmse_paired_mean", "rmse_paired_sd")} for r in t4],
    }

    # =========================================================== figures
    def legend_below(fig, ax, ncol):
        h, lab = ax.get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.06), fontsize=9)

    # 1. validation, LOM vs varying
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    offs = [("baseline", "lom", -0.27), ("baseline", "vexp", -0.09), ("detach", "lom", 0.09), ("detach", "vexp", 0.27)]
    for ax, (m, lab) in zip(axes, (("psnr_gtmean", "brightness-matched PSNR (dB) ↑"), ("lpips", "LPIPS ↓"))):
        for j, sc in enumerate(SCENES):
            for v, rn, dx in offs:
                ys = [runs[(rn, sc, v, s)]["val"][m] for s in SEEDS]
                filled = rn == "vexp"
                ax.scatter([j + dx] * len(ys), ys, s=30, marker=VARIANT_MARKER[v], linewidths=1.4, zorder=3,
                           facecolors=VARIANT_COLOR[v] if filled else "none", edgecolors=VARIANT_COLOR[v],
                           label=f"{LABEL[v]}, {'varying exposure' if filled else 'LOM'}" if j == 0 else None)
                ax.hlines(np.mean(ys), j + dx - 0.07, j + dx + 0.07, color=VARIANT_COLOR[v], linewidth=2, zorder=2)
        ax.set_xticks(range(len(SCENES)))
        ax.set_xticklabels(SCENES)
        ax.set_ylabel(lab)
        ax.grid(axis="x", visible=False)
    legend_below(fig, axes[0], 4)
    fig.suptitle("Validation per run: LOM (hollow) vs the varying-exposure set (filled); dots = seeds 0-2, bar = mean",
                 x=0.01, ha="left")
    fig.tight_layout()
    save_fig(fig, "fig1_val_lom_vs_varying")

    # 2. fitted s_k against u_k, seed 0
    fig, axes = plt.subplots(1, len(SCENES), figsize=(20, 4.3), sharey=True)
    lim = np.array([-1.1, 1.1])
    for ax, sc in zip(axes, SCENES):
        o = oracle[sc]
        uo = np.array([u_by[sc][n] for n in o["vexp"]["names"]])
        ax.plot(lim, -lim, color=MUTED, linestyle="--", linewidth=1.2, zorder=1, label="ideal: s = −u")
        ax.scatter(uo, o["vexp"]["init"], marker="x", s=30, color=INK2, linewidths=1.2, zorder=2, label="init log2(m̄ / m_k)")
        ax.scatter(uo, o["vexp"]["oracle"], marker="^", s=34, facecolors="none", edgecolors=ORACLE_COLOR, linewidths=1.3,
                   zorder=2, label="histogram oracle")
        for v in VARIANTS:
            vw = expo_views[(sc, v, 0)]
            ax.scatter(vw["u"], vw["s_v"], marker=VARIANT_MARKER[v], s=26, color=VARIANT_COLOR[v], alpha=0.85, zorder=3,
                       label=f"{LABEL[v]} curves (seed 0)")
        ax.set_title(sc, loc="left")
        ax.set_xlabel("applied gain u_k (stops)")
    axes[0].set_ylabel("fitted exposure s_k (stops)")
    legend_below(fig, axes[0], 5)
    fig.suptitle("Per-view exposure fitted to the final curves vs the applied gain, varying-exposure set", x=0.01, ha="left")
    fig.tight_layout()
    save_fig(fig, "fig2_s_fit_vs_u")

    # 3 and 4. paired per-view quantities against u_k, all seeds
    def paired_panels(ycol, ylabel, title, name, lines):
        fig, axes = plt.subplots(1, len(SCENES), figsize=(20, 4.3), sharey=True)
        for ax, sc in zip(axes, SCENES):
            for lab, (xs, ys, style) in lines.items():
                ax.plot(xs, ys, color=MUTED, linewidth=1.2, linestyle=style, zorder=1, label=lab)
            for v in VARIANTS:
                rows = [r for r in per_view if r["scene"] == sc and r["variant"] == LABEL[v]]
                ax.scatter([r["u"] for r in rows], [r[ycol] for r in rows], marker=VARIANT_MARKER[v], s=20,
                           color=VARIANT_COLOR[v], alpha=0.7, zorder=3, label=f"{LABEL[v]} (seeds 0-2)")
            ax.set_title(sc, loc="left")
            ax.set_xlabel("applied gain u_k (stops)")
        axes[0].set_ylabel(ylabel)
        legend_below(fig, axes[0], len(lines) + 2)
        fig.suptitle(title, x=0.01, ha="left")
        fig.tight_layout()
        save_fig(fig, name)

    paired_panels("log2_pseudo_ratio", "log2(mean pseudo-target, varying / LOM)",
                  "Pseudo-target brightness change per view (same view and seed) vs the applied gain",
                  "fig3_pseudo_brightness_paired_vs_u",
                  {"0: fully compensated": (lim, [0, 0], "--"), "1: passes straight through": (lim, lim, ":")})
    paired_panels("fit_psnr_lowrender_vs_input_diff", "Δ dark-branch fit PSNR, varying − LOM (dB)",
                  "Shared dark render vs the dark input photo: paired change per view (same view and seed) vs the applied gain",
                  "fig4_lowbranch_fit_paired_vs_u",
                  {"no change": (lim, [0, 0], "--")})

    # =========================================================== write
    with open(OUT / "tables.md", "w", encoding="utf-8") as f:
        f.write("# M1 tables (generated by examples/vexp_analyze.py)\n\n"
                "official = `baseline` (the official code); varying = `results/vexp/` (data/LOM_vexp); LOM = `results/rung0/`. "
                "Effects are paired by seed; (x×) = mean effect / noise, with noise = sqrt(mean of the seed variances of the "
                f"two arms); * = |effect| > {REAL:g}× noise. Raw PSNR, SSIM and LPIPS are taken from the re-score "
                "(identical to the training-time eval, see the integrity checks).\n\n")
        for title, rows in tables.items():
            f.write(f"## {title}\n\n{md_table(rows)}\n")
    with open(OUT / "numbers.json", "w") as f:
        json.dump(numbers, f, indent=1, default=float)
    print(f"[vexp_analyze] wrote {OUT}")


if __name__ == "__main__":
    main()
