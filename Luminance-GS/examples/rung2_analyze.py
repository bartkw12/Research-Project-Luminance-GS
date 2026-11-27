"""M3 analysis (Implementation_guides/M2_rung2_plan.md, section 6): Rung 2 against `detach` and the official code
on LOM (H1), the 2x2 on the varying-exposure set (H2: curve model x dark branch), and the mechanism diagnostics
(pseudo-target leak, dark-branch fit, exposure recovery). CPU only. Reads finished, re-scored runs and writes only
to results/rung2/summary*/ (or, in the test modes, to git-ignored folders under results/rung2/smoke_m2/).

    python rung2_analyze.py              seeds 0-2 -> results/rung2/summary/
    python rung2_analyze.py --seeds 0    seed-0 mode, "preliminary, one seed" -> results/rung2/summary_seed0/
    python rung2_analyze.py --smoke      code test on the 300-step part C smoke runs (Rung 2 cells; buu, seed 0)
    python rung2_analyze.py --selftest   code test of the full-seed tables: the Rung 2 cells are read from the
                                         detach / detachexp runs, so every curve-model effect must be exactly 0

Inputs (read only):
    LOM      results/rung0/<scene>_{baseline,detach}_seed*, results/rung2/<scene>_rung2_seed*
    varying  results/vexp/<scene>_{baseline,detach,detachexp,rung2,rung2exp}_seed*
    per run  cfg.json, stats/val_step9999.json, rescore/{val_summary.json, pseudo_targets.csv, consistency.csv},
             diagnostics/{pseudo_targets.csv, rung2_state.json, exposures_by_step.csv, curves_step10000.npz (local)},
             and an ok row in the root's runs.csv
    other    results/vexp/dataset/{gains,oracle}.json; M1's results/vexp/summary/{T1_val_varying, T4_exposure,
             T5b_paired_vs_signed_u}.csv; results/rung2/preview_final/views.csv
M5 (M5_plan.md step 3, E1 / H2b): --rung2_variant rung2id reads the Rung 2 cells from the identity-init runs
(rung2id on LOM and the varying set, rung2idexp in the exp column) and names them so; the analysis logic is
unchanged. Its H1 verdict on LOM is exploratory (H1 is judged on the cdf init), and the H2 rule's "between 1x and 2x:
run seeds 3 and 4" outcome reads "inconclusive" (M5 3.2(b)). Default output results/rung2/summary_rung2id/.
--out writes elsewhere. The guard refuses every finished folder (CLAUDE.md rule 6, listed in protected.py: M3's summary/,
summary_seed0/, summary_headstart*/, preview_final/ and, since 2026-11-26, M5's and E5's outputs such as
summary_rung2id/) and anything outside results/rung2/summary*/ or a git-ignored results/**/smoke*/ folder. So the
default command (summary/) and the E1 command below (summary_rung2id/, finished 2026-11-25) now refuse:
    python rung2_analyze.py --rung2_variant rung2id                         -> results/rung2/summary_rung2id/ (finished)
    python rung2_analyze.py --out ../results/rung2/smoke_m5/analyze_default  the M5 byte-for-byte test of M3's tables

A cell (set, variant) enters the tables only when every scene has a usable run for every requested seed; otherwise
it is listed as pending. The script stops before any table if a run has the wrong data, flags, step count or
reference depth, or if rung2 and rung2exp (same seed) differ in F, s or pseudo-targets (plan 6.1). With seeds 0-2
it also checks that it reproduces M1's numbers for official and detach (T1, leak, dark-fit slopes) exactly.

Tables (tables.md, CSVs, numbers.json): R1 H1 on LOM, R2 H1 extras, R3 the 2x2 cells, R4 factorial effects,
R5 mechanism (a leak, b dark fit, c exposure), R6 side effects, and the decision rules of plan 6.5 (seeds 0-2
only). Figures 1-6 of plan 6.4. Noise as in M1: pair by seed, noise = sqrt(mean of the two arms' seed variances),
real above 2x; the factorial main effects and the interaction use the noise over the four cells
(rung05_analyze.effects). One-seed mode has no sd, no x-noise and no "real": each difference sits next to the
reference cell's seed sd over seeds 0-2 ("ref sd").
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))

from build_vexp import fit_exposure_form, slope_r  # noqa: E402
from rung0_analyze import (INK, INK2, MUTED, SCENES, SERIES, TIMING_EXCLUDE, md_table, read_csv,  # noqa: E402
                           run_name, savefig)
from vexp_analyze import as_path, fx, last_ok_minutes, paired, pm, recovery_stats, rmse_c  # noqa: E402
from protected import refuse_finished  # noqa: E402  (rule 6 finished folders: protected.FINISHED)

RESULTS = EXAMPLES.parent / "results"
RUNG0, RUNG2, VEXP = RESULTS / "rung0", RESULTS / "rung2", RESULTS / "vexp"
DATASET = VEXP / "dataset"
M1_SUMMARY = VEXP / "summary"
PREVIEW = RUNG2 / "preview_final"
SMOKE = RUNG2 / "smoke_m2"
LOM_DATA = EXAMPLES.parent / "data" / "LOM_full"
VEXP_DATA = EXAMPLES.parent / "data" / "LOM_vexp"

ALL_SEEDS = [0, 1, 2]
REAL = 2.0
REF_VARIANT, REF_SEED = "baseline", 0  # LOM <scene>_baseline_seed0: the shared reference depth
H1_SCENES = ["buu", "chair", "sofa", "bike"]  # shrub is reported but does not decide (protocol rule 7)
LOM_VARIANTS = ["baseline", "detach", "rung2"]
VEXP_VARIANTS = ["baseline", "detach", "detachexp", "rung2", "rung2exp"]
REFERENCE_CELLS = {"baseline", "detach", "detachexp"}  # always loaded for seeds 0-2 (their seed sd is the yardstick)
LABEL = {"baseline": "official", "detach": "detach", "detachexp": "detachexp", "rung2": "rung2", "rung2exp": "rung2exp"}
ROW = {"baseline": "baseline", "detach": "detach", "detachexp": "detach", "rung2": "rung2", "rung2exp": "rung2"}
COLOR = {"baseline": SERIES[0], "detach": SERIES[1], "detachexp": SERIES[1], "rung2": SERIES[4], "rung2exp": SERIES[4]}
MARKER = {"baseline": "o", "detach": "s", "detachexp": "s", "rung2": "D", "rung2exp": "D"}
HOLLOW = {"detachexp", "rung2exp"}  # exp cells: their row's colour, hollow markers, dashed lines

RUNG2_FLAGS = {"curve_model": "rung2", "rung2_knots": 32, "rung2_knot_spacing": "quadratic", "rung2_F_init": "cdf",
               "rung2_s_init": "measured", "rung2_attach": "none", "rung2_prior": "F", "rung2_smooth_lambda": 0.1,
               "rung2_lr_F": 0.01, "rung2_lr_s": 0.001}  # the M3 configuration (decided 2026-11-13)
EXPECTED = {  # plan 2.3; keys missing from older cfg.json files take the official default (CFG_DEFAULTS)
    "baseline": {"detach_pseudo_target": False, "curve_model": "official", "dark_branch": "shared"},
    "detach": {"detach_pseudo_target": True, "curve_model": "official", "dark_branch": "shared"},
    "detachexp": {"detach_pseudo_target": True, "curve_model": "official", "dark_branch": "measured"},
    "rung2": {"detach_pseudo_target": False, "dark_branch": "shared", **RUNG2_FLAGS},
    "rung2exp": {"detach_pseudo_target": False, "dark_branch": "model", **RUNG2_FLAGS},
    # M5 E1: the identity-init variants (runner variants rung2id, rung2idexp)
    "rung2id": {"detach_pseudo_target": False, "dark_branch": "shared", **RUNG2_FLAGS, "rung2_F_init": "identity"},
    "rung2idexp": {"detach_pseudo_target": False, "dark_branch": "model", **RUNG2_FLAGS, "rung2_F_init": "identity"},
}
RUNG2_SOURCES = {"rung2": {"rung2": "rung2", "rung2exp": "rung2exp"},  # --rung2_variant -> run folders of the cells
                 "rung2id": {"rung2": "rung2id", "rung2exp": "rung2idexp"}}
CFG_DEFAULTS = {"detach_pseudo_target": False, "curve_model": "official", "dark_branch": "shared",
                "freeze_color_basis": False, "detach_curve_in_target": False}
VAL = ["psnr_gtmean", "psnr", "ssim", "lpips"]
NAME = {"psnr_gtmean": "bm PSNR", "psnr": "raw PSNR", "ssim": "SSIM", "lpips": "LPIPS", "sat_ratio": "sat_ratio",
        "ang_err_deg": "angle err (°)", "hue_err_sat_deg": "hue err (°)", "pseudo_bm": "pseudo bm PSNR",
        "cons_pseudo": "pseudo consistency (× GT)", "cons_render": "render consistency (× GT)",
        "dark_fit": "dark fit (dB)", "minutes": "minutes/run", "num_GS": "Gaussians", "params": "curve params"}
ND = {"psnr_gtmean": 2, "psnr": 2, "ssim": 4, "lpips": 4, "sat_ratio": 3, "ang_err_deg": 2, "hue_err_sat_deg": 2,
      "pseudo_bm": 2, "cons_pseudo": 2, "cons_render": 2, "dark_fit": 2, "minutes": 1, "num_GS": 0, "params": 0}
OFFICIAL_CURVE_PARAMS = 255 + 143_743 + 111_235  # plus 5 per training view (colour basis); plan 2.2
FIG_SCENE = "sofa"  # figure 6


# ------------------------------------------------------------------ mode and output guard
def guard_out(out: Path) -> Path:
    """M5 overwrite guard: never a finished folder (rule 6, protected.py); otherwise only results/rung2/summary*/ or a
    git-ignored results/**/smoke*/ folder."""
    out = refuse_finished(out, "rung2_analyze")
    in_smoke = any(p.name.startswith("smoke") and RESULTS.resolve() in p.parents for p in out.parents)
    if not (in_smoke or (out.parent == RUNG2.resolve() and out.name.startswith("summary"))):
        raise SystemExit(f"[rung2_analyze] refusing to write to {out}")
    return out


class Ctx:
    def __init__(self, a):
        self.smoke, self.selftest = a.smoke, a.selftest
        self.rung2_variant = a.rung2_variant
        self.seeds = [0] if a.smoke else sorted(set(a.seeds))
        self.scenes = ["buu"] if a.smoke else [s for s in SCENES if s in a.scenes]
        self.one_seed = len(self.seeds) == 1
        if a.out is not None:
            self.out = a.out
        elif a.smoke:
            self.out = SMOKE / "summary_test"
        elif a.selftest:
            self.out = SMOKE / "summary_selftest"
        else:
            tag = "" if a.rung2_variant == "rung2" else "_" + a.rung2_variant
            self.out = RUNG2 / ("summary" + tag + ("" if self.seeds == ALL_SEEDS
                                                   else "_seed" + "".join(map(str, self.seeds))))
        self.out = guard_out(self.out)
        self.fig = self.out / "figures"
        self.fig.mkdir(parents=True, exist_ok=True)

    def source(self, set_, v):
        """(root, run-folder variant, expected max_steps) of a cell."""
        if self.smoke and v in ("rung2", "rung2exp"):
            return (SMOKE if set_ == "lom" else VEXP / "smoke_m2"), v, 300
        if self.selftest and v in ("rung2", "rung2exp"):
            return (RUNG0 if set_ == "lom" else VEXP), {"rung2": "detach", "rung2exp": "detachexp"}[v], 10_000
        if v in ("rung2", "rung2exp"):
            v = RUNG2_SOURCES[self.rung2_variant][v]
            return (RUNG2 if set_ == "lom" else VEXP), v, 10_000
        return ((RUNG2 if v == "rung2" else RUNG0) if set_ == "lom" else VEXP), v, 10_000

    def groups(self):
        g = list(self.scenes)
        if len(self.scenes) > 1:
            g.append("pooled")
        if "shrub" in self.scenes and len(self.scenes) > 2:
            g.append("pooled_no_shrub")
        return g

    def members(self, g):
        return {"pooled": self.scenes, "pooled_no_shrub": [s for s in self.scenes if s != "shrub"]}.get(g, [g])


# ------------------------------------------------------------------ loading and checks
def load_run(root: Path, scene: str, variant: str, seed: int, ok: dict):
    """A finished, re-scored run with an ok row in runs.csv, or None."""
    name = run_name(scene, variant, seed)
    d = root / name
    if name not in ok or not (d / "rescore" / "val_summary.json").exists():
        return None
    cfg = json.loads((d / "cfg.json").read_text())
    n = int(cfg["max_steps"])
    dd, rs = d / "diagnostics", d / "rescore"
    r = {"dir": d, "cfg": cfg, "scene": scene, "seed": seed, "minutes": ok[name],
         "stats": json.loads((d / "stats" / f"val_step{n - 1:04d}.json").read_text()),
         "val": json.loads((rs / "val_summary.json").read_text()),
         "pseudo": {x["view"]: x for x in read_csv(rs / "pseudo_targets.csv")},
         "own_pseudo": {x["view"]: x for x in read_csv(dd / "pseudo_targets.csv")},
         "cons": read_csv(rs / "consistency.csv"), "npz": dd / f"curves_step{n:05d}.npz", "state": None, "steps": None}
    if (dd / "rung2_state.json").exists():
        r["state"] = json.loads((dd / "rung2_state.json").read_text())
        r["steps"] = read_csv(dd / "exposures_by_step.csv")
    return r


def load_all(ctx):
    runs = {}
    for set_, variants in (("lom", LOM_VARIANTS), ("vexp", VEXP_VARIANTS)):
        for v in variants:
            root, fv, steps = ctx.source(set_, v)
            ok = last_ok_minutes(root / "runs.csv") if (root / "runs.csv").exists() else {}
            seeds = ALL_SEEDS if (v in REFERENCE_CELLS or fv in REFERENCE_CELLS) else ctx.seeds
            for sc in ctx.scenes:
                for s in seeds:
                    r = load_run(root, sc, fv, s, ok)
                    if r is not None:
                        r.update(set=set_, variant=v, source_variant=fv, expected_steps=steps)
                        runs[(set_, v, sc, s)] = r
    return runs


def check_inputs(runs):
    """Plan 6.1: stop before any table if a run used the wrong data, flags, step count or reference depth."""
    bad = []
    for (set_, v, sc, s), r in runs.items():
        name = f"{r['dir'].parent.name}/{r['dir'].name}"
        c = r["cfg"]
        want = ((VEXP_DATA if set_ == "vexp" else LOM_DATA) / sc).resolve()
        if as_path(c["data_dir"]) != want:
            bad.append(f"{name}: data_dir {c['data_dir']} is not {want}")
        for k, x in {**{"freeze_color_basis": False, "detach_curve_in_target": False},
                     **EXPECTED[r["source_variant"]]}.items():
            got = c.get(k, CFG_DEFAULTS.get(k))
            if (abs(got - x) > 1e-12) if isinstance(x, float) else got != x:
                bad.append(f"{name}: {k} = {got!r}, expected {x!r} for {r['source_variant']}")
        if int(c["max_steps"]) != r["expected_steps"]:
            bad.append(f"{name}: max_steps {c['max_steps']}, expected {r['expected_steps']}")
        ref = (RUNG0 / run_name(sc, REF_VARIANT, REF_SEED)).resolve()
        if as_path(r["val"]["ref_depth_run"]) != ref:
            bad.append(f"{name}: re-scored with reference depth {r['val']['ref_depth_run']}, not {ref} "
                       f"(rescore_runs.py --ref_root ../results/rung0)")
    if bad:
        raise SystemExit("[rung2_analyze] input check failed:\n  " + "\n  ".join(bad))


# ------------------------------------------------------------------ per-run quantities
def fnum(x):
    return float(x) if x not in (None, "") else float("nan")


def dark_fit(row) -> float:
    """The dark render as its loss sees it (plan 6.3b): the exp column where it exists; for runs re-scored before
    part A (shared dark branch only) the old column, which it equals there."""
    x = row.get("fit_psnr_lowrender_exp_vs_input")
    return fnum(x) if x not in (None, "") else fnum(row["fit_psnr_lowrender_vs_input"])


def cons_ratio(r, key):
    """Mean relative L1 after the shared-depth warp, as a multiple of the GT's under the same warp."""
    return float(np.mean([fnum(c[f"rel_l1_{key}"]) for c in r["cons"]]) / np.mean([fnum(c["rel_l1_gt"]) for c in r["cons"]]))


def curve_params(r) -> int:
    if r["state"] is not None:
        return int(r["state"]["n_params"])
    return OFFICIAL_CURVE_PARAMS + 5 * len(r["pseudo"])


def run_value(r, key) -> float:
    """Scalar per run for the extras tables."""
    if key in ("psnr_gtmean", "psnr", "ssim", "lpips", "sat_ratio", "ang_err_deg", "hue_err_sat_deg"):
        return fnum(r["val"][key])
    if key == "pseudo_bm":
        return float(np.nanmean([fnum(x["pseudo_psnr_gtmean"]) for x in r["pseudo"].values()]))
    if key == "dark_fit":
        return float(np.nanmean([dark_fit(x) for x in r["pseudo"].values()]))
    if key in ("cons_pseudo", "cons_render"):
        if r["set"] == "lom" and r["source_variant"] == REF_VARIANT and r["seed"] == REF_SEED:
            return float("nan")  # the reference-depth run itself: left out, as in M1 T6
        return cons_ratio(r, key.split("_")[1])
    if key == "minutes":  # TIMING_EXCLUDE names LOM runs (the PC slept); the varying set reuses the run names
        excluded = r["set"] == "lom" and r["dir"].parent == RUNG0 and r["dir"].name in TIMING_EXCLUDE
        return float("nan") if excluded else float(r["minutes"])
    if key == "num_GS":
        return float(r["stats"]["num_GS"])
    if key == "params":
        return float(curve_params(r))
    raise KeyError(key)


def leak_stats(rv, rl, u_by):
    """Slopes of log2 mean brightness on u_k (0 = compensated, 1 = passes through), unpaired and paired with the
    same view on LOM, for the pseudo-targets, the renders at the training views and the dark inputs (M1 T4b)."""
    names = list(rv["pseudo"])
    assert set(names) == set(rl["pseudo"]) == set(u_by), f"{rv['dir'].name}: view names do not join"
    u = np.array([u_by[n] for n in names])
    out = {}
    for k, key in (("pseudo", "mean_pseudo"), ("render", "mean_render"), ("input", "mean_input")):
        a = np.array([fnum(rv["pseudo"][n][key]) for n in names])
        b = np.array([fnum(rl["pseudo"][n][key]) for n in names])
        out[f"{k}_leak"] = slope_r(np.log2(a), u)[0]
        out[f"{k}_leak_paired"] = slope_r(np.log2(a / b), u)[0]
    return out


# ------------------------------------------------------------------ formatting
def stat(x, nd):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "n/a"
    return pm(float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan"), nd)


def diff_str(a, b, nd, ref_sd=float("nan")):
    """Seeds 0-2: the paired effect (x-noise, * = real). One seed: the difference next to the reference sd."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) > 1:
        keep = np.isfinite(a) & np.isfinite(b)  # e.g. a wall time left out (TIMING_EXCLUDE): drop that seed pair
        if keep.sum() < 2:
            return "n/a", {"diff": float("nan"), "noise": float("nan"), "x_noise": float("nan"), "real": False}
        e = paired(a[keep], b[keep])
        return fx(e, nd) + ("" if keep.all() else f" [{int(keep.sum())} seeds]"), e
    d = float(a[0] - b[0])
    return (f"{d:+.{nd}f} (ref sd {ref_sd:.{nd}f})" if np.isfinite(ref_sd) else f"{d:+.{nd}f}"), \
        {"diff": d, "ref_sd": float(ref_sd)}


def factorial(cells: dict):
    """cells[variant] = per-seed arrays for detach, detachexp, rung2, rung2exp -> effects per seed and the noise over
    the four cells (rung05_analyze.effects); the simple effects are also judged with their two arms' noise."""
    D, DX, R, RX = (np.asarray(cells[k], float) for k in ("detach", "detachexp", "rung2", "rung2exp"))
    r, rx = LABEL["rung2"], LABEL["rung2exp"]  # M5: rung2id / rung2idexp with --rung2_variant rung2id
    main = {f"curve model (H2): ½[({r} − detach) + ({rx} − detachexp)]": 0.5 * ((R - D) + (RX - DX)),
            f"dark branch: ½[(detachexp − detach) + ({rx} − {r})]": 0.5 * ((DX - D) + (RX - R)),
            f"interaction: ({rx} − {r}) − (detachexp − detach)": (RX - R) - (DX - D)}
    simple = {f"{r} − detach (shared column)": (R, D), f"{rx} − detachexp (exp column)": (RX, DX),
              "detachexp − detach (detach row)": (DX, D), f"{rx} − {r} ({r} row)": (RX, R)}
    noise = float(np.sqrt(np.mean([np.var(x, ddof=1) for x in (D, DX, R, RX)]))) if len(D) > 1 else float("nan")
    return main, simple, noise


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=ALL_SEEDS)
    ap.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    ap.add_argument("--smoke", action="store_true", help="code test on the 300-step smoke runs (buu, seed 0)")
    ap.add_argument("--selftest", action="store_true", help="code test: Rung 2 cells read from detach / detachexp")
    ap.add_argument("--rung2_variant", choices=list(RUNG2_SOURCES), default="rung2",
                    help="M5 E1: rung2id reads the Rung 2 cells from rung2id / rung2idexp")
    ap.add_argument("--out", type=Path, default=None,
                    help="output folder (guarded: results/rung2/summary*/ or a results/**/smoke*/ folder; never M3's)")
    a = ap.parse_args()
    if a.smoke and a.selftest:
        raise SystemExit("pick one of --smoke and --selftest")
    if a.rung2_variant != "rung2" and (a.smoke or a.selftest):
        raise SystemExit("--rung2_variant applies to the real runs only")
    LABEL.update({k: v for k, v in RUNG2_SOURCES[a.rung2_variant].items()})
    ident = a.rung2_variant == "rung2id"
    ctx = Ctx(a)
    t_start = time.time()
    G = ctx.groups()
    S = ctx.seeds

    gains = json.loads((DATASET / "gains.json").read_text())["rows"]
    u_by = {sc: {r["view"]: r["u"] for r in gains if r["scene"] == sc} for sc in SCENES}
    ueff_by = {sc: {r["view"]: r["log2_ratio"] for r in gains if r["scene"] == sc} for sc in SCENES}
    oracle_json = {r["scene"]: r for r in json.loads((DATASET / "oracle.json").read_text())["rows"]}

    runs = load_all(ctx)
    check_inputs(runs)

    def have(set_, v, seeds=None):
        return all((set_, v, sc, s) in runs for sc in ctx.scenes for s in (seeds or S))

    status, cells = [], {}
    for set_, variants in (("lom", LOM_VARIANTS), ("vexp", VEXP_VARIANTS)):
        for v in variants:
            n = sum((set_, v, sc, s) in runs for sc in ctx.scenes for s in S)
            root, fv, _ = ctx.source(set_, v)
            cells[(set_, v)] = have(set_, v)
            status.append({"set": set_, "cell": LABEL[v], "runs": f"{n}/{len(ctx.scenes) * len(S)}",
                           "source": f"{root.relative_to(RESULTS).as_posix()}/<scene>_{fv}_seed*",
                           "status": "used" if cells[(set_, v)] else "pending"})
    print(f"[rung2_analyze] {len(runs)} runs loaded and checked; cells used: "
          f"{[f'{s}/{LABEL[v]}' for (s, v), ok in cells.items() if ok]}", flush=True)

    def per_seed(fn, g, seeds=None):
        return np.array([np.mean([fn(sc, s) for sc in ctx.members(g)]) for s in (seeds or S)])

    def getter(set_, v, key):
        return lambda sc, s: run_value(runs[(set_, v, sc, s)], key)

    def ref_sd(set_, v, key, g):
        """The seed sd over seeds 0-2 of a reference cell (one-seed mode's yardstick)."""
        if not have(set_, v, ALL_SEEDS):
            return float("nan")
        return float(np.std(per_seed(getter(set_, v, key), g, ALL_SEEDS), ddof=1))

    tables, numbers, integrity = {}, {}, []

    def check(name, value, tol, stop=True):
        ok = bool(np.isfinite(value) and value <= tol)
        integrity.append({"check": name, "value": f"{value:.2e}", "tolerance": f"{tol:.0e}",
                          "ok": ok if stop else ("report" if ok else "report (above)")})
        return ok

    def save(name, rows):
        """CSV in UTF-8 (rung0_analyze.write_csv uses the Windows codepage, which has no '−' or '½')."""
        if not rows:
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(ctx.out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # ================================================================ integrity
    for (set_, v, sc, s), r in runs.items():
        for m, tol in (("psnr", 1e-3), ("ssim", 1e-4), ("lpips", 1e-4)):
            d = abs(fnum(r["val"][m]) - fnum(r["stats"][m]))
            if d > tol:
                check(f"{r['dir'].parent.name}/{r['dir'].name}: |rescore − training eval| {m}", d, tol)
    check("max |rescore − training eval| psnr over all loaded runs",
          max(abs(fnum(r["val"]["psnr"]) - fnum(r["stats"]["psnr"])) for r in runs.values()), 1e-3)

    # built-in checks (plan 2.1, 6.1): an exp cell's curve side equals its row's
    for row_v, exp_v in (("detach", "detachexp"), ("rung2", "rung2exp")):
        worst = 0.0
        n_pairs = 0
        for sc in ctx.scenes:
            for s in S:
                ka, kb = ("vexp", row_v, sc, s), ("vexp", exp_v, sc, s)
                if ka in runs and kb in runs:
                    a, b = runs[ka]["own_pseudo"], runs[kb]["own_pseudo"]
                    worst = max(worst, max(abs(np.log2(fnum(a[n]["mean_pseudo"]) / fnum(b[n]["mean_pseudo"]))) for n in a))
                    n_pairs += 1
                    sa, sb = runs[ka]["state"], runs[kb]["state"]
                    if sa is not None and sb is not None:
                        check(f"{sc} seed {s}: {LABEL['rung2']} vs {LABEL['rung2exp']}, max |Δ s|", float(np.max(np.abs(
                            np.subtract(sa["s_final"], sb["s_final"])))), 1e-5)
                        check(f"{sc} seed {s}: {LABEL['rung2']} vs {LABEL['rung2exp']}, max |Δ F heights|", float(np.max(np.abs(
                            np.subtract(sa["heights"], sb["heights"])))), 1e-5)
        if n_pairs:
            check(f"{LABEL[row_v]} vs {LABEL[exp_v]} (same seed, {n_pairs} pairs): max |Δ log2 mean_pseudo|", worst,
                  1e-5 if row_v == "rung2" else 1e-3)

    # Rung 2 states: s_init vs oracle.json, fit_exposure_form on the composed curves vs the run's own s, mean s
    preview = [r for r in read_csv(PREVIEW / "views.csv") if r["config"] == "quadratic-cdf-priorF"] \
        if (PREVIEW / "views.csv").exists() else []
    for (set_, v, sc, s), r in sorted(runs.items(), key=lambda kv: str(kv[0])):
        st = r["state"]
        if st is None or v == "rung2exp":
            continue
        tag = f"{set_} {sc} seed {s}"
        if set_ == "vexp":
            want = dict(zip(oracle_json[sc]["view_names"], oracle_json[sc]["s_init"]))
            check(f"{tag}: {LABEL['rung2']} s_init vs oracle.json s_init (stops)",
                  max(abs(x - want[n]) for n, x in zip(st["view_names"], st["s_init"])), 1e-6)
        check(f"{tag}: {LABEL['rung2']} |mean s|", abs(st["mean_s"]), 1e-5)
        if r["npz"].exists():
            z = np.load(r["npz"])
            assert [str(x) for x in z["view_names"]] == st["view_names"], f"{tag}: npz view order differs"
            s_fit = fit_exposure_form(z["curve_adj"].astype(np.float64), z["bin_hist"].astype(np.float64))[0]
            check(f"{tag}: fit_exposure_form(curve_adj) vs the run's s (stops)",
                  float(np.max(np.abs(s_fit - np.array(st["s_final"])))), 1e-2)
        pv = {x["view"]: fnum(x["s_final"]) for x in preview if x["scene"] == sc and x["set"] == set_}
        if pv and r["expected_steps"] == 10_000:
            check(f"{tag}: trainer s vs the preview's (report only; the view order differs)",
                  max(abs(x - pv[n]) for n, x in zip(st["view_names"], st["s_final"])), 5e-2, stop=False)

    # reproduce M1 (official and detach on both sets, seeds 0-2)
    reproduce = S == ALL_SEEDS and ctx.scenes == SCENES and not ctx.smoke
    if reproduce and all(cells[(st_, v)] for st_ in ("lom", "vexp") for v in ("baseline", "detach")):
        t1 = {(r["scene"], r["variant"]): r for r in read_csv(M1_SUMMARY / "T1_val_varying.csv")}
        worst = max(abs(float(per_seed(getter("vexp", v, m), g).mean()) - float(t1[(g, LABEL[v])][f"{m}_mean"]))
                    for v in ("baseline", "detach") for g in G if (g, LABEL[v]) in t1 for m in VAL)
        check("M1 reproduction: T1 validation means (official, detach)", worst, 1e-9)
    stops = [c for c in integrity if c["ok"] is False]
    if stops:
        print(md_table(integrity))
        raise SystemExit("[rung2_analyze] integrity check failed (table above)")
    print("[rung2_analyze] integrity checks passed", flush=True)

    # ================================================================ R1 H1 (LOM)
    lom_v = [v for v in LOM_VARIANTS if cells[("lom", v)]]
    r1a, r1b, r1_csv, h1 = [], [], [], {}
    for g in G:
        for v in lom_v:
            vals = {m: per_seed(getter("lom", v, m), g) for m in VAL}
            r1a.append({"scene": g, "variant": LABEL[v], **{NAME[m]: stat(vals[m], ND[m]) for m in VAL}})
            r1_csv.append({"scene": g, "variant": LABEL[v], **{f"{m}_seed{s}": float(x) for m in VAL
                                                               for s, x in zip(S, vals[m])}})
        if cells[("lom", "rung2")] and cells[("lom", "detach")]:
            row, es = {"scene": g}, {}
            for m in VAL:
                row[NAME[m]], es[m] = diff_str(per_seed(getter("lom", "rung2", m), g),
                                               per_seed(getter("lom", "detach", m), g), ND[m],
                                               ref_sd("lom", "detach", m, g))
            if not ctx.one_seed and g in H1_SCENES:
                fails = [NAME[m] for m in VAL if m != "psnr" and (
                    es[m]["x_noise"] > REAL if m == "lpips" else es[m]["x_noise"] < -REAL)]
                row["H1 rule (plan 6.5)"] = "meets" if not fails else "fails: " + ", ".join(fails)
                h1[g] = row["H1 rule (plan 6.5)"]
            elif not ctx.one_seed:
                row["H1 rule (plan 6.5)"] = "reported only" if g == "shrub" else "—"
            r1b.append(row)
            numbers.setdefault("R1_rung2_minus_detach", {})[g] = es
    save("R1_H1_lom_cells", r1_csv)
    tables["R1a. H1 on LOM: validation per cell (mean ± sd over seeds; pooled = mean over scenes per seed). "
           "bm PSNR = brightness-matched PSNR"] = r1a
    if r1b:
        tables[f"R1b. {LABEL['rung2']} − detach on LOM, paired by seed (seeds 0-2: ×noise, * = real; one seed: the "
               "difference next to detach's seed sd over seeds 0-2). H1 rule: bm PSNR not below −2× noise, SSIM and "
               "LPIPS not worse beyond 2× noise, in each of buu, chair, sofa and bike; shrub reported only"
               + ("; exploratory for the identity init (H1 is judged on the cdf init)" if ident else "")] = r1b

    # ================================================================ R2 H1 extras (LOM) and R6 side effects (varying)
    def extras(set_, variants, keys, title, csv_name, diffs=()):
        rows, out_csv = [], []
        for g in G:
            for v in variants:
                row = {"scene": g, "variant": LABEL[v]}
                for k in keys:
                    x = per_seed(getter(set_, v, k), g)
                    row[NAME[k]] = stat(x, ND[k])
                    out_csv.append({"scene": g, "variant": LABEL[v], "metric": k,
                                    **{f"seed{s}": float(y) for s, y in zip(S, x)}})
                rows.append(row)
            for va, vb in diffs:
                if va in variants and vb in variants:
                    row = {"scene": g, "variant": f"{LABEL[va]} − {LABEL[vb]}"}
                    for k in keys:
                        if k == "params":
                            row[NAME[k]] = ""
                            continue
                        row[NAME[k]] = diff_str(per_seed(getter(set_, va, k), g), per_seed(getter(set_, vb, k), g),
                                                ND[k], ref_sd(set_, vb, k, g))[0]
                    rows.append(row)
        save(csv_name, out_csv)
        tables[title] = rows

    extras("lom", lom_v, ["pseudo_bm", "cons_pseudo", "cons_render", "ang_err_deg", "sat_ratio", "hue_err_sat_deg",
                          "dark_fit", "minutes", "num_GS", "params"],
           "R2. H1 extras on LOM: pseudo-target bm PSNR against GT, cross-view consistency of pseudo-targets and "
           "renders (relative L1 after the shared-depth warp, as a multiple of the GT's; the reference-depth run "
           "official seed 0 is left out), colour on validation, dark fit (dark render vs input, mean over training "
           "views), wall time (runs.csv), final Gaussians, curve parameters", "R2_H1_extras",
           diffs=[("rung2", "detach")])

    vexp_v = [v for v in VEXP_VARIANTS if cells[("vexp", v)]]
    extras("vexp", vexp_v, ["sat_ratio", "ang_err_deg", "hue_err_sat_deg", "cons_pseudo", "cons_render", "minutes",
                            "num_GS"],
           "R6. Side effects on the varying set: colour on validation, consistency (× GT, shared LOM depth), wall "
           "time, final Gaussians. The exp rows are the column effect within each curve row (the dark branch "
           "re-weights each view's L_low gradient, plan 2.1)", "R6_side_effects",
           diffs=[("detachexp", "detach"), ("rung2exp", "rung2")])

    # ================================================================ R3 cells and R4 factorial (varying)
    r3, r3_csv = [], []
    for g in G:
        for v in vexp_v:
            vals = {m: per_seed(getter("vexp", v, m), g) for m in VAL}
            r3.append({"scene": g, "cell": LABEL[v], **{NAME[m]: stat(vals[m], ND[m]) for m in VAL}})
            r3_csv.append({"scene": g, "cell": LABEL[v], **{f"{m}_seed{s}": float(x) for m in VAL
                                                           for s, x in zip(S, vals[m])}})
    save("R3_2x2_cells", r3_csv)
    tables["R3. The 2×2 on the varying-exposure set (official for reference): validation per cell"] = r3

    four = all(cells[("vexp", v)] for v in ("detach", "detachexp", "rung2", "rung2exp"))
    r4, r4_csv = [], []
    if four:
        for g in G:
            wide = {}  # effect -> row with one column per metric
            for m in VAL:
                vals = {v: per_seed(getter("vexp", v, m), g) for v in ("detach", "detachexp", "rung2", "rung2exp")}
                main_e, simple_e, noise = factorial(vals)
                if ctx.one_seed:
                    ref = float(np.sqrt(np.mean([ref_sd("vexp", v, m, g) ** 2 for v in ("detach", "detachexp")])))
                for name, x in main_e.items():
                    mu = float(np.mean(x))
                    if ctx.one_seed:
                        txt, xn, real = f"{mu:+.{ND[m]}f} (ref sd {ref:.{ND[m]}f})", float("nan"), None
                    else:
                        xn = mu / noise if noise > 0 else float("nan")
                        real = bool(noise > 0 and abs(mu) > REAL * noise)
                        txt = f"{mu:+.{ND[m]}f} ({xn:+.1f}×){' *' if real else ''}"
                    wide.setdefault(name, {"scene": g, "effect": name})[NAME[m]] = txt
                    r4_csv.append({"scene": g, "metric": m, "effect": name, "mean": mu, "noise": noise,
                                   "x_noise": xn, "real": real})
                for name, (xa, xb) in simple_e.items():
                    ref_v = "detachexp" if "exp column" in name else "detach"
                    txt, e = diff_str(xa, xb, ND[m], ref_sd("vexp", ref_v, m, g))
                    wide.setdefault(name, {"scene": g, "effect": name})[NAME[m]] = txt
                    r4_csv.append({"scene": g, "metric": m, "effect": name, "mean": e["diff"],
                                   "noise": e.get("noise", float("nan")), "x_noise": e.get("x_noise", float("nan")),
                                   "real": e.get("real")})
            r4.extend(wide.values())
        save("R4_factorial", r4_csv)
        numbers["R4"] = r4_csv
        tables["R4. Factorial effects on the varying set. Main effects and interaction: noise over the four cells "
               "(rung05_analyze.effects); simple effects: their two arms' noise (M1). One seed: ref sd = seed sd of "
               "detach and detachexp over seeds 0-2 (main effects) or of the second arm's cell"] = r4

    # ================================================================ R5a leak
    rows_leak = [v for v in ("baseline", "detach", "rung2") if cells[("vexp", v)] and cells[("lom", v)]]
    leak = {(v, sc, s): leak_stats(runs[("vexp", v, sc, s)], runs[("lom", v, sc, s)], u_by[sc])
            for v in rows_leak for sc in ctx.scenes for s in ALL_SEEDS
            if ("vexp", v, sc, s) in runs and ("lom", v, sc, s) in runs}  # all seeds: detach's sd is the yardstick
    detach_leak_all = all(("detach", sc, s) in leak for sc in ctx.scenes for s in ALL_SEEDS)
    r5a, r5a_csv = [], []
    for g in ctx.scenes + (["pooled"] if len(ctx.scenes) > 1 else []):
        for v in rows_leak:
            row = {"scene": g, "curve row": LABEL[v]}
            for k in ("pseudo_leak_paired", "pseudo_leak", "input_leak_paired"):
                x = per_seed(lambda sc, s, k=k, v=v: leak[(v, sc, s)][k], g)
                row[k] = stat(x, 3)
                r5a_csv.append({"scene": g, "row": LABEL[v], "metric": k, **{f"seed{s}": float(y) for s, y in zip(S, x)}})
            row["render_leak_paired"] = ""
            r5a.append(row)
        for c in vexp_v:  # the render leak per cell; LOM partner = the row's LOM run
            if cells[("lom", ROW[c])]:
                x = per_seed(lambda sc, s, c=c: leak_stats(runs[("vexp", c, sc, s)], runs[("lom", ROW[c], sc, s)],
                                                           u_by[sc])["render_leak_paired"], g)
                r5a.append({"scene": g, "curve row": f"{LABEL[c]} (render)", "pseudo_leak_paired": "",
                            "pseudo_leak": "", "input_leak_paired": "", "render_leak_paired": stat(x, 3)})
                r5a_csv.append({"scene": g, "row": LABEL[c], "metric": "render_leak_paired",
                                **{f"seed{s}": float(y) for s, y in zip(S, x)}})
        if "rung2" in rows_leak and "detach" in rows_leak:
            ref = float(np.std(per_seed(lambda sc, s: leak[("detach", sc, s)]["pseudo_leak_paired"], g, ALL_SEEDS),
                               ddof=1)) if detach_leak_all else float("nan")
            txt, e = diff_str(per_seed(lambda sc, s: leak[("rung2", sc, s)]["pseudo_leak_paired"], g),
                              per_seed(lambda sc, s: leak[("detach", sc, s)]["pseudo_leak_paired"], g), 3, ref)
            r5a.append({"scene": g, "curve row": f"{LABEL['rung2']} − detach", "pseudo_leak_paired": txt, "pseudo_leak": "",
                        "input_leak_paired": "", "render_leak_paired": ""})
            numbers.setdefault("R5a_leak_rung2_minus_detach", {})[g] = e
    save("R5a_leak", r5a_csv)
    tables["R5a. Pseudo-target leak (plan 6.3a): slope of log2(mean pseudo-target, varying / LOM) on u_k, same view, "
           "seed and curve row (0 = compensated, 1 = passes straight through); unpaired = log2 mean on the varying set "
           "alone; input = the dark inputs (check: about 1); render rows = the enhanced renders at the training views, "
           "per cell (LOM partner: the row's LOM run)"] = r5a
    if reproduce and "detach" in rows_leak and "baseline" in rows_leak:
        t4 = {(r["scene"], r["method"]): r for r in read_csv(M1_SUMMARY / "T4_exposure.csv")}
        worst = max(abs(float(per_seed(lambda sc, s, v=v: leak[(v, sc, s)]["pseudo_leak_paired"], g).mean())
                        - float(t4[(g, LABEL[v])]["pseudo_leak_paired_mean"]))
                    for v in ("baseline", "detach") for g in SCENES + ["pooled"])
        check("M1 reproduction: paired pseudo-target leak (official, detach) vs T4_exposure.csv", worst, 1e-9)

    # ================================================================ R5b dark fit
    r5b, r5b_csv, slopes = [], [], []
    dark_cells = [c for c in ("detach", "detachexp", "rung2", "rung2exp") if cells[("vexp", c)]]
    for g in G:
        for c in dark_cells:
            x = per_seed(getter("vexp", c, "dark_fit"), g)
            row = {"scene": g, "cell": LABEL[c], "dark fit, varying": stat(x, 2), "row's LOM level": "",
                   "column effect (exp − shared)": ""}
            if cells[("lom", ROW[c])]:
                row["row's LOM level"] = stat(per_seed(getter("lom", ROW[c], "dark_fit"), g), 2)
            if c in HOLLOW and cells[("vexp", ROW[c])]:
                row["column effect (exp − shared)"] = diff_str(x, per_seed(getter("vexp", ROW[c], "dark_fit"), g), 2,
                                                              ref_sd("vexp", ROW[c], "dark_fit", g))[0]
            r5b.append(row)
            r5b_csv.append({"scene": g, "cell": LABEL[c], **{f"seed{s}": float(y) for s, y in zip(S, x)}})
    pts = {}
    for c in dark_cells:
        if not cells[("lom", ROW[c])]:
            continue
        for sc in ctx.scenes:
            for s in S:
                rv, rl = runs[("vexp", c, sc, s)], runs[("lom", ROW[c], sc, s)]
                for n in rv["pseudo"]:
                    pts.setdefault((c, sc), []).append((u_by[sc][n], dark_fit(rv["pseudo"][n]) - dark_fit(rl["pseudo"][n]), s))
    for g in ctx.scenes + (["pooled"] if len(ctx.scenes) > 1 else []):
        for c in dark_cells:
            if (c, ctx.members(g)[0]) not in pts:
                continue
            p = np.array([q for sc in ctx.members(g) for q in pts[(c, sc)]])
            for sign in ("u<0", "u>0"):
                q = p[(p[:, 0] < -1e-9) if sign == "u<0" else (p[:, 0] > 1e-9)]
                q = q[np.isfinite(q[:, 1])]
                slopes.append({"scene": g, "cell": LABEL[c], "u_sign": sign, "n_view_seeds": len(q),
                               "mean_diff_db": float(q[:, 1].mean()),
                               "slope_db_per_stop": float(np.polyfit(q[:, 0], q[:, 1], 1)[0])})
    save("R5b_dark_fit", r5b_csv)
    save("R5b_dark_fit_slopes", slopes)
    tables["R5b. Dark-branch fit (plan 6.3b): PSNR of the dark render as its loss sees it against the input, mean over "
           "training views; the row's LOM level; the column effect within each curve row"] = r5b
    tables["R5b'. Per view: paired change of the dark fit, varying − LOM (same view, seed and curve row), against "
           "signed u_k, darkened (u<0) and brightened (u>0) views separately (M1 T5b; views with u = 0 left out)"] = [
        {**r, "mean_diff_db": f"{r['mean_diff_db']:+.2f}", "slope_db_per_stop": f"{r['slope_db_per_stop']:+.2f}"}
        for r in slopes]
    if reproduce and cells[("vexp", "detach")]:
        t5b = {(r["scene"], r["u_sign"]): r for r in read_csv(M1_SUMMARY / "T5b_paired_vs_signed_u.csv")
               if r["variant"] == "detach" and r["metric"] == "fit_psnr_lowrender_vs_input"}
        worst = max(max(abs(r["mean_diff_db"] - float(t5b[(r["scene"], r["u_sign"])]["mean_diff_db"])),
                        abs(r["slope_db_per_stop"] - float(t5b[(r["scene"], r["u_sign"])]["slope_db_per_stop"])))
                    for r in slopes if r["cell"] == "detach")
        check("M1 reproduction: detach dark-fit change vs signed u (T5b)", worst, 1e-9)

    # ================================================================ R5c exposure (Rung 2)
    r5c, r5c_csv, by_step = [], [], {}
    if cells[("vexp", "rung2")] and cells[("lom", "rung2")]:
        rec = {}
        for sc in ctx.scenes:
            for s in S:
                sv_st, sl_st = runs[("vexp", "rung2", sc, s)]["state"], runs[("lom", "rung2", sc, s)]["state"]
                if sv_st is None or sl_st is None:
                    continue
                names = sv_st["view_names"]
                assert names == sl_st["view_names"], f"{sc} seed {s}: view order differs between the sets"
                u = np.array([u_by[sc][n] for n in names])
                ue = np.array([ueff_by[sc][n] for n in names])
                s_v, s_l = np.array(sv_st["s_final"]), np.array(sl_st["s_final"])
                rec[(sc, s)] = {**recovery_stats(s_v, s_l, u, ue),
                                "move_from_init": rmse_c(s_v - np.array(sv_st["s_init"]))}
                steps_v = {}
                for row in runs[("vexp", "rung2", sc, s)]["steps"]:
                    steps_v.setdefault(int(row["step"]), {})[row["view"]] = fnum(row["s"])
                steps_l = {}
                for row in runs[("lom", "rung2", sc, s)]["steps"]:
                    steps_l.setdefault(int(row["step"]), {})[row["view"]] = fnum(row["s"])
                for st in sorted(set(steps_v) & set(steps_l)):
                    by_step.setdefault((sc, s), []).append(
                        (st, rmse_c(np.array([steps_v[st][n] - steps_l[st][n] for n in names]) + u)))
        keys = ["slope", "r", "rmse", "rmse_paired", "move_from_init"]
        if rec:
            t4 = {(r["scene"], r["method"]): r for r in read_csv(M1_SUMMARY / "T4_exposure.csv")}
            for g in ctx.scenes + (["pooled"] if len(ctx.scenes) > 1 else []):
                row = {"scene": g, "method": f"{LABEL['rung2']} (s_k)"}
                for k in keys:
                    x = per_seed(lambda sc, s, k=k: rec[(sc, s)][k], g)
                    row[k] = stat(x, 2 if k == "slope" else 3)
                    r5c_csv.append({"scene": g, "metric": k, **{f"seed{s}": float(y) for s, y in zip(S, x)}})
                r5c.append(row)
                for m in ("init", "oracle", "official", "detach"):
                    if (g, m) in t4:
                        t = t4[(g, m)]
                        r5c.append({"scene": g, "method": f"{m} (M1)" + (" post-hoc s_fit" if m in ("official", "detach") else ""),
                                    **{k: pm(fnum(t[f"{k}_mean"]), fnum(t[f"{k}_sd"]), 2 if k == "slope" else 3)
                                       if f"{k}_mean" in t else "" for k in keys}})
            save("R5c_exposure", r5c_csv)
            save("R5c_paired_rmse_by_step", [{"scene": sc, "seed": s, "step": st, "rmse_paired": v}
                                             for (sc, s), lst in by_step.items() for st, v in lst])
            tables["R5c. Exposure recovery (plan 6.3c): Rung 2's final s_k against the applied gains u_k (ideal slope "
                   "−1, RMSE 0); rmse = RMS of s + u (unpaired, floor near the natural spread); rmse_paired = with the "
                   "same view's s on LOM subtracted (LOM partner: results/rung2); move = rmse_c(s − s_init). Reference "
                   "rows from M1 (results/vexp/summary/T4_exposure.csv). A sanity check, not a test: the init already "
                   "sits at the quantisation floor"] = r5c

    # ================================================================ decision rules (plan 6.5)
    decisions = []
    if not ctx.one_seed:
        if h1:
            h1_name = (f"H1 rule applied to {LABEL['rung2']} (LOM; exploratory, cannot change H1)" if ident
                       else "H1 (LOM)")
            decisions.append(f"- **{h1_name}:** {'; '.join(f'{k}: {v}' for k, v in h1.items())} -> "
                             f"**{'supported' if all(v == 'meets' for v in h1.values()) else 'not supported'}** "
                             f"(all of buu, chair, sofa and bike must meet the rule).")
        if four:
            def eff(g, m, name):
                return next(r for r in r4_csv if r["scene"] == g and r["metric"] == m and r["effect"].startswith(name))
            pooled_groups = [g for g in ("pooled", "pooled_no_shrub") if g in G]
            for g in pooled_groups:
                cm, cl = eff(g, "psnr_gtmean", "curve model"), eff(g, "lpips", "curve model")
                ex = eff(g, "psnr_gtmean", f"{LABEL['rung2exp']} − detachexp")
                pos = [sc for sc in ctx.scenes if sc != "shrub" and eff(sc, "psnr_gtmean", "curve model")["mean"] > 0]
                n_inf = len([sc for sc in ctx.scenes if sc != "shrub"])
                if cm["real"] and cm["mean"] > 0 and not (cl["x_noise"] > REAL) and len(pos) > n_inf / 2:
                    verdict = "supported"
                elif ex["real"] and ex["mean"] > 0:
                    verdict = "supported only with the dark-branch fix"
                elif 1.0 <= abs(cm["x_noise"]) < REAL:
                    verdict = ("inconclusive (between 1× and 2× noise; no seeds-3-4 slot before the freeze, M5 3.2(b))"
                               if ident else
                               "between 1× and 2× noise: run seeds 3 and 4 (plan 5.3) before concluding")
                else:
                    verdict = "not supported"
                decisions.append(
                    f"- **{'H2b(b), H2 rule for ' + LABEL['rung2'] if ident else 'H2'} ({g}):** curve-model main effect "
                    f"on bm PSNR {cm['mean']:+.2f} dB ({cm['x_noise']:+.1f}×, "
                    f"{'real' if cm['real'] else 'not real'}); on LPIPS {cl['mean']:+.4f} ({cl['x_noise']:+.1f}×); "
                    f"positive in {len(pos)} of {n_inf} non-shrub scenes ({', '.join(pos) or 'none'}); exp-column "
                    f"simple effect {ex['mean']:+.2f} dB ({ex['x_noise']:+.1f}×) -> **{verdict}**.")
        e = numbers.get("R5a_leak_rung2_minus_detach", {}).get("pooled")
        if e is not None and "x_noise" in e:
            decisions.append(f"- **Mechanism:** Rung 2's pooled paired leak minus detach's {e['diff']:+.3f} "
                             f"({e['x_noise']:+.1f}×) -> **{'supports H2' if e['real'] and e['diff'] < 0 else 'does not support H2'}** "
                             f"(needs below detach's by more than 2× noise).")
        if not decisions:
            decisions.append("- Pending: the Rung 2 cells are not complete for seeds 0-2.")
    numbers["decisions"] = decisions

    stops = [c for c in integrity if c["ok"] is False]  # the M1 reproduction checks above
    if stops:
        print(md_table(integrity))
        raise SystemExit("[rung2_analyze] integrity check failed (table above)")

    # ================================================================ figures (plan 6.4)
    figs = make_figures(ctx, runs, cells, u_by, oracle_json, by_step)

    # ================================================================ outputs
    meta = {"date": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t_start), "seeds": S,
            "scenes": ctx.scenes, "mode": "smoke" if ctx.smoke else "selftest" if ctx.selftest else "runs",
            "rung2_variant": ctx.rung2_variant,
            "script_sha256_16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}
    save("cells", status)
    save("integrity", integrity)
    (ctx.out / "numbers.json").write_text(json.dumps({"meta": meta, "cells": status, "integrity": integrity,
                                                     **numbers}, indent=1, default=float))
    banner = ("**Preliminary, one seed (seed {}).** No sd, no ×noise and no verdicts: each difference is shown next to "
              "the reference cell's seed sd over seeds 0-2.".format(S[0]) if ctx.one_seed else
              f"Seeds {', '.join(map(str, S))}. Effects paired by seed; ×noise; * = real (above 2× noise).")
    if ctx.smoke:
        banner += " **CODE TEST** on the 300-step smoke runs: the Rung 2 numbers mean nothing."
    if ctx.selftest:
        banner += (" **CODE TEST:** the Rung 2 cells are copies of detach and detachexp, so every curve-model effect "
                   "must be exactly 0.")
    if ident:
        banner += (" **M5 E1 (H2b):** the Rung 2 cells are the identity-init runs `rung2id` (LOM, varying set) and "
                   "`rung2idexp` (varying set, exp column). Their H1 verdict is exploratory; H2b(a), the init effect, "
                   "is `rung2_ablation.py --mode init`.")
    title = ("# Rung 2 with F initialised at the identity (M5 E1, H2b): H1 rule on LOM (exploratory) and the 2×2 on the "
             "varying-exposure set\n" if ident else
             "# Rung 2 results (M3): H1 on LOM and the 2×2 on the varying-exposure set\n")
    md = [title,
          f"{banner} Generated {meta['date']} in {meta['seconds']} s by `rung2_analyze.py` ({meta['script_sha256_16']}); "
          f"plan `M2_rung2_plan.md` section 6{', readings M5_plan.md 3.2' if ident else ''}. Every number is "
          f"*measured*; readings are the report's job.\n",
          "## Cells\n", md_table(status), "\n## Integrity and input checks\n", md_table(integrity),
          "\n## Decision rules (plan 6.5)\n",
          "\n".join(decisions) if decisions else "_One seed: no verdicts._", "\n"]
    for title, rows in tables.items():
        md += [f"\n## {title}\n", md_table(rows)]
    md += ["\n## Figures\n"] + [f"- `figures/{f}.png`" for f in figs]
    (ctx.out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[rung2_analyze] wrote {ctx.out} in {time.time() - t_start:.0f} s")


# ------------------------------------------------------------------ figures
def make_figures(ctx, runs, cells, u_by, oracle_json, by_step):
    S, done = ctx.seeds, []

    def style(v):
        return dict(marker=MARKER[v], edgecolors=COLOR[v], linewidths=1.4,
                    facecolors="none" if v in HOLLOW else COLOR[v])

    def legend_below(fig, ax, ncol):
        h, lab = ax.get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.08), fontsize=9)

    # 1. H1: bm PSNR and LPIPS per scene on LOM, dots per seed
    lom_v = [v for v in LOM_VARIANTS if cells[("lom", v)]]
    if lom_v:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        offs = np.linspace(-0.22, 0.22, len(lom_v))
        for ax, (m, lab) in zip(axes, (("psnr_gtmean", "brightness-matched PSNR (dB) ↑"), ("lpips", "LPIPS ↓"))):
            for j, sc in enumerate(ctx.scenes):
                for v, dx in zip(lom_v, offs):
                    ys = [fnum(runs[("lom", v, sc, s)]["val"][m]) for s in S]
                    ax.scatter([j + dx] * len(ys), ys, s=30, zorder=3, label=LABEL[v] if j == 0 else None, **style(v))
                    ax.hlines(np.mean(ys), j + dx - 0.07, j + dx + 0.07, color=COLOR[v], linewidth=2, zorder=2)
            ax.set_xticks(range(len(ctx.scenes)))
            ax.set_xticklabels(ctx.scenes)
            ax.set_ylabel(lab)
            ax.grid(axis="x", visible=False)
        legend_below(fig, axes[0], len(lom_v))
        fig.suptitle("H1 on LOM: validation per run (dots = seeds, bar = mean)", x=0.01, ha="left")
        fig.tight_layout()
        savefig(fig, "fig1_H1_lom", ctx.fig)
        done.append("fig1_H1_lom")

    # 2. the 2x2 interaction per scene (bm PSNR)
    if all(cells[("vexp", v)] for v in ("detach", "detachexp", "rung2", "rung2exp")):
        panels = ctx.scenes + (["pooled"] if len(ctx.scenes) > 1 else [])
        fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.8), squeeze=False)
        for ax, g in zip(axes[0], panels):
            for row, exp_v in (("detach", "detachexp"), ("rung2", "rung2exp")):
                ys = [[np.mean([fnum(runs[("vexp", v, sc, s)]["val"]["psnr_gtmean"]) for sc in ctx.members(g)])
                       for s in S] for v in (row, exp_v)]
                for x, (v, y) in enumerate(zip((row, exp_v), ys)):
                    ax.scatter([x] * len(y), y, s=26, zorder=3, label=f"{LABEL[v]}" if g == panels[0] else None,
                               **style(v))
                ax.plot([0, 1], [np.mean(ys[0]), np.mean(ys[1])], color=COLOR[row], linewidth=1.8, zorder=2)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["shared", "exposure-aware"])
            ax.set_xlim(-0.4, 1.4)
            ax.set_title(g)
            ax.grid(axis="x", visible=False)
        axes[0, 0].set_ylabel("brightness-matched PSNR (dB) ↑")
        legend_below(fig, axes[0, 0], 4)
        fig.suptitle("Varying-exposure set: dark branch (x) × curve row (colour); dots = seeds, lines join cell means",
                     x=0.01, ha="left")
        fig.tight_layout()
        savefig(fig, "fig2_2x2_interaction", ctx.fig)
        done.append("fig2_2x2_interaction")

    # 3. leak: paired pseudo-target brightness change against u_k, seed 0
    leak_rows = [v for v in ("baseline", "detach", "rung2") if cells[("vexp", v)] and cells[("lom", v)]]
    s0 = S[0]
    if leak_rows:
        fig, axes = plt.subplots(1, len(ctx.scenes), figsize=(3.4 * len(ctx.scenes), 3.6), sharey=True, squeeze=False)
        for ax, sc in zip(axes[0], ctx.scenes):
            uu = np.array(sorted(u_by[sc].values()))
            ax.plot(uu, uu, color=MUTED, linestyle="--", linewidth=1.1, label="passes through (slope 1)")
            ax.axhline(0, color=INK2, linestyle=":", linewidth=1.1, label="compensated (slope 0)")
            for v in leak_rows:
                rv, rl = runs[("vexp", v, sc, s0)], runs[("lom", v, sc, s0)]
                names = list(rv["pseudo"])
                x = [u_by[sc][n] for n in names]
                y = [np.log2(fnum(rv["pseudo"][n]["mean_pseudo"]) / fnum(rl["pseudo"][n]["mean_pseudo"])) for n in names]
                ax.scatter(x, y, s=22, zorder=3, label=LABEL[v], **style(v))
            ax.set_title(sc)
            ax.set_xlabel("applied gain u_k (stops)")
        axes[0, 0].set_ylabel("log2(mean pseudo-target, varying / LOM)")
        legend_below(fig, axes[0, 0], 5)
        fig.suptitle(f"Pseudo-target leak, seed {s0}: how much of a view's gain reaches its pseudo-target", x=0.01, ha="left")
        fig.tight_layout()
        savefig(fig, "fig3_leak", ctx.fig)
        done.append("fig3_leak")

    # 4. dark fit: paired change against signed u_k, all four cells, seed 0
    dark = [c for c in ("detach", "detachexp", "rung2", "rung2exp") if cells[("vexp", c)] and cells[("lom", ROW[c])]]
    if dark:
        fig, axes = plt.subplots(1, len(ctx.scenes), figsize=(3.4 * len(ctx.scenes), 3.6), sharey=True, squeeze=False)
        for ax, sc in zip(axes[0], ctx.scenes):
            ax.axhline(0, color=MUTED, linewidth=1.0)
            for c in dark:
                rv, rl = runs[("vexp", c, sc, s0)], runs[("lom", ROW[c], sc, s0)]
                names = list(rv["pseudo"])
                ax.scatter([u_by[sc][n] for n in names], [dark_fit(rv["pseudo"][n]) - dark_fit(rl["pseudo"][n])
                                                          for n in names], s=22, zorder=3, label=LABEL[c], **style(c))
            ax.set_title(sc)
            ax.set_xlabel("applied gain u_k (stops)")
        axes[0, 0].set_ylabel("dark fit, varying − LOM (dB)")
        legend_below(fig, axes[0, 0], 4)
        fig.suptitle(f"Dark-branch fit per view, seed {s0} (hollow = exposure-aware dark branch)", x=0.01, ha="left")
        fig.tight_layout()
        savefig(fig, "fig4_dark_fit", ctx.fig)
        done.append("fig4_dark_fit")

    # 5. exposure: Rung 2's s_k against u_k (seed 0) and paired RMSE against step
    if cells[("vexp", "rung2")] and runs[("vexp", "rung2", ctx.scenes[0], s0)]["state"] is not None:
        fig, axes = plt.subplots(1, len(ctx.scenes) + 1, figsize=(3.4 * (len(ctx.scenes) + 1), 3.6), squeeze=False)
        for ax, sc in zip(axes[0], ctx.scenes):
            st = runs[("vexp", "rung2", sc, s0)]["state"]
            o = oracle_json[sc]
            u = np.array([u_by[sc][n] for n in st["view_names"]])
            line = np.linspace(u.min(), u.max(), 2)
            ax.plot(line, -(line - u.mean()), color=MUTED, linestyle="--", linewidth=1.1, label="ideal: s = −(u − mean u)")
            ax.scatter(o["u"], o["s_init"], marker="x", s=30, color=INK2, linewidths=1.2, zorder=2, label="init")
            ax.scatter(o["u"], o["s_oracle"], marker="^", s=32, facecolors="none", edgecolors=SERIES[2], linewidths=1.2,
                       zorder=2, label="histogram oracle")
            ax.scatter(u, st["s_final"], s=22, zorder=3, label=f"{LABEL['rung2']} s_k", **style("rung2"))
            ax.set_title(sc)
            ax.set_xlabel("applied gain u_k (stops)")
        axes[0, 0].set_ylabel("exposure s_k (stops)")
        ax = axes[0, -1]
        for j, sc in enumerate(ctx.scenes):
            if (sc, s0) in by_step:
                st_, v = zip(*by_step[(sc, s0)])
                ax.plot(st_, v, marker="o", markersize=3, color=SERIES[j % len(SERIES)], label=sc)
        ax.set_xlabel("training step")
        ax.set_ylabel("paired RMSE of s + u (stops)")
        ax.set_title("paired RMSE during training")
        ax.legend(fontsize=8)
        legend_below(fig, axes[0, 0], 4)
        fig.suptitle(f"Rung 2 exposures on the varying-exposure set, seed {s0}", x=0.01, ha="left")
        fig.tight_layout()
        savefig(fig, "fig5_exposure", ctx.fig)
        done.append("fig5_exposure")

    # 6. curves, one scene, seed 0: F, the per-view composed curves, the HE-CDFs and detach's per-view curves
    sc = FIG_SCENE if FIG_SCENE in ctx.scenes else ctx.scenes[0]
    keys = [("lom", "rung2"), ("vexp", "rung2")]
    if all(cells[k] and runs[(k[0], k[1], sc, s0)]["npz"].exists() for k in keys):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, (set_, _) in zip(axes, keys):
            z = np.load(runs[(set_, "rung2", sc, s0)]["npz"])
            x = np.linspace(0, 1, z["curve_adj"].shape[1])
            cdf = z["cdf_target"]
            for k in range(len(cdf)):
                ax.plot(x, cdf[k], color=MUTED, lw=0.8, alpha=0.6, label="views' HE-CDFs" if k == 0 else None)
            if cells[(set_, "detach")] and runs[(set_, "detach", sc, s0)]["npz"].exists():
                zd = np.load(runs[(set_, "detach", sc, s0)]["npz"])
                for k, cv in enumerate(zd["curve_adj"]):
                    ax.plot(x, cv, color=COLOR["detach"], lw=0.8, alpha=0.5, label="detach, per-view curves" if k == 0 else None)
            for k, cv in enumerate(z["curve_adj"]):
                ax.plot(x, cv, color=COLOR["rung2"], lw=1.0, alpha=0.7,
                        label=f"{LABEL['rung2']}, F(min(1, 2^s_k x))" if k == 0 else None)
            ax.plot(x, z["curve_global"], color=INK, lw=2.2, label=f"{LABEL['rung2']}, shared F")
            xmax = min(1.0, 1.25 * float(np.argmax(cdf >= 0.995, axis=1).max()) / (len(x) - 1))
            ax.set_xlim(0, xmax)
            ax.set_ylim(0, 1.02)
            ax.set_title("LOM" if set_ == "lom" else "varying-exposure set")
            ax.set_xlabel("input value x (0-1)")
        axes[0].set_ylabel("output")
        legend_below(fig, axes[0], 4)
        fig.suptitle(f"{sc}, seed {s0}: Rung 2's curves after training against detach's", x=0.01, ha="left")
        fig.tight_layout()
        savefig(fig, "fig6_curves", ctx.fig)
        done.append("fig6_curves")
    return done


if __name__ == "__main__":
    main()
