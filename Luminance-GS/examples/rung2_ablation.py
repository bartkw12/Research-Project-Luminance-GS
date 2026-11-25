"""M5 paired comparisons (Implementation_guides/M5_plan.md step 3; readings pre-registered in docs/RESEARCH_LOG.md,
2026-11-24, and M5_plan.md 3.1-3.6). CPU only. Reads finished, re-scored runs; writes one summary folder per job.

Modes (pair by seed; noise = sqrt(mean of the arms' seed variances); real above 2x; pooled = mean over the five
scenes per seed, checked against pooled without shrub; shrub never decides):
    h3       E2 (H3, restated): rung2att - rung2 on LOM (results/rung2) and the varying set (results/vexp, shared
             column), help and harm signals of M5 3.3, leak, pooled s_k slope, s drift   -> results/rung2/summary_h3/
    noprior  E4: rung2noprior - rung2 on LOM, every metric plus F at grid levels 1-4    -> results/rung2/summary_noprior/
    h1       the H1 rule (plan 6.5) for A - B on LOM data, official alongside; arms may span roots by seed; checks every
             run's exp_name. E3: rung2id - detach in results/h1b (seeds 3-5); E6: rung2 - detach in results/overexp
             (--exp_name over_exp), with --extra rung2id                                   -> --out
    init     E1 H2b(a): the init x dark-branch 2x2 on the varying set (rung2, rung2exp, rung2id, rung2idexp), noise over
             the four cells (rung05_analyze.effects)                                       -> results/rung2/summary_h2b/

Every run must exist, have an ok row in its root's runs.csv, be re-scored against results/rung0/<scene>_baseline_seed0,
and match its variant's flags, data set, exp_name, seed and step count; otherwise the script stops before any table.
--selftest reads the arm under test from its reference arm's runs (h1: A from B), so every effect must be exactly 0;
it writes to the git-ignored results/rung2/smoke_m5/. The output guard refuses M3's finished folders (rule 6).

Run from Luminance-GS/examples:
    python rung2_ablation.py --mode h3
    python rung2_ablation.py --mode noprior
    python rung2_ablation.py --mode init
    python rung2_ablation.py --mode h1 --a rung2id --a_roots ../results/h1b --b detach --b_roots ../results/h1b
        --seeds 3 4 5 --out ../results/h1b/summary                                         (E3, H1b)
    python rung2_ablation.py --mode h1 --a rung2 --a_roots ../results/overexp --b detach --b_roots ../results/overexp
        --official_roots ../results/overexp --exp_name over_exp --extra rung2id --out ../results/overexp/summary   (E6)
    python rung2_ablation.py --mode <mode> --selftest [...]
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))

from build_vexp import slope_r  # noqa: E402,F401
from rung0_analyze import SCENES, TIMING_EXCLUDE, md_table, run_name  # noqa: E402
from rung2_analyze import (CFG_DEFAULTS, DATASET, EXPECTED, H1_SCENES, LOM_DATA, ND, NAME, PROTECTED, REAL,  # noqa: E402
                           REF_SEED, REF_VARIANT, RESULTS, RUNG0, RUNG2, RUNG2_FLAGS, VAL, VEXP, VEXP_DATA,
                           dark_fit, fnum, leak_stats, load_run, stat)
from vexp_analyze import as_path, fx, last_ok_minutes, paired, recovery_stats, rmse_c  # noqa: E402

H1B, OVEREXP = RESULTS / "h1b", RESULTS / "overexp"
SMOKE = RUNG2 / "smoke_m5"
STEPS = 10_000
EXPECTED_ALL = {**EXPECTED,
                "rung2att": {"detach_pseudo_target": False, "dark_branch": "shared", **RUNG2_FLAGS, "rung2_attach": "s"},
                "rung2noprior": {"detach_pseudo_target": False, "dark_branch": "shared", **RUNG2_FLAGS,
                                 "rung2_prior": "none"}}
M5_TIMING_EXCLUDE = {("vexp", "chair_rung2id_seed2")}  # the PC slept during it (log, 2026-11-25)
EXTRA = ["hue_err_sat_deg", "ang_err_deg", "sat_ratio", "pseudo_bm", "dark_fit", "num_GS", "params", "minutes"]
WORSE = {"psnr_gtmean": -1, "ssim": -1, "lpips": +1}  # the H1 rule's worse side per metric (plan 6.5)
LEAK_MAX, SLOPE_RANGE = 0.1, (-1.1, -0.9)  # M5 3.3 harm signals (varying set, pooled)
DEFAULT_OUT = {"h3": RUNG2 / "summary_h3", "noprior": RUNG2 / "summary_noprior", "init": RUNG2 / "summary_h2b"}


# ------------------------------------------------------------------ output guard
def guard_out(out: Path) -> Path:
    """Never M3's finished folders (rule 6); otherwise only results/{rung2,h1b,overexp}/summary*/ or a git-ignored
    results/**/smoke*/ folder."""
    out = out.resolve()
    if any(out == p.resolve() or p.resolve() in out.parents for p in PROTECTED):
        raise SystemExit(f"[rung2_ablation] refusing to write to {out}: a finished M3 folder (rule 6)")
    in_smoke = any(p.name.startswith("smoke") and RESULTS.resolve() in p.parents for p in out.parents)
    roots = {RUNG2.resolve(), H1B.resolve(), OVEREXP.resolve()}
    if not (in_smoke or (out.parent in roots and out.name.startswith("summary"))):
        raise SystemExit(f"[rung2_ablation] refusing to write to {out}")
    return out


# ------------------------------------------------------------------ arms and loading
def arm(name, variant, roots, set_="lom", exp_name="low", ref=None):
    """One arm: its label, run-folder variant, roots (searched in order; a run in two roots is an error), data set and
    exp_name. `ref` = the reference arm's variant and roots, used by --selftest."""
    return {"name": name, "variant": variant, "expect": variant, "roots": [Path(r).resolve() for r in roots],
            "set": set_, "exp_name": exp_name, "ref": ref}


def as_selftest(a):
    """--selftest: read this arm from its reference arm's runs (same label, so every effect against the reference is 0)."""
    if a["ref"] is None:
        return a
    v, roots = a["ref"]
    return {**a, "variant": v, "expect": v, "roots": [Path(r).resolve() for r in roots]}


_OK = {}


def ok_rows(root):
    if root not in _OK:
        _OK[root] = last_ok_minutes(root / "runs.csv") if (root / "runs.csv").exists() else {}
    return _OK[root]


def load_arm(a, scenes, seeds, problems):
    out = {}
    for sc in scenes:
        for s in seeds:
            name = run_name(sc, a["variant"], s)
            found = [rt for rt in a["roots"] if (rt / name / "rescore" / "val_summary.json").exists()
                     and name in ok_rows(rt)]
            if not found:
                problems.append(f"{a['name']}: {name} missing, not re-scored or no ok row in "
                                f"{' / '.join(rt.name + '/runs.csv' for rt in a['roots'])}")
                continue
            if len(found) > 1:
                problems.append(f"{a['name']}: {name} found in several roots {[rt.name for rt in found]}")
                continue
            r = load_run(found[0], sc, a["variant"], s, ok_rows(found[0]))
            r.update(set=a["set"], variant=a["variant"], source_variant=a["variant"])
            check_run(r, a, sc, s, problems)
            out[(sc, s)] = r
    return out


def check_run(r, a, sc, s, problems):
    """M2 plan 6.1 plus M5 3.6: data set, flags, exp_name, seed, steps, reference depth, rescore = training eval."""
    c, tag = r["cfg"], f"{r['dir'].parent.name}/{r['dir'].name}"
    want = ((VEXP_DATA if a["set"] == "vexp" else LOM_DATA) / sc).resolve()
    if as_path(c["data_dir"]) != want:
        problems.append(f"{tag}: data_dir {c['data_dir']} is not {want}")
    if c.get("exp_name", "low") != a["exp_name"]:
        problems.append(f"{tag}: exp_name {c.get('exp_name')!r}, expected {a['exp_name']!r}")
    for k, x in {"freeze_color_basis": False, "detach_curve_in_target": False, **EXPECTED_ALL[a["expect"]]}.items():
        got = c.get(k, CFG_DEFAULTS.get(k))
        if (abs(got - x) > 1e-12) if isinstance(x, float) else got != x:
            problems.append(f"{tag}: {k} = {got!r}, expected {x!r} for {a['expect']}")
    if int(c["max_steps"]) != STEPS:
        problems.append(f"{tag}: max_steps {c['max_steps']}")
    if int(c["seed"]) != s:
        problems.append(f"{tag}: seed {c['seed']}, expected {s}")
    ref = (RUNG0 / run_name(sc, REF_VARIANT, REF_SEED)).resolve()
    if as_path(r["val"]["ref_depth_run"]) != ref:
        problems.append(f"{tag}: reference depth {r['val']['ref_depth_run']}, not {ref}")
    d = abs(fnum(r["val"]["psnr"]) - fnum(r["stats"]["psnr"]))
    if d > 1e-3:
        problems.append(f"{tag}: |rescore - training eval| psnr {d:.2e}")


def load_all(arms, scenes, seeds):
    problems = []
    runs = {a["name"]: load_arm(a, scenes, seeds, problems) for a in arms}
    if problems:
        raise SystemExit("[rung2_ablation] stopped before any table:\n  " + "\n  ".join(problems))
    return runs


# ------------------------------------------------------------------ per-run values and effects
def value(r, key):
    if key in ("psnr_gtmean", "psnr", "ssim", "lpips", "sat_ratio", "ang_err_deg", "hue_err_sat_deg"):
        return fnum(r["val"][key])
    if key == "pseudo_bm":
        return float(np.nanmean([fnum(x["pseudo_psnr_gtmean"]) for x in r["pseudo"].values()]))
    if key == "dark_fit":
        return float(np.nanmean([dark_fit(x) for x in r["pseudo"].values()]))
    if key == "num_GS":
        return float(r["stats"]["num_GS"])
    if key == "params":
        return float(r["state"]["n_params"]) if r["state"] is not None else float("nan")
    if key == "minutes":
        root, name = r["dir"].parent, r["dir"].name
        if (root.name, name) in M5_TIMING_EXCLUDE or (root == RUNG0 and name in TIMING_EXCLUDE):
            return float("nan")
        return float(r["minutes"])
    raise KeyError(key)


def effect(a, b):
    """paired(a, b), dropping seed pairs with a missing value (a wall time left out); needs two pairs."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 2:
        return {"diff": float("nan"), "diff_sd": float("nan"), "noise": float("nan"), "x_noise": float("nan"),
                "real": False, "n": int(keep.sum())}
    return {**paired(a[keep], b[keep]), "n": int(keep.sum())}


def groups(scenes):
    g = {sc: [sc] for sc in scenes}
    g["pooled"] = list(scenes)
    g["pooled_no_shrub"] = [s for s in scenes if s != "shrub"]
    return g


class Book:
    """Collects tables, CSV rows and numbers for one job."""

    def __init__(self, runs, scenes, seeds):
        self.runs, self.scenes, self.seeds, self.G = runs, scenes, seeds, groups(scenes)
        self.tables, self.cells, self.effects, self.numbers = {}, [], [], {}

    def per_seed(self, arm_name, fn, g):
        return np.array([np.mean([fn(self.runs[arm_name][(sc, s)]) for sc in self.G[g]]) for s in self.seeds])

    def vals(self, arm_name, key, g):
        return self.per_seed(arm_name, lambda r: value(r, key), g)

    def cell_rows(self, title, arm_names, keys):
        rows = []
        for g in self.G:
            for an in arm_names:
                row = {"scene": g, "arm": an}
                for k in keys:
                    x = self.vals(an, k, g)
                    row[NAME[k]] = stat(x, ND[k])
                    self.cells.append({"scene": g, "arm": an, "metric": k,
                                       **{f"seed{s}": float(y) for s, y in zip(self.seeds, x)}})
                rows.append(row)
        self.tables[title] = rows

    def diff_rows(self, title, pairs, keys):
        """pairs = [(label, arm a, arm b)] -> a - b per metric, paired by seed."""
        rows = []
        for g in self.G:
            for label, an, bn in pairs:
                row = {"scene": g, "comparison": label}
                for k in keys:
                    e = effect(self.vals(an, k, g), self.vals(bn, k, g))
                    row[NAME[k]] = fx(e, ND[k]) if np.isfinite(e["diff"]) else "n/a"
                    self.effects.append({"scene": g, "comparison": label, "metric": k, **e})
                    self.numbers.setdefault("effects", {}).setdefault(label, {}).setdefault(g, {})[k] = e
                rows.append(row)
        self.tables[title] = rows

    def eff(self, label, g, k):
        return self.numbers["effects"][label][g][k]


def h1_rule(book, label, scenes=H1_SCENES):
    """Plan 6.5: bm PSNR not below -2x noise, SSIM and LPIPS not worse beyond 2x noise, in each decision scene."""
    rows, n_pass, per_scene = [], 0, {}
    for sc in scenes:
        fails = []
        for k, sign in WORSE.items():
            e = book.eff(label, sc, k)
            ok = not (sign * e["x_noise"] > REAL)
            n_pass += ok
            if not ok:
                fails.append(NAME[k])
            rows.append({"scene": sc, "metric": NAME[k], label: fx(e, ND[k]),
                         "bar (2× noise, worse side)": f"{sign * REAL * e['noise']:+.{ND[k]}f}",
                         "check": "meets" if ok else "fails"})
        per_scene[sc] = "meets" if not fails else "fails: " + ", ".join(fails)
    return rows, n_pass, per_scene


# ------------------------------------------------------------------ modes
def mode_h1(a, book_args):
    A = arm(a.a, a.a, a.a_roots, exp_name=a.exp_name, ref=(a.b, a.b_roots))
    B = arm(a.b, a.b, a.b_roots, exp_name=a.exp_name)
    arms = [A, B]
    if a.official_roots:
        arms.append(arm("official", "baseline", a.official_roots, exp_name=a.exp_name))
    for v in a.extra or []:
        arms.append(arm(v, v, a.a_roots, exp_name=a.exp_name, ref=(a.b, a.b_roots)))
    if a.selftest:
        arms = [as_selftest(x) for x in arms]
    runs = load_all(arms, *book_args)
    book = Book(runs, *book_args)
    names = [x["name"] for x in arms]
    lab = f"{A['name']} − {B['name']}"
    book.cell_rows("Validation per arm (mean ± sd over seeds; pooled = mean over scenes per seed)", names, VAL)
    pairs = [(lab, A["name"], B["name"])]
    if "official" in names:
        pairs += [(f"{B['name']} − official", B["name"], "official"), (f"{A['name']} − official", A["name"], "official")]
    for v in a.extra or []:
        pairs += [(f"{v} − {B['name']} (exploratory)", v, B["name"]), (f"{v} − {A['name']} (exploratory)", v, A["name"])]
    book.diff_rows("Paired by seed (×noise, * = real above 2×)", pairs, VAL)
    book.diff_rows("Extras, paired by seed (reported only)", pairs, EXTRA)
    rows, n_pass, per_scene = h1_rule(book, lab)
    verdict = "supported" if n_pass == 3 * len(H1_SCENES) else "not supported"
    dec = [f"- **H1 rule for {lab}** ({a.exp_name}; seeds {', '.join(map(str, book.seeds))}): "
           f"{'; '.join(f'{k}: {v}' for k, v in per_scene.items())} -> {n_pass} of {3 * len(H1_SCENES)} checks meet "
           f"-> **{verdict}**."]
    if "official" in names:
        e = book.eff(f"{B['name']} − official", "pooled", "psnr_gtmean")
        dec.append(f"- {B['name']} − official, pooled bm PSNR {fx(e, 2)}"
                   + ("" if e["real"] and e["diff"] > 0 else
                      f" -> {B['name']} does not beat the official code here beyond 2× noise, so matching "
                      f"{B['name']} is a weaker bar (M5 3.6)."))
    book.tables[f"H1 rule (plan 6.5) for {lab}: 12 checks on buu, chair, sofa and bike; shrub reported only"] = rows
    book.numbers["h1"] = {"label": lab, "n_pass": n_pass, "per_scene": per_scene, "verdict": verdict}
    return book, dec


def mode_noprior(a, book_args):
    R = arm("rung2", "rung2", [RUNG2])
    N = arm("rung2noprior", "rung2noprior", [RUNG2], ref=("rung2", [RUNG2]))
    arms = [as_selftest(x) for x in (N, R)] if a.selftest else [N, R]
    runs = load_all(arms, *book_args)
    book = Book(runs, *book_args)
    lab = "rung2noprior − rung2"
    book.cell_rows("Validation and extras per arm (LOM)", ["rung2noprior", "rung2"], VAL + EXTRA)
    book.diff_rows("rung2noprior − rung2 on LOM, paired by seed (×noise, * = real above 2×); no decision depends on it "
                   "(M5 3.4)", [(lab, "rung2noprior", "rung2")], VAL + EXTRA)
    rows = []
    for sc in book.scenes:
        row = {"scene": sc}
        for an in ("rung2noprior", "rung2"):
            F = np.array([np.asarray(runs[an][(sc, s)]["state"]["F_grid"])[1:5] * 255 for s in book.seeds])
            row[f"{an}: F at levels 1–4 (×255)"] = "[" + ", ".join(f"{m:.2f}" for m in F.mean(0)) + "]"
            g = [fnum(runs[an][(sc, s)]["state"]["gamma"]) for s in book.seeds]
            row[f"{an}: γ"] = stat(g, 3)
            book.numbers.setdefault("dark_end", {}).setdefault(sc, {})[an] = {
                "F_levels_1_4_x255_by_seed": F.tolist(), "gamma_by_seed": g,
                "n_params": int(runs[an][(sc, book.seeds[0])]["state"]["n_params"])}
        rows.append(row)
    book.tables["Dark end: F on the 255-bin grid at b = 1–4 (x = b/254), ×255, mean over seeds (2026-11-16 "
                "correction); γ (NaN without the prior)"] = rows
    return book, ["- E4 is reported only: no decision depends on it (M5 3.4)."]


def mode_init(a, book_args):
    cells = [arm("rung2", "rung2", [VEXP], "vexp"), arm("rung2exp", "rung2exp", [VEXP], "vexp"),
             arm("rung2id", "rung2id", [VEXP], "vexp", ref=("rung2", [VEXP])),
             arm("rung2idexp", "rung2idexp", [VEXP], "vexp", ref=("rung2exp", [VEXP]))]
    if a.selftest:
        cells = [as_selftest(x) for x in cells]
    runs = load_all(cells, *book_args)
    book = Book(runs, *book_args)
    book.cell_rows("The init × dark-branch 2×2 on the varying set: validation per cell", [c["name"] for c in cells], VAL)
    main_rows, csv_rows = [], []
    names = {"init (H2b(a)): ½[(rung2id − rung2) + (rung2idexp − rung2exp)]": lambda R, RX, I, IX: 0.5 * ((I - R) + (IX - RX)),
             "dark branch: ½[(rung2exp − rung2) + (rung2idexp − rung2id)]": lambda R, RX, I, IX: 0.5 * ((RX - R) + (IX - I)),
             "interaction: (rung2idexp − rung2id) − (rung2exp − rung2)": lambda R, RX, I, IX: (IX - I) - (RX - R)}
    for g in book.G:
        wide = {}
        for k in VAL:
            R, RX, I, IX = (book.vals(c, k, g) for c in ("rung2", "rung2exp", "rung2id", "rung2idexp"))
            noise = float(np.sqrt(np.mean([np.var(x, ddof=1) for x in (R, RX, I, IX)])))
            for name, fn in names.items():
                mu = float(np.mean(fn(R, RX, I, IX)))
                xn = mu / noise if noise > 0 else float("nan")
                real = bool(noise > 0 and abs(mu) > REAL * noise)
                e = {"diff": mu, "noise": noise, "x_noise": xn, "real": real}
                wide.setdefault(name, {"scene": g, "effect": name})[NAME[k]] = fx(e, ND[k])
                csv_rows.append({"scene": g, "comparison": name, "metric": k, **e})
                book.numbers.setdefault("effects", {}).setdefault(name.split(":")[0], {}).setdefault(g, {})[k] = e
        main_rows.extend(wide.values())
    book.effects.extend(csv_rows)
    book.tables["Main effects and interaction, noise over the four cells (rung05_analyze.effects)"] = main_rows
    book.diff_rows("Simple effects, paired by seed (their two arms' noise)",
                   [("rung2id − rung2 (shared column)", "rung2id", "rung2"),
                    ("rung2idexp − rung2exp (exp column)", "rung2idexp", "rung2exp")], VAL)
    dec = []
    lab = "init (H2b(a))"
    for g in ("pooled", "pooled_no_shrub", "bike"):
        bm, lp = book.eff(lab, g, "psnr_gtmean"), book.eff(lab, g, "lpips")
        if bm["real"] and bm["diff"] > 0:
            verdict = "mixed" if lp["x_noise"] > REAL else "confirmed"
        elif bm["real"] and bm["diff"] < 0:
            verdict = "reversed"
        else:
            verdict = "not confirmed"
        book.numbers.setdefault("verdicts", {})[g] = verdict
        dec.append(f"- **H2b(a), {g}:** init main effect on bm PSNR {fx(bm, 2)}; LPIPS {fx(lp, 4)} -> **{verdict}** "
                   f"(confirmed: bm real and positive, LPIPS not worse beyond 2×; mixed: LPIPS worse beyond 2×; "
                   f"reversed: bm real and negative; else not confirmed).")
    return book, dec


def mode_h3(a, book_args):
    scenes, seeds = book_args
    gains = json.loads((DATASET / "gains.json").read_text())["rows"]
    u_by = {sc: {r["view"]: r["u"] for r in gains if r["scene"] == sc} for sc in SCENES}
    ueff_by = {sc: {r["view"]: r["log2_ratio"] for r in gains if r["scene"] == sc} for sc in SCENES}
    arms = [arm("lom rung2att", "rung2att", [RUNG2], "lom", ref=("rung2", [RUNG2])),
            arm("lom rung2", "rung2", [RUNG2], "lom"),
            arm("vexp rung2att", "rung2att", [VEXP], "vexp", ref=("rung2", [VEXP])),
            arm("vexp rung2", "rung2", [VEXP], "vexp")]
    if a.selftest:
        arms = [as_selftest(x) for x in arms]
    runs = load_all(arms, *book_args)
    book = Book(runs, *book_args)
    keys = VAL + ["hue_err_sat_deg", "ang_err_deg", "sat_ratio", "pseudo_bm", "dark_fit", "num_GS", "minutes"]
    book.cell_rows("Validation and extras per arm", [x["name"] for x in arms], keys)
    book.diff_rows("rung2att − rung2, paired by seed (×noise, * = real above 2×)",
                   [("LOM: rung2att − rung2", "lom rung2att", "lom rung2"),
                    ("varying: rung2att − rung2", "vexp rung2att", "vexp rung2")], keys)

    # mechanism per run: leak (varying run against the same arm's LOM run), s_k recovery, s drift
    mech = {}
    for v in ("rung2att", "rung2"):
        for sc in scenes:
            for s in seeds:
                rv, rl = runs[f"vexp {v}"][(sc, s)], runs[f"lom {v}"][(sc, s)]
                sv, sl = rv["state"], rl["state"]
                names = sv["view_names"]
                assert names == sl["view_names"], f"{sc} seed {s}: view order differs between the sets"
                u = np.array([u_by[sc][n] for n in names])
                ue = np.array([ueff_by[sc][n] for n in names])
                s_v, s_l = np.array(sv["s_final"]), np.array(sl["s_final"])
                rec = recovery_stats(s_v, s_l, u, ue)
                mech[(v, sc, s)] = {"pseudo_leak_paired": leak_stats(rv, rl, u_by[sc])["pseudo_leak_paired"],
                                    "slope": rec["slope"], "rmse_paired": rec["rmse_paired"],
                                    "drift_vexp": rmse_c(s_v - np.array(sv["s_init"])),
                                    "drift_lom": rmse_c(s_l - np.array(sl["s_init"]))}
    mkeys = {"pseudo_leak_paired": 3, "slope": 3, "rmse_paired": 3, "drift_vexp": 4, "drift_lom": 4}
    mrows = []
    for g in book.G:
        for v in ("rung2att", "rung2"):
            row = {"scene": g, "arm": v}
            for k, nd in mkeys.items():
                x = np.array([np.mean([mech[(v, sc, s)][k] for sc in book.G[g]]) for s in seeds])
                row[k] = stat(x, nd)
                book.cells.append({"scene": g, "arm": v, "metric": k, **{f"seed{s}": float(y) for s, y in zip(seeds, x)}})
                book.numbers.setdefault("mechanism", {}).setdefault(g, {}).setdefault(v, {})[k] = x.tolist()
            mrows.append(row)
        row = {"scene": g, "arm": "rung2att − rung2"}
        for k, nd in mkeys.items():
            xa = np.array([np.mean([mech[("rung2att", sc, s)][k] for sc in book.G[g]]) for s in seeds])
            xb = np.array([np.mean([mech[("rung2", sc, s)][k] for sc in book.G[g]]) for s in seeds])
            e = effect(xa, xb)
            row[k] = fx(e, nd)
            book.numbers.setdefault("effects", {}).setdefault("mechanism: rung2att − rung2", {}).setdefault(g, {})[k] = e
        mrows.append(row)
    book.tables["Mechanism (varying set unless named): paired pseudo-target leak (slope of log2 varying/LOM mean pseudo-"
                "target on u_k, same arm), s_k slope on u_k (ideal −1), paired s RMSE against the gains, s drift = "
                "rmse_c(s_final − s_init) on each set"] = mrows

    # readings (M5 3.3)
    lom, var = "LOM: rung2att − rung2", "varying: rung2att − rung2"
    help_, harm = [], []
    for lab, setname in ((lom, "LOM"), (var, "varying")):
        for g in ("pooled", "pooled_no_shrub"):
            bm, lp = book.eff(lab, g, "psnr_gtmean"), book.eff(lab, g, "lpips")
            if bm["real"] and bm["diff"] > 0:
                help_.append((g, f"{setname} {g} bm PSNR {fx(bm, 2)}"))
            if lp["real"] and lp["diff"] < 0:
                help_.append((g, f"{setname} {g} LPIPS {fx(lp, 4)}"))
    for sc in H1_SCENES:
        e = book.eff(lom, sc, "psnr_gtmean")
        if e["x_noise"] < -REAL:
            harm.append(f"LOM {sc} bm PSNR {fx(e, 2)} below −2× noise")
    for lab, setname in ((lom, "LOM"), (var, "varying")):
        e = book.eff(lab, "bike", "hue_err_sat_deg")
        if e["real"] and e["diff"] > 0:
            harm.append(f"bike hue error worse on {setname}: {fx(e, 2)}")
    leak = float(np.mean(book.numbers["mechanism"]["pooled"]["rung2att"]["pseudo_leak_paired"]))
    if leak > LEAK_MAX:
        harm.append(f"varying pooled leak {leak:.3f} > {LEAK_MAX}")
    slope = float(np.mean(book.numbers["mechanism"]["pooled"]["rung2att"]["slope"]))
    if not (SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1]):
        harm.append(f"varying pooled s_k slope {slope:.3f} outside {SLOPE_RANGE[0]} to {SLOPE_RANGE[1]}")
    helps = [t for g, t in help_ if g == "pooled"]
    helps_ns = [t for g, t in help_ if g == "pooled_no_shrub"]
    verdict = ("mixed" if helps and harm else "helps" if helps else "hurts" if harm else "no effect")
    book.numbers["h3"] = {"help": helps, "help_pooled_no_shrub": helps_ns, "harm": harm, "pooled_leak": leak,
                          "pooled_slope": slope, "verdict": verdict}
    dec = [f"- **Help signals (pooled):** {'; '.join(helps) or 'none'}. Pooled without shrub: "
           f"{'; '.join(helps_ns) or 'none'}{'' if bool(helps) == bool(helps_ns) else ' -> **differs from pooled**'}.",
           f"- **Harm signals:** {'; '.join(harm) or 'none'} (pooled leak {leak:.3f}, limit {LEAK_MAX}; pooled s_k slope "
           f"{slope:.3f}, range {SLOPE_RANGE[0]} to {SLOPE_RANGE[1]}).",
           f"- **H3 (restated):** **{verdict}**."]
    return book, dec


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["h3", "noprior", "h1", "init"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    ap.add_argument("--selftest", action="store_true", help="read the arm under test from its reference arm's runs")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--a", help="h1: arm A's variant")
    ap.add_argument("--a_roots", nargs="+", type=Path)
    ap.add_argument("--b", help="h1: arm B's variant (the reference)")
    ap.add_argument("--b_roots", nargs="+", type=Path)
    ap.add_argument("--official_roots", nargs="+", type=Path, default=None, help="h1: where the baseline runs are")
    ap.add_argument("--extra", nargs="+", default=None, help="h1: further variants in A's roots (exploratory)")
    ap.add_argument("--exp_name", default="low", choices=["low", "over_exp"])
    ap.add_argument("--title", default="", help="h1: a note for the heading (e.g. 'context only')")
    a = ap.parse_args()
    if a.mode == "h1" and not (a.a and a.a_roots and a.b and a.b_roots and (a.out or a.selftest)):
        raise SystemExit("--mode h1 needs --a, --a_roots, --b, --b_roots and --out")
    if a.mode != "h1" and (a.a or a.b or a.extra or a.official_roots or a.exp_name != "low"):
        raise SystemExit("--a/--b/--extra/--official_roots/--exp_name apply to --mode h1 only")
    if a.selftest:
        out = a.out or SMOKE / f"ablation_{a.mode}_selftest"
        if not any(p.name.startswith("smoke") for p in out.resolve().parents):
            raise SystemExit("--selftest writes only into a results/**/smoke*/ folder")
    else:
        out = a.out or DEFAULT_OUT[a.mode]
    out = guard_out(out)
    t0 = time.time()
    scenes = [s for s in SCENES if s in a.scenes]
    seeds = sorted(set(a.seeds))
    book, dec = {"h1": mode_h1, "noprior": mode_noprior, "init": mode_init, "h3": mode_h3}[a.mode](a, (scenes, seeds))

    out.mkdir(parents=True, exist_ok=True)
    meta = {"date": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0), "mode": a.mode,
            "selftest": a.selftest, "seeds": seeds, "scenes": scenes, "argv": sys.argv[1:],
            "script_sha256_16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}
    for name, rows in (("cells", book.cells), ("effects", book.effects)):
        if rows:
            with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader()
                w.writerows(rows)
    (out / "numbers.json").write_text(json.dumps({"meta": meta, "decisions": dec, **book.numbers}, indent=1,
                                                 default=float), encoding="utf-8")
    heading = {"h3": "H3 (restated): rung2att − rung2 on LOM and the varying set (M5 E2)",
               "noprior": "No shape prior: rung2noprior − rung2 on LOM (M5 E4)",
               "init": "H2b(a): the identity init on the varying set, init × dark-branch 2×2 (M5 E1)",
               "h1": f"H1 rule: {a.a} − {a.b}, LOM {a.exp_name} (M5 E3/E6){' — ' + a.title if a.title else ''}"}[a.mode]
    md = [f"# {heading}\n",
          ("**SELFTEST:** the arm under test is read from its reference arm's runs; every effect must be 0. " if a.selftest
           else "") +
          f"Seeds {', '.join(map(str, seeds))}. Paired by seed; noise = sqrt(mean of the arms' seed variances); ×noise; "
          f"* = real (above 2×). Pooled = mean over the five scenes per seed. Readings pre-registered in "
          f"docs/RESEARCH_LOG.md (2026-11-24), M5_plan.md 3.1–3.6. Generated {meta['date']} in {meta['seconds']} s by "
          f"`rung2_ablation.py` ({meta['script_sha256_16']}). Every number is *measured*.\n",
          "## Readings\n", "\n".join(dec), ""]
    for title, rows in book.tables.items():
        md += [f"\n## {title}\n", md_table(rows)]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(dec))
    print(f"[rung2_ablation] wrote {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
