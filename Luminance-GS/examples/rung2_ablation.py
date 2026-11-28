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
    lum      E5 (E5_luminance_plan.md; readings in the log, 2026-11-25): rung2idlum - rung2id on LOM (results/rung2),
             the varying set (results/vexp) and over_exp (results/overexp, exploratory), detach alongside; the built-in
             F/s/gamma identity check; the clipped share from each run's final F and s and its training photos (CPU)
                                                                                           -> results/rung2/summary_e5/
    rung3    M6 (M6_rung3_plan.md section 4; readings logged 2026-11-26): rung3 - rung2id on LOM (results/rung3 against
             results/rung2), the varying set (results/vexp) and over_exp (results/overexp, exploratory), detach alongside;
             H4's help and harm signals, bike's hue, the 4.3 mechanism check against the preview's predictions (refused
             unless results/rung3/preview/numbers.json has the sha256 logged before the batch), mechanism, zone curves,
             curve losses, figures                                                         -> results/rung3/summary/

Every run must exist, have an ok row in its root's runs.csv, be re-scored against results/rung0/<scene>_baseline_seed0,
and match its variant's flags, data set, exp_name, seed and step count; otherwise the script stops before any table.
--selftest reads the arm under test from its reference arm's runs (h1: A from B), so every effect must be exactly 0;
it writes to the git-ignored results/rung2/smoke_m5/. The output guard refuses every finished folder (CLAUDE.md rule 6,
listed in protected.py). Since 2026-11-26 that includes M5's and E5's summaries, so every command below that would
rewrite one of them now refuses (their outputs are finished); --selftest still runs.

Run from Luminance-GS/examples:
    python rung2_ablation.py --mode h3
    python rung2_ablation.py --mode noprior
    python rung2_ablation.py --mode init
    python rung2_ablation.py --mode h1 --a rung2id --a_roots ../results/h1b --b detach --b_roots ../results/h1b
        --seeds 3 4 5 --out ../results/h1b/summary                                         (E3, H1b)
    python rung2_ablation.py --mode h1 --a rung2 --a_roots ../results/overexp --b detach --b_roots ../results/overexp
        --official_roots ../results/overexp --exp_name over_exp --extra rung2id --out ../results/overexp/summary   (E6)
    python rung2_ablation.py --mode lum                                                    (E5)
    python rung2_ablation.py --mode h1 --a rung2idlum --a_roots ../results/rung2 --b detach --b_roots ../results/rung0
        --official_roots ../results/rung0 --out ../results/rung2/summary_e5_h1             (E5, exploratory H1 rule)
    python rung2_ablation.py --mode rung3                                                  (M6)
    python rung2_ablation.py --mode h1 --a rung3 --a_roots ../results/rung3 ../results/rung2 --b detach --b_roots
        ../results/rung0 --official_roots ../results/rung0 --extra rung2id --title "M6, exploratory"
        --out ../results/rung3/summary_h1                                                  (M6, exploratory H1 rule)
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
from protected import refuse_finished  # noqa: E402
from rung2_analyze import (CFG_DEFAULTS, DATASET, EXPECTED, H1_SCENES, LOM_DATA, ND, NAME, REAL,  # noqa: E402
                           REF_SEED, REF_VARIANT, RESULTS, RUNG0, RUNG2, RUNG2_FLAGS, VAL, VEXP, VEXP_DATA,
                           dark_fit, fnum, leak_stats, load_run, stat)
from vexp_analyze import as_path, fx, last_ok_minutes, paired, recovery_stats, rmse_c  # noqa: E402

H1B, OVEREXP, RUNG3 = RESULTS / "h1b", RESULTS / "overexp", RESULTS / "rung3"
SMOKE = RUNG2 / "smoke_m5"
STEPS = 10_000
EXPECTED_ALL = {**EXPECTED,
                "rung2att": {"detach_pseudo_target": False, "dark_branch": "shared", **RUNG2_FLAGS, "rung2_attach": "s"},
                "rung2noprior": {"detach_pseudo_target": False, "dark_branch": "shared", **RUNG2_FLAGS,
                                 "rung2_prior": "none"},
                # E5: the colour mode is checked for the identity pair (cfg.json files before E5 have no key: rgb)
                # M6: rung2id must have one zone, so a Rung 3 run cannot pass as the base (cfg.json files before M6 have
                # no key: 1)
                "rung2id": {**EXPECTED["rung2id"], "rung2_color": "rgb", "rung3_zones": 1},
                "rung2idlum": {**EXPECTED["rung2id"], "rung2_color": "luminance"},
                "rung3": {**EXPECTED["rung2id"], "rung2_color": "rgb", "rung3_zones": 2, "rung3_blur": 16}}
CFG_DEFAULTS_ALL = {**CFG_DEFAULTS, "rung2_color": "rgb", "rung3_zones": 1, "rung3_blur": 16}
NAME = {**NAME, "clip_share": "clipped share", "rel_l1_pseudo": "pseudo inconsistency (rel. L1)"}
ND = {**ND, "clip_share": 4, "rel_l1_pseudo": 4}
M5_TIMING_EXCLUDE = {("vexp", "chair_rung2id_seed2")}  # the PC slept during it (log, 2026-11-25)
EXTRA = ["hue_err_sat_deg", "ang_err_deg", "sat_ratio", "pseudo_bm", "dark_fit", "num_GS", "params", "minutes"]
WORSE = {"psnr_gtmean": -1, "ssim": -1, "lpips": +1}  # the H1 rule's worse side per metric (plan 6.5)
LEAK_MAX, SLOPE_RANGE = 0.1, (-1.1, -0.9)  # M5 3.3 harm signals (varying set, pooled)
FS_TOL = 1e-5  # E5 built-in check: F, s and gamma of rung2idlum equal rung2id's at the same seed (M3's tolerance)
DEFAULT_OUT = {"h3": RUNG2 / "summary_h3", "noprior": RUNG2 / "summary_noprior", "init": RUNG2 / "summary_h2b",
               "lum": RUNG2 / "summary_e5", "rung3": RUNG3 / "summary"}
# M6 (M6_rung3_plan.md section 4; readings in the log, 2026-11-26 "M6 step 1"; predictions 2026-11-27 "M6 step 3, test 5")
PREVIEW_M6 = RUNG3 / "preview" / "numbers.json"
PREVIEW_M6_SHA256 = "991f45a9d21bf631d0a7a77a2736b167a8cc07e3ce2d0c0c0a3d77731b93974f"  # as logged before the batch
CONS_HARM = 0.20       # harm 3 (D5): mean per-scene relative change of rel_l1_pseudo over the decision scenes, LOM
Z_LEVELS = [1, 2, 4, 8, 16, 32, 64]  # grid indices b (x = b / 254) at which the zone curves are reported
BIKE_VIEW = "29"       # diagnostics.RESCORE_EXAMPLES


# ------------------------------------------------------------------ output guard
def guard_out(out: Path) -> Path:
    """Never a finished folder (rule 6, protected.py); otherwise only results/{rung2,h1b,overexp,rung3}/summary*/ or a
    git-ignored results/**/smoke*/ folder."""
    out = refuse_finished(out, "rung2_ablation")
    in_smoke = any(p.name.startswith("smoke") and RESULTS.resolve() in p.parents for p in out.parents)
    roots = {RUNG2.resolve(), H1B.resolve(), OVEREXP.resolve(), RUNG3.resolve()}
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
        got = c.get(k, CFG_DEFAULTS_ALL.get(k))
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
    if key == "clip_share":
        return float(r.get("clip_share", float("nan")))
    if key == "rel_l1_pseudo":  # M6 harm 3: mean over the run's rescore/consistency.csv (shared reference depth)
        return float(np.mean([fnum(c["rel_l1_pseudo"]) for c in r["cons"]]))
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


def add_clip_shares(runs, arms, scenes, seeds):
    """E5: each Rung 2 run's clipped share (rung2.ExposureCurve.clipped_share), the mean over its training views of the
    share of channel values its pseudo-target clips, from the run's final F and s and the training photos loaded as
    the trainer loads them (rung2_preview.load_set, factor 1 as the runs). The colour mode is the run's own cfg value,
    so --selftest stays exact. CPU."""
    import torch
    from rung2 import ExposureCurve
    from rung2_preview import load_set
    photos = {}
    for a in arms:
        for sc in scenes:
            for s in seeds:
                r = runs[a["name"]][(sc, s)]
                if r["state"] is None:
                    continue
                key = (a["set"], a["exp_name"], sc)
                if key not in photos:
                    d = load_set(sc, a["set"], a["exp_name"])
                    photos[key] = (d["names"], d["images"])
                names, images = photos[key]
                st = r["state"]
                assert names == st["view_names"], f"{r['dir'].name}: training views differ from rung2_state.json"
                m = ExposureCurve(torch.tensor(st["knots"]), torch.tensor(st["heights"]), torch.tensor(st["s_final"]))
                color = r["cfg"].get("rung2_color", "rgb")
                r["clip_share"] = float(np.mean([m.clipped_share(img, k, color) for k, img in enumerate(images)]))


def mode_lum(a, book_args):
    scenes, seeds = book_args
    sets = [("LOM", "lom", "low", RUNG2), ("varying", "vexp", "low", VEXP), ("over_exp", "lom", "over_exp", OVEREXP)]
    arms, ctx = [], []
    for tag, set_, exp, root in sets:
        arms += [arm(f"{tag} rung2idlum", "rung2idlum", [root], set_, exp, ref=("rung2id", [root])),
                 arm(f"{tag} rung2id", "rung2id", [root], set_, exp)]
        ctx.append(arm(f"{tag} detach", "detach", [RUNG0 if tag == "LOM" else root], set_, exp))
    if a.selftest:
        arms = [as_selftest(x) for x in arms]
    runs = load_all(arms + ctx, *book_args)
    add_clip_shares(runs, arms, scenes, seeds)
    book = Book(runs, *book_args)
    keys = VAL + EXTRA + ["clip_share"]
    book.cell_rows("Validation and extras per arm (detach: context only)", [x["name"] for x in arms + ctx], keys)
    lab = {tag: f"{tag}: rung2idlum − rung2id" + (" (exploratory)" if tag == "over_exp" else "") for tag, *_ in sets}
    book.diff_rows("rung2idlum − rung2id, paired by seed (×noise, * = real above 2×)",
                   [(lab[tag], f"{tag} rung2idlum", f"{tag} rung2id") for tag, *_ in sets], keys)
    book.diff_rows("Against detach, paired by seed (context, reported only)",
                   [(f"{tag}: {v} − detach", f"{tag} {v}", f"{tag} detach") for tag, *_ in sets
                    for v in ("rung2idlum", "rung2id")], keys)

    # built-in check: the curve loss never sees the pseudo-target, so F, s and gamma must equal rung2id's (plan 2)
    fs_rows, worst = [], 0.0
    for tag, *_ in sets:
        for sc in scenes:
            d = {"F": [], "s": [], "gamma": []}
            for s in seeds:
                sl, si = runs[f"{tag} rung2idlum"][(sc, s)]["state"], runs[f"{tag} rung2id"][(sc, s)]["state"]
                assert sl["view_names"] == si["view_names"], f"{tag} {sc} seed {s}: view order differs"
                d["F"].append(float(np.max(np.abs(np.subtract(sl["F_grid"], si["F_grid"])))))
                d["s"].append(float(np.max(np.abs(np.subtract(sl["s_final"], si["s_final"])))))
                d["gamma"].append(abs(fnum(sl["gamma"]) - fnum(si["gamma"])))
            row = {"set": tag, "scene": sc, **{f"max |Δ{k}| over seeds": f"{max(v):.1e}" for k, v in d.items()}}
            worst = max(worst, *(max(v) for v in d.values()))
            fs_rows.append(row)
            book.numbers.setdefault("fs_check", {}).setdefault(tag, {})[sc] = d
    fs_ok = worst <= FS_TOL
    book.numbers["fs_check_max"], book.numbers["fs_check_ok"] = worst, fs_ok
    book.tables[f"Built-in check: F on the 255-bin grid, s_final and γ of rung2idlum against rung2id, same seed "
                f"(tolerance {FS_TOL:g})"] = fs_rows

    # readings (log 2026-11-25, "E5 plan approved"; E5_luminance_plan.md section 4)
    lom, var, over = lab["LOM"], lab["varying"], lab["over_exp"]
    hue = book.eff(var, "bike", "hue_err_sat_deg")
    hue_fix = bool(hue["real"] and hue["diff"] < 0)
    costs = []
    for lb, setname in ((lom, "LOM"), (var, "varying")):
        for sc in H1_SCENES:
            e = book.eff(lb, sc, "psnr_gtmean")
            if e["x_noise"] < -REAL:
                costs.append(f"{setname} {sc} bm PSNR {fx(e, 2)}")
    verdict = "fixes" if hue_fix and not costs else "fixes at a cost" if hue_fix else "no fix"
    worse = {g: [] for g in ("pooled", "pooled_no_shrub")}
    for lb, setname in ((lom, "LOM"), (var, "varying")):
        for g in worse:
            ss, lp = book.eff(lb, g, "ssim"), book.eff(lb, g, "lpips")
            if ss["x_noise"] < -REAL:
                worse[g].append(f"{setname} SSIM {fx(ss, 4)}")
            if lp["x_noise"] > REAL:
                worse[g].append(f"{setname} LPIPS {fx(lp, 4)}")
    adopt = verdict == "fixes" and not worse["pooled"]
    ob, obn = book.eff(over, "pooled", "psnr_gtmean"), book.eff(over, "pooled_no_shrub", "psnr_gtmean")
    gap = book.eff("over_exp: rung2id − detach", "pooled", "psnr_gtmean")
    part = bool(ob["real"] and ob["diff"] > 0)
    frac = ob["diff"] / -gap["diff"] if gap["diff"] < 0 else float("nan")
    sat = {v: stat(book.vals(f"over_exp {v}", "sat_ratio", "pooled"), 3) for v in ("rung2idlum", "rung2id", "detach")}
    book.numbers["e5"] = {"verdict": verdict, "bike_hue_varying": hue, "costs": costs, "adopt": adopt,
                          "adoption_worse": worse, "fs_check_ok": fs_ok, "overexp_part": part,
                          "overexp_gap_closed_fraction": frac}
    dec = [f"- **Built-in check** (F, s, γ identical to rung2id): max |Δ| {worst:.1e} -> "
           f"**{'passes' if fs_ok else 'FAILS: the colour mode changed the curve training; stop and investigate'}**.",
           f"- **Bike hue error on the varying set** (rung2idlum − rung2id): {fx(hue, 2)} -> "
           f"{'falls beyond 2× noise' if hue_fix else 'no fall beyond 2× noise'}.",
           f"- **bm PSNR cost in a decision scene** (below −2× noise, LOM or varying): {'; '.join(costs) or 'none'}.",
           f"- **Primary reading:** **{verdict}** (fixes: hue falls beyond 2× and no cost; fixes at a cost: hue falls, "
           f"some cost; no fix: otherwise).",
           f"- **Adoption into M6's base:** pooled SSIM/LPIPS worse beyond 2×: {'; '.join(worse['pooled']) or 'none'} "
           f"(without shrub: {'; '.join(worse['pooled_no_shrub']) or 'none'}) -> "
           f"**{'rung2idlum becomes the base' if adopt else 'M6 builds on rung2id; E5 is reported as an ablation'}**.",
           f"- **Overexposure (exploratory):** pooled bm PSNR {fx(ob, 2)} (without shrub {fx(obn, 2)}) -> "
           f"{'**colour accounts for part of the overexposure gap**' if part else 'no rise beyond 2× noise'}; "
           f"rung2id − detach there {fx(gap, 2)}, fraction of that gap closed {frac:+.2f}; pooled saturation ratio "
           f"rung2idlum {sat['rung2idlum']}, rung2id {sat['rung2id']}, detach {sat['detach']}. It cannot change H1-over."]
    return book, dec


# ------------------------------------------------------------------ M6: Rung 3
def load_preview_predictions() -> dict:
    """The preview's numbers.json (plan 4.3's predicted directions), only if its sha256 is the one logged before the
    batch (2026-11-27, "M6 step 3, test 5"); otherwise stop."""
    if not PREVIEW_M6.exists():
        raise SystemExit(f"[rung2_ablation] {PREVIEW_M6} is missing: the pre-registered predictions (plan 4.3) are needed")
    sha = hashlib.sha256(PREVIEW_M6.read_bytes()).hexdigest()
    if sha != PREVIEW_M6_SHA256:
        raise SystemExit(f"[rung2_ablation] {PREVIEW_M6} has sha256 {sha}, not the logged {PREVIEW_M6_SHA256}: the "
                         f"pre-registered predictions changed; stop and investigate")
    return {**json.loads(PREVIEW_M6.read_text()), "sha256": sha}


def mechanism_rows(book, runs, scenes, seeds, arms):
    """Plan 4.1 harm 4 and 4.4 (as mode_h3): per arm (variant), the varying-set leak (the varying run against the same
    variant's LOM run, same seed), the s_k slope on u_k, the paired s RMSE, the s drift on each set; and per seed
    max |s_rung3 - s_rung2id| on each set."""
    gains = json.loads((DATASET / "gains.json").read_text())["rows"]
    u_by = {sc: {r["view"]: r["u"] for r in gains if r["scene"] == sc} for sc in SCENES}
    ueff_by = {sc: {r["view"]: r["log2_ratio"] for r in gains if r["scene"] == sc} for sc in SCENES}
    mech = {}
    for v in arms:
        for sc in scenes:
            for s in seeds:
                rv, rl = runs[f"varying {v}"][(sc, s)], runs[f"LOM {v}"][(sc, s)]
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
    for sc in scenes:
        for s in seeds:
            for tag in ("LOM", "varying"):
                a = np.array(runs[f"{tag} rung3"][(sc, s)]["state"]["s_final"])
                b = np.array(runs[f"{tag} rung2id"][(sc, s)]["state"]["s_final"])
                mech[("rung3", sc, s)][f"max_ds_vs_rung2id_{tag.lower()}"] = float(np.abs(a - b).max())
    mkeys = {"pseudo_leak_paired": 3, "slope": 3, "rmse_paired": 3, "drift_vexp": 4, "drift_lom": 4}
    extra = {"max_ds_vs_rung2id_lom": 4, "max_ds_vs_rung2id_varying": 4}  # rung3 only
    rows = []
    for g in book.G:
        for v in arms:
            keys = dict(mkeys, **(extra if v == "rung3" else {}))
            row = {"scene": g, "arm": v, **({} if v == "rung3" else {k: "–" for k in extra})}
            for k, nd in keys.items():
                x = np.array([np.mean([mech[(v, sc, s)][k] for sc in book.G[g]]) for s in seeds])
                row[k] = stat(x, nd)
                book.cells.append({"scene": g, "arm": v, "metric": k, **{f"seed{s}": float(y) for s, y in zip(seeds, x)}})
                book.numbers.setdefault("mechanism", {}).setdefault(g, {}).setdefault(v, {})[k] = x.tolist()
            rows.append(row)
    return rows


def mode_rung3(a, book_args):
    """M6 (M6_rung3_plan.md section 4): rung3 - rung2id on LOM (results/rung3 vs results/rung2), the varying set (both in
    results/vexp) and over_exp (results/overexp, exploratory), detach alongside as context; the readings of 4.1-4.3 as
    pre-registered; mechanism, zone curves and curve losses reported (4.4); figures."""
    from rung2_headstart import curve_loss as headstart_curve_loss
    scenes, seeds = book_args
    pred = load_preview_predictions()
    sets = [("LOM", "lom", "low", RUNG3, RUNG2, RUNG0), ("varying", "vexp", "low", VEXP, VEXP, VEXP),
            ("over_exp", "lom", "over_exp", OVEREXP, OVEREXP, OVEREXP)]
    arms, ctx = [], []
    for tag, set_, exp, r3, r2, r0 in sets:
        arms += [arm(f"{tag} rung3", "rung3", [r3], set_, exp, ref=("rung2id", [r2])),
                 arm(f"{tag} rung2id", "rung2id", [r2], set_, exp)]
        ctx.append(arm(f"{tag} detach", "detach", [r0], set_, exp))
    if a.selftest:
        arms = [as_selftest(x) for x in arms]
    runs = load_all(arms + ctx, *book_args)
    book = Book(runs, *book_args)
    keys = VAL + EXTRA + ["rel_l1_pseudo"]
    book.cell_rows("Validation and extras per arm (detach: context only)", [x["name"] for x in arms + ctx], keys)
    lab = {tag: f"{tag}: rung3 − rung2id" + (" (exploratory)" if tag == "over_exp" else "") for tag, *_ in sets}
    book.diff_rows("rung3 − rung2id, paired by seed (×noise, * = real above 2×)",
                   [(lab[tag], f"{tag} rung3", f"{tag} rung2id") for tag, *_ in sets], keys)
    book.diff_rows("Against detach, paired by seed (context, reported only)",
                   [(f"{tag}: {v} − detach", f"{tag} {v}", f"{tag} detach") for tag, *_ in sets
                    for v in ("rung3", "rung2id")], keys)
    lom, var, over = lab["LOM"], lab["varying"], lab["over_exp"]

    # harm 3 (D5): consistency on LOM, per-scene relative change of rel_l1_pseudo, mean over the decision scenes
    cons_rows, rel = [], {}
    for sc in scenes:
        c3, c2 = (np.array([value(runs[f"LOM {v}"][(sc, s)], "rel_l1_pseudo") for s in seeds]) for v in ("rung3", "rung2id"))
        rel[sc] = float(c3.mean() / c2.mean() - 1)
        cons_rows.append({"scene": sc, "rung3": stat(c3, 4), "rung2id": stat(c2, 4), "relative change": f"{rel[sc]:+.1%}",
                          "decides": "yes" if sc in H1_SCENES else "no (reported)"})
    dec_sc = [sc for sc in H1_SCENES if sc in rel]
    cons_mean = float(np.mean([rel[sc] for sc in dec_sc]))
    pooled3 = np.mean([np.mean([value(runs["LOM rung3"][(sc, s)], "rel_l1_pseudo") for s in seeds]) for sc in scenes])
    pooled2 = np.mean([np.mean([value(runs["LOM rung2id"][(sc, s)], "rel_l1_pseudo") for s in seeds]) for sc in scenes])
    cons_rows.append({"scene": "decision-scene mean (decides)", "rung3": "", "rung2id": "",
                      "relative change": f"{cons_mean:+.1%}", "decides": f"harm if > +{CONS_HARM:.0%}"})
    cons_rows.append({"scene": "five-scene pooled value (reported)", "rung3": f"{pooled3:.4f}", "rung2id": f"{pooled2:.4f}",
                      "relative change": f"{pooled3 / pooled2 - 1:+.1%}", "decides": "no"})
    book.tables["Harm signal 3 (D5): cross-view pseudo-target inconsistency on LOM (rel_l1_pseudo, mean over "
                "rescore/consistency.csv, shared reference depth; mean ± sd over seeds)"] = cons_rows
    book.numbers["consistency"] = {"relative_change": rel, "decision_mean": cons_mean,
                                   "pooled_relative": float(pooled3 / pooled2 - 1)}

    # mechanism (harm 4 and 4.4)
    book.tables["Mechanism (plan 4.1 harm 4, 4.4): paired pseudo-target leak and s_k slope on u_k (varying set; leak "
                "pairs each varying run with the same arm's LOM run), paired s RMSE, s drift = rmse_c(s_final − s_init) "
                "on each set, and max |s_rung3 − s_rung2id| at the same seed"] = \
        mechanism_rows(book, runs, scenes, seeds, ("rung3", "rung2id"))

    # readings: H4 (4.1)
    help_, harm = [], []
    for lb, setname in ((lom, "LOM"), (var, "varying")):
        for g in ("pooled", "pooled_no_shrub"):
            bm, lp = book.eff(lb, g, "psnr_gtmean"), book.eff(lb, g, "lpips")
            if bm["real"] and bm["diff"] > 0:
                help_.append((g, f"{setname} {g} bm PSNR {fx(bm, 2)}"))
            if lp["real"] and lp["diff"] < 0:
                help_.append((g, f"{setname} {g} LPIPS {fx(lp, 4)}"))
        for sc in H1_SCENES:
            e = book.eff(lb, sc, "psnr_gtmean")
            if e["x_noise"] < -REAL:
                harm.append(f"1: {setname} {sc} bm PSNR {fx(e, 2)} below −2× noise")
        bm, lp = book.eff(lb, "pooled", "psnr_gtmean"), book.eff(lb, "pooled", "lpips")
        if bm["real"] and bm["diff"] < 0:
            harm.append(f"2: {setname} pooled bm PSNR {fx(bm, 2)}")
        if lp["real"] and lp["diff"] > 0:
            harm.append(f"2: {setname} pooled LPIPS {fx(lp, 4)}")
    if cons_mean > CONS_HARM:
        harm.append(f"3: LOM decision-scene mean consistency change {cons_mean:+.1%} > +{CONS_HARM:.0%}")
    leak = float(np.mean(book.numbers["mechanism"]["pooled"]["rung3"]["pseudo_leak_paired"]))
    slope = float(np.mean(book.numbers["mechanism"]["pooled"]["rung3"]["slope"]))
    if leak > LEAK_MAX:
        harm.append(f"4: varying pooled leak {leak:.3f} > {LEAK_MAX}")
    if not (SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1]):
        harm.append(f"4: varying pooled s_k slope {slope:.3f} outside {SLOPE_RANGE[0]} to {SLOPE_RANGE[1]}")
    helps = [t for g, t in help_ if g == "pooled"]
    helps_ns = [t for g, t in help_ if g == "pooled_no_shrub"]
    h4 = "mixed" if helps and harm else "helps" if helps else "hurts" if harm else "no effect"

    # readings: bike's hue (4.2)
    hue_lom, hue_var = book.eff(lom, "bike", "hue_err_sat_deg"), book.eff(var, "bike", "hue_err_sat_deg")
    hue_ok = bool(hue_lom["real"] and hue_lom["diff"] < 0) and not bool(hue_var["real"] and hue_var["diff"] > 0)
    bike_cost = [f"{setname} bike bm PSNR {fx(book.eff(lb, 'bike', 'psnr_gtmean'), 2)}" for lb, setname in
                 ((lom, "LOM"), (var, "varying")) if book.eff(lb, "bike", "psnr_gtmean")["x_noise"] < -REAL]
    hue_verdict = "fixes at a cost" if hue_ok and bike_cost else "fixes" if hue_ok else "no fix"

    # readings: the mechanism check (4.3), against the logged predictions
    checks = []
    for sc, p in pred["predictions"]["saturation"].items():
        if p["prediction"] is None:
            continue
        e = book.eff(lom, sc, "sat_ratio")
        sign = 1 if p["prediction"] == "+" else -1
        state = ("holds" if e["real"] and sign * e["diff"] > 0 else "reversed" if e["real"] and sign * e["diff"] < 0
                 else "not detectable")
        checks.append({"check": f"{sc} saturation", "preview Δ": f"{p['delta']:+.4f}", "predicted": p["prediction"],
                       "LOM val sat ratio, rung3 − rung2id": fx(e, 3), "result": state})
    for sc, p in pred["predictions"]["hue"].items():
        if p["prediction"] is None:
            continue
        e = book.eff(lom, sc, "hue_err_sat_deg")
        sign = -1 if p["prediction"] == "lower" else 1
        state = ("holds" if e["real"] and sign * e["diff"] > 0 else "reversed" if e["real"] and sign * e["diff"] < 0
                 else "not detectable")
        checks.append({"check": f"{sc} hue", "preview Δ": f"{p['delta_deg']:+.2f}°", "predicted": p["prediction"],
                       "LOM val sat ratio, rung3 − rung2id": f"hue err {fx(e, 2)}", "result": state})
    states = [c["result"] for c in checks]
    mech43 = ("no prediction" if not checks else "not as predicted" if "reversed" in states
              else "as predicted" if all(s == "holds" for s in states) else "partly" if "holds" in states
              else "not as predicted")
    book.tables["Mechanism check (plan 4.3): the logged photo-only predictions against LOM validation"] = checks

    # reported only (4.4): zone curves, gamma, shares, curve losses; skipped in --selftest (rung2id has no zones)
    prev = {r["scene"]: r for r in pred["lom"]}
    if not a.selftest:
        zrows = []
        for sc in scenes:
            st3 = [runs["LOM rung3"][(sc, s)]["state"] for s in seeds]
            st2 = [runs["LOM rung2id"][(sc, s)]["state"] for s in seeds]
            Fz = np.mean([np.asarray(x["F_grid_zones"]) for x in st3], axis=0)
            F2 = np.mean([np.asarray(x["F_grid"]) for x in st2], axis=0)
            zrows.append({"scene": sc,
                          "rung3 F_sh (×255)": json.dumps([round(255 * float(Fz[0, b]), 1) for b in Z_LEVELS]),
                          "rung3 F_lit (×255)": json.dumps([round(255 * float(Fz[1, b]), 1) for b in Z_LEVELS]),
                          "rung2id F (×255)": json.dumps([round(255 * float(F2[b]), 1) for b in Z_LEVELS]),
                          "max |F_sh − F_lit|": f"{float(np.abs(Fz[0] - Fz[1]).max()):.3f}",
                          "γ rung3": stat([fnum(x["gamma"]) for x in st3], 4),
                          "γ rung2id": stat([fnum(x["gamma"]) for x in st2], 4),
                          "zone shares (shadow, lit)": json.dumps(np.round(np.mean(st3[0]["zone_shares"], 0), 3).tolist()),
                          "chromatic lit share hard / soft (preview)":
                              f"{prev[sc]['chroma_lit_hard']:.2f} / {prev[sc]['chroma_lit_soft']:.2f}"})
            book.numbers.setdefault("zone_curves", {})[sc] = {"F_zones_mean": Fz.tolist(), "F_rung2id_mean": F2.tolist()}
        book.tables["Zone curves on LOM at grid levels b = 1, 2, 4, 8, 16, 32, 64 (mean over seeds; reported only)"] = zrows
    lrows = []
    for sc in scenes:
        l3 = [headstart_curve_loss(runs["LOM rung3"][(sc, s)]["dir"]) for s in seeds]
        l2 = [headstart_curve_loss(runs["LOM rung2id"][(sc, s)]["dir"]) for s in seeds]
        lrows.append({"scene": sc, "rung3, last 10 logged hist_loss": stat(l3, 5), "rung2id, same": stat(l2, 5),
                      "preview all-view objective rung3": f"{prev[sc]['objective_rung3']:.5f}",
                      "preview all-view objective rung2id": f"{prev[sc]['objective_rung2id']:.5f}"})
    book.tables["Curve loss at 10k (plan 4.4; different objectives, reported side by side)"] = lrows

    figures_rung3(a, book, runs, pred, scenes, seeds, lom)

    book.numbers["m6"] = {"h4": h4, "help": helps, "help_pooled_no_shrub": helps_ns, "harm": harm, "pooled_leak": leak,
                          "pooled_slope": slope, "consistency_decision_mean": cons_mean, "bike_hue": hue_verdict,
                          "bike_hue_lom": hue_lom, "bike_hue_varying": hue_var, "bike_cost": bike_cost,
                          "mechanism_4_3": mech43, "checks_4_3": checks, "preview_sha256": pred["sha256"]}
    ob, obn = book.eff(over, "pooled", "psnr_gtmean"), book.eff(over, "pooled_no_shrub", "psnr_gtmean")
    dec = [f"- **Pre-registered predictions** (`{PREVIEW_M6.relative_to(RESULTS.parent)}`): sha256 {pred['sha256']} = the "
           f"value logged before the batch.",
           f"- **Help signals (pooled):** {'; '.join(helps) or 'none'}. Pooled without shrub: {'; '.join(helps_ns) or 'none'}"
           f"{'' if bool(helps) == bool(helps_ns) else ' -> **differs from pooled**'}.",
           f"- **Harm signals:** {'; '.join(harm) or 'none'} (consistency: decision-scene mean {cons_mean:+.1%}, limit "
           f"+{CONS_HARM:.0%}; varying pooled leak {leak:.3f}, limit {LEAK_MAX}; pooled s_k slope {slope:.3f}, range "
           f"{SLOPE_RANGE[0]} to {SLOPE_RANGE[1]}).",
           f"- **H4 (plan 4.1):** **{h4}**.",
           f"- **Bike's hue (plan 4.2):** LOM {fx(hue_lom, 2)}, varying {fx(hue_var, 2)}; bike bm PSNR cost: "
           f"{'; '.join(bike_cost) or 'none'} -> **{hue_verdict}**.",
           f"- **Mechanism check (plan 4.3):** " + ("; ".join(f"{c['check']}: {c['result']}" for c in checks) or "no check")
           + f" -> **{mech43}**.",
           f"- **Overexposure (exploratory; cannot change H1-over):** pooled bm PSNR {fx(ob, 2)} (without shrub "
           f"{fx(obn, 2)})."]
    return book, dec


def figures_rung3(a, book, runs, pred, scenes, seeds, lom):
    """The plan section 7 figures: zone curves per scene, bike's view 29, validation saturation change against the
    preview's predicted change. The first two need Rung 3 runs, so --selftest skips them."""
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    from rung0_analyze import INK, MUTED, SERIES, savefig
    figs = a.out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    if not a.selftest:
        fig, axes = plt.subplots(2, len(scenes), figsize=(3.4 * len(scenes), 6.6), squeeze=False)
        x = np.linspace(0, 1, 255)
        for j, sc in enumerate(scenes):
            zc = book.numbers["zone_curves"][sc]
            Fz, F2 = np.asarray(zc["F_zones_mean"]), np.asarray(zc["F_rung2id_mean"])
            for i, xmax in enumerate((1.0, 64 / 254)):
                ax, k = axes[i, j], x <= xmax + 1e-9
                ax.plot(x[k], x[k], color=MUTED, lw=1.0, ls="--", label="identity")
                ax.plot(x[k], F2[k], color=INK, lw=2.0, label="rung2id F")
                ax.plot(x[k], Fz[0][k], color=SERIES[0], lw=1.6, label="rung3 F_sh (shadow)")
                ax.plot(x[k], Fz[1][k], color=SERIES[1], lw=1.6, label="rung3 F_lit (lit)")
                ax.set_title(sc + ("" if i == 0 else " (dark end)"))
                ax.set_xlabel("input x")
        h, lab = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
        savefig(fig, "zone_curves_lom", figs)
        if "bike" in scenes:
            d3, d2 = runs["LOM rung3"][("bike", seeds[0])]["dir"] / "rescore", runs["LOM rung2id"][("bike", seeds[0])]["dir"] / "rescore"
            f3, f2, fw = (d3 / f"example_{BIKE_VIEW}_in_pseudo_render_gt.png", d2 / f"example_{BIKE_VIEW}_in_pseudo_render_gt.png",
                          d3 / f"example_{BIKE_VIEW}_wlit.png")
            if f3.exists() and f2.exists() and fw.exists():
                e3, e2, ew = imageio.imread(f3), imageio.imread(f2), imageio.imread(fw)
                p4, p3 = e3.shape[1] // 4, ew.shape[1] // 3
                panels = [ew[:, :p3], ew[:, p3:2 * p3], e2[:, p4:2 * p4], e3[:, p4:2 * p4], e3[:, 3 * p4:]]
                imageio.imwrite(figs / f"bike_{BIKE_VIEW}_input_wlit_rung2id_rung3_gt.png", np.concatenate(panels, axis=1))
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for j, (sc, p) in enumerate(pred["predictions"]["saturation"].items()):
        e = book.eff(lom, sc, "sat_ratio")
        ax.errorbar(p["delta"], e["diff"], yerr=REAL * e["noise"], fmt="o", color=SERIES[j % len(SERIES)], label=sc,
                    capsize=3)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("preview: Δ mean pseudo-target saturation, chromatic pixels")
    ax.set_ylabel("LOM validation: Δ saturation ratio (±2× noise)")
    ax.set_title("rung3 − rung2id: predicted vs measured")
    ax.legend(fontsize=8)
    savefig(fig, "saturation_predicted_vs_measured", figs)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["h3", "noprior", "h1", "init", "lum", "rung3"])
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
    a.out_dir = out  # M6: mode_rung3 writes figures there
    t0 = time.time()
    scenes = [s for s in SCENES if s in a.scenes]
    seeds = sorted(set(a.seeds))
    book, dec = {"h1": mode_h1, "noprior": mode_noprior, "init": mode_init, "h3": mode_h3,
                 "lum": mode_lum, "rung3": mode_rung3}[a.mode](a, (scenes, seeds))

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
               "lum": "E5: the colour-preserving variant, rung2idlum − rung2id on LOM, the varying set and over_exp",
               "rung3": "M6 (H4): Rung 3, rung3 − rung2id on LOM, the varying set and over_exp (exploratory)",
               "h1": f"H1 rule: {a.a} − {a.b}, LOM {a.exp_name} (M5 E3/E6){' — ' + a.title if a.title else ''}"}[a.mode]
    prereg = {"lum": "docs/RESEARCH_LOG.md (2026-11-25, \"E5 plan approved\"), E5_luminance_plan.md section 4",
              "rung3": "docs/RESEARCH_LOG.md (2026-11-26, \"M6 step 1\"; the 4.3 predictions 2026-11-27, \"M6 step 3, "
                       "test 5\"), M6_rung3_plan.md section 4"}.get(a.mode, "docs/RESEARCH_LOG.md (2026-11-24), "
                                                                            "M5_plan.md 3.1–3.6")
    md = [f"# {heading}\n",
          ("**SELFTEST:** the arm under test is read from its reference arm's runs; every effect must be 0. " if a.selftest
           else "") +
          f"Seeds {', '.join(map(str, seeds))}. Paired by seed; noise = sqrt(mean of the arms' seed variances); ×noise; "
          f"* = real (above 2×). Pooled = mean over the five scenes per seed. Readings pre-registered in "
          f"{prereg}. Generated {meta['date']} in {meta['seconds']} s by "
          f"`rung2_ablation.py` ({meta['script_sha256_16']}). Every number is *measured*.\n",
          "## Readings\n", "\n".join(dec), ""]
    for title, rows in book.tables.items():
        md += [f"\n## {title}\n", md_table(rows)]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(dec))
    print(f"[rung2_ablation] wrote {out} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
