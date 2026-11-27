"""M5 curve-only replays (Implementation_guides/M5_plan.md): Rung 2's F and s_k trained on the training photos alone,
with no 3DGS, by rung2_preview.py's machinery (the trainer's data loading, model, optimisers, schedule and gauge).

    python rung2_replay.py --mode overexp   E6's pre-launch check (M5_plan.md step 2): the default configuration on
                                            the LOM over_exp photos, 10k steps, curve weight as --exp_name over_exp
                                            (0.1 from step 3000) -> results/overexp/preview/
    python rung2_replay.py --mode long      A1 (M5_plan.md 3.5): cdf and identity inits on the LOM low-light photos,
                                            50k steps, F gap and curve loss every 1k steps, plus 10k replays checked
                                            against the trainer's gap -> results/rung2/replay_long/
    python rung2_replay.py --mode rung3     M6's preview (M6_rung3_plan.md test 5): rung3's and rung2id's curve training
                                            replayed 10k steps per scene on the LOM low-light photos (and, for the
                                            exposure check, on the varying set), same seeded view order; the curves,
                                            gamma, zone shares, s against rung2id, section 1.4's photo statistics, the
                                            zone brightness of both models' pseudo-targets and the predicted directions
                                            of plan 4.3 -> results/rung3/preview/ (finished once logged: protected.py)

Training photos only: no normal-light image is read. CPU only (--threads, default 2, leaves room for a training
batch). Readings fixed in the research log on 2026-11-24 before any run:
- overexp, stop rule: leave E6 out of the launch and ask Bart if, in any scene, gamma <= 1 or F(x) < x at fewer than
  80% of the interior grid points (b = 1..253), or if the pooled fit residual is above 2x the low-light preview's
  (results/rung2/preview_final/P1_fit.csv, same configuration). Reported only: s_k movement from the init, |mean s|,
  the share of pixels at level 255 and the mean CDF[253] (the shifted target's top value; F(1) = 1 is pinned).
- long: gap(t) = max over the 255-bin grid of |F_identity(t) - F_cdf(t)|, same step. Per scene: "early stopping" if
  the final gap < 0.01; "two end points" if not and the gap fell by less than 10% over the last fifth of the run
  (40k to 50k); "undecided" otherwise. The learning rate decays to 1% over the replay, so a 50k replay models a
  50k-step run. Built-in check: the 10k replays' final gap is within 0.01 of the trainer's
  (results/rung2/summary_headstart_seeds012/curves.csv, seeds 0-2).
Writes only to its mode's folder or to a git-ignored smoke* folder under results/ (tests), never to a folder that
CLAUDE.md rule 6 protects (protected.py). Both modes' folders are finished since 2026-11-26, so --mode overexp and
--mode long now refuse unless --out names a smoke* folder.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))

from protected import refuse_finished  # noqa: E402
from rung0_analyze import INK, MUTED, SCENES, SERIES, md_table, savefig, write_csv  # noqa: E402
from rung2_preview import (DEFAULT, F_CLOSE, IDENTITY, LATE_STEP, build, load_set, objective, replay,  # noqa: E402
                           weighted_resid255)
from vexp_analyze import rmse_c  # noqa: E402

RESULTS = EXAMPLES.parent / "results"
OUT = {"overexp": RESULTS / "overexp" / "preview", "long": RESULTS / "rung2" / "replay_long",
       "rung3": RESULTS / "rung3" / "preview"}
LOW_PREVIEW = RESULTS / "rung2" / "preview_final" / "P1_fit.csv"
HEADSTART = RESULTS / "rung2" / "summary_headstart_seeds012" / "curves.csv"
LR_F, LR_S, K, LAM = 1e-2, 1e-3, 32, 0.1  # the trainer's Rung 2 defaults (simple_trainer_ours.Config)
GAMMA_MIN, BELOW_FRAC, RESID_RATIO = 1.0, 0.80, 2.0  # overexp stop rule
SHRINK, CHECK_TOL = 0.10, 0.01  # long: "stopped shrinking" over the last fifth; built-in check tolerance
INTERIOR = slice(1, 254)  # grid points b = 1..253 (F(0) = 0 and F(1) = 1 are pinned)


def guard(out: Path, mode: str) -> Path:
    out = refuse_finished(out, "rung2_replay")  # CLAUDE.md rule 6 (protected.py)
    res = RESULTS.resolve()
    test = res in out.parents and any(part.startswith("smoke") for part in out.relative_to(res).parts)
    if not (out == OUT[mode].resolve() or test):
        raise SystemExit(f"[rung2_replay] refusing to write to {out}: use {OUT[mode]} or a smoke* folder under results/")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


def settings(steps: int, every: int, seed: int) -> SimpleNamespace:
    """The attributes rung2_preview.replay reads."""
    return SimpleNamespace(steps=steps, every=every, seed=seed, lr_s=LR_S, smooth_lambda=LAM)


def meta(a, t0: float) -> dict:
    sha = {f"{f}_sha256_16": hashlib.sha256((EXAMPLES / f).read_bytes()).hexdigest()[:16]
           for f in ("rung2_replay.py", "rung2_preview.py", "rung2.py", "rung3.py")}
    return {"date": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0), "torch": torch.__version__,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}, **sha}


# ------------------------------------------------------------------ E6 pre-launch check
def run_overexp(a, out: Path, t0: float):
    steps = a.steps or 10_000
    every = a.every or 100
    low = {r["scene"]: float(r["resid255"]) for r in csv.DictReader(open(LOW_PREVIEW, newline=""))
           if r["config"] == DEFAULT and r["set"] == "lom"}
    rows, curves_fig = [], {}
    for sc in a.scenes:
        tc = time.time()
        d = load_set(sc, "lom", "over_exp")
        model, targets, _ = build(d, DEFAULT, K, {})
        traj = replay(model, targets, settings(steps, every, a.seed), LR_F, "over_exp")
        with torch.no_grad():
            x = model.grid.numpy()
            Fg = model.F(model.grid).numpy()
            curves = model.composed_curve(torch.arange(model.N)).numpy()
            prior = model.prior_curve()[0].numpy()
        s, s0 = model.s.detach().double().numpy(), d["s0"].double().numpy()
        dF = (Fg - x)[INTERIOR]
        rows.append({"scene": sc, "views": model.N, "mean_input": float(d["mean_input"].mean()),
                     "gamma": float(model.gamma()), "F_below_identity_frac": float(np.mean(dF < 0)),
                     "mean_F_minus_x_x255": 255 * float(dF.mean()), "min_F_minus_x_x255": 255 * float(dF.min()),
                     "max_F_minus_x_x255": 255 * float(dF.max()),
                     "F_at_64_127_191_x255": [round(255 * float(Fg[b]), 1) for b in (64, 127, 191)],
                     "resid255": weighted_resid255(curves, targets.numpy(), d["hist"]),
                     "low_light_resid255": low.get(sc, float("nan")),
                     "s_move_rmse": rmse_c(s - s0), "abs_mean_s": abs(float(s.mean())), "s_init_spread": rmse_c(s0),
                     "share_level_255": float(d["counts"][255] / d["counts"].sum()),
                     "mean_cdf253": float(d["cdf"][:, 253].mean()), "min_cdf253": float(d["cdf"][:, 253].min()),
                     "objective_final": traj[-1][1], "seconds": round(time.time() - tc, 1)})
        curves_fig[sc] = (x, Fg, prior, targets.numpy().mean(0), float(model.gamma()))
        print(f"[rung2_replay] {sc}: gamma {rows[-1]['gamma']:.3f}, F below identity at "
              f"{100 * rows[-1]['F_below_identity_frac']:.0f}% ({rows[-1]['seconds']:.0f} s)", flush=True)

    pooled = float(np.mean([r["resid255"] for r in rows]))
    pooled_low = float(np.mean([r["low_light_resid255"] for r in rows]))
    reasons = [f"{r['scene']}: gamma {r['gamma']:.3f} <= {GAMMA_MIN:g}" for r in rows if r["gamma"] <= GAMMA_MIN]
    reasons += [f"{r['scene']}: F below the identity at {100 * r['F_below_identity_frac']:.0f}% of interior points "
                f"(< {BELOW_FRAC:.0%})" for r in rows if r["F_below_identity_frac"] < BELOW_FRAC]
    if pooled > RESID_RATIO * pooled_low:
        reasons.append(f"pooled residual {pooled:.2f} > {RESID_RATIO:g} x low light {pooled_low:.2f}")
    verdict = "STOP: leave E6 out of the launch and ask Bart" if reasons else "pass: E6 can join the launch"

    write_csv("overexp_check", [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows], out)
    m = meta(a, t0)
    (out / "numbers.json").write_text(json.dumps({"meta": m, "rows": rows, "pooled_resid255": pooled,
                                                 "pooled_low_light_resid255": pooled_low, "stop_reasons": reasons,
                                                 "verdict": verdict}, indent=1))
    fig, axes = plt.subplots(1, len(curves_fig), figsize=(3.4 * len(curves_fig), 3.6), sharey=True, squeeze=False)
    for ax, (sc, (x, Fg, prior, T, g)) in zip(axes[0], curves_fig.items()):
        ax.plot(x, x, color=MUTED, lw=1.0, ls="--", label="identity")
        ax.plot(x, T, color=SERIES[0], lw=1.2, label="mean target (shifted HE-CDF)")
        ax.plot(x, prior, color=SERIES[3], lw=1.4, ls=":", label="shape prior x^gamma")
        ax.plot(x, Fg, color=INK, lw=2.0, label="shared F")
        ax.set_title(f"{sc} (gamma {g:.2f})")
        ax.set_xlabel("input value x (0-1)")
    axes[0, 0].set_ylabel("output")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    savefig(fig, "overexp_curves", out / "figures")
    show = ["scene", "views", "mean_input", "gamma", "F_below_identity_frac", "mean_F_minus_x_x255",
            "F_at_64_127_191_x255", "resid255", "low_light_resid255", "s_move_rmse", "abs_mean_s",
            "share_level_255", "mean_cdf253", "min_cdf253"]
    md = ["# E6 pre-launch check: Rung 2's curve on the LOM over_exp photos (curve loss only, no 3DGS)\n",
          f"Default configuration (`{DEFAULT}`), {steps} steps, curve weight as `--exp_name over_exp`, seed {a.seed}. "
          f"Training photos only. Generated {m['date']} in {m['seconds']} s by `rung2_replay.py` "
          f"({m['rung2_replay.py_sha256_16']}). Every number is *measured* on the replay.\n",
          f"**Stop rule** (log, 2026-11-24): gamma <= {GAMMA_MIN:g} or F below the identity at fewer than "
          f"{BELOW_FRAC:.0%} of interior grid points in any scene, or pooled residual above {RESID_RATIO:g}x the "
          f"low-light preview's.\n",
          f"- Pooled residual x255: {pooled:.2f} (low light, `preview_final`: {pooled_low:.2f}; ratio "
          f"{pooled / pooled_low:.2f}).",
          f"- **{verdict}**" + (": " + "; ".join(reasons) if reasons else "."), "",
          md_table([{k: r[k] for k in show} for r in rows]),
          "\nFigure: `figures/overexp_curves.png`.\n"]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[rung2_replay] {verdict}{': ' + '; '.join(reasons) if reasons else ''}")
    print(f"[rung2_replay] wrote {out} in {time.time() - t0:.0f} s")


# ------------------------------------------------------------------ A1
def trainer_gaps() -> dict:
    """The trainer's final F gap per scene and seed (rung2_headstart.py's curves.csv, x255 -> 0-1)."""
    out = {}
    with open(HEADSTART, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    for r in rows[1:]:
        out.setdefault(r[0], []).append(float(r[2].split()[0]) / 255)
    return out


def run_long(a, out: Path, t0: float):
    steps = a.steps or 50_000
    every = a.every or 1000
    tgap = trainer_gaps()
    traj_rows, rows, fig_data = [], [], {}
    for sc in a.scenes:
        tc = time.time()
        d = load_set(sc, "lom")
        tr, chk = {}, {}
        for cfg in (DEFAULT, IDENTITY):
            model, targets, _ = build(d, cfg, K, {})
            tr[cfg] = replay(model, targets, settings(steps, every, a.seed), LR_F)
            model, targets, _ = build(d, cfg, K, {})
            chk[cfg] = replay(model, targets, settings(a.check_steps, every, a.seed), LR_F)
        gap = {}
        for (t, jc, Fc), (t2, ji, Fi) in zip(tr[DEFAULT], tr[IDENTITY]):
            assert t == t2
            gap[t] = float(np.abs(Fi - Fc).max())
            traj_rows.append({"scene": sc, "step": t, "gap": gap[t], "objective_cdf": jc, "objective_identity": ji})
        last = max(gap)
        ref = min((t for t in gap if t >= 0.8 * last), default=last)  # 40k for a 50k run
        g_last, g_ref = gap[last], gap[ref]
        if g_last < F_CLOSE:
            outcome = "early stopping"
        elif g_ref - g_last < SHRINK * g_ref:
            outcome = "two end points"
        else:
            outcome = "undecided"
        first = min((t for t in gap if t >= every), default=None)
        halve = next((t for t in sorted(gap) if first is not None and t >= first and gap[t] <= 0.5 * gap[first]), None)
        g_chk = float(np.abs(chk[IDENTITY][-1][2] - chk[DEFAULT][-1][2]).max())
        tg = tgap.get(sc, [])
        check_ok = bool(tg) and abs(g_chk - float(np.mean(tg))) <= CHECK_TOL if a.check_steps == 10_000 else None
        rows.append({"scene": sc, "steps": last, "gap_first": gap[first] if first is not None else float("nan"),
                     "gap_at_ref": g_ref, "ref_step": ref, "gap_final": g_last, "outcome": outcome,
                     "step_gap_halves": halve, "objective_cdf_final": tr[DEFAULT][-1][1],
                     "objective_identity_final": tr[IDENTITY][-1][1], f"check_gap_{a.check_steps}": g_chk,
                     "trainer_gap_10k_seeds": [round(v, 4) for v in tg], "check_ok": check_ok,
                     "seconds": round(time.time() - tc, 1)})
        fig_data[sc] = (sorted(gap), [gap[t] for t in sorted(gap)])
        print(f"[rung2_replay] {sc}: gap {gap[first] if first is not None else float('nan'):.4f} at {first} -> "
              f"{g_last:.4f} at {last}: {outcome}; check {g_chk:.4f} vs trainer {tg} ({rows[-1]['seconds']:.0f} s)",
              flush=True)

    write_csv("gap_by_step", traj_rows, out)
    write_csv("long_summary", [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows], out)
    m = meta(a, t0)
    (out / "numbers.json").write_text(json.dumps({"meta": m, "rows": rows}, indent=1))
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for (sc, (t, g)), col in zip(fig_data.items(), SERIES):
        ax.plot(t, g, color=col, lw=1.6, label=sc)
    ax.axhline(F_CLOSE, color=MUTED, lw=1.0, ls="--", label=f"{F_CLOSE} (early-stopping line)")
    ax.set_yscale("log")
    ax.set_xlabel("replay step")
    ax.set_ylabel("max |F_identity - F_cdf| (0-1 scale)")
    ax.legend(fontsize=8)
    savefig(fig, "gap_by_step", out / "figures")
    md = ["# A1: does the identity-init F converge to the cdf-init F with longer curve training? (curve loss only)\n",
          f"LOM low-light training photos, both inits, {steps} steps (learning rate decayed to 1% over the replay, so "
          f"this models a {steps}-step run), seed {a.seed}. Generated {m['date']} in {m['seconds']} s by "
          f"`rung2_replay.py` ({m['rung2_replay.py_sha256_16']}). Every number is *measured* on the replay.\n",
          f"Reading (log, 2026-11-24): early stopping if the final gap < {F_CLOSE}; two end points if not and the gap "
          f"fell by less than {SHRINK:.0%} over the last fifth; undecided otherwise. Built-in check: the "
          f"{a.check_steps}-step replay's final gap within {CHECK_TOL} of the trainer's (seeds 0-2).\n",
          md_table(rows), "\nFigure: `figures/gap_by_step.png`; per-step values: `gap_by_step.csv`.\n"]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[rung2_replay] wrote {out} in {time.time() - t0:.0f} s")


# ------------------------------------------------------------------ M6: the Rung 3 preview (plan test 5)
Z_LEVELS = [1, 2, 4, 8, 16, 32, 64]  # grid indices b (x = b / 254) at which the plan reports the curves
BLUR = 16                            # --rung3_blur's default; the variant keeps it
DECISION = ["buu", "chair", "sofa", "bike"]
CHROMA_SAT, CHROMA_LEVEL = 0.3, 4    # plan 1.4: photo saturation (max - min) / max > 0.3, brightest channel >= level 4
SAT_MIN, HUE_MIN = 0.02, 1.0         # plan 4.3: smallest preview change that makes a prediction
SPLIT_MIN, S_RMS_MAX = 0.01, 0.05    # plan test 5's stop rule
BIKE_VIEW = "29"                     # diagnostics.RESCORE_EXAMPLES


def zone_model(d: dict):
    """Rung 3 as the trainer builds it (Runner.build_rung3): build_zones on the photos at s_init, identity curves."""
    from rung2 import init_heights, make_knots
    from rung3 import ZoneExposureCurve, build_zones
    knots = make_knots(K, "quadratic")
    return ZoneExposureCurve(knots, init_heights("identity", knots), d["s0"], build_zones(d["images"], d["s0"], BLUR),
                             blur=BLUR)


@torch.no_grad()
def zone_objective(model, lam: float, exp_name: str = "low") -> float:
    """The zone curve loss over all views at the late weight (rung2_preview.objective's counterpart)."""
    views = torch.arange(model.N)
    return float(model.loss(model.composed_curves(views), model.zone_targets, model.zone_shares, LATE_STEP, exp_name,
                            lam)[0])


def replay_zones(model, a, lr_F: float, exp_name: str = "low"):
    """rung2_preview.replay for Rung 3: one view per step, 10 x the zone curve loss on that view's fixed zone targets,
    Adam on (theta, rho) at lr_F and on s at a.lr_s, ExponentialLR to 1%, the gauge projection; the same seeded view
    order as rung2_preview.replay. Returns checkpoints (step, objective, the curves on the grid)."""
    opts = [torch.optim.Adam(model.f_params(), lr=lr_F, weight_decay=0.0),
            torch.optim.Adam([model.s], lr=a.lr_s, weight_decay=0.0)]
    scheds = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=0.01 ** (1.0 / a.steps)) for o in opts]
    gen = torch.Generator().manual_seed(a.seed)
    order, traj = [], []
    for step in range(a.steps):
        if step % a.every == 0:
            traj.append((step, zone_objective(model, a.smooth_lambda, exp_name), model.F_zones(model.grid).detach().numpy()))
        if not order:
            order = torch.randperm(model.N, generator=gen).tolist()
        k = order.pop(0)
        loss, _ = model.loss(model.composed_curves(k), model.zone_targets[k:k + 1], model.zone_shares[k:k + 1], step,
                             exp_name, a.smooth_lambda)
        for o in opts:
            o.zero_grad(set_to_none=True)
        (10 * loss).backward()
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        model.project_gauge()
    traj.append((a.steps, zone_objective(model, a.smooth_lambda, exp_name), model.F_zones(model.grid).detach().numpy()))
    return traj


@torch.no_grad()
def pseudo_and_photo_stats(d: dict, r2, z) -> dict:
    """Photo-only statistics (no normal-light image): section 1.4's chromatic shares, and for the photo and both
    models' pseudo-targets, the mean HSV saturation and hue drift from the photo on chromatic pixels (pooled over the
    views' pixels) and the zone brightness (w-weighted mean grey per view; 'inverted' = shadow brighter than lit)."""
    from diagnostics import HUE_PRED_SAT_MIN, rgb_to_hsv
    names = ("photo", "rung2id", "rung3")
    acc = {n: {"sat": 0.0, "hue": 0.0, "n_hue": 0, "sh": [], "lit": [], "inv": 0} for n in names}
    n_chroma = n_px = hard = 0
    soft = ramp = 0.0
    levels = []
    for k, img in enumerate(d["images"]):
        x = img[0]
        mx, mn = x.max(-1).values, x.min(-1).values
        chroma = ((mx - mn) / mx.clamp_min(1e-8) > CHROMA_SAT) & (mx * 255 >= CHROMA_LEVEL - 0.5)
        ws, wl = z.weights[k, 0], z.weights[k, 1]
        n_chroma += int(chroma.sum())
        n_px += chroma.numel()
        hard += int((wl[chroma] > 0.5).sum())
        soft += float(wl[chroma].double().sum())
        ramp += int(((wl[chroma] > 0) & (wl[chroma] < 1)).sum())
        levels.append((mx[chroma] * 255).numpy())
        hp, _, _ = rgb_to_hsv(x)
        for n, P in (("photo", x), ("rung2id", r2.map_pixels(img, k)[0]), ("rung3", z.map_pixels(img, k)[0])):
            h, s, _ = rgb_to_hsv(P)
            a = acc[n]
            a["sat"] += float(s[chroma].double().sum())
            ok = chroma & (s > HUE_PRED_SAT_MIN)
            dh = torch.remainder((h - hp)[ok].abs(), 360.0)
            a["hue"] += float(torch.minimum(dh, 360.0 - dh).double().sum())
            a["n_hue"] += int(ok.sum())
            Y = P.mean(-1)
            m_sh, m_lit = float((ws * Y).sum() / ws.sum()), float((wl * Y).sum() / wl.sum())
            a["sh"].append(m_sh)
            a["lit"].append(m_lit)
            a["inv"] += int(m_sh > m_lit)
    n_views = len(d["images"])
    out = {"chroma_px_frac": n_chroma / n_px, "chroma_lit_hard": hard / n_chroma, "chroma_lit_soft": soft / n_chroma,
           "chroma_in_ramp": ramp / n_chroma, "chroma_maxlevel_median": float(np.median(np.concatenate(levels)))}
    for n, a in acc.items():
        sh, lit = float(np.mean(a["sh"])), float(np.mean(a["lit"]))
        out[n] = {"sat_chroma": a["sat"] / n_chroma, "hue_drift_deg": a["hue"] / max(a["n_hue"], 1),
                  "hue_px_frac": a["n_hue"] / n_chroma, "zone_sh_x255": 255 * sh, "zone_lit_x255": 255 * lit,
                  "lit_over_sh": lit / sh, "inverted_views": a["inv"], "inverted_share": a["inv"] / n_views}
    return out


def preview_scene(sc: str, which: str, a, photo_stats: bool) -> tuple:
    """Both replays on one scene and set, and the per-scene numbers (plus the figure data)."""
    from rung3 import guide, lit_weight
    tc = time.time()
    d = load_set(sc, which)
    r2, targets, _ = build(d, IDENTITY, K, {})
    tr2 = replay(r2, targets, settings(a.steps or 10_000, a.every or 1000, a.seed), LR_F)
    z = zone_model(d)
    tr3 = replay_zones(z, settings(a.steps or 10_000, a.every or 1000, a.seed), LR_F)
    with torch.no_grad():
        Fz, F2 = z.F_zones(z.grid).numpy(), r2.F(r2.grid).numpy()
        s0, s3, s2 = d["s0"].double().numpy(), z.s.detach().double().numpy(), r2.s.detach().double().numpy()
        c1, c2 = (float(c) for c in z.centres)
        dw = max(float((lit_weight(guide(img, float(z.s[k]), BLUR), c1, c2) - z.weights[k, 1]).abs().max())
                 for k, img in enumerate(d["images"]))
    row = {"scene": sc, "set": which, "views": z.N,
           "centres_x255": [round(255 * 2 ** c1, 2), round(255 * 2 ** c2, 2)], "c2_minus_c1": c2 - c1,
           "lit_share_px": float(z.weights[:, 1].mean()), "zone_shares_mean": z.zone_shares.mean(0).tolist(),
           "cdf_gap_mean": float((z.zone_targets[:, 0] - z.zone_targets[:, 1]).abs().amax(dim=1).mean()),
           "gamma_rung3": float(z.gamma()), "gamma_rung2id": float(r2.gamma()),
           "F_sh_x255": [round(255 * float(Fz[0, b]), 1) for b in Z_LEVELS],
           "F_lit_x255": [round(255 * float(Fz[1, b]), 1) for b in Z_LEVELS],
           "F_rung2id_x255": [round(255 * float(F2[b]), 1) for b in Z_LEVELS],
           "max_F_sh_minus_lit": float(np.abs(Fz[0] - Fz[1]).max()),
           "increasing_rung3": z.strictly_increasing(), "increasing_rung2id": r2.strictly_increasing(),
           "s_move_rung3": rmse_c(s3 - s0), "s_move_rung2id": rmse_c(s2 - s0), "s_rms_rung3_vs_rung2id": rmse_c(s3 - s2),
           "max_abs_dw_lit_final_vs_init": dw,
           "objective_rung3": tr3[-1][1], "objective_rung2id": tr2[-1][1]}
    fig = {"x": z.grid.numpy(), "Fz": Fz, "F2": F2, "gamma": float(z.gamma())}
    if photo_stats:
        st = pseudo_and_photo_stats(d, r2, z)
        row.update({k: v for k, v in st.items() if not isinstance(v, dict)})
        for n in ("photo", "rung2id", "rung3"):
            row.update({f"{n}_{k}": v for k, v in st[n].items()})
        row["d_sat_chroma"] = st["rung3"]["sat_chroma"] - st["rung2id"]["sat_chroma"]
        row["d_hue_drift_deg"] = st["rung3"]["hue_drift_deg"] - st["rung2id"]["hue_drift_deg"]
        T2 = targets.numpy()
        row["T8_sh_lit_all_x255"] = [round(255 * float(z.zone_targets[:, m, 8].mean()), 1) for m in (0, 1)] + \
            [round(255 * float(T2[:, 8].mean()), 1)]
        row["T4_sh_lit_all_x255"] = [round(255 * float(z.zone_targets[:, m, 4].mean()), 1) for m in (0, 1)] + \
            [round(255 * float(T2[:, 4].mean()), 1)]
        if sc == "bike":
            fig["example"] = bike_example(d, r2, z)
    row["seconds"] = round(time.time() - tc, 1)
    print(f"[rung2_replay] {sc}/{which}: max |F_sh - F_lit| {row['max_F_sh_minus_lit']:.3f}, gamma {row['gamma_rung3']:.3f} "
          f"(rung2id {row['gamma_rung2id']:.3f}), s RMS vs rung2id {row['s_rms_rung3_vs_rung2id']:.4f}, max |dw| {dw:.4f}"
          + (f", d sat {row['d_sat_chroma']:+.3f}, d hue drift {row['d_hue_drift_deg']:+.2f} deg" if photo_stats else "")
          + f" ({row['seconds']:.0f} s)", flush=True)
    return row, fig


@torch.no_grad()
def bike_example(d: dict, r2, z):
    """Bike's view 29 (photo-only): photo x 4, w_lit, rung2id's and rung3's pseudo-targets, side by side."""
    from diagnostics import to_uint8
    k = next(i for i, n in enumerate(d["names"]) if Path(n).stem == BIKE_VIEW)
    img = d["images"][k]
    return np.concatenate([to_uint8(torch.clamp(4 * img[0], 0, 1)), to_uint8(z.weights[k, 1]),
                           to_uint8(r2.map_pixels(img, k)[0]), to_uint8(z.map_pixels(img, k)[0])], axis=1)


def run_rung3(a, out: Path, t0: float):
    import imageio.v2 as imageio
    lom, vexp, figs = [], [], {}
    for sc in a.scenes:
        row, fig = preview_scene(sc, "lom", a, photo_stats=True)
        lom.append(row)
        figs[sc] = fig
    for sc in a.scenes:
        vexp.append(preview_scene(sc, "vexp", a, photo_stats=False)[0])

    # stop rule (plan test 5; photo-only): a bug, a collapse to Rung 2, or a disturbed exposure model
    reasons = [f"{r['set']} {r['scene']}: a curve is not strictly increasing" for r in lom + vexp
               if not (r["increasing_rung3"] and r["increasing_rung2id"])]
    dec = [r for r in lom if r["scene"] in DECISION]
    if dec and all(r["max_F_sh_minus_lit"] < SPLIT_MIN for r in dec):
        reasons.append(f"every decision scene has max |F_sh - F_lit| < {SPLIT_MIN} on LOM (Rung 3 collapses to Rung 2)")
    reasons += [f"{r['set']} {r['scene']}: s differ from the rung2id replay by {r['s_rms_rung3_vs_rung2id']:.4f} stops RMS "
                f"(> {S_RMS_MAX})" for r in lom + vexp if r["s_rms_rung3_vs_rung2id"] > S_RMS_MAX]
    verdict = "STOP: ask Bart before the batch" if reasons else "pass: the batch can be proposed"

    # predicted directions for plan 4.3 (photo-only; LOM)
    sat_pred = {}
    for r in dec:
        dlt = float(r["d_sat_chroma"])
        sat_pred[r["scene"]] = {"delta": dlt, "prediction": ("+" if dlt > 0 else "-") if abs(dlt) >= SAT_MIN else None}
    bike = next((r for r in lom if r["scene"] == "bike"), None)
    hue_pred = {}
    if bike is not None:
        dh = float(bike["d_hue_drift_deg"])
        hue_pred["bike"] = {"delta_deg": dh, "prediction": ("lower" if dh < 0 else "higher") if abs(dh) >= HUE_MIN else None}
    predictions = {
        "rule": "plan 4.3: saturation, a decision scene's preview change in mean pseudo-target saturation on chromatic "
                f"pixels (rung3 - rung2id) of at least {SAT_MIN} predicts the sign of its LOM validation saturation-ratio "
                f"change; hue, a bike preview hue-drift change of at least {HUE_MIN} deg predicts the direction of its "
                "LOM validation hue error (smaller drift -> lower error)",
        "saturation": sat_pred, "hue": hue_pred}

    write_csv("preview_lom", [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
                              for r in lom], out)
    write_csv("preview_vexp", [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}
                               for r in vexp], out)
    m = meta(a, t0)
    (out / "numbers.json").write_text(json.dumps({"meta": m, "lom": lom, "vexp": vexp, "stop_reasons": reasons,
                                                  "verdict": verdict, "predictions": predictions}, indent=1))

    # figures: the curves per scene (full range and the dark end), bike's view 29
    fig, axes = plt.subplots(2, len(figs), figsize=(3.4 * len(figs), 6.6), squeeze=False)
    for j, (sc, f) in enumerate(figs.items()):
        for i, xmax in enumerate((1.0, 64 / 254)):
            ax = axes[i, j]
            keep = f["x"] <= xmax + 1e-9
            x = f["x"][keep]
            ax.plot(x, x, color=MUTED, lw=1.0, ls="--", label="identity")
            ax.plot(x, np.clip(x, 1e-3, 1) ** f["gamma"], color=SERIES[3], lw=1.2, ls=":", label="prior x^gamma (rung3)")
            ax.plot(x, f["F2"][keep], color=INK, lw=2.0, label="rung2id F")
            ax.plot(x, f["Fz"][0][keep], color=SERIES[0], lw=1.6, label="rung3 F_sh (shadow)")
            ax.plot(x, f["Fz"][1][keep], color=SERIES[1], lw=1.6, label="rung3 F_lit (lit)")
            ax.set_title(f"{sc}" + ("" if i == 0 else " (dark end)"))
            ax.set_xlabel("input x")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.0))
    savefig(fig, "zone_curves", out / "figures")
    if "example" in figs.get("bike", {}):
        imageio.imwrite(out / "figures" / f"bike_{BIKE_VIEW}_photo4_wlit_rung2id_rung3.png", figs["bike"]["example"])

    def pick(rows, keys):
        return [{k: (json.dumps(r[k]) if isinstance(r[k], list) else r[k]) for k in keys} for r in rows]

    md = ["# M6 preview: Rung 3's and rung2id's curve training replayed on the training photos (no 3DGS)\n",
          f"10k steps per scene (learning rate decayed to 1%), identity init, seed {a.seed}, the same view order for both "
          f"models; Rung 3 with fixed zone weights at s_init and a {BLUR}-px guide, as the trainer builds it. Training "
          f"photos only: no normal-light image is read. Generated {m['date']} in {m['seconds']} s by `rung2_replay.py` "
          f"({m['rung2_replay.py_sha256_16']}; rung3.py {m['rung3.py_sha256_16']}). Every number is *measured* on the "
          f"replay. Plan: `M6_rung3_plan.md` test 5 and section 4.3.\n",
          f"**Stop rule** (plan test 5): a curve not strictly increasing; every decision scene with max |F_sh − F_lit| < "
          f"{SPLIT_MIN}; s more than {S_RMS_MAX} stops RMS from the rung2id replay's. **{verdict}**"
          + (": " + "; ".join(reasons) if reasons else "."), "",
          "## Predicted directions (plan 4.3; logged before the batch)\n",
          md_table([{"scene": sc, "Δ sat (rung3 − rung2id)": p["delta"],
                     "prediction (LOM val sat ratio)": p["prediction"] or "no prediction"} for sc, p in sat_pred.items()],
                   nd=4),
          md_table([{"scene": sc, "Δ hue drift (°)": p["delta_deg"],
                     "prediction (LOM val hue error)": p["prediction"] or "no prediction"} for sc, p in hue_pred.items()],
                   nd=3),
          "\n## Curves on the 255-bin grid at b = 1, 2, 4, 8, 16, 32, 64 (×255), γ, split, exposures (LOM)\n",
          md_table(pick(lom, ["scene", "F_sh_x255", "F_lit_x255", "F_rung2id_x255", "max_F_sh_minus_lit", "gamma_rung3",
                              "gamma_rung2id", "s_rms_rung3_vs_rung2id", "max_abs_dw_lit_final_vs_init",
                              "objective_rung3", "objective_rung2id"]), nd=4),
          "\n## Zone brightness of the pseudo-targets (w-weighted mean grey ×255; inverted = shadow zone brighter)\n",
          md_table(pick(lom, ["scene", "photo_zone_sh_x255", "photo_zone_lit_x255", "rung2id_zone_sh_x255",
                              "rung2id_zone_lit_x255", "rung2id_lit_over_sh", "rung3_zone_sh_x255", "rung3_zone_lit_x255",
                              "rung3_lit_over_sh", "rung2id_inverted_share", "rung3_inverted_share"]), nd=3),
          "\n## Colour on chromatic photo pixels (mean HSV saturation; hue drift from the photo, °)\n",
          md_table(pick(lom, ["scene", "photo_sat_chroma", "rung2id_sat_chroma", "rung3_sat_chroma", "d_sat_chroma",
                              "rung2id_hue_drift_deg", "rung3_hue_drift_deg", "d_hue_drift_deg", "rung3_hue_px_frac"]),
                   nd=4),
          "\n## Section 1.4 recomputed with the trainer's loader\n",
          md_table(pick(lom, ["scene", "views", "centres_x255", "c2_minus_c1", "lit_share_px", "zone_shares_mean",
                              "cdf_gap_mean", "chroma_px_frac", "chroma_lit_hard", "chroma_lit_soft", "chroma_in_ramp",
                              "chroma_maxlevel_median", "T8_sh_lit_all_x255", "T4_sh_lit_all_x255"]), nd=3),
          "\n## Varying set (exposure check only)\n",
          md_table(pick(vexp, ["scene", "centres_x255", "zone_shares_mean", "F_sh_x255", "F_lit_x255", "F_rung2id_x255",
                               "gamma_rung3", "s_move_rung3", "s_move_rung2id", "s_rms_rung3_vs_rung2id",
                               "max_abs_dw_lit_final_vs_init", "increasing_rung3"]), nd=4),
          "\nFigures: `figures/zone_curves.png`; `figures/bike_29_photo4_wlit_rung2id_rung3.png` (photo ×4 | w_lit | "
          "rung2id pseudo-target | rung3 pseudo-target).\n"]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    sha = hashlib.sha256((out / "numbers.json").read_bytes()).hexdigest()
    print(f"[rung2_replay] {verdict}{': ' + '; '.join(reasons) if reasons else ''}")
    print(f"[rung2_replay] predictions: {json.dumps(predictions['saturation'])} {json.dumps(predictions['hue'])}")
    print(f"[rung2_replay] numbers.json sha256 {sha}")
    print(f"[rung2_replay] wrote {out} in {time.time() - t0:.0f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, choices=list(OUT))
    ap.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    ap.add_argument("--steps", type=int, default=None, help="default: 10000 (overexp, rung3), 50000 (long)")
    ap.add_argument("--every", type=int, default=None,
                    help="checkpoint interval; default: 100 (overexp), 1000 (long, rung3)")
    ap.add_argument("--check_steps", type=int, default=10_000, help="long: the replays checked against the trainer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2, help="CPU threads (2 leaves room for a training batch)")
    ap.add_argument("--out", type=Path, default=None, help="default: the mode's folder; tests: a smoke* folder")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = guard(a.out or OUT[a.mode], a.mode)
    t0 = time.time()
    {"overexp": run_overexp, "long": run_long, "rung3": run_rung3}[a.mode](a, out, t0)


if __name__ == "__main__":
    main()
