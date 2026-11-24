"""M5 curve-only replays (Implementation_guides/M5_plan.md): Rung 2's F and s_k trained on the training photos alone,
with no 3DGS, by rung2_preview.py's machinery (the trainer's data loading, model, optimisers, schedule and gauge).

    python rung2_replay.py --mode overexp   E6's pre-launch check (M5_plan.md step 2): the default configuration on
                                            the LOM over_exp photos, 10k steps, curve weight as --exp_name over_exp
                                            (0.1 from step 3000) -> results/overexp/preview/
    python rung2_replay.py --mode long      A1 (M5_plan.md 3.5): cdf and identity inits on the LOM low-light photos,
                                            50k steps, F gap and curve loss every 1k steps, plus 10k replays checked
                                            against the trainer's gap -> results/rung2/replay_long/

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
CLAUDE.md rule 6 protects.
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

from rung0_analyze import INK, MUTED, SCENES, SERIES, md_table, savefig, write_csv  # noqa: E402
from rung2_preview import DEFAULT, F_CLOSE, IDENTITY, build, load_set, replay, weighted_resid255  # noqa: E402
from vexp_analyze import rmse_c  # noqa: E402

RESULTS = EXAMPLES.parent / "results"
OUT = {"overexp": RESULTS / "overexp" / "preview", "long": RESULTS / "rung2" / "replay_long"}
PROTECTED = [RESULTS / "rung0" / "summary", RESULTS / "rung0" / "summary_rung05", RESULTS / "vexp" / "summary",
             RESULTS / "rung2" / "summary", RESULTS / "rung2" / "summary_seed0", RESULTS / "rung2" / "summary_headstart",
             RESULTS / "rung2" / "summary_headstart_seeds012", RESULTS / "rung2" / "preview_final"]  # CLAUDE.md rule 6
LOW_PREVIEW = RESULTS / "rung2" / "preview_final" / "P1_fit.csv"
HEADSTART = RESULTS / "rung2" / "summary_headstart_seeds012" / "curves.csv"
LR_F, LR_S, K, LAM = 1e-2, 1e-3, 32, 0.1  # the trainer's Rung 2 defaults (simple_trainer_ours.Config)
GAMMA_MIN, BELOW_FRAC, RESID_RATIO = 1.0, 0.80, 2.0  # overexp stop rule
SHRINK, CHECK_TOL = 0.10, 0.01  # long: "stopped shrinking" over the last fifth; built-in check tolerance
INTERIOR = slice(1, 254)  # grid points b = 1..253 (F(0) = 0 and F(1) = 1 are pinned)


def guard(out: Path, mode: str) -> Path:
    out = out.resolve()
    for p in PROTECTED:
        p = p.resolve()
        if out == p or p in out.parents:
            raise SystemExit(f"[rung2_replay] refusing to write to {out}: protected by CLAUDE.md rule 6")
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
           for f in ("rung2_replay.py", "rung2_preview.py", "rung2.py")}
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", required=True, choices=list(OUT))
    ap.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    ap.add_argument("--steps", type=int, default=None, help="default: 10000 (overexp), 50000 (long)")
    ap.add_argument("--every", type=int, default=None, help="checkpoint interval; default: 100 (overexp), 1000 (long)")
    ap.add_argument("--check_steps", type=int, default=10_000, help="long: the replays checked against the trainer")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2, help="CPU threads (2 leaves room for a training batch)")
    ap.add_argument("--out", type=Path, default=None, help="default: the mode's folder; tests: a smoke* folder")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = guard(a.out or OUT[a.mode], a.mode)
    t0 = time.time()
    (run_overexp if a.mode == "overexp" else run_long)(a, out, t0)


if __name__ == "__main__":
    main()
