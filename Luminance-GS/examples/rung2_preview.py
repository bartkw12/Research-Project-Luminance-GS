"""M2 step 6: what Rung 2's curve will do, before any 3DGS run (Implementation_guides/M2_rung2_plan.md).

In the default (detached) mode F and s_k learn only from L_curve, and L_curve depends only on the training
photos' HE-CDFs, never on the Gaussians. So a run's whole F and s_k trajectory can be replayed here without
rendering. The replay differs from a real run only in the random view order (a seeded permutation per epoch
here, the DataLoader's shuffle there).

    python rung2_preview.py                     (from Luminance-GS/examples, luminancegs env; CPU, ~25 min)
    python rung2_preview.py --scenes buu --steps 2000 --out ../results/rung2/preview_quick     (quick look)

Per scene, on LOM and on the varying-exposure set: load the training photos with the trainer's own Parser and
Dataset; compute the HE-CDFs, the measured exposures (as Runner.measure_exposures) and the 1/4-size grey
histograms; for each configuration replay the steps of 10 * L_curve with the trainer's optimisers (Adam on F's
side and on s, no weight decay, ExponentialLR to 1%) and the gauge projection; map every training photo through
the result.

Configurations. The default is the trainer's (decided 2026-11-13): quadratic knots, cdf init, one-bin-shifted
target, and the official shape-prior term with one learned global gamma on the shared F (priorF). Around it: the
usedrange knots and the identity init (the step 6 rules), the unshifted target (alignment check), no prior (the
plan's first draft; an M5 ablation) and the prior on each view's composed curve (priorL, the official form;
rejected 2026-11-13 because it pulls s_k toward 0).

Learning-rate rule (plan step 6, applied with the prior on): the default is also replayed at the other rates of
the ladder 1e-3, 3e-3, 1e-2 (F and gamma; s keeps --lr_s). The rule takes the first rate whose final objective is
within 5% of the full-batch optimum in every scene and set. The optimum minimises the same objective over all
views at the late weight: 10k full-batch Adam steps at lr 3e-2 from the cdf init, or a polish of the best ladder
replay if that ends lower. (The plan's 3k steps at 1e-2 suffice without the prior but stop up to 13% above the
optimum with it, because F then travels far from its init: scratch check of 2026-11-13.)

Tables (tables.md, one CSV each, numbers.json):
    P1 fit       residual x255 of the composed curves against their targets (grey-histogram weighted, as in
                 build_vexp.fit_exposure_form) next to the histogram oracle's on the same targets; final
                 objective against the full-batch optimum; steps to within 5% of the final objective; the
                 learning-rate ladder; steps the identity init needs to come within 0.01 of the cdf init's F
    P2 exposure  (varying set) s_k against u_k: slope, r, unpaired and paired RMSE, movement from the init,
                 next to the init and the histogram oracle (vexp_analyze.recovery_stats)
    P3 leak      paired pseudo-target leak: slope of log2(mean, varying / LOM) on u_k, next to M1's
    P4 dark end  pseudo-target value at input levels 0-16 and the dark-end slope (levels 1-8), next to detach's
                 (seed-0 curves_step10000.npz through the official LUT rule: level l reads curve[max(l-1, 0)]);
                 fraction of training pixels at levels 0-2; distance of the applied curves to detach's
    P5 quality   pseudo-target PSNR, raw and brightness-matched, against high/, next to detach's.
                 DIAGNOSTIC ONLY: it uses the normal-light images, so it never chooses a setting.
Writes only under results/rung2/ into a folder named preview* (default results/rung2/preview_final/; preview/ and
preview_lrF1e-2/ hold the 2026-11-11 runs the decision was based on, with the earlier configuration names).
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

EXAMPLES = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLES))

from build_vexp import fit_exposure_form, slope_r  # noqa: E402
from datasets.colmap import Dataset, Parser  # noqa: E402
from diagnostics import load_train_gt, nchw, psnr, psnr_gt_mean  # noqa: E402
from losses import HistogramPriorLoss  # noqa: E402
from rung0_analyze import INK, INK2, MUTED, SCENES, SEEDS, SERIES, md_table, read_csv, savefig, write_csv  # noqa: E402
from rung2 import ExposureCurve, hist_target, init_heights, make_knots, used_range_max  # noqa: E402
from vexp_analyze import recovery_stats, rmse_c  # noqa: E402

RESULTS = EXAMPLES.parent / "results"
LOM_DATA = EXAMPLES.parent / "data" / "LOM_full"
VEXP_DATA = EXAMPLES.parent / "data" / "LOM_vexp"
DATASET = RESULTS / "vexp" / "dataset"
RUNG2 = RESULTS / "rung2"
OUT_DEFAULT = RUNG2 / "preview_final"
SETS = {"lom": (LOM_DATA, RESULTS / "rung0"), "vexp": (VEXP_DATA, RESULTS / "vexp")}

# name: (knot spacing, F init, target alignment, shape prior)
CONFIGS = {
    "quadratic-cdf-priorF": ("quadratic", "cdf", "shift", "F"),             # the default (2026-11-13)
    "usedrange-cdf-priorF": ("usedrange", "cdf", "shift", "F"),             # knot-spacing rule
    "quadratic-identity-priorF": ("quadratic", "identity", "shift", "F"),   # F-init rule (head start, risk 5)
    "quadratic-cdf-priorF-unshifted": ("quadratic", "cdf", "none", "F"),    # alignment check
    "quadratic-cdf-noprior": ("quadratic", "cdf", "shift", "none"),         # the plan's first draft; M5 ablation
    "quadratic-cdf-priorL": ("quadratic", "cdf", "shift", "composed"),      # the official form; rejected
}
DEFAULT = "quadratic-cdf-priorF"
USEDRANGE = "usedrange-cdf-priorF"
IDENTITY = "quadratic-identity-priorF"
ALIGN_CHECK = "quadratic-cdf-priorF-unshifted"
NOPRIOR = "quadratic-cdf-noprior"
PRIOR_L = "quadratic-cdf-priorL"
SHOWN = [DEFAULT, ALIGN_CHECK, NOPRIOR, PRIOR_L]  # the configurations in the P2 to P5 tables
OPTIMUM_OF = {DEFAULT: DEFAULT, IDENTITY: DEFAULT, NOPRIOR: NOPRIOR}  # configs judged against a full-batch optimum
LR_LADDER = [1e-3, 3e-3, 1e-2]  # plan step 6: keep 1e-3 unless > 5% above the optimum, then 3e-3, then 1e-2
LEVELS = [0, 1, 2, 4, 8, 16]
LATE_STEP = 3000          # the objective is compared at the late-training weight w = 0.5
F_CLOSE = 0.01            # identity-init convergence: max |F_identity - F_cdf final| on the 255-bin grid
WITHIN = 0.05             # "within 5% of the final objective"
SPACING_GAIN = 0.30       # usedrange must lower the pooled P1 residual by at least 30% to become the default
LR_GAP = 5.0              # replay objective more than 5% above the full-batch optimum -> next rate of the ladder


# ------------------------------------------------------------------ output guard
def guard(out: Path) -> Path:
    """Write only under results/rung2/, into a folder whose name starts with "preview"."""
    out = out.resolve()
    if RUNG2.resolve() not in out.parents or not out.name.startswith("preview"):
        raise SystemExit(f"[rung2_preview] refusing to write to {out}: use results/rung2/preview*")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    return out


# ------------------------------------------------------------------ data
def load_set(scene: str, which: str, exp_name: str = "low") -> dict:
    """The training views exactly as the trainer loads them, with their HE-CDFs, grey histograms (the weights of
    fit_exposure_form), measured exposures (as Runner.measure_exposures) and 8-bit level counts. exp_name picks the
    input folder, as the trainer's --exp_name does (M5: "over_exp" for rung2_replay.py; this script uses "low")."""
    root = SETS[which][0]
    with contextlib.redirect_stdout(io.StringIO()):
        p = Parser(data_dir=str(root / scene), exp_name=exp_name, method="", factor=1, normalize=True, test_every=8)
        ds = Dataset(p, split="train")
    hp = HistogramPriorLoss()
    names, idx, images, m, cdfs, hists = [], [], [], [], [], []
    counts = np.zeros(256)
    for i in range(len(ds)):
        img = ds[i]["image"][None] / 255.0  # [1, H, W, 3]
        names.append(p.image_names[int(ds.indices[i])])
        idx.append(int(ds.indices[i]))
        images.append(img)
        m.append(float(img.mean()))
        cdfs.append(hp.compute_histogram_equalization(img))
        grey = torch.mean(torch.nn.functional.interpolate(img.permute(0, 3, 1, 2), scale_factor=0.25), dim=1)
        hists.append(torch.histc(grey.flatten(), bins=255, min=0.0, max=1.0).numpy())
        counts += np.bincount(np.clip(np.rint(img.numpy().ravel() * 255), 0, 255).astype(np.int64), minlength=256)
    m = np.array(m)
    s0 = np.log2(m).mean() - np.log2(m)
    return {"parser": p, "names": names, "idx": idx, "images": images, "mean_input": m,
            "s0": torch.tensor(s0, dtype=torch.float32), "cdf": torch.cat(cdfs), "hist": np.stack(hists),
            "counts": counts}


def load_gt(d: dict) -> list:
    out = []
    for img, i in zip(d["images"], d["idx"]):
        gt = torch.from_numpy(load_train_gt(d["parser"], i)).float()[None] / 255.0
        assert gt.shape == img.shape, f"GT {tuple(gt.shape)} vs input {tuple(img.shape)}"
        out.append(gt)
    return out


# ------------------------------------------------------------------ fitting
def build(d: dict, config: str, K: int, y_max_cache: dict):
    spacing, init, align, prior = CONFIGS[config]
    targets = hist_target(d["cdf"], align)
    y_max = None
    if spacing == "usedrange":
        if "y_max" not in y_max_cache:
            y_max_cache["y_max"] = used_range_max(d["images"], d["s0"])
        y_max = y_max_cache["y_max"]
    knots = make_knots(K, spacing, y_max)
    model = ExposureCurve(knots, init_heights(init, knots, targets, d["s0"]), d["s0"], prior=prior)
    return model, targets, y_max


@torch.no_grad()
def objective(model: ExposureCurve, targets: torch.Tensor, lam: float, exp_name: str = "low") -> float:
    """The curve loss (with the model's shape prior, if any) over all views at the late-training weight (exp_name
    sets that weight as rung2.curve_weight does: 0.5, or 0.1 for over_exp)."""
    return float(model.loss(model.composed_curve(torch.arange(model.N)), targets, LATE_STEP, exp_name, lam)[0])


def replay(model: ExposureCurve, targets: torch.Tensor, a, lr_F: float, exp_name: str = "low"):
    """The trainer's Rung 2 curve updates (simple_trainer_ours.Runner.train): one view per step, 10 * L_curve with
    the model's shape prior, Adam on F's side (theta, rho) at lr_F and on s at a.lr_s, ExponentialLR to 1% over
    the run, then the gauge projection. Returns checkpoints (step, objective, F on the 255-bin grid). exp_name
    sets the curve weight as the trainer's --exp_name does (M5: "over_exp" for rung2_replay.py)."""
    opts = [torch.optim.Adam(model.f_params(), lr=lr_F, weight_decay=0.0),
            torch.optim.Adam([model.s], lr=a.lr_s, weight_decay=0.0)]
    scheds = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=0.01 ** (1.0 / a.steps)) for o in opts]
    gen = torch.Generator().manual_seed(a.seed)
    order, traj = [], []
    for step in range(a.steps):
        if step % a.every == 0:
            traj.append((step, objective(model, targets, a.smooth_lambda, exp_name),
                         model.F(model.grid).detach().numpy()))
        if not order:
            order = torch.randperm(model.N, generator=gen).tolist()
        k = order.pop(0)
        loss, _ = model.loss(model.composed_curve(k), targets[k:k + 1], step, exp_name, a.smooth_lambda)
        for o in opts:
            o.zero_grad(set_to_none=True)
        (10 * loss).backward()
        for o in opts:
            o.step()
        for s in scheds:
            s.step()
        model.project_gauge()
    traj.append((a.steps, objective(model, targets, a.smooth_lambda, exp_name), model.F(model.grid).detach().numpy()))
    return traj


def full_batch(model: ExposureCurve, targets: torch.Tensor, steps: int, lr: float, lam: float) -> float:
    """All views per step at the late weight; Adam on every parameter, ExponentialLR to 1%; gauge projection."""
    opt = torch.optim.Adam(list(model.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.01 ** (1.0 / steps))
    views = torch.arange(model.N)
    for _ in range(steps):
        loss, _ = model.loss(model.composed_curve(views), targets, LATE_STEP, "low", lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        model.project_gauge()
    return objective(model, targets, lam)


def optimum(d: dict, config: str, replayed: list, a, y_cache: dict) -> dict:
    """The full-batch optimum of a configuration's objective: a cold start from its init, and a polish of the
    lowest-ending replay of the same objective; the lower of the two."""
    m, t, _ = build(d, config, a.K, y_cache)
    cold = full_batch(m, t, a.full_steps, a.full_lr, a.smooth_lambda)
    best = min(replayed, key=lambda r: r["traj"][-1][1])
    polish = full_batch(copy.deepcopy(best["model"]), best["targets"], a.polish_steps, a.polish_lr, a.smooth_lambda)
    return {"J": min(cold, polish), "cold": cold, "polish": polish, "from": "cold" if cold <= polish else "polish"}


def weighted_resid255(curves: np.ndarray, targets: np.ndarray, hist: np.ndarray) -> float:
    """fit_exposure_form's residual: grey-histogram weights normalised per view, RMS over views, x255."""
    W = hist / hist.sum(1, keepdims=True)
    return float(np.sqrt(((curves - targets) ** 2 * W).sum() / len(curves)) * 255)


def first_step_within(traj: list, frac: float) -> int:
    """Earliest checkpoint after which the objective stays within frac of its final value."""
    final = traj[-1][1]
    step = traj[-1][0]
    for t, j, _ in reversed(traj):
        if j > final * (1 + frac):
            break
        step = t
    return step


def first_step_close(traj: list, F_ref: np.ndarray, tol: float):
    for t, _, Fg in traj:
        if float(np.abs(Fg - F_ref).max()) < tol:
            return t
    return float("nan")


# ------------------------------------------------------------------ per-config outputs
@torch.no_grad()
def evaluate(model: ExposureCurve, d: dict, gt: list) -> dict:
    """Pseudo-targets of every training view, their means and PSNRs, and the curve at the dark levels."""
    mean_pseudo, p_raw, p_bm = [], [], []
    for k, (img, g) in enumerate(zip(d["images"], gt)):
        pseudo = model.map_pixels(img, k, "none")
        mean_pseudo.append(float(pseudo.mean()))
        p_raw.append(psnr(nchw(pseudo), nchw(g)))
        p_bm.append(psnr_gt_mean(nchw(pseudo), nchw(g)))
    x = torch.arange(256, dtype=torch.float32) / 255.0  # every 8-bit input level
    L = model.F(torch.clamp(model.gain(torch.arange(model.N))[:, None] * x[None], max=1.0)).numpy()  # [V, 256]
    return {"mean_pseudo": np.array(mean_pseudo), "pseudo_psnr": np.array(p_raw), "pseudo_psnr_gtmean": np.array(p_bm),
            "L_levels": L, "F_levels": model.F(x).numpy(), "gamma": float(model.gamma()),
            "slope_1_8": float(((L[:, 8] - L[:, 1]) / (7 / 255)).mean()),
            "curves": model.composed_curve(torch.arange(model.N)).numpy()}


def detach_reference(scene: str, which: str, names: list) -> dict:
    """detach on the same set: seed-0 per-view curves (local npz) and the seed mean/sd of its pseudo-target PSNR."""
    root = SETS[which][1]
    z = np.load(root / f"{scene}_detach_seed0" / "diagnostics" / "curves_step10000.npz")
    assert [str(v) for v in z["view_names"]] == names, f"{scene}/{which}: detach view order differs"
    c = z["curve_adj"].astype(np.float64)  # [V, 255]
    lut = np.stack([c[:, int(254 * l / 255)] for l in range(256)], 1)  # the official truncating lookup, levels 0..255
    out = {"L_levels": lut, "slope_1_8": float(((lut[:, 8] - lut[:, 1]) / (7 / 255)).mean()),
           "gamma": float(z["gamma"].mean())}
    for key in ("pseudo_psnr", "pseudo_psnr_gtmean"):
        per_seed = [np.mean([float(r[key]) for r in
                             read_csv(root / f"{scene}_detach_seed{s}" / "rescore" / "pseudo_targets.csv")])
                    for s in SEEDS]
        out[key] = (float(np.mean(per_seed)), float(np.std(per_seed, ddof=1)))
    return out


# ------------------------------------------------------------------ figures
def fig_curves(scene: str, per_set: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (which, (model, targets, model_noprior, detach_lut)) in zip(axes, per_set.items()):
        T = targets.numpy()
        x = model.grid.numpy()
        with torch.no_grad():
            curves = model.composed_curve(torch.arange(model.N)).numpy()
            Fg = model.F(model.grid).numpy()
            p = model.prior_curve()[0].numpy()
            Fn = model_noprior.F(model.grid).numpy()
        xmax = min(1.0, 1.25 * float(np.argmax(T >= 0.995, axis=1).max()) / 254)
        for k in range(model.N):
            ax.plot(x, T[k], color=MUTED, lw=0.8, alpha=0.6, label="views' targets (shifted HE-CDF)" if k == 0 else None)
            ax.plot(x, curves[k], color=SERIES[0], lw=1.0, alpha=0.6,
                    label="per-view curve F(min(1, 2^s_k x))" if k == 0 else None)
        ax.plot(x, p, color=SERIES[3], lw=1.6, ls=":", label=f"gamma prior x^{float(model.gamma()):.3f}")
        ax.plot(x, Fn, color=SERIES[1], lw=1.6, label="shared F without prior (first draft)")
        ax.plot(np.arange(256) / 255, detach_lut.mean(0), color=SERIES[2], lw=2.0, ls="--",
                label="detach, mean applied curve (seed 0)")
        ax.plot(x, Fg, color=INK, lw=2.2, label="shared F (default: with gamma prior)")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1.02)
        ax.set_title("LOM" if which == "lom" else "varying-exposure set")
        ax.set_xlabel("input value x (0-1)")
    axes[0].set_ylabel("output")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{scene}: Rung 2 preview ({DEFAULT}, curve loss only, no 3DGS)", color=INK)
    return fig


def fig_exposure(rows: dict):
    scenes = list(rows)
    fig, axes = plt.subplots(1, len(scenes), figsize=(3.3 * len(scenes), 3.6), sharey=True, squeeze=False)
    for ax, sc in zip(axes[0], scenes):
        r = rows[sc]
        u = r["u"]
        line = np.linspace(u.min(), u.max(), 2)
        ax.plot(line, -(line - u.mean()), color=MUTED, lw=1.2, ls="--", label="ideal: s = -(u - mean u)")
        # Rung 2 last and smallest: its s_k stay close to the init and would hide under it
        for (key, label), color, marker, size, z in zip(
                (("rung2", "Rung 2 preview"), ("init", "init log2(m_bar/m_k)"), ("oracle", "histogram oracle")),
                SERIES, ("o", "s", "^"), (22, 40, 34), (5, 3, 4)):
            ax.scatter(u, r[key], s=size, color=color, marker=marker, edgecolors="white", linewidths=0.8,
                       label=label, zorder=z)
        ax.set_title(sc)
        ax.set_xlabel("applied gain u_k (stops)")
    axes[0, 0].set_ylabel("exposure s_k (stops)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08))
    return fig


# ------------------------------------------------------------------ main
def f3(x):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--every", type=int, default=100, help="checkpoint interval of the replay")
    ap.add_argument("--full_steps", type=int, default=10_000, help="cold-start full-batch steps for the optimum")
    ap.add_argument("--full_lr", type=float, default=3e-2)
    ap.add_argument("--polish_steps", type=int, default=3000, help="full-batch polish of the best replay")
    ap.add_argument("--polish_lr", type=float, default=1e-3)
    ap.add_argument("--lr_F", type=float, default=1e-2, help="F's side (theta, gamma); the trainer's default")
    ap.add_argument("--lr_s", type=float, default=1e-3)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--smooth_lambda", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2, help="CPU threads (2 leaves room for a training batch)")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    a = ap.parse_args()
    if a.lr_F not in LR_LADDER:
        raise SystemExit(f"--lr_F must be on the ladder {LR_LADDER}")

    torch.set_num_threads(a.threads)
    cv2.setNumThreads(a.threads)
    out = guard(a.out)
    fig_dir = out / "figures"
    t_start = time.time()

    gains = json.loads((DATASET / "gains.json").read_text())["rows"]
    oracle_json = {r["scene"]: r for r in json.loads((DATASET / "oracle.json").read_text())["rows"]}
    t4 = {(r["scene"], r["method"]): r for r in read_csv(RESULTS / "vexp" / "summary" / "T4_exposure.csv")}

    P1, LADDER, P2, P3, P4, P4s, P4d, P5, VIEWS, CHECKS = [], [], [], [], [], [], [], [], [], []
    exp_fig = {}
    for scene in a.scenes:
        t0 = time.time()
        data = {w: load_set(scene, w) for w in SETS}
        names = data["lom"]["names"]
        assert data["vexp"]["names"] == names, f"{scene}: view order differs between LOM and LOM_vexp"
        gt = load_gt(data["lom"])  # LOM_vexp copies high/ unchanged, so one GT serves both sets
        u = np.array([{r["view"]: r["u"] for r in gains if r["scene"] == scene}[n] for n in names])
        ue = np.array([{r["view"]: r["log2_ratio"] for r in gains if r["scene"] == scene}[n] for n in names])

        # checks against M1's numbers
        oj = oracle_json[scene]
        assert oj["view_names"] == names, f"{scene}: oracle.json view order differs"
        oracle = {w: fit_exposure_form(data[w]["cdf"].double().numpy(), data[w]["hist"])[0] for w in SETS}
        CHECKS.append({"scene": scene,
                       "init_vs_oracle_json_s_init": float(np.abs(data["vexp"]["s0"].double().numpy() - oj["s_init"]).max()),
                       "oracle_vs_oracle_json_s_oracle": float(np.abs(oracle["vexp"] - oj["s_oracle"]).max())})

        res, opt = {}, {}
        for w, d in data.items():
            y_cache = {}
            for name in CONFIGS:
                model, targets, y_max = build(d, name, a.K, y_cache)
                tc = time.time()
                traj = replay(model, targets, a, a.lr_F)
                res[(w, name)] = {"model": model, "targets": targets, "traj": traj, "y_max": y_max,
                                  "secs": time.time() - tc, **evaluate(model, d, gt)}
            # the learning-rate ladder: the default at the other rates (same seed, init and view order)
            ladder = {a.lr_F: res[(w, DEFAULT)]}
            for lr in LR_LADDER:
                if lr not in ladder:
                    model, targets, _ = build(d, DEFAULT, a.K, y_cache)
                    ladder[lr] = {"model": model, "targets": targets, "traj": replay(model, targets, a, lr)}
            opt[w] = {DEFAULT: optimum(d, DEFAULT, list(ladder.values()) + [res[(w, IDENTITY)]], a, y_cache),
                      NOPRIOR: optimum(d, NOPRIOR, [res[(w, NOPRIOR)]], a, y_cache)}
            oc = opt[w][DEFAULT]
            for lr in LR_LADDER:
                r = ladder[lr]
                J = r["traj"][-1][1]
                LADDER.append({"scene": scene, "set": w, "lr_F": lr, "objective": J, "objective_opt": oc["J"],
                               "gap_pct": 100 * (J / oc["J"] - 1), "step_within_5pct": first_step_within(r["traj"], WITHIN),
                               "gamma": float(r["model"].gamma()),
                               "F_maxdiff_to_main": float(np.abs(r["traj"][-1][2] - ladder[a.lr_F]["traj"][-1][2]).max())})
        print(f"[rung2_preview] {scene}: fitted {len(CONFIGS)} configurations and the ladder on both sets "
              f"({time.time() - t0:.0f} s)", flush=True)

        # P1 fit
        refs = {w: detach_reference(scene, w, names) for w in SETS}
        nan = float("nan")
        oracle_resid = {(w, al): fit_exposure_form(hist_target(d["cdf"], al).double().numpy(), d["hist"])[2]
                        for w, d in data.items() for al in ("shift", "none")}
        for w, d in data.items():
            F_ref = res[(w, DEFAULT)]["traj"][-1][2]
            for name, (spacing, init, align, prior) in CONFIGS.items():
                r = res[(w, name)]
                J = r["traj"][-1][1]
                oc = opt[w].get(OPTIMUM_OF.get(name))
                P1.append({"scene": scene, "set": w, "config": name, "y_max": r["y_max"] if r["y_max"] else nan,
                           "resid255": weighted_resid255(r["curves"], r["targets"].numpy(), d["hist"]),
                           "oracle_resid255": oracle_resid[(w, align)],
                           "objective": J, "objective_opt": oc["J"] if oc else nan,
                           "objective_opt_cold": oc["cold"] if oc else nan,
                           "objective_opt_polish": oc["polish"] if oc else nan,
                           "gap_pct": 100 * (J / oc["J"] - 1) if oc else nan,
                           "step_within_5pct": first_step_within(r["traj"], WITHIN),
                           "gamma": r["gamma"], "detach_gamma_mean": refs[w]["gamma"],
                           "init_to_final_F_maxdiff": float(np.abs(r["traj"][0][2] - r["traj"][-1][2]).max()),
                           "steps_to_default_F_0.01": first_step_close(r["traj"], F_ref, F_CLOSE) if name == IDENTITY else nan,
                           "replay_seconds": r["secs"]})

        # P2 exposure (varying set), P3 leak
        s_v = {n: res[("vexp", n)]["model"].s.detach().double().numpy() for n in CONFIGS}
        s_l = {n: res[("lom", n)]["model"].s.detach().double().numpy() for n in CONFIGS}
        s0_v, s0_l = data["vexp"]["s0"].double().numpy(), data["lom"]["s0"].double().numpy()
        methods = [("init", s0_v, s0_l), ("oracle", oracle["vexp"], oracle["lom"])] + \
                  [(n, s_v[n], s_l[n]) for n in CONFIGS]
        for name, sv, sl in methods:
            rec = recovery_stats(sv, sl, u, ue)
            P2.append({"scene": scene, "method": name, "slope": rec["slope"], "r": rec["r"], "rmse": rec["rmse"],
                       "slope_paired": rec["slope_paired"], "rmse_paired": rec["rmse_paired"],
                       "rmse_ueff": rec["rmse_ueff"], "move_from_init": rmse_c(sv - s0_v)})
        exp_fig[scene] = {"u": u, "rung2": s_v[DEFAULT], "init": s0_v, "oracle": oracle["vexp"]}

        row = {"scene": scene,
               "input": slope_r(np.log2(data["vexp"]["mean_input"] / data["lom"]["mean_input"]), u)[0],
               "official_M1": float(t4[(scene, "official")]["pseudo_leak_paired_mean"]),
               "detach_M1": float(t4[(scene, "detach")]["pseudo_leak_paired_mean"])}
        for n in CONFIGS:
            row[n] = slope_r(np.log2(res[("vexp", n)]["mean_pseudo"] / res[("lom", n)]["mean_pseudo"]), u)[0]
        row["unpaired_" + DEFAULT] = slope_r(np.log2(res[("vexp", DEFAULT)]["mean_pseudo"]), u)[0]
        P3.append(row)

        # P4 dark end, P5 quality
        for w, d in data.items():
            ref = refs[w]
            frac = float(d["counts"][:3].sum() / d["counts"].sum())
            P4s.append({"scene": scene, "set": w, "frac_levels_0_2": frac, "detach": ref["slope_1_8"],
                        **{n: res[(w, n)]["slope_1_8"] for n in SHOWN}})
            for l in LEVELS:
                P4.append({"scene": scene, "set": w, "level": l,
                           "detach": 255 * float(ref["L_levels"][:, l].mean()),
                           **{n: 255 * float(res[(w, n)]["L_levels"][:, l].mean()) for n in SHOWN},
                           "F_s0_" + DEFAULT: 255 * float(res[(w, DEFAULT)]["F_levels"][l])})
            # whole-curve distance to detach as applied to input levels 0..255, weighted by how often each level
            # occurs in this set's training photos (all views and channels; training photos only), x255
            wts = d["counts"] / d["counts"].sum()
            for n in SHOWN:
                diff = np.abs(res[(w, n)]["L_levels"] - ref["L_levels"]).mean(0)  # [256], mean over views
                P4d.append({"scene": scene, "set": w, "config": n, "wmean_abs_diff_to_detach_x255": 255 * float(diff @ wts),
                            "max_abs_diff_to_detach_x255": 255 * float(diff.max())})
            for n in CONFIGS:
                r = res[(w, n)]
                m_raw, m_bm = float(r["pseudo_psnr"].mean()), float(r["pseudo_psnr_gtmean"].mean())
                (d_raw, sd_raw), (d_bm, sd_bm) = ref["pseudo_psnr"], ref["pseudo_psnr_gtmean"]
                P5.append({"scene": scene, "set": w, "config": n, "pseudo_psnr": m_raw, "pseudo_psnr_gtmean": m_bm,
                           "detach_psnr": d_raw, "detach_psnr_sd": sd_raw,
                           "detach_psnr_gtmean": d_bm, "detach_psnr_gtmean_sd": sd_bm,
                           "gap_psnr": m_raw - d_raw, "gap_gtmean": m_bm - d_bm,
                           "gap_gtmean_x_sd": (m_bm - d_bm) / sd_bm if sd_bm > 0 else float("nan")})
                for k, view in enumerate(names):
                    VIEWS.append({"scene": scene, "set": w, "config": n, "view": view,
                                  "u": u[k] if w == "vexp" else 0.0, "s_init": float(d["s0"][k]),
                                  "s_final": float(r["model"].s[k]), "mean_input": float(d["mean_input"][k]),
                                  "mean_pseudo": float(r["mean_pseudo"][k]),
                                  "pseudo_psnr": float(r["pseudo_psnr"][k]),
                                  "pseudo_psnr_gtmean": float(r["pseudo_psnr_gtmean"][k])})

        savefig(fig_curves(scene, {w: (res[(w, DEFAULT)]["model"], res[(w, DEFAULT)]["targets"],
                                       res[(w, NOPRIOR)]["model"], refs[w]["L_levels"]) for w in SETS}),
                f"curves_{scene}", fig_dir)
        print(f"[rung2_preview] {scene}: done ({time.time() - t0:.0f} s)", flush=True)

    savefig(fig_exposure(exp_fig), "exposure_vs_u", fig_dir)

    # ------------------------------------------------------------------ decision rules (training photos only)
    def pooled(config, key="resid255"):
        return float(np.mean([r[key] for r in P1 if r["config"] == config]))

    q, ur = pooled(DEFAULT), pooled(USEDRANGE)
    ladder_max = {lr: float(np.max([r["gap_pct"] for r in LADDER if r["lr_F"] == lr])) for lr in LR_LADDER}
    lr_pick = next((lr for lr in LR_LADDER if ladder_max[lr] <= LR_GAP), None)
    ident = [r["steps_to_default_F_0.01"] for r in P1 if r["config"] == IDENTITY]
    p5_lom = [r for r in P5 if r["config"] == DEFAULT and r["set"] == "lom"]
    worse = [r["scene"] for r in p5_lom if r["gap_gtmean"] < -r["detach_psnr_gtmean_sd"]]
    decisions = {
        "knot_spacing": {"pooled_resid255_quadratic": q, "pooled_resid255_usedrange": ur,
                         "reduction": 1 - ur / q, "rule": f"switch to usedrange if reduction >= {SPACING_GAIN:.0%}",
                         "outcome": "usedrange" if 1 - ur / q >= SPACING_GAIN else "quadratic"},
        "learning_rate": {"ladder_max_gap_pct": {f"{lr:g}": v for lr, v in ladder_max.items()},
                          "rule": f"first lr_F of {LR_LADDER} whose largest gap is <= {LR_GAP}%",
                          "outcome": lr_pick, "main_tables_lr_F": a.lr_F, "consistent": lr_pick == a.lr_F},
        "identity_init_steps_to_0.01": [float(x) for x in ident],
        "p5_lom_scenes_worse_than_detach_sd": worse,
        "p5_rule": "stop and ask Bart if 2 or more LOM scenes (diagnostic only; never chooses a setting)",
        "p5_outcome": "stop and ask Bart" if len(worse) >= 2 else "no flag",
    }

    # ------------------------------------------------------------------ outputs
    for name, rows in (("P1_fit", P1), ("P1_lr_ladder", LADDER), ("P2_exposure", P2), ("P3_leak", P3),
                       ("P4_dark_end", P4), ("P4_dark_slope", P4s), ("P4_distance_to_detach", P4d),
                       ("P5_pseudo_quality", P5), ("views", VIEWS), ("checks", CHECKS)):
        write_csv(name, rows, out)
    meta = {"date": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t_start),
            "script_sha256_16": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
            "rung2_sha256_16": hashlib.sha256((EXAMPLES / "rung2.py").read_bytes()).hexdigest()[:16],
            "torch": torch.__version__, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
            "configs": CONFIGS, "default": DEFAULT, "lr_ladder": LR_LADDER}
    (out / "numbers.json").write_text(json.dumps({"meta": meta, "decisions": decisions, "checks": CHECKS}, indent=1))

    def sel(rows, **kw):
        return [r for r in rows if all(r[k] == v for k, v in kw.items())]

    def steps_str(v):
        return ", ".join("never" if np.isnan(x) else str(int(x)) for x in v)

    row_order = ", ".join(f"{p['scene']}/{p['set']}" for p in sel(P1, config=DEFAULT))
    lr_str = "; ".join(f"{lr:g}: {ladder_max[lr]:+.2f}%" for lr in LR_LADDER)
    if lr_pick is None:
        lr_outcome = "**none of the ladder passes: ask Bart**"
    else:
        lr_outcome = f"**{lr_pick:g}**" + ("" if lr_pick == a.lr_F else
                                          f" (these tables used lr_F {a.lr_F:g}: rerun with --lr_F {lr_pick:g})")

    md = [f"# Rung 2 preview (M2 step 6), final configuration\n",
          f"Curve loss only, no 3DGS: a replay of the trainer's F and s_k updates on the training photos. "
          f"{a.steps} steps, lr {a.lr_F:g} (F and gamma) and {a.lr_s:g} (s), K = {a.K}, smoothness "
          f"{a.smooth_lambda:g}, seed {a.seed}. Default configuration `{DEFAULT}` (decided 2026-11-13: one-bin-shifted "
          f"target, and the official shape-prior term 0.5 [mse(p, F) + 0.01 mse(p, T_k)] with one learned global "
          f"gamma, p = clamp(x, 1e-3, 1)^gamma, on the shared F). `{ALIGN_CHECK}` is the alignment check, `{NOPRIOR}` "
          f"the plan's first draft (an M5 ablation), `{PRIOR_L}` the prior in the official form (rejected). "
          f"Generated {meta['date']} in {meta['seconds']} s by `rung2_preview.py` ({meta['script_sha256_16']}). "
          f"Every number here is *measured* on the replay; how a 3DGS run follows it is *inferred*.\n",
          "## Decision rules (training photos only)\n",
          f"- **Learning rate (F and gamma), plan step 6 with the prior on:** largest gap of the replay objective above "
          f"the full-batch optimum, over scenes and sets, per rate: {lr_str} (rule: the first rate of the ladder "
          f"within {LR_GAP:g}%) -> {lr_outcome}.",
          f"- **Knot spacing:** pooled P1 residual quadratic {q:.3f}/255, usedrange {ur:.3f}/255, reduction "
          f"{100 * (1 - ur / q):.1f}% (rule: switch at >= 30%) -> **{decisions['knot_spacing']['outcome']}**.",
          f"- **F init:** steps the identity init needs to come within {F_CLOSE} of the cdf init's final F: "
          f"{steps_str(ident)} (in the order {row_order}).",
          f"- **P5 flag (diagnostic only):** LOM scenes where `{DEFAULT}` is below `detach` by more than detach's "
          f"brightness-matched seed sd: {', '.join(worse) if worse else 'none'} -> **{decisions['p5_outcome']}**.\n"]

    md += ["## P1 learning-rate ladder (default configuration)\n",
           f"Objective = the curve loss with the prior over all views at the late weight. Optimum = the lower of "
           f"{a.full_steps} cold-start full-batch Adam steps (lr {a.full_lr:g}) and a {a.polish_steps}-step polish "
           f"(lr {a.polish_lr:g}) of the lowest-ending replay (ladder and identity init).\n"]
    md += [md_table([{"scene": r["scene"], "set": r["set"], "lr_F": f"{r['lr_F']:g}", "objective": f"{r['objective']:.4e}",
                      "optimum": f"{r['objective_opt']:.4e}", "gap %": f"{r['gap_pct']:+.2f}",
                      "step within 5%": r["step_within_5pct"], "gamma": f3(r["gamma"]),
                      f"max dF to lr {a.lr_F:g}": f3(r["F_maxdiff_to_main"])} for r in LADDER])]

    md += ["\n## P1 fit: weighted residual x255 against the targets\n"]
    t = []
    for r in sel(P1, config=DEFAULT):
        row = {"scene": r["scene"], "set": r["set"], "oracle": f3(r["oracle_resid255"])}
        for n in CONFIGS:
            if n != ALIGN_CHECK:
                row[n] = f3(sel(P1, scene=r["scene"], set=r["set"], config=n)[0]["resid255"])
        row["usedrange y_max"] = f3(sel(P1, scene=r["scene"], set=r["set"], config=USEDRANGE)[0]["y_max"])
        row["gamma"] = f3(r["gamma"])
        row["detach gamma (mean)"] = f3(r["detach_gamma_mean"])
        t.append(row)
    md += [md_table(t), "\nObjective against the full-batch optimum, and convergence.\n"]
    md += [md_table([{"scene": r["scene"], "set": r["set"], "config": r["config"], "objective": f"{r['objective']:.4e}",
                      "optimum": f"{r['objective_opt']:.4e}", "optimum from": "cold" if r["objective_opt"] == r["objective_opt_cold"] else "polish",
                      "gap %": f"{r['gap_pct']:+.2f}", "step within 5%": r["step_within_5pct"],
                      "init-to-final max dF": f3(r["init_to_final_F_maxdiff"]),
                      "replay s": f"{r['replay_seconds']:.1f}"} for r in P1 if r["config"] in (DEFAULT, NOPRIOR)])]

    md += ["\n## P2 exposure on the varying set: s_k against u_k (ideal slope -1, RMSE 0)\n"]
    md += [md_table([{"scene": r["scene"], "method": r["method"], "slope": f"{r['slope']:+.3f}", "r": f"{r['r']:+.3f}",
                      "rmse": f3(r["rmse"]), "rmse paired": f3(r["rmse_paired"]),
                      "move from init": f3(r["move_from_init"])}
                     for r in P2 if r["method"] in ["init", "oracle", USEDRANGE] + SHOWN])]

    md += ["\n## P3 paired pseudo-target leak (slope of log2(mean, varying / LOM) on u_k; 0 = compensated, "
           "1 = passes through)\n",
           "M1 columns: mean over 3 seeds from `results/vexp/summary/T4_exposure.csv`.\n"]
    md += [md_table([{k: (f"{v:+.3f}" if isinstance(v, float) else v) for k, v in r.items()} for r in P3])]

    md += ["\n## P4 dark end (LOM): mean pseudo-target value over views at input level l, in 8-bit levels\n",
           "detach: seed-0 curves through the official LUT (level l reads curve[max(l-1, 0)]).\n"]
    md += [md_table([{"scene": r["scene"], "level": r["level"], "detach": f"{r['detach']:.2f}",
                      **{n: f"{r[n]:.2f}" for n in SHOWN}} for r in P4 if r["set"] == "lom"])]
    md += ["\nDark-end slope (mean output gain per input step over levels 1-8; how much photo noise is amplified)"
           " and the fraction of training pixels at levels 0-2.\n"]
    md += [md_table([{"scene": r["scene"], "set": r["set"], "pixels at 0-2": f"{100 * r['frac_levels_0_2']:.2f}%",
                      "detach": f"{r['detach']:.2f}", **{n: f"{r[n]:.2f}" for n in SHOWN}} for r in P4s])]
    md += ["\nDistance to detach's curves as applied to input levels 0-255 (mean over views, weighted by how often "
           "each level occurs in the training photos), x255.\n"]
    t = []
    for r in sel(P4d, config=DEFAULT):
        t.append({"scene": r["scene"], "set": r["set"],
                  **{n: f"{sel(P4d, scene=r['scene'], set=r['set'], config=n)[0]['wmean_abs_diff_to_detach_x255']:.2f}"
                     for n in SHOWN}})
    md += [md_table(t)]

    md += ["\n## P5 pseudo-target quality against high/ (DIAGNOSTIC ONLY: never chooses a setting)\n",
           "detach: mean over views per run, then mean ± sd over seeds 0-2 (`rescore/pseudo_targets.csv`).\n"]
    md += [md_table([{"scene": r["scene"], "set": r["set"], "config": r["config"],
                      "PSNR": f"{r['pseudo_psnr']:.2f}", "detach PSNR": f"{r['detach_psnr']:.2f} ± {r['detach_psnr_sd']:.2f}",
                      "bm PSNR": f"{r['pseudo_psnr_gtmean']:.2f}",
                      "detach bm": f"{r['detach_psnr_gtmean']:.2f} ± {r['detach_psnr_gtmean_sd']:.2f}",
                      "bm gap": f"{r['gap_gtmean']:+.2f}", "gap / sd": f"{r['gap_gtmean_x_sd']:+.1f}"}
                     for r in P5 if r["config"] in SHOWN])]

    md += ["\n## Checks against M1\n",
           md_table([{k: (f"{v:.2e}" if isinstance(v, float) else v) for k, v in r.items()} for r in CHECKS]),
           "\nTolerances as in `vexp_analyze.py`: init 1e-6, oracle 2e-3 stops.\n",
           "\nFigures: `figures/curves_<scene>.png` (default configuration), `figures/exposure_vs_u.png`.\n"]
    (out / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[rung2_preview] wrote {out} in {time.time() - t_start:.0f} s")
    for k, v in decisions.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
