"""Rung 0 diagnostics for Luminance-GS (measurement only, no new method).

Logs the loss components, snapshots every training view's tone curve at chosen
steps, and after training scores the pseudo-enhanced targets (against the 3DGS
render and, as an oracle, the normal-light images in ``high/``) and their
cross-view consistency via depth-based warping.

Every entry point runs under ``torch.no_grad()`` inside ``rng_guard()`` and
reads views with ``trainset[i]`` rather than a DataLoader (creating a
DataLoader iterator draws from the global torch RNG even with shuffle=False),
so enabling diagnostics does not change the training trajectory.

Do not import ``simple_trainer_ours`` here: it parses the CLI at import time.
"""

import contextlib
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from losses import HistogramPriorLoss

LOSS_FIELDS = [
    "step", "total", "loss_regress_low", "loss_regress_enh", "loss_co", "hist_loss",
    "hist_cdf", "hist_pseudo", "hist_smooth", "w_cdf",
]
N_EXAMPLE_VIEWS = 3
N_NEIGHBOURS = 2
ALPHA_MIN = 0.5         # pixels with lower accumulated opacity have unreliable depth
DEPTH_REL_TOL = 0.05    # occlusion test: |z_proj - z_rendered| < 5% of z_rendered

# Rung 0.5 colour metrics
COLOR_DARK_NORM = 0.05 * 3 ** 0.5   # GT pixels with |rgb| below this are too dark for a reliable colour
HUE_GT_SAT_MIN = 0.3                # hue error only where the GT is clearly coloured ...
HUE_GT_VAL_MIN = 0.1
HUE_PRED_SAT_MIN = 0.05             # ... and the prediction has a defined hue
HUE_MIN_PIXELS = 500                # fewer masked pixels in a view -> NaN
# fixed views for re-score example images (the Rung 0 qualitative figures)
RESCORE_EXAMPLES = {"bike": "29", "shrub": "26"}


@contextlib.contextmanager
def rng_guard():
    """Save and restore every RNG the training loop depends on."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@contextlib.contextmanager
def eval_mode(*modules):
    """Put modules in eval mode and restore their previous mode afterwards."""
    was_training = [m.training for m in modules]
    for m in modules:
        m.eval()
    try:
        yield
    finally:
        for m, t in zip(modules, was_training):
            m.train(t)


def _f(x) -> float:
    return x.item() if torch.is_tensor(x) else float(x)


def nchw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 3, 1, 2)


def psnr(pred, target) -> float:
    return peak_signal_noise_ratio(pred, target, data_range=1.0).item()


def ssim(pred, target) -> float:
    return structural_similarity_index_measure(pred, target, data_range=1.0).item()


def psnr_gt_mean(pred, target) -> float:
    """PSNR after scaling pred to the GT mean brightness ("GT-mean" protocol)."""
    pred = torch.clamp(pred * (target.mean() / pred.mean().clamp_min(1e-8)), 0.0, 1.0)
    return psnr(pred, target)


def to_png_levels(img: torch.Tensor) -> torch.Tensor:
    """The 8-bit values eval() writes to renders/*.png ((x * 255).astype(uint8) truncates), back in [0, 1]."""
    return torch.floor(img.clamp(0, 1) * 255.0) / 255.0


def rgb_to_hsv(img: torch.Tensor):
    """[..., 3] RGB in [0, 1] -> hue (degrees, 0 where undefined), saturation, value."""
    v, arg = img.max(-1)
    delta = v - img.min(-1).values
    s = torch.where(v > 0, delta / v.clamp_min(1e-8), torch.zeros_like(v))
    r, g, b = img.unbind(-1)
    d = delta.clamp_min(1e-8)
    h = torch.where(arg == 0, torch.remainder((g - b) / d, 6.0),
                    torch.where(arg == 1, (b - r) / d + 2.0, (r - g) / d + 4.0)) * 60.0
    return torch.where(delta > 0, h, torch.zeros_like(h)), s, v


def color_metrics(pred: torch.Tensor, gt: torch.Tensor, prefix: str = "") -> Dict[str, float]:
    """Brightness-invariant colour errors of pred vs gt, both [H, W, 3] in [0, 1].
    ang_err: angle between RGB vectors; hue_err_sat: circular HSV hue difference on coloured GT pixels;
    sat_ratio: mean prediction saturation / mean GT saturation (< 1 = washed out)."""
    nan = float("nan")
    pn, gn = pred.norm(dim=-1), gt.norm(dim=-1)
    m = (gn > COLOR_DARK_NORM) & (pn > 1e-6)
    out = {}
    if m.any():
        cos = ((pred * gt).sum(-1)[m] / (pn[m] * gn[m])).clamp(-1, 1)
        ang = torch.rad2deg(torch.arccos(cos))
        out["ang_err_deg"], out["ang_err_p95_deg"] = _f(ang.mean()), _f(torch.quantile(ang, 0.95))
    else:
        out["ang_err_deg"] = out["ang_err_p95_deg"] = nan
    hp, sp, _ = rgb_to_hsv(pred)
    hg, sg, vg = rgb_to_hsv(gt)
    out["sat_ratio"] = _f(sp[m].mean() / sg[m].mean().clamp_min(1e-8)) if m.any() else nan
    coloured = (sg > HUE_GT_SAT_MIN) & (vg > HUE_GT_VAL_MIN)
    mh = coloured & (sp > HUE_PRED_SAT_MIN)
    if int(coloured.sum()) >= HUE_MIN_PIXELS and int(mh.sum()) > 0:
        dh = torch.remainder((hp - hg)[mh].abs(), 360.0)
        out["hue_err_sat_deg"] = _f(torch.minimum(dh, 360.0 - dh).mean())
    else:
        out["hue_err_sat_deg"] = nan
    out["hue_px_frac"] = _f(coloured.float().mean())
    out["hue_excluded_frac"] = _f(1.0 - mh.sum() / coloured.sum().clamp_min(1))
    return {prefix + k: v for k, v in out.items()}


@contextlib.contextmanager
def swapped_splats(runner, splats_state: Dict[str, torch.Tensor]):
    """Render with another run's Gaussians (e.g. a shared reference depth), then restore."""
    saved = runner.splats
    runner.splats = torch.nn.ParameterDict(
        {k: torch.nn.Parameter(v.to(runner.device), requires_grad=False) for k, v in splats_state.items()})
    try:
        yield
    finally:
        runner.splats = saved


def final_ckpt_path(run_dir: Path) -> Path:
    cfg = json.loads((run_dir / "cfg.json").read_text())
    return run_dir / "ckpts" / f"ckpt_{cfg['max_steps'] - 1}.pt"


def to_uint8(img: torch.Tensor) -> np.ndarray:
    """[H, W, C] or [H, W] float tensor in [0, 1] -> uint8 HxWx3 array."""
    img = img.detach().float().clamp(0, 1).cpu().numpy()
    if img.ndim == 2:
        img = img[..., None]
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    return (img * 255.0 + 0.5).astype(np.uint8)


def load_train_gt(parser, index: int) -> np.ndarray:
    """Normal-light image for a training view, undistorted and cropped exactly as
    ``datasets.colmap.Dataset.__getitem__`` does for the dark image."""
    low_path = Path(parser.image_paths[index])
    gt_path = low_path.parent.parent / "high" / low_path.name
    if not gt_path.exists():
        for alt in (gt_path.with_suffix(gt_path.suffix.upper()), gt_path.with_suffix(gt_path.suffix.lower())):
            if alt.exists():
                gt_path = alt
                break
    image = imageio.imread(gt_path)[..., :3]
    camera_id = parser.camera_ids[index]
    if len(parser.params_dict[camera_id]) > 0:
        mapx, mapy = parser.mapx_dict[camera_id], parser.mapy_dict[camera_id]
        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = parser.roi_undist_dict[camera_id]
        image = image[y : y + h, x : x + w]
    return image


class Diagnostics:
    def __init__(self, runner, out_dir=None, loss_log: bool = True):
        self.runner = runner
        self.cfg = runner.cfg
        self.device = runner.device
        self.out_dir = Path(out_dir) if out_dir is not None else Path(self.cfg.result_dir) / "diagnostics"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.hist = HistogramPriorLoss()
        self.timing: Dict[str, List[float]] = {}

        trainset = runner.trainset
        self.view_index = [int(trainset.indices[i]) for i in range(len(trainset))]
        self.view_names = [runner.parser.image_names[k] for k in self.view_index]

        self._loss_file = None
        if loss_log:
            self._loss_file = open(self.out_dir / "loss_log.csv", "w", newline="")
            self._loss_csv = csv.DictWriter(self._loss_file, fieldnames=LOSS_FIELDS)
            self._loss_csv.writeheader()

    # ------------------------------------------------------------------ utils
    def _view(self, i: int):
        """Training view i as the training loop sees it (batch of 1, on device)."""
        data = self.runner.trainset[i]
        return (
            (data["image"][None].to(self.device) / 255.0),          # pixels [1, H, W, 3]
            data["camtoworld"][None].to(self.device),               # [1, 4, 4]
            data["K"][None].to(self.device),                        # [1, 3, 3]
            torch.tensor([data["image_id"]], device=self.device),   # [1]
        )

    def _curve_modules(self):
        return self.runner.curve_adjust, self.runner.curve_adjust_gamma

    def _time(self, key: str, seconds: float):
        self.timing.setdefault(key, []).append(seconds)

    # ------------------------------------------------------------ loss log
    def log_losses(self, step: int, total, loss_regress_low, loss_regress_enh, loss_co, hist_loss, hist_parts):
        cfg = self.cfg
        if not (cfg.tb_every > 0 and step % cfg.tb_every == 0):
            return
        row = {
            "step": step,
            "total": _f(total),
            "loss_regress_low": _f(loss_regress_low),
            "loss_regress_enh": _f(loss_regress_enh),
            "loss_co": _f(loss_co),
            "hist_loss": _f(hist_loss),
            "hist_cdf": _f(hist_parts["cdf"]),
            "hist_pseudo": _f(hist_parts["pseudo"]),
            "hist_smooth": _f(hist_parts["smooth"]),
            "w_cdf": _f(hist_parts["w_cdf"]),
        }
        self._loss_csv.writerow(row)
        self._loss_file.flush()
        for k, v in row.items():
            if k != "step":
                self.runner.writer.add_scalar(f"diag/{k}", v, step)

    # ------------------------------------------------------ curve snapshot
    @torch.no_grad()
    def snapshot_curves(self, step: int):
        """Recompute every training view's curve exactly as the training step does."""
        tic = time.time()
        runner = self.runner
        keys = ["curve_bias", "curve_preclamp", "curve_adj", "pesdo_curve", "cdf_target", "bin_hist",
                "gamma", "alpha", "beta", "axis1", "axis2", "mean_brightness"]
        acc = {k: [] for k in keys}
        with rng_guard(), eval_mode(*self._curve_modules()):
            for i in range(len(runner.trainset)):
                pixels, c2w, _, ids = self._view(i)
                out = runner.curve_forward(pixels, c2w, ids)
                n_bins = out["curve_adj"].shape[-1]
                # same truncation as tools.LUT_mapping
                idx = torch.cat([(torch.clamp(t, 0, 1) * (n_bins - 1)).to(torch.int32).long().flatten()
                                 for t in out["projected"]])
                acc["bin_hist"].append(torch.bincount(idx, minlength=n_bins).cpu().numpy())
                acc["cdf_target"].append(self.hist.compute_histogram_equalization(pixels)[0].cpu().numpy())
                for k in ("curve_bias", "curve_preclamp", "curve_adj", "pesdo_curve"):
                    acc[k].append(out[k][0].cpu().numpy())
                for k in ("gamma", "alpha", "beta"):
                    acc[k].append(_f(out[k][0]))
                acc["axis1"].append(runner.axis1_para[i].detach().cpu().numpy())
                acc["axis2"].append(runner.axis2_para[i].detach().cpu().numpy())
                acc["mean_brightness"].append(_f(pixels.mean()))
            curve_global = runner.curve[0].detach().cpu().numpy()
        np.savez_compressed(
            self.out_dir / f"curves_step{step:05d}.npz",
            step=step,
            view_names=np.array(self.view_names),
            curve_global=curve_global,
            **{k: np.stack(v) if isinstance(v[0], np.ndarray) else np.array(v) for k, v in acc.items()},
        )
        self._time("snapshot_curves", time.time() - tic)

    # ------------------------------------------------------ final snapshot
    @torch.no_grad()
    def final_snapshot(self, step: int):
        """Pseudo-target quality and cross-view consistency at the end of training."""
        tic = time.time()
        with rng_guard(), eval_mode(*self._curve_modules()):
            views = self._score_views(step)
            self._consistency(views)
            self._save_curve_state(step)
        self._time("final_snapshot", time.time() - tic)
        self.close()

    def _render(self, c2w, K, width: int, height: int):
        cfg = self.cfg
        return self.runner.rasterize_splats(
            camtoworlds=c2w, Ks=K, width=width, height=height, sh_degree=cfg.sh_degree,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane, render_mode="RGB+ED",
        )

    def _score_views(self, step: int, ref_splats=None, extended: bool = False):
        """Training-view pseudo-targets and renders vs GT. ref_splats: take depth and alpha for the
        warp from these Gaussians instead of the run's own. extended (re-scoring): add colour metrics
        and the zero-basis counterfactual pseudo-target, and save only the fixed example views."""
        runner, cfg = self.runner, self.cfg
        n = len(runner.trainset)
        if extended:
            stem = RESCORE_EXAMPLES.get(Path(cfg.data_dir).name)
            examples = [i for i in range(n) if Path(self.view_names[i]).stem == stem]
        else:
            examples = sorted({0, n // 2, n - 1})[:N_EXAMPLE_VIEWS]
        views, rows = [], []
        for i in range(n):
            pixels, c2w, K, ids = self._view(i)
            height, width = pixels.shape[1:3]
            pseudo_raw = runner.curve_forward(pixels, c2w, ids)["pixels_enh"]
            renders_enh, renders_low, alphas_enh, _, _ = self._render(c2w, K, width, height)
            render = torch.clamp(renders_enh[..., :3], 0.0, 1.0)
            render_low = renders_low[..., :3]
            depth = renders_enh[..., 3]                                     # [1, H, W], camera-space z
            if ref_splats is not None:
                with swapped_splats(runner, ref_splats):
                    ref_enh, _, alphas_enh, _, _ = self._render(c2w, K, width, height)
                depth = ref_enh[..., 3]
            pseudo = torch.clamp(pseudo_raw, 0.0, 1.0)
            gt = torch.from_numpy(load_train_gt(runner.parser, self.view_index[i])).float().to(self.device)[None] / 255.0
            assert gt.shape == pixels.shape, f"GT {tuple(gt.shape)} vs input {tuple(pixels.shape)}"

            row = {
                "view": self.view_names[i],
                "mean_input": _f(pixels.mean()),
                "mean_pseudo": _f(pseudo_raw.mean()),
                "mean_render": _f(render.mean()),
                "mean_gt": _f(gt.mean()),
                "pseudo_out_of_range_frac": _f(((pseudo_raw < 0) | (pseudo_raw > 1)).float().mean()),
                "fit_l1_render_vs_pseudo": _f((render - pseudo).abs().mean()),
                "fit_psnr_render_vs_pseudo": psnr(nchw(render), nchw(pseudo)),
                "fit_psnr_lowrender_vs_input": psnr(nchw(render_low.clamp(0, 1)), nchw(pixels)),
                "pseudo_psnr": psnr(nchw(pseudo), nchw(gt)),
                "pseudo_ssim": ssim(nchw(pseudo), nchw(gt)),
                "pseudo_psnr_gtmean": psnr_gt_mean(nchw(pseudo), nchw(gt)),
                "render_psnr": psnr(nchw(render), nchw(gt)),
                "render_ssim": ssim(nchw(render), nchw(gt)),
                "render_psnr_gtmean": psnr_gt_mean(nchw(render), nchw(gt)),
            }
            view = {"pseudo": pseudo[0], "render": render[0], "gt": gt[0], "depth": depth[0],
                    "alpha": alphas_enh[0, ..., 0], "K": K[0], "c2w": c2w[0]}
            if extended:
                # counterfactual: the same curve with the colour basis reset to RGB
                pseudo0 = torch.clamp(runner.curve_forward(pixels, c2w, ids, zero_basis=True)["pixels_enh"], 0.0, 1.0)
                view["pseudo0"] = pseudo0[0]
                row.update({
                    "pseudo0_psnr": psnr(nchw(pseudo0), nchw(gt)),
                    "pseudo0_psnr_gtmean": psnr_gt_mean(nchw(pseudo0), nchw(gt)),
                    "pseudo0_vs_pseudo_l1": _f((pseudo0 - pseudo).abs().mean()),
                    "pseudo0_vs_pseudo_maxabs": _f((pseudo0 - pseudo).abs().max()),
                    **color_metrics(pseudo[0], gt[0], "pseudo_"),
                    **color_metrics(pseudo0[0], gt[0], "pseudo0_"),
                    **color_metrics(render[0], gt[0], "render_"),
                })
            rows.append(row)
            if i in examples:
                stem = Path(self.view_names[i]).stem
                canvas = np.concatenate([to_uint8(pixels[0]), to_uint8(pseudo[0]), to_uint8(render[0]), to_uint8(gt[0])], axis=1)
                imageio.imwrite(self.out_dir / f"example_{stem}_in_pseudo_render_gt.png", canvas)
                if extended:
                    diff = ((pseudo0 - pseudo).abs() * 4.0)[0]
                    canvas = np.concatenate([to_uint8(pseudo[0]), to_uint8(pseudo0[0]), to_uint8(diff)], axis=1)
                    imageio.imwrite(self.out_dir / f"example_{stem}_pseudo_zerobasis_diffx4.png", canvas)
            views.append(view)

        with open(self.out_dir / "pseudo_targets.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return views

    def _consistency(self, views, keys=("pseudo", "render", "gt")):
        centres = torch.stack([v["c2w"][:3, 3] for v in views])
        dist = torch.cdist(centres, centres)
        dist.fill_diagonal_(float("inf"))
        rows = []
        for i in range(len(views)):
            for j in torch.argsort(dist[i])[:N_NEIGHBOURS].tolist():
                warped, valid = self._warp(views[i], views[j], list(keys))
                row = {"view_i": self.view_names[i], "view_j": self.view_names[j],
                       "cam_dist": _f(dist[i, j]), "valid_frac": _f(valid.float().mean())}
                for key in keys:
                    diff = (views[i][key] - warped[key]).abs().mean(-1)
                    l1 = diff[valid].mean() if valid.any() else torch.tensor(float("nan"))
                    row[f"l1_{key}"] = _f(l1)
                    # L1 scales with brightness; relative L1 makes dark and bright image sets comparable
                    row[f"rel_l1_{key}"] = _f(l1 / views[i][key].mean(-1)[valid].mean().clamp_min(1e-8)) if valid.any() else float("nan")
                rows.append(row)
                if i == 0 and not (self.out_dir / "warp_check.png").exists():
                    self._save_warp_check(views[i], warped, valid)
        with open(self.out_dir / "consistency.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    def _warp(self, vi, vj, keys):
        """Warp images of view j into view i using view i's rendered depth.
        gsplat convention: camtoworld is OpenCV (x right, y down, z forward) and
        pixel (u, v) has its centre at (u + 0.5, v + 0.5)."""
        H, W = vi["depth"].shape
        dev = vi["depth"].device
        v, u = torch.meshgrid(torch.arange(H, device=dev, dtype=torch.float32),
                              torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
        pix = torch.stack([u + 0.5, v + 0.5, torch.ones_like(u)], -1).reshape(-1, 3)   # [HW, 3]
        z_i = vi["depth"].reshape(-1)
        x_cam_i = (torch.linalg.inv(vi["K"]) @ pix.T).T * z_i[:, None]
        x_world = x_cam_i @ vi["c2w"][:3, :3].T + vi["c2w"][:3, 3]
        w2c_j = torch.linalg.inv(vj["c2w"])
        x_cam_j = x_world @ w2c_j[:3, :3].T + w2c_j[:3, 3]
        z_j = x_cam_j[:, 2]
        proj = x_cam_j @ vj["K"].T
        u_j = proj[:, 0] / proj[:, 2].clamp_min(1e-8)
        v_j = proj[:, 1] / proj[:, 2].clamp_min(1e-8)
        # align_corners=False: normalised -1/+1 are the outer image edges
        grid = torch.stack([u_j / W * 2 - 1, v_j / H * 2 - 1], -1).reshape(1, H, W, 2)

        def sample(img):  # img [H, W, C] or [H, W]
            x = img[..., None] if img.dim() == 2 else img
            out = F.grid_sample(x.permute(2, 0, 1)[None], grid, mode="bilinear",
                                padding_mode="border", align_corners=False)
            return out[0].permute(1, 2, 0)

        depth_j = sample(vj["depth"])[..., 0].reshape(-1)
        alpha_j = sample(vj["alpha"])[..., 0].reshape(-1)
        inside = (u_j >= 0.5) & (u_j <= W - 0.5) & (v_j >= 0.5) & (v_j <= H - 0.5) & (z_j > 1e-6)
        visible = (z_j - depth_j).abs() < DEPTH_REL_TOL * depth_j.clamp_min(1e-6)
        valid = inside & visible & (vi["alpha"].reshape(-1) > ALPHA_MIN) & (alpha_j > ALPHA_MIN)
        return {k: sample(vj[k]) for k in keys}, valid.reshape(H, W)

    def _save_warp_check(self, vi, warped, valid):
        rows = []
        for key in ("gt", "pseudo"):
            diff = (vi[key] - warped[key]).abs() * 4.0 * valid[..., None]
            rows.append(np.concatenate([to_uint8(vi[key]), to_uint8(warped[key] * valid[..., None]),
                                        to_uint8(diff), to_uint8(valid.float())], axis=1))
        imageio.imwrite(self.out_dir / "warp_check.png", np.concatenate(rows, axis=0))

    def _save_curve_state(self, step: int):
        runner = self.runner
        torch.save({
            "step": step,
            "curve": runner.curve.detach().cpu(),
            "curve_adjust": runner.curve_adjust.state_dict(),
            "curve_adjust_gamma": runner.curve_adjust_gamma.state_dict(),
            "axis1": torch.stack([p.detach().cpu() for p in runner.axis1_para]),
            "axis2": torch.stack([p.detach().cpu() for p in runner.axis2_para]),
        }, self.out_dir / "curve_state.pt")

    # ------------------------------------------------------------ re-score
    def _load_run_state(self, run_dir: Path) -> str:
        """Load a finished run's splats (final ckpt) and curve state (diagnostics/curve_state.pt)."""
        runner, cfg = self.runner, self.cfg
        run_cfg = json.loads((run_dir / "cfg.json").read_text())
        for k in ("data_dir", "exp_name", "data_factor", "test_every"):
            assert run_cfg[k] == getattr(cfg, k), f"{k}: run has {run_cfg[k]!r}, re-score got {getattr(cfg, k)!r}"
        ckpt_path = final_ckpt_path(run_dir)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]
        st = torch.load(run_dir / "diagnostics" / "curve_state.pt", map_location=self.device)
        assert st["axis1"].shape[0] == len(runner.axis1_para), "curve_state.pt has a different number of views"
        runner.curve.data.copy_(st["curve"].to(self.device))
        runner.curve_adjust.load_state_dict(st["curve_adjust"])
        runner.curve_adjust_gamma.load_state_dict(st["curve_adjust_gamma"])
        for p, a in zip(runner.axis1_para, st["axis1"]):
            p.data.copy_(a.to(self.device))
        for p, a in zip(runner.axis2_para, st["axis2"]):
            p.data.copy_(a.to(self.device))
        return str(ckpt_path)

    @torch.no_grad()
    def rescore(self, run_dir, ref_dir=None):
        """Re-score a finished run without training: training-view pseudo-targets and renders with colour
        metrics and the zero-basis counterfactual, cross-view consistency (warp with ref_dir's depth if
        given), and validation metrics. Writes pseudo_targets.csv, consistency.csv, val_metrics.csv and
        val_summary.json to out_dir."""
        tic = time.time()
        run_dir = Path(run_dir)
        ckpt_path = self._load_run_state(run_dir)
        ref_splats = None
        if ref_dir is not None:
            ref_splats = torch.load(final_ckpt_path(Path(ref_dir)), map_location=self.device)["splats"]
        with rng_guard(), eval_mode(*self._curve_modules()):
            views = self._score_views(int(json.loads((run_dir / "cfg.json").read_text())["max_steps"]),
                                      ref_splats=ref_splats, extended=True)
            self._consistency(views, keys=("pseudo", "pseudo0", "render", "gt"))
            del views
            val_rows = self._score_val()
        summary = {k: float(np.nanmean([r[k] for r in val_rows])) for k in val_rows[0] if k != "view"}
        summary.update({"n_views": len(val_rows), "run": str(run_dir), "ckpt": ckpt_path,
                        "ref_depth_run": None if ref_dir is None else str(ref_dir), "seconds": time.time() - tic})
        with open(self.out_dir / "val_summary.json", "w") as f:
            json.dump(summary, f, indent=1)
        self._time("rescore", time.time() - tic)
        print(f"[rescore] {run_dir.name}: val PSNR {summary['psnr']:.3f}, PSNR gt-mean {summary['psnr_gtmean']:.3f}, "
              f"LPIPS {summary['lpips']:.3f}, hue err {summary['hue_err_sat_deg']:.2f} deg ({summary['seconds']:.0f} s)")
        self.close()

    def _score_val(self):
        """Validation views rendered as Runner.eval() does, against the normal-light validation images."""
        runner = self.runner
        valset = runner.valset
        rows = []
        for i in range(len(valset)):
            data = valset[i]
            gt = data["image"][None].to(self.device) / 255.0
            c2w, K = data["camtoworld"][None].to(self.device), data["K"][None].to(self.device)
            height, width = gt.shape[1:3]
            colors_enh, _, _, _, _ = self._render(c2w, K, width, height)
            pred = torch.clamp(colors_enh[..., :3], 0.0, 1.0)
            p, g = nchw(pred), nchw(gt)
            png = nchw(to_png_levels(pred)).double()
            rows.append({
                "view": runner.parser.image_names[int(valset.indices[i])],
                "psnr": runner.psnr(p, g).item(),
                "ssim": runner.ssim(p, g).item(),
                "lpips": runner.lpips(p, g).item(),
                "psnr_gtmean": psnr_gt_mean(p, g),
                # same 8-bit truncation and float64 maths as rung0_analyze.py, for checking against Rung 0
                "psnr_png": psnr(png, g.double()),
                "psnr_gtmean_png": psnr_gt_mean(png, g.double()),
                "mean_render": _f(pred.mean()),
                "mean_gt": _f(gt.mean()),
                **color_metrics(pred[0], gt[0]),
            })
        with open(self.out_dir / "val_metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return rows

    def close(self):
        if self._loss_file is not None and not self._loss_file.closed:
            self._loss_file.close()
        summary = {k: {"calls": len(v), "total_s": sum(v), "mean_s": sum(v) / len(v)} for k, v in self.timing.items()}
        with open(self.out_dir / "timing.json", "w") as f:
            json.dump(summary, f, indent=2)
