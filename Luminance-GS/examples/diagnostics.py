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
    def __init__(self, runner):
        self.runner = runner
        self.cfg = runner.cfg
        self.device = runner.device
        self.out_dir = Path(self.cfg.result_dir) / "diagnostics"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.hist = HistogramPriorLoss()
        self.timing: Dict[str, List[float]] = {}

        trainset = runner.trainset
        self.view_index = [int(trainset.indices[i]) for i in range(len(trainset))]
        self.view_names = [runner.parser.image_names[k] for k in self.view_index]

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

    def _score_views(self, step: int):
        runner, cfg = self.runner, self.cfg
        n = len(runner.trainset)
        examples = sorted({0, n // 2, n - 1})[:N_EXAMPLE_VIEWS]
        views, rows = [], []
        for i in range(n):
            pixels, c2w, K, ids = self._view(i)
            height, width = pixels.shape[1:3]
            pseudo_raw = runner.curve_forward(pixels, c2w, ids)["pixels_enh"]
            renders_enh, renders_low, alphas_enh, _, _ = runner.rasterize_splats(
                camtoworlds=c2w, Ks=K, width=width, height=height, sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane, far_plane=cfg.far_plane, render_mode="RGB+ED",
            )
            render = torch.clamp(renders_enh[..., :3], 0.0, 1.0)
            render_low = renders_low[..., :3]
            depth = renders_enh[..., 3]                                     # [1, H, W], camera-space z
            pseudo = torch.clamp(pseudo_raw, 0.0, 1.0)
            gt = torch.from_numpy(load_train_gt(runner.parser, self.view_index[i])).float().to(self.device)[None] / 255.0
            assert gt.shape == pixels.shape, f"GT {tuple(gt.shape)} vs input {tuple(pixels.shape)}"

            rows.append({
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
            })
            if i in examples:
                canvas = np.concatenate([to_uint8(pixels[0]), to_uint8(pseudo[0]), to_uint8(render[0]), to_uint8(gt[0])], axis=1)
                imageio.imwrite(self.out_dir / f"example_{Path(self.view_names[i]).stem}_in_pseudo_render_gt.png", canvas)
            views.append({"pseudo": pseudo[0], "render": render[0], "gt": gt[0], "depth": depth[0],
                          "alpha": alphas_enh[0, ..., 0], "K": K[0], "c2w": c2w[0]})

        with open(self.out_dir / "pseudo_targets.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return views

    def _consistency(self, views):
        centres = torch.stack([v["c2w"][:3, 3] for v in views])
        dist = torch.cdist(centres, centres)
        dist.fill_diagonal_(float("inf"))
        rows = []
        for i in range(len(views)):
            for j in torch.argsort(dist[i])[:N_NEIGHBOURS].tolist():
                warped, valid = self._warp(views[i], views[j], ["pseudo", "render", "gt"])
                row = {"view_i": self.view_names[i], "view_j": self.view_names[j],
                       "cam_dist": _f(dist[i, j]), "valid_frac": _f(valid.float().mean())}
                for key in ("pseudo", "render", "gt"):
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

    def close(self):
        if not self._loss_file.closed:
            self._loss_file.close()
        summary = {k: {"calls": len(v), "total_s": sum(v), "mean_s": sum(v) / len(v)} for k, v in self.timing.items()}
        with open(self.out_dir / "timing.json", "w") as f:
            json.dump(summary, f, indent=2)
