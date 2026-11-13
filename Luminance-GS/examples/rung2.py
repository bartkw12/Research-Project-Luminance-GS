"""Rung 2: one shared monotone tone curve F plus one exposure s_k per training view,

    L_k(x) = F(min(1, 2^{s_k} x)),   applied to R, G and B,

in place of the official 255-entry curve, its bias network, its shape-prior network and the per-view colour
basis (Implementation_guides/M2_rung2_plan.md, section 2.2). Torch and numpy only: the trainer imports this
module (M2 part C), never the other way round. Tests: test_rung2.py. Preview without 3DGS: rung2_preview.py.

Conventions
- Knots y_0 = 0 < ... < y_K = 1. Heights h = [0, c / c_K] with c = cumsum(softplus(theta)), so F(0) = 0,
  F(1) = 1 and F is strictly increasing by construction (up to float32: a theta far below -15 gives a segment
  too flat to resolve). Between knots F is linear, and differentiable in its input and in h, so gradient reaches
  y and therefore s_k (the official truncating LUT gives no gradient to pixel values).
- The official curve grid is x_b = b / 254, b = 0..254 (torch.linspace(0, 1, 255)). composed_curve(k) samples
  L_k there, so the curve loss, the diagnostics and the M1 analysis code read Rung 2 like the official curve.
- Curve-loss target (hist_target; decided by Bart 2026-11-11): the HE-CDF of
  losses.HistogramPriorLoss.compute_histogram_equalization, shifted one bin, T_k = [0, CDF_k[0], ..., CDF_k[253]].
  The official LUT truncates, so input level l >= 1 reads curve[l - 1] ~ CDF[l - 1]; an interpolated fit to the
  unshifted CDF would map level l to about CDF[l], one bin brighter, which lifts the darkest shadows most.
  With the shift both read the same bin, level 0 stays black, and F(0) = 0 agrees with the target at b = 0.
- 2^s is computed as exp(ln 2 * s), never torch.exp2: torch 1.13 compiles exp2's CUDA kernel at run time
  (NVRTC), which fails in this env (nvrtc-builtins64_117.dll missing).
"""

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as nnf
from torch import Tensor

LN2 = math.log(2.0)
N_BINS = 255          # length of the official curve; grid x_b = b / (N_BINS - 1)
INC_FLOOR = 1e-4      # smallest initial increment of F between two knots, before renormalising
SPACINGS = ("quadratic", "uniform", "usedrange")
F_INITS = ("cdf", "identity")
ATTACH = ("none", "s", "all")
ALIGNS = ("shift", "none")

Index = Union[int, Tensor]


# ------------------------------------------------------------------ knots, target, initialisation
def make_knots(K: int, spacing: str = "quadratic", y_max: Optional[float] = None) -> Tensor:
    """[K+1] float32 knots with y_0 = 0, y_K = 1, strictly increasing.
    quadratic: (j/K)^2, dense near black. uniform: j/K.
    usedrange: y_max * (j/(K-1))^2 for j < K and y_K = 1, so K knots cover [0, y_max], the range the photos'
    gained pixels actually use (y_max from used_range_max); the last segment spans (y_max, 1]."""
    if K < 2:
        raise ValueError(f"need K >= 2 segments, got {K}")
    j = np.arange(K + 1, dtype=np.float64)
    if spacing == "quadratic":
        y = (j / K) ** 2
    elif spacing == "uniform":
        y = j / K
    elif spacing == "usedrange":
        if y_max is None or not 0.0 < y_max < 1.0:
            raise ValueError(f"usedrange knots need 0 < y_max < 1, got {y_max}")
        y = np.append(y_max * (j[:K] / (K - 1)) ** 2, 1.0)
    else:
        raise ValueError(f"unknown knot spacing {spacing!r}; expected one of {SPACINGS}")
    return torch.tensor(y, dtype=torch.float32)


def used_range_max(images: Sequence[Tensor], s_init: Tensor, q: float = 0.999, cap: float = 0.99) -> float:
    """y_max for usedrange knots: the q-quantile of min(1, 2^{s_k} x) over every pixel and channel of every
    training view (images in [0, 1] holding 8-bit levels, any shape, one per view, in s_init's order). Exact,
    from per-view level counts. Capped at `cap` so the last segment (y_max, 1] is never empty."""
    vals, wts = [], []
    for img, s in zip(images, s_init.detach().double().cpu().tolist()):
        lv = np.clip(np.rint(img.detach().float().cpu().numpy().ravel() * 255.0), 0, 255).astype(np.int64)
        wts.append(np.bincount(lv, minlength=256).astype(np.float64))
        vals.append(np.minimum(1.0, 2.0 ** s * np.arange(256) / 255.0))
    v, w = np.concatenate(vals), np.concatenate(wts)
    order = np.argsort(v, kind="stable")
    v, cw = v[order], np.cumsum(w[order]) / w.sum()
    i = min(int(np.searchsorted(cw, q)), len(v) - 1)  # smallest value whose cumulative share reaches q
    return float(min(v[i], cap))


def hist_target(cdf: Tensor, align: str = "shift") -> Tensor:
    """[B, 255] curve-loss target from the HE-CDF [B, 255]. "shift" (the default; see the module docstring):
    [0, CDF[0], ..., CDF[253]]. "none": the CDF itself (the plan's first draft; used by the preview's
    alignment check only)."""
    if align == "shift":
        return torch.cat([torch.zeros_like(cdf[:, :1]), cdf[:, :-1]], dim=1)
    if align == "none":
        return cdf
    raise ValueError(f"unknown alignment {align!r}; expected one of {ALIGNS}")


def init_heights(mode: str, knots: Tensor, targets: Optional[Tensor] = None,
                 s_init: Optional[Tensor] = None) -> Tensor:
    """[K+1] float32 initial heights of F at the knots.
    identity: F(y) = y.
    cdf: every view's target (from hist_target) aligned by its initial exposure, T_k(2^{-s_k} y), taken as 1
    beyond the photo's range (2^{-s_k} y > 1), averaged over views. This inverts the model T_k(x) = F(2^{s_k} x).
    Then h_0 is pinned to 0, the increments are floored at INC_FLOOR and renormalised, so F(0) = 0, F(1) = 1
    and F is strictly increasing."""
    y = knots.detach().double().cpu().numpy()
    if mode == "identity":
        h = y.copy()
    elif mode == "cdf":
        if targets is None or s_init is None:
            raise ValueError("the cdf init needs the per-view targets and initial exposures")
        T = targets.detach().double().cpu().numpy()
        s = s_init.detach().double().cpu().numpy()
        if T.shape[0] != s.shape[0]:
            raise ValueError(f"{T.shape[0]} targets but {s.shape[0]} exposures")
        x = np.arange(T.shape[1]) / (T.shape[1] - 1)
        h = np.mean([np.interp(2.0 ** (-s_k) * y, x, T_k, right=1.0) for T_k, s_k in zip(T, s)], axis=0)
    else:
        raise ValueError(f"unknown F init {mode!r}; expected one of {F_INITS}")
    d = np.maximum(np.diff(np.concatenate([[0.0], h[1:]])), INC_FLOOR)
    return torch.tensor(np.concatenate([[0.0], np.cumsum(d) / d.sum()]), dtype=torch.float32)


def theta_from_heights(h: Tensor) -> Tensor:
    """Inverse of ExposureCurve.heights: theta_j = log(expm1(K d_j)), with d_j the increments of h (sum 1), so
    softplus(theta_j) = K d_j and theta is of order 1."""
    h = h.detach().double().cpu()
    d = torch.diff(h)
    if abs(float(h[0])) > 1e-6 or abs(float(h[-1]) - 1.0) > 1e-6 or not bool((d > 0).all()):
        raise ValueError("heights must start at 0, end at 1 and increase strictly")
    d = d / d.sum()
    return torch.log(torch.expm1(d * d.numel())).float()


def interp(yq: Tensor, knots: Tensor, h: Tensor) -> Tensor:
    """Piecewise-linear F at yq in [0, 1] (any shape): F(y_j) = h_j. Differentiable in yq and in h.
    knots and h are [K+1] with the dtype of yq."""
    j = torch.searchsorted(knots[1:-1].contiguous(), yq.contiguous(), right=True)  # segment index, 0..K-1
    y0, y1 = knots[j], knots[j + 1]
    h0, h1 = h[j], h[j + 1]
    return h0 + (h1 - h0) * (yq - y0) / (y1 - y0)


# ------------------------------------------------------------------ the model
class ExposureCurve(torch.nn.Module):
    """Rung 2: L_k(x) = F(min(1, 2^{s_k} x)); one shared monotone F, one exposure s_k per training view.
    Parameters: theta [K] (F's increments, through softplus) and s [N]. Buffers: knots [K+1], the official
    grid [255] and s_init [N] (kept for the diagnostics)."""

    def __init__(self, knots: Tensor, h_init: Tensor, s_init: Tensor):
        super().__init__()
        if h_init.numel() != knots.numel():
            raise ValueError(f"{h_init.numel()} heights for {knots.numel()} knots")
        self.register_buffer("knots", knots.detach().clone().float())
        self.register_buffer("grid", torch.linspace(0, 1, N_BINS))
        self.register_buffer("s_init", s_init.detach().clone().float().reshape(-1))
        self.theta = torch.nn.Parameter(theta_from_heights(h_init).to(self.knots.device))
        self.s = torch.nn.Parameter(self.s_init.clone())

    @property
    def K(self) -> int:
        return self.theta.numel()

    @property
    def N(self) -> int:
        return self.s.numel()

    def n_params(self) -> int:
        return self.K + self.N

    def heights(self) -> Tensor:
        """[K+1]: F at the knots, [0, c / c_K] with c = cumsum(softplus(theta))."""
        c = torch.cumsum(nnf.softplus(self.theta), 0)
        return torch.cat([c.new_zeros(1), c / c[-1]])

    def F(self, y: Tensor) -> Tensor:
        """The shared curve at y in [0, 1] (any shape); gradient to theta."""
        return interp(y, self.knots, self.heights())

    def gain(self, k: Index) -> Tensor:
        """[B] 2^{s_k} for an int or a [B] index tensor; gradient to s."""
        return torch.exp(LN2 * self.s[k].reshape(-1))

    def composed_curve(self, k: Index) -> Tensor:
        """[B, 255]: L_k on the official grid, F(min(1, 2^{s_k} x_b)); gradient to theta and s. The curve loss
        uses this; it is also what the diagnostics store as curve_adj."""
        y = torch.clamp(self.gain(k)[:, None] * self.grid[None], max=1.0)
        return interp(y, self.knots, self.heights())

    def map_pixels(self, pixels: Tensor, k: Index, attach: str = "none") -> Tensor:
        """[B, H, W, 3] pseudo-target L_k(pixels), per channel. attach:
        none: no gradient (the default, detached mode);
        s:    gradient to s_k only; F's heights are detached, y = 2^{s_k} x stays differentiable, so
              d L / d s_k = F'(y) y ln 2 (0 where 2^{s_k} x >= 1);
        all:  gradient to theta and s."""
        if attach not in ATTACH:
            raise ValueError(f"unknown attach {attach!r}; expected one of {ATTACH}")
        if attach == "none":
            with torch.no_grad():
                return self._map(pixels, k, self.heights())
        h = self.heights()
        return self._map(pixels, k, h if attach == "all" else h.detach())

    def _map(self, pixels: Tensor, k: Index, h: Tensor) -> Tensor:
        y = torch.clamp(self.gain(k).view(-1, 1, 1, 1) * pixels, max=1.0)
        return interp(y, self.knots, h)

    @torch.no_grad()
    def project_gauge(self):
        """Only relative exposure is identifiable: keep mean s = 0, so F carries the overall brightening."""
        self.s.sub_(self.s.mean())

    @torch.no_grad()
    def summary(self) -> Dict[str, object]:
        """Plain numbers for json (the diagnostics' rung2_state.json)."""
        return {"K": self.K, "N": self.N, "n_params": self.n_params(),
                "knots": self.knots.cpu().tolist(), "heights": self.heights().cpu().tolist(),
                "s_init": self.s_init.cpu().tolist(), "s_final": self.s.cpu().tolist(),
                "mean_s": float(self.s.mean())}


# ------------------------------------------------------------------ the curve loss
def curve_weight(step: int, exp_name: str = "") -> float:
    """w(t) of the official HistogramPriorLoss: 1 before step 3000, then 0.5 (0.1 for over_exp)."""
    if step < 3000:
        return 1.0
    return 0.1 if exp_name == "over_exp" else 0.5


def curve_loss(curve: Tensor, target: Tensor, step: int, exp_name: str = "",
               smooth_lambda: float = 0.1) -> Tuple[Tensor, Dict[str, object]]:
    """L_curve = w(t) mean (curve - target)^2 + smooth_lambda mean (curve[b+1] - curve[b])^2, on [B, 255].
    The official L_hist (losses.HistogramPriorLoss.forward) minus its shape-prior term. parts has the keys
    HistogramPriorLoss returns ("pseudo" = NaN: no shape prior), so Diagnostics.log_losses works unchanged."""
    cdf_term = torch.mean((curve - target) ** 2)
    smooth = torch.mean((curve[:, 1:] - curve[:, :-1]) ** 2)
    w = curve_weight(step, exp_name)
    total = w * cdf_term + smooth_lambda * smooth
    return total, {"cdf": cdf_term, "pseudo": float("nan"), "smooth": smooth, "w_cdf": w}
