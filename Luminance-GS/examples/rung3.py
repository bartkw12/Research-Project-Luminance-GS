"""Rung 3 (M6): two shared monotone curves mixed per pixel by fixed brightness-zone weights,

    L_k(x, p) = w_sh(p) F_sh(min(1, 2^{s_k} x)) + w_lit(p) F_lit(min(1, 2^{s_k} x)),   applied to R, G and B,

on Rung 2's knots, with one exposure s_k per training view and one shape-prior gamma shared by both curves
(Implementation_guides/M6_rung3_plan.md section 2; design and readings pre-registered in docs/RESEARCH_LOG.md on
2026-11-26). Torch and numpy only. It imports rung2's helpers and never changes rung2.py; the trainer imports this
module with --rung3_zones 2 (--rung3_zones 1 runs Rung 2's own code). Tests: test_rung3.py.

Conventions
- Zones m = 0 (shadow) and m = 1 (lit). With M = 1 (one zone, weight 1 everywhere) the model, its pseudo-target and
  its curve loss equal rung2.ExposureCurve's term for term; test_rung3.py checks that bit for bit.
- The weights are fixed per view (decided by Bart 2026-11-26, plan D6). build_zones computes them once, on the CPU,
  from the photo at its initial exposure s_init, so they have no gradient path and the curve side stays deterministic:
    G_k   = bilinear-up( area-average over blur x blur blocks of min(1, 2^{s_init,k} Y_k) ),  Y = (R + G + B) / 3,
            the area step via adaptive average pooling to (H // blur, W // blur), the same as
            interpolate(mode="area", scale_factor=1/blur) but without a floating-point scale factor;
    g_k   = log2(max(G_k, 1/255)), computed as ln / ln 2 (no torch.log2, for the same NVRTC reason as exp2);
    w_lit = clamp((g_k - c_1) / (c_2 - c_1), 0, 1),  w_sh = 1 - w_lit,  c_1, c_2 = the 25th and 75th percentiles of g
            over every full-resolution pixel of every training view (numpy float64, method "linear").
- Zone targets, also once per view: the grey and the nearest-sampled 1/4-size grid of
  losses.HistogramPriorLoss.compute_histogram_equalization, with each pixel counted with weight w_m. The bin index is
  histc's own float32 arithmetic (x * 255, truncated; x = 1 goes to bin 254), the weighted counts are summed in float64
  and cast to float32, so all-ones weights reproduce histc's counts exactly. CDF_k,m = cumsum(h) / (sum h + 1e-8),
  T_k,m = rung2.hist_target(CDF_k,m) (the one-bin shift), and the zone share pi_k,m = sum(w4_m) / number of pixels.
- Curve loss per view: sum_m pi_k,m * rung2.curve_loss(L_k,m, T_k,m, prior = x^gamma on F_m), i.e.
  sum_m pi_k,m [w(t) mse(L_k,m, T_k,m) + lambda smooth(L_k,m) + 0.5 (mse(p, F_m) + 0.01 mse(p, T_k,m))].
- Parameters: theta [M, K] (both curves, through softplus as in Rung 2), s [N], rho [] (gamma = exp(rho)):
  M K + N + 1. Only the shape prior "F" is supported (the trainer refuses the others).
- Method names differ from ExposureCurve's on purpose (F_zones, zone_heights, composed_curves): a Rung 2 code path
  that is handed this model stops with an AttributeError instead of misreading a [M, ...] array as one curve.
"""

import math
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as nnf
from torch import Tensor

from rung2 import GAMMA_X_FLOOR, LN2, N_BINS, curve_loss, hist_target, interp, theta_from_heights

MIN_ZONE_GAP = 0.05   # c_2 - c_1 below this (stops) and the zones would not separate (plan 2.4)
HIST_SCALE = 0.25     # the 1/4-size grid of HistogramPriorLoss.compute_histogram_equalization
Index = Union[int, Tensor]


# ------------------------------------------------------------------ zone constants (CPU, once per run)
def guide(img: Tensor, s: float, blur: int) -> Tensor:
    """[H, W] float32 g = log2 of the blurred, exposure-aligned grey of one photo img [1, H, W, 3] in [0, 1]."""
    Y = img.float().mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)            # [1, 1, H, W]
    Yg = torch.clamp(math.exp(LN2 * float(s)) * Y, max=1.0)
    H, W = Yg.shape[-2:]
    if blur < 1 or H // blur < 1 or W // blur < 1:
        raise ValueError(f"blur {blur} does not fit a {H} x {W} photo")
    G = nnf.adaptive_avg_pool2d(Yg, (H // blur, W // blur))                  # = interpolate(mode="area")
    G = nnf.interpolate(G, size=(H, W), mode="bilinear", align_corners=False)
    return (torch.log(torch.clamp(G, min=1.0 / 255.0)) / LN2)[0, 0]


def centres(guides: Sequence[Tensor]) -> Tuple[float, float]:
    """c_1, c_2: the 25th and 75th percentiles of every pixel of every guide (numpy float64, linear)."""
    g = np.concatenate([x.detach().double().cpu().numpy().ravel() for x in guides])
    c1, c2 = np.percentile(g, [25.0, 75.0], method="linear")
    return float(c1), float(c2)


def lit_weight(g: Tensor, c1: float, c2: float) -> Tensor:
    """w_lit = clamp((g - c_1) / (c_2 - c_1), 0, 1), computed in float64 and returned as float32."""
    return torch.clamp((g.double() - c1) / (c2 - c1), 0.0, 1.0).float()


def hist_bins(grey: Tensor) -> Tuple[Tensor, Tensor]:
    """torch.histc(bins=255, min=0, max=1)'s bin of each value of a float32 1-D grey: (x - 0) * 255 / (1 - 0) in float32,
    truncated, and x = 1 in the last bin. Also the mask of the values histc counts (inside [0, 1])."""
    keep = (grey >= 0) & (grey <= 1)
    pos = (grey.float() * float(N_BINS)).long()
    pos = torch.where(pos == N_BINS, pos - 1, pos)
    return pos, keep


def zone_histograms(img: Tensor, weights: Tensor) -> Tuple[Tensor, Tensor]:
    """Weighted grey histograms [M, 255] (float32) and zone shares [M] (float32) of one photo img [1, H, W, 3] with
    weight maps weights [M, H, W], on compute_histogram_equalization's nearest-sampled 1/4-size grid."""
    grey = torch.mean(nnf.interpolate(img.float().permute(0, 3, 1, 2), scale_factor=HIST_SCALE), dim=1).flatten()
    w4 = nnf.interpolate(weights.float()[None], scale_factor=HIST_SCALE)[0].reshape(weights.shape[0], -1)
    pos, keep = hist_bins(grey)
    hists, shares = [], []
    for wm in w4:
        h = torch.bincount(pos[keep], weights=wm[keep].double(), minlength=N_BINS)  # float64 sums on the CPU
        hists.append(h.float())
        shares.append(float(wm[keep].double().sum() / keep.sum()))
    return torch.stack(hists), torch.tensor(shares, dtype=torch.float32)


def zone_targets(hists: Tensor) -> Tensor:
    """[M, 255] one-bin-shifted HE-CDFs of weighted histograms [M, 255], normalised as compute_histogram_equalization."""
    cdf = torch.cumsum(hists, dim=1)
    cdf = cdf / (cdf[:, -1:] + 1e-8)
    return hist_target(cdf)


def zone_constants(images: Sequence[Tensor], weights: Tensor) -> Dict[str, Tensor]:
    """Targets [N, M, 255] and shares [N, M] for given weight maps [N, M, H, W] (build_zones' second half; the tests
    also call it with hand-made weights)."""
    targets, shares = [], []
    for img, w in zip(images, weights):
        h, pi = zone_histograms(img, w)
        targets.append(zone_targets(h))
        shares.append(pi)
    return {"targets": torch.stack(targets), "shares": torch.stack(shares)}


def build_zones(images: Sequence[Tensor], s_init: Tensor, blur: int = 16, zones: int = 2) -> Dict[str, Tensor]:
    """The zone constants of a run, from its training photos (each [1, H, W, 3] in [0, 1], in s_init's order, all one
    size) and initial exposures s_init [N]: weights [N, M, H, W] (float32), targets [N, M, 255], shares [N, M],
    centres [2] (float64; NaN for M = 1). zones = 1 gives weight 1 everywhere (Rung 2; tests only)."""
    if zones not in (1, 2):
        raise ValueError(f"zones must be 1 or 2, got {zones}")
    s = s_init.detach().double().cpu().tolist()
    if len(s) != len(images):
        raise ValueError(f"{len(images)} photos but {len(s)} exposures")
    sizes = {tuple(img.shape[1:3]) for img in images}
    if len(sizes) != 1:
        raise ValueError(f"training views differ in size: {sorted(sizes)}")
    H, W = sizes.pop()
    if zones == 1:
        weights = torch.ones(len(images), 1, H, W)
        c = torch.full((2,), float("nan"), dtype=torch.float64)
    else:
        guides = [guide(img, sk, blur) for img, sk in zip(images, s)]
        c1, c2 = centres(guides)
        if not c2 - c1 >= MIN_ZONE_GAP:
            raise ValueError(f"guide percentiles c_2 - c_1 = {c2 - c1:.4f} stops < {MIN_ZONE_GAP}: the zones would not "
                             f"separate")
        w_lit = torch.stack([lit_weight(g, c1, c2) for g in guides])                 # [N, H, W]
        weights = torch.stack([1.0 - w_lit, w_lit], dim=1)                           # [N, 2, H, W]: shadow, lit
        c = torch.tensor([c1, c2], dtype=torch.float64)
    return {"weights": weights, **zone_constants(images, weights), "centres": c}


# ------------------------------------------------------------------ the model
class ZoneExposureCurve(torch.nn.Module):
    """Rung 3: L_k(x, p) = sum_m w_m(p) F_m(min(1, 2^{s_k} x)) with M shared monotone curves F_m on Rung 2's knots, one
    exposure s_k per view and one shape-prior gamma. The zone constants of build_zones are buffers: centres, targets and
    shares persistent (saved in curve_state.pt and checked at re-score), the weight maps not (rebuilt in build)."""

    def __init__(self, knots: Tensor, h_init: Tensor, s_init: Tensor, zones: Dict[str, Tensor], prior: str = "F",
                 blur: int = 16):
        super().__init__()
        if prior != "F":
            raise ValueError(f"Rung 3 supports only the shape prior 'F', got {prior!r}")
        if h_init.numel() != knots.numel():
            raise ValueError(f"{h_init.numel()} heights for {knots.numel()} knots")
        N, M = zones["targets"].shape[:2]
        if s_init.numel() != N or zones["weights"].shape[:2] != (N, M) or zones["shares"].shape != (N, M):
            raise ValueError("zone constants and exposures do not agree in N or M")
        self.prior, self.blur = prior, int(blur)
        self.register_buffer("knots", knots.detach().clone().float())
        self.register_buffer("grid", torch.linspace(0, 1, N_BINS))
        self.register_buffer("s_init", s_init.detach().clone().float().reshape(-1))
        self.register_buffer("centres", zones["centres"].detach().clone().double())
        self.register_buffer("zone_targets", zones["targets"].detach().clone().float())
        self.register_buffer("zone_shares", zones["shares"].detach().clone().float())
        self.register_buffer("weights", zones["weights"].detach().clone().float(), persistent=False)
        self.theta = torch.nn.Parameter(theta_from_heights(h_init).repeat(M, 1).to(self.knots.device))  # [M, K]
        self.s = torch.nn.Parameter(self.s_init.clone())
        self.rho = torch.nn.Parameter(torch.zeros((), device=self.knots.device))  # gamma = 1 at the start

    @property
    def M(self) -> int:
        return self.theta.shape[0]

    @property
    def K(self) -> int:
        return self.theta.shape[1]

    @property
    def N(self) -> int:
        return self.s.numel()

    def n_params(self) -> int:
        return self.M * self.K + self.N + 1

    def f_params(self) -> list:
        """The curves' side (theta of both curves and the prior's rho), which shares F's learning rate."""
        return [self.theta, self.rho]

    def zone_heights(self) -> Tensor:
        """[M, K+1]: each F_m at the knots, [0, c / c_K] with c = cumsum(softplus(theta_m))."""
        c = torch.cumsum(nnf.softplus(self.theta), dim=-1)
        return torch.cat([c.new_zeros(c.shape[0], 1), c / c[:, -1:]], dim=1)

    def F_zones(self, y: Tensor) -> Tensor:
        """[M, *y.shape]: every curve at y in [0, 1]; gradient to theta."""
        h = self.zone_heights()
        return torch.stack([interp(y, self.knots, h[m]) for m in range(self.M)])

    def gain(self, k: Index) -> Tensor:
        """[B] 2^{s_k} for an int or a [B] index tensor; gradient to s."""
        return torch.exp(LN2 * self.s[k].reshape(-1))

    def composed_curves(self, k: Index) -> Tensor:
        """[B, M, 255]: F_m(min(1, 2^{s_k} x_b)) on the official grid; gradient to theta and s (the curve loss)."""
        y = torch.clamp(self.gain(k)[:, None] * self.grid[None], max=1.0)
        h = self.zone_heights()
        return torch.stack([interp(y, self.knots, h[m]) for m in range(self.M)], dim=1)

    def view_weights(self, k: Index) -> Tensor:
        """[B, M, H, W] the fixed weight maps of the views k."""
        return self.weights[torch.as_tensor(k, device=self.weights.device).reshape(-1)]

    @torch.no_grad()
    def map_pixels(self, pixels: Tensor, k: Index, attach: str = "none", color: str = "rgb") -> Tensor:
        """[B, H, W, 3] pseudo-target sum_m w_m F_m(min(1, 2^{s_k} x)) on R, G and B; never carries gradient (Rung 3 is
        detached: --rung2_attach none, --rung2_color rgb; anything else is refused)."""
        if attach != "none" or color != "rgb":
            raise ValueError(f"Rung 3 maps pixels detached and per channel only (attach {attach!r}, color {color!r})")
        w = self.view_weights(k)
        if w.shape[-2:] != pixels.shape[1:3]:
            raise ValueError(f"weight maps {tuple(w.shape[-2:])} do not fit pixels {tuple(pixels.shape[1:3])}")
        y = torch.clamp(self.gain(k).view(-1, 1, 1, 1) * pixels, max=1.0)
        h = self.zone_heights()
        out = None
        for m in range(self.M):
            term = w[:, m, :, :, None] * interp(y, self.knots, h[m])
            out = term if out is None else out + term
        return out

    @torch.no_grad()
    def project_gauge(self):
        """Only relative exposure is identifiable: keep mean s = 0, so the curves carry the overall brightening."""
        self.s.sub_(self.s.mean())

    # ---------------------------------------------------------------- shape prior and curve loss
    def gamma(self) -> Tensor:
        return torch.exp(self.rho)

    def prior_curve(self) -> Tensor:
        """[1, 255] p = clamp(x_b, 1e-3, 1)^gamma, the shape prior both curves are pulled toward; gradient to rho."""
        return torch.clamp(self.grid, GAMMA_X_FLOOR, 1.0)[None] ** self.gamma()

    def loss(self, curves: Tensor, targets: Tensor, shares: Tensor, step: int, exp_name: str = "",
             smooth_lambda: float = 0.1) -> Tuple[Tensor, Dict[str, object]]:
        """The zone curve loss of composed curves [B, M, 255] (composed_curves, attached) against zone targets
        [B, M, 255] with shares [B, M]: per view, sum_m pi_m rung2.curve_loss(curve_m, target_m, prior x^gamma on F_m);
        the mean over views. parts has HistogramPriorLoss's keys (cdf, pseudo, smooth: the same weighted sums; w_cdf),
        so Diagnostics.log_losses works unchanged. With B = 1 and M = 1 (pi = 1) this is ExposureCurve.loss exactly."""
        p, Fg = self.prior_curve(), self.F_zones(self.grid)
        totals, all_parts = [], []
        for b in range(curves.shape[0]):
            total, parts = None, {}
            for m in range(self.M):
                t, pm = curve_loss(curves[b:b + 1, m], targets[b:b + 1, m], step, exp_name, smooth_lambda,
                                   prior=p, prior_x=Fg[m][None])
                pi = shares[b, m]
                total = pi * t if total is None else total + pi * t
                for key in ("cdf", "pseudo", "smooth"):
                    parts[key] = pi * pm[key] if key not in parts else parts[key] + pi * pm[key]
                parts["w_cdf"] = pm["w_cdf"]
            totals.append(total)
            all_parts.append(parts)
        if len(totals) == 1:
            return totals[0], all_parts[0]
        mean = {key: torch.stack([q[key] for q in all_parts]).mean() for key in ("cdf", "pseudo", "smooth")}
        return torch.stack(totals).mean(), {**mean, "w_cdf": all_parts[0]["w_cdf"]}

    @torch.no_grad()
    def strictly_increasing(self) -> bool:
        return bool((torch.diff(self.zone_heights(), dim=1) > 0).all())

    @torch.no_grad()
    def summary(self) -> Dict[str, object]:
        """Plain numbers for json (the diagnostics' rung2_state.json). Per-zone arrays only under new keys (no "heights",
        no "F_grid"), so a single-curve reader stops instead of misreading them (plan 2.5)."""
        return {"K": self.K, "N": self.N, "zones": self.M, "n_params": self.n_params(), "prior": self.prior,
                "gamma": float(self.gamma()), "knots": self.knots.cpu().tolist(),
                "heights_zones": self.zone_heights().cpu().tolist(), "F_strictly_increasing": self.strictly_increasing(),
                "s_init": self.s_init.cpu().tolist(), "s_final": self.s.cpu().tolist(), "mean_s": float(self.s.mean()),
                "blur": self.blur, "centres": self.centres.cpu().tolist(), "zone_shares": self.zone_shares.cpu().tolist()}
