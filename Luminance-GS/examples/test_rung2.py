"""Unit tests for rung2.py (Implementation_guides/M2_rung2_plan.md, step 5). CPU, a few seconds, plain asserts.

    python test_rung2.py        (from Luminance-GS/examples, luminancegs env)

Tests 1 to 10 are the plan's. 11 to 13 were added with the code: the target shift and the initialisation round
trip (11), used_range_max (12), and the one-bin alignment against the official LUT (13; decided 2026-11-11).
14 (added 2026-11-13): the shape prior with one global gamma (decided 2026-11-13).
15 and 16 (added 2026-11-25, E5): the colour modes of map_pixels (rgb unchanged; luminance) and clipped_share.
"""

import math
import sys
import traceback

import numpy as np
import torch

from rung2 import (LN2, PRIOR_WEIGHT, ExposureCurve, curve_loss, hist_target, init_heights, interp, make_knots,
                   theta_from_heights, used_range_max)

K = 32


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _model(N=6, spacing="quadratic", seed=0, s_scale=0.5, theta_scale=1.0, dtype=torch.float32):
    """A model with random F (theta) and random mean-zero exposures."""
    g = _gen(seed)
    knots = make_knots(K, spacing, y_max=0.6 if spacing == "usedrange" else None)
    m = ExposureCurve(knots, init_heights("identity", knots), torch.zeros(N))
    with torch.no_grad():
        m.theta.copy_(theta_scale * torch.randn(K, generator=g))
        m.s.copy_(s_scale * torch.randn(N, generator=g))
    m.project_gauge()
    return m.to(dtype)


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


# ------------------------------------------------------------------ the plan's tests
def test_1_knots():
    for spacing in ("quadratic", "uniform", "usedrange"):
        for k in (2, 8, 32, 64):
            y = make_knots(k, spacing, y_max=0.4 if spacing == "usedrange" else None)
            assert y.shape == (k + 1,), f"{spacing}, K={k}: shape {tuple(y.shape)}"
            assert y[0] == 0 and y[-1] == 1, f"{spacing}, K={k}: ends {float(y[0])}, {float(y[-1])}"
            assert bool((torch.diff(y) > 0).all()), f"{spacing}, K={k}: not strictly increasing"
    n = int((make_knots(K, "quadratic") <= 0.25).sum())
    assert n >= K // 2, f"quadratic puts only {n} knots in [0, 0.25]"
    n = int((make_knots(K, "usedrange", y_max=0.4) <= 0.4 + 1e-7).sum())
    assert n == K, f"usedrange puts {n} knots at or below y_max, expected K = {K}"
    for bad in (None, 0.0, 1.0, 1.2):
        assert _raises(lambda: make_knots(K, "usedrange", y_max=bad)), f"usedrange accepted y_max = {bad}"
    assert _raises(lambda: make_knots(K, "cubic")), "unknown spacing accepted"


def test_2_heights():
    for seed in range(5):
        h = _model(seed=seed, theta_scale=2.0).heights().detach()
        assert float(h[0]) == 0.0, f"seed {seed}: h_0 = {float(h[0])}"
        assert abs(float(h[-1]) - 1.0) <= 1e-6, f"seed {seed}: h_K = {float(h[-1])}"
        assert bool((torch.diff(h) > 0).all()), f"seed {seed}: heights not strictly increasing"


def test_3_interp():
    m = _model(seed=1, theta_scale=1.5)
    y, h = m.knots, m.heights().detach()
    yq = torch.rand(10_000, generator=_gen(1))
    yq[:4] = torch.tensor([0.0, 1.0, float(y[5]), float(y[17])])  # the ends and two knots
    got = interp(yq, y, h).numpy()
    want = np.interp(yq.double().numpy(), y.double().numpy(), h.double().numpy())
    err = float(np.abs(got - want).max())
    assert err <= 1e-6, f"interp differs from np.interp by {err:.2e}"
    md = _model(seed=1, theta_scale=1.5, dtype=torch.float64)
    fine = torch.linspace(0, 1, 200_001, dtype=torch.float64)
    step = float(torch.diff(interp(fine, md.knots, md.heights().detach())).min())
    assert step >= -1e-12, f"interp decreases somewhere (smallest step {step:.2e})"


def test_4_identity():
    knots = make_knots(K, "uniform")
    m = ExposureCurve(knots, init_heights("identity", knots), torch.zeros(3))
    px = torch.rand(1, 7, 9, 3, generator=_gen(4))
    err = float((m.map_pixels(px, 1) - px).abs().max())
    assert err <= 1e-6, f"uniform knots, identity F, s = 0 do not return the input (max error {err:.2e})"


def test_5_gradients():
    m = _model(N=4, seed=2, dtype=torch.float64)
    k = 2
    with torch.no_grad():
        m.s[k] = 0.4  # gain 2^0.4 = 1.32, so pixels above 0.76 saturate
    px = torch.rand(1, 6, 6, 3, generator=_gen(5), dtype=torch.float64) * 0.7

    out = m.map_pixels(px, k, "none")
    assert not out.requires_grad, "attach none: the output carries gradient"

    m.zero_grad(set_to_none=True)
    m.map_pixels(px, k, "s").sum().backward()
    assert m.theta.grad is None or float(m.theta.grad.abs().max()) == 0.0, "attach s: theta gets gradient"
    others = [i for i in range(m.N) if i != k]
    assert float(m.s.grad[k]) != 0.0, "attach s: no gradient to s_k"
    assert float(m.s.grad[others].abs().max()) == 0.0, "attach s: gradient leaks to other views"
    grad_s = float(m.s.grad[k])

    m.zero_grad(set_to_none=True)
    m.map_pixels(px, k, "all").sum().backward()
    assert m.theta.grad is not None and float(m.theta.grad.abs().max()) > 0, "attach all: no gradient to theta"
    assert float(m.s.grad[k]) != 0.0, "attach all: no gradient to s_k"

    # per pixel: d F(2^s x) / d s = F'(y) y ln 2, against central differences (pixels away from knots)
    s0 = float(m.s[k])
    with torch.no_grad():
        y = torch.clamp(2.0 ** s0 * px, max=1.0)
        h = m.heights()
        j = torch.searchsorted(m.knots[1:-1].contiguous(), y.contiguous(), right=True)
        analytic = (h[j + 1] - h[j]) / (m.knots[j + 1] - m.knots[j]) * y * LN2

    def at(s, img=px):
        with torch.no_grad():
            m.s[k] = s
        return m.map_pixels(img, k, "none")

    eps = 1e-7
    fd = (at(s0 + eps) - at(s0 - eps)) / (2 * eps)
    at(s0)
    keep = ((y[..., None] - m.knots).abs().min(-1).values > 1e-4) & (y < 1 - 1e-4)
    assert int(keep.sum()) > 50, "too few test pixels away from the knots"
    rel = float(((fd - analytic).abs() / analytic.abs().clamp_min(1e-12))[keep].max())
    assert rel <= 1e-3, f"finite differences differ from F'(y) y ln 2 by {rel:.2e} (relative)"
    assert math.isclose(grad_s, float(analytic.sum()), rel_tol=1e-9), \
        f"autograd d/ds_k {grad_s:.6g} differs from the analytic sum {float(analytic.sum()):.6g}"

    sat = torch.full((1, 2, 2, 3), 0.9, dtype=torch.float64)  # 1.32 * 0.9 > 1
    m.zero_grad(set_to_none=True)
    m.map_pixels(sat, k, "s").sum().backward()
    assert float(m.s.grad[k]) == 0.0, "saturated pixels give gradient to s_k"
    fd_sat = float(((at(s0 + 1e-4, sat) - at(s0 - 1e-4, sat)) / 2e-4).abs().max())
    at(s0)
    assert fd_sat == 0.0, f"saturated pixels move with s_k ({fd_sat:.2e})"


def test_6_composed_vs_map():
    m = _model(N=5, seed=3)
    img = m.grid.view(1, 1, -1, 1).expand(1, 1, 255, 3).contiguous()  # pixels at x_b = b / 254
    for k in range(m.N):
        a = m.composed_curve(k).detach()[0]
        b = m.map_pixels(img, k, "none")[0, 0, :, 0]
        err = float((a - b).abs().max())
        assert err <= 1e-7, f"view {k}: composed_curve and map_pixels differ by {err:.2e}"
    batched = m.composed_curve(torch.arange(m.N)).detach()
    single = torch.cat([m.composed_curve(k).detach() for k in range(m.N)])
    assert batched.shape == (m.N, 255) and torch.equal(batched, single), "batched composed_curve differs"


def test_7_gauge():
    m = _model(N=7, seed=4)
    with torch.no_grad():
        m.s.add_(0.3)
    before = m.s.detach().clone()
    m.project_gauge()
    after = m.s.detach()
    assert abs(float(after.mean())) <= 1e-6, f"mean s after projection {float(after.mean()):.2e}"
    d = float(((after[:, None] - after[None]) - (before[:, None] - before[None])).abs().max())
    assert d <= 1e-6, f"projection changed differences between views by {d:.2e}"


def test_8_n_params():
    for n in (1, 22, 43):
        m = _model(N=n)
        assert m.n_params() == K + n, f"n_params {m.n_params()} != K + N = {K + n}"
        assert sum(p.numel() for p in m.parameters()) == K + n, "parameter tensors do not add up to K + N"


def test_9_synthetic_recovery():
    """CDFs of the model's own form, G(min(1, 2^{s_k} x)), with G reaching 1 at 0.7 < 2^{min s_k} = 0.707."""
    def G(y):
        return 1.0 - (1.0 - np.minimum(1.0, y / 0.7)) ** 3

    n = 8
    s_true = np.linspace(-0.5, 0.5, n)  # mean 0
    x = np.arange(255) / 254
    T = torch.tensor(np.stack([G(np.minimum(1.0, 2.0 ** s * x)) for s in s_true]), dtype=torch.float32)
    knots = make_knots(K, "quadratic")

    h = init_heights("cdf", knots, T, torch.tensor(s_true))
    err_h = float(np.abs(h.double().numpy() - G(knots.double().numpy())).max())
    assert err_h <= 0.02, f"cdf init with the true s misses G at the knots by {err_h:.4f}"

    m = ExposureCurve(knots, init_heights("cdf", knots, T, torch.zeros(n)), torch.zeros(n))
    steps = 3000
    opt = torch.optim.Adam([m.theta, m.s], lr=1e-2)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.01 ** (1.0 / steps))
    views = torch.arange(n)
    for step in range(steps):
        loss, _ = curve_loss(m.composed_curve(views), T, step, "low", 0.1)
        opt.zero_grad(set_to_none=True)
        (10 * loss).backward()
        opt.step()
        sched.step()
        m.project_gauge()
    err_s = float(np.abs(m.s.detach().double().numpy() - s_true).max())
    assert err_s <= 0.03, f"Adam from s = 0 recovers s within {err_s:.4f} stops, not 0.03"
    print(f"    init vs G at the knots: {err_h:.4f}; s recovered within {err_s:.4f} stops after {steps} steps")


def test_10_curve_loss():
    g = _gen(5)
    c, t = torch.rand(2, 255, generator=g), torch.rand(2, 255, generator=g)
    cn, tn = c.double().numpy(), t.double().numpy()
    mse = float(((cn - tn) ** 2).mean())
    sm = float(((cn[:, 1:] - cn[:, :-1]) ** 2).mean())
    for step, exp, w in ((0, "low", 1.0), (2999, "low", 1.0), (3000, "low", 0.5), (9999, "low", 0.5),
                         (2999, "over_exp", 1.0), (3000, "over_exp", 0.1)):
        total, parts = curve_loss(c, t, step, exp, 0.1)
        assert parts["w_cdf"] == w, f"step {step}, {exp}: w = {parts['w_cdf']}, expected {w}"
        assert abs(float(total) - (w * mse + 0.1 * sm)) <= 1e-6, f"step {step}, {exp}: total differs"
        assert abs(float(parts["cdf"]) - mse) <= 1e-6 and abs(float(parts["smooth"]) - sm) <= 1e-7
        assert math.isnan(parts["pseudo"]), "pseudo part is not NaN"

    # the same terms as the official HistogramPriorLoss, which Rung 2 keeps minus the shape prior
    from losses import HistogramPriorLoss
    hp = HistogramPriorLoss()
    img = torch.rand(1, 64, 64, 3, generator=g)
    curve = torch.rand(1, 255, generator=g).sort().values
    _, official = hp(curve, img, torch.linspace(0, 1, 255)[None], 3000, exp_name="low", return_parts=True)
    _, ours = curve_loss(curve, hp.compute_histogram_equalization(img), 3000, "low", hp.lambda_smooth)
    for key in ("cdf", "smooth", "w_cdf"):
        assert abs(float(official[key]) - float(ours[key])) <= 1e-7, f"{key} differs from HistogramPriorLoss"


# ------------------------------------------------------------------ added with the code
def test_11_target_and_roundtrip():
    cdf = torch.rand(3, 255, generator=_gen(6)).sort(dim=1).values
    cdf[:, -1] = 1.0
    T = hist_target(cdf, "shift")
    assert bool((T[:, 0] == 0).all()) and torch.equal(T[:, 1:], cdf[:, :-1]), "the shift is not [0, CDF[0..253]]"
    assert hist_target(cdf, "none") is cdf, "align none changes the CDF"
    assert _raises(lambda: hist_target(cdf, "half")), "unknown alignment accepted"
    s = torch.tensor([0.3, -0.1, -0.2])
    for spacing in ("quadratic", "uniform"):
        knots = make_knots(K, spacing)
        h = init_heights("cdf", knots, T, s)
        assert float(h[0]) == 0.0 and abs(float(h[-1]) - 1) <= 1e-6 and bool((torch.diff(h) > 0).all()), \
            f"{spacing}: cdf init is not a valid F"
        err = float((ExposureCurve(knots, h, s).heights().detach() - h).abs().max())
        assert err <= 1e-6, f"{spacing}: heights do not survive the theta round trip ({err:.2e})"
    assert _raises(lambda: theta_from_heights(torch.tensor([0.0, 0.5, 0.5, 1.0]))), "flat heights accepted"


def test_12_used_range_max():
    img = (torch.arange(100).repeat_interleave(3) / 255.0).view(1, 10, 10, 3)  # levels 0..99, 3 values each
    assert abs(used_range_max([img], torch.tensor([0.0])) - 99 / 255) <= 1e-9, "s = 0: quantile is not level 99"
    assert abs(used_range_max([img], torch.tensor([1.0])) - 2 * 99 / 255) <= 1e-9, "s = 1: not twice level 99"
    assert used_range_max([img], torch.tensor([3.0])) == 0.99, "a saturated range is not capped at 0.99"
    mixed = used_range_max([img, img * 0], torch.tensor([0.0, 0.0]), q=0.5)  # half the values are 0
    assert mixed == 0.0, f"q = 0.5 over a half-black set gives {mixed}"


def test_13_alignment_matches_official_lut():
    """With the shift, an interpolated curve on the grid x_b = b/254 that equals its target reads the same CDF
    bin as the official truncating LUT fitted to the CDF (level l -> CDF[l-1]); without it, about one bin brighter.
    A dark photo: geometric grey levels with mean 8."""
    from tools import LUT_mapping
    p = 1.0 / 9.0
    pmf = p * (1 - p) ** np.arange(255)
    pmf[-1] += 1 - pmf.sum()
    cdf = torch.tensor(np.cumsum(pmf), dtype=torch.float32)
    levels = torch.arange(0, 17)
    x = levels / 255.0
    official = LUT_mapping((x.view(1, 1, -1), None, None), cdf.view(1, -1))[0, 0]
    grid = torch.linspace(0, 1, 255)
    shifted = interp(x, grid, hist_target(cdf[None], "shift")[0])
    unshifted = interp(x, grid, cdf)
    assert float(shifted[0]) == 0.0, "level 0 is not black with the shift"
    e_s, e_n = (shifted - official)[1:].abs(), (unshifted - official)[1:].abs()
    assert bool((e_s <= 0.1 * e_n).all()), \
        f"levels 1-16: the shift does not track the official LUT (max {float(e_s.max()):.4f} vs {float(e_n.max()):.4f})"
    print(f"    levels 1-16, |F - official LUT|: shifted max {float(e_s.max()):.4f}, unshifted max {float(e_n.max()):.4f}")


def test_14_shape_prior():
    """The shape prior (decided 2026-11-13): the official term with one global gamma, on F or on the composed
    curves; gamma starts at 1; with prior "F" the term gives no gradient to s, so exposure compensation is the
    HE term's alone."""
    from losses import HistogramPriorLoss, gamma_curve
    g = _gen(7)
    knots = make_knots(K, "quadratic")
    n = 4
    for prior, extra in (("none", 0), ("F", 1), ("composed", 1)):
        m = ExposureCurve(knots, init_heights("identity", knots), torch.zeros(n), prior=prior)
        assert m.n_params() == K + n + extra, f"{prior}: n_params {m.n_params()} != {K + n + extra}"
        assert sum(p.numel() for p in m.parameters()) == m.n_params(), f"{prior}: parameter tensors do not add up"
        assert sum(p.numel() for p in m.f_params()) == K + extra, f"{prior}: f_params is not theta (+ rho)"
    assert _raises(lambda: ExposureCurve(knots, init_heights("identity", knots), torch.zeros(n), prior="beta")), \
        "unknown prior accepted"
    assert math.isnan(float(ExposureCurve(knots, init_heights("identity", knots), torch.zeros(n)).gamma())), \
        "gamma is not NaN without a prior"

    m = _model(N=n, seed=8)
    mp = ExposureCurve(knots, m.heights().detach(), m.s.detach(), prior="F")
    assert float(mp.gamma()) == 1.0, "gamma does not start at 1"
    with torch.no_grad():
        mp.rho.fill_(math.log(0.15))
    want = gamma_curve(mp.grid[None], torch.tensor([0.15]))
    assert float((mp.prior_curve().detach() - want).abs().max()) <= 1e-6, "prior_curve is not losses.gamma_curve"

    # the algebra of the official HistogramPriorLoss, with its pseudo curve replaced by the gamma curve
    hp = HistogramPriorLoss()
    img = torch.rand(1, 64, 64, 3, generator=g)
    curve = torch.rand(1, 255, generator=g).sort().values
    p = gamma_curve(torch.linspace(0, 1, 255)[None], torch.tensor([0.3]))
    for step in (0, 3000):
        tot_o, par_o = hp(curve, img, p, step, exp_name="low", return_parts=True)
        tot_r, par_r = curve_loss(curve, hp.compute_histogram_equalization(img), step, "low", hp.lambda_smooth, prior=p)
        assert abs(float(tot_o) - float(tot_r)) <= 1e-7, f"step {step}: total differs from HistogramPriorLoss"
        for key in ("cdf", "pseudo", "smooth", "w_cdf"):
            assert abs(float(par_o[key]) - float(par_r[key])) <= 1e-7, f"step {step}: {key} differs"

    # where the prior acts: on F it trains theta and rho but never s; on the composed curves it also pulls s
    T = hist_target(torch.rand(n, 255, generator=g).sort(dim=1).values)
    views = torch.arange(n)
    for prior, s_moves in (("F", False), ("composed", True)):
        mq = ExposureCurve(knots, m.heights().detach(), m.s.detach(), prior=prior)
        with torch.no_grad():
            mq.rho.fill_(math.log(0.3))
        _, parts = mq.loss(mq.composed_curve(views), T, 3000, "low", 0.1)
        (PRIOR_WEIGHT * parts["pseudo"]).backward()
        gs = 0.0 if mq.s.grad is None else float(mq.s.grad.abs().max())
        assert (gs > 0) == s_moves, f"prior {prior}: |d prior / d s| = {gs:.2e}"
        assert float(mq.theta.grad.abs().max()) > 0 and float(mq.rho.grad.abs()) > 0, \
            f"prior {prior}: theta or rho gets no gradient from the prior"
        total, parts = mq.loss(mq.composed_curve(views), T, 3000, "low", 0.1)
        base, _ = curve_loss(mq.composed_curve(views), T, 3000, "low", 0.1)
        assert abs(float(total) - float(base) - PRIOR_WEIGHT * float(parts["pseudo"])) <= 1e-7, \
            f"prior {prior}: total is not the curve loss plus 0.5 pseudo"


def test_15_color_modes():
    """E5 (E5_luminance_plan.md section 3): rgb is the M3 code bit for bit; luminance gives grey pixels the rgb
    value, keeps channel ratios where nothing clips, puts the output's channel mean at L_k(Y), keeps black black and
    stays in [0, 1]."""
    m = _model(N=5, seed=9, s_scale=0.8)
    g = _gen(10)
    px = torch.rand(1, 40, 50, 3, generator=g)
    px[0, 0, :3] = torch.tensor([[1.0, 0.2, 0.1], [0.0, 0.0, 0.0], [0.95, 0.95, 0.0]])  # saturated, black, yellow
    for k in range(m.N):
        old = interp(torch.clamp(m.gain(k).view(-1, 1, 1, 1) * px, max=1.0), m.knots, m.heights()).detach()
        for attach in ("none", "s", "all"):
            assert torch.equal(m.map_pixels(px, k, attach).detach(), old), f"view {k}, {attach}: default changed"
            assert torch.equal(m.map_pixels(px, k, attach, "rgb").detach(), old), f"view {k}, {attach}: rgb changed"
    assert _raises(lambda: m.map_pixels(px, 0, "none", "hsv")), "unknown colour mode accepted"
    assert _raises(lambda: m.clipped_share(px, 0, "hsv")), "unknown colour mode accepted by clipped_share"

    grey = torch.rand(1, 30, 30, 1, generator=g).expand(1, 30, 30, 3).contiguous()
    err_grey = max(float((m.map_pixels(grey, k, "none", "luminance") - m.map_pixels(grey, k, "none")).abs().max())
                   for k in range(m.N))
    assert err_grey <= 1e-6, f"grey pixels: luminance and rgb differ by {err_grey:.2e}"

    dark = px * 0.2  # a low-light photo's range (levels 0-51), where most pixels do not clip
    err_ratio = err_mean = 0.0
    for k in range(m.N):
        out = m.map_pixels(px, k, "none", "luminance")
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0, f"view {k}: output leaves [0, 1]"
        assert float(out[0, 0, 1].abs().max()) == 0.0, f"view {k}: black is not black"
        out = m.map_pixels(dark, k, "none", "luminance")
        Y = dark.mean(-1, keepdim=True)
        LY = m.F(torch.clamp(m.gain(k).view(-1, 1, 1, 1) * Y, max=1.0)).detach()  # L_k(Y), [1, H, W, 1]
        free = (Y[..., 0] > 0.004) & (dark * LY / Y < 1 - 1e-4).all(-1)  # pixels the clamp leaves alone
        assert int(free.sum()) > 1000, f"view {k}: only {int(free.sum())} unclipped test pixels"
        r = (out / dark.clamp_min(1e-4))[free & (dark > 0.01).all(-1)]
        err_ratio = max(err_ratio, float(((r - r.mean(-1, keepdim=True)).abs() / r.mean(-1, keepdim=True)).max()))
        err_mean = max(err_mean, float((out.mean(-1) - LY[..., 0])[free].abs().max()))
    assert err_ratio <= 1e-5, f"channel ratios change by up to {err_ratio:.2e} (relative)"
    assert err_mean <= 1e-6, f"unclipped output's channel mean misses L_k(Y) by {err_mean:.2e}"
    print(f"    grey: |lum - rgb| max {err_grey:.1e}; ratio change max {err_ratio:.1e}; |mean - L_k(Y)| max {err_mean:.1e}")

    # clipped share, on an identity F: s = 0 clips nothing; s = 0.6 (gain 1.52) clips R of (0.9, 0.3, 0.3) only
    knots = make_knots(K, "uniform")
    mi = ExposureCurve(knots, init_heights("identity", knots), torch.tensor([0.0, 0.6]))
    red = torch.tensor([0.9, 0.3, 0.3]).view(1, 1, 1, 3).expand(1, 4, 4, 3).contiguous()
    assert mi.clipped_share(red, 0, "luminance") == 0.0 and mi.clipped_share(red, 0, "rgb") == 0.0
    assert abs(mi.clipped_share(red, 1, "luminance") - 1 / 3) <= 1e-7, "luminance: R alone should clip"
    assert abs(mi.clipped_share(red, 1, "rgb") - 1 / 3) <= 1e-7, "rgb: 1.52 * 0.9 alone exceeds 1"
    lum = mi.map_pixels(red, 1, "none", "luminance")[0, 0, 0]
    assert float(lum[0]) == 1.0 and abs(float(lum[1]) - 0.3 * 2 ** 0.6) <= 1e-6, f"clipped pixel is {lum.tolist()}"


def test_16_color_gradients():
    """E5: the luminance mode's gradients follow --rung2_attach as rgb's do (none: none; s: s_k only; all: theta and
    s_k), and d/ds_k matches central differences away from knots and clipping."""
    m = _model(N=4, seed=11, dtype=torch.float64)
    k = 1
    with torch.no_grad():
        m.s[k] = 0.3
    px = torch.rand(1, 8, 8, 3, generator=_gen(12), dtype=torch.float64) * 0.2

    assert not m.map_pixels(px, k, "none", "luminance").requires_grad, "attach none: the output carries gradient"
    m.zero_grad(set_to_none=True)
    m.map_pixels(px, k, "s", "luminance").sum().backward()
    assert m.theta.grad is None or float(m.theta.grad.abs().max()) == 0.0, "attach s: theta gets gradient"
    others = [i for i in range(m.N) if i != k]
    assert float(m.s.grad[k]) != 0.0 and float(m.s.grad[others].abs().max()) == 0.0, "attach s: wrong s gradient"
    grad_s = float(m.s.grad[k])
    m.zero_grad(set_to_none=True)
    m.map_pixels(px, k, "all", "luminance").sum().backward()
    assert m.theta.grad is not None and float(m.theta.grad.abs().max()) > 0, "attach all: no gradient to theta"
    assert float(m.s.grad[k]) != 0.0, "attach all: no gradient to s_k"

    y = torch.clamp(2.0 ** 0.3 * px.mean(-1), max=1.0)
    keep = ((y[..., None] - m.knots).abs().min(-1).values > 1e-7).all()  # 1e-7 is ~10x the step in y below
    out = m.map_pixels(px, k, "none", "luminance")
    assert bool(keep) and float(out.max()) < 1.0, "test pixels touch a knot or clip; change the seed"
    s0, eps = 0.3, 1e-7

    def total(s):
        with torch.no_grad():
            m.s[k] = s
        return float(m.map_pixels(px, k, "none", "luminance").sum())

    fd = (total(s0 + eps) - total(s0 - eps)) / (2 * eps)
    total(s0)
    assert math.isclose(grad_s, fd, rel_tol=1e-5), f"autograd d/ds_k {grad_s:.8g} vs central differences {fd:.8g}"


TESTS = [test_1_knots, test_2_heights, test_3_interp, test_4_identity, test_5_gradients, test_6_composed_vs_map,
         test_7_gauge, test_8_n_params, test_9_synthetic_recovery, test_10_curve_loss,
         test_11_target_and_roundtrip, test_12_used_range_max, test_13_alignment_matches_official_lut,
         test_14_shape_prior, test_15_color_modes, test_16_color_gradients]


def main():
    torch.set_num_threads(2)  # light enough to run beside a training batch
    failed = []
    for t in TESTS:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - len(failed)}/{len(TESTS)} passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
