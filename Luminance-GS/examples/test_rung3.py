"""Unit tests for rung3.py (Implementation_guides/M6_rung3_plan.md section 3, tests 1 and 2). CPU, plain asserts.

    python test_rung3.py          (from Luminance-GS/examples, luminancegs env): the unit tests (test 1)
    python test_rung3.py --cuda   the CUDA op check (test 2): the convex mix and the moved buffers on the GPU

Test 1's items, in plan order: M = 1 is Rung 2 bit for bit (1); two equal curves with constant weights 1/2 are Rung 2
(2); the weights (3); monotone per pixel (4); the histogram against torch.histc on synthetic levels and on three real
training photos (5); exposure invariance of the guide (6); gradients (7). Added with the code: build determinism, the
state and the refusals (8); the loss's share weighting and its batch mean (9).
"""

import math
import sys
import traceback

import numpy as np
import torch

from losses import HistogramPriorLoss
from rung2 import ExposureCurve, curve_loss, hist_target, init_heights, make_knots
from rung3 import (MIN_ZONE_GAP, ZoneExposureCurve, build_zones, centres, guide, hist_bins, lit_weight,
                   zone_constants, zone_histograms, zone_targets)

K = 32
H, W = 100, 130  # not multiples of 16 or 4, as the real photos (374 x 499) are not


def _gen(seed):
    return torch.Generator().manual_seed(seed)


def _photos(n=5, seed=0, h=H, w=W):
    """Dark 8-bit test photos: a shadowed left half and a lit right half, textured colour, a saturated red patch, and a
    different exposure per view."""
    g = _gen(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    light = 0.03 + 0.12 * torch.sigmoid((xx - 0.5) * 12)
    gains = 2.0 ** (0.6 * torch.rand(n, generator=g) - 0.3)
    out = []
    for k in range(n):
        refl = 0.4 + 0.6 * torch.rand(h, w, 3, generator=g)
        refl[20:40, 30:60] = torch.tensor([0.9, 0.2, 0.1])
        x = torch.clamp(light[..., None] * refl * gains[k], 0, 1)
        out.append((torch.round(x * 255) / 255)[None].float())
    return out


def _s_init(images):
    """As Runner.measure_exposures: log2(m_bar / m_k), geometric mean, float32."""
    m = np.array([float(img.mean()) for img in images])
    return torch.tensor(np.log2(m).mean() - np.log2(m), dtype=torch.float32)


def _pair(images, zones, seed=0, s_scale=0.3, blur=16):
    """A Rung 3 model and a Rung 2 model (prior F) with the same random F (every zone), s and gamma."""
    s0 = _s_init(images)
    zd = build_zones(images, s0, blur, zones)
    knots = make_knots(K, "quadratic")
    z = ZoneExposureCurve(knots, init_heights("identity", knots), s0, zd, blur=blur)
    r = ExposureCurve(knots, init_heights("identity", knots), s0, prior="F")
    g = _gen(seed)
    with torch.no_grad():
        th, ds = torch.randn(K, generator=g), s_scale * torch.randn(len(images), generator=g)
        z.theta.copy_(th.repeat(z.M, 1))
        r.theta.copy_(th)
        z.s.add_(ds)
        r.s.add_(ds)
        z.rho.fill_(math.log(0.2))
        r.rho.fill_(math.log(0.2))
    z.project_gauge()
    r.project_gauge()
    return z, r, zd


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


# ------------------------------------------------------------------ test 1, the plan's items
def test_1_m1_is_rung2():
    """With one zone (weight 1 everywhere): zone targets, composed curves, pseudo-targets and the curve loss equal
    Rung 2's bit for bit (torch.equal)."""
    images = _photos()
    z, r, zd = _pair(images, zones=1)
    hp = HistogramPriorLoss()
    assert z.M == 1 and bool((zd["shares"] == 1.0).all()), "one zone: shares are not exactly 1"
    for k, img in enumerate(images):
        target = hist_target(hp.compute_histogram_equalization(img))
        assert torch.equal(z.zone_targets[k:k + 1, 0], target), f"view {k}: zone target differs from Rung 2's"
        assert torch.equal(z.composed_curves(k)[:, 0].detach(), r.composed_curve(k).detach()), \
            f"view {k}: composed curve differs"
        assert torch.equal(z.map_pixels(img, k), r.map_pixels(img, k)), f"view {k}: pseudo-target differs"
        for step, exp in ((0, "low"), (3000, "low"), (3000, "over_exp")):
            tz, pz = z.loss(z.composed_curves(k), z.zone_targets[k:k + 1], z.zone_shares[k:k + 1], step, exp, 0.1)
            tr, pr = r.loss(r.composed_curve(k), target, step, exp, 0.1)
            assert torch.equal(tz, tr), f"view {k}, step {step}, {exp}: loss {float(tz)!r} vs {float(tr)!r}"
            for key in ("cdf", "pseudo", "smooth"):
                assert torch.equal(pz[key], pr[key]), f"view {k}, step {step}: part {key} differs"
            assert pz["w_cdf"] == pr["w_cdf"], f"view {k}, step {step}: w_cdf differs"


def test_2_equal_curves_half_weights():
    """F_sh = F_lit = F and w = 1/2 everywhere give Rung 2's pseudo-target and loss (to 1e-6)."""
    images = _photos(seed=1)
    z, r, _ = _pair(images, zones=2, seed=1)
    half = 0.5 * torch.ones(len(images), 2, H, W)
    zc = zone_constants(images, half)
    z.weights = half
    hp = HistogramPriorLoss()
    err_px = err_loss = 0.0
    for k, img in enumerate(images):
        err_px = max(err_px, float((z.map_pixels(img, k) - r.map_pixels(img, k)).abs().max()))
        target = hist_target(hp.compute_histogram_equalization(img))
        tz, _ = z.loss(z.composed_curves(k), zc["targets"][k:k + 1], zc["shares"][k:k + 1], 3000, "low", 0.1)
        tr, _ = r.loss(r.composed_curve(k), target, 3000, "low", 0.1)
        err_loss = max(err_loss, abs(float(tz) - float(tr)))
    assert err_px <= 1e-6, f"pseudo-targets differ by {err_px:.2e}"
    assert err_loss <= 1e-6, f"losses differ by {err_loss:.2e}"
    print(f"    w = 1/2, equal curves: pseudo-target max |diff| {err_px:.1e}, loss max |diff| {err_loss:.1e}")


def test_3_weights():
    """Non-negative, summing to 1, exactly 0 below c_1 and 1 above c_2; the lit zone is the lit half; shares sum to 1."""
    images = _photos(seed=2)
    s0 = _s_init(images)
    zd = build_zones(images, s0, 16)
    w = zd["weights"]
    assert w.shape == (len(images), 2, H, W), f"weights shape {tuple(w.shape)}"
    assert float(w.min()) >= 0.0, "a negative weight"
    err = float((w.sum(1) - 1).abs().max())
    assert err <= 1e-6, f"weights sum to 1 only within {err:.2e}"
    gs = [guide(img, float(s), 16) for img, s in zip(images, s0)]
    c1, c2 = centres(gs)
    assert torch.equal(zd["centres"], torch.tensor([c1, c2], dtype=torch.float64)), "stored centres differ"
    assert c2 - c1 >= MIN_ZONE_GAP, f"c_2 - c_1 = {c2 - c1:.3f}"
    for k, g in enumerate(gs):
        wl = w[k, 1]
        assert bool((wl[g <= c1] == 0).all()) and bool((wl[g >= c2] == 1).all()), f"view {k}: not 0 / 1 outside"
        assert float(wl[:, W // 2:].mean()) > float(wl[:, :W // 2].mean()) + 0.3, f"view {k}: lit zone is not the lit half"
    assert float((zd["shares"].sum(1) - 1).abs().max()) <= 1e-6, "zone shares do not sum to 1"
    print(f"    centres {255 * 2 ** c1:.2f} / {255 * 2 ** c2:.2f} (x255), c_2 - c_1 = {c2 - c1:.3f} stops; "
          f"mean shares {zd['shares'].mean(0).tolist()}")


def test_4_monotone():
    """For random curves and fixed per-pixel weights, the output never decreases as a channel rises (float64, dense
    sweep); black maps to 0; 1 maps to 1 when s_k >= 0 and below 1 when s_k < 0."""
    images = _photos(n=4, seed=3)
    z, _, _ = _pair(images, zones=2, seed=3, s_scale=0.6)
    z = z.double()
    g = _gen(4)
    with torch.no_grad():
        z.theta.copy_(1.5 * torch.randn(2, K, generator=g, dtype=torch.float64))
    rows, L = 6, 20_001
    w_lit = torch.rand(rows, 1, generator=g, dtype=torch.float64).expand(rows, L)  # one weight per row (= pixel)
    z.weights = torch.stack([1 - w_lit, w_lit])[None].expand(z.N, 2, rows, L).contiguous()
    sweep = torch.linspace(0, 1, L, dtype=torch.float64)
    worst = 0.0
    for c in range(3):
        px = torch.rand(1, rows, 1, 3, generator=g, dtype=torch.float64).expand(1, rows, L, 3).clone()
        px[..., c] = sweep
        for k in range(z.N):
            out = z.map_pixels(px, k)[..., c]
            worst = min(worst, float(torch.diff(out, dim=-1).min()))
    assert worst >= -1e-12, f"the output decreases somewhere (smallest step {worst:.2e})"
    s = z.s.detach()
    assert bool((s > 0).any()) and bool((s < 0).any()), "test needs views with s_k > 0 and s_k < 0"
    for k in range(z.N):
        black = z.map_pixels(torch.zeros(1, rows, L, 3, dtype=torch.float64), k)
        white = z.map_pixels(torch.ones(1, rows, L, 3, dtype=torch.float64), k)
        assert float(black.abs().max()) == 0.0, f"view {k}: black is not black"
        if float(s[k]) >= 0:
            assert float((white - 1).abs().max()) <= 1e-12, f"view {k} (s >= 0): 1 does not map to 1"
        else:
            assert float(white.max()) < 1.0, f"view {k} (s < 0): 1 maps to 1"


def test_5_histogram():
    """All-ones weights reproduce torch.histc's counts exactly and compute_histogram_equalization's target, on bin-edge
    levels and on three real training photos."""
    hp = HistogramPriorLoss()

    def check(img, tag):
        grey = torch.mean(torch.nn.functional.interpolate(img.permute(0, 3, 1, 2), scale_factor=0.25), dim=1).flatten()
        want = torch.histc(grey, bins=255, min=0.0, max=1.0)
        hists, shares = zone_histograms(img, torch.ones(1, *img.shape[1:3]))
        assert torch.equal(hists[0], want), f"{tag}: counts differ in {int((hists[0] != want).sum())} bins"
        assert float(shares[0]) == 1.0, f"{tag}: share {float(shares[0])}"
        assert torch.equal(zone_targets(hists)[0], hist_target(hp.compute_histogram_equalization(img))[0]), \
            f"{tag}: target differs"

    # 4 x 4 blocks of one value each, so the 1/4-size nearest sampling sees every value: every grey level l/255, grey
    # means of random level triples (values like (l1 + l2 + l3) / 765 sit on or near bin edges), 0 and 1
    g = _gen(5)
    lv = torch.arange(256, dtype=torch.float32) / 255
    vals = torch.cat([lv[:, None].expand(256, 3), torch.randint(0, 256, (3840, 3), generator=g).float() / 255])
    img = vals.view(64, 64, 3).repeat_interleave(4, 0).repeat_interleave(4, 1)[None]
    check(img, "bin-edge levels")
    pos, keep = hist_bins(torch.tensor([0.0, 1 / 255, 254 / 255, 1.0, 1.5, -0.1]))
    assert pos[:4].tolist() == [0, 1, 254, 254] and keep.tolist() == [True] * 4 + [False] * 2, \
        f"bins of edge values {pos.tolist()}, kept {keep.tolist()}"
    from rung2_preview import load_set
    d = load_set("buu", "lom")
    for i in (0, len(d["images"]) // 2, len(d["images"]) - 1):
        check(d["images"][i], f"buu view {d['names'][i]}")
    print(f"    histc counts and targets equal on 4096 edge levels and 3 real photos ({tuple(d['images'][0].shape[1:3])})")


def test_6_exposure_invariance():
    """Scaling a photo by 2^u and its s_init by -u leaves the guide and the weights unchanged (to 1e-6; no clipping).
    The guide is float32, so it agrees to about one ulp (4.8e-7 at |g| in [4, 8)); the weights scale that by
    1 / (c_2 - c_1), so the centres come from the structured test photos (about 2 stops apart; the LOM scenes have
    0.62 to 1.34), not from a noise image whose blurred guide is nearly constant."""
    images = _photos(seed=6)
    c1, c2 = centres([guide(img, float(s), 16) for img, s in zip(images, _s_init(images))])
    s, u = 0.4, 0.7  # brightest level about 0.18, so 2^0.4 x 0.18 < 1 and 2^0.7 x 0.18 < 1: nothing clips
    err_g = err_w = 0.0
    for img in images:
        g1, g2 = guide(img, s, 16), guide(img * 2 ** u, s - u, 16)
        err_g = max(err_g, float((g1 - g2).abs().max()))
        err_w = max(err_w, float((lit_weight(g1, c1, c2) - lit_weight(g2, c1, c2)).abs().max()))
    assert err_g <= 1e-6 and err_w <= 1e-6, f"guide differs by {err_g:.2e}, weights by {err_w:.2e}"
    print(f"    guide max |diff| {err_g:.1e}, weights max |diff| {err_w:.1e} (c_2 - c_1 = {c2 - c1:.2f} stops)")


def test_7_gradients():
    """map_pixels never carries gradient; the loss trains both curves, s_k (only that view's) and rho; the zone
    constants are buffers without gradient."""
    images = _photos(seed=7)
    z, _, _ = _pair(images, zones=2, seed=7)
    z = z.double()
    k = 1
    assert not z.map_pixels(images[k].double(), k).requires_grad, "the pseudo-target carries gradient"
    for name in ("weights", "zone_targets", "zone_shares", "centres"):
        assert not getattr(z, name).requires_grad, f"{name} requires gradient"
    assert bool((z.zone_shares[k] > 0.05).all()), "test view needs both zones"
    loss, _ = z.loss(z.composed_curves(k), z.zone_targets[k:k + 1], z.zone_shares[k:k + 1], 3000, "low", 0.1)
    loss.backward()
    assert bool((z.theta.grad.abs().amax(dim=1) > 0).all()), "a curve gets no gradient"
    others = [i for i in range(z.N) if i != k]
    assert float(z.s.grad[k]) != 0.0 and float(z.s.grad[others].abs().max()) == 0.0, "wrong s gradient"
    assert float(z.rho.grad.abs()) > 0, "rho gets no gradient"


# ------------------------------------------------------------------ added with the code
def test_8_build_state_refusals():
    images = _photos(seed=8)
    s0 = _s_init(images)
    a, b = build_zones(images, s0, 16), build_zones(images, s0, 16)
    for key in a:
        assert torch.equal(a[key], b[key]), f"build_zones is not deterministic in {key}"
    z, _, _ = _pair(images, zones=2, seed=8)
    n = len(images)
    assert z.n_params() == 2 * K + n + 1 == sum(p.numel() for p in z.parameters()), f"n_params {z.n_params()}"
    assert sum(p.numel() for p in z.f_params()) == 2 * K + 1, "f_params is not theta and rho"
    assert z.strictly_increasing(), "curves are not strictly increasing"
    summ = z.summary()
    assert summ["zones"] == 2 and len(summ["heights_zones"]) == 2 and len(summ["zone_shares"]) == n
    assert "heights" not in summ and "F_grid" not in summ, "single-curve keys in the Rung 3 summary"
    for attr in ("F", "heights", "composed_curve"):
        assert not hasattr(z, attr), f"Rung 3 model has Rung 2's {attr}: a Rung 2 reader would not stop"
    sd = z.state_dict()
    assert {"centres", "zone_targets", "zone_shares", "theta", "s", "rho"} <= set(sd) and "weights" not in sd, \
        f"state keys {sorted(sd)}"
    knots = make_knots(K, "quadratic")
    z2 = ZoneExposureCurve(knots, init_heights("identity", knots), s0, a)
    z2.load_state_dict(sd)
    assert torch.equal(z2.zone_heights(), z.zone_heights()) and torch.equal(z2.s, z.s), "state round trip"
    assert _raises(lambda: ExposureCurve(knots, init_heights("identity", knots), s0, prior="F").load_state_dict(sd),
                   RuntimeError), "a Rung 2 model accepted a Rung 3 state"
    flat = [torch.full((1, H, W, 3), 0.05) for _ in range(3)]
    assert _raises(lambda: build_zones(flat, torch.zeros(3), 16)), "constant photos accepted (c_2 - c_1 = 0)"
    assert _raises(lambda: build_zones(images[:2] + [images[2][:, :-1]], s0[:3], 16)), "mixed sizes accepted"
    assert _raises(lambda: build_zones(images, s0[:-1], 16)), "photo and exposure counts differ, accepted"
    assert _raises(lambda: build_zones(images, s0, 200)), "a blur larger than the photo accepted"
    assert _raises(lambda: build_zones(images, s0, 16, zones=3)), "three zones accepted"
    assert _raises(lambda: z.map_pixels(images[0], 0, "s")), "attach s accepted"
    assert _raises(lambda: z.map_pixels(images[0], 0, "none", "luminance")), "luminance accepted"
    assert _raises(lambda: z.map_pixels(images[0][:, :50], 0)), "pixels of another size accepted"
    assert _raises(lambda: ZoneExposureCurve(knots, init_heights("identity", knots), s0, a, prior="none")), \
        "prior none accepted"


def test_9_loss_weighting_and_batch():
    """Per view the loss is sum_m pi_m curve_loss(zone m); over several views it is the mean of the single-view losses."""
    images = _photos(seed=9)
    z, _, _ = _pair(images, zones=2, seed=9)
    with torch.no_grad():
        z.theta[1].add_(0.5 * torch.randn(K, generator=_gen(10)))  # two different curves
    k = 2
    curves = z.composed_curves(k)
    shares = torch.tensor([[0.25, 0.75]])
    total, parts = z.loss(curves, z.zone_targets[k:k + 1], shares, 3000, "low", 0.1)
    p, Fg = z.prior_curve(), z.F_zones(z.grid)
    want = sum(w * curve_loss(curves[:, m], z.zone_targets[k:k + 1, m], 3000, "low", 0.1, prior=p,
                              prior_x=Fg[m][None])[0] for m, w in enumerate((0.25, 0.75)))
    assert abs(float(total) - float(want)) <= 1e-7, f"share weighting: {float(total)} vs {float(want)}"
    assert set(parts) == {"cdf", "pseudo", "smooth", "w_cdf"}, f"parts keys {sorted(parts)}"
    views = torch.arange(z.N)
    batch, _ = z.loss(z.composed_curves(views), z.zone_targets, z.zone_shares, 3000, "low", 0.1)
    single = np.mean([float(z.loss(z.composed_curves(i), z.zone_targets[i:i + 1], z.zone_shares[i:i + 1], 3000,
                                   "low", 0.1)[0]) for i in range(z.N)])
    assert abs(float(batch) - single) <= 1e-6, f"batch loss {float(batch)} vs mean of single views {single}"


TESTS = [test_1_m1_is_rung2, test_2_equal_curves_half_weights, test_3_weights, test_4_monotone, test_5_histogram,
         test_6_exposure_invariance, test_7_gradients, test_8_build_state_refusals, test_9_loss_weighting_and_batch]


# ------------------------------------------------------------------ test 2: the CUDA op check
def cuda_check():
    """The convex mix, the composed curves and the loss on the GPU against the CPU (to 1e-6), with the image index as
    the trainer passes it (a CUDA LongTensor), and the buffers moved unchanged."""
    if not torch.cuda.is_available():
        raise SystemExit("--cuda: no CUDA device")
    images = _photos(seed=11)
    z, _, _ = _pair(images, zones=2, seed=11)
    zc = ZoneExposureCurve(z.knots, init_heights("identity", z.knots), z.s_init,
                           {"weights": z.weights, "targets": z.zone_targets, "shares": z.zone_shares,
                            "centres": z.centres}).to("cuda")
    zc.load_state_dict(z.state_dict())
    for name in ("weights", "zone_targets", "zone_shares", "centres", "knots", "grid"):
        assert torch.equal(getattr(zc, name).cpu(), getattr(z, name)), f"{name} changed on the way to the GPU"
    worst = {"pseudo-target": 0.0, "composed curves": 0.0, "loss": 0.0}
    for k, img in enumerate(images):
        ids = torch.tensor([k], device="cuda")
        worst["pseudo-target"] = max(worst["pseudo-target"],
                                     float((zc.map_pixels(img.cuda(), ids).cpu() - z.map_pixels(img, k)).abs().max()))
        cg, cc = zc.composed_curves(ids), z.composed_curves(k)
        worst["composed curves"] = max(worst["composed curves"], float((cg.detach().cpu() - cc.detach()).abs().max()))
        lg = zc.loss(cg, zc.zone_targets[ids], zc.zone_shares[ids], 3000, "low", 0.1)[0]
        lc = z.loss(cc, z.zone_targets[k:k + 1], z.zone_shares[k:k + 1], 3000, "low", 0.1)[0]
        worst["loss"] = max(worst["loss"], abs(float(lg) - float(lc)))
        lg.backward()
    assert zc.theta.grad is not None and zc.s.grad is not None and zc.rho.grad is not None, "no gradient on the GPU"
    bad = {k: v for k, v in worst.items() if v > 1e-6}
    assert not bad, f"GPU differs from CPU: {bad}"
    print("cuda check passed: buffers moved unchanged; GPU vs CPU max |diff| " +
          ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))


def main():
    torch.set_num_threads(2)  # light enough to run beside a training batch
    if "--cuda" in sys.argv[1:]:
        cuda_check()
        return
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
