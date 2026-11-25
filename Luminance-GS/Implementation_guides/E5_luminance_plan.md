# E5 plan: the colour-preserving variant on the identity base (before Rung 3)

Written 2026-11-25 by Claude Code, after Bart's decisions of the same day (log: "Decisions after M5"). It replaces `M5_plan.md` step 6. **Status: draft for Bart's approval.** No code and no run before his OK; the readings in section 4 are logged before any run (rule 4).

## 1. Question and why now

- **Question:** is bike's hue gap on the varying-exposure set caused by applying F to R, G and B separately?
  - On the varying set: `rung2id` 27.0° ± 5.6, `detach` 12.3° ± 1.0 (*measured*, seeds 0–2).
- **Why before Rung 3:** fix the known weakness in the base first, so whatever Rung 3 then shows is its own effect.
- **Base:** `rung2id`, M6's base (decided 2026-11-25). The comparison is `rung2idlum` − `rung2id`. Every comparison run already exists:
  - `rung2id` on LOM, seeds 0–2 (`results/rung2/`);
  - on the varying set (E1);
  - on overexposure (E6b).
- **Overexposure:** an exploratory check joins the batch (Bart). It cannot change the H1-over verdict.

## 2. Design: `--rung2_color {rgb, luminance}` (default `rgb` reproduces the current code exactly)

```
Y      = (R + G + B) / 3                                   the channel-mean grey
out_c  = clamp( x_c · F(min(1, 2^{s_k} Y)) / max(Y, 1e-6), 0, 1 )   c ∈ {R, G, B}
```

- **Why this grey:**
  - The curve loss's target is already the HE-CDF of exactly this grey: `losses.py` line 56 averages the three channels before the histogram. So F is applied to the quantity it was fitted on.
  - Before clipping, the output's channel mean is exactly F(2^{s_k} Y). The pseudo-target's grey levels are then the equalised ones the curve loss aims at; per-channel application only approximates this (Jensen).
  - Rec.709 luminance was considered and not chosen: F would then act on a different grey from the one its target is built on.
- **What it changes, and what it does not:**
  - The channel ratios are kept, so the HSV hue and saturation of each pixel are kept (before clipping). A grey pixel gets exactly what `rgb` gives it.
  - **F, s_k and γ are untouched.** In the detached default they are trained only by the curve loss on the photos, which never sees the pseudo-target.
  - So `rung2idlum` and `rung2id` must have identical F and s at the same seed: the same built-in check as `rung2` against `rung2exp` in M3 (max |Δ| ≤ 1e-5). Any difference in results is the colour handling alone.
- **Clipping** is per channel at 1, as a camera saturates. Bright saturated pixels can clip in low light (the scale factor is large there), and those lose some hue. The analysis reports the share of clipped channel values.
- **Diagnostics:** the `projected` key becomes (Y, Y, Y), F's real input, so `bin_hist` in `curves_step*.npz` follows.
- **Attached modes:** `--rung2_attach s|all` still work (the gradient goes through y = 2^{s_k} Y), but E5 does not use them.
- **Refusals:** `--rung2_color luminance` without `--curve_model rung2` is refused.
- **Runner variant:** `rung2idlum` = `["--curve_model", "rung2", "--rung2_F_init", "identity", "--rung2_color", "luminance"]`, added to `NOT_IN_RUNG0`. On the varying set it is the shared column, like `rung2id`.

## 3. Code and tests (after Bart's OK; about half a day, *inferred*)

| File | Change |
|---|---|
| `rung2.py` | `map_pixels(..., color="rgb")`; the luminance branch in `_map` |
| `simple_trainer_ours.py` | `Config.rung2_color`; pass it to `map_pixels` in `rung2_forward`; `projected`; the refusal |
| `diagnostics.py` | `MODEL_DEFAULTS["rung2_color"] = "rgb"` (the trainer asserts it matches `Config`) |
| `rescore_runs.py` | `"rung2_color"` added to `MODEL_FLAGS` |
| `run_rung0.py` | the variant `rung2idlum` and its `NOT_IN_RUNG0` entry |
| `test_rung2.py` | new tests (below) |
| `rung2_ablation.py` | an `EXPECTED_ALL` entry for `rung2idlum`; a new `--mode lum` (section 5) |
| `docs/curve_estimation.md`, `CLAUDE.md` | the flag, the variant and the repo map |

**Tests, all before the batch (rule 3):**
1. **Unit tests** (CPU), in `test_rung2.py`. All existing tests still pass, and:
   - `rgb` reproduces the current `map_pixels` bit for bit;
   - grey pixels give the same output in both modes (to 1e-7);
   - channel ratios are kept where nothing clips;
   - black stays black, and the output stays in [0, 1];
   - gradients: none with `none`, s only with `s`, θ and s with `all`.
2. **Regression:** a 300-step `rung2id` smoke on the varying set (buu, seed 0) with the new code. It must match the existing `results/vexp/smoke_m5/buu_rung2id_seed0` in F, s and pseudo-targets (M3's tolerance 1e-5).
3. **Smoke runs:** 300 steps, buu, seed 0, in `smoke_e5/` under each root, each re-scored with `--ref_root ../results/rung0`:
   - `rung2idlum` on LOM, the varying set and `over_exp`;
   - plus a `rung2id` LOM smoke as its partner.

   Check, against the `rung2id` smoke of the same set: F and s are identical; the pseudo-targets differ; the `over_exp` run reads `over_exp/`.
4. `rung2_ablation.py --mode lum --selftest` gives every effect exactly 0.

## 4. Pre-registered readings (logged before any run, once Bart confirms)

General rules as in M5:
- pair by seed, seeds 0–2;
- noise = sqrt(mean of the arms' seed variances);
- an effect is real above 2× noise;
- pooled = the five-scene mean, checked against pooled without shrub;
- shrub never decides.

**Primary reading (M5 plan step 6, now on the identity base).** Comparison `rung2idlum` − `rung2id`:
- **Fixes bike's hue:** bike's hue error on the varying set falls by more than 2× noise, and no decision scene (buu, chair, sofa, bike) loses bm PSNR beyond 2× noise, on LOM or on the varying set.
- **Fixes at a cost:** the hue falls beyond 2× noise, but some decision scene loses bm PSNR beyond 2× noise.
- **No fix:** anything else, including a hue change within 2× noise.

**Adoption into M6's base.** `rung2idlum` replaces `rung2id` as M6's base only if the verdict is "fixes" **and** pooled SSIM and LPIPS are not worse beyond 2× noise on either set. Otherwise M6 builds on `rung2id`, and E5 is reported as an ablation.

**Reported without a decision:**
- every metric on both sets, per scene and pooled;
- the hue error in every scene, the saturation ratio, the angle error and the pseudo-target bm PSNR;
- the share of clipped channel values;
- the H1 rule for `rung2idlum` − `detach` on LOM. This is exploratory: it changes neither H1 nor H1b.

**Overexposure (exploratory).** It cannot change H1-over, and nothing is tuned on it.
- Comparison: `rung2idlum` − `rung2id` on `over_exp`.
- **"Colour accounts for part of the overexposure gap"** if pooled bm PSNR rises beyond 2× noise. Report what fraction of `rung2id`'s pooled gap to `detach` (−1.20 dB) it closes, and the saturation ratios.
- **The direction is not predicted:**
  - every method oversaturates here (saturation ratio 1.15–1.64 on chair, sofa and bike);
  - per-channel darkening re-saturates washed-out photos;
  - luminance scaling keeps each photo's own saturation.

## 5. What E5 can and cannot show (state it in the results)

- **What it isolates:** how F is applied. F, s and γ are identical by construction, and this is checked.
- **The detection limit:**
  - `rung2id`'s bike hue sd on the varying set is 5.6° (*measured*), so 2× noise is roughly 8–11°, depending on the new arm's spread (*inferred*).
  - The gap to `detach` is 14.7°.
  - So a partial fix smaller than about 8° reads "no fix".
- **Part of the gap is not Rung 2's:** `detach`'s own bike hue error falls from 31.8° on LOM to 12.3° on the varying set (*measured*; cause not established). E5 can remove only the part caused by per-channel application.
- **A risk to expect (*inferred*):** luminance scaling keeps the dark photo's colour noise and multiplies it by the brightening factor. Per-channel F compresses chroma, which pales colours but also hides noise. LPIPS could get worse even if hue improves; the adoption rule checks this.

## 6. Runs, analysis and schedule

**Batch:** 45 runs, about 3.5 h (*inferred* from M5's per-run minutes: 4.4 min on LOM and on the varying set, 5.1 min on `over_exp`). Ask Bart before the launch, and keep the PC awake (M5's helper, pointed at the new launcher). From `examples/`, in this order:
```
python run_rung0.py --root ../results/rung2 --variants rung2idlum --seeds 0 1 2
python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants rung2idlum --seeds 0 1 2
python run_rung0.py --exp_name over_exp --root ../results/overexp --variants rung2idlum --seeds 0 1 2
```
Then re-score the three roots with `--ref_root ../results/rung0` (about 10 s per run).

**Analysis:**
- **`rung2_ablation.py --mode lum`** → `results/rung2/summary_e5/`. It covers the paired comparison on all three sets, the built-in F/s check, the readings above and the clipped share (computed on the CPU from each run's final F and s and the training photos).
- **The exploratory H1 checks**, with the tested `h1` mode:
  - `--mode h1 --a rung2idlum --a_roots ../results/rung2 --b detach --b_roots ../results/rung0 --official_roots ../results/rung0 --out ../results/rung2/summary_e5_h1`
  - the same on `over_exp`, writing `results/overexp/summary_e5/`.
- **Write-up:** one log entry, and a short `E5_results.md`.

**Schedule** (*inferred*):

| When | What |
|---|---|
| Wed 25 Nov | Bart approves; readings logged; code and tests |
| Wed evening or Thu 26 | Batch, then re-score and analysis; the M6 base is decided by the adoption rule. `M6_rung3_plan.md` is written while the batch runs |
| Fri 27 | **Cutoff:** if E5 has not launched by then, Rung 3 goes ahead on `rung2id` |
