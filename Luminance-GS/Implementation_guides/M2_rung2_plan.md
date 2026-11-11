# M2: build Rung 2 behind flags, with an exposure-aware dark branch

Plan for Claude Code on Bart's PC. Written 2026-11-09 from `M1_vexp_results.md`, `M1_vexp_plan.md`, `Rung0.5_interpretation.md`, the proposal rationale (sections 5 and 6), the explainer's section 8.6 and an audit of the code on `main` at commit `c5fa94c` (M1 done). It records Bart's 2026-11-09 decision on the dark branch (section 1). It covers M2 (build and smoke-test) and also specifies the M3 runs and reporting (sections 5 and 6), so M3 needs no new plan unless something here changes.

In the repository this file lives at `Luminance-GS/Implementation_guides/M2_rung2_plan.md`. A copy is in the project's shared files as `docs/M2_handoff.md`.

**Reviewed 2026-11-10 by Claude Code on Bart's PC** against the code at `c5fa94c` (read-only, plus in-memory checks: torch 1.13.1 has `torch.exp2` and `searchsorted`; tyro 0.9.1 parses `--dark_branch measured` and rejects bad values). Part A matches the code. Folded in below, each marked "review":
- **Fixes:** the sign in unit test 9 (step 5); milestone numbers (the deck has M4 = the 13 Nov check-in and M5 = H3 and ablations, so the ablations here are M5); the LOM row in 2.1 (M1 found the split between content and exposure unknown); the guard order in step 1.
- **Additions:** the gradient re-weighting caveat and `num_GS` in R6 (2.1, 6.2); a ratio-based speed estimate (8.4, 5.1); same-seed repeats as the reference for the determinism checks (8.2, 8.4, 6.1); notes on reusing the analysis helpers (6, 6.6); the prompt for the M3 command chain (5.1); the `--ref_root` rule in the run-batch skill and `rescore/exposure_init.json` in the commit list (step 9).
- **Open, for Bart:** the dark-end alignment of the curve loss (2.2, before part C) and its pre-declared fallback (step 6); the H1 rule (6.5 against the explainer's 8.11); whether overexposure is in H1 (section 9, a question for Wei).
- **Branch:** `main` (Bart, 2026-11-10). The doc updates that do not depend on new code were made on 2026-11-11, before part A; step 9 lists what is left.

**Rules (from `CLAUDE.md`; they override anything below):**
- Never `git commit`, `push`, merge, rebase, reset, stash or switch branches. Ask Bart before creating a branch (M1 worked on `main`; if he wants one, suggest `m2-rung2`).
- Ask before any job longer than about 30 minutes, with a time estimate. Before anything over an hour, remind him to stop the PC sleeping.
- New behaviour goes behind a `Config` flag whose default reproduces the current code. Smoke-test (about 300 steps) before any batch.
- Log every step with `/log-experiment`. Leave finished results alone: `results/rung0/summary*/`, `results/vexp/summary/`, `Rung0_results.md`, `Rung0.5_results.md`, `M1_vexp_results.md`.
- Run Python with the `luminancegs` interpreter from `CLAUDE.local.md`.
- **While a batch runs, create new files only.** `run_rung0.py` starts every run as a fresh `python simple_trainer_ours.py`, so editing the trainer or anything it imports (`diagnostics.py`, `losses.py`, `tools.py`, `utils.py`, `datasets/`) changes the runs still in the queue.

---

## 1. M2 in plain terms

**What it is.** Two code changes behind flags, their tests, and the first batch:
1. **An exposure-aware dark branch** (`--dark_branch`, off by default). View k's dark render is scaled by 2^{−s_k} so that it matches a photo taken at a different exposure.
2. **Rung 2** (`--curve_model rung2`). One shared monotone curve F plus one exposure s_k per view, `L_k(x) = F(min(1, 2^{s_k} x))`, replacing the official curve, bias network, shape network and colour basis.

**Why the dark-branch switch (Bart chose option A on 2026-11-09).**
- Luminance-GS renders every view twice from the same Gaussians. The dark render must match the dark photo (`loss_regress_low`); the bright render must match the pseudo-target.
- Each Gaussian has one dark colour, the same for every view. On LOM the dark photos agree, and the dark render fits them at 46.2 dB (pooled, `detach`).
- On the varying set the photos differ by up to 2 stops. One dark colour cannot match them all, and the fit drops to 38.8 dB (M1 Q4, *measured*).
- Rung 2 replaces only the curve, so it would inherit this misfit. A weak H2 result could then not be pinned on either the curve model or the dark branch.
- The switch reuses the one-number-per-photo idea: no new parameters.

**The 2×2 on the varying-exposure set.**

| | Shared dark branch (official) | Exposure-aware dark branch |
|---|---|---|
| **`detach` curve** | `detach` (M1, done) | `detachexp` (M2 part A): s_k = measured log2(m̄/m_k), fixed |
| **Rung 2 curve** | `rung2` (M3) | `rung2exp` (M3): Rung 2's own s_k, stop-gradient |

- Comparing rows gives the curve model's effect. That is H2.
- Comparing columns gives the dark branch's effect: how much of the varying-exposure cost it causes.
- The interaction says whether the curve model's gain depends on the dark-branch fix.
- The analysis is the factorial one from Rung 0.5 (`rung05_analyze.effects`).

The stop-gradient keeps H3 separate. In `rung2exp` the dark loss never trains s_k, so s_k is trained only by the curve loss (default) and, in the H3 ablation, by the image loss.

**What stays as approved.**
- H1 is still Rung 2 against `detach` on LOM, with the shared dark branch (switch off).
- Every default reproduces the official code; `baseline` stays in every table as "official".
- The option-A extension goes slightly beyond the approved plan; mention it to Wei on 13 Nov (section 9).

**Milestones.** M2 = parts A to C below. M3 = the runs, re-scoring, `rung2_analyze.py` and `Rung2_results.md` (sections 5 and 6). M4 = the 13 Nov check-in with Wei. M5 = ablations, H3 first (review: the deck's numbering; this plan first said M4).

**Target dates** (times *inferred* from M1's run times):

| When | What | GPU |
|---|---|---|
| Day 1 | Part A: switch, checks, smoke test. Launch `detachexp` on the varying set. | 15 runs, about 2 h |
| Day 2 | Part B while the batch runs (new files only): `rung2.py`, tests, preview. Part C after it: wire Rung 2 in, regression checks, smoke tests. | smoke only |
| Night of day 2 | M3 batch, seed-major: seed 0 of all three new configurations first, then seeds 1 and 2 | 45 runs, about 6 h |
| Day 3 | Re-score, `rung2_analyze.py`, first numbers to Wei | |
| Fri 13 Nov | Check-in with Wei: at least seed 0 of everything | |

---

## 2. Design (agreed; ask Bart before changing any of it)

### 2.1 Exposure-aware dark branch

| Item | Choice | Reason |
|---|---|---|
| Where | In `train()`, multiply view k's rendered dark image by 2^{−s_k} right after rasterisation, before `loss_regress_low`. Nothing in `gsplat/` changes. | Alpha compositing is linear in the Gaussians' colours and the background is black, so scaling the image equals scaling every Gaussian's dark colour for that view. |
| Flag | `--dark_branch {shared, measured, model, gains}`, default `shared` | `shared` is the official code. |
| `measured` | s_k = log2(m̄/m_k): m_k is view k's mean as the trainer loads it (undistorted, all pixels and channels, /255), m̄ the geometric mean, so the mean is 0. Fixed for the whole run. On the varying set, equal to `oracle.json["s_init"]`. | Bart's choice for the `detach` row: works for any photo set, nothing learned. |
| `model` | Rung 2's current s_k, `.detach()` | The dark loss must not train s_k (keeps H3 separate). Needs `--curve_model rung2`. |
| `gains` | s_k = −(log2_ratio_k − mean), the realised gains in `<data_dir>/exposure_gains.json` | An oracle for diagnosis only: it uses the true gains, so it is not a method. A few lines to build; run only if needed (5.3). |
| Gauge | Every source has mean 0 over the training views | The dark render then shows the scene at the views' geometric-mean exposure, the same gauge as Rung 2's s_k. |
| Test views | Unchanged | Metrics use only the enhanced render, and test views have no s_k. |
| LOM | Switch off for H1 | LOM views are evenly exposed: their measured s_k has sd 0.08 to 0.20 stops (*measured*), and how much of that is exposure and how much is content is unknown (M1). So scaling may add error there (*inferred*; review: this row first said "mostly content"). |

**What it cannot fix** (*inferred*).
- The measured s_k mixes exposure with content. On the varying set its error against the true gains is the unpaired init RMSE: 0.08 to 0.19 stops (M1 plan, step 1 table), about 6 to 14% in brightness.
- So `detachexp`'s dark fit should climb from 38.8 dB toward LOM's 46.2 dB but probably not reach it. Rung 2's s_k ends near the histogram oracle, whose unpaired RMSE (0.07 to 0.17) is similar.
- 8-bit quantisation of the darkened views is not recoverable by any cell (M1 caveat).
- `gains` removes the estimation error, which is why it exists.

**A built-in check** (*follows from the code*).
- With a detached curve, nothing in the curve path sees the dark render. The switch also draws no random numbers, so the view order is unchanged.
- So `detach` and `detachexp` with the same seed have the same pseudo-targets, up to GPU float noise in the bias network. `rung2` and `rung2exp` have identical F, s_k and pseudo-targets.
- The column effect therefore acts only through the shared Gaussians. Steps 3 and 8 and section 6.1 check this.
- It acts on them in two ways (review, *follows from the code*). The dark target now matches each photo's exposure, and view k's `L_low` gradient is multiplied by 2^{−s_k}: ½× to 2×, arithmetic mean about 1.08 on this set. That also changes the screen-space gradients that drive densification. So report `num_GS` per cell (R6) before reading the column effect as "better fit" alone (*inferred*).

### 2.2 Rung 2 (explainer 8.6 and the proposal rationale, with the open details fixed)

| Component | Default | Flag (default) | Notes |
|---|---|---|---|
| Curve model | official | `--curve_model {official, rung2}` (`official`) | `rung2` does not build L_g, the bias network, the shape network, `pesdo_curve` or the colour basis, nor their optimisers. |
| F | Monotone piecewise-linear with K segments. Knots y_0 = 0 < … < y_K = 1. Heights h = [0, c / c_K] with c = cumsum(softplus(θ)), so h_0 = 0 and h_K = 1 exactly. | `--rung2_knots` (32) | Strictly increasing, F(0) = 0 and F(1) = 1 by construction. |
| Knot spacing | quadratic, y_j = (j/K)² | `--rung2_knot_spacing {quadratic, uniform, usedrange}` (`quadratic`) | `usedrange`: y_j = y_max·(j/(K−1))² for j < K and y_K = 1, with y_max the 99.9th percentile of min(1, 2^{s_k⁰} x) over all training pixels and channels. The default may switch to `usedrange` by the preview rule in step 6. |
| Lookup | Linear interpolation of F at y = min(1, 2^{s_k} x), applied to R, G and B | none | Replaces the truncated 255-entry LUT, so gradient reaches y and therefore s_k. |
| F initialisation | `cdf`: each view's HE-CDF aligned by its initial s_k (CDF_k(2^{−s_k⁰} y), 1 above the photo's range), averaged over views and sampled at the knots. Increments floored at 1e-4, then renormalised. | `--rung2_F_init {cdf, identity}` (`cdf`) | Close to where the curve loss drives F anyway; the preview checks how fast `identity` gets there (risk 5 in section 7). |
| s_k | A free scalar per view, initialised at log2(m̄/m_k), the same numbers as `measured` | `--rung2_s_init {measured, zero}` (`measured`) | `zero` is the proposal's "compare both" ablation (M5). |
| Gauge | After every optimiser step, subtract the mean of s | none | mean s = 0, so s_k is relative exposure and F carries the overall brightening. |
| Curve loss | L_curve = w(t)·mean_b (L_k[b] − CDF_k[b])² + λ·mean_b (L_k[b+1] − L_k[b])², with L_k[b] = F(min(1, 2^{s_k} x_b)) and x_b = b/254. CDF_k from `HistogramPriorLoss.compute_histogram_equalization`. w = 1, then 0.5 from step 3000 (0.1 for `over_exp`). The total loss uses 10·L_curve where the official uses 10·L_hist. | `--rung2_smooth_lambda` (0.1) | The official L_hist minus the shape-prior term. With F detached, Rung 2 then differs from `detach` only in the curve model. The rationale's slope-change smoother needs a new weight to tune, so it stays an ablation. |
| Optimisers | Adam for θ and Adam for s, lr 1e-3 each, weight decay 0, ExponentialLR to 1% over the run (as the official curve) | `--rung2_lr_F`, `--rung2_lr_s` (1e-3) | No weight decay: on s it would pull every view toward "no exposure correction". The preview checks convergence before M3. |
| Image loss into F and s | None (detached) | `--rung2_attach {none, s, all}` (`none`) | `s` is the H3 ablation: F's knot heights detached, y differentiable, so ∂F(2^{s_k}x)/∂s_k = F′(y)·y·ln 2. `all` is the attached-F ablation. |
| Colour | Fixed RGB basis: F is applied to each channel | none | As in `detach`, whose basis never moves. |
| Not built in M2 | Colour-preserving luminance variant (reserve `--rung2_color luminance`), slope-change smoother, one global gamma prior | | M5 |

- **Parameter count:** K + N = 32 + N, which is 54 to 75 for the LOM scenes (22 to 43 training views). The official curve machinery has 255 + 143,743 + 111,235 + 5N. Print the count at start-up and store it.
- **Flag guard:** with `--curve_model rung2`, assert that `--detach_pseudo_target`, `--detach_curve_in_target` and `--freeze_color_basis` are False. Rung 2 controls its own gradient flow through `--rung2_attach`.
- **Known systematic difference** (*follows from the code*). The official LUT truncates: level 0 and level 1 both read `curve[0]`, and level ℓ ≥ 1 reads `curve[ℓ−1]` ≈ CDF[ℓ−1]. Rung 2 maps level 0 to black (F(0) = 0) and level ℓ to about CDF[ℓ].
  - So Rung 2's pseudo-targets are black at level 0 and about one CDF bin brighter elsewhere than `detach`'s.
  - That shifts raw brightness, which brightness-matched PSNR corrects, and the darkest shadows. See risk 1 in section 7.
  - **Open, for Bart before part C (review).** A one-bin shift of the target removes most of this difference: fit F(min(1, 2^{s_k} x_b)) to [0, CDF_k[0], …, CDF_k[253]], and use the same target in the `cdf` init.
    - Rung 2 then reads the same CDF bin as `detach` for every level ≥ 1. Level 0 stays black, and F(0) = 0 now matches the target at b = 0, so there is no fixed residual there.
    - The snapshot grid stays x_b = b/254, so the M1 analysis code is unaffected.
    - Recommended, because the one-bin lift is largest where risk 1 sits: bike and shrub, whose dark-end CDFs rise steeply. The preview reports P2, P4 and P5 for both alignments (step 6).

### 2.3 Runs and names

| Variant | Flags | Roots |
|---|---|---|
| `detachexp` | `--detach_pseudo_target --dark_branch measured` | `results/vexp` |
| `rung2` | `--curve_model rung2` | `results/rung2` (LOM), `results/vexp` |
| `rung2exp` | `--curve_model rung2 --dark_branch model` | `results/vexp` |
| `rung2att` | `--curve_model rung2 --rung2_attach s` | defined in M2, run in M5 |

- "exp" in a variant name always means the exposure-aware dark branch.
- LOM Rung 2 runs go to a new root, `results/rung2/`, because rule 6 keeps `results/rung0/` for Rungs 0 and 0.5. Varying-set runs join M1's in `results/vexp/`, so all four 2×2 cells sit in one root.
- Every run is re-scored with the LOM reference depth: `--ref_root ../results/rung0`.

---

## 3. Audit: what the code needs

### 3.1 Changes

| File | Part | Change | Default behaviour |
|---|---|---|---|
| `examples/simple_trainer_ours.py` | A | `--dark_branch`, `Runner.measure_exposures`, `Runner.load_gains`, `Runner.dark_scale`; scale `colors_low` in `train()`; writes `<result_dir>/exposure_init.json` when exposures are computed | Unchanged: nothing is computed or scaled with `shared`. |
| `examples/diagnostics.py` | A | Two new columns at the end of `pseudo_targets.csv`: `s_dark` and `fit_psnr_lowrender_exp_vs_input`. `_load_run_state` asserts that the run's model flags match the re-score's. | Existing columns unchanged. For `shared` runs the new fit column equals the old one. |
| `examples/rescore_runs.py` | A, C | `RUN_RE` variant `[a-z]+` → `[a-z0-9]+`. `MODEL_FLAGS` passed through from each run's `cfg.json`. | Old run names still match, and old `cfg.json` files have none of these keys. |
| `examples/run_rung0.py` | A, C | `VARIANTS`: `detachexp` (A); `rung2`, `rung2exp`, `rung2att` (C) | Existing variants and commands unchanged. |
| `examples/rung2.py` | B | **New.** The Rung 2 curve module, knots, initialisation and curve loss (step 5). | Not imported until part C. |
| `examples/test_rung2.py` | B | **New.** Unit tests (step 5), plain asserts, `__main__` guard. | n/a |
| `examples/rung2_preview.py` | B | **New.** Fits Rung 2's F and s_k from the training photos alone, with no 3DGS, and previews pseudo-targets, leak and dark end (step 6). Writes `results/rung2/preview/`. | n/a |
| `examples/simple_trainer_ours.py`, `diagnostics.py` | C | `--curve_model rung2` and the `--rung2_*` flags wired in (step 7) | Unchanged with `official`. |
| `examples/rung2_analyze.py` | M3 | **New.** Section 6. | n/a |
| `CLAUDE.md`, `docs/experiment_protocol.md`, `docs/curve_estimation.md`, explainer 8.6, `.claude/settings.json` | C | Step 9 | |

### 3.2 Things found in the audit

1. **`rescore_runs.py` would skip every Rung 2 run silently.** Its `RUN_RE` allows only letters in the variant, so `rung2_*` folders never match. `summarize_runs.py` already uses `[a-z0-9]+`.
2. **A re-score rebuilds the model from defaults.** `rescore_runs.py` passes only `data_dir`, `exp_name`, `data_factor` and `test_every`, so a re-score of a Rung 2 or `exp` run would build the official model with a shared dark branch. Pass the model flags from the run's `cfg.json`, and assert them in `_load_run_state`.
3. **`*.pt` is git-ignored**, so `curve_state.pt`, and Rung 2's learned F and s_k in it, stay local. Write `diagnostics/rung2_state.json` and `diagnostics/exposures_by_step.csv` (committed) for the report.
4. **Diagnostics reach into the official modules:** `_curve_modules`, `snapshot_curves` (`curve_bias`, `gamma`, `axis1/2`, `runner.curve`), `_save_curve_state`, `_load_run_state`, and `Runner.grad_check`. Each needs a Rung 2 branch.
   - In Rung 2 mode, `curve_forward` should return the same keys (zeros or NaN where meaningless) plus `s_k`.
   - Return `projected` = (R, G, B), so `snapshot_curves`' bin histogram works unchanged.
   - `curve_adj` = the composed per-view curve on the 255-bin grid, so the M1 analysis code (`curve_adj`, `bin_hist`, `fit_exposure_form`) reads Rung 2 runs as well.
5. **`train()` builds schedulers with `self.curve_optimizers[0]`, `self.adjust_optimizers[0]` and `self.sat_optimizers[0]`.** These are empty in Rung 2 mode. Build the schedulers from one list of curve-side optimisers.
6. **Random numbers.** `measure_exposures` reads `trainset[i]` (no DataLoader, no random draws); keep it that way.
   - Rung 2 does not create the two networks, so its random stream after initialisation differs from `detach`'s. That is fine, because runs are paired by seed number, not by trajectory. Do not claim per-seed trajectory identity between rows.
7. **`L_spa` needs no exposure handling** (*follows from the code*). Its per-view degree c_k = 0.5/mean(I_k) cancels a per-view gain: scaling I_k by g scales its edges by g and c_k by 1/g.
8. **Protected outputs.** `rung0_analyze.write_csv` and `savefig` default to the protected `results/rung0/summary/`. New scripts must pass their own folder and guard it the way `vexp_analyze._inside_out` does.
9. **M1's outputs are now finished results.** Add `Edit(/Luminance-GS/results/vexp/summary/**)` and `Edit(/Luminance-GS/Implementation_guides/M1_vexp_results.md)` to the ask list in `.claude/settings.json`, and both to rule 6 in `CLAUDE.md`.

---

## 4. Steps

### Step 0. Start (10 minutes)

1. Run `git status` and `git log -1`. Confirm the working copy is at `c5fa94c` (or later) and clean; tell Bart if not.
2. Ask Bart about a branch. (Answered 2026-11-10: stay on `main`.)
3. Write a decision entry with `/log-experiment`:
   - option A and the 2×2;
   - the stop-gradient rule;
   - switch off on LOM, H1 unchanged;
   - sources: M1 report Q4 and this plan.

### Part A: the dark-branch switch (about 1 to 1.5 h of work, *inferred*)

#### Step 1. Trainer

```python
# imports
from pathlib import Path
from typing import Literal

# Config, after ref_depth_run
    # M2: dark branch. "shared" is the official code: one dark render serves every view. Otherwise view k's dark
    # render is scaled by 2^{-s_k} before loss_regress_low, with s_k from "measured" (log2(m_bar/m_k), geometric
    # mean, fixed), "model" (Rung 2's own s_k, stop-gradient) or "gains" (the set's true gains; diagnosis only)
    dark_branch: Literal["shared", "measured", "model", "gains"] = "shared"

# Runner.__init__, after self.valset
        # M2: per-view exposures for the dark branch (part C: also for Rung 2's s_k init). Nothing runs in the default.
        if cfg.dark_branch == "model":  # review: checked first, so nothing is written before the exit
            raise SystemExit("--dark_branch model needs --curve_model rung2")   # part C relaxes this
        self.s_measured = self.s_gains = None
        if cfg.dark_branch != "shared":
            self.s_measured = self.measure_exposures()
        if cfg.dark_branch == "gains":
            self.s_gains = self.load_gains()

# new Runner methods
    def train_view_names(self):
        return [self.parser.image_names[int(self.trainset.indices[i])] for i in range(len(self.trainset))]

    @torch.no_grad()
    def measure_exposures(self) -> Tensor:
        """s_k = log2(m_bar / m_k) per training view: m_k = mean of the view as training loads it, m_bar = geometric
        mean, so mean_k s_k = 0. Same numbers as build_vexp's s_init. trainset[i], no DataLoader: no RNG is drawn."""
        n = len(self.trainset)
        m = np.array([float((self.trainset[i]["image"] / 255.0).mean()) for i in range(n)])
        s = np.log2(m).mean() - np.log2(m)
        with open(f"{self.cfg.result_dir}/exposure_init.json", "w") as f:
            json.dump({"view": self.train_view_names(), "mean_input": m.tolist(), "s_measured": s.tolist()}, f, indent=1)
        return torch.tensor(s, dtype=torch.float32, device=self.device)

    def load_gains(self) -> Tensor:
        """Oracle for diagnosis only: s_k = -(realised log2 gain), centred, from <data_dir>/exposure_gains.json."""
        rows = json.loads((Path(self.cfg.data_dir) / "exposure_gains.json").read_text())["views"]
        g = {r["view"]: r["log2_ratio"] for r in rows}
        g = np.array([g[n] for n in self.train_view_names()])
        return torch.tensor(-(g - g.mean()), dtype=torch.float32, device=self.device)

    def dark_scale(self, image_ids: Tensor) -> Optional[Tensor]:
        """Factor 2^{-s_k} that puts view k's dark render at that photo's exposure; None for the official shared
        dark branch. Used by train() and the diagnostics, so the two cannot drift apart. Never carries gradient."""
        src = self.cfg.dark_branch
        if src == "shared":
            return None
        s = self.rung2.s.detach() if src == "model" else (self.s_measured if src == "measured" else self.s_gains)
        return torch.exp(-math.log(2.0) * s[image_ids]).view(-1, 1, 1, 1)  # not torch.exp2: see the note below

# train(), right after colors_low / colors_enh are split from the renders
            scale = self.dark_scale(image_ids)
            if scale is not None:
                colors_low = colors_low * scale  # M2: view k's dark render at its photo's exposure
```

`measure_exposures` does the same float32 mean and float64 log2 as `build_vexp.verify_scene`, so the values should equal `oracle.json["s_init"]` exactly.

**Implementation note (2026-11-11, found by the smoke test).** The first `detachexp` smoke run failed in `torch.exp2` on CUDA: torch 1.13.1 compiles that kernel at run time (NVRTC), and this env lacks `nvrtc-builtins64_117.dll`. The official code never uses such ops.
- `dark_scale` therefore computes exp(−ln2·s), and `s_dark` is computed with numpy on the CPU.
- Checked on CUDA the same day: softplus, expm1, log1p, log2, searchsorted, cumsum, pow and the backward of an indexed gather all work. So does exp2's replacement.
- Probe any other new op on CUDA before relying on it (part B and part C).

#### Step 2. Diagnostics, re-scorer, runner

- `diagnostics._score_views`: after `render_low`, compute `scale = runner.dark_scale(ids)` and `render_low_exp = render_low if scale is None else render_low * scale`. Add two keys at the **end** of `row`:
  - `"s_dark"`: 0.0 for `shared`, else `-log2(scale)`;
  - `"fit_psnr_lowrender_exp_vs_input"`: `psnr(nchw(render_low_exp.clamp(0, 1)), nchw(pixels))`.
- `diagnostics._load_run_state`: next to the data-key assert, check the model flags. `diagnostics.py` must not import the trainer (it parses the CLI at import), so keep a small `MODEL_DEFAULTS` dict there, e.g. `{"dark_branch": "shared"}`, and assert `run_cfg.get(k, MODEL_DEFAULTS[k]) == getattr(cfg, k)`.
- `rescore_runs.py`:

```python
RUN_RE = re.compile(r"^(?P<scene>[a-z]+)_(?P<variant>[a-z0-9]+)_seed(?P<seed>\d+)$")
# flags that change the model or the dark branch: passed through from the run's cfg.json so the re-score rebuilds it
MODEL_FLAGS = ["dark_branch"]  # part C adds curve_model and the rung2_* flags
...
        for flag in MODEL_FLAGS:  # not `k`: main() uses k as its progress counter (the smoke test caught this)
            if flag in cfg:
                cmd += [f"--{flag}", str(cfg[flag])]
```

- `run_rung0.py`:

```python
    # M2: "exp" = exposure-aware dark branch (view k's dark render scaled by 2^{-s_k} before loss_regress_low)
    "detachexp": ["--detach_pseudo_target", "--dark_branch", "measured"],
```

#### Step 3. Checks and smoke test (about 10 minutes)

1. **Default path unchanged.**
   - `python run_rung0.py --dry_run --scenes buu --seeds 0 --variants baseline detach` prints the same commands as before (diff them).
   - `python rescore_runs.py --root ../results/vexp --dry_run` still reports 30 runs already re-scored.
2. **Smoke runs:**

```
python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp/smoke_m2 --scenes buu --variants detach detachexp --seeds 0 --max_steps 300
python rescore_runs.py --root ../results/vexp/smoke_m2 --ref_root ../results/rung0
```

3. **Pass if:**
   - Both runs are `ok`, and `detachexp/cfg.json` has `dark_branch: measured`.
   - `detachexp/exposure_init.json` `s_measured` equals buu's `s_init` in `results/vexp/dataset/oracle.json` within 1e-6, joined by view name.
   - In `rescore/pseudo_targets.csv`:
     - `detach`'s new fit column equals its old one exactly;
     - `detachexp`'s `fit_psnr_lowrender_exp_vs_input` is higher than `detach`'s `fit_psnr_lowrender_vs_input` on most views, most clearly at large |u_k|. At 300 steps the dark fit is immature, so this checks only the direction. Report the numbers.
   - `mean_pseudo` per view agrees between the two runs within about 1e-3 (section 2.1, built-in check). A larger gap means something in the curve path sees the dark branch: find it before the batch.
   - The step-0 row of `diagnostics/loss_log.csv` for `detach` equals M1's smoke run (`results/vexp/smoke/buu_detach_seed0`, git-ignored; use it if it is still on disk) to every printed digit.
     - If that folder is gone, export the committed `examples/` with `git archive` into a scratch folder next to `examples/` (this changes no git state), run the same 300 steps from there, and compare.
     - Later rows only need to agree within GPU noise.
4. Optional: one 300-step run with `--dark_branch gains` (direct trainer call). Check that `s_gains` equals −(log2_ratio − mean) from `exposure_gains.json`.
5. Log the code change and the smoke results with `/log-experiment`.

#### Step 4. ASK Bart, then run `detachexp` on the varying set (about 2 h)

Use `/run-batch` (Bart starts it) with:

```
python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants detachexp --seeds 0 1 2
```

- 15 runs × 8.1 min (`detach` on this set in M1; the scaling costs nothing measurable, *inferred*) ≈ 2 h. Remind Bart about sleep.
- Then re-score the 15 new runs (a few seconds each): `python rescore_runs.py --root ../results/vexp --ref_root ../results/rung0`.
- Quick look, not the full analysis: `summarize_runs.py` for `detach` and `detachexp` on `results/vexp`, plus the mean trained dark fit per scene from `rescore/pseudo_targets.csv`. Log it.
- This is the first column effect, for `detach`. It is worth sending to Wei whatever Rung 2 does.

### Part B: Rung 2 as new files (while the step 4 batch runs)

#### Step 5. `examples/rung2.py` and `examples/test_rung2.py`

`rung2.py` (no import of the trainer):

```python
def make_knots(K, spacing, y_max=None) -> Tensor          # [K+1], y_0 = 0, y_K = 1, strictly increasing
def init_heights(mode, knots, cdfs, s_init) -> Tensor     # [K+1]; "identity" -> knots, "cdf" -> section 2.2
def interp(yq, knots, h) -> Tensor                        # piecewise-linear F, differentiable in yq and h

class ExposureCurve(torch.nn.Module):
    """Rung 2: L_k(x) = F(min(1, 2^{s_k} x)); one shared monotone F, one exposure s_k per training view."""
    def __init__(self, knots, h_init, s_init): ...        # buffer knots; Parameter theta [K] from h_init; Parameter s [N]
    def heights(self) -> Tensor                           # [K+1]: [0, c / c[-1]], c = cumsum(softplus(theta))
    def composed_curve(self, k) -> Tensor                 # [1, 255]: F(min(1, 2^{s_k} x_b)), x_b = b/254; grad to theta and s
    def map_pixels(self, pixels, k, attach) -> Tensor     # [1, H, W, 3]; attach "none" | "s" | "all"
    @torch.no_grad()
    def project_gauge(self): self.s -= self.s.mean()
    def n_params(self) -> int                             # K + N

def curve_loss(curve, cdf, step, exp_name, smooth_lambda) -> (total, parts)
    # w(t) * mean((curve - cdf)^2) + smooth_lambda * mean(diff(curve)^2); parts as HistogramPriorLoss returns them,
    # with "pseudo" = NaN (no shape prior), so Diagnostics.log_losses keeps working
```

Interpolation, with knots `y` of length K+1 and heights `h`:

```python
def interp(yq, y, h):
    j = torch.searchsorted(y[1:-1].contiguous(), yq.contiguous(), right=True)  # segment index 0..K-1
    return h[j] + (h[j + 1] - h[j]) * (yq - y[j]) / (y[j + 1] - y[j])
```

- `map_pixels` uses `h.detach()` for `attach` = `none` or `s`, and `s.detach()` for `none`.
- The `cdf` init: `softplus(θ_j) ∝ max(h_j − h_{j−1}, 1e-4)`, i.e. θ_j = log(expm1(d_j)) with d_j the increment times K, so θ is of order 1.

`test_rung2.py` (CPU, a few seconds). Each test is an assert with a message:
1. `make_knots`: y_0 = 0, y_K = 1, strictly increasing for every spacing. Quadratic puts at least K/2 knots in [0, 0.25]; `usedrange` puts K knots at or below y_max.
2. For random θ (several seeds): heights strictly increasing, h_0 = 0, h_K = 1 (atol 1e-6).
3. `interp` equals `np.interp` on 10,000 random points (atol 1e-6), and is non-decreasing on a fine grid.
4. Uniform knots, identity init and s = 0: `map_pixels` returns the input (atol 1e-6).
5. Gradients:
   - `none`: the output has no grad;
   - `s`: d/ds is non-zero and θ gets none;
   - `all`: both get gradient.
   - A finite-difference check of ∂F(2^s x)/∂s = F′(y)·y·ln 2 (rtol 1e-3). Pixels with 2^s x ≥ 1 get zero gradient.
6. `composed_curve(k)[b]` equals `map_pixels` on an image whose pixels are x_b, so the loss and the target use the same function.
7. `project_gauge` gives mean 0 and leaves differences between views unchanged.
8. `n_params() == K + N`.
9. Synthetic recovery: CDFs built as G(min(1, 2^{s_k} x)), the model's own form, from a known monotone G and known mean-zero s. (Review: this first said G(2^{−s_k} x), with which the init gives G(2^{−2s_k} y) and Adam recovers −s.) Choose a G that reaches 1 below 2^{min_k s_k}, as real HE-CDFs of dark photos do. Otherwise the synthetic CDFs are not 1 at x = 1, and the init cannot reproduce G.
   - `init_heights("cdf", …)` with the true s recovers G at the knots within 0.02.
   - 3,000 steps of Adam on `curve_loss` from s = 0 recover s within 0.03 stops.
10. `curve_loss` matches a hand computation on a toy input, and w switches at step 3000.

#### Step 6. `examples/rung2_preview.py`: what Rung 2's curve will do, before any 3DGS run

**Why it works.** In the default (detached) mode, F and s_k learn only from `L_curve`. `L_curve` depends only on the photos' HE-CDFs, never on the Gaussians. So the whole F and s_k trajectory can be replayed in seconds without rendering. The replay differs from a real run only in the random view order.

**What it does.** Per scene, on LOM and on the varying set:
1. Load the training photos with the trainer's own `Parser`/`Dataset`, as `vexp_analyze.histogram_oracle` does.
2. Compute the CDFs, m_k and the ¼-size grey histograms.
3. Replay 10k steps: per epoch a seeded random permutation of views. At each step, `10·curve_loss(composed_curve(k), cdf_k, step)`, Adam with the trainer's learning rates and schedule, then `project_gauge`.
4. Also run a full-batch fit (all views per step, 3,000 Adam steps) as the optimum.
5. Configurations: knot spacing {quadratic, usedrange} × F init {cdf, identity}.

Default `--device cpu` with `torch.set_num_threads(2)`, so it can run during a batch. About 15 minutes (*inferred*). It writes only to `results/rung2/preview/` (`tables.md`, CSVs, `figures/`), guarded.

| Table | Content |
|---|---|
| P1 fit | Weighted residual ×255 of the final composed curves against the CDFs, weighted by the grey histogram as in `fit_exposure_form`, next to the histogram oracle's residual on the same CDFs. Also the final loss against the full-batch optimum, and the step at which the loss is within 5% of its final value. |
| P2 exposure (varying set) | s_k against u_k: slope, r, unpaired RMSE of s + u, paired RMSE (minus the same view's s on LOM, same configuration), and movement from the init, rmse_c(s − s_init). Next to the init and the oracle; reuse `vexp_analyze.recovery_stats`. |
| P3 leak preview | Pseudo-targets F(min(1, 2^{s_k} x)) of every training view on both sets: paired leak (slope of log2(mean, varying / LOM) against u_k). Next to M1's `detach` (0.36 to 0.65) and official (0.27 to 0.56) from `results/vexp/summary/T4_exposure.csv`. |
| P4 dark end | F(ℓ/255) for ℓ ∈ {1, 2, 4, 8, 16} at s = 0, next to `detach`'s mean curve at the same input levels through the official LUT rule (level ℓ reads `curve[max(ℓ−1, 0)]`), from the LOM `detach` seed-0 `curves_step10000.npz` (local). Also the fraction of training pixels at levels 0 to 2 per scene, and (review) the dark-end slope F′(y)·2^{s} at levels 1 to 8, i.e. how much the curve amplifies each photo's noise, next to `detach`'s. |
| P5 pseudo-target quality (diagnostic only) | Brightness-matched PSNR of the preview pseudo-targets against the training views' normal-light images, next to `detach`'s final `pseudo_psnr_gtmean` from `results/rung0/<scene>_detach_seed*/rescore/pseudo_targets.csv` (LOM) and `results/vexp/` (varying). **Never used to choose a setting**: that would tune an unsupervised method on normal-light images. |

Figures: F and each view's composed curve against the views' CDFs, per scene (default configuration); s_k against u_k (varying set).

**Decision rules (training photos only; Bart confirms before part C's smoke tests):**
- **Knot spacing:** keep `quadratic` unless `usedrange` lowers P1's weighted residual, pooled over scenes and both sets, by at least 30%. The proposal flagged this risk: 1.83/255 against 0.70/255 on sofa's official mean curve.
- **Learning rates:** keep 1e-3 unless P1's final loss is more than 5% above the full-batch optimum. Then try 3e-3 and recheck.
- **F init:** keep `cdf`. Report how many steps `identity` needs to come within 0.01 of the same F.
- **Stop and ask Bart** if P4 or P5 show Rung 2's pseudo-targets clearly worse than `detach`'s on LOM, for example a P5 gap larger than LOM's brightness-matched seed noise in two or more scenes. That puts H1 at risk; see risk 1 in section 7.
- **Pre-declared remedy (review).** If P5, which uses the normal-light images, triggered a change of default, the default would be informed by the ground truth. So the remedy is fixed now and judged on training photos only (P1, P4): first the target alignment of 2.2, if not already adopted; then the smoothness ablation in M5. λ stays at 0.1 by default.
- If Bart has not decided the alignment of 2.2 by then, report P2, P4 and P5 for both alignments.

Write a log entry with the P1 to P5 headline numbers. P3 is the first read on H2's mechanism.

### Part C: wire Rung 2 in (after the step 4 batch has finished)

#### Step 7. Trainer and diagnostics

1. **`Config`:**
   - `curve_model: Literal["official", "rung2"] = "official"`;
   - `rung2_knots: int = 32`;
   - `rung2_knot_spacing: Literal["quadratic", "uniform", "usedrange"] = "quadratic"` (or the step 6 choice);
   - `rung2_F_init: Literal["cdf", "identity"] = "cdf"`;
   - `rung2_s_init: Literal["measured", "zero"] = "measured"`;
   - `rung2_attach: Literal["none", "s", "all"] = "none"`;
   - `rung2_smooth_lambda: float = 0.1`, `rung2_lr_F: float = 1e-3`, `rung2_lr_s: float = 1e-3`.
2. **`Runner.__init__`, Rung 2 mode:**
   - Assert the flag guard (section 2.2).
   - Compute `s_measured` (also when `--dark_branch shared`), the per-view CDFs and, for `usedrange`, y_max.
   - Build `self.rung2 = ExposureCurve(...)` with its two Adam optimisers.
   - Do not build `curve`, `curve_adjust`, `curve_adjust_gamma`, `pesdo_curve` or `axis*_para`. Set the official optimiser lists to `[]`.
   - Print `[rung2] curve parameters: K + N = …`.
   - Relax the `--dark_branch model` guard from step 1.
3. **`curve_forward`, Rung 2 branch:** return
   - `curve_adj` = `composed_curve(k)` (attached; used by the curve loss);
   - `pixels_enh` = `map_pixels(pixels, k, cfg.rung2_attach)`;
   - `projected` = (R, G, B);
   - `s_k`;
   - placeholders: `curve_bias` zeros, `curve_preclamp` = `curve_adj`, `pesdo_curve` NaN, `gamma`/`alpha`/`beta` NaN.
   - `zero_basis` is a no-op in this branch.
4. **`train()`:**
   - `hist_loss, hist_parts = rung2.curve_loss(...)` in place of `loss_histo`.
   - Schedulers from one list of curve-side optimisers.
   - `self.rung2.project_gauge()` after the optimiser steps.
   - At the end, assert |mean s| < 1e-5 and that F is strictly increasing.
5. **`grad_check`:** in Rung 2 mode, report the gradient norm on θ and on s from `loss_regress_enh` **and** from `loss_regress_low`. (Review: `train()` passes only `loss_regress_enh` today, so pass `loss_regress_low` as well.)
6. **Diagnostics:**
   - `_curve_modules` returns `(runner.rung2,)`.
   - `snapshot_curves` fills the official-only keys with zeros or NaN, adds `s`, `F_knots` and `F_heights`, and sets `curve_global` = F on the 255-bin grid. It appends `step, view, s` rows to `diagnostics/exposures_by_step.csv`.
   - `_save_curve_state` and `_load_run_state` handle a Rung 2 state: `curve_model`, knots, θ, s, `s_init`. Load the knots from the state; for `usedrange`, assert they match the recomputed ones within 1e-6.
   - `final_snapshot` writes `diagnostics/rung2_state.json` with: view names, `s_init`, `s_final`, knots, heights, K, spacing, `n_params` and the final mean of s.
7. **`MODEL_FLAGS` and `MODEL_DEFAULTS`:** add `curve_model` and every `rung2_*` flag. **`run_rung0.py`:** add `rung2`, `rung2exp` and `rung2att` (section 2.3).

#### Step 8. Regression checks and smoke tests

1. `python test_rung2.py` passes.
2. **Default path unchanged:**
   - Repeat step 3.1 (dry-run diff).
   - Re-score `results/vexp/buu_detach_seed0` and `results/rung0/buu_baseline_seed0` with the new code into `results/rung2/smoke_regress/<run>/`. Call the trainer directly with `--rescore_run` and `--ref_depth_run`, and never write into the runs' own `rescore/`.
   - Every shared column of `pseudo_targets.csv`, `consistency.csv` and `val_metrics.csv`, and every key of `val_summary.json` except paths and `seconds`, must equal the existing `rescore/` outputs within 1e-6. Forward rendering is deterministic; M1 measured re-score = training eval within 2.5e-6 dB. (Review: if an LPIPS or SSIM column misses 1e-6 but stays below 1e-4, re-score the same run twice with the new code. A gap no larger than between those two repeats is GPU float noise, not a regression.)
   - Repeat the step 3.3 loss-log comparison for a 300-step `detach` run.
3. **Rung 2 smoke runs** (300 steps), re-scored with `--ref_root ../results/rung0`:

```
python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp/smoke_m2 --scenes buu --variants rung2 rung2exp rung2att --seeds 0 --max_steps 300
python run_rung0.py --root ../results/rung2/smoke_m2 --scenes buu bike --variants rung2 --seeds 0 --max_steps 300
```

   For the gradient check, run each of `rung2`, `rung2exp` and `rung2att` once directly with `--grad_check` and the protocol's smoke settings.
4. **Pass if:**
   - Start-up prints 54 curve parameters for buu (32 + 22).
   - `grad_check` output:
     - `rung2`: `loss_regress_enh` gives no gradient to θ or s.
     - `rung2att`: s gets a non-zero gradient from `loss_regress_enh`; θ gets none.
     - `rung2exp`: s gets no gradient from `loss_regress_low`.
   - All runs `ok`. At the end |mean s| < 1e-5 and F is strictly increasing. `rung2_state.json` exists and its `s_init` equals `oracle.json["s_init"]` (varying set).
   - `rung2` and `rung2exp` (same seed): s and F agree within 1e-6, and `mean_pseudo` per view within 1e-5. (Review: the backward of the knot gather `h[j]` uses CUDA atomics, so float noise alone may exceed 1e-6. If it does, run `rung2` twice with the same seed; only a gap larger than that repeat's is a leak.)
   - `build_vexp.fit_exposure_form` on the snapshot's `curve_adj` and `bin_hist` returns the run's own s_k within about 0.01 stops. This checks that the M1 analysis code reads Rung 2 correctly.
   - Re-scoring works, and the re-scored `mean_pseudo` equals the run's own `diagnostics/pseudo_targets.csv`.
   - Speed: the ratio of minutes per 300-step smoke to `detach`'s smoke, times `detach`'s 8.1 min per 10k-step run. This sets M3's estimate; replace it with the first full run's time. (Review: do not scale a 300-step run by 10000/300. M1's `detach` smoke took 0.63 min, which scales to 21 min against the real 8.1, because start-up, eval and the trajectory video dominate a short run.)
   - Look at the bike example PNG (`diagnostics/example_*_in_pseudo_render_gt.png`) next to `detach`'s from `results/rung0/bike_detach_seed0`. Check the dark end and the noise.
5. Write a log entry with `/log-experiment`.

#### Step 9. Docs, log, commit list

1. **`CLAUDE.md`:**
   - Rung 2 row: M2 status.
   - Repo map: `rung2.py`, `test_rung2.py`, `rung2_preview.py`, `results/rung2/`.
   - "Our flags": `--dark_branch`, `--curve_model`, the `--rung2_*` flags.
   - Rule 6: add `results/vexp/summary/` and `M1_vexp_results.md`.
2. **`docs/experiment_protocol.md`** (add notes; do not rewrite old text):
   - the `results/rung2` root and the new variants;
   - the columns `s_dark` and `fit_psnr_lowrender_exp_vs_input`, and the files `exposure_init.json`, `rung2_state.json` and `exposures_by_step.csv`;
   - the 2×2 and the decision rules of section 6.5;
   - Rung 2's minutes per run.
3. **`docs/curve_estimation.md`:** it says "Flag names are decided in the Rung 2 plan". Add them, and a paragraph on the dark branch.
4. **Explainer 8.6:** a dated note on the dark-branch extension.
5. **`.claude/settings.json`:** the ask-list entries from section 3.2, item 9.
6. **`.claude/skills/run-batch/SKILL.md`** (review): re-score every root other than `results/rung0` with `--ref_root ../results/rung0`. `results/rung2/` has no `<scene>_baseline_seed0`, so a plain re-score there skips every run.
   - Done on 2026-11-11, before part A: the parts of items 1 to 6 that do not depend on new code (the decisions, the dark-branch description, stale status, settings and skill). Left for this step: flag lists, repo-map entries and numbers that need the code or the runs (for example Rung 2's minutes per run).
7. **Do not commit.** Propose two commits:
   - **Part A:** `simple_trainer_ours.py`, `diagnostics.py`, `rescore_runs.py`, `run_rung0.py`, `Implementation_guides/M2_rung2_plan.md`, `docs/RESEARCH_LOG.md`, and the docs updated before part A (`CLAUDE.md`, `docs/Luminance_GS_Research_Project_Explained.md`, `docs/experiment_protocol.md`, `docs/curve_estimation.md`, `docs/repo_vs_paper.md`, `.claude/settings.json`, `.claude/skills/run-batch/SKILL.md`). After the batch, add the 15 `detachexp` run folders (`cfg.json`, `exposure_init.json`, `stats/`, `diagnostics/*.csv`, `diagnostics/timing.json`, `rescore/*.csv`, `rescore/*.json`; the re-score writes its own `rescore/exposure_init.json` too) and `results/vexp/runs.csv`.
     Message: `M2 part A: exposure-aware dark branch (--dark_branch) and detachexp on the varying-exposure set`.
   - **Parts B and C:** `rung2.py`, `test_rung2.py`, `rung2_preview.py`, the trainer, `diagnostics.py`, `run_rung0.py`, `rescore_runs.py`, `results/rung2/preview/`, the docs, `.claude/settings.json`, `docs/RESEARCH_LOG.md`.
     Message: `M2: Rung 2 behind --curve_model rung2 (shared monotone F + per-view exposure), preview and tests`.

---

## 5. M3: the runs

### 5.1 Main batch: ASK Bart (about 6 h; seed 0 of everything after about 2 h)

Seed-major across both roots: seed 0 of all three new configurations first, then seeds 1 and 2. Chain the commands with `;` in PowerShell (the Anaconda PowerShell Prompt; in the plain Anaconda Prompt, which is cmd, use `&` instead: review), each with its output redirected to its root's `batch_console_m3*.log`:

```
python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants rung2 rung2exp --seeds 0
python run_rung0.py --root ../results/rung2 --variants rung2 --seeds 0
python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants rung2 rung2exp --seeds 1 2
python run_rung0.py --root ../results/rung2 --variants rung2 --seeds 1 2
```

- 45 runs × about 8 min ≈ 6 h (*inferred*: `detach` takes 8.1 min on this set, and Rung 2 drops two networks). Replace the estimate with the step 8 speed estimate (the ratio to `detach`'s smoke, not 300 steps scaled up).
- `--root ../results/rung2` with the default LOM data root is allowed by the runner's guard.

### 5.2 Re-score

```
python rescore_runs.py --root ../results/vexp --ref_root ../results/rung0
python rescore_runs.py --root ../results/rung2 --ref_root ../results/rung0
```

### 5.3 Conditional runs (pre-declared; each needs Bart's OK)

| Condition | Runs | Cost |
|---|---|---|
| The pooled H2 curve-model effect lies between 1× and 2× noise after 3 seeds | Seeds 3 and 4 for `detach`, `detachexp`, `rung2` and `rung2exp` on the varying set | 40 runs, about 5.5 h |
| The dark-branch effect differs between rows (real interaction) | `rung2` with `--dark_branch measured` (variant `rung2meas`), varying set, 3 seeds: both rows then use the same exposures | 15 runs, about 2 h |
| `detachexp`'s dark fit stays more than 2 dB below LOM's | `detach` with `--dark_branch gains` (variant `detachgains`), varying set, 3 seeds: how much is exposure-estimation error | 15 runs, about 2 h |
| Bart wants the switch in the final method | `detachexp` and `rung2exp` on LOM: the switch must not hurt evenly exposed scenes | 30 runs, about 4 h |
| Rung 2's LOM result is borderline for H1 | `rung2` with `--rung2_F_init identity`, seed 0, LOM: head-start check (risk 5) | 5 runs, about 40 min |

H3 (`rung2att` on LOM and the varying set) is M5.

---

## 6. Reporting: `rung2_analyze.py` and `Rung2_results.md`

Same style as `vexp_analyze.py`: `pathlib`, `__main__` guard, helpers imported rather than copied. Use `vexp_analyze`'s `paired`, `rmse_c`, `recovery_stats`, `exposure_run`, `histogram_oracle`, `seeds_of` and `group_scenes`, and the factorial pattern of `rung05_analyze.effects`. (Review: that function is hard-coded to the Rung 0.5 cell names, so write an `effects(vals, cells)` in `rung2_analyze.py` that copies its pattern. Do not edit `rung05_analyze.py`: its outputs are finished.) Write only to `results/rung2/summary/` (guarded). Read every other folder without writing to it.

### 6.1 Inputs and checks

- **Inputs:**
  - LOM: `results/rung0/<scene>_{baseline,detach}_seed*`, `results/rung2/<scene>_rung2_seed*`.
  - Varying set: `results/vexp/<scene>_{baseline,detach,detachexp,rung2,rung2exp}_seed*`.
  - Dataset: `results/vexp/dataset/{gains,oracle}.json`.
  - M1's exposure numbers for official and `detach`: `results/vexp/summary/T4_exposure.csv` and `numbers.json`.
  - `results/rung2/preview/`.
- **Stop before any table if:**
  - a run's `data_dir` does not match its root;
  - its `cfg.json` flags do not match its variant (section 2.3);
  - `max_steps` is not 10k;
  - its re-score did not use `results/rung0/<scene>_baseline_seed0` as reference depth;
  - it has no `ok` row in its root's `runs.csv`.
- **Built-in checks** (section 2.1):
  - `rung2` against `rung2exp`, same seed: s, F and `mean_pseudo` identical within 1e-5. Stop and report if not, after comparing with a same-seed repeat as in step 8.4 (review).
  - `detach` against `detachexp`, same seed: report the largest |Δ log2 `mean_pseudo`|; expected at the float-noise level.
  - The trainer's final s_k against the preview's: report only, expected within a few hundredths of a stop because the view order differs.

### 6.2 Tables

| Table | Content |
|---|---|
| R1 H1 (LOM) | `rung2`, `detach` and official per scene, pooled and pooled without shrub: raw and brightness-matched PSNR, SSIM, LPIPS (mean ± sd). `rung2` − `detach` paired by seed, with ×noise, and a verdict per scene (section 6.5). |
| R2 H1 extras (LOM) | Pseudo-target brightness-matched PSNR, pseudo-target and render consistency (× GT), `ang_err_deg`, `sat_ratio`, bike `hue_err_sat_deg`, dark fit, minutes per run, curve parameters. |
| R3 2×2 cells (varying set) | The four cells plus official, per scene and pooled: raw and brightness-matched PSNR, SSIM, LPIPS. |
| R4 factorial (varying set) | Curve-model main effect, dark-branch main effect, interaction, and the four simple effects, for brightness-matched PSNR, raw PSNR, LPIPS and SSIM, with ×noise and "real". |
| R5 mechanism | Leak, dark fit and exposure (section 6.3). |
| R6 side effects (varying set) | Colour, consistency, wall time and final `num_GS` per cell (review: the dark branch re-weights gradients, section 2.1). |

### 6.3 How the three mechanism diagnostics are reported

**(a) Pseudo-target leak** (does the curve side remove the exposure differences?)
- **Per curve row:** official, `detach`, `rung2`. The exp cells' pseudo-targets equal their row's (check in 6.1).
- **Paired leak:** slope of log2(mean pseudo-target, varying / LOM) against u_k, same view, seed and row. The LOM partner is `results/rung0` for official and `detach`, and `results/rung2` for `rung2`.
- **Also report:** the unpaired slope; mean ± sd over seeds; and, as in M1 T4b, rows for the dark input (check ≈ 1) and the enhanced render at the training view. The render leak is reported per cell, because the Gaussians can differ between columns.
- **Reading:** 0 means fully compensated and 1 means passes straight through. M1 found `detach` 0.36 to 0.65 (pooled 0.51) and official 0.27 to 0.56. Expected for Rung 2 (*inferred*): near 0, slightly negative, because its s_k should end near the histogram oracle, which overshoots the slope by 4 to 11%.

**(b) Dark-branch fit** (does the exposure-aware branch remove the misfit?)
- **Per cell, the trained dark fit:** PSNR of the dark render as its loss sees it. That is `fit_psnr_lowrender_vs_input` for shared cells and `fit_psnr_lowrender_exp_vs_input` for exp cells. Mean over training views, then mean ± sd over seeds.
- **Next to:** the same row's LOM level (`detach` 46.2 dB pooled from `results/rung0`; `rung2` from `results/rung2`) and the column effect with ×noise.
- **Per view:** the paired change (varying − LOM, same view, seed and row) against signed u_k, with slopes for u < 0 and u > 0 separately, as in M1 T5b. For the `detach` cell this reproduces M1 (+5.2 and −12.9 dB per stop).
- **Reading:** the switch works if the per-view slopes flatten and most of `detach`'s 7.5 dB drop comes back. What remains against LOM is exposure-estimation error (the `gains` cell, if run) plus quantisation.

**(c) Exposure recovery** (Rung 2 only; `detach` has no s_k, so its post-hoc fit from M1 is shown for comparison)
- **Values:** the final s_k from `rung2_state.json`.
- **Statistics:**
  - slope and r against u_k (a sanity check: they saturate on this set);
  - unpaired RMSE of s + u (floor near the natural spread, 0.08 to 0.20 stops);
  - **paired RMSE** of (s_varying − s_LOM) + u, same view and seed, LOM partner from `results/rung2`;
  - movement from the init, rmse_c(s − s_init).
- **Reference rows:** init (paired 0.004 to 0.019), histogram oracle (paired 0.038 to 0.082), and the official and `detach` post-hoc s_fit (0.61 to 0.84, M1 T4a).
- **Plus:** paired RMSE against training step from `exposures_by_step.csv`. Check that `fit_exposure_form` on Rung 2's composed curves returns its own s_k within 0.01 stops.
- **Reading:** this is not a "beat the initialisation" test, because the init already sits at the quantisation floor. It shows whether training keeps s_k on the compensating side (paired RMSE around the oracle's or below, against 0.6 to 0.8 for the official model), and how far the histogram fit pulls s_k away from the measured init.

### 6.4 Figures

Keep the colours: official `SERIES[0]` and `detach` `SERIES[1]` as in M1, `rung2` `SERIES[4]`. Exp cells use their row's colour with hollow markers and dashed lines. Init and oracle as in M1. One idea per figure, axes with units.
1. H1: brightness-matched PSNR and LPIPS per scene on LOM for official, `detach` and `rung2`, one dot per seed.
2. 2×2 interaction per scene (brightness-matched PSNR): dark branch on the x axis, one line per curve row, dots per seed.
3. Leak: paired pseudo-target brightness change against u_k (seed 0) for official, `detach` and `rung2`, with the 0 and 1 lines.
4. Dark fit: paired change (varying − LOM) against signed u_k, all four cells.
5. Exposure: Rung 2's s_k against u_k per scene (seed 0), with the init, the oracle and the ideal slope −1. Second panel: paired RMSE against training step.
6. Curves, one scene (chair or sofa), seed 0: F, the per-view composed curves F(2^{s_k} x) over the used range, the views' HE-CDFs, and `detach`'s per-view curves.

### 6.5 Decision rules (proposed and pre-registered; Bart confirms, then mentions them to Wei on 13 Nov)

- **Noise and "real":** as in M1. Pair by seed; noise = sqrt(mean of the arms' seed variances); real above 2×. For the factorial, noise over the four cells, as in `rung05_analyze.effects`. Shrub is reported but not used to decide (protocol rule 7). Check pooled verdicts against pooled-without-shrub and say so if they differ.
- **H1 (LOM):** supported if, in each of buu, chair, sofa and bike, `rung2` − `detach` in brightness-matched PSNR is not below −2× noise, and SSIM and LPIPS are not worse by more than 2× noise (explainer 8.11).
  - **Open, for Bart before M3 (review).** This rule differs from the explainer's 8.11, which uses every scene and "within the detectable effect" (about 1.6× the per-run std). Pick one before it is pre-registered with Wei. The recommendation is this one, which follows protocol rules 5 and 7; the explainer is then updated to match.
- **H2 (varying set), read from the 2×2:**
  - **Supported:** the pooled curve-model main effect on brightness-matched PSNR is real and positive, LPIPS is not worse beyond noise, and the effect is positive in most informative scenes.
  - **Supported only with the dark-branch fix:** only the simple effect in the exposure-aware column (`rung2exp` − `detachexp`) is real.
  - **Not supported:** neither. The leak and dark-fit diagnostics then say which side failed.
  - **Between 1× and 2× noise pooled:** run seeds 3 and 4 (5.3) before concluding. Do not reinterpret.
  - **Power** (*inferred*): if Rung 2's seed spread is like `detach`'s on this set (pooled sd 0.72 dB; per scene 0.03 to 1.51), a pooled effect needs about 1.4 to 1.5 dB to count. Per-scene tests will rarely pass on buu and sofa.
- **Mechanism** (supports H2, does not replace it): Rung 2's pooled paired leak is near 0 and below `detach`'s by more than 2× noise.
- **Exposure recovery:** a sanity check only (6.3c).

### 6.6 Seed-0 mode (for the 13 Nov check-in)

`python rung2_analyze.py --seeds 0` writes the same tables to `results/rung2/summary_seed0/`:
- n = 1, so no sd, no ×noise and no "real" (review: `vexp_analyze.paired` returns NaN noise for n = 1, with numpy warnings, so do not call it here);
- each difference is shown next to M1's seed sd for that scene and metric (`results/vexp/summary/T1_val_varying.csv`; `results/rung0/summary_rung05/` for LOM);
- every table is headed "preliminary, one seed".

The full run later writes `results/rung2/summary/` and leaves the seed-0 snapshot as it was.

### 6.7 `Implementation_guides/Rung2_results.md` (M3, by hand from the summary)

Same rules as M1's report: direct answers first, every claim *measured* or *inferred*, raw and brightness-matched PSNR side by side, real only above 2× noise, shrub reported but not used to decide.
- **Q1 (H1):** does Rung 2 match `detach` on LOM, with how many parameters and how much time?
- **Q2 (dark branch):** how much of the varying-exposure cost is the dark branch, for each curve model?
- **Q3 (H2):** does Rung 2 beat `detach` on the varying set, with and without the dark-branch fix?
- **Q4 (mechanism):** is Rung 2's pseudo-target leak near 0, and do its s_k stay on the compensating side?
- **Q5 (side effects):** colour, consistency, and dark-end behaviour on bike and shrub.
- **Closing:** what this means for M5 (H3, colour-preserving variant, knots).

---

## 7. Risks to watch

1. **The dark end.** F(0) = 0 with interpolation maps level 0 to black and level 1 to F(2^{s_k}/255). The official LUT maps levels 0 and 1 to the same lifted `curve[0]` (section 2.2).
   - On the darkest scenes (bike mean 7 to 8 levels, shrub 7 to 10, *measured* in M1), Rung 2's pseudo-targets could be noisier or darker in the shadows.
   - Preview tables P4 and P5 show this before any GPU time is spent.
   - If it is clearly worse, bring it to Bart. Do not quietly tune λ on normal-light images.
2. **Knot spacing.** Quadratic knots may underfit where the pixels are (the proposal's sofa check). The preview rule in step 6 decides from training photos only.
3. **Exposure-estimation error** (section 2.1). Neither `measured` nor Rung 2's s_k is exact, so no exp cell returns fully to LOM's dark fit. `gains` separates estimation error from misfit if needed.
4. **Power.** Seed noise on the varying set is large (6.5). Hence the pre-declared extra seeds.
5. **Head start.** `cdf` init gives bright targets from step 0, while `detach`'s curve starts at the identity. The preview shows how fast `identity` converges, and 5.3 has a check if H1 is borderline.
6. **Quantisation** (M1). Darkened views lost tonal levels, so no cell returns fully to its LOM level.
7. **GPU nondeterminism.** Runs with an identical seed differed by up to about 1 dB in the 2026-10-17 reproduction. This is inside the seed noise; never compare single runs.
8. **Editing code during a batch.** See the rules at the top.

---

## 8. Done when

**M2**
- [x] Step 0: decision logged, branch question asked (2026-11-11; branch: `main`).
- [ ] Part A: switch built; step 3 checks passed; `detachexp` batch `ok` (15 runs), re-scored, quick look logged.
- [ ] Part B: `rung2.py` and `test_rung2.py` pass; preview tables P1 to P5 written and logged; knot-spacing and learning-rate choices confirmed by Bart.
- [ ] Part C: Rung 2 wired in; step 8 regression and smoke checks passed; docs updated; commit lists and messages given to Bart.

**M3**
- [ ] 45 runs `ok`, all re-scored with `--ref_root ../results/rung0`.
- [ ] `rung2_analyze.py` written; `summary_seed0/` for the check-in, `summary/` once all seeds are done.
- [ ] `Rung2_results.md` answers Q1 to Q5. Research log, `CLAUDE.md` rung table and protocol updated.

---

## 9. For the 13 Nov check-in with Wei

**Show:**
- M1 in three lines: the H2 bar, the wrong-sign finding, the dark branch.
- `detachexp`'s column effect.
- The preview's leak and exposure results (P2, P3).
- R1 and R3/R4 at seed 0 or better, with figures 2, 3 and 5.

**Ask:**
1. Is Wei happy with the 2×2 and the dark-branch switch? They go slightly beyond the approved plan.
2. Is the factorial reading of H2 in section 6.5 acceptable?
3. Rung 3: the recommendation is to drop it and spend the time on H3 and the colour-preserving variant.
4. Is LOM overexposure in scope for H1? It needs official and `detach` baselines on `over_exp` (30 runs, about 5 h) plus Rung 2 (15 runs).
