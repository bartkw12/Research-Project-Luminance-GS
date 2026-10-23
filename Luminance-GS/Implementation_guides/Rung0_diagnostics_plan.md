# Rung 0: Diagnostics Plan for Luminance-GS (prompt for Claude Code)

> **How to use this file:** open Claude Code in the repository root (the folder that contains this file and the `Luminance-GS/` code folder) and say:
> *"Read `Rung0_diagnostics_plan.md` and carry out the plan. Start with Phase 0 and stop at each checkpoint marked STOP AND REPORT."*

---

## 1. Who you are working for and why

You are helping **Bart**, a Master of Engineering student in Computer Engineering at McMaster University, with his ECE 701 independent research project. His supervisor is Prof. Chen and his day-to-day PhD mentor is Wei. Experiments must be finished by **early December 2026**, then he writes the report.

The project extends **Luminance-GS** (Cui, Chu, Harada, CVPR 2025, arXiv 2504.01503). Luminance-GS reconstructs a normally lit 3D Gaussian Splatting scene from low-light photos without any normal-light supervision. This repository is the authors' official code (built on gsplat), copied unmodified. Bart has already reproduced it on three LOM scenes (see `average_results.md`).

### The research direction we chose

Wei asked for "a new method to transform the dark rendering into a normal-light one, with different curve estimation or another technique". The chosen research question is:

> Does constraining Luminance-GS's view-adaptive tone curve to a **monotonic, physically grounded, low-dimensional** model make its pseudo-enhanced training targets more consistent across views and more stable across runs? Can a **spatially-varying monotonic** version recover local detail that one global curve cannot?

The work is a ladder; each rung only replaces the curve module:

| Rung | What | Status |
| --- | --- | --- |
| **0** | **Diagnostics of the existing curve. No new method.** | **This file** |
| 1 | Compact monotonic curve: K positive increments (softplus, cumsum, normalise), linear interpolation instead of 255-bin lookup | Later |
| 2 | Exposure model: one shared monotonic curve F plus one exposure value per view, `L_k(x) = F(min(1, 2^{s_k} x))` | Later |
| 3 | Local monotonic curves: M shared monotonic basis curves blended per pixel by a low-res softmax weight map (stretch goal) | Later |

### Why Rung 0 comes first

The case for Rungs 1 to 3 rests on claims about the current curve that nobody has measured. Rung 0 measures them, so the proposal to Wei and Prof. Chen is backed by data, and so we do not build a method that fixes a problem that does not exist. Specifically:

- **Is the learned curve actually non-monotonic?** It is pulled toward a histogram-equalisation CDF (always rising) and smoothed by a TV term, so it may already rise everywhere. If so, "enforce monotonicity" alone will change little and we lean on Rungs 2 and 3.
- **How many degrees of freedom does it really use?** If the 255-value per-view curves are well described by ~16 numbers, Rung 1's compact form is justified.
- **Does the "view-adaptive shape prior" adapt at all?** Its parameters are scaled by 0.1 and 0.002 from near-zero outputs.
- **Does it matter that the pseudo-target is not detached?** The enhancement loss currently also trains the curve, so the target can chase the render.
- **How big is real seed-to-seed noise?** Every run so far used seed 42.
- **How good are the pseudo-targets, and how consistent are they across views?** These are the baseline numbers later rungs must beat.

The plots and tables from this phase become the **motivation section of the final report**, so they must be clean, labelled and reproducible.

---

## 2. What the code does today (read this before touching anything)

All paths below are relative to `Luminance-GS/examples/`.

**Per training step** (`simple_trainer_ours.py`, `Runner.train`, roughly lines 540 to 625):

1. `pixels` = the dark training photo, `[1, H, W, 3]` in [0, 1].
2. `curve_adj_bias = self.curve_adjust(pixels, camtoworlds)` → `[1, 255]` per-view bias from `CrossAttention` (`utils.py:14`), conditioned on the image and the flattened 4×4 camera matrix.
3. `gamma_alpha_beta = self.curve_adjust_gamma(pixels, camtoworlds)` → `[1, 3]` from `CrossAttention_Curve` (`utils.py:70`).
4. `curve_adj = torch.clamp(self.curve + curve_adj_bias, 0, 1)` (**line 582**). `self.curve` is the shared global curve, initialised to `linspace(0, 1, 255)` (line 311). **No monotonicity constraint.**
5. `pixel_project` (`tools.py`) projects RGB onto a per-view orthonormal basis built from `self.axis1_para[id]` (3 params) and `self.axis2_para[id]` (2 params) (line 336), min-max normalised to [0, 1].
6. `LUT_mapping(t, curve_adj)` (`tools.py:7-17`) rounds each value to one of 255 bins and indexes the curve. **No interpolation**, so only bins that pixels land in receive gradient. The **same** curve is applied to all three projected channels (lines 591-593).
7. `pixels_enh = pixel_project_back(...)` (**line 595**) is the pseudo-enhanced target. **It is not detached.**
8. Shape prior: `gamma = 1 + 0.1*g`, `alpha = 0.5 + 0.002*a`, `beta = 1 + 0.002*b` (**lines 600-602**); `pesdo_curve = s_curve(gamma_curve(linspace, gamma), alpha, beta)`.
9. Losses (line 621): `loss = loss_regress_low + 0.5*loss_regress_enh + loss_co + 10*hist_loss`, where `loss_regress_enh` compares the enhanced render `colors_enh` to `pixels_enh` (line 615), `loss_co` is `L_spa` (`losses.py:275`), and `hist_loss` is `HistogramPriorLoss` (`losses.py:47`): MSE to the per-image HE CDF + 0.5 × MSE to the pseudo curve + 0.1 × smoothness, with the CDF weight dropping to 0.5 after step 3000.

**Other facts you will need:**

- The seed is hard-coded: `set_random_seed(42)` in `Runner.__init__` (**line 253**). `set_random_seed` (`utils.py:297`) seeds `random`, `numpy` and `torch` only.
- `cfg = tyro.cli(Config)` runs at **module import time** (around line 172) as well as in `__main__`. New config fields must be added to the `Config` dataclass.
- Checkpoints (`ckpts/ckpt_<step>.pt`) store **only the splats**, not the curve networks, global curve or colour-basis parameters. So curve diagnostics must be captured **during training**, not afterwards.
- `eval()` renders the validation views and scores the **enhanced** render against the normal-light ground truth (`high/` folder).
- Dataset (`datasets/colmap.py`): training images come from `<data_dir>/low/`, validation ground truth from `<data_dir>/high/`. The split comes from `transforms_train.json` / `transforms_test.json`. **Check whether `high/` also contains normal-light images for the training views.** If it does, pseudo-targets can be scored against real ground truth (an oracle diagnostic, never used for training).
- Data lives at `Luminance-GS/data/LOM_full/<scene>/` (git-ignored). Bart has run `buu`, `chair` and `sofa`. Check which of the five LOM scenes (`buu`, `chair`, `sofa`, `bike`, `shrub`) are present; use all that exist and tell Bart which are missing.
- Bart's machine: **Windows**, conda env `luminancegs`, gsplat 1.0.0, one NVIDIA GPU. A 10k-step run takes about **12 minutes**. His usual command (from `anaconda_10k_cmdline_output.txt`), run inside `Luminance-GS/examples`:
  ```
  python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name low --result_dir ../results/buu_low
  ```
- Reproduction numbers to compare against are in `average_results.md` (buu 21.30, chair 17.48, sofa 23.41 dB PSNR, n = 3, all seed 42).

---

## 3. Ground rules

1. **Do not change default behaviour.** Every change sits behind a new `Config` flag whose default reproduces the original code exactly (seed 42, attached pseudo-target, no diagnostics). A run with no new flags must behave as before.
2. **Diagnostics must not affect training.** Run them under `torch.no_grad()`, in `eval` mode for any module with mode-dependent layers (there are none today, but check), and do not draw from any random number generator. Restore anything you change (for example `.train()` mode).
3. **Keep diagnostics code out of the trainer where possible.** Put it in a new module `examples/diagnostics.py`, and call into it from the trainer with a few short hooks.
4. **Work on a git branch** named `rung0-diagnostics`. Commit in small, clearly described steps. Do not commit `data/`, checkpoints (`*.pt`), or rendered PNGs of every view; do commit code, the aggregated CSV/JSON tables, the summary figures and the report.
5. **Windows-friendly:** use `pathlib`, no bash-only scripts. Any batch runner is a Python script. `num_workers=4` with `persistent_workers=True` is already used by the trainer, so guard any new script entry points with `if __name__ == "__main__":`.
6. **Ask before long jobs.** Smoke tests are fine to run. Ask Bart before starting anything longer than about 30 minutes, and tell him how long it will take.
7. When something in this plan turns out to be wrong about the code, trust the code, note the discrepancy in the report, and adapt.

---

## 4. The plan

### Phase 0: orient and confirm (no code changes)

1. Read `simple_trainer_ours.py`, `tools.py`, `losses.py`, `utils.py` and `datasets/colmap.py`, and confirm the facts in section 2 (line numbers may have drifted slightly).
2. List which LOM scenes exist under `Luminance-GS/data/LOM_full/` and whether each has `low/`, `high/`, `transforms_train.json`, `transforms_test.json`.
3. For one scene, check whether `high/` contains the **training** view filenames, not just the validation ones.
4. Confirm the conda env imports work: `python -c "import torch, gsplat; print(torch.__version__, torch.cuda.is_available())"`.

**STOP AND REPORT** to Bart: scenes available, whether training-view ground truth exists, anything in section 2 that was wrong.

### Phase 1: add flags and hooks to the trainer

Add these `Config` fields (all defaults preserve current behaviour):

| Field | Default | Effect |
| --- | --- | --- |
| `seed: int` | `42` | Passed to `set_random_seed(cfg.seed)` instead of the literal 42. |
| `detach_pseudo_target: bool` | `False` | If true, `pixels_enh = pixels_enh.detach()` before the losses, so the enhancement loss stops training the curve. |
| `diagnostics: bool` | `False` | Turns on all logging below. |
| `diag_steps: List[int]` | `[1, 500, 1000, 2000, 3000, 4000, 5000, 7000, 10000]` | Steps (1-indexed, same convention as `eval_steps`) at which to snapshot every training view's curve. |

Also, when `diagnostics` is on, log each loss component separately every `tb_every` steps, both to TensorBoard and to `<result_dir>/diagnostics/loss_log.csv`: `loss_regress_low`, `loss_regress_enh`, `loss_co`, `hist_loss`, and the three parts of `hist_loss` (CDF term, pseudo-curve term, smoothness term). Getting the parts may need `HistogramPriorLoss.forward` to optionally return them; keep its default return value unchanged.

### Phase 2: curve snapshots (`diagnostics.py`)

At each `diag_step`, loop over **every training view** (build a separate `DataLoader` over `self.trainset` with `shuffle=False`, `num_workers=0`) and, under `torch.no_grad()`, recompute exactly what the training step computes for that view:

- `curve_global` = `self.curve` (`[255]`)
- `curve_bias` (`[255]`), `curve_preclamp` = global + bias, `curve_adj` = clamped
- `gamma`, `alpha`, `beta` **after** the scaling at lines 600-602, and the resulting `pesdo_curve` (`[255]`)
- `cdf_target` from `HistogramPriorLoss.compute_histogram_equalization(pixels)` (`[255]`)
- `axis1`, `axis2` colour-basis parameters for the view
- `bin_hist`: a 255-bin histogram of the integer LUT indices hit by the three projected channels (reuse the same rounding as `LUT_mapping`)
- image name and mean brightness of the dark input

Save one file per step: `<result_dir>/diagnostics/curves_step{step:05d}.npz` with arrays stacked over views, plus the view names.

Refactor the per-view curve computation into one helper used by **both** the training step and the snapshot, so they cannot drift apart. Verify on one view that the helper reproduces the training step's `curve_adj` exactly.

### Phase 3: pseudo-target snapshot (end of training only)

After the final step, for every training view, under `torch.no_grad()`:

1. Compute `pixels_enh` (the pseudo-target) and render `colors_enh`, `colors_low` and the depth for that view (`render_mode="RGB+ED"`, as `eval()` does).
2. Save per view: the pseudo-target as PNG (only for 3 fixed example views, to keep the folder small), and for **all** views the numbers below in `<result_dir>/diagnostics/pseudo_targets.csv`:
   - mean brightness of input, pseudo-target and enhanced render
   - L1 and PSNR between enhanced render and pseudo-target (how well the 3DGS fits its own targets)
   - **if training-view ground truth exists in `high/`:** PSNR, SSIM, and **exposure-aligned PSNR** of both the pseudo-target and the enhanced render against the ground truth. Exposure-aligned means scaling the prediction by `mean(gt) / mean(pred)` and clamping to [0, 1] before scoring (the "GT-mean" protocol). This is an oracle diagnostic only.
3. **Cross-view consistency of pseudo-targets.** For each training view i, pick its 2 nearest training views j by camera-centre distance. Using the trained model's rendered depth at view i, unproject every pixel to 3D, project into view j, and bilinearly sample view j's pseudo-target. Keep only pixels that land inside view j and pass an occlusion check (projected depth within 5% of the depth rendered at view j). Report the masked mean L1 between pseudo-target i and the warped pseudo-target j, and the fraction of valid pixels. For reference, compute the same number for (a) the enhanced renders, which should be near zero apart from view-dependent effects, and (b) the training-view ground truth if it exists, which sets the "natural" level of view-dependent difference. Save to `<result_dir>/diagnostics/consistency.csv`.

Get the camera conventions right: gsplat uses `camtoworlds` (camera-to-world, OpenCV convention: x right, y down, z forward) and intrinsics `Ks`. Check the warp on one pair by visual inspection of a warped image before trusting the numbers, and save that check image.

### Phase 4: smoke test

Run a short job with diagnostics on, for example:

```
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name low --result_dir ../results/rung0/smoke --max_steps 600 --eval_steps 600 --save_steps 600 --diagnostics --diag_steps 1 300 600
```

Check that every expected file appears, the arrays have sensible shapes and ranges, the warp check image looks right, and the run did not crash at the end. Also confirm a run **without** the new flags still starts and trains normally.

**STOP AND REPORT** to Bart with what was built, the smoke-test outputs, and the time and disk cost of diagnostics per run.

### Phase 5: experiment batch (ask Bart before starting)

Write `examples/run_rung0.py`, a resumable batch runner that skips any run whose `stats/val_step9999.json` already exists and writes a log per run. Runs, at the default 10,000 steps with `--diagnostics` on:

| Variant | Flags | Scenes | Seeds |
| --- | --- | --- | --- |
| `baseline` | (none) | all available | 0, 1, 2 |
| `detach` | `--detach_pseudo_target` | all available | 0, 1, 2 |

Result folders: `Luminance-GS/results/rung0/<scene>_<variant>_seed<seed>/`.

With 5 scenes that is 30 runs, about 6 to 7 hours plus diagnostics overhead, so it is an overnight job. Tell Bart the exact estimate from the smoke test and wait for his go-ahead. He may prefer to launch it himself.

Note: seeds 0, 1, 2 differ from the seed-42 runs in `average_results.md`. That is intended; it measures real seed variance. Keep the old seed-42 numbers as an extra reference row.

### Phase 6: analysis and report

Write `examples/rung0_analyze.py`, which reads every run folder and produces `Luminance-GS/results/rung0/summary/` containing CSV tables, PNG figures (matplotlib, labelled axes with units, readable fonts, one idea per figure), and `Rung0_results.md` at the **repository root**.

Compute and plot:

1. **Monotonicity** of `curve_adj` per view and step: number of decreasing steps (`L[i+1] < L[i] - 1e-4`), total negative mass `sum(max(0, L[i] - L[i+1]))`, and fraction of entries clamped at 0 or 1 before clamping. Plot over training steps, mean and range across views and seeds, per scene.
2. **Curve shapes:** for one representative scene and seed, overlay every view's final curve with the global curve, the CDF target and the pseudo curve.
3. **Cross-view spread:** per-bin std of `curve_adj` across views, averaged over bins; and ‖bias‖ relative to ‖global − identity‖.
4. **Effective degrees of freedom:** PCA over the stacked final per-view curves (all views, per scene): components needed for 95% and 99% of variance.
5. **Compact monotone fit (key result for Rung 1):** fit every final learned curve with a monotone piecewise-linear curve with K equally spaced knots for K ∈ {4, 8, 16, 32, 64} (least squares with non-negative increments, for example `scipy.optimize.nnls` on an increment basis or isotonic regression on knot values). Report mean and max absolute error in units of 1/255. Plot error vs K.
6. **Bin coverage:** fraction of the 255 bins hit by any pixel, per view and per scene, and a plot of the average `bin_hist`. This shows how much of the curve ever receives gradient.
7. **Shape-prior parameters:** `gamma`, `alpha`, `beta` over training (mean and range across views). State plainly whether they move meaningfully from 1, 0.5 and 1.
8. **Seed variance:** PSNR, SSIM, LPIPS mean ± std per scene for `baseline` seeds 0 to 2, beside the seed-42 numbers from `average_results.md`.
9. **Detach ablation:** paired per-scene, per-seed differences (`detach − baseline`) in PSNR, SSIM, LPIPS, with mean and std. Say whether the difference exceeds seed noise.
10. **Pseudo-target quality** (if training-view ground truth exists): pseudo-target vs GT and enhanced render vs GT, raw and exposure-aligned. Is the render better or worse than the targets it was trained on?
11. **Pseudo-target consistency:** the warp error from Phase 3 for pseudo-targets, renders and ground truth, per scene. This is the number Rungs 1 and 2 must improve.
12. **Loss components over training** for one scene, to see which term dominates when.

`Rung0_results.md` must open with direct answers to these questions, each in one or two sentences with the number that supports it:

- Q1. Are the learned curves non-monotonic, and by how much?
- Q2. How many parameters does a per-view curve effectively use? What K gives a monotone fit within 1/255 on average?
- Q3. Do the shape-prior parameters (gamma, alpha, beta) adapt meaningfully?
- Q4. Does detaching the pseudo-target change results beyond seed noise?
- Q5. What is the real seed-to-seed spread per scene?
- Q6. How good are the pseudo-targets against ground truth, and does the 3DGS improve on them?
- Q7. How consistent are pseudo-targets across views, compared with real ground-truth images?

Then a short **"Implications for Rung 1 and Rung 2"** section using these rules of thumb:

- If curves are essentially monotonic already (Q1), say so: Rung 1's value would come from compactness, interpolation and stability, not from fixing inversions, and Rung 2 becomes the main contribution.
- If a small K fits well (Q2), Rung 1's compact form is justified; recommend the K to start from.
- If gamma, alpha and beta barely move (Q3), the hand-designed shape prior is effectively fixed, which supports replacing it with a structured curve.
- If detaching matters (Q4), Rungs 1 to 3 should be run in both modes or the better one chosen with a reason.
- Use the seed spread (Q5) to state the smallest effect later rungs can reliably detect.

Keep claims to what the numbers show. Mark anything inferred rather than measured.

**STOP AND REPORT** to Bart with a short summary of the answers and a link to `Rung0_results.md`.

---

## 5. Deliverables checklist

- [ ] `rung0-diagnostics` branch with small, described commits
- [ ] New `Config` flags: `seed`, `detach_pseudo_target`, `diagnostics`, `diag_steps` (defaults unchanged)
- [ ] `examples/diagnostics.py` (curve snapshots, pseudo-target snapshot, consistency warp)
- [ ] Shared per-view curve helper used by both training and diagnostics
- [ ] `examples/run_rung0.py` (resumable batch runner)
- [ ] `examples/rung0_analyze.py` (tables, figures, report)
- [ ] `Luminance-GS/results/rung0/summary/` with CSVs and figures
- [ ] `Rung0_results.md` at the repo root answering Q1 to Q7

## 6. Background reading, if needed

- Luminance-GS paper: arXiv 2504.01503 (method in section 3, curve losses in Eqs. 9 to 12, supplementary section 9 for learning rates).
- Luminance-GS++ (same authors, Feb 2026): arXiv 2602.18322. Adds a clipped per-pixel residual and a colour-correction loss on top of the same curve.
- Lyu, Cui and Zhang (Jun 2026): arXiv 2606.11841. Nonlinear tone curves for low-light 3DGS pseudo ground truth; reports the gain is "curve-agnostic".
- The full research direction write-up Bart is presenting to Wei: [Luminance-GS Extension: Research Direction and Gameplan](https://claude.ai/code/artifact/d99c7f93-2da7-40ec-a8d2-c7a445add89a) (may require Bart's login).
