# Rung 0.5: Untangling the "detach" result (prompt for Claude Code)

> **How to use this file:** check out the `rung0-diagnostics` branch, open Claude Code in the repository root (the folder that contains this file, `Rung0_results.md` and the `Luminance-GS/` code folder) and say:
> *"Read `Rung0.5_diagnostics_plan.md` and carry out the plan. Start with Phase 0 and stop at each checkpoint marked STOP AND REPORT."*

---

## 1. Context

### Who you are working for

You are helping **Bart**, a Master of Engineering student in Computer Engineering at McMaster University, with his ECE 701 independent research project (supervisor Prof. Chen, PhD mentor Wei). Experiments must be frozen by **5 December 2026**; the report follows.

The project extends **Luminance-GS** (Cui, Chu, Harada, CVPR 2025, arXiv 2504.01503), which reconstructs a normally lit 3D Gaussian Splatting scene from low-light photos with no normal-light supervision. The code in `Luminance-GS/` is the authors' official implementation on gsplat. On the `rung0-diagnostics` branch it already has diagnostics hooks added by an earlier Claude Code session.

### Where the project stands

**Read `Rung0_results.md` (repository root) and `Rung0_diagnostics_plan.md` first.** In short:

- **Research direction:** replace Luminance-GS's free-form, view-adaptive tone curve with an explicit model: one shared monotone curve F, anchored at black, plus **one exposure value per view**, `L_k(x) = F(min(1, 2^{s_k} x))`. This is "Rung 2".
- **Rung 0** (done, 30 runs: 5 LOM scenes × {baseline, detach} × seeds 0 to 2) showed that:
  - the learned curves are already monotonic;
  - the 255-value per-view curve bias is almost the same for every view, and the remaining variation is one-dimensional and tracks input brightness (|r| ≥ 0.97);
  - the S-curve prior never moves;
  - output brightness is set by the histogram prior;
  - **detaching the pseudo-target** (`--detach_pseudo_target`) improved brightness-matched validation PSNR by about +2.2 dB on chair, +2.2 dB on sofa and +1.9 dB on bike, at 5 to 19× seed noise. It also improved bike LPIPS by 0.118 and made runs about 40% faster.

### Why Rung 0.5 is needed

**The detach result is confounded.** On this branch, the per-view colour basis (`self.axis1_para`, `self.axis2_para`, the paper's per-view colour matrix M_k) enters the loss **only** through the pseudo-target `pixels_enh`:

- The parameters are created at `simple_trainer_ours.py:349-350` and optimised by `self.sat_optimizers` (line 352, Adam, lr 2e-4, weight decay 1e-4).
- They are used only inside `Runner.curve_forward` (line 473; basis built at lines 485-487), which projects the dark image onto the basis, applies the LUT and projects back.
- `pixels_enh` is the only output of that path that reaches a loss (`loss_regress_enh`).

So `--detach_pseudo_target` (lines 642-643) did two things at once:

1. It stopped the enhancement loss from training the **curve** (the target no longer "chases" the render).
2. It cut **all** gradient to the colour basis. Adam skips parameters whose `.grad` is `None`, so the basis stayed exactly at its RGB initialisation (zeros). The per-view colour matrix was effectively switched off.

There is visual evidence that (2) matters. In `Luminance-GS/results/rung0/summary/figures/qual_bike_baseline.png`, the baseline pseudo-target turns one yellow wheel red while the other stays yellow. In `qual_bike_detach.png`, both stay yellow. The paper's own ablation (Table 4) credits M_k with only +0.04 dB PSNR on buu low-light.

### What Rung 0.5 decides

A 2 × 2 factorial design separates the two effects:

| | Colour basis learned | Colour basis frozen at RGB |
| --- | --- | --- |
| **Curve trained by the enhancement loss** | `baseline` (exists) | **`frozenbasis` (new)** |
| **Curve not trained by the enhancement loss** | **`detachcurve` (new)** | `detach` (exists) |

The outcome decides two design choices for Rung 2:

- Whether Rung 2 keeps a per-view colour matrix or uses a fixed RGB basis.
- Whether Rung 2's curve is trained attached (with image gradient) or detached (priors only).

It also produces the first test of hypothesis **H3**: "a fixed colour basis removes the hue flips and accounts for most of the detach gain".

Rung 0.5 also fixes one measurement problem from Rung 0. The cross-view consistency metric warped each run's images with that run's own depth, so the ground-truth reference level moved between variants (bike: 0.066 baseline vs 0.043 detach). Rung 0.5 re-scores **all** runs with **one shared reference depth per scene**. It also adds **brightness-invariant colour metrics**, so hue flips are measured directly instead of eyeballed.

**Out of scope:** anything belonging to Rung 2. That includes the new curve parameterisation, the exposure model, fixing the gamma-prior black level and the varying-exposure dataset. Do not start them.

---

## 2. Ground rules

1. **Branch:** create `rung05-diagnostics` from the current `rung0-diagnostics`. Commit in small, clearly described steps.
2. **Defaults unchanged:** every new behaviour sits behind a new `Config` flag whose default reproduces the current code exactly. A run with no new flags must behave as before.
3. **Do not change the Rung 0 outputs.** Leave `Luminance-GS/results/rung0/summary/` and `Rung0_results.md` as they are. Write new outputs to `Luminance-GS/results/rung0/summary_rung05/` and `Rung0.5_results.md`. New training runs go in the same root (`Luminance-GS/results/rung0/`) so one analysis can read all four cells.
4. **Diagnostics must not affect training.** Keep using `rng_guard()`, `eval_mode()` and `trainset[i]` (not a DataLoader) as `diagnostics.py` already does.
5. **Windows:** Bart's machine runs Windows with conda env `luminancegs`, gsplat 1.0.0 and one NVIDIA GPU. Use `pathlib`, write Python batch scripts (no bash), and guard entry points with `if __name__ == "__main__":`. Before any batch longer than an hour, **ask Bart to stop the PC from sleeping** (the Rung 0 batch lost 11 hours to sleep). For example: `powercfg /change standby-timeout-ac 0`, restored afterwards.
6. **Ask before long jobs.** Smoke tests are fine. Give Bart a time estimate and wait for his go-ahead before anything over about 30 minutes.
7. **Git hygiene:** follow the existing `.gitignore`. Commit code, per-run `cfg.json`, `stats/*.json`, `diagnostics/*.csv`, `rescore/*.csv`, the new summary folder and the report. Never commit checkpoints, `.npz`, `.pt` or per-view PNGs.
8. If something in this plan is wrong about the code, trust the code, note the discrepancy in the report and adapt.

---

## 3. The plan

### Phase 0: confirm the confound (no code changes, minutes)

All Rung 0 run folders are on Bart's disk under `Luminance-GS/results/rung0/<scene>_<variant>_seed<seed>/`. Checkpoints, `.npz` and `.pt` files are git-ignored, so they exist only locally.

1. For every `*_detach_seed*` run, load `diagnostics/curve_state.pt` and check that `axis1` and `axis2` are **exactly zero** for every view. Cross-check with `diagnostics/curves_step10000.npz`.
2. For every `*_baseline_seed*` run, measure how far the learned basis moved. Rebuild the three basis vectors exactly as `tools.pixel_project` does: `normal = axis1 + [1,0,0]`; `normal2` from `axis2 + [1,0]` via the formula in `pixel_project`; `normal3` = cross product; all normalised. Report, per scene, the mean and max angle (degrees) between each basis vector and its RGB starting axis, and the largest rotation across views.
3. Check that every Rung 0 run still has `ckpts/ckpt_9999.pt` and `diagnostics/curve_state.pt`. Phase 3 needs both.

**STOP AND REPORT** to Bart:
- whether the confound is confirmed (detach axes exactly zero);
- the size of the baseline basis rotations;
- whether all checkpoints are present.

If the detach axes are **not** zero, the confound does not exist: stop and say so, because the rest of this plan is then unnecessary. If the baseline rotations are tiny (well under 1°), say so too: Rung 0.5 is still worth running, but a null result for `frozenbasis` becomes likely.

### Phase 1: two new training flags

Add to `Config` (defaults keep current behaviour):

| Field | Default | Effect |
| --- | --- | --- |
| `freeze_color_basis: bool` | `False` | In `curve_forward`, use `self.axis1_para[id].detach()` and `self.axis2_para[id].detach()`. The basis stays at its zero (RGB) initialisation while the curve still receives gradient from the enhancement loss. |
| `detach_curve_in_target: bool` | `False` | In `curve_forward`, pass `curve_adj.detach()` to the three `LUT_mapping` calls (line 492 onward) but return the **non-detached** `curve_adj` for `hist_loss`. The curve then gets gradient only from the priors, while the colour basis still gets gradient through `pixel_project_back`. |
| `grad_check: bool` | `False` | At the first training step only, print which parameters receive gradient from `loss_regress_enh`. Use `torch.autograd.grad(loss_regress_enh, [self.curve, self.axis1_para[id], self.axis2_para[id]], allow_unused=True, retain_graph=True)` and print, for each, whether the gradient is `None` or its norm. Training then continues unchanged. |

Notes:
- `curve_forward` is shared with `diagnostics.py`. Read the flags from `self.cfg` inside it so training and diagnostics stay identical.
- `LUT_mapping` converts its input to integer indices, so no gradient flows to the projected values through the lookup. The basis gradient in `detachcurve` comes only through `pixel_project_back` (the decode with `t_max`/`t_min` and the basis vectors). Confirm with `--grad_check` that it is non-zero.
- With `freeze_color_basis`, add an end-of-training assertion (only when the flag is on) that every `axis1_para`/`axis2_para` is still exactly zero.
- Expected `--grad_check` output for each configuration (verify all four):

| Flags | `curve` gets grad from `loss_regress_enh` | basis gets grad from `loss_regress_enh` |
| --- | --- | --- |
| none (baseline) | yes | yes |
| `--freeze_color_basis` | yes | no (`None`) |
| `--detach_curve_in_target` | no (`None`) | yes |
| `--detach_pseudo_target` | no | no |

### Phase 2: smoke tests

Run about 300 steps for each of the four flag sets on `buu` with `--grad_check --diagnostics`, for example:

```
python simple_trainer_ours.py --data_dir ../data/LOM_full/buu --exp_name low --result_dir ../results/rung0/smoke05_frozenbasis --max_steps 300 --eval_steps 300 --save_steps 300 --diag_steps 1 300 --diagnostics --grad_check --freeze_color_basis
```

Check that:
- the grad-check table above holds;
- the runs finish and write `stats/val_step0299.json`, `diagnostics/curve_state.pt` and `diagnostics/pseudo_targets.csv`;
- the frozen-basis assertion passes;
- a run with no new flags still trains normally;
- the per-step time of each variant is recorded (with `detachcurve` expected to be faster, since there is no backward pass through the LUT gather).

`smoke*` folders are already git-ignored.

### Phase 3: re-scoring mode (no training)

Rung 0.5 needs two new measurements for **all 60 runs** (30 existing, 30 new): colour metrics and consistency against a shared reference depth. Re-training is unnecessary, because each run has its splats (`ckpts/ckpt_9999.pt`) and its curve modules and basis (`diagnostics/curve_state.pt`).

Add a re-scoring mode to the trainer:

| Field | Default | Effect |
| --- | --- | --- |
| `rescore_run: Optional[str]` | `None` | Path to a finished run folder. Build the `Runner` as usual, then skip training and load the splats from `<run>/ckpts/ckpt_9999.pt` (as `main()` does for `--ckpt`). Load `curve`, `curve_adjust`, `curve_adjust_gamma`, `axis1` and `axis2` from `<run>/diagnostics/curve_state.pt`, then run the scoring below. |
| `ref_depth_run: Optional[str]` | `None` | Run folder whose splats provide the depth and alpha used for warping. If unset, use the run's own splats, which reproduces Rung 0. |

Invoke it with the run's own `--data_dir`/`--exp_name` (read them from the run's `cfg.json`) and `--result_dir <run>/rescore`, so the original run folder is never overwritten. `Runner.__init__` creates `ckpts/`, `renders/`, `tb/` and similar folders under the result dir, so extend `.gitignore` with `Luminance-GS/results/rung0/*/rescore/*` and then un-ignore only `*.csv` and `*.json` inside it.

Scoring, inside `Diagnostics` and under `rng_guard()`/`eval_mode()`:

1. **Training views** (as `_score_views` does now): recompute pseudo-targets with `curve_forward`, render the enhanced image, load the training-view ground truth with `load_train_gt`, and compute the existing metrics plus the colour metrics below. Use the **reference run's** depth and alpha for the warp, and compute the cross-view consistency (`_consistency`/`_warp`) for `pseudo`, `render` and `gt` with that shared depth. Write `rescore/pseudo_targets.csv` and `rescore/consistency.csv`.
2. **Validation views:** render the enhanced image as `eval()` does and compute PSNR, SSIM, LPIPS, brightness-matched PSNR and the colour metrics against the validation ground truth. Write `rescore/val_metrics.csv` (per view) and `rescore/val_summary.json`.
3. **Example images:** for one fixed bike view and one fixed shrub view, save `rescore/example_<view>_in_pseudo_render_gt.png` (git-ignored PNG).

**Colour metrics** (add helpers to `diagnostics.py`; all on [0, 1] RGB, pixels where the ground truth is too dark excluded):

- `ang_err_deg`: mean per-pixel angle between the predicted and ground-truth RGB vectors, `arccos(<p, g> / (|p| |g|))` in degrees, over pixels with `|g| > 0.05·√3`. Also report `ang_err_p95_deg`. This is brightness-invariant, so it complements brightness-matched PSNR.
- `hue_err_sat_deg`: mean circular hue difference (HSV hue, degrees) over pixels whose ground-truth saturation > 0.3 and value > 0.1. This is the direct "red wheel instead of yellow" measure.
- `sat_ratio`: mean HSV saturation of the prediction divided by that of the ground truth, over the same mask as `ang_err_deg`. Values below 1 mean washed out.

**Validate the re-scoring before using it.** Re-score one existing Rung 0 run (for example `buu_baseline_seed0`) with `ref_depth_run` unset. Its `pseudo_psnr`, `render_psnr` and consistency numbers must match that run's original `diagnostics/*.csv` to within about 1e-3, and its validation PSNR must match `stats/val_step9999.json` to within about 0.01 dB. If they do not match, find out why before going further.

Then write `examples/rescore_runs.py`: a resumable batch script that re-scores every run under `results/rung0/` that has a `ckpts/ckpt_9999.pt`, skips runs whose `rescore/val_summary.json` exists, uses `<scene>_baseline_seed0` as `ref_depth_run` for every run of that scene, and logs to `results/rung0/logs/rescore_<run>.log`. Each re-score should take about a minute. Measure it in the smoke test.

**STOP AND REPORT** to Bart:
- the grad-check table;
- smoke-test timings per variant;
- the re-scoring validation result;
- the time estimate for Phase 4.

### Phase 4: the batch (ask Bart first)

1. Extend `VARIANTS` in `examples/run_rung0.py` with `"frozenbasis": ["--freeze_color_basis"]` and `"detachcurve": ["--detach_curve_in_target"]`. The runner is resumable, so running it with `--variants frozenbasis detachcurve` trains only the 30 new runs: 5 scenes × 2 variants × seeds 0, 1, 2, at 10,000 steps with `--diagnostics`.
2. Then run `python rescore_runs.py` over all 60 runs.

Estimated time from Rung 0's timings:
- `frozenbasis`: about 12 to 14 minutes per run, so about 3.3 hours.
- `detachcurve`: probably about 8 minutes per run, so about 2 hours.
- Re-scoring: about 1 hour.

That is about 6 to 7 hours in total. Replace these numbers with the smoke-test measurements when you ask Bart.

### Phase 5: analysis and report

Write `examples/rung05_analyze.py`. Reuse helpers from `rung0_analyze.py` by importing them, not copying. It reads all four variants and writes CSVs, figures and `tables.md` to `Luminance-GS/results/rung0/summary_rung05/`. Use the existing figure style: labelled axes with units, one idea per figure, the same colours for `baseline` and `detach`, and two new colours for the new cells.

Compute, per scene and pooled over scenes:

1. **Validation metrics per cell:** mean ± std over seeds for PSNR, SSIM, LPIPS, brightness-matched PSNR, `ang_err_deg`, `hue_err_sat_deg` and `sat_ratio`.
2. **Factorial effects**, paired by seed, with mean ± std over seeds and the effect divided by pooled seed std, for each metric above:
   - basis effect = ½[(frozenbasis − baseline) + (detach − detachcurve)]
   - curve-detach effect = ½[(detachcurve − baseline) + (detach − frozenbasis)]
   - interaction = (detach − frozenbasis) − (detachcurve − baseline)
   - **share of the detach gain explained by freezing the basis** = (frozenbasis − baseline) / (detach − baseline), for brightness-matched PSNR, LPIPS and `hue_err_sat_deg`. Report only where the detach gain exceeds 2× seed noise; elsewhere write "n/a".
3. **Pseudo-target quality per cell** (training views, oracle): brightness-matched PSNR and the colour metrics.
4. **Consistency with shared depth:** relative L1 for pseudo-targets and renders per cell, with the ground-truth level now one number per scene. Compare with Rung 0's own-depth numbers to show how much the old metric moved.
5. **Basis rotation vs colour error:** for baseline runs, scatter each view's basis rotation (Phase 0) against its pseudo-target `hue_err_sat_deg`, per scene, with Spearman ρ.
6. **Timing per cell** (minutes per run, excluding `sofa_detach_seed1`, whose wall time was corrupted by sleep).
7. **Figures:**
   - a 2 × 2 grid of example pseudo-targets and renders for the bike view (one row per cell);
   - bar charts of the factorial effects with seed-noise bands;
   - the rotation-vs-hue-error scatter.

Write `Rung0.5_results.md` at the repository root, in the style of `Rung0_results.md` (measured vs *inferred* labelled). Open with direct answers:

- **Q1.** Was the confound real, and how far did the baseline colour basis rotate?
- **Q2.** How much of the detach gain (brightness-matched PSNR, LPIPS) comes from freezing the colour basis?
- **Q3.** How much comes from stopping the enhancement loss from training the curve?
- **Q4.** Is there an interaction, i.e. is `detach` more or less than the sum of the two effects?
- **Q5.** Does freezing the basis remove the hue flips (`hue_err_sat_deg`, the bike example)?
- **Q6.** With a shared reference depth, how do the four cells compare on pseudo-target consistency?
- **Q7.** Which cell is fastest, and why (inferred)?

Then an **"Implications for Rung 2"** section using these rules:

- If freezing the basis explains most (≥ about 75%) of the brightness-matched detach gain on the scenes where that gain is real, or removes the hue flips, then **Rung 2 uses a fixed RGB basis**, and H3 is supported.
- If stopping the curve's image gradient explains most of it, then **Rung 2 trains its curve from the priors only** (detached), and the colour-basis question is secondary.
- If both contribute, recommend the best of the four cells as Rung 2's default, report the 2 × 2 as an ablation in the final report, and say which cell the Rung 2 baseline comparison should use.
- Name the cell that should serve as the **reference baseline** for all Rung 2 comparisons from now on, and say whether the original official configuration should also stay in the tables (recommended: yes, as "official").

Keep claims to what the numbers show. With 3 seeds, call an effect real only if it exceeds about 2× seed noise.

**STOP AND REPORT** to Bart with a short summary of Q1 to Q7 and the implications, and a link to `Rung0.5_results.md`.

---

## 4. Deliverables checklist

- [ ] Branch `rung05-diagnostics`, small described commits
- [ ] Phase 0 confound check reported
- [ ] `Config` flags `freeze_color_basis`, `detach_curve_in_target`, `grad_check`, `rescore_run`, `ref_depth_run` (defaults unchanged)
- [ ] Colour metrics and re-scoring support in `diagnostics.py`; re-scoring validated against Rung 0
- [ ] `run_rung0.py` with the two new variants; `examples/rescore_runs.py`
- [ ] 30 new training runs and 60 re-scored runs
- [ ] `examples/rung05_analyze.py` and `Luminance-GS/results/rung0/summary_rung05/`
- [ ] `Rung0.5_results.md` at the repository root answering Q1 to Q7

## 5. Reference

- Code on `rung0-diagnostics`: `Runner.curve_forward` (`simple_trainer_ours.py:473`), training loss (lines 642-658), optimiser steps (around line 778), `diagnostics.py` (`Diagnostics`, `_score_views`, `_consistency`, `_warp`, `load_train_gt`, `psnr_gt_mean`), `run_rung0.py`, `rung0_analyze.py`.
- `Rung0_results.md` and `Luminance-GS/results/rung0/summary/tables.md` for all Rung 0 numbers.
- The research plan Bart is presenting: [Luminance-GS Extension: Research Direction and Gameplan](https://claude.ai/code/artifact/d99c7f93-2da7-40ec-a8d2-c7a445add89a) (may need Bart's login), section "Rung 0 results and revised plan".
