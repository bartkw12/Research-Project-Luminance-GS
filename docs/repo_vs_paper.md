# Paper vs repo: where the official code differs from the paper

"Paper" is Cui, Chu and Harada, *Luminance-GS*, CVPR 2025 (arXiv 2504.01503), Section 3 and Eq. 3 to 12. "Repo" is the authors' official code in `Luminance-GS/`. **All our experiments run the repo's code**, with new behaviour behind flags whose defaults reproduce it, so the baseline in every table is the authors' actual implementation. When describing the baseline in the report, describe the repo and cite these differences.

Read this when the paper and the code seem to disagree, or when writing the method or baseline section. How each piece works is in `curve_estimation.md`.

## Differences

| # | Aspect | Paper | Repo | Where in the code |
|---|---|---|---|---|
| 1 | Colour roles | base colour c renders the dark image; c_out = a·c + b renders the bright one | **Reversed.** Base SH colours render the bright image; `c·adjust_k + adjust_b` renders the dark one | `rasterize_splats` |
| 2 | Colour matrix M_k | free invertible 3×3 per view (9 numbers) | orthonormal basis from 5 numbers (3 real degrees of freedom), with min-max encoding per axis | `tools.py:pixel_project` |
| 3 | Curve length | 256 | 255 | `self.curve`, `CrossAttention` output |
| 4 | Lookup | not specified | truncation `int(254·t)`, no interpolation, no gradient to pixel values | `tools.py:LUT_mapping` |
| 5 | Monotonicity | not enforced | not enforced (clamp to [0, 1] only) | `curve_forward` |
| 6 | Curve per channel | L_k applied to the mapped image | one curve for all three projected channels | `curve_forward` |
| 7 | Shape prior | L_po · L_s with learnable G, A, B; ε = 1e-4 | composition S(gamma curve); gamma = 1 + 0.1g, alpha = 0.5 + 0.002a, beta = 1 + 0.002b; input clamped at 1e-3 (lifts black to 0.30 to 0.64) | `curve_forward`, `losses.py:gamma_curve`, `s_curve` |
| 8 | CDF weight after step 3000 | 0.1 | **0.5** for low light (0.1 only for overexposure) | `HistogramPriorLoss.forward` |
| 9 | Smoothness | separate L_tv = (1/255)·Σ squared differences | inside L_hist as 0.1·mean of squared differences, then ×10 with the rest | `HistogramPriorLoss.forward` |
| 10 | Extra term | none | 0.01·MSE(P_k, CDF_k) | `HistogramPriorLoss.forward` |
| 11 | CDF source | input image | channel mean of a ¼-size image, 255 bins | `HistogramPriorLoss.compute_histogram_equalization` |
| 12 | Regression weights | L_in + L_out with equal weight | L_low + **0.5**·L_enh | `train()` |
| 13 | Pseudo-target gradient | not discussed | not detached, so L_enh trains the curve, the bias net and the colour basis | `train()`; our `--detach_pseudo_target` |
| 14 | Seed | not stated | hard-coded 42 | our `--seed` (default 42) |

## Why each one matters for this project

- **1:** the test-time image comes from the colours trained by L_enh and L_spa. Washed-out or hue-flipped pseudo-targets therefore show up directly in the test renders.
- **2 and 13:** the colour basis is trained **only** through the attached pseudo-target. Detaching the target also freezes the basis at RGB. That is the Rung 0 "detach" confound and the reason for Rung 0.5. The paper's own ablation credits M_k with only +0.04 dB on buu.
- **4 and 5:** with truncation and no monotonicity, flat or decreasing steps are possible and the basis gets no signal from the lookup. Rung 0 found the curves monotone in practice (except shrub baseline, at most 5/255), so the case for Rung 2 rests on structure and interpolation, not on monotonicity.
- **7:** the 1e-3 clamp combined with gamma ≈ 0.1 lifts the black level. It is the *inferred* cause of the grey look on bike and shrub. Rung 2 anchors F(0) = 0.
- **8, 9, 10 and the ×10 weight:** the curve terms dominate the loss (about 5× the image terms at the end), so the curve is shaped mostly by priors. Any comparison should keep these weights unless the change is the point of the experiment.
- **12:** quote 0.5·L_enh, not the paper's equal weighting, in the method section.
- **14:** the original "3 runs" of the reproduction all used seed 42 and differed only through GPU nondeterminism. Rung 0 onwards uses seeds 0, 1 and 2.

## Reported numbers

The paper reports a LOM low-light mean of 18.34 dB PSNR, 0.799 SSIM and 0.294 LPIPS, and 18.09 / 0.877 / 0.193 on buu. Our seed-42 reproduction gives 21.30 dB PSNR on buu (`Luminance-GS/results/reproduction/average_results.md`). The authors' README notes they fixed training bugs in May 2026, after publication, which may explain part of the gap (*inferred*).

## Additions on our branches (not in the official code)

All default to off or to the official value. `--seed`, `--detach_pseudo_target`, `--diagnostics`, `--diag_steps` (Rung 0); `--freeze_color_basis`, `--detach_curve_in_target`, `--grad_check`, `--rescore_run`, `--ref_depth_run` (Rung 0.5). Rung 0 checked that a run with no new flags is bit-identical in the curve helper and keeps the same training order.
