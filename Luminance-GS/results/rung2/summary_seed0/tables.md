# Rung 2 results (M3): H1 on LOM and the 2×2 on the varying-exposure set

**Preliminary, one seed (seed 0).** No sd, no ×noise and no verdicts: each difference is shown next to the reference cell's seed sd over seeds 0-2. Generated 2026-11-14 20:48 in 39 s by `rung2_analyze.py` (cfcde97621bd4381); plan `M2_rung2_plan.md` section 6. Every number is *measured*; readings are the report's job.

## Cells

| set | cell | runs | source | status |
|---|---|---|---|---|
| lom | official | 5/5 | rung0/<scene>_baseline_seed* | used |
| lom | detach | 5/5 | rung0/<scene>_detach_seed* | used |
| lom | rung2 | 5/5 | rung2/<scene>_rung2_seed* | used |
| vexp | official | 5/5 | vexp/<scene>_baseline_seed* | used |
| vexp | detach | 5/5 | vexp/<scene>_detach_seed* | used |
| vexp | detachexp | 5/5 | vexp/<scene>_detachexp_seed* | used |
| vexp | rung2 | 5/5 | vexp/<scene>_rung2_seed* | used |
| vexp | rung2exp | 5/5 | vexp/<scene>_rung2exp_seed* | used |


## Integrity and input checks

| check | value | tolerance | ok |
|---|---|---|---|
| max |rescore − training eval| psnr over all loaded runs | 2.54e-06 | 1e-03 | True |
| detach vs detachexp (same seed, 5 pairs): max |Δ log2 mean_pseudo| | 6.18e-06 | 1e-03 | True |
| buu seed 0: rung2 vs rung2exp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| buu seed 0: rung2 vs rung2exp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| chair seed 0: rung2 vs rung2exp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| chair seed 0: rung2 vs rung2exp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| sofa seed 0: rung2 vs rung2exp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| sofa seed 0: rung2 vs rung2exp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| bike seed 0: rung2 vs rung2exp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| bike seed 0: rung2 vs rung2exp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| shrub seed 0: rung2 vs rung2exp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| shrub seed 0: rung2 vs rung2exp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| rung2 vs rung2exp (same seed, 5 pairs): max |Δ log2 mean_pseudo| | 0.00e+00 | 1e-05 | True |
| lom bike seed 0: rung2 |mean s| | 1.66e-09 | 1e-05 | True |
| lom bike seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 2.37e-03 | 1e-02 | True |
| lom bike seed 0: trainer s vs the preview's (report only; the view order differs) | 2.82e-03 | 5e-02 | report |
| lom buu seed 0: rung2 |mean s| | 2.71e-09 | 1e-05 | True |
| lom buu seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.00e-03 | 1e-02 | True |
| lom buu seed 0: trainer s vs the preview's (report only; the view order differs) | 2.63e-03 | 5e-02 | report |
| lom chair seed 0: rung2 |mean s| | 0.00e+00 | 1e-05 | True |
| lom chair seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 8.45e-04 | 1e-02 | True |
| lom chair seed 0: trainer s vs the preview's (report only; the view order differs) | 3.32e-03 | 5e-02 | report |
| lom shrub seed 0: rung2 |mean s| | 1.99e-09 | 1e-05 | True |
| lom shrub seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 6.60e-03 | 1e-02 | True |
| lom shrub seed 0: trainer s vs the preview's (report only; the view order differs) | 8.93e-04 | 5e-02 | report |
| lom sofa seed 0: rung2 |mean s| | 5.14e-10 | 1e-05 | True |
| lom sofa seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.56e-03 | 1e-02 | True |
| lom sofa seed 0: trainer s vs the preview's (report only; the view order differs) | 5.34e-04 | 5e-02 | report |
| vexp bike seed 0: rung2 s_init vs oracle.json s_init (stops) | 2.62e-08 | 1e-06 | True |
| vexp bike seed 0: rung2 |mean s| | 1.32e-08 | 1e-05 | True |
| vexp bike seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 6.17e-03 | 1e-02 | True |
| vexp bike seed 0: trainer s vs the preview's (report only; the view order differs) | 4.36e-03 | 5e-02 | report |
| vexp buu seed 0: rung2 s_init vs oracle.json s_init (stops) | 2.72e-08 | 1e-06 | True |
| vexp buu seed 0: rung2 |mean s| | 0.00e+00 | 1e-05 | True |
| vexp buu seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.08e-03 | 1e-02 | True |
| vexp buu seed 0: trainer s vs the preview's (report only; the view order differs) | 2.66e-03 | 5e-02 | report |
| vexp chair seed 0: rung2 s_init vs oracle.json s_init (stops) | 3.60e-08 | 1e-06 | True |
| vexp chair seed 0: rung2 |mean s| | 1.66e-08 | 1e-05 | True |
| vexp chair seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.63e-03 | 1e-02 | True |
| vexp chair seed 0: trainer s vs the preview's (report only; the view order differs) | 2.31e-03 | 5e-02 | report |
| vexp shrub seed 0: rung2 s_init vs oracle.json s_init (stops) | 4.18e-08 | 1e-06 | True |
| vexp shrub seed 0: rung2 |mean s| | 1.29e-08 | 1e-05 | True |
| vexp shrub seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 4.67e-03 | 1e-02 | True |
| vexp shrub seed 0: trainer s vs the preview's (report only; the view order differs) | 3.02e-03 | 5e-02 | report |
| vexp sofa seed 0: rung2 s_init vs oracle.json s_init (stops) | 2.97e-08 | 1e-06 | True |
| vexp sofa seed 0: rung2 |mean s| | 8.22e-09 | 1e-05 | True |
| vexp sofa seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.92e-03 | 1e-02 | True |
| vexp sofa seed 0: trainer s vs the preview's (report only; the view order differs) | 2.60e-04 | 5e-02 | report |


## Decision rules (plan 6.5)

_One seed: no verdicts._



## R1a. H1 on LOM: validation per cell (mean ± sd over seeds; pooled = mean over scenes per seed). bm PSNR = brightness-matched PSNR

| scene | variant | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | official | 26.02 | 21.02 | 0.9088 | 0.1884 |
| buu | detach | 26.50 | 21.75 | 0.9070 | 0.1957 |
| buu | rung2 | 26.07 | 20.22 | 0.8987 | 0.1747 |
| chair | official | 20.76 | 17.36 | 0.8360 | 0.3458 |
| chair | detach | 23.12 | 18.85 | 0.8518 | 0.3126 |
| chair | rung2 | 23.76 | 20.19 | 0.8599 | 0.2963 |
| sofa | official | 24.73 | 22.68 | 0.8907 | 0.2857 |
| sofa | detach | 27.64 | 23.81 | 0.9061 | 0.2576 |
| sofa | rung2 | 27.33 | 23.60 | 0.9060 | 0.2422 |
| bike | official | 19.52 | 18.68 | 0.7555 | 0.4022 |
| bike | detach | 21.27 | 15.96 | 0.7675 | 0.2933 |
| bike | rung2 | 21.16 | 16.36 | 0.7595 | 0.3008 |
| shrub | official | 17.79 | 11.41 | 0.6031 | 0.2640 |
| shrub | detach | 17.69 | 11.22 | 0.5947 | 0.2584 |
| shrub | rung2 | 18.23 | 11.92 | 0.5906 | 0.3389 |
| pooled | official | 21.76 | 18.23 | 0.7988 | 0.2972 |
| pooled | detach | 23.25 | 18.32 | 0.8054 | 0.2635 |
| pooled | rung2 | 23.31 | 18.46 | 0.8030 | 0.2706 |
| pooled_no_shrub | official | 22.76 | 19.94 | 0.8477 | 0.3055 |
| pooled_no_shrub | detach | 24.63 | 20.09 | 0.8581 | 0.2648 |
| pooled_no_shrub | rung2 | 24.58 | 20.09 | 0.8560 | 0.2535 |


## R1b. rung2 − detach on LOM, paired by seed (seeds 0-2: ×noise, * = real; one seed: the difference next to detach's seed sd over seeds 0-2). H1 rule: bm PSNR not below −2× noise, SSIM and LPIPS not worse beyond 2× noise, in each of buu, chair, sofa and bike; shrub reported only

| scene | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| buu | -0.43 (ref sd 0.25) | -1.52 (ref sd 0.52) | -0.0083 (ref sd 0.0045) | -0.0210 (ref sd 0.0093) |
| chair | +0.63 (ref sd 0.12) | +1.34 (ref sd 0.03) | +0.0081 (ref sd 0.0007) | -0.0163 (ref sd 0.0020) |
| sofa | -0.31 (ref sd 0.40) | -0.21 (ref sd 0.29) | -0.0001 (ref sd 0.0044) | -0.0154 (ref sd 0.0057) |
| bike | -0.11 (ref sd 0.05) | +0.40 (ref sd 0.10) | -0.0080 (ref sd 0.0040) | +0.0075 (ref sd 0.0069) |
| shrub | +0.54 (ref sd 0.50) | +0.70 (ref sd 0.66) | -0.0040 (ref sd 0.0043) | +0.0805 (ref sd 0.0525) |
| pooled | +0.06 (ref sd 0.07) | +0.14 (ref sd 0.02) | -0.0024 (ref sd 0.0016) | +0.0071 (ref sd 0.0116) |
| pooled_no_shrub | -0.05 (ref sd 0.08) | +0.00 (ref sd 0.18) | -0.0020 (ref sd 0.0018) | -0.0113 (ref sd 0.0014) |


## R2. H1 extras on LOM: pseudo-target bm PSNR against GT, cross-view consistency of pseudo-targets and renders (relative L1 after the shared-depth warp, as a multiple of the GT's; the reference-depth run official seed 0 is left out), colour on validation, dark fit (dark render vs input, mean over training views), wall time (runs.csv), final Gaussians, curve parameters

| scene | variant | pseudo bm PSNR | pseudo consistency (× GT) | render consistency (× GT) | angle err (°) | sat_ratio | hue err (°) | dark fit (dB) | minutes/run | Gaussians | curve params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buu | official | 26.17 | n/a | n/a | 4.13 | 0.751 | 11.52 | 46.15 | 12.1 | 84427 | 255343 |
| buu | detach | 26.80 | 1.57 | 0.71 | 3.97 | 0.933 | 8.24 | 45.95 | 7.4 | 79376 | 255343 |
| buu | rung2 | 26.72 | 1.53 | 0.80 | 3.57 | 0.902 | 7.15 | 46.02 | 4.0 | 85288 | 55 |
| buu | rung2 − detach | -0.08 (ref sd 0.02) | -0.04 (ref sd 0.02) | +0.09 (ref sd 0.06) | -0.40 (ref sd 0.06) | -0.031 (ref sd 0.008) | -1.09 (ref sd 0.13) | +0.08 (ref sd 0.26) | -3.4 (ref sd 0.1) | +5912 (ref sd 3145) |  |
| chair | official | 20.91 | n/a | n/a | 7.53 | 0.315 | 22.32 | 48.96 | 11.7 | 46452 | 255448 |
| chair | detach | 23.43 | 0.97 | 0.38 | 5.89 | 0.495 | 18.49 | 48.97 | 7.3 | 61430 | 255448 |
| chair | rung2 | 23.48 | 1.04 | 0.39 | 5.68 | 0.532 | 18.14 | 48.86 | 3.9 | 52049 | 76 |
| chair | rung2 − detach | +0.05 (ref sd 0.00) | +0.07 (ref sd 0.00) | +0.00 (ref sd 0.01) | -0.21 (ref sd 0.03) | +0.038 (ref sd 0.003) | -0.34 (ref sd 0.19) | -0.11 (ref sd 0.05) | -3.4 (ref sd 0.2) | -9381 (ref sd 3404) |  |
| sofa | official | 26.47 | n/a | n/a | 3.45 | 0.660 | 8.10 | 47.44 | 10.8 | 33871 | 255378 |
| sofa | detach | 28.74 | 1.67 | 0.76 | 2.55 | 0.969 | 11.22 | 47.54 | 7.2 | 39598 | 255378 |
| sofa | rung2 | 28.54 | 1.82 | 1.11 | 2.55 | 0.926 | 11.72 | 47.82 | 3.7 | 41993 | 62 |
| sofa | rung2 − detach | -0.19 (ref sd 0.00) | +0.14 (ref sd 0.02) | +0.35 (ref sd 0.01) | -0.00 (ref sd 0.03) | -0.042 (ref sd 0.011) | +0.50 (ref sd 1.02) | +0.28 (ref sd 0.10) | -3.5 | +2395 (ref sd 3002) |  |
| bike | official | 19.49 | n/a | n/a | 2.63 | 0.674 | 21.32 | 47.58 | 13.8 | 176678 | 255413 |
| bike | detach | 22.49 | 1.70 | 0.58 | 2.50 | 0.488 | 33.66 | 47.85 | 8.1 | 235209 | 255413 |
| bike | rung2 | 22.51 | 1.58 | 0.56 | 2.53 | 0.527 | 37.57 | 47.57 | 4.7 | 230035 | 69 |
| bike | rung2 − detach | +0.01 (ref sd 0.01) | -0.12 (ref sd 0.02) | -0.01 (ref sd 0.01) | +0.03 (ref sd 0.02) | +0.039 (ref sd 0.010) | +3.90 (ref sd 1.88) | -0.28 (ref sd 0.12) | -3.4 (ref sd 0.0) | -5174 (ref sd 3979) |  |
| shrub | official | 17.34 | n/a | n/a | 12.96 | 0.422 | 11.60 | 40.82 | 13.7 | 856769 | 255383 |
| shrub | detach | 17.38 | 0.95 | 0.46 | 11.64 | 0.487 | 8.44 | 40.77 | 9.3 | 881809 | 255383 |
| shrub | rung2 | 17.74 | 1.33 | 0.56 | 9.01 | 0.720 | 9.90 | 40.68 | 6.5 | 890700 | 63 |
| shrub | rung2 − detach | +0.36 (ref sd 0.17) | +0.38 (ref sd 0.41) | +0.09 (ref sd 0.09) | -2.63 (ref sd 1.30) | +0.233 (ref sd 0.244) | +1.45 (ref sd 0.72) | -0.10 (ref sd 0.06) | -2.8 (ref sd 0.4) | +8891 (ref sd 17980) |  |
| pooled | official | 22.08 | n/a | n/a | 6.14 | 0.564 | 14.97 | 46.19 | 12.4 | 239639 | 255393 |
| pooled | detach | 23.77 | 1.37 | 0.58 | 5.31 | 0.674 | 16.01 | 46.22 | 7.9 | 259484 | 255393 |
| pooled | rung2 | 23.80 | 1.46 | 0.68 | 4.67 | 0.722 | 16.90 | 46.19 | 4.6 | 260013 | 65 |
| pooled | rung2 − detach | +0.03 (ref sd 0.03) | +0.09 (ref sd 0.08) | +0.10 (ref sd 0.03) | -0.64 (ref sd 0.26) | +0.047 (ref sd 0.050) | +0.89 (ref sd 0.24) | -0.02 (ref sd 0.03) | -3.3 | +529 (ref sd 4160) |  |
| pooled_no_shrub | official | 23.26 | n/a | n/a | 4.43 | 0.600 | 15.81 | 47.53 | 12.1 | 85357 | 255396 |
| pooled_no_shrub | detach | 25.36 | 1.48 | 0.61 | 3.73 | 0.721 | 17.90 | 47.58 | 7.5 | 103903 | 255396 |
| pooled_no_shrub | rung2 | 25.31 | 1.49 | 0.72 | 3.58 | 0.722 | 18.65 | 47.57 | 4.1 | 102341 | 66 |
| pooled_no_shrub | rung2 − detach | -0.05 (ref sd 0.00) | +0.01 (ref sd 0.01) | +0.11 (ref sd 0.02) | -0.14 (ref sd 0.01) | +0.001 (ref sd 0.006) | +0.74 (ref sd 0.45) | -0.01 (ref sd 0.03) | -3.4 | -1562 (ref sd 940) |  |


## R6. Side effects on the varying set: colour on validation, consistency (× GT, shared LOM depth), wall time, final Gaussians. The exp rows are the column effect within each curve row (the dark branch re-weights each view's L_low gradient, plan 2.1)

| scene | variant | sat_ratio | angle err (°) | hue err (°) | pseudo consistency (× GT) | render consistency (× GT) | minutes/run | Gaussians |
|---|---|---|---|---|---|---|---|---|
| buu | official | 0.812 | 3.77 | 9.96 | 4.00 | 2.67 | 12.5 | 141976 |
| buu | detach | 0.966 | 3.88 | 11.18 | 4.78 | 3.05 | 7.8 | 147637 |
| buu | detachexp | 0.905 | 3.32 | 8.72 | 4.78 | 1.82 | 8.0 | 107025 |
| buu | rung2 | 0.909 | 3.84 | 10.06 | 1.65 | 0.93 | 4.1 | 118676 |
| buu | rung2exp | 0.927 | 3.55 | 7.19 | 1.65 | 0.98 | 4.0 | 102520 |
| buu | detachexp − detach | -0.061 (ref sd 0.025) | -0.56 (ref sd 0.23) | -2.46 (ref sd 1.34) | -0.00 (ref sd 0.05) | -1.23 (ref sd 0.43) | +0.2 (ref sd 0.0) | -40612 (ref sd 14505) |
| buu | rung2exp − rung2 | +0.018 | -0.28 | -2.87 | +0.00 | +0.05 | -0.1 | -16156 |
| chair | official | 0.209 | 8.42 | 19.94 | 3.33 | 0.42 | 12.4 | 75680 |
| chair | detach | 0.350 | 6.96 | 14.27 | 5.36 | 1.45 | 7.7 | 120046 |
| chair | detachexp | 0.338 | 7.06 | 14.67 | 5.36 | 0.57 | 7.9 | 89591 |
| chair | rung2 | 0.575 | 5.41 | 14.25 | 1.32 | 0.41 | 3.9 | 88556 |
| chair | rung2exp | 0.566 | 5.44 | 13.52 | 1.32 | 0.40 | 3.8 | 50368 |
| chair | detachexp − detach | -0.012 (ref sd 0.031) | +0.10 (ref sd 0.24) | +0.40 (ref sd 0.61) | -0.00 (ref sd 0.04) | -0.88 (ref sd 0.44) | +0.2 (ref sd 0.0) | -30455 (ref sd 15646) |
| chair | rung2exp − rung2 | -0.008 | +0.04 | -0.73 | +0.00 | -0.01 | -0.1 | -38188 |
| sofa | official | 0.534 | 3.85 | 11.64 | 6.29 | 2.79 | 11.4 | 65713 |
| sofa | detach | 0.848 | 2.74 | 9.36 | 9.00 | 3.93 | 7.4 | 55451 |
| sofa | detachexp | 0.871 | 2.74 | 9.03 | 9.00 | 0.90 | 7.8 | 40755 |
| sofa | rung2 | 0.985 | 2.66 | 12.86 | 2.17 | 1.16 | 3.7 | 56800 |
| sofa | rung2exp | 1.018 | 2.75 | 13.10 | 2.17 | 1.26 | 3.7 | 46527 |
| sofa | detachexp − detach | +0.023 (ref sd 0.005) | -0.00 (ref sd 0.21) | -0.34 (ref sd 0.69) | +0.00 (ref sd 0.08) | -3.03 (ref sd 1.58) | +0.4 (ref sd 0.0) | -14696 (ref sd 4561) |
| sofa | rung2exp − rung2 | +0.033 | +0.09 | +0.24 | +0.00 | +0.11 | -0.0 | -10273 |
| bike | official | 0.558 | 2.82 | 17.82 | 2.27 | 0.84 | 15.0 | 205620 |
| bike | detach | 0.436 | 2.56 | 11.80 | 4.53 | 1.75 | 8.2 | 280233 |
| bike | detachexp | 0.416 | 2.55 | 10.79 | 4.53 | 0.79 | 8.7 | 235644 |
| bike | rung2 | 0.581 | 2.57 | 36.00 | 1.89 | 0.66 | 4.8 | 276025 |
| bike | rung2exp | 0.557 | 2.57 | 38.80 | 1.89 | 0.60 | 4.8 | 241316 |
| bike | detachexp − detach | -0.020 (ref sd 0.005) | -0.02 (ref sd 0.03) | -1.01 (ref sd 1.02) | -0.00 (ref sd 0.08) | -0.96 (ref sd 0.35) | +0.4 (ref sd 0.0) | -44589 (ref sd 2774) |
| bike | rung2exp − rung2 | -0.023 | -0.00 | +2.81 | +0.00 | -0.06 | -0.1 | -34709 |
| shrub | official | 0.367 | 13.93 | 11.10 | 1.71 | 0.73 | 14.4 | 876887 |
| shrub | detach | 0.403 | 13.17 | 9.55 | 1.63 | 0.60 | 9.6 | 913657 |
| shrub | detachexp | 0.392 | 13.27 | 7.11 | 1.63 | 0.42 | 9.8 | 879688 |
| shrub | rung2 | 0.759 | 8.77 | 9.10 | 1.41 | 0.58 | 6.7 | 957263 |
| shrub | rung2exp | 0.751 | 8.83 | 9.13 | 1.41 | 0.57 | 6.6 | 932445 |
| shrub | detachexp − detach | -0.011 (ref sd 0.010) | +0.10 (ref sd 0.13) | -2.44 (ref sd 1.49) | +0.00 (ref sd 0.03) | -0.19 (ref sd 0.10) | +0.3 (ref sd 0.1) | -33969 (ref sd 20689) |
| shrub | rung2exp − rung2 | -0.009 | +0.06 | +0.03 | +0.00 | -0.01 | -0.2 | -24818 |
| pooled | official | 0.496 | 6.56 | 14.09 | 3.52 | 1.49 | 13.1 | 273175 |
| pooled | detach | 0.601 | 5.86 | 11.23 | 5.06 | 2.16 | 8.1 | 303405 |
| pooled | detachexp | 0.584 | 5.79 | 10.06 | 5.06 | 0.90 | 8.4 | 270541 |
| pooled | rung2 | 0.762 | 4.65 | 16.45 | 1.69 | 0.75 | 4.7 | 299464 |
| pooled | rung2exp | 0.764 | 4.63 | 16.35 | 1.69 | 0.76 | 4.6 | 274635 |
| pooled | detachexp − detach | -0.016 (ref sd 0.012) | -0.07 (ref sd 0.11) | -1.17 (ref sd 0.60) | +0.00 (ref sd 0.03) | -1.26 (ref sd 0.39) | +0.3 (ref sd 0.0) | -32864 (ref sd 4974) |
| pooled | rung2exp − rung2 | +0.002 | -0.02 | -0.10 | +0.00 | +0.02 | -0.1 | -24829 |
| pooled_no_shrub | official | 0.528 | 4.71 | 14.84 | 3.97 | 1.68 | 12.8 | 122247 |
| pooled_no_shrub | detach | 0.650 | 4.04 | 11.65 | 5.92 | 2.55 | 7.8 | 150842 |
| pooled_no_shrub | detachexp | 0.632 | 3.92 | 10.80 | 5.92 | 1.02 | 8.1 | 118254 |
| pooled_no_shrub | rung2 | 0.762 | 3.62 | 18.29 | 1.76 | 0.79 | 4.1 | 135014 |
| pooled_no_shrub | rung2exp | 0.767 | 3.58 | 18.15 | 1.76 | 0.81 | 4.1 | 110183 |
| pooled_no_shrub | detachexp − detach | -0.018 (ref sd 0.013) | -0.12 (ref sd 0.12) | -0.85 (ref sd 0.38) | +0.00 (ref sd 0.04) | -1.52 (ref sd 0.48) | +0.3 (ref sd 0.0) | -32588 (ref sd 1747) |
| pooled_no_shrub | rung2exp − rung2 | +0.005 | -0.04 | -0.14 | +0.00 | +0.02 | -0.1 | -24832 |


## R3. The 2×2 on the varying-exposure set (official for reference): validation per cell

| scene | cell | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | official | 21.76 | 21.24 | 0.8835 | 0.2446 |
| buu | detach | 23.99 | 22.26 | 0.8652 | 0.2649 |
| buu | detachexp | 25.86 | 21.43 | 0.9091 | 0.1968 |
| buu | rung2 | 25.37 | 19.26 | 0.8803 | 0.2103 |
| buu | rung2exp | 26.70 | 20.62 | 0.8910 | 0.1733 |
| chair | official | 17.63 | 14.68 | 0.7921 | 0.4086 |
| chair | detach | 19.99 | 15.10 | 0.8148 | 0.3426 |
| chair | detachexp | 20.20 | 16.28 | 0.8341 | 0.3121 |
| chair | rung2 | 23.41 | 20.22 | 0.8511 | 0.2777 |
| chair | rung2exp | 23.96 | 20.64 | 0.8597 | 0.2882 |
| sofa | official | 19.46 | 17.79 | 0.8423 | 0.3852 |
| sofa | detach | 23.45 | 19.27 | 0.8716 | 0.3225 |
| sofa | detachexp | 23.66 | 22.85 | 0.8930 | 0.3080 |
| sofa | rung2 | 26.59 | 23.36 | 0.8936 | 0.2617 |
| sofa | rung2exp | 27.49 | 23.75 | 0.9055 | 0.2513 |
| bike | official | 19.10 | 14.55 | 0.7352 | 0.4155 |
| bike | detach | 20.07 | 13.34 | 0.7283 | 0.3406 |
| bike | detachexp | 20.62 | 14.90 | 0.7658 | 0.3326 |
| bike | rung2 | 21.18 | 16.51 | 0.7532 | 0.2909 |
| bike | rung2exp | 21.15 | 16.42 | 0.7570 | 0.2946 |
| shrub | official | 16.69 | 10.17 | 0.5665 | 0.2856 |
| shrub | detach | 16.68 | 9.88 | 0.5495 | 0.2967 |
| shrub | detachexp | 16.83 | 9.80 | 0.5551 | 0.2907 |
| shrub | rung2 | 18.40 | 12.10 | 0.5948 | 0.3477 |
| shrub | rung2exp | 18.42 | 12.08 | 0.5958 | 0.3475 |
| pooled | official | 18.93 | 15.69 | 0.7639 | 0.3479 |
| pooled | detach | 20.83 | 15.97 | 0.7659 | 0.3134 |
| pooled | detachexp | 21.43 | 17.05 | 0.7914 | 0.2880 |
| pooled | rung2 | 22.99 | 18.29 | 0.7946 | 0.2776 |
| pooled | rung2exp | 23.54 | 18.70 | 0.8018 | 0.2710 |
| pooled_no_shrub | official | 19.49 | 17.06 | 0.8133 | 0.3635 |
| pooled_no_shrub | detach | 21.87 | 17.49 | 0.8200 | 0.3176 |
| pooled_no_shrub | detachexp | 22.58 | 18.86 | 0.8505 | 0.2874 |
| pooled_no_shrub | rung2 | 24.14 | 19.84 | 0.8446 | 0.2601 |
| pooled_no_shrub | rung2exp | 24.83 | 20.36 | 0.8533 | 0.2518 |


## R4. Factorial effects on the varying set. Main effects and interaction: noise over the four cells (rung05_analyze.effects); simple effects: their two arms' noise (M1). One seed: ref sd = seed sd of detach and detachexp over seeds 0-2 (main effects) or of the second arm's cell

| scene | effect | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | curve model (H2): ½[(rung2 − detach) + (rung2exp − detachexp)] | +1.11 (ref sd 1.57) | -1.91 (ref sd 0.58) | -0.0015 (ref sd 0.0103) | -0.0391 (ref sd 0.0169) |
| buu | dark branch: ½[(detachexp − detach) + (rung2exp − rung2)] | +1.60 (ref sd 1.57) | +0.26 (ref sd 0.58) | +0.0273 (ref sd 0.0103) | -0.0525 (ref sd 0.0169) |
| buu | interaction: (rung2exp − rung2) − (detachexp − detach) | -0.54 (ref sd 1.57) | +2.19 (ref sd 0.58) | -0.0332 (ref sd 0.0103) | +0.0310 (ref sd 0.0169) |
| buu | rung2 − detach (shared column) | +1.38 (ref sd 1.51) | -3.00 (ref sd 0.75) | +0.0151 (ref sd 0.0135) | -0.0546 (ref sd 0.0142) |
| buu | rung2exp − detachexp (exp column) | +0.84 (ref sd 1.64) | -0.81 (ref sd 0.33) | -0.0181 (ref sd 0.0056) | -0.0236 (ref sd 0.0193) |
| buu | detachexp − detach (detach row) | +1.88 (ref sd 1.51) | -0.83 (ref sd 0.75) | +0.0439 (ref sd 0.0135) | -0.0680 (ref sd 0.0142) |
| buu | rung2exp − rung2 (rung2 row) | +1.33 (ref sd 1.51) | +1.36 (ref sd 0.75) | +0.0107 (ref sd 0.0135) | -0.0370 (ref sd 0.0142) |
| chair | curve model (H2): ½[(rung2 − detach) + (rung2exp − detachexp)] | +3.59 (ref sd 0.60) | +4.74 (ref sd 1.04) | +0.0309 (ref sd 0.0180) | -0.0444 (ref sd 0.0377) |
| chair | dark branch: ½[(detachexp − detach) + (rung2exp − rung2)] | +0.38 (ref sd 0.60) | +0.80 (ref sd 1.04) | +0.0140 (ref sd 0.0180) | -0.0100 (ref sd 0.0377) |
| chair | interaction: (rung2exp − rung2) − (detachexp − detach) | +0.34 (ref sd 0.60) | -0.77 (ref sd 1.04) | -0.0107 (ref sd 0.0180) | +0.0410 (ref sd 0.0377) |
| chair | rung2 − detach (shared column) | +3.42 (ref sd 0.82) | +5.13 (ref sd 1.38) | +0.0363 (ref sd 0.0253) | -0.0649 (ref sd 0.0533) |
| chair | rung2exp − detachexp (exp column) | +3.76 (ref sd 0.21) | +4.36 (ref sd 0.50) | +0.0256 (ref sd 0.0026) | -0.0240 (ref sd 0.0002) |
| chair | detachexp − detach (detach row) | +0.21 (ref sd 0.82) | +1.18 (ref sd 1.38) | +0.0193 (ref sd 0.0253) | -0.0305 (ref sd 0.0533) |
| chair | rung2exp − rung2 (rung2 row) | +0.55 (ref sd 0.82) | +0.41 (ref sd 1.38) | +0.0086 (ref sd 0.0253) | +0.0105 (ref sd 0.0533) |
| sofa | curve model (H2): ½[(rung2 − detach) + (rung2exp − detachexp)] | +3.49 (ref sd 1.17) | +2.50 (ref sd 2.17) | +0.0172 (ref sd 0.0108) | -0.0588 (ref sd 0.0239) |
| sofa | dark branch: ½[(detachexp − detach) + (rung2exp − rung2)] | +0.56 (ref sd 1.17) | +1.99 (ref sd 2.17) | +0.0167 (ref sd 0.0108) | -0.0125 (ref sd 0.0239) |
| sofa | interaction: (rung2exp − rung2) − (detachexp − detach) | +0.69 (ref sd 1.17) | -3.19 (ref sd 2.17) | -0.0095 (ref sd 0.0108) | +0.0041 (ref sd 0.0239) |
| sofa | rung2 − detach (shared column) | +3.14 (ref sd 1.11) | +4.09 (ref sd 2.59) | +0.0220 (ref sd 0.0132) | -0.0609 (ref sd 0.0235) |
| sofa | rung2exp − detachexp (exp column) | +3.83 (ref sd 1.23) | +0.90 (ref sd 1.66) | +0.0125 (ref sd 0.0075) | -0.0567 (ref sd 0.0244) |
| sofa | detachexp − detach (detach row) | +0.21 (ref sd 1.11) | +3.59 (ref sd 2.59) | +0.0214 (ref sd 0.0132) | -0.0145 (ref sd 0.0235) |
| sofa | rung2exp − rung2 (rung2 row) | +0.90 (ref sd 1.11) | +0.40 (ref sd 2.59) | +0.0119 (ref sd 0.0132) | -0.0104 (ref sd 0.0235) |
| bike | curve model (H2): ½[(rung2 − detach) + (rung2exp − detachexp)] | +0.82 (ref sd 0.29) | +2.34 (ref sd 0.87) | +0.0081 (ref sd 0.0156) | -0.0439 (ref sd 0.0102) |
| bike | dark branch: ½[(detachexp − detach) + (rung2exp − rung2)] | +0.26 (ref sd 0.29) | +0.73 (ref sd 0.87) | +0.0206 (ref sd 0.0156) | -0.0021 (ref sd 0.0102) |
| bike | interaction: (rung2exp − rung2) − (detachexp − detach) | -0.58 (ref sd 0.29) | -1.65 (ref sd 0.87) | -0.0337 (ref sd 0.0156) | +0.0117 (ref sd 0.0102) |
| bike | rung2 − detach (shared column) | +1.11 (ref sd 0.35) | +3.17 (ref sd 1.18) | +0.0249 (ref sd 0.0216) | -0.0497 (ref sd 0.0120) |
| bike | rung2exp − detachexp (exp column) | +0.53 (ref sd 0.21) | +1.52 (ref sd 0.32) | -0.0088 (ref sd 0.0046) | -0.0380 (ref sd 0.0080) |
| bike | detachexp − detach (detach row) | +0.55 (ref sd 0.35) | +1.55 (ref sd 1.18) | +0.0375 (ref sd 0.0216) | -0.0080 (ref sd 0.0120) |
| bike | rung2exp − rung2 (rung2 row) | -0.03 (ref sd 0.35) | -0.09 (ref sd 1.18) | +0.0038 (ref sd 0.0216) | +0.0037 (ref sd 0.0120) |
| shrub | curve model (H2): ½[(rung2 − detach) + (rung2exp − detachexp)] | +1.66 (ref sd 0.03) | +2.25 (ref sd 0.15) | +0.0430 (ref sd 0.0014) | +0.0539 (ref sd 0.0013) |
| shrub | dark branch: ½[(detachexp − detach) + (rung2exp − rung2)] | +0.08 (ref sd 0.03) | -0.05 (ref sd 0.15) | +0.0032 (ref sd 0.0014) | -0.0031 (ref sd 0.0013) |
| shrub | interaction: (rung2exp − rung2) − (detachexp − detach) | -0.14 (ref sd 0.03) | +0.06 (ref sd 0.15) | -0.0046 (ref sd 0.0014) | +0.0057 (ref sd 0.0013) |
| shrub | rung2 − detach (shared column) | +1.72 (ref sd 0.03) | +2.22 (ref sd 0.19) | +0.0453 (ref sd 0.0007) | +0.0511 (ref sd 0.0017) |
| shrub | rung2exp − detachexp (exp column) | +1.59 (ref sd 0.03) | +2.28 (ref sd 0.08) | +0.0407 (ref sd 0.0018) | +0.0568 (ref sd 0.0007) |
| shrub | detachexp − detach (detach row) | +0.15 (ref sd 0.03) | -0.08 (ref sd 0.19) | +0.0055 (ref sd 0.0007) | -0.0060 (ref sd 0.0017) |
| shrub | rung2exp − rung2 (rung2 row) | +0.01 (ref sd 0.03) | -0.02 (ref sd 0.19) | +0.0010 (ref sd 0.0007) | -0.0002 (ref sd 0.0017) |
| pooled | curve model (H2): ½[(rung2 − detach) + (rung2exp − detachexp)] | +2.13 (ref sd 0.56) | +1.99 (ref sd 0.83) | +0.0196 (ref sd 0.0090) | -0.0264 (ref sd 0.0128) |
| pooled | dark branch: ½[(detachexp − detach) + (rung2exp − rung2)] | +0.58 (ref sd 0.56) | +0.75 (ref sd 0.83) | +0.0164 (ref sd 0.0090) | -0.0160 (ref sd 0.0128) |
| pooled | interaction: (rung2exp − rung2) − (detachexp − detach) | -0.05 (ref sd 0.56) | -0.67 (ref sd 0.83) | -0.0183 (ref sd 0.0090) | +0.0187 (ref sd 0.0128) |
| pooled | rung2 − detach (shared column) | +2.16 (ref sd 0.72) | +2.32 (ref sd 1.15) | +0.0287 (ref sd 0.0126) | -0.0358 (ref sd 0.0167) |
| pooled | rung2exp − detachexp (exp column) | +2.11 (ref sd 0.31) | +1.65 (ref sd 0.25) | +0.0104 (ref sd 0.0019) | -0.0171 (ref sd 0.0069) |
| pooled | detachexp − detach (detach row) | +0.60 (ref sd 0.72) | +1.08 (ref sd 1.15) | +0.0255 (ref sd 0.0126) | -0.0254 (ref sd 0.0167) |
| pooled | rung2exp − rung2 (rung2 row) | +0.55 (ref sd 0.72) | +0.41 (ref sd 1.15) | +0.0072 (ref sd 0.0126) | -0.0067 (ref sd 0.0167) |
| pooled_no_shrub | curve model (H2): ½[(rung2 − detach) + (rung2exp − detachexp)] | +2.25 (ref sd 0.69) | +1.92 (ref sd 1.02) | +0.0137 (ref sd 0.0111) | -0.0465 (ref sd 0.0161) |
| pooled_no_shrub | dark branch: ½[(detachexp − detach) + (rung2exp − rung2)] | +0.70 (ref sd 0.69) | +0.94 (ref sd 1.02) | +0.0197 (ref sd 0.0111) | -0.0193 (ref sd 0.0161) |
| pooled_no_shrub | interaction: (rung2exp − rung2) − (detachexp − detach) | -0.02 (ref sd 0.69) | -0.85 (ref sd 1.02) | -0.0218 (ref sd 0.0111) | +0.0219 (ref sd 0.0161) |
| pooled_no_shrub | rung2 − detach (shared column) | +2.26 (ref sd 0.91) | +2.35 (ref sd 1.41) | +0.0246 (ref sd 0.0156) | -0.0575 (ref sd 0.0212) |
| pooled_no_shrub | rung2exp − detachexp (exp column) | +2.24 (ref sd 0.38) | +1.49 (ref sd 0.31) | +0.0028 (ref sd 0.0020) | -0.0356 (ref sd 0.0084) |
| pooled_no_shrub | detachexp − detach (detach row) | +0.71 (ref sd 0.91) | +1.37 (ref sd 1.41) | +0.0305 (ref sd 0.0156) | -0.0302 (ref sd 0.0212) |
| pooled_no_shrub | rung2exp − rung2 (rung2 row) | +0.69 (ref sd 0.91) | +0.52 (ref sd 1.41) | +0.0088 (ref sd 0.0156) | -0.0083 (ref sd 0.0212) |


## R5a. Pseudo-target leak (plan 6.3a): slope of log2(mean pseudo-target, varying / LOM) on u_k, same view, seed and curve row (0 = compensated, 1 = passes straight through); unpaired = log2 mean on the varying set alone; input = the dark inputs (check: about 1); render rows = the enhanced renders at the training views, per cell (LOM partner: the row's LOM run)

| scene | curve row | pseudo_leak_paired | pseudo_leak | input_leak_paired | render_leak_paired |
|---|---|---|---|---|---|
| buu | official | 0.566 | 0.588 | 1.008 |  |
| buu | detach | 0.656 | 0.676 | 1.008 |  |
| buu | rung2 | 0.001 | -0.001 | 1.008 |  |
| buu | official (render) |  |  |  | 0.465 |
| buu | detach (render) |  |  |  | 0.479 |
| buu | detachexp (render) |  |  |  | 0.296 |
| buu | rung2 (render) |  |  |  | -0.007 |
| buu | rung2exp (render) |  |  |  | -0.000 |
| buu | rung2 − detach | -0.656 (ref sd 0.004) |  |  |  |
| chair | official | 0.323 | 0.336 | 1.015 |  |
| chair | detach | 0.492 | 0.516 | 1.015 |  |
| chair | rung2 | 0.008 | 0.011 | 1.015 |  |
| chair | official (render) |  |  |  | 0.061 |
| chair | detach (render) |  |  |  | 0.166 |
| chair | detachexp (render) |  |  |  | 0.077 |
| chair | rung2 (render) |  |  |  | 0.004 |
| chair | rung2exp (render) |  |  |  | 0.002 |
| chair | rung2 − detach | -0.484 (ref sd 0.004) |  |  |  |
| sofa | official | 0.402 | 0.415 | 1.002 |  |
| sofa | detach | 0.549 | 0.559 | 1.002 |  |
| sofa | rung2 | -0.003 | -0.004 | 1.002 |  |
| sofa | official (render) |  |  |  | 0.186 |
| sofa | detach (render) |  |  |  | 0.240 |
| sofa | detachexp (render) |  |  |  | 0.081 |
| sofa | rung2 (render) |  |  |  | 0.002 |
| sofa | rung2exp (render) |  |  |  | -0.000 |
| sofa | rung2 − detach | -0.551 (ref sd 0.005) |  |  |  |
| bike | official | 0.265 | 0.239 | 0.996 |  |
| bike | detach | 0.498 | 0.470 | 0.996 |  |
| bike | rung2 | 0.029 | 0.031 | 0.996 |  |
| bike | official (render) |  |  |  | 0.089 |
| bike | detach (render) |  |  |  | 0.210 |
| bike | detachexp (render) |  |  |  | 0.085 |
| bike | rung2 (render) |  |  |  | 0.002 |
| bike | rung2exp (render) |  |  |  | 0.003 |
| bike | rung2 − detach | -0.469 (ref sd 0.008) |  |  |  |
| shrub | official | 0.369 | 0.356 | 1.009 |  |
| shrub | detach | 0.362 | 0.348 | 1.009 |  |
| shrub | rung2 | 0.026 | 0.024 | 1.009 |  |
| shrub | official (render) |  |  |  | 0.181 |
| shrub | detach (render) |  |  |  | 0.155 |
| shrub | detachexp (render) |  |  |  | 0.054 |
| shrub | rung2 (render) |  |  |  | 0.012 |
| shrub | rung2exp (render) |  |  |  | 0.004 |
| shrub | rung2 − detach | -0.336 (ref sd 0.005) |  |  |  |
| pooled | official | 0.385 | 0.387 | 1.006 |  |
| pooled | detach | 0.511 | 0.514 | 1.006 |  |
| pooled | rung2 | 0.012 | 0.012 | 1.006 |  |
| pooled | official (render) |  |  |  | 0.197 |
| pooled | detach (render) |  |  |  | 0.250 |
| pooled | detachexp (render) |  |  |  | 0.118 |
| pooled | rung2 (render) |  |  |  | 0.002 |
| pooled | rung2exp (render) |  |  |  | 0.002 |
| pooled | rung2 − detach | -0.499 (ref sd 0.001) |  |  |  |


## R5b. Dark-branch fit (plan 6.3b): PSNR of the dark render as its loss sees it against the input, mean over training views; the row's LOM level; the column effect within each curve row

| scene | cell | dark fit, varying | row's LOM level | column effect (exp − shared) |
|---|---|---|---|---|
| buu | detach | 39.96 | 45.95 |  |
| buu | detachexp | 42.43 | 45.95 | +2.47 (ref sd 1.49) |
| buu | rung2 | 38.64 | 46.02 |  |
| buu | rung2exp | 43.91 | 46.02 | +5.27 |
| chair | detach | 39.05 | 48.97 |  |
| chair | detachexp | 46.74 | 48.97 | +7.69 (ref sd 1.44) |
| chair | rung2 | 36.44 | 48.86 |  |
| chair | rung2exp | 47.16 | 48.86 | +10.72 |
| sofa | detach | 37.68 | 47.54 |  |
| sofa | detachexp | 45.74 | 47.54 | +8.06 (ref sd 2.08) |
| sofa | rung2 | 34.17 | 47.82 |  |
| sofa | rung2exp | 46.01 | 47.82 | +11.84 |
| bike | detach | 42.28 | 47.85 |  |
| bike | detachexp | 47.30 | 47.85 | +5.02 (ref sd 0.62) |
| bike | rung2 | 39.77 | 47.57 |  |
| bike | rung2exp | 47.38 | 47.57 | +7.61 |
| shrub | detach | 36.91 | 40.77 |  |
| shrub | detachexp | 40.64 | 40.77 | +3.73 (ref sd 0.33) |
| shrub | rung2 | 36.64 | 40.68 |  |
| shrub | rung2exp | 40.51 | 40.68 | +3.87 |
| pooled | detach | 39.18 | 46.22 |  |
| pooled | detachexp | 44.57 | 46.22 | +5.39 (ref sd 0.45) |
| pooled | rung2 | 37.13 | 46.19 |  |
| pooled | rung2exp | 44.99 | 46.19 | +7.86 |
| pooled_no_shrub | detach | 39.75 | 47.58 |  |
| pooled_no_shrub | detachexp | 45.55 | 47.58 | +5.81 (ref sd 0.64) |
| pooled_no_shrub | rung2 | 37.26 | 47.57 |  |
| pooled_no_shrub | rung2exp | 46.11 | 47.57 | +8.86 |


## R5b'. Per view: paired change of the dark fit, varying − LOM (same view, seed and curve row), against signed u_k, darkened (u<0) and brightened (u>0) views separately (M1 T5b; views with u = 0 left out)

| scene | cell | u_sign | n_view_seeds | mean_diff_db | slope_db_per_stop |
|---|---|---|---|---|---|
| buu | detach | u<0 | 11 | -5.66 | +5.74 |
| buu | detach | u>0 | 11 | -6.30 | +0.26 |
| buu | detachexp | u<0 | 11 | -1.62 | -5.35 |
| buu | detachexp | u>0 | 11 | -5.42 | -3.40 |
| buu | rung2 | u<0 | 11 | -6.74 | +9.56 |
| buu | rung2 | u>0 | 11 | -8.03 | -4.75 |
| buu | rung2exp | u<0 | 11 | -0.22 | -4.02 |
| buu | rung2exp | u>0 | 11 | -4.02 | -4.20 |
| chair | detach | u<0 | 21 | -8.63 | +8.97 |
| chair | detach | u>0 | 21 | -11.35 | -14.34 |
| chair | detachexp | u<0 | 21 | -0.18 | -1.84 |
| chair | detachexp | u>0 | 21 | -4.32 | -4.75 |
| chair | rung2 | u<0 | 21 | -10.99 | +5.91 |
| chair | rung2 | u>0 | 21 | -14.29 | -18.20 |
| chair | rung2exp | u<0 | 21 | +0.26 | -1.28 |
| chair | rung2exp | u>0 | 21 | -3.69 | -5.13 |
| sofa | detach | u<0 | 14 | -7.69 | +4.00 |
| sofa | detach | u>0 | 14 | -12.01 | -19.45 |
| sofa | detachexp | u<0 | 14 | +0.36 | -2.49 |
| sofa | detachexp | u>0 | 14 | -4.01 | -4.92 |
| sofa | rung2 | u<0 | 14 | -10.74 | +10.88 |
| sofa | rung2 | u>0 | 14 | -16.96 | -17.37 |
| sofa | rung2exp | u<0 | 14 | +0.36 | -2.98 |
| sofa | rung2exp | u>0 | 14 | -4.07 | -5.25 |
| bike | detach | u<0 | 18 | -3.88 | +3.21 |
| bike | detach | u>0 | 18 | -7.24 | -10.91 |
| bike | detachexp | u<0 | 18 | +1.86 | -4.40 |
| bike | detachexp | u>0 | 18 | -2.94 | -4.92 |
| bike | rung2 | u<0 | 18 | -6.53 | +5.98 |
| bike | rung2 | u>0 | 18 | -9.07 | -15.46 |
| bike | rung2exp | u<0 | 18 | +2.19 | -3.60 |
| bike | rung2exp | u>0 | 18 | -2.56 | -4.98 |
| shrub | detach | u<0 | 15 | -2.01 | +4.64 |
| shrub | detach | u>0 | 15 | -5.72 | -12.69 |
| shrub | detachexp | u<0 | 15 | +2.32 | -4.58 |
| shrub | detachexp | u>0 | 15 | -2.59 | -5.03 |
| shrub | rung2 | u<0 | 15 | -1.94 | +4.87 |
| shrub | rung2 | u>0 | 15 | -6.13 | -13.75 |
| shrub | rung2exp | u<0 | 15 | +2.36 | -4.91 |
| shrub | rung2exp | u>0 | 15 | -2.69 | -5.29 |
| pooled | detach | u<0 | 79 | -5.71 | +5.66 |
| pooled | detach | u>0 | 79 | -8.76 | -12.22 |
| pooled | detachexp | u<0 | 79 | +0.66 | -3.53 |
| pooled | detachexp | u>0 | 79 | -3.78 | -4.71 |
| pooled | rung2 | u<0 | 79 | -7.62 | +7.33 |
| pooled | rung2 | u>0 | 79 | -11.15 | -14.97 |
| pooled | rung2exp | u<0 | 79 | +1.05 | -3.15 |
| pooled | rung2exp | u>0 | 79 | -3.36 | -5.05 |


## R5c. Exposure recovery (plan 6.3c): Rung 2's final s_k against the applied gains u_k (ideal slope −1, RMSE 0); rmse = RMS of s + u (unpaired, floor near the natural spread); rmse_paired = with the same view's s on LOM subtracted (LOM partner: results/rung2); move = rmse_c(s − s_init). Reference rows from M1 (results/vexp/summary/T4_exposure.csv). A sanity check, not a test: the init already sits at the quantisation floor

| scene | method | slope | r | rmse | rmse_paired | move_from_init |
|---|---|---|---|---|---|---|
| buu | rung2 (s_k) | -1.04 | -0.957 | 0.183 | 0.010 | 0.013 |
| buu | init (M1) | -1.04 | -0.952 | 0.193 | 0.006 |  |
| buu | oracle (M1) | -1.08 | -0.968 | 0.170 | 0.039 |  |
| buu | official (M1) post-hoc s_fit | 0.07 ± 0.00 | 0.801 ± 0.006 | 0.617 ± 0.001 | 0.613 ± 0.001 |  |
| buu | detach (M1) post-hoc s_fit | 0.10 ± 0.00 | 0.834 ± 0.004 | 0.635 ± 0.001 | 0.631 ± 0.001 |  |
| chair | rung2 (s_k) | -1.03 | -0.993 | 0.072 | 0.012 | 0.027 |
| chair | init (M1) | -1.04 | -0.991 | 0.083 | 0.013 |  |
| chair | oracle (M1) | -1.09 | -0.995 | 0.083 | 0.043 |  |
| chair | official (M1) post-hoc s_fit | 0.31 ± 0.06 | 0.963 ± 0.024 | 0.756 ± 0.036 | 0.749 ± 0.035 |  |
| chair | detach (M1) post-hoc s_fit | 0.27 ± 0.02 | 0.951 ± 0.004 | 0.735 ± 0.009 | 0.730 ± 0.009 |  |
| sofa | rung2 (s_k) | -1.02 | -0.990 | 0.084 | 0.013 | 0.013 |
| sofa | init (M1) | -1.02 | -0.991 | 0.079 | 0.004 |  |
| sofa | oracle (M1) | -1.07 | -0.996 | 0.066 | 0.038 |  |
| sofa | official (M1) post-hoc s_fit | 0.17 ± 0.01 | 0.947 ± 0.003 | 0.677 ± 0.004 | 0.672 ± 0.004 |  |
| sofa | detach (M1) post-hoc s_fit | 0.20 ± 0.00 | 0.905 ± 0.002 | 0.697 ± 0.001 | 0.693 ± 0.001 |  |
| bike | rung2 (s_k) | -0.91 | -0.989 | 0.094 | 0.037 | 0.025 |
| bike | init (M1) | -0.95 | -0.990 | 0.083 | 0.019 |  |
| bike | oracle (M1) | -1.04 | -0.989 | 0.093 | 0.056 |  |
| bike | official (M1) post-hoc s_fit | 0.23 ± 0.04 | 0.982 ± 0.000 | 0.712 ± 0.023 | 0.716 ± 0.022 |  |
| bike | detach (M1) post-hoc s_fit | 0.26 ± 0.01 | 0.965 ± 0.003 | 0.730 ± 0.004 | 0.736 ± 0.004 |  |
| shrub | rung2 (s_k) | -0.98 | -0.982 | 0.110 | 0.018 | 0.012 |
| shrub | init (M1) | -0.98 | -0.983 | 0.105 | 0.014 |  |
| shrub | oracle (M1) | -1.11 | -0.980 | 0.145 | 0.082 |  |
| shrub | official (M1) post-hoc s_fit | 0.26 ± 0.01 | 0.960 ± 0.009 | 0.729 ± 0.008 | 0.735 ± 0.008 |  |
| shrub | detach (M1) post-hoc s_fit | 0.44 ± 0.01 | 0.967 ± 0.002 | 0.832 ± 0.007 | 0.841 ± 0.008 |  |
| pooled | rung2 (s_k) | -1.00 | -0.982 | 0.109 | 0.018 | 0.018 |
| pooled | init (M1) | -1.01 | -0.982 | 0.108 | 0.011 |  |
| pooled | oracle (M1) | -1.08 | -0.986 | 0.111 | 0.052 |  |
| pooled | official (M1) post-hoc s_fit | 0.21 ± 0.02 | 0.931 ± 0.008 | 0.698 ± 0.010 | 0.697 ± 0.010 |  |
| pooled | detach (M1) post-hoc s_fit | 0.25 ± 0.01 | 0.924 ± 0.002 | 0.726 ± 0.004 | 0.726 ± 0.004 |  |


## Figures

- `figures/fig1_H1_lom.png`
- `figures/fig2_2x2_interaction.png`
- `figures/fig3_leak.png`
- `figures/fig4_dark_fit.png`
- `figures/fig5_exposure.png`
- `figures/fig6_curves.png`