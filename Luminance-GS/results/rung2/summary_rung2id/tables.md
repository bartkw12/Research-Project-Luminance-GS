# Rung 2 with F initialised at the identity (M5 E1, H2b): H1 rule on LOM (exploratory) and the 2×2 on the varying-exposure set

Seeds 0, 1, 2. Effects paired by seed; ×noise; * = real (above 2× noise). **M5 E1 (H2b):** the Rung 2 cells are the identity-init runs `rung2id` (LOM, varying set) and `rung2idexp` (varying set, exp column). Their H1 verdict is exploratory; H2b(a), the init effect, is `rung2_ablation.py --mode init`. Generated 2026-11-25 11:05 in 106 s by `rung2_analyze.py` (8a7554a82752535d); plan `M2_rung2_plan.md` section 6, readings M5_plan.md 3.2. Every number is *measured*; readings are the report's job.

## Cells

| set | cell | runs | source | status |
|---|---|---|---|---|
| lom | official | 15/15 | rung0/<scene>_baseline_seed* | used |
| lom | detach | 15/15 | rung0/<scene>_detach_seed* | used |
| lom | rung2id | 15/15 | rung2/<scene>_rung2id_seed* | used |
| vexp | official | 15/15 | vexp/<scene>_baseline_seed* | used |
| vexp | detach | 15/15 | vexp/<scene>_detach_seed* | used |
| vexp | detachexp | 15/15 | vexp/<scene>_detachexp_seed* | used |
| vexp | rung2id | 15/15 | vexp/<scene>_rung2id_seed* | used |
| vexp | rung2idexp | 15/15 | vexp/<scene>_rung2idexp_seed* | used |


## Integrity and input checks

| check | value | tolerance | ok |
|---|---|---|---|
| max |rescore − training eval| psnr over all loaded runs | 2.54e-06 | 1e-03 | True |
| detach vs detachexp (same seed, 15 pairs): max |Δ log2 mean_pseudo| | 6.18e-06 | 1e-03 | True |
| buu seed 0: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| buu seed 0: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| buu seed 1: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| buu seed 1: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| buu seed 2: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| buu seed 2: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| chair seed 0: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| chair seed 0: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| chair seed 1: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| chair seed 1: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| chair seed 2: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| chair seed 2: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| sofa seed 0: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| sofa seed 0: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| sofa seed 1: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| sofa seed 1: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| sofa seed 2: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| sofa seed 2: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| bike seed 0: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| bike seed 0: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| bike seed 1: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| bike seed 1: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| bike seed 2: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| bike seed 2: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| shrub seed 0: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| shrub seed 0: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| shrub seed 1: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| shrub seed 1: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| shrub seed 2: rung2id vs rung2idexp, max |Δ s| | 0.00e+00 | 1e-05 | True |
| shrub seed 2: rung2id vs rung2idexp, max |Δ F heights| | 0.00e+00 | 1e-05 | True |
| rung2id vs rung2idexp (same seed, 15 pairs): max |Δ log2 mean_pseudo| | 0.00e+00 | 1e-05 | True |
| lom bike seed 0: rung2id |mean s| | 1.66e-09 | 1e-05 | True |
| lom bike seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 2.92e-03 | 1e-02 | True |
| lom bike seed 0: trainer s vs the preview's (report only; the view order differs) | 1.98e-02 | 5e-02 | report |
| lom bike seed 1: rung2id |mean s| | 1.66e-09 | 1e-05 | True |
| lom bike seed 1: fit_exposure_form(curve_adj) vs the run's s (stops) | 3.08e-03 | 1e-02 | True |
| lom bike seed 1: trainer s vs the preview's (report only; the view order differs) | 2.30e-02 | 5e-02 | report |
| lom bike seed 2: rung2id |mean s| | 1.66e-09 | 1e-05 | True |
| lom bike seed 2: fit_exposure_form(curve_adj) vs the run's s (stops) | 2.94e-03 | 1e-02 | True |
| lom bike seed 2: trainer s vs the preview's (report only; the view order differs) | 2.29e-02 | 5e-02 | report |
| lom buu seed 0: rung2id |mean s| | 5.42e-09 | 1e-05 | True |
| lom buu seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.01e-03 | 1e-02 | True |
| lom buu seed 0: trainer s vs the preview's (report only; the view order differs) | 1.85e-02 | 5e-02 | report |
| lom buu seed 1: rung2id |mean s| | 0.00e+00 | 1e-05 | True |
| lom buu seed 1: fit_exposure_form(curve_adj) vs the run's s (stops) | 9.03e-04 | 1e-02 | True |
| lom buu seed 1: trainer s vs the preview's (report only; the view order differs) | 1.79e-02 | 5e-02 | report |
| lom buu seed 2: rung2id |mean s| | 0.00e+00 | 1e-05 | True |
| lom buu seed 2: fit_exposure_form(curve_adj) vs the run's s (stops) | 8.61e-04 | 1e-02 | True |
| lom buu seed 2: trainer s vs the preview's (report only; the view order differs) | 1.85e-02 | 5e-02 | report |
| lom chair seed 0: rung2id |mean s| | 2.43e-09 | 1e-05 | True |
| lom chair seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.58e-03 | 1e-02 | True |
| lom chair seed 0: trainer s vs the preview's (report only; the view order differs) | 4.75e-02 | 5e-02 | report |
| lom chair seed 1: rung2id |mean s| | 3.47e-10 | 1e-05 | True |
| lom chair seed 1: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.40e-03 | 1e-02 | True |
| lom chair seed 1: trainer s vs the preview's (report only; the view order differs) | 4.80e-02 | 5e-02 | report |
| lom chair seed 2: rung2id |mean s| | 0.00e+00 | 1e-05 | True |
| lom chair seed 2: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.49e-03 | 1e-02 | True |
| lom chair seed 2: trainer s vs the preview's (report only; the view order differs) | 4.75e-02 | 5e-02 | report |
| lom shrub seed 0: rung2id |mean s| | 9.93e-10 | 1e-05 | True |
| lom shrub seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 7.41e-03 | 1e-02 | True |
| lom shrub seed 0: trainer s vs the preview's (report only; the view order differs) | 2.09e-02 | 5e-02 | report |
| lom shrub seed 1: rung2id |mean s| | 4.97e-10 | 1e-05 | True |
| lom shrub seed 1: fit_exposure_form(curve_adj) vs the run's s (stops) | 7.68e-03 | 1e-02 | True |
| lom shrub seed 1: trainer s vs the preview's (report only; the view order differs) | 2.05e-02 | 5e-02 | report |
| lom shrub seed 2: rung2id |mean s| | 9.93e-10 | 1e-05 | True |
| lom shrub seed 2: fit_exposure_form(curve_adj) vs the run's s (stops) | 7.70e-03 | 1e-02 | True |
| lom shrub seed 2: trainer s vs the preview's (report only; the view order differs) | 2.06e-02 | 5e-02 | report |
| lom sofa seed 0: rung2id |mean s| | 1.28e-09 | 1e-05 | True |
| lom sofa seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 8.47e-04 | 1e-02 | True |
| lom sofa seed 0: trainer s vs the preview's (report only; the view order differs) | 1.30e-02 | 5e-02 | report |
| lom sofa seed 1: rung2id |mean s| | 0.00e+00 | 1e-05 | True |
| lom sofa seed 1: fit_exposure_form(curve_adj) vs the run's s (stops) | 9.51e-04 | 1e-02 | True |
| lom sofa seed 1: trainer s vs the preview's (report only; the view order differs) | 1.08e-02 | 5e-02 | report |
| lom sofa seed 2: rung2id |mean s| | 1.03e-09 | 1e-05 | True |
| lom sofa seed 2: fit_exposure_form(curve_adj) vs the run's s (stops) | 9.12e-04 | 1e-02 | True |
| lom sofa seed 2: trainer s vs the preview's (report only; the view order differs) | 1.18e-02 | 5e-02 | report |
| vexp bike seed 0: rung2id s_init vs oracle.json s_init (stops) | 2.62e-08 | 1e-06 | True |
| vexp bike seed 0: rung2id |mean s| | 0.00e+00 | 1e-05 | True |
| vexp bike seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 7.07e-03 | 1e-02 | True |
| vexp bike seed 0: trainer s vs the preview's (report only; the view order differs) | 5.34e-02 | 5e-02 | report (above) |
| vexp bike seed 1: rung2id s_init vs oracle.json s_init (stops) | 2.62e-08 | 1e-06 | True |
| vexp bike seed 1: rung2id |mean s| | 0.00e+00 | 1e-05 | True |
| vexp bike seed 1: fit_exposure_form(curve_adj) vs the run's s (stops) | 6.98e-03 | 1e-02 | True |
| vexp bike seed 1: trainer s vs the preview's (report only; the view order differs) | 5.30e-02 | 5e-02 | report (above) |
| vexp bike seed 2: rung2id s_init vs oracle.json s_init (stops) | 2.62e-08 | 1e-06 | True |
| vexp bike seed 2: rung2id |mean s| | 0.00e+00 | 1e-05 | True |
| vexp bike seed 2: fit_exposure_form(curve_adj) vs the run's s (stops) | 6.98e-03 | 1e-02 | True |
| vexp bike seed 2: trainer s vs the preview's (report only; the view order differs) | 5.34e-02 | 5e-02 | report (above) |
| vexp buu seed 0: rung2id s_init vs oracle.json s_init (stops) | 2.72e-08 | 1e-06 | True |
| vexp buu seed 0: rung2id |mean s| | 1.08e-08 | 1e-05 | True |
| vexp buu seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 7.16e-04 | 1e-02 | True |
| vexp buu seed 0: trainer s vs the preview's (report only; the view order differs) | 2.63e-02 | 5e-02 | report |
| vexp buu seed 1: rung2id s_init vs oracle.json s_init (stops) | 2.72e-08 | 1e-06 | True |
| vexp buu seed 1: rung2id |mean s| | 1.08e-08 | 1e-05 | True |
| vexp buu seed 1: fit_exposure_form(curve_adj) vs the run's s (stops) | 6.73e-04 | 1e-02 | True |
| vexp buu seed 1: trainer s vs the preview's (report only; the view order differs) | 2.57e-02 | 5e-02 | report |
| vexp buu seed 2: rung2id s_init vs oracle.json s_init (stops) | 2.72e-08 | 1e-06 | True |
| vexp buu seed 2: rung2id |mean s| | 0.00e+00 | 1e-05 | True |
| vexp buu seed 2: fit_exposure_form(curve_adj) vs the run's s (stops) | 7.26e-04 | 1e-02 | True |
| vexp buu seed 2: trainer s vs the preview's (report only; the view order differs) | 2.64e-02 | 5e-02 | report |
| vexp chair seed 0: rung2id s_init vs oracle.json s_init (stops) | 3.60e-08 | 1e-06 | True |
| vexp chair seed 0: rung2id |mean s| | 2.77e-09 | 1e-05 | True |
| vexp chair seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.60e-03 | 1e-02 | True |
| vexp chair seed 0: trainer s vs the preview's (report only; the view order differs) | 5.67e-02 | 5e-02 | report (above) |
| vexp chair seed 1: rung2id s_init vs oracle.json s_init (stops) | 3.60e-08 | 1e-06 | True |
| vexp chair seed 1: rung2id |mean s| | 8.32e-09 | 1e-05 | True |
| vexp chair seed 1: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.82e-03 | 1e-02 | True |
| vexp chair seed 1: trainer s vs the preview's (report only; the view order differs) | 5.69e-02 | 5e-02 | report (above) |
| vexp chair seed 2: rung2id s_init vs oracle.json s_init (stops) | 3.60e-08 | 1e-06 | True |
| vexp chair seed 2: rung2id |mean s| | 1.39e-08 | 1e-05 | True |
| vexp chair seed 2: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.82e-03 | 1e-02 | True |
| vexp chair seed 2: trainer s vs the preview's (report only; the view order differs) | 5.45e-02 | 5e-02 | report (above) |
| vexp shrub seed 0: rung2id s_init vs oracle.json s_init (stops) | 4.18e-08 | 1e-06 | True |
| vexp shrub seed 0: rung2id |mean s| | 4.97e-09 | 1e-05 | True |
| vexp shrub seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 7.62e-03 | 1e-02 | True |
| vexp shrub seed 0: trainer s vs the preview's (report only; the view order differs) | 3.14e-02 | 5e-02 | report |
| vexp shrub seed 1: rung2id s_init vs oracle.json s_init (stops) | 4.18e-08 | 1e-06 | True |
| vexp shrub seed 1: rung2id |mean s| | 9.93e-10 | 1e-05 | True |
| vexp shrub seed 1: fit_exposure_form(curve_adj) vs the run's s (stops) | 7.28e-03 | 1e-02 | True |
| vexp shrub seed 1: trainer s vs the preview's (report only; the view order differs) | 3.11e-02 | 5e-02 | report |
| vexp shrub seed 2: rung2id s_init vs oracle.json s_init (stops) | 4.18e-08 | 1e-06 | True |
| vexp shrub seed 2: rung2id |mean s| | 8.94e-09 | 1e-05 | True |
| vexp shrub seed 2: fit_exposure_form(curve_adj) vs the run's s (stops) | 7.68e-03 | 1e-02 | True |
| vexp shrub seed 2: trainer s vs the preview's (report only; the view order differs) | 3.24e-02 | 5e-02 | report |
| vexp sofa seed 0: rung2id s_init vs oracle.json s_init (stops) | 2.97e-08 | 1e-06 | True |
| vexp sofa seed 0: rung2id |mean s| | 8.22e-09 | 1e-05 | True |
| vexp sofa seed 0: fit_exposure_form(curve_adj) vs the run's s (stops) | 9.65e-04 | 1e-02 | True |
| vexp sofa seed 0: trainer s vs the preview's (report only; the view order differs) | 2.03e-02 | 5e-02 | report |
| vexp sofa seed 1: rung2id s_init vs oracle.json s_init (stops) | 2.97e-08 | 1e-06 | True |
| vexp sofa seed 1: rung2id |mean s| | 0.00e+00 | 1e-05 | True |
| vexp sofa seed 1: fit_exposure_form(curve_adj) vs the run's s (stops) | 1.06e-03 | 1e-02 | True |
| vexp sofa seed 1: trainer s vs the preview's (report only; the view order differs) | 2.07e-02 | 5e-02 | report |
| vexp sofa seed 2: rung2id s_init vs oracle.json s_init (stops) | 2.97e-08 | 1e-06 | True |
| vexp sofa seed 2: rung2id |mean s| | 0.00e+00 | 1e-05 | True |
| vexp sofa seed 2: fit_exposure_form(curve_adj) vs the run's s (stops) | 8.91e-04 | 1e-02 | True |
| vexp sofa seed 2: trainer s vs the preview's (report only; the view order differs) | 2.01e-02 | 5e-02 | report |
| M1 reproduction: T1 validation means (official, detach) | 0.00e+00 | 1e-09 | True |
| M1 reproduction: paired pseudo-target leak (official, detach) vs T4_exposure.csv | 0.00e+00 | 1e-09 | True |
| M1 reproduction: detach dark-fit change vs signed u (T5b) | 0.00e+00 | 1e-09 | True |


## Decision rules (plan 6.5)

- **H1 rule applied to rung2id (LOM; exploratory, cannot change H1):** buu: meets; chair: meets; sofa: meets; bike: meets -> **supported** (all of buu, chair, sofa and bike must meet the rule).
- **H2b(b), H2 rule for rung2id (pooled):** curve-model main effect on bm PSNR +2.72 dB (+6.8×, real); on LPIPS -0.0360 (-3.9×); positive in 4 of 4 non-shrub scenes (buu, chair, sofa, bike); exp-column simple effect +2.75 dB (+12.0×) -> **supported**.
- **H2b(b), H2 rule for rung2id (pooled_no_shrub):** curve-model main effect on bm PSNR +2.99 dB (+6.0×, real); on LPIPS -0.0594 (-5.1×); positive in 4 of 4 non-shrub scenes (buu, chair, sofa, bike); exp-column simple effect +3.04 dB (+10.8×) -> **supported**.
- **Mechanism:** Rung 2's pooled paired leak minus detach's -0.503 (-725.7×) -> **supports H2** (needs below detach's by more than 2× noise).



## R1a. H1 on LOM: validation per cell (mean ± sd over seeds; pooled = mean over scenes per seed). bm PSNR = brightness-matched PSNR

| scene | variant | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | official | 26.00 ± 0.33 | 20.81 ± 0.21 | 0.9089 ± 0.0016 | 0.1862 ± 0.0047 |
| buu | detach | 26.37 ± 0.25 | 21.56 ± 0.52 | 0.9039 ± 0.0045 | 0.2026 ± 0.0093 |
| buu | rung2id | 26.89 ± 0.20 | 20.51 ± 0.37 | 0.9097 ± 0.0027 | 0.1575 ± 0.0036 |
| chair | official | 20.94 ± 0.16 | 17.76 ± 0.36 | 0.8394 ± 0.0032 | 0.3379 ± 0.0069 |
| chair | detach | 23.09 ± 0.12 | 18.82 ± 0.03 | 0.8525 ± 0.0007 | 0.3105 ± 0.0020 |
| chair | rung2id | 23.51 ± 0.22 | 20.01 ± 0.27 | 0.8691 ± 0.0012 | 0.2842 ± 0.0011 |
| sofa | official | 25.22 ± 0.46 | 23.22 ± 0.58 | 0.8994 ± 0.0077 | 0.2757 ± 0.0098 |
| sofa | detach | 27.41 ± 0.40 | 23.47 ± 0.29 | 0.9042 ± 0.0044 | 0.2599 ± 0.0057 |
| sofa | rung2id | 28.28 ± 0.40 | 23.77 ± 0.07 | 0.9130 ± 0.0033 | 0.2325 ± 0.0058 |
| bike | official | 19.45 ± 0.13 | 18.42 ± 0.45 | 0.7520 ± 0.0043 | 0.4089 ± 0.0206 |
| bike | detach | 21.31 ± 0.05 | 16.02 ± 0.10 | 0.7637 ± 0.0040 | 0.2910 ± 0.0069 |
| bike | rung2id | 21.41 ± 0.00 | 16.21 ± 0.03 | 0.7882 ± 0.0016 | 0.3021 ± 0.0059 |
| shrub | official | 18.37 ± 0.99 | 12.13 ± 1.13 | 0.6161 ± 0.0203 | 0.2632 ± 0.0047 |
| shrub | detach | 18.06 ± 0.50 | 11.76 ± 0.66 | 0.5967 ± 0.0043 | 0.2877 ± 0.0525 |
| shrub | rung2id | 18.24 ± 0.01 | 11.95 ± 0.03 | 0.5875 ± 0.0011 | 0.3409 ± 0.0013 |
| pooled | official | 22.00 ± 0.32 | 18.47 ± 0.40 | 0.8032 ± 0.0049 | 0.2944 ± 0.0039 |
| pooled | detach | 23.25 ± 0.07 | 18.33 ± 0.02 | 0.8042 ± 0.0016 | 0.2704 ± 0.0116 |
| pooled | rung2id | 23.67 ± 0.07 | 18.49 ± 0.12 | 0.8135 ± 0.0003 | 0.2634 ± 0.0017 |
| pooled_no_shrub | official | 22.90 ± 0.16 | 20.05 ± 0.21 | 0.8499 ± 0.0019 | 0.3022 ± 0.0057 |
| pooled_no_shrub | detach | 24.55 ± 0.08 | 19.97 ± 0.18 | 0.8561 ± 0.0018 | 0.2660 ± 0.0014 |
| pooled_no_shrub | rung2id | 25.02 ± 0.09 | 20.12 ± 0.16 | 0.8700 ± 0.0001 | 0.2441 ± 0.0018 |


## R1b. rung2id − detach on LOM, paired by seed (seeds 0-2: ×noise, * = real; one seed: the difference next to detach's seed sd over seeds 0-2). H1 rule: bm PSNR not below −2× noise, SSIM and LPIPS not worse beyond 2× noise, in each of buu, chair, sofa and bike; shrub reported only; exploratory for the identity init (H1 is judged on the cdf init)

| scene | bm PSNR | raw PSNR | SSIM | LPIPS | H1 rule (plan 6.5) |
|---|---|---|---|---|---|
| buu | +0.52 (+2.3×) * | -1.05 (-2.3×) * | +0.0058 (+1.6×) | -0.0451 (-6.4×) * | meets |
| chair | +0.41 (+2.4×) * | +1.19 (+6.2×) * | +0.0165 (+16.3×) * | -0.0264 (-16.6×) * | meets |
| sofa | +0.87 (+2.2×) * | +0.29 (+1.4×) | +0.0087 (+2.2×) * | -0.0274 (-4.8×) * | meets |
| bike | +0.10 (+3.1×) * | +0.19 (+2.5×) * | +0.0244 (+8.0×) * | +0.0111 (+1.7×) | meets |
| shrub | +0.17 (+0.5×) | +0.19 (+0.4×) | -0.0091 (-2.9×) * | +0.0532 (+1.4×) | reported only |
| pooled | +0.41 (+6.0×) * | +0.16 (+1.9×) | +0.0093 (+7.9×) * | -0.0069 (-0.8×) | — |
| pooled_no_shrub | +0.47 (+5.7×) * | +0.16 (+0.9×) | +0.0139 (+10.8×) * | -0.0220 (-13.5×) * | — |


## R2. H1 extras on LOM: pseudo-target bm PSNR against GT, cross-view consistency of pseudo-targets and renders (relative L1 after the shared-depth warp, as a multiple of the GT's; the reference-depth run official seed 0 is left out), colour on validation, dark fit (dark render vs input, mean over training views), wall time (runs.csv), final Gaussians, curve parameters

| scene | variant | pseudo bm PSNR | pseudo consistency (× GT) | render consistency (× GT) | angle err (°) | sat_ratio | hue err (°) | dark fit (dB) | minutes/run | Gaussians | curve params |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buu | official | 26.22 ± 0.10 | 1.29 ± 0.02 | 0.70 ± 0.01 | 4.08 ± 0.09 | 0.756 ± 0.008 | 11.25 ± 0.35 | 46.22 ± 0.07 | 12.1 ± 0.1 | 83448 ± 1578 | 255343 ± 0 |
| buu | detach | 26.79 ± 0.02 | 1.56 ± 0.02 | 0.79 ± 0.06 | 4.03 ± 0.06 | 0.934 ± 0.008 | 8.23 ± 0.13 | 46.25 ± 0.26 | 7.4 ± 0.1 | 82022 ± 3145 | 255343 ± 0 |
| buu | rung2id | 27.57 ± 0.00 | 1.51 ± 0.00 | 0.79 ± 0.03 | 3.07 ± 0.07 | 0.866 ± 0.006 | 7.35 ± 0.25 | 46.04 ± 0.24 | 4.0 ± 0.1 | 86160 ± 2392 | 55 ± 0 |
| buu | rung2id − detach | +0.78 (+69.8×) * | -0.05 (-4.4×) * | +0.00 (+0.0×) | -0.97 (-15.2×) * | -0.068 (-9.7×) * | -0.88 (-4.4×) * | -0.20 (-0.8×) | -3.4 (-47.4×) * | +4138 (+1.5×) |  |
| chair | official | 21.09 ± 0.16 | 0.69 ± 0.01 | 0.30 ± 0.00 | 7.46 ± 0.06 | 0.322 ± 0.007 | 21.98 ± 0.49 | 49.01 ± 0.10 | 12.2 ± 0.9 | 50738 ± 4223 | 255448 ± 0 |
| chair | detach | 23.43 ± 0.00 | 0.97 ± 0.00 | 0.38 ± 0.01 | 5.87 ± 0.03 | 0.498 ± 0.003 | 18.33 ± 0.19 | 49.02 ± 0.05 | 7.4 ± 0.2 | 57621 ± 3404 | 255448 ± 0 |
| chair | rung2id | 23.76 ± 0.00 | 0.97 ± 0.00 | 0.36 ± 0.01 | 5.96 ± 0.01 | 0.465 ± 0.001 | 16.68 ± 0.16 | 48.92 ± 0.11 | 3.7 ± 0.0 | 50779 ± 3112 | 76 ± 0 |
| chair | rung2id − detach | +0.33 (+178.4×) * | +0.01 (+4.4×) * | -0.02 (-1.9×) | +0.09 (+4.7×) * | -0.032 (-14.2×) * | -1.65 (-9.2×) * | -0.10 (-1.2×) | -3.6 (-28.7×) * | -6842 (-2.1×) * |  |
| sofa | official | 26.78 ± 0.27 | 1.16 ± 0.01 | 0.58 ± 0.01 | 3.36 ± 0.08 | 0.704 ± 0.038 | 7.21 ± 0.80 | 47.30 ± 0.23 | 10.9 ± 0.1 | 37442 ± 3102 | 255378 ± 0 |
| sofa | detach | 28.74 ± 0.00 | 1.67 ± 0.02 | 0.76 ± 0.01 | 2.57 ± 0.03 | 0.981 ± 0.011 | 11.21 ± 1.02 | 47.42 ± 0.10 | 7.2 ± 0.0 | 40364 ± 3002 | 255378 ± 0 |
| sofa | rung2id | 29.65 ± 0.00 | 1.70 ± 0.00 | 0.95 ± 0.01 | 2.13 ± 0.04 | 0.856 ± 0.004 | 7.64 ± 0.19 | 47.60 ± 0.05 | 3.7 ± 0.1 | 37944 ± 1214 | 62 ± 0 |
| sofa | rung2id − detach | +0.91 (+2596.0×) * | +0.03 (+2.4×) * | +0.19 (+19.5×) * | -0.44 (-12.7×) * | -0.126 (-14.9×) * | -3.57 (-4.9×) * | +0.18 (+2.3×) * | -3.5 (-63.2×) * [2 seeds] | -2419 (-1.1×) |  |
| bike | official | 19.48 ± 0.08 | 1.11 ± 0.14 | 0.47 ± 0.31 | 2.89 ± 0.41 | 0.766 ± 0.167 | 21.46 ± 1.27 | 47.15 ± 0.52 | 14.1 ± 0.3 | 204630 ± 44411 | 255413 ± 0 |
| bike | detach | 22.50 ± 0.01 | 1.72 ± 0.02 | 0.58 ± 0.01 | 2.48 ± 0.02 | 0.497 ± 0.010 | 31.78 ± 1.88 | 47.72 ± 0.12 | 8.1 ± 0.0 | 237520 ± 3979 | 255413 ± 0 |
| bike | rung2id | 22.83 ± 0.00 | 1.42 ± 0.00 | 0.47 ± 0.01 | 2.58 ± 0.02 | 0.522 ± 0.003 | 33.58 ± 0.47 | 47.66 ± 0.10 | 4.6 ± 0.0 | 207119 ± 18968 | 69 ± 0 |
| bike | rung2id − detach | +0.33 (+50.4×) * | -0.30 (-23.9×) * | -0.11 (-12.2×) * | +0.10 (+5.9×) * | +0.024 (+3.3×) * | +1.80 (+1.3×) | -0.06 (-0.6×) | -3.5 (-129.9×) * | -30401 (-2.2×) * |  |
| shrub | official | 17.81 ± 0.82 | 1.27 ± 0.45 | 0.52 ± 0.14 | 11.33 ± 2.94 | 0.537 ± 0.213 | 9.55 ± 2.41 | 40.89 ± 0.06 | 13.8 ± 0.3 | 866517 ± 41175 | 255383 ± 0 |
| shrub | detach | 17.52 ± 0.17 | 1.19 ± 0.41 | 0.52 ± 0.09 | 10.77 ± 1.30 | 0.634 ± 0.244 | 9.16 ± 0.72 | 40.81 ± 0.06 | 9.1 ± 0.4 | 875042 ± 17980 | 255383 ± 0 |
| shrub | rung2id | 17.73 ± 0.00 | 1.35 ± 0.00 | 0.56 ± 0.00 | 9.00 ± 0.00 | 0.726 ± 0.001 | 10.22 ± 0.03 | 40.65 ± 0.03 | 6.5 ± 0.1 | 887640 ± 6458 | 63 ± 0 |
| shrub | rung2id − detach | +0.21 (+1.8×) | +0.15 (+0.5×) | +0.05 (+0.7×) | -1.78 (-1.9×) | +0.092 (+0.5×) | +1.06 (+2.1×) * | -0.16 (-3.4×) * | -2.6 (-8.4×) * | +12598 (+0.9×) |  |
| pooled | official | 22.28 ± 0.26 | 1.10 ± 0.11 | 0.51 ± 0.09 | 5.82 ± 0.53 | 0.617 ± 0.083 | 14.29 ± 0.59 | 46.12 ± 0.15 | 12.6 ± 0.2 | 248555 ± 17402 | 255393 ± 0 |
| pooled | detach | 23.80 ± 0.03 | 1.42 ± 0.08 | 0.61 ± 0.03 | 5.15 ± 0.26 | 0.709 ± 0.050 | 15.74 ± 0.24 | 46.24 ± 0.03 | 7.8 ± 0.1 | 258514 ± 4160 | 255393 ± 0 |
| pooled | rung2id | 24.31 ± 0.00 | 1.39 ± 0.00 | 0.63 ± 0.01 | 4.55 ± 0.01 | 0.687 ± 0.001 | 15.09 ± 0.15 | 46.17 ± 0.04 | 4.5 ± 0.0 | 253929 ± 4992 | 65 ± 0 |
| pooled | rung2id − detach | +0.51 (+22.7×) * | -0.03 (-0.6×) | +0.02 (+1.0×) | -0.60 (-3.3×) * | -0.022 (-0.6×) | -0.65 (-3.3×) * | -0.07 (-2.1×) * | -3.3 (-61.0×) * [2 seeds] | -4585 (-1.0×) |  |
| pooled_no_shrub | official | 23.39 ± 0.13 | 1.06 ± 0.03 | 0.51 ± 0.07 | 4.45 ± 0.07 | 0.637 ± 0.052 | 15.47 ± 0.49 | 47.42 ± 0.20 | 12.3 ± 0.3 | 94064 ± 12066 | 255396 ± 0 |
| pooled_no_shrub | detach | 25.36 ± 0.00 | 1.48 ± 0.01 | 0.63 ± 0.02 | 3.74 ± 0.01 | 0.728 ± 0.006 | 17.39 ± 0.45 | 47.60 ± 0.03 | 7.5 ± 0.0 | 104382 ± 940 | 255396 ± 0 |
| pooled_no_shrub | rung2id | 25.95 ± 0.00 | 1.40 ± 0.00 | 0.64 ± 0.01 | 3.44 ± 0.01 | 0.677 ± 0.002 | 16.31 ± 0.18 | 47.56 ± 0.06 | 4.0 ± 0.0 | 95501 ± 5606 | 66 ± 0 |
| pooled_no_shrub | rung2id − detach | +0.59 (+274.6×) * | -0.08 (-19.3×) * | +0.02 (+1.2×) | -0.30 (-25.4×) * | -0.051 (-11.5×) * | -1.07 (-3.2×) * | -0.05 (-1.0×) | -3.5 (-110.6×) * [2 seeds] | -8881 (-2.2×) * |  |


## R6. Side effects on the varying set: colour on validation, consistency (× GT, shared LOM depth), wall time, final Gaussians. The exp rows are the column effect within each curve row (the dark branch re-weights each view's L_low gradient, plan 2.1)

| scene | variant | sat_ratio | angle err (°) | hue err (°) | pseudo consistency (× GT) | render consistency (× GT) | minutes/run | Gaussians |
|---|---|---|---|---|---|---|---|---|
| buu | official | 0.816 ± 0.053 | 4.09 ± 0.28 | 10.36 ± 0.42 | 4.08 ± 0.07 | 2.69 ± 0.12 | 12.5 ± 0.0 | 140722 ± 11253 |
| buu | detach | 0.940 ± 0.025 | 3.64 ± 0.23 | 9.68 ± 1.34 | 4.73 ± 0.05 | 3.35 ± 0.43 | 7.7 ± 0.0 | 153614 ± 14505 |
| buu | detachexp | 0.847 ± 0.055 | 3.80 ± 0.51 | 9.68 ± 0.92 | 4.73 ± 0.05 | 2.26 ± 0.48 | 7.9 ± 0.2 | 105295 ± 1904 |
| buu | rung2id | 0.888 ± 0.009 | 3.25 ± 0.07 | 9.33 ± 0.33 | 1.65 ± 0.00 | 0.99 ± 0.09 | 3.9 ± 0.1 | 124029 ± 5841 |
| buu | rung2idexp | 0.896 ± 0.003 | 3.01 ± 0.07 | 7.13 ± 0.36 | 1.65 ± 0.00 | 0.93 ± 0.05 | 3.8 ± 0.1 | 94605 ± 2616 |
| buu | detachexp − detach | -0.093 (-2.2×) * | +0.16 (+0.4×) | +0.00 (+0.0×) | -0.00 (-0.0×) | -1.09 (-2.4×) * | +0.2 (+1.3×) | -48320 (-4.7×) * |
| buu | rung2idexp − rung2id | +0.008 (+1.1×) | -0.24 (-3.5×) * | -2.20 (-6.3×) * | +0.00 (+0.0×) | -0.06 (-0.8×) | -0.1 (-1.2×) | -29425 (-6.5×) * |
| chair | official | 0.221 ± 0.010 | 8.35 ± 0.08 | 22.43 ± 2.18 | 3.58 ± 0.31 | 0.75 ± 0.32 | 12.3 ± 0.0 | 86845 ± 12706 |
| chair | detach | 0.333 ± 0.031 | 7.12 ± 0.24 | 14.89 ± 0.61 | 5.41 ± 0.04 | 1.13 ± 0.44 | 7.7 ± 0.0 | 111862 ± 15646 |
| chair | detachexp | 0.335 ± 0.007 | 7.10 ± 0.04 | 14.58 ± 0.70 | 5.41 ± 0.04 | 0.75 ± 0.18 | 7.8 ± 0.1 | 82035 ± 6803 |
| chair | rung2id | 0.492 ± 0.002 | 5.67 ± 0.04 | 12.55 ± 0.83 | 1.26 ± 0.00 | 0.42 ± 0.01 | 38.6 ± 60.3 | 73650 ± 6661 |
| chair | rung2idexp | 0.496 ± 0.004 | 5.60 ± 0.04 | 12.19 ± 0.33 | 1.26 ± 0.00 | 0.39 ± 0.01 | 3.6 ± 0.1 | 54581 ± 2194 |
| chair | detachexp − detach | +0.002 (+0.1×) | -0.02 (-0.1×) | -0.30 (-0.5×) | -0.00 (-0.0×) | -0.38 (-1.1×) | +0.1 (+2.4×) * | -29827 (-2.5×) * |
| chair | rung2idexp − rung2id | +0.004 (+1.5×) | -0.06 (-1.7×) | -0.36 (-0.6×) | +0.00 (+0.0×) | -0.03 (-2.4×) * | -34.9 (-0.8×) | -19069 (-3.8×) * |
| sofa | official | 0.517 ± 0.017 | 3.68 ± 0.15 | 8.89 ± 2.39 | 6.25 ± 0.07 | 1.89 ± 0.79 | 11.4 ± 0.1 | 61335 ± 4056 |
| sofa | detach | 0.843 ± 0.005 | 2.52 ± 0.21 | 8.63 ± 0.69 | 8.98 ± 0.08 | 2.16 ± 1.58 | 7.4 ± 0.0 | 53775 ± 4561 |
| sofa | detachexp | 0.904 ± 0.066 | 2.82 ± 0.51 | 10.58 ± 2.64 | 8.98 ± 0.08 | 0.96 ± 0.21 | 7.6 ± 0.2 | 41237 ± 1279 |
| sofa | rung2id | 0.948 ± 0.012 | 2.17 ± 0.09 | 8.87 ± 0.32 | 2.00 ± 0.00 | 0.99 ± 0.03 | 3.6 ± 0.1 | 57322 ± 4598 |
| sofa | rung2idexp | 0.947 ± 0.005 | 2.15 ± 0.06 | 8.49 ± 0.07 | 2.00 ± 0.00 | 1.07 ± 0.09 | 3.6 ± 0.1 | 44829 ± 8021 |
| sofa | detachexp − detach | +0.061 (+1.3×) | +0.30 (+0.8×) | +1.96 (+1.0×) | +0.00 (+0.0×) | -1.20 (-1.1×) | +0.2 (+1.1×) | -12538 (-3.7×) * |
| sofa | rung2idexp − rung2id | -0.001 (-0.1×) | -0.03 (-0.4×) | -0.37 (-1.6×) | +0.00 (+0.0×) | +0.08 (+1.2×) | +0.0 (+0.3×) | -12493 (-1.9×) |
| bike | official | 0.543 ± 0.034 | 2.72 ± 0.12 | 16.91 ± 0.80 | 2.32 ± 0.05 | 0.52 ± 0.29 | 15.0 ± 0.0 | 203818 ± 3209 |
| bike | detach | 0.442 ± 0.005 | 2.54 ± 0.03 | 12.34 ± 1.02 | 4.61 ± 0.08 | 1.37 ± 0.35 | 8.3 ± 0.0 | 277262 ± 2774 |
| bike | detachexp | 0.417 ± 0.006 | 2.50 ± 0.04 | 8.99 ± 1.56 | 4.61 ± 0.08 | 0.66 ± 0.12 | 8.5 ± 0.3 | 231587 ± 8104 |
| bike | rung2id | 0.558 ± 0.006 | 2.59 ± 0.01 | 27.01 ± 5.56 | 1.64 ± 0.00 | 0.60 ± 0.09 | 4.5 ± 0.0 | 238666 ± 2628 |
| bike | rung2idexp | 0.545 ± 0.006 | 2.63 ± 0.01 | 31.98 ± 1.15 | 1.64 ± 0.00 | 0.50 ± 0.01 | 4.4 ± 0.0 | 207566 ± 3485 |
| bike | detachexp − detach | -0.024 (-4.4×) * | -0.04 (-1.0×) | -3.35 (-2.5×) * | +0.00 (+0.0×) | -0.72 (-2.7×) * | +0.3 (+1.3×) | -45675 (-7.5×) * |
| bike | rung2idexp − rung2id | -0.013 (-2.2×) * | +0.04 (+3.3×) * | +4.97 (+1.2×) | +0.00 (+0.0×) | -0.10 (-1.5×) | -0.1 (-1.4×) | -31100 (-10.1×) * |
| shrub | official | 0.489 ± 0.210 | 12.42 ± 2.49 | 9.89 ± 1.07 | 2.08 ± 0.70 | 0.78 ± 0.20 | 14.6 ± 0.2 | 883061 ± 36191 |
| shrub | detach | 0.396 ± 0.010 | 13.27 ± 0.13 | 8.74 ± 1.49 | 1.60 ± 0.03 | 0.55 ± 0.10 | 9.5 ± 0.1 | 914712 ± 20689 |
| shrub | detachexp | 0.390 ± 0.005 | 13.31 ± 0.09 | 7.08 ± 0.04 | 1.60 ± 0.03 | 0.41 ± 0.00 | 9.6 ± 0.2 | 868769 ± 17138 |
| shrub | rung2id | 0.755 ± 0.002 | 8.85 ± 0.03 | 9.15 ± 0.05 | 1.42 ± 0.00 | 0.58 ± 0.00 | 6.3 ± 0.0 | 951859 ± 8771 |
| shrub | rung2idexp | 0.756 ± 0.000 | 8.83 ± 0.03 | 9.18 ± 0.04 | 1.42 ± 0.00 | 0.57 ± 0.00 | 6.2 ± 0.0 | 917850 ± 7782 |
| shrub | detachexp − detach | -0.005 (-0.7×) | +0.04 (+0.4×) | -1.66 (-1.6×) | +0.00 (+0.0×) | -0.14 (-2.0×) | +0.1 (+0.5×) | -45943 (-2.4×) * |
| shrub | rung2idexp − rung2id | +0.001 (+0.4×) | -0.02 (-0.5×) | +0.04 (+0.8×) | +0.00 (+0.0×) | -0.00 (-1.2×) | -0.1 (-2.9×) * | -34009 (-4.1×) * |
| pooled | official | 0.517 ± 0.038 | 6.25 ± 0.47 | 13.70 ± 0.36 | 3.66 ± 0.19 | 1.33 ± 0.19 | 13.2 ± 0.0 | 275156 ± 8204 |
| pooled | detach | 0.590 ± 0.012 | 5.81 ± 0.11 | 10.85 ± 0.60 | 5.06 ± 0.03 | 1.71 ± 0.39 | 8.1 ± 0.0 | 302245 ± 4974 |
| pooled | detachexp | 0.579 ± 0.008 | 5.91 ± 0.20 | 10.18 ± 0.38 | 5.06 ± 0.03 | 1.01 ± 0.13 | 8.3 ± 0.1 | 265785 ± 6044 |
| pooled | rung2id | 0.728 ± 0.001 | 4.51 ± 0.02 | 13.38 ± 1.30 | 1.59 ± 0.00 | 0.71 ± 0.03 | 11.4 ± 12.1 | 289105 ± 1439 |
| pooled | rung2idexp | 0.728 ± 0.001 | 4.44 ± 0.02 | 13.80 ± 0.24 | 1.59 ± 0.00 | 0.69 ± 0.02 | 4.3 ± 0.0 | 263886 ± 3900 |
| pooled | detachexp − detach | -0.012 (-1.1×) | +0.09 (+0.6×) | -0.67 (-1.3×) | -0.00 (-0.0×) | -0.70 (-2.4×) * | +0.2 (+1.6×) | -36460 (-6.6×) * |
| pooled | rung2idexp − rung2id | -0.000 (-0.1×) | -0.06 (-2.8×) * | +0.41 (+0.4×) | +0.00 (+0.0×) | -0.02 (-0.9×) | -7.0 (-0.8×) | -25219 (-8.6×) * |
| pooled_no_shrub | official | 0.524 ± 0.005 | 4.71 ± 0.04 | 14.65 ± 0.19 | 4.06 ± 0.08 | 1.46 ± 0.22 | 12.8 ± 0.0 | 123180 ± 1349 |
| pooled_no_shrub | detach | 0.639 ± 0.013 | 3.95 ± 0.12 | 11.38 ± 0.38 | 5.93 ± 0.04 | 2.00 ± 0.48 | 7.8 ± 0.0 | 149128 ± 1747 |
| pooled_no_shrub | detachexp | 0.626 ± 0.011 | 4.06 ± 0.23 | 10.96 ± 0.48 | 5.93 ± 0.04 | 1.16 ± 0.16 | 7.9 ± 0.1 | 115038 ± 3403 |
| pooled_no_shrub | rung2id | 0.721 ± 0.001 | 3.42 ± 0.02 | 14.44 ± 1.64 | 1.64 ± 0.00 | 0.75 ± 0.03 | 12.6 ± 15.1 | 123417 ± 671 |
| pooled_no_shrub | rung2idexp | 0.721 ± 0.002 | 3.35 ± 0.02 | 14.95 ± 0.31 | 1.64 ± 0.00 | 0.72 ± 0.03 | 3.9 ± 0.1 | 100395 ± 3726 |
| pooled_no_shrub | detachexp − detach | -0.013 (-1.1×) | +0.10 (+0.6×) | -0.42 (-1.0×) | -0.00 (-0.0×) | -0.85 (-2.3×) * | +0.2 (+2.1×) * | -34090 (-12.6×) * |
| pooled_no_shrub | rung2idexp − rung2id | -0.000 (-0.3×) | -0.07 (-3.6×) * | +0.51 (+0.4×) | +0.00 (+0.0×) | -0.03 (-0.8×) | -8.8 (-0.8×) | -23022 (-8.6×) * |


## R3. The 2×2 on the varying-exposure set (official for reference): validation per cell

| scene | cell | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | official | 22.33 ± 0.73 | 22.01 ± 0.84 | 0.8768 ± 0.0065 | 0.2607 ± 0.0141 |
| buu | detach | 24.26 ± 1.51 | 22.81 ± 0.75 | 0.8807 ± 0.0135 | 0.2515 ± 0.0142 |
| buu | detachexp | 24.02 ± 1.64 | 21.68 ± 0.33 | 0.9030 ± 0.0056 | 0.2164 ± 0.0193 |
| buu | rung2id | 25.96 ± 0.38 | 19.85 ± 0.76 | 0.8891 ± 0.0063 | 0.1958 ± 0.0075 |
| buu | rung2idexp | 27.32 ± 0.23 | 20.92 ± 0.34 | 0.9091 ± 0.0020 | 0.1593 ± 0.0035 |
| chair | official | 17.82 ± 0.18 | 14.76 ± 0.11 | 0.7972 ± 0.0066 | 0.3967 ± 0.0132 |
| chair | detach | 19.89 ± 0.82 | 15.49 ± 1.38 | 0.8145 ± 0.0253 | 0.3559 ± 0.0533 |
| chair | detachexp | 20.36 ± 0.21 | 16.16 ± 0.50 | 0.8358 ± 0.0026 | 0.3122 ± 0.0002 |
| chair | rung2id | 24.02 ± 0.28 | 20.45 ± 0.28 | 0.8660 ± 0.0031 | 0.2635 ± 0.0045 |
| chair | rung2idexp | 24.26 ± 0.21 | 20.66 ± 0.23 | 0.8722 ± 0.0007 | 0.2649 ± 0.0034 |
| sofa | official | 20.04 ± 0.50 | 18.77 ± 0.85 | 0.8534 ± 0.0111 | 0.3753 ± 0.0201 |
| sofa | detach | 23.84 ± 1.11 | 21.67 ± 2.59 | 0.8841 ± 0.0132 | 0.2997 ± 0.0235 |
| sofa | detachexp | 24.38 ± 1.23 | 23.55 ± 1.66 | 0.8972 ± 0.0075 | 0.2973 ± 0.0244 |
| sofa | rung2id | 28.66 ± 0.70 | 24.05 ± 0.20 | 0.9083 ± 0.0003 | 0.2214 ± 0.0117 |
| sofa | rung2idexp | 28.79 ± 0.48 | 24.38 ± 0.20 | 0.9131 ± 0.0019 | 0.2300 ± 0.0046 |
| bike | official | 19.24 ± 0.15 | 15.48 ± 0.93 | 0.7421 ± 0.0064 | 0.4027 ± 0.0111 |
| bike | detach | 20.42 ± 0.35 | 14.55 ± 1.18 | 0.7448 ± 0.0216 | 0.3288 ± 0.0120 |
| bike | detachexp | 20.84 ± 0.21 | 15.25 ± 0.32 | 0.7711 ± 0.0046 | 0.3235 ± 0.0080 |
| bike | rung2id | 21.55 ± 0.03 | 16.39 ± 0.07 | 0.7846 ± 0.0015 | 0.2795 ± 0.0025 |
| bike | rung2idexp | 21.40 ± 0.01 | 16.32 ± 0.03 | 0.7855 ± 0.0011 | 0.2961 ± 0.0052 |
| shrub | official | 17.13 ± 0.61 | 10.74 ± 0.68 | 0.5727 ± 0.0054 | 0.3231 ± 0.0685 |
| shrub | detach | 16.71 ± 0.03 | 9.83 ± 0.19 | 0.5496 ± 0.0007 | 0.2959 ± 0.0017 |
| shrub | detachexp | 16.82 ± 0.03 | 9.76 ± 0.08 | 0.5545 ± 0.0018 | 0.2908 ± 0.0007 |
| shrub | rung2id | 18.38 ± 0.02 | 12.11 ± 0.03 | 0.5926 ± 0.0016 | 0.3512 ± 0.0014 |
| shrub | rung2idexp | 18.41 ± 0.00 | 12.13 ± 0.02 | 0.5953 ± 0.0014 | 0.3506 ± 0.0020 |
| pooled | official | 19.31 ± 0.33 | 16.35 ± 0.59 | 0.7685 ± 0.0040 | 0.3517 ± 0.0073 |
| pooled | detach | 21.03 ± 0.72 | 16.87 ± 1.15 | 0.7748 ± 0.0126 | 0.3064 ± 0.0167 |
| pooled | detachexp | 21.28 ± 0.31 | 17.28 ± 0.25 | 0.7923 ± 0.0019 | 0.2880 ± 0.0069 |
| pooled | rung2id | 23.71 ± 0.14 | 18.57 ± 0.10 | 0.8081 ± 0.0009 | 0.2623 ± 0.0024 |
| pooled | rung2idexp | 24.04 ± 0.10 | 18.88 ± 0.07 | 0.8150 ± 0.0005 | 0.2602 ± 0.0021 |
| pooled_no_shrub | official | 19.86 ± 0.33 | 17.76 ± 0.65 | 0.8174 ± 0.0038 | 0.3589 ± 0.0080 |
| pooled_no_shrub | detach | 22.11 ± 0.91 | 18.63 ± 1.41 | 0.8310 ± 0.0156 | 0.3090 ± 0.0212 |
| pooled_no_shrub | detachexp | 22.40 ± 0.38 | 19.16 ± 0.31 | 0.8518 ± 0.0020 | 0.2874 ± 0.0084 |
| pooled_no_shrub | rung2id | 25.05 ± 0.17 | 20.18 ± 0.12 | 0.8620 ± 0.0011 | 0.2400 ± 0.0027 |
| pooled_no_shrub | rung2idexp | 25.44 ± 0.12 | 20.57 ± 0.09 | 0.8699 ± 0.0005 | 0.2376 ± 0.0029 |


## R4. Factorial effects on the varying set. Main effects and interaction: noise over the four cells (rung05_analyze.effects); simple effects: their two arms' noise (M1). One seed: ref sd = seed sd of detach and detachexp over seeds 0-2 (main effects) or of the second arm's cell

| scene | effect | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | curve model (H2): ½[(rung2id − detach) + (rung2idexp − detachexp)] | +2.50 (+2.2×) * | -1.87 (-3.2×) * | +0.0072 (+0.9×) | -0.0564 (-4.4×) * |
| buu | dark branch: ½[(detachexp − detach) + (rung2idexp − rung2id)] | +0.56 (+0.5×) | -0.03 (-0.0×) | +0.0211 (+2.6×) * | -0.0358 (-2.8×) * |
| buu | interaction: (rung2idexp − rung2id) − (detachexp − detach) | +1.60 (+1.4×) | +2.21 (+3.8×) * | -0.0023 (-0.3×) | -0.0013 (-0.1×) |
| buu | rung2id − detach (shared column) | +1.70 (+1.5×) | -2.97 (-3.9×) * | +0.0083 (+0.8×) | -0.0557 (-4.9×) * |
| buu | rung2idexp − detachexp (exp column) | +3.30 (+2.8×) * | -0.76 (-2.3×) * | +0.0061 (+1.4×) | -0.0571 (-4.1×) * |
| buu | detachexp − detach (detach row) | -0.24 (-0.2×) | -1.13 (-2.0×) | +0.0222 (+2.2×) * | -0.0351 (-2.1×) * |
| buu | rung2idexp − rung2id (rung2id row) | +1.36 (+4.3×) * | +1.07 (+1.8×) | +0.0200 (+4.3×) * | -0.0364 (-6.2×) * |
| chair | curve model (H2): ½[(rung2id − detach) + (rung2idexp − detachexp)] | +4.01 (+8.7×) * | +4.73 (+6.3×) * | +0.0439 (+3.4×) * | -0.0698 (-2.6×) * |
| chair | dark branch: ½[(detachexp − detach) + (rung2idexp − rung2id)] | +0.35 (+0.8×) | +0.44 (+0.6×) | +0.0137 (+1.1×) | -0.0212 (-0.8×) |
| chair | interaction: (rung2idexp − rung2id) − (detachexp − detach) | -0.23 (-0.5×) | -0.46 (-0.6×) | -0.0152 (-1.2×) | +0.0452 (+1.7×) |
| chair | rung2id − detach (shared column) | +4.13 (+6.7×) * | +4.96 (+5.0×) * | +0.0515 (+2.9×) * | -0.0924 (-2.4×) * |
| chair | rung2idexp − detachexp (exp column) | +3.89 (+18.6×) * | +4.50 (+11.6×) * | +0.0363 (+19.1×) * | -0.0473 (-19.4×) * |
| chair | detachexp − detach (detach row) | +0.47 (+0.8×) | +0.67 (+0.6×) | +0.0213 (+1.2×) | -0.0437 (-1.2×) |
| chair | rung2idexp − rung2id (rung2id row) | +0.24 (+1.0×) | +0.21 (+0.8×) | +0.0061 (+2.7×) * | +0.0014 (+0.4×) |
| sofa | curve model (H2): ½[(rung2id − detach) + (rung2idexp − detachexp)] | +4.62 (+5.0×) * | +1.61 (+1.0×) | +0.0200 (+2.6×) * | -0.0729 (-4.0×) * |
| sofa | dark branch: ½[(detachexp − detach) + (rung2idexp − rung2id)] | +0.33 (+0.4×) | +1.11 (+0.7×) | +0.0089 (+1.2×) | +0.0031 (+0.2×) |
| sofa | interaction: (rung2idexp − rung2id) − (detachexp − detach) | -0.40 (-0.4×) | -1.54 (-1.0×) | -0.0082 (-1.1×) | +0.0109 (+0.6×) |
| sofa | rung2id − detach (shared column) | +4.82 (+5.2×) * | +2.38 (+1.3×) | +0.0241 (+2.6×) * | -0.0783 (-4.2×) * |
| sofa | rung2idexp − detachexp (exp column) | +4.42 (+4.7×) * | +0.83 (+0.7×) | +0.0159 (+2.9×) * | -0.0674 (-3.8×) * |
| sofa | detachexp − detach (detach row) | +0.53 (+0.5×) | +1.88 (+0.9×) | +0.0130 (+1.2×) | -0.0024 (-0.1×) |
| sofa | rung2idexp − rung2id (rung2id row) | +0.13 (+0.2×) | +0.33 (+1.7×) | +0.0048 (+3.5×) * | +0.0086 (+1.0×) |
| bike | curve model (H2): ½[(rung2id − detach) + (rung2idexp − detachexp)] | +0.84 (+4.1×) * | +1.46 (+2.4×) * | +0.0271 (+2.4×) * | -0.0384 (-4.9×) * |
| bike | dark branch: ½[(detachexp − detach) + (rung2idexp − rung2id)] | +0.14 (+0.7×) | +0.31 (+0.5×) | +0.0136 (+1.2×) | +0.0057 (+0.7×) |
| bike | interaction: (rung2idexp − rung2id) − (detachexp − detach) | -0.56 (-2.7×) * | -0.76 (-1.2×) | -0.0253 (-2.3×) * | +0.0219 (+2.8×) * |
| bike | rung2id − detach (shared column) | +1.12 (+4.5×) * | +1.84 (+2.2×) * | +0.0398 (+2.6×) * | -0.0493 (-5.7×) * |
| bike | rung2idexp − detachexp (exp column) | +0.56 (+3.7×) * | +1.08 (+4.7×) * | +0.0145 (+4.4×) * | -0.0274 (-4.0×) * |
| bike | detachexp − detach (detach row) | +0.42 (+1.4×) | +0.69 (+0.8×) | +0.0263 (+1.7×) | -0.0052 (-0.5×) |
| bike | rung2idexp − rung2id (rung2id row) | -0.14 (-5.7×) * | -0.07 (-1.4×) | +0.0009 (+0.7×) | +0.0167 (+4.1×) * |
| shrub | curve model (H2): ½[(rung2id − detach) + (rung2idexp − detachexp)] | +1.63 (+64.2×) * | +2.32 (+22.3×) * | +0.0419 (+29.3×) * | +0.0575 (+37.1×) * |
| shrub | dark branch: ½[(detachexp − detach) + (rung2idexp − rung2id)] | +0.07 (+2.7×) * | -0.02 (-0.2×) | +0.0038 (+2.6×) * | -0.0029 (-1.9×) |
| shrub | interaction: (rung2idexp − rung2id) − (detachexp − detach) | -0.08 (-3.0×) * | +0.09 (+0.9×) | -0.0023 (-1.6×) | +0.0045 (+2.9×) * |
| shrub | rung2id − detach (shared column) | +1.66 (+55.5×) * | +2.28 (+16.8×) * | +0.0430 (+34.9×) * | +0.0553 (+34.5×) * |
| shrub | rung2idexp − detachexp (exp column) | +1.59 (+80.9×) * | +2.37 (+40.8×) * | +0.0408 (+25.5×) * | +0.0598 (+39.8×) * |
| shrub | detachexp − detach (detach row) | +0.11 (+3.4×) * | -0.07 (-0.5×) | +0.0049 (+3.6×) * | -0.0052 (-3.9×) * |
| shrub | rung2idexp − rung2id (rung2id row) | +0.03 (+1.8×) | +0.02 (+1.0×) | +0.0026 (+1.8×) | -0.0007 (-0.4×) |
| pooled | curve model (H2): ½[(rung2id − detach) + (rung2idexp − detachexp)] | +2.72 (+6.8×) * | +1.65 (+2.8×) * | +0.0280 (+4.4×) * | -0.0360 (-3.9×) * |
| pooled | dark branch: ½[(detachexp − detach) + (rung2idexp − rung2id)] | +0.29 (+0.7×) | +0.36 (+0.6×) | +0.0122 (+1.9×) | -0.0102 (-1.1×) |
| pooled | interaction: (rung2idexp − rung2id) − (detachexp − detach) | +0.07 (+0.2×) | -0.09 (-0.2×) | -0.0107 (-1.7×) | +0.0162 (+1.8×) |
| pooled | rung2id − detach (shared column) | +2.69 (+5.2×) * | +1.70 (+2.1×) * | +0.0334 (+3.7×) * | -0.0441 (-3.7×) * |
| pooled | rung2idexp − detachexp (exp column) | +2.75 (+12.0×) * | +1.60 (+8.7×) * | +0.0227 (+16.6×) * | -0.0279 (-5.5×) * |
| pooled | detachexp − detach (detach row) | +0.26 (+0.5×) | +0.41 (+0.5×) | +0.0176 (+1.9×) | -0.0183 (-1.4×) |
| pooled | rung2idexp − rung2id (rung2id row) | +0.32 (+2.7×) * | +0.31 (+3.7×) * | +0.0069 (+9.7×) * | -0.0021 (-0.9×) |
| pooled_no_shrub | curve model (H2): ½[(rung2id − detach) + (rung2idexp − detachexp)] | +2.99 (+6.0×) * | +1.48 (+2.0×) * | +0.0246 (+3.1×) * | -0.0594 (-5.1×) * |
| pooled_no_shrub | dark branch: ½[(detachexp − detach) + (rung2idexp − rung2id)] | +0.35 (+0.7×) | +0.46 (+0.6×) | +0.0143 (+1.8×) | -0.0120 (-1.0×) |
| pooled_no_shrub | interaction: (rung2idexp − rung2id) − (detachexp − detach) | +0.10 (+0.2×) | -0.14 (-0.2×) | -0.0128 (-1.6×) | +0.0192 (+1.7×) |
| pooled_no_shrub | rung2id − detach (shared column) | +2.94 (+4.5×) * | +1.55 (+1.6×) | +0.0309 (+2.8×) * | -0.0690 (-4.6×) * |
| pooled_no_shrub | rung2idexp − detachexp (exp column) | +3.04 (+10.8×) * | +1.41 (+6.2×) * | +0.0182 (+12.4×) * | -0.0498 (-7.9×) * |
| pooled_no_shrub | detachexp − detach (detach row) | +0.29 (+0.4×) | +0.53 (+0.5×) | +0.0207 (+1.9×) | -0.0216 (-1.3×) |
| pooled_no_shrub | rung2idexp − rung2id (rung2id row) | +0.40 (+2.7×) * | +0.39 (+3.6×) * | +0.0080 (+9.3×) * | -0.0024 (-0.9×) |


## R5a. Pseudo-target leak (plan 6.3a): slope of log2(mean pseudo-target, varying / LOM) on u_k, same view, seed and curve row (0 = compensated, 1 = passes straight through); unpaired = log2 mean on the varying set alone; input = the dark inputs (check: about 1); render rows = the enhanced renders at the training views, per cell (LOM partner: the row's LOM run)

| scene | curve row | pseudo_leak_paired | pseudo_leak | input_leak_paired | render_leak_paired |
|---|---|---|---|---|---|
| buu | official | 0.564 ± 0.003 | 0.588 ± 0.004 | 1.008 ± 0.000 |  |
| buu | detach | 0.653 ± 0.004 | 0.672 ± 0.004 | 1.008 ± 0.000 |  |
| buu | rung2id | 0.002 ± 0.000 | 0.001 ± 0.000 | 1.008 ± 0.000 |  |
| buu | official (render) |  |  |  | 0.454 ± 0.017 |
| buu | detach (render) |  |  |  | 0.536 ± 0.064 |
| buu | detachexp (render) |  |  |  | 0.393 ± 0.094 |
| buu | rung2id (render) |  |  |  | -0.002 ± 0.002 |
| buu | rung2idexp (render) |  |  |  | -0.003 ± 0.002 |
| buu | rung2id − detach | -0.650 (-239.8×) * |  |  |  |
| chair | official | 0.345 ± 0.024 | 0.359 ± 0.025 | 1.015 ± 0.000 |  |
| chair | detach | 0.496 ± 0.004 | 0.519 ± 0.003 | 1.015 ± 0.000 |  |
| chair | rung2id | 0.006 ± 0.000 | 0.008 ± 0.000 | 1.015 ± 0.000 |  |
| chair | official (render) |  |  |  | 0.089 ± 0.024 |
| chair | detach (render) |  |  |  | 0.146 ± 0.049 |
| chair | detachexp (render) |  |  |  | 0.099 ± 0.024 |
| chair | rung2id (render) |  |  |  | 0.004 ± 0.003 |
| chair | rung2idexp (render) |  |  |  | 0.001 ± 0.001 |
| chair | rung2id − detach | -0.490 (-191.4×) * |  |  |  |
| sofa | official | 0.399 ± 0.005 | 0.412 ± 0.006 | 1.002 ± 0.000 |  |
| sofa | detach | 0.547 ± 0.005 | 0.558 ± 0.005 | 1.002 ± 0.000 |  |
| sofa | rung2id | 0.000 ± 0.000 | -0.001 ± 0.000 | 1.002 ± 0.000 |  |
| sofa | official (render) |  |  |  | 0.136 ± 0.043 |
| sofa | detach (render) |  |  |  | 0.149 ± 0.084 |
| sofa | detachexp (render) |  |  |  | 0.083 ± 0.018 |
| sofa | rung2id (render) |  |  |  | 0.001 ± 0.001 |
| sofa | rung2idexp (render) |  |  |  | -0.000 ± 0.002 |
| sofa | rung2id − detach | -0.547 (-163.7×) * |  |  |  |
| bike | official | 0.271 ± 0.006 | 0.246 ± 0.006 | 0.996 ± 0.000 |  |
| bike | detach | 0.506 ± 0.008 | 0.477 ± 0.007 | 0.996 ± 0.000 |  |
| bike | rung2id | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.996 ± 0.000 |  |
| bike | official (render) |  |  |  | 0.056 ± 0.035 |
| bike | detach (render) |  |  |  | 0.156 ± 0.058 |
| bike | detachexp (render) |  |  |  | 0.059 ± 0.023 |
| bike | rung2id (render) |  |  |  | 0.003 ± 0.003 |
| bike | rung2idexp (render) |  |  |  | 0.000 ± 0.000 |
| bike | rung2id − detach | -0.499 (-94.0×) * |  |  |  |
| shrub | official | 0.445 ± 0.136 | 0.418 ± 0.118 | 1.009 ± 0.000 |  |
| shrub | detach | 0.357 ± 0.005 | 0.342 ± 0.006 | 1.009 ± 0.000 |  |
| shrub | rung2id | 0.027 ± 0.000 | 0.024 ± 0.000 | 1.009 ± 0.000 |  |
| shrub | official (render) |  |  |  | 0.194 ± 0.046 |
| shrub | detach (render) |  |  |  | 0.126 ± 0.054 |
| shrub | detachexp (render) |  |  |  | 0.053 ± 0.002 |
| shrub | rung2id (render) |  |  |  | 0.008 ± 0.001 |
| shrub | rung2idexp (render) |  |  |  | 0.005 ± 0.000 |
| shrub | rung2id − detach | -0.330 (-89.1×) * |  |  |  |
| pooled | official | 0.405 ± 0.032 | 0.404 ± 0.028 | 1.006 ± 0.000 |  |
| pooled | detach | 0.512 ± 0.001 | 0.514 ± 0.001 | 1.006 ± 0.000 |  |
| pooled | rung2id | 0.009 ± 0.000 | 0.009 ± 0.000 | 1.006 ± 0.000 |  |
| pooled | official (render) |  |  |  | 0.186 ± 0.018 |
| pooled | detach (render) |  |  |  | 0.222 ± 0.024 |
| pooled | detachexp (render) |  |  |  | 0.137 ± 0.019 |
| pooled | rung2id (render) |  |  |  | 0.003 ± 0.002 |
| pooled | rung2idexp (render) |  |  |  | 0.001 ± 0.000 |
| pooled | rung2id − detach | -0.503 (-725.7×) * |  |  |  |


## R5b. Dark-branch fit (plan 6.3b): PSNR of the dark render as its loss sees it against the input, mean over training views; the row's LOM level; the column effect within each curve row

| scene | cell | dark fit, varying | row's LOM level | column effect (exp − shared) |
|---|---|---|---|---|
| buu | detach | 41.18 ± 1.49 | 46.25 ± 0.26 |  |
| buu | detachexp | 42.41 ± 0.08 | 46.25 ± 0.26 | +1.23 (+1.2×) |
| buu | rung2id | 39.32 ± 1.39 | 46.04 ± 0.24 |  |
| buu | rung2idexp | 43.83 ± 0.34 | 46.04 ± 0.24 | +4.50 (+4.4×) * |
| chair | detach | 38.02 ± 1.44 | 49.02 ± 0.05 |  |
| chair | detachexp | 46.74 ± 0.00 | 49.02 ± 0.05 | +8.72 (+8.5×) * |
| chair | rung2id | 36.63 ± 0.32 | 48.92 ± 0.11 |  |
| chair | rung2idexp | 46.79 ± 0.13 | 48.92 ± 0.11 | +10.16 (+42.1×) * |
| sofa | detach | 35.57 ± 2.08 | 47.42 ± 0.10 |  |
| sofa | detachexp | 45.44 ± 0.33 | 47.42 ± 0.10 | +9.88 (+6.6×) * |
| sofa | rung2id | 34.57 ± 1.12 | 47.60 ± 0.05 |  |
| sofa | rung2idexp | 45.90 ± 0.08 | 47.60 ± 0.05 | +11.33 (+14.2×) * |
| bike | detach | 41.86 ± 0.62 | 47.72 ± 0.12 |  |
| bike | detachexp | 47.30 ± 0.07 | 47.72 ± 0.12 | +5.44 (+12.4×) * |
| bike | rung2id | 40.32 ± 0.75 | 47.66 ± 0.10 |  |
| bike | rung2idexp | 47.61 ± 0.07 | 47.66 ± 0.10 | +7.29 (+13.6×) * |
| shrub | detach | 37.11 ± 0.33 | 40.81 ± 0.06 |  |
| shrub | detachexp | 40.68 ± 0.04 | 40.81 ± 0.06 | +3.57 (+15.3×) * |
| shrub | rung2id | 36.48 ± 0.39 | 40.65 ± 0.03 |  |
| shrub | rung2idexp | 40.51 ± 0.04 | 40.65 ± 0.03 | +4.02 (+14.4×) * |
| pooled | detach | 38.75 ± 0.45 | 46.24 ± 0.03 |  |
| pooled | detachexp | 44.52 ± 0.05 | 46.24 ± 0.03 | +5.77 (+17.9×) * |
| pooled | rung2id | 37.46 ± 0.17 | 46.17 ± 0.04 |  |
| pooled | rung2idexp | 44.93 ± 0.11 | 46.17 ± 0.04 | +7.46 (+51.7×) * |
| pooled_no_shrub | detach | 39.16 ± 0.64 | 47.60 ± 0.03 |  |
| pooled_no_shrub | detachexp | 45.48 ± 0.07 | 47.60 ± 0.03 | +6.32 (+13.8×) * |
| pooled_no_shrub | rung2id | 37.71 ± 0.27 | 47.56 ± 0.06 |  |
| pooled_no_shrub | rung2idexp | 46.03 ± 0.13 | 47.56 ± 0.06 | +8.32 (+39.2×) * |


## R5b'. Per view: paired change of the dark fit, varying − LOM (same view, seed and curve row), against signed u_k, darkened (u<0) and brightened (u>0) views separately (M1 T5b; views with u = 0 left out)

| scene | cell | u_sign | n_view_seeds | mean_diff_db | slope_db_per_stop |
|---|---|---|---|---|---|
| buu | detach | u<0 | 33 | -4.67 | +2.05 |
| buu | detach | u>0 | 33 | -5.47 | -2.01 |
| buu | detachexp | u<0 | 33 | -2.03 | -5.43 |
| buu | detachexp | u>0 | 33 | -5.64 | -4.43 |
| buu | rung2id | u<0 | 33 | -5.56 | +4.41 |
| buu | rung2id | u>0 | 33 | -7.89 | -3.95 |
| buu | rung2idexp | u<0 | 33 | -0.09 | -4.56 |
| buu | rung2idexp | u>0 | 33 | -4.35 | -4.85 |
| chair | detach | u<0 | 63 | -9.96 | +7.22 |
| chair | detach | u>0 | 63 | -12.28 | -16.12 |
| chair | detachexp | u<0 | 63 | -0.30 | -1.97 |
| chair | detachexp | u>0 | 63 | -4.33 | -4.73 |
| chair | rung2id | u<0 | 63 | -11.16 | +5.01 |
| chair | rung2id | u>0 | 63 | -13.84 | -19.43 |
| chair | rung2idexp | u<0 | 63 | -0.11 | -2.25 |
| chair | rung2idexp | u>0 | 63 | -4.18 | -5.19 |
| sofa | detach | u<0 | 42 | -9.87 | +8.96 |
| sofa | detach | u>0 | 42 | -14.10 | -15.89 |
| sofa | detachexp | u<0 | 42 | +0.26 | -3.19 |
| sofa | detachexp | u>0 | 42 | -4.24 | -5.20 |
| sofa | rung2id | u<0 | 42 | -11.20 | +12.26 |
| sofa | rung2id | u>0 | 42 | -15.22 | -13.50 |
| sofa | rung2idexp | u<0 | 42 | +0.39 | -3.21 |
| sofa | rung2idexp | u>0 | 42 | -3.86 | -4.93 |
| bike | detach | u<0 | 54 | -3.28 | +1.97 |
| bike | detach | u>0 | 54 | -8.43 | -12.30 |
| bike | detachexp | u<0 | 54 | +1.98 | -4.24 |
| bike | detachexp | u>0 | 54 | -2.81 | -5.08 |
| bike | rung2id | u<0 | 54 | -5.38 | +5.70 |
| bike | rung2id | u>0 | 54 | -9.30 | -16.24 |
| bike | rung2idexp | u<0 | 54 | +2.39 | -4.07 |
| bike | rung2idexp | u>0 | 54 | -2.49 | -4.95 |
| shrub | detach | u<0 | 45 | -1.39 | +3.82 |
| shrub | detach | u>0 | 45 | -6.00 | -13.51 |
| shrub | detachexp | u<0 | 45 | +2.35 | -4.66 |
| shrub | detachexp | u>0 | 45 | -2.60 | -5.03 |
| shrub | rung2id | u<0 | 45 | -1.97 | +4.81 |
| shrub | rung2id | u>0 | 45 | -6.36 | -13.74 |
| shrub | rung2idexp | u<0 | 45 | +2.39 | -4.74 |
| shrub | rung2idexp | u>0 | 45 | -2.67 | -5.22 |
| pooled | detach | u<0 | 237 | -6.06 | +5.18 |
| pooled | detach | u>0 | 237 | -9.58 | -12.94 |
| pooled | detachexp | u<0 | 237 | +0.58 | -3.67 |
| pooled | detachexp | u>0 | 237 | -3.82 | -4.94 |
| pooled | rung2id | u<0 | 237 | -7.33 | +6.56 |
| pooled | rung2id | u>0 | 237 | -10.80 | -14.65 |
| pooled | rung2idexp | u<0 | 237 | +1.02 | -3.58 |
| pooled | rung2idexp | u>0 | 237 | -3.47 | -5.09 |


## R5c. Exposure recovery (plan 6.3c): Rung 2's final s_k against the applied gains u_k (ideal slope −1, RMSE 0); rmse = RMS of s + u (unpaired, floor near the natural spread); rmse_paired = with the same view's s on LOM subtracted (LOM partner: results/rung2); move = rmse_c(s − s_init). Reference rows from M1 (results/vexp/summary/T4_exposure.csv). A sanity check, not a test: the init already sits at the quantisation floor

| scene | method | slope | r | rmse | rmse_paired | move_from_init |
|---|---|---|---|---|---|---|
| buu | rung2id (s_k) | -1.04 ± 0.00 | -0.952 ± 0.000 | 0.195 ± 0.000 | 0.007 ± 0.000 | 0.006 ± 0.000 |
| buu | init (M1) | -1.04 | -0.952 | 0.193 | 0.006 |  |
| buu | oracle (M1) | -1.08 | -0.968 | 0.170 | 0.039 |  |
| buu | official (M1) post-hoc s_fit | 0.07 ± 0.00 | 0.801 ± 0.006 | 0.617 ± 0.001 | 0.613 ± 0.001 |  |
| buu | detach (M1) post-hoc s_fit | 0.10 ± 0.00 | 0.834 ± 0.004 | 0.635 ± 0.001 | 0.631 ± 0.001 |  |
| chair | rung2id (s_k) | -1.04 ± 0.00 | -0.992 ± 0.000 | 0.080 ± 0.000 | 0.012 ± 0.000 | 0.004 ± 0.000 |
| chair | init (M1) | -1.04 | -0.991 | 0.083 | 0.013 |  |
| chair | oracle (M1) | -1.09 | -0.995 | 0.083 | 0.043 |  |
| chair | official (M1) post-hoc s_fit | 0.31 ± 0.06 | 0.963 ± 0.024 | 0.756 ± 0.036 | 0.749 ± 0.035 |  |
| chair | detach (M1) post-hoc s_fit | 0.27 ± 0.02 | 0.951 ± 0.004 | 0.735 ± 0.009 | 0.730 ± 0.009 |  |
| sofa | rung2id (s_k) | -1.02 ± 0.00 | -0.991 ± 0.000 | 0.083 ± 0.000 | 0.005 ± 0.000 | 0.005 ± 0.000 |
| sofa | init (M1) | -1.02 | -0.991 | 0.079 | 0.004 |  |
| sofa | oracle (M1) | -1.07 | -0.996 | 0.066 | 0.038 |  |
| sofa | official (M1) post-hoc s_fit | 0.17 ± 0.01 | 0.947 ± 0.003 | 0.677 ± 0.004 | 0.672 ± 0.004 |  |
| sofa | detach (M1) post-hoc s_fit | 0.20 ± 0.00 | 0.905 ± 0.002 | 0.697 ± 0.001 | 0.693 ± 0.001 |  |
| bike | rung2id (s_k) | -0.94 ± 0.00 | -0.990 ± 0.000 | 0.084 ± 0.000 | 0.021 ± 0.000 | 0.006 ± 0.000 |
| bike | init (M1) | -0.95 | -0.990 | 0.083 | 0.019 |  |
| bike | oracle (M1) | -1.04 | -0.989 | 0.093 | 0.056 |  |
| bike | official (M1) post-hoc s_fit | 0.23 ± 0.04 | 0.982 ± 0.000 | 0.712 ± 0.023 | 0.716 ± 0.022 |  |
| bike | detach (M1) post-hoc s_fit | 0.26 ± 0.01 | 0.965 ± 0.003 | 0.730 ± 0.004 | 0.736 ± 0.004 |  |
| shrub | rung2id (s_k) | -0.98 ± 0.00 | -0.983 ± 0.000 | 0.105 ± 0.000 | 0.013 ± 0.000 | 0.004 ± 0.000 |
| shrub | init (M1) | -0.98 | -0.983 | 0.105 | 0.014 |  |
| shrub | oracle (M1) | -1.11 | -0.980 | 0.145 | 0.082 |  |
| shrub | official (M1) post-hoc s_fit | 0.26 ± 0.01 | 0.960 ± 0.009 | 0.729 ± 0.008 | 0.735 ± 0.008 |  |
| shrub | detach (M1) post-hoc s_fit | 0.44 ± 0.01 | 0.967 ± 0.002 | 0.832 ± 0.007 | 0.841 ± 0.008 |  |
| pooled | rung2id (s_k) | -1.00 ± 0.00 | -0.981 ± 0.000 | 0.109 ± 0.000 | 0.011 ± 0.000 | 0.005 ± 0.000 |
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