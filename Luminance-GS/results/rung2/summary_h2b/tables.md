# H2b(a): the identity init on the varying set, init × dark-branch 2×2 (M5 E1)

Seeds 0, 1, 2. Paired by seed; noise = sqrt(mean of the arms' seed variances); ×noise; * = real (above 2×). Pooled = mean over the five scenes per seed. Readings pre-registered in docs/RESEARCH_LOG.md (2026-11-24), M5_plan.md 3.1–3.6. Generated 2026-11-25 11:03 in 2 s by `rung2_ablation.py` (a7dafbc0a6ba9b5a). Every number is *measured*.

## Readings

- **H2b(a), pooled:** init main effect on bm PSNR +0.61 (+5.1×) *; LPIPS -0.0117 (-3.1×) * -> **confirmed** (confirmed: bm real and positive, LPIPS not worse beyond 2×; mixed: LPIPS worse beyond 2×; reversed: bm real and negative; else not confirmed).
- **H2b(a), pooled_no_shrub:** init main effect on bm PSNR +0.76 (+5.2×) *; LPIPS -0.0155 (-3.2×) * -> **confirmed** (confirmed: bm real and positive, LPIPS not worse beyond 2×; mixed: LPIPS worse beyond 2×; reversed: bm real and negative; else not confirmed).
- **H2b(a), bike:** init main effect on bm PSNR +0.28 (+9.7×) *; LPIPS +0.0018 (+0.3×) -> **confirmed** (confirmed: bm real and positive, LPIPS not worse beyond 2×; mixed: LPIPS worse beyond 2×; reversed: bm real and negative; else not confirmed).


## The init × dark-branch 2×2 on the varying set: validation per cell

| scene | arm | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | rung2 | 25.53 ± 0.32 | 19.63 ± 0.64 | 0.8808 ± 0.0072 | 0.2053 ± 0.0214 |
| buu | rung2exp | 26.43 ± 0.23 | 20.50 ± 0.18 | 0.8936 ± 0.0023 | 0.1791 ± 0.0052 |
| buu | rung2id | 25.96 ± 0.38 | 19.85 ± 0.76 | 0.8891 ± 0.0063 | 0.1958 ± 0.0075 |
| buu | rung2idexp | 27.32 ± 0.23 | 20.92 ± 0.34 | 0.9091 ± 0.0020 | 0.1593 ± 0.0035 |
| chair | rung2 | 23.63 ± 0.33 | 20.03 ± 0.19 | 0.8520 ± 0.0009 | 0.2826 ± 0.0043 |
| chair | rung2exp | 23.80 ± 0.26 | 20.39 ± 0.22 | 0.8589 ± 0.0009 | 0.2893 ± 0.0017 |
| chair | rung2id | 24.02 ± 0.28 | 20.45 ± 0.28 | 0.8660 ± 0.0031 | 0.2635 ± 0.0045 |
| chair | rung2idexp | 24.26 ± 0.21 | 20.66 ± 0.23 | 0.8722 ± 0.0007 | 0.2649 ± 0.0034 |
| sofa | rung2 | 26.79 ± 0.18 | 23.03 ± 0.28 | 0.8936 ± 0.0010 | 0.2559 ± 0.0076 |
| sofa | rung2exp | 27.33 ± 0.17 | 23.58 ± 0.15 | 0.9030 ± 0.0025 | 0.2502 ± 0.0120 |
| sofa | rung2id | 28.66 ± 0.70 | 24.05 ± 0.20 | 0.9083 ± 0.0003 | 0.2214 ± 0.0117 |
| sofa | rung2idexp | 28.79 ± 0.48 | 24.38 ± 0.20 | 0.9131 ± 0.0019 | 0.2300 ± 0.0046 |
| bike | rung2 | 21.23 ± 0.04 | 16.48 ± 0.09 | 0.7537 ± 0.0075 | 0.2876 ± 0.0062 |
| bike | rung2exp | 21.15 ± 0.02 | 16.46 ± 0.03 | 0.7639 ± 0.0060 | 0.2843 ± 0.0090 |
| bike | rung2id | 21.55 ± 0.03 | 16.39 ± 0.07 | 0.7846 ± 0.0015 | 0.2795 ± 0.0025 |
| bike | rung2idexp | 21.40 ± 0.01 | 16.32 ± 0.03 | 0.7855 ± 0.0011 | 0.2961 ± 0.0052 |
| shrub | rung2 | 18.39 ± 0.02 | 12.07 ± 0.04 | 0.5941 ± 0.0007 | 0.3468 ± 0.0010 |
| shrub | rung2exp | 18.41 ± 0.01 | 12.08 ± 0.01 | 0.5962 ± 0.0004 | 0.3485 ± 0.0009 |
| shrub | rung2id | 18.38 ± 0.02 | 12.11 ± 0.03 | 0.5926 ± 0.0016 | 0.3512 ± 0.0014 |
| shrub | rung2idexp | 18.41 ± 0.00 | 12.13 ± 0.02 | 0.5953 ± 0.0014 | 0.3506 ± 0.0020 |
| pooled | rung2 | 23.11 ± 0.11 | 18.25 ± 0.10 | 0.7949 ± 0.0031 | 0.2756 ± 0.0066 |
| pooled | rung2exp | 23.42 ± 0.12 | 18.60 ± 0.09 | 0.8031 ± 0.0011 | 0.2703 ± 0.0020 |
| pooled | rung2id | 23.71 ± 0.14 | 18.57 ± 0.10 | 0.8081 ± 0.0009 | 0.2623 ± 0.0024 |
| pooled | rung2idexp | 24.04 ± 0.10 | 18.88 ± 0.07 | 0.8150 ± 0.0005 | 0.2602 ± 0.0021 |
| pooled_no_shrub | rung2 | 24.30 ± 0.14 | 19.79 ± 0.11 | 0.8450 ± 0.0038 | 0.2578 ± 0.0083 |
| pooled_no_shrub | rung2exp | 24.68 ± 0.15 | 20.23 ± 0.12 | 0.8549 ± 0.0013 | 0.2507 ± 0.0027 |
| pooled_no_shrub | rung2id | 25.05 ± 0.17 | 20.18 ± 0.12 | 0.8620 ± 0.0011 | 0.2400 ± 0.0027 |
| pooled_no_shrub | rung2idexp | 25.44 ± 0.12 | 20.57 ± 0.09 | 0.8699 ± 0.0005 | 0.2376 ± 0.0029 |


## Main effects and interaction, noise over the four cells (rung05_analyze.effects)

| scene | effect | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | init (H2b(a)): ½[(rung2id − rung2) + (rung2idexp − rung2exp)] | +0.66 (+2.2×) * | +0.32 (+0.6×) | +0.0119 (+2.4×) * | -0.0146 (-1.2×) |
| buu | dark branch: ½[(rung2exp − rung2) + (rung2idexp − rung2id)] | +1.13 (+3.8×) * | +0.97 (+1.8×) | +0.0164 (+3.3×) * | -0.0313 (-2.7×) * |
| buu | interaction: (rung2idexp − rung2id) − (rung2exp − rung2) | +0.46 (+1.5×) | +0.21 (+0.4×) | +0.0072 (+1.4×) | -0.0103 (-0.9×) |
| chair | init (H2b(a)): ½[(rung2id − rung2) + (rung2idexp − rung2exp)] | +0.42 (+1.5×) | +0.35 (+1.5×) | +0.0136 (+7.9×) * | -0.0218 (-6.0×) * |
| chair | dark branch: ½[(rung2exp − rung2) + (rung2idexp − rung2id)] | +0.20 (+0.7×) | +0.28 (+1.2×) | +0.0065 (+3.8×) * | +0.0041 (+1.1×) |
| chair | interaction: (rung2idexp − rung2id) − (rung2exp − rung2) | +0.07 (+0.3×) | -0.15 (-0.6×) | -0.0007 (-0.4×) | -0.0053 (-1.4×) |
| sofa | init (H2b(a)): ½[(rung2id − rung2) + (rung2idexp − rung2exp)] | +1.67 (+3.8×) * | +0.90 (+4.2×) * | +0.0123 (+7.4×) * | -0.0274 (-2.9×) * |
| sofa | dark branch: ½[(rung2exp − rung2) + (rung2idexp − rung2id)] | +0.34 (+0.8×) | +0.44 (+2.1×) * | +0.0071 (+4.3×) * | +0.0015 (+0.2×) |
| sofa | interaction: (rung2idexp − rung2id) − (rung2exp − rung2) | -0.41 (-0.9×) | -0.22 (-1.0×) | -0.0046 (-2.8×) * | +0.0142 (+1.5×) |
| bike | init (H2b(a)): ½[(rung2id − rung2) + (rung2idexp − rung2exp)] | +0.28 (+9.7×) * | -0.11 (-1.8×) | +0.0262 (+5.4×) * | +0.0018 (+0.3×) |
| bike | dark branch: ½[(rung2exp − rung2) + (rung2idexp − rung2id)] | -0.11 (-3.8×) * | -0.05 (-0.8×) | +0.0056 (+1.1×) | +0.0067 (+1.1×) |
| bike | interaction: (rung2idexp − rung2id) − (rung2exp − rung2) | -0.06 (-2.0×) | -0.05 (-0.7×) | -0.0093 (-1.9×) | +0.0199 (+3.2×) * |
| shrub | init (H2b(a)): ½[(rung2id − rung2) + (rung2idexp − rung2exp)] | -0.01 (-0.6×) | +0.05 (+1.7×) | -0.0012 (-1.0×) | +0.0033 (+2.3×) * |
| shrub | dark branch: ½[(rung2exp − rung2) + (rung2idexp − rung2id)] | +0.03 (+1.7×) | +0.02 (+0.5×) | +0.0023 (+2.1×) * | +0.0005 (+0.4×) |
| shrub | interaction: (rung2idexp − rung2id) − (rung2exp − rung2) | +0.01 (+0.7×) | +0.02 (+0.7×) | +0.0006 (+0.5×) | -0.0024 (-1.7×) |
| pooled | init (H2b(a)): ½[(rung2id − rung2) + (rung2idexp − rung2exp)] | +0.61 (+5.1×) * | +0.30 (+3.3×) * | +0.0126 (+7.3×) * | -0.0117 (-3.1×) * |
| pooled | dark branch: ½[(rung2exp − rung2) + (rung2idexp − rung2id)] | +0.32 (+2.7×) * | +0.33 (+3.7×) * | +0.0076 (+4.4×) * | -0.0037 (-1.0×) |
| pooled | interaction: (rung2idexp − rung2id) − (rung2exp − rung2) | +0.01 (+0.1×) | -0.04 (-0.4×) | -0.0014 (-0.8×) | +0.0032 (+0.9×) |
| pooled_no_shrub | init (H2b(a)): ½[(rung2id − rung2) + (rung2idexp − rung2exp)] | +0.76 (+5.2×) * | +0.36 (+3.3×) * | +0.0160 (+7.5×) * | -0.0155 (-3.2×) * |
| pooled_no_shrub | dark branch: ½[(rung2exp − rung2) + (rung2idexp − rung2id)] | +0.39 (+2.7×) * | +0.41 (+3.7×) * | +0.0089 (+4.2×) * | -0.0048 (-1.0×) |
| pooled_no_shrub | interaction: (rung2idexp − rung2id) − (rung2exp − rung2) | +0.01 (+0.1×) | -0.05 (-0.5×) | -0.0019 (-0.9×) | +0.0047 (+1.0×) |


## Simple effects, paired by seed (their two arms' noise)

| scene | comparison | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | rung2id − rung2 (shared column) | +0.43 (+1.2×) | +0.22 (+0.3×) | +0.0083 (+1.2×) | -0.0095 (-0.6×) |
| buu | rung2idexp − rung2exp (exp column) | +0.89 (+3.8×) * | +0.42 (+1.6×) | +0.0155 (+7.2×) * | -0.0198 (-4.4×) * |
| chair | rung2id − rung2 (shared column) | +0.38 (+1.3×) | +0.42 (+1.8×) | +0.0140 (+6.0×) * | -0.0191 (-4.4×) * |
| chair | rung2idexp − rung2exp (exp column) | +0.46 (+1.9×) | +0.27 (+1.2×) | +0.0132 (+17.0×) * | -0.0244 (-9.0×) * |
| sofa | rung2id − rung2 (shared column) | +1.87 (+3.7×) * | +1.01 (+4.2×) * | +0.0146 (+19.6×) * | -0.0345 (-3.5×) * |
| sofa | rung2idexp − rung2exp (exp column) | +1.46 (+4.0×) * | +0.80 (+4.5×) * | +0.0100 (+4.5×) * | -0.0203 (-2.2×) * |
| bike | rung2id − rung2 (shared column) | +0.31 (+8.1×) * | -0.09 (-1.1×) | +0.0309 (+5.7×) * | -0.0082 (-1.7×) |
| bike | rung2idexp − rung2exp (exp column) | +0.26 (+17.2×) * | -0.13 (-4.3×) * | +0.0216 (+5.0×) * | +0.0118 (+1.6×) |
| shrub | rung2id − rung2 (shared column) | -0.01 (-0.7×) | +0.04 (+1.1×) | -0.0015 (-1.2×) | +0.0045 (+3.6×) * |
| shrub | rung2idexp − rung2exp (exp column) | -0.00 (-0.7×) | +0.06 (+3.7×) * | -0.0009 (-0.8×) | +0.0020 (+1.3×) |
| pooled | rung2id − rung2 (shared column) | +0.60 (+4.8×) * | +0.32 (+3.3×) * | +0.0133 (+5.8×) * | -0.0134 (-2.7×) * |
| pooled | rung2idexp − rung2exp (exp column) | +0.61 (+5.5×) * | +0.28 (+3.5×) * | +0.0119 (+13.5×) * | -0.0101 (-4.9×) * |
| pooled_no_shrub | rung2id − rung2 (shared column) | +0.75 (+4.9×) * | +0.39 (+3.3×) * | +0.0169 (+6.0×) * | -0.0178 (-2.9×) * |
| pooled_no_shrub | rung2idexp − rung2exp (exp column) | +0.77 (+5.5×) * | +0.34 (+3.3×) * | +0.0151 (+15.0×) * | -0.0132 (-4.7×) * |
