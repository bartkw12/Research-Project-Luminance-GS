# H1 rule: rung2id − detach, LOM low (M5 E3/E6)

Seeds 3, 4, 5. Paired by seed; noise = sqrt(mean of the arms' seed variances); ×noise; * = real (above 2×). Pooled = mean over the five scenes per seed. Readings pre-registered in docs/RESEARCH_LOG.md (2026-11-24), M5_plan.md 3.1–3.6. Generated 2026-11-25 11:03 in 1 s by `rung2_ablation.py` (a7dafbc0a6ba9b5a). Every number is *measured*.

## Readings

- **H1 rule for rung2id − detach** (low; seeds 3, 4, 5): buu: meets; chair: meets; sofa: meets; bike: meets -> 12 of 12 checks meet -> **supported**.


## Validation per arm (mean ± sd over seeds; pooled = mean over scenes per seed)

| scene | arm | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | rung2id | 26.93 ± 0.17 | 20.43 ± 0.06 | 0.9104 ± 0.0005 | 0.1605 ± 0.0036 |
| buu | detach | 26.30 ± 0.08 | 21.39 ± 0.04 | 0.9058 ± 0.0008 | 0.1986 ± 0.0007 |
| chair | rung2id | 23.68 ± 0.17 | 20.28 ± 0.04 | 0.8705 ± 0.0011 | 0.2829 ± 0.0023 |
| chair | detach | 23.20 ± 0.34 | 19.15 ± 0.42 | 0.8552 ± 0.0025 | 0.3049 ± 0.0039 |
| sofa | rung2id | 28.36 ± 0.31 | 23.81 ± 0.03 | 0.9145 ± 0.0055 | 0.2399 ± 0.0155 |
| sofa | detach | 27.68 ± 0.48 | 23.50 ± 0.42 | 0.9070 ± 0.0055 | 0.2543 ± 0.0074 |
| bike | rung2id | 21.44 ± 0.03 | 16.19 ± 0.02 | 0.7893 ± 0.0022 | 0.2968 ± 0.0086 |
| bike | detach | 21.30 ± 0.07 | 15.97 ± 0.13 | 0.7623 ± 0.0069 | 0.2906 ± 0.0039 |
| shrub | rung2id | 18.23 ± 0.01 | 11.95 ± 0.02 | 0.5882 ± 0.0016 | 0.3441 ± 0.0006 |
| shrub | detach | 18.30 ± 0.63 | 12.04 ± 0.94 | 0.5925 ± 0.0065 | 0.3180 ± 0.0517 |
| pooled | rung2id | 23.73 ± 0.12 | 18.53 ± 0.01 | 0.8146 ± 0.0011 | 0.2648 ± 0.0028 |
| pooled | detach | 23.36 ± 0.08 | 18.41 ± 0.19 | 0.8045 ± 0.0012 | 0.2733 ± 0.0110 |
| pooled_no_shrub | rung2id | 25.10 ± 0.14 | 20.18 ± 0.01 | 0.8712 ± 0.0014 | 0.2450 ± 0.0036 |
| pooled_no_shrub | detach | 24.62 ± 0.18 | 20.00 ± 0.20 | 0.8576 ± 0.0031 | 0.2621 ± 0.0018 |


## Paired by seed (×noise, * = real above 2×)

| scene | comparison | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | rung2id − detach | +0.62 (+4.7×) * | -0.96 (-19.2×) * | +0.0046 (+6.7×) * | -0.0380 (-14.6×) * |
| chair | rung2id − detach | +0.48 (+1.8×) | +1.12 (+3.7×) * | +0.0154 (+7.9×) * | -0.0221 (-6.9×) * |
| sofa | rung2id − detach | +0.68 (+1.7×) | +0.31 (+1.0×) | +0.0076 (+1.4×) | -0.0144 (-1.2×) |
| bike | rung2id − detach | +0.14 (+2.6×) * | +0.22 (+2.3×) * | +0.0270 (+5.3×) * | +0.0062 (+0.9×) |
| shrub | rung2id − detach | -0.07 (-0.2×) | -0.09 (-0.1×) | -0.0043 (-0.9×) | +0.0261 (+0.7×) |
| pooled | rung2id − detach | +0.37 (+3.7×) * | +0.12 (+0.9×) | +0.0101 (+8.6×) * | -0.0084 (-1.1×) |
| pooled_no_shrub | rung2id − detach | +0.48 (+3.0×) * | +0.17 (+1.2×) | +0.0136 (+5.7×) * | -0.0171 (-6.0×) * |


## Extras, paired by seed (reported only)

| scene | comparison | hue err (°) | angle err (°) | sat_ratio | pseudo bm PSNR | dark fit (dB) | Gaussians | curve params | minutes/run |
|---|---|---|---|---|---|---|---|---|---|
| buu | rung2id − detach | -0.95 (-6.4×) * | -0.95 (-36.2×) * | -0.067 (-22.2×) * | +0.78 (+49.3×) * | -0.16 (-0.5×) | -3982 (-1.0×) | n/a | -2.7 (-17.8×) * |
| chair | rung2id − detach | -2.29 (-7.3×) * | +0.13 (+3.0×) * | -0.040 (-7.2×) * | +0.18 (+2.1×) * | -0.13 (-1.4×) | -12811 (-2.5×) * | n/a | -2.8 (-24.9×) * |
| sofa | rung2id − detach | -3.71 (-5.2×) * | -0.24 (-1.4×) | -0.079 (-3.7×) * | +0.93 (+78.1×) * | -0.18 (-0.6×) | +2727 (+0.9×) | n/a | -2.7 (-41.5×) * |
| bike | rung2id − detach | +1.51 (+1.1×) | +0.09 (+8.3×) * | +0.021 (+3.6×) * | +0.33 (+46.5×) * | -0.02 (-0.1×) | -37379 (-5.2×) * | n/a | -2.8 (-20.5×) * |
| shrub | rung2id − detach | +0.86 (+1.3×) | -1.19 (-1.1×) | -0.039 (-0.2×) | +0.19 (+0.9×) | +0.06 (+0.6×) | +10422 (+1.1×) | n/a | -2.6 (-8.3×) * |
| pooled | rung2id − detach | -0.92 (-3.7×) * | -0.43 (-2.0×) | -0.041 (-1.2×) | +0.48 (+13.2×) * | -0.09 (-1.0×) | -8205 (-2.2×) * | n/a | -2.7 (-31.6×) * |
| pooled_no_shrub | rung2id − detach | -1.36 (-4.1×) * | -0.24 (-5.2×) * | -0.041 (-9.2×) * | +0.56 (+19.4×) * | -0.12 (-1.1×) | -12861 (-4.0×) * | n/a | -2.8 (-44.5×) * |


## H1 rule (plan 6.5) for rung2id − detach: 12 checks on buu, chair, sofa and bike; shrub reported only

| scene | metric | rung2id − detach | bar (2× noise, worse side) | check |
|---|---|---|---|---|
| buu | bm PSNR | +0.62 (+4.7×) * | -0.27 | meets |
| buu | SSIM | +0.0046 (+6.7×) * | -0.0014 | meets |
| buu | LPIPS | -0.0380 (-14.6×) * | +0.0052 | meets |
| chair | bm PSNR | +0.48 (+1.8×) | -0.54 | meets |
| chair | SSIM | +0.0154 (+7.9×) * | -0.0039 | meets |
| chair | LPIPS | -0.0221 (-6.9×) * | +0.0064 | meets |
| sofa | bm PSNR | +0.68 (+1.7×) | -0.81 | meets |
| sofa | SSIM | +0.0076 (+1.4×) | -0.0110 | meets |
| sofa | LPIPS | -0.0144 (-1.2×) | +0.0243 | meets |
| bike | bm PSNR | +0.14 (+2.6×) * | -0.11 | meets |
| bike | SSIM | +0.0270 (+5.3×) * | -0.0102 | meets |
| bike | LPIPS | +0.0062 (+0.9×) | +0.0134 | meets |
