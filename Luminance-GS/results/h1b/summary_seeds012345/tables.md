# H1 rule: rung2id − detach, LOM low (M5 E3/E6) — context only, seeds 0-5 (detach code versions mixed)

Seeds 0, 1, 2, 3, 4, 5. Paired by seed; noise = sqrt(mean of the arms' seed variances); ×noise; * = real (above 2×). Pooled = mean over the five scenes per seed. Readings pre-registered in docs/RESEARCH_LOG.md (2026-11-24), M5_plan.md 3.1–3.6. Generated 2026-11-25 11:03 in 1 s by `rung2_ablation.py` (a7dafbc0a6ba9b5a). Every number is *measured*.

## Readings

- **H1 rule for rung2id − detach** (low; seeds 0, 1, 2, 3, 4, 5): buu: meets; chair: meets; sofa: meets; bike: meets -> 12 of 12 checks meet -> **supported**.


## Validation per arm (mean ± sd over seeds; pooled = mean over scenes per seed)

| scene | arm | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | rung2id | 26.91 ± 0.17 | 20.47 ± 0.24 | 0.9101 ± 0.0018 | 0.1590 ± 0.0036 |
| buu | detach | 26.34 ± 0.17 | 21.47 ± 0.34 | 0.9049 ± 0.0031 | 0.2006 ± 0.0063 |
| chair | rung2id | 23.59 ± 0.20 | 20.15 ± 0.22 | 0.8698 ± 0.0013 | 0.2835 ± 0.0017 |
| chair | detach | 23.15 ± 0.23 | 18.99 ± 0.32 | 0.8539 ± 0.0022 | 0.3077 ± 0.0041 |
| sofa | rung2id | 28.32 ± 0.32 | 23.79 ± 0.05 | 0.9137 ± 0.0042 | 0.2362 ± 0.0112 |
| sofa | detach | 27.54 ± 0.42 | 23.49 ± 0.32 | 0.9056 ± 0.0047 | 0.2571 ± 0.0066 |
| bike | rung2id | 21.43 ± 0.03 | 16.20 ± 0.02 | 0.7887 ± 0.0018 | 0.2994 ± 0.0072 |
| bike | detach | 21.30 ± 0.05 | 16.00 ± 0.11 | 0.7630 ± 0.0051 | 0.2908 ± 0.0050 |
| shrub | rung2id | 18.23 ± 0.01 | 11.95 ± 0.02 | 0.5879 ± 0.0012 | 0.3425 ± 0.0020 |
| shrub | detach | 18.18 ± 0.52 | 11.90 ± 0.74 | 0.5946 ± 0.0054 | 0.3028 ± 0.0495 |
| pooled | rung2id | 23.70 ± 0.09 | 18.51 ± 0.08 | 0.8140 ± 0.0009 | 0.2641 ± 0.0022 |
| pooled | detach | 23.30 ± 0.09 | 18.37 ± 0.13 | 0.8044 ± 0.0013 | 0.2718 ± 0.0102 |
| pooled_no_shrub | rung2id | 25.06 ± 0.11 | 20.15 ± 0.10 | 0.8706 ± 0.0011 | 0.2445 ± 0.0026 |
| pooled_no_shrub | detach | 24.58 ± 0.13 | 19.99 ± 0.17 | 0.8568 ± 0.0024 | 0.2641 ± 0.0026 |


## Paired by seed (×noise, * = real above 2×)

| scene | comparison | bm PSNR | raw PSNR | SSIM | LPIPS |
|---|---|---|---|---|---|
| buu | rung2id − detach | +0.57 (+3.4×) * | -1.01 (-3.4×) * | +0.0052 (+2.1×) * | -0.0416 (-8.1×) * |
| chair | rung2id − detach | +0.45 (+2.1×) * | +1.16 (+4.2×) * | +0.0160 (+8.8×) * | -0.0242 (-7.6×) * |
| sofa | rung2id − detach | +0.77 (+2.1×) * | +0.30 (+1.3×) | +0.0081 (+1.8×) | -0.0209 (-2.3×) * |
| bike | rung2id − detach | +0.12 (+2.9×) * | +0.20 (+2.6×) * | +0.0257 (+6.7×) * | +0.0086 (+1.4×) |
| shrub | rung2id − detach | +0.05 (+0.1×) | +0.05 (+0.1×) | -0.0067 (-1.7×) | +0.0397 (+1.1×) |
| pooled | rung2id − detach | +0.39 (+4.3×) * | +0.14 (+1.3×) | +0.0097 (+8.5×) * | -0.0077 (-1.0×) |
| pooled_no_shrub | rung2id − detach | +0.48 (+3.9×) * | +0.16 (+1.2×) | +0.0138 (+7.4×) * | -0.0195 (-7.5×) * |


## Extras, paired by seed (reported only)

| scene | comparison | hue err (°) | angle err (°) | sat_ratio | pseudo bm PSNR | dark fit (dB) | Gaussians | curve params | minutes/run |
|---|---|---|---|---|---|---|---|---|---|
| buu | rung2id − detach | -0.92 (-5.7×) * | -0.96 (-21.7×) * | -0.068 (-13.9×) * | +0.78 (+63.2×) * | -0.18 (-0.6×) | +78 (+0.0×) | n/a | -3.1 (-8.7×) * |
| chair | rung2id − detach | -1.97 (-6.6×) * | +0.11 (+3.4×) * | -0.036 (-8.2×) * | +0.26 (+3.3×) * | -0.12 (-1.4×) | -9827 (-2.4×) * | n/a | -3.2 (-8.4×) * |
| sofa | rung2id − detach | -3.64 (-5.6×) * | -0.34 (-2.8×) * | -0.102 (-5.3×) * | +0.92 (+90.2×) * | +0.00 (+0.0×) | +154 (+0.1×) | n/a | -3.0 (-8.9×) * |
| bike | rung2id − detach | +1.66 (+1.4×) | +0.10 (+7.2×) * | +0.023 (+3.8×) * | +0.33 (+53.8×) * | -0.04 (-0.4×) | -33890 (-3.2×) * | n/a | -3.2 (-8.6×) * |
| shrub | rung2id − detach | +0.96 (+1.8×) | -1.48 (-1.6×) | +0.026 (+0.2×) | +0.20 (+1.3×) | -0.05 (-0.5×) | +11510 (+1.1×) | n/a | -2.6 (-7.7×) * |
| pooled | rung2id − detach | -0.78 (-3.6×) * | -0.51 (-2.7×) * | -0.031 (-1.0×) | +0.50 (+16.8×) * | -0.08 (-1.1×) | -6395 (-1.6×) | n/a | -3.0 (-9.7×) * |
| pooled_no_shrub | rung2id − detach | -1.22 (-3.9×) * | -0.27 (-7.7×) * | -0.046 (-9.7×) * | +0.57 (+26.0×) * | -0.08 (-1.0×) | -10871 (-3.1×) * | n/a | -3.1 (-8.7×) * |


## H1 rule (plan 6.5) for rung2id − detach: 12 checks on buu, chair, sofa and bike; shrub reported only

| scene | metric | rung2id − detach | bar (2× noise, worse side) | check |
|---|---|---|---|---|
| buu | bm PSNR | +0.57 (+3.4×) * | -0.34 | meets |
| buu | SSIM | +0.0052 (+2.1×) * | -0.0050 | meets |
| buu | LPIPS | -0.0416 (-8.1×) * | +0.0102 | meets |
| chair | bm PSNR | +0.45 (+2.1×) * | -0.43 | meets |
| chair | SSIM | +0.0160 (+8.8×) * | -0.0036 | meets |
| chair | LPIPS | -0.0242 (-7.6×) * | +0.0063 | meets |
| sofa | bm PSNR | +0.77 (+2.1×) * | -0.75 | meets |
| sofa | SSIM | +0.0081 (+1.8×) | -0.0089 | meets |
| sofa | LPIPS | -0.0209 (-2.3×) * | +0.0184 | meets |
| bike | bm PSNR | +0.12 (+2.9×) * | -0.08 | meets |
| bike | SSIM | +0.0257 (+6.7×) * | -0.0077 | meets |
| bike | LPIPS | +0.0086 (+1.4×) | +0.0124 | meets |
