# Luminance-GS reproduction: 3-run averages

Setup: `simple_trainer_ours.py --exp_name low`, 10k steps, final eval at step 9999. Run 1 = original run in this repo; runs 2 and 3 = fresh reruns with identical config.

## Averages (mean ± sample std, n=3)

| Scene | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| buu | 21.30 ± 0.38 | 0.9116 ± 0.0010 | 0.1809 ± 0.0016 |
| chair | 17.48 ± 0.05 | 0.8377 ± 0.0014 | 0.3317 ± 0.0047 |
| sofa | 23.41 ± 0.54 | 0.9061 ± 0.0022 | 0.2569 ± 0.0104 |
| **mean of 3 scenes** | 20.73 | 0.8851 | 0.2565 |

## Per-run values

| Scene | Run | PSNR | SSIM | LPIPS | #GS |
|---|---|---|---|---|---|
| buu | 1 | 21.730 | 0.9127 | 0.1798 | 81900 |
| buu | 2 | 21.137 | 0.9108 | 0.1827 | 82506 |
| buu | 3 | 21.025 | 0.9112 | 0.1801 | 78367 |
| chair | 1 | 17.507 | 0.8373 | 0.3341 | 50750 |
| chair | 2 | 17.423 | 0.8364 | 0.3347 | 49953 |
| chair | 3 | 17.508 | 0.8392 | 0.3262 | 54816 |
| sofa | 1 | 23.149 | 0.9041 | 0.2687 | 36457 |
| sofa | 2 | 24.032 | 0.9084 | 0.2489 | 37361 |
| sofa | 3 | 23.042 | 0.9059 | 0.2530 | 37088 |

Paper reference (buu only, as noted in README): PSNR 18.09 / SSIM 0.877 / LPIPS 0.193.

## Notes

- The trainer calls `set_random_seed(42)` and it was not changed, so all runs share the same seed. The variation between runs comes from GPU nondeterminism (e.g. atomic ops in rasterization and densification), which still changes the Gaussian count and metrics noticeably (see PSNR spread on sofa).
- Std is the sample std (n=3); with so few runs treat it as a rough indication only.
- Raw `val_step9999.json` for every run are in this folder (`run{1,2,3}_<scene>_val_step9999.json`).
