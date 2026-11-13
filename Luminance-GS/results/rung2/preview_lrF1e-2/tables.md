# Rung 2 preview (M2 step 6)

Curve loss only, no 3DGS: a replay of the trainer's F and s_k updates on the training photos. 10000 steps, lr 0.01 (F) and 0.001 (s), K = 32, smoothness 0.1, seed 0. Default configuration `quadratic-cdf` (one-bin-shifted target, decided 2026-11-11); `quadratic-cdf-unshifted` is the alignment check. Generated 2026-11-11 15:10 in 918 s by `rung2_preview.py` (e57354181bd189ab). Every number here is *measured* on the replay; how a 3DGS run follows it is *inferred*.

## Decision rules (training photos only; Bart confirms)

- **Knot spacing:** pooled P1 residual quadratic 7.851/255, usedrange 7.137/255, reduction 9.1% (rule: switch at >= 30%) -> **quadratic**.
- **Learning rate:** largest gap of the replay objective above the full-batch optimum 0.33% (rule: try 3e-3 above 5%) -> **keep 1e-3**.
- **F init:** steps the identity init needs to come within 0.01 of the cdf init's final F: quadratic-identity: never, never, never, never, never, never, never, never, never, never; usedrange-identity: never, never, never, never, never, never, never, never, never, never (in the order buu/lom, buu/vexp, chair/lom, chair/vexp, sofa/lom, sofa/vexp, bike/lom, bike/vexp, shrub/lom, shrub/vexp).
- **P5 flag (diagnostic only):** LOM scenes where `quadratic-cdf` is below `detach` by more than detach's brightness-matched seed sd: buu, chair, sofa, bike -> **stop and ask Bart**.

Exploratory configurations (not in the plan, added for Bart's decision): quadratic-cdf-priorF, quadratic-cdf-priorL. They add the official shape-prior term 0.5 [mse(p, X) + 0.01 mse(p, T_k)] with one learned global gamma, p = clamp(x, 1e-3, 1)^gamma, on the shared F (priorF) or on each view's composed curve (priorL, the official form). They take no part in the rules above.

## P1 fit: weighted residual x255 against the targets

| scene | set | oracle | quadratic-cdf | usedrange-cdf | quadratic-identity | usedrange-identity | quadratic-cdf-priorF | quadratic-cdf-priorL | usedrange y_max | gamma priorF | gamma priorL | detach gamma (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buu | lom | 11.665 | 13.183 | 12.765 | 15.237 | 26.328 | 40.405 | 40.378 | 0.236 | 0.164 | 0.164 | 0.195 |
| buu | vexp | 11.676 | 13.487 | 13.037 | 15.530 | 27.026 | 39.202 | 41.959 | 0.236 | 0.165 | 0.165 | 0.378 |
| chair | lom | 5.832 | 8.406 | 5.861 | 12.403 | 31.514 | 43.617 | 43.658 | 0.122 | 0.108 | 0.108 | 0.095 |
| chair | vexp | 7.345 | 9.541 | 7.459 | 13.105 | 32.663 | 42.425 | 45.253 | 0.123 | 0.108 | 0.108 | -0.065 |
| sofa | lom | 7.841 | 8.729 | 8.592 | 13.230 | 29.103 | 42.041 | 41.839 | 0.165 | 0.136 | 0.136 | 0.149 |
| sofa | vexp | 8.008 | 9.034 | 8.872 | 13.377 | 30.069 | 40.970 | 42.833 | 0.166 | 0.137 | 0.138 | 0.073 |
| bike | lom | 3.998 | 4.617 | 3.867 | 26.520 | 53.149 | 53.215 | 53.021 | 0.151 | 0.075 | 0.075 | 0.065 |
| bike | vexp | 6.512 | 7.071 | 6.613 | 28.743 | 53.952 | 52.574 | 54.181 | 0.152 | 0.075 | 0.075 | -0.145 |
| shrub | lom | 1.888 | 2.026 | 1.923 | 13.096 | 34.321 | 45.755 | 45.871 | 0.188 | 0.077 | 0.077 | 0.068 |
| shrub | vexp | 2.492 | 2.414 | 2.382 | 13.625 | 33.882 | 43.680 | 45.626 | 0.188 | 0.077 | 0.077 | -0.151 |


Default configuration: objective against the full-batch optimum, and convergence.

| scene | set | objective | optimum | gap % | step within 5% | init-to-final max dF | replay s |
|---|---|---|---|---|---|---|---|
| buu | lom | 2.056e-04 | 2.054e-04 | +0.06 | 100 | 0.017 | 7.6 |
| buu | vexp | 2.163e-04 | 2.161e-04 | +0.09 | 100 | 0.019 | 7.6 |
| chair | lom | 6.294e-05 | 6.273e-05 | +0.33 | 200 | 0.023 | 7.7 |
| chair | vexp | 7.747e-05 | 7.734e-05 | +0.17 | 200 | 0.026 | 7.6 |
| sofa | lom | 7.750e-05 | 7.751e-05 | -0.02 | 100 | 0.014 | 7.9 |
| sofa | vexp | 8.982e-05 | 8.983e-05 | -0.00 | 100 | 0.019 | 7.3 |
| bike | lom | 5.039e-05 | 5.042e-05 | -0.05 | 100 | 0.024 | 7.7 |
| bike | vexp | 5.969e-05 | 5.970e-05 | -0.03 | 200 | 0.023 | 7.7 |
| shrub | lom | 2.597e-05 | 2.598e-05 | -0.04 | 100 | 0.007 | 7.7 |
| shrub | vexp | 2.860e-05 | 2.860e-05 | -0.03 | 100 | 0.007 | 9.2 |


## P2 exposure on the varying set: s_k against u_k (ideal slope -1, RMSE 0)

| scene | method | slope | r | rmse | rmse paired | move from init |
|---|---|---|---|---|---|---|
| buu | init | -1.037 | -0.952 | 0.193 | 0.006 | 0.000 |
| buu | oracle | -1.083 | -0.968 | 0.170 | 0.039 | 0.044 |
| buu | quadratic-cdf | -1.026 | -0.958 | 0.178 | 0.007 | 0.017 |
| buu | usedrange-cdf | -1.025 | -0.960 | 0.172 | 0.007 | 0.022 |
| buu | quadratic-cdf-unshifted | -1.068 | -0.959 | 0.185 | 0.024 | 0.022 |
| buu | quadratic-cdf-priorF | -1.039 | -0.957 | 0.183 | 0.010 | 0.013 |
| buu | quadratic-cdf-priorL | -0.820 | -0.958 | 0.176 | 0.117 | 0.135 |
| chair | init | -1.042 | -0.991 | 0.083 | 0.013 | 0.000 |
| chair | oracle | -1.093 | -0.995 | 0.083 | 0.043 | 0.044 |
| chair | quadratic-cdf | -1.032 | -0.993 | 0.072 | 0.012 | 0.022 |
| chair | usedrange-cdf | -1.024 | -0.994 | 0.065 | 0.011 | 0.029 |
| chair | quadratic-cdf-unshifted | -1.095 | -0.994 | 0.088 | 0.042 | 0.042 |
| chair | quadratic-cdf-priorF | -1.034 | -0.993 | 0.072 | 0.012 | 0.027 |
| chair | quadratic-cdf-priorL | -0.886 | -0.992 | 0.092 | 0.080 | 0.093 |
| sofa | init | -1.022 | -0.991 | 0.079 | 0.004 | 0.000 |
| sofa | oracle | -1.069 | -0.996 | 0.066 | 0.038 | 0.041 |
| sofa | quadratic-cdf | -1.007 | -0.991 | 0.078 | 0.007 | 0.011 |
| sofa | usedrange-cdf | -1.007 | -0.991 | 0.077 | 0.007 | 0.011 |
| sofa | quadratic-cdf-unshifted | -1.066 | -0.991 | 0.089 | 0.030 | 0.027 |
| sofa | quadratic-cdf-priorF | -1.023 | -0.990 | 0.084 | 0.013 | 0.013 |
| sofa | quadratic-cdf-priorL | -0.805 | -0.991 | 0.129 | 0.120 | 0.127 |
| bike | init | -0.952 | -0.990 | 0.083 | 0.019 | 0.000 |
| bike | oracle | -1.043 | -0.989 | 0.093 | 0.056 | 0.058 |
| bike | quadratic-cdf | -0.893 | -0.989 | 0.100 | 0.048 | 0.037 |
| bike | usedrange-cdf | -0.893 | -0.989 | 0.100 | 0.047 | 0.036 |
| bike | quadratic-cdf-unshifted | -1.031 | -0.990 | 0.089 | 0.048 | 0.049 |
| bike | quadratic-cdf-priorF | -0.912 | -0.989 | 0.093 | 0.037 | 0.026 |
| bike | quadratic-cdf-priorL | -0.768 | -0.989 | 0.150 | 0.119 | 0.107 |
| shrub | init | -0.984 | -0.983 | 0.105 | 0.014 | 0.000 |
| shrub | oracle | -1.108 | -0.980 | 0.145 | 0.082 | 0.077 |
| shrub | quadratic-cdf | -0.952 | -0.981 | 0.112 | 0.018 | 0.021 |
| shrub | usedrange-cdf | -0.953 | -0.981 | 0.111 | 0.018 | 0.020 |
| shrub | quadratic-cdf-unshifted | -1.076 | -0.982 | 0.127 | 0.062 | 0.056 |
| shrub | quadratic-cdf-priorF | -0.977 | -0.982 | 0.110 | 0.018 | 0.012 |
| shrub | quadratic-cdf-priorL | -0.782 | -0.982 | 0.153 | 0.116 | 0.119 |


## P3 paired pseudo-target leak (slope of log2(mean, varying / LOM) on u_k; 0 = compensated, 1 = passes through)

M1 columns: mean over 3 seeds from `results/vexp/summary/T4_exposure.csv`.

| scene | input | official_M1 | detach_M1 | quadratic-cdf | usedrange-cdf | quadratic-identity | usedrange-identity | quadratic-cdf-unshifted | quadratic-cdf-priorF | quadratic-cdf-priorL | unpaired_quadratic-cdf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buu | +1.008 | +0.564 | +0.653 | +0.013 | +0.012 | +0.012 | +0.009 | -0.026 | +0.001 | +0.122 | +0.012 |
| chair | +1.015 | +0.345 | +0.496 | +0.020 | +0.025 | +0.017 | +0.007 | -0.047 | +0.008 | +0.086 | +0.024 |
| sofa | +1.002 | +0.399 | +0.547 | +0.007 | +0.009 | +0.006 | +0.004 | -0.049 | -0.003 | +0.115 | +0.007 |
| bike | +0.996 | +0.271 | +0.506 | +0.098 | +0.100 | +0.014 | +0.007 | -0.087 | +0.029 | +0.121 | +0.098 |
| shrub | +1.009 | +0.445 | +0.357 | +0.026 | +0.026 | +0.019 | +0.027 | -0.045 | +0.026 | +0.096 | +0.024 |


## P4 dark end (LOM): mean pseudo-target value over views at input level l, in 8-bit levels

detach: seed-0 curves through the official LUT (level l reads curve[max(l-1, 0)]).

| scene | level | detach | quadratic-cdf | quadratic-cdf-unshifted | quadratic-cdf-priorF | quadratic-cdf-priorL |
|---|---|---|---|---|---|---|
| buu | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| buu | 1 | 0.00 | 0.04 | 0.11 | 45.75 | 47.43 |
| buu | 2 | 44.63 | 0.14 | 0.71 | 54.04 | 56.30 |
| buu | 4 | 58.89 | 3.92 | 9.16 | 66.69 | 66.19 |
| buu | 8 | 78.33 | 23.44 | 26.13 | 84.21 | 84.29 |
| buu | 16 | 119.39 | 76.64 | 88.91 | 119.09 | 118.88 |
| chair | 0 | 68.09 | 0.00 | 0.00 | 0.00 | 0.00 |
| chair | 1 | 68.09 | 0.02 | 0.69 | 63.98 | 64.32 |
| chair | 2 | 76.21 | 0.41 | 8.14 | 72.32 | 73.26 |
| chair | 4 | 97.06 | 25.38 | 43.10 | 95.46 | 94.75 |
| chair | 8 | 124.77 | 67.22 | 73.52 | 121.37 | 121.29 |
| chair | 16 | 172.53 | 145.58 | 160.85 | 166.70 | 166.41 |
| sofa | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sofa | 1 | 0.00 | 0.04 | 0.05 | 55.52 | 55.05 |
| sofa | 2 | 54.53 | 0.07 | 0.28 | 63.70 | 64.50 |
| sofa | 4 | 68.62 | 1.03 | 3.14 | 73.95 | 73.43 |
| sofa | 8 | 84.43 | 20.16 | 35.60 | 89.26 | 88.89 |
| sofa | 16 | 153.74 | 135.74 | 141.43 | 155.39 | 156.03 |
| bike | 0 | 83.74 | 0.00 | 0.00 | 0.00 | 0.00 |
| bike | 1 | 83.74 | 0.10 | 2.17 | 78.01 | 77.38 |
| bike | 2 | 90.60 | 1.53 | 8.81 | 86.95 | 88.34 |
| bike | 4 | 108.88 | 24.73 | 53.90 | 106.70 | 105.03 |
| bike | 8 | 181.06 | 159.84 | 189.29 | 178.36 | 178.52 |
| bike | 16 | 228.40 | 243.90 | 245.52 | 225.69 | 225.63 |
| shrub | 0 | 85.63 | 0.00 | 0.00 | 0.00 | 0.00 |
| shrub | 1 | 85.63 | 6.43 | 21.73 | 80.41 | 79.51 |
| shrub | 2 | 100.04 | 22.61 | 44.14 | 96.00 | 97.84 |
| shrub | 4 | 129.40 | 67.80 | 90.19 | 128.06 | 126.68 |
| shrub | 8 | 173.24 | 145.06 | 160.60 | 170.23 | 170.14 |
| shrub | 16 | 220.75 | 230.01 | 234.98 | 218.11 | 218.36 |


Dark-end slope (mean output gain per input step over levels 1-8; how much photo noise is amplified) and the fraction of training pixels at levels 0-2.

| scene | set | pixels at 0-2 | detach | quadratic-cdf | quadratic-cdf-unshifted | quadratic-cdf-priorF | quadratic-cdf-priorL |
|---|---|---|---|---|---|---|---|
| buu | lom | 0.74% | 11.19 | 3.34 | 3.72 | 5.49 | 5.27 |
| buu | vexp | 1.53% | 8.27 | 3.94 | 4.95 | 6.42 | 5.91 |
| chair | lom | 2.17% | 8.10 | 9.60 | 10.41 | 8.20 | 8.14 |
| chair | vexp | 6.10% | 19.37 | 10.51 | 11.90 | 9.65 | 9.17 |
| sofa | lom | 0.28% | 12.06 | 2.87 | 5.08 | 4.82 | 4.83 |
| sofa | vexp | 1.40% | 13.81 | 5.96 | 7.62 | 7.16 | 6.32 |
| bike | lom | 3.60% | 13.90 | 22.82 | 26.73 | 14.33 | 14.45 |
| bike | vexp | 6.56% | 12.16 | 21.84 | 24.32 | 14.79 | 14.28 |
| shrub | lom | 22.52% | 12.51 | 19.80 | 19.84 | 12.83 | 12.95 |
| shrub | vexp | 24.40% | 11.39 | 19.81 | 20.07 | 13.65 | 13.06 |


Distance to detach's curves as applied to input levels 0-255 (mean over views, weighted by how often each level occurs in the training photos), x255.

| scene | set | quadratic-cdf | quadratic-cdf-unshifted | quadratic-cdf-priorF | quadratic-cdf-priorL |
|---|---|---|---|---|---|
| buu | lom | 38.41 | 34.31 | 14.28 | 11.92 |
| buu | vexp | 51.57 | 51.27 | 34.78 | 32.25 |
| chair | lom | 41.38 | 35.15 | 5.78 | 5.22 |
| chair | vexp | 65.50 | 61.40 | 31.19 | 27.29 |
| sofa | lom | 32.23 | 26.22 | 6.34 | 5.26 |
| sofa | vexp | 55.47 | 53.65 | 30.17 | 24.57 |
| bike | lom | 46.19 | 28.84 | 5.42 | 4.79 |
| bike | vexp | 72.82 | 63.58 | 33.10 | 28.88 |
| shrub | lom | 43.41 | 32.61 | 11.40 | 10.27 |
| shrub | vexp | 63.83 | 55.71 | 33.60 | 29.90 |


## P5 pseudo-target quality against high/ (DIAGNOSTIC ONLY: never chooses a setting)

detach: mean over views per run, then mean ± sd over seeds 0-2 (`rescore/pseudo_targets.csv`).

| scene | set | config | PSNR | detach PSNR | bm PSNR | detach bm | bm gap | gap / sd |
|---|---|---|---|---|---|---|---|---|
| buu | lom | quadratic-cdf | 16.11 | 22.41 ± 0.10 | 15.94 | 26.79 ± 0.02 | -10.85 | -690.3 |
| buu | lom | quadratic-cdf-unshifted | 16.44 | 22.41 ± 0.10 | 17.11 | 26.79 ± 0.02 | -9.68 | -615.9 |
| buu | lom | quadratic-cdf-priorF | 20.78 | 22.41 ± 0.10 | 26.72 | 26.79 ± 0.02 | -0.07 | -4.7 |
| buu | lom | quadratic-cdf-priorL | 20.72 | 22.41 ± 0.10 | 26.64 | 26.79 ± 0.02 | -0.15 | -9.6 |
| buu | vexp | quadratic-cdf | 16.07 | 19.81 ± 0.02 | 15.92 | 27.04 ± 0.07 | -11.12 | -152.9 |
| buu | vexp | quadratic-cdf-unshifted | 16.32 | 19.81 ± 0.02 | 17.03 | 27.04 ± 0.07 | -10.01 | -137.7 |
| buu | vexp | quadratic-cdf-priorF | 20.85 | 19.81 ± 0.02 | 26.54 | 27.04 ± 0.07 | -0.50 | -6.9 |
| buu | vexp | quadratic-cdf-priorL | 20.90 | 19.81 ± 0.02 | 26.31 | 27.04 ± 0.07 | -0.73 | -10.1 |
| chair | lom | quadratic-cdf | 16.78 | 19.53 ± 0.01 | 16.41 | 23.43 ± 0.00 | -7.02 | -2697.7 |
| chair | lom | quadratic-cdf-unshifted | 17.20 | 19.53 ± 0.01 | 17.42 | 23.43 ± 0.00 | -6.01 | -2309.6 |
| chair | lom | quadratic-cdf-priorF | 20.35 | 19.53 ± 0.01 | 23.48 | 23.43 ± 0.00 | +0.06 | +21.8 |
| chair | lom | quadratic-cdf-priorL | 20.36 | 19.53 ± 0.01 | 23.50 | 23.43 ± 0.00 | +0.07 | +28.6 |
| chair | vexp | quadratic-cdf | 16.72 | 16.77 ± 0.18 | 16.41 | 21.49 ± 0.16 | -5.08 | -31.3 |
| chair | vexp | quadratic-cdf-unshifted | 17.07 | 16.77 ± 0.18 | 17.37 | 21.49 ± 0.16 | -4.13 | -25.4 |
| chair | vexp | quadratic-cdf-priorF | 20.61 | 16.77 ± 0.18 | 23.57 | 21.49 ± 0.16 | +2.07 | +12.7 |
| chair | vexp | quadratic-cdf-priorL | 20.20 | 16.77 ± 0.18 | 23.23 | 21.49 ± 0.16 | +1.74 | +10.7 |
| sofa | lom | quadratic-cdf | 16.69 | 25.75 ± 0.02 | 15.91 | 28.74 ± 0.00 | -12.82 | -45689.0 |
| sofa | lom | quadratic-cdf-unshifted | 18.05 | 25.75 ± 0.02 | 17.71 | 28.74 ± 0.00 | -11.02 | -39280.8 |
| sofa | lom | quadratic-cdf-priorF | 24.79 | 25.75 ± 0.02 | 28.54 | 28.74 ± 0.00 | -0.19 | -694.5 |
| sofa | lom | quadratic-cdf-priorL | 24.79 | 25.75 ± 0.02 | 28.48 | 28.74 ± 0.00 | -0.25 | -894.7 |
| sofa | vexp | quadratic-cdf | 16.64 | 19.73 ± 0.01 | 15.90 | 26.73 ± 0.10 | -10.82 | -110.6 |
| sofa | vexp | quadratic-cdf-unshifted | 17.81 | 19.73 ± 0.01 | 17.55 | 26.73 ± 0.10 | -9.18 | -93.8 |
| sofa | vexp | quadratic-cdf-priorF | 24.64 | 19.73 ± 0.01 | 27.83 | 26.73 ± 0.10 | +1.10 | +11.3 |
| sofa | vexp | quadratic-cdf-priorL | 24.30 | 19.73 ± 0.01 | 27.76 | 26.73 ± 0.10 | +1.03 | +10.5 |
| bike | lom | quadratic-cdf | 14.84 | 16.71 ± 0.12 | 14.08 | 22.50 ± 0.01 | -8.42 | -912.3 |
| bike | lom | quadratic-cdf-unshifted | 14.99 | 16.71 ± 0.12 | 17.04 | 22.50 ± 0.01 | -5.46 | -591.9 |
| bike | lom | quadratic-cdf-priorF | 17.13 | 16.71 ± 0.12 | 22.51 | 22.50 ± 0.01 | +0.00 | +0.2 |
| bike | lom | quadratic-cdf-priorL | 17.12 | 16.71 ± 0.12 | 22.47 | 22.50 ± 0.01 | -0.04 | -4.1 |
| bike | vexp | quadratic-cdf | 14.75 | 15.08 ± 0.30 | 14.02 | 21.61 ± 0.11 | -7.60 | -67.5 |
| bike | vexp | quadratic-cdf-unshifted | 14.77 | 15.08 ± 0.30 | 16.86 | 21.61 ± 0.11 | -4.76 | -42.3 |
| bike | vexp | quadratic-cdf-priorF | 17.21 | 15.08 ± 0.30 | 22.22 | 21.61 ± 0.11 | +0.60 | +5.4 |
| bike | vexp | quadratic-cdf-priorL | 17.26 | 15.08 ± 0.30 | 22.28 | 21.61 ± 0.11 | +0.67 | +5.9 |
| shrub | lom | quadratic-cdf | 14.47 | 11.68 ± 0.43 | 19.04 | 17.52 ± 0.17 | +1.53 | +9.2 |
| shrub | lom | quadratic-cdf-unshifted | 13.21 | 11.68 ± 0.43 | 19.53 | 17.52 ± 0.17 | +2.01 | +12.1 |
| shrub | lom | quadratic-cdf-priorF | 11.87 | 11.68 ± 0.43 | 17.74 | 17.52 ± 0.17 | +0.22 | +1.3 |
| shrub | lom | quadratic-cdf-priorL | 11.87 | 11.68 ± 0.43 | 17.73 | 17.52 ± 0.17 | +0.21 | +1.3 |
| shrub | vexp | quadratic-cdf | 14.42 | 9.87 ± 0.14 | 18.85 | 16.22 ± 0.06 | +2.63 | +44.3 |
| shrub | vexp | quadratic-cdf-unshifted | 13.10 | 9.87 ± 0.14 | 19.30 | 16.22 ± 0.06 | +3.08 | +52.0 |
| shrub | vexp | quadratic-cdf-priorF | 12.00 | 9.87 ± 0.14 | 17.83 | 16.22 ± 0.06 | +1.61 | +27.2 |
| shrub | vexp | quadratic-cdf-priorL | 11.92 | 9.87 ± 0.14 | 17.67 | 16.22 ± 0.06 | +1.45 | +24.5 |


## Checks against M1

| scene | init_vs_oracle_json_s_init | oracle_vs_oracle_json_s_oracle |
|---|---|---|
| buu | 1.84e-07 | 0.00e+00 |
| chair | 3.39e-07 | 0.00e+00 |
| sofa | 2.02e-07 | 0.00e+00 |
| bike | 2.59e-07 | 0.00e+00 |
| shrub | 1.97e-07 | 0.00e+00 |


Tolerances as in `vexp_analyze.py`: init 1e-6, oracle 2e-3 stops.


Figures: `figures/curves_<scene>.png` (default configuration), `figures/exposure_vs_u.png`.
