# Rung 2 preview (M2 step 6)

Curve loss only, no 3DGS: a replay of the trainer's F and s_k updates on the training photos. 10000 steps, lr 0.001 (F) and 0.001 (s), K = 32, smoothness 0.1, seed 0. Default configuration `quadratic-cdf` (one-bin-shifted target, decided 2026-11-11); `quadratic-cdf-unshifted` is the alignment check. Generated 2026-11-11 14:53 in 920 s by `rung2_preview.py` (015fa3ab9db6765b). Every number here is *measured* on the replay; how a 3DGS run follows it is *inferred*.

## Decision rules (training photos only; Bart confirms)

- **Knot spacing:** pooled P1 residual quadratic 7.797/255, usedrange 7.140/255, reduction 8.4% (rule: switch at >= 30%) -> **quadratic**.
- **Learning rate:** largest gap of the replay objective above the full-batch optimum 2.64% (rule: try 3e-3 above 5%) -> **keep 1e-3**.
- **F init:** steps the identity init needs to come within 0.01 of the cdf init's final F: quadratic-identity: never, never, never, never, never, never, never, never, never, never; usedrange-identity: never, never, never, never, never, never, never, never, never, never (in the order buu/lom, buu/vexp, chair/lom, chair/vexp, sofa/lom, sofa/vexp, bike/lom, bike/vexp, shrub/lom, shrub/vexp).
- **P5 flag (diagnostic only):** LOM scenes where `quadratic-cdf` is below `detach` by more than detach's brightness-matched seed sd: buu, chair, sofa, bike -> **stop and ask Bart**.

Exploratory configurations (not in the plan, added for Bart's decision): quadratic-cdf-priorF, quadratic-cdf-priorL. They add the official shape-prior term 0.5 [mse(p, X) + 0.01 mse(p, T_k)] with one learned global gamma, p = clamp(x, 1e-3, 1)^gamma, on the shared F (priorF) or on each view's composed curve (priorL, the official form). They take no part in the rules above.

## P1 fit: weighted residual x255 against the targets

| scene | set | oracle | quadratic-cdf | usedrange-cdf | quadratic-identity | usedrange-identity | quadratic-cdf-priorF | quadratic-cdf-priorL | usedrange y_max | gamma priorF | gamma priorL | detach gamma (mean) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buu | lom | 11.665 | 13.163 | 12.745 | 58.305 | 74.271 | 19.696 | 18.545 | 0.236 | 0.211 | 0.212 | 0.195 |
| buu | vexp | 11.676 | 13.434 | 13.005 | 61.164 | 76.434 | 19.888 | 19.984 | 0.236 | 0.210 | 0.224 | 0.378 |
| chair | lom | 5.832 | 8.307 | 5.969 | 79.003 | 81.966 | 12.275 | 12.305 | 0.122 | 0.190 | 0.190 | 0.095 |
| chair | vexp | 7.345 | 9.339 | 7.539 | 82.419 | 85.285 | 13.161 | 13.576 | 0.123 | 0.190 | 0.194 | -0.065 |
| sofa | lom | 7.841 | 8.716 | 8.606 | 69.059 | 77.500 | 16.070 | 15.735 | 0.165 | 0.201 | 0.202 | 0.149 |
| sofa | vexp | 8.008 | 8.933 | 8.849 | 72.372 | 80.297 | 16.043 | 16.331 | 0.166 | 0.201 | 0.210 | 0.073 |
| bike | lom | 3.998 | 4.602 | 3.837 | 97.811 | 104.468 | 23.004 | 22.883 | 0.151 | 0.183 | 0.183 | 0.065 |
| bike | vexp | 6.512 | 7.079 | 6.579 | 99.248 | 105.195 | 22.508 | 21.791 | 0.152 | 0.182 | 0.184 | -0.145 |
| shrub | lom | 1.888 | 1.981 | 1.888 | 97.527 | 107.587 | 22.315 | 22.336 | 0.188 | 0.184 | 0.184 | 0.068 |
| shrub | vexp | 2.492 | 2.421 | 2.384 | 99.938 | 109.095 | 21.592 | 21.312 | 0.188 | 0.184 | 0.186 | -0.151 |


Default configuration: objective against the full-batch optimum, and convergence.

| scene | set | objective | optimum | gap % | step within 5% | init-to-final max dF | replay s |
|---|---|---|---|---|---|---|---|
| buu | lom | 2.079e-04 | 2.054e-04 | +1.22 | 0 | 0.012 | 7.4 |
| buu | vexp | 2.195e-04 | 2.161e-04 | +1.58 | 100 | 0.011 | 8.7 |
| chair | lom | 6.407e-05 | 6.273e-05 | +2.14 | 400 | 0.014 | 7.8 |
| chair | vexp | 7.938e-05 | 7.734e-05 | +2.64 | 600 | 0.015 | 8.7 |
| sofa | lom | 7.807e-05 | 7.751e-05 | +0.71 | 100 | 0.009 | 8.2 |
| sofa | vexp | 9.117e-05 | 8.983e-05 | +1.50 | 300 | 0.012 | 8.1 |
| bike | lom | 5.066e-05 | 5.042e-05 | +0.49 | 200 | 0.025 | 7.4 |
| bike | vexp | 6.004e-05 | 5.970e-05 | +0.56 | 400 | 0.023 | 7.2 |
| shrub | lom | 2.605e-05 | 2.598e-05 | +0.27 | 100 | 0.007 | 8.1 |
| shrub | vexp | 2.865e-05 | 2.860e-05 | +0.17 | 100 | 0.007 | 7.6 |


## P2 exposure on the varying set: s_k against u_k (ideal slope -1, RMSE 0)

| scene | method | slope | r | rmse | rmse paired | move from init |
|---|---|---|---|---|---|---|
| buu | init | -1.037 | -0.952 | 0.193 | 0.006 | 0.000 |
| buu | oracle | -1.083 | -0.968 | 0.170 | 0.039 | 0.044 |
| buu | quadratic-cdf | -1.027 | -0.956 | 0.182 | 0.006 | 0.013 |
| buu | usedrange-cdf | -1.025 | -0.959 | 0.175 | 0.007 | 0.020 |
| buu | quadratic-cdf-unshifted | -1.064 | -0.958 | 0.187 | 0.022 | 0.018 |
| buu | quadratic-cdf-priorF | -1.038 | -0.952 | 0.194 | 0.008 | 0.007 |
| buu | quadratic-cdf-priorL | -0.903 | -0.958 | 0.166 | 0.070 | 0.086 |
| chair | init | -1.042 | -0.991 | 0.083 | 0.013 | 0.000 |
| chair | oracle | -1.093 | -0.995 | 0.083 | 0.043 | 0.044 |
| chair | quadratic-cdf | -1.031 | -0.993 | 0.072 | 0.011 | 0.023 |
| chair | usedrange-cdf | -1.025 | -0.994 | 0.065 | 0.010 | 0.028 |
| chair | quadratic-cdf-unshifted | -1.096 | -0.994 | 0.089 | 0.043 | 0.041 |
| chair | quadratic-cdf-priorF | -1.031 | -0.991 | 0.080 | 0.015 | 0.021 |
| chair | quadratic-cdf-priorL | -0.969 | -0.992 | 0.072 | 0.036 | 0.046 |
| sofa | init | -1.022 | -0.991 | 0.079 | 0.004 | 0.000 |
| sofa | oracle | -1.069 | -0.996 | 0.066 | 0.038 | 0.041 |
| sofa | quadratic-cdf | -1.009 | -0.991 | 0.078 | 0.007 | 0.010 |
| sofa | usedrange-cdf | -1.007 | -0.991 | 0.077 | 0.007 | 0.011 |
| sofa | quadratic-cdf-unshifted | -1.064 | -0.991 | 0.089 | 0.029 | 0.025 |
| sofa | quadratic-cdf-priorF | -1.026 | -0.991 | 0.079 | 0.007 | 0.005 |
| sofa | quadratic-cdf-priorL | -0.898 | -0.993 | 0.084 | 0.067 | 0.075 |
| bike | init | -0.952 | -0.990 | 0.083 | 0.019 | 0.000 |
| bike | oracle | -1.043 | -0.989 | 0.093 | 0.056 | 0.058 |
| bike | quadratic-cdf | -0.894 | -0.989 | 0.099 | 0.047 | 0.036 |
| bike | usedrange-cdf | -0.893 | -0.989 | 0.100 | 0.047 | 0.036 |
| bike | quadratic-cdf-unshifted | -1.031 | -0.990 | 0.089 | 0.048 | 0.049 |
| bike | quadratic-cdf-priorF | -0.960 | -0.989 | 0.086 | 0.031 | 0.018 |
| bike | quadratic-cdf-priorL | -0.871 | -0.991 | 0.102 | 0.054 | 0.049 |
| shrub | init | -0.984 | -0.983 | 0.105 | 0.014 | 0.000 |
| shrub | oracle | -1.108 | -0.980 | 0.145 | 0.082 | 0.077 |
| shrub | quadratic-cdf | -0.952 | -0.981 | 0.112 | 0.018 | 0.021 |
| shrub | usedrange-cdf | -0.953 | -0.981 | 0.111 | 0.018 | 0.020 |
| shrub | quadratic-cdf-unshifted | -1.075 | -0.982 | 0.126 | 0.062 | 0.055 |
| shrub | quadratic-cdf-priorF | -0.995 | -0.984 | 0.105 | 0.018 | 0.008 |
| shrub | quadratic-cdf-priorL | -0.910 | -0.984 | 0.109 | 0.042 | 0.045 |


## P3 paired pseudo-target leak (slope of log2(mean, varying / LOM) on u_k; 0 = compensated, 1 = passes through)

M1 columns: mean over 3 seeds from `results/vexp/summary/T4_exposure.csv`.

| scene | input | official_M1 | detach_M1 | quadratic-cdf | usedrange-cdf | quadratic-identity | usedrange-identity | quadratic-cdf-unshifted | quadratic-cdf-priorF | quadratic-cdf-priorL | unpaired_quadratic-cdf |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buu | +1.008 | +0.564 | +0.653 | +0.013 | +0.013 | +0.010 | +0.005 | -0.022 | +0.003 | +0.129 | +0.012 |
| chair | +1.015 | +0.345 | +0.496 | +0.022 | +0.025 | +0.008 | +0.003 | -0.047 | +0.012 | +0.085 | +0.026 |
| sofa | +1.002 | +0.399 | +0.547 | +0.008 | +0.010 | +0.009 | +0.003 | -0.047 | -0.001 | +0.129 | +0.006 |
| bike | +0.996 | +0.271 | +0.506 | +0.095 | +0.100 | +0.008 | +0.003 | -0.087 | -0.024 | +0.119 | +0.095 |
| shrub | +1.009 | +0.445 | +0.357 | +0.026 | +0.027 | +0.013 | +0.008 | -0.043 | -0.002 | +0.058 | +0.024 |


## P4 dark end (LOM): mean pseudo-target value over views at input level l, in 8-bit levels

detach: seed-0 curves through the official LUT (level l reads curve[max(l-1, 0)]).

| scene | level | detach | quadratic-cdf | quadratic-cdf-unshifted | quadratic-cdf-priorF | quadratic-cdf-priorL |
|---|---|---|---|---|---|---|
| buu | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| buu | 1 | 0.00 | 0.05 | 0.12 | 0.21 | 0.20 |
| buu | 2 | 44.63 | 0.18 | 0.82 | 0.70 | 0.64 |
| buu | 4 | 58.89 | 4.23 | 8.97 | 9.26 | 8.77 |
| buu | 8 | 78.33 | 22.98 | 25.79 | 37.52 | 37.24 |
| buu | 16 | 119.39 | 75.80 | 87.80 | 88.24 | 86.62 |
| chair | 0 | 68.09 | 0.00 | 0.00 | 0.00 | 0.00 |
| chair | 1 | 68.09 | 0.06 | 0.67 | 0.15 | 0.15 |
| chair | 2 | 76.21 | 1.08 | 8.39 | 2.30 | 2.31 |
| chair | 4 | 97.06 | 25.89 | 42.95 | 31.73 | 31.75 |
| chair | 8 | 124.77 | 67.66 | 73.88 | 74.93 | 75.03 |
| chair | 16 | 172.53 | 147.67 | 162.24 | 134.22 | 134.26 |
| sofa | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sofa | 1 | 0.00 | 0.04 | 0.05 | 0.13 | 0.13 |
| sofa | 2 | 54.53 | 0.07 | 0.25 | 0.21 | 0.21 |
| sofa | 4 | 68.62 | 1.00 | 2.83 | 2.45 | 2.37 |
| sofa | 8 | 84.43 | 20.99 | 36.21 | 27.35 | 27.12 |
| sofa | 16 | 153.74 | 135.44 | 141.59 | 130.94 | 131.10 |
| bike | 0 | 83.74 | 0.00 | 0.00 | 0.00 | 0.00 |
| bike | 1 | 83.74 | 0.08 | 1.28 | 0.11 | 0.10 |
| bike | 2 | 90.60 | 1.57 | 8.69 | 1.87 | 1.86 |
| bike | 4 | 108.88 | 23.99 | 53.41 | 20.87 | 20.52 |
| bike | 8 | 181.06 | 160.04 | 189.43 | 131.50 | 131.40 |
| bike | 16 | 228.40 | 244.84 | 246.03 | 211.38 | 211.34 |
| shrub | 0 | 85.63 | 0.00 | 0.00 | 0.00 | 0.00 |
| shrub | 1 | 85.63 | 5.61 | 21.90 | 4.91 | 4.86 |
| shrub | 2 | 100.04 | 22.46 | 44.00 | 20.73 | 20.74 |
| shrub | 4 | 129.40 | 67.86 | 90.23 | 61.46 | 61.42 |
| shrub | 8 | 173.24 | 145.08 | 160.60 | 123.10 | 123.04 |
| shrub | 16 | 220.75 | 230.02 | 234.98 | 193.16 | 193.23 |


Dark-end slope (mean output gain per input step over levels 1-8; how much photo noise is amplified) and the fraction of training pixels at levels 0-2.

| scene | set | pixels at 0-2 | detach | quadratic-cdf | quadratic-cdf-unshifted | quadratic-cdf-priorF | quadratic-cdf-priorL |
|---|---|---|---|---|---|---|---|
| buu | lom | 0.74% | 11.19 | 3.28 | 3.67 | 5.33 | 5.29 |
| buu | vexp | 1.53% | 8.27 | 3.90 | 4.85 | 5.66 | 5.20 |
| chair | lom | 2.17% | 8.10 | 9.66 | 10.46 | 10.68 | 10.70 |
| chair | vexp | 6.10% | 19.37 | 10.48 | 11.84 | 10.82 | 10.70 |
| sofa | lom | 0.28% | 12.06 | 2.99 | 5.17 | 3.89 | 3.86 |
| sofa | vexp | 1.40% | 13.81 | 5.96 | 7.61 | 6.49 | 5.94 |
| bike | lom | 3.60% | 13.90 | 22.85 | 26.88 | 18.77 | 18.76 |
| bike | vexp | 6.56% | 12.16 | 21.83 | 24.30 | 18.29 | 18.34 |
| shrub | lom | 22.52% | 12.51 | 19.92 | 19.81 | 16.88 | 16.88 |
| shrub | vexp | 24.40% | 11.39 | 19.77 | 19.80 | 16.57 | 16.61 |


Distance to detach's curves as applied to input levels 0-255 (mean over views, weighted by how often each level occurs in the training photos), x255.

| scene | set | quadratic-cdf | quadratic-cdf-unshifted | quadratic-cdf-priorF | quadratic-cdf-priorL |
|---|---|---|---|---|---|
| buu | lom | 38.55 | 34.45 | 29.79 | 28.66 |
| buu | vexp | 51.67 | 51.20 | 43.77 | 40.65 |
| chair | lom | 40.96 | 35.08 | 40.82 | 40.76 |
| chair | vexp | 65.31 | 61.23 | 61.66 | 61.27 |
| sofa | lom | 32.21 | 26.28 | 31.63 | 31.51 |
| sofa | vexp | 55.45 | 53.52 | 51.82 | 49.44 |
| bike | lom | 46.21 | 28.88 | 61.34 | 61.43 |
| bike | vexp | 72.85 | 63.58 | 75.20 | 74.89 |
| shrub | lom | 43.47 | 32.61 | 55.24 | 55.25 |
| shrub | vexp | 63.82 | 55.54 | 69.22 | 68.76 |


## P5 pseudo-target quality against high/ (DIAGNOSTIC ONLY: never chooses a setting)

detach: mean over views per run, then mean ± sd over seeds 0-2 (`rescore/pseudo_targets.csv`).

| scene | set | config | PSNR | detach PSNR | bm PSNR | detach bm | bm gap | gap / sd |
|---|---|---|---|---|---|---|---|---|
| buu | lom | quadratic-cdf | 16.12 | 22.41 ± 0.10 | 15.95 | 26.79 ± 0.02 | -10.85 | -690.2 |
| buu | lom | quadratic-cdf-unshifted | 16.49 | 22.41 ± 0.10 | 17.13 | 26.79 ± 0.02 | -9.67 | -615.0 |
| buu | lom | quadratic-cdf-priorF | 19.27 | 22.41 ± 0.10 | 19.11 | 26.79 ± 0.02 | -7.68 | -488.9 |
| buu | lom | quadratic-cdf-priorL | 19.25 | 22.41 ± 0.10 | 18.98 | 26.79 ± 0.02 | -7.82 | -497.3 |
| buu | vexp | quadratic-cdf | 16.09 | 19.81 ± 0.02 | 15.93 | 27.04 ± 0.07 | -11.11 | -152.8 |
| buu | vexp | quadratic-cdf-unshifted | 16.40 | 19.81 ± 0.02 | 17.06 | 27.04 ± 0.07 | -9.98 | -137.3 |
| buu | vexp | quadratic-cdf-priorF | 19.11 | 19.81 ± 0.02 | 18.97 | 27.04 ± 0.07 | -8.07 | -111.1 |
| buu | vexp | quadratic-cdf-priorL | 18.78 | 19.81 ± 0.02 | 18.67 | 27.04 ± 0.07 | -8.37 | -115.1 |
| chair | lom | quadratic-cdf | 16.83 | 19.53 ± 0.01 | 16.43 | 23.43 ± 0.00 | -7.00 | -2691.1 |
| chair | lom | quadratic-cdf-unshifted | 17.21 | 19.53 ± 0.01 | 17.42 | 23.43 ± 0.00 | -6.01 | -2310.1 |
| chair | lom | quadratic-cdf-priorF | 17.94 | 19.53 ± 0.01 | 17.65 | 23.43 ± 0.00 | -5.78 | -2221.8 |
| chair | lom | quadratic-cdf-priorL | 17.96 | 19.53 ± 0.01 | 17.66 | 23.43 ± 0.00 | -5.77 | -2217.8 |
| chair | vexp | quadratic-cdf | 16.77 | 16.77 ± 0.18 | 16.42 | 21.49 ± 0.16 | -5.08 | -31.2 |
| chair | vexp | quadratic-cdf-unshifted | 17.14 | 16.77 ± 0.18 | 17.42 | 21.49 ± 0.16 | -4.07 | -25.0 |
| chair | vexp | quadratic-cdf-priorF | 17.87 | 16.77 ± 0.18 | 17.62 | 21.49 ± 0.16 | -3.87 | -23.8 |
| chair | vexp | quadratic-cdf-priorL | 17.68 | 16.77 ± 0.18 | 17.55 | 21.49 ± 0.16 | -3.95 | -24.3 |
| sofa | lom | quadratic-cdf | 16.72 | 25.75 ± 0.02 | 15.92 | 28.74 ± 0.00 | -12.81 | -45653.4 |
| sofa | lom | quadratic-cdf-unshifted | 18.08 | 25.75 ± 0.02 | 17.72 | 28.74 ± 0.00 | -11.02 | -39258.8 |
| sofa | lom | quadratic-cdf-priorF | 17.92 | 25.75 ± 0.02 | 17.40 | 28.74 ± 0.00 | -11.33 | -40382.2 |
| sofa | lom | quadratic-cdf-priorL | 17.91 | 25.75 ± 0.02 | 17.38 | 28.74 ± 0.00 | -11.36 | -40465.7 |
| sofa | vexp | quadratic-cdf | 16.68 | 19.73 ± 0.01 | 15.91 | 26.73 ± 0.10 | -10.82 | -110.5 |
| sofa | vexp | quadratic-cdf-unshifted | 17.86 | 19.73 ± 0.01 | 17.56 | 26.73 ± 0.10 | -9.17 | -93.6 |
| sofa | vexp | quadratic-cdf-priorF | 17.79 | 19.73 ± 0.01 | 17.24 | 26.73 ± 0.10 | -9.49 | -96.9 |
| sofa | vexp | quadratic-cdf-priorL | 17.63 | 19.73 ± 0.01 | 17.10 | 26.73 ± 0.10 | -9.63 | -98.4 |
| bike | lom | quadratic-cdf | 14.81 | 16.71 ± 0.12 | 14.05 | 22.50 ± 0.01 | -8.45 | -915.4 |
| bike | lom | quadratic-cdf-unshifted | 14.97 | 16.71 ± 0.12 | 17.02 | 22.50 ± 0.01 | -5.49 | -594.4 |
| bike | lom | quadratic-cdf-priorF | 15.11 | 16.71 ± 0.12 | 14.38 | 22.50 ± 0.01 | -8.12 | -879.4 |
| bike | lom | quadratic-cdf-priorL | 15.09 | 16.71 ± 0.12 | 14.36 | 22.50 ± 0.01 | -8.14 | -882.0 |
| bike | vexp | quadratic-cdf | 14.75 | 15.08 ± 0.30 | 14.02 | 21.61 ± 0.11 | -7.60 | -67.5 |
| bike | vexp | quadratic-cdf-unshifted | 14.76 | 15.08 ± 0.30 | 16.85 | 21.61 ± 0.11 | -4.76 | -42.3 |
| bike | vexp | quadratic-cdf-priorF | 15.11 | 15.08 ± 0.30 | 14.39 | 21.61 ± 0.11 | -7.23 | -64.2 |
| bike | vexp | quadratic-cdf-priorL | 15.10 | 15.08 ± 0.30 | 14.41 | 21.61 ± 0.11 | -7.21 | -64.0 |
| shrub | lom | quadratic-cdf | 14.47 | 11.68 ± 0.43 | 19.02 | 17.52 ± 0.17 | +1.50 | +9.1 |
| shrub | lom | quadratic-cdf-unshifted | 13.21 | 11.68 ± 0.43 | 19.53 | 17.52 ± 0.17 | +2.02 | +12.1 |
| shrub | lom | quadratic-cdf-priorF | 18.10 | 11.68 ± 0.43 | 19.50 | 17.52 ± 0.17 | +1.98 | +12.0 |
| shrub | lom | quadratic-cdf-priorL | 18.10 | 11.68 ± 0.43 | 19.50 | 17.52 ± 0.17 | +1.98 | +12.0 |
| shrub | vexp | quadratic-cdf | 14.42 | 9.87 ± 0.14 | 18.85 | 16.22 ± 0.06 | +2.63 | +44.4 |
| shrub | vexp | quadratic-cdf-unshifted | 13.11 | 9.87 ± 0.14 | 19.33 | 16.22 ± 0.06 | +3.11 | +52.5 |
| shrub | vexp | quadratic-cdf-priorF | 17.89 | 9.87 ± 0.14 | 19.34 | 16.22 ± 0.06 | +3.12 | +52.7 |
| shrub | vexp | quadratic-cdf-priorL | 17.80 | 9.87 ± 0.14 | 19.34 | 16.22 ± 0.06 | +3.12 | +52.7 |


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
