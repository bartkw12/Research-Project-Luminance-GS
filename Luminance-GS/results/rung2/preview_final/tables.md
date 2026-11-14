# Rung 2 preview (M2 step 6), final configuration

Curve loss only, no 3DGS: a replay of the trainer's F and s_k updates on the training photos. 10000 steps, lr 0.01 (F and gamma) and 0.001 (s), K = 32, smoothness 0.1, seed 0. Default configuration `quadratic-cdf-priorF` (decided 2026-11-13: one-bin-shifted target, and the official shape-prior term 0.5 [mse(p, F) + 0.01 mse(p, T_k)] with one learned global gamma, p = clamp(x, 1e-3, 1)^gamma, on the shared F). `quadratic-cdf-priorF-unshifted` is the alignment check, `quadratic-cdf-noprior` the plan's first draft (an M5 ablation), `quadratic-cdf-priorL` the prior in the official form (rejected). Generated 2026-11-13 17:41 in 1286 s by `rung2_preview.py` (b2db513d3c0f95fb). Every number here is *measured* on the replay; how a 3DGS run follows it is *inferred*.

## Decision rules (training photos only)

- **Learning rate (F and gamma), plan step 6 with the prior on:** largest gap of the replay objective above the full-batch optimum, over scenes and sets, per rate: 0.001: +201.54%; 0.003: +7.00%; 0.01: +0.82% (rule: the first rate of the ladder within 5%) -> **0.01**.
- **Knot spacing:** pooled P1 residual quadratic 44.389/255, usedrange 44.513/255, reduction -0.3% (rule: switch at >= 30%) -> **quadratic**.
- **F init:** steps the identity init needs to come within 0.01 of the cdf init's final F: never, never, never, never, never, never, never, never, never, never (in the order buu/lom, buu/vexp, chair/lom, chair/vexp, sofa/lom, sofa/vexp, bike/lom, bike/vexp, shrub/lom, shrub/vexp).
- **P5 flag (diagnostic only):** LOM scenes where `quadratic-cdf-priorF` is below `detach` by more than detach's brightness-matched seed sd: buu, sofa -> **stop and ask Bart**.

## P1 learning-rate ladder (default configuration)

Objective = the curve loss with the prior over all views at the late weight. Optimum = the lower of 10000 cold-start full-batch Adam steps (lr 0.03) and a 3000-step polish (lr 0.001) of the lowest-ending replay (ladder and identity init).

| scene | set | lr_F | objective | optimum | gap % | step within 5% | gamma | max dF to lr 0.01 |
|---|---|---|---|---|---|---|---|---|
| buu | lom | 0.001 | 1.2513e-02 | 7.4558e-03 | +67.82 | 7100 | 0.211 | 0.231 |
| buu | lom | 0.003 | 7.5224e-03 | 7.4558e-03 | +0.89 | 3300 | 0.167 | 0.074 |
| buu | lom | 0.01 | 7.4635e-03 | 7.4558e-03 | +0.10 | 3100 | 0.164 | 0.000 |
| buu | vexp | 0.001 | 1.2525e-02 | 7.6359e-03 | +64.03 | 7100 | 0.210 | 0.217 |
| buu | vexp | 0.003 | 7.7510e-03 | 7.6359e-03 | +1.51 | 3400 | 0.168 | 0.077 |
| buu | vexp | 0.01 | 7.6555e-03 | 7.6359e-03 | +0.26 | 3100 | 0.165 | 0.000 |
| chair | lom | 0.001 | 1.2527e-02 | 5.7078e-03 | +119.47 | 7800 | 0.190 | 0.275 |
| chair | lom | 0.003 | 5.8195e-03 | 5.7078e-03 | +1.96 | 3600 | 0.113 | 0.125 |
| chair | lom | 0.01 | 5.7111e-03 | 5.7078e-03 | +0.06 | 3100 | 0.108 | 0.000 |
| chair | vexp | 0.001 | 1.2511e-02 | 5.8215e-03 | +114.92 | 7800 | 0.190 | 0.257 |
| chair | vexp | 0.003 | 5.9711e-03 | 5.8215e-03 | +2.57 | 3700 | 0.113 | 0.114 |
| chair | vexp | 0.01 | 5.8471e-03 | 5.8215e-03 | +0.44 | 3100 | 0.108 | 0.000 |
| sofa | lom | 0.001 | 1.2291e-02 | 6.3199e-03 | +94.48 | 7400 | 0.201 | 0.282 |
| sofa | lom | 0.003 | 6.4000e-03 | 6.3199e-03 | +1.27 | 3500 | 0.140 | 0.091 |
| sofa | lom | 0.01 | 6.3222e-03 | 6.3199e-03 | +0.04 | 3100 | 0.136 | 0.000 |
| sofa | vexp | 0.001 | 1.2383e-02 | 6.5394e-03 | +89.36 | 7400 | 0.201 | 0.271 |
| sofa | vexp | 0.003 | 6.6628e-03 | 6.5394e-03 | +1.89 | 3600 | 0.141 | 0.089 |
| sofa | vexp | 0.01 | 6.5572e-03 | 6.5394e-03 | +0.27 | 3100 | 0.137 | 0.000 |
| bike | lom | 0.001 | 1.2590e-02 | 4.4300e-03 | +184.20 | 8000 | 0.183 | 0.340 |
| bike | lom | 0.003 | 4.7053e-03 | 4.4300e-03 | +6.21 | 5200 | 0.086 | 0.191 |
| bike | lom | 0.01 | 4.4364e-03 | 4.4300e-03 | +0.14 | 3100 | 0.075 | 0.000 |
| bike | vexp | 0.001 | 1.2672e-02 | 4.5009e-03 | +181.55 | 8000 | 0.182 | 0.326 |
| bike | vexp | 0.003 | 4.8161e-03 | 4.5009e-03 | +7.00 | 5000 | 0.086 | 0.185 |
| bike | vexp | 0.01 | 4.5298e-03 | 4.5009e-03 | +0.64 | 3200 | 0.075 | 0.000 |
| shrub | lom | 0.001 | 1.0944e-02 | 3.6294e-03 | +201.54 | 8200 | 0.184 | 0.303 |
| shrub | lom | 0.003 | 3.7151e-03 | 3.6294e-03 | +2.36 | 5000 | 0.085 | 0.105 |
| shrub | lom | 0.01 | 3.6407e-03 | 3.6294e-03 | +0.31 | 3100 | 0.077 | 0.000 |
| shrub | vexp | 0.001 | 1.1016e-02 | 3.6855e-03 | +198.90 | 8200 | 0.184 | 0.295 |
| shrub | vexp | 0.003 | 3.8245e-03 | 3.6855e-03 | +3.77 | 4900 | 0.085 | 0.117 |
| shrub | vexp | 0.01 | 3.7158e-03 | 3.6855e-03 | +0.82 | 3100 | 0.077 | 0.000 |


## P1 fit: weighted residual x255 against the targets

| scene | set | oracle | quadratic-cdf-priorF | usedrange-cdf-priorF | quadratic-identity-priorF | quadratic-cdf-noprior | quadratic-cdf-priorL | usedrange y_max | gamma | detach gamma (mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| buu | lom | 11.665 | 40.405 | 40.436 | 41.417 | 13.183 | 40.378 | 0.236 | 0.164 | 0.195 |
| buu | vexp | 11.676 | 39.202 | 39.227 | 40.193 | 13.487 | 41.959 | 0.236 | 0.165 | 0.378 |
| chair | lom | 5.832 | 43.617 | 43.600 | 44.675 | 8.406 | 43.658 | 0.122 | 0.108 | 0.095 |
| chair | vexp | 7.345 | 42.425 | 42.685 | 43.342 | 9.541 | 45.253 | 0.123 | 0.108 | -0.065 |
| sofa | lom | 7.841 | 42.041 | 42.058 | 44.006 | 8.729 | 41.839 | 0.165 | 0.136 | 0.149 |
| sofa | vexp | 8.008 | 40.970 | 41.084 | 42.624 | 9.034 | 42.833 | 0.166 | 0.137 | 0.073 |
| bike | lom | 3.998 | 53.215 | 53.409 | 58.455 | 4.617 | 53.021 | 0.151 | 0.075 | 0.065 |
| bike | vexp | 6.512 | 52.574 | 53.256 | 58.776 | 7.071 | 54.181 | 0.152 | 0.075 | -0.145 |
| shrub | lom | 1.888 | 45.755 | 45.707 | 45.411 | 2.026 | 45.871 | 0.188 | 0.077 | 0.068 |
| shrub | vexp | 2.492 | 43.680 | 43.665 | 43.404 | 2.414 | 45.626 | 0.188 | 0.077 | -0.151 |


Objective against the full-batch optimum, and convergence.

| scene | set | config | objective | optimum | optimum from | gap % | step within 5% | init-to-final max dF | replay s |
|---|---|---|---|---|---|---|---|---|---|
| buu | lom | quadratic-cdf-priorF | 7.4635e-03 | 7.4558e-03 | cold | +0.10 | 3100 | 0.249 | 11.3 |
| buu | lom | quadratic-cdf-noprior | 2.0556e-04 | 2.0541e-04 | polish | +0.08 | 100 | 0.017 | 7.6 |
| buu | vexp | quadratic-cdf-priorF | 7.6555e-03 | 7.6359e-03 | cold | +0.26 | 3100 | 0.236 | 11.8 |
| buu | vexp | quadratic-cdf-noprior | 2.1631e-04 | 2.1606e-04 | polish | +0.12 | 100 | 0.019 | 7.7 |
| chair | lom | quadratic-cdf-priorF | 5.7111e-03 | 5.7078e-03 | cold | +0.06 | 3100 | 0.281 | 12.3 |
| chair | lom | quadratic-cdf-noprior | 6.2938e-05 | 6.2387e-05 | cold | +0.88 | 200 | 0.023 | 8.0 |
| chair | vexp | quadratic-cdf-priorF | 5.8471e-03 | 5.8215e-03 | cold | +0.44 | 3100 | 0.273 | 11.4 |
| chair | vexp | quadratic-cdf-noprior | 7.7470e-05 | 7.6975e-05 | cold | +0.64 | 200 | 0.026 | 7.5 |
| sofa | lom | quadratic-cdf-priorF | 6.3222e-03 | 6.3199e-03 | cold | +0.04 | 3100 | 0.287 | 11.2 |
| sofa | lom | quadratic-cdf-noprior | 7.7502e-05 | 7.7490e-05 | cold | +0.01 | 100 | 0.014 | 7.4 |
| sofa | vexp | quadratic-cdf-priorF | 6.5572e-03 | 6.5394e-03 | cold | +0.27 | 3100 | 0.276 | 11.1 |
| sofa | vexp | quadratic-cdf-noprior | 8.9823e-05 | 8.9807e-05 | cold | +0.02 | 100 | 0.019 | 7.2 |
| bike | lom | quadratic-cdf-priorF | 4.4364e-03 | 4.4300e-03 | cold | +0.14 | 3100 | 0.333 | 11.1 |
| bike | lom | quadratic-cdf-noprior | 5.0389e-05 | 5.0385e-05 | polish | +0.01 | 100 | 0.024 | 7.2 |
| bike | vexp | quadratic-cdf-priorF | 4.5298e-03 | 4.5009e-03 | cold | +0.64 | 3200 | 0.321 | 11.1 |
| bike | vexp | quadratic-cdf-noprior | 5.9687e-05 | 5.9677e-05 | cold | +0.02 | 200 | 0.023 | 7.3 |
| shrub | lom | quadratic-cdf-priorF | 3.6407e-03 | 3.6294e-03 | cold | +0.31 | 3100 | 0.301 | 11.2 |
| shrub | lom | quadratic-cdf-noprior | 2.5973e-05 | 2.5970e-05 | cold | +0.01 | 100 | 0.007 | 7.2 |
| shrub | vexp | quadratic-cdf-priorF | 3.7158e-03 | 3.6855e-03 | cold | +0.82 | 3100 | 0.293 | 11.0 |
| shrub | vexp | quadratic-cdf-noprior | 2.8595e-05 | 2.8587e-05 | cold | +0.03 | 100 | 0.007 | 7.2 |


## P2 exposure on the varying set: s_k against u_k (ideal slope -1, RMSE 0)

| scene | method | slope | r | rmse | rmse paired | move from init |
|---|---|---|---|---|---|---|
| buu | init | -1.037 | -0.952 | 0.193 | 0.006 | 0.000 |
| buu | oracle | -1.083 | -0.968 | 0.170 | 0.039 | 0.044 |
| buu | quadratic-cdf-priorF | -1.039 | -0.957 | 0.183 | 0.010 | 0.013 |
| buu | usedrange-cdf-priorF | -1.041 | -0.960 | 0.177 | 0.012 | 0.019 |
| buu | quadratic-cdf-priorF-unshifted | -1.082 | -0.959 | 0.189 | 0.034 | 0.029 |
| buu | quadratic-cdf-noprior | -1.026 | -0.958 | 0.178 | 0.007 | 0.017 |
| buu | quadratic-cdf-priorL | -0.820 | -0.958 | 0.176 | 0.117 | 0.135 |
| chair | init | -1.042 | -0.991 | 0.083 | 0.013 | 0.000 |
| chair | oracle | -1.093 | -0.995 | 0.083 | 0.043 | 0.044 |
| chair | quadratic-cdf-priorF | -1.034 | -0.993 | 0.072 | 0.012 | 0.027 |
| chair | usedrange-cdf-priorF | -1.027 | -0.995 | 0.062 | 0.014 | 0.035 |
| chair | quadratic-cdf-priorF-unshifted | -1.106 | -0.995 | 0.090 | 0.049 | 0.046 |
| chair | quadratic-cdf-noprior | -1.032 | -0.993 | 0.072 | 0.012 | 0.022 |
| chair | quadratic-cdf-priorL | -0.886 | -0.992 | 0.092 | 0.080 | 0.093 |
| sofa | init | -1.022 | -0.991 | 0.079 | 0.004 | 0.000 |
| sofa | oracle | -1.069 | -0.996 | 0.066 | 0.038 | 0.041 |
| sofa | quadratic-cdf-priorF | -1.023 | -0.990 | 0.084 | 0.013 | 0.013 |
| sofa | usedrange-cdf-priorF | -1.033 | -0.990 | 0.086 | 0.018 | 0.015 |
| sofa | quadratic-cdf-priorF-unshifted | -1.086 | -0.991 | 0.100 | 0.041 | 0.039 |
| sofa | quadratic-cdf-noprior | -1.007 | -0.991 | 0.078 | 0.007 | 0.011 |
| sofa | quadratic-cdf-priorL | -0.805 | -0.991 | 0.129 | 0.120 | 0.127 |
| bike | init | -0.952 | -0.990 | 0.083 | 0.019 | 0.000 |
| bike | oracle | -1.043 | -0.989 | 0.093 | 0.056 | 0.058 |
| bike | quadratic-cdf-priorF | -0.912 | -0.989 | 0.093 | 0.037 | 0.026 |
| bike | usedrange-cdf-priorF | -0.910 | -0.988 | 0.098 | 0.045 | 0.032 |
| bike | quadratic-cdf-priorF-unshifted | -1.033 | -0.990 | 0.088 | 0.051 | 0.049 |
| bike | quadratic-cdf-noprior | -0.893 | -0.989 | 0.100 | 0.048 | 0.037 |
| bike | quadratic-cdf-priorL | -0.768 | -0.989 | 0.150 | 0.119 | 0.107 |
| shrub | init | -0.984 | -0.983 | 0.105 | 0.014 | 0.000 |
| shrub | oracle | -1.108 | -0.980 | 0.145 | 0.082 | 0.077 |
| shrub | quadratic-cdf-priorF | -0.977 | -0.982 | 0.110 | 0.018 | 0.012 |
| shrub | usedrange-cdf-priorF | -0.976 | -0.980 | 0.114 | 0.017 | 0.018 |
| shrub | quadratic-cdf-priorF-unshifted | -1.077 | -0.982 | 0.127 | 0.066 | 0.056 |
| shrub | quadratic-cdf-noprior | -0.952 | -0.981 | 0.112 | 0.018 | 0.021 |
| shrub | quadratic-cdf-priorL | -0.782 | -0.982 | 0.153 | 0.116 | 0.119 |


## P3 paired pseudo-target leak (slope of log2(mean, varying / LOM) on u_k; 0 = compensated, 1 = passes through)

M1 columns: mean over 3 seeds from `results/vexp/summary/T4_exposure.csv`.

| scene | input | official_M1 | detach_M1 | quadratic-cdf-priorF | usedrange-cdf-priorF | quadratic-identity-priorF | quadratic-cdf-priorF-unshifted | quadratic-cdf-noprior | quadratic-cdf-priorL | unpaired_quadratic-cdf-priorF |
|---|---|---|---|---|---|---|---|---|---|---|
| buu | +1.008 | +0.564 | +0.653 | +0.001 | -0.001 | +0.002 | -0.022 | +0.013 | +0.122 | -0.000 |
| chair | +1.015 | +0.345 | +0.496 | +0.008 | +0.011 | +0.006 | -0.031 | +0.020 | +0.086 | +0.011 |
| sofa | +1.002 | +0.399 | +0.547 | -0.003 | -0.011 | +0.000 | -0.034 | +0.007 | +0.115 | -0.004 |
| bike | +0.996 | +0.271 | +0.506 | +0.029 | +0.035 | +0.007 | -0.043 | +0.098 | +0.121 | +0.031 |
| shrub | +1.009 | +0.445 | +0.357 | +0.026 | +0.028 | +0.027 | -0.004 | +0.026 | +0.096 | +0.024 |


## P4 dark end (LOM): mean pseudo-target value over views at input level l, in 8-bit levels

detach: seed-0 curves through the official LUT (level l reads curve[max(l-1, 0)]).

| scene | level | detach | quadratic-cdf-priorF | quadratic-cdf-priorF-unshifted | quadratic-cdf-noprior | quadratic-cdf-priorL |
|---|---|---|---|---|---|---|
| buu | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| buu | 1 | 0.00 | 45.75 | 46.22 | 0.04 | 47.43 |
| buu | 2 | 44.63 | 54.04 | 55.80 | 0.14 | 56.30 |
| buu | 4 | 58.89 | 66.69 | 71.29 | 3.92 | 66.19 |
| buu | 8 | 78.33 | 84.21 | 87.02 | 23.44 | 84.29 |
| buu | 16 | 119.39 | 119.09 | 126.74 | 76.64 | 118.88 |
| chair | 0 | 68.09 | 0.00 | 0.00 | 0.00 | 0.00 |
| chair | 1 | 68.09 | 63.98 | 65.80 | 0.02 | 64.32 |
| chair | 2 | 76.21 | 72.32 | 79.03 | 0.41 | 73.26 |
| chair | 4 | 97.06 | 95.46 | 106.82 | 25.38 | 94.75 |
| chair | 8 | 124.77 | 121.37 | 126.27 | 67.22 | 121.29 |
| chair | 16 | 172.53 | 166.70 | 175.84 | 145.58 | 166.41 |
| sofa | 0 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| sofa | 1 | 0.00 | 55.52 | 57.22 | 0.04 | 55.05 |
| sofa | 2 | 54.53 | 63.70 | 65.86 | 0.07 | 64.50 |
| sofa | 4 | 68.62 | 73.95 | 76.94 | 1.03 | 73.43 |
| sofa | 8 | 84.43 | 89.26 | 98.70 | 20.16 | 88.89 |
| sofa | 16 | 153.74 | 155.39 | 159.66 | 135.74 | 156.03 |
| bike | 0 | 83.74 | 0.00 | 0.00 | 0.00 | 0.00 |
| bike | 1 | 83.74 | 78.01 | 81.89 | 0.10 | 77.38 |
| bike | 2 | 90.60 | 86.95 | 93.04 | 1.53 | 88.34 |
| bike | 4 | 108.88 | 106.70 | 123.90 | 24.73 | 105.03 |
| bike | 8 | 181.06 | 178.36 | 195.50 | 159.84 | 178.52 |
| bike | 16 | 228.40 | 225.69 | 228.39 | 243.90 | 225.63 |
| shrub | 0 | 85.63 | 0.00 | 0.00 | 0.00 | 0.00 |
| shrub | 1 | 85.63 | 80.41 | 91.21 | 6.43 | 79.51 |
| shrub | 2 | 100.04 | 96.00 | 109.94 | 22.61 | 97.84 |
| shrub | 4 | 129.40 | 128.06 | 141.60 | 67.80 | 126.68 |
| shrub | 8 | 173.24 | 170.23 | 180.24 | 145.06 | 170.14 |
| shrub | 16 | 220.75 | 218.11 | 222.45 | 230.01 | 218.36 |


Dark-end slope (mean output gain per input step over levels 1-8; how much photo noise is amplified) and the fraction of training pixels at levels 0-2.

| scene | set | pixels at 0-2 | detach | quadratic-cdf-priorF | quadratic-cdf-priorF-unshifted | quadratic-cdf-noprior | quadratic-cdf-priorL |
|---|---|---|---|---|---|---|---|
| buu | lom | 0.74% | 11.19 | 5.49 | 5.83 | 3.34 | 5.27 |
| buu | vexp | 1.53% | 8.27 | 6.42 | 7.13 | 3.94 | 5.91 |
| chair | lom | 2.17% | 8.10 | 8.20 | 8.64 | 9.60 | 8.14 |
| chair | vexp | 6.10% | 19.37 | 9.65 | 10.57 | 10.51 | 9.17 |
| sofa | lom | 0.28% | 12.06 | 4.82 | 5.93 | 2.87 | 4.83 |
| sofa | vexp | 1.40% | 13.81 | 7.16 | 8.26 | 5.96 | 6.32 |
| bike | lom | 3.60% | 13.90 | 14.33 | 16.23 | 22.82 | 14.45 |
| bike | vexp | 6.56% | 12.16 | 14.79 | 16.16 | 21.84 | 14.28 |
| shrub | lom | 22.52% | 12.51 | 12.83 | 12.72 | 19.80 | 12.95 |
| shrub | vexp | 24.40% | 11.39 | 13.65 | 13.79 | 19.81 | 13.06 |


Distance to detach's curves as applied to input levels 0-255 (mean over views, weighted by how often each level occurs in the training photos), x255.

| scene | set | quadratic-cdf-priorF | quadratic-cdf-priorF-unshifted | quadratic-cdf-noprior | quadratic-cdf-priorL |
|---|---|---|---|---|---|
| buu | lom | 14.28 | 15.34 | 38.41 | 11.92 |
| buu | vexp | 34.78 | 37.39 | 51.57 | 32.25 |
| chair | lom | 5.78 | 6.87 | 41.38 | 5.22 |
| chair | vexp | 31.19 | 30.96 | 65.50 | 27.29 |
| sofa | lom | 6.34 | 9.66 | 32.23 | 5.26 |
| sofa | vexp | 30.17 | 31.47 | 55.47 | 24.57 |
| bike | lom | 5.42 | 13.24 | 46.19 | 4.79 |
| bike | vexp | 33.10 | 32.08 | 72.82 | 28.88 |
| shrub | lom | 11.40 | 14.25 | 43.41 | 10.27 |
| shrub | vexp | 33.60 | 31.70 | 63.83 | 29.90 |


## P5 pseudo-target quality against high/ (DIAGNOSTIC ONLY: never chooses a setting)

detach: mean over views per run, then mean ± sd over seeds 0-2 (`rescore/pseudo_targets.csv`).

| scene | set | config | PSNR | detach PSNR | bm PSNR | detach bm | bm gap | gap / sd |
|---|---|---|---|---|---|---|---|---|
| buu | lom | quadratic-cdf-priorF | 20.78 | 22.41 ± 0.10 | 26.72 | 26.79 ± 0.02 | -0.07 | -4.7 |
| buu | lom | quadratic-cdf-priorF-unshifted | 19.44 | 22.41 ± 0.10 | 26.42 | 26.79 ± 0.02 | -0.37 | -23.8 |
| buu | lom | quadratic-cdf-noprior | 16.11 | 22.41 ± 0.10 | 15.94 | 26.79 ± 0.02 | -10.85 | -690.3 |
| buu | lom | quadratic-cdf-priorL | 20.72 | 22.41 ± 0.10 | 26.64 | 26.79 ± 0.02 | -0.15 | -9.6 |
| buu | vexp | quadratic-cdf-priorF | 20.85 | 19.81 ± 0.02 | 26.54 | 27.04 ± 0.07 | -0.50 | -6.9 |
| buu | vexp | quadratic-cdf-priorF-unshifted | 19.51 | 19.81 ± 0.02 | 26.43 | 27.04 ± 0.07 | -0.61 | -8.3 |
| buu | vexp | quadratic-cdf-noprior | 16.07 | 19.81 ± 0.02 | 15.92 | 27.04 ± 0.07 | -11.12 | -152.9 |
| buu | vexp | quadratic-cdf-priorL | 20.90 | 19.81 ± 0.02 | 26.31 | 27.04 ± 0.07 | -0.73 | -10.1 |
| chair | lom | quadratic-cdf-priorF | 20.35 | 19.53 ± 0.01 | 23.48 | 23.43 ± 0.00 | +0.06 | +21.8 |
| chair | lom | quadratic-cdf-priorF-unshifted | 18.28 | 19.53 ± 0.01 | 22.80 | 23.43 ± 0.00 | -0.63 | -242.0 |
| chair | lom | quadratic-cdf-noprior | 16.78 | 19.53 ± 0.01 | 16.41 | 23.43 ± 0.00 | -7.02 | -2697.7 |
| chair | lom | quadratic-cdf-priorL | 20.36 | 19.53 ± 0.01 | 23.50 | 23.43 ± 0.00 | +0.07 | +28.6 |
| chair | vexp | quadratic-cdf-priorF | 20.61 | 16.77 ± 0.18 | 23.57 | 21.49 ± 0.16 | +2.07 | +12.7 |
| chair | vexp | quadratic-cdf-priorF-unshifted | 18.51 | 16.77 ± 0.18 | 23.03 | 21.49 ± 0.16 | +1.53 | +9.4 |
| chair | vexp | quadratic-cdf-noprior | 16.72 | 16.77 ± 0.18 | 16.41 | 21.49 ± 0.16 | -5.08 | -31.3 |
| chair | vexp | quadratic-cdf-priorL | 20.20 | 16.77 ± 0.18 | 23.23 | 21.49 ± 0.16 | +1.74 | +10.7 |
| sofa | lom | quadratic-cdf-priorF | 24.79 | 25.75 ± 0.02 | 28.54 | 28.74 ± 0.00 | -0.19 | -694.5 |
| sofa | lom | quadratic-cdf-priorF-unshifted | 22.61 | 25.75 ± 0.02 | 28.52 | 28.74 ± 0.00 | -0.22 | -780.0 |
| sofa | lom | quadratic-cdf-noprior | 16.69 | 25.75 ± 0.02 | 15.91 | 28.74 ± 0.00 | -12.82 | -45689.0 |
| sofa | lom | quadratic-cdf-priorL | 24.79 | 25.75 ± 0.02 | 28.48 | 28.74 ± 0.00 | -0.25 | -894.7 |
| sofa | vexp | quadratic-cdf-priorF | 24.64 | 19.73 ± 0.01 | 27.83 | 26.73 ± 0.10 | +1.10 | +11.3 |
| sofa | vexp | quadratic-cdf-priorF-unshifted | 22.62 | 19.73 ± 0.01 | 28.17 | 26.73 ± 0.10 | +1.45 | +14.8 |
| sofa | vexp | quadratic-cdf-noprior | 16.64 | 19.73 ± 0.01 | 15.90 | 26.73 ± 0.10 | -10.82 | -110.6 |
| sofa | vexp | quadratic-cdf-priorL | 24.30 | 19.73 ± 0.01 | 27.76 | 26.73 ± 0.10 | +1.03 | +10.5 |
| bike | lom | quadratic-cdf-priorF | 17.13 | 16.71 ± 0.12 | 22.51 | 22.50 ± 0.01 | +0.00 | +0.2 |
| bike | lom | quadratic-cdf-priorF-unshifted | 14.29 | 16.71 ± 0.12 | 22.11 | 22.50 ± 0.01 | -0.39 | -42.6 |
| bike | lom | quadratic-cdf-noprior | 14.84 | 16.71 ± 0.12 | 14.08 | 22.50 ± 0.01 | -8.42 | -912.3 |
| bike | lom | quadratic-cdf-priorL | 17.12 | 16.71 ± 0.12 | 22.47 | 22.50 ± 0.01 | -0.04 | -4.1 |
| bike | vexp | quadratic-cdf-priorF | 17.21 | 15.08 ± 0.30 | 22.22 | 21.61 ± 0.11 | +0.60 | +5.4 |
| bike | vexp | quadratic-cdf-priorF-unshifted | 14.29 | 15.08 ± 0.30 | 21.94 | 21.61 ± 0.11 | +0.33 | +2.9 |
| bike | vexp | quadratic-cdf-noprior | 14.75 | 15.08 ± 0.30 | 14.02 | 21.61 ± 0.11 | -7.60 | -67.5 |
| bike | vexp | quadratic-cdf-priorL | 17.26 | 15.08 ± 0.30 | 22.28 | 21.61 ± 0.11 | +0.67 | +5.9 |
| shrub | lom | quadratic-cdf-priorF | 11.87 | 11.68 ± 0.43 | 17.74 | 17.52 ± 0.17 | +0.22 | +1.3 |
| shrub | lom | quadratic-cdf-priorF-unshifted | 10.74 | 11.68 ± 0.43 | 17.10 | 17.52 ± 0.17 | -0.42 | -2.5 |
| shrub | lom | quadratic-cdf-noprior | 14.47 | 11.68 ± 0.43 | 19.04 | 17.52 ± 0.17 | +1.53 | +9.2 |
| shrub | lom | quadratic-cdf-priorL | 11.87 | 11.68 ± 0.43 | 17.73 | 17.52 ± 0.17 | +0.21 | +1.3 |
| shrub | vexp | quadratic-cdf-priorF | 12.00 | 9.87 ± 0.14 | 17.83 | 16.22 ± 0.06 | +1.61 | +27.2 |
| shrub | vexp | quadratic-cdf-priorF-unshifted | 10.87 | 9.87 ± 0.14 | 17.23 | 16.22 ± 0.06 | +1.01 | +17.1 |
| shrub | vexp | quadratic-cdf-noprior | 14.42 | 9.87 ± 0.14 | 18.85 | 16.22 ± 0.06 | +2.63 | +44.3 |
| shrub | vexp | quadratic-cdf-priorL | 11.92 | 9.87 ± 0.14 | 17.67 | 16.22 ± 0.06 | +1.45 | +24.5 |


## Checks against M1

| scene | init_vs_oracle_json_s_init | oracle_vs_oracle_json_s_oracle |
|---|---|---|
| buu | 2.23e-07 | 0.00e+00 |
| chair | 3.28e-07 | 0.00e+00 |
| sofa | 2.80e-07 | 0.00e+00 |
| bike | 2.59e-07 | 0.00e+00 |
| shrub | 2.52e-07 | 0.00e+00 |


Tolerances as in `vexp_analyze.py`: init 1e-6, oracle 2e-3 stops.


Figures: `figures/curves_<scene>.png` (default configuration), `figures/exposure_vs_u.png`.
