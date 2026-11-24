# E6 pre-launch check: Rung 2's curve on the LOM over_exp photos (curve loss only, no 3DGS)

Default configuration (`quadratic-cdf-priorF`), 10000 steps, curve weight as `--exp_name over_exp`, seed 0. Training photos only. Generated 2026-11-24 12:51 in 61 s by `rung2_replay.py` (e0f28deb42522953). Every number is *measured* on the replay.

**Stop rule** (log, 2026-11-24): gamma <= 1 or F below the identity at fewer than 80% of interior grid points in any scene, or pooled residual above 2x the low-light preview's.

- Pooled residual x255: 18.44 (low light, `preview_final`: 45.01; ratio 0.41).
- **pass: E6 can join the launch**.

| scene | views | mean_input | gamma | F_below_identity_frac | mean_F_minus_x_x255 | F_at_64_127_191_x255 | resid255 | low_light_resid255 | s_move_rmse | abs_mean_s | share_level_255 | mean_cdf253 | min_cdf253 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buu | 22 | 0.915 | 12.765 | 1.000 | -108.661 | [0.1, 0.5, 9.6] | 17.942 | 40.405 | 0.006 | 0.000 | 0.101 | 0.999 | 0.995 |
| chair | 43 | 0.794 | 4.052 | 1.000 | -76.918 | [0.8, 16.7, 78.4] | 16.564 | 43.617 | 0.007 | 0.000 | 0.000 | 1.000 | 1.000 |
| sofa | 29 | 0.839 | 5.262 | 1.000 | -87.204 | [0.4, 5.9, 54.6] | 14.641 | 42.041 | 0.004 | 0.000 | 0.000 | 1.000 | 1.000 |
| bike | 36 | 0.811 | 4.441 | 1.000 | -80.735 | [0.5, 10.9, 68.2] | 27.400 | 53.215 | 0.006 | 0.000 | 0.011 | 0.998 | 0.996 |
| shrub | 30 | 0.657 | 1.938 | 1.000 | -40.761 | [14.8, 65.3, 149.4] | 15.669 | 45.755 | 0.005 | 0.000 | 0.000 | 1.000 | 1.000 |


Figure: `figures/overexp_curves.png`.
