# M6 preview: Rung 3's and rung2id's curve training replayed on the training photos (no 3DGS)

10k steps per scene (learning rate decayed to 1%), identity init, seed 0, the same view order for both models; Rung 3 with fixed zone weights at s_init and a 16-px guide, as the trainer builds it. Training photos only: no normal-light image is read. Generated 2026-11-27 13:49 in 307 s by `rung2_replay.py` (2d2838098c61da64; rung3.py 77beed49af5ae30a). Every number is *measured* on the replay. Plan: `M6_rung3_plan.md` test 5 and section 4.3.

**Stop rule** (plan test 5): a curve not strictly increasing; every decision scene with max |F_sh − F_lit| < 0.01; s more than 0.05 stops RMS from the rung2id replay's. **pass: the batch can be proposed**.

## Predicted directions (plan 4.3; logged before the batch)

| scene | Δ sat (rung3 − rung2id) | prediction (LOM val sat ratio) |
|---|---|---|
| buu | 0.1016 | + |
| chair | 0.0962 | + |
| sofa | 0.0685 | + |
| bike | 0.0243 | + |

| scene | Δ hue drift (°) | prediction (LOM val hue error) |
|---|---|---|
| bike | -0.723 | no prediction |


## Curves on the 255-bin grid at b = 1, 2, 4, 8, 16, 32, 64 (×255), γ, split, exposures (LOM)

| scene | F_sh_x255 | F_lit_x255 | F_rung2id_x255 | max_F_sh_minus_lit | gamma_rung3 | gamma_rung2id | s_rms_rung3_vs_rung2id | max_abs_dw_lit_final_vs_init | objective_rung3 | objective_rung2id |
|---|---|---|---|---|---|---|---|---|---|---|
| buu | [40.2, 51.3, 67.7, 91.3, 150.4, 215.6, 229.6] | [41.9, 51.7, 63.0, 72.4, 84.9, 152.2, 230.5] | [42.2, 52.2, 66.1, 82.8, 118.8, 187.6, 230.0] | 0.3862 | 0.1649 | 0.1647 | 0.0318 | 0.0607 | 0.0099 | 0.0075 |
| chair | [58.6, 78.2, 113.1, 161.9, 213.3, 230.1, 238.4] | [54.7, 65.2, 76.6, 89.2, 140.5, 229.8, 238.8] | [56.9, 71.6, 93.3, 120.8, 173.5, 229.4, 238.6] | 0.3438 | 0.1087 | 0.1086 | 0.0039 | 0.0098 | 0.0072 | 0.0058 |
| sofa | [43.1, 54.8, 71.7, 110.5, 198.9, 224.1, 233.9] | [47.1, 57.8, 69.5, 80.3, 103.3, 194.6, 234.3] | [47.2, 57.9, 72.0, 96.3, 152.7, 209.9, 234.0] | 0.3751 | 0.1372 | 0.1368 | 0.0086 | 0.0147 | 0.0086 | 0.0064 |
| bike | [68.5, 90.9, 131.2, 198.1, 230.7, 238.6, 244.1] | [63.0, 78.7, 102.6, 155.0, 221.1, 236.9, 244.0] | [65.9, 84.9, 116.7, 176.3, 225.8, 237.6, 244.0] | 0.1751 | 0.0746 | 0.0745 | 0.0013 | 0.0067 | 0.0049 | 0.0045 |
| shrub | [81.1, 108.1, 148.1, 193.1, 226.5, 237.4, 243.7] | [66.8, 84.3, 110.3, 150.7, 209.4, 235.6, 243.7] | [73.8, 95.8, 128.6, 171.3, 218.0, 236.3, 243.7] | 0.1711 | 0.0765 | 0.0765 | 0.0007 | 0.0070 | 0.0040 | 0.0037 |


## Zone brightness of the pseudo-targets (w-weighted mean grey ×255; inverted = shadow zone brighter)

| scene | photo_zone_sh_x255 | photo_zone_lit_x255 | rung2id_zone_sh_x255 | rung2id_zone_lit_x255 | rung2id_lit_over_sh | rung3_zone_sh_x255 | rung3_zone_lit_x255 | rung3_lit_over_sh | rung2id_inverted_share | rung3_inverted_share |
|---|---|---|---|---|---|---|---|---|---|---|
| buu | 15.326 | 31.665 | 114.717 | 177.149 | 1.544 | 130.796 | 160.997 | 1.231 | 0.000 | 0.000 |
| chair | 7.487 | 17.361 | 115.151 | 180.597 | 1.568 | 131.444 | 166.597 | 1.267 | 0.000 | 0.000 |
| sofa | 11.118 | 25.217 | 118.214 | 184.205 | 1.558 | 133.695 | 165.529 | 1.238 | 0.000 | 0.000 |
| bike | 5.304 | 9.116 | 135.181 | 174.717 | 1.292 | 145.579 | 163.680 | 1.124 | 0.000 | 0.000 |
| shrub | 5.582 | 10.325 | 129.045 | 165.140 | 1.280 | 138.680 | 157.045 | 1.132 | 0.000 | 0.000 |


## Colour on chromatic photo pixels (mean HSV saturation; hue drift from the photo, °)

| scene | photo_sat_chroma | rung2id_sat_chroma | rung3_sat_chroma | d_sat_chroma | rung2id_hue_drift_deg | rung3_hue_drift_deg | d_hue_drift_deg | rung3_hue_px_frac |
|---|---|---|---|---|---|---|---|---|
| buu | 0.5249 | 0.3555 | 0.4571 | 0.1016 | 4.8766 | 4.5020 | -0.3746 | 1.0000 |
| chair | 0.3700 | 0.2246 | 0.3207 | 0.0962 | 4.2721 | 3.8574 | -0.4147 | 0.9999 |
| sofa | 0.5779 | 0.3885 | 0.4571 | 0.0685 | 2.5804 | 3.4211 | 0.8407 | 1.0000 |
| bike | 0.6325 | 0.3651 | 0.3894 | 0.0243 | 10.4000 | 9.6772 | -0.7228 | 0.9897 |
| shrub | 0.7003 | 0.4432 | 0.4572 | 0.0140 | 4.4755 | 4.2459 | -0.2296 | 0.9976 |


## Section 1.4 recomputed with the trainer's loader

| scene | views | centres_x255 | c2_minus_c1 | lit_share_px | zone_shares_mean | cdf_gap_mean | chroma_px_frac | chroma_lit_hard | chroma_lit_soft | chroma_in_ramp | chroma_maxlevel_median | T8_sh_lit_all_x255 | T4_sh_lit_all_x255 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buu | 22 | [15.98, 31.36] | 0.973 | 0.473 | [0.5350906848907471, 0.46490925550460815] | 0.764 | 0.763 | 0.559 | 0.551 | 0.480 | 38.000 | [43.5, 2.4, 24.3] | [4.7, 0.3, 2.7] |
| chair | 43 | [7.4, 18.67] | 1.336 | 0.564 | [0.4390755891799927, 0.5609244108200073] | 0.729 | 0.486 | 0.796 | 0.759 | 0.472 | 22.000 | [151.0, 3.1, 68.1] | [60.0, 0.4, 26.7] |
| sofa | 29 | [10.48, 24.56] | 1.229 | 0.479 | [0.519137442111969, 0.4808625876903534] | 0.771 | 0.267 | 0.119 | 0.188 | 0.567 | 14.000 | [28.7, 0.6, 15.2] | [1.7, 0.1, 0.9] |
| bike | 36 | [5.44, 8.53] | 0.649 | 0.517 | [0.4849761426448822, 0.5150238871574402] | 0.573 | 0.134 | 0.746 | 0.729 | 0.421 | 13.000 | [229.9, 93.8, 159.8] | [41.2, 8.8, 24.5] |
| shrub | 30 | [6.09, 9.38] | 0.623 | 0.520 | [0.4845634400844574, 0.515436589717865] | 0.376 | 0.812 | 0.612 | 0.593 | 0.543 | 11.000 | [193.0, 101.1, 145.6] | [109.6, 30.4, 68.8] |


## Varying set (exposure check only)

| scene | centres_x255 | zone_shares_mean | F_sh_x255 | F_lit_x255 | F_rung2id_x255 | gamma_rung3 | s_move_rung3 | s_move_rung2id | s_rms_rung3_vs_rung2id | max_abs_dw_lit_final_vs_init | increasing_rung3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buu | [15.83, 31.28] | [0.5329015851020813, 0.4670984148979187] | [37.6, 48.6, 65.0, 89.9, 151.9, 217.0, 231.2] | [34.4, 45.1, 57.9, 68.8, 82.3, 151.7, 232.1] | [37.6, 48.2, 62.7, 80.5, 117.9, 188.0, 231.5] | 0.1657 | 0.0307 | 0.0058 | 0.0325 | 0.0632 | True |
| chair | [7.26, 18.57] | [0.43971771001815796, 0.5602823495864868] | [57.9, 77.2, 112.5, 162.5, 214.8, 231.6, 239.5] | [46.1, 57.7, 71.5, 86.1, 140.8, 231.2, 239.8] | [53.3, 68.0, 90.4, 119.2, 174.3, 230.7, 239.7] | 0.1085 | 0.0080 | 0.0039 | 0.0046 | 0.0140 | True |
| sofa | [10.62, 24.58] | [0.5204935669898987, 0.4795064330101013] | [40.1, 51.3, 68.6, 108.8, 198.7, 225.7, 235.2] | [38.5, 49.8, 63.5, 76.2, 99.8, 194.6, 235.5] | [41.8, 52.8, 68.0, 93.2, 150.6, 210.9, 235.2] | 0.1385 | 0.0110 | 0.0048 | 0.0086 | 0.0182 | True |
| bike | [5.45, 8.51] | [0.48632732033729553, 0.5136726498603821] | [68.3, 90.4, 131.4, 197.7, 231.3, 239.4, 244.5] | [60.0, 75.8, 100.6, 153.8, 221.4, 237.8, 244.4] | [64.5, 83.2, 115.7, 175.6, 226.4, 238.5, 244.4] | 0.0743 | 0.0059 | 0.0054 | 0.0017 | 0.0193 | True |
| shrub | [6.07, 9.4] | [0.48437270522117615, 0.5156272649765015] | [80.5, 106.7, 146.8, 192.8, 227.3, 238.3, 244.2] | [63.6, 80.9, 107.2, 148.9, 209.1, 236.5, 244.1] | [72.1, 93.6, 126.4, 170.3, 218.4, 237.2, 244.1] | 0.0764 | 0.0037 | 0.0038 | 0.0009 | 0.0119 | True |


Figures: `figures/zone_curves.png`; `figures/bike_29_photo4_wlit_rung2id_rung3.png` (photo ×4 | w_lit | rung2id pseudo-target | rung3 pseudo-target).
