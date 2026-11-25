# A1: does the identity-init F converge to the cdf-init F with longer curve training? (curve loss only)

LOM low-light training photos, both inits, 50000 steps (learning rate decayed to 1% over the replay, so this models a 50000-step run), seed 0. Generated 2026-11-25 10:33 in 698 s by `rung2_replay.py` (e0f28deb42522953). Every number is *measured* on the replay.

Reading (log, 2026-11-24): early stopping if the final gap < 0.01; two end points if not and the gap fell by less than 10% over the last fifth; undecided otherwise. Built-in check: the 10000-step replay's final gap within 0.01 of the trainer's (seeds 0-2).

| scene | steps | gap_first | gap_at_ref | ref_step | gap_final | outcome | step_gap_halves | objective_cdf_final | objective_identity_final | check_gap_10000 | trainer_gap_10k_seeds | check_ok | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| buu | 50000 | 0.078 | 0.002 | 40000 | 0.002 | early stopping | 4000 | 0.007 | 0.007 | 0.034 | [0.0345, 0.0345, 0.0345] | True | 142.900 |
| chair | 50000 | 0.126 | 0.000 | 40000 | 0.000 | early stopping | 3000 | 0.006 | 0.006 | 0.031 | [0.031, 0.031, 0.031] | True | 139.700 |
| sofa | 50000 | 0.095 | 0.001 | 40000 | 0.001 | early stopping | 4000 | 0.006 | 0.006 | 0.035 | [0.0349, 0.0349, 0.0349] | True | 139.000 |
| bike | 50000 | 0.177 | 0.001 | 40000 | 0.001 | early stopping | 4000 | 0.004 | 0.004 | 0.051 | [0.051, 0.051, 0.051] | True | 139.800 |
| shrub | 50000 | 0.106 | 0.002 | 40000 | 0.002 | early stopping | 2000 | 0.004 | 0.004 | 0.031 | [0.031, 0.031, 0.031] | True | 136.800 |


Figure: `figures/gap_by_step.png`; per-step values: `gap_by_step.csv`.
