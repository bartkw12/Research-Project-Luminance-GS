# M5 plan: H3, the identity-init follow-up (H1b, H2b) and the road to the 5 Dec freeze

Written 2026-11-24 in the project thread, for Claude Code on Bart's PC, after Bart chose option B ("add a fresh H1b test") on 2026-11-24. It continues `M2_rung2_plan.md` (M2 and M3 are done) and `Rung2_results.md`.

**Status: step 0 done (2026-11-24).** Bart confirms the pre-registered readings in section 3 before step 1 and any run.

**Updated 2026-11-24, with Bart's notes:**
- M4 is done: Bart emailed Wei the M3 results.
- Rung 3 stays, as M6 (section 8).
- Both batches run today, back to back, and M5 should finish today (section 5).
- Overexposure is added as E6 (H1 on LOM `over_exp`, H1's original scope), running overnight after the day batches. Bart takes Wei's general go-ahead as approval.

**Revised 2026-11-24 after step 0** (Claude Code checked the plan against the code; Bart approved the fixes):
- **Seven steps could not run as written.** They are fixed in sections 2 and 4:
  - the `over_exp` preview and A1 need a new script;
  - E3 goes through the generic script;
  - the `rung2_analyze.py` regression test masks its date and fingerprint lines;
  - H2b(a) gets a script;
  - the w = 0.1 check is done in the code;
  - a `rung2id` smoke run on the varying set is added.
- **The readings in section 3 gained definitions:**
  - H3's pooled scope and s_k slope;
  - a "mixed" outcome for H2b(a);
  - no extra seeds for H2b(b);
  - A1's gap and outcomes;
  - E6's stop rule.

  These are the defaults Bart approved with the review. He confirms section 3 before step 1.
- **Smaller fixes:**
  - smoke coverage and re-scoring;
  - the runner's `over_exp` guard;
  - the generic script's scope;
  - two doc notes;
  - GPU overlap during re-scoring;
  - dates and sources.

---

## 1. Where we stand, and what is still open

**Measured so far** (`Rung2_results.md`, log 2026-11-14 to 2026-11-16):
- **H2 supported.** On the varying set, Rung 2 beats `detach` by +2.11 dB bm PSNR pooled (5.3×). The leak is 0.012, against `detach`'s 0.512. The s_k follow the true gains (slope −1.00).
- **H1 not supported** (pre-registered, default `cdf` init). 11 of 12 checks pass. Bike's bm PSNR is −0.155 dB at −3.8× its 0.041 dB noise. Pooled, the two match (−0.02 dB).
- **Identity init (`rung2id`), exploratory.** Confirmed over seeds 0–2: +0.44 dB pooled (6.2×) and +0.26 dB on bike (10.9×). Against `detach` it meets all 12 H1 checks.
  - F does not depend on the seed: the final F is identical across seeds to within 0.1 of a level.
  - The curve loss prefers `cdf` in all 15 pairs.
  - The deep-shadow lift comes from the one-γ shape prior, not from the equalisation target (correction, 2026-11-16).

**Still open:**
1. **H3 has not been run,** and its success criterion is already met by the default (section 3.3), so it has to be restated before any run.
2. **The identity init is untested on the varying set,** both for H2 and for whether its gain carries over to new photos.
3. **Is the identity gain just early stopping?** The log infers that longer training would erode it, but this has not been measured.
4. **Bike's hue on the varying set** is the largest remaining weakness: 35.4° against `detach`'s 12.3°.

**Time:** experiments freeze on Sat 5 Dec, 11 days from now. The report is due 18 Dec. M4 (the 13 Nov check-in) is done: Bart sent Wei the M3 results on 2026-11-24. Rung 3 stays, as M6 after M5 (Bart, 2026-11-24). Overexposure, part of H1's original scope and pending Wei since M4, is now E6 in this plan (Bart, 2026-11-24).

---

## 2. The experiments

| ID | Question | Runs | Root | GPU time (*inferred* from M3's per-run times) | Priority |
|---|---|---|---|---|---|
| **E1 (H2b)** | Does the identity init hold on new photos, and does H2 hold for it? | `rung2id` and `rung2idexp` (new), varying set, seeds 0–2: 30 runs | `results/vexp/` | about 2.4 h | Must |
| **E2 (H3)** | Does 3D feedback into the exposures help, hurt or do nothing? | `rung2att`, LOM and varying set (shared column), seeds 0–2: 30 runs | `results/rung2/`, `results/vexp/` | about 2.4 h | Must |
| **E3 (H1b)** | Does the identity variant meet the H1 rule on fresh seeds? | `detach` and `rung2id`, LOM, seeds 3–5: 30 runs | `results/h1b/` (new) | about 3.1 h | Bart's choice |
| **E4** | What does dropping the shape prior cost? (The ablation planned since 2026-11-13.) | `rung2noprior` (new), LOM, seeds 0–2: 15 runs | `results/rung2/` | about 1.2 h | Should |
| **A1** | Does the identity-init F converge to the `cdf` F if the curve trains longer? | CPU replay only, no 3DGS (new script, step 3) | `results/rung2/replay_long/` | CPU, about 6–7 min (*inferred*), alongside a batch | Should |
| **E5** | Is bike's hue gap from applying F per RGB channel? | Colour-preserving luminance variant (`--rung2_color luminance`, reserved in M2): LOM and varying set, 30 runs | `results/rung2/`, `results/vexp/` | about 1 day of code plus 2.4 h | Only if time allows (section 5) |
| **E6 (H1-over)** | Does Rung 2 match `detach` on LOM overexposure, H1's original second half? | `baseline` (official), `detach` and `rung2` with `--exp_name over_exp`, seeds 0–2: 45 runs. Optional: `rung2id`, 15 runs | `results/overexp/` (new) | about 6.2 h, or 7.4 h with `rung2id` | Should (overnight, Bart's choice) |

**New runner variants** (`run_rung0.py`, add all of them before the first batch, because the runner is running during the batches):
- `rung2idexp`: `["--curve_model", "rung2", "--rung2_F_init", "identity", "--dark_branch", "model"]`
- `rung2noprior`: `["--curve_model", "rung2", "--rung2_prior", "none"]`
- Add both to `NOT_IN_RUNG0`.

`rung2att` and `rung2id` already exist:
- `rung2att` passed a smoke test on the varying set only (M2 part C, 2026-11-13);
- `rung2id` passed one on LOM only (head-start check, 2026-11-15).

Step 2 adds the missing combinations.

**Why E1 uses both columns.** F and s_k are identical in the shared and exp columns, as `rung2` and `rung2exp` were. Both columns let `rung2_analyze.py` keep its analysis logic, with the identity variant swapped in by a new flag (step 3), so H2b is read exactly the way H2 was.

**Why E3 goes to a new root.** `results/rung0` holds Rungs 0 and 0.5 only (rule 6), and `detach` at seeds 3–5 is new. Re-score `results/h1b` with `--ref_root ../results/rung0`, like every other root.

**E6 must have its own root.** Run names (`<scene>_<variant>_seed<seed>`) do not include the exposure type. Pointed at `results/rung2`, the runner would find `buu_rung2_seed0` already finished and silently skip every overexposure run.
- Add `--exp_name {low, over_exp}` to `run_rung0.py`, default `low`, so every existing command is unchanged.
- The runner refuses `over_exp` outside `results/overexp/`, and refuses `low` inside it.
  - "Inside" is a subtree test (`Path.is_relative_to`, Python 3.9), so `results/overexp/smoke_m5/` counts as inside.
- It also refuses `over_exp` when `<data_root>/<scene>/over_exp/` is missing. `LOM_vexp` has none, so every run would otherwise fail in the loader.
- Re-score with `--ref_root ../results/rung0`; the poses are the same, so the low-light reference depth applies.

---

## 3. Pre-registered readings (log them before any run; Bart confirms)

General rules, unchanged: pair by seed; noise = sqrt(mean of the two arms' seed variances); an effect is real above 2× noise; shrub is reported but not used to decide; check pooled verdicts against pooled without shrub, and say so if they differ; report raw and bm PSNR side by side. "Pooled" is the mean over the five scenes per seed, as in M3.

### 3.1 H1b: the identity variant on fresh seeds (E3)

- **Statement:** "Rung 2 with F initialised at the identity matches `detach` on LOM by the H1 rule (plan 6.5), on seeds 3–5."
- **Rule:** identical to H1. 12 checks (buu, chair, sofa, bike × bm PSNR, SSIM, LPIPS), with noise from seeds 3–5 of both arms only.
- **Context only:** the 6-seed view (seeds 0–5).
  - It mixes `detach` code versions: Rung 0's runs from October and today's.
  - M2's regression check (identical step-0 loss, within 1e-5 to step 200) makes pooling them acceptable as context. Say so in the results.
- **Analysis:** the generic script's `h1` mode (step 3). `rung2_headstart.py` cannot do it: it needs `rung2` runs at the same seeds, and seeds 3–5 have none.
- **Label:** pre-registered on 2026-11-24, after H1 failed; the variant was chosen from seed-0 validation results. It does not change H1.
- **What it can and cannot show** (state it in the results):
  - F is seed-independent, so fresh seeds re-test only the 3DGS randomness. Seeds 1–2 already did that once.
  - So H1b mainly turns the exploratory 12/12 into a pre-registered verdict.
  - It cannot rule out that the init was picked to suit these five scenes. E1 is the stronger test of that.

### 3.2 H2b: the identity variant on the varying set (E1)

- **(a) The init effect carries over to new photos.**
  - Measure the identity-row minus default-row main effect over both columns: ½[(`rung2id` − `rung2`) + (`rung2idexp` − `rung2exp`)], with noise over the four cells as in `rung05_analyze.effects`.
  - **Confirmed:** pooled bm PSNR is real and positive, and LPIPS is not worse beyond 2× noise.
  - **Not confirmed:** pooled bm PSNR is within 2× noise.
  - **Reversed:** pooled bm PSNR is real and negative.
  - **Mixed:** pooled bm PSNR is real and positive, but LPIPS is worse beyond 2× noise.
  - Bike is judged the same way.
  - **Analysis:** the generic script's `init` mode (step 3). No existing script computes this effect.
- **(b) H2 holds for the identity variant.** The plan 6.5 H2 rule, with `rung2id` and `rung2idexp` in place of `rung2` and `rung2exp`, against `detach` and `detachexp`.
  - That rule's "between 1× and 2× noise pooled: run seeds 3 and 4" clause (40 runs, about 5.5 h) is replaced by "inconclusive": there is no slot before the freeze.
- **(c) Mechanism.** Report the leak and the s_k slope; expected unchanged, because the init does not touch s_k. A leak above 0.1 pooled would be a surprise to flag.
- **Also report** bike's hue error on the varying set, for the E5 decision.
- **Label:** pre-registered on 2026-11-24. The photos are new (gained) inputs, but the scenes and the ground truth are the ones the init was chosen on.

### 3.3 H3, restated before any H3 run (E2)

**Why it must be restated.**
- The current H3 is "attaching only s_k to the image loss recovers the buu LPIPS lost by detaching, without bringing back the hue flips, and keeps the `detach` gains elsewhere".
- The default already does that. Buu LPIPS on LOM (R1a, *measured*):

  | Official | `detach` | `rung2` |
  |---|---|---|
  | 0.1862 | 0.2026 | 0.1824 |

- So the old success criterion cannot tell `rung2att` from `rung2`.
- Keep the old text in `CLAUDE.md` with a dated note. Nothing has been run, so restating it now is legitimate. Tell Wei.

**Restated H3:** "Sending the image loss into the per-view exposures (F stays detached) changes quality on LOM or the varying set."
- **Comparison:** `rung2att` − `rung2`, paired by seed, seeds 0–2.
  - **On LOM:** all 5 scenes. "Pooled" is the five-scene mean as in M3, checked against pooled without shrub. The per-scene harm check uses the four decision scenes.
  - **On the varying set:** the shared column.
- **Helps:** on either set, pooled bm PSNR higher, or pooled LPIPS lower, beyond 2× noise, with no harm signal.
- **Harm signals:**
  - any decision scene's bm PSNR below −2× noise on LOM (the H1-style check). Only bm PSNR is judged here; SSIM and LPIPS are reported;
  - bike's hue error worse beyond 2× noise on either set (the Rung 0.5 hue-flip signature);
  - on the varying set, a pooled leak above 0.1 (a fifth of `detach`'s 0.51);
  - on the varying set, a pooled s_k slope (mean over scenes per seed) outside −0.9 to −1.1. Per-scene slopes are reported only, because the default's own bike slope is −0.91.
- **Verdict:** "helps" if a help criterion is met and there is no harm signal; "hurts" if any harm signal fires and there is no help; "mixed" if both; "no effect" otherwise.
- **Also report:** s_k drift from the initialisation (RMSE of s_final − s_init), and on the varying set the paired s RMSE against the gains.
- H3 runs on the default (`cdf` init) so that it ablates the pre-registered model.

### 3.4 E4: no shape prior

- **Report** `rung2noprior` − `rung2` on LOM, every metric, plus the dark-end curve values at levels 1–4.
  - "Level" here means F on the 255-bin grid at b = 1–4 (x = b/254), × 255, as in the 2026-11-16 correction.
- **Expectation, from the final preview** (`preview_final`, 2026-11-13, final learning rate): much worse pseudo-targets. Their bm PSNR falls short of `detach` by:
  - 10.85 dB on buu, 7.02 on chair, 12.82 on sofa and 8.42 on bike;
  - shrub is +1.53.
- **Parameters:** K + N, because no γ is created without the prior. That gives 54, 75, 61, 68 and 62 for buu, chair, sofa, bike and shrub.
- No decision depends on it. It is the measured cost of the D1 fix for the report.
- If it unexpectedly helps bike, report that as a finding about the prior. It does not change any default.

### 3.5 A1: long curve replay (CPU)

- **What:** replay the curve training (the `rung2_preview.py` machinery, photos only) for both inits to 50k steps on the LOM photos. Record the F gap and the curve loss every 1k steps.
  - **F gap:** gap(t) = max over the 255-bin grid of |F_id(t) − F_cdf(t)|, both at the same step t.
  - The learning rate decays to 1% over the replay. So a 50k replay models a 50k-step run, not the 10k run continued.
  - **Built-in check:** a 10k replay reproduces the trainer's bike gap (0.051 at 10k) within view-order noise.
- **Reading:**
  - **Early stopping:** gap(50k) < 0.01. The identity gain is then early stopping and would fade with longer training.
  - **Two end points:** gap(50k) ≥ 0.01 and it has stopped shrinking, meaning gap(40k) − gap(50k) < 10% of gap(40k). The curve loss is then flat at the dark end, and where F ends depends on where it starts.
  - **Undecided:** gap(50k) ≥ 0.01 and still shrinking.
  - Report which, with the step at which bike's gap halves.
- **Cost:** about 6–7 min of CPU (*inferred*: `preview_final` replays take about 7.6 s per 10k steps).
  - Time a 1k-step replay of the new mode first, and ask Bart if the estimate is over about 30 minutes.
  - Use `--threads 2` while a batch runs.

### 3.6 H1-over: overexposure (E6)

- **Statement:** "On LOM overexposure, Rung 2 (the default, unchanged from low light) matches `detach` by the H1 rule." Same 12 checks (buu, chair, sofa and bike × bm PSNR, SSIM and LPIPS), with noise from the `over_exp` runs. Shrub is reported only.
- **Input check:** every run's `cfg.json` has `exp_name` `over_exp`. No analysis script has checked `exp_name` so far, because every run was `low`.
- **Also reported:**
  - the official code next to both;
  - `detach` − official. Rung 0.5's detach gain was measured on low light only. If `detach` does not beat the official code here, "matching `detach`" is a weaker bar; say so;
  - `rung2id` − `detach` and `rung2id` − `rung2`, exploratory, if the optional runs are in.
- **Label:** pre-registered on 2026-11-24, before any overexposure run. It is the second half of H1's original scope, and its verdict is reported next to the low-light one, not merged with it.

---

## 4. Steps

### Step 0: check this plan, then stop (done 2026-11-24)
- Read `CLAUDE.md`, this plan, the last three entries of `docs/RESEARCH_LOG.md`, and in `Rung2_results.md` the Q1 confirmation section and "What this means for M5".
- Check the plan against the code: variant flags, roots, the runner's refusals, and the analysis scripts it names. Report errors or gaps, then wait for Bart's OK.
- Branch: stay on `main`. No commits.
- **Done:** the fixes Bart approved are folded into this revision (note at the top).

### Step 1: log the decisions and readings (before any run)
- One `/log-experiment` decision entry with:
  - option B (Bart, 2026-11-24);
  - the H3 restatement and why;
  - the readings of 3.1 to 3.6, as Bart confirms them.
- Update `CLAUDE.md`'s H3 bullet with a dated note, keeping the old text.
- Add dated notes where `CLAUDE.md`'s H1 bullet and `docs/experiment_protocol.md` (lines 8–9 and 94) still say overexposure is pending Wei: it is now E6.
- Propose to Bart (his call): add M3's finished outputs to rule 6 and to the `ask` list in `.claude/settings.json`.
  - The outputs: `results/rung2/summary/`, `summary_seed0/`, `summary_headstart*/`, `preview_final/` and `Rung2_results.md`.
  - Why: `rung2_analyze.py` may write to any `results/rung2/summary*` folder.

### Step 2: code needed before the first batch
- `run_rung0.py`: the two new variants (section 2).
- Smoke tests (rule 3, 300 steps, buu, seed 0), in `smoke_m5` folders under each root. Re-score each one with `--ref_root ../results/rung0`.
  - **Varying set:** `rung2id` and `rung2idexp`. Check `rung2idexp`'s F and s against `rung2id`'s, bit for bit (max |Δ| = 0, as in M3).
  - **LOM:** `rung2att` (never smoke-tested on LOM) and `rung2noprior`. For `rung2noprior`, check:
    - it reports K + N = 54 parameters on buu;
    - γ is NaN in its `rung2_state.json`;
    - its re-score works (there is no `rho` in the state).
- Add the `results/h1b/` and `results/overexp/` roots to `CLAUDE.md`'s repo map.
- **Overwrite guard (Bart, 2026-11-24):** every analysis script M5 writes or changes must refuse to write into a protected folder (rule 6), unless it is a git-ignored scratch folder. That covers `rung2_analyze.py`'s new `--out`, `rung2_ablation.py` and `rung2_replay.py`. The protected folders include M3's `results/rung2/summary/`, `summary_seed0/`, `summary_headstart/`, `summary_headstart_seeds012/` and `preview_final/`. Test each guard once with a refused path before use.
- **E6 before the launch** (all GPU smoke tests must finish before the batch starts, because the GPU is busy afterwards):
  - `run_rung0.py`: `--exp_name` and the root guards (section 2). Its default commands must be unchanged (diff a dry run).
  - **Data (checked in step 0):**
    - Every `over_exp/` holds the same image names as `low/` (lowercase `.jpg` against `.JPG`).
    - It also holds `transforms_train.json`, a copy of the scene's training split with `over_exp/` paths. The loader never reads it: it reads `<scene>/transforms_train.json` and looks up the COLMAP names case-insensitively.
    - The validation map `over_exp` → `high` works on Windows.
    - Chair's extra file in `high/` is a `.DS_Store`.
  - Smoke `baseline`, `detach` and `rung2` on buu `over_exp`, 300 steps, in `results/overexp/smoke_m5/`, and re-score all three. Check:
    - **the training set reads `over_exp/`:** `exp_name` in `cfg.json`, and `mean_input` far above the low-light smokes' (`diagnostics/pseudo_targets.csv`; also `exposure_init.json` for `rung2`);
    - **the curve weight:** a 300-step run cannot show w = 0.1, because `w_cdf` stays 1 until step 3000. Check it in the code instead:
      - the trainer passes `cfg.exp_name` to both curve losses (`simple_trainer_ours.py` lines 754–756);
      - `rung2.curve_weight` and `losses.HistogramPriorLoss` both use 0.1 for `over_exp`;
      - add a CPU unit call, `curve_weight(3000, "over_exp") == 0.1`.
  - **Curve-only preview on the `over_exp` photos** (CPU, 10k steps, default configuration, training photos only, writes `results/overexp/preview/`).
    - **Why new code:** `rung2_preview.py` as it stands reads `low/` and uses w = 0.5. It always runs every configuration on both sets, with the learning-rate ladder and the optima, and scores against `high/`. Its guard refuses folders not named `preview*`.
    - **Code:** add default-preserving `exp_name` parameters to `rung2_preview.load_set`, `replay` and `objective` (default `low`, so `preview_final` is unchanged). Then add a new script, `rung2_replay.py`, with two modes: `overexp` (this preview) and `long` (A1).
    - **No normal-light images:** no P5.
    - **Report:**
      - γ;
      - F against the identity;
      - the fit residual, next to the low-light preview's pooled residual;
      - s_k movement from the init, and |mean s|;
      - per scene, the share of pixels at level 255 and the mean CDF[253]. That is the shifted target's top value, while F(1) = 1 is pinned.
    - **Stop rule:** leave E6 out of the launch and ask Bart if any of these holds:
      - γ stays at or below 1;
      - F(x) < x at fewer than 80% of the interior grid points (b = 1–253);
      - the pooled fit residual is above 2× the low-light preview's.

      The curve has to darken here. A D1-style design mismatch should be caught before 6 GPU hours, not after.

### Step 3: batch 1, E1 + E2 (about 4.8 h)
Bart starts both batches as one launch with `/run-batch`, option (a), with sleep off: the two commands here, then step 4's two and step 4b's (all six in one chain), chained for the prompt in use (`;` in PowerShell, `&` in the Anaconda Prompt). From `Luminance-GS/examples`, in this order:
```
python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants rung2id rung2idexp rung2att --seeds 0 1 2
python run_rung0.py --root ../results/rung2 --variants rung2att --seeds 0 1 2
```

**While it runs** (CPU only; do not edit any file the trainer imports):
- **E1:** add `--rung2_variant {rung2, rung2id}` and `--out` to `rung2_analyze.py`. With `rung2id`, the Rung 2 cells on LOM and the varying set become `rung2id` and `rung2idexp`; write to `results/rung2/summary_rung2id/`.
  - **Also change:**
    - add `EXPECTED` entries for `rung2id` and `rung2idexp` (F init identity);
    - the tables name the cells `rung2id` and `rung2idexp`, and label the LOM H1 verdict exploratory;
    - the guard accepts a scratch `--out` in a git-ignored `smoke*` folder.
  - **Test before use,** with the default variant and `--out` in that scratch folder (never overwrite `results/rung2/summary/`):
    - every CSV matches the committed folder byte for byte;
    - `tables.md` and `numbers.json` match after masking the "Generated" line and `meta`. They record the date, the seconds and the script's own SHA, which changes with any edit;
    - `--selftest` still gives 0;
    - with `--rung2_variant rung2id`, the LOM H1 table equals `summary_headstart_seeds012/`'s.
- **E2, E3, E4, E6 and H2b(a):** one paired-comparison script, `rung2_ablation.py`, with arms in any roots and one mode per job:
  - **`h3`:** `rung2att` − `rung2` on LOM and the varying set. It covers per-scene harm, bike hue, the paired leak (LOM and varying runs paired, plus `gains.json`), the pooled s_k slope and s drift. Writes `results/rung2/summary_h3/`.
  - **`noprior`:** `rung2noprior` − `rung2` on LOM, with the dark end at grid levels 1–4. Writes `results/rung2/summary_noprior/`.
  - **`h1`:** the H1 rule for A − B, with official alongside. Its input check includes `exp_name`. Two uses:
    - E3: `rung2id` − `detach` in `results/h1b/`, writing `results/h1b/summary/`;
    - E6: see step 4b.
  - **`init`:** H2b(a), the init × dark-branch 2×2 on the varying set (`rung2`, `rung2exp`, `rung2id`, `rung2idexp`), with noise over the four cells. Writes `results/rung2/summary_h2b/`.
  - **Order:** write `h3` and `noprior` first (their runs finish first), then `h1`, then `init`.
  - **Tests before use:**
    - `--selftest` reads the variant from `rung2`, and every effect comes out 0;
    - `h1` pointed at `results/rung2` and `results/rung0` reproduces M3's R1b exactly;
    - `h1` pointed at the old roots reproduces `summary_headstart_seeds012/`'s H1 table.
- **A1:** `rung2_replay.py --mode long`, writing `results/rung2/replay_long/`, if Bart approves the time estimate.

### Step 4: batch 2, E3 + E4 (about 4.3 h), straight after batch 1
```
python run_rung0.py --root ../results/h1b --variants detach rung2id --seeds 3 4 5
python run_rung0.py --root ../results/rung2 --variants rung2noprior --seeds 0 1 2
```
Re-score every root touched with `--ref_root ../results/rung0`. This overlaps E6 on the one GPU:
- check the VRAM headroom with `nvidia-smi` first;
- note which E6 runs overlapped, because their wall times will read high.

### Step 4b: E6 overnight (about 6.2 h, or 7.4 h with `rung2id`), chained after step 4
```
python run_rung0.py --exp_name over_exp --root ../results/overexp --variants baseline detach rung2 --seeds 0 1 2
python run_rung0.py --exp_name over_exp --root ../results/overexp --variants rung2id --seeds 0 1 2
```
The second line is the optional `rung2id`. Re-score with `--ref_root ../results/rung0`.

**One launch:** all six commands run back to back, about 15 to 16.5 h in total (Bart, 2026-11-24). Started about 1 pm, they end about 4 to 5:30 am.
- E1 and E2 come first because they are the must-haves; E6 comes last, so it runs overnight.
- The runner is resumable, so a crash costs only the run in progress.

**E6 analysis.** The generic script's `h1` mode (step 3) applies the H1 rule to `rung2` − `detach` on `results/overexp/`, with official alongside, and writes `results/overexp/summary/`. It checks that every run has `exp_name` `over_exp`. Test it on low light first: pointed at `results/rung2` and `results/rung0`, it must reproduce M3's R1b exactly.

### Step 5: analyse and report
- Run the analysis for E1 to E4, E6 and A1 and apply the readings as logged. Write one log entry per experiment.
- Write `Implementation_guides/M5_results.md` (new), with:
  - H2b, H3, H1b and H1-over verdicts;
  - the E4 and A1 findings;
  - a section "What the final report says", with the side-by-side H1/H1b wording and the overexposure verdict next to the low-light one.
- Leave `Rung2_results.md` as M3's report, apart from a one-line pointer to `M5_results.md`.
- Update `CLAUDE.md`'s rung table and the protocol. Propose the commits.

### Step 6: E5, only if section 5 allows
- Design `--rung2_color luminance`: apply F to a luminance channel and scale RGB by the ratio, so hue is preserved. Default `rgb` reproduces the current code.
- Smoke test, then 30 runs (LOM and the varying set, seeds 0–2).
- Reading, logged before the runs:
  - `rung2lum` − `rung2`;
  - "fixes bike's hue" if bike's hue error on the varying set falls by more than 2× noise with no bm PSNR loss beyond 2× noise in any decision scene;
  - an ablation finding, not a new default.

---

## 5. Schedule (*inferred*; Bart wants M5 finished today, 2026-11-24)

| When | What |
|---|---|
| Tue 24, late morning | Step 0 review (done); Bart confirms section 3; steps 1–2 (log, variants, runner option, the `over_exp` preview script, seven smoke tests of about a minute each with their re-scores, the `over_exp` preview) |
| Tue 24, from about 1 pm | One launch: E1 + E2, then E3 + E4 (about 9 h, to about 10 pm), then E6 overnight (to about 4 to 5:30 am). Meanwhile, the analysis scripts are written and tested, and A1 runs on the CPU |
| Tue 24, late | E1 to E4: re-score (about 20 min), run the analysis, write the log entries, start `M5_results.md` |
| Wed 25 | E6: re-score, analyse, log; finish `M5_results.md`; message Wei; decide E5; write the M6 plan (section 8) |
| Thu 26 – Thu 3 Dec | M6: Rung 3, plus E5 or overexposure if chosen |
| Fri 4 – Sat 5 Dec | Final checks; **freeze on 5 Dec** (M7) |
| 6 – 18 Dec | Report writing (start the outline now, in parallel) |

E5 cannot fit today: it needs about a day of code before its 2.4 h of runs.

**If time runs short,** drop in this order: E5, A1, E6's optional `rung2id`, E4, E3, E6. E1 and E2 are the must-haves: H3 is a stated hypothesis, and E1 decides whether the identity variant is worth reporting at all.

---

## 6. Risks

1. **E1 reverses the init effect on the varying set.** The identity gain is then LOM-specific: report it that way, and the default is unchanged either way.
2. **`rung2att` lets s_k chase the renders,** so the leak rises. That is a legitimate H3 result ("hurts"), not a bug. Check `grad_check` first if it looks extreme.
3. **H1b fails on fresh seeds** (bike's noise is small and itself uncertain). Then report H1b as not supported; the 3-seed exploratory pass stays exploratory.
4. **GPU nondeterminism and sleep:** as before. Never compare single runs, and turn sleep off.
5. **Forking paths.** Every variant here is labelled pre-registered or exploratory in the log and the results, and no default changes in M5.
6. **E6 is the first overexposure run of anything in this project.**
   - The official and `detach` runs could fail or behave unlike low light. The smoke tests and the preview are the guard.
   - Rung 2's knots are dense near black, while overexposed detail sits near white. With w = 0.1 the shape prior dominates, so F should be close to a smooth gamma curve (*inferred*). The preview's fit residual checks this.
   - The one-bin shift makes the target's top value CDF[253], while F(1) = 1 is pinned. That is harmless in low light (CDF[253] ≈ 1), but a misfit on clipped photos (*inferred*; the bike photo viewed in step 0 is washed out but not broadly clipped). The preview reports it.
   - If H1-over fails because of a low-light-specific default (as D1 nearly made H1 fail), report it that way. Do not tune on the overexposure validation results.
7. **The analysis code is on today's critical path:** the `rung2_analyze.py` changes, `rung2_ablation.py` with four modes, and `rung2_replay.py`. If it slips, the batches still run and their results wait. No run is read before its script passes its tests.

---

## 7. For Wei (the next update, after M5; M4's M3 results went out on 2026-11-24)

- **H1b and H2b results,** next to the side-by-side H1 story:
  - the pre-registered H1 is not supported (bike −0.155 dB, against a 0.08 dB allowance);
  - the identity init gives 12/12 (exploratory).
- **The H3 restatement,** and why: Rung 2 already recovers buu's LPIPS (0.1824 against official's 0.1862).
- **Overexposure:** run as E6 on his general go-ahead; its verdict comes in the next update.
- **Rung 3:** it stays, as M6.
- **The shape prior is back in Rung 2** (one γ), if not already mentioned.

---

## 8. After M5: M6 (Rung 3) and M7 (the freeze)

**M6 = Rung 3**, local monotone curves (a per-pixel mix of shared curves). Kept by Bart on 2026-11-24.
- **When to plan it:** write a separate `M6_rung3_plan.md` on Wed 25, once M5's results are in. M5 decides which Rung 2 version Rung 3 builds on: `cdf` or identity init, and attached s_k or not.
- **What the plan needs:**
  - its own question and pre-registered reading. Rung 3 adds parameters back, so it must say what it should fix. Candidates: bike's hue, and the dark end;
  - the design behind a new flag, whose default reproduces Rung 2;
  - unit tests, smoke tests, then LOM and the varying set at seeds 0–2.
- **Budget** (*inferred*): about 7 working days, Thu 26 Nov to Thu 3 Dec. That is enough for design, code and about 30 to 45 runs if the design is settled by Fri 27.

**Overexposure** moved into M5 as E6 (Bart, 2026-11-24).

**E5** (the colour-preserving variant) is also an M6 candidate if E1 leaves bike's hue gap on the varying set in place.

**M7** = the freeze on Sat 5 Dec: final tables and figures, and nothing new after that date.
