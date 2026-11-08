# M1: build the varying-exposure LOM set and run the official and `detach` baselines on it

Plan for Claude Code on Bart's PC. Written 2026-11-08 from the approved proposal (slides 20, 22 and 24 of the storyboard, rationale section 8) and an audit of the code on `main` (commit `52e03fe`; the files M1 touches match Bart's working copy, checked by file size allowing for Windows line endings).

**Reviewed 2026-11-08 by Claude Code on Bart's PC** against the code, the data and the Rung 0 run folders (read-only checks, plus an in-memory run of the `--verify` logic on all five scenes). The corrections are folded in below: the brightness-ratio tolerances (step 1, step 3), the all-scene clipping and rounding numbers (section 2), the per-scene reference table and a stronger H2 caveat (step 1), a geometric-mean initialisation and a manifest merge in `build_vexp.py`, two guards that protect `results/rung0/` (step 2, section 5), a `.gitignore` rule, and the T4/T5 spec changes (section 5). Slides 9, 13, 20, 22 and 24 were then checked from screenshots Bart provided (the PC has no PDF reader). Slide 20's recipe matches the gain model, the clipping, the re-quantisation, the unchanged test views and ground truth, and the saved u_k. Its "LOM range" labels (0.63, 0.31, 0.31, 0.29, 0.40 stops) come from the Rung 0 snapshots relative to the geometric mean. But it says "draw u_k uniformly", i.e. independent draws, as do the explainer and the protocol, so the stratified sampling below is a refinement to mention to Wei. Slide 22's rules (pair by seed, 2× noise, official / `detach` / Rung 2 in every table) match section 5. Slide 24 dated M1 to 30 Oct.

In the repository this file lives at `Luminance-GS/Implementation_guides/M1_vexp_plan.md`. The dataset builder it refers to is `Luminance-GS/examples/build_vexp.py` (new, untracked). A copy of both is in the project's shared files under `docs/`.

**Rules (from `CLAUDE.md`, they override anything below):** never `git commit`, `push`, merge, rebase, reset or switch branches; ask Bart before creating a branch (suggested name `m1-vexp`); ask before any job longer than about 30 minutes, and remind him to stop the PC sleeping before anything over an hour; new behaviour goes behind an option whose default reproduces the current code; log every step with `/log-experiment`; never touch `results/rung0/summary/`, `summary_rung05/` or the finished reports. Run Python with the `luminancegs` interpreter from `CLAUDE.local.md`.

---

## 1. M1 in plain terms

**What it is.** A second version of the LOM dataset in which every *training* photo has been made brighter or darker by a known amount, plus the two existing methods (official Luminance-GS and our `detach` reference) trained on it.

**Why it is needed.** The real LOM photos of a scene are all exposed almost the same (they vary by only 0.29 to 0.63 stops within a scene; slide 20). The Rung 2 idea, one shared curve plus one exposure number s_k per photo, can therefore at best *tie* on LOM (that is H1). H2 claims that Rung 2 *wins* when exposure really varies, and that its learned s_k recover the true exposures. Testing that needs data where exposure varies and where the true per-photo exposure is known. Synthetic gains on LOM give exactly that, and the normal-light ground truth stays valid because only the dark training inputs change.

**Why the baselines run first (this milestone) and Rung 2 later (M3).** M1 sets the bar Rung 2 has to beat, and measures how much a 2-stop spread of exposures hurts each existing method. It also checks whether the official per-view curves compensate for exposure at all, which tells us whether H2 is plausible before any Rung 2 code exists.

**Inputs**
- `Luminance-GS/data/LOM_full/<scene>/` for buu, chair, sofa, bike, shrub (read only).
- The existing LOM runs `results/rung0/<scene>_{baseline,detach}_seed{0,1,2}/` (read only, for the LOM-vs-varying comparison).

**Outputs**
- `Luminance-GS/data/LOM_vexp/<scene>/` in the exact LOM layout (git-ignored, rebuildable from the script and seed). The authoritative record of the set is `gains.json`, because a different numpy version may draw a different permutation from the same seed.
- `results/vexp/dataset/`: every u_k, per-view checks, and the "oracle" exposure-recovery numbers (committed).
- 30 runs in `results/vexp/<scene>_<variant>_seed<seed>/`: 5 scenes × {baseline, detach} × seeds {0, 1, 2}, 10k steps, diagnostics on, re-scored.
- `results/vexp/summary/` (tables, figures, numbers) and a short hand-written `Implementation_guides/M1_vexp_results.md`.
- Research-log entries.

---

## 2. Design (agreed in the proposal; ask Bart before changing any of it)

| Item | Choice | Reason |
|---|---|---|
| Gain model | `I'_k = clip(round(2^{u_k} · I_k), 0, 255)` on the 8-bit sRGB values, per pixel and channel | Slide 20 recipe. Exact for linear or power-law encoding, an approximation for real sRGB (caveat for the report). |
| Range | u_k in [−1, 1] stops (½× to 2×) | About 3 to 7 times LOM's natural spread (0.29 to 0.63 stops within a scene on slide 20, from the Rung 0 snapshots of the undistorted views; 0.29 to 0.62 on the raw files, *measured*). At 2× nothing clips: the clipped fraction is 0 in every scene except bike, where it is 1.8e-6, a few hot pixels (*measured* on all five scenes, 2026-11-08). |
| Sampling | **Stratified uniform**: N evenly spaced values `−1 + (2j+1)/N`, in a random order | Still uniform on [−1, 1], but the mean of u_k is exactly 0, so the set keeps LOM's average *log*-brightness (the geometric mean of the gains is 1; the arithmetic mean is 1.08, +0.11 stops) and a LOM-vs-varying difference comes mostly from the *spread*. The range is covered evenly, which gives the best slope estimate for s_k vs u_k; u_k sd is 0.58 to 0.59 in every scene. `--sampling iid` reproduces independent draws if Wei prefers the literal recipe (the explainer and the protocol describe independent draws). |
| Seed | Dataset seed 701, per-scene stream `[701, crc32(scene)]` | One fixed set; all three training seeds see the same photos. Independent of which scenes are built. |
| What changes | Training views listed in `transforms_train.json` only | Validation reads `high/`, so test views never reach training or scoring. |
| What is copied unchanged | `high/`, `colmap_sparse/`, `transforms_{train,test,val}.json`, and the test views in `low/` (byte for byte) | Same cameras, same split, same ground truth, same initial point cloud. The loader needs the test-view files to exist in `low/` even though it never reads them. |
| File format | JPEG, quality 100, no chroma subsampling, original file names | The loader matches files to COLMAP image names (`10.JPG`), so PNG would need loader changes. Re-encoding error measured on buu: 0.49 to 0.77 levels RMS, 50.4 to 54.3 dB against the exact 8-bit image, far below the seed noise of the metrics; the minimum over all five scenes is 50.4 dB (bike 55.2). Logged per view. |
| Folder | `data/LOM_vexp/<scene>/`, run with `--exp_name low` | The trainer then takes exactly the LOM code path; only `--data_dir` changes. The folder name must not contain `low` (see 3.2). |

**Caveat for the report: quantisation (*measured*).** LOM's dark views are very dark: their mean is 19 to 29 levels of 255 on buu, 12 to 15 on chair, 16 to 20 on sofa, 7 to 10 on shrub and 7 to 8 on bike. At a few levels, rounding the gained value to 8 bits is not symmetric, so the realised brightness ratio differs from 2^{u_k} by up to 0.06 stops (bike; per scene in step 1). Halving also merges tonal levels that doubling does not recreate, so the darkened views lose information that the brightened views do not gain. Part of any cost of varying exposure is therefore a quantisation cost, and it is not symmetric in u. `gains.csv` stores the realised ratio per view (`log2_ratio`).

---

## 3. Audit: what the code needs

### 3.1 Changes

| File | Change | Default behaviour |
|---|---|---|
| `examples/build_vexp.py` | **New**, already written and placed. Builds the set, writes the manifest and a preview, and with `--verify` loads every scene through the trainer's own `Parser`/`Dataset` and runs the checks. Tested in the cloud on a copy of buu (build, manifest, preview, byte-identical copies, 'low' guard). The `--verify` core (trainer's `Parser`/`Dataset`, HE-CDF, oracle fit) was run in memory on Bart's PC on 2026-11-08 for all five scenes, with the gains and the JPEG round trip simulated (numbers in step 1); the full `--verify` runs for the first time at step 1. **Changed in the review:** (1) the initialisation uses the geometric mean, `log2(m̄/m_k) = mean(log2 m) − log2 m_k`, so it has mean 0 like Rung 2's s_k (the arithmetic mean shifted it by +0.11 to +0.13 stops; slope and r are unchanged); (2) rebuilding or re-verifying some scenes merges them into `gains.*`/`oracle.*` instead of replacing the other scenes' rows, and it refuses up front, before writing anything, if those rows used another seed, range or sampling; (3) the `low` guard checks every `<dst>/<scene>` path. Merge, guards and dry run were tested in a scratch folder. | n/a (new script) |
| `examples/run_rung0.py` | Add `--data_root` (default `data/LOM_full`), and refuse a non-default `--data_root` when `--root` is the default `results/rung0`. A forgotten `--root` would otherwise put varying-exposure runs into the protected Rung 0 root (rule 6). Diff in step 2. | Identical commands (checked with `--dry_run`). |
| `examples/rescore_runs.py` | Add `--ref_root` (default `--root`) so the varying-exposure runs use the **LOM** `<scene>_baseline_seed0` as the shared reference depth. Cameras are identical, so LOM and varying-exposure consistency numbers then use the same warp and are directly comparable. Diff in step 2. | Identical. |
| `examples/vexp_analyze.py` | **New**: tables and figures for M1. Spec in section 5. | n/a |
| `CLAUDE.md` | Rung table: Rung 2 "in progress (M1: varying-exposure set + baselines)". Repo map: `build_vexp.py`, `vexp_analyze.py`, `data/LOM_vexp/`, `results/vexp/`. "How to run": `--data_root`. | |
| `docs/experiment_protocol.md` | Data section: the set as built (path, stratified sampling, seed 701, JPEG settings, where u_k live, the quantisation caveat). Runs: root `results/vexp/`. Metrics: the exposure metrics from section 5, with a note that slope and r saturate on this set (step 1). Add a correcting note rather than rewriting the planned-recipe text. | |
| `docs/Luminance_GS_Research_Project_Explained.md` | Section 8.11 ("How you will know it worked"), H2 row: add a note that the proposed \|r\| ≥ 0.9 is already met by the initialisation on this set (step 1 table), so the exposure part of H2 is a sanity check and the PSNR comparison carries H2. | |
| `.claude/skills/run-batch/SKILL.md` | Pre-flight step 2 checks only `data/LOM_full`; make it check the runner's `--data_root`. In the finish step, give the varying-exposure re-score command `rescore_runs.py --root <root> --ref_root ../results/rung0`. | |
| `.gitignore` | **Done in the review:** added `Luminance-GS/results/**/rescore/*.png` (re-scoring writes `warp_check.png` and example PNGs into `rescore/`; before, only `results/rung0/*/rescore/` was covered), and fixed the capital P in Bart's `ECE701_Research_Proposal_Bart_Kowal.pdf` rule. That rule had matched only because Windows git ignores case. | |

### 3.2 No change needed (checked in the code)

- `simple_trainer_ours.py`, `datasets/colmap.py`, `diagnostics.py`, `losses.py`, `gsplat/`: the new folder has the LOM layout, so `--data_dir ../data/LOM_vexp/<scene> --exp_name low` works as is. `exp_name` only selects the dataset class (`low` → `datasets/colmap.py`) and the histogram weight after step 3000 (0.5 unless `over_exp`), so both stay as on LOM.
- Facts that constrain the layout:
  - `Parser` reads `colmap_sparse/0`, `transforms_train.json` and `transforms_test.json` from `data_dir`, and finds every image (train and test) in `data_dir/low/` by its COLMAP name.
  - `Dataset.__getitem__` finds a validation image with `image_path.replace(exp_name, "high")`. That replaces *every* `low` in the absolute path, so no folder above the images may contain `low`. `build_vexp.py` refuses such a `--dst`.
  - `diagnostics.load_train_gt` finds a training view's ground truth at `<low>/../high/<name>`, so `high/` must sit next to `low/` (it does).
  - The trainset order is the order of `transforms_train.json`, and `image_id` is the position in it; `build_vexp.py` stores that as `train_index`, and every diagnostics file stores the view name, so u_k can be joined by name.
- `rescore_runs.py` takes `data_dir` from each run's `cfg.json`, so re-scoring finds `LOM_vexp` by itself.

### 3.3 Things in the official model that react to per-view exposure (useful for reading the results)

1. The bias network sees each dark photo and adds a per-view offset to the curve.
2. The histogram-equalisation target is computed per view from that view's photo.
3. The shape-prior network (gamma) sees the photo.
4. The contrast loss uses a per-view degree `0.5 / mean(photo)` (paper Eq. 8).
5. **The dark branch is shared across views.** `colors_low = colors · adjust_k + adjust_b` is per Gaussian, not per view, and `loss_regress_low` asks that one dark render to match every dark photo. With a 2-stop spread it cannot match them all; only the view-dependent SH terms can absorb some of it. *Inferred:* this hurts every method on the varying set, Rung 2 included, because Rung 2 keeps this loss. M1 measures it with `fit_psnr_lowrender_vs_input` (section 5, T5). If it is large, Rung 2 may want `2^{−s_k}` on the dark branch too; that is an M2 design question for Bart, not part of M1.

---

## 4. Steps

### Step 1. Build and verify the set (about 5 minutes, *inferred*)

1. Read `examples/build_vexp.py` and check it against section 2. Change nothing without asking.
2. From `Luminance-GS/examples`:
   - `python build_vexp.py --dry_run` prints the u_k per view; nothing is written.
   - `python build_vexp.py --verify` builds all five scenes into `../data/LOM_vexp/` and writes `../results/vexp/dataset/{gains.csv,gains.json,oracle.csv,oracle.json,preview_<scene>.jpg}`.
   - If `--verify` fails after the build, fix it and rerun with `--verify_only`. A plain rerun refuses to overwrite the built scenes; a rebuild needs `--overwrite`.
3. Checks (the script asserts most of them; report the rest):
   - every scene: train and test counts as LOM (buu 22/3, chair 43/5, sofa 29/4, bike 36/4, shrub 30/5), validation images identical, cameras identical;
   - `log2(mean_out / mean_in)` within about **0.07 stops** of u_k. The darkest scenes deviate most because 8-bit rounding at a few levels is not symmetric (section 2 caveat). Expected maxima, *measured* in memory: buu 0.020, chair 0.046, sofa 0.011, bike 0.060, shrub 0.028. Bike will print the script's "differs from 2^u by up to … stops" note (threshold 0.05); that is expected, not a failure;
   - `clip_frac` 0 or tiny (expected: 1.8e-6 on bike, 0 elsewhere), `reencode_psnr_db` above 45 dB (expected minimum 50.4 dB);
   - look at the preview images and confirm the bottom row gets brighter from left to right.
4. Reference numbers (*measured* 2026-11-08 on Bart's PC, in memory: the trainer's own `Parser`/`Dataset` and undistortion, gains and JPEG round trip exactly as the builder does them, init with the geometric mean). The real `--verify` reads the same JPEG bytes through the same decoder, so it should reproduce these to rounding. "Paired" means minus the same view's value on the original LOM views, which removes the scene's natural brightness differences. RMSE is in stops, of s + u_k after removing the mean, so 0 means exact recovery.

   | scene | ratio error max (stops) | init slope / r | oracle slope / r | paired r, init / oracle | RMSE init / oracle | paired RMSE init / oracle | natural LOM sd (stops) |
   |---|---|---|---|---|---|---|---|
   | buu | 0.020 | −1.04 / −0.952 | −1.08 / −0.968 | −1.000 / −1.000 | 0.193 / 0.170 | 0.006 / 0.039 | 0.20 |
   | chair | 0.046 | −1.04 / −0.991 | −1.09 / −0.995 | −1.000 / −1.000 | 0.083 / 0.083 | 0.013 / 0.043 | 0.08 |
   | sofa | 0.011 | −1.02 / −0.991 | −1.07 / −0.996 | −1.000 / −1.000 | 0.079 / 0.066 | 0.004 / 0.038 | 0.08 |
   | bike | 0.060 | −0.95 / −0.990 | −1.04 / −0.989 | −0.999 / −0.999 | 0.083 / 0.093 | 0.019 / 0.056 | 0.08 |
   | shrub | 0.028 | −0.98 / −0.983 | −1.11 / −0.980 | −1.000 / −1.000 | 0.105 / 0.145 | 0.014 / 0.082 | 0.10 |

   Init = `log2(m̄/m_k)` with m̄ the geometric mean (mean exactly 0). Oracle = fit of the views' HE-CDF targets with one shared F and one s_k (`fit_exposure_form`); it explains 96 to 98% of the cross-view curve spread. u_k sd is 0.58 to 0.59.

   **What this means for H2 (tell Bart and Wei).**
   - *Measured:* the initialisation alone already tracks u_k at r = −0.95 on buu and −0.98 to −0.99 on the other four scenes, because LOM's own spread (sd 0.08 to 0.20 stops) is small next to u_k's (sd 0.59). Paired, r is −0.999 to −1.000 for both the init and the oracle, and the init's paired RMSE (0.004 to 0.019 stops) is just the 8-bit rounding error. The oracle overshoots slightly (slope −1.04 to −1.11), so even a perfect histogram fit is not exact.
   - *Inferred:* the explainer's H2 exposure threshold (\|r\| ≥ 0.9, slope near −1) is met before any learning, and the paired check is ≈ −1 for anything that follows brightness. Exposure recovery on this set is therefore a sanity check, not evidence for H2. The **PSNR comparison against `detach` on this set is the primary H2 test.**
   - The unpaired RMSE has a floor near LOM's natural spread (0.08 to 0.19 stops). How much of that spread is real exposure and how much is scene content is unknown, so a learned s_k that moves away from the init toward −u_k is not unambiguously "better". Report it, but do not lean on it.

### Step 2. Runner options (10 minutes)

`run_rung0.py`:

```python
# docstring, after the --root example
    python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants baseline detach
                                              # M1: the same runs on the varying-exposure set

# in main(), after the --root argument
    ap.add_argument("--data_root", type=Path, default=DATA,
                    help="folder holding <scene>/ in the LOM layout (default: data/LOM_full)")

# replace the scene check
    data = args.data_root.resolve()
    scenes = [s for s in args.scenes if (data / s).is_dir()]
    missing = sorted(set(args.scenes) - set(scenes))
    if missing:
        print(f"[run_rung0] scenes not found under {data}, skipped: {missing}")

# after root = args.root.resolve(): never put runs on another dataset into the protected Rung 0 root
    if data != DATA.resolve() and root == (EXAMPLES.parent / "results" / "rung0").resolve():
        raise SystemExit(f"[run_rung0] --data_root {data} is not LOM_full: pass a --root other than results/rung0")

# in cmd
            "--data_dir", str(data / scene), "--exp_name", "low", "--result_dir", str(run_dir),
```

`rescore_runs.py`:

```python
# in main(), after --runs
    ap.add_argument("--ref_root", type=Path, default=None,
                    help="where <scene>_baseline_seed0 (the shared reference depth) lives (default: --root)")
# after root = args.root.resolve()
    ref_root = args.ref_root.resolve() if args.ref_root is not None else root
# in the loop
        ref = ref_root / f"{scene}_{REF_VARIANT}_seed{REF_SEED}"
```

Check: `python run_rung0.py --dry_run --scenes buu --seeds 0` prints the same command as before the change (`DATA` is already absolute, so the `--data_dir` string is identical). Then `python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp --variants baseline detach --dry_run` lists 30 runs. Both were checked this way in the cloud. Also check that `python run_rung0.py --data_root ../data/LOM_vexp --dry_run`, with no `--root`, refuses. Note that the runner creates `<root>/logs/` even in a dry run; that folder is git-ignored.

### Step 3. Smoke test (about 5 minutes, *inferred*)

```
python run_rung0.py --data_root ../data/LOM_vexp --root ../results/vexp/smoke --scenes buu --variants baseline detach --seeds 0 --max_steps 300
python rescore_runs.py --root ../results/vexp/smoke --ref_root ../results/rung0
```

Pass if: both runs `ok` in `results/vexp/smoke/runs.csv`; each `cfg.json` has `data_dir` under `LOM_vexp`; in `diagnostics/pseudo_targets.csv`, `mean_input` per view divided by the same view's `mean_input` in `results/rung0/buu_baseline_seed0/diagnostics/pseudo_targets.csv` equals `2^{u_k}` to within about 0.03 stops (expected maximum 0.020 on buu, *measured* in memory; this proves the trainer reads the gained images); re-scoring writes `rescore/val_summary.json`. `smoke*` folders are git-ignored.

Then log the dataset with `/log-experiment` (what was built, seed, sampling, the step-1 checks and the oracle table from `oracle.csv`).

### Step 4. ASK Bart, then run the batch (about 5 h 10 min)

Use the `/run-batch` skill (Bart starts it) with
`--data_root ../data/LOM_vexp --root ../results/vexp --variants baseline detach --seeds 0 1 2`.

- 30 runs. Estimate from the LOM means: 15 × 12.6 min (baseline) + 15 × 7.9 min (detach) ≈ 307 min. Runs are seed-major, so seed 0 for every scene and variant is done after about 1 h 45 min, which is enough for a first look.
- Remind Bart to stop the PC sleeping (`powercfg /change standby-timeout-ac 0`, restore afterwards). Rung 0 lost 11 hours to sleep.
- Bart launches it himself (option (a) of the skill), output to `..\results\vexp\batch_console_m1.log`.
- Do not edit `simple_trainer_ours.py`, `diagnostics.py` or the dataset while it runs. M2 coding on a separate copy or after the batch.

### Step 5. Re-score (a few minutes; Rung 0.5 took about 4 s of scoring per run, *measured*)

`python rescore_runs.py --root ../results/vexp --ref_root ../results/rung0`

### Step 6. Analysis: write and run `vexp_analyze.py` (section 5)

### Step 7. Report and log

1. Write `Luminance-GS/Implementation_guides/M1_vexp_results.md` by hand from `results/vexp/summary/`: direct answers to Q1 to Q4 below first, every claim labelled *measured* or *inferred*, raw and brightness-matched PSNR side by side, effects called real only above 2× seed noise, shrub reported but not used to decide, and a closing "what this means for Rung 2".
   - **Q1.** How much does the 2-stop spread cost each baseline (varying − LOM, per scene)?
   - **Q2.** Is `detach` still the stronger reference on the varying set, i.e. what is the bar for H2?
   - **Q3.** Do the official per-view curves compensate for exposure (fitted s_k and pseudo-target brightness against u_k), and does `detach` behave differently?
   - **Q4.** How much does the shared dark branch suffer (paired low-branch fit, varying − LOM, against signed u_k, darkened and brightened views separately)?
2. `/log-experiment` entry with the results.
3. Update `CLAUDE.md`, `docs/experiment_protocol.md` and the explainer's H2 criterion (3.1), and the run-batch skill.
4. **Remind Bart to message Wei** (he asked for this on 2026-11-08) with:
   - the H2 bar (T1 and T3);
   - the fact that the initialisation already meets the H2 exposure threshold, so PSNR carries H2;
   - stratified draws instead of slide 20's independent draws;
   - why slide 20's "s_k must move toward the truth" check is ambiguous on this set (step 1 note).
5. Do not commit. List the files and propose a message, for example:
   `M1: build varying-exposure LOM set (LOM_vexp) and run official and detach baselines on it`.
   Files: `examples/build_vexp.py`, `examples/vexp_analyze.py`, `examples/run_rung0.py`, `examples/rescore_runs.py`, `results/vexp/dataset/`, `results/vexp/runs.csv`, every `results/vexp/<run>/` (`cfg.json`, `stats/`, `diagnostics/*.csv`, `diagnostics/timing.json`, `rescore/*.csv`, `rescore/*.json`), `results/vexp/summary/`, `Implementation_guides/M1_vexp_plan.md`, `Implementation_guides/M1_vexp_results.md`, `CLAUDE.md`, `docs/experiment_protocol.md`, `docs/Luminance_GS_Research_Project_Explained.md`, `docs/RESEARCH_LOG.md`, `.claude/skills/run-batch/SKILL.md`, `.gitignore`. The `.gitignore` drops checkpoints, renders, `.npz`, `tb/`, logs, `smoke*/` and (since the review) `rescore/*.png` everywhere under `results/`, plus all of `data/` and the proposal PDF.

---

## 5. Spec for `examples/vexp_analyze.py`

Same style as `rung05_analyze.py`: `pathlib`, `__main__` guard, import helpers from `rung0_analyze` (`read_csv`, `write_csv`, `md_table`, `mean_std`, `savefig`, `run_name`, `SERIES`) and the exposure fit from `build_vexp` (`fit_exposure_form`, `slope_r`). Write only to `results/vexp/summary/` (`tables.md`, one CSV per table, `numbers.json`, `figures/`). Read `results/rung0/` without writing to it. **`write_csv` and `savefig` default to `results/rung0/summary/`, a protected folder:** always pass `out_dir`, and assert that every output path is under `results/vexp/summary/`.

**Inputs per run** (both roots, variants `baseline` and `detach`, seeds 0 to 2): `rescore/val_summary.json`, `rescore/pseudo_targets.csv`, `rescore/consistency.csv`, `diagnostics/curves_step10000.npz` (local only, git-ignored; `curve_adj`, `bin_hist`, `mean_brightness`, `view_names`), `cfg.json`, and minutes from `runs.csv` (the last `ok` row per run, since retries add rows). u_k and the realised `log2_ratio` from `results/vexp/dataset/gains.json`, joined by view name; oracle and init numbers from `oracle.json`. All these inputs were checked to exist, with these columns, for the 30 Rung 0 runs.

**Checks before any table:** every varying-exposure run's `cfg.json` `data_dir` is under `LOM_vexp`, and its `rescore/val_summary.json["ref_depth_run"]` is under `results/rung0/`. Stop otherwise: T6 needs the LOM reference depth, and a plain `rescore_runs.py --root ../results/vexp` without `--ref_root` would silently use the varying-exposure `<scene>_baseline_seed0` instead.

**Noise and "real":** as `rung05_analyze.effects`: pair by seed; noise = sqrt(mean of the seed variances of the arms being compared); real if |mean effect| > 2 × noise.

| Table | Content |
|---|---|
| T1 | Validation on the varying set, per scene and pooled, per variant: raw PSNR, brightness-matched PSNR (`psnr_gtmean`), SSIM, LPIPS, mean ± sd over seeds. Label `baseline` "official". |
| T2 | Cost of varying exposure: varying − LOM, same variant, scene and seed, for the four metrics; mean ± sd and ×noise. |
| T3 | `detach` − official on the varying set next to the same difference on LOM (paired by seed, ×noise). This is the H2 bar. |
| T4 | Exposure handling, per run then mean ± sd over seeds, with u_k as x: **(a)** `s_fit` from fitting the final per-view curves (`curve_adj`, weights `bin_hist`) with `fit_exposure_form`: slope and r against u_k, plus the paired version `s_fit(varying) − s_fit(LOM)` for the same view and seed; **(b)** \|r\| of the first principal component score of `curve_adj` against u_k; **(c)** pseudo-target leak: slope of `log2(mean_pseudo)` on u_k, and the paired version `log2(mean_pseudo_vexp / mean_pseudo_LOM)`. 0 means the pseudo-targets fully compensate the exposure, 1 means the change passes straight through; **(d)** the same for `mean_render` (enhanced render at the training view); **(e)** cross-view curve spread (×255, as in `build_vexp.fit_exposure_form`) on varying vs LOM; **(f)** reference rows from `oracle.json`: the init and the histogram oracle; **(g)** residual RMSE in stops of s + u_k after removing the mean, unpaired and paired, for `s_fit`, init and oracle. Slope and r saturate on this set (step 1), so this is the column that can separate methods. The paired RMSE is the clean recovery error: init 0.004 to 0.019, oracle 0.04 to 0.08. The unpaired RMSE has a floor near LOM's natural spread; **(h)** sensitivity: (a) and (g) with the realised `u_eff = log2_ratio` in place of u_k (they differ by up to 0.06 stops on bike). |
| T5 | Training-view quality: mean `pseudo_psnr`, `pseudo_psnr_gtmean`, `fit_psnr_lowrender_vs_input` on varying vs LOM. Per view, the **paired** difference (varying − LOM, same view and seed) against **signed** u_k, with the slope and Spearman reported separately for u_k < 0 and u_k > 0. Do not correlate with \|u_k\|: absolute-error PSNR penalises brightened views more (their residual scales with brightness), and darkened views lose tonal levels (section 2 caveat), so \|u_k\| mixes two different effects. |
| T6 | Pseudo-target and render consistency (`rel_l1_pseudo`, `rel_l1_render`, `rel_l1_gt`) on varying vs LOM, both re-scored with the LOM `<scene>_baseline_seed0` depth. |
| T7 | Colour (`sat_ratio`, `ang_err_deg`) varying vs LOM, and minutes per run. |

**Figures** (one idea each, axes with units, official and `detach` in the same colours as in Rung 0.5):
1. Brightness-matched PSNR and LPIPS per scene, LOM vs varying, both variants, one dot per seed.
2. Per scene, `s_fit` against u_k for both variants (seed 0), with the init and the oracle as reference and the ideal slope −1 line.
3. Per scene, paired `log2(mean_pseudo_vexp / mean_pseudo_LOM)` against u_k for both variants, with the 0 (compensates) and 1 (passes through) lines.
4. Paired low-branch fit PSNR (varying − LOM, same view and seed) against signed u_k, all scenes, both variants.

**numbers.json:** the pooled T2 and T3 effects with ×noise, the T4 slopes and residual RMSEs per scene and variant, and the dataset's u_k sd per scene.

---

## 6. Done when

- [ ] `data/LOM_vexp/` has all five scenes and `build_vexp.py --verify` passes; `results/vexp/dataset/` holds `gains.*`, `oracle.*` and the previews.
- [ ] Smoke test passed and the dataset is logged.
- [ ] 30 runs `ok` in `results/vexp/runs.csv`, all re-scored with `--ref_root ../results/rung0`.
- [ ] `results/vexp/summary/` written by `vexp_analyze.py`; `M1_vexp_results.md` answers Q1 to Q4.
- [ ] Research log, `CLAUDE.md`, `experiment_protocol.md`, the explainer's H2 note and the run-batch skill updated; commit list and message given to Bart.

## 7. What M1 hands to M2 and M3

- u_k per view: `results/vexp/dataset/gains.json` (and `data/LOM_vexp/<scene>/exposure_gains.json`), keyed by view name and `train_index`.
- The H2 bar: T3 and T1 (`detach` on the varying set).
- For the H2 analysis in M3, report learned s_k against u_k three ways: raw, against the initialisation `log2(m̄/m_k)` with m̄ the geometric mean (`oracle.json["s_init"]`, mean 0, the same gauge as s_k), and paired with the same view's s_k from the Rung 2 run on LOM. Give the residual RMSE in stops next to slope and r, because slope and r already saturate for the init (step 1). The PSNR comparison against `detach` carries H2.
- The dark-branch question from 3.3, with the T5 numbers.
