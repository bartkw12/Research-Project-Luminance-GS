# Rung 0.5: what it means for Rungs 1, 2 and 3

Interpretation of `Rung0.5_results.md` (Oct 24). Numbers are quoted from that report or from the summary tables (`results/rung0/summary_rung05/tables.md`, `results/rung0/summary/tables.md`). *Inferred* marks my reading of them. Everything else is either a measured result or follows directly from the code. Section 5 lists the corrections made when this was reviewed against those tables.

## Verdict

The report's analysis is sound and follows the plan's decision rules. Rung 2 should use:

- a **fixed RGB basis**;
- a curve trained **from the priors only** (detached);
- the **`detach` cell as the reference baseline**, with the official configuration also kept in every table.

H3 is rejected as written, so it needs replacing.

The results also expose one consequence that the report does not spell out. It changes how Rung 2 should be framed and adds one important ablation (section 2).

## 1. What the results establish

| Question | Answer | Strength |
|---|---|---|
| Was the confound real? | Yes. The basis stayed at exactly 0 in every detached run. | Measured, all 15 runs |
| Where does the detach gain come from? | Mostly from cutting the curve's image gradient: 85–100% where the gain is real. Freezing the basis explains 10–42% pooled, and the share depends on the order of comparison. | Real at 5–27× seed noise |
| What causes the bike hue flip? | The curve chasing the render, not the colour basis. Freezing the basis leaves the red wheel. Both detached cells fix it. | Pseudo-target hue error 20.7° → 10.8°. The images are from seed 0 only. |
| Does the learned basis do anything useful? | Not really. It rotates 0.02–0.49°. Resetting it to RGB improves pseudo-targets in every scene, by +0.16 to +0.93 dB. | Measured counterfactual, not retrained |
| What does detaching cost? | Detached pseudo-targets are less consistent across views (0.97–1.72× the ground-truth level, vs 0.69–1.29× for the baseline). buu LPIPS gets worse (+0.026, 4.5×). Raw PSNR drops (curve effect −0.82 dB pooled, 3.1×). Saturation is **not** a cost: detaching the curve raises the render saturation ratio on buu, chair and sofa (+0.10 to +0.28, 12–22×), and bike's drop (−0.14) is only 1.6× noise. | Measured; consistency with the shared reference depth |

## 2. The consequence not stated in the report: detached means 2D-only

In detached mode the curve receives gradient only from `L_hist`. `L_hist` depends on the curve, the dark photo and the prior network, and on nothing rendered (section 6.6 of the explainer).

The detached pseudo-targets are therefore a function of each photo, its pose and the training step only. **No information from the 3D model reaches them.** In effect, the winning `detach` cell is "2D histogram-based enhancement, then 3DGS". The paper argues against exactly that recipe, although here the enhancement is still tied together by one shared curve and the priors.

This changes Rung 2 in three ways:

1. **The default Rung 2 curve could be fitted before training.** F and s_k could be fitted from the training photos' histograms alone, which makes the method simpler and faster than it looked. The framing becomes "the right view-consistent enhancement model": 3D consistency comes from one shared F with exposure alignment, instead of from joint training.
2. **Cross-view consistency will not come from the structure alone.** Detaching raised pseudo-target inconsistency to as much as 1.7× the ground-truth level. But the per-view curve variation is already one-dimensional and tiny in *both* modes (Rung 0: PC1 ≥ 99.2%; RMS deviation from the scene-mean curve 1.4–4.5/255 attached, 1.5–4.9/255 detached). So the extra inconsistency does not come from per-view freedom, and a shared F with one scalar per view will not remove it by itself.
   - *Inferred:* the difference lies in the shared curve's shape. The attached curve is reshaped by the render (Rung 0's reading), which can damp how much each photo's noise is amplified near black. The detached curve follows only the HE-CDF and the prior. F's dark-end slope (Rung 1 note below) is the lever a detached F has.
   - Track consistency as a diagnostic, not a headline metric. It is not a quality proxy: `detach` has the least consistent pseudo-targets and the best renders.
3. **New key ablation: detach F, attach s_k.** With interpolation, the image loss can reach the exposures without reaching the curve. That puts 3D feedback back in through only one number per view, so the target cannot "chase" the render the way a 255-value curve can.
   - *Inferred:* this is the most likely fix for the buu LPIPS regression. It is less likely to restore consistency, because one scalar per view cannot change the curve's shape (item 2). The hue flip should not come back, because one gain per view cannot recolour a single wheel.
   - *Inferred:* it matters most on the varying-exposure set. There a histogram-only s_k can confuse a view's content with its exposure, and the render can tell them apart. Run it there as well as on LOM.
   - Use this as the replacement for H3.

## 3. Changes per rung

**Rung 1 (already folded into Rung 2).** No change: monotone, K = 32, F(0) = 0, knots dense near black, interpolated. One addition: with the curve detached, F is fitted only to the CDF targets plus smoothness. Check that the smoothness weight does not let F follow the HE-CDF's steep jump at the dark end. That jump posterises the targets and compresses colour. In the detached default nothing else can change the curve's shape, so this is also the main lever on pseudo-target noise and consistency (section 2, item 2).

**Rung 2.** Updated defaults and comparisons:

- Defaults: fixed RGB basis, detached F, free s_k with mean 0. Drop the bias network, the shape network and the colour basis.
- Reference baseline: the `detach` cell, pooled: 23.25 dB brightness-matched PSNR, 0.270 LPIPS, 5.15° angle error, 18.33 dB raw PSNR. Keep the official configuration as "official".
- Ablations, in priority order:
  1. attached s_k (above), on LOM and on the varying-exposure set;
  2. a colour-preserving variant, out = rgb · F(2^{s_k}·Y)/Y with Y the luminance;
  3. K and knot spacing;
  4. fully attached F (the `frozenbasis`-style variant, which was best on buu and on raw PSNR).
- Variant 2 is *inferred* to help with paleness. Paleness is not a cost of detaching (section 1), but every cell is less saturated than the ground truth. The `detach` cell's saturation ratio is 0.50 on chair and bike, 0.63 on shrub, 0.93 on buu and 0.98 on sofa. A steep per-channel curve squeezes colour ratios (explainer, 6.3 worked example). Applying it to luminance and scaling RGB keeps hue and saturation fixed by construction, apart from clipping in highlights.
- **One correction to the report:** it says Rung 2's exposure model will fix overall brightness (bike renders 0.61 vs ground truth 0.49). It won't. With the mean of s_k fixed at 0, the exposures are relative only. Overall brightness is still set by the CDF prior and `L_spa` (which pulls the mean towards 0.5). No unsupervised model knows the true brightness, so brightness-matched PSNR stays the primary metric.

**Rung 3.** Demote it further. Its case rested on global curves limiting colour. Rung 0.5 shows the hue flip came from curve chasing, which detaching fixes. Paleness is a colour-compression problem that local brightness curves don't fix. Local curves would also add per-pixel freedom, which tends to reduce cross-view consistency. That is the metric detaching already hurt. If time allows, spend the stretch slot on the colour-preserving variant and the attached-exposure ablation instead.

**Hypotheses, revised.**

- **H1** (unchanged in spirit): Rung 2 matches the `detach` reference on LOM within the detectable effect, with about 255k fewer parameters.
- **H2** (unchanged): on the varying-exposure set, Rung 2 beats the reference, and s_k tracks the true gains.
- **H3 (new):** attaching only the per-view exposure to the image loss recovers the buu LPIPS lost by detaching, without bringing back the hue flips, and keeps the `detach` gains elsewhere. Pseudo-target consistency is reported, but little change is expected (section 2, item 2).
- Report the old H3 as answered: the colour basis is not the cause.

## 4. Checks on the report's caveats

- **Hue-flip evidence.** The wheel images come from seed 0, one view. The one supporting number is the pseudo-target hue error on training views. On bike, the validation hue metric ranks the cells the other way, because the detached bike renders are desaturated (0.50 vs 0.77; in buu, chair and sofa detaching raises saturation instead). The report's saturation floor (agreed before implementation) is the right fix, and the report is honest about the metric's limits.
  - Before showing this to Wei, render the bike 2×2 for seeds 1 and 2. A re-score takes about 9 s.
  - For Rung 2, add a colour metric that stays stable on pale images: chroma error in CIELAB (a\*, b\*) after brightness matching, reported next to the saturation ratio.
  - Optionally, hand-mask the bike wheels in a few test views to get a direct "flip" metric.
- **buu exception.** By the plan's rule, buu is correctly left out of the "share of gain" calculation, because its detach gain was not real in Rung 0 (+0.35 dB, 1.2×). But the LPIPS regression from detaching is real (4.5×), and `frozenbasis` is buu's best cell. So detached is not free on buu, and the attached-exposure ablation is the planned test for it.
  - buu is the brightest scene (measured, bin coverage in `results/rung0/summary/tables.md`). Its highest used bin averages about 62, with 99% of pixels below bin 52. For the other scenes these are 35–58 and 25–36. *Inferred:* it may benefit most from image feedback because the HE prior fits it worst.
- **Shrub.** Neither factor has a real effect: seed noise is 0.75 dB, the largest of any scene. Treat shrub as uninformative in all comparisons and say so in the report.
- **Order-dependent shares.** Shares swing between −29% and 42% depending on which factor is counted first. That means the share statistic is unstable here. In the report and the presentation, lead with the factorial main effects and the interaction, and cite shares only as a range.

## 5. Review notes (Claude Code, 2026-10-24)

Every number was checked against the summary tables. Changes from the first version:

- **Section 1, cost of detaching.** Removed "renders are paler (saturation ratio 0.71 pooled, 0.50 on bike)". 0.71 is the `detach` cell's level; the baseline's is lower (0.62). Detaching the curve raises saturation on buu, chair and sofa, at 12–22× noise. Added the raw PSNR cost. Source: `summary_rung05/tables.md`, tables 1 and 2a (`sat_ratio`, `psnr`).
- **Section 2, item 2.** Withdrew "a shared F should improve consistency; make it a headline metric". Per-view curve variation is equally small with and without detach. Source: `summary/tables.md`, Q2 PCA table (`pc1_frac`, `total_std_x255`).
- **Section 2, item 3, and H3.** Softened the consistency prediction. Added the varying-exposure set as where attached s_k matters most.
- **Section 3, Rung 1.** Noted that F's dark-end slope is the lever on pseudo-target noise.
- **Section 3, Rung 2.** Reframed paleness as a weakness of every cell, with the `detach` cell's saturation ratios (table 1).
- **Section 4, buu.** Corrected the bin numbers from "68, against 45–60". Source: `summary/tables.md`, bin coverage table (`top_bin_hit_mean`, `bin_99pct_mean`).
- **Checked and unchanged:**
  - the 2D-only argument (`losses.py:HistogramPriorLoss.forward` takes the curve, the dark photo and the prior, and nothing rendered);
  - the brightness correction (`L_spa` trains only the Gaussians);
  - the reference-baseline numbers and the rest of section 1.
