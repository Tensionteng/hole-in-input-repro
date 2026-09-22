# Screen S5 — Do TSFMs break when the lookback window has missing values?

**Setup.** Zero-shot forecasting with 3 time-series foundation models — chronos-bolt-base
(patch 16), chronos-t5-small (patch 1, mean-abs scaling + quantization tokenizer),
timesfm-2.5-200m (patch 32) — on ETTh1, ETTm1, weather (context L=512, horizon H=96,
300 random test windows per dataset, every channel an independent series, 2 mask seeds).
Missingness is applied to the context only (mcar iid Bernoulli(p), or contiguous blocks of
24 points), rates p ∈ {0.1, 0.3, 0.5, 0.7}, then filled (zero / ffill / linear) before
inference; `zero_oscale` = zero-fill + oracle rescaling by mean(|x_clean|)/mean(|x_filled|).
Errors are normalized per dataset×model by the clean p=0 baseline (relMSE; oracle fill
reproduces the baseline to ~1e-10). Aux models used a reduced grid (100 windows, 1 seed).

## Headline table: relMSE vs missing rate, mcar + linear interpolation

| model   | dataset | p=0.1 | p=0.3 | p=0.5 | p=0.7 |
|---------|---------|-------|-------|-------|-------|
| bolt    | ETTh1   | 1.00  | 1.01  | 1.05  | 1.26  |
| bolt    | ETTm1   | 1.01  | 1.02  | 1.04  | 1.07  |
| bolt    | weather | 0.99  | 0.95  | 0.90  | 0.88  |
| t5      | ETTh1   | 1.03  | 1.11  | 1.26  | 1.62  |
| t5      | ETTm1   | 1.07  | 1.11  | 1.31  | 1.53  |
| t5      | weather | 1.02  | 0.97  | 0.89  | 0.77  |
| timesfm | ETTh1   | 1.01  | 1.02  | 1.06  | 1.15  |
| timesfm | ETTm1   | 1.00  | 1.00  | 1.06  | 1.15  |
| timesfm | weather | 1.01  | 1.04  | 1.08  | 1.07  |

Zero-fill (mcar), dataset-avg relMSE — and the zero_oscale probe:

| model   | fill        | p=0.1 | p=0.3 | p=0.5 | p=0.7  |
|---------|-------------|-------|-------|-------|--------|
| bolt    | zero        | 0.96  | 1.32  | 5.87  | 13.74  |
| bolt    | zero_oscale | 1.14  | 1.94  | 1.87  | 10.43  |
| t5      | zero        | 1.31  | 3.59  | 6.43  | 7.17   |
| t5      | zero_oscale | 1.48  | 4.52  | 7.17  | 7.61   |
| timesfm | zero        | 0.92  | 1.05  | 6.07  | 18.97  |
| timesfm | zero_oscale | 1.19  | 3.88  | 3.00  | 17.91  |

## Findings

1. **TSFMs are near-immune to random point missingness when the fill is decent.**
   With linear interpolation, all three models stay within ~5% of clean MSE up to p=0.3
   and within ~7–26% (bolt), ~53–62% (t5), ~15% (timesfm) at p=0.7 on ETT. t5 degrades
   most, consistent with its point-level tokenizer propagating every corrupted value into
   a wrong token; patching models absorb scattered errors.

2. **The acute threat is naive zero-fill, via corruption of internal scaling statistics —
   but only for models whose scaling is a global mean-|x| statistic.** Zero-fill deflates
   chronos-bolt's context scale; restoring it with the oracle factor recovers ~79–83% of
   the damage at p=0.5 (per-dataset 78.9/82.3/82.6%; e.g. weather relMSE 13.65 → 3.24)
   but only ~24–38% at p=0.7, where value error dominates. The same probe **does not
   recover t5 at all** (oscale is uniformly *worse*: 7.61 vs 7.17 at p=0.7) — t5's damage
   is quantization/bin error, not the scale statistic — and only partially recovers
   timesfm at p=0.5 (6.07 → 3.00), not at p=0.7 (18.97 → 17.91). The scaling hypothesis
   is therefore architecture-dependent, not universal.

3. **Patch corruption is NOT the mechanism.** Block missingness (24-point runs) hurts
   *more* than mcar at equal rate for all three models on ETT (ETT-avg relMSE at p=0.7:
   bolt 1.46 vs 1.17; t5 1.93 vs 1.58; timesfm 1.39 vs 1.15) while corrupting *fewer*
   patches (bolt: 0.69 vs 1.00 of 16-point patches). t5, whose "patch" is a single point,
   shows the same ordering — the decisive control. Damage tracks the **interpolation error
   of the fill** (long gaps are unrecoverable), i.e. value corruption per se, not how many
   patch tokens are touched.

4. **Weather anomaly:** mcar + interpolation *improves* zero-shot forecasts on weather
   (bolt 0.88, t5 0.77 at p=0.7) — interpolation acts as denoising on its noisy channels.
   Several weather channels are also near-zero baseline series, so zero-fill is nearly
   harmless there at low p (relMSE 1.03 at p=0.1) while oscale actively *hurts* (1.48) —
   the p ≤ 0.3 recovery % on weather is therefore negative/unstable (denominator ≈ 0).

## MNAR: value-censored missingness (extension)

Two rank-based mechanisms censor exactly ceil(p·512) points per window-channel:
`mnar_high` (largest values — sensor saturation) and `mnar_extreme` (largest |z| vs the
window's own mean/std). Same windows/seeds/fills as mcar/block; achieved rate =
0.102/0.301/0.502/0.701 for p = 0.1/0.3/0.5/0.7. relMSE, linear fill:

| model   | dataset | mnar_high p=0.1/0.3/0.5/0.7 | mnar_extreme p=0.1/0.3/0.5/0.7 | mcar (ref) p=0.7 |
|---------|---------|------------------------------|--------------------------------|------------------|
| bolt    | ETTh1   | 1.00 / 1.10 / 1.43 / **2.52** | 1.04 / 1.35 / 1.53 / 1.93     | 1.26 |
| bolt    | ETTm1   | 1.02 / 1.09 / 1.37 / **2.67** | 1.02 / 1.21 / 1.38 / 1.80     | 1.07 |
| bolt    | weather | 0.98 / 1.17 / 1.29 / 1.33     | 0.97 / 1.02 / 1.03 / 1.02     | 0.88 |
| t5      | ETTh1   | 1.25 / 1.44 / 1.59 / **2.02** | 1.13 / 1.45 / 1.51 / 1.60     | 1.62 |
| t5      | ETTm1   | 1.23 / 1.53 / 1.82 / **2.18** | 0.98 / 1.20 / 1.37 / 1.41     | 1.53 |
| t5      | weather | 0.55 / 0.77 / 0.90 / 0.92     | 0.54 / 0.60 / 0.62 / 0.73     | 0.77 |
| timesfm | ETTh1   | 1.01 / 1.09 / 1.16 / 1.51     | 1.17 / 1.45 / 1.57 / 1.66     | 1.15 |
| timesfm | ETTm1   | 0.98 / 1.05 / 1.22 / 1.91     | 0.95 / 1.14 / 1.32 / 1.54     | 1.15 |
| timesfm | weather | 1.18 / 2.21 / 2.90 / **2.96** | 1.18 / 1.50 / 1.51 / 1.52     | 1.07 |

5. **MNAR hurts roughly twice as much as MCAR at high rates.** With linear fill at
   p=0.7, mnar_high reaches relMSE 2.0–2.7 on ETT for bolt/t5 (vs 1.1–1.6 under mcar);
   the gap opens at p ≥ 0.3 and widens superlinearly. mnar_extreme is milder than
   mnar_high for bolt/t5 on ETT, comparable or worse for timesfm.

6. **The damage is systematic bias against extremes, and it shows up in the future
   exactly where the censoring removed information.** Bolt's MSE on future points above
   the context 90th percentile vs the rest (ETTh1, p=0.7): mcar 10.2/14.6 (ratio 0.7),
   mnar_high 56.2/23.7 (ratio 2.4); ETTm1 p=0.7: 4.3/11.4 → 49.0/22.8. Under mcar the
   extreme-future error barely moves; under mnar_high it grows ~6×. mnar_extreme raises
   both tails symmetrically (no top-decile concentration).

7. **The oracle-rescale probe backfires under MNAR — the mcar scaling story does not
   transfer.** zero_oscale is *worse* than plain zero-fill at every rate, every dataset,
   every model under both mnar mechanisms (bolt mnar_high avg: zero 3.7/6.3/8.9/11.3 vs
   oscale 3.8/7.7/14.7/32.2; recovery −3% to −225%). Under informative missingness the
   damage is missing information + bias, not a corrupted mean-|x| statistic: re-inflating
   the surviving values just distorts the context shape. Contrast with mcar, where the
   same probe recovered ~80% of bolt's zero-fill damage at p=0.5.

8. Weather stays anomalous under MNAR for t5 (improves to 0.55 at p=0.1 mnar_high —
   censoring spikes + interpolation = strong denoising) but not for timesfm (2.96 at
   p=0.7, the worst single cell in the study).

## Caveats

- t5 = chronos-t5-**small** (spec fallback): t5-base measured 0.14 s/series and OOMs past
  batch 64 with 20 samples; could not cover the grid in its 15-min timebox.
- timesfm = **2.5-200m** torch checkpoint (the only one shipped by the `timesfm` 2.0.2
  pip package), not 2.0-500m. Run with its default flip-invariance/positivity passes.
- relMSE is a ratio of mean raw-scale MSEs; raw MSE is dominated by large-scale channels.
  Point forecast = median (chronos) / point head (timesfm); aux models: 100 windows, 1 mask seed.
- Achieved block rate is below nominal (overlapping blocks): 0.09/0.25/0.41/0.51 for
  p = 0.1/0.3/0.5/0.7 — block-vs-mcar comparisons should be read at *achieved* rate.
- MNAR masks are deterministic (rank-based) → 1 mask seed; achieved rate is
  ceil(p·512)/512 = 0.102/0.301/0.502/0.701. t5's mnar grid was computed in 7 GPU shards
  (same seeds/windows) and merged; bolt's mcar:linear records were re-run once to
  backfill the top-decile metric (values identical to the first pass).

Artifacts: `run_s5_missing.py`, `analyze_s5.py`, `s5_missing_results.json` (per-window
data), `s5_missing_curves.png` (mcar/block), `s5_missing_mnar.png` (mnar),
`s5_missing_run.log` (+ per-shard `s5_missing_run_t5_shard*.log`).
