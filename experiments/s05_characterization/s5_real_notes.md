# Screen S5-real — Do the synthetic missing-context findings replicate on REAL gaps?

**Setup.** Zero-shot forecasting with chronos-bolt-base and timesfm-2.5-200m on
PhysioNet'12 (tsdm sparse version, sets A+B+C, 11981 ICU patients, hourly grid over
the first 48h, 37 variables, raw units; read from the `~/.tsdm` cache populated by
repos/APN's pipeline — no download fight, so no USHCN fallback was needed). One
window per (patient, channel): context = hours 0–23, forecast = hours 24–35
(H=12; APN's own 36h→3h convention is too short-horizon, and the spec's 48–72h
context does not fit a 48h record). A window is eligible iff all 12 target hours
are observed (natural missingness never enters the target) and ≥ 2 context anchors
exist; ≤ 90 windows sampled per channel → **990 windows over 11 channels** (the
vitals; labs never have 12 consecutive hourly measurements). Fills: zero / ffill /
linear / zero_oscale (zero-fill, then rescale by mean(|x_observed|)/mean(|x_filled|)
per window — the deployable observed-stats version of the synthetic oracle probe).
Errors are compared per channel first (raw scales differ wildly across clinical
variables), then averaged: relMSE vs the per-channel best fill / vs linear.

**Natural missingness.** Per-variable observed fraction on the full 48h grid:
min 0.0017 (Cholesterol), median 0.074, max 0.698 (HR) — i.e. missing rates
30.2% / 92.6% / 99.8% (min/median/max). In the evaluated windows the context is
denser by construction: mean missing 18.7%, median 12.5%, max 91.7% (mean 19.5
of 24 anchors). Gaps are blocky and care-driven (informative), not iid.

## Headline: relMSE per fill (channel-mean; median in parentheses)

| model   | linear | ffill | zero | zero+oscale |
|---------|--------|-------|------|-------------|
| bolt    | 1.000  | 0.973 | 61.2 (2.98) | 176.3 (5.31) |
| timesfm | 1.000  | 0.989 | 65.5 (2.42) | 2119.5 (19.63) |

(relMAE agrees: zero = 6.0×/5.1× vs best fill; ffill ≈ linear within 3–6%.)

## Findings

1. **The fill ranking replicates, including the zero-fill catastrophe.** Decent
   fills are within ~3% of each other (ffill slightly beats linear on step-like
   ICU vitals), while zero-fill is 2.4–3.0× worse at the median channel and
   255–394× worse on Temp/Weight. Natural gaps are ~19% missing, yet damage far
   exceeds synthetic MCAR at p=0.7 (13.7–19.0×) because real gaps are blocky and
   informative, and short L=24 contexts have few anchors to begin with.
2. **Exception that matches the synthetic weather anomaly: Urine.** Zero-fill
   *wins* there (relMSE 0.23 bolt / 0.12 timesfm) — urine output is frequently
   0 mL/h, so zeros approximate the series' resting value. Zero-fill is only
   safe when the series naturally lives at zero.
3. **The observed-stats rescale backfires, as predicted for informative
   missingness.** oscale recovery = (mse_zero − mse_oscale)/(mse_zero −
   mse_linear) = **−137% (bolt), −1472% (timesfm)**; oscale is worse than plain
   zero-fill in 10/11 channels for both models (the exception is Urine: a tie
   for bolt, slightly better for timesfm — the near-zero-baseline channel). Under
   synthetic MCAR the same probe recovered ~80% of bolt's zero-fill damage; under
   synthetic MNAR it backfired (−3% to −225%). Real clinical gaps behave like
   the MNAR case: the damage is missing information + selection bias, not a
   corrupted mean-|x| statistic, and re-inflating the survivors distorts the
   context shape.
4. **Stress test: flat at +10%, mild at +30% — essentially replicates.** Extra
   MCAR on observed points, linear fill: +10% → median relMSE 1.02/1.00 (worst
   channel 1.07); +30% → median 1.11/1.14, worst vital 1.38 (achieved total
   missing ≈ 43%). The channel-mean blow-up (2.6×/5.6×) is entirely Weight, a
   static admission-value channel whose near-zero baseline error makes the ratio
   degenerate (17.5×/49.8×) — the same denominator instability as synthetic
   weather. The mild rise above the synthetic ~1.0–1.05 is expected: a 24h
   context holds ~20 anchors, so +30% MCAR removes a third of a handful of
   points (≤2 windows lose all anchors).
5. **The MNAR extreme-value signature does NOT appear in natural gaps.** Linear
   fill, MSE on future points above the observed-context q90 vs the rest: ratio
   0.92 (bolt), 0.64 (timesfm) — vs 2.4 under synthetic mnar_high. The channels
   that can pass a complete-target filter are routinely-measured vitals whose
   gaps are schedule-driven, not value-censored; the plausibly-MNAR labs
   (Troponin, Bilirubin, …) are excluded by design (never 12 hourly
   observations in a row).

## Caveats

- Window eligibility selects densely-monitored (presumably stabler) patients and
  routinely-measured channels — the evaluated missingness is the *benign* slice
  of clinical missingness; lab-channel MNAR is untestable under a
  complete-target rule.
- Convention deviation: context 24h → horizon 12h (P12 records are 48h total);
  990 windows, 2 mask seeds (stress only), t5 not run (spec: bolt + timesfm).
- Channel-mean relMSE is sensitive to near-zero-denominator channels (Temp,
  Weight static); medians reported alongside. Raw P12 units; models see filled
  contexts only, targets never filled.
- timesfm = 2.5-200m (only checkpoint shipped by the pip package), default
  compile flags as in the synthetic study.

Artifacts: `run_s5_real.py`, `analyze_s5_real.py`, `s5_real_results.json`
(per-window data + window metadata), `s5_real_analysis.json`, `s5_real.png`,
`s5_real_run.log`.
