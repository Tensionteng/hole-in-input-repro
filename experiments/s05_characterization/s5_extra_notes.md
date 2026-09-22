# Track C — supplementary mechanism experiments for S5 (missing CONTEXT values)

Same conventions as S5 (`run_s5_missing.py`): L=512, H=96, test windows in the last 20%,
univariate per channel, fill-then-feed, relMSE vs the clean baseline. New: 150 windows × 1
mask seed (Part 4: 100), seed base 20250810. New masks: `blockB` = exact-rate disjoint
length-B gaps (+remainder); `mar` = AR(1) latent (phi=0.95) thresholded at its per-channel
(1−p) quantile — temporally clustered, value-independent, nested across rates; `bolt_cal`
= bolt quantile (coverage) runs. Files: `run_s5_extra.py`, `analyze_s5_extra.py`,
`s5_extra_results.json` (merged; shards `s5_extra_p2_timesfm.json`, `s5_extra_p4_moirai.json`),
`s5_extra.png`, logs `s5_extra_p123_bolt.log` / `s5_extra_p2_timesfm.log` / `s5_extra_p4_moirai.log`.

## Q1 — at what gap length does interpolation stop being free? (Part 1, bolt)
Gap length ~24–48 points is the threshold; below it interpolation is free, above it damage
grows. ETTh1 p=0.3 relMSE: 1.00 (B4) → 1.04 (B8) → 1.11 (B16) → 1.12 (B24) → 1.18 (B48) →
1.16 (B96) → 1.12 (B168, a single 154-gap). p=0.1: flat ≈1.0 through B48, ~1.04–1.06 for a
single maximal 51-gap. weather p=0.3: 0.89–0.94 through B24 (denoising, cf. S5 weather
anomaly), crosses 1.0 at B≈48–96, 1.11 (B96), 1.23 (single 154-gap). Non-monotonic tail:
at fixed rate one huge gap hurts slightly *less* than several 48-gaps — damage localizes.

## Q2 — does pure temporal clustering (no value dependence) reproduce block damage or mcar immunity? (Part 2)
It reproduces — and exceeds — block damage. ETT-avg relMSE at p=0.7, bolt: mcar 1.22 /
**mar 2.50** / block24 1.82; timesfm: 1.25 / **2.95** / 2.05 (per-dataset max: ETTm1
timesfm mar 3.38). At p=0.1 mar ≈ mcar ≈ 1.0 (mean run 4.3 pts — below the Part-1
threshold). mar mean run length = 4.3/6.7/9.6/14.9 at p=0.1/0.3/0.5/0.7 with a heavy tail
(max run ~94 at p=0.7) — the long-gap tail does the damage. Conclusion: value dependence
is NOT needed; exogenous clustered missingness suffices. The operative variable is gap
structure → interpolation error, confirming S5 finding 3 (not patch corruption, not MNAR
bias).

## Q3 — does bolt's uncertainty know the input is corrupted? (Part 3)
Under **mcar**: mostly yes — the 80% PI widens slightly (ETTh1 width 4.51→5.34 at p=0.7)
and coverage stays ~flat (0.74 clean → 0.70; all datasets 0.70–0.71 at p=0.7). Under
**mnar_high**: silent failure — coverage collapses (ETTh1 0.66→0.51→0.36→**0.22**; ETTm1
0.20, weather 0.32 at p=0.7) while the PI *narrows* (ETTh1 4.51→3.13; weather 40.2→4.2,
~10×). Coverage of future extremes (top decile) is **0.000 at p≥0.3** — the interval never
contains an extreme future. The censored+interpolated context looks *calmer*, so the model
gets more confident exactly when it should be least confident. (Clean coverage is 0.74–0.75
vs 0.80 nominal: bolt is mildly overconfident even without missingness.)

## Part 4 — Moirai (shipped): moirai-1.1-R-base (91M) via uni2ts, 100 windows
mcar+linear: immune (0.89–1.11, all datasets/rates — strongest of the 4 models).
mnar_high+linear: nearly immune (ETTh1 1.27 at p=0.7, weather ≤1.10, ETTm1 *improves* to
0.60–0.83) — stark contrast with bolt (2.5) and timesfm on ETT. zero-fill: ETT remarkably
robust (mcar:zero ETTh1 1.48 at p=0.7; bolt ≈3.3) but weather catastrophic (mcar:zero 28.5,
mnar_high:zero 21.7 at p=0.7). block(24)+linear: mild (≤1.12 ETT; weather 1.9 at p=0.5).

## Caveats / honest negatives
- 150 windows × 1 mask seed (100 for moirai); the 150-window sample differs from S5's
  300-window one (`rng.choice` depends on k), so Part-2 mcar/block24 references were re-run
  in-file — don't compare absolute MSEs across files.
- Part-1 `blockB` masks differ from S5's block(24): exact achieved rate, disjoint gaps.
  At p=0.1 gaps >51 pts are infeasible (B96/B168 → a single 51-gap); at p=0.3 B168 → a
  single 154-gap (annotated in the figure).
- mar masks nest across rates (one latent, rate-dependent threshold): smoother curves, but
  rates are not independent draws.
- Moirai install was off-label: uni2ts 2.0.0 pins torch<2.5/scipy<1.12/numpy-1.26/gluonts
  0.14/pandas<2.2, so it was installed `--no-deps` plus gluonts 0.17, lightning,
  jaxtyping+jax[cpu], hydra-core. torch 2.6.0/numpy 2.4.4/pandas 2.3.3/scipy 1.17.1
  untouched. Inference ran unpatched; clean baselines are sane (moirai is a small base
  model: ETTh1 clean MSE 28.8 vs bolt 11.6). All series stamped freq="h" (freq-blind, like
  bolt/timesfm in S5); 20 samples → median.
