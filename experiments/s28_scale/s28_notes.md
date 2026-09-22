# S28 notes — does robustness to missing data improve with model scale?

## Question and pre-registration

The field's reflex answer to a TSFM weakness is "scale it up". Four released Chronos-Bolt
sizes share an architecture and a training recipe, so scale is the only variable:
**tiny 8.7M / mini 21.2M / small 47.7M / base 205.3M** (24x span).

Metric: **relative** degradation -- each model normalised by its OWN clean forecast, paired
per window, median. Pre-registered prediction: **flat**, because S25 Part 0 located the
damage in the input interface (content zeroed at masked positions, statistics computed
before masking), and the interface does not change with parameter count.

Anchor gate (bolt-base vs stored S5 cells): PASS, +0.0000% on all three.

## Result — flat, and if anything worsening

Best-of-{linear, zero, nan} fill, p=0.7, median paired relMSE, averaged over
ETTh1/ETTm1/weather, 300 windows:

| size | params | mcar | block | mnar_high | mnar_extreme | ETTh1 clean MSE (median) |
|---|---|---|---|---|---|---|
| tiny | 8.7M | 1.103 | 1.113 | 2.852 | 1.762 | 1.1144 |
| mini | 21.2M | 1.117 | 1.093 | 2.862 | 1.1091 | 1.1091 |
| small | 47.7M | 1.117 | 1.091 | 2.864 | 1.794 | 1.0719 |
| base | 205.3M | 1.121 | 1.086 | 2.875 | 1.827 | 1.1150 |

- **mnar_high is 2.852 -> 2.875 across a 24x span** -- unchanged to within 1%.
- **mcar and mnar_extreme get monotonically WORSE with scale** (1.103 -> 1.121 and
  1.762 -> 1.827).
- Only block improves, and only slightly (1.113 -> 1.086).

Scale does not buy robustness to missing data. The pre-registration is confirmed.

## The honest caveat that limits this round

**Clean accuracy is also flat on these three datasets** (ETTh1 median clean MSE 1.1144 ->
1.1150; ETTm1 0.7083 -> 0.7106; weather 2.357 -> 2.284). So on this evidence alone we cannot
separate "scale does not buy robustness" from "scale does not buy anything here" -- the
intended contrast (accuracy improves, robustness does not) is not demonstrated, because the
first half fails too on ETT/weather.

That is a real limitation of the round, not a presentational one. Closing it requires
datasets where these models' clean accuracy demonstrably improves with size; the GIFT-Eval
extension (downloading) is the intended fix, since it is the benchmark on which the family's
scaling was reported in the first place. Until then the defensible claim is narrower:

> across a 24x parameter span, relative degradation under missingness is flat to slightly
> worsening, on datasets where clean accuracy is also flat.

## Artifacts
`run_s28_scale.py`, `s28_results.json` (per-model, per-cell, 4 mechanisms x 4 rates x
3 fills x 3 datasets), `s28_full.log`. Checkpoints for tiny/mini/small were fetched to
`models_local/` by direct curl -- the hf-mirror `snapshot_download` path fails on the small
metadata files while the weight blobs download fine.
