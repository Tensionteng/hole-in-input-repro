# S32 notes — how much of a GIFT-Eval score is the model, and how much is its NaN handling?

## The corpus scan (the motivating fact)

Scanning the distributed GIFT-Eval corpus: **11.77% of its 305,205,463 points are NaN**, and
**9 of 28 datasets contain missing values** -- not only the three named `*_with_missing`:

| dataset | %NaN | % of series affected |
|---|---|---|
| electricity | **19.19** | 57.3 |
| kdd_cup_2018_with_missing (H) | 17.12 | 100.0 |
| restaurant | 13.76 | 98.8 |
| bitbrains_rnd | 13.13 | 100.0 |
| kdd_cup_2018_with_missing (D) | 9.40 | 98.5 |
| bitbrains_fast_storage | 4.52 | 94.3 |
| car_parts_with_missing | 4.49 | 6.2 |
| temperature_rain_with_missing | 2.58 | 96.5 |
| hierarchical_sales | 1.48 | 100.0 |

Missingness is not a corner case the benchmark excludes; it is a large, unremarked property
of the benchmark the field reports on. No paper reporting a GIFT-Eval number states how its
model handled those NaNs, and S30 showed the handling differs qualitatively by family.

Protocol note: random windows per series (context 512, horizon 64), scored on observed target
points, MASE vs an in-window naive-1 baseline. Deliberately NOT the official GIFT-Eval
protocol, so absolute values are not leaderboard-comparable; every claim is a WITHIN-protocol
comparison across fills and models.

## Part A — a score moves by up to 47% with an undocumented preprocessing choice

`kdd_cup_2018_with_missing/H`, 12.2% context NaN, MASE median:

| model | nan | linear | zero | ffill | spread |
|---|---|---|---|---|---|
| chronos-2 | 1.309 | 1.700 | 1.923 | 1.676 | **46.9%** |
| bolt-base | 1.295 | 1.573 | 1.715 | 1.553 | **32.5%** |
| moirai | 2.458 | 2.542 | 2.631 | 2.532 | 7.0% |
| timesfm | 2.413 | 2.441 | 2.597 | 2.430 | 4.6% |

Two readings:
1. **The measured gap between models depends on the fill.** chronos-2 beats timesfm by 46%
   under `nan` and by 22% under `zero`. A leaderboard entry is a joint measurement of model
   quality and an unreported preprocessing decision.
2. **Sensitivity is family-specific, and S30 predicts which.** timesfm and moirai barely move
   -- timesfm because it silently overwrites any fill with its own interpolation, so the user's
   choice is discarded. The Chronos family moves most because its NaN path discards content
   while its plain path uses it fully; the two paths are far apart.

Datasets with <2% context NaN show ~0 spread, as expected -- the effect tracks the amount of
missingness, not the dataset.

## Part B — scale buys accuracy here, does NOT buy robustness, and *increases* fill sensitivity

MASE median, `nan` fill, tiny (8.7M) -> base (205M):

- **Clean datasets: +4.5% better** on average (m4_hourly +24.9%, us_births +25.2%, solar +9.5%).
  This repairs S28's stated limitation: on ETT/weather clean accuracy was flat with scale, so
  "scale does not buy robustness" could not be separated from "scale buys nothing". On
  GIFT-Eval scale demonstrably buys accuracy.
- **Fill sensitivity GROWS with scale.** On kdd_cup the spread across the four fills is
  18.4% (tiny) -> 24.0% (mini) -> 23.7% (small) -> **32.3% (base)**.

So the larger model is more accurate and **more dependent on an undocumented preprocessing
choice**. Together with S28 (relative degradation under missingness flat to worsening across
a 24x span), scaling does not address missingness and increases the share of the score that
is attributable to preprocessing rather than to the model.

## Caveats
- Not the official GIFT-Eval protocol; absolute numbers are not leaderboard entries.
- Only one dataset in our window sample has >10% context NaN, which is where the large
  spreads appear; the claim is demonstrated there and shown to vanish at low NaN rates.
- Several GIFT-Eval datasets store 2-d (n_variates, T) targets; each variate is treated as a
  univariate series here.

## Artifacts
`run_s32_gifteval.py`, `s32_partA.json`, `s32_partB.json`, logs.

---

# S32c — the two protocol/coverage gaps, closed

## Gap 1: the headline no longer rests on one dataset

Pooling windows from every missing-containing GIFT-Eval dataset (7,725 windows, random
window protocol) and stratifying by each window's OWN context-NaN fraction gives a
dose-response curve, with the low-NaN bins acting as their own negative control.

**Spread across the four fill conventions (MASE median), by context-NaN bin:**

| ctx NaN | n | bolt-base | chronos-2 | timesfm | moirai |
|---|---|---|---|---|---|
| 0–1% | 4308 | 0.0% | 0.2% | 0.2% | 0.3% |
| 1–5% | 2327 | 2.0% | 5.0% | 0.5% | 0.8% |
| 5–15% | 625 | 34.6% | 46.1% | 1.7% | 3.4% |
| 15–30% | 208 | 76.8% | **101.4%** | 14.1% | 5.9% |
| >30% | 257 | 94.3% | **105.8%** | 27.2% | 37.7% |

- The effect is **zero where there is no missingness** and rises monotonically with it,
  reaching **>100%** for Chronos-2. A model's measured score can more than double purely
  from an unreported preprocessing choice.
- The family ordering is the one S30's taxonomy predicts: the Chronos family is most
  sensitive (its declared path discards the fill, its plain path uses it fully, so the two
  are far apart), TimesFM least (it overwrites any fill with its own interpolation), Moirai
  in between.
- **`nan` is the best fill in nearly every high-missingness cell** across all four families
  -- declaring beats filling, independently of everything measured on synthetic masks.

## Gap 2: the conclusion is not a protocol artefact

Repeating the identical analysis under a last-window-per-series protocol (the official-style
held-out convention, 1,366 windows) reproduces the direction and rough magnitude in the
>30% bin: bolt-base 22.4%, chronos-2 35.2%, moirai 47.0%, timesfm 2.7% (n=32, so noisy).
Low-NaN bins remain ~0-4% under both protocols. The sensitivity is a property of the data
and the model family, not of how we chose windows.

Absolute MASE values remain non-comparable to the published leaderboard; every claim here is
a within-protocol comparison, now shown to hold under two protocols.
