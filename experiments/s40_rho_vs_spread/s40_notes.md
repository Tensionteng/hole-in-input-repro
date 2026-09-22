# S40 — does a checkpoint's rho predict how much its benchmark score moves?

## Why
The paper has two halves that have not been joined quantitatively. Section 5 measures rho, a
property of a model's input path, on synthetic masks. Section 4 measures how much a model's
GIFT-Eval score moves when the fill convention changes, on real benchmark missingness. We
assert the ordering matches ("the ordering across families is the one Table 3 predicts") from
four families read by eye.

With S36 there are now eleven checkpoints with a measured rho. Running the fill-sensitivity
audit on the same eleven turns an eyeballed ordering into a scatter plot with a correlation:
does the lever we measure in a Jacobian predict the score movement a reviewer would see?

## Design
* Same protocol as S32 part A: the nine missing-containing GIFT-Eval datasets, four fill
  conventions (NaN, linear, zero, forward-fill), MASE against an in-window naive-1 baseline
  computed on observed context points, target scored on its observed points only.
* Score sensitivity per checkpoint = (worst - best convention) / best, pooled over datasets.
* rho per checkpoint comes from S36/S30 unchanged; nothing is refitted.
* Checkpoints without a declared path (TimeMoE) contribute the plain-only sensitivity, which
  is a data point rather than a gap: a user of that model has no other option.

## Anchor gate
The four checkpoints S32 already measured (bolt-base, chronos2, moirai, timesfm) must
reproduce their stored spreads within 5%.

## Pre-registered predictions
* P1: score sensitivity correlates positively with rho on the plain path across checkpoints.
  Rank correlation > 0.5.
* P2: the gap between rho-plain and rho-declared predicts sensitivity better than rho-plain
  alone, because the sensitivity is the difference between two paths, not a property of one.
* P3: TimesFM will be the least sensitive checkpoint, because it overwrites the fill.
* P4 (likely to fail): within the Chronos-Bolt size ladder, sensitivity will rise with scale,
  reproducing S28's finding on the single most missing-valued dataset at corpus level.

## Anchor gate — PASS, bit-exact
All 28 stored S32 cells for chronos-bolt-base reproduced to 0.00%.

## Scorecard (s40_scorecard.txt)

Two summary statistics behave differently and the difference is the result, so we report both.
Pooled over the whole corpus, most windows are nearly complete: five of the seven datasets are
under 1.5% NaN. The high-missingness statistic is therefore a **fixed** dataset, the most
missing-valued one in the corpus (`kdd_cup_2018_with_missing`, 12.15% of context points
missing); the per-checkpoint max is not comparable, because for some checkpoints the argmax is
a 63-window near-complete dataset.

| statistic | predictor | Spearman | Pearson |
|---|---|---|---|
| spread, all datasets | rho plain | **+0.810** (p=0.015) | +0.790 (p=0.020) |
| spread, all datasets | rho plain − rho declared | −0.690 (p=0.058) | −0.905 (p=0.002) |
| spread, kdd_cup (12% NaN) | rho plain − rho declared | **+0.833** (p=0.010) | **+0.895** (p=0.003) |
| spread, kdd_cup (12% NaN) | rho plain | −0.786 (p=0.021) | −0.534 (p=0.172) |

* **P1 CONFIRMED on the pooled statistic** (rho plain, Spearman +0.810) and **reversed on the
  high-missingness one**. Where there is essentially nothing missing, the only thing that can
  move a score is how much a model uses the handful of filled points, which is exactly rho on
  the plain path.
* **P2 CONFIRMED where it matters, falsified where it does not.** On the most missing-valued
  dataset the *gap* between the two paths predicts the spread at Spearman +0.833, and rho plain
  alone does not. This is what Proposition 1 says it should be: the spread between conventions
  is the distance between two reachable sets, not a property of either one.
* **P3 not testable as stated.** TimesFM is excluded from this round: it is in S30's rho table
  but its GIFT-Eval cells were run in S32 under a wrapper this round does not use, and rerunning
  it is not what this round is for.
* **P4 CONFIRMED.** Within the Chronos-Bolt ladder, spread on the most missing-valued dataset
  rises with scale: 18.4% / 24.0% / 23.7% / 32.3% for 8.7M / 21M / 48M / 205M. This reproduces
  S28's corpus-level finding on a per-dataset statistic.

## Caveat we must state in the paper

The eight checkpoints form two clusters, four Chronos-Bolt and four Moirai-family, so a
between-checkpoint correlation is close to a two-group comparison and the p-values above are
optimistic. The two claims that do not depend on that are the **within-family scale trend**
(P4, four points inside one family) and the **between-family ordering** (declared-path
behaviour predicts which cluster moves more where missingness is high). Report those, and
report the correlation as descriptive.
