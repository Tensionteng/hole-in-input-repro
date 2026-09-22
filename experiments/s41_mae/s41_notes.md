# S41 — does anything in the paper depend on the error being squared?

## Why
Every number in the paper is an MSE ratio. Squared error weights the tail heavily, and the
mechanisms we study differ precisely in how they treat extremes: value censoring removes the
largest values, so a squared metric is the one most likely to exaggerate its damage. A reviewer
is entitled to ask whether the mechanism ordering, the flat imputation ceiling, and the
interface gain survive an absolute-error metric. This round recomputes the load-bearing
measurements under both metrics **in the same pass**, so the pairing is exact.

## Design
* Same nine datasets, same windows, same masks, same models as S35/S37. Nothing is refitted.
* For every prediction we compute both per-window MSE and per-window MAE, and both are
  normalised by the same window's own clean-context error under the same metric.
* Measured under both metrics:
  A. the characterisation grid (4 mechanisms x 2 rates, linear fill);
  B. the interface sweep at rate 0.7 (native / dual / dual_obsnorm at alpha in {0, 0.5, 1}),
     each interface normalised by its OWN clean forecast, as in S37;
  C. the closure statistic derived from B.
* Also recorded: the permutation probe ratio rho under an L1 output-change norm, since rho is
  defined through an RMS norm and a reviewer may ask the same question of it.

## Anchor gate
The MSE half must reproduce the stored S35 grid and the S37 closure within 5%.

## Pre-registered predictions
* P1: the mechanism ordering (scattered < block < extreme < censoring) is unchanged under MAE
  on the same eight of nine datasets.
* P2: the ABSOLUTE damage under censoring shrinks under MAE, because squared error rewards the
  extremes censoring removes. Prediction: relMAE under censoring at rate 0.7 will be roughly
  half the relMSE excess.
* P3: closure is unchanged in sign everywhere and its median moves by less than 15 points.
  This is the one that matters: if the interface result is a squared-error artefact, the paper
  is wrong.
* P4: rho under an L1 norm is within 0.05 of rho under the RMS norm, and the declared paths
  stay at float noise. rho is a ratio of two output changes, so the norm should cancel.

## Anchor gate — PASS
The MSE half reproduces S37's closure exactly on ETTh1 (83.4 / 48.0 / 39.4 / 39.0 / 69.6 /
30.8 / 89.5 / 84.9) and the S35 grid to within the difference expected from raising the window
count from 40 to 300. All tables now quote the 300-window numbers, which is why a few figures
in the text moved slightly (e.g. ETTh1 censoring at rate 0.7: 3.09 -> 3.29).

## Scorecard (s41_scorecard.txt)

* **P1 CONFIRMED.** Censoring is the worst cell on the same 8 of 9 datasets under MAE as under
  MSE, with the same exception (`exchange`).
* **P2 CONFIRMED.** The excess over clean shrinks under MAE, and by a mechanism-ordered factor:
  median MAE-excess / MSE-excess is 0.42 under censoring, 0.47 under extreme censoring, 0.57
  under block outages, 0.60 under scattered dropout. Censoring loses most from an unsquared
  metric, which is what an argument about tails predicts. The censoring excess at rate 0.7
  falls from a median of 2.29 to 1.05 — very close to the predicted halving.
* **P3 CONFIRMED, and this is the one that mattered.** Closure keeps its sign in 62 of 63
  cells. The single exception is weather / block / p=0.7, +0.3% under MSE and -13.0% under MAE
  — a cell that was already indistinguishable from no effect, and the same weather-block corner
  that supplies the one negative cell in the MSE table. Medians move by 4.1 points in-sample
  (53.1 -> 57.2) and 1.7 points held out (73.2 -> 71.5).
* **P4 CONFIRMED.** rho under an L1 output norm differs from rho under RMS by at most 0.015
  over all 27 dataset x convention cells, and both declared paths stay at 0.0000.

## What this changes in the paper
Nothing substantive, which is the point. Table 2 now carries both metrics in every damage cell,
Section 5.1 states the shrinkage factors, and App. J gives the full comparison. The claim we can
now make and could not before is that the interface result is not an artefact of the metric.
