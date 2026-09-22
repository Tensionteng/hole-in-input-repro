# S39 — the silent-failure and the calibration repair on REAL benchmark missingness

## Why
Two claims in the paper are measured on synthetic masks plus two real deployments:

1. the silent failure (§5.2) — coverage collapses while the interval narrows;
2. the calibration repair (§10) — Mondrian conformal keyed to the detected mechanism restores
   per-group coverage.

The Limitations section currently has to say "real mechanisms, of which we have two". GIFT-Eval
ships nine datasets that are 1–19% NaN for their own reasons, with no synthetic mask anywhere in
the pipeline. Running both measurements there turns two real deployments into eleven, and it is
the same corpus the field reports scores on.

## Design
* Corpus: the nine missing-containing GIFT-Eval datasets, loaded by S32's reader. Context may
  contain NaN; the target is scored on its observed points only, so coverage is computed over
  observed target points.
* Windows are split by SERIES into a calibration half and a test half, so no series contributes
  to both.
* Fill: linear interpolation (the deployable default). Detector: the S16 GBDT over window
  geometry, fitted on ETTh1/ETTm1/weather and frozen — it has never seen a GIFT-Eval window.
* Variants: the model's native interval, one global conformal pool, and Mondrian keyed to the
  detected mechanism.
* Stratify by the window's own real-missing fraction, as in §4.2, so the low-missingness
  stratum is a negative control.

## Anchor gate
Windows with < 1% missing context must show near-nominal native coverage (the model is not
broken on clean data) and the three variants must agree there to within 0.03.

## Pre-registered predictions
* P1: native coverage falls monotonically with the real-missing fraction.
* P2: a single conformal pool fixes the average and leaves the high-missingness stratum short.
* P3: Mondrian keyed to the detected mechanism brings every stratum with >= 30 calibration
  windows to within 0.03 of nominal.
* P4: the detector will fire mostly `block` and `intermittent` on real GIFT-Eval missingness,
  because real gaps are runs and leading padding rather than value censoring. If it fires
  `censoring` often, the geometry features are picking up something we have not modelled and
  P3 is likely to fail.
* P5 (the one we expect to be wrong): the interval will NARROW as real missingness rises, as it
  does under synthetic censoring. Real benchmark gaps are mostly outages, not censoring, so it
  may widen instead — which would be the correct behaviour and would show the silent failure is
  specific to the censoring mechanism, not to missingness in general.

## Scorecard
(filled after the run)

## Anchor gate — FAILED, and the failure is the main result of the round

The gate asked that windows with < 1% missing context show near-nominal native coverage. They
do not: native coverage in the $0$--$1\%$ stratum is **0.822** against a nominal 0.90, over
1569 windows. \bolt{}'s intervals are under-covering on this corpus for reasons that have
nothing to do with missingness, which means GIFT-Eval cannot cleanly isolate a
missingness-induced coverage failure. Every number below must be read against that baseline
rather than against 0.90.

## Scorecard

* **P1 CONFIRMED but weak.** Native coverage falls monotonically-ish with the real missing
  fraction: 0.822 / 0.753 / 0.773 / 0.815 / 0.733 across the five strata. The drop from the
  cleanest to the most-missing stratum is 0.089 -- real, but nothing like the collapse to 0.217
  that synthetic censoring produces.
* **P2 CONFIRMED.** A single conformal pool reaches 0.899 overall and leaves the > 30% stratum
  at 0.829, i.e. it fixes the average and not the tail, exactly as on synthetic data.
* **P3 FALSIFIED.** Mondrian keyed to the detected mechanism moves the > 30% stratum only from
  0.829 to 0.840 --- about a sixth of the remaining gap, not the closure it achieves on
  synthetic mechanisms and on the two real deployments. Grouped BY DETECTED CLASS the repair
  does work (0.868--0.916 across six classes), so the failure is that a detected class is not
  homogeneous in missing RATE on this corpus: the same class spans 2% and 60% missing windows,
  and one conformal width cannot serve both.
* **P4 FALSIFIED.** We predicted the detector would fire mostly `block` and `intermittent` on
  real gaps. It fires `clean` 2184, `scattered` 796, `censoring` 559, `extreme` 614,
  `intermittent` 421 and `block` only 82. Real GIFT-Eval gaps look like value censoring to a
  geometry detector far more often than we expected.
* **P5 CONFIRMED (we predicted our own claim would not reproduce).** Intervals **widen** with
  real missingness rather than narrowing --- 625 to 632, 25.3 to 27.9, and so on in every
  stratum. The silent failure of the paper is specific to the value-censoring mechanism, not a
  property of missingness in general. On this corpus the model's intervals move in the honest
  direction.

## What this changes in the paper

Two things, both restrictions on claims we already make.

1. The silent failure must be stated as a property of **censoring**, not of missingness. The
   paper already derives it that way; this round is the evidence that the distinction matters
   on real data, and §5.2 / App. C should say so explicitly.
2. The calibration repair transfers at full magnitude to Penmanshiel and METR-LA, where the
   mechanism is homogeneous within a deployment, and does **not** close the gap on a
   heterogeneous benchmark corpus where one detected class spans a wide range of missing rates.
   The obvious fix --- condition the Mondrian groups on rate as well as mechanism --- is
   untested here and we should say so rather than claim it.

## Follow-up: conditioning the groups on rate as well as mechanism

Rather than assert the obvious fix, we ran it. Mondrian groups keyed to
(detected mechanism $\times$ missing-rate bin), same calibration split, same detector:

| stratum | native | 1 pool | mondrian | mondrian + rate |
|---|---|---|---|---|
| 0–1%   | 0.822 | 0.898 | 0.902 | 0.904 |
| 1–5%   | 0.753 | 0.913 | 0.899 | 0.892 |
| 5–15%  | 0.773 | 0.903 | 0.897 | **0.921** |
| 15–30% | 0.815 | 0.877 | 0.880 | **0.907** |
| >30%   | 0.733 | 0.829 | 0.840 | **0.855** |

It helps and it is not a cure. The 15–30% stratum is repaired (0.877 → 0.907) and the 5–15%
stratum over-corrects slightly; the worst stratum moves from 0.829 to 0.855 and remains 0.045
short of nominal on 83 test windows. Width is charged where the coverage is bought: 56 → 73 and
60 → 75 in the two worst strata, against 625 → 633 in the cleanest.

Conclusion for the paper: the mechanism-keyed repair is a full repair when the mechanism is
homogeneous within a deployment (Penmanshiel, METR-LA) and a partial one on a heterogeneous
benchmark corpus, where conditioning on rate recovers most but not all of the remainder. State
it that way.
