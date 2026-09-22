# S37 — does the interface prescription transfer to datasets the adapter never saw?

## Why
S27's headline (restoring the content path removes 40–93% of the error the declared path
cannot touch) is measured on the same three datasets the input-embedding adapter was trained
on: ETTh1, ETTm1, weather. A reviewer will ask whether that is a prescription or a fit.

S35 gives six further benchmark datasets. The adapter checkpoints from S27 are on disk, so
the clean test is: freeze them, evaluate on the six unseen datasets, change nothing else.
This turns the section-6 claim from "we can fit an interface per dataset" into "the interface
fix is a property of the interface, and it transfers".

## Design
* Model: chronos-bolt-base. Interfaces: `native` (declared, S27 baseline) and `dual`
  (declared + content restored), each with its own S27 adapter, loaded frozen.
* Datasets: the 9 of S35. ETTh1/ETTm1/weather are IN-SAMPLE for the adapter; ETTh2, ETTm2,
  electricity, traffic, exchange, illness are HELD OUT.
* Sweep: alpha in {0, 0.25, 0.5, 0.75, 1.0}; mechanisms mcar/block/mnar_high/mnar_extreme;
  rates 0.3/0.7. Paired per-window relMSE against each series' own clean forecast, median.
* Closure = (native@a=1 - dual@a=1) / (native@a=1 - 1), the S27 definition.

## Anchor gate
The three in-sample datasets must reproduce the S27 cells within 5%.

## Pre-registered predictions (recorded before the run)
* P1: closure > 0 in a majority of held-out cells. Reason: the mechanism is architectural,
  not dataset-specific.
* P2: median held-out closure will be SMALLER than the median in-sample closure. Reason: the
  migration law — structure transfers, magnitude does not. This is the prediction most likely
  to be falsified, and it is the one that matters for the paper's honesty.
* P3: closure will be ordered by mechanism as in S27 — largest under censoring, smallest
  under block.
* P4: at alpha=0 the restored interface will not be worse than the declared one in a majority
  of held-out cells (the "costs nothing when the imputer is poor" claim).

## Scorecard
(filled after the run)

## Anchor gate

FAILED as first written, and the failure found a real bug in S27 rather than in S37.

ETTh1 reproduced S27 bit-exactly (max deviation 0.00% over 120 cells); ETTm1 deviated by a
median of 1.5% and weather by 3.8%, with 40 cells outside +-5%. Disabling our
degenerate-channel filter did not close the gap, which ruled out the obvious explanation.

`verify_s27_base.py` isolates the cause: S27's `sweep()` computes the clean denominator with
whatever `input_patch_embedding` is loaded at that moment, and inside its dataset loop the
previous dataset's last adapted interface is still loaded. Every dataset after the first was
therefore normalised by an *adapted* clean forecast. Recomputing `weather|block|0.7` with
`dual_obsnorm` loaded reproduces S27's stored values to four decimals; recomputing it with the
released weights does not. S27's notes now carry the correction.

S37 resets to the released weights before computing any denominator, and additionally
normalises **each interface by its own clean forecast**, which is what `app:method` requires:
the adaptation edits a module the model uses on every input, so a generic clean-data gain or
loss must not be charged to the interface. `run_clean_control.py` shows that generic effect is
real and signed both ways --- the adapter improves the clean forecast by 0.5-3% on the ETT and
weather family and *degrades* it by 5-9% on electricity, traffic and illness.

## Scorecard (see s37_scorecard.txt for the full grid)

Cells where the declared path's own excess over clean is < 0.02 are reported n/a: there is
nothing to close and the ratio is dominated by its denominator.

* **P1 CONFIRMED, more strongly than predicted.** Closure is positive in 44/44 held-out cells
  and 22/23 in-sample cells. The single negative is weather, block, p=0.3 (-33% of an excess
  of 0.03).
* **P2 FALSIFIED.** We predicted held-out closure would be *smaller* than in-sample, by the
  migration law. It is larger: median 73.2% held out against 53.1% in-sample, IQR [47.8, 93.1]
  against [37.9, 72.7]. The interface fix is not fitted to the datasets it was trained on. Note
  this is a transfer across *datasets*, not from synthetic to real missingness, so it does not
  contradict the migration law --- but we predicted it would behave the same way and it did not.
* **P3 CONFIRMED.** Closure is largest under extreme censoring (median ~80%) and smallest under
  block outages (median ~44%), the S27 ordering.
* **P4 PARTLY FALSIFIED.** Restoring the content path is no worse than declaring at alpha=0 in
  only 41/67 cells (61%), against 6/8 in S27. Under the corrected normalisation, restoring the
  content path does cost something in about two cells in five when the imputer is poor. This
  must be stated in the paper; it is the price of the lever.

Cells above 100% (illness, exchange censoring) are cells where the restored interface beats its
own clean-context forecast. At alpha=1 the fill IS the truth, so the context differs from clean
only by the missingness flag; an adapted embedding can use that flag as a mild hint. These are
small-denominator cells and we do not lean on them.
