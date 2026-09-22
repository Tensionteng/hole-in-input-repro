# S29 notes — three audits, no new data

Anchor gates: all eight stored cells (S10 Penmanshiel x4, S14 METR-LA x4) reproduce to
|dev| <= 0.0001%. Arm C additionally reproduces the stored `cens|zero` numbers *implicitly*:
its relaxed window builder, restricted to complete-target windows, returns mean NMSE
28.42657 and median 1.65783 -- identical to S10's stored values.

---

## Arm A — the missingness pattern is itself a forecast signal

A network that sees ONLY the binary mask (plus mu, sd of the observed points so the output
has units, and never a single observed value):

| domain | bolt, best fill | mask-only | context-mean baseline | mask-only beats bolt on |
|---|---|---|---|---|
| Penmanshiel curtailment | 1.566 | 2.130 | 5.290 | **43% of windows** |
| METR-LA outages | 1.244 | 2.282 | 1.474 | 27% of windows |

Median NMSE, held-out half. The mask alone is far better than knowing only the level
(2.13 vs 5.29 on Penmanshiel) and it beats a foundation model **that sees every observed
value** on 43% of curtailment windows. Missingness in these systems is not noise: curtailment
happens *because* the wind is high, so the pattern of holes carries the signal the holes
removed. Note the mean is poor (855) -- the mask-only model has a heavy tail, as expected
from a model with no level information beyond two scalars.

Consequence for the framing: "repair the holes so the model can forecast" is only half the
story; the holes are also evidence.

---

## Arm B — the imputer as an attack surface, and why the interface is the defence

Threat model: the attacker controls the imputation step (a third-party library, an upstream
vendor, or whatever `fillna` the pipeline calls) but not the observed values, and poisoned
fills must lie inside the window's observed [min, max], so the context still passes a range
check. Untargeted objective (maximise forecast error), 400 Adam steps.

| domain | missing rate | plain fill | fill+mask (rank 2) | NaN (rank 0) |
|---|---|---|---|---|
| Penmanshiel curtailment | 14.1% | x1.01 / q90 x1.25 / max x11 | x1.00 / x1.13 / x5 | **x1.00 / x1.00 / x1** |
| METR-LA outages | 22.4% | **x3.06 / q90 x165.7 / max x3.7e5** | x1.18 / x2.21 / x19 | **x1.00 / x1.00 / x1** |
| | | >2x on 60% (METR) | >2x on 12% | **>2x on 0%** |

Two findings, and the second is the one that matters.

**(1) The attack is regime-dependent, not universal.** On Penmanshiel it barely works
(median damage x1.01): 14% missing plus a range constraint leaves the attacker almost no
leverage. On METR-LA it is devastating: the median window's error **triples**, 60% of windows
more than double, and the tail reaches x3.7e5 -- with every poisoned value inside the
observed range. So S25's spectacular steerability figure (0.002x clean) does translate into
a real attack, but only where missing rates and gap geometry give the attacker room. Any
security claim has to be stated with the regime attached.

**(2) The attack-surface dimension is rank(J_M).** The ordering plain > fill+mask > NaN
reproduces S25 Part 0's rank ordering |M| > 2 > 0 exactly, and the NaN interface is damaged
by **x1.00 with a maximum of x1.00** -- not "hard to attack" but structurally without a
lever, since the fill is not an input at all.

This creates a genuine, quantified design trade-off that pairs with S27:

> A full-rank interface is what lets a better imputer help you (S27: 40-93% of the error
> recovered) and is also what lets an adversary move your forecast by 3x. A rank-0 interface
> is provably immune and provably unable to benefit. **You cannot have both, and the choice
> is made at pretraining time.**

---

## Arm C — reported error is measured on a biased slice

Every window list in this project, and in the papers it builds on, requires the forecast
TARGET to be fully observed. That is a selection on the outcome. Rebuilding Penmanshiel's
censored windows without that filter and scoring only the observed target points:

| | windows | NMSE median | NMSE mean |
|---|---|---|---|
| complete target (what everyone scores) | 602 | **1.658** | 28.4 |
| partial target (everyone drops these) | 212 | **17.648** | 691.6 |
| all | 814 | 1.914 | — |

- The filter silently removes **26% of the data**.
- The removed quarter is **10.6x harder** by median NMSE (17.65 vs 1.66).
- Reported median error is therefore **15.4% optimistic**; by the mean the gap is far larger.

The mechanism is intuitive in hindsight and nobody states it: a window whose target is partly
missing is a window where the system was in an abnormal state, which is exactly when
forecasting is hard. Filtering on the target selects the calm periods. The same logic
appeared incidentally in this project's PhysioNet round, where windows passing a
complete-target rule turned out to be the densely-monitored, stabler patients.

This is a benchmarking claim, not a modelling one, and it is the cheapest of the three to
extend: it applies to every leaderboard that filters on target completeness.

## Artifacts
`run_s29_audit.py`, `s29_results.json` (per-window arrays for all arms), `s29_full.log`.
