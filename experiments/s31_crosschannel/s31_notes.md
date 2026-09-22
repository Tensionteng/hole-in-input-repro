# S31 notes — when one channel goes dark, can a multivariate TSFM borrow from its neighbours?

## Setup

Chronos-2 with `past_covariates`. Per (dataset, target channel, mechanism, rate): the target
channel is holed; its K=4 most-correlated neighbours are supplied **fully observed**
(correlation computed on the first 80% of the timeline, so the neighbour choice cannot leak
the evaluation window). This is the most favourable possible case for borrowing, so a null
result is strong. 150 windows, 6 target channels per dataset, ETTh1/ETTm1/weather, 18 cells
per (mechanism, rate). Paired per-window relMSE against the model's own clean univariate
forecast, median.

Four conditions: (a) clean univariate = 1.000 by construction; (b) holed + filled,
univariate; (c) holed + filled, + neighbours; (d) holed **declared as NaN**, + neighbours.

## Result — and the answer depends on both the mechanism and whether you declare the hole

| mechanism | p | (b) holed, univariate | (c) + neighbours | (d) NaN + neighbours |
|---|---|---|---|---|
| mcar | 0.3 | 1.019 | **0.996** | 1.012 |
| mcar | 0.7 | 1.211 | **1.084** (60% recovered) | 1.096 |
| block | 0.3 | 1.067 | 1.107 (worse) | **1.005** |
| block | 0.7 | 1.357 | 1.357 (**0% recovered**) | **1.023 (95% recovered)** |
| mnar_high | 0.3 | 1.227 | 1.276 (worse) | 1.278 |
| mnar_high | 0.7 | 3.059 | 2.967 (4%) | 3.503 (worse) |

Three distinct regimes:

1. **Scattered dropout (mcar): neighbours help directly.** 60% of the damage recovered at
   p=0.7, and at p=0.3 the multivariate forecast is *better than the clean univariate one* --
   correlated channels add information beyond the target's own history.

2. **Block outages: neighbours help only if you DECLARE the hole.** With a filled target,
   neighbours buy back **nothing** (1.357 -> 1.357). Declare the same hole as NaN and the
   damage almost vanishes (1.357 -> **1.023**, 95% recovered). The fabricated straight line
   that linear interpolation writes across a long gap drowns out the neighbour signal; remove
   it and the model reads the neighbours instead. **This is the single largest repair
   measured anywhere in this project, it needs no training, and it exists only in the
   multivariate setting.**

3. **Value censoring: neighbours do not help, and declaring makes it worse.** 4% recovered
   at p=0.7 with a filled target; declaring the hole *raises* error to 3.503. Consistent with
   the information floor -- and with the S25/S27 finding that censoring is the one regime
   where the model needs content injected rather than withheld.

## Why this matters beyond the practical advice

It closes a gap in S25/S26. Those rounds concluded that block missingness is "saturated" --
that fixed fills already sit at the achievable frontier and there is nothing left to
recover. That conclusion was correct **univariately** and wrong in general: the missing
information was in the neighbouring channels the whole time, and what was blocking access to
it was the fill, not the model. The univariate ceiling was an artefact of the setting.

It also gives the interface argument its third independent confirmation, from a new
direction: **declaring the hole is what unlocks cross-channel borrowing.** S27 said the
declaration channel caps what a better imputer can do; S29 said its rank is the attack
surface; S31 says it also gates whether a multivariate model can use its other inputs at all.

## Practical rule

> Under outages: declare the hole (NaN, not a fill) and give the model correlated channels --
> that removes ~95% of the damage for free. Under value censoring: neither buys anything.

## Caveats

- Chronos-2 only; Moirai's covariate path is not yet run.
- Neighbours are always fully observed. Real deployments often lose correlated sensors
  together (a substation trip, a weather event), which is the case this round does not test
  and where the benefit should be much smaller.
- Synthetic masks on real multivariate data; the real-mechanism version (METR-LA neighbouring
  loop detectors) is the natural follow-up and is directly supported by the S14 window set.
- The recovery percentages are unstable where the univariate gap is tiny (a near-zero
  denominator), which is why the table reports levels; read the percentages only for the
  p=0.7 rows.

## Artifacts
`run_s31_crosschannel.py`, `s31_results.json` (per-cell levels + recovery + neighbour ids),
`s31_full.log`.
