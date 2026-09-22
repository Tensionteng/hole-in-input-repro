# S31b notes — cross-channel borrowing under a REAL mechanism (METR-LA outages)

S31 found, on synthetic masks over real multivariate data, that declaring a hole and adding
correlated neighbours repairs ~95% of block-outage damage. This round tests it on METR-LA's
real sensor outages (S14's stored window list, Chronos-2 with `past_covariates`).

## Result 1 (the important one): the neighbours are usually down too

**Only 38 of 618 real outage windows (6.1%) have even 2 fully healthy top-correlated
neighbours.** Relaxing to neighbours that are themselves <=5% missing yields 98 usable
windows. In real deployments, correlated sensors fail together -- a substation trip, a
weather event, a comms outage takes out a neighbourhood, not a single sensor. Cross-channel
borrowing is mostly *unavailable* exactly when it is needed.

## Result 2: where it is available, the median gain is small and the tail gain is large

98 windows, mean target missing rate 16.3%, NMSE (var floor 4.0):

| condition | median | mean | vs filled-uni (paired median) | win rate |
|---|---|---|---|---|
| filled, univariate | 1.0738 | 2.078 | 1.000 | — |
| filled + neighbours | 1.0488 | 2.412 | x0.992 | 53% |
| **NaN-declared + neighbours** | **1.0183** | **1.238** | **x0.978** | 57% |
| NaN-declared, univariate | 1.0158 | 1.191 | x0.999 | 52% |

- Median gain from declaring + neighbours: **2%** (versus 95% recovery on synthetic masks).
- Mean error drops **40%** (2.078 -> 1.238) -- the benefit is concentrated in the tail, i.e.
  in the windows that were failing badly.
- The ordering from S31 survives (declaring beats filling), the magnitude does not.

## Reading

A fourth instance of the project's migration law: **the structure transfers, the magnitude
does not.** The synthetic 95% figure must be labelled synthetic. The deployable statement is
narrower and still useful: *declaring the hole rather than filling it cuts mean error 40% on
the windows where neighbours are available -- which is 6% of them.*

## Caveats
- Chronos-2 only; n=98 after the availability filter, so the median estimate is noisy.
- Neighbour choice uses correlations from the first 80% of the timeline; no leakage.
- "Fully healthy" is defined on the context window only.

## Artifacts
`run_s31b.py`, `s31b_results.json`, `s31b.log`.
