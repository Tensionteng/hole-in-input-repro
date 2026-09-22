# S26 notes — does S25's recoverable fill gap (term 3) survive on REAL missingness?

**Status: PRE-REGISTRATION (written before any result). Results appended below.**

## Why this round

S25's decomposition, on synthetic mechanisms:

```
R(f,g) - R_clean = (1) information + (2) architecture + (3) fill/routing
```

with term (3) ~0 (in fact slightly negative) under mcar/block, and **dominant under value
censoring** — 1.203 of a 1.875 total at mnar_high p=0.7, learned fill beating the best fixed
fill on 81% of windows paired. That single number is what re-writes the repair ladder and
gives MechGate a real job. It is currently supported **only by synthetic masks**, and the
project's own history (S6E's Tobit winning on synthetic mnar and then *losing* on real
curtailment, S10) is a direct precedent for a synthetic repair failing to migrate.

## Pre-registered predictions (risky and directional; can fail either way)

- **H1 — Penmanshiel curtailment (real value censoring).** A fill net trained through the
  frozen model **beats the best fixed fill** (zero, stored NMSE 28.43) on the held-out test
  half, and wins on >50% of windows paired. Rationale: real curtailment is the real-world
  instance of mnar_high, where S25 puts term (3) at 64% of the damage.
- **H2 — METR-LA sensor outages (real block geometry).** The learned fill **does NOT beat**
  the best fixed fill (nan, stored NMSE 2.00). Rationale: S25 puts term (3) at ~0 (negative)
  under block geometry — fixed fills are already at the frontier there.

**The test is the contrast, not either half.**
- H1 and not-H2 -> term (3) is real AND mechanism-specific: the revised ladder and the
  "give the router a better action set" story both hold.
- H1 and H2 -> learning the fill just always helps; the mechanism-specific claim is wrong
  and the routing argument weakens to "always use a learned fill".
- not-H1 -> term (3) is a synthetic artefact; the revised ladder is wrong and S25's system
  conclusion must be withdrawn. This is the outcome that would hurt, and it is exactly the
  S6E->S10 failure pattern, so it is a live possibility.

Secondary, unregistered-but-recorded: whether `fill_mask` (rank-2) or `plain_fill` (full
rank) wins per domain. S25's P3 falsification predicts **plain fill on Penmanshiel**
(censoring needs fabrication) and **fill+mask on METR-LA** (blocks punish fabrication).

## Setup

- Windows/masks/metrics are the **stored** ones: `run_s6_real_censor.build_windows/extract`
  (CTX=144, H=24, 602 cens + 602 ctrl, NMSE with a 100 kW^2 variance floor) and
  `run_s14_metrla.build_windows/extract` (CTX=512, 618 miss + 618 ctrl, 4.0 mph^2 floor).
  Anchor gates re-run four stored S10 cells and four stored S14 cells first.
- **Split**: the S10 per-turbine / per-sensor 50/50 cal/test split (seed 20250812). The fill
  net sees the CAL half only; every reported number is on the held-out TEST half, with the
  fixed-fill baselines recomputed on that same half so comparisons are paired.
- **Fill class**: identical to S25 — a tanh-bounded correction within 3 observed-sigma of the
  linear interpolant, zero-init (so the net starts exactly at linear). Same NaN guards
  (bolt's InstanceNorm back-propagates sqrt(0)=inf on a constant filled context).
- **Mask**: oracle (operator status codes / exact zeros), as in S10/S14's main tracks. This
  isolates "does a better fill exist" from "can the mechanism be detected"; S16 measures the
  latter (0.733 censoring recall with the deployable status-free mask). A deployed system
  needs both, and the honest headline for S26 is an upper bound on the deployable gain.
- **Deviation**: the METR-LA learned-fill arm uses H=64 (bolt's native single block) because
  H=96 triggers the pipeline's autoregressive quantile-mixing rollout, which is not a clean
  object to differentiate through. Its gate runs at the stored H=96; its arm carries its own
  recomputed H=64 baselines.
- Penmanshiel's censoring **clamps rather than deletes** (y_obs = min(y_true, cap)), so
  `keep` (feed the clamped value) is a distinct and, per S10, strong baseline.

---

# RESULTS

## Anchor gates — PASS (exact)

Eight stored cells re-run: S10 Penmanshiel (cens keep/zero/linear, ctrl clean) and S14
METR-LA (miss keep/linear/nan, ctrl clean). All reproduce to |dev| <= 0.0001%, so S26's
windows/masks/metrics are the stored ones and every comparison is paired with S10/S14.

## A design flaw found and fixed mid-round (disclosed)

The first full run centred the admissible fill class on the **linear** interpolant, inherited
from S25. On Penmanshiel the best fixed fill is **zero**, and measurement showed zero lies
*outside* the +-3 observed-sigma ball around linear in **66.3%** of windows (median distance
3.37 sigma, q90 7.17, max 20.8). The learned fill was therefore structurally unable to
represent the baseline it was being scored against, and its apparent loss said nothing.
Re-run with the class **centred on the best fixed fill**, so zero-init reproduces the
baseline exactly and any movement is a genuine improvement over it. All results below use
the centred class; the linear-centred run is kept in `s26_results.json` under `c-linear`
keys for Penmanshiel as the record of the flawed comparison.

## Headline: paired per-window results (held-out test half)

**Penmanshiel — real value censoring (306 test windows), learned fill vs `zero`:**

| variant | NMSE mean | NMSE median | paired ratio median | win% | >2x worse | sign test |
|---|---|---|---|---|---|---|
| **plain fill** | 26.45 (vs 19.08) | **1.4985 (vs 1.5660)** | **0.938** | **62%** | 9.2% | **p = 1.7e-05** |
| fill + mask | 29.73 | 1.7374 | 0.997 | 51% | 22.2% | p = 0.86 |

**METR-LA — real block outages (412 test windows), learned fill vs `nan` / vs `linear`:**

| variant | vs | NMSE median | paired ratio median | win% | sign test |
|---|---|---|---|---|---|
| plain fill | nan | 1.3273 (vs 1.2770) | 1.003 | 48% | p = 1.00 |
| fill + mask | nan | 1.3186 | 0.986 | 54% | p = 0.10 |
| plain fill | linear | 1.3273 (vs 1.2441) | 1.005 | 49% | p = 1.00 |
| fill + mask | linear | 1.3186 | 0.991 | 52% | p = 0.52 |

## Scorecard

| # | prediction | outcome |
|---|---|---|
| **H1** | Penmanshiel: learned fill beats the best fixed fill AND wins on >50% of windows | **SPLIT.** The paired clause is met decisively: 62% win, median paired ratio 0.938, sign test p=1.7e-5. The aggregate-NMSE clause is **not**: mean 26.45 vs 19.08, because 9.2% of windows are >2x worse and 3.6% are >5x worse (max ratio 2.1e4). |
| **H2** | METR-LA: learned fill does NOT beat the best fixed fill | **CONFIRMED.** No variant is significant against either baseline (p = 1.00 / 0.10 / 1.00 / 0.52). |
| contrast | H1 (paired) and not-H2 | **HOLDS.** A learned fill significantly beats the best fixed fill on real value censoring and does not on real outages — the mechanism-specific structure S25 predicted. |
| secondary (from S25's P3) | plain fill wins on censoring, fill+mask on blocks | **CONFIRMED on Penmanshiel** (plain 62%/p=1.7e-5 vs mask 51%/p=0.86) and **directionally consistent on METR-LA** (mask 54% vs plain 48%). Censoring needs fabrication; blocks punish it. |

## What this means — the honest reading

**Term (3) is real on real value censoring, but far smaller than synthetic and it has a
tail.** Synthetically (S25) the learned fill cut the excess by ~64% (2.875 -> 1.672). On real
curtailment the typical-window gain is **6%** (paired median ratio 0.938) and the aggregate
mean gets *worse*, because a minority of windows fail catastrophically. So:

- The **direction and the mechanism-specificity replicate** — that is the part the revised
  story needed, and it survived a test designed to kill it.
- The **magnitude does not**. Any claim of the form "64% of censoring damage is recoverable"
  is a synthetic-only statement and must be labelled as such.
- A learned fill is **not safe to deploy unguarded**: it improves the median window and
  ruins the tail. This is precisely the case for the abstention + group-conformal layer the
  project already has — routing to a learned fill should be paired with the calibration
  machinery, not offered as a standalone repair.

**Synthetic-to-real augmentation actively hurts.** Adding 1776 synthetically curtailed clean
windows (clamped to the k-th largest, mimicking a real curtailment plateau) degraded every
Penmanshiel cell (plain fill test mean 26.45 -> 42.86, win 62% -> 45%) and every METR-LA
cell. This is a third independent instance of the project's migration failure, after S6E's
Tobit (won synthetic mnar 2.52->1.97, lost on real curtailment) and S10's general finding:
**synthetic censoring geometry does not transfer to real curtailment**, now shown for
learned repairs and for data augmentation, not just for hand-designed fills.

**The binding constraint on real data is training windows, not the existence of the gap.**
The train/test diagnostic separates the readings: on the training half the learned fill wins
70% (mean 11.79 vs 38.08) versus 62% on the held-out half. The gap is exploitable in-sample
and only partly generalises from 296 real censored windows. Whether more real censored data
would close the synthetic-vs-real magnitude gap is untested and is the obvious follow-up.

## Caveats

- Oracle missingness mask (operator status codes / exact zeros). Deployment additionally
  needs the detector; S16 measures that separately (0.733 censoring recall status-free).
- Single model (chronos-bolt-base, frozen), single split (seed 20250812). The paired sign
  tests are over windows, not over splits; a repeated-split version would be stronger.
- METR-LA's learned-fill arm uses H=64 vs the stored H=96 (gate runs at 96 and passes).
- METR-LA's best fixed fill `nan` has a rank-0 reachable set and is unreachable by any
  value-fill class by construction; `linear` is reported as the best reachable baseline and
  the conclusion is unchanged against either.
