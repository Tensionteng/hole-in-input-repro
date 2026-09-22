# S14 notes — second real missingness mechanism: METR-LA sensor failures

**Question:** S6/S10 (Penmanshiel 2016 wind curtailment) showed the optimal
context fill depends on the missingness geometry: under value-censored (MNAR)
contexts zero-fill beat linear by 2–4.5× (mean-NMSE ratios zero/linear =
0.47 bolt / 0.41 timesfm / 0.57 moirai) and Tobit failed. S14 tests the
converse on a **non-value-dependent** real mechanism: METR-LA loop-sensor
failures (0-encoded, blocky, value-independent — exploration confirms:
boundary-speed distributions identical to marginal, hourly miss-rate vs speed
r=0.027). See `dataset_metrla/DATA_README.md` for provenance (canonical
DCRNN file via HF mirror, signature verified) and exploration numbers.

## Pre-registered predictions (written 2026-08-13 BEFORE any full model run;
smoke tests on 5 windows had been run and are consistent but uninformative)

- **P1 (main claim):** on miss windows, the fill ranking REVERSES vs
  Penmanshiel — zero/keep damage, linear wins:
  mean NMSE(zero)/mean NMSE(linear) ≥ 1.5 for bolt AND timesfm (Penmanshiel
  direction was 0.41–0.57, i.e. zero won by ~2×). Median-NMSE ratio in the
  same direction but smaller.
- **P2 (native NaN):** bolt nan ≈ linear (mean-NMSE ratio within [0.7, 1.4];
  bolt masks NaN in attention = true native handling). timesfm nan ≈ linear
  almost exactly (timesfm internally np.interp's NaNs — same operation;
  residual difference only from leading-NaN stripping). moirai nan ≤ linear
  (native observed-mask).
- **P3 (sanity):** on fully-observed ctrl windows all fills are bit-identical
  to clean (zero/linear/nan never touch observed values; moirai seeded
  per-call in S14). On METR-LA, keep ≡ zero exactly (missing is stored as 0).
- **P4 (detector):** S6/S10 bridge-rank detector on the 618 miss windows
  labels <15% "upper"+"two-tail" (Penmanshiel oracle mask: 65.1%) —
  clustered/random dominate; mean r_bar within [0.4, 0.6] (value-independent).
- **P5 (conformal):** under the winning fill (linear/nan), naive (clean-cal)
  split-conformal coverage on miss test windows lands within [0.85, 0.95] —
  no large clean→damaged transfer gap like Penmanshiel (native 0.71–0.77,
  clean-cal 0.80–0.89); mech-aware within ±0.02 of naive (missing windows are
  not a harder regime once filled). Under zero fill coverage degrades
  (directional).
- **P6 (persistence):** mse/pers(zero) > 1.3 × mse/pers(linear) on miss
  windows for bolt/timesfm (zero-fill erodes the model edge); all models beat
  persistence on ctrl clean (ratio < 1).
- **Anchor expectation:** ctrl clean mean NMSE same order of magnitude as
  Track D (bolt 9.91, timesfm 5.85 — wind power is intrinsically harder than
  traffic, so S14 values plausibly lower; gate band ratio [0.02, 50]) and
  mse/pers < 1 on ctrl. A pipeline bug would show as order-of-magnitude
  failure or broken identities.

## Setup (Track-D-aligned; details in DATA_README.md)

METR-LA, univariate per sensor; CTX=512 (42.7 h), H=96 (8 h); 618 miss
windows (context missing ≥5.1%, target fully observed; rate mean 0.232 /
median 0.191 / p90 0.512 / max 0.826) + 618 ctrl windows matched 1:1 per
sensor by nearest time-of-day (median match 0 min). Fills: keep (=as-recorded
0s), zero, linear (np.interp), nan (model-native). Models: chronos-bolt-base,
timesfm-2.5-200m, moirai-1.1-R-base (median of 20 samples, seeded per call in
S14). Metrics: NMSE = MSE/max(var(target), 4.0 mph²), mean+median;
persistence = last observed context value (primary; last-as-recorded variant
stored — Track D's as-recorded convention is undefined when the last context
point is a missing-encoded 0); top-decile split kept for Track D parity.
Conformal: native [q10,q90] → per-step split conformal target 0.9, cal/test =
first/second half BY TIME within group; naive=ctrl-cal, mech-aware=miss-cal,
pooled. Detector: S6 rules verbatim. Script `run_s14_metrla.py`; results
`s14_metrla_results.json`; logs `s14_*.log`.

## Results (filled after runs)

All runs done 2026-08-13, GPUs 4–7 (bolt cuda:4, timesfm GPU5 via
CUDA_VISIBLE_DEVICES, moirai cuda:6); gate 19/19 PASS (`s14_gate.log`):
ctrl fills bit-identical to clean for all models; miss keep≡zero identical
(METR-LA missing IS 0); ctrl clean NMSE same order as Track D (bolt 3.20 vs
9.91, timesfm 2.60 vs 5.85, moirai 2.30 vs 7.20 — ratios 0.32–0.45, traffic
intrinsically more predictable than wind power); all models beat persistence
on ctrl (0.58–0.87); all NMSE finite.

### Main table (618 miss / 618 ctrl windows; mean NMSE | median NMSE | mse/pers)

| model | ctrl clean | miss keep | miss zero | miss linear | miss nan |
|---|---|---|---|---|---|
| bolt | 3.20 / 1.300 / 0.78 | 21.19 / 1.415 / 1.69 | 21.19 / 1.415 / 1.69 | 2.75 / 1.243 / 0.78 | **2.00 / 1.234 / 0.74** |
| timesfm | 2.60 / 1.206 / 0.58 | 1.91 / 1.016 / 0.48 | 1.91 / 1.016 / 0.48 | 1.80 / 1.110 / 0.56 | 1.78 / 1.113 / 0.53 |
| moirai | 2.30 / 1.568 / 0.87 | 14.79 / 2.024 / 1.59 | 14.79 / 2.024 / 1.59 | **2.52 / 1.486 / 0.81** | 2.66 / 1.540 / 0.82 |

zero/linear mean-NMSE ratio: **bolt 7.71×, moirai 5.86×** (Penmanshiel:
0.47/0.57 — zero won there; full reversal) but **timesfm 1.06×** (median
0.92× — zero even slightly better). Paired over windows: zero worse than
linear on 59.2% (bolt, Wilcoxon p=2.8e-12) / 69.4% (moirai, p=2.4e-37) /
41.9% (timesfm, p=1.2e-06) of windows.

### Why timesfm is the exception (fill-level bias diagnostic, by missing-rate tercile)

Forecast-mean minus target-mean (mph), low/mid/high missing rate:

| model | low (8%) | mid (24%) | high (55%) | mse zero/lin (low/mid/high) |
|---|---|---|---|---|
| bolt zero-fill | −0.25 | −2.49 | **−11.51** | 1.15 / 2.19 / 3.91 |
| bolt linear | +0.70 | +0.40 | +2.90 | |
| timesfm zero-fill | +0.81 | +1.65 | +1.95 | 0.95 / 0.85 / 0.73 |
| timesfm linear | +1.48 | +1.38 | +3.79 | |

bolt reads the fill literally: zero blocks → fake standstill regime →
under-prediction growing with dose (−11.5 mph bias at 55% missing).
timesfm (per-input normalization + zero-rich pretraining priors, e.g.
intermittent series) re-anchors near the observed level: zero-fill bias stays
≤ +2 mph at any dose, and its linear-fill overshoots more (+3.79) — linear
ramps across long gaps create fake drift that timesfm extrapolates. So the
reversal is real but MODEL-DEPENDENT: it holds for models that take the
context level at face value (bolt, moirai), not for timesfm.

### Conformal (miss_test, native/naive/pooled/mech-aware coverage; target 0.90)

| model/fill | native | naive (ctrl cal) | pooled | mech-aware (miss cal) |
|---|---|---|---|---|
| bolt/zero | 0.796 | 0.950 | 0.947 | 0.941 |
| bolt/linear | 0.710 | 0.935 | 0.916 | **0.897** |
| bolt/nan | 0.685 | 0.940 | 0.918 | 0.882 |
| timesfm/zero | 0.856 | 0.960 | 0.946 | **0.909** |
| timesfm/linear | 0.783 | 0.940 | 0.934 | **0.921** |
| timesfm/nan | 0.785 | 0.940 | 0.933 | **0.919** |

ctrl_test: native 0.688 (bolt) / 0.769 (timesfm); naive-cal 0.958 / 0.952.
Opposite direction vs Penmanshiel: there clean→censored transfer
UNDER-covered (0.80–0.89, silent failures); here naive OVER-covers on miss
windows (0.935–0.960) — filled miss windows are slightly easier than clean
ctrl, not harder. Mech-aware calibration is still the closest to target in
5/6 cells (0.882–0.941), it just matters less.

### Detector (S6/S10 bridge-rank rules on the 618 miss masks)

Labels: upper 28.3%, two-tail 26.4%, clustered 45.1%, random 0.2%.
r_bar: mean 0.472, median 0.461, p10 0.160, p90 0.812; 28.5% of windows ≥0.62
AND 37.7% ≤0.38 (symmetric tails). Aggregate confirms value-independence,
but the per-window labels are unstable: with blocky masks the value-rank
features measure WHERE the block landed relative to the window's own regime
mix, not selection-on-value. The S6 detector was built for scattered masks
(CTX=144, run≈1); under long blocks its "upper"/"two-tail" rules misfire at
chance level in both directions.

## Verdicts vs pre-registration

- **P1 (fill reversal): CONFIRMED for bolt (7.71×) and moirai (5.86×),
  REJECTED for timesfm (1.06× mean / 0.92× median).** The mechanism-dependence
  claim survives with a model-dependence clause: the optimal fill is decided
  by (missingness geometry) × (model's input convention). On Penmanshiel
  timesfm agreed with the others because zero-fill happened to suit it there;
  on METR-LA it is simply insensitive.
- **P2 (native nan):** bolt nan/linear 0.727 mean / 0.993 median ✓ (within
  [0.7, 1.4]; nan is bolt's best fill). timesfm nan ≈ linear (0.989/1.002) ✓
  (it internally np.interp's — verified). moirai nan ≈ linear (1.054/1.036;
  pre-registered ≤, got ≈). ✓ mostly.
- **P3 (identities): ✓ exact** for all three models (gate max|ΔNMSE| = 0).
- **P4 (detector): REJECTED as pre-registered** (<15% upper/two-tail
  predicted; got 54.7%). The aggregate r_bar (0.472 ∈ [0.4,0.6]) ✓ diagnoses
  value-independence, but per-window labels are chance placements under block
  geometry — a documented detector limitation, not evidence of MNAR.
- **P5 (conformal): mostly confirmed.** naive on miss test 0.935–0.940 ∈
  [0.85, 0.95] ✓ but over-covers (direction flipped vs Penmanshiel, as
  expected from miss-not-harder). mech-aware within ±0.02 of naive: timesfm ✓
  (0.921 vs 0.940), bolt ✗ (0.897 vs 0.935, diff 0.038 — mech-aware closer to
  target). Zero-fill native coverage degrades (0.796/0.856) ✓ directional.
- **P6 (persistence):** bolt ✓ (zero 1.69 vs linear 0.78 → 2.17× > 1.3;
  zero-fill flips bolt BELOW persistence). timesfm ✗ (0.48 vs 0.56 — zero
  improves its relative standing, same exception as P1). ctrl clean all <1 ✓.

**Bottom line:** the Penmanshiel conclusion — "the best fill depends on the
physical mechanism" — replicates in reverse on METR-LA for 2 of 3 models:
under value-independent block missingness, zero-fill is catastrophic for
level-reading models (up to 7.7× mean NMSE, worse than persistence), while
linear/native-nan are essentially free (miss linear/nan ≈ ctrl clean on
median NMSE and mse/pers). timesfm-2.5 breaks the pattern: its input
normalization absorbs zero blocks (bias ≤2 mph at 55% missing), so its fill
ranking did NOT reverse. Mechanism × model convention jointly determine the
best fill; neither alone suffices.

## Caveats

- keep ≡ zero exactly on METR-LA (missing stored as 0.0); both were run and
  used as an identity check, not two independent baselines.
- Outages are system-wide synchronized (missing-mask corr 0.81): miss windows
  cluster in calendar time; the conformal time-split shares episodes between
  cal and test (same caveat as S10's farm-wide curtailment episodes).
- 10.4% of miss windows have >50% context missing (max 82.6%) — no rate cap,
  as Track D; dose-response reported above instead.
- Mean NMSE is tail-driven; miss-linear/nan mean NMSE < ctrl clean while
  medians and mse/pers are comparable — a variance/regime mix effect of the
  NMSE normalization (floor 4 mph²), not "missing helps". Per-window
  variances stored for sensitivity.
- moirai seeded per call in S14 (Track D/S10 were unseeded); 20 samples,
  median. bolt nan = true attention masking; timesfm nan = internal np.interp
  (so its nan≈linear is by construction, verified as a sanity).
- Data provenance: official DCRNN repo no longer hosts metr-la.h5 (Drive/
  Baidu only); used exact-copy HF mirror, canonical signature verified
  (34272×207, 2012-03-01..06-27, 8.11% zeros, max 70 mph). See DATA_README.
- Detector labels (upper/two-tail/clustered) are unreliable under blocky
  masks; only its aggregate r_bar is informative here.
