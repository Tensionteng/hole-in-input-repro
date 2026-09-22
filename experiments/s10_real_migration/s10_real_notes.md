# S10 notes — S6 fixes migrated to REAL curtailment censoring (Penmanshiel 2016)

**One line:** the two S6 fixes split on real data — **mechanism-aware conformal
transfers** (censored-window coverage restored to 0.90–0.92 at the 0.9 target
where clean-only calibration stalls at 0.80–0.89), but **Tobit imputation does
not**: the synthetic mnar_high geometry it repairs (peaks *removed*, threshold
known) is not the real geometry (values *clamped to an observed cap*, future
level mean-reverts), and on 602 real curtailed windows Tobit loses even to
linear fill, while zero-fill stays best. Same pipeline as Track D (window list
verified byte-identical; bolt/timesfm anchor numbers reproduce to 0.00%).

## Real censoring geometry (results["explore"]; 5268 effective-censored points)

Effective censoring mask as in Track D (status codes 9000/9210 ∩ binding
rule). Observed-power / available-power ratio at censored points is **bimodal**:

- **32% at ratio < 0.02** — code 9210 "externally stopped": output forced to
  ~0 or negative (turbine motoring). 33.5% at ratio < 0.1.
- **41% at ratio 0.70–0.85** (+13% at 0.5–0.7) — code 9000 clamps at plateau
  caps; recorded-power histogram peaks at 0 kW (32%), ~1700 kW (21%),
  500–700 kW (13%): caps ~1650/750/500/0 kW as in Track D's visual check.
- Would-be power at censored points (q95 power curve): median 1751 kW,
  p90 2059 kW — censoring binds at HIGH values (MNAR), confirmed again.
- Within-mask runs are mostly short (median 1, mean 4.4 10-min points, 12.5%
  ≥ 6, max 146) but windows were selected to hold ≥ 7 censored points.

So real curtailment **is** top-censoring, `y_obs = min(y_true, tau_event)` —
but with an **unknown, per-event, time-varying threshold** (and a forced-stop
sub-mechanism at tau ≈ 0). Unlike synthetic mnar_high, the censored value is
not deleted; it is observed AS the cap.

## Track 1 — Tobit migration

**Threshold estimators** (the synthetic study knew tau; here it is estimated):
- `tobit`: per censored run, tau_e = median of the *recorded* (clamped) values
  in the run — the plateau IS the cap. Fill = E[x|x > tau_e] = mu +
  sigma*lambda((tau_e−mu)/sigma) under a median/MAD Gaussian fit on the
  window's **uncensored** values, clipped to [0, 2050]. For forced-stop runs
  (tau_e ≈ 0) this degenerates to ≈ the context median.
- `pc` (geometry-matched variant): fill = clip(max(avail(wind_t), y_obs), 0,
  2050) — the per-turbine q95 power-curve available-power mapping, floored at
  the observed cap (censoring constraint y_true ≥ cap). Uses wind, not the
  operator log.

**Results** (602 real curtailed windows; Track D metrics; moirai = S10 draw,
its anchor medians matched Track D within 1.7–3.8%):

| model | clean | keep | zero | linear | tobit | pc |
|---|---|---|---|---|---|---|
| bolt (mean NMSE) | 9.91 | 40.29 | **28.43** | 60.16 | 65.01 | 66.25 |
| timesfm | 5.85 | 72.63 | **34.82** | 85.54 | 93.48 | 144.57 |
| moirai | 6.15 | 36.12 | **25.24** | 44.50 | 56.40 | 49.13 |
| bolt (median NMSE) | 1.73 | 2.16 | **1.66** | 2.12 | 2.22 | 2.17 |
| timesfm (median) | 1.63 | 2.10 | 2.21 | 1.96 | **1.79** | 2.10 |
| moirai (median) | 2.07 | 2.12 | **2.18** | 2.26 | 2.38 | 2.39 |

**Verdict: reversal, not confirmation.** Tobit loses to linear on mean NMSE
for all three models (and pc catastrophically for timesfm, 144.6); zero-fill
remains the best repair exactly as Track D found. The only Tobit wins are
timesfm's median NMSE (1.79 vs 1.96 linear; and raw MSE 279k vs 293k) —
NMSE up-weights the low-variance windows where high fills hurt most.

**Why the synthetic fix fails here** (fill-level diagnostic, mean over
censored points): recorded cap 864 kW; Tobit fill 1578 kW; pc fill 1937 kW;
avail 1943 kW — but the **4h-ahead target mean is only 942 kW**. Curtailment
is called during high-wind ramping episodes that subsequently decay; the
would-be power during the censored context is a *biased estimate of the future
level*. Filling high teaches the model a false high regime and inflates the
whole forecast (Track D's level-shift failure mode), while zero/keep anchor
low and win. The synthetic mnar_high damage Tobit repaired was peak-specific
bias at a *known* threshold with a *stationary* level — neither holds here.
There is no universal fill: the right repair is decided by the geometry *and*
by what the fill implies about the future regime.

**Threshold sensitivity** (tau_e ± 10%, mean NMSE): bolt 65.25/65.01/64.93
(±0.4% — insensitive); moirai 54.10/56.40/56.04 (−4.1%/−0.6%); timesfm
104.31/93.48/104.67 (+11.6%/+12.0%, non-monotone — both perturbations worse,
the plateau is a local optimum for timesfm). Conclusion: where Tobit fails it
fails for structural reasons (fill level), not threshold mis-estimation; the
estimator itself is robust.

## Track 2 — mechanism-aware conformal migration

bolt + timesfm native [q10,q90] (80% nominal), per-step split conformal,
target 90%; per-turbine 50/50 split → 301 cal / 301 test windows per group.
naive = calibrate on CLEAN windows; mech-aware = calibrate on real CENSORED
windows (status-code mask = oracle indicator); pooled = blind mix.

**Coverage on censored test windows** (width in kW):

| model/fill | native | naive | pooled | mech-aware |
|---|---|---|---|---|
| bolt/zero | 0.900 (1304) | 0.945 (1548) | 0.936 (1454) | **0.921** (1374) |
| bolt/linear | 0.710 (950) | 0.804 (1194) | 0.864 (1426) | **0.903** (1646) |
| timesfm/zero | 0.767 (1045) | 0.870 (1385) | 0.888 (1490) | **0.904** (1604) |
| timesfm/linear | 0.733 (895) | 0.841 (1235) | 0.884 (1484) | **0.914** (1731) |

**On clean control windows** (does the wrapper break normal windows?):
naive (clean→clean) lands on target (bolt 0.904, timesfm 0.911) as split
conformal must. mech-aware deployed on clean windows miscalibrates BOTH ways:
bolt/linear 0.962 (over-wide, 1336 vs 884 kW) but bolt/zero 0.823 (too narrow
— cens-calibrated widening is small because zero-filled censored contexts
already produce very wide native intervals, 1304 vs 640 kW clean). So the
mechanism-aware wrapper should be **gated per window by the detector**, not
applied globally.

**Verdict: the S6 conclusion transfers.** Under the damaging fills
(linear, and zero for timesfm) native intervals fail on real censoring
(0.71/0.73/0.77), clean-only calibration cannot fully repair them
(0.80–0.89), and only mechanism-matched calibration hits the 0.9 target
(0.90–0.92). New wrinkle vs synthetic: bolt+zero native coverage is already
0.900 — zero-filled clamped contexts make bolt *visibly* uncertain (intervals
2× wider than clean), so the silent-failure mode is fill-dependent. Width
cost of honesty: ×1.05 (bolt/zero) to ×1.83 (timesfm/linear) over native.

## Detector (no operator annotation needed?)

S6 bridge-rank rules on the 602 censored windows:
- **Oracle mask** (status codes): 56.1% "upper" + 9.0% "two-tail" = **65.1%
  detected as censored**; 25.2% "clustered", 9.6% "random" (safe fallbacks).
- **Auto mask** (status-free candidate: avail−power > 0.15·rated &
  avail > 0.30·rated): candidate mask has recall 1.000 / precision 0.642 vs
  the oracle effective mask; detector then labels **55.6% censoring-like**
  (46.2% upper + 9.5% two-tail), 33.2% random. On clean windows the candidate
  mask fires ≥3 points in 126/602 windows, of which 49.2% get labeled "upper"
  — those are real power-curve undershoot events (unlogged derating/icing
  etc.), so the false-alarm cost is a spurious high fill on ~10% of clean
  windows if the Tobit gate were opened (which Track 1 shows you should not do
  here anyway).

## Failure modes / caveats (honest)

- Tobit-on-real-data fails because the would-be level is anti-informative
  about the future level in curtailment regimes (mean-reverting episodes);
  do not read this as "Tobit is wrong" — it remains the S6 winner where the
  mechanism is stationary top-deletion (ETT). The reversal is about
  **mechanism semantics**, not estimation quality (threshold sensitivity is
  small).
- moirai forecasts by sampling (20 samples, unseeded in both Track D and S10):
  mean NMSE is tail-draw noise (Track D vs S10 anchor means differ up to 33%
  while medians agree within 1.7–3.8%). Its anchor gate was therefore on
  median NMSE. moirai skipped for track 2 (20 samples too coarse for 90%
  tails).
- Conformal cal/test is a random per-turbine split — curtailment arrives in
  farm-wide episodes, so cal and test share episodes; a blocked-by-episode
  split would be sterner. mech-aware calibration assumes censored windows are
  identifiable at deployment (status log or ~56% detector recall).
- avail (q95 power curve) is an estimate of the would-be value, used only for
  detection and the pc fill, never as an eval target.
- Control-window evaluations of the fills are the identity by construction
  (no mask → no fill); the "does it break clean windows" question is answered
  by the detector false-alarm rate and the conformal ctrl_deploy rows.

## Artifacts

`run_s10_real.py` (modes: explore, anchor, gate, track1, track2, detector,
finalize, figure; --model/--device/--shard/--smoke), `s10_real_results.json`
(anchor + track1 + track2 per-config and per-window metrics, explore,
detector, gate), `s10_real.png` (4 panels: geometry, fills, conformal,
detector), logs `s10_anchor_*.log`, `s10_track{1,2}_*.log`, `s10_explore.log`,
`s10_gate.log`, `s10_smoke*.log`, `s10_figure.log`.
