# S6 notes — real value-censored context: Penmanshiel 2016 wind curtailment

**One line:** on real, documented curtailment-censored contexts all three
zero-shot TSFMs degrade well beyond what window difficulty explains at the
mean (4–12× clean mean-NMSE as-observed), and the synthetic MNAR signature
replicates on the same series with paired masks — but the median effect is
modest (1.1–1.3×) and the error is a broad level shift, not peak-specific.

## Dataset and censoring evidence (how clean is it?)

Penmanshiel wind farm 2016 (Cubico, Zenodo 16807304, CC-BY-4.0), 14× Senvion
MM82, 10-min SCADA + per-turbine Status event logs. Curtailment is
**documented in the operator's own event log**: code 9000 "P output externally
reduced" / 9210 "Externally stopped" (IEC "Partial Performance"), covering
29–37% of observed intervals per turbine in 2016 (farm-wide episodes).
Effective (binding, high-value) censoring = flag ∩ {q95 power-curve available
power − observed power > 15% rated, available > 30% rated}: 0.5–3.5% of
intervals, and **monotone in wind speed** (0% in the lowest wind quintiles,
1.5–10.7% in the top) — genuine value-dependent (MNAR) censoring. Visual
validation: censored points form horizontal cap lines (~1650/1000/750/500/0
kW) cutting the power curve (`validate_scatter.png`, figure panel A).
Evidence quality: strong on provenance (operator log) and value-dependence;
the counterfactual itself is unobservable (avail curve is only an estimate).

## Setup

Univariate power; context 144×10-min (24 h), horizon 24 (4 h). 602 censored
windows (≥7 binding points; targets always fully observed & uncurtailed) +
602 clean windows matched 1:1 per turbine; per-window censor rate mean 0.139
(range 0.05–0.63). Fills: linear / zero / as-observed. Controls: clean;
clean+MCAR and clean+rank-based top-value (synth-MNAR) masks at the paired
per-window rate. Models: chronos-bolt-base, timesfm-2.5-200m,
moirai-1.1-R-base (median point forecasts). `run_s6_real_censor.py`,
incremental results in `s6_real_censor_results.json`.

## Degradation numbers vs control (mean NMSE / median NMSE)

| config | bolt | timesfm | moirai |
|---|---|---|---|
| clean control | 9.9 / 1.73 | 5.9 / 1.63 | 7.2 / 2.01 |
| REAL cens as-observed | 40.3 / 2.16 | 72.6 / 2.10 | 40.4 / 2.20 |
| REAL cens linear-fill | 60.2 / 2.12 | 85.5 / 1.96 | 50.3 / 2.33 |
| REAL cens zero-fill | 28.4 / 1.66 | 34.8 / 2.21 | 19.0 / 2.22 |
| synth-MNAR zero (matched) | 162.3 / 2.02 | 134.6 / 2.00 | 140.0 / 2.19 |
| MCAR linear (matched) | 8.4 / 1.79 | 5.5 / 1.65 | 8.0 / 2.07 |

- As-observed real censoring: **4.1× / 12.4× / 5.6× mean NMSE vs clean**
  (bolt/timesfm/moirai). After variance-matching on target variance the
  median ratio is only **1.08–1.29** — censored windows follow high-wind
  regimes and are intrinsically harder (median target var 63k vs 19k kW²).
- Difficulty-adjusted: the models' advantage over persistence collapses on
  censored windows — MSE/persistence goes from 1.04 (clean) to 1.47 (cens)
  for bolt, 1.03→1.32 timesfm, 1.15→1.21 moirai.
- Dose-response on real windows (bolt, as-observed): median NMSE 1.70 → 2.30
  → 2.77 across censor-rate terciles (Spearman ρ=0.166, p=4e-5; linear fill
  ρ=0.148, p=3e-4). timesfm/moirai: not significant.
- **Paired synthetic replication on the same real series** (clean
  counterfactual exists): rank-based top-value masking + zero-fill inflates
  the top-decile future MSE by a median 1.68×/1.55×/1.83×
  (bolt/timesfm/moirai) and mean NMSE by 5.5–19.9×, while rate-matched MCAR
  + linear fill is free (ratio ≈ 1.00). This is the synthetic study's MNAR
  signature, reproduced on real data (figure panel C).

## What failed / caveats (honest)

- The real-data top-decile/rest ratio does **not** exceed the control
  (0.8–3.2 vs 2.2–3.2): real curtailment clamps long contiguous runs, so the
  model reads a false low-power regime and under-predicts the *whole* target
  (level shift), unlike scattered synthetic mnar_high spikes. The
  peak-specific signature is clean only in the paired synthetic masks.
- Median effects are modest; the big mean ratios are tail-driven (a minority
  of catastrophic windows). Confound: censoring co-occurs with stormy,
  hard-to-forecast regimes; variance-matching and the persistence comparison
  only partially remove it.
- Kelmarsh (all years 2016–2024 probed): essentially no curtailment (only
  WT4-2023, 944 h of short "Technical curtailment", likely bat/noise).
  Penmanshiel 2021–2022: ≤52 grid-curtailment events/yr — too sparse; 2016
  was the rich year. PhysioNet'12 (tsdm cache): no detection-limit value
  spikes found. ENGIE La Haute Borne: not needed after Penmanshiel panned out.
- 2016 predates the "Turbine Power setpoint" signal and Greenbyte's
  lost-production-to-curtailment ledger is all-zero in 2016; the Status event
  log is the documented source instead.
