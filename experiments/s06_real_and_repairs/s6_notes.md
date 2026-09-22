# S6-E notes — censoring-aware imputation + conformal honesty wrapper

Same windows/seeds as S5 (masks identical; relMSE vs the shared clean baselines).

## Piece 1 — censoring-aware imputation (`run_s6_mnarfix.py`)

**Detector** (per window-channel, masked context only): features = r_bar (mean rank of
linear-bridge values at missing points within the observed CDF), e_bar (mean |rank−0.5|
of gap-flanking observed endpoints), x_bar, mean run length; fixed rule thresholds
(upper-censored if r_bar ≥ 0.62; two-tail if e_bar ≥ count-aware 3σ threshold or
x_bar ≥ 0.80; clustered if mean run ≥ 6; else random). Accuracy on the synthetic grids
(dataset-avg over p=0.1–0.7): **mcar 0.82–0.89, block 0.51–0.71, mnar_high 0.75–0.96,
mnar_extreme 0.55–0.86**. Main error modes: blocks that land on peaks look upper-censored
(information-theoretically ambiguous); both mnar mechanisms at p=0.7 partially look
"clustered" as censored regions merge. Failures are safe — they fall back to linear.

**Imputer**: detector-gated; for gaps whose endpoints are extreme-consistent, add a tent
bump on the linear bridge peaking at gap center. Two apex estimators: `tail_tri` (mean
excess of observed local extrema over their 3-point bridges, shrunk by n/(n+5)) and
`tail_tobit` (E[x|x>τ] under a median/MAD Gaussian with τ = observed max; symmetric for
the lower tail). tobit ≥ tri everywhere, so we report tobit.

**Verdict: yes, it beats linear under mnar_high — on ETT, by up to ~20% relative.**
bolt, relMSE vs clean at p=0.5/p=0.7: ETTh1 1.43→1.21 / 2.52→1.97 (**+15.3%/+21.9%**);
ETTm1 1.37→1.27 / 2.67→2.42 (+7.5%/+9.7%); weather 1.29→1.29 / 1.33→1.33 (±0%).
timesfm: ETTh1 +1.4%/+12.7%, ETTm1 +2.9%/+8.4%, weather ±0. Under mnar_extreme the gain
is smaller but consistent on ETT (+4.5–9.7% bolt); one negative cell: timesfm weather
p=0.7 (−7.0%). The extreme-future bias is reduced but not eliminated: bolt top-decile/rest
ratio under mnar_high moves 2.37→2.25 (ETTh1 p=0.7; mcar level ≈ 0.70) — the correction
is structurally conservative (observed peaks are the ones that *survived* censoring, so
the excess estimate is a lower bound). Reduce-to-linear cost when the detector false-fires:
≈0 on mcar (≤0.3%) except weather + tail_tri (+12–20% in one cell — tobit is the safe
default); ~2–3% on block (blocks on peaks get a spurious bump).

## Piece 2 — split-conformal wrapper (`run_s6_conformal.py`, bolt, linear fill)

Per dataset: 300 eval windows = the S5 windows; 300 disjoint calibration windows from the
same test region. Nonconformity E_t = max(q10_t − y_t, y_t − q90_t) per horizon step;
widening w_t = split-conformal quantile at 0.8.

- **Native intervals fail silently, as Track C found**: mnar_high coverage collapses
  0.75 → 0.67/0.53/0.39/0.25 (p=0.1–0.7, dataset avg) **while intervals narrow** to
  0.46× of clean width at p=0.7. mcar barely degrades natively (0.74→0.71).
- **Mechanism-matched conformal restores coverage to 0.79–0.81 at every rate** (mcar and
  mnar_high, all datasets). Width cost: ×1.05–1.07 under mcar; ×1.14/1.57/2.47/4.02
  under mnar_high at p=0.1/0.3/0.5/0.7 — honest intervals are expensive exactly where the
  model was confidently wrong. Per-step coverage is flat across the horizon after
  conformal (native coverage decays with step).
- **Mismatch cost is severe**: calibrate mcar → deploy mnar_high gives only
  0.73/0.60/0.47/0.35 — at p=0.7 the mismatched wrapper recovers just 0.10 of the 0.55
  coverage gap. The calibration set must match the missingness mechanism, not just the
  rate. (Rate-adaptive pooled calibration ≡ per-rate matched at grid rates by
  construction; kept for reference in the JSON.)

## Caveats

- Detector thresholds were tuned on the same synthetic grids used for evaluation
  (a priori reasoning + one feedback iteration on ETTh1 tiny) — treat absolute accuracy
  as indicative, not validated in the wild.
- The imputer helps where censoring is detectable and the series has recoverable peak
  structure (ETT); on weather (sparse near-zero channels) it is neutral.
- Conformal calibration uses synthetic masks at the deployment mechanism/rate — it
  assumes the mechanism is *known* (or detectable, cf. piece 1) at deployment.
- timesfm kept its S5 convention (100 windows, 1 seed); bolt 300 windows (mnar: 1 seed,
  deterministic masks; mcar sanity: 2 seeds).

Artifacts: `run_s6_mnarfix.py`, `run_s6_conformal.py`, `analyze_s6.py`,
`s6_mnarfix_results.json`, `s6_conformal_results.json`, `s6_mnarfix.png`,
`s6_conformal.png`, logs `s6_mnarfix_run.log`, `s6_conformal_run.log`.
