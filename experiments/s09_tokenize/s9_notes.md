# S9 — tokenizer-level attribution of Chronos-T5's missing-data degradation

Question: S5 only ever tested fill-then-feed for every model. Chronos-T5's
tokenizer (MeanScaleUniformBins) has a NATIVE NaN path — nan-aware scale =
mean(|x_observed|), NaN positions become pad_token_id with attention_mask=0
(excised, not filled). How does t5-nan compare to the S5 fills, and how much of
the damage travels through the scale statistic vs the token stream?

Setup: identical windows/masks/inference as S5 (t5 aux grid: 100 windows,
L=512, H=96, mask_seed 0, fp32, batch 128, median of 20 samples). Anchor gate
reproduced t5 × {clean, mcar:linear:0.3, mcar:zero:0.3} × ETTh1 from
s5_missing_results.json with **0.000% deviation** (bit-exact). Part B tokenizes
the same windows (ETTh1 + weather) with the real MeanScaleUniformBins
(n_tokens=4096, limits ±15, real bins 2..4095). Numbers below are dataset-avg
relMSE (MSE / clean MSE) at p = 0.1/0.3/0.5/0.7 unless stated.

## Part A — t5-nan completes the S5 grid (dataset-avg relMSE)

| mechanism    | fill   | p=0.1 | p=0.3 | p=0.5 | p=0.7 | verdict vs linear |
|--------------|--------|-------|-------|-------|-------|-------------------|
| mcar         | nan    | 1.013 | 1.060 | 1.138 | 1.328 | wins ≤0.5, ≈tie 0.7 (lin 1.037/1.066/1.156/1.306) |
| block        | nan    | 1.108 | 1.154 | 1.325 | 1.426 | **wins at 0.3–0.7** (lin 1.099/1.182/1.360/1.560) |
| mnar_high    | nan    | 0.988 | 1.233 | 1.442 | 1.845 | wins ≤0.3, **loses ≥0.5** (lin 1.008/1.244/1.438/1.709) |
| mnar_extreme | nan    | 0.995 | 1.232 | 1.308 | 1.380 | **loses everywhere** (lin 0.882/1.082/1.167/1.248) |

Same qualitative shape as bolt-nan (s5_fix): the native-NaN path is the best
default under spatial randomness (mcar/block) but loses to interpolation under
value censoring — excising censored extremes cannot fabricate the missing
amplitude, while linear interpolation at least fabricates a plausible ramp.
bolt-nan remains strictly better under mcar/block (1.131 / 1.085 @ p=0.7):
patch-16 + explicit mask channels + nan-aware InstanceNorm beats pad-and-skip.

Top-decile split (mnar_high p=0.7, relMSE over future points above/below the
clean-context q90): t5-nan ETTh1 td/rest = 3.38/1.76 vs linear 3.81/1.72 —
nan is slightly better ON extremes on ETTh1 but worse on ETTm1 (4.29 vs 3.33);
no consistent rescue of extreme futures. weather td ≈ 5.5 both.

## Part B — what missingness does to the token stream

**(1) scale_ratio (mode / clean).** The two sub-mechanisms separate cleanly:
- nan mode = **observed-set bias only**: unbiased under mcar/block (≈1.000 at
  all rates, ETTh1), but 0.911/0.786/0.689/0.626 under mnar_high — censoring
  large values shrinks mean|x_obs| by up to 37%.
- linear mode = **fill pollution**: mild downward drift under mcar/block
  (0.972–0.994 @ p=0.7), and 0.955/0.844/0.714/0.572 under mnar_high —
  interpolation toward the local mean compresses the amplitude as much as
  (p=0.7: more than) the observed-set bias.
- Since output_transform multiplies bin centres by scale, a 0.63× scale is a
  37% multiplicative under-prediction prior — the largest single effect found.

**(2) bin drift.** Surviving tokens are re-labelled massively even when the
scale is right: ETTh1 mcar p=0.7 bin_change_frac = 0.86 (nan) / 0.86 (linear)
despite scale_ratio ≈ 1.000 — sub-1% scale wobble is amplified by the 4093-bin
resolution (bin width ≈ 0.0073 scaled units); mean |Δbin| = 3.5/5.6. Under
mnar_high ~97–99% of surviving tokens change bin, mean |Δbin| = 53/67. Block
sits between (0.92, |Δbin| 6.2/7.2). Token-space "value pollution" is
near-total under value censoring.

**(3) resolution collapse.** The two modes corrupt the marginal token
distribution in OPPOSITE directions: nan subsamples the support
(distinct_bins/clean = 0.67 mcar / 0.78 block / 0.37 mnar_high @ p=0.7;
entropy/clean = 0.92/0.95/0.72), while linear fill SMEARS it — distinct bins
INFLATE to 1.52×/1.50× clean under mcar/block @ p=0.7 (interpolant ramps land
in bins between observed clusters; entropy 1.06×) but still collapse to 0.70×
under mnar_high. Neither direction is "the clean distribution".

**(4) edge bins — a non-channel.** Saturation never happens: ETTh1 edge_frac =
0.0000 everywhere (Spearman undefined, reported as null); weather 0.13% clean,
flat or decreasing with rate. The ±15 limits are never reached because
scale ≥ max|x|/n_obs keeps scaled values well inside.

**(5) money plot (|scale_ratio−1| vs per-window relMSE, Spearman ρ).** MUST be
cut per dataset — pooled ρ ≈ 0 is Simpson cancellation:
- ETTh1 (the clean read): nan mode ρ = +0.06/+0.18/**+0.39** (mcar/block/
  mnar_high, p ≈ 2e-1/2e-4/1e-15); linear mode +0.12/+0.07/+0.31. The more
  informative the missingness, the more relMSE tracks scale distortion.
- weather: INVERTED (ρ = −0.02/−0.10/−0.23 nan; −0.19/−0.10/−0.21 linear).
  The windows with the largest scale_ratio are degenerate all-zero rain
  channels whose relMSE *drops below 1* under missingness (see anomaly 2).

## Anomalies / caveats

1. **Tokenizer fallback `scale[~(scale>0)] = 1.0` fires in production** — the
   spec's assumption "rate ≤ 0.7 won't trigger it" is wrong for degenerate
   channels. weather `rain (mm)` is 95% zeros (clean mean|x| = 0.0055) and
   `raining (s)` 84% zeros: 63/2100 window-channels hit the fallback even in
   CLEAN contexts; under mnar_high p=0.7 it is 524/2100 (censoring the spikes
   leaves all-zero observations). The fallback silently switches the series to
   absolute units (scale_ratio ≈ 26 mean on weather). ETT datasets: 0 rows.
2. **weather relMSE < 1 under mnar** (nan @ p=0.1: dataset-avg 0.988, weather
   alone 0.72): missing rain spikes -> model predicts ~0 -> error DROPS vs
   clean on all-zero futures. Same family as S5's "zero-fill ≈ clean on
   weather" artifact; weather mnar numbers should not steer conclusions.
3. Pooled-across-datasets Spearman ≈ 0 (Simpson's paradox, anomaly 2's flip).
4. block p=0.7 achieves only 0.509 actual missing (overlapping runs) —
   identical to S5 (same seeds), so pairing is unaffected.
5. Anchor reproduction was bit-exact (0.000%) — same seeding, same A800s; a
   different GPU model would likely break bit-exactness but stay within the 5%
   gate.

## Bottom line

t5's degradation decomposes into (a) a **scale-statistic channel** —
nan mode keeps it honest under random missingness (≈1.00) but inherits the
observed-set bias under value censoring (0.63× @ mnar_high p=0.7), while linear
fill pollutes it by interpolation (0.57×); per-window relMSE tracks this
channel on ETT (ρ up to +0.39); and (b) a **token-stream channel** — 86–99% of
surviving tokens are re-binned and the marginal support is either subsampled
(nan) or smeared (linear, +52% distinct bins). Edge-bin saturation plays no
role. Practically: feeding Chronos-T5 raw NaN beats every fill under
mcar/block; under value-censored missingness, linear interpolation is still
the least bad option — no input-side trick recovers censored amplitude
(consistent with the S5/S6 conclusion).

Artifacts: `run_s9_tokenize.py`, `s9_results.json` (Part A per-config +
per-window, Part B full stats, correlations, anchor gate), `s9.png`,
`s9_anchor.json`, `s9_partA_{ETTh1,ETTm1,weather}.json`, `s9_partB.json`,
logs `s9_partA_*.log`, `s9_partB.log`.
