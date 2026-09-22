# S73 — absolute-scale CPT-vs-vanilla win rates and deltas

Reviewer question: the paper's missingness tables report relMSE against each model's
OWN clean accuracy plus closure shares. On the RAW absolute scale, what are the win
rates and mean deltas of the CPT retrofit vs vanilla? And is there any dataset where
the retrofit's clean accuracy slightly degrades AND its absolute error under
missingness is still worse than vanilla (relative metrics masking an absolute loss)?

Date: 2026-09-10. All artifacts in `tsfm_missing/experiments/s73_absolute/`.

## TL;DR — direct answers

**(a) Absolute win rates (bolt panel, the exact Table-5 cells: Q50-base vs stock
bolt-base, rate 0.7, declared path, 150 windows/cell).** Pooled over all
9 datasets x 4 mechanisms (66,668 series-rows per fill): **zero fill 74.0%,
linear 62.1%, oracle 61.1%** of rows have strictly lower absolute MSE for the
retrofit (all Wilcoxon p ≈ 0). Per dataset x fill, the win rate is **above 50%
on all 27 cells** (min 50.5%, weather/oracle; max 89.4%, illness/zero); every
per-dataset median absolute delta is negative (favors CPT) on all 27 cells.

**(b) The worry cell (clean slightly worse AND absolute-under-missingness worse).**
Under the paper's median convention: **no dataset is in the worry cell on any
fill.** 7 of 9 datasets pay a small clean cost (median clean-MSE ratios
1.018–1.131, median over datasets 1.042); all 9 are absolutely better under
missingness on all three fills by both median delta and per-row win rate. The one
partial exception under a mean-based readout is **weather**: clean +13.1%, and its
MEAN absolute error under oracle and linear fills is worse for the retrofit
(+3106 and +265 MSE) — entirely a tail effect: the top 10 of 12,600 rows supply
61–65% of the total positive delta, the median row is an exact tie (median delta
≈ −1e-14), and the per-row win rate is still ≥ 50.5%. Under zero fill weather is
better on every statistic. So the relative metrics are not masking a *typical-row*
absolute loss anywhere; they do mask a sparse weather tail on the two
imputation fills, which we now report explicitly.

**(c) Rebuttal-ready paragraph** — see bottom of this file.

## Provenance and method

### Why a re-eval was necessary (bolt) and sufficient (c2/m2)

- All stored DS9 grids (s57_bolt.json, s57_q50.json, s60/s61/s65 sweeps,
  s64_leaderboard_cells.json) keep, per cell, only the median/mean of the
  per-window relMSE ratio `w[i] = miss[i] / clean_own[i]`, and only median/mean
  of the clean MSE. Per-window clean errors are stored **nowhere**, so absolute
  per-window errors `w[i] x clean_own[i]` are **not reconstructable** (clean
  per-window variance spans orders of magnitude, e.g. ETTh1 median 1.2 vs mean
  11.8 — a median-imputed reconstruction would be fabricated precision).
- The GIFT dose-response (s64_bench3/s64_gift_scores.npz) **does** store
  per-window ABSOLUTE MASE (n=7,725) for c2-stock/c2-w50/m2-stock/m2-w50 x
  {nan, linear, zero, ffill}, so the c2/m2 absolute question is answered from
  stored data, no re-eval.

### Checkpoint identification (the paper's bolt CPT row)

The task allowed "P5-base or the WiSE-FT W50 checkpoint". Verified numerically:
**the paper's bolt "+CPT (ours)" row in tab:leaderboard is Q50-base**
(P8-base CPT x WiSE-FT 0.5, `s45_ckpt/arm_Q50_base.pt`): its per-dataset
perfect-fill medians from s57_q50.json are 1.28/1.27/1.18/1.22/1.16/1.73/1.61/
1.17/1.08 (med 1.22) — every digit of the paper row; P5-base is the blend-only
ablation that is zero-fill blind (7–520) and is NOT the paper row. Vanilla is
"bolt-base-stock" = arm P0 = `s45_ckpt/arm_P0_base.pt`, the stock
chronos-bolt-base dump (G2' max|diff| = 0 vs HF, s56 notes). The c2/m2 paper
rows are likewise the WiSE-FT-0.5 arms (c2-w50, m2-w50 in s64_leaderboard_rows).

### Re-eval harness and gate

`eval_s73.py` reuses `eval_s57_leaderboard.py` verbatim (same imports, same
seeded window/mask generators, TF32 state inherited from train_s45 exactly as
in the original run), but builds each batch once and scores both arms on the
identical in-memory arrays, recording per-row absolute MSE. Grid: DS9 x 4 mechs
x rate 0.7 x {zero, linear, oracle} + clean, 150 windows/cell, declared path.
**Gate: all 216 relMSE cell medians and all 18 clean medians reproduce the
stored s57 values bit-for-bit (max rel dev = 0.0)** — the re-run IS the paper's
cells, with absolute errors recorded. (One infrastructure repair: the
`legacy_nonstationary/tslib/dataset` symlink to the Time-Series-Library CSVs
had gone missing and was recreated; the bit-exact gate proves the data are the
same bytes the original evals used.)

Row granularity: a "row" is one window x channel series forecast (the
granularity of the stored relMSE arrays; 150 windows x C channels per cell,
66,668 rows pooled per fill). A per-window variant (channels averaged within
each window first) is in s73_absolute.json under `per_window_fill`.

## A. Bolt DS9 — absolute scale, Q50 vs stock

### A1. Overall per fill (rows pooled over 9 ds x 4 mechs, n = 66,668)

Win rate = share of rows with strictly lower absolute MSE for the retrofit;
exact ties (550 / 550 / 24 rows for zero / linear / oracle) are excluded from
the denominator and reported as `n_tie` in the JSON.

| fill | win rate | median ratio (CPT/vanilla) | ds-median win rate | min ds win rate | Wilcoxon p |
|---|---|---|---|---|---|
| zero | **0.740** | 0.763 | 0.734 | 0.579 (ETTm2) | ≈0 |
| linear | **0.621** | 0.942 | 0.599 | 0.539 (ETTm2) | ≈0 |
| oracle | **0.611** | 0.862 | 0.592 | 0.505 (weather) | ≈0 |

### A2. Per dataset x fill (rows pooled over 4 mechs)

win = per-row win rate; dMed/dMean = median/mean of (CPT − vanilla) absolute MSE;
ratio = median of per-row ratios; p = Wilcoxon signed-rank on log-ratios.

| dataset | fill | win | dMed | dMean | ratio | p |
|---|---|---|---|---|---|---|
| ETTh1 | zero | 0.734 | −0.290 | −3.30 | 0.806 | 2e-216 |
| ETTh1 | linear | 0.625 | −0.041 | −2.34 | 0.941 | 8e-44 |
| ETTh1 | oracle | 0.568 | −0.025 | −0.85 | 0.921 | 3e-36 |
| ETTm1 | zero | 0.659 | −0.108 | −0.58 | 0.866 | 8e-95 |
| ETTm1 | linear | 0.550 | −0.0076 | −0.31 | 0.977 | 2e-04 |
| ETTm1 | oracle | 0.618 | −0.055 | −1.51 | 0.882 | 1e-66 |
| weather | zero | 0.724 | −0.974 | −1831 | 0.707 | ≈0 |
| weather | linear | 0.544 | −6e-08 | **+265** | 0.987 | 1e-34 |
| weather | oracle | 0.505 | −1e-14 | **+3106** | 0.997 | 8e-10 |
| ETTh2 | zero | 0.600 | −1.089 | −8.14 | 0.856 | 1e-05 |
| ETTh2 | linear | 0.599 | −0.086 | −2.37 | 0.958 | 9e-47 |
| ETTh2 | oracle | 0.592 | −0.164 | −3.69 | 0.907 | 2e-55 |
| ETTm2 | zero | 0.579 | −0.766 | −8.13 | 0.844 | 2e-02 |
| ETTm2 | linear | 0.539 | −0.013 | −0.62 | 0.988 | 1e-05 |
| ETTm2 | oracle | 0.571 | −0.101 | −2.73 | 0.925 | 2e-41 |
| electricity | zero | 0.778 | −3383 | −6.1e5 | 0.626 | ≈0 |
| electricity | linear | 0.632 | −157 | −3.9e5 | 0.933 | 2e-85 |
| electricity | oracle | 0.660 | −497 | −1.0e6 | 0.736 | ≈0 |
| traffic | zero | 0.760 | −1.3e-4 | −2.9e-4 | 0.839 | ≈0 |
| traffic | linear | 0.687 | −8.8e-5 | −1.3e-4 | 0.918 | 1e-153 |
| traffic | oracle | 0.624 | −5.5e-5 | −4.0e-4 | 0.830 | ≈0 |
| exchange | zero | 0.821 | −0.0056 | −0.010 | 0.267 | ≈0 |
| exchange | linear | 0.599 | −4.7e-7 | −1.2e-4 | 0.914 | 1e-42 |
| exchange | oracle | 0.584 | −4.0e-7 | −3.5e-4 | 0.886 | 2e-41 |
| illness | zero | 0.894 | −1.0e6 | −1.0e10 | 0.908 | ≈0 |
| illness | linear | 0.788 | −1.2e5 | −2.5e9 | 0.931 | 2e-187 |
| illness | oracle | 0.877 | −1.2e6 | −1.2e10 | 0.716 | ≈0 |

Per-window granularity (150 windows, channels averaged): 26 of 27 cells still
above 50%; the single exception is **weather/oracle at 0.375**, the same tail
effect seen through window means.

### A3. Clean cost per dataset (same windows, clean input)

| dataset | median ratio CPT/vanilla | clean win rate | note |
|---|---|---|---|
| ETTh1 | 0.943 | 0.450 | clean BETTER on median |
| ETTm1 | 1.071 | 0.435 | small cost |
| weather | 1.131 | 0.421 | largest cost |
| ETTh2 | 0.989 | 0.502 | ≈ tie |
| ETTm2 | 1.042 | 0.421 | small cost |
| electricity | 1.085 | 0.294 | cost, broad |
| traffic | 1.032 | 0.409 | small cost |
| exchange | 1.018 | 0.499 | ≈ tie |
| illness | 1.057 | 0.276 | cost, broad |

Median over datasets: 1.042 (+4.2% median-of-medians on the exact Table-5
windows). The paper's clean column quotes +5.8%: that is the mean-based
aggregation of the same stored numbers — we reproduce 1.0577 as the mean over
datasets of the per-dataset mean-MSE ratios (median of mean-ratios: 1.053).

### A4. The 2x2 contingency (dataset level, median convention)

For every fill, the classification is IDENTICAL:
ETTh1, ETTh2 = (clean better, missing absolutely better); the other seven
datasets = (clean worse, missing absolutely better).
**(clean worse, missing absolutely worse): zero datasets on zero fills.**
No fill has any dataset in the reviewer's worry cell.

### A5. The weather tail (the one place the mean flips)

weather under oracle fill, CPT − vanilla delta over 12,600 rows: 49.4% of rows
are worse, median exactly ≈ 0, but the mean is +3106 because the top 10 rows
alone contribute 64.6% of the total positive delta (top 1% contribute 102%;
max single-row delta 8.8e6). Linear fill is the same shape (top-10 = 61.5%).
Under zero fill the tail disappears and CPT wins on every statistic
(win 72.4%, mean −1831). Per mechanism, the oracle mean loss is largest under
mnar_extreme (+6.2e3) and mnar_high (+4.7e3) — a few high-value spikes that the
retrofit's declared path handles worse than stock. This is a genuine, if sparse,
absolute-scale loss that the relMSE medians hide; it is now on the record.

## B. GIFT-Eval real missingness — per-stratum absolute win rates (c2/m2)

From stored per-window MASE (s64_gift_scores.npz; w50 = the paper's +CPT row vs
stock). Pool recomputed CPU-only via the deterministic s32c protocol;
**validation: pool n = 7,725 as expected, all 20 stored bin counts and bin
medians reproduced exactly (max abs dev 0.0)**, so row alignment is proven.

Win rate = share of windows in the stratum with MASE(w50) < MASE(stock).
Strata = the window's own context-NaN fraction; stratum populations
4,308 / 2,327 / 625 / 208 / 257, of which 4,041 / 2,316 / 625 / 207 / 248 are
scored (windows whose in-window naive-1 denominator is ~0 have undefined MASE
on every fill and drop out identically for both arms). Stored linear-fill
median MASE in parentheses (stock → w50).

### chronos-2 (c2-w50 vs c2-stock)

| stratum (ctx NaN) | nan | linear | zero | ffill | linear medians (stock → w50) |
|---|---|---|---|---|---|
| 0–1% | 0.441 | 0.431 | 0.434 | 0.432 | 0.629 → 0.627 |
| 1–5% | 0.434 | 0.529 | 0.561 | 0.534 | 0.860 → 0.860 |
| 5–15% | 0.288 | 0.541 | 0.566 | 0.530 | 1.445 → 1.373 |
| 15–30% | 0.135 | 0.643 | 0.705 | 0.599 | 2.020 → 1.694 |
| >30% | 0.198 | 0.669 | 0.706 | 0.601 | 1.822 → 1.495 |
| pooled | 0.409 | 0.485 | — | — | ratio 1.0007, p = 7e-07 |

### Moirai 2.0 (m2-w50 vs m2-stock)

| stratum (ctx NaN) | nan | linear | zero | ffill | linear medians (stock → w50) |
|---|---|---|---|---|---|
| 0–1% | 0.389 | 0.390 | 0.390 | 0.388 | 0.633 → 0.641 |
| 1–5% | 0.415 | 0.524 | 0.534 | 0.526 | 0.892 → 0.873 |
| 5–15% | 0.344 | 0.528 | 0.539 | 0.520 | 1.829 → 1.745 |
| 15–30% | 0.512 | 0.609 | 0.609 | 0.594 | 2.406 → 2.068 |
| >30% | 0.544 | 0.581 | 0.633 | 0.560 | 1.938 → 1.795 |
| pooled | 0.402 | 0.456 | — | — | ratio 1.0032, p = 7e-15 |

Read: on fixed fills the retrofit's win rate rises monotonically with the
missingness dose — a coin flip slightly below 50% in the near-clean stratum
(the dose-response's own negative-control range, which dominates the pool),
53–55% from 1–15% missing, and 56–71% above 15%. Pooled over all windows the
linear-fill win rate is slightly BELOW 50% for both families (c2: 0.485,
m2: 0.456) precisely because 52% of the pool has <1% missing; the pooled
median ratio is within 0.3% of 1.0. On the `nan` path (no imputation: stock's
native NaN handling vs the retrofit's declaration endpoint) c2-w50 loses more
often than it wins in every stratum — worst exactly where missingness is heavy
(0.135–0.198 above 15%); m2-w50 is dose-dependent on nan too (0.39 near-clean,
0.51–0.54 above 15%). GIFT admits no clean counterfactual (the missingness is
real), so the near-clean stratum is the clean-cost proxy: the small loss there
mirrors the DS9 clean deltas (c2: median 0.993 but illness 1.48; m2: median
1.011, 6 of 9 datasets slightly worse — stored values, Part C in the JSON).

## C. c2/m2 on the DS9 grid — what stored data can and cannot say

The s60/s61/s65/s64 DS9 grids store per-window relMSE ratios but only
median/mean clean MSE, so per-window absolute errors are not reconstructable
and no per-window win rate can be computed without a re-eval (out of scope per
the S73 plan; bolt carries the paper's flagship comparison). Stored clean
ratios of medians (w50 vs stock) are in s73_absolute.json
(`c2_m2_ds9_stored`); the GIFT table above is the absolute-scale evidence for
these two families.

## D. Rebuttal-ready answer (c)

> On the raw absolute scale the retrofit's advantage is not an artefact of the
> relative normalisation. Re-running the Table-5 bolt cells with per-window
> absolute MSE recorded (reproducing all 216 stored relMSE cells bit-for-bit),
> the retrofit beats the vanilla checkpoint on 74.0% / 62.1% / 61.1% of all
> 66,668 window×channel forecasts under zero / linear / oracle fills, with the
> per-dataset win rate above 50% on every one of the 27 dataset×fill cells and
> every per-dataset median absolute delta negative — so although 7 of 9
> datasets pay the reported small clean cost (median +4.2%), no dataset
> combines a clean degradation with an absolute loss under missingness. The
> single place an absolute loss exists is a sparse tail: on weather under
> oracle/linear fills ~10 of 12,600 rows (high-value spikes under the MNAR
> mechanisms) flip the MEAN delta against the retrofit while the median row is
> an exact tie; we will add this tail explicitly to the paper. On real
> GIFT-Eval missingness (absolute MASE, no clean counterfactual), both the
> Chronos-2 and Moirai retrofits win 53–71% of windows once context
> missingness exceeds ~1%, sit at a slight coin-flip loss in the near-clean
> stratum that dominates the pool (the dose-response's negative-control
> range), and under the raw NaN path the Chronos-2 retrofit still trails its
> stock NaN handling — all consistent with the paper's claim that the retrofit
> buys content-aware robustness at a small, now explicitly quantified,
> near-clean cost.

## File inventory

- `eval_s73.py` — re-eval script (P0 vs Q50, per-row absolute MSE; gate vs s57)
- `s73_eval.log`, `s73_eval_gate.json` — run log; per-cell absolutes + gate
  (216/216 relMSE cells, 18/18 clean cells, max rel dev 0.0)
- `s73_perwindow.npz` — per-row absolute MSE arrays (`clean|{arm}|{ds}`,
  `miss|{arm}|{ds}|{mech}|{fill}`, plus `rows|{ds}` window indices)
- `analyze_s73.py` — this analysis (bolt DS9 stats + 2x2 + tail focus; GIFT
  per-stratum win rates from stored npz with full pool-validation)
- `s73_absolute.json` — all numbers (per dataset×fill, per dataset×mech×fill,
  per-window variant, tail focus, GIFT bins, stored c2/m2 clean deltas)

Reproduce: `CUDA_VISIBLE_DEVICES=<free> python eval_s73.py --device cuda`
(~2 min on one A800), then `python analyze_s73.py` (~1 min, CPU).
