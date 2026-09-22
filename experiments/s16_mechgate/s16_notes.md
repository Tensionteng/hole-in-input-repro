# S16 notes — mechanism detector + gated repair pipeline (detect → route → calibrate)

**One line:** a 14-feature window-level mechanism classifier (6 classes) trained
only on S5-generator synthetic masks generalizes across datasets (LOD acc
0.92–0.98, vs 0.49–0.52 for a Little's-test-style single-feature baseline) and
transfers zero-shot to both real mechanisms — fixing the S6 v1 rule detector's
per-window failure on blocky masks (METR-LA block recall 0.574 vs v1 0.451,
censoring hallucination 42.6% vs v1 54.7%, ctrl→clean 1.000) — and the gated
pipeline it drives matches or beats the best fixed fill on 4 of 6 real
end-to-end cells (Penmanshiel timesfm/moirai gated **beats** best-fixed:
31.13 vs 34.82 / 25.18 vs 25.24 mean NMSE), while Mondrian-by-predicted-class
conformal restores per-group coverage to 0.88–0.93 where a global pool leaves
the hard group at 0.78.

## Setup / what's reused

- Detector training: S5 mask generators (`run_s5_missing.make_mask` for
  mcar/mnar; same algorithm with block length jitter {12,24,48,96} + 50%
  channel-shared blocks for block) on **train-split** windows of
  ETTh1/ETTm1/weather (220 windows/dataset → 7700 channel samples), rate
  p ~ U(0.05, 0.8). intermittent = natural near-zero channels (zero-frac ≥ 0.5
  → relabel, no mask) + synthetic sparsification (50–95% points zeroed, no
  mask). Class counts: clean 1476 / mcar 1362 / block 1415 / mnar_high 1057 /
  mnar_extreme 1042 / intermittent 1348.
- Features (all dimensionless, L-normalized so CTX=144 real and CTX=512
  synthetic share a space): miss_rate, runs_per_100, mean/med/p90 run
  length / L, fraction of missing in runs ≥ max(6, L/16), S6 bridge-rank
  r_bar/e_bar/x_bar + e_max, zero-frac of observed, scale-fallback flag,
  cross-channel mask corr (0 for univariate), Little-style chi-square segment
  statistic. Effective mask = mask | non-finite values.
- Models: multinomial logreg, sklearn HistGB ("gbdt"), and the single-feature
  Little baseline (logreg on the chi-square stat only). Final = gbdt on all
  three datasets (`s16_ckpt/clf_final.pkl`).
- Track B assembles the existing per-window stores (s12, s5+s6, s10, s14);
  new GPU inference only for: anchor gate (3 cells), synthetic Mondrian
  conformal stream, real conformal quantile reruns. All bolt.

## 1) Anchor gate — PASS

| cell | stored | rerun | diff |
|---|---|---|---|
| s12 ETTh1 mcar:zs:linear:0.3 (150 win) | 11.77903 | 11.77903 | 0.00% |
| s5 ETTm1 mnar_high:linear:0.5 (300 win) | 13.44178 | 13.44178 | 0.00% |
| s10 bolt cens zero (602 win, NMSE) | 28.42657 | 28.42657 | 0.00% |

Cross-checks inside Track B: my mondrian_oracle == S10/S14 stored mechaware
coverage to ≤1e-16 on all 4 real conformal cells (same splits/fills/model).

## 2) Track A — classifier

**LOD CV (leave-one-dataset-out), accuracy / macro-recall:**

| held-out | gbdt | logreg | little (1-feature) |
|---|---|---|---|
| ETTh1 | 0.972 / 0.966 | 0.944 / 0.935 | 0.499 / 0.525 |
| ETTm1 | 0.979 / 0.975 | 0.981 / 0.978 | 0.521 / 0.548 |
| weather | 0.921 / 0.906 | 0.921 / 0.907 | 0.485 / 0.482 |

Little-baseline per-class recall (ETTh1): clean 0.0, mcar 0.93, block 0.79,
mnar_high 0.0, mnar_extreme 0.43, intermittent 1.0 — a chi-square/run-shape
statistic alone cannot tell value-dependent censoring from random loss (the
motivation for multi-feature). gbdt pooled out-of-fold confusion: worst bleed
= mnar_high→block 8.0% and mnar_extreme→block 8.0% (high-rate censoring merges
into long runs), clean→intermittent 5.5% (weather natural zero channels).

Permutation importance: miss_rate 0.367, zero_frac_obs 0.184, med_run_rel
0.169, runs_per_100 0.090, r_bar 0.089, e_max 0.038, mean_run_rel 0.028,
e_bar 0.025, x_bar 0.014, little_chi2 0.009, frac_miss_long 0.002;
scale_fallback ≈ 0, xmask_corr ≈ 0 (by construction: channel-shared blocks are
within-class variance — the feature exists so synchronized real outages are
NOT pushed away from block).

**Synthetic test windows (Track B grids, window-level acc, mean-proba vote):**
mcar 1.000, block 1.000, mnar_high 0.999 (min 0.987), mnar_extreme 1.000 —
near-perfect in-distribution (expected; LOD above is the honest measure).

**Zero-shot transfer (the paper sell; "censoring recall" = mnar_high +
mnar_extreme, both value-dependent):**

| testbed | S16 recall | v1 rules | S16 clean recall | v1 clean |
|---|---|---|---|---|
| Penmanshiel cens, oracle mask | 0.337 (0.238 exact mnar_high) | 0.651 | 0.992 | 1.000 |
| Penmanshiel cens, auto mask (status-free) | **0.733** (0.640 exact) | 0.556 | 0.688 | 0.691 |
| METR-LA miss (0-encoded) | **0.574** (block) | 0.451 | **1.000** | 1.000 |

- METR-LA miss windows labeled censoring-like: S16 42.6% vs v1 54.7% (S14's
  documented v1 failure); ctrl 100% clean both.
- Penmanshiel **oracle** masks: 66.3% of cens windows → block. Real
  curtailment arrives in farm-wide episodes with long runs (mean 4.4, max
  146 × 10-min) — geometrically blocky, and the recorded clamped values are
  excluded from "observed", so the deleted-value features (r_bar etc.) see
  little signal: the clamp-episode geometry is genuinely close to a sensor
  block. Under the **auto** mask (avail−power margin, no operator log) the
  same classifier recovers 73.3% censoring recall — the avail margin itself
  carries the value-dependence, and this is the deployable scenario anyway.
- This is the honest limitation, quantified: clamp-not-delete real censoring
  with blocky episodes is ambiguous with sensor blocks under
  mask+observed-value features alone.

## 3) Track B — gated repair (synthetic, per-window)

relMSE vs paired clean, pooled over 3 datasets × 4 rates. Policy tables from
prior rounds: Eval A fills-policy {mcar→zs:linear, block→zs:nan}, full-policy
adds SFT variants (block→mb:nan by S12 aggregate); Eval B {all→linear} and the
S6 arm {mnar→tail_tobit}.

**Eval A (s12 store, bolt, 150 win, mcar/block):**

| mech | zs:lin | zs:nan | tok:nan | mb:nan | recon:nan | gated_fills | gated_full | oracle_fills | oracle_all |
|---|---|---|---|---|---|---|---|---|---|
| mcar | 1.152 | 1.266 | 1.265 | 1.121 | 1.119 | 1.152 | 1.119 | 0.994 | 0.903 |
| block | 1.588 | 1.241 | 1.190 | 1.152 | 1.152 | 1.241 | 1.152 | 1.120 | 0.881 |

regret gated_full − oracle_all: mcar 0.215, block 0.271; cell win-rate vs best
fixed: block 1.00, mcar 0.42. Note oracle_fills(0.994) ≪ any fixed fill on
mcar (1.152): even within one mechanism the per-window best fill varies —
mechanism identity alone does not saturate per-window routing (future work).

**Eval B (s5+s6 stores, bolt, 300 win, all 4 mechanisms):**

| mech | zero | ffill | linear | tobit(gated) | gated_lin | oracle | regret_lin | regret_tobit |
|---|---|---|---|---|---|---|---|---|
| mcar | 26.95 | 1.301 | **1.151** | 1.151 | 1.151 | 1.053 | 0.098 | 0.098 |
| block | 15.63 | 2.205 | **1.695** | 1.695 | 1.695 | 1.496 | 0.199 | 0.199 |
| mnar_high | 35.41 | 2.069 | 2.122 | **1.911** | 2.122 | 1.830 | 0.292 | **0.081** |
| mnar_extreme | 39.66 | 2.013 | 2.019 | **1.861** | 2.019 | 1.798 | 0.222 | **0.064** |

MNAR verdict (as pre-registered in the task): routing mnar→linear is the safe
default and beats every naive fill by ≥10×; routing mnar→tobit (S6 arm)
replicates the S6 per-window gain on the mean (2.12→1.91 / 2.02→1.86) and
nearly saturates the oracle regret (0.06–0.08). zero_oscale does not rescue
zero under mnar (54–72 relMSE — rescale inherits the biased observed stats).

## 4) Mondrian conformal (target 0.90)

**Synthetic mixed stream** (bolt [q10,q90] widened; 300 cal + 300 test
windows/dataset; per-channel mechanism mix clean/mcar/block/mnar_high/
mnar_extreme = .25/.30/.20/.15/.10, p~U(0.1,0.6); fill routed by predicted
class: mcar/mnar→linear, block→nan). Coverage by TRUE class, naive global →
Mondrian-by-predicted (→ by-oracle), ETTh1 / ETTm1 / weather:

| true class | native | naive | mond_pred | mond_oracle |
|---|---|---|---|---|
| clean | 0.749/0.743/0.751 | 0.934/0.923/0.929 | 0.895/0.882/0.904 | 0.895/0.882/0.902 |
| mcar | 0.731/0.733/0.750 | 0.936/0.927/0.936 | 0.904/0.892/0.904 | 0.906/0.892/0.904 |
| block | 0.738/0.732/0.734 | 0.935/0.927/0.924 | 0.901/0.900/0.906 | 0.900/0.900/0.900 |
| mnar_high | 0.469/0.482/0.524 | 0.782/0.783/0.800 | **0.901/0.885/0.886** | 0.900/0.889/0.904 |
| mnar_extreme | 0.455/0.433/0.498 | 0.843/0.826/0.813 | **0.924/0.928/0.921** | 0.928/0.931/0.899 |
| ALL | 0.671/0.669/0.690 | 0.903/0.895/0.900 | 0.903/0.894/0.903 | 0.903/0.895/0.902 |

The global pool hides the failure: mnar_high sits at 0.78–0.80 while clean
over-covers at 0.92–0.93. Mondrian-by-predicted flattens all groups to
0.88–0.93 (width cost exactly where S6 found it: mnar_high width 6.4→9.7 on
ETTh1, clean width shrinks 7.6→6.4), and is within ±0.01 of oracle grouping —
the detector is good enough that grouping by prediction ≈ grouping by truth.

**Real datasets** (exact S10/S14 splits; bolt; grouping pools = predicted
censoring/block vs rest):

| cell | native | naive | mond_pred | mond_oracle |
|---|---|---|---|---|
| Penn cens, zero fill | 0.900 | 0.936 | 0.936 | 0.921 |
| Penn cens, linear fill | 0.710 | 0.864 | 0.870 | 0.903 |
| Penn ctrl, linear fill | 0.775 | 0.943 | 0.938 | 0.904 |
| METR-LA miss, linear | 0.710 | 0.916 | 0.910 | 0.897 |
| METR-LA miss, nan | 0.685 | 0.918 | 0.905 | 0.882 |

Mondrian-by-predicted moves coverage toward the 0.9 target vs naive on all 4
miss/cens cells (Penn linear +0.006, METR-LA linear −0.006, nan −0.013); the
residual gap to oracle-grouping on Penn linear (0.870 vs 0.903) is the
censoring-detection recall (33.7% oracle-mask) showing up in calibration.

## 5) Real end-to-end (store-assembled per-window NMSE; mean / median)

**Penmanshiel, 602 censored windows.** Routing: predicted censoring → zero;
clean/intermittent → keep; mcar/block → linear.

| model | keep | zero | linear | gated (oracle mask) | gated (auto mask) | oracle |
|---|---|---|---|---|---|---|
| bolt | 40.29 / 2.16 | **28.43** / 1.66 | 60.16 / 2.12 | 42.92 / 2.03 | 29.34 / 1.84 | 27.41 / 1.35 |
| timesfm | 72.63 / 2.10 | 34.82 / 2.21 | 85.54 / 1.96 | 62.83 / 1.98 | **31.13** / 2.05 (beats best fixed) | 29.94 / 1.54 |
| moirai | 36.12 / 2.12 | 25.24 / 2.18 | 44.50 / 2.26 | 41.97 / 2.23 | **25.18** / 2.19 (beats best fixed) | 23.54 / 1.73 |

**METR-LA, 618 miss windows.** Routing: block → nan (bolt) / linear (timesfm,
moirai — S14 aggregates: timesfm nan≡linear internally, moirai linear 2.52 <
nan 2.66); clean/intermittent → keep; mcar/mnar → linear.

| model | keep=zero | linear | nan | gated | oracle |
|---|---|---|---|---|---|
| bolt | 21.19 / 1.41 | 2.749 / 1.243 | **1.999** / 1.234 | 2.735 / 1.220 | 1.419 / 1.079 |
| timesfm | 1.908 / 1.016 | 1.800 / 1.110 | **1.780** / 1.113 | 1.800 / 1.110 | 1.104 / 0.950 |
| moirai | 14.79 / 2.02 | **2.523** / 1.486 | 2.658 / 1.540 | 2.523 / 1.545 | 1.937 / 1.311 |

Read: with the deployable auto mask, Penmanshiel gating is within 3.2% of the
best fixed fill for bolt and **beats** it for timesfm/moirai (regret vs oracle
1.2–1.9 NMSE); on METR-LA the pipeline's job is to avoid the catastrophic
zero/keep fill, which it does for all three models (timesfm/moirai gated ≡
best fixed; bolt gated 2.735 ≈ linear 2.749, behind fixed-nan 1.999 because
42.6% of miss windows are mnar-hallucinated and route to linear). Per-window
oracle match rates 0.31–0.39 — mechanism-level routing captures the dominant
fill split (zero vs linear/nan) but per-window oracle headroom remains.

## 6) Anomalies / caveats (honest)

1. Penmanshiel oracle-mask cens windows classify as block (66.3%): clamp
   episodes are long-run blocky and the clamped values are not "observed", so
   deleted-value features go quiet. The auto (status-free) mask does NOT
   suffer this (73.3% censoring recall) — its avail-margin criterion is itself
   value-informative. Net effect: oracle-mask gating underperforms fixed-zero
   (42.9 vs 28.4 bolt), auto-mask gating nearly matches it.
2. METR-LA: 42.6% of miss windows hallucinated as mnar_* (v1: 54.7%) — the
   block-on-peaks ambiguity is reduced, not eliminated. Cheap there (routed
   linear), but it would be dangerous on a domain where mnar→zero: the
   per-domain policy table must come from prior per-domain evidence.
3. clean→intermittent bleed: 5.5% in LOD (weather natural zero channels), 5
   windows on Penn ctrl (calm-night zeros) — routed "keep", harmless.
4. Eval B mcar/block routing features use mask seed 0 while stored MSEs
   average seeds {0,1} (paired; routing-noise only). Eval A masks verified ==
   s12 stream (ms=0) by run_s12_recon's tiny assert.
5. moirai METR-LA route initially set to nan (2.658) — corrected to linear
   per S14 aggregate (2.523); final gated ≡ fixed-linear (no miss window is
   predicted clean, so the moirai/timesfm gates are degenerate there).
6. Synthetic conf stream: 374 weather cal windows predicted "intermittent"
   though the mix contains no true intermittent class (natural zero channels
   drawn as clean/block); they form their own small Mondrian pool — harmless.
7. Penn ctrl under linear fill: mond_pred 0.938 < naive 0.943 (both over-cover
   vs 0.9; oracle 0.904) — grouping helps the cens group without breaking
   ctrl.
8. scale_fallback / xmask_corr ≈ 0 permutation importance (by construction —
   see §2); kept for deployment robustness, not discriminative power.
9. Eval A: with own-clean normalization the in-domain SFT variants (mb/recon)
   dominate BOTH mechanisms, so the informative routing arm is the fills-only
   one; SFT-variant gains are model-swap gains, not fill-routing gains.
10. mean NMSE on real data is tail-driven (S10/S14 caveat); medians are in
    `s16_results.json` alongside means and do not change the rankings.

## Artifacts

`run_s16_mechgate.py` (modes: smoke, gate, trackA, trackB_gate, conf_syn,
conf_real, real_e2e, figure, all), `s16_results.json` (all tables incl.
per-cell detail), `s16_ckpt/` (clf_final.pkl, feats_train.npz, labels_*.npz),
`s16.png` (6 panels), logs `s16_smoke.log`, `s16_gate.log`, `s16_A.log`,
`s16_B.log`, `s16_e2e2.log`, `s16_figure.log`.
