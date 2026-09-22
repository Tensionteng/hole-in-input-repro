# S23 — learned imputation baselines (SAITS/BRITS) vs free fills on the mechanism grid

Question: does a **learned deep imputer** (SAITS / BRITS, per-dataset trained) beat
linear interpolation as a pre-processing step for a frozen TSFM (chronos-bolt-base),
once the missingness is stratified by mechanism? Literature (s20 B1): BRITS/SAITS/CSDI
were only ever evaluated under artificial MCAR-ish corruption, with imputation error
as the metric — never mechanism-stratified, never downstream. S23 closes the baseline
ring of the repair ladder: free fills (S5) → rescale/adapter (S5-fix) → **learned
imputation (S23)**.

## Design (identical to S5 unless stated)

- Grid: mechanisms {mcar, block, mnar_high, mnar_extreme} × p ∈ {0.1, 0.3, 0.5, 0.7}
  × {ETTh1, ETTm1, weather}; 300 windows × 2 mask seeds (mnar: deterministic, 1 seed);
  same windows/seeds/masks as S5 (imported from `run_s5_missing.py`).
- Metric: relMSE = config MSE / S5 bolt clean MSE of the same dataset (anchor gate
  verified our harness reproduces S5 cells exactly, deviation 0.00%).
- Downstream: chronos-bolt-base fp32, median quantile, context L=512, horizon H=96.
- Imputers (PyPOTS 1.3): SAITS (n_layers=2, d_model=256, n_heads=4, d_k=d_v=64,
  d_ffn=256, dropout=0.1, diagonal_attention_mask default); BRITS (rnn_hidden_size=256).
  batch 64, ≤50 epochs, patience 8, val = 512 augmented windows from the 60–80%
  timeline slice. Multivariate per dataset (n_features = C), per-channel
  standardization with train-split (first 60%) mean/std; eval windows standardized →
  imputed → de-standardized → bolt.
- Training augmentation: random windows from the first 60% of the timeline, one random
  mechanism + rate ~ U(0.05, 0.7) per sample. 6000 (ETTh1) / 8000 (ETTm1, weather)
  train windows.
  - variant `mb`: mechanisms ~ U{mcar, block} — the mainstream convention;
  - variant `all`: mechanisms ~ U{mcar, block, mnar_high, mnar_extreme} —
    "mechanism-matched / mnar-augmented" training.

## Pre-registered look points

(i)  Can learned imputation beat linear under mcar/block? If not, the S5 conclusion
     "free repair is enough" strengthens.
(ii) Does learned imputation also fail under mnar? Expectation: yes — value-censored
     context removes the information; this is the information-floor 8th test.
(iii) Does mnar-augmented training still lose under mnar? If `all` ≫ `mb` under mnar,
      failure was mechanism-specificity; if `all` ≈ `mb` ≈ linear, it is the floor.

## Anchor gate

Reproduce S5 bolt on ETTh1 {clean, mcar:linear/zero:0.3, mnar_high:linear/zero:0.7}
within ±5% relMSE through the S23 harness. **Result: PASS — all five cells deviate
0.00%** (bit-identical pipeline via import; see s23_ckpt/anchor.json).

## Results

Main table (dataset-avg relMSE vs clean; zero/linear from s5_missing_results.json,
nan = bolt native NaN from s5_fix_results.json):

| mech | p | zero | linear | nan | saits_mb | saits_all | brits_mb | brits_all |
|---|---|---|---|---|---|---|---|---|
| mcar | 0.1 | 0.957 | 1.001 | 1.010 | 1.010 | 1.019 | 1.019 | 1.021 |
| mcar | 0.3 | 1.320 | 0.993 | 1.006 | 1.023 | 1.040 | 1.026 | 1.031 |
| mcar | 0.5 | 5.871 | 0.997 | 1.042 | 1.048 | 1.057 | 1.057 | 1.066 |
| mcar | 0.7 | 13.744 | 1.072 | 1.131 | 1.064 | 1.065 | 1.113 | 1.121 |
| block | 0.1 | 1.358 | 0.994 | 1.007 | 1.000 | 1.003 | 0.999 | 1.000 |
| block | 0.3 | 1.907 | 1.048 | 1.012 | 1.000 | 0.997 | 0.986 | 0.987 |
| block | 0.5 | 4.199 | 1.180 | 1.046 | 1.011 | 0.996 | 0.993 | 1.003 |
| block | 0.7 | 5.927 | 1.360 | 1.085 | 1.019 | 1.000 | 0.985 | 1.002 |
| mnar_high | 0.1 | 3.727 | 0.997 | 1.003 | 1.041 | 1.011 | 1.032 | 1.036 |
| mnar_high | 0.3 | 6.312 | 1.119 | 1.188 | 1.126 | 1.038 | 1.160 | 1.139 |
| mnar_high | 0.5 | 8.911 | 1.364 | 1.688 | 1.333 | 1.290 | 1.533 | 1.376 |
| mnar_high | 0.7 | 11.318 | 2.174 | 3.393 | 1.459 | 1.364 | 1.722 | 1.567 |
| mnar_extreme | 0.1 | 3.920 | 1.010 | 1.055 | 1.105 | 1.066 | 1.042 | 1.048 |
| mnar_extreme | 0.3 | 6.183 | 1.193 | 1.328 | 1.385 | 1.391 | 1.286 | 1.231 |
| mnar_extreme | 0.5 | 9.361 | 1.313 | 1.423 | 1.539 | 1.435 | 1.325 | 1.356 |
| mnar_extreme | 0.7 | 12.125 | 1.580 | 1.698 | 1.793 | 1.696 | 1.565 | 1.652 |

Per-dataset highlights (relMSE, linear → best learned):
- block p=0.7: ETTh1 1.513 → 0.967 (saits_all); ETTm1 1.399 → 0.983 (brits_mb);
  weather 1.168 → 0.955 (saits_all). Learned imputation wins on all three datasets.
- mcar p=0.7: ETTh1 1.263 → 1.014 (saits_all); ETTm1 1.073 → 1.014 (saits_mb);
  weather 0.881 → 1.142 (learned LOSES; linear<1 on weather is the S5 denoising
  artifact — mild corruption regularises this dataset).
- mnar_high p=0.7: ETTh1 2.519 → 1.281 (saits_all); ETTm1 2.674 → 1.442 (saits_all);
  weather 1.329 → 1.133 (saits_mb). All four learned variants beat linear on ETT at
  p≥0.5; on weather saits_mb/brits_all win but saits_all/brits_mb lose (see anomalies).
- mnar_extreme p=0.7: ETTh1 1.929 → 1.634 (brits_mb); ETTm1 1.797 → 1.893 (best
  learned still loses); weather 1.015 → 1.104 (loses). Dataset-split outcome.

Paired per-window ΔrelMSE vs linear (900 windows pooled over datasets, mean±se, win%):
- block p=0.7: saits_mb −0.341±0.033 (68%), saits_all −0.360±0.033 (68%),
  brits_mb −0.376±0.033 (71%), brits_all −0.358±0.032 (67%) — decisive.
- mnar_high p=0.7: saits_mb −0.715±0.034 (86%), saits_all −0.810±0.041 (80%),
  brits_mb −0.452±0.058 (72%), brits_all −0.607±0.042 (74%) — decisive.
- mnar_extreme p=0.7: saits_mb +0.213±0.015 (30%), saits_all +0.116±0.016 (47%),
  brits_mb −0.016±0.024 (51%), brits_all +0.072±0.022 (44%) — no consistent win.
- mcar: |Δ| ≤ 0.06 everywhere except saits p=0.7 ≈ 0; win% 39–51% — a wash on average.

Extreme-value split (mse_topdecile / mse_rest, fixed clean-context q90 threshold),
mnar_high p=0.7 — the mechanism of the win:
- ETTh1: linear td 56.20 / rest 23.73 (ratio 2.37) → saits_all td 20.47 / rest 13.07
  (1.57); saits_mb td 31.85.
- ETTm1: linear td 48.97 / rest 22.79 (2.15) → saits_all td 17.38 / rest 13.60 (1.28).
- weather: linear td 29780 / rest 1343 (22.2) → saits_mb td 25658 / rest 1103;
  saits_all td 22315 but rest 2608 (worse rest — its win there is mixed).
Under one-sided censoring, linear's downward bias hits exactly the extreme futures;
the learned shape prior halves that top-decile error on ETT.
mnar_extreme p=0.7 shows the flip side: saits_all improves td (ETTm1 13.50 → 11.75)
but worsens rest (18.52 → 24.49) — reconstructing censored extremes injects values
that corrupt the bulk distribution on ETTm1/weather; net loss vs linear.

## Verdicts on look points

(i) **mcar: no (linear is enough); block: yes, decisively.** Dataset-avg under mcar the
    learned imputers sit within ±6% of linear (losing on weather, winning on ETT at
    high p) — S5's "free repair suffices under MCAR" stands. Under block all four
    learned variants beat linear at p≥0.3 on every dataset (avg 1.00 vs 1.36 at p=0.7,
    paired win 67–71%), and also beat S5-fix's best block repair (bolt native NaN,
    1.085 avg). Deep imputation is the first repair that clearly improves on linear
    for contiguous gaps.
(ii) **mnar_high: the "floor" breaks — learned priors win.** All four learned variants
    beat linear at p≥0.5 on both ETT datasets (best avg 1.364 vs 2.174 at p=0.7), by
    halving the top-decile future error. The censored values themselves remain
    unpredictable (imp_mae under mnar is 3–8× that under mcar), but a learned prior
    over segment shapes restores downstream-usable structure that interpolation
    provably cannot. **mnar_extreme: the floor holds on average** — no learned variant
    beats linear consistently (avg best ≈ linear at 0.7; ETTm1/weather lose, ETTh1
    wins). So the information-floor conclusion survives for two-sided censoring but
    was overstated for one-sided censoring: "free repairs cannot beat linear under
    MNAR" (S5) is true; "no repair can" (stronger reading) is false for mnar_high.
(iii) **Mechanism-matched training is real but second-order.** `all` vs `mb` under
    mnar on ETT: mnar_high p=0.7 ETTh1 1.281 vs 1.733 (saits), mnar_extreme p=0.7
    ETTh1 1.700 vs 1.945; gains shrink at lower p. Under mcar/block `all` ≈ `mb`
    (no cost). On weather the sign flips for saits (1.441/1.368 vs 1.159/1.133 at
    mnar_high 0.5/0.7) but not for brits (1.277/1.282 vs 1.835/1.928). Net: training
    mechanism distribution matters most exactly where the eval mechanism is hardest,
    but the dominant factor is the learned prior itself, not mechanism matching.

## Anomalies / notes

- brits_mb on weather mnar_high p=0.5/0.7: relMSE 1.835/1.928 — worse than linear
  (1.290/1.329) and far worse than its own `all` variant (1.277/1.282). A
  mcar/block-only-trained BRITS catastrophically misfills one-sided-censored contexts
  on this heterogeneous 21-channel data; mechanism-matched training fixes it.
- weather mcar linear relMSE < 1 (0.881 at p=0.7) and zero-fill avg 0.957 at p=0.1:
  the known S5 denoising artifact (mild corruption regularises weather forecasts);
  learned imputers sit at 1.03–1.17 there and "lose" to it.
- ETTh1 block learned relMSE < 1.00 (down to 0.967): imputation-as-denoising on the
  smoothest dataset.
- Imputation quality and downstream quality decouple: e.g. saits imp_mae under
  mnar_high is ~0.7 (vs ~0.15 under mcar) yet delivers the largest downstream wins
  there; brits_mb weather has the lowest imp_mae (0.037) but loses to linear under
  mcar. Imputation-error leaderboards would not have predicted any of this.
- Ops: PyPOTS 1.3 (pinned to keep transformers 5.14.1 / huggingface-hub untouched;
  pypots 1.5 would force a downgrade). torch 2.6 rejects pypots' `model.load`
  (weights_only default change) — worked around by loading `model_state_dict`
  directly (own checkpoints). torch.load glob also matches tensorboard `*.pypots`
  event files — filtered. BRITS first two ETTh1 runs used batch 64 (~4.5 min/epoch);
  restarted at batch 128 (~2 min/epoch), all 6 BRITS runs use bs128 for uniformity.
  SAITS training 5–9 min/run; BRITS 83–133 min/run (RNN on 512-step windows).

Artifacts: `run_s23_impute.py`, `s23_results.json`, `s23.png`, `s23_ckpt/`,
logs `s23_train_*.log`, `s23_eval_*.log`.
