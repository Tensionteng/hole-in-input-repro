# S62 notes — TempoPFN in the missingness-interface census

TempoPFN (`AutoML-org/TempoPFN`, checkpoint_38M, HF sha f88ff0ab; Moroshan, Siems, Zela,
Carstensen, Hutter 2025, arXiv:2510.25502) is the closest prior work to the paper's
prescription: it pretrains WITH mechanism-diverse missingness
(`configs/example.yaml`: `data_augmentation.nan_augmentation: true`;
`src/data/augmentations.py::NanAugmenter` samples NaN ratios and run-lengths from the
GIFT-Eval NaN statistics in `data/nan_stats.json` — scattered AND block patterns) — but
its interface discards the fill's content into a learned NaN embedding. This round turns
the paper's sentence "the two halves of the prescription remain uncombined" into numbers.

## How the interface actually works (code evidence)

- `src/models/model.py::_compute_embeddings`:
  `nan_mask = torch.isnan(scaled_history)`; `channel_embeddings[nan_mask] = self.nan_embedding`.
  At a NaN position the value's embedding is REPLACED by a single learned vector (same
  token at every position). Whatever value sat there is gone.
- `src/data/scalers.py::RobustScaler.compute_statistics`:
  `valid_data = valid_data[torch.isfinite(valid_data)]` — median/IQR from observed
  (finite) points only. The fill cannot leak through the scaling statistics either.
- The container's `history_mask` (`src/data/containers.py`) is a per-series PADDING mask
  that zeroes whole embeddings ("Suppress padded time steps completely so padding is a
  pure batching artifact") — NOT a per-point missingness channel.
- The official GIFT-Eval predictor (`src/gift_eval/predictor.py:199-237`) passes raw
  NaN-containing targets straight into `history_values`. NaN-in-input IS the model's
  declared missingness path, and it has no slot that both declares a point missing and
  carries a fill value. TempoPFN also does not impute internally (contrast TimesFM's
  "overwrites" — TimesFM replaces the fill with its own interpolation; TempoPFN
  substitutes a learned token and keeps observed-only statistics).

Consequence: the declared path's forecast is a deterministic function of (observed
values, mask pattern, time features) ONLY. Verified numerically: permuting or redrawing
the fill under the declared path moves the forecast by exactly 0.0 (perm=0, redraw=0 on
all 9 census cells; smoke test: fill-permutation effect 0.000e+00).

Two probe paths, as elsewhere: `plain` = linear fill fed as ordinary values, nothing
declared; `nan` = NaN at the missing positions (the declared path).

## Gate (all passed)

- Determinism: two identical calls, max|diff| = 0.0 on both paths (bf16 autocast — the
  ONLY supported mode: fla's `chunk_gated_delta_product` kernel asserts against fp32
  inputs; the official quick start runs the same bf16-autocast/fp32-weights setup).
- Clean-context MSE vs chronos-bolt-base on the same 60 windows/dataset (L=512, H=64):
  ETTh1 12.25 vs 10.36 (1.18x), ETTm1 8.60 vs 7.94 (1.08x), weather 2408 vs 3740
  (0.64x — TempoPFN better). Sane range.
- Redraw denominator on the plain path: 0.246 (ETTh1/mcar) — clearly nonzero.

## Census probe (rho), rate 0.3, linear fill, 60 windows/cell

| dataset | mech | rho plain | rho declared (nan) |
|---|---|---|---|
| ETTh1 | mcar | 0.970 | 0 (exact) |
| ETTh1 | block | 0.843 | 0 (exact) |
| ETTh1 | mnar_high | 0.370 | 0 (exact) |
| ETTm1 | mcar | 0.979 | 0 (exact) |
| ETTm1 | block | 0.901 | 0 (exact) |
| ETTm1 | mnar_high | 0.327 | 0 (exact) |
| weather | mcar | 0.087 (median-based 0.96, see caveat) | 0 (exact) |
| weather | block | 0.324 | 0 (exact) |
| weather | mnar_high | 1.956 | 0 (exact) |

plain median = 0.84 over all 9 cells, 0.87 over the 6 ETT cells — the same "uses the
fill's content" range as every other family (bolt 0.91–1.00, TiRex 0.90–1.12).
declared = exactly 0 with perm = redraw = 0 — the fifth "discards" convention in the
families table, and the cleanest possible instance: not even (loc, scale) leak, because
the scaler excludes non-finite values.

## Fill-quality sweep (relMSE vs own clean, mean ratio)

fill = (1-a)*linear + a*truth; 3 datasets x 4 mechs x rates {0.3, 0.7} x a {0, 0.5, 1},
60 windows/cell. Both predictions of the paper confirmed exactly:

- **Declared path flat**: relMSE identical at all three alphas in all 48 cells (input
  is provably alpha-invariant; computed once, copied, documented in the JSON). The
  augmentation cannot open the content path — nothing can, the path does not exist.
- **Plain path reaches 1.000 at a=1** in all 48 cells (input becomes the clean context;
  also an end-to-end determinism check).

Declared vs plain at a=0 (linear fill): declared wins 11 of 24 cells — every
value-independent cell at rate 0.7 and several at 0.3 (ETTh1 block 0.7: 0.998 vs 2.175;
ETTm1 mcar 0.3: 0.927 vs 1.047; ETTm1 block 0.7: 1.549 vs 2.185) — while plain wins the
censoring cells where the interpolant leaks the censored magnitude (ETTm1 mnar_high 0.7:
3.509 vs 4.694; weather mnar_extreme 0.7: 1.122 vs 1.583). Several declared cells sit
BELOW 1.0 (ETTm1 mcar 0.3: 0.927, mcar 0.7: 0.972; ETTh1 block 0.7: 0.998): declaring
30–70% of the context missing can BEAT feeding the complete clean context — the
augmentation half of the prescription is real and it works. What it cannot do is use a
better fill: at a=1 the plain path is at 1.000 everywhere while the declared path is
stuck at its flat value, better or worse.

## Calibration at rate 0.7 (coverage80 / pi_width80, H=64, 60 windows)

| dataset | cell | tempopfn plain | tempopfn declared | bolt plain (ref) |
|---|---|---|---|---|
| ETTh1 | clean | 0.796 / 5.20 | 0.796 / 5.20 | 0.792 / 4.71 |
| ETTh1 | mcar | 0.749 / 6.07 | 0.784 / 5.45 | 0.735 / 5.56 |
| ETTh1 | mnar_high | 0.236 / 3.48 | 0.258 / 4.76 | 0.238 / 3.25 |
| ETTm1 | clean | 0.819 / 4.64 | 0.819 / 4.64 | 0.793 / 3.84 |
| ETTm1 | mcar | 0.750 / 4.55 | 0.794 / 4.53 | 0.737 / 4.14 |
| ETTm1 | mnar_high | 0.222 / 2.76 | 0.305 / 5.92 | 0.206 / 2.28 |
| weather | clean | 0.780 / 29.8 | 0.780 / 29.8 | 0.782 / 43.8 |
| weather | mcar | 0.692 / 29.7 | 0.798 / 27.2 | 0.722 / 48.6 |
| weather | mnar_high | 0.307 / 4.96 | 0.381 / 8.70 | 0.332 / 4.46 |

Answers to the bonus question: censoring-augmented pretraining does NOT protect
TempoPFN's intervals from the censoring collapse — declared coverage under mnar_high is
0.26–0.38 vs bolt's 0.21–0.33 (paper value 0.217 at H=96/150 win; 0.238 here at
H=64/60 win). But the MECHANISM of failure differs by path: the plain path reproduces
bolt's signature exactly (coverage collapses AND the interval narrows, 5.20→3.48 on
ETTh1), while the declared path's interval does NOT narrow (5.20→4.76; on ETTm1 it even
widens 4.64→5.92) — it stays honestly wide yet still misses, because the censored
extremes are gone from the input. Under scattered dropout the declared path is nearly
nominal (0.784/0.794/0.798 vs clean 0.796/0.819/0.780) — the augmentation patterns
(scattered/block, value-independent) transfer to calibration too; censoring is not in
the augmentation and it shows.

## Caveats

- **bf16 only.** fla's `chunk_gated_delta_product` rejects fp32 activations. Weights are
  fp32, activations bf16 (official inference mode). Determinism verified exactly
  (max|diff| = 0.0 across identical calls, both paths, and a=1 sweep cells reproduce
  the clean reference to 1.000 in all 48 cells).
- **Weather rho is tail-dominated.** On weather/mcar/plain the redraw arm's MEAN is
  6.26x forecast scale, but the per-series distribution is median 0.064 / p90 1.04 /
  p99 7.3 / max 6992 — 9.2% of weather series have context IQR < 0.01 and hit the
  RobustScaler's min_scale=1e-3 floor, so a redrawn spiky observed value moves the
  scaled input enormously (perm arm bounded by the fill multiset: max 123). Median-based
  rho on that cell is ~0.96, in line with the other families; the stored mean-based
  ratios on weather (0.087 / 0.324 / 1.956) reflect the tail, not the typical series.
  Same protocol kept for cross-model comparability — flagging, not re-defining.
- **mnar_high plain rho on ETT (0.33–0.37)** matches the pattern in every other family
  (bolt 0.36/0.80, FlowState 0.29): under value censoring the fill values are near-
  constant interpolants at the censored extremes, so their permutation matters less.
- Time features: all series were fed with a fixed start (2017-01-01) and Frequency.H —
  no calendar information, same for every cell and both paths (census models receive no
  timestamps at all).
- Declared-path sweep cells are computed once per (ds, mech, rate) and copied across
  alphas (input provably alpha-invariant); each copied record carries a note saying so.

## Files

- `run_s62_smoke.py` — load/determinism/interface-invariance smoke (also catches the
  fp32 kernel assertion).
- `run_s62_tempopfn.py` — gate + probe + sweep + calib, writes `s62_results.json`
  (resumable, atomic saves).
- `run_s62_weather_redraw.py` — the weather redraw dissection (appends
  `followup.weather_redraw_dissection` to the results JSON).
- Model: `models_local/tempopfn` (full HF snapshot incl. source; sha f88ff0ab).
  Requires `flash-linear-attention` (fla-core 0.5.2 installed into the venv) and
  `PYTHONPATH=models_local/tempopfn`.
