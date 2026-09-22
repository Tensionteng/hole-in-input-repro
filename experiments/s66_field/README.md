# S66 — the 2025 zero-shot models under the Table-4 (GIFT-Eval dose-response) protocol

Extends the paper's Table 4 pool to the three shipped 2025 models — FlowState r1
(`models_local/flowstate-r1`, IBM), TiRex 1.1 (`models_local/tirex`, NX-AI) and
TimesFM 2.5 (`google/timesfm-2.5-200m-pytorch`, HF cache). Zero-shot inference
only; no training exists for these arms. Write-only directory.

## Scripts and artifacts

- `run_s66_gift.py` → `s66_gift_dose.json` + `s66_gift_scores.npz` (+ logs)
  The s32c pipeline verbatim (`run_s32c_stratified.collect("random", 6, 250)`,
  `s32.fill`, `s32.mase`), identical to `s64_bench3/run_s64_gift.py`: 7,725 pooled
  random windows from the nine missing-containing GIFT-Eval subsets (seed 1),
  context 512, horizon 64, per-window MASE vs in-window naive-1 on observed target
  points, 4 fills x 4 models, stratified by the window's own context-NaN fraction.
  Resumable: per-window scores are checkpointed to the npz after every fill.
- `aggregate_s66_gift.py` → `s66_gift_rows.json`
  Gates + Table-4-format rows (linear fill) with all four fills kept, per-stratum
  deltas vs the three stock rows in `s32c_results.json`, and overall
  (unstratified) pooled medians.

## What each model actually receives

Fill routing is `s32.forecast` unchanged: `nan` -> the model's declared/NaN input
convention; `linear`/`zero`/`ffill` -> the "plain" convention (the filled array is
fed as the series content). Per the s30/s49 permutation probes (fill-content
sensitivity ratios):

- **flowstate** — plain rows: our filled array (uses fill content, ratio 0.85).
  nan row: its declared path zeroes the missing content and keeps only the
  missingness flag (ratio 0.00) — the fill is discarded, statistics come from
  observed points only.
- **tirex** — plain rows: our filled array (ratio 0.96). nan row: same (a)+flag
  declared convention as FlowState (ratio 0.00); fill discarded.
- **timesfm** — plain rows: our filled array (ratio 0.86). nan row: TimesFM 2.5
  applies its OWN internal interpolation over the NaN positions; any fill we
  supply on that path is never seen (ratio 0.00). Its internal fill is
  near-equivalent to `np.interp` for interior gaps: in s32c the timesfm nan and
  linear rows coincide in the three low-NaN strata and split only above 15%
  missing (boundary/long-gap handling).

So the Table-4 (linear) value for all three is genuinely the model forecasting
our linearly-filled context; the `nan` column is each model's own declared
missing-data convention.

## Gates

- **A (required by the task, tol 1e-5):** the s66 bolt-base stock arm goes through
  `s32.get_model` + `s32.forecast` — the verbatim s32c code path — and must
  reproduce `s32c_results.json`'s `bolt-base|random` bin medians on all four
  fills. Result: **PASS, max rel dev 0.0** (bit-exact, 20 comparisons; pool
  n=7,725 with stratum counts 4,308/2,327/625/208/257 identical to s32c).
- **B (supplementary):** the s66 timesfm run vs s32c's `timesfm|random` rows —
  validates the s30 wrapper and the cached 2.5 weights end to end. See
  `s66_gift_rows.json` -> `gates.B_timesfm_vs_s32c` (filled in after the run).

## Devices / seeds

cuda:0, pool seed 1 (the s32c/s64 pool), `s25.SEED` for torch/numpy, strict fp32
(TF32 off). Stock-overall reference pools: c2-stock/m2-stock from
`s64_bench3/s64_gift_scores.npz` (s64 gated that pool as identical), bolt-base
stock from s66's own gated arm.

## Gates (final)

- **A (required, tol 1e-5): PASS, max rel dev 0.0** — the s66 bolt-base stock arm
  reproduces `s32c_results.json`'s `bolt-base|random` bin medians bit-exactly
  (20 comparisons: 5 strata x 4 fills; identical stratum ns 4,308/2,327/625/208/257).
- **B (supplementary): PASS, max rel dev 0.0** — the s66 timesfm run reproduces
  s32c's `timesfm|random` rows bit-exactly (20 comparisons), validating the s30
  TimesFM wrapper and the cached 2.5 weights end to end.

## Results (all from `s66_gift_rows.json`; linear fill, random protocol)

n = 4,308 / 2,327 / 625 / 208 / 257 per stratum (identical to Table 4).

| model | <1% | 1-5% | 5-15% | 15-30% | >30% | overall |
|---|---|---|---|---|---|---|
| FlowState r1 | 0.703 | 0.908 | 1.924 | 2.285 | 1.924 | 0.839 |
| TiRex 1.1 | 0.621 | 0.827 | 1.137 | 1.453 | 1.468 | 0.742 |
| TimesFM 2.5 | 0.655 | 0.877 | 1.867 | 2.175 | 1.843 | 0.793 |

Same models under their declared NaN convention (the `nan` fill row):

| model | <1% | 1-5% | 5-15% | 15-30% | >30% | overall |
|---|---|---|---|---|---|---|
| FlowState r1 | 0.703 | 0.908 | 1.899 | 2.222 | 1.782 | 0.841 |
| TiRex 1.1 | 0.621 | 0.826 | 1.037 | 1.018 | 0.928 | 0.732 |
| TimesFM 2.5 | 0.655 | 0.877 | 1.867 | 2.167 | 1.818 | 0.793 |

Headline observations:

- **TiRex's declared path is dose-INSENSITIVE**: its nan row stays at 0.93-1.04
  from the 5-15% stratum up through >30%, so the nan-vs-linear gap widens with
  missingness (-0.1% at 1-5% -> -36.8% at >30%). On the linear fill it is the
  best of the three in every stratum and beats all three stock families from the
  paper (e.g. -37.8% vs moirai2 stock at 5-15%); on its own declared path the
  margin is larger still.
- **FlowState degrades the most** under linear fill (+40.8% vs bolt-base stock at
  5-15%, roughly TimesFM's level), and its declared path buys back only a few
  percent (-1.3% to -7.4% nan-vs-linear).
- **TimesFM's nan and linear rows nearly coincide** (its internal interpolation ~
  np.interp for interior gaps); it tracks the bolt/c2 stocks below 5% missing but
  degrades faster above it (+36.6% vs bolt-base stock at 5-15%).

Per-stratum deltas vs all three stock rows (bolt-base / chronos2 / moirai2, s32c
artifacts) and the overall pooled medians are in `s66_gift_rows.json`.

## Wall-clock (cuda:0, alone on the device)

bolt-base gate arm 23 s (+49 s pool collection); timesfm 528 s; tirex 280 s;
flowstate 44 s; pool re-collection for the resumed run ~40 s. Total ~16 min
across two invocations (the first was killed by a 600 s launcher timeout after
timesfm finished; the npz checkpoint made the resume lossless).

## Caveats

- Within-protocol numbers only: this is the paper's documented random-window,
  in-window-naive-1 MASE protocol, NOT official GIFT-Eval; absolute values are
  not leaderboard-comparable (same caveat as Table 4).
- 288 of the 7,725 windows have undefined MASE (naive-1 denominator ~ 0) and are
  excluded by nanmedian — identically for every model and identical to the s64
  pool (7,437 finite).
- TimesFM 2.5 has no local copy under `models_local/`; it loads from the HF
  cache (`google/timesfm-2.5-200m-pytorch`, snapshot with config + safetensors)
  via the mirror endpoint. Gate B pins this exact weight set to s32c's numbers.

