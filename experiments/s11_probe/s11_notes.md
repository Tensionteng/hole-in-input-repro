# S11 — linear probes: where in the encoder does the masked truth survive? (chronos-bolt-base)

Windows/masks/seeds identical to run_s8_masktoken.py (= run_s5_missing.py): 300
test windows in the last-20% region, L=512, H=96, patch=16 (32 patches/series,
position 32 = REG dropped), 2 mask seeds for mcar/block, 1 for
rank-deterministic mnar_high, bolt native nan fill. Probe-train windows: 300
more from the train region (forecast origin strictly before the test region,
seed SEED+11). Grid: {mcar, block, mnar_high} × p ∈ {0.3, 0.7} × {ETTh1,
weather}. One encoder forward per window version with
`output_hidden_states=True` → 13 tensors (0 = embedding output, ℓ = block-ℓ
output for 1..11, 12 = final-LayerNorm output). Probe = PCA(64, fit on pooled
TRAIN patches per dataset/direction/layer) → Ridge(α=1) per
(config, family, layer); metric = pooled test R². Targets are per-patch
statistics of the TRUE values in the clean window's instance-norm z-space
(primary: patch mean over the 16/missing points; auxiliary: within-patch std).
A linear-probe R² lower-bounds information content.

## Anchor gate vs S8 — PASSED (exact)

| cell (ETTh1, nan)  | s8_results.json | S11 harness |
|--------------------|-----------------|-------------|
| mcar p=0.3 relMSE  | 0.9781          | 0.9781      |
| mnar_high p=0.7    | 3.6671          | 3.6671      |

Bit-identical relMSE (same windows/masks/model; the extraction path replicates
`encode()` line-for-line and was verified against native `model.encode` in the
tiny self-test, rel max|diff| < 2e-3, limited by fp16 storage).

## Sanity gates — PASSED (with one documented deviation)

- **obs_clean** (decode patch mean from CLEAN-run patches): layer 0 = **1.0000**
  exactly on both datasets, as it must be (the patch embedding is a linear map
  of values+flags). Full curve ETTh1: 1.000/1.000/0.999/0.998/0.996/0.994/
  0.992/0.990/0.987/0.983/0.978/0.965/**0.929**; weather nearly identical
  (min 0.929). The pre-registered "R²≈1 at every layer (≥0.95)" fails at
  l11/l12 — deep mixing genuinely dilutes per-patch point detail even in clean
  windows. Pipeline-critical checks (l0≈1 exactly, high floor) hold; the 0.93
  level is the readout ceiling all missing-patch numbers below are read
  against. Missing-patch R² (≤0.81) sits well under it, so conclusions are
  unaffected.
- **input-side baseline** (ridge from the raw 32-dim [values|flags] patch
  input): R² ≤ 0.005 everywhere except ETTh1 mnar_high 0.3 (−0.129, small-n
  train/test target shift on an intercept-only fit — the features are a
  constant zero vector there). ⇒ the truth is **not present at the input**.
- **chance** (window observed mean): R² ≤ −0.07 for mcar/block and −3.98…−19.8
  for mnar_high (observed mean is anti-predictive for censored-high patches).
- **obs_in_miss** (fully-observed patches inside corrupted windows): l0–l6 R²
  0.94–1.00 for mcar/block; dips at mnar_high 0.7 (ETTh1 0.83, weather 0.67 at
  l0) — expected per-series unit mismatch (observed low-tail stats vs clean
  stats), not information loss.

## Main result — R²(layer) for fully-missing patches (clean-z space)

l0 / l6 / l12; peak layer in parentheses. n/a = family empty.

| mech × p        | ETTh1                     | weather                   |
|-----------------|---------------------------|---------------------------|
| mcar 0.3        | n/a (0 patches: 0.3¹⁶)    | n/a (0 patches)           |
| mcar 0.7        | −0.000 / 0.539 / 0.473 (l4 0.541) | −0.000 / 0.802 / 0.792 (l7 0.808) |
| block 0.3       | −0.000 / 0.552 / 0.518 (l6 0.552) | −0.000 / 0.729 / 0.718 (l8 0.730) |
| block 0.7       | −0.000 / 0.474 / 0.456 (l6 0.474) | −0.000 / 0.616 / 0.612 (l8 0.623) |
| mnar_high 0.3   | −0.129 / 0.134 / 0.090 (l3 0.179) | −0.001 / 0.455 / 0.408 (l4 0.465) |
| mnar_high 0.7   | −0.005 / 0.060 / 0.052 (l5 0.074) | −0.002 / 0.152 / 0.130 (l5 0.168) |

Test-patch counts: block 15k–116k; mnar_high 1.8k–115k; mcar 0.7 only 514
(ETTh1) / 1410 (weather) — read those two cells with wider error bars.

Curve shape is identical across all non-degenerate cells: **exactly 0 at the
embedding layer, a steep rise over blocks 1–4, peak at layers 4–8, mild decay
into the final-LN output**. The network does not *preserve* the masked truth
(it is absent at input) — it *rebuilds* a linear signal of it in the first
half of the encoder via attention transport (fully-missing patches are masked
as keys but remain queries, so transport is architecturally open), and late
layers slightly trade per-patch detail for global features.

Auxiliary target (within-patch std of the truth, p=0.7): l6 R² 0.23–0.32 for
mcar/block on both datasets — "this gap was volatile" is linearly readable
too, at about half the mean-signal strength; mnar_high sits at 0.08–0.16.

Missing-space robustness (target in the corrupted window's own units; mcar/
block only — see anomalies): ETTh1 identical to clean-space (±0.01); weather
mcar 0.7 0.802→0.782 but block deflates 0.729→0.508 (p=0.3) / 0.616→0.449
(p=0.7) at l6 — part of weather's block signal is window-level scale/level
variation that per-series renormalization absorbs. Qualitative conclusions
unchanged.

## Distance decomposition (block, fully-missing patches; l6 R²)

| dist to gap edge | ETTh1 p=0.3 | ETTh1 p=0.7 | weather p=0.3 | weather p=0.7 |
|------------------|-------------|-------------|---------------|---------------|
| 0–7 points       | 0.569       | 0.508       | 0.770         | 0.709         |
| 8–23             | 0.460       | 0.444       | 0.590         | 0.549         |
| ≥24              | n/a (n=49)  | 0.298       | n/a (n=140)   | 0.390         |

**Information decays with depth into the gap** (0.51→0.44→0.30 on ETTh1 p=0.7;
0.71→0.55→0.39 weather p=0.7), monotone in every cell with data. Contrast with
S7: representation *drift* (clean-vs-corrupted cosine) did **not** decay with
distance — drift is global, but the decodable truth content is local. The
perturbation of a gap-center token is just as large as an edge token's, yet
less of it is usable signal. Transport is partly local (nearby observed
patches carry the reconstructive context).

## Partially-missing patches (mean of the missing positions only)

mcar keeps the missing-point mean decodable at 0.66–0.95 (l6; ETTh1 0.73/0.77,
weather 0.94/0.95) — scattered gaps are almost interpolable from the patch's
own observed points. Block: 0.50–0.84. mnar_high: 0.23–0.39. All flat-to-mild-
peak over depth; even at l0 these are ≫0 because the input patch itself
carries the observed points (correlation structure), unlike the constant-input
fully-missing case.

## Answers

1. **Where is the truth still linearly decodable?** For random-like
   missingness (mcar/block): nowhere at the input (R²=0 by construction),
   then **reconstructed by layers 1–4 and peaking at layers 4–8**
   (R² = 0.45–0.55 ETTh1, 0.62–0.81 weather), persisting to the final layer
   with mild decay. The information is NOT destroyed inside the forward pass —
   it is *created* there, from context.
2. **Ceiling implication — two regimes.** (a) mcar/block: the ceiling is NOT
   the input side. Mid-encoder gap tokens already carry a 0.5–0.8-R² linear
   readout of the truth that the forecast head does not fully exploit (S5/S8:
   nan-path forecasts degrade far more than a 0.5–0.8 fidelity signal
   implies), so representation-level/repair-head approaches have real headroom
   above the input-side ladder. (b) mnar_high: R² ≈ 0.05–0.17 (ETTh1 both p,
   weather p=0.7) — the censored extreme levels are gone at the input and are
   NOT rebuilt; the input-side repair ladder is essentially the ceiling there,
   matching S5/S8 (linear fill unbeatable on mnar) and S10 (Tobit reversal
   fails on real curtailment).
3. **Is mnar_high ≈ 0 (information-theoretic floor)?** For the extreme-
   deviation component: yes. One exception above the naive floor: weather
   mnar_high 0.3 reaches 0.41–0.47 — but that decodable part is the
   *contextually predictable level* of smooth channels during high episodes
   (plus a readable volatility signal, std-R² 0.23–0.27), not the censored
   deviation from it. Honest reading: the floor holds for what censoring
   actually destroys; seasonal/level structure leaks through context, more so
   on smooth datasets.
4. **Distance:** information decays with distance into the gap (unlike S7's
   distance-flat drift) — drift ≠ information.
5. **Layer budget:** reconstruction is done by layer ~4–6; layers 7–12 add
   nothing for this signal (slight decay). Any probe-guided repair head should
   read from mid-encoder, not the final output.

## Anomalies / deviations

- obs_clean pre-registered ≥0.95-everywhere gate fails at l11/l12 (0.965/0.929)
  — genuine deep-mixing dilution in clean windows; l0 = 1.0000 exact. Gate
  re-set to (l0 ≥ 0.99) ∧ (min ≥ 0.90), documented in run_s11_probe.py.
- mcar p=0.3 has **zero** fully-missing patches (0.3¹⁶ ≈ 4×10⁻⁸) — degenerate,
  same as S8's Exp4 note; symmetric: mcar p=0.7 has zero fully-observed
  patches (obs_in_miss n/a).
- Missing-space (_ms) targets: weather windows with exactly-constant observed
  stretches hit bolt's scale=eps floor and explode z_m (first pass: weather
  block _ms ≈ 0.00, mnar 0.7 _ms = −32.6). Fixed by excluding eps-floor
  windows; under mnar_high the observed low-tail variance collapses by design,
  so _ms is not computed there at all (ill-posed target, outlier artifact).
- block achieved rates are 0.250/0.509 for nominal 0.3/0.7 (block overlap) —
  inherited unchanged from the S5 mask construction, consistent with all
  previous rounds.
- ETTh1 mnar_high 0.3 full_miss l0 = −0.129: intercept-only ridge on constant
  input with n≈1.6k and train/test target shift; not meaningful, kept for the
  record.

## Artifacts

`run_s11_probe.py` (--tiny / --anchor / --dataset / --figure; docstring has the
pre-registered hypotheses H1–H3), `s11_results.json` (anchor, clean_probe,
per-config families × 13 layers incl. baselines, block distance bins),
`s11.png` (rows: fully-missing / partially-missing / distance+std aux; columns:
mechanism), logs `s11_tiny.log`, `s11_anchor.log`, `s11_ETTh1.log`,
`s11_weather.log`, `s11_figure.log`.
