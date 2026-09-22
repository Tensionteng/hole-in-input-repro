# S13 — attention knockout: causal evidence for where the masked truth is rebuilt (chronos-bolt-base)

S11 (linear probes) found the true values of fully-missing patches become
linearly decodable in the mid-encoder (R² = 0 at embedding → rise over blocks
1–4 → peak at layers 4–8 → mild decay). Attention was implicated by
elimination only. S13 intervenes: in each encoder block of a knocked segment,
every fully-missing patch QUERY has all 32 patch keys set to −inf before the
softmax, leaving only the appended [REG] token (position 32) — attention mass
redirects onto REG exactly (verified bit-exact in smoke). Tensor shapes,
position_bias sharing, and all unknocked rows/positions are untouched
(smoke: untouched patch positions bitwise identical to native).

Windows/masks/seeds identical to S11 (= S5/S8): 300 test + 300 probe-train
windows, L=512, H=96, patch=16, 2 mask seeds, native nan fill. Grid: {ETTh1,
weather} × {block 0.3, block 0.7, mcar 0.7} × knockout {none, l0-3, l4-7,
l8-11, all}. Measurements per cell: (1) full_miss probe R² (S11 pipeline,
PCA(64)+Ridge(1.0), but PCA basis fit per (dataset, segment) on that
segment's own train patches — each condition measured on its own merits);
(2) forecast relMSE vs native clean.

## Pre-registered predictions (frozen 2026-08-13 before any grid run; only
## smoke + relMSE anchor had executed)

- **P1**: knocking l0-3 OR l4-7 collapses full_miss probe R² at l6 (≥ 50%
  relative drop vs no-knockout) — S11's curve rises over blocks 1–4 and peaks
  at 4–8, so cutting transport in either half of the rise should remove most
  of the l6 signal. (User's registered form: "敲除 l0-7 时 R² 大幅塌陷".)
- **P2**: knocking l8-11 leaves l6 R² essentially unchanged (≤ 10% relative
  drop; l6 = block-5 output is upstream of the knocked layers) and reduces
  l12 R² only partially (≥ 50% retention) — the residual stream carries what
  was rebuilt by layer 7 straight through the knocked layers.
- **P3**: knockout=all drives l6 R² back to the embedding level (|R²| ≤ 0.05).
  If not → hook broken or bypass exists → investigate before reporting.
- **P4** (revised at smoke time, before any grid run): relMSE under
  full-miss-query knockout is EXACTLY unchanged at every segment. bolt's
  `decode()` cross-attends with `encoder_attention_mask` = the patch mask
  (chronos_bolt.py l.429-435), which excludes fully-missing positions, and
  gap tokens are never keys in the encoder — their rebuilt content is
  architecturally invisible to the forecast head. The naive form ("relMSE
  degrades where R² collapses") is falsified by architecture inspection; the
  grid verifies exact-zero at scale.
- **P5 (sanity)**: obs-query knockout (all layers) degrades relMSE clearly
  more than full-miss-query knockout — proves the knockout machinery bites.

## Gates — ALL PASSED

- **smoke** (s13_smoke.log): empty knockout ≡ native encode (rel 2.7e-4,
  fp16 storage noise) and ≡ native predict (0.0); all-knockout patch-key mass
  exactly 0 / REG mass 1.0 at all 12 layers (259 queries); untouched patch
  positions bitwise 0; knocked positions move (max 2.15); predictions bitwise
  identical under all-knockout (decoder masks gap positions); segment
  selectivity (l0 knocked exactly, l4 free); no NaN.
- **relMSE anchor** (s13_anchor.log): all 6 grid cells reproduce
  s8_results.json exp1 'correct' relMSE **bit-exactly** (4 decimals), clean
  MSE bit-exact too.
- **R² anchor** (none-segment vs s11_results.json, ±5% on l6/l12): PASS on
  all 12 cells, max deviation 0.6% (e.g. ETTh1 block 0.3 l6 0.5529 vs S11
  0.5522; weather mcar 0.7 l6 0.8031 vs 0.8019). l0 values bit-identical
  (full-rank PCA at the embedding layer is basis-independent).

## Result 1 — knockout segment × probe R² (l6 / l12), full_miss family

ETTh1:

| mech        | none        | l0-3        | l4-7        | l8-11       | all           |
|-------------|-------------|-------------|-------------|-------------|---------------|
| block 0.3   | 0.553/0.518 | 0.494/0.488 | 0.532/0.506 | 0.553/0.517 | **0.006**/−0.003 |
| block 0.7   | 0.475/0.456 | 0.428/0.423 | 0.454/0.438 | 0.475/0.447 | **0.003**/0.001  |
| mcar 0.7    | 0.536/0.479 | 0.510/0.481 | 0.521/0.480 | 0.536/0.478 | **−0.017**/−0.012 |

weather:

| mech        | none        | l0-3        | l4-7        | l8-11       | all           |
|-------------|-------------|-------------|-------------|-------------|---------------|
| block 0.3   | 0.729/0.718 | 0.688/0.706 | 0.707/0.687 | 0.729/0.705 | **0.054**/0.024  |
| block 0.7   | 0.615/0.612 | 0.579/0.598 | 0.584/0.568 | 0.615/0.593 | **0.016**/0.010  |
| mcar 0.7    | 0.803/0.791 | 0.797/0.792 | 0.775/0.769 | 0.803/0.787 | **−0.025**/−0.030 |

Relative l6 drop: l0-3 → −0.7…−10.7%; l4-7 → −2.8…−5.0%; l8-11 → exactly 0.0
in all 6 cells (upstream readout, as it must be); all → −92.6…−103%.

Full 13-layer curves (s13_results.json) give the clean causal signatures:

- **l0-3 knocked**: R² flat ≈ 0 over hidden 1–4 (e.g. ETTh1 block 0.3: 0.009
  at layer 4 vs 0.540 unknocked — the native early rise is abolished), then a
  **jump at layer 5** (0.460, the first free block) and 89–94% recovery by l6.
  Blocks 4–5 alone rebuild nearly the whole signal from scratch.
- **l4-7 knocked**: normal rise over blocks 0–3 (0.540 at layer 4 = 97.7% of
  the none-l6 value), flat across the knocked region, **no rebound at 8–11**.
- **l8-11 knocked**: identical to none through layer 8; l12 retention
  96.9–99.8% — the residual stream carries the rebuilt signal through the
  knocked layers; blocks 8–11 contribute nothing new in any condition.
- **all knocked**: flat at the REG floor everywhere (≤0.026, except weather
  block 0.3 = 0.054 — see anomalies). Internal consistency check: within
  knocked layers the l0-3 and all curves are **numerically identical**
  (weather block 0.3 layers 2–4: 0.037/0.053/0.059 in both) — that floor is
  exactly the "REG-only" channel, independent of segment.

## Result 2 — knockout × relMSE

All 24 cells (2 ds × 3 mech × 4 active segments) are **float64-identical** to
the no-knockout relMSE (e.g. ETTh1 block 0.7 = 1.0310734378415765 in every
segment; verified numerically in s13_results.json, not just at print
precision). P4 confirmed exactly: the rebuilt truth in gap tokens
has **zero** causal channel to the forecast — gap tokens are never keys in
the encoder (chronos_bolt.py l.303) and are masked out of the decoder's
cross-attention (l.429-435). S11's "the forecast head does not fully exploit
the mid-encoder signal" is sharpened to "**the head cannot read it at all**".

Sanity (obs-query knockout, all layers) relMSE: ETTh1 block 0.3 0.985→**1.960**
(+99%), ETTh1 mcar 0.7 1.053→1.126 (+7.0%); weather block 0.3 1.031→1.079
(+4.6%), weather mcar 0.7 1.320→1.362 (+3.2%). The same operation strongly
moves predictions when applied to tokens the head actually reads — the
null result above is therefore informative, not a dead hook.

## Pre-registration scorecard

- **P1 FALSIFIED.** Max single-segment l6 drop is 10.7% (ETTh1 block 0.3,
  l0-3), nowhere near ≥50%. Reconstruction is **not** localized to a
  necessary 4-block segment: blocks 4–5 alone rebuild 89–94% of the signal
  from zero (l0-3 knocked), blocks 0–3 alone reach 96–98% (l4-7 knocked).
  Transport capacity is redundant across blocks 0–7.
- **P2 CONFIRMED** (stronger than registered): l6 exactly unchanged; l12
  retention 96.9–99.8% (bar was ≥50%).
- **P3 CONFIRMED on 5/6 cells** (|R²| ≤ 0.026); weather block 0.3 = 0.054
  marginally over the 0.05 bar — REG-channel window-level leak (below), not
  a hook defect (smoke proved exact-zero patch-key attention).
- **P4 CONFIRMED EXACTLY** (bit-identical relMSE in all 24 cells).
- **P5 CONFIRMED**: obs-query knockout degrades relMSE in all 4 sanity cells;
  full-miss-query knockout moves it by exactly 0.

## Conclusions

1. **Attention is causally necessary for the reconstruction.** Cutting every
   full-miss query's attention returns the probe to the input level
   (R² ≈ 0.00, embedding floor) in all 6 cells. S11's elimination-based
   attribution ("attention is the only cross-position channel") is now an
   intervention result.
2. **But the layer budget is redundant, not localized.** S11's curve (rise
   blocks 1–4, peak 4–8) describes the *preferred* route of an unperturbed
   network, not a necessary one: when blocks 0–3 are knocked, blocks 4–5
   rebuild ~90% of the signal in a single step (layer-5 jump); when 4–7 are
   knocked, 0–3 have already built ~97%; 8–11 never contribute. S11's
   "reconstruction is done by layer ~4–6" refines to: *preferred early,
   substitutable across blocks 0–7*.
3. **The rebuilt signal has zero native predictive function.** relMSE is
   bit-unchanged under every knockout because bolt's decoder masks gap
   positions out of cross-attention (and gap tokens are never encoder keys).
   The 0.5–0.8-R² signal S11 found is computed as a byproduct (plausibly
   useful for pretraining-style objectives) and is architecturally unread by
   the forecast head. For the repair agenda this is the best possible
   outcome: a repair head reading mid-encoder gap tokens has the **full**
   0.5–0.8 R² headroom — none of it is already "used" by the model.
4. **Consistency with S11**: the correlational curve and the causal curves
   agree on everything testable — zero at input, early-mid encoder
   construction, late-layer irrelevance, residual-stream persistence. The
   only correction is P1: correlation-localization ≠ causal-localization;
   the causal assay shows distributed redundant capacity.

## Anomalies / deviations

- P1 falsified (headline deviation, above).
- weather block 0.3 all-knockout l6 R² = 0.054 (bar: ≤0.05): the [REG]
  redirect — by design the one channel left open — leaks window-level
  level/scale info, which is mildly predictive of patch means on smooth
  weather channels (S11's missing-space analysis found the same window-scale
  component in weather block). Proof it is REG and not a patch-key leak:
  smoke verified exact-zero patch-key mass, and the l0-3/all curves coincide
  numerically inside the knocked layers (the REG floor is segment-independent).
  ETTh1's floor is ≤0.006.
- Sanity obs-knockout moves mcar 0.7 cells (+3.2%/+7.0%) although a
  block-1 context has essentially zero fully-observed patches (0.3¹⁶≈4×10⁻⁸):
  the hit lands in the second autoregressive block, where the appended
  64-point first-block forecast forms 4 fully-observed patches that do get
  knocked. Expected, kept for the record.
- weather obs-knockout effect (+4.6%) is much smaller than ETTh1's (+99%):
  an attention knockout leaves each observed token's residual stream (its own
  embedding — an exact linear readout of its values, S11 l0 R² = 1.0 — plus
  per-position FFN) intact; only cross-patch context is cut. ETTh1 forecasts
  lean on cross-patch context far more than smooth-weather ones. The
  asymmetry is mechanistically expected, not a dose-response problem.

## Artifacts

`run_s13_knockout.py` (docstring = hypotheses + design), `s13_results.json`
(anchor incl. bit-exact S8 replay, r2_anchor gate detail, probe_ko per
ds×segment×mech full 13-layer R² + n's, relmse_ko, sanity_obs), `s13.png`
(rows 1–2: R²(layer) per ds×mech with S11 reference; row 3: relMSE bars +
sanity, l6-R²-by-segment summary), logs `s13_smoke.log`, `s13_anchor.log`,
`s13_ETTh1.log`, `s13_weather.log`. Runtime: smoke ~1 min, anchor ~5 min,
grid ~15 min on 4×A800 (GPU 0–3).
