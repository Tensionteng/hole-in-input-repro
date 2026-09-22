# S7 — deep attribution of missing-context damage (chronos-bolt-base)

Grid: mechanisms {mcar, block, mnar_high, mnar_extreme} × p ∈ {0.3,0.5,0.7} ×
{ETTh1, ETTm1, weather}; windows/mask seeds identical to run_s5_missing.py
(300 windows, 2 mask seeds for mcar/block, 1 for rank-deterministic mnar,
L=512, H=96, median quantile). Layers 1–2 use {mcar, block, mnar_high} ×
p ∈ {0.3,0.7} × fills {zero, linear, nan}. bolt only (timesfm/moirai deferred:
different attention interfaces and fill conventions).

## Hard-anchor check vs s5_fix_notes.md / s5_missing_results.json — PASSED

Cell A (= plain zero-fill), dataset-avg relMSE at p=0.7, S5 value → S7 value:

| mechanism    | S5    | S7     | Δ      |
|--------------|-------|--------|--------|
| mcar         | 13.74 | 13.744 | +0.0%  |
| block        | 5.93  |  5.927 | −0.1%  |
| mnar_high    | 11.32 | 11.318 | −0.0%  |
| mnar_extreme | 12.13 | 12.125 | −0.0%  |

Cell B recovery (relA−relB)/(relA−1) at mcar p=0.5: S7 82.6/78.9/82.3% vs
S5 82.3/78.7/82.3% (ETTh1/ETTm1/weather) — OK.

## Methodological correction to the S5 interpretation (important)

The S5 "oracle statistics probe" (`zero_oscale`: rescale input by
s = mean|x_clean|/mean|x_zero|) is, by the **scale-invariance of bolt's
InstanceNorm**, exactly a *pure rescale of the forecast by s*: the input
rescale cancels inside (x−loc)/scale, so the network sees bit-identical
normalized values and only the output unscale changes (verified numerically:
B-cell relMSE matches s5_missing_results.json zero_oscale to 4 decimals).
So the 82.3% recovery at mcar p=0.5 must be credited to the **output-unscale
magnitude channel**, not to input-side normalization.

The literal InstanceNorm loc/scale swap (hook forcing clean-context mean/std,
cell **Bh**) was implemented and verified (identity test: overriding with the
context's own stats reproduces the no-hook forecast to 5.7e-6). It does **not**
reproduce the 82.3%: it recovers only 62.7/58.4% on ETTh1/ETTm1 and *explodes*
on weather (relMSE 452.7 at mcar p=0.5). Reason: with clean stats, a missing
position becomes the token −loc/scale = −mean/std in normalized space, which
for low-CV channels (e.g. pressure ≈ 1000±5 → −200σ) is an extreme
out-of-distribution input; under mcar essentially *every* patch contains such
tokens. Forcing clean stats into the AR second block as well (--block2
override) also blows up weather (relMSE 280.7) via tiny-std channels; all
reported hook cells therefore use natural block-2 stats.

## Layer 0 — 2×2 decomposition (cells: A zero+polluted, B zero+oracle-scale
[=S5 zero_oscale], C clean+polluted-iNorm [hook], D clean; diagnostics Bh, Cr
[= clean × mean|x_zero|/mean|x_clean|])

Two exact factorizations of the damage relA−1 (dataset-avg):
"scale" uses stats-only cell Cr, values-only cell B;
"inorm" uses C and Bh.

**mcar** (scale factorial is near-additive at p≥0.5):

| p   | A      | B      | Cr     | stats% | values% | inter% | B-recovery |
|-----|--------|--------|--------|--------|---------|--------|------------|
| 0.3 | 1.320  | 1.936  | 2.185  | (ill-conditioned: damage tiny, B overshoots — the S5 p≤0.3 artifact) | | | −192.8% |
| 0.5 | 5.871  | 1.874  | 4.481  | 71.5   | 17.9    | 10.6   | 82.1% |
| 0.7 | 13.744 | 10.428 | 7.986  | 54.8   | 74.0    | −28.8  | 26.0% |

Per-dataset stats share at p=0.5: 58.7/62.9/73.1% (ETTh1/ETTm1/weather);
at p=0.7: 53.1/54.0/55.0% — remarkably stable across datasets. p-dependence:
**the scale-statistic share peaks at p=0.5; at p=0.7 the value-error share
dominates** (62–76%), consistent with S5's 24–38% recovery at p=0.7
(S7 per-dataset: 38.2/36.3/24.3%).

**block / mnar**: the oracle scale correction *backfires* (B > A everywhere;
e.g. mnar_high p=0.7: B=32.2 vs A=11.3) — under informative/contiguous
missingness the deflated scale is partially *correct* (values really are
censored), so re-inflating the forecast overshoots; the decomposition is
non-additive there (values share >100%, interaction negative). Hook cell C
(polluted stats, clean values) is nearly harmless under mcar/block
(relMSE ≤1.08–1.50) but grows under mnar_high to 3.13 at p=0.7
(ETTh1 3.33 / ETTm1 1.71 / weather 4.35): biased observed statistics hurt
even with perfect values.

**Bh (literal iNorm swap) is mechanism-dependent**: explodes under mcar
(weather 387–523, ETT 1.0–9.0) but is the *best* zero-value cell under block
(p=0.3: 1.07/1.17/1.45 vs A 1.08/1.21/3.43; weather p=0.7: 5.62 vs A 14.03) —
contiguous gaps produce few all-missing extreme-token patches, and correct
stats then realign everything else.

## Layer 1 — representation drift (cosine / relative L2 vs paired clean run)

Drift grows with depth (embed cos 0.77–0.90 → final layer 0.17–0.64 depending
on condition) and is smallest for linear, largest for zero; nan sits between.

- **mcar (scattered)**: drift is a **uniform global shift**, not localized —
  corrupted vs fully-observed patches drift identically (p=0.3 zero: cos
  0.485 vs 0.475; only ~10% of series even have a fully-observed patch).
  Cause: the polluted InstanceNorm statistics shift/scale *every* token, and
  every patch embeds the same −loc/scale garbage points.
- **block (contiguous)**: **binary localization on top of a global floor** —
  corrupted patches drift more (final-layer cos 0.32–0.39) than observed
  patches (0.46–0.61), but observed patches still drift far from clean.
- **Distance decay hypothesis: REJECTED.** Binned by distance to nearest
  observed point, block p=0.7 zero-fill drift is a step function: cos drops
  from 0.54 (patches touching an observed point) to ≈0.44–0.45 for patches
  fully inside a gap, then stays flat from 1 to 63 time-steps into the gap
  (0.438 @1 step vs 0.448 @32–63 steps). No gradation.
- nan fill drifts less than zero at the final layer (e.g. block p=0.7: 0.602
  vs 0.360) — the mask channel + observed-stats normalization visibly reduce
  representation damage.

## Layer 2 — attention audit (encoder self-attention, 12 layers × 12 heads,
33 positions = 32 patches + REG at the last position)

- **Entropy: essentially unchanged** (negative result). Head-averaged entropy
  2.04–2.10 nats across all conditions vs 2.066 clean; per-head spread
  (0.6–3.2 nats) is also unchanged. No entropy collapse, no global rewiring
  of attention patterns. (nan under block/mnar p=0.7 drops slightly to
  1.86–1.91, mechanically, because fully-missing keys are masked out.)
- **No attention sink on garbage patches.** Under zero-fill, corrupted-patch
  attention mass tracks the uniform baseline almost exactly (enrichment
  0.987–1.024 across mechanisms, p, and layers): attention is *blind* to
  corruption. At mcar p=0.7 that means **96% of attention mass lands on
  corrupted tokens** — the damage mechanism is mass *dilution* onto garbage,
  i.e. destructive, not a protective sink. linear-fill behaves identically
  (enrichment ≈0.99) — its patches are not discounted either.
- **The nan path is the protective one**: enrichment 0.03–0.50 (corrupted
  patches strongly avoided; fully-missing patches hard-masked). bolt's native
  mask channel does exactly what it should — S5's "nan beats linear on block"
  has its attention-level signature here.
- **REG token**: mild extra sink at layer 0 only under zero-fill (mass 0.088
  clean → 0.145 zero, mcar p=0.7), flat ~0.03 elsewhere; slight REG-mass rise
  under nan block/mnar p=0.7 (0.046/0.062 avg). **First patch: no sink**
  (~0.02–0.03, all conditions). No head specialization appears under
  corruption (per-head corrupted-mass range is similar in clean runs).

## Answers to the three questions

1. **Stats vs values vs p**: under mcar the global scale-statistic share of
   zero-fill damage is ~55–73% at p=0.5 (peaks there, near-additive with a
   ~10–24% interaction), falling to ~54% at p=0.7 where value corruption
   dominates (62–76%); at p=0.3 the damage is too small and the scale
   correction overshoots (ill-conditioned). Under block/mnar the scale fix
   backfires — damage there is value-side / information loss, with biased
   observed statistics adding a growing hook-C component at high p
   (mnar_high p=0.7: 3.13). Crucially, the fixable "statistics" damage is the
   **output-unscale magnitude**; literal input-side InstanceNorm stat swaps
   (Bh) are mechanism-dependent and dangerous under scattered missingness.
2. **Drift**: uniform global shift under scattered missingness (mcar, and
   mnar at high p); binary localization on a global floor under block; grows
   with depth; **no distance decay** into gaps — a step, not a gradient.
3. **Sink**: none on garbage patches. Zero-fill attention mass is diluted
   onto corrupted tokens at exactly their frequency (destructive dilution);
   the nan path's mask channel produces strong protective avoidance
   (enrichment 0.03–0.5). REG absorbs a mild layer-0 excess under zero-fill;
   entropy is unchanged everywhere.

## Artifacts

`run_s7_attrib.py` (main, 8-way sharded), `analyze_s7.py` (merge+summary+
figure), `s7_attrib_results.json` (per-window aggregates only, 43.8MB),
`s7_attrib.png`, logs `s7_shard0..7.log` + `s7_anchor_test.log`.
