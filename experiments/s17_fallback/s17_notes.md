# S17 — does Chronos-T5's silent `scale=1.0` tokenizer fallback actually hurt?

Question: S9 (anomaly 1) found that MeanScaleUniformBins'
`scale[~(scale>0)] = 1.0` fallback (chronos.py:178) fires in production —
weather clean 63/2100 rows, mnar_high p=0.7 524/2100 — silently switching
all-zero-observed channels to absolute units. But S9 never showed this hurts
accuracy: near-zero channels are predicted ≈0 anyway. S17 is the go/no-go:
quantify the damage, and only build a fix if the damage clears a
pre-registered bar.

**Structural fact (smoke-verified, s17_smoke.log).** For a fallback row the
observed context is all zeros, so the token stream is invariant to scale
(0/s = 0 → centre bin; NaN → pad). `output_transform` is linear in scale, and
the pass-2 scale of the H=96 two-pass loop is proportional to the pass-1
scale, so the whole prediction is exactly homogeneous: **pred(s) =
pred(1.0)·s** (smoke: bit-exact mirror of `pipe.predict`; identity exact up to
float32 rounding on ~1e-8 values). Scale-only fixes on fallback rows are
therefore analytic — no GPU rerun needed — and MSE(s) per row is the quadratic
`A·s² − 2B·s + C` with (A,B,C) the prediction/target moments.

**Bolt contrast.** Chronos-Bolt's InstanceNorm (chronos_bolt.py:111-113) uses
loc = nanmean (nan→0), scale = nan-std with `scale==0 → eps=1e-5`. An all-zero
context gets scale=1e-5, so bolt's denormalised predictions collapse to
~loc = 0 — a *deflating* fallback, the opposite sign of t5's *inflating* 1.0.
Smoke: bolt on an all-zero context predicts max|pred| = 1.2e-7.

## Anchor gate — PASS (bit-exact)

| check | S9 | S17 | result |
|---|---|---|---|
| weather clean fallback rows | 63/2100 | 63/2100 | exact |
| weather mnar_high p=0.7 fallback rows | 524/2100 | 524/2100 | exact |
| t5 ETTh1 clean MSE | 14.219885 | 14.219885 | dev −2e-16 |
| t5 weather clean MSE (bonus) | 8567.094 | 8567.094 | dev −2e-16 |
| ETT clean fallback rows | 0 | 0 | exact |

New channel detail: the clean fallback rows are ch14 `rain (mm)` (38) + ch15
`raining (s)` (25) only. Under mnar_high p=0.7 the fallback additionally fires
on the **solar channels** ch16 SWDR / ch17 PAR / ch18 maxPAR (100 each — long
exact-zero night stretches plus spike censoring leave all-zero observations)
and ch7 VPdef (24). Measured zero fractions: rain 96.4%, raining 93.6%.

## Part A — damage verification

### Synthetic intermittent grid
Two-state dry/wet Markov regimes (mean wet spell 24 steps, amplitude ~Exp(1),
noise σ ∈ {0, 0.1, 0.3} on active points, exact zeros preserved), activity
a ∈ {0.02…0.4}, 256 series/cell, L=512+H=96. Fallback rates: 62–66% at
a=0.02, 29–35% at a=0.05, 7–9% at a=0.1, ~0 at a≥0.2 — 788 fallback rows
pooled. A global-noise diagnostic (σ_g ∈ {0.01, 0.05} added everywhere) gives
**zero** triggers at any activity: the fallback requires *exact* zeros.

On fallback rows (pooled, n=788):
- prior fix (scale := cell mean|x|): agg improvement **−1.3e-08** (CI
  [−1.4e-08, −1.3e-08]) — nothing.
- bolt vs t5 native: **+7.6e-08** — identical.
- oracle s* (per-row least-squares, uses the future): **+17.4%** — the only
  thing that helps is knowing the future.
- mean|pred| of t5 on fallback rows: 1.4e-8 (clean-context rows) — t5's
  response to an all-centre-bin stream is numerically zero.
- Split by future: on future-zero rows the prior fix "improves" 99.8% (MSE
  ~1e-16 → ~1e-21, absolute nothing); on future-active rows (−1.3e-8) — the
  spike is missed at scale 1.0 and at any other scale.

### Real weather (S9-paired windows)
| config | n_fb | IMP_prior | IMP_oracle | bolt vs t5 | t5 mean\|pred\| |
|---|---|---|---|---|---|
| clean | 63 | +0.000 | +2.1% | +0.000 | 1.4e-8 |
| mnar_high p=0.7 | 524 | −0.000 | +36.8% | +0.000 | 6.2e-3 |

Pooled real fallback rows (n=587): prior-fix agg improvement **−1.9e-05**,
median per-row 0.0; bolt identical (0.0). On future-active rows (392/587):
exactly 0.0 — the censored spikes are missed regardless of scale. On
future-zero rows the aggregate is −97.6, a ratio-of-sums artifact that is
itself instructive: ch15 `raining (s)` has train prior mean|x| = 26.5 (>1), so
the "fix" *inflates* near-zero predictions 26× (ΣMSE 0.42 → 295 over 60
rows), while ch14 rain (prior 0.014) improves 2.57 → 5e-4. Both are absolute
peanuts against these channels' ~7500 per-row MSE scale — but it proves a
blind prior-floor can backfire on channels whose natural scale exceeds 1.

### Why the fallback is harmless
Chronos-T5's response to an all-centre-bin token stream is ≈0 (the same
fallback existed throughout pretraining — the model has internalised it).
The residual error on fallback rows is an **information problem** (a spike
after 512 zeros is unpredictable), not a scale problem: only the oracle scale
— which implicitly encodes the future — recovers anything (17–37%). bolt's
eps floor lands on the same ≈0 predictions, so bolt shows identical MSE on
every fallback row (per-row diff 0.0).

## Go / No-Go — **NO-GO**

Pre-registered rule: GO iff aggregate prior-fix improvement ≥5% on either
pooled fallback population with top-5-row share <50%.
Measured: real **−1.9e-05**, synth **−1.3e-08**. No damage → Part B degraded
to a short verification. Conclusion: document the trap, don't patch.

## Part B (degraded verification — no candidate patch adopted)

Decision was NO-GO, so Part B ran the short form: two representative
candidates per model, checking (i) fallback-row effect, (ii) collateral damage
on normal rows, (iii) ETT clean regression.

**t5 `floor10`** — scale := max(mean|x_obs|, channel/cell prior):
- Fallback rows: Δ ≈ 0 by construction (analytic): weather −1.9e-05, synth
  −1.3e-08. Confirms Part A exactly.
- Collateral on non-fallback rows (scale bumped where 0 < mean|x_obs| <
  prior): synth cell ratios range 0.94 (a0.4:s0.3) to **1.266** (a0.2:s0.3),
  +5–15% worse on all a=0.1 cells. ETT regression: ETTh1 1.0108, ETTm1
  0.9892 — ±1% both ways.
- weather clean whole-set ratio 0.727 is the S9-anomaly-2 artifact (bumping
  near-zero rain windows collapses tokens to centre bins → predictions →0 →
  error drops on zero futures); not a real improvement.

**t5 `nanskip`** — all-zero-observed rows → all-NaN (model treats as missing):
- **Catastrophic on synthetic fallback rows**: a0.02:s0.0 agg improvement
  −22.9 (ΣMSE 4.71 → 112.9, 24× worse; median row −89%; 1.3% of rows
  improved). Whole-set up to 7.6×. The all-centre-bin token stream IS the
  right evidence; replacing it with an empty context makes the model sample
  its global prior (scale 1.0), far from zero.
- weather fallback rows: clean −14.7%, mnar +0.36% — noise, not a fix.
- ETT exactly 1.0 (no ETT row triggers).

**bolt `stdfloor10`** — scale := max(nan-std, channel prior-std):
- weather clean fallback rows **−9.9%** (worse): replacing eps=1e-5 by a real
  std inflates predictions on zero-future rows; mnar +1.2% (noise). Synth fb
  +0.2–0.3% (negligible).
- Collateral: weather clean whole-set 1.427 (the floor bumps ordinary
  low-variance windows), ETT regression +2.07% (ETTh1) / +0.81% (ETTm1).

**bolt `nanskip`**: ≈0 everywhere (+0.07% / +0.03% weather fb), ETT exactly
1.0. bolt's native eps path is already the best behaviour.

### Bottom line
The scale=1.0 fallback is a real code smell but **empirically harmless**:
Chronos-T5's answer to an all-zero token stream is ≈0 regardless of scale
(pretraining internalised the fallback, which exists in the training pipeline
too), and bolt's eps floor lands on the same ≈0 predictions. The error on
fallback rows is an information problem (unpredictable spikes after long
zeros), not a scale problem — only the future-aware oracle scale recovers
anything (17–37%). Patching the scale gains nothing and can backfire (prior >
1 channels; near-zero normal windows). Recommendation: **do not patch;
document the trap** (a global-prior floor is *not* a free win — it can inflate
near-zero predictions on channels whose natural scale exceeds 1, and
"improvements" on weather are the zero-artifact family). Residual action:
none for t5/bolt inference; for future tokenizer work, note the fallback only
fires on EXACT zeros (any sub-quantisation noise disables it — global-noise
diagnostic: 0 triggers at σ_g ≥ 0.01).

## Part A-tail — edge-bin clamp on near-zero-scale channels: also harmless

Second path flagged after the fallback: on a near-zero-but-positive scale
(rain: window scale ≈ 0.005–0.03), a context spike x/scale ≫ 15 exceeds the
tokenizer's ±15 bin limits and is **clamped to the edge bin** — the model sees
"maximal bin", not the amplitude. S9 measured edge_frac ≈ 0.13% on weather
clean. Test (3 arms, paired per row): (i) native; (ii) oracle global scale
(train nonzero-mean|x| per channel, e.g. rain 0.38 — spikes land mid-scale);
(iii) log1p transform + expm1 back (median is monotone-equivariant, so the
point forecast stays a median). Row selection uses the real tokenizer's output
(token id ∈ {2, 4095}).

**Real weather** (rain/raining/SWDR/PAR/maxPAR rows of the 100 S9 windows,
n=500): 121 clamped rows — rain 62/100, raining 59/100, **solar 0/100**
(day-night windows keep mean|x| high, so max/mean < 15; solar never clamps).
On clamped rows: oracle agg improvement **+1.7e-08**, top-decile (future
spike points) Σ-ratio **0.99999997**; log1p agg −9e-10, td 1.0000000011.
Direct inspection of the most extreme clamped rows (raining = 600 s spikes at
19–40× the window scale, all pinned to the edge bin; futures with up to
Σ|y| = 6340 s of rain): t5's mean|pred| is ≈0.0000 under **all three arms**.
The model's strategy on these channels at H=96 is simply "predict ~0",
whatever the amplitude representation — so the clamp has nothing to damage.

**Synthetic heavy-tail** (same two-state Markov generator, amplitude
Weibull(0.5), σ=0.1; clamp rates 179/256 at a=0.05, 254/256 at a=0.2):
a=0.05 → oracle −0.99% / log1p −0.71%, td ratios 1.009 / 1.003; a=0.2 →
oracle +0.97% / log1p −0.54%, td 0.997 / 0.998. All within ±1% — no harm, no
benefit.

**Judgment (same 5% bar): clamp_harmful = False, log1p_mitigates = False.**
The clamp path is documented as harmless for inference, same as the fallback.
Two honest caveats: (a) this is an inference-time statement — whether
edge-bin saturation during *pretraining* suppressed amplitude learning on
such channels is not testable by inference-time patches (our arms show the
model ignores the amplitude even when given it); (b) on NON-clamped rain
rows the oracle/log1p arms show agg MSE "improvements" (+37% / +27%) with
top-decile ratios 1.7× / 2.5× WORSE — the S9-anomaly-2 zero-artifact family
(shrinking predictions toward 0 helps MSE on zero futures while losing the
rain signal); not a real effect, and another warning against reading weather
rain MSE at face value. bolt has no quantizer (continuous patches) — no clamp
path exists there by construction.

## Caveats
- The homogeneity identity makes the "fix" space one-dimensional; a fix that
  changes the *token stream* (e.g. nanskip) is not covered by it and is tested
  explicitly in Part B.
- Oracle s* uses future information — an upper bound, not deployable.
- Synthetic amplitudes are O(1); the conclusion is scale-free by construction
  (predictions are exactly homogeneous in s on fallback rows).

Artifacts: `run_s17_fallback.py`, `s17_results.json` (keys: partA, partB,
partA_clamp), `s17.png`, `s17_notes.md`; logs `s17_smoke.log`, `s17_A.log`,
`s17_B.log`, `s17_clamp.log`.
