# S70B — R/C probe in the long/full-outage regime: notes

Date: 2026-09-03. Inference only; no training. Same pipeline, conventions,
checkpoints, and discipline as S70 (see s70_notes.md). Work confined to
`tsfm_missing/experiments/s70_rcprobe/`.

## Motivation

S70's P2 failed informatively: at rates 0.3/0.7 the CPT retrofit never NEEDS
the context — enough target history survives that its declared-flag robustness
keeps it at ~1.01–1.06 relMSE regardless of q_c, so the substitution signature
had no room to appear. S70B measures the discriminating regime: long/full
outages, where univariate extrapolation is impossible and the model must choose
between the repair R and the context C.

## Grid

(mech, rate) ∈ { (mcar, 0.9), (block, 0.9), (block, 1.0) } — (mcar, 1.0)
skipped: degenerate with (block, 1.0). Everything else identical to S70:
q_r ∈ {zero, linear, oracle} x q_c ∈ {clean, corrupt} x 3 models (vanilla /
ctl5k / cpt, same checkpoints as S70) x 3 datasets x 6 target channels x 150
windows = 972 cells. Paired per-window relMSE vs the model's OWN clean
univariate forecast, median per cell.

Conventions carried over from S70 (documented in s70_notes.md): fills declared
through the dual endpoint for ctl5k/cpt (target AND corrupt neighbours), silent
for vanilla; neighbour corruption uses independent masks (mask_seed=1) at the
same rate/mechanism; K=4 neighbours from first-80% correlations; same windows,
same seeds, strict fp32.

### Rate-1.0 interpretation (flagged explicitly)

`make_mask("block", 1.0)` draws 21 nominal blocks of 24 over L=512, which with
overlap covers only ~62.5% ± 6 of the window (measured; block 0.9 covers ~57%
actual). That is NOT the intended regime: the motivation is full outages where
univariate extrapolation is impossible, and (block, 1.0) must be degenerate
with (mcar, 1.0) — which masks everything. S70B therefore implements
**(block, 1.0) = all L positions masked** (target, and neighbours when
corrupt). (mcar, 0.9) and (block, 0.9) use the standard `make_mask` unchanged;
the nominal-vs-actual coverage gap for block 0.9 is noted here.

Consequence at (block, 1.0): q_r=zero and q_r=linear are identical inputs
(linear fill with no valid positions falls back to zeros), so those two rows of
the grid are expected to coincide for every model.

## Envelope caveat (verified before running)

`train_s60.py:7-10` (the CPT recipe, regime "mechdiv", t45.augment verbatim):
"50% of windows clean; the rest get mechanism ~ U{mcar, block, mnar_high,
mnar_extreme}, **rate ~ U(0.05, 0.7)**". Rates 0.9 and 1.0 are therefore
EXTRAPOLATION for the cpt arm (and for ctl5k, which saw no missingness at
all). Absolute levels are recorded and compared against cpt's own rate-0.7
cells from `s70_rcprobe.json`.

## Pre-registered predictions (written 2026-09-03 before any pooled result)

- **P1b**: for the cpt arm at q_r=zero, clean context beats corrupt context by
  >= 0.1 relMSE pooled at rate >= 0.9, i.e. median(corrupt) − median(clean)
  >= 0.10 over the 54 strata (18 ds-ch x 3 mech-rate).
- **P2b**: the substitution ratio (q_c effect at q_r=zero) / (q_c effect at
  q_r=oracle) for the cpt arm is >= 2 in this regime, with the q_c effect
  defined as corrupt − clean pooled median relMSE. (Reported alongside the raw
  effects; if the oracle denominator is ~0 the ratio is flagged unstable.)
- **P3b**: vanilla remains far above cpt at (zero, .) overall, and vanilla's
  (zero,clean) vs (zero,corrupt) gap stays large (silent switching persists).
  Operationalized (before seeing results): vanilla/cpt pooled ratio >= 1.5 at
  both q_c values, and vanilla's corrupt − clean gap at q_r=zero >= 0.3 relMSE.
- **Envelope clause**: if the cpt arm collapses across the board at rate >= 0.9
  relative to its rate-0.7 cells, report the operating-envelope boundary as the
  finding instead of substitution.

## Pipeline check

Before the new cells: re-run S70's anchor cells (block 0.7, all 3 models,
ETTh1 = one dataset, 6 channels x 3 q_r x 2 q_c = 108 cells) through the S70B
code path and confirm each cell median matches `s70_rcprobe.json` within 5%.

## Pipeline check — PASS

`s70b_anchor.json`: S70's (block, 0.7) cells for ETTh1 (6 channels x 3 q_r x
2 q_c x 3 models = 108 cells) re-run through the S70B code path on 2026-09-03:
**108/108 within 5%** — in fact every cell matched `s70_rcprobe.json` at
+0.00% (bit-stable).

## Results

Full 972-cell table in `s70b_rcprobe.json` (per-window ratios stored per
cell); machine-readable verdicts in `s70b_verdicts.json`.

Pooled median relMSE over all 54 strata (rates 0.9/1.0 pooled):

| model | q_r | clean ctx | corrupt ctx | q_c effect |
|---|---|---|---|---|
| vanilla | zero | 8.122 | 8.511 | +0.388 |
| vanilla | linear | 2.188 | 2.119 | −0.068 |
| vanilla | oracle | 0.986 | 0.993 | +0.007 |
| ctl5k | zero | 5.957 | 5.979 | +0.022 |
| ctl5k | linear | 1.712 | 1.765 | +0.053 |
| ctl5k | oracle | 1.301 | 1.359 | +0.057 |
| cpt | zero | 1.872 | 1.753 | −0.119 |
| cpt | linear | 1.483 | 1.562 | +0.079 |
| cpt | oracle | 1.155 | 1.178 | +0.023 |

Per (mech, rate), clean / corrupt pooled medians:

| stratum | vanilla zero | vanilla linear | vanilla oracle | ctl5k zero | ctl5k linear | ctl5k oracle | cpt zero | cpt linear | cpt oracle |
|---|---|---|---|---|---|---|---|---|---|
| mcar 0.9 | 12.1 / 12.1 | 1.71 / 1.92 | 0.99 / 1.00 | 12.1 / 12.3 | 1.62 / 1.65 | 1.28 / 1.35 | 1.87 / 1.75 | 1.47 / 1.49 | 1.09 / 1.11 |
| block 0.9 | 3.78 / 5.12 | 1.52 / 1.53 | 0.99 / 0.99 | 1.87 / 1.88 | 1.20 / 1.22 | 1.07 / 1.09 | 1.16 / 1.18 | 1.13 / 1.13 | 1.09 / 1.10 |
| block 1.0 | 12.1 / 12.1 | 12.1 / 12.1 | 0.99 / 0.99 | 10.6 / 10.6 | 10.6 / 10.6 | 2.73 / 2.73 | 8.35 / 8.35 | 8.35 / 8.35 | **3.4e9 / 3.4e9** |

At (block, 1.0) the zero and linear rows coincide exactly (documented
degeneracy), and clean/corrupt coincide for every model: at full outage no
model's target forecast uses the covariates through this corner.

Envelope comparison (cpt pooled medians; s70 rate-0.7 vs s70b):

| stratum | zero clean/corrupt | linear clean/corrupt | oracle clean/corrupt |
|---|---|---|---|
| s70 mcar 0.7 | 1.15 / 1.18 | 1.04 / 1.07 | 1.01 / 1.04 |
| s70 block 0.7 | 1.09 / 1.08 | 1.08 / 1.09 | 1.07 / 1.07 |
| s70b mcar 0.9 | 1.87 / 1.75 | 1.47 / 1.49 | 1.09 / 1.11 |
| s70b block 0.9 | 1.16 / 1.18 | 1.14 / 1.13 | 1.09 / 1.10 |
| s70b block 1.0 | 8.35 / 8.35 | 8.35 / 8.35 | 3.4e9 / 3.4e9 |

## Verdicts

- **P1b: FAIL.** cpt's q_c effect at q_r=zero is −0.119 pooled (need >=
  +0.10); at rate 0.9 only it is −0.016. Clean context does NOT beat corrupt
  context for cpt even at long outages — cpt remains context-insensitive.
- **P2b: FAIL.** Substitution ratio −5.27 (eff0 −0.119 / effO +0.023); at rate
  0.9 only: eff0 −0.016, effO +0.011. There is no measurable context shift at
  any fill quality, so the ratio is noise-signed. Verdict robust to excluding
  the degenerate (block, 1.0) strata.
- **P3b: PASS.** vanilla/cpt at (zero, .): 4.34x clean, 4.86x corrupt pooled
  (weather 12.9x/17.2x, ETTh1 2.7x/2.8x, ETTm1 4.6x/4.9x). Vanilla's
  (zero,clean)→(zero,corrupt) gap: +0.388 pooled, +1.233 at rate 0.9 only,
  +1.34 at block 0.9 — silent switching persists and grows with outage length.
- **Envelope clause: NOT triggered as pre-registered** (no across-the-board
  collapse at rate >= 0.9 — block 0.9 sits at cpt's rate-0.7 levels), so the
  substitution question stands and fails on its own terms. But the boundary is
  real and appears at the (block, 1.0) corner, reported here as a finding:
  - cpt holds through rate 0.9 (block 0.9 with a garbage repair: 1.16, vs
    vanilla 3.78 and ctl5k 1.87; mcar 0.9 degrades ~1.6x vs 0.7 but stays
    6.5x better than vanilla).
  - At (block, 1.0) — never trained (recipe rates <= 0.7) — the declared
    full-outage corner breaks for cpt: zero/linear-declared collapses to an
    input-independent near-zero forecast (8.35), and **oracle-declared
    numerically explodes** (median relMSE ~3.4e9; forecasts |y| ~ 1e4–1e6 on
    O(10) data, verified finite, deterministic, covariate-independent).
    ctl5k on the IDENTICAL declared inputs stays finite (10.6 / 2.73), and
    vanilla-silent is sane (12.1 / 0.99). The explosion is therefore a
    cpt-weights property of the all-missing-flag + real-content corner, not
    a pipeline artifact: CPT's training taught it to trust flags over
    content, but only up to rate 0.7; at rate 1.0 the flag pattern is so far
    outside the recipe that the output head amplifies instead of ignoring.

**S70B takeaway**: the selective-trust signature does not extend to the
long-outage regime for the opposite reason S70 found at low rates — cpt never
learned to substitute context for repair because its training (rates <= 0.7,
no covariates) never required it. Its garbage-repair immunity (P3, P3b) is a
univariate-flag mechanism, not source selection. The operating envelope ends
at full outage, where the declared corner first goes input-independent, then
(oracle content) numerically unstable.

## Reproduction

```
cd tsfm_missing/experiments/s70_rcprobe
# pipeline check (must pass before the grid):
CUDA_VISIBLE_DEVICES=<idle> ../../../.venv/bin/python run_s70b_rcprobe.py --anchor-check --n-win 150
# grid (two shards, ~5-8 min each; written 2026-09-03 on idle-polled GPUs 0/2):
CUDA_VISIBLE_DEVICES=<a> python run_s70b_rcprobe.py --datasets weather --out s70b_shard_weather.json
CUDA_VISIBLE_DEVICES=<b> python run_s70b_rcprobe.py --datasets ETTh1,ETTm1 --out s70b_shard_ett.json
# merge shards into s70b_rcprobe.json (shard "cells"/"clean_mse" union), then:
python analyze_s70b.py    # writes s70b_verdicts.json
```

Shard logs: `s70b_weather.log`, `s70b_ett.log`; shard PIDs in
`s70b_probe.pid`. GPU discipline: the box was fully occupied by other
experiments at submission time; a poller waited for two cards showing < 2 GiB
(GPUs 0 and 2 became free ~25 min later) and only those were used.
