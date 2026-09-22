# S70 — R/C selective-trust probe: notes

Date: 2026-09-03. Inference only; no training. GPUs 4 (weather shard) and 5
(ETTh1+ETTm1 shard). Strict fp32 (TF32 off), matching the s60/s65 eval
convention.

## What was run

The full pre-registered grid: 3 datasets (ETTh1, ETTm1, weather) x 6 target
channels (S31's exact channel draw, reproduced bit-for-bit) x {mcar, block} x
rate {0.3, 0.7} x q_r {zero, linear, oracle} x q_c {clean, corrupt} x 3 models,
150 windows per cell, paired per-window relMSE vs the model's OWN clean
univariate forecast, median per cell (S31 convention). Neighbours: K=4 most
target-correlated channels, correlation on the first 80% of the timeline only;
supplied as `past_covariates` (vanilla) / as extra same-group context rows
(dual arms) — the pipeline stacks past covariates into the target's group, and
the dual replica does the same via `group_ids`.

**Cell-count note (deviation in the DESIGN's arithmetic, not in coverage):**
DESIGN says "3 x 6 x 2 x 2 x 3 x 2 x 3 = 2592 cells", but that product is
**1296**. All enumerated factor combinations were run — 1296 cells, n=150
windows each. Nothing was dropped; the DESIGN's total is simply miscomputed.

## Checkpoints (and why)

| model | checkpoint | source |
|---|---|---|
| vanilla | `models_local/chronos-2` (stock, fp32) | S31's model |
| ctl5k | `s65_fairbudget/s65_ckpt/c2_nomiss.pt` | the paper's c2 **+5k fair-budget control**: 5,000 CPT steps on the no-missingness filtered regime, dual interface (`s65_ckpt/c2_nomiss_log.json`: steps=5000, regime=filtered; `eval_s65_c2.py` loads this file as arm C2CTL for `s65_c2_leaderboard.json`) |
| cpt | `s60_c2retrofit/s60_ckpt/arm_CPT_s20260903.pt` | the paper's current c2 **CPT retrofit**: seed 20260903, 5,000 steps, dual interface. `eval_s60.py --suffix _s20260903` maps this file to the CPT arm of `s60_leaderboard_s20260903.json`; the eval log (`s60_eval_s20260903.log`) numbers match that JSON (e.g. clean ETTh1 CPT 1.237) |

The undated `arm_CPT.pt` (seed 20260901) and `arm_CPT_s20260902.pt` are earlier
seed replicas of the same recipe; the dated s20260903 set is the current
leaderboard candidate per the DESIGN and was verified against the leaderboard
JSON before running.

## Conventions (the load-bearing choices)

Following S69's arena semantics ("native arms receive fills silently; dual arms
receive them declared"):

- **vanilla**: all fills SILENT (plain convention: content present, positions
  unmarked), through the S31 `Chronos2Pipeline` code path. The native
  interface cannot carry a content-preserving declaration, so q_r enters only
  as context content. Consequence: vanilla at (linear, clean) IS S31 condition
  (c) — verified bit-exact (max deviation 0.0000 over all 72 strata).
- **ctl5k / cpt**: fills DECLARED through the restored dual endpoint (content +
  per-position flag), for the target repair AND for corrupt neighbours. The
  DESIGN specifies declaration only for the target fill ("declared through its
  restored endpoint"); declaring the neighbour corruption as well follows
  S69's dual-arm q_c semantics. Vanilla's corrupt context is silent
  (zero-filled, unmarked).
- Corrupt neighbours: holed at the same rate/mechanism as the target with
  INDEPENDENT masks (`make_mask(..., mask_seed=1)`; the DESIGN says "same
  rate/mechanism" and does not pin positions), zero-filled.
- Dual-arm forward: `forward_grouped()` in `run_s70_rcprobe.py` —
  `c2_iface.forward_iface` with `group_ids` passed to the encoder (the only
  change). With singleton groups it reduces to the S60-gated replica; the
  stock pipeline's all-NaN `future_covariates` for past-only covariates is
  op-for-op identical to `future_covariates=None`, so the grouped replica
  matches the pipeline path except for the removed zeroing line.
- n_win=150 as pre-registered (no subsampling needed; full grid took ~6
  minutes per shard, far under the 12 h budget).

## Gate G0 — PASS (run before any new cell)

Reproduce S31's stored anchors, conditions (c) `holed_multi` and (d)
`nan_multi` at block 0.7, within 10%: **all 18 (ds, ch) pairs reproduced with
0.0% deviation on both conditions** (36/36 values identical at printed
precision; strict fp32 reproduces S31's numbers exactly). Verdict: PASS, no
pipeline drift. Per-cell numbers in `s70_rcprobe.json["g0"]`.

## Results

Pooled median relMSE over the 72 (ds, ch, mech, rate) strata
(full 1296-cell table in `s70_rcprobe.json`; per-window ratios stored per cell):

| model | q_r | clean ctx | corrupt ctx |
|---|---|---|---|
| vanilla | zero | 1.913 | 1.984 |
| vanilla | linear | 1.084 | 1.149 |
| vanilla | oracle | 0.986 | 1.006 |
| ctl5k | zero | 1.629 | 1.580 |
| ctl5k | linear | 1.044 | 1.061 |
| ctl5k | oracle | 1.029 | 1.040 |
| cpt | zero | 1.063 | 1.058 |
| cpt | linear | 1.021 | 1.026 |
| cpt | oracle | 1.009 | 1.016 |

Per dataset (clean context; the P3 contrast):

| dataset | vanilla zero/oracle | ctl5k zero/oracle | cpt zero/oracle |
|---|---|---|---|
| weather | 2.483 / 0.954 = 2.60 | 1.638 / 1.015 = 1.61 | 1.027 / 0.975 = 1.05 |
| ETTh1 | 1.544 / 0.998 = 1.55 | 1.377 / 1.034 = 1.33 | 1.053 / 1.038 = 1.01 |
| ETTm1 | 2.056 / 0.991 = 2.08 | 1.826 / 1.072 = 1.70 | 1.090 / 1.013 = 1.08 |

q_c effect (clean − corrupt, relMSE points) by (mech, rate):

| stratum | vanilla zero / oracle | ctl5k zero / oracle | cpt zero / oracle |
|---|---|---|---|
| mcar 0.3 | −0.029 / −0.028 | −0.008 / −0.026 | +0.002 / −0.009 |
| mcar 0.7 | +0.042 / −0.011 | −0.282 / −0.029 | −0.022 / −0.030 |
| block 0.3 | −0.169 / −0.022 | +0.007 / +0.003 | −0.003 / −0.003 |
| block 0.7 | **−0.672** / −0.016 | +0.052 / −0.016 | +0.004 / +0.000 |

## Verdicts on the pre-registered predictions

- **P1 (vanilla flat in q_r, <5%): FAIL.** Vanilla varies by 70% across q_r
  (pooled spread 0.698 clean / 0.709 corrupt context; stratum-median spread
  0.717). Fed silently, vanilla eats the fill content: zero fill at 70%
  missing costs ~2x relMSE, oracle fill returns to ~1.0. The pre-registered
  rationale ("its interface discards fill content") does not apply to the
  silent-feed convention the DESIGN specifies for vanilla — this mirrors
  S69's arena P1, where native arms' MSE varies with q_r even though their
  per-window trust weight cannot.
- **P2 (substitution, CPT ratio ≥ 2, others ~1): FAIL as pre-registered.**
  Pooled q_c effects at q_r=zero vs oracle: vanilla −0.071/−0.020 (ratio 3.6),
  ctl5k +0.049/−0.011, cpt +0.005/−0.007 (ratio −0.7). CPT shows **no**
  measurable context sensitivity at any q_r (all effects within ±0.03 relMSE
  points), so the ratio is noise-dominated. The large ratio belongs to
  VANILLA, concentrated at block 0.7 (−0.672 at zero fill vs −0.016 at
  oracle): vanilla borrows silently and is hurt by context corruption exactly
  when the repair is garbage. CPT does not need the context at these rates —
  its declared-flag robustness keeps it at ~1.01–1.06 everywhere, so the
  selective-shift signature has no room to appear in relMSE.
- **P3 (garbage repair ignored): PASS.** cpt at (zero, clean) = 1.063 vs
  (oracle, clean) = 1.009 → ratio 1.054, within 10%. Vanilla: 1.913 vs 0.986 →
  1.940, far outside 10% (it eats the garbage). Control: 1.584 — the
  no-missingness control is NOT immune to the garbage fill, so the
  garbage-ignoring is attributable to the missingness-diverse CPT training,
  not to the interface or the step budget alone. Per-dataset cpt ratios:
  1.05 / 1.01 / 1.08 — all within 10%.

Summary: the selective-trust signature survives on the real foundation model
in its clearest form — the CPT retrofit ignores a garbage repair (P3, all
three datasets, and the control separates the mechanism from the budget) —
but the context-substitution half (P2) does not materialize in relMSE,
because CPT is barely context-dependent here; the silent borrower (vanilla)
shows the largest context sensitivity instead. P1's failure is itself the
S69-consistent result: fill content moves the native model, declaration or
not.

## Reproduction

```
cd tsfm_missing/experiments/s70_rcprobe
# gate (must pass before the grid):
CUDA_VISIBLE_DEVICES=4 ../../../.venv/bin/python run_s70_rcprobe.py --g0-only --n-win 150
# grid (two shards; each ~6 min):
CUDA_VISIBLE_DEVICES=4 python run_s70_rcprobe.py --datasets weather --out s70_shard_weather.json
CUDA_VISIBLE_DEVICES=5 python run_s70_rcprobe.py --datasets ETTh1,ETTm1 --out s70_shard_ett.json
# merge shards into s70_rcprobe.json, then:
python analyze_s70.py    # writes s70_verdicts.json
```

Run PIDs were recorded in `s70_probe.pid`; shard logs: `s70_weather.log`,
`s70_ett.log`.
