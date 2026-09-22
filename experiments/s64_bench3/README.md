# S64 — benchmark evaluation of the three retrofit families (Tables 3 and 4)

Completes the paper's benchmark evaluation: the Moirai-2.0 and Chronos-2 retrofits
under the EXACT Table-3 (missingness leaderboard) protocol, and both retrofits under
the Table-4 (GIFT-Eval dose-response) protocol. Write-only directory.

## Scripts and artifacts

- `run_s64_leaderboard.py` → `s64_leaderboard_cells.json` (+ `.log`)
  Raw cells: DS9 x 4 mechanisms x rate 0.7 x fills {zero, linear, oracle (=perfect)}
  x arms {m2-stock, m2-w50, c2-stock, c2-w50}, each on its own declared path, median
  paired per-window relMSE vs the arm's own clean-context error; per-window ratios
  kept (`w`). Mirrors `s45_pretrain/eval_s57_leaderboard.py` line for line
  (s35.load(ds, 150), s25.build_eval_batch, s25.fill_context, ok = base > 1e-12).
- `aggregate_s64_leaderboard.py` → `s64_leaderboard_rows.json`
  Paper-format rows + reproduction gates G1-G4.
- `run_s64_gift.py` → `s64_gift_dose.json` + `s64_gift_scores.npz` (+ `.log`)
  The s32c pipeline verbatim (`run_s32c_stratified.collect("random", 6, 250)`,
  `s32.fill`, `s32.mase`): 7,725 pooled windows, per-window MASE vs in-window
  naive-1 on observed target points, 4 fills x 4 models, stratified by the window's
  own context-NaN fraction.
- `aggregate_s64_gift.py` → `s64_gift_report.json`
  Table-4-format strata (linear fill) + stock reproduction gates.

## Table-3 row aggregation (recovered from the paper's artifacts)

`tab_leaderboard.tex` rows are NOT the flat median over the 36 (ds x mech) cells.
The printed rows for bolt-base stock (1.51/1.57/3.51, worst 46.5), Q50-base
(1.22/1.51/2.26, 7.2) and Moirai 2.0 stock (1.18/1.60/11.56, 82.2) are reproduced
to the printed decimals by:

    per-dataset cell = median over the 4 mechanism cells (of per-(ds,mech) medians)
    row value        = median over the 9 per-dataset cells
    worst zero cell  = max over the 9 per-dataset zero cells

The paper's Chronos-2 row (1.61/1.57/2.03, 31.0; W50 1.26/1.34/1.81, 79.3) was
instead computed as the flat median over the 36 ds x mech cells
(verify_main_text.py `_lb`) -- an aggregation inconsistency in the paper's table.
The s60 run's CELLS are the s57 protocol's cells (gate G3): same windows, masks,
fills, declared paths, own-clean relMSE. Only the row aggregation differed. S64
reports all four arms under the Table-3 (per-dataset) aggregation so the families
are comparable, and keeps the flat-36 numbers alongside for continuity with the
printed C2 row.

## Gates (all PASS after the TF32 fix)

- G1: m2-stock's 108 strict-fp32 cells vs `s45_pretrain/s57_moirai2.json` — max rel
  dev 0.32%, which is exactly the TF32 difference: s57_moirai2.json was computed
  with TF32 ENABLED (eval_s57 imports eval_s45 -> train_s45, whose module top sets
  allow_tf32=True).
- G1b: the same script run with TF32 on (matching s57's effective environment) +
  batch=64 reproduces all 108 cells BIT-EXACT (max rel dev 0.0; cells saved as
  `s64_m2stock_tf32on_cells.json`). Moirai-2.0 IS TF32-sensitive at the per-cell
  level (~0.3%), unlike what a first batch-64 check suggested.
- G2: m2-stock row = 1.18/1.60/11.56, worst 82.2 — the paper's Moirai 2.0 row,
  reproduced at the paper's print precision under strict fp32 (strict value
  11.5556 vs TF32 11.5585; worst 82.18 vs 82.20).
- G3: c2-stock strict-fp32 cells vs s60 artifacts (zero/linear vs
  s60_leaderboard.json, perfect vs s60_grid.json alpha=1.0 declared) — max rel dev
  0.0 (bit-exact): the s60 protocol IS the s57 protocol cell-for-cell.
- G4: c2-stock row under the flat-36 aggregation = 1.61/1.57/2.03, worst 31.0 —
  reproduces the paper's printed C2 row (and G4b: W50 = 1.26/1.34/1.81, 79.3).
- GIFT: c2-stock and m2-stock bin medians reproduce s32c_results.json's
  chronos2|random and moirai2|random rows exactly (all four fills, n=7,725 pool,
  identical stratum counts).

## The TF32 import trap (why there are two leaderboard runs)

`train_s45.py:30-31` sets `torch.backends.*.allow_tf32 = True` at MODULE level, so
`import eval_s45` (which imports train_s45) silently re-enables TF32 after an eval
script has disabled it. The first s64 leaderboard run imported eval_s45 for the
DS9/MECHS constants and computed all four arms with TF32 matmuls (per-window
relMSE off by up to ~0.5% for Chronos-2, ~0.3% for Moirai-2.0, caught because
c2-stock no longer matched s60_grid's stored per-window ratios — strict fp32
matches bit-exact). Note the same trap fired when the s57 artifacts were produced:
eval_s57_leaderboard.py imports eval_s45, so s57_moirai2.json (and the bolt rows
of Table 3) are TF32-on numbers; the s60/s61 artifacts are strict fp32 (their eval
scripts set the flags and never import train_s45). The s64 headline numbers are
strict fp32, matching the s60/s61 standard; at the paper's print precision the
Moirai stock row is identical under both (11.56 vs 11.56, worst 82.2). Fix: define
DS9/MECHS locally, never import eval_s45, assert the flag at runtime. The GIFT
scripts never imported eval_s45 and were unaffected (stock gates exact on the
first pass).

## Fill routing (Table 4)

Following s32.forecast + the Q50Bolt precedent: stock arms take each fixed fill
through the PLAIN path (content, all-ones flag; `nan` = the stock NaN path);
retrofit arms take fixed fills through their DECLARED path (content + true flag)
and `nan` as their declaration endpoint (zeroed content + flag = 0). Table 4
reports the linear fill.

## Devices / seeds

All runs: cuda:0, seed 20250810 (windows/masks), strict fp32 (TF32 off), n_win=150
(leaderboard), s32c pool seed 1 (GIFT). Inference coexisted with the s62 TempoPFN
probe on cuda:0 without OOM.

## Results (all from the saved JSONs; strict fp32)

### Table-3-protocol rows (per-dataset cells -> median; worst = max ds zero cell)

| arm | perfect | linear | zero | worst zero cell |
|---|---|---|---|---|
| m2-stock | 1.18 | 1.60 | 11.56 | 82.2 (electricity) |
| m2-w50 (m2-retrofit) | 1.11 | 1.49 | 2.96 | 21.7 (electricity) |
| c2-stock | 1.54 | 1.52 | 2.10 | 6.0 (electricity) |
| c2-w50 (c2-retrofit) | 1.29 | 1.44 | 1.66 | 8.9 (electricity) |
| bolt-base stock [s57 artifact] | 1.51 | 1.57 | 3.51 | 46.5 (exchange) |
| bolt-base Q50 retrofit [s57 artifact] | 1.22 | 1.51 | 2.26 | 7.2 (exchange) |

Flat-36 aggregation (the paper's printed C2 row; kept for continuity):
c2-stock 1.61/1.57/2.03 (31.0), c2-w50 1.26/1.34/1.81 (79.3),
m2-stock 1.18/1.68/8.97 (248.9), m2-w50 1.12/1.50/2.79 (48.7).

### Table-4-protocol dose-response (linear fill, random windows, median MASE)

Chronos-2 (n = 4,308 / 2,327 / 625 / 208 / 257):

| stratum | stock | retrofit | Delta |
|---|---|---|---|
| <1% | 0.629 | 0.627 | -0.4% |
| 1-5% | 0.860 | 0.860 | -0.0% |
| 5-15% | 1.445 | 1.373 | -5.0% |
| 15-30% | 2.020 | 1.694 | -16.1% |
| >30% | 1.822 | 1.495 | -17.9% |

Moirai 2.0 (same strata and counts):

| stratum | stock | retrofit | Delta |
|---|---|---|---|
| <1% | 0.633 | 0.641 | +1.3% |
| 1-5% | 0.892 | 0.873 | -2.1% |
| 5-15% | 1.829 | 1.745 | -4.6% |
| 15-30% | 2.406 | 2.068 | -14.1% |
| >30% | 1.938 | 1.795 | -7.4% |

Same crossover shape as the bolt family's printed Table 4: a small clean cost in
the lowest stratum, growing gains once the window is actually missing.

### Wall-clock (cuda:0, coexisting with the s62 probe)

Leaderboard (4 arms, 9 ds x 4 mechs x 3 fills, 150 win): m2-stock 51s, m2-w50 46s,
c2-stock 170s, c2-w50 171s (~7.5 min total; plus one discarded 6.5-min TF32 run).
GIFT (pool collection ~3 min; 4 models x 4 fills over 7,725 windows): c2-stock 55s,
c2-w50 49s, m2-stock 9s, m2-w50 8s (~5 min total). Aggregation < 1s each.
