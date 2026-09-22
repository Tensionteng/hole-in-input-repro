# S59 — the any-fill matrix, and the zero-fill blindness it exposed

Grid: 9 benchmarks × 4 mechanisms × rate 0.7 × fills {zero, ffill, linear, oracle}
on the declared path (eval_s45.py `--parts sweep9fills`). Data: `s59_fills_tiny.json`,
`s59_fills_base.json`, `s59_fills_C.json`.

## The matrix (tiny; base is qualitatively identical, worse in magnitude)

| fill | P5 vs P0 (9 datasets) |
|---|---|
| oracle (α=1) | **9/9 win** |
| linear | **9/9 win** |
| ffill | 7 win, 2 tie |
| **zero** | **0/9 — LOSES ALL, 3×–19× worse (tiny); up to 11× worse at base** |

## Diagnosis (confirmed, not hypothesised)

The H recipe (block_filldiv) alpha-blends between linear extrapolation and truth at
masked positions — the content at a flagged position is NEVER zero during training.
Zero fill (the most common deployment fill, `fillna(0)`) is therefore
out-of-distribution for P5. The C arm, whose mechdiv regime includes a declare-only
branch (holes zeroed + flag), shows NO zero-fill blindness: 1.03–1.77 across all 9
datasets (`s59_fills_C.json`). Cause and cure are both pinned to the recipe, not the
interface.

## Fix: P6 = H + declaration endpoint

`block_declblend` regime (train_s45.py): block outages, 50% alpha-blend, 50%
declared-zero, keeping H's single mechanism (the thing that made H scale-stable in
S48) while covering the declaration endpoint (the thing C had and H lacked). P6
retrains via train_s53.py at tiny and base; re-verification on this matrix and on
the S55 real-missingness windows follows. Notes will record whether the fix holds
and whether it costs closure on the oracle column.

## Status

PENDING P6 eval. The "never worse under any fill" claim is BLOCKED on P6; with P5
it is false (zero column).
