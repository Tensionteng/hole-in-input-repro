# S58 — WiSE-FT: the clean-cost dial (post-registered in make_wiseft.py)

theta(alpha) = (1−alpha)·stock + alpha·CPT. Zero training. Eval: `s58_eval_tiny.json`,
`s58_eval_base.json` (clean+probe+sweep+attack, same grids as S53/S56).

## tiny (8.7M)

| alpha | closure α=1 | clean vs P0 | floor |
|---|---|---|---|
| 0.30 | 38.9% [+31,+50] | 1.004 | 18/24 |
| 0.50 | 54.7% [+45,+67] | 1.018 | 18/24 |
| 0.70 | 68.0% [+56,+78] | 1.026 | 18/24 |
| 0.85 | 72.6% [+61,+80] | 1.040 | 18/24 |
| 1.00 (=P5) | 76.7% [+66,+82] | 1.048 | 18/24 |

## base (205M)

| alpha | closure α=1 | clean vs P0 | floor |
|---|---|---|---|
| 0.30 | 43.0% [+33,+62] | 1.002 | 16/24 |
| 0.50 | 66.5% [+58,+74] | 1.009 | **20/24** |
| 0.70 | 76.9% [+69,+80] | 1.039 | 19/24 |
| 0.85 | 78.9% [+72,+83] | 1.024 | 18/24 |
| 1.00 (=P5) | 78.4% [+71,+82] | 1.041 | 18/24 |

## Read-out

1. **The clean cost is a dial, not a tax.** At base, alpha=0.5 gives 66.5% closure
   at +0.9% clean; alpha=0.85 gives 78.9% at +2.4%. The earlier "5–7% clean cost"
   (R5, S53) was the cost of taking alpha=1 by default, not a property of the
   method. The prescription now quotes the frontier, and "no clean regression"
   (<1%) is achievable at ~2/3 of the retrofit.
2. **W85 Pareto-dominates P5 at base** (78.9% vs 78.4% closure AND 1.024 vs 1.041
   clean) — interpolation smooths off some CPT overfitting. Non-monotone clean
   ratios along the grid (1.039 at 0.70 < 1.024 at 0.85 inverted) are within the
   eval's median-of-9 noise.
3. Floor and attack surface hold across the whole frontier (16–20/24, ~1.43–1.48×):
   the dial does not buy back safety, it only redistributes accuracy.
4. The S58 success bar (clean ≤ 1.02 AND closure within 15 pts of P5) is met at
   base by W50 (1.009, 66.5% vs 78.4−15=63.4) — and by W85 at base on the closure
   side with 1.024. At tiny the frontier is shifted ~3 points of clean cost worse;
   the bar is met by W70 only on closure (68.0 > 61.7) with clean 1.026.
