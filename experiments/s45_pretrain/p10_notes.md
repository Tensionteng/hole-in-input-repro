# P10 — clean-anchor distillation: negative result

Setup: mechdiv CPT (the P8 recipe) + a distillation pull toward the frozen stock
model's quantiles on CLEAN windows only (window-scaled L1, lambda=10), 5k steps,
both sizes. Rationale: anchor clean behaviour to stock while leaving missingness
learning unconstrained.

## Results (reference = stock P0)

| size | closure | clean | attack |
|---|---|---|---|
| tiny | 33.5% | 1.110 | 1.19× |
| base | 45.9% | 1.038 | 1.22× |

vs Q50 (the WiSE-FT arm): closure 62.3%, clean 1.058.

## Verdict: negative, abandoned

The clean anchor suppresses missingness learning through shared weights (closure
roughly halves at both sizes) while NOT fully restoring clean accuracy (1.038–1.110,
never ≤1.02). Combined with Q30's demonstration that WiSE-FT cannot undo
distribution-specific overfitting either (m4_hourly +30% at alpha=0.3), the
conclusion is that **the ~3–5% clean cost of the retrofit is a property of
continued training itself, not of the missingness recipe** — P4 (same 5k steps,
no missingness) already pays 3.8–12.9%. The honest paper position is therefore
the dose-response framing (cost is flat, payoff grows with missing rate,
crossover ~3%) plus a deployment routing rule, not "zero cost".
