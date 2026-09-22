# S512 — the retrofit's content-reading, visible in the attention maps

Setup: encoder self-attention over patches (bolt-tiny, patch 16, L=512 → 32
patches + reg token), stock (P0) vs retrofit (P5), same windows, block mechanism at
rate 0.5, 20 windows × 3 datasets, at the two ends of the fill-quality axis.
Quantity: attention share received by masked patches ÷ uniform share (median).
`s512_attn.json`, `eval_s512_attn.py`.

## Results (median share ratio)

| dataset | P0, α=0 | P0, α=1 | P5, α=0 | P5, α=1 |
|---|---|---|---|
| ETTh1 | 0.804 | 0.808 | 0.832 | **0.859** |
| ETTm1 | 0.816 | 0.818 | 0.849 | **0.863** |
| weather | 0.832 | 0.833 | 0.872 | **0.881** |

## Read-out

1. **Stock's attention is flat across the fill-quality axis** (Δ ≤ 0.004 everywhere)
   — the attention-level counterpart of the permutation probe's ρ ≈ 0. The ceiling
   is visible inside the mechanism, not only in the outputs.
2. **P5 attends to masked patches MORE when the fill is good** (α=1 > α=0 on all
   three datasets, +0.027/+0.014/+0.009) and less when it is poor — the mechanism
   has learned to gate content by declared quality, exactly the behaviour the
   alpha-blend recipe was meant to teach.
3. Both models attend to masked patches slightly LESS than uniform (< 1.0): the
   flag-down weighting is learned even by stock (flag semantics survive pretraining)
   — the difference the retrofit makes is CONTENT-conditional attention, not
   attention per se. Consistent with S53's P1 finding (the flag perturbs, the
   content is what must be learned to use).
