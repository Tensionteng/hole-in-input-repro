# S53 (stage 3) — CPT-from-stock: can a shipped checkpoint be retrofitted?

**Status: DESIGN (pre-registered before any training). Requires stage 1/2 (S45/S47)
findings; the causal claim does not depend on this stage, but the builder prescription does.**

## Question

Stages 1–2 prove the recipe causally on models we trained ourselves. A builder already
holding a stock checkpoint will not retrain from scratch, so stage 3 asks the
retrofit question: **can a shipped checkpoint acquire missingness awareness through
continued pretraining (CPT) with the restored interface, at no clean-accuracy cost?**
A positive answer upgrades the prescription from "train it right" to "fix what shipped".

## Arms

All bolt-tiny, **all initialised from the stock chronos-bolt-tiny checkpoint**; only the
interface and the continued-training regime vary.

| arm | stage 1 | stage 2 | reads as |
|---|---|---|---|
| P0 | — | — | stock, untouched (reference) |
| P1 | — | — | P0 weights read through the dual interface, zero training (interface break alone) |
| P2 | input_patch_embedding only, 600 steps bs48 lr 1e-4, mechdiv (the S27 adapter recipe) | — | is the cheap adapter enough? |
| P3 | same adapter as P2 | full model, 5k steps bs1024 lr 1e-4, block_filldiv (the H recipe) | the two-stage prescription |
| P4 | — | full model, 5k steps bs1024 lr 1e-4, filtered, **native** interface | budget control: same extra FLOPs, no missingness |
| P5 | — | full model, 5k steps bs1024 lr 1e-4, block_filldiv, dual | is stage 1 needed at all? |

- Corpus, sampler, augmentation, pinball loss and evaluation harness identical to
  S45/S47; the only differences vs stages 1–2 are the initialisation (stock instead of
  random) and the budget (5k instead of 15k steps — a retrofit must be cheap to be
  worth reporting). Seed 20260826 for every arm; two replicate seeds follow if R1/R2
  hold in direction.
- P2/P3 stage 1 mirrors S27's adapter (600 steps, bs 48, lr 1e-4, cosine, embedding
  only, mechanism-diverse missingness with alpha-blend), but the windows come from the
  S45 pretraining corpus so that P2 vs P3 differ ONLY in stage 2, and so that no
  benchmark dataset leaks into any arm.
- P0's checkpoint is a plain dump of the stock state dict; P1 shares P0's weights (a
  symlink) and differs only in the interface used at evaluation.

## Pre-registered predictions (reference = P0 throughout, metrics as in analyze_s48.py)

- **R1 (the retrofit works)**: P3 rho_declared ≥ 0.5 and median alpha=1 closure ≥ 50%
  of P0's untouchable excess, on the same 3-dataset × 4-mechanism × 2-rate grid.
- **R2 (the cause is the regime, not the extra FLOPs)**: P4 stays flat —
  rho_declared ≈ 0, closure ≈ 0.
- **R3 (the adapter alone is not enough)**: P2 < P3 in median closure — the S45
  prediction that pretraining beats the 2-minute adapter generalises to CPT.
- **R4 (stage 1 matters, or it doesn't)**: P3 vs P5 reported openly; if P5 ≈ P3 the
  prescription drops the adapter stage and says so.
- **R5 (no clean cost)**: P3 clean mse_median within ±5% of P0 (median ratio over the
  9 benchmark datasets).
- **R6 (the trade-off returns, honestly)**: P3's declared path reopens the attack
  surface relative to P0 (attack ratio > P0's ≈1.05); reported, not hidden.

Falsification: if R1 fails at 5k steps we say so and report a budget-doubling arm
(10k) labelled as such, not as a silent fix. Every arm is reported, win or lose, in
the paper's prereg scorecard.

## Gates

- G2′: with STOCK weights, the encode replica at plain matches the stock forward to
  max|diff| < 1e-3 (catches any load/config drift before training).
- G1 (stored stock probe reproduction) as in eval_s45.py, before any arm is scored.

## Deliverables

`train_s53.py`, four trained checkpoints (P2/P3/P4/P5) plus the P0 dump and P1 symlink
in `s45_ckpt/`, `s53_eval.json` via eval_s45.py, `s53_notes.md` with the R1–R6
scorecard, and a paper subsection ("retrofitting a shipped checkpoint") that the
builder prescription cites.
