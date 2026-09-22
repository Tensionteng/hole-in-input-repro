# S47 (stage 2) — the recipe ablation ladder: which ingredients of the prescription carry the weight?

**Status: DESIGN (pre-registered before any training). Requires stage 1 (S45) to confirm
P2/P4 first; if stage 1 falsifies the factorial, this design is void.**

## Question

Stage 1 asks "does interface × mechanism-diversity matter" (the causal claim). Stage 2 asks
the narrower, builder-facing question: **among the recipes on the table, which is best, and
which ingredient carries the weight?** The candidates are the ones the 2025 literature
actually uses (single scattered patch mask, CPM block mask) and the two halves of our
prescription separated (mechanism diversity vs fill-quality diversity).

## Arms (same architecture, corpus, budget, sampler as S45; interface = restored throughout)

| arm | augmentation | reads as |
|---|---|---|
| F | CPM contiguous block-mask (TiRex/Toto/FlowState recipe: run length ~ U(1,5) patches, p ~ U(0,0.25)) | the 2025 consensus recipe |
| G | mechanism-diverse, **declare-only** (holes always zeroed + flag, no alpha-blend) | is fill-quality diversity needed? |
| H | **fill-diverse only** (alpha-blend, single mechanism = block) | is mechanism diversity needed? |
| (reference) C, E from S45 | mechanism-diverse / single scattered patch mask | already trained, no new cost |
| winner@small | the winning recipe at bolt-small (48M) | scale invariance of the recipe |

Arms F/G/H: bolt-tiny, 15k steps, batch 1024, fp32, identical to S45 in every other detail.

## Pre-registered predictions

- **Q1**: on the median alpha=1 closure and the alpha=0 floor, C > E and C > F. (Mechanism
  diversity beats both single-mechanism recipes.)
- **Q2**: G ~= C at alpha=0 (declare-only suffices at the floor) but G < C at alpha >= 0.5
  (fill diversity pays when the fill is good).
- **Q3**: H < C under block outages (diversity is what handles long gaps), H ~= C under
  scattered dropout.
- **Q4**: the winner's median alpha=1 closure at bolt-small is within 15 points of
  bolt-tiny's.
- **Q5 (safety)**: none of F/G/H beats C on the full-grid median closure. A falsification
  here weakens the mechanism-diversity claim to "the best recipe is F/G/H", and the paper
  reports that instead.

## Evaluation and anti-overfitting discipline

1. Same harness as S45 (eval_s45.py), same grid (4 mech x 2 rates x 3 datasets + 9-dataset
   clean check), arms read against the same stored A/C/E cells.
2. **Second axis**: the winner is re-scored on real missingness (Penmanshiel curtailment and
   METR-LA outages, S26/S29 windows) before any "best recipe" sentence is written; a recipe
   that wins on the synthetic grid and loses on real missingness is reported as such.
3. Clean accuracy must not regress vs arm A by more than 5% (the prescription cannot be
   bought with clean performance).
4. Every arm is reported, win or lose, in the paper's prereg scorecard.

## Gates

- G1/G2 as in S45 (stock bolt-tiny probe reproduction; encode replica exactness).
- The A/C/E cells are re-read from s45_eval.json, not re-run: stage 2 adds only F/G/H/small.

## Deliverables

`train_s45.py --arm {F,G,H}` (regimes added), eval results, `s47_notes.md` with the Q1-Q5
scorecard, and a short paper section ("which ingredients of the recipe matter") that the
builder prescription in the conclusion cites.
