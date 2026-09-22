# S71 — the boundary, measured: CPT attempts on the non-restorable conventions

## Question

Section 8 of the paper asserts: the overwriting (TimesFM), learned-token (TempoPFN) and
no-declared-path (Timer-XL) conventions would need an input channel *added*, not restored —
architecture work, not a post-hoc fix. That claim currently rests on source-level reading
of their input handling. S71 turns it into a measurement: give each boundary model the
same 5,000-step missingness-diverse CPT that fixed bolt/c2/m2, through whatever input path
the model natively exposes, WITHOUT adding channels, and see whether any fill-reading
emerges.

## Models and feeding conventions

For each model, first establish from the project's own audit code how a fill can be
presented at all (the s29/s30 cross-model harness and the s57 leaderboard runs encode the
per-model conventions; s62 has the TempoPFN path):

- **TempoPFN** (learned-token convention; `models_local/tempopfn`; s62 pipeline).
- **Timer** (`models_local/timer-base-84m`; s57 ran a "timerxl" row — resolve EXACTLY
  which checkpoint and label the paper's leaderboard used, and document it; the boundary
  sentence's "Timer-XL" must refer to what was actually evaluated).
- **TimesFM** (overwriting convention; NOT in models_local as of 2026-09-03 — s57 ran it
  somehow; find that path. If a runnable torch checkpoint can be fetched the same way s57
  did (HF mirror), include it; if not, document the failure and carry the other two).

For each model, document precisely how missing positions and fills are presented during
both CPT and eval. If a model cannot ingest fill content at all through its native path
(e.g. it overwrites filled values internally), that is itself the measurement for its
row; still run the CPT to show post-CPT behaviour is unchanged.

## Arms (per model)

1. `stock` — no training (evaluation only).
2. `cpt` — 5,000 steps on the same mechanism-diverse missingness corpus recipe used for
   the bolt retrofit (reuse the s54/P5 corpus and augmentation code; fills presented
   through the model's NATIVE path, alpha-blend quality spectrum included).
3. `ctl5k` — 5,000 steps on the same corpus with missingness filtered out (budget
   control).

Same optimizer family the model was shipped with where practical (AdamW, lr 1e-4 with
500-step warmup, cosine to 0; bs as memory allows, record it). Two seeds for `cpt` if
time allows. bolt's own P0/P4/P5 rows from s53/s54 serve as the reference column — no new
bolt training.

## Evaluation (per model, per arm)

The paper's own harness, adapted per model from the s57/s30 conventions:

- **probe**: fill-permutation sensitivity rho under the model's native missingness
  convention (s30 protocol), 3 datasets x 3 mechs.
- **sweep**: alpha-blend fill-quality sweep, alpha in {0, .25, .5, .75, 1}, block + mcar
  at rate 0.7, ETTh1/ETTm1/weather, 150 windows, relMSE vs the arm's own clean.
- **clean**: clean-context MSE on the same datasets (cost check).
- **leaderboard row**: the s57 missingness leaderboard grid (9 benchmarks x 4 mechanisms,
  rate 0.7, {zero, linear, oracle} fills) for the post-CPT checkpoint.

## Pre-registered predictions

- **Q1**: no boundary model gains fill-reading from plain CPT: post-CPT rho <= 0.1 and
  alpha-sweep closure <= 10% (reference: bolt CPT rho = 0.88).
- **Q2**: the ctl5k arms are unchanged from stock (rho within 0.05 of stock).
- **Q3 (falsification clause)**: if any boundary model DOES gain fill-reading without
  architecture change (rho > 0.3 AND closure > 25%), the boundary claim is wrong for that
  model — report it, and the paper sentence gets revised to name only the models where
  the claim survives.

## Gates

- **G0**: reproduce each model's own s57 leaderboard stock numbers (within 5% on the
  shared cells) before training anything. If a model's stock row cannot be reproduced,
  drop that model and document why.
- **G1**: CPT runs must show decreasing train loss; a diverged/flat run is a plumbing
  bug to fix, not a result.

## Deliverables

- `run_s71_<model>.py` (CPT per model), `eval_s71.py`, `s71_<model>.json`,
  `s71_notes.md` with the per-model verdict table and the conventions documentation.
- GPUs: idle-poller discipline — only take cards showing < 2 GiB in use (several
  experiments are running concurrently on this box). Expect 6 CPT runs x 1-3 h plus
  evals.
