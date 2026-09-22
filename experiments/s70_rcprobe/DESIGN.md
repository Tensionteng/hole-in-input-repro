# S70 — the R/C probe on a real foundation model (external validity for S69)

## Question

Does the Chronos-2 CPT retrofit show the selective-trust signature on real multivariate
data — leaning on the context when the target repair is garbage, and on the repair when
the context is corrupted — where the vanilla model cannot modulate anything?

S69 proves the mechanism in a controlled arena; S70 checks the signature survives on a
real foundation model and real benchmarks, at inference only (no new training).

## Setup

- Datasets: ETTh1, ETTm1, weather. Target channels: the 6 per dataset used by S31.
  Neighbours: the K=4 most target-correlated channels, correlation computed on the first
  80% of the timeline only (S31's no-leak rule), supplied as `past_covariates`.
- Missingness on the target: {mcar, block} x rate {0.3, 0.7}; 150 windows per cell.
- Manipulations:
  - **q_r** (target repair quality): fill the target's holes with {zero, linear, oracle}
    fill. For the CPT retrofit the fill is declared through its restored endpoint; for
    the vanilla model the fill is silent (its interface cannot carry a declaration).
  - **q_c** (context quality): neighbours either fully observed (clean) or holed at the
    same rate/mechanism and zero-filled (corrupt).
- Models (3): c2 vanilla (stock checkpoint), c2 +5k fair-budget control, c2 CPT retrofit
  — the exact checkpoints behind the paper's Table-4/leaderboard c2 rows (identify from
  `../s60_c2retrofit/` notes/logs; the dated `s20260903` set is the current candidate —
  verify against `s60_leaderboard_s20260903.json` before running).
- Metric: paired per-window relMSE against the model's OWN clean univariate forecast
  (S31 convention), median per cell.
- Total: 3 ds x 6 targets x 2 mech x 2 rate x 3 q_r x 2 q_c x 3 models = 2592 cells,
  inference only.

## Pre-registered predictions

- **P1 (vanilla is flat in q_r)**: across q_r in {zero, linear, oracle}, vanilla c2's
  relMSE varies by less than 5% (its interface discards fill content); q_c affects it
  only silently (clean neighbours help regardless of declaration).
- **P2 (substitution, CPT only)**: for the CPT model, the q_c effect (clean minus corrupt
  context, relMSE points) at q_r = zero is at least 2x the q_c effect at q_r = oracle.
  For vanilla and the +5k control the ratio is ~1 (no modulation).
- **P3 (garbage repair ignored)**: CPT at (q_r=zero, q_c=clean) is within 10% of CPT at
  (q_r=oracle, q_c=clean); vanilla at (zero, clean) is not (it eats the garbage).

## Gates

- **G0**: reproduce S31's stored anchors before any new cell: condition (c)/(d) at block
  0.7 within 10% of `../s31_crosschannel/s31_results.json`. If G0 fails, stop and
  diagnose the pipeline drift instead of proceeding.

## Deliverables

- `run_s70_rcprobe.py`, `s70_rcprobe.json`, `s70_notes.md` (with a cell-level table for
  the paper and the verdict on P1-P3).
- GPUs 4-5 (reserved); split the cell grid across the two cards. Expected runtime: hours.
