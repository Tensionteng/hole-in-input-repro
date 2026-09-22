# S69 — the trust arena: selective source weighting under independently manipulated repair quality

## Question

When the target's repair R and a correlated context C are both available, has a model
trained under the restored interface with reliability-diverse fills learned *graded,
per-source* trust — weighting each source by its declared quality, and shifting weight
between them when one degrades? Which ingredient (interface vs reliability-diverse
training) is necessary? And how does the learned weighting compare with the computable
optimum?

This is the controlled-arena version of the paper's layer-2 claim ("the model learns how
much to trust the fill"). The S45 factorial showed *that* the prescription works on real
benchmarks; S69 shows *what exactly* was learned, in a setting where the Bayes-optimal
source weights are known in closed form, so "trust" becomes measurable rather than
metaphorical.

## Arena (synthetic, fully known ground truth)

Per series (all draws per-series, fixed for the series' lifetime):

- target: AR(1), x_t = phi * x_{t-1} + sigma * eps_t, phi ~ U(0.7, 0.95), standardized to
  unit variance;
- context: c_t = rho * x_t + sqrt(1 - rho^2) * eta_t, rho ~ U(0.5, 0.9), eta iid N(0,1);
- length 1024; train pool 100,000 series, eval pool 256 series from held-out generator
  seeds (disjoint rng stream).

Window: L=512 context + H=64 horizon (same as S45). Task: forecast the target's horizon
from the bivariate context window.

Everything is jointly Gaussian given (phi, rho), so the Bayes predictor of the future
given any declared/corrupted observation set is exact numeric conditioning (E3 below).

## Arms (2 x 2, in-arena lettering = the paper's quadrant semantics)

| arm | interface | training corpus | = user quadrant | = S45 analog |
|-----|-----------|-----------------|-----------------|--------------|
| A | native (missing -> 0, no flag) | filtered (clean only) | no training, blocked | A |
| B | native | diverse missingness | training, blocked | D |
| C | dual (content + per-position flag, both channels) | filtered | no training, accessible | B |
| D | dual | diverse missingness | full prescription | C |

Diverse regime, per training window, independently on EACH channel: clean with p=0.25;
else mechanism in {mcar (rate ~ U(0.05, 0.9)), block (runs covering total rate ~
U(0.1, 0.9))}; masked positions filled with a q-blend, fill = q * truth + (1 - q) *
N(0,1) noise, q ~ U(0, 1) per window-channel. Native arms: identical corruption
geometry, but masked positions carry 0 and no flag exists (augmentation through the
blocked interface, exactly as S45 arm D). Dual arms: content + flag per channel
(4 input channels: x_content, x_flag, c_content, c_flag); native: 2 channels.

The noise endpoint (q=0 carries zero information about x) is what makes the yardstick
clean: optimal w_R(q_r=0) = 0 exactly.

Same backbone as S45 (bolt-tiny, 8.7M-class; input embedding widened to the channel
count), same optimizer schedule (15k steps, bs 1024, lr 3e-4, warmup 1k), checkpoints
every 1k steps. If loss and probe metrics have clearly saturated by 8k, stop there and
record it; do not extend beyond 15k. Seeds: 3 per arm (12 jobs).

## Evaluation (held-out 256 series)

Notation: q_r = quality of the target repair (blend coefficient toward truth),
q_c = quality of the context repair (whole-context block repair at blend q_c, declared).

- **E1 heatmap**: q_r x q_c in {0, .25, .5, .75, 1}^2, target missingness mcar & block
  x rate {0.3, 0.7}. Native arms receive fills silently (content present, positions not
  marked); dual arms receive them declared (flag=missing). Record absolute MSE and
  relMSE vs the arm's own clean MSE.
- **E2 implied weights** (perturbation): at each grid cell, delta = +/-0.1 on the target
  fill at missing positions -> w_R = mean(d_yhat)/delta over positions and horizons
  (two-sided average); same on the context fill -> w_C. Flag-flip F1 (dual arms only):
  content fixed, re-flag repaired target positions as observed -> Delta forecast
  (declarations are load-bearing iff this is nonzero).
- **E3 Bayes yardstick**: per (phi, rho, q_r, q_c, mechanism, rate), exact
  linear-Gaussian conditioning of the future on the declared observation set; report the
  optimal implied weights w*_R, w*_C (same perturbation definition applied to the Bayes
  predictor = analytic regression coefficients) and the MSE* floor.

## Pre-registered predictions

- **P1 (the gate)**: native arms' w_R(q_r, .) is flat in q_r (content used silently, no
  per-window modulation); their MSE still varies with q_r (better content helps), but the
  weight does not adapt.
- **P2 (in-arena replica of the S45 collapse)**: arm C (dual + filtered) is
  indistinguishable from arm A on E1/E2 within tolerance; F1 flag-flip changes its
  forecast by ~0 (bitwise or near-zero).
- **P3 (substitution — the kill shot)**: arm D shows dw_R/dq_r > 0, dw_C/dq_c > 0, and
  dw_C/dq_r < 0 (context weight rises as repair quality drops); Spearman rank correlation
  between arm D's w_R field and the w*_R field over the 25 cells >= 0.9. Arms A/B show no
  such ordering (|Spearman| < 0.3 against w*_R).
- **P4 (where the interface pays)**: at q_r = 0 (any q_c), arm D's MSE is at least 20%
  below the native arms' (it can ignore the garbage; they cannot); the F1 flag-flip at
  q_r = 0 (declaring garbage as observed) raises arm D's MSE toward the native level.
- **P5 (cost)**: arm D's clean MSE within 5% of arm A's.

## Gates

- **G0 (yardstick sanity)**: w*_R(q_r=0) = 0 exactly; MSE*(q_r=1, q_c=1, rate 0.7 block)
  approximately equals the clean-floor MSE*.
- **G1 (comparability)**: all four arms' clean MSE within 10% of each other before any
  missing-data eval is interpreted.
- **G2 (measurement sanity)**: a ridge regression fitted on the true features recovers
  w*_R, w*_C within tolerance, validating the perturbation pipeline.

## Risks and handling

- AR(1) may be too easy (all arms saturate). If G1 shows trivially equal performance AND
  E1 shows no separation anywhere, add per-series harmonic components and mark the
  yardstick as the AR-approximation of the true optimum (flagged in the writeup).
- The transformer may not reach the Bayes floor: P3 is ordinal (rank correlation), not
  absolute. We do not claim optimality, only ordering.
- q_c as whole-context repair is at the edge of the training distribution (rate-1 block
  on the context channel): the diverse regime includes block rates up to 0.9, so rate 1.0
  is a mild extrapolation; if arm D behaves erratically specifically at the q_c cells,
  add rate-1.0 context blocks to training and re-run arm D only.

## Compute plan (8 x A800, all free as of 2026-09-03)

- GPU 0-3: arms A-D, seed 0. GPU 6-7: arms A-D, seed 1 (two jobs per GPU; bolt-tiny bs
  1024 fits easily). GPU 4-5: reserved for S70 (inference probe).
- Seed-2 replicas chain onto GPU 0-3 when seed 0 finishes. Each training job chains its
  own E1/E2 eval on completion (writes s69_eval_<arm>_s<seed>.json).
- Scheduler: per-GPU FIFO runner over queue/*.sh job files; logs per job.

## Deliverables

- `gen_s69_corpus.py`, `train_s69.py` (adapted from `../s45_pretrain/train_s45.py`),
  `eval_s69.py` (E1/E2), `bayes_s69.py` (E3), `analyze_s69.py`, `run_queue.sh`.
- `s69_ckpt/arm_{A,B,C,D}_s{0,1,2}.pt`, `s69_eval_*.json`, `s69_bayes.json`.
- Figure sources for the paper: four heatmap panels (one per quadrant) with the Bayes
  floor as reference panel, and the w_R(q_r) curves at q_c=1 per arm overlaid with w*.
- This DESIGN.md is the pre-registration; deviations are recorded in s69_notes.md with
  reasons.
