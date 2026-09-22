# S69 notes — implementation decisions, smoke validation, deviations

Pre-registered contract: DESIGN.md. This file records the interpretations made where the
DESIGN leaves implementation latitude, the smoke-run evidence, and (below the line) any
deviations discovered after launch. Current status: no deviations from DESIGN.md.

## Interpretations (implementation latitude, not deviations)

1. **"Standardized to unit variance"**: the target is generated as a *stationary* AR(1)
   with sigma = sqrt(1 - phi^2) and x_0 ~ N(0,1), so its law has exactly unit variance
   and the joint law of (x, c) given (phi, rho) is exactly Gaussian -- the Bayes
   yardstick is then exact, not approximate. No empirical re-scaling of sampled series
   (that would distort the AR covariance and break exactness).
2. **Eval window**: each of the 256 held-out series is evaluated at its final window
   (context [448, 960), horizon [960, 1024)). The process is stationary, so the window
   position is immaterial to both the model and the yardstick.
3. **E2 perturbation estimator**: simultaneous two-sided +/-0.1 perturbation of *all*
   repaired positions of a source, with the horizon-mean response normalized by the
   number of perturbed positions. This is the only reading under which the DESIGN's
   yardstick equivalence ("same perturbation definition applied to the Bayes predictor =
   analytic regression coefficients") holds exactly, and a single shared implementation
   (`gen_s69_corpus.perturbed_weight`) is used for the model (eval_s69) and for the G2
   ridge check (bayes_s69).
4. **Weight magnitudes**: under this normalization the Bayes weights are O(1e-4)
   (AR(1) is Markov -- only positions near the forecast origin carry weight, and the
   mean is over all repaired positions). The q-ordered *field* (which P3's Spearman
   uses) spans 0 -> ~1.5e-4 in q_r. Consequence: eval forwards run in strict fp32
   (TF32 disabled in eval_s69.py) so the ~1e-3-scale aggregate perturbation responses
   are resolved; TF32 remains on for training (as in S45).
5. **Interface wiring**: dual = 4 patch channels [x_content | x_flag | c_content |
   c_flag], native = 2 [x_content | c_content] (DESIGN: "no flag exists" for native).
   `input_patch_embedding` widened to 16*n_ch with the chronos init scheme (std =
   initializer_factor * in_dim^-0.5). Instance norm per content channel; forecast
   inverse-transformed with the target channel's loc/scale; flags never normalized; all
   patches attend (content is always present in the arena; flags carry observability, so
   there is no attention-level masking -- unlike S45's masked-patch attention).
6. **Init identity across arms**: torch.manual_seed(seed) before backbone construction
   (identical backbone across all four arms at a seed); the widened embedding is drawn
   under seed+913 (shared between same-width arms: A/B share, C/D share).
7. **Block geometry**: runs of 24 (the project's established BLOCK constant from S25),
   number of runs covering the drawn total rate.
8. **q_c eval semantics**: the whole context channel is one declared rate-1 block repair
   (flag=missing everywhere on c for dual arms; silent content for native), per the
   DESIGN's risk-section wording.
9. **Checkpoint cadence**: every 1k steps (`s69_ckpt/arm_<A>_s<S>_step<k>.pt`) per the
   saturation clause, plus the final `arm_<A>_s<S>.pt`. Not needed in the event: the
   measured budget (below) is far under the 6 h threshold, so all arms run the full 15k.

## Smoke validation (2026-09-03, GPUs 0/1, smoke corpus 4k/64 series)

- Corpus sanity (100k/256 full pool, same for smoke): var(x) 0.998, var(c) 1.000,
  mean lag-1 autocorr 0.825 vs E[phi] 0.825, mean corr(x,c) 0.703 vs E[rho] 0.700.
- 300-step training runs of all four arms (bs 128): losses 0.40 -> 0.34, no NaN skips,
  val clean pinball ~0.32; 2000-step arm D (bs 512): loss 0.311, val pinball 0.303.
- Wiring check at init (fresh dual model): swapping context content / x-flag / c-flag
  each moves the forecast (max|dy| 1e-4..4e-4) -- all four input channels live.
- Gates on the smoke eval pool:
  - **G0 PASS**: max|w*_R(q_r=0)| = 0.00e+00; MSE*(q_r=1, q_c=1, block, 0.7) vs
    clean-floor MSE* max per-series rel diff = 0.00e+00.
  - **G2 PASS**: ridge on simulated true features recovers w*_R / w*_C through the
    shared perturbation procedure, max abs err 0.0013 (tol 0.02), incl. the exact
    endpoint w*_R(q_r=0)=0.
  - Yardstick field sanity (smoke cells): w*_R monotone in q_r (0 -> 1e-4 -> 1.5e-4),
    w*_C monotone in q_c and decreasing in q_r (the substitution direction), MSE*
    decreasing in both -- the pre-registered ordering exists to be found.
- Throughput: 10.6 steps/s at bs 1024 (A800) -> ~23.5 min per 15k-step arm; two arms
  per GPU plus chained eval -> ~55 min per GPU. Far under 6 h: full 15k steps for all
  12 jobs, saturation clause not invoked.

## Deviations

None as of launch.

---

# s69b (round 2) — 2026-09-03

## Why

v1 ran cleanly (all gates passed) but the arena had no adversity: the hardest cell
(worst arm, mcar 0.7, garbage fill + garbage context) sat at relMSE 1.014 vs clean --
real-data missingness hurts ~250x more (bolt vanilla 3.51). AR(1) is Markov and nearly
unpredictable beyond a few steps at H=64, so ignoring fill AND context cost almost
nothing; arm D never needed to learn trust (flat relMSE 0.996-1.007, wR ~0.002-0.004
unmodulated, wC = 0). P4 failed (D ~ A at q_r=0), P3 indeterminate. This is the
DESIGN.md "AR(1) may be too easy" risk case; round 2 applies the extended remedy.

## s69b changes vs v1

1. **Process**: x = stationary AR(1) (variance 1-h) + K=3 Gaussian harmonics
   a_k cos(2 pi t / p_k) + b_k sin(2 pi t / p_k), a_k,b_k ~ N(0, h/3) iid; periods p_k
   drawn per series WITHOUT replacement from the bank; harmonic energy share
   h ~ U(0.6, 0.85). Context c as in v1. Still jointly Gaussian -> the yardstick stays
   exact: Cov(x_t, x_s) = (1-h) phi^|t-s| + sum_k (h/3) cos(2 pi (t-s)/p_k).
2. **Masks**: new tailblock mechanism (the LAST ceil(rate*L) positions holed) in both
   training augmentation (diverse regime: clean 25%, then thirds over
   {mcar U(0.05,0.9), block U(0.1,0.9), tailblock U(0.1,0.9)}) and eval
   (mcar/block/tailblock x rates {0.3, 0.7}).
3. **G0b headroom gate (AMENDED, see below)** runs before any training.
4. Full pre-registered protocol otherwise unchanged: 4 arms x 3 seeds, same budgets and
   interfaces; E1+E2+F1 evals write s69b_eval_<arm>_s<seed>.json; yardstick writes
   s69b_bayes.json; analysis writes s69b_analysis.json. v1 outputs untouched; v1
   reproducible via --round s69 everywhere.

## G0b amendment (approved 2026-09-03)

Original wording: "MSE*(q_r=0,q_c=0) / MSE*(1,1) >= 2.0 at tailblock 0.7 and >= 1.5 at
block 0.7, median over eval series; tune harmonic energy/periods until it clears."

**Structural finding**: the block-0.7 information-loss bar is unreachable by tuning.
Random (scattered) noiseless subsampling of a stationary Gaussian process has NO
aliasing floor -- 154 scattered noiseless points pin the harmonic amplitudes at any
period (two-regressor fits are well-conditioned unless period -> infinity, and then the
component is smooth and still pinned); and the AR part's exploitable predictability at
H=64 is only ~3-5% of its variance (mean_h phi^{2h} ~ 0.03-0.10 for phi <= 0.95), so
losing the recent tail at block 0.7 costs ~0.05 of total variance. Since q=0 fills
carry zero information by construction, the information-loss gap is purely a function
of the observed positions' geometry, and scattered geometry loses almost nothing.
Measured ceiling across 14 (bank, share) parameterizations: block 0.7 ratio 1.04-1.15.

**Decomposition** (both quantities exact, from the same covariance blocks; kept in
separate JSON fields `mse` vs `mse_naive` everywhere):
- information-loss gap = MSE*(q=0, observed-only) / clean floor -- large only for
  contiguous/tail censoring;
- fill-poisoning gap = MSE of the naive-trust linear predictor (treats q=0 fills as
  clean) / clean floor -- the level an interface that cannot modulate trust (native
  arms, vanilla bolt) lives at.

Tuning table (64-series medians, exact conditioning):

| bank | share | floor | tail0.7 info | block0.7 info | tail0.7 naive | block0.7 naive | mcar0.7 naive |
|---|---|---|---|---|---|---|---|
| {16,32,64} (spec example) | (0.3,0.6) | 0.545 | 1.12 | 1.04 | 1.56 | 1.24 | 1.49 |
| {64,128,256} | (0.3,0.6) | 0.568 | 1.15 | 1.04 | -- | -- | -- |
| {128,256,512} | (0.5,0.8) | 0.364 | 1.13 | 1.05 | -- | -- | -- |
| {192,384,768} | (0.6,0.85) | 0.299 | 2.41 | 1.08 | 2.97 | 1.70 | 2.39 |
| {256,512,1024} | (0.6,0.85) | 0.360 | 2.02 | 1.07 | -- | -- | -- |
| {512,1024,2048} | (0.6,0.9) | 0.305 | 3.17 | 1.05 | 3.70 | 1.35 | 2.48 |

Amended G0b (approved): **(a)** information-loss gap >= 2.0 at tailblock 0.7 (kept);
**(b)** fill-poisoning (naive-trust) gap >= 1.5 at block 0.7 (replaces the unreachable
block information-loss bar). Block stays scattered-runs-of-24 (NOT redefined as
contiguous) for continuity with v1 and the paper's mechanism taxonomy. Parameters:
bank {192,384,768}, share U(0.6,0.85) -- the {512,1024,2048} alternative's longest
component covers only 1/4 cycle per window (a real learning problem for the model), so
the milder bank was chosen with tail margin 2.41.

**Full 256-series G0b: PASS** -- info-loss tailblock0.7 median 2.385 (>= 2.0, q10
1.87); naive-trust block0.7 median 1.643 (>= 1.5, q10 1.28); references: info-loss
block0.7 = 1.070, naive-trust tailblock0.7 = 2.934. Clean floor MSE* mean ~0.30.

## P4 reframing under the decomposition

The expected native-arm level at q_r=0 IS the naive-trust line. At tailblock 0.7,
q_r=0, q_c=1: optimal floor ~1.06x clean (clean context substitutes for the holed
tail) vs naive trust ~2.97x -- a ~2.8x spread for "D beats native arms by >= 20% at
q_r=0, q_c=1" to live in. P4's flip clause is unchanged.

## s69b deviations from the round-2 spec

- G0b amended as above (approved).
- Harmonic bank {192,384,768} and share U(0.6,0.85) replace the spec's example
  {16,32,64} / U(0.3,0.6) (spec explicitly allowed tuning; approved).
- Training-mechanism mix extended to thirds {mcar, block, tailblock} within the
  non-clean 75% (spec: "add tailblock to training augmentation"; split not specified).
- Tailblock training rate range U(0.1,0.9) (same as block; unspecified by spec).
- Periods drawn without replacement (3 distinct components per series; spec said
  "drawn per series from a bank" without specifying).
- GPU discipline: idle-poller (< 2 GiB) per-card before each job, per the round-2
  compute note; queues may drain later than the v1 schedule when other tenants hold
  cards.

## s69b results addendum (2026-09-04, 10/12 evals; A_s0/A_s2 pending on a held GPU)

Gates: G0 PASS, G0b(amended) PASS (2.385 / 1.643), G2 PASS (max err 0.0003), G1 PASS
(clean-MSE spread 6-8% per seed <= 10%).

- P1 (native flat w_R): holds. Arm A's w_R range over q_r ~ 1.8e-4 vs arm D's
  modulation ~ 5e-3 (~28x). Note: arm B (native + diverse) shows intermediate
  modulation (~5e-3 range) -- content statistics of the fill leak q even without flags;
  the interface is what makes the modulation graded/reliable, not what makes some
  modulation possible. Worth stating precisely in the paper.
- P2 (C == A): FAILS informatively. Dual interface + filtered corpus is not inert but
  BRITTLE: at flagged-OOD cells C's relMSE reaches 6-24x A's, F1 flip deltas up to 3.3
  (vs A's ~1.0-2.5 relMSE at the same cells). The v1/S45 collapse does not replicate
  in-arena; declaration semantics are learned from diverse training, not from the
  interface alone. (Single seed-pair s1 as of writing.)
- P3: MIXED, and the E2 scalar needs care under harmonic processes. The optimal
  coefficient pattern is oscillatory, so the position/horizon-averaged w* is a small
  signed residual whose sign is phase-dependent (w*_R at mcar 0.7 goes 0 -> -0.0017 as
  q_r rises) while the model's aggregate response goes 0.0048 -> +0.0102. Hence the
  pre-registered SIGNED Spearman(w_R, w*) reads ~-0.9 for arm D at the mcar panels
  while the MAGNITUDE Spearman(|w_R|, |w*|) reads +0.92/+0.91 there -- D tracks the
  magnitude field nearly perfectly at mcar. At block panels the ordering is absent-to-
  inverted (signed +0.6/+0.7, magnitude -0.6/-0.7); at tailblock it is weak (+0.25).
  Countercheck arms A/B are not cleanly < 0.3 everywhere (up to 0.68 on near-flat
  fields -- wiggle-level correlations). Verdict: graded trust is cleanly demonstrated
  at mcar in the magnitude reading; the 25-cell signed-field formulation of P3 is
  contaminated by sign cancellation and should be refined (e.g. near-origin-windowed
  weights or per-position weight profiles) before the paper claims it.
- P4: PASSES decisively at every q_r=0 cell: D is 17-69% below the best native arm.
  At tailblock 0.7, q_c=1 (the updated P4 cell): D 69% below native, sitting AT the
  yardstick floor (rel 1.06 vs floor* 1.062) while natives sit at the naive-trust line
  (~2.9 vs naive* 2.94) -- the information-loss/fill-poisoning decomposition is borne
  out quantitatively. F1 flip at q_r=0 raises D's relMSE to ~2.0 (toward native) at
  tailblock 0.7. Even at q_c=0 (information gone, floor 2.38) D beats natives by ~43%.
- P5: PASS at s1 (D within 1.0% of A); awaits A_s0/A_s2 for the other seeds.
