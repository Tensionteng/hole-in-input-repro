# S25 notes — two-floor decomposition: information floor vs architectural floor

**Status: PRE-REGISTRATION (written before any run). Results appended below after execution.**

## Motivation

Every repair round so far (S5F/S6E/S6F/S8/S12/S15/S18/S20/S21/S23) reports *a* number
for *a* repair. What none of them separate is **why** a repair fails:

```
R(f, g0) - R_clean  =  [R_stat - R_clean] + [R*(f) - R_stat] + [R(f,g0) - R*(f)]
                        (1) information      (2) architecture   (3) fill/routing
```

- **(1) information floor** — the target information the missingness actually destroyed.
  Model-independent. Estimated by the best *direct* predictor Z=(x_obs, mask) -> Y trained
  in-domain (no TSFM in the loop), normalized by its own clean baseline.
- **(2) architectural floor** — information that survives in Z but that a *frozen* f cannot
  be made to use, because the only lever we have is the input fill. Model-specific.
  R*(f) = inf over fills g of E[loss(f(g(Z)), Y)]; estimated from above by an amortized
  fill network trained end-to-end through frozen f.
- **(3) fill/routing suboptimality** — the only term MechGate can touch.

If (2) is large, no amount of detector/router work can help and the paper should say so
with a number. If (2) is ~0, the whole repair ladder is a policy problem and the failures
of S12/S15/S18/S21 were failures of *method*, not of *possibility*.

## The architectural claim this round tests (found by reading chronos_bolt.py:277-300)

`encode()` computes `context, loc_scale = self.instance_norm(context)` **before** applying
the observation mask, and `InstanceNorm` excludes only NaN (`torch.nanmean`), not mask==0.
The mask is applied afterwards: `patched_context = where(patched_mask > 0, patched_context, 0)`.
Therefore, for Chronos-Bolt:

> **Proposition (reachable set).** Let x_M be the values placed at the missing positions.
> (a) If x_M is passed as NaN (no explicit mask), the forecast is *exactly invariant* to
>     x_M: rank(J_M) = 0, where J_M = d(forecast)/d(x_M).
> (b) If x_M is passed as finite values together with an explicit observation mask, the
>     entire influence of x_M factors through the two scalars (loc, scale) =
>     (nanmean, nanstd) of the full context: **rank(J_M) <= 2**.
> (c) If x_M is passed as finite values with no mask (the ubiquitous fill-then-feed
>     practice), J_M is full rank: fabricated content enters the content channel directly.

This makes "geometry x input convention" (the S6D/S14 reversal principle) a statement
about the *rank of the fill-reachable set*, and it retro-predicts S5-fix1: the
observed-stats rescale probe recovers ~82% of zero-fill damage under mcar because it moves
exactly the reachable `scale` dimension, and backfires under mnar because that dimension is
estimated from a biased sample. Part 0 tests (a)-(c) numerically.

## Pre-registered predictions (recorded before running; falsifications stay verbatim)

- **P0-a** bolt NaN path: max |J_M| == 0.0 **exactly** (not merely small). Confidence high
  (code path is unambiguous). Falsified if any nonzero entry appears.
- **P0-b** bolt fill+mask path: numerical rank of J_M == 2 (singular values sigma_3.. below
  1e-4 of sigma_1). Confidence high. Falsified if rank > 2.
- **P0-c** bolt plain-fill path: numerical rank >> 2 (>= 20 at p=0.3). Confidence high.
- **P0-d** timesfm plain fill: nonzero J_M (finite-difference), i.e. no uncorruptible path.
  Confidence high.
- **P1 (Part A, oracle fill)** the per-window gradient-optimized fill reaches risk *below*
  the clean-context risk on the plain-fill path (the true values are feasible, so <= clean
  is guaranteed if the optimizer works; going *below* measures adversarial steerability).
  Pre-registered as an optimizer sanity gate, not a scientific claim.
- **P2 (Part B, the real question)** on the plain-fill path the amortized fill net closes
  most of the mcar/block gap (relMSE -> <= 1.1) but leaves a large residual under
  mnar_high at p=0.7 (relMSE >= 1.6, vs linear 2.17 from S5). Rationale: S23 showed
  mechanism-matched in-domain *training* recovers part of mnar_high (2.174 -> 1.364), and a
  fill net has strictly less freedom than retraining the forecaster.
- **P3** the fill+mask (rank-2) path is *worse* than the plain-fill path under mcar/block
  (only 2 dof to work with) but *better or equal* under mnar_high (2 dof cannot fabricate
  misleading content). Falsified if the ordering is uniform in either direction.
- **P4 (Part C)** the direct predictor's own-clean relMSE under mnar_high p=0.7 is
  materially below the fill net's, i.e. **(2) > 0**: the frozen model is a real bottleneck,
  not just the fill. This is the claim I am least sure about (55/45) -- the opposite result
  (direct ~= fillnet) would say the damage is pure information loss and would *strengthen*
  the information-floor story instead.
- **P5** term (3), measured as best-fixed-fill minus fill net, is small compared to (1)+(2)
  under mnar, consistent with S16's finding that mechanism routing adds ~0 on top of a
  fixed linear fill.

## Setup

- L=512. **H=64** for all optimization/decomposition parts -- bolt's native
  `model_prediction_length`, so a single forward pass; S5's H=96 triggers an autoregressive
  rollout with a 9x quantile-mixing heuristic that would contaminate a gradient measurement.
  The anchor gate is run at S5's H=96 to prove the harness reproduces the stored numbers.
- Windows/masks byte-identical to S5: SEED=20250810, `load_windows` with k=300 on the last
  20%, `make_mask` copied verbatim (cheaper parts take a *prefix* of the same `starts`, so
  window_idx and hence masks are unchanged).
- Train region for the learned components: first 80% of the timeline, disjoint from every
  evaluated window; starts drawn with SEED+1.
- Datasets ETTh1/ETTm1/weather; mechanisms mcar/block/mnar_high/mnar_extreme; eval rates
  {0.3, 0.7}. All relMSE are **own-clean** (S6F convention onwards).

---

# RESULTS

## Anchor gate — PASS (exact)

Re-running the S25 harness against three stored S5 cells (bolt, ETTh1, H=96, 300 windows,
S5's own mask seeds):

| cell | stored (S5) | rerun (S25) | dev |
|---|---|---|---|
| clean:none:0.0 | 11.199030 | 11.199030 | +0.0000% |
| mnar_high:linear:0.5 | 16.050925 | 16.050925 | +0.0000% |
| mcar:zero:0.3 | 12.359439 | 12.359439 | +0.0000% |

Not merely within +-5%: bit-identical to 5 decimals, so the copied `load_windows`/
`make_mask`/`fill_context` reproduce S5's windows and masks exactly and every S25 number is
paired with the S5 stores.

## Part 0 — the reachable-set proposition is CONFIRMED, exactly

Two independent measurements per (dataset, mechanism, convention):
**(i)** numerical rank of J_M = d(median forecast)/d(x_M), singular values thresholded at
1e-4 x sigma_1; **(ii)** a gradient-free permutation test — permute the fill values *among
the missing positions* (which preserves the context's mean and std exactly, up to float
summation order, while destroying the fill's content) and compare the resulting output
change to that of an independent redraw of the fill.

| convention | rank(J_M) med [min,max] | max abs(J_M) | perm / redraw output change |
|---|---|---|---|
| plain fill (no mask) — the ubiquitous fill-then-feed | 64 [54, 64] (saturated: J is 64 x abs(M)) | 7.78e0 | **0.72** median (1.16 max) |
| fill + explicit mask | **2 [2, 2] — every one of 72 series** | 1.85e-2 | **1e-6** (float summation noise) |
| NaN (bolt's native path) | **0 [0, 0]** | **0.0 exactly, all series** | **0** |

Full run: 3 datasets x 3 mechanisms x 8 series per cell, rate 0.3. The fill+mask rank
distribution over all 72 series is the single value {2}; the NaN-path Jacobian is
identically zero on every series, not merely small.

Pre-registered P0-a / P0-b / P0-c: **all three confirmed**, and more sharply than predicted —
the fill+mask rank is exactly 2 in every cell measured, and the NaN-path Jacobian is
identically zero rather than merely small.

**Proposition (fill-reachable set of Chronos-Bolt) — verified.** Because `encode()` applies
`instance_norm` *before* the mask and `InstanceNorm` excludes only NaN (`torch.nanmean`),
while the mask is applied afterwards as
`patched_context = where(patched_mask > 0, patched_context, 0)`:

- (a) NaN input: the missing values are not inputs at all. rank(J_M) = 0, exactly.
- (b) filled input + explicit mask: the *entire* influence of the fill factors through the
  two scalars (loc, scale) = (nanmean, nanstd) of the full context. rank(J_M) <= 2, and the
  permutation test shows the content is irrelevant to 5.5 orders of magnitude.
- (c) filled input, no mask: J_M is full rank; fabricated content enters the content channel
  and moves the forecast as much as an arbitrary redraw (ratio ~1.0 under mcar/block).

**Why this matters for the paper.** "Optimal fill = missingness geometry x model input
convention" (the S6D/S14 reversal principle) stops being an empirical slogan and becomes a
statement about the *rank of the fill-reachable set*: three conventions of the SAME model
give reachable sets of dimension 0, 2, and |M|. It also retro-explains S5-fix1: the
observed-stats rescale probe recovers ~82% of zero-fill damage under mcar because it moves
exactly the reachable `scale` dimension, and backfires under mnar because that dimension is
estimated from a value-biased sample. And it makes the zero-fill catastrophe a corollary:
only convention (c) can be catastrophically wrong, because only (c) has an unbounded lever.

Secondary reading (full run, plain-fill permutation ratio by mechanism): mcar 1.073,
block 0.703, **mnar_high 0.381**. Under scattered MCAR gaps only the fill's moments matter
even on the content path (permuting is as damaging as redrawing); under value censoring
*where* the fabricated values are placed matters much more than their multiset. So even the
content channel is mechanism-sensitive, and the gap geometry -> "which positions" ordering
is itself part of what a repair must get right.

## Methodology change made mid-round (recorded, with the evidence that forced it)

**The admissible fill class G must be bounded, and Part A is why.**

The pre-registration defined R*(f) = inf over fills g of E[loss(f(g(Z)), Y)] with g
unrestricted. Part A measured that infimum per window by gradient descent and it is
*adversarial*: on ETTh1 the optimised fill reaches 0.0021-0.0958 x the CLEAN-context error
(conv=1.00, i.e. every series converged past the truth-feasible point). In other words, by
choosing the values at the 30-70% missing positions one can make the frozen model forecast
*better than it does on the complete, true context*, by two to three orders of magnitude.

Two consequences:

1. **An architectural floor defined over unrestricted fills measures steerability, not
   repairability.** The infimum is not a floor at all; it is an input attack. (It is also
   E[inf], not inf E, so it was never a valid bound on an amortised policy -- the
   pre-registration flagged this, but the magnitude of the effect is what makes the
   unrestricted definition useless rather than merely loose.)
2. Amortising the same unrestricted objective *diverges*: the fill net walks off into the
   adversarial region, final loss 27-3e7 from an init of 0.9, and emits non-finite
   forecasts on the fill+mask path.

So Part B now optimises over an explicit, documented class: **fills within `radius`=3
observed-sigma of the linear interpolant** (zero-init = exactly the linear fill, so every
measured gain is a gain over linear). R*(f) is henceforth reported *relative to a stated
fill class*, which is the honest object anyway -- the architectural floor is only defined
once you say which fills are admissible.

This is itself a finding worth a paragraph in the paper: **TSFM forecasts are highly
steerable through the missing positions.** An adversary (or a badly-tuned imputer) that
controls which values land in the gaps controls the forecast. That is a second silent
failure alongside the calibration collapse, and it is the reason "learn the best fill" is
not automatically a safe objective.

## Part A — per-window oracle fill (ETTh1, complete; other datasets running)

| cell | linear | oracle fill (unrestricted) | oracle fill, identified set x_M >= tau |
|---|---|---|---|
| mcar 0.3 | 1.023 | 0.0195 | (n/a) |
| mcar 0.7 | 1.477 | 0.0021 | (n/a) |
| block 0.3 | 1.208 | 0.0958 | (n/a) |
| block 0.7 | 1.576 | 0.0088 | (n/a) |
| mnar_high 0.3 | 1.105 | 0.0509 | **1.0555** |
| mnar_high 0.7 | 1.900 | 0.0028 | **1.2916** |
| mnar_extreme 0.3 | 1.730 | 0.0302 | (n/a: two-sided exclusion, not a lower bound) |

`conv=1.00` everywhere: every series reached at least the truth-feasible risk, so the
optimiser is not the binding constraint.

**The informative column is the constrained one.** tau = max(observed) is the sharp lower
end of the identified set under top-k censoring and is computable without the truth (the
run asserts the true censored values all lie above it). Restricted to *statistically
admissible* fills, the best reachable risk through frozen bolt is 1.056 / 1.292 at
p=0.3/0.7 -- versus linear's 1.105 / 1.900.

This exposes a concrete defect in the default repair: **under value censoring, linear
interpolation produces fills that lie outside the identified set.** It fills below tau when
every censored value is provably above it. That is a theory-backed criticism of the
ubiquitous default, and it retro-explains why S6E's `tail_tobit` beat linear on synthetic
mnar (2.52 -> 1.97) -- a Tobit fill is precisely a fill pushed back inside the identified
set. Caveat: this is still E[inf] (per-window oracle within the constraint), so it is a
lower bound on what an amortised policy can reach, not an achievable number; Part B's
fill net is the achievable upper bound.

## Parts B+C — the two-floor decomposition (PAIRED, the headline result)

**Statistic.** Per-window relMSE r_w = MSE(method, w) / MSE(clean context, w), reported as
the **median over windows**, then averaged over the three datasets. The ratio-of-means
version originally computed is tail-dominated and not defensible here: on ETTh1 the clean
per-window MSE has mean 10.357 vs median 1.115, so a handful of windows set the mean and
the decomposition terms flip sign between the two statistics. Everything below is paired;
`s25_paired.json` holds per-cell median/mean/q75/q95 and paired win-rates.

| mechanism | p | best fixed fill | fill net (plain) | fill net (+mask) | direct | **(1) information** | **(2) architecture** | **(3) fill/routing** | win% |
|---|---|---|---|---|---|---|---|---|---|
| mcar | 0.3 | 1.013 | 1.026 | 1.048 | 1.013 | 0.013 | 0.013 | −0.013 | 50% |
| mcar | 0.7 | 1.121 | 1.187 | 1.236 | 1.057 | 0.057 | 0.130 | −0.066 | 48% |
| block | 0.3 | 1.020 | 1.042 | 1.029 | 1.036 | 0.036 | −0.007 | −0.009 | 50% |
| block | 0.7 | 1.086 | 1.220 | 1.121 | 1.094 | 0.094 | 0.027 | −0.035 | 48% |
| mnar_high | 0.3 | 1.193 | 1.094 | 1.294 | 1.102 | 0.102 | −0.008 | 0.099 | 60% |
| **mnar_high** | **0.7** | **2.875** | **1.672** | 2.272 | **1.308** | **0.308** | **0.364** | **1.203** | **81%** |
| mnar_extreme | 0.3 | 1.210 | 1.095 | 1.214 | 1.101 | 0.101 | −0.006 | 0.115 | 64% |
| mnar_extreme | 0.7 | 1.827 | 1.646 | 1.966 | 1.325 | 0.325 | 0.320 | 0.182 | 66% |

(1) = direct − 1; (2) = min(fill net, fill net+mask) − direct; (3) = best fixed − min(fill net).
win% = fraction of windows where a learned fill beats the best fixed fill, paired.

### What this says

**MCAR and block are saturated.** Total excess at p=0.7 is only 0.09–0.12, term (3) is
*negative* (a learned fill cannot beat linear/nan), and terms (1)+(2) are small. There is
nothing left to win. This is the quantitative explanation of S16's null result: mechanism
routing among fixed fills adds nothing on mcar/block **because the fixed fills are already
at the floor there**, not because routing is a bad idea.

**Value censoring at high rate is the opposite regime.** At mnar_high p=0.7 the total excess
is 1.875, and it splits as information 0.308 / architecture 0.364 / **fill 1.203**. That is,
**~64% of the damage is recoverable by changing the fill alone, with the foundation model
completely frozen**, and the learned fill wins on **81% of windows** paired. Best fixed fill
2.875 -> learned fill 1.672.

This is the single most consequential number in the round, and it **contradicts the current
repair ladder** in README section 4, which prescribes "mnar_high -> free repairs: linear,
full stop". Linear is the best *fixed* fill, but it is nowhere near the achievable frontier.

**The information floor is real but much smaller than the observed damage.** A
mechanism-matched, in-domain-trained direct predictor -- which is unconstrained by any TSFM
architecture and whose clean MSE actually *beats* zero-shot bolt (10.10 vs 10.36 ETTh1,
8.41 vs 8.14 ETTm1, 2088 vs 3675 weather) -- still degrades by 0.31 at mnar_high p=0.7 and
0.33 at mnar_extreme p=0.7, versus 0.06–0.09 under mcar/block. So censoring really does
destroy information, and the amount is now measured rather than asserted.

**The architectural floor is ~0 except under high-rate censoring**, where it is 0.32–0.36.
Caveat: term (2) is a difference between the relative degradations of two different models,
not a strict bound, so read it as "no evidence of an architectural penalty under
mcar/block/low-rate mnar; evidence of a real one under high-rate censoring".

### Pre-registration scorecard

| # | prediction | outcome |
|---|---|---|
| P0-a | NaN path rank(J_M) = 0 exactly | **CONFIRMED** (0.0 on all 72 series) |
| P0-b | fill+mask rank = 2 | **CONFIRMED** (exactly 2 on all 72) |
| P0-c | plain fill rank >> 2 | **CONFIRMED** (54–64, saturated) |
| P0-d | timesfm has no uncorruptible path | **NOT RUN** (deferred with the cross-model extension) |
| P1 | oracle fill reaches below clean | **CONFIRMED** (0.002–0.096x clean, conv=1.00) |
| P2 | fill net closes mcar/block to <=1.1, leaves >=1.6 at mnar_high p=0.7 | **SPLIT**: the mnar half is right (1.672); the mcar/block half is wrong in *direction* -- there was no gap to close, fixed fills were already optimal and the fill net is slightly worse |
| P3 | fill+mask worse than plain under mcar/block, better under mnar_high | **FALSIFIED, and instructively so** -- exactly reversed. fill+mask wins under **block** (1.121 vs 1.220) and loses badly under **mnar_high** (2.272 vs 1.672) |
| P4 | direct predictor materially better than fill net, i.e. (2) > 0 | **CONFIRMED only under high-rate censoring** (0.32–0.36); ~0 elsewhere |
| P5 | term (3) small compared to (1)+(2) under mnar | **FALSIFIED** -- (3)=1.203 exceeds (1)+(2)=0.672 at mnar_high p=0.7 |

### Why P3's falsification is the useful one

P3 was reversed, and the reason unifies Part 0 with the repair ladder: **the optimal rank of
the fill-reachable set is mechanism-dependent.**

- Under **block**, fabricating content is *dangerous* -- linear interpolation across a long
  gap invents a straight line the model then trusts. The rank-2 (fill+mask) and rank-0 (NaN)
  conventions cannot fabricate, so they win (block p=0.7: nan 1.086, fill+mask 1.121, plain
  learned fill 1.220).
- Under **value censoring**, fabricating content is *necessary* -- the truth lies above the
  threshold and the model must be told so. A rank-2 convention can only shift (loc, scale)
  and is therefore crippled (mnar_high p=0.7: fill+mask 2.272 vs plain 1.672).

So "optimal fill = geometry x input convention" is no longer an empirical slogan: the
convention selects the *dimension of the reachable set*, and the mechanism determines
whether you need that dimension to be large (censoring: fabricate) or zero (blocks: refuse
to fabricate). Part 0 supplies the dimensions, Parts B/C supply which one you want.

## What S25 changes for the project

**1. The repair ladder's mnar entry is wrong, and now measurably so.** README section 4
prescribes "mnar_high -> free repairs: linear, full stop; the information floor forbids more".
S25 splits that claim: linear is the best *fixed* fill (2.875 paired median at p=0.7), but a
small input-side network trained through the **frozen** model reaches 1.672 and wins on 81%
of windows, while the information floor is only 1.308. The floor is real; it just sits far
below where the fixed-fill ladder stops. Roughly **64% of the censoring damage is
recoverable without touching the foundation model**.

**2. It explains S16's null routing result instead of excusing it.** Routing among fixed
fills adds ~0 on mcar/block because term (3) is *negative* there -- the fixed fills are
already at the achievable frontier, so there is nothing for a router to win. The place where
term (3) is huge (censoring, 1.203) is exactly the place the current router has only fixed
fills to choose between. The system's problem was never the detector; it was the action set.

**3. It reframes MechGate's contribution.** The pipeline should route to *learned,
mechanism-matched fills*, not among {linear, zero, nan}. That gives the detector a real job
(the payoff for knowing the mechanism is now 1.2 relMSE, not 0.0) and turns the five rounds
of "learned repair failed" into a diagnosis: S12/S15/S18/S21 all tried to change the model or
its internals; S25 changes only the input, which is the one lever whose reachable set Part 0
shows is full-rank.

**4. It gives "geometry x input convention" a derivation.** Part 0 fixes the dimension of
the reachable set per convention (|M| / 2 / 0); P3's falsification shows which dimension you
want: rank-0/2 when fabrication is dangerous (blocks), full rank when fabrication is
necessary (censoring). The empirical reversal between Penmanshiel and METR-LA is the same
statement.

**5. Two new negative/practical findings worth a paragraph each.**
   - **TSFM forecasts are highly steerable through the missing positions** (Part A: an
     optimised fill reaches 0.002-0.096x the clean-context error). Whoever controls the
     imputer controls the forecast.
   - **chronos-bolt has a backward-pass silent failure.** `InstanceNorm` computes
     `scale = (x-loc).square().nanmean().sqrt()` and then `where(scale == 0, eps, scale)`;
     the `where` repairs the value but the gradient still flows through `sqrt(0) = inf`, so a
     single window with a constant filled context turns any input-side module trained through
     the frozen model entirely to NaN. Symptom is deceptive: NaN weights -> all-NaN context ->
     bolt masks everything -> it emits its no-information constant, so the forecasts stay
     *finite* and merely become input-independent. This belongs next to S17's forward-path
     fallbacks; anyone training an adapter in front of bolt (S5-fix2, S8) is exposed.

## Honest caveats

- The fill net is **mechanism-matched** (one net per mechanism, trained with the true
  mechanism known) and **in-domain** (train split of the same three datasets). So term (3)
  is "what a perfect detector plus a learned in-domain fill could recover", an upper bound
  on a deployable system, not a zero-shot result. The fixed-fill baseline is likewise given
  oracle mechanism knowledge (best of {linear, zero, nan} chosen per cell), so the
  comparison is not tilted toward the learned fill on that axis.
- Term (2) is a difference between the relative degradations of two different models
  (frozen bolt vs a from-scratch direct predictor). It is a diagnostic, not a bound.
- Fill nets were trained at rate ~U(0.1,0.8) and evaluated at 0.3/0.7 (not rate-matched):
  conservative.
- H=64 here vs H=96 in S5. The anchor gate reproduces S5 exactly at H=96; all S25 numbers
  are internally consistent and own-clean normalised, but absolute MSEs are not comparable
  to the S5 stores across horizons.
- Single model family (chronos-bolt). Part 0's proposition is bolt-specific by construction;
  the cross-model version (P0-d) is deferred.

## Part A — complete (24/24 cells, 40 windows x C channels, 600 Adam steps)

Statistic here is the ratio of means over 40 windows (not Part B's paired median), so the
columns are comparable within Part A but not directly against the Parts B/C table.

| cell | linear | oracle fill (unrestricted) | oracle fill, identified set | conv |
|---|---|---|---|---|
| ETTh1 mcar 0.3 / 0.7 | 1.023 / 1.477 | 0.0195 / 0.0021 | — | 1.00 |
| ETTh1 block 0.3 / 0.7 | 1.208 / 1.576 | 0.0958 / 0.0088 | — | 1.00 |
| ETTh1 mnar_high 0.3 / 0.7 | 1.105 / 1.900 | 0.0509 / 0.0028 | **1.055 / 1.292** | 1.00 |
| ETTh1 mnar_extreme 0.3 / 0.7 | 1.730 / 1.906 | 0.0302 / 0.0207 | — | 1.00 |
| ETTm1 mcar 0.3 / 0.7 | 1.013 / 1.209 | 0.0132 / 0.0031 | — | 1.00 |
| ETTm1 block 0.3 / 0.7 | 1.086 / 2.516 | 0.0391 / 0.0094 | — | 1.00 |
| ETTm1 mnar_high 0.3 / 0.7 | 1.175 / 2.497 | 0.0345 / 0.0058 | **1.115 / 1.745** | 1.00 |
| ETTm1 mnar_extreme 0.3 / 0.7 | 1.903 / 2.604 | 0.0226 / 0.0088 | — | 1.00 |
| weather mcar 0.3 / 0.7 | 0.974 / 0.833 | 0.0376 / 0.0079 | — | 1.00 |
| weather block 0.3 / 0.7 | 0.911 / 1.069 | 0.0737 / 0.0187 | — | 0.99 |
| weather mnar_high 0.3 / 0.7 | 1.017 / 1.200 | 0.1511 / 1.1817 | **0.988 / 1.196** | 0.89–0.98 |
| weather mnar_extreme 0.3 / 0.7 | 0.894 / 0.898 | 0.1254 / 0.0844 | — | 0.98 |

**Steerability (unrestricted column).** Across all 24 cells the optimised fill reaches a
median of **0.022x** the clean-context error (min 0.0021, max 1.18). Choosing the values at
the missing positions lets one drive the frozen model's forecast roughly **50x better than
it does on the complete, true context** — this is an input attack, not a repair, and it is
why Part B had to fix an admissible fill class (see the methodology note above).

**Identified-set column (mnar_high).** Restricted to fills that respect
x_M >= max(observed) -- the sharp lower end of the identified set under top-k censoring,
computable without the truth -- the reachable risk beats linear in 5 of 6 cells and the gap
widens with the rate: ETTh1 1.292 vs 1.900, ETTm1 1.745 vs 2.497 at p=0.7 (weather ties,
1.196 vs 1.200). Consistent in direction with Part B's learned fill, which is the achievable
(amortised, non-oracle) version of the same statement.

**Reading.** Part A is a lower bound (E[inf]) and Part B an upper bound (amortised, learned)
on the same quantity. They agree that under value censoring the default fill is far from the
frontier, and they disagree with each other by the amount you would expect from the
oracle/amortised gap.

---

## Round status

- Anchor gate: PASS (exact). Parts 0, A, B, C: complete.
- Deferred: P0-d (cross-model reachable-set structure: timesfm / moirai / chronos-2).
- Artifacts: `run_s25_twofloor.py`, `analyze_s25.py`, `analyze_s25_paired.py`,
  `make_s25_figure.py`, `s25_results.json` (merged), `s25_results_p{0,A,B,C}.json`,
  `s25_paired.json` (per-cell median/mean/q75/q95 + paired win-rates),
  `s25_paired_agg.json`, `s25.png`, logs `s25_*.log`, checkpoints `s25_ckpt/`.
