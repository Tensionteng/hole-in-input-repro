# S19 — mechanism of the S15 mnar_high unmask gain (appendix-level exploration)

**Status: pre-registration frozen 2026-08-13 after smoke (`s19_smoke.log`)
passed and BEFORE any grid eval. Smoke proved, on mnar_high p=0.7 ETTh1 with
the s15 checkpoints: (1) S13 full_miss knockout (all 12 layers) is EXACTLY
inert under the native decode mask (mb+KO vs mb max|diff| = 0.0, replicating
S13 P4 under LoRA SFT); (2) the same knockout BITES once S15's unmask decode
is active (max|diff| 7.89); (3) under LoRA + knockout, knocked gap queries put
exactly 0 mass on patch keys and 1.0 on [REG] at all 12 layers; (4) ZS pipe
composition works; (5) post-restore bitwise clean. Results sections below are
filled in after the runs; nothing above this line is edited post-hoc.**

## Question

S15 opened bolt's decoder cross-attention mask on fully-missing patch
positions. Pre-registered mcar/block endpoint: nothing (Outcome B). But the
mnar_high p=0.7 negative control showed an unexpected REAL gain on ETT
(sft:unmask vs sft:mb, paired per-window: ETTh1 +5.67% t=-15.5 win 0.98,
ETTm1 +6.91% t=-7.6 win 0.82; weather flat +0.24%; survives own-clean
normalization: 4.34<4.58, 5.27<5.66, 2.70<2.73). The cells remain ~4x
degraded vs clean, so this is not repair. Working hypothesis ("phase
scaffold"): the rebuilt gap content — a smooth, position-aware interpolation
across the censored peak regions — gives the forecast head a better
phase/continuity prior than isolated low-value islands, reducing SECONDARY
damage (level/phase misalignment), not restoring the censored peaks. S19
tests the mechanism; it does not propose a method.

## Design (all on mnar_high p=0.7, nan fill, 150 windows s5.SEED, 1
## deterministic mask; ETTh1/ETTm1 primary, weather flat control)

Conditions (7): `zs_masked`, `zs_unmask`, `zs_unmask_ko`, `mb` (= S15
mb_anchor, bit-identical to S12 mb), `mb_ko` (registered control: knockout
must be exactly inert under the native mask), `unmask`, `unmask_ko` (the
ablation). Knockout = S13 `Knockout(raw, all 12 layers, "full_miss")`:
gap queries are redirected to [REG], so the rebuilt content never forms;
the decoder mask stays open, so any pure normalization side-effect of the
unmask (softmax mass redistribution over observed keys) is preserved.

1. **Causal ablation**: `unmask_ko` vs `unmask` vs `mb`. Gain retention =
   rel_gain(unmask_ko vs mb) / rel_gain(unmask vs mb).
2. **Error anatomy** (paired, per window): horizon quarters (h0-24 … h72-96);
   exact level/shape decomposition (mse = (μp−μg)² + centered-mse); dose
   response by #fully-missing patches (the amount of scaffold readable).
3. **Top-decile split**: gain inside vs outside (window,channel) pairs whose
   FUTURE contains a high peak (fut_peak_clean = (max future − clean-context
   mean)/std; top decile per dataset).
4. **SFT necessity**: zs:unmask vs zs:masked on the same cell (S15: flat)
   plus zs:unmask+KO — the gain must be an unmask × SFT interaction.

## Pre-registered predictions (scored post-hoc; no tuning toward them)

- **P1 (content causality)**: unmask_ko retains ≤ 20% of the unmask-vs-mb
  gain on ETTh1/ETTm1. Retention ≥ 80% ⇒ the gain is an architectural
  side-effect of opening the mask and the scaffold story is dead.
- **P2 (phase prior)**: horizon-quarter gains front-loaded: rel_gain(q1) >
  rel_gain(q4) on ETT. A flat profile fits a pure level-shift story better.
- **P3 (secondary damage, not peak recovery)**: the gain is NOT concentrated
  in future-peak pairs (top-decile gain ≤ rest gain), and a visible share of
  it comes from centered/shape error, not only the window mean.
- **P4 (dose response)**: gain grows with the fully-missing-patch tercile on
  ETT (monotone).
- **P5 (controls)**: weather |gain| < 1% for every paired comparison;
  mb_ko == mb bitwise (max per-window diff exactly 0.0).

## Gates

- **smoke** (`s19_smoke.log`): PASSED — see status header.
- **anchor** (`s19_anchor.log`): the 4 non-KO conditions re-evaluated through
  this harness must match `s15_results.json` neg cells (mnar_high p=0.7, all
  3 datasets) within ±5% (expected bit-exact: same code path, deterministic
  masks). Merge/analysis refuses to run on a failed gate.

## Results

**Anchor gate (`s19_anchor.log`): PASS, bit-exact.** All 12 base cells (4
non-KO conditions × 3 datasets) reproduce `s15_results.json` at ratio
1.000000 with max per-window diff exactly 0.0 — far inside the ±5% tolerance.
My `mb`/`unmask` therefore are S15's cells; the KO conditions differ by
exactly one controlled variable.

**Controls (P5: PASS).** `mb_ko == mb` bitwise on all 3 datasets (max
per-window diff exactly 0.0) — the knockout is provably inert under the
native mask, so any effect under the open mask is content, not machinery.
Weather paired rows all |gain| < 1% (un +0.24%, unko −0.01%, zs +0.28%).

**1) Causal ablation (P1: PASS, stronger than predicted).** Paired per-window
vs mb (n=150): the unmask gain VANISHES and INVERTS under the knockout —
ETTh1: unmask +5.67% (t=+15.5, win 0.98) → unmask_ko **−2.88%** (t=−5.3);
ETTm1: +6.91% (t=+7.6, win 0.82) → unmask_ko **−22.04%** (t=−11.7); direct
unmask_ko vs unmask: −9.07% / −31.10%. Gain retention = **−0.51 / −3.19**
(registered threshold ≤ 0.2). The SFT-unmask head has become *dependent* on
the rebuilt scaffold: forcing it to read null (REG-only) gap tokens is far
worse than never reading gap positions at all. The S15 gain is entirely
content-driven; the architectural side-effect explanation (softmax
renormalization, which survives the knockout) is ruled out.

**2) Error anatomy.**
- Horizon (P2: **REJECTED**): the gain is horizon-UNIFORM, not front-loaded —
  ETTh1 q1..q4 = +4.96/+5.15/+6.42/+6.12% (q4 > q1); ETTm1 +7.25/+6.54/
  +6.68/+7.31%. A phase prior should decay with distance from the context;
  a level correction should not. (The KO penalty on ETTm1 is back-loaded,
  q4 −31.1%: null tokens mislead more at long horizon.)
- Level vs shape (exact decomposition mse = (μp−μg)² + centered-mse): the
  gain is a LEVEL correction — ETTh1 level share **106%** (shape −6%;
  shape_corr even ticks down 0.2607→0.2565, t=−1.9); ETTm1 level 72% /
  shape 28% (shape_corr 0.2150→0.2541, t=+8.4 — a real but secondary shape
  component on ETTm1 only).
- Signed bias (supplementary eval, `s19_bias.log`): all conditions
  UNDER-predict the future mean under mnar_high (ETTh1: zs −3.569, mb
  −3.934; ETTm1: zs −3.298, mb −3.290, original units). Unmask pulls the
  bias toward zero (ETTh1 −3.759, ETTm1 −3.082; paired |bias| gain
  +4.29% t=+19.8 / +4.63% t=+7.9); the knockout deepens it (−4.072 /
  −3.939). Side observation: on ETTh1 SFT-mb is MORE biased than ZS
  (−3.93 < −3.57) — mcar/block-augmented SFT deepened the mnar
  under-prediction; unmask recovers part of exactly that deficit.
- Dose response (P4: PASS): gain grows with the number of fully-missing
  patches — ETTh1 terciles +4.07/+5.97/+8.01% (monotone; Spearman ρ=+0.315,
  t=+10.7); ETTm1 +5.49/+5.05/+7.31% (ρ=+0.169, t=+5.6; t1≈t2, t3 clearly
  highest). Under the knockout the dose slope inverts (ETTh1 t3 −9.22%,
  ETTm1 t3 −22.7%): the more scaffold there was to lose, the worse the
  damage — a mirror-image confirmation.

**3) Future-peak top decile (P3: PASS).** The gain is NOT concentrated in
windows/channels whose future contains a high peak; it is weaker there —
ETTh1: top-decile +4.12% vs rest +5.69% (ρ(fut_peak, gain)=−0.151); ETTm1:
+1.91% vs +6.96% (ρ=−0.199). Window-level (n=15 top windows, weak power):
ETTh1 +4.06% vs +5.82%; ETTm1 +7.24% vs +6.88% (mixed, as expected at
n=15). The scaffold does not recover peaks — it helps exactly where the
future is flat and the error was collateral.

**4) SFT necessity: confirmed interaction.** zs:unmask vs zs:masked on the
same cell: ETTh1 −0.84% (t=−4.8), ETTm1 −0.27% (t=−0.3) — flat-to-slightly-
harmful zero-shot. Interaction (sft gain − zs gain): +6.51 / +7.18 pp. Yet
the ZS unmask read is itself content-dependent: zs_unmask under KO collapses
to −9.58% / −19.92% — even the pretrained head reads the rebuilt content
when the mask is opened; it just reads it badly. Gain = unmask × SFT.

## Verdict

P1 PASS, P2 REJECTED, P3 PASS, P4 PASS (ETTm1 t1≈t2 wrinkle), P5 PASS.

**One-sentence mechanism conclusion:** the S15 mnar_high gain is a *level
scaffold* effect — reading only the low-value islands that survive high-value
censoring makes the head systematically under-predict the future mean, and
the rebuilt smooth interpolation across the censored peaks restores the
context's effective level (bias −3.93→−3.76 on ETTh1, −3.29→−3.08 on ETTm1;
horizon-uniform gain, ~100% level share on ETTh1, dose-dependent on
fully-missing patches, absent in future-peak windows); it is wholly
content-caused (knockout inverts it to −2.9%/−22.0%) and SFT-gated (ZS
flat) — the pre-registered "phase scaffold" framing survives only in
narrowed form: continuity helps, but through level calibration, not phase.

## Caveats / honest notes

- This is a mechanism appendix, not a method claim: the cells remain ~4x
  degraded vs clean, the pre-registered S15 mcar/block endpoint was Outcome
  B, and mcar/block are unaffected by all of the above.
- ETTm1's −22% KO penalty shows sft:unmask is *fragile* to gap-content
  corruption on that dataset — dependence, not just use.
- Weather tercile-1 sub-bin shows +2.41% (t=3.8) against an overall +0.24%:
  a post-hoc sub-bin fluctuation, not a registered comparison; all
  registered weather rows are < 1% (P5 intact).
- Window-level peak split has n=15 in the top decile — descriptive only;
  the (window,channel)-level split (n=105/decile) is the powered test.
- Alternative "renormalization side-effect" explanation is excluded by
  design: the knockout leaves the decoder mask open (same attention
  normalization over observed keys) but removes the rebuilt content.

## Artifacts

`run_s19_mnarunmask.py` (--smoke/--eval --jobs {anatomy,ablation}/--bias/
--anchor/--merge/--summary/--figure), `s19_results.json` (anchor, controls,
21 cells with per-window + per-(window,channel) vectors, maskstats, bias
analysis, full analysis tables), `s19.png`, logs `s19_smoke.log`,
`s19_anatomy.log`, `s19_ablation.log`, `s19_bias.log`, `s19_anchor.log`,
`s19_merge.log`, `s19_summary.log`. Runtime: smoke ~2 min, evals ~3 min on
2 GPUs (pinned 3-4), bias ~2 min, merge/figure seconds.
