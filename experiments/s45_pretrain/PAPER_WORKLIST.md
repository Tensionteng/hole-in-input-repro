# Paper revision work list (from the story walkthrough, 2026-08-27/28)

Everything below was agreed during the part-by-part story check. Ordered by paper
section. Experiment-side facts live in the stage notes; this file is the writing
guide. Nothing here has been applied to main.tex yet.

## Narrative structure

1. **Open with real data, not the probe.** Motivation = Penmanshiel curtailment
   (stock error 3× its clean level; 64% of the degradation recoverable, S25/S26)
   → then the mechanism diagnosis (probe/rank). Current order is reversed.
2. **Frame the fix as "a cheap, post-hoc retrofit"**: 5k steps, 40 min on one
   GPU, no access to the original pretraining corpus, no architecture change.
   Not "can we retrain".
3. **No failure-history narration.** P5's zero-fill blindness is presented as an
   ablation arm ("without the declare endpoint, the recipe fails on zero fill"),
   not as "our first attempt broke". The recipe map (P5/P6/P8/Q50) is a
   component×behaviour ablation matrix, not a timeline.
4. **P4 goes into the main table** (stock + same 5k steps, no missingness): the
   fair-training-budget control. Its closure is −19% and its clean degrades
   3.8–12.9% — the retrofit's win is larger under the fair comparison.

## Metrics and definitions

5. ρ and closure definitions go to the appendix with formulas AND one worked
   numeric example each (use the ETTm1|mnar_high|0.7 cell); main text gets a
   one-sentence intuitive version at first use.
6. Disclose the small-denominator rules: closure drops cells with excess ≤ 0.02;
   ρ's denominator is guarded. Report absolute errors for presentation; closure
   is used only for attribution analysis.
7. E-arm's "−65% ± 106" is rewritten robustly ("median closure fluctuates around
   zero across seeds, range −187% to +1%: zero benefit from a perfect fill").

## Evidence and citations

8. "Zero fill is the deployment default": cite the TSLib / GluonTS / Chronos
   papers, one sentence that all three codebases fill with 0 (footnote with file
   and line: data_loader.py:434,439; split.py dummy_value=0.0; chronos2 model).
9. Premise evidence table (S513) into the paper: GIFT-Eval per-subset NaN rates
   (electricity 19.2%, kdd_cup 17.1%, ... 11.77% overall), Penmanshiel 26% of
   windows, METR-LA 23.2%, UCI original electricity 20.15% zeros / 57% clients
   starting unconnected, benchmark zero-fill fingerprints (labelled as
   fingerprints, no overclaim), the 26%-hardest-windows-dropped selection effect.
10. Same-name benchmark versioning: TSLib electricity is 2016–2019 while UCI is
    2011–2014 — non-overlapping distributions under one name; used as the
    preprocessing-black-box argument.

## Results to update

11. Main table becomes the S57 leaderboard (8 models, Q50 row) + the S59
    any-fill matrix. TimesFM 2.5 joins the convention census as "overwrite"
    (its three fill columns are identical; illness 19.29 vs stock's 2.02).
12. The GIFT-Eval dose-response goes into the main text: flat ~5% median cost
    (+53% worst cell, m4_hourly), payoff growing to −16%…−32% above 30% missing,
    crossover ~3%; deployments sit above the crossover (S513 rates).
13. Clean-cost language: report the range (4.4–7.0%) + the WiSE-FT dial, NOT
    the pre-registered ±5% gate verdict (P5 narrowly missed it at tiny, met at
    base). Q30/P10 are negative results for the appendix: neither interpolation
    nor distillation removes the residual cost — it is a property of continued
    training (P4 control), priced into the dose-response story.
14. Deployment routing rule: missing rate <3% → stock, ≥3% → Q50. This is the
    honest "worst case is parity" implementation.
15. M_ladder.tex: fix the H@small denominator (was A@tiny; correct same-size
    numbers: 76.3% [+71,+89], floor 9/24, clean 0.929 — s48_eval_small.json).
16. S512 attention figure: P0's attention is flat over fill quality, P5
    up-weights masked patches at α=1 (mechanism-level evidence, one figure).
