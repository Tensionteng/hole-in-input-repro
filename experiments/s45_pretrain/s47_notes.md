# S47 notes — the recipe ablation ladder (stage 2)

**Status: results (F/G/H evaluated; I and H@small training).**

## The question and the arms

Stage 1 (S45) showed interface x mechanism-diversity matters. Stage 2 asks which ingredient
carries the weight, using single-ingredient arms on the same budget (bolt-tiny, 15k steps,
bs 1024, restored interface throughout):

- F: CPM block-mask (TiRex/Toto/FlowState 2025 recipe; zeroed content + flag)
- G: mechanism-diverse, declare-only (no fill mixture)
- H: fill-diverse (alpha-blend), single mechanism (block)
- I: mechanism-diverse at FULL fill dosage (the deconfounder vs C, training)
- references: C (mechdiv mixture = 25% declare + 25% alpha-blend), E (Moirai-2.0 recipe)

## Headline numbers (21 closeable cells, median closure of A's excess at alpha=1)

| arm | closure a=1 | floor wins vs A (a=0, of 24) | clean ratio vs A (median) | attack (median) |
|---|---|---|---|---|
| C | 55.2% | 17 | 1.055 | x1.16 |
| E | 1.0% | 13 | 1.023 | x1.22 |
| F | -63.7% | 8 | 1.095 | x1.26 |
| G | 1.6% | 13 | 1.036 | x1.36 |
| H | **91.7%** | **20** | **0.975** | x1.20 |

By mechanism (closure): scattered C 33.1 / H 87.5; block C 52.1 / H 82.1.

## Scorecard against DESIGN2's pre-registered predictions

- **Q1 (C > E and C > F): CONFIRMED.** 55.2% vs 1.0% and -63.7%. The 2025 recipes read
  content (rho 0.82-0.98) but do not use it; CPM is actively harmful at high fill quality.
- **Q2 (G ~= C at a=0; G < C at a>=0.5): SPLIT.** G is worse than C nearly everywhere
  (C<G in 18/24 at a=0, 24/24 at a>=0.5): declare-only teaches ignoring, not using.
- **Q3 (H < C under block, H ~= C under scattered): FALSIFIED.** H beats C under BOTH
  (82.1 vs 52.1 block; 87.5 vs 33.1 scattered). The fill-quality axis transfers across
  mechanisms even when trained on one.
- **Q4 (winner at bolt-small within 15 pts of tiny): PENDING (H@small training).**
- **Q5 (nothing beats C on full-grid median): FALSIFIED.** H beats C decisively
  (91.7 vs 55.2). Per DESIGN2's own rule, the claim is revised to: the dominant
  ingredient is fill-quality diversity with the flag; mechanism diversity helps against
  single-mechanism recipes but is not the main driver.

## Interpretation (provisional, pending arm I)

- Closure tracks the DOSAGE of alpha-blend fill-diversity in training: G/E ~0 dosage
  (1-2%), C half dosage (55%), H full dosage (92%).
- Arm I (mechdiv at full dosage) separates "mechanism diversity adds something" (I > H)
  from "it is all fill dosage" (I ~= H).
- F's -63.7% deserves its own sentence in the paper: the 2025 consensus recipe teaches
  distrust past the point of use where the fill is good.
