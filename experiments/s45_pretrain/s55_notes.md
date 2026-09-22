# S55 — the winning arms on REAL missingness (DESIGN2's pre-registered second axis)

Data: `s55_realfill.json` via `eval_s55_realfill.py`. Windows are the S26 held-out
TEST halves (per-turbine/-sensor 50/50, seed 20250812): Penmanshiel curtailment
(CTX=144, H=24, NMSE var floor 100) and METR-LA outages (CTX=512, H=64, floor 4.0).
Fixed fills only; every number is NMSE per window, mean and median reported.

## Penmanshiel (real value censoring), TEST half

| arm | ctrl clean | keep | zero | linear |
|---|---|---|---|---|
| P0 (stock) | 5.85 / 1.79 | 27.5 / 1.82 | 19.9 / 1.86 | 45.2 / 1.80 |
| A (from-scratch, filtered) | 76.1 / 2.21 | 544.8 / 4.37 | 513.5 / 6.10 | 342.3 / 2.61 |
| H (from-scratch, winning recipe) | 16.9 / 2.78 | 307.6 / 3.28 | 322.2 / 4.11 | 281.8 / 2.98 |
| P5 (CPT-from-stock) | **3.80** / 1.82 | 67.9 / 1.95 | 46.8 / 1.78 | 59.6 / 1.86 |

(mean / median)

## METR-LA (real block outages), TEST half

| arm | ctrl clean | keep = zero | linear | nan |
|---|---|---|---|---|
| P0 (stock) | 2.99 / 1.29 | 2.38 / 1.42 | 2.14 / 1.26 | 2.19 / 1.26 |
| A | 9.90 / 1.79 | 23.4 / 10.7 | 10.6 / 1.97 | 10.3 / 1.91 |
| H | 4.77 / 1.66 | 54.3 / 7.08 | 6.79 / 1.74 | — |
| P5 (CPT-from-stock) | **2.20** / 1.30 | 4.40 / 1.57 | 2.44 / 1.37 | — |

## Read-out (recorded as-is, per the pre-registration)

1. **The from-scratch arms are unusable on real data, and it is the corpus, not the
   recipe.** A/H's *clean* NMSE on Penmanshiel is 76.1/16.9 vs stock's 5.85 — the
   5.7B-point corpus does not cover these domains. This kills any "pretrain from
   scratch with our recipe" deployment story on real data and independently confirms
   why the builder-facing prescription must be CPT-from-stock (P5 keeps stock's
   corpus breadth: its clean NMSE is *better* than stock's on both datasets —
   3.80 vs 5.85 and 2.20 vs 2.99).
2. **P5 on real missingness: medians are at parity, means are worse in the tail.**
   Penmanshiel keep: median 1.95 vs stock 1.82, but mean 67.9 vs 27.5. METR-LA
   linear: median 1.37 vs 1.26, mean 2.44 vs 2.14. The synthetic-grid win does NOT
   transfer to real missingness under the fixed fills available at deployment (keep/
   zero/linear are all poor fills here, and Penmanshiel's `keep` is actively wrong —
   clamped values). Per DESIGN2: "a recipe that wins on the synthetic grid and loses
   on real missingness is reported as such" — the 'best recipe' sentence in the paper
   must be scoped to the synthetic grid + the CPT deployment path, and this table is
   the disclosed counterweight.
3. Real-missingness gains would need a GOOD fill (the closure axis is gated by fill
   quality — exactly the paper's fill-quality-diversity claim); on these datasets the
   best fixed fills are poor, so the retrofit's advantage does not show up. This is
   consistent with, and strengthens, the paper's "the fill is the bottleneck" message.
