# S57 — the missingness leaderboard

Grid: 9 benchmarks × 4 mechanisms × rate 0.7 × fills {zero, linear, oracle}, each
model on its OWN declared path, 150 windows/cell, relMSE vs own clean. Data:
`s57_{bolt,moirai2,tirex,flowstate,timerxl_plain,q50}.json`, eval_s57_leaderboard.py.

## Headline: Q50 is the ONLY model that is non-trivial on all three fills

| model | oracle | linear | zero | reads fill content? |
|---|---|---|---|---|
| **Q50-base (ours)** | 1.08–1.73 | 1.10–4.53 | **1.56–7.18** | yes |
| P5-base (ours, blend-only) | 1.07–1.42 | 1.10–4.32 | blind (7–520) | yes |
| P8-base (ours, no interp) | 1.12–3.28 | 1.04–4.41 | 1.27–7.24 | yes |
| Moirai 2.0 | 1.04–1.84 | 1.20–5.42 | **blind (4.8–82)** | yes (its E-recipe) |
| bolt-base stock | 1.30–4.88 | 1.31–5.75 | 2.21–46.48 | no (ceiling) |
| **TimesFM 2.5** | **fill-invariant** 1.13–19.29 | = | = | **no (overwrites: all fills identical — its own interpolant is the only one it ever uses; illness 19.29)** |
| TiRex | fill-invariant 1.34–6.82 | = | = | no (fill-blind) |
| FlowState | fill-invariant 1.09–5.13 | = | = | no (fill-blind) |
| Timer-XL | declared path emits NaN; plain path shown (trivial 1.00 on oracle) | | | no interface |

## Column verdicts

- **oracle**: P5 wins 4 (ETTm1/ETTh2/ETTm2/traffic), Moirai 2.0 wins 4
  (ETTh1/weather/electricity/exchange), Q50 wins 1 (illness). Ours and Moirai split
  the column; every other model loses by 2–5×.
- **linear**: ours (Q50/P5/P8) win 7–8 of 9; Moirai 2.0 only contests ETTh1.
- **zero**: Q50 beats stock bolt-base 9/9 and Moirai 2.0 9/9. TiRex/FlowState's
  "normal" zero numbers are their fill-BLINDNESS (identical on all fills — they
  lose the oracle column by up to 5×); TimesFM's fill-invariance is the overwrite
  convention made visible: every fill is replaced by its own interpolant, so it
  cannot benefit from a good fill either (oracle = zero on every column) — and its
  interpolant is not even good (illness 19.29, worse than stock's ceiling 2.02);
  Timer-XL has no declared path at all.

## What the table proves

Every shipped model sits in exactly one failure mode: fill-blind (TiRex,
FlowState), interface-without-declare-training (Moirai 2.0: reads content but zero
is OOD; bolt-stock: full ceiling), or broken interface (Timer-XL). The retrofit is
the only entry that reads content AND survives every fill — the paper's problem
statement and prescription in one table.

## Fair-budget control (P4@base, added 2026-08-28)

P4 = stock + the same 5k CPT steps on our corpus WITHOUT missingness (native
interface, filtered regime). On the leaderboard grid:

- oracle column: P4 ≈ P0 (1.31–4.19) — continued training alone does not break
  or fix the ceiling.
- zero column: **P4 collapses (4.74–393, exchange 393.4 vs stock's 46.5)** —
  continued training on a filtered corpus ERODES the stock model's native
  zero-fill robustness. Q50 (7.18 on exchange) is 55× better than the fair
  control and 6× better than stock. The retrofit's advantage is therefore
  larger under the fair training-budget comparison, not smaller.

## Caveats

- timerxl-plain's oracle=1.00 is the trivial no-interface baseline (filled truth =
  clean input), not a win; shown for completeness.
- Clean MSEs are stored per model per dataset; Q50-base's clean is within ~5% of
  stock bolt-base (median), Timer-XL's clean is far off the field (weather 44.5).
- Single seed for the P8@base run behind Q50@base.
