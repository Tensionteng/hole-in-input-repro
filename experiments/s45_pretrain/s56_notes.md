# S56 — CPT-from-stock at bolt-BASE (205M): does the retrofit scale?

Setup: identical to S53 (P5 arm, one-stage CPT on the H recipe, 5k steps bs1024
lr 1e-4, seed 20260826), but initialised from the stock **bolt-base** checkpoint
(205M, from the HF cache; `arm_P0_base.pt` is its dump, G2′ max|diff| = 0).
Training took 43 min on one A800. Eval: `s56_eval_base.json` (eval_s45.py
`--size base`, clean+probe+sweep+attack+sweep9, same grids as tiny).

## Scorecard at base (reference = stock bolt-base)

| arm | ρ_declared | closure α=1 | 95% CI | floor α=0 | clean vs P0 | attack |
|---|---|---|---|---|---|---|
| P0 (stock bolt-base) | 0.000 | — | — | — | 1.000 | 1.05× |
| P5 (CPT, H recipe) | 0.832 | 78.4% | [+71, +82]% | 18/24 | 1.041 | 1.43× |

Tiny reference (S53, P5): closure 76.7% [+66,+82], floor 18/24, clean 1.048,
attack 1.43×. **The retrofit transfers to 205M essentially unchanged**; the clean
cost is slightly SMALLER at base (4.1%, inside the ±5% gate that tiny narrowly
missed).

## Main-table rows at base (9 benchmarks, sweep9 grid)

Panel A (perfect fill): ours wins 9/9 — stock 1.30–4.88, ours 1.07–1.42.
Panel B (deployment fill): ours wins **9/9** at base (tiny won 6/9 with two ties);
largest margins electricity 5.75→4.32, traffic 4.09→3.55.
Clean cost per dataset: 0.75–1.17, median 1.041.

## Read-out

- The S53 story is not a tiny-model artefact: same closure, same floor, same attack
  ratio, smaller clean cost, at 23× the parameters. The builder prescription now
  holds at both ends of the bolt family used in the paper.
- One seed at base (cost); direction matches the tiny 3-seed spread
  (73.3–77.2%) with the base point (78.4%) just above it.
