# S53–S56 roundup: the CPT-from-stock round (2026-08-26/27)

Everything below is pre-registered-then-run; per-stage notes carry the details.
Nothing here is in the paper yet (held back per instruction).

| stage | question | answer | artefacts |
|---|---|---|---|
| S53 | Can a shipped ckpt be retrofitted? | Yes: 5k CPT steps on the H recipe recovers 73–85% of the stock ceiling's excess | `DESIGN3.md` (prereg), `train_s53.py`, `s53_notes.md`, `s53_eval_{a,b,seeds_a,seeds_b}.json` |
| S53 seeds | Is it stable? | Yes: P5 75.7% ± 2.1, P3 82.0% ± 4.0 (3 seeds). R4 revised: stage-1 adapter = small consistent gain. R5 revised: clean cost 5–7%, just outside the ±5% gate | `s53_notes.md` (updated scorecard) |
| S54 | Leaderboard-style main table? | Perfect fill: ours 9/9. Deployment fill: ours 6/9 + 2 ties (tiny); 9/9 (base) | `s54_notes.md`, `main_table.tex`, `make_main_table.py`, `s54_sweep9_{a,b}.json` |
| S55 | Real missingness (DESIGN2 axis 2)? | From-scratch arms unusable off-corpus (clean NMSE 76/17 vs 5.9) → CPT is THE deployment path. P5: medians at parity with stock, means worse in the tail — the synthetic win needs a good fill, which real deployments lack. 'Best recipe' claims must be scoped | `s55_notes.md`, `s55_realfill.json`, `eval_s55_realfill.py` |
| S56 | Does it scale to 205M? | Yes: bolt-base closure 78.4% [+71,+82], floor 18/24, clean 1.041 (inside gate), attack 1.43×, Panel-B 9/9 | `s56_notes.md`, `s56_eval_base.json`, `arm_P{0,5}_base.pt` |

## Checkpoints (all in `s45_ckpt/`)

- `arm_P0_{tiny,base}.pt` — stock dumps (G2′ = 0 vs stock forward); `arm_P1_tiny.pt`
  symlinks P0 (the untrained restored-interface arm)
- `arm_P{2,3,4,5}_tiny.pt` — S53 arms, seed 20260826
- `arm_P{2,3,4,5}_tiny_s{20260901,20260902}.pt` — replicate seeds
- `arm_P5_base.pt` — the 205M retrofit

## The builder prescription as it now stands

Restore the dual interface, then 5k steps of full-model CPT on the H regime (block
outages, alpha-blend fill); optionally prepend the 600-step embedding-only adapter
for ~6 more closure points at a wider seed spread. Recovers ~76–82% of the ceiling,
floor 17–18/24, clean cost 4–7%, attack surface 1.33–1.43×, verified at 8.7M (3
seeds) and 205M (1 seed). On real missingness the gain shows only when the deployed
fill is good — the fill remains the bottleneck, which is the paper's point.

## Known loose ends for the paper pass (not done, per instruction)

- `M_ladder.tex`: the H@small row still divides by A@tiny (recipe × capacity
  conflation); correct same-size numbers are in s48_eval_small.json (H@small 76.3%
  [+71,+89], floor 9/24, clean 0.929 vs A@small).
- S55's parity-on-real-missingness table must accompany any "best recipe" sentence.
- R5's ±5% gate was narrowly missed at tiny (5.3% median) but held at base (4.1%) —
  the main text should quote the range, not the gate.
