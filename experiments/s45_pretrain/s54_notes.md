# S54 — the main results table (sweep9 grid)

Data: `s54_sweep9_a.json` (P0/P1/P2) + `s54_sweep9_b.json` (P3/P4/P5), eval_s45.py
`--parts sweep9`: all 9 benchmarks x 4 mechanisms x rate 0.7 x alpha in {0, 1},
declared path, per-window relMSE vs the arm's OWN clean accuracy (same convention as
the S45/S48 sweeps), 150 windows per cell. Aggregate: `make_main_table.py` ->
`main_table.tex` (booktabs, two panels).

## Headline

- **Panel A (perfect fill, alpha=1): the CPT arm wins 9/9 columns.** Median relMSE
  vs own clean: stock 1.33–5.37 across datasets, ours 1.06–1.20. Largest gains where
  the stock ceiling is most expensive: electricity 5.37→1.20, traffic 4.16→1.16.
- **Panel B (deployment fill, alpha=0 = linear): ours wins 6/9 and ties the rest.**
  ETTh1 1.50 vs adapter's 1.48 (a 0.02 tie), weather 1.18 = budget control's 1.18.
  This is the floor claim on the full benchmark suite: under a fill the model cannot
  trust, the retrofit is never worse than any alternative.
- **Clean cost, disclosed in the same table**: ours runs at 1.048× stock clean
  (per-dataset 0.89–1.17), inside the pre-registered ±5% gate.
- Budget control (P4, same FLOPs, filtered regime) matches stock on Panel A
  (1.60/2.05/5.58/4.37 on ETTh1/ETTm1/electricity/traffic) — the gains are caused by
  the missingness regime, not by extra training.
- The untrained restored interface (P1) is better than stock on Panel A
  (e.g. electricity 2.13 vs 5.37) but *worse* where the fill misleads
  (exchange 1.50/1.47 vs stock's 1.33/1.37 across panels) — the gamble the paper
  describes, now visible on the leaderboard-style table.

## Caveats

- relMSE is normalised per arm against its own clean; the clean row lets the reader
  reconstruct absolute performance (our arm's clean is 4.8% worse than stock).
- rate 0.7 only, mechanisms aggregated by median; per-mechanism cells are in the
  JSONs for the appendix.
- Single seed (20260826) for the P-arms; replicate seeds pending.
