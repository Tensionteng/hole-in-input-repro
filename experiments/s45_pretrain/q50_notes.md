# Q50 — the candidate final model, three seeds (bolt-tiny)

Q50 = P8 (mechdiv CPT: 4 mechanisms, 50% declare + 50% alpha-blend, 5k steps from
stock) × WiSE-FT 0.5 (equal interpolation with the stock weights). Seeds
20260826/20260901/20260902. Data: s58_eval_Q{,_s01,_s02}.json.

## Scorecard (reference = stock P0)

| seed | ρ_declared | closure α=1 | 95% CI | floor α=0 | clean | attack |
|---|---|---|---|---|---|---|
| 20260826 | 0.748 | 62.3% | [+48,+66] | 18/24 | 1.058 | 1.20× |
| 20260901 | 0.753 | 59.9% | [+50,+69] | 17/24 | 1.070 | 1.18× |
| 20260902 | 0.730 | 62.8% | [+54,+71] | 17/24 | 1.044 | 1.18× |
| **spread** | | **61.7% ± 1.6** | | | 1.044–1.070 | |

## zero-fill column across seeds (the S59 hardest fill; P0 row for reference)

Per-dataset values are near-identical across seeds (e.g. exchange 6.79/7.17/6.29,
weather 1.43/1.42/1.46). In every seed only 1–2 of 9 columns sit >2% above P0
(ETTh2/ETTm2/illness, all ≤ 1.15×), the rest are wins — exchange ~2× better,
weather ~1.25×, electricity ~1.2×.

## Verdict

The balanced recipe is seed-stable on every axis: closure 61.7% ± 1.6, clean cost
4.4–7.0%, any-fill robustness (including zero) with ≤2 small losses per seed, and
a reduced attack surface (1.18–1.20× vs the blend-only recipe's 1.43×). Together
with the S57 leaderboard row (Q50@base) and the S55 v2 real-data parity, this is
the model the paper's prescription should point at.

## Base-size leaderboard rows, three seeds (S57 grid, vs stock bolt-base)

P8@base replicates landed and were interpolated identically (×0.5 with stock).
Data: s57_q50{,_s01,_s02}.json.

- **oracle column: 9/9 wins vs stock in EVERY seed** — ranges 1.06–1.73 across
  seeds vs stock's 1.30–4.88; seed spread per dataset ≤ 0.1.
- **zero column: 9/9 wins vs stock in EVERY seed** — 1.56–7.25 vs 2.21–46.48,
  including the columns (ETTh2/ETTm2/illness) that were small losses at tiny.
  The base-size any-fill robustness is strictly cleaner than tiny's.

Base-size seeds: P8@base replicates (20260901/20260902) are training now; the
Q50@base leaderboard row will be re-read per seed when they land.
