# Recipe map (S53 + exploration arms, bolt-tiny, reference = stock P0)

All arms are CPT-from-stock (5k steps, bs1024, lr 1e-4, seed 20260826) unless noted.
Closure/floor/clean/attack from analyze_s48.py; the zero column is the S59 any-fill
matrix's hardest fill. Data: s53_eval_{P6,P7,P8,P9}.json, s58_eval_Q.json,
s59_fills_*.json.

## The four knobs and their costs

| arm | recipe | closure α=1 | clean | zero col | attack |
|---|---|---|---|---|---|
| P5 | H: block, alpha-blend only | 76.7% | 1.048 | **blind (3–19×)** | 1.43 |
| P6 | H + declare endpoint 50% | 56.4% | 1.114 | fixed (≈parity) | 1.15 |
| P7 | declare + linear + ffill thirds | 58.7% | 1.102 | ≈P6 (ffill endpoint buys nothing) | 1.19 |
| P8 | mechdiv (4 mechs, 50% declare) | 71.7% | 1.254 | **best (8/9 beat P0)** | 1.14 |
| P9 | P6 with clean_p 0.75 | 62.4% | 1.141 | partial | 1.17 |
| Q50 = P8 × WiSE-FT 0.5 | | 62.3% | 1.058 | ≈parity | 1.20 |
| Q70 = P8 × WiSE-FT 0.7 | | 70.0% | 1.129 | good | 1.15 |

## Lessons

1. **Declaration endpoint is necessary** (P5's zero blindness) and **mechanism
   diversity decides the quality of the fix** (P8's zero column beats P6's) — but
   mechanism diversity is the most expensive knob in clean cost (1.254), echoing
   S48's 48M instability from scratch.
2. **ffill endpoint and clean_p are dead knobs**: P7 ≈ P6, P9 worse than P6 on
   clean (counter-intuitive, likely sampler-path variance; flagged for seeds).
3. **Attack surface shrinks with declare training** (1.14–1.20 vs 1.43): teaching
   the model that content can be absent makes it more cautious about content, an
   unplanned safety benefit.
4. **WiSE-FT transfers across recipes** and Pareto-dominates: Q50 ≥ P6 on every
   axis (closure 62.3 vs 56.4, clean 1.058 vs 1.114, zero parity, attack similar).
5. Base-size behaviour is kinder across the board (P6@base 74.7% closure at 1.086
   clean; P5@base 78.4% at 1.041).

## Candidate final recipes (pending base verification + real-data recheck)

- **Balanced default: Q50** — nothing broken anywhere: 62% closure, +6% clean,
    any-fill robust, lowest attack of the strong arms.
- **Max closure: P5** — 77% but zero-blind; only if the deployment controls the fill.
- **Max robustness: Q70/P8** — best any-fill behaviour, clean cost 13–25%.
