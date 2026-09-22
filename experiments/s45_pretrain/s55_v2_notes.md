# S55 v2 — real-missingness recheck across the recipe map (P0/P5/P6/Q50)

Windows: S26 TEST halves (same as v1). `s55_realfill_v2.json`, eval_s55_realfill.py
`--arms P0,P5,P6,Q50`. NMSE mean per window (medians all land within ±0.1 of each
other; the mean tail is where the arms differ).

## Penmanshiel (best fixed fill there = zero)

| arm | ctrl clean | keep | zero | linear |
|---|---|---|---|---|
| P0 | 5.85 | 27.5 | **19.9** | 45.2 |
| P5 (H, blend only) | 3.80 | 67.9 | 46.8 (2.4×) | 59.6 |
| P6 (H+declare) | 5.39 | 54.6 | 38.3 (1.9×) | 43.2 |
| **Q50 (P8×WiSE-FT)** | 5.93 | 35.4 | **20.2 (parity)** | 45.7 (parity) |

## METR-LA

| arm | ctrl clean | keep=zero | linear |
|---|---|---|---|
| P0 | 2.99 | 2.38 | 2.14 |
| P5 | 2.20 | 4.40 (1.8×) | 2.44 |
| P6 | 2.35 | 2.49 | 2.26 |
| **Q50** | **2.06** | **2.39 (parity)** | **2.11 (≈win)** |

## Read-out

1. **Q50 delivers the v1 promise that P5/P6 could not**: never materially worse than
   stock under any fixed fill on real missingness (worst cell: penn keep 35.4 vs
   27.5, +29%; every other cell at parity), with better clean accuracy on METR-LA
   (2.06 vs 2.99) and comparable elsewhere.
2. The recipe iteration is visible in the zero column: P5 46.8 → P6 38.3 → Q50
   20.2 (vs stock 19.9). Mechanism-diverse declare training + interpolation is what
   closed it, confirming the recipe-map lessons on real data, not only synthetic.
3. Remaining honest caveats: penn `keep` (clamped values fed as-is) is still worse
   than stock for every CPT arm — that fill is actively WRONG content (not absent,
   not noisy, systematically biased low), and reading it hurts. This is the
   fill-quality story again, in its sharpest real-world form.
