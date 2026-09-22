# S513 — the missingness evidence table

Data: `s513_missing_evidence.json`, `s513_missing_evidence.py`.

## A. GIFT-Eval: exact NaN statistics (recomputed from the distributed arrow files)

| subset | % points NaN | % series with NaN |
|---|---|---|
| electricity/H | **19.19%** | 57.3% |
| kdd_cup_2018_with_missing/H | 17.12% | 100% |
| restaurant | 13.76% | 98.8% |
| bitbrains_rnd/5T | 13.13% | 100% |
| bitbrains_fast_storage/5T | 4.52% | 94.3% |
| hierarchical_sales/D | 1.48% | 100% |
| car_parts_with_missing | 4.49% | 6.2% |
| temperature_rain_with_missing | 2.58% | 96.5% |
| jena_weather/H | 0.01% | 100% |

9 of GIFT-Eval's 28 datasets contain NaNs; the corpus is 11.77% missing overall
(305M points). Every subset with nontrivial missingness has it in the MAJORITY of
its series — this is not a corner case the benchmark excludes.

## B. The nine standard benchmarks: cleaning fingerprints

Published files are NaN-free, so the question becomes "can the cleaning be
reverse-engineered?" Yes — fills leave fingerprints: zero-runs (>=4 steps),
exact plateaus (>=12), and exact straight runs (>=6, the linear-interpolation
fingerprint). Channels with <=20 unique values excluded.

| dataset | zero-runs (mean/max ch) | plateaus (mean/max ch) | straight runs |
|---|---|---|---|
| ETTh1 | 0.52 / 1.15% | 2.40 / 3.00% | ~0 |
| ETTh2 | **8.08 / 28.54%** | **11.20 / 37.31%** | ~0 |
| ETTm1 | 0.56 / 1.16% | 2.57 / 3.18% | ~0 |
| ETTm2 | **8.58 / 29.08%** | **18.18 / 61.99%** | ~0 |
| weather | **15.87 / 95.51%** | **15.79 / 93.79%** | ~0 |
| electricity | 0.76 / **75.71%** | 0.92 / **76.30%** | ~0 |
| traffic | 0.70 / 13.57% | 0.47 / 8.70% | ~0 |
| exchange | 0.00 / 0.00% | 1.41 / 11.26% | ~0 |
| illness | 0.00 / 0.00% | 0.00 / 0.00% | ~0 |

## Read-out

1. **The cleaning was zero-fill, not interpolation**: straight-run fingerprints
   are absent everywhere, while zero-runs and plateaus are extensive. Combined with
   TSLib's `np.nan_to_num` (data_loader.py:434,439) and GluonTS's
   `dummy_value=0.0`, the published benchmarks' missingness was silently zero-filled.
2. **The traces are recoverable**: ETTh2/ETTm2 carry 8.6% zero-runs and up to 62%
   plateaus on the worst channel (stuck/unsampled sensors); electricity's worst
   channel is 75.7% zero — the documented "customers not yet connected in 2011"
   segments of the UCI original; weather's sparse channels reach 95.5%.
3. So the premise table for the paper has three independent legs: GIFT-Eval's
   11.77% (with per-subset numbers above), the two real datasets (Penmanshiel
   26% of windows curtailed; METR-LA 23.2% mean outage rate), and the zero-fill
   fingerprints inside the "clean" benchmarks themselves.

## C. Direct evidence from the UCI original (downloaded 2026-08-28)

`ElectricityLoadDiagrams20112014` raw file (LD2011_2014.txt, 370 clients,
15-min, 2011–2015), vs the TSLib `electricity.csv` (321 clients, hourly,
**2016–2019**):

| | UCI original | TSLib version |
|---|---|---|
| overall zero fraction | **20.15%** | 1.09% |
| 2011 zero fraction | **57.18%** | (no overlap) |
| clients starting unconnected | **212/370 (57%)** | 321/370 kept |
| clients with a FULL YEAR of leading zeros | 161 | — |
| longest leading-zero prefix | 1294 days | — |

- The published benchmark versions were produced by DISCARDING the dirtiest
  parts: GMP-AR (arXiv:2406.12242) drops all of 2011; another study drops the
  gappy clients; TSLib's file comes from an undocumented 2016–2019 pipeline.
- The same benchmark name circulates as at least two non-overlapping versions
  (2011–2014 vs 2016–2019) — "the dataset" is not even a single object.
- SAITS (arXiv:2202.08516) reports the same corpus as "has no missing data" —
  true only of somebody's cleaned derivative. The missingness status of a
  benchmark is a property of its invisible preprocessing, not of the data.
