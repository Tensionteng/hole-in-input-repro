# S511 — do the standard benchmarks already contain missingness?

Data: `s511_benchmark_missing.json` via `s511_benchmark_missing.py`. Per dataset:
NaN fraction; fraction of points inside zero-runs >= 4 steps; fraction inside
constant plateaus >= 12 identical steps (channels with <= 20 unique values
excluded — binary/count channels plateau by nature, not by fault).

## Findings (reported as-is)

| dataset | NaN | zero-run (mean/max ch) | const-plateau (mean/max ch) |
|---|---|---|---|
| ETTh1 | 0.00% | 0.00 / 0.00% | 2.39 / 3.00% |
| ETTh2 | 0.00% | 0.00 / 0.00% | 10.94 / 35.98% |
| ETTm1 | 0.00% | 0.00 / 0.00% | 2.56 / 3.14% |
| ETTm2 | 0.00% | 0.00 / 0.00% | 17.64 / 60.30% |
| weather | 0.00% | 0.09 / 0.81% | 15.76 / 93.68% |
| electricity | 0.00% | 0.00 / 0.61% | 0.89 / 75.70% |
| traffic | 0.00% | 0.00 / 0.14% | 0.43 / 8.39% |
| exchange | 0.00% | 0.00 / 0.00% | 1.39 / 11.12% |
| illness | 0.00% | 0.00 / 0.00% | 0.00 / 0.00% |

## What this supports, and what it does not

- **Supported**: the published benchmark files are NaN-free because the missingness
  was silently cleaned upstream (interpolated or dropped before release) — the
  evaluation pipeline has no missingness semantics at all. Separately, several
  datasets carry long constant plateaus even after excluding discrete channels
  (ETTm2's worst continuous channel is flat 60% of the time; weather 94%;
  electricity 76%) — the fingerprint of stuck/faulted sensors, i.e. *undeclared*
  missingness sitting inside the "clean" benchmark.
- **NOT supported**: a claim like "standard benchmarks contain X% NaN" — they
  contain none as published. The paper's evidence for real-world missingness rests
  on the two real datasets (Penmanshiel curtailment, METR-LA outages) and on the
  silent-cleaning observation above, not on benchmark NaN counts.
