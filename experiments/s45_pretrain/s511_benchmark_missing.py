#!/usr/bin/env python
"""S511: how much missing/anomalous content do the STANDARD benchmarks already have?

The reviewer's question is "why should a benchmark audience care about missingness?"
Answer: the nine standard forecasting benchmarks themselves contain NaNs, long zero
runs (sensor outages read as zeros) and constant plateaus (stuck sensors) -- i.e.
deployment missingness is already inside the evaluation, it is just silently
mishandled (usually dropped or zero-filled by the dataloader).

For every dataset x channel: NaN fraction, exact-zero run fraction (runs >= 4 steps),
constant-plateau fraction (runs >= 12 identical steps). Reported per dataset as the
max over channels (the worst channel is what a univariate window sampler hits) and
the mean.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TSLIB = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                     "legacy_nonstationary", "tslib", "dataset"))
DATASETS = {
    "ETTh1": "ETT-small/ETTh1.csv", "ETTh2": "ETT-small/ETTh2.csv",
    "ETTm1": "ETT-small/ETTm1.csv", "ETTm2": "ETT-small/ETTm2.csv",
    "weather": "weather/weather.csv", "electricity": "electricity/electricity.csv",
    "traffic": "traffic/traffic.csv", "exchange": "exchange_rate/exchange_rate.csv",
    "illness": "illness/national_illness.csv",
}


def run_fraction(v, kind, min_run):
    """Fraction of points inside runs of length >= min_run of the given kind."""
    if kind == "zero":
        b = v == 0.0
    elif kind == "const":
        b = np.r_[True, v[1:] == v[:-1]]
    else:
        raise ValueError(kind)
    n = len(b)
    inside = np.zeros(n, bool)
    i = 0
    while i < n:
        if kind == "zero":
            j = i + 1
            while j < n and b[j]:
                j += 1
            if b[i] and j - i >= min_run:
                inside[i:j] = True
            i = j
        else:  # const: b marks "same as previous"
            j = i
            while j < n and b[j]:
                j += 1
            if j - i >= min_run:
                inside[i:j] = True
            i = max(j, i + 1)
    return float(inside.mean())


def main():
    out = {}
    for name, rel in DATASETS.items():
        df = pd.read_csv(os.path.join(TSLIB, rel))
        df = df.drop(columns=[c for c in df.columns
                              if c.lower() in ("date", "unnamed: 0")])
        X = df.to_numpy(np.float32)
        nan_frac = float(np.isnan(X).mean())
        zr, cr = [], []
        n_discrete = 0
        for c in range(X.shape[1]):
            v = X[:, c]
            v = v[np.isfinite(v)]
            if len(v) < 100:
                continue
            if len(np.unique(v)) <= 20:      # binary/count channels: plateaus are
                n_discrete += 1              # their nature, not sensor faults
                continue
            zr.append(run_fraction(v, "zero", 4))
            cr.append(run_fraction(v, "const", 12))
        out[name] = {
            "n_rows": int(X.shape[0]), "n_channels": int(X.shape[1]),
            "n_discrete_channels_excluded": n_discrete,
            "nan_frac": nan_frac,
            "zero_run_frac_mean": float(np.mean(zr)) if zr else 0.0,
            "zero_run_frac_max": float(np.max(zr)) if zr else 0.0,
            "const_run_frac_mean": float(np.mean(cr)) if cr else 0.0,
            "const_run_frac_max": float(np.max(cr)) if cr else 0.0,
        }
        o = out[name]
        print(f"{name:12s} nan={100*o['nan_frac']:5.2f}%  "
              f"zero-run(mean/max)={100*o['zero_run_frac_mean']:5.2f}/"
              f"{100*o['zero_run_frac_max']:5.2f}%  "
              f"const-plateau(mean/max)={100*o['const_run_frac_mean']:5.2f}/"
              f"{100*o['const_run_frac_max']:5.2f}%", flush=True)
    json.dump(out, open(os.path.join(HERE, "s511_benchmark_missing.json"), "w"),
              indent=1)
    print("\nwrote s511_benchmark_missing.json")


if __name__ == "__main__":
    main()
