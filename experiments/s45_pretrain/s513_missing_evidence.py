#!/usr/bin/env python
"""S513: the missingness evidence table for the paper's premise.

Two blocks:

A. GIFT-Eval: exact NaN statistics per missing-containing subset (points and
   series), recomputed from the distributed arrow files.

B. The nine standard benchmarks: CLEANING FINGERPRINTS. Published files are
   NaN-free, but upstream cleaning leaves traces:
     - zero-runs (>= 4 steps)          -> zero-fill fingerprint
     - exact plateaus (diff == 0, >=12) -> forward-fill / stuck sensor
     - exact straight runs (diff2 == 0, >= 6, with nonzero slope)
                                          -> linear-interpolation fingerprint
   Channels with <= 20 unique values are excluded (discrete by nature).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow as pa

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GIFT = os.path.join(ROOT, "tsfm_missing", "data", "gifteval")
TSLIB = os.path.join(ROOT, "legacy_nonstationary", "tslib", "dataset")

GIFT_MISSING = ["electricity/H", "kdd_cup_2018_with_missing/H", "restaurant",
                "bitbrains_rnd/5T", "bitbrains_fast_storage/5T", "hierarchical_sales/D",
                "car_parts_with_missing", "temperature_rain_with_missing",
                "jena_weather/H"]
DS9 = {
    "ETTh1": "ETT-small/ETTh1.csv", "ETTh2": "ETT-small/ETTh2.csv",
    "ETTm1": "ETT-small/ETTm1.csv", "ETTm2": "ETT-small/ETTm2.csv",
    "weather": "weather/weather.csv", "electricity": "electricity/electricity.csv",
    "traffic": "traffic/traffic.csv", "exchange": "exchange_rate/exchange_rate.csv",
    "illness": "illness/national_illness.csv",
}


def gift_stats():
    out = {}
    for dsfreq in GIFT_MISSING:
        d = os.path.join(GIFT, dsfreq)
        n_pts = n_nan = n_ser = n_ser_nan = 0
        for f in sorted(glob.glob(os.path.join(d, "*.arrow"))):
            with pa.memory_map(f, "rb") as src:
                t = pa.ipc.open_stream(src).read_all()
            for s in t.column("target").to_pylist():
                a = np.asarray(s, dtype=np.float32)
                rows = [a] if a.ndim == 1 else [a[i] for i in range(a.shape[0])]
                for r in rows:
                    n_ser += 1
                    n_pts += r.size
                    k = int(np.isnan(r).sum())
                    n_nan += k
                    n_ser_nan += k > 0
        out[dsfreq] = {"n_points": n_pts, "nan_points": n_nan,
                       "nan_frac": n_nan / max(n_pts, 1),
                       "n_series": n_ser, "series_with_nan": n_ser_nan,
                       "series_nan_frac": n_ser_nan / max(n_ser, 1)}
        o = out[dsfreq]
        print(f"{dsfreq:35s} {100*o['nan_frac']:6.2f}% of points | "
              f"{100*o['series_nan_frac']:6.2f}% of series "
              f"({o['n_series']} series, {o['n_points']} pts)", flush=True)
    return out


def run_fracs(v):
    """(zero-run frac, plateau frac, straight-line frac) for one channel."""
    d1 = np.diff(v)
    d2 = np.diff(v, 2)
    def frac(mask_bool, min_run):
        b = np.asarray(mask_bool, bool)
        d = np.diff(np.r_[False, b, False].astype(np.int8))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        inside = np.zeros(len(b), bool)
        for s, e in zip(starts, ends):
            if e - s >= min_run:
                inside[s:e] = True
        return float(inside.mean())
    z = frac(v == 0.0, 4)
    p = frac(d1 == 0.0, 11)              # d1-run 11 == 12-point plateau
    # straight run: d2 == 0 AND the segment is not flat (d1 != 0 somewhere inside)
    s_raw = frac(d2 == 0.0, 4)           # d2-run 4 == 6-point straight segment
    p_raw = frac(d1 == 0.0, 5)           # d1-run 5 == 6-point plateau
    s = max(0.0, s_raw - p_raw)          # plateaus are d2==0 too; remove them
    return z, p, s


def bench_stats():
    out = {}
    for name, rel in DS9.items():
        df = pd.read_csv(os.path.join(TSLIB, rel))
        df = df.drop(columns=[c for c in df.columns
                              if c.lower() in ("date", "unnamed: 0")])
        X = df.to_numpy(np.float32)
        zs, ps, ss = [], [], []
        for c in range(X.shape[1]):
            v = X[:, c]
            v = v[np.isfinite(v)]
            if len(v) < 100 or len(np.unique(v)) <= 20:
                continue
            z, p, s = run_fracs(v)
            zs.append(z); ps.append(p); ss.append(s)
        out[name] = {"zero_run_mean": float(np.mean(zs)),
                     "zero_run_max": float(np.max(zs)),
                     "plateau_mean": float(np.mean(ps)),
                     "plateau_max": float(np.max(ps)),
                     "straight_mean": float(np.mean(ss)),
                     "straight_max": float(np.max(ss))}
        o = out[name]
        print(f"{name:12s} zero {100*o['zero_run_mean']:5.2f}/{100*o['zero_run_max']:5.2f}%  "
              f"plateau {100*o['plateau_mean']:5.2f}/{100*o['plateau_max']:5.2f}%  "
              f"straight {100*o['straight_mean']:5.2f}/{100*o['straight_max']:5.2f}%",
              flush=True)
    return out


if __name__ == "__main__":
    print("== A. GIFT-Eval missing subsets ==")
    a = gift_stats()
    print("\n== B. benchmark cleaning fingerprints (mean/max over continuous channels) ==")
    b = bench_stats()
    json.dump({"gifteval": a, "benchmarks": b},
              open(os.path.join(HERE, "s513_missing_evidence.json"), "w"), indent=1)
    print("\nwrote s513_missing_evidence.json")
