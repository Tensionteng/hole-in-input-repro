#!/usr/bin/env python
"""S45 corpus prep: download the public Chronos training corpus and cache it as
univariate series for the pretraining PoC.

Anti-leakage: every source the paper evaluates on is excluded -- the 9 benchmark datasets
(ETT h/m 1/2, weather, electricity, traffic, exchange, illness) and the GIFT-Eval
missing-containing sources (kdd_cup_2018, temperature_rain, car_parts, covid_deaths).
This repo carries no ETT, so the exclusions are by subset name.

Each parquet row is a series; every array-valued column is one univariate variable. The
cache is one .npz per subset: a flat float32 array + per-series (start, len) offsets.
"""
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(HERE := os.path.dirname(os.path.abspath(__file__)),
                                     "..", "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "corpus_cache")
REPO = "autogluon/chronos_datasets"

# subsets to use, and roughly how much of each to keep (cap on series)
SUBSETS = {
    "dominick": None,
    "m4_daily": None, "m4_hourly": None, "m4_weekly": None, "m4_monthly": None,
    "m4_quarterly": None, "m4_yearly": None,
    "m5": None,
    "mexico_city_bikes": None,
    "monash_cif_2016": None, "monash_fred_md": None, "monash_hospital": None,
    "monash_london_smart_meters": None,
    "monash_m1_monthly": None, "monash_m1_quarterly": None, "monash_m1_yearly": None,
    "monash_m3_monthly": None, "monash_m3_quarterly": None, "monash_m3_yearly": None,
    "monash_nn5_weekly": None, "monash_pedestrian_counts": None,
    "monash_rideshare": None, "monash_saugeenday": None,
    "monash_tourism_monthly": None, "monash_tourism_quarterly": None,
    "monash_tourism_yearly": None,
    "nn5": None, "ercot": None,
    "solar_1h": None, "solar": None,
    "taxi_1h": None, "taxi_30min": None,
    "uber_tlc_daily": None, "uber_tlc_hourly": None,
    "ushcn_daily": None,
    "wiki_daily_100k": None,
    "wind_farms_daily": None, "wind_farms_hourly": None,
    "weatherbench_daily": None,
    "training_corpus": 20,          # KernelSynth 1M: keep 20 of 220 shards
}
EXCLUDED = {"exchange_rate", "monash_weather", "electricity_15min",
            "monash_electricity_hourly", "monash_electricity_weekly",
            "monash_australian_electricity", "monash_traffic",
            "monash_kdd_cup_2018", "monash_temperature_rain",
            "monash_car_parts", "monash_covid_deaths",
            "weatherbench_hourly", "weatherbench_weekly"}
MIN_LEN = 640                     # need 512 context + 64 horizon + slack


def download():
    from huggingface_hub import snapshot_download
    pats = []
    for sub, cap in SUBSETS.items():
        if sub in EXCLUDED:
            continue
        if sub == "training_corpus":
            pats.append("training_corpus/kernel_synth_1m/**")  # the 1M KernelSynth shards
        else:
            pats.append(f"{sub}/**")
    return snapshot_download(REPO, repo_type="dataset", allow_patterns=pats)


def build_cache(base):
    stats = {}
    for sub in sorted(SUBSETS):
        if sub in EXCLUDED:
            continue
        d = os.path.join(base, sub)
        if sub == "training_corpus":
            d = os.path.join(base, "training_corpus", "kernel_synth_1m")
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
        if SUBSETS[sub] is not None:
            files = files[:SUBSETS[sub]]
        flat, offs = [], []
        n_points = 0
        for f in files:
            df = pd.read_parquet(os.path.join(d, f))
            for col in df.columns:
                if col.lower() in ("item_id", "id", "start", "freq", "timestamp"):
                    continue
                for arr in df[col].to_numpy():
                    if not isinstance(arr, np.ndarray) or arr.dtype == object:
                        continue
                    v = np.asarray(arr, dtype=np.float32).ravel()
                    if len(v) < MIN_LEN:
                        continue
                    offs.append((n_points, len(v)))
                    flat.append(v)
                    n_points += len(v)
        if not offs:
            continue
        flat = np.concatenate(flat)
        np.savez(os.path.join(CACHE, f"{sub}.npz"),
                 flat=flat, offs=np.asarray(offs, dtype=np.int64))
        stats[sub] = {"series": len(offs), "points": int(n_points),
                      "median_len": float(np.median([o[1] for o in offs]))}
        print(f"{sub:32s} series={len(offs):8d} points={n_points:12,}",
              flush=True)
    json.dump(stats, open(os.path.join(CACHE, "stats.json"), "w"), indent=1)
    return stats


def main():
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    print("== download ==", flush=True)
    base = download()
    print(f"downloaded to {base} in {(time.time()-t0)/60:.1f} min", flush=True)
    print("== build cache ==", flush=True)
    build_cache(base)
    print(f"done in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
