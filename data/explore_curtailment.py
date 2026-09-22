#!/usr/bin/env python
"""Explore Kelmarsh 2016 SCADA: quantify curtailment censoring.

Detection rule (documented in DATA_README.md):
  PRIMARY: Greenbyte's own "Lost Production to Curtailment (Total) (kWh)" > 0
           for the 10-min interval (the operator's curtailment ledger).
  CROSS-CHECKS: "Turbine Power setpoint (kW)" < rated (2050),
           and actual power well below "Potential power default PC (kW)".
Outputs: dataset_censor/processed_kelmarsh2016.npz with per-turbine regular
10-min grids (power, wind, setpoint, curtail flag), plus stats printed and
saved to dataset_censor/censor_stats.json. Validation plots -> validate_*.png
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCADA_DIR = os.path.join(HERE, "dataset_censor", "scada_2016")
OUT_NPZ = os.path.join(HERE, "dataset_censor", "processed_kelmarsh2016.npz")
OUT_STATS = os.path.join(HERE, "dataset_censor", "censor_stats.json")
RATED = 2050.0

COLS = {
    "Date and time": "time",
    "Wind speed (m/s)": "wind",
    "Power (kW)": "power",
    "Potential power default PC (kW)": "potential",
    "Turbine Power setpoint (kW)": "setpoint",
    "Lost Production to Curtailment (Total) (kWh)": "lost_curt",
    "Lost Production to Curtailment (Grid) (kWh)": "lost_curt_grid",
}


def load_turbine(path):
    df = pd.read_csv(path, skiprows=9, low_memory=False)
    df = df.rename(columns={df.columns[0]: "Date and time"})
    have = {c: v for c, v in COLS.items() if c in df.columns}
    df = df[list(have)].rename(columns=have)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    # regular 10-min grid over the file's stated span
    full = pd.date_range(df.index.min().floor("10min"),
                         df.index.max().ceil("10min"), freq="10min")
    df = df.reindex(full)
    return df


def main():
    all_stats = {}
    store = {}
    for path in sorted(glob.glob(os.path.join(SCADA_DIR, "Turbine_Data_*.csv"))):
        turb = "T" + os.path.basename(path).split("_")[2]
        df = load_turbine(path)
        n = len(df)
        curt = (df["lost_curt"].fillna(0) > 0)
        curt_grid = (df["lost_curt_grid"].fillna(0) > 0)
        sp_low = df["setpoint"] < RATED - 1e-6
        under = (df["potential"] - df["power"]) > 0.1 * RATED
        obs = df["power"].notna() & df["wind"].notna()
        hiwind = obs & (df["wind"] >= 10.0)
        st = dict(
            n_intervals=n,
            power_obs_rate=float(df["power"].notna().mean()),
            wind_obs_rate=float(df["wind"].notna().mean()),
            curtailed_frac_all=float(curt.mean()),
            curtailed_frac_obs=float(curt[obs].mean()),
            curtailed_grid_frac_obs=float(curt_grid[obs].mean()),
            setpoint_below_rated_frac=float(sp_low[df["setpoint"].notna()].mean()),
            hiwind_frac_of_obs=float(hiwind[obs].mean()),
            curtailed_given_hiwind=float(curt[hiwind].mean()) if hiwind.any() else None,
            agree_curt_vs_setpoint=float((curt == sp_low)[obs].mean()),
            curt_and_under_frac=float((curt & under)[obs].mean()),
        )
        # episode-length distribution of the curtail flag (on the regular grid)
        flag = curt.to_numpy()
        runs, cur = [], 0
        for v in flag:
            cur = cur + 1 if v else 0
            if not v and cur == 0:
                continue
            if not v:
                runs.append(cur); cur = 0
        if cur:
            runs.append(cur)
        runs = np.array(runs) if runs else np.array([0])
        st["episodes"] = int((runs > 0).sum())
        st["episode_len_median"] = float(np.median(runs[runs > 0])) if (runs > 0).any() else 0
        st["episode_len_max"] = int(runs.max())
        all_stats[turb] = st
        for k in ["power", "wind", "setpoint", "potential"]:
            store[f"{turb}_{k}"] = df[k].to_numpy(np.float32)
        store[f"{turb}_curt"] = curt.to_numpy()
        store[f"{turb}_curt_grid"] = curt_grid.to_numpy()
        store[f"{turb}_time"] = df.index.astype("datetime64[s]").to_numpy()
        print(turb, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in st.items()}))
    np.savez_compressed(OUT_NPZ, **store)
    with open(OUT_STATS, "w") as f:
        json.dump(all_stats, f, indent=2)
    print("saved", OUT_NPZ, "and", OUT_STATS)


if __name__ == "__main__":
    main()
