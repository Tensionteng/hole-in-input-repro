#!/usr/bin/env python
"""Build the processed Penmanshiel-2016 censoring dataset.

Reads the two Penmanshiel 2016 SCADA zips (WT01-10, WT11-15) under
dataset_censor/, and for each of the 14 turbines produces a regular 10-min
grid with:
  power   : "Power (kW)" 10-min mean (NaN where not observed)
  wind    : "Wind speed (m/s)" 10-min mean
  curt    : documented curtailment flag from the turbine's Status event log:
            intervals overlapping an event with Code 9000 ("P output
            externally reduced", IEC category "Partial Performance") or
            Code 9210 ("Externally stopped"). This is the grid operator
            actively clamping/stopping output -- the documented censoring.
  avail   : empirical available-power curve (q95 of power per 0.5 m/s wind
            bin over the turbine-year; robust to curtailment)
  cens    : EFFECTIVE censor mask = curt & power observed &
            (avail(wind) - power > 0.15*rated) & (avail(wind) > 0.30*rated)
            i.e. the clamp is binding and the censored value was a HIGH one.

Outputs dataset_censor/penmanshiel2016_processed.npz and
dataset_censor/censor_stats_penmanshiel2016.json.
"""
import glob
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DCDIR = os.path.join(HERE, "dataset_censor")
RATED = 2050.0
CURT_CODES = {"9000", "9210"}
BIND_MARGIN = 0.15 * RATED     # avail - power margin for the clamp to count as binding
HIGH_AVAIL = 0.30 * RATED      # censored value must have been at least this available power

OUT_NPZ = os.path.join(DCDIR, "penmanshiel2016_processed.npz")
OUT_STATS = os.path.join(DCDIR, "censor_stats_penmanshiel2016.json")


def read_member(zf, name, **kw):
    with zf.open(name) as fh:
        return pd.read_csv(io.BytesIO(fh.read()), **kw)


def process(zip_path):
    out = {}
    zf = zipfile.ZipFile(zip_path)
    names = zf.namelist()
    turb_data = sorted(n for n in names if n.startswith("Turbine_Data_"))
    for td in turb_data:
        tag = td.split("_")[2] + "_" + td.split("_")[3]  # e.g. Penmanshiel_01
        sid = td.replace("Turbine_Data", "Status")
        df = read_member(zf, td, skiprows=9, low_memory=False)
        df = df.rename(columns={df.columns[0]: "time"})
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()
        full = pd.date_range(df.index.min().floor("10min"),
                             df.index.max().ceil("10min"), freq="10min")
        p = pd.to_numeric(df["Power (kW)"], errors="coerce").reindex(full)
        w = pd.to_numeric(df["Wind speed (m/s)"], errors="coerce").reindex(full)
        # status-derived curtailment flag
        curt = pd.Series(False, index=full)
        if sid in names:
            st = read_member(zf, sid, comment="#", dtype=str)
            st["Code"] = st["Code"].astype(str).str.strip()
            st = st[st["Code"].isin(CURT_CODES)]
            for _, r in st.iterrows():
                a = pd.to_datetime(r["Timestamp start"], errors="coerce")
                b = pd.to_datetime(r["Timestamp end"], errors="coerce")
                if pd.isna(a) or pd.isna(b):
                    continue
                curt.loc[(curt.index >= a.floor("10min")) &
                         (curt.index <= b.ceil("10min"))] = True
        # empirical available-power curve (q95 per 0.5 m/s bin)
        obs = p.notna() & w.notna()
        bins = np.arange(0, 30.5, 0.5)
        wb = pd.cut(w[obs], bins, labels=bins[:-1] + 0.25)
        q95 = p[obs].groupby(wb, observed=True).quantile(0.95)
        xs = q95.index.to_numpy(float)
        avail = pd.Series(np.interp(w.fillna(0), xs, q95.to_numpy(float)),
                          index=full)
        avail[w.isna()] = np.nan
        avail[w < xs[0] - 0.25] = np.nan
        avail[w > xs[-1] + 0.25] = np.nan
        cens = (curt & p.notna() & avail.notna()
                & ((avail - p) > BIND_MARGIN) & (avail > HIGH_AVAIL))
        out[tag] = dict(power=p.to_numpy(np.float32), wind=w.to_numpy(np.float32),
                        curt=curt.to_numpy(), cens=cens.to_numpy(),
                        avail=avail.to_numpy(np.float32),
                        time=full.astype("datetime64[s]").to_numpy(),
                        curve_x=xs, curve_y=q95.to_numpy(np.float32))
    return out


def main():
    store, stats = {}, {}
    for zp in sorted(glob.glob(os.path.join(DCDIR, "Penmanshiel_SCADA_2016_*.zip"))):
        for tag, d in process(zp).items():
            n = len(d["power"])
            obs = np.isfinite(d["power"]) & np.isfinite(d["wind"])
            cens, curt = d["cens"], d["curt"]
            # censor rate by wind-speed decile (documenting value-dependence)
            wv = d["wind"]
            edges = np.nanquantile(wv[obs], np.linspace(0, 1, 6))
            by_dec = []
            for i in range(5):
                m = obs & (wv >= edges[i]) & (wv <= edges[i + 1])
                by_dec.append(float(cens[m].mean()) if m.any() else None)
            # censored episode lengths
            runs, cur = [], 0
            for v in cens:
                if v:
                    cur += 1
                elif cur:
                    runs.append(cur); cur = 0
            if cur:
                runs.append(cur)
            runs = np.array(runs) if runs else np.zeros(1, int)
            stats[tag] = dict(
                n_intervals=n, obs_rate=float(obs.mean()),
                curt_frac=float(curt[obs].mean()),
                cens_frac=float(cens[obs].mean()),
                cens_episodes=int((runs > 0).sum()),
                cens_ep_len_median=float(np.median(runs[runs > 0])) if (runs > 0).any() else 0,
                cens_ep_len_max=int(runs.max()),
                cens_frac_by_wind_quintile=by_dec,
                power_mean=float(np.nanmean(d["power"])),
            )
            for k, v in d.items():
                store[f"{tag}_{k}"] = v
            print(tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in stats[tag].items()}))
    np.savez_compressed(OUT_NPZ, **store)
    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print("saved", OUT_NPZ)


if __name__ == "__main__":
    main()
