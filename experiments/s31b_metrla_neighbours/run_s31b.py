#!/usr/bin/env python
"""S31b: cross-channel borrowing under a REAL missingness mechanism (METR-LA outages).

S31 found, on synthetic masks over real multivariate data, that supplying correlated
neighbours repairs ~95% of block-outage damage -- but ONLY if the hole is declared as NaN
rather than filled (with a fill, neighbours bought back 0%). That number is the largest
repair measured anywhere in this project, and it is currently synthetic-mask-only.

This round repeats it on METR-LA's real sensor outages, using S14's stored window list. For
each `miss` window on sensor s, the K most-correlated OTHER sensors over the same timestamps
are supplied as covariates, and only neighbours that are themselves fully observed in that
window are used -- so the comparison isolates "can it borrow", not "are the neighbours also
down". Correlations come from the first 80% of the timeline.

Conditions (paired, same windows): target filled univariate / target filled + neighbours /
target declared NaN + neighbours. Chronos-2 with `past_covariates`.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "tsfm_missing", "data")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch

sys.path.insert(0, os.path.join(EXP, "s14_metrla"))
sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s26_realfill"))
sys.path.insert(0, os.path.join(EXP, "s31_crosschannel"))
import run_s14_metrla as s14
s14.H5 = os.path.join(DATA, "dataset_metrla", "metr-la.h5")
import run_s26_realfill as s26
s26.s14.H5 = s14.H5
import run_s31_crosschannel as s31

CTX, H = 512, 64
K = 4
VAR_FLOOR = 4.0
OUT = os.path.join(HERE, "s31b_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=618)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--nb-max-miss", type=float, default=0.05,
                    help="allow neighbours that are themselves slightly incomplete; "
                         "requiring perfectly healthy neighbours keeps only 6% of real "
                         "outage windows, which is itself a finding")
    args = ap.parse_args()
    np.random.seed(20250810)
    torch.manual_seed(20250810)

    idx, sensors, V = s14.load_matrix()
    T, S = V.shape
    ntr = int(0.8 * T)
    Z = V[:ntr].astype(np.float64)
    obs = Z != 0.0
    mu = np.array([Z[obs[:, i], i].mean() if obs[:, i].any() else 0 for i in range(S)])
    sd = np.array([Z[obs[:, i], i].std() if obs[:, i].sum() > 1 else 1 for i in range(S)]) + 1e-9
    Zs = np.where(obs, (Z - mu) / sd, 0.0)
    C = np.abs((Zs.T @ Zs) / max(ntr, 1))
    np.fill_diagonal(C, -np.inf)
    nb_map = {i: np.argsort(-C[i])[:K + 20].tolist() for i in range(S)}

    windows = s14.build_windows()
    sidx = {s: i for i, s in enumerate(sensors)}
    miss = [w for w in windows if w["kind"] == "miss"]
    rng = np.random.default_rng(0)
    if len(miss) > args.max_windows:
        miss = [miss[i] for i in sorted(rng.choice(len(miss), args.max_windows, replace=False))]

    tg_fill, tg_nan, cov, gt, kept, n_healthy = [], [], [], [], 0, 0
    for w in miss:
        si, t = sidx[w["sensor"]], w["start"]
        v = V[:, si]
        c = v[t - CTX:t].astype(np.float32)
        m = c == 0.0
        nbs = [j for j in nb_map[si]
               if (V[t - CTX:t, j] == 0.0).mean() <= args.nb_max_miss][:args.k]
        n_healthy += int(len([j for j in nb_map[si]
                              if (V[t - CTX:t, j] != 0.0).all()]) >= args.k)
        if len(nbs) < args.k:
            continue
        lin = np.interp(np.arange(CTX), np.flatnonzero(~m), c[~m]).astype(np.float32)
        nn_ = c.copy()
        nn_[m] = np.nan
        tg_fill.append(lin)
        tg_nan.append(nn_)
        cov.append(V[t - CTX:t, nbs].T.astype(np.float32))
        gt.append(V[t:t + H, si].astype(np.float32))
        kept += 1
    tg_fill = np.stack(tg_fill); tg_nan = np.stack(tg_nan)
    cov = np.stack(cov); gt = np.stack(gt)
    print(f"AVAILABILITY: {n_healthy}/{len(miss)} miss windows have {args.k} FULLY healthy "
          f"top-correlated neighbours ({100*n_healthy/max(len(miss),1):.1f}%); "
          f"{kept} usable at <= {100*args.nb_max_miss:.0f}% neighbour missingness\n"
          f"windows: {kept} kept; "
          f"mean target missing rate {100*(tg_fill.shape[1] and np.mean([np.isnan(x).mean() for x in tg_nan])):.1f}%",
          flush=True)

    model = s31.C2()
    nmse = lambda p: ((p - gt) ** 2).mean(1) / np.maximum(gt.var(axis=1), VAR_FLOOR)
    res = {"meta": {"CTX": CTX, "H": H, "K": args.k, "nb_max_miss": args.nb_max_miss, "n_fully_healthy": int(n_healthy), "n": int(kept),
                    "model": "chronos-2 past_covariates", "var_floor": VAR_FLOOR,
                    "windows": "S14 METR-LA miss windows, real sensor outages"}}
    conds = {}
    t0 = time.time()
    conds["filled_uni"] = nmse(model.fc(tg_fill))
    conds["filled_plus_nb"] = nmse(model.fc(tg_fill, cov))
    conds["nan_plus_nb"] = nmse(model.fc(tg_nan, cov))
    conds["nan_uni"] = nmse(model.fc(tg_nan))
    base = conds["filled_uni"]
    print(f"\n== METR-LA real outages, chronos-2, {kept} windows ({time.time()-t0:.0f}s) ==")
    for k, v in conds.items():
        r = v / np.maximum(base, 1e-12)
        res[k] = {"nmse_median": float(np.median(v)), "nmse_mean": float(v.mean()),
                  "ratio_vs_filled_uni_median": float(np.median(r)),
                  "winrate_vs_filled_uni": float((v < base).mean()),
                  "per_window": v.tolist()}
        print(f"  {k:16s} NMSE med={np.median(v):7.4f} mean={v.mean():8.3f}  "
              f"vs filled-uni x{np.median(r):.3f}  wins {100*(v<base).mean():.0f}%")
    json.dump(res, open(OUT, "w"))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
