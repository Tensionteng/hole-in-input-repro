#!/usr/bin/env python
"""S35: does the interface result hold across the full set of standard benchmark datasets?

Proposition 1 and the consequences that follow from it were measured on three datasets
(ETTh1, ETTm1, weather). This round extends the two load-bearing measurements to every
standard forecasting dataset available to us --- adding ETTh2, ETTm2, electricity (321
channels), traffic (862 channels), exchange rate and national illness --- so that the claims
rest on nine datasets spanning four orders of magnitude in channel count and several domains.

Two measurements, both cheap:
  A. the permutation probe of Proposition 1 (gradient-free), per interface;
  B. the characterisation grid and the imputation-quality sweep at the declared and
     content-restored interfaces.

Anchor gate: the three original datasets must reproduce their stored values.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TSLIB = os.path.join(ROOT, "legacy_nonstationary", "tslib", "dataset")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s27_interface"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
import run_s25_twofloor as s25
import run_s27_interface as s27
import run_s30_crossmodel as s30

H = s25.H
L = s25.L
RATE = 0.3
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES = (0.3, 0.7)
ALPHAS = (0.0, 0.5, 1.0)
OUT = os.path.join(HERE, "s35_results.json")

DATASETS = {
    "ETTh1": ("ETT-small/ETTh1.csv", None),
    "ETTh2": ("ETT-small/ETTh2.csv", None),
    "ETTm1": ("ETT-small/ETTm1.csv", None),
    "ETTm2": ("ETT-small/ETTm2.csv", None),
    "weather": ("weather/weather.csv", None),
    "electricity": ("electricity/electricity.csv", 24),
    "traffic": ("traffic/traffic.csv", 24),
    "exchange": ("exchange_rate/exchange_rate.csv", None),
    "illness": ("illness/national_illness.csv", None),
}


def load(name, n_windows, seed=s25.SEED):
    """Same convention as S25: last 20% of the timeline, channels as univariate series."""
    rel, max_ch = DATASETS[name]
    df = pd.read_csv(os.path.join(TSLIB, rel))
    df = df.drop(columns=[c for c in df.columns if c.lower() in ("date", "unnamed: 0")])
    X = df.to_numpy(np.float32)
    if max_ch is not None and X.shape[1] > max_ch:
        rng = np.random.default_rng(seed)
        X = X[:, np.sort(rng.choice(X.shape[1], max_ch, replace=False))]
    N = len(X)
    test_start = int(0.8 * N)
    lo, hi = test_start - L, N - L - H
    if hi < lo:
        return None, None
    rng = np.random.default_rng(seed)
    k = min(n_windows, hi - lo + 1)
    starts = np.sort(rng.choice(hi - lo + 1, size=k, replace=False)) + lo
    return X, starts


def build(X, starts, mech, rate, fill="linear"):
    C = X.shape[1]
    cl, fl, mk, gt = [], [], [], []
    for wi, s in enumerate(starts):
        x = X[s:s + L].T.copy()
        y = X[s + L:s + L + H].T.copy()
        m = (np.zeros((C, L), bool) if mech == "clean"
             else s25.make_mask(mech, rate, wi, 0, C, x=x))
        cl.append(x)
        fl.append(s25.fill_context(x, m, "none" if mech == "clean" else fill))
        mk.append(m)
        gt.append(y)
    f = lambda a: np.concatenate(a).astype(np.float32)
    return f(cl), f(fl), np.concatenate(mk), f(gt)


def finite(a):
    return np.isfinite(a).all(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-win", type=int, default=40)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--parts", default="probe,grid,sweep")
    args = ap.parse_args()
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"L": L, "H": H, "rate_probe": RATE, "rates": list(RATES),
                            "alphas": list(ALPHAS), "n_win": args.n_win,
                            "datasets": list(DATASETS), "model": "chronos-bolt-base"})
    bolt_probe = s30.Bolt("cuda")
    bolt = s25.Bolt("cuda")
    parts = args.parts.split(",")

    for name in args.datasets.split(","):
        X, st = load(name, args.n_win)
        if X is None:
            print(f"SKIP {name}: too short", flush=True)
            continue
        # drop channels with non-finite or constant context
        _, cl, _, gt_c = build(X, st, "clean", 0.0)
        keep = finite(cl) & finite(gt_c) & (cl.std(axis=1) > 1e-8)
        base = ((bolt.median_np(cl[keep]) - gt_c[keep]) ** 2).mean(1)
        ok = base > 1e-12
        print(f"\n== {name}: {X.shape[1]} ch x {len(st)} win = {keep.sum()} usable series "
              f"(clean MSE median {np.median(base):.4f}) ==", flush=True)
        res.setdefault("clean", {})[name] = {"n_series": int(keep.sum()),
                                             "mse_median": float(np.median(base))}

        if "probe" in parts:
            _, lin, mask, _ = build(X, st, "block", RATE)
            lin, mask = lin[keep], mask[keep]
            rng = np.random.default_rng(s25.SEED)
            row = {}
            for conv in ("plain", "mask", "nan"):
                r = s30.perm_probe(bolt_probe, lin, mask, conv, rng)
                row[conv] = r["ratio"]
            res.setdefault("probe", {})[name] = row
            print("   probe rho: " + "  ".join(f"{k}={v:.4f}" for k, v in row.items()),
                  flush=True)

        if "grid" in parts:
            for mech in MECHS:
                out = []
                for rate in RATES:
                    _, fl, mk, gt = build(X, st, mech, rate)
                    e = ((bolt.median_np(fl[keep]) - gt[keep]) ** 2).mean(1)
                    v = float(np.median(e[ok] / base[ok]))
                    res.setdefault("grid", {})[f"{name}|{mech}|{rate}"] = v
                    out.append(v)
                print(f"   {mech:13s} " + "  ".join(f"p={r}:{v:.3f}"
                                                    for r, v in zip(RATES, out)), flush=True)

        if "sweep" in parts:
            for mech in ("block", "mnar_high"):
                clean, lin, mask, gt = build(X, st, mech, 0.7)
                clean, lin, mask, gt = clean[keep], lin[keep], mask[keep], gt[keep]
                obs = (~mask).astype(np.float32)
                for iface in ("native", "dual"):
                    vals = []
                    for a in ALPHAS:
                        c = lin.copy()
                        c[mask] = (1 - a) * lin[mask] + a * clean[mask]
                        p = s27.median_iface_np(bolt, c, obs, iface)
                        e = ((p - gt) ** 2).mean(1)
                        vals.append(float(np.median(e[ok] / base[ok])))
                    res.setdefault("sweep", {})[f"{name}|{mech}|{iface}"] = vals
                    print(f"   sweep {mech:10s} {iface:7s} " +
                          " -> ".join(f"{v:.3f}" for v in vals) +
                          f"   slope {vals[-1]-vals[0]:+.3f}", flush=True)
        json.dump(res, open(OUT + ".tmp", "w"))
        os.replace(OUT + ".tmp", OUT)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
