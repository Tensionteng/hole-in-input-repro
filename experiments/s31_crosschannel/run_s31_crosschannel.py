#!/usr/bin/env python
"""S31: when one channel goes dark, can a multivariate TSFM borrow from its neighbours?

This is the question a practitioner actually asks. A sensor fails, but a highly correlated
sensor next to it is fine. Multivariate foundation models (Chronos-2 with covariates, Moirai)
advertise cross-series information sharing. Does that sharing actually repair a hole?

If it does, the practical advice under missingness is "use a multivariate model", and the
whole single-series repair ladder matters much less. If it does not, then the interface
result from S25/S27 explains why -- the model is never told which channel is degraded, so it
has no way to down-weight it -- and cross-channel borrowing is another lever that the input
convention silently disables.

Design, per (dataset, mechanism, rate):
  (a) target channel CLEAN, univariate                  -> upper bound
  (b) target channel holed + filled, univariate         -> the baseline everyone lives with
  (c) target holed + filled, + K observed neighbours    -> the question
  (d) target holed as NaN, + K observed neighbours      -> does declaring it help borrowing?
Recovery = (b - c) / (b - a): the fraction of the damage that neighbours buy back.

Neighbours are the K most-correlated channels, correlation computed on the TRAIN region
(first 80%) so the choice does not leak the evaluation window. They are always fully
observed -- this is the most favourable case for borrowing, so a null result here is strong.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
import run_s25_twofloor as s25

L, H = s25.L, 64
K_NEIGH = 4
MECHS = ("mcar", "block", "mnar_high")
RATES = (0.3, 0.7)
OUT = os.path.join(HERE, "s31_results.json")


def neighbours(X, n_train, k=K_NEIGH):
    """k most-correlated channels per channel, from the TRAIN region only."""
    Z = X[:n_train]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    C = np.abs(np.corrcoef(Z.T))
    np.fill_diagonal(C, -np.inf)
    C = np.nan_to_num(C, nan=-np.inf)
    return {c: np.argsort(-C[c])[:k].tolist() for c in range(X.shape[1])}


class C2:
    name = "chronos-2"

    def __init__(self, dev="cuda"):
        from chronos import Chronos2Pipeline
        self.p = Chronos2Pipeline.from_pretrained(
            os.path.join(ROOT, "models_local", "chronos-2"),
            device_map=dev, torch_dtype=torch.float32)

    @torch.no_grad()
    def fc(self, targets, covs=None, batch=64):
        """targets: [n, L]; covs: None or [n, K, L]. Returns median forecast [n, H]."""
        outs = []
        for i in range(0, len(targets), batch):
            chunk = []
            for j in range(i, min(i + batch, len(targets))):
                d = {"target": targets[j]}
                if covs is not None:
                    d["past_covariates"] = {f"c{k}": covs[j][k] for k in range(covs.shape[1])}
                chunk.append(d)
            q = self.p.predict(chunk, prediction_length=H, batch_size=batch)
            a = np.stack([np.asarray(x.cpu() if torch.is_tensor(x) else x) for x in q])
            a = a.squeeze(1) if a.ndim == 4 else a
            outs.append(a[:, a.shape[1] // 2, :])
        return np.concatenate(outs)


def build(X, starts, ch, nb, mech, rate, fill="linear"):
    """Target channel `ch` holed; neighbour channels `nb` fully observed."""
    C = X.shape[1]
    tg_clean, tg_fill, tg_nan, cov, gt, msk = [], [], [], [], [], []
    for wi, s in enumerate(starts):
        w = X[s:s + L].T                     # [C, L]
        x = w[ch].copy()
        m = (np.zeros(L, bool) if mech == "clean"
             else s25.make_mask(mech, rate, wi, 0, 1, x=x[None])[0])
        f = s25.fill_context(x[None], m[None], fill)[0]
        n_ = x.copy()
        n_[m] = np.nan
        tg_clean.append(x)
        tg_fill.append(f)
        tg_nan.append(n_)
        cov.append(w[nb].copy())
        gt.append(X[s + L:s + L + H].T[ch].copy())
        msk.append(m)
    f32 = lambda a: np.asarray(a, np.float32)
    return f32(tg_clean), f32(tg_fill), f32(tg_nan), f32(cov), f32(gt), np.asarray(msk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-win", type=int, default=150)
    ap.add_argument("--n-ch", type=int, default=6, help="target channels per dataset")
    ap.add_argument("--datasets", default="weather,ETTh1,ETTm1")
    args = ap.parse_args()
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"K_neighbours": K_NEIGH, "mechs": list(MECHS),
                            "rates": list(RATES), "H": H, "n_win": args.n_win,
                            "model": "chronos-2 (past_covariates)",
                            "recovery": "(b - c) / (b - a)"})
    model = C2()
    cells = {}
    for ds in args.datasets.split(","):
        path = s25.DATASETS[ds]
        import pandas as pd
        Xall = pd.read_csv(path).drop(columns=["date"]).to_numpy(np.float32)
        nb_map = neighbours(Xall, int(0.8 * len(Xall)))
        X, st = s25.load_windows(path, args.n_win, s25.SEED, H)
        rng = np.random.default_rng(s25.SEED)
        chans = sorted(rng.choice(X.shape[1], size=min(args.n_ch, X.shape[1]),
                                  replace=False).tolist())
        for ch in chans:
            nb = nb_map[ch]
            for mech in MECHS:
                for rate in RATES:
                    t0 = time.time()
                    cl, fl, nn_, cov, gt, m = build(X, st, ch, nb, mech, rate)
                    e = lambda p: ((p - gt) ** 2).mean(1)
                    a = e(model.fc(cl))                       # clean univariate
                    b = e(model.fc(fl))                       # holed univariate
                    c = e(model.fc(fl, cov))                  # holed + neighbours
                    d = e(model.fc(nn_, cov))                 # NaN + neighbours
                    ok = a > 1e-12
                    rel = lambda v: float(np.median(v[ok] / a[ok]))
                    gap = np.median(b[ok] / a[ok]) - 1.0
                    rec_c = (np.median(b[ok] / a[ok]) - np.median(c[ok] / a[ok])) / max(gap, 1e-9)
                    rec_d = (np.median(b[ok] / a[ok]) - np.median(d[ok] / a[ok])) / max(gap, 1e-9)
                    key = f"{ds}|ch{ch}|{mech}|{rate}"
                    cells[key] = {"clean": 1.0, "holed_uni": rel(b), "holed_multi": rel(c),
                                  "nan_multi": rel(d), "recovery_fill": float(rec_c),
                                  "recovery_nan": float(rec_d),
                                  "neighbours": nb, "n": int(ok.sum())}
                    print(f"  {key:28s} uni={rel(b):6.3f} +nb={rel(c):6.3f} "
                          f"nan+nb={rel(d):6.3f}  recovery {100*rec_c:+5.0f}% / "
                          f"{100*rec_d:+5.0f}% ({time.time()-t0:.0f}s)", flush=True)
        json.dump({**res, "cells": cells}, open(OUT + ".tmp", "w"))
        os.replace(OUT + ".tmp", OUT)
    res["cells"] = cells
    agg = {}
    for mech in MECHS:
        for rate in RATES:
            v = [c["recovery_fill"] for k, c in cells.items() if f"|{mech}|{rate}" in k]
            w = [c["recovery_nan"] for k, c in cells.items() if f"|{mech}|{rate}" in k]
            if v:
                agg[f"{mech}|{rate}"] = {"recovery_fill_median": float(np.median(v)),
                                         "recovery_nan_median": float(np.median(w)),
                                         "n_cells": len(v)}
    res["agg"] = agg
    print("\n== recovery from neighbours (median over channels/datasets) ==")
    for k, v in agg.items():
        print(f"  {k:22s} fill+nb {100*v['recovery_fill_median']:+6.1f}%   "
              f"nan+nb {100*v['recovery_nan_median']:+6.1f}%  (n={v['n_cells']})")
    json.dump(res, open(OUT + ".tmp", "w"))
    os.replace(OUT + ".tmp", OUT)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
