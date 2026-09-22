#!/usr/bin/env python
"""S62 follow-up: why is TempoPFN's weather plain-path redraw arm so large?

The census probe on weather/mcar/plain gave perm=0.545 but redraw=6.26 (rho=0.087),
while every other family sits at perm~redraw~0.5 there. Hypothesis: weather has
near-constant channels; TempoPFN's RobustScaler clamps the IQR at min_scale=1e-3
(and scaled values at +-50), so a redrawn OBSERVED value (spiky, from the observed
pool) at a missing position lands orders of magnitude further from the linear fill
than any permutation of the fill can -- the redraw arm measures a marginal shift the
perm arm cannot produce. This script reports the per-series distribution of both
arms and the per-channel IQR floor hits to make the caveat concrete.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tempopfn_triton_cache")

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, HERE)

import numpy as np
import torch

import run_s25_twofloor as s25
from run_s62_tempopfn import TempoPFN, OUT

L, H = s25.L, 64


def main():
    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    tempo = TempoPFN("cuda:0", bf16=True)

    X, st = s25.load_windows(s25.DATASETS["weather"], 60, s25.SEED, H)
    _, lin, mask, _ = s25.build_eval_batch(X, st, "mcar", 0.3, H, fill="linear")
    y0 = tempo.fc(lin, mask, "plain")
    scale = np.sqrt((y0 ** 2).mean(1)) + 1e-9

    rng = np.random.default_rng(s25.SEED)
    dp_all, dr_all = [], []
    for _ in range(4):
        cp, cr = lin.copy(), lin.copy()
        for i in range(len(lin)):
            idx = np.flatnonzero(mask[i])
            if len(idx) < 2:
                continue
            cp[i, idx] = lin[i, idx][rng.permutation(len(idx))]
            o = lin[i, ~mask[i]]
            cr[i, idx] = rng.choice(o, size=len(idx)) if len(o) else 0.0
        dp_all.append(np.sqrt(((tempo.fc(cp, mask, "plain") - y0) ** 2).mean(1)) / scale)
        dr_all.append(np.sqrt(((tempo.fc(cr, mask, "plain") - y0) ** 2).mean(1)) / scale)
    dp, dr = np.mean(dp_all, 0), np.mean(dr_all, 0)

    def stats(a):
        return {q: float(np.percentile(a, q)) for q in (50, 90, 99, 100)}

    # channel scale diagnostics: how many series sit at the IQR min_scale floor,
    # and does the redraw arm move the SCALED input much further than the perm arm?
    iqr = np.percentile(lin, 75, axis=1) - np.percentile(lin, 25, axis=1)
    chan_std = lin.std(1)
    fill_dev_perm, fill_dev_redraw = [], []
    rng2 = np.random.default_rng(s25.SEED)
    for i in range(len(lin)):
        idx = np.flatnonzero(mask[i])
        if len(idx) < 2:
            continue
        pv = lin[i, idx][rng2.permutation(len(idx))]
        rv = rng2.choice(lin[i, ~mask[i]], size=len(idx))
        fill_dev_perm.append(float(np.abs(pv - lin[i, idx]).max()))
        fill_dev_redraw.append(float(np.abs(rv - lin[i, idx]).max()))

    out = {
        "weather_mcar_plain_perm_per_series": stats(dp),
        "weather_mcar_plain_redraw_per_series": stats(dr),
        "redraw_mean_over_median": float(dr.mean() / np.median(dr)),
        "perm_mean_over_median": float(dp.mean() / np.median(dp)),
        "frac_series_iqr_below_1e-2": float((iqr < 1e-2).mean()),
        "frac_series_std_below_1e-2": float((chan_std < 1e-2).mean()),
        "max_abs_fill_change_perm": {"p50": float(np.percentile(fill_dev_perm, 50)),
                                     "max": float(np.max(fill_dev_perm))},
        "max_abs_fill_change_redraw": {"p50": float(np.percentile(fill_dev_redraw, 50)),
                                       "max": float(np.max(fill_dev_redraw))},
    }
    for k, v in out.items():
        print(f"{k}: {v}", flush=True)

    res = {}
    if os.path.exists(OUT):
        res = __import__("json").load(open(OUT))
    res.setdefault("followup", {})["weather_redraw_dissection"] = out
    tmp = OUT + ".tmp"
    __import__("json").dump(res, open(tmp, "w"))
    os.replace(tmp, OUT)
    print("appended to", OUT)


if __name__ == "__main__":
    main()
