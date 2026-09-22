#!/usr/bin/env python
"""S32c: fill sensitivity as a function of how much is actually missing, pooled and
protocol-checked.

Two weaknesses of S32-A, both fixed here.

1. ONE DATASET carried the headline. Only `kdd_cup_2018_with_missing` had >10% context NaN
   in the window sample, so the 47% spread rested on a single dataset. Fix: pool windows from
   every missing-containing GIFT-Eval dataset and stratify by the window's OWN context-NaN
   fraction. The claim then becomes a dose-response curve over thousands of windows rather
   than one cell, and low-NaN bins act as their own negative control.

2. PROTOCOL DEPENDENCE. S32-A sampled random windows, which is not GIFT-Eval's protocol.
   Fix: run the identical analysis under two window protocols -- random windows, and
   last-window-per-series (the official-style held-out convention) -- and report both. The
   claim is a within-protocol sensitivity, so agreement across protocols is what it needs.

Metric as in S32: MASE vs an in-window naive-1 baseline, scored on observed target points.
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GIFT = os.path.join(ROOT, "tsfm_missing", "data", "gifteval")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import pyarrow as pa
import torch

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import run_s32_gifteval as s32

L, H = s32.L, s32.H
FILLS = s32.FILLS
BINS = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 1.01)]
OUT = os.path.join(HERE, "s32c_results.json")


def collect(protocol, per_series, max_series, seed=1):
    """Pool windows from every missing-containing dataset under one window protocol."""
    ctxs, tgts, oks, src = [], [], [], []
    for dsfreq in s32.MISSING_DS:
        try:
            ser = s32.read_series(dsfreq, max_series=max_series)
        except Exception:
            continue
        if not ser:
            continue
        rng = np.random.default_rng(seed)
        for a in ser:
            n = len(a)
            if n < L + H:
                continue
            if protocol == "last":
                starts = [n - L - H]
            else:
                k = min(per_series, n - L - H + 1)
                starts = rng.choice(n - L - H + 1, size=k, replace=False)
            for s0 in np.atleast_1d(starts):
                c, y = a[s0:s0 + L], a[s0 + L:s0 + L + H]
                m = np.isfinite(y)
                if m.sum() < 8 or np.isfinite(c).sum() < 16:
                    continue
                ctxs.append(c)
                tgts.append(np.nan_to_num(y))
                oks.append(m)
                src.append(dsfreq)
    if not ctxs:
        return None
    return np.stack(ctxs), np.stack(tgts), np.stack(oks), np.array(src)


def run_model(res, name, protocol, ctx, tgt, ok, src, batch_report=True):
    model = s32.get_model(name)
    nanfrac = (~np.isfinite(ctx)).mean(axis=1)
    scores = {}
    for how in FILLS:
        f_, miss = s32.fill(ctx, how)
        f_in = ctx.astype(np.float32) if how == "nan" else f_
        t0 = time.time()
        pred = s32.forecast(model, f_in, miss, how)
        scores[how] = s32.mase(pred, tgt, ok, ctx)
        print(f"    {name:10s} {protocol:6s} {how:7s} done ({time.time()-t0:.0f}s)", flush=True)
    for lo, hi in BINS:
        sel = (nanfrac >= lo) & (nanfrac < hi)
        if sel.sum() < 20:
            continue
        med = {h: float(np.nanmedian(scores[h][sel])) for h in FILLS}
        best, worst = min(med.values()), max(med.values())
        key = f"{name}|{protocol}|{lo:.2f}-{hi:.2f}"
        res.setdefault("bins", {})[key] = {
            "n": int(sel.sum()), "median_by_fill": med,
            "spread_pct": 100 * (worst - best) / max(best, 1e-9),
            "best_fill": min(med, key=med.get),
            "datasets": sorted(set(src[sel].tolist()))}
        print(f"  {name:10s} {protocol:6s} nan in [{lo:.0%},{hi:.0%})  n={sel.sum():5d}  " +
              "  ".join(f"{h}={med[h]:.3f}" for h in FILLS) +
              f"   spread {100*(worst-best)/max(best,1e-9):5.1f}%  best={min(med, key=med.get)}",
              flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bolt-base,chronos2,timesfm,moirai")
    ap.add_argument("--per-series", type=int, default=6)
    ap.add_argument("--max-series", type=int, default=250)
    args = ap.parse_args()
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"bins": BINS, "fills": list(FILLS), "L": L, "H": H,
                            "protocols": ["random", "last"],
                            "datasets": s32.MISSING_DS})
    for protocol in ("random", "last"):
        got = collect(protocol, args.per_series, args.max_series)
        if got is None:
            print(f"{protocol}: no windows", flush=True)
            continue
        ctx, tgt, ok, src = got
        nf = (~np.isfinite(ctx)).mean(axis=1)
        print(f"\n=== protocol={protocol}: {len(ctx)} pooled windows, "
              f"ctx NaN mean {100*nf.mean():.2f}%, "
              f">10% NaN in {100*(nf > 0.10).mean():.1f}% of windows ===", flush=True)
        res.setdefault("pool", {})[protocol] = {
            "n": int(len(ctx)), "nan_mean": float(nf.mean()),
            "frac_gt10pct": float((nf > 0.10).mean())}
        for name in args.models.split(","):
            try:
                res = run_model(res, name, protocol, ctx, tgt, ok, src)
            except Exception as e:
                print(f"  SKIP {name}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            json.dump(res, open(OUT + ".tmp", "w"))
            os.replace(OUT + ".tmp", OUT)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
