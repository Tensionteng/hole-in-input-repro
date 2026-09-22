#!/usr/bin/env python
"""S50: is rho meaningful for a forecaster that samples?

The permutation probe reads rho = E||f(permuted fill) - f(fill)|| / E||f(redrawn fill) -
f(fill)||, and reads rho near 1 as "the model uses the fill's content". For a DETERMINISTIC
forecaster that reading is sound. For a forecaster that samples, both the numerator and the
denominator also contain the model's own sampling noise, which pushes the ratio toward 1
whether or not the model reads anything. Moirai 1.1 and Moirai-MoE sample; Moirai 2.0,
Chronos-Bolt, Chronos-2 and TimesFM do not.

So we measure the noise floor directly. Call the model twice on the IDENTICAL input and form
the same ratio,

    rho_self = E||f(x) - f(x)|| / E||f(redrawn fill) - f(x)||,

which is what the probe would report for a model that ignores the fill completely. rho_self
near 0 means the probe's reading is signal; rho_self near rho_declared means the probe cannot
distinguish retention from noise for that checkpoint, and its row should not be read as
evidence of retention.

Run against the same cells, windows, masks and seeds as S30/S36.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
sys.path.insert(0, os.path.join(EXP, "s36_models"))
import run_s25_twofloor as s25
import run_s30_crossmodel as s30
import run_s36_models as s36

# S30 carries the deterministic controls and Moirai 1.1 base; S36 carries the other
# Moirai sizes and the MoE variants, which are the remaining stochastic checkpoints.
REGISTRY = {**s30.REGISTRY, **s36.REGISTRY}

H, L = s30.H, s30.L
RATE, MECHS, N_PERM = s30.RATE, s30.MECHS, s30.N_PERM
OUT = os.path.join(HERE, "s52_results.json")


def probe_with_noise(model, ctx, miss, conv, rng):
    """S30's perm_probe plus a self arm: f(x) against f(x) on identical input."""
    y0 = model.fc(ctx, miss, conv)
    scale = np.sqrt((y0 ** 2).mean(1)) + 1e-9
    dp, dr, dsf = [], [], []
    for _ in range(N_PERM):
        cp, cr = ctx.copy(), ctx.copy()
        for i in range(len(ctx)):
            idx = np.flatnonzero(miss[i])
            if len(idx) < 2:
                continue
            cp[i, idx] = ctx[i, idx][rng.permutation(len(idx))]
            o = ctx[i, ~miss[i]]
            cr[i, idx] = rng.choice(o, size=len(idx)) if len(o) else 0.0
        dp.append(np.sqrt(((model.fc(cp, miss, conv) - y0) ** 2).mean(1)) / scale)
        dr.append(np.sqrt(((model.fc(cr, miss, conv) - y0) ** 2).mean(1)) / scale)
        # the noise arm: the SAME context, so any difference is the model's own stochasticity
        dsf.append(np.sqrt(((model.fc(ctx.copy(), miss, conv) - y0) ** 2).mean(1)) / scale)
    dp, dr, dsf = np.mean(dp, 0), np.mean(dr, 0), np.mean(dsf, 0)
    den = max(dr.mean(), 1e-15)
    return {"perm_rms_rel": float(dp.mean()), "redraw_rms_rel": float(dr.mean()),
            "self_rms_rel": float(dsf.mean()),
            "ratio": float(dp.mean() / den), "ratio_self": float(dsf.mean() / den)}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="moirai,bolt,chronos2")
    ap.add_argument("--n-win", type=int, default=s30.N_WIN)
    args = ap.parse_args()
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"rate": RATE, "mechs": list(MECHS), "n_win": args.n_win,
                            "n_perm": N_PERM, "H": H,
                            "probe": "S30's permutation probe plus a same-input noise arm"})
    for key in args.models.split(","):
        try:
            model = REGISTRY[key]("cuda")
        except Exception as e:
            print(f"SKIP {key}: {type(e).__name__}: {str(e)[:140]}", flush=True)
            continue
        print(f"\n== {model.name} ==", flush=True)
        cells = {}
        for ds, path in s25.DATASETS.items():
            X, st = s25.load_windows(path, args.n_win, s25.SEED, H)
            for mech in MECHS:
                clean, lin, mask, _ = s25.build_eval_batch(X, st, mech, RATE, H, fill="linear")
                rng = np.random.default_rng(s25.SEED)
                for conv in model.convs:
                    t0 = time.time()
                    try:
                        r = probe_with_noise(model, lin.copy(), mask, conv, rng)
                    except Exception as e:
                        print(f"  {ds} {mech} {conv} FAIL {type(e).__name__}: {str(e)[:90]}",
                              flush=True)
                        continue
                    cells[f"{ds}|{mech}|{conv}"] = r
                    print(f"  {ds:8s} {mech:10s} {conv:6s} rho={r['ratio']:.4f} "
                          f"rho_self={r['ratio_self']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        agg = {}
        for conv in model.convs:
            v = [c["ratio"] for k, c in cells.items() if k.endswith("|" + conv)]
            vs = [c["ratio_self"] for k, c in cells.items() if k.endswith("|" + conv)]
            if v:
                agg[conv] = {"ratio_median": float(np.median(v)),
                             "ratio_self_median": float(np.median(vs)), "n_cells": len(v)}
        res.setdefault("models", {})[key] = {"name": model.name, "cells": cells, "agg": agg}
        print(f"  AGG {model.name}: " + "  ".join(
            f"{c}: rho={agg[c]['ratio_median']:.4f} rho_self={agg[c]['ratio_self_median']:.4f}"
            for c in agg), flush=True)
        del model
        torch.cuda.empty_cache()
        json.dump(res, open(OUT + ".tmp", "w"))
        os.replace(OUT + ".tmp", OUT)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
