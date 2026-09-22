#!/usr/bin/env python
"""S64: GIFT-Eval dose-response (Table-4 protocol) for the Chronos-2 and Moirai-2.0
retrofits, stock vs WiSE-FT 0.5, on the EXACT s32c pipeline.

Protocol (run_s32c_stratified.py, random-window arm, reused verbatim):
  windows pooled from the nine missing-containing GIFT-Eval subsets
  (s32.MISSING_DS), random-window protocol (per_series=6, max_series=250, seed=1 --
  the identical pool as s32c_results.json, n=7725), context 512, horizon 64, scored
  on OBSERVED target points, MASE against an in-window naive-1 baseline computed on
  the observed context points. Stratified by the window's OWN context-NaN fraction
  into bins (0-1%, 1-5%, 5-15%, 15-30%, >30%). Table 4 reports the LINEAR fill.

Fill routing (s32.forecast + the Q50Bolt precedent for retrofit arms):
  c2-stock  native iface: nan -> the NaN path (mask from NaN, observed-only stats);
            fixed fills -> plain (fill content, all-ones flag)  [= s32c's chronos2]
  c2-w50    dual iface:   nan -> declaration endpoint (zeroed content + flag=0);
            fixed fills -> declared (fill content + true flag)  [Q50Bolt precedent]
  m2-stock  nan -> declared (content nan_to_num + obs=~miss); fixed fills -> plain
            (content + all-ones flag)                            [= s32c's moirai2]
  m2-w50    nan -> declared; fixed fills -> declared (fill + true flag)
                                                               [Q50Bolt precedent]

Gate: c2-stock and m2-stock bin medians must reproduce s32c_results.json's
chronos2|random and moirai2|random rows (all four fills) -- that validates the
window pool, the metric, and the wrapper equivalence in one comparison.
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

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s32_gifteval"))
sys.path.insert(0, os.path.join(EXP, "s60_c2retrofit"))
sys.path.insert(0, os.path.join(EXP, "s61_m2retrofit"))
import run_s25_twofloor as s25
import run_s32_gifteval as s32
import run_s32c_stratified as s32c
import c2_iface
import m2_iface

L, H = s32.L, s32.H
FILLS = s32.FILLS
BINS = s32c.BINS
S60_CK = os.path.join(EXP, "s60_c2retrofit", "s60_ckpt")
S61_CK = os.path.join(EXP, "s61_m2retrofit", "s61_ckpt")
OUT = os.path.join(HERE, "s64_gift_dose.json")
SCORES = os.path.join(HERE, "s64_gift_scores.npz")


def make_model(name, dev):
    """Returns predict(ctx_filled, miss, how) -> [n, H] median forecast."""
    if name.startswith("c2"):
        model = c2_iface.load_stock(dev)
        iface = "native" if name == "c2-stock" else "dual"
        if name == "c2-w50":
            sd = torch.load(os.path.join(S60_CK, "arm_W50.pt"), map_location=dev,
                            weights_only=True)
            model.load_state_dict(sd)
            model.eval()

        def predict(ctx_filled, miss, how):
            if how == "nan":
                if iface == "native":
                    cx, ob = c2_iface.conv_ctx(ctx_filled, miss, "nan", "native")
                else:
                    # declaration endpoint: zeroed content + true flag (dual)
                    cx = np.nan_to_num(ctx_filled).astype(np.float32)
                    ob = (~miss).astype(np.float32)
            elif iface == "native":
                cx, ob = c2_iface.conv_ctx(ctx_filled, miss, "plain", "native")
            else:
                cx, ob = c2_iface.conv_ctx(ctx_filled, miss, "declared", "dual")
            return c2_iface.median_fwd(model, cx, ob, iface, horizon=H, batch=256)
        return predict, model
    if name.startswith("m2"):
        fc_mod = m2_iface.load_stock(dev)
        if name == "m2-w50":
            sd = torch.load(os.path.join(S61_CK, "arm_W50.pt"), map_location=dev,
                            weights_only=True)
            fc_mod.module.load_state_dict(sd)
            fc_mod.eval()

        def predict(ctx_filled, miss, how):
            if name == "m2-stock" and how != "nan":
                content, obs = m2_iface.conv_ctx(ctx_filled, miss, "plain")
            else:
                # nan how: declared (nan_to_num + ~miss) for both arms;
                # fixed fills on w50: declared (fill + true flag)
                content, obs = m2_iface.conv_ctx(ctx_filled, miss, "nan")
            return m2_iface.median_fwd(fc_mod, content, obs, horizon=H, batch=512)
        return predict, fc_mod
    raise ValueError(name)


def save(res, path=OUT):
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="c2-stock,c2-w50,m2-stock,m2-w50")
    ap.add_argument("--per-series", type=int, default=6)
    ap.add_argument("--max-series", type=int, default=250)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"protocol": "s32c random-window arm, verbatim pool",
                            "per_series": args.per_series,
                            "max_series": args.max_series, "seed": 1,
                            "bins": BINS, "fills": list(FILLS), "L": L, "H": H,
                            "datasets": s32.MISSING_DS,
                            "metric": "MASE vs in-window naive-1, observed target pts",
                            "table4_fill": "linear"})

    got = s32c.collect("random", args.per_series, args.max_series)
    assert got is not None, "no windows collected"
    ctx, tgt, ok, src = got
    nf = (~np.isfinite(ctx)).mean(axis=1)
    print(f"pool: {len(ctx)} windows, ctx NaN mean {100*nf.mean():.2f}%", flush=True)
    res.setdefault("pool", {})["random"] = {
        "n": int(len(ctx)), "nan_mean": float(nf.mean()),
        "frac_gt10pct": float((nf > 0.10).mean())}

    scores = {}
    if os.path.exists(SCORES):
        scores = dict(np.load(SCORES))

    for name in args.models.split(","):
        t0 = time.time()
        predict, handle = make_model(name, args.device)
        for how in FILLS:
            key = f"{name}|{how}"
            if key not in scores:
                f_, miss = s32.fill(ctx, how)
                f_in = ctx.astype(np.float32) if how == "nan" else f_
                t1 = time.time()
                pred = predict(f_in, miss, how)
                scores[key] = s32.mase(pred, tgt, ok, ctx)
                print(f"    {name:8s} {how:7s} done ({time.time()-t1:.0f}s)",
                      flush=True)
                np.savez(SCORES, **scores)
        for lo, hi in BINS:
            sel = (nf >= lo) & (nf < hi)
            if sel.sum() < 20:
                continue
            med = {h: float(np.nanmedian(scores[f"{name}|{h}"][sel])) for h in FILLS}
            res.setdefault("bins", {})[f"{name}|random|{lo:.2f}-{hi:.2f}"] = {
                "n": int(sel.sum()), "median_by_fill": med,
                "datasets": sorted(set(src[sel].tolist()))}
            print(f"  {name:8s} nan in [{lo:.0%},{hi:.0%})  n={sel.sum():5d}  " +
                  "  ".join(f"{h}={med[h]:.3f}" for h in FILLS), flush=True)
        save(res)
        del handle
        torch.cuda.empty_cache()
        print(f"== {name} done in {time.time()-t0:.0f}s ==", flush=True)
    np.savez(SCORES, **scores)
    save(res)
    print("\nwrote", OUT, "and", SCORES)


if __name__ == "__main__":
    main()
