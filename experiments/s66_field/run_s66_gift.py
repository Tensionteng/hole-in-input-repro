#!/usr/bin/env python
"""S66: GIFT-Eval dose-response (Table-4 protocol) for the three shipped 2025
zero-shot models -- FlowState r1, TiRex 1.1, TimesFM 2.5 -- on the EXACT s32c
pipeline, plus a bolt-base stock gate arm.

Protocol (run_s32c_stratified.py, random-window arm, reused verbatim -- identical
to s64_bench3/run_s64_gift.py):
  windows pooled from the nine missing-containing GIFT-Eval subsets
  (s32.MISSING_DS), random-window protocol (per_series=6, max_series=250, seed=1 --
  the identical pool as s32c_results.json, n=7725), context 512, horizon 64, scored
  on OBSERVED target points, MASE against an in-window naive-1 baseline computed on
  the observed context points. Stratified by the window's OWN context-NaN fraction
  into bins (0-1%, 1-5%, 5-15%, 15-30%, >30%). Table 4 reports the LINEAR fill;
  all four fills are scored and kept.

Fill routing (s32.forecast, unchanged): how == "nan" -> the model's declared/NaN
input convention; fixed fills (linear/zero/ffill) -> the "plain" convention, i.e.
the filled array is fed as the series content. What each 2025 model then does with
it (measured in s30/s49 by the permutation probe):
  flowstate  plain: uses the fill content (ratio 0.85). nan: declared path zeroes
             the content and keeps only the missingness flag (ratio 0.00) --
             statistics from observed points only.
  tirex      plain: uses the fill content (ratio 0.96). nan: same (a)+flag
             declared path as FlowState (ratio 0.00); fill content discarded.
  timesfm    plain: uses the fill content (ratio 0.86). nan: TimesFM 2.5 applies
             its OWN internal interpolation over the NaN positions (ratio 0.00 to
             any fill we supply); s32c's timesfm nan==linear rows in the low-NaN
             strata show its internal fill is near-equivalent to np.interp there.
So the "linear" row for all three is genuinely the model forecasting our
linearly-filled context; the "nan" row is each model's own declared missing-data
convention, which never sees our fill.

Gate arm: bolt-base stock goes through s32.get_model + s32.forecast -- the
verbatim s32c code path -- and must reproduce s32c_results.json's
bolt-base|random bin medians to <=1e-5 relative deviation (all four fills)
before the three new models are trusted. A second, supplementary gate compares
the new timesfm run against s32c's timesfm|random rows (validates the s30
wrapper and the HF-cache weights).
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
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
sys.path.insert(0, os.path.join(EXP, "s32_gifteval"))
sys.path.insert(0, os.path.join(EXP, "s49_2025models"))
import run_s25_twofloor as s25
import run_s30_crossmodel as s30
import run_s32_gifteval as s32
import run_s32c_stratified as s32c

L, H = s32.L, s32.H
FILLS = s32.FILLS
BINS = s32c.BINS
OUT = os.path.join(HERE, "s66_gift_dose.json")
SCORES = os.path.join(HERE, "s66_gift_scores.npz")


def make_model(name, dev):
    """Returns predict(ctx_filled, miss, how) -> [n, H] median forecast."""
    if name == "bolt-base":
        # gate arm: the verbatim s32c path (s30.Bolt via s32.get_model/forecast)
        model = s32.get_model("bolt-base")

        def predict(ctx_filled, miss, how):
            return s32.forecast(model, ctx_filled, miss, how)
        return predict, model
    if name == "timesfm":
        model = s30.TimesFM(dev)
    elif name == "tirex":
        from run_s49_2025 import TiRex
        model = TiRex(dev)
    elif name == "flowstate":
        from run_s49_2025 import FlowState
        model = FlowState(dev)
    else:
        raise ValueError(name)

    def predict(ctx_filled, miss, how):
        # s32.forecast's routing for every non-Bolt wrapper, verbatim
        return model.fc(ctx_filled, miss, "nan" if how == "nan" else "plain")
    return predict, model


def save(res, path=OUT):
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bolt-base,timesfm,tirex,flowstate")
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
                            "table4_fill": "linear",
                            "models": "bolt-base (gate), timesfm-2.5-200m, tirex-1.1, "
                                      "flowstate-r1"})

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

    fills = {}                      # fill is deterministic; compute each once
    for name in args.models.split(","):
        t0 = time.time()
        predict, handle = make_model(name, args.device)
        for how in FILLS:
            key = f"{name}|{how}"
            if key not in scores:
                if how not in fills:
                    fills[how] = s32.fill(ctx, how)
                f_, miss = fills[how]
                f_in = ctx.astype(np.float32) if how == "nan" else f_
                t1 = time.time()
                pred = predict(f_in, miss, how)
                scores[key] = s32.mase(pred, tgt, ok, ctx)
                print(f"    {name:10s} {how:7s} done ({time.time()-t1:.0f}s)",
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
            print(f"  {name:10s} nan in [{lo:.0%},{hi:.0%})  n={sel.sum():5d}  " +
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
