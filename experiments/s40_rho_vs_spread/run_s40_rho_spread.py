#!/usr/bin/env python
"""S40: does a checkpoint's rho predict how much its GIFT-Eval score moves with the fill?

Reuses S32's corpus reader, window sampler, fill conventions and MASE unchanged, and S36's
model wrappers, so the only new thing is the model axis. See s40_notes.md.
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import torch

for d in ("s25_twofloor", "s30_crossmodel", "s32_gifteval", "s36_models"):
    sys.path.insert(0, os.path.join(EXP, d))
import run_s32_gifteval as s32
import run_s36_models as s36

FILLS = s32.FILLS if hasattr(s32, "FILLS") else ("nan", "linear", "zero", "ffill")
OUT = os.path.join(HERE, "s40_results.json")

# key -> (S36/S30 registry key, display name, params)
MODELS = {
    "bolt_tiny":   ("chronos-bolt-tiny", 8_700_000),
    "bolt_mini":   ("chronos-bolt-mini", 21_000_000),
    "bolt_small":  ("chronos-bolt-small", 48_000_000),
    "bolt_base":   ("chronos-bolt-base", 205_000_000),
    "t5_base":     ("chronos-t5-base", 200_000_000),
    "moirai_s":    ("moirai-1.1-R-small", 14_000_000),
    "moirai_l":    ("moirai-1.1-R-large", 311_000_000),
    "moe_s":       ("moirai-moe-1.0-R-small", 117_000_000),
    "moe_b":       ("moirai-moe-1.0-R-base", 935_000_000),
    "timemoe_50":  ("TimeMoE-50M", 50_000_000),
}


def forecast(model, ctx, miss, how):
    """Route a fill convention to the model's matching input path.

    A model with no declared path gets the plain path for every convention; that is what its
    users get too, and the resulting spread is real rather than an artefact."""
    conv = "nan" if (how == "nan" and "nan" in model.convs) else "plain"
    return model.fc(ctx, miss, conv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--max-series", type=int, default=200)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    np.random.seed(0)
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("meta", {"fills": list(FILLS), "datasets": s32.MISSING_DS,
                            "max_series": args.max_series, "L": s32.L, "H": s32.H,
                            "metric": "MASE vs in-window naive-1 on observed context points",
                            "note": "S32 protocol, S36 model axis"})

    # cache the corpus once: every checkpoint must see identical windows
    data = {}
    for ds in s32.MISSING_DS:
        try:
            w = s32.windows(s32.read_series(ds, max_series=args.max_series))
        except Exception as e:
            print(f"SKIP {ds}: {type(e).__name__}", flush=True)
            continue
        if w is None:
            print(f"SKIP {ds}: no usable window", flush=True)
            continue
        data[ds] = w
        print(f"  {ds:36s} {len(w[0]):5d} windows, "
              f"{(~np.isfinite(w[0])).mean()*100:5.2f}% ctx NaN", flush=True)

    for key in args.models.split(","):
        disp, npar = MODELS[key]
        try:
            model = s36.REGISTRY[key]("cuda")
        except Exception as e:
            print(f"SKIP {key}: {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        print(f"\n== {disp} ==", flush=True)
        for ds, (ctx, tgt, ok) in data.items():
            line = f"  {ds:36s} "
            for how in FILLS:
                f_, miss = s32.fill(ctx, how)
                f_in = (np.where(np.isfinite(ctx), ctx, np.nan).astype(np.float32)
                        if how == "nan" else f_)
                try:
                    t0 = time.time()
                    p = forecast(model, f_in, miss, how)
                    v = float(np.nanmedian(s32.mase(p, tgt, ok, ctx)))
                except Exception as e:
                    v = float("nan")
                    print(f"[{how} FAIL {type(e).__name__}: {str(e)[:50]}]", end="",
                          flush=True)
                res.setdefault("cells", {})[f"{key}|{ds}|{how}"] = {
                    "mase_median": v, "n": int(len(ctx))}
                line += f"{how}={v:7.3f} "
            print(line, flush=True)
        res.setdefault("params", {})[key] = npar
        res.setdefault("name", {})[key] = disp
        del model
        torch.cuda.empty_cache()
        json.dump(res, open(args.out + ".tmp", "w"))
        os.replace(args.out + ".tmp", args.out)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
