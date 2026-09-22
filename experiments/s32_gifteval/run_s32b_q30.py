#!/usr/bin/env python
"""Q50 (our retrofit, bolt-base) on the full S32 GIFT-Eval sweep, part-B protocol.

Reuses run_s32_gifteval's series loading / windowing / fill / MASE verbatim, and
reads the stored bolt-base cells from s32_partB.json as the reference, so the
comparison is paired per dataset per fill. Output: s32_q30.json.
"""
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
sys.path.insert(0, os.path.join(EXP, "s27_interface"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
sys.path.insert(0, os.path.join(EXP, "s45_pretrain"))
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import run_s32_gifteval as s32
import eval_s45 as e45

H = s32.H


class Q50Bolt:
    """Our retrofit, exposing the S32 model interface. "plain" (a fixed fill is
    supplied) is routed to our declared path; "nan" zero-fills + declares."""

    name = "q30-base"
    n_params = 205_000_000

    def __init__(self, dev):
        self.m = e45.load_arm("Q30", dev, "base")
        self.dev = dev

    @torch.no_grad()
    def fc(self, ctx, miss, conv, batch=256):
        if conv == "nan":
            obs = np.isfinite(ctx).astype(np.float32)
            c = np.nan_to_num(ctx).astype(np.float32)
        else:
            obs = (~miss).astype(np.float32)
            c = ctx.astype(np.float32)
        return e45.median_fwd(self.m, c, obs, "dual")[:, :H]


def main():
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    model = Q50Bolt("cuda")
    res = {"meta": {"protocol": "S32 part B verbatim; q50 declared path on fills, "
                                "zero-fill+declare on nan",
                    "reference": "bolt-base cells from s32_partB.json"},
           "cells": {}}
    for dsfreq in s32.MISSING_DS + s32.CLEAN_DS:
        try:
            ser = s32.read_series(dsfreq, max_series=300)
            w = s32.windows(ser)
        except Exception as e:
            print(f"  {dsfreq:44s} READ-FAIL {type(e).__name__}", flush=True)
            continue
        if w is None:
            print(f"  {dsfreq:44s} no usable windows", flush=True)
            continue
        ctx, tgt, ok = w
        line = f"  {dsfreq:44s} n={len(ctx):4d} "
        for how in s32.FILLS:
            f_, miss = s32.fill(ctx, how)
            try:
                if how == "nan":
                    f_in = np.where(np.isfinite(ctx), ctx, np.nan).astype(np.float32)
                    pred = model.fc(f_in, miss, "nan")
                else:
                    pred = model.fc(f_, miss, "declared")
                v = float(np.nanmedian(s32.mase(pred, tgt, ok, ctx)))
            except Exception as e:
                v = float("nan")
                print(f"[{how} FAIL {type(e).__name__}: {str(e)[:60]}]", end="",
                      flush=True)
            res["cells"][f"q30-base|{dsfreq}|{how}"] = {
                "mase_median": v, "n": int(len(ctx)),
                "ctx_nan_frac": float((~np.isfinite(ctx)).mean())}
            line += f"{how}={v:7.3f} "
        print(line, flush=True)
        json.dump(res, open(os.path.join(HERE, "s32_q30.json"), "w"), indent=1)
    print("\nwrote s32_q30.json")


if __name__ == "__main__":
    main()
