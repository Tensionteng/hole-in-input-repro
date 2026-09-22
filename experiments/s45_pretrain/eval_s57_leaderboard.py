#!/usr/bin/env python
"""S57: the missingness leaderboard -- our retrofit vs other TSFMs as shipped.

Same grid as the main table's hardest column: 9 benchmarks x 4 mechanisms x rate
0.7, fills {zero, linear, oracle}, on each model's OWN declared path (nan/flag
semantics per its native interface), 150 windows per cell. relMSE vs the model's
own clean accuracy, exactly as the sweep9 convention; clean MSEs stored too, so
absolute performance is reconstructable.

Models: bolt-base stock (P0@base), our retrofit (P5@base; Q50@base appended when
the P8@base run lands), Moirai 2.0, TiRex, FlowState, Timer-XL -- every one with
its native missingness convention, none retrained by us.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s27_interface"))
sys.path.insert(0, os.path.join(EXP, "s35_breadth"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
sys.path.insert(0, os.path.join(EXP, "s46_moirai2"))
sys.path.insert(0, os.path.join(EXP, "s49_2025models"))
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import run_s35_breadth as s35
import run_s30_crossmodel as s30mod
import eval_s45 as e45
from run_s46_moirai2 import Moirai2
from run_s49_2025 import TiRex, FlowState, TimerXL

L, H = s25.L, 64
DS9 = e45.DS9
MECHS = e45.MECHS
FILLS = ("zero", "linear", "oracle")


def bolt_fc(arm, size, dev):
    m = e45.load_arm(arm, dev, size)
    iface = e45.ARM_IFACE[arm.partition("@")[0]]

    def fc(ctx, miss, conv):
        cx, ob = e45.conv_ctx(ctx, miss, conv, iface)
        return e45.median_fwd(m, cx, ob, iface if conv == "declared" else "plain")[:, :H]
    return fc


def wrap_fc(model):
    def fc(ctx, miss, conv):
        return model.fc(ctx, miss, "nan" if conv == "declared" else "plain")[:, :H]
    return fc


REGISTRY = {
    "bolt-base-stock": lambda dev: bolt_fc("P0", "base", dev),
    "P5-base": lambda dev: bolt_fc("P5", "base", dev),
    "P8-base": lambda dev: bolt_fc("P8", "base", dev),
    "Q50-base": lambda dev: bolt_fc("Q50", "base", dev),
    "moirai2": lambda dev: wrap_fc(Moirai2(dev)),
    "tirex": lambda dev: wrap_fc(TiRex(dev)),
    "flowstate": lambda dev: wrap_fc(FlowState(dev)),
    # TimesFM 2.5: its declared path (nan) triggers its OWN interpolation -- the
    # "overwrite" convention, scored as-is
    "timesfm": lambda dev: wrap_fc(s30mod.TimesFM(dev)),
    # Timer-XL's declared (nan) path emits NaN logits -- its only usable path is
    # plain fill without declaration; reported as such
    "timerxl-plain": lambda dev: (lambda m: lambda ctx, miss, conv: m.fc(
        ctx, miss, "plain")[:, :H])(TimerXL(dev)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--n-win", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = args.device

    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("meta", {"grid": "DS9 x 4 mechs x rate 0.7 x {zero,linear,oracle}",
                            "n_win": args.n_win, "conv": "each model's declared path"})
    for name in args.models.split(","):
        if name in REGISTRY:
            fc = REGISTRY[name](dev)
        else:
            # "Q50-base@20260901" -> bolt arm Q50@20260901 at size base
            base_name, _, seed = name.partition("@")
            arm, _, size = base_name.rpartition("-")
            fc = bolt_fc(f"{arm}@{seed}" if seed else arm, size, dev)
        res.setdefault("clean_mse", {})
        for ds in DS9:
            X, st = s35.load(ds, args.n_win)
            if X is None:
                continue
            _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
            zmask = np.zeros_like(cl, bool)
            pred_c = fc(cl, zmask, "declared")
            base = ((pred_c - gt_c) ** 2).mean(1)
            res["clean_mse"].setdefault(name, {})[ds] = {
                "median": float(np.median(base)), "mean": float(base.mean())}
            for mech in MECHS:
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, 0.7, H,
                                                            fill="linear")
                for fill in FILLS:
                    ctx = s25.fill_context(clean, mask, fill)
                    e = ((fc(ctx, mask, "declared") - gt) ** 2).mean(1)
                    ok = base > 1e-12
                    r = e[ok] / base[ok]
                    res.setdefault("grid", {})[f"{ds}|{mech}|0.7|{fill}|{name}"] = {
                        "median": float(np.median(r)), "mean": float(r.mean())}
                print(f"  {name:15s} {ds:11s} {mech:13s} " + " ".join(
                    f"{f}:{res['grid'][f'{ds}|{mech}|0.7|{f}|{name}']['median']:.2f}"
                    for f in FILLS), flush=True)
            json.dump(res, open(args.out, "w"))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
