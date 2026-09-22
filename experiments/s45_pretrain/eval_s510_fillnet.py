#!/usr/bin/env python
"""S510: the combination punch -- learned fill x (stock vs retrofit) on REAL missingness.

S26 showed a learned fill (fillnet, tanh-bounded around the interpolant) beats fixed
fills on real curtailment, through a frozen STOCK model that cannot read the fill's
content. S53/S58 gave us a model that can. This experiment pairs them:

  P0 (stock, native)  -- fillnet can only act through the content channel (plain_fill)
  P5 (CPT retrofit)   -- fillnet acts through content + flag (fill_mask)

Same S26 protocol throughout: cal/test 50/50 split, 800 steps, center fills
(zero for penn, linear for metr), NMSE on the held-out TEST half. Fixed-fill
baselines are re-read from s55_realfill.json.
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
sys.path.insert(0, os.path.join(EXP, "s27_interface"))
sys.path.insert(0, os.path.join(EXP, "s26_realfill"))
sys.path.insert(0, HERE)
import run_s26_realfill as s26
import train_s45 as t45
import eval_s45 as e45

CENTER = {"penn": "zero", "metr": "linear"}
KIND = {"penn": "cens", "metr": "miss"}


class ArmBolt:
    """An eval_s45 arm wrapped to s26's train/eval_fillnet bolt interface."""

    def __init__(self, model, iface, device):
        self.model, self.iface, self.device = model, iface, device

    def median(self, ctx, mask=None):
        obs = mask if mask is not None else torch.ones_like(ctx)
        q = t45.quantiles_fwd(self.model, ctx, obs.float(), self.iface)
        return q[:, 4, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="P0,P5")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "s510_fillnet.json"))
    args = ap.parse_args()
    dev = args.device
    steps = 60 if args.smoke else args.steps

    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("meta", {
        "protocol": "S26 verbatim (cal/test split, fillnet class, centers)",
        "arms": {a: e45.ARM_IFACE[a] for a in args.arms.split(",")},
        "note": "plain_fill = content channel only; fill_mask = content + flag"})

    penn, metr = s26.Penn(), s26.Metr()
    models = {a: ArmBolt(e45.load_arm(a, dev), e45.ARM_IFACE[a], dev)
              for a in args.arms.split(",")}
    for ds in (penn, metr):
        kind = KIND[ds.name]
        cal, test = ds.split(kind)
        print(f"\n== {ds.name}/{kind}: {len(cal)} cal / {len(test)} test", flush=True)
        for a, bolt in models.items():
            for conv in ("plain_fill", "fill_mask"):
                t0 = time.time()
                net, hist = s26.train_fillnet(bolt, ds, cal, conv, steps=steps,
                                              center=CENTER[ds.name])
                e = s26.eval_fillnet(bolt, net, ds, test, conv,
                                     center=CENTER[ds.name])
                res.setdefault(ds.name, {}).setdefault(a, {})[conv] = {
                    "mean": float(e.mean()), "median": float(np.median(e)),
                    "per_window": [round(float(x), 5) for x in e],
                    "final_loss": float(np.mean(hist[-50:]))}
                print(f"  {ds.name:5s} {a:3s} {conv:11s} TEST mean={e.mean():9.4f} "
                      f"med={np.median(e):7.4f} ({time.time()-t0:.0f}s)", flush=True)
                json.dump(res, open(args.out, "w"))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
