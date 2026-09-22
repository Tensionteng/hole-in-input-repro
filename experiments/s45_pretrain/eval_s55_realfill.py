#!/usr/bin/env python
"""S55: the winning arms re-scored on REAL missingness (DESIGN2's second axis).

DESIGN2 pre-registered: "the winner is re-scored on real missingness (Penmanshiel
curtailment and METR-LA outages, S26/S29 windows) before any 'best recipe' sentence
is written; a recipe that wins on the synthetic grid and loses on real missingness is
reported as such." This scores, on the S26 held-out TEST halves:

  P0  stock bolt-tiny (reference, native interface)
  A   from-scratch, filtered regime (the status-quo replica)
  H   from-scratch, the winning recipe (block outages + fill-diverse)
  P5  CPT-from-stock on the H recipe (the shipped-checkpoint retrofit)

Fixed fills only (keep/zero/linear; nan additionally for the native arms, where it is
the stock declaration semantics -- the dual arms' equivalent is zero+flag=0, which is
already the 'zero' row with obs flags). NMSE per window, the S26 var floors.
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
sys.path.insert(0, os.path.join(EXP, "s26_realfill"))
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import run_s26_realfill as s26
import train_s45 as t45
import eval_s45 as e45

ARMS = {"P0": "native", "A": "native", "H": "dual", "P5": "dual",
        "P6": "dual", "Q50": "dual"}
FILLS = {"penn": ("keep", "zero", "linear"),
         "metr": ("keep", "zero", "linear", "nan")}
KIND = {"penn": "cens", "metr": "miss"}


@torch.no_grad()
def median_np(model, ctx, obs, iface, H, dev, batch=256):
    outs = []
    for i in range(0, len(ctx), batch):
        c = torch.from_numpy(ctx[i:i + batch].astype(np.float32)).to(dev)
        o = torch.from_numpy(obs[i:i + batch].astype(np.float32)).to(dev)
        q = t45.quantiles_fwd(model, c, o, iface)
        outs.append(q[:, 4, :H].float().cpu().numpy())
    return np.concatenate(outs)


def score_arm(res, model, arm, iface, ds, dev):
    kind = KIND[ds.name]
    _, test = ds.split(kind)
    clean_kind = "ctrl"
    _, test_c = ds.split(clean_kind)
    # clean reference on the held-out ctrl half: needed because the from-scratch arms
    # can be far off-distribution on these datasets (their NMSEs only read against
    # their own clean level)
    ctx_c = s26.fill_ctx(ds.ctx_obs[test_c], ds.ctx_rec[test_c], ds.mask[test_c], "linear")
    obs_c = np.ones_like(ctx_c)
    pred_c = median_np(model, ctx_c, obs_c, iface, ds.H, dev)
    e_c = s26.nmse_per_window(pred_c, ds.tgt[test_c], ds.VAR_FLOOR)
    res.setdefault(ds.name, {}).setdefault(arm, {})["_ctrl_clean"] = {
        "mean": float(e_c.mean()), "median": float(np.median(e_c))}
    print(f"  {ds.name:5s} {arm:3s} CTRL-CLEAN NMSE mean={e_c.mean():9.4f} "
          f"med={np.median(e_c):7.4f}", flush=True)
    for f in FILLS[ds.name]:
        ctx = s26.fill_ctx(ds.ctx_obs[test], ds.ctx_rec[test], ds.mask[test], f)
        if f == "nan" and iface != "native":
            continue                       # nan is the native (rank-0) declaration
        obs = (~np.isnan(ctx)).astype(np.float32) if f == "nan" \
            else (~ds.mask[test]).astype(np.float32)
        pred = median_np(model, ctx, obs, iface, ds.H, dev)
        e = s26.nmse_per_window(pred, ds.tgt[test], ds.VAR_FLOOR)
        res.setdefault(ds.name, {}).setdefault(arm, {})[f] = {
            "mean": float(e.mean()), "median": float(np.median(e)),
            "per_window": [round(float(x), 5) for x in e]}
        print(f"  {ds.name:5s} {arm:3s} {f:7s} NMSE mean={e.mean():9.4f} "
              f"med={np.median(e):7.4f}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="P0,A,H,P5")
    ap.add_argument("--n-win", type=int, default=0, help="truncate test half (smoke)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "s55_realfill.json"))
    args = ap.parse_args()
    dev = args.device

    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("meta", {
        "windows": "S26 TEST halves (per-turbine/-sensor 50/50, seed 20250812)",
        "penn": "CTX=144 H=24, curtailment (oracle mask), NMSE var floor 100",
        "metr": "CTX=512 H=64, outages (exact zeros), NMSE var floor 4.0",
        "arms": ARMS, "note": "NMSE per window; fills fixed (no learned fill)"})

    penn, metr = s26.Penn(), s26.Metr()
    if args.n_win:
        # smoke: truncate via monkey-patched split
        for ds in (penn, metr):
            orig = ds.split
            ds.split = lambda kind, orig=orig: (lambda c, t: (c, t[:args.n_win]))(*orig(kind))

    models = {a: e45.load_arm(a, dev) for a in args.arms.split(",")}
    for ds in (penn, metr):
        for a, m in models.items():
            res = score_arm(res, m, a, ARMS[a], ds, dev)
            json.dump(res, open(args.out, "w"))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
