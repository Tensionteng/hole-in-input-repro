#!/usr/bin/env python
"""S64: the three retrofit families under the EXACT Table-3 (S57) leaderboard protocol.

Protocol (mirrors s45_pretrain/eval_s57_leaderboard.py line for line):
  DS9 (9 benchmarks, s35.load, n_win=150, seed 20250810, last-20% windows, channels
  as univariate series, electricity/traffic subsampled to 24 channels) x 4 mechanisms
  (mcar, block, mnar_high, mnar_extreme) at rate 0.7 x fills {zero, linear, oracle},
  each model on its OWN declared path, median paired per-window relMSE vs the model's
  OWN clean-context error (clean = declared pass with an all-observed mask).

Arms (all fp32 strict, same windows/masks for every arm):
  m2-stock   Moirai 2.0 R-small as shipped (s61 m2_iface; declared = "nan" conv:
             content nan_to_num + observed_mask = ~miss -- identical to s46/s57)
  m2-w50     m2-retrofit: WiSE-FT 0.5*stock + 0.5*CPT (s61_ckpt/arm_W50.pt), same iface
  c2-stock   Chronos-2 as shipped, NATIVE interface (s60 c2_iface replica, gated
             bit-exact vs the stock module by run_s60_gate.py); declared = explicit
             mask, content zeroed by the model
  c2-w50     c2-retrofit: WiSE-FT 0.5*stock + 0.5*CPT (s60_ckpt/arm_W50.pt), DUAL
             interface (fill content kept + flag kept); declared = fill + true flag

The script writes raw cells (per ds|mech|rate|fill|arm median/mean + per-window
ratios) to s64_leaderboard_cells.json. aggregate_s64_leaderboard.py builds the
paper-format rows and runs the reproduction gates.
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

torch.backends.cuda.matmul.allow_tf32 = False     # strict fp32 eval (as s60/s61)
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s35_breadth"))
sys.path.insert(0, os.path.join(EXP, "s60_c2retrofit"))
sys.path.insert(0, os.path.join(EXP, "s61_m2retrofit"))
import run_s25_twofloor as s25
import run_s35_breadth as s35
import c2_iface
import m2_iface

L, H = s25.L, 64
# verbatim from eval_s45.py -- NOT imported, because eval_s45 imports train_s45,
# whose module top sets allow_tf32=True and would silently undo the strict-fp32
# flags above (this exact bug bit the first s64 run; caught by the stock gates)
DS9 = ("ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness")
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
FILLS = ("zero", "linear", "oracle")             # s57's FILLS; oracle = the "perfect" fill
S60_CK = os.path.join(EXP, "s60_c2retrofit", "s60_ckpt")
S61_CK = os.path.join(EXP, "s61_m2retrofit", "s61_ckpt")
OUT = os.path.join(HERE, "s64_leaderboard_cells.json")


# ------------------------------------------------------------------ arms ----

def make_fc(name, dev):
    """Returns fc(ctx, miss, conv) -> [n, H] median forecast, s57 wrap semantics."""
    if name.startswith("m2"):
        fc_mod = m2_iface.load_stock(dev)
        if name == "m2-w50":
            sd = torch.load(os.path.join(S61_CK, "arm_W50.pt"), map_location=dev,
                            weights_only=True)
            fc_mod.module.load_state_dict(sd)
            fc_mod.eval()

        def fc(ctx, miss, conv):
            # s57's wrap_fc: declared -> the model's "nan" (declaration) path;
            # batch=64 reproduces s57_moirai2.json bit-exactly
            content, obs = m2_iface.conv_ctx(
                ctx, miss, "nan" if conv == "declared" else "plain")
            return m2_iface.median_fwd(fc_mod, content, obs, horizon=H, batch=64)
        return fc, fc_mod
    if name.startswith("c2"):
        model = c2_iface.load_stock(dev)
        iface = "native" if name == "c2-stock" else "dual"
        if name == "c2-w50":
            sd = torch.load(os.path.join(S60_CK, "arm_W50.pt"), map_location=dev,
                            weights_only=True)
            model.load_state_dict(sd)
            model.eval()

        def fc(ctx, miss, conv):
            cx, ob = c2_iface.conv_ctx(ctx, miss, conv, iface)
            return c2_iface.median_fwd(model, cx, ob, iface, horizon=H, batch=512)
        return fc, model
    raise ValueError(name)


def save(res, path=OUT):
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="m2-stock,m2-w50,c2-stock,c2-w50")
    ap.add_argument("--n-win", type=int, default=150)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    dev = args.device
    assert not torch.backends.cuda.matmul.allow_tf32, \
        "TF32 got re-enabled by an import -- eval must be strict fp32"

    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("meta", {"grid": "DS9 x 4 mechs x rate 0.7 x {zero,linear,oracle}",
                            "n_win": args.n_win, "conv": "each model's declared path",
                            "protocol": "eval_s57_leaderboard.py exact",
                            "L": L, "H": H, "seed": s25.SEED, "dtype": "fp32 strict",
                            "arms": {"m2-stock": "Moirai 2.0 R-small stock (declared=nan)",
                                     "m2-w50": "m2-retrofit WiSE-FT0.5 (declared=nan)",
                                     "c2-stock": "Chronos-2 stock, native iface",
                                     "c2-w50": "c2-retrofit WiSE-FT0.5, dual iface"}})
    for name in args.arms.split(","):
        t0 = time.time()
        fc, handle = make_fc(name, dev)
        res.setdefault("clean_mse", {})
        for ds in DS9:
            X, st = s35.load(ds, args.n_win)
            if X is None:
                print(f"  {name} {ds}: no valid windows, skipped", flush=True)
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
                        "median": float(np.median(r)), "mean": float(r.mean()),
                        "w": [round(float(x), 6) for x in r]}
                print(f"  {name:8s} {ds:11s} {mech:13s} " + " ".join(
                    f"{f}:{res['grid'][f'{ds}|{mech}|0.7|{f}|{name}']['median']:.2f}"
                    for f in FILLS), flush=True)
            save(res, args.out)
        del handle
        torch.cuda.empty_cache()
        print(f"== {name} done in {time.time()-t0:.0f}s ==", flush=True)
    res["meta"]["wall_clock_note"] = "per-arm times printed to log"
    save(res, args.out)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
