#!/usr/bin/env python
"""S69 E1/E2 evaluation on the held-out 256-series pool.

Per arm (checkpoint s69_ckpt/arm_<arm>_s<seed>.pt) and per grid cell
(mech in {mcar, block}) x (rate in {0.3, 0.7}) x (q_r, q_c in {0,.25,.5,.75,1}^2):
  E1  absolute MSE and relMSE vs the arm's own clean MSE (per-window ratio, median/mean
      over windows -- eval_s45 conventions). Native arms receive the fills silently
      (content present, positions not marked); dual arms receive them declared
      (flag=missing at repaired target positions and over the whole context channel).
  E2  implied source weights by the shared two-sided perturbation measurement
      (g69.perturbed_weight, delta=0.1): w_R w.r.t. the target fill at that window's
      repaired positions, w_C w.r.t. the whole context fill. Flags stay fixed during
      content perturbation.
  F1  flag-flip (dual arms only): content fixed, repaired target positions re-flagged
      as observed -> forecast delta stats and post-flip MSE (declarations are
      load-bearing iff nonzero).

Masks and blend noises are deterministic per (series, mech, rate) and shared across the
q grid and across arms (paired design). Writes s69_eval_<arm>_s<seed>.json.
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

sys.path.insert(0, HERE)
import gen_s69_corpus as g69
import train_s69 as t69

L, H = g69.L, g69.H
ARMS = g69.ARMS


def load_arm(arm, seed, dev, ck_dir, size="tiny"):
    iface = ARMS[arm]["iface"]
    m = t69.build_model(dev, iface, size, seed=seed)
    p = os.path.join(ck_dir, f"arm_{arm}_s{seed}.pt")
    m.load_state_dict(torch.load(p, map_location=dev, weights_only=True))
    m.eval()
    for prm in m.parameters():
        prm.requires_grad_(False)
    return m, iface


def make_predict_fn(model, iface, dev, flag_x, flag_c, batch=256):
    """predict_fn(z [N, 2L] float32) -> median forecast [N, H]; flags fixed (closure)."""
    fx_t = torch.from_numpy(flag_x).to(dev)
    fc_t = torch.from_numpy(flag_c).to(dev)

    @torch.no_grad()
    def predict(z):
        outs = []
        for i in range(0, len(z), batch):
            zb = torch.from_numpy(z[i:i + batch]).to(dev)
            q = t69.quantiles_fwd(model, zb[:, :L], zb[:, L:],
                                  fx_t[i:i + batch], fc_t[i:i + batch], iface)
            outs.append(q[:, 4, :H].float().cpu().numpy())
        return np.concatenate(outs)

    return predict


@torch.no_grad()
def clean_mse(model, iface, dev, x_ctx, c_ctx, fut, batch=256):
    one = np.ones_like(x_ctx)
    outs = []
    for i in range(0, len(x_ctx), batch):
        q = t69.quantiles_fwd(model, torch.from_numpy(x_ctx[i:i + batch]).to(dev),
                              torch.from_numpy(c_ctx[i:i + batch]).to(dev),
                              torch.from_numpy(one[i:i + batch]).to(dev),
                              torch.from_numpy(one[i:i + batch].copy()).to(dev), iface)
        outs.append(q[:, 4, :H].float().cpu().numpy())
    y = np.concatenate(outs)
    return ((y - fut) ** 2).mean(1), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", default="tiny")
    ap.add_argument("--round", default="s69", choices=["s69", "s69b"])
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--ck-dir", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-win", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False   # E2 weights are O(1e-4): keep the
    torch.backends.cudnn.allow_tf32 = False         # eval forwards in strict fp32
    torch.manual_seed(6900 + args.seed)
    np.random.seed(6900 + args.seed)
    dev = args.device
    t0 = time.time()

    v2 = args.round == "s69b"
    corpus = args.corpus or os.path.join(HERE, "s69b_corpus" if v2 else "s69_corpus")
    ck_dir = args.ck_dir or os.path.join(HERE, "s69b_ckpt" if v2 else "s69_ckpt")
    n_win = args.n_win or (32 if args.smoke else 256)
    mech_set = g69.EVAL_MECHS_V2 if v2 else g69.EVAL_MECHS
    mechs = ("mcar",) if args.smoke else mech_set
    rates = (0.3,) if args.smoke else g69.EVAL_RATES
    q_grid = (0.0, 0.5, 1.0) if args.smoke else g69.Q_GRID
    out_path = args.out or os.path.join(
        HERE, f"s69b_eval_{args.arm}_s{args.seed}.json" if v2 else
        f"s69_eval_{args.arm}_s{args.seed}.json")

    x_ctx, c_ctx, fut, phi, rho = t69.eval_windows(corpus, n_win)
    model, iface = load_arm(args.arm, args.seed, dev, ck_dir, args.size)
    e_clean, _ = clean_mse(model, iface, dev, x_ctx, c_ctx, fut)
    print(f"[{args.arm}s{args.seed}] clean MSE median={np.median(e_clean):.4f} "
          f"mean={e_clean.mean():.4f} ({n_win} windows)", flush=True)

    res = {"meta": {"arm": args.arm, **ARMS[args.arm], "seed": args.seed,
                    "size": args.size, "n_win": n_win, "L": L, "H": H,
                    "round": args.round,
                    "q_grid": list(q_grid), "mechs": list(mechs), "rates": list(rates),
                    "delta": g69.DELTA, "corpus": os.path.basename(
                        corpus.rstrip("/")),
                    "code_hash": g69.code_hash(), "torch": torch.__version__,
                    "smoke": bool(args.smoke)},
           "clean": {"mse_median": float(np.median(e_clean)),
                     "mse_mean": float(e_clean.mean()),
                     "mse_w": [round(float(v), 6) for v in e_clean]},
           "cells": {}}

    ctx_cols = np.zeros(2 * L, bool)
    ctx_cols[L:] = True
    for mech in mechs:
        for rate in rates:
            # per-window masks/noises, deterministic and paired across arms/q cells
            masks, zs_clean = [], []
            for i in range(n_win):
                masks.append(g69.eval_target_mask(i, mech, rate))
            masks = np.stack(masks)                          # [N, L] bool
            n_miss = masks.sum(1)
            noises = [g69.eval_noises(i, mech, rate) for i in range(n_win)]
            nt = np.stack([n[0] for n in noises]).astype(np.float32)
            nc = np.stack([n[1] for n in noises]).astype(np.float32)
            tgt_cols = np.zeros((n_win, 2 * L), bool)
            tgt_cols[:, :L] = masks
            for q_r in q_grid:
                for q_c in q_grid:
                    fx, fc, fl_t, fl_c = [], [], [], []
                    for i in range(n_win):
                        a, b, c_, d = g69.cell_inputs(x_ctx[i], c_ctx[i], masks[i],
                                                      nt[i], nc[i], q_r, q_c)
                        fx.append(a)
                        fc.append(b)
                        fl_t.append(c_)
                        fl_c.append(d)
                    fx = np.stack(fx)
                    fc = np.stack(fc)
                    fl_t = np.stack(fl_t)
                    fl_c = np.stack(fl_c)
                    if iface == "native":                    # fills silent, no flags
                        fl_t = np.ones_like(fl_t)
                        fl_c = np.ones_like(fl_c)
                    z0 = np.concatenate([fx, fc], axis=1)    # [N, 2L]
                    predict = make_predict_fn(model, iface, dev, fl_t, fl_c)
                    y0 = predict(z0)
                    e = ((y0 - fut) ** 2).mean(1)
                    ok = e_clean > 1e-12
                    r = e[ok] / e_clean[ok]
                    wR = g69.perturbed_weight(predict, z0, tgt_cols)
                    wC = g69.perturbed_weight(predict, z0, ctx_cols)
                    key = f"{mech}|{rate}|{q_r}|{q_c}"
                    cell = {"mech": mech, "rate": rate, "q_r": q_r, "q_c": q_c,
                            "n_miss_mean": float(n_miss.mean()),
                            "mse_median": float(np.median(e)),
                            "mse_mean": float(e.mean()),
                            "rel_median": float(np.median(r)),
                            "rel_mean": float(r.mean()),
                            "rel_w": [round(float(v), 6) for v in r],
                            "wR_mean": float(wR.mean()),
                            "wR_median": float(np.median(wR)),
                            "wR_w": [round(float(v), 8) for v in wR],
                            "wC_mean": float(wC.mean()),
                            "wC_median": float(np.median(wC)),
                            "wC_w": [round(float(v), 8) for v in wC]}
                    if iface == "dual":                      # F1: re-flag as observed
                        fl_flip = fl_t.copy()
                        fl_flip[masks] = 1.0
                        predict_flip = make_predict_fn(model, iface, dev, fl_flip, fl_c)
                        y_flip = predict_flip(z0)
                        d = y_flip - y0
                        e_flip = ((y_flip - fut) ** 2).mean(1)
                        cell["flip"] = {
                            "abs_delta_mean": float(np.abs(d).mean()),
                            "abs_delta_max": float(np.abs(d).max()),
                            "mse_median": float(np.median(e_flip)),
                            "rel_median": float(np.median(e_flip[ok] / e_clean[ok]))}
                    res["cells"][key] = cell
                    print(f"  {key}: relMSE {cell['rel_median']:.3f} "
                          f"wR {cell['wR_mean']:.4f} wC {cell['wC_mean']:.4f}"
                          + (f" flip|d| {cell['flip']['abs_delta_mean']:.4f}"
                             if "flip" in cell else ""), flush=True)
            json.dump(res, open(out_path, "w"))
    res["meta"]["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(out_path, "w"))
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
