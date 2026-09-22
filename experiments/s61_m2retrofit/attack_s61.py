#!/usr/bin/env python
"""S61 attack surface: zeroth-order poisoning of Moirai 2.0, stock vs CPT vs
m2-retrofit (W50).

The s60_c2retrofit/attack_s60.py protocol (itself s44's threat model), adapted to
a RETAINER: for this interface the fill content ALWAYS reaches the model (flag
accompanies it), so there is exactly one path to attack -- the declared path
(content + flag = 0 at missing positions). There is no content-free declared
path, hence no x1.00-by-construction immunity cell; that is the retainer's
trade-off, stated plainly.

Hill-climb: 8 rounds x 32 candidates, missing positions only, clamped to the
window's observed [min, max]; oracle untargeted objective (maximise true NMSE).
Windows: METR-LA missing-outage test half via s26 (same split as s44/s60, 412 win).
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

sys.path.insert(0, os.path.join(EXP, "s14_metrla"))
sys.path.insert(0, os.path.join(EXP, "s26_realfill"))
sys.path.insert(0, HERE)
import run_s26_realfill as s26
import m2_iface
from eval_s61 import load_arm

OUT = os.path.join(HERE, "s61_attack.json")
SEED = s26.SEED_SPLIT
ROUNDS, CAND = 8, 32


def make_fc(m, H):
    """Declared path: fill content + flag = 0 at missing."""
    def fc(ctx, miss):
        content, obs = m2_iface.conv_ctx(ctx, miss, "declared")
        return m2_iface.median_fwd(m, content, obs, horizon=H, batch=1024)
    return fc


def attack_zeroth_order(fc, base, miss, obs_lo, obs_hi, tgt, var_floor, rng,
                        rounds=ROUNDS, cand=CAND):
    """s60's hill-climb, verbatim but for the fc(ctx, miss) signature."""
    n, L = base.shape

    def nmse(ctx):
        p = fc(ctx.reshape(-1, L), np.tile(miss, (len(ctx) // n, 1))
               if len(ctx) > n else miss)
        k = len(p) // n
        t = np.tile(tgt, (k, 1)) if k > 1 else tgt
        return s26.nmse_per_window(p, t, var_floor)

    e_base = nmse(base)
    best = base.copy()
    e_best = e_base.copy()
    width = obs_hi - obs_lo
    for r in range(rounds):
        if r == 0:
            noise = rng.uniform(obs_lo, obs_hi, size=(cand, n, L)).astype(np.float32)
        else:
            scale = width * (0.5 ** r) * 0.5
            noise = best[None] + rng.normal(0, 1, size=(cand, n, L)).astype(np.float32) \
                * scale[None]
        noise = np.clip(noise, obs_lo[None], obs_hi[None])
        cands = np.where(miss[None], noise, base[None]).astype(np.float32)
        e = nmse(cands.reshape(cand * n, L)).reshape(cand, n)
        k = e.argmax(0)
        improved = e[k, np.arange(n)] > e_best
        best[improved] = cands[k, np.arange(n)][improved]
        e_best = np.maximum(e_best, e[k, np.arange(n)])
        print(f"    round {r}: median NMSE {np.median(e_best):.4f} "
              f"(base {np.median(e_base):.4f})", flush=True)
    return best, e_base, e_best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="stock,CPT,W50")
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"attack": "zeroth-order hill-climb (s44/s60 protocol)",
                            "rounds": ROUNDS, "candidates_per_round": CAND,
                            "threat_model": "fill only, clamped to observed range",
                            "objective": "oracle untargeted (maximise true NMSE)",
                            "path": "declared (content + flag) -- the ONLY path; "
                                    "this interface has no content-free convention",
                            "data": "METR-LA missing-outage test half (s26/s44 windows)"})
    ds = s26.Metr()
    _, test = ds.split("miss")
    H = ds.H
    miss = ds.mask[test]
    tgt = ds.tgt[test]
    lo = np.array([np.nanmin(r) for r in ds.ctx_obs[test]], np.float32)[:, None]
    hi = np.array([np.nanmax(r) for r in ds.ctx_obs[test]], np.float32)[:, None]
    arms = {a: load_arm(a, args.device) for a in args.arms.split(",")}

    for a, m in arms.items():
        cell = res.setdefault("cells", {}).setdefault(a, {})
        fc = make_fc(m, H)
        def_tbl = {}
        for f in ("zero", "linear", "nan"):
            ctx = s26.fill_ctx(ds.ctx_obs[test], ds.ctx_rec[test], miss, f)
            p = fc(ctx, miss)      # declared: nan_to_num + flag (nan == zero here)
            def_tbl[f] = float(np.median(s26.nmse_per_window(p, tgt, ds.VAR_FLOOR)))
        cell["defender_nmse_median"] = def_tbl
        print(f"[{a}] defender declared-path NMSE: " +
              "  ".join(f"{f}={v:.4f}" for f, v in def_tbl.items()), flush=True)

        base = s26.fill_ctx(ds.ctx_obs[test], ds.ctx_rec[test], miss, "linear")
        t0 = time.time()
        _, e_c, e_a = attack_zeroth_order(fc, base, miss, lo, hi, tgt,
                                          ds.VAR_FLOOR, rng)
        ratio = e_a / np.maximum(e_c, 1e-12)
        cell["declared|untargeted"] = {
            "nmse_clean_median": float(np.median(e_c)),
            "nmse_attacked_median": float(np.median(e_a)),
            "damage_ratio_median": float(np.median(ratio)),
            "damage_ratio_q90": float(np.quantile(ratio, 0.9)),
            "damage_ratio_max": float(ratio.max()),
            "frac_windows_doubled": float((ratio > 2).mean())}
        print(f"[{a}] declared damage "
              f"x{cell['declared|untargeted']['damage_ratio_median']:.2f} "
              f"(q90 x{cell['declared|untargeted']['damage_ratio_q90']:.2f}, "
              f">2x on {100*cell['declared|untargeted']['frac_windows_doubled']:.0f}%) "
              f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT, "w"))
    res["n_windows"] = int(len(test))
    json.dump(res, open(OUT, "w"))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
