#!/usr/bin/env python
"""S63 eval: the blend-only C2 ablation arm on the paper's grid (s60 protocol).

Arm:
  BLEND   S63 continued-pretrained checkpoint (mechdiv_filldiv -- S60's mechdiv
          with declaration-endpoint share 0; dual interface)

References (NOT re-evaluated here -- s60's eval is deterministic: same seeds,
same windows, strict fp32): stock, CPT (full recipe), W50, read from
s60_c2retrofit/s60_{grid,probe,leaderboard}.json by analyze_s63.py.

Parts (each writes its own JSON, s60 naming with the s63_ prefix):
  clean + grid   9 benchmarks x 4 mechanisms x rates {0.3, 0.7} x alphas {0, .5, 1},
                 declared and plain views, own-clean relMSE with per-window ratios
  probe          permutation rho, 3 ds x 3 mechs at rate 0.3, 60 windows
  leaderboard    9 benchmarks x 4 mechanisms at 70% missing under {zero, linear,
                 nan} fills (nan = declaration endpoint for the dual arm)

All eval strict fp32 (no TF32), seeds/windows identical to s60's.
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
sys.path.insert(0, os.path.join(EXP, "s35_breadth"))
sys.path.insert(0, os.path.join(EXP, "s60_c2retrofit"))
import run_s25_twofloor as s25
import run_s35_breadth as s35
import c2_iface

L, H = s25.L, 64
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES = (0.3, 0.7)
ALPHAS = (0.0, 0.5, 1.0)
DS9 = ("ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness")
CK = os.path.join(HERE, "s63_ckpt")
ARM_IFACE = {"BLEND": "dual"}
ARM_FILE = {"BLEND": "arm_BLEND.pt"}


def load_arm(arm, dev):
    m = c2_iface.load_stock(dev)
    sd = torch.load(os.path.join(CK, ARM_FILE[arm]), map_location=dev,
                    weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m


def run_conv(m, arm_iface, ctx, miss, conv, batch=512):
    """Median forecast [n, H] under a convention (see c2_iface.conv_ctx)."""
    if conv == "nan":
        cx, ob = c2_iface.conv_ctx(ctx, miss, "nan", "native")
        return c2_iface.median_fwd(m, cx, ob, "native", horizon=H, batch=batch)
    cx, ob = c2_iface.conv_ctx(ctx, miss, conv, arm_iface)
    return c2_iface.median_fwd(m, cx, ob, arm_iface, horizon=H, batch=batch)


def blend(clean, lin, mask, a):
    out = lin.copy()
    out[mask] = (1.0 - a) * lin[mask] + a * clean[mask]
    return out


def perm_probe(m, arm_iface, filled, mask, conv, rng, n_perm=4):
    """S30/S36 permutation probe through the interface switch."""
    y0 = run_conv(m, arm_iface, filled, mask, conv, batch=512)
    scale = np.sqrt((y0 ** 2).mean(1)) + 1e-9
    dp, dr = [], []
    for _ in range(n_perm):
        cp, cr = filled.copy(), filled.copy()
        for i in range(len(filled)):
            idx = np.flatnonzero(mask[i])
            if len(idx) < 2:
                continue
            cp[i, idx] = filled[i, idx][rng.permutation(len(idx))]
            src = filled[i, ~mask[i]]
            cr[i, idx] = rng.choice(src, size=len(idx)) if len(src) else 0.0
        dp.append(np.sqrt(((run_conv(m, arm_iface, cp, mask, conv) - y0) ** 2).mean(1)) / scale)
        dr.append(np.sqrt(((run_conv(m, arm_iface, cr, mask, conv) - y0) ** 2).mean(1)) / scale)
    dp, dr = np.mean(dp, 0), np.mean(dr, 0)
    return {"rho": float(dp.mean() / max(dr.mean(), 1e-15)),
            "perm": float(dp.mean()), "redraw": float(dr.mean())}


def save(res, path):
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="clean,probe,grid,leaderboard")
    ap.add_argument("--arms", default="BLEND")
    ap.add_argument("--n-win", type=int, default=150)
    ap.add_argument("--device", default="cuda:2")
    args = ap.parse_args()
    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    dev = args.device
    arms = {a: load_arm(a, dev) for a in args.arms.split(",")}
    print("loaded arms:", list(arms), flush=True)

    # ------------------------------------------------------- clean + grid ----
    if "clean" in args.parts or "grid" in args.parts:
        out = os.path.join(HERE, "s63_grid.json")
        res = json.load(open(out)) if os.path.exists(out) else {}
        res.setdefault("meta", {"grid": "DS9 x 4 mechs x rates {0.3,0.7} x alphas "
                                "{0,0.5,1} x {declared,plain}", "n_win": args.n_win,
                                "relMSE": "per-window vs each arm's OWN clean",
                                "arms": ARM_IFACE, "H": H, "dtype": "fp32 strict",
                                "references": "stock/CPT/W50 cells live in "
                                "s60_c2retrofit/s60_grid.json (deterministic, "
                                "same seeds/windows)"})
        for ds in DS9:
            X, st = s35.load(ds, args.n_win)
            if X is None:
                print(f"  grid {ds}: no valid windows, skipped", flush=True)
                continue
            _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
            zmask = np.zeros_like(cl, bool)
            base = {}
            for a, m in arms.items():
                e = ((run_conv(m, ARM_IFACE[a], cl, zmask, "declared") - gt_c) ** 2).mean(1)
                base[a] = e
                res.setdefault("clean", {}).setdefault(a, {})[ds] = {
                    "mse_median": float(np.median(e)), "mse_mean": float(e.mean())}
            print(f"  clean {ds:11s} " + " ".join(
                f"{a}:{res['clean'][a][ds]['mse_median']:.4g}" for a in arms), flush=True)
            if "grid" not in args.parts:
                save(res, out)
                continue
            for mech in MECHS:
                for rate in RATES:
                    clean, lin, mask, gt = s25.build_eval_batch(
                        X, st, mech, rate, H, fill="linear")
                    for al in ALPHAS:
                        ctx = blend(clean, lin, mask, al)
                        for a, m in arms.items():
                            for view in ("declared", "plain"):
                                e = ((run_conv(m, ARM_IFACE[a], ctx, mask, view)
                                      - gt) ** 2).mean(1)
                                ok = base[a] > 1e-12
                                r = e[ok] / base[a][ok]
                                res.setdefault("grid", {})[
                                    f"{ds}|{mech}|{rate}|{al}|{a}|{view}"] = {
                                    "median": float(np.median(r)),
                                    "mean": float(r.mean()),
                                    "w": [round(float(x), 6) for x in r]}
                    print(f"  grid {ds:11s} {mech:13s} p={rate} " + " ".join(
                        f"{a}:{res['grid'][f'{ds}|{mech}|{rate}|0.0|{a}|declared']['median']:.2f}"
                        f"->{res['grid'][f'{ds}|{mech}|{rate}|1.0|{a}|declared']['median']:.2f}"
                        for a in arms), flush=True)
                    save(res, out)
        save(res, out)
        print("wrote", out, flush=True)

    # -------------------------------------------------------------- probe ----
    if "probe" in args.parts:
        out = os.path.join(HERE, "s63_probe.json")
        res = json.load(open(out)) if os.path.exists(out) else {}
        res.setdefault("meta", {"protocol": "S30/S36: 3 ds x 3 mechs, rate 0.3, "
                                "60 win, linear fill, 4 perms", "arms": ARM_IFACE,
                                "convs": ["declared", "plain", "nan"]})
        for ds, path in s25.DATASETS.items():
            X, st = s25.load_windows(path, 60, s25.SEED, H)
            for mech in ("mcar", "block", "mnar_high"):
                _, lin, mask, _ = s25.build_eval_batch(X, st, mech, 0.3, H,
                                                       fill="linear")
                for a, m in arms.items():
                    for conv in ("declared", "plain", "nan"):
                        rng = np.random.default_rng(s25.SEED)
                        r = perm_probe(m, ARM_IFACE[a], lin, mask, conv, rng)
                        res.setdefault("probe", {})[f"{ds}|{mech}|{a}|{conv}"] = r
                print(f"  probe {ds:8s} {mech:10s} " + " ".join(
                    f"{a}.{c}:{res['probe'][f'{ds}|{mech}|{a}|{c}']['rho']:.3f}"
                    for a in arms for c in ("declared", "plain", "nan")), flush=True)
                save(res, out)
        agg = {}
        for a in arms:
            for conv in ("declared", "plain", "nan"):
                v = [c["rho"] for k, c in res["probe"].items()
                     if k.endswith(f"|{a}|{conv}")]
                if v:
                    agg[f"{a}|{conv}"] = {"rho_median": float(np.median(v)),
                                          "rho_max": float(np.max(v)), "n": len(v)}
        res["agg"] = agg
        save(res, out)
        print("  AGG " + "  ".join(f"{k}={v['rho_median']:.4f}" for k, v in agg.items()),
              flush=True)
        print("wrote", out, flush=True)

    # -------------------------------------------------------- leaderboard ----
    if "leaderboard" in args.parts:
        out = os.path.join(HERE, "s63_leaderboard.json")
        res = json.load(open(out)) if os.path.exists(out) else {}
        res.setdefault("meta", {
            "grid": "DS9 x 4 mechs x rate 0.7 x fills {zero, linear, nan}",
            "n_win": args.n_win,
            "fills": {"zero": "zero fill, declared (arm's own iface)",
                      "linear": "linear fill, declared (arm's own iface)",
                      "nan": "dual arm: the declaration endpoint (zeroed content "
                             "+ flag) -- NEVER SEEN by BLEND during training"},
            "relMSE": "per-window vs each arm's OWN clean",
            "references": "stock/CPT/W50 cells live in "
                          "s60_c2retrofit/s60_leaderboard.json"})
        for ds in DS9:
            X, st = s35.load(ds, args.n_win)
            if X is None:
                print(f"  board {ds}: no valid windows, skipped", flush=True)
                continue
            _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
            zmask = np.zeros_like(cl, bool)
            base = {}
            for a, m in arms.items():
                e = ((run_conv(m, ARM_IFACE[a], cl, zmask, "declared") - gt_c) ** 2).mean(1)
                base[a] = e
                res.setdefault("clean_mse", {}).setdefault(a, {})[ds] = {
                    "median": float(np.median(e)), "mean": float(e.mean())}
            for mech in MECHS:
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, 0.7, H,
                                                            fill="linear")
                for fill in ("zero", "linear", "nan"):
                    if fill != "nan":
                        ctx = s25.fill_context(clean, mask, fill)
                    for a, m in arms.items():
                        if fill == "nan":
                            # declaration endpoint: zeroed content + flag (dual)
                            z = s25.fill_context(clean, mask, "zero")
                            e = ((run_conv(m, ARM_IFACE[a], z, mask, "declared")
                                  - gt) ** 2).mean(1)
                        else:
                            e = ((run_conv(m, ARM_IFACE[a], ctx, mask, "declared")
                                  - gt) ** 2).mean(1)
                        ok = base[a] > 1e-12
                        r = e[ok] / base[a][ok]
                        res.setdefault("grid", {})[
                            f"{ds}|{mech}|0.7|{fill}|{a}"] = {
                            "median": float(np.median(r)), "mean": float(r.mean())}
                print(f"  board {ds:11s} {mech:13s} " + " ".join(
                    f"{a}." + "/".join(
                        f"{res['grid'][f'{ds}|{mech}|0.7|{f}|{a}']['median']:.2f}"
                        for f in ("zero", "linear", "nan"))
                    for a in arms), flush=True)
                save(res, out)
        save(res, out)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
