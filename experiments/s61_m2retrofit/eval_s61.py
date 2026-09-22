#!/usr/bin/env python
"""S61 eval: stock Moirai 2.0 vs CPT vs m2-retrofit (W50) on the paper's grid.

All three arms share Moirai 2.0's native retaining interface -- fill content AND
flag both enter the content path; there is no interface switch, only conventions:
  declared   content (nan_to_num at missing) + observed_mask = ~miss  (s46 "nan")
  plain      content + observed_mask all ones                         (s46 "plain")

Parts (each writes its own JSON, merged across runs so arms can be added later):
  clean + grid   9 benchmarks x 4 mechanisms x rates {0.3, 0.7} x alphas {0, .5, 1},
                 declared and plain views, own-clean relMSE with per-window ratios
  probe          permutation rho (s46 protocol: 3 ds x 3 mechs, rate 0.3, 60 win,
                 linear fill, 4 perms), declared and plain views
  leaderboard    9 benchmarks x 4 mechanisms at 70% missing under three deployable
                 fills: zero (declared), linear (declared), nan-declared (the
                 declaration endpoint: content zeroed + flag = 0 -- numerically
                 identical to zero-declared for this interface, reported as such)

All eval strict fp32 (no TF32), seeds/windows identical across arms.
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

torch.backends.cuda.matmul.allow_tf32 = False     # strict fp32 eval
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s35_breadth"))
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import run_s35_breadth as s35
import m2_iface

L, H = s25.L, 64
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES = (0.3, 0.7)
ALPHAS = (0.0, 0.5, 1.0)
DS9 = ("ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness")
CK = os.path.join(HERE, "s61_ckpt")
ARM_FILE = {"CPT": "arm_CPT.pt", "W50": "arm_W50.pt"}
BATCH = 1024


def load_arm(arm, dev):
    fc = m2_iface.load_stock(dev)
    if arm != "stock":
        sd = torch.load(os.path.join(CK, ARM_FILE[arm]), map_location=dev,
                        weights_only=True)
        fc.module.load_state_dict(sd)
        fc.eval()
    return fc


def run_conv(m, ctx, miss, conv, batch=BATCH):
    content, obs = m2_iface.conv_ctx(ctx, miss, conv)
    return m2_iface.median_fwd(m, content, obs, horizon=H, batch=batch)


def blend(clean, lin, mask, a):
    out = lin.copy()
    out[mask] = (1.0 - a) * lin[mask] + a * clean[mask]
    return out


def perm_probe(m, filled, mask, conv, rng, n_perm=4):
    """S30/S36/S46 permutation probe."""
    y0 = run_conv(m, filled, mask, conv)
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
        dp.append(np.sqrt(((run_conv(m, cp, mask, conv) - y0) ** 2).mean(1)) / scale)
        dr.append(np.sqrt(((run_conv(m, cr, mask, conv) - y0) ** 2).mean(1)) / scale)
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
    ap.add_argument("--arms", default="stock,CPT,W50")
    ap.add_argument("--n-win", type=int, default=150)
    ap.add_argument("--device", default="cuda:2")
    # --suffix added for seed replications (2026-09-01): default '' reproduces
    # the original seed-20260901 behavior exactly.
    ap.add_argument("--suffix", default="",
                    help="seed-replication suffix (e.g. _s20260902): load "
                         "arm_{CPT,W50}{suffix}.pt and write "
                         "s61_{grid,probe,leaderboard}{suffix}.json; default "
                         "'' = the seed-20260901 artifacts")
    args = ap.parse_args()
    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    dev = args.device
    if args.suffix:
        ARM_FILE.update({"CPT": f"arm_CPT{args.suffix}.pt",
                         "W50": f"arm_W50{args.suffix}.pt"})
    arms = {a: load_arm(a, dev) for a in args.arms.split(",")}
    print("loaded arms:", list(arms), flush=True)

    # ------------------------------------------------------- clean + grid ----
    if "clean" in args.parts or "grid" in args.parts:
        out = os.path.join(HERE, f"s61_grid{args.suffix}.json")
        res = json.load(open(out)) if os.path.exists(out) else {}
        res.setdefault("meta", {"grid": "DS9 x 4 mechs x rates {0.3,0.7} x alphas "
                                "{0,0.5,1} x {declared,plain}", "n_win": args.n_win,
                                "relMSE": "per-window vs each arm's OWN clean",
                                "arms": {a: "native (retainer)" for a in arms},
                                "H": H, "dtype": "fp32 strict"})
        for ds in DS9:
            X, st = s35.load(ds, args.n_win)
            if X is None:
                print(f"  grid {ds}: no valid windows, skipped", flush=True)
                continue
            _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
            zmask = np.zeros_like(cl, bool)
            base = {}
            for a, m in arms.items():
                e = ((run_conv(m, cl, zmask, "declared") - gt_c) ** 2).mean(1)
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
                                e = ((run_conv(m, ctx, mask, view)
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
        out = os.path.join(HERE, f"s61_probe{args.suffix}.json")
        res = json.load(open(out)) if os.path.exists(out) else {}
        res.setdefault("meta", {"protocol": "s46: 3 ds x 3 mechs, rate 0.3, "
                                "60 win, linear fill, 4 perms",
                                "convs": {"declared": "content + flag (s46 'nan')",
                                          "plain": "content only (s46 'plain')"},
                                "stock_ref": {"declared": 0.9492, "plain": 0.9672}})
        for ds, path in s25.DATASETS.items():
            X, st = s25.load_windows(path, 60, s25.SEED, H)
            for mech in ("mcar", "block", "mnar_high"):
                _, lin, mask, _ = s25.build_eval_batch(X, st, mech, 0.3, H,
                                                       fill="linear")
                for a, m in arms.items():
                    for conv in ("declared", "plain"):
                        rng = np.random.default_rng(s25.SEED)
                        r = perm_probe(m, lin, mask, conv, rng)
                        res.setdefault("probe", {})[f"{ds}|{mech}|{a}|{conv}"] = r
                print(f"  probe {ds:8s} {mech:10s} " + " ".join(
                    f"{a}.{c}:{res['probe'][f'{ds}|{mech}|{a}|{c}']['rho']:.3f}"
                    for a in arms for c in ("declared", "plain")), flush=True)
                save(res, out)
        agg = {}
        for a in arms:
            for conv in ("declared", "plain"):
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
        out = os.path.join(HERE, f"s61_leaderboard{args.suffix}.json")
        res = json.load(open(out)) if os.path.exists(out) else {}
        res.setdefault("meta", {
            "grid": "DS9 x 4 mechs x rate 0.7 x fills {zero, linear, nan}",
            "n_win": args.n_win,
            "fills": {"zero": "zero fill, declared (content zeroed + flag)",
                      "linear": "linear fill, declared",
                      "nan": "declaration endpoint: content zeroed + flag = 0 "
                             "(numerically identical to zero-declared here; "
                             "Moirai 2.0 has no content-free NaN path)"},
            "relMSE": "per-window vs each arm's OWN clean"})
        for ds in DS9:
            X, st = s35.load(ds, args.n_win)
            if X is None:
                print(f"  board {ds}: no valid windows, skipped", flush=True)
                continue
            _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
            zmask = np.zeros_like(cl, bool)
            base = {}
            for a, m in arms.items():
                e = ((run_conv(m, cl, zmask, "declared") - gt_c) ** 2).mean(1)
                base[a] = e
                res.setdefault("clean_mse", {}).setdefault(a, {})[ds] = {
                    "median": float(np.median(e)), "mean": float(e.mean())}
            for mech in MECHS:
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, 0.7, H,
                                                            fill="linear")
                for fill in ("zero", "linear", "nan"):
                    # nan -> NaN at missing -> nan_to_num zeroes it in conv_ctx
                    ctx = s25.fill_context(clean, mask, fill)
                    for a, m in arms.items():
                        e = ((run_conv(m, ctx, mask, "declared") - gt) ** 2).mean(1)
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
