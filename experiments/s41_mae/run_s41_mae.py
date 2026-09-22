#!/usr/bin/env python
"""S41: every load-bearing measurement under MAE as well as MSE, in one pass.

Both metrics are computed from the same forecasts on the same windows, and each is normalised
by that window's own clean-context error under the same metric, so the comparison isolates the
metric and nothing else. See s41_notes.md.
"""
import argparse, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch

for d in ("s25_twofloor", "s27_interface", "s30_crossmodel", "s35_breadth"):
    sys.path.insert(0, os.path.join(EXP, d))
import run_s25_twofloor as s25
import run_s27_interface as s27
import run_s30_crossmodel as s30
import run_s35_breadth as s35

H, L = s27.H, s25.L
CKPT = os.path.join(EXP, "s27_interface", "s27_ckpt")
IFACES = ("native", "dual", "dual_obsnorm")
ALPHAS = (0.0, 0.5, 1.0)
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES = (0.3, 0.7)
INSAMPLE = ("ETTh1", "ETTm1", "weather")
OUT = os.path.join(HERE, "s41_results.json")


def errs(pred, truth):
    """Per-window squared and absolute error, from the same forecast."""
    d = pred - truth
    return (d ** 2).mean(1), np.abs(d).mean(1)


def ratio(e, base, ok):
    r = e[ok] / np.maximum(base[ok], 1e-12)
    return {"median": float(np.median(r)), "mean": float(r.mean()),
            "q95": float(np.quantile(r, 0.95))}


def l1_probe(model, ctx, miss, conv, rng, n_perm=4):
    """The permutation probe with an L1 output-change norm instead of RMS."""
    y0 = model.fc(ctx, miss, conv)
    scale = np.abs(y0).mean(1) + 1e-9
    dp, dr = [], []
    for _ in range(n_perm):
        cp, cr = ctx.copy(), ctx.copy()
        for i in range(len(ctx)):
            idx = np.flatnonzero(miss[i])
            if len(idx) < 2:
                continue
            cp[i, idx] = ctx[i, idx][rng.permutation(len(idx))]
            o = ctx[i, ~miss[i]]
            cr[i, idx] = rng.choice(o, size=len(idx)) if len(o) else 0.0
        dp.append(np.abs(model.fc(cp, miss, conv) - y0).mean(1) / scale)
        dr.append(np.abs(model.fc(cr, miss, conv) - y0).mean(1) / scale)
    dp, dr = np.mean(dp, 0), np.mean(dr, 0)
    return float(dp.mean() / max(dr.mean(), 1e-15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(s35.DATASETS))
    ap.add_argument("--n-win", type=int, default=120)
    ap.add_argument("--parts", default="grid,sweep,probe")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)

    bolt = s25.Bolt("cuda")
    base_sd = s27.save_proj(bolt)
    adapters = {i: torch.load(os.path.join(CKPT, f"proj_{i}.pt"), map_location="cuda")
                for i in IFACES}
    probe_model = s30.Bolt("cuda") if "probe" in args.parts else None
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("meta", {"L": L, "H": H, "n_win": args.n_win, "alphas": list(ALPHAS),
                            "mechs": list(MECHS), "rates": list(RATES),
                            "metrics": ["mse", "mae"], "insample": list(INSAMPLE),
                            "note": "both metrics from the same forecasts, each normalised "
                                    "by the same window's own clean error under that metric"})
    parts = args.parts.split(",")

    for name in args.datasets.split(","):
        X, st = s35.load(name, args.n_win)
        if X is None:
            print(f"SKIP {name}", flush=True)
            continue
        _, cl, _, gt_c = s35.build(X, st, "clean", 0.0)
        keep = s35.finite(cl) & s35.finite(gt_c) & (cl.std(axis=1) > 1e-8)
        cl, gt_c = cl[keep], gt_c[keep]
        s27.load_proj(bolt, base_sd)
        b2, b1 = errs(bolt.median_np(cl), gt_c)
        ok = (b2 > 1e-12) & (b1 > 1e-9)
        tag = "in-sample" if name in INSAMPLE else "HELD OUT"
        print(f"\n== {name} ({tag}, {int(ok.sum())} series) ==", flush=True)

        if "grid" in parts:
            print(f"  {'mechanism':14s} {'rate':>5s} {'relMSE':>9s} {'relMAE':>9s} "
                  f"{'MSE excess':>11s} {'MAE excess':>11s}", flush=True)
            for mech in MECHS:
                for rate in RATES:
                    _, fl, _, gt = s35.build(X, st, mech, rate)
                    e2, e1 = errs(bolt.median_np(fl[keep]), gt[keep])
                    r2, r1 = ratio(e2, b2, ok), ratio(e1, b1, ok)
                    res.setdefault("grid", {})[f"{name}|{mech}|{rate}"] = {
                        "mse": r2, "mae": r1}
                    print(f"  {mech:14s} {rate:5} {r2['median']:9.3f} {r1['median']:9.3f} "
                          f"{r2['median']-1:11.3f} {r1['median']-1:11.3f}", flush=True)

        if "sweep" in parts:
            own2, own1 = {}, {}
            obs_clean = np.ones_like(cl)
            for iface in IFACES:
                s27.load_proj(bolt, adapters[iface])
                own2[iface], own1[iface] = errs(
                    s27.median_iface_np(bolt, cl, obs_clean, iface), gt_c)
            s27.load_proj(bolt, base_sd)
            for mech in MECHS:
                for rate in RATES:
                    clean, lin, mask, gt = s35.build(X, st, mech, rate)
                    clean, lin = clean[keep], lin[keep]
                    mask, gt = mask[keep], gt[keep]
                    obs = (~mask).astype(np.float32)
                    line = {}
                    for iface in IFACES:
                        s27.load_proj(bolt, adapters[iface])
                        for a in ALPHAS:
                            ctx = s27.blend_fill(clean, lin, mask, a)
                            e2, e1 = errs(s27.median_iface_np(bolt, ctx, obs, iface), gt)
                            cell = {"mse": ratio(e2, own2[iface], ok),
                                    "mae": ratio(e1, own1[iface], ok)}
                            res.setdefault("sweep", {})[
                                f"{name}|{mech}|{rate}|{a}|{iface}"] = cell
                            line[(iface, a)] = cell
                    out = {}
                    for met in ("mse", "mae"):
                        n1 = line[("native", 1.0)][met]["median"]
                        b = min(line[("dual", 1.0)][met]["median"],
                                line[("dual_obsnorm", 1.0)][met]["median"])
                        n0 = line[("native", 0.0)][met]["median"]
                        b0 = min(line[("dual", 0.0)][met]["median"],
                                 line[("dual_obsnorm", 0.0)][met]["median"])
                        out[met] = {"native_a1": n1, "restored_a1": b,
                                    "native_a0": n0, "restored_a0": b0,
                                    "closure": ((n1 - b) / (n1 - 1.0)
                                                if n1 > 1.0 else float("nan"))}
                    res.setdefault("closure", {})[f"{name}|{mech}|{rate}"] = out
                    print(f"  {mech:14s} {rate:5}  closure MSE "
                          f"{out['mse']['closure']*100:6.1f}%   MAE "
                          f"{out['mae']['closure']*100:6.1f}%   "
                          f"(native {out['mse']['native_a1']:.3f}/"
                          f"{out['mae']['native_a1']:.3f})", flush=True)
            s27.load_proj(bolt, base_sd)

        if "probe" in parts:
            _, lin, mask, _ = s35.build(X, st, "block", 0.3)
            lin, mask = lin[keep], mask[keep]
            row = {}
            for conv in ("plain", "mask", "nan"):
                rng = np.random.default_rng(s25.SEED)
                rms = s30.perm_probe(probe_model, lin, mask, conv,
                                     np.random.default_rng(s25.SEED))["ratio"]
                row[conv] = {"rms": rms,
                             "l1": l1_probe(probe_model, lin, mask, conv, rng)}
            res.setdefault("probe", {})[name] = row
            print("  probe rho  " + "   ".join(
                f"{c}: rms={row[c]['rms']:.4f} l1={row[c]['l1']:.4f}" for c in row),
                flush=True)

        json.dump(res, open(args.out + ".tmp", "w"))
        os.replace(args.out + ".tmp", args.out)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
