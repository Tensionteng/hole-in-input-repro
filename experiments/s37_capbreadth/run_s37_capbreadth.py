#!/usr/bin/env python
"""S37: does the S27 interface fix transfer to datasets its adapter never saw?

The input-patch-embedding adapters of S27 were trained on ETTh1/ETTm1/weather. We freeze them
and run the identical imputation-quality sweep on all nine datasets of S35, six of which the
adapters have never seen. Nothing is retrained. See s37_notes.md for the pre-registration.
"""
import argparse, json, os, sys

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
import run_s25_twofloor as s25
import run_s27_interface as s27
import run_s35_breadth as s35

H, L = s27.H, s25.L
CKPT = os.path.join(EXP, "s27_interface", "s27_ckpt")
OUT = os.path.join(HERE, "s37_results.json")
IFACES = ("plain", "native", "dual", "dual_obsnorm")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES = (0.3, 0.7)
INSAMPLE = ("ETTh1", "ETTm1", "weather")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(s35.DATASETS))
    ap.add_argument("--n-win", type=int, default=60)
    ap.add_argument("--out", default=OUT)
    # the degenerate-channel filter is needed for electricity/traffic/illness; disable it
    # to reproduce S27 exactly, which kept every channel of the three original datasets.
    ap.add_argument("--no-filter", action="store_true")
    args = ap.parse_args()
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)

    bolt = s25.Bolt("cuda")
    base_sd = s27.save_proj(bolt)
    adapters = {i: torch.load(os.path.join(CKPT, f"proj_{i}.pt"), map_location="cuda")
                for i in IFACES}
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("meta", {"H": H, "L": L, "alphas": list(ALPHAS), "mechs": list(MECHS),
                            "rates": list(RATES), "n_win": args.n_win,
                            "ifaces": list(IFACES), "insample": list(INSAMPLE),
                            "adapters": "S27, frozen, not retrained",
                            "metric": "paired per-window relMSE vs own clean, median"})

    for name in args.datasets.split(","):
        X, st = s35.load(name, args.n_win)
        if X is None:
            print(f"SKIP {name}", flush=True)
            continue
        _, cl, _, gt_c = s35.build(X, st, "clean", 0.0)
        keep = s35.finite(cl) & s35.finite(gt_c) & (cl.std(axis=1) > 1e-8)
        if args.no_filter:
            keep = np.ones(len(cl), bool)
        cl, gt_c = cl[keep], gt_c[keep]
        s27.load_proj(bolt, base_sd)
        base = ((bolt.median_np(cl) - gt_c) ** 2).mean(1)
        ok = base > 1e-12
        # Own-clean baselines. The adaptation edits a module the model uses on every input,
        # so each interface must be normalised by ITS OWN clean forecast; otherwise a generic
        # gain (or loss) on clean data is charged to the interface. See app:method.
        obs_clean = np.ones_like(cl)
        own = {}
        for iface in IFACES:
            s27.load_proj(bolt, adapters[iface])
            own[iface] = ((s27.median_iface_np(bolt, cl, obs_clean, iface) - gt_c) ** 2).mean(1)
        s27.load_proj(bolt, base_sd)
        res.setdefault("clean_own", {})[name] = {
            i: float(np.median(own[i][ok] / base[ok])) for i in IFACES}
        tag = "in-sample" if name in INSAMPLE else "HELD OUT"
        print(f"\n== {name} ({tag}, {int(ok.sum())} series) ==", flush=True)
        for mech in MECHS:
            for rate in RATES:
                clean, lin, mask, gt = s35.build(X, st, mech, rate)
                clean, lin, mask, gt = clean[keep], lin[keep], mask[keep], gt[keep]
                obs = (~mask).astype(np.float32)
                line = {}
                for iface in IFACES:
                    s27.load_proj(bolt, adapters[iface])
                    vals = []
                    for a in ALPHAS:
                        ctx = s27.blend_fill(clean, lin, mask, a)
                        p = s27.median_iface_np(bolt, ctx, obs, iface)
                        e = ((p - gt) ** 2).mean(1)
                        r = e[ok] / base[ok]
                        ro = e[ok] / np.maximum(own[iface][ok], 1e-12)
                        vals.append({"median": float(np.median(r)), "mean": float(r.mean()),
                                     "q95": float(np.quantile(r, 0.95)),
                                     "median_own": float(np.median(ro)),
                                     "mean_own": float(ro.mean())})
                        res.setdefault("sweep", {})[
                            f"{name}|{mech}|{rate}|{a}|{iface}"] = vals[-1]
                    line[iface] = [v["median_own"] for v in vals]
                    res.setdefault("released_norm", {})[f"{name}|{mech}|{rate}|{iface}"] = \
                        [v["median"] for v in vals]
                nat1 = line["native"][-1]
                best = min(line["dual"][-1], line["dual_obsnorm"][-1])
                clo = (nat1 - best) / (nat1 - 1.0) if nat1 > 1.0 else float("nan")
                res.setdefault("closure", {})[f"{name}|{mech}|{rate}"] = {
                    "norm": "own-clean per interface",
                    "native_a1": nat1, "restored_a1": best, "closure": clo,
                    "native_a0": line["native"][0],
                    "restored_a0": min(line["dual"][0], line["dual_obsnorm"][0])}
                print(f"  {mech:13s} p={rate}  native {line['native'][0]:.3f}->"
                      f"{nat1:.3f}   restored {min(line['dual'][0],line['dual_obsnorm'][0]):.3f}"
                      f"->{best:.3f}   closure {clo*100:5.1f}%", flush=True)
        s27.load_proj(bolt, base_sd)
        json.dump(res, open(args.out + ".tmp", "w"))
        os.replace(args.out + ".tmp", args.out)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
