#!/usr/bin/env python
"""S45-9ds: the factorial alpha-sweep grid extended to all 9 benchmarks.

The scorecard sweep (s45_eval.json / s48_eval_*.json) covers ETTh1/ETTm1/weather only
(24 cells). This fills the SAME key schema -- ds|mech|rate|alpha|arm|view, relMSE vs the
arm's own clean on the same windows -- for all 9 datasets x 4 mechs x 2 rates x 5 alphas
x 2 views, for every tiny arm whose checkpoint exists (including @seed replicates).

Windows come from s35.load (the sweep9 convention), so this file is analysed standalone;
the 3 scorecard datasets are re-computed under this loader rather than merged with the
scorecard's window set.

Usage: eval_s45_9ds.py --arms A,B,C --device cuda --out s48_eval_9ds.json
Missing checkpoints are skipped with a warning (seed replicates arrive as s48f lands).
"""
import argparse
import json
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
import eval_s45 as e45                     # noqa: E402  (sets sys.path, constants)
import run_s25_twofloor as s25             # noqa: E402
import run_s35_breadth as s35              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="comma list, @seed tags allowed")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-win", type=int, default=150)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    dev = args.device

    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("meta", {"grid": "9ds x 4mech x 2rates x 5alphas x 2views",
                            "loader": "s35.load", "n_win": args.n_win,
                            "alphas": list(e45.ALPHAS), "mechs": list(e45.MECHS),
                            "rates": list(e45.RATES), "ds9": e45.DS9})

    arms = {}
    for a in args.arms.split(","):
        try:
            arms[a] = e45.load_arm(a, dev, size="tiny")
        except FileNotFoundError as exc:
            print(f"  SKIP {a}: {exc}", flush=True)
    print(f"loaded {len(arms)} arms: {sorted(arms)}", flush=True)

    for ds in e45.DS9:
        t0 = time.time()
        X, st = s35.load(ds, args.n_win)
        if X is None:
            print(f"  {ds}: no valid windows, skipped", flush=True)
            continue
        _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, e45.H)
        obs_c = np.ones_like(cl)
        base = {a: ((e45.median_fwd(m, cl, obs_c, e45.ARM_IFACE[a.partition("@")[0]])
                     - gt_c) ** 2).mean(1) for a, m in arms.items()}
        for mech in e45.MECHS:
            for rate in e45.RATES:
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, rate, e45.H,
                                                            fill="linear")
                for al in e45.ALPHAS:
                    ctx = e45.blend(clean, lin, mask, al)
                    for a, m in arms.items():
                        for view in ("declared", "plain"):
                            iface = e45.ARM_IFACE[a.partition("@")[0]] if view == "declared" \
                                else "plain"
                            cx, ob = e45.conv_ctx(ctx, mask, view, iface)
                            e = ((e45.median_fwd(m, cx, ob, iface) - gt) ** 2).mean(1)
                            ok = base[a] > 1e-12
                            r = e[ok] / base[a][ok]
                            res.setdefault("sweep", {})[
                                f"{ds}|{mech}|{rate}|{al}|{a}|{view}"] = {
                                "median": float(np.median(r)), "mean": float(r.mean()),
                                "w": [round(float(x), 6) for x in r]}
                print(f"  {ds:11s} {mech:13s} p={rate} done "
                      f"({time.time() - t0:.0f}s)", flush=True)
                json.dump(res, open(args.out, "w"))
    print("all done", flush=True)


if __name__ == "__main__":
    main()
