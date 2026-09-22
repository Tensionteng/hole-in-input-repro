#!/usr/bin/env python
"""S63 analysis: blend-only ablation vs stock / full-recipe CPT / W50.

Conventions identical to analyze_s60.py:
  closure@a=1  (e_stock - e_arm) / (e_stock - 1), per cell (ds|mech|rate, declared
               view), median over cells with stock excess e_stock - 1 > 0.02
  floor@a=0    count of cells where the arm is no worse than stock at alpha=0
  clean ratio  median over the 9 datasets of arm clean MSE / stock clean MSE

Reference cells (stock, CPT, W50) are read from s60_c2retrofit's JSONs -- s60's
eval is deterministic (fixed seeds/windows, strict fp32), and eval_s63.py drives
the identical loaders, so cells are directly comparable.

Headline question: does the blend-only arm collapse on ZERO fill the way
Chronos-Bolt's P5 did (3-19x worse than stock on every dataset)?

Writes s63_summary.json.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S60 = os.path.join(HERE, "..", "s60_c2retrofit")
ARM = "BLEND"
REFS = ("stock", "CPT", "W50")
ALPHAS = ("0.0", "0.5", "1.0")
FILLS = ("zero", "linear", "nan")


def main():
    g63 = json.load(open(os.path.join(HERE, "s63_grid.json")))
    g60 = json.load(open(os.path.join(S60, "s60_grid.json")))
    cells = dict(g60["grid"])          # stock/CPT/W50
    cells.update(g63["grid"])          # BLEND
    clean = {a: d for a, d in g60.get("clean", {}).items()}
    clean.update(g63.get("clean", {}))
    out = {"meta": {"arm": "BLEND = S60 CPT with declaration-endpoint share 0 "
                    "(mechdiv_filldiv; all holed windows alpha-blend filled)",
                    "closure": "(e_stock - e_arm)/(e_stock - 1), median over cells "
                    "with stock excess > 0.02 (declared view, cell = ds|mech|rate)",
                    "floor": "cells where arm <= stock at alpha=0 (declared)",
                    "clean_ratio": "median over DS9 of arm/stock clean MSE median",
                    "references": "stock/CPT/W50 from s60_c2retrofit JSONs "
                    "(deterministic eval, same seeds/windows)"}}

    keys = set()
    for k in cells:
        ds, mech, rate, al, arm, view = k.rsplit("|", 5)
        if arm == "stock" and view == "declared":
            keys.add((ds, mech, rate))
    keys = sorted(keys)

    clo, clo_cells, floor_ct, floor_tot = {}, {}, 0, 0
    for al in ALPHAS:
        clo[al] = []
        clo_cells[al] = {}
    for ds, mech, rate in keys:
        for al in ALPHAS:
            ks = cells.get(f"{ds}|{mech}|{rate}|{al}|stock|declared")
            ka = cells.get(f"{ds}|{mech}|{rate}|{al}|{ARM}|declared")
            if ks is None or ka is None:
                continue
            e_s, e_a = ks["median"], ka["median"]
            if al == "0.0":
                floor_tot += 1
                floor_ct += int(e_a <= e_s)
            if e_s - 1 > 0.02:
                c = (e_s - e_a) / (e_s - 1)
                clo[al].append(c)
                clo_cells[al][f"{ds}|{mech}|{rate}"] = {
                    "e_stock": e_s, "e_arm": e_a, "closure": c}
    res = {
        "closure_median": {al: float(np.median(v)) if v else None
                           for al, v in clo.items()},
        "closure_mean": {al: float(np.mean(v)) if v else None
                         for al, v in clo.items()},
        "closure_n_cells": {al: len(v) for al, v in clo.items()},
        "floor_at_alpha0": {"count": floor_ct, "total": floor_tot},
        "closure_cells": clo_cells,
    }
    if ARM in clean and "stock" in clean:
        ratios = [clean[ARM][ds]["mse_median"] / clean["stock"][ds]["mse_median"]
                  for ds in clean["stock"] if ds in clean[ARM]]
        res["clean_ratio_median"] = float(np.median(ratios))
        res["clean_ratio_per_ds"] = {ds: clean[ARM][ds]["mse_median"]
                                     / clean["stock"][ds]["mse_median"]
                                     for ds in clean["stock"] if ds in clean[ARM]}
    out[ARM] = res
    print(f"[{ARM}] closure@1.0 median={res['closure_median']['1.0']:.3f} "
          f"(n={res['closure_n_cells']['1.0']}), floor@0 "
          f"{floor_ct}/{floor_tot}, clean ratio "
          f"{res.get('clean_ratio_median', float('nan')):.4f}", flush=True)

    # ------------------------------------------------------ leaderboard ----
    lb60 = json.load(open(os.path.join(S60, "s60_leaderboard.json")))["grid"]
    lb63 = json.load(open(os.path.join(HERE, "s63_leaderboard.json")))["grid"]
    lb = dict(lb60)
    lb.update(lb63)

    board = {}
    for fill in FILLS:
        per_arm = {}
        for a in (ARM,) + REFS:
            vals = {k.rsplit("|", 1)[0]: v["median"] for k, v in lb.items()
                    if k.endswith(f"|{fill}|{a}")}
            if vals:
                per_arm[a] = {"median_of_cells": float(np.median(list(vals.values()))),
                              "max": float(max(vals.values())), "n": len(vals)}
        # wins/losses of BLEND vs each reference
        vs = {}
        for ref in REFS:
            wins = losses = ties = 0
            marg = []
            for k, v in lb.items():
                ds, mech, rate, f, a = k.rsplit("|", 4)
                if f != fill or a != ARM:
                    continue
                vr = lb.get(f"{ds}|{mech}|{rate}|{fill}|{ref}")
                if vr is None:
                    continue
                marg.append(vr["median"] - v["median"])
                wins += int(v["median"] < vr["median"] - 1e-9)
                losses += int(v["median"] > vr["median"] + 1e-9)
                ties += int(abs(v["median"] - vr["median"]) <= 1e-9)
            vs[ref] = {"wins": wins, "losses": losses, "ties": ties,
                       "median_margin_relMSE": float(np.median(marg)) if marg else None}
        board[fill] = {"per_arm": per_arm, "BLEND_vs": vs}
    out["leaderboard"] = board
    for fill in FILLS:
        pa = board[fill]["per_arm"]
        print(f"[board {fill:6s}] " + "  ".join(
            f"{a}={pa[a]['median_of_cells']:.3f}" for a in pa) + "   BLEND vs: " +
            "  ".join(f"{r} {board[fill]['BLEND_vs'][r]['wins']}W/"
                      f"{board[fill]['BLEND_vs'][r]['losses']}L" for r in REFS),
              flush=True)

    # per-dataset zero-fill collapse check (the Bolt P5 pattern: 3-19x per ds)
    per_ds = {}
    for a in (ARM,) + REFS:
        for k, v in lb.items():
            ds, mech, rate, f, arm = k.rsplit("|", 4)
            if f == "zero" and arm == a:
                per_ds.setdefault(ds, {})[a] = per_ds.setdefault(ds, {}).get(a, []) + [v["median"]]
    zds = {}
    for ds, arms in sorted(per_ds.items()):
        row = {a: float(np.median(v)) for a, v in arms.items()}
        if ARM in row and "stock" in row:
            row["BLEND/stock"] = row[ARM] / row["stock"]
        if ARM in row and "CPT" in row:
            row["BLEND/CPT"] = row[ARM] / row["CPT"]
        zds[ds] = row
    out["leaderboard"]["zero_per_ds"] = zds
    print("[zero per ds] " + "  ".join(
        f"{ds}:{r.get('BLEND/stock', float('nan')):.2f}x" for ds, r in zds.items()),
        flush=True)

    # ------------------------------------------------------------ probe ----
    pr63 = json.load(open(os.path.join(HERE, "s63_probe.json")))
    pr60 = json.load(open(os.path.join(S60, "s60_probe.json")))
    out["probe"] = {"BLEND": pr63["agg"], "references": pr60["agg"]}
    for conv in ("declared", "plain"):
        print(f"[probe {conv:8s}] BLEND rho="
              f"{pr63['agg'][f'BLEND|{conv}']['rho_median']:.3f}  (CPT "
              f"{pr60['agg'][f'CPT|{conv}']['rho_median']:.3f}, W50 "
              f"{pr60['agg'][f'W50|{conv}']['rho_median']:.3f}, stock "
              f"{pr60['agg'][f'stock|{conv}']['rho_median']:.3f})", flush=True)

    out["verdict"] = (
        "The declaration endpoint is necessary on Chronos-2 too. The blend-only "
        "arm (mechdiv_filldiv, endpoint share 0) collapses on zero fill: "
        "leaderboard median relMSE 20.13 vs stock 2.03 (9.9x) and full-recipe "
        "CPT 1.50 (13.4x), losing 36/36 cells to every reference; per dataset "
        "it is 2.7x (illness) to 332x (exchange) worse than stock -- the Bolt "
        "P5 pattern (3-19x) reproduced at C2 scale. Worst cell "
        "exchange|mcar|0.7: 3327.8 vs stock 3.59. Electricity censoring cells: "
        "mcar 202.6 vs stock 3.21, mnar_high 178.6 vs 31.0, mnar_extreme "
        "163.0 vs 8.76. The collapse is specific to the never-trained "
        "content-zeroed+flag endpoint: on linear fill BLEND is healthy "
        "(median 1.243, 32W/4L vs stock, second only to CPT 1.193), "
        "closure@a=1 0.846 (CPT 0.795), floor@a=0 65/72 (CPT 66/72), clean "
        "ratio 1.020, probe rho declared 0.864 (CPT 0.815).")

    json.dump(out, open(os.path.join(HERE, "s63_summary.json"), "w"), indent=1)
    print("wrote", os.path.join(HERE, "s63_summary.json"))


if __name__ == "__main__":
    main()
