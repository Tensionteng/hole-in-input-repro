#!/usr/bin/env python
"""S60 analysis: closure / floor / clean-ratio from s60_grid.json + leaderboard wins.

  closure@a=1  (e_stock - e_arm) / (e_stock - 1), per cell (ds|mech|rate, declared
               view), median over cells with stock excess e_stock - 1 > 0.02
  floor@a=0    count of cells where the arm is no worse than stock at alpha=0
  clean ratio  median over the 9 datasets of arm clean MSE / stock clean MSE

Writes s60_summary.json. Every headline number in the S60 report comes from here.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("CPT", "W50")
ALPHAS = ("0.0", "0.5", "1.0")


def main():
    grid = json.load(open(os.path.join(HERE, "s60_grid.json")))
    cells = grid["grid"]
    out = {"meta": {"closure": "(e_stock - e_arm)/(e_stock - 1), median over cells "
                    "with stock excess > 0.02 (declared view, cell = ds|mech|rate)",
                    "floor": "cells where arm <= stock at alpha=0 (declared)",
                    "clean_ratio": "median over DS9 of arm/stock clean MSE median"}}

    # cell inventory
    keys = set()
    for k in cells:
        ds, mech, rate, al, arm, view = k.rsplit("|", 5)
        if arm == "stock" and view == "declared":
            keys.add((ds, mech, rate))
    keys = sorted(keys)

    for arm in ARMS:
        clo, clo_cells, floor_ct, floor_tot = {}, {}, 0, 0
        for al in ALPHAS:
            clo[al] = []
            clo_cells[al] = {}
        for ds, mech, rate in keys:
            for al in ALPHAS:
                ks = cells.get(f"{ds}|{mech}|{rate}|{al}|stock|declared")
                ka = cells.get(f"{ds}|{mech}|{rate}|{al}|{arm}|declared")
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
        # clean ratio
        cl = grid.get("clean", {})
        if arm in cl and "stock" in cl:
            ratios = [cl[arm][ds]["mse_median"] / cl["stock"][ds]["mse_median"]
                      for ds in cl["stock"] if ds in cl[arm]]
            res["clean_ratio_median"] = float(np.median(ratios))
            res["clean_ratio_per_ds"] = {ds: cl[arm][ds]["mse_median"]
                                         / cl["stock"][ds]["mse_median"]
                                         for ds in cl["stock"] if ds in cl[arm]}
        out[arm] = res
        print(f"[{arm}] closure@1.0 median={res['closure_median']['1.0']:.3f} "
              f"(n={res['closure_n_cells']['1.0']}), floor@0 "
              f"{floor_ct}/{floor_tot}, clean ratio "
              f"{res.get('clean_ratio_median', float('nan')):.4f}", flush=True)

    # leaderboard: win/loss vs stock on linear fill (the deployable default) + per-fill
    lb_path = os.path.join(HERE, "s60_leaderboard.json")
    if os.path.exists(lb_path):
        lb = json.load(open(lb_path))["grid"]
        for arm in ARMS:
            wl = {}
            for fill in ("zero", "linear", "nan"):
                wins = losses = ties = 0
                marg = []
                for k, v in lb.items():
                    ds, mech, rate, f, a = k.rsplit("|", 4)
                    if f != fill or a != arm:
                        continue
                    vs = lb.get(f"{ds}|{mech}|{rate}|{fill}|stock")
                    if vs is None:
                        continue
                    marg.append(vs["median"] - v["median"])
                    wins += int(v["median"] < vs["median"] - 1e-9)
                    losses += int(v["median"] > vs["median"] + 1e-9)
                    ties += int(abs(v["median"] - vs["median"]) <= 1e-9)
                wl[fill] = {"wins": wins, "losses": losses, "ties": ties,
                            "median_margin_relMSE": float(np.median(marg)) if marg else None}
            out[arm]["leaderboard_vs_stock"] = wl
            print(f"[{arm}] board vs stock: " + "  ".join(
                f"{f} {wl[f]['wins']}W/{wl[f]['losses']}L/{wl[f]['ties']}T"
                for f in wl), flush=True)

    json.dump(out, open(os.path.join(HERE, "s60_summary.json"), "w"), indent=1)
    print("wrote", os.path.join(HERE, "s60_summary.json"))


if __name__ == "__main__":
    main()
