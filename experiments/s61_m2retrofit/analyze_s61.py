#!/usr/bin/env python
"""S61 analysis: closure / floor / clean-ratio from s61_grid.json + leaderboard.

  closure@a     (e_stock - e_arm) / (e_stock - 1), per cell (ds|mech|rate, declared
                view), median over cells with stock excess e_stock - 1 > 0.02.
                Reference = STOCK-DECLARED (the retainer's fill-reading path).
  sweep ends    stock's (and each arm's) absolute median relMSE at alpha=0 and
                alpha=1 across cells -- stock itself moves with alpha, so closure
                must be read against its endpoints.
  floor@a=0     count of cells where the arm is no worse than stock at alpha=0
  clean ratio   median over the 9 datasets of arm clean MSE / stock clean MSE

Writes s61_summary.json. Every headline number in the S61 report comes from here.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ("CPT", "W50")
ALPHAS = ("0.0", "0.5", "1.0")


def main():
    grid = json.load(open(os.path.join(HERE, "s61_grid.json")))
    cells = grid["grid"]
    out = {"meta": {"closure": "(e_stock - e_arm)/(e_stock - 1), median over cells "
                    "with stock excess > 0.02 (declared view, cell = ds|mech|rate); "
                    "reference = stock-DECLARED",
                    "floor": "cells where arm <= stock at alpha=0 (declared)",
                    "clean_ratio": "median over DS9 of arm/stock clean MSE median"}}

    keys = set()
    for k in cells:
        ds, mech, rate, al, arm, view = k.rsplit("|", 5)
        if arm == "stock" and view == "declared":
            keys.add((ds, mech, rate))
    keys = sorted(keys)

    # absolute sweep endpoints (median relMSE over cells) for every arm
    ends = {}
    for arm in ("stock",) + ARMS:
        for view in ("declared", "plain"):
            e0 = [cells[f"{ds}|{mech}|{rate}|0.0|{arm}|{view}"]["median"]
                  for ds, mech, rate in keys
                  if f"{ds}|{mech}|{rate}|0.0|{arm}|{view}" in cells]
            e1 = [cells[f"{ds}|{mech}|{rate}|1.0|{arm}|{view}"]["median"]
                  for ds, mech, rate in keys
                  if f"{ds}|{mech}|{rate}|1.0|{arm}|{view}" in cells]
            ends[f"{arm}|{view}"] = {
                "alpha0_median": float(np.median(e0)) if e0 else None,
                "alpha1_median": float(np.median(e1)) if e1 else None,
                "alpha0_mean": float(np.mean(e0)) if e0 else None,
                "alpha1_mean": float(np.mean(e1)) if e1 else None,
                "n_cells": len(e0)}
    out["sweep_endpoints"] = ends
    # per-rate endpoints (declared): the paper's moirai2 reference is the
    # scattered (mcar) dropout at rate 0.7 -- report it separately
    per_rate = {}
    for rate in ("0.3", "0.7"):
        for arm in ("stock",) + ARMS:
            sel = [(ds, mech, r) for ds, mech, r in keys if r == rate]
            e0 = [cells[f"{ds}|{mech}|{r}|0.0|{arm}|declared"]["median"]
                  for ds, mech, r in sel
                  if f"{ds}|{mech}|{r}|0.0|{arm}|declared" in cells]
            e1 = [cells[f"{ds}|{mech}|{r}|1.0|{arm}|declared"]["median"]
                  for ds, mech, r in sel
                  if f"{ds}|{mech}|{r}|1.0|{arm}|declared" in cells]
            e0m = [cells[f"{ds}|mcar|{r}|0.0|{arm}|declared"]["median"]
                   for ds, mech, r in sel
                   if f"{ds}|mcar|{r}|0.0|{arm}|declared" in cells]
            e1m = [cells[f"{ds}|mcar|{r}|1.0|{arm}|declared"]["median"]
                   for ds, mech, r in sel
                   if f"{ds}|mcar|{r}|1.0|{arm}|declared" in cells]
            per_rate[f"{arm}|{rate}"] = {
                "alpha0_median": float(np.median(e0)) if e0 else None,
                "alpha1_median": float(np.median(e1)) if e1 else None,
                "mcar_alpha0_median": float(np.median(e0m)) if e0m else None,
                "mcar_alpha1_median": float(np.median(e1m)) if e1m else None}
    out["sweep_endpoints_by_rate"] = per_rate
    print("[sweep endpoints, declared] " + "  ".join(
        f"{a}: {ends[f'{a}|declared']['alpha0_median']:.3f} -> "
        f"{ends[f'{a}|declared']['alpha1_median']:.3f}" for a in ("stock",) + ARMS),
        flush=True)

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

    # leaderboard: per-fill medians + win/loss vs stock
    lb_path = os.path.join(HERE, "s61_leaderboard.json")
    if os.path.exists(lb_path):
        lb = json.load(open(lb_path))["grid"]
        out["leaderboard_medians"] = {}
        for arm in ("stock",) + ARMS:
            for fill in ("zero", "linear", "nan"):
                v = [c["median"] for k, c in lb.items() if k.endswith(f"|{fill}|{arm}")]
                if v:
                    out["leaderboard_medians"][f"{arm}|{fill}"] = {
                        "median_relMSE": float(np.median(v)), "n": len(v)}
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

    json.dump(out, open(os.path.join(HERE, "s61_summary.json"), "w"), indent=1)
    print("wrote", os.path.join(HERE, "s61_summary.json"))


if __name__ == "__main__":
    main()
