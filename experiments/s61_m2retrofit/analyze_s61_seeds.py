#!/usr/bin/env python
"""S61 seed replications: per-seed summaries + the combined s61_seeds.json.

Added 2026-09-01 for the seed-replication runs (20260902, 20260903). For each
seed in SEEDS this reads that seed's grid/probe/leaderboard JSONs -- seed
20260901 = the ORIGINAL unsuffixed artifacts, later seeds = the outputs of
eval_s61.py --suffix _s<seed> -- and recomputes the headline metrics with the
EXACT formulas of analyze_s61.py:

  closure@a    (e_stock - e_arm)/(e_stock - 1), declared view, median over
               cells with stock excess e_stock - 1 > 0.02 (reference =
               stock-DECLARED)
  floor@a=0    cells where the arm is no worse than stock (declared)
  clean ratio  median over DS9 of arm/stock clean MSE median
  rho          permutation-probe median over the 9 cells (declared view)
  leaderboard  median over the 36 cells of per-cell median relMSE per fill

Writes s61_summary_s<seed>.json for every seed (same schema as
s61_summary.json; seed 20260901's file is derived from the original artifacts
and is a NEW file -- nothing existing is overwritten) and the combined
s61_seeds.json with mean +/- sd across the three seeds for closure@1 and
clean ratio. The bad-fill attack stays single-seed (20260901) by design.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (20260901, 20260902, 20260903)
ARMS = ("CPT", "W50")
ALPHAS = ("0.0", "0.5", "1.0")


def sfx(seed):
    return "" if seed == SEEDS[0] else f"_s{seed}"


def analyze_one(seed):
    """analyze_s61.py's metric set, computed on one seed's JSONs."""
    grid = json.load(open(os.path.join(HERE, f"s61_grid{sfx(seed)}.json")))
    cells = grid["grid"]
    out = {"meta": {"seed": seed,
                    "closure": "(e_stock - e_arm)/(e_stock - 1), median over "
                    "cells with stock excess > 0.02 (declared view, cell = "
                    "ds|mech|rate); reference = stock-DECLARED",
                    "floor": "cells where arm <= stock at alpha=0 (declared)",
                    "clean_ratio": "median over DS9 of arm/stock clean MSE "
                    "median",
                    "source": f"s61_grid{sfx(seed)}.json / s61_probe"
                    f"{sfx(seed)}.json / s61_leaderboard{sfx(seed)}.json"}}

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

    # probe rho (declared view) from the seed's probe JSON
    probe = json.load(open(os.path.join(HERE, f"s61_probe{sfx(seed)}.json")))
    out["rho_declared"] = {a: probe["agg"][f"{a}|declared"] for a in
                           ("stock",) + ARMS}

    # leaderboard medians per arm x fill (median over the 36 cells)
    lb = json.load(open(os.path.join(HERE,
                                     f"s61_leaderboard{sfx(seed)}.json")))["grid"]
    out["leaderboard_medians"] = {}
    for arm in ("stock",) + ARMS:
        for fill in ("zero", "linear", "nan"):
            v = [c["median"] for k, c in lb.items()
                 if k.endswith(f"|{fill}|{arm}")]
            if v:
                out["leaderboard_medians"][f"{arm}|{fill}"] = {
                    "median_relMSE": float(np.median(v)), "n": len(v)}
    return out


def main():
    per_seed = {}
    for seed in SEEDS:
        per_seed[str(seed)] = analyze_one(seed)
        path = os.path.join(HERE, f"s61_summary_s{seed}.json")
        json.dump(per_seed[str(seed)], open(path, "w"), indent=1)
        print("wrote", path, flush=True)
        for arm in ARMS:
            a = per_seed[str(seed)][arm]
            print(f"  [{seed} {arm}] rho={per_seed[str(seed)]['rho_declared'][arm]['rho_median']:.4f} "
                  f"closure@1={a['closure_median']['1.0']:.4f} "
                  f"(n={a['closure_n_cells']['1.0']}) "
                  f"floor@0 {a['floor_at_alpha0']['count']}/{a['floor_at_alpha0']['total']} "
                  f"clean={a.get('clean_ratio_median', float('nan')):.4f}",
                  flush=True)

    def stat(vals):
        return {"mean": float(np.mean(vals)),
                "sd": float(np.std(vals, ddof=1)),
                "n": len(vals),
                "values": [float(v) for v in vals]}

    sids = [str(s) for s in SEEDS]
    across = {
        "closure_at_1": {
            arm: stat([per_seed[s][arm]["closure_median"]["1.0"] for s in sids])
            for arm in ARMS},
        "clean_ratio_median": {
            arm: stat([per_seed[s][arm]["clean_ratio_median"] for s in sids])
            for arm in ARMS},
    }
    combined = {
        "meta": {
            "experiment": "s61_m2retrofit: Moirai 2.0 R-small 11M, 5000-step "
                          "mechdiv CPT from stock + WiSE-FT 0.5 (corpus-quality "
                          "filter identical across seeds: skip windows with "
                          "max|(fut-loc)/scale| > 100)",
            "seeds": list(SEEDS),
            "seed_role": "CPT data sampler + augmentation draws; init is "
                         "always the STOCK checkpoint",
            "rho": "permutation-probe median rho over 9 cells, declared view",
            "closure_at_1": "(e_stock - e_arm)/(e_stock - 1) at alpha=1, "
                            "declared, median over cells with stock excess "
                            "> 0.02 (analyze_s61.py formula)",
            "floor_at_0": "cells where arm <= stock at alpha=0 (declared)",
            "clean": "median over DS9 of arm/stock clean MSE median",
            "leaderboard_median": "median over 36 cells (9 ds x 4 mechs, rate "
                                  "0.7) of per-cell median relMSE",
            "attack": "bad-fill attack kept single-seed (20260901, "
                      "s61_attack.json) per task scope",
        },
        "seeds": {s: {
            "rho": {arm: per_seed[s]["rho_declared"][arm]["rho_median"]
                    for arm in ARMS},
            "closure_at_1": {arm: per_seed[s][arm]["closure_median"]["1.0"]
                             for arm in ARMS},
            "closure_at_1_n_cells": {
                arm: per_seed[s][arm]["closure_n_cells"]["1.0"]
                for arm in ARMS},
            "floor_at_0": {arm: per_seed[s][arm]["floor_at_alpha0"]
                           for arm in ARMS},
            "clean": {arm: per_seed[s][arm]["clean_ratio_median"]
                      for arm in ARMS},
            "leaderboard_median": {
                f"W50|{f}": per_seed[s]["leaderboard_medians"][f"W50|{f}"]["median_relMSE"]
                for f in ("zero", "linear")},
        } for s in sids},
        "across_seeds": across,
    }
    path = os.path.join(HERE, "s61_seeds.json")
    json.dump(combined, open(path, "w"), indent=1)
    print("wrote", path, flush=True)
    for metric in ("closure_at_1", "clean_ratio_median"):
        for arm in ARMS:
            st = across[metric][arm]
            print(f"  {metric} {arm}: mean={st['mean']:.4f} sd={st['sd']:.4f} "
                  f"values={[round(v, 4) for v in st['values']]}", flush=True)


if __name__ == "__main__":
    main()
