#!/usr/bin/env python
"""S65 analysis: the fair-budget control scorecard for both families.

Per family (c2 = Chronos-2, m2 = Moirai 2.0), from the s65 eval JSONs:
  rho          median permutation rho on the declared (restored/retaining) path
               (s65_{c2,m2}_probe.json agg; plain/nan kept for context)
  closure@a    (e_stock - e_arm)/(e_stock - 1), per cell (ds|mech|rate, declared
               view), median over cells with stock excess > 0.02 -- analyze_s60 /
               analyze_s61 verbatim
  floor@a=0    cells where the arm is no worse than stock at alpha=0 (declared)
  clean ratio  median over the 9 datasets of arm clean MSE / stock clean MSE
  leaderboard  4 mechs x rate 0.7 under {zero, linear} (+ oracle from the grid's
               alpha=1.0 declared cells -- s64 G3 showed these ARE the oracle
               cells), median relMSE under BOTH aggregations:
               flat-36 (the paper's C2-row convention) and per-dataset Table-3
               (median of 9 per-dataset medians-of-4; the paper's M2/Bolt-row
               convention) -- s64_bench3/aggregate_s64_leaderboard.py helpers

Writes s65_summary.json. Reference recipe-arm numbers (s60/s61 summaries) and
the paper's Bolt control row are embedded for side-by-side reading.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "s64_bench3"))
import aggregate_s64_leaderboard as agg64

ALPHAS = ("0.0", "0.5", "1.0")
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
DS9 = ("ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness")

# measured 2026-09-01 on the s65 artifacts (see s65_summary.json for provenance)
VERDICTS = {
    "c2": "continued training without the recipe DEGRADES robustness rather than "
          "adding to it: the declaration endpoint the control never saw collapses "
          "(leaderboard zero 2.03 -> 10.78 flat-36 / 2.10 -> 20.16 per-dataset, "
          "worst cell 31.0 -> 2404.9, 0W/36L vs stock) and declared-path closure "
          "at realistic fills is +15% (alpha=0) against the recipe arm's +68%. "
          "Its rho (0.848) and oracle-fill closure (70%) are interface "
          "throughput, not learning: stock's untrained plain view already reads "
          "fill content (rho 0.898, closure@1 = 100%). The recipe's gain is the "
          "recipe's, not the budget's.",
    "m2": "continued training without the recipe leaves the retainer's "
          "missingness behaviour untouched: leaderboard zero 8.97 -> 8.64 flat-36 "
          "/ 11.56 -> 11.94 per-dataset (worst cell ~250 both), closure@1 +13% "
          "against the recipe arm's +36% (CPT) / +52% (W50), floor 48/72, rho "
          "0.897 vs stock's 0.949. No missingness awareness is gained from clean "
          "data and none of stock's native behaviour is lost -- the recipe "
          "carries the retrofit's gains.",
}

# context: the recipe arms (s60/s61 summaries) and the paper's Bolt control
RECIPE_REF = {
    "c2_CPT": {"closure1": 0.7950, "floor": "66/72", "clean": 1.0387,
               "rho_declared": 0.8149},
    "c2_W50": {"closure1": 0.5386, "floor": "59/72", "clean": 0.9934,
               "rho_declared": 0.8546},
    "m2_CPT": {"closure1": 0.3588, "floor": "65/72", "clean": 1.0312,
               "rho_declared": 0.8826},
    "m2_W50": {"closure1": 0.5245, "floor": "66/72", "clean": 1.0113,
               "rho_declared": 0.8776},
    "bolt_control_P4": {"closure1": -0.189, "floor": "9/24", "clean": 1.038,
                        "rho_declared": 0.000,
                        "leaderboard_zero": "3.51 -> 9.80 (worst 46.5 -> 393.4)"},
}


def closure_floor_clean(grid, arms):
    """analyze_s60/analyze_s61's exact closure/floor/clean-ratio computation."""
    cells = grid["grid"]
    keys = sorted({tuple(k.rsplit("|", 5)[i] for i in (0, 1, 2))
                   for k in cells
                   if k.rsplit("|", 5)[4] == "stock"
                   and k.rsplit("|", 5)[5] == "declared"})
    out = {}
    for arm in arms:
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
        res = {"closure_median": {al: float(np.median(v)) if v else None
                                  for al, v in clo.items()},
               "closure_mean": {al: float(np.mean(v)) if v else None
                                for al, v in clo.items()},
               "closure_n_cells": {al: len(v) for al, v in clo.items()},
               "floor_at_alpha0": {"count": floor_ct, "total": floor_tot},
               "closure_cells": clo_cells}
        cl = grid.get("clean", {})
        if arm in cl and "stock" in cl:
            ratios = [cl[arm][ds]["mse_median"] / cl["stock"][ds]["mse_median"]
                      for ds in cl["stock"] if ds in cl[arm]]
            res["clean_ratio_median"] = float(np.median(ratios))
            res["clean_ratio_per_ds"] = {ds: cl[arm][ds]["mse_median"]
                                         / cl["stock"][ds]["mse_median"]
                                         for ds in cl["stock"] if ds in cl[arm]}
        out[arm] = res
    return out


def closure_by_view(grid, arms):
    """Closure vs stock-DECLARED for every arm x view at alpha {0,1}.

    The diagnostic that separates interface throughput from learned missingness
    awareness: at alpha=1 the 'fill' IS the ground truth, so any content-reading
    view (including stock's untrained PLAIN view) mechanically closes ~100% of
    stock's excess; the declared view at alpha=0 (realistic fill, true flag) is
    where learning must show up.
    """
    cells = grid["grid"]
    keys = sorted({tuple(k.rsplit("|", 5)[i] for i in (0, 1, 2))
                   for k in cells
                   if k.rsplit("|", 5)[4] == "stock"
                   and k.rsplit("|", 5)[5] == "declared"})
    out = {}
    for armview in [("stock", "plain")] + [(a, v) for a in arms
                                           for v in ("declared", "plain")]:
        arm, view = armview
        acc = {"0.0": [], "1.0": []}
        for ds, mech, rate in keys:
            for al in ("0.0", "1.0"):
                ks = cells.get(f"{ds}|{mech}|{rate}|{al}|stock|declared")
                ka = cells.get(f"{ds}|{mech}|{rate}|{al}|{arm}|{view}")
                if ks and ka and ks["median"] - 1 > 0.02:
                    acc[al].append((ks["median"] - ka["median"])
                                   / (ks["median"] - 1))
        out[f"{arm}|{view}"] = {
            f"closure_at_alpha{al}": {"median": float(np.median(v)) if v else None,
                                      "n": len(v)} for al, v in acc.items()}
    return out


def board_with_oracle(lb, grid, arms):
    """Leaderboard cells {zero,linear,nan} + oracle mapped from the alpha-sweep
    grid (rate 0.7, alpha 1.0, declared) -- s64 G3's identity."""
    cells = dict(lb["grid"])
    for ds in DS9:
        for m in MECHS:
            for a in arms + ("stock",):
                g = grid["grid"].get(f"{ds}|{m}|0.7|1.0|{a}|declared")
                if g is not None:
                    cells[f"{ds}|{m}|0.7|oracle|{a}"] = {
                        "median": g["median"], "mean": g["mean"]}
    return cells


def board_rows(cells, arms):
    """Both aggregations for every arm, via the s64 helpers."""
    out = {}
    for a in ("stock",) + arms:
        out[a] = {"table3_per_dataset": agg64.row_table3(cells, a),
                  "flat36": agg64.row_flat36(cells, a)}
    # win/loss vs stock per fill (analyze_s60 convention)
    for a in arms:
        wl = {}
        for fill in ("zero", "linear", "nan"):
            wins = losses = ties = 0
            marg = []
            for k, v in cells.items():
                ds, mech, rate, f, arm = k.rsplit("|", 4)
                if f != fill or arm != a:
                    continue
                vs = cells.get(f"{ds}|{mech}|{rate}|{fill}|stock")
                if vs is None:
                    continue
                marg.append(vs["median"] - v["median"])
                wins += int(v["median"] < vs["median"] - 1e-9)
                losses += int(v["median"] > vs["median"] + 1e-9)
                ties += int(abs(v["median"] - vs["median"]) <= 1e-9)
            wl[fill] = {"wins": wins, "losses": losses, "ties": ties,
                        "median_margin_relMSE": float(np.median(marg)) if marg else None}
        out[a]["vs_stock"] = wl
    return out


def family(tag, arms):
    grid = json.load(open(os.path.join(HERE, f"s65_{tag}_grid.json")))
    probe = json.load(open(os.path.join(HERE, f"s65_{tag}_probe.json")))
    lb = json.load(open(os.path.join(HERE, f"s65_{tag}_leaderboard.json")))
    res = {"arms": arms,
           "probe_rho": probe["agg"],
           "scorecard": closure_floor_clean(grid, arms),
           "closure_by_view_vs_stockdeclared": closure_by_view(grid, arms)}
    cells = board_with_oracle(lb, grid, arms)
    res["leaderboard"] = board_rows(cells, arms)
    res["verdict"] = VERDICTS[tag]
    return res


def main():
    out = {"meta": {
        "experiment": "S65 fair-budget control: 5,000 CPT steps, identical "
                      "corpus/batch/optimizer/seed to the recipe arms (s60/s61), "
                      "augmentation regime 'filtered' (no missingness at all)",
        "protocols": "eval_s60.py / eval_s61.py verbatim (same windows per cell, "
                     "strict fp32); analyze_s60/analyze_s61 closure/floor/clean; "
                     "s64_bench3 aggregations (flat-36 and per-dataset Table-3)",
        "question": "are the retrofit gains from the recipe or merely from 5,000 "
                    "more steps of training?",
        "bolt_control_reference": "train_s53.py P4: closure -18.9% +/- 9.4, "
                                  "rho 0.000, clean 1.038, leaderboard zero "
                                  "3.51 -> 9.80, worst cell 46.5 -> 393.4"},
        "recipe_reference": RECIPE_REF}
    for tag, arms in (("c2", ("C2CTL", "C2W50")), ("m2", ("M2CTL", "M2W50"))):
        out[tag] = family(tag, arms)

    json.dump(out, open(os.path.join(HERE, "s65_summary.json"), "w"), indent=1)

    for tag in ("c2", "m2"):
        f = out[tag]
        print(f"\n=== {tag.upper()} fair-budget control ===")
        for arm in f["arms"]:
            sc = f["scorecard"][arm]
            rho = f["probe_rho"].get(f"{arm}|declared", {})
            lb = f["leaderboard"][arm]
            st = f["leaderboard"]["stock"]
            print(f"[{arm}] rho(declared) median={rho.get('rho_median'):.4f} "
                  f"(max {rho.get('rho_max'):.4f})")
            print(f"  closure@1 median={sc['closure_median']['1.0']:.4f} "
                  f"(n={sc['closure_n_cells']['1.0']}), floor@0 "
                  f"{sc['floor_at_alpha0']['count']}/{sc['floor_at_alpha0']['total']}, "
                  f"clean ratio {sc.get('clean_ratio_median', float('nan')):.4f}")
            for agg_name in ("table3_per_dataset", "flat36"):
                row, srow = lb[agg_name], st[agg_name]
                print(f"  board[{agg_name}] zero {srow['zero']:.2f} -> "
                      f"{row['zero']:.2f}, linear {srow['linear']:.2f} -> "
                      f"{row['linear']:.2f}, perfect {srow['perfect']:.2f} -> "
                      f"{row['perfect']:.2f}")
            print("  vs stock: " + "  ".join(
                f"{fl} {v['wins']}W/{v['losses']}L/{v['ties']}T"
                for fl, v in lb["vs_stock"].items()))
        print("verdict:", f["verdict"])
    print("\nwrote", os.path.join(HERE, "s65_summary.json"))


if __name__ == "__main__":
    main()
