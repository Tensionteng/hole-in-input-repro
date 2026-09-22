#!/usr/bin/env python
"""S64: build the paper-format Table-3 rows from s64_leaderboard_cells.json and run
the reproduction gates.

Table-3 row aggregation (reverse-engineered from the paper's own artifacts:
tab_leaderboard.tex vs s57_moirai2.json / s57_bolt.json / s57_q50.json -- see
s64_bench3/README.md):
  per-dataset cell  = median over the 4 mechanism cells of the per-(ds,mech) medians
  row value         = median over the 9 per-dataset cells
  worst zero cell   = max over the 9 per-dataset zero cells
This aggregation reproduces the paper's rows to the printed 2 decimals for
bolt-base stock (1.51/1.57/3.51, 46.5), Q50-base (1.22/1.51/2.26, 7.2) and
Moirai 2.0 stock (1.18/1.60/11.56, 82.2).

Gates:
  G1  m2-stock's 108 raw cells == s45_pretrain/s57_moirai2.json (rel dev <= 1%)
  G2  m2-stock row (Table-3 agg) == the paper's printed row (1.18/1.60/11.56, 82.2)
  G3  c2-stock's raw cells == s60's artifacts (zero/linear vs s60_leaderboard.json,
      perfect vs s60_grid.json alpha=1.0 declared) -> the s60 run's cells ARE the
      s57 protocol's cells; only the paper's row aggregation for C2 differed
      (flat 36-cell median, checked as G4)
  G4  c2-stock row under the flat-36 aggregation == the paper's C2 row
      (1.61/1.57/2.03, 31.0)
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
DS9 = ("ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness")
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
CELLS = os.path.join(HERE, "s64_leaderboard_cells.json")
OUT = os.path.join(HERE, "s64_leaderboard_rows.json")

PAPER = {  # tab_leaderboard.tex (perfect, linear, zero, worst zero cell)
    "m2-stock": (1.18, 1.60, 11.56, 82.2),
    "c2-stock": (1.61, 1.57, 2.03, 31.0),
    "bolt-stock": (1.51, 1.57, 3.51, 46.5),
    "bolt-q50": (1.22, 1.51, 2.26, 7.2),
}


def per_ds_cells(grid, arm, fill):
    """{ds: median over the 4 mech cell medians}."""
    return {d: float(np.median([grid[f"{d}|{m}|0.7|{fill}|{arm}"]["median"]
                                for m in MECHS])) for d in DS9
            if all(f"{d}|{m}|0.7|{fill}|{arm}" in grid for m in MECHS)}


def row_table3(grid, arm):
    """The paper's Table-3 aggregation: per-dataset cells -> median / max."""
    row = {"per_dataset": {}}
    for fill, pretty in (("oracle", "perfect"), ("linear", "linear"), ("zero", "zero")):
        pds = per_ds_cells(grid, arm, fill)
        row["per_dataset"][pretty] = pds
        row[pretty] = float(np.median(list(pds.values())))
    z = row["per_dataset"]["zero"]
    worst_ds = max(z, key=z.get)
    row["worst_zero_cell"] = {"value": z[worst_ds], "dataset": worst_ds}
    return row


def row_flat36(grid, arm, perfect_from=None):
    """Flat median over the 36 (ds x mech) cells -- the aggregation the paper's
    C2 row was actually computed with (verify_main_text.py `_lb`)."""
    out = {}
    for fill, pretty in (("oracle", "perfect"), ("linear", "linear"), ("zero", "zero")):
        src = perfect_from if (fill == "oracle" and perfect_from is not None) else grid
        vals = [src[f"{d}|{m}|0.7|{fill}|{arm}"]["median"]
                for d in DS9 for m in MECHS
                if f"{d}|{m}|0.7|{fill}|{arm}" in src]
        out[pretty] = float(np.median(vals))
        if fill == "zero":
            out["worst_zero_cell"] = max(vals)
    return out


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def main():
    cells = json.load(open(CELLS))
    grid = cells["grid"]
    arms = ["m2-stock", "m2-w50", "c2-stock", "c2-w50"]
    res = {"meta": {"aggregation": "per-dataset cell = median of 4 mech medians; "
                                   "row = median of 9 ds cells; worst zero = max ds cell",
                    "source_cells": "s64_leaderboard_cells.json"},
           "rows": {}, "rows_flat36": {}, "gates": {}}

    for arm in arms:
        res["rows"][arm] = row_table3(grid, arm)
        res["rows_flat36"][arm] = row_flat36(grid, arm)

    # the two bolt rows from the s57 artifacts, same aggregation (context)
    s57 = {}
    for tag, f, model in (("bolt-stock", "s57_bolt.json", "bolt-base-stock"),
                          ("bolt-q50", "s57_q50.json", "Q50-base")):
        g = json.load(open(os.path.join(EXP, "s45_pretrain", f)))["grid"]
        s57[tag] = row_table3(g, model)
    res["rows_s57_bolt_context"] = s57

    # --------------------------------------------------------------- G1 ----
    ref = json.load(open(os.path.join(EXP, "s45_pretrain", "s57_moirai2.json")))["grid"]
    devs = []
    for d in DS9:
        for m in MECHS:
            for f in ("zero", "linear", "oracle"):
                a = grid[f"{d}|{m}|0.7|{f}|m2-stock"]["median"]
                b = ref[f"{d}|{m}|0.7|{f}|moirai2"]["median"]
                devs.append((rel(a, b), f"{d}|{m}|{f}"))
    devs.sort(reverse=True)
    res["gates"]["G1_m2stock_vs_s57_moirai2"] = {
        "max_rel_dev": devs[0][0], "worst_cell": devs[0][1],
        "n_cells": len(devs), "pass": devs[0][0] <= 0.005,
        "note": "s57_moirai2.json was computed with TF32 ENABLED (eval_s57 "
                "imports eval_s45 -> train_s45 sets allow_tf32=True); the s64 "
                "strict-fp32 cells differ only at that numerical level"}

    # G1b: under the original TF32-on settings the reproduction is bit-exact
    tf = json.load(open(os.path.join(HERE, "s64_m2stock_tf32on_cells.json")))["grid"]
    mx = max(rel(tf[f"{d}|{m}|0.7|{f}|m2-stock"]["median"],
                 ref[f"{d}|{m}|0.7|{f}|moirai2"]["median"])
             for d in DS9 for m in MECHS for f in ("zero", "linear", "oracle"))
    res["gates"]["G1b_m2stock_tf32on_bitexact"] = {
        "max_rel_dev": mx, "n_cells": 108, "pass": mx == 0.0,
        "note": "same script with TF32 on (as s57 effectively ran) + batch=64 "
                "reproduces all 108 s57 cells bit-exactly"}

    # --------------------------------------------------------------- G2 ----
    row = res["rows"]["m2-stock"]
    exp = PAPER["m2-stock"]
    got = (row["perfect"], row["linear"], row["zero"], row["worst_zero_cell"]["value"])
    res["gates"]["G2_m2stock_row_vs_paper"] = {
        "expected": exp, "got": got,
        "pass": all(round(g, 2) == round(e, 2) for g, e in zip(got[:3], exp[:3]))
                and round(got[3], 1) == round(exp[3], 1),
        "note": "compared at the paper's own print precision (2 decimals for the "
                "row, 1 decimal for the worst cell)"}

    # --------------------------------------------------------------- G3 ----
    lb60 = json.load(open(os.path.join(EXP, "s60_c2retrofit",
                                       "s60_leaderboard.json")))["grid"]
    g60 = json.load(open(os.path.join(EXP, "s60_c2retrofit", "s60_grid.json")))["grid"]
    devs = []
    for d in DS9:
        for m in MECHS:
            for f in ("zero", "linear"):
                a = grid[f"{d}|{m}|0.7|{f}|c2-stock"]["median"]
                b = lb60[f"{d}|{m}|0.7|{f}|stock"]["median"]
                devs.append((rel(a, b), f"{d}|{m}|{f}"))
            a = grid[f"{d}|{m}|0.7|oracle|c2-stock"]["median"]
            b = g60[f"{d}|{m}|0.7|1.0|stock|declared"]["median"]
            devs.append((rel(a, b), f"{d}|{m}|perfect"))
    devs.sort(reverse=True)
    res["gates"]["G3_c2stock_vs_s60_artifacts"] = {
        "max_rel_dev": devs[0][0], "worst_cell": devs[0][1],
        "n_cells": len(devs), "pass": devs[0][0] <= 0.01,
        "note": "s60 cells == s57-protocol cells; only the paper's C2 row "
                "aggregation differed (flat-36, see G4)"}

    # --------------------------------------------------------------- G4 ----
    flat = res["rows_flat36"]["c2-stock"]
    exp = PAPER["c2-stock"]
    got = (flat["perfect"], flat["linear"], flat["zero"], flat["worst_zero_cell"])
    res["gates"]["G4_c2stock_flat36_vs_paper"] = {
        "expected": exp, "got": got,
        "pass": all(abs(g - e) <= 0.02 * max(abs(e), 1e-9) + 0.005
                    for g, e in zip(got, exp)),
        "note": "the paper's C2 row (and W50 row) used the flat 36-cell median, "
                "not the per-dataset aggregation of the bolt/M2 rows"}

    # also: the paper's c2-w50 printed row under flat-36 (1.26/1.34/1.81, 79.3)
    flatw = res["rows_flat36"]["c2-w50"]
    res["gates"]["G4b_c2w50_flat36_vs_paper"] = {
        "expected": (1.26, 1.34, 1.81, 79.3),
        "got": (flatw["perfect"], flatw["linear"], flatw["zero"],
                flatw["worst_zero_cell"]),
        "pass": None}

    json.dump(res, open(OUT, "w"), indent=1)

    # ------------------------------------------------------------- print ----
    print(f"{'arm':10s} {'perfect':>8s} {'linear':>8s} {'zero':>8s} {'worst0':>8s}")
    for arm in arms:
        r = res["rows"][arm]
        print(f"{arm:10s} {r['perfect']:8.2f} {r['linear']:8.2f} {r['zero']:8.2f} "
              f"{r['worst_zero_cell']['value']:8.2f} ({r['worst_zero_cell']['dataset']})")
    for tag, r in s57.items():
        print(f"{tag:10s} {r['perfect']:8.2f} {r['linear']:8.2f} {r['zero']:8.2f} "
              f"{r['worst_zero_cell']['value']:8.2f} ({r['worst_zero_cell']['dataset']})"
              "  [s57 artifact]")
    print("\ngates:")
    for k, g in res["gates"].items():
        print(f"  {k}: {'PASS' if g['pass'] else 'FAIL' if g['pass'] is False else 'info'}"
              f"  {json.dumps({kk: vv for kk, vv in g.items() if kk in ('max_rel_dev', 'expected', 'got')})}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
