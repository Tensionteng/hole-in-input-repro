#!/usr/bin/env python
"""S54: the paper's main results table from the sweep9 grid.

Rows = retrofit ladder on stock bolt-tiny (P0..P5), columns = the 9 benchmarks,
two panels: deployment fill (alpha=0, linear) and perfect fill (alpha=1).
Cell = median over the 4 mechanisms of the per-cell median relMSE vs the arm's OWN
clean accuracy (the sweep9 convention; clean ratios are reported as a separate row so
absolute performance can be reconstructed).

Usage: make_main_table.py --files s54_sweep9_a.json s54_sweep9_b.json [--tex]
"""
import argparse
import json
import numpy as np

DS9 = ("ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness")
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
ARMS = ["P0", "P1", "P2", "P4", "P5"]
ROW_LABELS = {
    "P0": "Stock bolt-tiny (as shipped)",
    "P1": "+ restored interface (no training)",
    "P2": "+ adapter only (S27 recipe)",
    "P4": "+ CPT on filtered (budget control)",
    "P5": "+ CPT on the H recipe (ours)",
}


def load(paths):
    sw, cl = {}, {}
    for p in paths:
        d = json.load(open(p))
        sw.update(d.get("sweep", {}))
        cl.update(d.get("clean", {}))
    return sw, cl


def cell(sw, ds, alpha, arm):
    vals = []
    for m in MECHS:
        v = sw.get(f"{ds}|{m}|0.7|{alpha}|{arm}|declared")
        if v is not None:
            vals.append(v["median"])
    return float(np.median(vals)) if vals else float("nan")


def clean_ratio(cl, arm, ref="P0"):
    r = [cl[arm][d]["mse_median"] / cl[ref][d]["mse_median"]
         for d in DS9 if d in cl.get(arm, {}) and d in cl.get(ref, {})]
    return r


def panel(sw, alpha, arms):
    tab = {a: [cell(sw, ds, alpha, a) for ds in DS9] for a in arms}
    best = {}
    for j, ds in enumerate(DS9):
        col = [(a, tab[a][j]) for a in arms if np.isfinite(tab[a][j])]
        best[ds] = min(col, key=lambda x: x[1])[0] if col else None
    return tab, best


def fmt(v, bold, tex):
    s = f"{v:.2f}"
    if bold:
        s = f"\\textbf{{{s}}}" if tex else f"**{s}**"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--tex", action="store_true")
    ap.add_argument("--arms", default=",".join(ARMS),
                        help="subset of arms, e.g. P0,P5 for the base-size table")
    args = ap.parse_args()
    arms = args.arms.split(",")
    sw, cl = load(args.files)

    wins = {a: 0 for a in arms}
    for alpha, title in ((1.0, "Panel A: perfect fill (alpha=1)"),
                         (0.0, "Panel B: deployment fill (alpha=0, linear)")):
        tab, best = panel(sw, alpha, arms)
        if args.tex:
            print(f"% {title}")
            print("\\begin{tabular}{l" + "c" * len(DS9) + "}")
            print("\\toprule")
            print("Method & " + " & ".join(DS9) + " \\\\")
            print("\\midrule")
            for a in arms:
                cells = [fmt(tab[a][j], best[DS9[j]] == a, True) for j in range(len(DS9))]
                print(f"{ROW_LABELS[a]} & " + " & ".join(cells) + " \\\\")
            print("\\bottomrule\n\\end{tabular}\n")
        else:
            print(f"\n## {title}\n")
            print("| Method | " + " | ".join(DS9) + " |")
            print("|" + "---|" * (len(DS9) + 1))
            for a in arms:
                cells = [fmt(tab[a][j], best[DS9[j]] == a, False) for j in range(len(DS9))]
                print(f"| {ROW_LABELS[a]} | " + " | ".join(cells) + " |")
        for j, ds in enumerate(DS9):
            if alpha == 1.0 and best[ds]:
                wins[best[ds]] += 1

    print("\n-- wins per arm (panel A columns): " +
          ", ".join(f"{a}={wins[a]}" for a in arms))
    print("-- clean ratio vs P0 (median over DS9):")
    for a in arms:
        r = clean_ratio(cl, a)
        if r:
            print(f"   {a}: {np.median(r):.3f}  (per-ds: "
                  + ", ".join(f"{x:.2f}" for x in r) + ")")


if __name__ == "__main__":
    main()
