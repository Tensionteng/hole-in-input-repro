#!/usr/bin/env python
"""Analysis for S5 Track A (run_s5_fix.py): repair methods.

Reads s5_missing_results.json (plain zero/linear/oracle-oscale baselines) and
s5_fix_results.json (fix0=nan, fix1=observed-stats rescale, fix2=bolt adapter).
Prints relMSE/recovery tables and writes s5_fix.png:
  (a)-(d) bolt, one panel per mechanism: relMSE vs rate for the plain fills
          (zero, linear), the oracle probe (mcar only), fix0 (nan), fix1
          (zero_obsrescale / linear_obsrescale) and fix2 (adapter)
  (e)     timesfm: fix0 identity check (nan vs linear) + fix1 under mcar/mnar
  (f)     p=0.7 summary bars, bolt: every method x mechanism
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S5 = json.load(open(os.path.join(HERE, "s5_missing_results.json")))
FX = json.load(open(os.path.join(HERE, "s5_fix_results.json")))
FIG = os.path.join(HERE, "s5_fix.png")
RATES = [0.1, 0.3, 0.5, 0.7]
DATASETS = ["ETTh1", "ETTm1", "weather"]
MECHS = ["mcar", "block", "mnar_high", "mnar_extreme"]


def mse(model, ds, mech, fill, rate):
    key = f"{mech}:{fill}:{rate}"
    rec = FX.get(model, {}).get(ds, {}).get(key)          # fix results first
    if rec is None and model != "bolt_fix2":              # then plain-run results
        rec = S5.get(model, {}).get(ds, {}).get(key)
    return rec["mse"] if rec else None


def clean(model, ds):
    if model in FX and "clean:none:0.0" in FX[model].get(ds, {}):
        return FX[model][ds]["clean:none:0.0"]["mse"]
    return S5[{"bolt_fix2": "bolt"}.get(model, model)][ds]["clean:none:0.0"]["mse"]


def rel(model, ds, mech, fill, rate, own_clean=False):
    f = mse(model, ds, mech, fill, rate)
    if f is None:
        return None
    if own_clean and model in FX and "clean:none:0.0" in FX[model].get(ds, {}):
        c = clean(model, ds)
    else:
        c = clean({"bolt_fix2": "bolt"}.get(model, model), ds)
    return f / c


def avg_rel(model, mech, fill, rate, own_clean=False):
    v = [rel(model, ds, mech, fill, rate, own_clean) for ds in DATASETS]
    v = [x for x in v if v is not None]
    return np.mean(v) if v else None


def series(model, mech, fill):
    return [avg_rel(model, mech, fill, r) for r in RATES]


def main():
    # ------------------------------------------------ printed tables ----
    print("=" * 84)
    print("Fix1 zero_obsrescale: recovery% of plain zero-fill damage (bolt, mcar)")
    for ds in DATASETS:
        row = []
        for r in RATES:
            z, o, c = (mse("bolt", ds, "mcar", "zero", r),
                       mse("bolt", ds, "mcar", "zero_obsrescale", r),
                       clean("bolt", ds))
            row.append(f"{100 * (z - o) / (z - c):7.1f}%" if z and o and abs(z - c) > 1e-9 else "      -")
        print(f"  {ds:8s} " + " ".join(row))

    print("\n" + "=" * 84)
    print("all methods, dataset-avg relMSE")
    for mech in MECHS:
        print(f"  {mech}:")
        for model, fill, tag in (
                ("bolt", "zero", "bolt zero"), ("bolt", "linear", "bolt linear"),
                ("bolt", "nan", "bolt nan (fix0)"),
                ("bolt", "zero_obsrescale", "bolt zero+rescale (fix1)"),
                ("bolt", "linear_obsrescale", "bolt linear+rescale (fix1)"),
                ("bolt_fix2", "fix2", "bolt fix2 (adapter)")):
            ys = series(model, mech, fill)
            if any(ys):
                print(f"    {tag:26s} " + " ".join(f"{y:6.3f}" if y is not None else "     -" for y in ys))

    # ------------------------------------------------------- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    style = {"zero": ("tab:blue", "o", "-"), "linear": ("tab:green", "^", "-"),
             "zero_oscale": ("tab:cyan", "v", ":"),
             "nan": ("tab:red", "s", "-"),
             "zero_obsrescale": ("tab:orange", "D", "--"),
             "linear_obsrescale": ("tab:olive", "P", "--"),
             "fix2": ("tab:purple", "X", "-")}

    def plot(ax, entries, title):
        for model, mech, fill, label in entries:
            ys = series(model, mech, fill)
            if not any(ys):
                continue
            col, mk, ls = style[fill]
            ax.plot([r for r, y in zip(RATES, ys) if y is not None],
                    [y for y in ys if y is not None], ls, color=col, marker=mk,
                    ms=4, lw=1.2, label=label)
        ax.axhline(1.0, color="gray", lw=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("context missing rate p")
        ax.set_ylabel("MSE / clean MSE")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6.5)

    plot(axes[0, 0], [("bolt", "mcar", "zero", "zero"),
                      ("bolt", "mcar", "zero_oscale", "zero + ORACLE rescale"),
                      ("bolt", "mcar", "zero_obsrescale", "zero + obs rescale (fix1)"),
                      ("bolt", "mcar", "linear", "linear"),
                      ("bolt", "mcar", "nan", "NaN native (fix0)"),
                      ("bolt_fix2", "mcar", "fix2", "adapter (fix2)")],
         "(a) bolt — mcar")
    plot(axes[0, 1], [("bolt", "block", "zero", "zero"),
                      ("bolt", "block", "linear", "linear"),
                      ("bolt", "block", "nan", "NaN native (fix0)"),
                      ("bolt", "block", "zero_obsrescale", "zero + obs rescale (fix1)"),
                      ("bolt_fix2", "block", "fix2", "adapter (fix2)")],
         "(b) bolt — block")
    plot(axes[0, 2], [("bolt", "mnar_high", "zero", "zero"),
                      ("bolt", "mnar_high", "linear", "linear"),
                      ("bolt", "mnar_high", "nan", "NaN native (fix0)"),
                      ("bolt", "mnar_high", "zero_obsrescale", "zero + obs rescale (fix1)"),
                      ("bolt_fix2", "mnar_high", "fix2", "adapter (fix2)")],
         "(c) bolt — mnar_high")
    plot(axes[1, 0], [("bolt", "mnar_extreme", "zero", "zero"),
                      ("bolt", "mnar_extreme", "linear", "linear"),
                      ("bolt", "mnar_extreme", "nan", "NaN native (fix0)"),
                      ("bolt", "mnar_extreme", "zero_obsrescale", "zero + obs rescale (fix1)"),
                      ("bolt_fix2", "mnar_extreme", "fix2", "adapter (fix2)")],
         "(d) bolt — mnar_extreme")
    plot(axes[1, 1], [("timesfm", "mcar", "linear", "linear (explicit)"),
                      ("timesfm", "mcar", "nan", "NaN -> internal interp (fix0)"),
                      ("timesfm", "mcar", "zero", "zero"),
                      ("timesfm", "mcar", "zero_obsrescale", "zero + obs rescale (fix1)"),
                      ("timesfm", "mnar_high", "linear", "linear, mnar_high"),
                      ("timesfm", "mnar_high", "nan", "NaN, mnar_high"),
                      ("timesfm", "mnar_high", "zero_obsrescale", "zero+rescale, mnar_high")],
         "(e) timesfm — fix0 identity + fix1")
    # (f) p=0.7 summary bars, bolt
    ax = axes[1, 2]
    methods = [("bolt", "zero", "zero"), ("bolt", "linear", "linear"),
               ("bolt", "nan", "nan (fix0)"),
               ("bolt", "zero_obsrescale", "zero+resc (fix1)"),
               ("bolt", "linear_obsrescale", "lin+resc (fix1)"),
               ("bolt_fix2", "fix2", "adapter (fix2)")]
    width = 0.13
    xs = np.arange(len(MECHS))
    for i, (model, fill, label) in enumerate(methods):
        vals = [avg_rel(model, mech, fill, 0.7) for mech in MECHS]
        ax.bar(xs + (i - 2.5) * width, [v if v is not None else np.nan for v in vals],
               width, label=label, color=style[fill][0])
    ax.set_xticks(xs, MECHS)
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.set_title("(f) bolt — all repairs at p=0.7 (dataset avg)", fontsize=10)
    ax.set_ylabel("MSE / clean MSE")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {FIG}")


if __name__ == "__main__":
    main()
