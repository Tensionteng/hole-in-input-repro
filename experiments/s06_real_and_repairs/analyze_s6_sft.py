#!/usr/bin/env python
"""Analysis + figure for S6-F (run_s6_sft.py): masked-augmentation SFT of
chronos-bolt-base. relMSE is always vs the PAIRED zero-shot clean baseline
(bolt_zs clean:none:0.0 on the same 150 windows).

Prints: (1) per-mechanism relMSE tables (zs vs sft_mb vs sft_all, both fills),
(2) SFT-all vs SFT-mb on the mnar grids, (3) clean-MSE drift.
Writes s6_sft.png: one panel per mechanism (dataset-avg relMSE vs rate, 6
model-fill lines), a clean-drift panel, and a per-dataset mnar_high p=0.7 bar
panel.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "s6_sft_results.json")
FIG = os.path.join(HERE, "s6_sft.png")
RATES = [0.1, 0.3, 0.5, 0.7]
MECHS = ["mcar", "block", "mnar_high", "mnar_extreme"]
DATASETS = ["ETTh1", "ETTm1", "weather"]
MODELS = [("bolt_zs", "zero-shot"), ("sft_mb", "SFT-mb"), ("sft_all", "SFT-all")]
FILLS = ["nan", "linear"]


def get(res, model, ds, mech, fill, rate):
    return res.get(model, {}).get(ds, {}).get(f"{mech}:{fill}:{rate}")


def clean_mse(res, ds):
    return res["bolt_zs"][ds]["clean:none:0.0"]["mse"]


def relmse(res, model, ds, mech, fill, rate):
    c = get(res, model, ds, mech, fill, rate)
    return None if c is None else c["mse"] / clean_mse(res, ds)


def avg_rel(res, model, mech, fill, rate, dss=DATASETS):
    vs = [relmse(res, model, ds, mech, fill, rate) for ds in dss]
    vs = [v for v in vs if v is not None]
    return np.mean(vs) if vs else None


def tables(res):
    print("=" * 84)
    print("(1) relMSE vs zero-shot, dataset-avg (per-dataset in parentheses ETT-avg only)")
    for mech in MECHS:
        print(f"\n  {mech}:")
        print(f"    {'model':9s} {'fill':7s} " + " ".join(f"p={r:<5}" for r in RATES))
        for m, _lab in MODELS:
            for fill in FILLS:
                vals = [avg_rel(res, m, mech, fill, r) for r in RATES]
                ett = [avg_rel(res, m, mech, fill, r, ("ETTh1", "ETTm1")) for r in RATES]
                print(f"    {m:9s} {fill:7s} " + " ".join(
                    f"{v:6.2f}" if v is not None else "     -" for v in vals)
                    + "   ETT: " + " ".join(
                    f"{v:5.2f}" if v is not None else "    -" for v in ett))

    print("\n" + "=" * 84)
    print("(2) SFT-all vs SFT-mb on mnar grids (relMSE; negative delta = SFT-all better)")
    for mech in ("mnar_high", "mnar_extreme"):
        for fill in FILLS:
            print(f"  {mech} / {fill}:")
            print(f"    {'dataset':9s} " + " ".join(f"p={r:<13}" for r in RATES))
            for ds in DATASETS:
                cells = []
                for r in RATES:
                    a = relmse(res, "sft_all", ds, mech, fill, r)
                    b = relmse(res, "sft_mb", ds, mech, fill, r)
                    z = relmse(res, "bolt_zs", ds, mech, fill, r)
                    cells.append(f"{z:.2f}/{b:.2f}/{a:.2f}"
                                 if None not in (a, b, z) else "-")
                print(f"    {ds:9s} " + " ".join(f"{c:14}" for c in cells))
            print("    (cell = zs / sft_mb / sft_all)")

    print("\n" + "=" * 84)
    print("(3) clean-MSE drift: SFT clean / zero-shot clean")
    for m in ("sft_mb", "sft_all"):
        row = []
        for ds in DATASETS:
            c = get(res, m, ds, "clean", "none", 0.0)
            row.append(f"{ds}={c['mse'] / clean_mse(res, ds):.3f}" if c else f"{ds}=-")
        print(f"  {m:9s} " + "  ".join(row))


def figure(res):
    style = {("bolt_zs", "nan"): ("tab:gray", "--", "o"),
             ("bolt_zs", "linear"): ("tab:gray", "-", "o"),
             ("sft_mb", "nan"): ("tab:blue", "--", "s"),
             ("sft_mb", "linear"): ("tab:blue", "-", "s"),
             ("sft_all", "nan"): ("tab:red", "--", "^"),
             ("sft_all", "linear"): ("tab:red", "-", "^")}
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.33, wspace=0.28)

    for i, mech in enumerate(MECHS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        for (m, fill), (col, ls, mk) in style.items():
            ys = [avg_rel(res, m, mech, fill, r) for r in RATES]
            if any(v is not None for v in ys):
                ax.plot([r for r, y in zip(RATES, ys) if y is not None],
                        [y for y in ys if y is not None],
                        ls, color=col, marker=mk, ms=4,
                        label=f"{m} {fill}")
        ax.axhline(1.0, color="gray", lw=0.5)
        ax.set_title(f"({chr(97 + i)}) {mech} — dataset-avg relMSE", fontsize=10)
        ax.set_xlabel("context missing rate p")
        ax.set_ylabel("MSE / zero-shot clean MSE")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # (e) clean drift
    ax = fig.add_subplot(gs[1, 1])
    width, xs = 0.35, np.arange(len(DATASETS))
    for j, (m, col) in enumerate((("sft_mb", "tab:blue"), ("sft_all", "tab:red"))):
        vals = []
        for ds in DATASETS:
            c = get(res, m, ds, "clean", "none", 0.0)
            vals.append(c["mse"] / clean_mse(res, ds) if c else np.nan)
        ax.bar(xs + (j - 0.5) * width, vals, width, label=m)
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.set_xticks(xs, DATASETS)
    ax.set_title("(e) clean-MSE drift (SFT clean / zero-shot clean)", fontsize=10)
    ax.set_ylabel("ratio")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # (f) mnar_high p=0.7 per dataset
    ax = fig.add_subplot(gs[1, 2])
    combos = [("bolt_zs", "nan"), ("bolt_zs", "linear"), ("sft_mb", "nan"),
              ("sft_mb", "linear"), ("sft_all", "nan"), ("sft_all", "linear")]
    width, xs = 0.13, np.arange(len(DATASETS))
    for j, (m, fill) in enumerate(combos):
        col, ls, mk = style[(m, fill)]
        vals = [relmse(res, m, ds, "mnar_high", fill, 0.7) or np.nan for ds in DATASETS]
        ax.bar(xs + (j - 2.5) * width, vals, width,
               color=col, alpha=0.55 if fill == "nan" else 1.0,
               hatch="//" if fill == "nan" else "", label=f"{m} {fill}")
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.set_xticks(xs, DATASETS)
    ax.set_title("(f) mnar_high p=0.7 relMSE by dataset", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3, axis="y")

    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {FIG}")


def main():
    res = json.load(open(RESULTS))
    tables(res)
    figure(res)


if __name__ == "__main__":
    main()
