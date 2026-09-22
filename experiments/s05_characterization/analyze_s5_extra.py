#!/usr/bin/env python
"""Analysis + figure for Track C supplementary experiments (run_s5_extra.py).

Prints headline tables and writes s5_extra.png:
  (a) interpolability curve: relMSE vs gap length B (exact-rate blockB masks),
      bolt, ETTh1 + weather, p in {0.1, 0.3}
  (b) MAR control: relMSE vs rate for mcar / block24 / mar, bolt + timesfm
  (c) calibration: empirical coverage of bolt's central 80% PI vs rate,
      mcar vs mnar_high, all datasets
  (d) moirai: relMSE vs rate per mechanism x fill (dataset avg)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "s5_extra_results.json")
FIG = os.path.join(HERE, "s5_extra.png")
RATES = [0.1, 0.3, 0.5, 0.7]
SWEEP_RATES = [0.1, 0.3]
BLOCKS = [4, 8, 16, 24, 48, 96, 168]
DATASETS = ["ETTh1", "ETTm1", "weather"]


def get(res, model, ds, key):
    return res.get(model, {}).get(ds, {}).get(key)


def relmse(res, model, ds, key):
    cfg, clean = get(res, model, ds, key), get(res, model, ds, "clean:none:0.0")
    if cfg is None or clean is None:
        return None
    return cfg["mse"] / clean["mse"]


def table_part1(res):
    print("=" * 78)
    print("Part 1: relMSE vs gap length (exact-rate disjoint blocks, linear fill, bolt)")
    for ds in ("ETTh1", "weather"):
        print(f"  {ds}:")
        print(f"    {'p':>4} " + " ".join(f"B={b:<4}" for b in BLOCKS))
        for p in SWEEP_RATES:
            cells = []
            for b in BLOCKS:
                v = relmse(res, "bolt", ds, f"block{b}:linear:{p}")
                c = get(res, "bolt", ds, f"block{b}:linear:{p}")
                cells.append(f"{v:5.3f}/{c['mean_maxrun']:>3.0f}" if v is not None else "    -")
            print(f"    {p:4.1f} " + " ".join(f"{c:6}" for c in cells))
        print("    (cell = relMSE/realized max gap)")


def table_part2(res):
    print("\n" + "=" * 78)
    print("Part 2: MAR control (linear fill) -- relMSE, and mar mean run length")
    for m in ("bolt", "timesfm"):
        print(f"  {m}:")
        print(f"    {'dataset':9s} {'mech':8s} " + " ".join(f"p={r:<5}" for r in RATES))
        for ds in DATASETS:
            for mech in ("mcar", "mar", "block24"):
                vals = [relmse(res, m, ds, f"{mech}:linear:{r}") for r in RATES]
                print(f"    {ds:9s} {mech:8s} " + " ".join(
                    f"{v:6.3f}" if v is not None else "     -" for v in vals))
        rl = [get(res, m, "ETTh1", f"mar:linear:{r}")["mean_runlen"] for r in RATES]
        print(f"    mar mean run length: " + " ".join(f"{v:5.1f}" for v in rl))


def table_part3(res):
    print("\n" + "=" * 78)
    print("Part 3: bolt central-80%-PI coverage (linear fill); cov_td = coverage on future extremes")
    for ds in DATASETS:
        print(f"  {ds}: clean coverage={get(res, 'bolt_cal', ds, 'clean:none:0.0')['coverage80']:.3f}")
        print(f"    {'mech':10s} " + " ".join(f"p={r:<11}" for r in RATES))
        for mech in ("mcar", "mnar_high"):
            cells = []
            for r in RATES:
                c = get(res, "bolt_cal", ds, f"{mech}:linear:{r}")
                cells.append(f"{c['coverage80']:.3f}/{c.get('coverage80_topdecile', float('nan')):.3f}"
                             if c else "-")
            print(f"    {mech:10s} " + " ".join(f"{c:12}" for c in cells))
        print("    (cell = coverage80 / coverage80_topdecile)")


def table_part4(res):
    if "moirai" not in res:
        print("\nPart 4: moirai not present")
        return
    print("\n" + "=" * 78)
    print("Part 4: moirai-1.1-R-base relMSE (dataset avg), 100 windows")
    print(f"  {'mech':10s} {'fill':7s} " + " ".join(f"p={r:<5}" for r in RATES))
    for mech in ("mcar", "block", "mnar_high"):
        for fill in ("zero", "linear"):
            vals = []
            for r in RATES:
                vs = [relmse(res, "moirai", ds, f"{mech}:{fill}:{r}") for ds in DATASETS]
                vs = [v for v in vs if v is not None]
                vals.append(np.mean(vs) if vs else None)
            print(f"  {mech:10s} {fill:7s} " + " ".join(
                f"{v:6.2f}" if v is not None else "     -" for v in vals))


def figure(res):
    colors = {"mcar": "tab:blue", "mar": "tab:orange", "block24": "tab:red",
              "mnar_high": "tab:purple", "block": "tab:red"}
    ds_color = {"ETTh1": "tab:blue", "ETTm1": "tab:orange", "weather": "tab:green"}
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    # (a) interpolability curve
    for j, ds in enumerate(("ETTh1", "weather")):
        ax = fig.add_subplot(gs[0, j])
        for p, mk in zip(SWEEP_RATES, ("o", "s")):
            xs, ys, gl = [], [], []
            for b in BLOCKS:
                v = relmse(res, "bolt", ds, f"block{b}:linear:{p}")
                c = get(res, "bolt", ds, f"block{b}:linear:{p}")
                if v is not None:
                    xs.append(b)
                    ys.append(v)
                    gl.append(c["mean_maxrun"])
            ax.plot(xs, ys, marker=mk, ms=5, label=f"p={p}")
            for x, y, g in zip(xs, ys, gl):
                if g < x:  # gap infeasible at this rate: annotate realized max gap
                    ax.annotate(f"g={g:.0f}", (x, y), textcoords="offset points",
                                xytext=(0, 7), ha="center", fontsize=7, color="gray")
        ax.set_xscale("log", base=2)
        ax.set_xticks(BLOCKS, [str(b) for b in BLOCKS])
        ax.axhline(1.0, color="gray", lw=0.5)
        ax.set_title(f"(a) relMSE vs gap length — bolt, {ds}, linear fill", fontsize=10)
        ax.set_xlabel("gap length B (points)")
        ax.set_ylabel("MSE / clean MSE")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # (c) coverage
    ax = fig.add_subplot(gs[0, 2])
    for ds in DATASETS:
        cl = get(res, "bolt_cal", ds, "clean:none:0.0")
        for mech, ls in (("mcar", "-"), ("mnar_high", "--")):
            xs, ys = [0.0], [cl["coverage80"]]
            for r in RATES:
                c = get(res, "bolt_cal", ds, f"{mech}:linear:{r}")
                if c:
                    xs.append(r)
                    ys.append(c["coverage80"])
            ax.plot(xs, ys, ls, color=ds_color[ds], marker="o", ms=4,
                    label=f"{mech} / {ds}")
    ax.axhline(0.8, color="k", lw=0.8, ls=":")
    ax.set_title("(c) bolt central-80% PI coverage — linear fill", fontsize=10)
    ax.set_xlabel("context missing rate p")
    ax.set_ylabel("empirical coverage")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # (b) MAR control, one panel per model
    for j, m in enumerate(("bolt", "timesfm")):
        ax = fig.add_subplot(gs[1, j])
        for mech in ("mcar", "mar", "block24"):
            ys = []
            for r in RATES:
                vs = [relmse(res, m, ds, f"{mech}:linear:{r}") for ds in DATASETS]
                vs = [v for v in vs if v is not None]
                ys.append(np.mean(vs) if vs else None)
            ax.plot([r for r, y in zip(RATES, ys) if y], [y for y in ys if y],
                    color=colors[mech], marker="o", ms=5, lw=2, label=f"{mech} (avg)")
            for ds in DATASETS:
                ys_ds = [relmse(res, m, ds, f"{mech}:linear:{r}") for r in RATES]
                ax.plot([r for r, y in zip(RATES, ys_ds) if y],
                        [y for y in ys_ds if y], color=colors[mech], lw=0.6, alpha=0.35)
        ax.axhline(1.0, color="gray", lw=0.5)
        lab = {"bolt": "chronos-bolt-base", "timesfm": "timesfm-2.5-200m"}[m]
        ax.set_title(f"(b) MAR control — {lab}, linear fill", fontsize=10)
        ax.set_xlabel("context missing rate p")
        ax.set_ylabel("MSE / clean MSE")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # (d) moirai
    ax = fig.add_subplot(gs[1, 2])
    if "moirai" in res:
        mech_ls = {"mcar": "-", "block": "--", "mnar_high": ":"}
        for mech in ("mcar", "block", "mnar_high"):
            for fill, col in (("linear", "tab:blue"), ("zero", "tab:red")):
                ys = []
                for r in RATES:
                    vs = [relmse(res, "moirai", ds, f"{mech}:{fill}:{r}") for ds in DATASETS]
                    vs = [v for v in vs if v is not None]
                    ys.append(np.mean(vs) if vs else None)
                ax.plot([r for r, y in zip(RATES, ys) if y], [y for y in ys if y],
                        mech_ls[mech], color=col, marker="o", ms=4,
                        label=f"{mech} {fill}")
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=2)
    else:
        ax.text(0.5, 0.5, "moirai skipped", ha="center", va="center")
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.set_title("(d) moirai-1.1-R-base — relMSE (dataset avg, log)", fontsize=10)
    ax.set_xlabel("context missing rate p")
    ax.set_ylabel("MSE / clean MSE")
    ax.grid(alpha=0.3)

    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {FIG}")


def main():
    res = json.load(open(RESULTS))
    table_part1(res)
    table_part2(res)
    table_part3(res)
    table_part4(res)
    figure(res)


if __name__ == "__main__":
    main()
