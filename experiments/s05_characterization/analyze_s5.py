#!/usr/bin/env python
"""Analysis for screen S5 (see run_s5_missing.py for the design).

Prints:
  1. relMSE vs missing rate (mcar, linear fill) per dataset x model
  2. mcar vs block at equal nominal rate and at equal corrupted-patch fraction
  3. zero_oscale recovery: what fraction of zero-fill damage the oracle
     rescaling removes (scaling-statistic probe)
  4. t5 / timesfm vs bolt on the configs they share
  5. sanity: oracle anchor == clean

Writes s5_missing_curves.png:
  (a) relMSE vs rate, mcar+linear, one line per dataset x model
  (b) relMSE vs rate, mcar, per fill strategy (dataset-average, bolt)
  (c) relMSE vs corrupted-patch fraction, mcar vs block (bolt, linear),
      with t5 (patch=1) and timesfm (patch=32) overlaid where available
  (d) zero vs zero_oscale bars per rate (bolt, mcar, dataset-average)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "s5_missing_results.json")
FIG = os.path.join(HERE, "s5_missing_curves.png")
FIG_MNAR = os.path.join(HERE, "s5_missing_mnar.png")
RATES = [0.1, 0.3, 0.5, 0.7]
DATASETS = ["ETTh1", "ETTm1", "weather"]
MODELS = ["bolt", "t5", "timesfm"]


def get(res, model, ds, mech, fill, rate):
    return res.get(model, {}).get(ds, {}).get(f"{mech}:{fill}:{rate}")


def relmse(res, model, ds, mech, fill, rate):
    cfg, clean = get(res, model, ds, mech, fill, rate), get(res, model, ds, "clean", "none", 0.0)
    if cfg is None or clean is None:
        return None
    return cfg["mse"] / clean["mse"]


def main():
    res = json.load(open(RESULTS))

    print("=" * 78)
    print("sanity: oracle anchor (mcar:oracle:0.3) must equal clean")
    for m in MODELS:
        for ds in DATASETS:
            c, o = get(res, m, ds, "clean", "none", 0.0), get(res, m, ds, "mcar", "oracle", 0.3)
            if c and o:
                print(f"  {m:8s} {ds:8s} clean={c['mse']:.4f} oracle={o['mse']:.4f} "
                      f"reldiff={abs(o['mse'] - c['mse']) / c['mse']:.2e}")

    print("\n" + "=" * 78)
    print("(1) relMSE, mcar + linear fill, vs missing rate")
    print(f"{'model':9s} {'dataset':9s} " + " ".join(f"p={r:<4}" for r in RATES))
    for m in MODELS:
        for ds in DATASETS:
            vals = [relmse(res, m, ds, "mcar", "linear", r) for r in RATES]
            if any(v is not None for v in vals):
                print(f"{m:9s} {ds:9s} " + " ".join(
                    f"{v:6.3f}" if v is not None else "     -" for v in vals))

    print("\n" + "=" * 78)
    print("(2) mcar vs block (bolt, linear fill): equal rate and equal patch fraction")
    for ds in DATASETS:
        print(f"  {ds}:")
        print(f"    {'rate':>5s} {'cpf_mcar':>8s} {'cpf_block':>8s} "
              f"{'relM_mcar':>9s} {'relM_block':>9s} {'ratio':>6s}")
        for r in RATES:
            a, b = get(res, "bolt", ds, "mcar", "linear", r), get(res, "bolt", ds, "block", "linear", r)
            if not (a and b):
                continue
            ra, rb = relmse(res, "bolt", ds, "mcar", "linear", r), relmse(res, "bolt", ds, "block", "linear", r)
            if ra is None or rb is None:
                continue
            print(f"    {r:5.1f} {a['corr_patch_frac']:8.3f} {b['corr_patch_frac']:8.3f} "
                  f"{ra:9.3f} {rb:9.3f} {ra / rb:6.2f}")
        # equal-patch-fraction: interpolate the mcar curve at block's cpf
        mc = [(c["corr_patch_frac"], relmse(res, "bolt", ds, "mcar", "linear", r))
              for r in RATES if (c := get(res, "bolt", ds, "mcar", "linear", r))]
        bl = [(c["corr_patch_frac"], relmse(res, "bolt", ds, "block", "linear", r), r)
              for r in RATES if (c := get(res, "bolt", ds, "block", "linear", r))]
        mc = sorted((x, y) for x, y in mc if x is not None and y is not None)
        for cpf_b, rel_b, r in bl:
            if cpf_b is None or rel_b is None:
                continue
            if mc and mc[0][0] <= cpf_b <= mc[-1][0]:
                rel_m = np.interp(cpf_b, [x for x, _ in mc], [y for _, y in mc])
                print(f"    @cpf={cpf_b:.3f} (block p={r}): relMSE block={rel_b:.3f} "
                      f"vs mcar(interp)={rel_m:.3f} -> mcar/block={rel_m / rel_b:.2f}")

    print("\n" + "=" * 78)
    print("(3) zero_oscale recovery, bolt mcar: (MSE_zero-MSE_oscale)/(MSE_zero-MSE_clean)")
    for m in MODELS:
        recs = []
        for ds in DATASETS:
            row = []
            for r in RATES:
                z, o, c = (get(res, m, ds, "mcar", "zero", r),
                           get(res, m, ds, "mcar", "zero_oscale", r),
                           get(res, m, ds, "clean", "none", 0.0))
                row.append(None if not (z and o and c) or abs(z["mse"] - c["mse"]) < 1e-12
                           else 100 * (z["mse"] - o["mse"]) / (z["mse"] - c["mse"]))
            if any(v is not None for v in row):
                recs.append(row)
                print(f"  {m:8s} {ds:8s} " + " ".join(
                    f"{v:7.1f}%" if v is not None else "       -" for v in row))
        if recs:
            avg = [np.nanmean([row[i] for row in recs if row[i] is not None])
                   for i in range(len(RATES))]
            print(f"  {m:8s} {'avg':8s} " + " ".join(f"{v:7.1f}%" for v in avg))

    print("\n" + "=" * 78)
    print("(4) aux models vs bolt on shared configs (relMSE)")
    for mech, fill in (("mcar", "zero"), ("mcar", "linear"), ("block", "linear")):
        for r in RATES:
            cells = []
            for m in MODELS:
                vals = [relmse(res, m, ds, mech, fill, r) for ds in DATASETS]
                vals = [v for v in vals if v is not None]
                cells.append(f"{np.mean(vals):6.3f}" if vals else "     -")
            print(f"  {mech:5s} {fill:6s} p={r}: bolt={cells[0]} t5={cells[1]} timesfm={cells[2]}")

    # ------------------------------------------------------------- figure ----
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.30)
    colors = {"ETTh1": "tab:blue", "ETTm1": "tab:orange", "weather": "tab:green"}
    mstyle = {"bolt": "-", "t5": "--", "timesfm": ":"}
    ds_marker = {"ETTh1": "o", "ETTm1": "^", "weather": "s"}

    # (a) relMSE vs rate, mcar + linear fill, one line per dataset x model
    ax = fig.add_subplot(gs[0, 0:2])
    for m in MODELS:
        for ds in DATASETS:
            ys = [relmse(res, m, ds, "mcar", "linear", r) for r in RATES]
            if any(ys):
                ax.plot([r for r, y in zip(RATES, ys) if y],
                        [y for y in ys if y], mstyle[m], color=colors[ds],
                        marker=ds_marker[ds], ms=4, label=f"{m} / {ds}")
    ax.set_title("(a) relMSE vs missing rate — mcar, linear fill")
    ax.set_xlabel("context missing rate p")
    ax.set_ylabel("MSE / clean MSE")
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)

    # (b) relMSE vs rate per fill strategy, bolt, dataset average
    ax = fig.add_subplot(gs[0, 2:4])
    for fill, mk in (("zero", "o"), ("ffill", "s"), ("linear", "^"), ("zero_oscale", "D")):
        ys = []
        for r in RATES:
            vals = [relmse(res, "bolt", ds, "mcar", fill, r) for ds in DATASETS]
            vals = [v for v in vals if v is not None]
            ys.append(np.mean(vals) if vals else None)
        ax.plot([r for r, y in zip(RATES, ys) if y], [y for y in ys if y],
                marker=mk, label=fill)
    ax.set_title("(b) relMSE vs rate by fill — bolt, mcar, dataset avg")
    ax.set_xlabel("context missing rate p")
    ax.set_ylabel("MSE / clean MSE")
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) patch-corruption collapse test, one small multiple per model
    mech_color = {"mcar": "tab:red", "block": "tab:blue"}
    patch_of = {"bolt": 16, "t5": 1, "timesfm": 32}
    label_of = {"bolt": "chronos-bolt-base", "t5": "chronos-t5-small",
                "timesfm": "timesfm-2.5-200m"}
    for i, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[1, i])
        for mech in ("mcar", "block"):
            for ds in DATASETS:
                pts = []
                for r in RATES:
                    c = get(res, m, ds, mech, "linear", r)
                    cl = get(res, m, ds, "clean", "none", 0.0)
                    if c and cl:
                        pts.append((c["corr_patch_frac"], c["mse"] / cl["mse"]))
                if pts:
                    xs, ys = zip(*sorted(pts))
                    ax.plot(xs, ys, color=mech_color[mech], marker=ds_marker[ds],
                            ms=4, lw=1.1, alpha=0.85)
        ax.set_title(f"(c) {label_of[m]} (patch={patch_of[m]})", fontsize=10)
        ax.set_xlabel("corrupted-patch fraction")
        if i == 0:
            ax.set_ylabel("MSE / clean MSE")
        ax.axhline(1.0, color="gray", lw=0.5)
        ax.grid(alpha=0.3)
        if i == 0:
            handles = ([Line2D([], [], color=mech_color[k], lw=1.1, label=k)
                        for k in ("mcar", "block")] +
                       [Line2D([], [], color="gray", marker=ds_marker[ds], ls="",
                               label=ds) for ds in DATASETS])
            ax.legend(handles=handles, fontsize=7, loc="upper left")

    # (d) scaling probe: zero vs zero_oscale bars, bolt, mcar, dataset average
    ax = fig.add_subplot(gs[1, 3])
    width, xs = 0.35, np.arange(len(RATES))
    zbar, obar, recs = [], [], []
    for r in RATES:
        z = [relmse(res, "bolt", ds, "mcar", "zero", r) for ds in DATASETS]
        o = [relmse(res, "bolt", ds, "mcar", "zero_oscale", r) for ds in DATASETS]
        z = [v for v in z if v is not None]
        o = [v for v in o if v is not None]
        zbar.append(np.mean(z) if z else np.nan)
        obar.append(np.mean(o) if o else np.nan)
        rec = [(zz - oo) / (zz - 1.0) * 100 for zz, oo in zip(z, o) if abs(zz - 1.0) > 1e-9]
        recs.append(np.mean(rec) if rec else np.nan)
    ax.bar(xs - width / 2, zbar, width, label="zero fill")
    ax.bar(xs + width / 2, obar, width, label="zero_oscale (oracle rescale)")
    for i, rc in enumerate(recs):
        if not np.isnan(rc):
            ax.text(i, max(zbar[i], obar[i]) + 0.01, f"rec {rc:.0f}%",
                    ha="center", fontsize=8)
    ax.set_xticks(xs, [f"p={r}" for r in RATES])
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.set_title("(d) scaling probe — bolt, mcar, dataset avg", fontsize=10)
    ax.set_ylabel("MSE / clean MSE")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"\nfigure -> {FIG}")
    mnar_figure(res)


def mnar_figure(res):
    """Second figure: the value-censored (MNAR) mechanisms vs mcar."""
    mech_style = {"mcar": ("tab:blue", "o"), "mnar_high": ("tab:red", "s"),
                  "mnar_extreme": ("tab:purple", "^")}
    label_of = {"bolt": "chronos-bolt-base", "t5": "chronos-t5-small",
                "timesfm": "timesfm-2.5-200m"}
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 6, hspace=0.35, wspace=0.30)

    # (a) relMSE vs rate, linear fill, dataset avg — one multiple per model
    for i, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[0, 2 * i:2 * i + 2])
        for mech, (col, mk) in mech_style.items():
            ys = []
            for r in RATES:
                vals = [relmse(res, m, ds, mech, "linear", r) for ds in DATASETS]
                vals = [v for v in vals if v is not None]
                ys.append(np.mean(vals) if vals else None)
            ax.plot([r for r, y in zip(RATES, ys) if y], [y for y in ys if y],
                    color=col, marker=mk, ms=4, label=mech)
        ax.set_title(f"(a) {label_of[m]} — linear fill, dataset avg", fontsize=10)
        ax.set_xlabel("context missing rate p")
        if i == 0:
            ax.set_ylabel("MSE / clean MSE")
        ax.axhline(1.0, color="gray", lw=0.5)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # (b) bolt: error concentration on extreme futures (top-decile / rest ratio)
    ax = fig.add_subplot(gs[1, 0:3])
    for mech, (col, mk) in mech_style.items():
        ys = []
        for r in RATES:
            ratios = []
            for ds in DATASETS:
                c = get(res, "bolt", ds, mech, "linear", r)
                if c and "mse_topdecile" in c and c["mse_rest"]:
                    ratios.append(c["mse_topdecile"] / c["mse_rest"])
            ys.append(np.mean(ratios) if ratios else None)
        ax.plot([r for r, y in zip(RATES, ys) if y], [y for y in ys if y],
                color=col, marker=mk, ms=5, label=mech)
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.set_title("(b) bolt: MSE on extreme futures / MSE on the rest", fontsize=10)
    ax.set_xlabel("context missing rate p")
    ax.set_ylabel("mse_topdecile / mse_rest (dataset avg)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) oracle rescale under mnar_high: zero vs zero_oscale per model
    ax = fig.add_subplot(gs[1, 3:6])
    mcol = {"bolt": "tab:blue", "t5": "tab:orange", "timesfm": "tab:green"}
    for m in MODELS:
        for fill, ls in (("zero", "-"), ("zero_oscale", "--")):
            ys = []
            for r in RATES:
                vals = [relmse(res, m, ds, "mnar_high", fill, r) for ds in DATASETS]
                vals = [v for v in vals if v is not None]
                ys.append(np.mean(vals) if vals else None)
            ax.plot([r for r, y in zip(RATES, ys) if y], [y for y in ys if y],
                    ls, color=mcol[m], marker="o", ms=4,
                    label=f"{m} {'zero' if fill == 'zero' else 'zero_oscale'}")
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.set_title("(c) oracle rescale fails under mnar_high (dataset avg)", fontsize=10)
    ax.set_xlabel("context missing rate p")
    ax.set_ylabel("MSE / clean MSE")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.savefig(FIG_MNAR, dpi=150, bbox_inches="tight")
    print(f"figure -> {FIG_MNAR}")


if __name__ == "__main__":
    main()
