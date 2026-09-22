#!/usr/bin/env python
"""Appendix: does anything depend on the error being squared?

Both panels plot the MSE reading against the MAE reading of the same measurement, taken from
the same forecasts on the same windows. Points on the diagonal are metric-independent.
"""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
R = {}
for f in sorted(glob.glob(os.path.join(EXP, "s41_mae", "part_*.json"))):
    d = json.load(open(f))
    for k in ("grid", "closure"):
        R.setdefault(k, {}).update(d.get(k, {}))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})

MECH = [("mcar", "scattered", "#2c6fad", "o"), ("block", "block", "#7ba7d1", "s"),
        ("mnar_high", "censoring", "#c0392b", "^"), ("mnar_extreme", "extreme", "#e08a5d", "v")]
RATES = (0.3, 0.7)
MIN = 0.02

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85))

ax = axes[0]
for m, lab, c, mk in MECH:
    xs, ys = [], []
    for k, v in R["grid"].items():
        if k.split("|")[1] != m:
            continue
        xs.append(v["mse"]["median"] - 1)
        ys.append(v["mae"]["median"] - 1)
    ax.scatter(xs, ys, s=22, color=c, marker=mk, label=lab, alpha=.85, zorder=3)
lim = [8e-4, 22]
ax.plot(lim, lim, "k--", lw=.8, zorder=1)
ax.plot(lim, [0.5 * v for v in lim], color="0.55", ls=":", lw=.9, zorder=1)
ax.text(9, 11.5, "equal", fontsize=6, color="0.3", rotation=32)
ax.text(11, 4.2, "MAE $=\\,$half", fontsize=6, color="0.45", rotation=32)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(*lim); ax.set_ylim(*lim)
ax.set_xlabel("excess relMSE over a clean context")
ax.set_ylabel("excess relMAE")
ax.set_title("(a)", fontsize=8)
ax.legend(frameon=False, fontsize=6.4, loc="upper left", handletextpad=.3,
          borderpad=.1, labelspacing=.25)

ax = axes[1]
flip = tot = 0
for m, lab, c, mk in MECH:
    xs, ys = [], []
    for k, v in R["closure"].items():
        if k.split("|")[1] != m:
            continue
        if v["mse"]["native_a1"] - 1 < MIN or v["mae"]["native_a1"] - 1 < MIN:
            continue
        a, b = v["mse"]["closure"], v["mae"]["closure"]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        flip += (a > 0) != (b > 0); tot += 1
        xs.append(min(a, 1.2) * 100); ys.append(min(b, 1.2) * 100)
    ax.scatter(xs, ys, s=22, color=c, marker=mk, label=lab, alpha=.85, zorder=3)
ax.plot([0, 125], [0, 125], "k--", lw=.8, zorder=1)
ax.axhline(0, color="0.6", lw=.7); ax.axvline(0, color="0.6", lw=.7)
ax.set_xlim(0, 125); ax.set_ylim(0, 125)
ax.set_xlabel("closure under MSE (%)"); ax.set_ylabel("closure under MAE (%)")
ax.set_title("(b)", fontsize=8)
ax.text(.04, .95, f"sign flips: {flip} of {tot}", transform=ax.transAxes, va="top",
        fontsize=6.8, color="0.25")
fig.tight_layout(pad=.4, w_pad=1.6)
out = os.path.join(HERE, "app_mae.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out, "| sign flips:", flip)
