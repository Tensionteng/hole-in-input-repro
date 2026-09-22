#!/usr/bin/env python
"""Heatmap of the full characterisation grid: mechanism x rate x fill."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
R = json.load(open(os.path.join(EXP, "s05_characterization", "s5_missing_results.json")))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
                     "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
                     "pdf.fonttype": 42, "figure.dpi": 200})
DS = ("ETTh1", "ETTm1", "weather")
RATES = (0.1, 0.3, 0.5, 0.7)
MECHS = [("mcar", "scattered"), ("block", "block"),
         ("mnar_high", "censoring"), ("mnar_extreme", "extreme cens.")]
FILLS = [("zero", "zero"), ("ffill", "forward"), ("linear", "linear")]
MODELS = [("bolt", "Chronos-Bolt"), ("timesfm", "TimesFM")]

fig, axes = plt.subplots(1, len(MODELS), figsize=(7.1, 2.6))
norm = TwoSlopeNorm(vmin=0.8, vcenter=1.0, vmax=4.0)
for ax, (mkey, mlab) in zip(np.atleast_1d(axes), MODELS):
    M = R.get(mkey, {})
    rows, ylab = [], []
    for mech, mlabel in MECHS:
        for fk, fl in FILLS:
            r = []
            for p in RATES:
                v = [M[d][f"{mech}:{fk}:{p}"]["mse"] / M[d]["clean:none:0.0"]["mse"]
                     for d in DS if d in M and f"{mech}:{fk}:{p}" in M[d]]
                r.append(np.mean(v) if v else np.nan)
            rows.append(r); ylab.append(f"{mlabel} · {fl}")
    A = np.array(rows)
    im = ax.imshow(A, cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(len(RATES))); ax.set_xticklabels([f"{p:g}" for p in RATES])
    ax.set_xlabel("missingness rate")
    ax.set_yticks(range(len(ylab)))
    ax.set_yticklabels(ylab if ax is np.atleast_1d(axes)[0] else [""] * len(ylab), fontsize=6.2)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if np.isfinite(A[i, j]):
                ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=5.6,
                        color="white" if (A[i, j] > 2.2 or A[i, j] < 0.88) else "0.15")
    for k in range(1, len(MECHS)):
        ax.axhline(k * len(FILLS) - .5, color="white", lw=1.6)
    ax.set_title(mlab)
cb = fig.colorbar(im, ax=list(np.atleast_1d(axes)), fraction=.03, pad=.02,
                  ticks=[0.8, 1.0, 2.0, 3.0, 4.0])
cb.set_label("relMSE vs clean", fontsize=7.2); cb.ax.tick_params(labelsize=6.6)
out = os.path.join(HERE, "app_grid.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
