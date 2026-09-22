#!/usr/bin/env python
"""Heatmap: probe R^2 across layers x knockout segment."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
P = json.load(open(os.path.join(EXP, "s13_knockout", "s13_results.json")))["probe_ko"]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.4, "axes.labelsize": 8,
                     "xtick.labelsize": 6.8, "ytick.labelsize": 7.0,
                     "pdf.fonttype": 42, "figure.dpi": 200})
SEGS = ["none", "l0-3", "l4-7", "l8-11", "all"]
LAB = ["no knockout", "layers 0-3", "layers 4-7", "layers 8-11", "all layers"]
CELLS = [("ETTh1", "block:0.3", "ETTh1, block 30%"),
         ("ETTh1", "mcar:0.7", "ETTh1, scattered 70%"),
         ("weather", "block:0.7", "weather, block 70%")]

fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.95))
for ax, (ds, cell, title) in zip(axes, CELLS):
    A = np.array([P[ds][s][cell]["r2_mean"] for s in SEGS], dtype=float)
    im = ax.imshow(A, cmap="viridis", vmin=0, vmax=0.85, aspect="auto")
    ax.set_xticks(range(0, A.shape[1], 2))
    ax.set_xticklabels(range(0, A.shape[1], 2))
    ax.set_xlabel("encoder layer")
    ax.set_yticks(range(len(SEGS)))
    ax.set_yticklabels(LAB if ax is axes[0] else [""] * len(SEGS), fontsize=6.6)
    ax.set_title(title, fontsize=7.6)
cb = fig.colorbar(im, ax=list(axes), fraction=.022, pad=.015)
cb.set_label(r"probe $R^2$", fontsize=7.2); cb.ax.tick_params(labelsize=6.6)
out = os.path.join(HERE, "app_koheat.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
