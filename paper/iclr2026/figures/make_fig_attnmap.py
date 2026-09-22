#!/usr/bin/env python
"""Figure: attention share received by masked patches, per encoder layer, vanilla vs CPT retrofit.

Reads experiments/s45_pretrain/s512_attn_layers.json (eval_s512_attn.py): per-layer median
share ratio (attention received by patches containing masked points / uniform share) for
the vanilla model (P0) and the one-stage CPT retrofit (P5), at the two ends of the fill-quality axis
(alpha=0 linear fill, alpha=1 oracle), on ETTh1/ETTm1/weather. Two annotated heatmaps on a
shared colour scale. Writes figures/fig_attnmap.pdf (main text, retrofit experiment).
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
D = json.load(open(os.path.join(EXP, "s45_pretrain", "s512_attn_layers.json")))

DS = ("ETTh1", "ETTm1", "weather")
AL = (("0.0", "poor fill"), ("1.0", "perfect fill"))
ARMS = (("P0", "vanilla"), ("P5", "CPT retrofit"))

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                     "ytick.labelsize": 7.5, "pdf.fonttype": 42, "figure.dpi": 200})

grids = {}
for arm, _ in ARMS:
    cols = None
    for d in DS:
        for a, _ in AL:
            col = np.array(D[f"{d}|a={a}|{arm}"]["median"])
            cols = col if cols is None else np.column_stack([cols, col])
    grids[arm] = cols.T                     # [3 datasets x 2 fills, n_layer]

vmin = min(g.min() for g in grids.values())
vmax = max(g.max() for g in grids.values())

ROWLAB = [f"{d} {('poor' if a == '0.0' else 'perfect')}" for d in DS for a, _ in AL]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.9), sharex=True)
for ax, (arm, lab) in zip(axes, ARMS):
    g = grids[arm]
    im = ax.imshow(g, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    n_layer = g.shape[1]
    ax.set_xticks(range(n_layer))
    ax.set_xticklabels([f"{i}" for i in range(n_layer)])
    ax.set_xlabel("encoder layer")
    ax.set_yticks(range(g.shape[0]))
    ax.set_yticklabels(ROWLAB, fontsize=9.5)
    for i in range(g.shape[0]):
        for j in range(n_layer):
            ax.text(j, i, f"{g[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if g[i, j] < vmin + 0.6 * (vmax - vmin) else "black")
    ax.set_title(lab, fontsize=12)
    for s in ax.spines.values():
        s.set_visible(False)
axes[1].tick_params(labelleft=False)
fig.subplots_adjust(left=0.14, right=0.86, top=0.88, bottom=0.16, wspace=0.25)
cax = fig.add_axes([0.89, 0.16, 0.018, 0.72])
fig.colorbar(im, cax=cax)
cax.set_ylabel("attention share of masked patches (uniform = 1)", fontsize=9.5)
cax.tick_params(labelsize=9)
out = os.path.join(HERE, "fig_attnmap.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
