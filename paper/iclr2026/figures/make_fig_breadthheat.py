#!/usr/bin/env python
"""Appendix: the damage grid and the permutation probe across nine benchmark datasets."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
S = json.load(open(os.path.join(EXP, "s35_breadth", "s35_results.json")))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})

DS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic",
      "exchange", "illness"]
MECH = [("mcar", "scat."), ("block", "block"), ("mnar_high", "cens."),
        ("mnar_extreme", "extr.")]
RATES = (0.3, 0.7)
cols = [(m, r) for m, _ in MECH for r in RATES]
collab = [f"{l}\n{r}" for _, l in MECH for r in RATES]

M = np.array([[S["grid"][f"{d}|{m}|{r}"] for m, r in cols] for d in DS])

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9),
                         gridspec_kw={"width_ratios": [1.75, 1]})
ax = axes[0]
im = ax.imshow(M, cmap="YlOrRd", norm=LogNorm(vmin=0.95, vmax=M.max()), aspect="auto")
for i in range(len(DS)):
    for j in range(len(cols)):
        v = M[i, j]
        ax.text(j, i, f"{v:.2f}" if v < 10 else f"{v:.1f}", ha="center", va="center",
                fontsize=5.9, color="white" if v > 4 else "0.1")
ax.set_xticks(range(len(cols))); ax.set_xticklabels(collab, fontsize=6.2)
ax.set_yticks(range(len(DS))); ax.set_yticklabels(DS, fontsize=6.8)
ax.set_title("(a)", fontsize=8)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=.032, pad=.02)
cb.ax.tick_params(labelsize=6)

# (b) the probe: plain vs the two declared paths
ax = axes[1]
y = np.arange(len(DS))
ax.barh(y + .19, [S["probe"][d]["plain"] for d in DS], .36, color="#8c8c8c",
        label="plain fill")
ax.barh(y - .19, [max(S["probe"][d]["mask"], S["probe"][d]["nan"]) for d in DS], .36,
        color="#2c6fad", label="declared (mask or NaN)")
for i, d in enumerate(DS):
    ax.plot(0, i - .19, "|", color="#2c6fad", ms=5, mew=1.4)
    ax.text(.025, i - .19, "declared: 0.0000", va="center", fontsize=5.6, color="#1a4c7c")
ax.set_yticks(y); ax.set_yticklabels(DS, fontsize=6.8); ax.invert_yaxis()
ax.set_xlim(-0.01, 1.45); ax.set_xlabel(r"permutation ratio $\rho$", fontsize=7.4)
ax.set_title("(b)", fontsize=8, pad=20)
ax.legend(frameon=False, fontsize=6.4, ncol=2, loc="lower center",
          bbox_to_anchor=(.5, 1.005), handlelength=1.2, columnspacing=1.0)
fig.tight_layout(pad=.4, w_pad=1.4)
out = os.path.join(HERE, "app_breadthheat.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
