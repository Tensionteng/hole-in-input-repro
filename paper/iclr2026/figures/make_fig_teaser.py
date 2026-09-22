#!/usr/bin/env python
"""Intro teaser: one imputation, three interfaces -- and the trade-off they trace.

Panels share axes and the same 150 windows, three datasets, block outages at rate 0.7.
Left to right the declared path reads more of the fill, so the alpha=1 end improves
monotonically -- and the alpha=0 end degrades monotonically, which is the trade-off the
paper is about, visible in one figure.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
EXP = os.path.join(ROOT, "tsfm_missing", "experiments")

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7.2, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})

S46 = json.load(open(os.path.join(EXP, "s46_moirai2", "s46_results.json")))["sweep"]
S33 = json.load(open(os.path.join(EXP, "s33_moirai_causal", "s33_results.json")))["sweep"]
AL = (0.0, 0.25, 0.5, 0.75, 1.0)
DS = ("ETTh1", "ETTm1", "weather")


def cur(S, name, mech="block", rate="0.7"):
    return [float(np.mean([S[f"{d}|{mech}|{rate}|{a}|{name}|declared"]["median"] for d in DS]))
            for a in AL]


# Bolt is measured in both rounds and the two curves agree to every printed digit; we take
# the s46 copy and check the s33 one against it, which is one of the cross-round gates.
b46, b33 = cur(S46, "bolt"), cur(S33, "bolt")
assert max(abs(x - y) for x, y in zip(b46, b33)) < 5e-3, (b46, b33)

PANELS = [
    (b46, "#c0392b", "Chronos-Bolt"),
    (cur(S46, "moirai2"), "#b07d2b", "Moirai 2.0"),
    (cur(S33, "moirai"), "#2c6fad", "Moirai 1.1"),
]

fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=True)
for ax, (y, col, title) in zip(axes, PANELS):
    ax.plot(AL, y, "-o", color=col, lw=2, ms=4.5, zorder=3)
    ax.axhline(1.0, color="k", ls=":", lw=.8)
    ax.set_ylim(0.95, 1.45)
    ax.set_xticks([0, .5, 1.0])
    ax.set_xlabel("imputation quality  $\\alpha$")
    ax.set_title(title, fontsize=8.6, pad=3)
    # the two ends carry the whole story, so mark them rather than the curve
    ax.plot([0.0], [y[0]], "o", ms=9, mfc="none", mec="0.35", mew=1.0, zorder=4)
    ax.plot([1.0], [y[-1]], "o", ms=9, mfc="none", mec="0.35", mew=1.0, zorder=4)
    ax.text(.03, .06, f"{y[0]:.2f} $\\rightarrow$ {y[-1]:.2f}", transform=ax.transAxes,
            fontsize=8.4, fontweight="bold", color=col)

axes[0].set_ylabel("forecast error\n(relative to a clean context)")
axes[0].annotate("linear fill", xy=(0.0, PANELS[0][0][0]), xytext=(0.16, 1.34),
                 fontsize=6.8, color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.5", lw=.8))
axes[2].annotate("perfect\nimputer", xy=(1.0, PANELS[2][0][-1]), xytext=(0.52, 1.16),
                 fontsize=6.8, color="0.3",
                 arrowprops=dict(arrowstyle="->", color="0.5", lw=.8))

fig.tight_layout(pad=.4, w_pad=1.0)
out = os.path.join(HERE, "fig_teaser.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
for y, _, t in PANELS:
    print(f"  {t.splitlines()[0]:52s} {y[0]:.3f} -> {y[-1]:.3f}")
