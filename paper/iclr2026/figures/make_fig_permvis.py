#!/usr/bin/env python
"""Figure: the permutation test on one window -- vanilla invariance vs CPT divergence.

Reads experiments/s67_permvis/s67_permvis.json (run_s67_permvis.py): one weather
window (channel 0, block outages rate 0.5, declared path), fill A = linear, fill B =
the missing-position fill values permuted (mean/variance preserved bitwise). Two
columns (vanilla bolt tiny = P0, CPT retrofit = P5): top row the horizon forecast
under fill A vs fill B, bottom row the per-layer attention share of masked patches
under fill A vs fill B. Writes figures/fig_permvis.pdf.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
R = json.load(open(os.path.join(EXP, "s67_permvis", "s67_permvis.json")))
H = R["meta"]["protocol"]["H"]

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7.2,
                     "ytick.labelsize": 7.2, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42,
                     "figure.dpi": 200})

BLUE, RED = "#2c6fad", "#c0392b"
ARMS = (("P0", "vanilla bolt tiny"), ("P5", "CPT retrofit"))
TAGS = ("(a)", "(b)", "(c)", "(d)")


def fmt_sci(v):
    """2.014e-03 -> 2\\times10^{-3} (mathtext)."""
    if v == 0:
        return "0"
    m, e = f"{v:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.9), sharey="row")
th = 1 + np.arange(H)
truth = np.asarray(R["window_data"]["truth"])

for j, (arm, coltitle) in enumerate(ARMS):
    d = R[arm]
    qA = np.asarray(d["quantiles_A"])
    qB = np.asarray(d["quantiles_B"])
    sA = np.asarray(d["attn"]["share_A"])
    sB = np.asarray(d["attn"]["share_B"])
    floor_f = d["float_floor"]["forecast_maxabs"]
    floor_s = d["float_floor"]["share_maxabs"]
    dmed = d["delta"]["median_maxabs"]
    dall = d["delta"]["all_maxabs"]
    dsh = np.abs(np.asarray(d["attn"]["delta_share"]))

    # ------------------------------------------------ top: horizon forecast ----
    ax = axes[0, j]
    ax.fill_between(th, qA[0], qA[8], color=BLUE, alpha=0.22, lw=0, zorder=2)
    ax.plot(th, qA[4], color=BLUE, lw=1.2, zorder=4)
    ax.plot(th, qB[4], color=RED, lw=1.1, ls="--", zorder=5)
    ax.plot(th, truth, color="k", lw=1.2, zorder=6)
    ax.set_title(f"{TAGS[j]} {coltitle}", pad=3)
    ax.set_xticks([1, 16, 32, 48, 64])
    ax.set_xlim(0, 65)
    ax.set_xlabel("horizon step")
    if j == 0:
        txt = ("fill B = fill A permuted:\n"
               r"max $|\Delta|$ forecast $=0$ (bitwise)" + "\n"
               rf"batch floor {fmt_sci(floor_f)}")
    else:
        txt = (rf"max $|\Delta|$ median $={dmed:.2f}$" + "\n"
               rf"max $|\Delta|$ any quantile $={dall:.2f}$" + "\n"
               rf"batch floor {fmt_sci(floor_f)}")
    ax.text(0.03, 0.03, txt, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.6, color="0.2",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.2))

    # --------------------------------------- bottom: per-layer attention bars ----
    ax = axes[1, j]
    x = np.arange(len(sA))
    w = 0.38
    ax.bar(x - w / 2, sA, w, color=BLUE, zorder=3)
    ax.bar(x + w / 2, sB, w, color=RED, alpha=0.85, zorder=3)
    ax.axhline(1.0, color="0.45", lw=0.8, ls=":", zorder=2)
    ax.set_xticks(x)
    ax.set_xlim(-0.6, len(sA) - 0.4)
    ax.set_xlabel("encoder layer")
    ax.set_ylim(0, 1.28)
    ax.set_title(f"{TAGS[2 + j]} attention share of masked patches", pad=3)
    imax = int(np.argmax(dsh))
    if j == 0:
        txt = (r"max $|\Delta|$ share $=0$ (bitwise)" + "\n"
               rf"batch floor {fmt_sci(floor_s)}")
    else:
        txt = (rf"max $|\Delta|$ share $={dsh[imax]:.3f}$ (layer {imax})" + "\n"
               rf"batch floor {fmt_sci(floor_s)}")
    ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.6, color="0.2",
            bbox=dict(fc="white", ec="none", alpha=1.0, pad=1.2))
    print(f"{arm}: forecast dmed={dmed:.4g} dall={dall:.4g} floor={floor_f:.3g} | "
          f"share dmax={dsh.max():.4g} floor={floor_s:.3g}")

axes[0, 0].set_ylabel("weather, channel 0")
axes[1, 0].set_ylabel("attn share / uniform")
axes[0, 1].tick_params(labelleft=False)
axes[1, 1].tick_params(labelleft=False)

HANDLES = [
    Line2D([], [], color="k", lw=1.2, label="realized future"),
    Line2D([], [], color=BLUE, lw=1.2, label="median, fill A (linear)"),
    Line2D([], [], color=RED, lw=1.1, ls="--", label="median, fill B (permuted)"),
    Patch(fc=BLUE, alpha=0.22, lw=0, label="80% band, fill A"),
    Patch(fc=BLUE, lw=0, label="share, fill A"),
    Patch(fc=RED, alpha=0.85, lw=0, label="share, fill B"),
]
fig.legend(handles=HANDLES, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.04), handlelength=1.7, columnspacing=1.2)
fig.tight_layout(pad=0.45, w_pad=0.9, h_pad=1.0, rect=(0, 0, 1, 0.93))
out = os.path.join(HERE, "fig_permvis.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
