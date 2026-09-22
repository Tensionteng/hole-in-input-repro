#!/usr/bin/env python
"""Figure: the imputation ceiling, and what removing one line recovers.

Reads the S37 sweep, which supersedes S27: it resets the input-patch-embedding before
computing any clean denominator, and normalises each interface by its OWN clean forecast so a
generic gain from the adaptation is not charged to the interface. Panel set (a) is the three
datasets the adapter was trained on; (b) is the six it never saw.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
SW = {}
for f in ("own_insample.json", "own_held_a.json", "own_held_b.json", "own_held_c.json"):
    p = os.path.join(EXP, "s37_capbreadth", f)
    if os.path.exists(p):
        SW.update(json.load(open(p))["sweep"])

AL = (0.0, 0.25, 0.5, 0.75, 1.0)
DS_IN = ("ETTh1", "ETTm1", "weather")
DS_OUT = ("ETTh2", "ETTm2", "electricity", "traffic", "exchange", "illness")
MECHS_ALL = [("mcar", "scattered dropout"), ("block", "block outage"),
             ("mnar_high", "value censoring"), ("mnar_extreme", "extreme censoring")]
# The main-text figure is the held-out row across all four mechanisms; the appendix keeps
# the in-sample row, same layout.
MECHS = MECHS_ALL
RATE = 0.7

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 200,
})

C_NATIVE = "#c0392b"
C_DUAL = "#2c6fad"
C_PLAIN = "#8c8c8c"


def curve(iface, mech, dss):
    """Median over datasets of the paired per-window median, own-clean normalised."""
    return np.array([np.median([SW[f"{d}|{mech}|{RATE}|{a}|{iface}"]["median_own"]
                                for d in dss
                                if f"{d}|{mech}|{RATE}|{a}|{iface}" in SW]) for a in AL])


# Two renderings from one source: the main text shows the held-out row, which is the
# stronger claim; the appendix keeps the in-sample row in the same 1x4 layout.
import sys
ROWS = ([(DS_IN, "adapter's own\n3 datasets")] if "--insample" in sys.argv
        else [(DS_OUT, "6 held-out\ndatasets")])
handles = []
fig, axes = plt.subplots(len(ROWS), len(MECHS),
                         figsize=(7.1 * len(MECHS) / 4,
                                  2.15 if len(ROWS) == 1 else 3.75),
                         sharex=True, squeeze=False)
for row, (dss, rlab) in enumerate(ROWS):
 for ax, (mech, title) in zip(axes[row], MECHS):
    nat = curve("native", mech, dss)
    dual = np.minimum(curve("dual", mech, dss), curve("dual_obsnorm", mech, dss))
    ax.axhline(1.0, color="k", ls=":", lw=0.7, zorder=1)
    l1, = ax.plot(AL, nat, "-s", color=C_NATIVE, lw=1.6, ms=3.4, zorder=3)
    l2, = ax.plot(AL, dual, "-^", color=C_DUAL, lw=1.6, ms=3.6, zorder=3)
    handles[:] = [l1, l2]
    if row == len(ROWS) - 1:
        ax.set_xlabel(r"imputation quality $\alpha$")
    ax.set_xticks([0, 0.5, 1.0])
    ax.margins(x=0.06)
    # annotate the gap at alpha = 1
    lo, hi = dual[-1], nat[-1]
    if hi - lo > 0.02:
        ax.annotate("", xy=(1.0, hi), xytext=(1.0, lo),
                    arrowprops=dict(arrowstyle="<->", color="0.35", lw=0.9))
        ax.text(0.94, (hi + lo) / 2, f"{100*(hi-lo)/max(hi-1,1e-9):.0f}%",
                ha="right", va="center", fontsize=7.5, color="0.2",
                bbox=dict(fc="white", ec="none", pad=0.6))
for row, (_, rlab) in enumerate(ROWS):
    axes[row][0].set_ylabel((rlab + "\n" if len(ROWS) > 1 else "") + "relMSE vs own clean")
fig.legend(handles, ["declared path (content zeroed)", "declared path + content restored"],
           loc="lower center", ncol=2, frameon=False, handlelength=1.8,
           bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(pad=0.35, w_pad=1.1, rect=(0, 0.09, 1, 1))
out = os.path.join(HERE, "app_cap_full.pdf" if "--insample" in sys.argv
                   else "fig_cap.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
