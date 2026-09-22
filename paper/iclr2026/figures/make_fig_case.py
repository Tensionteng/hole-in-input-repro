#!/usr/bin/env python
"""Case-study figures: one window's forecast fan against the realized truth.

Reads experiments/s68_casefig/s68_cases.json (run_s68_casefig.py) and writes
  fig_case.pdf  main text, 2 panels: same dataset (meta.selection.main_dataset),
                same window, (a) value censoring, (b) scattered dropout
  app_case.pdf  appendix, 3 x 2 grid: ETTh1 / weather / electricity x the two
                mechanisms, each at the window index chosen by the mechanical
                selection rule recorded in the JSON.

Every panel: context region with missing positions shaded and the withheld truth
dotted, then the horizon with the realized future (black), the declared-path
median and 80% band (blue), the clean-context 80% band (dashed grey) and red
dots where the truth exits the declared band.
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
R = json.load(open(os.path.join(EXP, "s68_casefig", "s68_cases.json")))
L, H = R["meta"]["protocol"]["L"], R["meta"]["protocol"]["H"]
CHOSEN = R["meta"]["selection"]["chosen"]
MAIN_DS = R["meta"]["selection"]["main_dataset"]
STATS = R["stats"]

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7.2,
                     "ytick.labelsize": 7.2, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42,
                     "figure.dpi": 200})

BLUE, RED = "#2c6fad", "#c0392b"
MECH_TITLE = {"mnar_high": "value censoring", "mcar": "scattered dropout"}
TAIL = 128          # only the last TAIL of the 512 context steps are drawn


def runs(mask):
    """Contiguous True runs of a 1-D bool array, as (start, stop) index pairs."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return zip(idx[0::2], idx[1::2])


def panel(ax, ds, mech, title=None, ylabel=None, xlabel=False, stats_text=True):
    c = R["cases"][ds][mech][str(CHOSEN[ds])]
    ctx = np.asarray(c["context"])[-TAIL:]
    mask = np.asarray(c["mask"], bool)[-TAIL:]
    filled = np.asarray(c["filled"])[-TAIL:]
    truth = np.asarray(c["truth"])
    qc = np.asarray(c["q_clean"])          # [9, H]
    qd = np.asarray(c["q_decl"])
    tc = np.arange(L - TAIL, L)
    th = L + np.arange(H)

    for a, b in runs(mask):                # missing positions, shaded
        ax.axvspan(tc[0] + a - 0.5, tc[0] + b - 0.5, color=RED, alpha=0.10,
                   lw=0, zorder=0)
    ax.plot(tc, ctx, color="0.72", lw=0.7, zorder=1)          # full true context
    ax.plot(tc, filled, color="0.25", lw=0.9, zorder=2)       # as fed (fill)
    withheld = np.where(mask, ctx, np.nan)
    ax.plot(tc, withheld, color=RED, lw=0.9, ls=":", zorder=3)

    ax.fill_between(th, qd[0], qd[8], color=BLUE, alpha=0.22, lw=0, zorder=3)
    ax.plot(th, qc[0], color="0.45", lw=0.8, ls="--", zorder=4)
    ax.plot(th, qc[8], color="0.45", lw=0.8, ls="--", zorder=4)
    ax.plot(th, qd[4], color=BLUE, lw=1.2, zorder=5)
    ax.plot(th, truth, color="k", lw=1.2, zorder=6)
    out = (truth < qd[0]) | (truth > qd[8])
    ax.plot(th[out], truth[out], "o", color=RED, ms=3.2, lw=0, zorder=7)

    ax.axvline(L - 0.5, color="0.4", lw=0.7, ls="-", zorder=2)
    ax.set_xlim(tc[0] - 1, L + H + 1)
    ax.set_xticks([L - TAIL, L - TAIL // 2, L, L + H])
    wi = CHOSEN[ds]
    cov = STATS[ds][mech]["coverage"][wi]
    wrel = STATS[ds][mech]["width_rel_clean"][wi]
    if stats_text:
        ax.text(0.03, 0.97,
                f"80% band: covers {cov:.2f},\nwidth {wrel:.2f}$\\times$ clean",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.6,
                color="0.2", bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
    if title:
        ax.set_title(title, pad=3)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("time step")
    return cov, wrel, int(out.sum())


HANDLES = [
    Line2D([], [], color="k", lw=1.2, label="realized future"),
    Line2D([], [], color=BLUE, lw=1.2, label="median, declared path"),
    Patch(fc=BLUE, alpha=0.22, lw=0, label="80% band, declared path"),
    Line2D([], [], color="0.45", lw=0.8, ls="--", label="80% band, clean context"),
    Line2D([], [], color="0.25", lw=0.9, label="context as fed (linear fill)"),
    Line2D([], [], color=RED, lw=0.9, ls=":", label="withheld truth (missing, shaded)"),
]

# ------------------------------------------------------------- main figure ----
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.15))
for ax, mech, tag in zip(axes, ("mnar_high", "mcar"), ("(a)", "(b)")):
    cov, wrel, nout = panel(
        ax, MAIN_DS, mech,
        title=f"{tag} {MECH_TITLE[mech]}, 70% missing",
        ylabel=f"{MAIN_DS}, channel 0" if mech == "mnar_high" else None,
        xlabel=True)
    print(f"fig_case {tag} {mech:9s} {MAIN_DS} wi={CHOSEN[MAIN_DS]} "
          f"cov={cov:.3f} width={wrel:.3f}x clean exits={nout}/{H}")
fig.legend(handles=HANDLES, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.06), handlelength=1.7, columnspacing=1.2)
fig.tight_layout(pad=0.4, w_pad=1.2, rect=(0, 0, 1, 0.90))
out = os.path.join(HERE, "fig_case.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)

# ---------------------------------------------------------- appendix grid ----
DS3 = ("ETTh1", "weather", "electricity")
fig, axes = plt.subplots(3, 2, figsize=(7.0, 5.9))
for i, ds in enumerate(DS3):
    for j, mech in enumerate(("mnar_high", "mcar")):
        ax = axes[i, j]
        cov, wrel, nout = panel(
            ax, ds, mech,
            title=(f"{MECH_TITLE[mech]}, 70% missing" if i == 0 else None),
            ylabel=f"{ds}\nchannel 0" if j == 0 else None,
            xlabel=(i == 2))
        print(f"app_case {ds:11s} {mech:9s} wi={CHOSEN[ds]} "
              f"cov={cov:.3f} width={wrel:.3f}x clean exits={nout}/{H}")
fig.legend(handles=HANDLES, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.03), handlelength=1.7, columnspacing=1.2)
fig.tight_layout(pad=0.5, w_pad=1.2, h_pad=0.9, rect=(0, 0, 1, 0.955))
out = os.path.join(HERE, "app_case.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
