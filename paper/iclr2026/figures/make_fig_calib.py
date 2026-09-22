#!/usr/bin/env python
"""Calibration: the one repair that transfers, and the corpus where it only half transfers.

(a) nine datasets, synthetic mechanisms, detector fitted on three of them and frozen (S38)
(b) two real deployments (S16)
(c) real GIFT-Eval missingness, by the window's own missing fraction (S39)
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
S16 = json.load(open(os.path.join(EXP, "s16_mechgate", "s16_results.json")))["trackB"]
S38 = {}
for f in ("gate.json", "heldout_a.json", "heldout_b.json"):
    p = os.path.join(EXP, "s38_calibbreadth", f)
    if os.path.exists(p):
        S38.update(json.load(open(p))["datasets"])
S39 = json.load(open(os.path.join(EXP, "s39_realcalib", "s39_results.json")))

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
                     "legend.fontsize": 7.2, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})
C_NAT, C_POOL, C_MOND = "#8c8c8c", "#e08a5d", "#2c6fad"
VAR = [("native", "model's own interval", C_NAT),
       ("naive", "conformal, single pool", C_POOL),
       ("mondrian_pred", "conformal, grouped by detected mechanism", C_MOND)]

# Two renderings: the main text needs the claim (synthetic breadth + the two real
# deployments); the GIFT-Eval residual is a caveat and travels with its appendix.
import sys
GIFT_PANEL = "--with-gift" in sys.argv
n = 3 if GIFT_PANEL else 2
fig, axes = plt.subplots(1, n, figsize=(7.1 if GIFT_PANEL else 5.4, 2.5),
                         gridspec_kw={"width_ratios": ([1.25, .95, 1.05] if GIFT_PANEL
                                                       else [1.25, .95])})
handles = []

# (a) nine datasets, synthetic mechanisms -----------------------------------------------
cls = ["clean", "mcar", "block", "mnar_high", "mnar_extreme"]
lab = ["clean", "scattered", "block", "censoring", "extreme"]
x = np.arange(len(cls))
for j, (v, l, c) in enumerate(VAR):
    ys = [[S38[d]["groups"][k][v]["coverage"] for d in S38 if k in S38[d]["groups"]]
          for k in cls]
    m = [np.mean(y) for y in ys]
    err = np.array([[np.mean(y) - np.min(y) for y in ys],
                    [np.max(y) - np.mean(y) for y in ys]])
    b = axes[0].bar(x + (j - 1) * .27, m, .27, color=c,
                    yerr=err, error_kw=dict(lw=.7, ecolor="0.3", capsize=1.6))
    handles.append(b)
axes[0].axhline(.9, color="k", ls="--", lw=.9)
axes[0].text(4.55, .925, "target 90%", fontsize=6.4, color="0.35", ha="right")
axes[0].set_xticks(x); axes[0].set_xticklabels(lab, fontsize=7, rotation=18, ha="right")
axes[0].set_ylim(.1, 1.02); axes[0].set_ylabel("empirical coverage")
axes[0].set_title("(a)", fontsize=8)

# (b) two real deployments ---------------------------------------------------------------
cr = S16["conf_real"]
cells = [("penn", "linear", "cens_test", "Penmanshiel\ncurtailment"),
         ("metrla", "linear", "miss_test", "METR-LA\noutages"),
         ("metrla", "nan", "miss_test", "METR-LA\noutages (NaN)")]
x = np.arange(len(cells))
for j, (v, l, c) in enumerate(VAR):
    axes[1].bar(x + (j - 1) * .27, [cr[d][f][g][v]["coverage"] for d, f, g, _ in cells],
                .27, color=c)
axes[1].axhline(.9, color="k", ls="--", lw=.9)
axes[1].set_xticks(x); axes[1].set_xticklabels([t.replace("\n", " ") for *_, t in cells], fontsize=6.2,
                        rotation=18, ha="right")
axes[1].set_ylim(.1, 1.02)
axes[1].set_title("(b)", fontsize=8)

# (c) real GIFT-Eval missingness ----------------------------------------------------------
if GIFT_PANEL:
 ST = S39["strata"]
 keys = list(ST)
 labs = [f"{float(k.split('-')[0])*100:.0f}--{min(float(k.split('-')[1]),1.0)*100:.0f}%"
         for k in keys]
 x = np.arange(len(keys))
 for v, c, mk, l in (("native", C_NAT, "o", "model's own interval"),
                     ("naive", C_POOL, "s", "conformal, single pool"),
                     ("mondrian", C_MOND, "^", "grouped by detected mechanism"),
                     ("mondrian_rate", "#1a4c7c", "D", "grouped by mechanism $\\times$ rate")):
     if v not in ST[keys[0]]:
         continue
     axes[2].plot(x, [ST[k][v]["coverage"] for k in keys], "-" + mk, color=c, lw=1.4,
                  ms=3.6, label=l)
 axes[2].axhline(.9, color="k", ls="--", lw=.9)
 axes[2].set_xticks(x); axes[2].set_xticklabels(labs, fontsize=6.4, rotation=18, ha="right")
 # (c) lives in a narrow band around the target; a shared axis with (a) and (b) would hide
 # the very differences the panel exists to show.
 axes[2].set_ylim(.66, .99)
 axes[2].set_xlabel("real missing fraction of the context", fontsize=7)
 axes[2].set_title("(c)", fontsize=8)
 axes[2].legend(frameon=False, fontsize=5.9, loc="lower left", handlelength=1.3,
                handletextpad=.4, borderpad=.1, labelspacing=.25, ncol=1)
 axes[2].set_ylabel("empirical coverage", fontsize=7.4)

fig.legend(handles, [l for _, l, _ in VAR], loc="lower center", ncol=3, frameon=False,
           fontsize=7, handlelength=1.4, bbox_to_anchor=(.5, -.075))
fig.tight_layout(pad=.4, w_pad=1.3)
out = os.path.join(HERE, "app_calib_gift.pdf" if GIFT_PANEL else "fig_calib.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
