#!/usr/bin/env python
"""Appendix: the interface fix across nine datasets, three of which the adapter saw."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
R = {}
for f in ("own_insample.json", "own_held_a.json", "own_held_b.json", "own_held_c.json"):
    p = os.path.join(EXP, "s37_capbreadth", f)
    if os.path.exists(p):
        d = json.load(open(p))
        for k in ("closure", "clean_own"):
            R.setdefault(k, {}).update(d.get(k, {}))
CC = json.load(open(os.path.join(EXP, "s37_capbreadth", "clean_control.json")))

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})

DS = ["ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
      "exchange", "illness"]
INS = ("ETTh1", "ETTm1", "weather")
MECH = [("mcar", "scat."), ("block", "block"), ("mnar_high", "cens."),
        ("mnar_extreme", "extr.")]
RATES = (0.3, 0.7)
MIN_EXCESS = 0.02

cols, collab = [], []
for m, ml in MECH:
    for r in RATES:
        cols.append((m, r)); collab.append(f"{ml}\n{r}")
M = np.full((len(DS), len(cols)), np.nan)
for i, ds in enumerate(DS):
    for j, (m, r) in enumerate(cols):
        c = R["closure"].get(f"{ds}|{m}|{r}")
        if c and c["native_a1"] - 1.0 >= MIN_EXCESS:
            M[i, j] = np.clip(c["closure"], -1.0, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85),
                         gridspec_kw={"width_ratios": [1.55, 1]})
ax = axes[0]
im = ax.imshow(M, cmap="BrBG", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), aspect="auto")
for i in range(len(DS)):
    for j in range(len(cols)):
        v = M[i, j]
        ax.text(j, i, "n/a" if not np.isfinite(v) else f"{v*100:.0f}", ha="center",
                va="center", fontsize=5.9,
                color="0.45" if not np.isfinite(v) else ("white" if abs(v) > .62 else "0.1"))
ax.axhline(2.5, color="k", lw=1.4)
ax.set_xticks(range(len(cols))); ax.set_xticklabels(collab, fontsize=6.2)
ax.set_yticks(range(len(DS)))
ax.set_yticklabels([d + (" *" if d in INS else "") for d in DS], fontsize=6.8)
ax.set_title("(a)", fontsize=8)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
cb.ax.tick_params(labelsize=6); cb.set_ticks([-1, 0, 1])
cb.set_ticklabels(["$-100$", "0", "100"])

# (b) the control: what the adaptation does on a clean context
ax = axes[1]
x = np.arange(len(DS))
for j, (k, lab, c) in enumerate((("native", "declared path", "#c0392b"),
                                 ("dual", "content restored", "#2c6fad"))):
    ax.bar(x + (j - .5) * .38, [CC[d]["clean_relMSE"][k] - 1 for d in DS], .38,
           color=c, label=lab)
ax.axhline(0, color="k", lw=.8)
ax.set_xticks(x)
ax.set_xticklabels([d + (" *" if d in INS else "") for d in DS], rotation=38, ha="right",
                   fontsize=6.2)
ax.set_ylabel("clean-context relMSE $-\\,1$", fontsize=7.4)
ax.set_title("(b)", fontsize=8, pad=20)
ax.legend(frameon=False, fontsize=6.4, ncol=2, loc="lower center",
          bbox_to_anchor=(.5, 1.005), handlelength=1.2, columnspacing=1.2)
fig.text(.005, .015, "* one of the three datasets the input-embedding adapter was trained on",
         fontsize=6, color="0.35")
fig.tight_layout(pad=.4, w_pad=1.5, rect=(0, .045, 1, 1))
out = os.path.join(HERE, "app_capbreadth.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
