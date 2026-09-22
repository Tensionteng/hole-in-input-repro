#!/usr/bin/env python
"""Attribution figures: channel decomposition, probe curves, knockout."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
J = lambda *p: json.load(open(os.path.join(EXP, *p)))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
                     "legend.fontsize": 7.0, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})

P11 = J("s11_probe", "s11_results.json")["probe"]
P13 = J("s13_knockout", "s13_results.json")

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.72))

# (a) probe R^2 by depth, several mechanisms
ax = axes[0]
cells = [("ETTh1", "block:0.3", "block 30%", "#2c6fad"),
         ("ETTh1", "block:0.7", "block 70%", "#7ba7d1"),
         ("ETTh1", "mcar:0.7", "scattered 70%", "#5aa0d0"),
         ("ETTh1", "mnar_high:0.7", "censoring 70%", "#c0392b")]
for ds, cell, lab, col in cells:
    try:
        y = P11[ds][cell]["families"]["full_miss"]["r2_mean"]
    except KeyError:
        continue
    ax.plot(range(len(y)), y, "-o", color=col, lw=1.5, ms=2.6, label=lab)
ax.axhline(0, color="k", ls=":", lw=.7)
ax.set_xlabel("encoder layer"); ax.set_ylabel(r"probe $R^2$ for the missing values")
ax.set_title("(a)", fontsize=8, pad=31)
ax.legend(frameon=False, handlelength=1.2, fontsize=6.2, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.005), columnspacing=1.0, handletextpad=.4)

# (b) knockout
ax = axes[1]
segs = [("none", "no knockout", "#2c6fad", "-"), ("l0-3", "layers 0–3", "#e08a5d", "-"),
        ("l4-7", "layers 4–7", "#8c8c8c", "-"), ("all", "all layers", "#c0392b", "-")]
for seg, lab, col, ls in segs:
    try:
        y = P13["probe_ko"]["ETTh1"][seg]["block:0.3"]["r2_mean"]
    except KeyError:
        continue
    ax.plot(range(len(y)), y, ls, marker="o", color=col, lw=1.5, ms=2.6, label=lab)
ax.axhline(0, color="k", ls=":", lw=.7)
ax.set_xlabel("encoder layer"); ax.set_ylabel(r"probe $R^2$")
ax.set_title("(b)", fontsize=8, pad=31)
ax.legend(frameon=False, handlelength=1.2, fontsize=6.2, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.005), columnspacing=1.0, handletextpad=.4)

# (c) but it changes nothing downstream
ax = axes[2]
rk = P13["relmse_ko"]["ETTh1"]
order = ["none", "l0-3", "l4-7", "l8-11", "all"]
vals = []
for s in order:
    vals.append(rk.get(s, {}).get("block:0.7", np.nan))
ax.bar(range(len(order)), vals, .6, color=["#2c6fad"] + ["#c0392b"] * 4)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(["none", "l0–3", "l4–7", "l8–11", "all"], fontsize=7)
lo = np.nanmin(vals)
ax.set_ylim(lo * 0.97, lo * 1.06)
ax.set_ylabel("forecast relMSE")
ax.set_title("(c)", fontsize=8)
ax.text(.5, .88, "all five values identical\nto float64 precision", transform=ax.transAxes,
        ha="center", fontsize=6.8, color="0.3")

fig.tight_layout(pad=.4, w_pad=1.6)
out = os.path.join(HERE, "app_probe.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
