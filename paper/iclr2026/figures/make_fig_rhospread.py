#!/usr/bin/env python
"""Does the Jacobian we measure predict the benchmark score movement a reviewer would see?"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
R = json.load(open(os.path.join(EXP, "s40_rho_vs_spread", "s40b_summary.json")))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})

FAM = {"chronos-bolt": ("Chronos-Bolt (declared path discards)", "#c0392b", "s"),
       "chronos-t5": ("Chronos-T5 (discards)", "#e08a5d", "o"),
       "chronos-2": ("Chronos-2 (discards)", "#d35400", "D"),
       "timesfm-2.5": ("TimesFM (overwrites)", "#7ba7d1", "v"),
       "tirex-1.1": ("TiRex ((a)+flag)", "#16a085", "P"),
       "flowstate-r1": ("FlowState ((a)+flag)", "#27ae60", "X"),
       "moirai-2.0": ("Moirai 2.0 (declared path retains)", "#2c6fad", "^")}


def fam(name):
    if name.startswith("chronos-bolt"):
        return "chronos-bolt"
    if name.startswith("chronos-t5"):
        return "chronos-t5"
    if name.startswith("chronos-2"):
        return "chronos-2"
    if name.startswith("timesfm"):
        return "timesfm-2.5"
    if name.startswith("tirex"):
        return "tirex-1.1"
    if name.startswith("flowstate"):
        return "flowstate-r1"
    return "moirai-2.0"


fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
per_ax = []
for ax, (xk, xlab, yk, ylab, title) in zip(axes, [
        ("gap", r"$\rho_{\mathrm{plain}} - \rho_{\mathrm{declared}}$", "spread_kdd",
         "score spread (%)", "(a)"),
        ("rho_plain", r"$\rho$ on the plain path", "spread_median",
         "score spread (%)", "(b)")]):
    xs, ys = [], []
    seen = set()
    pts = []
    for r in R:
        if r[xk] is None or r[yk] is None:
            continue
        f = fam(r["name"])
        lab, c, mk = FAM[f]
        ax.scatter(r[xk], r[yk] * 100, s=34, color=c, marker=mk, zorder=3,
                   label=lab if f not in seen else None)
        seen.add(f)
        xs.append(r[xk]); ys.append(r[yk] * 100)
        pts.append(r)
    sr = stats.spearmanr(xs, ys)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=8)
    ax.text(.03, .95, f"Spearman {sr.statistic:+.2f}\n($p={sr.pvalue:.3f}$)",
            transform=ax.transAxes, va="top", fontsize=6.6, color="0.25")
    ax.margins(x=.16, y=.22)
    per_ax.append((xk, yk, pts))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=6.4,
           bbox_to_anchor=(.5, -.03), handletextpad=.3)
fig.tight_layout(pad=.4, w_pad=1.6)
fig.canvas.draw()
# labels go on after the layout is final, each nudged to a free slot around its point.
# transData is in display pixels while annotate offsets are in points; work in points.
pt = 72.0 / fig.dpi
for ax, (xk, yk, pts) in zip(axes, per_ax):
    xy = [ax.transData.transform((r[xk], r[yk] * 100)) * pt for r in pts]
    placed = [(px - 3.5, py - 3.5, px + 3.5, py + 3.5) for px, py in xy]  # the markers
    for r, (x0, y0) in zip(pts, xy):
        short = (r["name"].replace("chronos-bolt-", "bolt-")
                 .replace("moirai-1.1-R-", "moirai-").replace("moirai-moe-1.0-R-", "moe-").replace("chronos-t5-", "t5-"))
        w, hgt = len(short) * 3.6 + 2, 7.5
        for dx, dy, ha in ((4, 2.5, "left"), (4, -9.5, "left"), (-4, 2.5, "right"),
                           (-4, -9.5, "right"), (4, 10.5, "left"), (-4, 10.5, "right"),
                           (0, 9.0, "center"), (0, -14.0, "center"),
                           (4, -17.5, "left"), (-4, -17.5, "right")):
            bx = x0 + dx if ha == "left" else (x0 + dx - w if ha == "right" else x0 + dx - w / 2)
            by = y0 + dy
            if all(bx + w < b[0] or bx > b[2] or by + hgt < b[1] or by > b[3]
                   for b in placed):
                break
        else:
            dx, dy, ha, bx, by = 4, 2.5, "left", x0 + 4, y0 + 2.5
        ax.annotate(short, (r[xk], r[yk] * 100), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=5.6, color="0.3")
        placed.append((bx, by, bx + w, by + hgt))
out = os.path.join(HERE, "app_rhospread.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
