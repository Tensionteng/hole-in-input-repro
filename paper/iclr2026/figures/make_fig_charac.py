#!/usr/bin/env python
"""Characterisation figure: mechanism not rate, the silent failure, the real-data reversal."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
J = lambda *p: json.load(open(os.path.join(EXP, *p)))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
                     "legend.fontsize": 7.2, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})
DS = ("ETTh1", "ETTm1", "weather")
RATES = (0.1, 0.3, 0.5, 0.7)
MECHS = [("mcar", "scattered", "#2c6fad"), ("block", "block", "#5aa0d0"),
         ("mnar_high", "censoring", "#c0392b"), ("mnar_extreme", "extreme cens.", "#e08a5d")]

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15))

# (a) mechanism, not rate
R = J("s05_characterization", "s5_missing_results.json")["bolt"]
ax = axes[0]
for m, lab, col in MECHS:
    y = []
    for p in RATES:
        v = [R[d][f"{m}:linear:{p}"]["mse"] / R[d]["clean:none:0.0"]["mse"] for d in DS
             if f"{m}:linear:{p}" in R[d]]
        y.append(np.mean(v))
    ax.plot(RATES, y, "-o", color=col, lw=1.6, ms=3.6, label=lab)
ax.axhline(1, color="k", ls=":", lw=.7)
ax.set_xlabel("missingness rate"); ax.set_ylabel("relMSE vs clean")
ax.set_title("(a)", loc="left", fontsize=8)
ax.legend(frameon=False, handlelength=1.5, loc="upper left")

# (b) the silent failure, as a trajectory in (width, coverage)
C = J("s05_characterization", "s5_extra_results.json")["bolt_cal"]
ax = axes[1]
for m, lab, col in (("mcar", "scattered", "#2c6fad"), ("mnar_high", "censoring", "#c0392b")):
    cov, wid = [], []
    for p in (0.0,) + RATES:
        k = "clean:none:0.0" if p == 0.0 else f"{m}:linear:{p}"
        cov.append(np.mean([C[d][k]["coverage80"] for d in DS if k in C[d]]))
        wid.append(np.mean([C[d][k]["pi_width80"] / C[d]["clean:none:0.0"]["pi_width80"]
                            for d in DS if k in C[d]]))
    ax.plot(wid, cov, "-o", color=col, lw=1.7, ms=3.6, label=lab)
    ax.annotate("", xy=(wid[-1], cov[-1]), xytext=(wid[-2], cov[-2]),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.7))
    ax.annotate(f"{int(RATES[-1]*100)}%", (wid[-1], cov[-1]), textcoords="offset points",
                xytext=(4, -8), fontsize=6.6, color=col)
ax.axhline(0.8, color="k", ls=":", lw=.8)
ax.text(0.62, 0.812, "nominal 80%", fontsize=6.4, color="0.4")
ax.plot(wid[0], cov[0], "ks", ms=4.5, zorder=5)
ax.annotate("clean", (wid[0], cov[0]), textcoords="offset points", xytext=(-6, 6),
            fontsize=6.6, color="0.25")
ax.set_xlabel("interval width, relative to clean")
ax.set_ylabel("empirical coverage")
ax.set_title("(b)", loc="left", fontsize=8)
ax.legend(frameon=False, loc="lower right", handlelength=1.5)

# (c) the reversal on two real deployments
ax = axes[2]
penn = {"as-recorded": 40.29, "zero": 28.43, "linear": 60.16}
metr = {"as-recorded": 21.19, "zero": 21.19, "linear": 2.75}
fills = ["as-recorded", "zero", "linear"]
x = np.arange(3)
pn = [penn[f] / min(penn.values()) for f in fills]
mt = [metr[f] / min(metr.values()) for f in fills]
ax.bar(x - .18, pn, .36, color="#c0392b", label="curtailment (censoring)")
ax.bar(x + .18, mt, .36, color="#2c6fad", label="outages (blocks)")
ax.set_yscale("log"); ax.axhline(1, color="k", ls=":", lw=.7)
ax.set_xticks(x); ax.set_xticklabels(fills, fontsize=7)
ax.set_ylabel("NMSE / best fill in that domain", fontsize=7.4)
ax.set_ylim(0.8, 40)
ax.set_title("(c)", loc="left", fontsize=8)
ax.legend(frameon=False, fontsize=6.6, loc="upper center", bbox_to_anchor=(.5,1.0))

fig.tight_layout(pad=.4, w_pad=2.2)
out = os.path.join(HERE, "fig_charac.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
