#!/usr/bin/env python
"""Probe follow-ups: the information gradient, and the reconstruction/forecast dissociation."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
P11 = json.load(open(os.path.join(EXP, "s11_probe", "s11_results.json")))["probe"]
S12 = json.load(open(os.path.join(EXP, "s12_recon_sft", "s12_results.json")))["tables"]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
                     "legend.fontsize": 7.0, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.62))

# (a) information gradient: peak probe R^2 by target family
ax = axes[0]
cells = [("mcar:0.7", "scattered 70%"), ("block:0.7", "block 70%"),
         ("mnar_high:0.7", "censoring 70%")]
fams = [("obs_in_miss", "observed values", "#2c6fad"),
        ("part_miss", "partly-missing patches", "#5aa0d0"),
        ("full_miss", "fully-missing patches", "#c0392b")]
x = np.arange(len(cells))
for j, (f, lab, col) in enumerate(fams):
    y = []
    for c, _ in cells:
        vs = []
        for d in P11:
            fv = P11[d].get(c, {}).get("families", {}).get(f)
            if fv:
                r = np.asarray(fv["r2_mean"], dtype=float)
                if np.isfinite(r).any():
                    vs.append(np.nanmax(r))
        y.append(np.mean(vs) if vs else np.nan)
    ax.bar(x + (j - 1) * .27, np.nan_to_num(y), .27, color=col, label=lab)
    for xi, v in zip(x, y):
        if not np.isfinite(v):
            ax.text(xi + (j - 1) * .27, .03, "n/a", ha="center", fontsize=6,
                    color="0.45", rotation=90)
ax.axhline(0, color="k", lw=.7)
ax.set_xticks(x); ax.set_xticklabels([l for _, l in cells], fontsize=7)
ax.set_ylabel(r"peak probe $R^2$"); ax.set_ylim(0, 1.05)
ax.set_title("(a)", fontsize=8, pad=20)
ax.legend(frameon=False, fontsize=6.6, ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.005), columnspacing=1.1, handlelength=1.2,
          handletextpad=.5)

# (b) dissociation: forecast vs reconstruction across fine-tuning arms
ax = axes[1]
probe, main = S12["probe"], S12["main"]
arms = [("zs", "zs:nan", "zero-shot", "#8c8c8c", "o"),
        ("mb", "mb:nan", "masked-block SFT", "#2c6fad", "s"),
        ("recon", "recon:nan", "reconstruction-aware SFT", "#c0392b", "^")]
for key, marm, lab, col, mk in arms:
    xs, ys = [], []
    for pk, pv in probe.items():
        ds, mech, rate = pk.split(":")
        cell = f"{mech}:{rate}"
        if key in pv and cell in main and marm in main[cell]:
            xs.append(pv[key]); ys.append(main[cell][marm][ds])
    if xs:
        ax.scatter(xs, ys, s=26, color=col, marker=mk, label=lab, zorder=3, alpha=.9)
ax.set_xlabel(r"probe $R^2$ (reconstruction quality)")
ax.set_ylabel("forecast relMSE")
ax.set_title("(b)", fontsize=8, pad=20)
ax.legend(frameon=False, fontsize=6.6, ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.005), columnspacing=1.0, handlelength=1.0,
          handletextpad=.4)

fig.tight_layout(pad=.4, w_pad=1.7)
out = os.path.join(HERE, "app_probe2.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
