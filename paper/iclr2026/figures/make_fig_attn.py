#!/usr/bin/env python
"""Attention statistics are unchanged by missingness: the damage does not travel here."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
A = json.load(open(os.path.join(EXP, "s07_component_attrib", "s7_attrib_results.json")))["attn"]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
                     "legend.fontsize": 7.0, "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})
DS = ["ETTh1", "ETTm1", "weather"]
CONDS = [("clean", None, "clean context", "#111111", "-"),
         ("mcar", "linear:0.7", "scattered 70%", "#2c6fad", "-"),
         ("block", "linear:0.7", "block 70%", "#5aa0d0", "-"),
         ("mnar_high", "linear:0.7", "censoring 70%", "#c0392b", "-")]

def get(field, mech, cell):
    out = []
    for d in DS:
        node = A[d][mech] if cell is None else A[d][mech].get(cell)
        if node is None or field not in node:
            continue
        out.append(np.asarray(node[field], dtype=float))
    return np.mean(out, axis=0) if out else None

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.62))

# (a) attention entropy per layer
ax = axes[0]
for mech, cell, lab, col, ls in CONDS:
    y = get("ent", mech, cell)
    if y is None: continue
    ax.plot(range(1, len(y) + 1), y, ls, marker="o", color=col, lw=1.5, ms=2.8, label=lab)
ax.set_xlabel("encoder layer"); ax.set_ylabel("attention entropy (nats)")
ax.set_title("(a)", fontsize=8, pad=31)
ax.legend(frameon=False, handlelength=1.2, fontsize=6.2, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.005), columnspacing=1.0, handletextpad=.4)

# (b) attention mass on the register / sink token
ax = axes[1]
for mech, cell, lab, col, ls in CONDS:
    y = get("mreg", mech, cell)
    if y is None: continue
    ax.plot(range(1, len(y) + 1), y, ls, marker="o", color=col, lw=1.5, ms=2.8, label=lab)
ax.set_xlabel("encoder layer"); ax.set_ylabel("mass on the register token")
ax.set_title("(b)", fontsize=8)

# (c) representation drift does grow, and it is not information
ax = axes[2]
D = json.load(open(os.path.join(EXP, "s07_component_attrib",
                                "s7_attrib_results.json")))["drift"]
for mech, lab, col in (("mcar", "scattered", "#2c6fad"), ("block", "block", "#5aa0d0"),
                       ("mnar_high", "censoring", "#c0392b")):
    ys = []
    for d in DS:
        cell = D[d][mech].get(f"{mech}:linear:0.7")
        if cell and "cos_mean_pw" in cell:
            ys.append(np.asarray(cell["cos_mean_pw"], dtype=float).mean(axis=0))
    if ys:
        y = np.mean(ys, axis=0)
        ax.plot(range(len(y)), 1 - y, "-o", color=col, lw=1.5, ms=2.8, label=lab)
ax.set_xlabel("encoder layer"); ax.set_ylabel(r"$1-\cos$ to the clean state")
ax.set_title("(c)", fontsize=8)
ax.legend(frameon=False, handlelength=1.4, fontsize=6.6, loc="upper left")

fig.tight_layout(pad=.4, w_pad=1.5)
out = os.path.join(HERE, "app_attn.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
