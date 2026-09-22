#!/usr/bin/env python
"""S26 figure: does the recoverable fill gap survive on real missingness?"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "s26_results.json")))["arms"]

fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

# (a) paired ratio distributions
ax = axes[0]
specs = [("penn|cens|aug0|c-zero", "plain_fill", "zero", "Penmanshiel\n(real censoring)", "#c0392b"),
         ("metr|miss|aug0|c-linear", "plain_fill", "nan", "METR-LA\n(real outages)", "#3b6ea5")]
data, labs, cols = [], [], []
for key, cv, base, lab, col in specs:
    c = r[key]
    e = np.array(c[f"learned_{cv}"]["per_window"])
    b = np.array(c[f"fixed_{base}"]["per_window"])
    data.append(np.clip(e / np.maximum(b, 1e-12), 0.2, 5.0)); labs.append(lab); cols.append(col)
bp = ax.boxplot(data, tick_labels=labs, showfliers=False, patch_artist=True, widths=0.5)
for p, c in zip(bp["boxes"], cols):
    p.set_facecolor(c); p.set_alpha(0.55)
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_yscale("log"); ax.set_ylabel("paired NMSE ratio  learned fill / best fixed fill")
ax.set_title("(a) Learned fill vs best fixed fill,\nwindow by window", fontsize=10)
for i, (d, key) in enumerate(zip(data, [s[0] for s in specs])):
    ax.text(i + 1, 0.23, f"win {100*(d<1).mean():.0f}%", ha="center", fontsize=9,
            fontweight="bold")

# (b) win-rate + significance
ax = axes[1]
rows = [("Penmanshiel\nplain fill", "penn|cens|aug0|c-zero", "plain_fill", "zero", 1.7e-5),
        ("Penmanshiel\nfill+mask", "penn|cens|aug0|c-zero", "fill_mask", "zero", 0.86),
        ("METR-LA\nplain fill", "metr|miss|aug0|c-linear", "plain_fill", "nan", 1.00),
        ("METR-LA\nfill+mask", "metr|miss|aug0|c-linear", "fill_mask", "nan", 0.10)]
w = []
for _, key, cv, base, _p in rows:
    c = r[key]
    e = np.array(c[f"learned_{cv}"]["per_window"]); b = np.array(c[f"fixed_{base}"]["per_window"])
    w.append(100 * (e < b).mean())
cols = ["#c0392b", "#e08a5d", "#3b6ea5", "#7ba7d1"]
ax.bar(range(4), w, 0.6, color=cols)
ax.axhline(50, color="k", ls="--", lw=1)
for i, (lab, _, _, _, p) in enumerate(rows):
    ax.text(i, w[i] + 1.2, f"p={p:.0e}" if p < 0.01 else f"p={p:.2f}", ha="center", fontsize=8.5)
ax.set_xticks(range(4)); ax.set_xticklabels([x[0] for x in rows], fontsize=8)
ax.set_ylabel("% of windows where the learned fill wins"); ax.set_ylim(0, 72)
ax.set_title("(b) Mechanism-specific, as predicted:\nsignificant only under censoring", fontsize=10)

# (c) synthetic vs real magnitude
ax = axes[2]
labels = ["synthetic\nmnar_high p=0.7\n(S25)", "real curtailment\n(S26)"]
excess_fixed = [1.875, np.median(np.array(r["penn|cens|aug0|c-zero"]["fixed_zero"]["per_window"])) - 1]
e = np.array(r["penn|cens|aug0|c-zero"]["learned_plain_fill"]["per_window"])
b = np.array(r["penn|cens|aug0|c-zero"]["fixed_zero"]["per_window"])
recov = [64.0, 100 * (1 - np.median(e / b))]
ax.bar([0, 1], recov, 0.5, color=["#8c8c8c", "#c0392b"])
for i, v in enumerate(recov):
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=12, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("% of the gap recovered by a learned fill"); ax.set_ylim(0, 75)
ax.set_title("(c) Direction replicates, magnitude does not", fontsize=10)

fig.suptitle("S26 — does S25's recoverable fill gap survive on REAL missingness? "
             "(chronos-bolt-base, frozen; held-out half)", fontsize=11.5)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "s26.png"), dpi=150)
print("figure ->", os.path.join(HERE, "s26.png"))
