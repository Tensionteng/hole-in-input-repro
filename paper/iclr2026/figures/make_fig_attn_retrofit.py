#!/usr/bin/env python
"""Figure: attention share received by masked patches, vanilla vs CPT retrofit (S512).

Reads experiments/s45_pretrain/s512_attn.json (produced by eval_s512_attn.py): encoder
self-attention share received by patches containing masked points, divided by the uniform
share, per window; median over windows, for the vanilla model (P0) and the CPT retrofit (P5) at the two
ends of the fill-quality axis (alpha=0 linear fill, alpha=1 oracle), on ETTh1/ETTm1/weather.
Writes figures/fig_attn.pdf (used in the retrofit appendix, "Mechanism-level evidence").
NB: make_fig_attn.py is a different figure (app_attn.pdf, the S7 attribution panels).
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
D = json.load(open(os.path.join(EXP, "s45_pretrain", "s512_attn.json")))

DS = ("ETTh1", "ETTm1", "weather")
ARMS = [("P0", "vanilla", ("#a9c8e8", "#2c6fad")),
        ("P5", "CPT retrofit", ("#f2b27c", "#c1502e"))]
AL = ("0.0", "1.0")
FILL = {"0.0": "poor fill", "1.0": "perfect fill"}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})

fig, ax = plt.subplots(figsize=(3.6, 1.95))
w = 0.2
x = np.arange(len(DS))
for i, (arm, lab, (c0, c1)) in enumerate(ARMS):
    for j, a in enumerate(AL):
        y = [D[f"{d}|a={a}|{arm}"]["median_share_ratio"] for d in DS]
        off = (i * 2 + j - 1.5) * w
        ax.bar(x + off, y, w * 0.92, color=(c0 if a == "0.0" else c1),
               label=f"{lab}, {FILL[a]}")
# mark the alpha=0 -> 1 movement above each arm's pair
for i, (arm, _, _) in enumerate(ARMS):
    y0 = [D[f"{d}|a=0.0|{arm}"]["median_share_ratio"] for d in DS]
    y1 = [D[f"{d}|a=1.0|{arm}"]["median_share_ratio"] for d in DS]
    for k in range(len(DS)):
        if abs(y1[k] - y0[k]) > 5e-3:
            ax.annotate("", xy=(x[k] + (i * 2 + 1 - 1.5) * w, y1[k] + 0.004),
                        xytext=(x[k] + (i * 2 - 1.5) * w, y0[k] + 0.004),
                        arrowprops=dict(arrowstyle="->", color="#a32b20", lw=0.9))
ax.set_xticks(x)
ax.set_xticklabels(DS)
ax.set_ylabel("attention share of masked patches\n(uniform $=1.0$)")
ax.set_ylim(0.70, 0.95)
ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.2,
          columnspacing=0.9, bbox_to_anchor=(0.0, 1.13))
fig.tight_layout(pad=0.35)
out = os.path.join(HERE, "fig_attn.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
for d in DS:
    print(f"  {d}: vanilla {D[f'{d}|a=0.0|P0']['median_share_ratio']:.4f} -> "
          f"{D[f'{d}|a=1.0|P0']['median_share_ratio']:.4f}, CPT retrofit "
          f"{D[f'{d}|a=0.0|P5']['median_share_ratio']:.4f} -> "
          f"{D[f'{d}|a=1.0|P5']['median_share_ratio']:.4f}")
