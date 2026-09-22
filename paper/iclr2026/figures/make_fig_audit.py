#!/usr/bin/env python
"""Figure: score sensitivity to the fill convention, as a dose-response in missingness."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
R = json.load(open(os.path.join(EXP, "s32_gifteval", "s32c_results.json")))
B, POOL = R["bins"], R["pool"]

BINS = ["0.00-0.01", "0.01-0.05", "0.05-0.15", "0.15-0.30", "0.30-1.01"]
LAB = ["0–1%", "1–5%", "5–15%", "15–30%", ">30%"]
MODELS = [("bolt-base", "Chronos-Bolt", "#c0392b", "s"),
          ("chronos2", "Chronos-2", "#e08a5d", "o"),
          ("moirai2", "Moirai 2.0", "#2c6fad", "^"),
          ("timesfm", "TimesFM", "#7ba7d1", "v")]

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})

# One figure: (a) the dose-response, (b) the same stratum under a second window protocol.
fig, axes = plt.subplots(1, 2,
                         figsize=(7.1, 2.05),
                         gridspec_kw={"width_ratios": [1.35, 1]},
                         squeeze=False)
axes = axes[0]

ax = axes[0]
x = np.arange(len(BINS))
for key, lab, col, mk in MODELS:
    y = [B.get(f"{key}|random|{b}", {}).get("spread_pct", np.nan) for b in BINS]
    ax.plot(x, y, marker=mk, color=col, lw=1.6, ms=4, label=lab)
ax.axhline(0, color="k", lw=0.7, ls=":")
ns = [B.get(f"bolt-base|random|{b}", {}).get("n", 0) for b in BINS]
ax.set_xticks(x)
ax.set_xticklabels([f"{l}\n$n$={n}" for l, n in zip(LAB, ns)], fontsize=7)
ax.set_xlabel("fraction of the context that is missing")
ax.set_ylabel("score change from the\nfill convention alone (%)")
ax.legend(frameon=False, loc="upper left", handlelength=1.6)
ax = axes[1]
prot = [("random", "random\nwindows"), ("last", "last window\n/ series")]
# two protocol groups on the x-axis; bars keep the model colours of the left panel
for j, (p, plab) in enumerate(prot):
    y = [B.get(f"{k}|{p}|0.30-1.01", {}).get("spread_pct", np.nan) for k, _, _, _ in MODELS]
    x = j * (len(MODELS) + 1) + np.arange(len(MODELS))
    ax.bar(x, y, 0.8, color=[m[2] for m in MODELS], edgecolor="none")
ax.set_xticks([j * (len(MODELS) + 1) + (len(MODELS) - 1) / 2 for j in range(2)])
ax.set_xticklabels([plab for _, plab in prot], fontsize=7)
ax.set_ylabel("score change (%)")

fig.tight_layout(pad=0.4, w_pad=1.4)
out = os.path.join(HERE, "fig_audit.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
