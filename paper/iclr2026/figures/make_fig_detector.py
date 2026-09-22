#!/usr/bin/env python
"""Detector: per-class recall, confusion, feature importance."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
A = json.load(open(os.path.join(EXP, "s16_mechgate", "s16_results.json")))["trackA"]
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.4,
                     "legend.fontsize": 7.0, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})
CLS = ["clean", "mcar", "block", "mnar_high", "mnar_extreme", "intermittent"]
LAB = ["clean", "scattered", "block", "censoring", "extreme", "intermittent"]
DS = ["ETTh1", "ETTm1", "weather"]

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.15),
                         gridspec_kw={"width_ratios": [1.05, 1.1, 1]})

# (a) per-class recall, gbdt vs single-statistic baseline
ax = axes[0]
x = np.arange(len(CLS))
for j, (mdl, lab, col) in enumerate([("gbdt", "geometry features", "#2c6fad"),
                                     ("little", "single statistic", "#c0392b")]):
    y = []
    for c in CLS:
        v = [A["lod"][d][mdl]["recall"].get(c, np.nan) for d in DS if mdl in A["lod"][d]]
        y.append(np.nanmean(v) if v else np.nan)
    ax.bar(x + (j - .5) * .38, y, .38, color=col, label=lab)
ax.set_xticks(x); ax.set_xticklabels(LAB, rotation=30, ha="right", fontsize=6.6)
ax.set_ylabel("recall (leave-one-dataset-out)"); ax.set_ylim(0, 1.30)
ax.set_title("(a)", fontsize=8)
# Two rows: the panel is only a third of the text width and a single row of these two
# labels runs past its right edge.
ax.legend(frameon=False, loc="upper center", fontsize=6.6, ncol=1,
          columnspacing=1.0, handlelength=1.2, handletextpad=.5, labelspacing=.25,
          borderpad=.1)

# (b) confusion matrix
ax = axes[1]
C = np.asarray(A["lod_confusion_gbdt"], dtype=float)
C = C / np.maximum(C.sum(1, keepdims=True), 1e-9)
im = ax.imshow(C, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(CLS))); ax.set_yticks(range(len(CLS)))
ax.set_xticklabels(LAB, rotation=40, ha="right", fontsize=6.2)
ax.set_yticklabels(LAB, fontsize=6.2)
for i in range(len(CLS)):
    for j in range(len(CLS)):
        if C[i, j] > .02:
            ax.text(j, i, f"{100*C[i,j]:.0f}", ha="center", va="center", fontsize=5.8,
                    color="white" if C[i, j] > .55 else "0.2")
ax.set_ylabel("true"); ax.set_xlabel("predicted")
ax.set_title("(b)", fontsize=8)

# (c) feature importance
ax = axes[2]
fi = A["gbdt_feat_importance"]
it = sorted(fi.items(), key=lambda kv: kv[1])[-7:]
ax.barh(range(len(it)), [v for _, v in it], color="#5aa0d0", height=.7)
ax.set_yticks(range(len(it)))
ax.set_yticklabels([k.replace("_", " ") for k, _ in it], fontsize=6.4)
ax.set_xlabel("permutation importance")
ax.set_title("(c)", fontsize=8)

fig.tight_layout(pad=.4, w_pad=1.5)
out = os.path.join(HERE, "app_detector.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
