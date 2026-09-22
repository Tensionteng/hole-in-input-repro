#!/usr/bin/env python
"""S27 figure: the input interface caps how much any imputer can help."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
r = json.load(open(os.path.join(HERE, "s27_results.json")))["sweep"]
AL = (0.0, 0.25, 0.5, 0.75, 1.0)
DS = ("ETTh1", "ETTm1", "weather")
ME = ("mcar", "block", "mnar_high", "mnar_extreme")
STYLE = {"plain": ("#8c8c8c", "o", "plain fill (no flag)"),
         "native": ("#c0392b", "s", "native: content zeroed (Chronos-Bolt)"),
         "dual": ("#3b6ea5", "^", "dual: value + flag"),
         "dual_obsnorm": ("#5aa0d0", "v", "dual + observed-only stats")}

def c(tag, m, rt, i):
    return np.array([np.mean([r[f"{tag}|{d}|{m}|{rt}|{a}|{i}"]["median"] for d in DS]) for a in AL])

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for row, tag in enumerate(("zeroshot", "adapted")):
    for col, m in enumerate(ME):
        ax = axes[row, col]
        for i, (col_, mk, lab) in STYLE.items():
            v = c(tag, m, 0.7, i)
            ax.plot(AL, v, marker=mk, color=col_, lw=2, ms=5, label=lab if (row == 0 and col == 0) else None)
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_xlabel("imputation quality  α  (1 = perfect imputer)", fontsize=8)
        if col == 0:
            ax.set_ylabel(f"{'zero-shot' if row==0 else 'adapted'}\npaired relMSE", fontsize=9)
        ax.set_title(f"{m}  p=0.7", fontsize=10)
        ax.tick_params(labelsize=8)
fig.legend(loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(0.5, 0.965), frameon=False)
fig.suptitle("S27 — a content-zeroing mask interface caps the benefit of imputation at ~zero\n"
             "(red is flat: a perfect imputer buys nothing.  blue = same model, one line removed, "
             "input projection adapted 2 min)", fontsize=12, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(HERE, "s27.png"), dpi=140, bbox_inches="tight")
print("figure ->", os.path.join(HERE, "s27.png"))
