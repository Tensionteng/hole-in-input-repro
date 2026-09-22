#!/usr/bin/env python
"""S25 figure: reachable-set structure (Part 0) + the paired two-floor decomposition."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CONVS = ("plain_fill", "fill_mask", "nan")
LBL = {"plain_fill": "plain fill\n(no mask)", "fill_mask": "fill + mask", "nan": "NaN"}
ORDER = ["mcar:0.3", "mcar:0.7", "block:0.3", "block:0.7",
         "mnar_high:0.3", "mnar_high:0.7", "mnar_extreme:0.3", "mnar_extreme:0.7"]


def main():
    p0 = json.load(open(os.path.join(HERE, "s25_results_p0.json")))["part0"]["cells"]
    agg = json.load(open(os.path.join(HERE, "s25_paired_agg.json")))
    keys = [k for k in ORDER if k in agg]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))

    # (a) reachable-set rank. No dual axis: a log-scaled companion bar visually competes
    # with rank 64 and misleads. Rank is the bar; the permutation sensitivity is annotated.
    ax = axes[0]
    rank, perm = [], []
    for c in CONVS:
        sel = [v for k, v in p0.items() if k.endswith(":" + c)]
        rank.append(np.median([s_["rank_1e-4"] for v in sel for s_ in v["per_series"]]))
        perm.append(np.median([v["perm_invariance"]["ratio_perm_over_redraw"] for v in sel]))
    xs = np.arange(3)
    cols = ["#c0392b", "#e08a5d", "#3b6ea5"]
    ax.bar(xs, rank, 0.55, color=cols)
    ax.set_ylabel("numerical rank of $J_M$  (dim. of reachable set)")
    ax.set_ylim(0, 88)
    for x, r, pv in zip(xs, rank, perm):
        ax.text(x, r + 3, f"rank {r:.0f}", ha="center", fontsize=12, fontweight="bold")
        lab = f"{pv:.0e}" if pv > 0 else "0 (exactly)"
        ax.text(x, r + 11, f"fill-content\nsensitivity\n{lab}", ha="center", fontsize=8.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{LBL[c]}\n\n{n}" for c, n in zip(
        CONVS, ["fill content fully enters\nthe forecast",
                "only (loc, scale) —\ncontent invisible",
                "no lever at all"])], fontsize=8.5)
    ax.set_title("(a) The fill-reachable set is fixed by\nthe input convention, not the model",
                 fontsize=10)

    # (b) achievable levels
    ax = axes[1]
    xs = np.arange(len(keys))
    series = [("best fixed fill", "best_fixed", "#8c8c8c"),
              ("learned fill (plain)", "fillnet", "#3b6ea5"),
              ("learned fill (+mask)", "fill_mask", "#7ba7d1"),
              ("direct predictor", "direct", "#c0392b")]
    for i, (lab, fld, col) in enumerate(series):
        ax.bar(xs + (i - 1.5) * 0.2, [agg[k][fld] for k in keys], 0.2, label=lab, color=col)
    ax.axhline(1.0, color="k", lw=0.9, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([k.replace(":", " p=") for k in keys], fontsize=7.5,
                       rotation=30, ha="right")
    ax.set_ylabel("paired relMSE (median over windows)")
    ax.set_title("(b) What each lever can achieve", fontsize=10)
    ax.legend(fontsize=8)

    # (c) the decomposition
    ax = axes[2]
    t1 = np.array([agg[k]["term1_information"] for k in keys])
    t2 = np.array([agg[k]["term2_architecture"] for k in keys])
    t3 = np.array([agg[k]["term3_fill"] for k in keys])
    ax.bar(xs, t1, 0.62, label="(1) information floor", color="#c0392b")
    ax.bar(xs, t2, 0.62, bottom=t1, label="(2) architectural floor", color="#e08a5d")
    ax.bar(xs, t3, 0.62, bottom=t1 + np.maximum(t2, 0), label="(3) fill / routing (recoverable)",
           color="#3b6ea5")
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([k.replace(":", " p=") for k in keys], fontsize=7.5,
                       rotation=30, ha="right")
    ax.set_ylabel("excess relMSE over clean")
    ax.set_title("(c) Two-floor decomposition:\nwhere the damage actually lives", fontsize=10)
    ax.legend(fontsize=8)
    ax.annotate("64% of the damage is\nrecoverable by the fill alone",
                xy=(5, t1[5] + t2[5] + t3[5] * 0.55), xytext=(1.2, 1.55),
                fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=1.1))

    fig.suptitle("S25 — information floor vs architectural floor "
                 "(chronos-bolt-base, frozen; paired per-window medians)", fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(HERE, "s25.png")
    fig.savefig(out, dpi=150)
    print("figure ->", out)


if __name__ == "__main__":
    main()
