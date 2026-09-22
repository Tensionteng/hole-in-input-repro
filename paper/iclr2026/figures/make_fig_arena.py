#!/usr/bin/env python
"""The trust arena: four-quadrant heatmaps with the two theory references.

Panels: the 2x2 factorial arms (interface x corpus) plus the Bayes-optimal floor and the
naive-trust reference, all as 5x5 heatmaps over (q_r, q_c) at tailblock rate 0.7, mean
over three seeds. Arm cells: relMSE vs the arm's own clean (measured). Reference panels:
MSE ratio to the clean Bayes floor (exact conditioning). Shared colorscale clipped at 3;
clipped cells are annotated with their exact values.

Data: s69b_eval_<arm>_s<seed>.json + s69b_bayes.json in experiments/s69_arena.
"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ARENA = os.path.join(ROOT, "tsfm_missing", "experiments", "s69_arena")

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 8,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "pdf.fonttype": 42, "figure.dpi": 200})

QS = ["0.0", "0.25", "0.5", "0.75", "1.0"]
QLAB = ["0", ".25", ".5", ".75", "1"]
MECH, RATE = "tailblock", "0.7"
VMIN, VMAX = 0.95, 3.0

def arm_grid(arm):
    mats = []
    for s in (0, 1, 2):
        d = json.load(open(os.path.join(ARENA, f"s69b_eval_{arm}_s{s}.json")))
        cells = d["cells"]
        mats.append(np.array([[cells[f"{MECH}|{RATE}|{qr}|{qc}"]["rel_median"]
                               for qc in QS] for qr in QS]))
    return np.mean(mats, 0)

def bayes_grid(field):
    b = json.load(open(os.path.join(ARENA, "s69b_bayes.json")))
    floor = b["clean_floor"]["mse_median"]
    cells = b["cells"]
    return np.array([[cells[f"{MECH}|{RATE}|{qr}|{qc}"][f"{field}_median"] / floor
                      for qc in QS] for qr in QS])

PANELS = [
    ("A", arm_grid("A"), "A: blocked interface,\nfiltered corpus"),
    ("B", arm_grid("B"), "B: blocked interface,\ndiverse corpus"),
    ("C", arm_grid("C"), "C: accessible interface,\nfiltered corpus"),
    ("D", arm_grid("D"), "D: accessible interface,\ndiverse corpus (ours)"),
    ("F", bayes_grid("mse"), "Bayes floor\n(exact conditioning)"),
    ("N", bayes_grid("mse_naive"), "naive trust\n(fill taken as real)"),
]

fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.6))
cmap = plt.get_cmap("viridis_r")
for ax, (tag, M, title) in zip(axes.flat, PANELS):
    im = ax.imshow(np.clip(M, VMIN, VMAX), cmap=cmap, vmin=VMIN, vmax=VMAX,
                   origin="lower", aspect="equal")
    for i in range(5):
        for j in range(5):
            v = M[i, j]
            txt = f"{v:.2f}" if v <= VMAX else f"{v:.1f}>"
            dark = min(v, VMAX) < 0.55 * (VMIN + VMAX) + 0.45 * VMIN
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=5.6, color="white" if not dark else "0.15",
                    fontweight="bold" if tag == "D" else "normal")
    ax.set_xticks(range(5), QLAB); ax.set_yticks(range(5), QLAB)
    ax.set_xlabel(r"context quality $q_c$")
    ax.set_title(title, fontsize=7.6, pad=3)
    if tag in ("A", "D"):
        ax.set_ylabel(r"repair quality $q_r$")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

fig.suptitle(r"forecast error (relMSE) over the two source-quality dials, tail outage at rate $0.7$",
             fontsize=8.2, y=0.995)
fig.tight_layout(pad=.5, w_pad=1.1, h_pad=1.0, rect=(0, 0.04, 1, 0.97))
cax = fig.add_axes([0.08, 0.005, 0.84, 0.018])
fig.colorbar(im, cax=cax, orientation="horizontal").set_label(
    "relMSE vs own clean (arms) / vs clean Bayes floor (references), clipped at 3", fontsize=6.5)

out = os.path.join(HERE, "fig_arena.pdf")
fig.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
for tag, M, _ in PANELS:
    print(f"  {tag}: range {M.min():.2f}-{M.max():.2f} | (0,0)={M[0,0]:.2f} (0,1)={M[0,-1]:.2f} (1,0)={M[-1,0]:.2f} (1,1)={M[-1,-1]:.2f}")
