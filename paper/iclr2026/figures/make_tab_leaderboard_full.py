#!/usr/bin/env python
"""The missingness leaderboard in full: every model x every dataset x {perfect, linear, zero}.

Cell convention (identical to Table 3's aggregation): per-dataset cell = median of the
four mechanism medians (70% missing, declared path, paired relMSE vs own clean); the
right-hand "med." column is the median over the nine datasets, i.e. the row values of
Table 3. Sources: s57_* for the bolt-family arms and the shipped models, s64_bench3 for
Chronos-2 and Moirai 2.0 (strict fp32). Writes figures/tab_leaderboard_full.tex.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))

D57 = {
    "bolt": json.load(open(os.path.join(EXP, "s45_pretrain", "s57_bolt.json"))),
    "p4": json.load(open(os.path.join(EXP, "s45_pretrain", "s57_p4.json"))),
    "q50": json.load(open(os.path.join(EXP, "s45_pretrain", "s57_q50.json"))),
    "moirai2": json.load(open(os.path.join(EXP, "s45_pretrain", "s57_moirai2.json"))),
    "timesfm": json.load(open(os.path.join(EXP, "s45_pretrain", "s57_timesfm.json"))),
    "tirex": json.load(open(os.path.join(EXP, "s45_pretrain", "s57_tirex.json"))),
    "flowstate": json.load(open(os.path.join(EXP, "s45_pretrain", "s57_flowstate.json"))),
    "timerxl": json.load(open(os.path.join(EXP, "s45_pretrain", "s57_timerxl.json"))),
}
D64 = json.load(open(os.path.join(EXP, "s64_bench3", "s64_leaderboard_cells.json")))

DS9 = ["ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness"]
DSLAB = ["ETTh1", "ETTm1", "weath.", "ETTh2", "ETTm2", "electr.", "traffic",
         "exch.", "illness"]
MECHS = ["mcar", "block", "mnar_high", "mnar_extreme"]
FILLS = [("oracle", "perfect"), ("linear", "linear"), ("zero", "zero")]

# (row label, source, arm)
ROWS = [
    (r"\bolt{} base (vanilla)", D57["bolt"], "bolt-base-stock"),
    (r"\quad $+$ $5$k steps, no missingness", D57["p4"], "P4-base"),
    ("Moirai 2.0 (vanilla)", D57["moirai2"], "moirai2"),
    ("TimesFM 2.5", D57["timesfm"], "timesfm"),
    ("TiRex", D57["tirex"], "tirex"),
    ("FlowState", D57["flowstate"], "flowstate"),
    ("Timer-XL", D57["timerxl"], "timerxl"),
    (r"\bolt{} base $+$ CPT (ours)", D57["q50"], "Q50-base"),
    (r"\ctwo{} (vanilla)", D64, "c2-stock"),
    (r"\ctwo{} $+$ CPT (ours)", D64, "c2-w50"),
    ("Moirai 2.0 $+$ CPT (ours)", D64, "m2-w50"),
]


def cell(src, ds, mech, fill, arm):
    v = src["grid"].get(f"{ds}|{mech}|0.7|{fill}|{arm}")
    if v is None:
        return np.nan
    return v["median"]


def ds_cell(src, ds, fill, arm):
    return float(np.nanmedian([cell(src, ds, m, fill, arm) for m in MECHS]))


def fmt(v):
    if not np.isfinite(v):
        return "--"
    return f"{v:.2f}" if v < 100 else f"{v:.0f}"


lines = []
lines.append(r"\begin{table}[!p]")
lines.append(r"\centering")
lines.append(r"\caption{\textbf{The missingness leaderboard in full.} Median relMSE against "
             r"each model's own clean accuracy at $70\%$ missing on its declared path, "
             r"on nine benchmarks under the three fills a deployment can produce; each "
             r"dataset cell is the median of the four mechanism medians, and \emph{med.} "
             r"is the median over datasets. The main-text Table~\ref{tab:leaderboard} "
             r"shows the trained models' rows. \textbf{Bold} marks the best model in a "
             r"column. TimesFM, TiRex and FlowState discard or overwrite the fill, so "
             r"their rows are identical on every fill by construction; Timer-XL's "
             r"declared path is non-finite.}")
lines.append(r"\label{tab:leaderboardfull}")
lines.append(r"\vspace{2pt}")
lines.append(r"\scriptsize")
lines.append(r"\setlength{\tabcolsep}{2.6pt}")
lines.append(r"\resizebox{\textwidth}{!}{%")
lines.append(r"\begin{tabular}{@{} l " + "c" * 10 + r" @{}}")
lines.append(r"\toprule")
lines.append(" & ".join(["Model"] + DSLAB + [r"\emph{med.}"]) + r" \\")

for fi, (fkey, flab) in enumerate(FILLS):
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{11}{@{}l}{\emph{" + flab + r" fill}} \\[1pt]")
    table = []
    for lab, src, arm in ROWS:
        row = [ds_cell(src, d, fkey, arm) for d in DS9]
        med = float(np.nanmedian(row))
        table.append((lab, row, med))
    # bold the best (min, finite) in each column
    for j in range(len(DS9) + 1):
        vals = [t[1][j] if j < 9 else t[2] for t in table]
        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            continue
        best = min(finite)
        for k, (lab, row, med) in enumerate(table):
            v = row[j] if j < 9 else med
            if np.isfinite(v) and abs(v - best) < 1e-9:
                if j < 9:
                    table[k][1][j] = ("B", v)
                else:
                    table[k] = (lab, row, ("B", med))
    for lab, row, med in table:
        if lab == "Timer-XL":
            lines.append(lab + r" & \multicolumn{10}{c}{declared path non-finite} \\")
            continue
        cells = []
        for v in row:
            cells.append(r"\textbf{" + fmt(v[1]) + "}" if isinstance(v, tuple) else fmt(v))
        m = r"\textbf{" + fmt(med[1]) + "}" if isinstance(med, tuple) else fmt(med)
        lines.append(lab + " & " + " & ".join(cells) + " & " + m + r" \\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}}")
lines.append(r"\end{table}")

out = os.path.join(HERE, "tab_leaderboard_full.tex")
open(out, "w").write("\n".join(lines) + "\n")
print("wrote", out)
# verification: print the med. column per fill (should match Table 3's rows)
for fkey, flab in FILLS:
    print(flab)
    for lab, src, arm in ROWS:
        med = float(np.nanmedian([ds_cell(src, d, fkey, arm) for d in DS9]))
        print(f"    {lab:42s} {med:.2f}")
