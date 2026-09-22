#!/usr/bin/env python
"""The dataset table: what each source is, and the context/horizon used on it.

Channel counts, timeline lengths and usable-series counts are read from the data and from the
round results rather than typed in; only the domain and frequency labels are editorial.
"""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
sys.path.insert(0, os.path.join(EXP, "s35_breadth"))
sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
import run_s35_breadth as s35

S35 = json.load(open(os.path.join(EXP, "s35_breadth", "s35_results.json")))
META = [
    # key, printed name, domain, frequency, group
    ("ETTh1", "ETTh1", "electr. transformer", "1\\,h", "synthetic"),
    ("ETTh2", "ETTh2", "electr. transformer", "1\\,h", "synthetic"),
    ("ETTm1", "ETTm1", "electr. transformer", "15\\,min", "synthetic"),
    ("ETTm2", "ETTm2", "electr. transformer", "15\\,min", "synthetic"),
    ("weather", "weather", "meteorology", "10\\,min", "synthetic"),
    ("electricity", "electricity", "household load", "1\\,h", "synthetic"),
    ("traffic", "traffic", "road occupancy", "1\\,h", "synthetic"),
    ("exchange", "exchange rate", "finance", "1\\,d", "synthetic"),
    ("illness", "illness", "public health", "1\\,w", "synthetic"),
]
# Window counts read from the rounds that produced them, not typed in.
S10 = json.load(open(os.path.join(EXP, "s10_real_migration", "s10_real_results.json")))
S14 = json.load(open(os.path.join(EXP, "s14_metrla", "s14_metrla_results.json")))
n_penn = S10["records"]["anchor|bolt|cens|linear"]["n_windows"]
n_metr = S14["records"]["anchor|bolt|miss|linear"]["n_windows"] \
    if "anchor|bolt|miss|linear" in S14["records"] \
    else S14["records"]["anchor|moirai|miss|linear"]["n_windows"]
REAL = [
    ("Penmanshiel", "wind-farm curtailm.", "10\\,min", "144", "24", f"{n_penn}"),
    ("METR-LA", "traffic-sensor out.", "5\\,min", "512", "64", f"{n_metr}"),
]
GIFT = ("GIFT-Eval (9 subsets)", "mixed", "5\\,min--1\\,d", "512", "64", "7{,}725")

L, H = s35.L, s35.H
rows = []
for key, name, dom, freq, _ in META:
    df = pd.read_csv(os.path.join(s35.TSLIB, s35.DATASETS[key][0]))
    df = df.drop(columns=[c for c in df.columns if c.lower() in ("date", "unnamed: 0")])
    X, st = s35.load(key, 300)
    cap = s35.DATASETS[key][1]
    ch = (f"{X.shape[1]} of {df.shape[1]}" if cap and df.shape[1] > cap
          else str(df.shape[1]))
    _, cl0, _, gt0 = s35.build(X, st, "clean", 0.0)
    keep0 = s35.finite(cl0) & s35.finite(gt0) & (cl0.std(axis=1) > 1e-8)
    nser = int(keep0.sum())
    rows.append((name, dom, freq, f"{len(df):,}".replace(",", "{,}"), ch, str(L), str(H),
                 str(len(st)), f"{nser:,}".replace(",", "{,}")))

L2 = [r"\begin{table}[H]", r"\centering",
      r"\caption{\textbf{Datasets, and the context and horizon used on each.} A `series' is one"
      r" channel of one window; the probe and grid measurements of Table~\ref{tab:breadth} are"
      r" taken over all of them. Windows are drawn once per source and reused by every round"
      r" except the calibration one, which needs disjoint calibration and test halves and"
      r" draws $120$ of each. Channel counts written $a$ of $b$ are a fixed random"
      r" subsample drawn once with the global seed, so that an $862$-channel source does not"
      r" dominate the pooled statistics."
      r" \hi{Red} marks the one source whose $L$ and $H$ differ from the $512/64$ used"
      r" everywhere else; it inherits them from the study it extends."
      r" GIFT-Eval's per-subset missingness is in Table~\ref{tab:missingrates}.}",
      r"\label{tab:datasets}", r"\vspace{2pt}", r"\footnotesize",
      r"\setlength{\tabcolsep}{3.2pt}",
      r"\begin{tabular}{llrrrrrrr}", r"\toprule",
      r"& & & \multicolumn{2}{c}{source} & \multicolumn{2}{c}{lengths} &"
      r" \multicolumn{2}{c}{evaluation} \\",
      r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
      r"Dataset & Domain & Freq. & steps & chan. & $L$ & $H$ & win. & series \\",
      r"\midrule",
      r"\multicolumn{9}{l}{\emph{Synthetic mechanisms applied to a clean source}} \\[1pt]"]
prev_dom = None
for i, r in enumerate(rows):
    dom = r[1]
    if dom != prev_dom:
        n = sum(1 for q in rows if q[1] == dom)
        cell = r"\multirow{" + str(n) + r"}{*}{" + dom + "}" if n > 1 else dom
    else:
        cell = ""
    prev_dom = dom
    L2.append(" & ".join([r[0], cell, r[2], r[3], r[4], r[5], r[6], r[7], r[8]]) + r" \\")
L2 += [r"\midrule",
       r"\multicolumn{9}{l}{\emph{Real missingness, no mask applied by us}} \\[1pt]"]
for name, dom, freq, l, h, nw in REAL:
    # Flag any row whose lengths differ from the 512/64 used everywhere else, since a reader
    # skimming will otherwise carry the standard pair across the whole table.
    odd = (l, h) != (str(L), str(H))
    lc = (r"\hi{" + l + "}") if odd else l
    hc = (r"\hi{" + h + "}") if odd else h
    L2.append(f"{name} & {dom} & {freq} & -- & -- & {lc} & {hc} & {nw} & -- \\\\")
L2.append(f"{GIFT[0]} & {GIFT[1]} & {GIFT[2]} & -- & -- & {GIFT[3]} & {GIFT[4]} & "
          f"{GIFT[5]} & -- \\\\")
L2 += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(HERE, "tab_datasets_full.tex"), "w").write("\n".join(L2) + "\n")

# Series counts at the 300-window setting used by the sweeps, so tab_breadth can quote the
# same n as the measurements it reports.
counts = {}
for key, name, *_ in META:
    X, st = s35.load(key, 300)
    _, cl, _, gt = s35.build(X, st, "clean", 0.0)
    keep = s35.finite(cl) & s35.finite(gt) & (cl.std(axis=1) > 1e-8)
    counts[key] = {"n_win": int(len(st)), "n_series": int(keep.sum()),
                   "n_channels": int(X.shape[1])}
json.dump(counts, open(os.path.join(HERE, "dataset_counts.json"), "w"), indent=1)
print("wrote tab_datasets_full.tex and dataset_counts.json")
