#!/usr/bin/env python
"""M3 analysis: does the unreported fill convention move leaderboard RANKINGS, or only
absolute scores?

Re-analysis of S32 part A (no new runs): 4 models x 9 missing-containing GIFT-Eval datasets
x 4 fill conventions, MASE median per cell, last-window protocol.

Protocols compared:
  fixed-<fill> : every model scored under the same fill (linear / zero / ffill / nan)
  best-per-model : each model scored under its own best fill (per dataset)

For each protocol we rank models two ways: (i) mean of per-dataset MASE ranks,
(ii) mean of per-dataset MASE normalised by the best model in that dataset.
We then count pairwise inversions between protocols and report the margins.
"""
import json, os
from itertools import combinations
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "s32_partA.json")))
cells = {k: v for k, v in d["cells"].items() if not k.startswith("moirai|")}
cells.update(json.load(open(os.path.join(HERE, "s32_partA_moirai2.json")))["cells"])

models = sorted({k.split("|")[0] for k in cells})
datasets = sorted({k.split("|")[1] for k in cells})
fills = ["nan", "linear", "zero", "ffill"]
print(f"models: {models}\ndatasets: {datasets}\nfills: {fills}\n")

M = {m: {ds: {f: cells[f"{m}|{ds}|{f}"]["mase_median"] for f in fills}
         for ds in datasets} for m in models}


def protocol_scores(fill):
    """Return {model: (mean_rank, mean_rel_mase)} under one fixed fill."""
    rank_agg, rel_agg = {}, {}
    for m in models:
        ranks, rels = [], []
        for ds in datasets:
            scores = sorted(M[m2][ds][fill] for m2 in models)
            ranks.append(1 + sum(s < M[m][ds][fill] for s in scores))
            rels.append(M[m][ds][fill] / scores[0])
        rank_agg[m] = float(np.mean(ranks))
        rel_agg[m] = float(np.mean(rels))
    return rank_agg, rel_agg


def best_per_model():
    rank_agg, rel_agg, pick = {}, {}, {}
    for m in models:
        ranks, rels, picks = [], [], []
        for ds in datasets:
            bf = min(fills, key=lambda f: M[m][ds][f])
            picks.append(bf)
            v = M[m][ds][bf]
            scores = sorted(min(M[m2][ds][f] for f in fills) for m2 in models)
            ranks.append(1 + sum(s < v for s in scores))
            rels.append(v / scores[0])
        rank_agg[m] = float(np.mean(ranks))
        rel_agg[m] = float(np.mean(rels))
        pick[m] = picks
    return rank_agg, rel_agg, pick


def ranking(scores):
    return [m for m, _ in sorted(scores.items(), key=lambda kv: kv[1])]


def inversions(order_a, order_b):
    inv = []
    for a, b in combinations(order_a, 2):
        if order_b.index(a) > order_b.index(b):
            inv.append((a, b))
    return inv


protocols = {f: protocol_scores(f) for f in fills}
best_rank, best_rel, pick = best_per_model()
protocols["best-per-model"] = (best_rank, best_rel)

print("== per-protocol standings (mean per-dataset rank / mean relative MASE) ==")
for name, (rk, rl) in protocols.items():
    order = ranking(rk)
    print(f"\n[{name}]")
    for m in order:
        print(f"  {m:12s} rank {rk[m]:.2f}   relMASE {rl[m]:.4f}")

print("\n== pairwise inversions vs the 'linear' protocol ==")
base_rank, base_rel = protocols["linear"]
for name, (rk, rl) in protocols.items():
    if name == "linear":
        continue
    for agg, sc in (("rank", rk), ("relMASE", rl)):
        inv = inversions(ranking(base_rank if agg == "rank" else base_rel), ranking(sc))
        print(f"  linear vs {name:15s} ({agg:7s}): {len(inv)} inversion(s) {inv}")

print("\n== margins: adjacent-model gaps under 'linear' vs within-model spread across fills ==")
for agg, sc in (("rank", base_rank), ("relMASE", base_rel)):
    order = ranking(sc)
    gaps = [(order[i], order[i + 1], sc[order[i + 1]] - sc[order[i]])
            for i in range(len(order) - 1)]
    print(f"  [{agg}] adjacent gaps: " +
          ", ".join(f"{a}>{b}: {g:.4f}" for a, b, g in gaps))
spread = {}
for m in models:
    per_fill_rank, per_fill_rel = [], []
    for f in fills:
        rk, rl = protocol_scores(f)
        per_fill_rank.append(rk[m]); per_fill_rel.append(rl[m])
    spread[m] = (max(per_fill_rank) - min(per_fill_rank),
                 max(per_fill_rel) - min(per_fill_rel))
print("  within-model spread across fills (rank, relMASE):")
for m in models:
    print(f"    {m:12s} Drank {spread[m][0]:.2f}  DrelMASE {spread[m][1]:.4f}")

print("\n== best fill picked per model (per dataset) ==")
for m in models:
    print(f"  {m:12s} {pick[m]}")
