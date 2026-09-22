#!/usr/bin/env python
"""s40b: the rho-vs-spread analysis on the Moirai-2.0-era census (9 rho-bearing checkpoints).

Same methodology as s40 (per-dataset fill spread, median / >3% NaN / kdd-12%), same rho
sources (S36/S30/S46 probes); the checkpoint set is the paper's updated census:
bolt tiny/mini/small/base, chronos2, t5 small, t5 base, timesfm, moirai2.
"""
import json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
G32 = os.path.join(EXP, "s32_gifteval")
OUT = os.path.join(HERE, "s40b_summary.json")

DS_HIGH = ("electricity/H", "kdd_cup_2018_with_missing/H", "restaurant",
           "bitbrains_rnd/5T")   # >3% context NaN datasets in the pool
KDD = "kdd_cup_2018_with_missing/H"
FILLS = ("nan", "linear", "zero", "ffill")

CELL_SOURCES = [
    (os.path.join(HERE, "part_a.json"), None),                    # bolt tiny/mini/small/base
    (os.path.join(HERE, "part_e_t5.json"), None),                 # t5 base
    (os.path.join(G32, "s32_partA.json"), ("chronos2", "timesfm")),
    (os.path.join(G32, "s32_partA_moirai2.json"), None),
    (os.path.join(G32, "s32_partA_t5small.json"), None),
    (os.path.join(G32, "s32_partA_tirex.json"), None),
    (os.path.join(G32, "s32_partA_flowstate.json"), None),
]

RHO = {  # (rho_plain, rho_declared), from tab_families_full (S36/S30/S46/S49 medians)
    "bolt_tiny": (0.87, 0.00), "bolt_mini": (0.96, 0.00),
    "bolt_small": (0.97, 0.00), "bolt_base": (0.91, 0.00),
    "chronos2": (0.90, 0.00), "t5-small": (0.95, 0.00), "t5_base": (0.91, 0.00),
    "timesfm": (0.86, 0.00), "moirai2": (0.97, 0.95),
    "tirex": (0.97, 0.00), "flowstate": (0.85, 0.00),
}
NAME = {"bolt_tiny": "chronos-bolt-tiny", "bolt_mini": "chronos-bolt-mini",
        "bolt_small": "chronos-bolt-small", "bolt_base": "chronos-bolt-base",
        "chronos2": "chronos-2", "t5-small": "chronos-t5-small",
        "t5_base": "chronos-t5-base", "timesfm": "timesfm-2.5",
        "moirai2": "moirai-2.0-R-small", "tirex": "tirex-1.1",
        "flowstate": "flowstate-r1"}
PARAMS = {"bolt_tiny": 8.7e6, "bolt_mini": 21e6, "bolt_small": 48e6,
          "bolt_base": 205e6, "chronos2": 120e6, "t5-small": 46e6,
          "t5_base": 200e6, "timesfm": 200e6, "moirai2": 11.4e6,
          "tirex": 35e6, "flowstate": 9.1e6}

cells = {}
for path, keep in CELL_SOURCES:
    if not os.path.exists(path):
        print("missing (ok if t5-small still running):", path)
        continue
    d = json.load(open(path))
    cs = d.get("cells", d)
    for k, v in cs.items():
        m = k.split("|")[0]
        if keep is None or m in keep:
            cells[k] = v

models = sorted({k.split("|")[0] for k in cells})
datasets = sorted({k.split("|")[1] for k in cells})

rows = []
for m in models:
    per = []
    for ds in datasets:
        try:
            vals = [cells[f"{m}|{ds}|{f}"]["mase_median"] for f in FILLS]
        except KeyError:
            continue
        per.append((ds, (max(vals) - min(vals)) / max(min(vals), 1e-9)))
    if not per:
        continue
    hi = [s for ds, s in per if ds in DS_HIGH]
    kdd = dict(per).get(KDD)
    pl, dec = RHO.get(m, (None, None))
    rows.append({"model": m, "name": NAME.get(m, m), "n_ds": len(per),
                 "spread_median": float(np.median([s for _, s in per])),
                 "spread_max": float(max(s for _, s in per)),
                 "spread_high": float(np.mean(hi)) if hi else None,
                 "spread_kdd": kdd,
                 "rho_plain": pl, "rho_dec": dec,
                 "gap": (pl - dec) if pl is not None and dec is not None else None,
                 "params": PARAMS.get(m)})

print(f"{'checkpoint':22s} {'rho pl':>7s} {'rho dec':>8s} {'gap':>6s} "
      f"{'all ds':>8s} {'>3% NaN':>9s} {'kdd 12%':>9s}")
for r in sorted(rows, key=lambda x: -x["spread_median"]):
    f = lambda v: f"{v:7.3f}" if v is not None else f"{'--':>7s}"
    sh = f"{r['spread_high']*100:8.1f}%" if r["spread_high"] is not None else f"{'--':>9s}"
    sk = f"{r['spread_kdd']*100:8.1f}%" if r["spread_kdd"] is not None else f"{'--':>9s}"
    print(f"{r['name']:22s} {f(r['rho_plain'])} {f(r['rho_dec'])} {f(r['gap'])} "
          f"{r['spread_median']*100:7.1f}% {sh} {sk}")

PAIRS = [("rho_plain", "spread_median"), ("gap", "spread_median"),
         ("rho_plain", "spread_high"), ("gap", "spread_high"),
         ("gap", "spread_kdd"), ("rho_plain", "spread_kdd")]
print()
for xk, yk in PAIRS:
    xs = [(r[xk], r[yk]) for r in rows if r[xk] is not None and r[yk] is not None]
    if len(xs) < 3:
        continue
    x, y = np.array([a for a, _ in xs]), np.array([b for _, b in xs])
    sr = stats.spearmanr(x, y)
    pr = stats.pearsonr(x, y)
    print(f"{xk:10s} vs {yk:14s} (n={len(x)}): Spearman {sr.statistic:+.3f} "
          f"(p={sr.pvalue:.3f}), Pearson {pr.statistic:+.3f} (p={pr.pvalue:.3f})")

json.dump(rows, open(OUT, "w"), indent=1)
print("\nwrote", OUT)
