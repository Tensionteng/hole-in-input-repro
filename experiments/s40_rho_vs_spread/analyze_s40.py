#!/usr/bin/env python
"""S40: does rho predict how far a checkpoint's benchmark score moves with the fill?"""
import json, os, glob
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
R = {"cells": {}, "params": {}, "name": {}}
for f in sorted(glob.glob(os.path.join(HERE, "part_*.json"))) + \
         [os.path.join(HERE, "gate.json")]:
    if os.path.exists(f):
        d = json.load(open(f))
        for k in R:
            R[k].update(d.get(k, {}))

# rho from S30 (bolt base, chronos2, t5 small, timesfm, moirai base) and S36 (the rest)
RHO = {}
s30 = json.load(open(os.path.join(EXP, "s30_crossmodel", "s30_results.json")))["models"]
for src_key, key in (("bolt", "bolt_base"), ("chronos2", "chronos2"), ("t5", "t5_small"),
                     ("timesfm", "timesfm"), ("moirai", "moirai_b")):
    a = s30.get(src_key, {}).get("agg", {})
    if a:
        RHO[key] = a
for f in glob.glob(os.path.join(EXP, "s36_models", "part_*.json")):
    for k, v in json.load(open(f)).get("models", {}).items():
        if "agg" in v:
            RHO[k] = v["agg"]

FILLS = ("nan", "linear", "zero", "ffill")
# Context-missing fraction per dataset, as measured by the reader. Section 4 stratifies by
# the window's own missing fraction and reads its ordering off the HIGH stratum; pooling
# every dataset instead is dominated by the near-complete ones, so we report both.
NANFRAC = {"electricity/H": .0033, "kdd_cup_2018_with_missing/H": .1215,
           "bitbrains_rnd/5T": .0414, "bitbrains_fast_storage/5T": .0043,
           "hierarchical_sales/D": .0125, "temperature_rain_with_missing": .0328,
           "jena_weather/H": .0002}
HIGH = 0.03
# The max over datasets is not comparable across checkpoints -- for some the argmax is a
# near-complete dataset with 63 windows -- so the high-missingness statistic is a FIXED
# dataset: the most missing-valued one in the corpus.
HIGH_DS = "kdd_cup_2018_with_missing/H"
rows = []
for m in sorted({k.split("|")[0] for k in R["cells"]}):
    per, hi, rows_high = [], [], None
    for ds in sorted({k.split("|")[1] for k in R["cells"] if k.startswith(m + "|")}):
        v = [R["cells"].get(f"{m}|{ds}|{h}", {}).get("mase_median", np.nan) for h in FILLS]
        v = np.array(v, float)
        if not np.isfinite(v).all() or np.nanmin(v) <= 0:
            continue
        sp = (np.max(v) - np.min(v)) / np.min(v)
        per.append(sp)
        if NANFRAC.get(ds, 0) >= HIGH:
            hi.append(sp)
        if ds == HIGH_DS:
            rows_high = sp
    if not per:
        continue
    a = RHO.get(m, {})
    pl = a.get("plain", {}).get("ratio_median")
    dec = None
    for c in ("mask", "nan"):
        if a.get(c, {}).get("ratio_median") is not None:
            dec = a[c]["ratio_median"]
            break
    rows.append({"model": m, "name": R["name"].get(m, m), "n_ds": len(per),
                 "spread_median": float(np.median(per)), "spread_max": float(np.max(per)),
                 "spread_high": float(np.mean(hi)) if hi else None, "n_high": len(hi),
                 "spread_kdd": rows_high,
                 "rho_plain": pl, "rho_dec": dec,
                 "gap": (pl - dec) if (pl is not None and dec is not None) else None,
                 "params": R["params"].get(m)})

print(f"{'checkpoint':22s} {'params':>8s} {'rho pl':>7s} {'rho dec':>8s} {'gap':>6s} "
      f"{'all ds':>8s} {'>3% NaN':>9s} {'kdd 12%':>9s}")
for r in sorted(rows, key=lambda x: -(x["spread_median"])):
    f = lambda v, w=7, d=3: (f"{v:{w}.{d}f}" if v is not None else f"{'--':>{w}}")
    sh = f"{r['spread_high']*100:8.1f}%" if r["spread_high"] is not None else f"{'--':>9}"
    sk = f"{r['spread_kdd']*100:8.1f}%" if r["spread_kdd"] is not None else f"{'--':>9}"
    print(f"{r['name']:22s} {(r['params'] or 0)/1e6:7.1f}M {f(r['rho_plain'])} "
          f"{f(r['rho_dec'],8)} {f(r['gap'],6)} {r['spread_median']*100:7.1f}% {sh} {sk}")

PAIRS = [("rho_plain", "spread_median", "rho plain", "spread over all datasets"),
         ("gap", "spread_median", "rho plain - rho declared", "spread over all datasets"),
         ("rho_plain", "spread_high", "rho plain", "spread on >3% NaN datasets"),
         ("gap", "spread_high", "rho plain - rho declared", "spread on >3% NaN datasets"),
         ("gap", "spread_kdd", "rho plain - rho declared",
          "spread on the most missing-valued dataset"),
         ("rho_plain", "spread_kdd", "rho plain",
          "spread on the most missing-valued dataset")]
for xk, yk, lab, ylab in PAIRS:
    xs = [(r[xk], r[yk]) for r in rows if r[xk] is not None and r[yk] is not None]
    if len(xs) < 3:
        continue
    x, y = np.array([a for a, _ in xs]), np.array([b for _, b in xs])
    sr = stats.spearmanr(x, y)
    pr = stats.pearsonr(x, y)
    print(f"{lab} vs {ylab} (n={len(x)}): "
          f"Spearman {sr.statistic:+.3f} (p={sr.pvalue:.3f}), "
          f"Pearson {pr.statistic:+.3f} (p={pr.pvalue:.3f})")
json.dump(rows, open(os.path.join(HERE, "s40_summary.json"), "w"), indent=1)
