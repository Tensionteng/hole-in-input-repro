#!/usr/bin/env python
"""S38 scorecard: does the mechanism-keyed repair hold on datasets the detector never saw?"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in ("gate.json", "heldout_a.json", "heldout_b.json"):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        R.update(json.load(open(p))["datasets"])
INS = ("ETTh1", "ETTm1", "weather")
GRP = ["clean", "mcar", "block", "mnar_high", "mnar_extreme"]
NAME = {"mcar": "scattered", "mnar_high": "censoring", "mnar_extreme": "extreme"}
V = ["native", "naive", "mondrian_pred", "mondrian_oracle"]

print("empirical coverage of the nominal 0.90 interval, by TRUE mechanism\n")
print(f"{'dataset':13s} {'acc':>5s} {'group':12s} {'n':>5s} " +
      " ".join(f"{v.replace('mondrian_','m-'):>10s}" for v in V) + f"  {'width x':>8s}")
agg = {v: {"in": [], "out": []} for v in V}
for ds, r in R.items():
    t = "in" if ds in INS else "out"
    for gi, g in enumerate(GRP):
        row = r["groups"].get(g)
        if not row:
            continue
        for v in V:
            agg[v][t].append(row[v]["coverage"])
        wx = row["mondrian_pred"]["width"] / max(row["native"]["width"], 1e-12)
        print(f"{(ds + (' *' if t=='in' else '')) if gi == 0 else '':13s} "
              f"{(f'{r[chr(100)+chr(101)+chr(116)+chr(101)+chr(99)+chr(116)+chr(111)+chr(114)+chr(95)+chr(97)+chr(99)+chr(99)]:.2f}') if gi == 0 else '':>5s} "
              f"{NAME.get(g, g):12s} {row['n']:5d} " +
              " ".join(f"{row[v]['coverage']:10.3f}" for v in V) + f"  {wx:7.2f}x")
print("\n* = one of the three datasets the detector was fitted on\n")
print(f"{'':26s}" + " ".join(f"{v.replace('mondrian_','m-'):>16s}" for v in V))
for t, lab in (("in", "in-sample (3 datasets)"), ("out", "held out (6 datasets)")):
    print(f"{lab:26s}" + " ".join(
        f"{np.mean(agg[v][t]):8.3f}+-{np.std(agg[v][t]):5.3f}" for v in V))
cens = [r["groups"][g]["mondrian_pred"]["coverage"] for r in R.values()
        for g in ("mnar_high", "mnar_extreme") if g in r["groups"]]
censn = [r["groups"][g]["native"]["coverage"] for r in R.values()
         for g in ("mnar_high", "mnar_extreme") if g in r["groups"]]
censv = [r["groups"][g]["naive"]["coverage"] for r in R.values()
         for g in ("mnar_high", "mnar_extreme") if g in r["groups"]]
print(f"\ncensoring groups only ({len(cens)} cells across all 9 datasets):")
print(f"  native   {np.mean(censn):.3f}  (min {min(censn):.3f})")
print(f"  1 pool   {np.mean(censv):.3f}  (min {min(censv):.3f})")
print(f"  mondrian {np.mean(cens):.3f}  (min {min(cens):.3f}); "
      f"within 0.03 of nominal in {sum(abs(c-0.9)<=0.03 for c in cens)}/{len(cens)}")
d = [abs(r["groups"][g]["mondrian_pred"]["coverage"] -
         r["groups"][g]["mondrian_oracle"]["coverage"])
     for r in R.values() for g in GRP if g in r["groups"]]
print(f"\npredicted vs oracle grouping: max |delta| {max(d):.4f}, mean {np.mean(d):.4f}")
print("detector accuracy: " + "  ".join(f"{k}={v['detector_acc']:.3f}" for k, v in R.items()))
