#!/usr/bin/env python
"""S37 scorecard: closure by dataset and mechanism, in-sample vs held out."""
import json, os, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in ("own_insample.json", "own_held_a.json", "own_held_b.json", "own_held_c.json"):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        d = json.load(open(p))
        for k in ("closure", "sweep", "clean_own", "released_norm"):
            R.setdefault(k, {}).update(d.get(k, {}))
DS = ["ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
      "exchange", "illness"]
INS = ("ETTh1", "ETTm1", "weather")
MECH = [("mcar", "scattered"), ("block", "block"), ("mnar_high", "censoring"),
        ("mnar_extreme", "extreme")]
RATES = (0.3, 0.7)

MIN_EXCESS = 0.02   # below this the declared path has essentially nothing to close and the
                    # ratio is dominated by its denominator; those cells are reported as n/a
print("closure = (declared - restored) / (declared - 1) at alpha=1, own-clean normalised")
print(f"cells where the declared path's own excess is < {MIN_EXCESS} are n/a: "
      "nothing to close\n")
hdr = f"{'dataset':12s} {'':9s}" + "".join(f"{n:>11s}" for _, n in MECH)
print(hdr); print("-" * len(hdr))
allv = {"in": [], "out": []}
for ds in DS:
    if f"{ds}|mcar|0.3" not in R["closure"]:
        continue
    tag = "in" if ds in INS else "out"
    for r in RATES:
        row = []
        for m, _ in MECH:
            c = R["closure"].get(f"{ds}|{m}|{r}")
            v = (c["closure"] if (c and c["native_a1"] - 1.0 >= MIN_EXCESS)
                 else float("nan"))
            row.append(v)
            if np.isfinite(v):
                allv[tag].append(v)
        print(f"{ds if r == RATES[0] else '':12s} {'p=' + str(r):9s}" +
              "".join((f"{v*100:10.1f}%" if np.isfinite(v) else f"{'n/a':>11s}")
                      for v in row))
print()
for t, lab in (("in", "in-sample (3 datasets)"), ("out", "held out (6 datasets)")):
    v = np.array(allv[t])
    print(f"{lab:24s} n={len(v):3d}  positive {int((v>0).sum())}/{len(v)}  "
          f"median {np.median(v)*100:5.1f}%  IQR [{np.percentile(v,25)*100:.1f},"
          f"{np.percentile(v,75)*100:.1f}]%  min {v.min()*100:.1f}%")

print("\nalpha=0 control (restored must not be worse than declared when the imputer is poor)")
w = {"in": [0, 0], "out": [0, 0]}
for k, c in R["closure"].items():
    ds = k.split("|")[0]
    t = "in" if ds in INS else "out"
    if c["native_a1"] - 1.0 < MIN_EXCESS:
        continue
    w[t][1] += 1
    w[t][0] += c["restored_a0"] <= c["native_a0"] + 1e-9
for t, lab in (("in", "in-sample"), ("out", "held out")):
    print(f"  {lab:12s} restored no worse in {w[t][0]}/{w[t][1]} cells")

print("\nclean-context control (adapted vs released weights, no missingness at all)")
for ds in DS:
    c = R.get("clean_own", {}).get(ds)
    if c:
        print(f"  {ds:12s} " + "  ".join(f"{i}={c[i]:.4f}" for i in c))
