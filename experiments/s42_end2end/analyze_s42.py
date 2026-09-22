#!/usr/bin/env python
"""S42 scorecard: declared vs restored under REAL imputers.

Per (dataset, mechanism, rate) cell and per fill, reports the own-clean median relMSE of
native / dual / dual_obsnorm and the paired win share of the restored paths vs native.
Then the headline summaries:
  - for each fill, in how many cells does dual beat native (median), and the median win share
  - where real imputers sit on the alpha axis (imputation MSE vs linear and oracle)
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "s42_results.json")))
cells, imp = d["cells"], d["imp_mse_std"]
DS = ("ETTh1", "ETTm1", "weather")
MECHS = (("mcar", "scattered"), ("block", "block"),
         ("mnar_high", "censoring"), ("mnar_extreme", "extreme"))
RATES = ("0.3", "0.7")
FILLS = ("linear", "saits_mb", "saits_all", "brits_mb", "brits_all", "oracle")

print("== per-cell medians (own-clean relMSE) ==")
hdr = f"{'dataset':9s}{'mech':10s}{'p':5s}{'fill':10s}" + \
      "".join(f"{i:>14s}" for i in ("native", "dual", "dual_obsnorm")) + \
      f"{'dual wins%':>11s}"
print(hdr)
for ds in DS:
    for mech, mname in MECHS:
        for rate in RATES:
            for fill in FILLS:
                k = f"{ds}|{mech}|{rate}|{fill}"
                row = [cells[f"{k}|{i}"]["median"]
                       for i in ("native", "dual", "dual_obsnorm")]
                w = cells[f"{k}|dual"].get("winshare_vs_native", float("nan"))
                print(f"{ds:9s}{mname:10s}{rate:5s}{fill:10s}" +
                      "".join(f"{v:14.3f}" for v in row) + f"{100*w:10.0f}%")
            print()

print("== headline: does restored beat declared under each fill? ==")
for fill in FILLS:
    wins, shares, marg = 0, [], []
    n = 0
    for ds in DS:
        for mech, _ in MECHS:
            for rate in RATES:
                k = f"{ds}|{mech}|{rate}|{fill}"
                nat, du = cells[f"{k}|native"]["median"], cells[f"{k}|dual"]["median"]
                n += 1
                wins += du < nat
                shares.append(cells[f"{k}|dual"]["winshare_vs_native"])
                marg.append((nat - du) / max(nat - 1.0, 1e-9))
    print(f"  {fill:10s} dual<native in {wins:2d}/{n} cells   "
          f"median paired win share {100*np.median(shares):5.1f}%   "
          f"median excess closed {100*np.median(marg):5.1f}%")

print("\n== where real imputers sit on the alpha axis ==")
print("(imputation MSE on masked positions, standardized; "
      "alpha_eff = 1 - mse(fill)/mse(linear))")
for ds in DS:
    for mech, mname in MECHS:
        for rate in RATES:
            base = imp[f"{ds}|{mech}|{rate}|linear"]
            line = f"  {ds:8s} {mname:10s} p={rate}  linear={base:.3f}"
            for fill in ("saits_mb", "saits_all", "brits_mb", "brits_all"):
                m = imp[f"{ds}|{mech}|{rate}|{fill}"]
                line += f"  {fill}={m:.3f}(a~{1 - m / base:+.2f})"
            print(line)
