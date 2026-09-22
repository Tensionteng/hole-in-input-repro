#!/usr/bin/env python
"""Does the patch-DC share of an imputer's residual predict which interface wants it?

Joins S51's residual-structure statistics to S42's downstream outcome on the same cells.
Outcome = share of the declared path's excess that the restored path closes,
(native - dual) / (native - 1), exactly as analyze_s42.py defines it.

Two competing predictors of that outcome:
  rmse  -- imputation accuracy, the one-dimensional axis the paper already uses
  psi   -- patch-DC share, the structure statistic this round proposes
"""
import json
import os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)

S51 = json.load(open(os.path.join(HERE, "s51_results.json")))["cells"]
S42 = json.load(open(os.path.join(EXP, "s42_end2end", "s42_results.json")))["cells"]

rows = []
for k, v in S51.items():
    ds, mech, rate, fill = k.split("|")
    nat, du = S42.get(f"{k}|native"), S42.get(f"{k}|dual")
    if nat is None or du is None:
        continue
    excess = nat["median"] - 1.0
    if excess <= 0.02:                       # nothing for either path to close
        continue
    rows.append(dict(cell=k, ds=ds, mech=mech, rate=float(rate), fill=fill,
                     psi=v["psi_median"], ac1=v["ac1_median"], rmse=v["rmse_median"],
                     closed=(nat["median"] - du["median"]) / excess,
                     native=nat["median"], dual=du["median"]))

print(f"{len(rows)} cells with an excess worth closing\n")
print(f"{'cell':34s} {'rmse':>6s} {'psi':>6s} {'ac1':>7s} {'native':>7s} {'dual':>7s} {'closed':>8s}")
for r in sorted(rows, key=lambda r: -r["psi"]):
    print(f"{r['cell']:34s} {r['rmse']:6.3f} {r['psi']:6.3f} {r['ac1']:+7.3f} "
          f"{r['native']:7.3f} {r['dual']:7.3f} {100*r['closed']:7.1f}%")

y = np.array([r["closed"] for r in rows])
print("\n== which statistic predicts the outcome ==")
for name in ("rmse", "psi", "ac1"):
    x = np.array([r[name] for r in rows])
    ok = np.isfinite(x) & np.isfinite(y)
    sp, ps = stats.spearmanr(x[ok], y[ok])
    pe, pp = stats.pearsonr(x[ok], y[ok])
    print(f"  {name:5s} vs closure:  Spearman {sp:+.3f} (p={ps:.4f})   "
          f"Pearson {pe:+.3f} (p={pp:.4f})   n={ok.sum()}")

print("\n== controlling for what psi could be a proxy for ==")
for key in ("mech", "fill"):
    sps = []
    for g in sorted(set(r[key] for r in rows)):
        sub = [r for r in rows if r[key] == g]
        if len(sub) < 5:
            continue
        sp, p = stats.spearmanr([r["psi"] for r in sub], [r["closed"] for r in sub])
        sps.append(sp)
        print(f"  within {key}={g:11s} n={len(sub):2d}  Spearman(psi, closure) {sp:+.3f} (p={p:.3f})")
    if sps:
        print(f"  -> median within-{key} Spearman {np.median(sps):+.3f}")

print("\n== as a decision rule: restore the content path only when psi is low ==")
best = None
for thr in np.quantile([r["psi"] for r in rows], np.linspace(.2, .8, 13)):
    lo = [r["closed"] for r in rows if r["psi"] <= thr]
    hi = [r["closed"] for r in rows if r["psi"] > thr]
    if len(lo) < 4 or len(hi) < 4:
        continue
    gap = np.median(lo) - np.median(hi)
    if best is None or gap > best[1]:
        best = (thr, gap, len(lo), len(hi), np.median(lo), np.median(hi))
if best:
    thr, gap, nlo, nhi, mlo, mhi = best
    print(f"  psi <= {thr:.3f}: median closure {100*mlo:+.1f}% (n={nlo})")
    print(f"  psi >  {thr:.3f}: median closure {100*mhi:+.1f}% (n={nhi})")
    print(f"  separation {100*gap:+.1f} points")
    lo = [r["closed"] for r in rows if r["psi"] <= thr]
    hi = [r["closed"] for r in rows if r["psi"] > thr]
    u, p = stats.mannwhitneyu(lo, hi, alternative="greater")
    print(f"  Mann-Whitney one-sided p = {p:.4f}")


# ---------------------------------------------------------------------------
# The confound is obvious in the sorted table: psi is near 1 under censoring and
# near 0.3 under scattered dropout, so a pooled correlation could be re-discovering
# the mechanism. Two tests that cannot.
print("\n== test 1: does psi add anything to the mechanism label? ==")
z = []
for m in sorted(set(r["mech"] for r in rows)):
    sub = [r for r in rows if r["mech"] == m]
    p = np.array([r["psi"] for r in sub])
    thr = np.median(p)
    lo = [r["closed"] for r in sub if r["psi"] <= thr]
    hi = [r["closed"] for r in sub if r["psi"] > thr]
    print(f"  {m:13s} psi<=med: {100*np.median(lo):+7.1f}%   psi>med: {100*np.median(hi):+7.1f}%"
          f"   gap {100*(np.median(lo)-np.median(hi)):+6.1f} pts   (n={len(sub)})")
    z += [(r["closed"], r["psi"] <= thr) for r in sub]
lo = [c for c, is_lo in z if is_lo]
hi = [c for c, is_lo in z if not is_lo]
u, p = stats.mannwhitneyu(lo, hi, alternative="greater")
print(f"  pooled within-mechanism split: {100*np.median(lo):+.1f}% vs {100*np.median(hi):+.1f}%"
      f"  (Mann-Whitney one-sided p = {p:.4f}, n={len(lo)}+{len(hi)})")

print("\n== test 2: the BRITS-vs-SAITS puzzle of Section 6.1, cell by cell ==")
print("  Section 6.1 reports that BRITS reaches an effective alpha up to 0.8 by imputation")
print("  error yet makes the restored path worse. If psi is the missing axis, then in the")
print("  cells where BRITS loses to SAITS on closure it should also carry the higher psi.")
agree = tot = 0
by = {}
for r in rows:
    by[(r["ds"], r["mech"], r["rate"], r["fill"])] = r
for (ds, mech, rate, f), r in sorted(by.items()):
    if not f.startswith("brits"):
        continue
    s = by.get((ds, mech, rate, f.replace("brits", "saits")))
    if s is None:
        continue
    tot += 1
    dpsi, dclo = r["psi"] - s["psi"], r["closed"] - s["closed"]
    agree += (dpsi > 0) == (dclo < 0)
print(f"  psi correctly ordered the pair in {agree}/{tot} matched BRITS-vs-SAITS cells")
b = stats.binomtest(agree, tot, 0.5, alternative="greater")
print(f"  binomial one-sided p = {b.pvalue:.4f}")
d_psi = [by[(d, m, rt, f)]["psi"] - by[(d, m, rt, f.replace("brits", "saits"))]["psi"]
         for (d, m, rt, f) in by if f.startswith("brits")
         and (d, m, rt, f.replace("brits", "saits")) in by]
d_rmse = [by[(d, m, rt, f)]["rmse"] - by[(d, m, rt, f.replace("brits", "saits"))]["rmse"]
          for (d, m, rt, f) in by if f.startswith("brits")
          and (d, m, rt, f.replace("brits", "saits")) in by]
print(f"  BRITS - SAITS: median dpsi {np.median(d_psi):+.3f}, median drmse {np.median(d_rmse):+.3f}")
print("  (a positive dpsi with a non-positive drmse is the signature the alpha axis misses)")
