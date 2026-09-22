#!/usr/bin/env python
"""Alpha breakeven: at what imputation quality does the restored interface start beating
the declared path?

Re-analysis of the S37 sweep (no new model runs). For each (dataset, mechanism, rate) cell
we take the own-clean-normalised median relMSE of the declared path (native) and the
restored path (dual) at alpha in {0, .25, .5, .75, 1}; the declared path is ~flat in alpha
(the ceiling result), the restored path improves with alpha. The breakeven alpha* is the
first crossing point (linear interpolation between sampled alphas).

alpha* = 0   -> restored already wins with a linear fill
alpha* > 1   -> restored does not catch up even at the oracle
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in ("own_insample.json", "own_held_a.json", "own_held_b.json", "own_held_c.json"):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        d = json.load(open(p))
        R.update(d.get("sweep", {}))

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
INS = ("ETTh1", "ETTm1", "weather")
MECHS = (("mcar", "scattered"), ("block", "block"),
         ("mnar_high", "censoring"), ("mnar_extreme", "extreme"))
RATES = ("0.3", "0.7")
DS = ["ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
      "exchange", "illness"]


def series(ds, mech, rate, iface):
    out = []
    for a in ALPHAS:
        c = R.get(f"{ds}|{mech}|{rate}|{a}|{iface}")
        out.append(np.nan if c is None else c["median_own"])
    return np.array(out, dtype=float)


def series_min(ds, mech, rate):
    """restored = min(dual, dual_obsnorm): the convention of the paper's alpha=0 control."""
    a = series(ds, mech, rate, "dual")
    b = series(ds, mech, rate, "dual_obsnorm")
    return np.fmin(a, b)


def breakeven(native, dual):
    """First alpha where dual <= native, linear-interpolated; nan if never by alpha=1."""
    if np.any(np.isnan(native)) or np.any(np.isnan(dual)):
        return np.nan
    d = dual - native
    if d[0] <= 0:
        return 0.0
    for i in range(1, len(ALPHAS)):
        if d[i] <= 0:
            # interpolate between i-1 (positive) and i (non-positive)
            w = d[i - 1] / (d[i - 1] - d[i]) if d[i] != d[i - 1] else 0.0
            return ALPHAS[i - 1] + w * (ALPHAS[i] - ALPHAS[i - 1])
    return np.inf


rows = []
for ds in DS:
    for mech, mname in MECHS:
        for rate in RATES:
            nat = series(ds, mech, rate, "native")
            du = series_min(ds, mech, rate)
            be = breakeven(nat, du)
            rows.append(dict(ds=ds, mech=mech, mname=mname, rate=rate, be=be,
                             native0=nat[0], native1=nat[-1],
                             dual0=du[0], dual1=du[-1],
                             insample=ds in INS,
                             closeable=nat[-1] - 1.0 >= 0.02))

print(f"{'dataset':12s}{'mechanism':11s}{'rate':5s}{'native(a=0)':>12s}{'dual(a=0)':>11s}"
      f"{'dual(a=1)':>11s}{'alpha*':>9s}")
for r in rows:
    be = r["be"]
    s = "0" if be == 0 else (">1" if np.isinf(be) else
                             ("n/a" if np.isnan(be) else f"{be:.2f}"))
    print(f"{r['ds']:12s}{r['mname']:11s}{r['rate']:5s}{r['native0']:12.3f}"
          f"{r['dual0']:11.3f}{r['dual1']:11.3f}{s:>9s}")

print("\n== summary by mechanism (closeable cells only: declared-path excess >= 0.02) ==")
for mech, mname in MECHS:
    for grp, lab in ((True, "in-sample"), (False, "held-out")):
        v = [r["be"] for r in rows
             if r["mech"] == mech and r["insample"] == grp and r["closeable"]]
        v = [x for x in v if not np.isnan(x)]
        if not v:
            continue
        v = np.array(v)
        fin = v[np.isfinite(v)]
        line = (f"{mname:10s} {lab:9s} n={len(v):2d}  a*=0: {int((v==0).sum()):2d}  "
                f">1: {int(np.isinf(v).sum()):2d}")
        if len(fin):
            line += (f"  finite median {np.median(fin):.2f} "
                     f"IQR [{np.percentile(fin,25):.2f},{np.percentile(fin,75):.2f}]")
        print(line)

print("\n== pooled (closeable cells only) ==")
tot = {"in": [], "out": []}
for grp, lab in ((True, "in-sample"), (False, "held-out")):
    v = np.array([r["be"] for r in rows if r["insample"] == grp and r["closeable"]
                  and not np.isnan(r["be"])])
    tot["in" if grp else "out"] = v
    fin = v[np.isfinite(v)]
    print(f"{lab:9s} n={len(v):2d}  a*=0: {int((v==0).sum()):2d}  "
          f"in (0,0.25]: {int(((v>0)&(v<=0.25)).sum()):2d}  "
          f"in (0.25,0.5]: {int(((v>0.25)&(v<=0.5)).sum()):2d}  "
          f"in (0.5,1]: {int(((v>0.5)&(v<=1)).sum()):2d}  >1: {int(np.isinf(v).sum()):2d}"
          f"  finite median {np.median(fin):.2f}")
v = np.concatenate([tot["in"], tot["out"]])
fin = v[np.isfinite(v)]
print(f"{'ALL':9s} n={len(v):2d}  a*=0: {int((v==0).sum()):2d}  "
      f"in (0,0.25]: {int(((v>0)&(v<=0.25)).sum()):2d}  "
      f"in (0.25,0.5]: {int(((v>0.25)&(v<=0.5)).sum()):2d}  "
      f"in (0.5,1]: {int(((v>0.5)&(v<=1)).sum()):2d}  >1: {int(np.isinf(v).sum()):2d}"
      f"  finite median {np.median(fin):.2f}")
