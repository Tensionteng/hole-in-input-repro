#!/usr/bin/env python
"""S48: seed spread and same-size ladder for the pretraining round.

Two things the stage-1/stage-2 scorecards could not report:

  (1) Every arm was a single seed, so "H closes 92% where C closes 55%" carried no
      uncertainty at all. Three seeds per arm at tiny turn that into a range.
  (2) H@48M's clean ratio and closure were computed against arm A at TINY, so they
      conflated recipe with capacity. A and C are now trained at 48M and 205M as well,
      and every ratio below uses a same-size denominator.

Closure of an arm against its reference, per cell (dataset x mech x rate):
    (ref_declared(alpha=1) - arm_declared(alpha=1)) / (ref_declared(alpha=1) - 1)
matching analyze_s45.py; cells whose reference excess is <= 0.02 are not closeable.
Bootstrap is paired on windows, using the per-window ratios eval_s45.py now stores.
"""
import argparse
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# the scorecard's sweep grid is three datasets (analyze_s45.py); the clean-accuracy
# ratio is taken over all nine, so the two are kept separate here
DS3 = ("ETTh1", "ETTm1", "weather")
DS9 = ("ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness")
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES = (0.3, 0.7)
RNG = np.random.default_rng(0)


def load(paths):
    """Merge several eval JSONs; later files win on key collisions."""
    out = {"clean": {}, "sweep": {}, "attack": {}, "probe": {}}
    for p in paths:
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        for sec in out:
            out[sec].update(d.get(sec, {}))
    return out


def cells(E, arm, ref, alpha="1.0"):
    """Per-cell (ref_excess, arm_value) with the per-window vectors when present."""
    got = []
    for ds in DS3:
        for m in MECHS:
            for r in RATES:
                k = f"{ds}|{m}|{r}|{alpha}"
                a, b = E["sweep"].get(f"{k}|{arm}|declared"), E["sweep"].get(f"{k}|{ref}|declared")
                if a is None or b is None:
                    continue
                got.append((f"{ds}|{m}|{r}", b, a))
    return got


def closure(E, arm, ref, alpha="1.0"):
    vals = []
    for _, b, a in cells(E, arm, ref, alpha):
        ex = b["median"] - 1.0
        if ex <= 0.02:
            continue
        vals.append((b["median"] - a["median"]) / ex)
    return np.median(vals) if vals else float("nan"), len(vals)


def closure_boot(E, arm, ref, alpha="1.0", B=2000):
    """Paired-on-windows bootstrap of the median closure across closeable cells."""
    pairs = []
    for _, b, a in cells(E, arm, ref, alpha):
        if b["median"] - 1.0 <= 0.02 or "w" not in b or "w" not in a:
            continue
        wb, wa = np.asarray(b["w"]), np.asarray(a["w"])
        if len(wb) != len(wa):
            continue
        pairs.append((wb, wa))
    if not pairs:
        return None
    draws = np.empty(B)
    n = len(pairs[0][0])
    for i in range(B):
        idx = RNG.integers(0, n, n)
        per = []
        for wb, wa in pairs:
            mb, ma = np.median(wb[idx]), np.median(wa[idx])
            if mb - 1.0 > 0.02:
                per.append((mb - ma) / (mb - 1.0))
        if per:
            draws[i] = np.median(per)
        else:
            draws[i] = np.nan
    d = draws[np.isfinite(draws)]
    return float(np.quantile(d, .025)), float(np.quantile(d, .975)), len(pairs)


def floor_cells(E, arm, ref):
    """Cells of 24 (three ETT-family datasets are the scorecard's grid) where arm <= ref
    at alpha = 0, i.e. the arm is no worse than the reference given a poor fill."""
    n = ok = 0
    for _, b, a in cells(E, arm, ref, alpha="0.0"):
        n += 1
        ok += a["median"] <= b["median"] + 1e-9
    return ok, n


def clean_ratio(E, arm, ref):
    r = [E["clean"][arm][d]["mse_median"] / E["clean"][ref][d]["mse_median"]
         for d in DS9 if d in E["clean"].get(arm, {}) and d in E["clean"].get(ref, {})]
    return float(np.median(r)) if r else float("nan"), len(r)


def attack_median(E, arm):
    v = [x["ratio_median"] for k, x in E["attack"].items() if k.endswith(f"|{arm}")]
    return float(np.median(v)) if v else float("nan")


def rho_declared(E, arm):
    v = [x["rho"] for k, x in E["probe"].items() if k.endswith(f"|{arm}|declared")]
    return float(np.median(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--ref", default="A", help="reference arm (the status quo)")
    ap.add_argument("--arms", default="C,H")
    ap.add_argument("--seeds", default="", help="comma-separated replicate labels, e.g. 20260901,20260902")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    E = load(args.files)

    print(f"\n=== {args.label or 'ladder'} (reference = {args.ref}) ===")
    print(f"{'arm':16s}{'rho_dec':>9s}{'closure a=1':>13s}{'95% CI':>18s}"
          f"{'floor a=0':>11s}{'clean':>8s}{'attack':>9s}")
    labels = [args.ref] + args.arms.split(",")
    seeds = [s for s in args.seeds.split(",") if s]
    for base in labels:
        for sfx in [""] + [f"@{s}" for s in seeds]:
            arm = base + sfx
            if arm not in E["clean"]:
                continue
            ref = args.ref + sfx if (args.ref + sfx) in E["clean"] else args.ref
            if arm == ref:
                cl, _ = clean_ratio(E, arm, ref)
                print(f"{arm:16s}{rho_declared(E,arm):9.3f}{'---':>13s}{'---':>18s}"
                      f"{'---':>11s}{cl:8.3f}{attack_median(E,arm):8.2f}x")
                continue
            c, n = closure(E, arm, ref)
            ci = closure_boot(E, arm, ref)
            ok, tot = floor_cells(E, arm, ref)
            cr, _ = clean_ratio(E, arm, ref)
            cis = f"[{100*ci[0]:+.0f}, {100*ci[1]:+.0f}]%" if ci else "n/a"
            print(f"{arm:16s}{rho_declared(E,arm):9.3f}{100*c:12.1f}%{cis:>18s}"
                  f"{ok:8d}/{tot:<3d}{cr:8.3f}{attack_median(E,arm):8.2f}x")

    if seeds:
        print("\n-- seed spread (the quantity the single-seed scorecard could not report) --")
        for base in args.arms.split(","):
            vals = []
            for sfx in [""] + [f"@{s}" for s in seeds]:
                arm, ref = base + sfx, args.ref + (sfx if (args.ref + sfx) in E["clean"] else "")
                if arm in E["clean"]:
                    vals.append(closure(E, arm, ref)[0])
            if len(vals) > 1:
                print(f"  {base}: closure {100*np.mean(vals):.1f}% "
                      f"+- {100*np.std(vals, ddof=1):.1f} (n={len(vals)} seeds, "
                      f"range {100*min(vals):.1f}-{100*max(vals):.1f}%)")


if __name__ == "__main__":
    main()
