#!/usr/bin/env python
"""S41 scorecard: what changes when the metric is MAE instead of MSE."""
import glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for f in sorted(glob.glob(os.path.join(HERE, "part_*.json"))):
    d = json.load(open(f))
    for k in ("grid", "sweep", "closure", "probe"):
        R.setdefault(k, {}).update(d.get(k, {}))
DS = ["ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
      "exchange", "illness"]
DS = [d for d in DS if f"{d}|mcar|0.3" in R.get("grid", {})]
INS = ("ETTh1", "ETTm1", "weather")
MECH = [("mcar", "scattered"), ("block", "block"), ("mnar_high", "censoring"),
        ("mnar_extreme", "extreme")]
MIN = 0.02

print(f"A. damage grid at rate 0.7 ({len(DS)} datasets)\n")
print(f"{'dataset':13s}" + "".join(f"{n:>21s}" for _, n in MECH))
print(f"{'':13s}" + "".join(f"{'relMSE':>10s}{'relMAE':>11s}" for _ in MECH))
order_mse = order_mae = 0
for d in DS:
    row = []
    for m, _ in MECH:
        c = R["grid"][f"{d}|{m}|0.7"]
        row += [c["mse"]["median"], c["mae"]["median"]]
    order_mse += (np.argmax(row[0::2]) == 2)
    order_mae += (np.argmax(row[1::2]) == 2)
    print(f"{d:13s}" + "".join(f"{v:10.2f}{row[i*2+1]:11.2f}" if i * 2 == j else ""
                               for i, v in enumerate(row[0::2]) for j in [i * 2]))
print(f"\ncensoring is the worst cell:  under MSE {order_mse}/{len(DS)}   "
      f"under MAE {order_mae}/{len(DS)}")
ex = {"mse": [], "mae": []}
for d in DS:
    for met in ex:
        ex[met].append(R["grid"][f"{d}|mnar_high|0.7"][met]["median"] - 1)
print(f"censoring excess at rate 0.7, median over datasets: "
      f"MSE {np.median(ex['mse']):.3f}   MAE {np.median(ex['mae']):.3f}   "
      f"ratio {np.median(ex['mae'])/np.median(ex['mse']):.2f}")
for m, n in MECH:
    e = [(R["grid"][f"{d}|{m}|0.7"]["mae"]["median"] - 1) /
         max(R["grid"][f"{d}|{m}|0.7"]["mse"]["median"] - 1, 1e-9) for d in DS
         if R["grid"][f"{d}|{m}|0.7"]["mse"]["median"] - 1 > 0.05]
    if e:
        print(f"  {n:11s} MAE excess / MSE excess: median {np.median(e):.2f} (n={len(e)})")

print(f"\nB. closure, MSE vs MAE\n")
print(f"{'dataset':13s} {'':7s}" + "".join(f"{n:>19s}" for _, n in MECH))
agg = {"mse": {"in": [], "out": []}, "mae": {"in": [], "out": []}}
flips = []
for d in DS:
    for r in (0.3, 0.7):
        line = f"{d if r == 0.3 else '':13s} p={r:<4}"
        for m, _ in MECH:
            c = R["closure"][f"{d}|{m}|{r}"]
            v2, v1 = c["mse"]["closure"], c["mae"]["closure"]
            skip2 = c["mse"]["native_a1"] - 1 < MIN
            skip1 = c["mae"]["native_a1"] - 1 < MIN
            if not skip2 and np.isfinite(v2):
                agg["mse"]["in" if d in INS else "out"].append(v2)
            if not skip1 and np.isfinite(v1):
                agg["mae"]["in" if d in INS else "out"].append(v1)
            if not skip2 and not skip1 and np.isfinite(v2) and np.isfinite(v1) \
                    and (v2 > 0) != (v1 > 0):
                flips.append((d, m, r, v2, v1))
            f2 = f"{v2*100:6.0f}" if not skip2 and np.isfinite(v2) else "   n/a"
            f1 = f"{v1*100:6.0f}" if not skip1 and np.isfinite(v1) else "   n/a"
            line += f"  {f2}/{f1}%"
        print(line)
for t in ("in", "out"):
    a, b = agg["mse"][t], agg["mae"][t]
    print(f"\n{t:>3}-sample: MSE median {np.median(a)*100:5.1f}% (n={len(a)}, "
          f"{sum(v>0 for v in a)} positive)   "
          f"MAE median {np.median(b)*100:5.1f}% (n={len(b)}, {sum(v>0 for v in b)} positive)")
print(f"\ncells where the SIGN of closure flips between metrics: {len(flips)}")
for f in flips:
    print(f"   {f[0]:12s} {f[1]:13s} p={f[2]}  MSE {f[3]*100:+.1f}%  MAE {f[4]*100:+.1f}%")

if "probe" in R:
    print("\nC. the permutation probe under an L1 output norm")
    for d, v in R["probe"].items():
        print(f"  {d:13s} " + "  ".join(
            f"{c}: rms={v[c]['rms']:.4f} l1={v[c]['l1']:.4f}" for c in v))
    dif = [abs(v[c]["rms"] - v[c]["l1"]) for v in R["probe"].values() for c in v]
    print(f"  max |rms - l1| over all cells: {max(dif):.4f}")
