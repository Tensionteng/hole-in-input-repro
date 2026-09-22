#!/usr/bin/env python
"""S45 scorecard: the pre-registered predictions of DESIGN.md against the eval results.

Note on arm B: the filtered corpus makes the interface inert during training (on clean
windows native and dual are the same operation), so B is expected to be bit-identical to A
-- a design-validation check, not a factorial cell with information. The interface x
augmentation separation is therefore read from D vs A (augmentation alone) and C vs A
(interface + augmentation), and P4 is reported as partially vacuous by construction.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "s45_eval.json")))
DS = ("ETTh1", "ETTm1", "weather")
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES = (0.3, 0.7)
SW = R["sweep"]
PR = R["probe"]
AT = R["attack"]
CL = R["clean"]

ok = lambda pred, name: print(f"  [{'CONFIRMED' if pred else 'FALSIFIED'}] {name}")


def sw(ds, mech, rate, a, arm, view="declared"):
    return SW[f"{ds}|{mech}|{rate}|{a}|{arm}|{view}"]["median"]


def cells(mechs=MECHS, rates=RATES, datasets=DS):
    for ds in datasets:
        for m in mechs:
            for r in rates:
                yield ds, m, r


print("== design-validation: is arm B bit-identical to arm A? ==")
same = all(abs(CL["A"][d]["mse_median"] - CL["B"][d]["mse_median"]) < 1e-6 for d in CL["A"])
print(f"  clean MSE identical on all datasets: {same}")
rho_B_diff = max(abs(PR[f"{ds}|{m}|B|{c}"]["rho"] - PR[f"{ds}|{m}|A|{c}"]["rho"])
                 for ds in DS for m in ("mcar", "block", "mnar_high") for c in ("plain",))
print(f"  max |rho_B - rho_A| on plain: {rho_B_diff:.3f}")

print("\n== P1 (replica): arm A behaves like stock bolt ==")
p1a = max(PR[f"{ds}|{m}|A|declared"]["rho"] for ds in DS
          for m in ("mcar", "block", "mnar_high")) < 0.01
ok(p1a, "P1a: A declared rho ~ 0")
slopes = [sw(ds, m, r, 1.0, "A") - sw(ds, m, r, 0.0, "A")
          for ds, m, r in cells(mechs=("mcar", "block"))]
p1b = np.median(np.abs(slopes)) < 0.05
ok(p1b, f"P1b: A alpha sweep flat (median |slope| {np.median(np.abs(slopes)):.3f})")

print("\n== P2 (prescription): arm C reads content and uses it ==")
rhoC = np.median([PR[f"{ds}|{m}|C|declared"]["rho"] for ds in DS
                  for m in ("mcar", "block")])
ok(rhoC >= 0.5, f"P2a: C declared rho >= 0.5 (median {rhoC:.2f})")
closures = []
for ds, m, r in cells():
    excA = sw(ds, m, r, 1.0, "A") - 1.0
    if excA < 0.02:
        continue
    cl = (sw(ds, m, r, 1.0, "A") - sw(ds, m, r, 1.0, "C")) / excA
    closures.append(cl)
ok(np.median(closures) >= 0.5,
   f"P2b: C closes >= 50% of A's untouchable excess at alpha=1 (median {np.median(closures)*100:.0f}%, n={len(closures)})")

print("\n== P3 (floor ~ declaration): C no worse than A at alpha=0 ==")
wins = [sw(ds, m, r, 0.0, "C") <= sw(ds, m, r, 0.0, "A") + 1e-9
        for ds, m, r in cells()]
ok(np.mean(wins) >= 0.6, f"P3: C <= A at alpha=0 in {int(np.sum(wins))}/{len(wins)} cells")

print("\n== P4 (factor separation; reported with the B-vacuity caveat) ==")
d_slopes = [sw(ds, m, r, 1.0, "D") - sw(ds, m, r, 0.0, "D")
            for ds, m, r in cells(mechs=("mcar", "block"))]
ok(np.median(np.abs(d_slopes)) < 0.05,
   f"P4a: D flat in alpha (median |slope| {np.median(np.abs(d_slopes)):.3f})")
c_better_a1 = np.mean([sw(ds, m, r, 1.0, "C") < sw(ds, m, r, 1.0, "A")
                       for ds, m, r in cells()])
c_better_a0 = np.mean([sw(ds, m, r, 0.0, "C") < sw(ds, m, r, 0.0, "A")
                       for ds, m, r in cells()])
print(f"  C<A at a=1 in {c_better_a1*100:.0f}% of cells; C<A at a=0 in {c_better_a0*100:.0f}%")

print("\n== P5 (no clean cost): C clean within 5% of A ==")
ratios = []
for d in CL["A"]:
    a, c = CL["A"][d]["mse_median"], CL["C"][d]["mse_median"]
    ratios.append(c / max(a, 1e-12))
ok(np.median(ratios) <= 1.05,
   f"P5: median clean ratio C/A = {np.median(ratios):.3f} (range {min(ratios):.3f}-{max(ratios):.3f})")

print("\n== P6 (trade-off returns): C's declared path is attackable ==")
ratiosA = [AT[f"{ds}|{m}|A"]["ratio_median"] for ds in DS for m in ("block", "mnar_high")]
ratiosC = [AT[f"{ds}|{m}|C"]["ratio_median"] for ds in DS for m in ("block", "mnar_high")]
ok(np.median(ratiosC) > np.median(ratiosA) + 0.05,
   f"P6: attack damage C (median x{np.median(ratiosC):.2f}) > A (x{np.median(ratiosA):.2f})")

print("\n== bonus: E (Moirai-2.0 recipe) vs C (mechanism-diverse) ==")
clE, clC = [], []
for ds, m, r in cells():
    excA = sw(ds, m, r, 1.0, "A") - 1.0
    if excA < 0.02:
        continue
    clE.append((sw(ds, m, r, 1.0, "A") - sw(ds, m, r, 1.0, "E")) / excA)
    clC.append((sw(ds, m, r, 1.0, "A") - sw(ds, m, r, 1.0, "C")) / excA)
print(f"  closure at a=1: E {np.median(clE)*100:.0f}% vs C {np.median(clC)*100:.0f}% "
      f"(mechanism diversity delta {(np.median(clC)-np.median(clE))*100:+.0f} pts)")
winsE = [sw(ds, m, r, 0.0, "E") <= sw(ds, m, r, 0.0, "A") + 1e-9 for ds, m, r in cells()]
print(f"  E <= A at a=0 in {int(np.sum(winsE))}/{len(winsE)} cells "
      f"(C: {int(np.sum(wins))})")
