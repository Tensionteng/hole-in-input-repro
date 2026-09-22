#!/usr/bin/env python
"""S70 analysis: P1/P2/P3 verdicts from s70_rcprobe.json.

Pooling: cell medians over the 72 (ds, ch, mech, rate) strata per
(model, q_r, q_c). All definitions follow DESIGN.md verbatim:

  P1  vanilla relMSE spread across q_r {zero, linear, oracle}: per stratum
      (max - min) / mean over the three fills; pooled spread on pooled medians.
      Supported iff < 5%.
  P2  q_c effect e(model, q_r) = median(clean) - median(corrupt) relMSE points
      (clean minus corrupt, as pre-registered; typically negative since
      corruption hurts). Substitution: e(cpt, zero) / e(cpt, oracle) >= 2;
      vanilla and ctl5k ratio ~ 1.
  P3  median(cpt, zero, clean) vs median(cpt, oracle, clean) within 10%;
      vanilla at (zero, clean) NOT within 10% of (oracle, clean).
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "s70_rcprobe.json")

Q_R = ("zero", "linear", "oracle")
Q_C = ("clean", "corrupt")
MODELS = ("vanilla", "ctl5k", "cpt")


def load():
    r = json.load(open(RES))
    cell = defaultdict(dict)
    for k, c in r["cells"].items():
        ds, ch, mech, rate, q_r, q_c, model = k.split("|")
        cell[(ds, ch, mech, rate)][(q_r, q_c, model)] = c["median"]
    return r, cell


def pooled(cell):
    """med[(model, q_r, q_c)] = median over strata of cell medians."""
    acc = defaultdict(list)
    for strata, d in cell.items():
        for key, v in d.items():
            acc[key].append(v)
    return {k: float(np.median(v)) for k, v in acc.items()}


def main():
    res, cell = load()
    med = pooled(cell)
    out = {"pooled_median": {f"{m}|{qr}|{qc}": med[(qr, qc, m)]
                             for m in MODELS for qr in Q_R for qc in Q_C}}

    # ---- P1: vanilla flat in q_r -----------------------------------------
    spreads = []
    for strata, d in cell.items():
        for qc in Q_C:
            v = [d[(qr, qc, "vanilla")] for qr in Q_R]
            spreads.append((max(v) - min(v)) / np.mean(v))
    pooled_spread = {}
    for qc in Q_C:
        v = [med[(qr, qc, "vanilla")] for qr in Q_R]
        pooled_spread[qc] = (max(v) - min(v)) / np.mean(v)
    p1 = {"stratum_spread_median": float(np.median(spreads)),
          "stratum_spread_max": float(np.max(spreads)),
          "pooled_spread": pooled_spread,
          "pass": bool(max(pooled_spread.values()) < 0.05)}
    out["P1"] = p1

    # ---- P2: substitution -------------------------------------------------
    eff = {m: {qr: med[(qr, "clean", m)] - med[(qr, "corrupt", m)]
               for qr in Q_R} for m in MODELS}
    ratio = {m: (eff[m]["zero"] / eff[m]["oracle"]
                 if abs(eff[m]["oracle"]) > 1e-12 else float("nan"))
             for m in MODELS}
    p2 = {"q_c_effect": eff, "ratio_zero_over_oracle": ratio,
          "pass": bool(ratio["cpt"] >= 2.0 and abs(ratio["vanilla"] - 1) < 0.5
                       and abs(ratio["ctl5k"] - 1) < 0.5)}
    out["P2"] = p2

    # ---- P3: garbage repair ignored ---------------------------------------
    cpt_ratio = med[("zero", "clean", "cpt")] / med[("oracle", "clean", "cpt")]
    van_ratio = med[("zero", "clean", "vanilla")] / med[("oracle", "clean", "vanilla")]
    ctl_ratio = med[("zero", "clean", "ctl5k")] / med[("oracle", "clean", "ctl5k")]
    p3 = {"cpt_zero_over_oracle_clean": cpt_ratio,
          "vanilla_zero_over_oracle_clean": van_ratio,
          "ctl5k_zero_over_oracle_clean": ctl_ratio,
          "pass": bool(abs(cpt_ratio - 1) <= 0.10 and abs(van_ratio - 1) > 0.10)}
    out["P3"] = p3

    # ---- per-dataset breakdown of the pooled table ------------------------
    per_ds = {}
    for ds in ("weather", "ETTh1", "ETTm1"):
        sub = {s: d for s, d in cell.items() if s[0] == ds}
        m_ds = pooled(sub)
        per_ds[ds] = {f"{m}|{qr}|{qc}": m_ds[(qr, qc, m)]
                      for m in MODELS for qr in Q_R for qc in Q_C}
    out["per_dataset_median"] = per_ds

    json.dump(out, open(os.path.join(HERE, "s70_verdicts.json"), "w"), indent=1)

    f = lambda m, qr, qc: med[(qr, qc, m)]
    print("== pooled median relMSE (72 strata) ==")
    print(f"{'model':8s} {'q_r':7s} {'clean':>7s} {'corrupt':>8s}")
    for m in MODELS:
        for qr in Q_R:
            print(f"{m:8s} {qr:7s} {f(m, qr, 'clean'):7.3f} "
                  f"{f(m, qr, 'corrupt'):8.3f}")
    print("\nP1 vanilla q_r spread: pooled clean %.3f corrupt %.3f "
          "(strata median %.3f, max %.3f) -> %s"
          % (pooled_spread["clean"], pooled_spread["corrupt"],
             np.median(spreads), np.max(spreads),
             "PASS (<5%)" if p1["pass"] else "FAIL (>=5%)"))
    print("P2 q_c effect (clean - corrupt):")
    for m in MODELS:
        print("   %-7s zero %+.3f  linear %+.3f  oracle %+.3f  ratio %.2f"
              % (m, eff[m]["zero"], eff[m]["linear"], eff[m]["oracle"],
                 ratio[m]))
    print("   -> %s" % ("PASS (cpt ratio >= 2, others ~1)" if p2["pass"]
                         else "FAIL"))
    print("P3 (zero, clean) / (oracle, clean): cpt %.3f (%s 10%%), "
          "vanilla %.3f (%s 10%%), ctl5k %.3f -> %s"
          % (cpt_ratio, "within" if abs(cpt_ratio - 1) <= .1 else "outside",
             van_ratio, "within" if abs(van_ratio - 1) <= .1 else "outside",
             ctl_ratio, "PASS" if p3["pass"] else "FAIL"))


if __name__ == "__main__":
    main()
