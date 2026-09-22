#!/usr/bin/env python
"""S70B analysis: P1b/P2b/P3b verdicts + operating-envelope comparison.

Pooling: cell medians over the 54 strata (18 ds-ch x 3 mech-rate) per
(model, q_r, q_c); q_c effect = corrupt - clean pooled median (positive =
context corruption hurts). Definitions as pre-registered in s70b_notes.md.
Envelope: cpt pooled medians at rate >= 0.9 vs its own s70 rate-0.7 cells
(s70_rcprobe.json, same q_r/q_c, mcar 0.7 and block 0.7 pooled over 18 strata).
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "s70b_rcprobe.json")
S70 = os.path.join(HERE, "s70_rcprobe.json")

Q_R = ("zero", "linear", "oracle")
Q_C = ("clean", "corrupt")
MODELS = ("vanilla", "ctl5k", "cpt")


def load_cells(path):
    r = json.load(open(path))
    cell = defaultdict(dict)
    for k, c in r["cells"].items():
        ds, ch, mech, rate, q_r, q_c, model = k.split("|")
        cell[(ds, ch, mech, rate)][(q_r, q_c, model)] = c["median"]
    return r, cell


def pooled(cell, keys=None):
    acc = defaultdict(list)
    for strata, d in cell.items():
        for key, v in d.items():
            if keys is None or strata[2:] in keys:
                acc[key].append(v)
    return {k: float(np.median(v)) for k, v in acc.items()}


def main():
    res, cell = load_cells(RES)
    med = pooled(cell)
    out = {"pooled_median": {f"{m}|{qr}|{qc}": med[(qr, qc, m)]
                             for m in MODELS for qr in Q_R for qc in Q_C}}

    eff = {m: {qr: med[(qr, "corrupt", m)] - med[(qr, "clean", m)]
               for qr in Q_R} for m in MODELS}
    ratio_cpt = (eff["cpt"]["zero"] / eff["cpt"]["oracle"]
                 if abs(eff["cpt"]["oracle"]) > 1e-12 else float("nan"))

    p1b = {"effect_cpt_zero": eff["cpt"]["zero"],
           "pass": bool(eff["cpt"]["zero"] >= 0.10)}
    p2b = {"effect_cpt_zero": eff["cpt"]["zero"],
           "effect_cpt_oracle": eff["cpt"]["oracle"],
           "ratio": ratio_cpt,
           "pass": bool(ratio_cpt >= 2.0)}
    v_zc, c_zc = med[("zero", "clean", "vanilla")], med[("zero", "clean", "cpt")]
    v_zx, c_zx = med[("zero", "corrupt", "vanilla")], med[("zero", "corrupt", "cpt")]
    p3b = {"vanilla_zero_clean": v_zc, "cpt_zero_clean": c_zc,
           "vanilla_zero_corrupt": v_zx, "cpt_zero_corrupt": c_zx,
           "ratio_clean": v_zc / c_zc, "ratio_corrupt": v_zx / c_zx,
           "vanilla_gap_zero": eff["vanilla"]["zero"],
           "pass": bool(v_zc / c_zc >= 1.5 and v_zx / c_zx >= 1.5
                        and eff["vanilla"]["zero"] >= 0.3)}
    out.update(P1b=p1b, P2b=p2b, P3b=p3b,
               q_c_effects={m: eff[m] for m in MODELS})

    # per (mech, rate) pooling
    per_mr = {}
    for mr in (("mcar", "0.9"), ("block", "0.9"), ("block", "1.0")):
        m_mr = pooled(cell, keys={mr})
        per_mr["|".join(mr)] = {
            f"{m}|{qr}|{qc}": m_mr[(qr, qc, m)]
            for m in MODELS for qr in Q_R for qc in Q_C}
    out["per_mechrate_median"] = per_mr

    # per-dataset pooling
    per_ds = {}
    for ds in ("weather", "ETTh1", "ETTm1"):
        sub = {s: d for s, d in cell.items() if s[0] == ds}
        m_ds = pooled(sub)
        per_ds[ds] = {f"{m}|{qr}|{qc}": m_ds[(qr, qc, m)]
                      for m in MODELS for qr in Q_R for qc in Q_C}
    out["per_dataset_median"] = per_ds

    # envelope: cpt here vs its s70 rate-0.7 cells
    _, c70 = load_cells(S70)
    env = {}
    for mr in (("mcar", "0.7"), ("block", "0.7")):
        m70 = pooled(c70, keys={mr})
        env["s70_" + "|".join(mr)] = {f"{qr}|{qc}": m70[(qr, qc, "cpt")]
                                      for qr in Q_R for qc in Q_C}
    env["s70b_pooled_cpt"] = {f"{qr}|{qc}": med[(qr, qc, "cpt")]
                              for qr in Q_R for qc in Q_C}
    for mr in (("mcar", "0.9"), ("block", "0.9"), ("block", "1.0")):
        m_mr = pooled(cell, keys={mr})
        env["s70b_" + "|".join(mr) + "_cpt"] = {
            f"{qr}|{qc}": m_mr[(qr, qc, "cpt")] for qr in Q_R for qc in Q_C}
    out["envelope"] = env

    json.dump(out, open(os.path.join(HERE, "s70b_verdicts.json"), "w"), indent=1)

    f = lambda m, qr, qc: med[(qr, qc, m)]
    print("== S70B pooled median relMSE (54 strata: rates 0.9/1.0) ==")
    print(f"{'model':8s} {'q_r':7s} {'clean':>7s} {'corrupt':>8s} {'eff':>7s}")
    for m in MODELS:
        for qr in Q_R:
            print(f"{m:8s} {qr:7s} {f(m, qr, 'clean'):7.3f} "
                  f"{f(m, qr, 'corrupt'):8.3f} {eff[m][qr]:+7.3f}")
    print("\nP1b cpt q_c effect at zero: %+.3f (need >= +0.10) -> %s"
          % (eff["cpt"]["zero"], "PASS" if p1b["pass"] else "FAIL"))
    print("P2b cpt substitution ratio: %.2f (eff0 %+.3f / effO %+.3f; need >= 2)"
          " -> %s" % (ratio_cpt, eff["cpt"]["zero"], eff["cpt"]["oracle"],
                      "PASS" if p2b["pass"] else "FAIL"))
    print("P3b vanilla/cpt at zero: clean %.2fx (%.3f vs %.3f), corrupt %.2fx "
          "(%.3f vs %.3f); vanilla gap %+.3f -> %s"
          % (v_zc / c_zc, v_zc, c_zc, v_zx / c_zx, v_zx, c_zx,
             eff["vanilla"]["zero"], "PASS" if p3b["pass"] else "FAIL"))
    print("\n== envelope: cpt pooled medians ==")
    for k in ("s70_mcar|0.7", "s70_block|0.7", "s70b_mcar|0.9_cpt",
              "s70b_block|0.9_cpt", "s70b_block|1.0_cpt"):
        print(f"  {k:22s} " + "  ".join(f"{kk}:{vv:.3f}"
                                        for kk, vv in env[k].items()))


if __name__ == "__main__":
    main()
