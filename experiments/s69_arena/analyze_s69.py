#!/usr/bin/env python
"""S69 analysis: gates G1 and the pre-registered predictions P1-P5 (DESIGN.md), from
s69_eval_<arm>_s<seed>.json (E1/E2, per arm/seed) and s69_bayes.json (E3 yardstick).

Outputs s69_analysis.json with per-seed and seed-mean verdicts, plus the figure sources
(per-arm 5x5 relMSE heatmaps per (mech, rate) panel, the Bayes reference panel, and the
w_R(q_r) curves at q_c=1 overlaid with w*_R).

No scipy: Spearman is Pearson on average-tie ranks, implemented inline.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import gen_s69_corpus as g69

ARMS = g69.ARMS
Q = list(g69.Q_GRID)


def ranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float(ra @ rb / d) if d > 0 else float("nan")


def field(cells, mech, rate, key):
    """The 25-cell (q_r, q_c) field of `key`, as (matrix [q_r x q_c], flat list)."""
    m = np.full((len(Q), len(Q)), np.nan)
    for i, qr in enumerate(Q):
        for j, qc in enumerate(Q):
            c = cells.get(f"{mech}|{rate}|{qr}|{qc}")
            if c is not None:
                m[i, j] = c[key]
    return m, [m[i, j] for i in range(len(Q)) for j in range(len(Q))
               if not np.isnan(m[i, j])]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="s69", choices=["s69", "s69b"])
    args = ap.parse_args()
    v2 = args.round == "s69b"
    ap_out = os.path.join(HERE, f"s69b_analysis.json" if v2 else "s69_analysis.json")
    pat = "s69b_eval_*_s*.json" if v2 else "s69_eval_*_s*.json"
    evals = {}
    for p in sorted(glob.glob(os.path.join(HERE, pat))):
        d = json.load(open(p))
        if d["meta"].get("smoke"):
            continue
        evals[(d["meta"]["arm"], d["meta"]["seed"])] = d
    bayes_path = os.path.join(HERE, "s69b_bayes.json" if v2 else "s69_bayes.json")
    bayes = json.load(open(bayes_path)) if os.path.exists(bayes_path) else None
    seeds = sorted({s for _, s in evals})
    arms = sorted({a for a, _ in evals})
    print(f"found evals: {sorted(evals)}; bayes: {'yes' if bayes else 'no'}")
    res = {"meta": {"round": args.round,
                    "evals": [f"{a}_s{s}" for a, s in sorted(evals)],
                    "code_hash": g69.code_hash()},
           "gates": {}, "predictions": {}, "figure_sources": {}}
    if not evals:
        json.dump(res, open(ap_out, "w"))
        sys.exit("no eval jsons yet")

    # ---- G1: all arms' clean MSE within 10% of each other (per seed)
    g1 = {}
    for s in seeds:
        cms = {a: evals[(a, s)]["clean"]["mse_median"] for a in arms
               if (a, s) in evals}
        if len(cms) < 2:
            continue
        lo, hi = min(cms.values()), max(cms.values())
        g1[f"s{s}"] = {"clean_mse_median": cms, "spread": hi / lo - 1,
                       "pass": bool(hi / lo - 1 <= 0.10)}
    res["gates"]["G1"] = g1

    mechs_rates = [(m, r) for m in
                   (g69.EVAL_MECHS_V2 if v2 else g69.EVAL_MECHS)
                   for r in g69.EVAL_RATES]

    def cell_field(arm, seed, key, mech, rate):
        return field(evals[(arm, seed)]["cells"], mech, rate, key)

    # ---- P1: native arms' w_R flat in q_r (range and slope per panel)
    p1 = {}
    for a in arms:
        if ARMS[a]["iface"] != "native":
            continue
        for s in seeds:
            if (a, s) not in evals:
                continue
            for mech, rate in mechs_rates:
                mvals, _ = cell_field(a, s, "wR_mean", mech, rate)
                row = np.nanmean(mvals, axis=1)          # w_R(q_r) averaged over q_c
                p1[f"{a}|s{s}|{mech}|{rate}"] = {
                    "wR_by_qr": [float(v) for v in row],
                    "range": float(row.max() - row.min()),
                    "spearman_vs_qr": spearman(
                        np.repeat(np.arange(len(Q)), len(Q)),
                        [v for v in mvals.flatten() if not np.isnan(v)])}
    res["predictions"]["P1"] = p1

    # ---- P2: arm C == arm A on E1/E2; C's F1 flip ~ 0
    p2 = {}
    for s in seeds:
        if ("A", s) not in evals or ("C", s) not in evals:
            continue
        for mech, rate in mechs_rates:
            ma, fa = cell_field("A", s, "rel_median", mech, rate)
            mc, fc = cell_field("C", s, "rel_median", mech, rate)
            wa, _ = cell_field("A", s, "wR_mean", mech, rate)
            wc, _ = cell_field("C", s, "wR_mean", mech, rate)
            flips = [evals[("C", s)]["cells"][f"{mech}|{rate}|{qr}|{qc}"]
                     ["flip"]["abs_delta_mean"]
                     for qr in Q for qc in Q
                     if f"{mech}|{rate}|{qr}|{qc}" in evals[("C", s)]["cells"]]
            p2[f"s{s}|{mech}|{rate}"] = {
                "relmse_max_abs_diff": float(np.nanmax(np.abs(ma - mc))),
                "wR_max_abs_diff": float(np.nanmax(np.abs(wa - wc))),
                "flip_abs_delta_max": float(np.max(flips)) if flips else None}
    res["predictions"]["P2"] = p2

    # ---- P3: arm D substitution + Spearman vs the Bayes field; A/B countercheck.
    # Two field scorings, both reported (s69b notes): the PRE-REGISTERED signed field
    # (spearman_wR_wstar) and the magnitude field (|w|), which is the meaningful one
    # when the optimal coefficient pattern is oscillatory (harmonic process: the
    # position/horizon-averaged coefficient is a small signed residual whose sign is
    # phase-dependent, while |w| still grows with q).
    p3 = {}
    for a in arms:
        for s in seeds:
            if (a, s) not in evals:
                continue
            for mech, rate in mechs_rates:
                mvals, flat = cell_field(a, s, "wR_mean", mech, rate)
                cvals, cflat = cell_field(a, s, "wC_mean", mech, rate)
                entry = {"spearman_wR_wstar": None, "spearman_abs_wR_wstar": None}
                if bayes and len(flat) == 25:
                    bflat = field(bayes["cells"], mech, rate, "wR_mean")[1]
                    entry["spearman_wR_wstar"] = spearman(flat, bflat)
                    entry["spearman_abs_wR_wstar"] = spearman(
                        [abs(v) for v in flat], [abs(v) for v in bflat])
                qr_idx = np.repeat(np.arange(len(Q)), len(Q))
                qc_idx = np.tile(np.arange(len(Q)), len(Q))
                entry["spearman_wR_vs_qr"] = spearman(qr_idx, flat)
                entry["spearman_absW_vs_qr"] = spearman(
                    qr_idx, [abs(v) for v in flat])
                entry["spearman_wC_vs_qc"] = spearman(qc_idx, cflat)
                entry["spearman_wC_vs_qr"] = spearman(qr_idx, cflat)
                entry["spearman_absWC_vs_qr"] = spearman(
                    qr_idx, [abs(v) for v in cflat])
                p3[f"{a}|s{s}|{mech}|{rate}"] = entry
    res["predictions"]["P3"] = p3

    # ---- P4: at q_r=0, arm D MSE >= 20% below native arms; flip raises it back.
    # s69b amendment: the EXPECTED native level is the yardstick's naive-trust line
    # (mse_naive / clean floor); both reference levels are included per cell.
    p4 = {}
    for s in seeds:
        if ("D", s) not in evals:
            continue
        natives = [a for a in arms if ARMS[a]["iface"] == "native"
                   and (a, s) in evals]
        for mech, rate in mechs_rates:
            for qc in Q:
                k = f"{mech}|{rate}|0.0|{qc}"
                cd = evals[("D", s)]["cells"].get(k)
                if cd is None:
                    continue
                row = {"mse_D": cd["mse_median"],
                       "mse_native": {a: evals[(a, s)]["cells"][k]["mse_median"]
                                      for a in natives},
                       "flip_rel_median": cd.get("flip", {}).get("rel_median")}
                if bayes and k in bayes["cells"]:
                    bc = bayes["cells"][k]
                    row["bayes_floor_rel"] = (bc["mse_mean"] /
                                              bayes["clean_floor"]["mse_mean"])
                    if "mse_naive_mean" in bc:
                        row["naive_trust_rel"] = (bc["mse_naive_mean"] /
                                                  bayes["clean_floor"]["mse_mean"])
                if natives:
                    worst_native = min(row["mse_native"].values())
                    row["D_below_native"] = float(1 - cd["mse_median"] / worst_native)
                p4[f"s{s}|{k}"] = row
    res["predictions"]["P4"] = p4

    # ---- P5: arm D clean MSE within 5% of arm A
    p5 = {}
    for s in seeds:
        if ("A", s) in evals and ("D", s) in evals:
            ca, cd = (evals[("A", s)]["clean"]["mse_median"],
                      evals[("D", s)]["clean"]["mse_median"])
            p5[f"s{s}"] = {"clean_A": ca, "clean_D": cd, "diff": cd / ca - 1,
                           "pass": bool(abs(cd / ca - 1) <= 0.05)}
    res["predictions"]["P5"] = p5

    # ---- figure sources: relMSE heatmaps per arm x panel + Bayes reference panel,
    #      and w_R(q_r) curves at q_c=1 overlaid with w*
    figs = {"q_grid": Q, "panels": [f"{m}|{r}" for m, r in mechs_rates]}
    for a in arms:
        for mech, rate in mechs_rates:
            mats = []
            for s in seeds:
                if (a, s) in evals:
                    mats.append(cell_field(a, s, "rel_median", mech, rate)[0])
            if mats:
                figs[f"relMSE_heatmap|{a}|{mech}|{rate}"] = np.nanmean(
                    np.stack(mats), 0).tolist()
            curves = []
            for s in seeds:
                if (a, s) in evals:
                    curves.append(cell_field(a, s, "wR_mean", mech, rate)[0][:, -1])
            if curves:
                figs[f"wR_curve_qc1|{a}|{mech}|{rate}"] = np.nanmean(
                    np.stack(curves), 0).tolist()
    if bayes:
        for mech, rate in mechs_rates:
            figs[f"bayes_mse_floor|{mech}|{rate}"] = field(
                bayes["cells"], mech, rate, "mse_mean")[0].tolist()
            figs[f"bayes_wR_curve_qc1|{mech}|{rate}"] = field(
                bayes["cells"], mech, rate, "wR_mean")[0][:, -1].tolist()
            if v2:
                figs[f"bayes_naive_trust|{mech}|{rate}"] = field(
                    bayes["cells"], mech, rate, "mse_naive_mean")[0].tolist()
    res["figure_sources"] = figs

    json.dump(res, open(ap_out, "w"), indent=1)
    print("wrote", ap_out)
    for g, v in res["gates"].items():
        print(g, json.dumps(v, indent=1)[:400])
    for p in ("P1", "P3", "P4", "P5"):
        keys = list(res["predictions"].get(p, {}))[:4]
        for k in keys:
            print(p, k, json.dumps(res["predictions"][p][k])[:240])


if __name__ == "__main__":
    main()
