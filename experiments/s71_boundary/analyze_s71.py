#!/usr/bin/env python
"""S71 verdict extraction: the DESIGN's Q1/Q2/Q3 numbers per model.

Per model and arm (stock / cpt / ctl5k / cpt@seed):
  rho_declared   median probe rho over the 9 cells (3 ds x 3 mechs), on the model's
                 native missingness convention (timesfm/tempopfn: nan; timer: plain
                 -- its ONLY path; the plain-path rho is the trivial no-interface
                 baseline, per s57 notes, and is not evidence of an interface)
  closure        per sweep cell (3 ds x {block,mcar}, declared view, rate 0.7):
                 (r0 - r1)/(r0 - 1) with r0/r1 the median relMSE at alpha 0/1;
                 cells with r0 <= 1.02 carry no fill-quality signal to close and are
                 reported as n/a; aggregate = median over defined cells
  clean_cost     median over ds of (arm clean MSE / stock clean MSE) - 1
  leaderboard    per fill: median over the 36 grid cells of median relMSE

Q1 (per DESIGN): no boundary model gains fill-reading from plain CPT: post-CPT
rho <= 0.1 AND closure <= 10%. Q2: ctl5k rho within 0.05 of stock. Q3
(falsification): rho > 0.3 AND closure > 25% on the native path.

Writes s71_verdicts.json and prints the verdict table.
"""
import argparse
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

DS3 = ("ETTh1", "ETTm1", "weather")
MECHS_PROBE = ("mcar", "block", "mnar_high")
MECHS_SWEEP = ("block", "mcar")
FILLS = ("zero", "linear", "oracle")

# bolt reference column (stored s53/s56 values; no new bolt training per DESIGN)
BOLT_REF = {"P5-base (bolt CPT ref)": {"rho": 0.832, "closure": None},
            "P0-base (stock ref)": {"rho": 0.0, "closure": None},
            "note": "s56_eval_base.json probe rho_declared medians; P5-tiny "
                    "closure 76.7% (s53_notes.md scorecard)"}


def rho_row(res, arm):
    out = {}
    for conv in ("nan", "plain"):
        vs = [res.get("probe", {}).get(f"{ds}|{mech}|{arm}|{conv}")
              for ds in DS3 for mech in MECHS_PROBE]
        vs = [v["ratio"] for v in vs if v is not None]
        if vs:
            out[conv] = float(np.median(vs))
    return out


def closure_row(res, arm, view="declared"):
    cls = []
    per = {}
    for ds in DS3:
        for mech in MECHS_SWEEP:
            v0 = res.get("sweep", {}).get(f"{ds}|{mech}|0.7|0.0|{arm}|{view}")
            v1 = res.get("sweep", {}).get(f"{ds}|{mech}|0.7|1.0|{arm}|{view}")
            if v0 is None and view == "declared":
                v0 = res.get("sweep", {}).get(f"{ds}|{mech}|0.7|0.0|{arm}|plain")
                v1 = res.get("sweep", {}).get(f"{ds}|{mech}|0.7|1.0|{arm}|plain")
            if v0 is None or v1 is None:
                continue
            r0, r1 = v0["median"], v1["median"]
            if r0 <= 1.02:
                per[f"{ds}|{mech}"] = None
                continue
            c = (r0 - r1) / (r0 - 1.0)
            cls.append(c)
            per[f"{ds}|{mech}"] = c
    return (float(np.median(cls)) if cls else None), per


def clean_cost(res, arm):
    cl = res.get("clean", {})
    ratios = []
    for ds in DS3:
        a, s = cl.get(f"{arm}|{ds}"), cl.get(f"stock|{ds}")
        if a and s:
            ratios.append(a["mse_median"] / s["mse_median"])
    return float(np.median(ratios)) if ratios else None


def lb_summary(res, arm):
    g = res.get("leaderboard", {}).get("grid", {})
    out = {}
    for fill in FILLS:
        vs = [v["median"] for k, v in g.items()
              if k.endswith(f"|{fill}|{arm}")]
        if vs:
            out[fill] = {"median": float(np.median(vs)), "max": float(np.max(vs)),
                         "n": len(vs)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="timesfm,tempopfn,timer")
    ap.add_argument("--arms", default="stock,cpt,ctl5k,cpt@20260904")
    args = ap.parse_args()
    out = {"bolt_ref": BOLT_REF}
    for model in args.models.split(","):
        p = os.path.join(HERE, f"s71_{model}.json")
        if not os.path.exists(p):
            continue
        res = json.load(open(p))
        mrow = {}
        for arm in args.arms.split(","):
            rho = rho_row(res, arm)
            clo, per = closure_row(res, arm)
            if not rho and clo is None:
                continue
            mrow[arm] = {
                "rho_median": rho,
                "closure_median": clo, "closure_cells": per,
                "clean_cost_ratio": clean_cost(res, arm),
                "leaderboard": lb_summary(res, arm),
            }
        out[model] = mrow
    json.dump(out, open(os.path.join(HERE, "s71_verdicts.json"), "w"), indent=1)
    for model, mrow in out.items():
        if model == "bolt_ref":
            continue
        print(f"\n== {model} ==")
        print(f"{'arm':16s} {'rho(nan)':>9s} {'rho(plain)':>11s} {'closure':>9s} "
              f"{'clean/stock':>11s}  lb zero/linear/oracle (median)")
        for arm, r in mrow.items():
            rho = r["rho_median"]
            lb = r["leaderboard"]
            lbs = "/".join(f"{lb[f]['median']:.2f}" if f in lb else "-"
                           for f in FILLS)
            print(f"{arm:16s} {rho.get('nan', float('nan')):9.3f} "
                  f"{rho.get('plain', float('nan')):11.3f} "
                  f"{(r['closure_median'] if r['closure_median'] is not None else float('nan')):9.3f} "
                  f"{(r['clean_cost_ratio'] if r['clean_cost_ratio'] is not None else float('nan')):11.3f}  {lbs}")
    print("\nwrote s71_verdicts.json")


if __name__ == "__main__":
    main()
