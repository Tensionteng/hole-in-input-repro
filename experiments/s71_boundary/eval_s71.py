#!/usr/bin/env python
"""S71 evaluation: G0 stock reproduction + the DESIGN's per-arm eval battery.

Per model (timesfm / tempopfn / timer) and arm (stock / cpt / ctl5k / cpt@seed):
  probe       s30 permutation rho, 3 ds x 3 mechs, 60 win, rate 0.3, linear fill,
              each model's convs (timer: plain only -- its nan path emits NaN logits)
  sweep       alpha-blend fill-quality sweep, alpha in {0,.25,.5,.75,1},
              block+mcar at rate 0.7, ETTh1/ETTm1/weather, 150 win,
              relMSE vs the arm's own clean; declared + plain views (declared views
              that are provably alpha-invariant are computed once and copied, s62
              convention)
  clean       clean-context MSE on the same 3 datasets
  leaderboard the s57 grid: 9 ds x 4 mechs x rate 0.7 x {zero,linear,oracle},
              each model's declared path (timer: plain, as the paper's row), 150 win,
              relMSE vs the arm's own clean; clean MSEs stored
  g0          stock arm vs the stored references (s57_timesfm.json,
              s57_timerxl_plain.json, s62_results.json for tempopfn -- it has no s57
              row; documented deviation), 5% tolerance on shared cells

Resumable at the cell level; writes s71_<model>.json atomically.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tempopfn_triton_cache")

import numpy as np
import torch

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s35_breadth"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
sys.path.insert(0, HERE)

import run_s25_twofloor as s25
import run_s35_breadth as s35
import run_s30_crossmodel as s30mod
import s71_common as C

L, H = s25.L, 64
DS3 = ("ETTh1", "ETTm1", "weather")
DS9 = ("ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness")
MECHS_PROBE = ("mcar", "block", "mnar_high")
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
MECHS_SWEEP = ("block", "mcar")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
FILLS = ("zero", "linear", "oracle")

G0_REFS = {
    "timesfm": os.path.join(EXP, "s45_pretrain", "s57_timesfm.json"),
    "timer": os.path.join(EXP, "s45_pretrain", "s57_timerxl_plain.json"),
    "tempopfn": os.path.join(EXP, "s62_tempopfn", "s62_results.json"),
}
G0_REF_NAME = {"timesfm": "timesfm", "timer": "timerxl-plain",
               "tempopfn": "tempopfn-38m"}


def out_path(model):
    return os.path.join(HERE, f"s71_{model}.json")


def load_res(model):
    p = out_path(model)
    return json.load(open(p)) if os.path.exists(p) else {}


def save(model, res):
    tmp = f"{out_path(model)}.{os.getpid()}.tmp"   # PID-unique: concurrent eval
    json.dump(res, open(tmp, "w"))                 # processes for different arms of
    os.replace(tmp, out_path(model))               # the same model must not race


def blend(clean, lin, mask, a):
    out = lin.copy()
    out[mask] = (1.0 - a) * lin[mask] + a * clean[mask]
    return out


def declared_conv(model):
    return {"timesfm": "nan", "tempopfn": "nan", "timer": "plain"}[model]


def relmse_cell(pred, gt, base):
    e = ((pred.astype(np.float64) - gt.astype(np.float64)) ** 2).mean(1)
    ok = base > 1e-12
    r = e[ok] / base[ok]
    return {"median": float(np.median(r)), "mean": float(r.mean()),
            "w": [round(float(x), 6) for x in r]}


# ------------------------------------------------------------------ parts ----

def part_probe(model, arm, wrapper, res):
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, 60, s25.SEED, H)
        for mech in MECHS_PROBE:
            _, lin, mask, _ = s25.build_eval_batch(X, st, mech, 0.3, H,
                                                   fill="linear")
            for conv in wrapper.convs:
                key = f"{ds}|{mech}|{arm}|{conv}"
                if key in res.setdefault("probe", {}):
                    continue
                rng = np.random.default_rng(s25.SEED)
                r = s30mod.perm_probe(wrapper, lin, mask, conv, rng)
                res["probe"][key] = r
                print(f"  probe {ds:8s} {mech:10s} {arm:6s} {conv:6s} "
                      f"rho={r['ratio']:.4f}", flush=True)
            save(model, res)


def part_sweep(model, arm, wrapper, res):
    sw = res.setdefault("sweep", {})
    views = [c for c in ("declared", "plain")
             if not (model == "timer" and c == "declared")]
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, 150, s25.SEED, H)
        _, cl, mk0, gt = s25.build_eval_batch(X, st, "clean", 0.0, H)
        pc = wrapper.fc(cl, mk0, declared_conv(model))
        base = ((pc.astype(np.float64) - gt.astype(np.float64)) ** 2).mean(1)
        for mech in MECHS_SWEEP:
            clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, 0.7, H,
                                                        fill="linear")
            nan_cache = {}
            for a in ALPHAS:
                ctx = blend(clean, lin, mask, a)
                for view in views:
                    key = f"{ds}|{mech}|0.7|{a}|{arm}|{view}"
                    if key in sw:
                        continue
                    conv = declared_conv(model) if view == "declared" else "plain"
                    if conv == "nan":
                        # declared path input is provably alpha-invariant
                        if mech not in nan_cache:
                            pred = wrapper.fc(lin, mask, "nan")
                            nan_cache[mech] = relmse_cell(pred, gt, base)
                        sw[key] = dict(nan_cache[mech])
                        sw[key]["note"] = "alpha-invariant input; computed once"
                        continue
                    pred = wrapper.fc(ctx, mask, conv)
                    sw[key] = relmse_cell(pred, gt, base)
            line = " ".join(
                f"{v}:{sw[f'{ds}|{mech}|0.7|0.0|{arm}|{v}']['median']:.2f}->"
                f"{sw[f'{ds}|{mech}|0.7|1.0|{arm}|{v}']['median']:.2f}"
                for v in views)
            print(f"  sweep {ds:8s} {mech:6s} {arm:6s} {line}", flush=True)
            save(model, res)


def part_clean(model, arm, wrapper, res):
    for ds, path in s25.DATASETS.items():
        key = f"{arm}|{ds}"
        if key in res.setdefault("clean", {}):
            continue
        X, st = s25.load_windows(path, 150, s25.SEED, H)
        _, cl, mk0, gt = s25.build_eval_batch(X, st, "clean", 0.0, H)
        pred = wrapper.fc(cl, mk0, declared_conv(model))
        e = ((pred.astype(np.float64) - gt.astype(np.float64)) ** 2).mean(1)
        res["clean"][key] = {"mse_median": float(np.median(e)),
                             "mse_mean": float(e.mean())}
        print(f"  clean {ds:8s} {arm:6s} {res['clean'][key]['mse_median']:.4g}",
              flush=True)
        save(model, res)


def part_leaderboard(model, arm, wrapper, res, n_win=150):
    lb = res.setdefault("leaderboard", {})
    for ds in DS9:
        X, st = s35.load(ds, n_win)
        if X is None:
            print(f"  leaderboard {ds}: no valid windows, skipped", flush=True)
            continue
        _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
        zmask = np.zeros_like(cl, bool)
        pc = wrapper.fc(cl, zmask, declared_conv(model))
        base = ((pc.astype(np.float64) - gt_c.astype(np.float64)) ** 2).mean(1)
        lb.setdefault("clean_mse", {}).setdefault(arm, {})[ds] = {
            "median": float(np.median(base)), "mean": float(base.mean())}
        for mech in MECHS:
            clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, 0.7, H,
                                                        fill="linear")
            for fill in FILLS:
                key = f"{ds}|{mech}|0.7|{fill}|{arm}"
                if key in lb.setdefault("grid", {}):
                    continue
                ctx = s25.fill_context(clean, mask, fill)
                pred = wrapper.fc(ctx, mask, declared_conv(model))
                lb["grid"][key] = relmse_cell(pred, gt, base)
            print(f"  lb {ds:11s} {mech:13s} {arm:6s} " + " ".join(
                f"{f}:{lb['grid'][f'{ds}|{mech}|0.7|{f}|{arm}']['median']:.2f}"
                for f in FILLS), flush=True)
        save(model, res)


def _g0_tempopfn(wrapper, ref, res):
    """Reproduce s62's stock cells at s62's own protocol (60 win, alphas {0,.5,1},
    rates {0.3,0.7}, 4 mechs, both convs) and compare. s62's sweep relMSE uses
    ratio-of-means / median-of-ratios on 60 windows, so the 150-win arm sweep is NOT
    a substitute -- this pass is self-contained."""
    cmp, worst = [], 0.0
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, 60, s25.SEED, H)   # s62 N_WIN=60
        for mech in MECHS_PROBE:
            _, lin, mask, _ = s25.build_eval_batch(X, st, mech, 0.3, H,
                                                   fill="linear")
            for conv in wrapper.convs:
                key = f"tempopfn-38m|{ds}|{mech}|{conv}"
                if key not in ref.get("probe", {}):
                    continue
                rng = np.random.default_rng(s25.SEED)
                r = s30mod.perm_probe(wrapper, lin, mask, conv, rng)
                d = abs(r["ratio"] - ref["probe"][key]["ratio"])
                cmp.append({"cell": key, "mine": r["ratio"],
                            "ref": ref["probe"][key]["ratio"], "abs_dev": d})
                worst = max(worst, d / 0.05)     # tol: abs 0.05 on rho
                print(f"  g0 probe {ds:8s} {mech:10s} {conv:6s} "
                      f"rho={r['ratio']:.4f} ref={ref['probe'][key]['ratio']:.4f}",
                      flush=True)
        save("tempopfn", res)
        _, cl, mk0, gt = s25.build_eval_batch(X, st, "clean", 0.0, H)
        pc = wrapper.fc(cl, mk0, "plain")
        mse_clean = ((pc.astype(np.float64) - gt.astype(np.float64)) ** 2).mean(1)
        for mech in MECHS:
            for rate in (0.3, 0.7):
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, rate, H,
                                                            fill="linear")
                nan_pred = None
                for conv in wrapper.convs:
                    for a in (0.0, 0.5, 1.0):
                        key = f"tempopfn-38m|{ds}|{mech}|{rate}|{conv}|a={a}"
                        if key not in ref.get("sweep", {}):
                            continue
                        if conv == "nan":
                            if nan_pred is None:
                                nan_pred = wrapper.fc(lin, mask, "nan")
                            pred = nan_pred
                        else:
                            pred = wrapper.fc(blend(clean, lin, mask, a), mask,
                                              conv)
                        e = ((pred.astype(np.float64) - gt.astype(np.float64))
                             ** 2).mean(1)
                        ok = mse_clean > 1e-12
                        mine = float(np.median(e[ok] / mse_clean[ok]))
                        refv = ref["sweep"][key]["relMSE_median"]
                        d = abs(mine - refv) / max(1e-9, abs(refv))
                        cmp.append({"cell": key, "mine": mine, "ref": refv,
                                    "rel_dev": d,
                                    "stat": "median of per-series ratios"})
                        worst = max(worst, d / 0.05)
            print(f"  g0 sweep {ds:8s} {mech:13s} done", flush=True)
        save("tempopfn", res)
    return cmp, worst


def part_g0(model, wrapper_stock, res):
    """Compare the stock arm's freshly computed cells against the stored refs."""
    name = G0_REF_NAME[model]
    ref = json.load(open(G0_REFS[model]))
    if model in ("timesfm", "timer"):
        cmp, worst = [], 0.0
        for key, cell in ref["grid"].items():
            ds, mech, rate, fill, _ = key.rsplit("|", 4)
            mine = res["leaderboard"]["grid"].get(f"{ds}|{mech}|{rate}|{fill}|stock")
            if mine is None or not np.isfinite(cell["median"]):
                continue
            d = abs(mine["median"] - cell["median"]) / max(1e-9,
                                                           abs(cell["median"]))
            worst = max(worst, d)
            cmp.append({"cell": key, "mine": mine["median"],
                        "ref": cell["median"], "rel_dev": d})
        for ds, cell in ref["clean_mse"].get(name, {}).items():
            mine = res["leaderboard"]["clean_mse"].get("stock", {}).get(ds)
            if mine is None:
                continue
            d = abs(mine["median"] - cell["median"]) / max(1e-9,
                                                           abs(cell["median"]))
            worst = max(worst, d)
            cmp.append({"cell": f"clean|{ds}", "mine": mine["median"],
                        "ref": cell["median"], "rel_dev": d})
        passed, tol_note = worst <= 0.05, "rel dev <= 0.05 on shared cells"
    else:
        cmp, worst = _g0_tempopfn(wrapper_stock, ref, res)
        passed = worst <= 1.0
        tol_note = ("probe: abs dev <= 0.05 on rho; sweep: rel dev <= 0.05 on "
                    "median-of-ratios (worst is in tolerance units)")
    res["g0"] = {"ref": G0_REFS[model], "ref_name": name, "n_cells": len(cmp),
                 "worst": worst, "tolerance": tol_note, "pass": bool(passed),
                 "cells": cmp[:200]}
    save(model, res)
    print(f"GATE G0 {model}: {len(cmp)} cells, worst {worst:.4f} "
          f"{'PASS' if passed else 'FAIL'}", flush=True)
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--arms", default="stock")
    ap.add_argument("--parts", default="g0,probe,sweep,clean,leaderboard")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    dev = args.device
    parts = args.parts.split(",")

    # serialise writers: one eval process per model JSON at a time (incremental
    # saves are load-modify-write; two concurrent arms would drop each other's
    # cells). The lock is held for the whole run; waiters block.
    import fcntl
    locks = []
    for model in args.models.split(","):
        lf = open(os.path.join(HERE, f"s71_{model}.lock"), "w")
        fcntl.flock(lf, fcntl.LOCK_EX)
        locks.append(lf)

    for model in args.models.split(","):
        res = load_res(model)
        res.setdefault("meta", {
            "model": model, "declared_conv": declared_conv(model),
            "L": L, "H": H, "seed": s25.SEED,
            "probe": "s30 protocol, 60 win, rate 0.3, linear fill",
            "sweep": "alphas x {block,mcar} x 0.7 x 150 win, relMSE vs own clean",
            "leaderboard": "s57 grid, 150 win, declared path (timer: plain)"})
        arms = args.arms.split(",")
        if "g0" in parts and "stock" not in arms:
            arms = ["stock"] + arms
        for arm in arms:
            ckpt = C.resolve_ckpt(model, arm)
            wrapper = C.WRAPPERS[model](dev, ckpt)
            if "g0" in parts and arm == "stock":
                # G0 consumes the stock leaderboard (timesfm/timer) or runs its
                # own s62-protocol reproduction (tempopfn)
                if model != "tempopfn":
                    part_leaderboard(model, arm, wrapper, res)
                if not part_g0(model, wrapper, res):
                    print(f"G0 FAIL for {model} -- per DESIGN, drop the model "
                          f"and document", flush=True)
                # fall through: stock is also a regular arm for the other parts
            if "probe" in parts:
                part_probe(model, arm, wrapper, res)
            if "sweep" in parts:
                part_sweep(model, arm, wrapper, res)
            if "clean" in parts:
                part_clean(model, arm, wrapper, res)
            if "leaderboard" in parts:
                part_leaderboard(model, arm, wrapper, res)
            del wrapper
            torch.cuda.empty_cache()
        save(model, res)
    print("done", flush=True)


if __name__ == "__main__":
    main()
