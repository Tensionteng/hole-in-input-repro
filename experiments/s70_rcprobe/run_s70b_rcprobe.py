#!/usr/bin/env python
"""S70B: the R/C selective-trust probe in the long/full-outage regime.

Motivation: at S70's rates (0.3/0.7) the CPT retrofit never needs the context
(enough target history survives), so the substitution signature (P2) had no
room to appear. S70B reruns the probe at (mcar, 0.9), (block, 0.9) and
(block, 1.0) -- outages long enough that univariate extrapolation is
impossible and the model must choose between the repair R and the context C.

Everything else is identical to S70 (same checkpoints, conventions, windows,
seeds, metric; see s70_notes.md). All numerics-critical code is imported from
run_s70_rcprobe; this script only re-defines the grid, the rate-1.0 mask rule,
and the orchestration.

Rate-1.0 rule (flagged in s70b_notes.md): make_mask("block", 1.0) covers only
~62% of the window (21 nominal blocks of 24 with overlap), which is not a full
outage. (block, 1.0) here masks ALL L positions (target and, when corrupt,
neighbours) -- degenerate with (mcar, 1.0), as the design intends. At
(block, 1.0) q_r=zero and q_r=linear are identical inputs (linear fill with no
valid positions falls back to zeros).

--anchor-check: re-run S70's (block, 0.7) cells for ETTh1 (all 6 channels,
3 q_r x 2 q_c x 3 models = 108 cells) through this code path and require each
cell median within 5% of s70_rcprobe.json before any new cell runs.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False     # strict fp32 eval
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s31_crosschannel"))
sys.path.insert(0, os.path.join(EXP, "s60_c2retrofit"))
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import run_s31_crosschannel as s31
import run_s70_rcprobe as s70

L, H = s70.L, s70.H
Q_R = s70.Q_R
Q_C = s70.Q_C
GRID = (("mcar", 0.9), ("block", 0.9), ("block", 1.0))   # (mcar,1.0) skipped
OUT = os.path.join(HERE, "s70b_rcprobe.json")
S70_RES = os.path.join(HERE, "s70_rcprobe.json")
ANCHOR_TOL = 0.05


def build_masks_b(mech, rate, n_win, K):
    """S70's build_masks, except (block, 1.0) = full outage (all positions)."""
    if mech == "block" and rate == 1.0:
        return (np.ones((n_win, L), bool), np.ones((n_win, K, L), bool))
    return s70.build_masks(mech, rate, n_win, K)


def cell_arrays_b(X, starts, ch, nb, mech, rate, n_win):
    """s70.cell_arrays with the S70B mask rule."""
    clean, gt, cov = [], [], []
    for s in starts[:n_win]:
        w = X[s:s + L].T
        clean.append(w[ch].copy())
        cov.append(w[nb].copy())
        gt.append(X[s + L:s + L + H].T[ch].copy())
    f32 = lambda a: np.asarray(a, np.float32)
    clean, gt, cov = f32(clean), f32(gt), f32(cov)
    mt, mc = build_masks_b(mech, rate, len(clean), cov.shape[1])
    fills = {q: s25.fill_context(clean, mt, q) for q in Q_R}
    cov_corrupt = s25.fill_context(cov, mc, "zero")
    return clean, gt, mt, fills, cov, cov_corrupt, mc


def run_dataset_b(ds, models, grid, n_win, res, max_ch=None, tag="",
                  persist=True):
    """s70.run_dataset over an explicit (mech, rate) grid."""
    import pandas as pd
    path = s25.DATASETS[ds]
    Xall = pd.read_csv(path).drop(columns=["date"]).to_numpy(np.float32)
    nb_map = s31.neighbours(Xall, int(0.8 * len(Xall)))
    X, st = s25.load_windows(path, n_win, s25.SEED, H)
    chans = s70.s31_channels(Xall)
    if max_ch:
        chans = chans[:max_ch]
    print(f"[{tag}] {ds}: channels {chans}, {len(st)} windows", flush=True)

    for ch in chans:
        nb = nb_map[ch]
        clean, gt0, _, _, _, _, _ = cell_arrays_b(X, st, ch, nb, "mcar", 0.9,
                                                  len(st))
        base = {}
        for name, model in models.items():
            if name == "vanilla":
                e0 = ((model.fc(clean) - gt0) ** 2).mean(1)
            else:
                obs1 = np.ones_like(clean)
                y0 = s70.median_fwd_grouped(model, clean, obs1,
                                            s70.MODEL_IFACE[name], 1)
                e0 = ((y0 - gt0) ** 2).mean(1)
            base[name] = e0
            res.setdefault("clean_mse", {}).setdefault(name, {})[f"{ds}|ch{ch}"] = {
                "median": float(np.median(e0)), "mean": float(e0.mean())}
        print(f"[{tag}] {ds}|ch{ch} clean mse " + " ".join(
            f"{m}:{np.median(base[m]):.4g}" for m in models), flush=True)

        for mech, rate in grid:
            t0 = time.time()
            clean, gt, mt, fills, cov, covc, mc = cell_arrays_b(
                X, st, ch, nb, mech, rate, len(st))
            for q_r in Q_R:
                for q_c in Q_C:
                    cv = cov if q_c == "clean" else covc
                    for name, model in models.items():
                        if name == "vanilla":
                            y = model.fc(fills[q_r], cv)
                        else:
                            y = s70.fc_dual(model, fills[q_r], cv, mt, mc,
                                            s70.MODEL_IFACE[name])
                        e = ((y - gt) ** 2).mean(1)
                        med, mean, n_ok, r = s70.rel_median(e, base[name])
                        key = f"{ds}|ch{ch}|{mech}|{rate}|{q_r}|{q_c}|{name}"
                        res.setdefault("cells", {})[key] = {
                            "median": med, "mean": mean, "n": n_ok,
                            "neighbours": nb,
                            "w": [round(float(x), 6) for x in r]}
            print(f"[{tag}] {ds}|ch{ch}|{mech}|{rate} done "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if persist:
                save(res)
    return res


def save(res, path=None):
    path = path or OUT
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, path)


def anchor_check(models, n_win, tol=ANCHOR_TOL):
    """Re-run S70's (block, 0.7) cells for ETTh1 (all models) through the S70B
    code path; each cell median must match s70_rcprobe.json within `tol`."""
    old = json.load(open(S70_RES))["cells"]
    res = {}
    run_dataset_b("ETTh1", models, (("block", 0.7),), n_win, res, tag="anchor",
                  persist=False)
    bad = []
    for k, c in res["cells"].items():
        ref = old[k]["median"]
        dev = c["median"] / ref - 1.0
        status = "ok" if abs(dev) <= tol else "FAIL"
        if status == "FAIL":
            bad.append(k)
        print(f"  anchor {k:44s} {c['median']:.4f} vs {ref:.4f} "
              f"({100 * dev:+5.2f}%) {status}", flush=True)
    n = len(res["cells"])
    print(f"ANCHOR CHECK: {n - len(bad)}/{n} within {tol:.0%} -> "
          f"{'PASS' if not bad else 'FAIL'}", flush=True)
    return {"tolerance": tol, "n_cells": n, "n_fail": len(bad),
            "failed": bad, "pass": not bad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="weather,ETTh1,ETTm1")
    ap.add_argument("--models", default="vanilla,ctl5k,cpt")
    ap.add_argument("--n-win", type=int, default=150)
    ap.add_argument("--max-ch", type=int, default=None)
    ap.add_argument("--anchor-check", action="store_true",
                    help="run the S70 (block, 0.7) anchor cells on ETTh1 and "
                         "exit nonzero if any deviates > 5%")
    ap.add_argument("--tag", default="s70b")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out:
        global OUT
        OUT = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    assert not torch.backends.cuda.matmul.allow_tf32, \
        "TF32 got re-enabled by an import -- eval must be strict fp32"
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    dev = "cuda:0"   # one card per process; visibility via CUDA_VISIBLE_DEVICES

    models = {}
    for name in args.models.split(","):
        models[name] = s31.C2(dev=dev) if name == "vanilla" else s70.load_dual(name, dev)
    print(f"[{args.tag}] loaded models: {list(models)}", flush=True)

    if args.anchor_check:
        r = anchor_check(models, args.n_win)
        out = os.path.join(HERE, "s70b_anchor.json")
        json.dump(r, open(out, "w"))
        print("wrote", out, flush=True)
        sys.exit(0 if r["pass"] else 1)

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {
        "design": "s70b: long/full-outage R/C probe; extends s70_rcprobe/DESIGN.md",
        "grid": "3 ds x 6 targets x (mcar,0.9),(block,0.9),(block,1.0) x q_r "
                "{zero,linear,oracle} x q_c {clean,corrupt} x 3 models = 972 cells",
        "rate_1.0_rule": "(block,1.0) = ALL L positions masked (full outage); "
                         "make_mask('block',1.0) would cover only ~62%",
        "degeneracy": "at (block,1.0) q_r=zero and q_r=linear are identical "
                      "inputs (linear fill with no valid positions -> zeros)",
        "envelope": "cpt recipe trained at rate ~ U(0.05, 0.7) (train_s60.py "
                    "mechdiv regime); rates 0.9/1.0 are extrapolation for cpt",
        "n_win": args.n_win, "K_neighbours": s70.K_NEIGH, "H": H, "L": L,
        "seed": s25.SEED, "dtype": "fp32 strict",
        "relMSE": "per-window vs each model's OWN clean univariate forecast; "
                  "median of per-window ratios (S31 convention)",
        "conventions": "identical to s70 (see s70_notes.md): dual arms "
                       "declared (target + corrupt neighbours), vanilla silent",
        "checkpoints": "identical to s70 (see s70_notes.md / s70 meta)"})
    for ds in args.datasets.split(","):
        res = run_dataset_b(ds, models, GRID, args.n_win, res, args.max_ch,
                            args.tag)
        save(res)
    save(res)
    print(f"[{args.tag}] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
