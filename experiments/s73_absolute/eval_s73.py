#!/usr/bin/env python
"""S73: per-window ABSOLUTE errors for the bolt CPT retrofit vs vanilla.

The paper's missingness tables report relMSE against each model's OWN clean
accuracy. The stored grids (s57_bolt.json / s57_q50.json, and the s60/s61/s65
sweeps) keep only per-cell medians/means of the per-window ratio plus
median/mean clean MSE -- per-window clean errors are nowhere stored, so the
absolute per-window errors (ratio x own clean) are NOT reconstructable from
stored data. This script re-runs the exact Table-5 (tab:leaderboard) bolt
cells and records per-window absolute MSE directly.

Arms (verified against the paper's table, see s73_notes.md):
  vanilla = "bolt-base-stock" (arm P0: the stock chronos-bolt-base dump,
             arm_P0_base.pt; G2' max|diff|=0 vs the HF checkpoint, s56 notes)
  cpt     = "Q50-base" (arm Q50: P8-base CPT x WiSE-FT 0.5, arm_Q50_base.pt)
            -- Q50-base reproduces every digit of the paper's bolt "+CPT
            (ours)" row (tab_leaderboard); P5-base is the blend-only ablation
            that is zero-fill blind and is NOT the paper row.

Grid: eval_s57_leaderboard.py verbatim -- DS9 x 4 mechs x rate 0.7 x fills
{zero, linear, oracle} on each arm's declared path, 150 windows/cell, plus the
clean cell per dataset. Batches are built ONCE per (ds, mech) and both arms
are scored on the identical in-memory arrays, so row i of the vanilla array
and row i of the CPT array are the same window x channel series.

Outputs:
  s73_perwindow.npz   per-row absolute MSE arrays:
                        clean|{arm}|{ds}                    [S*C]
                        miss|{arm}|{ds}|{mech}|{fill}       [S*C]
                      plus row meta: rows|{ds} = window index per row
  s73_eval_gate.json  per-cell medians of the absolute errors and of the
                      ratio vs own clean, and the gate deviation against the
                      stored s57 medians (must be ~0: same seeds/checkpoints).

TF32: deliberately NOT overridden -- eval_s57_leaderboard.py imports
train_s45, which sets allow_tf32=True, so the stored bolt numbers were
produced with TF32 on; we replicate that environment exactly.
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

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s27_interface"))
sys.path.insert(0, os.path.join(EXP, "s35_breadth"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
sys.path.insert(0, os.path.join(EXP, "s46_moirai2"))
sys.path.insert(0, os.path.join(EXP, "s49_2025models"))
sys.path.insert(0, os.path.join(EXP, "s45_pretrain"))
import run_s25_twofloor as s25
import run_s35_breadth as s35
import eval_s45 as e45
import eval_s57_leaderboard as e57

L, H = s25.L, 64
DS9 = e45.DS9
MECHS = e45.MECHS
FILLS = ("zero", "linear", "oracle")
ARMS = {"vanilla": "bolt-base-stock", "cpt": "Q50-base"}
NPZ = os.path.join(HERE, "s73_perwindow.npz")
GATE = os.path.join(HERE, "s73_eval_gate.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-win", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)

    scores = dict(np.load(NPZ)) if os.path.exists(NPZ) else {}
    gate = json.load(open(GATE)) if os.path.exists(GATE) else {
        "meta": {"grid": "DS9 x 4 mechs x rate 0.7 x {zero,linear,oracle} + clean",
                 "n_win": args.n_win, "arms": ARMS,
                 "convention": "per-row absolute MSE; ratio medians gated vs s57 stored cells"},
        "cells": {}}

    fcs = {k: e57.REGISTRY[v](dev) for k, v in ARMS.items()}

    for ds in DS9:
        X, st = s35.load(ds, args.n_win)
        if X is None:
            continue
        # row -> window index map (build_eval_batch concatenates windows in order)
        C = X.shape[1]
        rows_key = f"rows|{ds}"
        if rows_key not in scores:
            scores[rows_key] = np.repeat(np.arange(len(st)), C).astype(np.int32)

        t0 = time.time()
        # ---- clean cell
        cl, _, zm, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
        zmask = np.zeros_like(cl, bool)
        base = {}
        for aname, fc in fcs.items():
            key = f"clean|{aname}|{ds}"
            if key not in scores:
                pred = fc(cl, zmask, "declared")
                scores[key] = ((pred - gt_c) ** 2).mean(1).astype(np.float32)
            base[aname] = scores[key]
            gate["cells"].setdefault(f"clean|{aname}|{ds}", {
                "median": float(np.median(scores[key])),
                "mean": float(scores[key].mean()), "n": int(len(scores[key]))})

        # ---- missingness cells
        for mech in MECHS:
            clean, _, mask, gt = s25.build_eval_batch(X, st, mech, 0.7, H,
                                                      fill="linear")
            for fill in FILLS:
                ctx = s25.fill_context(clean, mask, fill)
                for aname, fc in fcs.items():
                    key = f"miss|{aname}|{ds}|{mech}|{fill}"
                    if key not in scores:
                        e = ((fc(ctx, mask, "declared") - gt) ** 2).mean(1)
                        scores[key] = e.astype(np.float32)
                    e = scores[key]
                    b = base[aname]
                    ok = b > 1e-12
                    r = e[ok] / b[ok]
                    gate["cells"][f"{ds}|{mech}|0.7|{fill}|{aname}"] = {
                        "abs_median": float(np.median(e)),
                        "abs_mean": float(e.mean()),
                        "rel_median": float(np.median(r)),
                        "rel_mean": float(r.mean()),
                        "n": int(len(e)), "n_ok": int(ok.sum())}
        np.savez(NPZ, **scores)
        json.dump(gate, open(GATE, "w"))
        print(f"== {ds} done in {time.time()-t0:.0f}s ==", flush=True)

    # ---- gate vs stored s57 cells
    s57 = {}
    for f, armfile in (("vanilla", "s57_bolt.json"), ("cpt", "s57_q50.json")):
        d = json.load(open(os.path.join(EXP, "s45_pretrain", armfile)))
        name = {"vanilla": "bolt-base-stock", "cpt": "Q50-base"}[f]
        for k, v in d["grid"].items():
            if k.endswith("|" + name):
                s57[k.replace("|" + name, "") + "|" + f] = v
    devs = []
    for k, stored in s57.items():
        mine = gate["cells"].get(k)
        if mine is None:
            continue
        rel = abs(mine["rel_median"] - stored["median"]) / max(stored["median"], 1e-9)
        devs.append({"cell": k, "stored_rel_median": stored["median"],
                     "mine_rel_median": mine["rel_median"], "rel_dev": rel})
    gate["gate_vs_s57"] = {
        "n_cells_compared": len(devs),
        "max_rel_dev": max((d["rel_dev"] for d in devs), default=None),
        "worst5": sorted(devs, key=lambda d: -d["rel_dev"])[:5]}
    json.dump(gate, open(GATE, "w"))
    np.savez(NPZ, **scores)
    print("gate max rel dev:", gate["gate_vs_s57"]["max_rel_dev"], flush=True)
    print("wrote", NPZ, "and", GATE)


if __name__ == "__main__":
    main()
