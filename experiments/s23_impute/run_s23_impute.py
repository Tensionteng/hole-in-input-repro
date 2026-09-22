#!/usr/bin/env python
"""Screen S23: do LEARNED imputers (SAITS/BRITS, PyPOTS) beat free fills on the
mechanism-stratified missing grid?

S5 (run_s5_missing.py / s5_fix_notes.md) showed that on the grid
  mechanisms {mcar, block, mnar_high, mnar_extreme} x p {0.1,0.3,0.5,0.7}
  x {ETTh1, ETTm1, weather}
linear interpolation is nearly lossless under MCAR and no free repair wins
under MNAR (value-censored context = unrecoverable information). The literature
(s20_prompt_lit_notes.md B1) shows BRITS/SAITS/CSDI were only ever evaluated on
artificial MCAR-ish corruption -- never on block/MNAR, and never by their effect
on a frozen downstream forecaster. S23 fills that gap: train deep imputers per
dataset, fill the S5 eval windows, feed chronos-bolt-base, score relMSE vs
clean, exactly like S5 (same windows/seeds/masks via import).

Imputer variants (mechanism distribution seen in TRAINING):
  mb  : masks sampled from {mcar, block}          -- the mainstream convention
  all : masks sampled from {mcar, block, mnar_high, mnar_extreme} -- mechanism-
        matched ("mnar-augmented") training, to discriminate mechanism-specific
        failure from an information floor.

Fairness protocol: imputers are trained on the train split (first 60% of the
timeline, val = 60-80% for early stopping), windows of L=512, one random
mechanism + rate ~ U(0.05, 0.7) per sample; per-channel standardisation with
train-split stats. Eval never touches the test region for training.

Pre-registered look points:
  (i)  can learned imputation beat linear under mcar/block? (if not, the
       "free repair is enough" conclusion strengthens)
  (ii) does learned imputation also fail under mnar? (expected: information
       floor, 8th test)
  (iii) does mnar-augmented training still lose under mnar? (mechanism-
        specificity vs information floor)

Modes:
  smoke   -- tiny end-to-end check (ETTh1, few samples, tiny SAITS, save/load)
  anchor  -- reproduce S5 bolt {clean, mcar:linear/zero:0.3, mnar_high:linear/zero:0.7}
             on ETTh1 within +-5% relMSE; gate for everything else
  train   -- train one imputer: --model {saits,brits} --variant {mb,all} --dataset DS
  eval    -- evaluate one imputer on the full mechanism x rate grid for one dataset
  collect -- merge anchor + baselines (S5 linear/zero, S5-fix nan) + eval shards
             into s23_results.json
  report  -- print the main table and write s23.png
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import pandas as pd
import torch

from run_s5_missing import (  # identical windows/masks/fills/metrics as S5
    L, H, RATES, BLOCK, SEED, DATASETS,
    load_windows, make_mask, run_config, BoltModel,
)

MECHS = ["mcar", "block", "mnar_high", "mnar_extreme"]
VARIANTS = {"mb": ["mcar", "block"], "all": MECHS}
MODELS = ["saits", "brits"]
CKPT_DIR = os.path.join(HERE, "s23_ckpt")
RESULTS_PATH = os.path.join(HERE, "s23_results.json")
S5_PATH = os.path.join(HERE, "s5_missing_results.json")
S5FIX_PATH = os.path.join(HERE, "s5_fix_results.json")
ANCHOR_PATH = os.path.join(CKPT_DIR, "anchor.json")
TRAIN_FRAC, VAL_FRAC = 0.6, 0.8  # train [0,.6), val [.6,.8), eval = last 20% (S5)
N_TRAIN = {"ETTh1": 6000, "ETTm1": 8000, "weather": 8000}
N_VAL = 512
AUG_RATE = (0.05, 0.7)  # training augmentation missing-rate range
ANCHOR_TOL = 0.05

SAITS_HP = dict(n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=256,
                dropout=0.1)
BRITS_HP = dict(rnn_hidden_size=256)


# ------------------------------------------------------------ helpers ----

def tag_of(model, variant, dataset):
    return f"{model}_{variant}_{dataset}"


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def split_bounds(N):
    return int(TRAIN_FRAC * N), int(VAL_FRAC * N)


def channel_mask(rng, mech, rate, xc):
    """[L] bool training-augmentation mask for one standardized channel xc.
    Mirrors S5's constructions but driven by the training rng."""
    if mech == "mcar":
        return rng.random(L) < rate
    if mech == "block":
        m = np.zeros(L, bool)
        nb = max(1, int(round(rate * L / BLOCK)))
        for s in rng.integers(0, L - BLOCK + 1, size=nb):
            m[s:s + BLOCK] = True
        return m
    k = int(np.ceil(rate * L))
    if mech == "mnar_high":
        key = xc
    else:  # mnar_extreme
        mu, sd = xc.mean(), xc.std()
        key = np.abs((xc - mu) / (sd if sd > 1e-12 else 1.0))
    m = np.zeros(L, bool)
    order = np.argsort(-key, kind="stable")
    m[order[:k]] = True
    return m


def build_augmented(X, lo, hi, n_samples, mechs, rng, mu, sd):
    """Returns (X_nan, X_clean): [n, L, C] float32 standardized windows, X_nan
    with NaN at augmented masks, X_clean the intact standardized windows
    (PyPOTS validation needs them as 'X_ori')."""
    C = X.shape[1]
    out = np.empty((n_samples, L, C), np.float32)
    ori = np.empty((n_samples, L, C), np.float32)
    starts = rng.integers(lo, hi - L + 1, size=n_samples)
    for i, s in enumerate(starts):
        xw = ((X[s:s + L] - mu) / sd).astype(np.float32)  # [L, C]
        ori[i] = xw
        mech = mechs[rng.integers(len(mechs))]
        rate = rng.uniform(*AUG_RATE)
        for c in range(C):
            m = channel_mask(rng, mech, rate, xw[:, c])
            xw[m, c] = np.nan
        out[i] = xw
    return out, ori


def build_imputer(model_name, n_features, device, epochs, patience, batch_size,
                  saving_path=None):
    from pypots.imputation import SAITS, BRITS
    if model_name == "saits":
        return SAITS(n_steps=L, n_features=n_features, **SAITS_HP,
                     batch_size=batch_size, epochs=epochs, patience=patience,
                     num_workers=0, device=device, saving_path=saving_path,
                     model_saving_strategy="best", verbose=True)
    if model_name == "brits":
        return BRITS(n_steps=L, n_features=n_features, **BRITS_HP,
                     batch_size=batch_size, epochs=epochs, patience=patience,
                     num_workers=0, device=device, saving_path=saving_path,
                     model_saving_strategy="best", verbose=True)
    raise ValueError(model_name)


def load_weights(imputer, path, device):
    """pypots<=1.3's .load() calls torch.load without weights_only=False, which
    torch 2.6 rejects; the .pypots file (>=0.13 format) is a dict holding
    'model_state_dict'. Load it ourselves (files are our own checkpoints)."""
    loaded = torch.load(path, map_location=device, weights_only=False)
    sd = (loaded["model_state_dict"]
          if isinstance(loaded, dict) and "model_state_dict" in loaded
          else loaded.state_dict())
    cur = imputer.model.state_dict()
    cur.update(sd)
    imputer.model.load_state_dict(cur)
    imputer.model.eval()
    return imputer


def load_split(dataset):
    df = pd.read_csv(DATASETS[dataset])
    return df.drop(columns=["date"]).to_numpy(np.float32)


def channel_stats(X):
    tr, _ = split_bounds(len(X))
    mu = np.nanmean(X[:tr], axis=0).astype(np.float32)
    sd = np.nanstd(X[:tr], axis=0).astype(np.float32)
    sd = np.where(sd > 1e-8, sd, 1.0).astype(np.float32)
    return mu, sd


# ------------------------------------------------------------ smoke ----

def cmd_smoke(args):
    tag = "smoke"
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    X = load_split("ETTh1")
    mu, sd = channel_stats(X)
    tr, va = split_bounds(len(X))
    mechs = VARIANTS["all"]
    Xtr, _ = build_augmented(X, 0, tr, 384, mechs, rng, mu, sd)
    Xva, Xva_ori = build_augmented(X, tr, va, 64, mechs, rng, mu, sd)
    print(f"smoke train {Xtr.shape} val {Xva.shape} "
          f"nan-frac {np.isnan(Xtr).mean():.3f}", flush=True)

    from pypots.imputation import SAITS
    model = SAITS(n_steps=L, n_features=X.shape[1], n_layers=1, d_model=64,
                  n_heads=2, d_k=32, d_v=32, d_ffn=64, dropout=0.1,
                  batch_size=32, epochs=2, patience=None, num_workers=0,
                  device=args.device,
                  saving_path=os.path.join(CKPT_DIR, tag + "_pypots"),
                  model_saving_strategy="best", verbose=True)
    t0 = time.time()
    model.fit({"X": Xtr}, val_set={"X": Xva, "X_ori": Xva_ori})
    print(f"smoke fit done in {(time.time() - t0) / 60:.1f} min", flush=True)

    # save -> reload roundtrip, then impute fresh masked windows
    cands = sorted((p for p in glob.glob(
        os.path.join(CKPT_DIR, tag + "_pypots", "**", "*.pypots"), recursive=True)
        if "tfevents" not in os.path.basename(p)), key=os.path.getmtime)
    assert cands, "no .pypots checkpoint written by fit()"
    src = cands[-1]
    model2 = SAITS(n_steps=L, n_features=X.shape[1], n_layers=1, d_model=64,
                   n_heads=2, d_k=32, d_v=32, d_ffn=64, dropout=0.1,
                   batch_size=32, epochs=1, num_workers=0, device=args.device,
                   verbose=False)
    load_weights(model2, src, args.device)
    Xte, _ = build_augmented(X, tr, va, 32, mechs, rng, mu, sd)
    out1 = model.predict({"X": Xte})
    out2 = model2.predict({"X": Xte})
    assert isinstance(out1, dict) and "imputation" in out1, out1.keys()
    imp1, imp2 = out1["imputation"], out2["imputation"]
    print(f"predict ok: keys={sorted(out1.keys())} shape={imp1.shape} "
          f"nan_left={int(np.isnan(imp1).sum())}", flush=True)
    diff = float(np.nanmean(np.abs(imp1 - imp2)))
    print(f"in-place vs reloaded mean|diff| = {diff:.5f}", flush=True)

    # bolt on imputed context: one mcar p=0.3 cell, 20 windows
    _, starts = load_windows(DATASETS["ETTh1"], 20, SEED)
    bolt = BoltModel(args.device)
    cfg = dict(mech="mcar", fill="linear", rate=0.3)
    res_lin = run_config(bolt, X, starts, cfg, n_seeds=2)
    C = X.shape[1]
    samples, gts = [], []
    for wi, s in enumerate(starts):
        xc = X[s:s + L].T.copy()
        for ms in range(2):
            mask = make_mask("mcar", 0.3, wi, ms, C)
            xw = ((xc.T - mu) / sd).astype(np.float32)
            xw[mask.T] = np.nan
            samples.append(xw)
            gts.append(X[s + L:s + L + H].T.copy())
    samples = np.stack(samples)
    imp = model.predict({"X": samples})["imputation"]
    ctx = (imp * sd + mu).transpose(0, 2, 1).reshape(-1, L).astype(np.float32)
    pred = bolt.predict_point(ctx).reshape(len(samples), C, H).astype(np.float64)
    gt = np.stack(gts).astype(np.float64)
    mse = float(((pred - gt) ** 2).mean())
    print(f"smoke ETTh1 mcar:0.3 (20 win): saits-filled bolt mse={mse:.4f} "
          f"vs linear-filled bolt mse={res_lin['mse']:.4f}", flush=True)
    print("SMOKE OK", flush=True)


# ------------------------------------------------------------ anchor ----

def cmd_anchor(args):
    """Gate: our harness (imported S5 code, same windows/seeds) must reproduce
    S5's bolt numbers within +-5% relMSE before any imputer numbers count."""
    s5 = json.load(open(S5_PATH))
    clean_s5 = s5["bolt"]["ETTh1"]["clean:none:0.0"]["mse"]
    cells = [dict(mech="clean", fill="none", rate=0.0),
             dict(mech="mcar", fill="linear", rate=0.3),
             dict(mech="mcar", fill="zero", rate=0.3),
             dict(mech="mnar_high", fill="linear", rate=0.7),
             dict(mech="mnar_high", fill="zero", rate=0.7)]
    bolt = BoltModel(args.device)
    X, starts = load_windows(DATASETS["ETTh1"], 300, SEED)
    out = {"tol": ANCHOR_TOL, "clean_s5": clean_s5, "cells": {}}
    gate = True
    for cfg in cells:
        t0 = time.time()
        res = run_config(bolt, X, starts, cfg, n_seeds=2)
        key = f"{cfg['mech']}:{cfg['fill']}:{cfg['rate']}"
        s5_mse = s5["bolt"]["ETTh1"][key]["mse"]
        rel_ours, rel_s5 = res["mse"] / clean_s5, s5_mse / clean_s5
        dev = abs(rel_ours - rel_s5) / rel_s5
        ok = bool(dev <= ANCHOR_TOL)
        gate &= ok
        out["cells"][key] = {"relMSE_ours": rel_ours, "relMSE_s5": rel_s5,
                             "dev": dev, "pass": ok}
        print(f"anchor {key:22s} ours={rel_ours:8.4f} s5={rel_s5:8.4f} "
              f"dev={dev * 100:5.2f}% {'OK' if ok else 'FAIL'} "
              f"({time.time() - t0:.0f}s)", flush=True)
    out["gate"] = "PASS" if gate else "FAIL"
    save_json(out, ANCHOR_PATH)
    print(f"ANCHOR GATE: {out['gate']}", flush=True)
    if not gate:
        sys.exit(1)


# ------------------------------------------------------------ train ----

def cmd_train(args):
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = tag_of(args.model, args.variant, args.dataset)
    seed = (SEED + zlib.crc32(tag.encode())) % (2 ** 31)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = load_split(args.dataset)
    N, C = X.shape
    mu, sd = channel_stats(X)
    tr, va = split_bounds(N)
    mechs = VARIANTS[args.variant]
    n_train = args.n_train or N_TRAIN[args.dataset]
    t0 = time.time()
    Xtr, _ = build_augmented(X, 0, tr, n_train, mechs, rng, mu, sd)
    Xva, Xva_ori = build_augmented(X, tr, va, N_VAL, mechs, rng, mu, sd)
    print(f"[{tag}] train {Xtr.shape} val {Xva.shape} mechs={mechs} "
          f"nan-frac {np.isnan(Xtr).mean():.3f} "
          f"(built in {(time.time() - t0) / 60:.1f} min)", flush=True)

    model = build_imputer(args.model, C, args.device, args.epochs, args.patience,
                          args.batch_size,
                          saving_path=os.path.join(CKPT_DIR, tag + "_pypots"))
    t1 = time.time()
    model.fit({"X": Xtr}, val_set={"X": Xva, "X_ori": Xva_ori})
    mins = (time.time() - t1) / 60
    cands = sorted((p for p in glob.glob(
        os.path.join(CKPT_DIR, tag + "_pypots", "**", "*.pypots"), recursive=True)
        if "tfevents" not in os.path.basename(p)), key=os.path.getmtime)
    assert cands, f"[{tag}] no checkpoint written"
    best = [p for p in cands if "best" in os.path.basename(p).lower()]
    src = best[-1] if best else cands[-1]
    dst = os.path.join(CKPT_DIR, tag + ".pypots")
    shutil.copy2(src, dst)
    meta = {"tag": tag, "model": args.model, "variant": args.variant,
            "dataset": args.dataset, "mechs": mechs, "n_train": n_train,
            "n_val": N_VAL, "aug_rate": AUG_RATE, "L": L, "seed": seed,
            "mu": mu.tolist(), "sd": sd.tolist(), "train_minutes": round(mins, 1),
            "hp": SAITS_HP if args.model == "saits" else BRITS_HP,
            "epochs_max": args.epochs, "patience": args.patience,
            "batch_size": args.batch_size, "ckpt": dst}
    save_json(meta, os.path.join(CKPT_DIR, tag + "_meta.json"))
    print(f"[{tag}] TRAIN DONE in {mins:.1f} min -> {dst}", flush=True)


# ------------------------------------------------------------ eval ----

def eval_config(imputer, bolt, X, starts, mech, rate, mu, sd, clean_mse):
    """One (mechanism, rate) cell: impute the masked S5 eval windows, forecast
    with bolt, return metrics incl. the S5 top-decile split (fixed per-channel
    clean-context q90 threshold, so the extreme set is identical across fills)."""
    C = X.shape[1]
    det = mech in ("mnar_high", "mnar_extreme")
    samples, gts, clean_std, masks, q90s = [], [], [], [], []
    for wi, s in enumerate(starts):
        xc = X[s:s + L].T.copy()              # [C, L]
        y = X[s + L:s + L + H].T.copy()       # [C, H]
        q90 = np.quantile(xc, 0.9, axis=1)    # per-channel context q90 [C]
        for ms in ((0,) if det else range(2)):
            mask = make_mask(mech, rate, wi, ms, C, x=xc)
            xw = ((xc.T - mu) / sd).astype(np.float32)  # [L, C]
            clean_std.append(xw.copy())
            xw[mask.T] = np.nan
            samples.append(xw)
            gts.append(y)
            masks.append(mask)
            q90s.append(q90)
    samples = np.stack(samples)               # [S, L, C]
    imp = imputer.predict({"X": samples})["imputation"]  # standardized
    clean_std = np.stack(clean_std)
    mask_arr = np.stack([m.T for m in masks])            # [S, L, C]
    with np.errstate(all="ignore"):
        imp_mae = float(np.abs(imp - clean_std)[mask_arr].mean())
        imp_mse = float(((imp - clean_std) ** 2)[mask_arr].mean())
    ctx = (imp * sd + mu).transpose(0, 2, 1)  # [S, C, L] original scale
    S = len(samples)
    pred = bolt.predict_point(
        ctx.reshape(S * C, L).astype(np.float32)).reshape(S, C, H)
    gt = np.stack(gts).astype(np.float64)
    err2 = (pred.astype(np.float64) - gt) ** 2
    mse_w = err2.mean(axis=(1, 2))
    nw = len(starts)
    top = gt > np.stack(q90s)[:, :, None]     # [S, C, H]
    with np.errstate(all="ignore"):
        td = np.nanmean(np.where(top, err2, np.nan), axis=(1, 2))
        rs = np.nanmean(np.where(~top, err2, np.nan), axis=(1, 2))
    td_w = np.nanmean(td.reshape(nw, -1), axis=1)
    rs_w = np.nanmean(rs.reshape(nw, -1), axis=1)
    return {"mse": float(mse_w.mean()),
            "mae": float(np.abs(pred - gt).mean()),
            "relMSE": float(mse_w.mean() / clean_mse),
            "imp_mae_std": imp_mae, "imp_mse_std": imp_mse,
            "achieved_rate": float(np.stack([m.mean() for m in masks]).mean()),
            "n_windows": nw, "n_seeds": 1 if det else 2,
            "mse_per_window": mse_w.reshape(nw, -1).mean(axis=1).tolist(),
            "mse_topdecile": float(np.nanmean(td)),
            "mse_rest": float(np.nanmean(rs)),
            "mse_topdecile_per_window":
                [None if np.isnan(v) else v for v in td_w.tolist()],
            "mse_rest_per_window":
                [None if np.isnan(v) else v for v in rs_w.tolist()]}


def cmd_eval(args):
    tag = tag_of(args.model, args.variant, args.dataset)
    meta = json.load(open(os.path.join(CKPT_DIR, tag + "_meta.json")))
    mu = np.array(meta["mu"], np.float32)
    sd = np.array(meta["sd"], np.float32)
    s5 = json.load(open(S5_PATH))
    clean_mse = s5["bolt"][args.dataset]["clean:none:0.0"]["mse"]
    X, starts = load_windows(DATASETS[args.dataset], 300, SEED)

    imputer = build_imputer(args.model, X.shape[1], args.device,
                            epochs=1, patience=None, batch_size=128)
    load_weights(imputer, meta["ckpt"], args.device)
    bolt = BoltModel(args.device)

    partial_path = os.path.join(CKPT_DIR, f"eval_{tag}.json")
    partial = json.load(open(partial_path)) if os.path.exists(partial_path) else {}
    for mech in MECHS:
        for rate in RATES:
            key = f"{mech}:{rate}"
            if key in partial and "mse_topdecile" in partial[key]:
                continue
            t0 = time.time()
            rec = eval_config(imputer, bolt, X, starts, mech, rate,
                              mu, sd, clean_mse)
            rec["seconds"] = round(time.time() - t0, 1)
            old = partial.get(key, {})
            old.update(rec)  # backfills top-decile keys on recompute
            partial[key] = old
            save_json(partial, partial_path)
            print(f"[{tag}] {key:20s} relMSE={rec['relMSE']:8.4f} "
                  f"td={rec['mse_topdecile']:9.3f} rest={rec['mse_rest']:8.3f} "
                  f"imp_mae={rec['imp_mae_std']:6.4f} ({rec['seconds']}s)",
                  flush=True)
    print(f"[{tag}] EVAL DONE", flush=True)


# ------------------------------------------------------------ collect ----

def cmd_collect(args):
    s5 = json.load(open(S5_PATH))
    s5f = json.load(open(S5FIX_PATH))
    results = {"meta": {
        "L": L, "H": H, "rates": RATES, "block": BLOCK, "seed": SEED,
        "datasets": DATASETS, "windows": 300, "seeds": 2,
        "mechanisms": MECHS,
        "variants": {"mb": "train masks ~ U{mcar, block} (mainstream)",
                     "all": "train masks ~ U{mcar, block, mnar_high, mnar_extreme}"},
        "imputers": {"saits": f"PyPOTS SAITS {SAITS_HP}",
                     "brits": f"PyPOTS BRITS {BRITS_HP}"},
        "train_region": "first 60% timeline, val 60-80%, eval = S5 test windows",
        "standardization": "per-channel train-split mean/std",
        "relMSE": "config MSE / S5 bolt clean:none:0.0 MSE of the same dataset",
        "downstream": "amazon/chronos-bolt-base fp32, median quantile",
    }}
    if os.path.exists(ANCHOR_PATH):
        results["anchor"] = json.load(open(ANCHOR_PATH))

    clean = {ds: s5["bolt"][ds]["clean:none:0.0"]["mse"] for ds in DATASETS}
    base = {}
    for ds in DATASETS:
        base[ds] = {}
        for mech in MECHS:
            for r in RATES:
                cell = {}
                for fill in ("zero", "linear"):
                    k = f"{mech}:{fill}:{r}"
                    if k in s5["bolt"][ds]:
                        cell[fill] = {"relMSE": s5["bolt"][ds][k]["mse"] / clean[ds],
                                      "mse_per_window":
                                          s5["bolt"][ds][k].get("mse_per_window")}
                k = f"{mech}:nan:{r}"
                if k in s5f["bolt"][ds]:
                    cell["nan"] = {"relMSE": s5f["bolt"][ds][k]["mse"] / clean[ds]}
                base[ds][f"{mech}:{r}"] = cell
    results["baselines"] = base
    results["clean_mse"] = clean

    res = {}
    for model in MODELS:
        for variant in VARIANTS:
            tagp = f"{model}_{variant}"
            for ds in DATASETS:
                p = os.path.join(CKPT_DIR, f"eval_{tagp}_{ds}.json")
                if os.path.exists(p):
                    res.setdefault(tagp, {})[ds] = json.load(open(p))
    results["results"] = res
    save_json(results, RESULTS_PATH)
    n = sum(len(v) for tagp in res.values() for v in tagp.values())
    print(f"collected -> {RESULTS_PATH}: variants={list(res.keys())} "
          f"cells={n}", flush=True)


# ------------------------------------------------------------ report ----

def cmd_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = json.load(open(RESULTS_PATH))
    dss = list(DATASETS)
    methods = [("zero", "#888888", "--", "o", "zero fill"),
               ("linear", "#000000", "-", "s", "linear interp"),
               ("nan", "#8c564b", ":", "^", "bolt native NaN"),
               ("saits_mb", "#1f77b4", "-", "o", "SAITS (train: mcar+block)"),
               ("saits_all", "#1f77b4", "-.", "v", "SAITS (train: +mnar)"),
               ("brits_mb", "#d62728", "-", "o", "BRITS (train: mcar+block)"),
               ("brits_all", "#d62728", "-.", "v", "BRITS (train: +mnar)")]

    def ds_avg_rel(mech, rate, method):
        vals = []
        for ds in dss:
            key = f"{mech}:{rate}"
            if method in ("zero", "linear", "nan"):
                cell = R["baselines"].get(ds, {}).get(key, {})
                v = cell.get(method, {}).get("relMSE")
            else:
                v = R["results"].get(method, {}).get(ds, {}).get(key, {}).get("relMSE")
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharex=True)
    for ax, mech in zip(axes, MECHS):
        for m, color, ls, mk, label in methods:
            ys = [ds_avg_rel(mech, r, m) for r in RATES]
            if all(y is None for y in ys):
                continue
            ax.plot(RATES, [y if y is not None else np.nan for y in ys],
                    color=color, ls=ls, marker=mk, ms=4, lw=1.6, label=label)
        ax.set_yscale("log")
        ax.set_title(mech)
        ax.set_xlabel("missing rate p")
        ax.grid(alpha=0.3, which="both")
        ax.axhline(1.0, color="k", lw=0.8, alpha=0.4)
    axes[0].set_ylabel("relMSE vs clean (dataset-avg, log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("S23: learned imputers (SAITS/BRITS) vs free fills — "
                 "chronos-bolt-base downstream relMSE", y=1.13, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "s23.png"), dpi=200, bbox_inches="tight")
    print("wrote s23.png", flush=True)

    # main table, dataset-avg relMSE
    print("\n== dataset-avg relMSE ==", flush=True)
    hdr = ["mech", "p"] + [m for m, *_ in methods]
    print(f"{hdr[0]:14s}{hdr[1]:>4s}" + "".join(f"{h:>12s}" for h in hdr[2:]),
          flush=True)
    for mech in MECHS:
        for r in RATES:
            row = [ds_avg_rel(mech, r, m) for m, *_ in methods]
            print(f"{mech:14s}{r:>4.1f}" + "".join(
                f"{v:12.3f}" if v is not None else f"{'—':>12s}" for v in row),
                flush=True)

    # per-dataset table for the learned imputers vs linear
    learned = [m for m in ("saits_mb", "saits_all", "brits_mb", "brits_all")
               if m in R["results"]]
    print("\n== per-dataset relMSE ==", flush=True)
    hdr = ["dataset", "mech", "p", "linear"] + learned
    print(f"{hdr[0]:9s}{hdr[1]:14s}{hdr[2]:>4s}{hdr[3]:>9s}"
          + "".join(f"{h:>11s}" for h in hdr[4:]), flush=True)
    for ds in dss:
        for mech in MECHS:
            for r in RATES:
                key = f"{mech}:{r}"
                lin = R["baselines"][ds].get(key, {}).get("linear", {}).get("relMSE")
                row = [R["results"][m].get(ds, {}).get(key, {}).get("relMSE")
                       for m in learned]
                print(f"{ds:9s}{mech:14s}{r:>4.1f}"
                      + (f"{lin:9.3f}" if lin is not None else f"{'—':>9s}")
                      + "".join(f"{v:11.3f}" if v is not None else f"{'—':>11s}"
                                for v in row), flush=True)

    # paired per-window comparison vs linear: delta relMSE mean +- se, win rate
    print("\n== paired vs linear (per-window ΔrelMSE, mean±se; win%% = fraction "
          "of windows where imputer beats linear) ==", flush=True)
    for m in learned:
        for mech in MECHS:
            parts = []
            for r in RATES:
                key = f"{mech}:{r}"
                diffs = []
                for ds in dss:
                    ours = R["results"][m].get(ds, {}).get(key, {})
                    lin = R["baselines"][ds].get(key, {}).get("linear", {})
                    ow = ours.get("mse_per_window")
                    lw = lin.get("mse_per_window")
                    if ow and lw and len(ow) == len(lw):
                        cm = R["clean_mse"][ds]
                        diffs += [(o - l) / cm for o, l in zip(ow, lw)]
                if diffs:
                    d = np.array(diffs)
                    parts.append(f"p={r}: {d.mean():+.3f}±{d.std()/np.sqrt(len(d)):.3f} "
                                 f"win={100 * (d < 0).mean():.0f}%")
            if parts:
                print(f"  {m:10s} {mech:14s} " + "  ".join(parts), flush=True)


# ------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("smoke", "anchor"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cuda")
    p = sub.add_parser("train")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-train", type=int, default=0)
    p = sub.add_parser("eval")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--variant", required=True, choices=list(VARIANTS))
    p.add_argument("--dataset", required=True, choices=list(DATASETS))
    p.add_argument("--device", default="cuda")
    sub.add_parser("collect")
    sub.add_parser("report")
    args = ap.parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)
    {"smoke": cmd_smoke, "anchor": cmd_anchor, "train": cmd_train,
     "eval": cmd_eval, "collect": cmd_collect, "report": cmd_report}[args.mode](args)


if __name__ == "__main__":
    main()
