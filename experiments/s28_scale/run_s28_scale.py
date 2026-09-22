#!/usr/bin/env python
"""S28: does robustness to missing data improve with model scale?

The field's default answer to almost any TSFM weakness is "scale it up". This round asks
whether scale buys missingness robustness, using the four released Chronos-Bolt sizes --
tiny 8.7M, mini 21.2M, small 47.7M, base 205.3M (a 24x span, identical architecture and
identical training recipe, so scale is the only variable).

The measurement that matters is RELATIVE degradation: each model is normalised by its OWN
clean-context forecast, so "robustness" means "how much worse does missingness make me",
not "how good am I". Absolute clean error is reported alongside, to confirm that scale does
buy accuracy on clean data -- which makes a flat robustness curve a statement about what
scale does NOT buy.

Pre-registered prediction: clean error falls with scale, while relative degradation is
FLAT. Rationale: S25 Part 0 located the damage in the input interface (content zeroed at
masked positions, statistics computed before masking), and the interface is identical
across the four sizes -- it does not get better with parameters. If instead robustness
improves markedly with scale, the paper's architectural story weakens and "just scale up"
becomes a legitimate answer.

Windows, masks and metrics are S25's (hence S5's, gate-verified). Paired per-window relMSE,
median over windows.
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
import run_s25_twofloor as s25

MODELS = [("tiny", os.path.join(ROOT, "models_local", "chronos-bolt-tiny")),
          ("mini", os.path.join(ROOT, "models_local", "chronos-bolt-mini")),
          ("small", os.path.join(ROOT, "models_local", "chronos-bolt-small")),
          ("base", "amazon/chronos-bolt-base")]
MECHS = s25.MECHS
RATES = (0.1, 0.3, 0.5, 0.7)
FILLS = ("linear", "zero", "nan")
H = s25.H
OUT = os.path.join(HERE, "s28_results.json")


class SizedBolt(s25.Bolt):
    def __init__(self, path, device="cuda"):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            path, device_map=device, torch_dtype=torch.float32)
        self.device = device
        self.model = self.pipe.inner_model if hasattr(self.pipe, "inner_model") else self.pipe.model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        assert self.model.chronos_config.prediction_length >= H


def gate(res, bolt):
    """bolt-base must still reproduce the stored S5 cells."""
    out = {}
    X, st = s25.load_windows(s25.DATASETS["ETTh1"], 300, s25.SEED, s25.H_GATE)
    for key, stored in s25.ANCHORS.items():
        mech, fill, rate = key.split(":")
        rate = float(rate)
        seeds = (0,) if mech in ("clean", "mnar_high", "mnar_extreme") else (0, 1)
        vals = []
        for ms in seeds:
            _, filled, _, gt = s25.build_eval_batch(X, st, mech, rate, s25.H_GATE,
                                                    fill=fill, mask_seed=ms)
            pred = bolt.median_rollout_np(filled, s25.H_GATE).astype(np.float64)
            vals.append(float(((pred - gt.astype(np.float64)) ** 2).mean()))
        v = float(np.mean(vals))
        d = abs(v - stored) / stored
        out[key] = {"stored": stored, "rerun": v, "rel_dev": d, "pass": d <= 0.05}
        print(f"GATE {key:24s} stored={stored:10.5f} rerun={v:10.5f} dev={d:+.4%} "
              f"{'PASS' if d <= 0.05 else 'FAIL'}", flush=True)
    res["gate"] = out
    assert all(v["pass"] for v in out.values()), "gate FAILED"
    return res


def run_model(res, name, path, n_win=300):
    bolt = SizedBolt(path)
    print(f"\n== {name}: {bolt.n_params/1e6:.1f}M params ==", flush=True)
    if name == "base":
        res = gate(res, bolt)
    cells = {}
    for ds, dpath in s25.DATASETS.items():
        X, st = s25.load_windows(dpath, n_win, s25.SEED, H)
        _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
        base = ((bolt.median_np(cl) - gt_c) ** 2).mean(1)
        ok = base > 1e-12
        cells[f"{ds}|clean"] = {"mse_mean": float(base.mean()),
                                "mse_median": float(np.median(base)),
                                "n": int(ok.sum())}
        for mech in MECHS:
            for rate in RATES:
                for fill in FILLS:
                    _, f_, mk, gt = s25.build_eval_batch(X, st, mech, rate, H, fill=fill)
                    e = ((bolt.median_np(f_) - gt) ** 2).mean(1)
                    r = e[ok] / base[ok]
                    cells[f"{ds}|{mech}|{rate}|{fill}"] = {
                        "rel_median": float(np.median(r)), "rel_mean": float(r.mean()),
                        "rel_q75": float(np.quantile(r, 0.75))}
        print(f"  {ds:8s} clean MSE med={np.median(base):9.4f}", flush=True)
    summ = {}
    for m in MECHS:
        for rate in RATES:
            v = float(np.mean([min(cells[f"{d}|{m}|{rate}|{f}"]["rel_median"] for f in FILLS)
                               for d in s25.DATASETS]))
            summ[f"{m}|{rate}"] = v
    print("  best-fill relMSE (median, dataset-avg): " + "  ".join(
        f"{m}@0.7={summ[f'{m}|0.7']:.3f}" for m in MECHS), flush=True)
    res.setdefault("models", {})[name] = {
        "n_params": int(bolt.n_params), "path": path, "cells": cells, "summary": summ,
        "clean_mse_median_avg": float(np.mean([cells[f"{d}|clean"]["mse_median"]
                                               for d in s25.DATASETS]))}
    del bolt
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-win", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"models": [m[0] for m in MODELS], "rates": list(RATES),
                            "mechs": list(MECHS), "fills": list(FILLS), "H": H,
                            "metric": "paired per-window relMSE vs each model's OWN clean"})
    nwin = 30 if args.smoke else args.n_win
    for name, path in MODELS:
        t0 = time.time()
        res = run_model(res, name, path, n_win=nwin)
        json.dump(res, open(OUT + ".tmp", "w"))
        os.replace(OUT + ".tmp", OUT)
        print(f"  [{name}] done in {time.time()-t0:.0f}s", flush=True)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
