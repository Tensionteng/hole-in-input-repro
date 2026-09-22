#!/usr/bin/env python
"""Screen S6-E, piece 2: conformal honesty wrapper for bolt prediction intervals.

S5 Track C found that bolt's native 80% prediction intervals fail SILENTLY
under informative missingness: coverage collapses (0.74 -> 0.22 under
mnar_high) while the intervals NARROW -- the model is confidently wrong.
Split conformal is the standard distribution-free fix; the open question is
whether the calibration set must match the deployment missingness mechanism.

Design (per dataset):
  * eval windows  = the SAME 300 test-region windows as the S5 grids
    (run_s5_missing.load_windows, seed 20250810).
  * calibration windows = 300 further starts from the same test region,
    disjoint from the eval starts (seed 20250810+999).
  * deployment fill = linear interpolation (the S5 best practice); the mask
    seen at calibration is synthetic, at a chosen mechanism and rate.
  * nonconformity for the native 80% interval [q10, q90]: E_t = max(q10_t -
    y_t, y_t - q90_t), per horizon step t = 1..96, over calibration series.
    Widened interval: [q10_t - w_t, q90_t + w_t], w_t = the split-conformal
    quantile ceil((n+1)*0.8)/n of {E_t}.
  * variants: native (no conformal); matched (calibration mechanism+rate =
    deployment); mismatched (calibrate mcar, deploy mnar_high) -- the mismatch
    cost; rate-pooled adaptive (one calibration pool spanning all rates, w_t(p)
    by per-rate-bin quantiles + linear interpolation in p).

Metrics per config: pointwise coverage and mean interval width (pooled and
per-step). bolt only, 1 mask seed. Saved incrementally to
s6_conformal_results.json.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import run_s5_missing as s5

HERE = s5.HERE
RESULTS_PATH = os.path.join(HERE, "s6_conformal_results.json")
L, H = s5.L, s5.H
ALPHA = 0.8  # target coverage of the native [q10, q90] interval
MECHS = ("mcar", "mnar_high")


def calib_starts(ds, n=300):
    """300 test-region starts disjoint from the S5 eval windows."""
    X, eval_starts = s5.load_windows(s5.DATASETS[ds], 300, s5.SEED)
    N = len(X)
    test_start = int(0.8 * N)
    lo, hi = test_start - L, N - L - H
    valid = np.arange(lo, hi + 1)
    pool = np.setdiff1d(valid, eval_starts)
    rng = np.random.default_rng(s5.SEED + 999)
    return X, np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))


@torch.no_grad()
def predict_intervals(pipe, ctx, batch=512):
    """ctx [n, L] -> q10, q90 of the 96-step forecast."""
    q10, q90 = [], []
    for i in range(0, len(ctx), batch):
        xb = torch.from_numpy(ctx[i:i + batch]).to("cuda")
        q = pipe.predict(xb, prediction_length=H)  # [b, 9, H]
        q10.append(q[:, 0, :].float().cpu().numpy())
        q90.append(q[:, 8, :].float().cpu().numpy())
    return np.concatenate(q10), np.concatenate(q90)


def build_split(pipe, X, starts, mech, rate, n_seeds=1):
    """Linear-filled contexts + ground truth for one (mech, rate) over starts.
    Returns q10, q90, y each [n_series, H]."""
    C = X.shape[1]
    ctxs, gts = [], []
    for wi, s in enumerate(starts):
        x_clean = X[s:s + L].T.copy()
        y = X[s + L:s + L + H].T.copy()
        for ms in range(n_seeds):
            mask = (np.zeros((C, L), bool) if mech == "clean"
                    else s5.make_mask(mech, rate, wi, ms, C, x=x_clean))
            ctxs.append(s5.fill_context(x_clean, mask, "none" if mech == "clean" else "linear"))
            gts.append(y)
    S = len(ctxs)
    ctx = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
    q10, q90 = predict_intervals(pipe, ctx)
    gt = np.stack(gts).reshape(S * C, s5.H).astype(np.float64)
    return q10.astype(np.float64), q90.astype(np.float64), gt


def conformal_w(E, alpha=ALPHA):
    """E: [n, H] nonconformity -> per-step widening [H] (split-conformal quantile)."""
    n = E.shape[0]
    k = min(int(np.ceil((n + 1) * alpha)), n)
    return np.sort(E, axis=0)[k - 1]  # [H]


def coverage(q10, q90, y, w=None):
    if w is None:
        w = np.zeros(q10.shape[-1])
    inside = (y >= q10 - w) & (y <= q90 + w)
    return inside.mean(axis=0), (q90 - q10 + 2 * w).mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RESULTS_PATH)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(s5.SEED)
    np.random.seed(s5.SEED)
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map="cuda", torch_dtype=torch.float32)

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    results.setdefault("meta", {
        "piece": "S6-E piece 2: split-conformal wrapper for bolt 80% intervals",
        "alpha": ALPHA, "fill": "linear", "mask_seeds": 1,
        "eval": "the 300 S5 eval windows", "calib": "300 disjoint test-region windows",
        "nonconformity": "E_t = max(q10_t - y_t, y_t - q90_t), per horizon step"})

    dss = ["ETTh1"] if args.tiny else list(s5.DATASETS)
    rates = [0.3] if args.tiny else s5.RATES
    for ds in dss:
        t0 = time.time()
        X, ev_starts = s5.load_windows(s5.DATASETS[ds], 300, s5.SEED)
        _, ca_starts = calib_starts(ds)
        mech_rate = [("clean", 0.0)] + [(m, r) for m in MECHS for r in rates]
        calib, evalp = {}, {}
        for mech, rate in mech_rate:
            calib[(mech, rate)] = build_split(pipe, X, ca_starts, mech, rate)
            evalp[(mech, rate)] = build_split(pipe, X, ev_starts, mech, rate)

        def nonconf(split):
            q10, q90, y = split
            return np.maximum(q10 - y, y - q90)  # [n, H]

        recs = {}
        for dmech, rate in mech_rate:
            variants = [(dmech, "")]                      # matched calibration
            if dmech == "mnar_high":
                variants.append(("mcar", "mismatch"))     # calibrate mcar, deploy mnar
            if dmech != "clean":
                variants.append(("pooled", ""))           # rate-adaptive pooled pool
            for cmech, vtag in variants:
                if cmech == "pooled":
                    # rate-adaptive: one pool over all rates of the DEPLOY
                    # mechanism; w_t(p) = per-rate-bin conformal quantiles
                    E_bin = np.stack([conformal_w(nonconf(calib[(dmech, r)]))
                                      for r in rates])  # [n_rates, H]
                    w = E_bin[rates.index(rate)]  # linear interp in p; grid pt = exact
                    tag = f"{dmech}|calib=pooled_adaptive|{rate}"
                else:
                    w = conformal_w(nonconf(calib[(cmech, rate)]))
                    tag = f"{dmech}|calib={cmech}{'(' + vtag + ')' if vtag else ''}|{rate}"
                q10, q90, y = evalp[(dmech, rate)]
                cov_nat, wid_nat = coverage(q10, q90, y)
                cov_cf, wid_cf = coverage(q10, q90, y, w)
                recs[tag] = {
                    "coverage_native": float(cov_nat.mean()),
                    "coverage_conformal": float(cov_cf.mean()),
                    "width_native": float(wid_nat.mean()),
                    "width_conformal": float(wid_cf.mean()),
                    "coverage_native_per_step": cov_nat.tolist(),
                    "coverage_conformal_per_step": cov_cf.tolist(),
                    "w_per_step": w.tolist(),
                }
        results.setdefault("bolt", {})[ds] = recs
        tmp = args.out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(results, fh)
        os.replace(tmp, args.out)
        print(f"{ds} done ({time.time() - t0:.0f}s)", flush=True)
        for tag, rec in recs.items():
            print(f"  {tag:38s} cov {rec['coverage_native']:.3f} -> "
                  f"{rec['coverage_conformal']:.3f}  width {rec['width_native']:.2f} -> "
                  f"{rec['width_conformal']:.2f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
