#!/usr/bin/env python
"""Screen S6-E, piece 1: censoring-aware imputation for MNAR contexts.

S5 established that under value-censored missingness (mnar_high / mnar_extreme)
NO repair beats plain linear interpolation, and that the residual damage is
systematic: linear bridges across censored peaks bias forecasts down exactly on
extreme futures (bolt top-decile/rest MSE ratio 2.4 vs 0.7 under mcar). This
script tests whether a cheap, deployable two-stage fix helps:

  Stage 1 -- mechanism detector (per window-channel), using only the masked
  context. Features: r_bar = mean rank (within the observed values' empirical
  CDF) of the linear-bridge values at missing positions; x_bar = mean of
  max(r, 1-r); e_bar = mean |rank - 0.5| of the observed endpoints flanking
  missing runs; mean_run = mean length of missing runs. Rules (a priori):
      r_bar >= 0.62                          -> "upper"     (bridges sit high)
      e_bar >= 0.265 and r_bar < 0.56        -> "two-tail"  (gap ends at edges)
      x_bar >= 0.80                          -> "two-tail"
      mean_run >= 6                          -> "clustered" (long runs; block=24)
      else                                   -> "random"
  Rationale: for iid/block missingness gap endpoints are uniform in the
  observed CDF (r_bar ~ 0.5, e_bar ~ 0.25 exactly); censored peaks force
  bridges between high flank values (r_bar -> 1); two-tail censoring pushes
  gap endpoints toward both edges of the observed range (e_bar up, r_bar
  stays ~ 0.5); block masks are value-blind with 24-long runs. Censoring
  tests come FIRST because at high rates censored points merge into long runs
  that would otherwise trip the run-length rule. Accuracy is evaluated on the
  synthetic grids where the truth is known.

  Stage 2 -- tail-aware imputation, gated by the detector (a channel judged
  non-censored gets plain linear, so the imputer reduces to linear when no
  censoring is present). For each missing run with BOTH endpoints observed and
  endpoint ranks consistent with censoring (min endpoint rank >= 0.5 for
  upper-censoring, max(|rank-0.5|) >= 0.5 i.e. endpoints near a tail for
  two-tail), a tent-shaped bump is added on top of the linear bridge, peaking
  at the gap midpoint. Two bump-height estimators:
    * tail_tri  -- "peak excess": mean excess of OBSERVED local extrema over
      their 3-point linear bridges, shrunk by evidence count:
      e = mean(e_i) * n/(n+5). Conservative: observed peaks are the ones that
      survived censoring, so e underestimates the excess of censored peaks.
    * tail_tobit -- Tobit-style conditional expectation: with mu = median(obs),
      sigma = 1.4826*MAD(obs), censoring point tau = max(obs) (upper) or
      min(obs) (lower), bump apex = E[x | x > tau] = mu + sigma*lambda(alpha),
      alpha = (tau-mu)/sigma, lambda = pdf/(1-cdf) (and symmetrically below).
  Fills are evaluated with bolt and timesfm on the mnar_high / mnar_extreme
  grids (same windows/seeds as S5 -> identical masks), reporting relMSE vs
  clean and vs linear, and the top-decile/rest MSE ratio that diagnosed the
  bias in S5. mcar/block at p=0.3 are included as reduce-to-linear sanity.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import run_s5_missing as s5

HERE = s5.HERE
RESULTS_PATH = os.path.join(HERE, "s6_mnarfix_results.json")
L = s5.L
TAIL_FILLS = ("tail_tri", "tail_tobit")


# ------------------------------------------------------ detector ----------

def bridge_rank_features(x1d, mask):
    """x1d: [L] with observed values (mask True = missing). Returns features."""
    valid = np.flatnonzero(~mask)
    nmiss = int(mask.sum())
    dm = np.diff(mask.astype(np.int8))
    n_runs = int((dm == 1).sum()) + (1 if mask[0] else 0)
    mean_run = nmiss / max(n_runs, 1)
    obs = x1d[valid]
    filled = np.interp(np.arange(L), valid, obs)
    obs_sorted = np.sort(obs)
    n_obs = len(obs)
    ranks = np.searchsorted(obs_sorted, filled[mask]) / n_obs  # in [0,1]
    r_bar = float(ranks.mean())
    x_bar = float(np.maximum(ranks, 1.0 - ranks).mean())
    # e_bar: mean |rank - 0.5| of the OBSERVED endpoints flanking missing runs.
    # Value-blind masks (mcar/block) give exactly ~0.25; tail censoring pushes
    # gap endpoints toward the edges of the observed range -> elevated.
    starts = list(np.flatnonzero(dm == 1) + 1) + ([0] if mask[0] else [])
    ends = list(np.flatnonzero(dm == -1)) + ([L - 1] if mask[-1] else [])
    end_absrank = []
    for i, j in zip(sorted(starts), sorted(ends)):
        for t in (i - 1, j + 1):
            if 0 <= t < L and not mask[t]:
                end_absrank.append(abs(np.searchsorted(obs_sorted, x1d[t]) / n_obs - 0.5))
    e_bar = float(np.mean(end_absrank)) if end_absrank else 0.25
    return dict(mean_run=mean_run, r_bar=r_bar, x_bar=x_bar, e_bar=e_bar,
                n_runs=n_runs)


def detect_mechanism(x1d, mask):
    """Rule-based; thresholds fixed a priori (see module docstring)."""
    f = bridge_rank_features(x1d, mask)
    # count-aware two-tail threshold: e_bar is 0.25 for value-blind masks with
    # per-endpoint sd ~0.0735; require a >=3sd (floor 0.015) elevation
    thr_e = 0.25 + max(0.015, 3.0 * 0.0735 / np.sqrt(max(2 * f["n_runs"], 1)))
    if f["r_bar"] >= 0.62:
        mech = "upper"
    elif (f["e_bar"] >= thr_e and f["r_bar"] < 0.56) or f["x_bar"] >= 0.80:
        mech = "two-tail"
    elif f["mean_run"] >= 6:
        mech = "clustered"
    else:
        mech = "random"
    return mech, f


# ------------------------------------------------------ imputer -----------

def _extrema_excess(x1d, valid):
    """Mean excess of observed local maxima (minima) over their 3-point bridges,
    shrunk toward 0 by evidence count n/(n+5)."""
    hi, lo = [], []
    vset = set(valid.tolist())
    for t in valid:
        if t - 1 in vset and t + 1 in vset:
            b = 0.5 * (x1d[t - 1] + x1d[t + 1])
            if x1d[t] > x1d[t - 1] and x1d[t] >= x1d[t + 1]:
                hi.append(x1d[t] - b)
            elif x1d[t] < x1d[t - 1] and x1d[t] <= x1d[t + 1]:
                lo.append(b - x1d[t])
    e_hi = float(np.mean(hi)) * len(hi) / (len(hi) + 5.0) if hi else 0.0
    e_lo = float(np.mean(lo)) * len(lo) / (len(lo) + 5.0) if lo else 0.0
    return e_hi, e_lo


def _tobit_levels(obs):
    """E[x | x > tau_hi] and E[x | x < tau_lo] under a robust Gaussian model."""
    from scipy.stats import norm
    med = float(np.median(obs))
    sigma = max(1.4826 * float(np.median(np.abs(obs - med))), 1e-8)
    out = {}
    for tag, tau in (("hi", float(obs.max())), ("lo", float(obs.min()))):
        a = (tau - med) / sigma
        a = min(max(a, -8.0), 8.0)
        if tag == "hi":
            lam = norm.pdf(a) / max(1.0 - norm.cdf(a), 1e-12)
            out["hi"] = med + sigma * lam
        else:
            lam = norm.pdf(a) / max(norm.cdf(a), 1e-12)
            out["lo"] = med - sigma * lam
    return out


def tail_impute(x1d, mask, mode):
    """Detector-gated tail-aware fill of one channel. Returns (filled, mech)."""
    base = s5.fill_context(x1d[None, :], mask[None, :], "linear")[0]
    if mask.sum() == 0:
        return base, "clean"
    mech, feats = detect_mechanism(x1d, mask)
    if mech not in ("upper", "two-tail"):
        return base, mech  # reduce to linear
    valid = np.flatnonzero(~mask)
    obs = x1d[valid]
    ranks_obs_sorted = np.sort(obs)
    n_obs = len(obs)

    def rank_of(v):
        return np.searchsorted(ranks_obs_sorted, v) / n_obs

    e_hi, e_lo = _extrema_excess(x1d, valid)
    tb = _tobit_levels(obs)

    out = base.copy()
    dm = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(dm == 1) + 1) + ([0] if mask[0] else [])
    ends = list(np.flatnonzero(dm == -1)) + ([L - 1] if mask[-1] else [])
    for i, j in zip(sorted(starts), sorted(ends)):
        if i == 0 or j == L - 1:
            continue  # boundary gaps: one-sided evidence only, stay linear
        l_v, r_v = x1d[i - 1], x1d[j + 1]
        r_l, r_r = rank_of(l_v), rank_of(r_v)
        hi_gap = min(r_l, r_r) >= 0.5                     # both endpoints high
        lo_gap = max(r_l, r_r) <= 0.5                     # both endpoints low
        span = j - i + 1
        mid = 0.5 * (i + j)
        for t in range(i, j + 1):
            tent = 1.0 - abs(t - mid) / (span / 2.0 + 1e-9)
            if mech == "upper" and hi_gap:
                target = (base[t] + e_hi) if mode == "tail_tri" else tb["hi"]
                out[t] = max(base[t], base[t] + (target - base[t]) * tent)
            elif mech == "two-tail":
                if hi_gap:
                    target = (base[t] + e_hi) if mode == "tail_tri" else tb["hi"]
                    out[t] = max(base[t], base[t] + (target - base[t]) * tent)
                elif lo_gap:
                    target = (base[t] - e_lo) if mode == "tail_tri" else tb["lo"]
                    out[t] = min(base[t], base[t] + (target - base[t]) * tent)
    return out, mech


# ------------------------------------------------------ evaluation --------

def run_config(model, X, starts, mech, fill, rate, n_seeds):
    """Same mask/window conventions as s5.run_config; adds detector + fills."""
    C = X.shape[1]
    det = mech in ("clean", "mnar_high", "mnar_extreme")
    ctxs, gts, masks, q90s, det_hits = [], [], [], [], []
    for wi, s in enumerate(starts):
        x_clean = X[s:s + L].T.copy()
        y = X[s + L:s + L + s5.H].T.copy()
        q90s_w = np.quantile(x_clean, 0.9, axis=1)
        for ms in ((0,) if det else range(n_seeds)):
            mask = (np.zeros((C, L), bool) if mech == "clean"
                    else s5.make_mask(mech, rate, wi, ms, C, x=x_clean))
            if fill in TAIL_FILLS and mech != "clean":
                filled = np.empty_like(x_clean)
                for c in range(C):
                    filled[c], _ = tail_impute(x_clean[c], mask[c], fill)
            else:
                filled = s5.fill_context(x_clean, mask, "none" if mech == "clean" else fill)
            ctxs.append(filled)
            gts.append(y)
            masks.append(mask)
            q90s.append(q90s_w)
    S = len(ctxs)
    ctx = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
    pred = model.predict_point(ctx).reshape(S, C, s5.H).astype(np.float64)
    gt = np.stack(gts).astype(np.float64)
    err2 = (pred - gt) ** 2
    mse = err2.mean(axis=(1, 2))
    mae = np.abs(pred - gt).mean(axis=(1, 2))
    nw = len(starts)
    res = {
        "mech": mech, "fill": fill, "rate": rate,
        "n_windows": nw, "n_seeds": 1 if det else n_seeds,
        "mse": float(mse.mean()), "mae": float(mae.mean()),
        "achieved_rate": float(np.stack([m.mean() for m in masks]).mean()),
        "mse_per_window": mse.reshape(nw, -1).mean(axis=1).tolist(),
        "mae_per_window": mae.reshape(nw, -1).mean(axis=1).tolist(),
    }
    if mech != "clean":  # top-decile diagnostic (threshold from clean context)
        top = gt > np.stack(q90s)[:, :, None]
        with np.errstate(all="ignore"):
            td = np.nanmean(np.where(top, err2, np.nan), axis=(1, 2))
            rs = np.nanmean(np.where(~top, err2, np.nan), axis=(1, 2))
        res["mse_topdecile"] = float(np.nanmean(td))
        res["mse_rest"] = float(np.nanmean(rs))
        res["topdecile_frac"] = float(top.mean())
    return res


def eval_detector(X, starts, n_seeds=1):
    """Accuracy of detect_mechanism on synthetic masks with known truth."""
    truth = {"mcar": "random", "block": "clustered",
             "mnar_high": "upper", "mnar_extreme": "two-tail"}
    C = X.shape[1]
    out = {}
    for mech, want in truth.items():
        for rate in s5.RATES:
            n_ok = n_tot = 0
            feat_acc = []
            for wi, s in enumerate(starts):
                x_clean = X[s:s + L].T
                for ms in (range(n_seeds) if mech in ("mcar", "block") else (0,)):
                    mask = s5.make_mask(mech, rate, wi, ms, C, x=x_clean)
                    for c in range(C):
                        got, f = detect_mechanism(x_clean[c], mask[c])
                        n_ok += int(got == want)
                        n_tot += 1
                        feat_acc.append((f["r_bar"], f["x_bar"], f["mean_run"], f["e_bar"]))
            feat_acc = np.array(feat_acc)
            out[f"{mech}:{rate}"] = {
                "acc": n_ok / n_tot, "n": n_tot,
                "r_bar": float(feat_acc[:, 0].mean()),
                "x_bar": float(feat_acc[:, 1].mean()),
                "mean_run": float(feat_acc[:, 2].mean()),
                "e_bar": float(feat_acc[:, 3].mean())}
    return out


def save_results(results, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--models", default="bolt,timesfm")
    ap.add_argument("--out", default=RESULTS_PATH)
    args = ap.parse_args()

    torch.manual_seed(s5.SEED)
    np.random.seed(s5.SEED)

    if args.tiny:
        bolt = s5.BoltModel("cuda")
        X, starts = s5.load_windows(s5.DATASETS["ETTh1"], 20, s5.SEED)
        print("detector features @ ETTh1 tiny (truth -> r_bar/x_bar/mean_run/acc):")
        det = eval_detector(X, starts)
        for k, v in det.items():
            print(f"  {k:18s} r_bar={v['r_bar']:.3f} x_bar={v['x_bar']:.3f} "
                  f"run={v['mean_run']:5.1f} acc={v['acc']:.3f}")
        for mech in ("mnar_high", "mnar_extreme"):
            for fill in ("linear",) + TAIL_FILLS:
                t0 = time.time()
                res = run_config(bolt, X, starts, mech, fill, 0.3, n_seeds=2)
                print(f"{mech:13s} {fill:10s} mse={res['mse']:8.4f} "
                      f"td={res['mse_topdecile']:8.4f} rest={res['mse_rest']:8.4f} "
                      f"({time.time() - t0:.1f}s)", flush=True)
        res = run_config(bolt, X, starts, "mcar", "tail_tri", 0.3, n_seeds=2)
        print(f"mcar sanity   tail_tri   mse={res['mse']:8.4f} "
              f"(expect ~= mcar:linear 12.8624)", flush=True)
        return

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
        print(f"resume: loaded {args.out}", flush=True)
    results.setdefault("meta", {
        "piece": "S6-E piece 1: censoring-aware imputation",
        "detector": "rules on bridge-rank features (see script docstring)",
        "fills": {"tail_tri": "linear + shrunk observed-extrema excess tent",
                  "tail_tobit": "linear + Tobit E[x|x>tau] tent, tau=observed max/min"},
        "conventions": "same windows/seeds as s5_missing_results.json"})

    # ---------------- detector accuracy (CPU, cheap) ----------------
    det_out = results.setdefault("detector", {})
    for ds in s5.DATASETS:
        if ds in det_out:
            continue
        X, starts = s5.load_windows(s5.DATASETS[ds], 300, s5.SEED)
        t0 = time.time()
        det_out[ds] = eval_detector(X, starts)
        save_results(results, args.out)
        print(f"detector {ds} done ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- imputation grid ----------------
    grid = []
    for r in s5.RATES:
        for mech in ("mnar_high", "mnar_extreme"):
            for fill in TAIL_FILLS:
                grid.append((mech, fill, r))
    for fill in TAIL_FILLS:  # reduce-to-linear sanity
        grid.append(("mcar", fill, 0.3))
        grid.append(("block", fill, 0.3))

    for name in args.models.split(","):
        cls = {"bolt": s5.BoltModel, "timesfm": s5.TimesFmModel}[name]
        model = cls("cuda")
        results.setdefault(name, {})
        n_win = 300 if name == "bolt" else 100
        n_seeds = 2 if name == "bolt" else 1
        jobs = [(ds, cfg) for ds in s5.DATASETS for cfg in grid]
        loaded = {}
        for ds, (mech, fill, rate) in jobs:
            key = f"{mech}:{fill}:{rate}"
            if key in results[name].get(ds, {}):
                continue
            if ds not in loaded:
                loaded[ds] = s5.load_windows(s5.DATASETS[ds], 300, s5.SEED)
            X, starts_all = loaded[ds]
            t0 = time.time()
            res = run_config(model, X, starts_all[:n_win], mech, fill, rate, n_seeds)
            res["seconds"] = round(time.time() - t0, 1)
            results[name].setdefault(ds, {})[key] = res
            save_results(results, args.out)
            print(f"{name:8s} {ds:8s} {key:28s} mse={res['mse']:9.4f} "
                  f"td={res.get('mse_topdecile', float('nan')):9.2f} "
                  f"rest={res.get('mse_rest', float('nan')):8.2f} ({res['seconds']}s)",
                  flush=True)
        del model
        torch.cuda.empty_cache()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
