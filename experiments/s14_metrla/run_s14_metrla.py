#!/usr/bin/env python
"""S14: second REAL missingness mechanism -- METR-LA traffic sensor failures.

Claim under test (pre-registered in s14_metrla_notes.md BEFORE model runs):
S6/S10 (Penmanshiel 2016 wind curtailment) found the optimal context fill is
mechanism-dependent: under VALUE-CENSORED (MNAR) missingness zero-fill beat
linear 2-4.5x and Tobit failed. METR-LA missingness is the opposite geometry:
loop-sensor failures encoded as exact 0.0 (DCRNN convention), long blocks,
value-INdependent. Prediction: the fill ranking REVERSES -- linear and
native-NaN win, zero/keep damage.

Pipeline mirrors Track D (run_s6_real_censor.py) / S10 (run_s10_real.py):
  univariate per sensor; context CTX=512 (42.7h @5min), horizon H=96 (8h);
  miss windows: >= MIN_MISS missing points in context, target fully observed;
  ctrl windows: fully observed context+target, matched 1:1 per sensor to the
  miss windows by nearest time-of-day (no replacement, seeded).
Fills: keep (as-recorded, i.e. 0s kept -- physically identical to zero on
  METR-LA), zero, linear (np.interp, Track D convention), nan (model-native:
  chronos-bolt masks NaN in attention; timesfm internally np.interp's).
Metrics: MSE/MAE, NMSE = MSE / max(var(target), VAR_FLOOR) per window,
  mean/median NMSE, mse/persistence (Track D), top-decile split (Track D).
  Persistence: last OBSERVED value (primary; Track D's last-as-recorded is
  undefined when the last context point is missing-encoded-0; both stored).
Conformal: native [q10,q90] -> per-step split conformal, target 0.9; cal/test
  = first/second half BY TIME within each group; naive = calibrate on ctrl,
  mech-aware = calibrate on miss, pooled = mix.
Detector: S6/S10 bridge-rank rules, ported verbatim (CTX-agnostic).

Data: dataset_metrla/metr-la.h5 (canonical DCRNN file; the official repo now
  only links Google Drive/Baidu, so an exact-copy HF mirror was used -- see
  dataset_metrla/DATA_README.md; signature verified: 34272 x 207, 5-min,
  2012-03-01..2012-06-27, 8.11% exact zeros, max 70 mph).

Modes: explore, smoke, anchor, conformal, detector, gate, figure.
Parallelism: one process per model per GPU (--model/--device). Results
flock-merged into s14_metrla_results.json.
"""
import argparse
import fcntl
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import pandas as pd
import torch

CTX, H = 512, 96               # 42.7h context, 8h horizon @ 5-min
STRIDE = 12                    # candidate grid every 1h
MIN_MISS = 26                  # >= ~5% of context missing (Track D used 7/144)
QUOTA = 3                      # max miss windows per sensor (cap ~621 pairs)
SEED = 20250813
VAR_FLOOR = 4.0                # mph^2 (2 mph std) floor for NMSE denominator
H5 = os.path.join(HERE, "dataset_metrla", "metr-la.h5")
OUT = os.path.join(HERE, "s14_metrla_results.json")
ALPHA = 0.9                    # conformal target
# Track D ctrl:clean reference (run_s6_real_censor.py, Penmanshiel):
TRACKD_CTRL = {"bolt": dict(nmse=9.91, nmse_median=1.73),
               "timesfm": dict(nmse=5.85, nmse_median=1.63),
               "moirai": dict(nmse=7.2, nmse_median=2.01)}


# ------------------------------------------------------------------ models --
class BoltModel:
    name, batch = "bolt", 1024

    def __init__(self, device):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
        self.device = device

    @torch.no_grad()
    def predict_point(self, ctx):  # ctx: [n, CTX] float32 (NaN ok) -> [n, H]
        outs = []
        for i in range(0, len(ctx), self.batch):
            xb = torch.from_numpy(np.ascontiguousarray(ctx[i:i + self.batch])).to(self.device)
            q = self.pipe.predict(xb, prediction_length=H)
            outs.append(q[:, 4, :].float().cpu().numpy())
        return np.concatenate(outs)

    @torch.no_grad()
    def predict_quantiles(self, ctx):
        q10, q90 = [], []
        for i in range(0, len(ctx), self.batch):
            xb = torch.from_numpy(np.ascontiguousarray(ctx[i:i + self.batch])).to(self.device)
            q = self.pipe.predict(xb, prediction_length=H)  # [b, 9, H]
            q10.append(q[:, 0, :].float().cpu().numpy())
            q90.append(q[:, 8, :].float().cpu().numpy())
        return np.concatenate(q10), np.concatenate(q90)


class TimesFmModel:
    name, batch = "timesfm", 512

    def __init__(self, device):
        import timesfm
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            "google/timesfm-2.5-200m-pytorch")
        self.m.compile(timesfm.ForecastConfig(
            max_context=1024, max_horizon=128, per_core_batch_size=512,
            normalize_inputs=True, use_continuous_quantile_head=True))

    def predict_point(self, ctx):
        outs = []
        for i in range(0, len(ctx), self.batch):
            chunk = [row for row in ctx[i:i + self.batch]]
            pf, _ = self.m.forecast(horizon=H, inputs=chunk)
            outs.append(np.asarray(pf, dtype=np.float32))
        return np.concatenate(outs)

    def predict_quantiles(self, ctx):
        q10, q90 = [], []
        for i in range(0, len(ctx), self.batch):
            chunk = [row for row in ctx[i:i + self.batch]]
            _, qf = self.m.forecast(horizon=H, inputs=chunk)
            qf = np.asarray(qf)  # [b, H, 10]: col0=mean, cols1..9 = q0.1..q0.9
            q10.append(qf[:, :, 1].astype(np.float32))
            q90.append(qf[:, :, 9].astype(np.float32))
        return np.concatenate(q10), np.concatenate(q90)


class MoiraiModel:
    """Salesforce/moirai-1.1-R-base via uni2ts; median of 20 samples."""
    name, batch = "moirai", 250
    num_samples = 20

    def __init__(self, device):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        forecast = MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base"),
            prediction_length=H, context_length=CTX, patch_size=16,
            num_samples=self.num_samples, target_dim=1,
            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        self.predictor = forecast.create_predictor(batch_size=self.batch,
                                                   device=device)

    def predict_point(self, ctx):
        # fixed seed per call: sampling-based -> make configs comparable
        # (identical inputs must give identical outputs; S10 moirai was
        # unseeded and its mean NMSE was tail-draw noise)
        torch.manual_seed(20250813)
        outs = []
        for i in range(0, len(ctx), self.batch):
            chunk = ctx[i:i + self.batch]
            ds = [{"target": row, "start": pd.Period("2000-01-01", freq="5min")}
                  for row in chunk]
            fcsts = list(self.predictor.predict(ds))
            outs.append(np.stack([np.asarray(f.median, dtype=np.float32)
                                  for f in fcsts]))
        return np.concatenate(outs)


MODEL_REGISTRY = {"bolt": BoltModel, "timesfm": TimesFmModel, "moirai": MoiraiModel}


# ------------------------------------------------------------------- data ---
def load_matrix():
    df = pd.read_hdf(H5)
    return df.index, [str(c) for c in df.columns], df.values.astype(np.float32)


def build_windows(seed=SEED):
    """Track-D-style window list. miss: >=MIN_MISS missing (=0.0) in context,
    target fully observed. ctrl: fully observed context+target, matched 1:1
    per sensor to miss windows by nearest time-of-day (no replacement)."""
    idx, sensors, V = load_matrix()
    T, S = V.shape
    rng = np.random.default_rng(seed)
    windows = []
    for si in range(S):
        ob = V[:, si] != 0.0
        miss_cand, ctrl_cand = [], []
        for t in range(CTX, T - H + 1, STRIDE):
            if not ob[t:t + H].all():
                continue
            nmiss = CTX - int(ob[t - CTX:t].sum())
            if nmiss >= MIN_MISS:
                miss_cand.append((t, nmiss / CTX))
            elif nmiss == 0:
                ctrl_cand.append(t)
        if not miss_cand or not ctrl_cand:
            continue
        k = min(QUOTA, len(miss_cand))
        take = rng.choice(len(miss_cand), size=k, replace=False)
        tod = idx.hour * 60 + idx.minute          # minutes-of-day per timestep
        pool = list(ctrl_cand)
        base = len(windows)
        kept = 0
        for j in sorted(take):
            t, rate = miss_cand[j]
            d = np.abs(tod[pool] - tod[t])
            d = np.minimum(d, 1440 - d)           # circular tod distance
            bi = int(np.argmin(d))
            tc = pool.pop(bi)
            windows.append(dict(window_id=len(windows), sensor=sensors[si],
                                start=int(t), kind="miss", rate=float(rate),
                                pair=-1, tod_match_min=int(d[bi])))
            windows.append(dict(window_id=len(windows), sensor=sensors[si],
                                start=int(tc), kind="ctrl", rate=float(rate),
                                pair=base + 2 * kept, tod_match_min=int(d[bi])))
            kept += 1
    n_miss = sum(1 for w in windows if w["kind"] == "miss")
    print(f"[data] {len(windows)} windows ({n_miss} miss + {len(windows) - n_miss} ctrl), "
          f"{len({w['sensor'] for w in windows})} sensors; miss rate mean "
          f"{np.mean([w['rate'] for w in windows if w['kind'] == 'miss']):.3f}; "
          f"tod match median {np.median([w['tod_match_min'] for w in windows if w['kind'] == 'miss']):.0f} min")
    return windows


def extract(windows):
    """Per-window arrays: ctx_rec (as-recorded, 0=missing), ctx_obs (NaN at
    missing), mask (True=missing), tgt (as-recorded; fully observed)."""
    _, sensors, V = load_matrix()
    sidx = {s: i for i, s in enumerate(sensors)}
    W = len(windows)
    ctx_rec = np.zeros((W, CTX), np.float32)
    ctx_obs = np.zeros((W, CTX), np.float32)
    mask = np.zeros((W, CTX), bool)
    tgt = np.zeros((W, H), np.float32)
    for i, w in enumerate(windows):
        v = V[:, sidx[w["sensor"]]]
        t = w["start"]
        c = v[t - CTX:t]
        ctx_rec[i] = c
        m = c == 0.0
        mask[i] = m
        co = c.copy()
        co[m] = np.nan
        ctx_obs[i] = co
        tgt[i] = v[t:t + H]
    return dict(ctx_rec=ctx_rec, ctx_obs=ctx_obs, mask=mask, tgt=tgt)


def group_ids(windows, kind):
    return np.array([i for i, w in enumerate(windows) if w["kind"] == kind])


# ------------------------------------------------------------------- fills --
def fill_window(fill, x_obs, x_rec):
    """One context. x_obs has NaN at missing; x_rec as-recorded (0=missing).
    'nan' returns x_obs unchanged (model-native handling)."""
    if fill in ("keep", "clean"):   # clean: ctrl windows are fully observed
        return x_rec.astype(np.float64)
    if fill == "zero":
        return np.nan_to_num(x_obs.astype(np.float64))
    if fill == "nan":
        return x_obs.astype(np.float64)
    valid = np.flatnonzero(np.isfinite(x_obs))
    if len(valid) == 0:
        return np.zeros(CTX)
    if fill == "linear":
        return np.interp(np.arange(CTX), valid, x_obs[valid])
    raise ValueError(fill)


def build_X(fill, ids, A):
    return np.stack([fill_window(fill, A["ctx_obs"][i], A["ctx_rec"][i])
                     for i in ids]).astype(np.float32)


# ------------------------------------------------- point-forecast metrics ---
def point_metrics(pred, Y, q90_ref, ctx_rec_ids, ctx_obs_ids):
    """Track D metrics + median NMSE + persistence (obs & rec variants)."""
    err2 = (pred - Y) ** 2
    mse_w = err2.mean(axis=1)
    var = np.maximum(Y.var(axis=1), VAR_FLOOR)
    nmse_w = mse_w / var
    q90 = np.array([np.nanquantile(q90_ref[k], 0.9) for k in range(len(Y))])
    top = Y > q90[:, None]
    with np.errstate(all="ignore"):
        td = np.nanmean(np.where(top, err2, np.nan), axis=1)
        rs = np.nanmean(np.where(~top, err2, np.nan), axis=1)
    # persistence: last OBSERVED context value (primary) / last as-recorded
    last_obs = np.array([row[np.isfinite(row)][-1] if np.isfinite(row).any()
                         else 0.0 for row in ctx_obs_ids])
    pers_obs = np.repeat(last_obs[:, None], H, axis=1)
    pers_rec = np.repeat(ctx_rec_ids[:, -1:], H, axis=1)
    mse_pers_obs = ((pers_obs - Y) ** 2).mean(axis=1)
    mse_pers_rec = ((pers_rec - Y) ** 2).mean(axis=1)
    return dict(
        mse=float(mse_w.mean()), mae=float(np.abs(pred - Y).mean()),
        nmse=float(nmse_w.mean()), nmse_median=float(np.median(nmse_w)),
        mse_topdecile=float(np.nanmean(td)), mse_rest=float(np.nanmean(rs)),
        topdecile_frac=float(top.mean()),
        persistence_mse=float(mse_pers_obs.mean()),
        mse_over_persistence=float(mse_w.mean() / max(mse_pers_obs.mean(), 1e-9)),
        persistence_mse_rec=float(mse_pers_rec.mean()),
        mse_over_persistence_rec=float(mse_w.mean() / max(mse_pers_rec.mean(), 1e-9)),
        target_var_median=float(np.median(Y.var(axis=1))),
        nmse_per_window=nmse_w.tolist(), mse_per_window=mse_w.tolist(),
        var_per_window=Y.var(axis=1).tolist(),
    )


def run_point_config(model, windows, A, group, fill, ids):
    X = build_X(fill, ids, A)
    Y = A["tgt"][ids].astype(np.float64)
    pred = model.predict_point(X).astype(np.float64)
    if not np.isfinite(pred).all():
        n_bad = int((~np.isfinite(pred).all(axis=1)).sum())
        print(f"  [warn] {model.name}/{group}/{fill}: {n_bad} non-finite preds",
              flush=True)
    # Track D q90 reference: observed (NaN) context for miss, recorded for ctrl
    q90_ref = A["ctx_obs"][ids] if group == "miss" else A["ctx_rec"][ids]
    rec = point_metrics(pred, Y, q90_ref, A["ctx_rec"][ids], A["ctx_obs"][ids])
    rec.update(model=model.name, group=group, fill=fill, n_windows=len(ids),
               window_ids=[int(i) for i in ids],
               mean_missing_rate=float(A["mask"][ids].mean()))
    return rec


# ------------------------------------ mechanism detector (ported from S6) ---
def bridge_rank_features(x1d, mask):
    """S6 detector features, CTX-agnostic. x1d values at masked points unused."""
    L = len(x1d)
    valid = np.flatnonzero(~mask)
    nmiss = int(mask.sum())
    dm = np.diff(mask.astype(np.int8))
    n_runs = int((dm == 1).sum()) + (1 if mask[0] else 0)
    mean_run = nmiss / max(n_runs, 1)
    obs = x1d[valid]
    filled = np.interp(np.arange(L), valid, obs)
    obs_sorted = np.sort(obs)
    n_obs = len(obs)
    ranks = np.searchsorted(obs_sorted, filled[mask]) / n_obs
    r_bar = float(ranks.mean())
    x_bar = float(np.maximum(ranks, 1.0 - ranks).mean())
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
    """S6's a priori rules (unchanged thresholds)."""
    if mask.sum() == 0:
        return "clean", {}
    if (~mask).sum() < 2:
        return "undetermined", {}
    f = bridge_rank_features(x1d, mask)
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


def run_detector(windows, A):
    ids = group_ids(windows, "miss")
    labels, feats = [], []
    for i in ids:
        mech, f = detect_mechanism(A["ctx_rec"][i], A["mask"][i])
        labels.append(mech)
        if f:
            feats.append(f)
    cnt = {k: labels.count(k)
           for k in ("upper", "two-tail", "clustered", "random", "undetermined")}
    n = max(len(labels), 1)
    return dict(n_windows=len(ids), label_counts=cnt,
                label_frac={k: v / n for k, v in cnt.items()},
                r_bar_mean=float(np.mean([f["r_bar"] for f in feats])),
                r_bar_median=float(np.median([f["r_bar"] for f in feats])),
                r_bar_p10=float(np.percentile([f["r_bar"] for f in feats], 10)),
                r_bar_p90=float(np.percentile([f["r_bar"] for f in feats], 90)),
                r_bar_per_window=[round(f["r_bar"], 4) for f in feats],
                e_bar_mean=float(np.mean([f["e_bar"] for f in feats])),
                mean_run_mean=float(np.mean([f["mean_run"] for f in feats])),
                mean_run_median=float(np.median([f["mean_run"] for f in feats])))


# --------------------------------------------------------- conformal --------
def time_split(windows, kind):
    """First/second half BY TIME (user requirement: temporal split)."""
    ids = [i for i, w in enumerate(windows) if w["kind"] == kind]
    ids.sort(key=lambda i: (windows[i]["start"], windows[i]["sensor"]))
    half = len(ids) // 2
    return np.array(ids[:half]), np.array(ids[half:])


def conformal_w(E, alpha=ALPHA):
    n = E.shape[0]
    k = min(int(np.ceil((n + 1) * alpha)), n)
    return np.sort(E, axis=0)[k - 1]


def cov_width(q10, q90, y, w=None):
    if w is None:
        w = np.zeros(q10.shape[-1])
    inside = (y >= q10 - w) & (y <= q90 + w)
    return dict(coverage=float(inside.mean()), width=float((q90 - q10 + 2 * w).mean()),
                coverage_per_window=inside.mean(axis=1).tolist(),
                width_per_step=(q90 - q10 + 2 * w).mean(axis=0).tolist())


def run_conformal(model, windows, A):
    cal_m, test_m = time_split(windows, "miss")
    cal_c, test_c = time_split(windows, "ctrl")
    recs = {}
    for fill in ("zero", "linear", "nan"):
        splits = {}
        for name, ids in [("miss_cal", cal_m), ("miss_test", test_m),
                          ("ctrl_cal", cal_c), ("ctrl_test", test_c)]:
            X = build_X(fill, ids, A)
            q10, q90 = model.predict_quantiles(X)
            q10, q90 = q10.astype(np.float64), q90.astype(np.float64)
            y = A["tgt"][ids].astype(np.float64)
            splits[name] = (q10, q90, y, ids)
        E = {name: np.maximum(q10 - y, y - q90)
             for name, (q10, q90, y, _) in splits.items()}
        W_cal = {"naive": conformal_w(E["ctrl_cal"]),
                 "mechaware": conformal_w(E["miss_cal"]),
                 "pooled": conformal_w(np.concatenate([E["ctrl_cal"], E["miss_cal"]]))}
        for deploy in ("miss_test", "ctrl_test"):
            q10, q90, y, ids = splits[deploy]
            variants = [("native", None)] + list(W_cal.items())
            if deploy == "ctrl_test":
                variants = [("native", None), ("naive", W_cal["naive"]),
                            ("mechaware", W_cal["mechaware"])]
            for vname, w in variants:
                r = cov_width(q10, q90, y, w)
                r.update(model=model.name, fill=fill, deploy=deploy, variant=vname,
                         n_cal={"naive": len(cal_c), "mechaware": len(cal_m),
                                "pooled": len(cal_c) + len(cal_m)}.get(vname, 0),
                         n_test=len(ids), window_ids=[int(i) for i in ids],
                         target=ALPHA)
                key = f"conformal|{model.name}|{fill}|{deploy}|{vname}"
                recs[key] = r
                print(f"    {key:44s} cov={r['coverage']:.3f} wid={r['width']:7.2f}",
                      flush=True)
    return recs


# ----------------------------------------------------------------- explore --
def run_explore():
    idx, sensors, V = load_matrix()
    T, S = V.shape
    miss = V == 0.0
    print(f"[explore] {T} x {S}, {idx[0]} .. {idx[-1]}, "
          f"zeros {miss.mean() * 100:.2f}%", flush=True)

    per_sensor = miss.mean(axis=0)
    # run lengths of zero-runs, pooled over sensors
    runs = []
    for si in range(S):
        m = miss[:, si]
        dm = np.diff(m.astype(np.int8))
        starts = list(np.flatnonzero(dm == 1) + 1) + ([0] if m[0] else [])
        ends = list(np.flatnonzero(dm == -1)) + ([T - 1] if m[-1] else [])
        runs += [j - i + 1 for i, j in zip(sorted(starts), sorted(ends))]
    runs = np.array(runs)
    rbins = [1, 2, 3, 6, 12, 36, 72, 288, 10 ** 9]
    rhist = [int(((runs >= lo) & (runs < hi)).sum()) for lo, hi in zip(rbins[:-1], rbins[1:])]

    # value-dependence: observed speeds adjacent to zero-runs vs all observed
    bnd = []
    for si in range(S):
        m = miss[:, si]
        dm = np.diff(m.astype(np.int8))
        starts = list(np.flatnonzero(dm == 1) + 1) + ([0] if m[0] else [])
        ends = list(np.flatnonzero(dm == -1)) + ([T - 1] if m[-1] else [])
        for i, j in zip(sorted(starts), sorted(ends)):
            for t in (i - 1, j + 1):
                if 0 <= t < T and not m[t]:
                    bnd.append(V[t, si])
    bnd = np.array(bnd)
    obs_all = V[~miss]
    vbins = np.arange(0, 75, 5)
    h_bnd = np.histogram(bnd, bins=vbins)[0] / max(len(bnd), 1)
    h_all = np.histogram(obs_all, bins=vbins)[0] / max(len(obs_all), 1)

    # hourly: missing rate vs mean speed (24 values) -> correlation
    hours = idx.hour.values
    hmiss, hspd = [], []
    for h in range(24):
        sel = hours == h
        vh, mh = V[sel], miss[sel]
        hmiss.append(mh.mean())
        hspd.append(vh[~mh].mean())
    hmiss, hspd = np.array(hmiss), np.array(hspd)
    r_hour = float(np.corrcoef(hmiss, hspd)[0, 1])
    # day-of-week
    dow = idx.dayofweek.values
    dowmiss = [float(miss[dow == d].mean()) for d in range(7)]

    # inter-sensor correlation (sample 40 sensors, pairwise-complete)
    rng = np.random.default_rng(0)
    sub = rng.choice(S, size=40, replace=False)
    corrs, mcorrs = [], []
    for a in range(len(sub)):
        for b in range(a + 1, len(sub)):
            x, y = V[:, sub[a]], V[:, sub[b]]
            ok = (x != 0) & (y != 0)
            if ok.sum() > 1000:
                corrs.append(float(np.corrcoef(x[ok], y[ok])[0, 1]))
            mcorrs.append(float(np.corrcoef(miss[:, sub[a]].astype(float),
                                            miss[:, sub[b]].astype(float))[0, 1]))

    # window candidate counts under S14 rules
    n_miss_cand, n_ctrl_cand, n_sens_with = 0, 0, 0
    ob = ~miss
    for si in range(S):
        mc = cc = 0
        o = ob[:, si]
        for t in range(CTX, T - H + 1, STRIDE):
            if not o[t:t + H].all():
                continue
            nm = CTX - int(o[t - CTX:t].sum())
            if nm >= MIN_MISS:
                mc += 1
            elif nm == 0:
                cc += 1
        n_miss_cand += mc
        n_ctrl_cand += cc
        n_sens_with += bool(mc and cc)

    return dict(
        shape=[T, S], start=str(idx[0]), end=str(idx[-1]),
        freq_seconds=int(np.median(np.diff(idx.values).astype('timedelta64[s]').astype(int))),
        value_min=float(V.min()), value_max=float(V.max()),
        missing_frac=float(miss.mean()),
        per_sensor_missing=dict(
            min=float(per_sensor.min()), p25=float(np.percentile(per_sensor, 25)),
            median=float(np.median(per_sensor)), mean=float(per_sensor.mean()),
            p75=float(np.percentile(per_sensor, 75)), max=float(per_sensor.max()),
            n_sensors_gt20pct=int((per_sensor > 0.2).sum()),
            n_sensors_lt1pct=int((per_sensor < 0.01).sum())),
        runlen=dict(n_runs=int(len(runs)), median=float(np.median(runs)),
                    mean=float(runs.mean()), p90=float(np.percentile(runs, 90)),
                    p99=float(np.percentile(runs, 99)), max=int(runs.max()),
                    frac_ge12=float((runs >= 12).mean()),
                    frac_ge288=float((runs >= 288).mean()),
                    bin_edges=rbins, bin_counts=rhist,
                    frac_zeros_in_runs_ge12=float(runs[runs >= 12].sum() / runs.sum())),
        boundary_speed=dict(n=int(len(bnd)), mean=float(bnd.mean()),
                            median=float(np.median(bnd)),
                            p10=float(np.percentile(bnd, 10)),
                            p90=float(np.percentile(bnd, 90))),
        all_speed=dict(n=int(len(obs_all)), mean=float(obs_all.mean()),
                       median=float(np.median(obs_all)),
                       p10=float(np.percentile(obs_all, 10)),
                       p90=float(np.percentile(obs_all, 90))),
        speed_hist_bins=vbins.tolist(), speed_hist_boundary=h_bnd.tolist(),
        speed_hist_all=h_all.tolist(),
        hourly_missing=hmiss.tolist(), hourly_speed=hspd.tolist(),
        hourly_corr_missing_speed=r_hour,
        dow_missing=dowmiss,
        sensor_speed_corr=dict(n_pairs=len(corrs), mean=float(np.mean(corrs)),
                               median=float(np.median(corrs)),
                               p10=float(np.percentile(corrs, 10)),
                               p90=float(np.percentile(corrs, 90))),
        sensor_mask_corr=dict(n_pairs=len(mcorrs), mean=float(np.mean(mcorrs)),
                              median=float(np.median(mcorrs)),
                              p90=float(np.percentile(mcorrs, 90))),
        window_rule=dict(ctx=CTX, horizon=H, stride=STRIDE, min_miss=MIN_MISS,
                         quota=QUOTA, n_miss_cand=n_miss_cand,
                         n_ctrl_cand=n_ctrl_cand,
                         n_sensors_with_both=n_sens_with),
    )


# -------------------------------------------------------------------- io ----
def save_records(recs, path=OUT):
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        txt = fh.read()
        try:
            data = json.loads(txt) if txt.strip() else {}
        except json.JSONDecodeError:
            data = {}
        data.setdefault("meta", dict(
            design="S14: second real missingness mechanism (METR-LA sensor "
                   "failure, 0-encoded) -- does the optimal fill reverse vs "
                   "Penmanshiel value-censoring?",
            ctx=CTX, horizon=H, stride=STRIDE, min_miss=MIN_MISS, quota=QUOTA,
            seed=SEED, var_floor=VAR_FLOOR, alpha=ALPHA,
            provenance="dataset_metrla/metr-la.h5 = canonical DCRNN METR-LA; "
                       "official repo hosts only Drive/Baidu links; exact-copy "
                       "HF mirror jimmygao3218/METRLA used (see DATA_README.md)",
            fills=dict(keep="as-recorded (0s kept; == zero on METR-LA)",
                       zero="missing->0", linear="missing->np.interp",
                       nan="missing->NaN, model-native (bolt masks; timesfm "
                           "internally np.interp)"),
            persistence="last OBSERVED context value (primary); last "
                        "as-recorded also stored (Track D variant)"))
        data.setdefault("records", {}).update(recs)
        fh.seek(0)
        fh.truncate()
        json.dump(data, fh)
        fh.flush()
        os.fsync(fh.fileno())
        fcntl.flock(fh, fcntl.LOCK_UN)


def load_records(path=OUT):
    with open(path) as f:
        return json.load(f)["records"]


# ------------------------------------------------------------------- gate ---
def run_gate(recs):
    rows, all_pass = [], True

    def add(name, ok, detail):
        nonlocal all_pass
        all_pass &= bool(ok)
        rows.append(dict(check=name, passed=bool(ok), detail=detail))
        print(f"  [{ 'OK' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    # 1. identity: fills must not touch fully-observed ctrl windows
    for model in ("bolt", "timesfm", "moirai"):
        base = recs.get(f"anchor|{model}|ctrl|clean")
        if base is None:
            continue
        for fill in ("zero", "linear", "nan"):
            r = recs.get(f"anchor|{model}|ctrl|{fill}")
            if r is None:
                continue
            d = float(np.max(np.abs(np.array(r["nmse_per_window"])
                                    - np.array(base["nmse_per_window"]))))
            add(f"{model} ctrl {fill}==clean", d < 1e-5, f"max|dNMSE|={d:.2e}")
    # 2. keep == zero on miss windows (METR-LA missing IS 0)
    for model in ("bolt", "timesfm", "moirai"):
        a, b = recs.get(f"anchor|{model}|miss|keep"), recs.get(f"anchor|{model}|miss|zero")
        if a is None or b is None:
            continue
        d = float(np.max(np.abs(np.array(a["nmse_per_window"])
                                - np.array(b["nmse_per_window"]))))
        add(f"{model} miss keep==zero", d < 1e-5, f"max|dNMSE|={d:.2e}")
    # 3. anchor vs Track D clean level (order-of-magnitude) + beat persistence
    for model, ref in TRACKD_CTRL.items():
        r = recs.get(f"anchor|{model}|ctrl|clean")
        if r is None:
            continue
        ratio = r["nmse"] / ref["nmse"]
        add(f"{model} ctrl clean NMSE order vs TrackD",
            np.isfinite(r["nmse"]) and 0.02 <= ratio <= 50,
            f"S14={r['nmse']:.3f} (med {r['nmse_median']:.3f}) TrackD={ref['nmse']:.2f} "
            f"ratio={ratio:.3f}")
        add(f"{model} ctrl beats persistence",
            r["mse_over_persistence"] < 1.0,
            f"mse/pers={r['mse_over_persistence']:.3f}")
    # 4. all NMSE finite
    bad = [k for k, r in recs.items() if k.startswith("anchor|")
           and not np.isfinite(r["nmse"])]
    add("all anchor NMSE finite", not bad, f"bad={bad}")
    print(f"GATE {'PASS' if all_pass else 'FAIL'}", flush=True)
    return {"gate": dict(passed=bool(all_pass), rows=rows)}


# ------------------------------------------------------------------ figure --
def run_figure(recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    ex = recs["explore"]

    # A: run-length distribution
    ax = axes[0][0]
    edges, counts = ex["runlen"]["bin_edges"], ex["runlen"]["bin_counts"]
    labels = ["1", "2", "3-5", "6-11", "12-35", "36-71", "72-287", ">=288"]
    ax.bar(range(len(counts)), counts, color="#88419d")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, fontsize=8, rotation=30)
    ax.set_yscale("log")
    ax.set_xlabel("zero-run length (5-min points)")
    ax.set_ylabel("# runs (log)")
    ax.set_title(f"A. Missing geometry: blocky sensor outages\n"
                 f"(missing {ex['missing_frac'] * 100:.1f}% overall; "
                 f"{ex['runlen']['frac_zeros_in_runs_ge12']:.0%} of missing pts in "
                 f"runs>=12; max run {ex['runlen']['max']})", fontsize=10)

    # B: value-independence of missingness
    ax = axes[0][1]
    bins = np.array(ex["speed_hist_bins"])
    ctr = (bins[:-1] + bins[1:]) / 2
    ax.plot(ctr, ex["speed_hist_all"], "k-", lw=2, label="all observed speeds")
    ax.plot(ctr, ex["speed_hist_boundary"], "r--", lw=2,
            label="speeds at missing-run boundaries")
    ax.set_xlabel("speed (mph)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    ax.set_title(f"B. Missingness is NOT value-dependent\n"
                 f"(boundary median {ex['boundary_speed']['median']:.1f} vs overall "
                 f"{ex['all_speed']['median']:.1f} mph; hourly miss-rate vs speed "
                 f"r={ex['hourly_corr_missing_speed']:.2f})", fontsize=10)

    # C: hourly pattern
    ax = axes[0][2]
    ax2 = ax.twinx()
    ax.plot(range(24), np.array(ex["hourly_missing"]) * 100, "r-o", ms=3,
            label="missing %")
    ax2.plot(range(24), ex["hourly_speed"], "b-s", ms=3, label="mean speed")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("missing rate (%)", color="r")
    ax2.set_ylabel("mean speed (mph)", color="b")
    ax.set_title("C. Missingness vs time-of-day\n(sensor outages, not congestion "
                 "zeros)", fontsize=10)

    # D: fills on miss windows (mean NMSE) + ctrl clean reference
    ax = axes[1][0]
    models = [m for m in ("bolt", "timesfm", "moirai")
              if f"anchor|{m}|miss|linear" in recs]
    fills = ["keep", "zero", "linear", "nan"]
    wdt = 0.8 / max(len(models), 1)
    for mi, model in enumerate(models):
        vals = [recs[f"anchor|{model}|miss|{f}"]["nmse"] for f in fills]
        x = np.arange(len(fills)) + (mi - (len(models) - 1) / 2) * wdt
        ax.bar(x, vals, width=wdt, label=model, alpha=0.85,
               color=["#1f77b4", "#ff7f0e", "#2ca02c"][mi])
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        clean = recs[f"anchor|{model}|ctrl|clean"]["nmse"]
        if mi == 0:
            ax.axhline(clean, color="k", ls="--", lw=1)
            ax.text(len(fills) - 0.5, clean, f"clean ctrl ({clean:.2f})",
                    fontsize=8, ha="right", va="bottom")
    ax.set_xticks(np.arange(len(fills)))
    ax.set_xticklabels(fills)
    ax.set_ylabel("mean NMSE on miss windows")
    zl = [recs[f"anchor|{m}|miss|zero"]["nmse"] / recs[f"anchor|{m}|miss|linear"]["nmse"]
          for m in models]
    ax.set_title(f"D. Fill ranking on sensor-failure missingness\n"
                 f"(zero/linear = {'/'.join(f'{v:.2f}' for v in zl)}; Penmanshiel "
                 f"was 0.41-0.57, zero won)", fontsize=10)
    ax.legend(fontsize=8)

    # E: conformal coverage on miss test
    ax = axes[1][1]
    variants = ["native", "naive", "pooled", "mechaware"]
    qmodels = [m for m in ("bolt", "timesfm")
               if f"conformal|{m}|linear|miss_test|native" in recs]
    wdt = 0.8 / 6
    ci = 0
    for model in qmodels:
        for fill in ("zero", "linear", "nan"):
            vals = []
            for v in variants:
                k = f"conformal|{model}|{fill}|miss_test|{v}"
                vals.append(recs[k]["coverage"] if k in recs else np.nan)
            x = np.arange(len(variants)) + (ci - 2.5) * wdt
            ax.bar(x, vals, width=wdt, label=f"{model}/{fill}",
                   color=["#1f77b4", "#9ecae1", "#08306b",
                          "#d62728", "#f4b6b6", "#67000d"][ci])
            ci += 1
    ax.axhline(ALPHA, color="k", ls="--", lw=1)
    ax.text(0, ALPHA + 0.01, f"target {ALPHA}", fontsize=8)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(["native", "naive\n(ctrl cal)", "pooled", "mech-aware\n(miss cal)"],
                       fontsize=8)
    ax.set_ylabel("coverage on miss test windows")
    ax.set_ylim(0, 1.02)
    ax.set_title("E. Split-conformal on miss windows (time split)\n"
                 "(native [q10,q90] widened to 90%)", fontsize=10)
    ax.legend(fontsize=6, ncol=2)

    # F: detector labels
    ax = axes[1][2]
    det = recs.get("detector|miss")
    if det:
        mechs = ["upper", "two-tail", "clustered", "random", "undetermined"]
        vals = [det["label_frac"].get(m, 0) for m in mechs]
        ax.bar(range(len(mechs)), vals,
               color=["#2ca02c", "#9467bd", "#8c564b", "#c7c7c7", "#eeeeee"])
        ax.set_xticks(range(len(mechs)))
        ax.set_xticklabels(mechs, fontsize=8, rotation=20)
        ax.set_ylabel(f"fraction of {det['n_windows']} miss windows")
        ax.set_title(f"F. S6/S10 mechanism detector on METR-LA\n"
                     f"(r_bar mean {det['r_bar_mean']:.2f}; Penmanshiel: 65% "
                     f"upper/two-tail)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "s14_metrla.png"), dpi=150)
    print("saved s14_metrla.png", flush=True)


# ------------------------------------------------------------------- smoke --
def _smoke(model, windows, A):
    ids_m = group_ids(windows, "miss")[:5]
    ids_c = group_ids(windows, "ctrl")[:5]
    print(f"[smoke] {model.name}: {len(ids_m)} miss + {len(ids_c)} ctrl windows")
    for group, fill in [("ctrl", "clean"), ("ctrl", "zero"), ("ctrl", "nan"),
                        ("miss", "keep"), ("miss", "zero"), ("miss", "linear"),
                        ("miss", "nan")]:
        ids = ids_m if group == "miss" else ids_c
        r = run_point_config(model, windows, A, group, fill, ids)
        print(f"  {group:4s} {fill:7s} nmse={r['nmse']:8.3f} mse={r['mse']:9.2f} "
              f"mse/pers={r['mse_over_persistence']:.2f}", flush=True)
    # literal single-window sanity: zero-fill of a clean window == clean input
    i = ids_c[0]
    Xa = fill_window("linear", A["ctx_obs"][i], A["ctx_rec"][i])
    Xb = fill_window("zero", A["ctx_obs"][i], A["ctx_rec"][i])
    assert np.array_equal(Xa, A["ctx_rec"][i]) and np.array_equal(Xb, A["ctx_rec"][i]), \
        "fill touched a clean window"
    pa = model.predict_point(Xa[None].astype(np.float32))
    pb = model.predict_point(A["ctx_rec"][i][None].astype(np.float32))
    print(f"  clean-window zero-fill max|dpred|={np.abs(pa - pb).max():.2e}")
    if hasattr(model, "predict_quantiles"):
        q10, q90 = model.predict_quantiles(build_X("linear", ids_m, A))
        y = A["tgt"][ids_m].astype(np.float64)
        cov = ((y >= q10) & (y <= q90)).mean()
        print(f"  quantiles ok: shapes {q10.shape}, native cov(linear)={cov:.3f}")
        assert (q90 > q10).mean() > 0.99, "quantile order broken"
    for i in ids_m[:3]:
        mech, f = detect_mechanism(A["ctx_rec"][i], A["mask"][i])
        print(f"  detector window {i}: {mech} r_bar={f['r_bar']:.2f} "
              f"e_bar={f['e_bar']:.2f} run={f['mean_run']:.1f}")
    print("[smoke OK]")


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="anchor",
                    help="comma list: explore,smoke,anchor,conformal,detector,gate,figure")
    ap.add_argument("--model", default="bolt,timesfm,moirai")
    ap.add_argument("--device", default="cuda:4")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    modes = args.mode.split(",")
    t0 = time.time()

    if "explore" in modes:
        ex = run_explore()
        print(json.dumps(ex, indent=1), flush=True)
        save_records({"explore": ex}, args.out)
        return

    windows = build_windows()
    A = extract(windows)

    if "detector" in modes:
        det = run_detector(windows, A)
        print(json.dumps(det, indent=1), flush=True)
        save_records({"detector|miss": det}, args.out)
        if set(modes) == {"detector"}:
            return

    model_names = args.model.split(",")
    point_modes = [m for m in modes if m in ("anchor", "smoke")]
    if point_modes or "conformal" in modes:
        for mname in model_names:
            model = MODEL_REGISTRY[mname](args.device)
            print(f"[model] {mname} on {args.device} ({time.time() - t0:.0f}s)",
                  flush=True)
            if "smoke" in modes:
                _smoke(model, windows, A)
                return
            if "anchor" in modes:
                for group, fill in [("ctrl", "clean"), ("ctrl", "zero"),
                                    ("ctrl", "linear"), ("ctrl", "nan"),
                                    ("miss", "keep"), ("miss", "zero"),
                                    ("miss", "linear"), ("miss", "nan")]:
                    ids = group_ids(windows, group)
                    r = run_point_config(model, windows, A, group, fill, ids)
                    key = f"anchor|{mname}|{group}|{fill}"
                    print(f"  {key:30s} nmse={r['nmse']:8.4f} med={r['nmse_median']:7.3f} "
                          f"mse/pers={r['mse_over_persistence']:5.2f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                    save_records({key: r}, args.out)
            if "conformal" in modes:
                if hasattr(model, "predict_quantiles"):
                    for k2, v2 in run_conformal(model, windows, A).items():
                        save_records({k2: v2}, args.out)
                else:
                    print(f"[skip] {mname}: no quantile head for conformal", flush=True)
            del model
            torch.cuda.empty_cache()

    if "gate" in modes:
        save_records(run_gate(load_records(args.out)), args.out)
    if "figure" in modes:
        run_figure(load_records(args.out))
    print(f"[done] modes={modes} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
