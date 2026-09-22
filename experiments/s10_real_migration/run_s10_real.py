#!/usr/bin/env python
"""S10: migrate S6's artificial-MNAR fixes to REAL curtailment censoring.

S6 (run_s6_mnarfix.py / run_s6_conformal.py) validated two fixes on synthetic
value-censored contexts (top-p% values removed): (1) Tobit truncation-aware
imputation, (2) mechanism-aware split conformal. Track D
(run_s6_real_censor.py) characterized REAL censoring (Penmanshiel 2016 wind
curtailment, operator Status-log codes 9000/9210) but applied no fix. S10
closes the loop: same real windows, same metrics, now with the fixes.

KEY ASSUMPTIONS (measured in mode=explore, stored under results["explore"]):
  * Real geometry is BIMODAL top-censoring with an UNKNOWN, time-varying
    threshold: ~55% of effectively-censored points sit at a plateau of
    50-85% of available power (code 9000 clamps at caps ~1650/1000/750/500
    kW), ~31% are forced stops at ~0/negative (code 9210). Unlike synthetic
    mnar_high the censored value is NOT removed -- it is observed AS the cap:
    y_obs = min(y_true, tau_event). So the Tobit threshold tau must be
    ESTIMATED per event. Two estimators are tested:
      - "tobit": tau_e = median of the recorded (clamped) values within the
        censored run (the plateau IS the cap); fill = E[x | x > tau_e] under a
        robust Gaussian (median/MAD) fit on the window's UNCENSORED values;
        fill clipped to [0, RATED]. Threshold sensitivity: tau_e x {0.9, 1.1}.
      - "pc": power-curve mapping; fill = clip(max(avail(wind_t), y_obs), 0,
        RATED) per censored point. avail is the per-turbine-year q95 power
        curve (side information that does not use the operator log; the max()
        enforces the censoring constraint y_true >= cap).
  * For forced-stop events (tau_e ~ 0) plateau-Tobit degenerates to ~the
    context median (lambda(alpha->-inf) -> 0); the pc variant still sees the
    high wind. The contrast is part of the science question.
  * Windows, masks, metrics (NMSE with var floor 100 kW^2, top-decile split at
    the observed-context q90, persistence = last as-recorded value, ratio of
    mean MSEs) are EXACTLY Track D's: run_s6_real_censor is imported and the
    window list is verified identical to s6_real_censor_results.json.
  * Conformal: base interval = native [q10, q90] (80% nominal; bolt quantiles
    0.1..0.9, timesfm continuous quantile head), split-conformal target 90%,
    per-horizon-step widening. cal/test = per-turbine 50/50 split of the 602
    cens (resp. 602 ctrl) windows. naive = calibrate on CLEAN windows (the
    mechanism-blind standard); mechanism-aware = calibrate on CENSORED windows
    (status-code mask = oracle censor indicator); pooled = mechanism-blind
    mix of both. Deploy fills: zero (Track D's winner) and linear (S6 parity).
    moirai is point-only here (20 samples too coarse for 90% tails).
  * Detector add-on: S6's bridge-rank rules (ported with CTX=144) on (a) the
    oracle status-code mask and (b) a status-free candidate mask
    (avail - power > 0.15*rated & avail > 0.30*rated) -- the "no operator
    annotation" deployment scenario.

Modes (comma-separated): explore, anchor, gate, track1, track2, detector,
  finalize, figure. Parallelism: one process per model per GPU
  (--model/--device), or window-sharded via --shard k/m (strided window
  subset; record keys get a shard suffix and mode=finalize merges them).
Results flock-merged into s10_real_results.json after every config.
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
import torch
from scipy.stats import norm

import run_s6_real_censor as s6r  # Track D pipeline: windows/metrics identical

CTX, H = s6r.CTX, s6r.H          # 144, 24
RATED = 2050.0                   # kW (Senvion MM82)
VAR_FLOOR = s6r.VAR_FLOOR        # 100 kW^2
NPZ = s6r.NPZ
OUT = os.path.join(HERE, "s10_real_results.json")
S6_RESULTS = os.path.join(HERE, "s6_real_censor_results.json")
SEED_SPLIT = 20250812            # cal/test split seed (track2)
ALPHA = 0.9                      # conformal target coverage
MIN_DET_POINTS = 3               # min candidate-mask points to run detector


# ------------------------------------------------------------------ models --
class BoltQ(s6r.BoltModel):
    """chronos-bolt-base + native quantile access (q10/q90 of 0.1..0.9 grid)."""

    @torch.no_grad()
    def predict_quantiles(self, ctx):  # ctx: [n, CTX] -> (q10, q90) each [n, H]
        q10, q90 = [], []
        for i in range(0, len(ctx), self.batch):
            xb = torch.from_numpy(ctx[i:i + self.batch]).to(self.device)
            q = self.pipe.predict(xb, prediction_length=H)  # [b, 9, H]
            q10.append(q[:, 0, :].float().cpu().numpy())
            q90.append(q[:, 8, :].float().cpu().numpy())
        return np.concatenate(q10), np.concatenate(q90)


class TimesFmQ(s6r.TimesFmModel):
    """timesfm-2.5-200m + continuous quantile head (mean + 0.1..0.9)."""

    def predict_quantiles(self, ctx):
        q10, q90 = [], []
        for i in range(0, len(ctx), self.batch):
            chunk = [row for row in ctx[i:i + self.batch]]
            _, qf = self.m.forecast(horizon=H, inputs=chunk)
            qf = np.asarray(qf)  # [b, H, 10]: col0=mean, cols1..9 = q0.1..q0.9
            assert qf.ndim == 3 and qf.shape[2] == 10, f"timesfm qf shape {qf.shape}"
            q10.append(qf[:, :, 1].astype(np.float32))
            q90.append(qf[:, :, 9].astype(np.float32))
        return np.concatenate(q10), np.concatenate(q90)


class MoiraiS10(s6r.MoiraiModel):
    """MoiraiModel with the predictor pinned to the requested device."""

    def __init__(self, device):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        forecast = MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base"),
            prediction_length=H, context_length=CTX, patch_size=16,
            num_samples=self.num_samples, target_dim=1,
            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        self.predictor = forecast.create_predictor(batch_size=self.batch,
                                                   device=device)


MODEL_REGISTRY = {"bolt": BoltQ, "timesfm": TimesFmQ, "moirai": MoiraiS10}


# ------------------------------------------------------------------- data ---
def load_all():
    """Track D windows (verified identical) + per-window arrays."""
    windows, series = s6r.build_windows()
    with open(S6_RESULTS) as f:
        old = json.load(f)
    assert old["meta"]["windows"] == windows, "window list drifted from Track D"
    d = np.load(NPZ)
    W = len(windows)
    ctx_obs = np.full((W, CTX), np.nan, np.float32)   # censored points -> NaN
    ctx_rec = np.full((W, CTX), np.nan, np.float32)   # as-recorded (clamped)
    avail = np.full((W, CTX), np.nan, np.float32)     # q95 power-curve estimate
    mask = np.zeros((W, CTX), bool)                   # effective-censor mask
    tgt = np.full((W, H), np.nan, np.float32)
    for i, w in enumerate(windows):
        t = w["start"]
        tag = w["turbine"]
        p = d[f"{tag}_power"]
        ctx_rec[i] = p[t - CTX:t]
        avail[i] = d[f"{tag}_avail"][t - CTX:t]
        tgt[i] = p[t:t + H]
        if w["kind"] == "cens":
            m = d[f"{tag}_cens"][t - CTX:t]
            mask[i] = m
            c = ctx_rec[i].copy()
            c[m] = np.nan
            ctx_obs[i] = c
        else:
            ctx_obs[i] = ctx_rec[i]
    return windows, dict(ctx_obs=ctx_obs, ctx_rec=ctx_rec, avail=avail,
                         mask=mask, tgt=tgt)


def group_ids(windows, kind):
    return np.array([i for i, w in enumerate(windows) if w["kind"] == kind])


# ------------------------------------------------------------------- fills --
def cens_runs(mask1d):
    """[L] bool -> list of (start, end) inclusive runs."""
    dm = np.diff(mask1d.astype(np.int8))
    starts = list(np.flatnonzero(dm == 1) + 1) + ([0] if mask1d[0] else [])
    ends = list(np.flatnonzero(dm == -1)) + ([len(mask1d) - 1] if mask1d[-1] else [])
    return list(zip(sorted(starts), sorted(ends)))


def fill_window(fill, x_obs, x_rec, avail1d, mask1d):
    """One context. x_obs has NaN at censored points; x_rec is as-recorded."""
    t = np.arange(CTX)
    valid = np.flatnonzero(np.isfinite(x_obs))
    if fill == "keep":
        return x_rec.astype(np.float64)
    if fill == "zero":
        return np.nan_to_num(x_obs.astype(np.float64))
    if len(valid) == 0:
        return np.zeros(CTX)
    base = np.interp(t, valid, x_obs[valid])  # linear (Track D convention)
    if fill == "linear" or mask1d.sum() == 0:
        return base
    # ---- censoring-aware fills -------------------------------------------
    tau_scale = {"tobit": 1.0, "tobit_up": 1.1, "tobit_dn": 0.9}.get(fill, 1.0)
    obs = x_obs[valid].astype(np.float64)
    mu = float(np.median(obs))
    sigma = max(1.4826 * float(np.median(np.abs(obs - mu))), 1e-6)
    out = base.copy()
    for i, j in cens_runs(mask1d):
        if fill == "pc":
            # power-curve would-be estimate, floored at the observed cap
            out[i:j + 1] = np.clip(np.maximum(avail1d[i:j + 1], x_rec[i:j + 1]),
                                   0.0, RATED)
        else:  # plateau-threshold Tobit: tau_e = in-run median of recorded
            tau = float(np.median(x_rec[i:j + 1])) * tau_scale
            a = min(max((tau - mu) / sigma, -8.0), 8.0)
            lam = norm.pdf(a) / max(1.0 - norm.cdf(a), 1e-12)
            out[i:j + 1] = min(max(mu + sigma * lam, 0.0), RATED)
    return out


def build_X(fill, ids, A):
    return np.stack([fill_window(fill, A["ctx_obs"][i], A["ctx_rec"][i],
                                 A["avail"][i], A["mask"][i])
                     for i in ids]).astype(np.float32)


# ------------------------------------------------- point-forecast metrics ---
def point_metrics(pred, Y, q90_ref, ctx_rec_ids):
    """Mirror s6r.run_config metrics + persistence. All per-window + pooled."""
    err2 = (pred - Y) ** 2
    mse_w = err2.mean(axis=1)
    var = np.maximum(Y.var(axis=1), VAR_FLOOR)
    nmse_w = mse_w / var
    q90 = np.array([np.nanquantile(q90_ref[k], 0.9) for k in range(len(Y))])
    top = Y > q90[:, None]
    with np.errstate(all="ignore"):
        td = np.nanmean(np.where(top, err2, np.nan), axis=1)
        rs = np.nanmean(np.where(~top, err2, np.nan), axis=1)
    pers = np.repeat(ctx_rec_ids[:, -1:], H, axis=1).astype(np.float64)
    mse_pers = ((pers - Y) ** 2).mean(axis=1)
    return dict(
        mse=float(mse_w.mean()), mae=float(np.abs(pred - Y).mean()),
        nmse=float(nmse_w.mean()), nmse_median=float(np.median(nmse_w)),
        mse_topdecile=float(np.nanmean(td)), mse_rest=float(np.nanmean(rs)),
        topdecile_frac=float(top.mean()),
        persistence_mse=float(mse_pers.mean()),
        mse_over_persistence=float(mse_w.mean() / max(mse_pers.mean(), 1e-9)),
        nmse_per_window=nmse_w.tolist(), mse_per_window=mse_w.tolist(),
        mse_topdecile_per_window=[None if np.isnan(v) else v for v in td.tolist()],
        mse_rest_per_window=[None if np.isnan(v) else v for v in rs.tolist()],
    )


def run_point_config(model, windows, A, group, fill, ids):
    """ids: window indices (already group-filtered and sharded)."""
    X = build_X(fill, ids, A)
    Y = A["tgt"][ids].astype(np.float64)
    pred = model.predict_point(X).astype(np.float64)
    # Track D q90 reference: observed (censored->NaN) context for cens windows,
    # clean context for ctrl windows
    q90_ref = A["ctx_obs"][ids] if group == "cens" else A["ctx_rec"][ids]
    rec = point_metrics(pred, Y, q90_ref, A["ctx_rec"][ids])
    rec.update(model=model.name, group=group, fill=fill, n_windows=len(ids),
               window_ids=[int(i) for i in ids],
               mean_censor_rate=float(A["mask"][ids].mean(axis=1).mean()))
    return rec


# --------------------------------------------------------- conformal (T2) ---
def cal_test_split(windows, kind, seed=SEED_SPLIT):
    """Per-turbine 50/50 split of the group's window ids (deterministic)."""
    by_t = {}
    for i, w in enumerate(windows):
        if w["kind"] == kind:
            by_t.setdefault(w["turbine"], []).append(i)
    rng = np.random.default_rng(seed)
    cal, test = [], []
    for tag in sorted(by_t):
        lst = by_t[tag]
        perm = rng.permutation(len(lst))
        half = len(lst) // 2
        cal += [lst[k] for k in perm[:half]]
        test += [lst[k] for k in perm[half:]]
    return np.array(sorted(cal)), np.array(sorted(test))


def conformal_w(E, alpha=ALPHA):
    """E: [n, H] nonconformity -> per-step widening [H]."""
    n = E.shape[0]
    k = min(int(np.ceil((n + 1) * alpha)), n)
    return np.sort(E, axis=0)[k - 1]


def cov_width(q10, q90, y, w=None):
    if w is None:
        w = np.zeros(q10.shape[-1])
    inside = (y >= q10 - w) & (y <= q90 + w)
    cov_w = inside.mean(axis=1)          # per window
    return dict(coverage=float(inside.mean()), width=float((q90 - q10 + 2 * w).mean()),
                coverage_per_step=inside.mean(axis=0).tolist(),
                width_per_step=(q90 - q10 + 2 * w).mean(axis=0).tolist(),
                coverage_per_window=cov_w.tolist())


def run_track2(model, windows, A, shard):
    """Native / naive / pooled / mechanism-aware conformal on real censoring."""
    cal_c, test_c = cal_test_split(windows, "cens")
    cal_t, test_t = cal_test_split(windows, "ctrl")
    k, m = shard
    if m > 1:  # shard the deploy sets only; calibration always full
        test_c, test_t = test_c[k::m], test_t[k::m]
    recs = {}
    for fill in ("zero", "linear"):
        splits = {}
        for name, ids, grp in [("cens_cal", cal_c, "cens"),
                               ("cens_test", test_c, "cens"),
                               ("ctrl_cal", cal_t, "ctrl"),
                               ("ctrl_test", test_t, "ctrl")]:
            X = build_X(fill, ids, A)
            q10, q90 = model.predict_quantiles(X)
            q10, q90 = q10.astype(np.float64), q90.astype(np.float64)
            y = A["tgt"][ids].astype(np.float64)
            splits[name] = (q10, q90, y, ids)
        E = {name: np.maximum(q10 - y, y - q90)
             for name, (q10, q90, y, _) in splits.items()}
        W_cal = {
            "naive": conformal_w(E["ctrl_cal"]),                    # clean-calib
            "mechaware": conformal_w(E["cens_cal"]),                # cens-calib
            "pooled": conformal_w(np.concatenate([E["ctrl_cal"], E["cens_cal"]])),
        }
        for deploy in ("cens_test", "ctrl_test"):
            q10, q90, y, ids = splits[deploy]
            variants = [("native", None)] + list(W_cal.items())
            if deploy == "ctrl_test":
                variants = [("native", None), ("naive", W_cal["naive"]),
                            ("mechaware", W_cal["mechaware"])]
            for vname, w in variants:
                r = cov_width(q10, q90, y, w)
                r.update(model=model.name, fill=fill, deploy=deploy, variant=vname,
                         n_cal={"naive": len(cal_t), "mechaware": len(cal_c),
                                "pooled": len(cal_t) + len(cal_c)}.get(vname, 0),
                         n_test=len(ids), window_ids=[int(i) for i in ids],
                         target=ALPHA)
                tag = f"track2|{model.name}|{fill}|{deploy}|{vname}"
                recs[tag + shard_suffix(shard)] = r
                print(f"    {tag:44s} cov={r['coverage']:.3f} wid={r['width']:8.1f}",
                      flush=True)
    return recs


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
        return "undetermined", {}  # mask swallows (nearly) the whole context
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


def candidate_mask(power1d, avail1d):
    """Status-free curtailment candidate: binding clamp, no operator log."""
    return (np.isfinite(power1d) & np.isfinite(avail1d)
            & ((avail1d - power1d) > 0.15 * RATED) & (avail1d > 0.30 * RATED))


def run_detector(windows, A):
    out = {}
    for mode in ("oracle", "auto"):
        stats = {}
        for grp in ("cens", "ctrl"):
            ids = group_ids(windows, grp)
            labels, feats, n_skip = [], [], 0
            overlap = dict(tp=0, cand=0, cens=0)
            for i in ids:
                m = (A["mask"][i] if mode == "oracle"
                     else candidate_mask(A["ctx_rec"][i], A["avail"][i]))
                if mode == "auto" and grp == "cens":
                    om = A["mask"][i]
                    overlap["tp"] += int((m & om).sum())
                    overlap["cand"] += int(m.sum())
                    overlap["cens"] += int(om.sum())
                if m.sum() < MIN_DET_POINTS:
                    n_skip += 1
                    continue
                mech, f = detect_mechanism(A["ctx_rec"][i], m)
                labels.append(mech)
                if f:
                    feats.append(f)
            cnt = {k: labels.count(k)
                   for k in ("upper", "two-tail", "clustered", "random", "undetermined")}
            n = max(len(labels), 1)
            stats[grp] = dict(
                n_windows=len(ids), n_skipped_too_few=n_skip,
                label_counts=cnt,
                label_frac={k: v / n for k, v in cnt.items()},
                r_bar=float(np.mean([f["r_bar"] for f in feats])) if feats else None,
                e_bar=float(np.mean([f["e_bar"] for f in feats])) if feats else None,
                mean_run=float(np.mean([f["mean_run"] for f in feats])) if feats else None,
                cand_points_per_window=float(np.mean([
                    candidate_mask(A["ctx_rec"][i], A["avail"][i]).sum()
                    for i in ids])) if mode == "auto" else None,
            )
            if mode == "auto" and grp == "cens":
                stats[grp]["cand_recall_vs_oracle"] = overlap["tp"] / max(overlap["cens"], 1)
                stats[grp]["cand_precision_vs_oracle"] = overlap["tp"] / max(overlap["cand"], 1)
        out[mode] = stats
    return out


# ----------------------------------------------------------------- explore --
def run_explore():
    d = np.load(NPZ)
    tags = sorted({k[:-len("_power")] for k in d.files if k.endswith("_power")})
    ratio, pow_c, av_c, rl = [], [], [], []
    per_turb = {}
    for tag in tags:
        p, cens = d[f"{tag}_power"], d[f"{tag}_cens"]
        av = d[f"{tag}_avail"]
        m = cens & np.isfinite(p)
        r = p[m] / np.maximum(av[m], 1e-6)
        ratio.append(r); pow_c.append(p[m]); av_c.append(av[m])
        rl += [j - i + 1 for i, j in cens_runs(m)]
        per_turb[tag] = dict(n_cens=int(m.sum()),
                             frac_ratio_lt_0p1=float((r < 0.1).mean()),
                             frac_pow_lt_50kW=float((p[m] < 50).mean()),
                             med_ratio=float(np.median(r)))
    r = np.concatenate(ratio); pw = np.concatenate(pow_c)
    av = np.concatenate(av_c); rl = np.array(rl)
    edges = [-np.inf, 0.02, 0.1, 0.25, 0.5, 0.7, 0.85, 1.01]
    hist = [float(((r >= lo) & (r < hi)).mean()) for lo, hi in zip(edges[:-1], edges[1:])]
    pcap_edges = np.arange(0, 2200, 100)
    pcap_hist = np.histogram(np.clip(pw, 0, None), bins=pcap_edges)[0] / len(pw)
    return dict(
        n_censored_points=int(len(r)), per_turbine=per_turb,
        ratio_bins=[f"{lo}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])],
        ratio_hist=hist,
        frac_ratio_lt_0p1=float((r < 0.1).mean()),
        frac_pow_lt_50kW=float((pw < 50).mean()),
        plateau_bin_edges_kW=pcap_edges.tolist(),
        plateau_hist=pcap_hist.tolist(),
        avail_at_cens=dict(median=float(np.median(av)),
                           p10=float(np.percentile(av, 10)),
                           p90=float(np.percentile(av, 90))),
        runlen=dict(median=float(np.median(rl)), mean=float(rl.mean()),
                    p90=float(np.percentile(rl, 90)), max=int(rl.max()),
                    frac_ge6=float((rl >= 6).mean())),
    )


# -------------------------------------------------------------------- io ----
def shard_suffix(shard):
    k, m = shard
    return f"|shard{k}of{m}" if m > 1 else ""


def save_records(recs, path=OUT):
    """flock-locked read-merge-write; safe for concurrent model processes."""
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        txt = fh.read()
        try:
            data = json.loads(txt) if txt.strip() else {}
        except json.JSONDecodeError:
            data = {}
        data.setdefault("meta", dict(
            design="S10: S6 fixes (Tobit imputation, mechanism-aware conformal) "
                   "migrated to real Penmanshiel-2016 curtailment censoring",
            ctx=CTX, horizon=H, alpha=ALPHA, seed_split=SEED_SPLIT,
            fills=dict(keep="as-recorded clamped values",
                       zero="censored->0", linear="censored->np.interp",
                       tobit="per-run plateau threshold tau_e (in-run median of "
                             "recorded); fill E[x|x>tau_e], median/MAD Gaussian "
                             "on uncensored context, clip [0, 2050]",
                       tobit_up="tobit with tau_e x 1.1 (sensitivity)",
                       tobit_dn="tobit with tau_e x 0.9 (sensitivity)",
                       pc="power-curve fill clip(max(avail(wind), recorded), 0, 2050)"),
            track2=dict(base_interval="native [q10,q90]", target=ALPHA,
                        split="per-turbine 50/50 cal/test within cens/ctrl",
                        naive="calibrate on clean ctrl windows",
                        mechaware="calibrate on real censored windows",
                        pooled="mechanism-blind ctrl+cens calibration"),
            anchor_of="s6_real_censor_results.json"))
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


# -------------------------------------------------------------- gate/final --
def run_gate():
    """Anchor: S10 anchor records must reproduce Track D within +/-5%.
    bolt/timesfm are deterministic -> gate on mean NMSE. moirai forecasts by
    sampling (20 samples, median) and neither Track D nor S10 seeds it, so its
    mean NMSE is dominated by tail-draw noise; gate moirai on MEDIAN NMSE (its
    means are still reported in the notes)."""
    s6 = json.load(open(S6_RESULTS))["records"]
    s10 = load_records()
    rows, all_pass = [], True
    for model in ("bolt", "timesfm", "moirai"):
        fld = "nmse_median" if model == "moirai" else "nmse"
        for group, fill in [("ctrl", "clean"), ("cens", "keep"),
                            ("cens", "zero"), ("cens", "linear")]:
            key = f"anchor|{model}|{group}|{fill}"
            if key not in s10:
                print(f"  MISSING {key}")
                all_pass = False
                continue
            ref = next(r for r in s6 if r["model"] == model
                       and r["group"] == ("ctrl_clean" if group == "ctrl" else group)
                       and (r["fill"] == fill or (fill == "clean" and r["fill"] == "clean")))
            got = s10[key][fld]
            want = (float(np.median(ref["nmse_per_window"])) if fld == "nmse_median"
                    else ref["nmse"])
            rel = abs(got - want) / want
            ok = rel <= 0.05
            all_pass &= ok
            rows.append(dict(model=model, group=group, fill=fill, metric=fld,
                             s6=want, s10=got, rel_diff=rel, passed=bool(ok),
                             s10_mean_nmse=s10[key]["nmse"], s6_mean_nmse=ref["nmse"]))
            print(f"  {model:8s} {group:4s} {fill:6s} {fld:12s} s6={want:9.4f} "
                  f"s10={got:9.4f} diff={rel * 100:5.2f}% {'OK' if ok else 'FAIL'}",
                  flush=True)
    print(f"GATE {'PASS' if all_pass else 'FAIL'}", flush=True)
    return {"gate": dict(passed=bool(all_pass), tol=0.05,
                         metric="mean NMSE (bolt/timesfm), median NMSE (moirai, "
                                "sampling-based -> mean is tail-draw noise)", rows=rows)}


def run_finalize():
    """Merge sharded records (per-window arrays concatenated, aggregates
    recomputed) into bare record keys."""
    recs = load_records()
    groups = {}
    for key, r in recs.items():
        if "|shard" not in key:
            continue
        base = key.split("|shard")[0]
        groups.setdefault(base, []).append(r)
    out = {}
    for base, rs in groups.items():
        order = np.argsort(np.concatenate([r["window_ids"] for r in rs]))
        r0 = rs[0]
        merged = {k: v for k, v in r0.items()
                  if not k.endswith("_per_window") and k != "window_ids"}
        merged["window_ids"] = np.concatenate([r["window_ids"] for r in rs])[order].tolist()
        merged["n_windows"] = len(merged["window_ids"])
        for fld in ("nmse_per_window", "mse_per_window", "coverage_per_window"):
            if fld in r0:
                merged[fld] = np.concatenate([r[fld] for r in rs])[order].tolist()
        if "nmse_per_window" in merged:
            nm = np.array(merged["nmse_per_window"])
            ms = np.array(merged["mse_per_window"])
            merged.update(nmse=float(nm.mean()), nmse_median=float(np.median(nm)),
                          mse=float(ms.mean()))
        if "coverage_per_window" in merged:
            merged["coverage"] = float(np.mean(merged["coverage_per_window"]))
        out[base] = merged
        print(f"  merged {len(rs)} shards -> {base}", flush=True)
    return out


# ------------------------------------------------------------------ figure --
def run_figure(recs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    models = ("bolt", "timesfm", "moirai")
    colors = {"keep": "#999999", "zero": "#1f77b4", "linear": "#d62728",
              "tobit": "#2ca02c", "pc": "#9467bd"}

    # -- A: real censoring geometry -----------------------------------------
    ax = axes[0][0]
    ex = recs["explore"]
    bins = ex["ratio_bins"]
    ax.bar(range(len(bins)), ex["ratio_hist"], color="#88419d")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(["<0.02", "0.02-0.1", "0.1-0.25", "0.25-0.5",
                        "0.5-0.7", "0.7-0.85", "0.85-1"], fontsize=8, rotation=30)
    ax.set_xlabel("observed power / available power at censored points")
    ax.set_ylabel("fraction")
    ax.set_title("A. Real curtailment geometry: clamped to plateau OR ~0\n"
                 f"(n={ex['n_censored_points']} pts; {ex['frac_ratio_lt_0p1']:.0%} at "
                 f"ratio<0.1 = forced stop, rest at plateau caps)", fontsize=10)

    # -- B: track1 fills -----------------------------------------------------
    ax = axes[0][1]
    fills = ["keep", "zero", "linear", "tobit", "pc"]
    wdt = 0.25
    for mi, model in enumerate(models):
        clean = next(r for k, r in recs.items()
                     if k == f"anchor|{model}|ctrl|clean")
        vals = []
        for f in fills:
            key = (f"anchor|{model}|cens|{f}" if f in ("keep", "zero", "linear")
                   else f"track1|{model}|cens|{f}")
            vals.append(recs[key]["nmse"])
        x = np.arange(len(fills)) + (mi - 1) * wdt
        ax.bar(x, vals, width=wdt, label=model,
               color=["#1f77b4", "#ff7f0e", "#2ca02c"][mi], alpha=0.85)
        for xi, v in zip(x, vals):
            ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        if mi == 0:
            ax.axhline(clean["nmse"], color="k", ls="--", lw=1)
            ax.text(len(fills) - 0.5, clean["nmse"], f"clean ctrl ({clean['nmse']:.1f})",
                    fontsize=8, ha="right", va="bottom")
    ax.set_xticks(np.arange(len(fills)))
    ax.set_xticklabels(fills)
    ax.set_ylabel("mean NMSE on censored windows")
    ax.set_title("B. Track 1: censoring-aware fills vs Track D fills\n"
                 "(602 real curtailed windows; lower = better)", fontsize=10)
    ax.legend(fontsize=8)

    # -- C: track2 conformal coverage ---------------------------------------
    ax = axes[1][0]
    qmodels = ("bolt", "timesfm")
    variants = ["native", "naive", "pooled", "mechaware"]
    wdt = 0.18
    for mi, model in enumerate(qmodels):
        for fi, fill in enumerate(("zero", "linear")):
            vals = []
            for v in variants:
                key = f"track2|{model}|{fill}|cens_test|{v}"
                vals.append(recs[key]["coverage"] if key in recs else np.nan)
            x = np.arange(len(variants)) + (mi * 2 + fi - 1.5) * wdt
            ax.bar(x, vals, width=wdt,
                   color=["#1f77b4", "#9ecae1", "#d62728", "#f4b6b6"][mi * 2 + fi],
                   label=f"{model}/{fill}")
    ax.axhline(ALPHA, color="k", ls="--", lw=1)
    ax.text(0, ALPHA + 0.01, f"target {ALPHA}", fontsize=8)
    ax.set_xticks(np.arange(len(variants)))
    ax.set_xticklabels(["native", "naive\n(clean cal)", "pooled", "mech-aware\n(cens cal)"])
    ax.set_ylabel("coverage on censored test windows")
    ax.set_ylim(0, 1.02)
    ax.set_title("C. Track 2: split-conformal on real censoring\n"
                 "(native [q10,q90] widened to 90%; 301 cal / 301 test)", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    # -- D: detector ----------------------------------------------------------
    ax = axes[1][1]
    det_o = recs["detector|oracle"]["cens"]["label_frac"]
    det_a = recs["detector|auto"]["cens"]["label_frac"]
    mechs = ["upper", "two-tail", "clustered", "random"]
    bottoms = np.zeros(2)
    for mech, c in zip(mechs, ["#2ca02c", "#9467bd", "#8c564b", "#c7c7c7"]):
        vals = [det_o.get(mech, 0), det_a.get(mech, 0)]
        ax.bar([0, 1], vals, bottom=bottoms, color=c, label=mech, width=0.5)
        bottoms += vals
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["oracle mask\n(status codes)", "auto mask\n(avail-power margin)"])
    ax.set_ylabel("fraction of 602 censored windows")
    ax.set_title("D. S6 mechanism detector on real curtailment\n"
                 "(upper = censoring detected -> Tobit gate would open)", fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "s10_real.png"), dpi=150)
    print("saved s10_real.png", flush=True)


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="anchor",
                    help="comma list: explore,anchor,gate,track1,track2,detector,"
                         "finalize,figure")
    ap.add_argument("--model", default="bolt,timesfm,moirai")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shard", default="0/1", help="k/m: strided window shard")
    ap.add_argument("--smoke", action="store_true",
                    help="5 cens + 5 ctrl windows, bolt, print only, no writes")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    shard = tuple(int(v) for v in args.shard.split("/"))
    assert len(shard) == 2 and 0 <= shard[0] < shard[1]
    modes = args.mode.split(",")
    t0 = time.time()

    if "explore" in modes:
        ex = run_explore()
        print(json.dumps(ex["per_turbine"], indent=1), flush=True)
        print(f"[explore] ratio hist {ex['ratio_hist']}", flush=True)
        if not args.smoke:
            save_records({"explore": ex}, args.out)

    need_windows = (set(modes) & {"anchor", "track1", "track2", "detector", "smoke"}
                    or args.smoke)
    windows = A = None
    if need_windows:
        windows, A = load_all()
        print(f"[data] {len(windows)} windows ({time.time() - t0:.0f}s)", flush=True)

    if "detector" in modes:
        det = run_detector(windows, A)
        print(json.dumps(det, indent=1), flush=True)
        if not args.smoke:
            save_records({f"detector|{k}": v for k, v in det.items()}, args.out)

    model_names = args.model.split(",")
    point_modes = [mm for mm in modes if mm in ("anchor", "track1")]
    if point_modes or "track2" in modes or args.smoke:
        ids_c, ids_t = group_ids(windows, "cens"), group_ids(windows, "ctrl")
        if args.smoke:
            ids_c, ids_t = ids_c[:5], ids_t[:5]
        if shard[1] > 1 and not args.smoke:
            ids_c, ids_t = ids_c[shard[0]::shard[1]], ids_t[shard[0]::shard[1]]
        for mname in model_names:
            if mname == "moirai" and ("track2" in modes) and not point_modes:
                print("[skip] moirai has no native quantiles for track2", flush=True)
                continue
            model = MODEL_REGISTRY[mname](args.device)
            print(f"[model] {mname} on {args.device} ({time.time() - t0:.0f}s)",
                  flush=True)
            if args.smoke:
                _smoke(model, windows, A, ids_c, ids_t)
                return
            recs = {}
            for mode in point_modes:
                cfgs = ([("ctrl", "clean"), ("cens", "keep"),
                         ("cens", "zero"), ("cens", "linear")] if mode == "anchor"
                        else [("cens", "tobit"), ("cens", "tobit_up"),
                              ("cens", "tobit_dn"), ("cens", "pc")])
                for group, fill in cfgs:
                    ids = ids_c if group == "cens" else ids_t
                    r = run_point_config(model, windows, A, group, fill, ids)
                    key = f"{mode}|{mname}|{group}|{fill}"
                    recs[key + shard_suffix(shard)] = r
                    print(f"  {key:34s} nmse={r['nmse']:9.4f} med={r['nmse_median']:7.3f} "
                          f"mse/pers={r['mse_over_persistence']:5.2f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                    save_records({k: v for k, v in recs.items()}, args.out)
                    recs = {}
            if "track2" in modes:
                if hasattr(model, "predict_quantiles"):
                    for k2, v2 in run_track2(model, windows, A, shard).items():
                        save_records({k2: v2}, args.out)
                else:
                    print(f"[skip] {mname}: no quantile head for track2", flush=True)
            del model
            torch.cuda.empty_cache()

    if "gate" in modes:
        save_records(run_gate(), args.out)
    if "finalize" in modes:
        save_records(run_finalize(), args.out)
    if "figure" in modes:
        run_figure(load_records(args.out))
    print(f"[done] modes={modes} ({time.time() - t0:.0f}s)", flush=True)


def _smoke(model, windows, A, ids_c, ids_t):
    """5-window sanity: fills, metrics, quantiles, detector. Prints only."""
    print(f"[smoke] {model.name}: {len(ids_c)} cens + {len(ids_t)} ctrl windows")
    for group, fill in [("ctrl", "clean"), ("cens", "keep"), ("cens", "zero"),
                        ("cens", "linear"), ("cens", "tobit"), ("cens", "tobit_up"),
                        ("cens", "tobit_dn"), ("cens", "pc")]:
        ids = ids_c if group == "cens" else ids_t
        r = run_point_config(model, windows, A, group, fill, ids)
        print(f"  {group:4s} {fill:9s} nmse={r['nmse']:8.3f} mse={r['mse']:11.1f} "
              f"mse/pers={r['mse_over_persistence']:.2f}", flush=True)
    # fill sanity: tobit vs pc vs avail at the first censored window
    i = ids_c[0]
    m = A["mask"][i]
    ii, jj = cens_runs(m)[0]
    x_t = fill_window("tobit", A["ctx_obs"][i], A["ctx_rec"][i], A["avail"][i], m)
    x_p = fill_window("pc", A["ctx_obs"][i], A["ctx_rec"][i], A["avail"][i], m)
    print(f"  window {i} first censored run [{ii},{jj}]: recorded="
          f"{np.round(A['ctx_rec'][i][ii:jj + 1], 0).tolist()}")
    print(f"    avail={np.round(A['avail'][i][ii:jj + 1], 0).tolist()}")
    print(f"    tobit={np.round(x_t[ii:jj + 1], 0).tolist()}  pc={np.round(x_p[ii:jj + 1], 0).tolist()}")
    if hasattr(model, "predict_quantiles"):
        q10, q90 = model.predict_quantiles(build_X("zero", ids_c, A))
        y = A["tgt"][ids_c].astype(np.float64)
        cov = ((y >= q10) & (y <= q90)).mean()
        print(f"  quantiles ok: q10/q90 shapes {q10.shape}, native cov(zero fill)={cov:.3f}")
        assert (q90 > q10).mean() > 0.99, "quantile order broken"
    for i in ids_c[:3]:
        mech, f = detect_mechanism(A["ctx_rec"][i], A["mask"][i])
        print(f"  detector window {i}: {mech} r_bar={f['r_bar']:.2f} "
              f"e_bar={f['e_bar']:.2f} run={f['mean_run']:.1f}")
    print("[smoke OK]")


if __name__ == "__main__":
    main()
