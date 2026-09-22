#!/usr/bin/env python
"""Screen S5-real: does the synthetic missing-context story replicate on REAL gaps?

run_s5_missing.py established the synthetic ground truth on ETT/weather:
MCAR + linear interpolation is nearly free, zero-fill is catastrophic (it
corrupts internal scaling statistics), and the oracle-rescale probe only
helps when the damage is scaling-statistic corruption (MCAR) -- under
informative (MNAR) missingness it backfires. This script re-runs the same
probes on PhysioNet'12, where the gaps are real: ICU measurements are
missing because clinicians chose not to measure, i.e. informative by
construction.

Data: tsdm's sparse Physionet'12 (sets A+B+C, 11981 patients), already
cached by repos/APN's pipeline at ~/.tsdm/datasets/Physionet2012 (read
only). Values are raw units on an hourly grid over the first 48h of each
ICU stay. APN's own convention (seq_len=36, pred_len=3) is too short-horizon
to be informative, and the tail of the 48h record is often absent entirely,
so windows are: context = hours 0..23, forecast = hours 24..35 (H=12).
One window per (patient, channel); a window is eligible iff the 12 target
hours are FULLY observed (natural missingness is never allowed in the
target) and the context has >= 2 observed points. <= 90 windows sampled per
channel (balanced; 11 eligible channels -> 990 windows).

Probes per model (chronos-bolt-base, timesfm-2.5-200m; t5 skipped -- the
small variant's tokenizer noise dominated its synthetic signal):
  fills on the natural context: zero / ffill / linear / zero_oscale
  (zero-fill + rescale by mean(|x_observed|)/mean(|x_zerofilled|) per
  window -- the deployable, observed-stats version of the synthetic
  study's oracle probe; under MCAR they coincide in expectation, under
  informative missingness the synthetic study predicts it should NOT
  fully recover).
  stress: extra MCAR {0.1, 0.3} on top of the natural gaps, linear fill,
  2 mask seeds -- the synthetic curve was flat, so it should stay flat.

Metrics: MSE/MAE per (model, fill), per channel and overall; per-channel
relMSE against the best fill; oscale recovery = (mse_zero - mse_oscale) /
(mse_zero - mse_linear). For the linear fill we also record the MNAR
signature from the synthetic study: MSE over future points above the
OBSERVED-context 90th percentile vs the rest. Results are appended to
s5_real_results.json after every config, so a crash loses nothing.
"""
import argparse
import json
import os
import tarfile
import io
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import pandas as pd
import torch

CTX, H = 24, 12          # context hours 0..23, target hours 24..35
GRID = 48
MIN_CTX_OBS = 2
CAP_PER_CHANNEL = 90
SEED = 20250811
EXTRA_RATES = [0.1, 0.3]
N_MASK_SEEDS = 2
RESULTS_PATH = os.path.join(HERE, "s5_real_results.json")
TSDM_CACHE = os.path.expanduser(
    "~/.tsdm/datasets/Physionet2012/Physionet2012-set-{}-sparse.tar")


# ---------------------------------------------------------------- models ----
class BoltModel:
    name, batch = "bolt", 1024

    def __init__(self, device):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
        self.device = device

    @torch.no_grad()
    def predict_point(self, ctx):  # ctx: [n, CTX] float32 -> [n, H]
        outs = []
        for i in range(0, len(ctx), self.batch):
            xb = torch.from_numpy(ctx[i:i + self.batch]).to(self.device)
            q = self.pipe.predict(xb, prediction_length=H)
            outs.append(q[:, 4, :].float().cpu().numpy())
        return np.concatenate(outs)


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


MODEL_REGISTRY = {"bolt": BoltModel, "timesfm": TimesFmModel}


# ------------------------------------------------------------- data prep ----

def load_p12_grid():
    """Returns values [P, 48, V] float32 (NaN = not observed), patient ids,
    variable names. Read-only against the tsdm cache created by repos/APN."""
    frames = []
    for s in "ABC":
        with tarfile.open(TSDM_CACHE.format(s)) as tf:
            frames.append(pd.read_feather(io.BytesIO(
                tf.extractfile("series.feather").read())))
    df = pd.concat(frames, ignore_index=True)
    varcols = [c for c in df.columns if c not in ("RecordID", "Time")]
    g = df.set_index(["RecordID", "Time"]).sort_index()
    pids = g.index.get_level_values(0).unique().to_numpy()
    pid_pos = {p: i for i, p in enumerate(pids)}
    X = np.full((len(pids), GRID, len(varcols)), np.nan, dtype=np.float32)
    ri = g.index.get_level_values(0).map(pid_pos).to_numpy()
    ti = g.index.get_level_values(1).to_numpy()
    for j, c in enumerate(varcols):
        X[ri, ti, j] = g[c].to_numpy(np.float32)
    return X, pids, varcols


def build_windows(X, pids, varcols, cap=CAP_PER_CHANNEL, seed=SEED):
    """Eligible (patient, channel) windows: target hours CTX..CTX+H-1 fully
    observed, >= MIN_CTX_OBS observed points in the context. Sample <= cap
    per channel. Returns list of dicts with window metadata."""
    rng = np.random.default_rng(seed)
    obs = np.isfinite(X)
    tgt_ok = obs[:, CTX:CTX + H, :].all(axis=1)          # [P, V]
    ctx_nobs = obs[:, :CTX, :].sum(axis=1)               # [P, V]
    windows = []
    for j, var in enumerate(varcols):
        elig = np.flatnonzero(tgt_ok[:, j] & (ctx_nobs[:, j] >= MIN_CTX_OBS))
        if len(elig) < 10:
            continue
        take = np.sort(rng.choice(elig, size=min(cap, len(elig)),
                                  replace=False)) if len(elig) > cap else elig
        for pi in take:
            ctx_obs = obs[pi, :CTX, j]
            windows.append(dict(
                window_id=len(windows), patient=int(pids[pi]), channel=var,
                ctx_missing_rate=float(1.0 - ctx_obs.mean()),
                ctx_nobs=int(ctx_obs.sum())))
    return windows


def extract(windows, X, pids, varcols):
    """Per-window observed context [W, CTX] (NaN = naturally missing) and
    fully-observed target [W, H]."""
    pid_pos = {p: i for i, p in enumerate(pids)}
    var_pos = {v: j for j, v in enumerate(varcols)}
    W = len(windows)
    ctx = np.full((W, CTX), np.nan, dtype=np.float32)
    tgt = np.full((W, H), np.nan, dtype=np.float32)
    for w, meta in enumerate(windows):
        pi, j = pid_pos[meta["patient"]], var_pos[meta["channel"]]
        ctx[w] = X[pi, :CTX, j]
        tgt[w] = X[pi, CTX:CTX + H, j]
    return ctx, tgt


# --------------------------------------------------------------- fills ------

def fill_context(x, fill):
    """x: [W, CTX] with NaN = missing. Returns filled copy (no NaN)."""
    t = np.arange(CTX)
    if fill == "zero":
        return np.nan_to_num(x)
    if fill == "zero_oscale":
        out = np.nan_to_num(x)
        num = np.nanmean(np.abs(x), axis=1, keepdims=True)   # observed stats
        den = np.abs(out).mean(axis=1, keepdims=True)
        scale = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 1.0)
        return out * scale
    out = np.empty_like(x)
    for w in range(len(x)):
        valid = np.flatnonzero(np.isfinite(x[w]))
        if len(valid) == 0:
            # extra MCAR can empty a sparse context entirely; fall back to
            # the zero-fill behaviour used in the synthetic study
            out[w] = 0.0
        elif fill == "ffill":
            pos = np.maximum(np.searchsorted(valid, t, side="right") - 1, 0)
            out[w] = x[w, valid[pos]]      # leading gaps take first valid value
        elif fill == "linear":
            out[w] = np.interp(t, valid, x[w, valid])  # edges -> ffill/bfill
        else:
            raise ValueError(fill)
    return out


def add_extra_mcar(ctx_obs, rate, mask_seed):
    """ctx_obs: [W, CTX] with NaN = naturally missing. Returns a copy with
    an extra iid Bernoulli(rate) of the OBSERVED points set to NaN.
    Deterministic in (window, seed, rate) for paired comparison."""
    out = ctx_obs.copy()
    rng = np.random.default_rng(np.random.SeedSequence(
        [SEED, mask_seed, int(round(rate * 100))]))
    drop = (rng.random(ctx_obs.shape) < rate) & np.isfinite(ctx_obs)
    # per-window RNG would be cleaner but slow; stream is deterministic and
    # identical across fills/models, which is all the pairing needs
    out[drop] = np.nan
    return out


# ------------------------------------------------------------ one config ----

def run_config(model, ctx_obs, tgt, windows, fill, extra_rate, mask_seed):
    ctx = add_extra_mcar(ctx_obs, extra_rate, mask_seed) if extra_rate > 0 \
        else ctx_obs
    miss_rate = float(np.isnan(ctx).mean())
    filled = fill_context(ctx, fill).astype(np.float32)
    pred = model.predict_point(filled).astype(np.float64)
    err2 = (pred - tgt.astype(np.float64)) ** 2
    res = {
        "model": model.name, "fill": fill,
        "extra_mcar": extra_rate, "mask_seed": mask_seed,
        "n_windows": len(windows),
        "ctx_missing_rate": miss_rate,
        "mse": float(err2.mean()), "mae": float(np.abs(pred - tgt).mean()),
        "mse_per_window": err2.mean(axis=1).tolist(),
        "mae_per_window": np.abs(pred - tgt).mean(axis=1).tolist(),
    }
    # per-channel aggregates (window order is fixed across configs)
    chans = [w["channel"] for w in windows]
    res["mse_per_channel"] = {c: float(err2.mean(axis=1)[[i for i, cc in enumerate(chans) if cc == c]].mean())
                              for c in sorted(set(chans))}
    if fill == "linear":
        # MNAR signature: error on future points above the OBSERVED-context
        # q90 vs the rest (fixed per-window threshold -> identical point set
        # across extra-mcar rates)
        q90 = np.nanquantile(ctx_obs, 0.9, axis=1, keepdims=True)
        top = tgt > q90
        with np.errstate(all="ignore"):
            td = np.nanmean(np.where(top, err2, np.nan), axis=1)
            rs = np.nanmean(np.where(~top, err2, np.nan), axis=1)
        res["mse_topdecile"] = float(np.nanmean(td))
        res["mse_rest"] = float(np.nanmean(rs))
        res["topdecile_frac"] = float(top.mean())
    return res


def cfgkey(r):
    return f"{r['model']}:{r['fill']}:{r['extra_mcar']}:{r['mask_seed']}"


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bolt,timesfm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true",
                    help="30 windows, 1 config, print and exit")
    args = ap.parse_args()

    t0 = time.time()
    X, pids, varcols = load_p12_grid()
    windows = build_windows(X, pids, varcols)
    ctx_obs, tgt = extract(windows, X, pids, varcols)
    chans = sorted({w["channel"] for w in windows})
    print(f"[data] {len(pids)} patients, {len(varcols)} vars, "
          f"{len(windows)} windows over {len(chans)} channels: {chans}")
    print(f"[data] window ctx missing rate: mean "
          f"{np.mean([w['ctx_missing_rate'] for w in windows]):.3f} "
          f"(min {np.min([w['ctx_missing_rate'] for w in windows]):.3f}, "
          f"max {np.max([w['ctx_missing_rate'] for w in windows]):.3f})")

    if args.smoke:
        ctx_obs, tgt = ctx_obs[:30], tgt[:30]
        windows = windows[:30]

    results = {"meta": dict(
        dataset="Physionet2012 (tsdm sparse, sets A+B+C, hourly grid, raw units)",
        ctx_hours=CTX, horizon_hours=H, min_ctx_obs=MIN_CTX_OBS,
        cap_per_channel=CAP_PER_CHANNEL, seed=SEED,
        channels=chans, windows=windows), "records": []}
    if os.path.exists(RESULTS_PATH) and not args.smoke:
        with open(RESULTS_PATH) as f:
            old = json.load(f)
        if old.get("meta", {}).get("windows") == results["meta"]["windows"]:
            results["records"] = old["records"]
            print(f"[resume] {len(old['records'])} records loaded")
    done = {cfgkey(r) for r in results["records"]}

    for mname in args.model.split(","):
        model = MODEL_REGISTRY[mname](args.device)
        print(f"[model] {mname} loaded ({time.time() - t0:.0f}s)")
        cfgs = [(f, 0.0, 0) for f in ("linear", "zero", "ffill", "zero_oscale")]
        cfgs += [("linear", r, s) for r in EXTRA_RATES
                 for s in range(N_MASK_SEEDS)]
        for fill, rate, ms in cfgs:
            key = f"{mname}:{fill}:{rate}:{ms}"
            if key in done:
                continue
            res = run_config(model, ctx_obs, tgt, windows, fill, rate, ms)
            print(f"  {key}: mse={res['mse']:.4f} mae={res['mae']:.4f} "
                  f"miss={res['ctx_missing_rate']:.3f} ({time.time() - t0:.0f}s)")
            if args.smoke:
                return
            results["records"].append(res)
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f)
        del model
        torch.cuda.empty_cache()

    print(f"[done] {len(results['records'])} records "
          f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
