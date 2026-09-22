#!/usr/bin/env python
"""Screen S5: do time-series foundation models break when the INPUT is missing?

TSFMs (Chronos-Bolt, Chronos-T5, TimesFM) are almost always evaluated on clean
lookback windows, but real deployment feeds them contexts with gaps. The naive
practice is fill-then-feed. This screen measures the degradation curve of
zero-shot forecasts as the missing rate in the CONTEXT rises, and localises the
damage to one of three components:

  (a) internal scaling statistics  -- e.g. Chronos normalises by mean(|x|);
      zero-filling 30% of the context deflates that statistic ~30%. Probed by
      `zero_oscale`: zero-fill, then rescale the context by the ORACLE factor
      mean(|x_clean|)/mean(|x_zerofilled|). Whatever error this removes was
      scaling-statistic corruption.
  (b) patching  -- Chronos-Bolt embeds 16-point patches; a single missing
      point (under naive fill) corrupts the whole patch token. Probed by
      contrasting `mcar` (scattered) vs `block` (contiguous length-24 runs):
      at equal missing rate mcar corrupts far more patches; plotting relMSE
      against the achieved corrupted-patch fraction tests whether the two
      mechanisms collapse onto one curve (patch-level story) or not
      (value-level story). Chronos-T5 (patch = 1) and TimesFM (patch = 32)
      bracket the patch-size axis.
  (c) value corruption per se -- whatever remains.

Design: context L=512, horizon H=96, scored future inside the last 20% of each
series; 300 window starts uniform at random per dataset; every channel is an
independent univariate series, masked independently; 2 mask seeds per
(window, config). Rates {0.1,0.3,0.5,0.7}, mechanisms {mcar, block}, fills
{zero, ffill, linear} + zero_oscale (mcar only) + an oracle anchor at p=0.3
that must reproduce the clean baseline. Models: chronos-bolt-base (full grid),
chronos-t5-small and timesfm-2.5-200m (reduced priority grid, time-boxed --
t5-base was benchmarked at 0.14 s/series and cannot leave batch 64 without
OOM, so the spec's permitted fallback t5-small is used; the timesfm pip
package only ships the 2.5-200M torch checkpoint, not 2.0-500M).

Point forecast: median (chronos: 0.5 quantile / median of 20 samples;
timesfm: the model's point head). MSE/MAE over the 96 steps, averaged over
channels, windows, mask seeds. Results (incl. per-window values) are appended
to s5_missing_results.json after every config, so a crash loses nothing and
re-running resumes where it stopped.

Update (MNAR extension, --mnar): value-censored mechanisms that hit exactly
ceil(p*L) masked points per window-channel by rank construction --
`mnar_high` censors the largest values (sensor saturation), `mnar_extreme`
censors the largest |z| against the window's own mean/std. Masks are
deterministic, so one mask seed suffices. Linear-fill configs additionally
record mse_topdecile / mse_rest: MSE over future ground-truth points
above/below the channel's CLEAN-context 90th percentile (fixed threshold, so
the extreme set is identical across mechanisms/fills) -- a direct test of
whether value-censored interpolation bias turns into systematic
under-prediction of extreme futures. Bolt also re-runs mcar:linear to
backfill the metric on the mcar records.
"""
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import pandas as pd
import torch

L, H = 512, 96
RATES = [0.1, 0.3, 0.5, 0.7]
BLOCK = 24
SEED = 20250810
RESULTS_PATH = os.path.join(HERE, "s5_missing_results.json")

DATASETS = {
    "ETTh1":  "tslib/dataset/ETT-small/ETTh1.csv",
    "ETTm1":  "tslib/dataset/ETT-small/ETTm1.csv",
    "weather": "tslib/dataset/weather/weather.csv",
}


# ---------------------------------------------------------------- models ----
class BoltModel:
    name, patch, batch = "bolt", 16, 1024

    def __init__(self, device):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
        self.device = device

    @torch.no_grad()
    def predict_point(self, ctx):  # ctx: [n, L] float32 -> [n, H]
        outs = []
        for i in range(0, len(ctx), self.batch):
            xb = torch.from_numpy(ctx[i:i + self.batch]).to(self.device)
            q = self.pipe.predict(xb, prediction_length=H)  # [b, 9, H], q[..,4]=median
            outs.append(q[:, 4, :].float().cpu().numpy())
        return np.concatenate(outs)


class T5Model:
    """chronos-t5-small: mean-scale + quantile-bin tokenizer, so 'patch' = 1."""
    name, patch, batch = "t5", 1, 128
    num_samples = 20

    def __init__(self, device):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small", device_map=device, torch_dtype=torch.float32)
        self.device = device

    @torch.no_grad()
    def predict_point(self, ctx):
        outs = []
        for i in range(0, len(ctx), self.batch):
            torch.manual_seed(SEED + i)  # deterministic sampling per batch
            # keep the context on CPU: MeanScaleUniformBins.boundaries is a
            # plain CPU tensor, and bucketize(cuda_input, cpu_boundaries) fails
            xb = torch.from_numpy(ctx[i:i + self.batch])
            s = self.pipe.predict(xb, prediction_length=H,
                                  num_samples=self.num_samples)  # [b, S, H]
            outs.append(s.median(dim=1).values.float().cpu().numpy())
        return np.concatenate(outs)


class TimesFmModel:
    name, patch, batch = "timesfm", 32, 512

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


MODEL_REGISTRY = {"bolt": BoltModel, "t5": T5Model, "timesfm": TimesFmModel}

# ------------------------------------------------------------ data/masks ----

def load_windows(path, n_windows, seed):
    df = pd.read_csv(path)
    X = df.drop(columns=["date"]).to_numpy(np.float32)  # [N, C]
    N = len(X)
    test_start = int(0.8 * N)
    lo, hi = test_start - L, N - L - H  # forecast origin in the last 20%
    n_valid = hi - lo + 1
    rng = np.random.default_rng(seed)
    k = min(n_windows, n_valid)
    starts = np.sort(rng.choice(n_valid, size=k, replace=False)) + lo
    return X, starts


def make_mask(mech, rate, window_idx, mask_seed, C, x=None):
    """[C, L] bool. Deterministic in (window, mask_seed, rate, mech) only, so
    all fills/models share identical masks for paired comparison. The mnar_*
    mechanisms are rank-based on the clean context x [C, L] and censor exactly
    ceil(p*L) positions per channel, so they need no randomness at all."""
    if mech in ("mnar_high", "mnar_extreme"):
        assert x is not None, "mnar masks need the clean context"
        k = int(np.ceil(rate * L))
        if mech == "mnar_high":            # sensor saturation: censor the largest values
            key = x
        else:                              # censor the largest |z| vs window mean/std
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True)
            key = np.abs((x - mu) / np.where(sd > 1e-12, sd, 1.0))
        m = np.zeros((C, L), bool)
        order = np.argsort(-key, axis=1, kind="stable")  # descending rank
        np.put_along_axis(m, order[:, :k], True, axis=1)
        return m
    rng = np.random.default_rng(np.random.SeedSequence(
        [SEED, window_idx, mask_seed, int(round(rate * 100)),
         0 if mech == "mcar" else 1]))
    if mech == "mcar":
        return rng.random((C, L)) < rate
    m = np.zeros((C, L), bool)
    n_blocks = int(round(rate * L / BLOCK))
    for c in range(C):
        for s in rng.integers(0, L - BLOCK + 1, size=n_blocks):
            m[c, s:s + BLOCK] = True
    return m


def fill_context(x, mask, fill):
    """x: [C, L] clean context; mask: [C, L] True=missing. Returns filled copy."""
    if fill in ("oracle", "none"):
        return x.copy()
    if fill == "zero":
        out = x.copy()
        out[mask] = 0.0
        return out
    if fill == "zero_oscale":
        out = x.copy()
        out[mask] = 0.0
        num = np.abs(x).mean(axis=1, keepdims=True)
        den = np.abs(out).mean(axis=1, keepdims=True)
        scale = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 1.0)
        return out * scale
    out = x.copy()
    t = np.arange(L)
    for c in range(len(x)):
        valid = np.flatnonzero(~mask[c])
        if len(valid) == 0:
            out[c] = 0.0
        elif fill == "ffill":
            pos = np.maximum(np.searchsorted(valid, t, side="right") - 1, 0)
            out[c] = x[c, valid[pos]]  # leading gaps take the first valid value
        elif fill == "linear":
            out[c] = np.interp(t, valid, x[c, valid])  # edges -> ffill/bfill
        else:
            raise ValueError(fill)
    return out


def corr_patch_frac(mask, patch):
    """[C, L] bool -> mean over channels of fraction of patches with >=1 miss."""
    if patch == 1:
        return float(mask.mean())
    P = L // patch
    return float(mask.reshape(mask.shape[0], P, patch).any(axis=2).mean())


# ------------------------------------------------------------ one config ----

def run_config(model, X, starts, cfg, n_seeds):
    C = X.shape[1]
    mech, fill, rate = cfg["mech"], cfg["fill"], cfg["rate"]
    # mnar masks are deterministic (rank-based), so extra mask seeds are
    # identical duplicates -- run one seed only
    det = mech in ("clean", "mnar_high", "mnar_extreme")
    ctxs, gts, masks, q90s = [], [], [], []
    for wi, s in enumerate(starts):
        x_clean = X[s:s + L].T.copy()       # [C, L]
        y = X[s + L:s + L + H].T.copy()     # [C, H]
        q90s_w = np.quantile(x_clean, 0.9, axis=1)  # per-channel context q90 [C]
        for ms in ((0,) if det else range(n_seeds)):
            mask = (np.zeros((C, L), bool) if mech == "clean"
                    else make_mask(mech, rate, wi, ms, C, x=x_clean))
            ctxs.append(fill_context(x_clean, mask, "none" if mech == "clean" else fill))
            gts.append(y)
            masks.append(mask)
            q90s.append(q90s_w)
    S = len(ctxs)
    ctx = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
    pred = model.predict_point(ctx).reshape(S, C, H).astype(np.float64)
    gt = np.stack(gts).astype(np.float64)
    err2 = (pred - gt) ** 2
    mse = err2.mean(axis=(1, 2))   # per (window, mask_seed)
    mae = np.abs(pred - gt).mean(axis=(1, 2))
    rate_sw = np.stack([m.mean() for m in masks])
    cpf_sw = np.array([corr_patch_frac(m, model.patch) for m in masks])
    nw = len(starts)
    res = {
        "mech": mech, "fill": fill, "rate": rate,
        "n_windows": nw, "n_seeds": 1 if det else n_seeds,
        "mse": float(mse.mean()), "mae": float(mae.mean()),
        "achieved_rate": float(rate_sw.mean()), "corr_patch_frac": float(cpf_sw.mean()),
        "mse_per_window": mse.reshape(nw, -1).mean(axis=1).tolist(),
        "mae_per_window": mae.reshape(nw, -1).mean(axis=1).tolist(),
        "achieved_rate_per_window": rate_sw.reshape(nw, -1).mean(axis=1).tolist(),
        "corr_patch_frac_per_window": cpf_sw.reshape(nw, -1).mean(axis=1).tolist(),
    }
    if fill == "linear" and mech != "clean":
        # extreme-value metric: error over future points above/below the
        # channel's CLEAN-context 90th percentile (fixed threshold -> the
        # top-decile point set is identical across fills and mechanisms)
        top = gt > np.stack(q90s)[:, :, None]          # [S, C, H]
        with np.errstate(all="ignore"):
            td = np.nanmean(np.where(top, err2, np.nan), axis=(1, 2))
            rs = np.nanmean(np.where(~top, err2, np.nan), axis=(1, 2))
        td_w = np.nanmean(td.reshape(nw, -1), axis=1)
        rs_w = np.nanmean(rs.reshape(nw, -1), axis=1)
        res["mse_topdecile"] = float(np.nanmean(td))
        res["mse_rest"] = float(np.nanmean(rs))
        res["topdecile_frac"] = float(top.mean())
        res["mse_topdecile_per_window"] = [None if np.isnan(v) else v for v in td_w.tolist()]
        res["mse_rest_per_window"] = [None if np.isnan(v) else v for v in rs_w.tolist()]
    return res


def cfgkey(cfg):
    return f"{cfg['mech']}:{cfg['fill']}:{cfg['rate']}"


def full_grid():
    cfgs = [dict(mech="clean", fill="none", rate=0.0)]
    for r in RATES:
        for mech in ("mcar", "block"):
            for fill in ("zero", "ffill", "linear"):
                cfgs.append(dict(mech=mech, fill=fill, rate=r))
    for r in RATES:
        cfgs.append(dict(mech="mcar", fill="zero_oscale", rate=r))
    cfgs.append(dict(mech="mcar", fill="oracle", rate=0.3))  # sanity anchor
    return cfgs


def reduced_grid():
    """Priority order for the time-boxed aux models: configs outer, datasets
    inner, so a budget cut truncates the tail instead of a whole dataset."""
    cfgs = [dict(mech="clean", fill="none", rate=0.0),
            dict(mech="mcar", fill="linear", rate=0.3),
            dict(mech="mcar", fill="zero", rate=0.3)]
    cfgs += [dict(mech="mcar", fill="linear", rate=r) for r in (0.1, 0.5, 0.7)]
    cfgs += [dict(mech="mcar", fill="zero", rate=r) for r in (0.1, 0.5, 0.7)]
    cfgs += [dict(mech="block", fill="linear", rate=r) for r in RATES]
    cfgs += [dict(mech="mcar", fill="zero_oscale", rate=r) for r in RATES]
    return cfgs


def mnar_grid():
    """Value-censored mechanisms, all fills, all models -- same windows/seeds
    as the mcar/block runs so results are directly comparable."""
    cfgs = []
    for r in RATES:
        for mech in ("mnar_high", "mnar_extreme"):
            for fill in ("zero", "ffill", "linear", "zero_oscale"):
                cfgs.append(dict(mech=mech, fill=fill, rate=r))
    return cfgs


# ------------------------------------------------------------------ main ----

def save_results(results, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true",
                    help="20 windows, ETTh1, bolt, rates {0,0.3} mcar all fills; prints, no save")
    ap.add_argument("--tiny-mnar", action="store_true",
                    help="20 windows, ETTh1, bolt, p=0.3 mnar_high all fills; prints, no save")
    ap.add_argument("--mnar", action="store_true",
                    help="run the mnar_high/mnar_extreme grid instead of the mcar/block grid;"
                         " bolt also re-runs mcar:linear to add the top-decile metric")
    ap.add_argument("--models", default="bolt,t5,timesfm")
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--aux-windows", type=int, default=100)
    ap.add_argument("--aux-seeds", type=int, default=1)
    ap.add_argument("--t5-budget", type=float, default=18.0, help="minutes")
    ap.add_argument("--timesfm-budget", type=float, default=15.0, help="minutes")
    ap.add_argument("--shard", type=int, default=0, help="shard index (0-based)")
    ap.add_argument("--nshards", type=int, default=1, help="number of shards")
    ap.add_argument("--out", default=RESULTS_PATH)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda"

    if args.tiny:
        model = BoltModel(device)
        X, starts = load_windows(DATASETS["ETTh1"], 20, SEED)
        cfgs = [dict(mech="clean", fill="none", rate=0.0)]
        for r in (0.0, 0.3):
            for fill in ("zero", "ffill", "linear", "oracle", "zero_oscale"):
                cfgs.append(dict(mech="mcar", fill=fill, rate=r))
        print(f"tiny: ETTh1 {len(starts)} windows x {X.shape[1]} ch, bolt, 2 mask seeds")
        for cfg in cfgs:
            t0 = time.time()
            res = run_config(model, X, starts, cfg, n_seeds=2)
            print(f"{cfgkey(cfg):24s} mse={res['mse']:10.4f} mae={res['mae']:7.4f} "
                  f"ach_rate={res['achieved_rate']:.3f} cpf={res['corr_patch_frac']:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        return

    if args.tiny_mnar:
        model = BoltModel(device)
        X, starts = load_windows(DATASETS["ETTh1"], 20, SEED)
        cfgs = [dict(mech="clean", fill="none", rate=0.0),
                dict(mech="mcar", fill="linear", rate=0.3)]
        for fill in ("zero", "ffill", "linear", "zero_oscale"):
            cfgs.append(dict(mech="mnar_high", fill=fill, rate=0.3))
        cfgs.append(dict(mech="mnar_extreme", fill="linear", rate=0.3))
        print(f"tiny-mnar: ETTh1 {len(starts)} windows x {X.shape[1]} ch, bolt, p=0.3")
        for cfg in cfgs:
            t0 = time.time()
            res = run_config(model, X, starts, cfg, n_seeds=2)
            td, rs = res.get("mse_topdecile"), res.get("mse_rest")
            extra = (f" td={td:9.4f} rest={rs:8.4f} frac={res['topdecile_frac']:.3f}"
                     if td is not None else "")
            print(f"{cfgkey(cfg):26s} mse={res['mse']:10.4f} ach={res['achieved_rate']:.6f} "
                  f"cpf={res['corr_patch_frac']:.3f}{extra} ({time.time() - t0:.1f}s)",
                  flush=True)
        return

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
        print(f"resume: loaded {args.out}", flush=True)
    results.setdefault("meta", {
        "L": L, "H": H, "rates": RATES, "block": BLOCK, "seed": SEED,
        "datasets": DATASETS, "windows": args.windows, "seeds": args.seeds,
        "aux_windows": args.aux_windows, "aux_seeds": args.aux_seeds,
        "models": {"bolt": "amazon/chronos-bolt-base (fp32, patch=16, median)",
                   "t5": "amazon/chronos-t5-small (fp32, patch=1, median of 20 samples)",
                   "timesfm": "google/timesfm-2.5-200m-pytorch (patch=32, point head)"},
        "test_region": "last 20% of timeline; forecast origin inside it",
        "point_forecast": "median (chronos) / point head (timesfm)",
    })

    for name in args.models.split(","):
        cls = MODEL_REGISTRY[name]
        budget = (None if name == "bolt" else
                  args.t5_budget * 60 if name == "t5" else args.timesfm_budget * 60)
        t_model0 = time.time()
        model = cls(device)
        results.setdefault(name, {})
        if args.mnar:
            grid = mnar_grid()
            if name == "bolt":
                # re-run bolt mcar:linear so those records gain the top-decile
                # metric; all other pre-existing keys are left untouched
                for ds in DATASETS:
                    for r in RATES:
                        results[name].get(ds, {}).pop(f"mcar:linear:{r}", None)
                grid = [dict(mech="mcar", fill="linear", rate=r) for r in RATES] + grid
        else:
            grid = full_grid() if name == "bolt" else reduced_grid()
        n_win = args.windows if name == "bolt" else args.aux_windows
        n_seeds = args.seeds if name == "bolt" else args.aux_seeds

        if name == "bolt":  # dataset-major: complete one dataset before moving on
            jobs = [(ds, cfg) for ds in DATASETS for cfg in grid]
        else:               # config-major (priority order), datasets inner
            jobs = [(ds, cfg) for cfg in grid for ds in DATASETS]
        if args.nshards > 1:  # strided slice spreads the costly weather jobs evenly
            jobs = jobs[args.shard::args.nshards]
            print(f"shard {args.shard}/{args.nshards}: {len(jobs)} jobs", flush=True)

        loaded = {}
        truncated = False
        for ds, cfg in jobs:
            key = cfgkey(cfg)
            if key in results[name].get(ds, {}):
                continue
            if budget is not None and time.time() - t_model0 > budget:
                print(f"{name}: budget exceeded, skipping {ds} {key} and rest", flush=True)
                truncated = True
                break
            if ds not in loaded:
                loaded[ds] = load_windows(DATASETS[ds], args.windows, SEED)
            X, starts_all = loaded[ds]
            starts = starts_all[:n_win]
            t0 = time.time()
            res = run_config(model, X, starts, cfg, n_seeds)
            res["seconds"] = round(time.time() - t0, 1)
            results[name].setdefault(ds, {})[key] = res
            save_results(results, args.out)
            print(f"{name:8s} {ds:8s} {key:24s} mse={res['mse']:10.4f} "
                  f"ach={res['achieved_rate']:.3f} cpf={res['corr_patch_frac']:.3f} "
                  f"({res['seconds']}s)", flush=True)
        results[name].setdefault("_status", {})
        results[name]["_status"] = {"truncated": truncated,
                                    "minutes": round((time.time() - t_model0) / 60, 1)}
        save_results(results, args.out)
        del model
        torch.cuda.empty_cache()

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
