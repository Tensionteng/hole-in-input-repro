#!/usr/bin/env python
"""Screen S6: REAL value-censored context -- Penmanshiel 2016 wind curtailment.

The synthetic S5 study established that MNAR (value-censored) context gaps
cause systematic under-prediction of peaks in zero-shot TSFM forecasts. This
script anchors that claim on a real dataset: Penmanshiel wind farm 2016
(Cubico, Zenodo 16807304, CC-BY-4.0), where the grid operator externally
clamped turbine output (Status-log code 9000 "P output externally reduced" /
9210 "Externally stopped") -- documented, value-dependent censoring of high
power values (see dataset_censor/DATA_README.md).

Design (univariate power, 10-min grid; context CTX=144 = 24h, horizon H=24 = 4h):
  cens   : windows whose context contains >= MIN_CENS effectively-censored
           points (clamp binding at high available power; treated as missing
           and filled). Target always fully observed & never curtailed.
  ctrl   : clean windows (no curtailment, no NaN in context+target), sampled
           1:1 per turbine against cens windows.
  ctrl_mcar / ctrl_mnar : the SAME control windows with synthetic masks at the
           per-window rate r_w of the paired cens window (mcar = iid
           Bernoulli; mnar = top-ceil(r_w*CTX) by value, rank-based like S5's
           mnar_high) -- the equal-rate random/extreme-missingness controls.
Configs per model: cens x {linear, zero, keep-as-is} + ctrl x {clean} +
  ctrl_mcar x {linear, zero} + ctrl_mnar x {linear, zero}.
Metrics: MSE/MAE, NMSE = MSE / max(var(target), 100 kW^2) per window, and the
  MNAR signature: MSE over future points above the window's context q90 vs
  the rest (clean-context q90 for ctrl windows, observed-context q90 for cens
  windows -- fixed threshold per window, so the split is identical across
  fills). relNMSE is against ctrl:clean.
Models: chronos-bolt-base, timesfm-2.5-200m, moirai-1.1-R-base (all cached).
Results appended to s6_real_censor_results.json after every config.
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

CTX, H = 144, 24
MIN_CENS = 7                 # >= ~5% of the context effectively censored
STRIDE = 6
SEED = 20250811
NPZ = os.path.join(HERE, "dataset_censor", "penmanshiel2016_processed.npz")
RESULTS_PATH = os.path.join(HERE, "s6_real_censor_results.json")
VAR_FLOOR = 100.0            # kW^2 (10 kW std) floor for the NMSE denominator


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
        try:
            self.predictor = forecast.create_predictor(batch_size=self.batch)
        except TypeError:
            import lightning as pl
            trainer = pl.Trainer(accelerator="gpu", devices=[0], logger=False,
                                 enable_checkpointing=False)
            self.predictor = forecast.create_predictor(trainer, batch_size=self.batch)

    def predict_point(self, ctx):
        outs = []
        for i in range(0, len(ctx), self.batch):
            chunk = ctx[i:i + self.batch]
            ds = [{"target": row, "start": pd.Period("2000-01-01", freq="10min")}
                  for row in chunk]
            fcsts = list(self.predictor.predict(ds))
            outs.append(np.stack([np.asarray(f.median, dtype=np.float32)
                                  for f in fcsts]))
        return np.concatenate(outs)


MODEL_REGISTRY = {"bolt": BoltModel, "timesfm": TimesFmModel, "moirai": MoiraiModel}


# ------------------------------------------------------------- data prep ----

def build_windows(npz_path=NPZ, seed=SEED):
    """Returns (windows, series) where windows is a list of dicts:
    kind in {cens, ctrl}; for ctrl, 'pair' = index of its cens window.
    series[tag] = dict(power, cens, curt)."""
    d = np.load(npz_path)
    tags = sorted({k[:-len("_power")] for k in d.files if k.endswith("_power")})
    rng = np.random.default_rng(seed)
    windows, series = [], {}
    for tag in tags:
        p, cens, curt = d[f"{tag}_power"], d[f"{tag}_cens"], d[f"{tag}_curt"]
        series[tag] = dict(power=p, cens=cens, curt=curt)
        n = len(p)
        cens_starts, ctrl_starts, rates = [], [], []
        for t in range(CTX, n - H, STRIDE):
            ctx_p = p[t - CTX:t]
            tgt_p = p[t:t + H]
            ctx_cens = cens[t - CTX:t]
            ctx_curt = curt[t - CTX:t]
            tgt_curt = curt[t:t + H]
            if not np.isfinite(tgt_p).all() or tgt_curt.any():
                continue
            n_cens = int(ctx_cens.sum())
            if not (np.isfinite(ctx_p) | ctx_cens).all():
                continue  # NaN in context outside the censor mask
            if n_cens >= MIN_CENS:
                cens_starts.append(t)
                rates.append(n_cens / CTX)
            elif not ctx_curt.any() and np.isfinite(ctx_p).all():
                ctrl_starts.append(t)
        # sample control starts 1:1 with cens starts for this turbine
        if cens_starts:
            take = rng.choice(len(ctrl_starts),
                              size=min(len(cens_starts), len(ctrl_starts)),
                              replace=False) if ctrl_starts else []
            base = len(windows)
            for i, t in enumerate(cens_starts):
                windows.append(dict(window_id=len(windows), turbine=tag, start=int(t),
                                    kind="cens", rate=float(rates[i]), pair=-1))
            for j, ti in enumerate(take):
                t = ctrl_starts[ti]
                windows.append(dict(window_id=len(windows), turbine=tag, start=int(t),
                                    kind="ctrl", rate=float(rates[j]), pair=base + j))
        print(f"  {tag}: {len(cens_starts)} cens, {len(ctrl_starts)} ctrl candidates")
    return windows, series


def extract(windows, series):
    """Per-window context (censored points -> NaN for kind=cens), clean target,
    clean context (for q90 of ctrl windows), and the censor mask."""
    W = len(windows)
    ctx_obs = np.full((W, CTX), np.nan, np.float32)   # censored points = NaN
    ctx_clean = np.full((W, CTX), np.nan, np.float32)  # as-recorded context
    tgt = np.full((W, H), np.nan, np.float32)
    for i, w in enumerate(windows):
        s = series[w["turbine"]]
        t = w["start"]
        c = s["power"][t - CTX:t].copy()
        ctx_clean[i] = c
        if w["kind"] == "cens":
            c[s["cens"][t - CTX:t]] = np.nan
        ctx_obs[i] = c
        tgt[i] = s["power"][t:t + H]
    return ctx_obs, ctx_clean, tgt


# --------------------------------------------------------------- masks ------

def fill_ctx(x, fill):
    """x: [CTX] with NaN = missing. Returns filled copy (no NaN)."""
    t = np.arange(CTX)
    valid = np.flatnonzero(np.isfinite(x))
    if fill == "zero":
        return np.nan_to_num(x)
    if len(valid) == 0:
        return np.zeros(CTX, np.float32)
    if fill == "linear":
        return np.interp(t, valid, x[valid]).astype(np.float32)
    raise ValueError(fill)


def synth_mask(kind, rate, window_id):
    """[CTX] bool mask for a control window at the given rate. Deterministic."""
    k = int(np.ceil(rate * CTX))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, window_id, int(round(rate * 1000))]))
    m = np.zeros(CTX, bool)
    if kind == "mcar":
        m[rng.choice(CTX, size=k, replace=False)] = True
    return m


def mnar_mask(x_clean, rate):
    """Top-ceil(rate*CTX) values of the clean context masked (S5 mnar_high)."""
    k = int(np.ceil(rate * CTX))
    m = np.zeros(CTX, bool)
    order = np.argsort(-x_clean, kind="stable")
    m[order[:k]] = True
    return m


# ------------------------------------------------------------ one config ----

def run_config(model, group, fill, windows, ctx_obs, ctx_clean, tgt):
    """group: cens|ctrl_clean|ctrl_mcar|ctrl_mnar. Returns metrics dict."""
    idx = np.array([i for i, w in enumerate(windows)
                    if (w["kind"] == "cens") == group.startswith("cens")])
    X, Y, q90, rates = [], [], [], []
    for i in idx:
        w = windows[i]
        if group == "cens" and fill == "keep":
            X.append(ctx_clean[i])          # feed clamped values as-is
        elif group == "cens":
            X.append(fill_ctx(ctx_obs[i], fill))
        elif group == "ctrl_clean":
            X.append(ctx_clean[i])
        else:
            m = (synth_mask("mcar", w["rate"], w["window_id"]) if group == "ctrl_mcar"
                 else mnar_mask(ctx_clean[i], w["rate"]))
            xm = ctx_clean[i].copy()
            xm[m] = np.nan
            X.append(fill_ctx(xm, fill))
        Y.append(tgt[i])
        # fixed per-window top-decile threshold: clean-context q90 for ctrl
        # windows, observed-context q90 for cens windows
        ref = ctx_clean[i] if w["kind"] == "ctrl" else ctx_obs[i]
        q90.append(np.nanquantile(ref, 0.9))
        rates.append(w["rate"] if group != "ctrl_clean" else 0.0)
    X = np.stack(X).astype(np.float32)
    Y = np.stack(Y).astype(np.float64)
    pred = model.predict_point(X).astype(np.float64)
    err2 = (pred - Y) ** 2
    mse_w = err2.mean(axis=1)
    var = np.maximum(Y.var(axis=1), VAR_FLOOR)
    nmse_w = mse_w / var
    top = Y > np.array(q90)[:, None]
    with np.errstate(all="ignore"):
        td = np.nanmean(np.where(top, err2, np.nan), axis=1)
        rs = np.nanmean(np.where(~top, err2, np.nan), axis=1)
    return dict(
        model=model.name, group=group, fill=fill, n_windows=len(idx),
        mean_censor_rate=float(np.mean(rates)),
        mse=float(mse_w.mean()), mae=float(np.abs(pred - Y).mean()),
        nmse=float(nmse_w.mean()),
        mse_topdecile=float(np.nanmean(td)), mse_rest=float(np.nanmean(rs)),
        topdecile_frac=float(top.mean()),
        nmse_per_window=nmse_w.tolist(), mse_per_window=mse_w.tolist(),
        mse_topdecile_per_window=[None if np.isnan(v) else v for v in td.tolist()],
        mse_rest_per_window=[None if np.isnan(v) else v for v in rs.tolist()],
    )


CONFIGS = [("ctrl_clean", "clean"),
           ("ctrl_mcar", "linear"), ("ctrl_mcar", "zero"),
           ("ctrl_mnar", "linear"), ("ctrl_mnar", "zero"),
           ("cens", "linear"), ("cens", "zero"), ("cens", "keep")]


def cfgkey(r):
    return f"{r['model']}:{r['group']}:{r['fill']}"


# ------------------------------------------------------------------ main ----

def save(results):
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, RESULTS_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bolt,timesfm,moirai")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--smoke", action="store_true",
                    help="bolt only, 40 cens + 40 ctrl windows, print and exit")
    args = ap.parse_args()

    t0 = time.time()
    windows, series = build_windows()
    n_cens = sum(1 for w in windows if w["kind"] == "cens")
    print(f"[data] {len(windows)} windows ({n_cens} cens, {len(windows) - n_cens} ctrl); "
          f"censor rates: mean {np.mean([w['rate'] for w in windows if w['kind'] == 'cens']):.3f}")
    ctx_obs, ctx_clean, tgt = extract(windows, series)

    if args.smoke:
        keep = [w for w in windows if w["kind"] == "cens"][:40] + \
               [w for w in windows if w["kind"] == "ctrl"][:40]
        idx = [w["window_id"] for w in keep]
        windows = keep
        ctx_obs, ctx_clean, tgt = ctx_obs[idx], ctx_clean[idx], tgt[idx]
        args.model = "bolt"

    results = {"meta": dict(
        dataset="Penmanshiel 2016 (Cubico, Zenodo 16807304, CC-BY-4.0), 10-min power",
        ctx=CTX, horizon=H, min_cens=MIN_CENS, stride=STRIDE, seed=SEED,
        censoring="Status-log codes 9000/9210 (external reduction/stop) + binding rule "
                  "(avail-power margin 0.15*rated, avail>0.30*rated)",
        windows=windows), "records": []}
    if os.path.exists(RESULTS_PATH) and not args.smoke:
        with open(RESULTS_PATH) as f:
            old = json.load(f)
        if old.get("meta", {}).get("windows") == results["meta"]["windows"]:
            results["records"] = old["records"]
            print(f"[resume] {len(old['records'])} records")
    done = {cfgkey(r) for r in results["records"]}

    for mname in args.model.split(","):
        model = MODEL_REGISTRY[mname](args.device)
        print(f"[model] {mname} ({time.time() - t0:.0f}s)", flush=True)
        for group, fill in CONFIGS:
            key = f"{mname}:{group}:{fill}"
            if key in done:
                continue
            res = run_config(model, group, fill, windows, ctx_obs, ctx_clean, tgt)
            td, rs = res["mse_topdecile"], res["mse_rest"]
            print(f"  {key:28s} nmse={res['nmse']:.4f} mse={res['mse']:9.1f} "
                  f"td/rest={td / max(rs, 1e-9):5.2f} ({time.time() - t0:.0f}s)", flush=True)
            if args.smoke:
                continue
            results["records"].append(res)
            save(results)
        del model
        torch.cuda.empty_cache()
    print(f"[done] {len(results['records'])} records ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
