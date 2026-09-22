#!/usr/bin/env python
"""Track C: supplementary mechanism experiments for screen S5 (see run_s5_missing.py).

Reuses S5's data loading, window sampling, fill and inference conventions
(imported from run_s5_missing). Adds three new mask constructions / metrics:

  Part 1 -- block-length sweep ("interpolability curve"): bolt only, ETTh1 +
    weather, rates {0.1, 0.3}, gap lengths {4,8,16,24,48,96,168}, linear fill.
    Unlike S5's overlapping-block mask (achieved rate < nominal), `blockB`
    masks here censor EXACTLY round(p*L) points per channel, arranged as
    floor(k/B) disjoint length-B gaps plus one remainder gap of k%B -- so the
    rate is held exactly constant along the gap-length axis. (At p=0.1 a gap
    longer than k=51 is infeasible: B=96/168 collapse to a single maximal
    51-point gap; realized max gap length is recorded per config.)

  Part 2 -- MAR control: `mar` masks are driven by an EXTERNAL process, not
    the value: per channel an independent AR(1) latent (phi=0.95, unit gaussian
    innovations, stationary init), standardized over the window, thresholded at
    its own (1-p) quantile -> temporally clustered but value-independent
    missingness. The latent is rate-independent, so masks nest across rates.
    Compared against `mcar` and exact-rate `block24` references at the same
    windows/seeds (equal achieved rate), bolt + timesfm, all 3 datasets.

  Part 3 -- calibration under missingness: bolt's predictive quantiles
    (0.1..0.9). Empirical coverage of the central 80% PI (q0.1..q0.9) on the
    future, for mcar and mnar_high (rank-based censoring of the largest values,
    reused from run_s5_missing.make_mask), linear fill, rates {0.1..0.7}, all 3
    datasets. Also records coverage split by future-extreme vs rest (same
    clean-context q90 threshold as S5's top-decile metric) and mean PI width.
    Results are stored under the model key "bolt_cal" so they never collide
    with part-2 point-forecast records.

  Part 4 (optional) -- Moirai (Salesforce/moirai-1.1-R-base via uni2ts):
    mechanisms {mcar, block(24 orig), mnar_high} x fills {zero, linear} x rates
    {0.1,0.3,0.5,0.7}, 100 windows x 1 seed, all 3 datasets. Median of
    20 samples as point forecast.

Common: context L=512, horizon H=96, test region = last 20%, 150 windows
(100 for part 4) x 1 mask seed, same seed base 20250810 as S5. Note the
150-window sample differs from S5's 300-window sample (rng.choice depends on
k), so part-2 mcar/block references are re-run here rather than reused.

Results appended to --out after every config (crash-safe, resumable). Separate
processes must write separate --out files (shard convention, merged later).
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

from run_s5_missing import (L, H, RATES, SEED, DATASETS, load_windows,
                            make_mask, fill_context, corr_patch_frac,
                            save_results)

RESULTS_PATH = os.path.join(HERE, "s5_extra_results.json")
BLOCK_SWEEP = (4, 8, 16, 24, 48, 96, 168)
MAR_PHI = 0.95


# ---------------------------------------------------------------- models ----
class BoltModel:
    name, patch, batch = "bolt", 16, 1024

    def __init__(self, device):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
        self.device = device

    @torch.no_grad()
    def predict_quant(self, ctx):  # ctx: [n, L] float32 -> [n, 9, H] (q0.1..q0.9)
        outs = []
        for i in range(0, len(ctx), self.batch):
            xb = torch.from_numpy(ctx[i:i + self.batch]).to(self.device)
            q = self.pipe.predict(xb, prediction_length=H)
            outs.append(q.float().cpu().numpy())
        return np.concatenate(outs)

    @torch.no_grad()
    def predict_point(self, ctx):  # -> [n, H] median
        return self.predict_quant(ctx)[:, 4, :]


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


class MoiraiModel:
    """Salesforce/moirai-1.1-R-base via uni2ts; univariate, median of samples."""
    name, patch, batch = "moirai", 16, 250
    num_samples = 20

    def __init__(self, device):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        forecast = MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base"),
            prediction_length=H, context_length=L, patch_size=16,
            num_samples=self.num_samples, target_dim=1,
            feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        try:
            self.predictor = forecast.create_predictor(batch_size=self.batch)
        except TypeError:  # older uni2ts wants an explicit lightning trainer
            import lightning as pl
            trainer = pl.Trainer(accelerator="gpu", devices=[0], logger=False,
                                 enable_checkpointing=False)
            self.predictor = forecast.create_predictor(trainer, batch_size=self.batch)

    def predict_point(self, ctx):
        outs = []
        for i in range(0, len(ctx), self.batch):
            chunk = ctx[i:i + self.batch]
            ds = [{"target": row, "start": pd.Period("2000-01-01", freq="h")}
                  for row in chunk]
            fcsts = list(self.predictor.predict(ds))
            outs.append(np.stack([np.asarray(f.median, dtype=np.float32)
                                  for f in fcsts]))
        return np.concatenate(outs)


MODEL_REGISTRY = {"bolt": BoltModel, "timesfm": TimesFmModel, "moirai": MoiraiModel}

# ------------------------------------------------------------ new masks ----

def make_block_sweep_mask(rate, block, window_idx, mask_seed, C):
    """[C, L] bool. EXACTLY round(rate*L) points per channel, as floor(k/B)
    disjoint length-B gaps + one remainder gap of k%B (rejection-sampled
    starts, non-overlapping). If B > k the whole budget is one maximal gap of
    length k. Deterministic in (window, seed, rate, B)."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [SEED, window_idx, mask_seed, int(round(rate * 100)), block]))
    k = int(round(rate * L))
    m = np.zeros((C, L), bool)
    if k == 0:
        return m
    n_full, rem = divmod(k, block)
    for c in range(C):
        for b in [block] * n_full + ([rem] if rem else []):
            for _ in range(500):
                s = int(rng.integers(0, L - b + 1))
                if not m[c, s:s + b].any():
                    m[c, s:s + b] = True
                    break
            else:  # pragma: no cover - unreachable at p <= 0.3 densities
                s = int(rng.integers(0, L - b + 1))
                m[c, s:s + b] = True
    return m


def make_mar_latent(window_idx, mask_seed, C, phi=MAR_PHI):
    """[C, L] AR(1) latent, rate-independent so masks nest across rates."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [SEED, window_idx, mask_seed, 777]))
    e = rng.standard_normal((C, L))
    z = np.empty((C, L))
    z[:, 0] = rng.standard_normal(C) / np.sqrt(1.0 - phi ** 2)  # stationary init
    for t in range(1, L):
        z[:, t] = phi * z[:, t - 1] + e[:, t]
    sd = z.std(axis=1, keepdims=True)
    return (z - z.mean(axis=1, keepdims=True)) / np.where(sd > 1e-12, sd, 1.0)


def make_mar_mask(rate, window_idx, mask_seed, C):
    """Threshold the AR(1) latent at its per-channel (1-rate) quantile:
    temporally clustered, value-independent missingness at rate ~= rate."""
    z = make_mar_latent(window_idx, mask_seed, C)
    thr = np.quantile(z, 1.0 - rate, axis=1, keepdims=True)
    return z > thr


def runlen_stats(mask):
    """[C, L] bool -> (mean run length, mean max run length) over channels."""
    means, maxs = [], []
    for row in mask:
        d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
        lens = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
        means.append(float(lens.mean()) if len(lens) else 0.0)
        maxs.append(float(lens.max()) if len(lens) else 0.0)
    return float(np.mean(means)), float(np.mean(maxs))


def dispatch_mask(mech, rate, wi, ms, C, x_clean):
    if mech == "clean":
        return np.zeros((C, L), bool)
    if mech == "mar":
        return make_mar_mask(rate, wi, ms, C)
    if mech.startswith("block") and mech != "block":
        return make_block_sweep_mask(rate, int(mech[len("block"):]), wi, ms, C)
    return make_mask(mech, rate, wi, ms, C, x=x_clean)  # mcar / block(24) / mnar_*


# ------------------------------------------------------------ one config ----

def collect_contexts(X, starts, cfg, n_seeds):
    """Shared by point/quant runners: build (ctx_filled, gt, mask, q90ctx) per
    (window, mask_seed), flattened the same way as run_s5_missing.run_config."""
    C = X.shape[1]
    mech, fill, rate = cfg["mech"], cfg["fill"], cfg["rate"]
    det = mech in ("clean", "mnar_high", "mnar_extreme")
    ctxs, gts, masks, q90s = [], [], [], []
    for wi, s in enumerate(starts):
        x_clean = X[s:s + L].T.copy()
        y = X[s + L:s + L + H].T.copy()
        q90s_w = np.quantile(x_clean, 0.9, axis=1)
        for ms in ((0,) if det else range(n_seeds)):
            mask = dispatch_mask(mech, rate, wi, ms, C, x_clean)
            ctxs.append(fill_context(x_clean, mask, "none" if mech == "clean" else fill))
            gts.append(y)
            masks.append(mask)
            q90s.append(q90s_w)
    return ctxs, gts, masks, q90s


def base_record(cfg, starts, masks, n_seeds):
    rate_sw = np.stack([m.mean() for m in masks])
    rl = np.array([runlen_stats(m) for m in masks])  # [S, 2] (mean runlen, max runlen)
    nw = len(starts)
    return {
        "mech": cfg["mech"], "fill": cfg["fill"], "rate": cfg["rate"],
        "n_windows": nw,
        "n_seeds": len(masks) // nw,
        "achieved_rate": float(rate_sw.mean()),
        "mean_runlen": float(rl[:, 0].mean()),
        "mean_maxrun": float(rl[:, 1].mean()),
        "achieved_rate_per_window": rate_sw.reshape(nw, -1).mean(axis=1).tolist(),
        "mean_runlen_per_window": rl[:, 0].reshape(nw, -1).mean(axis=1).tolist(),
        "mean_maxrun_per_window": rl[:, 1].reshape(nw, -1).mean(axis=1).tolist(),
    }


def run_config_point(model, X, starts, cfg, n_seeds):
    C = X.shape[1]
    ctxs, gts, masks, _ = collect_contexts(X, starts, cfg, n_seeds)
    S = len(ctxs)
    ctx = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
    pred = model.predict_point(ctx).reshape(S, C, H).astype(np.float64)
    gt = np.stack(gts).astype(np.float64)
    err2 = (pred - gt) ** 2
    mse = err2.mean(axis=(1, 2))
    mae = np.abs(pred - gt).mean(axis=(1, 2))
    nw = len(starts)
    res = base_record(cfg, starts, masks, n_seeds)
    res.update({
        "mse": float(mse.mean()), "mae": float(mae.mean()),
        "corr_patch_frac": float(np.mean([corr_patch_frac(m, model.patch) for m in masks])),
        "mse_per_window": mse.reshape(nw, -1).mean(axis=1).tolist(),
        "mae_per_window": mae.reshape(nw, -1).mean(axis=1).tolist(),
    })
    return res


def run_config_quant(model, X, starts, cfg, n_seeds):
    """Bolt quantiles -> coverage of the central 80% PI (q0.1..q0.9)."""
    C = X.shape[1]
    ctxs, gts, masks, q90s = collect_contexts(X, starts, cfg, n_seeds)
    S = len(ctxs)
    ctx = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
    q = model.predict_quant(ctx).reshape(S, C, 9, H).astype(np.float64)
    gt = np.stack(gts).astype(np.float64)
    q10, q50, q90 = q[:, :, 0], q[:, :, 4], q[:, :, 8]
    hit = (gt >= q10) & (gt <= q90)                      # [S, C, H]
    cov = hit.mean(axis=(1, 2))
    width = (q90 - q10).mean(axis=(1, 2))
    err2 = (q50 - gt) ** 2
    mse = err2.mean(axis=(1, 2))
    nw = len(starts)
    res = base_record(cfg, starts, masks, n_seeds)
    res.update({
        "coverage80": float(cov.mean()),
        "pi_width80": float(width.mean()),
        "mse": float(mse.mean()),
        "coverage80_per_window": cov.reshape(nw, -1).mean(axis=1).tolist(),
        "pi_width80_per_window": width.reshape(nw, -1).mean(axis=1).tolist(),
        "mse_per_window": mse.reshape(nw, -1).mean(axis=1).tolist(),
    })
    if cfg["mech"] != "clean":
        # split coverage / median-MSE by future-extreme vs rest, using the same
        # fixed clean-context q90 threshold as S5's top-decile metric
        top = gt > np.stack(q90s)[:, :, None]            # [S, C, H]
        with np.errstate(all="ignore"):
            cov_td = np.nanmean(np.where(top, hit.astype(float), np.nan), axis=(1, 2))
            cov_rs = np.nanmean(np.where(~top, hit.astype(float), np.nan), axis=(1, 2))
            td = np.nanmean(np.where(top, err2, np.nan), axis=(1, 2))
            rs = np.nanmean(np.where(~top, err2, np.nan), axis=(1, 2))
        res.update({
            "coverage80_topdecile": float(np.nanmean(cov_td)),
            "coverage80_rest": float(np.nanmean(cov_rs)),
            "mse_topdecile": float(np.nanmean(td)),
            "mse_rest": float(np.nanmean(rs)),
            "coverage80_topdecile_per_window":
                [None if np.isnan(v) else v for v in
                 np.nanmean(cov_td.reshape(nw, -1), axis=1).tolist()],
            "coverage80_rest_per_window":
                [None if np.isnan(v) else v for v in
                 np.nanmean(cov_rs.reshape(nw, -1), axis=1).tolist()],
        })
    return res


# ----------------------------------------------------------------- grids ----

def cfgkey(cfg):
    return f"{cfg['mech']}:{cfg['fill']}:{cfg['rate']}"


def part_grid(part):
    cfgs = [dict(mech="clean", fill="none", rate=0.0)]
    if part == 1:
        for r in (0.1, 0.3):
            for b in BLOCK_SWEEP:
                cfgs.append(dict(mech=f"block{b}", fill="linear", rate=r))
    elif part == 2:
        for r in RATES:
            for mech in ("mcar", "block24", "mar"):
                cfgs.append(dict(mech=mech, fill="linear", rate=r))
    elif part == 3:
        for r in RATES:
            for mech in ("mcar", "mnar_high"):
                cfgs.append(dict(mech=mech, fill="linear", rate=r))
    elif part == 4:
        for r in RATES:
            for mech in ("mcar", "block", "mnar_high"):
                for fill in ("zero", "linear"):
                    cfgs.append(dict(mech=mech, fill=fill, rate=r))
    else:
        raise ValueError(part)
    return cfgs


PART_SPECS = {
    # part: (models, datasets, windows, seeds, runner)
    1: (("bolt",), ("ETTh1", "weather"), 150, 1, "point"),
    2: (("bolt", "timesfm"), ("ETTh1", "ETTm1", "weather"), 150, 1, "point"),
    3: (("bolt_cal",), ("ETTh1", "ETTm1", "weather"), 150, 1, "quant"),
    4: (("moirai",), ("ETTh1", "ETTm1", "weather"), 100, 1, "point"),
}


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="1,2,3")
    ap.add_argument("--models", default=None,
                    help="restrict within-part models, e.g. 'timesfm' or 'bolt'")
    ap.add_argument("--out", default=RESULTS_PATH)
    ap.add_argument("--tiny", action="store_true",
                    help="smoke test: 10 windows, ETTh1, bolt, a few configs, no save")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda"

    if args.tiny:
        model = BoltModel(device)
        X, starts = load_windows(DATASETS["ETTh1"], 10, SEED)
        cfgs = [dict(mech="clean", fill="none", rate=0.0),
                dict(mech="block4", fill="linear", rate=0.1),
                dict(mech="block168", fill="linear", rate=0.1),
                dict(mech="mar", fill="linear", rate=0.3),
                dict(mech="mcar", fill="linear", rate=0.3),
                dict(mech="mnar_high", fill="linear", rate=0.3)]
        for cfg in cfgs:
            t0 = time.time()
            rp = run_config_point(model, X, starts, cfg, n_seeds=1)
            rq = run_config_quant(model, X, starts, cfg, n_seeds=1)
            print(f"{cfgkey(cfg):22s} mse={rp['mse']:8.4f} ach={rp['achieved_rate']:.3f} "
                  f"runlen={rp['mean_runlen']:5.1f}/{rp['mean_maxrun']:5.1f} "
                  f"cov80={rq['coverage80']:.3f} w80={rq['pi_width80']:.2f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        return

    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
        print(f"resume: loaded {args.out}", flush=True)
    results.setdefault("meta", {
        "track": "C supplementary (parts 1-4); see run_s5_extra.py docstring",
        "L": L, "H": H, "rates": RATES, "block_sweep": BLOCK_SWEEP,
        "mar_phi": MAR_PHI, "seed": SEED, "datasets": DATASETS,
        "windows": 150, "seeds": 1, "part4_windows": 100,
        "masks": {"blockB": "exact round(p*L) pts, disjoint length-B gaps + remainder",
                  "mar": "AR(1) phi=0.95 latent, per-channel (1-p) quantile threshold; nested across rates",
                  "mcar/block/mnar_high": "reused from run_s5_missing.make_mask"},
        "note": "150-window sample differs from S5's 300-window sample; "
                "part-2 mcar/block24 references re-run at these windows. "
                "bolt_cal = bolt quantile (coverage) runs.",
    })

    loaded = {}
    for part in [int(p) for p in args.parts.split(",")]:
        models, datasets, n_win, n_seeds, runner = PART_SPECS[part]
        if args.models:
            keep = set(args.models.split(","))
            models = tuple(m for m in models
                           if m in keep or (m == "bolt_cal" and "bolt" in keep))
        grid = part_grid(part)
        for name in models:
            cls = MODEL_REGISTRY["bolt" if name == "bolt_cal" else name]
            t_model0 = time.time()
            model = cls(device)
            results.setdefault(name, {})
            jobs = [(ds, cfg) for ds in datasets for cfg in grid]
            for ds, cfg in jobs:
                key = cfgkey(cfg)
                if key in results[name].get(ds, {}):
                    continue
                if (ds, n_win) not in loaded:
                    loaded[(ds, n_win)] = load_windows(DATASETS[ds], n_win, SEED)
                X, starts = loaded[(ds, n_win)]
                t0 = time.time()
                res = (run_config_quant if runner == "quant" else run_config_point)(
                    model, X, starts, cfg, n_seeds)
                res["seconds"] = round(time.time() - t0, 1)
                results[name].setdefault(ds, {})[key] = res
                save_results(results, args.out)
                extra = (f"cov80={res['coverage80']:.3f}" if runner == "quant"
                         else f"runlen={res['mean_runlen']:.1f}/{res['mean_maxrun']:.0f}")
                print(f"p{part} {name:8s} {ds:8s} {key:22s} mse={res['mse']:10.4f} "
                      f"ach={res['achieved_rate']:.3f} {extra} ({res['seconds']}s)",
                      flush=True)
            results[name].setdefault("_status", {})
            results[name]["_status"][f"part{part}"] = {
                "done": True, "minutes": round((time.time() - t_model0) / 60, 1)}
            save_results(results, args.out)
            del model
            torch.cuda.empty_cache()

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
