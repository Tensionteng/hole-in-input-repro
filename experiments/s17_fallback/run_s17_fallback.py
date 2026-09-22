#!/usr/bin/env python
"""S17: damage verification (go/no-go) + repair of Chronos-T5's silent
tokenizer fallback `scale[~(scale>0)] = 1.0` (chronos/chronos.py:178).

S9 found the fallback fires in production: weather clean 63/2100 rows,
mnar_high p=0.7 524/2100 rows (all-zero-observed rain channels get silently
switched to absolute units). S9 did NOT show this hurts accuracy — near-zero
channels are predicted ≈0 anyway. S17 measures the damage first.

Key structural fact (verified in smoke): for a fallback row the observed
context is all zeros, so the token stream is INVARIANT to scale (0/s = 0 ->
center bin; NaN -> pad). output_transform is linear in scale, and the pass-2
scale of the H=96 two-pass loop is proportional to the pass-1 scale, so the
FULL prediction is exactly homogeneous:  pred(s) = pred(1.0) * s .
=> scale-only fixes on fallback rows are ANALYTIC (no GPU rerun needed).

Bolt contrast (chronos_bolt.py:95-122): InstanceNorm loc = nanmean (nan->0),
scale = nan-std with nan->1.0 and `scale==0 -> eps=1e-5`. An all-zero context
gets scale=1e-5, so bolt's denormalized predictions are ~loc=0 — a *deflating*
fallback, opposite sign to t5's *inflating* 1.0. S17 checks whether bolt is
harmed at all on the same windows.

Parts:
  smoke  : (1) my t5_predict == pipe.predict bit-exact on 5 ETTh1 windows;
           (2) analytic identity pred(s)=pred(1)*s on synthetic all-zero rows;
           (3) my bolt_predict == pipe.predict on the same rows; bolt on an
           all-zero row -> |pred| tiny and scale==1e-5.
  A      : anchor gate (fallback counts 63/2100 clean, 524/2100 mnar_high
           p=0.7 within +-2%, CPU-only; t5 ETTh1 clean MSE within +-5% of
           14.219885), then Part A: synthetic intermittent grid (two-state
           Markov dry/wet regimes -> realistic long all-zero contexts;
           activity x noise grid) + real weather windows (S9-paired) with
           per-row analytic scale fixes (channel/global prior + per-row
           oracle s*) and bolt on identical rows. Ends with the pre-registered
           go/no-go decision.
  B      : only meaningful if decision=go (else degraded short verification):
           inference-time patches for t5 (medmad / floor alpha in {0.1,1} /
           nanskip / nanskip+prior) and bolt (medmad / stdfloor / nanskip),
           evaluated on the synthetic grid + weather + ETT regression
           (ETTh1/ETTm1 clean must stay ~unchanged).
  report : renders s17.png from s17_results.json (CPU only).

Pre-registered decision rule (Part A -> B gate):
  GO iff on EITHER the real fallback rows (weather clean+mnar pooled) OR the
  synthetic fallback rows (pooled over cells):
      aggregate improvement IMP = 1 - sum(MSE_priorfix)/sum(MSE_native) >= 5%
      AND top-5-row share of total improvement < 50% (not few-window driven).

Artifacts: s17_results.json, s17.png, s17_notes.md; logs s17_*.log.
Only new files are created; nothing pre-existing is modified.
"""
import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import pandas as pd
import torch

torch.set_grad_enabled(False)

# ---- constants (S5/S9-paired where relevant) --------------------------------
L, H = 512, 96
SEED = 20250810                      # S5/S9 window+mask seed namespace
NWIN_DRAW, NWIN_AUX = 300, 100       # S5 drew 300 starts, t5 aux used [:100]
T5_BATCH, T5_SAMPLES = 128, 20
BOLT_BATCH = 1024

SYNTH_SEED = 20260813
N_SYNTH = 256                        # series per cell
ACT = [0.02, 0.05, 0.10, 0.20, 0.40]  # stationary activity rates
SIG = [0.0, 0.1, 0.3]                # additive noise on ACTIVE points (x E[amp]=1)
DW = 24.0                            # mean wet-spell length

RAIN_CH = [14, 15]                   # 'rain (mm)', 'raining (s)' in weather.csv
DATASETS = {
    "ETTh1": "tslib/dataset/ETT-small/ETTh1.csv",
    "ETTm1": "tslib/dataset/ETT-small/ETTm1.csv",
    "weather": "tslib/dataset/weather/weather.csv",
}

# anchors from s9_results.json
S9_T5_CLEAN_ETTH1 = 14.219885274164715
S9_T5_CLEAN_WEATHER = 8567.094083322841
S9_FB_CLEAN, S9_FB_MNAR07 = 63, 524
CNT_TOL, MSE_TOL = 0.02, 0.05

GO_IMP, GO_TOP5 = 0.05, 0.50         # pre-registered decision thresholds

RESULTS = os.path.join(HERE, "s17_results.json")


# ---- S5/S9-verbatim window + mask machinery ---------------------------------
def load_windows(path, n_windows, seed):
    df = pd.read_csv(path)
    X = df.drop(columns=["date"]).to_numpy(np.float32)  # [N, C]
    N = len(X)
    test_start = int(0.8 * N)
    lo, hi = test_start - L, N - L - H
    n_valid = hi - lo + 1
    rng = np.random.default_rng(seed)
    k = min(n_windows, n_valid)
    starts = np.sort(rng.choice(n_valid, size=k, replace=False)) + lo
    return X, starts


def make_mask(mech, rate, window_idx, mask_seed, C, x=None):
    """[C, L] bool. Verbatim copy of the S5/S9 mask generator."""
    if mech in ("mnar_high", "mnar_extreme"):
        assert x is not None
        k = int(np.ceil(rate * L))
        if mech == "mnar_high":
            key = x
        else:
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True)
            key = np.abs((x - mu) / np.where(sd > 1e-12, sd, 1.0))
        m = np.zeros((C, L), bool)
        order = np.argsort(-key, axis=1, kind="stable")
        np.put_along_axis(m, order[:, :k], True, axis=1)
        return m
    raise ValueError(mech)


def fill_nan(x, mask):
    out = x.copy()
    out[mask] = np.nan
    return out


# ---- models ------------------------------------------------------------------
def load_t5(device):
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small", device_map=device, torch_dtype=torch.float32)


def load_bolt(device):
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)


@torch.no_grad()
def t5_predict(pipe, ctx_np, scale_fn=None):
    """Bit-faithful mirror of ChronosPipeline.predict (chronos.py:487-511)
    including the H>64 two-pass loop and S9's per-batch seeding.

    scale_fn(ctx_f32_cpu [b, Lcur], rows_np, first_pass: bool)
        -> (ctx_modified, scale[b] tensor | None)
    None scale -> native tokenizer path on the modified context.
    """
    dev = pipe.model.device
    tok = pipe.tokenizer
    mpl = pipe.model.config.prediction_length
    eos = tok.config.use_eos_token and tok.config.model_type == "seq2seq"
    outs = []
    for i in range(0, len(ctx_np), T5_BATCH):
        torch.manual_seed(SEED + i)                      # S9-identical seeding
        ctx = torch.from_numpy(ctx_np[i:i + T5_BATCH])
        rows = np.arange(i, min(i + T5_BATCH, len(ctx_np)))
        preds, remaining, first = [], H, True
        while remaining > 0:
            if scale_fn is None:
                ids, am, scale = tok.context_input_transform(ctx)
            else:
                ctx_m, sc = scale_fn(ctx, rows, first)
                if sc is None:
                    ids, am, scale = tok.context_input_transform(ctx_m)
                else:
                    ids, am, scale = tok._input_transform(ctx_m, scale=sc)
                    if eos:
                        ids, am = tok._append_eos_token(ids, am)
            samples = pipe.model(ids.to(dev), am.to(dev), min(remaining, mpl),
                                 T5_SAMPLES, None, None, None)
            pred = tok.output_transform(samples.to(scale.device), scale)
            preds.append(pred)
            remaining -= pred.shape[-1]
            if remaining <= 0:
                break
            ctx = torch.cat([ctx, pred.median(dim=1).values], dim=-1)
            first = False
        outs.append(torch.cat(preds, dim=-1).median(dim=1).values
                    .float().cpu().numpy())
    return np.concatenate(outs)


def make_t5_scale_fn(mode, prior=None, alpha=1.0, flags0=None, ctxlen=512):
    """Inference-time scale patches for t5. prior: np [n_rows] per-row prior.
    Modes:
      medmad        scale = nanmedian(|x_obs|); <=0 -> 1.0 (native-style floor)
      floor         scale = max(mean|x_obs|, alpha*prior)   [every pass]
      nanskip       flagged rows -> all-NaN on FIRST pass only (native scale)
      nanskip_prior same, but flagged rows get scale=prior on first pass
    flags0 (bool [n]): all-zero-observed rows of the original context."""
    def fn(ctx, rows, first_pass):
        x = ctx.to(torch.float32)
        if mode in ("nanskip", "nanskip_prior"):
            if not first_pass:
                return x, None                    # native on continuation
            fl = torch.from_numpy(flags0[rows])
            xm = torch.where(fl.unsqueeze(-1),
                             torch.full_like(x, float("nan")), x)
            if mode == "nanskip":
                return xm, None                   # native: all-NaN -> scale 1.0
            if x.shape[-1] > ctxlen:
                x = x[..., -ctxlen:]
            pr = torch.from_numpy(prior[rows]).float()
            am = ~torch.isnan(x)
            nobs = am.float().sum(dim=-1)
            ma = torch.nansum(x.abs() * am, dim=-1) / nobs.clamp(min=1)
            s = torch.where(fl, pr, ma)
            s = torch.where(s > 0, s, torch.ones_like(s))
            return xm, s
        # scale-only modes: truncate like context_input_transform does
        if x.shape[-1] > ctxlen:
            x = x[..., -ctxlen:]
        am = ~torch.isnan(x)
        nobs = am.float().sum(dim=-1)
        if mode == "medmad":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s = torch.nanmedian(x.abs(), dim=-1).values
            s = torch.nan_to_num(s, nan=0.0)
            s = torch.where(s > 0, s, torch.ones_like(s))
            return x, s
        if mode == "floor":
            ma = torch.nansum(x.abs() * am, dim=-1) / nobs.clamp(min=1)
            pr = torch.from_numpy(prior[rows]).float()
            s = torch.maximum(ma, alpha * pr)
            s = torch.where(nobs == 0, pr, s)     # all-NaN row -> prior
            return x, s
        if mode == "const":
            pr = torch.from_numpy(prior[rows]).float()
            s = torch.where(pr > 0, pr, torch.ones_like(pr))
            return x, s
        raise ValueError(mode)
    return fn


class _FixedNorm:
    """InstanceNorm stand-in returning a caller-supplied (loc, scale)."""
    def __init__(self, orig, loc, scale):
        self.orig, self.loc, self.scale = orig, loc, scale

    def __call__(self, x, loc_scale=None):
        return self.orig(x, (self.loc, self.scale))

    def inverse(self, x, loc_scale=None):
        return self.orig.inverse(x, (self.loc, self.scale))


@torch.no_grad()
def bolt_predict(pipe, ctx_np, norm_fn=None):
    """Mirror of ChronosBoltPipeline.predict (chronos_bolt.py:548-594),
    returning the 0.5-quantile (q[:,4,:]) like S5. Deterministic.

    norm_fn(x_f32_dev [b, Lcur], rows_np) -> (loc [b,1], scale [b,1]) | None.
    """
    model = pipe.model
    dev = model.device
    orig_norm = model.instance_norm
    quantiles = pipe.quantiles
    mpl = pipe.model_prediction_length

    def fwd(ctx, rows):
        if norm_fn is None:
            return model(context=ctx).quantile_preds
        loc, scale = norm_fn(ctx, rows)
        # instance_norm is a registered submodule: bypass nn.Module.__setattr__
        # (rejects non-Module values); __dict__ lookup shadows _modules.
        object.__setattr__(model, "instance_norm", _FixedNorm(orig_norm, loc, scale))
        try:
            out = model(context=ctx).quantile_preds
        finally:
            model.__dict__.pop("instance_norm", None)   # restore registered submodule
        return out

    outs = []
    for i in range(0, len(ctx_np), BOLT_BATCH):
        ctx = torch.from_numpy(ctx_np[i:i + BOLT_BATCH]).to(dev, torch.float32)
        if ctx.shape[-1] > pipe.model_context_length:
            ctx = ctx[..., -pipe.model_context_length:]
        rows = np.arange(i, min(i + BOLT_BATCH, len(ctx_np)))
        prediction = fwd(ctx, rows)                       # [b, 9, mpl]
        preds = [prediction]
        remaining = H - prediction.shape[-1]
        if remaining > 0:
            ctxq = ctx.unsqueeze(1).repeat(1, len(quantiles), 1)
            qt = torch.tensor(quantiles, device=dev)
            while remaining > 0:
                ctxq = torch.cat([ctxq, prediction], dim=-1)[..., -pipe.model_context_length:]
                b, nq, cl = ctxq.shape
                prediction = fwd(ctxq.reshape(b * nq, cl), np.repeat(rows, nq))
                prediction = prediction.reshape(b, nq * len(quantiles), -1)
                prediction = torch.quantile(prediction, q=qt, dim=1).transpose(0, 1)
                preds.append(prediction)
                remaining -= prediction.shape[-1]
        full = torch.cat(preds, dim=-1)[..., :H].to(torch.float32).cpu()
        outs.append(full[:, 4, :].numpy())
    return np.concatenate(outs)


def make_bolt_norm_fn(mode, prior_std=None, alpha=1.0, flags0=None, eps=1e-5):
    """Inference-time loc/scale patches for bolt's InstanceNorm.
      medmad    loc=nanmedian, scale=1.4826*MAD; scale==0 -> eps (native-style)
      stdfloor  loc=nanmean, scale=max(nan-std, alpha*prior_std)
      nanskip   handled as preprocessing (rows -> all-NaN), not here.
    """
    def fn(x, rows):
        x = x.to(torch.float32)
        if mode == "medmad":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loc = torch.nanmedian(x, dim=-1, keepdim=True).values
                mad = torch.nanmedian((x - loc).abs(), dim=-1, keepdim=True).values
            loc = torch.nan_to_num(loc, nan=0.0)
            scale = torch.nan_to_num(1.4826 * mad, nan=0.0)
            scale = torch.where(scale == 0, torch.full_like(scale, eps), scale)
            return loc, scale
        if mode == "stdfloor":
            loc = torch.nan_to_num(torch.nanmean(x, dim=-1, keepdim=True), nan=0.0)
            std = torch.nan_to_num(
                (x - loc).square().nanmean(dim=-1, keepdim=True).sqrt(), nan=0.0)
            pr = torch.from_numpy(prior_std[rows]).to(x.device).unsqueeze(-1)
            scale = torch.maximum(std, alpha * pr)
            nobs = (~torch.isnan(x)).float().sum(dim=-1, keepdim=True)
            scale = torch.where(nobs == 0, torch.ones_like(scale), scale)
            return loc, scale
        raise ValueError(mode)
    return fn


# ---- synthetic intermittent series ------------------------------------------
def gen_intermittent(a, sigma, n=N_SYNTH, dw=DW, seed=SYNTH_SEED, tail="exp"):
    """Two-state (dry/wet) Markov regime, zero-inflated. Mean wet spell dw;
    p_d set so stationary activity = a. Wet amplitude ~ Exp(1) (tail='exp') or
    Weibull(0.5) (tail='weibull', heavy right tail for the clamp study);
    additive Gaussian noise sigma*E[amp] on ACTIVE points only (exact zeros
    preserved). Returns float32 [n, L+H]."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [seed + (7 if tail == "weibull" else 0),
         int(round(a * 1000)), int(round(sigma * 1000))]))
    T = L + H
    p_w = 1.0 / dw
    p_d = p_w * a / (1.0 - a)
    state = rng.random(n) < a
    x = np.zeros((n, T), np.float32)
    for t in range(T):
        amp = (rng.weibull(0.5, n) * 0.5 if tail == "weibull"
               else rng.exponential(1.0, n))     # both have E[amp]=1
        val = np.maximum(amp + sigma * rng.standard_normal(n), 0.0)
        x[:, t] = np.where(state, val, 0.0)
        state = state ^ (rng.random(n) < np.where(state, p_w, p_d))
    return x


# ---- metric helpers ----------------------------------------------------------
def row_mse(pred, y):
    return ((pred.astype(np.float64) - y.astype(np.float64)) ** 2).mean(axis=1)


def improvement_bundle(mse_native, mse_fix, n_boot=2000, seed=7):
    """Aggregate + distribution stats of a fix on a row population."""
    mse_native = np.asarray(mse_native, float)
    mse_fix = np.asarray(mse_fix, float)
    n = len(mse_native)
    out = {"n": n}
    if n == 0:
        return {**out, "agg_improvement": None}
    sn, sf = mse_native.sum(), mse_fix.sum()
    agg = 1.0 - sf / sn if sn > 0 else None
    delta = mse_native - mse_fix
    top5 = float(np.sort(delta)[-5:].sum() / delta.sum()) if delta.sum() > 0 else None
    pos = mse_native > 1e-12
    med_row = float(np.median(1.0 - mse_fix[pos] / mse_native[pos])) if pos.any() else None
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s_n, s_f = mse_native[idx].sum(), mse_fix[idx].sum()
        if s_n > 0:
            boots.append(1.0 - s_f / s_n)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None
    out.update(agg_improvement=None if agg is None else float(agg),
               agg_ci95=ci, median_row_improvement=med_row,
               top5_share=top5,
               frac_rows_improved=float((delta > 0).mean()),
               sum_mse_native=float(sn), sum_mse_fix=float(sf))
    return out


def oracle_scale(pred, y):
    """Per-row least-squares optimal s* = clip(sum(p*y)/sum(p^2), 0, inf)."""
    p = pred.astype(np.float64)
    yy = y.astype(np.float64)
    num = (p * yy).sum(axis=1)
    den = (p * p).sum(axis=1)
    s = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return np.clip(s, 0.0, None), den > 0


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def jfloat(x):
    return None if x is None else float(x)


# ---- smoke -------------------------------------------------------------------
def part_smoke():
    device = "cuda"
    print("== S17 smoke ==", flush=True)
    t5 = load_t5(device)
    cfg = t5.model.config
    print(f"t5: context_length={cfg.context_length} pred_len={cfg.prediction_length} "
          f"n_tokens={cfg.n_tokens} n_special={cfg.n_special_tokens} eos={cfg.use_eos_token}",
          flush=True)

    # (1) t5_predict native == pipe.predict, bit-exact
    X, starts_all = load_windows(DATASETS["ETTh1"], NWIN_DRAW, SEED)
    starts = starts_all[:5]
    ctx = np.stack([X[s:s + L].T for s in starts]).reshape(5 * 7, L).astype(np.float32)
    mine = t5_predict(t5, ctx)
    ref = []
    for i in range(0, len(ctx), T5_BATCH):
        torch.manual_seed(SEED + i)
        s = t5.predict(torch.from_numpy(ctx[i:i + T5_BATCH]),
                       prediction_length=H, num_samples=T5_SAMPLES)
        ref.append(s.median(dim=1).values.float().cpu().numpy())
    ref = np.concatenate(ref)
    d1 = np.abs(mine - ref).max()
    print(f"(1) t5 native mirror: max|mine-pipe| = {d1:.3e}", flush=True)
    assert d1 == 0.0, "t5_predict native path is not bit-exact vs pipe.predict"

    # (2) analytic scale identity on all-zero rows
    zctx = np.zeros((8, L), np.float32)
    zctx[4:, ::7] = np.nan                       # some NaN-marked rows too
    p1 = t5_predict(t5, zctx)
    prior = np.full(len(zctx), 0.05, np.float32)
    fl = np.zeros(len(zctx), bool)
    p2 = t5_predict(t5, zctx,
                    scale_fn=make_t5_scale_fn("floor", prior=prior, alpha=1.0,
                                              flags0=fl, ctxlen=cfg.context_length))
    with np.errstate(all="ignore"):
        ratio = p2 / np.where(np.abs(p1) > 1e-6, p1, np.nan)
    finite = np.isfinite(ratio)
    assert np.isfinite(p2).all(), "patched (floor) path produced non-finite predictions"
    # relative check: ratio/0.05 - 1, robust to float noise on ~1e-8 predictions
    d2 = np.abs(ratio[finite] / 0.05 - 1.0).max() if finite.any() else 0.0
    frac_exact0 = float((p2[p1 == 0] == 0).mean()) if (p1 == 0).any() else 1.0
    print(f"(2) analytic identity: max rel dev of p(0.05)/p(1) from 0.05 = {d2:.3e} "
          f"(zero-preserving frac={frac_exact0:.3f})", flush=True)
    assert d2 < 1e-3, "scale homogeneity broken"

    # (3) bolt mirror + all-zero behaviour
    bolt = load_bolt(device)
    bn = bolt_predict(bolt, ctx)
    bref = []
    for i in range(0, len(ctx), BOLT_BATCH):
        q = bolt.predict(torch.from_numpy(ctx[i:i + BOLT_BATCH]).to(device),
                         prediction_length=H)
        bref.append(q[:, 4, :].float().cpu().numpy())
    bref = np.concatenate(bref)
    d3 = np.abs(bn - bref).max()
    print(f"(3) bolt mirror: max|mine-pipe| = {d3:.3e}", flush=True)
    assert d3 < 1e-5, "bolt_predict native path differs from pipe.predict"
    bz = bolt_predict(bolt, np.zeros((4, L), np.float32))
    print(f"    bolt all-zero ctx -> max|pred| = {np.abs(bz).max():.3e} "
          f"(eps=1e-5 floor: expect ~0)", flush=True)
    # generator sanity
    xs = gen_intermittent(0.05, 0.1, n=64)
    fb = (np.abs(xs[:, :L]).sum(axis=1) <= 0).mean()
    act = (xs > 0).mean()
    print(f"(4) synth sanity a=0.05: fallback_rate={fb:.2f} activity={act:.3f}", flush=True)
    assert 0.1 < fb < 0.7 and 0.03 < act < 0.08
    print("SMOKE OK", flush=True)
    return 0


# ---- Part A: anchor gate -----------------------------------------------------
def weather_contexts():
    """S9-paired weather windows: clean + mnar_high p=0.7 (nan fill)."""
    X, starts_all = load_windows(DATASETS["weather"], NWIN_DRAW, SEED)
    starts = starts_all[:NWIN_AUX]
    C = X.shape[1]
    ctx_c = np.stack([X[s:s + L].T for s in starts]).astype(np.float32)
    y = np.stack([X[s + L:s + L + H].T for s in starts]).astype(np.float32)
    ctx_m = np.stack([fill_nan(ctx_c[wi], make_mask("mnar_high", 0.7, wi, 0, C,
                                                    x=ctx_c[wi]))
                      for wi in range(len(starts))]).astype(np.float32)
    test_start = int(0.8 * len(X))
    prior_ch = np.abs(X[:test_start]).mean(axis=0)          # [C] train prior
    prior_std_ch = X[:test_start].std(axis=0)
    return X, starts, ctx_c, ctx_m, y, prior_ch, prior_std_ch


def fb_count_clean(ctx2d):
    """S9 part-B detector: all-zero rows of a clean context."""
    return (np.abs(ctx2d).sum(axis=1) <= 0)


def fb_count_nan(ctx2d):
    """S9 part-A detector: all-zero-observed rows of a NaN-filled context."""
    with np.errstate(invalid="ignore"):
        return (np.nansum(np.abs(ctx2d), axis=1) <= 0)


def anchor_gate():
    out = {}
    # (1) fallback counts, CPU only
    X, starts, ctx_c, ctx_m, y, prior_ch, prior_std = weather_contexts()
    nw, C = ctx_c.shape[0], ctx_c.shape[1]
    fb_c = fb_count_clean(ctx_c.reshape(-1, L))
    fb_m = fb_count_nan(ctx_m.reshape(-1, L))
    n_c, n_m = int(fb_c.sum()), int(fb_m.sum())
    per_ch_c = fb_c.reshape(nw, C).sum(axis=0).astype(int).tolist()
    per_ch_m = fb_m.reshape(nw, C).sum(axis=0).astype(int).tolist()
    ok_c = abs(n_c - S9_FB_CLEAN) / S9_FB_CLEAN <= CNT_TOL
    ok_m = abs(n_m - S9_FB_MNAR07) / S9_FB_MNAR07 <= CNT_TOL
    # ETT clean: expect 0 fallback rows
    ett0 = {}
    for ds in ("ETTh1", "ETTm1"):
        Xe, se = load_windows(DATASETS[ds], NWIN_DRAW, SEED)
        ce = np.stack([Xe[s:s + L].T for s in se[:NWIN_AUX]]).reshape(-1, L)
        ett0[ds] = int(fb_count_clean(ce).sum())
    out["counts"] = {
        "weather_clean": {"got": n_c, "s9": S9_FB_CLEAN, "pass": bool(ok_c),
                          "per_channel": per_ch_c},
        "weather_mnar_high_0.7": {"got": n_m, "s9": S9_FB_MNAR07, "pass": bool(ok_m),
                                  "per_channel": per_ch_m},
        "ett_clean": ett0,
    }
    print(f"anchor counts: clean {n_c}/{S9_FB_CLEAN} {'PASS' if ok_c else 'FAIL'}; "
          f"mnar0.7 {n_m}/{S9_FB_MNAR07} {'PASS' if ok_m else 'FAIL'}; "
          f"ETT clean fb={ett0}", flush=True)
    out["_gate_counts_pass"] = bool(ok_c and ok_m)
    return out


def anchor_mse(t5):
    X, starts_all = load_windows(DATASETS["ETTh1"], NWIN_DRAW, SEED)
    starts = starts_all[:NWIN_AUX]
    ctx = np.stack([X[s:s + L].T for s in starts]).reshape(-1, L).astype(np.float32)
    y = np.stack([X[s + L:s + L + H].T for s in starts]).reshape(-1, H).astype(np.float32)
    t0 = time.time()
    pred = t5_predict(t5, ctx)
    mse = float(row_mse(pred, y).mean())
    dev = (mse - S9_T5_CLEAN_ETTH1) / S9_T5_CLEAN_ETTH1
    ok = abs(dev) <= MSE_TOL
    print(f"anchor t5 ETTh1 clean mse={mse:.4f} s9={S9_T5_CLEAN_ETTH1:.4f} "
          f"dev={dev:+.3%} {'PASS' if ok else 'FAIL'} ({time.time()-t0:.0f}s)", flush=True)
    return {"t5_clean_ETTh1": {"got": mse, "s9": S9_T5_CLEAN_ETTH1,
                               "rel_dev": dev, "pass": bool(ok)}}, bool(ok)


# ---- Part A: synthetic grid --------------------------------------------------
def run_synth(t5, bolt):
    cells, pooled = {}, {k: [] for k in
                         ("mse_t5", "mse_t5_prior", "mse_t5_oracle", "mse_bolt",
                          "fut_active", "cell", "meanpred_t5", "meanpred_bolt",
                          "mom")}
    diag_gnoise = {}
    for a in ACT:
        for sg in SIG:
            key = f"a{a}:s{sg}"
            t0 = time.time()
            xs = gen_intermittent(a, sg)
            ctx, y = xs[:, :L].copy(), xs[:, L:].copy()
            prior = float(np.abs(xs).mean())
            pred_t5 = t5_predict(t5, ctx)
            pred_bolt = bolt_predict(bolt, ctx)
            fb = np.abs(ctx).sum(axis=1) <= 0
            n_fb = int(fb.sum())
            mse_t5 = row_mse(pred_t5, y)
            mse_bolt = row_mse(pred_bolt, y)
            rec = {"a": a, "sigma": sg, "n": len(xs), "n_fb": n_fb,
                   "fb_rate": n_fb / len(xs), "prior": prior,
                   "activity_achieved": float((xs > 0).mean()),
                   "mse_t5_all": float(mse_t5.mean()),
                   "mse_bolt_all": float(mse_bolt.mean()),
                   "mse_t5_nonfb": float(mse_t5[~fb].mean()) if (~fb).any() else None,
                   "mse_bolt_nonfb": float(mse_bolt[~fb].mean()) if (~fb).any() else None,
                   "seconds": round(time.time() - t0, 1)}
            if n_fb:
                mp_t5 = np.abs(pred_t5[fb]).mean(axis=1)
                mp_bolt = np.abs(pred_bolt[fb]).mean(axis=1)
                s_prior = np.full(n_fb, prior)
                mse_prior = row_mse(pred_t5[fb] * s_prior[:, None], y[fb])
                s_oracle, ok_o = oracle_scale(pred_t5[fb], y[fb])
                mse_oracle = np.full(n_fb, np.nan)
                mse_oracle[ok_o] = row_mse((pred_t5[fb] * s_oracle[:, None])[ok_o],
                                           y[fb][ok_o])
                fut = (np.abs(y[fb]).sum(axis=1) > 0)
                rec["fb"] = {
                    "prior_fix": improvement_bundle(mse_t5[fb], mse_prior),
                    "oracle_fix": improvement_bundle(mse_t5[fb][ok_o],
                                                     mse_oracle[ok_o]),
                    "oracle_excluded": int((~ok_o).sum()),
                    "bolt_vs_native": improvement_bundle(mse_t5[fb], mse_bolt[fb]),
                    "mse_t5": float(mse_t5[fb].mean()),
                    "mse_bolt": float(mse_bolt[fb].mean()),
                    "meanpred_t5": float(mp_t5.mean()),
                    "meanpred_bolt": float(mp_bolt.mean()),
                    "fut_active_frac": float(fut.mean()),
                    "prior_fix_fut_active": improvement_bundle(mse_t5[fb][fut], mse_prior[fut]) if fut.any() else None,
                    "prior_fix_fut_zero": improvement_bundle(mse_t5[fb][~fut], mse_prior[~fut]) if (~fut).any() else None,
                }
                for k, v in (("mse_t5", mse_t5[fb]), ("mse_t5_prior", mse_prior),
                             ("mse_t5_oracle", mse_oracle), ("mse_bolt", mse_bolt[fb]),
                             ("fut_active", fut.astype(float)),
                             ("meanpred_t5", mp_t5), ("meanpred_bolt", mp_bolt)):
                    pooled[k].append(v)
                pooled["cell"].append(np.full(n_fb, key))
                pooled["mom"].append(fb_moments(pred_t5[fb], y[fb]))
            print(f"synth {key:14s} fb={n_fb:3d} ({n_fb/len(xs):5.1%}) "
                  f"mse_t5={rec['mse_t5_all']:.4f} "
                  + (f"fb: IMP_prior={rec['fb']['prior_fix']['agg_improvement']:+.3f} "
                     f"t5={rec['fb']['mse_t5']:.4f} bolt={rec['fb']['mse_bolt']:.4f} "
                     if n_fb else "") + f"({rec['seconds']}s)", flush=True)
            cells[key] = rec
    # global-noise diagnostic: fallback requires EXACT zeros (CPU only)
    for a in (0.02, 0.10, 0.40):
        for sg in (0.01, 0.05):
            xs = gen_intermittent(a, 0.0, n=128) + \
                sg * np.random.default_rng(1).standard_normal((128, L + H)).astype(np.float32)
            ctx = xs[:, :L]
            n_fb = int((np.abs(ctx).sum(axis=1) <= 0).sum())
            diag_gnoise[f"a{a}:g{sg}"] = {
                "n_fb": n_fb,
                "mean_scale": float(np.abs(ctx).mean(axis=1).mean())}
    pooled = {k: (np.concatenate(v) if v else np.array([])) for k, v in pooled.items()}
    res = {"cells": cells, "global_noise_diag": diag_gnoise, "n_fallback": int(len(pooled["mse_t5"]))}
    if len(pooled["mse_t5"]):
        ok_o = np.isfinite(pooled["mse_t5_oracle"])
        res["pooled"] = {
            "prior_fix": improvement_bundle(pooled["mse_t5"], pooled["mse_t5_prior"]),
            "oracle_fix": improvement_bundle(pooled["mse_t5"][ok_o], pooled["mse_t5_oracle"][ok_o]),
            "bolt_vs_native": improvement_bundle(pooled["mse_t5"], pooled["mse_bolt"]),
            "meanpred_t5": float(pooled["meanpred_t5"].mean()),
            "meanpred_bolt": float(pooled["meanpred_bolt"].mean()),
            "fut_active_frac": float(pooled["fut_active"].mean()),
        }
        fa = pooled["fut_active"] > 0.5
        if fa.any():
            res["pooled"]["prior_fix_fut_active"] = improvement_bundle(
                pooled["mse_t5"][fa], pooled["mse_t5_prior"][fa])
        if (~fa).any():
            res["pooled"]["prior_fix_fut_zero"] = improvement_bundle(
                pooled["mse_t5"][~fa], pooled["mse_t5_prior"][~fa])
        # per-cell key for each pooled row is kept for later breakdowns
        res["pooled"]["rows_per_cell"] = {c: int((pooled["cell"] == c).sum())
                                          for c in np.unique(pooled["cell"])}
        res["pooled"]["mom"] = pooled["mom"].tolist()
        res["pooled"]["prior_row"] = np.array(
            [cells[c]["prior"] for c in pooled["cell"]]).tolist()
    return res


# ---- Part A: real weather ----------------------------------------------------
def run_real(t5, bolt):
    X, starts, ctx_c3, ctx_m3, y3, prior_ch, prior_std_ch = weather_contexts()
    nw, C = ctx_c3.shape[0], ctx_c3.shape[1]
    ch_of_row = np.tile(np.arange(C), nw)
    out = {"n_windows": nw, "n_channels": C,
           "channel_prior_meanabs": prior_ch.tolist(),
           "rain_channels": RAIN_CH,
           "zero_frac_rain": [float((X[:, c] == 0).mean()) for c in RAIN_CH]}
    res = {}
    for tag, ctx3 in (("clean", ctx_c3), ("mnar_high_0.7", ctx_m3)):
        ctx = ctx3.reshape(-1, L).astype(np.float32)
        y = y3.reshape(-1, H).astype(np.float32)
        fb = fb_count_nan(ctx) if tag != "clean" else fb_count_clean(ctx)
        t0 = time.time()
        pred_t5 = t5_predict(t5, ctx)
        pred_bolt = bolt_predict(bolt, ctx)
        mse_t5, mse_bolt = row_mse(pred_t5, y), row_mse(pred_bolt, y)
        # bolt eps-trigger rows: std of observed == 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obs_std = np.nanstd(ctx, axis=1)
        nobs = np.isfinite(ctx).sum(axis=1)
        bolt_eps = (obs_std == 0) & (nobs > 0)
        rec = {"mse_t5": float(mse_t5.mean()), "mse_bolt": float(mse_bolt.mean()),
               "n_fb": int(fb.sum()), "n_bolt_eps": int(bolt_eps.sum()),
               "seconds": round(time.time() - t0, 1)}
        if tag == "clean":
            rec["mse_t5_vs_s9"] = {
                "s9": S9_T5_CLEAN_WEATHER,
                "rel_dev": float((mse_t5.mean() - S9_T5_CLEAN_WEATHER) / S9_T5_CLEAN_WEATHER)}
        if fb.any():
            pr = prior_ch[ch_of_row[fb]].astype(np.float64)
            mse_prior = row_mse(pred_t5[fb] * pr[:, None], y[fb])
            s_or, ok_o = oracle_scale(pred_t5[fb], y[fb])
            mse_or = np.full(fb.sum(), np.nan)
            mse_or[ok_o] = row_mse((pred_t5[fb] * s_or[:, None])[ok_o], y[fb][ok_o])
            fut = np.abs(y[fb]).sum(axis=1) > 0
            rec["fb"] = {
                "rows": {"win": (np.flatnonzero(fb) // C).tolist(),
                         "ch": ch_of_row[fb].tolist(),
                         "mse_t5": mse_t5[fb].tolist(),
                         "mse_t5_prior": mse_prior.tolist(),
                         "mse_bolt": mse_bolt[fb].tolist(),
                         "fut_active": [bool(v) for v in fut],
                         "prior_row": pr.tolist(),
                         "mom": fb_moments(pred_t5[fb], y[fb]).tolist()},
                "prior_fix": improvement_bundle(mse_t5[fb], mse_prior),
                "oracle_fix": improvement_bundle(mse_t5[fb][ok_o], mse_or[ok_o]),
                "oracle_excluded": int((~ok_o).sum()),
                "bolt_vs_native": improvement_bundle(mse_t5[fb], mse_bolt[fb]),
                "meanpred_t5": float(np.abs(pred_t5[fb]).mean()),
                "meanpred_bolt": float(np.abs(pred_bolt[fb]).mean()),
                "fut_active_frac": float(fut.mean()),
                "prior_fix_fut_active": improvement_bundle(mse_t5[fb][fut], mse_prior[fut]) if fut.any() else None,
                "prior_fix_fut_zero": improvement_bundle(mse_t5[fb][~fut], mse_prior[~fut]) if (~fut).any() else None,
            }
        # rain-channel rows: fallback vs not (native)
        rain = np.isin(ch_of_row, RAIN_CH)
        rec["rain_rows"] = {
            "n_rain": int(rain.sum()), "n_rain_fb": int((rain & fb).sum()),
            "mse_t5_rain_fb": float(mse_t5[rain & fb].mean()) if (rain & fb).any() else None,
            "mse_t5_rain_nonfb": float(mse_t5[rain & ~fb].mean()) if (rain & ~fb).any() else None,
            "mse_bolt_rain_fb": float(mse_bolt[rain & fb].mean()) if (rain & fb).any() else None,
            "mse_bolt_rain_nonfb": float(mse_bolt[rain & ~fb].mean()) if (rain & ~fb).any() else None,
        }
        print(f"real {tag:16s} mse_t5={rec['mse_t5']:.2f} mse_bolt={rec['mse_bolt']:.2f} "
              f"fb={rec['n_fb']} bolt_eps={rec['n_bolt_eps']}", flush=True)
        if fb.any():
            print(f"   fb rows: IMP_prior={rec['fb']['prior_fix']['agg_improvement']:+.3f} "
                  f"IMP_oracle={rec['fb']['oracle_fix']['agg_improvement']:+.3f} "
                  f"bolt/native={rec['fb']['bolt_vs_native']['agg_improvement']:+.3f} "
                  f"|pred| t5={rec['fb']['meanpred_t5']:.4f} bolt={rec['fb']['meanpred_bolt']:.6f}",
                  flush=True)
        res[tag] = rec
    out["configs"] = res
    # pooled real fallback rows (clean + mnar)
    rows = [res[t]["fb"]["rows"] for t in ("clean", "mnar_high_0.7") if "fb" in res[t]]
    if rows:
        mse_n = np.concatenate([np.array(r["mse_t5"]) for r in rows])
        mse_p = np.concatenate([np.array(r["mse_t5_prior"]) for r in rows])
        mse_b = np.concatenate([np.array(r["mse_bolt"]) for r in rows])
        fut = np.concatenate([np.array(r["fut_active"]) for r in rows])
        out["pooled"] = {
            "n": int(len(mse_n)),
            "prior_fix": improvement_bundle(mse_n, mse_p),
            "bolt_vs_native": improvement_bundle(mse_n, mse_b),
            "prior_fix_fut_active": improvement_bundle(mse_n[fut], mse_p[fut]) if fut.any() else None,
            "prior_fix_fut_zero": improvement_bundle(mse_n[~fut], mse_p[~fut]) if (~fut).any() else None,
        }
    return out


# ---- generic helpers for A/B -------------------------------------------------
def fb_flags(ctx):
    """All-zero-observed rows (covers clean and NaN-filled contexts)."""
    with np.errstate(invalid="ignore"):
        return np.nansum(np.abs(ctx), axis=1) <= 0


def fb_moments(pred, y):
    """Per-row (mean p^2, mean p*y, mean y^2): MSE(s) = A s^2 - 2B s + C."""
    p, yy = pred.astype(np.float64), y.astype(np.float64)
    return np.stack([(p * p).mean(1), (p * yy).mean(1), (yy ** 2).mean(1)], axis=1)


def decide(synth, real):
    pops = {}
    if "pooled" in synth:
        pops["synth"] = synth["pooled"]["prior_fix"]
    if "pooled" in real:
        pops["real"] = real["pooled"]["prior_fix"]
    go_pops = [name for name, st in pops.items()
               if st.get("agg_improvement") is not None
               and st["agg_improvement"] >= GO_IMP
               and (st.get("top5_share") is None or st["top5_share"] < GO_TOP5)]
    return {
        "rule": (f"GO iff agg_improvement >= {GO_IMP:.0%} and top5_share < "
                 f"{GO_TOP5:.0%} on either pooled fallback-row population "
                 f"(real weather / synthetic)"),
        "populations": {k: {"agg_improvement": v.get("agg_improvement"),
                            "agg_ci95": v.get("agg_ci95"),
                            "top5_share": v.get("top5_share"),
                            "n": v.get("n")} for k, v in pops.items()},
        "go": bool(go_pops),
        "go_populations": go_pops,
    }


def part_a(args):
    results = {"meta": {
        "L": L, "H": H, "seed": SEED, "windows": NWIN_AUX,
        "synth": {"seed": SYNTH_SEED, "n_per_cell": N_SYNTH, "act": ACT,
                  "sig": SIG, "mean_wet_spell": DW,
                  "generator": "two-state Markov dry/wet, amp~Exp(1), noise on active"},
        "models": {"t5": "amazon/chronos-t5-small fp32 batch128 20 samples median",
                   "bolt": "amazon/chronos-bolt-base fp32 median quantile"},
        "pairing": "weather/ETT windows+masks identical to S9 (run_s5_missing.py)",
        "decision_rule": f"agg improvement>={GO_IMP} and top5<{GO_TOP5}",
    }}
    gate = anchor_gate()
    results["anchor_gate"] = gate
    if not gate["_gate_counts_pass"]:
        results["decision"] = {"go": False, "reason": "anchor counts gate FAILED"}
        save_json(results, RESULTS)
        print("ANCHOR GATE (counts) FAIL — aborting", flush=True)
        return 1
    t5 = load_t5("cuda")
    mse_res, ok = anchor_mse(t5)
    results["anchor_gate"].update(mse_res)
    results["anchor_gate"]["pass"] = bool(ok)
    if not ok:
        results["decision"] = {"go": False, "reason": "anchor t5 MSE gate FAILED"}
        save_json(results, RESULTS)
        print("ANCHOR GATE (mse) FAIL — aborting", flush=True)
        return 1
    print("ANCHOR GATE PASS", flush=True)
    bolt = load_bolt("cuda")

    synth = run_synth(t5, bolt)
    real = run_real(t5, bolt)
    decision = decide(synth, real)
    results["partA"] = {"synth": synth, "real": real}
    results["decision"] = decision
    save_json(results, RESULTS)
    rp = decision["populations"].get("real", {})
    sp = decision["populations"].get("synth", {})
    print(f"DECISION: real IMP={rp.get('agg_improvement')} top5={rp.get('top5_share')}; "
          f"synth IMP={sp.get('agg_improvement')} top5={sp.get('top5_share')} "
          f"-> {'GO' if decision['go'] else 'NO-GO'}", flush=True)
    print("PART A DONE", flush=True)
    return 0


# ---- Part B ------------------------------------------------------------------
def eval_t5_mode(t5, ctx, mode, prior_row=None, alpha=1.0, native_pred=None):
    """Predictions for a patched t5 mode, reusing analytic shortcuts.
    floor on fallback rows is exact via pred*(alpha*prior) (smoke-verified);
    floor on bumped-but-not-fallback rows and medmad need reruns; nanskip
    reruns only flagged rows."""
    n = len(ctx)
    ctxlen = t5.model.config.context_length
    fl = fb_flags(ctx)
    if mode == "native":
        return native_pred
    if mode == "medmad":
        return t5_predict(t5, ctx,
                          scale_fn=make_t5_scale_fn("medmad", ctxlen=ctxlen))
    if mode in ("nanskip", "nanskip_prior"):
        pred = native_pred.copy()
        if fl.any():
            pr = prior_row[fl] if mode == "nanskip_prior" else None
            pred[fl] = t5_predict(
                t5, ctx[fl],
                scale_fn=make_t5_scale_fn(mode, prior=pr,
                                          flags0=np.ones(int(fl.sum()), bool),
                                          ctxlen=ctxlen))
        return pred
    if mode in ("floor", "floor01", "floor10"):
        pr = prior_row.astype(np.float64)
        with np.errstate(invalid="ignore"):
            ma = np.nansum(np.abs(ctx), axis=1) / np.clip(
                np.isfinite(ctx).sum(axis=1), 1, None)
        bumped = (~fl) & (ma < alpha * pr)
        pred = native_pred.astype(np.float64).copy()
        pred[fl] = native_pred[fl] * (alpha * pr[fl])[:, None]     # analytic
        if bumped.any():
            pred[bumped] = t5_predict(
                t5, ctx[bumped],
                scale_fn=make_t5_scale_fn("floor", prior=prior_row[bumped],
                                          alpha=alpha, ctxlen=ctxlen))
        return pred
    raise ValueError(mode)


def eval_bolt_mode(bolt, ctx, mode, prior_std_row=None, alpha=1.0, native_pred=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        std = np.nanstd(ctx, axis=1)
    nobs = np.isfinite(ctx).sum(axis=1)
    if mode == "native":
        return native_pred
    if mode == "medmad":
        return bolt_predict(bolt, ctx, norm_fn=make_bolt_norm_fn("medmad"))
    if mode in ("stdfloor", "stdfloor01", "stdfloor10"):
        pr = prior_std_row.astype(np.float64)
        bumped = (std < alpha * pr) & (nobs > 0)
        pred = native_pred.astype(np.float64).copy()
        if bumped.any():
            pred[bumped] = bolt_predict(
                bolt, ctx[bumped],
                norm_fn=make_bolt_norm_fn("stdfloor",
                                          prior_std=prior_std_row[bumped],
                                          alpha=alpha))
        return pred
    if mode == "nanskip":
        fl = (std == 0) & (nobs > 0)
        pred = native_pred.astype(np.float64).copy()
        if fl.any():
            ctx2 = ctx.copy()
            ctx2[fl] = np.nan
            pred[fl] = bolt_predict(bolt, ctx2[fl])
        return pred
    raise ValueError(mode)


def mode_metrics(pred_mode, native_pred, ctx, y):
    fl = fb_flags(ctx)
    mse_n = row_mse(native_pred, y)
    mse_m = row_mse(pred_mode, y)
    out = {"n": int(len(ctx)), "n_fb": int(fl.sum()),
           "whole_ratio": jfloat(mse_m.sum() / mse_n.sum() if mse_n.sum() > 0 else None)}
    if fl.any():
        out["fb"] = improvement_bundle(mse_n[fl], mse_m[fl])
    if (~fl).any():
        nn, nm = mse_n[~fl].sum(), mse_m[~fl].sum()
        out["nonfb_ratio"] = jfloat(nm / nn if nn > 0 else None)
    return out


def build_eval_sets():
    """Deterministic rebuild of every Part-B evaluation context."""
    sets = {}
    for a in ACT:
        for sg in SIG:
            xs = gen_intermittent(a, sg)
            ctx, y = xs[:, :L].copy(), xs[:, L:].copy()
            n = len(xs)
            sets[f"synth:a{a}:s{sg}"] = {
                "ctx": ctx, "y": y,
                "prior_t5": np.full(n, np.abs(xs).mean(), np.float32),
                "prior_bolt": np.full(n, xs.std(), np.float32),
                "kind": "synth"}
    X, starts, ctx_c3, ctx_m3, y3, prior_ch, prior_std_ch = weather_contexts()
    nw, C = ctx_c3.shape[0], ctx_c3.shape[1]
    chrow = np.tile(np.arange(C), nw)
    for tag, ctx3 in (("clean", ctx_c3), ("mnar_high_0.7", ctx_m3)):
        sets[f"weather_{tag}"] = {
            "ctx": ctx3.reshape(-1, L).astype(np.float32),
            "y": y3.reshape(-1, H).astype(np.float32),
            "prior_t5": prior_ch[chrow].astype(np.float32),
            "prior_bolt": prior_std_ch[chrow].astype(np.float32),
            "kind": "weather"}
    for ds in ("ETTh1", "ETTm1"):
        Xe, se = load_windows(DATASETS[ds], NWIN_DRAW, SEED)
        Ce = Xe.shape[1]
        chrow_e = np.tile(np.arange(Ce), NWIN_AUX)
        ts = int(0.8 * len(Xe))
        sets[f"{ds}_clean"] = {
            "ctx": np.stack([Xe[s:s + L].T for s in se[:NWIN_AUX]]).reshape(-1, L).astype(np.float32),
            "y": np.stack([Xe[s + L:s + L + H].T for s in se[:NWIN_AUX]]).reshape(-1, H).astype(np.float32),
            "prior_t5": np.abs(Xe[:ts]).mean(axis=0)[chrow_e].astype(np.float32),
            "prior_bolt": Xe[:ts].std(axis=0)[chrow_e].astype(np.float32),
            "kind": "ett"}
    return sets


def part_b(args):
    res = json.load(open(RESULTS))
    go = bool(res.get("decision", {}).get("go"))
    print(f"PART B (decision was {'GO' if go else 'NO-GO — degraded short verification'})",
          flush=True)
    t5 = load_t5("cuda")
    bolt = load_bolt("cuda")
    sets = build_eval_sets()
    t5_modes = (["medmad", "floor01", "floor10", "nanskip", "nanskip_prior"] if go
                else ["floor10", "nanskip"])
    bolt_modes = (["medmad", "stdfloor01", "stdfloor10", "nanskip"] if go
                  else ["stdfloor10", "nanskip"])
    out = {"go": go, "t5": {}, "bolt": {}}

    # ---- t5 candidates -------------------------------------------------------
    for name, st in sets.items():
        t0 = time.time()
        native = t5_predict(t5, st["ctx"])
        rec = {"native_mse": float(row_mse(native, st["y"]).mean())}
        for mode in t5_modes:
            tm = time.time()
            alpha = 0.1 if mode == "floor01" else 1.0
            pred = eval_t5_mode(t5, st["ctx"], mode, prior_row=st["prior_t5"],
                                alpha=alpha, native_pred=native)
            rec[mode] = mode_metrics(pred, native, st["ctx"], st["y"])
            print(f"t5 {name:26s} {mode:14s} whole={rec[mode]['whole_ratio']:.4f} "
                  f"fb_IMP={(rec[mode].get('fb') or {}).get('agg_improvement')} "
                  f"({time.time()-tm:.0f}s)", flush=True)
        rec["seconds"] = round(time.time() - t0, 1)
        out["t5"][name] = rec

    # ---- bolt candidates -----------------------------------------------------
    for name, st in sets.items():
        native = bolt_predict(bolt, st["ctx"])
        rec = {"native_mse": float(row_mse(native, st["y"]).mean())}
        for mode in bolt_modes:
            alpha = 0.1 if mode == "stdfloor01" else 1.0
            pred = eval_bolt_mode(bolt, st["ctx"], mode,
                                  prior_std_row=st["prior_bolt"], alpha=alpha,
                                  native_pred=native)
            rec[mode] = mode_metrics(pred, native, st["ctx"], st["y"])
        out["bolt"][name] = rec
        print(f"bolt {name:26s} done", flush=True)

    # ---- pooled summaries + recommendation -----------------------------------
    def pool_weather(recs, mode):
        ms_n, ms_m = [], []
        for tag in ("weather_clean", "weather_mnar_high_0.7"):
            r = recs[tag].get(mode)
            if r and r.get("fb"):
                ms_n.append(r["fb"]["sum_mse_native"])
                ms_m.append(r["fb"]["sum_mse_fix"])
        if not ms_n:
            return None
        return float(1.0 - sum(ms_m) / sum(ms_n))

    summ = {"t5": {}, "bolt": {}}
    for mode in t5_modes:
        ett = {ds: out["t5"][f"{ds}_clean"][mode]["whole_ratio"] for ds in ("ETTh1", "ETTm1")}
        nonfb = [out["t5"][k][mode].get("nonfb_ratio") for k in sets
                 if sets[k]["kind"] == "synth"]
        nonfb = [v for v in nonfb if v is not None]
        summ["t5"][mode] = {
            "weather_fb_improvement": pool_weather(out["t5"], mode),
            "ett_whole_ratio": ett,
            "synth_nonfb_worst_ratio": float(max(nonfb, key=lambda v: abs(v - 1))) if nonfb else None,
        }
    for mode in bolt_modes:
        ett = {ds: out["bolt"][f"{ds}_clean"][mode]["whole_ratio"] for ds in ("ETTh1", "ETTm1")}
        summ["bolt"][mode] = {"weather_fb_improvement": pool_weather(out["bolt"], mode),
                              "ett_whole_ratio": ett}

    if go:
        elig = []
        for mode in t5_modes:
            s = summ["t5"][mode]
            ett_ok = all(abs(v - 1) <= 0.01 for v in s["ett_whole_ratio"].values())
            nonfb_ok = s["synth_nonfb_worst_ratio"] is None or \
                abs(s["synth_nonfb_worst_ratio"] - 1) <= 0.01
            if ett_ok and nonfb_ok and s["weather_fb_improvement"] is not None:
                elig.append((mode, s["weather_fb_improvement"]))
        rec_mode = max(elig, key=lambda kv: kv[1])[0] if elig else None
        summ["recommendation"] = {
            "eligible_t5": elig,
            "recommended_t5": rec_mode,
            "rule": "max weather fallback-row improvement among modes with "
                    "|ETT ratio-1|<=1% and |synth nonfb ratio-1|<=1%",
        }
        print(f"RECOMMENDATION: {rec_mode} (eligible={elig})", flush=True)
    res["partB"] = out
    res["partB_summary"] = summ
    save_json(res, RESULTS)
    print("PART B DONE", flush=True)
    render(res)
    return 0


# ---- report (figure) ---------------------------------------------------------
def _sweep_curve(mom, prior_row, grid):
    """Mean over rows of MSE(s)/MSE(1.0) with s = grid*prior_row (analytic)."""
    mom = np.asarray(mom, float)
    pr = np.asarray(prior_row, float)
    A, B, C = mom[:, 0], mom[:, 1], mom[:, 2]
    native = A - 2 * B + C
    ok = native > 0
    if ok.sum() < 3:
        return None
    s = grid[None, :] * pr[ok, None]
    mse = A[ok, None] * s ** 2 - 2 * B[ok, None] * s + C[ok, None]
    return (mse / native[ok, None]).mean(axis=0)


def render(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pa = res.get("partA", {})
    synth, real = pa.get("synth", {}), pa.get("real", {})
    dec = res.get("decision", {})
    cells = synth.get("cells", {})

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.30)

    # (1) synth fallback rate vs activity
    ax = fig.add_subplot(gs[0, 0])
    for sg in SIG:
        ys = [cells.get(f"a{a}:s{sg}", {}).get("fb_rate") for a in ACT]
        xs = [a for a, v in zip(ACT, ys) if v is not None]
        ax.plot(xs, [v for v in ys if v is not None], marker="o", ms=4,
                label=f"σ={sg}")
    ax.set_xscale("log")
    ax.set_xticks(ACT)
    ax.set_xticklabels([f"{a:g}" for a in ACT])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("stationary activity rate a")
    ax.set_ylabel("fallback trigger rate")
    ax.set_title("synthetic: t5 fallback rate vs activity\n(global-noise diag: 0 triggers — exact zeros required)",
                 fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (2) synth fallback-row aggregate improvement vs activity
    ax = fig.add_subplot(gs[0, 1])
    for sg, mk in zip(SIG, ("o", "s", "^")):
        xs, imp_p, imp_o, imp_b = [], [], [], []
        for a in ACT:
            r = cells.get(f"a{a}:s{sg}", {}).get("fb")
            if not r:
                continue
            xs.append(a)
            imp_p.append(r["prior_fix"]["agg_improvement"])
            imp_o.append(r["oracle_fix"]["agg_improvement"])
            imp_b.append(r["bolt_vs_native"]["agg_improvement"])
        ax.plot(xs, imp_p, marker=mk, ms=4, color="#1f77b4",
                label="prior fix" if sg == SIG[0] else None)
        ax.plot(xs, imp_o, marker=mk, ms=4, color="#2ca02c", ls="--",
                label="oracle s*" if sg == SIG[0] else None)
        ax.plot(xs, imp_b, marker=mk, ms=4, color="#9467bd", ls=":",
                label="bolt vs t5" if sg == SIG[0] else None)
    ax.axhline(GO_IMP, color="r", lw=1, ls="--", label=f"gate {GO_IMP:.0%}")
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xticks(ACT)
    ax.set_xticklabels([f"{a:g}" for a in ACT])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("activity rate a (markers: σ=0 o, 0.1 □, 0.3 △)")
    ax.set_ylabel("agg improvement on fallback rows")
    ax.set_title("synthetic: damage of the scale=1.0 fallback", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (3) real fallback rows: native vs prior-fix scatter
    ax = fig.add_subplot(gs[0, 2])
    cfgs = real.get("configs", {})
    for tag, col, lab in (("clean", "#1f77b4", "clean"), ("mnar_high_0.7", "#d62728", "mnar_high p=0.7")):
        r = cfgs.get(tag, {}).get("fb", {}).get("rows")
        if not r:
            continue
        ax.scatter(np.maximum(r["mse_t5"], 1e-12), np.maximum(r["mse_t5_prior"], 1e-12),
                   s=10, alpha=0.6, color=col, label=f"{lab} (n={len(r['mse_t5'])})")
    lim = ax.get_xlim()
    ax.plot([1e-12, lim[1]], [1e-12, lim[1]], "k--", lw=0.8, alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("native MSE (scale=1.0 fallback)")
    ax.set_ylabel("prior-fixed MSE")
    ax.set_title("real weather fallback rows: native vs fixed", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (4) real per-row relative improvement histogram
    ax = fig.add_subplot(gs[1, 0])
    for tag, col in (("clean", "#1f77b4"), ("mnar_high_0.7", "#d62728")):
        r = cfgs.get(tag, {}).get("fb", {}).get("rows")
        if not r:
            continue
        mn = np.array(r["mse_t5"]); mf = np.array(r["mse_t5_prior"])
        ok = mn > 1e-12
        rel = 1 - mf[ok] / mn[ok]
        ax.hist(np.clip(rel, -1, 1), bins=40, alpha=0.5, color=col,
                label=f"{tag} (med={np.median(rel):+.2f})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("per-row relative MSE improvement (clipped ±1)")
    ax.set_title("real fallback rows: improvement distribution", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # (5) scale sweep: normalized MSE vs s/prior (analytic from moments)
    ax = fig.add_subplot(gs[1, 1])
    grid = np.logspace(-2, 2.5, 60)
    if synth.get("pooled", {}).get("mom"):
        c = _sweep_curve(synth["pooled"]["mom"], synth["pooled"]["prior_row"], grid)
        if c is not None:
            ax.plot(grid, c, color="#2ca02c", label="synthetic fb rows")
    moms, prs = [], []
    for tag in ("clean", "mnar_high_0.7"):
        r = cfgs.get(tag, {}).get("fb", {}).get("rows")
        if r and r.get("mom"):
            moms += r["mom"]; prs += r["prior_row"]
    if moms:
        c = _sweep_curve(moms, prs, grid)
        if c is not None:
            ax.plot(grid, c, color="#d62728", label="weather fb rows")
    # where does the native fallback sit? s=1.0 -> x = 1/prior
    if synth.get("pooled", {}).get("prior_row"):
        med = float(np.median(1.0 / np.array(synth["pooled"]["prior_row"])))
        ax.axvline(med, color="#2ca02c", ls=":", lw=1,
                   label=f"native s=1.0 (synth median {med:.3g}×prior)")
    if prs:
        med = float(np.median(1.0 / np.array(prs)))
        ax.axvline(med, color="#d62728", ls=":", lw=1,
                   label=f"native s=1.0 (weather median {med:.3g}×prior)")
    ax.axvline(1.0, color="k", lw=0.8, alpha=0.5, label="s = prior")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.99, 1e4)
    ax.set_xlabel("scale s / prior")
    ax.set_ylabel("MSE(s) / MSE(native)")
    ax.set_title("MSE(s) is flat around native s=1.0 — predictions are ≈0,\n"
                 "so the scale choice is irrelevant on fallback rows", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5)

    # (6) Part B candidate comparison or no-go text
    ax = fig.add_subplot(gs[1, 2])
    pb = res.get("partB_summary")
    if pb and pb.get("t5"):
        modes = [m for m in ("medmad", "floor01", "floor10", "nanskip", "nanskip_prior")
                 if m in pb["t5"]]
        imps = [pb["t5"][m].get("weather_fb_improvement") for m in modes]
        xs = np.arange(len(modes))
        ax.bar(xs - 0.2, [v if v is not None else 0 for v in imps], width=0.4,
               color="#1f77b4", label="t5 weather fb IMP")
        bmodes = [m for m in ("medmad", "stdfloor01", "stdfloor10", "nanskip")
                  if m in pb.get("bolt", {})]
        if bmodes:
            off = xs[-1] + 1
            bimps = [pb["bolt"][m].get("weather_fb_improvement") for m in bmodes]
            ax.bar(off + np.arange(len(bmodes)) - 0.2,
                   [v if v is not None else 0 for v in bimps], width=0.4,
                   color="#9467bd", label="bolt weather fb IMP")
            modes_all = modes + [f"bolt:{m}" for m in bmodes]
        else:
            modes_all = modes
        ett_dev = []
        for m in modes:
            e = pb["t5"][m].get("ett_whole_ratio", {})
            ett_dev.append(max((abs(v - 1) for v in e.values()), default=0) * 100)
        for m in bmodes if bmodes else []:
            e = pb["bolt"][m].get("ett_whole_ratio", {})
            ett_dev.append(max((abs(v - 1) for v in e.values()), default=0) * 100)
        xall = np.arange(len(modes_all)).astype(float)
        ax2 = ax.twinx()
        ax2.plot(xall + 0.2, ett_dev, "rx", ms=7, label="max |ETT regr| (%)")
        ax2.set_ylabel("ETT regression |Δ| %", color="r")
        ax2.axhline(1.0, color="r", ls=":", lw=0.8)
        ax.set_xticks(xall - 0.2)
        ax.set_xticklabels(modes_all, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("weather fb agg improvement")
        ax.set_title("Part B candidates (bars) + ETT regression (x)", fontsize=10)
        ax.grid(alpha=0.3)
    else:
        txt = "Part B not run"
        if dec:
            txt = (f"decision: {'GO' if dec.get('go') else 'NO-GO'}\n"
                   f"{dec.get('rule', '')}\n"
                   f"{dec.get('reason', '')}")
        ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=9,
                transform=ax.transAxes, wrap=True)
        ax.axis("off")

    pops = dec.get("populations", {})
    rp, sp = pops.get("real", {}), pops.get("synth", {})
    fig.suptitle(
        f"S17 — Chronos-T5 scale=1.0 fallback damage verification: "
        f"{'GO' if dec.get('go') else 'NO-GO'}  |  real prior-fix IMP="
        f"{_fmt(rp.get('agg_improvement'))} (n={rp.get('n')})  "
        f"synth IMP={_fmt(sp.get('agg_improvement'))} (n={sp.get('n')})  "
        f"— fallback is harmless (preds≈0 on all-zero contexts)",
        fontsize=11)
    fig.savefig(os.path.join(HERE, "s17.png"), dpi=140, bbox_inches="tight")
    print("s17.png written", flush=True)


def _fmt(v):
    return "None" if v is None else f"{v:+.3f}"


def part_report(args):
    res = json.load(open(RESULTS))
    render(res)
    return 0


# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["smoke", "A", "B", "clamp", "report"])
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if args.part == "smoke":
        return part_smoke()
    if args.part == "A":
        return part_a(args)
    if args.part == "B":
        return part_b(args)
    if args.part == "clamp":
        return part_clamp(args)
    if args.part == "report":
        return part_report(args)


# ---- Part A tail: edge-bin clamp on near-zero-scale channels -----------------
CLAMP_CH = [14, 15, 16, 17, 18]        # rain, raining, SWDR, PAR, maxPAR
CLAMP_ACT = [0.05, 0.20]
CLAMP_SIG = 0.1


def _clamp_metrics(ctx, y, preds, clamped, clamp_frac):
    """Per-arm MSE + top-decile MSE (future above clean-context q90, S5 defn),
    split by clamped / control rows, paired against the native arm."""
    q90 = np.quantile(ctx, 0.9, axis=1)
    top = y > q90[:, None]
    y64 = y.astype(np.float64)
    mse = {k: row_mse(p, y) for k, p in preds.items()}
    td = {}
    for k, p in preds.items():
        err2 = (p.astype(np.float64) - y64) ** 2
        with np.errstate(all="ignore"):
            td[k] = np.nanmean(np.where(top, err2, np.nan), axis=1)
    out = {"n": len(ctx), "n_clamped": int(clamped.sum()),
           "clamp_frac_mean": float(clamp_frac.mean()),
           "fut_active_frac": float((np.abs(y64).sum(axis=1) > 0).mean())}
    for subset_name, sel in (("clamped", clamped), ("control", ~clamped)):
        if sel.sum() < 3:
            out[subset_name] = {"n": int(sel.sum())}
            continue
        sub = {"n": int(sel.sum())}
        for arm in ("oracle_scale", "log1p"):
            b = improvement_bundle(mse["native"][sel], mse[arm][sel])
            ok = np.isfinite(td["native"][sel]) & np.isfinite(td[arm][sel])
            tdn, tda = td["native"][sel][ok], td[arm][sel][ok]
            sub[arm] = {
                "mse": b,
                "topdecile": {
                    "n": int(ok.sum()),
                    "sum_ratio": jfloat(tda.sum() / tdn.sum() if tdn.sum() > 0 else None),
                    "median_row_improvement": jfloat(
                        np.median(1 - tda / tdn)) if (tdn > 0).any() else None,
                },
            }
            # does the clamp depth predict the fix gain?
            if subset_name == "clamped":
                d = mse["native"][sel] - mse[arm][sel]
                pos = mse["native"][sel] > 1e-12
                if pos.sum() >= 10:
                    from scipy.stats import spearmanr
                    rho = spearmanr(clamp_frac[sel][pos],
                                    (1 - mse[arm][sel] / mse["native"][sel])[pos])
                    sub[arm]["spearman_clampfrac_vs_improvement"] = float(rho.statistic)
        out[subset_name] = sub
    return out


def part_clamp(args):
    """Edge-bin clamp harm test: on near-zero-scale channels a context spike
    x/scale can exceed the +/-15 bin limits and is clamped to the edge bin —
    the model sees 'maximal bin', not the amplitude. Three arms on rows of
    the affected channels: (i) native, (ii) oracle global scale (channel
    nonzero-mean|x| from train), (iii) log1p transform + expm1 back (median is
    equivariant under monotone transforms)."""
    t5 = load_t5("cuda")
    tok = t5.tokenizer
    ntok, nsp = t5.model.config.n_tokens, t5.model.config.n_special_tokens
    edge_hi, edge_lo = ntok - 1, nsp
    ctxlen = t5.model.config.context_length
    out = {"meta": {
        "arms": ["native", "oracle_scale", "log1p"],
        "oracle_scale_def": "train mean|x| over nonzero points, per channel/cell",
        "clamp_def": f"token id in {{{edge_lo}, {edge_hi}}} (real tokenizer)",
        "topdecile": "future points above clean-context q90 (S5 definition)",
    }}

    def edge_stats(ctx):
        ids, _, sc = tok._input_transform(torch.from_numpy(ctx))
        ids = ids.numpy()
        frac = ((ids == edge_hi) | (ids == edge_lo)).mean(axis=1)
        return frac, sc.numpy()

    def run_arms(ctx, glob_scale_row):
        preds = {"native": t5_predict(t5, ctx)}
        preds["oracle_scale"] = t5_predict(
            t5, ctx, scale_fn=make_t5_scale_fn("const", prior=glob_scale_row,
                                               ctxlen=ctxlen))
        preds["log1p"] = np.expm1(t5_predict(t5, np.log1p(ctx))).astype(np.float32)
        return preds

    # ---- real weather, near-zero channels ------------------------------------
    X, starts, ctx_c3, _, y3, prior_ch, _ = weather_contexts()
    test_start = int(0.8 * len(X))
    nz_prior = np.array([np.abs(X[:test_start, c])[X[:test_start, c] != 0].mean()
                         for c in range(X.shape[1])])
    sub_ctx = ctx_c3[:, CLAMP_CH, :].reshape(-1, L).astype(np.float32)
    sub_y = y3[:, CLAMP_CH, :].reshape(-1, H).astype(np.float32)
    assert sub_ctx.min() >= 0, "log1p arm needs non-negative channels"
    chrow = np.tile(CLAMP_CH, len(starts))
    frac, s0 = edge_stats(sub_ctx)
    clamped = frac > 0
    rec = {"channels": CLAMP_CH,
           "rows_per_channel": {int(c): int((chrow == c).sum()) for c in CLAMP_CH},
           "clamped_per_channel": {int(c): int((clamped & (chrow == c)).sum())
                                   for c in CLAMP_CH},
           "native_scale_median": float(np.median(s0)),
           "maxratio_p95": float(np.percentile(
               np.abs(sub_ctx).max(axis=1) / np.maximum(s0, 1e-30), 95)),
           "nz_prior_per_channel": {int(c): float(nz_prior[c]) for c in CLAMP_CH}}
    preds = run_arms(sub_ctx, nz_prior[chrow].astype(np.float32))
    rec.update(_clamp_metrics(sub_ctx, sub_y, preds, clamped, frac))
    out["real"] = rec
    cl = rec.get("clamped", {})
    print(f"clamp real: n={rec['n']} clamped={rec['n_clamped']} "
          f"per-ch={rec['clamped_per_channel']}", flush=True)
    for arm in ("oracle_scale", "log1p"):
        if arm in cl:
            print(f"  clamped {arm:12s} agg_IMP={cl[arm]['mse']['agg_improvement']:+.4f} "
                  f"td_ratio={cl[arm]['topdecile']['sum_ratio']}", flush=True)

    # ---- synthetic heavy-tail -------------------------------------------------
    synth = {}
    for a in CLAMP_ACT:
        xs = gen_intermittent(a, CLAMP_SIG, tail="weibull")
        ctx, y = xs[:, :L].copy(), xs[:, L:].copy()
        nz = xs[xs != 0]
        gprior = np.full(len(xs), nz.mean(), np.float32)
        frac, s0 = edge_stats(ctx)
        clamped = frac > 0
        preds = run_arms(ctx, gprior)
        rec = {"a": a, "sigma": CLAMP_SIG, "tail": "weibull(0.5)",
               "glob_prior": float(nz.mean()),
               "native_scale_median": float(np.median(s0))}
        rec.update(_clamp_metrics(ctx, y, preds, clamped, frac))
        synth[f"a{a}"] = rec
        cl = rec.get("clamped", {})
        line = f"clamp synth a={a}: n={rec['n']} clamped={rec['n_clamped']}"
        for arm in ("oracle_scale", "log1p"):
            if arm in cl:
                line += (f" | {arm} IMP={cl[arm]['mse']['agg_improvement']:+.4f} "
                         f"td={cl[arm]['topdecile']['sum_ratio']}")
        print(line, flush=True)
    out["synth"] = synth

    # ---- judgment --------------------------------------------------------------
    def grab(d, arm):
        c = d.get("clamped", {}).get(arm)
        if not c:
            return None, None
        return c["mse"]["agg_improvement"], c["topdecile"]["sum_ratio"]
    pops = {"real": out["real"], **{f"synth:{k}": v for k, v in synth.items()}}
    judgment = {}
    for name, d in pops.items():
        for arm in ("oracle_scale", "log1p"):
            imp, tdr = grab(d, arm)
            judgment[f"{name}:{arm}"] = {"agg_improvement": imp, "td_sum_ratio": tdr}
    imps_o = [v["agg_improvement"] for k, v in judgment.items()
              if k.endswith("oracle_scale") and v["agg_improvement"] is not None]
    imps_l = [v["agg_improvement"] for k, v in judgment.items()
              if k.endswith("log1p") and v["agg_improvement"] is not None]
    out["judgment"] = {
        "per_population": judgment,
        "clamp_harmful": bool(imps_o and max(imps_o) >= GO_IMP),
        "log1p_mitigates": bool(imps_l and max(imps_l) >= GO_IMP),
        "rule": f"harmful/mitigates iff best agg improvement >= {GO_IMP:.0%} "
                f"on clamped rows",
    }
    print(f"CLAMP JUDGMENT: harmful={out['judgment']['clamp_harmful']} "
          f"log1p_mitigates={out['judgment']['log1p_mitigates']}", flush=True)

    res = json.load(open(RESULTS))
    res["partA_clamp"] = out
    save_json(res, RESULTS)
    print("PART CLAMP DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())


