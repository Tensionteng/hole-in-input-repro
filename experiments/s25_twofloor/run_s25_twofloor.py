#!/usr/bin/env python
"""S25: two-floor decomposition -- information floor vs architectural floor.

Every prior repair round reports a number for a repair; none separate WHY a repair
fails. This round decomposes the degradation of a frozen TSFM under missing data:

    R(f, g0) - R_clean = [R_stat - R_clean] + [R*(f) - R_stat] + [R(f,g0) - R*(f)]
                          (1) information     (2) architecture   (3) fill/routing

(1) is model-independent (the information the mechanism destroyed); estimated by the best
    direct predictor Z=(x_obs, mask) -> Y trained in-domain, normalised by its OWN clean
    baseline.
(2) is model-specific: information that survives in Z but that a FROZEN f cannot be made
    to use, because the only lever is the input fill. R*(f) = inf_g E[loss(f(g(Z)), Y)];
    estimated from above by an amortised fill network trained end-to-end through frozen f.
(3) is the only term a detector/router can touch.

Part 0 additionally tests an exact architectural claim read off chronos_bolt.py:277-300 --
`instance_norm` runs BEFORE the mask is applied and excludes only NaN, so

  (a) NaN input, no mask      -> forecast exactly invariant to the missing values, rank 0
  (b) filled input + mask     -> all influence factors through (loc, scale), rank <= 2
  (c) filled input, no mask   -> full rank; fabricated content enters the content channel

i.e. "geometry x input convention" becomes a statement about the rank of the
fill-reachable set.

Conventions: L=512; H=64 (bolt's native single-block horizon -- S5's H=96 triggers an
autoregressive rollout with a 9x quantile-mixing heuristic that would contaminate a
gradient measurement; the anchor gate runs at H=96 to prove the harness reproduces S5).
Windows and masks are byte-identical to S5 (SEED=20250810, load_windows k=300 on the last
20%, make_mask copied verbatim). Learned components train on the first 80% only.
All relMSE are own-clean.
"""
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

L = 512
H = 64                 # bolt native prediction length; single forward pass
H_GATE = 96            # S5's horizon, used only by the anchor gate
BLOCK = 24
SEED = 20250810
RATES_EVAL = (0.3, 0.7)
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
RESULTS_PATH = os.path.join(HERE, "s25_results.json")

DATA_ROOT = os.path.join(ROOT, "legacy_nonstationary", "tslib", "dataset")
DATASETS = {
    "ETTh1": os.path.join(DATA_ROOT, "ETT-small", "ETTh1.csv"),
    "ETTm1": os.path.join(DATA_ROOT, "ETT-small", "ETTm1.csv"),
    "weather": os.path.join(DATA_ROOT, "weather", "weather.csv"),
}

# stored S5 numbers the anchor gate must reproduce within +-5% (bolt, ETTh1, H=96, 300 win)
ANCHORS = {
    "clean:none:0.0": 11.199030024288856,
    "mnar_high:linear:0.5": 16.050924603797416,
    "mcar:zero:0.3": 12.359439230779998,
}


# ------------------------------------------------------- S5 helpers (verbatim) ----

def load_windows(path, n_windows, seed, horizon):
    df = pd.read_csv(path)
    X = df.drop(columns=["date"]).to_numpy(np.float32)
    N = len(X)
    test_start = int(0.8 * N)
    lo, hi = test_start - L, N - L - horizon
    n_valid = hi - lo + 1
    rng = np.random.default_rng(seed)
    k = min(n_windows, n_valid)
    starts = np.sort(rng.choice(n_valid, size=k, replace=False)) + lo
    return X, starts


def load_train_windows(path, n_windows, horizon, seed=SEED + 1):
    """Window starts drawn from the FIRST 80% only, disjoint from every eval window."""
    df = pd.read_csv(path)
    X = df.drop(columns=["date"]).to_numpy(np.float32)
    N = len(X)
    hi = int(0.8 * N) - L - horizon           # last train start
    rng = np.random.default_rng(seed)
    k = min(n_windows, hi + 1)
    starts = np.sort(rng.choice(hi + 1, size=k, replace=False))
    return X, starts


def make_mask(mech, rate, window_idx, mask_seed, C, x=None):
    """[C, L] bool. Copied verbatim from run_s5_missing.py so masks are identical."""
    if mech in ("mnar_high", "mnar_extreme"):
        assert x is not None, "mnar masks need the clean context"
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
    if fill in ("oracle", "none"):
        return x.copy()
    if fill == "zero":
        out = x.copy()
        out[mask] = 0.0
        return out
    if fill == "nan":
        out = x.copy()
        out[mask] = np.nan
        return out
    out = x.copy()
    t = np.arange(L)
    for c in range(len(x)):
        valid = np.flatnonzero(~mask[c])
        if len(valid) == 0:
            out[c] = 0.0
        elif fill == "ffill":
            pos = np.maximum(np.searchsorted(valid, t, side="right") - 1, 0)
            out[c] = x[c, valid[pos]]
        elif fill == "linear":
            out[c] = np.interp(t, valid, x[c, valid])
        else:
            raise ValueError(fill)
    return out


# ------------------------------------------------------------------- model ----

class Bolt:
    """chronos-bolt-base with a DIFFERENTIABLE single-block forecast head.

    `pipe.predict` wraps the model in no_grad and, for horizons beyond the native 64,
    performs an autoregressive rollout with a quantile-mixing heuristic. For every
    optimisation-based part we call the raw module once with H<=64 so the measured
    Jacobian is the model's, not the rollout heuristic's.
    """
    name, patch = "bolt", 16

    def __init__(self, device="cuda"):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
        self.device = device
        self.model = self.pipe.inner_model if hasattr(self.pipe, "inner_model") else self.pipe.model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        assert self.model.chronos_config.prediction_length >= H, "H exceeds native block"

    def median(self, ctx, mask=None):
        """ctx: [B, L] float32 tensor (may contain NaN). Differentiable in ctx.
        mask: optional [B, L] float/bool tensor, 1 = observed. Returns [B, H]."""
        out = self.model(context=ctx, mask=mask)
        return out.quantile_preds[:, 4, :H]

    @torch.no_grad()
    def median_np(self, ctx_np, mask_np=None, batch=512):
        outs = []
        for i in range(0, len(ctx_np), batch):
            xb = torch.from_numpy(ctx_np[i:i + batch]).to(self.device)
            mb = None
            if mask_np is not None:
                mb = torch.from_numpy(mask_np[i:i + batch]).to(self.device).float()
            outs.append(self.median(xb, mb).float().cpu().numpy())
        return np.concatenate(outs)

    @torch.no_grad()
    def median_rollout_np(self, ctx_np, horizon, batch=1024):
        """The stock pipeline path (used only by the anchor gate at H=96)."""
        outs = []
        for i in range(0, len(ctx_np), batch):
            xb = torch.from_numpy(ctx_np[i:i + batch]).to(self.device)
            q = self.pipe.predict(xb, prediction_length=horizon)
            outs.append(q[:, 4, :].float().cpu().numpy())
        return np.concatenate(outs)


# ------------------------------------------------------------------- nets ----

class TCN(nn.Module):
    """Shared backbone: dilated conv stack over [value, mask] -> per-step features."""

    def __init__(self, ch=64, n_layers=6):
        super().__init__()
        self.inp = nn.Conv1d(2, ch, 5, padding=2)
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            d = 2 ** i
            self.blocks.append(nn.Sequential(
                nn.Conv1d(ch, ch, 3, padding=d, dilation=d), nn.GELU(),
                nn.Conv1d(ch, ch, 1), nn.GELU()))
        self.ch = ch

    def forward(self, v, m):           # v, m: [B, L]
        h = self.inp(torch.stack([v, m], dim=1))
        for blk in self.blocks:
            h = h + blk(h)
        return h                        # [B, ch, L]


class FillNet(nn.Module):
    """g_theta: (standardised observed context, mask) -> values at the missing positions.

    The output is a BOUNDED correction around the linear interpolant: at most `radius`
    observed-sigma away from it. This defines the admissible fill class G explicitly, and
    it is a scientific choice, not just a numerical one. Part A showed that optimising over
    ALL fills is adversarial -- a per-window oracle fill drives the error to 0.003x the
    clean-context error, i.e. the model can be steered almost anywhere by choosing 30-70%
    of its input. An architectural floor defined against that set would measure
    steerability, not repairability, and amortising it diverges. Zero-init makes the net
    start exactly at the linear fill, so any measured gain is a gain over linear."""

    def __init__(self, ch=64, radius=3.0):
        super().__init__()
        self.body = TCN(ch)
        self.head = nn.Conv1d(ch, 1, 1)
        self.radius = radius
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, v, m):           # v standardised, m 1=observed
        corr = self.radius * torch.tanh(self.head(self.body(v, m)).squeeze(1))
        return v + (1 - m) * corr


class DirectNet(nn.Module):
    """q_phi: (standardised observed context, mask) -> H-step forecast, no TSFM in loop."""

    def __init__(self, ch=64):
        super().__init__()
        self.body = TCN(ch)
        self.head = nn.Sequential(nn.Linear(ch * 4, 256), nn.GELU(), nn.Linear(256, H))

    def forward(self, v, m):
        h = self.body(v, m)                                   # [B, ch, L]
        feat = torch.cat([h.mean(-1), h.max(-1).values, h[..., -1], h[..., -64:].mean(-1)], -1)
        return 10.0 * torch.tanh(self.head(feat) / 10.0)      # bounded in sigma units


# ------------------------------------------------------------- batch build ----

def build_eval_batch(X, starts, mech, rate, horizon, fill="linear", mask_seed=0):
    """Returns clean ctx, filled ctx, mask(True=missing), future -- all [S*C, .]."""
    C = X.shape[1]
    cl, fl, mk, gt = [], [], [], []
    for wi, s in enumerate(starts):
        x = X[s:s + L].T.copy()
        y = X[s + L:s + L + horizon].T.copy()
        m = (np.zeros((C, L), bool) if mech == "clean"
             else make_mask(mech, rate, wi, mask_seed, C, x=x))
        cl.append(x)
        fl.append(fill_context(x, m, "none" if mech == "clean" else fill))
        mk.append(m)
        gt.append(y)
    r = lambda a: np.concatenate(a).astype(np.float32)
    return r(cl), r(fl), np.concatenate(mk), r(gt)


def standardise(filled, obs):
    """Per-series standardisation by OBSERVED-only stats (deployable)."""
    n = obs.sum(1, keepdims=True).clip(1)
    mu = (filled * obs).sum(1, keepdims=True) / n
    sd = np.sqrt(((filled - mu) ** 2 * obs).sum(1, keepdims=True) / n) + 1e-6
    return mu, sd


# ---------------------------------------------------------------- part gate ----

def part_gate(res, device):
    bolt = Bolt(device)
    X, starts = load_windows(DATASETS["ETTh1"], 300, SEED, H_GATE)
    out = {}
    for key, stored in ANCHORS.items():
        mech, fill, rate = key.split(":")
        rate = float(rate)
        # S5 averaged 2 mask seeds for the stochastic mechanisms, 1 for clean/mnar_*
        seeds = (0,) if mech in ("clean", "mnar_high", "mnar_extreme") else (0, 1)
        mses = []
        for ms in seeds:
            _, filled, _, gt = build_eval_batch(X, starts, mech, rate, H_GATE,
                                                fill=fill, mask_seed=ms)
            pred = bolt.median_rollout_np(filled, H_GATE).astype(np.float64)
            mses.append(float(((pred - gt.astype(np.float64)) ** 2).mean()))
        mse = float(np.mean(mses))
        dev = abs(mse - stored) / stored
        out[key] = {"stored": stored, "rerun": mse, "rel_dev": dev, "pass": dev <= 0.05}
        print(f"GATE {key:24s} stored={stored:10.5f} rerun={mse:10.5f} dev={dev:+.4%} "
              f"{'PASS' if dev <= 0.05 else 'FAIL'}", flush=True)
    res["gate"] = out
    assert all(v["pass"] for v in out.values()), "anchor gate FAILED -- stop"
    return res


# ------------------------------------------------------------------ part 0 ----

def perm_invariance(bolt, filled, mask, n_perm=4, rng=None):
    """Decisive, gradient-free test of the reachable-set proposition.

    Permuting the fill values AMONG the missing positions leaves (nanmean, nanstd) of the
    full context unchanged (up to float summation order) while changing the fill CONTENT
    completely. So:
      - if the forecast depends on the fill only through (loc, scale)  -> output unchanged
      - if the fill enters the content channel                          -> output moves
    Reported as the RMS output change under permutation, normalised by the RMS output
    change under an independent redraw of the fill (same marginal, different moments).
    """
    rng = rng or np.random.default_rng(SEED)
    out = {}
    for conv in ("plain_fill", "fill_mask", "nan"):
        base_ctx = filled.copy()
        if conv == "nan":
            base_ctx[mask] = np.nan
        mk = (~mask).astype(np.float32) if conv == "fill_mask" else None
        y0 = bolt.median_np(base_ctx, mk)
        d_perm, d_redraw = [], []
        for _ in range(n_perm):
            cp, cr = filled.copy(), filled.copy()
            for i in range(len(filled)):
                idx = np.flatnonzero(mask[i])
                if len(idx) < 2:
                    continue
                cp[i, idx] = filled[i, idx][rng.permutation(len(idx))]
                obs = filled[i, ~mask[i]]
                cr[i, idx] = rng.choice(obs, size=len(idx)) if len(obs) else 0.0
            if conv == "nan":
                cp[mask] = np.nan
                cr[mask] = np.nan
            d_perm.append(np.sqrt(((bolt.median_np(cp, mk) - y0) ** 2).mean(1)))
            d_redraw.append(np.sqrt(((bolt.median_np(cr, mk) - y0) ** 2).mean(1)))
        scale = np.sqrt((y0 ** 2).mean(1)) + 1e-12
        dp = np.mean(d_perm, 0) / scale
        dr = np.mean(d_redraw, 0) / scale
        out[conv] = {"perm_rms_rel": float(dp.mean()), "redraw_rms_rel": float(dr.mean()),
                     "ratio_perm_over_redraw": float(dp.mean() / max(dr.mean(), 1e-15)),
                     "perm_rms_rel_max": float(dp.max())}
    return out


def part0(res, device, n_win=8, rate=0.3):
    """Numerical rank of J_M = d(forecast)/d(x_M) for each input convention."""
    bolt = Bolt(device)
    out = {}
    for ds, path in DATASETS.items():
        X, starts = load_windows(path, 300, SEED, H)
        for mech in ("mcar", "block", "mnar_high"):
            clean, filled, mask, _ = build_eval_batch(X, starts[:n_win], mech, rate, H)
            pinv = perm_invariance(bolt, filled, mask)
            for conv, v in pinv.items():
                print(f"P0-perm {ds:8s} {mech:13s} {conv:11s} "
                      f"perm={v['perm_rms_rel']:.3e} redraw={v['redraw_rms_rel']:.3e} "
                      f"ratio={v['ratio_perm_over_redraw']:.4f}", flush=True)
            rows = []
            for i in range(min(len(filled), n_win)):        # one series per window
                mi = mask[i]
                if mi.sum() < 4:
                    continue
                idx = torch.from_numpy(np.flatnonzero(mi)).to(device)
                x_f = torch.from_numpy(filled[i:i + 1]).to(device)
                obs = torch.from_numpy((~mi)[None].astype(np.float32)).to(device)
                x_nan = x_f.clone()
                x_nan[0, idx] = float("nan")

                for conv, ctx, mk in (("plain_fill", x_f, None),
                                      ("fill_mask", x_f, obs),
                                      ("nan", x_nan, None)):
                    c = ctx.clone().requires_grad_(True)
                    y = bolt.median(c, mk)[0]                # [H]
                    J = []
                    for h in range(H):
                        g, = torch.autograd.grad(y[h], c, retain_graph=(h < H - 1))
                        J.append(g[0, idx].detach())
                    J = torch.stack(J)                       # [H, |M|]
                    J = torch.nan_to_num(J, nan=0.0)
                    sv = torch.linalg.svdvals(J.double()).cpu().numpy()
                    s1 = float(sv[0]) if sv[0] > 0 else 0.0
                    rank = int((sv > 1e-4 * max(s1, 1e-30)).sum()) if s1 > 0 else 0
                    rows.append({"conv": conv, "n_missing": int(mi.sum()),
                                 "absmax": float(J.abs().max()), "sv1": s1,
                                 "sv_top5": sv[:5].tolist(), "rank_1e-4": rank})
            for conv in ("plain_fill", "fill_mask", "nan"):
                sel = [r for r in rows if r["conv"] == conv]
                out[f"{ds}:{mech}:{conv}"] = {
                    "n": len(sel),
                    "rank_median": float(np.median([r["rank_1e-4"] for r in sel])),
                    "rank_max": int(max(r["rank_1e-4"] for r in sel)),
                    "absmax_max": float(max(r["absmax"] for r in sel)),
                    "per_series": sel,
                    "perm_invariance": pinv[conv],
                }
                print(f"P0 {ds:8s} {mech:13s} {conv:11s} rank(med/max)="
                      f"{out[f'{ds}:{mech}:{conv}']['rank_median']:.1f}/"
                      f"{out[f'{ds}:{mech}:{conv}']['rank_max']:d} "
                      f"|J|max={out[f'{ds}:{mech}:{conv}']['absmax_max']:.3e}", flush=True)
    res["part0"] = {"rate": rate, "n_win": n_win, "cells": out}
    return res


# ------------------------------------------------------------------ part A ----

def oracle_fill(bolt, clean, filled, mask, gt, conv, steps=300, lr=0.05, lo=None,
                chunk=192):
    """Series are optimised independently, so chunking is exact (not an approximation)."""
    if len(filled) > chunk:
        return np.concatenate([
            _oracle_fill(bolt, clean[i:i + chunk], filled[i:i + chunk], mask[i:i + chunk],
                         gt[i:i + chunk], conv, steps, lr,
                         None if lo is None else lo[i:i + chunk])
            for i in range(0, len(filled), chunk)])
    return _oracle_fill(bolt, clean, filled, mask, gt, conv, steps, lr, lo)


def _oracle_fill(bolt, clean, filled, mask, gt, conv, steps=300, lr=0.05, lo=None):
    """Per-window gradient-optimised fill against the TRUE future (E[inf], a lower bound
    on what any fill could achieve; an optimiser sanity gate, not a deployable method)."""
    dev = bolt.device
    xf = torch.from_numpy(filled).to(dev)
    m_miss = torch.from_numpy(mask.astype(np.float32)).to(dev)
    obs = 1.0 - m_miss
    y = torch.from_numpy(gt).to(dev)
    sd = torch.from_numpy(np.asarray(standardise(filled, obs.cpu().numpy())[1])).to(dev)
    v = torch.zeros_like(xf, requires_grad=True)
    opt = torch.optim.Adam([v], lr=lr)
    mk = obs if conv == "fill_mask" else None
    lo_t = None if lo is None else torch.from_numpy(lo).to(dev)
    best = None
    for it in range(steps):
        opt.zero_grad()
        cand = xf + m_miss * v * sd
        if lo_t is not None:                       # identified-set constraint x_M >= tau
            cand = torch.where(m_miss > 0, torch.maximum(cand, lo_t), cand)
        pred = bolt.median(cand, mk)
        loss = (((pred - y) / sd) ** 2).mean(1)
        loss.sum().backward()
        opt.step()
        with torch.no_grad():
            cur = ((pred - y) ** 2).mean(1)
            best = cur if best is None else torch.minimum(best, cur)
    return best.detach().cpu().numpy()


def partA(res, device, n_win=40, steps=600):
    bolt = Bolt(device)
    out = {}
    for ds, path in DATASETS.items():
        X, starts = load_windows(path, 300, SEED, H)
        st = starts[:n_win]
        _, cl_f, cl_m, gt_c = build_eval_batch(X, st, "clean", 0.0, H)
        mse_clean = ((bolt.median_np(cl_f) - gt_c) ** 2).mean(1)
        for mech in MECHS:
            for rate in RATES_EVAL:
                clean, filled, mask, gt = build_eval_batch(X, st, mech, rate, H)
                base = ((bolt.median_np(filled) - gt) ** 2).mean(1)
                lo = None
                if mech == "mnar_high":
                    # sharp identified-set bound, computable WITHOUT the truth: top-k
                    # censoring implies every censored value is >= max(observed).
                    tau = np.where(mask, -np.inf, clean).max(1, keepdims=True)
                    lo = np.broadcast_to(tau, clean.shape).astype(np.float32).copy()
                    tau_true = np.where(mask, clean, np.inf).min(1)
                    assert (tau_true + 1e-4 >= tau[:, 0]).all(), "identified set excludes truth"
                t0 = time.time()
                orc = oracle_fill(bolt, clean, filled, mask, gt, "plain_fill", steps=steps)
                orc_c = (oracle_fill(bolt, clean, filled, mask, gt, "plain_fill",
                                     steps=steps, lo=lo) if lo is not None else orc)
                key = f"{ds}:{mech}:{rate}"
                conv_frac = float((orc <= mse_clean + 1e-12).mean())
                out[key] = {
                    "converged_frac": conv_frac,
                    "rel_linear": float(base.mean() / mse_clean.mean()),
                    "rel_oracle_fill": float(orc.mean() / mse_clean.mean()),
                    "rel_oracle_fill_constrained": float(orc_c.mean() / mse_clean.mean()),
                    "mse_clean": float(mse_clean.mean()), "n": int(len(base)),
                    "per_series_oracle": orc.tolist(),
                    "per_series_linear": base.tolist(),
                    "per_series_clean": mse_clean.tolist(),
                }
                print(f"PA {key:26s} linear={out[key]['rel_linear']:7.3f} "
                      f"oracle={out[key]['rel_oracle_fill']:7.4f} "
                      f"cons={out[key]['rel_oracle_fill_constrained']:7.4f} "
                      f"conv={conv_frac:.2f} ({time.time()-t0:.0f}s)", flush=True)
    res["partA"] = {"n_win": n_win, "steps": steps, "cells": out}
    return res


# ---------------------------------------------------------------- training ----

class TrainStream:
    """Pooled train-region windows from all datasets, masked on the fly."""

    def __init__(self, mech, n_per_ds=4000, horizon=H, seed=SEED + 7):
        self.mech, self.horizon = mech, horizon
        self.rng = np.random.default_rng(seed)
        self.pool = []
        for ds, path in DATASETS.items():
            X, starts = load_train_windows(path, n_per_ds, horizon)
            self.pool.append((X, starts))

    @staticmethod
    def _degenerate(c, f):
        sd = float(np.std(c))
        return (not np.isfinite(sd)) or sd <= 1e-3 * (abs(float(np.mean(c))) + 1e-6) \
            or sd == 0.0 or not np.isfinite(f).all()

    @staticmethod
    def _flat_filled(f):
        """A constant filled context makes bolt's InstanceNorm sqrt(0) -> inf gradient."""
        return float(np.std(f)) <= 1e-8 * (abs(float(np.mean(f))) + 1e-6)

    def batch(self, bs, rate=None):
        ctx, fut = [], []
        for _ in range(bs):
            for _try in range(50):
                X, starts = self.pool[self.rng.integers(len(self.pool))]
                s = int(starts[self.rng.integers(len(starts))])
                c = int(self.rng.integers(X.shape[1]))
                cc, ff = X[s:s + L, c], X[s + L:s + L + self.horizon, c]
                if not self._degenerate(cc, ff):
                    break
            ctx.append(cc)
            fut.append(ff)
        ctx = np.stack(ctx).astype(np.float32)
        fut = np.stack(fut).astype(np.float32)
        p = rate if rate is not None else float(self.rng.uniform(0.1, 0.8))
        mask = np.zeros_like(ctx, bool)
        for i in range(bs):
            mask[i] = make_mask(self.mech, p, int(self.rng.integers(10 ** 6)),
                                0, 1, x=ctx[i][None])[0]
        filled = np.stack([fill_context(ctx[i][None], mask[i][None], "linear")[0]
                           for i in range(bs)])
        keep = np.array([not self._flat_filled(filled[i]) for i in range(bs)])
        if keep.sum() < 2:
            keep[:] = True                      # degenerate batch: let the grad guard catch it
        return ctx[keep], filled[keep], mask[keep], fut[keep]


def prep(filled, mask, device):
    obs = (~mask).astype(np.float32)
    mu, sd = standardise(filled, obs)
    v = torch.from_numpy((filled - mu) / sd).to(device)
    m = torch.from_numpy(obs).to(device)
    return v, m, torch.from_numpy(mu).to(device), torch.from_numpy(sd).to(device)


def loss_scale(ref, device):
    """Per-window loss WEIGHT, computed from the CLEAN context (available at training time
    because we synthesise the masks). Never enters the model input.

    Using the observed-only std here would be wrong twice over: under value censoring it is
    exactly the biased statistic this paper studies, so it shrinks with the censoring rate,
    the objective's scale becomes mechanism-dependent (init loss 44 for mnar_high vs 0.30
    for mcar/block) and training diverges."""
    sd = ref.std(axis=1, keepdims=True)
    absm = np.abs(ref).mean(axis=1, keepdims=True)
    sc = np.maximum(sd, 0.01 * absm)
    return torch.from_numpy(np.maximum(sc, 1e-6).astype(np.float32)).to(device)


def huber_scaled(pred, y, scale, delta=5.0):
    """Quadratic in the normal range, linear in the tail: bounded gradients."""
    r = torch.clamp((pred - y) / scale, -1e3, 1e3)
    return torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=delta)


def train_fillnet(bolt, mech, conv, steps=700, bs=48, lr=1e-4, log=100):
    dev = bolt.device
    net = FillNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    stream = TrainStream(mech)
    hist, n_skip = [], 0
    for it in range(steps):
        ctx, filled, mask, fut = stream.batch(bs)
        v, m, mu, sd = prep(filled, mask, dev)
        y = torch.from_numpy(fut).to(dev)
        x = net(v, m) * sd + mu
        pred = bolt.median(x, m if conv == "fill_mask" else None)
        loss = huber_scaled(pred, y, loss_scale(ctx, y.device))
        if not torch.isfinite(loss):
            n_skip += 1
            continue
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad()
            n_skip += 1
            continue
        opt.step()
        sch.step()
        hist.append(float(loss))
        if it == 0:
            print(f"    fillnet[{mech}/{conv}] init_loss={float(loss):.4f}", flush=True)
        if (it + 1) % log == 0:
            print(f"    fillnet[{mech}/{conv}] {it+1}/{steps} "
                  f"loss={np.mean(hist[-log:]):.4f} skip={n_skip}", flush=True)
    return net, hist


def train_directnet(mech, device, steps=1500, bs=128, lr=1e-3, log=300, clean=False):
    net = DirectNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    stream = TrainStream(mech if not clean else "mcar")
    hist, n_skip = [], 0
    for it in range(steps):
        ctx, filled, mask, fut = stream.batch(bs)
        if clean:
            filled, mask = ctx, np.zeros_like(mask)
        v, m, mu, sd = prep(filled, mask, device)
        y = torch.from_numpy(fut).to(device)
        pred = net(v, m) * sd + mu
        loss = huber_scaled(pred, y, loss_scale(ctx, y.device))
        if not torch.isfinite(loss):
            n_skip += 1
            continue
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad()
            n_skip += 1
            continue
        opt.step()
        sch.step()
        hist.append(float(loss))
        if (it + 1) % log == 0:
            print(f"    direct[{'clean' if clean else mech}] {it+1}/{steps} "
                  f"loss={np.mean(hist[-log:]):.4f} skip={n_skip}", flush=True)
    return net, hist


@torch.no_grad()
def eval_fillnet(bolt, net, X, starts, mech, rate, conv, batch=256):
    _, filled, mask, gt = build_eval_batch(X, starts, mech, rate, H)
    outs = []
    for i in range(0, len(filled), batch):
        v, m, mu, sd = prep(filled[i:i + batch], mask[i:i + batch], bolt.device)
        x = net(v, m) * sd + mu
        assert torch.isfinite(x).all(), "fill net emitted non-finite fill values"
        pred = bolt.median(x, m if conv == "fill_mask" else None)
        outs.append(pred.float().cpu().numpy())
    pred = np.concatenate(outs)
    assert np.isfinite(pred).all(), "fill net produced non-finite forecasts"
    return ((pred - gt) ** 2).mean(1)


@torch.no_grad()
def eval_directnet(net, X, starts, mech, rate, device, batch=256, clean=False):
    cl, filled, mask, gt = build_eval_batch(X, starts, mech, rate, H)
    if clean:
        filled, mask = cl, np.zeros_like(mask)
    outs = []
    for i in range(0, len(filled), batch):
        v, m, mu, sd = prep(filled[i:i + batch], mask[i:i + batch], device)
        outs.append((net(v, m) * sd + mu).float().cpu().numpy())
    pred = np.concatenate(outs)
    return ((pred - gt) ** 2).mean(1)


# ------------------------------------------------------------------ part B ----

def partB(res, device, steps=700, n_win=300, mechs=MECHS):
    bolt = Bolt(device)
    ck = os.path.join(HERE, "s25_ckpt")
    os.makedirs(ck, exist_ok=True)
    cells, base = {}, {}
    evald = {ds: load_windows(p, n_win, SEED, H) for ds, p in DATASETS.items()}

    for ds, (X, st) in evald.items():                       # fixed-fill references
        _, cl_f, _, gt_c = build_eval_batch(X, st, "clean", 0.0, H)
        mse_clean = ((bolt.median_np(cl_f) - gt_c) ** 2).mean(1)
        base[f"{ds}:clean"] = float(mse_clean.mean())
        for mech in mechs:
            for rate in RATES_EVAL:
                for fill in ("linear", "zero", "nan"):
                    _, f_, mk, gt = build_eval_batch(X, st, mech, rate, H, fill=fill)
                    mse = ((bolt.median_np(f_) - gt) ** 2).mean(1)
                    base[f"{ds}:{mech}:{rate}:{fill}"] = float(mse.mean() / mse_clean.mean())
        print(f"PB base {ds} clean={base[f'{ds}:clean']:.4f}", flush=True)

    for mech in mechs:
        for conv in ("plain_fill", "fill_mask"):
            t0 = time.time()
            net, hist = train_fillnet(bolt, mech, conv, steps=steps)
            assert all(torch.isfinite(q).all() for q in net.parameters()), \
                f"fill net weights went non-finite ({mech}/{conv})"
            torch.save(net.state_dict(), os.path.join(ck, f"fillnet_{mech}_{conv}.pt"))
            for ds, (X, st) in evald.items():
                for rate in RATES_EVAL:
                    mse = eval_fillnet(bolt, net, X, st, mech, rate, conv)
                    key = f"{ds}:{mech}:{rate}:{conv}"
                    cells[key] = {"rel": float(mse.mean() / base[f"{ds}:clean"]),
                                  "rel_median": float(np.median(mse) / base[f"{ds}:clean"]),
                                  "per_window": mse.tolist()}
                    print(f"PB {key:34s} rel={cells[key]['rel']:7.3f} "
                          f"(linear {base[f'{ds}:{mech}:{rate}:linear']:.3f}, "
                          f"nan {base[f'{ds}:{mech}:{rate}:nan']:.3f})", flush=True)
            print(f"  [{mech}/{conv}] trained in {time.time()-t0:.0f}s "
                  f"final_loss={np.mean(hist[-50:]):.4f}", flush=True)
    res["partB"] = {"steps": steps, "n_win": n_win, "base": base, "cells": cells}
    return res


# ------------------------------------------------------------------ part C ----

def partC(res, device, steps=1500, n_win=300):
    ck = os.path.join(HERE, "s25_ckpt")
    os.makedirs(ck, exist_ok=True)
    evald = {ds: load_windows(p, n_win, SEED, H) for ds, p in DATASETS.items()}
    cells, base = {}, {}

    net_c, _ = train_directnet("mcar", device, steps=steps, clean=True)
    torch.save(net_c.state_dict(), os.path.join(ck, "directnet_clean.pt"))
    for ds, (X, st) in evald.items():
        mse = eval_directnet(net_c, X, st, "clean", 0.0, device, clean=True)
        base[f"{ds}:clean"] = float(mse.mean())
        print(f"PC direct clean {ds} mse={base[f'{ds}:clean']:.4f}", flush=True)

    for mech in MECHS:
        net, hist = train_directnet(mech, device, steps=steps)
        torch.save(net.state_dict(), os.path.join(ck, f"directnet_{mech}.pt"))
        for ds, (X, st) in evald.items():
            for rate in RATES_EVAL:
                mse = eval_directnet(net, X, st, mech, rate, device)
                key = f"{ds}:{mech}:{rate}"
                cells[key] = {"rel": float(mse.mean() / base[f"{ds}:clean"]),
                              "per_window": mse.tolist()}
                print(f"PC {key:26s} rel={cells[key]['rel']:7.3f}", flush=True)
    res["partC"] = {"steps": steps, "n_win": n_win, "base": base, "cells": cells}
    return res


# ------------------------------------------------------------------- main ----

def save(res, path=RESULTS_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="gate,0,A,B,C")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--b-steps", type=int, default=700)
    ap.add_argument("--c-steps", type=int, default=1500)
    ap.add_argument("--n-win", type=int, default=300)
    ap.add_argument("--a-win", type=int, default=40)
    ap.add_argument("--a-steps", type=int, default=600)
    ap.add_argument("--mechs", default=",".join(MECHS))
    ap.add_argument("--out", default=RESULTS_PATH)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda"

    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("meta", {"L": L, "H": H, "H_gate": H_GATE, "seed": SEED,
                            "rates_eval": list(RATES_EVAL), "mechs": list(MECHS),
                            "model": "amazon/chronos-bolt-base (fp32, frozen)",
                            "datasets": DATASETS})

    parts = args.parts.split(",")
    if args.smoke:
        res = part_gate(res, device)
        res = part0(res, device, n_win=2, rate=0.3)
        res = partA(res, device, n_win=4, steps=40)
        res = partB(res, device, steps=20, n_win=20)
        res = partC(res, device, steps=40, n_win=20)
        print("SMOKE OK")
        return

    for p in parts:
        t0 = time.time()
        if p == "gate":
            res = part_gate(res, device)
        elif p == "0":
            res = part0(res, device)
        elif p == "A":
            res = partA(res, device, n_win=args.a_win, steps=args.a_steps)
        elif p == "B":
            res = partB(res, device, steps=args.b_steps, n_win=args.n_win,
                        mechs=tuple(args.mechs.split(",")))
        elif p == "C":
            res = partC(res, device, steps=args.c_steps, n_win=args.n_win)
        else:
            raise ValueError(p)
        save(res, args.out)
        print(f"== part {p} done in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
