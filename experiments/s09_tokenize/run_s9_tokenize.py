#!/usr/bin/env python
"""S9: attribute Chronos-T5's missing-data degradation to its tokenizer.

S5 (run_s5_missing.py) only ever tested fill-then-feed (zero/ffill/linear) for
every model, so Chronos-T5's NATIVE NaN path was never measured: the
MeanScaleUniformBins tokenizer (chronos.py:154-196) computes scale as the
nan-aware mean(|x_observed|), and turns NaN positions into pad_token_id with
attention_mask=0 -- the point is excised from the sequence, not filled.

Two sub-mechanisms can therefore drive t5's degradation under missingness:
  (a) "statistical channel": the scale statistic changes (fill pollution under
      linear -- interpolation compresses mean|x| toward the local mean; or
      observed-set bias under nan -- MNAR censors extremes so the observed
      points are a biased sample). output_transform multiplies bin centres by
      scale, so any scale bias is a MULTIPLICATIVE amplitude error.
  (b) "token channel": bin ids of surviving points drift (resolution collapse,
      edge-bin saturation), and nan mode shortens the effective sequence.

Design -- everything paired against S5:
  * Identical window selection: load_windows(path, 300, SEED)[:100] (S5 aux).
  * Identical masks: make_mask copied verbatim; mcar/block mask_seed=0
    (S5 aux_seeds=1), mnar deterministic.
  * Identical t5 inference: fp32, batch 128, num_samples=20, median of
    samples, torch.manual_seed(SEED+i) per batch, CPU input tensors.
  * Anchor gate (--part anchor) MUST pass before the full grid: reproduce
    t5 x {clean, mcar:linear:0.3, mcar:zero:0.3} x ETTh1 from
    s5_missing_results.json within 5%.

Part A (GPU): grid = mechanisms {mcar, block, mnar_high, mnar_extreme}
  x rates {0.1,0.3,0.5,0.7} x {ETTh1, ETTm1, weather}, fill=nan, plus the
  clean anchor per dataset. Per-config + per-window MSE, top-decile/rest
  decomposition (same fixed clean-context q90 threshold as S5), and NaN-path
  sanity counters (all-missing rows / zero-scale fallback rows / finite preds).

Part B (CPU, tokenizer only, no model): ETTh1 + weather, same 100 windows,
  mechanisms {mcar, block, mnar_high} x same rates. Each window is tokenized
  in 3 forms (clean / missing+NaN / missing+linear) with the REAL
  MeanScaleUniformBins from the cached chronos-t5-small config, recording per
  window: scale_ratio vs clean (both modes), bin-id drift on positions
  observed in both modes (bin_change_frac, mean|dbin|), resolution collapse
  (distinct bins, token entropy, observed positions only), and edge-bin
  (ids 2 and n_tokens-1) frequency.

--part report merges A+B (+S5/s5_fix for t5-{zero,ffill,linear} and bolt-nan)
into s9_results.json, computes Spearman correlations of |scale_ratio-1| and
edge-bin freq against per-window relMSE, and renders s9.png.

Artifacts: s9_partA_{ds}.json, s9_partB.json, s9_anchor.json, s9_results.json,
s9.png; logs s9_partA_{ds}.log / s9_partB.log (shell redirects).
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import pandas as pd
import torch

# ---- shared constants: identical to run_s5_missing.py ----------------------
L, H = 512, 96
RATES = [0.1, 0.3, 0.5, 0.7]
BLOCK = 24
SEED = 20250810
NWIN_DRAW, NWIN_AUX, AUX_SEEDS = 300, 100, 1   # S5 drew 300 starts, t5 used [:100]
MECHS_A = ["mcar", "block", "mnar_high", "mnar_extreme"]
MECHS_B = ["mcar", "block", "mnar_high"]

DATASETS = {
    "ETTh1": "tslib/dataset/ETT-small/ETTh1.csv",
    "ETTm1": "tslib/dataset/ETT-small/ETTm1.csv",
    "weather": "tslib/dataset/weather/weather.csv",
}
DATASETS_B = ["ETTh1", "weather"]
S5_PATH = os.path.join(HERE, "s5_missing_results.json")
S5_FIX_PATH = os.path.join(HERE, "s5_fix_results.json")


# ---- copied verbatim from run_s5_missing.py (pairing depends on it) ---------
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
    """[C, L] bool. Verbatim copy of the S5 mask generator."""
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
    """S5 fills + the new 'nan' mode: raw NaN straight into the tokenizer."""
    if fill in ("oracle", "none"):
        return x.copy()
    if fill == "nan":                       # NEW in S9: native NaN path
        out = x.copy()
        out[mask] = np.nan
        return out
    if fill == "zero":
        out = x.copy()
        out[mask] = 0.0
        return out
    out = x.copy()
    t = np.arange(L)
    for c in range(len(x)):
        valid = np.flatnonzero(~mask[c])
        if len(valid) == 0:
            out[c] = 0.0
        elif fill == "linear":
            out[c] = np.interp(t, valid, x[c, valid])  # edges -> ffill/bfill
        else:
            raise ValueError(fill)
    return out


def corr_patch_frac(mask, patch):
    if patch == 1:
        return float(mask.mean())
    P = L // patch
    return float(mask.reshape(mask.shape[0], P, patch).any(axis=2).mean())


# ---------------------------------------------------------------- Part A ----

def run_config_t5(model, X, starts, cfg, n_seeds=AUX_SEEDS):
    """S5's run_config specialised to t5 + extended with fill='nan', NaN-path
    sanity counters, and the top-decile split recorded for nan as well."""
    C = X.shape[1]
    mech, fill, rate = cfg["mech"], cfg["fill"], cfg["rate"]
    det = mech in ("clean", "mnar_high", "mnar_extreme")
    ctxs, gts, masks, q90s = [], [], [], []
    for wi, s in enumerate(starts):
        x_clean = X[s:s + L].T.copy()       # [C, L]
        y = X[s + L:s + L + H].T.copy()     # [C, H]
        q90s_w = np.quantile(x_clean, 0.9, axis=1)
        for ms in ((0,) if det else range(n_seeds)):
            mask = (np.zeros((C, L), bool) if mech == "clean"
                    else make_mask(mech, rate, wi, ms, C, x=x_clean))
            ctxs.append(fill_context(x_clean, mask, "none" if mech == "clean" else fill))
            gts.append(y)
            masks.append(mask)
            q90s.append(q90s_w)
    S = len(ctxs)
    ctx = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
    # NaN-path safety checks: the tokenizer's fallbacks must never trigger at
    # rate <= 0.7 (scale[~(scale>0)] = 1.0 fires only if a row is all-NaN or
    # all-zero on observed points)
    n_all_missing = int((~np.isfinite(ctx)).all(axis=1).sum()) if fill == "nan" else 0
    n_zero_scale = 0
    if fill == "nan":
        with np.errstate(invalid="ignore"):
            n_zero_scale = int((np.nansum(np.abs(ctx), axis=1) <= 0).sum())
    pred = model.predict_point(ctx).reshape(S, C, H).astype(np.float64)
    finite = bool(np.isfinite(pred).all())
    gt = np.stack(gts).astype(np.float64)
    err2 = (pred - gt) ** 2
    mse = err2.mean(axis=(1, 2))
    mae = np.abs(pred - gt).mean(axis=(1, 2))
    rate_sw = np.stack([m.mean() for m in masks])
    nw = len(starts)
    res = {
        "mech": mech, "fill": fill, "rate": rate,
        "n_windows": nw, "n_seeds": 1 if det else n_seeds,
        "mse": float(mse.mean()), "mae": float(mae.mean()),
        "achieved_rate": float(rate_sw.mean()),
        "mse_per_window": mse.reshape(nw, -1).mean(axis=1).tolist(),
        "mae_per_window": mae.reshape(nw, -1).mean(axis=1).tolist(),
        "n_all_missing_rows": n_all_missing,
        "n_zero_scale_rows": n_zero_scale,
        "pred_finite": finite,
    }
    if fill in ("linear", "nan") and mech != "clean":
        # extreme-value metric, same fixed threshold definition as S5
        top = gt > np.stack(q90s)[:, :, None]
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


def grid_a():
    cfgs = [dict(mech="clean", fill="none", rate=0.0)]
    for r in RATES:
        for mech in MECHS_A:
            cfgs.append(dict(mech=mech, fill="nan", rate=r))
    return cfgs


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def part_anchor(args):
    """Reproduction gate vs s5_missing_results.json (t5, ETTh1)."""
    model = T5Model("cuda")
    X, starts_all = load_windows(DATASETS["ETTh1"], NWIN_DRAW, SEED)
    starts = starts_all[:NWIN_AUX]
    cfgs = [dict(mech="clean", fill="none", rate=0.0),
            dict(mech="mcar", fill="linear", rate=0.3),
            dict(mech="mcar", fill="zero", rate=0.3)]
    s5 = json.load(open(S5_PATH))["t5"]["ETTh1"]
    out, ok_all = {}, True
    for cfg in cfgs:
        t0 = time.time()
        res = run_config_t5(model, X, starts, cfg)
        key = cfgkey(cfg)
        ref = s5[key]["mse"]
        dev = (res["mse"] - ref) / ref
        ok = abs(dev) <= 0.05
        ok_all &= ok
        out[key] = {"s9_mse": res["mse"], "s5_mse": ref, "rel_dev": dev, "pass": ok}
        print(f"anchor {key:20s} s9={res['mse']:10.4f} s5={ref:10.4f} "
              f"dev={dev:+.3%} {'PASS' if ok else 'FAIL'} ({time.time()-t0:.1f}s)",
              flush=True)
    out["_gate"] = {"threshold": 0.05, "pass": ok_all}
    save_json(out, os.path.join(HERE, "s9_anchor.json"))
    print(f"ANCHOR GATE {'PASS' if ok_all else 'FAIL'}", flush=True)
    return 0 if ok_all else 1


def part_smoke(args):
    """5 windows, ETTh1, nan path end-to-end: finite preds, no fallback."""
    model = T5Model("cuda")
    X, starts_all = load_windows(DATASETS["ETTh1"], NWIN_DRAW, SEED)
    starts = starts_all[:5]
    cfgs = [dict(mech="clean", fill="none", rate=0.0)]
    cfgs += [dict(mech=m, fill="nan", rate=0.3) for m in MECHS_A]
    ok_all = True
    for cfg in cfgs:
        res = run_config_t5(model, X, starts, cfg)
        ok = res["pred_finite"] and res["n_all_missing_rows"] == 0 \
            and res["n_zero_scale_rows"] == 0
        ok_all &= ok
        print(f"smoke {cfgkey(cfg):22s} mse={res['mse']:9.4f} finite={res['pred_finite']} "
              f"allmiss={res['n_all_missing_rows']} zeroscale={res['n_zero_scale_rows']} "
              f"{'OK' if ok else 'BAD'}", flush=True)
    print(f"SMOKE {'OK' if ok_all else 'BAD'}", flush=True)
    return 0 if ok_all else 1


def part_a(args):
    ds = args.shard
    assert ds in DATASETS, f"--shard must be one of {list(DATASETS)} for part A"
    path = os.path.join(HERE, f"s9_partA_{ds}.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    if results:
        print(f"resume: {path} has {len(results)} configs", flush=True)
    model = T5Model("cuda")
    X, starts_all = load_windows(DATASETS[ds], NWIN_DRAW, SEED)
    starts = starts_all[:NWIN_AUX]
    for cfg in grid_a():
        key = cfgkey(cfg)
        if key in results:
            continue
        t0 = time.time()
        res = run_config_t5(model, X, starts, cfg)
        res["seconds"] = round(time.time() - t0, 1)
        results[key] = res
        save_json(results, path)
        print(f"t5-nan {ds:8s} {key:22s} mse={res['mse']:10.4f} "
              f"ach={res['achieved_rate']:.3f} finite={res['pred_finite']} "
              f"zeroscale={res['n_zero_scale_rows']} ({res['seconds']}s)", flush=True)
    print(f"PART A {ds} DONE", flush=True)
    return 0


# ---------------------------------------------------------------- Part B ----

def build_tokenizer():
    """The real MeanScaleUniformBins of the cached chronos-t5-small."""
    from chronos import ChronosConfig
    hits = glob.glob(os.path.join(
        os.environ["HF_HOME"], "hub", "models--amazon--chronos-t5-small",
        "snapshots", "*", "config.json"))
    assert hits, "chronos-t5-small config.json not found in HF cache"
    cc = json.load(open(hits[0]))["chronos_config"]
    cfg = ChronosConfig(**cc)
    return cfg.create_tokenizer(), cfg


@torch.no_grad()
def tokenize(tok, ctx):
    """ctx [n, L] float32 numpy (may hold NaN) -> (ids [n,L] int64,
    attn [n,L] bool, scale [n] float32) via the real _input_transform (no EOS:
    the appended token carries no information about the context stats)."""
    ids, amask, scale = tok._input_transform(torch.from_numpy(ctx))
    return ids.numpy(), amask.numpy(), scale.numpy()


def row_stats(ids, obs, n_tokens, edge_lo, edge_hi):
    """Per-row distinct-bin count, token entropy (bits) and edge-bin fraction,
    observed positions only. ids [n, L] int64, obs [n, L] bool."""
    n = ids.shape[0]
    distinct = np.zeros(n); entropy = np.zeros(n); edge = np.zeros(n)
    for i in range(n):
        v = ids[i][obs[i]]
        cnt = np.bincount(v, minlength=n_tokens).astype(np.float64)
        p = cnt[cnt > 0] / cnt.sum()
        distinct[i] = len(p)
        entropy[i] = float(-(p * np.log2(p)).sum())
        edge[i] = float(((v == edge_lo) | (v == edge_hi)).mean())
    return distinct, entropy, edge


def part_b(args):
    tok, cfg = build_tokenizer()
    n_tokens, nsp = cfg.n_tokens, cfg.n_special_tokens
    edge_lo, edge_hi = nsp, n_tokens - 1          # real bin ids 2 and 4095
    print(f"tokenizer: n_tokens={n_tokens} edge bins=({edge_lo},{edge_hi}) "
          f"limits=({cfg.tokenizer_kwargs})", flush=True)
    out = {"meta": {"n_tokens": n_tokens, "n_special_tokens": nsp,
                    "edge_bins": [edge_lo, edge_hi], "windows": NWIN_AUX,
                    "mask_seed": 0, "L": L, "H": H, "seed": SEED,
                    "datasets": DATASETS_B, "mechs": MECHS_B, "rates": RATES},
           "clean": {}, "stats": {}}
    for ds in DATASETS_B:
        X, starts_all = load_windows(DATASETS[ds], NWIN_DRAW, SEED)
        starts = starts_all[:NWIN_AUX]
        C = X.shape[1]
        nw = len(starts)
        # clean tokenization once per window (mechanism-independent)
        clean_ids = np.empty((nw, C, L), np.int64)
        clean_scale = np.empty((nw, C))
        for wi, s in enumerate(starts):
            xc = X[s:s + L].T.copy().astype(np.float32)
            ids, _, sc = tokenize(tok, xc)
            clean_ids[wi], clean_scale[wi] = ids, sc
        full = np.ones((nw, C, L), bool)
        d_cl, e_cl, g_cl = row_stats(clean_ids.reshape(nw * C, L),
                                     full.reshape(nw * C, L), n_tokens,
                                     edge_lo, edge_hi)
        # tokenizer fallback `scale[~(scale>0)] = 1.0` fires when a row's
        # observed values are all exactly 0 (degenerate channels such as
        # weather 'rain (mm)'); detect it from the raw context, since the
        # returned scale has already been patched to 1.0
        n_fb_clean = int((np.abs(np.stack(
            [X[s:s + L].T for s in starts]).reshape(nw * C, L)).sum(axis=1) <= 0).sum())
        out["clean"][ds] = {
            "scale": clean_scale.mean(axis=1).tolist(),
            "distinct": d_cl.reshape(nw, C).mean(axis=1).tolist(),
            "entropy": e_cl.reshape(nw, C).mean(axis=1).tolist(),
            "edge_frac": g_cl.reshape(nw, C).mean(axis=1).tolist(),
            "n_fallback_rows": n_fb_clean,
            "fallback_frac": n_fb_clean / (nw * C),
        }
        print(f"{ds}: clean tokenized ({nw}w x {C}ch), "
              f"mean distinct={d_cl.mean():.1f} entropy={e_cl.mean():.2f}b "
              f"edge={g_cl.mean():.4f}", flush=True)
        out["stats"][ds] = {}
        for mech in MECHS_B:
            out["stats"][ds][mech] = {}
            for rate in RATES:
                t0 = time.time()
                # build masked contexts for all windows
                nan_rows, lin_rows, obs_rows, idx_rows = [], [], [], []
                for wi, s in enumerate(starts):
                    xc = X[s:s + L].T.copy().astype(np.float32)
                    mask = make_mask(mech, rate, wi, 0, C, x=xc)
                    nan_rows.append(fill_context(xc, mask, "nan"))
                    lin_rows.append(fill_context(xc, mask, "linear"))
                    obs_rows.append(~mask)
                nan_ctx = np.concatenate(nan_rows)          # [nw*C, L]
                lin_ctx = np.concatenate(lin_rows)
                obs = np.concatenate(obs_rows)              # True = observed
                ids_n, am_n, sc_n = tokenize(tok, nan_ctx)
                ids_l, _, sc_l = tokenize(tok, lin_ctx)
                assert (am_n == obs).all()
                ids_c = clean_ids.reshape(nw * C, L)
                sc_c = clean_scale.reshape(nw * C)
                # fallback counters, detected from the raw contexts
                with np.errstate(invalid="ignore"):
                    fb_nan = int((np.nansum(np.abs(nan_ctx), axis=1) <= 0).sum())
                fb_lin = int((np.abs(lin_ctx).sum(axis=1) <= 0).sum())
                rec = {}
                for mode, ids_m, sc_m in (("nan", ids_n, sc_n), ("linear", ids_l, sc_l)):
                    sr = sc_m / sc_c
                    # bin drift on positions observed in BOTH modes == obs
                    chg = ((ids_m != ids_c) & obs).sum(axis=1) / obs.sum(axis=1)
                    adb = (np.abs(ids_m - ids_c) * obs).sum(axis=1) / obs.sum(axis=1)
                    ob_m = obs if mode == "nan" else np.ones_like(obs)
                    dst, ent, edg = row_stats(ids_m, ob_m, n_tokens, edge_lo, edge_hi)
                    rec[mode] = {
                        "scale_ratio": sr.reshape(nw, C).mean(axis=1).tolist(),
                        "bin_change_frac": chg.reshape(nw, C).mean(axis=1).tolist(),
                        "mean_abs_dbin": adb.reshape(nw, C).mean(axis=1).tolist(),
                        "distinct": dst.reshape(nw, C).mean(axis=1).tolist(),
                        "distinct_frac": (dst / d_cl).reshape(nw, C).mean(axis=1).tolist(),
                        "entropy": ent.reshape(nw, C).mean(axis=1).tolist(),
                        "edge_frac": edg.reshape(nw, C).mean(axis=1).tolist(),
                    }
                rec["nan"]["n_fallback_rows"] = fb_nan
                rec["linear"]["n_fallback_rows"] = fb_lin
                rec["achieved_rate"] = float((~obs).mean())
                out["stats"][ds][mech][str(rate)] = rec
                print(f"{ds:8s} {mech:12s} p={rate:<4} sr_nan={np.mean(rec['nan']['scale_ratio']):.4f} "
                      f"sr_lin={np.mean(rec['linear']['scale_ratio']):.4f} "
                      f"chg_nan={np.mean(rec['nan']['bin_change_frac']):.3f} "
                      f"chg_lin={np.mean(rec['linear']['bin_change_frac']):.3f} "
                      f"fb_nan={fb_nan} fb_lin={fb_lin} ({time.time()-t0:.1f}s)", flush=True)
        save_json(out, os.path.join(HERE, "s9_partB.json"))
        print(f"{ds} saved", flush=True)
    print("PART B DONE", flush=True)
    return 0


# ---------------------------------------------------------------- report ----

def relmse_table(records, clean_key="clean:none:0.0"):
    """records: {ds: {key: res}} -> {(mech, fill, rate): {ds: relMSE}}."""
    out = {}
    for ds, cfgs in records.items():
        if ds.startswith("_"):
            continue
        clean = cfgs[clean_key]["mse"]
        for key, r in cfgs.items():
            out.setdefault((r["mech"], r["fill"], r["rate"]), {})[ds] = r["mse"] / clean
    return out


def ds_avg(tbl, mech, fill, rate, datasets):
    vals = [tbl[(mech, fill, rate)][ds] for ds in datasets
            if (mech, fill, rate) in tbl and ds in tbl[(mech, fill, rate)]]
    return float(np.mean(vals)) if vals else None


def spearman(x, y):
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return None, int(ok.sum())
    rho, p = spearmanr(x[ok], y[ok])
    if not np.isfinite(rho):  # constant input (e.g. ETTh1 edge_frac == 0 everywhere)
        return None, int(ok.sum())
    return float(rho), int(ok.sum())


def part_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- load Part A (t5-nan) and references --------------------------------
    pa, pa_relw = {}, {}
    for ds in DATASETS:
        pa[ds] = json.load(open(os.path.join(HERE, f"s9_partA_{ds}.json")))
    s5 = json.load(open(S5_PATH))
    s5fix = json.load(open(S5_FIX_PATH))
    pb = json.load(open(os.path.join(HERE, "s9_partB.json")))

    tbl_nan = relmse_table(pa)                       # t5 nan (S9)
    tbl_s5 = relmse_table(s5["t5"])                  # t5 zero/ffill/linear
    tbl_bolt = relmse_table(s5fix["bolt"])           # bolt nan (fix0)
    datasets = list(DATASETS)

    # per-window relMSE: nan vs its own clean, linear vs S5 clean
    relw_nan, relw_lin = {}, {}
    for ds in DATASETS:
        cw = np.array(pa[ds]["clean:none:0.0"]["mse_per_window"])
        relw_nan[ds] = {k: (np.array(r["mse_per_window"]) / cw).tolist()
                        for k, r in pa[ds].items() if r["mech"] != "clean"}
        cw5 = np.array(s5["t5"][ds]["clean:none:0.0"]["mse_per_window"])
        relw_lin[ds] = {k: (np.array(r["mse_per_window"]) / cw5).tolist()
                        for k, r in s5["t5"][ds].items()
                        if isinstance(r, dict) and r.get("fill") == "linear"}

    # ---- comparison table: t5-nan vs t5-linear vs bolt-nan -------------------
    comp = {}
    print("\n== dataset-avg relMSE: t5-nan / t5-linear / bolt-nan ==", flush=True)
    for mech in MECHS_A:
        for r in RATES:
            row = {"t5_nan": ds_avg(tbl_nan, mech, "nan", r, datasets),
                   "t5_linear": ds_avg(tbl_s5, mech, "linear", r, datasets),
                   "bolt_nan": ds_avg(tbl_bolt, mech, "nan", r, datasets)}
            cands = {k: v for k, v in row.items() if v is not None}
            row["winner"] = min(cands, key=cands.get) if cands else None
            comp[f"{mech}:{r}"] = row
            print(f"{mech:14s} p={r:<4} nan={row['t5_nan']:.3f} "
                  f"lin={row['t5_linear']:.3f} bolt_nan={row['bolt_nan']:.3f} "
                  f"-> {row['winner']}", flush=True)

    # ---- correlations: scale_ratio & edge bins vs per-window relMSE ----------
    # IMPORTANT: report per dataset. Pooling ETTh1+weather cancels the signal
    # (Simpson's paradox): on ETTh1 scale distortion -> higher relMSE, while on
    # weather the largest scale_ratio windows are degenerate all-zero channels
    # whose relMSE *drops* below 1, inverting the correlation.
    corr = {}
    xy_ds = {}   # (mode, mech, ds) -> (|scale_ratio-1|, relMSE) per window
    for mode, relw, fill in (("nan", relw_nan, "nan"), ("linear", relw_lin, "linear")):
        for mech in MECHS_B:
            for ds in DATASETS_B:
                xs, ys = [], []
                for r in RATES:
                    key = f"{mech}:{fill}:{r}"
                    if key not in relw.get(ds, {}):
                        continue
                    rec = pb["stats"][ds][mech][str(r)][mode]
                    xs += [abs(v - 1.0) for v in rec["scale_ratio"]]
                    ys += relw[ds][key]
                xy_ds[(mode, mech, ds)] = (xs, ys)
                rho, n = spearman(xs, ys)
                corr[f"scale_vs_relmse:{ds}:{mech}:{mode}"] = {"rho": rho, "n": n}
        # per-dataset pooled over mechanisms, for both stats
        for ds in DATASETS_B + ["pooled"]:
            dss = DATASETS_B if ds == "pooled" else [ds]
            xs_s = [v for mech in MECHS_B for d in dss for v in xy_ds[(mode, mech, d)][0]]
            ys_s = [v for mech in MECHS_B for d in dss for v in xy_ds[(mode, mech, d)][1]]
            rho, n = spearman(xs_s, ys_s)
            corr[f"scale_vs_relmse:{ds}:all:{mode}"] = {"rho": rho, "n": n}
            xs_e, ys_e = [], []
            for mech in MECHS_B:
                for d in dss:
                    for r in RATES:
                        key = f"{mech}:{fill}:{r}"
                        if key not in relw.get(d, {}):
                            continue
                        rec = pb["stats"][d][mech][str(r)][mode]
                        xs_e += rec["edge_frac"]
                        ys_e += relw[d][key]
            rho, n = spearman(xs_e, ys_e)
            corr[f"edge_vs_relmse:{ds}:all:{mode}"] = {"rho": rho, "n": n}
    print("\n== Spearman correlations ==", flush=True)
    for k, v in corr.items():
        print(f"  {k:38s} rho={v['rho']} n={v['n']}", flush=True)

    # ---- s9_results.json ------------------------------------------------------
    anchor = json.load(open(os.path.join(HERE, "s9_anchor.json"))) \
        if os.path.exists(os.path.join(HERE, "s9_anchor.json")) else None
    results = {
        "meta": {"L": L, "H": H, "seed": SEED, "windows": NWIN_AUX,
                 "mask_seed_mcar_block": 0, "rates": RATES,
                 "mechanisms_A": MECHS_A, "mechanisms_B": MECHS_B,
                 "model": "amazon/chronos-t5-small (fp32, batch 128, median of 20 samples)",
                 "pairing": "identical windows/masks/inference as run_s5_missing.py (t5 aux grid)",
                 "relmse": "config mse / clean mse per dataset; dataset-avg = mean over datasets"},
        "anchor_gate": anchor,
        "partA": pa,
        "partB": pb,
        "comparison_dataset_avg_relmse": comp,
        "correlations": corr,
    }
    save_json(results, os.path.join(HERE, "s9_results.json"))
    print("\ns9_results.json written", flush=True)

    # ---- figure ----------------------------------------------------------------
    mech_titles = {"mcar": "MCAR", "block": "Block (24-step runs)",
                   "mnar_high": "MNAR-high (censor largest)",
                   "mnar_extreme": "MNAR-extreme (censor |z|)"}
    fills = [("zero", "t5 zero (S5)", "#d62728", "o"),
             ("ffill", "t5 ffill (S5)", "#ff7f0e", "s"),
             ("linear", "t5 linear (S5)", "#1f77b4", "^"),
             ("nan", "t5 nan (S9)", "#2ca02c", "D")]
    fig = plt.figure(figsize=(17, 13))
    gs = fig.add_gridspec(3, 4, hspace=0.32, wspace=0.30,
                          height_ratios=[1.0, 1.0, 0.95])
    # (i) relMSE curves per mechanism
    for j, mech in enumerate(MECHS_A):
        ax = fig.add_subplot(gs[0, j])
        for fill, lab, col, mk in fills:
            src = tbl_nan if fill == "nan" else tbl_s5
            ys = [ds_avg(src, mech, fill, r, datasets) for r in RATES]
            if any(v is None for v in ys):
                continue
            ax.plot(RATES, ys, marker=mk, ms=4, color=col, label=lab)
        ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
        ax.set_yscale("log")
        ax.set_title(mech_titles[mech], fontsize=10)
        ax.set_xlabel("missing rate"); ax.grid(alpha=0.3)
        if j == 0:
            ax.set_ylabel("relMSE (dataset avg, log)")
            ax.legend(fontsize=7, loc="upper left")
    # (ii) money plot: |scale_ratio-1| vs per-window relMSE
    mech_col = {"mcar": "#1f77b4", "block": "#ff7f0e", "mnar_high": "#d62728"}
    ds_mark = {"ETTh1": "o", "weather": "x"}
    for j, (mode, ttl) in enumerate((("nan", "nan mode (vs t5-nan relMSE)"),
                                     ("linear", "linear mode (vs t5-linear relMSE)"))):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        for mech in MECHS_B:
            for ds in DATASETS_B:
                xs, ys = xy_ds[(mode, mech, ds)]
                ax.scatter(np.maximum(xs, 1e-5), ys, s=6, alpha=0.35,
                           color=mech_col[mech], marker=ds_mark[ds],
                           label=(mech_titles[mech].split(" (")[0] if ds == "ETTh1" else None))
        rhos = [corr[f"scale_vs_relmse:ETTh1:{m}:{mode}"]["rho"] for m in MECHS_B]
        rho_w = corr["scale_vs_relmse:weather:all:" + mode]["rho"]
        ax.set_title(f"|scale_ratio - 1| vs window relMSE — {ttl}\n"
                     f"Spearman ρ ETTh1 mcar/block/mnar_high = "
                     f"{rhos[0]:+.2f}/{rhos[1]:+.2f}/{rhos[2]:+.2f}; weather(all) = {rho_w:+.2f}",
                     fontsize=9)
        ax.set_xlabel("|scale_ratio - 1| per window (log; o = ETTh1, x = weather)")
        ax.set_ylabel("window relMSE")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8, markerscale=2)
    # (iii) resolution collapse: distinct-bin fraction vs rate
    for j, mech in enumerate(MECHS_B):
        ax = fig.add_subplot(gs[2, j])
        for mode, lab, col, mk in (("nan", "nan", "#2ca02c", "D"),
                                   ("linear", "linear", "#1f77b4", "^")):
            ys = []
            for r in RATES:
                vals = [v for ds in DATASETS_B
                        for v in pb["stats"][ds][mech][str(r)][mode]["distinct_frac"]]
                ys.append(float(np.mean(vals)))
            ax.plot(RATES, ys, marker=mk, ms=4, color=col, label=lab)
        ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5, label="clean")
        ax.set_title(mech_titles[mech].split(" (")[0], fontsize=10)
        ax.set_xlabel("missing rate"); ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        if j == 0:
            ax.set_ylabel("distinct bins / clean")
            ax.legend(fontsize=8)
    # entropy summary panel
    ax = fig.add_subplot(gs[2, 3])
    for mech in MECHS_B:
        for mode, col, ls in (("nan", "#2ca02c", "-"), ("linear", "#1f77b4", "--")):
            ys = []
            for r in RATES:
                vals = [v / c for ds in DATASETS_B
                        for v, c in zip(pb["stats"][ds][mech][str(r)][mode]["entropy"],
                                        pb["clean"][ds]["entropy"])]
                ys.append(float(np.mean(vals)))
            ax.plot(RATES, ys, color=col, ls=ls, ms=4, marker="o",
                    label=f"{mech.split('_')[0]} {mode}")
    ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_title("token entropy / clean", fontsize=10)
    ax.set_xlabel("missing rate"); ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("S9 — Chronos-T5 tokenizer under missing data: native-NaN vs fill-then-feed", fontsize=12)
    fig.savefig(os.path.join(HERE, "s9.png"), dpi=140, bbox_inches="tight")
    print("s9.png written", flush=True)
    return 0


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["anchor", "smoke", "A", "B", "report"])
    ap.add_argument("--shard", default="",
                    help="part A: dataset name (ETTh1/ETTm1/weather)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if args.part == "anchor":
        return part_anchor(args)
    if args.part == "smoke":
        return part_smoke(args)
    if args.part == "A":
        return part_a(args)
    if args.part == "B":
        return part_b(args)
    if args.part == "report":
        return part_report(args)


if __name__ == "__main__":
    sys.exit(main())
