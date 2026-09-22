#!/usr/bin/env python
"""S70: the R/C selective-trust probe on Chronos-2 (external validity for S69).

On real benchmarks (ETTh1, ETTm1, weather) a target channel is holed
({mcar, block} x rate {0.3, 0.7}) and its K=4 most target-correlated neighbour
channels (correlation on the first 80% of the timeline only, S31's no-leak rule)
are supplied as past_covariates. Two factors are manipulated independently:

  q_r (target repair quality): holes filled with {zero, linear, oracle}
  q_c (context quality):       neighbours {clean, corrupt} -- corrupt = holed at
                               the same rate/mechanism (independent masks,
                               mask_seed=1) and zero-filled

across three models:

  vanilla  stock Chronos-2 through the S31 pipeline path. Fills are SILENT
           (plain convention: content present, positions not marked) -- the
           native interface cannot carry a content-preserving declaration.
           This mirrors S69's native arms and makes vanilla at
           (q_r=linear, q_c=clean) identical to S31 condition (c).
  ctl5k    S65 fair-budget control (s65_ckpt/c2_nomiss.pt): 5,000 CPT steps on
           the no-missingness filtered regime, dual interface. Fills DECLARED
           through the dual endpoint (content + per-position flag), for the
           target and for corrupt neighbours alike (S69 dual-arm semantics).
  cpt      S60 CPT retrofit (s60_ckpt/arm_CPT_s20260903.pt, seed 20260903 --
           the checkpoint behind s60_leaderboard_s20260903.json). Declared,
           same as ctl5k.

Metric: paired per-window relMSE vs the model's OWN clean univariate forecast
(S31 convention: median of per-window ratios over windows with clean MSE >
1e-12). 150 windows per cell (DESIGN default; --n-win for smoke).

GATE G0 (--g0-only): reproduce S31's stored anchors -- conditions (c)
(holed_multi) and (d) (nan_multi) at block 0.7 -- within 10% per (ds, ch)
before any new cell runs. Uses the same code path as S31 (Chronos2Pipeline
with past_covariates).

Dual arms run through a grouped extension of s60's c2_iface replica: the only
change vs c2_iface.forward_iface is that group_ids are passed to the encoder
so the target row and its neighbour rows mix, exactly as the stock pipeline
does when it stacks target + past_covariates into one group. The stock
pipeline's all-NaN future_covariates for past-only covariates is op-for-op
identical to future_covariates=None (both yield zero patched future
covariates), so the replica matches the pipeline path except for the removed
zeroing line (dual) and the explicit mask.

All eval strict fp32 (TF32 off), matching s60/s65 eval conventions.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False     # strict fp32 eval
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s31_crosschannel"))
sys.path.insert(0, os.path.join(EXP, "s60_c2retrofit"))
import run_s25_twofloor as s25
import run_s31_crosschannel as s31
import c2_iface

L, H = s25.L, 64
K_NEIGH = s31.K_NEIGH
MECHS = ("mcar", "block")
RATES = (0.3, 0.7)
Q_R = ("zero", "linear", "oracle")
Q_C = ("clean", "corrupt")
MODELS = ("vanilla", "ctl5k", "cpt")
OUT = os.path.join(HERE, "s70_rcprobe.json")
S31_ANCHORS = os.path.join(EXP, "s31_crosschannel", "s31_results.json")
CKPT = {
    "ctl5k": os.path.join(EXP, "s65_fairbudget", "s65_ckpt", "c2_nomiss.pt"),
    "cpt": os.path.join(EXP, "s60_c2retrofit", "s60_ckpt", "arm_CPT_s20260903.pt"),
}
MODEL_IFACE = {"vanilla": "native", "ctl5k": "dual", "cpt": "dual"}
G0_TOL = 0.10


# ------------------------------------------------- grouped dual forward ----
# c2_iface.forward_iface with one change: group_ids are accepted and passed to
# the encoder (stock behaviour when group_ids is given). With singleton groups
# (arange) this reduces to the S60-gated replica bit-for-bit.

def forward_grouped(model, context, context_mask, iface, group_ids,
                    num_output_patches):
    cfg = model.chronos_config
    batch_size = context.shape[0]
    patched_context, attention_mask, loc_scale = c2_iface.prepare_ctx(
        model, context, context_mask, iface)
    num_context_patches = attention_mask.shape[-1]

    input_embeds = model.input_patch_embedding(patched_context)
    if cfg.use_reg_token:
        reg_input_ids = torch.full((batch_size, 1), model.config.reg_token_id,
                                   device=input_embeds.device)
        reg_embeds = model.shared(reg_input_ids)
        input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
        attention_mask = torch.cat(
            [attention_mask.to(model.dtype),
             torch.ones_like(reg_input_ids).to(model.dtype)], dim=-1)

    patched_future, _ = model._prepare_patched_future(
        future_covariates=None, future_covariates_mask=None, loc_scale=loc_scale,
        num_output_patches=num_output_patches, batch_size=batch_size)
    future_attention_mask = torch.ones(batch_size, num_output_patches,
                                       dtype=model.dtype, device=model.device)
    future_embeds = model.input_patch_embedding(patched_future)

    input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
    attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)

    encoder_outputs = model.encoder(attention_mask=attention_mask,
                                    inputs_embeds=input_embeds,
                                    group_ids=group_ids)
    hidden_states = encoder_outputs[0]
    assert hidden_states.shape == (batch_size, num_context_patches + 1
                                   + num_output_patches, model.model_dim)

    from einops import rearrange
    forecast_embeds = hidden_states[:, -num_output_patches:]
    quantile_preds = model.output_patch_embedding(forecast_embeds)
    quantile_preds = rearrange(
        quantile_preds, "b n (q p) -> b q (n p)",
        n=num_output_patches, q=model.num_quantiles, p=cfg.output_patch_size)
    h = num_output_patches * cfg.output_patch_size
    quantile_preds = rearrange(quantile_preds, "b q h -> b (q h)",
                               b=batch_size, q=model.num_quantiles, h=h)
    quantile_preds = model.instance_norm.inverse(quantile_preds, loc_scale)
    quantile_preds = rearrange(quantile_preds, "b (q h) -> b q h",
                               q=model.num_quantiles, h=h)
    return quantile_preds


@torch.no_grad()
def median_fwd_grouped(model, ctx_np, obs_np, iface, n_rows_per_group,
                       horizon=H, batch_rows=500):
    """Median-quantile forecast for every TARGET row (row 0 of each group).

    ctx_np/obs_np: [n_groups * n_rows_per_group, L] ordered group-by-group.
    Groups are never split across batches.
    """
    assert horizon % model.chronos_config.output_patch_size == 0
    nop = horizon // model.chronos_config.output_patch_size
    q_med = model.chronos_config.quantiles.index(0.5)
    dev = next(model.parameters()).device
    r = n_rows_per_group
    n_groups = len(ctx_np) // r
    gids = np.repeat(np.arange(n_groups, dtype=np.int64), r)
    outs = []
    step = (batch_rows // r) * r
    for i in range(0, len(ctx_np), step):
        c = torch.from_numpy(ctx_np[i:i + step]).to(dev)
        o = torch.from_numpy(obs_np[i:i + step]).to(dev)
        g = torch.from_numpy(gids[i:i + step]).to(dev)
        q = forward_grouped(model, c, o, iface, g, nop)
        outs.append(q[:, q_med, :horizon].float().cpu().numpy())
    q = np.concatenate(outs)
    return q[0::r]   # target = first row of a group


# ------------------------------------------------------------- data prep ----

def s31_channels(Xall):
    """Reproduce S31's per-dataset target-channel draw exactly."""
    rng = np.random.default_rng(s25.SEED)
    return sorted(rng.choice(Xall.shape[1], size=min(6, Xall.shape[1]),
                             replace=False).tolist())


def build_masks(mech, rate, n_win, K):
    """Per-window target masks [n, L] (mask_seed=0, C=1 -- S31's draw) and
    neighbour masks [n, K, L] (mask_seed=1, independent of the target's)."""
    mt, mc = [], []
    for wi in range(n_win):
        mt.append(s25.make_mask(mech, rate, wi, 0, 1)[0])
        mc.append(s25.make_mask(mech, rate, wi, 1, K))
    return np.asarray(mt), np.asarray(mc)


def cell_arrays(X, starts, ch, nb, mech, rate, n_win):
    """All deterministic per-(cell-sans-model) arrays.

    Returns clean [n,L], gt [n,H], target masks [n,L], filled targets per q_r
    {q_r: [n,L]}, clean covariates [n,K,L], corrupt covariates [n,K,L],
    neighbour masks [n,K,L].
    """
    clean, gt, cov = [], [], []
    for s in starts[:n_win]:
        w = X[s:s + L].T
        clean.append(w[ch].copy())
        cov.append(w[nb].copy())
        gt.append(X[s + L:s + L + H].T[ch].copy())
    f32 = lambda a: np.asarray(a, np.float32)
    clean, gt, cov = f32(clean), f32(gt), f32(cov)
    mt, mc = build_masks(mech, rate, len(clean), cov.shape[1])
    fills = {q: s25.fill_context(clean, mt, q) for q in Q_R}
    cov_corrupt = s25.fill_context(cov, mc, "zero")
    return clean, gt, mt, fills, cov, cov_corrupt, mc


# --------------------------------------------------------------- models -----

def load_dual(arm, dev):
    m = c2_iface.load_stock(dev)
    sd = torch.load(CKPT[arm], map_location=dev, weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    return m


def fc_dual(model, targets, covs, tmask, cmask, iface):
    """Grouped declared forecast. targets [n,L] (filled, finite); covs [n,K,L]
    (filled, finite); tmask/cmask bool hole masks. Returns [n, H] medians."""
    n, K = targets.shape[0], covs.shape[1]
    ctx = np.concatenate([targets[:, None], covs], axis=1)          # [n,1+K,L]
    obs = np.ones_like(ctx)
    obs[:, 0] = (~tmask).astype(np.float32)                          # declared
    for k in range(K):
        obs[:, 1 + k] = (~cmask[:, k]).astype(np.float32)
    ctx = ctx.reshape(n * (1 + K), L).astype(np.float32)
    obs = obs.reshape(n * (1 + K), L).astype(np.float32)
    return median_fwd_grouped(model, ctx, obs, iface, 1 + K)


# ----------------------------------------------------------------- eval -----

def rel_median(e, e0):
    ok = e0 > 1e-12
    r = e[ok] / e0[ok]
    return float(np.median(r)), float(r.mean()), int(ok.sum()), r


def run_dataset(ds, models, n_win, res, max_ch=None, tag=""):
    import pandas as pd
    path = s25.DATASETS[ds]
    Xall = pd.read_csv(path).drop(columns=["date"]).to_numpy(np.float32)
    nb_map = s31.neighbours(Xall, int(0.8 * len(Xall)))
    X, st = s25.load_windows(path, n_win, s25.SEED, H)
    chans = s31_channels(Xall)
    if max_ch:
        chans = chans[:max_ch]
    print(f"[{tag}] {ds}: channels {chans}, {len(st)} windows", flush=True)

    for ch in chans:
        nb = nb_map[ch]
        # clean univariate reference per model (paired per window); the clean
        # context/horizon do not depend on (mech, rate), so computed once
        clean, gt0, _, _, _, _, _ = cell_arrays(X, st, ch, nb, "mcar", 0.3,
                                                len(st))
        base = {}
        for name, model in models.items():
            if name == "vanilla":
                e0 = ((model.fc(clean) - gt0) ** 2).mean(1)
            else:
                obs1 = np.ones_like(clean)
                y0 = median_fwd_grouped(model, clean, obs1, MODEL_IFACE[name], 1)
                e0 = ((y0 - gt0) ** 2).mean(1)
            base[name] = e0
            res.setdefault("clean_mse", {}).setdefault(name, {})[f"{ds}|ch{ch}"] = {
                "median": float(np.median(e0)), "mean": float(e0.mean())}
        print(f"[{tag}] {ds}|ch{ch} clean mse " + " ".join(
            f"{m}:{np.median(base[m]):.4g}" for m in models), flush=True)

        for mech in MECHS:
            for rate in RATES:
                t0 = time.time()
                clean, gt, mt, fills, cov, covc, mc = cell_arrays(
                    X, st, ch, nb, mech, rate, len(st))
                for q_r in Q_R:
                    for q_c in Q_C:
                        cv = cov if q_c == "clean" else covc
                        for name, model in models.items():
                            if name == "vanilla":
                                y = model.fc(fills[q_r], cv)
                            else:
                                y = fc_dual(model, fills[q_r], cv, mt, mc,
                                            MODEL_IFACE[name])
                            e = ((y - gt) ** 2).mean(1)
                            med, mean, n_ok, r = rel_median(e, base[name])
                            key = (f"{ds}|ch{ch}|{mech}|{rate}|{q_r}|{q_c}|{name}")
                            res.setdefault("cells", {})[key] = {
                                "median": med, "mean": mean, "n": n_ok,
                                "neighbours": nb,
                                "w": [round(float(x), 6) for x in r]}
                print(f"[{tag}] {ds}|ch{ch}|{mech}|{rate} done "
                      f"({time.time() - t0:.0f}s)", flush=True)
                save(res)
    return res


def save(res, path=None):
    path = path or OUT
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, path)


# ------------------------------------------------------------------ G0 ------

def run_g0(dev, n_win):
    """Reproduce S31 conditions (c)/(d) at block 0.7 within 10% of the stored
    anchors, using S31's exact code path (pipeline + past_covariates)."""
    import pandas as pd
    anchors = json.load(open(S31_ANCHORS))["cells"]
    model = s31.C2(dev=dev)
    g0, allpass = {}, True
    for ds in ("weather", "ETTh1", "ETTm1"):
        path = s25.DATASETS[ds]
        Xall = pd.read_csv(path).drop(columns=["date"]).to_numpy(np.float32)
        nb_map = s31.neighbours(Xall, int(0.8 * len(Xall)))
        X, st = s25.load_windows(path, n_win, s25.SEED, H)
        for ch in s31_channels(Xall):
            nb = nb_map[ch]
            cl, fl, nn_, cov, gt, m = s31.build(X, st, ch, nb, "block", 0.7)
            e = lambda p: ((p - gt) ** 2).mean(1)
            a = e(model.fc(cl))
            ok = a > 1e-12
            c_new = float(np.median(e(model.fc(fl, cov))[ok] / a[ok]))
            d_new = float(np.median(e(model.fc(nn_, cov))[ok] / a[ok]))
            key = f"{ds}|ch{ch}|block|0.7"
            c_st = anchors[key]["holed_multi"]
            d_st = anchors[key]["nan_multi"]
            pc = abs(c_new / c_st - 1.0) <= G0_TOL
            pd_ = abs(d_new / d_st - 1.0) <= G0_TOL
            g0[key] = {"c_stored": c_st, "c_new": c_new, "c_dev": c_new / c_st - 1.0,
                       "d_stored": d_st, "d_new": d_new, "d_dev": d_new / d_st - 1.0,
                       "pass": bool(pc and pd_)}
            allpass &= pc and pd_
            print(f"  G0 {key:24s} (c) {c_new:.4f} vs {c_st:.4f} "
                  f"({100 * (c_new / c_st - 1):+5.1f}%)  (d) {d_new:.4f} vs "
                  f"{d_st:.4f} ({100 * (d_new / d_st - 1):+5.1f}%)  "
                  f"{'ok' if pc and pd_ else 'FAIL'}", flush=True)
    return {"tolerance": G0_TOL, "n_win": n_win, "pass": bool(allpass),
            "cells": g0}


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="weather,ETTh1,ETTm1")
    ap.add_argument("--models", default="vanilla,ctl5k,cpt")
    ap.add_argument("--n-win", type=int, default=150)
    ap.add_argument("--max-ch", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--g0-only", action="store_true")
    ap.add_argument("--tag", default="s70")
    ap.add_argument("--out", default=None,
                    help="output JSON path (default: s70_rcprobe.json in THIS "
                         "directory); shards must write disjoint files, merged "
                         "afterwards")
    args = ap.parse_args()
    if args.out:
        global OUT
        OUT = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    assert not torch.backends.cuda.matmul.allow_tf32, \
        "TF32 got re-enabled by an import -- eval must be strict fp32"
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    dev = args.device

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {
        "design": "s70_rcprobe/DESIGN.md",
        "grid": "3 ds x 6 targets x {mcar,block} x {0.3,0.7} x q_r {zero,"
                "linear,oracle} x q_c {clean,corrupt} x 3 models",
        "n_win": args.n_win, "K_neighbours": K_NEIGH, "H": H, "L": L,
        "seed": s25.SEED, "dtype": "fp32 strict",
        "relMSE": "per-window vs each model's OWN clean univariate forecast; "
                  "median of per-window ratios (S31 convention)",
        "neighbours": "K=4 most target-correlated channels, correlation on the "
                      "first 80% of the timeline only (S31 no-leak rule)",
        "corrupt_context": "neighbours holed at the same rate/mechanism with "
                           "independent masks (mask_seed=1), zero-filled",
        "conventions": {
            "vanilla": "stock pipeline (S31 code path); fills SILENT (plain: "
                       "content present, positions unmarked) -- the native "
                       "interface cannot carry a content-preserving declaration",
            "ctl5k": "dual interface, fills DECLARED (content + flag) for the "
                     "target and for corrupt neighbours (S69 dual semantics)",
            "cpt": "same as ctl5k"},
        "checkpoints": {
            "vanilla": "models_local/chronos-2 (stock)",
            "ctl5k": "s65_fairbudget/s65_ckpt/c2_nomiss.pt (5,000 CPT steps, "
                     "no-missingness filtered regime, dual) -- the fair-budget "
                     "control behind the s65 c2 leaderboard",
            "cpt": "s60_c2retrofit/s60_ckpt/arm_CPT_s20260903.pt (seed "
                   "20260903, dual) -- the checkpoint behind "
                   "s60_leaderboard_s20260903.json"},
    })

    if args.g0_only:
        g0 = run_g0(dev, args.n_win)
        res["g0"] = g0
        save(res)
        print(f"\nG0 {'PASS' if g0['pass'] else 'FAIL'} "
              f"(tol {G0_TOL:.0%}, n_win={args.n_win})", flush=True)
        sys.exit(0 if g0["pass"] else 1)

    models = {}
    for name in args.models.split(","):
        if name == "vanilla":
            models[name] = s31.C2(dev=dev)
        else:
            models[name] = load_dual(name, dev)
    print(f"[{args.tag}] loaded models: {list(models)}", flush=True)

    for ds in args.datasets.split(","):
        res = run_dataset(ds, models, args.n_win, res, args.max_ch, args.tag)
        save(res)
    save(res)
    print(f"[{args.tag}] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
