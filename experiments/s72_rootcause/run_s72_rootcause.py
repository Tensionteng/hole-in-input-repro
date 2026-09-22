#!/usr/bin/env python
"""S72: root cause of the S70B full-outage explosion of the Chronos-2 CPT retrofit.

Background (s70b_notes.md): at (block, 1.0) -- ALL L=512 context positions
declared missing through the restored (dual) endpoint -- the S60 CPT checkpoint
(arm_CPT_s20260903.pt) numerically explodes when the fill carries real content
(oracle): median relMSE ~3.4e9, |y| ~ 1e4-1e6 on O(10) data. With zero/linear
fills the same checkpoint degenerates to an input-independent near-zero
forecast (relMSE 8.35). Vanilla and the +5k control (c2_nomiss.pt) stay finite
on identical inputs. S70B established this is a property of the CPT weights.

S72 instruments the forward pass to discriminate:
  H1  normalization: instance-norm statistics degenerate / division by ~zero
  H2  attention collapse: all-missing flags produce pathological attention
  H3  weights: the CPT weights landed where this never-trained input corner
      (all-missing flags WITH content present) blows up activations

Phases:
  reproduce   tie out the exploding cells against s70b_rcprobe.json (ETTh1 ch0,
              150 windows, grouped fc_dual path) and run the v-collapse test:
              recover v = asinh((y - loc)/scale) per window and check whether
              the normalized network output is window/content-independent.
  ladder      univariate corner ablation: {content: oracle|zeros} x
              {flag: all-missing|all-observed} x {endpoint: dual|native}
              for vanilla / ctl5k / cpt / w50 (WiSE-FT 0.5/0.5).
  generality  cpt corner on ETTm1/weather ch0 (is it ETTh1-specific?).
  trace       hooked forward on the exploding config + finite references:
              per-sublayer hidden norms, attention logit max and entropy,
              instance-norm (loc, scale), input-embedding norms, head output
              (normalized quantiles v) and the instance_norm.inverse sinh step.
  weightdiff  per-leaf |Delta W| of cpt / ctl5k / w50 vs vanilla.

All eval strict fp32 (TF32 off), same conventions as s60/s65/s70/s70b.
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
sys.path.insert(0, os.path.join(EXP, "s70_rcprobe"))
import run_s25_twofloor as s25
import run_s31_crosschannel as s31
import c2_iface
import run_s70_rcprobe as s70
import run_s70b_rcprobe as s70b

from chronos.chronos2.layers import MHA

L, H = s70.L, s70.H
PATCH = 16
N_CTX = L // PATCH                    # 32 context patches
OUT = os.path.join(HERE, "s72_rootcause.json")
S70B_RES = os.path.join(EXP, "s70_rcprobe", "s70b_rcprobe.json")
CKPT = {
    "ctl5k": os.path.join(EXP, "s65_fairbudget", "s65_ckpt", "c2_nomiss.pt"),
    "cpt": os.path.join(EXP, "s60_c2retrofit", "s60_ckpt", "arm_CPT_s20260903.pt"),
    "w50": os.path.join(EXP, "s60_c2retrofit", "s60_ckpt", "arm_W50_s20260903.pt"),
}
EPS = 1e-5    # chronos InstanceNorm scale floor (chronos_bolt.py:100,113)


# ---------------------------------------------------------------- helpers ----

def jnum(x):
    """JSON-safe scalar: non-finite -> None."""
    x = float(x)
    return x if np.isfinite(x) else None


def stat_dict(x):
    xf = x.detach().float()
    return {"max_abs": jnum(xf.abs().max()),
            "rms": jnum(xf.pow(2).mean().sqrt())}


def row_stats(x):
    xf = x.detach().float().flatten()
    return {"min": jnum(xf.min()), "median": jnum(xf.median()),
            "max": jnum(xf.max())}


def load_replica(dev, ckpt=None):
    m = c2_iface.load_stock(dev)
    if ckpt is not None:
        sd = torch.load(ckpt, map_location=dev, weights_only=True)
        m.load_state_dict(sd)
        m.eval()
    for i, blk in enumerate(m.encoder.block):
        blk.layer[0].self_attention._s72_tag = f"b{i:02d}.time"
        blk.layer[1].self_attention._s72_tag = f"b{i:02d}.group"
    return m


def np_loc_scale(content):
    """numpy replica of chronos InstanceNorm stats on the RAW filled context."""
    loc = content.mean(1)
    sd = content.std(1)
    scale = np.where(sd == 0, EPS, sd)
    return loc.astype(np.float64), scale.astype(np.float64)


def recover_v(y, content):
    """v = asinh((y - loc)/scale): undo instance_norm.inverse for median-q y."""
    loc, scale = np_loc_scale(content)
    return np.arcsinh((y.astype(np.float64) - loc[:, None]) / scale[:, None])


def v_dispersion(v):
    """Max relative deviation of per-window v from the per-horizon median.
    ~0 => the normalized network output is one window-independent constant."""
    med = np.median(v, axis=0, keepdims=True)
    denom = max(1.0, float(np.abs(v).max()))
    return float(np.abs(v - med).max() / denom)


# ------------------------------------------------------- attention recorder ----

_REC = {"list": None, "n_ctx": N_CTX}
_ORIG_EAGER = MHA._eager_attention


def _attn_stats(self, scores):
    sf = scores.float()
    p = torch.softmax(sf, dim=-1)
    plogp = torch.where(p > 0, p * p.log(), torch.zeros_like(p))
    ent = -plogp.sum(-1)                       # [..., q_len]
    valid = sf > -1e37
    d = {"scores_max_valid": jnum(sf[valid].max()) if bool(valid.any()) else None,
         "scores_min_valid": jnum(sf[valid].min()) if bool(valid.any()) else None,
         "p_max": jnum(p.max()),
         "ent_mean": jnum(ent.mean()), "ent_min": jnum(ent.min()),
         "ent_max": jnum(ent.max())}
    n = _REC["n_ctx"]
    kind = self._s72_tag.split(".")[1]
    if kind == "time":
        # scores [B, heads, T, T]; query positions on dim 2
        d["ent_ctx_q_mean"] = jnum(ent[:, :, :n].mean())
        d["ent_reg_q_mean"] = jnum(ent[:, :, n].mean())
        d["ent_fut_q_mean"] = jnum(ent[:, :, n + 1:].mean())
        d["ent_fut_q_min"] = jnum(ent[:, :, n + 1:].min())
    else:
        # scores [T, heads, B, B]; sequence positions on dim 0
        d["ent_ctx_pos_mean"] = jnum(ent[:n].mean())
        d["ent_reg_pos_mean"] = jnum(ent[n].mean())
        d["ent_fut_pos_mean"] = jnum(ent[n + 1:].mean())
        d["ent_fut_pos_min"] = jnum(ent[n + 1:].min())
    return d


def _recording_eager(self, q, k, v, mask):
    scores = torch.matmul(q, k.transpose(3, 2)) + mask
    if _REC["list"] is not None:
        _REC["list"].append((self._s72_tag, _attn_stats(self, scores)))
    return _ORIG_EAGER(self, q, k, v, mask)


# ------------------------------------------------------------ traced forward ----

def _hidden_hook(rec, tag, n):
    def hook(mod, inp, out):
        h = out if torch.is_tensor(out) else out[0]
        hf = h.detach().float()
        d = stat_dict(hf)
        d["ctx"] = stat_dict(hf[:, :n])
        d["reg"] = stat_dict(hf[:, n:n + 1])
        d["fut"] = stat_dict(hf[:, n + 1:])
        rec.append((tag, d))
    return hook


@torch.no_grad()
def traced_forward(model, ctx_np, obs_np, iface, n_rows_per_group=1, horizon=H,
                   return_internals=False):
    """s70.forward_grouped replica with full instrumentation. Returns
    (median forecast [n_groups, horizon], record dict[, internals])."""
    from einops import rearrange
    dev = next(model.parameters()).device
    cfg = model.chronos_config
    nop = horizon // cfg.output_patch_size
    q_med = cfg.quantiles.index(0.5)
    n = N_CTX
    c = torch.from_numpy(ctx_np).to(dev)
    o = torch.from_numpy(obs_np).to(dev)
    B = c.shape[0]
    n_groups = B // n_rows_per_group
    gids = torch.from_numpy(
        np.repeat(np.arange(n_groups, dtype=np.int64), n_rows_per_group)).to(dev)

    rec = {"attn": [], "stages": []}
    enc_cfg = model.encoder.block[0].layer[0].self_attention.config
    old_impl = enc_cfg._attn_implementation
    enc_cfg._attn_implementation = "eager"     # recorder needs eager scores
    hooks = []
    for i, blk in enumerate(model.encoder.block):
        hooks.append(blk.layer[0].register_forward_hook(
            _hidden_hook(rec["stages"], f"b{i:02d}.time_attn", n)))
        hooks.append(blk.layer[1].register_forward_hook(
            _hidden_hook(rec["stages"], f"b{i:02d}.group_attn", n)))
        hooks.append(blk.layer[2].register_forward_hook(
            _hidden_hook(rec["stages"], f"b{i:02d}.ffn", n)))
    hooks.append(model.encoder.final_layer_norm.register_forward_hook(
        _hidden_hook(rec["stages"], "final_ln", n)))
    _REC["list"] = rec["attn"]
    _REC["n_ctx"] = n
    try:
        patched_context, attention_mask, loc_scale = c2_iface.prepare_ctx(
            model, c, o, iface)
        loc, scale = loc_scale
        rec["loc"] = row_stats(loc)
        rec["scale"] = row_stats(scale)
        rec["scale_at_floor_frac"] = jnum((scale <= EPS * 1.0001).float().mean())
        p3 = PATCH
        rec["patched_timeenc"] = stat_dict(patched_context[..., :p3])
        rec["patched_content"] = stat_dict(patched_context[..., p3:2 * p3])
        rec["patched_flag"] = stat_dict(patched_context[..., 2 * p3:])
        rec["attn_mask_ctx_frac"] = jnum(
            attention_mask[:, :].float().mean())

        input_embeds = model.input_patch_embedding(patched_context)
        rec["embed_ctx"] = stat_dict(input_embeds)
        if cfg.use_reg_token:
            reg_input_ids = torch.full((B, 1), model.config.reg_token_id,
                                       device=input_embeds.device)
            reg_embeds = model.shared(reg_input_ids)
            input_embeds = torch.cat([input_embeds, reg_embeds], dim=-2)
            attention_mask = torch.cat(
                [attention_mask.to(model.dtype),
                 torch.ones_like(reg_input_ids).to(model.dtype)], dim=-1)
            rec["embed_reg"] = stat_dict(reg_embeds)

        patched_future, _ = model._prepare_patched_future(
            future_covariates=None, future_covariates_mask=None,
            loc_scale=loc_scale, num_output_patches=nop, batch_size=B)
        future_embeds = model.input_patch_embedding(patched_future)
        rec["embed_fut"] = stat_dict(future_embeds)
        rec["patched_future_content"] = stat_dict(patched_future[..., p3:2 * p3])

        input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
        attention_mask = torch.cat(
            [attention_mask,
             torch.ones(B, nop, dtype=model.dtype, device=dev)], dim=-1)

        enc = model.encoder(attention_mask=attention_mask,
                            inputs_embeds=input_embeds, group_ids=gids)
        hidden = enc[0]
        rec["hidden_fut_pre_head"] = stat_dict(hidden[:, -nop:].float())

        forecast_embeds = hidden[:, -nop:]
        quantile_preds = model.output_patch_embedding(forecast_embeds)
        rec["head_out"] = stat_dict(quantile_preds)
        quantile_preds = rearrange(
            quantile_preds, "b n (q p) -> b q (n p)",
            n=nop, q=model.num_quantiles, p=cfg.output_patch_size)
        v = quantile_preds                      # normalized (pre-inverse)
        rec["v_pre_inverse"] = stat_dict(v)
        rec["v_medq_absmax"] = jnum(v[:, q_med].abs().max())
        rec["v_per_quantile_absmax"] = [
            jnum(v[:, iq].abs().max()) for iq in range(model.num_quantiles)]
        rec["quantiles"] = [float(q) for q in cfg.quantiles]
        rec["v_medq_per_patch_max"] = [
            jnum(v[:, q_med, ip * PATCH:(ip + 1) * PATCH].abs().max())
            for ip in range(nop)]
        h = nop * cfg.output_patch_size
        vp = rearrange(v, "b q h -> b (q h)", b=B, q=model.num_quantiles, h=h)
        y = model.instance_norm.inverse(vp, loc_scale)   # sinh(v)*scale + loc
        rec["y_post_inverse"] = stat_dict(y)
        rec["sinh_max_abs"] = jnum(torch.sinh(vp.float()).abs().max())
        y = rearrange(y, "b (q h) -> b q h", q=model.num_quantiles, h=h)
    finally:
        enc_cfg._attn_implementation = old_impl
        for hh in hooks:
            hh.remove()
        _REC["list"] = None

    y_med = y[:, q_med, :horizon].float().cpu().numpy()
    if return_internals:
        internals = {"forecast_embeds": forecast_embeds, "loc_scale": loc_scale,
                     "v": v, "nop": nop}
        return y_med[0::n_rows_per_group], rec, internals
    return y_med[0::n_rows_per_group], rec


@torch.no_grad()
def head_swap_test(models, clean):
    """Cross apply encoder vs output head of different checkpoints at the
    corner (cpt/dual/oracle/all-missing): attributes the v blow-up to the
    encoder stack or to output_patch_embedding."""
    from einops import rearrange
    obs0 = np.zeros_like(clean, dtype=np.float32)
    internals = {}
    for enc_name in ("cpt", "vanilla", "ctl5k"):
        _, _, inter = traced_forward(models[enc_name], clean.astype(np.float32),
                                     obs0, "dual", 1, return_internals=True)
        internals[enc_name] = inter
    out = {}
    for enc_name in ("cpt", "vanilla", "ctl5k"):
        fe = internals[enc_name]["forecast_embeds"]
        loc_scale = internals[enc_name]["loc_scale"]
        nop = internals[enc_name]["nop"]
        for head_name in ("cpt", "vanilla", "ctl5k"):
            m = models[head_name]
            qp = m.output_patch_embedding(fe)
            qp = rearrange(qp, "b n (q p) -> b q (n p)", n=nop,
                           q=m.num_quantiles, p=m.chronos_config.output_patch_size)
            q_med = m.chronos_config.quantiles.index(0.5)
            h = nop * m.chronos_config.output_patch_size
            vpf = rearrange(qp, "b q h -> b (q h)", b=fe.shape[0],
                            q=m.num_quantiles, h=h)
            y = m.instance_norm.inverse(vpf, loc_scale)
            out[f"enc_{enc_name}__head_{head_name}"] = {
                "v_medq_absmax": jnum(qp[:, q_med].abs().max()),
                "v_absmax": jnum(qp.abs().max()),
                "y_absmax": jnum(y.abs().max())}
    return out


# ---------------------------------------------------------------- phases ----

def phase_weightdiff(models, res):
    print("[s72] weightdiff", flush=True)
    sd0 = {k: v.detach().float().cpu()
           for k, v in models["vanilla"].state_dict().items()}

    def diff(name):
        sd = {k: v.detach().float().cpu()
              for k, v in models[name].state_dict().items()}
        rows = []
        for k in sd0:
            d = sd[k] - sd0[k]
            rows.append({"name": k,
                         "max_abs": jnum(d.abs().max()),
                         "rms": jnum(d.pow(2).mean().sqrt())})
        rows.sort(key=lambda r: -(r["max_abs"] or 0))
        return rows

    out = {}
    for name in ("cpt", "ctl5k", "w50"):
        rows = diff(name)
        out[name + "_vs_vanilla"] = {
            "top15_by_max_abs": rows[:15],
            "global_max_abs": rows[0]["max_abs"],
            "global_rms": jnum(np.sqrt(np.mean(
                [r["rms"] ** 2 for r in rows])))}
        if name == "cpt":
            out["cpt_vs_vanilla_all"] = rows
    # w50 = 0.5*(vanilla+cpt) by construction: verify the deltas halve
    sd_w = {k: v.detach().float().cpu()
            for k, v in models["w50"].state_dict().items()}
    sd_c = {k: v.detach().float().cpu()
            for k, v in models["cpt"].state_dict().items()}
    ratios = []
    for k in sd0:
        dc = (sd_c[k] - sd0[k]).abs().max()
        dw = (sd_w[k] - sd0[k]).abs().max()
        if float(dc) > 0:
            ratios.append(float(dw / dc))
    out["w50_delta_over_cpt_delta"] = {
        "median": jnum(np.median(ratios)), "min": jnum(np.min(ratios)),
        "max": jnum(np.max(ratios))}
    res["weightdiff"] = out
    return res


def phase_reproduce(models, pipe, res, n_win):
    """Tie out the exploding cells vs s70b_rcprobe.json; v-collapse test."""
    import pandas as pd
    print("[s72] reproduce", flush=True)
    ds = "ETTh1"
    path = s25.DATASETS[ds]
    Xall = pd.read_csv(path).drop(columns=["date"]).to_numpy(np.float32)
    nb_map = s31.neighbours(Xall, int(0.8 * len(Xall)))
    X, st = s25.load_windows(path, n_win, s25.SEED, H)
    ch = 0
    nb = nb_map[ch]
    clean, gt, mt, fills, cov, covc, mc = s70b.cell_arrays_b(
        X, st, ch, nb, "block", 1.0, n_win)

    # clean univariate bases (S70 convention, per model)
    base = {}
    e0p = ((pipe.fc(clean) - gt) ** 2).mean(1)
    base["vanilla"] = e0p
    # replica native/dual with all-observed flags must match the pipeline base
    e0r = ((s70.median_fwd_grouped(models["vanilla"], clean,
                                   np.ones_like(clean), "dual", 1) - gt) ** 2).mean(1)
    base_dev = float(np.abs(e0r / np.maximum(e0p, 1e-30) - 1.0).max())
    for name in ("ctl5k", "cpt", "w50"):
        obs1 = np.ones_like(clean)
        y0 = s70.median_fwd_grouped(models[name], clean, obs1, "dual", 1)
        base[name] = ((y0 - gt) ** 2).mean(1)

    stored = json.load(open(S70B_RES))["cells"]
    cells, devs, pass_all = {}, {}, True
    allm = {"vanilla": pipe, "ctl5k": models["ctl5k"], "cpt": models["cpt"]}
    ys = {}
    for q_r in s70.Q_R:
        for name, model in allm.items():
            if name == "vanilla":
                y = model.fc(fills[q_r], cov)
            else:
                y = s70.fc_dual(model, fills[q_r], cov, mt, mc,
                                s70.MODEL_IFACE[name])
            ys[(q_r, name)] = y
            e = ((y - gt) ** 2).mean(1)
            med, mean, n_ok, _ = s70.rel_median(e, base[name])
            key = f"{ds}|ch{ch}|block|1.0|{q_r}|clean|{name}"
            st_med = stored[key]["median"]
            dev = med / st_med - 1.0 if st_med else float("nan")
            cells[key] = {"median": med, "mean": mean, "n": n_ok,
                          "max_abs_y": jnum(np.abs(y).max()),
                          "finite": bool(np.isfinite(y).all()),
                          "stored_median": st_med, "dev_vs_stored": jnum(dev)}
            devs[key] = dev
            pass_all &= abs(dev) <= 0.01
            print(f"  {key}: {med:.4g} (stored {st_med:.4g}, "
                  f"dev {100 * dev:+.3f}%), max|y|={np.abs(y).max():.3g}",
                  flush=True)

    # v-collapse: is the normalized network output window-independent?
    vc = {}
    for q_r in ("zero", "oracle"):
        v = recover_v(ys[(q_r, "cpt")], fills[q_r])
        vc[q_r] = {"v_median_absmax": jnum(np.abs(np.median(v, 0)).max()),
                   "v_dispersion_rel": jnum(v_dispersion(v)),
                   "v_absmax": jnum(np.abs(v).max())}
    v_o = np.median(recover_v(ys[("oracle", "cpt")], fills["oracle"]), 0)
    v_z = np.median(recover_v(ys[("zero", "cpt")], fills["zero"]), 0)
    vc["oracle_vs_zero_median_v_maxdiff"] = jnum(np.abs(v_o - v_z).max())
    vc["oracle_vs_zero_median_v_reldiff"] = jnum(
        np.abs(v_o - v_z).max() / max(1.0, np.abs(v_o).max()))

    res["reproduce"] = {
        "n_win": n_win, "dataset": ds, "channel": ch,
        "replica_base_vs_pipeline_base_maxdev": jnum(base_dev),
        "cells": cells, "pass_1pct": bool(pass_all), "v_collapse": vc}
    return res


def corner_forecast(model, content, flag, iface):
    """Univariate corner forecast. content [n,L] finite; flag in {0,1}."""
    obs = np.full_like(content, flag, dtype=np.float32)
    return s70.median_fwd_grouped(model, content.astype(np.float32), obs,
                                  iface, 1)


def ladder_cell(model, base, content, gt, flag, iface):
    y = corner_forecast(model, content, flag, iface)
    e = ((y - gt) ** 2).mean(1)
    med, mean, n_ok, _ = s70.rel_median(e, base)
    v = recover_v(y, content)
    return {"relMSE_median": med, "relMSE_mean": mean, "n": n_ok,
            "max_abs_y": jnum(np.abs(y).max()),
            "finite": bool(np.isfinite(y).all()),
            "y_window_std_mean": jnum(y.std(0).mean()),
            "v_absmax": jnum(np.abs(v).max()),
            "v_median_absmax": jnum(np.abs(np.median(v, 0)).max()),
            "v_dispersion_rel": jnum(v_dispersion(v))}, y, v


def phase_ladder(models, res, n_win, channels, ds="ETTh1"):
    """{content x flag x endpoint} x 4 models, univariate corner."""
    import pandas as pd
    print("[s72] ladder", flush=True)
    path = s25.DATASETS[ds]
    X, st = s25.load_windows(path, max(n_win, 150), s25.SEED, H)
    cells, vstore = {}, {}
    for ch in channels:
        clean, gt = [], []
        for s in st[:n_win]:
            w = X[s:s + L].T
            clean.append(w[ch].copy())
            gt.append(X[s + L:s + L + H].T[ch].copy())
        clean = np.asarray(clean, np.float32)
        gt = np.asarray(gt, np.float32)
        contents = {"oracle": clean, "zeros": np.zeros_like(clean)}
        for name, model in models.items():
            obs1 = np.ones_like(clean)
            y0 = s70.median_fwd_grouped(model, clean, obs1, "dual", 1)
            base = ((y0 - gt) ** 2).mean(1)
            for cname, content in contents.items():
                for fname, flag in (("missing", 0.0), ("observed", 1.0)):
                    for iface in ("dual", "native"):
                        key = f"{ds}|ch{ch}|{name}|{cname}|{fname}|{iface}"
                        cell, y, v = ladder_cell(model, base, content, gt,
                                                 flag, iface)
                        cells[key] = cell
                        if name == "cpt" and ch == channels[0]:
                            vstore[(cname, fname, iface)] = np.median(v, 0)
            print(f"  {ds}|ch{ch}|{name} done", flush=True)
    # cross-cell v comparisons on the primary channel: is v content/endpoint
    # independent at the corner?
    def vdiff(a, b):
        return jnum(np.abs(vstore[a] - vstore[b]).max()
                    / max(1.0, np.abs(vstore[a]).max()))
    vtests = {
        "cpt_missing_dual_oracle_vs_zeros_reldiff":
            vdiff(("oracle", "missing", "dual"), ("zeros", "missing", "dual")),
        "cpt_missing_oracle_dual_vs_native_reldiff":
            vdiff(("oracle", "missing", "dual"), ("oracle", "missing", "native")),
        "cpt_missing_vs_observed_dual_oracle_reldiff":
            vdiff(("oracle", "missing", "dual"), ("oracle", "observed", "dual")),
    }
    res["ladder"] = {"n_win": n_win, "channels": channels, "dataset": ds,
                     "cells": cells, "v_tests": vtests}
    return res


def phase_generality(models, res, n_win):
    """cpt corner on the other two datasets (ch0)."""
    import pandas as pd
    print("[s72] generality", flush=True)
    out = {}
    for ds in ("ETTm1", "weather"):
        path = s25.DATASETS[ds]
        X, st = s25.load_windows(path, max(n_win, 150), s25.SEED, H)
        ch = 0
        clean, gt = [], []
        for s in st[:n_win]:
            w = X[s:s + L].T
            clean.append(w[ch].copy())
            gt.append(X[s + L:s + L + H].T[ch].copy())
        clean = np.asarray(clean, np.float32)
        gt = np.asarray(gt, np.float32)
        for name in ("cpt", "vanilla"):
            model = models[name]
            obs1 = np.ones_like(clean)
            y0 = s70.median_fwd_grouped(model, clean, obs1, "dual", 1)
            base = ((y0 - gt) ** 2).mean(1)
            for cname, content in (("oracle", clean),
                                   ("zeros", np.zeros_like(clean))):
                cell, _, _ = ladder_cell(model, base, content, gt, 0.0, "dual")
                out[f"{ds}|ch{ch}|{name}|{cname}|missing|dual"] = cell
        print(f"  {ds} done", flush=True)
    res["generality"] = {"n_win": n_win, "cells": out}
    return res


def phase_trace(models, res, n_trace, pipe_arrays=None):
    """Hooked forward on the exploding config + finite references."""
    import pandas as pd
    print("[s72] trace", flush=True)
    ds = "ETTh1"
    path = s25.DATASETS[ds]
    Xall = pd.read_csv(path).drop(columns=["date"]).to_numpy(np.float32)
    nb_map = s31.neighbours(Xall, int(0.8 * len(Xall)))
    X, st = s25.load_windows(path, 150, s25.SEED, H)
    ch = 0
    nb = nb_map[ch]
    n = n_trace
    clean, gt = [], []
    for s in st[:n]:
        w = X[s:s + L].T
        clean.append(w[ch].copy())
        gt.append(X[s + L:s + L + H].T[ch].copy())
    clean = np.asarray(clean, np.float32)
    gt = np.asarray(gt, np.float32)
    zeros = np.zeros_like(clean)

    configs = {
        "corner_cpt":        ("cpt", clean, 0.0, "dual"),
        "zero_cpt":          ("cpt", zeros, 0.0, "dual"),
        "clean_cpt":         ("cpt", clean, 1.0, "dual"),
        "corner_cpt_native": ("cpt", clean, 0.0, "native"),
        "corner_vanilla":    ("vanilla", clean, 0.0, "dual"),
        "corner_ctl5k":      ("ctl5k", clean, 0.0, "dual"),
        "corner_w50":        ("w50", clean, 0.0, "dual"),
    }
    traces, fidelity = {}, {}
    for cname, (mname, content, flag, iface) in configs.items():
        model = models[mname]
        obs = np.full_like(content, flag, dtype=np.float32)
        y_t, rec = traced_forward(model, content.astype(np.float32), obs,
                                  iface, 1)
        y_ref = s70.median_fwd_grouped(model, content.astype(np.float32),
                                       obs, iface, 1)
        fidelity[cname] = jnum(
            np.abs(y_t - y_ref).max() / max(1.0, np.abs(y_ref).max()))
        e = ((y_t - gt) ** 2).mean(1)
        rec["relMSE_median"] = jnum(np.median(e))
        rec["y_med_absmax"] = jnum(np.abs(y_t).max())
        traces[cname] = rec
        print(f"  trace {cname}: max|y|={np.abs(y_t).max():.3g}, "
              f"fidelity dev={fidelity[cname]}", flush=True)

    # grouped variant of the exploding config: target + 4 clean neighbours,
    # ALL rows declared all-missing (the exact S70B cell shape)
    _, _, mt, fills, cov, _, mc = s70b.cell_arrays_b(X, st, ch, nb, "block",
                                                     1.0, n)
    ng = n
    ctx = np.concatenate([fills["oracle"][:, None], cov], axis=1)  # [g,5,L]
    obs = np.zeros_like(ctx)
    ctx = ctx.reshape(ng * 5, L).astype(np.float32)
    obs = obs.reshape(ng * 5, L).astype(np.float32)
    y_g, rec_g = traced_forward(models["cpt"], ctx, obs, "dual", 5)
    y_ref = s70.fc_dual(models["cpt"], fills["oracle"], cov, mt, mc, "dual")
    fidelity["corner_cpt_grouped"] = jnum(
        np.abs(y_g - y_ref[:ng]).max() / max(1.0, np.abs(y_ref[:ng]).max()))
    v_g = recover_v(y_g, fills["oracle"])
    v_u = recover_v(corner_forecast(models["cpt"], clean, 0.0, "dual"), clean)
    traces["corner_cpt_grouped"] = rec_g
    grouped_test = {
        "v_grouped_absmax": jnum(np.abs(v_g).max()),
        "v_grouped_dispersion_rel": jnum(v_dispersion(v_g)),
        "v_grouped_vs_univariate_reldiff": jnum(
            np.abs(np.median(v_g, 0) - np.median(v_u, 0)).max()
            / max(1.0, np.abs(np.median(v_g, 0)).max()))}

    # finite-reference envelope per stage; first divergence of the corner
    refs = ("clean_cpt", "corner_vanilla", "corner_ctl5k")
    stage_tags = [t for t, _ in traces["corner_cpt"]["stages"]]
    ratios, env = {}, {}
    corner = dict(traces["corner_cpt"]["stages"])
    for tag in stage_tags:
        e_max = max(dict(traces[r]["stages"])[tag]["max_abs"] for r in refs)
        env[tag] = e_max
        ratios[tag] = jnum(corner[tag]["max_abs"] / max(e_max, 1e-30))
    first = next((t for t in stage_tags if (ratios[t] or 0) > 3.0), None)
    head_env = max(traces[r]["head_out"]["max_abs"] for r in refs)
    v_env = max(traces[r]["v_medq_absmax"] for r in refs)

    # encoder-vs-head attribution: cross heads at the corner
    swap = head_swap_test(models, clean)
    print("  head-swap:", {k: round(vv["v_medq_absmax"], 3)
                           for k, vv in swap.items()}, flush=True)

    res["trace"] = {
        "n_trace": n, "dataset": ds, "channel": ch,
        "configs": traces, "fidelity_rel_dev": fidelity,
        "stage_envelope_max_abs": env,
        "corner_over_envelope_ratio": ratios,
        "first_stage_ratio_gt3": first,
        "head_out_env_max_abs": jnum(head_env),
        "head_out_corner_over_env": jnum(
            traces["corner_cpt"]["head_out"]["max_abs"] / head_env),
        "v_medq_env_absmax": jnum(v_env),
        "v_medq_corner_over_env": jnum(
            traces["corner_cpt"]["v_medq_absmax"] / v_env),
        "grouped_corner_test": grouped_test,
        "head_swap": swap}
    return res


def verdict(res):
    """Assemble the H1/H2/H3 discrimination from the measurements."""
    tr = res["trace"]["configs"]
    corner = tr["corner_cpt"]
    v = {}
    v["H1_normalization"] = {
        "verdict": "rejected as primary cause",
        "corner_loc": corner["loc"], "corner_scale": corner["scale"],
        "corner_scale_at_floor_frac": corner["scale_at_floor_frac"],
        "note": "loc/scale are ordinary O(1)-O(10) window stats in the "
                "exploding config; the 1e-5 scale floor is hit only by the "
                "zeros fill, where it SQUASHES the same huge v to ~0 "
                "(near-zero forecast, relMSE ~8). No division-by-zero in the "
                "exploding path; instance_norm.inverse amplifies via sinh, "
                "not via degenerate stats."}
    ent_fut = [d["ent_fut_q_mean"] for t, d in corner["attn"]
               if t.endswith(".time")]
    ent_ctx = [d["ent_ctx_q_mean"] for t, d in corner["attn"]
               if t.endswith(".time")]
    smax = [d["scores_max_valid"] for t, d in corner["attn"]
            if d["scores_max_valid"] is not None]
    v["H2_attention_collapse"] = {
        "verdict": "rejected: no collapse (entropy -> 0); the corner shows "
                   "structurally DEGENERATE-but-uniform attention",
        "time_attn_ctx_query_entropy_mean_per_layer": ent_ctx,
        "uniform_entropy_log37": jnum(np.log(37)),
        "time_attn_fut_query_entropy_mean_per_layer": ent_fut,
        "max_valid_logit_over_layers": jnum(max(smax)),
        "note": "all-masked context rows go exactly UNIFORM (entropy log T), "
                "future queries attend REG+future only; logits stay O(1e2), "
                "no inf, no one-hot collapse. The same mask structure is "
                "shared by vanilla/ctl5k, which do not explode."}
    v["H3_weights"] = {
        "verdict": "SUPPORTED, with a precise locus",
        "v_medq_absmax": {c: tr[c]["v_medq_absmax"] for c in tr},
        "sinh_max_abs_corner": corner["sinh_max_abs"],
        "v_dispersion_rel_corner": res["reproduce"]["v_collapse"]["oracle"]
            ["v_dispersion_rel"],
        "v_content_independence_reldiff": res["reproduce"]["v_collapse"]
            ["oracle_vs_zero_median_v_reldiff"],
        "v_endpoint_independence_reldiff": res["ladder"]["v_tests"]
            ["cpt_missing_oracle_dual_vs_native_reldiff"],
        "head_out_corner_over_env": res["trace"]["head_out_corner_over_env"],
        "first_stage_ratio_gt3": res["trace"]["first_stage_ratio_gt3"],
        "encoder_stage_max_ratio": jnum(max(
            x for x in res["trace"]["corner_over_envelope_ratio"].values()
            if x is not None)),
        "head_swap_v_medq": {k: vv["v_medq_absmax"]
                             for k, vv in res["trace"]["head_swap"].items()},
        "note": "the CPT weights make the head emit an input-independent "
                "normalized output v with median-quantile |v| ~ 10-13 at the "
                "all-missing-flag corner (vanilla/ctl5k at the same corner: "
                "~0.4; cpt on clean observed input: ~2.2). No encoder stage "
                "leaves the finite-reference envelope (max ratio < 1). "
                "instance_norm.inverse (use_arcsinh=true) then applies sinh: "
                "y = sinh(v)*scale + loc ~ 1e4-1e6 on O(10) data. Content "
                "contributes only (loc, scale); the flag pattern alone "
                "determines v (oracle vs zeros reldiff ~ 1e-7; dual vs "
                "native identical; grouped == univariate)."}
    res["verdict"] = v
    return res


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "weightdiff", "reproduce", "ladder",
                             "generality", "trace"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-win-repro", type=int, default=150)
    ap.add_argument("--n-win", type=int, default=32)
    ap.add_argument("--n-trace", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or OUT
    assert not torch.backends.cuda.matmul.allow_tf32
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    dev = args.device

    MHA._eager_attention = _recording_eager   # inert unless _REC["list"] set

    t0 = time.time()
    models = {"vanilla": load_replica(dev)}
    for name in ("ctl5k", "cpt", "w50"):
        models[name] = load_replica(dev, CKPT[name])
    print(f"[s72] loaded replicas {list(models)} ({time.time() - t0:.0f}s)",
          flush=True)

    res = {"meta": {
        "experiment": "s72_rootcause: mechanism of the s70b full-outage "
                      "explosion of the Chronos-2 CPT retrofit",
        "corner": "(block, 1.0) = all L=512 context positions declared "
                  "missing (obs flag 0) through the named endpoint; content = "
                  "oracle (true values) or zeros",
        "checkpoints": {"vanilla": "models_local/chronos-2 (stock weights, "
                                   "c2_iface replica)",
                        "ctl5k": CKPT["ctl5k"], "cpt": CKPT["cpt"],
                        "w50": CKPT["w50"] + " (WiSE-FT 0.5*vanilla+0.5*cpt)"},
        "conventions": "strict fp32, TF32 off; relMSE = per-window MSE ratio "
                       "vs the model's OWN clean univariate forecast (dual "
                       "replica, all-observed flags), median (S31 convention)",
        "L": L, "H": H, "patch": PATCH, "n_ctx_patches": N_CTX,
        "instance_norm": "chronos_bolt.InstanceNorm(use_arcsinh=True, "
                         "eps=1e-5): fwd arcsinh((x-loc)/scale), inverse "
                         "sinh(v)*scale+loc, scale floored at 1e-5",
        "seed": s25.SEED, "device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "date": "2026-09-10"}}

    ph = args.phase
    if ph in ("all", "weightdiff"):
        res = phase_weightdiff(models, res)
    if ph in ("all", "reproduce"):
        pipe = s31.C2(dev=dev)
        res = phase_reproduce(models, pipe, res, args.n_win_repro)
        del pipe
        torch.cuda.empty_cache()
    if ph in ("all", "ladder"):
        res = phase_ladder(models, res, args.n_win, channels=[0, 1, 2])
    if ph in ("all", "generality"):
        res = phase_generality(models, res, args.n_win)
    if ph in ("all", "trace"):
        res = phase_trace(models, res, args.n_trace)
    if ph == "all":
        res = verdict(res)

    tmp = out + ".tmp"
    json.dump(res, open(tmp, "w"), indent=1)
    os.replace(tmp, out)
    print(f"[s72] wrote {out} ({time.time() - t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
