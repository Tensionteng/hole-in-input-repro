#!/usr/bin/env python
"""S512: does the retrofit's content-reading show up in the attention maps?

The permutation probe (S30 lineage) proves behaviourally that P5's output depends
on the fill content while P0's does not. This experiment looks INSIDE: encoder
self-attention weights over patches, for stock (P0) vs retrofit (P5), on the same
windows, at the two ends of the fill-quality axis (alpha=0 linear, alpha=1 oracle).

Quantities per window (bolt-tiny: patch 16, L=512 -> 32 patches + 1 reg token):
  share_miss = attention received by patches CONTAINING masked points, averaged
               over layers/heads/query positions (reg token excluded), divided by
               the share they would get under uniform attention
Reported: median share ratio per arm per alpha, on 3 datasets x 20 windows, and a
saved heatmap figure for one example window.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s27_interface"))
sys.path.insert(0, os.path.join(EXP, "s35_breadth"))
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import run_s27_interface as s27
import run_s35_breadth as s35
import train_s45 as t45
import eval_s45 as e45

L, H = s25.L, 64
PATCH = 16
DS3 = ("ETTh1", "ETTm1", "weather")


@torch.no_grad()
def attn_share(model, ctx_np, obs_np, iface):
    """Attention share received by masked patches / uniform share, per window."""
    dev = next(model.parameters()).device
    c = torch.from_numpy(ctx_np).to(dev)
    o = torch.from_numpy(obs_np).to(dev)
    # replicate the encode path up to the encoder call, with attentions on
    cfg = model.chronos_config
    mask = o.to(c.dtype)
    context, loc_scale = model.instance_norm(c)
    context = context.to(model.dtype)
    pc = model.patch(context)
    pm = torch.nan_to_num(model.patch(mask.to(model.dtype)), nan=0.0)
    if iface == "native":
        pc = torch.where(pm > 0.0, pc, 0.0)
    pc = torch.cat([pc, pm], dim=-1)
    embeds = model.input_patch_embedding(pc)
    attn_mask = pm.sum(dim=-1) > 0
    if cfg.use_reg_token:
        reg = torch.full((c.shape[0], 1), model.config.reg_token_id,
                         device=embeds.device)
        embeds = torch.cat([model.shared(reg), embeds], dim=-2)
        attn_mask = torch.cat([attn_mask.to(model.dtype),
                               torch.ones(reg.shape[0], 1, dtype=model.dtype,
                                          device=embeds.device)], dim=-1)
    out = model.encoder(attention_mask=attn_mask, inputs_embeds=embeds,
                        output_attentions=True)
    n_patch = L // PATCH
    miss_patch = (pm.float().mean(dim=-1) < 1.0).cpu().numpy()      # [B, n_patch]
    shares = []
    for att in out.attentions:                      # per layer [B, heads, q, kv]
        a = att.float().cpu().numpy()
        a = a[:, :, 1:, 1:]                          # drop reg token q/kv
        a = a.mean(axis=(1, 2))                      # [B, kv] mean over heads & queries
        shares.append(a)
    shares = np.stack(shares, axis=0)                # [n_layer, B, kv]
    per_layer = []
    for li in range(shares.shape[0]):
        a = shares[li]
        a = a / a.sum(axis=1, keepdims=True)
        uni = 1.0 / a.shape[1]
        got = []
        for i in range(len(a)):
            m = miss_patch[i]
            if m.any() and (~m).any():
                got.append(a[i, m].mean() / uni)
        per_layer.append(np.array(got))
    a = np.mean(shares, axis=0)                      # [B, kv] mean over layers
    a = a / a.sum(axis=1, keepdims=True)
    uni = 1.0 / a.shape[1]
    got = []
    for i in range(len(a)):
        m = miss_patch[i]
        if m.any() and (~m).any():
            got.append(a[i, m].mean() / uni)
    return np.array(got), per_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="P0,P5")
    ap.add_argument("--n-win", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(HERE, "s512_attn.json"))
    ap.add_argument("--layers-out", default=os.path.join(HERE, "s512_attn_layers.json"),
                    help="per-layer share ratios are always written here")
    args = ap.parse_args()
    dev = args.device

    res = {"meta": {"patch": PATCH, "L": L, "mech": "block", "rate": 0.5,
                    "datasets": DS3, "alphas": [0.0, 1.0], "n_win": args.n_win,
                    "note": "share ratio = attention on masked patches / uniform"}}
    layers = {"meta": {"note": "per-layer median/mean share ratio; layer 0 = first encoder block"}}
    models = {a: e45.load_arm(a, dev) for a in args.arms.split(",")}
    example = {}
    for ds in DS3:
        X, st = s35.load(ds, args.n_win)
        clean, lin, mask, gt = s25.build_eval_batch(X, st, "block", 0.5, H,
                                                    fill="linear")
        for al in (0.0, 1.0):
            ctx = e45.blend(clean, lin, mask, al)
            obs = (~mask).astype(np.float32)
            for a, m in models.items():
                iface = e45.ARM_IFACE[a.partition("@")[0]]
                r, pl = attn_share(m, ctx, obs, iface)
                res[f"{ds}|a={al}|{a}"] = {
                    "median_share_ratio": float(np.median(r)),
                    "mean_share_ratio": float(r.mean()), "n": int(len(r))}
                layers[f"{ds}|a={al}|{a}"] = {
                    "median": [float(np.median(x)) for x in pl],
                    "mean": [float(x.mean()) for x in pl],
                    "n": [int(len(x)) for x in pl]}
                print(f"  {ds:8s} a={al} {a:3s} share-ratio med={np.median(r):.3f} "
                      f"mean={r.mean():.3f}", flush=True)
                if ds == "ETTh1" and al == 1.0 and a in ("P0", "P5"):
                    example[a] = {"ctx": ctx[0].tolist(), "mask": mask[0].tolist()}
    json.dump(res, open(args.out, "w"), indent=1)
    json.dump(layers, open(args.layers_out, "w"), indent=1)
    print("\nwrote", args.out, "and", args.layers_out)


if __name__ == "__main__":
    main()
