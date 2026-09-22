#!/usr/bin/env python
"""S61 interface wrapper for Moirai 2.0 R-small (the RETAINING family, third
retrofit target).

Moirai 2.0's input tokens are cat([scaled_target, observed_mask]): fill content
AND the missingness flag enter the content path together -- the interface already
retains the fill, so no interface surgery is needed. Conventions:

  declared / nan   content at missing positions zeroed (nan_to_num), flag = 0
                   there (observed_mask = ~miss). This is the declaration
                   endpoint of the paper's recipe in Moirai 2.0's own convention,
                   and it is exactly the s46/s57 'nan' conv.
  plain            content as given (fill values), flag = 1 everywhere.

Eval forward: Moirai2Forecast.forward (the released inference module, quantiles
deterministic) -> median quantile. Training forward: Moirai2Module in
training_mode + the released PackedQuantileMAELoss (uni2ts.loss.packed) on the
map the model actually uses at inference with H=64 (last context token, its
num_predict_token=4 slots -> the 4 future tokens), i.e. exactly
structure_multi_predict's branch per_var_predict_token == num_predict_token.
"""
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CKPT = os.path.join(ROOT, "models_local", "moirai-2.0-R-small")

PATCH = 16           # module patch_size (checkpoint config)
NPT = 4              # num_predict_token
NQ = 9               # len(quantile_levels); median = index 4


def load_stock(dev, train=False):
    """Moirai2Forecast wrapping the stock module (fp32). train=True -> grads on."""
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    fc = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(CKPT),
        prediction_length=64, target_dim=1, feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0, context_length=512).to(dev)
    if train:
        fc.train()
        fc.module.train()
        for p in fc.module.parameters():
            p.requires_grad_(True)
    else:
        fc.eval()
        for p in fc.module.parameters():
            p.requires_grad_(False)
    return fc


def conv_ctx(ctx, miss, conv):
    """Returns (content [n, L] float32, observed [n, L] bool) for a convention."""
    ctx = np.asarray(ctx, np.float32)
    if conv in ("declared", "nan"):
        obs = ~miss
        return np.nan_to_num(ctx), obs
    if conv == "plain":
        return np.nan_to_num(ctx), np.ones_like(ctx, bool)
    raise ValueError(conv)


@torch.no_grad()
def median_fwd(fc, content, obs, horizon=64, batch=512):
    """Median forecast [n, horizon] through the released inference forward."""
    outs = []
    for i in range(0, len(content), batch):
        c = torch.from_numpy(np.asarray(content[i:i + batch], np.float32)).to(
            fc.device).unsqueeze(-1)
        o = torch.from_numpy(np.asarray(obs[i:i + batch], bool)).to(
            fc.device).unsqueeze(-1)
        pad = torch.zeros(c.shape[:2], dtype=torch.bool, device=fc.device)
        q = fc(past_target=c, past_observed_target=o, past_is_pad=pad)
        outs.append(q[:, NQ // 2, :].float().cpu().numpy())
    return np.concatenate(outs)[:, :horizon]


def train_forward(fc, content, obs, fut):
    """Native Moirai 2.0 training loss on one batch.

    content [B, 512] float32 (fill values at missing positions, zeros at declared
    ones), obs [B, 512] bool (False = missing/declared), fut [B, 64] float32
    (fully observed horizon). Runs the module in training_mode and applies the
    released PackedQuantileMAELoss to the inference map: last context token's
    NPT slots -> the NPT future tokens (H = NPT*PATCH = 64).
    """
    from uni2ts.loss.packed import PackedQuantileMAELoss
    dev = fc.device
    pt = torch.as_tensor(content, dtype=torch.float32, device=dev).unsqueeze(-1)
    po = torch.as_tensor(obs, dtype=torch.bool, device=dev).unsqueeze(-1)
    pad = torch.zeros(pt.shape[:2], dtype=torch.bool, device=dev)
    ft = torch.as_tensor(fut, dtype=torch.float32, device=dev).unsqueeze(-1)
    fo = torch.ones_like(ft, dtype=torch.bool)
    (target, observed_mask, sample_id, time_id, variate_id,
     prediction_mask) = fc._convert(
         fc.module.patch_size, pt, po, pad,
         future_target=ft, future_observed_target=fo)
    preds, scaled_target = fc.module(
        target, observed_mask, sample_id, time_id, variate_id, prediction_mask,
        training_mode=True)
    B = pt.shape[0]
    n_ctx = 512 // PATCH
    # [B, seq, NPT*NQ*PATCH] -> last context token's NPT slots -> [B, NPT, NQ*PATCH]
    preds = preds.view(B, -1, NPT, NQ, PATCH)[:, n_ctx - 1]        # [B, NPT, NQ, PATCH]
    pred = preds.reshape(B, NPT, NQ * PATCH)
    tgt = scaled_target[:, n_ctx:n_ctx + NPT, :]                   # [B, NPT, PATCH]
    pmask = torch.ones(B, NPT, dtype=torch.bool, device=dev)
    omask = torch.ones(B, NPT, PATCH, dtype=torch.bool, device=dev)
    sid = torch.zeros(B, NPT, dtype=torch.long, device=dev)
    vid = torch.zeros(B, NPT, dtype=torch.long, device=dev)
    return PackedQuantileMAELoss()(pred, tgt, pmask, omask, sid, vid)
