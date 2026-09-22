#!/usr/bin/env python
"""S60: interface-switchable forward for Chronos-2 (the "dual" retrofit replica).

Mirrors s27_interface/run_s27_interface.py for Chronos-Bolt. Chronos-2's stock
`_prepare_patched_context` (chronos/chronos2/model.py:392-442) discards the CONTENT
at masked positions:

    patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

and keeps only the flag, concatenated with a time encoding into the input patch
embedding. The two interfaces here:

  native   stock semantics, op-for-op: mask from the argument or from NaN; content
           zeroed where mask == 0. With a NaN context (mask=None) the InstanceNorm
           (nanmean) uses observed points only -- the declared missing-data path.
           With a finite filled context + explicit mask the stats come from the
           full (filled) context and the content is still zeroed.
  dual     the retrofit: fill CONTENT kept AND the flag kept; InstanceNorm over the
           filled context (same convention as Bolt's "dual" in S27/S45). The ONLY
           difference from native is the removed zeroing line -- and, when the
           context carries NaN, that dual requires a finite (filled) context.

Everything downstream (input_patch_embedding, [REG] token, future patches, encoder,
output head, native quantile loss) calls the model's own submodules in the stock
order, so the replica diverges from stock only by that one line.

GATE (run_s60_gate.py): on fp32, eval mode, the native replica must match
Chronos2Model.forward bit-exactly (max|Delta| = 0) on clean and holed contexts.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch
from einops import rearrange, repeat

C2_PATH = os.path.join(ROOT, "models_local", "chronos-2")


def load_stock(dev, train=False):
    """The shipped Chronos-2 checkpoint, fp32."""
    from chronos.chronos2.model import Chronos2Model
    m = Chronos2Model.from_pretrained(C2_PATH, torch_dtype=torch.float32).to(dev)
    if train:
        m.train()
        for p in m.parameters():
            p.requires_grad_(True)
    else:
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    return m


# ---------------------------------------------------------- context side ----

def prepare_ctx(model, context, context_mask, iface):
    """Replica of Chronos2Model._prepare_patched_context with the interface switch.

    context: [B, L] (NaN allowed only for native with context_mask=None).
    context_mask: [B, L] 1 = observed, or None (native only: inferred from NaN).
    Returns (patched_context [B, n, 3p], attention_mask [B, n] bool, loc_scale).
    """
    assert iface in ("native", "dual")
    cfg = model.chronos_config
    if iface == "native":
        cmask = context_mask.to(context.dtype) if context_mask is not None else \
            torch.isnan(context).logical_not().to(context.dtype)
        zero_content = True
    else:
        assert context_mask is not None, "dual needs an explicit observation mask"
        assert not torch.isnan(context).any(), "dual needs a finite (filled) context"
        cmask = context_mask.to(context.dtype)
        zero_content = False

    batch_size, context_length = context.shape
    assert context_length <= cfg.context_length, "replica does not truncate; L <= 8192"

    # scaling (identical call in both interfaces: nanmean => observed-only stats iff
    # the context carries NaN, i.e. the stock declared path)
    context, loc_scale = model.instance_norm(context)
    context = context.to(model.dtype)
    cmask = cmask.to(model.dtype)

    # patching
    patched_context = model.patch(context)
    patched_mask = torch.nan_to_num(model.patch(cmask), nan=0.0)
    if zero_content:
        # the line the retrofit removes: content discarded, flag kept
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)

    # attention_mask = 1 if at least one item in the patch is observed
    attention_mask = patched_mask.sum(dim=-1) > 0
    num_context_patches = attention_mask.shape[-1]

    # context time encoding (verbatim from the stock method)
    final_context_length = num_context_patches * cfg.input_patch_size
    context_time_enc = torch.arange(start=-final_context_length, end=0,
                                    device=model.device, dtype=torch.float32)
    context_time_enc = (
        repeat(context_time_enc, "(n p) -> b n p",
               b=batch_size, n=num_context_patches, p=cfg.input_patch_size)
        .div(cfg.time_encoding_scale)
        .to(model.dtype)
    )
    patched_context = torch.cat([context_time_enc, patched_context, patched_mask], dim=-1)
    return patched_context, attention_mask, loc_scale


# ------------------------------------------------------------ full forward ----

def forward_iface(model, context, context_mask, iface, num_output_patches,
                  future_target=None, future_target_mask=None):
    """Replica of Chronos2Model.encode+forward through the interface switch.

    Returns (quantile_preds [B, Q, num_output_patches*p] unscaled, loss or None).
    The loss is the model's OWN _compute_loss on the normalized quantiles -- the
    native Chronos-2 training loss, not a re-implementation.
    """
    cfg = model.chronos_config
    batch_size = context.shape[0]
    patched_context, attention_mask, loc_scale = prepare_ctx(
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

    patched_future, patched_future_covariates_mask = model._prepare_patched_future(
        future_covariates=None, future_covariates_mask=None, loc_scale=loc_scale,
        num_output_patches=num_output_patches, batch_size=batch_size)
    future_attention_mask = torch.ones(batch_size, num_output_patches,
                                       dtype=model.dtype, device=model.device)
    future_embeds = model.input_patch_embedding(patched_future)

    input_embeds = torch.cat([input_embeds, future_embeds], dim=-2)
    attention_mask = torch.cat([attention_mask, future_attention_mask], dim=-1)

    group_ids = torch.arange(batch_size, dtype=torch.long, device=model.device)
    encoder_outputs = model.encoder(attention_mask=attention_mask,
                                    inputs_embeds=input_embeds,
                                    group_ids=group_ids)
    hidden_states = encoder_outputs[0]
    assert hidden_states.shape == (batch_size, num_context_patches + 1
                                   + num_output_patches, model.model_dim)

    forecast_embeds = hidden_states[:, -num_output_patches:]
    quantile_preds = model.output_patch_embedding(forecast_embeds)
    quantile_preds = rearrange(
        quantile_preds, "b n (q p) -> b q (n p)",
        n=num_output_patches, q=model.num_quantiles, p=cfg.output_patch_size)

    loss = None
    if future_target is not None:
        loss = model._compute_loss(
            quantile_preds=quantile_preds, future_target=future_target,
            future_target_mask=future_target_mask,
            patched_future_covariates_mask=patched_future_covariates_mask,
            loc_scale=loc_scale, num_output_patches=num_output_patches)

    h = num_output_patches * cfg.output_patch_size
    quantile_preds = rearrange(quantile_preds, "b q h -> b (q h)",
                               b=batch_size, q=model.num_quantiles, h=h)
    quantile_preds = model.instance_norm.inverse(quantile_preds, loc_scale)
    quantile_preds = rearrange(quantile_preds, "b (q h) -> b q h",
                               q=model.num_quantiles, h=h)
    return quantile_preds, loss


# -------------------------------------------------------------- conveniences ----

def conv_ctx(ctx, miss, conv, iface):
    """(ctx_to_feed, obs_to_feed) under a convention (eval_s45.conv_ctx analogue).

    plain    fill value, flag all-ones (fill-then-feed; no declaration)
    declared fill value + the true flag (native: content zeroed by the model;
             dual: content kept) -- the interface the arm was pretrained with
    nan      NaN at missing, mask=None (stock native only: the declared NaN path)
    """
    if conv == "plain":
        return ctx, np.ones_like(ctx)
    if conv == "declared":
        return ctx, (~miss).astype(np.float32)
    if conv == "nan":
        assert iface == "native", "the NaN path exists only on the native interface"
        c = ctx.copy()
        c[miss] = np.nan
        return c, None
    raise ValueError(conv)


@torch.no_grad()
def median_fwd(model, ctx_np, obs_np, iface, horizon=64, batch=256):
    """Batched median-quantile forecast [n, horizon] through the interface switch."""
    assert horizon % model.chronos_config.output_patch_size == 0
    nop = horizon // model.chronos_config.output_patch_size
    q_med = model.chronos_config.quantiles.index(0.5)
    dev = next(model.parameters()).device
    outs = []
    for i in range(0, len(ctx_np), batch):
        c = torch.from_numpy(ctx_np[i:i + batch]).to(dev)
        o = None if obs_np is None else torch.from_numpy(obs_np[i:i + batch]).to(dev)
        q, _ = forward_iface(model, c, o, iface, nop)
        outs.append(q[:, q_med, :horizon].float().cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def quantile_fwd(model, ctx_np, obs_np, iface, horizon=64, batch=256):
    """Batched all-quantile forecast [n, Q, horizon]."""
    assert horizon % model.chronos_config.output_patch_size == 0
    nop = horizon // model.chronos_config.output_patch_size
    dev = next(model.parameters()).device
    outs = []
    for i in range(0, len(ctx_np), batch):
        c = torch.from_numpy(ctx_np[i:i + batch]).to(dev)
        o = None if obs_np is None else torch.from_numpy(obs_np[i:i + batch]).to(dev)
        q, _ = forward_iface(model, c, o, iface, nop)
        outs.append(q[:, :, :horizon].float().cpu().numpy())
    return np.concatenate(outs)
