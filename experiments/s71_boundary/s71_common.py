#!/usr/bin/env python
"""S71 shared infrastructure: per-model native-path wrappers, training forwards,
and the mechdiv-recipe mapping through each boundary model's native input path.

Models (see s71_notes.md for provenance + conventions):
  timesfm   TimesFM 2.5 200M (google/timesfm-2.5-200m-pytorch via HF mirror cache).
            Overwriting convention: the wrapper linear-interpolates every NaN before
            the module runs; the module mask channel is padding-only in the shipped
            path. Evaluated quantity: wrapper point forecast full_forecast[..., 5].
  tempopfn  TempoPFN 38M (models_local/tempopfn). Learned-token convention: NaN
            positions are replaced by nan_embedding; scaler stats observed-only.
            bf16 autocast only (fla kernel rejects fp32).
  timer     Timer base 84M (models_local/timer-base-84m; the paper's "Timer-XL"
            row). No declared path: NaN input -> NaN logits. Eval = last-token
            next-96 forecast, first 64 values, context cropped to last 480.

CPT mapping of the s53/P5 mechdiv recipe (documented in s71_notes.md sec. 3):
  clean 50% | declared 25% | alpha-blend-filled 25%, mech in 4, rate U(.05,.7).
  declared half presented natively: timesfm -> linear fill (its own interpolant),
  tempopfn -> NaN (nan_embedding), timer -> zero fill (no flag exists).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tempopfn_triton_cache")

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True   # same convention as train_s45.py
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s35_breadth"))
sys.path.insert(0, os.path.join(EXP, "s45_pretrain"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
sys.path.insert(0, os.path.join(EXP, "s49_2025models"))
sys.path.insert(0, os.path.join(ROOT, "models_local", "tempopfn"))

import run_s25_twofloor as s25
import train_s45 as t45

L, H = s25.L, 64
SEED = 20260903
CK = os.path.join(HERE, "s71_ckpt")

STEPS, WARMUP, LR, WD = 5000, 500, 1e-4, 0.01   # DESIGN: AdamW lr1e-4, 500 warmup,
                                                 # cosine to 0 (bs per model, recorded)

# ------------------------------------------------------------------ recipe ----

def s71_augment(ctx, model_key, regime, rng, wi, clean_p=0.5):
    """The s45/s53 mechdiv recipe, presented through model_key's native path.

    Returns (ctx_to_feed [L] float32, miss [L] bool). rng draw order is identical to
    train_s45.augment's mechdiv branch so the augmentation stream is the same recipe.
    """
    miss = np.zeros(L, bool)
    if regime == "filtered" or rng.random() < clean_p:
        return ctx, miss
    mech = ("mcar", "block", "mnar_high", "mnar_extreme")[int(rng.integers(4))]
    rate = float(rng.uniform(0.05, 0.7))
    mask = s25.make_mask(mech, rate, wi, int(rng.integers(1 << 30)), 1,
                         x=ctx[None])[0]
    declared = rng.random() < 0.5
    if declared:
        if model_key == "timesfm":
            # overwrite made literal: declared positions carry the model's own
            # (linear) interpolant -- the only content its shipped path ever uses
            out = t45.linear_fill(ctx, mask)
        elif model_key == "tempopfn":
            out = ctx.copy()
            out[mask] = np.nan          # native declared path (learned nan token)
        elif model_key == "timer":
            out = ctx.copy()
            out[mask] = 0.0             # values-only realisation of "declared"
        else:
            raise ValueError(model_key)
    else:
        a = float(rng.uniform(0, 1))
        lin = t45.linear_fill(ctx, mask)
        out = lin.copy()
        out[mask] = (1 - a) * lin[mask] + a * ctx[mask]
    return out.astype(np.float32), mask


def make_sched(opt, steps, warmup=WARMUP):
    def f(t):
        if t < warmup:
            return (t + 1) / warmup
        p = (t - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def train_loop(model_key, params, fwd_loss, arm, seed, bs, steps, dev,
               smoke=False, log_every=100):
    """Shared CPT loop. fwd_loss(ctx_batch, miss_batch, fut_batch) -> scalar loss.
    params = the model's trainable parameters. Returns (log dict, history)."""
    regime = "mechdiv" if arm == "cpt" else "filtered"
    steps = 40 if smoke else steps
    rng = np.random.default_rng(seed)
    sampler = t45.Sampler(seed=seed + 1)
    gen = sampler.window(require_clean_ctx=True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sch = make_sched(opt, steps)
    hist, n_skip, wi, t0 = [], 0, 0, time.time()
    for it in range(steps):
        ctxs, misss, futs = [], [], []
        while len(ctxs) < bs:
            c, f = next(gen)
            c, m = s71_augment(c, model_key, regime, rng, wi)
            wi += 1
            ctxs.append(c)
            misss.append(m)
            futs.append(f)
        loss = fwd_loss(np.stack(ctxs), np.stack(misss), np.stack(futs))
        if not torch.isfinite(loss):
            n_skip += 1
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            n_skip += 1
            opt.zero_grad()
            continue
        opt.step()
        sch.step()
        hist.append(float(loss))
        if (it + 1) % log_every == 0:
            print(f"  [{model_key}:{arm}] {it+1}/{steps} "
                  f"loss={np.mean(hist[-log_every:]):.4f} skip={n_skip} "
                  f"lr={sch.get_last_lr()[0]:.2e} ({(time.time()-t0)/60:.1f}m)",
                  flush=True)
    return {"regime": regime, "steps": steps, "bs": bs, "lr": LR, "wd": WD,
            "warmup": WARMUP, "seed": seed, "n_skip": n_skip,
            "loss_head": hist[:log_every], "loss_tail": hist[-2 * log_every:],
            "minutes": (time.time() - t0) / 60}, hist


# ----------------------------------------------------------------- TimesFM ----

TIMESFM_REPO = "google/timesfm-2.5-200m-pytorch"


def timesfm_snapshot():
    hub = os.path.join(os.environ["HF_HOME"], "hub",
                       "models--google--timesfm-2.5-200m-pytorch", "snapshots")
    snaps = sorted(os.listdir(hub))
    return os.path.join(hub, snaps[0])


def build_timesfm(dev, ckpt=None, torch_compile=False):
    """The torch module (fp32), stock or from an S71 checkpoint. For TRAINING.
    Eval uses TimesFMEval below (the exact s30/s57 wrapper path)."""
    import timesfm
    m = timesfm.TimesFM_2p5_200M_torch(torch_compile=torch_compile)
    m.load_checkpoint(timesfm_snapshot())
    if ckpt is not None:
        m.model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                           weights_only=True))
    m.model.to(dev)
    return m


def timesfm_outer(ctx_np, dev):
    """The wrapper's input preprocessing (forecast()/compiled_decode): front-pad to
    max_context=1024 with mask=True on the pad, outer revin (normalize_inputs=True,
    stats over the padded vector). Returns (mk, mu, sigma, xn)."""
    from timesfm.torch import util
    B = len(ctx_np)
    values = np.zeros((B, 1024), np.float32)
    values[:, -L:] = ctx_np
    masks = np.ones((B, 1024), bool)
    masks[:, -L:] = False
    x = torch.from_numpy(values).to(dev)
    mk = torch.from_numpy(masks).to(dev)
    mu = x.mean(-1, keepdim=True)
    sigma = x.std(-1, keepdim=True)
    xn = util.revin(x, mu, sigma, reverse=False)
    return mk, mu, sigma, xn


def timesfm_fwd_train(module, ctx_np, fut_np, dev):
    """Grad-enabled replica of decode()'s prefill (the module's own input handling:
    causal running-stats revin over observed patches, content zeroed where masked).

    NOTE: the eval wrapper's decode TAIL flags (force_flip_invariance=True,
    infer_is_positive=True, use_continuous_quantile_head=True -- the s30/s57
    ForecastConfig inherits the dataclass defaults for the first two) are eval
    post-processing, not part of the trained function; CPT trains the raw decode,
    exactly as the model's own pretraining would have. G2t verifies this replica
    against module.decode() exactly (0.0).
    Returns quantile forecast [B, H, 10] in raw space (channels 1..9 = 0.1..0.9,
    channel 5 = median = the evaluated point forecast).
    """
    from timesfm.torch import util
    revin = util.revin
    B = len(ctx_np)
    mk, mu, sigma, xn = timesfm_outer(ctx_np, dev)
    p = module.p
    pi = xn.reshape(B, -1, p)
    pm = mk.reshape(B, -1, p)
    n = torch.zeros(B, device=dev)
    mu_r = torch.zeros(B, device=dev)
    sg = torch.zeros(B, device=dev)
    mus, sgs = [], []
    for i in range(pi.shape[1]):
        (n, mu_r, sg), _ = util.update_running_stats(n, mu_r, sg, pi[:, i],
                                                     pm[:, i])
        mus.append(mu_r)
        sgs.append(sg)
    ctx_mu = torch.stack(mus, 1)
    ctx_sg = torch.stack(sgs, 1)
    normed = revin(pi, ctx_mu, ctx_sg, reverse=False)
    normed = torch.where(pm, torch.zeros_like(normed), normed)
    (_, _, out_ts, _), _ = module(normed, pm, None)
    ren = revin(out_ts, ctx_mu, ctx_sg, reverse=True)
    ren = ren.reshape(B, -1, module.o, module.q)
    pf = ren[:, -1, :H, :]                            # [B, H, 10], outer-normed space
    return revin(pf, mu, sigma, reverse=True)         # back to raw space


def timesfm_pinball(module, ctx_np, fut_np, dev):
    q = timesfm_fwd_train(module, ctx_np, fut_np, dev)
    levels = list(module.config.quantiles)            # 0.1..0.9
    qq = q[..., 1:10]                                 # channels 1..9 = the 9 quantiles
    y = torch.from_numpy(fut_np).to(dev).unsqueeze(-1)  # [B, H, 1]
    e = y - qq
    t = torch.as_tensor(levels, device=dev, dtype=qq.dtype).view(1, 1, -1)
    return torch.maximum(t * e, (t - 1) * e).mean()


class TimesFMEval:
    """The exact s30/s57 eval wrapper (compiled decode), with optional S71 ckpt."""
    name = "timesfm-2.5-200m"
    convs = ("plain", "nan")

    def __init__(self, dev, ckpt=None):
        import timesfm
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(TIMESFM_REPO)
        if ckpt is not None:
            self.m.model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                                    weights_only=True))
            self.m.model.to(self.m.model.device)
        self.m.compile(timesfm.ForecastConfig(
            max_context=1024, max_horizon=128, per_core_batch_size=256,
            normalize_inputs=True, use_continuous_quantile_head=True))

    def fc(self, ctx, miss, conv, batch=256):
        outs = []
        for i in range(0, len(ctx), batch):
            c = ctx[i:i + batch].copy()
            if conv == "nan":
                c[miss[i:i + batch]] = np.nan
            pf, _ = self.m.forecast(horizon=H, inputs=[r for r in c])
            outs.append(np.asarray(pf, dtype=np.float32))
        return np.concatenate(outs)[:, :H]


# ----------------------------------------------------------------- TempoPFN ---

TEMPO = os.path.join(ROOT, "models_local", "tempopfn")


def build_tempopfn(dev, ckpt=None):
    import yaml
    from src.models.model import TimeSeriesModel
    with open(os.path.join(TEMPO, "configs", "example.yaml")) as f:
        cfg = yaml.safe_load(f)
    m = TimeSeriesModel(**cfg["TimeSeriesModel"]).to(dev)
    sd = torch.load(os.path.join(TEMPO, "models", "checkpoint_38M.pth"),
                    map_location="cpu", weights_only=False)["model_state_dict"]
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    return m


def tempopfn_container(ctx_np, fut_np, dev):
    """s62 container conventions: fixed start, Frequency.H, [B, T, 1] tensors."""
    from src.data.containers import BatchTimeSeriesContainer
    from src.data.frequency import Frequency
    B = len(ctx_np)
    c = torch.from_numpy(ctx_np.copy()).to(dev)
    f = (torch.from_numpy(fut_np.copy()).to(dev) if fut_np is not None else
         torch.zeros(B, H, 1, device=dev))
    return BatchTimeSeriesContainer(
        history_values=c.unsqueeze(-1),
        future_values=f.unsqueeze(-1) if f.dim() == 2 else f,
        start=[np.datetime64("2017-01-01")] * B,
        frequency=[Frequency.H] * B)


class TempoPFNEval:
    """s62 eval path (bf16 autocast), optional S71 ckpt."""
    name = "tempopfn-38m"
    convs = ("plain", "nan")

    def __init__(self, dev, ckpt=None, bf16=True):
        self.m = build_tempopfn(dev, ckpt)
        self.m.eval()
        self.dev = dev
        self.bf16 = bf16
        self.qidx_med = self.m.quantiles.index(0.5)

    @torch.no_grad()
    def fc_quant(self, ctx, miss, conv, batch=256):
        outs = []
        for i in range(0, len(ctx), batch):
            c = ctx[i:i + batch].copy()
            if conv == "nan":
                c[miss[i:i + batch]] = np.nan
            cont = tempopfn_container(c, None, self.dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=self.bf16):
                out = self.m(cont)
            pred = self.m.scaler.inverse_scale(out["result"].float(),
                                               out["scale_statistics"])
            outs.append(pred[:, :, 0, :].cpu().numpy())
        return np.concatenate(outs)

    def fc(self, ctx, miss, conv, batch=256):
        return self.fc_quant(ctx, miss, conv, batch)[..., self.qidx_med]


# ------------------------------------------------------------------- Timer ----

def build_timer(dev, ckpt=None):
    """s49 load path incl. the transformers-5.x shims and the rope buffer rebuild."""
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "seen_tokens"):
        DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, *a, **k: self.get_seq_length()
    m = AutoModelForCausalLM.from_pretrained(
        os.path.join(ROOT, "models_local", "timer-base-84m"),
        trust_remote_code=True, dtype=torch.float32).to(dev)
    for layer in m.model.layers:
        rope = layer.self_attn.rotary_emb
        dim = rope.dim
        inv = 1.0 / (rope.base ** (torch.arange(0, dim, 2, dtype=torch.int64)
                                   .float().to(rope.inv_freq.device) / dim))
        rope.inv_freq.copy_(inv)
        rope._set_cos_sin_cache(seq_len=rope.max_seq_len_cached,
                                device=rope.inv_freq.device, dtype=torch.float32)
    if ckpt is not None:
        m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
    return m


TIMER_CTX = 480          # eval crops the 512 context to the last 480 (96 x 5)
TIMER_PAD = 96 - H % 96  # future padded to a full patch (64 -> 96, 32 pad)


def timer_fwd_train(model, ctx_np, miss_np, fut_np, dev):
    """Next-patch MSE in the model's NATIVE normalised regime, aligned with the
    eval semantics (token t predicts patch t+1; the eval reads the last token's
    next-96 forecast).

    Why not the raw (revin=False) regime: the shipped weights were pretrained on
    instance-normalised windows -- measured 2026-09-03: stock outputs saturate at
    ~+-10 for any raw input scale, and the raw-regime BACKWARD overflows (grad
    norm NaN on every batch with |x|max >~ 1e5, i.e. most of the corpus). The raw
    forward is the s57 EVAL convention (kept for eval, unchanged); it is not a
    trainable regime. CPT therefore continues the model's own regime: inputs
    normalised by observed-context stats (deployable; TimesFM/TempoPFN scalers are
    likewise observed-only), the recipe's "declared" half presented as content=0
    at missing positions AFTER normalisation (TimesFM-module convention:
    norm-then-zero), loss = native next-patch MSE in normalised space with
    elementwise target masking (fabricated/pad target positions contribute 0;
    loss = the model's native next-patch objective in normalised space, Huberised at
    delta=20: raw MSE's gradient is dominated by corpus windows whose future leaves
    the context range by >1000 sd (measured: batch losses up to 4e6 with ~50% of
    batches tail-dominated); delta=20 keeps the objective quadratic in the bulk and
    bounds any single window's pull. Elementwise target masking for fabricated/pad
    target positions as before.
    """
    B = len(ctx_np)
    c = ctx_np[:, -TIMER_CTX:]
    obs = ~miss_np[:, -TIMER_CTX:]
    cnt = obs.sum(1).clip(1)
    mu = (c * obs).sum(1) / cnt
    sd = np.sqrt((((c - mu[:, None]) ** 2) * obs).sum(1) / cnt)
    sd = np.maximum(sd, 1e-3)
    cn = ((c - mu[:, None]) / sd[:, None]).astype(np.float32)
    cn[miss_np[:, -TIMER_CTX:]] = 0.0          # declared: content zeroed, no flag
    fn = ((fut_np - mu[:, None]) / sd[:, None]).astype(np.float32)
    x = np.concatenate([cn, fn,
                        np.zeros((B, TIMER_PAD), np.float32)], axis=1)  # [B, 576]
    w = np.zeros((B, TIMER_CTX + H + TIMER_PAD), np.float32)
    w[:, :TIMER_CTX] = (~miss_np[:, -TIMER_CTX:]).astype(np.float32)
    w[:, TIMER_CTX:TIMER_CTX + H] = 1.0              # future: fully observed
    t = torch.from_numpy(x).to(dev)
    out = model(input_ids=t, use_cache=False, output_hidden_states=True)
    hid = out.hidden_states[-1]                      # post-norm [B, 6, 1024]
    pred = model.lm_heads[0](hid)                    # [B, 6, 96]
    tgt = t[:, 96:].unfold(-1, 96, 96)               # token t -> patch t+1
    wt = torch.from_numpy(w[:, 96:]).to(dev)
    losses = torch.nn.functional.huber_loss(pred[:, :-1], tgt, delta=20.0,
                                            reduction="none")
    num = (losses.reshape(B, -1) * wt.reshape(B, -1)).sum(1)
    den = wt.reshape(B, -1).sum(1).clamp_min(1.0)
    return (num / den).mean()


class TimerEval:
    """s49/s57 eval path (the paper's timerxl-plain row). conv 'nan' emits NaN
    logits -- structurally broken, kept for documentation, not scored."""
    name = "timer-base-84m"
    convs = ("plain",)

    def __init__(self, dev, ckpt=None):
        self.m = build_timer(dev, ckpt)
        self.m.eval()
        self.dev = dev

    @torch.no_grad()
    def fc(self, ctx, miss, conv, batch=64):
        assert conv == "plain", "Timer-XL has no declared path (nan -> NaN logits)"
        outs = []
        for i in range(0, len(ctx), batch):
            c = ctx[i:i + batch].copy()
            c = c[:, -(c.shape[1] // 96 * 96):]
            t = torch.from_numpy(c).to(self.dev)
            o = self.m(input_ids=t, use_cache=False)
            outs.append(o.logits[:, :H].float().cpu().numpy())
        return np.concatenate(outs)


# ------------------------------------------------------------------ misc ------

WRAPPERS = {"timesfm": TimesFMEval, "tempopfn": TempoPFNEval, "timer": TimerEval}


def resolve_ckpt(model_key, arm):
    """arm: stock | cpt | ctl5k | cpt@<seed> etc."""
    if arm == "stock":
        return None
    a, _, seed = arm.partition("@")
    tag = f"{model_key}_{a}" + (f"_s{seed}" if seed else "")
    p = os.path.join(CK, f"{tag}.pt")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p
