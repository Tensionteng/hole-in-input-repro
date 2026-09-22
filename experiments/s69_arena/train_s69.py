#!/usr/bin/env python
"""S69 ("the trust arena"): train the 2x2 factorial of bolt-tiny-scale transformers on the
synthetic bivariate corpus. Adapted from ../s45_pretrain/train_s45.py -- same backbone
(bolt-tiny), same optimizer schedule (15k steps, bs 1024, lr 3e-4, warmup 1k, linear
decay), checkpoints every 1k steps (DESIGN.md saturation clause).

Differences from S45, all mandated by DESIGN.md:
  - corpus is the bivariate (AR(1) target + correlated context) synthetic pool from
    gen_s69_corpus.py, not univariate npz caches;
  - the model input is 2 channels (native: x_content, c_content) or 4 channels
    (dual: x_content, x_flag, c_content, c_flag) per patch; input_patch_embedding is
    widened accordingly and re-initialized with the chronos scheme (std scaled by
    in_dim**-0.5); everything else is the stock bolt-tiny init at torch seed --seed;
  - the diverse regime corrupts EACH channel independently per window
    (gen_s69_corpus.corrupt_channel); native arms get identical corruption geometry but
    masked positions carry 0 and no flag channel exists ("augmentation through the
    blocked interface", as S45 arm D);
  - instance norm is per content channel (each channel standardized on its own context
    window); the forecast is inverse-transformed with the TARGET channel's loc/scale;
    flags are never normalized; all patches attend (content is always present; the flag
    channels carry observability -- there is no attention-level masking in the arena).

Arms (DESIGN.md): A native+filtered, B native+diverse, C dual+filtered, D dual+diverse.
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

torch.backends.cuda.matmul.allow_tf32 = True   # A800: fp32 storage, TF32 matmul speed;
torch.backends.cudnn.allow_tf32 = True         # evaluation rounds keep strict fp32

sys.path.insert(0, HERE)
import gen_s69_corpus as g69

L, H, T = g69.L, g69.H, g69.T
ARMS = g69.ARMS
CK = os.path.join(HERE, "s69_ckpt")


# ------------------------------------------------------------------ corpus ----

class Sampler:
    """Uniform windows over the synthetic train pool (all windows valid and finite)."""

    def __init__(self, corpus, seed):
        self.rng = np.random.default_rng(seed)
        z = np.load(os.path.join(corpus, "train.npz"))
        self.x, self.c = z["x"], z["c"]

    def batch(self, bs):
        """Vectorized draw of bs windows: (ctx_x, ctx_c, fut), each [bs, L] / [bs, H]."""
        rng = self.rng
        i = rng.integers(0, len(self.x), bs)
        st = rng.integers(0, T - L - H, bs)
        cols = st[:, None] + np.arange(L)
        ctx_x = self.x[i[:, None], cols]
        ctx_c = self.c[i[:, None], cols]
        fut = self.x[i[:, None], st[:, None] + L + np.arange(H)]
        return ctx_x, ctx_c, fut


def eval_windows(corpus, n=None):
    """Held-out eval pool at the fixed final window: (x_ctx, c_ctx, fut, phi, rho)."""
    z = np.load(os.path.join(corpus, "eval.npz"))
    x, c = z["x"], z["c"]
    if n is not None:
        x, c, = x[:n], c[:n]
    s = g69.EVAL_START
    return (x[:, s:s + L], c[:, s:s + L], x[:, s + L:s + L + H],
            z["phi"][:len(x)], z["rho"][:len(x)])


# ------------------------------------------------------------ augmentation ----

def augment(ctx_x, ctx_c, regime, iface, rng, v2=False):
    """Per-window corruption. Returns (fx, fc, flag_x, flag_c) float32 [L] each.
    filtered: clean window, flags all-observed.
    diverse:  each channel corrupted independently (g69.corrupt_channel; s69b uses
              corrupt_channel_v2 which adds the tail-censor mechanism); dual keeps the
              q-blend content + flag; native zeroes the masked positions and the flags are
              all-ones placeholders (no flag channel exists for native arms)."""
    if regime == "filtered":
        one = np.ones(L, np.float32)
        return ctx_x.copy(), ctx_c.copy(), one, one.copy()
    corrupt = g69.corrupt_channel_v2 if v2 else g69.corrupt_channel
    fx, ox = corrupt(ctx_x, rng)
    fc, oc = corrupt(ctx_c, rng)
    if iface == "native":
        fx = np.where(ox > 0, fx, 0.0).astype(np.float32)   # masked positions carry 0,
        fc = np.where(oc > 0, fc, 0.0).astype(np.float32)   # and no flag exists
        one = np.ones(L, np.float32)
        return fx, fc, one, one.copy()
    return fx, fc, ox, oc


# ------------------------------------------------------------------ model ----

def build_model(dev, iface="native", size="tiny", seed=0):
    """bolt-tiny from scratch; input_patch_embedding widened to the channel count.
    Backbone init identical across arms at a given --seed; the widened embedding is
    re-initialized with the chronos scheme under a fixed sub-seed (shared between the
    two arms of the same interface width)."""
    from transformers import AutoConfig
    from chronos.chronos_bolt import ChronosBoltModelForForecasting, ResidualBlock
    cfg = AutoConfig.from_pretrained(os.path.join(ROOT, "models_local",
                                                  f"chronos-bolt-{size}"))
    torch.manual_seed(seed)
    m = ChronosBoltModelForForecasting(cfg)
    n_ch = 4 if iface == "dual" else 2
    in_dim = m.chronos_config.input_patch_size * n_ch
    torch.manual_seed(seed + 913)
    emb = ResidualBlock(in_dim=in_dim, h_dim=cfg.d_ff, out_dim=cfg.d_model,
                        act_fn_name=cfg.dense_act_fn, dropout_p=cfg.dropout_rate)
    factor = cfg.initializer_factor
    torch.nn.init.normal_(emb.hidden_layer.weight, std=factor * in_dim ** -0.5)
    torch.nn.init.zeros_(emb.hidden_layer.bias)
    torch.nn.init.normal_(emb.residual_layer.weight, std=factor * in_dim ** -0.5)
    torch.nn.init.zeros_(emb.residual_layer.bias)
    torch.nn.init.normal_(emb.output_layer.weight, std=factor * cfg.d_ff ** -0.5)
    torch.nn.init.zeros_(emb.output_layer.bias)
    m.input_patch_embedding = emb
    m = m.to(dev)
    for p in m.parameters():
        p.requires_grad_(True)
    return m


def encode_bi(model, fx, fc, flag_x, flag_c, iface):
    """Bivariate replica of ChronosBoltModelForForecasting.encode.
    fx/fc: [B, L] finite content; flag_x/flag_c: [B, L] 1=observed (dual only).
    Patch-channel order (fixed): dual = [x_content | x_flag | c_content | c_flag],
    native = [x_content | c_content]."""
    xn, loc_scale = model.instance_norm(fx)
    cn, _ = model.instance_norm(fc)
    xn, cn = xn.to(model.dtype), cn.to(model.dtype)
    parts = [model.patch(xn)]
    if iface == "dual":
        parts.append(torch.nan_to_num(model.patch(flag_x.to(model.dtype)), nan=0.0))
    parts.append(model.patch(cn))
    if iface == "dual":
        parts.append(torch.nan_to_num(model.patch(flag_c.to(model.dtype)), nan=0.0))
    patched = torch.cat(parts, dim=-1)
    attention_mask = torch.ones(patched.shape[0], patched.shape[1],
                                dtype=model.dtype, device=patched.device)
    input_embeds = model.input_patch_embedding(patched)
    if model.chronos_config.use_reg_token:
        reg_ids = torch.full((fx.shape[0], 1), model.config.reg_token_id,
                             device=input_embeds.device)
        input_embeds = torch.cat([input_embeds, model.shared(reg_ids)], dim=-2)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(reg_ids).to(model.dtype)], dim=-1)
    enc = model.encoder(attention_mask=attention_mask, inputs_embeds=input_embeds)
    return enc[0], loc_scale, input_embeds, attention_mask


def quantiles_fwd(model, fx, fc, flag_x, flag_c, iface):
    """All-quantile forecast [B, n_quantiles, H] in raw (target) units."""
    hidden, loc_scale, input_embeds, attn = encode_bi(model, fx, fc, flag_x, flag_c,
                                                      iface)
    seq = model.decode(input_embeds, attn, hidden)
    B = fx.shape[0]
    q = model.output_patch_embedding(seq).view(B, model.num_quantiles,
                                               model.chronos_config.prediction_length)
    return model.instance_norm.inverse(q.view(B, -1), loc_scale).view(
        B, model.num_quantiles, model.chronos_config.prediction_length)


def pinball(q, y, levels):
    """q: [B, Q, H] predicted quantiles; y: [B, H]; levels: [Q]."""
    e = y.unsqueeze(1) - q
    t = torch.as_tensor(levels, device=q.device, dtype=q.dtype).view(1, -1, 1)
    return torch.maximum(t * e, (t - 1) * e).mean()


# ------------------------------------------------------------------ train ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--size", default="tiny", choices=["tiny", "small", "base"])
    ap.add_argument("--seed", type=int, default=0, help="replicate seed (0/1/2): varies "
                                                    "init, sampler and augmentation")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--round", default="s69", choices=["s69", "s69b"],
                    help="s69b: harmonic corpus, tail-censor augmentation, b-dirs")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--ck-dir", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    v2 = args.round == "s69b"
    corpus = args.corpus or os.path.join(HERE, "s69b_corpus" if v2 else "s69_corpus")
    arm = ARMS[args.arm]
    steps = 300 if args.smoke else args.steps
    bs = 128 if args.smoke else args.bs
    dev = args.device
    ck_dir = args.ck_dir or os.path.join(HERE, "s69b_ckpt" if v2 else "s69_ckpt")
    os.makedirs(ck_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed + 100 * (ord(args.arm) - 65))
    sampler = Sampler(corpus, seed=args.seed + 1)

    tag = f"{args.arm}_s{args.seed}"
    model = build_model(dev, arm["iface"], args.size, seed=args.seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] iface={arm['iface']} regime={arm['regime']} params={n_params/1e6:.2f}M",
          flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / args.warmup) * (1 - t / steps))
    levels = list(model.chronos_config.quantiles)

    hist, n_skip, t0 = [], 0, time.time()
    for it in range(steps):
        cx, cc, fut = sampler.batch(bs)
        fxs, fcs, ox, oc = [], [], [], []
        for w in range(bs):
            a, b, c_, d = augment(cx[w], cc[w], arm["regime"], arm["iface"], rng, v2)
            fxs.append(a)
            fcs.append(b)
            ox.append(c_)
            oc.append(d)
        xt = torch.from_numpy(np.stack(fxs)).to(dev)
        ct = torch.from_numpy(np.stack(fcs)).to(dev)
        ot = torch.from_numpy(np.stack(ox)).to(dev)
        pt = torch.from_numpy(np.stack(oc)).to(dev)
        yt = torch.from_numpy(fut).to(dev)
        q = quantiles_fwd(model, xt, ct, ot, pt, arm["iface"])[:, :, :H]
        loss = pinball(q, yt, levels)
        if not torch.isfinite(loss):
            n_skip += 1
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            n_skip += 1
            opt.zero_grad()
            continue
        opt.step()
        sch.step()
        hist.append(float(loss))
        if (it + 1) % 100 == 0:
            print(f"[{tag}] {it+1}/{steps} loss={np.mean(hist[-100:]):.4f} "
                  f"skip={n_skip} lr={sch.get_last_lr()[0]:.2e} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if (it + 1) % 1000 == 0 or it + 1 == steps:
            v = quick_eval(model, corpus, arm["iface"], dev)
            print(f"[{tag}]   val clean pinball={v:.4f}", flush=True)
            torch.save(model.state_dict(),
                       os.path.join(ck_dir, f"arm_{tag}_step{it+1}.pt"))
    torch.save(model.state_dict(), os.path.join(ck_dir, f"arm_{tag}.pt"))
    meta = {"arm": args.arm, **arm, "size": args.size, "seed": args.seed,
            "steps": steps, "bs": bs, "lr": args.lr, "warmup": args.warmup,
            "n_params": n_params, "code_hash": g69.code_hash(),
            "torch": torch.__version__, "numpy": np.__version__,
            "corpus": os.path.basename(corpus.rstrip("/")), "round": args.round,
            "loss_hist_tail": hist[-100:], "n_skip": n_skip,
            "minutes": (time.time() - t0) / 60}
    json.dump(meta, open(os.path.join(ck_dir, f"arm_{tag}_log.json"), "w"))
    print(f"[{tag}] done in {(time.time()-t0)/60:.1f} min", flush=True)


@torch.no_grad()
def quick_eval(model, corpus, iface, dev, n=64):
    """Clean-window pinball on the held-out eval pool."""
    was_training = model.training
    model.eval()
    x_ctx, c_ctx, fut, _, _ = eval_windows(corpus, n)
    one = np.ones_like(x_ctx)
    q = quantiles_fwd(model, torch.from_numpy(x_ctx).to(dev),
                      torch.from_numpy(c_ctx).to(dev), torch.from_numpy(one).to(dev),
                      torch.from_numpy(one.copy()).to(dev), iface)
    v = float(pinball(q[:, :, :H], torch.from_numpy(fut).to(dev),
                      list(model.chronos_config.quantiles)))
    if was_training:
        model.train()
    return v


if __name__ == "__main__":
    main()
