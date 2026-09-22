#!/usr/bin/env python
"""S45 stage 1: pretraining bolt-tiny from scratch under five arms (2x2 factorial + E).

Arms (same architecture, corpus, budget, seed schedule; only interface x corpus regime vary):
  A  stock interface (content zeroed)      + filtered corpus            (status quo replica)
  B  restored interface (content + flag)   + filtered corpus            (interface alone)
  C  restored interface                    + mechanism-diverse missing  (the prescription)
  D  stock interface                       + mechanism-diverse missing  (augmentation alone)
  E  restored interface                    + Moirai-2.0 recipe          (single scattered
                                                                          patch mask + causal-mean fill)

Design doc: DESIGN.md. Gates: G2 (encode replica matches stock forward at plain with the
random-init weights) runs before training; G1/G3 live in eval_s45.py.
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

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s27_interface"))
import run_s25_twofloor as s25
import run_s27_interface as s27

L, H = s25.L, 64
SEED = 20260824
CACHE = os.environ.get("S45_CACHE", os.path.join(HERE, "corpus_cache"))
CK = os.path.join(HERE, "s45_ckpt")
SYNTH = "training_corpus"          # KernelSynth shards = the synthetic pool
SYNTH_P = 0.15                     # real:synthetic = 85:15 (Chronos' 9:1, slightly diluted)
CAP_POINTS = 200_000_000           # per-subset contribution cap (anti-domination)

ARMS = {
    "A": {"iface": "native", "regime": "filtered"},
    "B": {"iface": "dual", "regime": "filtered"},
    "C": {"iface": "dual", "regime": "mechdiv"},
    "D": {"iface": "native", "regime": "mechdiv"},
    "E": {"iface": "dual", "regime": "moirai2"},
    # S47 (stage 2): the recipe ablation ladder, restored interface throughout
    "F": {"iface": "dual", "regime": "cpm"},            # TiRex/Toto/FlowState 2025 recipe
    "G": {"iface": "dual", "regime": "mechdiv_decl"},   # mechanism-diverse, declare-only
    "H": {"iface": "dual", "regime": "block_filldiv"},  # fill-diverse, single mechanism
    "I": {"iface": "dual", "regime": "mechdiv_filldiv"},  # both at full fill dosage
}


# ------------------------------------------------------------------ corpus ----

class Sampler:
    """Uniform-over-valid-windows slicing with per-subset caps and a held-out tail."""

    def __init__(self, seed, val_per_subset=64):
        self.rng = np.random.default_rng(seed)
        self.stats = json.load(open(os.path.join(CACHE, "stats.json")))
        self.subs = sorted(self.stats)
        self.flat, self.offs, self.vals = {}, {}, {}
        weights = []
        for s in self.subs:
            # load each member ONCE: numpy>=2 re-reads the zip member on every
            # NpzFile.__getitem__, which is ~1s/GB per access -- the old bottleneck
            z = np.load(os.path.join(CACHE, f"{s}.npz"))
            self.flat[s], self.offs[s] = z["flat"], z["offs"]
            n_series = len(self.offs[s])
            n_val = min(val_per_subset, n_series // 20)
            self.vals[s] = self.offs[s][-n_val:] if n_val else self.offs[s][:0]
            weights.append(min(self.stats[s]["points"], CAP_POINTS))
        w = np.asarray(weights, dtype=np.float64)
        self.p = w / w.sum()

    def _series(self, s, i):
        off, ln = self.offs[s][i]
        return self.flat[s][off:off + ln]

    def window(self, require_clean_ctx=False):
        rng = self.rng
        while True:
            if rng.random() < SYNTH_P and SYNTH in self.flat:
                s = SYNTH
            else:
                s = rng.choice(self.subs, p=self.p)
            n_tr = len(self.offs[s]) - len(self.vals[s])
            i = int(rng.integers(n_tr))
            v = self._series(s, i)
            if len(v) < L + H + 4:
                continue
            st = int(rng.integers(0, len(v) - L - H))
            ctx = v[st:st + L].copy()
            fut = v[st + L:st + L + H].copy()
            if not np.isfinite(fut).all():
                continue                       # loss needs a fully-observed horizon
            if require_clean_ctx and not np.isfinite(ctx).all():
                continue                       # the filtered regime drops NaN windows
            yield ctx, fut

    def val_windows(self, sub, n, rng):
        out = []
        for off, ln in self.vals[sub]:
            if ln < L + H + 4:
                continue
            v = self.flat[sub][off:off + ln]
            st = int(rng.integers(0, len(v) - L - H))
            ctx, fut = v[st:st + L], v[st + L:st + L + H]
            if np.isfinite(ctx).all() and np.isfinite(fut).all():
                out.append((ctx, fut))
            if len(out) >= n:
                break
        return out


# ------------------------------------------------------------ augmentation ----

def linear_fill(x, m):
    return s25.fill_context(x[None], m[None], "linear")[0]


def causal_mean_fill(x, m):
    out = x.copy()
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        return out
    obs = ~m
    csum = np.cumsum(np.where(obs, x, 0.0))
    cnt = np.cumsum(obs.astype(np.int64))
    first = x[np.flatnonzero(obs)[0]] if obs.any() else 0.0
    mean = np.where(cnt > 0, csum / np.maximum(cnt, 1), first)
    out[idx] = mean[idx]
    return out


def augment(ctx, regime, rng, wi, clean_p=0.5):
    """Returns (ctx_filled, obs) with obs = 1 at observed positions.
    clean_p: fraction of windows left untouched (P9 raises it to buy clean accuracy)."""
    if regime == "filtered":
        return ctx, np.ones_like(ctx)
    if rng.random() < clean_p:
        return ctx, np.ones_like(ctx)          # clean_p of windows stay clean
    if regime == "moirai2":
        n_patch = L // 16
        pm = rng.random(n_patch) < 0.5
        mask = np.repeat(pm, 16)
        if mask.all():
            mask[rng.integers(n_patch) * 16:(rng.integers(n_patch) + 1) * 16] = False
        return causal_mean_fill(ctx, mask), (~mask).astype(np.float32)
    if regime == "cpm":
        # TiRex/Toto/FlowState recipe: contiguous run of c~U(1,5) patches masked with
        # probability p~U(0,0.25); content zeroed, flag carries the declaration
        n_patch = L // 16
        c = int(rng.integers(1, 6))
        p = float(rng.uniform(0, 0.25))
        mask = np.zeros(L, bool)
        n_runs = max(1, int(round(p * n_patch / c)))
        for s in rng.integers(0, n_patch - c + 1, size=n_runs):
            mask[s * 16:(s + c) * 16] = True
        out = ctx.copy()
        out[mask] = 0.0
        return out, (~mask).astype(np.float32)
    if regime == "block_filldiv":
        # fill-diverse, single mechanism: block outages only, always alpha-blend filled
        rate = float(rng.uniform(0.05, 0.7))
        mask = s25.make_mask("block", rate, wi, int(rng.integers(1 << 30)), 1,
                             x=ctx[None])[0]
        a = float(rng.uniform(0, 1))
        lin = linear_fill(ctx, mask)
        out = lin.copy()
        out[mask] = (1 - a) * lin[mask] + a * ctx[mask]
        return out, (~mask).astype(np.float32)
    if regime == "block_declblend":
        # S59 fix: H recipe + the declaration endpoint. Block outages only; half
        # declared (holes zeroed + flag), half alpha-blend filled -- so the model
        # sees content=0 at flagged positions during training too.
        rate = float(rng.uniform(0.05, 0.7))
        mask = s25.make_mask("block", rate, wi, int(rng.integers(1 << 30)), 1,
                             x=ctx[None])[0]
        if rng.random() < 0.5:
            out = ctx.copy()
            out[mask] = 0.0
            return out, (~mask).astype(np.float32)
        a = float(rng.uniform(0, 1))
        lin = linear_fill(ctx, mask)
        out = lin.copy()
        out[mask] = (1 - a) * lin[mask] + a * ctx[mask]
        return out, (~mask).astype(np.float32)
    if regime == "block_declblend_ffill":
        # P7 exploration: cover the deployment fills explicitly. Thirds: declared
        # zero, alpha-blend from LINEAR, alpha-blend from FFILL.
        rate = float(rng.uniform(0.05, 0.7))
        mask = s25.make_mask("block", rate, wi, int(rng.integers(1 << 30)), 1,
                             x=ctx[None])[0]
        u = rng.random()
        if u < 1.0 / 3.0:
            out = ctx.copy()
            out[mask] = 0.0
            return out, (~mask).astype(np.float32)
        base_fill = linear_fill(ctx, mask) if u < 2.0 / 3.0 else \
            s25.fill_context(ctx[None], mask[None], "ffill")[0]
        a = float(rng.uniform(0, 1))
        out = base_fill.copy()
        out[mask] = (1 - a) * base_fill[mask] + a * ctx[mask]
        return out, (~mask).astype(np.float32)
    if regime == "mechdiv_filldiv":
        # mechanism-diverse at full fill dosage: always alpha-blend filled, never declared
        mech = ("mcar", "block", "mnar_high", "mnar_extreme")[int(rng.integers(4))]
        rate = float(rng.uniform(0.05, 0.7))
        mask = s25.make_mask(mech, rate, wi, int(rng.integers(1 << 30)), 1,
                             x=ctx[None])[0]
        a = float(rng.uniform(0, 1))
        lin = linear_fill(ctx, mask)
        out = lin.copy()
        out[mask] = (1 - a) * lin[mask] + a * ctx[mask]
        return out, (~mask).astype(np.float32)
    if regime == "mechdiv_decl":
        # mechanism-diverse, declare-only: holes always zeroed + flag, no fill mixture
        mech = ("mcar", "block", "mnar_high", "mnar_extreme")[int(rng.integers(4))]
        rate = float(rng.uniform(0.05, 0.7))
        mask = s25.make_mask(mech, rate, wi, int(rng.integers(1 << 30)), 1,
                             x=ctx[None])[0]
        out = ctx.copy()
        out[mask] = 0.0
        return out, (~mask).astype(np.float32)
    # mechdiv: mechanism-diverse missingness with deployment-realistic mixture
    mech = ("mcar", "block", "mnar_high", "mnar_extreme")[int(rng.integers(4))]
    rate = float(rng.uniform(0.05, 0.7))
    mask = s25.make_mask(mech, rate, wi, int(rng.integers(1 << 30)), 1, x=ctx[None])[0]
    if rng.random() < 0.5:
        out = ctx.copy()                       # declared: content zeroed, flag carries it
        out[mask] = 0.0
        return out, (~mask).astype(np.float32)
    a = float(rng.uniform(0, 1))               # filled+declared: alpha-blend content
    lin = linear_fill(ctx, mask)
    out = lin.copy()
    out[mask] = (1 - a) * lin[mask] + a * ctx[mask]
    return out, (~mask).astype(np.float32)


# ------------------------------------------------------------------ model ----

def build_model(dev, size="tiny", seed=None):
    from transformers import AutoConfig
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    cfg = AutoConfig.from_pretrained(os.path.join(ROOT, "models_local",
                                                f"chronos-bolt-{size}"))
    # identical init across arms at a given seed; --seed varies it across replicates
    torch.manual_seed(SEED if seed is None else seed)
    m = ChronosBoltModelForForecasting(cfg).to(dev)
    for p in m.parameters():
        p.requires_grad_(True)
    return m


def quantiles_fwd(model, ctx, obs, iface):
    """All-quantile forecast [B, n_quantiles, H] through the interface switch."""
    hidden, loc_scale, input_embeds, attn = s27.encode_iface(model, ctx, obs, iface)
    seq = model.decode(input_embeds, attn, hidden)
    B = ctx.shape[0]
    q = model.output_patch_embedding(seq).view(B, model.num_quantiles,
                                               model.chronos_config.prediction_length)
    return model.instance_norm.inverse(q.view(B, -1), loc_scale).view(
        B, model.num_quantiles, model.chronos_config.prediction_length)


def pinball(q, y, levels):
    """q: [B, Q, H] predicted quantiles; y: [B, H]; levels: [Q]."""
    e = y.unsqueeze(1) - q
    t = torch.as_tensor(levels, device=q.device, dtype=q.dtype).view(1, -1, 1)
    return torch.maximum(t * e, (t - 1) * e).mean()


def gate_g2(model, dev):
    """Encode replica (plain) must match the stock forward with random-init weights."""
    was_training = model.training
    model.eval()                               # dropout off: the two paths must match exactly
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, L)).astype(np.float32)
    obs = np.ones_like(x)
    xt = torch.from_numpy(x).to(dev)
    with torch.no_grad():
        a = quantiles_fwd(model, xt, torch.from_numpy(obs).to(dev), "plain")
        b = model(context=xt, mask=None).quantile_preds
    d = float((a - b).abs().max())
    if was_training:
        model.train()
    print(f"GATE G2 encode replica vs stock (random init): max|diff|={d:.3e}", flush=True)
    assert d < 1e-3, "G2 FAILED"
    return d


# ------------------------------------------------------------------ train ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--size", default="tiny", choices=["tiny", "small", "base"],
                    help="bolt size; tiny=8.7M, small=48M, base=205M")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="replicate seed: varies init, sampler and augmentation")
    ap.add_argument("--tag", default="",
                    help="checkpoint-name suffix, for runs that vary the recipe rather "
                         "than the seed (e.g. r50k); keeps them from shadowing the originals")
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    arm = ARMS[args.arm]
    steps = 300 if args.smoke else args.steps
    bs = 128 if args.smoke else args.bs
    dev = args.device
    os.makedirs(CK, exist_ok=True)
    rng = np.random.default_rng(args.seed + 100 * (ord(args.arm) - 65))
    sampler = Sampler(seed=args.seed + 1)
    gen = sampler.window(require_clean_ctx=True)
    # every arm trains on finite windows; the regimes differ in the AUGMENTATION
    # (filtered = none, mechdiv/moirai2 = on-the-fly missingness), not in corpus NaNs

    tag = f"{args.arm}_{args.size}"
    if args.seed != SEED:
        tag += f"_s{args.seed}"
    if args.tag:
        tag += f"_{args.tag}"
    model = build_model(dev, args.size, seed=args.seed)
    gate_g2(model, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / args.warmup) * (1 - t / steps))
    levels = list(model.chronos_config.quantiles)

    val_subs = ("m4_daily", "ushcn_daily", "m5")
    hist, n_skip, t0 = [], 0, time.time()
    wi = 0
    for it in range(steps):
        ctxs, obss, futs = [], [], []
        while len(ctxs) < bs:
            c, f = next(gen)
            c, o = augment(c, arm["regime"], rng, wi)
            wi += 1
            ctxs.append(c)
            obss.append(o)
            futs.append(f)
        xt = torch.from_numpy(np.stack(ctxs)).to(dev)
        ot = torch.from_numpy(np.stack(obss)).to(dev)
        yt = torch.from_numpy(np.stack(futs)).to(dev)
        q = quantiles_fwd(model, xt, ot, arm["iface"])[:, :, :H]
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
            print(f"[{args.arm}] {it+1}/{steps} loss={np.mean(hist[-100:]):.4f} "
                  f"skip={n_skip} lr={sch.get_last_lr()[0]:.2e} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if (it + 1) % 1000 == 0 or it + 1 == steps:
            v = quick_eval(model, sampler, val_subs, arm, dev)
            print(f"[{args.arm}]   val clean pinball={v:.4f}", flush=True)
        if (it + 1) % 5000 == 0 or it + 1 == steps:
            torch.save(model.state_dict(), os.path.join(CK, f"arm_{tag}.pt"))
    torch.save(model.state_dict(), os.path.join(CK, f"arm_{tag}.pt"))
    json.dump({"arm": args.arm, **arm, "size": args.size, "seed": args.seed,
               "steps": steps, "bs": bs, "lr": args.lr,
               "loss_hist_tail": hist[-100:], "n_skip": n_skip,
               "minutes": (time.time() - t0) / 60},
              open(os.path.join(CK, f"arm_{tag}_log.json"), "w"))
    print(f"[{args.arm}] done in {(time.time()-t0)/60:.1f} min", flush=True)


@torch.no_grad()
def quick_eval(model, sampler, subs, arm, dev, n=32):
    model.eval()
    rng = np.random.default_rng(SEED + 999)
    losses = []
    for s in subs:
        ws = sampler.val_windows(s, n, rng)
        if not ws:
            continue
        ctx = np.stack([w[0] for w in ws])
        fut = np.stack([w[1] for w in ws])
        q = quantiles_fwd(model, torch.from_numpy(ctx).to(dev),
                          torch.ones_like(torch.from_numpy(ctx)).to(dev), arm["iface"])
        losses.append(float(pinball(q, torch.from_numpy(fut).to(dev),
                                    list(model.chronos_config.quantiles))))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


if __name__ == "__main__":
    main()
