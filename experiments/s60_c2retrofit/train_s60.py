#!/usr/bin/env python
"""S60 CPT: retrofit the STOCK Chronos-2 checkpoint (the P8-arm recipe, second family).

Identical in spirit to train_s53.py stage 2 for Chronos-Bolt:
  init     stock models_local/chronos-2 (fp32)
  corpus   s45_pretrain/corpus_cache via t45.Sampler (real:synthetic 85:15, capped)
  regime   "mechdiv" (t45.augment, verbatim): 50% of windows clean; the rest get
           mechanism ~ U{mcar, block, mnar_high, mnar_extreme}, rate ~ U(0.05, 0.7);
           half DECLARED (content zeroed + flag) -- the declaration endpoint --
           half alpha-blend filled (content = (1-a)*linear + a*truth, a ~ U(0,1),
           + flag) -- the fill-quality spectrum
  iface    dual (fill content AND flag reach the input patch embedding)
  loss     Chronos-2's NATIVE quantile loss (model._compute_loss on the normalized
           quantiles), not a re-implementation
  budget   5000 steps, bs 1024, AdamW lr 1e-4 wd 0.01, warmup 100, linear decay to 0,
           grad clip 1.0, NaN-gradient guard (skip non-finite loss / grad norm),
           fp32 with TF32 matmul (train_s45 convention)

Gate G2' (replica vs stock forward with STOCK weights) runs before training.
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

torch.backends.cuda.matmul.allow_tf32 = True   # train_s45/train_s53 convention
torch.backends.cudnn.allow_tf32 = True         # (eval rounds run strict fp32)

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s45_pretrain"))
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import train_s45 as t45
import c2_iface

L, H = s25.L, 64
SEED = 20260901
CK = os.path.join(HERE, "s60_ckpt")
STEPS, BS, LR, WARMUP = 5000, 1024, 1e-4, 100


def gate_g2_stock(model, dev):
    """Encode replica (native) vs stock forward with the STOCK weights."""
    was_training = model.training
    model.eval()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, L)).astype(np.float32)
    mask = torch.from_numpy(rng.random(size=(8, L)) < 0.4).to(dev)
    xt = torch.from_numpy(x).to(dev)
    xnan = xt.clone()
    xnan[mask] = float("nan")
    nop = H // model.chronos_config.output_patch_size
    with torch.no_grad():
        ref = model(context=xnan, num_output_patches=nop).quantile_preds
        q, _ = c2_iface.forward_iface(model, xnan, None, "native", nop)
    d = float((q - ref).abs().max())
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    if was_training:
        model.train()
    print(f"GATE G2' c2 replica native vs stock (stock weights): max|diff|={d:.3e}",
          flush=True)
    assert d == 0.0, "G2' FAILED -- do not train on a divergent replica"
    return d


@torch.no_grad()
def quick_eval(model, sampler, dev, n=32):
    """Clean-context native Chronos-2 loss on held-out corpus tails (dual iface)."""
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(t45.SEED + 999)
    nop = H // model.chronos_config.output_patch_size
    losses = []
    for s in ("m4_daily", "ushcn_daily", "m5"):
        ws = sampler.val_windows(s, n, rng)
        if not ws:
            continue
        ctx = np.stack([w[0] for w in ws])
        fut = np.stack([w[1] for w in ws])
        _, loss = c2_iface.forward_iface(
            model, torch.from_numpy(ctx).to(dev),
            torch.ones_like(torch.from_numpy(ctx)).to(dev), "dual", nop,
            future_target=torch.from_numpy(fut).to(dev))
        losses.append(float(loss))
    if was_training:
        model.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--bs", type=int, default=BS)
    ap.add_argument("--micro", type=int, default=256,
                    help="micro-batch for gradient accumulation; Chronos-2's group "
                         "attention is O(bs^2), so bs 1024 OOMs in fp32. Singleton "
                         "groups make accumulation exact (no cross-series leakage).")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dev = args.device
    steps = 40 if args.smoke else args.steps
    bs = 128 if args.smoke else args.bs
    os.makedirs(CK, exist_ok=True)

    model = c2_iface.load_stock(dev, train=True)
    d = gate_g2_stock(model, dev)
    sampler = t45.Sampler(seed=args.seed + 1)
    gen = sampler.window(require_clean_ctx=True)
    rng = np.random.default_rng(args.seed)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / args.warmup) * (1 - t / steps))
    nop = H // model.chronos_config.output_patch_size

    tag = "arm_CPT" + (f"_s{args.seed}" if args.seed != SEED else "") + \
        (f"_{args.tag}" if args.tag else "")
    micro = min(args.micro, bs)
    assert bs % micro == 0, "bs must be a multiple of micro"
    accum = bs // micro
    hist, n_skip, wi, t0 = [], 0, 0, time.time()
    for it in range(steps):
        ctxs, obss, futs = [], [], []
        while len(ctxs) < bs:
            c, f = next(gen)
            c, o = t45.augment(c, "mechdiv", rng, wi)
            wi += 1
            ctxs.append(c)
            obss.append(o)
            futs.append(f)
        loss_acc, finite = 0.0, True
        opt.zero_grad()
        for mb in range(accum):
            sl = slice(mb * micro, (mb + 1) * micro)
            xt = torch.from_numpy(np.stack(ctxs[sl])).to(dev)
            ot = torch.from_numpy(np.stack(obss[sl])).to(dev)
            yt = torch.from_numpy(np.stack(futs[sl])).to(dev)
            _, loss = c2_iface.forward_iface(model, xt, ot, "dual", nop,
                                             future_target=yt)
            if not torch.isfinite(loss):
                finite = False
                break
            (loss / accum).backward()
            loss_acc += float(loss) / accum
        if not finite:
            n_skip += 1
            opt.zero_grad()
            continue
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            n_skip += 1
            opt.zero_grad()
            continue
        opt.step()
        sch.step()
        hist.append(loss_acc)
        if (it + 1) % 100 == 0:
            print(f"[CPT] {it+1}/{steps} loss={np.mean(hist[-100:]):.4f} "
                  f"skip={n_skip} lr={sch.get_last_lr()[0]:.2e} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if (it + 1) % 1000 == 0 or it + 1 == steps:
            v = quick_eval(model, sampler, dev)
            print(f"[CPT]   val clean native-loss={v:.4f}", flush=True)
        if (it + 1) % 2500 == 0 or it + 1 == steps:
            torch.save(model.state_dict(), os.path.join(CK, f"{tag}.pt"))
    torch.save(model.state_dict(), os.path.join(CK, f"{tag}.pt"))
    log = {"arm": "CPT", "iface": "dual", "regime": "mechdiv", "init": "stock",
           "model": "models_local/chronos-2", "seed": args.seed, "steps": steps,
           "bs": bs, "micro": micro, "accum": accum,
           "bs_note": "effective batch 1024 = 4x256 grad accumulation; exact for "
                      "singleton groups, forced by O(bs^2) group attention in fp32",
           "lr": args.lr, "warmup": args.warmup, "g2_stock": d,
           "loss_hist_tail": hist[-100:], "n_skip": n_skip,
           "minutes": (time.time() - t0) / 60, "smoke": args.smoke}
    json.dump(log, open(os.path.join(CK, f"{tag}_log.json"), "w"))
    print(f"[CPT] done in {log['minutes']:.1f} min -> {tag}.pt", flush=True)


if __name__ == "__main__":
    main()
