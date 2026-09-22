#!/usr/bin/env python
"""S65 fair-budget control, Chronos-2: the "CPT control (no missing)" arm.

Mirror of s60_c2retrofit/train_s60.py in EVERY budget dimension -- stock init,
the same s45 corpus through the same Sampler (seed schedule untouched, so the
window stream is bit-identical to the recipe arm's), 5000 steps, bs 1024 as
4x256 grad accumulation, AdamW lr 1e-4 wd 0.01, warmup 100, linear decay to 0,
grad clip 1.0, NaN-gradient guard, fp32 with TF32 matmul, dual interface -- with
the SINGLE change that isolates the recipe: the augmentation regime is
"filtered" (t45.augment's no-missingness branch: every window clean, obs all
ones) instead of "mechdiv". This is the Chronos-2 analogue of the paper's Bolt
budget control (train_s53.py arm P4: native + full-model CPT on filtered).

With all-clean windows the dual and native interfaces are numerically identical
(the removed zeroing line only acts where flag == 0, and there are none), so the
control also trains exactly what a no-missingness native CPT would -- while
keeping the restored-interface weight layout the recipe arm has.

Answers the reviewer: are the s60 gains from the recipe or merely from 5,000
more steps? The Bolt control learned nothing (closure -18.9%, rho = 0.000) and
eroded the stock zero-fill behaviour (3.51 -> 9.80).
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

torch.backends.cuda.matmul.allow_tf32 = True   # train_s45/train_s60 convention
torch.backends.cudnn.allow_tf32 = True         # (eval rounds run strict fp32)

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s45_pretrain"))
sys.path.insert(0, os.path.join(EXP, "s60_c2retrofit"))
import run_s25_twofloor as s25
import train_s45 as t45
import c2_iface

L, H = s25.L, 64
SEED = 20260901
CK = os.path.join(HERE, "s65_ckpt")
STEPS, BS, LR, WARMUP = 5000, 1024, 1e-4, 100
REGIME = "filtered"     # t45.augment's no-missingness branch: the ONLY change vs s60


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
    ap.add_argument("--device", default="cuda:0")
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

    tag = "c2_nomiss" + (f"_s{args.seed}" if args.seed != SEED else "") + \
        (f"_{args.tag}" if args.tag else "")
    micro = min(args.micro, bs)
    assert bs % micro == 0, "bs must be a multiple of micro"
    accum = bs // micro
    hist, n_skip, wi, t0 = [], 0, 0, time.time()
    for it in range(steps):
        ctxs, obss, futs = [], [], []
        while len(ctxs) < bs:
            c, f = next(gen)
            c, o = t45.augment(c, REGIME, rng, wi)   # filtered: clean ctx, obs=1
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
            print(f"[CTL] {it+1}/{steps} loss={np.mean(hist[-100:]):.4f} "
                  f"skip={n_skip} lr={sch.get_last_lr()[0]:.2e} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if (it + 1) % 1000 == 0 or it + 1 == steps:
            v = quick_eval(model, sampler, dev)
            print(f"[CTL]   val clean native-loss={v:.4f}", flush=True)
        if (it + 1) % 2500 == 0 or it + 1 == steps:
            torch.save(model.state_dict(), os.path.join(CK, f"{tag}.pt"))
    torch.save(model.state_dict(), os.path.join(CK, f"{tag}.pt"))
    log = {"arm": "CPT control (no missing)", "iface": "dual",
           "regime": REGIME, "init": "stock",
           "model": "models_local/chronos-2", "seed": args.seed, "steps": steps,
           "bs": bs, "micro": micro, "accum": accum,
           "bs_note": "effective batch 1024 = 4x256 grad accumulation; exact for "
                      "singleton groups, forced by O(bs^2) group attention in fp32",
           "control_note": "identical to s60 train_s60.py except regime='filtered' "
                           "(t45.augment no-missingness branch): every window clean, "
                           "obs all ones; the Chronos-2 analogue of train_s53.py arm "
                           "P4 (the paper's Bolt CPT control)",
           "lr": args.lr, "warmup": args.warmup, "g2_stock": d,
           "loss_hist_tail": hist[-100:], "n_skip": n_skip,
           "minutes": (time.time() - t0) / 60, "smoke": args.smoke}
    json.dump(log, open(os.path.join(CK, f"{tag}_log.json"), "w"))
    print(f"[CTL] done in {log['minutes']:.1f} min -> {tag}.pt", flush=True)


if __name__ == "__main__":
    main()
