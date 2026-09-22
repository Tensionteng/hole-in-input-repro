#!/usr/bin/env python
"""S65 fair-budget control, Moirai 2.0: the "CPT control (no missing)" arm.

Mirror of s61_m2retrofit/train_s61.py in EVERY budget dimension -- stock init,
the same s45 corpus through the same Sampler (seed schedule untouched, so the
window stream is bit-identical to the recipe arm's), 5000 steps, bs 1024 in a
single micro-batch, AdamW lr 1e-4 wd 0.01, warmup 100, linear decay to 0, grad
clip 1.0, NaN-gradient guard, fp32 with TF32 matmul, the native training loss
(training_mode module + released PackedQuantileMAELoss on the H=64 inference
map) -- with the SINGLE change that isolates the recipe: the augmentation
regime is "filtered" (t45.augment's no-missingness branch: every window clean,
obs all ones) instead of "mechdiv". The documented corpus-quality filter is
kept IDENTICAL (same code path; with obs all ones the observed-only stats are
the full-context stats, ~1.3% of windows skipped as before).

Since Moirai 2.0 is a RETAINER (fill content and flag already enter the content
path natively), the control needs no interface decision: it is exactly "5,000
more steps of clean-data training" for this family. The recipe arm's gains
(closure@1 0.36 CPT / 0.52 W50, leaderboard zero 8.97 -> 2.00/2.79) must NOT
appear here if they come from the recipe rather than the budget.
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

torch.backends.cuda.matmul.allow_tf32 = True   # train_s45/train_s61 convention
torch.backends.cudnn.allow_tf32 = True         # (eval rounds run strict fp32)

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s45_pretrain"))
sys.path.insert(0, os.path.join(EXP, "s61_m2retrofit"))
import run_s25_twofloor as s25
import train_s45 as t45
import m2_iface

L, H = s25.L, 64
SEED = 20260901
CK = os.path.join(HERE, "s65_ckpt")
STEPS, BS, LR, WARMUP = 5000, 1024, 1e-4, 100
REGIME = "filtered"     # t45.augment's no-missingness branch: the ONLY change vs s61


@torch.no_grad()
def quick_eval(fc, sampler, dev, n=32):
    """Clean-context native loss on held-out corpus tails (obs all ones)."""
    was_training = fc.training
    fc.eval()
    rng = np.random.default_rng(t45.SEED + 999)
    losses = []
    for s in ("m4_daily", "ushcn_daily", "m5"):
        ws = sampler.val_windows(s, n, rng)
        if not ws:
            continue
        kept = []
        for c, f in ws:                       # same corpus-quality filter as train
            mu = float(c.mean())
            var = float(((c - mu) ** 2).sum() / max(len(c) - 1, 1))
            if float(np.abs((f - mu) / np.sqrt(var + 1e-5)).max()) <= 100:
                kept.append((c, f))
        if not kept:
            continue
        ctx = np.stack([w[0] for w in kept])
        fut = np.stack([w[1] for w in kept])
        content, obs = m2_iface.conv_ctx(ctx, np.zeros_like(ctx, bool), "plain")
        losses.append(float(m2_iface.train_forward(fc, content, obs, fut)))
    if was_training:
        fc.train()
    return float(np.mean(losses)) if losses else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--bs", type=int, default=BS)
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

    fc = m2_iface.load_stock(dev, train=True)
    assert (fc.module.patch_size, fc.module.num_predict_token,
            fc.module.num_quantiles) == (m2_iface.PATCH, m2_iface.NPT, m2_iface.NQ)
    sampler = t45.Sampler(seed=args.seed + 1)
    gen = sampler.window(require_clean_ctx=True)
    rng = np.random.default_rng(args.seed)

    opt = torch.optim.AdamW(fc.module.parameters(), lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / args.warmup) * (1 - t / steps))

    tag = "m2_nomiss" + (f"_s{args.seed}" if args.seed != SEED else "") + \
        (f"_{args.tag}" if args.tag else "")
    hist, n_skip, n_filt, wi, t0 = [], 0, 0, 0, time.time()
    for it in range(steps):
        ctxs, obss, futs = [], [], []
        while len(ctxs) < bs:
            c, f = next(gen)
            c, o = t45.augment(c, REGIME, rng, wi)   # filtered: clean ctx, obs=1
            wi += 1
            ob = o > 0.5
            n_ob = int(ob.sum())
            if n_ob < 8:
                n_filt += 1
                continue
            mu = float(c[ob].mean())
            var = float(((c[ob] - mu) ** 2).sum() / max(n_ob - 1, 1))
            if float(np.abs((f - mu) / np.sqrt(var + 1e-5)).max()) > 100:
                n_filt += 1                       # corpus-quality filter (s61 header)
                continue
            ctxs.append(c)
            obss.append(o)
            futs.append(f)
        loss = m2_iface.train_forward(fc, np.stack(ctxs), np.stack(obss),
                                      np.stack(futs))
        if not torch.isfinite(loss):
            n_skip += 1
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(fc.module.parameters(), 1.0)
        if not torch.isfinite(gn):
            n_skip += 1
            opt.zero_grad()
            continue
        opt.step()
        sch.step()
        hist.append(float(loss))
        if (it + 1) % 100 == 0:
            print(f"[CTL] {it+1}/{steps} loss={np.mean(hist[-100:]):.4f} "
                  f"skip={n_skip} filt={n_filt} lr={sch.get_last_lr()[0]:.2e} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if (it + 1) % 1000 == 0 or it + 1 == steps:
            v = quick_eval(fc, sampler, dev)
            print(f"[CTL]   val clean native-loss={v:.4f}", flush=True)
        if (it + 1) % 2500 == 0 or it + 1 == steps:
            torch.save(fc.module.state_dict(), os.path.join(CK, f"{tag}.pt"))
    torch.save(fc.module.state_dict(), os.path.join(CK, f"{tag}.pt"))
    log = {"arm": "CPT control (no missing)",
           "iface": "native (retainer: content + flag, no surgery)",
           "regime": REGIME, "init": "stock",
           "model": "models_local/moirai-2.0-R-small", "seed": args.seed,
           "steps": steps, "bs": bs,
           "bs_note": "effective batch 1024 in a single micro-batch: 36 tokens "
                      "per series, no gradient accumulation needed",
           "loss": "released PackedQuantileMAELoss on the training_mode module "
                   "output; last context token's 4 slots -> 4 future tokens "
                   "(the H=64 inference map)",
           "control_note": "identical to s61 train_s61.py except regime='filtered' "
                           "(t45.augment no-missingness branch): every window clean, "
                           "obs all ones; the corpus-quality filter kept identical",
           "lr": args.lr, "warmup": args.warmup,
           "corpus_filter": "skip window if max|(fut-loc)/scale| > 100 "
                            "(observed-only context stats, sqrt(var+1e-5) floor) "
                            "-- Moirai 2.0-style data-quality filtering; see s61 header",
           "n_filtered": n_filt,
           "loss_hist_tail": hist[-100:], "n_skip": n_skip,
           "minutes": (time.time() - t0) / 60, "smoke": args.smoke}
    json.dump(log, open(os.path.join(CK, f"{tag}_log.json"), "w"))
    print(f"[CTL] done in {log['minutes']:.1f} min -> {tag}.pt", flush=True)


if __name__ == "__main__":
    main()
