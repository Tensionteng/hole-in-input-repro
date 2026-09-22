#!/usr/bin/env python
"""S61 CPT: retrofit the STOCK Moirai 2.0 R-small checkpoint (the mechdiv recipe,
third family -- a RETAINER: fill content and flag already enter the content path,
so no interface surgery; the recipe itself must teach it to USE what it sees).

Identical in spirit to train_s60.py (Chronos-2) / train_s53.py (Chronos-Bolt):
  init     stock models_local/moirai-2.0-R-small (fp32)
  corpus   s45_pretrain/corpus_cache via t45.Sampler (real:synthetic 85:15, capped)
  regime   "mechdiv" (t45.augment, verbatim): 50% of windows clean; the rest get
           mechanism ~ U{mcar, block, mnar_high, mnar_extreme}, rate ~ U(0.05, 0.7);
           half DECLARED -- content zeroed + flag = 0 at masked positions, exactly
           the declaration endpoint in Moirai 2.0's own convention (target=0,
           observed_mask=0) -- half alpha-blend filled (content =
           (1-a)*linear + a*truth, a ~ U(0,1), flag = 0 there)
  iface    native (no surgery): content AND observed_mask are both fed -- the
           augment output (filled_ctx, obs) maps 1:1 onto (target, observed_mask)
  loss     Moirai 2.0's NATIVE training loss: the module in training_mode
           (returns preds, scaled_target) + the released PackedQuantileMAELoss
           from uni2ts.loss.packed, applied to the map the model uses at
           inference with H=64: last context token's num_predict_token=4 slots
           -> the 4 future tokens (structure_multi_predict's non-recursive
           branch). Not a re-implementation.
  budget   5000 steps, bs 1024 (single micro-batch: 36 tokens/series, no
           accumulation needed), AdamW lr 1e-4 wd 0.01, warmup 100, linear decay
           to 0, grad clip 1.0, NaN-gradient guard, fp32 with TF32 matmul
           (train_s45/train_s60 convention; eval rounds run strict fp32)
  seed     20260901 (same as S60's CPT)
  filter   corpus-quality filter (data side, NOT a loss change): skip a window
           when the future lies beyond 100 observed-context stds,
           max|(fut - loc)/scale| > 100, loc/scale = observed-only context
           stats with Moirai 2.0's own sqrt(var+1e-5) floor. Moirai 2.0's
           std scaler has a 1e-5 floor, so near-constant corpus windows whose
           future moves produce |scaled targets| up to 3e4; unfiltered, ~1.3%
           of windows carry >99% of the batch loss (mean pinball spikes to
           3e3 while the median is 0.19). Moirai 2.0's own pretraining applied
           a data filtering mechanism "to filter out non-forecastable, low
           quality time series" (checkpoint README); this rule is that filter
           for the s45 cache. ~1.3% of windows are skipped. Eval is unaffected.
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
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import train_s45 as t45
import m2_iface

L, H = s25.L, 64
SEED = 20260901
CK = os.path.join(HERE, "s61_ckpt")
STEPS, BS, LR, WARMUP = 5000, 1024, 1e-4, 100


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
    ap.add_argument("--device", default="cuda:1")
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

    tag = "arm_CPT" + (f"_s{args.seed}" if args.seed != SEED else "") + \
        (f"_{args.tag}" if args.tag else "")
    hist, n_skip, n_filt, wi, t0 = [], 0, 0, 0, time.time()
    for it in range(steps):
        ctxs, obss, futs = [], [], []
        while len(ctxs) < bs:
            c, f = next(gen)
            c, o = t45.augment(c, "mechdiv", rng, wi)
            wi += 1
            ob = o > 0.5
            n_ob = int(ob.sum())
            if n_ob < 8:
                n_filt += 1
                continue
            mu = float(c[ob].mean())
            var = float(((c[ob] - mu) ** 2).sum() / max(n_ob - 1, 1))
            if float(np.abs((f - mu) / np.sqrt(var + 1e-5)).max()) > 100:
                n_filt += 1                       # corpus-quality filter (header)
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
            print(f"[CPT] {it+1}/{steps} loss={np.mean(hist[-100:]):.4f} "
                  f"skip={n_skip} filt={n_filt} lr={sch.get_last_lr()[0]:.2e} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if (it + 1) % 1000 == 0 or it + 1 == steps:
            v = quick_eval(fc, sampler, dev)
            print(f"[CPT]   val clean native-loss={v:.4f}", flush=True)
        if (it + 1) % 2500 == 0 or it + 1 == steps:
            torch.save(fc.module.state_dict(), os.path.join(CK, f"{tag}.pt"))
    torch.save(fc.module.state_dict(), os.path.join(CK, f"{tag}.pt"))
    log = {"arm": "CPT", "iface": "native (retainer: content + flag, no surgery)",
           "regime": "mechdiv", "init": "stock",
           "model": "models_local/moirai-2.0-R-small", "seed": args.seed,
           "steps": steps, "bs": bs,
           "bs_note": "effective batch 1024 in a single micro-batch: 36 tokens "
                      "per series, no gradient accumulation needed",
           "loss": "released PackedQuantileMAELoss on the training_mode module "
                   "output; last context token's 4 slots -> 4 future tokens "
                   "(the H=64 inference map)",
           "lr": args.lr, "warmup": args.warmup,
           "corpus_filter": "skip window if max|(fut-loc)/scale| > 100 "
                            "(observed-only context stats, sqrt(var+1e-5) floor) "
                            "-- Moirai 2.0-style data-quality filtering; see header",
           "n_filtered": n_filt,
           "loss_hist_tail": hist[-100:], "n_skip": n_skip,
           "minutes": (time.time() - t0) / 60, "smoke": args.smoke}
    json.dump(log, open(os.path.join(CK, f"{tag}_log.json"), "w"))
    print(f"[CPT] done in {log['minutes']:.1f} min -> {tag}.pt", flush=True)


if __name__ == "__main__":
    main()
