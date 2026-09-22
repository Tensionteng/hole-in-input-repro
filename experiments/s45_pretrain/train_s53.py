#!/usr/bin/env python
"""S53 (stage 3): CPT-from-stock -- retrofitting a shipped checkpoint.

Arms (all bolt-tiny, ALL initialised from the stock chronos-bolt-tiny checkpoint):
  P0  stock, untouched (reference; "--arm P0" just dumps the state dict)
  P1  P0 weights through the dual interface, zero training (symlink to P0's ckpt)
  P2  dual + input_patch_embedding only, 600 steps bs48 lr1e-4 on mechdiv
      (the S27 adapter recipe, re-run on the S45 corpus)
  P3  P2's adapter, then full-model CPT on block_filldiv (the H recipe),
      5k steps bs1024 lr1e-4  -- the two-stage prescription
  P4  native + full-model CPT on filtered, 5k steps bs1024 lr1e-4  (budget control)
  P5  dual + full-model CPT on block_filldiv directly, 5k steps     (no stage 1)

Design doc: DESIGN3.md. Gate G2' (encode replica vs stock forward with STOCK
weights) runs before any training.
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

torch.backends.cuda.matmul.allow_tf32 = True   # same convention as train_s45.py
torch.backends.cudnn.allow_tf32 = True

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s27_interface"))
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import train_s45 as t45

L, H = s25.L, 64
SEED = 20260826
CK = t45.CK

ARMS = {
    #                iface      stage-1 adapter?   stage-2 regime (None = adapter only)
    "P2": {"iface": "dual",   "stage1": True,  "regime2": None},
    "P3": {"iface": "dual",   "stage1": True,  "regime2": "block_filldiv"},
    "P4": {"iface": "native", "stage1": False, "regime2": "filtered"},
    "P5": {"iface": "dual",   "stage1": False, "regime2": "block_filldiv"},
    # S59 fix arm: H recipe + declaration endpoint (zero-fill blindness found by S59)
    "P6": {"iface": "dual",   "stage1": False, "regime2": "block_declblend"},
    # exploration arms (compute is cheap; pre-registered as exploration, not confirmatory)
    "P7": {"iface": "dual",   "stage1": False, "regime2": "block_declblend_ffill"},
    "P8": {"iface": "dual",   "stage1": False, "regime2": "mechdiv"},
    "P9": {"iface": "dual",   "stage1": False, "regime2": "block_declblend",
           "clean_p": 0.75},
    # P10: mechdiv CPT + clean-anchor distillation -- the "worst case should be
    # parity" arm: on untouched windows the loss pulls toward the STOCK model's
    # own quantiles, so clean behaviour cannot drift far from stock.
    "P10": {"iface": "dual",  "stage1": False, "regime2": "mechdiv",
            "distill": 10.0},
}
S1_STEPS, S1_BS, S1_LR = 600, 48, 1e-4       # the S27 adapter recipe
S2_STEPS, S2_BS, S2_LR, S2_WARMUP = 5000, 1024, 1e-4, 100


def build_stock(dev, size="tiny"):
    """The shipped checkpoint, not a random init. models_local ships configs only
    where the stock weights live in the HF cache (bolt-base); tiny/small have full
    local copies."""
    from chronos.chronos_bolt import ChronosBoltModelForForecasting
    local = os.path.join(ROOT, "models_local", f"chronos-bolt-{size}")
    src = local if os.path.exists(os.path.join(local, "model.safetensors")) \
        else f"amazon/chronos-bolt-{size}"
    m = ChronosBoltModelForForecasting.from_pretrained(src).to(dev)
    for p in m.parameters():
        p.requires_grad_(True)
    return m


def stage1_adapter(model, sampler, dev, seed, smoke=False):
    """S27's adapter on the S45 corpus: freeze all but input_patch_embedding."""
    steps, bs = (40, 48) if smoke else (S1_STEPS, S1_BS)
    for p in model.parameters():
        p.requires_grad_(False)
    mod = model.input_patch_embedding
    for p in mod.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(mod.parameters(), lr=S1_LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    gen = sampler.window(require_clean_ctx=True)
    levels = list(model.chronos_config.quantiles)
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
        xt = torch.from_numpy(np.stack(ctxs)).to(dev)
        ot = torch.from_numpy(np.stack(obss)).to(dev)
        yt = torch.from_numpy(np.stack(futs)).to(dev)
        q = t45.quantiles_fwd(model, xt, ot, "dual")[:, :, :H]
        loss = t45.pinball(q, yt, levels)
        if not torch.isfinite(loss):
            n_skip += 1
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        if not torch.isfinite(gn):
            n_skip += 1
            opt.zero_grad()
            continue
        opt.step()
        sch.step()
        hist.append(float(loss))
        if (it + 1) % 150 == 0:
            print(f"  [stage1] {it+1}/{steps} loss={np.mean(hist[-150:]):.4f} "
                  f"skip={n_skip}", flush=True)
    for p in model.parameters():
        p.requires_grad_(True)
    print(f"  [stage1] done in {(time.time()-t0)/60:.1f} min", flush=True)
    return {"loss_tail": hist[-50:], "n_skip": n_skip}


def stage2_cpt(model, sampler, dev, seed, regime, iface, smoke=False, clean_p=0.5,
               distill=0.0, ref_model=None):
    """Full-model CPT: identical loop to train_s45.py except the init and budget.
    distill > 0 adds, on CLEAN rows only (obs all ones), a lambda * MSE pull toward
    the frozen stock model's quantiles -- the clean anchor."""
    steps, bs = (100, 128) if smoke else (S2_STEPS, S2_BS)
    opt = torch.optim.AdamW(model.parameters(), lr=S2_LR, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / S2_WARMUP) * (1 - t / steps))
    rng = np.random.default_rng(seed)
    gen = sampler.window(require_clean_ctx=True)
    levels = list(model.chronos_config.quantiles)
    hist, n_skip, wi, t0 = [], 0, 0, time.time()
    for it in range(steps):
        ctxs, obss, futs = [], [], []
        while len(ctxs) < bs:
            c, f = next(gen)
            c, o = t45.augment(c, regime, rng, wi, clean_p=clean_p)
            wi += 1
            ctxs.append(c)
            obss.append(o)
            futs.append(f)
        xt = torch.from_numpy(np.stack(ctxs)).to(dev)
        ot = torch.from_numpy(np.stack(obss)).to(dev)
        yt = torch.from_numpy(np.stack(futs)).to(dev)
        q = t45.quantiles_fwd(model, xt, ot, iface)[:, :, :H]
        loss = t45.pinball(q, yt, levels)
        if distill > 0.0:
            clean_rows = (ot.sum(dim=1) == ot.shape[1])
            if clean_rows.any():
                with torch.no_grad():
                    q_ref = t45.quantiles_fwd(
                        ref_model, xt[clean_rows],
                        torch.ones_like(ot[clean_rows]), "native")[:, :, :H]
                # window-scaled L1 in quantile space: scale-free, O(0.1-1)
                sc = yt[clean_rows].std(dim=1).view(-1, 1, 1).clamp_min(1e-3)
                loss = loss + distill * (
                    (q[clean_rows] - q_ref).abs() / sc).mean()
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
            print(f"  [stage2:{regime}] {it+1}/{steps} loss={np.mean(hist[-100:]):.4f} "
                  f"skip={n_skip} lr={sch.get_last_lr()[0]:.2e} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
        if (it + 1) % 1000 == 0 or it + 1 == steps:
            v = t45.quick_eval(model, sampler, ("m4_daily", "ushcn_daily", "m5"),
                               {"iface": iface}, dev)
            print(f"  [stage2:{regime}]   val clean pinball={v:.4f}", flush=True)
    return {"loss_tail": hist[-100:], "n_skip": n_skip}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["P0"] + list(ARMS))
    ap.add_argument("--size", default="tiny", choices=["tiny", "small", "base"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device
    os.makedirs(CK, exist_ok=True)
    model = build_stock(dev, args.size)
    d = t45.gate_g2(model, dev)          # G2': replica vs stock forward, STOCK weights

    if args.arm == "P0":
        torch.save(model.state_dict(), os.path.join(CK, f"arm_P0_{args.size}.pt"))
        print(f"[P0] stock {args.size} dump saved (G2' max|diff|={d:.2e})", flush=True)
        return

    arm = ARMS[args.arm]
    sampler = t45.Sampler(seed=args.seed + 1)
    log = {"arm": args.arm, **arm, "init": "stock", "size": args.size,
           "seed": args.seed, "g2_stock": d, "smoke": args.smoke}
    t0 = time.time()
    if arm["stage1"]:
        log["stage1"] = stage1_adapter(model, sampler, dev, args.seed,
                                       smoke=args.smoke)
    if arm["regime2"] is not None:
        ref = None
        if arm.get("distill", 0.0) > 0.0:
            ref = build_stock(dev, args.size)
            ref.eval()
            for p in ref.parameters():
                p.requires_grad_(False)
        log["stage2"] = stage2_cpt(model, sampler, dev, args.seed + 1,
                                   arm["regime2"], arm["iface"], smoke=args.smoke,
                                   clean_p=arm.get("clean_p", 0.5),
                                   distill=arm.get("distill", 0.0), ref_model=ref)
    tag = f"arm_{args.arm}_{args.size}" + (f"_s{args.seed}" if args.seed != SEED else "")
    torch.save(model.state_dict(), os.path.join(CK, f"{tag}.pt"))
    log["minutes"] = (time.time() - t0) / 60
    json.dump(log, open(os.path.join(CK, f"{tag}_log.json"), "w"))
    print(f"[{args.arm}] done in {log['minutes']:.1f} min -> {tag}.pt", flush=True)


if __name__ == "__main__":
    main()
