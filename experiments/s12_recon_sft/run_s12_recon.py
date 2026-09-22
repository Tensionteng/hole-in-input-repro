#!/usr/bin/env python
"""S12: reconstruction-aware deep fine-tuning (RECON-SFT) -- does explicitly
training the mid-encoder reconstruction signal beat the input-side repair
ladder?

Background. S11 (linear probes on chronos-bolt-base hidden states) found that
for mcar/block missingness the TRUE values behind fully-missing patches are
linearly decodable from mid-encoder layers (l4-l8 peak, R2 = 0.45-0.81) even
though the signal is exactly 0 at the input embedding -- the encoder REBUILDS
it via attention transport (fully-missing patches are hard-masked as keys but
remain queries). The forecast head does not fully exploit this signal. S12
turns that headroom into a training objective and tests whether it pays.

Pre-registered outcomes (both publishable, no tuning toward A):
  Outcome A: RECON-SFT > SFT-mb on the mcar/block grid -> the deep objective
     converts S11's headroom into forecast gains; a new method chapter.
  Outcome B: RECON-SFT ~= SFT-mb -> plain masked-augmentation SFT already uses
     the reconstruction headroom; S11's value is explaining WHY SFT works.

Variants (identical trainable backbone params / steps / lr / augmentation;
the ONLY difference is the auxiliary loss):
  mb      : SFT-mb baseline rerun = run_s6_sft.py variant "mb", same seeds
            (seed = SEED + 1000*0 + DS_IDX -> bit-comparable to s6_sft_ckpt).
  recon   : mb + lambda=1.0 * auxiliary reconstruction loss.
  recon03 : same with lambda=0.3 (sensitivity arm, run if budget allows).

Auxiliary loss (recon variants). Per training step, take the encoder block-6
output hidden state (== hidden_states[6] in S11's convention; verified in
--tiny that it matches encoder.block[5] output to 0.0) at FULLY-MISSING patch
positions of the NaN-form series in the batch (the exact S11 decoding
setting: hard-masked keys that remain queries; linear-form series are excluded
because their gap values leak through the fill). A learnable linear readout
head Linear(768 -> 16) regresses the 16 true values of each such patch in the
CLEAN window's instance-norm z-space (loc/scale of the uncorrupted context,
S11's target space), pointwise MSE, elementwise-clipped at 100 like the main
loss. Total loss = clipped_pinball + lambda * clipped_aux_mse. The head is
trained jointly (same AdamW lr) and dropped at eval -- the forecast
architecture of RECON-SFT is identical to SFT-mb; only the training differs.

Training config (copied from S6, do not tune): LoRA r=16 alpha=32 dropout
0.05 on q/k/v/o (peft 0.20.0, 3.54M trainable), 3000 steps x batch 256,
AdamW lr 1e-4 + 200-step linear warmup, grad-clip 1.0, clipped native pinball
(clip=100). Augmentation: 20% clean; else mcar/block(24) p~U(0.05,0.8); input
50% raw-NaN / 50% linear-filled. Train split = first 70% of the timeline.
All variants of a dataset share the same seed -> identical augmentation
stream (paired training); the recon aux head init consumes extra torch RNG,
so dropout draws differ from the pure-mb run (data stream identical).

Evaluation (S6 protocol exactly, for anchor comparability): 150 test windows
(s5.load_windows(ds, 150, SEED) -- run_s5_extra's sample), 1 mask seed for
mcar/block, deterministic mnar, median quantile, H=96.
  Main grid: {mcar, block} x p {0.1,0.3,0.5,0.7} x {ETTh1, ETTm1, weather} x
    methods {zs:nan, zs:linear, tok:nan (s8_masktoken_{ds}.pt in-domain),
    mb:nan/linear, recon:nan/linear, recon03:nan/linear}.
  Negative controls: {mnar_high, mnar_extreme} x p=0.7 x methods
    {zs:nan, zs:linear, mb:nan, recon:nan, recon03:nan}. No gain expected;
    a gain triggers a leakage audit before any claim.
  relMSE = MSE / paired zs clean MSE (S5/S6 convention); own-clean-relative
  also stored for the SFT variants (S6's in-domain-adaptation guard).

Anchor gate (pre-registered tolerances, fixed-seed pipeline):
  zs cells vs s6_sft_results.json bolt_zs: |ratio-1| <= 2% (expect ~exact).
  mb cells vs s6_sft_results.json sft_mb:  |ratio-1| <= 5%.
  Cells: clean (zs, mb) + {mcar, block} x 4 rates x 3 ds x {zs:nan,
  zs:linear, mb:nan} = 78 cells. Gate = ALL pass; >10% on any cell = hard
  fail, stop and investigate before the full run.

Mechanism diagnostic (title-figure candidate): post-training S11 l6 probe
(full_miss r2_mean, configs {mcar 0.7, block 0.7} x 3 datasets, same
300/300 train/test windows and PCA(64)+Ridge(1.0) pipeline as S11) on
zs / mb / recon / recon03. Question: does training sharpen the representation
(R2 rises) or teach the head to read the existing signal (R2 flat, error
down)? ZS reference from s11_results.json: ETTh1 0.539/0.474, weather
0.802/0.616 (mcar/block).

Subcommands:
  --tiny      smoke test (hook check, S6 loss replication, 3-step train,
              mini eval, mini probe); prints only.
  --train     --variant {mb,recon,recon03} --dataset <ds>
  --eval      --shard i --nshards n [--anchor-subset]
  --anchor    gate vs s6_sft_results.json from the anchor-subset cells
  --probe     --dataset <ds> --model {zs,mb,recon,recon03}
  --merge     shards + probes + anchor -> s12_results.json
  --summary   print the main tables
  --figure    s12.png

Intermediate state lives in s12_ckpt/ (eval_shard*.json, probe_*.json,
anchor.json, s12_<variant>_<ds>.pt); the only root-level artifacts are
run_s12_recon.py, s12_results.json, s12.png, s12_notes.md, s12_*.log.
"""
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import torch

import run_s5_missing as s5
import run_s6_sft as s6
import run_s8_masktoken as s8
import run_s11_probe as s11

L, H = s5.L, s5.H                       # 512, 96
PATCH, NPATCH = s8.PATCH, s8.NPATCH     # 16, 32
L6 = 6                                  # S11 layer index (hidden_states[6])
L6_BLOCK = L6 - 1                       # encoder block whose output that is
DS = ("ETTh1", "ETTm1", "weather")
DS_IDX = s6.DS_IDX
RATES = (0.1, 0.3, 0.5, 0.7)
NEG_MECHS = ("mnar_high", "mnar_extreme")
VARIANTS = ("mb", "recon", "recon03")
LAM = {"mb": 0.0, "recon": 1.0, "recon03": 0.3}
PROBE_MODELS = ("zs",) + VARIANTS
CKPT_DIR = os.path.join(HERE, "s12_ckpt")
RESULTS = os.path.join(HERE, "s12_results.json")
S6_JSON = os.path.join(HERE, "s6_sft_results.json")
TOK_CKPT = s8.TOK_CKPT

MAIN_METHODS = [("zs", "nan"), ("zs", "linear"), ("tok", "nan"),
                ("mb", "nan"), ("mb", "linear"),
                ("recon", "nan"), ("recon", "linear"),
                ("recon03", "nan"), ("recon03", "linear")]
NEG_METHODS = [("zs", "nan"), ("zs", "linear"),
               ("mb", "nan"), ("recon", "nan"), ("recon03", "nan")]
ANCHOR_TOL = {"zs": 0.02, "mb": 0.05}   # fixed-seed pipeline tolerances

PCA_DIM, PCA_SUB = 64, 150_000          # S11 probe constants
PROBE_CFGS = ("mcar:0.7", "block:0.7")


def ckpt_path(variant, ds):
    return os.path.join(CKPT_DIR, f"s12_{variant}_{ds}.pt")


def shard_path(i):
    return os.path.join(CKPT_DIR, f"eval_shard{i}.json")


def probe_path(ds, model):
    return os.path.join(CKPT_DIR, f"probe_{ds}_{model}.json")


# ------------------------------------------------------------ training ----

def train(args):
    """S6 train loop (run_s6_sft.train) copied line-for-line, with the RECON
    auxiliary loss added. For variant=mb the rng/torch consumption is
    identical to S6's mb run (same seed) -> checkpoints should be near
    bit-identical; the anchor gate verifies the forecasts match."""
    ds, variant, lam = args.dataset, args.variant, LAM[args.variant]
    device = "cuda"
    # all variants share the mb seed -> identical augmentation stream
    seed = s5.SEED + 1000 * s6.VAR_IDX["mb"] + DS_IDX[ds]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
    model = pipe.model
    assert not model.instance_norm.use_arcsinh, "loss replication assumes no arcsinh"
    model.train()  # bolt has no dropout/BN; needed only for lora_dropout

    model = s6.wrap_lora(model)
    trainable = [p for p in model.parameters() if p.requires_grad]

    captured = {}
    hook = model.encoder.block[L6_BLOCK].register_forward_hook(
        lambda mod, inp, out: captured.__setitem__("l6", out[0]))
    head = None
    if lam > 0:
        # created AFTER wrap_lora: consumes extra torch RNG, so the recon
        # dropout stream differs from pure-mb (data stream identical)
        head = torch.nn.Linear(model.config.d_model, PATCH).to(device)
    params = trainable + (list(head.parameters()) if head is not None else [])
    print(f"variant={variant} lambda={lam} adapter=lora trainable params: "
          f"{sum(p.numel() for p in trainable):,}"
          + (f" + aux head {sum(p.numel() for p in head.parameters()):,}"
             f" (Linear {model.config.d_model}->{PATCH})" if head is not None
             else ""), flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / args.warmup))

    Xtr = s6.load_X(ds)[: int(0.7 * len(s6.load_X(ds)))]
    N, C = Xtr.shape
    t0 = time.time()
    run = {"loss": 0.0, "raw": 0.0, "aux": 0.0, "fm": 0, "n": 0}
    for step in range(1, args.steps + 1):
        starts = rng.integers(0, N - s5.L - s6.PRED, size=args.batch)
        chans = rng.integers(0, C, size=args.batch)
        ctx = np.stack([Xtr[s:s + s5.L, c] for s, c in zip(starts, chans)])
        fut = np.stack([Xtr[s + s5.L:s + s5.L + s6.PRED, c]
                        for s, c in zip(starts, chans)])
        m = np.stack([s6.aug_mask(rng, ctx[b], "mb") for b in range(args.batch)])
        forms = rng.random(args.batch) < 0.5
        xb_np = np.stack([s6.corrupt(ctx[b:b + 1], m[b:b + 1],
                                     "nan" if forms[b] else "linear")[0]
                          for b in range(args.batch)])
        xb = torch.from_numpy(xb_np).to(device)
        yb = torch.from_numpy(fut).to(device)
        if lam > 0:
            # fully-missing patches of NaN-form series (the S11 setting);
            # target = true values in the CLEAN window's instance-norm space
            fm_np = m.reshape(args.batch, NPATCH, PATCH).all(axis=2) & \
                forms[:, None]
            lc, sc = s11.norm_stats_np(ctx)
            z_np = ((ctx - lc[:, None]) / sc[:, None]
                    ).reshape(args.batch, NPATCH, PATCH).astype(np.float32)
        with torch.enable_grad():
            loss, raw = s6.clipped_pinball(model, xb, yb, args.loss_clip)
            total = loss
            if lam > 0:
                fm_t = torch.from_numpy(fm_np).to(device)
                if bool(fm_t.any()):
                    hsel = captured["l6"][:, :NPATCH][fm_t]           # [n, d]
                    zsel = torch.from_numpy(z_np[fm_np]).to(device)   # [n, 16]
                    aux = torch.clamp((head(hsel) - zsel).square(),
                                      max=args.loss_clip).mean()
                else:  # keep the graph alive on degenerate batches
                    aux = sum(p.sum() for p in head.parameters()) * 0.0
                total = loss + lam * aux
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        captured.pop("l6", None)
        run["loss"] += loss.item()
        run["raw"] += raw.item()
        run["n"] += 1
        if lam > 0:
            run["aux"] += aux.item()
            run["fm"] += int(fm_np.sum())
        if step % args.log_every == 0 or step == 1:
            k = run["n"]
            msg = (f"step {step}/{args.steps} loss={run['loss'] / k:.4f} "
                   f"raw={run['raw'] / k:.2f}")
            if lam > 0:
                msg += (f" aux={run['aux'] / k:.4f} fm/step={run['fm'] / k:.0f}")
            print(f"{msg} ({time.time() - t0:.0f}s)", flush=True)
            run = {"loss": 0.0, "raw": 0.0, "aux": 0.0, "fm": 0, "n": 0}

    hook.remove()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    name = args.ckpt_name or f"s12_{variant}_{ds}.pt"
    ckpt = os.path.join(args.ckpt_dir, name)
    torch.save({
        "adapter": "lora",
        "lora_config": s6.LORA_KW,
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters()
                      if p.requires_grad},
        "aux_head": (None if head is None else
                     {n: p.detach().cpu() for n, p in head.state_dict().items()}),
        "meta": {"variant": variant, "lambda": lam, "dataset": ds,
                 "steps": args.steps, "batch": args.batch, "lr": args.lr,
                 "warmup": args.warmup, "loss_clip": args.loss_clip,
                 "seed": seed, "train_frac": 0.7, "pred_len_train": s6.PRED,
                 "aug": "mcar+block24 p~U(.05,.8) (S6 mb mix, same seed)",
                 "input_forms": "50% raw-NaN / 50% linear-fill; 20% clean series",
                 "aux": (None if lam == 0 else
                         "l6 hidden @ fully-missing NaN-form patches -> "
                         "Linear(768->16), clean-z 16pt MSE clip=100")},
    }, ckpt)
    print(f"saved -> {ckpt} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------ model I/O ----

def raw_model(pipe):
    """The underlying ChronosBoltModelForForecasting (LoRA modules are
    injected in-place, so encoder blocks of the raw model include them)."""
    m = pipe.model
    if hasattr(m, "peft_config"):   # PeftModel wrapper -> unwrap
        return m.base_model.model
    return m


def load_sft_pipe(device, ckpt):
    """S6's SFTBoltModel loading pattern (fresh pipe + wrap_lora + load)."""
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
    sd = torch.load(ckpt, map_location=device)
    model = s6.wrap_lora(pipe.model)
    pipe.model = model
    model.load_state_dict(sd["trainable"], strict=False)
    model.eval()
    pipe._s12_meta = sd["meta"]
    return pipe


class ModelBank:
    """One ZS pipe + one LoRA-swappable SFT pipe per eval shard process."""

    def __init__(self, device):
        self.device = device
        self.zs = s8.load_bolt(device)
        self.sft = None
        self.cur = None
        self.tokens = {}

    def sft_pipe(self):
        if self.sft is None:
            from chronos import BaseChronosPipeline
            pipe = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-bolt-base", device_map=self.device,
                torch_dtype=torch.float32)
            pipe.model = s6.wrap_lora(pipe.model)
            self.sft = pipe
        return self.sft

    def get(self, model, ds):
        if model == "zs":
            return self.zs
        if model == "tok":
            if ds not in self.tokens:
                sd = torch.load(TOK_CKPT[ds], map_location=self.device)
                self.tokens[ds] = sd["mask_token"].to(self.device)
            return self.zs, self.tokens[ds]
        key = (model, ds)
        pipe = self.sft_pipe()
        if self.cur != key:
            sd = torch.load(ckpt_path(*key), map_location=self.device)
            pipe.model.load_state_dict(sd["trainable"], strict=False)
            pipe.model.eval()
            self.cur = key
        return pipe


# ------------------------------------------------------------ evaluation ----

def eval_jobs():
    """Canonical job list (strided sharding). kind in {clean, grid, neg}."""
    jobs = []
    for ds in DS:
        for model in VARIANTS + ("zs",):
            jobs.append(("clean", ds, None, 0.0, model, "none"))
    for ds in DS:
        for mech in ("mcar", "block"):
            for rate in RATES:
                for model, fill in MAIN_METHODS:
                    jobs.append(("grid", ds, mech, rate, model, fill))
    for ds in DS:
        for mech in NEG_MECHS:
            for model, fill in NEG_METHODS:
                jobs.append(("neg", ds, mech, 0.7, model, fill))
    return jobs


def is_anchor_job(job):
    kind, ds, mech, rate, model, fill = job
    if kind == "clean":
        return model in ("zs", "mb")
    if kind == "grid":
        return model == "zs" or (model == "mb" and fill == "nan")
    return False


def job_key(job):
    kind, ds, mech, rate, model, fill = job
    if kind == "clean":
        return f"clean:{model}"
    return f"{mech}:{model}:{fill}:{rate}"


def run_eval_job(bank, X, starts, ds, job):
    kind, _, mech, rate, model, fill = job
    t0 = time.time()
    if kind == "clean":
        pipe = bank.get(model, ds)
        res = s8.clean_baseline(pipe, X, starts)
        res["seconds"] = round(time.time() - t0, 1)
        return res
    ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, mech, rate,
                                                   n_seeds=1)
    if fill == "linear":
        x_in = s8.fill_linear(ctx, msk)
    else:
        x_in = ctx.copy()
        x_in[msk] = np.nan
    if model == "tok":
        pipe, token = bank.get(model, ds)
        with s8.EncodePatch(pipe.model, mask_mode="correct", mask_token=token):
            pred = s8.predict_median(pipe, x_in)
    else:
        pred = s8.predict_median(bank.get(model, ds), x_in)
    mse, mse_pw = s8.mse_windowed(pred, gt, S, C, nw)
    return {"mse": mse, "mse_per_window": mse_pw,
            "achieved_rate": float(msk.mean()),
            "n_windows": nw, "n_seeds": ns,
            "seconds": round(time.time() - t0, 1)}


def run_eval(args):
    jobs = eval_jobs()[args.shard::args.nshards]
    out = shard_path(args.shard)
    results = {}
    if os.path.exists(out):
        results = json.load(open(out))
        print(f"resume: loaded {out}", flush=True)
    results.setdefault("meta", {"track": "S12 RECON-SFT eval",
                                "windows": args.windows, "seeds": 1,
                                "methods": MAIN_METHODS, "neg": NEG_METHODS})
    save = lambda: s5.save_results(results, out)
    bank = ModelBank(args.device)
    loaded = {}
    n_done = 0
    for job in jobs:
        if args.anchor_subset and not is_anchor_job(job):
            continue
        kind, ds, mech, rate, model, fill = job
        sec = "clean" if kind == "clean" else ("neg" if kind == "neg" else "eval")
        key = job_key(job)
        if key in results.setdefault(sec, {}).setdefault(ds, {}):
            continue
        if ds not in loaded:
            loaded[ds] = s5.load_windows(s5.DATASETS[ds], args.windows, s5.SEED)
        X, starts = loaded[ds]
        res = run_eval_job(bank, X, starts, ds, job)
        results[sec][ds][key] = res
        save()
        n_done += 1
        print(f"  {sec:5s} {ds:8s} {key:26s} mse={res['mse']:12.4f} "
              f"({res['seconds']}s)", flush=True)
    print(f"EVAL SHARD {args.shard} DONE ({n_done} new jobs)", flush=True)


# ----------------------------------------------------------------- anchor ---

def run_anchor(args):
    """Compare the anchor-subset cells against s6_sft_results.json."""
    r6 = json.load(open(S6_JSON))
    got = {}
    for i in range(args.nshards):
        p = shard_path(i)
        if os.path.exists(p):
            r = json.load(open(p))
            for sec in ("clean", "eval"):
                for ds, cells in r.get(sec, {}).items():
                    got.setdefault(sec, {}).setdefault(ds, {}).update(cells)
    cells = []
    for ds in DS:
        for model, sec6 in (("zs", "bolt_zs"), ("mb", "sft_mb")):
            ref = r6[sec6][ds]["clean:none:0.0"]["mse"]
            mine = got.get("clean", {}).get(ds, {}).get(f"clean:{model}", {})
            cells.append((ds, "clean", model, mine.get("mse"), ref))
        for mech in ("mcar", "block"):
            for rate in RATES:
                for model, fill, sec6 in (("zs", "nan", "bolt_zs"),
                                          ("zs", "linear", "bolt_zs"),
                                          ("mb", "nan", "sft_mb")):
                    ref = r6[sec6][ds][f"{mech}:{fill}:{rate}"]["mse"]
                    mine = got.get("eval", {}).get(ds, {}).get(
                        f"{mech}:{model}:{fill}:{rate}", {})
                    cells.append((ds, f"{mech}:{fill}:{rate}", model,
                                  mine.get("mse"), ref))
    print("== S12 anchor gate vs s6_sft_results.json ==", flush=True)
    out = {"tolerances": ANCHOR_TOL, "cells": {}}
    gate, n_miss = True, 0
    for ds, cfg, model, mine, ref in cells:
        key = f"{ds}:{cfg}:{model}"
        if mine is None:
            out["cells"][key] = {"ref": ref, "status": "MISSING"}
            n_miss += 1
            gate = False
            print(f"  {key:34s} MISSING", flush=True)
            continue
        ratio = mine / ref
        tol = ANCHOR_TOL[model]
        ok = abs(ratio - 1) <= tol
        gate &= ok
        out["cells"][key] = {"mine": mine, "ref": ref, "ratio": ratio,
                             "tol": tol, "ok": ok}
        print(f"  {key:34s} mine={mine:12.4f} s6={ref:12.4f} "
              f"ratio={ratio:.4f} (tol {tol}) -> {'OK' if ok else 'FAIL'}",
              flush=True)
    out["gate"] = "PASS" if gate else "FAIL"
    out["n_missing"] = n_miss
    s5.save_results(out, os.path.join(CKPT_DIR, "anchor.json"))
    print(f"ANCHOR {out['gate']}", flush=True)
    if not gate:
        raise SystemExit(2)


# ----------------------------------------------------------------- probe ----

@torch.no_grad()
def extract_l6(model, x_in, batch=512):
    """s11.extract_hidden restricted to layer 6 (== encoder block 5 output).
    x_in [N, L] float32, NaN = missing -> [N, NPATCH, d] float16."""
    d = model.config.d_model
    dev = next(model.parameters()).device
    N = len(x_in)
    out = np.empty((N, NPATCH, d), np.float16)
    for i in range(0, N, batch):
        xb = torch.from_numpy(x_in[i:i + batch]).to(dev)
        B = xb.shape[0]
        mask = torch.isnan(xb).logical_not().to(xb.dtype)               # l.280
        xc, _ = model.instance_norm(xb)                                 # l.288
        xc = xc.to(model.dtype)                                         # l.292
        mask = mask.to(model.dtype)                                     # l.293
        pc = model.patch(xc)                                            # l.296
        pm = torch.nan_to_num(model.patch(mask), nan=0.0)               # l.297
        pc = torch.where(pm > 0.0, pc, 0.0)                             # l.298
        pc_in = torch.cat([pc, pm], dim=-1)                             # l.300
        am = pm.sum(dim=-1) > 0                                         # l.303
        emb = model.input_patch_embedding(pc_in)                        # l.305
        reg_ids = torch.full((B, 1), model.config.reg_token_id, device=dev)
        emb = torch.cat([emb, model.shared(reg_ids)], dim=-2)           # l.307-315
        am = torch.cat([am.to(model.dtype),
                        torch.ones_like(reg_ids).to(model.dtype)], dim=-1)
        enc = model.encoder(attention_mask=am, inputs_embeds=emb,
                            output_hidden_states=True)                  # l.324-327
        out[i:i + B] = enc.hidden_states[L6][:, :NPATCH].float().cpu().numpy() \
            .astype(np.float16)
    return out


def probe_split(model, X, starts, mech, p):
    """l6 reps + probe data for one (split, config)."""
    ctx, msk, _, S, C, nw, ns = s8.build_contexts(X, starts, mech, p)  # 2 seeds
    pd_ = s11.make_probe_data(ctx, msk)
    reps = extract_l6(model, pd_["x_nan"])
    return pd_, reps


def run_probe(args):
    ds, mname = args.dataset, args.model
    if mname == "zs":
        pipe = s8.load_bolt(args.device)
    else:
        pipe = load_sft_pipe(args.device, ckpt_path(mname, ds))
    model = raw_model(pipe)
    X, starts_te = s5.load_windows(s5.DATASETS[ds], args.probe_windows, s5.SEED)
    _, starts_tr = s11.load_train_windows(s5.DATASETS[ds], args.probe_windows,
                                          s11.TRAIN_SEED)
    print(f"probe {ds} {mname}: {len(starts_tr)} train / {len(starts_te)} "
          f"test windows, C={X.shape[1]}", flush=True)
    data = {}
    for cfg in PROBE_CFGS:
        mech, p = cfg.split(":")
        for split, starts in (("train", starts_tr), ("test", starts_te)):
            t0 = time.time()
            pd_, reps = probe_split(model, X, starts, mech, float(p))
            data[(cfg, split)] = (pd_, reps)
            print(f"  {cfg} {split}: {len(pd_['x_nan'])} series, full_miss="
                  f"{int(pd_['fam']['full_miss'].sum())} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    # PCA(64) on pooled train patches of both configs (S11 convention)
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(0)
    Xp = np.concatenate([data[(c, "train")][1].reshape(-1, data[(c, "train")][1].shape[-1])
                         for c in PROBE_CFGS]).astype(np.float32)
    if len(Xp) > PCA_SUB:
        Xp = Xp[rng.choice(len(Xp), PCA_SUB, replace=False)]
    pca = PCA(n_components=PCA_DIM, svd_solver="randomized", random_state=0)
    pca.fit(Xp)
    print(f"  pca: fit on {len(Xp)} patches, "
          f"ev={pca.explained_variance_ratio_.sum():.3f}", flush=True)
    res = {"dataset": ds, "model": mname, "layer": L6,
           "ev_pca": float(pca.explained_variance_ratio_.sum())}
    for cfg in PROBE_CFGS:
        dtr, rtr = data[(cfg, "train")]
        dte, rte = data[(cfg, "test")]
        ztr = pca.transform(rtr.reshape(-1, rtr.shape[-1]).astype(np.float32))
        zte = pca.transform(rte.reshape(-1, rte.shape[-1]).astype(np.float32))
        ftr, fte = dtr["fam"]["full_miss"].ravel(), dte["fam"]["full_miss"].ravel()
        r2 = s11.probe_r2(ztr[ftr], dtr["y_mean"].ravel()[ftr],
                          zte[fte], dte["y_mean"].ravel()[fte])
        res[cfg] = {"r2_l6_fullmiss": r2, "n_train": int(ftr.sum()),
                    "n_test": int(fte.sum()),
                    "y_var_test": float(np.var(dte["y_mean"].ravel()[fte]))
                    if fte.sum() else None}
        print(f"  {cfg}: r2_l6={r2} (n_tr={int(ftr.sum())}, "
              f"n_te={int(fte.sum())})", flush=True)
    s5.save_results(res, probe_path(ds, mname))
    print(f"PROBE {ds} {mname} DONE", flush=True)


# ----------------------------------------------------------------- merge ----

def run_merge(args):
    out = {"meta": {
        "track": "S12 reconstruction-aware deep SFT (RECON-SFT) on bolt-base",
        "seed": s5.SEED, "L": L, "H": H, "patch": PATCH, "n_patch": NPATCH,
        "variants": {v: {"lambda": LAM[v]} for v in VARIANTS},
        "train": ("LoRA r=16 a=32 do=0.05 q/k/v/o; 3000 steps x 256; AdamW "
                  "1e-4 + 200 warmup; grad-clip 1.0; clipped pinball 100; "
                  "aug mcar+block24 p~U(.05,.8), 20% clean, 50% nan/50% linear"),
        "aux_loss": ("l6 (encoder block 6 output) hidden at fully-missing "
                     "NaN-form patches -> Linear(768->16) -> clean-z 16-point "
                     "MSE, clip=100; head dropped at eval"),
        "eval": ("150 windows (s5.load_windows(ds,150,SEED)), 1 mask seed, "
                 "median; relMSE vs paired zs clean; own-clean-relative also "
                 "stored for SFT variants"),
        "neg_controls": f"{NEG_MECHS} p=0.7",
        "probe": (f"l6 full_miss r2_mean, cfgs {PROBE_CFGS}, 300 train + 300 "
                  f"test windows (S11 windows/seeds), PCA(64)+Ridge(1.0)"),
        "pre_registered_outcomes": {
            "A": "RECON-SFT > SFT-mb on mcar/block grid (deep objective pays)",
            "B": "RECON-SFT ~= SFT-mb (SFT already uses the headroom)"},
        "anchor_tolerances": ANCHOR_TOL,
    }}
    ap = os.path.join(CKPT_DIR, "anchor.json")
    if os.path.exists(ap):
        out["anchor"] = json.load(open(ap))
    for sec in ("clean", "eval", "neg"):
        out[sec] = {}
    for i in range(64):
        p = shard_path(i)
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        for sec in ("clean", "eval", "neg"):
            for ds, cells in r.get(sec, {}).items():
                out[sec].setdefault(ds, {}).update(cells)
    # relMSE columns
    for ds in DS:
        cz = out.get("clean", {}).get(ds, {}).get("clean:zs", {}).get("mse")
        for sec in ("eval", "neg"):
            for key, cell in out.get(sec, {}).get(ds, {}).items():
                model = key.split(":")[1] if sec != "clean" else None
                if cz:
                    cell["rel_zs"] = cell["mse"] / cz
                if model in VARIANTS:
                    cm = out["clean"].get(ds, {}).get(f"clean:{model}", {})
                    if cm.get("mse"):
                        cell["rel_own"] = cell["mse"] / cm["mse"]
    # probes + S11 ZS reference
    out["probe"] = {}
    for ds in DS:
        for mname in PROBE_MODELS:
            p = probe_path(ds, mname)
            if os.path.exists(p):
                r = json.load(open(p))
                for cfg in PROBE_CFGS:
                    out["probe"].setdefault(ds, {}).setdefault(cfg, {})[mname] = r[cfg]
    s11r = json.load(open(os.path.join(HERE, "s11_results.json")))
    out["probe_s11_zs_ref"] = {
        ds: {cfg: s11r["probe"][ds][cfg]["families"]["full_miss"]["r2_mean"][L6]
             for cfg in PROBE_CFGS} for ds in ("ETTh1", "weather")}
    # summary tables
    out["tables"] = build_tables(out)
    s5.save_results(out, RESULTS)
    print(f"merged -> {RESULTS}", flush=True)


def build_tables(out):
    """Main table: dataset-avg + per-dataset relMSE (vs zs clean) per
    (mech, rate, method); neg table; probe table."""
    tabs = {"main": {}, "neg": {}, "probe": {}}
    methods = [f"{m}:{f}" for m, f in MAIN_METHODS]
    for mech in ("mcar", "block"):
        for rate in RATES:
            row = {}
            for mth in methods:
                vals = []
                for ds in DS:
                    cell = out.get("eval", {}).get(ds, {}).get(
                        f"{mech}:{mth}:{rate}")
                    vals.append(None if cell is None else cell.get("rel_zs"))
                row[mth] = {"avg": (float(np.mean(vals)) if all(v is not None for v in vals) else None),
                            **{ds: v for ds, v in zip(DS, vals)}}
            tabs["main"][f"{mech}:{rate}"] = row
    for mech in NEG_MECHS:
        row = {}
        for mth in [f"{m}:{f}" for m, f in NEG_METHODS]:
            rels, owns = [], []
            for ds in DS:
                cell = out.get("neg", {}).get(ds, {}).get(f"{mech}:{mth}:0.7")
                rels.append(None if cell is None else cell.get("rel_zs"))
                owns.append(None if cell is None else cell.get("rel_own"))
            row[mth] = {
                "avg": (float(np.mean(rels)) if all(v is not None for v in rels) else None),
                "avg_own": (float(np.mean(owns)) if all(v is not None for v in owns) else None),
                **{ds: v for ds, v in zip(DS, rels)}}
        tabs["neg"][mech] = row
    for ds in DS:
        for cfg in PROBE_CFGS:
            tabs["probe"][f"{ds}:{cfg}"] = {
                m: out.get("probe", {}).get(ds, {}).get(cfg, {}).get(m, {}).get("r2_l6_fullmiss")
                for m in PROBE_MODELS}
    return tabs


def run_summary(args):
    out = json.load(open(RESULTS))
    print("== anchor gate:", out.get("anchor", {}).get("gate"), "==")
    methods = [f"{m}:{f}" for m, f in MAIN_METHODS]
    for mech in ("mcar", "block"):
        print(f"\n== main grid relMSE (vs zs clean), {mech} == ")
        print(f"{'rate':5s} " + " ".join(f"{m:>12s}" for m in methods))
        for rate in RATES:
            row = out["tables"]["main"][f"{mech}:{rate}"]
            print(f"{rate:<5.1f} " + " ".join(
                f"{(row[m]['avg'] if row[m]['avg'] is not None else float('nan')):12.4f}"
                for m in methods))
        # rate-average
        avg = {}
        for m in methods:
            vals = [out["tables"]["main"][f"{mech}:{r}"][m]["avg"] for r in RATES]
            avg[m] = float(np.mean(vals)) if all(v is not None for v in vals) else None
        print(f"{'avg':5s} " + " ".join(
            f"{(avg[m] if avg[m] is not None else float('nan')):12.4f}" for m in methods))
    print("\n== negative controls p=0.7 (relMSE vs zs clean | own-clean) ==")
    for mech in NEG_MECHS:
        row = out["tables"]["neg"][mech]
        for mth in [f"{m}:{f}" for m, f in NEG_METHODS]:
            e = row[mth]
            print(f"  {mech:14s} {mth:12s} avg={e['avg']} avg_own={e['avg_own']} "
                  f"per-ds={[round(e[ds], 4) if e[ds] is not None else None for ds in DS]}")
    print("\n== probe l6 full_miss R2 (zs ref from S11 in brackets) ==")
    ref = out.get("probe_s11_zs_ref", {})
    for ds in DS:
        for cfg in PROBE_CFGS:
            t = out["tables"]["probe"].get(f"{ds}:{cfg}", {})
            r = ref.get(ds, {}).get(cfg)
            print(f"  {ds:8s} {cfg:10s} " + " ".join(
                f"{m}={t.get(m) if t.get(m) is not None else float('nan'):.4f}"
                for m in PROBE_MODELS) + (f"  [s11 zs={r:.4f}]" if r else ""))


# ---------------------------------------------------------------- figure ----

def make_figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    show = [("zs:nan", "ZS native nan", "tab:blue", "-", "o"),
            ("zs:linear", "ZS linear fill", "tab:cyan", "-", "s"),
            ("tok:nan", "[MASK] token (S8)", "tab:orange", "-", "^"),
            ("mb:nan", "SFT-mb (S6 rerun)", "tab:green", "-", "d"),
            ("recon:nan", "RECON-SFT λ=1", "tab:red", "-", "*"),
            ("recon03:nan", "RECON-SFT λ=0.3", "tab:pink", "--", "v")]
    for ai, mech in enumerate(("mcar", "block")):
        ax = axes[0][ai]
        for mth, label, c, ls, mk in show:
            ys = [out["tables"]["main"][f"{mech}:{r}"][mth]["avg"] for r in RATES]
            if any(v is None for v in ys):
                continue
            ax.plot(RATES, ys, color=c, ls=ls, marker=mk, ms=4, label=label)
        ax.axhline(1.0, color="grey", lw=0.6, ls=":")
        ax.set_title(f"{mech}: relMSE vs missing rate (dataset-avg)")
        ax.set_xlabel("nominal missing rate p")
        ax.set_ylabel("relMSE (vs paired ZS clean)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    # probe shift
    ax = axes[1][0]
    labels, groups = [], {}
    for ds in DS:
        for cfg in PROBE_CFGS:
            labels.append(f"{ds}\n{cfg}")
            groups[f"{ds}:{cfg}"] = out["tables"]["probe"].get(f"{ds}:{cfg}", {})
    x = np.arange(len(labels))
    w = 0.2
    pcols = {"zs": "tab:blue", "mb": "tab:green", "recon": "tab:red",
             "recon03": "tab:pink"}
    for mi, m in enumerate(PROBE_MODELS):
        ys = [groups[k].get(m) for k in groups]
        ys = [np.nan if v is None else v for v in ys]
        ax.bar(x + (mi - 1.5) * w, ys, w, label=m, color=pcols[m])
    ref = out.get("probe_s11_zs_ref", {})
    for xi, k in enumerate(groups):
        ds, cfg = k.split(":", 1)
        v = ref.get(ds, {}).get(cfg)
        if v is not None:
            ax.plot([xi - 1.7 * w, xi - 1.3 * w], [v, v], color="k", lw=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("l6 full-miss probe R²")
    ax.set_title("mechanism diagnostic: l6 reconstruction signal after training\n"
                 "(black tick = S11 ZS reference)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    # negative controls
    ax = axes[1][1]
    nshow = [("zs:nan", "ZS nan"), ("zs:linear", "ZS linear"),
             ("mb:nan", "SFT-mb"), ("recon:nan", "RECON λ=1"),
             ("recon03:nan", "RECON λ=0.3")]
    x = np.arange(len(NEG_MECHS))
    w = 0.16
    for mi, (mth, label) in enumerate(nshow):
        ys = [out["tables"]["neg"][mech][mth]["avg"] for mech in NEG_MECHS]
        ys = [np.nan if v is None else v for v in ys]
        ax.bar(x + (mi - 2) * w, ys, w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\np=0.7" for m in NEG_MECHS], fontsize=8)
    ax.set_ylabel("relMSE (vs paired ZS clean)")
    ax.set_title("negative controls: MNAR (no gain expected)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    gate = out.get("anchor", {}).get("gate", "?")
    fig.suptitle(f"S12 RECON-SFT vs SFT-mb (chronos-bolt-base) — anchor gate {gate}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}", flush=True)


# ------------------------------------------------------------------ tiny ----

def run_tiny(args):
    device = args.device
    pipe = s8.load_bolt(device)
    model = pipe.model
    X, starts = s5.load_windows(s5.DATASETS["ETTh1"], 6, s5.SEED)
    ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, "block", 0.5,
                                                   n_seeds=1)
    x_nan = ctx.copy()
    x_nan[msk] = np.nan

    # 1) hook captures hidden_states[6] exactly (S11 encode replica inline)
    captured = {}
    h = model.encoder.block[L6_BLOCK].register_forward_hook(
        lambda mod, inp, out: captured.__setitem__("l6", out[0]))
    xb = torch.from_numpy(x_nan[:64]).to(device)
    B = xb.shape[0]
    mask = torch.isnan(xb).logical_not().to(xb.dtype)
    xc, _ = model.instance_norm(xb)
    xc = xc.to(model.dtype)
    mask = mask.to(model.dtype)
    pc = model.patch(xc)
    pm = torch.nan_to_num(model.patch(mask), nan=0.0)
    pc = torch.where(pm > 0.0, pc, 0.0)
    pc_in = torch.cat([pc, pm], dim=-1)
    am = pm.sum(dim=-1) > 0
    emb = model.input_patch_embedding(pc_in)
    reg_ids = torch.full((B, 1), model.config.reg_token_id, device=xb.device)
    emb = torch.cat([emb, model.shared(reg_ids)], dim=-2)
    am = torch.cat([am.to(model.dtype), torch.ones_like(reg_ids).to(model.dtype)], dim=-1)
    with torch.no_grad():
        enc = model.encoder(attention_mask=am, inputs_embeds=emb,
                            output_hidden_states=True)
    d = (enc.hidden_states[L6] - captured["l6"]).abs().max().item()
    print(f"tiny: hook-vs-hidden_states[{L6}] max|diff| = {d:.3e} (want 0)",
          flush=True)
    assert d == 0.0, "l6 hook mismatch"
    h.remove()

    # 2) mask stream of build_contexts(n_seeds=1) == s5.make_mask ms=0 (S6)
    m_ref = np.stack([s5.make_mask("block", 0.5, wi, 0, C,
                                   x=X[s:s + L].T.copy())
                      for wi, s in enumerate(starts)])
    assert (m_ref == msk.reshape(nw, ns, C, L)[:, 0]).all(), "mask stream mismatch"
    print("tiny: build_contexts(n_seeds=1) mask stream == S6 eval stream",
          flush=True)

    # 3) mb training replicates S6 step-1 loss (same seed/batch/data)
    args.steps, args.log_every = 3, 1
    args.variant, args.dataset = "mb", "ETTh1"
    args.ckpt_name = "tiny_mb_ETTh1.pt"
    print("tiny: training mb 3 steps (step-1 loss must match S6's 37.6535)",
          flush=True)
    train(args)

    # 4) recon training smoke: aux finite, grads flow to head and lora
    args.variant = "recon"
    args.ckpt_name = "tiny_recon_ETTh1.pt"
    print("tiny: training recon 3 steps (aux loss active)", flush=True)
    train(args)

    # 5) eval path smoke incl. SFT ckpt load + token
    bank = ModelBank(device)
    bank.sft_pipe()
    for job in [("grid", "ETTh1", "mcar", 0.3, "zs", "nan"),
                ("grid", "ETTh1", "mcar", 0.3, "tok", "nan")]:
        res = run_eval_job(bank, X, starts, "ETTh1", job)
        print(f"tiny: eval {job_key(job):24s} mse={res['mse']:.4f}", flush=True)
    for name, mname, fill in (("tiny_mb_ETTh1.pt", "mb", "linear"),
                              ("tiny_recon_ETTh1.pt", "recon", "nan")):
        sd = torch.load(os.path.join(args.ckpt_dir, name), map_location=device)
        bank.sft.model.load_state_dict(sd["trainable"], strict=False)
        bank.sft.model.eval()
        bank.cur = (mname, "ETTh1")
        job = ("grid", "ETTh1", "block", 0.3, mname, fill)
        res = run_eval_job(bank, X, starts, "ETTh1", job)
        print(f"tiny: eval {job_key(job):24s} mse={res['mse']:.4f}", flush=True)

    # 6) probe path smoke (5 train + 5 test windows, zs)
    _, starts_tr = s11.load_train_windows(s5.DATASETS["ETTh1"], 5, s11.TRAIN_SEED)
    dtr, rtr = probe_split(model, X, starts_tr, "block", 0.7)
    dte, rte = probe_split(model, X, starts, "block", 0.7)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(PCA_DIM, len(rtr.reshape(-1, 768)) - 1),
              svd_solver="randomized", random_state=0)
    pca.fit(rtr.reshape(-1, 768).astype(np.float32))
    ztr = pca.transform(rtr.reshape(-1, 768).astype(np.float32))
    zte = pca.transform(rte.reshape(-1, 768).astype(np.float32))
    ftr, fte = dtr["fam"]["full_miss"].ravel(), dte["fam"]["full_miss"].ravel()
    r2 = s11.probe_r2(ztr[ftr], dtr["y_mean"].ravel()[ftr],
                      zte[fte], dte["y_mean"].ravel()[fte])
    print(f"tiny: probe block:0.7 full_miss r2={r2} "
          f"(n_tr={int(ftr.sum())}, n_te={int(fte.sum())})", flush=True)
    for f in ("tiny_mb_ETTh1.pt", "tiny_recon_ETTh1.pt"):
        os.remove(os.path.join(args.ckpt_dir, f))
    print("TINY OK", flush=True)


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--anchor-subset", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--variant", choices=list(VARIANTS), default="mb")
    ap.add_argument("--dataset", choices=list(DS), default="ETTh1")
    ap.add_argument("--model", choices=list(PROBE_MODELS), default="zs")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--loss-clip", type=float, default=100.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--windows", type=int, default=150)
    ap.add_argument("--probe-windows", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--ckpt-name", default=None)
    ap.add_argument("--fig-out", default=os.path.join(HERE, "s12.png"))
    args = ap.parse_args()

    torch.manual_seed(s5.SEED)
    np.random.seed(s5.SEED)

    if args.tiny:
        run_tiny(args)
        return
    if args.train:
        train(args)
        return
    if args.eval:
        run_eval(args)
        return
    if args.anchor:
        run_anchor(args)
        return
    if args.probe:
        run_probe(args)
        return
    if args.merge:
        run_merge(args)
        return
    if args.summary:
        run_summary(args)
        return
    if args.figure:
        make_figure(json.load(open(RESULTS)), args.fig_out)
        return
    raise SystemExit("nothing to do")


if __name__ == "__main__":
    main()
