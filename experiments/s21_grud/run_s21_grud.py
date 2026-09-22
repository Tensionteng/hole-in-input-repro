#!/usr/bin/env python
"""S21: GRU-D-style missingness-aware INPUT features for chronos-bolt-base --
does a richer structured missingness signal beat the native 0/1 flag once
SFT-mb is in place?

Background. S8 showed bolt's concatenated 0/1 mask channel is strongly causal
but acts only through the embedding/value pathway. GRU-D (Che et al. 2018)
showed for RNNs that the missingness PATTERN itself is informative when given
as mask + time-since-last-observation (+ decay). S20 showed mechanism text
prompts are inert -- mechanism information must enter through STRUCTURED
inputs. S21 ports the GRU-D idea to bolt's input layer: per-point features
computed ONLY from the observed/missing indicator, patchified exactly like
bolt's own mask channel, and projected into the patch embedding by a
ZERO-INITIALIZED linear map (the S5F miss_proj pattern), so at init every
variant is bit-identical to plain SFT-mb.

Features (per time point t, from the observed indicator o_t in {0,1} only --
values are never read):
  delta_t  = log(1 + time since last observed point)   (GRU-D core; 0 if t observed)
  gap_t    = log(1 + distance to next observed point)  (forward-looking twin)
  rate_t   = cumulative observed fraction up to t      (obs_rate_so_far)
Each feature [B, L] is patchified with model.patch to [B, NPATCH, PATCH] and
concatenated feature-major into [B, NPATCH, k*PATCH]; the patch embedding
output gets  emb += W @ feats + b  with W, b zero-initialized.

Variants (trainable = S6/S12 LoRA set + zero-init feature projection;
EVERYTHING else identical to SFT-mb: LoRA r=16/a=32/do=0.05 on q/k/v/o, 3000
steps x 256, AdamW 1e-4 + 200 warmup, grad-clip 1.0, clipped pinball clip=100,
S6-mb augmentation mcar+block24 p~U(0.05,0.8), 20% clean, 50% NaN/50% linear
input, first-70% train split, seed = SEED + DS_IDX (the mb seed)):
  flag  -- plain SFT-mb rerun (anchor; expected bit-identical to s12_mb_*)
  delta -- flag + delta_t feature (k=1)
  full  -- flag + delta_t + gap_ahead + obs_rate (k=3)
Zero-init parameters are created with torch.zeros (no RNG consumption) and
the encode replica adds no RNG calls, so all three variants share the mb data
stream AND the torch dropout stream exactly; delta/full diverge from flag only
through the learned feature pathway.

Eval (S6/S12 protocol): 150 windows (s5.load_windows(ds,150,SEED)), 1 mask
seed, deterministic mnar, median quantile, H=96, NaN native fill only.
Grid: {mcar, block, mnar_high, mnar_extreme} x p {0.1,0.3,0.5,0.7} x
{ETTh1, ETTm1, weather} x {flag, delta, full}; clean cells for zs + all three
variants. relMSE = MSE / paired zs clean MSE; own-clean-relative stored too.
Per-window mse / level^2 / signed bias stored for paired tests and the S19
level-bias decomposition (does the mnar under-prediction shrink?).

Pre-registered primary endpoint: paired per-window test (variant - flag) of
relMSE, pooled over block+mnar_high+mnar_extreme (150 w x 4 rates x 3 ds x
3 mechs = 5400 window-means). EFFECTIVE iff relative improvement > 2% AND the
95% CI excludes 0 for delta or full; otherwise SFT-mb has saturated the
missingness signal (consistent with S12/S15 redundancy). mcar is secondary.

Anchor gate: flag cells vs s12_results.json (clean: flag vs clean:mb tol 5%,
zs vs clean:zs tol 2%; grid: mcar/block x 4 rates x 3 ds flag:nan vs
{mech}:mb:nan:{rate} tol 5%; expected bit-exact). Additionally the flag
checkpoints are compared tensor-wise (torch.equal) against s12_ckpt/s12_mb_*.

Negative/clean checks: clean:{delta,full}/clean:flag ratio ~1; leakage
assertion -- features computed after randomly permuting the observed VALUES
(mask fixed) must be bit-identical (features read the mask only).

Subcommands:
  --tiny      smoke (feature correctness vs numpy ref, leak assertion,
              zero-init bit-identity vs native, mask-stream check, 3-step
              train per variant with step-1 loss == S6's 37.6535, eval smoke)
  --train     --variant {flag,delta,full} --dataset <ds>
  --eval      --shard i --nshards n [--anchor-subset]
  --anchor    gate vs s12_results.json + checkpoint identity check
  --merge     shards -> s21_results.json (+ paired tests, bias tables)
  --summary   print the main tables
  --figure    s21.png

State lives in s21_ckpt/ (eval_shard*.json, anchor.json, s21_<variant>_<ds>.pt);
root-level artifacts: run_s21_grud.py, s21_results.json, s21.png,
s21_notes.md, s21_*.log. Nothing pre-existing is modified.
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
import torch.nn.functional as F

import run_s5_missing as s5
import run_s6_sft as s6
import run_s8_masktoken as s8

L, H = s5.L, s5.H                       # 512, 96
PATCH, NPATCH = s8.PATCH, s8.NPATCH     # 16, 32
DS = ("ETTh1", "ETTm1", "weather")
DS_IDX = s6.DS_IDX
RATES = (0.1, 0.3, 0.5, 0.7)
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
PRIMARY_MECHS = ("block", "mnar_high", "mnar_extreme")
VARIANTS = ("flag", "delta", "full")
FEAT_NAMES = {"flag": (), "delta": ("delta",), "full": ("delta", "gap", "rate")}
CKPT_DIR = os.path.join(HERE, "s21_ckpt")
RESULTS = os.path.join(HERE, "s21_results.json")
S12_JSON = os.path.join(HERE, "s12_results.json")
S12_CKPT = {ds: os.path.join(HERE, "s12_ckpt", f"s12_mb_{ds}.pt") for ds in DS}
ANCHOR_TOL = {"zs": 0.02, "flag": 0.05}


def ckpt_path(variant, ds):
    return os.path.join(CKPT_DIR, f"s21_{variant}_{ds}.pt")


def shard_path(i):
    return os.path.join(CKPT_DIR, f"eval_shard{i}.json")


# ------------------------------------------------------- GRU-D features ----

def grud_feats(obs, names, patch_fn):
    """obs: [B, L] float observed indicator (1=observed). Returns patchified
    features [B, NPATCH, k*PATCH] (feature-major). Reads the MASK ONLY.
      delta: log1p(time since last observed; 0 at observed t; t+1 if none yet)
      gap:   log1p(distance to next observed; 0 at observed t; L-t if none)
      rate:  cumulative observed fraction up to and including t."""
    B, L_ = obs.shape
    idx = torch.arange(L_, device=obs.device, dtype=obs.dtype
                       ).unsqueeze(0).expand(B, L_)
    neg = torch.full_like(idx, -1.0)
    ob = obs > 0
    feats = []
    for n in names:
        if n == "delta":
            last = torch.cummax(torch.where(ob, idx, neg), dim=-1).values
            feats.append(torch.log1p(torch.where(last >= 0, idx - last,
                                                 idx + 1.0)))
        elif n == "gap":
            rob = torch.flip(ob, [-1])
            rlast = torch.cummax(torch.where(rob, idx, neg), dim=-1).values
            gap = torch.where(rlast >= 0, idx - rlast, idx + 1.0)
            feats.append(torch.log1p(torch.flip(gap, [-1])))
        elif n == "rate":
            feats.append(torch.cumsum(obs, dim=-1) / (idx + 1.0))
        else:
            raise ValueError(n)
    fs = torch.stack(feats, dim=1)                          # [B, k, L]
    k = len(names)
    pf = patch_fn(fs.reshape(B * k, L_))                    # [B*k, n_patch, P]
    np_, ps = pf.shape[1], pf.shape[2]      # n_patch varies in AR predict (L<=2048)
    return pf.reshape(B, k, np_, ps).permute(0, 2, 1, 3).reshape(B, np_, k * ps)


def encode_grud(model, context, feat, mask=None):
    """Line-for-line replica of ChronosBoltModelForForecasting.encode()
    (chronos_bolt.py l.277-329, == s8.encode_replica mask_mode='correct')
    plus the S21 feature injection: emb += W @ patchified(mask-features).
    W/b zero-init => bit-identical to the native encode at init."""
    mask = mask.to(context.dtype) if mask is not None else \
        torch.isnan(context).logical_not().to(context.dtype)          # l.280
    B = context.shape[0]
    if context.shape[-1] > model.chronos_config.context_length:       # l.283-285
        context = context[..., -model.chronos_config.context_length:]
        mask = mask[..., -model.chronos_config.context_length:]
    context, loc_scale = model.instance_norm(context)                 # l.288
    context = context.to(model.dtype)                                 # l.292
    mask = mask.to(model.dtype)                                       # l.293
    pc = model.patch(context)                                         # l.296
    pm = torch.nan_to_num(model.patch(mask), nan=0.0)                 # l.297
    pc = torch.where(pm > 0.0, pc, 0.0)                               # l.298
    pc_in = torch.cat([pc, pm], dim=-1)                               # l.300
    am = pm.sum(dim=-1) > 0                                           # l.303
    emb = model.input_patch_embedding(pc_in)                          # l.305
    # ---- S21: structured missingness features (mask-only), zero-init proj --
    if feat is not None:
        pf = grud_feats(mask, feat["names"], model.patch)
        emb = emb + F.linear(pf, feat["w"], feat["b"]).to(emb.dtype)
    reg_ids = torch.full((B, 1), model.config.reg_token_id,
                         device=context.device)
    emb = torch.cat([emb, model.shared(reg_ids)], dim=-2)             # l.307-315
    am = torch.cat([am.to(model.dtype),
                    torch.ones_like(reg_ids).to(model.dtype)], dim=-1)
    enc = model.encoder(attention_mask=am, inputs_embeds=emb)         # l.324-327
    return enc.last_hidden_state, loc_scale, emb, am


class EncodeGrud:
    """Context manager installing encode_grud as model.encode (same pattern
    as s8.EncodePatch; `model` must be the RAW ChronosBolt model -- the peft
    wrapper's forward reaches it via self.encode)."""

    def __init__(self, model, feat):
        self.model, self.feat, self.orig = model, feat, None

    def __enter__(self):
        self.orig = self.model.encode

        def encode_fn(context, mask=None):
            return encode_grud(self.model, context, self.feat, mask)

        self.model.encode = encode_fn
        return self

    def __exit__(self, *exc):
        self.model.encode = self.orig


def raw_model(pipe):
    """The underlying ChronosBoltModelForForecasting. pipe.model is either the
    raw model (train path; get_peft_model mutates it in-place -- note a raw
    PreTrainedModel's .base_model returns ITSELF, so attribute sniffing is
    unreliable) or a PeftModel wrapper (eval path: PeftModel -> LoraModel
    (.base_model) -> ChronosBolt (.model))."""
    from peft import PeftModel
    m = pipe.model
    if isinstance(m, PeftModel):
        return m.base_model.model
    return m


# ------------------------------------------------------------ training ----

def train(args):
    """S6/S12 train loop with the S21 feature projection added. For
    variant=flag the rng/torch consumption is identical to S6's/S12's mb run
    (same seed) -> checkpoints expected bit-identical to s12_mb_*; the anchor
    gate verifies. delta/full share the identical data+dropout streams
    (zero-init params consume no RNG)."""
    ds, variant, names = args.dataset, args.variant, FEAT_NAMES[args.variant]
    device = "cuda"
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

    feat = None
    if names:
        d = model.config.d_model
        feat = {"names": names,
                # torch.zeros consumes NO RNG -> streams identical to flag
                "w": torch.nn.Parameter(torch.zeros(d, len(names) * PATCH,
                                                    device=device)),
                "b": torch.nn.Parameter(torch.zeros(d, device=device))}
    params = trainable + ([feat["w"], feat["b"]] if feat is not None else [])
    print(f"variant={variant} feats={names or '-'} adapter=lora trainable "
          f"params: {sum(p.numel() for p in trainable):,}"
          + (f" + feat proj {feat['w'].numel() + feat['b'].numel():,} "
             f"(zero-init Linear {len(names) * PATCH}->{model.config.d_model})"
             if feat is not None else ""), flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / args.warmup))

    Xtr = s6.load_X(ds)[: int(0.7 * len(s6.load_X(ds)))]
    N, C = Xtr.shape
    t0 = time.time()
    run = {"loss": 0.0, "raw": 0.0, "n": 0}
    from contextlib import nullcontext
    cm = EncodeGrud(raw_model(pipe), feat) if feat is not None else nullcontext()
    with cm:
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
            with torch.enable_grad():
                loss, raw = s6.clipped_pinball(model, xb, yb, args.loss_clip)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run["loss"] += loss.item()
            run["raw"] += raw.item()
            run["n"] += 1
            if step % args.log_every == 0 or step == 1:
                k = run["n"]
                msg = (f"step {step}/{args.steps} loss={run['loss'] / k:.4f} "
                       f"raw={run['raw'] / k:.2f}")
                if feat is not None:
                    msg += f" |W|={feat['w'].norm().item():.4f}"
                print(f"{msg} ({time.time() - t0:.0f}s)", flush=True)
                run = {"loss": 0.0, "raw": 0.0, "n": 0}

    os.makedirs(args.ckpt_dir, exist_ok=True)
    name = args.ckpt_name or f"s21_{variant}_{ds}.pt"
    ckpt = os.path.join(args.ckpt_dir, name)
    torch.save({
        "adapter": "lora",
        "lora_config": s6.LORA_KW,
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters()
                      if p.requires_grad},
        "feat": (None if feat is None else
                 {"names": list(names),
                  "w": feat["w"].detach().cpu(),
                  "b": feat["b"].detach().cpu()}),
        "meta": {"variant": variant, "features": list(names), "dataset": ds,
                 "steps": args.steps, "batch": args.batch, "lr": args.lr,
                 "warmup": args.warmup, "loss_clip": args.loss_clip,
                 "seed": seed, "train_frac": 0.7, "pred_len_train": s6.PRED,
                 "aug": "mcar+block24 p~U(.05,.8) (S6 mb mix, same seed)",
                 "input_forms": "50% raw-NaN / 50% linear-fill; 20% clean series",
                 "feat_def": ("delta=log1p(t-since-last-obs); "
                              "gap=log1p(dist-to-next-obs); "
                              "rate=cum-obs-fraction; mask-only, "
                              "zero-init proj into patch embedding"),
                 "final_loss": None},
    }, ckpt)
    print(f"saved -> {ckpt} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------ model I/O ----

class ModelBank:
    """One ZS pipe + one LoRA-swappable SFT pipe per eval shard process."""

    def __init__(self, device):
        self.device = device
        self.zs = s8.load_bolt(device)
        self.sft = None
        self.cur = None
        self.feat = None

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
        """-> (pipe, feat-or-None)."""
        if model == "zs":
            return self.zs, None
        key = (model, ds)
        pipe = self.sft_pipe()
        if self.cur != key:
            sd = torch.load(ckpt_path(*key), map_location=self.device)
            pipe.model.load_state_dict(sd["trainable"], strict=False)
            pipe.model.eval()
            f = sd["feat"]
            self.feat = (None if f is None else
                         {"names": f["names"],
                          "w": f["w"].to(self.device),
                          "b": f["b"].to(self.device)})
            self.cur = key
        return pipe, self.feat


# ------------------------------------------------------------ evaluation ----

def eval_jobs():
    """Canonical job list (strided sharding). kind in {clean, grid}."""
    jobs = []
    for ds in DS:
        for model in ("zs",) + VARIANTS:
            jobs.append(("clean", ds, None, 0.0, model, "nan"))
    for ds in DS:
        for mech in MECHS:
            for rate in RATES:
                for model in VARIANTS:
                    jobs.append(("grid", ds, mech, rate, model, "nan"))
    return jobs


def is_anchor_job(job):
    kind, ds, mech, rate, model, fill = job
    if kind == "clean":
        return model in ("zs", "flag")
    return model == "flag" and mech in ("mcar", "block")


def job_key(job):
    kind, ds, mech, rate, model, fill = job
    if kind == "clean":
        return f"clean:{model}"
    return f"{mech}:{model}:{fill}:{rate}"


def score(pred, gt, S, C, nw, ns):
    """S5-convention windowed MSE + S19-style level/bias per window."""
    pred = pred.reshape(S, C, H).astype(np.float64)
    err = pred - gt
    mse_wc = (err ** 2).mean(axis=2)                       # [S, C]
    mu_p, mu_g = pred.mean(axis=2), gt.mean(axis=2)
    lvl2_wc = (mu_p - mu_g) ** 2
    bias_wc = mu_p - mu_g
    per = ns * C
    f = lambda v: v.reshape(nw, per).mean(axis=1).tolist()
    return {"mse": float(mse_wc.mean()),
            "mse_per_window": f(mse_wc),
            "lvl2_per_window": f(lvl2_wc),
            "bias_per_window": f(bias_wc),
            "bias": float(bias_wc.mean()),
            "lvl2_share": float(lvl2_wc.mean() / max(mse_wc.mean(), 1e-12))}


def predict_with(pipe, feat, x_in):
    if feat is None:
        return s8.predict_median(pipe, x_in)
    with EncodeGrud(raw_model(pipe), feat):
        return s8.predict_median(pipe, x_in)


def run_eval_job(bank, X, starts, ds, job):
    kind, _, mech, rate, model, fill = job
    t0 = time.time()
    pipe, feat = bank.get(model, ds)
    if kind == "clean":
        C = X.shape[1]
        ctxs, gts = [], []
        for st in starts:
            ctxs.append(X[st:st + L].T.copy())
            gts.append(X[st + L:st + L + H].T.copy())
        S = len(ctxs)
        ctx = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
        gt = np.stack(gts).astype(np.float64)
        pred = predict_with(pipe, feat, ctx)
        res = score(pred, gt, S, C, len(starts), 1)
        res["seconds"] = round(time.time() - t0, 1)
        return res
    ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, mech, rate,
                                                   n_seeds=1)
    x_in = ctx.copy()
    x_in[msk] = np.nan
    pred = predict_with(pipe, feat, x_in)
    res = score(pred, gt, S, C, nw, ns)
    res.update({"achieved_rate": float(msk.mean()),
                "n_windows": nw, "n_seeds": ns,
                "seconds": round(time.time() - t0, 1)})
    return res


def run_eval(args):
    jobs = eval_jobs()[args.shard::args.nshards]
    out = shard_path(args.shard)
    results = {}
    if os.path.exists(out):
        results = json.load(open(out))
        print(f"resume: loaded {out}", flush=True)
    results.setdefault("meta", {"track": "S21 GRU-D missingness-input eval",
                                "windows": args.windows, "seeds": 1,
                                "variants": VARIANTS, "fill": "nan"})
    save = lambda: s5.save_results(results, out)
    bank = ModelBank(args.device)
    loaded = {}
    n_done = 0
    for job in jobs:
        if args.anchor_subset and not is_anchor_job(job):
            continue
        kind, ds, mech, rate, model, fill = job
        sec = "clean" if kind == "clean" else "eval"
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
        print(f"  {sec:5s} {ds:8s} {key:28s} mse={res['mse']:12.4f} "
              f"({res['seconds']}s)", flush=True)
    print(f"EVAL SHARD {args.shard} DONE ({n_done} new jobs)", flush=True)


# ----------------------------------------------------------------- anchor ---

def run_anchor(args):
    """Gate vs s12_results.json + checkpoint identity vs s12_ckpt/s12_mb_*."""
    r12 = json.load(open(S12_JSON))
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
        for model, sec12 in (("zs", "clean:zs"), ("flag", "clean:mb")):
            ref = r12["clean"][ds][sec12]["mse"]
            mine = got.get("clean", {}).get(ds, {}).get(f"clean:{model}", {})
            cells.append((ds, "clean", model, mine.get("mse"), ref))
        for mech in ("mcar", "block"):
            for rate in RATES:
                ref = r12["eval"][ds][f"{mech}:mb:nan:{rate}"]["mse"]
                mine = got.get("eval", {}).get(ds, {}).get(
                    f"{mech}:flag:nan:{rate}", {})
                cells.append((ds, f"{mech}:nan:{rate}", "flag",
                              mine.get("mse"), ref))
    print("== S21 anchor gate vs s12_results.json ==", flush=True)
    out = {"tolerances": ANCHOR_TOL, "cells": {}, "ckpt_identity": {}}
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
        print(f"  {key:34s} mine={mine:12.4f} s12={ref:12.4f} "
              f"ratio={ratio:.4f} (tol {tol}) -> {'OK' if ok else 'FAIL'}",
              flush=True)
    # checkpoint identity (informational, expected bit-exact)
    for ds in DS:
        p21, p12 = ckpt_path("flag", ds), S12_CKPT[ds]
        if not (os.path.exists(p21) and os.path.exists(p12)):
            out["ckpt_identity"][ds] = {"status": "MISSING"}
            gate = False
            continue
        a = torch.load(p21, map_location="cpu")["trainable"]
        b = torch.load(p12, map_location="cpu")["trainable"]
        same_keys = set(a) == set(b)
        eq = same_keys and all(torch.equal(a[k], b[k]) for k in a)
        md = 0.0 if eq else max(float((a[k].float() - b[k].float()).abs().max())
                                for k in a if k in b)
        out["ckpt_identity"][ds] = {"bit_equal": bool(eq), "max_abs_diff": md}
        print(f"  ckpt {ds:8s} bit_equal={eq} max|diff|={md:.3e}", flush=True)
    out["gate"] = "PASS" if gate else "FAIL"
    out["n_missing"] = n_miss
    s5.save_results(out, os.path.join(CKPT_DIR, "anchor.json"))
    print(f"ANCHOR {out['gate']}", flush=True)
    if not gate:
        raise SystemExit(2)


# ----------------------------------------------------------------- merge ----

def paired_stats(out, mechs, variant):
    """Paired per-window (variant - flag) in zs-clean-relative units, pooled
    over the given mechanisms x rates x datasets (n = 150 x 4 x 3 x |mechs|)."""
    diffs, base = [], []
    for mech in mechs:
        for ds in DS:
            cz = out["clean"][ds]["clean:zs"]["mse_per_window"]
            for rate in RATES:
                a = out["eval"][ds][f"{mech}:flag:nan:{rate}"]["mse_per_window"]
                b = out["eval"][ds][f"{mech}:{variant}:nan:{rate}"]["mse_per_window"]
                diffs += [(bi - ai) / ci for ai, bi, ci in zip(a, b, cz)]
                base += [ai / ci for ai, ci in zip(a, cz)]
    d = np.asarray(diffs)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    half = 1.96 * sd / np.sqrt(n)
    return {"n": n, "mean_diff": mean, "t": float(mean / (sd / np.sqrt(n))),
            "ci95": [mean - half, mean + half],
            "win_rate": float((d < 0).mean()),
            "rel_improvement": float(-mean / np.mean(base)),
            "flag_mean_rel": float(np.mean(base))}


def build_tables(out):
    tabs = {"main": {}, "clean": {}, "bias": {}}
    for mech in MECHS:
        for rate in RATES:
            row = {}
            for v in VARIANTS:
                vals = []
                for ds in DS:
                    cell = out.get("eval", {}).get(ds, {}).get(
                        f"{mech}:{v}:nan:{rate}")
                    vals.append(None if cell is None else cell.get("rel_zs"))
                row[v] = {"avg": (float(np.mean(vals))
                                  if all(v is not None for v in vals) else None),
                          **{ds: v for ds, v in zip(DS, vals)}}
            tabs["main"][f"{mech}:{rate}"] = row
    for ds in DS:
        cz = out["clean"][ds].get("clean:zs", {}).get("mse")
        row = {"zs_mse": cz}
        for v in VARIANTS:
            cm = out["clean"][ds].get(f"clean:{v}", {})
            row[v] = {"mse": cm.get("mse"),
                      "rel_zs": (cm.get("mse") / cz if cz and cm.get("mse") else None)}
        tabs["clean"][ds] = row
    # S19 level-bias decomposition on the MNAR mechanisms (per ds x variant)
    for mech in ("mnar_high", "mnar_extreme"):
        for rate in RATES:
            row = {}
            for v in VARIANTS:
                ent = {}
                for ds in DS:
                    cell = out.get("eval", {}).get(ds, {}).get(
                        f"{mech}:{v}:nan:{rate}")
                    ent[ds] = (None if cell is None else
                               {"bias": cell["bias"],
                                "lvl2_share": cell["lvl2_share"],
                                "rel_zs": cell["rel_zs"]})
                row[v] = ent
            tabs["bias"][f"{mech}:{rate}"] = row
    return tabs


def run_merge(args):
    out = {"meta": {
        "track": "S21 GRU-D missingness-aware input features on bolt-base",
        "seed": s5.SEED, "L": L, "H": H, "patch": PATCH, "n_patch": NPATCH,
        "variants": {v: {"features": list(FEAT_NAMES[v])} for v in VARIANTS},
        "feature_def": ("delta_t=log1p(time since last observed); "
                        "gap_t=log1p(distance to next observed); "
                        "rate_t=cumulative observed fraction; computed from "
                        "the observed indicator ONLY; patchified like bolt's "
                        "mask channel; zero-init linear proj added to the "
                        "patch embedding (S5F miss_proj pattern)"),
        "train": ("LoRA r=16 a=32 do=0.05 q/k/v/o + zero-init feat proj; "
                  "3000 steps x 256; AdamW 1e-4 + 200 warmup; grad-clip 1.0; "
                  "clipped pinball 100; aug mcar+block24 p~U(.05,.8), 20% "
                  "clean, 50% nan/50% linear; mb seed -> identical data+"
                  "dropout streams across variants"),
        "eval": ("150 windows (s5.load_windows(ds,150,SEED)), 1 mask seed, "
                 "nan native fill, median; relMSE vs paired zs clean; "
                 "per-window mse/lvl2/bias stored"),
        "grid": f"{MECHS} x {RATES} x {DS} x {VARIANTS}",
        "primary_endpoint": ("paired per-window relMSE diff (variant - flag) "
                             "pooled over block+mnar_high+mnar_extreme "
                             "(n=5400); EFFECTIVE iff rel improvement > 2% "
                             "and 95% CI excludes 0"),
        "anchor_tolerances": ANCHOR_TOL,
    }}
    ap = os.path.join(CKPT_DIR, "anchor.json")
    if os.path.exists(ap):
        out["anchor"] = json.load(open(ap))
    for sec in ("clean", "eval"):
        out[sec] = {}
    for i in range(64):
        p = shard_path(i)
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        for sec in ("clean", "eval"):
            for ds, cells in r.get(sec, {}).items():
                out[sec].setdefault(ds, {}).update(cells)
    # relMSE columns
    for ds in DS:
        cz = out.get("clean", {}).get(ds, {}).get("clean:zs", {}).get("mse")
        for key, cell in out.get("eval", {}).get(ds, {}).items():
            v = key.split(":")[1]
            if cz:
                cell["rel_zs"] = cell["mse"] / cz
            cm = out["clean"].get(ds, {}).get(f"clean:{v}", {})
            if cm.get("mse"):
                cell["rel_own"] = cell["mse"] / cm["mse"]
    out["tables"] = build_tables(out)
    # paired tests: primary (block+mnar) + per-mechanism secondary
    out["paired"] = {}
    for v in ("delta", "full"):
        out["paired"][f"{v}_vs_flag:primary"] = paired_stats(out, PRIMARY_MECHS, v)
        for mech in MECHS:
            out["paired"][f"{v}_vs_flag:{mech}"] = paired_stats(out, (mech,), v)
    s5.save_results(out, RESULTS)
    print(f"merged -> {RESULTS}", flush=True)


def run_summary(args):
    out = json.load(open(RESULTS))
    print("== anchor gate:", out.get("anchor", {}).get("gate"), "==")
    for ds in DS:
        row = out["tables"]["clean"][ds]
        print(f"  clean {ds:8s} zs={row['zs_mse']:.4f} " + " ".join(
            f"{v}={row[v]['mse']:.4f} (rel {row[v]['rel_zs']:.4f})"
            for v in VARIANTS))
    for mech in MECHS:
        print(f"\n== grid relMSE (vs zs clean), {mech} == ")
        print(f"{'rate':5s} " + " ".join(f"{v:>10s}" for v in VARIANTS))
        for rate in RATES:
            row = out["tables"]["main"][f"{mech}:{rate}"]
            print(f"{rate:<5.1f} " + " ".join(
                f"{(row[v]['avg'] if row[v]['avg'] is not None else float('nan')):10.4f}"
                for v in VARIANTS))
        avg = {v: float(np.mean([out["tables"]["main"][f"{mech}:{r}"][v]["avg"]
                                 for r in RATES])) for v in VARIANTS}
        print(f"{'avg':5s} " + " ".join(f"{avg[v]:10.4f}" for v in VARIANTS))
    print("\n== paired tests (variant - flag, zs-clean-relative) ==")
    for k, s in out["paired"].items():
        print(f"  {k:28s} n={s['n']:5d} diff={s['mean_diff']:+.5f} "
              f"rel={100 * s['rel_improvement']:+.2f}% t={s['t']:+.2f} "
              f"ci95=[{s['ci95'][0]:+.5f},{s['ci95'][1]:+.5f}] "
              f"win={s['win_rate']:.3f}")
    print("\n== mnar level bias (signed mu_pred-mu_gt, original units) "
          "and lvl2 share ==")
    for mech in ("mnar_high", "mnar_extreme"):
        for rate in RATES:
            row = out["tables"]["bias"][f"{mech}:{rate}"]
            print(f"  {mech:13s} p={rate:.1f} " + " | ".join(
                f"{v}: " + ",".join(
                    f"{ds} b={row[v][ds]['bias']:+.3f} l={row[v][ds]['lvl2_share']:.2f}"
                    for ds in DS) for v in VARIANTS))


# ---------------------------------------------------------------- figure ----

def make_figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    vstyle = {"flag": ("SFT-mb (flag only)", "tab:green", "-", "d"),
              "delta": ("flag + δ_t", "tab:red", "-", "*"),
              "full": ("flag + δ_t + gap + rate", "tab:purple", "--", "v")}
    for ai, mech in enumerate(("mcar", "block")):
        ax = axes[0][ai]
        for v, (label, c, ls, mk) in vstyle.items():
            ys = [out["tables"]["main"][f"{mech}:{r}"][v]["avg"] for r in RATES]
            if any(vv is None for vv in ys):
                continue
            ax.plot(RATES, ys, color=c, ls=ls, marker=mk, ms=4, label=label)
        ax.axhline(1.0, color="grey", lw=0.6, ls=":")
        ax.set_title(f"{mech}: relMSE vs missing rate (dataset-avg)")
        ax.set_xlabel("nominal missing rate p")
        ax.set_ylabel("relMSE (vs paired ZS clean)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    # MNAR curves
    ax = axes[1][0]
    for mech, mls in (("mnar_high", "-"), ("mnar_extreme", ":")):
        for v, (label, c, ls, mk) in vstyle.items():
            ys = [out["tables"]["main"][f"{mech}:{r}"][v]["avg"] for r in RATES]
            if any(vv is None for vv in ys):
                continue
            ax.plot(RATES, ys, color=c, ls=mls, marker=mk, ms=4,
                    label=f"{mech} {label}")
    ax.axhline(1.0, color="grey", lw=0.6, ls=":")
    ax.set_title("MNAR mechanisms: relMSE vs rate (dataset-avg)")
    ax.set_xlabel("nominal missing rate p")
    ax.set_ylabel("relMSE (vs paired ZS clean)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    # paired diffs with CI
    ax = axes[1][1]
    keys = [f"{v}_vs_flag:{m}" for v in ("delta", "full")
            for m in ("primary",) + MECHS]
    labels = [k.replace("_vs_flag:", "\n") for k in keys]
    means = [out["paired"][k]["mean_diff"] for k in keys]
    cis = [out["paired"][k]["ci95"] for k in keys]
    cols = ["tab:red" if k.startswith("delta") else "tab:purple" for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, means, color=cols, alpha=0.8)
    ax.errorbar(x, means, yerr=[m - c[0] for m, c in zip(means, cis)],
                fmt="none", ecolor="k", capsize=3, lw=1)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Δ relMSE (variant − flag, paired)")
    ax.set_title("paired diffs with 95% CI (negative = better than flag)")
    ax.grid(alpha=0.3, axis="y")
    gate = out.get("anchor", {}).get("gate", "?")
    fig.suptitle(f"S21 GRU-D missingness features vs SFT-mb (bolt-base) — "
                 f"anchor gate {gate}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}", flush=True)


# ------------------------------------------------------------------ tiny ----

def _np_feats_ref(m, names):
    """Independent numpy reference for grud_feats (point level)."""
    L_ = len(m)
    out = {}
    if "delta" in names:
        d = np.zeros(L_)
        last = -1
        for t in range(L_):
            if m[t]:
                last = t
            d[t] = (t - last) if last >= 0 else (t + 1)
        out["delta"] = np.log1p(d)
    if "gap" in names:
        g = np.zeros(L_)
        nxt = -1
        for t in range(L_ - 1, -1, -1):
            if m[t]:
                nxt = t
            g[t] = (nxt - t) if nxt >= 0 else (L_ - t)
        out["gap"] = np.log1p(g)
    if "rate" in names:
        out["rate"] = np.cumsum(m) / np.arange(1, L_ + 1)
    return out


def run_tiny(args):
    device = args.device
    pipe = s8.load_bolt(device)
    model = pipe.model
    X, starts = s5.load_windows(s5.DATASETS["ETTh1"], 6, s5.SEED)
    ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, "block", 0.5,
                                                   n_seeds=1)
    x_nan = ctx.copy()
    x_nan[msk] = np.nan

    # 1) feature correctness vs independent numpy reference (+clean constants)
    rng = np.random.default_rng(0)
    m_row = rng.random(L) < 0.4                                # True = missing
    obs_t = torch.from_numpy((~m_row).astype(np.float32))[None].to(device)
    got = grud_feats(obs_t, ("delta", "gap", "rate"), model.patch)[0].cpu().numpy()
    ref = _np_feats_ref(~m_row, ("delta", "gap", "rate"))
    for fi, n in enumerate(("delta", "gap", "rate")):
        got_f = got[:, fi * PATCH:(fi + 1) * PATCH].reshape(-1)
        d = np.abs(got_f - ref[n]).max()
        assert d < 1e-5, f"feature {n} mismatch {d}"
        print(f"tiny: feature {n:5s} vs numpy ref max|diff| = {d:.2e}", flush=True)
    full_obs = grud_feats(torch.ones(1, L, device=device),
                          ("delta", "gap", "rate"), model.patch)[0].cpu().numpy()
    assert np.all(full_obs[:, :2 * PATCH] == 0.0) and \
        np.allclose(full_obs[:, 2 * PATCH:], 1.0), "clean constants wrong"
    print("tiny: fully-observed series -> delta=0, gap=0, rate=1  OK", flush=True)

    # 2) leakage assertion: permuting observed VALUES (mask fixed) must not
    #    change the features
    x1 = x_nan[0].copy()
    obs_idx = np.flatnonzero(~np.isnan(x1))
    x2 = x1.copy()
    x2[obs_idx] = x1[rng.permutation(obs_idx)]
    f1 = grud_feats(torch.isnan(torch.from_numpy(x1[None]).to(device))
                    .logical_not().float(), ("delta", "gap", "rate"), model.patch)
    f2 = grud_feats(torch.isnan(torch.from_numpy(x2[None]).to(device))
                    .logical_not().float(), ("delta", "gap", "rate"), model.patch)
    assert torch.equal(f1, f2), "LEAK: features changed under value permutation"
    print("tiny: leak assertion (value permutation -> identical features) OK",
          flush=True)

    # 3) zero-init feature proj == native encode, bit-exact (both variants)
    p_ref = s8.predict_median(pipe, x_nan, batch=512)
    for v in ("delta", "full"):
        k = len(FEAT_NAMES[v])
        feat0 = {"names": FEAT_NAMES[v],
                 "w": torch.zeros(model.config.d_model, k * PATCH, device=device),
                 "b": torch.zeros(model.config.d_model, device=device)}
        with EncodeGrud(model, feat0):
            p_g = s8.predict_median(pipe, x_nan, batch=512)
        d = np.abs(p_ref - p_g).max()
        print(f"tiny: zero-init {v:5s} vs native max|diff| = {d:.3e} (want 0)",
              flush=True)
        assert d == 0.0, "zero-init projection is not bit-identical"

    # 4) mask stream of build_contexts(n_seeds=1) == s5.make_mask ms=0 (S6)
    m_ref = np.stack([s5.make_mask("block", 0.5, wi, 0, C,
                                   x=X[s:s + L].T.copy())
                      for wi, s in enumerate(starts)])
    assert (m_ref == msk.reshape(nw, ns, C, L)[:, 0]).all(), "mask stream mismatch"
    print("tiny: build_contexts(n_seeds=1) mask stream == S6 eval stream",
          flush=True)

    # 5) 3-step train per variant; step-1 loss must equal S6's 37.6535 for
    #    ALL variants (zero-init => identical forward at step 1)
    args.steps, args.log_every = 3, 1
    args.dataset = "ETTh1"
    for v in VARIANTS:
        args.variant, args.ckpt_name = v, f"tiny_{v}_ETTh1.pt"
        print(f"tiny: training {v} 3 steps (step-1 loss must be 37.6535)",
              flush=True)
        train(args)
        sd = torch.load(os.path.join(args.ckpt_dir, args.ckpt_name),
                        map_location="cpu")
        wn = 0.0 if sd["feat"] is None else sd["feat"]["w"].norm().item()
        print(f"tiny: {v} trained feat |W| = {wn:.5f}", flush=True)

    # 6) eval path smoke incl. tiny ckpts
    bank = ModelBank(device)
    bank.sft_pipe()
    res = run_eval_job(bank, X, starts, "ETTh1",
                       ("grid", "ETTh1", "mcar", 0.3, "zs", "nan"))
    print(f"tiny: eval mcar:zs:nan:0.3 mse={res['mse']:.4f}", flush=True)
    for v in VARIANTS:
        sd = torch.load(ckpt_path(v, "ETTh1").replace(
            f"s21_{v}_ETTh1.pt", f"tiny_{v}_ETTh1.pt"), map_location=device)
        bank.sft.model.load_state_dict(sd["trainable"], strict=False)
        bank.sft.model.eval()
        f = sd["feat"]
        bank.feat = (None if f is None else
                     {"names": f["names"], "w": f["w"].to(device),
                      "b": f["b"].to(device)})
        bank.cur = (v, "ETTh1")
        res = run_eval_job(bank, X, starts, "ETTh1",
                           ("grid", "ETTh1", "block", 0.3, v, "nan"))
        print(f"tiny: eval block:{v}:nan:0.3 mse={res['mse']:.4f}", flush=True)
    for v in VARIANTS:
        os.remove(os.path.join(args.ckpt_dir, f"tiny_{v}_ETTh1.pt"))
    print("TINY OK", flush=True)


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--anchor-subset", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--variant", choices=list(VARIANTS), default="flag")
    ap.add_argument("--dataset", choices=list(DS), default="ETTh1")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--loss-clip", type=float, default=100.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--windows", type=int, default=150)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--ckpt-name", default=None)
    ap.add_argument("--fig-out", default=os.path.join(HERE, "s21.png"))
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
