#!/usr/bin/env python
"""S15: reconstruction-routed decoding -- open the decoder cross-attention
mask on fully-missing patch positions and test whether the mid-encoder
reconstruction (S11: R2 0.5-0.8; S13: attention-causal, but architecturally
unread by the forecast head) can be turned into forecast gains.

Background. bolt's encode() hard-masks fully-missing patches out of the
encoder KEY set but keeps them as QUERIES; the mid-encoder rebuilds their
true values (S11/S13). bolt's decode() cross-attends with
encoder_attention_mask = the patch mask (chronos_bolt.py l.429-435), which
EXCLUDES fully-missing positions -- S13 proved the rebuilt signal has
exactly zero causal channel to the forecast (bit-identical relMSE under
full-miss-query knockout). S15 removes that architectural barrier: the
encoder is left untouched (the reconstruction machinery stays native), and
the decoder's cross-attention mask is replaced by all-ones, so the forecast
head can read the rebuilt content at gap positions ("reconstruction-routed
decoding").

Implementation (runtime monkey-patch, no library edits -- same style as
S8/S13): UnmaskDecode sets an instance-attribute `decode` on the raw
ChronosBoltModelForForecasting that replicates chronos_bolt.py decode()
line-for-line except `encoder_attention_mask=torch.ones_like(attention_mask)`.
Every position of the patched context (32 patches + [REG]) is a real
position, so all-ones is exactly "observe the gap positions too". Clean
input => native mask is already all-ones => unmask is bit-identical
(registered sanity). Active during BOTH training (sft:unmask) and eval.

Pre-registered outcomes (frozen 2026-08-13 before any run; no tuning toward A):
  Outcome A: sft:unmask significantly beats sft:mb (masked SFT, S12) on the
     mcar/block grid: paired per-window test, mean relative gain > 2% with
     the 95% CI of the paired diff excluding 0 -> the rebuilt signal was
     real forecast headroom and routing it to the head pays; new method
     chapter "reconstruction-routed decoding".
  Outcome B: sft:unmask ~= sft:mb -> the rebuilt content is redundant for
     the head (observed positions already carry equivalent information);
     the attribution chapter closes with ironclad evidence.

Conditions (grid = {mcar, block} x p {0.1,0.3,0.5,0.7} x {ETTh1, ETTm1,
weather}; S12 protocol exactly: 150 test windows s5.SEED, 1 mask seed,
median quantile, H=96, relMSE vs paired zs clean):
  zs:masked  -- native bolt-nan (control / anchor; must reproduce
                s12_results.json zs:nan bit-exactly).
  zs:unmask  -- zero-shot + unmasked cross-attention (OOD operation: the
                decoder never read these positions during pretraining;
                degradation is informative too).
  sft:unmask -- unmask + SFT, S12/S6 SFT-mb config and budget EXACTLY:
                LoRA r=16 alpha=32 dropout 0.05 on q/k/v/o, 3000 steps x
                batch 256, AdamW lr 1e-4 + 200-step linear warmup,
                grad-clip 1.0, clipped pinball clip=100, aug mcar+block24
                p~U(0.05,0.8) with 20% clean, input 50% raw-NaN / 50%
                linear-fill, first-70% train split, seed = S12-mb seed
                (identical augmentation/dropout stream; the ONLY difference
                vs S12's mb run is the cross-attention mask).
  References lifted from s12_results.json (harness is bit-identical, proven
  by the anchor gate): mb:nan (masked SFT) and tok:nan (S8 mask token).
  mb_anchor: SFT-mb retrained through THIS harness (user's anchor gate:
  eval cells vs s12 mb:nan within +-5%; checkpoint param diff vs
  s12_ckpt/s12_mb_*.pt reported, expected exactly 0).

Negative controls: {mnar_high, mnar_extreme} x p=0.7 x all 4 conditions --
no gain expected (censored information is not reconstructable; S5/S8/S12).

Gates (in order):
  smoke : (1) native decode cross-attention puts EXACTLY 0 mass on
          fully-missing keys, unmasked decode puts >0 (weights still sum
          to 1, no NaN); (2) unmasked predictions on masked input differ
          from native (>0) and contain no NaN; (3) clean input: unmasked ==
          native BITWISE; (4) UnmaskDecode restores cleanly.
  anchor: zs:masked cells vs s12_results.json zs:nan (+-2%, expect exact)
          + clean:zs_masked vs s12 clean:zs; after training, mb_anchor eval
          cells vs s12 mb:nan (+-5%) + ckpt diff report.

Subcommands:
  --smoke
  --train --variant {mb_anchor,unmask} --dataset <ds>
  --eval --shard i --nshards n [--anchor-subset]
  --anchor   (zs gate; + mb gate once mb_anchor cells exist)
  --merge    shards -> s15_results.json (+ paired tests + tables)
  --summary
  --figure   s15.png

Intermediate state in s15_ckpt/ (eval_shard*.json, s15_<variant>_<ds>.pt,
anchor.json). Root-level artifacts: run_s15_unmask.py, s15_results.json,
s15.png, s15_notes.md, s15_*.log.
"""
import argparse
import contextlib
import json
import os
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import torch

import run_s5_missing as s5
import run_s6_sft as s6
import run_s8_masktoken as s8

L, H = s5.L, s5.H                       # 512, 96
PATCH, NPATCH = s8.PATCH, s8.NPATCH     # 16, 32
DS = ("ETTh1", "ETTm1", "weather")
DS_IDX = s6.DS_IDX
RATES = (0.1, 0.3, 0.5, 0.7)
NEG_MECHS = ("mnar_high", "mnar_extreme")
CONDS = ("zs_masked", "zs_unmask", "mb_anchor", "unmask")
UNMASKED = {"zs_unmask", "unmask"}      # conditions that open the mask
SFT_CONDS = ("mb_anchor", "unmask")
CKPT_DIR = os.path.join(HERE, "s15_ckpt")
RESULTS = os.path.join(HERE, "s15_results.json")
S12_JSON = os.path.join(HERE, "s12_results.json")
S12_CKPT = os.path.join(HERE, "s12_ckpt")
ANCHOR_TOL = {"zs": 0.02, "mb": 0.05}   # pre-registered; both expect ~exact
ANCHOR_RATES = (0.3, 0.7)               # user's 6+ representative cells


def ckpt_path(variant, ds):
    return os.path.join(CKPT_DIR, f"s15_{variant}_{ds}.pt")


def shard_path(i):
    return os.path.join(CKPT_DIR, f"eval_shard{i}.json")


# ------------------------------------------------------- unmask machinery ----

def unmask_decode(self, input_embeds, attention_mask, hidden_states,
                  output_attentions=False):
    """Line-for-line replica of ChronosBoltModelForForecasting.decode()
    (chronos_bolt.py l.401-437) with ONE change: the cross-attention
    encoder_attention_mask is all-ones instead of the patch mask, so the
    forecast head reads the encoder hidden states at fully-missing patch
    positions (where S11/S13 showed the truth is rebuilt)."""
    batch_size = input_embeds.shape[0]
    decoder_input_ids = torch.full(
        (batch_size, 1),
        self.config.decoder_start_token_id,
        device=input_embeds.device,
    )
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=torch.ones_like(attention_mask),   # <-- S15
        output_attentions=output_attentions,
        return_dict=True,
    )
    return decoder_outputs.last_hidden_state


def raw_model(pipe):
    """The underlying ChronosBoltModelForForecasting (S12's unwrap: LoRA
    modules are injected in-place; PeftModel -> LoraModel -> raw)."""
    m = pipe.model
    if hasattr(m, "peft_config"):
        return m.base_model.model
    return m


class UnmaskDecode:
    """Context manager: install unmask_decode as an instance attribute on
    the raw ChronosBoltModelForForecasting (shadows the class method;
    deleting it restores native behavior -- S13's encode-patch pattern).
    Encoder untouched. Applies to the predict path AND the training forward
    (both go through model.forward -> self.decode)."""

    def __init__(self, model):
        assert hasattr(model, "decode") and hasattr(model, "encoder"), \
            "pass the raw ChronosBoltModelForForecasting (use raw_model())"
        self.raw = model

    def __enter__(self):
        assert "decode" not in self.raw.__dict__, "already patched"
        self.raw.decode = types.MethodType(unmask_decode, self.raw)
        return self

    def __exit__(self, *exc):
        del self.raw.decode


# --------------------------------------------------------------- training ----

def train(args):
    """S12's train() (= S6 mb loop) with the recon aux loss removed and the
    S15 decode-unmask patch added for variant=unmask. variant=mb_anchor
    consumes the exact same torch/numpy RNG stream as S12's mb run (same
    seed, same op sequence) -> checkpoint expected bit-identical to
    s12_ckpt/s12_mb_<ds>.pt; the anchor gate verifies."""
    ds, variant = args.dataset, args.variant
    device = "cuda"
    seed = s5.SEED + 1000 * s6.VAR_IDX["mb"] + DS_IDX[ds]   # == S12 mb seed
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
    model = pipe.model
    assert not model.instance_norm.use_arcsinh, "loss replication assumes no arcsinh"
    model.train()  # bolt has no dropout/BN; needed only for lora_dropout

    raw0 = model                               # raw ChronosBolt model object
    model = s6.wrap_lora(model)                # peft wrapper around raw0
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"variant={variant} adapter=lora trainable params: "
          f"{sum(p.numel() for p in trainable):,} "
          f"(decode unmask {'ON' if variant == 'unmask' else 'OFF'})",
          flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / args.warmup))

    Xtr = s6.load_X(ds)[: int(0.7 * len(s6.load_X(ds)))]
    N, C = Xtr.shape
    ctxman = UnmaskDecode(raw0) if variant == "unmask" \
        else contextlib.nullcontext()
    t0 = time.time()
    run = {"loss": 0.0, "raw": 0.0, "n": 0}
    with ctxman:
        for step in range(1, args.steps + 1):
            starts = rng.integers(0, N - s5.L - s6.PRED, size=args.batch)
            chans = rng.integers(0, C, size=args.batch)
            ctx = np.stack([Xtr[s:s + s5.L, c] for s, c in zip(starts, chans)])
            fut = np.stack([Xtr[s + s5.L:s + s5.L + s6.PRED, c]
                            for s, c in zip(starts, chans)])
            m = np.stack([s6.aug_mask(rng, ctx[b], "mb")
                          for b in range(args.batch)])
            forms = rng.random(args.batch) < 0.5
            xb_np = np.stack([s6.corrupt(ctx[b:b + 1], m[b:b + 1],
                                         "nan" if forms[b] else "linear")[0]
                              for b in range(args.batch)])
            xb = torch.from_numpy(xb_np).to(device)
            yb = torch.from_numpy(fut).to(device)
            with torch.enable_grad():
                loss, raw_loss = s6.clipped_pinball(model, xb, yb,
                                                    args.loss_clip)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run["loss"] += loss.item()
            run["raw"] += raw_loss.item()
            run["n"] += 1
            if step % args.log_every == 0 or step == 1:
                k = run["n"]
                print(f"step {step}/{args.steps} loss={run['loss'] / k:.4f} "
                      f"raw={run['raw'] / k:.2f} ({time.time() - t0:.0f}s)",
                      flush=True)
                run = {"loss": 0.0, "raw": 0.0, "n": 0}

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = ckpt_path(variant, ds)
    torch.save({
        "adapter": "lora",
        "lora_config": s6.LORA_KW,
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters()
                      if p.requires_grad},
        "meta": {"variant": variant, "dataset": ds,
                 "decode_unmask": variant == "unmask",
                 "steps": args.steps, "batch": args.batch, "lr": args.lr,
                 "warmup": args.warmup, "loss_clip": args.loss_clip,
                 "seed": seed, "train_frac": 0.7, "pred_len_train": s6.PRED,
                 "aug": "mcar+block24 p~U(.05,.8) (S6 mb mix, S12-mb seed)",
                 "input_forms": "50% raw-NaN / 50% linear-fill; 20% clean",
                 "decode": ("cross-attention encoder_attention_mask = all-ones"
                            if variant == "unmask" else "native patch mask")},
    }, ckpt)
    print(f"saved -> {ckpt} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------ evaluation ----

class ModelBank:
    """One ZS pipe + one LoRA-swappable SFT pipe per eval shard process
    (S12 pattern)."""

    def __init__(self, device):
        self.device = device
        self.zs = s8.load_bolt(device)
        self.sft = None
        self.cur = None

    def sft_pipe(self):
        if self.sft is None:
            from chronos import BaseChronosPipeline
            pipe = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-bolt-base", device_map=self.device,
                torch_dtype=torch.float32)
            pipe.model = s6.wrap_lora(pipe.model)
            self.sft = pipe
        return self.sft

    def get(self, cond, ds):
        """-> (pipe, unmask_flag)."""
        if cond in ("zs_masked", "zs_unmask"):
            return self.zs, cond in UNMASKED
        pipe = self.sft_pipe()
        if self.cur != (cond, ds):
            sd = torch.load(ckpt_path(cond, ds), map_location=self.device)
            pipe.model.load_state_dict(sd["trainable"], strict=False)
            pipe.model.eval()
            self.cur = (cond, ds)
        return pipe, cond in UNMASKED


def eval_jobs():
    """Canonical job list (strided sharding). kind in {clean, grid, neg}.
    All cells use nan fill (the S11/S13 decoding setting)."""
    jobs = []
    for ds in DS:
        for cond in CONDS:
            jobs.append(("clean", ds, None, 0.0, cond))
    for ds in DS:
        for mech in ("mcar", "block"):
            for rate in RATES:
                for cond in CONDS:
                    jobs.append(("grid", ds, mech, rate, cond))
    for ds in DS:
        for mech in NEG_MECHS:
            for cond in CONDS:
                jobs.append(("neg", ds, mech, 0.7, cond))
    return jobs


def is_anchor_job(job):
    """User's anchor subset: zs:masked clean + mcar/block x {0.3,0.7}."""
    kind, ds, mech, rate, cond = job
    if cond != "zs_masked":
        return False
    if kind == "clean":
        return True
    return kind == "grid" and rate in ANCHOR_RATES


def job_key(job):
    kind, ds, mech, rate, cond = job
    if kind == "clean":
        return f"clean:{cond}"
    return f"{mech}:{cond}:{rate}"


def run_eval_job(bank, X, starts, ds, job):
    kind, _, mech, rate, cond = job
    t0 = time.time()
    pipe, um = bank.get(cond, ds)
    ctxman = UnmaskDecode(raw_model(pipe)) if um else contextlib.nullcontext()
    with ctxman:
        if kind == "clean":
            res = s8.clean_baseline(pipe, X, starts)
            res["seconds"] = round(time.time() - t0, 1)
            return res
        ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, mech, rate,
                                                       n_seeds=1)
        x_in = ctx.copy()
        x_in[msk] = np.nan
        pred = s8.predict_median(pipe, x_in)
    mse, mse_pw = s8.mse_windowed(pred, gt, S, C, nw)
    return {"mse": mse, "mse_per_window": mse_pw,
            "achieved_rate": float(msk.mean()),
            "n_windows": nw, "n_seeds": ns,
            "seconds": round(time.time() - t0, 1)}


def run_eval(args):
    jobs = eval_jobs()[args.shard::args.nshards]
    os.makedirs(CKPT_DIR, exist_ok=True)
    out = shard_path(args.shard)
    results = {}
    if os.path.exists(out):
        results = json.load(open(out))
        print(f"resume: loaded {out}", flush=True)
    results.setdefault("meta", {"track": "S15 reconstruction-routed decoding",
                                "windows": args.windows, "seeds": 1,
                                "conds": CONDS})
    save = lambda: s5.save_results(results, out)
    bank = ModelBank(args.device)
    loaded = {}
    n_done = 0
    for job in jobs:
        if args.anchor_subset and not is_anchor_job(job):
            continue
        kind, ds, mech, rate, cond = job
        if cond in SFT_CONDS and not os.path.exists(ckpt_path(cond, ds)):
            print(f"  skip {job_key(job)} {ds}: ckpt missing", flush=True)
            continue
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


# ------------------------------------------------------------------ smoke ----

def run_smoke(args):
    device = args.device
    pipe = s8.load_bolt(device)
    model = pipe.model
    X, starts = s5.load_windows(s5.DATASETS["ETTh1"], 5, s5.SEED)
    ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, "block", 0.3,
                                                   n_seeds=1)
    x_nan = ctx.copy()
    x_nan[msk] = np.nan
    xb = torch.from_numpy(x_nan[:64]).to(device)

    # 1) cross-attention mass on fully-missing keys: native 0, unmasked >0
    with torch.no_grad():
        hidden, loc_scale, emb, am = model.encode(xb)
        B = xb.shape[0]
        dec_ids = torch.full((B, 1), model.config.decoder_start_token_id,
                             device=device)
        dec_nat = model.decoder(input_ids=dec_ids, encoder_hidden_states=hidden,
                                encoder_attention_mask=am,
                                output_attentions=True, return_dict=True)
        dec_unm = model.decoder(input_ids=dec_ids, encoder_hidden_states=hidden,
                                encoder_attention_mask=torch.ones_like(am),
                                output_attentions=True, return_dict=True)
    gap = am[:, :NPATCH] == 0                          # [B, 32] bool
    ngap = int(gap.sum())
    assert ngap > 100, "smoke batch should contain many fully-missing patches"
    ca_nat = dec_nat.cross_attentions
    ca_unm = dec_unm.cross_attentions
    assert ca_nat is not None and len(ca_nat) > 0, "no cross_attentions returned"
    print(f"smoke: decoder layers with cross-attn = {len(ca_nat)}, "
          f"shape {tuple(ca_nat[0].shape)}", flush=True)
    for l, (an, au) in enumerate(zip(ca_nat, ca_unm)):
        # [B, heads, 1, 33] -> mass on gap keys
        gm = gap[:, None, None, :].expand_as(an[..., :NPATCH])
        mn = an[..., :NPATCH][gm].abs().max().item()
        mu_min = au[..., :NPATCH][gm].min().item()
        mu_max = au[..., :NPATCH][gm].max().item()
        s_nat = an.sum(dim=-1).mean().item()
        s_unm = au.sum(dim=-1).mean().item()
        assert mn == 0.0, f"layer {l}: native cross-attn keeps {mn} on gap keys"
        assert mu_min > 0.0, f"layer {l}: unmasked cross-attn has 0 on gap keys"
        assert abs(s_unm - 1.0) < 1e-4 and abs(s_nat - 1.0) < 1e-4
        assert not torch.isnan(au).any() and not torch.isnan(an).any()
        print(f"smoke: layer {l:2d} gap-key mass native=0.0 unmasked "
              f"[{mu_min:.2e}, {mu_max:.3f}] (n_gap={ngap})", flush=True)
    # REG key must keep identical mass semantics: unmasked sums to 1 over 33
    d_last = (dec_nat.last_hidden_state - dec_unm.last_hidden_state)
    print(f"smoke: decoder last_hidden native-vs-unmasked max|diff| = "
          f"{d_last.abs().max().item():.4f} (want >0: OOD read of gap keys)",
          flush=True)
    assert d_last.abs().max().item() > 0

    # 2) predict-level: unmasked differs on masked input, no NaN
    p_native = s8.predict_median(pipe, x_nan, batch=512)
    with UnmaskDecode(model):
        p_unmask = s8.predict_median(pipe, x_nan, batch=512)
    assert not np.isnan(p_unmask).any(), "NaN in unmasked predictions"
    md = np.abs(p_native - p_unmask).max()
    print(f"smoke: predict max|unmask-native| on masked input = {md:.4f} "
          f"(want >0)", flush=True)
    assert md > 0

    # 3) clean sanity: unmasked == native BITWISE (no gap positions -> the
    #    all-ones mask equals the native mask exactly)
    C_ = X.shape[1]
    ctxs = np.stack([X[s:s + L].T.copy() for s in starts[:4]])
    x_clean = ctxs.reshape(-1, L).astype(np.float32)
    p_clean_native = s8.predict_median(pipe, x_clean, batch=512)
    with UnmaskDecode(model):
        p_clean_unmask = s8.predict_median(pipe, x_clean, batch=512)
    dc = np.abs(p_clean_native - p_clean_unmask).max()
    print(f"smoke: clean predict max|unmask-native| = {dc:.2e} "
          f"(want exactly 0)", flush=True)
    assert dc == 0.0, "clean sanity failed: unmask must be a no-op on clean input"

    # 4) restore check: after the context manager the class method is back
    assert "decode" not in model.__dict__, "UnmaskDecode did not restore"
    p_after = s8.predict_median(pipe, x_nan, batch=512)
    dr = np.abs(p_after - p_native).max()
    print(f"smoke: post-restore predict max|diff| vs native = {dr:.2e} "
          f"(want 0)", flush=True)
    assert dr == 0.0
    print("SMOKE OK", flush=True)


# ----------------------------------------------------------------- anchor ----

def collect_shards():
    got = {"clean": {}, "eval": {}, "neg": {}}
    for i in range(64):
        p = shard_path(i)
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        for sec in ("clean", "eval", "neg"):
            for ds, cells in r.get(sec, {}).items():
                got[sec].setdefault(ds, {}).update(cells)
    return got


def run_anchor(args):
    """zs part: my zs_masked cells vs s12_results.json zs:nan (+-2%, expect
    bit-exact). mb part (once mb_anchor cells exist): my rerun vs s12 mb:nan
    (+-5%) + checkpoint param diff vs s12_ckpt/s12_mb_*.pt."""
    r12 = json.load(open(S12_JSON))
    got = collect_shards()
    out = {"tolerances": ANCHOR_TOL, "cells": {}}
    gate, n_cells = True, 0
    print("== S15 anchor gate vs s12_results.json ==", flush=True)
    for ds in DS:
        todo = [("clean", None, 0.0)]
        for mech in ("mcar", "block"):
            for rate in RATES:
                todo.append(("eval", mech, rate))
        for kind, mech, rate in todo:
            if kind == "clean":
                mine = got["clean"].get(ds, {}).get("clean:zs_masked", {})
                ref = r12["clean"][ds]["clean:zs"]["mse"]
                ck = f"{ds}:clean:zs"
            else:
                mine = got["eval"].get(ds, {}).get(f"{mech}:zs_masked:{rate}", {})
                ref = r12["eval"][ds][f"{mech}:zs:nan:{rate}"]["mse"]
                ck = f"{ds}:{mech}:zs:{rate}"
            if not mine:
                status = "MISSING(subset-pending)" if (
                    kind == "eval" and rate not in ANCHOR_RATES) else "MISSING"
                if status == "MISSING":
                    gate = False
                out["cells"][ck] = {"ref": ref, "status": status}
                print(f"  {ck:28s} {status}", flush=True)
                continue
            ratio = mine["mse"] / ref
            ok = abs(ratio - 1) <= ANCHOR_TOL["zs"]
            gate &= ok
            n_cells += 1
            out["cells"][ck] = {"mine": mine["mse"], "ref": ref,
                                "ratio": ratio, "ok": ok}
            print(f"  {ck:28s} mine={mine['mse']:12.4f} s12={ref:12.4f} "
                  f"ratio={ratio:.6f} -> {'OK' if ok else 'FAIL'}", flush=True)
    out["zs_gate"] = "PASS" if gate else "FAIL"
    out["zs_cells"] = n_cells

    # ---- mb part (post-training) ----
    mb_cells = 0
    mb_gate = True
    mb_present = all(os.path.exists(ckpt_path("mb_anchor", ds)) for ds in DS)
    if mb_present:
        print("== mb_anchor rerun vs s12 mb:nan (+-5%) ==", flush=True)
        for ds in DS:
            for mech in ("mcar", "block"):
                for rate in RATES:
                    mine = got["eval"].get(ds, {}).get(
                        f"{mech}:mb_anchor:{rate}", {})
                    if not mine:
                        continue
                    ref = r12["eval"][ds][f"{mech}:mb:nan:{rate}"]["mse"]
                    ratio = mine["mse"] / ref
                    ok = abs(ratio - 1) <= ANCHOR_TOL["mb"]
                    mb_gate &= ok
                    mb_cells += 1
                    out["cells"][f"{ds}:{mech}:mb:{rate}"] = {
                        "mine": mine["mse"], "ref": ref, "ratio": ratio,
                        "ok": ok}
                    print(f"  {ds}:{mech}:{rate}:mb mine={mine['mse']:12.4f} "
                          f"s12={ref:12.4f} ratio={ratio:.6f} -> "
                          f"{'OK' if ok else 'FAIL'}", flush=True)
            mine = got["clean"].get(ds, {}).get("clean:mb_anchor", {})
            if mine:
                ref = r12["clean"][ds]["clean:mb"]["mse"]
                ratio = mine["mse"] / ref
                ok = abs(ratio - 1) <= ANCHOR_TOL["mb"]
                mb_gate &= ok
                mb_cells += 1
                print(f"  {ds}:clean:mb mine={mine['mse']:12.4f} "
                      f"s12={ref:12.4f} ratio={ratio:.6f} -> "
                      f"{'OK' if ok else 'FAIL'}", flush=True)
        # checkpoint param diff (expected exactly 0: same seed, same stream)
        diffs = {}
        for ds in DS:
            a = torch.load(ckpt_path("mb_anchor", ds), map_location="cpu")
            b = torch.load(os.path.join(S12_CKPT, f"s12_mb_{ds}.pt"),
                           map_location="cpu")
            assert set(a["trainable"]) == set(b["trainable"])
            d = max(float((a["trainable"][k] - b["trainable"][k]).abs().max())
                    for k in a["trainable"])
            diffs[ds] = d
            print(f"  ckpt param max|diff| s15_mb_anchor vs s12_mb {ds}: {d:.3e}",
                  flush=True)
        out["mb_ckpt_maxdiff"] = diffs
        out["mb_cells"] = mb_cells
        out["mb_gate"] = "PASS" if (mb_gate and mb_cells > 0) else "FAIL"
        out["gate"] = "PASS" if (gate and out["mb_gate"] == "PASS") else "FAIL"
    else:
        out["gate"] = out["zs_gate"]
    os.makedirs(CKPT_DIR, exist_ok=True)
    s5.save_results(out, os.path.join(CKPT_DIR, "anchor.json"))
    print(f"ANCHOR {out['gate']}", flush=True)
    if out["gate"] != "PASS":
        raise SystemExit(2)


# ------------------------------------------------------------------ merge ----

def paired_stats(unm_pw, mb_pw):
    """Paired per-window diff (unmask - mb): mean, t, 95% CI half-width,
    win-rate. n large -> normal approx."""
    d = np.asarray(unm_pw, np.float64) - np.asarray(mb_pw, np.float64)
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    t = m / (sd / np.sqrt(n)) if sd > 0 else 0.0
    return {"n": n, "mean_diff": m, "t": t, "ci95": 1.96 * sd / np.sqrt(n),
            "win_rate": float((d < 0).mean())}


def run_merge(args):
    r12 = json.load(open(S12_JSON))
    got = collect_shards()
    out = {"meta": {
        "track": "S15 reconstruction-routed decoding (decoder cross-attention "
                 "unmasked on fully-missing patch positions), chronos-bolt-base",
        "seed": s5.SEED, "L": L, "H": H, "patch": PATCH, "n_patch": NPATCH,
        "conds": {"zs_masked": "native bolt-nan (anchor vs S12)",
                  "zs_unmask": "ZS + cross-attn all-ones mask (OOD)",
                  "mb_anchor": "SFT-mb retrained through this harness (anchor)",
                  "unmask": "SFT-mb config/budget + cross-attn unmasked "
                            "(train+eval)"},
        "train": ("LoRA r=16 a=32 do=0.05 q/k/v/o; 3000 steps x 256; AdamW "
                  "1e-4 + 200 warmup; grad-clip 1.0; clipped pinball 100; "
                  "aug mcar+block24 p~U(.05,.8), 20% clean, 50% nan/50% "
                  "linear; S12-mb seed (identical data/dropout stream)"),
        "eval": ("150 windows (s5.load_windows(ds,150,SEED)), 1 mask seed, "
                 "median; relMSE vs paired zs clean; own-clean also stored"),
        "neg_controls": f"{NEG_MECHS} p=0.7",
        "pre_registered_outcomes": {
            "A": "sft:unmask > sft:mb: paired mean relative gain > 2% with "
                 "95% CI of the per-window diff excluding 0",
            "B": "sft:unmask ~= sft:mb: rebuilt content redundant for the "
                 "head; attribution chapter closes"},
        "references_from_s12": ["mb:nan", "tok:nan", "zs:nan", "clean:zs",
                                "clean:mb"],
    }}
    ap = os.path.join(CKPT_DIR, "anchor.json")
    if os.path.exists(ap):
        out["anchor"] = json.load(open(ap))
    out.update(got)

    # ---- clean sanity: zs_unmask must equal zs_masked BITWISE per window ----
    sanity = {}
    for ds in DS:
        a = out["clean"].get(ds, {}).get("clean:zs_masked", {})
        b = out["clean"].get(ds, {}).get("clean:zs_unmask", {})
        if a and b:
            dmax = max(abs(x - y) for x, y in zip(a["mse_per_window"],
                                                  b["mse_per_window"]))
            sanity[ds] = {"max_abs_window_diff": dmax, "bitwise": dmax == 0.0}
            assert dmax == 0.0, f"clean sanity FAILED on {ds}: {dmax}"
    out["clean_sanity"] = sanity

    # ---- relMSE columns (vs MY zs_masked clean; S12's is bit-identical) ----
    for ds in DS:
        cz = out["clean"].get(ds, {}).get("clean:zs_masked", {}).get("mse")
        for sec in ("eval", "neg"):
            for key, cell in out.get(sec, {}).get(ds, {}).items():
                cond = key.split(":")[1] if sec != "clean" else None
                if cz:
                    cell["rel_zs"] = cell["mse"] / cz
                if cond in SFT_CONDS:
                    cm = out["clean"].get(ds, {}).get(f"clean:{cond}", {})
                    if cm.get("mse"):
                        cell["rel_own"] = cell["mse"] / cm["mse"]

    # ---- main tables: my 4 conds + S12 references mb:nan / tok:nan ----
    tabs = {"main": {}, "neg": {}, "paired": {}}
    disp = {"zs_masked": "zs:masked", "zs_unmask": "zs:unmask",
            "mb_anchor": "mb:rerun", "unmask": "sft:unmask"}
    for mech in ("mcar", "block"):
        for rate in RATES:
            row = {}
            for cond in CONDS:
                vals = [out.get("eval", {}).get(ds, {}).get(
                    f"{mech}:{cond}:{rate}", {}).get("rel_zs") for ds in DS]
                row[disp[cond]] = {
                    "avg": (float(np.mean(vals))
                            if all(v is not None for v in vals) else None),
                    **{ds: v for ds, v in zip(DS, vals)}}
            for ref in ("mb:nan", "tok:nan"):
                e12 = r12["tables"]["main"][f"{mech}:{rate}"][ref]
                row[ref] = e12
            tabs["main"][f"{mech}:{rate}"] = row
    for mech in NEG_MECHS:
        row = {}
        for cond in CONDS:
            rels, owns = [], []
            for ds in DS:
                cell = out.get("neg", {}).get(ds, {}).get(f"{mech}:{cond}:0.7", {})
                rels.append(cell.get("rel_zs"))
                owns.append(cell.get("rel_own"))
            row[disp[cond]] = {
                "avg": (float(np.mean(rels))
                        if all(v is not None for v in rels) else None),
                "avg_own": (float(np.mean([o for o in owns if o is not None]))
                            if any(o is not None for o in owns) else None),
                **{ds: v for ds, v in zip(DS, rels)}}
        e12 = r12["tables"]["neg"][mech]["mb:nan"]
        row["mb:nan"] = e12
        tabs["neg"][mech] = row

    # ---- paired tests: sft:unmask vs S12 mb (per-window), zs pair secondary --
    for mech in ("mcar", "block"):
        all_u, all_m = [], []
        per_ds = {}
        for ds in DS:
            du, dm = [], []
            for rate in RATES:
                u = out["eval"].get(ds, {}).get(f"{mech}:unmask:{rate}", {})
                m12 = r12["eval"][ds].get(f"{mech}:mb:nan:{rate}", {})
                if u and m12:
                    du += u["mse_per_window"]
                    dm += m12["mse_per_window"]
                    all_u += u["mse_per_window"]
                    all_m += m12["mse_per_window"]
            if du:
                st = paired_stats(du, dm)
                st["mean_mb"] = float(np.mean(dm))
                st["rel_gain"] = -st["mean_diff"] / st["mean_mb"]
                per_ds[ds] = st
        st = paired_stats(all_u, all_m)
        st["mean_mb"] = float(np.mean(all_m))
        st["rel_gain"] = -st["mean_diff"] / st["mean_mb"]
        tabs["paired"][mech] = {"all": st, "per_ds": per_ds}
    # pooled mcar+block (the pre-registered primary endpoint)
    pu, pm = [], []
    for mech in ("mcar", "block"):
        for ds in DS:
            for rate in RATES:
                u = out["eval"].get(ds, {}).get(f"{mech}:unmask:{rate}", {})
                m12 = r12["eval"][ds].get(f"{mech}:mb:nan:{rate}", {})
                if u and m12:
                    pu += u["mse_per_window"]
                    pm += m12["mse_per_window"]
    st = paired_stats(pu, pm)
    st["mean_mb"] = float(np.mean(pm))
    st["rel_gain"] = -st["mean_diff"] / st["mean_mb"]
    tabs["paired"]["pooled"] = {"all": st}
    # negative controls paired: sft:unmask vs S12 mb
    for mech in NEG_MECHS:
        all_u, all_m = [], []
        per_ds = {}
        for ds in DS:
            u = out["neg"].get(ds, {}).get(f"{mech}:unmask:0.7", {})
            m12 = r12["neg"][ds].get(f"{mech}:mb:nan:0.7", {})
            if u and m12:
                all_u += u["mse_per_window"]
                all_m += m12["mse_per_window"]
                st = paired_stats(u["mse_per_window"], m12["mse_per_window"])
                st["rel_gain"] = -st["mean_diff"] / float(
                    np.mean(m12["mse_per_window"]))
                per_ds[ds] = st
        if all_u:
            st = paired_stats(all_u, all_m)
            st["mean_mb"] = float(np.mean(all_m))
            st["rel_gain"] = -st["mean_diff"] / st["mean_mb"]
            tabs["paired"][mech] = {"all": st, "per_ds": per_ds}
    # zs pair (secondary): zs:unmask vs zs:masked
    for mech in ("mcar", "block"):
        au, am_ = [], []
        for ds in DS:
            for rate in RATES:
                u = out["eval"].get(ds, {}).get(f"{mech}:zs_unmask:{rate}", {})
                m = out["eval"].get(ds, {}).get(f"{mech}:zs_masked:{rate}", {})
                if u and m:
                    au += u["mse_per_window"]
                    am_ += m["mse_per_window"]
        if au:
            st = paired_stats(au, am_)
            st["mean_masked"] = float(np.mean(am_))
            st["rel_gain"] = -st["mean_diff"] / st["mean_masked"]
            tabs["paired"][f"zs_{mech}"] = {"all": st}
    out["tables"] = tabs
    s5.save_results(out, RESULTS)
    print(f"merged -> {RESULTS}", flush=True)
    print(f"clean sanity (bitwise zs_unmask==zs_masked): "
          f"{json.dumps(out['clean_sanity'])}", flush=True)


def run_summary(args):
    out = json.load(open(RESULTS))
    print("== anchor:", json.dumps({k: v for k, v in out.get("anchor", {}).items()
                                   if k.endswith("gate") or k == "mb_ckpt_maxdiff"}))
    print("== clean sanity:", json.dumps(out.get("clean_sanity")))
    cols = ["zs:masked", "zs:unmask", "mb:nan", "sft:unmask", "tok:nan"]
    for mech in ("mcar", "block"):
        print(f"\n== main grid relMSE (vs zs clean), {mech} ==")
        print(f"{'rate':5s} " + " ".join(f"{c:>11s}" for c in cols))
        for rate in RATES:
            row = out["tables"]["main"][f"{mech}:{rate}"]
            print(f"{rate:<5.1f} " + " ".join(
                f"{(row[c]['avg'] if row[c]['avg'] is not None else float('nan')):11.4f}"
                for c in cols))
        avg = {}
        for c in cols:
            vals = [out["tables"]["main"][f"{mech}:{r}"][c]["avg"] for r in RATES]
            avg[c] = float(np.mean(vals)) if all(v is not None for v in vals) else None
        print(f"{'avg':5s} " + " ".join(
            f"{(avg[c] if avg[c] is not None else float('nan')):11.4f}" for c in cols))
    print("\n== per-dataset rate-avg relMSE (mcar/block avg) ==")
    for ds in DS:
        line = f"  {ds:8s} "
        for c in cols:
            vals = [out["tables"]["main"][f"{mech}:{r}"][c][ds]
                    for mech in ("mcar", "block") for r in RATES]
            vals = [v for v in vals if v is not None]
            line += f"{c}={np.mean(vals):.4f} " if len(vals) == 8 else f"{c}=n/a "
        print(line)
    print("\n== paired per-window test: sft:unmask vs sft:mb (S12) ==")
    for mech in ("mcar", "block"):
        st = out["tables"]["paired"].get(mech, {}).get("all")
        if st:
            print(f"  {mech:6s} n={st['n']} diff={st['mean_diff']:+.5f} "
                  f"(mb mean {st['mean_mb']:.4f}) rel_gain={st['rel_gain'] * 100:+.2f}% "
                  f"t={st['t']:+.2f} ci95=±{st['ci95']:.5f} win={st['win_rate']:.3f}")
            for ds, s in out["tables"]["paired"][mech].get("per_ds", {}).items():
                print(f"    {ds:8s} n={s['n']} rel_gain={s['rel_gain'] * 100:+.2f}% "
                      f"t={s['t']:+.2f} win={s['win_rate']:.3f}")
    print("\n== zs pair: zs:unmask vs zs:masked ==")
    for mech in ("mcar", "block"):
        st = out["tables"]["paired"].get(f"zs_{mech}", {}).get("all")
        if st:
            print(f"  {mech:6s} rel_gain={st['rel_gain'] * 100:+.2f}% "
                  f"t={st['t']:+.2f} win={st['win_rate']:.3f}")
    print("\n== negative controls p=0.7 (relMSE vs zs clean) ==")
    for mech in NEG_MECHS:
        row = out["tables"]["neg"][mech]
        for c in ("zs:masked", "zs:unmask", "mb:nan", "mb:rerun", "sft:unmask"):
            e = row.get(c)
            if e and e.get("avg") is not None:
                print(f"  {mech:14s} {c:10s} avg={e['avg']:.4f} "
                      f"per-ds={[round(e[ds], 4) if e.get(ds) is not None else None for ds in DS]}")


# ----------------------------------------------------------------- figure ----

def make_figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [("zs:masked", "ZS masked (native)", "tab:blue", "-", "o"),
            ("zs:unmask", "ZS unmask (OOD)", "tab:cyan", "-", "s"),
            ("mb:nan", "SFT-mb (S12)", "tab:green", "-", "d"),
            ("sft:unmask", "SFT-unmask (S15)", "tab:red", "-", "*"),
            ("tok:nan", "[MASK] token (S8)", "tab:orange", "--", "^")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
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
    # paired diffs per ds (relative gain %, normalized across datasets)
    ax = axes[1][0]
    labels, means, cis, ts = [], [], [], []
    for mech in ("mcar", "block"):
        pk = out["tables"]["paired"].get(mech, {})
        st = pk.get("all")
        if st:
            labels.append(f"{mech}\nall")
            means.append(st["rel_gain"] * 100)
            cis.append(st["ci95"] / st["mean_mb"] * 100)
            ts.append(f"t={-st['t']:+.1f}")
        for ds in DS:
            s = pk.get("per_ds", {}).get(ds)
            if s:
                labels.append(f"{mech}\n{ds}")
                means.append(s["rel_gain"] * 100)
                cis.append(s["ci95"] / s["mean_mb"] * 100)
                ts.append(f"t={-s['t']:+.1f}")
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=cis, color=["k" if "all" in l else "tab:grey"
                                      for l in labels], alpha=0.8, capsize=3)
    for xi, (m, c, t) in enumerate(zip(means, cis, ts)):
        ax.text(xi, m + (c if m >= 0 else -c) + (0.15 if m >= 0 else -0.35),
                t, ha="center", fontsize=7)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("paired relMSE gain of sft:unmask vs mb (%)")
    ax.set_title("paired test: sft:unmask vs sft:mb (95% CI; >0 = unmask better)\n"
                 "t>0 favors unmask")
    ax.grid(alpha=0.3, axis="y")
    # negative controls
    ax = axes[1][1]
    nshow = [("zs:masked", "ZS masked"), ("zs:unmask", "ZS unmask"),
             ("mb:nan", "SFT-mb"), ("sft:unmask", "SFT-unmask")]
    x = np.arange(len(NEG_MECHS))
    w = 0.18
    for mi, (mth, label) in enumerate(nshow):
        ys = [out["tables"]["neg"][mech][mth]["avg"] for mech in NEG_MECHS]
        ys = [np.nan if v is None else v for v in ys]
        ax.bar(x + (mi - 1.5) * w, ys, w, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\np=0.7" for m in NEG_MECHS], fontsize=8)
    ax.set_ylabel("relMSE (vs paired ZS clean)")
    ax.set_title("negative controls: MNAR (no gain expected)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)
    gate = out.get("anchor", {}).get("gate", "?")
    fig.suptitle(f"S15 reconstruction-routed decoding (chronos-bolt-base) — "
                 f"anchor gate {gate}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}", flush=True)


# ------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--anchor-subset", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--variant", choices=("mb_anchor", "unmask"),
                    default="unmask")
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
    ap.add_argument("--fig-out", default=os.path.join(HERE, "s15.png"))
    args = ap.parse_args()

    torch.manual_seed(s5.SEED)
    np.random.seed(s5.SEED)

    if args.smoke:
        run_smoke(args)
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
