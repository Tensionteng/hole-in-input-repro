#!/usr/bin/env python
"""S8: causal validation & generalization of the missingness signal in
chronos-bolt-base (see s5_fix_notes.md / s7_attrib_notes.md for anchors).

S7 showed the binary mask channel bolt concatenates into its patch-embedding
input (chronos_bolt.py l.300) is what lets attention avoid corrupted patches
(enrichment 0.03-0.50 under the native nan path vs ~1.0 for zero/linear fill).
S8 asks four questions:

  Exp1 -- causal ablation of the mask CHANNEL (inference only). Native nan
     path with four manipulations of the concatenated mask channel ONLY
     (value mean-fill l.298 and the attention mask l.303 always follow the
     TRUE mask): (a) correct, (b) zero (channel forced to 0 = "everything
     flagged missing"), (c) random (50% of bits flipped), (d) invert.
     Grid: {mcar, block, mnar_high} x p in {0.3, 0.7} x {ETTh1, ETTm1, weather}.
     Outputs MSE (full 2-block predict) + attention enrichment/entropy
     (first-block encode, S7 Layer-2 metric definitions). Question: does the
     channel itself causally matter, and does the model believe a lying mask?
  Exp2 -- learnable [MASK] token. One d_model=768 vector, zero-init, replaces
     the patch embedding of FULLY-missing patches (which then become
     attendable; partially-missing patches keep the native mean-fill + mask
     concat). Backbone frozen, only the vector trained: augmentation
     mcar+block(24), p~U(0.05,0.8), train split, clipped pinball loss
     (clip=100, run_s6_sft.clipped_pinball), AdamW lr 1e-3, 1000 steps,
     batch 256. One token per dataset.
  Exp3 -- generalization grid. Tokens from Exp2 evaluated on
     {train ds x test ds} x {mcar, block, mnar_high, mnar_extreme} x
     p in {0.1..0.7}; controls: native nan, linear fill, and the S5
     miss_proj-only adapter (s5_fix_bolt_adapter_mponly.pt). relMSE vs the
     paired native-clean baseline (S5 convention).
  Exp4 -- attention audit of the trained token: S7 enrichment/entropy on
     {mcar, block, mnar_high} x p in {0.3, 0.7} with the matched token
     installed; additionally tracks attention mass on the fully-missing
     (token) patch positions themselves. Positive = token positions are
     discounted into the native-mask band (enrichment 0.03-0.50).

Windows / mask seeds identical to run_s5_missing.py (300 windows, 2 mask seeds
for mcar/block, 1 for rank-deterministic mnar, L=512, H=96, median quantile).
Hard anchors (dataset-avg relMSE, p=0.7, must hold within +-5% before the main
runs): mcar nan 1.131 / linear 1.072; block nan 1.085 / linear 1.360;
mnar_high nan 3.393 / linear 2.174.

Sharding: every grid subcommand takes --shard/--nshards and writes
s8_results_<phase>_shard<i>.json incrementally (tmp+replace after every job);
--merge combines everything into s8_results.json.
"""
import argparse
import json
import os
import time
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import pandas as pd
import torch

import run_s5_missing as s5
import run_s5_fix as s5f
import run_s6_sft as s6

L, H = s5.L, s5.H                      # 512, 96
PATCH = 16
NPATCH = L // PATCH                    # 32
REG = NPATCH                           # REG token position (appended last)
NLAYERS = 12
PRED = 64                              # bolt native prediction length (training)
DS = ("ETTh1", "ETTm1", "weather")
DS_IDX = {d: i for i, d in enumerate(DS)}
MECHS1 = ("mcar", "block", "mnar_high")
P1 = (0.3, 0.7)
MODES = ("correct", "zero", "random", "invert")
MECHS3 = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES3 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
METHODS3 = ("nan", "linear", "mponly",
            "tok_ETTh1", "tok_ETTm1", "tok_weather")
MPONLY_CKPT = os.path.join(HERE, "s5_fix_bolt_adapter_mponly.pt")
TOK_CKPT = {ds: os.path.join(HERE, f"s8_masktoken_{ds}.pt") for ds in DS}
TRAIN_JSON = {ds: os.path.join(HERE, f"s8_train_{ds}.json") for ds in DS}

# S5 hard anchors: dataset-avg relMSE at p=0.7 (verified against
# s5_missing_results.json / s5_fix_results.json on 2026-08-12)
ANCHORS = {("mcar", "nan"): 1.131, ("mcar", "linear"): 1.072,
           ("block", "nan"): 1.085, ("block", "linear"): 1.360,
           ("mnar_high", "nan"): 3.393, ("mnar_high", "linear"): 2.174}


# ------------------------------------------------------------- model I/O ----

def load_bolt(device):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
    pipe.model.eval()  # config has dropout_rate=0.1; everything here is deterministic
    return pipe


def encode_replica(model, xb, mask_mode="correct", mask_token=None,
                   flip_gen=None, need_attn=False):
    """Line-for-line replica of ChronosBoltModelForForecasting.encode()
    (chronos_bolt.py l.277-329) with two S8 knobs:

      mask_mode  -- manipulates ONLY the concatenated mask channel (l.300).
                    The value mean-fill (l.298) and the attention mask (l.303)
                    always follow the TRUE mask, so the four modes isolate the
                    channel's own causal contribution.
      mask_token -- optional [d_model] tensor that REPLACES the patch
                    embedding of fully-missing patches; those positions become
                    attendable (Exp2/3/4). Partially-missing patches are
                    untouched (native mean-fill + mask concat).

    Returns (encoder_outputs, loc_scale, input_embeds, attention_mask)."""
    mask = torch.isnan(xb).logical_not().to(xb.dtype)                       # l.280
    B = xb.shape[0]
    if xb.shape[-1] > model.chronos_config.context_length:                  # l.283-285
        xb = xb[..., -model.chronos_config.context_length:]
        mask = mask[..., -model.chronos_config.context_length:]
    xc, loc_scale = model.instance_norm(xb)                                 # l.288
    xc = xc.to(model.dtype)                                                 # l.292
    mask = mask.to(model.dtype)                                             # l.293
    pc = model.patch(xc)                                                    # l.296
    pm = torch.nan_to_num(model.patch(mask), nan=0.0)                       # l.297
    pc = torch.where(pm > 0.0, pc, 0.0)                                     # l.298
    # ---- S8: the concatenated mask channel, manipulated ----
    if mask_mode == "correct":
        pm_chan = pm
    elif mask_mode == "zero":
        pm_chan = torch.zeros_like(pm)
    elif mask_mode == "invert":
        pm_chan = 1.0 - pm
    elif mask_mode == "random":
        flip = torch.rand(pm.shape, generator=flip_gen).to(pm.device) < 0.5
        pm_chan = torch.where(flip, 1.0 - pm, pm)
    else:
        raise ValueError(mask_mode)
    pc_in = torch.cat([pc, pm_chan], dim=-1)                                # l.300
    am = pm.sum(dim=-1) > 0                                                 # l.303 (TRUE mask)
    emb = model.input_patch_embedding(pc_in)                                # l.305
    # ---- S8: learned [MASK] token for fully-missing patches ----
    if mask_token is not None:
        full_miss = pm.sum(dim=-1) == 0                                     # [B, n_patch]
        if bool(full_miss.any()):
            emb = torch.where(full_miss.unsqueeze(-1),
                              mask_token.to(emb.dtype).view(1, 1, -1), emb)
            am = am | full_miss                                             # token positions attend
    reg_ids = torch.full((B, 1), model.config.reg_token_id, device=xb.device)
    emb = torch.cat([emb, model.shared(reg_ids)], dim=-2)                   # l.307-315
    am = torch.cat([am.to(model.dtype),
                    torch.ones_like(reg_ids).to(model.dtype)], dim=-1)      # l.316-321
    enc = model.encoder(attention_mask=am, inputs_embeds=emb,
                        output_attentions=need_attn)                        # l.324-327
    return enc, loc_scale, emb, am


class EncodePatch:
    """Context manager installing encode_replica as model.encode (the predict
    path calls self.encode in BOTH autoregressive blocks, so the manipulation
    applies everywhere). mode='random' draws flips from a generator seeded
    per (flip_seed, call_index) -> deterministic end to end."""

    def __init__(self, model, mask_mode="correct", mask_token=None, flip_seed=0):
        self.model, self.mask_mode, self.mask_token = model, mask_mode, mask_token
        self.flip_seed, self.calls, self.orig = flip_seed, 0, None

    def __enter__(self):
        self.orig = self.model.encode

        def encode_fn(context, mask=None):
            gen = None
            if self.mask_mode == "random":
                gen = torch.Generator()
                gen.manual_seed(self.flip_seed + self.calls)
            self.calls += 1
            enc, loc_scale, emb, am = encode_replica(
                self.model, context, self.mask_mode, self.mask_token, gen)
            return enc.last_hidden_state, loc_scale, emb, am

        self.model.encode = encode_fn
        return self

    def __exit__(self, *exc):
        self.model.encode = self.orig


@torch.no_grad()
def predict_median(pipe, ctx, batch=1024):
    """ctx: [n, L] float32 (may contain NaN) -> [n, H] median forecast."""
    outs = []
    for i in range(0, len(ctx), batch):
        xb = torch.from_numpy(ctx[i:i + batch]).to(pipe.model.device)
        q = pipe.predict(xb, prediction_length=H)  # [b, 9, H]
        outs.append(q[:, 4, :].float().cpu().numpy())
    return np.concatenate(outs)


# ------------------------------------------------------------ data/masks ----

def load_X(ds):
    df = pd.read_csv(s5.DATASETS[ds])
    return df.drop(columns=["date"]).to_numpy(np.float32)


def build_contexts(X, starts, mech, rate, n_seeds=2):
    """Identical windows/masks as run_s5_missing.run_config. Returns flat
    clean contexts [S*C, L], masks [S*C, L] bool, gt [S, C, H], and meta."""
    C = X.shape[1]
    det = mech in ("mnar_high", "mnar_extreme")
    ctxs, gts, masks = [], [], []
    for wi, st in enumerate(starts):
        xc = X[st:st + L].T.copy()
        y = X[st + L:st + L + H].T.copy()
        for ms in ((0,) if det else range(n_seeds)):
            ctxs.append(xc)
            gts.append(y)
            masks.append(s5.make_mask(mech, rate, wi, ms, C, x=xc))
    S = len(ctxs)
    ns = 1 if det else n_seeds
    ctx = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
    msk = np.stack(masks).reshape(S * C, L)
    gt = np.stack(gts).astype(np.float64)
    return ctx, msk, gt, S, C, len(starts), ns


def fill_linear(ctx, msk):
    """[N, L] row-wise linear fill (identical semantics to s5.fill_context)."""
    out = ctx.copy()
    t = np.arange(L)
    for i in range(len(out)):
        valid = np.flatnonzero(~msk[i])
        if len(valid) == 0:
            out[i] = 0.0
        else:
            out[i] = np.interp(t, valid, ctx[i, valid])
    return out


def mse_windowed(pred, gt, S, C, nw):
    """pred [S*C, H] -> overall mse + per-window means (S5 conventions)."""
    pred = pred.reshape(S, C, H).astype(np.float64)
    mse = ((pred - gt) ** 2).mean(axis=(1, 2))
    return float(mse.mean()), mse.reshape(nw, -1).mean(axis=1).tolist()


def clean_baseline(pipe, X, starts):
    """Native clean MSE per dataset (relMSE denominator for everything)."""
    C = X.shape[1]
    ctxs, gts = [], []
    for st in starts:
        ctxs.append(X[st:st + L].T.copy())
        gts.append(X[st + L:st + L + H].T.copy())
    S = len(ctxs)
    ctx = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
    gt = np.stack(gts).astype(np.float64)
    pred = predict_median(pipe, ctx)
    mse, mse_pw = mse_windowed(pred, gt, S, C, len(starts))
    return {"mse": mse, "mse_per_window": mse_pw}


# --------------------------------------------------- attention metrics -----

class AttnAcc:
    """S7 Layer-2 attention metrics (run_s7_attrib.DriftAttnAcc, attention
    subset, identical mass/entropy formulas): per-layer head/query-averaged
    entropy, mass on corrupted patches (>=1 missing point), REG, patch 0.
    Two enrichment denominators are stored side by side:
      *_am -- over the true attention mask (attendable keys; the causally
              interpretable 'uniform baseline');
      *_s7 -- S7's published convention (fully-OBSERVED patches + REG as the
              key set; reproduces the s7_attrib_notes.md nan band 0.03-0.50
              exactly, verified against s7_attrib_results.json).
    Optionally also tracks the fully-missing 'token' positions (Exp4).
    Per-window means stored for downstream stats."""

    def __init__(self, n_series, track_token=False):
        self.ent = np.zeros((n_series, NLAYERS), np.float64)
        self.mc = np.zeros((n_series, NLAYERS), np.float64)
        self.mreg = np.zeros((n_series, NLAYERS), np.float64)
        self.mfirst = np.zeros((n_series, NLAYERS), np.float64)
        self.frac_corr_am = np.zeros(n_series, np.float64)
        self.frac_corr_s7 = np.zeros(n_series, np.float64)
        self.track_token = track_token
        if track_token:
            self.mtok = np.zeros((n_series, NLAYERS), np.float64)
            self.frac_tok_am = np.zeros(n_series, np.float64)
            self.frac_tok_s7 = np.zeros(n_series, np.float64)

    def add(self, enc, corr, tokpos, valid_am, valid_s7, sl):
        """enc: encoder output with attentions; corr/tokpos: [B, NPATCH] bool;
        valid_am/valid_s7: [B, NPATCH+1] float key-set masks (incl. REG)."""
        B = corr.shape[0]
        dev = enc.attentions[0].device
        corr_t = torch.from_numpy(corr).to(dev)
        va = torch.from_numpy(valid_am).to(dev)
        v7 = torch.from_numpy(valid_s7).to(dev)
        ck = torch.cat([corr_t, torch.zeros(B, 1, dtype=torch.bool, device=dev)], dim=1)
        self.frac_corr_am[sl] = (ck.float().sum(1) / va.sum(1).clamp_min(1.0)).cpu().numpy()
        self.frac_corr_s7[sl] = (ck.float().sum(1) / v7.sum(1).clamp_min(1.0)).cpu().numpy()
        if self.track_token:
            tok_t = torch.from_numpy(tokpos).to(dev)
            tk = torch.cat([tok_t, torch.zeros(B, 1, dtype=torch.bool, device=dev)], dim=1)
            self.frac_tok_am[sl] = (tk.float().sum(1) / va.sum(1).clamp_min(1.0)).cpu().numpy()
            self.frac_tok_s7[sl] = (tk.float().sum(1) / v7.sum(1).clamp_min(1.0)).cpu().numpy()
        for l, a in enumerate(enc.attentions):              # [B,12,33,33]
            a = a.float()
            ent = -(a * (a + 1e-12).log()).sum(-1)          # [B,12,33]
            self.ent[sl, l] = ent.mean(dim=(1, 2)).cpu().numpy()
            mc = (a * ck[:, None, None, :].float()).sum(-1)  # [B,12,33]
            self.mc[sl, l] = mc.mean(dim=(1, 2)).cpu().numpy()
            self.mreg[sl, l] = a[..., REG].mean(dim=(1, 2)).cpu().numpy()
            self.mfirst[sl, l] = a[..., 0].mean(dim=(1, 2)).cpu().numpy()
            if self.track_token:
                mt = (a * tk[:, None, None, :].float()).sum(-1)
                self.mtok[sl, l] = mt.mean(dim=(1, 2)).cpu().numpy()

    def finalize(self, nw, ns, C):
        per = ns * C
        g = lambda v: v.reshape(nw, per, NLAYERS).mean(axis=1).tolist()
        f_am, f_s7 = self.frac_corr_am.mean(), self.frac_corr_s7.mean()
        res = {"ent_pw": g(self.ent), "mc_pw": g(self.mc),
               "mreg_pw": g(self.mreg), "mfirst_pw": g(self.mfirst),
               "ent": self.ent.mean(0).tolist(), "mc": self.mc.mean(0).tolist(),
               "mreg": self.mreg.mean(0).tolist(), "mfirst": self.mfirst.mean(0).tolist(),
               "frac_corr": float(f_am), "frac_corr_am": float(f_am),
               "frac_corr_s7": float(f_s7),
               "enrich": float(self.mc.mean() / max(f_am, 1e-9)),
               "enrich_am": float(self.mc.mean() / max(f_am, 1e-9)),
               "enrich_s7": float(self.mc.mean() / max(f_s7, 1e-9))}
        if self.track_token:
            t_am, t_s7 = self.frac_tok_am.mean(), self.frac_tok_s7.mean()
            res["mtok_pw"] = g(self.mtok)
            res["mtok"] = self.mtok.mean(0).tolist()
            res["frac_tok_am"] = float(t_am)
            res["frac_tok_s7"] = float(t_s7)
            res["enrich_tok_am"] = float(self.mtok.mean() / max(t_am, 1e-9))
            res["enrich_tok_s7"] = float(self.mtok.mean() / max(t_s7, 1e-9))
        return res


@torch.no_grad()
def attn_pass(model, x_in, msk, nw, ns, C, mask_mode="correct",
              mask_token=None, flip_seed=0, track_token=False, batch=256):
    """First-block encode_replica with attentions over all series. x_in:
    [S*C, L] with NaN at missing; msk: [S*C, L] bool."""
    n = len(x_in)
    acc = AttnAcc(n, track_token=track_token)
    corr = msk.reshape(n, NPATCH, PATCH).any(axis=2)                    # >=1 missing
    tokpos = msk.reshape(n, NPATCH, PATCH).all(axis=2)                  # fully missing
    if mask_token is not None:
        valid_patch = np.ones((n, NPATCH), np.float32)                  # all attend
    else:
        valid_patch = (~tokpos).astype(np.float32)                      # native: full-miss hard-masked
    valid_am = np.concatenate([valid_patch, np.ones((n, 1), np.float32)], axis=1)
    # S7's published denominator: fully-OBSERVED patches + REG (unchanged by
    # any S8 manipulation; keeps enrich_s7 directly comparable to S7's nan)
    valid_s7 = np.concatenate([(~corr).astype(np.float32),
                               np.ones((n, 1), np.float32)], axis=1)
    calls = 0
    for i in range(0, n, batch):
        xb = torch.from_numpy(x_in[i:i + batch]).to(model.device)
        gen = None
        if mask_mode == "random":
            gen = torch.Generator()
            gen.manual_seed(flip_seed + 1000 + calls)
        calls += 1
        enc, _, _, _ = encode_replica(model, xb, mask_mode=mask_mode,
                                      mask_token=mask_token, flip_gen=gen,
                                      need_attn=True)
        acc.add(enc, corr[i:i + len(xb)], tokpos[i:i + len(xb)],
                valid_am[i:i + len(xb)], valid_s7[i:i + len(xb)],
                slice(i, i + len(xb)))
    return acc.finalize(nw, ns, C)


# ------------------------------------------------------------ Exp1 ----------

def exp1_jobs():
    return [(ds, mech, p, mode) for ds in DS for mech in MECHS1
            for p in P1 for mode in MODES]


def run_exp1_job(pipe, X, starts, ds, mech, p, mode):
    ctx, msk, gt, S, C, nw, ns = build_contexts(X, starts, mech, p)
    x_nan = ctx.copy()
    x_nan[msk] = np.nan
    flip_seed = zlib.crc32(f"{ds}:{mech}:{p}:{mode}".encode()) % (2 ** 31)
    t0 = time.time()
    with EncodePatch(pipe.model, mask_mode=mode, flip_seed=flip_seed):
        pred = predict_median(pipe, x_nan)
    mse, mse_pw = mse_windowed(pred, gt, S, C, nw)
    res = attn_pass(pipe.model, x_nan, msk, nw, ns, C, mask_mode=mode,
                    flip_seed=flip_seed)
    res.update({"mech": mech, "p": p, "mode": mode, "mse": mse,
                "mse_per_window": mse_pw, "seconds": round(time.time() - t0, 1)})
    return res


# ------------------------------------------------------------ Exp2 ----------

def aug_mask(rng, row):
    """[L] bool; 50% mcar / 50% block(24), p ~ U(0.05, 0.8)."""
    m = np.zeros(L, bool)
    p = rng.uniform(0.05, 0.8)
    if rng.random() < 0.5:
        m[:] = rng.random(L) < p
    else:
        for st in rng.integers(0, L - s5.BLOCK + 1, size=int(round(p * L / s5.BLOCK))):
            m[st:st + s5.BLOCK] = True
    return m


def train_token(args):
    ds = args.dataset
    device = "cuda"
    seed = s5.SEED + 31 + DS_IDX[ds]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pipe = load_bolt(device)
    model = pipe.model
    token = torch.nn.Parameter(torch.zeros(model.config.d_model, device=device))
    for prm in model.parameters():
        prm.requires_grad_(False)
    opt = torch.optim.AdamW([token], lr=args.lr)
    print(f"[{ds}] trainable params: {token.numel():,} (zero-init [MASK] token)",
          flush=True)

    Xtr = load_X(ds)
    Xtr = Xtr[: int(0.7 * len(Xtr))]
    N, C = Xtr.shape
    t0 = time.time()
    losses = []
    run = [0.0, 0.0]
    with EncodePatch(model, mask_mode="correct", mask_token=token):
        for step in range(1, args.steps + 1):
            starts = rng.integers(0, N - L - PRED, size=args.batch)
            chans = rng.integers(0, C, size=args.batch)
            ctx = np.stack([Xtr[st:st + L, c] for st, c in zip(starts, chans)])
            fut = np.stack([Xtr[st + L:st + L + PRED, c] for st, c in zip(starts, chans)])
            m = np.stack([aug_mask(rng, ctx[b]) for b in range(args.batch)])
            xb = torch.from_numpy(np.where(m, np.nan, ctx)).to(device)
            yb = torch.from_numpy(fut).to(device)
            with torch.enable_grad():
                loss, raw = s6.clipped_pinball(model, xb, yb, args.loss_clip)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([token], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            run[0] += loss.item()
            run[1] += raw.item()
            if step % args.log_every == 0 or step == 1:
                k = args.log_every if step > 1 else 1
                losses.append([step, run[0] / k, run[1] / k])
                print(f"[{ds}] step {step}/{args.steps} loss={run[0] / k:.4f} "
                      f"raw={run[1] / k:.2f} |tok|={token.norm().item():.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                run[0] = run[1] = 0.0
    meta = {"dataset": ds, "steps": args.steps, "batch": args.batch, "lr": args.lr,
            "loss_clip": args.loss_clip, "seed": seed, "train_frac": 0.7,
            "pred_len_train": PRED, "aug": "50% mcar / 50% block(24), p~U(0.05,0.8)",
            "token_norm": float(token.norm().item())}
    torch.save({"mask_token": token.detach().cpu(), "meta": meta, "losses": losses},
               TOK_CKPT[ds])
    s5.save_results({"meta": meta, "losses": losses}, TRAIN_JSON[ds])
    print(f"[{ds}] saved -> {TOK_CKPT[ds]} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------ Exp3 ----------

def exp3_jobs():
    return [(ds, mech, r, m) for ds in DS for mech in MECHS3
            for r in RATES3 for m in METHODS3]


class MpOnlyAdapter:
    """S5 fix2 miss_proj-only adapter (s5_fix_bolt_adapter_mponly.pt):
    Linear(1, d_model) on the per-patch missing fraction, added to the patch
    embedding via a hook on input_patch_embedding. Hook is installed per job
    and removed afterwards so nan/linear/tok jobs in the same process are
    untouched."""

    def __init__(self, model):
        sd = torch.load(MPONLY_CKPT, map_location=model.device)
        self.miss_proj = torch.nn.Linear(1, model.config.d_model).to(model.device)
        self.miss_proj.load_state_dict(sd["miss_proj"])
        self.model = model

    def __enter__(self):
        self.handle = self.model.input_patch_embedding.register_forward_hook(
            s5f.make_miss_hook(self.miss_proj, PATCH))
        return self

    def __exit__(self, *exc):
        self.handle.remove()


def run_exp3_job(pipe, tokens, mponly, X, starts, ds, mech, rate, method):
    ctx, msk, gt, S, C, nw, ns = build_contexts(X, starts, mech, rate)
    t0 = time.time()
    if method == "linear":
        pred = predict_median(pipe, fill_linear(ctx, msk))
    else:
        x_nan = ctx.copy()
        x_nan[msk] = np.nan
        if method == "nan":
            pred = predict_median(pipe, x_nan)
        elif method == "mponly":
            with mponly:
                pred = predict_median(pipe, x_nan)
        else:  # tok_<ds>
            with EncodePatch(pipe.model, mask_mode="correct",
                             mask_token=tokens[method[4:]]):
                pred = predict_median(pipe, x_nan)
    mse, mse_pw = mse_windowed(pred, gt, S, C, nw)
    return {"mech": mech, "rate": rate, "method": method, "mse": mse,
            "mse_per_window": mse_pw, "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------ Exp4 ----------

def exp4_jobs():
    return [(ds, mech, p) for ds in DS for mech in MECHS1 for p in P1]


def run_exp4_job(pipe, token, X, starts, ds, mech, p):
    ctx, msk, gt, S, C, nw, ns = build_contexts(X, starts, mech, p)
    x_nan = ctx.copy()
    x_nan[msk] = np.nan
    t0 = time.time()
    res = attn_pass(pipe.model, x_nan, msk, nw, ns, C, mask_mode="correct",
                    mask_token=token, track_token=True)
    res.update({"mech": mech, "p": p, "token_ds": ds,
                "seconds": round(time.time() - t0, 1)})
    return res


# ------------------------------------------------------------- anchor -------

def run_anchor(args):
    """The 6 S5 anchor cells (dataset-avg relMSE, p=0.7) through THIS harness
    (nan via encode_replica 'correct' == native path; linear via native)."""
    device = "cuda"
    pipe = load_bolt(device)
    if args.out is None:
        args.out = os.path.join(HERE, "s8_anchor.json")
    out = {}
    if os.path.exists(args.out):
        out = json.load(open(args.out))
    for ds in args.datasets.split(","):
        X, starts = s5.load_windows(s5.DATASETS[ds], args.windows, s5.SEED)
        ent = out.setdefault(ds, {})
        if "clean" not in ent:
            ent["clean"] = clean_baseline(pipe, X, starts)
            s5.save_results(out, args.out)
            print(f"anchor {ds:8s} clean mse={ent['clean']['mse']:.4f}", flush=True)
        for mech in MECHS1:
            for fill in ("nan", "linear"):
                key = f"{mech}:{fill}"
                if key in ent:
                    continue
                ctx, msk, gt, S, C, nw, ns = build_contexts(X, starts, mech, 0.7)
                t0 = time.time()
                if fill == "linear":
                    pred = predict_median(pipe, fill_linear(ctx, msk))
                else:
                    x_nan = ctx.copy()
                    x_nan[msk] = np.nan
                    with EncodePatch(pipe.model, mask_mode="correct"):
                        pred = predict_median(pipe, x_nan)
                mse, mse_pw = mse_windowed(pred, gt, S, C, nw)
                rel = mse / ent["clean"]["mse"]
                ent[key] = {"mse": mse, "rel": rel, "mse_per_window": mse_pw}
                s5.save_results(out, args.out)
                print(f"anchor {ds:8s} {key:18s} rel={rel:8.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    # verdict over whatever datasets this shard covered
    print("== anchor verdict (dataset-avg relMSE vs S5, +-5%) ==", flush=True)
    allok = True
    for (mech, fill), tgt in ANCHORS.items():
        rels = [out[ds][f"{mech}:{fill}"]["rel"]
                for ds in args.datasets.split(",")
                if ds in out and f"{mech}:{fill}" in out[ds]]
        if len(rels) < 3:
            continue
        avg = sum(rels) / len(rels)
        ok = abs(avg / tgt - 1) <= 0.05
        allok &= ok
        print(f"  {mech:10s} {fill:7s} avg={avg:7.3f} vs {tgt:6.3f} -> "
              f"{'OK' if ok else 'FAIL'}", flush=True)
    print(f"ANCHOR {'PASS' if allok else 'FAIL'}", flush=True)


# ----------------------------------------------------------------- tiny -----

def run_tiny(args):
    device = "cuda"
    pipe = load_bolt(device)
    model = pipe.model
    X, starts = s5.load_windows(s5.DATASETS["ETTh1"], 6, s5.SEED)
    ctx, msk, gt, S, C, nw, ns = build_contexts(X, starts, "block", 0.5)
    x_nan = ctx.copy()
    x_nan[msk] = np.nan
    # 1) replica 'correct' == native predictions (both AR blocks)
    p_ref = predict_median(pipe, x_nan, batch=512)
    with EncodePatch(model, mask_mode="correct"):
        p_rep = predict_median(pipe, x_nan, batch=512)
    d = np.abs(p_ref - p_rep).max()
    print(f"tiny: replica-vs-native max|diff| = {d:.3e} (want <1e-4)", flush=True)
    assert d < 1e-4, "encode replica mismatch"
    # 2) zero token changes forecasts (fully-missing patches become attendable)
    tok = torch.zeros(model.config.d_model, device=device)
    with EncodePatch(model, mask_mode="correct", mask_token=tok):
        p_tok = predict_median(pipe, x_nan, batch=512)
    print(f"tiny: zero-token-vs-native mean|diff| = "
          f"{np.abs(p_ref - p_tok).mean():.4f} (want > 0)", flush=True)
    # 3) mask modes run and produce enrichment
    for mode in MODES:
        res = run_exp1_job(pipe, X, starts, "ETTh1", "block", 0.5, mode)
        print(f"tiny: exp1 block:0.5 mode={mode:8s} mse={res['mse']:9.4f} "
              f"enrich={res['enrich']:.3f} ent={np.mean(res['ent']):.3f}", flush=True)
    # 4) token training smoke test (5 steps, grad flows)
    args.dataset, args.steps, args.log_every, args.batch = "ETTh1", 5, 1, 64
    train_token(args)
    sd = torch.load(TOK_CKPT["ETTh1"], map_location="cpu")
    print(f"tiny: trained token |w|={sd['mask_token'].norm():.4f} (want > 0)", flush=True)
    os.remove(TOK_CKPT["ETTh1"])
    os.remove(TRAIN_JSON["ETTh1"])
    print("TINY OK", flush=True)


# ----------------------------------------------------------------- main -----

def load_tokens(device, needed):
    toks = {}
    for ds in needed:
        sd = torch.load(TOK_CKPT[ds], map_location=device)
        toks[ds] = sd["mask_token"].to(device)
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--train-token", action="store_true")
    ap.add_argument("--exp1", action="store_true")
    ap.add_argument("--exp3", action="store_true")
    ap.add_argument("--exp4", action="store_true")
    ap.add_argument("--zero-token", action="store_true",
                    help="Exp4 control: untrained all-zero [MASK] token "
                         "(section exp4_zero, shard files *_exp4z_*)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--dataset", choices=list(DS), default="ETTh1")
    ap.add_argument("--datasets", default=",".join(DS))
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss-clip", type=float, default=100.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(s5.SEED)
    np.random.seed(s5.SEED)

    if args.tiny:
        run_tiny(args)
        return
    if args.anchor:
        run_anchor(args)
        return
    if args.train_token:
        train_token(args)
        return
    if args.merge:
        merge_all()
        return

    phase = ("exp1" if args.exp1 else "exp3" if args.exp3 else
             "exp4" if args.exp4 else None)
    assert phase, "nothing to do"
    section = "exp4_zero" if (phase == "exp4" and args.zero_token) else phase
    tag = "exp4z" if (phase == "exp4" and args.zero_token) else phase
    out = args.out or os.path.join(HERE, f"s8_results_{tag}_shard{args.shard}.json")
    results = {}
    if os.path.exists(out):
        results = json.load(open(out))
        print(f"resume: loaded {out}", flush=True)
    results.setdefault("meta", {
        "track": "S8 missing-signal causality & generalization (bolt only)",
        "seed": s5.SEED, "L": L, "H": H, "patch": PATCH, "n_patch": NPATCH,
        "windows": args.windows, "seeds_mcar_block": args.seeds,
        "anchors": {f"{m}:{f}": v for (m, f), v in ANCHORS.items()},
    })
    save = lambda: s5.save_results(results, out)

    device = "cuda"
    pipe = load_bolt(device)
    jobs = {"exp1": exp1_jobs, "exp3": exp3_jobs, "exp4": exp4_jobs}[phase]()
    jobs = jobs[args.shard::args.nshards]
    print(f"{phase} shard {args.shard}/{args.nshards}: {len(jobs)} jobs", flush=True)

    tokens, mponly = {}, None
    if phase == "exp3":
        need = sorted({m[4:] for *_, m in jobs if m.startswith("tok_")})
        tokens = load_tokens(device, need)
        if any(m == "mponly" for *_, m in jobs):
            mponly = MpOnlyAdapter(pipe.model)
    if phase == "exp4":
        if args.zero_token:
            d_model = pipe.model.config.d_model
            tokens = {ds: torch.zeros(d_model, device=device)
                      for ds in sorted({j[0] for j in jobs})}
        else:
            tokens = load_tokens(device, sorted({j[0] for j in jobs}))

    loaded = {}
    sec = results.setdefault(section, {})
    for job in jobs:
        ds = job[0]
        if ds not in loaded:
            loaded[ds] = s5.load_windows(s5.DATASETS[ds], args.windows, s5.SEED)
        X, starts = loaded[ds]
        if phase == "exp1":
            _, mech, p, mode = job
            key = f"{mech}:{p}:{mode}"
        elif phase == "exp3":
            _, mech, rate, method = job
            key = f"{mech}:{rate}:{method}"
        else:
            _, mech, p = job
            key = f"{mech}:{p}"
        if key in sec.get(ds, {}):
            continue
        cl = results.setdefault("clean", {})
        if ds not in cl:
            t0 = time.time()
            cl[ds] = clean_baseline(pipe, X, starts)
            cl[ds]["seconds"] = round(time.time() - t0, 1)
            save()
            print(f"  clean {ds} mse={cl[ds]['mse']:.4f}", flush=True)
        if phase == "exp1":
            res = run_exp1_job(pipe, X, starts, *job)
            msg = (f"mse={res['mse']:10.4f} enrich={res['enrich']:.3f} "
                   f"ent={np.mean(res['ent']):.3f}")
        elif phase == "exp3":
            res = run_exp3_job(pipe, tokens, mponly, X, starts, *job)
            rel = res["mse"] / cl[ds]["mse"]
            msg = f"mse={res['mse']:10.4f} rel={rel:8.3f}"
        else:
            res = run_exp4_job(pipe, tokens[ds], X, starts, *job)
            msg = (f"enrich_tok_s7={res['enrich_tok_s7']:.3f} "
                   f"enrich_tok_am={res['enrich_tok_am']:.3f} "
                   f"ent={np.mean(res['ent']):.3f}")
        sec.setdefault(ds, {})[key] = res
        save()
        print(f"  {phase} {ds:8s} {key:34s} {msg} ({res['seconds']}s)", flush=True)
    save()
    print("SHARD DONE", flush=True)


def merge_all():
    out = os.path.join(HERE, "s8_results.json")
    merged = {}
    import glob
    for path in sorted(glob.glob(os.path.join(HERE, "s8_results_*_shard*.json"))):
        r = json.load(open(path))
        for sec, dv in r.items():
            if sec == "meta":
                merged.setdefault("meta", dv)
                continue
            tgt = merged.setdefault(sec, {})
            for k, v in dv.items():
                if isinstance(v, dict) and isinstance(tgt.get(k), dict):
                    tgt[k].update(v)
                else:
                    tgt[k] = v
    # anchor results (per-dataset files) + training curves
    for ds in DS:
        pa = os.path.join(HERE, f"s8_anchor_{ds}.json")
        if os.path.exists(pa):
            merged.setdefault("anchor", {})[ds] = json.load(open(pa)).get(ds, {})
        if os.path.exists(TRAIN_JSON[ds]):
            merged.setdefault("train", {})[ds] = json.load(open(TRAIN_JSON[ds]))
    s5.save_results(merged, out)
    print(f"merged -> {out}")


if __name__ == "__main__":
    main()
