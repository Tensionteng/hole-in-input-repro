#!/usr/bin/env python
"""S7: deep attribution of the missing-context damage in chronos-bolt-base.

S5 (run_s5_missing.py / run_s5_fix.py, s5_fix_notes.md) established that
zero-filling the context catastrophically damages bolt, and that an oracle
probe rescuing the InstanceNorm scaling statistics recovers ~82% of the
damage under mcar p=0.5. S7 decomposes *where* the damage lives, in three
layers. Main model: chronos-bolt-base only. Windows / mask seeds / rates are
identical to run_s5_missing.py (imported, not copied): 300 windows, forecast
origin in the last 20% of each series, 2 mask seeds for mcar/block, 1 for the
deterministic rank-based mnar_* mechanisms, L=512, H=96, median quantile as
point forecast.

Layer 0 -- 2x2 input decomposition. Cells per (dataset, mechanism, p in
  {0.3,0.5,0.7}):
    A: zero-filled values + polluted stats   (= plain zero-fill, S5 anchor)
    B: zero-filled values + oracle stats     (= the S5 oracle probe exactly:
        zero_oscale, input rescaled by mean|x_clean|/mean|x_zerofilled|.
        NOTE: bolt's InstanceNorm is scale-invariant, so an input rescale
        cancels inside the normalization and the whole effect is a global
        rescale of the FORECAST by that factor -- the S5 82.3% recovery is
        therefore a property of the output-unscale magnitude channel only)
    C: complete values + polluted stats      (clean context fed; a forward
        hook on model.instance_norm forces loc/scale computed on the
        ZERO-FILLED context -- undoes the natural scaling (x*scale+loc, bolt
        uses no arcsinh), re-applies the override, and returns it as
        loc_scale so the output unscale uses it too)
    D: complete values + true stats          (= clean, one run per dataset)
  Diagnostic extra cells (same grid, for the notes' factorization contrast):
    Bh: zero values + hook-forced CLEAN-context loc/scale (the literal
        InstanceNorm stats swap; explodes on low-CV channels because missing
        positions become -loc/scale = -mean/std tokens, e.g. -200 for a
        1000+-5 pressure channel)
    Cr: clean values rescaled by mean|x_zerofilled|/mean|x_clean| (the
        scale-factor symmetric counterpart of B: forecast deflated by 1/s)
  B/Cr form an exact 2x2 factorial on the global scale statistic; Bh/C form
  an exact 2x2 factorial on InstanceNorm loc/scale (input norm + unscale).
  H=96 > bolt's native prediction_length (64), so pipe.predict runs a second
  autoregressive block whose context (length 576, batch x9 quantiles) gets
  NATURAL stats (--block2 natural; forcing clean stats there too blows up
  weather via tiny-std channels, verified empirically). --block2 override
  exists for ablation only.
  Hard anchors (s5_missing_results.json / s5_fix_notes.md): cell A dataset-avg
  relMSE at p=0.7: mcar 13.74 / block 5.93 / mnar_high 11.32 / mnar_extreme
  12.13 (+-5%); cell B recovery (relA-relB)/(relA-1) at mcar p=0.5:
  82.3/78.7/82.3 % on ETTh1/ETTm1/weather (exact by construction: B IS the
  S5 zero_oscale probe; verified numerically before the full run).

Layer 1 -- representation drift map. {clean, zero, linear, nan} x
  {mcar, block, mnar_high} x p in {0.3,0.7}. A re-implementation of bolt's
  encode() (chronos_bolt.py l.277-329, line-for-line) calls the encoder with
  output_hidden_states=True; hidden_states[0] IS the input_patch_embedding
  output (eval mode => dropout identity), so the "embed" layer and all 12
  encoder layers are covered. Per patch position (32 patches; REG token
  excluded from drift) we record cosine similarity and relative L2
  (||h_dirty-h_clean||/||h_clean||) vs the paired clean run, aggregated:
  mean over all patches / corrupted patches (>=1 missing point) / clean
  patches, and binned by each patch's distance to the nearest observed point
  (time-step bins [0,1,2,4,8,16,32,64,129)).

Layer 2 -- attention audit (sink detection). Same configs as Layer 1, same
  forward, output_attentions=True (encoder self-attention, 12 layers x 12
  heads, 33 positions = 32 patches + REG at the LAST position). Per-window
  aggregated scalars only (raw tensors never stored): per-layer-per-head
  entropy; attention mass on corrupted patches vs observed patches and the
  enrichment ratio over the uniform baseline (mass_corrupt / fraction of
  corrupted valid key positions); mass on the REG token and on patch 0
  (classic sink positions). Question: do garbage patches (identical zero
  tokens under zero-fill / mean-fill+mask tokens under nan) ABSORB mass
  (protective sink, enrichment > 1) or DILUTE mass onto garbage (harmful)?

Sharding: 8 GPUs, explicit schedule (--nshards 8): shards 0-3 take
weather x {mcar, block, mnar_high, mnar_extreme}, shards 4-7 take two
(ETT, mechanism) jobs each; otherwise jobs are strided. Each shard writes
s7_attrib_results_shard{i}.json incrementally (tmp+replace after every
config); merge_shards() combines them into s7_attrib_results.json.
"""
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import pandas as pd
import torch

import run_s5_missing as s5

L, H = s5.L, s5.H           # 512, 96
PATCH = 16                  # bolt-base input_patch_size
NPATCH = L // PATCH         # 32
REG = NPATCH                # REG token position (appended last)
NHEADS, NLAYERS = 12, 12    # bolt-base encoder
NHID = NLAYERS + 1          # hidden states: embedding output + 12 layers
EPS = 1e-5                  # InstanceNorm eps

MECHS_ALL = ("mcar", "block", "mnar_high", "mnar_extreme")
MECHS_L12 = ("mcar", "block", "mnar_high")
P_L0 = (0.3, 0.5, 0.7)
P_L12 = (0.3, 0.7)
CONDS = ("zero", "linear", "nan")
CELLS = ("A", "B", "C", "Bh", "Cr")
DIST_EDGES = (0, 1, 2, 4, 8, 16, 32, 64, 129)  # time-step bins, patch dist = min over its points
NBINS = len(DIST_EDGES) - 1

# S5 hard anchors (dataset-avg relMSE, p=0.7, cell A) and B-cell recovery at mcar p=0.5
ANCHOR_A_P07 = {"mcar": 13.74, "block": 5.93, "mnar_high": 11.32, "mnar_extreme": 12.13}
ANCHOR_B_REC = {"ETTh1": 0.823, "ETTm1": 0.787, "weather": 0.823}


# ------------------------------------------------------------- model I/O ----

def load_bolt(device):
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
    pipe.model.eval()  # config has dropout_rate=0.1; everything here is deterministic
    return pipe


class InstanceNormOverride:
    """Forward hook on bolt's InstanceNorm forcing external (loc, scale).

    The module's own scaling is undone (x = scaled*scale + loc -- bolt uses
    no arcsinh) and the override re-applied; the override is also returned as
    loc_scale so the final unscale in forward() uses the same statistics.
    Fires only on first-block contexts (length L, matching batch); the AR
    second block (length L+64, batch x9) keeps natural stats unless
    mode='override'."""

    def __init__(self, model, mode="natural"):
        assert not model.instance_norm.use_arcsinh, "undo assumes no arcsinh"
        assert model.config.chronos_config["input_patch_size"] == PATCH
        self.device = model.device
        self.mode = mode          # block-2 handling: 'natural' | 'override'
        self.loc = None
        self.scale = None
        model.instance_norm.register_forward_hook(self._hook)

    def set(self, loc, scale):
        """loc/scale: np arrays [b] (per series), or None to disable."""
        if loc is None:
            self.loc = self.scale = None
            return
        self.loc = torch.as_tensor(loc, dtype=torch.float32, device=self.device).view(-1, 1)
        sc = torch.as_tensor(scale, dtype=torch.float32, device=self.device).view(-1, 1)
        self.scale = torch.where(sc == 0, torch.full_like(sc, EPS), sc)

    def _hook(self, module, inputs, output):
        if self.loc is None:
            return
        x = inputs[0]
        b, n = x.shape
        if n == L and b == self.loc.shape[0]:
            lo, sc = self.loc, self.scale
        elif self.mode == "override" and b % self.loc.shape[0] == 0 and b != self.loc.shape[0]:
            rep = b // self.loc.shape[0]
            lo = self.loc.repeat_interleave(rep, dim=0)
            sc = self.scale.repeat_interleave(rep, dim=0)
        else:
            return
        scaled, (loc, scale) = output
        x_raw = scaled.float() * scale + loc
        new = (x_raw - lo) / sc
        return new.to(scaled.dtype), (lo, sc)


@torch.no_grad()
def predict_median(pipe, hook, ctx, loc=None, scale=None, batch=512):
    """ctx: [n, L] float32 (no NaN on this path) -> [n, H] median forecast."""
    outs = []
    for i in range(0, len(ctx), batch):
        xb = torch.from_numpy(ctx[i:i + batch]).to(pipe.model.device)
        if hook is not None:
            hook.set(None if loc is None else loc[i:i + batch],
                     None if scale is None else scale[i:i + batch])
        q = pipe.predict(xb, prediction_length=H)  # [b, 9, H]
        outs.append(q[:, 4, :].float().cpu().numpy())
    if hook is not None:
        hook.set(None, None)
    return np.concatenate(outs)


@torch.no_grad()
def encode_full(model, xb, need_attn):
    """Line-for-line replica of ChronosBoltModelForForecasting.encode()
    (chronos_bolt.py l.277-329) exposing hidden states and attentions.
    xb: [B, L] float32 CUDA tensor, may contain NaN (native bolt path)."""
    mask = torch.isnan(xb).logical_not().to(xb.dtype)                       # l.280
    B = xb.shape[0]
    xc, _ = model.instance_norm(xb)                                         # l.288
    xc = xc.to(model.dtype)
    mask = mask.to(model.dtype)
    pc = model.patch(xc)                                                    # l.296
    pm = torch.nan_to_num(model.patch(mask), nan=0.0)                       # l.297
    pc = torch.where(pm > 0.0, pc, 0.0)                                     # l.298
    pc = torch.cat([pc, pm], dim=-1)                                        # l.300
    am = pm.sum(dim=-1) > 0                                                 # l.303
    emb = model.input_patch_embedding(pc)                                   # l.305
    reg_ids = torch.full((B, 1), model.config.reg_token_id, device=xb.device)
    emb = torch.cat([emb, model.shared(reg_ids)], dim=-2)                   # l.315
    am = torch.cat([am.to(model.dtype),
                    torch.ones_like(reg_ids).to(model.dtype)], dim=-1)      # l.316-321
    return model.encoder(attention_mask=am, inputs_embeds=emb,
                         output_hidden_states=True, output_attentions=need_attn)


# --------------------------------------------------------- data / masks ----

def fill_cond(x, mask, cond):
    """x: [N, L] clean; mask: [N, L] bool (True=missing)."""
    if cond == "zero":
        out = x.copy()
        out[mask] = 0.0
        return out
    if cond == "nan":
        out = x.copy()
        out[mask] = np.nan
        return out
    if cond == "linear":
        out = x.copy()
        t = np.arange(L)
        for i in range(len(x)):
            valid = np.flatnonzero(~mask[i])
            if len(valid) == 0:
                out[i] = 0.0
            else:
                out[i] = np.interp(t, valid, x[i, valid])
        return out
    raise ValueError(cond)


def dist_to_observed(obs):
    """obs: [N, L] bool -> [N, L] int32 distance to nearest observed point."""
    N, Lx = obs.shape
    big = Lx + 1
    d = np.where(obs, 0, big).astype(np.int32)
    for t in range(1, Lx):
        d[:, t] = np.minimum(d[:, t], d[:, t - 1] + 1)
    for t in range(Lx - 2, -1, -1):
        d[:, t] = np.minimum(d[:, t], d[:, t + 1] + 1)
    return d


def patch_meta(mask):
    """mask [N, L] bool -> corrupted-patch flags [N, NPATCH] bool and
    per-patch distance bin [N, NPATCH] (distance = min over patch points of
    the distance to the nearest observed point)."""
    N = mask.shape[0]
    corr = mask.reshape(N, NPATCH, PATCH).any(axis=2)
    d = dist_to_observed(~mask).reshape(N, NPATCH, PATCH).min(axis=2)
    bins = np.digitize(d, DIST_EDGES[1:-1])  # 0..NBINS-1
    return corr, bins.astype(np.int64)


def series_stats(x):
    """Population mean/std (ddof=0, matching InstanceNorm's nanmean version)
    per row of x [N, L]."""
    loc = x.mean(axis=1)
    scale = x.std(axis=1)
    return loc.astype(np.float32), scale.astype(np.float32)


# -------------------------------------------------------------- Layer 0 ----

def run_layer0(pipe, hook, X, starts, mech, p, n_seeds):
    """Cells A/B/C/Bh/Cr for one (dataset, mechanism, p)."""
    C = X.shape[1]
    det = mech in ("mnar_high", "mnar_extreme")
    ctxs, xz, gts = [], [], []
    for wi, s in enumerate(starts):
        x_clean = X[s:s + L].T.copy()
        y = X[s + L:s + L + H].T.copy()
        for ms in ((0,) if det else range(n_seeds)):
            mask = s5.make_mask(mech, p, wi, ms, C, x=x_clean)
            z = x_clean.copy()
            z[mask] = 0.0
            ctxs.append(x_clean)
            xz.append(z)
            gts.append(y)
    S = len(ctxs)
    x_clean = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
    x_zero = np.stack(xz).reshape(S * C, L).astype(np.float32)
    gt = np.stack(gts).astype(np.float64)
    loc_c, sc_c = series_stats(x_clean)   # oracle stats (cell Bh)
    loc_z, sc_z = series_stats(x_zero)    # polluted stats (cell C)
    # global scale-statistic factor s = mean|x_clean| / mean|x_zero| (S5 zero_oscale)
    num = np.abs(x_clean).mean(axis=1)
    den = np.abs(x_zero).mean(axis=1)
    s_fac = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 1.0).astype(np.float32)
    out = {}
    for cell, ctx, loc, sc in (("A", x_zero, None, None),
                               ("B", x_zero * s_fac[:, None], None, None),
                               ("C", x_clean, loc_z, sc_z),
                               ("Bh", x_zero, loc_c, sc_c),
                               ("Cr", x_clean / s_fac[:, None], None, None)):
        t0 = time.time()
        pred = predict_median(pipe, hook, ctx, loc, sc).reshape(S, C, H).astype(np.float64)
        mse = ((pred - gt) ** 2).mean(axis=(1, 2))
        nw = len(starts)
        out[cell] = {"mse": float(mse.mean()),
                     "mse_per_window": mse.reshape(nw, -1).mean(axis=1).tolist(),
                     "seconds": round(time.time() - t0, 1)}
    return out


def run_clean_mse(pipe, X, starts):
    """Cell D (= S5 clean): one plain predict over all windows/channels."""
    C = X.shape[1]
    ctxs, gts = [], []
    for s in starts:
        ctxs.append(X[s:s + L].T.copy())
        gts.append(X[s + L:s + L + H].T.copy())
    S = len(ctxs)
    ctx = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
    gt = np.stack(gts).astype(np.float64)
    pred = predict_median(pipe, None, ctx).reshape(S, C, H).astype(np.float64)
    mse = ((pred - gt) ** 2).mean(axis=(1, 2))
    return {"mse": float(mse.mean()),
            "mse_per_window": mse.reshape(len(starts), -1).mean(axis=1).tolist()}


# -------------------------------------------------------- Layers 1 + 2 ----

class DriftAttnAcc:
    """Per-config accumulators: per-series vectors -> per-window means, plus
    global (all-series) sums. Nothing bigger than [n_series, n_layers]."""

    def __init__(self, n_series):
        self.cos_mean = np.zeros((n_series, NHID), np.float64)
        self.cos_corr = np.zeros((n_series, NHID), np.float64)
        self.cos_obs = np.zeros((n_series, NHID), np.float64)
        self.l2_mean = np.zeros((n_series, NHID), np.float64)
        self.dist_cos = np.zeros((NHID, NBINS), np.float64)
        self.dist_l2 = np.zeros((NHID, NBINS), np.float64)
        self.dist_cnt = np.zeros(NBINS, np.float64)
        self.n_corr = np.zeros(n_series, np.float64)  # corrupted patches per series
        self.has_corr = np.zeros(n_series, np.float64)
        self.has_obs = np.zeros(n_series, np.float64)
        self.ent = np.zeros((n_series, NLAYERS), np.float64)
        self.mc = np.zeros((n_series, NLAYERS), np.float64)
        self.mreg = np.zeros((n_series, NLAYERS), np.float64)
        self.mfirst = np.zeros((n_series, NLAYERS), np.float64)
        self.ent_head = np.zeros((NLAYERS, NHEADS), np.float64)
        self.mc_head = np.zeros((NLAYERS, NHEADS), np.float64)
        self.frac_corr = np.zeros(n_series, np.float64)
        self.n_seen = 0

    def add(self, hidden_c, enc_x, corr, dbins, valid_key, sl):
        """hidden_c: tuple NHID x [B,33,D] (clean); enc_x: corrupted encoder
        output; corr/dbins: [B,NPATCH]; valid_key: [B,NPATCH+1] float mask."""
        B = corr.shape[0]
        corr_t = torch.from_numpy(corr).to(hidden_c[0].device)
        dbins_t = torch.from_numpy(dbins).to(hidden_c[0].device)
        for l in range(NHID):
            a = hidden_c[l][:, :NPATCH, :].float()
            b = enc_x.hidden_states[l][:, :NPATCH, :].float()
            cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)     # [B,32]
            l2 = (b - a).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-6)    # [B,32]
            ncorr = corr_t.sum(1)
            nobs = (~corr_t).sum(1)
            nan = torch.full_like(ncorr, float("nan"), dtype=cos.dtype)
            # NaN when the patch set is empty (e.g. mcar has ~no fully-clean
            # patches) -- a clamped 0/1=0 would fake maximal drift
            self.cos_mean[sl, l] = cos.mean(1).cpu().numpy()
            self.cos_corr[sl, l] = torch.where(
                ncorr > 0, (cos * corr_t).sum(1) / ncorr.clamp_min(1), nan).cpu().numpy()
            self.cos_obs[sl, l] = torch.where(
                nobs > 0, (cos * (~corr_t)).sum(1) / nobs.clamp_min(1), nan).cpu().numpy()
            self.l2_mean[sl, l] = l2.mean(1).cpu().numpy()
            for bn in range(NBINS):
                m = dbins_t == bn
                cnt = m.sum()
                if cnt > 0:
                    self.dist_cos[l, bn] += (cos * m).sum().item()
                    self.dist_l2[l, bn] += (l2 * m).sum().item()
                    if l == 0:
                        self.dist_cnt[bn] += cnt.item()
        self.n_corr[sl] = corr_t.sum(1).cpu().numpy()
        self.has_corr[sl] = (corr_t.sum(1) > 0).cpu().numpy()
        self.has_obs[sl] = ((~corr_t).sum(1) > 0).cpu().numpy()

        vk = torch.from_numpy(valid_key).to(hidden_c[0].device)           # [B,33]
        ck = torch.cat([corr_t, torch.zeros(B, 1, dtype=torch.bool,
                                            device=corr_t.device)], dim=1)  # REG not corrupted
        self.frac_corr[sl] = (ck.float().sum(1) / vk.sum(1).clamp_min(1.0)).cpu().numpy()
        if enc_x.attentions is None:
            return
        for l, a in enumerate(enc_x.attentions):                          # [B,12,33,33]
            a = a.float()
            ent = -(a * (a + 1e-12).log()).sum(-1)                        # [B,12,33]
            self.ent[sl, l] = ent.mean(dim=(1, 2)).cpu().numpy()
            self.ent_head[l] += ent.mean(dim=(0, 2)).cpu().numpy() * B
            mc = (a * ck[:, None, None, :].float()).sum(-1)               # [B,12,33]
            self.mc[sl, l] = mc.mean(dim=(1, 2)).cpu().numpy()
            self.mc_head[l] += mc.mean(dim=(0, 2)).cpu().numpy() * B
            self.mreg[sl, l] = a[..., REG].mean(dim=(1, 2)).cpu().numpy()
            self.mfirst[sl, l] = a[..., 0].mean(dim=(1, 2)).cpu().numpy()
        self.n_seen += B

    def finalize(self, n_windows, n_seeds, C):
        """Per-window means (over seeds x channels) + global means. cos_corr/
        cos_obs use NaN-aware means (empty patch sets excluded, not zeroed)."""
        per = n_seeds * C
        with np.errstate(invalid="ignore"):
            g = lambda v: np.nanmean(  # nanmean over (seed, channel); may warn on all-NaN
                v.reshape(n_windows, per, v.shape[-1]), axis=1)  # [300, Lx]
            res = {
                "cos_mean_pw": g(self.cos_mean).tolist(),
                "l2_mean_pw": g(self.l2_mean).tolist(),
                "ent_pw": g(self.ent).tolist(),
                "mc_pw": g(self.mc).tolist(),
                "mreg_pw": g(self.mreg).tolist(),
                "mfirst_pw": g(self.mfirst).tolist(),
                "cos_mean": np.nanmean(self.cos_mean, axis=0).tolist(),
                "cos_corr": np.nanmean(self.cos_corr, axis=0).tolist(),
                "cos_obs": np.nanmean(self.cos_obs, axis=0).tolist(),
                "l2_mean": np.nanmean(self.l2_mean, axis=0).tolist(),
                "ent": self.ent.mean(0).tolist(),
                "mc": self.mc.mean(0).tolist(),
                "mreg": self.mreg.mean(0).tolist(),
                "mfirst": self.mfirst.mean(0).tolist(),
                "frac_corr": float(self.frac_corr.mean()),
                "n_corr_patches": float(self.n_corr.mean()),
                "frac_series_with_corr": float(self.has_corr.mean()),
                "frac_series_with_obs": float(self.has_obs.mean()),
                "dist_cos": (self.dist_cos / np.maximum(self.dist_cnt[None, :], 1)).tolist(),
                "dist_l2": (self.dist_l2 / np.maximum(self.dist_cnt[None, :], 1)).tolist(),
                "dist_cnt": self.dist_cnt.tolist(),
            }
        # NaN is not JSON-serializable by strict parsers; use None
        def clean(o):
            if isinstance(o, float) and o != o:
                return None
            if isinstance(o, list):
                return [clean(v) for v in o]
            if isinstance(o, dict):
                return {k: clean(v) for k, v in o.items()}
            return o
        res = clean(res)
        if self.n_seen > 0:
            res["ent_head"] = (self.ent_head / self.n_seen).tolist()
            res["mc_head"] = (self.mc_head / self.n_seen).tolist()
        return res


def run_l12(pipe, X, starts, mech, ds_key, results, save, batch_series=256):
    """Layers 1+2 for one (dataset, mechanism): clean reference pass (once per
    dataset, attention included) then {p x seed x cond} forwards with paired
    drift/attention aggregation."""
    model = pipe.model
    dev = model.device
    C = X.shape[1]
    det = mech == "mnar_high"
    seeds = (0,) if det else (0, 1)
    nw = len(starts)
    wblk = max(1, batch_series // (len(seeds) * C))

    # clean contexts per window [nw, C, L]
    xc_all = np.stack([X[s:s + L].T.copy() for s in starts]).astype(np.float32)

    attn_ds = results.setdefault("attn", {}).setdefault(ds_key, {})
    if "clean" not in attn_ds:  # clean attention reference, once per dataset
        acc = DriftAttnAcc(nw * C)
        for w0 in range(0, nw, wblk * len(seeds)):
            w1 = min(w0 + wblk * len(seeds), nw)
            xb_np = xc_all[w0:w1].reshape(-1, L)
            sl = slice(w0 * C, w1 * C)
            enc = encode_full(model, torch.from_numpy(xb_np).to(dev), need_attn=True)
            zeros = np.zeros((len(xb_np), NPATCH), bool)
            acc.add(enc.hidden_states, enc, zeros, np.zeros((len(xb_np), NPATCH), np.int64),
                    np.ones((len(xb_np), NPATCH + 1), np.float32), sl)
        attn_ds["clean"] = acc.finalize(nw, 1, C)
        save()
        print(f"  [l12] {ds_key} clean attn ref done", flush=True)

    # masks per (p, seed): [nw, C, L]
    masks = {}
    for p in P_L12:
        for ms in seeds:
            m = np.stack([s5.make_mask(mech, p, wi, ms, C, x=xc_all[wi])
                          for wi in range(nw)])
            masks[(p, ms)] = m

    for p in P_L12:
        for cond in CONDS:
            key = f"{mech}:{cond}:{p}"
            drift_ds = results.setdefault("drift", {}).setdefault(ds_key, {}).setdefault(mech, {})
            if key in drift_ds:
                continue
            acc = DriftAttnAcc(nw * len(seeds) * C)
            t0 = time.time()
            for w0 in range(0, nw, wblk):
                w1 = min(w0 + wblk, nw)
                rows_x, rows_m = [], []
                for wi in range(w0, w1):          # order: (wi, ms, c)
                    for ms in seeds:
                        rows_x.append(xc_all[wi])
                        rows_m.append(masks[(p, ms)][wi])
                xb_clean = np.concatenate(rows_x)                      # [B, L]
                mb = np.concatenate(rows_m)
                B = len(xb_clean)
                sl = slice(w0 * len(seeds) * C, w1 * len(seeds) * C)
                hc = encode_full(model, torch.from_numpy(xb_clean).to(dev),
                                 need_attn=False).hidden_states
                ctx = fill_cond(xb_clean, mb, cond)
                enc = encode_full(model, torch.from_numpy(ctx).to(dev), need_attn=True)
                corr, dbins = patch_meta(mb)
                if cond == "nan":
                    valid = (~np.isnan(ctx).reshape(B, NPATCH, PATCH).any(axis=2)
                             ).astype(np.float32)
                else:
                    valid = np.ones((B, NPATCH), np.float32)
                valid = np.concatenate([valid, np.ones((B, 1), np.float32)], axis=1)
                acc.add(hc, enc, corr, dbins, valid, sl)
            res = acc.finalize(nw, len(seeds), C)
            res["seconds"] = round(time.time() - t0, 1)
            drift_ds[key] = res
            attn_ds.setdefault(mech, {})[f"{cond}:{p}"] = {
                k: res[k] for k in ("ent_pw", "mc_pw", "mreg_pw", "mfirst_pw",
                                    "ent", "mc", "mreg", "mfirst", "frac_corr",
                                    "ent_head", "mc_head")}
            save()
            print(f"  [l12] {ds_key} {key:24s} cos0={res['cos_mean'][0]:.4f} "
                  f"cosN={res['cos_mean'][-1]:.4f} ({res['seconds']}s)", flush=True)


# ----------------------------------------------------------------- main ----

def save_results(results, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh)
    os.replace(tmp, path)


def job_schedule(nshards):
    """Returns list of jobs per shard. For 8 shards: 0-3 weather x 4 mechs,
    4-7 two (ETT, mech) jobs each. Otherwise strided over the flat list."""
    ett_jobs = [(ds, m) for ds in ("ETTh1", "ETTm1") for m in MECHS_ALL]
    w_jobs = [("weather", m) for m in MECHS_ALL]
    if nshards == 8:
        sched = [[w] for w in w_jobs]
        for i in range(4):
            sched.append([ett_jobs[2 * i], ett_jobs[2 * i + 1]])
        return sched
    flat = [(ds, m) for ds in ("ETTh1", "ETTm1", "weather") for m in MECHS_ALL]
    return [flat[i::nshards] for i in range(nshards)]


def merge_shards(nshards=8, out=None):
    out = out or os.path.join(HERE, "s7_attrib_results.json")
    merged = {}
    for i in range(nshards):
        p = os.path.join(HERE, f"s7_attrib_results_shard{i}.json")
        if not os.path.exists(p):
            print(f"warn: {p} missing")
            continue
        r = json.load(open(p))
        for sec, dv in r.items():
            if sec == "meta":
                merged.setdefault("meta", dv)
                continue
            for ds, mv in dv.items():
                if not isinstance(mv, dict):
                    merged.setdefault(sec, {})[ds] = mv
                    continue
                tgt = merged.setdefault(sec, {}).setdefault(ds, {})
                for k, v in mv.items():
                    if isinstance(v, dict) and isinstance(tgt.get(k), dict):
                        tgt[k].update(v)
                    else:
                        tgt[k] = v
    save_results(merged, out)
    print(f"merged -> {out}")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--batch-predict", type=int, default=512)
    ap.add_argument("--batch-enc", type=int, default=256)
    ap.add_argument("--block2", choices=["natural", "override"], default="natural")
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--tiny", action="store_true",
                    help="hook correctness self-test + one mini config, prints only")
    ap.add_argument("--anchor-test", action="store_true",
                    help="300-window anchor check: A at p=0.7 (4 mechs) + B recovery at mcar p=0.5")
    ap.add_argument("--skip-l12", action="store_true")
    ap.add_argument("--rerun-l12", action="store_true",
                    help="drop existing drift/attn sections and recompute layers 1+2")
    args = ap.parse_args()

    if args.merge:
        merge_shards(args.nshards)
        return

    torch.manual_seed(s5.SEED)
    np.random.seed(s5.SEED)
    device = "cuda"
    pipe = load_bolt(device)
    hook = InstanceNormOverride(pipe.model, mode=args.block2)

    if args.tiny:
        # 1) hook identity: overriding clean stats with the context's own
        # natural stats must reproduce the no-hook forecast bit-near-exactly
        rng = np.random.default_rng(0)
        X, starts = s5.load_windows(s5.DATASETS["ETTh1"], 6, s5.SEED)
        x = np.stack([X[s:s + L, 0] for s in starts]).astype(np.float32)
        p_ref = predict_median(pipe, None, x, batch=4)
        loc, sc = series_stats(x)
        p_idn = predict_median(pipe, hook, x, loc, sc, batch=4)
        print(f"tiny: identity max|diff| = {np.abs(p_ref - p_idn).max():.3e} (want ~1e-5)")
        # 2) shift invariance sanity: stats from a shifted context move the forecast
        p_sh = predict_median(pipe, hook, x, loc + 1.0, sc, batch=4)
        print(f"tiny: loc+1 shift mean diff = {(p_sh - p_ref).mean():.3f} (want ~+1)")
        # 3) mini L0 config
        res = run_layer0(pipe, hook, X, starts, "mcar", 0.5, n_seeds=2)
        clean = run_clean_mse(pipe, X, starts)
        for c in "ABC":
            print(f"tiny L0 ETTh1 mcar:0.5 cell {c}: mse={res[c]['mse']:.4f} "
                  f"rel={res[c]['mse'] / clean['mse']:.4f}")
        # 4) mini l12 pass (2 windows weather)
        Xw, sw = s5.load_windows(s5.DATASETS["weather"], 2, s5.SEED)
        tmp = {}
        run_l12(pipe, Xw, sw, "mcar", "weather", tmp, lambda: None, batch_series=8)
        r = tmp["drift"]["weather"]["mcar"]["mcar:zero:0.7"]
        print(f"tiny l12 weather mcar:zero:0.7 cos embed={r['cos_mean'][0]:.4f} "
              f"layer11={r['cos_mean'][-1]:.4f} ent_l0={r['ent'][0]:.3f} "
              f"mreg_l0={r['mreg'][0]:.4f} mc_l0={r['mc'][0]:.4f}")
        print("TINY OK", flush=True)
        return

    if args.anchor_test:
        loaded = {}
        for ds in s5.DATASETS:
            X, starts = s5.load_windows(s5.DATASETS[ds], args.windows, s5.SEED)
            loaded[ds] = (X, starts)
        print("== anchor: cell A relMSE at p=0.7 (target: "
              + ", ".join(f"{m} {v}" for m, v in ANCHOR_A_P07.items()) + " +-5%) ==", flush=True)
        for mech in MECHS_ALL:
            rels = []
            for ds, (X, starts) in loaded.items():
                clean = run_clean_mse(pipe, X, starts)
                r = run_layer0(pipe, hook, X, starts, mech, 0.7, args.seeds)
                rel = r["A"]["mse"] / clean["mse"]
                rels.append(rel)
                print(f"  {ds:8s} {mech:14s} A rel={rel:8.3f}", flush=True)
            avg = sum(rels) / len(rels)
            ok = abs(avg / ANCHOR_A_P07[mech] - 1) <= 0.05
            print(f"  AVG {mech:14s} {avg:8.3f} vs {ANCHOR_A_P07[mech]:6.2f} "
                  f"-> {'OK' if ok else 'FAIL'}", flush=True)
        print("== anchor: cell B recovery at mcar p=0.5 "
              "(target 82.3/78.7/82.3%) ==", flush=True)
        for ds, (X, starts) in loaded.items():
            clean = run_clean_mse(pipe, X, starts)
            r = run_layer0(pipe, hook, X, starts, "mcar", 0.5, args.seeds)
            relA = r["A"]["mse"] / clean["mse"]
            relB = r["B"]["mse"] / clean["mse"]
            rec = (relA - relB) / (relA - 1.0)
            ok = abs(rec - ANCHOR_B_REC[ds]) <= 0.05
            print(f"  {ds:8s} relA={relA:8.3f} relB={relB:8.3f} rec={rec * 100:5.1f}% "
                  f"vs {ANCHOR_B_REC[ds] * 100:.1f}% -> {'OK' if ok else 'FAIL'}", flush=True)
        return

    out = args.out or os.path.join(HERE, f"s7_attrib_results_shard{args.shard}.json")
    results = {}
    if os.path.exists(out):
        results = json.load(open(out))
        print(f"resume: loaded {out}", flush=True)
    if args.rerun_l12:
        results.pop("drift", None)
        results.pop("attn", None)
    results.setdefault("meta", {
        "track": "S7 attribution (bolt only)",
        "seed": s5.SEED, "L": L, "H": H, "patch": PATCH, "n_patch": NPATCH,
        "reg_position": REG, "dist_edges": list(DIST_EDGES),
        "windows": args.windows, "seeds_mcar_block": args.seeds,
        "block2_stats": args.block2,
        "cells": {"A": "zero values + polluted stats (plain zero-fill)",
                  "B": "zero values x mean|x_clean|/mean|x_zero| rescale "
                       "(= S5 zero_oscale oracle probe; pure forecast rescale "
                       "by InstanceNorm scale-invariance)",
                  "C": "clean values + hook-forced zero-filled-context "
                       "InstanceNorm loc/scale",
                  "Bh": "zero values + hook-forced clean-context InstanceNorm "
                        "loc/scale (literal stats swap; diagnostic)",
                  "Cr": "clean values x mean|x_zero|/mean|x_clean| (scale-"
                        "factor symmetric counterpart of B; diagnostic)",
                  "D": "clean"},
        "anchors": {"A_p07_relMSE": ANCHOR_A_P07, "B_recovery_mcar_p05": ANCHOR_B_REC},
    })
    save = lambda: save_results(results, out)

    jobs = job_schedule(args.nshards)[args.shard]
    print(f"shard {args.shard}/{args.nshards}: {jobs}", flush=True)
    loaded = {}
    for ds, mech in jobs:
        if ds not in loaded:
            loaded[ds] = s5.load_windows(s5.DATASETS[ds], args.windows, s5.SEED)
        X, starts = loaded[ds]

        l0_ds = results.setdefault("layer0", {}).setdefault(ds, {})
        if "clean" not in l0_ds:
            t0 = time.time()
            l0_ds["clean"] = run_clean_mse(pipe, X, starts)
            l0_ds["clean"]["seconds"] = round(time.time() - t0, 1)
            save()
            print(f"[l0] {ds} clean mse={l0_ds['clean']['mse']:.4f}", flush=True)
        for p in P_L0:
            mkey = f"{mech}:{p}"
            if mkey not in l0_ds:
                t0 = time.time()
                l0_ds[mkey] = run_layer0(pipe, hook, X, starts, mech, p, args.seeds)
                rel = {c: l0_ds[mkey][c]["mse"] / l0_ds["clean"]["mse"] for c in CELLS}
                save()
                print(f"[l0] {ds:8s} {mkey:18s} relA={rel['A']:8.3f} "
                      f"relB={rel['B']:8.3f} relC={rel['C']:8.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)

        if not args.skip_l12 and mech in MECHS_L12:
            run_l12(pipe, X, starts, mech, ds, results, save,
                    batch_series=args.batch_enc)
    save()
    print("SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
