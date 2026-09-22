#!/usr/bin/env python
"""S13: attention knockout -- CAUSAL evidence for where the masked truth is
rebuilt inside the chronos-bolt-base encoder, and how much the forecast
actually relies on that rebuilding.

S11 (run_s11_probe.py) found with linear probes that the true values of
FULLY-missing patches become linearly decodable in the mid-encoder (R2 = 0 at
the embedding layer, rising over blocks 1-4, peaking at hidden-state layers
4-8, mild decay into the final LN). The attribution to ATTENTION was by
elimination (attention is the only cross-position channel in a transformer),
not by intervention. S13 supplies the intervention: knock out the attention
of fully-missing patch QUERIES layer-segment by layer-segment and watch (a)
the probe R2 collapse pattern (causal layer localization of the
reconstruction) and (b) the forecast relMSE degradation (causal contribution
of the reconstruction to prediction).

Knockout operation. bolt hard-masks fully-missing patches out of the KEY set
(chronos_bolt.py l.303) but keeps them as QUERIES. For every encoder block in
the knocked segment we wrap T5Attention.forward (runtime monkey-patch, same
replica style as s8.encode_replica / s11.extract_hidden -- no library edits)
and, right before the softmax, set to -inf every logit of a fully-missing
query row against all 32 patch keys. The only key left for such a query is
the appended [REG] token (position 32), so its attention mass is redirected
onto REG exactly (softmax over a single finite logit = 1.0, no NaN). Tensor
shapes are untouched; the returned position_bias is NOT contaminated, so the
knockout never leaks into unknocked layers. A "obs" target mode applies the
identical operation to fully-OBSERVED patch queries (sanity control).

Grid. Datasets {ETTh1, weather} x mechanisms {block p=0.3, block p=0.7, mcar
p=0.7} x knockout segments {none, l0-3, l4-7, l8-11, all}. Windows / masks /
seeds are inherited verbatim from run_s11_probe.py (= run_s5_missing.py):
300 test windows in the last-20% region (SEED), 300 probe-train windows from
the train region (SEED+11), L=512, H=96, patch=16, 2 mask seeds, bolt native
nan fill. Per config we measure:
  1. probe R2 -- the S11 pipeline (PCA(64) -> Ridge(alpha=1), pooled test R2,
     full_miss family, patch-mean target in clean-window z-space), with the
     PCA basis fit per (dataset, knockout segment) on that segment's own
     pooled train patches: each condition is measured on its own merits, so
     an R2 drop means information loss, not representation drift. We report
     hidden-state layers l6 and l12 (S11's readout convention: index 6 =
     block-5 output, 12 = final LN output) plus the full 13-layer curve.
  2. relMSE -- median-quantile forecast MSE under knockout / native clean MSE.

Pre-registered predictions (frozen BEFORE any S13 run; scored in s13_notes.md):
  P1: knocking l0-3 OR l4-7 collapses the full_miss probe R2 at l6 (>= 50%
      relative drop vs the no-knockout value) -- S11's curve rises over blocks
      1-4 and peaks at layers 4-8, so cutting transport in either half of the
      rise should remove most of the signal at l6.
  P2: knocking l8-11 leaves l6 R2 essentially unchanged (<= 10% relative drop;
      l6 sits upstream of the knocked layers) and reduces l12 R2 only
      partially (>= 50% retention) -- the residual stream carries whatever was
      rebuilt by layer 7 straight through the knocked layers; only NEW
      transport is cut.
  P3: knockout=all drives l6 R2 back to the embedding level (|R2| <= 0.05).
      If it does not, the hook is broken or a bypass exists -> investigate
      before reporting.
  P4 (revised at smoke time, BEFORE any grid run): relMSE under full-miss-
      query knockout is EXACTLY unchanged (0.0) at every segment. Reason found
      while building the smoke test: bolt's decode() cross-attends with
      encoder_attention_mask = the patch mask (chronos_bolt.py l.429-435),
      which excludes fully-missing positions, and gap tokens are never keys in
      the encoder -- so their rebuilt content is architecturally invisible to
      the forecast head. The naive expectation "relMSE degrades where R2
      collapses" is falsified by architecture; the grid run verifies the
      exact-zero claim at scale (any nonzero relMSE delta = a bypass to
      investigate). The obs-query sanity below is what proves the knockout
      machinery CAN move predictions.
  P5 (sanity): obs-query knockout (all layers) degrades relMSE clearly more
      than full-miss-query knockout -- proves the knockout itself bites.

Gates (in order):
  smoke   : 5 windows, block 0.3 -- with an empty knockout the patched
            attention forwards reproduce native encode bit-for-bit-ish
            (rel max|diff| < 2e-3, same bar as S11); under knockout=all the
            knocked queries put EXACTLY 0 mass on patch keys and 1.0 on REG
            at every layer; untouched patch positions are bitwise unaffected;
            knocked positions move; predictions stay bitwise identical (the
            decoder masks gap positions); no NaN anywhere.
  anchor  : no-knockout relMSE for the 6 grid cells vs s8_results.json exp1
            'correct' (+-5%) and no-knockout probe R2 (l6/l12) vs
            s11_results.json (+-5%; PCA pool here is the 3 S13 configs, S11's
            had 6, so exact bit-identity is NOT expected).

Subcommands: --smoke, --anchor, --dataset DS --segs a,b,c (worker; appends
RESULT json lines to its log), --merge (logs -> s13_results.json + gates),
--figure (s13.png). Logs: s13_smoke.log, s13_anchor.log, s13_{ds}.log.
"""
import argparse
import contextlib
import glob
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
import run_s8_masktoken as s8
import run_s11_probe as s11

L, H = s5.L, s5.H                      # 512, 96
PATCH = s8.PATCH                       # 16
NPATCH = s8.NPATCH                     # 32
REG = NPATCH                           # [REG] key position (appended last)
NLAYERS = s8.NLAYERS                   # 12 encoder blocks
NHID = s11.NHID                        # 13 hidden states
DS = ("ETTh1", "weather")
MECH_KEYS = ("block:0.3", "block:0.7", "mcar:0.7")
TRAIN_SEED = s11.TRAIN_SEED            # s5.SEED + 11
SEGS = {"none": [],
        "l0-3": [0, 1, 2, 3],
        "l4-7": [4, 5, 6, 7],
        "l8-11": [8, 9, 10, 11],
        "all": list(range(NLAYERS))}
RESULTS = os.path.join(HERE, "s13_results.json")
S8_JSON = os.path.join(HERE, "s8_results.json")
S11_JSON = os.path.join(HERE, "s11_results.json")


# ------------------------------------------------------- knockout machinery --

def make_ko_forward(layer_idx, ko, orig):
    """Replacement T5Attention.forward for the encoder self-attention of ONE
    block. Replicates the installed transformers forward (modeling_t5.py
    l.240-331, encoder path: no cross-attention, no cache) and inserts the
    S13 knockout right before the softmax. Falls back to the original forward
    whenever the knockout does not apply (layer not in segment, no batch
    qmask set, cross-attention / cache -- never hit on the bolt encoder)."""
    def fwd(self, hidden_states, mask=None, key_value_states=None,
            position_bias=None, past_key_values=None, output_attentions=False,
            **kwargs):
        qm = ko.qmask
        if (qm is None or layer_idx not in ko.layers
                or key_value_states is not None or past_key_values is not None):
            return orig(hidden_states, mask=mask,
                        key_value_states=key_value_states,
                        position_bias=position_bias,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions, **kwargs)
        B = hidden_states.shape[0]
        assert qm.shape[0] == B, f"qmask batch {qm.shape[0]} != hidden {B}"
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.key_value_proj_dim)
        query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        kv_shape = (*hidden_states.shape[:-1], -1, self.key_value_proj_dim)
        key_states = self.k(hidden_states).view(kv_shape).transpose(1, 2)
        value_states = self.v(hidden_states).view(kv_shape).transpose(1, 2)
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        if position_bias is None:
            key_length = key_states.shape[-2]
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, query_states.shape[1], input_shape[1], key_length),
                    device=scores.device, dtype=scores.dtype)
            else:
                position_bias = self.compute_bias(input_shape[1], key_length,
                                                  device=scores.device)
            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask
        scores = scores + position_bias
        # ---- S13 knockout: selected query rows keep ONLY the [REG] key ----
        np_ = qm.shape[1]                       # 32 patch positions
        kom = torch.zeros(B, 1, scores.shape[2], scores.shape[3],
                          dtype=torch.bool, device=scores.device)
        kom[:, :, :np_, :np_] = qm[:, None, :, None]
        scores = scores.masked_fill(kom, float("-inf"))
        attn_weights = torch.nn.functional.softmax(
            scores.float(), dim=-1).type_as(scores)
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o(attn_output)
        outputs = (attn_output, position_bias)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs
    return fwd


class Knockout:
    """Context manager: patch all 12 encoder T5Attention forwards + model.encode
    so that queries selected by `target` ("full_miss": cnt==0 observed points;
    "obs": cnt==16) are knocked out in the blocks listed in `layers`.
    `qmask` ([B, NPATCH] bool on device, None = inactive) is set per forward
    batch by encode_ko / extract_hidden and reset right after each encoder
    call, so a stale mask can never leak into an unrelated forward."""

    def __init__(self, model, layers, target):
        assert target in ("full_miss", "obs")
        self.model, self.layers, self.target = model, set(layers), target
        self.qmask = None
        self._patched = []

    def set_batch(self, pm):
        """pm [B, NPATCH, PATCH] float (1 = observed), bolt's patched_mask."""
        cnt = pm.sum(dim=-1)
        qm = (cnt == 0) if self.target == "full_miss" else (cnt == pm.shape[-1])
        self.qmask = qm if bool(qm.any()) else None

    def __enter__(self):
        for l, block in enumerate(self.model.encoder.block):
            attn = block.layer[0].SelfAttention
            orig = attn.forward
            attn.forward = types.MethodType(make_ko_forward(l, self, orig), attn)
            self._patched.append(attn)

        def encode_fn(context, mask=None):
            return encode_ko(self.model, context, self)

        self.model.encode = encode_fn
        return self

    def __exit__(self, *exc):
        for attn in self._patched:
            del attn.forward          # drop instance attr -> class method again
        self._patched = []
        del self.model.encode
        self.qmask = None


def encode_ko(model, xb, ko):
    """Line-for-line replica of ChronosBoltModelForForecasting.encode()
    (= s8.encode_replica 'correct'; verified there against the installed
    chronos_bolt.py) with ko.set_batch(patched_mask) injected right before the
    encoder call. Returns the native 4-tuple."""
    mask = torch.isnan(xb).logical_not().to(xb.dtype)                   # l.280
    B = xb.shape[0]
    if xb.shape[-1] > model.chronos_config.context_length:              # l.283-285
        xb = xb[..., -model.chronos_config.context_length:]
        mask = mask[..., -model.chronos_config.context_length:]
    xc, loc_scale = model.instance_norm(xb)                             # l.288
    xc = xc.to(model.dtype)                                             # l.292
    mask = mask.to(model.dtype)                                         # l.293
    pc = model.patch(xc)                                                # l.296
    pm = torch.nan_to_num(model.patch(mask), nan=0.0)                   # l.297
    pc = torch.where(pm > 0.0, pc, 0.0)                                 # l.298
    pc_in = torch.cat([pc, pm], dim=-1)                                 # l.300
    am = pm.sum(dim=-1) > 0                                             # l.303
    emb = model.input_patch_embedding(pc_in)                            # l.305
    reg_ids = torch.full((B, 1), model.config.reg_token_id, device=xb.device)
    emb = torch.cat([emb, model.shared(reg_ids)], dim=-2)               # l.307-315
    am = torch.cat([am.to(model.dtype),
                    torch.ones_like(reg_ids).to(model.dtype)], dim=-1)  # l.316-321
    ko.set_batch(pm)
    try:
        enc = model.encoder(attention_mask=am, inputs_embeds=emb)       # l.324-327
    finally:
        ko.qmask = None
    return enc.last_hidden_state, loc_scale, emb, am


@torch.no_grad()
def extract_hidden(pipe, x_in, ko=None, batch=512, need_attn=False):
    """s11.extract_hidden (identical preprocessing) + optional knockout.
    Returns (layer list of [N, NPATCH, d] fp16, encoder output or None)."""
    model = pipe.model
    d = model.config.d_model
    N = len(x_in)
    out = [np.empty((N, NPATCH, d), np.float16) for _ in range(NHID)]
    last_enc = None
    for i in range(0, N, batch):
        xb = torch.from_numpy(x_in[i:i + batch]).to(model.device)
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
        am = torch.cat([am.to(model.dtype),
                        torch.ones_like(reg_ids).to(model.dtype)], dim=-1)
        if ko is not None:
            ko.set_batch(pm)
        try:
            enc = model.encoder(attention_mask=am, inputs_embeds=emb,
                                output_hidden_states=True,
                                output_attentions=need_attn)
        finally:
            if ko is not None:
                ko.qmask = None
        assert len(enc.hidden_states) == NHID
        for l, hs in enumerate(enc.hidden_states):
            out[l][i:i + B] = hs[:, :NPATCH, :].float().cpu().numpy().astype(np.float16)
        if need_attn and i + batch >= N:
            last_enc = enc
    return (out, last_enc) if need_attn else out


# ------------------------------------------------------------------ smoke ----

def run_smoke(args):
    pipe = s8.load_bolt(args.device)
    model = pipe.model
    X, starts = s5.load_windows(s5.DATASETS["ETTh1"], 5, s5.SEED)
    ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, "block", 0.3)
    x_nan = ctx.copy()
    x_nan[msk] = np.nan

    # 1) empty knockout must reproduce native encode exactly (S11's bar: 2e-3)
    with Knockout(model, [], "full_miss") as ko:
        reps_ko, _ = extract_hidden(pipe, x_nan, ko=ko, batch=70, need_attn=True)
    xb = torch.from_numpy(x_nan[:70]).to(model.device)
    with torch.no_grad():
        ref = model.encode(xb)[0][:, :NPATCH].float().cpu().numpy()
    got = reps_ko[-1][:70].astype(np.float32)
    rel = np.abs(ref - got).max() / max(np.abs(ref).max(), 1e-9)
    print(f"smoke: empty-knockout vs native rel max|diff| = {rel:.2e} "
          f"(want <2e-3)", flush=True)
    assert rel < 2e-3, "patched forwards distort the native path"
    assert not np.isnan(got).any(), "NaN in empty-knockout hidden states"

    # 1b) empty-knockout PREDICT path (exercises encode_ko) == native predict
    p_native = s8.predict_median(pipe, x_nan, batch=70)
    with Knockout(model, [], "full_miss"):
        p_empty = s8.predict_median(pipe, x_nan, batch=70)
    dp = np.abs(p_native - p_empty).max()
    print(f"smoke: empty-knockout predict max|diff| = {dp:.2e} (want <1e-4)",
          flush=True)
    assert dp < 1e-4, "encode_ko replica mismatch"

    # 2) knockout=all: attention exactly 0 on patch keys / 1.0 on REG, and
    #    rows without fully-missing patches bitwise unaffected
    with Knockout(model, SEGS["all"], "full_miss") as ko:
        reps_all, enc = extract_hidden(pipe, x_nan, ko=ko, batch=70,
                                       need_attn=True)
    assert enc is not None and enc.attentions is not None
    pm_t = torch.from_numpy(
        (~msk.reshape(len(msk), NPATCH, PATCH)).astype(np.float32)).to(model.device)
    qm = (pm_t.sum(-1) == 0)                                # [B, 32] bool
    nq = int(qm.sum())
    assert nq > 100, "smoke batch should contain many fully-missing patches"
    for l, a in enumerate(enc.attentions):                  # [B, nh, 33, 33]
        nh = a.shape[1]
        qm4 = qm[:, None, :, None].expand(-1, nh, -1, NPATCH)
        w_patch = a[:, :, :NPATCH, :NPATCH][qm4]
        w_reg = a[:, :, :NPATCH, REG].masked_select(
            qm[:, None, :].expand(-1, nh, -1))
        mx = float(w_patch.abs().max()) if w_patch.numel() else 0.0
        mn = float(w_reg.min()) if w_reg.numel() else 1.0
        assert mx == 0.0, f"layer {l}: knocked query keeps {mx} mass on patch keys"
        assert abs(mn - 1.0) < 1e-6, f"layer {l}: REG mass {mn} != 1"
    print(f"smoke: knockout=all -> patch-key mass exactly 0, REG mass 1.0 "
          f"at all 12 layers ({nq} knocked queries)", flush=True)
    assert nq > 100, "smoke batch should contain many fully-missing patches"
    for l in range(NHID):
        assert not np.isnan(reps_all[l]).any(), f"NaN in hidden layer {l}"
    # fully-missing patches are never KEYS (native hard mask), so every
    # non-full-miss patch position must be bitwise unaffected by the knockout
    # (compare fp16-vs-fp16 against the empty-knockout extraction of step 1)
    sel = ~qm.cpu().numpy()                                 # [70, 32] bool
    d = np.abs(reps_all[-1][sel].astype(np.float32)
               - reps_ko[-1][sel].astype(np.float32)).max()
    print(f"smoke: untouched patch positions max|diff| vs native = "
          f"{d:.2e} (want <1e-5; n={int(sel.sum())})", flush=True)
    assert d < 1e-5, "knockout leaked into untouched positions"

    # 3) segment selectivity: l0-3 knocked, l4+ NOT forced to zero
    with Knockout(model, SEGS["l0-3"], "full_miss") as ko:
        _, enc4 = extract_hidden(pipe, x_nan, ko=ko, batch=70, need_attn=True)
    w0 = enc4.attentions[0][:, :, :NPATCH, :NPATCH][
        qm[:, :, None].expand(-1, -1, NPATCH).unsqueeze(1).expand(
            -1, enc4.attentions[0].shape[1], -1, -1)]
    w4 = enc4.attentions[4][:, :, :NPATCH, :NPATCH][
        qm[:, :, None].expand(-1, -1, NPATCH).unsqueeze(1).expand(
            -1, enc4.attentions[4].shape[1], -1, -1)]
    print(f"smoke: seg l0-3 -> layer0 patch-mass {float(w0.max()):.1e} "
          f"(want 0), layer4 patch-mass {float(w4.max()):.3f} (want >0)",
          flush=True)
    assert float(w0.max()) == 0.0 and float(w4.max()) > 0.0

    # 4) hidden states AT knocked positions must change (the knockout bites the
    #    representation) ...
    qm_np = qm.cpu().numpy()
    dk = np.abs(reps_all[-1][qm_np].astype(np.float32)
                - reps_ko[-1][qm_np].astype(np.float32)).max()
    print(f"smoke: knocked positions max|diff| vs empty-ko = {dk:.4f} "
          f"(want >>0; n={int(qm_np.sum())})", flush=True)
    assert dk > 1e-2, "knockout did not change the knocked tokens"
    # ... while PREDICTIONS must be bitwise identical: bolt's decode()
    # (chronos_bolt.py l.429-435) cross-attends with encoder_attention_mask =
    # the patch mask, which EXCLUDES fully-missing positions -- the forecast
    # head never reads gap tokens, and gap tokens are never keys in the
    # encoder, so their rebuilt content cannot reach the forecast at all.
    with Knockout(model, SEGS["all"], "full_miss"):
        p_ko = s8.predict_median(pipe, x_nan, batch=70)
    assert not np.isnan(p_ko).any(), "NaN in knocked predictions"
    md = np.abs(p_native - p_ko).max()
    print(f"smoke: predict max|ko-native| = {md:.2e} (want exactly 0: the "
          f"decoder masks fully-missing positions)", flush=True)
    assert md == 0.0
    print("SMOKE OK", flush=True)


# ----------------------------------------------------------------- anchor ----

def run_anchor(args):
    """No-knockout relMSE for the 6 grid cells vs s8_results.json (+-5%) and
    native clean MSE per dataset (relMSE denominator for everything)."""
    pipe = s8.load_bolt(args.device)
    r8 = json.load(open(S8_JSON))
    out = {}
    if os.path.exists(args.out):
        out = json.load(open(args.out))
    out.setdefault("meta", {
        "track": "S13 attention knockout (bolt encoder), full_miss queries -> [REG]",
        "model": "amazon/chronos-bolt-base (fp32)", "seed": s5.SEED,
        "train_seed": TRAIN_SEED, "L": L, "H": H, "patch": PATCH,
        "n_patch": NPATCH, "segments": {k: v for k, v in SEGS.items()},
        "datasets": DS, "mechanisms": MECH_KEYS,
        "windows_test": args.windows, "windows_train": args.train_windows,
        "probe": "S11 pipeline: PCA(64)+Ridge(1.0), per-(ds,segment) PCA basis",
        "targets": "full_miss patch mean in clean-window z-space",
        "readout_layers": {"l6": 6, "l12": 12},
    })
    save = lambda: s5.save_results(out, args.out)
    anch = out.setdefault("anchor", {})
    gate = True
    for ds in DS:
        X, starts = s5.load_windows(s5.DATASETS[ds], args.windows, s5.SEED)
        if ds not in out.setdefault("clean", {}):
            t0 = time.time()
            out["clean"][ds] = s8.clean_baseline(pipe, X, starts)
            save()
            print(f"anchor {ds:8s} clean mse={out['clean'][ds]['mse']:.4f} "
                  f"(s8: {r8['clean'][ds]['mse']:.4f}) ({time.time()-t0:.0f}s)",
                  flush=True)
        cl = out["clean"][ds]["mse"]
        for key in MECH_KEYS:
            ak = f"{ds}:{key}"
            if ak in anch:
                continue
            mech, p = key.split(":")
            ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, mech, float(p))
            x_nan = ctx.copy()
            x_nan[msk] = np.nan
            t0 = time.time()
            pred = s8.predict_median(pipe, x_nan)
            mse, _ = s8.mse_windowed(pred, gt, S, C, nw)
            rel = mse / cl
            tgt = r8["exp1"][ds][f"{mech}:{p}:correct"]["mse"] / r8["clean"][ds]["mse"]
            ok = abs(rel / tgt - 1) <= 0.05
            gate &= ok
            anch[ak] = {"rel": rel, "s8_rel": tgt, "ok": ok}
            save()
            print(f"anchor {ds:8s} {key:9s} nan rel={rel:.4f} vs s8 "
                  f"{tgt:.4f} -> {'OK' if ok else 'FAIL'} ({time.time()-t0:.0f}s)",
                  flush=True)
    anch["gate"] = "PASS" if gate else "FAIL"
    save()
    print(f"ANCHOR {anch['gate']}", flush=True)
    if not gate:
        raise SystemExit(2)


# ------------------------------------------------------------- grid worker ---

def check_anchor_gate(args):
    if os.path.exists(args.out):
        out = json.load(open(args.out))
        if out.get("anchor", {}).get("gate") == "PASS":
            return out
    raise SystemExit("anchor gate not passed -- run --anchor first and verify")


def emit(rec):
    print("RESULT " + json.dumps(rec), flush=True)


def done_keys(logpaths):
    """(kind, ds, seg) keys already present as RESULT lines in the logs."""
    done = set()
    for lp in logpaths:
        if not os.path.exists(lp):
            continue
        with open(lp) as fh:
            for line in fh:
                if line.startswith("RESULT "):
                    try:
                        r = json.loads(line[7:])
                        done.add((r["kind"], r["ds"], r.get("seg")))
                    except Exception:
                        pass
    return done


def run_worker(args):
    out = check_anchor_gate(args)
    ds = args.dataset
    segs = args.segs.split(",")
    for sg in segs:
        assert sg in SEGS, sg
    pipe = s8.load_bolt(args.device)
    clean_mse = out["clean"][ds]["mse"]
    X, starts_te = s5.load_windows(s5.DATASETS[ds], args.windows, s5.SEED)
    _, starts_tr = s11.load_train_windows(s5.DATASETS[ds], args.train_windows,
                                          TRAIN_SEED)
    done = done_keys(args.logs)
    print(f"[{ds}] segs={segs}: {len(starts_tr)} train / {len(starts_te)} "
          f"test windows, C={X.shape[1]}, resume skips={sorted(done)}",
          flush=True)

    for seg in segs:
        if ("segment", ds, seg) in done:
            print(f"[{ds}] {seg}: already in log, skip", flush=True)
            continue
        t_seg = time.time()
        layers = SEGS[seg]
        rec = {"kind": "segment", "ds": ds, "seg": seg, "layers": layers,
               "probe": {}, "relmse": {}}
        ko = None if not layers else Knockout(pipe.model, layers, "full_miss")
        ctxman = ko if ko is not None else contextlib.nullcontext()
        with ctxman:
            # ---- train-side extraction + per-segment PCA basis ----
            pdata_tr, ztr = {}, {}
            reps_tr = {}
            for key in MECH_KEYS:
                mech, p = key.split(":")
                ctx, msk, _, _, _, _, _ = s8.build_contexts(X, starts_tr, mech,
                                                            float(p))
                pd_ = s11.make_probe_data(ctx, msk)
                pdata_tr[key] = pd_
                t0 = time.time()
                reps_tr[key] = extract_hidden(pipe, pd_["x_nan"], ko=ko)
                print(f"[{ds}] {seg:5s} {key:9s} train: {len(ctx)} series "
                      f"({time.time()-t0:.0f}s)", flush=True)
            pcas = s11.fit_pcas([reps_tr[k] for k in MECH_KEYS], f"{ds}/{seg}")
            for key in MECH_KEYS:
                ztr[key] = s11.transform_reps(pcas, reps_tr.pop(key))
            # ---- test-side: extract -> transform -> probe -> relMSE ----
            for key in MECH_KEYS:
                mech, p = key.split(":")
                ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts_te,
                                                               mech, float(p))
                pd_te = s11.make_probe_data(ctx, msk)
                t0 = time.time()
                reps = extract_hidden(pipe, pd_te["x_nan"], ko=ko)
                zte = s11.transform_reps(pcas, reps)
                del reps
                r2, ntr, nte = s11.family_probe(
                    ztr[key], pdata_tr[key]["y_mean"],
                    pdata_tr[key]["fam"]["full_miss"],
                    zte, pd_te["y_mean"], pd_te["fam"]["full_miss"])
                del zte
                rec["probe"][key] = {"r2_mean": r2, "n_train": ntr,
                                     "n_test": nte,
                                     "achieved_rate": float(msk.mean())}
                msg = "n/a" if r2[6] is None else \
                    f"l0={r2[0]:.3f} l6={r2[6]:.3f} l12={r2[12]:.3f}"
                print(f"[{ds}] {seg:5s} {key:9s} probe: full_miss {msg} "
                      f"(n_te={nte}) ({time.time()-t0:.0f}s)", flush=True)
                if layers:      # none-segment relMSE comes from the anchor
                    t0 = time.time()
                    x_nan = ctx.copy()
                    x_nan[msk] = np.nan
                    pred = s8.predict_median(pipe, x_nan)
                    mse, _ = s8.mse_windowed(pred, gt, S, C, nw)
                    rec["relmse"][key] = mse / clean_mse
                    print(f"[{ds}] {seg:5s} {key:9s} relMSE={rec['relmse'][key]:.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            del ztr
        rec["seconds"] = round(time.time() - t_seg, 1)
        emit(rec)

    # ---- sanity control: fully-OBSERVED queries knocked out (with 'all') ----
    if "all" in segs and ("sanity_obs", ds, None) not in done:
        rec = {"kind": "sanity_obs", "ds": ds, "relmse_obs_ko": {}}
        for key in ("block:0.3", "mcar:0.7"):
            mech, p = key.split(":")
            ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts_te, mech,
                                                           float(p))
            x_nan = ctx.copy()
            x_nan[msk] = np.nan
            t0 = time.time()
            with Knockout(pipe.model, SEGS["all"], "obs"):
                pred = s8.predict_median(pipe, x_nan)
            mse, _ = s8.mse_windowed(pred, gt, S, C, nw)
            rec["relmse_obs_ko"][key] = mse / clean_mse
            print(f"[{ds}] sanity obs-ko {key:9s} relMSE="
                  f"{rec['relmse_obs_ko'][key]:.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
        emit(rec)
    print(f"[{ds}] WORKER DONE", flush=True)


# ------------------------------------------------------------------ merge ----

def run_merge(args):
    out = {}
    if os.path.exists(args.out):
        out = json.load(open(args.out))
    if out.get("anchor", {}).get("gate") != "PASS":
        raise SystemExit("anchor gate not passed")
    logs = sorted(glob.glob(os.path.join(HERE, "s13_*.log")))
    n_new = 0
    for lp in logs:
        with open(lp) as fh:
            for line in fh:
                if not line.startswith("RESULT "):
                    continue
                r = json.loads(line[7:])
                if r["kind"] == "segment":
                    sec = out.setdefault("probe_ko", {}).setdefault(
                        r["ds"], {}).setdefault(r["seg"], {})
                    if r["seg"] not in out.setdefault("probe_ko", {}).get(
                            r["ds"], {}) or sec != r["probe"]:
                        n_new += 1
                    out["probe_ko"][r["ds"]][r["seg"]] = r["probe"]
                    if r["relmse"]:
                        out.setdefault("relmse_ko", {}).setdefault(
                            r["ds"], {})[r["seg"]] = r["relmse"]
                elif r["kind"] == "sanity_obs":
                    out.setdefault("sanity_obs", {})[r["ds"]] = r["relmse_obs_ko"]
                    n_new += 1
    # none-segment relMSE = anchor values
    for ds in DS:
        if f"{ds}:block:0.3" in out["anchor"]:
            out.setdefault("relmse_ko", {}).setdefault(ds, {})["none"] = {
                key: out["anchor"][f"{ds}:{key}"]["rel"] for key in MECH_KEYS}

    # ---- R2 anchor gate: none-segment vs S11 (+-5% on l6/l12) ----
    r11 = json.load(open(S11_JSON))
    gate = True
    detail = {}
    for ds in DS:
        probe = out.get("probe_ko", {}).get(ds, {}).get("none", {})
        for key in MECH_KEYS:
            if key not in probe:
                continue
            for li, lname in ((6, "l6"), (12, "l12")):
                got = probe[key]["r2_mean"][li]
                ref = r11["probe"][ds][key]["families"]["full_miss"]["r2_mean"][li]
                if got is None or ref is None:
                    continue
                ok = abs(got / ref - 1) <= 0.05
                gate &= ok
                detail[f"{ds}:{key}:{lname}"] = {"s13": got, "s11": ref, "ok": ok}
    out["r2_anchor"] = {"gate": "PASS" if gate else "FAIL", "detail": detail}
    s5.save_results(out, args.out)
    print(f"merged ({n_new} new records) -> {args.out}", flush=True)
    print(f"R2 anchor gate vs S11: {out['r2_anchor']['gate']}", flush=True)
    for k, v in detail.items():
        print(f"  {k:28s} s13={v['s13']:.4f} s11={v['s11']:.4f} "
              f"{'OK' if v['ok'] else 'FAIL'}", flush=True)

    # ---- summary tables ----
    for ds in DS:
        pk = out.get("probe_ko", {}).get(ds, {})
        rk = out.get("relmse_ko", {}).get(ds, {})
        if not pk:
            continue
        print(f"\n== {ds}: full_miss probe R2 (l6 / l12) by knockout segment ==",
              flush=True)
        for key in MECH_KEYS:
            row = []
            for seg in SEGS:
                r2 = pk.get(seg, {}).get(key, {}).get("r2_mean")
                row.append("   n/a" if not r2 or r2[6] is None else
                           f"{r2[6]:.3f}/{r2[12]:.3f}")
            print(f"  {key:9s} " + "  ".join(f"{s}:{v}" for s, v in
                                             zip(SEGS, row)), flush=True)
        print(f"== {ds}: relMSE by knockout segment ==", flush=True)
        for key in MECH_KEYS:
            row = [f"{rk.get(seg, {}).get(key, float('nan')):.4f}"
                   for seg in SEGS]
            print(f"  {key:9s} " + "  ".join(f"{s}:{v}" for s, v in
                                             zip(SEGS, row)), flush=True)
        so = out.get("sanity_obs", {}).get(ds)
        if so:
            print(f"== {ds}: sanity obs-query knockout (all) relMSE == "
                  + json.dumps(so), flush=True)


# ----------------------------------------------------------------- figure ----

def make_figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r11 = json.load(open(S11_JSON))
    layers = list(range(NHID))
    seg_style = {"none": dict(color="k", lw=2.0, label="no knockout"),
                 "l0-3": dict(color="tab:blue", label="knock l0-3"),
                 "l4-7": dict(color="tab:orange", label="knock l4-7"),
                 "l8-11": dict(color="tab:green", label="knock l8-11"),
                 "all": dict(color="tab:red", label="knock all")}
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for row, ds in enumerate(DS):
        for col, key in enumerate(MECH_KEYS):
            ax = axes[row][col]
            ref = r11["probe"][ds][key]["families"]["full_miss"]["r2_mean"]
            ax.plot(layers, [np.nan if v is None else v for v in ref],
                    color="grey", ls="--", lw=1, alpha=0.7,
                    label="S11 no-knockout ref")
            for seg in SEGS:
                cfg = res.get("probe_ko", {}).get(ds, {}).get(seg, {}).get(key)
                if not cfg:
                    continue
                r2 = [np.nan if v is None else v for v in cfg["r2_mean"]]
                ax.plot(layers, r2, marker="o", ms=3, **seg_style[seg])
            ax.axhline(0.0, color="k", lw=0.5)
            for ls_, c in ((0.5, "tab:blue"), (5.5, "tab:orange"),
                           (8.5, "tab:green")):
                ax.axvline(ls_ - 0.5, color=c, lw=0.5, alpha=0.4)
            ax.set_title(f"{ds} — {key}", fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.1, 1.0)
            if row == 1:
                ax.set_xlabel("encoder layer (0 = embedding, 12 = final LN)")
            if col == 0:
                ax.set_ylabel("full-miss patch probe R² (test)")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="lower left")
    # row 3: relMSE bars per segment (2 ds) + sanity panel
    width = 0.25
    for col, ds in enumerate(DS):
        ax = axes[2][col]
        rk = res.get("relmse_ko", {}).get(ds, {})
        xs = np.arange(len(SEGS))
        for mi, key in enumerate(MECH_KEYS):
            vals = [rk.get(seg, {}).get(key, np.nan) for seg in SEGS]
            ax.bar(xs + (mi - 1) * width, vals, width, label=key)
        so = res.get("sanity_obs", {}).get(ds, {})
        for mi, key in enumerate(("block:0.3", "mcar:0.7")):
            if key in so:
                ax.bar(len(SEGS) + (mi - 1.5) * width, so[key], width,
                       color="purple", alpha=0.4 if mi else 0.8,
                       label=f"OBS-query knock-all {key}")
        ax.set_xticks(list(xs) + [len(SEGS) - 0.5 + width])
        ax.set_xticklabels(list(SEGS) + ["obs-ko\n(sanity)"], fontsize=8)
        ax.axhline(1.0, color="k", lw=0.5, ls=":")
        ax.set_title(f"{ds} — forecast relMSE under knockout", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylabel("relMSE vs clean")
        ax.legend(fontsize=7)
    ax = axes[2][2]
    for ds, ls_ in zip(DS, ("-", "--")):
        for key, m in zip(MECH_KEYS, ("o", "s", "^")):
            vals = []
            for seg in SEGS:
                cfg = res.get("probe_ko", {}).get(ds, {}).get(seg, {}).get(key)
                vals.append(np.nan if not cfg or cfg["r2_mean"][6] is None
                            else cfg["r2_mean"][6])
            ax.plot(list(SEGS), vals, marker=m, ls=ls_, label=f"{ds} {key}")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_title("l6 probe R² vs knockout segment", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylabel("full-miss patch probe R² at l6")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=7)
    fig.suptitle("S13 attention knockout on chronos-bolt-base: causal "
                 "localization of masked-truth reconstruction", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}", flush=True)


# ------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--dataset", choices=list(DS), default=None)
    ap.add_argument("--segs", default=",".join(SEGS))
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--train-windows", type=int, default=300)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--fig-out", default=os.path.join(HERE, "s13.png"))
    ap.add_argument("--logs", nargs="*",
                    default=[os.path.join(HERE, f"s13_{ds}.log") for ds in DS])
    args = ap.parse_args()

    torch.manual_seed(s5.SEED)
    np.random.seed(s5.SEED)

    if args.smoke:
        run_smoke(args)
        return
    if args.anchor:
        run_anchor(args)
        return
    if args.merge:
        run_merge(args)
        return
    if args.figure:
        res = json.load(open(args.out))
        make_figure(res, args.fig_out)
        return
    assert args.dataset, "nothing to do"
    run_worker(args)


if __name__ == "__main__":
    main()
