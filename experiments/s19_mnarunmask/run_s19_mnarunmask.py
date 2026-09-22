#!/usr/bin/env python
"""S19: mechanism of the S15 mnar_high unmask gain -- appendix-level causal
exploration. NOT a new method.

Background. S15 (run_s15_unmask.py, s15_results.json) opened bolt's decoder
cross-attention mask on fully-missing patch positions. On the pre-registered
mcar/block grid this bought exactly nothing (Outcome B). But on the negative
control mnar_high p=0.7 it found an unexpected REAL gain on ETT (paired
per-window sft:unmask vs sft:mb: ETTh1 +5.67% t=-15.5 win 0.98, ETTm1 +6.91%
t=-7.6 win 0.82; weather flat +0.24%; survives own-clean normalization).
The cells remain ~4x degraded vs clean, so this is not "repair". Working
hypothesis (the "phase scaffold" story): the rebuilt gap content -- a smooth,
position-aware interpolation across the censored peak regions -- gives the
forecast head a better phase/continuity prior than isolated low-value
islands, reducing SECONDARY damage (level/phase misalignment), not restoring
censored peaks. S19 tests that mechanism four ways, all on mnar_high p=0.7
x {ETTh1, ETTm1} with weather as flat control:

1. Causal ablation: sft:unmask + S13 full_miss knockout (all 12 layers; gap
   queries redirected to [REG], so the rebuilt content never forms) -- if the
   gain vanishes, it comes from the rebuilt CONTENT, not from an architectural
   side-effect of opening the mask (softmax mass redistribution over observed
   keys survives the knockout, so a side-effect explanation predicts the gain
   is KEPT under knockout).
2. Error anatomy (sft:unmask vs sft:mb, paired per window): horizon quarters
   (phase prior predicts early steps gain more), level-vs-shape decomposition
   (mse == (mu_p-mu_g)^2 + centered-mse exactly), dose-response by window
   censoring severity (n fully-missing patches = the amount of scaffold the
   head can read).
3. Top-decile split: gain inside vs outside windows/channels whose FUTURE
   contains a high peak. Gain concentrated in flat-future pairs supports the
   secondary-damage story; gain in peak-future pairs would force a rethink.
4. SFT necessity: zs:unmask vs zs:masked on the same cell (S15: flat) plus
   zs:unmask+knockout -- the gain must be an unmask x SFT interaction.

Implementation: runtime composition of S15's UnmaskDecode (decoder side) and
S13's Knockout (encoder side, full_miss target, all layers) on the raw
ChronosBoltModelForForecasting; no library edits. Checkpoints loaded from
s15_ckpt/ (s15_mb_anchor_*.pt == s12_ckpt/s12_mb_*.pt bit-identical, proven in
S15; s15_unmask_*.pt). Windows/masks/fill exactly as S15: 150 test windows
(s5.SEED), deterministic rank-based mnar_high mask (1 seed), nan fill, median
quantile, H=96.

Conditions (7): zs_masked, zs_unmask, zs_unmask_ko, mb (= s15 mb_anchor),
mb_ko (control: knockout must be EXACTLY inert under the native mask -- S13
P4), unmask, unmask_ko (the ablation).

Pre-registered predictions (frozen 2026-08-13, before any S19 eval; scored in
s19_notes.md):
  P1 (content causality): unmask_ko retains <= 20% of the unmask-vs-mb gain
     on ETTh1/ETTm1 (gain_retention = rel_gain(unmask_ko vs mb) /
     rel_gain(unmask vs mb)). Retention >= 80% => the gain is an architectural
     side-effect and the scaffold story is dead.
  P2 (phase prior): horizon-quarter gains are front-loaded: rel_gain(q1) >
     rel_gain(q4) on ETT. A flat profile would fit a pure level shift better.
  P3 (secondary damage, not peak recovery): the gain is NOT concentrated in
     future-peak pairs (top-decile fut_peak gain <= rest gain), and a visible
     share comes from centered/shape error, not only the window mean.
  P4 (dose response): gain grows with the number of fully-missing patches
     (tercile bins monotone on ETT).
  P5 (controls): weather |gain| < 1% for every paired comparison; mb_ko == mb
     bitwise (max per-window diff exactly 0.0).

Anchor gate: the 4 non-KO conditions re-evaluated through THIS harness must
match s15_results.json neg cells (mnar_high p=0.7, all 3 datasets) within +-5%
(expected bit-exact: same code path, deterministic masks). Gate runs after
the base-condition eval and BEFORE merge/analysis; merge refuses to run on a
failed gate.

Subcommands:
  --smoke                       KO x unmask composition checks (GPU)
  --eval --jobs {anatomy,ablation}   worker; appends RESULT lines to its log
  --anchor                      gate vs s15_results.json -> s19_results.json
  --merge                       logs -> cells + analyses -> s19_results.json
  --summary
  --figure                      s19.png

Logs: s19_smoke.log, s19_anatomy.log (4 base conds x 3 ds), s19_ablation.log
(3 KO conds x 3 ds), s19_anchor.log, s19_merge.log. Artifacts: this file,
s19_results.json, s19.png, s19_notes.md. Nothing pre-existing is written.
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
import run_s13_knockout as s13
import run_s15_unmask as s15

L, H = s5.L, s5.H                       # 512, 96
PATCH, NPATCH = s8.PATCH, s8.NPATCH     # 16, 32
REG = s13.REG                           # 32
NLAYERS = s8.NLAYERS                    # 12
DS = ("ETTh1", "ETTm1", "weather")
MECH, RATE = "mnar_high", 0.7
BASE_CONDS = ("zs_masked", "zs_unmask", "mb", "unmask")
KO_CONDS = ("zs_unmask_ko", "mb_ko", "unmask_ko")
CONDS = BASE_CONDS + KO_CONDS
BIAS_CONDS = ("zs_masked", "mb", "unmask", "unmask_ko")   # signed-bias eval
# cond -> (s15 ModelBank condition, apply S13 full_miss knockout?)
COND_MAP = {"zs_masked": ("zs_masked", False),
            "zs_unmask": ("zs_unmask", False),
            "zs_unmask_ko": ("zs_unmask", True),
            "mb": ("mb_anchor", False),
            "mb_ko": ("mb_anchor", True),
            "unmask": ("unmask", False),
            "unmask_ko": ("unmask", True)}
S15_CELL = {"zs_masked": "mnar_high:zs_masked:0.7",
            "zs_unmask": "mnar_high:zs_unmask:0.7",
            "mb": "mnar_high:mb_anchor:0.7",
            "unmask": "mnar_high:unmask:0.7"}
RESULTS = os.path.join(HERE, "s19_results.json")
S15_JSON = os.path.join(HERE, "s15_results.json")
ANCHOR_TOL = 0.05                       # pre-registered; expect bit-exact
LOGS = {"anatomy": os.path.join(HERE, "s19_anatomy.log"),
        "ablation": os.path.join(HERE, "s19_ablation.log"),
        "bias": os.path.join(HERE, "s19_bias.log")}


# ------------------------------------------------------------ small utils ----

def emit(rec):
    print("RESULT " + json.dumps(rec), flush=True)


def scan_logs(paths):
    """RESULT json lines -> {(ds, cond): rec}, {ds: maskstats}, bias cells."""
    cells, mstats, bias = {}, {}, {}
    for lp in paths:
        if not os.path.exists(lp):
            continue
        with open(lp) as fh:
            for line in fh:
                if not line.startswith("RESULT "):
                    continue
                try:
                    r = json.loads(line[7:])
                except Exception:
                    continue
                if r.get("kind") == "cell":
                    cells[(r["ds"], r["cond"])] = r
                elif r.get("kind") == "maskstats":
                    mstats[r["ds"]] = r
                elif r.get("kind") == "bias":
                    bias[(r["ds"], r["cond"])] = r
    return cells, mstats, bias


def pgain(a_mse, b_mse):
    """Paired gain of a over b on identical windows/masks (positive = a
    better). t > 0 favors a; rel_gain normalized by mean(b)."""
    a = np.asarray(a_mse, np.float64)
    b = np.asarray(b_mse, np.float64)
    d = b - a
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    t = m / (sd / np.sqrt(n)) if sd > 0 else 0.0
    return {"n": n, "gain": m, "mean_ref": float(b.mean()),
            "rel_gain": m / float(b.mean()), "t": t,
            "ci95": 1.96 * sd / np.sqrt(n), "win_rate": float((d > 0).mean())}


def _rankdata(a):
    """Average-tie ranks (0..n-1), no scipy dependency."""
    a = np.asarray(a, np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), np.float64)
    ranks[order] = np.arange(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    rho = float((rx * ry).sum() / den) if den > 0 else 0.0
    n = len(rx)
    t = rho * np.sqrt((n - 2) / max(1e-12, 1 - rho ** 2))
    return {"rho": rho, "t": float(t), "n": n}


def tercile_bins(v):
    r = _rankdata(v)
    return np.minimum((r / len(v) * 3).astype(int), 2)


# ------------------------------------------------------------- evaluation ----

def mask_stats(ctx, msk, gt, S, C):
    """Per-(window, channel) mask/future descriptors (identical for all
    conditions -- masks are deterministic)."""
    m3 = msk.reshape(S, C, L)
    xc = ctx.reshape(S, C, L).astype(np.float64)
    nfm = m3.reshape(S, C, NPATCH, PATCH).all(axis=3).sum(axis=2)
    flat = msk.reshape(S * C, L)
    longest = np.zeros(S * C, np.int64)
    for i in range(S * C):
        d = np.diff(np.concatenate(([0], flat[i].view(np.int8), [0])))
        runs = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
        longest[i] = runs.max() if len(runs) else 0
    recency = m3[:, :, -H:].mean(axis=2)
    mu = xc.mean(axis=2)
    sd = np.where(xc.std(axis=2) > 1e-12, xc.std(axis=2), 1.0)
    cens_mass = (((xc - mu[:, :, None]) / sd[:, :, None]) * m3).sum(axis=2)
    obs = np.where(m3, np.nan, xc)
    omu = np.nanmean(obs, axis=2)
    osd = np.nanstd(obs, axis=2)
    osd = np.where(osd > 1e-12, osd, 1.0)
    fmax = gt.max(axis=2)
    return {"kind": "maskstats", "ds": None, "S": S, "C": C,
            "nfm": nfm.reshape(-1).tolist(),
            "longest": longest.tolist(),
            "recency": recency.reshape(-1).tolist(),
            "cens_mass": cens_mass.reshape(-1).tolist(),
            "fut_peak_clean": ((fmax - mu) / sd).reshape(-1).tolist(),
            "fut_peak_obs": ((fmax - omu) / osd).reshape(-1).tolist()}


def eval_cell(bank, X, starts, ds, cond):
    t0 = time.time()
    s15cond, use_ko = COND_MAP[cond]
    pipe, um = bank.get(s15cond, ds)
    raw = s15.raw_model(pipe)
    ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, MECH, RATE,
                                                   n_seeds=1)
    x_in = ctx.copy()
    x_in[msk] = np.nan
    with contextlib.ExitStack() as stack:
        if um:
            stack.enter_context(s15.UnmaskDecode(raw))
        if use_ko:
            stack.enter_context(s13.Knockout(raw, list(range(NLAYERS)),
                                             "full_miss"))
        pred = s8.predict_median(pipe, x_in)
    p = pred.astype(np.float64).reshape(S, C, H)
    e2 = (p - gt) ** 2
    mu_p = p.mean(axis=2)
    mu_g = gt.mean(axis=2)
    pc = p - mu_p[:, :, None]
    gc = gt - mu_g[:, :, None]
    num = (pc * gc).sum(axis=2)
    den = np.sqrt((pc ** 2).sum(axis=2) * (gc ** 2).sum(axis=2))
    shape_corr = np.where(den > 0, num / np.where(den == 0, 1.0, den), 0.0)
    wc = {"mse": e2.mean(axis=2),
          "lvl2": (mu_p - mu_g) ** 2,
          "shape_mse": ((pc - gc) ** 2).mean(axis=2),
          "shape_corr": shape_corr}
    for q in range(4):
        wc[f"seg{q + 1}"] = e2[:, :, q * 24:(q + 1) * 24].mean(axis=2)
    mse, mse_pw = s8.mse_windowed(pred, gt, S, C, nw)
    return {"kind": "cell", "ds": ds, "cond": cond, "mse": mse,
            "mse_per_window": mse_pw, "achieved_rate": float(msk.mean()),
            "n_windows": nw,
            "wc": {k: v.reshape(-1).tolist() for k, v in wc.items()},
            "seconds": round(time.time() - t0, 1)}


def run_eval(args):
    conds = {"anatomy": BASE_CONDS, "ablation": KO_CONDS}[args.jobs]
    cells, mstats, _ = scan_logs([args.log])
    bank = s15.ModelBank(args.device)
    n_done = 0
    for ds in DS:
        X, starts = s5.load_windows(s5.DATASETS[ds], args.windows, s5.SEED)
        if ds not in mstats:
            ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, MECH,
                                                           RATE, n_seeds=1)
            ms = mask_stats(ctx, msk, gt, S, C)
            ms["ds"] = ds
            emit(ms)
        for cond in conds:
            if (ds, cond) in cells:
                print(f"  skip {ds}:{cond} (in log)", flush=True)
                continue
            rec = eval_cell(bank, X, starts, ds, cond)
            emit(rec)
            n_done += 1
            print(f"  {ds:8s} {cond:14s} mse={rec['mse']:12.4f} "
                  f"({rec['seconds']}s)", flush=True)
    print(f"EVAL {args.jobs} DONE ({n_done} new cells)", flush=True)


def run_bias(args):
    """Supplementary eval (added after the level-share finding): per-(w,c)
    SIGNED level bias mu_pred - mu_gt, to test whether the mnar_high level
    damage is a systematic UNDER-prediction that unmask corrects upward."""
    _, _, done = scan_logs([args.log])
    bank = s15.ModelBank(args.device)
    n_done = 0
    for ds in DS:
        X, starts = s5.load_windows(s5.DATASETS[ds], args.windows, s5.SEED)
        ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, MECH, RATE,
                                                       n_seeds=1)
        x_in = ctx.copy()
        x_in[msk] = np.nan
        for cond in BIAS_CONDS:
            if (ds, cond) in done:
                print(f"  skip {ds}:{cond} (in log)", flush=True)
                continue
            s15cond, use_ko = COND_MAP[cond]
            pipe, um = bank.get(s15cond, ds)
            raw = s15.raw_model(pipe)
            t0 = time.time()
            with contextlib.ExitStack() as stack:
                if um:
                    stack.enter_context(s15.UnmaskDecode(raw))
                if use_ko:
                    stack.enter_context(s13.Knockout(
                        raw, list(range(NLAYERS)), "full_miss"))
                pred = s8.predict_median(pipe, x_in)
            p = pred.astype(np.float64).reshape(S, C, H)
            mu_p = p.mean(axis=2)
            mu_g = gt.mean(axis=2)
            rec = {"kind": "bias", "ds": ds, "cond": cond,
                   "wc": {"bias": (mu_p - mu_g).reshape(-1).tolist(),
                          "mu_gt": mu_g.reshape(-1).tolist()},
                   "seconds": round(time.time() - t0, 1)}
            emit(rec)
            n_done += 1
            b = np.array(rec["wc"]["bias"])
            print(f"  {ds:8s} {cond:14s} mean_bias={b.mean():+9.4f} "
                  f"mean|bias|={np.abs(b).mean():8.4f} ({rec['seconds']}s)",
                  flush=True)
    print(f"BIAS DONE ({n_done} new cells)", flush=True)


# ------------------------------------------------------------------ smoke ----

def run_smoke(args):
    device = args.device
    from chronos import BaseChronosPipeline
    X, starts = s5.load_windows(s5.DATASETS["ETTh1"], 5, s5.SEED)
    ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, MECH, RATE,
                                                   n_seeds=1)
    nfm = int(msk.reshape(S * C, NPATCH, PATCH).all(axis=2).sum())
    print(f"smoke: {S*C} series, fully-missing patches = {nfm}", flush=True)
    assert nfm > 50, "mnar_high p=0.7 should produce many fully-missing patches"
    x_in = ctx.copy()
    x_in[msk] = np.nan
    ko_layers = list(range(NLAYERS))

    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
    pipe.model = s6.wrap_lora(pipe.model)
    raw = s15.raw_model(pipe)

    # 1) mb ckpt: knockout EXACTLY inert under the native decode mask (S13 P4)
    sd_mb = torch.load(s15.ckpt_path("mb_anchor", "ETTh1"),
                       map_location=device)
    pipe.model.load_state_dict(sd_mb["trainable"], strict=False)
    pipe.model.eval()
    p_mb = s8.predict_median(pipe, x_in)
    with s13.Knockout(raw, ko_layers, "full_miss"):
        p_mb_ko = s8.predict_median(pipe, x_in)
    d = np.abs(p_mb_ko - p_mb).max()
    print(f"smoke: mb + knockout vs mb max|diff| = {d:.2e} (want exactly 0)",
          flush=True)
    assert d == 0.0, "knockout leaked through the native decode mask"

    # 2) unmask ckpt: knockout must BITE once the mask is open
    sd_un = torch.load(s15.ckpt_path("unmask", "ETTh1"), map_location=device)
    pipe.model.load_state_dict(sd_un["trainable"], strict=False)
    pipe.model.eval()
    p_un_native = s8.predict_median(pipe, x_in)   # unmask weights, native decode
    with s15.UnmaskDecode(raw):
        p_un = s8.predict_median(pipe, x_in)
        with s13.Knockout(raw, ko_layers, "full_miss"):
            p_un_ko = s8.predict_median(pipe, x_in)
    assert not np.isnan(p_un).any() and not np.isnan(p_un_ko).any()
    d1 = np.abs(p_un - p_un_native).max()
    d2 = np.abs(p_un_ko - p_un).max()
    print(f"smoke: unmask vs native-decode max|diff| = {d1:.4f} (want >0)",
          flush=True)
    print(f"smoke: unmask+knockout vs unmask max|diff| = {d2:.4f} (want >0: "
          f"the knockout bites once gap positions are readable)", flush=True)
    assert d1 > 0 and d2 > 0

    # 3) attention-level: under LoRA + knockout, gap queries put exactly 0
    #    mass on patch keys and 1.0 on [REG] at all 12 layers
    ko = s13.Knockout(raw, ko_layers, "full_miss")
    nb = min(64, len(x_in))
    with ko:
        _, enc = s13.extract_hidden(types.SimpleNamespace(model=raw),
                                    x_in[:nb], ko=ko, batch=nb,
                                    need_attn=True)
    pm = (~msk[:nb]).reshape(nb, NPATCH, PATCH).astype(np.float32)
    qm = torch.from_numpy(pm.sum(-1) == 0).to(device)       # [nb, 32] bool
    nq = int(qm.sum())
    assert nq > 50
    for l, a in enumerate(enc.attentions):                  # [B, nh, 33, 33]
        nh = a.shape[1]
        qm4 = qm[:, None, :, None].expand(-1, nh, -1, NPATCH)
        w_patch = a[:, :, :NPATCH, :NPATCH][qm4]
        w_reg = a[:, :, :NPATCH, REG].masked_select(
            qm[:, None, :].expand(-1, nh, -1))
        mx = float(w_patch.abs().max()) if w_patch.numel() else 0.0
        mn = float(w_reg.min()) if w_reg.numel() else 1.0
        assert mx == 0.0, f"layer {l}: knocked query keeps {mx} on patch keys"
        assert abs(mn - 1.0) < 1e-6, f"layer {l}: REG mass {mn} != 1"
    print(f"smoke: knockout under LoRA: patch-key mass exactly 0, REG mass "
          f"1.0 at all 12 layers ({nq} knocked queries)", flush=True)

    # 4) ZS pipe: unmask + knockout composition also works there
    zs = s8.load_bolt(device)
    with s15.UnmaskDecode(zs.model):
        p_zu = s8.predict_median(zs, x_in)
        with s13.Knockout(zs.model, ko_layers, "full_miss"):
            p_zu_ko = s8.predict_median(zs, x_in)
    dz = np.abs(p_zu_ko - p_zu).max()
    print(f"smoke: zs unmask+knockout vs zs unmask max|diff| = {dz:.4f} "
          f"(want >0)", flush=True)
    assert dz > 0 and not np.isnan(p_zu_ko).any()

    # 5) restore: all patches gone, mb prediction reproduced bitwise
    assert "decode" not in raw.__dict__ and "encode" not in raw.__dict__
    pipe.model.load_state_dict(sd_mb["trainable"], strict=False)
    pipe.model.eval()
    p_after = s8.predict_median(pipe, x_in)
    dr = np.abs(p_after - p_mb).max()
    print(f"smoke: post-restore mb predict max|diff| = {dr:.2e} (want 0)",
          flush=True)
    assert dr == 0.0
    print("SMOKE OK", flush=True)


# ----------------------------------------------------------------- anchor ----

def run_anchor(args):
    r15 = json.load(open(S15_JSON))
    cells, _, _ = scan_logs([LOGS["anatomy"]])
    out = {}
    if os.path.exists(RESULTS):
        out = json.load(open(RESULTS))
    out.setdefault("meta", {
        "track": "S19 mechanism of the S15 mnar_high unmask gain "
                 "(appendix-level; no new method)",
        "cell": f"{MECH} p={RATE}, nan fill, 150 windows s5.SEED, "
                "deterministic mask, median quantile",
        "datasets": {"primary": ["ETTh1", "ETTm1"], "control": ["weather"]},
        "conds": {c: {"s15_cond": COND_MAP[c][0],
                      "knockout": "full_miss all-12-layers -> [REG]"
                      if COND_MAP[c][1] else None} for c in CONDS},
        "checkpoints": "s15_ckpt/ (mb_anchor == s12_mb bit-identical; unmask)",
        "pre_registered_predictions": {
            "P1": "unmask_ko retains <=20% of the unmask-vs-mb gain on ETT "
                  "(content causality; >=80% kills the scaffold story)",
            "P2": "horizon gains front-loaded: rel_gain(q1) > rel_gain(q4)",
            "P3": "gain not concentrated in future-peak top decile; visible "
                  "shape-error share",
            "P4": "gain monotone in fully-missing-patch terciles (ETT)",
            "P5": "weather flat (<1%) everywhere; mb_ko == mb bitwise"}})
    gate, n_cells = True, 0
    rep = {"tolerance": ANCHOR_TOL, "cells": {}}
    print("== S19 anchor gate vs s15_results.json (mnar_high p=0.7) ==",
          flush=True)
    for ds in DS:
        for cond in BASE_CONDS:
            mine = cells.get((ds, cond))
            ref = r15["neg"][ds][S15_CELL[cond]]
            ck = f"{ds}:{cond}"
            if mine is None:
                rep["cells"][ck] = {"ref": ref["mse"], "status": "MISSING"}
                gate = False
                print(f"  {ck:24s} MISSING", flush=True)
                continue
            ratio = mine["mse"] / ref["mse"]
            dpw = max(abs(x - y) for x, y in zip(mine["mse_per_window"],
                                                 ref["mse_per_window"]))
            ok = abs(ratio - 1) <= ANCHOR_TOL
            gate &= ok
            n_cells += 1
            rep["cells"][ck] = {"mine": mine["mse"], "ref": ref["mse"],
                                "ratio": ratio, "max_window_diff": dpw,
                                "ok": ok}
            print(f"  {ck:24s} mine={mine['mse']:12.4f} s15={ref['mse']:12.4f} "
                  f"ratio={ratio:.6f} maxwdiff={dpw:.2e} -> "
                  f"{'OK' if ok else 'FAIL'}", flush=True)
    rep["n_cells"] = n_cells
    rep["gate"] = "PASS" if gate else "FAIL"
    out["anchor"] = rep
    s5.save_results(out, RESULTS)
    print(f"ANCHOR {rep['gate']}", flush=True)
    if not gate:
        raise SystemExit(2)


# ------------------------------------------------------------------ merge ----

def run_merge(args):
    out = json.load(open(RESULTS))
    if out.get("anchor", {}).get("gate") != "PASS":
        raise SystemExit("anchor gate not passed -- run --anchor first")
    cells, mstats, bias = scan_logs([LOGS["anatomy"], LOGS["ablation"],
                                     LOGS["bias"]])
    missing = [(ds, c) for ds in DS for c in CONDS if (ds, c) not in cells]
    if missing:
        raise SystemExit(f"missing cells: {missing}")
    out["cells"] = {ds: {c: cells[(ds, c)] for c in CONDS} for ds in DS}
    out["maskstats"] = {ds: mstats[ds] for ds in DS}

    # ---- controls ----
    controls = {}
    for ds in DS:
        a = out["cells"][ds]["mb_ko"]["mse_per_window"]
        b = out["cells"][ds]["mb"]["mse_per_window"]
        d = max(abs(x - y) for x, y in zip(a, b))
        controls[ds] = {"mb_ko_vs_mb_max_window_diff": d, "bitwise": d == 0.0}
        assert d == 0.0, f"mb_ko != mb on {ds}: {d}"
    out["controls"] = controls
    print(f"control mb_ko==mb bitwise on all datasets: OK", flush=True)

    # ---- analyses ----
    an = {}
    for ds in DS:
        c = out["cells"][ds]
        S = c["mb"]["n_windows"]
        C = len(c["mb"]["wc"]["mse"]) // S
        wp = lambda cond: np.array(c[cond]["mse_per_window"], np.float64)
        wc = lambda cond, k: np.array(c[cond]["wc"][k], np.float64)
        e = {"S": S, "C": C, "paired": {}, "horizon": {}, "levelshape": {},
             "dose": {}, "peak": {}}

        # 1) ablation paired table (window level, n=150)
        pairs = [("un_vs_mb", "unmask", "mb"),
                 ("unko_vs_mb", "unmask_ko", "mb"),
                 ("unko_vs_un", "unmask_ko", "unmask"),
                 ("mbko_vs_mb", "mb_ko", "mb"),
                 ("zsu_vs_zsm", "zs_unmask", "zs_masked"),
                 ("zsuko_vs_zsu", "zs_unmask_ko", "zs_unmask")]
        for name, a, b in pairs:
            e["paired"][name] = pgain(wp(a), wp(b))
        gu = e["paired"]["un_vs_mb"]["rel_gain"]
        gk = e["paired"]["unko_vs_mb"]["rel_gain"]
        e["paired"]["gain_retention"] = (gk / gu) if abs(gu) > 1e-12 else None

        # 2) horizon quarters (window-level channel-avg segment MSE)
        for name, a, b in (("un_vs_mb", "unmask", "mb"),
                           ("unko_vs_mb", "unmask_ko", "mb")):
            qs = {}
            for q in range(4):
                va = wc(a, f"seg{q + 1}").reshape(S, C).mean(axis=1)
                vb = wc(b, f"seg{q + 1}").reshape(S, C).mean(axis=1)
                qs[f"q{q + 1}"] = pgain(va, vb)
            e["horizon"][name] = qs

        # 3) level vs shape (wc level; mse == lvl2 + shape_mse exactly)
        for name, a, b in (("un_vs_mb", "unmask", "mb"),
                           ("unko_vs_mb", "unmask_ko", "mb")):
            d_tot = wc(b, "mse") - wc(a, "mse")
            d_lvl = wc(b, "lvl2") - wc(a, "lvl2")
            d_shp = wc(b, "shape_mse") - wc(a, "shape_mse")
            mb_mean = float(wc(b, "mse").mean())
            st = {"d_total": pgain(wc(a, "mse"), wc(b, "mse")),
                  "d_level": pgain(wc(a, "lvl2"), wc(b, "lvl2")),
                  "d_shape": pgain(wc(a, "shape_mse"), wc(b, "shape_mse")),
                  "shape_corr": pgain(-wc(a, "shape_corr"),
                                      -wc(b, "shape_corr"))}
            st["shape_corr"]["mean_a"] = float(wc(a, "shape_corr").mean())
            st["shape_corr"]["mean_b"] = float(wc(b, "shape_corr").mean())
            st["level_share"] = (float(d_lvl.mean()) / float(d_tot.mean())
                                 if abs(d_tot.mean()) > 1e-12 else None)
            st["shape_share"] = (float(d_shp.mean()) / float(d_tot.mean())
                                 if abs(d_tot.mean()) > 1e-12 else None)
            st["rel_level"] = float(d_lvl.mean()) / mb_mean
            st["rel_shape"] = float(d_shp.mean()) / mb_mean
            e["levelshape"][name] = st

        # 4) dose response (wc level): terciles by fully-missing-patch count
        nfm = np.array(mstats[ds]["nfm"], np.float64)
        cmass = np.array(mstats[ds]["cens_mass"], np.float64)
        for name, a in (("un_vs_mb", "unmask"), ("unko_vs_mb", "unmask_ko")):
            d = wc("mb", "mse") - wc(a, "mse")
            g = d / wc("mb", "mse")
            bins = tercile_bins(nfm)
            tb = {}
            for bi in range(3):
                sel = bins == bi
                tb[f"t{bi + 1}"] = {
                    "n": int(sel.sum()),
                    "nfm_range": [float(nfm[sel].min()),
                                  float(nfm[sel].max())],
                    **pgain(wc(a, "mse")[sel], wc("mb", "mse")[sel])}
            e["dose"][name] = {"by_nfm_tercile": tb,
                               "spearman_nfm": spearman(nfm, g),
                               "spearman_cens_mass": spearman(cmass, g)}

        # 5) future-peak top decile (wc level + window level)
        fp = np.array(mstats[ds]["fut_peak_clean"], np.float64)
        fpo = np.array(mstats[ds]["fut_peak_obs"], np.float64)
        for name, a in (("un_vs_mb", "unmask"), ("unko_vs_mb", "unmask_ko")):
            r = _rankdata(fp) / len(fp)
            top = r >= 0.9
            e["peak"][name] = {
                "wc_top_decile": {"n": int(top.sum()),
                                  "fp_range": [float(fp[top].min()),
                                               float(fp[top].max())],
                                  **pgain(wc(a, "mse")[top],
                                          wc("mb", "mse")[top])},
                "wc_rest": {"n": int((~top).sum()),
                            **pgain(wc(a, "mse")[~top], wc("mb", "mse")[~top])},
                "spearman_fut_peak": spearman(
                    fp, (wc("mb", "mse") - wc(a, "mse")) / wc("mb", "mse")),
                "spearman_fut_peak_obs": spearman(
                    fpo, (wc("mb", "mse") - wc(a, "mse")) / wc("mb", "mse"))}
        # window level: window fut peak = max over channels; top 15 windows
        wf = fp.reshape(S, C).max(axis=1)
        wt = _rankdata(wf) / S >= 0.9
        e["peak"]["window_level"] = {
            "n_top": int(wt.sum()),
            "top": pgain(wp("unmask")[wt], wp("mb")[wt]),
            "rest": pgain(wp("unmask")[~wt], wp("mb")[~wt])}

        # 6) SFT necessity (interaction) on window level
        g_sft = e["paired"]["un_vs_mb"]["rel_gain"]
        g_zs = e["paired"]["zsu_vs_zsm"]["rel_gain"]
        e["sft_necessity"] = {"gain_sft": g_sft, "gain_zs": g_zs,
                              "interaction": g_sft - g_zs,
                              "gain_zs_under_ko":
                                  e["paired"]["zsuko_vs_zsu"]["rel_gain"]}
        an[ds] = e
    out["analysis"] = an

    # ---- supplementary signed-bias analysis (if s19_bias.log exists) ----
    if all((ds, cd) in bias for ds in DS for cd in BIAS_CONDS):
        ban = {}
        for ds in DS:
            be = {}
            for cd in BIAS_CONDS:
                b = np.array(bias[(ds, cd)]["wc"]["bias"], np.float64)
                be[cd] = {"mean_bias": float(b.mean()),
                          "median_bias": float(np.median(b)),
                          "mean_abs_bias": float(np.abs(b).mean())}
            for name, a in (("un_vs_mb", "unmask"),
                            ("unko_vs_mb", "unmask_ko"),
                            ("mb_vs_zsm", "mb")):
                ref = "mb" if name != "mb_vs_zsm" else "zs_masked"
                ba = np.abs(np.array(bias[(ds, a)]["wc"]["bias"], np.float64))
                bb = np.abs(np.array(bias[(ds, ref)]["wc"]["bias"],
                                    np.float64))
                be[f"absbias_paired:{name}"] = pgain(ba, bb)
            ban[ds] = be
        out["bias_analysis"] = ban
        print("bias analysis folded in", flush=True)
    s5.save_results(out, RESULTS)
    print(f"merged -> {RESULTS}", flush=True)


# ---------------------------------------------------------------- summary ----

def run_summary(args):
    out = json.load(open(RESULTS))
    print("== anchor:", out.get("anchor", {}).get("gate"),
          f"({out.get('anchor', {}).get('n_cells')} cells, "
          f"tol ±{out.get('anchor', {}).get('tolerance')})")
    print("== controls:", json.dumps(out.get("controls")))
    for ds in DS:
        a = out["analysis"][ds]
        print(f"\n===== {ds} (S={a['S']} windows x C={a['C']} channels) =====")
        print("-- paired (window level; gain>0 = first cond better) --")
        for k, v in a["paired"].items():
            if k == "gain_retention":
                print(f"  gain_retention = {v:.3f}")
                continue
            print(f"  {k:12s} rel_gain={v['rel_gain'] * 100:+7.2f}% "
                  f"t={v['t']:+8.2f} win={v['win_rate']:.3f} "
                  f"(ref mean {v['mean_ref']:.4f})")
        print("-- horizon quarters (rel gain %) --")
        for name, qs in a["horizon"].items():
            row = "  ".join(f"{q}:{qs[q]['rel_gain'] * 100:+.2f}%"
                            f"(t={qs[q]['t']:+.1f})" for q in qs)
            print(f"  {name:12s} {row}")
        print("-- level vs shape (share of total gain) --")
        for name, st in a["levelshape"].items():
            print(f"  {name:12s} level_share={st['level_share']:+.3f} "
                  f"shape_share={st['shape_share']:+.3f} "
                  f"(rel {st['rel_level'] * 100:+.2f}% / "
                  f"{st['rel_shape'] * 100:+.2f}% of mb mse) "
                  f"shape_corr {st['shape_corr']['mean_b']:.4f}->"
                  f"{st['shape_corr']['mean_a']:.4f} "
                  f"(t={st['shape_corr']['t']:+.1f})")
        print("-- dose response (nfm terciles, rel gain %) --")
        for name, dz in a["dose"].items():
            row = "  ".join(
                f"{t}:{dz['by_nfm_tercile'][t]['rel_gain'] * 100:+.2f}%"
                f"[nfm {dz['by_nfm_tercile'][t]['nfm_range'][0]:.0f}-"
                f"{dz['by_nfm_tercile'][t]['nfm_range'][1]:.0f}]"
                for t in ("t1", "t2", "t3"))
            sp = dz["spearman_nfm"]
            print(f"  {name:12s} {row}  spearman rho={sp['rho']:+.3f} "
                  f"(t={sp['t']:+.1f})")
        print("-- future-peak top decile (rel gain %) --")
        for name in ("un_vs_mb", "unko_vs_mb"):
            pk = a["peak"][name]
            print(f"  {name:12s} top={pk['wc_top_decile']['rel_gain'] * 100:+.2f}%"
                  f" (n={pk['wc_top_decile']['n']}, t={pk['wc_top_decile']['t']:+.1f})"
                  f" rest={pk['wc_rest']['rel_gain'] * 100:+.2f}%"
                  f" (t={pk['wc_rest']['t']:+.1f})"
                  f"  rho(fp,gain)={pk['spearman_fut_peak']['rho']:+.3f}")
        wl = a["peak"]["window_level"]
        print(f"  window-level top={wl['top']['rel_gain'] * 100:+.2f}% "
              f"(n={wl['n_top']}) rest={wl['rest']['rel_gain'] * 100:+.2f}%")
        sn = a["sft_necessity"]
        print(f"-- SFT necessity: sft {sn['gain_sft'] * 100:+.2f}% vs zs "
              f"{sn['gain_zs'] * 100:+.2f}% -> interaction "
              f"{sn['interaction'] * 100:+.2f}%; zs under KO "
              f"{sn['gain_zs_under_ko'] * 100:+.2f}%")
    ban = out.get("bias_analysis")
    if ban:
        print("\n===== signed level bias (mu_pred - mu_gt, per w,c) =====")
        for ds in DS:
            be = ban[ds]
            row = "  ".join(f"{cd}: mean={be[cd]['mean_bias']:+.4f} "
                            f"|b|={be[cd]['mean_abs_bias']:.4f}"
                            for cd in BIAS_CONDS)
            print(f"  {ds:8s} {row}")
            for k in be:
                if k.startswith("absbias_paired:"):
                    v = be[k]
                    print(f"  {ds:8s} {k[15:]:12s} |bias| rel_gain="
                          f"{v['rel_gain'] * 100:+.2f}% t={v['t']:+.2f} "
                          f"win={v['win_rate']:.3f}")


# ----------------------------------------------------------------- figure ----

def make_figure(out, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r15 = json.load(open(S15_JSON))
    clean_zs = {ds: r15["clean"][ds]["clean:zs_masked"]["mse"] for ds in DS}
    disp = {"zs_masked": "zs:masked", "zs_unmask": "zs:unmask",
            "zs_unmask_ko": "zs:unmask+KO", "mb": "sft:mb",
            "mb_ko": "sft:mb+KO", "unmask": "sft:unmask",
            "unmask_ko": "sft:unmask+KO"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # (a) ablation bars: relMSE vs s15 zs-clean, per condition x dataset
    ax = axes[0][0]
    x = np.arange(len(DS))
    w = 0.12
    for ci, cond in enumerate(CONDS):
        ys = [out["cells"][ds][cond]["mse"] / clean_zs[ds] for ds in DS]
        ax.bar(x + (ci - 3) * w, ys, w, label=disp[cond])
    ax.set_xticks(x)
    ax.set_xticklabels(DS)
    ax.set_ylabel("relMSE (vs zs clean)")
    ax.set_title(f"ablation on {MECH} p={RATE}: gain dies with the knockout?")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=6.5, ncol=2)

    # (b) horizon quarters: rel gain % of un vs mb (solid) and unko vs mb
    ax = axes[0][1]
    marks = {"ETTh1": "o", "ETTm1": "s", "weather": "^"}
    for ds in DS:
        qs = out["analysis"][ds]["horizon"]["un_vs_mb"]
        ys = [qs[f"q{i + 1}"]["rel_gain"] * 100 for i in range(4)]
        es = [qs[f"q{i + 1}"]["ci95"] / qs[f"q{i + 1}"]["mean_ref"] * 100
              for i in range(4)]
        ax.errorbar(np.arange(4) - 0.03, ys, yerr=es, marker=marks[ds],
                    ms=4, lw=1.4, label=f"{ds} unmask", capsize=2)
        qk = out["analysis"][ds]["horizon"]["unko_vs_mb"]
        yk = [qk[f"q{i + 1}"]["rel_gain"] * 100 for i in range(4)]
        ax.plot(np.arange(4) + 0.03, yk, marker=marks[ds], ms=3, lw=0.9,
                ls=":", alpha=0.6, color=ax.lines[-1].get_color(),
                label=f"{ds} unmask+KO")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([f"h{i * 24}-{(i + 1) * 24}" for i in range(4)])
    ax.set_ylabel("paired rel gain vs sft:mb (%)")
    ax.set_title("horizon decomposition (phase prior: front-loaded?)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5)

    # (c) level vs shape: stacked share of total gain, un vs mb
    ax = axes[1][0]
    x = np.arange(len(DS))
    lvl, shp, lvl_ko, shp_ko = [], [], [], []
    for ds in DS:
        st = out["analysis"][ds]["levelshape"]["un_vs_mb"]
        sk = out["analysis"][ds]["levelshape"]["unko_vs_mb"]
        lvl.append(st["rel_level"] * 100)
        shp.append(st["rel_shape"] * 100)
        lvl_ko.append(sk["rel_level"] * 100)
        shp_ko.append(sk["rel_shape"] * 100)
    ax.bar(x - 0.2, lvl, 0.35, label="level (mean shift)", color="tab:blue")
    ax.bar(x - 0.2, shp, 0.35, bottom=lvl, label="shape (centered)",
           color="tab:cyan")
    ax.bar(x + 0.2, lvl_ko, 0.35, color="tab:blue", alpha=0.35)
    ax.bar(x + 0.2, shp_ko, 0.35, bottom=lvl_ko, color="tab:cyan", alpha=0.35)
    for xi, ds in enumerate(DS):
        sc = out["analysis"][ds]["levelshape"]["un_vs_mb"]["shape_corr"]
        ax.text(xi - 0.2, lvl[xi] + shp[xi] + 0.15,
                f"corr {sc['mean_b']:.3f}->{sc['mean_a']:.3f}",
                ha="center", fontsize=7)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ds}\n(unmask | unmask+KO)" for ds in DS])
    ax.set_ylabel("gain vs sft:mb (% of mb MSE)")
    ax.set_title("level vs shape decomposition of the gain")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    # (d) dose terciles + future-peak decile, un vs mb
    ax = axes[1][1]
    cats = ["sev T1", "sev T2", "sev T3", "peak\ndecile", "rest"]
    x = np.arange(len(cats))
    w = 0.25
    for di, ds in enumerate(DS):
        dz = out["analysis"][ds]["dose"]["un_vs_mb"]["by_nfm_tercile"]
        pk = out["analysis"][ds]["peak"]["un_vs_mb"]
        ys = [dz["t1"]["rel_gain"] * 100, dz["t2"]["rel_gain"] * 100,
              dz["t3"]["rel_gain"] * 100,
              pk["wc_top_decile"]["rel_gain"] * 100,
              pk["wc_rest"]["rel_gain"] * 100]
        ax.bar(x + (di - 1) * w, ys, w, label=ds)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel("paired rel gain of sft:unmask vs mb (%)")
    ax.set_title("dose response (severity terciles) + future-peak decile")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    gate = out.get("anchor", {}).get("gate", "?")
    fig.suptitle(f"S19 mechanism of the mnar_high unmask gain "
                 f"(chronos-bolt-base) — anchor gate {gate}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    print(f"figure -> {path}", flush=True)


# ------------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--jobs", choices=("anatomy", "ablation"), default="anatomy")
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--windows", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log", default=None)
    ap.add_argument("--fig-out", default=os.path.join(HERE, "s19.png"))
    args = ap.parse_args()

    torch.manual_seed(s5.SEED)
    np.random.seed(s5.SEED)
    if args.log is None:
        args.log = LOGS["bias"] if args.bias else LOGS[args.jobs]

    if args.smoke:
        run_smoke(args)
        return
    if args.eval:
        run_eval(args)
        return
    if args.bias:
        run_bias(args)
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
