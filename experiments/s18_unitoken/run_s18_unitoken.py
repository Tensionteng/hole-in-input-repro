#!/usr/bin/env python
"""S18: universal [MASK] token -- can ONE gap-prior token trained on a
cross-dataset corpus escape the in-domain limitation found in S8?

Background (s8_notes.md / s8_multiseed_notes.md): a 768-d zero-init [MASK]
token (backbone frozen, clipped pinball clip=100, AdamW 1e-3, 1000 steps x
256 series, 50% mcar / 50% block(24), p~U(0.05,0.8), train split) beats every
control under block missingness in-domain, but the transfer matrix is
diagonal-dominant: the weather token does NOT transfer to ETT (worse than
nan). S18 trains ONE token on an equal-parts mixture of the ETTh1 + ETTm1 +
weather train splits and asks whether it is "decent everywhere".

Design (pre-registered judgement, see s18_notes.md):
  uni     -- one token, mixed corpus. Per batch of 256, element b draws from
             DS[b % 3] (exactly equal thirds: 86/85/85), everything else
             identical to S8 Exp2. 3 training seeds, S8b convention: base
             101/202/303 -> actual = base + 31 (132/233/334; DS_IDX=0 for the
             single mixed token).
  uniaug  -- same, plus per-window random scale jitter f~exp(U(-ln2,+ln2))
             and linear-trend perturbation (total drift d~U(-2,+2) x
             std(context) across the L+PRED window), drawn from a SEPARATE
             rng stream (seed + 777000) so uni and uniaug see identical
             windows/masks per step. Intent: stop the token from memorising
             dataset-specific levels. Known design caveat (recorded in the
             notes): bolt instance-normalises per series and the clipped
             pinball loss is computed in normalised units, so pure scale
             jitter cancels almost exactly; the trend perturbation is the
             operative ingredient.
  eval    -- S8 Exp3 harness REUSED VERBATIM (same 300 windows, same 2 mask
             seeds, same controls, same batch sizes) on the block grid
             p in {0.1..0.7} x {ETTh1, ETTm1, weather}; plus the no-gain
             check cells mcar@0.3 and mnar_high@0.7.
  methods -- anchor controls: nan / linear / mponly / tok_ETTh1 / tok_ETTm1 /
             tok_weather (loaded from s8_masktoken_{ds}.pt); new:
             uni_s{101,202,303}, uniaug_s{101,202,303}.
  anchor  -- the 6 control methods re-evaluated on ALL 27 cells/ds BEFORE
             training; gate: every cell within +-5% of s8_results.json
             (S8b showed the pipeline is bit-deterministic, so ~0% expected).

Pre-registered judgement (block rate-avg relMSE per test ds, uni = mean over
the 3 seeds; tok_own = matched S8 in-domain token):
  SUCCESS         : for every ds, uni <= nan AND uni - tok_own <= 0.02
                    ("nowhere bad");
  PARTIAL SUCCESS : for every ds, uni <= nan, but some ds has
                    uni - tok_own > 0.02 (out-of-domain collapse gone,
                    in-domain gap remains);
  FAILURE         : some ds has uni > nan (still collapses out-of-domain).
Same criteria applied to uniaug. Check cells: mcar@0.3 expects |uni-nan| <=
0.02 (S8: token is a no-op under mcar); mnar_high@0.7 expects uni <= nan +
0.10 (S8: no transfer, slight degradation tolerated, collapse = fail).

Sharding: --anchor / --eval take --shard/--nshards and append machine-lines
("RESULT {json}" / "CLEAN {json}") to s18_{anchor,eval}_shard<i>.log (human
progress on stdout). Resume-safe: completed cells in the log are skipped.
--gate / --merge / --report parse those logs (no GPU needed).
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import torch

import run_s5_missing as s5
import run_s6_sft as s6
import run_s8_masktoken as s8

L, H, PRED = s8.L, s8.H, s8.PRED          # 512, 96, 64
DS = list(s8.DS)                          # ETTh1, ETTm1, weather
RATES3 = list(s8.RATES3)                  # 0.1..0.7
CELLS = [("block", r) for r in RATES3] + [("mcar", 0.3), ("mnar_high", 0.7)]
ANCHOR_METHODS = ["nan", "linear", "mponly",
                  "tok_ETTh1", "tok_ETTm1", "tok_weather"]
SEED_BASES = (101, 202, 303)
VARIANTS = ("uni", "uniaug")
NEW_METHODS = [f"{v}_s{b}" for v in VARIANTS for b in SEED_BASES]
TOK18 = {m: os.path.join(HERE, f"s18_unitoken_{m}.pt") for m in NEW_METHODS}
RESULTS = os.path.join(HERE, "s18_results.json")
S8_RESULTS = os.path.join(HERE, "s8_results.json")

# uniaug perturbation ranges
SCALE_HALFLOG = math.log(2.0)   # f ~ exp(U(-ln2, +ln2)) in [0.5, 2]
DRIFT_MAX = 2.0                 # total trend drift ~ U(-2, +2) x std(context)
AUG_STREAM_OFFSET = 777000      # separate rng stream for the perturbations

STEPS, BATCH, LR, LOSS_CLIP, LOG_EVERY = 1000, 256, 1e-3, 100.0, 50


def actual_seed(base):
    """S8b seed convention (base + 31 + DS_IDX); single mixed token -> DS_IDX 0."""
    return base + 31


# ------------------------------------------------------------ training ------

def load_train_data():
    data = []
    for ds in DS:
        X = s8.load_X(ds)
        data.append(X[: int(0.7 * len(X))])
    return data


def mix_batch(rng, rng_aug, data, batch, variant):
    """One mixed-corpus batch. Element b draws its series from DS[b % 3]
    (exactly equal thirds per batch). rng draws (start, channel, then the
    aug_mask draws in the caller) are identical between uni and uniaug for the
    same seed; uniaug's scale/trend perturbations come from rng_aug, a
    separate stream. Returns ctx [B, L], fut [B, PRED] float32."""
    ctx = np.empty((batch, L), np.float32)
    fut = np.empty((batch, PRED), np.float32)
    ramp = np.linspace(-0.5, 0.5, L + PRED, dtype=np.float32)
    for b in range(batch):
        Xtr = data[b % len(DS)]
        n, c = Xtr.shape
        st = rng.integers(0, n - L - PRED)
        ch = rng.integers(0, c)
        w = Xtr[st:st + L + PRED, ch].astype(np.float32)      # [L+PRED]
        if variant == "uniaug":
            f = math.exp(rng_aug.uniform(-SCALE_HALFLOG, SCALE_HALFLOG))
            d = rng_aug.uniform(-DRIFT_MAX, DRIFT_MAX) * float(w[:L].std())
            w = f * w + d * ramp
        ctx[b] = w[:L]
        fut[b] = w[L:]
    return ctx, fut


def train(args):
    variant, base = args.variant, args.seed_base
    assert variant in VARIANTS and base in SEED_BASES
    method = f"{variant}_s{base}"
    seed = actual_seed(base)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    rng_aug = np.random.default_rng(seed + AUG_STREAM_OFFSET)
    device = "cuda"
    pipe = s8.load_bolt(device)
    model = pipe.model
    token = torch.nn.Parameter(torch.zeros(model.config.d_model, device=device))
    for prm in model.parameters():
        prm.requires_grad_(False)
    opt = torch.optim.AdamW([token], lr=args.lr)
    print(f"[{method}] trainable params: {token.numel():,} (zero-init [MASK] token); "
          f"actual seed {seed}; corpus = equal thirds of {','.join(DS)} train splits",
          flush=True)

    data = load_train_data()
    t0 = time.time()
    losses = []
    run = [0.0, 0.0]
    with s8.EncodePatch(model, mask_mode="correct", mask_token=token):
        for step in range(1, args.steps + 1):
            ctx, fut = mix_batch(rng, rng_aug, data, args.batch, variant)
            m = np.stack([s8.aug_mask(rng, ctx[b]) for b in range(args.batch)])
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
                print(f"[{method}] step {step}/{args.steps} loss={run[0] / k:.4f} "
                      f"raw={run[1] / k:.2f} |tok|={token.norm().item():.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
                run[0] = run[1] = 0.0
    meta = {"method": method, "variant": variant, "seed_base": base,
            "actual_seed": seed, "aug_stream_seed": seed + AUG_STREAM_OFFSET,
            "corpus": "equal thirds (b%3) of ETTh1/ETTm1/weather train splits (0.7)",
            "steps": args.steps, "batch": args.batch, "lr": args.lr,
            "loss_clip": args.loss_clip, "pred_len_train": PRED,
            "aug": ("50% mcar / 50% block(24), p~U(0.05,0.8)"
                    + ("; + scale f~exp(U(-ln2,ln2)), trend drift d~U(-2,2)*std(ctx)"
                       if variant == "uniaug" else "")),
            "token_norm": float(token.norm().item())}
    out = args.out_ckpt or TOK18[method]
    torch.save({"mask_token": token.detach().cpu(), "meta": meta, "losses": losses}, out)
    print(f"[{method}] saved -> {out} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------ eval ----------

def run_cell(pipe, tokens, mponly, X, starts, ds, mech, rate, method):
    """Identical semantics to s8.run_exp3_job; only the token lookup differs
    (keyed by full method name)."""
    ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, mech, rate)
    t0 = time.time()
    if method == "linear":
        pred = s8.predict_median(pipe, s8.fill_linear(ctx, msk))
    else:
        x_nan = ctx.copy()
        x_nan[msk] = np.nan
        if method == "nan":
            pred = s8.predict_median(pipe, x_nan)
        elif method == "mponly":
            with mponly:
                pred = s8.predict_median(pipe, x_nan)
        else:
            with s8.EncodePatch(pipe.model, mask_mode="correct",
                                mask_token=tokens[method]):
                pred = s8.predict_median(pipe, x_nan)
    mse, mse_pw = s8.mse_windowed(pred, gt, S, C, nw)
    return {"ds": ds, "mech": mech, "rate": rate, "method": method, "mse": mse,
            "mse_per_window": mse_pw, "seconds": round(time.time() - t0, 1)}


def parse_log(path):
    """RESULT/CLEAN lines -> {(ds, mech, rate, method): cell}, {ds: clean}."""
    cells, cleans = {}, {}
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                if line.startswith("RESULT "):
                    d = json.loads(line[len("RESULT "):])
                    cells[(d["ds"], d["mech"], d["rate"], d["method"])] = d
                elif line.startswith("CLEAN "):
                    d = json.loads(line[len("CLEAN "):])
                    cleans[d["ds"]] = d
    return cells, cleans


def shard_main(args, methods, kind):
    log = os.path.join(HERE, f"s18_{kind}_shard{args.shard}.log")
    done, cleans = parse_log(log)
    fh = open(log, "a", buffering=1)
    device = "cuda"
    pipe = s8.load_bolt(device)
    tokens = {}
    for m in methods:
        if m.startswith("tok_"):
            sd = torch.load(s8.TOK_CKPT[m[4:]], map_location=device)
            tokens[m] = sd["mask_token"].to(device)
        elif m in NEW_METHODS:
            sd = torch.load(TOK18[m], map_location=device)
            tokens[m] = sd["mask_token"].to(device)
        # nan / linear / mponly need no token
    mponly = s8.MpOnlyAdapter(pipe.model) if "mponly" in methods else None
    jobs = [(ds, mech, rate, m) for ds in DS for (mech, rate) in CELLS
            for m in methods]
    jobs = jobs[args.shard::args.nshards]
    print(f"{kind} shard {args.shard}/{args.nshards}: {len(jobs)} jobs "
          f"({len(done)} already in {log})", flush=True)
    loaded = {}
    for ds, mech, rate, m in jobs:
        if (ds, mech, rate, m) in done:
            continue
        if ds not in loaded:
            loaded[ds] = s5.load_windows(s5.DATASETS[ds], args.windows, s5.SEED)
        X, starts = loaded[ds]
        if ds not in cleans:
            t0 = time.time()
            cl = s8.clean_baseline(pipe, X, starts)
            cl = {"ds": ds, "mse": cl["mse"],
                  "mse_per_window": cl["mse_per_window"],
                  "seconds": round(time.time() - t0, 1)}
            fh.write("CLEAN " + json.dumps(cl) + "\n")
            fh.flush()
            cleans[ds] = cl
            print(f"  clean {ds} mse={cl['mse']:.4f}", flush=True)
        res = run_cell(pipe, tokens, mponly, X, starts, ds, mech, rate, m)
        fh.write("RESULT " + json.dumps(res) + "\n")
        fh.flush()
        rel = res["mse"] / cleans[ds]["mse"]
        print(f"  {kind} {ds:8s} {mech}:{rate}:{m:16s} mse={res['mse']:10.4f} "
              f"rel={rel:8.3f} ({res['seconds']}s)", flush=True)
    print("SHARD DONE", flush=True)


# ------------------------------------------------------- gate / merge -------

def collect(kind, nshards):
    cells, cleans = {}, {}
    for i in range(nshards):
        c, cl = parse_log(os.path.join(HERE, f"s18_{kind}_shard{i}.log"))
        cells.update(c)
        cleans.update(cl)
    return cells, cleans


def gate(args):
    s8r = json.load(open(S8_RESULTS))
    cells, cleans = collect("anchor", args.nshards)
    expect = [(ds, mech, rate, m) for ds in DS for (mech, rate) in CELLS
              for m in ANCHOR_METHODS]
    missing = [j for j in expect if j not in cells]
    if missing:
        print(f"ANCHOR INCOMPLETE: {len(missing)}/{len(expect)} cells missing, "
              f"e.g. {missing[:3]}", flush=True)
        sys.exit(1)
    per_method = {}
    worst = (0.0, None)
    for (ds, mech, rate, m), cell in cells.items():
        ref = s8r["exp3"][ds][f"{mech}:{rate}:{m}"]["mse"]
        dev = abs(cell["mse"] / ref - 1.0)
        per_method[m] = max(per_method.get(m, 0.0), dev)
        if dev > worst[0]:
            worst = (dev, (ds, mech, rate, m))
    cdev = {ds: abs(cleans[ds]["mse"] / s8r["clean"][ds]["mse"] - 1.0)
            for ds in DS}
    print("== S18 anchor gate vs s8_results.json (max |dev| per method) ==",
          flush=True)
    for m in ANCHOR_METHODS:
        print(f"  {m:16s} max|dev| = {per_method[m] * 100:.4f}%", flush=True)
    print("  clean mse dev: " + ", ".join(f"{ds} {v * 100:.4f}%"
                                          for ds, v in cdev.items()), flush=True)
    ok = all(v <= 0.05 for v in per_method.values()) and \
        all(v <= 0.05 for v in cdev.values())
    print(f"worst cell {worst[1]} dev={worst[0] * 100:.4f}%", flush=True)
    print(f"ANCHOR {'PASS' if ok else 'FAIL'}", flush=True)
    if ok:
        write_results(cells, cleans,
                      gate={"per_method_max_dev": per_method,
                            "clean_dev": cdev, "worst": worst[0]})
    sys.exit(0 if ok else 1)


def write_results(cells, cleans, gate=None):
    out = {"meta": {
        "track": "S18 universal [MASK] token (cross-dataset corpus) on bolt",
        "seed": s5.SEED, "L": L, "H": H, "windows": 300,
        "seeds_mcar_block": 2, "cells": [f"{m}:{r}" for m, r in CELLS],
        "anchor_methods": ANCHOR_METHODS, "new_methods": NEW_METHODS,
        "seed_bases": list(SEED_BASES),
        "actual_seeds": [actual_seed(b) for b in SEED_BASES],
        "judgement": "see s18_notes.md (pre-registered)"}}
    if gate is not None:
        out["gate"] = gate
    out["clean"] = {ds: cleans[ds] for ds in DS if ds in cleans}
    exp3 = {}
    for (ds, mech, rate, m), cell in cells.items():
        exp3.setdefault(ds, {})[f"{mech}:{rate}:{m}"] = cell
    out["exp3"] = exp3
    train = {}
    for m in NEW_METHODS:
        if os.path.exists(TOK18[m]):
            sd = torch.load(TOK18[m], map_location="cpu")
            train[m] = {"meta": sd["meta"], "losses": sd["losses"]}
    if train:
        out["train"] = train
    s5.save_results(out, RESULTS)
    print(f"saved -> {RESULTS}", flush=True)


def merge(args):
    cells, cleans = collect("anchor", args.nshards)
    ecells, ecleans = collect("eval", args.nshards)
    cells.update(ecells)
    cleans.update(ecleans)
    write_results(cells, cleans)


# ------------------------------------------------------------- report -------

def rateavg(exp3, clean, ds, method):
    """Block rate-avg relMSE."""
    vals = [exp3[ds][f"block:{r}:{method}"]["mse"] / clean[ds]["mse"]
            for r in RATES3]
    return float(np.mean(vals))


def rel(exp3, clean, ds, mech, rate, method):
    return exp3[ds][f"{mech}:{rate}:{method}"]["mse"] / clean[ds]["mse"]


def report(args):
    r = json.load(open(RESULTS))
    exp3, clean = r["exp3"], r["clean"]
    uni = [f"uni_s{b}" for b in SEED_BASES]
    uniaug = [f"uniaug_s{b}" for b in SEED_BASES]

    print("== block rate-avg relMSE (transfer matrix; uni/uniaug mean+-sd "
          "over 3 seeds) ==", flush=True)
    hdr = ["test ds"] + ANCHOR_METHODS + ["uni", "uniaug"]
    print(("{:9s}" + "{:>11s}" * (len(hdr) - 1)).format(*hdr), flush=True)
    mat = {}
    for ds in DS:
        row = [rateavg(exp3, clean, ds, m) for m in ANCHOR_METHODS]
        uv = [rateavg(exp3, clean, ds, m) for m in uni]
        av = [rateavg(exp3, clean, ds, m) for m in uniaug]
        mat[ds] = {"anchor": dict(zip(ANCHOR_METHODS, row)),
                   "uni_seeds": uv, "uniaug_seeds": av,
                   "uni": [float(np.mean(uv)), float(np.std(uv))],
                   "uniaug": [float(np.mean(av)), float(np.std(av))]}
        print("{:9s}".format(ds)
              + "".join(f"{v:11.4f}" for v in row)
              + f"{np.mean(uv):7.4f}+{np.std(uv):.4f}"
              + f"{np.mean(av):7.4f}+-{np.std(av):.4f}", flush=True)

    print("\n== judgement (pre-registered) ==", flush=True)
    verdicts = {}
    for tag, grp in (("uni", uni), ("uniaug", uniaug)):
        crit_nan, crit_gap = {}, {}
        for ds in DS:
            mv = np.mean([rateavg(exp3, clean, ds, m) for m in grp])
            nan_v = rateavg(exp3, clean, ds, "nan")
            own_v = rateavg(exp3, clean, ds, f"tok_{ds}")
            crit_nan[ds] = bool(mv <= nan_v)
            crit_gap[ds] = bool(mv - own_v <= 0.02)
            print(f"  {tag:6s} {ds:8s} uni={mv:.4f} nan={nan_v:.4f} "
                  f"tok_own={own_v:.4f}  uni<=nan: {crit_nan[ds]}  "
                  f"gap={mv - own_v:+.4f} (<=0.02: {crit_gap[ds]})", flush=True)
        if all(crit_nan.values()) and all(crit_gap.values()):
            v = "SUCCESS"
        elif all(crit_nan.values()):
            v = "PARTIAL SUCCESS"
        else:
            v = "FAILURE"
        verdicts[tag] = {"verdict": v, "uni<=nan": crit_nan,
                         "gap<=0.02": crit_gap}
        print(f"  {tag}: {v}", flush=True)

    print("\n== no-gain check cells (uni/uniaug mean+-sd vs nan, tok_own) ==",
          flush=True)
    checks = {}
    for mech, rate in (("mcar", 0.3), ("mnar_high", 0.7)):
        for ds in DS:
            nan_v = rel(exp3, clean, ds, mech, rate, "nan")
            own_v = rel(exp3, clean, ds, mech, rate, f"tok_{ds}")
            uv = [rel(exp3, clean, ds, mech, rate, m) for m in uni]
            av = [rel(exp3, clean, ds, mech, rate, m) for m in uniaug]
            checks[f"{mech}:{rate}"] = checks.get(f"{mech}:{rate}", {})
            checks[f"{mech}:{rate}"][ds] = {
                "nan": nan_v, "tok_own": own_v,
                "uni": [float(np.mean(uv)), float(np.std(uv))],
                "uniaug": [float(np.mean(av)), float(np.std(av))]}
            print(f"  {mech}:{rate} {ds:8s} nan={nan_v:.4f} own={own_v:.4f} "
                  f"uni={np.mean(uv):.4f}+-{np.std(uv):.4f} "
                  f"uniaug={np.mean(av):.4f}+-{np.std(av):.4f}", flush=True)

    print("\n== uni - nan, per ds (95% t-CI over 3 seeds, t=4.303) ==",
          flush=True)
    cis = {}
    for ds in DS:
        diffs = np.array([rateavg(exp3, clean, ds, m)
                          for m in uni]) - rateavg(exp3, clean, ds, "nan")
        gaps = np.array([rateavg(exp3, clean, ds, m)
                         for m in uni]) - rateavg(exp3, clean, ds, f"tok_{ds}")
        for tag, arr in (("uni-nan", diffs), ("uni-own", gaps)):
            mu, sd = arr.mean(), arr.std(ddof=1)
            half = 4.303 * sd / math.sqrt(3)
            cis[f"{tag}:{ds}"] = [float(mu), float(mu - half), float(mu + half)]
            print(f"  {tag:8s} {ds:8s} mean={mu:+.4f} sd={sd:.4f} "
                  f"CI=[{mu - half:+.4f}, {mu + half:+.4f}]", flush=True)

    r["report"] = {"matrix": mat, "verdicts": verdicts, "checks": checks,
                   "cis": cis}
    s5.save_results(r, RESULTS)

    # ------------------------------------------------------------ figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    methods_bar = ANCHOR_METHODS + ["uni", "uniaug"]
    colors = {"nan": "#888888", "linear": "#bbbbbb", "mponly": "#aaaaaa",
              "tok_ETTh1": "#1f77b4", "tok_ETTm1": "#2ca02c",
              "tok_weather": "#9467bd", "uni": "#d62728", "uniaug": "#ff7f0e"}
    for j, ds in enumerate(DS):
        ax = axes[0, j]
        vals, errs = [], []
        for m in methods_bar:
            if m == "uni":
                v = [rateavg(exp3, clean, ds, x) for x in uni]
                vals.append(np.mean(v)); errs.append(np.std(v))
            elif m == "uniaug":
                v = [rateavg(exp3, clean, ds, x) for x in uniaug]
                vals.append(np.mean(v)); errs.append(np.std(v))
            else:
                vals.append(rateavg(exp3, clean, ds, m)); errs.append(0.0)
        ax.bar(range(len(methods_bar)), vals, yerr=errs, capsize=3,
               color=[colors[m] for m in methods_bar])
        ax.axhline(vals[0], color="#888888", ls="--", lw=1)
        own = rateavg(exp3, clean, ds, f"tok_{ds}")
        ax.axhline(own, color=colors[f"tok_{ds}"], ls=":", lw=1)
        ax.set_xticks(range(len(methods_bar)))
        ax.set_xticklabels([m.replace("tok_", "t_") for m in methods_bar],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(f"test {ds} -- block rate-avg relMSE")
        if j == 0:
            ax.set_ylabel("relMSE")
        ax.grid(axis="y", alpha=0.3)

        ax = axes[1, j]
        nan_r = [rel(exp3, clean, ds, "block", p, "nan") for p in RATES3]
        lin_r = [rel(exp3, clean, ds, "block", p, "linear") for p in RATES3]
        own_r = [rel(exp3, clean, ds, "block", p, f"tok_{ds}") for p in RATES3]
        uu = np.array([[rel(exp3, clean, ds, "block", p, m) for p in RATES3]
                       for m in uni])
        aa = np.array([[rel(exp3, clean, ds, "block", p, m) for p in RATES3]
                       for m in uniaug])
        ax.plot(RATES3, nan_r, "o-", color="#888888", label="nan")
        ax.plot(RATES3, lin_r, "s:", color="#bbbbbb", label="linear")
        ax.plot(RATES3, own_r, "^-", color=colors[f"tok_{ds}"],
                label=f"tok_own ({ds})")
        ax.plot(RATES3, uu.mean(0), "o-", color=colors["uni"], label="uni")
        ax.fill_between(RATES3, uu.mean(0) - uu.std(0), uu.mean(0) + uu.std(0),
                        color=colors["uni"], alpha=0.2)
        ax.plot(RATES3, aa.mean(0), "o--", color=colors["uniaug"],
                label="uniaug")
        ax.fill_between(RATES3, aa.mean(0) - aa.std(0), aa.mean(0) + aa.std(0),
                        color=colors["uniaug"], alpha=0.2)
        ax.axhline(1.0, color="k", lw=0.5, alpha=0.4)
        ax.set_xlabel("block missing rate p")
        if j == 0:
            ax.set_ylabel("relMSE")
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
    fig.suptitle("S18 universal [MASK] token vs S8 in-domain tokens "
                 "(block missingness; uni/uniaug = mean+/-sd over seeds "
                 "101/202/303)")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "s18.png"), dpi=150)
    print("saved -> s18.png", flush=True)


# --------------------------------------------------------------- tiny -------

def run_tiny(args):
    device = "cuda"
    pipe = s8.load_bolt(device)
    model = pipe.model
    X, starts = s5.load_windows(s5.DATASETS["ETTh1"], 6, s5.SEED)
    ctx, msk, gt, S, C, nw, ns = s8.build_contexts(X, starts, "block", 0.5)
    x_nan = ctx.copy()
    x_nan[msk] = np.nan
    # 1) replica 'correct' == native predictions
    p_ref = s8.predict_median(pipe, x_nan, batch=512)
    with s8.EncodePatch(model, mask_mode="correct"):
        p_rep = s8.predict_median(pipe, x_nan, batch=512)
    d = np.abs(p_ref - p_rep).max()
    print(f"tiny: replica-vs-native max|diff| = {d:.3e} (want <1e-4)", flush=True)
    assert d < 1e-4
    # 2) mixed sampler: exact thirds, in-range, deterministic per seed
    data = load_train_data()
    for variant in VARIANTS:
        r1, a1 = np.random.default_rng(132), np.random.default_rng(132 + AUG_STREAM_OFFSET)
        c1, f1 = mix_batch(r1, a1, data, BATCH, variant)
        r2, a2 = np.random.default_rng(132), np.random.default_rng(132 + AUG_STREAM_OFFSET)
        c2, f2 = mix_batch(r2, a2, data, BATCH, variant)
        assert np.array_equal(c1, c2) and np.array_equal(f1, f2)
        assert np.isfinite(c1).all() and np.isfinite(f1).all()
        print(f"tiny: mix_batch {variant} deterministic, shapes {c1.shape}/{f1.shape}",
              flush=True)
    ru, au = np.random.default_rng(132), np.random.default_rng(132 + AUG_STREAM_OFFSET)
    cu, fu = mix_batch(ru, au, data, BATCH, "uni")
    rg, ag = np.random.default_rng(132), np.random.default_rng(132 + AUG_STREAM_OFFSET)
    cg, fg = mix_batch(rg, ag, data, BATCH, "uniaug")
    same = np.mean(np.all(cu == cg, axis=1))
    print(f"tiny: uni-vs-uniaug identical rows: {same:.2f} (perturbation active, "
          f"some rows must differ)", flush=True)
    assert 0.0 < same < 1.0
    # 3) scale jitter cancels under instance norm (design caveat check)
    w = cu[0]
    f = 3.7
    n1 = (w - w.mean()) / w.std()
    n2 = (f * w - (f * w).mean()) / (f * w).std()
    print(f"tiny: scale-invariance of instance norm max|diff| = "
          f"{np.abs(n1 - n2).max():.2e} (expected ~1e-7 -> scale jitter inert)",
          flush=True)
    # 4) train smoke: 5 steps both variants, grad flows, ckpt save/load
    args.steps, args.batch, args.log_every = 5, 32, 1
    args.out_ckpt = os.path.join(HERE, "s18_unitoken_smoke_tmp.pt")
    for variant in VARIANTS:
        args.variant, args.seed_base = variant, 101
        train(args)
        sd = torch.load(args.out_ckpt, map_location="cpu")
        nrm = sd["mask_token"].norm().item()
        assert nrm > 0 and np.isfinite(nrm)
        print(f"tiny: train {variant} ok, |tok|={nrm:.4f}", flush=True)
    os.remove(args.out_ckpt)
    # 5) eval dispatch incl. token path (zero token on one cell)
    tok0 = torch.zeros(model.config.d_model, device=device)
    res = run_cell(pipe, {"dbg": tok0}, None, X, starts, "ETTh1", "block", 0.3, "dbg")
    res2 = run_cell(pipe, {}, None, X, starts, "ETTh1", "block", 0.3, "nan")
    print(f"tiny: eval zero-token mse={res['mse']:.4f} vs nan mse={res2['mse']:.4f} "
          f"(both finite, different paths)", flush=True)
    print("TINY OK", flush=True)


# ---------------------------------------------------------------- main ------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--anchor", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--variant", choices=list(VARIANTS), default="uni")
    ap.add_argument("--seed-base", type=int, default=101)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--loss-clip", type=float, default=LOSS_CLIP)
    ap.add_argument("--log-every", type=int, default=LOG_EVERY)
    ap.add_argument("--windows", type=int, default=300)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=3)
    ap.add_argument("--out-ckpt", default=None)
    args = ap.parse_args()

    torch.manual_seed(s5.SEED)
    np.random.seed(s5.SEED)

    if args.tiny:
        run_tiny(args)
    elif args.anchor:
        shard_main(args, ANCHOR_METHODS, "anchor")
    elif args.train:
        train(args)
    elif args.eval:
        shard_main(args, NEW_METHODS, "eval")
    elif args.gate:
        gate(args)
    elif args.merge:
        merge(args)
    elif args.report:
        report(args)
    else:
        ap.error("nothing to do")


if __name__ == "__main__":
    main()
