#!/usr/bin/env python
"""S45 evaluation: the paper's own harness on the five pretrained arms.

Per arm (A/B/C/D/E), on held-out benchmark windows (the datasets are zero-shot for every
arm -- none was in the corpus):
  clean   clean-context relMSE on the 9 benchmark datasets (G3 reports arm A vs stock bolt-tiny)
  probe   permutation rho under plain / declared / nan (S30/S36 protocol, 3 ds x 3 mechs x 60 win)
  sweep   imputation-quality (alpha) sweep on the declared path, 4 mechs x 2 rates, 150 win/cell
  attack  gradient-free range-constrained attack through the declared path (S33 protocol)

Conventions per arm: "plain" = fill, no flag; "declared" = the interface the arm was
PRETRAINED with (native for A/D, dual for B/C/E) plus, for reference, the dual view of A/D
and the native view of B/C/E; "nan" = declared absent.

Gates: G1 reproduces stored stock bolt-tiny probe cells (S36) before any arm is scored.
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

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s27_interface"))
sys.path.insert(0, os.path.join(EXP, "s35_breadth"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import run_s27_interface as s27
import run_s35_breadth as s35
import run_s30_crossmodel as s30mod
import train_s45 as t45

L, H = s25.L, 64
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES = (0.3, 0.7)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
DS9 = ("ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness")
CK = os.path.join(HERE, "s45_ckpt")
OUT = os.path.join(HERE, "s45_eval.json")

ARM_IFACE = {"A": "native", "B": "dual", "C": "dual", "D": "native", "E": "dual",
             "F": "dual", "G": "dual", "H": "dual", "I": "dual",
             # S53 (CPT-from-stock): P0 stock as shipped, P4 the native budget control
             "P0": "native", "P1": "dual", "P2": "dual", "P3": "dual",
             "P4": "native", "P5": "dual",
             # S58 (WiSE-FT): P0/P5 interpolates, evaluated through the dual interface
             "W30": "dual", "W50": "dual", "W70": "dual", "W85": "dual",
             # S59 fix arm: H + declaration endpoint
             "P6": "dual",
             # exploration arms
             "P7": "dual", "P8": "dual", "P9": "dual",
             # P8 x WiSE-FT interpolates
             "Q30": "dual", "Q50": "dual", "Q70": "dual",
             # P10: mechdiv CPT + clean-anchor distillation
             "P10": "dual"}


def load_arm(arm, dev, size="tiny"):
    """arm is a bare letter, or "<letter>@<seed>" for a replicate checkpoint."""
    letter, _, seed = arm.partition("@")
    m = t45.build_model(dev, size)
    # "<letter>@<suffix>" resolves a replicate seed (s20260901) or a recipe tag (r50k)
    cands = ([f"arm_{letter}_{size}_s{seed}.pt", f"arm_{letter}_{size}_{seed}.pt"] if seed else
             [f"arm_{letter}_{size}.pt", f"arm_{letter}.pt"])
    for cand in cands:
        p = os.path.join(CK, cand)
        if os.path.exists(p):
            sd = torch.load(p, map_location=dev, weights_only=True)
            break
    else:
        raise FileNotFoundError(f"no checkpoint for arm {arm} ({size}): tried {cands}")
    m.load_state_dict(sd)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def median_fwd(model, ctx_np, obs_np, iface, batch=256):
    outs = []
    for i in range(0, len(ctx_np), batch):
        c = torch.from_numpy(ctx_np[i:i + batch]).to(next(model.parameters()).device)
        o = torch.from_numpy(obs_np[i:i + batch]).to(next(model.parameters()).device)
        q = t45.quantiles_fwd(model, c, o, iface)
        outs.append(q[:, 4, :H].float().cpu().numpy())
    return np.concatenate(outs)


def conv_ctx(ctx, miss, conv, iface_declared):
    """(ctx_to_feed, obs_to_feed) under a convention."""
    if conv == "plain":
        return ctx, np.ones_like(ctx)
    if conv == "declared":
        return ctx, (~miss).astype(np.float32)
    if conv == "nan":
        c = ctx.copy()
        c[miss] = np.nan
        return c, (~np.isnan(c)).astype(np.float32)   # the stock NaN path: obs = ~isnan
    raise ValueError(conv)


def blend(clean, lin, mask, a):
    out = lin.copy()
    out[mask] = (1.0 - a) * lin[mask] + a * clean[mask]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="g1,clean,probe,sweep,attack,g3")
    ap.add_argument("--arms", default="A,B,C,D,E")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-win", type=int, default=150)
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", default="tiny", choices=["tiny", "small", "base"])
    args = ap.parse_args()
    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    dev = args.device
    out_path = args.out or os.path.join(HERE, "s45_eval.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    res.setdefault("meta", {"arms": ARM_IFACE, "H": H, "alphas": list(ALPHAS),
                            "mechs": list(MECHS), "rates": list(RATES),
                            "ds9": DS9, "n_win": args.n_win})

    # -------------------------------------------------------------- G1 ----
    if "g1" in args.parts:
        # the eval harness must reproduce stock bolt-TINY's stored S36 probe values first
        class _Tiny(s30mod.Bolt):
            def __init__(self, dev):
                from chronos import BaseChronosPipeline as B
                self.name = "chronos-bolt-tiny"
                self.convs = ("plain", "mask", "nan")
                self.p = B.from_pretrained(os.path.join(ROOT, "models_local",
                                                        "chronos-bolt-tiny"),
                                           device_map=dev, torch_dtype=torch.float32)
                self.m = self.p.inner_model if hasattr(self.p, "inner_model") else self.p.model
                self.dev = dev
        tiny = _Tiny(dev)
        stored = json.load(open(os.path.join(EXP, "s36_models", "part_0.json"))
                           )["models"]["bolt_tiny"]["cells"]
        gate, worst = {}, 0.0
        for ds, path in s25.DATASETS.items():
            X, st = s25.load_windows(path, 60, s25.SEED, H)
            for mech in ("mcar", "block", "mnar_high"):
                _, lin, mask, _ = s25.build_eval_batch(X, st, mech, 0.3, H, fill="linear")
                rng = np.random.default_rng(s25.SEED)
                r = s30mod.perm_probe(tiny, lin, mask, "plain", rng)
                ref = stored[f"{ds}|{mech}|plain"]["ratio"]
                dev_abs = abs(r["ratio"] - ref)
                worst = max(worst, dev_abs)
                gate[f"{ds}|{mech}"] = {"mine": r["ratio"], "stored": ref,
                                        "abs_dev": dev_abs}
        res["g1"] = {"cells": gate, "max_abs_dev": worst, "pass": worst <= 0.05}
        print(f"GATE G1 stock bolt-tiny plain rho over 9 cells, max dev {worst:.3f} "
              f"{'PASS' if res['g1']['pass'] else 'FAIL'}", flush=True)
        assert res["g1"]["pass"], "G1 FAILED"
        del tiny
        torch.cuda.empty_cache()

    arms = {a: load_arm(a, dev, args.size) for a in args.arms.split(",")}
    print("loaded arms:", list(arms), flush=True)

    # ------------------------------------------------------------ clean ----
    if "clean" in args.parts:
        for ds in DS9:
            X, st = s35.load(ds, args.n_win)
            if X is None:
                print(f"  clean {ds}: no valid windows, skipped", flush=True)
                continue
            _, cl, _, gt = s25.build_eval_batch(X, st, "clean", 0.0, H)
            obs = np.ones_like(cl)
            for a, m in arms.items():
                e = ((median_fwd(m, cl, obs, ARM_IFACE[a.partition("@")[0]]) - gt) ** 2).mean(1)
                res.setdefault("clean", {}).setdefault(a, {})[ds] = {
                    "mse_median": float(np.median(e)), "mse_mean": float(e.mean())}
            print(f"  clean {ds:11s} " + " ".join(
                f"{a}:{res['clean'][a][ds]['mse_median']:.4g}" for a in arms), flush=True)
            json.dump(res, open(out_path, "w"))

    # ------------------------------------------------------------- probe ----
    if "probe" in args.parts:
        for ds, path in s25.DATASETS.items():
            X, st = s25.load_windows(path, 60, s25.SEED, H)
            for mech in ("mcar", "block", "mnar_high"):
                _, lin, mask, _ = s25.build_eval_batch(X, st, mech, 0.3, H, fill="linear")
                for a, m in arms.items():
                    for conv in ("plain", "declared", "nan"):
                        rng = np.random.default_rng(s25.SEED)
                        iface = ARM_IFACE[a.partition("@")[0]] if conv == "declared" else (
                            "native" if conv == "nan" else "plain")
                        # nan conv feeds NaN through the native stock path
                        r = perm_probe_iface(m, lin, mask, conv, iface, rng)
                        res.setdefault("probe", {})[f"{ds}|{mech}|{a}|{conv}"] = r
                print(f"  probe {ds:8s} {mech:10s} " + " ".join(
                    f"{a}.{c}:{res['probe'][f'{ds}|{mech}|{a}|{c}']['rho']:.3f}"
                    for a in arms for c in ("plain", "declared")), flush=True)
                json.dump(res, open(out_path, "w"))

    # ------------------------------------------------------------- sweep ----
    if "sweep" in args.parts:
        for ds, path in s25.DATASETS.items():
            X, st = s25.load_windows(path, args.n_win, s25.SEED, H)
            _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
            obs_c = np.ones_like(cl)
            base = {a: ((median_fwd(m, cl, obs_c, ARM_IFACE[a.partition("@")[0]]) - gt_c) ** 2).mean(1)
                    for a, m in arms.items()}
            for mech in MECHS:
                for rate in RATES:
                    clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, rate, H,
                                                                fill="linear")
                    for al in ALPHAS:
                        ctx = blend(clean, lin, mask, al)
                        for a, m in arms.items():
                            for view in ("declared", "plain"):
                                conv = view
                                iface = ARM_IFACE[a.partition("@")[0]] if view == "declared" else "plain"
                                cx, ob = conv_ctx(ctx, mask, conv, iface)
                                e = ((median_fwd(m, cx, ob, iface) - gt) ** 2).mean(1)
                                ok = base[a] > 1e-12
                                r = e[ok] / base[a][ok]
                                res.setdefault("sweep", {})[
                                    f"{ds}|{mech}|{rate}|{al}|{a}|{view}"] = {
                                    "median": float(np.median(r)), "mean": float(r.mean()),
                                    # per-window ratios, kept so closure can be bootstrapped
                                    # paired on windows rather than only summarised
                                    "w": [round(float(x), 6) for x in r]}
                    print(f"  sweep {ds:8s} {mech:13s} p={rate} " + " ".join(
                        f"{a}:{res['sweep'][f'{ds}|{mech}|{rate}|0.0|{a}|declared']['median']:.2f}"
                        f"->{res['sweep'][f'{ds}|{mech}|{rate}|1.0|{a}|declared']['median']:.2f}"
                        for a in arms), flush=True)
                    json.dump(res, open(out_path, "w"))

    # ----------------------------------------------------------- sweep9 ----
    # main-table grid: all 9 benchmarks x 4 mechs at rate 0.7, alpha in {0, 1}
    if "sweep9" in args.parts:
        for ds in DS9:
            X, st = s35.load(ds, args.n_win)
            if X is None:
                print(f"  sweep9 {ds}: no valid windows, skipped", flush=True)
                continue
            _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
            obs_c = np.ones_like(cl)
            base = {a: ((median_fwd(m, cl, obs_c, ARM_IFACE[a.partition("@")[0]]) - gt_c) ** 2).mean(1)
                    for a, m in arms.items()}
            for mech in MECHS:
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, 0.7, H,
                                                            fill="linear")
                for al in (0.0, 1.0):
                    ctx = blend(clean, lin, mask, al)
                    for a, m in arms.items():
                        for view in ("declared", "plain"):
                            iface = ARM_IFACE[a.partition("@")[0]] if view == "declared" else "plain"
                            cx, ob = conv_ctx(ctx, mask, view, iface)
                            e = ((median_fwd(m, cx, ob, iface) - gt) ** 2).mean(1)
                            ok = base[a] > 1e-12
                            r = e[ok] / base[a][ok]
                            res.setdefault("sweep", {})[f"{ds}|{mech}|0.7|{al}|{a}|{view}"] = {
                                "median": float(np.median(r)), "mean": float(r.mean()),
                                "w": [round(float(x), 6) for x in r]}
                print(f"  sweep9 {ds:11s} {mech:13s} " + " ".join(
                    f"{a}:{res['sweep'][f'{ds}|{mech}|0.7|0.0|{a}|declared']['median']:.2f}"
                    f"->{res['sweep'][f'{ds}|{mech}|0.7|1.0|{a}|declared']['median']:.2f}"
                    for a in arms), flush=True)
                json.dump(res, open(out_path, "w"))

    # ------------------------------------------------------- sweep9fills ----
    # any-fill grid: 9 benchmarks x 4 mechs at rate 0.7 x {zero, ffill, linear,
    # oracle} on the declared path -- the "no matter the fill, never worse" claim
    if "sweep9fills" in args.parts:
        for ds in DS9:
            X, st = s35.load(ds, args.n_win)
            if X is None:
                print(f"  fills {ds}: no valid windows, skipped", flush=True)
                continue
            _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
            obs_c = np.ones_like(cl)
            base = {a: ((median_fwd(m, cl, obs_c, ARM_IFACE[a.partition("@")[0]]) - gt_c) ** 2).mean(1)
                    for a, m in arms.items()}
            for mech in MECHS:
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, 0.7, H,
                                                            fill="linear")
                for fill in ("zero", "ffill", "linear", "oracle"):
                    ctx = s25.fill_context(clean, mask, fill)
                    for a, m in arms.items():
                        iface = ARM_IFACE[a.partition("@")[0]]
                        cx, ob = conv_ctx(ctx, mask, "declared", iface)
                        e = ((median_fwd(m, cx, ob, iface) - gt) ** 2).mean(1)
                        ok = base[a] > 1e-12
                        r = e[ok] / base[a][ok]
                        res.setdefault("fills", {})[
                            f"{ds}|{mech}|0.7|{fill}|{a}|declared"] = {
                            "median": float(np.median(r)), "mean": float(r.mean()),
                            "w": [round(float(x), 6) for x in r]}
                print(f"  fills {ds:11s} {mech:13s} " + " ".join(
                    f"{a}.z:{res['fills'][f'{ds}|{mech}|0.7|zero|{a}|declared']['median']:.2f}"
                    f"->o:{res['fills'][f'{ds}|{mech}|0.7|oracle|{a}|declared']['median']:.2f}"
                    for a in arms), flush=True)
                json.dump(res, open(out_path, "w"))

    # ------------------------------------------------------------ attack ----
    if "attack" in args.parts:
        for ds, path in s25.DATASETS.items():
            X, st = s25.load_windows(path, 60, s25.SEED, H)
            for mech in ("block", "mnar_high"):
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, 0.5, H,
                                                            fill="linear")
                for a, m in arms.items():
                    iface = ARM_IFACE[a.partition("@")[0]]
                    e0, w = rand_attack(m, iface, lin, mask, gt)
                    ratio = w / np.maximum(e0, 1e-12)
                    res.setdefault("attack", {})[f"{ds}|{mech}|{a}"] = {
                        "e0_median": float(np.median(e0)),
                        "ratio_median": float(np.median(ratio)),
                        "ratio_q90": float(np.quantile(ratio, 0.9))}
                print(f"  attack {ds:8s} {mech:10s} " + " ".join(
                    f"{a}:x{res['attack'][f'{ds}|{mech}|{a}']['ratio_median']:.2f}"
                    for a in arms), flush=True)
                json.dump(res, open(out_path, "w"))

    # --------------------------------------------------------------- G3 ----
    if "g3" in args.parts and "clean" in res and "A" in res["clean"]:
        stock = s25.Bolt(dev)   # NOTE: bolt-base, documented against arm A openly
        e_stock = {}
        for ds in DS9:
            X, st = s35.load(ds, args.n_win)
            if X is None:
                continue
            _, cl, _, gt = s25.build_eval_batch(X, st, "clean", 0.0, H)
            e_stock[ds] = float(np.median(((stock.median_np(cl) - gt) ** 2).mean(1)))
        res["g3"] = {"stock_bolt_base_clean_median": e_stock,
                     "note": "arm A vs STOCK bolt-base documents the corpus difference; "
                             "stock bolt-tiny was not used (the family checkpoint differs)"}
        json.dump(res, open(out_path, "w"))
    print("\nwrote", out_path)


def perm_probe_iface(model, filled, mask, conv, iface, rng, n_perm=4):
    cx0, ob0 = conv_ctx(filled, mask, conv, iface)
    y0 = median_fwd(model, cx0, ob0, iface if conv != "nan" else "native")
    scale = np.sqrt((y0 ** 2).mean(1)) + 1e-9
    dp, dr = [], []
    for _ in range(n_perm):
        cp, cr = filled.copy(), filled.copy()
        for i in range(len(filled)):
            idx = np.flatnonzero(mask[i])
            if len(idx) < 2:
                continue
            cp[i, idx] = filled[i, idx][rng.permutation(len(idx))]
            src = filled[i, ~mask[i]]
            cr[i, idx] = rng.choice(src, size=len(idx)) if len(src) else 0.0
        for cc, dd in ((cp, dp), (cr, dr)):
            cx, ob = conv_ctx(cc, mask, conv, iface)
            y = median_fwd(model, cx, ob, iface if conv != "nan" else "native")
            dd.append(np.sqrt(((y - y0) ** 2).mean(1)) / scale)
    dp, dr = np.mean(dp, 0), np.mean(dr, 0)
    return {"rho": float(dp.mean() / max(dr.mean(), 1e-15)),
            "perm": float(dp.mean()), "redraw": float(dr.mean())}


def rand_attack(model, iface, ctx, mask, gt, n_try=400, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([np.nanmin(c[~mk]) for c, mk in zip(ctx, mask)], np.float32)[:, None]
    hi = np.array([np.nanmax(c[~mk]) for c, mk in zip(ctx, mask)], np.float32)[:, None]
    obs = (~mask).astype(np.float32)
    e0 = ((median_fwd(model, ctx, obs, iface) - gt) ** 2).mean(1)
    worst = e0.copy()
    B = 20
    for _ in range(n_try // B):
        cand = ctx.copy()
        prop = lo + rng.random(ctx.shape).astype(np.float32) * (hi - lo)
        cand[mask] = prop[mask]
        e = ((median_fwd(model, cand, obs, iface) - gt) ** 2).mean(1)
        worst = np.maximum(worst, e)
    return e0, worst


if __name__ == "__main__":
    main()
