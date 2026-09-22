#!/usr/bin/env python
"""S33: the causal test of the interface claim, using Moirai as the natural experiment.

Everything in S27/S29/S31 is consistent with "the input interface caps what can be repaired",
but all of it was measured on Chronos-Bolt. A sceptic can answer: maybe missingness is simply
irreducibly hard, and the interface is a coincidence.

S30 supplies the discriminating case. Across five families, four discard or overwrite the
fill content once a hole is declared (permutation ratio 0.0000), but **Moirai does not**: with
its observed-mask set, permuting the fill content still moves the forecast (ratio 0.9739). If
the cap is caused by the interface, then the same imputation-quality sweep that is FLAT for
Chronos-Bolt's declared path must be DECREASING for Moirai's -- same task, same data, same
masks, different interface. That is a prediction the interface hypothesis makes and the
"missingness is just hard" hypothesis does not.

Sweep: fill = (1-alpha)*linear + alpha*truth, alpha in {0, 0.25, 0.5, 0.75, 1}. alpha=1 is a
perfect imputer. Gradient-free, so it works on any family.

Arms:
  moirai|declared   observed-mask marks the holes, values still supplied  <- the test
  moirai|plain      no mask, values supplied
  bolt|declared     explicit mask (rank 2)                                <- the flat control
  bolt|plain        no mask (full rank)                                   <- the steep control

Secondary: a gradient-free random-search attack through each declared path, to check S29's
"attack surface = rank" ordering on a family whose declared path is NOT rank-deficient.
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
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
import run_s25_twofloor as s25
import run_s30_crossmodel as s30

H = 64
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
MECHS = ("mcar", "block", "mnar_high")
RATES = (0.3, 0.7)
OUT = os.path.join(HERE, "s33_results.json")


def blend(clean, lin, mask, a):
    out = lin.copy()
    out[mask] = (1.0 - a) * lin[mask] + a * clean[mask]
    return out


def sweep(res, models, n_win=100):
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, n_win, s25.SEED, H)
        clean_ctx, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
        base = {}
        for name, (m, convs) in models.items():
            b = ((m.fc(cl, np.zeros_like(cl, bool), convs["plain"]) - gt_c) ** 2).mean(1)
            base[name] = b
        for mech in MECHS:
            for rate in RATES:
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, rate, H, fill="linear")
                for a in ALPHAS:
                    ctx = blend(clean, lin, mask, a)
                    for name, (m, convs) in models.items():
                        for arm, conv in convs.items():
                            e = ((m.fc(ctx, mask, conv) - gt) ** 2).mean(1)
                            ok = base[name] > 1e-12
                            r = e[ok] / base[name][ok]
                            res.setdefault("sweep", {})[
                                f"{ds}|{mech}|{rate}|{a}|{name}|{arm}"] = {
                                "median": float(np.median(r)), "mean": float(r.mean())}
                print(f"  {ds:8s} {mech:10s} p={rate} " + "  ".join(
                    f"{n}.{arm}:" + "/".join(
                        f"{res['sweep'][f'{ds}|{mech}|{rate}|{a}|{n}|{arm}']['median']:.2f}"
                        for a in (0.0, 1.0))
                    for n in models for arm in models[n][1]), flush=True)
    return res


def rand_attack(m, conv, ctx, mask, gt, n_try=400, seed=0):
    """Gradient-free range-constrained attack: sample fills inside the observed [min,max],
    keep the worst. Works on any family, including ones we cannot differentiate through."""
    rng = np.random.default_rng(seed)
    lo = np.array([np.nanmin(c[~mk]) for c, mk in zip(ctx, mask)], np.float32)[:, None]
    hi = np.array([np.nanmax(c[~mk]) for c, mk in zip(ctx, mask)], np.float32)[:, None]
    e0 = ((m.fc(ctx, mask, conv) - gt) ** 2).mean(1)
    worst = e0.copy()
    B = 20
    for _ in range(n_try // B):
        cand = ctx.copy()
        u = rng.random(ctx.shape).astype(np.float32)
        prop = lo + u * (hi - lo)
        cand[mask] = prop[mask]
        e = ((m.fc(cand, mask, conv) - gt) ** 2).mean(1)
        worst = np.maximum(worst, e)
    return e0, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-win", type=int, default=100)
    ap.add_argument("--attack-tries", type=int, default=400)
    ap.add_argument("--parts", default="sweep,attack")
    args = ap.parse_args()
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"alphas": list(ALPHAS), "mechs": list(MECHS),
                            "rates": list(RATES), "H": H, "n_win": args.n_win,
                            "claim": "if the cap is caused by the interface, moirai's "
                                     "declared path must NOT be flat in alpha"})
    moirai = s30.Moirai("cuda")
    bolt = s30.Bolt("cuda")
    models = {"moirai": (moirai, {"plain": "plain", "declared": "nan"}),
              "bolt": (bolt, {"plain": "plain", "declared": "mask"})}
    if "sweep" in args.parts.split(","):
        print("== imputation-quality sweep (median relMSE at alpha=0 / alpha=1) ==", flush=True)
        res = sweep(res, models, n_win=args.n_win)
        json.dump(res, open(OUT + ".tmp", "w"))
        os.replace(OUT + ".tmp", OUT)
    if "attack" in args.parts.split(","):
        print("\n== gradient-free range-constrained attack through the DECLARED path ==",
              flush=True)
        for ds, path in s25.DATASETS.items():
            X, st = s25.load_windows(path, min(args.n_win, 60), s25.SEED, H)
            for mech in ("block", "mnar_high"):
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, 0.5, H, fill="linear")
                for name, (m, convs) in models.items():
                    t0 = time.time()
                    e0, w = rand_attack(m, convs["declared"], lin, mask, gt,
                                        n_try=args.attack_tries)
                    ratio = w / np.maximum(e0, 1e-12)
                    res.setdefault("attack", {})[f"{ds}|{mech}|{name}"] = {
                        "damage_median": float(np.median(ratio)),
                        "damage_q90": float(np.quantile(ratio, 0.9)),
                        "damage_max": float(ratio.max()),
                        "frac_doubled": float((ratio > 2).mean())}
                    c = res["attack"][f"{ds}|{mech}|{name}"]
                    print(f"  {ds:8s} {mech:10s} {name:7s} declared-path damage "
                          f"x{c['damage_median']:.2f} (q90 x{c['damage_q90']:.2f}, "
                          f"max x{c['damage_max']:.1f}) >2x on "
                          f"{100*c['frac_doubled']:.0f}% ({time.time()-t0:.0f}s)", flush=True)
        json.dump(res, open(OUT + ".tmp", "w"))
        os.replace(OUT + ".tmp", OUT)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
