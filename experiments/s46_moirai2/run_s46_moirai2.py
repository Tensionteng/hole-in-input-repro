#!/usr/bin/env python
"""S46: Moirai 2.0 as the retaining family -- probe, library check, causal sweep.

The paper's natural experiment (S33) used Moirai 1.1, whose declared path retains the fill
content. Moirai 2.0 (Salesforce, Aug 2025) is the current generation: at the module level
its input tokens are `cat([scaled_target, observed_mask])` -- fill content AND missingness
flag into the content path together, i.e. the deployed instance of the paper's restored
interface (with a single scattered patch-mask training mechanism). This round re-runs the
measurements that make it the paper's retaining family:

  P1 rho probe (S30 protocol): plain vs declared, 3 datasets x 3 mechs x 60 windows.
     Expectation: declared rho high (content retained), unlike every (a)-convention model.
  V1 library check: our direct forward call vs the library's create_predictor path on
     identical clean windows (mirror of S33's V1 for 1.1; 2.0 emits quantiles directly,
     so agreement should be exact, not within sampling noise).
  S1 causal alpha-sweep (S33 design): moirai2 declared/plain vs bolt declared/plain,
     fill = (1-a)*linear + a*truth, 3 datasets x 3 mechs x 2 rates, 150 windows/cell.
     The interface-as-gate prediction: moirai2's declared path decreases with alpha,
     bolt's stays flat.
  A1 gradient-free range-constrained attack through the declared path (S33's A-arm):
     a retaining declared path should be attackable -- the trade-off.
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

L, H = s25.L, 64
MECHS = ("mcar", "block", "mnar_high")
RATES = (0.3, 0.7)
RATE_PROBE = 0.3
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_PERM = 4
OUT = os.path.join(HERE, "s46_results.json")
LOCAL = os.path.join(ROOT, "models_local")


class Moirai2:
    """Moirai 2.0 R-small. conv 'nan' = declared: fill content + observed mask both
    supplied (the model's input tokens are cat([scaled_target, observed_mask]))."""
    name = "moirai-2.0-R-small"
    convs = ("plain", "nan")

    def __init__(self, dev):
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        path = os.path.join(LOCAL, "moirai-2.0-R-small")
        self.mod = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(path),
            prediction_length=H, target_dim=1, feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0, context_length=L).to(dev)
        self.mod.eval()
        self.dev = dev

    @torch.no_grad()
    def fc(self, ctx, miss, conv, batch=64):
        outs = []
        for i in range(0, len(ctx), batch):
            c = ctx[i:i + batch].copy()
            mk = miss[i:i + batch]
            obs = np.ones_like(c, bool)
            if conv == "nan":
                obs = ~mk
                c = np.nan_to_num(c)
            pt = torch.from_numpy(c).to(self.dev).unsqueeze(-1)
            om = torch.from_numpy(obs).to(self.dev).unsqueeze(-1)
            pad = torch.zeros(pt.shape[:2], dtype=torch.bool, device=self.dev)
            s = self.mod(past_target=pt, past_observed_target=om, past_is_pad=pad)
            outs.append(s[:, 4, :].float().cpu().numpy())
        return np.concatenate(outs)


def blend(clean, lin, mask, a):
    out = lin.copy()
    out[mask] = (1.0 - a) * lin[mask] + a * clean[mask]
    return out


# ------------------------------------------------------------------ parts ----

def part_libcheck(res, m2):
    """V1: our forward vs the library's create_predictor path on identical clean windows.

    uni2ts 2.0.0 pins gluonts~=0.14.3, whose QuantileForecastGenerator drives the module
    through singledispatch `predict_to_numpy`; the pinned env has gluonts 0.17.0, whose
    generator expects a different net output convention, so we keep the library's full
    data path (input_transform + instance splitter + batching) and swap in a raw output
    collector with the 0.14.3 semantics -- the check then exercises everything except the
    one gluonts-version-dependent wrapper."""
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from gluonts.itertools import select

    class RawQuantileCollector:
        def __call__(self, inference_data_loader, prediction_net, input_names,
                     output_transform=None, num_samples=None, **kwargs):
            for batch in inference_data_loader:
                inputs = select(input_names, batch, ignore_missing=True)
                out = prediction_net(**inputs)
                for o in out.detach().float().cpu().numpy():
                    yield o

    path = os.path.join(LOCAL, "moirai-2.0-R-small")
    predictor = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(path),
        prediction_length=H, target_dim=1, feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0, context_length=L).create_predictor(
        batch_size=64, device="cuda")
    predictor.forecast_generator = RawQuantileCollector()
    X, st = s25.load_windows(s25.DATASETS["ETTh1"], 60, s25.SEED, H)
    _, cl, _, _ = s25.build_eval_batch(X, st, "clean", 0.0, H)
    import pandas as pd
    ds = [{"target": c.astype(np.float32), "start": pd.Period("2000-01-01", freq="min"),
           "item_id": str(i)} for i, c in enumerate(cl)]
    lib = np.stack([np.asarray(o, np.float32) for o in predictor.predict(ds)])
    lib = lib[:, 4, :]  # median quantile
    ours = m2.fc(cl, np.zeros_like(cl, bool), "plain")
    d = np.abs(lib - ours)
    rel = d.max() / (np.abs(lib).max() + 1e-9)
    corr = float(np.corrcoef(lib.ravel(), ours.ravel())[0, 1])
    res["libcheck"] = {"max_abs_diff": float(d.max()), "max_rel_diff": float(rel),
                       "corr": corr,
                       "note": "library data path (gluonts transform + splitter + "
                               "batching) vs our direct forward call"}
    print(f"V1 library check: max|diff|={d.max():.3e} rel={rel:.3e} corr={corr:.4f}",
          flush=True)
    return res


def part_probe(res, m2):
    bolt = s30.Bolt("cuda")
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, 60, s25.SEED, H)
        for mech in MECHS:
            _, lin, mask, _ = s25.build_eval_batch(X, st, mech, RATE_PROBE, H,
                                                   fill="linear")
            for name, m in (("moirai2", m2), ("bolt", bolt)):
                for conv in m.convs:
                    rng = np.random.default_rng(s25.SEED)
                    r = s30.perm_probe(m, lin, mask, conv, rng)
                    res.setdefault("probe", {})[f"{ds}|{mech}|{name}|{conv}"] = r
                    print(f"probe {ds:8s} {mech:10s} {name:7s} {conv:6s} "
                          f"rho={r['ratio']:.4f}", flush=True)
            json.dump(res, open(OUT, "w"))
    return res


def part_sweep(res, m2, n_win):
    bolt = s30.Bolt("cuda")
    models = {"moirai2": (m2, {"plain": "plain", "declared": "nan"}),
              "bolt": (bolt, {"plain": "plain", "declared": "mask"})}
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, n_win, s25.SEED, H)
        _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
        base = {}
        for name, (m, convs) in models.items():
            base[name] = ((m.fc(cl, np.zeros_like(cl, bool), convs["plain"])
                           - gt_c) ** 2).mean(1)
        for mech in MECHS:
            for rate in RATES:
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, rate, H,
                                                            fill="linear")
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
                line = "  ".join(
                    f"{n}.{arm}:" + "/".join(
                        f"{res['sweep'][f'{ds}|{mech}|{rate}|{a}|{n}|{arm}']['median']:.3f}"
                        for a in (0.0, 1.0))
                    for n in models for arm in models[n][1])
                print(f"sweep {ds:8s} {mech:10s} p={rate} {line}", flush=True)
                json.dump(res, open(OUT, "w"))
    return res


def part_attack(res, m2, n_win):
    bolt = s30.Bolt("cuda")
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, n_win, s25.SEED, H)
        for mech in ("block", "mnar_high"):
            clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, 0.5, H, fill="linear")
            for name, m, conv in (("moirai2", m2, "nan"), ("bolt", bolt, "mask")):
                t0 = time.time()
                e0, w = s33_rand_attack(m, conv, lin, mask, gt)
                ratio = w / np.maximum(e0, 1e-12)
                res.setdefault("attack", {})[f"{ds}|{mech}|{name}"] = {
                    "e0_median": float(np.median(e0)),
                    "worst_median": float(np.median(w)),
                    "ratio_median": float(np.median(ratio)),
                    "ratio_q90": float(np.quantile(ratio, 0.9)),
                    "ratio_max": float(ratio.max())}
                print(f"attack {ds:8s} {mech:10s} {name:7s} "
                      f"x{np.median(ratio):.2f} (q90 x{np.quantile(ratio, 0.9):.2f}) "
                      f"({time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(OUT, "w"))
    return res


def s33_rand_attack(m, conv, ctx, mask, gt, n_try=400, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([np.nanmin(c[~mk]) for c, mk in zip(ctx, mask)], np.float32)[:, None]
    hi = np.array([np.nanmax(c[~mk]) for c, mk in zip(ctx, mask)], np.float32)[:, None]
    e0 = ((m.fc(ctx, mask, conv) - gt) ** 2).mean(1)
    worst = e0.copy()
    B = 20
    for _ in range(n_try // B):
        cand = ctx.copy()
        prop = lo + rng.random(ctx.shape).astype(np.float32) * (hi - lo)
        cand[mask] = prop[mask]
        e = ((m.fc(cand, mask, conv) - gt) ** 2).mean(1)
        worst = np.maximum(worst, e)
    return e0, worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="libcheck,probe,sweep,attack")
    ap.add_argument("--n-win", type=int, default=150)
    args = ap.parse_args()
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"model": "moirai-2.0-R-small (uni2ts 2.0.0, local ckpt)",
                            "L": L, "H": H, "alphas": list(ALPHAS),
                            "probe_n_win": 60, "sweep_n_win": args.n_win,
                            "declared": "fill content + observed mask (cat into content path)"})
    m2 = Moirai2("cuda")
    for part in args.parts.split(","):
        t0 = time.time()
        print(f"\n== {part} ==", flush=True)
        if part == "libcheck":
            res = part_libcheck(res, m2)
        elif part == "probe":
            res = part_probe(res, m2)
        elif part == "sweep":
            res = part_sweep(res, m2, args.n_win)
        elif part == "attack":
            res = part_attack(res, m2, min(args.n_win, 60))
        json.dump(res, open(OUT, "w"))
        print(f"== {part} done in {time.time()-t0:.0f}s ==", flush=True)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
