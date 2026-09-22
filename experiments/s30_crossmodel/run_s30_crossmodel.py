#!/usr/bin/env python
"""S30: the fill-reachable set across model families (the P0-d owed since S25).

S25 Part 0 proved, exactly, that Chronos-Bolt's masked path discards the CONTENT of the
fill and keeps only the flag (rank 2), while its NaN path discards the fill entirely
(rank 0) and plain fill-then-feed uses it in full. S27 turned that into a prescription and
S29 into a security corollary (attack-surface dimension = rank). Both arguments are
currently bolt-only. This round measures the same structure across every family we can run.

Method: the gradient-free permutation test from S25 Part 0, which works on any black box.
Permuting the fill values AMONG the missing positions preserves their multiset -- hence the
context's mean and variance -- while destroying which value sits where. So

    output change under permutation / output change under an independent redraw

is ~1 if the model uses the fill's content, and ~0 if it only sees its summary statistics.
Run per model and per input convention, this yields a cross-family table of how much of a
lever an imputer actually has, and how much of an attack surface a defender is exposing.

Models: chronos-bolt-base, chronos-2, chronos-t5-small, timesfm-2.5, moirai-1.1-R-base.
Conventions per model: whatever that family actually offers (plain fill; an explicit mask
where the API has one; NaN where the model claims to handle it).
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
import run_s25_twofloor as s25

L, H = s25.L, 64
OUT = os.path.join(HERE, "s30_results.json")
MECHS = ("mcar", "block", "mnar_high")
RATE = 0.3
N_WIN = 60
N_PERM = 4


# --------------------------------------------------------------- wrappers ----

class Bolt:
    name = "chronos-bolt-base"
    convs = ("plain", "mask", "nan")

    def __init__(self, dev):
        from chronos import BaseChronosPipeline
        self.p = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base",
                                                     device_map=dev, torch_dtype=torch.float32)
        self.m = self.p.inner_model if hasattr(self.p, "inner_model") else self.p.model
        self.dev = dev

    @torch.no_grad()
    def fc(self, ctx, miss, conv, batch=256):
        outs = []
        for i in range(0, len(ctx), batch):
            c = ctx[i:i + batch].copy()
            mk = miss[i:i + batch]
            if conv == "nan":
                c[mk] = np.nan
            t = torch.from_numpy(c).to(self.dev)
            mask = None
            if conv == "mask":
                mask = torch.from_numpy((~mk).astype(np.float32)).to(self.dev)
            outs.append(self.m(context=t, mask=mask).quantile_preds[:, 4, :H]
                        .float().cpu().numpy())
        return np.concatenate(outs)


class Chronos2:
    name = "chronos-2"
    convs = ("plain", "nan")

    def __init__(self, dev):
        from chronos import Chronos2Pipeline
        self.p = Chronos2Pipeline.from_pretrained(os.path.join(ROOT, "models_local", "chronos-2"),
                                                  device_map=dev, torch_dtype=torch.float32)
        self.dev = dev

    @torch.no_grad()
    def fc(self, ctx, miss, conv, batch=128):
        outs = []
        for i in range(0, len(ctx), batch):
            c = ctx[i:i + batch].copy()
            if conv == "nan":
                c[miss[i:i + batch]] = np.nan
            q = self.p.predict(list(c), prediction_length=H, batch_size=batch)
            arr = np.stack([np.asarray(x if not torch.is_tensor(x) else x.cpu()) for x in q])
            arr = arr.squeeze(1) if arr.ndim == 4 else arr
            outs.append(arr[:, arr.shape[1] // 2, :])           # median quantile
        return np.concatenate(outs)


class T5:
    name = "chronos-t5-small"
    convs = ("plain", "nan")

    def __init__(self, dev):
        from chronos import BaseChronosPipeline
        self.p = BaseChronosPipeline.from_pretrained("amazon/chronos-t5-small",
                                                     device_map=dev, torch_dtype=torch.float32)
        self.dev = dev

    @torch.no_grad()
    def fc(self, ctx, miss, conv, batch=64):
        outs = []
        for i in range(0, len(ctx), batch):
            c = ctx[i:i + batch].copy()
            if conv == "nan":
                c[miss[i:i + batch]] = np.nan
            torch.manual_seed(s25.SEED)
            s = self.p.predict(torch.from_numpy(c), prediction_length=H, num_samples=20)
            outs.append(s.median(dim=1).values.float().cpu().numpy())
        return np.concatenate(outs)


class TimesFM:
    name = "timesfm-2.5-200m"
    convs = ("plain", "nan")

    def __init__(self, dev):
        import timesfm
        self.m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        self.m.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=128,
                                              per_core_batch_size=256, normalize_inputs=True,
                                              use_continuous_quantile_head=True))

    def fc(self, ctx, miss, conv, batch=256):
        outs = []
        for i in range(0, len(ctx), batch):
            c = ctx[i:i + batch].copy()
            if conv == "nan":
                c[miss[i:i + batch]] = np.nan
            pf, _ = self.m.forecast(horizon=H, inputs=[r for r in c])
            outs.append(np.asarray(pf, dtype=np.float32))
        return np.concatenate(outs)


class Moirai:
    name = "moirai-1.1-R-base"
    convs = ("plain", "nan")

    def __init__(self, dev):
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        self.mod = MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base"),
            prediction_length=H, context_length=L, patch_size=32,
            num_samples=20, target_dim=1, feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0).to(dev)
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
            outs.append(s.median(dim=1).values.squeeze(-1).float().cpu().numpy())
        return np.concatenate(outs)


REGISTRY = {"bolt": Bolt, "chronos2": Chronos2, "t5": T5, "timesfm": TimesFM, "moirai": Moirai}


# ------------------------------------------------------------------ probe ----

def perm_probe(model, ctx, miss, conv, rng):
    """Permutation vs redraw sensitivity of the forecast to the FILL CONTENT."""
    y0 = model.fc(ctx, miss, conv)
    scale = np.sqrt((y0 ** 2).mean(1)) + 1e-9
    dp, dr = [], []
    for _ in range(N_PERM):
        cp, cr = ctx.copy(), ctx.copy()
        for i in range(len(ctx)):
            idx = np.flatnonzero(miss[i])
            if len(idx) < 2:
                continue
            cp[i, idx] = ctx[i, idx][rng.permutation(len(idx))]
            o = ctx[i, ~miss[i]]
            cr[i, idx] = rng.choice(o, size=len(idx)) if len(o) else 0.0
        dp.append(np.sqrt(((model.fc(cp, miss, conv) - y0) ** 2).mean(1)) / scale)
        dr.append(np.sqrt(((model.fc(cr, miss, conv) - y0) ** 2).mean(1)) / scale)
    dp, dr = np.mean(dp, 0), np.mean(dr, 0)
    return {"perm_rms_rel": float(dp.mean()), "redraw_rms_rel": float(dr.mean()),
            "ratio": float(dp.mean() / max(dr.mean(), 1e-15)),
            "perm_max": float(dp.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bolt,chronos2,t5,timesfm,moirai")
    ap.add_argument("--n-win", type=int, default=N_WIN)
    args = ap.parse_args()
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"rate": RATE, "mechs": list(MECHS), "n_win": args.n_win,
                            "n_perm": N_PERM, "H": H,
                            "probe": "permutation vs redraw of the fill content"})
    for key in args.models.split(","):
        try:
            model = REGISTRY[key]("cuda")
        except Exception as e:
            print(f"SKIP {key}: {type(e).__name__}: {str(e)[:140]}", flush=True)
            res.setdefault("models", {})[key] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        print(f"\n== {model.name} ==", flush=True)
        cells = {}
        for ds, path in s25.DATASETS.items():
            X, st = s25.load_windows(path, args.n_win, s25.SEED, H)
            for mech in MECHS:
                clean, lin, mask, _ = s25.build_eval_batch(X, st, mech, RATE, H, fill="linear")
                for conv in model.convs:
                    rng = np.random.default_rng(s25.SEED)
                    t0 = time.time()
                    try:
                        r = perm_probe(model, lin, mask, conv, rng)
                    except Exception as e:
                        print(f"  {ds:8s} {mech:10s} {conv:6s} FAIL {type(e).__name__}: "
                              f"{str(e)[:90]}", flush=True)
                        continue
                    cells[f"{ds}|{mech}|{conv}"] = r
                    print(f"  {ds:8s} {mech:10s} {conv:6s} perm={r['perm_rms_rel']:.3e} "
                          f"redraw={r['redraw_rms_rel']:.3e} ratio={r['ratio']:.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        agg = {}
        for conv in model.convs:
            v = [c["ratio"] for k, c in cells.items() if k.endswith("|" + conv)]
            if v:
                agg[conv] = {"ratio_median": float(np.median(v)), "ratio_max": float(np.max(v)),
                             "n_cells": len(v)}
        res.setdefault("models", {})[key] = {"name": model.name, "cells": cells, "agg": agg}
        print(f"  AGG {model.name}: " + "  ".join(
            f"{c}={agg[c]['ratio_median']:.4f}" for c in agg), flush=True)
        del model
        torch.cuda.empty_cache()
        json.dump(res, open(OUT + ".tmp", "w"))
        os.replace(OUT + ".tmp", OUT)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
