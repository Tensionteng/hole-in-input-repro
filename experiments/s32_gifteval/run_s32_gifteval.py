#!/usr/bin/env python
"""S32: how much of a GIFT-Eval score is the model, and how much is its NaN handling?

Scanning the distributed GIFT-Eval corpus: **11.77% of its 305M points are NaN**, and 9 of
28 datasets contain missing values -- including `electricity` (19.2% NaN, 57% of series),
`restaurant` (13.8%), `bitbrains_rnd` (13.1%), `hierarchical_sales`, `jena_weather`, on top
of the three explicitly named `*_with_missing`. So missingness is not a corner case the
benchmark excludes; it is a large, unremarked property of the benchmark everyone reports on.

No paper reporting a GIFT-Eval number states how its model handled those NaNs, and S25/S30
showed the handling differs qualitatively by family (Chronos discards the fill content,
TimesFM overwrites it with its own interpolation, Moirai keeps using it). If a score moves
more when you change the fill convention than the gaps between models on the leaderboard,
then part of what the leaderboard ranks is undocumented preprocessing.

Two parts:
  A. FILL SENSITIVITY -- for each model, score the missing-containing datasets under four
     fill conventions (nan / linear / zero / ffill) and measure the spread.
  B. SCALE -- the four Chronos-Bolt sizes on both the clean and the missing datasets. This
     also repairs S28's stated limitation: on ETT/weather clean accuracy was flat with scale,
     so "scale does not buy robustness" could not be separated from "scale buys nothing".
     GIFT-Eval is the benchmark the family's scaling was reported on.

Protocol (documented, deliberately NOT the official GIFT-Eval one): last-window evaluation
per series, context 512, horizon 64, scored only on OBSERVED target points, metric MASE
against an in-window naive-1 forecast computed on observed points. Absolute values are
therefore not leaderboard-comparable; every claim here is a WITHIN-protocol comparison
across fills and across models, which is what the question needs.
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
GIFT = os.path.join(ROOT, "tsfm_missing", "data", "gifteval")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import pyarrow as pa
import torch

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
import run_s25_twofloor as s25
import run_s30_crossmodel as s30

L, H = 512, 64
FILLS = ("nan", "linear", "zero", "ffill")
OUT = os.path.join(HERE, "s32_results.json")

MISSING_DS = ["electricity/H", "kdd_cup_2018_with_missing/H", "restaurant",
              "bitbrains_rnd/5T", "bitbrains_fast_storage/5T", "hierarchical_sales/D",
              "car_parts_with_missing",
              "temperature_rain_with_missing", "jena_weather/H"]
CLEAN_DS = ["ett1/H", "ett2/H", "m4_hourly", "solar/H", "us_births/D",
            "covid_deaths", "hospital", "saugeenday/D"]


def read_series(dsfreq, max_series=400, seed=0):
    d = os.path.join(GIFT, dsfreq)
    files = sorted(glob.glob(os.path.join(d, "*.arrow")))
    out = []
    for f in files:
        with pa.memory_map(f, "rb") as src:
            t = pa.ipc.open_stream(src).read_all()
        for s in t.column("target").to_pylist():
            a = np.asarray(s, dtype=np.float32)
            # several GIFT-Eval datasets store a 2-d (n_variates, T) target; each variate is
            # its own univariate series here (the earlier column-0 reshape silently produced
            # length-n_variates arrays and dropped these datasets entirely)
            rowset = [a] if a.ndim == 1 else [a[i] for i in range(a.shape[0])]
            for r in rowset:
                if r.size >= L + H:
                    out.append(np.ascontiguousarray(r, dtype=np.float32))
        if len(out) >= max_series * 3:
            break
    rng = np.random.default_rng(seed)
    if len(out) > max_series:
        out = [out[i] for i in rng.choice(len(out), max_series, replace=False)]
    return out


def windows(series, per_series=3, seed=1):
    """Random windows across each series, not just the last one.

    Last-window-only evaluation systematically misses interior gaps and leading padding --
    electricity's 19% NaN is mostly leading NaN from unequal series start dates, which a
    last-window protocol never sees. Sampling uniformly represents what the corpus actually
    contains. Context may contain NaN; the target is scored on its observed points only.
    """
    rng = np.random.default_rng(seed)
    ctx, tgt, ok = [], [], []
    for a in series:
        n = len(a)
        if n < L + H:
            continue
        k = min(per_series, max(1, (n - L - H) // (L // 2) + 1))
        starts = rng.choice(n - L - H + 1, size=k, replace=False) if n - L - H + 1 >= k else [0]
        for s0 in np.atleast_1d(starts):
            c, y = a[s0:s0 + L], a[s0 + L:s0 + L + H]
            m = np.isfinite(y)
            if m.sum() < 8 or np.isfinite(c).sum() < 16:
                continue
            ctx.append(c)
            tgt.append(np.nan_to_num(y))
            ok.append(m)
    if not ctx:
        return None
    return np.stack(ctx), np.stack(tgt), np.stack(ok)


def fill(ctx, how):
    out = ctx.copy()
    miss = ~np.isfinite(out)
    if how == "nan":
        return out, miss
    t = np.arange(out.shape[1])
    for i in range(len(out)):
        v = np.flatnonzero(np.isfinite(out[i]))
        if len(v) == 0:
            out[i] = 0.0
            continue
        if how == "linear":
            out[i] = np.interp(t, v, out[i][v])
        elif how == "zero":
            out[i] = np.nan_to_num(out[i])
        elif how == "ffill":
            pos = np.maximum(np.searchsorted(v, t, side="right") - 1, 0)
            out[i] = out[i][v[pos]]
    return out.astype(np.float32), miss


def mase(pred, tgt, ok, ctx):
    """MASE vs an in-window naive-1 baseline computed on the observed context points."""
    num = np.where(ok, np.abs(pred - tgt), np.nan)
    num = np.nanmean(num, axis=1)
    den = []
    for c in ctx:
        v = c[np.isfinite(c)]
        den.append(np.mean(np.abs(np.diff(v))) if len(v) > 2 else np.nan)
    den = np.asarray(den)
    with np.errstate(all="ignore"):
        r = num / np.where(den > 1e-9, den, np.nan)
    return r


MODELS = {
    "bolt-tiny": lambda: s30.Bolt.__new__(s30.Bolt),
}


class Q50Bolt:
    """Our retrofit (S53/Q50, bolt-base CPT x WiSE-FT 0.5). "plain" (a fixed fill
    is supplied) is routed to our declared path; "nan" zero-fills + declares."""

    name = "q50-base"
    n_params = 205_000_000

    def __init__(self, dev):
        sys.path.insert(0, os.path.join(EXP, "s27_interface"))
        sys.path.insert(0, os.path.join(EXP, "s45_pretrain"))
        import eval_s45 as e45
        self.e45 = e45
        self.m = e45.load_arm("Q50", dev, "base")
        self.dev = dev

    @torch.no_grad()
    def fc(self, ctx, miss, conv, batch=256):
        if conv == "nan":
            obs = np.isfinite(ctx).astype(np.float32)
            c = np.nan_to_num(ctx).astype(np.float32)
        else:
            obs = (~miss).astype(np.float32)
            c = ctx.astype(np.float32)
        return self.e45.median_fwd(self.m, c, obs, "dual")[:, :H]


def get_model(name):
    if name.startswith("bolt-"):
        size = name.split("-")[1]
        path = ("amazon/chronos-bolt-base" if size == "base"
                else os.path.join(ROOT, "models_local", f"chronos-bolt-{size}"))
        m = s30.Bolt.__new__(s30.Bolt)
        from chronos import BaseChronosPipeline
        m.p = BaseChronosPipeline.from_pretrained(path, device_map="cuda",
                                                  torch_dtype=torch.float32)
        m.m = m.p.inner_model if hasattr(m.p, "inner_model") else m.p.model
        m.dev = "cuda"
        m.n_params = sum(q.numel() for q in m.m.parameters())
        return m
    if name == "chronos2":
        m = s30.Chronos2("cuda")
        m.n_params = 119_477_664
        return m
    if name == "timesfm":
        m = s30.TimesFM("cuda")
        m.n_params = 200_000_000
        return m
    if name == "moirai":
        m = s30.Moirai("cuda")
        m.n_params = 91_000_000
        return m
    if name == "moirai2":
        sys.path.insert(0, os.path.join(EXP, "s46_moirai2"))
        from run_s46_moirai2 import Moirai2
        m = Moirai2("cuda")
        m.n_params = 11_400_000
        return m
    if name == "t5-small":
        m = s30.T5("cuda")
        m.n_params = 46_000_000
        return m
    if name == "tirex":
        sys.path.insert(0, os.path.join(EXP, "s49_2025models"))
        from run_s49_2025 import TiRex
        m = TiRex("cuda")
        m.n_params = 35_000_000
        return m
    if name == "flowstate":
        sys.path.insert(0, os.path.join(EXP, "s49_2025models"))
        from run_s49_2025 import FlowState
        m = FlowState("cuda")
        m.n_params = 9_100_000
        return m
    if name == "q50base":
        return Q50Bolt("cuda")
    raise ValueError(name)


def forecast(model, ctx_filled, miss, how):
    """Route each fill convention to the model's matching input path."""
    if isinstance(model, s30.Bolt):
        return model.fc(ctx_filled, miss, "nan" if how == "nan" else "plain")
    if how == "nan":
        return model.fc(ctx_filled, miss, "nan")
    return model.fc(ctx_filled, miss, "plain")


def run(res, name, dslist, tag, max_series):
    model = get_model(name)
    print(f"\n== {name} ({tag}) ==", flush=True)
    for dsfreq in dslist:
        try:
            ser = read_series(dsfreq, max_series=max_series)
            w = windows(ser)
        except Exception as e:
            print(f"  {dsfreq:44s} READ-FAIL {type(e).__name__}", flush=True)
            continue
        if w is None:
            print(f"  {dsfreq:44s} no usable windows (n_series={len(ser)}, "
                  f"len_ok={sum(len(a) >= L + H for a in ser)})", flush=True)
            continue
        ctx, tgt, ok = w
        nanfrac = float((~np.isfinite(ctx)).mean())
        line = f"  {dsfreq:44s} n={len(ctx):4d} ctxNaN={100*nanfrac:5.2f}%  "
        for how in FILLS:
            f_, miss = fill(ctx, how)
            if how == "nan":
                f_in = np.where(np.isfinite(ctx), ctx, np.nan).astype(np.float32)
            else:
                f_in = f_
            try:
                t0 = time.time()
                pred = forecast(model, f_in if how == "nan" else f_, miss, how)
                m = mase(pred, tgt, ok, ctx)
                v = float(np.nanmedian(m))
            except Exception as e:
                v = float("nan")
                print(f"[{how} FAIL {type(e).__name__}: {str(e)[:60]}]", end="", flush=True)
            res.setdefault("cells", {})[f"{name}|{dsfreq}|{how}"] = {
                "mase_median": v, "n": int(len(ctx)), "ctx_nan_frac": nanfrac}
            line += f"{how}={v:7.3f} "
        print(line, flush=True)
    res.setdefault("params", {})[name] = int(getattr(model, "n_params", 0))
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bolt-base,chronos2,timesfm,moirai")
    ap.add_argument("--part", default="A", help="A = fill sensitivity, B = scale")
    ap.add_argument("--max-series", type=int, default=300)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("meta", {"L": L, "H": H, "fills": list(FILLS),
                            "protocol": "last window per series, scored on observed target "
                                        "points, MASE vs in-window naive-1",
                            "missing_ds": MISSING_DS, "clean_ds": CLEAN_DS})
    for name in args.models.split(","):
        dslist = MISSING_DS if args.part == "A" else (MISSING_DS + CLEAN_DS)
        try:
            res = run(res, name, dslist, args.part, args.max_series)
        except Exception as e:
            print(f"SKIP {name}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        json.dump(res, open(args.out + ".tmp", "w"))
        os.replace(args.out + ".tmp", args.out)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
