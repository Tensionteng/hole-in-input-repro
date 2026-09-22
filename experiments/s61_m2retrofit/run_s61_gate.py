#!/usr/bin/env python
"""S61 GATE: before any training, establish that

  G1  our direct call path (m2_iface.median_fwd through the released
      Moirai2Forecast.forward) reproduces the library's own inference path
      (gluonts input_transform + instance splitter + batching, raw output
      collector with the 0.14.3 generator semantics -- the s46 V1 check) on
      identical clean windows: require max|Delta| = 0 in strict fp32 (TF32 off).
  G2  the training forward (module training_mode + released
      PackedQuantileMAELoss on the last-context-token -> 4 future tokens map)
      produces a finite loss and finite gradients with STOCK weights.
  G3  the s46 permutation probe reproduces the paper's value for stock:
      declared-path rho ~ 0.95 (median over 3 ds x 3 mechs, rate 0.3, linear
      fill, 60 windows/cell, 4 perms).

Exits nonzero if G1 fails (do not train on a divergent replica).
"""
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

torch.backends.cuda.matmul.allow_tf32 = False     # strict fp32 for the gate
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import run_s30_crossmodel as s30
import m2_iface

L, H = s25.L, 64
OUT = os.path.join(HERE, "s61_gate.json")


def part_lib(res, dev):
    """G1: library data path vs m2_iface.median_fwd on identical clean windows."""
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

    predictor = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(m2_iface.CKPT),
        prediction_length=H, target_dim=1, feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0, context_length=L).create_predictor(
        batch_size=64, device=dev)
    predictor.forecast_generator = RawQuantileCollector()
    X, st = s25.load_windows(s25.DATASETS["ETTh1"], 60, s25.SEED, H)
    _, cl, _, _ = s25.build_eval_batch(X, st, "clean", 0.0, H)
    import pandas as pd
    ds = [{"target": c.astype(np.float32), "start": pd.Period("2000-01-01", freq="min"),
           "item_id": str(i)} for i, c in enumerate(cl)]
    lib = np.stack([np.asarray(o, np.float32) for o in predictor.predict(ds)])
    lib = lib[:, 4, :]                                   # median quantile
    fc = m2_iface.load_stock(dev)
    content, obs = m2_iface.conv_ctx(cl, np.zeros_like(cl, bool), "plain")
    ours = m2_iface.median_fwd(fc, content, obs, batch=64)   # library's batch size
    d = float(np.abs(lib - ours).max())
    ours512 = m2_iface.median_fwd(fc, content, obs, batch=512)
    d_bs = float(np.abs(ours - ours512).max())
    res["libcheck"] = {"max_abs_diff": d, "pass": d == 0.0,
                       "batchsize_numeric_gap": d_bs,
                       "note": "library data path (gluonts transform + splitter + "
                               "batching, bs 64) vs m2_iface.median_fwd at bs 64; "
                               "s46's V1 recorded max_abs_diff = 0.0 for the same "
                               "call. batchsize_numeric_gap = ours bs64 vs bs512 "
                               "(cuBLAS kernel choice; eval uses bs 1024)"}
    print(f"G1 library check: max|diff|={d:.3e} {'PASS' if d == 0.0 else 'FAIL'}",
          flush=True)
    return d == 0.0


def part_trainfw(res, dev):
    """G2: native training loss + backward are finite with stock weights."""
    fc = m2_iface.load_stock(dev, train=True)
    rng = np.random.default_rng(0)
    ctx = rng.normal(size=(16, L)).astype(np.float32)
    fut = rng.normal(size=(16, H)).astype(np.float32)
    miss = rng.random(size=(16, L)) < 0.4
    content, obs = m2_iface.conv_ctx(ctx, miss, "declared")
    loss = m2_iface.train_forward(fc, content, obs, fut)
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(fc.module.parameters(), 1.0)
    ok = bool(torch.isfinite(loss) and torch.isfinite(gn))
    res["train_forward"] = {"loss": float(loss), "grad_norm": float(gn), "pass": ok,
                            "note": "module training_mode + released "
                                    "PackedQuantileMAELoss, last-context-token -> "
                                    "4 future tokens (the H=64 inference map)"}
    print(f"G2 training forward: loss={float(loss):.4f} |grad|={float(gn):.3f} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def part_probe(res, dev):
    """G3: reproduce the s46 rho for stock (declared ~ 0.95)."""
    fc = m2_iface.load_stock(dev)

    class Wrap:                     # s30.perm_probe drives model.fc(ctx, miss, conv)
        convs = ("plain", "nan")

        def fc_(self, ctx, miss, conv):
            content, obs = m2_iface.conv_ctx(ctx, miss, conv)
            return m2_iface.median_fwd(fc, content, obs)
    w = Wrap()
    w.fc = w.fc_
    rhos = {"plain": [], "nan": []}
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, 60, s25.SEED, H)
        for mech in ("mcar", "block", "mnar_high"):
            _, lin, mask, _ = s25.build_eval_batch(X, st, mech, 0.3, H, fill="linear")
            for conv in ("plain", "nan"):
                rng = np.random.default_rng(s25.SEED)
                r = s30.perm_probe(w, lin, mask, conv, rng)
                res.setdefault("probe", {})[f"{ds}|{mech}|stock|{conv}"] = r
                rhos[conv].append(r["ratio"])
                print(f"G3 probe {ds:8s} {mech:10s} stock {conv:6s} "
                      f"rho={r['ratio']:.4f}", flush=True)
    agg = {c: {"rho_median": float(np.median(v)), "rho_mean": float(np.mean(v))}
           for c, v in rhos.items()}
    res["probe_agg"] = agg
    res["probe_ref"] = {"s46_moirai2_nan_rho_median": 0.9492,
                        "s46_moirai2_plain_rho_median": 0.9672}
    print(f"G3 stock rho: declared(nan) median={agg['nan']['rho_median']:.4f} "
          f"(s46: 0.9492), plain median={agg['plain']['rho_median']:.4f} (s46: 0.9672)",
          flush=True)
    return True


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    parts = sys.argv[2].split(",") if len(sys.argv) > 2 else ["lib", "trainfw", "probe"]
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"model": "models_local/moirai-2.0-R-small (uni2ts, local)",
                            "dtype": "float32", "tf32": False, "L": L, "H": H})
    ok = True
    for p in parts:
        t0 = time.time()
        print(f"\n== {p} ==", flush=True)
        if p == "lib":
            ok &= part_lib(res, dev)
        elif p == "trainfw":
            ok &= part_trainfw(res, dev)
        elif p == "probe":
            ok &= part_probe(res, dev)
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"== {p} done in {time.time()-t0:.0f}s ==", flush=True)
    res["pass"] = bool(ok and res.get("libcheck", {}).get("pass", True))
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"GATE overall: {'PASS' if res['pass'] else 'FAIL'}; wrote {OUT}", flush=True)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
