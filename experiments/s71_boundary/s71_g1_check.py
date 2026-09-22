#!/usr/bin/env python
"""S71 G1 substance check: CPT must actually LEARN the presented task.

The 100-step running-mean train losses are tail-noise dominated (corpus scale
tail; timer especially). This script evaluates each model's native CPT loss on a
FIXED set of corpus windows -- same windows, same augmentation stream for every
checkpoint -- for the stock / cpt / ctl5k checkpoints, split by presentation
(clean vs declared vs filled). If cpt < stock on the missing presentations, the
CPT learned the missingness task (G1 in substance); ctl5k should sit at stock
level on missing and slightly better on clean.

Usage: python s71_g1_check.py --models timer,timesfm,tempopfn --device cuda:0
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tempopfn_triton_cache")

import numpy as np
import torch

sys.path.insert(0, HERE)
import s71_common as C
import train_s45 as t45

OUT = os.path.join(HERE, "s71_g1_check.json")
N_WIN = 512
BS = 128


def batches(model_key):
    """Fixed window set + fixed augmentation stream (independent of checkpoint)."""
    rng = np.random.default_rng(C.SEED + 777)
    sampler = t45.Sampler(seed=C.SEED + 778)
    gen = sampler.window(require_clean_ctx=True)
    rows = {"clean": [], "declared": [], "filled": []}
    wi = 0
    while sum(len(v) for v in rows.values()) < N_WIN * 3:
        c, f = next(gen)
        out, mask = C.s71_augment(c, model_key, "mechdiv", rng, wi,
                                  clean_p=1.0 / 3.0)
        wi += 1
        kind = ("clean" if not mask.any() else
                ("declared" if _is_declared(model_key, out, c, mask)
                 else "filled"))
        if len(rows[kind]) >= N_WIN:
            continue
        rows[kind].append((out, mask, f))
    return rows


def _is_declared(model_key, out, ctx, mask):
    if model_key == "tempopfn":
        return bool(np.isnan(out[mask]).all())
    if model_key == "timer":
        return bool((out[mask] == 0.0).all())
    # timesfm: declared half is the model's own linear interpolant
    lin = t45.linear_fill(ctx, mask)
    return bool(np.allclose(out[mask], lin[mask]))


def loss_for(model_key, ckpt, rows, dev):
    """Per-window native losses (tail-robust median alongside the mean)."""
    if model_key == "timesfm":
        m = C.build_timesfm(dev, ckpt)
        module = m.model
        module.eval()

        def f(ctx, miss, fut):
            with torch.no_grad():
                q = C.timesfm_fwd_train(module, ctx, fut, dev)
                levels = list(module.config.quantiles)
                qq = q[..., 1:10]
                y = torch.from_numpy(fut).to(dev).unsqueeze(-1)
                e = y - qq
                t = torch.as_tensor(levels, device=dev,
                                    dtype=qq.dtype).view(1, 1, -1)
                return torch.maximum(t * e, (t - 1) * e).mean(dim=(1, 2))
    elif model_key == "tempopfn":
        model = C.build_tempopfn(dev, ckpt)
        model.eval()

        def f(ctx, miss, fut):
            cont = C.tempopfn_container(ctx, fut, dev)
            with torch.no_grad(), torch.autocast(device_type="cuda",
                                                 dtype=torch.bfloat16):
                out = model(cont)
            pred = out["result"]                       # [B, H, 1, Q] scaled
            fs = model.scaler.scale(cont.future_values, out["scale_statistics"])
            e = fs.unsqueeze(-1).float() - pred.float()
            qt = model.qt.float()
            return torch.maximum((qt - 1) * e, qt * e).mean(dim=(1, 2, 3))
    else:
        model = C.build_timer(dev, ckpt)
        model.eval()

        def f(ctx, miss, fut):
            with torch.no_grad():
                B = len(ctx)
                c = ctx[:, -C.TIMER_CTX:]
                obs = ~miss[:, -C.TIMER_CTX:]
                cnt = obs.sum(1).clip(1)
                mu = (c * obs).sum(1) / cnt
                sd = np.sqrt((((c - mu[:, None]) ** 2) * obs).sum(1) / cnt)
                sd = np.maximum(sd, 1e-3)
                cn = ((c - mu[:, None]) / sd[:, None]).astype(np.float32)
                cn[miss[:, -C.TIMER_CTX:]] = 0.0
                fn = ((fut - mu[:, None]) / sd[:, None]).astype(np.float32)
                x = np.concatenate([cn, fn, np.zeros((B, C.TIMER_PAD),
                                    np.float32)], axis=1)
                w = np.zeros((B, C.TIMER_CTX + C.H + C.TIMER_PAD), np.float32)
                w[:, :C.TIMER_CTX] = (~miss[:, -C.TIMER_CTX:]).astype(np.float32)
                w[:, C.TIMER_CTX:C.TIMER_CTX + C.H] = 1.0
                t = torch.from_numpy(x).to(dev)
                out = model(input_ids=t, use_cache=False,
                            output_hidden_states=True)
                pred = model.lm_heads[0](out.hidden_states[-1])
                tgt = t[:, 96:].unfold(-1, 96, 96)
                wt = torch.from_numpy(w[:, 96:]).to(dev)
                lo = torch.nn.functional.huber_loss(pred[:, :-1], tgt, delta=20.0,
                                                    reduction="none")
                num = (lo.reshape(B, -1) * wt.reshape(B, -1)).sum(1)
                return num / wt.reshape(B, -1).sum(1).clamp_min(1.0)
    res = {}
    for kind, rows_k in rows.items():
        losses = []
        for i in range(0, len(rows_k), BS):
            chunk = rows_k[i:i + BS]
            ctx = np.stack([r[0] for r in chunk])
            miss = np.stack([r[1] for r in chunk])
            fut = np.stack([r[2] for r in chunk])
            v = f(ctx, miss, fut)
            losses.extend(float(x) for x in v.float().cpu().numpy())
        res[kind] = {"median": float(np.median(losses)),
                     "mean": float(np.mean(losses)),
                     "q90": float(np.quantile(losses, 0.9)),
                     "n": len(losses)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for model in args.models.split(","):
        rows = batches(model)
        for arm in ("stock", "cpt", "ctl5k"):
            ckpt = C.resolve_ckpt(model, arm)
            r = loss_for(model, ckpt, rows, args.device)
            res.setdefault(model, {})[arm] = r
            print(f"{model:9s} {arm:6s} " + " ".join(
                f"{k}:med={v['median']:.4f},mean={v['mean']:.4f}"
                for k, v in r.items()), flush=True)
            json.dump(res, open(OUT, "w"))
        del rows
    print("wrote", OUT)


if __name__ == "__main__":
    main()
