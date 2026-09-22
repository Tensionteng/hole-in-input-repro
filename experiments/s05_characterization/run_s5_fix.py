#!/usr/bin/env python
"""Screen S5, Track A: can the missing-context damage be REPAIRED?

run_s5_missing.py established: (i) TSFMs are near-immune to mcar under decent
interpolation; (ii) naive zero-fill hurts via corrupted internal scaling
statistics (mcar only -- the oracle-rescale probe recovered ~80% of bolt's
damage at p=0.5); (iii) structured missingness (block, mnar) hurts via fill
error / information loss, and the oracle rescale BACKFIRES there. This script
evaluates three repair strategies, cheapest first:

  Fix 0 -- native NaN probe. Just hand the raw NaN context to the pipeline.
     * bolt (chronos_bolt.py): mask = ~isnan(context) (l.280); InstanceNorm
       uses nanmean/nan-std over OBSERVED positions (l.111-112) -- mask-aware
       scaling is native; missing values become 0 in normalized space
       (= observed-mean fill, l.298) and the binary mask is concatenated into
       the patch embedding input (l.300), so the model sees which points are
       missing. "Doing nothing" for bolt is already a strong repair.
     * timesfm (timesfm_2p5_base.py l.176): every input is silently passed
       through strip_leading_nans + np.interp linear interpolation. "Doing
       nothing" = our linear fill, approximately (leading gaps differ: strip
       + zero-pad-with-mask vs our edge-value backfill).
  Fix 1 -- black-box observed-stats rescale (no oracle, deployable): zero-fill
     then rescale by mean(|x_observed|)/mean(|x_zerofilled|), and the same
     on top of linear fill. Under mcar, mean|x_obs| is unbiased so this
     approximates the oracle probe (which recovered ~80% at p=0.5); under mnar
     the observed stats are themselves biased low, so it should under-correct
     or distort -- we measure which.
  Fix 2 -- white-box bolt surgery. bolt already has (i) mask-aware scaling
     natively; we add (ii) a per-patch missing-fraction feature: a new
     Linear(1, d_model) (zero-init => exactly Fix 0 at start) whose output is
     added to the patch embeddings via a forward hook on
     input_patch_embedding (the mask flags are already in its input, so the
     hook can compute the fraction). Backbone frozen; only miss_proj and
     input_patch_embedding train, with the model's OWN pinball loss
     (chronos_bolt.py l.372), on train-split (first 70%) windows with random
     mask augmentation (mech in {mcar, block, mnar_high, mnar_extreme},
     rate ~ U(0.05, 0.8)). Evaluated on the block/mnar grids.

Same windows/seeds conventions as run_s5_missing.py (imported, not copied):
bolt = 300 windows (2 mask seeds for mcar/block, 1 for deterministic mnar),
timesfm = first 100 of those windows, 1 seed. New keys go to
s5_fix_results.json (incremental, resumable); existing s5 files untouched.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import run_s5_missing as s5

HERE = s5.HERE
RESULTS_PATH = os.path.join(HERE, "s5_fix_results.json")
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
FIX_FILLS = ("nan", "zero_obsrescale", "linear_obsrescale")
CKPT = os.path.join(HERE, "s5_fix_bolt_adapter.pt")


# ----------------------------------------------------------------- fills ----

def fill_fix(x, mask, fill):
    """x: [C, L] clean context; mask: [C, L] True=missing."""
    if fill in ("nan", "fix2"):  # fix2 also feeds NaN (adapter intercepts it)
        out = x.copy()
        out[mask] = np.nan
        return out
    if fill in ("zero_obsrescale", "linear_obsrescale"):
        if fill == "zero_obsrescale":
            base = x.copy()
            base[mask] = 0.0
        else:
            base = s5.fill_context(x, mask, "linear")
        obs = np.where(mask, np.nan, x)
        with np.errstate(all="ignore"):
            num = np.nanmean(np.abs(obs), axis=1, keepdims=True)  # observed-only mean|x|
        den = np.abs(base).mean(axis=1, keepdims=True)
        ok = np.isfinite(num) & (den > 1e-12)
        scale = np.where(ok, np.nan_to_num(num) / np.maximum(den, 1e-12), 1.0)
        return base * scale
    if fill == "none":
        return x.copy()
    raise ValueError(fill)


def run_config(model, X, starts, cfg, n_seeds):
    """Same conventions as s5.run_config (identical masks/windows), new fills."""
    C = X.shape[1]
    mech, fill, rate = cfg["mech"], cfg["fill"], cfg["rate"]
    det = mech in ("clean", "mnar_high", "mnar_extreme")
    ctxs, gts, masks = [], [], []
    for wi, s in enumerate(starts):
        x_clean = X[s:s + s5.L].T.copy()
        y = X[s + s5.L:s + s5.L + s5.H].T.copy()
        for ms in ((0,) if det else range(n_seeds)):
            mask = (np.zeros((C, s5.L), bool) if mech == "clean"
                    else s5.make_mask(mech, rate, wi, ms, C, x=x_clean))
            ctxs.append(fill_fix(x_clean, mask, "none" if mech == "clean" else fill))
            gts.append(y)
            masks.append(mask)
    S = len(ctxs)
    ctx = np.stack(ctxs).reshape(S * C, s5.L).astype(np.float32)
    pred = model.predict_point(ctx).reshape(S, C, s5.H).astype(np.float64)
    gt = np.stack(gts).astype(np.float64)
    mse = ((pred - gt) ** 2).mean(axis=(1, 2))
    mae = np.abs(pred - gt).mean(axis=(1, 2))
    rate_sw = np.stack([m.mean() for m in masks])
    cpf_sw = np.array([s5.corr_patch_frac(m, model.patch) for m in masks])
    nw = len(starts)
    return {
        "mech": mech, "fill": fill, "rate": rate,
        "n_windows": nw, "n_seeds": 1 if det else n_seeds,
        "mse": float(mse.mean()), "mae": float(mae.mean()),
        "achieved_rate": float(rate_sw.mean()), "corr_patch_frac": float(cpf_sw.mean()),
        "mse_per_window": mse.reshape(nw, -1).mean(axis=1).tolist(),
        "mae_per_window": mae.reshape(nw, -1).mean(axis=1).tolist(),
        "achieved_rate_per_window": rate_sw.reshape(nw, -1).mean(axis=1).tolist(),
        "corr_patch_frac_per_window": cpf_sw.reshape(nw, -1).mean(axis=1).tolist(),
    }


def cfgkey(cfg):
    return f"{cfg['mech']}:{cfg['fill']}:{cfg['rate']}"


def fix_grid():
    cfgs = [dict(mech="clean", fill="none", rate=0.0)]
    for r in s5.RATES:
        for mech in MECHS:
            for fill in FIX_FILLS:
                cfgs.append(dict(mech=mech, fill=fill, rate=r))
    return cfgs


def fix2_grid():
    cfgs = [dict(mech="clean", fill="none", rate=0.0)]
    for r in s5.RATES:
        for mech in MECHS:
            cfgs.append(dict(mech=mech, fill="fix2", rate=r))
    return cfgs


def save_results(results, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(results, fh)
    os.replace(tmp, path)


# -------------------------------------------------------- fix 2: surgery ----

def load_X(ds):
    df = pd.read_csv(s5.DATASETS[ds])
    return df.drop(columns=["date"]).to_numpy(np.float32)


def random_masks(rng, ctx):
    """[B, L] bool; mech in {none, mcar, block, mnar_high, mnar_extreme} x20%,
    rate ~ U(0.05, 0.8)."""
    B, L = ctx.shape
    m = np.zeros((B, L), bool)
    for b in range(B):
        mech = rng.integers(0, 5)
        if mech == 0:
            continue
        p = rng.uniform(0.05, 0.8)
        if mech == 1:
            m[b] = rng.random(L) < p
        elif mech == 2:
            for s in rng.integers(0, L - s5.BLOCK + 1, size=int(round(p * L / s5.BLOCK))):
                m[b, s:s + s5.BLOCK] = True
        else:
            k = int(np.ceil(p * L))
            if mech == 3:
                key = ctx[b]
            else:
                sd = ctx[b].std()
                key = np.abs((ctx[b] - ctx[b].mean()) / (sd if sd > 1e-12 else 1.0))
            m[b, np.argsort(-key, kind="stable")[:k]] = True
    return m


def make_miss_hook(miss_proj, patch):
    def hook(module, inputs, output):
        # inputs[0]: [B, n_patches, 2*P]; last P dims are the observed flags
        mask_part = inputs[0][..., patch:]
        frac = (1.0 - mask_part).mean(dim=-1, keepdim=True)  # [B, n_patches, 1]
        return output + miss_proj(frac.to(output.dtype))
    return hook


def train_fix2(args):
    from chronos import BaseChronosPipeline
    device = "cuda"
    torch.manual_seed(s5.SEED + 7)
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
    model = pipe.model
    model.eval()  # no dropout; grads still flow to the trainable params
    patch = model.config.chronos_config["input_patch_size"]
    d_model = model.config.d_model
    miss_proj = torch.nn.Linear(1, d_model).to(device)
    torch.nn.init.zeros_(miss_proj.weight)
    torch.nn.init.zeros_(miss_proj.bias)  # zero-init => starts exactly at Fix 0
    model.input_patch_embedding.register_forward_hook(make_miss_hook(miss_proj, patch))

    for p in model.parameters():
        p.requires_grad_(False)
    trainable = list(miss_proj.parameters())
    if args.train_input_proj:
        for p in model.input_patch_embedding.parameters():
            p.requires_grad_(True)
        trainable += list(model.input_patch_embedding.parameters())
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    print(f"trainable params: {sum(p.numel() for p in trainable):,} "
          f"(train_input_proj={args.train_input_proj})", flush=True)

    data = {ds: load_X(ds) for ds in s5.DATASETS}
    train = {ds: X[: int(0.7 * len(X))] for ds, X in data.items()}
    rng = np.random.default_rng(s5.SEED + 13)
    DS = list(s5.DATASETS)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        ds = DS[step % len(DS)]
        Xtr = train[ds]
        N, C = Xtr.shape
        starts = rng.integers(0, N - s5.L - 64, size=args.batch)
        chans = rng.integers(0, C, size=args.batch)
        ctx = np.stack([Xtr[s:s + s5.L, c] for s, c in zip(starts, chans)])
        fut = np.stack([Xtr[s + s5.L:s + s5.L + 64, c] for s, c in zip(starts, chans)])
        m = random_masks(rng, ctx)
        xb = torch.from_numpy(np.where(m, np.nan, ctx)).to(device)
        yb = torch.from_numpy(fut).to(device)
        with torch.enable_grad():
            out = model(context=xb, target=yb)  # native pinball loss on normalized target
            out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % args.log_every == 0 or step == 1:
            print(f"step {step}/{args.steps} ds={ds} loss={out.loss.item():.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    torch.save({"miss_proj": miss_proj.state_dict(),
                "input_patch_embedding": model.input_patch_embedding.state_dict(),
                "steps": args.steps, "lr": args.lr,
                "train_input_proj": args.train_input_proj}, args.ckpt)
    print(f"saved -> {args.ckpt}", flush=True)


class Fix2BoltModel(s5.BoltModel):
    """bolt + trained miss-frac adapter; feed NaN contexts (fill 'fix2')."""
    name, patch, batch = "bolt_fix2", 16, 1024

    def __init__(self, device, ckpt):
        super().__init__(device)
        sd = torch.load(ckpt, map_location=device)
        model = self.pipe.model
        patch = model.config.chronos_config["input_patch_size"]
        self.miss_proj = torch.nn.Linear(1, model.config.d_model).to(device)
        self.miss_proj.load_state_dict(sd["miss_proj"])
        model.input_patch_embedding.load_state_dict(sd["input_patch_embedding"])
        model.input_patch_embedding.register_forward_hook(
            make_miss_hook(self.miss_proj, patch))
        model.eval()


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--models", default="bolt,timesfm")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=RESULTS_PATH)
    ap.add_argument("--train-fix2", action="store_true")
    ap.add_argument("--eval-fix2", action="store_true")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--steps", type=int, default=480)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=40)
    ap.add_argument("--train-input-proj", action="store_true",
                    help="also train input_patch_embedding (not just miss_proj)")
    args = ap.parse_args()

    torch.manual_seed(s5.SEED)
    np.random.seed(s5.SEED)

    if args.train_fix2:
        train_fix2(args)
        return

    if args.tiny:
        bolt = s5.BoltModel("cuda")
        X, starts = s5.load_windows(s5.DATASETS["ETTh1"], 20, s5.SEED)
        cfgs = [dict(mech="clean", fill="none", rate=0.0)]
        for mech in ("mcar", "mnar_high"):
            for fill in FIX_FILLS:
                cfgs.append(dict(mech=mech, fill=fill, rate=0.3))
        print("tiny-fix: ETTh1 20 windows, bolt, p=0.3 "
              "(reference: mcar:linear=12.8624, mcar:zero=13.8794, mnar_high:linear=13.8385)")
        for cfg in cfgs:
            t0 = time.time()
            res = run_config(bolt, X, starts, cfg, n_seeds=2)
            print(f"{cfgkey(cfg):32s} mse={res['mse']:10.4f} ach={res['achieved_rate']:.3f} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        del bolt
        torch.cuda.empty_cache()
        tfm = s5.TimesFmModel("cuda")
        for cfg in [dict(mech="mcar", fill="nan", rate=0.3)]:
            res = run_config(tfm, X, starts, cfg, n_seeds=1)
            print(f"timesfm {cfgkey(cfg):24s} mse={res['mse']:10.4f} "
                  f"(internal-interp probe)", flush=True)
        return

    if args.eval_fix2:
        model = Fix2BoltModel("cuda", args.ckpt)
        results = {}
        if os.path.exists(args.out):
            results = json.load(open(args.out))
        results.setdefault("bolt_fix2", {})
        grid = fix2_grid()
        jobs = [(ds, cfg) for ds in s5.DATASETS for cfg in grid]
        if args.nshards > 1:
            jobs = jobs[args.shard::args.nshards]
            print(f"shard {args.shard}/{args.nshards}: {len(jobs)} jobs", flush=True)
        for ds, cfg in jobs:
            key = cfgkey(cfg)
            if key in results["bolt_fix2"].get(ds, {}):
                continue
            X, starts = s5.load_windows(s5.DATASETS[ds], 300, s5.SEED)
            t0 = time.time()
            res = run_config(model, X, starts, cfg, n_seeds=2)
            res["seconds"] = round(time.time() - t0, 1)
            results["bolt_fix2"].setdefault(ds, {})[key] = res
            save_results(results, args.out)
            print(f"bolt_fix2 {ds:8s} {key:28s} mse={res['mse']:10.4f} ({res['seconds']}s)",
                  flush=True)
        print("EVAL FIX2 DONE", flush=True)
        return

    # ---------------- Fix 0 + Fix 1 grid ----------------
    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
        print(f"resume: loaded {args.out}", flush=True)
    results.setdefault("meta", {
        "track": "S5-A repair methods", "seed": s5.SEED,
        "fix0_nan": "raw NaN to pipeline (bolt: native mask-aware nanmean/nan-std "
                    "scaling + mean-fill-in-normalized-space + mask concat; "
                    "timesfm: silent internal linear interpolation)",
        "fix1": "observed-stats rescale: fill * mean(|x_observed|)/mean(|x_filled|)",
        "conventions": "same windows/seeds as s5_missing_results.json",
    })
    grid = fix_grid()
    for name in args.models.split(","):
        cls = {"bolt": s5.BoltModel, "timesfm": s5.TimesFmModel}[name]
        model = cls("cuda")
        results.setdefault(name, {})
        n_win = 300 if name == "bolt" else 100
        n_seeds = 2 if name == "bolt" else 1
        jobs = [(ds, cfg) for ds in s5.DATASETS for cfg in grid]
        if args.nshards > 1:
            jobs = jobs[args.shard::args.nshards]
            print(f"shard {args.shard}/{args.nshards}: {len(jobs)} jobs", flush=True)
        loaded = {}
        for ds, cfg in jobs:
            key = cfgkey(cfg)
            if key in results[name].get(ds, {}):
                continue
            if ds not in loaded:
                loaded[ds] = s5.load_windows(s5.DATASETS[ds], 300, s5.SEED)
            X, starts_all = loaded[ds]
            t0 = time.time()
            res = run_config(model, X, starts_all[:n_win], cfg, n_seeds)
            res["seconds"] = round(time.time() - t0, 1)
            results[name].setdefault(ds, {})[key] = res
            save_results(results, args.out)
            print(f"{name:8s} {ds:8s} {key:32s} mse={res['mse']:10.4f} "
                  f"ach={res['achieved_rate']:.3f} ({res['seconds']}s)", flush=True)
        del model
        torch.cuda.empty_cache()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
