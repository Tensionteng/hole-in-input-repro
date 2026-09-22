#!/usr/bin/env python
"""S6-F: masked-augmentation SFT of chronos-bolt-base -- the "expected baseline"
repair method for missing-context damage (see run_s5_missing.py / run_s5_fix.py).

Per-dataset SFT on the TRAIN split (first 70% of the timeline), two variants:

  SFT-mb : augmentation masks = mcar + block(24), rate ~ U(0.05, 0.8)
  SFT-all : same + MNAR-style censoring (top-p% values per series, p ~ U(0.05, 0.7))

Each augmented series is corrupted as raw NaN (bolt's native mask-aware path,
chronos_bolt.py encode) with prob 1/2, else linear-filled -- so both evaluation
input modes are in-distribution. 20% of series stay clean to anchor clean
performance. Loss = bolt's NATIVE pinball (quantile) loss, replicated
element-wise from the model's own InstanceNorm loc/scale so it can be CLIPPED
(--loss-clip, default 100 in normalized units): the s5_fix notes documented the
unclipped native loss spiking to 1e7 on tiny-std weather channels, turning
those batches into AdamW-normalized noise. Grad-norm clip 1.0 on top.

Adapter: LoRA (peft, r=16 alpha=32 dropout 0.05, target q/k/v/o of the T5
stacks) if peft works; else full fine-tune of encoder+decoder (--adapter
full). Checkpoints (trainable params only + meta) go to s6_sft_ckpt/.

Evaluation: S5 test grid -- mechanisms {mcar, block, mnar_high, mnar_extreme} x
p {0.1,0.3,0.5,0.7} x fills {nan, linear} x {ETTh1, ETTm1, weather}, 150
windows x 1 mask seed (same windows as run_s5_extra.py; seed base 20250810).
Per-dataset SFT models are evaluated on their OWN dataset (matched protocol).
Zero-shot bolt (model key "bolt_zs") is re-run on the same 150-window grid so
every relMSE uses a paired clean baseline. Results appended to --out after
every config (crash-safe, resumable).
"""
import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import pandas as pd
import torch

import run_s5_missing as s5

RESULTS_PATH = os.path.join(HERE, "s6_sft_results.json")
CKPT_DIR = os.path.join(HERE, "s6_sft_ckpt")
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
EVAL_FILLS = ("nan", "linear")
PRED = 64  # bolt native prediction length used in training
LORA_KW = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
               target_modules=["q", "k", "v", "o"])
DS_IDX = {"ETTh1": 0, "ETTm1": 1, "weather": 2}
VAR_IDX = {"mb": 0, "all": 1}


# ------------------------------------------------------------ training ----

def load_X(ds):
    df = pd.read_csv(s5.DATASETS[ds])
    return df.drop(columns=["date"]).to_numpy(np.float32)


def aug_mask(rng, ctx_row, variant):
    """[L] bool augmentation mask for one series. 20% clean; else uniform over
    the variant's mechanisms, rate ~ U(0.05, 0.8) (U(0.05, 0.7) for mnar)."""
    L = len(ctx_row)
    m = np.zeros(L, bool)
    if rng.random() < 0.2:
        return m
    mechs = ("mcar", "block") if variant == "mb" else ("mcar", "block", "mnar_high")
    mech = mechs[rng.integers(len(mechs))]
    p = rng.uniform(0.05, 0.7 if mech == "mnar_high" else 0.8)
    if mech == "mcar":
        m[:] = rng.random(L) < p
    elif mech == "block":
        for s in rng.integers(0, L - s5.BLOCK + 1, size=int(round(p * L / s5.BLOCK))):
            m[s:s + s5.BLOCK] = True
    else:  # mnar_high: censor the top ceil(p*L) values
        k = int(np.ceil(p * L))
        m[np.argsort(-ctx_row, kind="stable")[:k]] = True
    return m


def corrupt(ctx, m, form):
    """ctx [B, L], m [B, L] bool -> corrupted input (nan or linear-filled)."""
    if form == "nan":
        out = ctx.copy()
        out[m] = np.nan
        return out
    out = ctx.copy()
    t = np.arange(ctx.shape[1])
    for b in range(len(ctx)):
        valid = np.flatnonzero(~m[b])
        if len(valid) == 0:
            continue  # leave as-is (never happens: p <= 0.8)
        out[b] = np.interp(t, valid, ctx[b, valid])
    return out


def clipped_pinball(model, xb, yb, clip):
    """bolt's native pinball loss (chronos_bolt.py forward, target branch),
    replicated element-wise so it can be clipped. model(context) returns
    UNSCALED quantile preds [B, 9, P]; we re-normalize preds and target with
    the context's own nan-aware InstanceNorm loc/scale, exactly as the model
    does internally."""
    preds = model(context=xb).quantile_preds                     # [B, 9, P]
    loc = torch.nan_to_num(torch.nanmean(xb, dim=-1, keepdim=True), nan=0.0)
    scale = torch.nan_to_num((xb - loc).square().nanmean(dim=-1, keepdim=True).sqrt(),
                             nan=1.0)
    scale = torch.where(scale == 0, model.instance_norm.eps, scale)
    loc, scale = loc.unsqueeze(1), scale.unsqueeze(1)            # [B, 1, 1]
    yn = (yb.unsqueeze(1) - loc) / scale                         # [B, 1, H]
    pn = (preds - loc) / scale                                   # [B, 9, H]
    q = model.quantiles.view(1, -1, 1)
    elem = 2 * torch.abs((yn - pn) * ((yn <= pn).float() - q))   # [B, 9, H]
    raw = elem.mean(dim=1).sum(dim=-1).mean()                    # native reduction
    loss = torch.clamp(elem, max=clip).mean(dim=1).sum(dim=-1).mean()
    return loss, raw.detach()


def wrap_lora(model):
    """get_peft_model needs get_input_embeddings (transformers 5 raises
    NotImplementedError on ChronosBoltModelForForecasting); bolt's `shared`
    embedding (the [REG] token) is the honest answer -- nothing is tied to it,
    so peft's tied-module check becomes a no-op."""
    from peft import LoraConfig, get_peft_model
    model.get_input_embeddings = lambda: model.shared
    return get_peft_model(model, LoraConfig(**LORA_KW))


def train(args):
    device = "cuda"
    seed = s5.SEED + 1000 * VAR_IDX[args.variant] + DS_IDX[args.dataset]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
    model = pipe.model
    assert not model.instance_norm.use_arcsinh, "loss replication assumes no arcsinh"
    model.train()  # bolt has no dropout/BN; needed only for lora_dropout

    if args.adapter == "lora":
        model = wrap_lora(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:  # full fine-tune of encoder + decoder only
        for p in model.parameters():
            p.requires_grad_(False)
        for mod in (model.encoder, model.decoder):
            for p in mod.parameters():
                p.requires_grad_(True)
        trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"adapter={args.adapter} trainable params: "
          f"{sum(p.numel() for p in trainable):,}", flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / args.warmup))

    Xtr = load_X(args.dataset)[: int(0.7 * len(load_X(args.dataset)))]
    N, C = Xtr.shape
    t0 = time.time()
    run_loss, run_raw, run_n = 0.0, 0.0, 0
    for step in range(1, args.steps + 1):
        starts = rng.integers(0, N - s5.L - PRED, size=args.batch)
        chans = rng.integers(0, C, size=args.batch)
        ctx = np.stack([Xtr[s:s + s5.L, c] for s, c in zip(starts, chans)])
        fut = np.stack([Xtr[s + s5.L:s + s5.L + PRED, c] for s, c in zip(starts, chans)])
        m = np.stack([aug_mask(rng, ctx[b], args.variant) for b in range(args.batch)])
        forms = rng.random(args.batch) < 0.5
        xb_np = np.stack([corrupt(ctx[b:b + 1], m[b:b + 1],
                                  "nan" if forms[b] else "linear")[0]
                          for b in range(args.batch)])
        xb = torch.from_numpy(xb_np).to(device)
        yb = torch.from_numpy(fut).to(device)
        with torch.enable_grad():
            loss, raw = clipped_pinball(model, xb, yb, args.loss_clip)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        run_loss += loss.item()
        run_raw += raw.item()
        run_n += 1
        if step % args.log_every == 0 or step == 1:
            print(f"step {step}/{args.steps} loss={run_loss / run_n:.4f} "
                  f"raw={run_raw / run_n:.2f} ({time.time() - t0:.0f}s)", flush=True)
            run_loss, run_raw, run_n = 0.0, 0.0, 0

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = os.path.join(args.ckpt_dir, f"sft_{args.variant}_{args.dataset}.pt")
    torch.save({
        "adapter": args.adapter,
        "lora_config": LORA_KW if args.adapter == "lora" else None,
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters()
                      if p.requires_grad},
        "meta": {"variant": args.variant, "dataset": args.dataset,
                 "steps": args.steps, "batch": args.batch, "lr": args.lr,
                 "warmup": args.warmup, "loss_clip": args.loss_clip,
                 "seed": seed, "train_frac": 0.7, "pred_len_train": PRED,
                 "aug": ("mcar+block24 p~U(.05,.8)" if args.variant == "mb" else
                         "mcar+block24 p~U(.05,.8) + mnar_high p~U(.05,.7)"),
                 "input_forms": "50% raw-NaN / 50% linear-fill; 20% clean series"},
    }, ckpt)
    print(f"saved -> {ckpt} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------ evaluation ----

class SFTBoltModel:
    name, patch, batch = "sft", 16, 1024

    def __init__(self, device, ckpt):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)
        sd = torch.load(ckpt, map_location=device)
        model = self.pipe.model
        if sd["adapter"] == "lora":
            model = wrap_lora(model)
            self.pipe.model = model
        model.load_state_dict(sd["trainable"], strict=False)
        model.eval()
        self.device = device
        self.meta = sd["meta"]

    @torch.no_grad()
    def predict_point(self, ctx):  # ctx: [n, L] float32, may contain NaN
        outs = []
        for i in range(0, len(ctx), self.batch):
            xb = torch.from_numpy(ctx[i:i + self.batch]).to(self.device)
            q = self.pipe.predict(xb, prediction_length=s5.H)
            outs.append(q[:, 4, :].float().cpu().numpy())
        return np.concatenate(outs)


def fill_eval(x, mask, fill):
    if fill in ("none",):
        return x.copy()
    if fill == "nan":
        out = x.copy()
        out[mask] = np.nan
        return out
    return s5.fill_context(x, mask, fill)  # linear


def run_config(model, X, starts, cfg, n_seeds=1):
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
            ctxs.append(fill_eval(x_clean, mask, "none" if mech == "clean" else fill))
            gts.append(y)
            masks.append(mask)
    S = len(ctxs)
    ctx = np.stack(ctxs).reshape(S * C, s5.L).astype(np.float32)
    pred = model.predict_point(ctx).reshape(S, C, s5.H).astype(np.float64)
    gt = np.stack(gts).astype(np.float64)
    mse = ((pred - gt) ** 2).mean(axis=(1, 2))
    mae = np.abs(pred - gt).mean(axis=(1, 2))
    rate_sw = np.stack([m.mean() for m in masks])
    nw = len(starts)
    return {
        "mech": mech, "fill": fill, "rate": rate,
        "n_windows": nw, "n_seeds": 1 if det else n_seeds,
        "mse": float(mse.mean()), "mae": float(mae.mean()),
        "achieved_rate": float(rate_sw.mean()),
        "mse_per_window": mse.reshape(nw, -1).mean(axis=1).tolist(),
        "mae_per_window": mae.reshape(nw, -1).mean(axis=1).tolist(),
    }


def cfgkey(cfg):
    return f"{cfg['mech']}:{cfg['fill']}:{cfg['rate']}"


def eval_grid():
    cfgs = [dict(mech="clean", fill="none", rate=0.0)]
    for r in s5.RATES:
        for mech in MECHS:
            for fill in EVAL_FILLS:
                cfgs.append(dict(mech=mech, fill=fill, rate=r))
    return cfgs


def run_eval(model, section, datasets, out, shard=0, nshards=1):
    results = {}
    if os.path.exists(out):
        results = json.load(open(out))
        print(f"resume: loaded {out}", flush=True)
    results.setdefault(section, {})
    jobs = [(ds, cfg) for ds in datasets for cfg in eval_grid()]
    if nshards > 1:
        jobs = jobs[shard::nshards]
        print(f"shard {shard}/{nshards}: {len(jobs)} jobs", flush=True)
    loaded = {}
    for ds, cfg in jobs:
        key = cfgkey(cfg)
        if key in results[section].get(ds, {}):
            continue
        if ds not in loaded:
            loaded[ds] = s5.load_windows(s5.DATASETS[ds], 150, s5.SEED)
        X, starts = loaded[ds]
        t0 = time.time()
        res = run_config(model, X, starts, cfg, n_seeds=1)
        res["seconds"] = round(time.time() - t0, 1)
        results[section].setdefault(ds, {})[key] = res
        s5.save_results(results, out)
        print(f"{section:8s} {ds:8s} {key:26s} mse={res['mse']:10.4f} "
              f"ach={res['achieved_rate']:.3f} ({res['seconds']}s)", flush=True)
    return results


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval-zs", action="store_true", help="zero-shot bolt grid")
    ap.add_argument("--eval", action="store_true", help="SFT variant, matched datasets")
    ap.add_argument("--variant", choices=["mb", "all"], default="mb")
    ap.add_argument("--dataset", choices=list(DS_IDX), default="ETTh1")
    ap.add_argument("--adapter", choices=["lora", "full"], default="lora")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--loss-clip", type=float, default=100.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--out", default=RESULTS_PATH)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    if args.tiny:
        # 5 training steps + 1 eval config, ETTh1, prints only
        args.steps, args.log_every = 5, 1
        train(args)
        ckpt = os.path.join(args.ckpt_dir, f"sft_{args.variant}_{args.dataset}.pt")
        model = SFTBoltModel("cuda", ckpt)
        X, starts = s5.load_windows(s5.DATASETS[args.dataset], 10, s5.SEED)
        for cfg in [dict(mech="clean", fill="none", rate=0.0),
                    dict(mech="mcar", fill="nan", rate=0.3),
                    dict(mech="mnar_high", fill="linear", rate=0.3)]:
            res = run_config(model, X, starts, cfg)
            print(f"tiny-eval {cfgkey(cfg):24s} mse={res['mse']:.4f}", flush=True)
        return

    if args.train:
        train(args)
        return

    if args.eval_zs:
        model = s5.BoltModel("cuda")
        run_eval(model, "bolt_zs", list(DS_IDX), args.out, args.shard, args.nshards)
        print("EVAL ZS DONE", flush=True)
        return

    if args.eval:
        for ds in DS_IDX:
            ckpt = os.path.join(args.ckpt_dir, f"sft_{args.variant}_{ds}.pt")
            if not os.path.exists(ckpt):
                print(f"skip {ds}: {ckpt} missing", flush=True)
                continue
            model = SFTBoltModel("cuda", ckpt)
            run_eval(model, f"sft_{args.variant}", [ds], args.out,
                     args.shard, args.nshards)
            del model
            torch.cuda.empty_cache()
        print("EVAL DONE", flush=True)
        return


if __name__ == "__main__":
    main()
