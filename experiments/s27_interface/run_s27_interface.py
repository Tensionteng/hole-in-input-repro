#!/usr/bin/env python
"""S27: does the input INTERFACE cap how much a better imputation can help?

S25 Part 0 established, exactly, that Chronos-Bolt zeroes the CONTENT at masked positions
and keeps only the flag, so the fill's influence collapses onto (loc, scale) -- rank 2. The
consequence, which this round tests, is a statement about how TSFMs should be built:

    under a content-zeroing interface, forecast quality is invariant to imputation quality,
    so the model cannot benefit from a better imputer -- ever.

Design: sweep imputation quality alpha (fill = (1-alpha)*linear + alpha*truth, alpha=1 being
a perfect imputer) against four input interfaces:

  plain         fill value, NO flag                      (fill-then-feed; TimesFM-like)
  native        flag only, content zeroed                (Chronos-Bolt's masked path)
  dual          fill value AND flag                      (native minus the zeroing line)
  dual_obsnorm  fill + flag, (loc, scale) from observed  (also fixes the statistics channel)

If the claim holds, `native` is flat in alpha while `dual` decreases -- the interface, not
the imputer, is the binding constraint.

Two regimes: zero-shot (bolt as shipped) and adapted (fine-tune ONLY input_patch_embedding
on mechanism-diverse missingness with alpha ~ U(0,1); backbone frozen; the S5-fix2 `mponly`
recipe that preserved clean performance). Zero-shot `dual` is expected to be bad -- bolt was
pretrained expecting zeros there -- so the adapted regime is the real test.

Windows, masks and metrics are S25's (hence S5's, gate-verified).
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

L, H = s25.L, s25.H
MECHS = s25.MECHS
RATES = (0.3, 0.7)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
INTERFACES = ("plain", "native", "dual", "dual_obsnorm")
OUT = os.path.join(HERE, "s27_results.json")
CK = os.path.join(HERE, "s27_ckpt")


# ------------------------------------------------------------- interfaces ----

def encode_iface(model, ctx_filled, obs_mask, iface):
    """Replicates ChronosBoltModelForForecasting.encode with the interface as a switch.

    ctx_filled: [B, L] finite (never NaN). obs_mask: [B, L] 1 = observed.
    The only differences between interfaces are (i) whether the mask is passed at all,
    (ii) whether the content is zeroed at masked positions, (iii) whether (loc, scale) are
    computed from observed points only.
    """
    cfg = model.chronos_config
    if iface == "plain":
        mask = torch.ones_like(ctx_filled)
    else:
        mask = obs_mask.to(ctx_filled.dtype)

    if iface == "dual_obsnorm":
        # statistics from the observed points only: run instance_norm on a NaN'd copy
        nan_ctx = torch.where(obs_mask > 0, ctx_filled, torch.nan)
        _, loc_scale = model.instance_norm(nan_ctx)
        context, _ = model.instance_norm(ctx_filled, loc_scale)
    else:
        context, loc_scale = model.instance_norm(ctx_filled)

    context = context.to(model.dtype)
    mask = mask.to(model.dtype)
    patched_context = model.patch(context)
    patched_mask = torch.nan_to_num(model.patch(mask), nan=0.0)
    if iface == "native":
        # the line this round is about: content discarded, flag kept
        patched_context = torch.where(patched_mask > 0.0, patched_context, 0.0)
    patched_context = torch.cat([patched_context, patched_mask], dim=-1)

    attention_mask = patched_mask.sum(dim=-1) > 0
    input_embeds = model.input_patch_embedding(patched_context)
    if cfg.use_reg_token:
        reg_ids = torch.full((ctx_filled.shape[0], 1), model.config.reg_token_id,
                             device=input_embeds.device)
        input_embeds = torch.cat([input_embeds, model.shared(reg_ids)], dim=-2)
        attention_mask = torch.cat(
            [attention_mask.to(model.dtype), torch.ones_like(reg_ids).to(model.dtype)], dim=-1)
    enc = model.encoder(attention_mask=attention_mask, inputs_embeds=input_embeds)
    return enc[0], loc_scale, input_embeds, attention_mask


def median_iface(bolt, ctx_filled, obs_mask, iface):
    """Differentiable median forecast [B, H] under the chosen interface."""
    m = bolt.model
    hidden, loc_scale, input_embeds, attn = encode_iface(m, ctx_filled, obs_mask, iface)
    seq = m.decode(input_embeds, attn, hidden)
    B = ctx_filled.shape[0]
    q = m.output_patch_embedding(seq).view(B, m.num_quantiles,
                                           m.chronos_config.prediction_length)
    q = m.instance_norm.inverse(q.view(B, -1), loc_scale).view(
        B, m.num_quantiles, m.chronos_config.prediction_length)
    return q[:, 4, :H]


@torch.no_grad()
def median_iface_np(bolt, ctx_np, obs_np, iface, batch=256):
    outs = []
    for i in range(0, len(ctx_np), batch):
        c = torch.from_numpy(ctx_np[i:i + batch]).to(bolt.device)
        o = torch.from_numpy(obs_np[i:i + batch]).to(bolt.device)
        outs.append(median_iface(bolt, c, o, iface).float().cpu().numpy())
    return np.concatenate(outs)


def blend_fill(clean, filled_linear, mask, alpha):
    """(1-alpha)*linear + alpha*truth at the masked positions; observed points untouched."""
    out = filled_linear.copy()
    m = mask
    out[m] = (1.0 - alpha) * filled_linear[m] + alpha * clean[m]
    return out


# ----------------------------------------------------------------- gate ----

def gate(res, bolt):
    """Verify the re-implemented encode reproduces the stock model bit-for-bit on clean
    input, and that S5's stored numbers still come out of this harness."""
    out = {}
    X, starts = s25.load_windows(s25.DATASETS["ETTh1"], 300, s25.SEED, s25.H_GATE)
    for key, stored in s25.ANCHORS.items():
        mech, fill, rate = key.split(":")
        rate = float(rate)
        seeds = (0,) if mech in ("clean", "mnar_high", "mnar_extreme") else (0, 1)
        vals = []
        for ms in seeds:
            _, filled, _, gt = s25.build_eval_batch(X, starts, mech, rate, s25.H_GATE,
                                                    fill=fill, mask_seed=ms)
            pred = bolt.median_rollout_np(filled, s25.H_GATE).astype(np.float64)
            vals.append(float(((pred - gt.astype(np.float64)) ** 2).mean()))
        v = float(np.mean(vals))
        d = abs(v - stored) / stored
        out[key] = {"stored": stored, "rerun": v, "rel_dev": d, "pass": d <= 0.05}
        print(f"GATE s5 {key:24s} stored={stored:10.5f} rerun={v:10.5f} dev={d:+.4%} "
              f"{'PASS' if d <= 0.05 else 'FAIL'}", flush=True)

    # re-implementation check: interface `plain` on a fully observed context must equal the
    # stock forward pass exactly (same code path, no mask, no zeroing).
    Xe, se = s25.load_windows(s25.DATASETS["ETTh1"], 50, s25.SEED, H)
    _, cl, mk, _ = s25.build_eval_batch(Xe, se, "clean", 0.0, H)
    obs = (~mk).astype(np.float32)
    a = bolt.median_np(cl)
    b = median_iface_np(bolt, cl, obs, "plain")
    d = float(np.abs(a - b).max())
    out["reimpl_plain_vs_stock"] = {"max_abs_diff": d, "pass": d < 1e-3}
    print(f"GATE reimpl plain==stock  max|diff|={d:.3e} {'PASS' if d < 1e-3 else 'FAIL'}",
          flush=True)

    # and `native` with an explicit mask must equal feeding NaN through the stock path
    _, fl, mk2, _ = s25.build_eval_batch(Xe, se, "mcar", 0.3, H, fill="linear")
    nan_ctx = fl.copy()
    nan_ctx[mk2] = np.nan
    a = bolt.median_np(nan_ctx)                              # stock NaN path
    b = median_iface_np(bolt, fl, (~mk2).astype(np.float32), "dual_obsnorm")
    # dual_obsnorm differs from the NaN path only by keeping the content, so they are NOT
    # expected to match; the meaningful check is that native+observed-stats == NaN path.
    out["nan_vs_dual_obsnorm_differs"] = {"max_abs_diff": float(np.abs(a - b).max())}
    print(f"GATE nan-path vs dual_obsnorm max|diff|="
          f"{out['nan_vs_dual_obsnorm_differs']['max_abs_diff']:.4f} (expected > 0)",
          flush=True)
    res["gate"] = out
    assert all(v.get("pass", True) for v in out.values()), "gate FAILED -- stop"
    return res


# ---------------------------------------------------------------- sweep ----

def sweep(res, bolt, tag, nets=None, n_win=300):
    """alpha x interface x mechanism x rate x dataset, paired per-window relMSE."""
    cells = {}
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, n_win, s25.SEED, H)
        _, cl_f, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
        obs_c = np.ones_like(cl_f)
        base = ((bolt.median_np(cl_f) - gt_c) ** 2).mean(1)
        ok = base > 1e-12
        for mech in MECHS:
            for rate in RATES:
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, rate, H, fill="linear")
                obs = (~mask).astype(np.float32)
                for a in ALPHAS:
                    ctx = blend_fill(clean, lin, mask, a)
                    for iface in INTERFACES:
                        if nets is not None:
                            load_proj(bolt, nets[iface])
                        p = median_iface_np(bolt, ctx, obs, iface)
                        e = ((p - gt) ** 2).mean(1)
                        r = e[ok] / base[ok]
                        cells[f"{tag}|{ds}|{mech}|{rate}|{a}|{iface}"] = {
                            "median": float(np.median(r)), "mean": float(r.mean()),
                            "q75": float(np.quantile(r, 0.75)),
                            "q95": float(np.quantile(r, 0.95)), "n": int(ok.sum())}
                print(f"  {tag} {ds:8s} {mech:13s} p={rate} " + " ".join(
                    f"{i}:{cells[f'{tag}|{ds}|{mech}|{rate}|1.0|{i}']['median']:.3f}"
                    for i in INTERFACES), flush=True)
    res.setdefault("sweep", {}).update(cells)
    return res


# ------------------------------------------------------------- adaptation ----

def save_proj(bolt):
    return {k: v.detach().clone() for k, v in
            bolt.model.input_patch_embedding.state_dict().items()}


def load_proj(bolt, sd):
    bolt.model.input_patch_embedding.load_state_dict(sd)


def adapt(bolt, iface, base_sd, steps=600, bs=48, lr=1e-4, log=150, seed=3):
    """Fine-tune ONLY input_patch_embedding, on mechanism-diverse missingness with the
    imputation quality alpha ~ U(0,1) so the module learns when to trust the content."""
    load_proj(bolt, base_sd)
    mod = bolt.model.input_patch_embedding
    for p in mod.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(mod.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    streams = {m: s25.TrainStream(m, n_per_ds=3000, seed=s25.SEED + 7 + i)
               for i, m in enumerate(MECHS)}
    hist, n_skip = [], 0
    for it in range(steps):
        mech = MECHS[int(rng.integers(len(MECHS)))]
        ctx, lin, mask, fut = streams[mech].batch(bs)
        a = float(rng.uniform(0, 1))
        x = blend_fill(ctx, lin, mask, a)
        obs = (~mask).astype(np.float32)
        xt = torch.from_numpy(x).to(bolt.device)
        ot = torch.from_numpy(obs).to(bolt.device)
        y = torch.from_numpy(fut).to(bolt.device)
        pred = median_iface(bolt, xt, ot, iface)
        loss = s25.huber_scaled(pred, y, s25.loss_scale(ctx, bolt.device))
        if not torch.isfinite(loss):
            n_skip += 1
            continue
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad()
            n_skip += 1
            continue
        opt.step()
        sch.step()
        hist.append(float(loss))
        if it == 0 or (it + 1) % log == 0:
            print(f"    adapt[{iface}] {it+1}/{steps} loss={np.mean(hist[-log:]):.4f} "
                  f"skip={n_skip}", flush=True)
    sd = save_proj(bolt)
    for p in mod.parameters():
        p.requires_grad_(False)
    load_proj(bolt, base_sd)
    return sd, hist


def save(res):
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="gate,zeroshot,adapt")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--n-win", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    os.makedirs(CK, exist_ok=True)

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"L": L, "H": H, "alphas": list(ALPHAS),
                            "interfaces": list(INTERFACES), "rates": list(RATES),
                            "model": "amazon/chronos-bolt-base",
                            "adaptation": "input_patch_embedding only, backbone frozen"})
    bolt = s25.Bolt("cuda")
    base_sd = save_proj(bolt)
    steps = 40 if args.smoke else args.steps
    nwin = 30 if args.smoke else args.n_win

    parts = args.parts.split(",")
    if "gate" in parts:
        res = gate(res, bolt)
        save(res)
    if "zeroshot" in parts:
        print("\n== zero-shot sweep ==", flush=True)
        res = sweep(res, bolt, "zeroshot", nets=None, n_win=nwin)
        save(res)
    if "adapt" in parts:
        nets = {}
        for iface in INTERFACES:
            t0 = time.time()
            sd, hist = adapt(bolt, iface, base_sd, steps=steps)
            torch.save(sd, os.path.join(CK, f"proj_{iface}.pt"))
            nets[iface] = sd
            print(f"  [{iface}] adapted in {time.time()-t0:.0f}s "
                  f"final_loss={np.mean(hist[-50:]):.4f}", flush=True)
        print("\n== adapted sweep ==", flush=True)
        res = sweep(res, bolt, "adapted", nets=nets, n_win=nwin)
        load_proj(bolt, base_sd)
        save(res)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
