#!/usr/bin/env python
"""S67 permvis: the permutation test, made visible on one window.

The paper's claim, per window: on vanilla bolt's DECLARED (native) path the fill
content is dropped except through the instance-norm statistics, so PERMUTING the
fill values among the missing positions (same multiset -> same mean/variance) must
leave the forecast unchanged up to float noise. After the 5k-step CPT retrofit
(dual interface, block_filldiv regime; S53 arm P5) the model reads the fill, so
the same permutation moves the forecast and the attention.

Models (identical to eval_s512_attn.py / fig_attnmap.pdf):
  P0 = s45_ckpt/arm_P0_tiny.pt   stock chronos-bolt-tiny state-dict dump (train_s53
                                 --arm P0), declared interface = native
  P5 = s45_ckpt/arm_P5_tiny.pt   one-stage CPT from stock, dual interface,
                                 5000 steps bs1024 lr1e-4 on block_filldiv, seed
                                 20260826; declared interface = dual

Protocol: s25 harness (SEED, L=512, H=64, channel 0); mask = block outages at
rate 0.5, mask_seed 0 (Fig 7's protocol); fill A = linear; fill B = fill A with
the missing-position values permuted (fixed seed, non-identity; mean/std of the
missing multiset preserved bitwise). Forecasts via t45.quantiles_fwd (all 9
quantiles); attention shares via eval_s512_attn's hooking convention (share
received by patches containing masked points / uniform share, per layer), plus
per-patch kv detail and a horizon-adjacent-queries variant.

Float-noise floor: the same fill A forwarded (a) alone vs (b) inside a 2-row
batch [A, B] and (c) duplicated [A, A] -- batch composition changes kernel
tiling and gives the numeric floor any cross-input comparison sits above.

Window: weather wi=6 (one of the two s66 case-figure windows; chosen over
electricity wi=14 because the block/0.5 hole pattern leaves 15 of 32 patches
masked vs 17 fully observed -- suited to attention bars -- and the CPT
divergence on it is large in both forecast (max|Dmedian| 0.83 on a median span
of 0.76) and attention (per-layer share shift up to 0.125); electricity wi=14
qualifies on holes (19/32) but its attention shift is <=0.013 and its forecast
shift is small relative to the data range). Recorded in meta.

Sanity gate (printed and stored): vanilla max|Dforecast| must sit at/below the
float floor (~1e-6..1e-5 relative); CPT must be orders of magnitude larger. If
the vanilla delta is NOT tiny the script refuses to write results: the premise
failed and the figure must not be made.
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

torch.backends.cuda.matmul.allow_tf32 = False     # strict fp32 eval, as s65/s66
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s27_interface"))
sys.path.insert(0, os.path.join(EXP, "s45_pretrain"))
import run_s25_twofloor as s25
import train_s45 as t45
import eval_s45 as e45

L, H, SEED = s25.L, s25.H, s25.SEED
PATCH = 16
N_LAYER = 4                       # bolt-tiny encoder blocks
RATE = 0.5
PERM_SEED = 20260903
DS, WI = "weather", 6             # see module docstring / meta.window
TSLIB = os.path.join(ROOT, "legacy_nonstationary", "tslib", "dataset")
DATA = {"weather": os.path.join(TSLIB, "weather", "weather.csv"),
        "electricity": os.path.join(TSLIB, "electricity", "electricity.csv")}
ARMS = ("P0", "P5")
OUT = os.path.join(HERE, "s67_permvis.json")


def permute_fill(fill, mask, seed=PERM_SEED):
    """fill B: fill A with the missing-position values permuted (non-identity)."""
    rng = np.random.default_rng(seed)
    out = fill.copy()
    idx = np.flatnonzero(mask)
    p = rng.permutation(len(idx))
    assert not (p == np.arange(len(idx))).all(), "identity permutation drawn"
    out[idx] = fill[idx][p]
    return out, p


@torch.no_grad()
def quantiles_np(model, ctx_np, obs_np, iface):
    c = torch.from_numpy(ctx_np).to(next(model.parameters()).device)
    o = torch.from_numpy(obs_np).to(next(model.parameters()).device)
    return t45.quantiles_fwd(model, c, o, iface)[:, :, :H].float().cpu().numpy()


@torch.no_grad()
def attn_detail(model, ctx_np, obs_np, iface):
    """Encoder self-attention on the declared path, eval_s512_attn.py's hooking.

    Returns (share_ratio [n_layer], kv_all [n_layer, n_patch],
             kv_last4 [n_layer, n_patch]):
      share_ratio  attention received by patches containing masked points,
                   averaged over heads and all patch queries, divided by the
                   uniform share (1/n_patch) -- eval_s512_attn's quantity
      kv_all       the underlying per-kv-patch distributions (mean over heads,
                   all patch queries), rows normalised to sum 1
      kv_last4     same but averaged only over the last 4 patch queries
                   (horizon-adjacent queries)
    """
    dev = next(model.parameters()).device
    c = torch.from_numpy(ctx_np).to(dev)
    o = torch.from_numpy(obs_np).to(dev)
    cfg = model.chronos_config
    mask = o.to(c.dtype)
    context, _ = model.instance_norm(c)
    context = context.to(model.dtype)
    pc = model.patch(context)
    pm = torch.nan_to_num(model.patch(mask.to(model.dtype)), nan=0.0)
    if iface == "native":
        pc = torch.where(pm > 0.0, pc, 0.0)
    pc = torch.cat([pc, pm], dim=-1)
    embeds = model.input_patch_embedding(pc)
    attn_mask = pm.sum(dim=-1) > 0
    if cfg.use_reg_token:
        reg = torch.full((c.shape[0], 1), model.config.reg_token_id,
                         device=embeds.device)
        embeds = torch.cat([model.shared(reg), embeds], dim=-2)
        attn_mask = torch.cat([attn_mask.to(model.dtype),
                               torch.ones(reg.shape[0], 1, dtype=model.dtype,
                                          device=embeds.device)], dim=-1)
    out = model.encoder(attention_mask=attn_mask, inputs_embeds=embeds,
                        output_attentions=True)
    n_patch = L // PATCH
    miss_patch = (pm.float().mean(dim=-1) < 1.0).cpu().numpy()     # [B, n_patch]
    kv_all, kv_last4, shares = [], [], []
    for att in out.attentions:                   # per layer [B, heads, q, kv]
        a = att.float().cpu().numpy()
        a = a[:, :, 1:, 1:]                       # drop reg token q/kv
        a_all = a.mean(axis=(1, 2))               # [B, kv] over heads & queries
        a_l4 = a[:, :, -4:, :].mean(axis=(1, 2))  # [B, kv] horizon-adjacent q
        a_all = a_all / a_all.sum(axis=1, keepdims=True)
        a_l4 = a_l4 / a_l4.sum(axis=1, keepdims=True)
        kv_all.append(a_all)
        kv_last4.append(a_l4)
        uni = 1.0 / a_all.shape[1]
        got = [a_all[i, miss_patch[i]].mean() / uni for i in range(len(a_all))
               if miss_patch[i].any() and (~miss_patch[i]).any()]
        shares.append(np.array(got))
    return np.array(shares), np.stack(kv_all, 0), np.stack(kv_last4, 0)


def main():
    dev = "cuda"
    t0 = time.time()
    X, starts = s25.load_windows(DATA[DS], 300, SEED, H)
    s = int(starts[WI])
    x = X[s:s + L, 0].astype(np.float32)
    y = X[s + L:s + L + H, 0].astype(np.float32)
    mask = s25.make_mask("block", RATE, WI, 0, 1, x=x[None])[0]
    fillA = s25.fill_context(x[None], mask[None], "linear")[0]
    fillB, perm = permute_fill(fillA, mask)
    idx = np.flatnonzero(mask)
    # permutation preserves the missing-value multiset: mean/std check (float64)
    chk = {"n_missing": int(mask.sum()),
           "mean_absdiff": float(np.abs(fillA[idx].astype(np.float64).mean()
                                        - fillB[idx].astype(np.float64).mean())),
           "std_absdiff": float(np.abs(fillA[idx].astype(np.float64).std()
                                       - fillB[idx].astype(np.float64).std())),
           "max_moved": float(np.abs(fillA[idx] - fillB[idx]).max()),
           "n_positions_changed": int((fillA[idx] != fillB[idx]).sum())}
    assert chk["mean_absdiff"] == 0.0 and chk["std_absdiff"] == 0.0
    assert chk["n_positions_changed"] > 0
    obs = (~mask).astype(np.float32)
    patch_miss = mask.reshape(L // PATCH, PATCH).any(axis=1)
    print(f"window {DS} wi={WI} start={s}: missing={chk['n_missing']}/{L}, "
          f"patches containing missing {int(patch_miss.sum())}/{L // PATCH}; "
          f"perm changed {chk['n_positions_changed']} positions, "
          f"max move {chk['max_moved']:.3f}, mean/std bitwise preserved",
          flush=True)

    res = {"meta": {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": "S67 permutation-invariance, one window, made visible",
        "models": {
            "P0": {"ckpt": "experiments/s45_pretrain/s45_ckpt/arm_P0_tiny.pt",
                   "what": "stock chronos-bolt-tiny state-dict dump "
                           "(train_s53.py --arm P0; weights as shipped)",
                   "declared_iface": "native"},
            "P5": {"ckpt": "experiments/s45_pretrain/s45_ckpt/arm_P5_tiny.pt",
                   "what": "one-stage CPT retrofit from stock: dual interface, "
                           "5000 steps bs1024 lr1e-4 on block_filldiv "
                           "(train_s53.py --arm P5, seed 20260826)",
                   "declared_iface": "dual"}},
        "protocol": {"L": L, "H": H, "patch": PATCH, "seed": SEED,
                     "mech": "block", "rate": RATE, "mask_seed": 0,
                     "fill_A": "linear", "fill_B": "A with missing-position "
                             "values permuted (fixed seed, non-identity)",
                     "perm_seed": PERM_SEED, "channel": 0, "path": "declared "
                     "(filled context + observation mask)",
                     "precision": "strict fp32 (TF32 off)",
                     "quantiles": "q0..q8 = 0.1..0.9; band = [q0, q8]"},
        "window": {"dataset": DS, "wi": WI, "start": s,
                   "choice": ("one of the two s66 case-figure windows "
                              "(electricity wi=14, weather wi=6); picked over "
                              "electricity wi=14 because its block/0.5 hole "
                              "pattern (15 of 32 patches contain missing, 17 "
                              "fully observed) suits the attention bars and the "
                              "CPT divergence on it is large in both forecast "
                              "and attention; electricity wi=14 also qualifies "
                              "on holes (19/32) but its attention share shifts "
                              "are <=0.013 and its forecast shift is small "
                              "relative to the data range")},
        "permutation_check": chk,
        "n_missing_patches": int(patch_miss.sum()),
        "float_floor_def": ("same fill A forwarded alone vs inside a 2-row "
                            "batch [A,B] and duplicated [A,A]; batch "
                            "composition changes kernel tiling -- the numeric "
                            "floor for cross-input comparisons"),
    }}

    models = {a: e45.load_arm(a, dev) for a in ARMS}
    print("loaded arms:", {a: e45.ARM_IFACE[a] for a in ARMS}, flush=True)

    for a, m in models.items():
        iface = e45.ARM_IFACE[a]
        qA = quantiles_np(m, fillA[None], obs[None], iface)[0]          # [9, H]
        qB = quantiles_np(m, fillB[None], obs[None], iface)[0]
        # float floor: batch-composition variants of the SAME fill A
        two = np.concatenate([fillA[None], fillB[None]])
        obs2 = np.concatenate([obs[None], obs[None]])
        qA_inAB = quantiles_np(m, two, obs2, iface)[0]
        dup = np.concatenate([fillA[None], fillA[None]])
        qA_inAA = quantiles_np(m, dup, obs2, iface)[0]
        floor = max(float(np.abs(qA - qA_inAB).max()),
                    float(np.abs(qA - qA_inAA).max()))

        sA, kvA, kvA4 = attn_detail(m, fillA[None], obs[None], iface)
        sB, kvB, kvB4 = attn_detail(m, fillB[None], obs[None], iface)
        sA_inAB, _, _ = attn_detail(m, two, obs2, iface)   # floor on shares
        floor_share = float(np.abs(sA[:, 0] - sA_inAB[:, 0]).max())

        d_all = np.abs(qA - qB).max(axis=1)                # per quantile [9]
        d_med = float(d_all[4])
        sc = float(np.std(qA[4]))
        rel = float(d_all.max() / max(sc, 1e-9))
        res[a] = {
            "quantiles_A": np.round(qA, 5).tolist(),
            "quantiles_B": np.round(qB, 5).tolist(),
            "delta": {"median_maxabs": d_med,
                      "per_quantile_maxabs": [float(v) for v in d_all],
                      "all_maxabs": float(d_all.max()),
                      "rel_to_median_std": rel},
            "float_floor": {"forecast_maxabs": floor,
                            "share_maxabs": floor_share},
            "attn": {"share_A": [float(v) for v in sA[:, 0]],
                     "share_B": [float(v) for v in sB[:, 0]],
                     "delta_share": [float(v) for v in (sB - sA)[:, 0]],
                     "kv_all_A": np.round(kvA[:, 0, :], 6).tolist(),
                     "kv_all_B": np.round(kvB[:, 0, :], 6).tolist(),
                     "kv_last4_A": np.round(kvA4[:, 0, :], 6).tolist(),
                     "kv_last4_B": np.round(kvB4[:, 0, :], 6).tolist(),
                     "n_layers": N_LAYER},
        }
        print(f"{a} ({iface:6s}): max|Dmedian|={d_med:.3e}  "
              f"max|D| all quantiles={d_all.max():.3e}  "
              f"(rel {rel:.1e})  float floor={floor:.3e}  "
              f"share floor={floor_share:.2e}", flush=True)
        print(f"   share A={np.round(sA[:,0],4)} B={np.round(sB[:,0],4)} "
              f"D={np.round((sB-sA)[:,0],4)}", flush=True)

    # context / truth / fills for the figure
    res["window_data"] = {
        "context": np.round(x, 5).tolist(),
        "truth": np.round(y, 5).tolist(),
        "mask": mask.astype(int).tolist(),
        "fill_A": np.round(fillA, 5).tolist(),
        "fill_B": np.round(fillB, 5).tolist(),
    }

    # ---------------------------------------------------------------- gate ----
    v, c = res["P0"], res["P5"]
    gate = {
        "vanilla_at_or_below_floor":
            v["delta"]["all_maxabs"] <= max(v["float_floor"]["forecast_maxabs"],
                                            1e-6),
        "vanilla_tiny_rel": v["delta"]["rel_to_median_std"] <= 1e-3,
        "cpt_orders_larger":
            c["delta"]["median_maxabs"] >= 100 * max(v["delta"]["median_maxabs"],
                                                     1e-12),
    }
    res["sanity_gate"] = gate
    print("\nSANITY GATE:", gate, flush=True)
    if not all(gate.values()):
        print("PREMISE FAILED -- not writing results, figure must not be made",
              flush=True)
        sys.exit(1)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
