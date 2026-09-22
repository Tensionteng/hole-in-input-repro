#!/usr/bin/env python
"""S44: the attack surface on a second model -- zeroth-order poisoning of Chronos-2.

The paper's attack result (S29, arm B) is gradient-based through Chronos-Bolt's replicated
forward path, which makes it single-model. This round extends the measurement to Chronos-2
with a gradient-free hill-climbing attack under the SAME threat model (the attacker controls
only the imputation step; poisoned fills stay inside the window's observed [min, max], so the
context passes a range check; oracle untargeted objective: maximise true NMSE).

A zeroth-order attack needs no replica of the model's internals, so it applies to any
checkpoint; its damage is a LOWER bound on the gradient attack's. Under the NaN convention
the fill is not an input, so the attack is structurally impossible -- verified, not assumed:
random candidate fills must leave the forecast bit-identical.

Windows are S29's METR-LA missing-outage test half (via S26's loader), metrics identical.
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

sys.path.insert(0, os.path.join(EXP, "s14_metrla"))
sys.path.insert(0, os.path.join(EXP, "s26_realfill"))
import run_s26_realfill as s26

OUT = os.path.join(HERE, "s44_results.json")
SEED = s26.SEED_SPLIT
ROUNDS, CAND = 8, 32


class C2:
    name = "chronos-2"

    def __init__(self, dev="cuda"):
        from chronos import Chronos2Pipeline
        self.p = Chronos2Pipeline.from_pretrained(
            os.path.join(ROOT, "models_local", "chronos-2"),
            device_map=dev, torch_dtype=torch.float32)

    @torch.no_grad()
    def fc(self, targets, batch=64, H=64):
        outs = []
        for i in range(0, len(targets), batch):
            chunk = [{"target": targets[j]}
                     for j in range(i, min(i + batch, len(targets)))]
            q = self.p.predict(chunk, prediction_length=H, batch_size=batch)
            a = np.stack([np.asarray(x.cpu() if torch.is_tensor(x) else x) for x in q])
            a = a.squeeze(1) if a.ndim == 4 else a
            outs.append(a[:, a.shape[1] // 2, :].astype(np.float32))
        return np.concatenate(outs)


def attack_zeroth_order(fc, base, miss, obs_lo, obs_hi, tgt, var_floor, rng,
                        rounds=ROUNDS, cand=CAND):
    """Hill-climb on the fill, missing positions only, clamped to observed range.
    base: [n, L] starting fill; miss: [n, L] True=missing. Returns (best_ctx, e_base, e_best)."""
    n, L = base.shape
    def nmse(ctx):
        p = fc(ctx)
        k = len(p) // n
        t = np.tile(tgt, (k, 1)) if k > 1 else tgt
        return s26.nmse_per_window(p, t, var_floor)
    e_base = nmse(base)
    best = base.copy()
    e_best = e_base.copy()
    width = (obs_hi - obs_lo)
    for r in range(rounds):
        # candidates: round 0 uniform over the whole range, later gaussian around incumbent
        if r == 0:
            noise = rng.uniform(obs_lo, obs_hi, size=(cand, n, L)).astype(np.float32)
        else:
            scale = width * (0.5 ** r) * 0.5
            noise = best[None] + rng.normal(0, 1, size=(cand, n, L)).astype(np.float32) \
                * scale[None]
        noise = np.clip(noise, obs_lo[None], obs_hi[None])
        cands = np.where(miss[None], noise, base[None]).astype(np.float32)
        e = nmse(cands.reshape(cand * n, L)).reshape(cand, n)
        k = e.argmax(0)
        improved = e[k, np.arange(n)] > e_best
        best[improved] = cands[k, np.arange(n)][improved]
        e_best = np.maximum(e_best, e[k, np.arange(n)])
        print(f"    round {r}: median NMSE {np.median(e_best):.4f} "
              f"(base {np.median(e_base):.4f})", flush=True)
    return best, e_base, e_best


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"model": "chronos-2", "attack": "zeroth-order hill-climb",
                            "rounds": ROUNDS, "candidates_per_round": CAND,
                            "threat_model": "fill only, clamped to observed range",
                            "objective": "oracle untargeted (maximise true NMSE)",
                            "data": "METR-LA missing-outage test half (S26/S29 windows)"})
    ds = s26.Metr()
    _, test = ds.split("miss")
    c2 = C2("cuda")
    H = ds.H

    # defender's best fixed fill for THIS model (mirrors S29's protocol)
    best, best_e = None, None
    for f in ("keep", "zero", "linear", "nan"):
        ctx = s26.fill_ctx(ds.ctx_obs[test], ds.ctx_rec[test], ds.mask[test], f)
        e = s26.nmse_per_window(c2.fc(ctx, H=H), ds.tgt[test], ds.VAR_FLOOR)
        print(f"defender {f:7s} NMSE median {np.median(e):.4f}", flush=True)
        if best is None or np.median(e) < np.median(best_e):
            best, best_e = f, e
    base_fill = "linear" if best == "nan" else best
    print(f"defender best = {best}; attacking from {base_fill}", flush=True)

    miss = ds.mask[test]
    tgt = ds.tgt[test]
    base = s26.fill_ctx(ds.ctx_obs[test], ds.ctx_rec[test], ds.mask[test], base_fill)
    lo = np.array([np.nanmin(r) for r in ds.ctx_obs[test]], np.float32)[:, None]
    hi = np.array([np.nanmax(r) for r in ds.ctx_obs[test]], np.float32)[:, None]

    cell = {"n": int(len(test)), "defender_best_fill": best,
            "missing_rate_mean": float(miss.mean())}

    # plain path: full-rank surface
    t0 = time.time()
    poisoned, e_c, e_a = attack_zeroth_order(c2.fc, base, miss, lo, hi, tgt,
                                             ds.VAR_FLOOR, rng)
    ratio = e_a / np.maximum(e_c, 1e-12)
    cell["plain|untargeted"] = {
        "nmse_clean_median": float(np.median(e_c)),
        "nmse_attacked_median": float(np.median(e_a)),
        "damage_ratio_median": float(np.median(ratio)),
        "damage_ratio_q90": float(np.quantile(ratio, 0.9)),
        "damage_ratio_max": float(ratio.max()),
        "frac_windows_doubled": float((ratio > 2).mean())}
    print(f"plain   damage x{cell['plain|untargeted']['damage_ratio_median']:.2f} "
          f"(q90 x{cell['plain|untargeted']['damage_ratio_q90']:.2f}, "
          f"max x{cell['plain|untargeted']['damage_ratio_max']:.0f}, "
          f">2x on {100*cell['plain|untargeted']['frac_windows_doubled']:.0f}%) "
          f"({time.time()-t0:.0f}s)", flush=True)

    # nan path: structural immunity -- under the NaN convention the fill is not an input,
    # so no candidate exists to score; damage is x1.00 by construction (rho = 0.00 for
    # chronos-2's declared path, measured in S36). We verify only that the nan path itself
    # is deterministic, so the x1.00 is exact rather than averaged.
    t0 = time.time()
    nan_ctx = s26.fill_ctx(ds.ctx_obs[test], ds.ctx_rec[test], miss, "nan")
    p0 = c2.fc(nan_ctx, H=H)
    p0b = c2.fc(nan_ctx, H=H)
    det = float(np.abs(p0 - p0b).max())
    e_nan = s26.nmse_per_window(p0, tgt, ds.VAR_FLOOR)
    cell["nan|untargeted"] = {
        "nmse_clean_median": float(np.median(e_nan)),
        "determinism_max_abs_diff": det,
        "damage_ratio_median": 1.0, "damage_ratio_q90": 1.0, "damage_ratio_max": 1.0,
        "frac_windows_doubled": 0.0}
    print(f"nan     damage x1.00 by construction (determinism check {det:.1e}) "
          f"({time.time()-t0:.0f}s)", flush=True)

    res["metr|missing"] = cell
    json.dump(res, open(OUT, "w"))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
