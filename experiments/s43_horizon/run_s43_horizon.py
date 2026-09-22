#!/usr/bin/env python
"""S43: is the rank trichotomy an artefact of H=64? -- horizon spot-check.

The paper's characterisation is measured at L=512, H=64 (one native bolt block), and the
Limitations say it should be re-measured at other horizons. This round reruns the
gradient-free permutation probe (rho of Eq. (rho)) at H=24 (a slice of the native block)
and H=192 (a 3-block autoregressive median rollout through the same head), against the
H=64 baseline which must reproduce S30/S36's stored values (gate).

Rollout note: the stock pipeline does not accept an explicit observation mask, so for
fill_mask at H=192 we roll the differentiable head forward manually (append the median,
slide the window, extend the observation mask with ones). rho is a ratio of output changes
under the SAME forecast procedure, so the procedure only needs to be consistent within a
convention, which it is.
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

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
import run_s25_twofloor as s25

L = s25.L
MECHS = ("mcar", "block", "mnar_high")
RATE = 0.3
N_WIN = 60
HORIZONS = (24, 64, 128, 192)
CONVS = ("plain_fill", "fill_mask", "nan")
DATASETS = ("ETTh1", "ETTm1", "weather")
OUT = os.path.join(HERE, "s43_results.json")
SEED = s25.SEED


# ------------------------------------------------------------- forecasts ----

def make_fc(bolt, horizon):
    """fc(ctx_np, conv, obs) -> median forecast [n, horizon] under the convention.
    ctx is finite-filled; obs marks 1=observed (used by fill_mask and nan)."""
    def fc(ctx, conv, obs=None):
        if conv == "nan":
            c = ctx.copy()
            c[obs == 0] = np.nan
        else:
            c = ctx
        if horizon <= 64:
            if conv == "fill_mask":
                return bolt.median_np(c, obs.astype(np.float32))[:, :horizon]
            return bolt.median_np(c, None)[:, :horizon]
        # autoregressive median rollout through the same head
        cur = c.copy()
        cur_obs = obs.copy() if obs is not None else None
        out = []
        produced = 0
        while produced < horizon:
            if conv == "fill_mask":
                y = bolt.median_np(cur, cur_obs.astype(np.float32))
            else:
                y = bolt.median_np(cur, None)
            take = min(64, horizon - produced)
            out.append(y[:, :take])
            cur = np.concatenate([cur[:, take:], y[:, :take]], axis=1)
            if cur_obs is not None:
                cur_obs = np.concatenate(
                    [cur_obs[:, take:], np.ones_like(y[:, :take])], axis=1)
            produced += take
        return np.concatenate(out, axis=1)

    return fc


# ------------------------------------------------------------------ probe ----

def perm_probe(bolt, fc, filled, mask, n_perm=4, rng=None):
    """s25.perm_invariance with a parameterised forecast horizon. filled: [n, L] finite,
    mask: [n, L] True=missing."""
    rng = rng or np.random.default_rng(SEED)
    out = {}
    obs = (~mask).astype(np.float32)
    for conv in CONVS:
        y0 = fc(filled, conv, obs)
        d_perm, d_redraw = [], []
        for _ in range(n_perm):
            cp, cr = filled.copy(), filled.copy()
            for i in range(len(filled)):
                idx = np.flatnonzero(mask[i])
                if len(idx) < 2:
                    continue
                cp[i, idx] = filled[i, idx][rng.permutation(len(idx))]
                src = filled[i, ~mask[i]]
                cr[i, idx] = rng.choice(src, size=len(idx)) if len(src) else 0.0
            d_perm.append(np.sqrt(((fc(cp, conv, obs) - y0) ** 2).mean(1)))
            d_redraw.append(np.sqrt(((fc(cr, conv, obs) - y0) ** 2).mean(1)))
        scale = np.sqrt((y0 ** 2).mean(1)) + 1e-12
        dp = np.mean(d_perm, 0) / scale
        dr = np.mean(d_redraw, 0) / scale
        out[conv] = {"rho": float(dp.mean() / max(dr.mean(), 1e-15)),
                     "perm": float(dp.mean()), "redraw": float(dr.mean())}
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    bolt = s25.Bolt("cuda")
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"L": L, "rate": RATE, "n_win": N_WIN, "n_perm": 4,
                            "horizons": HORIZONS, "mechs": MECHS, "datasets": DATASETS,
                            "rollout": "autoregressive median through the native head for H>64"})
    rng = np.random.default_rng(SEED)
    for ds in DATASETS:
        X, st = s25.load_windows(s25.DATASETS[ds], N_WIN, SEED, 192)
        for mech in MECHS:
            _, filled, mask, _ = s25.build_eval_batch(X, st, mech, RATE, 192,
                                                      fill="linear")
            for hz in HORIZONS:
                key = f"{ds}|{mech}|{hz}"
                if key in res.get("cells", {}):
                    continue
                t0 = time.time()
                fc = make_fc(bolt, hz)
                r = perm_probe(bolt, fc, filled, mask, rng=rng)
                res.setdefault("cells", {})[key] = r
                json.dump(res, open(OUT, "w"))
                print(f"{key:24s} " + "  ".join(
                    f"{c}: rho={r[c]['rho']:.2e}" for c in CONVS) +
                    f"  ({time.time()-t0:.0f}s)", flush=True)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
