#!/usr/bin/env python
"""S69 corpus generator + shared torch-free helpers.

Arena (DESIGN.md): per series, phi ~ U(0.7, 0.95), stationary AR(1) target
x_t = phi x_{t-1} + sigma eps_t with sigma = sqrt(1 - phi^2) (exact unit variance --
"standardized to unit variance" is read as the stationary law having variance 1, so the
joint law of (x, c) given (phi, rho) is exactly Gaussian and the Bayes yardstick is exact;
no empirical re-scaling). Context c_t = rho x_t + sqrt(1 - rho^2) eta_t, rho ~ U(0.5, 0.9).
Series length 1024; train pool (default 100k) and eval pool (256) come from disjoint rng
streams (different SeedSequence stems), so no eval series can appear in training.

Also hosts the shared, torch-free pieces used by train_s69.py / eval_s69.py / bayes_s69.py:
  - training-time diverse-regime corruption (corrupt_channel)
  - deterministic eval masks / blend noises (paired across arms and q cells)
  - eval cell assembly (cell_inputs)
  - the two-sided perturbation implied-weight measurement (perturbed_weight), used
    verbatim for the model (eval_s69) and for the ridge gate G2 (bayes_s69)
  - code_hash() for git-style provenance in every output json.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

L, H, T = 512, 64, 1024          # context, horizon, series length
EVAL_START = T - L - H           # = 448: the final window of each eval series (stationary
                                 # process, so the window position is immaterial)
BLOCK = 24                       # run length for block masks (matches s25 geometry)
TRAIN_SEED = 690_000_1
EVAL_SEED = 690_000_2
EVAL_CELL_SEED = 690_000_3       # masks/noises of the eval grid
G2_SEED = 690_000_4              # ridge-gate simulations

ARMS = {"A": {"iface": "native", "regime": "filtered"},
        "B": {"iface": "native", "regime": "diverse"},
        "C": {"iface": "dual", "regime": "filtered"},
        "D": {"iface": "dual", "regime": "diverse"}}

Q_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EVAL_MECHS = ("mcar", "block")
EVAL_MECHS_V2 = ("mcar", "block", "tailblock")     # s69b adds the tail-censor regime
EVAL_RATES = (0.3, 0.7)
DELTA = 0.1                      # E2 perturbation size

# ---- s69b (round 2): harder process family -- AR(1) + K Gaussian harmonics ----
HARM_K = 3                       # harmonics per series
HARM_BANK = (192, 384, 768)      # period bank (points); tuned via G0b (tune_s69b.py),
HARM_SHARE = (0.6, 0.85)         # harmonic share ~ U(*) -- approved amendment 2026-09-03
TRAIN_SEED_V2 = 690_100_1
EVAL_SEED_V2 = 690_100_2


# ----------------------------------------------------------------- corpus ----

def gen_pool(n, seed):
    """Returns x, c [n, T] float32 and per-series phi, rho [n] float32."""
    rng = np.random.default_rng(np.random.SeedSequence([seed]))
    phi = rng.uniform(0.7, 0.95, n)
    rho = rng.uniform(0.5, 0.9, n)
    sigma = np.sqrt(1.0 - phi ** 2)              # stationary unit variance
    x = np.empty((n, T), np.float64)
    x[:, 0] = rng.standard_normal(n)             # stationary start N(0, 1)
    for t in range(1, T):
        x[:, t] = phi * x[:, t - 1] + sigma * rng.standard_normal(n)
    eta = rng.standard_normal((n, T))
    c = rho[:, None] * x + np.sqrt(1.0 - rho ** 2)[:, None] * eta
    return (x.astype(np.float32), c.astype(np.float32),
            phi.astype(np.float32), rho.astype(np.float32))


def pool_report(x, c, phi, rho, name, k=2000):
    """Sanity statistics over the first k series (printed, not gating)."""
    xs, cs = x[:k].astype(np.float64), c[:k].astype(np.float64)
    m = min(len(xs), 200)
    lag1 = [np.corrcoef(xs[i, 1:], xs[i, :-1])[0, 1] for i in range(m)]
    xyc = [np.corrcoef(xs[i], cs[i])[0, 1] for i in range(m)]
    return {f"{name}_var_x": float(xs.var()), f"{name}_var_c": float(cs.var()),
            f"{name}_lag1_mean": float(np.mean(lag1)),
            f"{name}_lag1_vs_phi_mae": float(np.abs(np.mean(lag1) - phi[:m].mean())),
            f"{name}_corr_xc_mean": float(np.mean(xyc)),
            f"{name}_corr_vs_rho_mae": float(np.abs(np.mean(xyc) - rho[:m].mean()))}


# ------------------------------------------------------- s69b process (v2) ----

def gen_pool_v2(n, seed, bank=HARM_BANK, share=HARM_SHARE, k=HARM_K):
    """AR(1) + K Gaussian harmonics, exactly unit variance, jointly Gaussian:
    x_t = ar_t + sum_k [a_k cos(2 pi t / p_k) + b_k sin(2 pi t / p_k)],
    a_k, b_k ~ N(0, h/k) iid, h ~ U(*share) the harmonic energy share, periods p_k
    drawn per series from `bank` WITHOUT replacement (distinct components);
    ar_t stationary AR(1) with Var = 1 - h (innovation sqrt((1-h)(1-phi^2))).
    Context c as in v1. Returns x, c [n, T] float32, phi, rho, h [n] float32,
    periods [n, k] int32."""
    rng = np.random.default_rng(np.random.SeedSequence([seed]))
    phi = rng.uniform(0.7, 0.95, n)
    rho = rng.uniform(0.5, 0.9, n)
    h = rng.uniform(share[0], share[1], n)
    bank = np.asarray(bank)
    pidx = np.stack([rng.choice(len(bank), size=k, replace=False)
                     for _ in range(n)])            # [n, k] distinct periods per series
    periods = bank[pidx].astype(np.int32)
    sig = np.sqrt(h / k)                           # per-harmonic amplitude std
    a = sig[:, None] * rng.standard_normal((n, k))
    b = sig[:, None] * rng.standard_normal((n, k))
    t = np.arange(T).astype(np.float64)
    x = np.zeros((n, T), np.float64)
    for j in range(k):
        w = 2.0 * np.pi / periods[:, j]
        x += a[:, j, None] * np.cos(w[:, None] * t) + \
            b[:, j, None] * np.sin(w[:, None] * t)
    ar = np.empty((n, T), np.float64)              # independent stationary AR(1) part
    sigma_ar = np.sqrt((1.0 - h) * (1.0 - phi ** 2))
    ar[:, 0] = np.sqrt(1.0 - h) * rng.standard_normal(n)
    for tt in range(1, T):
        ar[:, tt] = phi * ar[:, tt - 1] + sigma_ar * rng.standard_normal(n)
    x += ar
    eta = rng.standard_normal((n, T))
    c = rho[:, None] * x + np.sqrt(1.0 - rho ** 2)[:, None] * eta
    return (x.astype(np.float32), c.astype(np.float32), phi.astype(np.float32),
            rho.astype(np.float32), h.astype(np.float32), periods)


def sxx_kernel_v2(phi, h, periods, n_t):
    """Cov(x_t, x_s) kernel matrix [n_t, n_t] for the v2 process (float64)."""
    t = np.arange(n_t)
    d = np.abs(t[:, None] - t[None, :])
    sxx = (1.0 - h) * phi ** d
    for p in periods:
        sxx = sxx + (h / len(periods)) * np.cos(2.0 * np.pi * d / p)
    return sxx

def mask_block(rng, rate):
    """Runs of BLOCK covering ~rate of L in total (runs may overlap)."""
    m = np.zeros(L, bool)
    n_runs = max(1, int(round(rate * L / BLOCK)))
    for s in rng.integers(0, L - BLOCK + 1, size=n_runs):
        m[s:s + BLOCK] = True
    return m


def corrupt_channel(v, rng):
    """The diverse regime, one channel of one training window (DESIGN.md):
    clean with p=0.25; else mcar rate~U(0.05,0.9) or block rate~U(0.1,0.9), 50/50;
    masked positions filled with fill = q*truth + (1-q)*N(0,1), q~U(0,1).

    Returns (filled, obs) with obs = 1 at observed positions."""
    if rng.random() < 0.25:
        return v.copy(), np.ones(L, np.float32)
    if rng.random() < 0.5:
        mask = rng.random(L) < rng.uniform(0.05, 0.9)
    else:
        mask = mask_block(rng, rng.uniform(0.1, 0.9))
    q = float(rng.uniform(0.0, 1.0))
    noise = rng.standard_normal(L)
    out = v.copy()
    out[mask] = q * v[mask] + (1.0 - q) * noise[mask]
    return out, (~mask).astype(np.float32)


def mask_tailblock(rng, rate):
    """Tail-censor: the LAST ceil(rate*L) positions of the window holed (the
    forecast-relevant recent past). Deterministic given rate; rng unused."""
    m = np.zeros(L, bool)
    m[L - int(np.ceil(rate * L)):] = True
    return m


def corrupt_channel_v2(v, rng):
    """s69b diverse regime: as corrupt_channel, but the non-clean 75% is split in
    thirds over {mcar U(0.05,0.9), block U(0.1,0.9), tailblock U(0.1,0.9)}."""
    if rng.random() < 0.25:
        return v.copy(), np.ones(L, np.float32)
    u = rng.random()
    if u < 1.0 / 3.0:
        mask = rng.random(L) < rng.uniform(0.05, 0.9)
    elif u < 2.0 / 3.0:
        mask = mask_block(rng, rng.uniform(0.1, 0.9))
    else:
        mask = mask_tailblock(rng, rng.uniform(0.1, 0.9))
    q = float(rng.uniform(0.0, 1.0))
    noise = rng.standard_normal(L)
    out = v.copy()
    out[mask] = q * v[mask] + (1.0 - q) * noise[mask]
    return out, (~mask).astype(np.float32)


# ------------------------------------------------------------- eval cells ----

def _cell_rng(series, mech, rate, what, channel=0):
    """Deterministic per-(series, mechanism, rate) streams: masks and blend noises are
    paired across arms and across the q grid within a (mech, rate) column."""
    return np.random.default_rng(np.random.SeedSequence(
        [EVAL_CELL_SEED, int(series), {"mcar": 0, "block": 1, "tailblock": 2}[mech],
         int(round(rate * 100)), {"mask": 0, "noise": 1}[what], int(channel)]))


def eval_target_mask(series, mech, rate):
    """[L] bool, True = repaired (missing) on the target channel."""
    rng = _cell_rng(series, mech, rate, "mask")
    if mech == "mcar":
        return rng.random(L) < rate
    if mech == "tailblock":
        return mask_tailblock(rng, rate)
    return mask_block(rng, rate)


def eval_noises(series, mech, rate):
    """(noise_target, noise_context), each [L] standard normal, shared across the q grid."""
    nt = _cell_rng(series, mech, rate, "noise", 0).standard_normal(L)
    nc = _cell_rng(series, mech, rate, "noise", 1).standard_normal(L)
    return nt, nc


def cell_inputs(x_ctx, c_ctx, mask_t, noise_t, noise_c, q_r, q_c):
    """The (q_r, q_c) repair of one eval window. Returns (fx, fc, flag_t, flag_c):
    fx/fc are the CONTENT channels (finite); flags are 1=observed (dual arms only --
    native arms ignore them). The context channel is a whole-context repair: every
    position declared missing and blend-filled at q_c (DESIGN.md E1/E2, risks section)."""
    fx = x_ctx.copy()
    fx[mask_t] = q_r * x_ctx[mask_t] + (1.0 - q_r) * noise_t[mask_t]
    fc = q_c * c_ctx + (1.0 - q_c) * noise_c
    flag_t = (~mask_t).astype(np.float32)
    flag_c = np.zeros(L, np.float32)
    return fx, fc, flag_t, flag_c


def flat_obs(fx, fc):
    """The flat observation vector shared by the model eval and the Bayes/ridge paths:
    z = [target content (L), context content (L)]."""
    return np.concatenate([fx, fc])


TGT = np.arange(0, L)            # column slices of the flat observation
CTX = np.arange(L, 2 * L)


def perturbed_weight(predict_fn, z0, cols, delta=DELTA):
    """E2 implied weights, per row: simultaneous two-sided +/-delta perturbation of the
    selected entries, horizon-mean response normalized per perturbed position.
    cols: [D] bool (shared columns, e.g. the whole context channel or G2's fixed mask)
    or [N, D] bool (per-row columns, e.g. each window's own target-repair mask).
    Returns [N] per-row weights. For a linear predictor every row's weight equals the
    mean regression coefficient on the selected columns exactly -- the property G0/G2
    rely on (DESIGN.md E2/E3). predict_fn: [N, D] -> [N, H]."""
    if cols.ndim == 1:
        cols = np.broadcast_to(cols, z0.shape)
    zp = z0.copy()
    zp[cols] += delta
    zm = z0.copy()
    zm[cols] -= delta
    dy = predict_fn(zp) - predict_fn(zm)                    # [N, H]
    k = cols.sum(axis=1)
    return dy.mean(axis=1) / (2.0 * delta * k)


# ------------------------------------------------------------------ meta ----

def code_hash():
    h = hashlib.sha256()
    for f in ("DESIGN.md", "gen_s69_corpus.py", "train_s69.py", "eval_s69.py",
              "bayes_s69.py", "analyze_s69.py"):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            h.update(f.encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=100_000)
    ap.add_argument("--n-eval", type=int, default=256)
    ap.add_argument("--round", default="s69", choices=["s69", "s69b"])
    ap.add_argument("--bank", default=None,
                    help="s69b: comma-separated period bank override (G0b tuning)")
    ap.add_argument("--share", default=None,
                    help="s69b: harmonic share range override, e.g. 0.5,0.8")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(
        HERE, "s69_corpus" if args.round == "s69" else "s69b_corpus")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    if args.round == "s69":
        xt, ct, phit, rhot = gen_pool(args.n_train, TRAIN_SEED)
        xe, ce, phie, rhoe = gen_pool(args.n_eval, EVAL_SEED)
        np.savez(os.path.join(out, "train.npz"), x=xt, c=ct, phi=phit, rho=rhot)
        np.savez(os.path.join(out, "eval.npz"), x=xe, c=ce, phi=phie, rho=rhoe)
        stats = {"n_train": args.n_train, "n_eval": args.n_eval, "T": T, "L": L, "H": H,
                 "train_seed": TRAIN_SEED, "eval_seed": EVAL_SEED,
                 "minutes": (time.time() - t0) / 60, "code_hash": code_hash(),
                 **pool_report(xt, ct, phit, rhot, "train"),
                 **pool_report(xe, ce, phie, rhoe, "eval")}
    else:
        bank = tuple(int(v) for v in args.bank.split(",")) if args.bank else HARM_BANK
        share = (tuple(float(v) for v in args.share.split(",")) if args.share
                 else HARM_SHARE)
        xt, ct, phit, rhot, ht, pt = gen_pool_v2(args.n_train, TRAIN_SEED_V2, bank,
                                                 share)
        xe, ce, phie, rhoe, he, pe = gen_pool_v2(args.n_eval, EVAL_SEED_V2, bank, share)
        np.savez(os.path.join(out, "train.npz"), x=xt, c=ct, phi=phit, rho=rhot,
                 h=ht, periods=pt)
        np.savez(os.path.join(out, "eval.npz"), x=xe, c=ce, phi=phie, rho=rhoe,
                 h=he, periods=pe)
        stats = {"n_train": args.n_train, "n_eval": args.n_eval, "T": T, "L": L, "H": H,
                 "round": "s69b", "harm_k": HARM_K, "harm_bank": list(bank),
                 "harm_share": list(share),
                 "train_seed": TRAIN_SEED_V2, "eval_seed": EVAL_SEED_V2,
                 "minutes": (time.time() - t0) / 60, "code_hash": code_hash(),
                 **pool_report(xt, ct, phit, rhot, "train"),
                 **pool_report(xe, ce, phie, rhoe, "eval")}
    with open(os.path.join(out, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    print(json.dumps(stats, indent=1), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
