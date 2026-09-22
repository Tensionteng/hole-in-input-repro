#!/usr/bin/env python
"""S69 E3: the closed-form Bayes yardstick, and gates G0/G2 (v1) plus G0b (s69b).

Everything in the arena is jointly Gaussian given the per-series parameters, so the
Bayes predictor of the target horizon given any declared/corrupted observation set is
exact numeric conditioning. Observation model per eval cell (mech, rate, q_r, q_c),
matching the eval harness exactly (gen_s69_corpus.cell_inputs):
  target, observed position t:   o_t = x_t                      (noiseless)
  target, repaired position t:   o_t = q_r * x_t + e_t,  e_t ~ N(0, (1-q_r)^2)
  context, every position t:     o_{L+t} = q_c * c_t + e'_t, e'_t ~ N(0, (1-q_c)^2)
(the whole context channel is a declared repair -- DESIGN.md E1; q=1 recovers the
noiseless observation, q=0 a pure-noise one).

v2 (s69b) process: x = AR(1) (variance 1-h) + K Gaussian harmonics (share h, periods
drawn per series). Still jointly Gaussian: Cov(x_t, x_s) = (1-h) phi^|t-s| +
sum_k (h/K) cos(2 pi (t-s) / p_k) -- the conditioning code is unchanged, only the
kernel. v2 adds the tailblock mechanism (last ceil(rate*L) positions holed).

Per cell we report the optimal implied weights w*_R / w*_C (the analytic regression
coefficients of the Bayes predictor on the respective fill entries, averaged over
positions and horizons -- identical to applying the E2 perturbation definition to the
Bayes predictor, which is exactly linear) and the MSE* floor.

Gates:
  G0   w*_R(q_r=0) == 0 exactly;  MSE*(q_r=1, q_c=1, block, 0.7) ~= clean-floor MSE*.
  G0b  (s69b only, AMENDED 2026-09-03, see s69_notes.md): HEADROOM, two separate
       reference quantities -- information-loss gap median MSE*(0,0)/MSE*(1,1) >= 2.0
       at tailblock 0.7, AND fill-poisoning (naive-trust) gap median >= 1.5 at
       block 0.7. Every cell carries both `mse` (optimal floor) and `mse_naive`
       (naive-trust level); they are never mixed.
  G2   a ridge regression fitted on simulated true features, scored through the SHARED
       perturbation procedure (g69.perturbed_weight), recovers w*_R / w*_C within tol.

Pure numpy/CPU. --round s69 reproduces v1 (s69_bayes.json); --round s69b writes
s69b_bayes.json.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import gen_s69_corpus as g69

L, H = g69.L, g69.H
G2_TOL = 0.02
G0B_TAIL = 2.0
G0B_BLOCK = 1.5


# ------------------------------------------------------------- covariance ----

def _cov_from_sxx(sxx, rho):
    """Shared block builder: sxx [L+H, L+H] target kernel + context correlation rho.
    Returns S_obs [2L, 2L], S_fo [H, 2L], S_ff [H, H]."""
    tc = np.arange(L)
    dc = np.abs(tc[:, None] - tc[None, :])
    scc = rho ** 2 * sxx[:L, :L] + (1 - rho ** 2) * np.eye(L)
    sxc = rho * sxx[:L, :L]                            # Cov(x_t, c_s), ctx x ctx
    sfc = rho * sxx[L:, :L]                            # Cov(x_fut, c_ctx)
    S_obs = np.block([[sxx[:L, :L], sxc], [sxc.T, scc]])
    S_fo = np.concatenate([sxx[L:, :L], sfc], axis=1)
    return S_obs, S_fo, sxx[L:, L:]


def series_covariances(phi, rho):
    """v1: unit-variance AR(1) target."""
    t = np.arange(L + H)
    d = np.abs(t[:, None] - t[None, :])
    return _cov_from_sxx(phi ** d, rho)


def series_covariances_v2(phi, rho, h, periods):
    """v2: AR(1) + Gaussian harmonics kernel."""
    return _cov_from_sxx(g69.sxx_kernel_v2(phi, h, periods, L + H), rho)


def condition(S_obs, S_fo, S_ff, blend, noise_var):
    """Bayes predictor of the future given o = blend * [x_ctx, c_ctx] + e,
    e ~ N(0, diag(noise_var)). Returns (A [H, 2L], mse_star scalar)."""
    S_oo = blend[:, None] * S_obs * blend[None, :]
    S_oo[np.diag_indices_from(S_oo)] += noise_var
    S_fo_b = S_fo * blend[None, :]
    A = np.linalg.solve(S_oo.T, S_fo_b.T).T          # A = S_fo_b @ inv(S_oo)
    cond = S_ff - A @ S_fo_b.T
    return A, float(np.trace(cond) / H)


def naive_trust_mse(S_obs, S_fo, S_ff, A_clean, blend_true, nv_true):
    """Fill-poisoning reference line (s69b amendment, 2026-09-03): MSE of the linear
    predictor FIT as if all fills were clean (A_clean, computed once per series),
    evaluated on data whose fills actually follow (blend_true, nv_true). This is the
    expected level of an interface that cannot down-weight declared-garbage content --
    the native-arm reference for P4. Kept strictly separate from the optimal floor
    (mse) in all outputs."""
    S_oo_t = blend_true[:, None] * S_obs * blend_true[None, :]
    S_oo_t[np.diag_indices_from(S_oo_t)] += nv_true
    S_fo_t = S_fo * blend_true[None, :]
    t1 = float(np.sum(A_clean * S_fo_t))
    t2 = float(np.sum((A_clean @ S_oo_t) * A_clean))
    return float((np.trace(S_ff) - 2.0 * t1 + t2) / H)


def cell_obs_vectors(mask_t, q_r, q_c):
    """(blend, noise_var) for one cell: the eval observation structure."""
    blend = np.concatenate([np.where(mask_t, q_r, 1.0), np.full(L, q_c)])
    nv = np.concatenate([np.where(mask_t, (1 - q_r) ** 2, 0.0),
                         np.full(L, (1 - q_c) ** 2)])
    return blend, nv


# ------------------------------------------------------------------- G2 ----

def simulate_windows(phi, rho, n, rng):
    """v1 process: n independent (ctx, fut) draws. x_ctx [n, L], c_ctx [n, L],
    fut [n, H] (float64)."""
    sigma = np.sqrt(1 - phi ** 2)
    x = np.empty((n, L + H))
    x[:, 0] = rng.standard_normal(n)
    for t in range(1, L + H):
        x[:, t] = phi * x[:, t - 1] + sigma * rng.standard_normal(n)
    eta = rng.standard_normal((n, L))
    c = rho * x[:, :L] + np.sqrt(1 - rho ** 2) * eta
    return x[:, :L], c, x[:, L:]


def simulate_windows_v2(phi, rho, h, periods, n, rng):
    """v2 process: AR(1) (var 1-h) + K Gaussian harmonics (share h)."""
    t = np.arange(L + H, dtype=np.float64)
    k = len(periods)
    sig = np.sqrt(h / k)
    x = np.zeros((n, L + H))
    for p in periods:
        a = sig * rng.standard_normal(n)
        b = sig * rng.standard_normal(n)
        w = 2.0 * np.pi / p
        x += a[:, None] * np.cos(w * t) + b[:, None] * np.sin(w * t)
    ar = np.empty((n, L + H))
    sigma_ar = np.sqrt((1.0 - h) * (1.0 - phi ** 2))
    ar[:, 0] = np.sqrt(1.0 - h) * rng.standard_normal(n)
    for tt in range(1, L + H):
        ar[:, tt] = phi * ar[:, tt - 1] + sigma_ar * rng.standard_normal(n)
    x += ar
    eta = rng.standard_normal((n, L))
    c = rho * x[:, :L] + np.sqrt(1 - rho ** 2) * eta
    return x[:, :L], c, x[:, L:]


def gate_g2(series_params, cells, n_sim, smoke, v2):
    """Ridge on simulated true features must reproduce the analytic weights when scored
    by the shared perturbation procedure."""
    rng = np.random.default_rng(np.random.SeedSequence([g69.G2_SEED, int(smoke)]))
    out = []
    for si, prm in enumerate(series_params):
        if v2:
            x_ctx, c_ctx, fut = simulate_windows_v2(prm["phi"], prm["rho"], prm["h"],
                                                    prm["periods"], n_sim, rng)
            S_obs, S_fo, S_ff = series_covariances_v2(prm["phi"], prm["rho"], prm["h"],
                                                      prm["periods"])
        else:
            x_ctx, c_ctx, fut = simulate_windows(prm["phi"], prm["rho"], n_sim, rng)
            S_obs, S_fo, S_ff = series_covariances(prm["phi"], prm["rho"])
        for (mech, rate, q_r, q_c) in cells:
            mask_t = g69.eval_target_mask(si, mech, rate)
            nt = rng.standard_normal((n_sim, L))
            nc = rng.standard_normal((n_sim, L))
            fx = x_ctx.copy()
            fx[:, mask_t] = q_r * x_ctx[:, mask_t] + (1 - q_r) * nt[:, mask_t]
            fc = q_c * c_ctx + (1 - q_c) * nc
            Z = np.concatenate([fx, fc], axis=1)             # [n, 2L]
            W = np.linalg.solve(Z.T @ Z + 1.0 * np.eye(2 * L), Z.T @ fut)
            predict = lambda Zq: Zq @ W                      # exact linear predictor
            z0 = Z[:4096]
            cols_r = np.zeros(2 * L, bool)
            cols_r[:L][mask_t] = True
            cols_c = np.zeros(2 * L, bool)
            cols_c[L:] = True
            wR = float(g69.perturbed_weight(predict, z0, cols_r).mean())
            wC = float(g69.perturbed_weight(predict, z0, cols_c).mean())
            blend, nv = cell_obs_vectors(mask_t, q_r, q_c)
            A, _ = condition(S_obs, S_fo, S_ff, blend, nv)
            wR_star = float(A[:, :L][:, mask_t].mean())
            wC_star = float(A[:, L:].mean())
            out.append({"series": si, "cell": f"{mech}|{rate}|{q_r}|{q_c}",
                        "wR_ridge": wR, "wR_star": wR_star,
                        "wC_ridge": wC, "wC_star": wC_star,
                        "err": max(abs(wR - wR_star), abs(wC - wC_star))})
            print(f"  G2 s{si} {mech}|{rate}|qr={q_r}|qc={q_c}: "
                  f"wR {wR:.4f}/{wR_star:.4f}  wC {wC:.4f}/{wC_star:.4f}", flush=True)
    worst = max(o["err"] for o in out)
    return {"cells": out, "max_abs_err": worst, "tol": G2_TOL, "pass": worst <= G2_TOL,
            "n_sim": n_sim}


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="s69", choices=["s69", "s69b"])
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--n-series", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    v2 = args.round == "s69b"
    corpus = args.corpus or os.path.join(
        HERE, "s69b_corpus" if v2 else "s69_corpus")
    out_path = args.out or os.path.join(
        HERE, "s69b_bayes.json" if v2 else "s69_bayes.json")
    t0 = time.time()

    z = np.load(os.path.join(corpus, "eval.npz"))
    phi_all = z["phi"].astype(np.float64)
    rho_all = z["rho"].astype(np.float64)
    n_series = args.n_series or len(phi_all)
    phi_all, rho_all = phi_all[:n_series], rho_all[:n_series]
    if v2:
        h_all = z["h"].astype(np.float64)[:n_series]
        per_all = z["periods"][:n_series]

    def cov_fn(i):
        if v2:
            return series_covariances_v2(phi_all[i], rho_all[i], h_all[i], per_all[i])
        return series_covariances(phi_all[i], rho_all[i])

    mech_set = g69.EVAL_MECHS_V2 if v2 else g69.EVAL_MECHS
    mechs = ("mcar",) if args.smoke else mech_set
    rates = (0.3,) if args.smoke else g69.EVAL_RATES
    q_grid = (0.0, 0.5, 1.0) if args.smoke else g69.Q_GRID
    print(f"E3 yardstick [{args.round}]: {n_series} series x {len(mechs)}x"
          f"{len(rates)}x{len(q_grid)}^2 cells", flush=True)

    res = {"meta": {"round": args.round,
                    "corpus": os.path.basename(corpus.rstrip("/")),
                    "n_series": n_series, "L": L, "H": H, "q_grid": list(q_grid),
                    "mechs": list(mechs), "rates": list(rates),
                    "code_hash": g69.code_hash(), "numpy": np.__version__,
                    "smoke": bool(args.smoke)},
           "cells": {}, "clean_floor": {}}

    # clean floor: every position of both channels observed noiselessly
    clean = np.empty(n_series)
    for i in range(n_series):
        S_obs, S_fo, S_ff = cov_fn(i)
        _, clean[i] = condition(S_obs, S_fo, S_ff, np.ones(2 * L), np.zeros(2 * L))
    res["clean_floor"] = {"mse_per_series": [round(float(v), 6) for v in clean],
                          "mse_mean": float(clean.mean()),
                          "mse_median": float(np.median(clean))}
    print(f"clean-floor MSE*: mean={clean.mean():.4f} median={np.median(clean):.4f}",
          flush=True)

    wR_q0_max = 0.0                                  # G0 part 1 accumulator
    g0_cells = []                                    # G0 part 2 accumulator
    g0b_info = {"tailblock|0.7": [], "block|0.7": []}      # G0b information-loss ratios
    g0b_naive = {"tailblock|0.7": [], "block|0.7": []}     # G0b fill-poisoning ratios
    for i in range(n_series):
        S_obs, S_fo, S_ff = cov_fn(i)
        # clean-assuming coefficient matrix: shared by every cell's naive-trust line
        A_clean = np.linalg.solve(S_obs.T, S_fo.T).T
        for mech in mechs:
            for rate in rates:
                mask_t = g69.eval_target_mask(i, mech, rate)
                for q_r in q_grid:
                    for q_c in q_grid:
                        blend, nv = cell_obs_vectors(mask_t, q_r, q_c)
                        A, mse = condition(S_obs, S_fo, S_ff, blend, nv)
                        mse_nv = naive_trust_mse(S_obs, S_fo, S_ff, A_clean, blend, nv)
                        key = f"{mech}|{rate}|{q_r}|{q_c}"
                        cell = res["cells"].setdefault(
                            key, {"mech": mech, "rate": rate, "q_r": q_r, "q_c": q_c,
                                  "wR": [], "wC": [], "mse": [], "mse_naive": []})
                        cell["wR"].append(round(float(A[:, :L][:, mask_t].mean()), 8))
                        cell["wC"].append(round(float(A[:, L:].mean()), 8))
                        cell["mse"].append(round(mse, 6))
                        cell["mse_naive"].append(round(mse_nv, 6))
        # G0 reference cells, independent of the (possibly smoke-reduced) grid:
        # (a) w*_R at q_r=0 must be exactly zero (any mask / q_c)
        mask0 = g69.eval_target_mask(i, "mcar", 0.3)
        blend, nv = cell_obs_vectors(mask0, 0.0, 0.5)
        A0, _ = condition(S_obs, S_fo, S_ff, blend, nv)
        wR_q0_max = max(wR_q0_max, float(np.abs(A0[:, :L][:, mask0]).max()))
        # (b) MSE*(q_r=1, q_c=1, block, 0.7) must equal the clean floor -- built through
        # the same cell-construction expressions as the grid (regression protection)
        mask1 = g69.eval_target_mask(i, "block", 0.7)
        blend, nv = cell_obs_vectors(mask1, 1.0, 1.0)
        _, mse1 = condition(S_obs, S_fo, S_ff, blend, nv)
        g0_cells.append(mse1)
        # G0b headroom cells (v2, amended 2026-09-03): information-loss gap at
        # tailblock 0.7 AND fill-poisoning (naive-trust) gap at block 0.7; both
        # metrics computed for both mechanisms and stored separately.
        if v2:
            for gm in ("tailblock", "block"):
                gm_mask = g69.eval_target_mask(i, gm, 0.7)
                blend, nv = cell_obs_vectors(gm_mask, 0.0, 0.0)
                _, mse0 = condition(S_obs, S_fo, S_ff, blend, nv)
                g0b_info[f"{gm}|0.7"].append(mse0 / clean[i])
                g0b_naive[f"{gm}|0.7"].append(
                    naive_trust_mse(S_obs, S_fo, S_ff, A_clean, blend, nv) / clean[i])
        if (i + 1) % 32 == 0 or i + 1 == n_series:
            print(f"  {i+1}/{n_series} series ({(time.time()-t0)/60:.1f}m)", flush=True)

    for cell in res["cells"].values():
        for k in ("wR", "wC", "mse", "mse_naive"):
            v = np.asarray(cell[k])
            cell[k + "_mean"] = float(v.mean())
            cell[k + "_median"] = float(np.median(v))

    # ---- G0
    g0_cells = np.asarray(g0_cells)
    g0_rel = float(np.max(np.abs(g0_cells - clean) / clean))   # per-series worst case
    res["gates"] = {"G0": {
        "wR_qr0_max_abs": float(wR_q0_max),
        "mse_star_q1_block07_mean": float(g0_cells.mean()),
        "clean_floor_mean": float(clean.mean()),
        "max_rel_diff_per_series": g0_rel,
        "pass": bool(wR_q0_max < 1e-9 and g0_rel < 1e-6)}}
    print(f"GATE G0: max|w*_R(q_r=0)|={wR_q0_max:.2e}  "
          f"MSE*(1,1,block,0.7) vs clean floor max rel diff={g0_rel:.2e} "
          f"{'PASS' if res['gates']['G0']['pass'] else 'FAIL'}", flush=True)

    # ---- G0b (v2 only, AMENDED 2026-09-03 -- see s69_notes.md): the headroom gate.
    #   (a) information-loss gap:  median MSE*(0,0)/MSE*(1,1) >= 2.0 at tailblock 0.7
    #   (b) fill-poisoning gap:    median MSE_naive(0,0)/MSE*(1,1) >= 1.5 at block 0.7
    # (the original wording demanded >= 1.5 information-loss at block 0.7, which is
    # structurally unreachable: random noiseless subsampling de-aliases stationary
    # Gaussian structure. Both metrics are reported for both mechanisms regardless.)
    if v2:
        med_tail = float(np.median(g0b_info["tailblock|0.7"]))
        med_block_nv = float(np.median(g0b_naive["block|0.7"]))
        res["gates"]["G0b"] = {
            "amended": True,
            "info_loss_tailblock07_median": med_tail,
            "info_loss_block07_median": float(np.median(g0b_info["block|0.7"])),
            "naive_trust_tailblock07_median":
                float(np.median(g0b_naive["tailblock|0.7"])),
            "naive_trust_block07_median": med_block_nv,
            "info_loss_tailblock07_q10":
                float(np.quantile(g0b_info["tailblock|0.7"], 0.1)),
            "naive_trust_block07_q10":
                float(np.quantile(g0b_naive["block|0.7"], 0.1)),
            "thresholds": {"info_loss_tailblock|0.7": G0B_TAIL,
                           "naive_trust_block|0.7": G0B_BLOCK},
            "pass": bool(med_tail >= G0B_TAIL and med_block_nv >= G0B_BLOCK)}
        print(f"GATE G0b (amended): info-loss tailblock0.7 median={med_tail:.2f} "
              f"(>= {G0B_TAIL})   naive-trust block0.7 median={med_block_nv:.2f} "
              f"(>= {G0B_BLOCK})   "
              f"{'PASS' if res['gates']['G0b']['pass'] else 'FAIL'}", flush=True)

    # ---- G2
    n_sim = 8000 if args.smoke else 30000
    n_g2 = 2 if args.smoke else 4
    if v2:
        g2_pairs = [{"phi": phi_all[i], "rho": rho_all[i], "h": h_all[i],
                     "periods": per_all[i]} for i in range(n_g2)]
        g2_cells = ([("mcar", 0.3, 0.5, 0.5), ("mcar", 0.3, 0.0, 1.0)] if args.smoke
                    else [("block", 0.7, 0.5, 0.5), ("mcar", 0.3, 0.25, 0.75),
                          ("tailblock", 0.7, 0.0, 1.0)])
    else:
        g2_pairs = [{"phi": phi_all[i], "rho": rho_all[i]} for i in range(n_g2)]
        g2_cells = ([("mcar", 0.3, 0.5, 0.5), ("mcar", 0.3, 0.0, 1.0)] if args.smoke
                    else [("block", 0.7, 0.5, 0.5), ("mcar", 0.3, 0.25, 0.75),
                          ("mcar", 0.7, 0.0, 1.0)])
    res["gates"]["G2"] = gate_g2(g2_pairs, g2_cells, n_sim, args.smoke, v2)
    print(f"GATE G2: max abs err {res['gates']['G2']['max_abs_err']:.4f} "
          f"(tol {G2_TOL}) {'PASS' if res['gates']['G2']['pass'] else 'FAIL'}",
          flush=True)

    res["meta"]["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(out_path, "w"))
    print("wrote", out_path, flush=True)
    gates = [res["gates"]["G0"]["pass"], res["gates"]["G2"]["pass"]]
    if v2:
        gates.append(res["gates"]["G0b"]["pass"])
    if not all(gates):
        sys.exit("GATES FAILED")


if __name__ == "__main__":
    main()
