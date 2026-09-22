#!/usr/bin/env python
"""S42: real imputers across the interface -- the end-to-end test of the alpha sweep.

The paper's imputation-ceiling result (S27/S37) sweeps alpha between linear interpolation
and the truth, which brackets every real imputer but never puts one in the loop. This round
replaces the alpha-blend with REAL imputers -- the S23-trained SAITS/BRITS (variants `all`
mechanism-matched and `mb` mainstream) -- and asks, on identical windows and masks:

    declared (native, content zeroed) vs restored (dual / dual_obsnorm, fill + flag),
    all with the S27-adapted input projections, own-clean normalised.

If the breakeven analysis of the paper is right, the restored interface should beat the
declared one under every learned fill on most cells, since learned imputers sit well above
alpha=0.25 on these datasets.

Windows, masks (mask_seed=0), mechanisms, rates and the clean baselines are S25/S27's, so
the linear-fill cells must reproduce S37's own_insample sweep (gate, +-5%).
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
sys.path.insert(0, os.path.join(EXP, "s27_interface"))
import run_s25_twofloor as s25
import run_s27_interface as s27

L, H = s25.L, s25.H
MECHS = s25.MECHS
RATES = (0.3, 0.7)
DATASETS = ("ETTh1", "ETTm1", "weather")
FILLS = ("linear", "saits_all", "saits_mb", "brits_all", "brits_mb", "oracle")
INTERFACES = ("native", "dual", "dual_obsnorm")
S23_CK = os.path.join(EXP, "s23_impute", "s23_ckpt")
S27_CK = os.path.join(EXP, "s27_interface", "s27_ckpt")
OUT = os.path.join(HERE, "s42_results.json")

SAITS_HP = dict(n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=256,
                dropout=0.1)
BRITS_HP = dict(rnn_hidden_size=256)


# ------------------------------------------------------------ imputers ----

def build_imputer(name, n_features, device):
    from pypots.imputation import SAITS, BRITS
    if name == "saits":
        return SAITS(n_steps=L, n_features=n_features, **SAITS_HP,
                     batch_size=128, epochs=1, patience=None, num_workers=0,
                     device=device, saving_path=None, verbose=False)
    if name == "brits":
        return BRITS(n_steps=L, n_features=n_features, **BRITS_HP,
                     batch_size=128, epochs=1, patience=None, num_workers=0,
                     device=device, saving_path=None, verbose=False)
    raise ValueError(name)


def load_weights(imputer, path, device):
    """See S23: pypots<=1.3 .load() lacks weights_only=False; files are our own."""
    loaded = torch.load(path, map_location=device, weights_only=False)
    sd = (loaded["model_state_dict"]
          if isinstance(loaded, dict) and "model_state_dict" in loaded
          else loaded.state_dict())
    cur = imputer.model.state_dict()
    cur.update(sd)
    imputer.model.load_state_dict(cur)
    imputer.model.eval()
    return imputer


def load_imputer(fill, dataset, n_features, device):
    """fill = '{model}_{variant}'; returns (imputer, mu, sd)."""
    model, variant = fill.split("_")
    tag = f"{model}_{variant}_{dataset}"
    meta = json.load(open(os.path.join(S23_CK, tag + "_meta.json")))
    ck = os.path.join(S23_CK, tag + ".pypots")
    imp = build_imputer(model, n_features, device)
    load_weights(imp, ck, device)
    return imp, np.array(meta["mu"], np.float32), np.array(meta["sd"], np.float32)


# ------------------------------------------------------------- batches ----

def multivariate_masked(X, starts, mech, rate):
    """Per window: clean [C,L], mask [C,L] (identical to build_eval_batch), future [C,H]."""
    C = X.shape[1]
    det = mech in ("mnar_high", "mnar_extreme")
    assert det or True  # mask_seed = 0 throughout, matching S27's sweep
    cl, mk, gt = [], [], []
    for wi, s in enumerate(starts):
        x = X[s:s + L].T.copy()
        y = X[s + L:s + L + H].T.copy()
        m = s25.make_mask(mech, rate, wi, 0, C, x=x)
        cl.append(x)
        mk.append(m)
        gt.append(y)
    return cl, mk, gt


def impute_windows(imputer, mu, sd, cl, mk):
    """Standardise -> NaN -> impute -> de-standardise. Returns [S*C, L] filled ctx."""
    S = len(cl)
    samples = np.empty((S, L, cl[0].shape[0]), np.float32)
    for i, (x, m) in enumerate(zip(cl, mk)):
        xw = ((x.T - mu) / sd).astype(np.float32)
        xw[m.T] = np.nan
        samples[i] = xw
    imp = imputer.predict({"X": samples})["imputation"]     # [S, L, C] standardized
    ctx = (imp * sd + mu).transpose(0, 2, 1)                # [S, C, L]
    flat = np.concatenate([ctx[i] for i in range(S)]).astype(np.float32)
    return flat


# ----------------------------------------------------------------- main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    dev = args.device
    n_win = 20 if args.smoke else 300

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {
        "L": L, "H": H, "rates": RATES, "datasets": DATASETS, "fills": FILLS,
        "interfaces": INTERFACES, "n_win": n_win, "mask_seed": 0,
        "imputers": "S23 checkpoints (saits/brits x all/mb), per-dataset",
        "projections": "S27 adapted input_patch_embedding per interface",
        "norm": "own-clean per interface (median of per-series ratios)"})

    bolt = s25.Bolt(dev)
    projs = {i: torch.load(os.path.join(S27_CK, f"proj_{i}.pt"),
                           map_location=dev, weights_only=True)
             for i in INTERFACES}
    base_sd = s27.save_proj(bolt)

    imputers = {}
    for ds in DATASETS:
        C = pd_read_nfeatures(ds)
        for fill in FILLS:
            if fill in ("linear", "oracle"):
                continue
            t0 = time.time()
            imputers[(fill, ds)] = load_imputer(fill, ds, C, dev)
            print(f"loaded {fill}/{ds} ({time.time()-t0:.0f}s)", flush=True)

    cells, imp_err = {}, {}
    for ds in DATASETS:
        X, st = s25.load_windows(s25.DATASETS[ds], n_win, s25.SEED, H)

        # own-clean baselines, one per interface
        _, cl_f, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
        obs_c = np.ones_like(cl_f)
        clean_err = {}
        for iface in INTERFACES:
            s27.load_proj(bolt, projs[iface])
            p = s27.median_iface_np(bolt, cl_f, obs_c, iface)
            clean_err[iface] = ((p - gt_c) ** 2).mean(1)
        s27.load_proj(bolt, base_sd)

        for mech in MECHS:
            for rate in RATES:
                cl, mk, gt = multivariate_masked(X, st, mech, rate)
                mask_flat = np.concatenate(mk)
                gt_flat = np.concatenate(gt).astype(np.float32)
                obs = (~mask_flat).astype(np.float32)

                # mask/layout consistency with S25's harness (assert once per cell)
                clean_b, lin_b, mask_b, gt_b = s25.build_eval_batch(
                    X, st, mech, rate, H, fill="linear")
                assert np.array_equal(mask_flat, mask_b), f"mask mismatch {ds} {mech}"
                assert np.allclose(np.concatenate(cl), clean_b), "clean mismatch"

                fills = {"linear": lin_b,
                         "oracle": np.concatenate(cl).astype(np.float32)}
                for fill in FILLS:
                    if fill in ("linear", "oracle"):
                        continue
                    imp, mu, sd = imputers[(fill, ds)]
                    t0 = time.time()
                    fills[fill] = impute_windows(imp, mu, sd, cl, mk)
                    print(f"  imputed {ds:8s} {mech:13s} {rate} {fill:10s} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                # imputation MSE on masked positions (standardized space), for the
                # alpha-axis readout; all imputers of one dataset share mu/sd
                _, mu0, sd0 = imputers[("saits_all", ds)]
                for fill in FILLS:
                    imp_err[f"{ds}|{mech}|{rate}|{fill}"] = fill_imputation_mse(
                        fills[fill], cl, mk, mu0, sd0)

                for fill in FILLS:
                    ctx = fills[fill]
                    errs = {}
                    for iface in INTERFACES:
                        s27.load_proj(bolt, projs[iface])
                        p = s27.median_iface_np(bolt, ctx, obs, iface)
                        errs[iface] = ((p - gt_flat) ** 2).mean(1)
                    s27.load_proj(bolt, base_sd)
                    for iface in INTERFACES:
                        e = errs[iface]
                        ce = clean_err[iface]
                        ok = ce > 1e-12
                        r = e[ok] / ce[ok]
                        key = f"{ds}|{mech}|{rate}|{fill}|{iface}"
                        cells[key] = {"median": float(np.median(r)),
                                      "mean": float(r.mean()),
                                      "q95": float(np.quantile(r, 0.95)),
                                      "n": int(ok.sum())}
                    # paired win shares of the restored paths vs the declared one
                    okn = clean_err["native"] > 1e-12
                    for iface in ("dual", "dual_obsnorm"):
                        w = float((errs[iface][okn] < errs["native"][okn]).mean())
                        cells[f"{ds}|{mech}|{rate}|{fill}|{iface}"][
                            "winshare_vs_native"] = w
                line = " ".join(
                    f"{f}:{cells[f'{ds}|{mech}|{rate}|{f}|dual']['median']:.3f}"
                    for f in ("linear", "saits_all", "brits_all"))
                print(f"  {ds:8s} {mech:13s} p={rate} dual {line}", flush=True)
                res["cells"] = cells
                res["imp_mse_std"] = imp_err
                save(res)

    # gate: linear-fill cells must reproduce S37's own_insample sweep within 5%
    if not args.smoke:
        s37 = json.load(open(os.path.join(EXP, "s37_capbreadth", "own_insample.json")))
        sw = s37["sweep"]
        gate, worst = {}, 0.0
        for ds in DATASETS:
            for mech in MECHS:
                for rate in RATES:
                    for iface in INTERFACES:
                        mine = cells[f"{ds}|{mech}|{rate}|linear|{iface}"]["median"]
                        ref = sw[f"{ds}|{mech}|{rate}|0.0|{iface}"]["median_own"]
                        d = abs(mine - ref) / ref
                        worst = max(worst, d)
                        gate[f"{ds}|{mech}|{rate}|{iface}"] = {
                            "mine": mine, "s37": ref, "rel_dev": d, "pass": d <= 0.05}
        res["gate_linear_vs_s37"] = gate
        print(f"\nGATE linear vs s37 own_insample: worst rel dev {worst:.4%}", flush=True)
        assert worst <= 0.05, "gate FAILED -- harness does not reproduce S37"

    res["cells"] = cells
    res["imp_mse_std"] = imp_err
    save(res)
    print("\nwrote", OUT)


def fill_imputation_mse(flat_fill, cl, mk, mu, sd):
    """Imputation MSE on masked positions in per-channel standardized space.
    flat_fill: [S*C, L] original scale (window-major); cl/mk: per-window [C, L]."""
    C = cl[0].shape[0]
    clean_flat = np.concatenate(cl).astype(np.float32)
    mask_flat = np.concatenate(mk)
    tot, cnt = 0.0, 0
    for i in range(len(clean_flat)):
        ch = i % C
        xs = (clean_flat[i] - mu[ch]) / sd[ch]
        xf = (flat_fill[i] - mu[ch]) / sd[ch]
        m = mask_flat[i]
        if m.any():
            tot += float(((xf - xs) ** 2)[m].mean())
            cnt += 1
    return tot / max(cnt, 1)


def pd_read_nfeatures(ds):
    import pandas as pd
    df = pd.read_csv(s25.DATASETS[ds], nrows=2)
    return df.drop(columns=["date"]).shape[1]


def save(res):
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, OUT)


if __name__ == "__main__":
    main()
