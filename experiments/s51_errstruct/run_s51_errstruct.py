#!/usr/bin/env python
"""S51: what about an imputer's error decides whether the restored interface wants it?

Section 6.1 leaves a warning without a criterion: BRITS reaches an effective alpha of 0.8 by
imputation error yet makes the restored path WORSE than the declared one, so a one-dimensional
quality axis does not capture an imputer's error structure. This round proposes the missing
statistic and tests it out of sample.

The hypothesis is architectural. Bolt embeds non-overlapping patches of 16 steps, so the part
of an imputation residual that survives into the patch embedding as a level error is its
per-patch MEAN; residual that is zero-mean inside a patch partly averages out. Define the
patch-DC share of a residual r on the masked positions of one series,

    psi = sum_p n_p * mean_p(r)^2  /  sum_p sum_{t in p} r_t^2      in [0, 1],

with p ranging over the patches that contain at least one masked position. psi = 1 is a pure
per-patch offset -- an error the model reads as a genuine change of level -- and psi near
1/n_p is white jitter. The prediction: at matched imputation accuracy, high psi hurts the
restored interface and leaves the declared one alone, because only the restored one reads it.

Everything is reconstructed from S42's own windows, masks (mask_seed=0) and S23 checkpoints,
so the residuals are the ones that produced S42's forecasts.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s42_end2end"))
import run_s25_twofloor as s25
import run_s42_end2end as s42

L, H = s42.L, s42.H
PATCH = 16                      # bolt input_patch_size == input_patch_stride
DATASETS = s42.DATASETS
MECHS = s42.MECHS
RATES = s42.RATES
FILLS = [f for f in s42.FILLS if f not in ("oracle",)]


def patch_stats(r, m):
    """Residual structure on one series. r, m: [L]; m True at masked positions."""
    idx = np.flatnonzero(m)
    if idx.size < 4:
        return None
    rm = r[idx]
    tot = float((rm ** 2).sum())
    if tot <= 0:
        return None
    # patch-DC share: energy carried by the per-patch mean of the residual
    pid = idx // PATCH
    dc = 0.0
    for p in np.unique(pid):
        sel = rm[pid == p]
        dc += sel.size * float(sel.mean()) ** 2
    psi = dc / tot
    # lag-1 autocorrelation over consecutive masked positions (a norm-free second view)
    run = np.flatnonzero(np.diff(idx) == 1)
    if run.size >= 3:
        a, b = rm[run], rm[run + 1]
        sa, sb = a.std(), b.std()
        ac1 = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else 0.0
    else:
        ac1 = float("nan")
    return psi, ac1, float(np.sqrt((rm ** 2).mean()))


def main():
    dev = "cuda"
    n_win = 300
    out = {"meta": {"L": L, "H": H, "patch": PATCH, "n_win": n_win,
                    "datasets": DATASETS, "mechs": list(MECHS),
                    "rates": RATES, "fills": FILLS,
                    "note": "residuals reconstructed from S42 windows/masks and S23 imputers"}}
    cells = {}
    for ds in DATASETS:
        X, st = s25.load_windows(s25.DATASETS[ds], n_win, s25.SEED, H)
        C = X.shape[1]
        imps = {f: s42.load_imputer(f, ds, C, dev) for f in FILLS if f != "linear"}
        for mech in MECHS:
            for rate in RATES:
                cl, mk, _ = s42.multivariate_masked(X, st, mech, rate)
                clean_flat = np.concatenate(cl).astype(np.float32)
                mask_flat = np.concatenate(mk)
                _, lin_b, mask_b, _ = s25.build_eval_batch(
                    X, st, mech, rate, H, fill="linear")
                assert np.array_equal(mask_flat, mask_b), f"mask mismatch {ds} {mech} {rate}"
                # all imputers of a dataset share the standardisation, as in S42
                _, mu0, sd0 = imps["saits_all"]
                for fill in FILLS:
                    t0 = time.time()
                    if fill == "linear":
                        filled = lin_b
                    else:
                        imp, mu, sd = imps[fill]
                        filled = s42.impute_windows(imp, mu, sd, cl, mk)
                    psis, acs, rmses = [], [], []
                    for i in range(len(clean_flat)):
                        ch = i % C
                        r = (filled[i] - clean_flat[i]) / sd0[ch]
                        s = patch_stats(r, mask_flat[i])
                        if s is None:
                            continue
                        psis.append(s[0]); acs.append(s[1]); rmses.append(s[2])
                    k = f"{ds}|{mech}|{rate}|{fill}"
                    cells[k] = {"psi_median": float(np.median(psis)),
                                "psi_mean": float(np.mean(psis)),
                                "ac1_median": float(np.nanmedian(acs)),
                                "rmse_median": float(np.median(rmses)),
                                "n_series": len(psis)}
                    print(f"  {k:34s} psi={cells[k]['psi_median']:.3f} "
                          f"ac1={cells[k]['ac1_median']:+.3f} "
                          f"rmse={cells[k]['rmse_median']:.3f} ({time.time()-t0:.0f}s)",
                          flush=True)
    out["cells"] = cells
    json.dump(out, open(os.path.join(HERE, "s51_results.json"), "w"), indent=1)
    print("wrote s51_results.json")


if __name__ == "__main__":
    main()
