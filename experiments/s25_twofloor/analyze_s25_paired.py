#!/usr/bin/env python
"""S25 paired re-analysis: per-WINDOW relMSE ratios, not ratios of means.

The mean-based relMSE in partB/partC is a ratio of means, and the hand-check showed it is
dominated by a handful of extreme windows (ETTh1 mnar_high p=0.7: mean 1.390 vs a median
window ratio far below it; weather worse). For a decomposition whose whole point is "how
much of the damage is which term", a tail-dominated statistic is not defensible.

This script recomputes everything paired: for each window w,
    r_w = MSE(method, w) / MSE(clean-context, w)
and reports the median and the mean of r_w, plus a paired win-rate against the best fixed
fill. It reuses the trained checkpoints from partB/partC (no retraining).
"""
import json
import os

import numpy as np
import torch

import run_s25_twofloor as R

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "s25_ckpt")
FIXED = ("linear", "zero", "nan")
EPS = 1e-12


def per_window_mse(pred, gt):
    return ((pred.astype(np.float64) - gt.astype(np.float64)) ** 2).mean(1)


@torch.no_grad()
def main():
    bolt = R.Bolt("cuda")
    out = {}
    for ds, path in R.DATASETS.items():
        X, st = R.load_windows(path, 300, R.SEED, R.H)
        _, cl_f, _, gt_c = R.build_eval_batch(X, st, "clean", 0.0, R.H)
        mse_clean = per_window_mse(bolt.median_np(cl_f), gt_c)     # [S*C]
        ok = mse_clean > EPS                                        # guard degenerate rows
        print(f"{ds}: {len(mse_clean)} series, {int((~ok).sum())} degenerate dropped, "
              f"clean mean={mse_clean.mean():.4f} median={np.median(mse_clean):.4f}",
              flush=True)

        for mech in R.MECHS:
            for rate in R.RATES_EVAL:
                cell = {}
                for f in FIXED:
                    _, fl, _, gt = R.build_eval_batch(X, st, mech, rate, R.H, fill=f)
                    cell[f] = per_window_mse(bolt.median_np(fl), gt)
                for conv in ("plain_fill", "fill_mask"):
                    p = os.path.join(CK, f"fillnet_{mech}_{conv}.pt")
                    net = R.FillNet().to("cuda")
                    net.load_state_dict(torch.load(p))
                    net.eval()
                    cell[f"fillnet_{conv}"] = R.eval_fillnet(bolt, net, X, st, mech, rate, conv)
                netd = R.DirectNet().to("cuda")
                netd.load_state_dict(torch.load(os.path.join(CK, f"directnet_{mech}.pt")))
                netd.eval()
                netc = R.DirectNet().to("cuda")
                netc.load_state_dict(torch.load(os.path.join(CK, "directnet_clean.pt")))
                netc.eval()
                d_miss = R.eval_directnet(netd, X, st, mech, rate, "cuda")
                d_clean = R.eval_directnet(netc, X, st, "clean", 0.0, "cuda", clean=True)
                dok = ok & (d_clean > EPS)

                rec = {}
                for k, v in cell.items():
                    r = v[ok] / mse_clean[ok]
                    rec[k] = {"median": float(np.median(r)), "mean": float(r.mean()),
                              "q75": float(np.quantile(r, 0.75)),
                              "q95": float(np.quantile(r, 0.95))}
                rd = d_miss[dok] / d_clean[dok]
                rec["direct"] = {"median": float(np.median(rd)), "mean": float(rd.mean()),
                                 "q75": float(np.quantile(rd, 0.75)),
                                 "q95": float(np.quantile(rd, 0.95))}
                # paired win-rate: learned fill vs the best fixed fill, window by window
                bf_name = min(FIXED, key=lambda f: np.median(cell[f][ok] / mse_clean[ok]))
                best_fill = cell[bf_name]
                rec["_best_fixed_by_median"] = bf_name
                for conv in ("plain_fill", "fill_mask"):
                    rec[f"winrate_{conv}_vs_bestfixed"] = float(
                        (cell[f"fillnet_{conv}"][ok] < best_fill[ok]).mean())
                rec["n"] = int(ok.sum())
                out[f"{ds}:{mech}:{rate}"] = rec
                print(f"  {mech:13s} p={rate}  " + "  ".join(
                    f"{k}={rec[k]['median']:.3f}" for k in
                    ("linear", "nan", "fillnet_plain_fill", "fillnet_fill_mask", "direct")),
                    flush=True)

    json.dump(out, open(os.path.join(HERE, "s25_paired.json"), "w"), indent=1)
    table(out)


def table(out):
    print("\n== PAIRED two-floor decomposition (median over windows of per-window "
          "relMSE vs that window's own clean forecast) ==")
    print(f"{'mech':14s}{'p':>5s}{'bestfix':>9s}{'fillnet':>9s}{'f+mask':>9s}{'direct':>8s}"
          f"{'(1)info':>9s}{'(2)arch':>9s}{'(3)fill':>9s}{'win%':>7s}")
    agg = {}
    for mech in R.MECHS:
        for rate in R.RATES_EVAL:
            rows = [out[k] for k in out if k.endswith(f":{mech}:{rate}")]
            if not rows:
                continue
            g = lambda f: float(np.mean([r[f]["median"] for r in rows]))
            bf = float(np.mean([min(r[f]["median"] for f in FIXED) for r in rows]))
            fn, fm, dn = g("fillnet_plain_fill"), g("fillnet_fill_mask"), g("direct")
            through = min(fn, fm)
            win = float(np.mean([max(r["winrate_plain_fill_vs_bestfixed"],
                                     r["winrate_fill_mask_vs_bestfixed"]) for r in rows]))
            agg[f"{mech}:{rate}"] = {"best_fixed": bf, "fillnet": fn, "fill_mask": fm,
                                     "direct": dn, "term1_information": dn - 1.0,
                                     "term2_architecture": through - dn,
                                     "term3_fill": bf - through, "winrate": win}
            print(f"{mech:14s}{rate:5.1f}{bf:9.3f}{fn:9.3f}{fm:9.3f}{dn:8.3f}"
                  f"{dn-1:9.3f}{through-dn:9.3f}{bf-through:9.3f}{win*100:6.0f}%")
    json.dump(agg, open(os.path.join(HERE, "s25_paired_agg.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
