#!/usr/bin/env python
"""S73 analysis: absolute-scale win rates and deltas, CPT retrofit vs vanilla.

Part A (bolt, DS9 synthetic grid): from s73_perwindow.npz (eval_s73.py; gate
verified bit-exact against the stored s57 relMSE cells). Per dataset x fill
(rows pooled over the 4 mechanisms):
  - win rate: share of rows with abs MSE(Q50) < abs MSE(stock)  [ties reported]
  - mean/median absolute delta (Q50 - stock), median ratio Q50/stock
  - Wilcoxon signed-rank p on log ratios (paired rows)
  - clean deltas per dataset (same windows, clean input)
  - the 2x2 contingency {clean better/worse} x {missing absolute better/worse}
    per dataset x fill (dataset level, median convention) -- the reviewer's
    worry cell is (clean worse, missing absolute worse)
  - per-window granularity variant (rows averaged within each of the 150
    windows before comparing) as a robustness readout
Row granularity note: the paper's cells aggregate series-rows (150 windows x C
channels); win rates here use the same rows as the stored relMSE arrays.

Part B (GIFT-Eval real missingness, chronos-2 / Moirai-2.0): the stored
s64_gift_scores.npz holds per-window ABSOLUTE MASE for stock vs WiSE-FT-0.5
(the paper's tab:leaderboard c2/m2 retrofit rows) x 4 fills on the exact s32c
random-window pool. The pool is deterministic (seed 1), so stratum membership
(the window's own context-NaN fraction) is recomputed CPU-only via
run_s32c_stratified.collect -- no re-eval. Validated by reproducing every
stored bin count and bin median in s64_gift_dose.json exactly.

Part C (c2/m2 on the DS9 grid): stored grids keep per-window RATIOS but only
median/mean clean -> per-window absolute errors are not reconstructable; we
report the stored per-dataset clean deltas and say so.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s32_gifteval"))

NPZ = os.path.join(HERE, "s73_perwindow.npz")
OUT = os.path.join(HERE, "s73_absolute.json")

DS9 = ("ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness")
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
FILLS = ("zero", "linear", "oracle")


def pair_stats(ev, ec, rows=None):
    """Paired absolute-error comparison: ev=vanilla, ec=cpt, same row order."""
    ev = np.asarray(ev, np.float64)
    ec = np.asarray(ec, np.float64)
    finite = np.isfinite(ev) & np.isfinite(ec)
    ev, ec = ev[finite], ec[finite]
    n = len(ev)
    tie = ec == ev
    win = ec < ev
    d = ec - ev
    out = {"n": int(n), "n_nan_dropped": int((~finite).sum()),
           "win_rate": float(win.sum() / max(n - tie.sum(), 1)),
           "n_win": int(win.sum()), "n_tie": int(tie.sum()),
           "mean_delta": float(d.mean()), "median_delta": float(np.median(d))}
    ok = (ev > 1e-12) & (ec > 1e-12)
    if ok.sum() > 20:
        lr = np.log(ec[ok]) - np.log(ev[ok])
        out["median_ratio"] = float(np.median(ec[ok] / ev[ok]))
        out["wilcoxon_p"] = float(wilcoxon(lr).pvalue)
    else:
        out["median_ratio"] = None
        out["wilcoxon_p"] = None
    return out


def part_a(z):
    res = {"clean": {}, "per_dataset_fill": {}, "per_dataset_mech_fill": {},
           "per_window_fill": {}, "overall_fill": {}, "contingency_2x2": {}}
    # ---- clean per dataset
    clean_ratio = {}
    for ds in DS9:
        ev = z[f"clean|vanilla|{ds}"]
        ec = z[f"clean|cpt|{ds}"]
        st = pair_stats(ev, ec)
        st["median_vanilla"] = float(np.median(ev))
        st["median_cpt"] = float(np.median(ec))
        st["ratio_of_medians"] = st["median_cpt"] / st["median_vanilla"]
        res["clean"][ds] = st
        clean_ratio[ds] = st["ratio_of_medians"]
    res["clean_median_ratio_over_ds"] = float(np.median(list(clean_ratio.values())))

    # ---- missingness cells
    for ds in DS9:
        rows = z[f"rows|{ds}"]
        for fill in FILLS:
            ev_all, ec_all = [], []
            pw = {}  # per-window aggregated
            for mech in MECHS:
                ev = z[f"miss|vanilla|{ds}|{mech}|{fill}"]
                ec = z[f"miss|cpt|{ds}|{mech}|{fill}"]
                res["per_dataset_mech_fill"][f"{ds}|{mech}|{fill}"] = pair_stats(ev, ec)
                ev_all.append(ev)
                ec_all.append(ec)
                # per-window aggregation (mean over the window's channels)
                wv = np.bincount(rows, weights=ev) / np.bincount(rows)
                wc = np.bincount(rows, weights=ec) / np.bincount(rows)
                pw[mech] = (wv, wc)
            ev_all = np.concatenate(ev_all)
            ec_all = np.concatenate(ec_all)
            st = pair_stats(ev_all, ec_all)
            res["per_dataset_fill"][f"{ds}|{fill}"] = st
            w_win = np.concatenate([pw[m][0] for m in MECHS])
            w_cpt = np.concatenate([pw[m][1] for m in MECHS])
            res["per_window_fill"][f"{ds}|{fill}"] = pair_stats(w_win, w_cpt)

    # ---- overall per fill (rows pooled over datasets) + ds-median summaries
    for fill in FILLS:
        ev = np.concatenate([z[f"miss|vanilla|{ds}|{m}|{fill}"]
                             for ds in DS9 for m in MECHS])
        ec = np.concatenate([z[f"miss|cpt|{ds}|{m}|{fill}"]
                             for ds in DS9 for m in MECHS])
        st = pair_stats(ev, ec)
        wrs = [res["per_dataset_fill"][f"{ds}|{fill}"]["win_rate"] for ds in DS9]
        st["ds_win_rates"] = {ds: w for ds, w in zip(DS9, wrs)}
        st["min_ds_win_rate"] = float(min(wrs))
        st["median_ds_win_rate"] = float(np.median(wrs))
        mr = [res["per_dataset_fill"][f"{ds}|{fill}"]["median_ratio"] for ds in DS9]
        st["median_ds_ratio"] = float(np.median([x for x in mr if x]))
        res["overall_fill"][fill] = st

    # ---- 2x2 contingency per dataset x fill (dataset level, median convention)
    for fill in FILLS:
        tab = {}
        for ds in DS9:
            clean_worse = res["clean"][ds]["median_cpt"] > res["clean"][ds]["median_vanilla"]
            miss_worse = res["per_dataset_fill"][f"{ds}|{fill}"]["median_delta"] > 0
            tab[ds] = ("clean_worse" if clean_worse else "clean_better") + "|" + \
                      ("miss_worse" if miss_worse else "miss_better")
        res["contingency_2x2"][fill] = tab
    # ---- tail focus: any (ds, fill) whose MEAN delta favors vanilla
    for ds in DS9:
        for fill in FILLS:
            ev = np.concatenate([z[f"miss|vanilla|{ds}|{m}|{fill}"] for m in MECHS]).astype(np.float64)
            ec = np.concatenate([z[f"miss|cpt|{ds}|{m}|{fill}"] for m in MECHS]).astype(np.float64)
            d = ec - ev
            if d.mean() > 0:
                dsrt = np.sort(d)[::-1]
                res.setdefault("tail_focus", {})[f"{ds}|{fill}"] = {
                    "mean_delta": float(d.mean()),
                    "frac_rows_cpt_worse": float((d > 0).mean()),
                    "top10_share_of_total_delta": float(dsrt[:10].sum() / d.sum()),
                    "top1pct_share_of_total_delta": float(dsrt[:max(len(dsrt)//100, 1)].sum() / d.sum()),
                    "delta_quantiles": {str(q): float(np.quantile(d, q))
                                        for q in (0.5, 0.9, 0.99, 0.999, 1.0)}}
    return res


def part_b():
    """GIFT per-stratum win rates from stored per-window MASE (c2/m2)."""
    import run_s32_gifteval as s32
    import run_s32c_stratified as s32c
    z = np.load(os.path.join(EXP, "s64_bench3", "s64_gift_scores.npz"))
    dose = json.load(open(os.path.join(EXP, "s64_bench3", "s64_gift_dose.json")))
    got = s32c.collect("random", 6, 250)
    ctx, tgt, ok, src = got
    nf = (~np.isfinite(ctx)).mean(axis=1)
    n_pool = len(ctx)

    # validation: bin counts + bin medians must reproduce s64_gift_dose.json
    val = {"n_pool": int(n_pool), "expected": 7725, "bins_checked": 0,
           "count_mismatch": [], "median_max_abs_dev": 0.0}
    BINS = s32c.BINS
    for key, cell in dose["bins"].items():
        name, _, rng = key.partition("|random|")
        lo, hi = [float(x) for x in rng.split("-")]
        sel = (nf >= lo) & (nf < hi)
        val["bins_checked"] += 1
        if int(sel.sum()) != cell["n"]:
            val["count_mismatch"].append({key: [int(sel.sum()), cell["n"]]})
        for h, med in cell["median_by_fill"].items():
            mine = float(np.nanmedian(z[f"{name}|{h}"][sel]))
            val["median_max_abs_dev"] = max(val["median_max_abs_dev"],
                                            abs(mine - med))
    out = {"validation": val, "bins": {}}
    pairs = {"c2": ("c2-stock", "c2-w50"), "m2": ("m2-stock", "m2-w50")}
    all_fills = ["nan", "linear", "zero", "ffill"]
    for fam, (stock, w50) in pairs.items():
        for lo, hi in BINS:
            sel = (nf >= lo) & (nf < hi)
            bkey = f"{lo:.2f}-{hi:.2f}"
            for fill in all_fills:
                sv = z[f"{stock}|{fill}"][sel]
                cv = z[f"{w50}|{fill}"][sel]
                st = pair_stats(sv, cv)
                st["n_bin"] = int(sel.sum())
                st["median_stock"] = float(np.nanmedian(sv))
                st["median_w50"] = float(np.nanmedian(cv))
                out["bins"][f"{fam}|{bkey}|{fill}"] = st
        # pooled across strata, linear fill (Table 4's fill)
        for fill in ("linear", "nan"):
            st = pair_stats(z[f"{stock}|{fill}"], z[f"{w50}|{fill}"])
            out["bins"][f"{fam}|all|{fill}"] = st
    out["meta"] = {"metric": dose["meta"]["metric"], "table4_fill": "linear",
                   "pair": "w50 (paper's +CPT row) vs stock", "n_pool": int(n_pool)}
    return out


def part_c():
    """c2/m2 DS9: what stored data CAN say (clean deltas) + why not more."""
    d = json.load(open(os.path.join(EXP, "s64_bench3", "s64_leaderboard_cells.json")))
    cl = d["clean_mse"]
    out = {}
    for fam in ("c2", "m2"):
        stock, w50 = f"{fam}-stock", f"{fam}-w50"
        r = {}
        for ds in DS9:
            if ds in cl.get(w50, {}) and ds in cl.get(stock, {}):
                r[ds] = cl[w50][ds]["median"] / cl[stock][ds]["median"]
        out[fam] = {"arms": [a for a in cl if a.startswith(fam)],
                    "clean_ratio_of_medians_w50_vs_stock": r,
                    "clean_median_ratio_over_ds": float(np.median(list(r.values()))) if r else None}
    out["limitation"] = ("s60/s61/s65/s64 grids store per-window relMSE ratios but only "
                         "median/mean clean MSE; per-window clean errors are stored nowhere, "
                         "so per-window absolute errors (ratio x own clean) are not "
                         "reconstructable for c2/m2 on DS9 without a re-eval. Bolt was "
                         "re-evaluated instead (part A); c2/m2 absolute-scale evidence "
                         "comes from the stored per-window GIFT MASE (part B).")
    return out


def main():
    z = np.load(NPZ)
    res = {"meta": {
        "experiment": "S73: absolute-scale CPT-vs-vanilla win rates and deltas",
        "bolt_arms": {"vanilla": "bolt-base-stock (P0 = stock chronos-bolt-base)",
                      "cpt": "Q50-base (P8 CPT x WiSE-FT0.5; the paper's tab:leaderboard bolt +CPT row)"},
        "grid": "DS9 x 4 mechs x rate 0.7 x {zero,linear,oracle}, 150 windows/cell, declared path",
        "row_granularity": "series-row (window x channel), the granularity of the stored relMSE arrays",
        "gate": "all 216 relMSE cell medians reproduce s57_bolt.json/s57_q50.json bit-exactly",
        "date": "2026-09-10"}}
    print("part A: bolt DS9 ...", flush=True)
    res["bolt_ds9"] = part_a(z)
    print("part B: GIFT strata ...", flush=True)
    res["gift"] = part_b()
    print("part C: c2/m2 DS9 stored ...", flush=True)
    res["c2_m2_ds9_stored"] = part_c()
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)

    # console summary
    print("\n== bolt DS9: overall win rates (Q50 vs stock, per-row) ==")
    for fill in FILLS:
        o = res["bolt_ds9"]["overall_fill"][fill]
        print(f"  {fill:7s} win={o['win_rate']:.3f} (ds med {o['median_ds_win_rate']:.3f}, "
              f"min {o['min_ds_win_rate']:.3f})  medratio={o['median_ratio']:.3f}")
    print("\n== clean ratio of medians per ds (Q50/stock) ==")
    for ds in DS9:
        c = res["bolt_ds9"]["clean"][ds]
        print(f"  {ds:12s} {c['ratio_of_medians']:.4f}  win={c['win_rate']:.3f}")
    print("\n== 2x2 contingency (worry cell = clean_worse|miss_worse) ==")
    for fill in FILLS:
        print(f"  {fill}:", res["bolt_ds9"]["contingency_2x2"][fill])
    print("\n== GIFT validation ==", res["gift"]["validation"])


if __name__ == "__main__":
    main()
