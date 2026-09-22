#!/usr/bin/env python
"""S66: Table-4-format dose-response rows for the 2025 zero-shot models
(FlowState r1, TiRex 1.1, TimesFM 2.5) + the reproduction gates.

Gate A (required, <=1e-5): s66's bolt-base stock arm vs s32c_results.json's
bolt-base|random bin medians, all four fills -- validates the window pool, the
metric, and the routing in one comparison (same role as s64's stock gates).
Gate B (supplementary): s66's timesfm run vs s32c's timesfm|random rows --
validates the s30 TimesFM wrapper and the cached 2.5 weights end to end.

Rows: per model, per stratum -- n, median MASE under the linear fill (the
Table-4 value), all four fills, the declared-vs-linear delta (nan vs linear),
and deltas against the three stock rows already in the paper's artifacts
(bolt-base / chronos2 / moirai2 from s32c_results.json, same stratum, linear).
Overall (unstratified) medians are pooled per-window medians from the score
arrays; the c2/m2 stock pools come from s64_gift_scores.npz (s64 gated that
pool as identical), bolt-base stock from s66's own gated regeneration.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
BINS = ["0.00-0.01", "0.01-0.05", "0.05-0.15", "0.15-0.30", "0.30-1.01"]
PRETTY = {"0.00-0.01": "<1%", "0.01-0.05": "1-5%", "0.05-0.15": "5-15%",
          "0.15-0.30": "15-30%", "0.30-1.01": ">30%"}
FILLS = ("nan", "linear", "zero", "ffill")
NEW_MODELS = ("flowstate", "tirex", "timesfm")
STOCK_REFS = ("bolt-base", "chronos2", "moirai2")   # s32c_results.json arm names
GATE_TOL = 1e-5
OUT = os.path.join(HERE, "s66_gift_rows.json")


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def gate(mine_bins, ref_bins, mine_arm, ref_arm):
    devs = []
    for b in BINS:
        kref, km = f"{ref_arm}|random|{b}", f"{mine_arm}|random|{b}"
        if kref not in ref_bins or km not in mine_bins:
            continue
        assert ref_bins[kref]["n"] == mine_bins[km]["n"], \
            (kref, ref_bins[kref]["n"], mine_bins[km]["n"])
        for h, v in ref_bins[kref]["median_by_fill"].items():
            devs.append((rel(mine_bins[km]["median_by_fill"][h], v),
                         f"{ref_arm}|{b}|{h}"))
    devs.sort(reverse=True)
    return {"max_rel_dev": devs[0][0], "worst": devs[0][1],
            "n_comparisons": len(devs), "tol": GATE_TOL,
            "pass": devs[0][0] <= GATE_TOL}


def main():
    dose = json.load(open(os.path.join(HERE, "s66_gift_dose.json")))
    mine = dose["bins"]
    ref = json.load(open(os.path.join(EXP, "s32_gifteval", "s32c_results.json")))["bins"]
    scores = dict(np.load(os.path.join(HERE, "s66_gift_scores.npz")))

    gates = {"A_bolt_base_vs_s32c": gate(mine, ref, "bolt-base", "bolt-base")}
    if any(k.startswith("timesfm|") for k in mine):
        gates["B_timesfm_vs_s32c"] = gate(mine, ref, "timesfm", "timesfm")

    # overall (unstratified) pools for the stock references: s64's gated npz
    # for c2/m2 stock, s66's gated bolt-base arm for bolt stock
    s64 = dict(np.load(os.path.join(EXP, "s64_bench3", "s64_gift_scores.npz")))
    stock_overall = {"bolt-base": float(np.nanmedian(scores["bolt-base|linear"])),
                     "chronos2": float(np.nanmedian(s64["c2-stock|linear"])),
                     "moirai2": float(np.nanmedian(s64["m2-stock|linear"]))}

    models = {}
    for name in NEW_MODELS:
        if f"{name}|linear" not in scores:
            continue
        rows = []
        for b in BINS:
            cell = mine.get(f"{name}|random|{b}")
            if not cell:
                continue
            med = cell["median_by_fill"]
            deltas = {}
            for s in STOCK_REFS:
                rs = ref.get(f"{s}|random|{b}")
                if rs and rs["n"] == cell["n"]:
                    deltas[f"vs_{s}_stock_pct"] = \
                        100 * (med["linear"] - rs["median_by_fill"]["linear"]) \
                        / rs["median_by_fill"]["linear"]
            rows.append({"stratum": PRETTY[b], "bin": b, "n": cell["n"],
                         "linear": med["linear"], "median_by_fill": med,
                         "nan_vs_linear_pct":
                             100 * (med["nan"] - med["linear"]) / med["linear"],
                         "datasets": cell["datasets"], **deltas})
        overall = {h: float(np.nanmedian(scores[f"{name}|{h}"])) for h in FILLS}
        odelta = {f"vs_{s}_stock_pct":
                  100 * (overall["linear"] - stock_overall[s]) / stock_overall[s]
                  for s in STOCK_REFS}
        models[name] = {"overall_median_by_fill": overall,
                        "overall_nan_vs_linear_pct":
                            100 * (overall["nan"] - overall["linear"])
                            / overall["linear"],
                        "overall_deltas": odelta, "strata": rows}

    res = {"meta": {"table4_protocol": "s32c random-window, median MASE vs "
                                       "in-window naive-1, observed target points, "
                                       "n=7725 pool (seed 1, per_series 6, "
                                       "max_series 250)",
                    "table4_fill": "linear",
                    "nan_vs_linear": "100*(nan-linear)/linear: the model's declared "
                                     "NaN convention vs the Table-4 linear fill",
                    "deltas": "100*(model-stock)/stock, linear fill, same stratum; "
                              "stocks from s32c_results.json",
                    "overall": "unstratified pooled median over all 7,725 windows",
                    "model_inputs": {
                        "flowstate": "plain rows: our filled array as the series "
                                     "(uses fill content, s49 ratio 0.85); nan row: "
                                     "declared path, content zeroed + flag (ratio "
                                     "0.00) -- fill discarded",
                        "tirex": "plain rows: our filled array (ratio 0.96); nan "
                                 "row: declared path, content zeroed + flag (ratio "
                                 "0.00) -- fill discarded",
                        "timesfm": "plain rows: our filled array (ratio 0.86); nan "
                                   "row: TimesFM 2.5's own internal interpolation "
                                   "over the NaN positions -- our fill never seen"}},
           "pool": dose.get("pool", {}).get("random"),
           "gates": gates,
           "stock_overall_linear": stock_overall,
           "models": models}
    json.dump(res, open(OUT, "w"), indent=1)

    for k, g in gates.items():
        print(f"gate {k}: {'PASS' if g['pass'] else 'FAIL'} "
              f"max_rel_dev={g['max_rel_dev']:.3e} ({g['worst']}, "
              f"{g['n_comparisons']} comparisons)")
    for name, m in models.items():
        print(f"\n== {name} (linear fill, random protocol) ==")
        print(f"{'stratum':8s} {'n':>6s} {'linear':>8s} {'nan':>8s} "
              f"{'nan-lin':>8s} " +
                  "  ".join(f"d_{s}" for s in STOCK_REFS))
        for r in m["strata"]:
            print(f"{r['stratum']:8s} {r['n']:6d} {r['linear']:8.3f} "
                  f"{r['median_by_fill']['nan']:8.3f} "
                  f"{r['nan_vs_linear_pct']:+7.1f}% " +
                  "  ".join(f"{r.get(f'vs_{s}_stock_pct', float('nan')):+6.1f}%"
                            for s in STOCK_REFS))
        print(f"  overall linear={m['overall_median_by_fill']['linear']:.3f} "
              f"nan={m['overall_median_by_fill']['nan']:.3f}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
