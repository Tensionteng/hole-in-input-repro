#!/usr/bin/env python
"""S64: Table-4-format dose-response report for the C2 and M2 retrofits + the
stock-reproduction gate against s32c_results.json.

Table 4 (tab_doseresponse.tex) reports the LINEAR fill under the random-window
protocol: per stratum, stock median MASE, retrofit median MASE, Delta = retrofit
- stock relative to stock. This script emits the same columns for the C2 and M2
families, keeps all four fills in the JSON, and gates the two stock rows against
s32c_results.json (chronos2|random, moirai2|random).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
BINS = ["0.00-0.01", "0.01-0.05", "0.05-0.15", "0.15-0.30", "0.30-1.01"]
PRETTY = {"0.00-0.01": "<1%", "0.01-0.05": "1-5%", "0.05-0.15": "5-15%",
          "0.15-0.30": "15-30%", "0.30-1.01": ">30%"}
OUT = os.path.join(HERE, "s64_gift_report.json")


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def main():
    mine = json.load(open(os.path.join(HERE, "s64_gift_dose.json")))["bins"]
    ref = json.load(open(os.path.join(EXP, "s32_gifteval", "s32c_results.json")))["bins"]

    # ------------------------------------------------------------- gate ----
    gates = {}
    for mine_arm, ref_arm in (("c2-stock", "chronos2"), ("m2-stock", "moirai2")):
        devs = []
        for b in BINS:
            kref = f"{ref_arm}|random|{b}"
            km = f"{mine_arm}|random|{b}"
            if kref not in ref or km not in mine:
                continue
            assert ref[kref]["n"] == mine[km]["n"], (kref, ref[kref]["n"], mine[km]["n"])
            for h, v in ref[kref]["median_by_fill"].items():
                devs.append((rel(mine[km]["median_by_fill"][h], v),
                             f"{ref_arm}|{b}|{h}"))
        devs.sort(reverse=True)
        gates[f"stock_vs_s32c|{mine_arm}=={ref_arm}"] = {
            "max_rel_dev": devs[0][0], "worst": devs[0][1],
            "n Comparisons".replace(" ", ""): len(devs),
            "pass": devs[0][0] <= 0.02}

    # ------------------------------------------------------------ table ----
    families = {}
    for fam, stock, w50 in (("chronos2", "c2-stock", "c2-w50"),
                            ("moirai2", "m2-stock", "m2-w50")):
        rows = []
        for b in BINS:
            ks = mine.get(f"{stock}|random|{b}")
            kw = mine.get(f"{w50}|random|{b}")
            if not ks or not kw:
                continue
            s = ks["median_by_fill"]["linear"]
            r = kw["median_by_fill"]["linear"]
            rows.append({"stratum": PRETTY[b], "bin": b, "n": ks["n"],
                         "stock_linear": s, "retrofit_linear": r,
                         "delta_pct": 100 * (r - s) / s,
                         "stock_by_fill": ks["median_by_fill"],
                         "retrofit_by_fill": kw["median_by_fill"]})
        families[fam] = rows

    res = {"meta": {"table4_protocol": "s32c random-window, linear fill, median MASE "
                                       "vs in-window naive-1, observed target points",
                    "delta": "100*(retrofit-stock)/stock, linear fill"},
           "gates": gates, "families": families}
    json.dump(res, open(OUT, "w"), indent=1)

    for fam, rows in families.items():
        print(f"\n== {fam} (linear fill, random protocol) ==")
        print(f"{'stratum':8s} {'n':>6s} {'stock':>8s} {'retrofit':>9s} {'Delta':>8s}")
        for r in rows:
            print(f"{r['stratum']:8s} {r['n']:6d} {r['stock_linear']:8.3f} "
                  f"{r['retrofit_linear']:9.3f} {r['delta_pct']:+7.1f}%")
    print("\ngates:")
    for k, g in gates.items():
        print(f"  {k}: {'PASS' if g['pass'] else 'FAIL'} max_rel_dev={g['max_rel_dev']:.4f}"
              f" ({g['worst']})")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
