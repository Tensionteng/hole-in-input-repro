#!/usr/bin/env python
"""Assemble the S25 two-floor decomposition table + figure from s25_results.json."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "s25_results.json")
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES = (0.3, 0.7)
DSS = ("ETTh1", "ETTm1", "weather")
CONVS = ("plain_fill", "fill_mask", "nan")


def main():
    res = json.load(open(RES))
    out = {}

    # ---------------------------------------------------------------- gate ----
    if "gate" in res:
        print("== anchor gate (bolt, ETTh1, H=96, 300 windows) ==")
        for k, v in res["gate"].items():
            print(f"  {k:24s} stored={v['stored']:10.5f} rerun={v['rerun']:10.5f} "
                  f"dev={v['rel_dev']:+.4%} {'PASS' if v['pass'] else 'FAIL'}")

    # --------------------------------------------------- part 0: reachability ----
    if "part0" in res:
        print("\n== Part 0: fill-reachable set (rank of J_M; permutation invariance) ==")
        print(f"{'convention':12s} {'rank med':>9s} {'rank max':>9s} {'|J|max':>11s} "
              f"{'perm/redraw':>12s}  (pooled over datasets x mechanisms)")
        agg = {}
        for c in CONVS:
            sel = [v for k, v in res["part0"]["cells"].items() if k.endswith(":" + c)]
            rk = [v["rank_median"] for v in sel]
            rt = [v["perm_invariance"]["ratio_perm_over_redraw"] for v in sel]
            am = [v["absmax_max"] for v in sel]
            agg[c] = {"rank_median": float(np.median(rk)),
                      "rank_max": int(max(v["rank_max"] for v in sel)),
                      "absmax_max": float(max(am)),
                      "perm_over_redraw_median": float(np.median(rt)),
                      "perm_over_redraw_max": float(np.max(rt))}
            print(f"{c:12s} {agg[c]['rank_median']:9.1f} {agg[c]['rank_max']:9d} "
                  f"{agg[c]['absmax_max']:11.3e} {agg[c]['perm_over_redraw_median']:12.4f}")
        out["part0_agg"] = agg

    # ------------------------------------------------------- part A: oracle ----
    if "partA" in res:
        print("\n== Part A: per-window gradient oracle fill (E[inf]; optimiser gate) ==")
        print(f"{'cell':30s} {'linear':>8s} {'oracle':>8s} {'oracle|x>=tau':>14s}")
        for ds in DSS:
            for m in MECHS:
                for r in RATES:
                    k = f"{ds}:{m}:{r}"
                    if k not in res["partA"]["cells"]:
                        continue
                    v = res["partA"]["cells"][k]
                    print(f"{k:30s} {v['rel_linear']:8.3f} {v['rel_oracle_fill']:8.4f} "
                          f"{v['rel_oracle_fill_constrained']:14.4f}")

    # ---------------------------------------- parts B+C: the decomposition ----
    if "partB" in res and "partC" in res:
        B, C = res["partB"], res["partC"]
        print("\n== Two-floor decomposition (dataset-avg own-clean relMSE) ==")
        hdr = (f"{'mech':14s}{'p':>5s}{'best fix':>10s}{'fill net':>10s}"
               f"{'fill+mask':>11s}{'direct':>9s}{'(1) info':>10s}{'(2) arch':>10s}{'(3) fill':>10s}")
        print(hdr)
        table = {}
        for m in MECHS:
            for r in RATES:
                bf, fn, fm, dn = [], [], [], []
                for ds in DSS:
                    fills = [B["base"].get(f"{ds}:{m}:{r}:{f}") for f in ("linear", "zero", "nan")]
                    fills = [x for x in fills if x is not None]
                    kb = f"{ds}:{m}:{r}:plain_fill"
                    km = f"{ds}:{m}:{r}:fill_mask"
                    kc = f"{ds}:{m}:{r}"
                    if not fills or kb not in B["cells"] or kc not in C["cells"]:
                        continue
                    bf.append(min(fills))
                    fn.append(B["cells"][kb]["rel"])
                    fm.append(B["cells"][km]["rel"] if km in B["cells"] else np.nan)
                    dn.append(C["cells"][kc]["rel"])
                if not bf:
                    continue
                bf, fn, fm, dn = map(lambda a: float(np.nanmean(a)), (bf, fn, fm, dn))
                best_through_f = min(fn, fm) if not np.isnan(fm) else fn
                t1 = dn - 1.0                      # information floor (excess)
                t2 = best_through_f - dn           # architectural excess over the floor
                t3 = bf - best_through_f           # fill/routing suboptimality
                table[f"{m}:{r}"] = {"best_fixed": bf, "fillnet_plain": fn,
                                     "fillnet_mask": fm, "direct": dn,
                                     "term1_information": t1, "term2_architecture": t2,
                                     "term3_fill_routing": t3, "total_excess": bf - 1.0}
                print(f"{m:14s}{r:5.1f}{bf:10.3f}{fn:10.3f}{fm:11.3f}{dn:9.3f}"
                      f"{t1:10.3f}{t2:10.3f}{t3:10.3f}")
        out["decomposition"] = table
        print("\nclean baselines (MSE): bolt " +
              ", ".join(f"{d}={B['base'][f'{d}:clean']:.3f}" for d in DSS if f"{d}:clean" in B["base"]))
        print("                     direct " +
              ", ".join(f"{d}={C['base'][f'{d}:clean']:.3f}" for d in DSS if f"{d}:clean" in C["base"]))

    with open(os.path.join(HERE, "s25_analysis.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    figure(res, out)


def figure(res, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))

    ax = axes[0]
    if "part0_agg" in out:
        a = out["part0_agg"]
        xs = np.arange(len(CONVS))
        ax.bar(xs - 0.2, [a[c]["rank_median"] for c in CONVS], 0.4, label="median rank of $J_M$")
        ax2 = ax.twinx()
        ax2.bar(xs + 0.2, [max(a[c]["perm_over_redraw_median"], 1e-6) for c in CONVS], 0.4,
                color="tab:orange", label="perm / redraw output change")
        ax2.set_yscale("log")
        ax2.set_ylabel("permutation sensitivity (log)")
        ax.set_xticks(xs)
        ax.set_xticklabels(["plain fill\n(no mask)", "fill + mask", "NaN"])
        ax.set_ylabel("numerical rank of $J_M$")
        ax.set_title("(a) Fill-reachable set by input convention")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper center")

    ax = axes[1]
    if "decomposition" in out:
        keys = [k for k in out["decomposition"]]
        xs = np.arange(len(keys))
        for i, (lab, fld) in enumerate([("best fixed fill", "best_fixed"),
                                        ("fill net (plain)", "fillnet_plain"),
                                        ("fill net (+mask)", "fillnet_mask"),
                                        ("direct predictor", "direct")]):
            ax.bar(xs + (i - 1.5) * 0.2, [out["decomposition"][k][fld] for k in keys],
                   0.2, label=lab)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xticks(xs)
        ax.set_xticklabels([k.replace(":", "\np=") for k in keys], fontsize=7)
        ax.set_ylabel("own-clean relMSE")
        ax.set_title("(b) What is achievable, by lever")
        ax.legend(fontsize=8)

    ax = axes[2]
    if "decomposition" in out:
        keys = list(out["decomposition"])
        xs = np.arange(len(keys))
        t1 = np.array([out["decomposition"][k]["term1_information"] for k in keys])
        t2 = np.array([out["decomposition"][k]["term2_architecture"] for k in keys])
        t3 = np.array([out["decomposition"][k]["term3_fill_routing"] for k in keys])
        ax.bar(xs, t1, 0.6, label="(1) information floor")
        ax.bar(xs, t2, 0.6, bottom=t1, label="(2) architectural floor")
        ax.bar(xs, t3, 0.6, bottom=t1 + t2, label="(3) fill / routing")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels([k.replace(":", "\np=") for k in keys], fontsize=7)
        ax.set_ylabel("excess relMSE over clean")
        ax.set_title("(c) Two-floor decomposition of the degradation")
        ax.legend(fontsize=8)

    fig.suptitle("S25: information floor vs architectural floor (chronos-bolt-base, frozen)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "s25.png"), dpi=150)
    print(f"\nfigure -> {os.path.join(HERE, 's25.png')}")


if __name__ == "__main__":
    main()
