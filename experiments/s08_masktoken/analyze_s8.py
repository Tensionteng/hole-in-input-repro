#!/usr/bin/env python
"""S8 analysis: reads s8_results.json (run `run_s8_masktoken.py --merge`
first), prints the summary tables that go into s8_notes.md, renders s8.png.

Metric conventions (both stored by Exp4; Exp1 stores masses + am-fracs and the
s7-frac is recomputed here from the deterministic masks -- verified to
reproduce s7_attrib_results.json nan enrichments 0.032/0.030/0.493/0.252/
0.275/0.066 exactly):
  enrich_am : mass on corrupted patches / fraction of ATTENDABLE keys
  enrich_s7 : S7's published convention -- same mass, but the key set is the
              fully-OBSERVED patches + REG (the 0.03-0.50 "native mask band")
"""
import json
import os

import numpy as np

import run_s5_missing as s5

HERE = s5.HERE
RESULTS = os.path.join(HERE, "s8_results.json")
PNG = os.path.join(HERE, "s8.png")
DS = ("ETTh1", "ETTm1", "weather")
MECHS1 = ("mcar", "block", "mnar_high")
P1 = (0.3, 0.7)
MODES = ("correct", "zero", "random", "invert")
MECHS3 = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES3 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
METHODS3 = ("nan", "linear", "mponly", "tok_ETTh1", "tok_ETTm1", "tok_weather")
PATCH, NPATCH = 16, 32


def frac_corr_s7(ds, mech, p):
    """S7's frac_corr: #corrupted / (#fully-observed + REG), mean over series."""
    X, starts = s5.load_windows(s5.DATASETS[ds], 300, s5.SEED)
    C = X.shape[1]
    det = mech in ("mnar_high", "mnar_extreme")
    vals = []
    for wi, s in enumerate(starts):
        xc = X[s:s + s5.L].T.copy()
        for ms in ((0,) if det else (0, 1)):
            m = s5.make_mask(mech, p, wi, ms, C, x=xc)
            pm = m.reshape(C, NPATCH, PATCH)
            corr = pm.any(axis=2)
            vals.append(corr.sum(axis=1) / ((~corr).sum(axis=1) + 1.0))
    return float(np.concatenate(vals).mean())


# ------------------------------------------------------------- tables -------

def exp1_table(r):
    clean = {ds: r["clean"][ds]["mse"] for ds in DS}
    f7 = {(ds, m, p): frac_corr_s7(ds, m, p) for ds in DS for m in MECHS1 for p in P1}
    rows = {}
    for mech in MECHS1:
        for p in P1:
            for mode in MODES:
                rel, eam, e7, ent = [], [], [], []
                for ds in DS:
                    v = r["exp1"][ds][f"{mech}:{p}:{mode}"]
                    rel.append(v["mse"] / clean[ds])
                    eam.append(v.get("enrich_am", v["enrich"]))  # old shards: am only
                    e7.append(np.mean(v["mc"]) / f7[(ds, mech, p)])
                    ent.append(np.mean(v["ent"]))
                rows[(mech, p, mode)] = {"rel": float(np.mean(rel)),
                                         "rel_ds": rel,
                                         "enrich_am": float(np.mean(eam)),
                                         "enrich_s7": float(np.mean(e7)),
                                         "ent": float(np.mean(ent))}
    return rows


def exp3_tables(r):
    clean = {ds: r["clean"][ds]["mse"] for ds in DS}
    rel = {}  # (test_ds, mech, rate, method) -> relMSE
    for ds in DS:
        for mech in MECHS3:
            for rate in RATES3:
                for m in METHODS3:
                    v = r["exp3"][ds].get(f"{mech}:{rate}:{m}")
                    if v:
                        rel[(ds, mech, rate, m)] = v["mse"] / clean[ds]
    # method x rate, test-ds avg, per mech
    mr = {}
    for mech in MECHS3:
        for m in METHODS3:
            mr[(mech, m)] = [float(np.mean([rel[(ds, mech, p, m)] for ds in DS]))
                             for p in RATES3]
    # token train-ds x test-ds per mech per rate
    tt = {}
    for mech in MECHS3:
        for rate in RATES3:
            tt[(mech, rate)] = [[rel.get((tds, mech, rate, f"tok_{trs}"))
                                 for tds in DS] for trs in DS]
    return rel, mr, tt


def exp4_table(r):
    rows = {}
    for ds in DS:
        for mech in MECHS1:
            for p in P1:
                v = r["exp4"][ds][f"{mech}:{p}"]
                rows[(ds, mech, p)] = {
                    "enrich_tok_s7": v["enrich_tok_s7"],
                    "enrich_tok_am": v["enrich_tok_am"],
                    "enrich_corr_s7": v["enrich_s7"],
                    "enrich_corr_am": v["enrich_am"],
                    "ent": float(np.mean(v["ent"])),
                    "frac_tok_am": v["frac_tok_am"]}
    return rows


def print_all(r):
    print("=" * 78)
    print("ANCHOR check (dataset-avg relMSE, p=0.7)")
    anch = r.get("anchor", {})
    for m, f in [("mcar", "nan"), ("mcar", "linear"), ("block", "nan"),
                 ("block", "linear"), ("mnar_high", "nan"), ("mnar_high", "linear")]:
        vals = [anch[ds][f"{m}:{f}"]["rel"] for ds in DS if ds in anch]
        if len(vals) == 3:
            print(f"  {m:10s} {f:7s} avg={np.mean(vals):.3f} (S5 ref in notes)")

    print("\n" + "=" * 78)
    print("EXP1 -- mask-channel ablation (dataset-avg relMSE / enrichment)")
    rows = exp1_table(r)
    for mech in MECHS1:
        for p in P1:
            line = f"  {mech:10s} p={p}: "
            for mode in MODES:
                v = rows[(mech, p, mode)]
                line += (f"{mode}: rel={v['rel']:.3f} e_am={v['enrich_am']:.3f} "
                         f"e_s7={v['enrich_s7']:.3f} | ")
            print(line)

    print("\n" + "=" * 78)
    print("EXP3 -- generalization (test-ds-avg relMSE vs clean), method x rate")
    _, mr, tt = exp3_tables(r)
    hdr = "  " + f"{'mech':13s}{'method':13s}" + "".join(f"{p:7.1f}" for p in RATES3) + "   mean"
    print(hdr)
    for mech in MECHS3:
        for m in METHODS3:
            v = mr[(mech, m)]
            print(f"  {mech:13s}{m:13s}" + "".join(f"{x:7.3f}" for x in v)
                  + f" {np.mean(v):7.3f}")
        print()
    print("  token train-ds x test-ds (relMSE), per mech, p=0.5:")
    for mech in MECHS3:
        mat = tt[(mech, 0.5)]
        print(f"    {mech}: " + "; ".join(
            f"tok_{trs}->" + "/".join(f"{mat[DS.index(trs)][j]:.3f}" for j in range(3))
            for trs in DS))

    print("\n" + "=" * 78)
    print("EXP4 -- attention under the trained [MASK] token (matched ds token)")
    rows4 = exp4_table(r)
    for mech in MECHS1:
        for p in P1:
            ks = [(ds, mech, p) for ds in DS]
            et7 = np.mean([rows4[k]["enrich_tok_s7"] for k in ks])
            etam = np.mean([rows4[k]["enrich_tok_am"] for k in ks])
            ec7 = np.mean([rows4[k]["enrich_corr_s7"] for k in ks])
            ent = np.mean([rows4[k]["ent"] for k in ks])
            print(f"  {mech:10s} p={p}: tok e_s7={et7:.3f} e_am={etam:.3f} | "
                  f"corr e_s7={ec7:.3f} | ent={ent:.3f}")


# ------------------------------------------------------------- figure -------

def make_figure(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clean = {ds: r["clean"][ds]["mse"] for ds in DS}
    rows1 = exp1_table(r)
    _, mr, tt = exp3_tables(r)
    rows4 = exp4_table(r)

    fig, axes = plt.subplots(3, 4, figsize=(24, 14))
    colors = {"correct": "#2ca02c", "zero": "#d62728",
              "random": "#ff7f0e", "invert": "#9467bd"}

    # (a) Exp1 relMSE
    ax = axes[0, 0]
    xticks, xlabels = [], []
    for i, mech in enumerate(MECHS1):
        for j, p in enumerate(P1):
            x0 = (i * 2 + j) * 5
            xticks.append(x0 + 1.5)
            xlabels.append(f"{mech}\np={p}")
            for k, mode in enumerate(MODES):
                ax.bar(x0 + k, rows1[(mech, p, mode)]["rel"], width=0.9,
                       color=colors[mode], label=mode if i == 0 and j == 0 else None)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("relMSE vs clean (dataset-avg)")
    ax.set_title("(a) Exp1: mask-channel ablation -- MSE", fontsize=11)
    ax.legend(fontsize=8)

    # (b) Exp1 enrichment (bars: attendable-key convention; dots: S7 convention)
    ax = axes[0, 1]
    for i, mech in enumerate(MECHS1):
        for j, p in enumerate(P1):
            x0 = (i * 2 + j) * 5
            for k, mode in enumerate(MODES):
                v = rows1[(mech, p, mode)]
                ax.bar(x0 + k, v["enrich_am"], width=0.9, color=colors[mode])
                ax.plot(x0 + k, v["enrich_s7"], "k.", ms=7)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="uniform (blind)")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("corrupted-patch enrichment")
    ax.set_title("(b) Exp1: attention enrichment (bars: attendable-key;\n"
                 "dots: S7 convention) -- flat across modes", fontsize=11)
    ax.legend(fontsize=8)

    # (c) Exp4 token enrichment vs native band
    ax = axes[0, 2]
    ax.axhspan(0.03, 0.50, color="green", alpha=0.15,
               label="native-mask band 0.03-0.50")
    xt4, xl4 = [], []
    for i, mech in enumerate(MECHS1):
        for j, p in enumerate(P1):
            x0 = (i * 2 + j) * 4
            xt4.append(x0 + 1)
            xl4.append(f"{mech}\np={p}")
            for k, ds in enumerate(DS):
                v = rows4[(ds, mech, p)]
                ax.bar(x0 + k, v["enrich_tok_s7"], width=0.9,
                       color=plt.cm.tab10(k), label=ds if i == 0 and j == 0 else None)
                ax.plot(x0 + k, v["enrich_tok_am"], "k.", ms=6)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(xt4)
    ax.set_xticklabels(xl4, fontsize=8)
    ax.set_ylabel("token-position enrichment")
    ax.set_title("(c) Exp4: does the trained [MASK] token get ignored?\n"
                 "(bars: S7 convention; dots: attendable-key)", fontsize=11)
    ax.legend(fontsize=8)

    # (d) training curves (clipped loss only, linear scale; raw is 1e2-1e6 and
    # would squash the trend -- raw range noted in the notes)
    ax = axes[0, 3]
    for k, ds in enumerate(DS):
        losses = r["train"][ds]["losses"]
        ax.plot([x[0] for x in losses], [x[1] for x in losses],
                color=plt.cm.tab10(k), label=ds)
    ax.set_xlabel("step")
    ax.set_ylabel("clipped pinball loss")
    ax.set_title("(d) Exp2 training curves (clip=100;\nunclipped raw 1e2-1e6, see notes)", fontsize=11)
    ax.legend(fontsize=8)

    # (e-h) Exp3 method x rate heatmaps per test mechanism
    vmax = 4.0
    for j, mech in enumerate(MECHS3):
        ax = axes[1, j]
        mat = np.array([[mr[(mech, m)][i] for i in range(len(RATES3))]
                        for m in METHODS3])
        ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=1.0, vmax=vmax)
        ax.set_xticks(range(len(RATES3)))
        ax.set_xticklabels([f"{p}" for p in RATES3], fontsize=8)
        ax.set_yticks(range(len(METHODS3)))
        ax.set_yticklabels(METHODS3, fontsize=8)
        for a in range(mat.shape[0]):
            for b in range(mat.shape[1]):
                ax.text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center",
                        fontsize=6.5)
        ax.set_xlabel("missing rate p")
        ax.set_title(f"({chr(101 + j)}) Exp3 relMSE: test {mech} "
                     f"(test-ds avg)", fontsize=11)

    # (i-l) token train-ds x test-ds heatmaps at p=0.5
    for j, mech in enumerate(MECHS3):
        ax = axes[2, j]
        mat = np.array(tt[(mech, 0.5)], dtype=float)
        ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=1.0, vmax=vmax)
        ax.set_xticks(range(3))
        ax.set_xticklabels(DS, fontsize=8)
        ax.set_yticks(range(3))
        ax.set_yticklabels([f"tok_{d}" for d in DS], fontsize=8)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center",
                        fontsize=8)
        ax.set_title(f"({chr(105 + j)}) Exp3 token transfer: test {mech}, p=0.5",
                     fontsize=11)
        ax.set_xlabel("test dataset")

    fig.suptitle("S8 -- missing-signal causality & generalization (chronos-bolt-base)",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PNG, dpi=150)
    print(f"figure -> {PNG}")


def main():
    r = json.load(open(RESULTS))
    print_all(r)
    make_figure(r)


if __name__ == "__main__":
    main()
