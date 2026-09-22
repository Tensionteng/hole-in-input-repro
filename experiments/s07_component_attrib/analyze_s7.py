#!/usr/bin/env python
"""S7 analysis: merge shard results -> s7_attrib_results.json, compute the
Layer-0 decomposition (stats vs values vs interaction), drift summaries and
attention sink summaries, render s7_attrib.png, and print the numbers that go
into s7_attrib_notes.md.

Definitions (dataset-avg relMSE unless stated):
  damage        = relA - 1                (total zero-fill damage)
  stats-only    = relC - 1                (clean values, polluted stats)
  values-only   = relB - 1                (zero values, oracle stats)
  interaction   = relA - relB - relC + 1  (residual super-additivity)
  shares        = each / damage
  B recovery    = (relA - relB) / damage  (S5 anchor: 82.3/78.7/82.3% mcar p=0.5)
"""
import json
import os

import numpy as np

import run_s5_missing as s5
import run_s7_attrib as s7

HERE = s5.HERE
RESULTS = os.path.join(HERE, "s7_attrib_results.json")
PNG = os.path.join(HERE, "s7_attrib.png")
DS = ("ETTh1", "ETTm1", "weather")
MECHS = s7.MECHS_ALL
P_L0 = s7.P_L0
P_L12 = s7.P_L12
CONDS = s7.CONDS


def load():
    r = json.load(open(RESULTS))
    return r


def rel(r, ds, mech, p, cell):
    clean = r["layer0"][ds]["clean"]["mse"]
    return r["layer0"][ds][f"{mech}:{p}"][cell]["mse"] / clean


def layer0_table(r):
    """Returns {(mech, p): {cell: dataset-avg relMSE, ...}} + per-ds."""
    out = {}
    for mech in MECHS:
        for p in P_L0:
            per_ds = {ds: {c: rel(r, ds, mech, p, c) for c in s7.CELLS} for ds in DS}
            avg = {c: float(np.mean([per_ds[ds][c] for ds in DS])) for c in s7.CELLS}
            out[(mech, p)] = {"per_ds": per_ds, "avg": avg}
    return out


def decomp(avg, stats_cell="C", vals_cell="B"):
    """2x2 factorial decomposition given the stats-only and values-only cells."""
    damage = avg["A"] - 1
    stats = avg[stats_cell] - 1
    vals = avg[vals_cell] - 1
    inter = avg["A"] - avg[vals_cell] - avg[stats_cell] + 1
    return {"damage": damage, "stats": stats, "values": vals, "interaction": inter,
            "stats_share": stats / damage, "values_share": vals / damage,
            "interaction_share": inter / damage,
            "B_recovery": (avg["A"] - avg["B"]) / damage}


def print_summary(r):
    t = layer0_table(r)
    print("=" * 78)
    print("LAYER 0 -- dataset-avg relMSE (A/B/C) and decomposition shares")
    print("=" * 78)
    for mech in MECHS:
        for p in P_L0:
            e = t[(mech, p)]
            d = decomp(e["avg"])                       # spec cells: B (oscale), C (hook)
            ds_ = decomp(e["avg"], stats_cell="Cr")    # scale-factor factorial (B, Cr)
            di = decomp(e["avg"], stats_cell="C", vals_cell="Bh")  # InstanceNorm factorial
            print(f"{mech:14s} p={p}: A={e['avg']['A']:7.3f} B={e['avg']['B']:7.3f} "
                  f"C={e['avg']['C']:7.3f} Bh={e['avg']['Bh']:8.3f} Cr={e['avg']['Cr']:7.3f}")
            print(f"{'':16s}    spec  [B,C]  : stats {d['stats_share'] * 100:6.1f}% "
                  f"values {d['values_share'] * 100:6.1f}% inter {d['interaction_share'] * 100:6.1f}% "
                  f"| B-recovery {d['B_recovery'] * 100:5.1f}%")
            print(f"{'':16s}    scale [B,Cr] : stats {ds_['stats_share'] * 100:6.1f}% "
                  f"values {ds_['values_share'] * 100:6.1f}% inter {ds_['interaction_share'] * 100:6.1f}%")
            print(f"{'':16s}    inorm [Bh,C] : stats {di['stats_share'] * 100:6.1f}% "
                  f"values {di['values_share'] * 100:6.1f}% inter {di['interaction_share'] * 100:6.1f}%")
        print()
    print("Anchors: A p=0.7 avg relMSE vs S5")
    for mech in MECHS:
        got = t[(mech, 0.7)]["avg"]["A"]
        ref = s7.ANCHOR_A_P07[mech]
        print(f"  {mech:14s} {got:8.3f} vs {ref:6.2f}  ({(got / ref - 1) * 100:+.1f}%)")
    print("Anchors: B recovery at mcar p=0.5 per dataset vs S5 (82.3/78.7/82.3%)")
    for ds in DS:
        e = t[("mcar", 0.5)]["per_ds"][ds]
        rec = (e["A"] - e["B"]) / (e["A"] - 1)
        print(f"  {ds:8s} rec={rec * 100:5.1f}%  (relA={e['A']:.3f} relB={e['B']:.3f})")
    # decomposition per dataset at p=0.5 mcar (the stats story)
    print("\nDecomposition per dataset, mcar:")
    for p in P_L0:
        for ds in DS:
            e = t[("mcar", p)]["per_ds"][ds]
            d = decomp(e)
            print(f"  p={p} {ds:8s} stats {d['stats_share'] * 100:6.1f}% "
                  f"values {d['values_share'] * 100:6.1f}% inter {d['interaction_share'] * 100:6.1f}%")

    print("\n" + "=" * 78)
    print("LAYER 1 -- drift (dataset-avg over ds where available)")
    print("=" * 78)
    for mech in s7.MECHS_L12:
        for p in P_L12:
            for cond in CONDS:
                cos0, cosN, cosNc, cosNo, l2N, fobs = [], [], [], [], [], []
                for ds in DS:
                    try:
                        rr = r["drift"][ds][mech][f"{mech}:{cond}:{p}"]
                    except KeyError:
                        continue
                    cos0.append(rr["cos_mean"][0])
                    cosN.append(rr["cos_mean"][-1])
                    cosNc.append(np.nan if rr["cos_corr"][-1] is None else rr["cos_corr"][-1])
                    cosNo.append(np.nan if rr["cos_obs"][-1] is None else rr["cos_obs"][-1])
                    l2N.append(rr["l2_mean"][-1])
                    fobs.append(rr.get("frac_series_with_obs", np.nan))
                if cos0:
                    with np.errstate(invalid="ignore"):
                        obs_str = (f"{np.nanmean(cosNo):.3f}" if np.isfinite(np.nanmean(cosNo))
                                   else "n/a")
                    print(f"{mech:10s} p={p} {cond:6s} cos: embed={np.mean(cos0):.3f} "
                          f"Llast={np.mean(cosN):.3f} (corr {np.nanmean(cosNc):.3f} / "
                          f"obs {obs_str}, series-with-obs {np.nanmean(fobs):.2f}) "
                          f"l2_Llast={np.mean(l2N):.3f}")

    print("\n" + "=" * 78)
    print("LAYER 2 -- attention (dataset-avg); enrich = mass_corrupt/frac_corrupt")
    print("=" * 78)
    for mech in s7.MECHS_L12:
        for p in P_L12:
            line = f"{mech:10s} p={p}: "
            for cond in CONDS:
                ents, mcs, enr, mreg = [], [], [], []
                for ds in DS:
                    try:
                        rr = r["attn"][ds][mech][f"{cond}:{p}"]
                    except KeyError:
                        continue
                    ents.append(np.mean(rr["ent"]))
                    mcs.append(np.mean(rr["mc"]))
                    enr.append(np.mean(rr["mc"]) / max(rr["frac_corr"], 1e-9))
                    mreg.append(np.mean(rr["mreg"]))
                if ents:
                    line += (f"{cond}: ent={np.mean(ents):.3f} mc={np.mean(mcs):.3f} "
                             f"enr={np.mean(enr):.3f} reg={np.mean(mreg):.3f} | ")
            print(line)
        # clean reference
        ents = [np.mean(r["attn"][ds]["clean"]["ent"]) for ds in DS if "clean" in r["attn"].get(ds, {})]
        mregs = [np.mean(r["attn"][ds]["clean"]["mreg"]) for ds in DS if "clean" in r["attn"].get(ds, {})]
        if ents:
            print(f"{mech:10s} clean : ent={np.mean(ents):.3f} reg={np.mean(mregs):.3f}")


def make_figure(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = layer0_table(r)
    fig, axes = plt.subplots(3, 4, figsize=(22, 13))

    # row 1: Layer-0 relMSE curves per mechanism
    for j, mech in enumerate(MECHS):
        ax = axes[0, j]
        styles = {"A": ("o-", "zero+polluted (plain zero-fill)"),
                  "B": ("s-", "zero+oracle scale (= S5 zero_oscale)"),
                  "C": ("^-", "clean+polluted iNorm (hook)"),
                  "Bh": ("d--", "zero+oracle iNorm (hook)"),
                  "Cr": ("x--", "clean+deflated scale")}
        for cell in s7.CELLS:
            mk, lab = styles[cell]
            ys = [t[(mech, p)]["avg"][cell] for p in P_L0]
            ax.plot(P_L0, ys, mk, label=f"{cell}: {lab}", ms=4)
        ax.axhline(1.0, color="k", ls=":", lw=1, label="D (clean)")
        ax.set_yscale("log")
        ax.set_title(f"Layer0 decomposition: {mech}")
        ax.set_xlabel("missing rate p")
        ax.set_ylabel("relMSE (dataset-avg, log)")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)

    # row 2 c1/c2: decomposition shares (grouped bars; scale factorial B/Cr)
    for ax, mech in ((axes[1, 0], "mcar"), (axes[1, 1], "mnar_high")):
        width = 0.25
        xs = np.arange(len(P_L0))
        for k, (part, color, lab) in enumerate((("stats_share", "#d62728", "stats (Cr)"),
                                                ("values_share", "#1f77b4", "values (B)"),
                                                ("interaction_share", "#7f7f7f", "interaction"))):
            vals = np.array([decomp(t[(mech, p)]["avg"], stats_cell="Cr")[part] for p in P_L0])
            ax.bar(xs + (k - 1) * width, vals, width, label=lab, color=color, alpha=0.85)
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.axhline(0.0, color="k", lw=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(p) for p in P_L0])
        ax.set_title(f"Damage share ({mech}, scale factorial B/Cr)")
        ax.set_xlabel("missing rate p")
        ax.set_ylabel("share of (relA - 1)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # row 2 c3: drift cos vs layer, mcar p=0.7
    ax = axes[1, 2]
    for cond, sty in (("zero", "o-"), ("linear", "s-"), ("nan", "^-")):
        curves = []
        for ds in DS:
            try:
                curves.append(r["drift"][ds]["mcar"][f"mcar:{cond}:0.7"]["cos_mean"])
            except KeyError:
                pass
        if curves:
            ax.plot(range(s7.NHID), np.mean(curves, axis=0), sty, label=cond, ms=4)
    # corrupted vs observed for zero (NaN-safe: empty patch sets -> None)
    cc, co = [], []
    for ds in DS:
        try:
            rr = r["drift"][ds]["mcar"]["mcar:zero:0.7"]
            cc.append([np.nan if v is None else v for v in rr["cos_corr"]])
            co.append([np.nan if v is None else v for v in rr["cos_obs"]])
        except KeyError:
            pass
    if cc:
        with np.errstate(invalid="ignore"):
            ax.plot(range(s7.NHID), np.nanmean(cc, axis=0), "o--", color="C0", alpha=0.6, ms=4,
                    label="zero, corrupted patches")
            ax.plot(range(s7.NHID), np.nanmean(co, axis=0), "o:", color="C0", alpha=0.6, ms=4,
                    label="zero, observed patches (mcar: ~none)")
    ax.set_title("Drift: cosine vs clean (mcar p=0.7, ds-avg)")
    ax.set_xlabel("encoder layer (0 = patch embedding)")
    ax.set_ylabel("cosine similarity to clean")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # row 2 c4: drift vs distance to nearest observed (zero-fill)
    ax = axes[1, 2 + 1]
    edges = s7.DIST_EDGES
    labels = [f"{edges[i]}-{edges[i + 1] - 1}" for i in range(s7.NBINS)]
    for mech, sty in (("mcar", "o-"), ("block", "s-")):
        for layer, lw in ((0, 2), (12, 2)):
            curves, cnts = [], []
            for ds in DS:
                try:
                    rr = r["drift"][ds][mech][f"{mech}:zero:0.7"]
                    curves.append(np.array(rr["dist_cos"][layer], dtype=float))
                    cnts.append(np.array(rr["dist_cnt"], dtype=float))
                except KeyError:
                    pass
            if curves:
                c = np.mean(curves, axis=0)
                n = np.sum(cnts, axis=0)
                c = np.where(n > 100, c, np.nan)  # hide near-empty bins
                ax.plot(range(s7.NBINS), c, sty, lw=lw,
                        label=f"{mech} {'embed' if layer == 0 else 'final'}", ms=4)
    ax.set_title("Drift vs distance to nearest observed (zero p=0.7)")
    ax.set_xlabel("patch distance bin (time steps)")
    ax.set_ylabel("cosine similarity to clean")
    ax.set_xticks(range(s7.NBINS))
    ax.set_xticklabels(labels, fontsize=7, rotation=30)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # row 3: attention audit, mcar p=0.7 (dataset-avg)
    def avg_curve(mech, cond, p, key):
        curves = []
        for ds in DS:
            try:
                curves.append(r["attn"][ds][mech][f"{cond}:{p}"][key])
            except KeyError:
                pass
        return np.mean(curves, axis=0) if curves else None

    def clean_curve(key):
        curves = [r["attn"][ds]["clean"][key] for ds in DS if "clean" in r["attn"].get(ds, {})]
        return np.mean(curves, axis=0) if curves else None

    ax = axes[2, 0]
    c = clean_curve("ent")
    if c is not None:
        ax.plot(range(s7.NLAYERS), c, "k-", lw=2, label="clean")
    for cond, sty in (("zero", "o-"), ("linear", "s-"), ("nan", "^-")):
        v = avg_curve("mcar", cond, 0.7, "ent")
        if v is not None:
            ax.plot(range(s7.NLAYERS), v, sty, label=cond, ms=4)
    ax.set_title("Attention entropy vs layer (mcar p=0.7)")
    ax.set_xlabel("encoder layer")
    ax.set_ylabel("mean head-avg entropy (nats)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    for mech, mk in (("mcar", "o-"), ("block", "s-"), ("mnar_high", "^-")):
        for cond, col in (("zero", "C0"), ("nan", "C2")):
            v = avg_curve(mech, cond, 0.7, "mc")
            fr = np.mean([r["attn"][ds][mech][f"{cond}:0.7"]["frac_corr"]
                          for ds in DS if f"{cond}:0.7" in r["attn"].get(ds, {}).get(mech, {})])
            if v is not None:
                ax.plot(range(s7.NLAYERS), v / max(fr, 1e-9), mk, color=col,
                        label=f"{mech} {cond}", ms=4)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_title("Corrupted-patch attention mass / uniform baseline (p=0.7)")
    ax.set_xlabel("encoder layer")
    ax.set_ylabel("enrichment ratio")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2, 2]
    c = clean_curve("mreg")
    if c is not None:
        ax.plot(range(s7.NLAYERS), c, "k-", lw=2, label="clean")
    for cond, sty in (("zero", "o-"), ("linear", "s-"), ("nan", "^-")):
        v = avg_curve("mcar", cond, 0.7, "mreg")
        if v is not None:
            ax.plot(range(s7.NLAYERS), v, sty, label=f"mcar {cond}", ms=4)
    ax.set_title("Mass on REG token vs layer (mcar p=0.7)")
    ax.set_xlabel("encoder layer")
    ax.set_ylabel("attention mass")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2, 3]
    c = clean_curve("mfirst")
    if c is not None:
        ax.plot(range(s7.NLAYERS), c, "k-", lw=2, label="clean")
    for cond, sty in (("zero", "o-"), ("linear", "s-"), ("nan", "^-")):
        v = avg_curve("mcar", cond, 0.7, "mfirst")
        if v is not None:
            ax.plot(range(s7.NLAYERS), v, sty, label=f"mcar {cond}", ms=4)
    ax.set_title("Mass on first patch vs layer (mcar p=0.7)")
    ax.set_xlabel("encoder layer")
    ax.set_ylabel("attention mass")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("S7 attribution: chronos-bolt-base under missing context", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PNG, dpi=150)
    print(f"figure -> {PNG}")


def main():
    r = load()
    print_summary(r)
    make_figure(r)


if __name__ == "__main__":
    main()
