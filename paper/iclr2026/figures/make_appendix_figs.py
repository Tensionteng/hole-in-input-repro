#!/usr/bin/env python
"""All appendix figures, as vector PDFs."""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "experiments"))
J = lambda *p: json.load(open(os.path.join(EXP, *p)))

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "figure.dpi": 200})
DS = ("ETTh1", "ETTm1", "weather")
MECHS = [("mcar", "scattered"), ("block", "block"),
         ("mnar_high", "censoring"), ("mnar_extreme", "extreme cens.")]
save = lambda fig, n: (fig.savefig(os.path.join(HERE, n), format="pdf", bbox_inches="tight"),
                       print("wrote", n))


# ---- A: scale ---------------------------------------------------------------
def fig_scale():
    S = J("s28_scale", "s28_results.json")["models"]
    order = ["tiny", "mini", "small", "base"]
    p = [S[k]["n_params"] / 1e6 for k in order]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.25))

    ax = axes[0]
    for (m, lab), col in zip(MECHS, ["#2c6fad", "#5aa0d0", "#c0392b", "#e08a5d"]):
        ax.plot(p, [S[k]["summary"][f"{m}|0.7"] for k in order], "-o", color=col,
                lw=1.6, ms=4, label=lab)
    ax.set_xscale("log"); ax.set_xlabel("parameters (M)")
    ax.set_ylabel("relMSE at rate 0.7"); ax.set_ylim(0.9, 4.0)
    ax.legend(frameon=False, ncol=2, handlelength=1.4, fontsize=6.8, loc="upper left")

    # GIFT-Eval: accuracy improves, fill-sensitivity grows
    B = J("s32_gifteval", "s32_partB.json")["cells"]
    names = ["bolt-tiny", "bolt-mini", "bolt-small", "bolt-base"]
    clean_ds = ["ett1/H", "ett2/H", "m4_hourly", "solar/H", "us_births/D"]
    acc, spread = [], []
    for n in names:
        v = [B[f"{n}|{d}|nan"]["mase_median"] for d in clean_ds if f"{n}|{d}|nan" in B]
        acc.append(np.mean(v) if v else np.nan)
        k = "kdd_cup_2018_with_missing/H"
        q = [B.get(f"{n}|{k}|{f}", {}).get("mase_median", np.nan)
             for f in ("nan", "linear", "zero", "ffill")]
        spread.append(100 * (np.nanmax(q) - np.nanmin(q)) / np.nanmin(q))
    ax = axes[1]
    ax.plot(p, np.array(acc) / acc[0], "-o", color="#2c6fad", lw=1.7, ms=4,
            label="clean-data error (relative to smallest)")
    ax2 = ax.twinx()
    ax2.plot(p, spread, "-s", color="#c0392b", lw=1.7, ms=4,
             label="sensitivity to the fill convention")
    ax.set_xscale("log"); ax.set_xlabel("parameters (M)")
    ax.set_ylabel("clean error, relative", color="#2c6fad")
    ax2.set_ylabel("fill-convention spread (%)", color="#c0392b", fontsize=7.4)
    ax.tick_params(axis="y", colors="#2c6fad"); ax2.tick_params(axis="y", colors="#c0392b")
    fig.tight_layout(pad=.4, w_pad=1.9); save(fig, "app_scale.pdf")


# ---- B: the alpha sweep, zero-shot vs adapted, all mechanisms ---------------
def fig_sweep_full():
    """Zero-shot row from S27 (no adapter is ever loaded there, so its denominator is sound);
    adapted row from S37, which resets the projection before computing any denominator and
    normalises each interface by its own clean forecast."""
    SW = J("s27_interface", "s27_results.json")["sweep"]
    S37 = {}
    for f in ("own_insample.json", "own_held_a.json", "own_held_b.json", "own_held_c.json"):
        p = os.path.join(EXP, "s37_capbreadth", f)
        if os.path.exists(p):
            S37.update(json.load(open(p))["sweep"])
    AL = (0.0, 0.25, 0.5, 0.75, 1.0)
    DS9 = ["ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
           "exchange", "illness"]

    def cur(tag, i, m, r):
        if tag == "zeroshot":
            return [np.mean([SW[f"zeroshot|{d}|{m}|{r}|{a}|{i}"]["median"] for d in DS])
                    for a in AL]
        return [np.median([S37[f"{d}|{m}|{r}|{a}|{i}"]["median_own"] for d in DS9
                           if f"{d}|{m}|{r}|{a}|{i}" in S37]) for a in AL]

    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.75), sharex=True)
    for row, (tag, rlab) in enumerate((("zeroshot", "zero-shot, 3 datasets"),
                                       ("adapted", "adapted, 9 datasets"))):
        for col, (m, lab) in enumerate(MECHS):
            ax = axes[row, col]
            for i, c, mk, nm in (("plain", "#8c8c8c", "o", "plain fill"),
                                 ("native", "#c0392b", "s", "declared"),
                                 ("dual", "#2c6fad", "^", "declared+content"),
                                 ("dual_obsnorm", "#7ba7d1", "v", "+obs. stats")):
                try:
                    y = cur(tag, i, m, 0.7)
                except KeyError:
                    continue
                ax.plot(AL, y, "-", marker=mk, color=c, lw=1.3, ms=3,
                        label=nm if (row == 0 and col == 0) else None)
            ax.axhline(1, color="k", ls=":", lw=.7)
            if row == 0:
                ax.set_title(lab)
            if row == 1:
                ax.set_xlabel(r"$\alpha$")
            if col == 0:
                ax.set_ylabel(rlab + "\nrelMSE")
    fig.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.055))
    fig.tight_layout(pad=.4); save(fig, "app_sweep.pdf")


# ---- C: bad-fill sensitivity by interface ------------------------------------
def fig_attack():
    A = J("s29_audit", "s29_results.json")["armB"]
    doms = [("penn|cens", "Penmanshiel (14% missing)"), ("metr|miss", "METR-LA (22% missing)")]
    ifs = [("plain", "plain fill"), ("native", "declared (rank 2)"), ("nan", "NaN (rank 0)")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55))
    for ax, (k, t) in zip(axes, doms):
        med = [A[k][f"{i}|untargeted"]["damage_ratio_median"] for i, _ in ifs]
        q90 = [A[k][f"{i}|untargeted"]["damage_ratio_q90"] for i, _ in ifs]
        x = np.arange(3)
        ax.bar(x - .18, med, .36, color="#c0392b", label="median")
        ax.bar(x + .18, q90, .36, color="#e8b4a8", label="90th pct")
        ax.set_yscale("log"); ax.axhline(1, color="k", lw=.7, ls=":")
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in ifs], fontsize=7)
        ax.set_title(t); ax.set_ylabel("error multiplier, adversarial fill")
    axes[0].legend(frameon=False)
    fig.tight_layout(pad=.4, w_pad=1.2); save(fig, "app_attack.pdf")


# ---- D: cross-channel -------------------------------------------------------
def fig_crosschannel():
    C = J("s31_crosschannel", "s31_results.json")["cells"]
    B = J("s31b_metrla_neighbours", "s31b_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55))
    ms = ["mcar", "block", "mnar_high"]
    labs = ["scattered", "block", "censoring"]
    conds = [("holed_uni", "filled, univariate", "#8c8c8c"),
             ("holed_multi", "filled + neighbours", "#e08a5d"),
             ("nan_multi", "declared + neighbours", "#2c6fad")]
    x = np.arange(3)
    for j, (key, lab, col) in enumerate(conds):
        y = [np.median([v[key] for k, v in C.items() if f"|{m}|0.7" in k]) for m in ms]
        axes[0].bar(x + (j - 1) * .27, y, .27, color=col, label=lab)
    axes[0].axhline(1, color="k", ls=":", lw=.7)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labs)
    axes[0].set_ylabel("relMSE (rate 0.7)"); axes[0].set_title("Synthetic masks")
    axes[0].legend(frameon=False, fontsize=6.8)
    keys = [("filled_uni", "filled\nuni"), ("filled_plus_nb", "filled\n+nb"),
            ("nan_plus_nb", "declared\n+nb")]
    axes[1].bar(np.arange(3) - .18, [B[k]["nmse_median"] for k, _ in keys], .36,
                color="#2c6fad", label="median")
    axes[1].bar(np.arange(3) + .18, [B[k]["nmse_mean"] for k, _ in keys], .36,
                color="#a8c4de", label="mean")
    axes[1].set_xticks(np.arange(3)); axes[1].set_xticklabels([l for _, l in keys], fontsize=7)
    axes[1].set_ylabel("NMSE"); axes[1].set_title("METR-LA real outages ($n{=}98$)")
    axes[1].legend(frameon=False)
    fig.tight_layout(pad=.4, w_pad=1.2); save(fig, "app_crosschannel.pdf")


# ---- E: real-data migration -------------------------------------------------
def fig_migration():
    P = J("s26_realfill", "s26_results.json")["arms"]["penn|cens|aug0|c-zero"]
    e = np.array(P["learned_plain_fill"]["per_window"])
    z = np.array(P["fixed_zero"]["per_window"])
    r = np.clip(e / np.maximum(z, 1e-12), .2, 5)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55))
    axes[0].hist(np.log10(r), bins=40, color="#2c6fad", alpha=.85)
    axes[0].axvline(0, color="k", ls=":", lw=.9)
    axes[0].set_xlabel(r"$\log_{10}$(learned fill / best fixed fill)")
    axes[0].set_ylabel("windows")
    axes[0].set_title(f"Penmanshiel, paired ($n{{=}}{len(r)}$)")
    axes[0].text(.03, .95, f"wins {100*(r<1).mean():.0f}%\n$p=1.7\\times10^{{-5}}$",
                 transform=axes[0].transAxes, va="top", fontsize=7.5)
    syn, real = 64.0, 100 * (1 - np.median(e / z))
    axes[1].bar([0, 1], [syn, real], .5, color=["#8c8c8c", "#c0392b"])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["synthetic\ncensoring", "real\ncurtailment"])
    axes[1].set_ylabel("% of the gap recovered")
    for i, v in enumerate([syn, real]):
        axes[1].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout(pad=.4, w_pad=1.2); save(fig, "app_migration.pdf")


# ---- F: patch size ----------------------------------------------------------
def fig_patch():
    R = J("s34_patchsize", "s34_results.json")
    C, CL = R["cells"], R["clean"]
    P = [8, 16, 32, 64, 128]
    cl = [np.mean([CL[f"{d}|{p}"]["mse_median"] for d in DS]) for p in P]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.55))
    for m, lab in MECHS:
        axes[0].plot(P, [np.mean([C[f"{d}|{m}|0.7|{p}"]["rel_median"] for d in DS]) for p in P],
                     "-o", lw=1.4, ms=3.5, label=lab)
    axes[0].set_xscale("log", base=2); axes[0].set_xlabel("patch size")
    axes[0].set_ylabel("relMSE at rate 0.7")
    for m, lab in MECHS:
        y = [cl[i] * np.mean([C[f"{d}|{m}|0.7|{p}"]["rel_median"] for d in DS])
             for i, p in enumerate(P)]
        axes[1].plot(P, y, "-o", lw=1.4, ms=3.5, label=lab)
    axes[1].plot(P, cl, "--k", lw=1.2, label="clean")
    axes[1].set_xscale("log", base=2); axes[1].set_xlabel("patch size")
    axes[1].set_ylabel("absolute MSE")
    h, lb = axes[1].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, ncol=5, fontsize=6.8, loc="lower center",
               bbox_to_anchor=(0.5, -0.015), columnspacing=1.4, handlelength=1.5)
    fig.tight_layout(pad=.4, w_pad=1.2, rect=(0, .085, 1, 1)); save(fig, "app_patch.pdf")


for f in (fig_scale, fig_sweep_full, fig_attack, fig_crosschannel, fig_migration, fig_patch):
    try:
        f()
    except Exception as e:
        print("FAIL", f.__name__, type(e).__name__, str(e)[:140])
