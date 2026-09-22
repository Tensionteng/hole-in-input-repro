#!/usr/bin/env python
"""Analysis for S5-real (run_s5_real.py): fill ranking, oscale recovery,
stress flatness, MNAR top-decile signature. Writes s5_real.png and prints
the tables that go into s5_real_notes.md.

Conventions mirror analyze_s5.py: errors are compared per channel first
(raw MSE scales differ wildly across clinical variables), then averaged
across channels. relMSE(fill) = mean over channels of mse(fill)/mse(best
fill for that channel); the stress test is normalized by the natural-gap
linear baseline instead.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "s5_real_results.json")
PNG = os.path.join(HERE, "s5_real.png")

FILLS = ["linear", "ffill", "zero", "zero_oscale"]


def load():
    with open(RESULTS) as f:
        return json.load(f)


def natural_records(data, model):
    """fill -> record for the natural-gap configs of one model."""
    return {r["fill"]: r for r in data["records"]
            if r["model"] == model and r["extra_mcar"] == 0.0}


def channel_rel(data, model, ref="best"):
    """relMSE/relMAE per fill vs per-channel best (or vs linear if ref=
    'linear'), averaged over channels. Returns {fill: (relMSE, relMAE)}."""
    recs = natural_records(data, model)
    chans = data["meta"]["channels"]
    mse = {f: np.array([recs[f]["mse_per_channel"][c] for c in chans])
           for f in recs}
    base = (np.minimum.reduce(list(mse.values())) if ref == "best"
            else mse[ref])
    out = {}
    for f in FILLS:
        if f not in mse:
            continue
        out[f] = float(np.mean(mse[f] / base))
    return out


def oscale_recovery(data, model):
    """Per channel: (mse_zero - mse_oscale)/(mse_zero - mse_linear).
    >0: oscale helps; 1: full recovery to linear; <0: oscale is worse
    than plain zero-fill. Returns (per-channel dict, mean)."""
    recs = natural_records(data, model)
    chans = data["meta"]["channels"]
    out = {}
    for c in chans:
        z, o, l = (recs[f]["mse_per_channel"][c]
                   for f in ("zero", "zero_oscale", "linear"))
        out[c] = float((z - o) / (z - l)) if abs(z - l) > 1e-12 else float("nan")
    return out, float(np.nanmean(list(out.values())))


def stress_rel(data, model):
    """relMSE of extra-MCAR linear configs vs the natural-gap linear
    baseline, per channel then summarized over channels and mask seeds.
    Returns {rate: dict(mean, median, max_excl_static, per_channel)} --
    the mean is dominated by Weight (static admission value, near-zero
    baseline error -> degenerate ratio), so the median is the headline."""
    recs = natural_records(data, model)
    chans = data["meta"]["channels"]
    base = np.array([recs["linear"]["mse_per_channel"][c] for c in chans])
    out = {0.0: dict(mean=1.0, median=1.0, max_excl_static=1.0,
                     per_channel={c: 1.0 for c in chans})}
    for rate in (0.1, 0.3):
        rs = [r for r in data["records"] if r["model"] == model
              and r["fill"] == "linear" and r["extra_mcar"] == rate]
        rel = np.stack([np.array([r["mse_per_channel"][c] for c in chans])
                        / base for r in rs]).mean(axis=0)
        exw = [v for c, v in zip(chans, rel) if c != "Weight"]
        out[rate] = dict(mean=float(rel.mean()), median=float(np.median(rel)),
                         max_excl_static=float(max(exw)),
                         per_channel={c: float(v) for c, v in zip(chans, rel)})
    return out


def main():
    data = load()
    models = sorted({r["model"] for r in data["records"]})
    chans = data["meta"]["channels"]
    win = data["meta"]["windows"]
    mr = [w["ctx_missing_rate"] for w in win]
    print(f"windows: {len(win)} over {len(chans)} channels; "
          f"ctx missing mean {np.mean(mr):.3f} "
          f"(min {np.min(mr):.3f}, med {np.median(mr):.3f}, max {np.max(mr):.3f})")

    summary = {}
    for m in models:
        rel_best = channel_rel(data, m, ref="best")
        rel_lin = channel_rel(data, m, ref="linear")
        rec_per_ch, rec_mean = oscale_recovery(data, m)
        stress = stress_rel(data, m)
        lin = natural_records(data, m)["linear"]
        td_ratio = lin["mse_topdecile"] / lin["mse_rest"]
        best_fill = min(rel_best, key=rel_best.get)
        summary[m] = dict(rel_best=rel_best, rel_lin=rel_lin,
                          oscale_recovery_per_channel=rec_per_ch,
                          oscale_recovery_mean=rec_mean, stress=stress,
                          topdecile_ratio=float(td_ratio), best_fill=best_fill)
        print(f"\n=== {m} ===")
        print("  relMSE vs best fill: " + ", ".join(
            f"{f} {rel_best[f]:.3f}" for f in FILLS if f in rel_best))
        print("  relMSE vs linear   : " + ", ".join(
            f"{f} {rel_lin[f]:.3f}" for f in FILLS if f in rel_lin))
        print(f"  best fill (mean relMSE): {best_fill}")
        print(f"  oscale recovery mean: {rec_mean:+.3f}  per-channel: " +
              ", ".join(f"{c} {v:+.2f}" for c, v in rec_per_ch.items()))
        print("  stress relMSE (extra MCAR, linear):")
        for k in sorted(stress):
            s = stress[k]
            print(f"    +{int(k*100):2d}%: mean {s['mean']:.3f}  "
                  f"median {s['median']:.3f}  "
                  f"max excl Weight {s['max_excl_static']:.3f}")
        print(f"  linear top-decile/rest MSE ratio: {td_ratio:.2f} "
              f"(topdecile {lin['mse_topdecile']:.4f} vs rest {lin['mse_rest']:.4f}, "
              f"frac {lin['topdecile_frac']:.3f})")
        # per-channel fill table + relMAE (from per-window arrays)
        chans_ = data["meta"]["channels"]
        cw = [w["channel"] for w in data["meta"]["windows"]]
        recs = natural_records(data, m)
        mae = {f: np.array([np.mean([v for v, cc in zip(recs[f]["mae_per_window"], cw) if cc == c])
                            for c in chans_]) for f in FILLS if f in recs}
        mae_base = np.minimum.reduce(list(mae.values()))
        print("  relMAE vs best fill: " + ", ".join(
            f"{f} {float(np.mean(mae[f] / mae_base)):.3f}" for f in mae))
        print("  per-channel relMSE vs linear:")
        for c in chans_:
            row = " ".join(f"{f}={recs[f]['mse_per_channel'][c] / recs['linear']['mse_per_channel'][c]:8.2f}"
                           for f in ("ffill", "zero", "zero_oscale"))
            print(f"    {c:10s} {row}")

    # ------------------------------------------------------------ figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    x = np.arange(len(FILLS))
    width = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        vals = [summary[m]["rel_best"].get(f, np.nan) for f in FILLS]
        axes[0].bar(x + (i - (len(models) - 1) / 2) * width, vals, width,
                    label=m)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["linear", "ffill", "zero", "zero+\noscale"])
    axes[0].axhline(1.0, color="gray", lw=0.8, ls="--")
    axes[0].set_ylabel("relMSE vs best fill (log)")
    axes[0].set_title("(a) Fill ranking on natural gaps")
    axes[0].legend()

    for m in models:
        s = summary[m]["stress"]
        xs = sorted(s)
        axes[1].plot([100 * k for k in xs], [s[k]["median"] for k in xs],
                     "o-", label=f"{m} (median)")
        axes[1].plot([100 * k for k in xs], [s[k]["mean"] for k in xs],
                     "o--", alpha=0.45, label=f"{m} (mean, static-ch. infl.)")
    axes[1].axhline(1.0, color="gray", lw=0.8, ls="--")
    axes[1].set_xlabel("extra MCAR rate on observed context points (%)")
    axes[1].set_ylabel("relMSE vs natural-gap linear")
    axes[1].set_title("(b) Stress: extra MCAR + linear fill")
    axes[1].legend(fontsize=8)

    labels = {"zero": "zero-fill\nvs linear", "zero_oscale": "zero+oscale\nvs linear"}
    x = np.arange(2)
    for i, m in enumerate(models):
        rl = summary[m]["rel_lin"]
        axes[2].bar(x + (i - (len(models) - 1) / 2) * width,
                    [rl["zero"], rl["zero_oscale"]], width, label=m)
        rec = summary[m]["oscale_recovery_mean"]
        axes[2].text(x[1] + (i - (len(models) - 1) / 2) * width,
                     rl["zero_oscale"], f"rec {rec:+.0%}",
                     ha="center", va="bottom", fontsize=8)
    axes[2].set_yscale("log")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(list(labels.values()))
    axes[2].axhline(1.0, color="gray", lw=0.8, ls="--")
    axes[2].set_ylabel("relMSE vs linear (log)")
    axes[2].set_title("(c) Zero-fill damage and oscale repair")
    axes[2].legend()

    fig.suptitle("S5-real: PhysioNet'12 natural missingness "
                 "(context 24h, horizon 12h, 990 windows, 11 channels)")
    fig.tight_layout()
    fig.savefig(PNG, dpi=150)
    print(f"\n[figure] {PNG}")

    with open(os.path.join(HERE, "s5_real_analysis.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
