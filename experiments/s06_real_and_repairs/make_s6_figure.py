#!/usr/bin/env python
"""Make s6_real_censor.png and write derived analysis into the results json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "dataset_censor", "penmanshiel2016_processed.npz"))
res = json.load(open(os.path.join(HERE, "s6_real_censor_results.json")))
windows = res["meta"]["windows"]
recs = res["records"]
MODELS = ("bolt", "timesfm", "moirai")
COLORS = {"bolt": "tab:blue", "timesfm": "tab:green", "moirai": "tab:purple"}

fig, axes = plt.subplots(1, 4, figsize=(21, 4.8))

# --- panel A: validation scatter, WT14 ------------------------------------
ax = axes[0]
tag = "Penmanshiel_14"
p, w, cens, curt = d[f"{tag}_power"], d[f"{tag}_wind"], d[f"{tag}_cens"], d[f"{tag}_curt"]
obs = np.isfinite(p) & np.isfinite(w)
rng = np.random.default_rng(0)
idx = rng.choice(np.flatnonzero(obs & ~cens), size=5000, replace=False)
ax.scatter(w[idx], p[idx], s=1, c="0.75", rasterized=True, label="observed")
ci = np.flatnonzero(cens)
ax.scatter(w[ci], p[ci], s=4, c="tab:red", rasterized=True,
           label="effectively censored (clamp binding)")
ax.plot(d[f"{tag}_curve_x"], d[f"{tag}_curve_y"], "k-", lw=1.2, label="q95 power curve")
ax.set_xlabel("wind speed (m/s)"); ax.set_ylabel("power (kW)")
ax.set_title("A. Penmanshiel WT14, 2016: curtailment cap lines\n(documented external output reduction)")
ax.legend(fontsize=7, markerscale=4, loc="lower right")
ax.set_xlim(0, 27); ax.set_ylim(-80, 2300)

# --- panel B: mean NMSE by config ---------------------------------------
ax = axes[1]
groups = [("ctrl_clean", "clean", "clean\ncontrol"),
          ("ctrl_mcar", "linear", "MCAR\nlinear"),
          ("ctrl_mcar", "zero", "MCAR\nzero"),
          ("ctrl_mnar", "linear", "synth-MNAR\nlinear"),
          ("ctrl_mnar", "zero", "synth-MNAR\nzero"),
          ("cens", "linear", "REAL cens\nlinear"),
          ("cens", "zero", "REAL cens\nzero"),
          ("cens", "keep", "REAL cens\nas-observed")]
x = np.arange(len(groups))
for k, m in enumerate(MODELS):
    vals = []
    for g, f, _ in groups:
        r = next(r for r in recs if r["model"] == m and r["group"] == g and r["fill"] == f)
        vals.append(np.mean(r["nmse_per_window"]))
    ax.plot(x, vals, "o-", color=COLORS[m], label=m, lw=1.4, ms=4)
ax.set_xticks(x); ax.set_xticklabels([g[2] for g in groups], fontsize=6.5)
ax.set_ylabel("mean NMSE (per window, log scale)")
ax.set_title("B. Forecast error by context condition\n(602 real-censored + 602 matched clean windows)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax.set_yscale("log")

# --- panel C: paired top-decile MSE ratios (MNAR signature) ----------------
ax = axes[2]
data, labels, cols = [], [], []
for g, f, lab in [("ctrl_mcar", "linear", "MCAR linear"), ("ctrl_mnar", "linear", "MNAR linear"),
                  ("ctrl_mnar", "zero", "MNAR zero")]:
    for m in MODELS:
        base = next(r for r in recs if r["model"] == m and r["group"] == "ctrl_clean")
        btd = np.array([v if v is not None else np.nan for v in base["mse_topdecile_per_window"]])
        r = next(r for r in recs if r["model"] == m and r["group"] == g and r["fill"] == f)
        atd = np.array([v if v is not None else np.nan for v in r["mse_topdecile_per_window"]])
        ratio = atd / np.maximum(btd, 1e-9)
        ratio = ratio[np.isfinite(ratio)]
        data.append(np.clip(ratio, 0, 6))
        labels.append(f"{lab}\n{m}")
        cols.append(COLORS[m])
bp = ax.boxplot(data, showfliers=False, widths=0.6, patch_artist=True,
                medianprops=dict(color="black"))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.5)
ax.axhline(1.0, color="gray", ls="--", lw=1)
ax.set_xticklabels(labels, fontsize=6)
ax.set_ylabel("top-decile future MSE / clean (paired, clipped at 6)")
ax.set_title("C. MNAR signature on real data: error on high-value\nfuture points (synthetic masks, same windows)")
ax.grid(alpha=0.3)

# --- panel D: dose-response on real censored windows ----------------------
ax = axes[3]
rates = np.array([w["rate"] for w in windows if w["kind"] == "cens"])
t1, t2 = np.quantile(rates, [1 / 3, 2 / 3])
bins = np.digitize(rates, [t1, t2])
centers = [rates[bins == b].mean() for b in range(3)]
for m in MODELS:
    for fill, ls in [("keep", "-"), ("linear", "--")]:
        r = next(r for r in recs if r["model"] == m and r["group"] == "cens" and r["fill"] == fill)
        nm = np.array(r["nmse_per_window"])
        ax.plot(centers, [np.median(nm[bins == b]) for b in range(3)], ls,
                color=COLORS[m], marker="o", ms=4, lw=1.4, label=f"{m} ({fill})")
# var-matched clean reference (median clean NMSE per model, flat)
for m in MODELS:
    base = next(r for r in recs if r["model"] == m and r["group"] == "ctrl_clean")
    ax.axhline(np.median(base["nmse_per_window"]), color=COLORS[m], ls=":", lw=1, alpha=0.6)
ax.set_xlabel("window censor rate (tercile means)")
ax.set_ylabel("median NMSE")
ax.set_title("D. Dose-response on REAL censored windows\n(dotted = clean-control median per model)")
ax.legend(fontsize=6.5, ncol=2)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "s6_real_censor.png"), dpi=130)
print("saved s6_real_censor.png")

# --- derived stats into the json -------------------------------------------
def rec(m, g, f):
    return next(r for r in recs if r["model"] == m and r["group"] == g and r["fill"] == f)

analysis = {"median_nmse": {}, "mean_nmse": {}, "td_rest_ratio": {},
            "paired_topdecile_ratio_median": {}, "cens_dose_response": {}}
for m in MODELS:
    analysis["median_nmse"][m] = {f"{g}:{f}": float(np.median(rec(m, g, f)["nmse_per_window"]))
                                  for g, f, _ in groups}
    analysis["mean_nmse"][m] = {f"{g}:{f}": float(np.mean(rec(m, g, f)["nmse_per_window"]))
                                for g, f, _ in groups}
    analysis["td_rest_ratio"][m] = {
        f"{g}:{f}": rec(m, g, f)["mse_topdecile"] / max(rec(m, g, f)["mse_rest"], 1e-9)
        for g, f, _ in groups}
    base = rec(m, "ctrl_clean", "clean")
    btd = np.array([v if v is not None else np.nan for v in base["mse_topdecile_per_window"]])
    for g, f in [("ctrl_mcar", "linear"), ("ctrl_mnar", "linear"), ("ctrl_mnar", "zero")]:
        atd = np.array([v if v is not None else np.nan for v in rec(m, g, f)["mse_topdecile_per_window"]])
        analysis["paired_topdecile_ratio_median"][f"{m}:{g}:{f}"] = float(
            np.nanmedian(atd / np.maximum(btd, 1e-9)))
    for fill in ("keep", "linear", "zero"):
        nm = np.array(rec(m, "cens", fill)["nmse_per_window"])
        analysis["cens_dose_response"][f"{m}:{fill}"] = [
            float(np.median(nm[bins == b])) for b in range(3)]
res["analysis"] = analysis
with open(os.path.join(HERE, "s6_real_censor_results.json"), "w") as f:
    json.dump(res, f)
print("analysis block written into s6_real_censor_results.json")
