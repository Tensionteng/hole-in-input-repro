#!/usr/bin/env python
"""Analyze s6_real_censor_results.json: robust stats + baselines + pairing."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CTX, H = 144, 24
NPZ = os.path.join(HERE, "dataset_censor", "penmanshiel2016_processed.npz")
VAR_FLOOR = 100.0

res = json.load(open(os.path.join(HERE, "s6_real_censor_results.json")))
windows = res["meta"]["windows"]
d = np.load(NPZ)

# rebuild contexts/targets exactly like run_s6 (deterministic)
tags = sorted({w["turbine"] for w in windows})
series = {t: dict(power=d[f"{t}_power"], cens=d[f"{t}_cens"], curt=d[f"{t}_curt"]) for t in tags}
n = len(windows)
ctx_clean = np.full((n, CTX), np.nan, np.float32)
tgt = np.full((n, H), np.nan, np.float32)
for i, w in enumerate(windows):
    s = series[w["turbine"]]
    t = w["start"]
    ctx_clean[i] = s["power"][t - CTX:t]
    tgt[i] = s["power"][t:t + H]

kind = np.array([w["kind"] for w in windows])
cens_idx = np.flatnonzero(kind == "cens")
ctrl_idx = np.flatnonzero(kind == "ctrl")

# ---- baselines (no GPU): persistence (last value) and daily-naive ----------
def nmse_of(pred, Y):
    err2 = (pred - Y) ** 2
    var = np.maximum(Y.var(axis=1), VAR_FLOOR)
    return err2.mean(axis=1) / var, err2.mean(axis=1)

Yall = tgt.astype(np.float64)
pers = np.repeat(ctx_clean[:, -1:], H, axis=1).astype(np.float64)      # last observed
seas = ctx_clean[:, CTX - H:].astype(np.float64)                       # 24h earlier
for name, P in [("persistence", pers), ("daily-naive", seas)]:
    for grp, idx in [("cens", cens_idx), ("ctrl", ctrl_idx)]:
        nm, ms = nmse_of(P[idx], Yall[idx])
        print(f"baseline {name:12s} {grp}: medianNMSE={np.median(nm):.3f} "
              f"meanNMSE={nm.mean():.2f} mse={ms.mean():.0f}")

# target variance / level distributions: intrinsic difficulty check
v = Yall.var(axis=1)
print("\ntarget var: cens median", np.median(v[cens_idx]).round(0),
      " ctrl median", np.median(v[ctrl_idx]).round(0))
print("target mean: cens", Yall[cens_idx].mean().round(0), " ctrl", Yall[ctrl_idx].mean().round(0))

# ---- per-config robust stats ------------------------------------------------
print("\nper-config (NMSE median / mean, MSE):")
recs = res["records"]
for r in recs:
    nm = np.array(r["nmse_per_window"])
    ms = np.array(r["mse_per_window"])
    print(f"  {r['model']:8s} {r['group']:10s} {r['fill']:7s} "
          f"medNMSE={np.median(nm):8.3f} meanNMSE={nm.mean():9.2f} "
          f"medMSE={np.median(ms):9.0f} mse={ms.mean():9.0f} "
          f"td/rest={r['mse_topdecile']/max(r['mse_rest'],1e-9):5.2f}")

# ---- paired analysis: synthetic groups vs clean on the SAME ctrl windows ----
print("\npaired per-window NMSE ratios vs clean (ctrl windows):")
for model in ("bolt", "timesfm", "moirai"):
    base = next(r for r in recs if r["model"] == model and r["group"] == "ctrl_clean")
    b = np.array(base["nmse_per_window"])
    for grp, fill in [("ctrl_mcar", "linear"), ("ctrl_mcar", "zero"),
                      ("ctrl_mnar", "linear"), ("ctrl_mnar", "zero")]:
        r = next(r for r in recs if r["model"] == model and r["group"] == grp and r["fill"] == fill)
        a = np.array(r["nmse_per_window"])
        ratio = a / np.maximum(b, 1e-9)
        print(f"  {model:8s} {grp}:{fill:7s} median ratio={np.median(ratio):6.3f} "
              f"mean={ratio.mean():8.2f} p90={np.quantile(ratio,0.9):8.2f}")

# ---- cens group vs matched-difficulty control: compare windows with similar
# target variance (nearest neighbor on log var) -------------------------------
print("\nvariance-matched comparison (median NMSE):")
lv_c = np.log(np.maximum(v[cens_idx], 1.0))
lv_o = np.log(np.maximum(v[ctrl_idx], 1.0))
matched = []
for x in lv_c:
    matched.append(ctrl_idx[np.argmin(np.abs(lv_o - x))])
matched = np.array(matched)
for model in ("bolt", "timesfm", "moirai"):
    base = next(r for r in recs if r["model"] == model and r["group"] == "ctrl_clean")
    b = np.array(base["nmse_per_window"])
    w2i = {w["window_id"]: k for k, w in enumerate([w for w in windows if w["kind"]=="ctrl"])}
    # records store per-window arrays in window order of the group; ctrl order = ctrl_idx order
    bmap = dict(zip(ctrl_idx.tolist(), b.tolist()))
    mb = np.array([bmap[i] for i in matched])
    for fill in ("linear", "zero", "keep"):
        r = next(r for r in recs if r["model"] == model and r["group"] == "cens" and r["fill"] == fill)
        a = np.array(r["nmse_per_window"])
        print(f"  {model:8s} cens:{fill:7s} medNMSE={np.median(a):8.3f} vs "
              f"var-matched ctrl clean medNMSE={np.median(mb):8.3f} "
              f"(ratio of medians {np.median(a)/max(np.median(mb),1e-9):6.2f})")
