#!/usr/bin/env python
"""Hand-check every number the new sections claim, against the source JSONs.

Each entry is (claim as written in the paper, recomputed value). Nothing here reads a
scorecard or a notes file -- only the raw results.
"""
import glob, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda *p: json.load(open(os.path.join(HERE, *p)))
ok = bad = 0
CLAIMS = []   # every value this file asserts, for audit_prose_numbers.py
INS = ("ETTh1", "ETTm1", "weather")   # the three datasets every adapter/detector was fitted on


def chk(label, claim, got, tol=0.02):
    global ok, bad
    CLAIMS.append(float(claim))
    if got is None:
        print(f"  ??  {label:58s} claim {claim}   (not recomputable)"); bad += 1; return
    good = (abs(got - claim) <= tol * max(abs(claim), 1e-9) + 1e-12)
    print(f"  {'ok ' if good else 'BAD'} {label:58s} claim {claim:>10} got {got:>10.4g}")
    ok += good; bad += (not good)


print("== S41 / Tables 1-2 (nine datasets, both metrics) ==")
S41 = {}
for f in sorted(glob.glob(os.path.join(HERE, "s41_mae", "part_*.json"))):
    d = json.load(open(f))
    for k in ("grid", "probe", "closure"):
        S41.setdefault(k, {}).update(d.get(k, {}))
CNT = json.load(open(os.path.join(HERE, "..", "paper", "iclr2026", "figures",
                                  "dataset_counts.json")))
DS9 = list(CNT)
pl = [S41["probe"][d]["plain"]["rms"] for d in DS9]
chk("rho plain, min", 0.79, min(pl)); chk("rho plain, max", 0.99, max(pl))
chk("rho on the NaN path, max", 0.0, max(S41["probe"][d]["nan"]["rms"] for d in DS9), tol=0)
mk = [S41["probe"][d]["mask"]["rms"] for d in DS9]
chk("rho on the mask path, min (x1e-7)", 5.9, min(mk) * 1e7, tol=.02)
chk("rho on the mask path, max (x1e-6)", 9.6, max(mk) * 1e6, tol=.02)
chk("total series", 31983, sum(v["n_series"] for v in CNT.values()), tol=0)
mech = ["mcar", "block", "mnar_high", "mnar_extreme"]
worst = [max(range(4), key=lambda i: S41["grid"][f"{d}|{mech[i]}|0.7"]["mse"]["median"])
         for d in DS9]
chk("datasets where censoring is the worst cell", 8, sum(w == 2 for w in worst), tol=0)
chk("exchange, extreme at 0.7", 2.86, S41["grid"]["exchange|mnar_extreme|0.7"]["mse"]["median"])
chk("exchange, censoring at 0.7", 1.34, S41["grid"]["exchange|mnar_high|0.7"]["mse"]["median"])
sc = [S41["grid"][f"{d}|mcar|0.7"]["mse"]["median"] for d in DS9]
ce = [S41["grid"][f"{d}|mnar_high|0.7"]["mse"]["median"] for d in DS9]
chk("scattered at 0.7, min", 1.03, min(sc)); chk("scattered at 0.7, max", 3.07, max(sc))
chk("censoring at 0.7, min", 1.34, min(ce)); chk("censoring at 0.7, max", 22.35, max(ce))
chk("weather censoring at 0.7", 1.42, S41["grid"]["weather|mnar_high|0.7"]["mse"]["median"])

print("\n== S41 / App. MAE (does anything depend on squaring?) ==")
worst_mae = [max(range(4), key=lambda i: S41["grid"][f"{d}|{mech[i]}|0.7"]["mae"]["median"])
             for d in DS9]
chk("censoring worst under MAE too", 8, sum(w == 2 for w in worst_mae), tol=0)
for m, name, claim in (("mcar", "scattered", .60), ("block", "block", .57),
                       ("mnar_high", "censoring", .42), ("mnar_extreme", "extreme", .47)):
    r = [(S41["grid"][f"{d}|{m}|0.7"]["mae"]["median"] - 1) /
         (S41["grid"][f"{d}|{m}|0.7"]["mse"]["median"] - 1) for d in DS9
         if S41["grid"][f"{d}|{m}|0.7"]["mse"]["median"] - 1 > 0.05]
    chk(f"MAE/MSE excess ratio, {name}", claim, float(np.median(r)), tol=.02)
ce_mse = float(np.median([S41["grid"][f"{d}|mnar_high|0.7"]["mse"]["median"] - 1
                          for d in DS9]))
ce_mae = float(np.median([S41["grid"][f"{d}|mnar_high|0.7"]["mae"]["median"] - 1
                          for d in DS9]))
chk("censoring excess at 0.7, MSE", 2.29, ce_mse); chk("censoring excess at 0.7, MAE", 1.05, ce_mae)
fl = tot = 0
med = {"mse": {"in": [], "out": []}, "mae": {"in": [], "out": []}}
for k, c in S41["closure"].items():
    ds = k.split("|")[0]
    a, b = c["mse"]["closure"], c["mae"]["closure"]
    s2 = c["mse"]["native_a1"] - 1 >= 0.02
    s1 = c["mae"]["native_a1"] - 1 >= 0.02
    if s2 and np.isfinite(a):
        med["mse"]["in" if ds in INS else "out"].append(a)
    if s1 and np.isfinite(b):
        med["mae"]["in" if ds in INS else "out"].append(b)
    if s2 and s1 and np.isfinite(a) and np.isfinite(b):
        tot += 1; fl += (a > 0) != (b > 0)
chk("closure sign flips between metrics", 1, fl, tol=0)
chk("closure cells compared", 63, tot, tol=0)
chk("in-sample closure median, MAE (%)", 57.2, np.median(med["mae"]["in"]) * 100, tol=.02)
chk("held-out closure median, MAE (%)", 71.5, np.median(med["mae"]["out"]) * 100, tol=.02)
dif = [abs(S41["probe"][d][c]["rms"] - S41["probe"][d][c]["l1"])
       for d in DS9 for c in ("plain", "mask", "nan")]
chk("max |rho_rms - rho_L1|", 0.015, max(dif), tol=.05)

print("\n== S37 / Section 6.1 (the interface fix) ==")
S = {"closure": {}, "clean_own": {}}
for f in ("own_insample", "own_held_a", "own_held_b", "own_held_c"):
    d = J("s37_capbreadth", f + ".json")
    for k in S: S[k].update(d.get(k, {}))
MIN = 0.02
cl = {"in": [], "out": []}
a0 = [0, 0]
for k, c in S["closure"].items():
    if c["native_a1"] - 1.0 < MIN: continue
    cl["in" if k.split("|")[0] in INS else "out"].append(c["closure"])
    a0[1] += 1; a0[0] += c["restored_a0"] <= c["native_a0"] + 1e-9
chk("in-sample cells", 23, len(cl["in"]), tol=0)
chk("in-sample positive", 22, sum(v > 0 for v in cl["in"]), tol=0)
chk("in-sample median closure (%)", 53, np.median(cl["in"]) * 100, tol=.02)
chk("held-out cells", 44, len(cl["out"]), tol=0)
chk("held-out positive", 44, sum(v > 0 for v in cl["out"]), tol=0)
chk("held-out median closure (%)", 73, np.median(cl["out"]) * 100, tol=.02)
chk("alpha=0 no worse, count", 41, a0[0], tol=0)
chk("alpha=0 total cells", 67, a0[1], tol=0)
cc = J("s37_capbreadth", "clean_control.json")
ett = [cc[d]["clean_relMSE"]["dual"] for d in ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather")]
big = [cc[d]["clean_relMSE"]["dual"] for d in ("electricity", "traffic", "illness")]
chk("clean control, ETT/weather best gain (%)", 3.0, (1 - min(ett)) * 100, tol=.12)
chk("clean control, big-corpus worst loss (%)", 9.0, (max(big) - 1) * 100, tol=.12)

SW = J("s27_interface", "s27_results.json")["sweep"]
DS3 = ("ETTh1", "ETTm1", "weather")
flat = []
for m in ("mcar", "block"):
    for r in (0.3, 0.7):
        a0 = np.mean([SW[f"zeroshot|{d}|{m}|{r}|0.0|native"]["median"] for d in DS3])
        a1 = np.mean([SW[f"zeroshot|{d}|{m}|{r}|1.0|native"]["median"] for d in DS3])
        flat.append(abs(a1 - a0) / (a0 - 1.0))
chk("declared sweep, scattered+block min (%)", 3.8, min(flat) * 100, tol=.02)
chk("declared sweep, scattered+block max (%)", 5.2, max(flat) * 100, tol=.02)
cens = []
for m in ("mnar_high", "mnar_extreme"):
    for r in (0.3, 0.7):
        a0 = np.mean([SW[f"zeroshot|{d}|{m}|{r}|0.0|native"]["median"] for d in DS3])
        a1 = np.mean([SW[f"zeroshot|{d}|{m}|{r}|1.0|native"]["median"] for d in DS3])
        cens.append(abs(a1 - a0) / (a0 - 1.0))
chk("declared sweep, censoring min (%)", 6.7, min(cens) * 100, tol=.02)
chk("declared sweep, censoring max (%)", 40, max(cens) * 100, tol=.02)

print("\n== S38 / Section 10 (calibration across nine datasets) ==")
R = {}
for f in ("gate", "heldout_a", "heldout_b"):
    R.update(J("s38_calibbreadth", f + ".json")["datasets"])
cens = [(r["groups"][g]["native"]["coverage"], r["groups"][g]["naive"]["coverage"],
         r["groups"][g]["mondrian_pred"]["coverage"])
        for r in R.values() for g in ("mnar_high", "mnar_extreme") if g in r["groups"]]
chk("censoring cells", 18, len(cens), tol=0)
chk("censoring native mean", 0.442, np.mean([c[0] for c in cens]))
chk("censoring 1-pool mean", 0.808, np.mean([c[1] for c in cens]))
chk("censoring mondrian mean", 0.901, np.mean([c[2] for c in cens]))
chk("censoring within 0.03 of nominal", 16,
    sum(abs(c[2] - .9) <= .03 for c in cens), tol=0)
GR = ["clean", "mcar", "block", "mnar_high", "mnar_extreme"]
for t, lab, m, sd in (("in", INS, 0.901, 0.018), ("out", None, 0.901, 0.014)):
    v = [r["groups"][g]["mondrian_pred"]["coverage"] for k, r in R.items()
         for g in GR if g in r["groups"] and ((k in INS) == (t == "in"))]
    chk(f"mondrian mean, {t}-sample", m, float(np.mean(v)))
    chk(f"mondrian sd, {t}-sample", sd, float(np.std(v)), tol=.06)
d = [abs(r["groups"][g]["mondrian_pred"]["coverage"] -
         r["groups"][g]["mondrian_oracle"]["coverage"])
     for r in R.values() for g in GR if g in r["groups"]]
chk("pred-vs-oracle max |delta|", 0.029, max(d), tol=.05)
chk("pred-vs-oracle mean |delta|", 0.003, float(np.mean(d)), tol=.2)

print("\n== S39 / Section 10 (real GIFT-Eval missingness) ==")
s39 = J("s39_realcalib", "s39_results.json")["strata"]
ks = list(s39)
chk("cleanest stratum, native", 0.822, s39[ks[0]]["native"]["coverage"])
chk("most-missing stratum, native", 0.733, s39[ks[-1]]["native"]["coverage"])
chk("most-missing, single pool", 0.829, s39[ks[-1]]["naive"]["coverage"])
chk("most-missing, mondrian", 0.840, s39[ks[-1]]["mondrian"]["coverage"])
chk("most-missing, mondrian x rate", 0.855, s39[ks[-1]]["mondrian_rate"]["coverage"])
chk("15-30% stratum, single pool", 0.877, s39[ks[-2]]["naive"]["coverage"])
chk("15-30% stratum, mondrian x rate", 0.907, s39[ks[-2]]["mondrian_rate"]["coverage"])
wid = [(s39[k]["mondrian_rate"]["width"] > s39[k]["native"]["width"]) for k in ks]
chk("strata where the interval WIDENS", len(ks), sum(wid), tol=0)

print("\n== S36 / Table 3 (thirteen checkpoints) ==")
m36 = {}
for f in glob.glob(os.path.join(HERE, "s36_models", "part_*.json")):
    m36.update(json.load(open(f)).get("models", {}))
m30 = J("s30_crossmodel", "s30_results.json")["models"]
bolt_dec = [m36[k]["agg"]["mask"]["ratio_median"] for k in
            ("bolt_tiny", "bolt_mini", "bolt_small")] + \
           [m30["bolt"]["agg"]["mask"]["ratio_median"]]
chk("bolt ladder, max declared rho (x1e-6)", 2.2, max(bolt_dec) * 1e6, tol=.05)
moirai = [m36[k]["agg"]["nan"]["ratio_median"] for k in
          ("moirai_s", "moirai_l", "moe_s", "moe_b")] + \
         [m30["moirai"]["agg"]["nan"]["ratio_median"]]
chk("moirai family, min declared rho", 0.938, min(moirai), tol=.01)
chk("chronos-t5-base declared rho", 0.0, m36["t5_base"]["agg"]["nan"]["ratio_median"], tol=0)
chk("chronos-t5-base plain rho", 0.910, m36["t5_base"]["agg"]["plain"]["ratio_median"],
    tol=.01)

print("\n== S40 / Section 4 (rho predicts the score movement) ==")
rows = J("s40_rho_vs_spread", "s40_summary.json")
x = [r["gap"] for r in rows if r["gap"] is not None and r["spread_kdd"] is not None]
y = [r["spread_kdd"] for r in rows if r["gap"] is not None and r["spread_kdd"] is not None]
sr = stats.spearmanr(x, y)
chk("gap vs kdd spread, Spearman", 0.750, sr.statistic, tol=.02)
chk("gap vs kdd spread, p", 0.020, sr.pvalue, tol=.15)
pr = stats.pearsonr(x, y)
chk("gap vs kdd spread, Pearson", 0.763, pr.statistic, tol=.02)
xr = [r["rho_plain"] for r in rows if r["rho_plain"] is not None and r["spread_kdd"] is not None]
yr = [r["spread_kdd"] for r in rows if r["rho_plain"] is not None and r["spread_kdd"] is not None]
chk("rho_plain vs kdd spread, Spearman", -0.717, stats.spearmanr(xr, yr).statistic, tol=.02)
xp = [r["rho_plain"] for r in rows if r["rho_plain"] is not None]
yp = [r["spread_median"] for r in rows if r["rho_plain"] is not None]
chk("rho_plain vs pooled spread, Spearman", 0.800, stats.spearmanr(xp, yp).statistic, tol=.02)
t5 = [r for r in rows if r["name"] == "chronos-t5-base"][0]
chk("chronos-t5-base gap", 0.910, t5["gap"], tol=.01)
chk("chronos-t5-base kdd spread (%)", 11.1, t5["spread_kdd"] * 100, tol=.02)
bolt = {r["name"]: r["spread_kdd"] for r in rows if r["name"].startswith("chronos-bolt")}
for n, v in (("tiny", 18.4), ("mini", 24.0), ("small", 23.7), ("base", 32.3)):
    chk(f"bolt-{n} spread on kdd (%)", v, bolt[f"chronos-bolt-{n}"] * 100, tol=.02)

print("\n== s12558 merge registration ==")

# 305M: GIFT-Eval's total point count. No results JSON stores it; recompute the same
# full-corpus scan verify_main_text.py uses for the 11.77% NaN share (the count is
# documented as 305,205,463 in s32_gifteval/s32_notes.md).
GIFT = os.path.abspath(os.path.join(HERE, "..", "data", "gifteval"))
gift_m = None
if os.path.isdir(GIFT):
    try:
        import pyarrow as pa
        tot = 0
        for d_ in sorted(os.listdir(GIFT)):
            for f in glob.glob(os.path.join(GIFT, d_, "**", "*.arrow"), recursive=True):
                with pa.memory_map(f, "rb") as src:
                    t = pa.ipc.open_stream(src).read_all()
                for row in t.column("target").to_pylist():
                    tot += np.asarray(row, dtype=np.float64).size
        gift_m = tot / 1e6
    except Exception:
        gift_m = None
chk("GIFT-Eval corpus, total points (M)", 305, gift_m)

# 2.52 / 1.26: the S5 three-dataset sweep behind fig:problem (left), bolt + linear fill
# at rate 0.7; the prose quotes the sweep's ETTh1 row.
S5 = J("s05_characterization", "s5_missing_results.json")["bolt"]
chk("censoring at 0.7, linear, ETTh1 (S5 sweep)", 2.52,
    S5["ETTh1"]["mnar_high:linear:0.7"]["mse"] / S5["ETTh1"]["clean:none:0.0"]["mse"])
chk("scattered at 0.7, linear, ETTh1 (S5 sweep)", 1.26,
    S5["ETTh1"]["mcar:linear:0.7"]["mse"] / S5["ETTh1"]["clean:none:0.0"]["mse"])
chk("censoring at 0.7, linear, three-dataset mean (S5 sweep)", 2.17,
    float(np.mean([S5[d]["mnar_high:linear:0.7"]["mse"] / S5[d]["clean:none:0.0"]["mse"]
                   for d in INS])))
chk("scattered at 0.7, linear, three-dataset mean (S5 sweep)", 1.07,
    float(np.mean([S5[d]["mcar:linear:0.7"]["mse"] / S5[d]["clean:none:0.0"]["mse"]
                   for d in INS])))

# 40.2 / 2.799: SAITS/BRITS end-to-end (tab:endtoend's oracle row, and the declared-path
# relMSE that SAITS fill improves on).
S42 = J("s42_end2end", "s42_results.json")["cells"]
marg = []
for ds in INS:
    for m in ("mcar", "block", "mnar_high", "mnar_extreme"):
        for r in ("0.3", "0.7"):
            k = f"{ds}|{m}|{r}|oracle"
            nat, du = S42[k + "|native"]["median"], S42[k + "|dual"]["median"]
            marg.append((nat - du) / max(nat - 1.0, 1e-9))
chk("end-to-end oracle row, median closure (%)", 40.2, float(np.median(marg)) * 100)
chk("declared ETTh1 censoring 0.7 before SAITS (linear)", 2.799,
    S42["ETTh1|mnar_high|0.7|linear|native"]["median"])

# 2.49 / 1.29: gradient-free fill attack on Moirai 2.0's retaining path (S46 attack
# cells, three datasets x {block, censoring} at nominal rate 0.5); the prose quotes the
# range of the per-cell median damage.
S46A = J("s46_moirai2", "s46_results.json")["attack"]
m2 = [v["ratio_median"] for k, v in S46A.items() if k.endswith("|moirai2")]
chk("moirai2 retaining-path attack, max median damage", 2.49, max(m2))
chk("moirai2 retaining-path attack, min median damage", 1.29, min(m2))

# 72.7 / 0.852: bolt-tiny 2x2 factorial, restored + mechanism-diverse arm (C) against
# the same-seed baseline (A), three paired seeds.
S48S = J("s45_pretrain", "s48_eval_seeds.json")
SEEDS = ("", "@20260901", "@20260902")
MECHS4 = ("mcar", "block", "mnar_high", "mnar_extreme")


def s48_closure(E, arm, ref, ds_set):
    vals = []
    for ds in ds_set:
        for m in MECHS4:
            for r in (0.3, 0.7):
                k = f"{ds}|{m}|{r}|1.0"
                a = E["sweep"].get(f"{k}|{arm}|declared")
                b = E["sweep"].get(f"{k}|{ref}|declared")
                if a is None or b is None:
                    continue
                ex = b["median"] - 1.0
                if ex <= 0.02:
                    continue
                vals.append((b["median"] - a["median"]) / ex)
    return float(np.median(vals))


cl = [s48_closure(S48S, "C" + s, "A" + s, INS) for s in SEEDS]
chk("factorial restored+mech-div closure, 3-seed mean (%)", 72.7, np.mean(cl) * 100)
chk("factorial restored+mech-div closure, 3-seed sd", 16.1, np.std(cl, ddof=1) * 100)
rho = [float(np.median([S48S["probe"][f"{ds}|{m}|C{s}|declared"]["rho"]
                        for ds in INS for m in ("mcar", "block")])) for s in SEEDS]
chk("factorial restored+mech-div perm rho, 3-seed mean", 0.852, np.mean(rho))
chk("factorial restored+mech-div perm rho, 3-seed sd", 0.034, np.std(rho, ddof=1))

# 2.7 / 94.4: the nine-dataset, three-seed recipe ladder (S48 9ds grid).
S48N = J("s45_pretrain", "s48_eval_9ds.json")
gcl = [s48_closure(S48N, "G" + s, "A" + s, DS9) for s in SEEDS]
chk("recipe ladder declare-only closure, 3-seed mean (%)", 2.7, np.mean(gcl) * 100)
chk("recipe ladder declare-only closure, 3-seed sd", 63.0, np.std(gcl, ddof=1) * 100)
hcl = [s48_closure(S48N, "H" + s, "A" + s, DS9) for s in SEEDS]
chk("recipe ladder fill-diverse closure, 3-seed mean (%)", 94.4, np.mean(hcl) * 100)
chk("recipe ladder fill-diverse closure, 3-seed sd", 5.5, np.std(hcl, ddof=1) * 100)

# 1.37: Moirai 2.0's declared path on ETTh1 scattered dropout at rate 0.7, linear fill.
S61 = J("s61_m2retrofit", "s61_grid.json")["grid"]
chk("moirai2 declared ETTh1 scattered 0.7, linear", 1.37,
    S61["ETTh1|mcar|0.7|0.0|stock|declared"]["median"])

# 2.26: the bolt CPT retrofit (Q50) under zero fill, tab:leaderboard's hierarchical
# median -- per-dataset median over the four mechanism medians, then median over datasets.
S57Q = J("s45_pretrain", "s57_q50.json")["grid"]
zq = [float(np.median([S57Q[f"{d}|{m}|0.7|zero|Q50-base"]["median"] for m in MECHS4]))
      for d in DS9]
chk("bolt CPT retrofit, zero fill, hierarchical median", 2.26, float(np.median(zq)))

# tab:leaderboard clean-change column for the c2/m2 +5k no-missingness controls,
# n/r in the submitted table. Median over DS9 of per-dataset (control/stock - 1) in
# median clean MSE from the s65 grids' clean dicts. The same formula reproduces the
# printed bolt rows (+4.3/+4.2) and the s60/s61 CPT rows (-0.7/+1.1) exactly.
for name_, path_, ctl_, claim_ in (("c2", "s65_fairbudget/s65_c2_grid.json", "C2CTL", 5.2),
                                   ("m2", "s65_fairbudget/s65_m2_grid.json", "M2CTL", 2.5)):
    cl_ = J(*path_.split("/"))["clean"]
    ch_ = [(cl_[ctl_][d]["mse_median"] / cl_["stock"][d]["mse_median"] - 1) * 100
           for d in DS9]
    chk(f"{name_} no-missingness control, leaderboard clean change (%)", claim_,
        float(np.median(ch_)), tol=.02)

print(f"\n==== {ok} ok, {bad} bad ====")

json.dump(sorted(set(CLAIMS)), open(os.path.join(HERE, "claims_late_rounds.json"), "w"))
