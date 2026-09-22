#!/usr/bin/env python
"""Traces every numeric claim in the main text back to the run that produced it.

verify_paper_numbers_v2.py covers the rounds added late in the project. This file works
through the main text section by section, so that a claim's provenance is a check rather than
a memory. Numbers that are structural (rates, context lengths, section counts) are excluded by
audit_prose_numbers.py rather than checked here.
"""
import glob, json, os, re
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda *p: json.load(open(os.path.join(HERE, *p)))
ok = bad = 0
CLAIMS = []   # every value this file asserts, for audit_prose_numbers.py


def chk(label, claim, got, tol=0.02):
    global ok, bad
    CLAIMS.append(float(claim))
    good = got is not None and abs(got - claim) <= tol * max(abs(claim), 1e-9) + 1e-12
    print(f"  {'ok ' if good else 'BAD'} {label:60s} claim {claim:>10} got "
          + (f"{got:>10.4g}" if got is not None else f"{'n/a':>10}"))
    ok += good; bad += (not good)


# =============================================================== Section 9 ====
print("== Section 9: the error budget, migration and scale ==")
S25 = J("s25_twofloor", "s25_results.json")
AGG = J("s25_twofloor", "s25_paired_agg.json")

# the direct predictor's own clean error against zero-shot Bolt's
bolt_clean = {d: S25["partB"]["base"][f"{d}:clean"] for d in ("ETTh1", "ETTm1", "weather")}
dir_clean = {d: S25["partC"]["base"][f"{d}:clean"] for d in ("ETTh1", "ETTm1", "weather")}
for d, c in (("ETTh1", 10.10), ("ETTm1", 8.41), ("weather", 2088)):
    chk(f"direct predictor clean MSE, {d}", c, dir_clean[d])
for d, c in (("ETTh1", 10.36), ("ETTm1", 8.14), ("weather", 3675)):
    chk(f"zero-shot bolt clean MSE, {d}", c, bolt_clean[d])
wins = sum(dir_clean[d] < bolt_clean[d] for d in bolt_clean)
chk("datasets where the direct predictor beats bolt", 2, wins, tol=0)
chk("ETTm1 shortfall of the direct predictor (%)", 3.4,
    (dir_clean["ETTm1"] / bolt_clean["ETTm1"] - 1) * 100, tol=.05)

tot = {k: sum(AGG[k][t] for t in ("term1_information", "term2_architecture", "term3_fill"))
       for k in AGG}
chk("total excess, scattered at 0.7", 0.12, tot["mcar:0.7"], tol=.05)
chk("total excess, block at 0.7", 0.09, tot["block:0.7"], tol=.06)
chk("total excess, censoring at 0.7", 1.875, tot["mnar_high:0.7"])
chk("information term, censoring at 0.7", 0.308, AGG["mnar_high:0.7"]["term1_information"])
chk("architecture term, censoring at 0.7", 0.364, AGG["mnar_high:0.7"]["term2_architecture"])
chk("fill term, censoring at 0.7", 1.203, AGG["mnar_high:0.7"]["term3_fill"])
chk("fill share of the censoring excess (%)", 64,
    100 * AGG["mnar_high:0.7"]["term3_fill"] / tot["mnar_high:0.7"], tol=.05)
chk("learned fill win rate, censoring at 0.7 (%)", 81,
    100 * AGG["mnar_high:0.7"]["winrate"], tol=.02)
chk("information floor, censoring", 0.31, AGG["mnar_high:0.7"]["term1_information"], tol=.02)
chk("information floor, extreme censoring", 0.33,
    AGG["mnar_extreme:0.7"]["term1_information"], tol=.02)
chk("information floor, scattered", 0.06, AGG["mcar:0.7"]["term1_information"], tol=.05)
chk("information floor, block", 0.09, AGG["block:0.7"]["term1_information"], tol=.05)

# the real-curtailment replication (S26), recomputed from per-window errors
A26 = J("s26_realfill", "s26_results.json")["arms"]["penn|cens|aug0|c-zero"]
base = np.array(A26["fixed_zero"]["per_window"])
lrn = np.array(A26["learned_plain_fill"]["per_window"])
m = base > 0
r = lrn[m] / base[m]
chk("learned fill win rate on real curtailment (%)", 62, 100 * float((r < 1).mean()))
chk("paired median ratio on real curtailment", 0.938, float(np.median(r)))
chk("sign test p (x1e5)", 1.7, stats.binomtest(int((r < 1).sum()), len(r), 0.5).pvalue * 1e5,
    tol=.05)
chk("typical-window gain on real curtailment (%)", 6, 100 * (1 - float(np.median(r))), tol=.06)
chk("windows more than 2x worse (%)", 9.2, 100 * float((r > 2).mean()))

# scale (S28)
S28 = J("s28_scale", "s28_results.json")["models"]
for sz, c in (("tiny", 2.852), ("base", 2.875)):
    chk(f"censoring at 0.7, {sz}", c, S28[sz]["summary"]["mnar_high|0.7"])
for sz, c in (("tiny", 1.103), ("base", 1.121)):
    chk(f"scattered at 0.7, {sz}", c, S28[sz]["summary"]["mcar|0.7"])
for sz, c in (("tiny", 1.762), ("base", 1.827)):
    chk(f"extreme at 0.7, {sz}", c, S28[sz]["summary"]["mnar_extreme|0.7"])


# the corpus scan behind Section 4, re-read from the distributed arrow files
GIFT = os.path.abspath(os.path.join(HERE, "..", "data", "gifteval"))
if os.path.isdir(GIFT):
    import pyarrow as pa

    def nan_frac(ds):
        tot = nn = 0
        for f in sorted(glob.glob(os.path.join(GIFT, ds, "*.arrow"))):
            with pa.memory_map(f, "rb") as src:
                t = pa.ipc.open_stream(src).read_all()
            for row in t.column("target").to_pylist():
                a = np.asarray(row, dtype=np.float64)
                tot += a.size
                nn += int(np.isnan(a).sum())
        return 100 * nn / tot if tot else None

    chk("electricity, NaN share of points (%)", 19.2, nan_frac("electricity/H"))
    chk("restaurant, NaN share of points (%)", 13.8, nan_frac("restaurant"))
    if os.environ.get("FULL_SCAN"):      # ~2 min over 305M points
        tot = nn = 0
        for d_ in sorted(os.listdir(GIFT)):
            for f in glob.glob(os.path.join(GIFT, d_, "**", "*.arrow"), recursive=True):
                with pa.memory_map(f, "rb") as src:
                    t = pa.ipc.open_stream(src).read_all()
                for row in t.column("target").to_pylist():
                    a = np.asarray(row, dtype=np.float64)
                    tot += a.size
                    nn += int(np.isnan(a).sum())
        chk("GIFT-Eval corpus, NaN share of points (%)", 11.77, 100 * nn / tot, tol=.01)
        chk("GIFT-Eval corpus, total points (millions)", 305, tot / 1e6, tol=.01)

# The top-decile concentration of Section 5.1. Only the repaired arm survives in a results
# file; the 2.37 baseline it moves from lives in S5's summary, and is listed as a known gap by
# audit_prose_numbers.py rather than checked here.
T6 = J("s06_real_and_repairs", "s6_mnarfix_results.json")["bolt"]["ETTh1"]
c6 = T6["mnar_high:tail_tobit:0.7"]
chk("top-decile / rest after a truncation-aware repair, ETTh1", 2.25,
    c6["mse_topdecile"] / c6["mse_rest"])

# cross-channel recovery under a declared hole, synthetic block outages
chk("cross-channel recovery, block 0.7, declared (%)", 95,
    100 * J("s31_crosschannel", "s31_results.json")["agg"]["block|0.7"]["recovery_nan_median"],
    tol=.02)

# parameter counts quoted in Section 3, against the table that lists them
# the per-checkpoint parameter counts live in the full table, which the appendix carries
FAM = open(os.path.join(HERE, "..", "paper", "iclr2026", "figures",
                        "tab_families_full.tex")).read()
pars = [float(x) for x in re.findall(r"& ([\d.]+)M &", FAM)]
chk("smallest checkpoint (M params)", 8.7, min(pars))
chk("largest checkpoint (M params)", 935, max(pars))
chk("largest Chronos-Bolt (M params)", 205,
    max(float(x) for x in re.findall(r"\\bolt\{\} \w+ & ([\d.]+)M &", FAM)))

# channel counts quoted in Sections 5 and 6
CNT = json.load(open(os.path.join(HERE, "..", "paper", "iclr2026", "figures",
                                  "dataset_counts.json")))
import pandas as pd
tr = pd.read_csv(os.path.join(os.path.dirname(GIFT), "..", "..", "legacy_nonstationary",
                              "tslib", "dataset", "traffic", "traffic.csv"), nrows=1)
chk("traffic channel count", 862, tr.shape[1] - 1, tol=0)
chk("smallest channel count", 7, min(v["n_channels"] for v in CNT.values()), tol=0)



# =============================================================== Section 8 ====
print("\n== Section 8: the natural experiment ==")
S33 = J("s33_moirai_causal", "s33_results.json")
SW, M33 = S33["sweep"], S33["meta"]
DS3 = sorted({k.split("|")[0] for k in SW})


def sw(mech, rate, a, model, path="declared"):
    return float(np.mean([SW[f"{d}|{mech}|{rate}|{a}|{model}|{path}"]["median"] for d in DS3]))


chk("windows per cell", 150, M33["n_win"], tol=0)
for mech, rate, model, a, c in (
        ("mcar", 0.7, "moirai", 0.0, 1.092), ("mcar", 0.7, "moirai", 1.0, 0.999),
        ("mcar", 0.7, "bolt", 0.0, 1.215), ("mcar", 0.7, "bolt", 1.0, 1.219),
        ("block", 0.7, "moirai", 0.0, 1.348), ("block", 0.7, "moirai", 1.0, 1.002),
        ("block", 0.7, "bolt", 0.0, 1.094), ("block", 0.7, "bolt", 1.0, 1.086),
        ("mnar_high", 0.7, "bolt", 0.0, 3.067), ("mnar_high", 0.7, "bolt", 1.0, 2.213),
        ("mnar_high", 0.7, "moirai", 0.0, 2.132), ("mnar_high", 0.7, "moirai", 1.0, 1.642)):
    chk(f"{model} declared, {mech} p={rate}, alpha={a:.0f}", c, sw(mech, rate, a, model))

V33 = J("s33_moirai_causal", "s33_verify.json")
v1 = V33["v1"]
chk("library-path max relative difference", 2.14, v1["max_rel_diff"])
chk("library-path correlation", 0.975, v1["corr"])
chk("seed-to-seed max relative difference", 1.77, v1["self_max_rel_diff_two_seeds"])
chk("seed-to-seed correlation", 0.973, v1["self_corr_two_seeds"])
# the patch-size check: Moirai's declared-path slope under block missingness at rate 0.7
v2 = V33["v2"]
# these are ETTh1 values; the text now says so, and quotes ETTm1 beside them
for ds, ps, c in (("ETTh1", "patch32", -0.795), ("ETTh1", "patch64", -0.672),
                  ("ETTm1", "patch32", -0.247), ("ETTm1", "patch64", -0.502)):
    chk(f"moirai block slope, {ds} {ps}", c, v2[f"{ds}|block|{ps}"]["slope"], tol=.03)
for ds, c in (("ETTh1", 1.091), ("ETTm1", 0.973)):
    chk(f"moirai block relMSE at alpha=0, patch16, {ds}", c,
        v2[f"{ds}|block|patch16"]["rel_median"][0])


# =============================================================== Section 5 ====
print("\n== Section 5: what missingness does ==")
CAL = J("s05_characterization", "s5_extra_results.json")["bolt_cal"]["ETTh1"]
chk("clean coverage of the 80% interval, ETTh1", 0.742, CAL["clean:none:0.0"]["coverage80"])
chk("clean interval width, ETTh1", 4.51, CAL["clean:none:0.0"]["pi_width80"])
chk("scattered 0.7 coverage, ETTh1", 0.702, CAL["mcar:linear:0.7"]["coverage80"])
chk("censoring 0.7 coverage, ETTh1", 0.217, CAL["mnar_high:linear:0.7"]["coverage80"])
chk("censoring 0.7 interval width, ETTh1", 3.13, CAL["mnar_high:linear:0.7"]["pi_width80"])
chk("the interval narrows under censoring", 1,
    int(CAL["mnar_high:linear:0.7"]["pi_width80"] < CAL["clean:none:0.0"]["pi_width80"]),
    tol=0)
chk("the interval widens under scattered dropout", 1,
    int(CAL["mcar:linear:0.7"]["pi_width80"] > CAL["clean:none:0.0"]["pi_width80"]), tol=0)

# the two real deployments, from the stored-vs-rerun gate of S26
G = J("s26_realfill", "s26_results.json")["gate"]
chk("Penmanshiel, zero fill (NMSE)", 28.43, G["penn|cens|zero"]["rerun"])
chk("Penmanshiel, linear fill (NMSE)", 60.16, G["penn|cens|linear"]["rerun"])
chk("METR-LA, zero/as-recorded fill (NMSE)", 21.19, G["metr|miss|keep"]["rerun"])
chk("METR-LA, linear fill (NMSE)", 2.75, G["metr|miss|linear"]["rerun"])
chk("Penmanshiel: linear/zero", 2.1,
    G["penn|cens|linear"]["rerun"] / G["penn|cens|zero"]["rerun"], tol=.05)
chk("METR-LA: as-recorded/linear", 7.7,
    G["metr|miss|keep"]["rerun"] / G["metr|miss|linear"]["rerun"], tol=.05)

# the predictor that sees only the missingness pattern
A29 = J("s29_audit", "s29_results.json")["armA"]["penn|cens"]
chk("mask-only predictor, median NMSE", 2.13, A29["mask_only"]["median"])
chk("context-mean baseline, median NMSE", 5.29, A29["baseline_context_mean"]["median"])
chk("mask-only win rate against bolt (%)", 43, 100 * A29["mask_only_winrate_vs_bolt"])

# =========================================================== Section 4.3 =====
print("\n== Section 4.3: the target-completeness filter ==")
C29 = J("s29_audit", "s29_results.json")["armC"]
chk("windows removed by the filter (%)", 26,
    100 * C29["frac_windows_excluded_by_the_filter"])
chk("median NMSE of the removed windows", 17.65, C29["nmse_partial_median"])
chk("median NMSE of the surviving windows", 1.66, C29["nmse_complete_median"])
chk("how much harder the removed quarter is", 10.6,
    C29["nmse_partial_median"] / C29["nmse_complete_median"], tol=.02)
chk("optimism of the reported median (%)", 15.4, C29["optimism_median_pct"])

# =============================================================== Section 7 ====
print("\n== Section 7: the three consequences ==")
C31 = J("s31_crosschannel", "s31_results.json")["cells"]


def xc(mech, rate, field):
    return float(np.median([c[field] for k, c in C31.items()
                            if k.split("|")[2] == mech and k.split("|")[3] == str(rate)]))


chk("block 0.7, filled univariate", 1.357, xc("block", 0.7, "holed_uni"))
chk("block 0.7, filled + neighbours", 1.357, xc("block", 0.7, "holed_multi"))
chk("block 0.7, declared + neighbours", 1.023, xc("block", 0.7, "nan_multi"))
chk("scattered 0.7, filled univariate", 1.211, xc("mcar", 0.7, "holed_uni"))
chk("scattered 0.7, filled + neighbours", 1.084, xc("mcar", 0.7, "holed_multi"))
chk("censoring 0.7, filled univariate", 3.059, xc("mnar_high", 0.7, "holed_uni"))
chk("censoring 0.7, filled + neighbours", 2.967, xc("mnar_high", 0.7, "holed_multi"))

B31 = J("s31b_metrla_neighbours", "s31b_results.json")
S14 = J("s14_metrla", "s14_metrla_results.json")["records"]
n_miss = S14["detector|miss"]["n_windows"]
chk("METR-LA outage windows", 618, n_miss, tol=0)
chk("windows with two healthy neighbours (%)", 6.1,
    100 * B31["meta"]["n_fully_healthy"] / n_miss, tol=.02)
chk("declared + neighbours, median gain on real outages (%)", 2,
    100 * (1 - B31["nan_plus_nb"]["ratio_vs_filled_uni_median"]), tol=.15)
chk("filled univariate mean, real outages", 2.078, B31["filled_uni"]["nmse_mean"])
chk("declared + neighbours mean, real outages", 1.238, B31["nan_plus_nb"]["nmse_mean"])

B29 = J("s29_audit", "s29_results.json")["armB"]
M, P = B29["metr|miss"], B29["penn|cens"]
chk("METR-LA plain fill, median damage", 3.06, M["plain|untargeted"]["damage_ratio_median"])
chk("METR-LA plain fill, windows doubled (%)", 60,
    100 * M["plain|untargeted"]["frac_windows_doubled"], tol=.02)
chk("METR-LA plain fill, 90th percentile damage (x1e2)", 1.7,
    M["plain|untargeted"]["damage_ratio_q90"] / 1e2, tol=.05)
chk("METR-LA plain fill, maximum damage (x1e5)", 3.7,
    M["plain|untargeted"]["damage_ratio_max"] / 1e5, tol=.02)
chk("METR-LA declared path, median damage", 1.18,
    M["native|untargeted"]["damage_ratio_median"])
chk("METR-LA declared path, windows doubled (%)", 12,
    100 * M["native|untargeted"]["frac_windows_doubled"], tol=.02)
chk("METR-LA NaN path, maximum damage", 1.00, M["nan|untargeted"]["damage_ratio_max"], tol=0)
chk("Penmanshiel missing rate (%)", 14, 100 * P["missing_rate_mean"], tol=.02)
chk("Penmanshiel plain fill, median damage", 1.01,
    P["plain|untargeted"]["damage_ratio_median"], tol=.01)

# ============================================================== Section 10 ====
print("\n== Section 10: calibration ==")
S16 = J("s16_mechgate", "s16_results.json")
LOD = S16["trackA"]["lod"]
g = [LOD[d]["gbdt"]["acc"] for d in LOD]
lt = [LOD[d]["little"]["acc"] for d in LOD]
chk("detector leave-one-dataset-out accuracy, min", 0.92, min(g))
chk("detector leave-one-dataset-out accuracy, max", 0.98, max(g))
chk("single-statistic baseline, min", 0.49, min(lt))
chk("single-statistic baseline, max", 0.52, max(lt))

CR = S16["trackB"]["conf_real"]
chk("Penmanshiel, native coverage", 0.710, CR["penn"]["linear"]["cens_test"]["native"]["coverage"])
chk("Penmanshiel, mondrian coverage", 0.870,
    CR["penn"]["linear"]["cens_test"]["mondrian_pred"]["coverage"])
chk("METR-LA linear, native coverage", 0.710, CR["metrla"]["linear"]["miss_test"]["native"]["coverage"])
chk("METR-LA linear, mondrian coverage", 0.910,
    CR["metrla"]["linear"]["miss_test"]["mondrian_pred"]["coverage"])
chk("METR-LA declared, native coverage", 0.685, CR["metrla"]["nan"]["miss_test"]["native"]["coverage"])
chk("METR-LA declared, mondrian coverage", 0.905,
    CR["metrla"]["nan"]["miss_test"]["mondrian_pred"]["coverage"])

S38 = {}
for f in ("gate", "heldout_a", "heldout_b"):
    S38.update(J("s38_calibbreadth", f + ".json")["datasets"])
cens_pool = [r["groups"][q]["naive"]["coverage"] for r in S38.values()
             for q in ("mnar_high", "mnar_extreme") if q in r["groups"]]
chk("worst censoring coverage under a single pool", 0.732, min(cens_pool))
wid = {q: [r["groups"][q]["mondrian_pred"]["width"] / r["groups"][q]["native"]["width"]
           for r in S38.values() if q in r["groups"]]
       for q in ("clean", "mcar", "mnar_high", "mnar_extreme")}
calm = wid["clean"] + wid["mcar"]
cens = wid["mnar_high"] + wid["mnar_extreme"]
chk("median width multiplier, calm groups", 1.4, float(np.median(calm)), tol=.04)
chk("median width multiplier, censored groups", 3.7, float(np.median(cens)), tol=.04)
per_ds_ok = all(
    min(r["groups"][q]["mondrian_pred"]["width"] / r["groups"][q]["native"]["width"]
        for q in ("mnar_high", "mnar_extreme") if q in r["groups"])
    > max(r["groups"][q]["mondrian_pred"]["width"] / r["groups"][q]["native"]["width"]
          for q in ("clean", "mcar") if q in r["groups"])
    for r in S38.values())
chk("censored groups widen more than calm ones, on every dataset", 1, int(per_ds_ok), tol=0)
E = S38["ETTh1"]["groups"]; I = S38["illness"]["groups"]
chk("ETTh1 calm width multiplier", 1.4,
    max(E[q]["mondrian_pred"]["width"] / E[q]["native"]["width"] for q in ("clean", "mcar")),
    tol=.06)
chk("ETTh1 censored width multiplier", 3.5,
    max(E[q]["mondrian_pred"]["width"] / E[q]["native"]["width"]
        for q in ("mnar_high", "mnar_extreme")), tol=.03)
chk("illness calm width multiplier", 7.9,
    max(I[q]["mondrian_pred"]["width"] / I[q]["native"]["width"] for q in ("clean", "mcar")),
    tol=.02)
chk("illness censored width multiplier", 25.4,
    max(I[q]["mondrian_pred"]["width"] / I[q]["native"]["width"]
        for q in ("mnar_high", "mnar_extreme")), tol=.02)

# ========================================================= Sections 3, 4, 6 ===
print("\n== Sections 3, 4, 6 ==")
P0 = J("s25_twofloor", "s25_results.json")["part0"]["cells"]
K = "ratio_perm_over_redraw"


def rho(conv, mech=None):
    v = [P0[q]["perm_invariance"][K] for q in P0
         if q.endswith(conv) and (mech is None or q.split(":")[1] == mech)]
    return float(np.median(v))


ranks = {c: [s["rank_1e-4"] for q in P0 if q.endswith(c) for s in P0[q]["per_series"]]
         for c in ("plain_fill", "fill_mask", "nan")}
chk("series per convention", 72, len(ranks["plain_fill"]), tol=0)
chk("plain fill rank, min", 54, min(ranks["plain_fill"]), tol=0)
chk("plain fill rank, max", 64, max(ranks["plain_fill"]), tol=0)
chk("fill+mask rank is exactly 2 on every series", 72,
    sum(r == 2 for r in ranks["fill_mask"]), tol=0)
chk("NaN rank is exactly 0 on every series", 72, sum(r == 0 for r in ranks["nan"]), tol=0)
chk("max |J_M| under NaN", 0.0,
    max(s["absmax"] for q in P0 if q.endswith("nan") for s in P0[q]["per_series"]), tol=0)
chk("rho, plain fill (median)", 0.72, rho("plain_fill"))
chk("rho, fill+mask (median, x1e6)", 1.2, rho("fill_mask") * 1e6, tol=.05)
chk("rho, NaN", 0.0, rho("nan"), tol=0)
chk("rho plain, scattered", 1.05, rho("plain_fill", "mcar"))
chk("rho plain, block", 0.72, rho("plain_fill", "block"))
chk("rho plain, censoring", 0.34, rho("plain_fill", "mnar_high"))

C32 = J("s32_gifteval", "s32c_results.json")
chk("GIFT-Eval windows pooled", 7725, C32["pool"]["random"]["n"], tol=0)
chk("lowest stratum, windows", 4308, C32["bins"]["bolt-base|random|0.00-0.01"]["n"], tol=0)
low = [C32["bins"][f"{m}|random|0.00-0.01"]["spread_pct"]
       for m in ("bolt-base", "chronos2", "moirai", "timesfm")]
chk("lowest stratum, worst spread (%)", 0.3, max(low), tol=.35)
chk("top stratum, bolt spread (%)", 94, C32["bins"]["bolt-base|random|0.30-1.01"]["spread_pct"])
chk("top stratum, chronos-2 spread (%)", 106,
    C32["bins"]["chronos2|random|0.30-1.01"]["spread_pct"])
for m, c in (("bolt-base", 22), ("chronos2", 35), ("moirai", 47)):
    chk(f"last-window protocol, {m} spread (%)", c,
        C32["bins"][f"{m}|last|0.30-1.01"]["spread_pct"], tol=.03)

M26 = J("s26_realfill", "s26_results.json")["meta"]
chk("METR-LA variance floor is stated", 1, int("4.0 mph^2" in M26["metr"]), tol=0)
chk("Penmanshiel variance floor is stated", 1, int("100 kW^2" in M26["penn"]), tol=0)

# ================================= Sections 5, 8 and the new appendices ======
# Everything below was added in the revision round and had no automated check.

print("\n== Section 8: the same family, two recipes (Moirai 1.1 vs 2.0) ==")
S33 = J("s33_moirai_causal", "s33_results.json")["sweep"]
S46 = J("s46_moirai2", "s46_results.json")["sweep"]
DS3 = ("ETTh1", "ETTm1", "weather")


def dsavg(S, name, mech, rate, alpha, path="declared"):
    return float(np.mean([S[f"{d}|{mech}|{rate}|{alpha}|{name}|{path}"]["median"] for d in DS3]))


chk("Moirai 1.1, block 0.7, alpha=0 (dataset-averaged)", 1.348, dsavg(S33, "moirai", "block", 0.7, 0.0))
chk("Moirai 1.1, block 0.7, alpha=1 (dataset-averaged)", 1.002, dsavg(S33, "moirai", "block", 0.7, 1.0))
chk("Moirai 1.1, scattered 0.7, alpha=0", 1.092, dsavg(S33, "moirai", "mcar", 0.7, 0.0))
chk("Moirai 1.1, scattered 0.7, alpha=1", 0.999, dsavg(S33, "moirai", "mcar", 0.7, 1.0))
chk("Moirai 1.1 on ETTh1 alone, block 0.7, alpha=0", 1.804,
    S33["ETTh1|block|0.7|0.0|moirai|declared"]["median"])
chk("Moirai 1.1 on ETTh1 alone, block 0.7, alpha=1", 1.002,
    S33["ETTh1|block|0.7|1.0|moirai|declared"]["median"])
chk("Moirai 2.0, block 0.7, alpha=0", 1.244, dsavg(S46, "moirai2", "block", 0.7, 0.0))
chk("Moirai 2.0, block 0.7, alpha=1", 1.072, dsavg(S46, "moirai2", "block", 0.7, 1.0))
chk("Bolt, block 0.7, alpha=0", 1.094, dsavg(S46, "bolt", "block", 0.7, 0.0))
chk("Bolt, block 0.7, alpha=1", 1.086, dsavg(S46, "bolt", "block", 0.7, 1.0))
# the teaser's cross-round gate: Bolt is measured in S33 and S46 and must agree
chk("Bolt block-0.7 sweep agrees across S33 and S46", 0.0,
    max(abs(dsavg(S33, "bolt", "block", 0.7, a) - dsavg(S46, "bolt", "block", 0.7, a))
        for a in (0.0, 0.25, 0.5, 0.75, 1.0)), tol=0.005)

print("\n== Appendix: Moirai 1.1 patch-size robustness, read as closure ==")
V33 = J("s33_moirai_causal", "s33_verify.json")["v2"]
cl = {}
for k, v in V33.items():
    a0, a1 = v["rel_median"][0], v["rel_median"][-1]
    if a0 - 1.0 > 0.02:
        cl[k] = 100 * (a0 - a1) / (a0 - 1.0)
chk("cells with an excess to close, across patch sizes", 9, len(cl), tol=0)
chk("minimum closure over those cells (%)", 92.9, min(cl.values()), tol=.01)
chk("maximum closure over those cells (%)", 105.1, max(cl.values()), tol=.01)
chk("ETTh1 block excess at patch 16", 1.091, V33["ETTh1|block|patch16"]["rel_median"][0])
chk("ETTh1 block excess at patch 32", 1.807, V33["ETTh1|block|patch32"]["rel_median"][0])

print("\n== Appendix: the permutation probe against its own noise floor ==")
S52 = J("s52_rhonoise", "s52_results.json")["models"]
for key, name, conv, r, rs in (
        ("bolt", "Chronos-Bolt base", "plain", 0.907, 0.0),
        ("bolt", "Chronos-Bolt base", "nan", 0.0, 0.0),
        ("chronos2", "Chronos-2", "nan", 0.0, 0.0),
        ("moirai", "Moirai 1.1 base", "nan", 0.976, 0.498),
        ("moirai_s", "Moirai 1.1 small", "nan", 0.979, 0.573),
        ("moirai_l", "Moirai 1.1 large", "nan", 0.949, 0.466),
        ("moe_s", "Moirai-MoE small", "nan", 0.959, 0.597),
        ("moe_b", "Moirai-MoE base", "nan", 0.991, 0.605)):
    if key not in S52:
        chk(f"{name} {conv}: MISSING from s52_results.json", 1, 0, tol=0)
        continue
    agg = S52[key]["agg"][conv]
    chk(f"{name} {conv}: rho", r, agg["ratio_median"], tol=.01)
    chk(f"{name} {conv}: rho_self", rs, agg["ratio_self_median"], tol=.02)
chk("Chronos-Bolt mask path: rho (x1e6)", 1.2,
    S52["bolt"]["agg"]["mask"]["ratio_median"] * 1e6, tol=.1)
# the asymmetry the appendix rests on: a deterministic model has an exactly zero floor
chk("deterministic families: rho_self is exactly zero everywhere", 0.0,
    max(S52[k]["agg"][c]["ratio_self_median"] for k in ("bolt", "chronos2")
        for c in S52[k]["agg"]), tol=0)
# and every retaining checkpoint clears its own floor
chk("smallest gap between declared rho and its noise floor", 0.35,
    min(S52[k]["agg"]["nan"]["ratio_median"] - S52[k]["agg"]["nan"]["ratio_self_median"]
        for k in ("moirai", "moirai_s", "moirai_l", "moe_s", "moe_b")), tol=.05)

print("\n== Appendix: the patch-DC share of an imputer's residual ==")
S51 = J("s51_errstruct", "s51_results.json")["cells"]
S42 = J("s42_end2end", "s42_results.json")["cells"]
rows = []
for k, v in S51.items():
    nat, du = S42.get(f"{k}|native"), S42.get(f"{k}|dual")
    if nat is None or du is None or nat["median"] - 1.0 <= 0.02:
        continue
    rows.append((v["psi_median"], v["rmse_median"],
                 (nat["median"] - du["median"]) / (nat["median"] - 1.0), k.split("|")[1],
                 k.split("|")[3]))
chk("cells with an excess worth closing", 115, len(rows), tol=0)
sp = stats.spearmanr([r[0] for r in rows], [r[2] for r in rows])
chk("Spearman(psi, closure)", -0.272, sp.statistic, tol=.02)
chk("Spearman(psi, closure) p-value x1000", 3.3, sp.pvalue * 1000, tol=.2)
spr = stats.spearmanr([r[1] for r in rows], [r[2] for r in rows])
chk("Spearman(imputation RMSE, closure) is the wrong sign", 0.233, spr.statistic, tol=.02)
# within each mechanism, splitting at that mechanism's own median psi
lo, hi = [], []
for m in sorted(set(r[3] for r in rows)):
    sub = [r for r in rows if r[3] == m]
    thr = np.median([r[0] for r in sub])
    lo += [r[2] for r in sub if r[0] <= thr]
    hi += [r[2] for r in sub if r[0] > thr]
chk("within-mechanism split, low-psi median closure (%)", 12.8, 100 * np.median(lo), tol=.05)
chk("within-mechanism split, high-psi median closure (%)", -10.7, 100 * np.median(hi), tol=.05)
chk("within-mechanism split, Mann-Whitney p x10000", 4.0,
    stats.mannwhitneyu(lo, hi, alternative="greater").pvalue * 1e4, tol=.3)
# the BRITS-vs-SAITS pairing: at nearly matched imputation accuracy, psi should order
# the pair the way closure does
idx = {}
for k, v in S51.items():
    ds, m, rate, fill = k.split("|")
    nat, du = S42.get(f"{k}|native"), S42.get(f"{k}|dual")
    if nat is None or du is None or nat["median"] - 1.0 <= 0.02:
        continue
    idx[(ds, m, rate, fill)] = (v["psi_median"], v["rmse_median"],
                                (nat["median"] - du["median"]) / (nat["median"] - 1.0))
pairs, agree, dpsi, drmse = 0, 0, [], []
for (ds, m, rate, fill), b in idx.items():
    if not fill.startswith("brits"):
        continue
    sa = idx.get((ds, m, rate, fill.replace("brits", "saits")))
    if sa is None:
        continue
    pairs += 1
    agree += (b[0] > sa[0]) == (b[2] < sa[2])
    dpsi.append(b[0] - sa[0])
    drmse.append(b[1] - sa[1])
chk("matched BRITS-vs-SAITS cells", 44, pairs, tol=0)
chk("psi orders the pair correctly", 30, agree, tol=0)
chk("binomial one-sided p x1000", 11.3,
    stats.binomtest(agree, pairs, 0.5, alternative="greater").pvalue * 1000, tol=.1)
chk("median BRITS-minus-SAITS psi gap", 0.026, float(np.median(dpsi)), tol=.05)
chk("median BRITS-minus-SAITS imputation-RMSE gap", 0.008, float(np.median(drmse)), tol=.1)

print("\n== Section 6.2: the Chronos-2 retrofit (S60) ==")
S60 = os.path.join("s60_c2retrofit")
PR60 = J(S60, "s60_probe.json")["agg"]
SU60 = J(S60, "s60_summary.json")
LB60 = J(S60, "s60_leaderboard.json")["grid"]
AT60 = J(S60, "s60_attack.json")["cells"]
G60 = J(S60, "s60_grid.json")["grid"]
DS9 = ["ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness"]
MECHS4 = ["mcar", "block", "mnar_high", "mnar_extreme"]

chk("c2 stock declared rho ~ 0 (x1e6)", 5.4, PR60["stock|declared"]["rho_median"] * 1e6,
    tol=.1)
SE60 = J("s60_c2retrofit", "s60_seeds.json")["seeds"]
_seeds60 = sorted(SE60)
def _m60(metric, arm):
    return float(np.mean([SE60[s][metric][arm] for s in _seeds60]))

chk("c2 CPT dual rho (3-seed mean)", 0.825, _m60("rho", "CPT"), tol=.02)
chk("c2 retrofit (W50) dual rho (3-seed mean)", 0.880, _m60("rho", "W50"), tol=.02)
chk("c2 CPT closure at alpha=1 (%, 3-seed mean)", 80.8,
    _m60("closure_at_1", "CPT") * 100, tol=.02)
chk("c2 W50 closure at alpha=1 (%, 3-seed mean)", 55.5,
    _m60("closure_at_1", "W50") * 100, tol=.02)
chk("c2 CPT floor range (65-67 of 72)", 66,
    float(np.median([SE60[s]["floor_at_0"]["CPT"]["count"] for s in _seeds60])), tol=0)
chk("c2 W50 floor range (59-61 of 72)", 60,
    float(np.median([SE60[s]["floor_at_0"]["W50"]["count"] for s in _seeds60])), tol=0)
chk("c2 CPT clean ratio (3-seed mean)", 1.043, _m60("clean", "CPT"), tol=.01)
chk("c2 W50 clean ratio (parity, 3-seed mean)", 1.007, _m60("clean", "W50"), tol=.01)


def _lb(arm, fill):
    vals = [LB60[f"{d}|{m}|0.7|{fill}|{arm}"]["median"] for d in DS9 for m in MECHS4]
    return float(np.median(vals)), float(max(vals))


# the paper's Table 3 aggregates per-dataset (median of 4 mech medians) then over datasets;
# the s64 run reproduces that aggregation for all families (strict fp32)
R64 = J("s64_bench3", "s64_leaderboard_rows.json")["rows"]
for arm, exp in (("c2-stock", (1.54, 1.52, 2.10, 6.0)),
                 ("c2-w50", (1.29, 1.44, 1.66, 8.9))):
    r = R64[arm]
    chk(f"c2 leaderboard {arm} perfect (per-dataset)", exp[0], r["perfect"], tol=.02)
    chk(f"c2 leaderboard {arm} linear (per-dataset)", exp[1], r["linear"], tol=.02)
    chk(f"c2 leaderboard {arm} zero (per-dataset)", exp[2], r["zero"], tol=.02)
    chk(f"c2 leaderboard {arm} worst zero cell", exp[3],
        r["worst_zero_cell"]["value"], tol=.02)
for arm, exp in (("m2-stock", (1.18, 1.60, 11.56, 82.2)),
                 ("m2-w50", (1.11, 1.49, 2.96, 21.7))):
    r = R64[arm]
    chk(f"m2 leaderboard {arm} perfect (per-dataset)", exp[0], r["perfect"], tol=.02)
    chk(f"m2 leaderboard {arm} linear (per-dataset)", exp[1], r["linear"], tol=.02)
    chk(f"m2 leaderboard {arm} zero (per-dataset)", exp[2], r["zero"], tol=.02)
    chk(f"m2 leaderboard {arm} worst zero cell", exp[3],
        r["worst_zero_cell"]["value"], tol=.02)

G64 = J("s64_bench3", "s64_gift_report.json")["families"]
for fam, deltas in (("chronos2", (-0.4, -0.02, -5.0, -16.1, -17.9)),
                    ("moirai2", (1.3, -2.1, -4.6, -14.1, -7.4))):
    for row, dexp in zip(G64[fam], deltas):
        chk(f"gift dose {fam} {row['stratum']} delta (%)", dexp, row["delta_pct"],
            tol=max(.15, .03 / max(abs(dexp), 1e-9)))

chk("c2 attack, stock plain (median)", 1.40,
    AT60["stock"]["plain|untargeted"]["damage_ratio_median"], tol=.02)
chk("c2 attack, CPT dual (median)", 1.08,
    AT60["CPT"]["dual|untargeted"]["damage_ratio_median"], tol=.02)
chk("c2 attack, W50 dual (median)", 1.11,
    AT60["W50"]["dual|untargeted"]["damage_ratio_median"], tol=.02)

print("\n== Appendix G.4: the C2 blend-only ablation (S63) ==")
SU63 = J("s63_c2endpoint", "s63_summary.json")
LB63 = J("s63_c2endpoint", "s63_leaderboard.json")["grid"]
_blend_zero = float(np.median([LB63[f"{d}|{m}|0.7|zero|BLEND"]["median"]
                               for d in DS9 for m in MECHS4]))
_blend_linear = float(np.median([LB63[f"{d}|{m}|0.7|linear|BLEND"]["median"]
                                 for d in DS9 for m in MECHS4]))
_stock_zero = float(np.median([LB60[f"{d}|{m}|0.7|zero|stock"]["median"]
                               for d in DS9 for m in MECHS4]))
chk("c2 blend-only zero-fill median (collapse)", 20.13, _blend_zero, tol=.02)
chk("c2 blend-only zero / stock zero", 9.9, _blend_zero / _stock_zero, tol=.02)
chk("c2 blend-only linear median (healthy)", 1.24, _blend_linear, tol=.02)
chk("c2 blend-only closure at alpha=1 (%)", 84.6,
    SU63["BLEND"]["closure_median"]["1.0"] * 100, tol=.02)
chk("c2 blend-only clean ratio", 1.020, SU63["BLEND"]["clean_ratio_median"], tol=.01)

print("\n== Section 6.2: the fair-budget control on three families (S65) ==")
SU65 = J("s65_fairbudget", "s65_summary.json")
LB65C = J("s65_fairbudget", "s65_c2_leaderboard.json")["grid"]
LB65M = J("s65_fairbudget", "s65_m2_leaderboard.json")["grid"]
_c2ctl_zero = float(np.median([LB65C[f"{d}|{m}|0.7|zero|C2CTL"]["median"]
                               for d in DS9 for m in MECHS4]))
_m2ctl_zero = float(np.median([LB65M[f"{d}|{m}|0.7|zero|M2CTL"]["median"]
                               for d in DS9 for m in MECHS4]))
chk("c2 control closure at alpha=1 (mechanical) (%)", 70.0,
    SU65["c2"]["scorecard"]["C2CTL"]["closure_median"]["1.0"] * 100, tol=.02)
chk("c2 control closure at alpha=0 (%)", 15.4,
    SU65["c2"]["scorecard"]["C2CTL"]["closure_median"]["0.0"] * 100, tol=.05)
chk("c2 control floor (cells of 72)", 51,
    SU65["c2"]["scorecard"]["C2CTL"]["floor_at_alpha0"]["count"], tol=0)
chk("c2 control clean ratio", 1.052,
    SU65["c2"]["scorecard"]["C2CTL"]["clean_ratio_median"], tol=.01)
chk("c2 control zero-fill median (collapse)", 10.78, _c2ctl_zero, tol=.02)
chk("m2 control closure at alpha=1 (%)", 13.0,
    SU65["m2"]["scorecard"]["M2CTL"]["closure_median"]["1.0"] * 100, tol=.05)
chk("m2 control floor (cells of 72)", 48,
    SU65["m2"]["scorecard"]["M2CTL"]["floor_at_alpha0"]["count"], tol=0)
chk("m2 control clean ratio", 1.025,
    SU65["m2"]["scorecard"]["M2CTL"]["clean_ratio_median"], tol=.01)
chk("m2 control zero-fill median (inert)", 8.64, _m2ctl_zero, tol=.02)

print("\n== Section 6.2: the Moirai-2.0 retrofit (S61) ==")
S61 = os.path.join("s61_m2retrofit")
PR61 = J(S61, "s61_probe.json")["agg"]
SU61 = J(S61, "s61_summary.json")
AT61 = J(S61, "s61_attack.json")["cells"]

chk("m2 stock declared rho (census reproduction)", 0.949,
    PR61["stock|declared"]["rho_median"], tol=.01)
SE61 = J(S61, "s61_seeds.json")["seeds"]
_seeds61 = sorted(SE61)
def _m61(metric, arm):
    import numpy as _np
    return float(_np.mean([SE61[s][metric][arm] for s in _seeds61]))

chk("m2 CPT declared rho (3-seed mean)", 0.856, _m61("rho", "CPT"), tol=.02)
chk("m2 retrofit (W50) declared rho (3-seed mean)", 0.895, _m61("rho", "W50"), tol=.02)
chk("m2 stock sweep endpoint alpha0", 1.500,
    SU61["sweep_endpoints"]["stock|declared"]["alpha0_median"], tol=.01)
chk("m2 stock sweep endpoint alpha1", 1.141,
    SU61["sweep_endpoints"]["stock|declared"]["alpha1_median"], tol=.01)
chk("m2 CPT closure at alpha=1 (%, 3-seed mean)", 36.3,
    _m61("closure_at_1", "CPT") * 100, tol=.02)
chk("m2 W50 closure at alpha=1 (%, 3-seed mean)", 54.3,
    _m61("closure_at_1", "W50") * 100, tol=.02)
chk("m2 CPT floor (cells of 72)", 65, SU61["CPT"]["floor_at_alpha0"]["count"], tol=0)
chk("m2 W50 floor (cells of 72)", 66, SU61["W50"]["floor_at_alpha0"]["count"], tol=0)
chk("m2 CPT clean ratio (3-seed mean)", 1.033, _m61("clean", "CPT"), tol=.01)
chk("m2 W50 clean ratio (parity, 3-seed mean)", 1.016, _m61("clean", "W50"), tol=.01)
chk("m2 leaderboard stock zero", 8.97,
    SU61["leaderboard_medians"]["stock|zero"]["median_relMSE"], tol=.01)
chk("m2 leaderboard W50 zero", 2.80,
    SU61["leaderboard_medians"]["W50|zero"]["median_relMSE"], tol=.01)
chk("m2 leaderboard stock linear", 1.68,
    SU61["leaderboard_medians"]["stock|linear"]["median_relMSE"], tol=.01)
chk("m2 leaderboard W50 linear", 1.50,
    SU61["leaderboard_medians"]["W50|linear"]["median_relMSE"], tol=.01)
chk("m2 bad-fill, stock (median)", 1.41,
    AT61["stock"]["declared|untargeted"]["damage_ratio_median"], tol=.02)
chk("m2 bad-fill, W50 (median)", 1.30,
    AT61["W50"]["declared|untargeted"]["damage_ratio_median"], tol=.02)

print("\n== Appendix E: the field on GIFT-Eval's missing strata (S66) ==")
S66 = J("s66_field", "s66_gift_rows.json")
_ov = S66["stock_overall_linear"]
_mo = S66["models"]
chk("gift overall bolt stock", 0.782, _ov["bolt-base"], tol=.01)
chk("gift overall c2 stock", 0.761, _ov["chronos2"], tol=.01)
chk("gift overall m2 stock", 0.781, _ov["moirai2"], tol=.01)
chk("gift overall FlowState (7% above bolt stock)", 0.839,
    _mo["flowstate"]["overall_median_by_fill"]["linear"], tol=.01)
chk("gift overall TiRex (5% below bolt stock)", 0.742,
    _mo["tirex"]["overall_median_by_fill"]["linear"], tol=.01)
chk("gift overall TiRex declared path (dose-insensitive)", 0.732,
    _mo["tirex"]["overall_median_by_fill"]["nan"], tol=.01)
chk("gift overall TimesFM (1% above bolt stock)", 0.793,
    _mo["timesfm"]["overall_median_by_fill"]["linear"], tol=.01)
_tirex_nan_strata = [row["median_by_fill"]["nan"] for row in _mo["tirex"]["strata"]]
chk("TiRex declared >30% stratum", 0.928, _tirex_nan_strata[-1], tol=.02)

print("\n== The census counts the prose quotes ==")
import re as _re
CT = open(os.path.join(HERE, "..", "paper", "iclr2026", "figures", "counts.tex")).read()
words = {"seventeen": 17, "ten": 10, "twenty": 20, "eighteen": 18, "eleven": 11}
got = {m.group(1): words.get(m.group(2), -1)
       for m in _re.finditer(r"\\newcommand\{\\(\w+)\}\{([\w-]+)\}", CT)}
print("\n== The census: TempoPFN (S62) ==")
PR62 = J("s62_tempopfn", "s62_results.json")["probe"]
_cells62 = lambda conv: [v for k, v in PR62.items() if k.endswith("|" + conv)]
import numpy as _np62
_plain62 = [v["ratio"] for v in _cells62("plain")]
_decl62 = [v["ratio"] for v in _cells62("nan")]
chk("TempoPFN rho plain (median over 9 cells)", 0.84, float(_np62.median(_plain62)),
    tol=.02)
chk("TempoPFN rho declared == 0 on all 9 cells", 0.0,
    float(max(abs(x) for x in _decl62)), tol=0)
chk("TempoPFN declared perm term == 0 everywhere", 0.0,
    float(max(v["perm_rms_rel"] for v in _cells62("nan"))), tol=0)
chk("TempoPFN declared redraw term == 0 everywhere", 0.0,
    float(max(v["redraw_rms_rel"] for v in _cells62("nan"))), tol=0)
SW62 = J("s62_tempopfn", "s62_results.json")["sweep"]
_groups62 = {}
_plain_oracle = []
for k, v in SW62.items():
    if not isinstance(v, dict) or "relMSE_median" not in v:
        continue
    parts = k.split("|")          # tempopfn-38m|ds|mech|rate|path|a=X
    if len(parts) != 6:
        continue
    _, ds, mech, rate, path, al = parts
    if path == "nan":
        _groups62.setdefault((ds, mech, rate), []).append(v["relMSE_median"])
    if path == "plain" and al == "a=1.0":
        _plain_oracle.append(v["relMSE_median"])
_flat_spread = max((max(v) - min(v)) for v in _groups62.values())
chk("TempoPFN declared sweep flat: max spread over alpha", 0.0, float(_flat_spread),
    tol=0)
if _plain_oracle:
    chk("TempoPFN plain at alpha=1 == clean: max |relMSE - 1|", 0.0,
        float(max(abs(x - 1.0) for x in _plain_oracle)), tol=0)

chk("nummodels (checkpoints with a rho)", 18, got.get("nummodels"), tol=0)
chk("numfamilies (families with a rho)", 11, got.get("numfamilies"), tol=0)
chk("numallmodels (rows in the census table)", 18, got.get("numallmodels"), tol=0)

print(f"\n==== {ok} ok, {bad} bad ====")

json.dump(sorted(set(CLAIMS)), open(os.path.join(HERE, "claims_main_text.json"), "w"))
