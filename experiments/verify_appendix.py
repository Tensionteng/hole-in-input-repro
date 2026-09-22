#!/usr/bin/env python
"""Traces every numeric claim in the appendices back to the run that produced it.

Companion to verify_main_text.py. Where a value survives only in a round's log rather than in
its results JSON, the log is parsed rather than trusted.
"""
import glob, json, os, re, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda *p: json.load(open(os.path.join(HERE, *p)))
ok = bad = 0
CLAIMS = []   # every value this file asserts, for audit_prose_numbers.py


def chk(label, claim, got, tol=0.02):
    global ok, bad
    CLAIMS.append(float(claim))
    good = got is not None and abs(got - claim) <= tol * max(abs(claim), 1e-9) + 1e-12
    print(f"  {'ok ' if good else 'BAD'} {label:62s} claim {claim:>9} got "
          + (f"{got:>9.4g}" if got is not None else f"{'n/a':>9}"))
    ok += good; bad += (not good)


print("== C.1 the grid: what a zero fill costs ==")
S5 = J("s05_characterization", "s5_missing_results.json")


def zero_rel(model):
    v = []
    for ds in S5[model]:
        c = S5[model][ds]
        if "mcar:zero:0.7" in c and "clean:none:0.0" in c:
            v.append(c["mcar:zero:0.7"]["mse"] / c["clean:none:0.0"]["mse"])
    return float(np.mean(v)) if v else None


chk("zero fill at 70% scattered, bolt", 13.7, zero_rel("bolt"))
chk("zero fill at 70% scattered, timesfm", 19.0, zero_rel("timesfm"))

print("\n== C.3 the sweep, zero-shot and adapted ==")
SW27 = J("s27_interface", "s27_results.json")["sweep"]
DS3 = ("ETTh1", "ETTm1", "weather")
zs = lambda i: float(np.mean([SW27[f"zeroshot|{d}|mcar|0.7|0.0|{i}"]["median"] for d in DS3]))
chk("zero-shot scattered 0.7, declared at alpha=0", 1.21, zs("native"))
chk("zero-shot scattered 0.7, restored at alpha=0", 1.32,
    min(zs("dual"), zs("dual_obsnorm")))

S37 = {}
for f in ("own_insample", "own_held_a", "own_held_b", "own_held_c"):
    p = os.path.join(HERE, "s37_capbreadth", f + ".json")
    if os.path.exists(p):
        S37.update(json.load(open(p))["sweep"])
DS9 = ["ETTh1", "ETTm1", "weather", "ETTh2", "ETTm2", "electricity", "traffic",
       "exchange", "illness"]
ad = lambda m, a, i: float(np.median([S37[f"{d}|{m}|0.7|{a}|{i}"]["median_own"] for d in DS9
                                      if f"{d}|{m}|0.7|{a}|{i}" in S37]))
chk("adapted scattered 0.7, declared at alpha=0", 1.150, ad("mcar", 0.0, "native"))
chk("adapted scattered 0.7, restored at alpha=0", 1.148,
    min(ad("mcar", 0.0, "dual"), ad("mcar", 0.0, "dual_obsnorm")))
chk("adapted block 0.7, declared at alpha=0", 1.118, ad("block", 0.0, "native"))
chk("adapted block 0.7, restored at alpha=0", 1.129,
    min(ad("block", 0.0, "dual"), ad("block", 0.0, "dual_obsnorm")))

print("\n== C.4 the attack, targeted variant ==")
M = J("s29_audit", "s29_results.json")["armB"]["metr|miss"]
for k, c in (("plain|targeted", 2.43), ("native|targeted", 1.04), ("nan|targeted", 1.00)):
    chk(f"METR-LA {k}", c, M[k]["damage_ratio_median"], tol=.01)

print("\n== C.5 cross-channel, the declared column ==")
C31 = J("s31_crosschannel", "s31_results.json")["cells"]
xc = lambda m, f: float(np.median([c[f] for k, c in C31.items()
                                   if k.split("|")[2] == m and k.split("|")[3] == "0.7"]))
chk("scattered 0.7, declared + neighbours", 1.096, xc("mcar", "nan_multi"))
chk("censoring 0.7, declared + neighbours", 3.503, xc("mnar_high", "nan_multi"))

print("\n== C.6 migration: synthetic augmentation ==")
A26 = J("s26_realfill", "s26_results.json")["arms"]
for arm, med, win in (("penn|cens|aug0|c-zero", 1.499, 62), ("penn|cens|aug6|c-zero", 2.163, 45)):
    a = A26[arm]
    base = np.array(a["fixed_zero"]["per_window"])
    x = np.array(a["learned_plain_fill"]["per_window"])
    m = base > 0
    chk(f"{arm.split('|')[2]}: test median", med, float(np.median(x)))
    chk(f"{arm.split('|')[2]}: win rate (%)", win, 100 * float((x[m] / base[m] < 1).mean()),
        tol=.02)

print("\n== C.7 patch size ==")
S34 = J("s34_patchsize", "s34_results.json")
C34, DS34 = S34["cells"], sorted({k.split("|")[0] for k in S34["cells"]})
CL34 = S34["clean"]
gaps, worse_rel, worse_abs = [], 0, 0
for m in ("mcar", "block", "mnar_high", "mnar_extreme"):
    for r in ("0.1", "0.3", "0.5", "0.7"):
        f = lambda p: np.mean([C34[f"{d}|{m}|{r}|{p}"]["rel_median"] for d in DS34])
        a = lambda p: np.mean([CL34[f"{d}|{p}"]["mse_median"] * C34[f"{d}|{m}|{r}|{p}"]
                               ["rel_median"] for d in DS34])
        gaps.append(100 * (f(64) - f(16)) / f(16))
        worse_rel += f(16) < f(64)
        worse_abs += a(16) > a(64)
chk("mechanism-by-rate cells", 16, len(gaps), tol=0)
chk("patch 16 looks more robust in, cells", 15, worse_rel, tol=0)
chk("patch 16 apparent robustness gain, median (%)", 12, float(np.median(gaps)), tol=.05)
chk("patch 16 apparent robustness gain, max (%)", 83, max(gaps), tol=.02)
chk("patch 16 worse in absolute terms in, cells", 15, worse_abs, tol=0)

print("\n== E the mechanism detector, and F real GIFT-Eval ==")
S39 = J("s39_realcalib", "s39_results.json")
st = S39["strata"]
ks = list(st)
chk("cleanest stratum, windows", 1569, st[ks[0]]["n"], tol=0)
chk("coverage drop across strata", 0.089,
    st[ks[0]]["native"]["coverage"] - st[ks[-1]]["native"]["coverage"], tol=.02)
chk("overall coverage under a single pool", 0.899, S39["overall"]["naive"]["coverage"])
bd = [v["mondrian"]["coverage"] for v in S39["by_detected"].values()]
chk("grouped by detected class, min", 0.868, min(bd))
chk("grouped by detected class, max", 0.916, max(bd))
for cls, c in (("clean", 2184), ("mcar", 796), ("mnar_extreme", 614), ("mnar_high", 559),
               ("intermittent", 421), ("block", 82)):
    chk(f"detector fires {cls}", c, S39["detected"][cls], tol=0)

print("\n== G attribution: probes and the opened mask ==")
P11 = J("s11_probe", "s11_results.json")["probe"]


def fam_peak(cell, fam):
    v = []
    for d in P11:
        fv = P11[d].get(cell, {}).get("families", {}).get(fam)
        if fv:
            r = np.asarray(fv["r2_mean"], float)
            if np.isfinite(r).any():
                v.append(np.nanmax(r))
    return float(np.mean(v)) if v else None


obs = [fam_peak(c, "obs_in_miss") for c in ("block:0.7", "mnar_high:0.7")]
chk("observed values inside a holed window, min R2", 0.85, min(obs), tol=.02)
chk("observed values inside a holed window, max R2", 0.95, max(obs), tol=.02)
fm = [fam_peak(c, "full_miss") for c in ("mcar:0.7", "block:0.7")]
chk("fully-missing patches under scattered/block, min R2", 0.55, min(fm), tol=.02)
chk("fully-missing patches under scattered/block, max R2", 0.67, max(fm), tol=.02)
chk("fully-missing patches under censoring, R2", 0.12, fam_peak("mnar_high:0.7", "full_miss"),
    tol=.05)

T12 = J("s12_recon_sft", "s12_results.json")["tables"]
pr = T12["probe"]["ETTh1:block:0.7"]
chk("probe R2, masked-block SFT (ETTh1 block 0.7)", 0.478, pr["mb"])
chk("probe R2, reconstruction-aware SFT", 0.487, pr["recon"])
mn = T12["main"]["block:0.7"]
chk("forecast relMSE, masked-block SFT", 0.800, mn["mb:nan"]["avg"])
chk("forecast relMSE, reconstruction-aware SFT", 0.794, mn["recon:nan"]["avg"])

S19 = J("s19_mnarunmask", "s19_results.json")
chk("knockout under the native decode mask, max |delta|", 0.0,
    max(v["mb_ko_vs_mb_max_window_diff"] for v in S19["controls"].values()), tol=0)
log = open(os.path.join(HERE, "s19_mnarunmask", "s19_smoke.log")).read()
m = re.search(r"unmask\+knockout vs unmask max\|diff\| = ([\d.]+)", log)
chk("knockout once the decode mask is opened, max |delta|", 7.89,
    float(m.group(1)) if m else None)

nb = open(os.path.join(HERE, "s15_unmask", "s15_notes.md")).read()
for ds, c in (("ETTh1", 5.67), ("ETTm1", 6.91)):
    mm = re.search(ds + r" \*\*\+([\d.]+)%", nb)
    chk(f"opening the mask, gain on {ds} (%)", c, float(mm.group(1)) if mm else None)

print("\n== other appendix figures ==")
# the most missing-valued GIFT-Eval dataset, re-read from the arrow files
GIFT = os.path.abspath(os.path.join(HERE, "..", "data", "gifteval"))
if os.path.isdir(GIFT):
    import pyarrow as pa
    tot = nn = 0
    for f in sorted(glob.glob(os.path.join(GIFT, "kdd_cup_2018_with_missing/H", "*.arrow"))):
        with pa.memory_map(f, "rb") as src:
            t = pa.ipc.open_stream(src).read_all()
        for row in t.column("target").to_pylist():
            a = np.asarray(row, dtype=np.float64)
            tot += a.size
            nn += int(np.isnan(a).sum())
    chk("kdd_cup, NaN share of the raw series (%)", 17.1, 100 * nn / tot if tot else None)
# 12.15% is the context-NaN share of the WINDOWS the audit sampled, which is what the
# appendix quotes; it is re-derived here with S32's reader rather than taken from the log.
sys.path.insert(0, os.path.join(HERE, "s32_gifteval"))
import run_s32_gifteval as s32
_w = s32.windows(s32.read_series("kdd_cup_2018_with_missing/H", max_series=200))
chk("kdd_cup, NaN share of sampled context (%)", 12.15,
    100 * float((~np.isfinite(_w[0])).mean()) if _w else None)

# the TimeMoE number we discard: quoted in the appendix precisely because it did not reproduce
log36 = open(os.path.join(HERE, "s36_models", "log_4.log")).read()
m36 = re.search(r"AGG TimeMoE-50M: plain=([\d.]+)", log36)
chk("the TimeMoE rho we discard", 0.65, float(m36.group(1)) if m36 else None, tol=.01)

# the MSE closure median the MAE comparison is stated against
S41 = {}
for f in sorted(glob.glob(os.path.join(HERE, "s41_mae", "part_*.json"))):
    S41.update(json.load(open(f)).get("closure", {}))
INS = ("ETTh1", "ETTm1", "weather")
ins = [c["mse"]["closure"] for k, c in S41.items()
       if k.split("|")[0] in INS and c["mse"]["native_a1"] - 1 >= 0.02
       and np.isfinite(c["mse"]["closure"])]
chk("in-sample closure median under MSE (%)", 53.1, 100 * float(np.median(ins)), tol=.01)

json.dump(sorted(set(CLAIMS)), open(os.path.join(HERE, "claims_appendix.json"), "w"))
print(f"\n==== {ok} ok, {bad} bad ====")
