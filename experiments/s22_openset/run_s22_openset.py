#!/usr/bin/env python
"""S22: open-set fallback for the S16 MechGate mechanism detector.

S16's detector is closed-world: every window is hard-classified into
{clean, mcar, block, mnar_high, mnar_extreme, intermittent}. S22 adds an
open-set fallback: windows that match NO trained mechanism are flagged
"unknown" and routed to a safe default (linear fill + widest conformal
width) instead of being hard-routed.

Scenarios (pre-registered in s22_notes.md):
  holdout_mnar_extreme / holdout_block: retrain gbdt on the S16 training
    features with that class removed; the held-out class is the unknown.
  novel: keep the S16 6-class classifier; unknowns are two new mechanisms
    never trained -- iburst (pointwise missingness with rate drifting
    linearly over the window) and mnar_low (low-value rank censoring).

Detection scores (window level):
  maxprob: max of the channel-mean posterior (S16 vote); reject if < tau.
  maha:    mean over channels of the Mahalanobis distance to the nearest
           known-class prototype (Ledoit-Wolf pooled covariance, feature
           column 12 excluded -- it is constant 0 in training features);
           reject if > tau.
tau calibrated per scenario on 300 fresh TRAIN-region windows at known-class
false-rejection rate 2%.

Evaluation: anchor gate (3 S16 syn_eval cells recomputed from s16_ckpt,
±5% + >=99% label agreement); detection-rate-vs-FRR curves; end-to-end
relMSE on held-out mechanisms (arms fixed_linear / hard_gated /
openset_gated / openset_maha / oracle under policy tables T_safe and
T_danger); coverage on an ETTh1 cal/test stream (T_danger routing);
real-data probe on Penmanshiel + METR-LA.

Modes (comma list): smoke, gate, calib, eval, figure, all.
New files only: run_s22_openset.py, s22_results.json, s22.png,
s22_notes.md, s22_ckpt/, s22_*.log.
"""
import argparse
import json
import os
import pickle
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import torch

import run_s5_missing as s5
import run_s6_mnarfix as s6
import run_s6_conformal as s6c
import run_s16_mechgate as s16

L, H = s5.L, s5.H                       # 512, 96
SEED = 20260814
NPROC = 24
OUT = os.path.join(HERE, "s22_results.json")
CKPT = os.path.join(HERE, "s22_ckpt")
os.makedirs(CKPT, exist_ok=True)

CLASSES = s16.CLASSES
CLEAN, MCAR, BLOCK, MNAR_H, MNAR_E, INTER = range(6)
RATES4 = [0.1, 0.3, 0.5, 0.7]
RATES2 = [0.3, 0.5]
N_EVAL = 150                            # first 150 of S16's 300 eval windows
MAHA_EXCLUDE = [12]                     # xmask_corr: constant in train feats

TRAIN_PROBS = np.array([0.20, 0.20, 0.20, 0.15, 0.15, 0.10])

SCENARIOS = {
    "holdout_mnar_extreme": dict(drop=[MNAR_E], unknown=["mnar_extreme"],
                                 rates=RATES4, clf="retrain"),
    "holdout_block": dict(drop=[BLOCK], unknown=["block"], rates=RATES4,
                          clf="retrain"),
    "novel": dict(drop=[], unknown=["iburst", "mnar_low"], rates=RATES2,
                  clf="s16"),
}
# policy tables (predicted class -> fill); unknown handled separately
T_SAFE = {MCAR: "linear", BLOCK: "linear", MNAR_H: "tail_tobit",
          MNAR_E: "tail_tobit", CLEAN: "linear", INTER: "linear"}
T_DANGER = {MCAR: "linear", BLOCK: "linear", MNAR_H: "zero",
            MNAR_E: "zero", CLEAN: "linear", INTER: "linear"}
TABLES = {"T_safe": T_SAFE, "T_danger": T_DANGER}
FILLS_E2E = ["zero", "ffill", "linear", "tail_tobit"]


# ===================================================== new mask generators ====
def mnar_low_mask(rate, x):
    """[C, L] bool: censor the ceil(rate*L) LOWEST values (mirror of S5's
    rank-based mnar_high). Deterministic."""
    C = x.shape[0]
    k = int(np.ceil(rate * L))
    m = np.zeros((C, L), bool)
    order = np.argsort(x, axis=1, kind="stable")      # ascending rank
    np.put_along_axis(m, order[:, :k], True, axis=1)
    return m


def iburst_mask(rate, wi, ms, C):
    """[C, L] bool: pointwise Bernoulli missingness with the instantaneous
    rate ramping linearly over the window: p_t = clip(rate * (0.2 +
    1.6 t/(L-1)), 0, 0.97) (ramp mean = 1, so E[rate] ~= rate; late-window
    bursts). Deterministic per (window, mask_seed, rate)."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [s5.SEED, wi, ms, int(round(rate * 100)), 7]))
    ramp = 0.2 + 1.6 * np.arange(L) / (L - 1)
    p = np.clip(rate * ramp, 0.0, 0.97)
    return rng.random((C, L)) < p[None, :]


def make_mask_s22(mech, rate, wi, ms, C, x):
    if mech == "mnar_low":
        return mnar_low_mask(rate, x)
    if mech == "iburst":
        return iburst_mask(rate, wi, ms, C)
    return s5.make_mask(mech, rate, wi, ms, C, x=x)


def mask_variants(mech, rate, wi, C, x):
    """Store convention (s5/s6): deterministic mechanisms (mnar_*) use a
    single mask; stochastic ones average over mask seeds {0,1}."""
    if mech in ("mnar_high", "mnar_extreme", "mnar_low"):
        return [make_mask_s22(mech, rate, wi, 0, C, x)]
    return [make_mask_s22(mech, rate, wi, ms, C, x) for ms in (0, 1)]


# ==================================================== multiprocessing pool ====
_X = {}


def _init_worker():
    import pandas as pd
    global _X
    for ds, path in s5.DATASETS.items():
        df = pd.read_csv(path)
        _X[ds] = df.drop(columns=["date"]).to_numpy(np.float32)


def _starts_eval(ds):
    """Replica of s5.load_windows start selection (300 test-region windows)."""
    X = _X[ds]
    N = len(X)
    test_start = int(0.8 * N)
    lo, hi = test_start - L, N - L - H
    n_valid = hi - lo + 1
    rng = np.random.default_rng(s5.SEED)
    return np.sort(rng.choice(n_valid, size=min(300, n_valid),
                              replace=False)) + lo


def _starts_cal(ds, n=150):
    """Replica of s6c.calib_starts (test region, disjoint from eval)."""
    X = _X[ds]
    N = len(X)
    test_start = int(0.8 * N)
    lo, hi = test_start - L, N - L - H
    valid = np.arange(lo, hi + 1)
    pool = np.setdiff1d(valid, _starts_eval(ds))
    rng = np.random.default_rng(s5.SEED + 999)
    return np.sort(rng.choice(pool, size=min(n, len(pool)), replace=False))


def _starts_train(ds, n, seed):
    """Replica of s16.load_train_windows (train region)."""
    X = _X[ds]
    N = len(X)
    test_start = int(0.8 * N)
    lo, hi = 0, test_start - L - H
    rng = np.random.default_rng(seed)
    k = min(n, hi - lo + 1)
    return np.sort(rng.choice(hi - lo + 1, size=k, replace=False)) + lo


def _stream_channel(scen_key, sid, wi, c, x1d):
    """One channel of a mixed-mechanism calibration/coverage stream.
    Returns (vals [L], mask [L], tlab). Mirrors s16.run_conf_syn semantics:
    labels drawn from the scenario's known-class mix; rate ~ U(0.1, 0.6);
    intermittent = sparsified zeros (no mask)."""
    mix = [cl for cl in range(6) if cl not in SCENARIOS[scen_key]["drop"]]
    probs = TRAIN_PROBS[mix] / TRAIN_PROBS[mix].sum()
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 31 + sid, wi, c]))
    tlab = int(mix[rng.choice(len(mix), p=probs)])
    rate = float(rng.uniform(0.1, 0.6))
    m = np.zeros(L, bool)
    vals = x1d
    if tlab == MCAR:
        m = s5.make_mask("mcar", rate, wi, 7000 + sid * 991 + c, 1)[0]
    elif tlab == BLOCK:
        m = s16.block_mask_1d(rate, L, rng, 24)
    elif tlab in (MNAR_H, MNAR_E):
        mech = "mnar_high" if tlab == MNAR_H else "mnar_extreme"
        m = s5.make_mask(mech, rate, wi, 0, 1, x=x1d[None, :])[0]
    elif tlab == INTER and (x1d == 0).mean() < 0.5:
        keep = rng.random(L) < float(rng.uniform(0.05, 0.5))
        vals = x1d * keep
    return vals.astype(np.float32), m, tlab


def _calib_channel(scen_key, ds, wi, c, x1d):
    """One channel of the TRAIN-region tau-calibration stream. Mirrors
    s16.train_samples semantics (incl. natural-zero relabel to intermittent
    and no-visible-damage block -> clean), labels restricted to the
    scenario's known classes (renormalized training probs)."""
    known = [cl for cl in range(6) if cl not in SCENARIOS[scen_key]["drop"]]
    probs = TRAIN_PROBS[known] / TRAIN_PROBS[known].sum()
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 7, wi, c]))
    lab = int(known[rng.choice(len(known), p=probs)])
    if (x1d == 0).mean() >= 0.5:
        lab = INTER
    rate = float(rng.uniform(0.05, 0.8))
    mask = np.zeros(L, bool)
    vals = x1d
    if lab == MCAR:
        mask = s5.make_mask("mcar", rate, wi, 1000 + c, 1)[0]
    elif lab == BLOCK:
        blen = int(rng.choice([12, 24, 48, 96]))
        mask = s16.block_mask_1d(rate, L, rng, blen)
        if mask.sum() < max(2, 0.02 * L):
            lab, mask = CLEAN, np.zeros(L, bool)
    elif lab in (MNAR_H, MNAR_E):
        mech = "mnar_high" if lab == MNAR_H else "mnar_extreme"
        mask = s5.make_mask(mech, rate, wi, 0, 1, x=x1d[None, :])[0]
    elif lab == INTER and (x1d == 0).mean() < 0.5:
        keep = rng.random(L) < float(rng.uniform(0.05, 0.5))
        vals = x1d * keep
    return vals.astype(np.float32), mask, lab


def _job(j):
    kind = j[0]
    if kind == "feat":
        _, ds, stream, wi, mech, rate = j
        X = _X[ds]
        starts = _starts_eval(ds) if stream == "eval" else _starts_cal(ds)
        s = starts[wi]
        x = X[s:s + L].T.copy()
        m = make_mask_s22(mech, rate, wi, 0, X.shape[1], x)
        return j, s16.window_features(x, m), None, None
    if kind == "gate":
        _, ds, mech, rate, wi = j
        X = _X[ds]
        s = _starts_eval(ds)[wi]
        x = X[s:s + L].T.copy()
        m = s5.make_mask(mech, rate, wi, 0, X.shape[1], x=x)
        return j, s16.window_features(x, m), None, None
    if kind == "fills":
        # per-window filled contexts for all (mask variant, fill) combos
        _, ds, mech, rate, wi, fills = j
        X = _X[ds]
        C = X.shape[1]
        s = _starts_eval(ds)[wi]
        x = X[s:s + L].T.copy()
        mvs = mask_variants(mech, rate, wi, C, x)
        out = {}
        for vi, m in enumerate(mvs):
            for f in fills:
                if f == "tail_tobit":
                    filled = np.empty_like(x)
                    for c in range(C):
                        filled[c], _ = s6.tail_impute(x[c], m[c], "tail_tobit")
                else:
                    filled = s5.fill_context(x, m, f)
                out[(vi, f)] = filled.astype(np.float32)
        gt = X[s + L:s + L + H].T.copy()
        F = s16.window_features(x, mvs[0])      # routing features at ms=0
        return j, F, out, gt
    if kind == "covcal":
        _, scen_key, wi = j
        X = _X["ETTh1"]
        s = _starts_cal("ETTh1")[wi]
        x = X[s:s + L].T.copy()
        C = X.shape[1]
        vals = np.empty((C, L), np.float32)
        m = np.zeros((C, L), bool)
        tl = np.zeros(C, np.int8)
        for c in range(C):
            vals[c], m[c], tl[c] = _stream_channel(scen_key, 0, wi, c, x[c])
        return j, s16.window_features(vals, m), (vals, m, tl), None
    if kind == "covtest":
        _, scen_key, mech, rate, wi = j
        X = _X["ETTh1"]
        s = _starts_eval("ETTh1")[wi]
        x = X[s:s + L].T.copy()
        m = make_mask_s22(mech, rate, wi, 0, X.shape[1], x)
        return j, s16.window_features(x, m), (x.astype(np.float32), m), None
    raise ValueError(kind)


def run_jobs(jobs, nproc=NPROC):
    """Run jobs in a process pool; returns {job: (F, extra, gt)}."""
    import multiprocessing as mp
    out = {}
    if not jobs:
        return out
    with mp.Pool(nproc, initializer=_init_worker) as pool:
        for j, F, extra, gt in pool.imap_unordered(_job, jobs, chunksize=4):
            out[j] = (F, extra, gt)
    return out


# =========================================================== open-set core ====
def known_classes(scen_key):
    return [c for c in range(6) if c not in SCENARIOS[scen_key]["drop"]]


def train_scenario_clf(scen_key):
    """gbdt for the scenario: retrained on the S16 training features minus
    the dropped classes, or the S16 final classifier for 'novel'."""
    path = os.path.join(CKPT, f"clf_{scen_key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    if SCENARIOS[scen_key]["clf"] == "s16":
        with open(os.path.join(s16.CKPT, "clf_final.pkl"), "rb") as fh:
            clf = pickle.load(fh)["clfs"]["gbdt"]
    else:
        z = np.load(os.path.join(s16.CKPT, "feats_train.npz"))
        F, y = z["F"], z["y"]
        keep = np.isin(y, known_classes(scen_key))
        clf = s16.fit_clf("gbdt", F[keep], y[keep])
    with open(path, "wb") as fh:
        pickle.dump(clf, fh)
    return clf


def fit_maha(scen_key):
    """Class prototypes + pooled Ledoit-Wolf precision on the scenario's
    training features (feature col 12 excluded)."""
    path = os.path.join(CKPT, f"maha_{scen_key}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["mu"], z["prec"]
    from sklearn.covariance import LedoitWolf
    z = np.load(os.path.join(s16.CKPT, "feats_train.npz"))
    F, y = z["F"], z["y"]
    dims = [i for i in range(F.shape[1]) if i not in MAHA_EXCLUDE]
    kc = known_classes(scen_key)
    keep = np.isin(y, kc)
    F, y = F[keep][:, dims], y[keep]
    mus = np.stack([F[y == c].mean(axis=0) for c in kc])
    centered = F - mus[[kc.index(c) for c in y]]
    lw = LedoitWolf().fit(centered)
    prec = lw.precision_
    np.savez(path, mu=mus, prec=prec)
    return mus, prec


def window_scores(clf, mus, prec, F):
    """F [C,14] -> (pred_label, maxprob_score, maha_score)."""
    p = clf.predict_proba(F).mean(axis=0)          # S16 mean-proba vote
    pred = int(clf.classes_[np.argmax(p)])
    dims = [i for i in range(F.shape[1]) if i not in MAHA_EXCLUDE]
    D = F[:, dims][:, None, :] - mus[None, :, :]   # [C, K, d]
    d2 = np.einsum("ckd,de,cke->ck", D, prec, D)
    maha = float(np.sqrt(np.maximum(d2, 0.0)).min(axis=1).mean())
    return pred, float(p.max()), maha


def apply_scores(clf, mus, prec, feats):
    """feats: {wi: [C,14]} -> per-window (pred, maxprob, maha) arrays."""
    preds, mps, mhs = [], [], []
    for wi in sorted(feats):
        p, mp, mh = window_scores(clf, mus, prec, feats[wi])
        preds.append(p)
        mps.append(mp)
        mhs.append(mh)
    return (np.array(preds), np.array(mps), np.array(mhs))


# ------------------------------------------------------------------ smoke ----
def run_smoke(device):
    print("[smoke] bolt load ...", flush=True)
    model = s16.Bolt96(device)
    # starts replication must match the original loaders exactly
    _init_worker()
    for ds in s5.DATASETS:
        _, st = s5.load_windows(s5.DATASETS[ds], 300, s5.SEED)
        assert np.array_equal(st, _starts_eval(ds)), f"eval starts {ds}"
        _, sc = s6c.calib_starts(ds, 150)
        assert np.array_equal(sc, _starts_cal(ds)), f"cal starts {ds}"
    print("[smoke] starts replication OK (eval + cal, 3 datasets)", flush=True)
    # new mask generators
    X = _X["ETTh1"]
    s = _starts_eval("ETTh1")[0]
    x = X[s:s + L].T.copy()
    m_lo = mnar_low_mask(0.5, x)
    fd = dict(zip(s16.FEAT_NAMES, s16.channel_features(x[0], m_lo[0])))
    print(f"[smoke] mnar_low p=0.5: r_bar={fd['r_bar']:.3f} "
          f"miss={fd['miss_rate']:.3f}", flush=True)
    assert fd["r_bar"] < 0.45, "mnar_low should delete LOW ranks"
    m_ib = iburst_mask(0.4, 0, 0, x.shape[0])
    half = L // 2
    e, l_ = m_ib[0, :half].mean(), m_ib[0, half:].mean()
    print(f"[smoke] iburst p=0.4: miss={m_ib.mean():.3f} "
          f"early={e:.3f} late={l_:.3f}", flush=True)
    assert abs(m_ib.mean() - 0.4) < 0.08 and l_ > 1.8 * max(e, 1e-6)
    f_ib = dict(zip(s16.FEAT_NAMES, s16.channel_features(x[1], m_ib[1])))
    assert f_ib["mean_run_rel"] * L < 24, "iburst should look pointwise"
    # scenario classifier + maha on a small subset
    z = np.load(os.path.join(s16.CKPT, "feats_train.npz"))
    F, y = z["F"][:4000], z["y"][:4000]
    keep = np.isin(y, known_classes("holdout_block"))
    clf = s16.fit_clf("gbdt", F[keep], y[keep])
    mus, prec = fit_maha("holdout_block")
    p, mp, mh = window_scores(clf, mus, prec, F[:5])
    print(f"[smoke] scores ok: pred={p} maxprob={mp:.3f} maha={mh:.2f}",
          flush=True)
    # tail_impute on new mechanisms
    f1, m1 = s6.tail_impute(x[0], m_lo[0], "tail_tobit")
    assert np.isfinite(f1).all()
    # bolt forward on a filled iburst context
    ctx = s5.fill_context(x, m_ib, "linear").astype(np.float32)
    q10, q90 = model.predict_quantiles(ctx)
    assert q10.shape == (x.shape[0], H) and (q90 > q10).mean() > 0.99
    del model
    torch.cuda.empty_cache()
    print("[smoke OK]", flush=True)


# ----------------------------------------------------------------- anchor ----
GATE_CELLS = [("ETTh1", "mnar_extreme", 0.7),
              ("ETTm1", "block", 0.5),
              ("weather", "mnar_high", 0.3)]


def run_gate(res):
    """Recompute 3 S16 syn_eval cells with s16_ckpt/clf_final.pkl; acc within
    ±5% of stored AND >=99% label agreement with cached labels."""
    with open(os.path.join(s16.CKPT, "clf_final.pkl"), "rb") as fh:
        clf = pickle.load(fh)["clfs"]["gbdt"]
    cached = np.load(os.path.join(s16.CKPT, "labels_syn_gbdt.npz"))
    d16 = json.load(open(os.path.join(HERE, "s16_results.json")))
    stored_acc = d16["trackA"]["syn_eval_acc"]
    cells = []
    for ds, mech, rate in GATE_CELLS:
        t0 = time.time()
        jobs = [("gate", ds, mech, rate, wi) for wi in range(300)]
        r = run_jobs(jobs)
        wl = []
        for wi in range(300):
            F = r[("gate", ds, mech, rate, wi)][0]
            p = clf.predict_proba(F).mean(axis=0)
            wl.append(int(clf.classes_[np.argmax(p)]))
        wl = np.array(wl, np.int8)
        true = CLASSES.index(mech)
        acc = float((wl == true).mean())
        key = f"B|{ds}|{mech}|{rate}"
        ref = cached[key].astype(int)
        agree = float((wl == ref).mean())
        st = stored_acc[key]["acc"]
        ok = abs(acc - st) <= 0.05 and agree >= 0.99
        cells.append(dict(cell=key, stored_acc=float(st), recomputed_acc=acc,
                          label_agreement=agree, passed=bool(ok)))
        print(f"  [gate] {key}: stored={st:.4f} rerun={acc:.4f} "
              f"agree={agree:.4f} {'OK' if ok else 'FAIL'} "
              f"({time.time()-t0:.0f}s)", flush=True)
    passed = all(c["passed"] for c in cells)
    print(f"GATE {'PASS' if passed else 'FAIL'}", flush=True)
    res["gate"] = dict(passed=bool(passed), cells=cells)
    save_results(res)
    return passed


# ------------------------------------------------------------------- calib ----
def run_calib(res):
    """Train scenario classifiers, fit Mahalanobis prototypes, calibrate tau
    at known-class FRR=2% on 300 fresh train-region windows per scenario."""
    out = {}
    for scen in SCENARIOS:
        t0 = time.time()
        clf = train_scenario_clf(scen)
        mus, prec = fit_maha(scen)
        cache = os.path.join(CKPT, f"calib_{scen}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            mp, mh = z["mp"], z["mh"]
        else:
            mps, mhs = _calib_scores(scen, clf, mus, prec)
            np.savez(cache, mp=mps, mh=mhs)
            mp, mh = mps, mhs
        tau_mp = float(np.quantile(mp, 0.02))
        tau_mh = float(np.quantile(mh, 0.98))
        out[scen] = dict(
            n_windows=int(len(mp)), tau_maxprob=tau_mp, tau_maha=tau_mh,
            frr_target=0.02,
            mp_quantiles={str(q): float(np.quantile(mp, q))
                          for q in (0.02, 0.1, 0.5)},
            mh_quantiles={str(q): float(np.quantile(mh, q))
                          for q in (0.5, 0.9, 0.98)})
        print(f"  [calib] {scen}: tau_mp={tau_mp:.4f} tau_maha={tau_mh:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    res["calib"] = out
    save_results(res)
    return out


def _calib_gen(task):
    """Top-level (picklable) calibration-window feature generator."""
    scen, ds = task
    _init_worker()
    X = _X[ds]
    starts = _starts_train(ds, 100, SEED)
    rows = []
    for wi, s in enumerate(starts):
        x = X[s:s + L].T.copy()
        C = X.shape[1]
        Fs = []
        for c in range(C):
            vals, m, lab = _calib_channel(scen, ds, wi, c, x[c])
            Fs.append(s16.channel_features(vals, m))
        rows.append(np.stack(Fs))
    return rows


def _calib_scores(scen, clf, mus, prec):
    """300 windows (100/dataset) of known-class channels; window scores."""
    import multiprocessing as _mp
    mps, mhs = [], []
    with _mp.Pool(3, initializer=_init_worker) as pool:
        tasks = [(scen, ds) for ds in s5.DATASETS]
        for rows in pool.imap_unordered(_calib_gen, tasks):
            for F in rows:
                p, mp_, mh = window_scores(clf, mus, prec, F)
                mps.append(mp_)
                mhs.append(mh)
    return np.array(mps), np.array(mhs)


# ------------------------------------------------------------- detection -----
def grid_features(scen):
    """Features of the scenario's held-out-mechanism eval windows.
    Returns {mech: {rate: {ds: {wi: [C,14]}}}}; cached to s22_ckpt."""
    cache = os.path.join(CKPT, f"gridfeats_{scen}.npz")
    spec = SCENARIOS[scen]
    if os.path.exists(cache):
        z = np.load(cache)
        out = {}
        for k in z.files:
            mech, rate, ds, wi = k.split("|")
            out.setdefault(mech, {}).setdefault(float(rate), {}) \
               .setdefault(ds, {})[int(wi)] = z[k]
        return out
    jobs, keys = [], {}
    for mech in spec["unknown"]:
        for rate in spec["rates"]:
            for ds in s5.DATASETS:
                for wi in range(N_EVAL):
                    j = ("feat", ds, "eval", wi, mech, rate)
                    jobs.append(j)
                    keys[j] = (mech, rate, ds, wi)
    t0 = time.time()
    r = run_jobs(jobs)
    print(f"  [gridfeats] {scen}: {len(jobs)} windows "
          f"({time.time()-t0:.0f}s)", flush=True)
    out, store = {}, {}
    for j, (mech, rate, ds, wi) in keys.items():
        F = r[j][0]
        out.setdefault(mech, {}).setdefault(rate, {}).setdefault(ds, {})[wi] = F
        store[f"{mech}|{rate}|{ds}|{wi}"] = F
    np.savez(cache, **store)
    return out


def run_detect(res):
    out = {}
    for scen, spec in SCENARIOS.items():
        clf = train_scenario_clf(scen)
        mus, prec = fit_maha(scen)
        z = np.load(os.path.join(CKPT, f"calib_{scen}.npz"))
        cal_mp, cal_mh = z["mp"], z["mh"]
        tau_mp = res["calib"][scen]["tau_maxprob"]
        tau_mh = res["calib"][scen]["tau_maha"]
        grids = grid_features(scen)
        rec = {"tau_maxprob": tau_mp, "tau_maha": tau_mh}
        for mech in spec["unknown"]:
            mrec = {}
            for rate in spec["rates"]:
                # pooled over datasets
                preds, mps, mhs = [], [], []
                for ds in s5.DATASETS:
                    feats = grids[mech][rate][ds]
                    p, mp_, mh = apply_scores(clf, mus, prec, feats)
                    preds.append(p)
                    mps.append(mp_)
                    mhs.append(mh)
                preds = np.concatenate(preds)
                mps = np.concatenate(mps)
                mhs = np.concatenate(mhs)
                mrec[str(rate)] = dict(
                    n=int(len(mps)),
                    det_maxprob=float((mps < tau_mp).mean()),
                    det_maha=float((mhs > tau_mh).mean()),
                    pred_frac={CLASSES[c]: float((preds == c).mean())
                               for c in np.unique(preds)},
                    mp_mean=float(mps.mean()), mh_mean=float(mhs.mean()))
            rec[mech] = mrec
        # full curves: pooled over rates per mechanism
        qs = np.linspace(0.0, 1.0, 41)
        curves = {}
        for mech in spec["unknown"]:
            mps, mhs = [], []
            for rate in spec["rates"]:
                for ds in s5.DATASETS:
                    feats = grids[mech][rate][ds]
                    _, mp_, mh = apply_scores(clf, mus, prec, feats)
                    mps.append(mp_)
                    mhs.append(mh)
            mps = np.concatenate(mps)
            mhs = np.concatenate(mhs)
            taus_mp = np.quantile(cal_mp, qs)
            taus_mh = np.quantile(cal_mh, 1.0 - qs)
            curves[mech] = dict(
                frr=qs.tolist(),
                det_maxprob=[float((mps < t).mean()) for t in taus_mp],
                det_maha=[float((mhs > t).mean()) for t in taus_mh],
                det_at_2pct_maxprob=float((mps < np.quantile(cal_mp, 0.02)).mean()),
                det_at_2pct_maha=float((mhs > np.quantile(cal_mh, 0.98)).mean()))
        rec["curves"] = curves
        out[scen] = rec
        for mech in spec["unknown"]:
            d = curves[mech]
            print(f"  [detect] {scen}/{mech}: det@2%FRR "
                  f"maxprob={d['det_at_2pct_maxprob']:.3f} "
                  f"maha={d['det_at_2pct_maha']:.3f}", flush=True)
    res["detect"] = out
    save_results(res)
    return out


# ------------------------------------------------------------- e2e relMSE ----
def _store_rel(ds, mech, rate, fill, d5, d6):
    """Per-window relMSE from stores (first N_EVAL windows); None if
    missing."""
    src = d6 if fill in ("tail_tobit", "tail_tri") else d5
    k = f"{mech}:{fill}:{rate}"
    if ds not in src or k not in src[ds]:
        return None
    clean = np.array(d5[ds]["clean:none:0.0"]["mse_per_window"])[:N_EVAL]
    mse = np.array(src[ds][k]["mse_per_window"])
    return mse[:N_EVAL] / clean


def _mse_from_fills(r, model):
    """r: {job: (F, extra, gt)} from 'fills' jobs -> per-window MSE per fill,
    averaged over mask variants. Returns {fill: [N_EVAL]}."""
    gtm, rows = {}, []
    nvar = 0
    for j, (F, extra, gt) in r.items():
        gtm[j[4]] = gt
        for (vi, f), ctx in extra.items():
            rows.append((j[4], vi, f, ctx))
            nvar = max(nvar, vi + 1)
    fills = sorted({f for _, _, f, _ in rows})
    mse = {}
    for f in fills:
        acc = np.zeros(N_EVAL)
        for vi in range(nvar):
            sel = sorted([rr for rr in rows if rr[1] == vi and rr[2] == f])
            ctx = np.stack([rr[3] for rr in sel])
            wis = [rr[0] for rr in sel]
            gt = np.stack([gtm[w] for w in wis]).astype(np.float64)
            S, Cc = ctx.shape[0], ctx.shape[1]
            pred = model.predict_point(
                ctx.reshape(S * Cc, L)).reshape(S, Cc, H).astype(np.float64)
            acc[wis] += ((pred - gt) ** 2).mean(axis=(1, 2))
        mse[f] = acc / nvar
    return mse


def block_tobit_fillin(device):
    """block:tail_tobit at rates {0.1,0.5,0.7} (0.3 is stored in s6);
    store convention: MSE averaged over mask seeds {0,1}."""
    cache = os.path.join(CKPT, "mse_block_tobit.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    model = s5.BoltModel(device)
    out = {}
    for ds in s5.DATASETS:
        for rate in (0.1, 0.5, 0.7):
            t0 = time.time()
            jobs = [("fills", ds, "block", rate, wi, ("tail_tobit",))
                    for wi in range(N_EVAL)]
            r = run_jobs(jobs)
            mse_w = _mse_from_fills(r, model)["tail_tobit"]
            out[f"{ds}|{rate}"] = mse_w
            print(f"  [fillin] block:tail_tobit:{rate} {ds} "
                  f"mse={mse_w.mean():.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
    np.savez(cache, **out)
    del model
    torch.cuda.empty_cache()
    return out


def novel_mse(device):
    """Per-window MSE for the novel mechanisms (iburst, mnar_low) under the
    four fills; store convention for iburst: averaged over mask seeds."""
    cache = os.path.join(CKPT, "mse_novel.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    model = s5.BoltModel(device)
    out = {}
    for mech in ("iburst", "mnar_low"):
        for rate in RATES2:
            for ds in s5.DATASETS:
                t0 = time.time()
                jobs = [("fills", ds, mech, rate, wi, tuple(FILLS_E2E))
                        for wi in range(N_EVAL)]
                r = run_jobs(jobs)
                mse = _mse_from_fills(r, model)
                for f in FILLS_E2E:
                    out[f"{mech}|{rate}|{ds}|{f}"] = mse[f]
                print(f"  [novel-mse] {mech}:{rate} {ds} "
                      f"linear={mse['linear'].mean():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(cache, **out)
    del model
    torch.cuda.empty_cache()
    return out


def run_e2e(res, device):
    """Arms on held-out mechanisms: fixed_linear / hard_gated /
    openset_gated(maxprob) / openset_maha / oracle; tables T_safe, T_danger.
    relMSE vs paired clean (per window), mean + p95 + max."""
    d5 = json.load(open(os.path.join(HERE, "s5_missing_results.json")))["bolt"]
    d6 = json.load(open(os.path.join(HERE, "s6_mnarfix_results.json")))["bolt"]
    out = {}

    # ---- per-window relMSE tensor per scenario: mech,rate,ds,fill -> [150]
    rel, preds_w, unk_mp_w, unk_mh_w = {}, {}, {}, {}
    for scen, spec in SCENARIOS.items():
        clf = train_scenario_clf(scen)
        mus, prec = fit_maha(scen)
        tau_mp = res["calib"][scen]["tau_maxprob"]
        tau_mh = res["calib"][scen]["tau_maha"]
        grids = grid_features(scen)
        for mech in spec["unknown"]:
            for rate in spec["rates"]:
                for ds in s5.DATASETS:
                    feats = grids[mech][rate][ds]
                    p, mp_, mh = apply_scores(clf, mus, prec, feats)
                    key = (scen, mech, rate, ds)
                    preds_w[key] = p
                    unk_mp_w[key] = mp_ < tau_mp
                    unk_mh_w[key] = mh > tau_mh
    res.setdefault("e2e", {})
    # ---- relMSE matrices
    def rel_for(scen, mech, rate, ds, fill):
        if scen == "novel":
            mse_all = run_e2e._novel
            clean = np.array(
                d5[ds]["clean:none:0.0"]["mse_per_window"])[:N_EVAL]
            return mse_all[f"{mech}|{rate}|{ds}|{fill}"] / clean
        if fill == "tail_tobit" and mech == "block" and rate != 0.3:
            mse = run_e2e._fillin[f"{ds}|{rate}"]
            clean = np.array(
                d5[ds]["clean:none:0.0"]["mse_per_window"])[:N_EVAL]
            return mse / clean
        v = _store_rel(ds, mech, rate, fill, d5, d6)
        return v

    run_e2e._novel = None
    if "novel" in SCENARIOS:
        run_e2e._novel = novel_mse(device)
    run_e2e._fillin = block_tobit_fillin(device)

    for scen, spec in SCENARIOS.items():
        rec = {}
        for mech in spec["unknown"]:
            for table_name, table in TABLES.items():
                arms = {"fixed_linear": [], "hard_gated": [],
                        "openset_gated": [], "openset_maha": [], "oracle": []}
                per_rate = {}
                for rate in spec["rates"]:
                    arm_rate = {a: [] for a in arms}
                    for ds in s5.DATASETS:
                        key = (scen, mech, rate, ds)
                        p = preds_w[key]
                        u_mp = unk_mp_w[key]
                        u_mh = unk_mh_w[key]
                        per = {f: rel_for(scen, mech, rate, ds, f)
                               for f in FILLS_E2E}
                        if any(v is None for v in per.values()):
                            raise RuntimeError(f"missing store cell {key}")
                        M = np.stack([per[f] for f in FILLS_E2E])
                        i_lin = FILLS_E2E.index("linear")
                        oracle = M.min(axis=0)
                        def route_fill(lab):
                            return table.get(int(lab), "linear")
                        r_hard = np.array([FILLS_E2E.index(route_fill(l))
                                           for l in p])
                        r_mp = np.array([i_lin if u_mp[i] else r_hard[i]
                                         for i in range(len(p))])
                        r_mh = np.array([i_lin if u_mh[i] else r_hard[i]
                                         for i in range(len(p))])
                        n = len(p)
                        arm_rate["fixed_linear"].append(M[i_lin])
                        arm_rate["hard_gated"].append(M[r_hard, np.arange(n)])
                        arm_rate["openset_gated"].append(M[r_mp, np.arange(n)])
                        arm_rate["openset_maha"].append(M[r_mh, np.arange(n)])
                        arm_rate["oracle"].append(oracle)
                    per_rate[str(rate)] = {a: float(np.concatenate(v).mean())
                                           for a, v in arm_rate.items()}
                    for a in arms:
                        arms[a].append(np.concatenate(arm_rate[a]))
                row = {}
                for a, chunks in arms.items():
                    v = np.concatenate(chunks)
                    row[a] = dict(mean=float(v.mean()),
                                  p95=float(np.quantile(v, 0.95)),
                                  max=float(v.max()))
                # extra diagnostics
                ufrac_mp = float(np.mean([
                    unk_mp_w[(scen, mech, r, ds)].mean()
                    for r in spec["rates"] for ds in s5.DATASETS]))
                ufrac_mh = float(np.mean([
                    unk_mh_w[(scen, mech, r, ds)].mean()
                    for r in spec["rates"] for ds in s5.DATASETS]))
                rec[f"{mech}|{table_name}"] = dict(
                    arms=row, per_rate=per_rate,
                    unknown_frac_maxprob=ufrac_mp, unknown_frac_maha=ufrac_mh)
                print(f"  [e2e] {scen}/{mech}/{table_name}: "
                      + "  ".join(f"{a}={row[a]['mean']:.3f}"
                                  f"(p95 {row[a]['p95']:.2f})"
                                  for a in ("fixed_linear", "hard_gated",
                                            "openset_gated", "oracle")),
                      flush=True)
        out[scen] = rec
    res["e2e"] = out
    save_results(res)
    return out


# ---------------------------------------------------------------- coverage ----
def run_coverage(res, device):
    """ETTh1 cal/test streams, T_danger routing, bolt [q10,q90] widened to
    90%. Arms: fixed_linear (global width), hard_gated (Mondrian by predicted
    class), openset_gated (maxprob unknown -> linear + w_max), openset_maha
    (maha unknown -> linear + w_max; added after maxprob showed zero
    detection -- disclosed in notes). Unknown flags are series-level
    thresholdings of the window-calibrated scores."""
    model = s16.Bolt96(device)
    out = {}
    MIN_GROUP = 30
    for scen, spec in SCENARIOS.items():
        t0 = time.time()
        clf = train_scenario_clf(scen)
        mus, prec = fit_maha(scen)
        tau_mp = res["calib"][scen]["tau_maxprob"]
        tau_mh = res["calib"][scen]["tau_maha"]
        cache = os.path.join(CKPT, f"cov2_{scen}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            out[scen] = json.loads(z["summary"].item())
            print(f"  [cov] {scen}: cached", flush=True)
            continue
        # ---- cal stream (150 windows, known-mech mix)
        jobs = [("covcal", scen, wi) for wi in range(150)]
        r = run_jobs(jobs)
        cal = []
        for wi in range(150):
            F, extra, _ = r[("covcal", scen, wi)]
            vals, m, tl = extra
            for c in range(vals.shape[0]):
                p, mp_, mh = window_scores(clf, mus, prec, F[c:c + 1])
                cal.append(dict(vals=vals[c], m=m[c], tl=int(tl[c]),
                                pred=p, unk=bool(mp_ < tau_mp),
                                unk_mh=bool(mh > tau_mh)))
        # ---- test stream (held-out mech, rates {0.3,0.5})
        test = []
        trates = RATES2
        for mech in spec["unknown"]:
            for rate in trates:
                jobs = [("covtest", scen, mech, rate, wi)
                        for wi in range(150)]
                r = run_jobs(jobs)
                for wi in range(150):
                    F, extra, _ = r[("covtest", scen, mech, rate, wi)]
                    x, m = extra
                    for c in range(x.shape[0]):
                        p, mp_, mh = window_scores(clf, mus, prec,
                                                   F[c:c + 1])
                        test.append(dict(x=x[c], m=m[c], mech=mech,
                                         rate=rate, pred=p,
                                         unk=bool(mp_ < tau_mp),
                                         unk_mh=bool(mh > tau_mh)))
        # ---- routed contexts per arm
        def fill_of(vals, m, fill):
            if not m.any():
                return vals.astype(np.float64)
            return s5.fill_context(vals[None, :].astype(np.float64),
                                   m[None, :], fill)[0]

        def ctx_of(row, arm, key_x, key_m):
            if arm == "fixed_linear":
                return fill_of(row[key_x], row[key_m], "linear")
            fill = T_DANGER.get(row["pred"], "linear")
            if arm == "openset_gated" and row["unk"]:
                fill = "linear"
            if arm == "openset_maha" and row["unk_mh"]:
                fill = "linear"
            return fill_of(row[key_x], row[key_m], fill)

        rec = {"n_cal": len(cal), "n_test": len(test),
               "cal_unknown_frac": float(np.mean([r["unk"] for r in cal])),
               "cal_unknown_frac_maha": float(
                   np.mean([r["unk_mh"] for r in cal]))}
        # ---- quantiles per arm
        arms = ("fixed_linear", "hard_gated", "openset_gated", "openset_maha")
        QA = {}
        for arm in arms:
            Xc = np.stack([ctx_of(r, arm, "vals", "m")
                           for r in cal]).astype(np.float32)
            q10c, q90c = model.predict_quantiles(Xc)
            Xt = np.stack([ctx_of(r, arm, "x", "m")
                           for r in test]).astype(np.float32)
            q10t, q90t = model.predict_quantiles(Xt)
            QA[arm] = (q10c, q90c, q10t, q90t)
        # ---- ground truth for cal/test rows (rows are ordered
        # mech -> rate -> wi -> channel, matching construction above)
        _init_worker()
        X = _X["ETTh1"]
        starts = _starts_eval("ETTh1")
        yt = []
        for mech in spec["unknown"]:
            for rate in trates:
                for wi in range(150):
                    s = starts[wi]
                    yw = X[s + L:s + L + H].T.copy()
                    for c in range(X.shape[1]):
                        yt.append(yw[c])
        y_cal = []
        starts_cal = _starts_cal("ETTh1")
        for wi in range(150):
            s = starts_cal[wi]
            yw = X[s + L:s + L + H].T.copy()
            for c in range(X.shape[1]):
                y_cal.append(yw[c])
        yt = np.stack(yt).astype(np.float64)
        y_cal = np.stack(y_cal).astype(np.float64)
        # ---- widths per arm
        summary = {}
        for arm in arms:
            q10c, q90c, q10t, q90t = [a.astype(np.float64) for a in QA[arm]]
            E = np.maximum(q10c - y_cal, y_cal - q90c)
            w_glob = s16.conformal_w(E)
            plab_cal = np.array([r["pred"] for r in cal])
            pools = {}
            for c in known_classes(scen):
                sel = plab_cal == c
                if sel.sum() >= MIN_GROUP:
                    pools[c] = s16.conformal_w(E[sel])
            w_max = np.max(np.stack(list(pools.values())), axis=0) \
                if pools else w_glob
            # test application
            plab_te = np.array([r["pred"] for r in test])
            unk_te = np.array([r["unk"] for r in test])
            unk_te_mh = np.array([r["unk_mh"] for r in test])
            if arm == "fixed_linear":
                w_all = np.tile(w_glob, (len(test), 1))
            elif arm == "hard_gated":
                w_all = np.stack([pools.get(c, w_glob) for c in plab_te])
            elif arm == "openset_gated":
                w_all = np.stack([w_max if u else pools.get(c, w_glob)
                                  for c, u in zip(plab_te, unk_te)])
            else:
                w_all = np.stack([w_max if u else pools.get(c, w_glob)
                                  for c, u in zip(plab_te, unk_te_mh)])
            per_mech = {}
            for mech in spec["unknown"]:
                sel = np.array([r["mech"] == mech for r in test])
                inside = (yt[sel] >= q10t[sel] - w_all[sel]) & \
                         (yt[sel] <= q90t[sel] + w_all[sel])
                per_mech[mech] = dict(
                    coverage=float(inside.mean()),
                    width=float((q90t[sel] - q10t[sel]
                                 + 2 * w_all[sel]).mean()),
                    n=int(sel.sum()),
                    unknown_frac=float(
                        (unk_te_mh[sel] if arm == "openset_maha"
                         else unk_te[sel]).mean()))
            inside = (yt >= q10t - w_all) & (yt <= q90t + w_all)
            summary[arm] = dict(
                coverage=float(inside.mean()),
                width=float((q90t - q10t + 2 * w_all).mean()),
                per_mech=per_mech)
        rec["arms"] = summary
        rec["cal_pred_frac"] = {CLASSES[c]: float((plab_cal == c).mean())
                                for c in np.unique(plab_cal)}
        out[scen] = rec
        np.savez(cache, summary=json.dumps(rec))
        print(f"  [cov] {scen} ({time.time()-t0:.0f}s): " +
              "  ".join(f"{a} cov={summary[a]['coverage']:.3f} "
                        f"w={summary[a]['width']:.2f}" for a in arms),
              flush=True)
    res["coverage"] = out
    save_results(res)
    del model
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------- real probe ----
def run_real_probe(res):
    """Score S16's Penmanshiel + METR-LA windows with the 'novel' pipeline
    (full 6-class clf + tau_2%); fraction flagged unknown per group."""
    scen = "novel"
    clf = train_scenario_clf(scen)
    mus, prec = fit_maha(scen)
    tau_mp = res["calib"][scen]["tau_maxprob"]
    tau_mh = res["calib"][scen]["tau_maha"]
    out = {}
    import run_s10_real as s10
    windows, A = s10.load_all()
    ids_c = s10.group_ids(windows, "cens")
    ids_t = s10.group_ids(windows, "ctrl")
    for mode in ("oracle", "auto"):
        rows = []
        for i in np.concatenate([ids_c, ids_t]):
            m = (A["mask"][i] if mode == "oracle"
                 else s10.candidate_mask(A["ctx_rec"][i], A["avail"][i]))
            p, mp_, mh = window_scores(
                clf, mus, prec, s16.channel_features(A["ctx_rec"][i], m)[None])
            rows.append((i, p, mp_, mh))
        for grp, ids in (("cens", set(ids_c)), ("ctrl", set(ids_t))):
            sel = [r for r in rows if r[0] in ids]
            mp_ = np.array([r[2] for r in sel])
            mh = np.array([r[3] for r in sel])
            pred = np.array([r[1] for r in sel])
            out[f"penn_{mode}_{grp}"] = dict(
                n=len(sel),
                unknown_maxprob=float((mp_ < tau_mp).mean()),
                unknown_maha=float((mh > tau_mh).mean()),
                pred_mnar_frac=float(np.isin(pred, [MNAR_H, MNAR_E]).mean()))
        print(f"  [probe] penn_{mode}: " +
              "  ".join(f"{grp} unk_mp={out[f'penn_{mode}_{grp}']['unknown_maxprob']:.3f}"
                        for grp in ("cens", "ctrl")), flush=True)
    import run_s14_metrla as s14
    windows = s14.build_windows()
    A = s14.extract(windows)
    ids_m = s14.group_ids(windows, "miss")
    ids_t = s14.group_ids(windows, "ctrl")
    rows = []
    for i in np.concatenate([ids_m, ids_t]):
        p, mp_, mh = window_scores(
            clf, mus, prec, s16.channel_features(A["ctx_rec"][i],
                                                 A["mask"][i])[None])
        rows.append((i, p, mp_, mh))
    for grp, ids in (("miss", set(ids_m)), ("ctrl", set(ids_t))):
        sel = [r for r in rows if r[0] in ids]
        mp_ = np.array([r[2] for r in sel])
        mh = np.array([r[3] for r in sel])
        pred = np.array([r[1] for r in sel])
        mnar_sel = np.isin(pred, [MNAR_H, MNAR_E])
        out[f"metrla_{grp}"] = dict(
            n=len(sel),
            unknown_maxprob=float((mp_ < tau_mp).mean()),
            unknown_maha=float((mh > tau_mh).mean()),
            pred_mnar_frac=float(mnar_sel.mean()),
            unknown_maxprob_among_pred_mnar=float(
                (mp_[mnar_sel] < tau_mp).mean()) if mnar_sel.any() else None,
            unknown_maha_among_pred_mnar=float(
                (mh[mnar_sel] > tau_mh).mean()) if mnar_sel.any() else None)
        print(f"  [probe] metrla_{grp}: "
              f"unk_mp={out[f'metrla_{grp}']['unknown_maxprob']:.3f} "
              f"unk_maha={out[f'metrla_{grp}']['unknown_maha']:.3f}",
              flush=True)
    res["real_probe"] = out
    save_results(res)
    return out


# ------------------------------------------------------------------ figure ----
def run_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    scen_labels = {"holdout_mnar_extreme": "held-out mnar_extreme",
                   "holdout_block": "held-out block",
                   "novel": None}
    colors = {"holdout_mnar_extreme": "#d62728",
              "holdout_block": "#ff7f0e", "iburst": "#1f77b4",
              "mnar_low": "#2ca02c"}

    # A/B: detection-vs-FRR curves
    for ax, meth, det_key in ((axes[0][0], "maxprob", "det_maxprob"),
                              (axes[0][1], "maha", "det_maha")):
        for scen, spec in SCENARIOS.items():
            cur = res["detect"][scen]["curves"]
            for mech in spec["unknown"]:
                c = curves_key = f"{scen}:{mech}"
                lab = (f"{scen_labels.get(scen) or mech}"
                       if scen != "novel" else f"novel: {mech}")
                col = colors.get(mech, colors.get(scen, "#333333"))
                d = cur[mech]
                ax.plot(d["frr"], d[det_key], color=col,
                        ls="-" if scen != "novel" else "--", label=lab)
                i2 = int(np.argmin(np.abs(np.array(d["frr"]) - 0.02)))
                ax.scatter([d["frr"][i2]], [d[det_key][i2]], color=col,
                           zorder=5, s=25)
        ax.axvline(0.02, color="k", ls=":", lw=1)
        ax.set_xlabel("known-class false-rejection rate")
        ax.set_ylabel("unknown detection rate")
        ax.set_title(f"{'A' if meth == 'maxprob' else 'B'}. Open-set "
                     f"detection vs FRR ({meth})\n(dot = calibrated tau @ "
                     "2% FRR)", fontsize=10)
        ax.set_xlim(0, 0.3)
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    # C/D: e2e relMSE bars
    for ax, tname, tag in ((axes[0][2], "T_safe", "C"),
                           (axes[1][0], "T_danger", "D")):
        rows = []
        for scen, spec in SCENARIOS.items():
            for mech in spec["unknown"]:
                r = res["e2e"][scen][f"{mech}|{tname}"]
                lab = (scen_labels.get(scen) or mech)
                if scen == "novel":
                    lab = f"novel\n{mech}"
                rows.append((lab, r["arms"]))
        arms = ("fixed_linear", "hard_gated", "openset_gated",
                "openset_maha", "oracle")
        cols = {"fixed_linear": "#c7c7c7", "hard_gated": "#d62728",
                "openset_gated": "#7fb3d5", "openset_maha": "#1f77b4",
                "oracle": "#2ca02c"}
        x = np.arange(len(rows))
        w = 0.16
        for ai, a in enumerate(arms):
            vals = [r[1][a]["mean"] for r in rows]
            ax.bar(x + (ai - 2) * w, vals, width=w, label=a,
                   color=cols[a])
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], fontsize=7)
        ax.set_ylabel("mean relMSE vs clean")
        ax.set_title(f"{tag}. End-to-end on held-out mechanisms ({tname})\n"
                     "(bolt, pooled datasets x rates)", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")

    # E: worst-case p95 under T_danger
    ax = axes[1][1]
    rows = []
    for scen, spec in SCENARIOS.items():
        for mech in spec["unknown"]:
            r = res["e2e"][scen][f"{mech}|T_danger"]
            lab = scen_labels.get(scen) or f"novel\n{mech}"
            if scen == "novel":
                lab = f"novel\n{mech}"
            rows.append((lab, r["arms"]))
    arms = ("fixed_linear", "hard_gated", "openset_maha")
    cols = {"fixed_linear": "#c7c7c7", "hard_gated": "#d62728",
            "openset_maha": "#1f77b4"}
    x = np.arange(len(rows))
    w = 0.25
    for ai, a in enumerate(arms):
        vals = [r[1][a]["p95"] for r in rows]
        ax.bar(x + (ai - 1) * w, vals, width=w, label=a, color=cols[a])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylabel("p95 per-window relMSE (log)")
    ax.set_yscale("log")
    ax.set_title("E. Worst-case tail under T_danger\n(open-set fallback vs "
                 "hard gating)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # F: coverage on held-out test streams
    ax = axes[1][2]
    arms = ("fixed_linear", "hard_gated", "openset_gated", "openset_maha")
    rows = []
    for scen in SCENARIOS:
        cov = res["coverage"][scen]["arms"]
        rows.append((scen_labels.get(scen) or "novel", cov))
    x = np.arange(len(rows))
    w = 0.2
    cols2 = {"fixed_linear": "#c7c7c7", "hard_gated": "#d62728",
             "openset_gated": "#7fb3d5", "openset_maha": "#1f77b4"}
    for ai, a in enumerate(arms):
        vals = [r[1][a]["coverage"] for r in rows]
        ax.bar(x + (ai - 1.5) * w, vals, width=w, label=a, color=cols2[a])
        for xi, v in zip(x + (ai - 1.5) * w, vals):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax.axhline(0.9, color="k", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("coverage (target 0.9)")
    ax.set_title("F. Coverage on held-out-mechanism test streams\n"
                 "(ETTh1, T_danger routing, bolt [q10,q90]+conformal)",
                 fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "s22.png"), dpi=150)
    print("saved s22.png", flush=True)


# ------------------------------------------------------------- io helpers ----
def save_results(res, path=OUT):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
    os.replace(tmp, path)


def load_results(path=OUT):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


# -------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke",
                    help="comma list: smoke,gate,calib,eval,figure,all")
    ap.add_argument("--device", default="cuda:4")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    modes = args.mode.split(",")
    if "all" in modes:
        modes = ["gate", "calib", "eval", "figure"]
    t0 = time.time()
    res = load_results(args.out)
    res.setdefault("meta", dict(
        track="S22: open-set fallback for the S16 MechGate detector",
        seed=SEED, scenarios={k: dict(drop=[CLASSES[c] for c in v["drop"]],
                                      unknown=v["unknown"], rates=v["rates"])
                              for k, v in SCENARIOS.items()},
        methods=dict(
            maxprob="window score = max of channel-mean posterior; reject "
                    "< tau (2nd pct of 300 train-region calib windows)",
            maha="window score = mean over channels of min Mahalanobis "
                 "distance to known-class prototypes (Ledoit-Wolf pooled "
                 "cov, feat col 12 excluded); reject > tau (98th pct)"),
        routing=dict(unknown="linear fill + w_max conformal width",
                     T_safe={CLASSES[k]: v for k, v in T_SAFE.items()},
                     T_danger={CLASSES[k]: v for k, v in T_DANGER.items()}),
        eval_windows=f"first {N_EVAL} of S16's 300 eval windows",
        novel_masks="iburst: Bernoulli p_t=clip(rate*(0.2+1.6*t/(L-1)),0,"
                    "0.97), ms-averaged {0,1}; mnar_low: lowest-rank "
                    "censoring (deterministic)"))
    save_results(res, args.out)

    if "smoke" in modes:
        run_smoke(args.device)
        return
    if "gate" in modes:
        run_gate(res)
    if "calib" in modes:
        run_calib(res)
    if "eval" in modes:
        run_detect(res)
        run_e2e(res, args.device)
        run_coverage(res, args.device)
        run_real_probe(res)
    if "coverage" in modes:
        run_coverage(res, args.device)
    if "figure" in modes:
        run_figure(res)
    print(f"[done] modes={modes} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
