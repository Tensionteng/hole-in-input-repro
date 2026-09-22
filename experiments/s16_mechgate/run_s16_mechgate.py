#!/usr/bin/env python
"""S16: mechanism detector + gated repair pipeline (detect -> route -> calibrate).

Prior rounds established that the optimal context repair is mechanism-SPECIFIC:
S5/S5-fix (synthetic grid: linear wins mcar, bolt-nan wins block, linear wins
mnar), S6 (Tobit helps only artificial mnar_high on ETT; mechanism-matched
conformal restores coverage), S10 (real curtailment: zero wins, Tobit
reverses), S14 (real sensor outage: linear/nan win, zero catastrophic; S6
bridge-rank detector v1 is per-window unreliable under blocky masks).

S16 closes the loop with a deployable pipeline:

Track A -- multi-feature mechanism classifier.
  Classes: {clean, mcar, block, mnar_high, mnar_extreme, intermittent}.
  Window-channel features, all dimensionless (L-normalized so CTX=144 real
  windows and CTX=512 synthetic windows share a space):
    miss_rate, runs_per_100, mean/med/p90 run length / L, fraction of missing
    points in long runs (>= max(6, L/16)), S6 bridge-rank features
    (r_bar, e_bar, x_bar) + e_max (max flanking-endpoint |rank-0.5|),
    zero fraction among observed, scale-fallback flag (dead channel),
    cross-channel mask correlation (multi-channel windows; 0 for univariate),
    Little's-test-style chi-square segment statistic (also used ALONE as the
    single-feature literature baseline).
  Training: synthetic masks from the S5 generator (run_s5_missing.make_mask
  for mcar/mnar; same algorithm with jittered block length + optional
  channel-shared blocks for block) on TRAIN-split windows of
  ETTh1/ETTm1/weather, rates p ~ U(0.05, 0.8). intermittent = natural
  near-zero channels (weather) + synthetic sparsification (random 50-95% of
  points zeroed, NO mask -- the zeros are the data, not missingness).
  Models: multinomial logistic regression and gradient boosting
  (sklearn HistGB), leave-one-dataset-out CV. Baseline: single-feature
  classifier on the Little-style chi-square statistic.
  Transfer (zero-shot, the paper sell): Penmanshiel censored windows (oracle
  label mnar_high; masks = status-code oracle, plus the status-free
  auto-candidate variant) and METR-LA miss windows (oracle label block;
  mask = exact zeros), each with matched clean controls. Compared against
  the S6/S10 v1 rule detector recomputed per window.

Track B -- gated repair pipeline, per-window, assembled from the existing
  per-window result stores whenever possible:
  * Eval A (s12_results.json, bolt, 150 windows, masks == s5.make_mask
    ms=0 per run_s12_recon tiny assert): candidates {zs:linear, zs:nan,
    tok:nan, mb:nan, recon:nan}; policies over fills-only and full sets.
  * Eval B (s5_missing_results.json + s6_mnarfix_results.json, bolt,
    300 windows, identical masks across the two stores): candidates
    {zero, ffill, linear, zero_oscale(mcar+mnar), tail_tobit(mnar)}.
    The MNAR-routing-to-linear result is a feature, not a bug.
  * Mondrian conformal (synthetic): bolt native [q10,q90] widened to 90%
    target, split-conformal grouped by PREDICTED class vs naive global pool
    vs grouping by ORACLE class, on a mixed-mechanism deployment stream.
  * Real end-to-end: Penmanshiel + METR-LA gated NMSE assembled from
    s10/s14 per-window stores (zero new inference) + bolt quantile reruns
    for Mondrian-by-predicted conformal (naive / mondrian-pred /
    mondrian-oracle on the exact S10/S14 cal-test splits).

Anchor gate: 3 store cells re-inferred with bolt (s12 ETTh1 mcar:zs:linear:0.3,
s5 ETTm1 mnar_high:linear:0.5, s10 bolt cens zero) must reproduce within 5%.

Modes (comma list): smoke, gate, trackA, trackB_gate, conf_syn, conf_real,
real_e2e, figure, all. New files only: run_s16_mechgate.py, s16_results.json,
s16.png, s16_notes.md, s16_ckpt/, s16_*.log.
"""
import argparse
import json
import os
import pickle
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))

import numpy as np
import torch

import run_s5_missing as s5

L, H = s5.L, s5.H                      # 512, 96
SEED = 20250816
OUT = os.path.join(HERE, "s16_results.json")
CKPT = os.path.join(HERE, "s16_ckpt")
os.makedirs(CKPT, exist_ok=True)

CLASSES = ["clean", "mcar", "block", "mnar_high", "mnar_extreme", "intermittent"]
CLEAN, MCAR, BLOCK, MNAR_H, MNAR_E, INTER = range(6)
FEAT_NAMES = ["miss_rate", "runs_per_100", "mean_run_rel", "med_run_rel",
              "p90_run_rel", "frac_miss_long", "r_bar", "e_bar", "x_bar",
              "e_max", "zero_frac_obs", "scale_fallback", "xmask_corr",
              "little_chi2"]
RATES = [0.1, 0.3, 0.5, 0.7]


# ============================================================== features ====
def _runs(mask):
    """[L] bool -> list of (start, end) inclusive."""
    Lc = len(mask)
    dm = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(dm == 1) + 1) + ([0] if mask[0] else [])
    ends = list(np.flatnonzero(dm == -1)) + ([Lc - 1] if mask[-1] else [])
    return list(zip(sorted(starts), sorted(ends)))


def channel_features(x1d, mask):
    """x1d [L] as-recorded values (NaN allowed = unobserved); mask [L] bool
    True = flagged-missing. Effective missing = mask | ~isfinite. Returns a
    14-dim vector (FEAT_NAMES order); xmask_corr slot left 0 (caller fills)."""
    Lc = len(x1d)
    mask = np.asarray(mask, bool) | ~np.isfinite(x1d)
    nmiss = int(mask.sum())
    valid = np.flatnonzero(~mask)
    obs = x1d[valid]
    miss_rate = nmiss / Lc
    # value-side features on observed points
    if len(obs) > 1:
        zero_frac = float((obs == 0.0).mean())
        scale_fb = float(float(np.mean(np.abs(obs))) < 1e-8
                         or float(obs.std()) < 1e-8)
    else:
        zero_frac, scale_fb = 0.0, 1.0
    # run geometry
    runs = _runs(mask)
    n_runs = len(runs)
    lens = np.array([j - i + 1 for i, j in runs]) if runs else np.zeros(1)
    long_thr = max(6, Lc // 16)
    frac_long = float(lens[lens >= long_thr].sum() / nmiss) if nmiss else 0.0
    # Little-style chi-square: missing counts over 8 segments vs uniform
    seg = np.array_split(mask.astype(np.float64), 8)
    m_k = np.array([s.sum() for s in seg])
    mbar = m_k.mean()
    chi2 = float(((m_k - mbar) ** 2 / max(mbar, 1e-9)).sum())
    little = float(np.log1p(chi2 / 7.0))
    if nmiss == 0 or len(obs) < 2:
        return np.array([miss_rate, 100 * n_runs / Lc, 0, 0, 0, frac_long,
                         0.5, 0.25, 0.5, 0.0, zero_frac, scale_fb, 0.0,
                         little], np.float64)
    # S6 bridge-rank features (r_bar / e_bar / x_bar) + e_max
    filled = np.interp(np.arange(Lc), valid, obs)
    obs_sorted = np.sort(obs)
    n_obs = len(obs)
    ranks = np.searchsorted(obs_sorted, filled[mask]) / n_obs
    r_bar = float(ranks.mean())
    x_bar = float(np.maximum(ranks, 1.0 - ranks).mean())
    end_abs = []
    for i, j in runs:
        for t in (i - 1, j + 1):
            if 0 <= t < Lc and not mask[t]:
                end_abs.append(abs(np.searchsorted(obs_sorted, x1d[t]) / n_obs - 0.5))
    e_bar = float(np.mean(end_abs)) if end_abs else 0.25
    e_max = float(np.max(end_abs)) if end_abs else 0.0
    return np.array([miss_rate, 100 * n_runs / Lc,
                     lens.mean() / Lc, np.median(lens) / Lc,
                     np.percentile(lens, 90) / Lc, frac_long,
                     r_bar, e_bar, x_bar, e_max,
                     zero_frac, scale_fb, 0.0, little], np.float64)


def xmask_corr(M):
    """[C, L] bool -> mean pairwise correlation of channel masks (0 if <2
    informative channels or univariate)."""
    C = M.shape[0]
    if C < 2:
        return 0.0
    F = M.astype(np.float64)
    sd = F.std(axis=1)
    ok = sd > 1e-12
    if ok.sum() < 2:
        return 0.0
    Z = (F[ok] - F[ok].mean(axis=1, keepdims=True)) / sd[ok, None]
    Cm = (Z @ Z.T) / M.shape[1]
    iu = np.triu_indices(int(ok.sum()), 1)
    return float(Cm[iu].mean())


def window_features(Xw, Mw):
    """Xw [C, L] values, Mw [C, L] bool -> [C, 14] feature matrix."""
    C = Xw.shape[0]
    F = np.stack([channel_features(Xw[c], Mw[c]) for c in range(C)])
    F[:, 12] = xmask_corr(Mw)
    return F


# ==================================================== synthetic mask gen ====
def block_mask_1d(rate, Lc, rng, blen):
    """Same algorithm as s5.make_mask('block') with parameterized length."""
    m = np.zeros(Lc, bool)
    n_blocks = int(round(rate * Lc / blen))
    if n_blocks > 0:
        for st in rng.integers(0, Lc - blen + 1, size=n_blocks):
            m[st:st + blen] = True
    return m


def load_train_windows(path, n_windows, seed):
    """Train-region mirror of s5.load_windows (origins before the test 20%)."""
    import pandas as pd
    df = pd.read_csv(path)
    X = df.drop(columns=["date"]).to_numpy(np.float32)
    N = len(X)
    test_start = int(0.8 * N)
    lo, hi = 0, test_start - L - H
    rng = np.random.default_rng(seed)
    k = min(n_windows, hi - lo + 1)
    starts = np.sort(rng.choice(hi - lo + 1, size=k, replace=False)) + lo
    return X, starts


def train_samples(ds, n_win=220, seed=SEED):
    """Yield (x1d, mask, label, wi, c) for one dataset's train windows."""
    X, starts = load_train_windows(s5.DATASETS[ds], n_win, seed)
    C = X.shape[1]
    probs = [0.20, 0.20, 0.20, 0.15, 0.15, 0.10]
    for wi, s in enumerate(starts):
        xw = X[s:s + L].T.copy()
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7, wi]))
        shared = bool(rng.random() < 0.5)
        blen = int(rng.choice([12, 24, 48, 96]))
        shared_rate = float(rng.uniform(0.05, 0.8))
        shared_mask = None
        for c in range(C):
            x = xw[c]
            lab = int(rng.choice(6, p=probs))
            if (x == 0).mean() >= 0.5:
                lab = INTER           # natural near-zero channel
            rate = float(rng.uniform(0.05, 0.8))
            mask = np.zeros(L, bool)
            vals = x
            if lab == MCAR:
                mask = s5.make_mask("mcar", rate, wi, 1000 + c, 1)[0]
            elif lab == BLOCK:
                if shared:
                    if shared_mask is None:
                        shared_mask = block_mask_1d(shared_rate, L, rng, blen)
                    mask = shared_mask
                else:
                    mask = block_mask_1d(rate, L, rng, blen)
                if mask.sum() < max(2, 0.02 * L):
                    lab, mask = CLEAN, np.zeros(L, bool)  # no visible damage
            elif lab in (MNAR_H, MNAR_E):
                mech = "mnar_high" if lab == MNAR_H else "mnar_extreme"
                mask = s5.make_mask(mech, rate, wi, 0, 1, x=x[None, :])[0]
            elif lab == INTER and (x == 0).mean() < 0.5:
                keep = rng.random(L) < float(rng.uniform(0.05, 0.5))
                vals = x * keep       # synthetic sparsification, no mask
            yield vals, mask, lab, wi, c


def build_train_features(n_win=220):
    """Feature matrix + labels for all 3 datasets; cached to s16_ckpt."""
    cache = os.path.join(CKPT, "feats_train.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["F"], z["y"], z["g"]
    Fs, ys, gs = [], [], []
    for gi, ds in enumerate(s5.DATASETS):
        t0 = time.time()
        n = 0
        for vals, mask, lab, wi, c in train_samples(ds, n_win):
            Fs.append(channel_features(vals, mask))
            ys.append(lab)
            gs.append(gi)
            n += 1
        print(f"  [train-feats] {ds}: {n} channel samples ({time.time()-t0:.0f}s)",
              flush=True)
    F = np.stack(Fs)
    y = np.array(ys, np.int64)
    g = np.array(gs, np.int64)
    np.savez(cache, F=F, y=y, g=g)
    return F, y, g


# ============================================================= classifiers ====
def make_classifiers():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "gbdt": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, l2_regularization=1.0,
            random_state=SEED),
        "logreg": Pipeline([("sc", StandardScaler()),
                            ("lr", LogisticRegression(max_iter=3000))]),
        # Little's-test-style single-feature baseline (chi2 segment stat)
        "little": Pipeline([("sc", StandardScaler()),
                            ("lr", LogisticRegression(max_iter=3000))]),
    }


def fit_clf(name, F, y):
    clf = make_classifiers()[name]
    clf.fit(F[:, [13]] if name == "little" else F, y)
    return clf


def proba_clf(name, clf, F):
    return clf.predict_proba(F[:, [13]] if name == "little" else F)


# ---------------------------------------------------------------- Track A ----
def run_trackA(res, n_win=220):
    ds_names = list(s5.DATASETS)
    F, y, g = build_train_features(n_win)
    out = {"n_samples": int(len(y)),
           "class_counts": {CLASSES[i]: int((y == i).sum()) for i in range(6)},
           "feat_names": FEAT_NAMES}

    # ---------------- leave-one-dataset-out CV ----------------
    lod = {}
    for gi, ds in enumerate(ds_names):
        te, tr = g == gi, g != gi
        row = {}
        for name in ("gbdt", "logreg", "little"):
            clf = fit_clf(name, F[tr], y[tr])
            pred = np.argmax(proba_clf(name, clf, F[te]), axis=1)
            acc = float((pred == y[te]).mean())
            rec = {CLASSES[c]: (float(((pred == c) & (y[te] == c)).sum()
                                      / max((y[te] == c).sum(), 1))
                                if (y[te] == c).any() else None)
                   for c in range(6)}
            row[name] = dict(acc=acc, recall=rec,
                             macro_recall=float(np.nanmean(
                                 [v for v in rec.values() if v is not None])))
        lod[ds] = row
        print(f"  [LOD] {ds}: " + "  ".join(
            f"{n} acc={row[n]['acc']:.3f}" for n in row), flush=True)
    out["lod"] = lod
    # pooled gbdt confusion (out-of-fold)
    conf = np.zeros((6, 6), int)
    for gi in range(3):
        te, tr = g == gi, g != gi
        clf = fit_clf("gbdt", F[tr], y[tr])
        pred = np.argmax(proba_clf("gbdt", clf, F[te]), axis=1)
        for a, b in zip(y[te], pred):
            conf[a, b] += 1
    out["lod_confusion_gbdt"] = conf.tolist()

    # ---------------- final models on all data ----------------
    final = {name: fit_clf(name, F, y) for name in ("gbdt", "logreg", "little")}
    with open(os.path.join(CKPT, "clf_final.pkl"), "wb") as fh:
        pickle.dump({"clfs": final, "feat_names": FEAT_NAMES,
                     "classes": CLASSES}, fh)

    # feature importances (gbdt, permutation on a subsample)
    from sklearn.inspection import permutation_importance
    sub = np.random.default_rng(SEED).choice(len(y), size=min(4000, len(y)),
                                             replace=False)
    pi = permutation_importance(final["gbdt"], F[sub], y[sub], n_repeats=3,
                                random_state=SEED)
    out["gbdt_feat_importance"] = dict(zip(
        FEAT_NAMES, [float(v) for v in pi.importances_mean]))
    res["trackA"] = out
    save_results(res)
    return final


def labels_for_windows(clf, name, feats):
    """feats: list of [C,14] per window -> per-window predicted class via
    mean channel probability argmax; also per-channel predictions."""
    wl, cp = [], []
    for F in feats:
        p = proba_clf(name, clf, F).mean(axis=0)
        wl.append(int(np.argmax(p)))
        cp.append(int(np.argmax(p)))
    return np.array(wl, np.int8)


def eval_labels_synthetic(clf, name="gbdt"):
    """Predicted window labels on the S12 (Eval A) and S5/S6 (Eval B) eval
    grids, plus window-level accuracy vs the known config mechanism.
    Eval A masks: s5.make_mask(mech, rate, wi, 0, C) (== s12 stream, verified
    by run_s12_recon tiny assert). Eval B mcar/block use ms=0 features while
    stored MSEs average ms in {0,1} (paired; disclosed)."""
    cache = os.path.join(CKPT, f"labels_syn_{name}.npz")
    acc = {}
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}, None
    store = {}
    for ds in s5.DATASETS:
        X300, st300 = s5.load_windows(s5.DATASETS[ds], 300, s5.SEED)
        X150, st150 = s5.load_windows(s5.DATASETS[ds], 150, s5.SEED)
        C = X300.shape[1]
        for grid, X, starts, mechs in (("A", X150, st150, ("mcar", "block")),
                                       ("B", X300, st300, ("mcar", "block",
                                                           "mnar_high",
                                                           "mnar_extreme"))):
            for mech in mechs:
                for rate in RATES:
                    t0 = time.time()
                    wl = []
                    for wi, s in enumerate(starts):
                        x = X[s:s + L].T.copy()
                        m = s5.make_mask(mech, rate, wi, 0, C, x=x)
                        F = window_features(x, m)
                        p = proba_clf(name, clf, F).mean(axis=0)
                        wl.append(int(np.argmax(p)))
                    wl = np.array(wl, np.int8)
                    store[f"{grid}|{ds}|{mech}|{rate}"] = wl
                    true = CLASSES.index(mech)
                    acc[f"{grid}|{ds}|{mech}|{rate}"] = dict(
                        acc=float((wl == true).mean()), n=len(wl))
                    print(f"  [syn-lab] {grid} {ds} {mech} {rate}: "
                          f"acc={acc[f'{grid}|{ds}|{mech}|{rate}']['acc']:.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(cache, **store)
    return store, acc


# ---------------------------------------------------------- real transfer ----
def penn_windows():
    import run_s10_real as s10
    windows, A = s10.load_all()
    return s10, windows, A


def metrla_windows():
    import run_s14_metrla as s14
    windows = s14.build_windows()
    A = s14.extract(windows)
    return s14, windows, A


def confusion_from(pred, true, ncls=6):
    conf = np.zeros((ncls, ncls), int)
    for a, b in zip(true, pred):
        if a >= 0 and b >= 0:
            conf[a, b] += 1
    return conf


V1_MAP = {"upper": MNAR_H, "two-tail": MNAR_E, "clustered": BLOCK,
          "random": MCAR, "clean": CLEAN, "undetermined": CLEAN}


def run_transfer(clf, res, name="gbdt"):
    out = {}
    # ---- Penmanshiel: oracle mask + status-free auto mask ----
    s10, windows, A = penn_windows()
    ids_c = s10.group_ids(windows, "cens")
    ids_t = s10.group_ids(windows, "ctrl")
    for mode in ("oracle", "auto"):
        feats, true = [], []
        for i in np.concatenate([ids_c, ids_t]):
            m = (A["mask"][i] if mode == "oracle"
                 else s10.candidate_mask(A["ctx_rec"][i], A["avail"][i]))
            feats.append(channel_features(A["ctx_rec"][i], m))
            true.append(MNAR_H if i in set(ids_c) else CLEAN)
        F = np.stack(feats)
        pred = np.argmax(proba_clf(name, clf, F), axis=1)
        true = np.array(true)
        conf = confusion_from(pred, true)
        rec = {CLASSES[c]: (float(conf[c, c] / max(conf[c].sum(), 1)))
               for c in (CLEAN, MNAR_H)}
        # v1 rule detector on the same windows/masks
        v1 = []
        for k, i in enumerate(np.concatenate([ids_c, ids_t])):
            m = (A["mask"][i] if mode == "oracle"
                 else s10.candidate_mask(A["ctx_rec"][i], A["avail"][i]))
            xf = A["ctx_rec"][i].copy()
            ok = np.isfinite(xf)
            if ok.sum() >= 2:
                xf = np.interp(np.arange(len(xf)), np.flatnonzero(ok), xf[ok])
            lab, _ = s10.detect_mechanism(np.nan_to_num(xf), m)
            v1.append(V1_MAP[lab])
        v1 = np.array(v1)
        v1cens = float(np.isin(v1[true == MNAR_H], [MNAR_H, MNAR_E]).mean())
        key = f"penn_{mode}"
        out[key] = dict(
            n_cens=len(ids_c), n_ctrl=len(ids_t),
            confusion=conf.tolist(),
            acc_cens=float((pred[true == MNAR_H] == MNAR_H).mean()),
            acc_clean=float((pred[true == CLEAN] == CLEAN).mean()),
            pred_frac={CLASSES[c]: float((pred == c).mean()) for c in range(6)},
            v1_cens_recall=v1cens,
            v1_ctrl_clean=float((v1[true == CLEAN] == CLEAN).mean()),
        )
        # save predicted labels (window-index aligned) for Track B
        lab_full = np.full(len(windows), -1, np.int8)
        for k, i in enumerate(np.concatenate([ids_c, ids_t])):
            lab_full[i] = pred[k]
        np.savez(os.path.join(CKPT, f"labels_penn_{mode}_{name}.npz"),
                 labels=lab_full)
        print(f"  [transfer] {key}: cens->mnar_high {out[key]['acc_cens']:.3f}, "
              f"ctrl->clean {out[key]['acc_clean']:.3f}, v1 cens-recall "
              f"{v1cens:.3f}", flush=True)
    # ---- METR-LA: zero-encoded outages ----
    s14, windows, A = metrla_windows()
    ids_m = s14.group_ids(windows, "miss")
    ids_t = s14.group_ids(windows, "ctrl")
    feats, true = [], []
    for i in np.concatenate([ids_m, ids_t]):
        feats.append(channel_features(A["ctx_rec"][i], A["mask"][i]))
        true.append(BLOCK if i in set(ids_m) else CLEAN)
    F = np.stack(feats)
    pred = np.argmax(proba_clf(name, clf, F), axis=1)
    true = np.array(true)
    conf = confusion_from(pred, true)
    v1 = []
    for i in np.concatenate([ids_m, ids_t]):
        lab, _ = s14.detect_mechanism(A["ctx_rec"][i], A["mask"][i])
        v1.append(V1_MAP[lab])
    v1 = np.array(v1)
    out["metrla"] = dict(
        n_miss=len(ids_m), n_ctrl=len(ids_t),
        confusion=conf.tolist(),
        acc_block=float((pred[true == BLOCK] == BLOCK).mean()),
        acc_clean=float((pred[true == CLEAN] == CLEAN).mean()),
        pred_frac={CLASSES[c]: float((pred == c).mean()) for c in range(6)},
        v1_block_recall=float((v1[true == BLOCK] == BLOCK).mean()),
        v1_ctrl_clean=float((v1[true == CLEAN] == CLEAN).mean()),
    )
    lab_full = np.full(len(windows), -1, np.int8)
    for k, i in enumerate(np.concatenate([ids_m, ids_t])):
        lab_full[i] = pred[k]
    np.savez(os.path.join(CKPT, f"labels_metrla_{name}.npz"), labels=lab_full)
    print(f"  [transfer] metrla: miss->block {out['metrla']['acc_block']:.3f}, "
          f"ctrl->clean {out['metrla']['acc_clean']:.3f}, v1 block-recall "
          f"{out['metrla']['v1_block_recall']:.3f}", flush=True)
    res.setdefault("trackA", {})["transfer"] = out
    save_results(res)
    return out


# ------------------------------------------------------------- anchor gate ----
def run_gate(res, device):
    """Re-infer 3 store cells with bolt; compare within +/-5%."""
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map=device, torch_dtype=torch.float32)

    @torch.no_grad()
    def bolt_point(ctx, batch=1024):  # [n, L] -> [n, H] median
        outs = []
        for i in range(0, len(ctx), batch):
            xb = torch.from_numpy(np.ascontiguousarray(ctx[i:i + batch])).to(device)
            q = pipe.predict(xb, prediction_length=H)
            outs.append(q[:, 4, :].float().cpu().numpy())
        return np.concatenate(outs)

    cells = []

    def check(name, stored, got, tol=0.05):
        rel = abs(got - stored) / max(abs(stored), 1e-12)
        cells.append(dict(name=name, stored=float(stored), recomputed=float(got),
                          rel_diff=float(rel), passed=bool(rel <= tol)))
        print(f"  [gate] {name}: stored={stored:.5f} rerun={got:.5f} "
              f"diff={rel*100:.2f}% {'OK' if rel <= tol else 'FAIL'}", flush=True)

    # cell 1: s12 ETTh1 mcar:zs:linear:0.3 (150 windows, mask ms=0)
    d12 = json.load(open(os.path.join(HERE, "s12_results.json")))
    X, starts = s5.load_windows(s5.DATASETS["ETTh1"], 150, s5.SEED)
    C = X.shape[1]
    ctxs, gts = [], []
    for wi, s in enumerate(starts):
        x = X[s:s + L].T.copy()
        m = s5.make_mask("mcar", 0.3, wi, 0, C, x=x)
        ctxs.append(s5.fill_context(x, m, "linear"))
        gts.append(X[s + L:s + L + H].T.copy())
    S = len(ctxs)
    ctx = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
    pred = bolt_point(ctx).reshape(S, C, H).astype(np.float64)
    mse = ((pred - np.stack(gts)) ** 2).mean(axis=(1, 2))
    check("s12|ETTh1|mcar:zs:linear:0.3",
          d12["eval"]["ETTh1"]["mcar:zs:linear:0.3"]["mse"], float(mse.mean()))

    # cell 2: s5 ETTm1 mnar_high:linear:0.5 (300 windows, deterministic)
    d5 = json.load(open(os.path.join(HERE, "s5_missing_results.json")))
    X, starts = s5.load_windows(s5.DATASETS["ETTm1"], 300, s5.SEED)
    C = X.shape[1]
    ctxs, gts = [], []
    for wi, s in enumerate(starts):
        x = X[s:s + L].T.copy()
        m = s5.make_mask("mnar_high", 0.5, wi, 0, C, x=x)
        ctxs.append(s5.fill_context(x, m, "linear"))
        gts.append(X[s + L:s + L + H].T.copy())
    S = len(ctxs)
    ctx = np.stack(ctxs).reshape(S * C, L).astype(np.float32)
    pred = bolt_point(ctx).reshape(S, C, H).astype(np.float64)
    mse = ((pred - np.stack(gts)) ** 2).mean(axis=(1, 2))
    check("s5|ETTm1|mnar_high:linear:0.5",
          d5["bolt"]["ETTm1"]["mnar_high:linear:0.5"]["mse"], float(mse.mean()))

    # cell 3: s10 bolt cens zero (602 real curtailed windows, NMSE)
    import run_s10_real as s10
    windows, A = s10.load_all()
    ids = s10.group_ids(windows, "cens")
    Xz = s10.build_X("zero", ids, A)
    outs = []
    for i in range(0, len(Xz), 1024):
        xb = torch.from_numpy(Xz[i:i + 1024]).to(device)
        q = pipe.predict(xb, prediction_length=s10.H)
        outs.append(q[:, 4, :].float().cpu().numpy())
    pred = np.concatenate(outs).astype(np.float64)
    Y = A["tgt"][ids].astype(np.float64)
    mse_w = ((pred - Y) ** 2).mean(axis=1)
    nmse = float((mse_w / np.maximum(Y.var(axis=1), s10.VAR_FLOOR)).mean())
    d10 = json.load(open(os.path.join(HERE, "s10_real_results.json")))["records"]
    check("s10|bolt|cens|zero", d10["anchor|bolt|cens|zero"]["nmse"], nmse)

    passed = all(c["passed"] for c in cells)
    print(f"GATE {'PASS' if passed else 'FAIL'}", flush=True)
    res["gate"] = dict(passed=bool(passed), tol=0.05, cells=cells)
    save_results(res)
    del pipe
    torch.cuda.empty_cache()
    return passed


# ------------------------------------------------------------ io helpers ----
def save_results(res, path=OUT):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
    os.replace(tmp, path)


def load_results(path=OUT):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


# ============================================================== Track B =====
def _lab_get(labels, grid, ds, mech, rate):
    return labels[f"{grid}|{ds}|{mech}|{rate}"].astype(int)


def run_gate_eval_a(res, labels):
    """Eval A: s12 store (bolt, 150 windows, mcar/block). Candidates:
    zs:linear, zs:nan, tok:nan, mb:nan, recon:nan. relMSE per window vs the
    variant's OWN clean (tok uses zs clean -- clean:tok not stored; tok==zs on
    mcar to 3 decimals so clean:tok ~= clean:zs)."""
    d12 = json.load(open(os.path.join(HERE, "s12_results.json")))
    cands = ["zs:linear", "zs:nan", "tok:nan", "mb:nan", "recon:nan"]
    own_clean = {"zs": "clean:zs", "tok": "clean:zs", "mb": "clean:mb",
                 "recon": "clean:recon"}
    rel = {}   # (ds, mech, rate, strat) -> [150] relMSE
    for ds in s5.DATASETS:
        cleans = {v: np.array(d12["clean"][ds][k]["mse_per_window"])
                  for v, k in own_clean.items()}
        for mech in ("mcar", "block"):
            for rate in RATES:
                for st in cands:
                    v = st.split(":")[0]
                    mse = np.array(d12["eval"][ds][f"{mech}:{st}:{rate}"]
                                   ["mse_per_window"])
                    rel[(ds, mech, rate, st)] = mse / cleans[v]
    # policy tables (prior-round aggregate evidence, printed for the record)
    agg = {}
    for mech in ("mcar", "block"):
        for st in cands:
            agg[(mech, st)] = float(np.mean([
                rel[(ds, mech, r, st)].mean() for ds in s5.DATASETS
                for r in RATES]))
    out = {"candidates": cands, "aggregate_relMSE": {f"{m}|{s}": v
                                                     for (m, s), v in agg.items()}}
    pol_zs = {"mcar": "zs:linear", "block": "zs:nan"}        # fills only (S5-fix)
    pol_full = {"mcar": min(cands, key=lambda s: agg[("mcar", s)]),
                "block": min(cands, key=lambda s: agg[("block", s)])}
    out["policy_fills"] = pol_zs
    out["policy_full"] = pol_full
    print(f"  [evalA] aggregate relMSE: "
          + "; ".join(f"{m}:{s}={agg[(m, s)]:.3f}" for m in ("mcar", "block")
                      for s in cands), flush=True)

    def route(lab, pol):
        return pol.get(CLASSES[lab], "zs:linear")  # default fallback linear

    cells = {}
    for ds in s5.DATASETS:
        for mech in ("mcar", "block"):
            for rate in RATES:
                lab = _lab_get(labels, "A", ds, mech, rate)
                per = {st: rel[(ds, mech, rate, st)] for st in cands}
                n = len(lab)
                g_zs = np.array([per[route(l, pol_zs)][i]
                                 for i, l in enumerate(lab)])
                g_full = np.array([per[route(l, pol_full)][i]
                                   for i, l in enumerate(lab)])
                M = np.stack([per[st] for st in cands])       # [5, n]
                oracle_all = M.min(axis=0)
                i_zs = [cands.index(s) for s in ("zs:linear", "zs:nan")]
                oracle_fills = M[i_zs].min(axis=0)
                i_zst = [cands.index(s) for s in
                         ("zs:linear", "zs:nan", "tok:nan")]
                oracle_zstok = M[i_zst].min(axis=0)
                cells[f"{ds}|{mech}|{rate}"] = {
                    **{f"fixed|{st}": float(per[st].mean()) for st in cands},
                    "gated_fills": float(g_zs.mean()),
                    "gated_full": float(g_full.mean()),
                    "oracle_fills2": float(oracle_fills.mean()),
                    "oracle_zstok": float(oracle_zstok.mean()),
                    "oracle_all": float(oracle_all.mean()),
                    "oracle_match_rate_gatedfull": float(
                        (g_full <= np.min(M, axis=0) + 1e-12).mean()),
                }
    out["cells"] = cells
    # pooled summary per mechanism
    summ = {}
    for mech in ("mcar", "block"):
        arms = [f"fixed|{st}" for st in cands] + [
            "gated_fills", "gated_full", "oracle_fills2", "oracle_zstok",
            "oracle_all"]
        row = {}
        for a in arms:
            row[a] = float(np.mean([cells[f"{ds}|{mech}|{r}"][a]
                                    for ds in s5.DATASETS for r in RATES]))
        row["regret_gatedfull_vs_oracleall"] = row["gated_full"] - row["oracle_all"]
        row["regret_gatedfills_vs_oraclefills"] = (row["gated_fills"]
                                                   - row["oracle_fills2"])
        best_fixed = min(f"fixed|{st}" for st in cands)
        wins = [cells[f"{ds}|{mech}|{r}"]["gated_full"]
                <= cells[f"{ds}|{mech}|{r}"][best_fixed] + 1e-12
                for ds in s5.DATASETS for r in RATES]
        row["best_fixed_arm"] = best_fixed
        row["gatedfull_cell_winrate_vs_bestfixed"] = float(np.mean(wins))
        summ[mech] = row
    out["summary"] = summ
    for mech in summ:
        s = summ[mech]
        print(f"  [evalA:{mech}] gated_full={s['gated_full']:.4f} "
              f"oracle_all={s['oracle_all']:.4f} regret={s['regret_gatedfull_vs_oracleall']:.4f} "
              f"best_fixed={s['best_fixed_arm']}={s[s['best_fixed_arm']]:.4f} "
              f"winrate={s['gatedfull_cell_winrate_vs_bestfixed']:.2f}", flush=True)
    res.setdefault("trackB", {})["evalA"] = out
    save_results(res)
    return out


def run_gate_eval_b(res, labels):
    """Eval B: s5 + s6_mnarfix stores (bolt, 300 windows, mcar/block/mnar).
    Candidates: zero, ffill, linear, zero_oscale (mcar/mnar), tail_tobit
    (mnar). relMSE vs s5 clean:none:0.0 per window."""
    d5 = json.load(open(os.path.join(HERE, "s5_missing_results.json")))["bolt"]
    d6 = json.load(open(os.path.join(HERE, "s6_mnarfix_results.json")))["bolt"]
    mechs = ("mcar", "block", "mnar_high", "mnar_extreme")
    base_fills = ["zero", "ffill", "linear"]
    rel = {}
    for ds in s5.DATASETS:
        clean = np.array(d5[ds]["clean:none:0.0"]["mse_per_window"])
        for mech in mechs:
            for rate in RATES:
                for f in base_fills + ["zero_oscale"]:
                    k = f"{mech}:{f}:{rate}"
                    if k in d5[ds]:
                        rel[(ds, mech, rate, f)] = (np.array(
                            d5[ds][k]["mse_per_window"]) / clean)
                if mech.startswith("mnar"):
                    k = f"{mech}:tail_tobit:{rate}"
                    rel[(ds, mech, rate, "tail_tobit")] = (np.array(
                        d6[ds][k]["mse_per_window"]) / clean)
    out = {}
    cells = {}
    for ds in s5.DATASETS:
        for mech in mechs:
            for rate in RATES:
                lab = _lab_get(labels, "B", ds, mech, rate)
                cands = [f for f in ("zero", "ffill", "linear", "zero_oscale",
                                     "tail_tobit")
                         if (ds, mech, rate, f) in rel]
                per = {f: rel[(ds, mech, rate, f)] for f in cands}
                M = np.stack([per[f] for f in cands])
                oracle = M.min(axis=0)
                pol_lin = per["linear"]
                # gated-tobit: predicted mnar -> tail_tobit, else linear
                if "tail_tobit" in per:
                    g_t = np.array([per["tail_tobit"][i]
                                    if CLASSES[l] in ("mnar_high",
                                                      "mnar_extreme")
                                    else per["linear"][i]
                                    for i, l in enumerate(lab)])
                else:
                    g_t = per["linear"].copy()
                cells[f"{ds}|{mech}|{rate}"] = {
                    **{f"fixed|{f}": float(per[f].mean()) for f in cands},
                    "gated_linear": float(pol_lin.mean()),
                    "gated_tobit": float(g_t.mean()),
                    "oracle": float(oracle.mean()),
                }
    out["cells"] = cells
    summ = {}
    for mech in mechs:
        arms = set()
        for ds in s5.DATASETS:
            for r in RATES:
                arms |= {k for k in cells[f"{ds}|{mech}|{r}"]}
        row = {}
        for a in sorted(arms):
            vals = [cells[f"{ds}|{mech}|{r}"][a] for ds in s5.DATASETS
                    for r in RATES if a in cells[f"{ds}|{mech}|{r}"]]
            row[a] = float(np.mean(vals))
        row["regret_gatedlinear_vs_oracle"] = row["gated_linear"] - row["oracle"]
        row["regret_gatedtobit_vs_oracle"] = row["gated_tobit"] - row["oracle"]
        fixed_arms = [a for a in arms if a.startswith("fixed|")]
        best_fixed = min(fixed_arms, key=lambda a: row[a])
        wins = [cells[f"{ds}|{mech}|{r}"]["gated_linear"]
                <= cells[f"{ds}|{mech}|{r}"][best_fixed] + 1e-12
                for ds in s5.DATASETS for r in RATES]
        row["best_fixed_arm"] = best_fixed
        row["gatedlinear_cell_winrate_vs_bestfixed"] = float(np.mean(wins))
        summ[mech] = row
    out["summary"] = summ
    for mech in summ:
        s = summ[mech]
        print(f"  [evalB:{mech}] linear={s['gated_linear']:.4f} "
              f"tobit-gated={s['gated_tobit']:.4f} oracle={s['oracle']:.4f} "
              f"best_fixed={s['best_fixed_arm']}={s[s['best_fixed_arm']]:.4f}",
              flush=True)
    res.setdefault("trackB", {})["evalB"] = out
    save_results(res)
    return out


# ------------------------------------------- Mondrian conformal (synthetic) --
class Bolt96:
    """chronos-bolt-base, L=512/H=96, point + native [q10,q90]."""
    name, batch = "bolt", 1024

    def __init__(self, device):
        from chronos import BaseChronosPipeline
        self.pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map=device,
            torch_dtype=torch.float32)
        self.device = device

    @torch.no_grad()
    def predict_quantiles(self, ctx):
        q10, q90 = [], []
        for i in range(0, len(ctx), self.batch):
            xb = torch.from_numpy(np.ascontiguousarray(ctx[i:i + self.batch])
                                  ).to(self.device)
            q = self.pipe.predict(xb, prediction_length=H)
            q10.append(q[:, 0, :].float().cpu().numpy())
            q90.append(q[:, 8, :].float().cpu().numpy())
        return np.concatenate(q10), np.concatenate(q90)


def conformal_w(E, alpha=0.9):
    """E: [n, H] nonconformity -> per-step widening [H] (split conformal)."""
    n = E.shape[0]
    k = min(int(np.ceil((n + 1) * alpha)), n)
    return np.sort(E, axis=0)[k - 1]


def cov_wid(q10, q90, y, w):
    inside = (y >= q10 - w) & (y <= q90 + w)
    return float(inside.mean()), float((q90 - q10 + 2 * w).mean())


def routed_fill_synth(x1d, mask, pred_lab):
    """Deployment fill for the synthetic Mondrian stream (bolt):
    mcar/mnar -> linear; block -> native NaN; clean/intermittent -> no repair
    (NaN-native if a mask is nevertheless present = the missed-detection cost)."""
    cls = CLASSES[pred_lab]
    x = x1d.astype(np.float64)
    if cls in ("mcar", "mnar_high", "mnar_extreme"):
        valid = np.flatnonzero(~mask)
        if mask.any() and len(valid) >= 2:
            return np.interp(np.arange(len(x)), valid, x[valid])
        return x.copy()
    out = x.copy()
    out[mask] = np.nan
    return out


def run_conf_syn(res, clf, device, name="gbdt"):
    """Mixed-mechanism deployment stream; split-conformal grouped by
    predicted class (deployable Mondrian) vs naive global vs oracle class."""
    import run_s6_conformal as s6c
    model = Bolt96(device)
    mix = [(CLEAN, 0.25), (MCAR, 0.30), (BLOCK, 0.20), (MNAR_H, 0.15),
           (MNAR_E, 0.10)]
    mprobs = np.array([p for _, p in mix])
    MIN_GROUP = 30
    out = {}
    for ds in s5.DATASETS:
        t0 = time.time()
        X, ev_starts = s5.load_windows(s5.DATASETS[ds], 300, s5.SEED)
        _, ca_starts = s6c.calib_starts(ds)
        C = X.shape[1]
        rec = {"n_cal": len(ca_starts), "n_test": len(ev_starts), "C": C}
        splits = {}
        for sid, (tag, starts) in enumerate((("cal", ca_starts),
                                             ("test", ev_starts))):
            ctxs, ys, tlabs, plabs = [], [], [], []
            for wi, s in enumerate(starts):
                xw = X[s:s + L].T.copy()
                yw = X[s + L:s + L + H].T.copy()
                for c in range(C):
                    rng = np.random.default_rng(np.random.SeedSequence(
                        [SEED, 31 + sid, wi, c]))
                    mi = int(rng.choice(len(mix), p=mprobs))
                    tlab = mix[mi][0]
                    rate = float(rng.uniform(0.1, 0.6))
                    if tlab == CLEAN:
                        m = np.zeros(L, bool)
                    elif tlab == MCAR:
                        m = s5.make_mask("mcar", rate, wi, 5000 + sid * 997 + c,
                                         1)[0]
                    elif tlab == BLOCK:
                        m = block_mask_1d(rate, L, rng, 24)
                    else:
                        mech = "mnar_high" if tlab == MNAR_H else "mnar_extreme"
                        m = s5.make_mask(mech, rate, wi, 0, 1,
                                         x=xw[c][None, :])[0]
                    f1 = channel_features(xw[c], m)
                    plab = int(np.argmax(proba_clf(name, clf, f1[None])[0]))
                    ctxs.append(routed_fill_synth(xw[c], m, plab))
                    ys.append(yw[c])
                    tlabs.append(tlab)
                    plabs.append(plab)
            X_in = np.stack(ctxs).astype(np.float32)
            q10, q90 = model.predict_quantiles(X_in)
            splits[tag] = dict(q10=q10.astype(np.float64),
                               q90=q90.astype(np.float64),
                               y=np.stack(ys).astype(np.float64),
                               tlab=np.array(tlabs), plab=np.array(plabs))
        ca, te = splits["cal"], splits["test"]
        E_ca = np.maximum(ca["q10"] - ca["y"], ca["y"] - ca["q90"])
        w_naive = conformal_w(E_ca)
        # pools
        pool_pred = {c: conformal_w(E_ca[ca["plab"] == c])
                     if (ca["plab"] == c).sum() >= MIN_GROUP else None
                     for c in range(6)}
        pool_true = {c: conformal_w(E_ca[ca["tlab"] == c])
                     if (ca["tlab"] == c).sum() >= MIN_GROUP else None
                     for c in range(6)}
        rec["cal_group_counts_pred"] = {CLASSES[c]: int((ca["plab"] == c).sum())
                                        for c in range(6)}
        rec["cal_group_counts_true"] = {CLASSES[c]: int((ca["tlab"] == c).sum())
                                        for c in range(6)}
        # deploy: per test series pick w by variant
        variants = ("native", "naive", "mondrian_pred", "mondrian_oracle")
        rows = {}
        for tcls in range(6):
            sel = te["tlab"] == tcls
            if sel.sum() == 0:
                continue
            q10, q90, y = te["q10"][sel], te["q90"][sel], te["y"][sel]
            pl = te["plab"][sel]
            row = {}
            for v in variants:
                if v == "native":
                    w = np.zeros(H)
                    cov, wid = cov_wid(q10, q90, y, w)
                elif v == "naive":
                    cov, wid = cov_wid(q10, q90, y, w_naive)
                else:
                    pool = pool_pred if v == "mondrian_pred" else pool_true
                    key = pl if v == "mondrian_pred" else np.full(sel.sum(),
                                                                  tcls)
                    covs, wids = [], []
                    for c in range(6):
                        ss = key == c
                        if ss.sum() == 0:
                            continue
                        w = pool[c] if pool[c] is not None else w_naive
                        cv, wd = cov_wid(q10[ss], q90[ss], y[ss], w)
                        covs.append((cv, ss.sum()))
                        wids.append((wd, ss.sum()))
                    cov = float(np.average([a for a, n in covs],
                                           weights=[n for _, n in covs]))
                    wid = float(np.average([a for a, n in wids],
                                           weights=[n for _, n in wids]))
                row[v] = dict(coverage=cov, width=wid)
            rows[CLASSES[tcls]] = row
        # marginal (all test series)
        row = {}
        for v in variants:
            if v == "native":
                cov, wid = cov_wid(te["q10"], te["q90"], te["y"], np.zeros(H))
            elif v == "naive":
                cov, wid = cov_wid(te["q10"], te["q90"], te["y"], w_naive)
            else:
                pool = pool_pred if v == "mondrian_pred" else pool_true
                key = te["plab"] if v == "mondrian_pred" else te["tlab"]
                covs, wids = [], []
                for c in range(6):
                    ss = key == c
                    if ss.sum() == 0:
                        continue
                    w = pool[c] if pool[c] is not None else w_naive
                    cv, wd = cov_wid(te["q10"][ss], te["q90"][ss], te["y"][ss], w)
                    covs.append((cv, ss.sum()))
                    wids.append((wd, ss.sum()))
                cov = float(np.average([a for a, n in covs],
                                       weights=[n for _, n in covs]))
                wid = float(np.average([a for a, n in wids],
                                       weights=[n for _, n in wids]))
            row[v] = dict(coverage=cov, width=wid)
        rows["ALL"] = row
        rec["rows"] = rows
        out[ds] = rec
        print(f"  [conf_syn] {ds} ({time.time()-t0:.0f}s):", flush=True)
        for k, row in rows.items():
            print(f"    {k:12s} " + "  ".join(
                f"{v}:{row[v]['coverage']:.3f}" for v in variants), flush=True)
    res.setdefault("trackB", {})["conf_syn"] = out
    save_results(res)
    del model
    torch.cuda.empty_cache()
    return out


# ------------------------------------- Mondrian conformal (real datasets) ----
def _mondrian_pools(E_cal, key_cal, pos_mask_cal, min_group=30):
    """Pools keyed by a boolean (positive/negative) split of cal windows."""
    out = {}
    if pos_mask_cal.sum() >= min_group:
        out["pos"] = conformal_w(E_cal[pos_mask_cal])
    if (~pos_mask_cal).sum() >= min_group:
        out["neg"] = conformal_w(E_cal[~pos_mask_cal])
    return out


def _apply_pool(q10, q90, y, pos_mask_test, pools, w_global):
    covs, wids = [], []
    for tag, sel in (("pos", pos_mask_test), ("neg", ~pos_mask_test)):
        if sel.sum() == 0:
            continue
        w = pools.get(tag, w_global)
        cv, wd = cov_wid(q10[sel], q90[sel], y[sel], w)
        covs.append((cv, int(sel.sum())))
        wids.append((wd, int(sel.sum())))
    cov = float(np.average([a for a, n in covs], weights=[n for _, n in covs]))
    wid = float(np.average([a for a, n in wids], weights=[n for _, n in wids]))
    return dict(coverage=cov, width=wid)


def run_conf_real(res, device, name="gbdt"):
    """Mondrian-by-detected-class conformal on the two real datasets, reusing
    the exact S10/S14 cal/test splits. bolt only. Grouping: predicted class
    (deployable) vs naive global vs oracle class (upper bound)."""
    out = {}
    # ---------------- Penmanshiel (CTX=144, fills zero/linear) ----------------
    import run_s10_real as s10
    lab = np.load(os.path.join(CKPT, f"labels_penn_oracle_{name}.npz"))["labels"]
    windows, A = s10.load_all()
    model = s10.BoltQ(device)
    cal_c, test_c = s10.cal_test_split(windows, "cens")
    cal_t, test_t = s10.cal_test_split(windows, "ctrl")
    pos = np.isin(lab, [MNAR_H, MNAR_E])     # predicted censoring
    d10 = json.load(open(os.path.join(HERE, "s10_real_results.json")))["records"]
    rec_ds = {}
    for fill in ("zero", "linear"):
        splits = {}
        for tag, ids in (("cens_cal", cal_c), ("cens_test", test_c),
                         ("ctrl_cal", cal_t), ("ctrl_test", test_t)):
            X = s10.build_X(fill, ids, A)
            q10, q90 = model.predict_quantiles(X)
            splits[tag] = dict(q10=q10.astype(np.float64),
                               q90=q90.astype(np.float64),
                               y=A["tgt"][ids].astype(np.float64), ids=ids)
        cal_ids = np.concatenate([cal_c, cal_t])
        E_cal = np.concatenate([np.maximum(splits[t]["q10"] - splits[t]["y"],
                                           splits[t]["y"] - splits[t]["q90"])
                                for t in ("cens_cal", "ctrl_cal")])
        pos_cal = pos[cal_ids]
        w_naive = conformal_w(E_cal)
        pools_pred = _mondrian_pools(E_cal, None, pos_cal)
        w_cens = conformal_w(np.maximum(splits["cens_cal"]["q10"]
                                        - splits["cens_cal"]["y"],
                                        splits["cens_cal"]["y"]
                                        - splits["cens_cal"]["q90"]))
        w_ctrl = conformal_w(np.maximum(splits["ctrl_cal"]["q10"]
                                        - splits["ctrl_cal"]["y"],
                                        splits["ctrl_cal"]["y"]
                                        - splits["ctrl_cal"]["q90"]))
        rec_fill = {}
        for deploy, true_kind in (("cens_test", "cens"), ("ctrl_test", "ctrl")):
            sp = splits[deploy]
            pos_te = pos[sp["ids"]]
            row = {"native": dict(zip(("coverage", "width"),
                                      cov_wid(sp["q10"], sp["q90"], sp["y"],
                                              np.zeros(s10.H)))),
                   "naive": dict(zip(("coverage", "width"),
                                     cov_wid(sp["q10"], sp["q90"], sp["y"],
                                             w_naive)))}
            row["mondrian_pred"] = _apply_pool(sp["q10"], sp["q90"], sp["y"],
                                               pos_te, pools_pred, w_naive)
            pools_or = {"pos": w_cens, "neg": w_ctrl}
            true_pos = np.ones(len(sp["ids"]), bool) if true_kind == "cens" \
                else np.zeros(len(sp["ids"]), bool)
            row["mondrian_oracle"] = _apply_pool(sp["q10"], sp["q90"], sp["y"],
                                                 true_pos, pools_or, w_naive)
            row["n_test"] = len(sp["ids"])
            row["pred_pos_frac"] = float(pos_te.mean())
            rec_fill[deploy] = row
        # cross-anchor: mondrian_oracle on cens_test == s10 mechaware
        stored = d10[f"track2|bolt|{fill}|cens_test|mechaware"]["coverage"]
        rec_fill["crosscheck_oracle_vs_s10_mechaware"] = dict(
            stored=float(stored), mine=rec_fill["cens_test"]["mondrian_oracle"]["coverage"],
            abs_diff=abs(stored - rec_fill["cens_test"]["mondrian_oracle"]["coverage"]))
        rec_ds[fill] = rec_fill
        print(f"  [conf_real:penn:{fill}] cens_test "
              + "  ".join(f"{v}:{rec_fill['cens_test'][v]['coverage']:.3f}"
                          for v in ("native", "naive", "mondrian_pred",
                                    "mondrian_oracle")), flush=True)
    out["penn"] = rec_ds
    del model
    torch.cuda.empty_cache()

    # ---------------- METR-LA (CTX=512, fills linear/nan) ----------------
    import run_s14_metrla as s14
    lab = np.load(os.path.join(CKPT, f"labels_metrla_{name}.npz"))["labels"]
    windows = s14.build_windows()
    A = s14.extract(windows)
    model = s14.BoltModel(device)
    cal_m, test_m = s14.time_split(windows, "miss")
    cal_c, test_c = s14.time_split(windows, "ctrl")
    pos = lab == BLOCK                        # predicted sensor-outage block
    d14 = json.load(open(os.path.join(HERE, "s14_metrla_results.json")))["records"]
    rec_ds = {}
    for fill in ("linear", "nan"):
        splits = {}
        for tag, ids in (("miss_cal", cal_m), ("miss_test", test_m),
                         ("ctrl_cal", cal_c), ("ctrl_test", test_c)):
            X = s14.build_X(fill, ids, A)
            q10, q90 = model.predict_quantiles(X)
            splits[tag] = dict(q10=q10.astype(np.float64),
                               q90=q90.astype(np.float64),
                               y=A["tgt"][ids].astype(np.float64), ids=ids)
        cal_ids = np.concatenate([cal_m, cal_c])
        E_cal = np.concatenate([np.maximum(splits[t]["q10"] - splits[t]["y"],
                                           splits[t]["y"] - splits[t]["q90"])
                                for t in ("miss_cal", "ctrl_cal")])
        pos_cal = pos[cal_ids]
        w_naive = conformal_w(E_cal)
        pools_pred = _mondrian_pools(E_cal, None, pos_cal)
        w_miss = conformal_w(np.maximum(splits["miss_cal"]["q10"]
                                        - splits["miss_cal"]["y"],
                                        splits["miss_cal"]["y"]
                                        - splits["miss_cal"]["q90"]))
        w_ctrl = conformal_w(np.maximum(splits["ctrl_cal"]["q10"]
                                        - splits["ctrl_cal"]["y"],
                                        splits["ctrl_cal"]["y"]
                                        - splits["ctrl_cal"]["q90"]))
        rec_fill = {}
        for deploy, true_kind in (("miss_test", "miss"), ("ctrl_test", "ctrl")):
            sp = splits[deploy]
            pos_te = pos[sp["ids"]]
            row = {"native": dict(zip(("coverage", "width"),
                                      cov_wid(sp["q10"], sp["q90"], sp["y"],
                                              np.zeros(s14.H)))),
                   "naive": dict(zip(("coverage", "width"),
                                     cov_wid(sp["q10"], sp["q90"], sp["y"],
                                             w_naive)))}
            row["mondrian_pred"] = _apply_pool(sp["q10"], sp["q90"], sp["y"],
                                               pos_te, pools_pred, w_naive)
            pools_or = {"pos": w_miss, "neg": w_ctrl}
            true_pos = np.ones(len(sp["ids"]), bool) if true_kind == "miss" \
                else np.zeros(len(sp["ids"]), bool)
            row["mondrian_oracle"] = _apply_pool(sp["q10"], sp["q90"], sp["y"],
                                                 true_pos, pools_or, w_naive)
            row["n_test"] = len(sp["ids"])
            row["pred_pos_frac"] = float(pos_te.mean())
            rec_fill[deploy] = row
        stored = d14[f"conformal|bolt|{fill}|miss_test|mechaware"]["coverage"]
        rec_fill["crosscheck_oracle_vs_s14_mechaware"] = dict(
            stored=float(stored), mine=rec_fill["miss_test"]["mondrian_oracle"]["coverage"],
            abs_diff=abs(stored - rec_fill["miss_test"]["mondrian_oracle"]["coverage"]))
        rec_ds[fill] = rec_fill
        print(f"  [conf_real:metrla:{fill}] miss_test "
              + "  ".join(f"{v}:{rec_fill['miss_test'][v]['coverage']:.3f}"
                          for v in ("native", "naive", "mondrian_pred",
                                    "mondrian_oracle")), flush=True)
    out["metrla"] = rec_ds
    del model
    torch.cuda.empty_cache()
    res.setdefault("trackB", {})["conf_real"] = out
    save_results(res)
    return out


# ---------------------------------------------------- real end-to-end NMSE ----
def _rec_map(rec):
    return {int(i): v for i, v in zip(rec["window_ids"], rec["nmse_per_window"])}


def run_real_e2e(res, name="gbdt"):
    """Gated-pipeline NMSE on the two real datasets, assembled from the
    s10/s14 per-window stores (no new inference). Routing tables are
    per-domain (prior-round evidence):
      Penmanshiel: predicted censoring -> zero; clean/intermittent -> keep;
        mcar/block -> linear.
      METR-LA: predicted block -> nan (bolt/moirai) / linear (timesfm);
        clean/intermittent -> keep; mcar/mnar -> linear."""
    out = {}
    # ---- Penmanshiel ----
    d10 = json.load(open(os.path.join(HERE, "s10_real_results.json")))["records"]
    import run_s10_real as s10
    windows, A = s10.load_all()
    ids_c = s10.group_ids(windows, "cens")
    out_p = {}
    for model in ("bolt", "timesfm", "moirai"):
        fills = {f: _rec_map(d10[f"anchor|{model}|cens|{f}"])
                 for f in ("keep", "zero", "linear")}
        per = {f: np.array([fills[f][int(i)] for i in ids_c])
               for f in ("keep", "zero", "linear")}
        M = np.stack([per[f] for f in ("keep", "zero", "linear")])
        oracle = M.min(axis=0)
        rec_m = {"fixed|" + f: dict(mean=float(per[f].mean()),
                                    median=float(np.median(per[f])))
                 for f in ("keep", "zero", "linear")}
        rec_m["oracle"] = dict(mean=float(oracle.mean()),
                               median=float(np.median(oracle)))
        for labmode in ("oracle", "auto"):
            lab = np.load(os.path.join(CKPT, f"labels_penn_{labmode}_{name}.npz")
                          )["labels"]
            pl = lab[ids_c]
            route = np.where(np.isin(pl, [MNAR_H, MNAR_E]), 1,  # zero
                             np.where(np.isin(pl, [CLEAN, INTER]), 0, 2))
            gated = np.array([M[route[k], k] for k in range(len(ids_c))])
            rec_m[f"gated_{labmode}mask"] = dict(
                mean=float(gated.mean()), median=float(np.median(gated)),
                regret_vs_oracle=float(gated.mean() - oracle.mean()),
                oracle_match_rate=float((gated <= oracle + 1e-12).mean()))
        best_fixed = min(("keep", "zero", "linear"),
                         key=lambda f: per[f].mean())
        g = rec_m["gated_oraclemask"]["mean"]
        gated_pw = np.array([M[np.where(np.isin(lab[ids_c], [MNAR_H, MNAR_E]), 1,
                                np.where(np.isin(lab[ids_c], [CLEAN, INTER]),
                                         0, 2))[k], k]
                             for k in range(len(ids_c))])
        rec_m["gated_vs_bestfixed"] = dict(
            best_fixed=best_fixed,
            best_fixed_mean=rec_m[f"fixed|{best_fixed}"]["mean"],
            gated_mean=g,
            per_window_winrate=float((gated_pw <= per[best_fixed] + 1e-12).mean()))
        out_p[model] = rec_m
        print(f"  [e2e:penn:{model}] gated={g:.3f} oracle={rec_m['oracle']['mean']:.3f} "
              f"best_fixed={best_fixed}={rec_m[f'fixed|{best_fixed}']['mean']:.3f}",
              flush=True)
    out["penn"] = out_p
    # ---- METR-LA ----
    d14 = json.load(open(os.path.join(HERE, "s14_metrla_results.json")))["records"]
    import run_s14_metrla as s14
    windows = s14.build_windows()
    ids_m = s14.group_ids(windows, "miss")
    lab = np.load(os.path.join(CKPT, f"labels_metrla_{name}.npz"))["labels"]
    out_m = {}
    for model in ("bolt", "timesfm", "moirai"):
        fills = {f: _rec_map(d14[f"anchor|{model}|miss|{f}"])
                 for f in ("keep", "zero", "linear", "nan")}
        order = ("keep", "zero", "linear", "nan")
        per = {f: np.array([fills[f][int(i)] for i in ids_m]) for f in order}
        M = np.stack([per[f] for f in order])
        oracle = M.min(axis=0)
        rec_m = {"fixed|" + f: dict(mean=float(per[f].mean()),
                                    median=float(np.median(per[f])))
                 for f in order}
        rec_m["oracle"] = dict(mean=float(oracle.mean()),
                               median=float(np.median(oracle)))
        pl = lab[ids_m]
        # per-domain table (S14 aggregate evidence): bolt -> nan (2.00<2.75);
        # timesfm -> linear (nan==linear internally); moirai -> linear
        # (2.52<2.66).
        best_idx = order.index("nan" if model == "bolt" else "linear")
        lin_idx = order.index("linear")
        keep_idx = order.index("keep")
        route = np.where(pl == BLOCK, best_idx,
                         np.where(np.isin(pl, [CLEAN, INTER]), keep_idx, lin_idx))
        gated = np.array([M[route[k], k] for k in range(len(ids_m))])
        rec_m["gated"] = dict(mean=float(gated.mean()),
                              median=float(np.median(gated)),
                              regret_vs_oracle=float(gated.mean() - oracle.mean()),
                              oracle_match_rate=float((gated <= oracle + 1e-12).mean()))
        best_fixed = min(order, key=lambda f: per[f].mean())
        rec_m["gated_vs_bestfixed"] = dict(
            best_fixed=best_fixed,
            best_fixed_mean=rec_m[f"fixed|{best_fixed}"]["mean"],
            gated_mean=float(gated.mean()),
            per_window_winrate=float((gated <= per[best_fixed] + 1e-12).mean()))
        out_m[model] = rec_m
        print(f"  [e2e:metrla:{model}] gated={gated.mean():.3f} "
              f"oracle={rec_m['oracle']['mean']:.3f} best_fixed={best_fixed}="
              f"{rec_m[f'fixed|{best_fixed}']['mean']:.3f}", flush=True)
    out["metrla"] = out_m
    res.setdefault("trackB", {})["real_e2e"] = out
    save_results(res)
    return out


# ------------------------------------------------------------------ smoke ----
def run_smoke(device):
    print("[smoke] loading bolt ...", flush=True)
    model = Bolt96(device)
    X, starts = s5.load_windows(s5.DATASETS["ETTh1"], 6, s5.SEED)
    C = X.shape[1]
    wi, s = 0, starts[0]
    x = X[s:s + L].T.copy()
    m = s5.make_mask("mnar_high", 0.5, wi, 0, C, x=x)
    fd = dict(zip(FEAT_NAMES, channel_features(x[0], m[0])))
    print(f"[smoke] mnar_high p=0.5: r_bar={fd['r_bar']:.3f} "
          f"miss={fd['miss_rate']:.3f} e_max={fd['e_max']:.3f}", flush=True)
    assert fd["r_bar"] > 0.55, "mnar_high signature missing"
    m2 = s5.make_mask("block", 0.3, wi, 0, C, x=x)
    f2 = dict(zip(FEAT_NAMES, channel_features(x[1], m2[1])))
    print(f"[smoke] block p=0.3: mean_run={f2['mean_run_rel']*L:.1f} "
          f"little={f2['little_chi2']:.2f} r_bar={f2['r_bar']:.3f}", flush=True)
    assert f2["mean_run_rel"] * L > 10, "block run length missing"
    Fs, ys = [], []
    for vals, mask, lab, wj, c in train_samples("ETTh1", n_win=30):
        Fs.append(channel_features(vals, mask))
        ys.append(lab)
    clf = fit_clf("gbdt", np.stack(Fs), np.array(ys))
    p = proba_clf("gbdt", clf, np.stack(Fs[:5]))
    print(f"[smoke] mini gbdt ok, proba sums {p.sum(1).round(3)}", flush=True)
    ctx = s5.fill_context(x, m, "linear").reshape(C, L).astype(np.float32)
    q10, q90 = model.predict_quantiles(ctx)
    assert q10.shape == (C, H) and (q90 > q10).mean() > 0.99
    E = np.random.default_rng(0).random((100, H))
    w = conformal_w(E)
    assert w.shape == (H,) and (w >= np.median(E, axis=0)).all()
    print("[smoke OK]", flush=True)


# ----------------------------------------------------------------- figure ----
def run_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    ta = res.get("trackA", {})
    tb = res.get("trackB", {})

    # A: LOD accuracy
    ax = axes[0][0]
    lod = ta.get("lod", {})
    dss = list(lod)
    if dss:
        models = ("gbdt", "logreg", "little")
        w = 0.25
        for mi, mname in enumerate(models):
            vals = [lod[d][mname]["acc"] for d in dss]
            ax.bar(np.arange(len(dss)) + (mi - 1) * w, vals, width=w,
                   label=mname)
            for xi, v in zip(np.arange(len(dss)) + (mi - 1) * w, vals):
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(dss)))
        ax.set_xticklabels([f"held-out\n{d}" for d in dss], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("A. Track A LOD cross-validation\n(6-class mechanism classifier)",
                 fontsize=10)
    ax.legend(fontsize=8)

    # B: real transfer recall vs v1 detector
    ax = axes[0][1]
    tr = ta.get("transfer", {})
    labels_b, vals_b, v1_b = [], [], []
    if tr:
        for key, lab_t, lab_c in (("penn_oracle", "Penn cens\n(oracle mask)",
                                   "mnar_high"),):
            pass
        rows = [("Penn oracle\ncens->mnar", tr["penn_oracle"]["acc_cens"],
                 tr["penn_oracle"]["v1_cens_recall"]),
                ("Penn oracle\nctrl->clean", tr["penn_oracle"]["acc_clean"],
                 tr["penn_oracle"]["v1_ctrl_clean"]),
                ("Penn auto\ncens->mnar", tr["penn_auto"]["acc_cens"],
                 tr["penn_auto"]["v1_cens_recall"]),
                ("METRLA\nmiss->block", tr["metrla"]["acc_block"],
                 tr["metrla"]["v1_block_recall"]),
                ("METRLA\nctrl->clean", tr["metrla"]["acc_clean"],
                 tr["metrla"]["v1_ctrl_clean"])]
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [r[1] for r in rows], width=0.4, label="S16 classifier",
               color="#1f77b4")
        ax.bar(x + 0.2, [r[2] for r in rows], width=0.4, label="v1 rules (S6)",
               color="#c7c7c7")
        for xi, r in zip(x, rows):
            ax.text(xi - 0.2, r[1], f"{r[1]:.2f}", ha="center", va="bottom",
                    fontsize=7)
            ax.text(xi + 0.2, r[2], f"{r[2]:.2f}", ha="center", va="bottom",
                    fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("per-window recall")
    ax.set_title("B. Zero-shot transfer to real mechanisms\n(vs S6 v1 rules)",
                 fontsize=10)
    ax.legend(fontsize=8)

    # C: gating relMSE (pooled)
    ax = axes[0][2]
    ea, eb = tb.get("evalA", {}).get("summary", {}), tb.get("evalB", {}).get(
        "summary", {})
    groups, best_f, gated, orac = [], [], [], []
    if ea:
        for mech in ("mcar", "block"):
            s = ea[mech]
            groups.append(f"mcar\n(s12)" if mech == "mcar" else "block\n(s12)")
            best_f.append(s[s["best_fixed_arm"]])
            gated.append(s["gated_full"])
            orac.append(s["oracle_all"])
    if eb:
        for mech in ("mnar_high", "mnar_extreme"):
            s = eb[mech]
            groups.append(f"{mech}\n(s5/s6)")
            best_f.append(s[s["best_fixed_arm"]])
            gated.append(s["gated_linear"])
            orac.append(s["oracle"])
    if groups:
        x = np.arange(len(groups))
        w = 0.28
        ax.bar(x - w, best_f, width=w, label="best fixed", color="#c7c7c7")
        ax.bar(x, gated, width=w, label="gated (S16)", color="#1f77b4")
        ax.bar(x + w, orac, width=w, label="oracle route", color="#2ca02c")
        for xi, v in zip(x, gated):
            ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=8)
    ax.set_ylabel("mean relMSE vs clean")
    ax.set_title("C. Track B gating: per-window routing value\n(bolt, pooled "
                 "over datasets x rates)", fontsize=10)
    ax.legend(fontsize=8)

    # D: synthetic Mondrian conformal
    ax = axes[1][0]
    cs = tb.get("conf_syn", {})
    if cs:
        cls_rows = [c for c in CLASSES[:5] if any(c in cs[d]["rows"]
                                                  for d in cs)] + ["ALL"]
        variants = ("naive", "mondrian_pred", "mondrian_oracle")
        w = 0.25
        for vi, v in enumerate(variants):
            vals = []
            for c in cls_rows:
                covs = [cs[d]["rows"][c][v]["coverage"] for d in cs
                        if c in cs[d]["rows"]]
                vals.append(float(np.mean(covs)) if covs else np.nan)
            ax.bar(np.arange(len(cls_rows)) + (vi - 1) * w, vals, width=w,
                   label=v)
        ax.axhline(0.9, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(cls_rows)))
        ax.set_xticklabels(cls_rows, fontsize=8, rotation=20)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("coverage (target 0.9)")
    ax.set_title("D. Mondrian conformal, synthetic mixed stream\n(grouped by "
                 "predicted vs oracle class; bolt)", fontsize=10)
    ax.legend(fontsize=8)

    # E: real Mondrian conformal
    ax = axes[1][1]
    cr = tb.get("conf_real", {})
    if cr:
        rows = []
        if "penn" in cr:
            rows.append(("Penn cens\nzero fill", cr["penn"]["zero"]["cens_test"]))
            rows.append(("Penn cens\nlinear fill",
                         cr["penn"]["linear"]["cens_test"]))
        if "metrla" in cr:
            rows.append(("METRLA miss\nlinear fill",
                         cr["metrla"]["linear"]["miss_test"]))
            rows.append(("METRLA miss\nnan fill",
                         cr["metrla"]["nan"]["miss_test"]))
        variants = ("native", "naive", "mondrian_pred", "mondrian_oracle")
        w = 0.2
        cols = ["#999999", "#d62728", "#1f77b4", "#2ca02c"]
        for vi, v in enumerate(variants):
            vals = [r[1][v]["coverage"] for r in rows]
            ax.bar(np.arange(len(rows)) + (vi - 1.5) * w, vals, width=w,
                   label=v, color=cols[vi])
        ax.axhline(0.9, color="k", ls="--", lw=1)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("coverage (target 0.9)")
    ax.set_title("E. Real-data conformal: Mondrian by detected class\n"
                 "(S10/S14 splits, bolt)", fontsize=10)
    ax.legend(fontsize=7)

    # F: real end-to-end NMSE
    ax = axes[1][2]
    e2 = tb.get("real_e2e", {})
    if e2:
        rows = []
        if "penn" in e2 and "bolt" in e2["penn"]:
            r = e2["penn"]["bolt"]
            rows.append(("Penn\nkeep", r["fixed|keep"]["mean"]))
            rows.append(("Penn\nzero", r["fixed|zero"]["mean"]))
            rows.append(("Penn\nlinear", r["fixed|linear"]["mean"]))
            rows.append(("Penn\ngated\n(auto mask)", r["gated_automask"]["mean"]))
            rows.append(("Penn\noracle", r["oracle"]["mean"]))
        if "metrla" in e2 and "bolt" in e2["metrla"]:
            r = e2["metrla"]["bolt"]
            rows.append(("MLA\nzero", r["fixed|zero"]["mean"]))
            rows.append(("MLA\nlinear", r["fixed|linear"]["mean"]))
            rows.append(("MLA\nnan", r["fixed|nan"]["mean"]))
            rows.append(("MLA\ngated", r["gated"]["mean"]))
            rows.append(("MLA\noracle", r["oracle"]["mean"]))
        x = np.arange(len(rows))
        cols = (["#c7c7c7"] * 3 + ["#1f77b4", "#2ca02c"] +
                ["#c7c7c7"] * 3 + ["#1f77b4", "#2ca02c"])[:len(rows)]
        ax.bar(x, [r[1] for r in rows], color=cols)
        for xi, r in zip(x, rows):
            ax.text(xi, r[1], f"{r[1]:.2f}", ha="center", va="bottom",
                    fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([r[0] for r in rows], fontsize=7)
        ax.set_yscale("log")
    ax.set_ylabel("mean NMSE (log)")
    ax.set_title("F. Real end-to-end: gated pipeline vs fixed fills\n(bolt; "
                 "store-assembled per-window NMSE)", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "s16.png"), dpi=150)
    print("saved s16.png", flush=True)


# -------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke",
                    help="comma list: smoke,gate,trackA,trackB_gate,conf_syn,"
                         "conf_real,real_e2e,figure,all")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--windows", type=int, default=220,
                    help="train windows per dataset (Track A)")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    modes = args.mode.split(",")
    if "all" in modes:
        modes = ["gate", "trackA", "trackB_gate", "conf_syn", "conf_real",
                 "real_e2e", "figure"]
    t0 = time.time()
    res = load_results(args.out)
    res.setdefault("meta", dict(
        track="S16: mechanism detector + gated repair pipeline "
              "(detect->route->calibrate)",
        seed=SEED, classes=CLASSES, features=FEAT_NAMES,
        train=f"train-split windows ({args.windows}/dataset), S5 mask "
              "generators, p~U(0.05,0.8), block len jitter {{12,24,48,96}}, "
              "50% channel-shared blocks",
        evalA="s12_results.json bolt 150win masks==make_mask(ms=0)",
        evalB="s5_missing_results.json + s6_mnarfix_results.json bolt 300win "
              "(mcar/block routing features use ms=0; stored MSE averages "
              "ms={0,1})",
        stores=["s5_missing_results.json", "s5_fix_results.json",
                "s6_mnarfix_results.json", "s6_conformal_results.json",
                "s10_real_results.json", "s12_results.json",
                "s14_metrla_results.json"],
        anchor_cells=["s12|ETTh1|mcar:zs:linear:0.3",
                      "s5|ETTm1|mnar_high:linear:0.5",
                      "s10|bolt|cens|zero"]))
    save_results(res, args.out)

    if "smoke" in modes:
        run_smoke(args.device)
        return
    if "gate" in modes:
        run_gate(res, args.device)
    if "trackA" in modes:
        clfs = run_trackA(res, args.windows)
        labels, acc = eval_labels_synthetic(clfs["gbdt"])
        if acc is not None:
            res["trackA"]["syn_eval_acc"] = acc
            save_results(res)
        run_transfer(clfs["gbdt"], res)
    if "trackB_gate" in modes:
        z = np.load(os.path.join(CKPT, "labels_syn_gbdt.npz"))
        labels = {k: z[k] for k in z.files}
        run_gate_eval_a(res, labels)
        run_gate_eval_b(res, labels)
    if "conf_syn" in modes:
        with open(os.path.join(CKPT, "clf_final.pkl"), "rb") as fh:
            clf = pickle.load(fh)["clfs"]["gbdt"]
        run_conf_syn(res, clf, args.device)
    if "conf_real" in modes:
        run_conf_real(res, args.device)
    if "real_e2e" in modes:
        run_real_e2e(res)
    if "figure" in modes:
        run_figure(res)
    print(f"[done] modes={modes} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
