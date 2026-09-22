#!/usr/bin/env python
"""S24: generalization hardening pack for the S16 MechGate detector/router.

Reviewer-attack surface: "the heuristic components (gbdt detector, hard
routing table) may not generalize". S24 turns each component into measured
evidence, four sub-tasks:

  A. Soft routing (Bayesian decision). Replace the hard argmax->table route
     by argmin_a sum_c p(c|x) L(c,a): the detector's class posterior x an
     expected-loss matrix estimated from the existing per-window stores
     (synthetic: s5/s6 tobit + s22 block-tobit fill-in + S23 saits/brits
     learned imputers; real: s10/s14 NMSE stores, L-hat fit on the cal half
     of the S10/S14 splits, evaluated on the test half). gbdt posteriors are
     temperature-calibrated on the S22 calibration stream first (S22 showed
     raw gbdt posteriors are overconfident off-manifold); raw and calibrated
     soft routing are both reported. Synthetic protocol: L fit on windows
     wi<75 of the S16 Eval-B grid (first 150 windows), evaluated on
     75<=wi<150, per-window paired relMSE vs clean. Hard baselines: the S16
     deployment table AND a plug-in hard table (argmax -> argmin_a L(c,a))
     built from the SAME L, so soft-vs-hard isolates posterior weighting.

  B. Deep detector comparison. Two neural classifiers trained on the SAME
     S16 training samples: mlp_feat (MLP on the same 14-dim features --
     isolates the model contribution) and raw_net (small conv+transformer on
     [linear-interpolated values, mask] sequences, length-agnostic via global
     mean pooling -- the "learnable features" variant). Three axes: closed-
     set accuracy on the S16 synthetic grid B, zero-shot cross-domain
     transfer (Penmanshiel oracle/auto, METR-LA, PhysioNet'12), open-set
     detection on the S22 novel scenario (maxprob + maha double criterion;
     feature-space maha for gbdt/mlp_feat, embedding-space maha for raw_net).

  C. In-family parameter extrapolation. The detector was trained on block
     lengths {12,24,48,96} (jittered), rank-based censoring, rates
     U(0.05,0.8). Systematic shifts: block length {6,12,48,96} (6 = below
     family), quantile-threshold censoring at {q80,q85,q95,q99} (q99 = rate
     0.01, below the trained rate floor), and rate extrapolation to 0.9 for
     all four damaged mechanisms. Accuracy-vs-shift curves per model.

  D. Third real domain: PhysioNet'12 (informative, care-driven sampling).
     Missingness-geometry exploration, detector transfer (gbdt + neural
     variants), maha open-set flagging, and gated end-to-end routing from the
     s5_real per-window stores (bolt/timesfm x {zero,ffill,linear}), patients
     split cal/test, per-channel linear-normalized relMSE (S5-real
     convention).

Anchor gate: 3 S16 syn_eval cells recomputed from the grid-B feature cache
(+-5% acc and >=99% label agreement vs s16_ckpt/labels_syn_gbdt.npz), plus 2
S16 Eval-B arm means reproduced from stores.

Modes (comma list): smoke, gate, A, B, C, D, figure, all.
New files only: run_s24_generalization.py, s24_results.json, s24.png,
s24_notes.md, s24_ckpt/, s24_*.log. No existing file is modified; no git.
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
import run_s16_mechgate as s16
import run_s22_openset as s22

L, H = s5.L, s5.H                       # 512, 96
SEED = 20260817
NPROC = 24
OUT = os.path.join(HERE, "s24_results.json")
CKPT = os.path.join(HERE, "s24_ckpt")
os.makedirs(CKPT, exist_ok=True)

CLASSES = s16.CLASSES
CLEAN, MCAR, BLOCK, MNAR_H, MNAR_E, INTER = range(6)
RATES4 = [0.1, 0.3, 0.5, 0.7]
N_EVAL = 150                            # store fill-in coverage (S22 convention)
CAL_WI = 75                             # synthetic L: wi<75 fit, 75<=wi<150 eval
GRIDB_MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
ACTIONS_SYN = ["zero", "ffill", "linear", "tail_tobit", "saits_all",
               "brits_all"]
GATE_CELLS = [("ETTh1", "mnar_extreme", 0.7), ("ETTm1", "block", 0.5),
              ("weather", "mnar_high", 0.3)]

# Task C cell specs
C_BLENS = [6, 12, 48, 96]
C_QS = [0.80, 0.85, 0.95, 0.99]
C_RATE09_MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")


# ============================================================ io helpers ====
def save_results(res, path=OUT):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
    os.replace(tmp, path)


def load_results(path=OUT):
    if os.path.exists(path):
        return json.load(open(path))
    return {}


def load_gbdt():
    with open(os.path.join(s16.CKPT, "clf_final.pkl"), "rb") as fh:
        return pickle.load(fh)["clfs"]["gbdt"]


# ===================================================== multiprocessing pool ====
def _init_worker():
    s22._init_worker()


def _starts_eval(ds):
    return s22._starts_eval(ds)


def block_mask_blen(rate, wi, C, blen):
    """Task C: s5-algorithm block mask with parameterized length, per-channel
    independent, deterministic in (wi, rate, blen). Distinct spawn key from
    s5/s16 streams."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [s5.SEED, wi, 3, int(round(rate * 100)), 1000 + blen]))
    m = np.zeros((C, L), bool)
    n_blocks = int(round(rate * L / blen))
    if n_blocks > 0:
        for c in range(C):
            for st in rng.integers(0, L - blen + 1, size=n_blocks):
                m[c, st:st + blen] = True
    return m


def mnar_quantile_mask(x, q):
    """Task C: censor values strictly above the per-channel quantile q
    (threshold-censoring family; achieved rate ~= 1-q on continuous data).
    Deterministic."""
    thr = np.quantile(x, q, axis=1, keepdims=True)
    return x > thr


def shift_mask(cell, x, wi):
    """Mask [C,L] for a Task C cell spec tuple."""
    kind = cell[0]
    C = x.shape[0]
    if kind == "blen":
        _, blen, rate = cell
        return block_mask_blen(rate, wi, C, blen)
    if kind == "q":
        _, q = cell
        return mnar_quantile_mask(x, q)
    if kind == "rate09":
        _, mech = cell
        return s5.make_mask(mech, 0.9, wi, 0, C, x=x)
    raise ValueError(cell)


C_CELLS = ([("blen", b, r) for b in C_BLENS for r in RATES4]
           + [("q", q) for q in C_QS]
           + [("rate09", m) for m in C_RATE09_MECHS])
C_CELL_TRUE = {**{("blen", b, r): BLOCK for b in C_BLENS for r in RATES4},
               **{("q", q): MNAR_H for q in C_QS},
               **{("rate09", m): CLASSES.index(m) for m in C_RATE09_MECHS}}
C_CELL_NAME = {**{("blen", b, r): f"block_b{b}_r{r}"
                  for b in C_BLENS for r in RATES4},
               **{("q", q): f"mnar_q{int(q*100)}" for q in C_QS},
               **{("rate09", m): f"{m}_r0.9" for m in C_RATE09_MECHS}}


def _job(j):
    kind = j[0]
    if kind == "featB":
        # grid-B features: ("featB", ds, mech, rate, wi), wi in 0..299
        _, ds, mech, rate, wi = j
        X = s22._X[ds]
        s = _starts_eval(ds)[wi]
        x = X[s:s + L].T.copy()
        m = s5.make_mask(mech, rate, wi, 0, X.shape[1], x=x)
        return j, s16.window_features(x, m)
    if kind == "shiftC":
        _, ds, cell, wi = j
        X = s22._X[ds]
        s = _starts_eval(ds)[wi]
        x = X[s:s + L].T.copy()
        m = shift_mask(cell, x, wi)
        return j, s16.window_features(x, m)
    if kind == "calS":
        # S22 'novel' calibration stream (train region, 100 windows/dataset),
        # returning per-channel (features, label, vals, mask, window index)
        # so every S24 model can be scored on the identical stream.
        _, ds = j
        X = s22._X[ds]
        starts = s22._starts_train(ds, 100, s22.SEED)
        Fs, labs, vals, masks, wis = [], [], [], [], []
        for wi, s in enumerate(starts):
            x = X[s:s + L].T.copy()
            for c in range(X.shape[1]):
                v, m, lab = s22._calib_channel("novel", ds, wi, c, x[c])
                Fs.append(s16.channel_features(v, m))
                labs.append(lab)
                vals.append(v)
                masks.append(m)
                wis.append(wi)
        return j, (np.stack(Fs), np.array(labs, np.int64),
                   np.stack(vals).astype(np.float32), np.stack(masks),
                   np.array(wis, np.int64))
    if kind == "tobitA":
        # mcar:tail_tobit fill-in, mask seeds {0,1} (store convention)
        _, ds, rate, wi = j
        import run_s6_mnarfix as s6
        X = s22._X[ds]
        C = X.shape[1]
        s = _starts_eval(ds)[wi]
        x = X[s:s + L].T.copy()
        gt = X[s + L:s + L + H].T.copy()
        out = {}
        for ms in (0, 1):
            m = s5.make_mask("mcar", rate, wi, ms, C, x=x)
            filled = np.empty_like(x)
            for c in range(C):
                filled[c], _ = s6.tail_impute(x[c], m[c], "tail_tobit")
            out[ms] = filled.astype(np.float32)
        return j, (out, gt)
    raise ValueError(kind)


def run_jobs(jobs, nproc=NPROC):
    import multiprocessing as mp
    out = {}
    if not jobs:
        return out
    with mp.Pool(nproc, initializer=_init_worker) as pool:
        for j, payload in pool.imap_unordered(_job, jobs, chunksize=4):
            out[j] = payload
    return out


def eval_starts_main(X, n=300):
    """Main-process replica of s5.load_windows / s22._starts_eval start
    selection (n test-region window starts, deterministic)."""
    N = len(X)
    test_start = int(0.8 * N)
    lo, hi = test_start - L, N - L - H
    n_valid = hi - lo + 1
    rng = np.random.default_rng(s5.SEED)
    return np.sort(rng.choice(n_valid, size=min(n, n_valid),
                              replace=False)) + lo


def load_ds_main():
    """Load the 3 dataset matrices in the main process (small CSVs)."""
    import pandas as pd
    return {ds: pd.read_csv(p).drop(columns=["date"]).to_numpy(np.float32)
            for ds, p in s5.DATASETS.items()}


# ---------------------------------------------------- grid-B feature cache ----
def gridb_features():
    """{(ds,mech,rate): [300, C, 14]} feature cache for the S16 Eval-B grid."""
    cache = os.path.join(CKPT, "feats_gridB.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        out = {}
        for k in z.files:
            ds, mech, rate = k.split("|")
            out[(ds, mech, float(rate))] = z[k]
        return out
    jobs = [("featB", ds, mech, rate, wi)
            for ds in s5.DATASETS for mech in GRIDB_MECHS for rate in RATES4
            for wi in range(300)]
    t0 = time.time()
    r = run_jobs(jobs)
    store = {}
    for ds in s5.DATASETS:
        for mech in GRIDB_MECHS:
            for rate in RATES4:
                arr = np.stack([r[("featB", ds, mech, rate, wi)]
                                for wi in range(300)])
                store[f"{ds}|{mech}|{rate}"] = arr
    np.savez(cache, **store)
    print(f"  [gridB] {len(jobs)} windows featurized ({time.time()-t0:.0f}s)",
          flush=True)
    return {(ds, mech, float(rate)): store[f"{ds}|{mech}|{rate}"]
            for ds in s5.DATASETS for mech in GRIDB_MECHS for rate in RATES4}


def window_proba_from_feats(model_proba_fn, F):
    """F [..., C, 14] -> window posterior [..., 6] via channel-mean vote.
    model_proba_fn maps a FLAT [n, 14] matrix -> [n, 6] posteriors."""
    C = F.shape[-2]
    P = model_proba_fn(F.reshape(-1, F.shape[-1]))
    P = P.reshape(F.shape[:-1] + (P.shape[-1],))
    return P.mean(axis=-2)


def gbdt_channel_proba(clf, F):
    return clf.predict_proba(F)


# ------------------------------------------------------------------- gate ----
def run_gate(res):
    """Anchor: 3 S16 syn_eval cells from the grid-B cache (+-5% acc, >=99%
    label agreement) + 2 S16 Eval-B arm means reproduced from stores."""
    feats = gridb_features()
    clf = load_gbdt()
    cached = np.load(os.path.join(s16.CKPT, "labels_syn_gbdt.npz"))
    d16 = json.load(open(os.path.join(HERE, "s16_results.json")))
    stored_acc = d16["trackA"]["syn_eval_acc"]
    cells = []
    for ds, mech, rate in GATE_CELLS:
        F = feats[(ds, mech, rate)]
        wl = window_proba_from_feats(
            lambda b: gbdt_channel_proba(clf, b), F).argmax(axis=1)
        key = f"B|{ds}|{mech}|{rate}"
        acc = float((wl == CLASSES.index(mech)).mean())
        agree = float((wl == cached[key].astype(int)).mean())
        st = stored_acc[key]["acc"]
        ok = abs(acc - st) <= 0.05 and agree >= 0.99
        cells.append(dict(cell=key, stored_acc=float(st), recomputed_acc=acc,
                          label_agreement=agree, passed=bool(ok)))
        print(f"  [gate] {key}: stored={st:.4f} rerun={acc:.4f} "
              f"agree={agree:.4f} {'OK' if ok else 'FAIL'}", flush=True)
    # S16 Eval-B arm anchors (300-window store assembly, CPU only)
    d5 = json.load(open(os.path.join(HERE, "s5_missing_results.json")))["bolt"]
    d6 = json.load(open(os.path.join(HERE, "s6_mnarfix_results.json")))["bolt"]
    for mech, ref in (("mnar_high", 1.911), ("mnar_extreme", 1.861)):
        vals = []
        for ds in s5.DATASETS:
            clean = np.array(d5[ds]["clean:none:0.0"]["mse_per_window"])
            for rate in RATES4:
                lab = cached[f"B|{ds}|{mech}|{rate}"].astype(int)
                lin = np.array(d5[ds][f"{mech}:linear:{rate}"]
                               ["mse_per_window"]) / clean
                tob = np.array(d6[ds][f"{mech}:tail_tobit:{rate}"]
                               ["mse_per_window"]) / clean
                is_mnar = np.isin(lab, [MNAR_H, MNAR_E])
                vals.append(np.where(is_mnar, tob, lin).mean())
        got = float(np.mean(vals))
        ok = abs(got - ref) / ref <= 0.05
        cells.append(dict(cell=f"s16_evalB_gated_tobit|{mech}",
                          stored_acc=ref, recomputed_acc=got,
                          label_agreement=None, passed=bool(ok)))
        print(f"  [gate] evalB gated_tobit {mech}: stored={ref:.4f} "
              f"rerun={got:.4f} {'OK' if ok else 'FAIL'}", flush=True)
    passed = all(c["passed"] for c in cells)
    print(f"GATE {'PASS' if passed else 'FAIL'}", flush=True)
    res["gate"] = dict(passed=bool(passed), cells=cells)
    save_results(res)
    return passed


# ===================================================== calibration stream ====
def calib_stream():
    """S22 'novel' calibration stream with per-channel (features, label,
    vals, mask, window id); cached. Window ids are globally unique (dataset
    offsets added)."""
    cache = os.path.join(CKPT, "calib_stream.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["F"], z["lab"], z["vals"], z["mask"], z["win"]
    t0 = time.time()
    r = run_jobs([("calS", ds) for ds in s5.DATASETS], nproc=3)
    F = np.concatenate([r[("calS", ds)][0] for ds in s5.DATASETS])
    lab = np.concatenate([r[("calS", ds)][1] for ds in s5.DATASETS])
    vals = np.concatenate([r[("calS", ds)][2] for ds in s5.DATASETS])
    mask = np.concatenate([r[("calS", ds)][3] for ds in s5.DATASETS])
    win = np.concatenate([r[("calS", ds)][4] + 1000 * gi
                          for gi, ds in enumerate(s5.DATASETS)])
    np.savez(cache, F=F, lab=lab, vals=vals, mask=mask, win=win)
    print(f"  [calS] {len(lab)} channel samples ({time.time()-t0:.0f}s)",
          flush=True)
    return F, lab, vals, mask, win


def temperature_fit(proba, y):
    """Scalar temperature on log-posteriors, NLL fit (scipy bounded)."""
    from scipy.optimize import minimize_scalar
    logp = np.log(np.clip(proba, 1e-12, 1.0))

    def nll(T):
        z = logp / T
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())

    r = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    return float(r.x), float(r.fun), nll(1.0)


def apply_temperature(proba, T):
    """Temperature scaling on log-posteriors; softmax over the last axis
    (works for [n,6] and [n,C,6])."""
    z = np.log(np.clip(proba, 1e-12, 1.0)) / T
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def ece(proba, y, n_bins=15):
    conf = proba.max(axis=1)
    acc = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.sum():
            e += sel.mean() * abs(acc[sel].mean() - conf[sel].mean())
    return float(e)


def get_calibration():
    """Temperature for the gbdt posterior, fit on the calibration stream;
    returns (T, diagnostics dict)."""
    cache = os.path.join(CKPT, "calib_temperature.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        return d["T"], d
    F, lab, _, _, _win = calib_stream()
    clf = load_gbdt()
    P = gbdt_channel_proba(clf, F)
    T, nll_cal, nll_raw = temperature_fit(P, lab)
    d = dict(T=T, nll_raw=nll_raw, nll_cal=nll_cal,
             ece_raw=ece(P, lab), ece_cal=ece(apply_temperature(P, T), lab),
             acc_raw=float((P.argmax(1) == lab).mean()),
             acc_cal=float((apply_temperature(P, T).argmax(1) == lab).mean()),
             n=len(lab))
    json.dump(d, open(cache, "w"))
    print(f"  [calib] T={T:.3f} nll {nll_raw:.4f}->{nll_cal:.4f} "
          f"ece {d['ece_raw']:.4f}->{d['ece_cal']:.4f}", flush=True)
    return T, d


# ============================================================ Task A core ====
def syn_relMSE_store():
    """Per-window relMSE for every (ds, mech, rate, action) on the first
    N_EVAL windows, assembled from stores (+ mcar:tobit fill-in cache).
    Returns {(ds,mech,rate): {action: [N_EVAL]}}."""
    cache = os.path.join(CKPT, "relMSE_syn.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        out = {}
        for k in z.files:
            ds, mech, rate, a = k.split("|")
            out.setdefault((ds, mech, float(rate)), {})[a] = z[k]
        return out
    d5 = json.load(open(os.path.join(HERE, "s5_missing_results.json")))["bolt"]
    d6 = json.load(open(os.path.join(HERE, "s6_mnarfix_results.json")))["bolt"]
    blk_tob = np.load(os.path.join(s22.CKPT, "mse_block_tobit.npz"))
    mcar_tob_path = os.path.join(CKPT, "mse_mcar_tobit.npz")
    mcar_tob = (np.load(mcar_tob_path) if os.path.exists(mcar_tob_path)
                else None)
    out, store = {}, {}
    for ds in s5.DATASETS:
        clean = np.array(d5[ds]["clean:none:0.0"]["mse_per_window"])[:N_EVAL]
        d23 = {}
        for imp in ("saits", "brits"):
            d23[imp] = json.load(open(os.path.join(
                s23_dir(), f"eval_{imp}_all_{ds}.json")))
        for mech in GRIDB_MECHS:
            for rate in RATES4:
                row = {}
                for a in ("zero", "ffill", "linear"):
                    row[a] = np.array(
                        d5[ds][f"{mech}:{a}:{rate}"]["mse_per_window"]
                        )[:N_EVAL] / clean
                # tail_tobit: mnar from s6; block/mcar 0.3 from s6, other
                # rates from the s22 fill-in (block) / s24 fill-in (mcar)
                if mech.startswith("mnar") or rate == 0.3:
                    row["tail_tobit"] = np.array(
                        d6[ds][f"{mech}:tail_tobit:{rate}"]
                        ["mse_per_window"])[:N_EVAL] / clean
                elif mech == "block":
                    row["tail_tobit"] = blk_tob[f"{ds}|{rate}"] / clean
                else:
                    assert mcar_tob is not None, "run A_fillin first"
                    row["tail_tobit"] = mcar_tob[f"{ds}|{rate}"] / clean
                for imp in ("saits", "brits"):
                    row[f"{imp}_all"] = np.array(
                        d23[imp][f"{mech}:{rate}"]["mse_per_window"]
                        )[:N_EVAL] / clean
                for a, v in row.items():
                    store[f"{ds}|{mech}|{rate}|{a}"] = v
                out[(ds, mech, rate)] = row
    np.savez(cache, **store)
    return out


def s23_dir():
    return os.path.join(HERE, "s23_ckpt")


def run_A_fillin(device):
    """mcar:tail_tobit per-window MSE at rates {0.1,0.5,0.7} (0.3 is stored
    in s6); mask-seed {0,1} averaged, first N_EVAL windows (S22 convention)."""
    path = os.path.join(CKPT, "mse_mcar_tobit.npz")
    if os.path.exists(path):
        return
    model = s5.BoltModel(device)
    out = {}
    for ds in s5.DATASETS:
        for rate in (0.1, 0.5, 0.7):
            t0 = time.time()
            r = run_jobs([("tobitA", ds, rate, wi) for wi in range(N_EVAL)])
            ctxs, gts = [], []
            for wi in range(N_EVAL):
                fills, gt = r[("tobitA", ds, rate, wi)]
                for ms in (0, 1):
                    ctxs.append(fills[ms])
                    gts.append(gt)
            C = ctxs[0].shape[0]
            ctx = np.stack(ctxs).reshape(-1, L).astype(np.float32)
            pred = model.predict_point(ctx).reshape(len(ctxs), C, H)
            gt = np.stack(gts).astype(np.float64)
            mse = ((pred.astype(np.float64) - gt) ** 2).mean(axis=(1, 2))
            out[f"{ds}|{rate}"] = mse.reshape(N_EVAL, 2).mean(axis=1)
            print(f"  [fillin] mcar:tail_tobit:{rate} {ds} "
                  f"mse={out[f'{ds}|{rate}'].mean():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(path, **out)
    del model
    torch.cuda.empty_cache()


def syn_window_proba(T=None):
    """Window posteriors on the grid-B cache (first N_EVAL windows).
    Returns {(ds,mech,rate): [N_EVAL, 6]}."""
    tag = "raw" if T is None else f"T{T:.3f}"
    cache = os.path.join(CKPT, f"proba_gridB_{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {tuple(k.split("|")[:2]) + (float(k.split("|")[2]),): z[k]
                for k in z.files}
    clf = load_gbdt()
    feats = gridb_features()
    out, store = {}, {}
    for (ds, mech, rate), F in feats.items():
        F = F[:N_EVAL]
        P = gbdt_channel_proba(clf, F.reshape(-1, F.shape[-1]))
        P = P.reshape(F.shape[0], F.shape[1], -1)
        if T is not None:
            P = apply_temperature(P, T)
        W = P.mean(axis=1)
        out[(ds, mech, rate)] = W
        store[f"{ds}|{mech}|{rate}"] = W
    np.savez(cache, **store)
    return out


def estimate_L_syn(rel):
    """L(c,a) = mean relMSE of action a on windows of TRUE class c, fit on
    wi < CAL_WI pooled over datasets x rates. clean/intermittent rows are 1.0
    (fills are identity when no mask is present; the value is argmin-
    invariant -- it shifts all actions equally -- disclosed in notes)."""
    L_mat = np.ones((6, len(ACTIONS_SYN)))
    counts = np.zeros(6)
    for c, mech in enumerate(GRIDB_MECHS):
        cls = CLASSES.index(mech)
        for a_i, a in enumerate(ACTIONS_SYN):
            v = np.concatenate([rel[(ds, mech, rate)][a][:CAL_WI]
                                for ds in s5.DATASETS for rate in RATES4])
            L_mat[cls, a_i] = v.mean()
        counts[cls] = sum(len(rel[(ds, mech, RATES4[0])][a][:CAL_WI])
                          for ds in s5.DATASETS)
    return L_mat


SAFE_PRIORITY = ["linear", "ffill", "nan", "keep", "zero", "tail_tobit",
                 "saits_all", "brits_all"]


def safe_argmin_row(row, actions, tol=0.01):
    """argmin with a safety tie-break: among actions within tol (relative)
    of the row minimum, pick the one earliest in SAFE_PRIORITY (the S22
    safe-default convention: linear first). Prevents arbitrary tie-breaks
    onto dangerous actions (e.g. an all-tie row falling onto zero-fill)."""
    row = np.asarray(row, float)
    near = np.flatnonzero(row <= row.min() * (1 + tol) + 1e-12)
    pri = [SAFE_PRIORITY.index(actions[i]) if actions[i] in SAFE_PRIORITY
           else len(SAFE_PRIORITY) for i in near]
    return int(near[np.argmin(pri)])


def routing_metrics(M, actions_idx):
    """M [n_actions, n] per-window relMSE/NMSE; actions_idx [n] chosen
    action indices -> (mean, median, p95)."""
    n = M.shape[1]
    v = M[actions_idx, np.arange(n)]
    return float(v.mean()), float(np.median(v)), float(np.quantile(v, 0.95))


def run_A_syn(res):
    T, cal_diag = get_calibration()
    rel = syn_relMSE_store()
    L_mat = estimate_L_syn(rel)
    proba_raw = syn_window_proba(None)
    proba_cal = syn_window_proba(T)
    A = len(ACTIONS_SYN)
    # plug-in hard table from the same L (argmax class -> safest near-min
    # action in L's row); global safety reference = mean over damaged rows
    plug_table = {c: safe_argmin_row(L_mat[c], ACTIONS_SYN)
                  for c in range(6)}
    s16_table = {MCAR: ACTIONS_SYN.index("linear"),
                 BLOCK: ACTIONS_SYN.index("linear"),
                 MNAR_H: ACTIONS_SYN.index("tail_tobit"),
                 MNAR_E: ACTIONS_SYN.index("tail_tobit"),
                 CLEAN: ACTIONS_SYN.index("linear"),
                 INTER: ACTIONS_SYN.index("linear")}
    arms = ("fixed_linear", "hard_s16", "hard_plug", "soft_raw", "soft_cal",
            "oracle")
    out = {"actions": ACTIONS_SYN,
           "L_matrix": {CLASSES[c]: {a: float(L_mat[c, i])
                                     for i, a in enumerate(ACTIONS_SYN)}
                        for c in range(6)},
           "plug_table": {CLASSES[c]: ACTIONS_SYN[i]
                          for c, i in plug_table.items()},
           "calibration": cal_diag, "mechs": {}}
    for mech in GRIDB_MECHS:
        per_arm = {a: [] for a in arms}
        for ds in s5.DATASETS:
            for rate in RATES4:
                W_raw = proba_raw[(ds, mech, rate)][CAL_WI:]
                W_cal = proba_cal[(ds, mech, rate)][CAL_WI:]
                row = rel[(ds, mech, rate)]
                M = np.stack([row[a][CAL_WI:] for a in ACTIONS_SYN])
                n = M.shape[1]
                hard = W_raw.argmax(axis=1)
                idx = {
                    "fixed_linear": np.full(n, ACTIONS_SYN.index("linear")),
                    "hard_s16": np.array([s16_table[int(h)] for h in hard]),
                    "hard_plug": np.array([plug_table[int(h)] for h in hard]),
                    "soft_raw": (W_raw @ L_mat).argmin(axis=1),
                    "soft_cal": (W_cal @ L_mat).argmin(axis=1),
                    "oracle": M.argmin(axis=0),
                }
                for a in arms:
                    per_arm[a].append(M[idx[a], np.arange(n)])
        rec = {}
        for a in arms:
            v = np.concatenate(per_arm[a])
            rec[a] = dict(mean=float(v.mean()), median=float(np.median(v)),
                          p95=float(np.quantile(v, 0.95)))
        rec["regret_soft_cal"] = rec["soft_cal"]["mean"] - rec["oracle"]["mean"]
        rec["regret_hard_plug"] = (rec["hard_plug"]["mean"]
                                   - rec["oracle"]["mean"])
        out["mechs"][mech] = rec
        print(f"  [A:syn:{mech}] " + "  ".join(
            f"{a}={rec[a]['mean']:.3f}" for a in arms), flush=True)
    # pooled across mechanisms
    pooled = {}
    for a in arms:
        v = [out["mechs"][m][a]["mean"] for m in GRIDB_MECHS]
        pooled[a] = float(np.mean(v))
    out["pooled_mean_over_mechs"] = pooled
    print("  [A:syn pooled] " + "  ".join(f"{a}={pooled[a]:.3f}" for a in arms),
          flush=True)
    res.setdefault("A", {})["syn"] = out
    save_results(res)
    return out


# -------------------------------------------------- Task A: real datasets ----
def real_posteriors(T=None):
    """Window posteriors for Penn (oracle/auto masks) and METR-LA.
    Returns dict with proba arrays + window ids; cached."""
    tag = "raw" if T is None else f"T{T:.3f}"
    cache = os.path.join(CKPT, f"proba_real_{tag}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    clf = load_gbdt()
    out = {}
    import run_s10_real as s10
    windows, A = s10.load_all()
    ids_c = s10.group_ids(windows, "cens")
    ids_t = s10.group_ids(windows, "ctrl")
    for mode in ("oracle", "auto"):
        P = np.zeros((len(windows), 6))
        for k, i in enumerate(np.concatenate([ids_c, ids_t])):
            m = (A["mask"][i] if mode == "oracle"
                 else s10.candidate_mask(A["ctx_rec"][i], A["avail"][i]))
            p = gbdt_channel_proba(clf, s16.channel_features(
                A["ctx_rec"][i], m)[None])
            if T is not None:
                p = apply_temperature(p, T)
            P[i] = p[0]
        out[f"penn_{mode}"] = P
    import run_s14_metrla as s14
    windows = s14.build_windows()
    A = s14.extract(windows)
    ids_m = s14.group_ids(windows, "miss")
    ids_t = s14.group_ids(windows, "ctrl")
    P = np.zeros((len(windows), 6))
    for i in np.concatenate([ids_m, ids_t]):
        p = gbdt_channel_proba(clf, s16.channel_features(
            A["ctx_rec"][i], A["mask"][i])[None])
        if T is not None:
            p = apply_temperature(p, T)
        P[i] = p[0]
    out["metrla"] = P
    np.savez(cache, **out)
    return out


def _nmse_map(rec):
    return {int(i): v for i, v in zip(rec["window_ids"],
                                      rec["nmse_per_window"])}


def _real_route_eval(name, actions, nmse, cal_ids, test_ids, P_raw, P_cal,
                     s16_route, min_group=30):
    """Generic cal/test soft-vs-hard routing evaluation on one real group.
    nmse: {action: {window_id: nmse}}; P_raw/P_cal: [n_windows,6] posteriors;
    s16_route: pred class -> action index (S16 deployment table)."""
    A_ = len(actions)
    M_cal = np.stack([np.array([nmse[a][int(i)] for i in cal_ids])
                      for a in actions])
    M_te = np.stack([np.array([nmse[a][int(i)] for i in test_ids])
                     for a in actions])
    pred_cal = P_raw[cal_ids].argmax(axis=1)
    # L-hat rows = predicted class, fit on cal; fallback = cal global mean
    Lh = np.zeros((6, A_))
    glob = M_cal.mean(axis=1)
    for c in range(6):
        sel = pred_cal == c
        Lh[c] = M_cal[:, sel].mean(axis=1) if sel.sum() >= min_group else glob
    plug = {c: safe_argmin_row(Lh[c], actions) for c in range(6)}
    n = len(test_ids)
    hard = P_raw[test_ids].argmax(axis=1)
    idx = {f"fixed|{a}": np.full(n, k) for k, a in enumerate(actions)}
    idx["hard_s16"] = np.array([s16_route[int(h)] for h in hard])
    idx["hard_plug"] = np.array([plug[int(h)] for h in hard])
    idx["soft_raw"] = (P_raw[test_ids] @ Lh).argmin(axis=1)
    idx["soft_cal"] = (P_cal[test_ids] @ Lh).argmin(axis=1)
    idx["oracle"] = M_te.argmin(axis=0)
    rec = {"actions": actions, "n_cal": len(cal_ids), "n_test": n,
           "L_hat": {CLASSES[c]: {a: float(Lh[c, k])
                                  for k, a in enumerate(actions)}
                     for c in range(6)},
           "cal_pred_counts": {CLASSES[c]: int((pred_cal == c).sum())
                               for c in range(6)},
           "plug_table": {CLASSES[c]: actions[plug[c]] for c in range(6)}}
    for a, ii in idx.items():
        mean, med, p95 = routing_metrics(M_te, ii)
        rec[a] = dict(mean=mean, median=med, p95=p95)
    rec["regret_soft_cal"] = rec["soft_cal"]["mean"] - rec["oracle"]["mean"]
    rec["regret_hard_plug"] = rec["hard_plug"]["mean"] - rec["oracle"]["mean"]
    print(f"  [A:{name}] " + "  ".join(
        f"{a}={rec[a]['mean']:.3f}"
        for a in ("hard_s16", "hard_plug", "soft_raw", "soft_cal", "oracle")),
        flush=True)
    return rec


def run_A_real(res):
    T, _ = get_calibration()
    P_raw = real_posteriors(None)
    P_cal = real_posteriors(T)
    out = {}
    # ---------------- Penmanshiel ----------------
    import run_s10_real as s10
    d10 = json.load(open(os.path.join(HERE, "s10_real_results.json")))["records"]
    windows, A = s10.load_all()
    cal_c, test_c = s10.cal_test_split(windows, "cens")
    cal_t, test_t = s10.cal_test_split(windows, "ctrl")
    # ctrl NMSE under any fill == ctrl clean (contexts complete; fills no-op)
    for model in ("bolt", "timesfm", "moirai"):
        for mask_mode in ("oracle", "auto"):
            nmse = {}
            for f in ("keep", "zero", "linear"):
                m = _nmse_map(d10[f"anchor|{model}|cens|{f}"])
                mc = _nmse_map(d10[f"anchor|{model}|ctrl|clean"])
                m.update(mc)                     # ctrl identity
                nmse[f] = m
            # S16 deployment table: mnar->zero, clean/inter->keep, else linear
            s16_route = {MNAR_H: 1, MNAR_E: 1, CLEAN: 0, INTER: 0,
                         MCAR: 2, BLOCK: 2}
            rec = _real_route_eval(
                f"penn_{model}_{mask_mode}", ["keep", "zero", "linear"],
                nmse, np.concatenate([cal_c, cal_t]),
                np.concatenate([test_c, test_t]),
                P_raw[f"penn_{mask_mode}"], P_cal[f"penn_{mask_mode}"],
                s16_route)
            # also the cens-only view (S16's e2e population)
            rec_cens = _real_route_eval(
                f"penn_{model}_{mask_mode}|cens_only", ["keep", "zero",
                                                        "linear"],
                nmse, np.concatenate([cal_c, cal_t]), test_c,
                P_raw[f"penn_{mask_mode}"], P_cal[f"penn_{mask_mode}"],
                s16_route)
            rec["cens_only"] = {k: rec_cens[k] for k in (
                "hard_s16", "hard_plug", "soft_raw", "soft_cal", "oracle",
                "fixed|keep", "fixed|zero", "fixed|linear")}
            out[f"penn_{model}_{mask_mode}"] = rec
    # ---------------- METR-LA ----------------
    import run_s14_metrla as s14
    d14 = json.load(open(os.path.join(HERE, "s14_metrla_results.json")))["records"]
    windows = s14.build_windows()
    cal_m, test_m = s14.time_split(windows, "miss")
    cal_c, test_c = s14.time_split(windows, "ctrl")
    for model in ("bolt", "timesfm", "moirai"):
        acts = ["keep", "linear", "nan"]
        nmse = {}
        for f in acts:
            m = _nmse_map(d14[f"anchor|{model}|miss|{f}"])
            # ctrl: fills are the identity on complete contexts (S14 gate
            # verified bit-identical), so the stored ctrl|clean == any fill
            mc = _nmse_map(d14[f"anchor|{model}|ctrl|clean"])
            m.update(mc)
            nmse[f] = m
        # S16 table: block->nan(bolt)/linear(others), clean/inter->keep,
        # mcar/mnar->linear
        best = 2 if model == "bolt" else 1
        s16_route = {BLOCK: best, CLEAN: 0, INTER: 0, MCAR: 1, MNAR_H: 1,
                     MNAR_E: 1}
        rec = _real_route_eval(
            f"metrla_{model}", acts, nmse, np.concatenate([cal_m, cal_c]),
            np.concatenate([test_m, test_c]), P_raw["metrla"],
            P_cal["metrla"], s16_route)
        rec_miss = _real_route_eval(
            f"metrla_{model}|miss_only", acts, nmse,
            np.concatenate([cal_m, cal_c]), test_m, P_raw["metrla"],
            P_cal["metrla"], s16_route)
        rec["miss_only"] = {k: rec_miss[k] for k in (
            "hard_s16", "hard_plug", "soft_raw", "soft_cal", "oracle",
            "fixed|keep", "fixed|linear", "fixed|nan")}
        out[f"metrla_{model}"] = rec
    res.setdefault("A", {})["real"] = out
    save_results(res)
    return out


def run_A(res, device):
    run_A_fillin(device)
    run_A_syn(res)
    run_A_real(res)


# ============================================================ Task B core ====
def build_mlp():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(14, 64), nn.GELU(), nn.Linear(64, 64),
                         nn.GELU(), nn.Linear(64, 6))


class RawNet(torch.nn.Module):
    """Length-agnostic conv+transformer channel classifier.
    Input [B, 2, T]: channel 0 = linear-interpolated values standardized by
    observed stats; channel 1 = mask (1 = missing)."""

    def __init__(self, d=64, n_layers=2, n_heads=4, ff=128):
        super().__init__()
        import torch.nn as nn
        self.conv = nn.Sequential(
            nn.Conv1d(2, 32, 9, padding=4), nn.GELU(),
            nn.Conv1d(32, 32, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(32, d, 9, stride=4, padding=4), nn.GELU())
        layer = nn.TransformerEncoderLayer(
            d, n_heads, ff, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d, 6)

    def forward(self, x):
        h = self.conv(x).transpose(1, 2)       # [B, T', d]
        h = self.enc(h).mean(dim=1)            # global mean pool [B, d]
        return self.head(h), h


def raw_input(vals, mask):
    """vals [.., T] as-recorded (NaN = missing ok), mask [.., T] bool ->
    [.., 2, T] float32 tensor input."""
    v = np.where(mask, np.nan, vals).astype(np.float64)
    T = v.shape[-1]
    t = np.arange(T)
    flat = v.reshape(-1, T)
    out = np.zeros_like(flat)
    for i, row in enumerate(flat):
        ok = np.isfinite(row)
        if ok.sum() >= 2:
            out[i] = np.interp(t, np.flatnonzero(ok), row[ok])
        elif ok.sum() == 1:
            out[i] = row[ok][0]
    mu = np.nanmean(flat, axis=1, keepdims=True)
    sd = np.nanstd(flat, axis=1, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1.0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    z = (out - mu) / sd
    x = np.stack([z, mask.reshape(-1, T).astype(np.float64)], axis=1)
    return torch.from_numpy(x.reshape(vals.shape[:-1] + (2, T)).astype(
        np.float32))


def train_samples_raw():
    """Regenerate the exact S16 training samples as raw (vals, mask, label);
    cached."""
    cache = os.path.join(CKPT, "raw_train.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["vals"], z["mask"], z["lab"]
    vals, masks, labs = [], [], []
    for ds in s5.DATASETS:
        for v, m, lab, wi, c in s16.train_samples(ds, 220):
            vals.append(v)
            masks.append(m)
            labs.append(lab)
    vals = np.stack(vals).astype(np.float32)
    masks = np.stack(masks)
    labs = np.array(labs, np.int64)
    np.savez(cache, vals=vals, mask=masks, lab=labs)
    return vals, masks, labs


def train_neural(device):
    """Train mlp_feat + raw_net on the S16 training samples; cached to
    s24_ckpt. Returns (mlp_bundle, raw_bundle, doc dict)."""
    mlp_p = os.path.join(CKPT, "mlp_feat.pt")
    raw_p = os.path.join(CKPT, "raw_net.pt")
    doc = {}
    if os.path.exists(mlp_p) and os.path.exists(raw_p):
        mlp = torch.load(mlp_p, weights_only=False)
        raw = torch.load(raw_p, weights_only=False)
        return mlp, raw, None
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    import torch.nn as nn
    # ---------------- mlp_feat ----------------
    z = np.load(os.path.join(s16.CKPT, "feats_train.npz"))
    F, y = z["F"].astype(np.float32), z["y"]
    mu, sd = F.mean(0), F.std(0)
    # constant/near-constant columns (xmask_corr is identically 0 in the
    # per-channel training features) must not blow up standardized inputs
    # off-distribution -- floor the scale at 1.0 (same singular-direction
    # issue S22 documented for the maha distance).
    sd = np.where(sd > 1e-6, sd, 1.0)
    n = len(y)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    ntr = int(0.9 * n)
    tr, va = perm[:ntr], perm[ntr:]
    mlp = build_mlp()
    n_params = sum(p.numel() for p in mlp.parameters())
    opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.from_numpy((F[tr] - mu) / sd)
    ytr = torch.from_numpy(y[tr])
    Xva = torch.from_numpy((F[va] - mu) / sd)
    yva = torch.from_numpy(y[va])
    t0 = time.time()
    best = (0.0, None)
    for ep in range(60):
        mlp.train()
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 512):
            b = idx[i:i + 512]
            loss = nn.functional.cross_entropy(mlp(Xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        mlp.eval()
        with torch.no_grad():
            acc = (mlp(Xva).argmax(1) == yva).float().mean().item()
        if acc > best[0]:
            best = (acc, {k: v.clone() for k, v in mlp.state_dict().items()})
    if best[1] is not None:
        mlp.load_state_dict(best[1])
    doc["mlp_feat"] = dict(params=int(n_params), train_samples=int(ntr),
                           epochs=60, early_stop="best val acc",
                           val_acc=float(best[0]),
                           train_seconds=round(time.time() - t0, 1),
                           input="same 14-dim features as gbdt")
    mlp_bundle = {"sd": mlp.state_dict(), "mu": mu, "sd_": sd}
    torch.save(mlp_bundle, mlp_p)
    print(f"  [B] mlp_feat: {n_params} params, val acc {best[0]:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    # ---------------- raw_net ----------------
    vals, masks, labs = train_samples_raw()
    model = RawNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    n = len(labs)
    perm = rng.permutation(n)
    ntr = int(0.9 * n)
    tr, va = perm[:ntr], perm[ntr:]
    X_all = raw_input(vals, masks)
    y_all = torch.from_numpy(labs)
    t0 = time.time()
    best = (0.0, None)
    n_epochs = 20
    for ep in range(n_epochs):
        model.train()
        idx = torch.randperm(len(tr))
        for i in range(0, len(tr), 256):
            b = tr[idx[i:i + 256].numpy()]
            logits, _ = model(X_all[b].to(device))
            loss = nn.functional.cross_entropy(logits, y_all[b].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(va), 1024):
                lg, _ = model(X_all[va[i:i + 1024]].to(device))
                preds.append(lg.argmax(1).cpu())
            acc = (torch.cat(preds) == y_all[va]).float().mean().item()
        if acc > best[0]:
            best = (acc, {k: v.clone() for k, v in model.state_dict().items()})
        print(f"    [raw_net] epoch {ep}: val acc {acc:.4f}", flush=True)
    if best[1] is not None:
        model.load_state_dict(best[1])
    doc["raw_net"] = dict(params=int(n_params), train_samples=int(ntr),
                          epochs=n_epochs, early_stop="best val acc",
                          val_acc=float(best[0]),
                          train_seconds=round(time.time() - t0, 1),
                          input="[linear-interp standardized values, mask] "
                                "sequence, length-agnostic (mean pool)")
    torch.save({"sd": model.state_dict()}, raw_p)
    print(f"  [B] raw_net: {n_params} params, val acc {best[0]:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return mlp_bundle, {"sd": model.state_dict()}, doc


def mlp_channel_proba(bundle, F):
    """F [.., 14] -> [.., 6] posterior."""
    mlp = build_mlp()
    mlp.load_state_dict(bundle["sd"])
    mlp.eval()
    X = (F - bundle["mu"]) / bundle["sd_"]
    with torch.no_grad():
        p = torch.softmax(mlp(torch.from_numpy(
            X.reshape(-1, 14).astype(np.float32))), dim=1)
    return p.numpy().reshape(F.shape[:-1] + (6,))


class RawScorer:
    """Batch-scores raw (vals, mask) channels with raw_net; also exposes
    embeddings for the maha variant."""

    def __init__(self, bundle, device):
        self.model = RawNet().to(device)
        self.model.load_state_dict(bundle["sd"])
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def proba(self, vals, mask):
        X = raw_input(vals, mask)
        shp = X.shape[:-2]
        X = X.reshape(-1, *X.shape[-2:])
        ps, embs = [], []
        for i in range(0, len(X), 2048):
            lg, em = self.model(X[i:i + 2048].to(self.device))
            ps.append(torch.softmax(lg, 1).cpu().numpy())
            embs.append(em.cpu().numpy())
        P = np.concatenate(ps).reshape(shp + (6,))
        E = np.concatenate(embs).reshape(shp + (-1,))
        return P, E


def B_closed_set(res, mlp, raw, device):
    """Window accuracy of gbdt / mlp_feat / raw_net on the S16 grid B
    (300 windows/cell)."""
    cache = os.path.join(CKPT, "B_closedset.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    clf = load_gbdt()
    feats = gridb_features()
    scorer = RawScorer(raw, device)
    Xs = load_ds_main()
    out = {}
    for ds in s5.DATASETS:
        X = Xs[ds]
        starts = eval_starts_main(X)
        for mech in GRIDB_MECHS:
            for rate in RATES4:
                F = feats[(ds, mech, rate)]                    # [300,C,14]
                W_g = window_proba_from_feats(
                    lambda b: gbdt_channel_proba(clf, b), F)
                W_m = window_proba_from_feats(
                    lambda b: mlp_channel_proba(mlp, b), F)
                # raw: regenerate masks (deterministic), batch forward
                vals, masks = [], []
                for wi, s in enumerate(starts):
                    x = X[s:s + L].T.copy()
                    m = s5.make_mask(mech, rate, wi, 0, X.shape[1], x=x)
                    vals.append(x)
                    masks.append(m)
                V = np.stack(vals)                             # [300,C,L]
                Mk = np.stack(masks)
                P, _ = scorer.proba(V, Mk)                     # [300,C,6]
                W_r = P.mean(axis=1)
                true = CLASSES.index(mech)
                rec = {name: float((W.argmax(1) == true).mean())
                       for name, W in (("gbdt", W_g), ("mlp_feat", W_m),
                                       ("raw_net", W_r))}
                out[f"{ds}|{mech}|{rate}"] = rec
        print(f"  [B:closed] {ds} done", flush=True)
    # pooled per mechanism
    pooled = {}
    for mech in GRIDB_MECHS:
        pooled[mech] = {}
        for name in ("gbdt", "mlp_feat", "raw_net"):
            pooled[mech][name] = float(np.mean([
                out[f"{ds}|{mech}|{r}"][name]
                for ds in s5.DATASETS for r in RATES4]))
    out["_pooled"] = pooled
    json.dump(out, open(cache, "w"))
    print("  [B:closed pooled] " + "; ".join(
        f"{m}: " + "  ".join(f"{n}={pooled[m][n]:.3f}"
                             for n in ("gbdt", "mlp_feat", "raw_net"))
        for m in GRIDB_MECHS), flush=True)
    return out


def B_transfer(res, mlp, raw, device):
    """Zero-shot transfer of all models to Penn (oracle/auto), METR-LA and
    PhysioNet'12 (D's cache). Window posteriors cached for reuse by D."""
    cache = os.path.join(CKPT, "B_transfer.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    clf = load_gbdt()
    scorer = RawScorer(raw, device)
    out = {}

    def score_all(name, feats_list, raw_list):
        """feats_list: list of [1 or C,14]; raw_list: list of (vals[*,T],
        mask[*,T]) -> per-window predictions per model."""
        W = {"gbdt": [], "mlp_feat": [], "raw_net": []}
        for (F, (v, m)) in zip(feats_list, raw_list):
            W["gbdt"].append(gbdt_channel_proba(clf, F).mean(axis=0))
            W["mlp_feat"].append(mlp_channel_proba(mlp, F).mean(axis=0))
        # raw in bigger batches
        lens = [m.shape[-1] for _, m in raw_list]
        Cs = [m.shape[0] for _, m in raw_list]
        P = []
        for (v, m) in raw_list:
            p, _ = scorer.proba(v, m)
            P.append(p.mean(axis=0))
        W["raw_net"] = P
        return {k: np.stack(v) for k, v in W.items()}

    # ---- Penmanshiel ----
    import run_s10_real as s10
    windows, A = s10.load_all()
    ids_c = s10.group_ids(windows, "cens")
    ids_t = s10.group_ids(windows, "ctrl")
    true = np.array([MNAR_H] * len(ids_c) + [CLEAN] * len(ids_t))
    for mode in ("oracle", "auto"):
        feats, raws = [], []
        for i in np.concatenate([ids_c, ids_t]):
            m = (A["mask"][i] if mode == "oracle"
                 else s10.candidate_mask(A["ctx_rec"][i], A["avail"][i]))
            feats.append(s16.channel_features(A["ctx_rec"][i], m)[None])
            raws.append((A["ctx_rec"][i][None, :].astype(np.float32),
                         m[None, :]))
        W = score_all(f"penn_{mode}", feats, raws)
        rec = {}
        for name, p in W.items():
            pred = p.argmax(1)
            rec[name] = dict(
                cens_mnar_recall=float(np.isin(pred[true == MNAR_H],
                                               [MNAR_H, MNAR_E]).mean()),
                ctrl_clean_recall=float((pred[true == CLEAN] == CLEAN).mean()),
                pred_frac={CLASSES[c]: float((pred == c).mean())
                           for c in range(6)})
        out[f"penn_{mode}"] = rec
        print(f"  [B:transfer] penn_{mode}: " + "  ".join(
            f"{n} cens={rec[n]['cens_mnar_recall']:.3f}"
            for n in rec), flush=True)
    # ---- METR-LA ----
    import run_s14_metrla as s14
    windows = s14.build_windows()
    A = s14.extract(windows)
    ids_m = s14.group_ids(windows, "miss")
    ids_t = s14.group_ids(windows, "ctrl")
    true = np.array([BLOCK] * len(ids_m) + [CLEAN] * len(ids_t))
    feats, raws = [], []
    for i in np.concatenate([ids_m, ids_t]):
        feats.append(s16.channel_features(A["ctx_rec"][i], A["mask"][i])[None])
        raws.append((A["ctx_rec"][i][None, :].astype(np.float32),
                     A["mask"][i][None, :]))
    W = score_all("metrla", feats, raws)
    rec = {}
    for name, p in W.items():
        pred = p.argmax(1)
        rec[name] = dict(
            miss_block_recall=float((pred[true == BLOCK] == BLOCK).mean()),
            ctrl_clean_recall=float((pred[true == CLEAN] == CLEAN).mean()),
            miss_mnar_frac=float(np.isin(pred[true == BLOCK],
                                         [MNAR_H, MNAR_E]).mean()),
            pred_frac={CLASSES[c]: float((pred == c).mean())
                       for c in range(6)})
    out["metrla"] = rec
    print("  [B:transfer] metrla: " + "  ".join(
        f"{n} block={rec[n]['miss_block_recall']:.3f} "
        f"mnar={rec[n]['miss_mnar_frac']:.3f}" for n in rec), flush=True)
    # ---- PhysioNet'12 (features from D's cache; raw from same cache) ----
    p12 = p12_cache()
    if p12 is not None:
        ctx, tgt, chans, pids = p12
        mask = ~np.isfinite(ctx)
        W = score_all("p12", [s16.channel_features(ctx[w], mask[w])[None]
                              for w in range(len(ctx))],
                      [(np.nan_to_num(ctx[w], nan=0.0)[None, :], mask[w][None])
                       for w in range(len(ctx))])
        rec = {}
        for name, p in W.items():
            pred = p.argmax(1)
            rec[name] = dict(pred_frac={CLASSES[c]: float((pred == c).mean())
                                        for c in range(6)})
        out["p12"] = rec
        print("  [B:transfer] p12: " + "  ".join(
            f"{n} top={CLASSES[np.argmax([rec[n]['pred_frac'][c] for c in CLASSES])]}"
            for n in rec), flush=True)
    json.dump(out, open(cache, "w"))
    res.setdefault("B", {})["transfer"] = out
    save_results(res)
    return out


def B_openset(res, mlp, raw, device):
    """S22 'novel' open-set scenario for all models. maxprob per model;
    maha: feature space (gbdt/mlp_feat share the 14-dim space, identical to
    S22) and raw_net's 64-dim embedding space. tau at FRR=2% on the S24
    calibration stream (identical windows to S22's)."""
    cache = os.path.join(CKPT, "B_openset.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    clf = load_gbdt()
    mus, prec = s22.fit_maha("novel")
    F_cal, lab_cal, vals_cal, mask_cal, win_cal = calib_stream()
    scorer = RawScorer(raw, device)

    def window_mean(score, win):
        """Per-channel score -> per-window mean (S22 window-level conv)."""
        u = np.unique(win)
        out = np.array([score[win == w].mean() for w in u])
        return u, out

    def window_vote_maxprob(P, win):
        """Per-channel posteriors [n,6] -> per-window max of the channel-mean
        posterior (S16/S22 vote convention: vote first, then max)."""
        u = np.unique(win)
        return np.array([P[win == w].mean(axis=0).max() for w in u])

    # ---- calibration scores per model (window level, S22 convention) ----
    P_cal_g = gbdt_channel_proba(clf, F_cal)
    P_cal_m = mlp_channel_proba(mlp, F_cal)
    P_cal_r, E_cal_r = scorer.proba(vals_cal, mask_cal)
    mp_cal = {"gbdt": window_vote_maxprob(P_cal_g, win_cal),
              "mlp_feat": window_vote_maxprob(P_cal_m, win_cal),
              "raw_net": window_vote_maxprob(P_cal_r, win_cal)}
    # feature-space maha (S22 formula, col 12 excluded)
    dims = [i for i in range(14) if i not in s22.MAHA_EXCLUDE]

    def feat_maha(F):
        D = F[:, dims][:, None, :] - mus[None, :, :]
        d2 = np.einsum("ckd,de,cke->ck", D, prec, D)
        return np.sqrt(np.maximum(d2, 0.0)).min(axis=1)

    mh_cal_feat = window_mean(feat_maha(F_cal), win_cal)[1]
    # replication check vs s22's cached gbdt calibration scores
    z22 = np.load(os.path.join(s22.CKPT, "calib_novel.npz"))
    repl = dict(
        maxprob_maxabs=float(np.max(np.abs(np.sort(mp_cal["gbdt"])
                                           - np.sort(z22["mp"])))),
        maha_maxabs=float(np.max(np.abs(np.sort(mh_cal_feat)
                                        - np.sort(z22["mh"])))))
    print(f"  [B:openset] calib replication vs s22: maxprob "
          f"{repl['maxprob_maxabs']:.2e} maha {repl['maha_maxabs']:.2e}",
          flush=True)
    # embedding maha for raw_net
    from sklearn.covariance import LedoitWolf
    vals_t, mask_t, lab_t = train_samples_raw()
    _, E_tr = scorer.proba(vals_t, mask_t)
    mu_e = np.stack([E_tr[lab_t == c].mean(0) for c in range(6)])
    cen = E_tr - mu_e[lab_t]
    lw = LedoitWolf().fit(cen)

    def emb_maha(E):
        D = E[:, None, :] - mu_e[None, :, :]
        d2 = np.einsum("ckd,de,cke->ck", D, lw.precision_, D)
        return np.sqrt(np.maximum(d2, 0.0)).min(axis=1)

    mh_cal_emb = window_mean(emb_maha(E_cal_r), win_cal)[1]
    tau = {"gbdt|maxprob": float(np.quantile(mp_cal["gbdt"], 0.02)),
           "mlp_feat|maxprob": float(np.quantile(mp_cal["mlp_feat"], 0.02)),
           "raw_net|maxprob": float(np.quantile(mp_cal["raw_net"], 0.02)),
           "feat|maha": float(np.quantile(mh_cal_feat, 0.98)),
           "raw_emb|maha": float(np.quantile(mh_cal_emb, 0.98))}
    # ---- eval windows: s22 novel grid (features cached there) ----
    z = np.load(os.path.join(s22.CKPT, "gridfeats_novel.npz"))
    grids = {}
    for k in z.files:
        mech, rate, ds, wi = k.split("|")
        grids.setdefault((mech, float(rate), ds), []).append((int(wi), z[k]))
    Xs = load_ds_main()
    out = {"tau": tau, "calib_replication_vs_s22": repl, "mechs": {}}
    for mech in ("iburst", "mnar_low"):
        rec = {}
        for name, mh_m in (("gbdt", "feat"), ("mlp_feat", "feat"),
                           ("raw_net", "raw_emb")):
            dets_mp, dets_mh = [], []
            for rate in (0.3, 0.5):
                for ds in s5.DATASETS:
                    cells = sorted(grids[(mech, rate, ds)])
                    wis = [w for w, _ in cells]
                    F = np.concatenate([c for _, c in cells])  # [sumC, 14]
                    if name == "gbdt":
                        P = gbdt_channel_proba(clf, F)
                        E = None
                    elif name == "mlp_feat":
                        P = mlp_channel_proba(mlp, F)
                        E = None
                    else:
                        X = Xs[ds]
                        starts = eval_starts_main(X)
                        V, Mk = [], []
                        for wi in wis:
                            s = starts[wi]
                            x = X[s:s + L].T.copy()
                            m = s22.make_mask_s22(mech, rate, wi, 0,
                                                  X.shape[1], x)
                            V.append(x)
                            Mk.append(m)
                        P, E = scorer.proba(np.concatenate(V),
                                            np.concatenate(Mk))
                    mh = feat_maha(F) if mh_m == "feat" else emb_maha(E)
                    # window-level scores (S22): maxprob = max of the
                    # channel-mean posterior; maha = mean over channels
                    Cs = [c.shape[0] for _, c in cells]
                    idx = np.concatenate([[0], np.cumsum(Cs)])
                    w_mp = np.array([P[a:b].mean(axis=0).max()
                                     for a, b in zip(idx[:-1], idx[1:])])
                    w_mh = np.array([mh[a:b].mean()
                                     for a, b in zip(idx[:-1], idx[1:])])
                    dets_mp.append(w_mp < tau[f"{name}|maxprob"])
                    dets_mh.append(w_mh > tau["feat|maha" if mh_m == "feat"
                                              else "raw_emb|maha"])
            rec[name] = dict(
                det_maxprob=float(np.concatenate(dets_mp).mean()),
                det_maha=float(np.concatenate(dets_mh).mean()))
        out["mechs"][mech] = rec
        print(f"  [B:openset] {mech}: " + "  ".join(
            f"{n} mp={rec[n]['det_maxprob']:.3f} maha={rec[n]['det_maha']:.3f}"
            for n in rec), flush=True)
    json.dump(out, open(cache, "w"))
    res.setdefault("B", {})["openset"] = out
    save_results(res)
    return out


def run_B(res, device):
    mlp, raw, doc = train_neural(device)
    if doc:
        res.setdefault("B", {})["training_doc"] = doc
        save_results(res)
    closed = B_closed_set(res, mlp, raw, device)
    res.setdefault("B", {})["closed_set"] = closed
    save_results(res)
    B_transfer(res, mlp, raw, device)
    B_openset(res, mlp, raw, device)


# ============================================================ Task C core ====
def shiftc_features():
    """{(ds, cell): [150, C, 14]} features for the Task C shifted cells."""
    cache = os.path.join(CKPT, "feats_shiftC.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        out = {}
        for k in z.files:
            ds, cell_name = k.split("@")
            out[(ds, cell_name)] = z[k]
        return out
    jobs = [("shiftC", ds, cell, wi)
            for ds in s5.DATASETS for cell in C_CELLS for wi in range(N_EVAL)]
    t0 = time.time()
    r = run_jobs(jobs)
    store = {}
    for ds in s5.DATASETS:
        for cell in C_CELLS:
            arr = np.stack([r[("shiftC", ds, cell, wi)]
                            for wi in range(N_EVAL)])
            store[f"{ds}@{C_CELL_NAME[cell]}"] = arr
    np.savez(cache, **store)
    print(f"  [shiftC] {len(jobs)} windows ({time.time()-t0:.0f}s)", flush=True)
    return {(ds, k.split("@")[1]): v for k, v in
            store.items() for ds in [k.split("@")[0]]}


def run_C(res, device):
    feats = shiftc_features()
    clf = load_gbdt()
    mlp_p = os.path.join(CKPT, "mlp_feat.pt")
    raw_p = os.path.join(CKPT, "raw_net.pt")
    mlp = torch.load(mlp_p, weights_only=False) if os.path.exists(mlp_p) \
        else None
    raw = torch.load(raw_p, weights_only=False) if os.path.exists(raw_p) \
        else None
    scorer = RawScorer(raw, device) if raw else None
    Xs = load_ds_main() if scorer else {}
    out = {}
    for cell in C_CELLS:
        cname = C_CELL_NAME[cell]
        true = C_CELL_TRUE[cell]
        rec = {}
        Ws = {"gbdt": [], "mlp_feat": [], "raw_net": []}
        for ds in s5.DATASETS:
            F = feats[(ds, cname)]
            Ws["gbdt"].append(window_proba_from_feats(
                lambda b: gbdt_channel_proba(clf, b), F))
            if mlp:
                Ws["mlp_feat"].append(window_proba_from_feats(
                    lambda b: mlp_channel_proba(mlp, b), F))
            if scorer:
                X = Xs[ds]
                starts = eval_starts_main(X)
                V, Mk = [], []
                for wi in range(N_EVAL):
                    s = starts[wi]
                    x = X[s:s + L].T.copy()
                    V.append(x)
                    Mk.append(shift_mask(cell, x, wi))
                P, _ = scorer.proba(np.stack(V), np.stack(Mk))
                Ws["raw_net"].append(P.mean(axis=1))
        for name, W in Ws.items():
            if not W:
                continue
            pred = np.concatenate(W).argmax(1)
            rec[name] = dict(
                acc=float((pred == true).mean()),
                pred_frac={CLASSES[c]: float((pred == c).mean())
                           for c in range(6)})
        out[cname] = rec
        print(f"  [C] {cname}: " + "  ".join(
            f"{n}={rec[n]['acc']:.3f}" for n in rec), flush=True)
    res.setdefault("C", {})["cells"] = out
    # reference in-family accuracy from grid B (gbdt + whichever B models
    # are cached in B_closedset)
    bcs = res.get("B", {}).get("closed_set", {})
    ref = {}
    for mech in GRIDB_MECHS:
        for name in ("gbdt", "mlp_feat", "raw_net"):
            vals = [bcs[f"{ds}|{mech}|{r}"][name]
                    for ds in s5.DATASETS for r in RATES4
                    if f"{ds}|{mech}|{r}" in bcs]
            if vals:
                ref.setdefault(mech, {})[name] = float(np.mean(vals))
    res["C"]["reference_gridB_acc"] = ref
    save_results(res)
    return out


# ============================================================ Task D core ====
def p12_cache():
    """PhysioNet'12 windows from run_s5_real (cached arrays)."""
    path = os.path.join(CKPT, "p12_cache.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["ctx"], z["tgt"], z["chans"], z["pids"]
    import run_s5_real as s5r
    t0 = time.time()
    X, pids_all, varcols = s5r.load_p12_grid()
    windows = s5r.build_windows(X, pids_all, varcols)
    ctx, tgt = s5r.extract(windows, X, pids_all, varcols)
    chans = np.array([w["channel"] for w in windows])
    pids = np.array([w["patient"] for w in windows])
    np.savez(path, ctx=ctx, tgt=tgt, chans=chans, pids=pids)
    print(f"  [D] p12 cache built: {len(ctx)} windows "
          f"({time.time()-t0:.0f}s)", flush=True)
    return ctx, tgt, chans, pids


def run_D_explore(res):
    """Missingness-geometry exploration of the P12 windows."""
    ctx, tgt, chans, pids = p12_cache()
    mask = ~np.isfinite(ctx)
    out = {"n_windows": len(ctx), "channels": {}}
    miss_rate = mask.mean(axis=1)
    out["ctx_missing_rate"] = dict(mean=float(miss_rate.mean()),
                                   median=float(np.median(miss_rate)),
                                   p90=float(np.quantile(miss_rate, 0.9)),
                                   max=float(miss_rate.max()))
    # run geometry
    maxrun, nruns = [], []
    for w in range(len(ctx)):
        rs = s16._runs(mask[w])
        lens = np.array([j - i + 1 for i, j in rs]) if rs else np.zeros(1)
        maxrun.append(lens.max())
        nruns.append(len(rs))
    maxrun = np.array(maxrun)
    out["runs"] = dict(mean_n_runs=float(np.mean(nruns)),
                       maxrun_median=float(np.median(maxrun)),
                       frac_maxrun_ge6=float((maxrun >= 6).mean()),
                       frac_maxrun_ge12=float((maxrun >= 12).mean()))
    # informative-sampling probe: within-channel Spearman correlation between
    # a patient's context missing rate and the observed mean level
    from scipy.stats import spearmanr
    per_ch = {}
    for ch in sorted(set(chans)):
        sel = chans == ch
        mr = miss_rate[sel]
        lvl = np.nanmean(ctx[sel], axis=1)
        ok = np.isfinite(lvl)
        rho = float(spearmanr(mr[ok], lvl[ok]).statistic) if ok.sum() > 10 \
            else None
        per_ch[ch] = dict(n=int(sel.sum()), miss_mean=float(mr.mean()),
                          spearman_miss_vs_level=rho)
    out["per_channel"] = per_ch
    res.setdefault("D", {})["explore"] = out
    save_results(res)
    print(f"  [D:explore] miss mean {out['ctx_missing_rate']['mean']:.3f}, "
          f"frac maxrun>=6 {out['runs']['frac_maxrun_ge6']:.3f}", flush=True)
    return out


def run_D_transfer(res):
    """gbdt detector on P12 windows: predicted-class distribution + maha
    unknown fraction (S22 novel pipeline). Neural variants live in B."""
    ctx, tgt, chans, pids = p12_cache()
    mask = ~np.isfinite(ctx)
    cache = os.path.join(CKPT, "D_transfer.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        proba, mh = z["proba"], z["maha"]
    else:
        clf = load_gbdt()
        mus, prec = s22.fit_maha("novel")
        dims = [i for i in range(14) if i not in s22.MAHA_EXCLUDE]
        proba = np.zeros((len(ctx), 6))
        mh = np.zeros(len(ctx))
        for w in range(len(ctx)):
            F = s16.channel_features(ctx[w], mask[w])[None]
            proba[w] = gbdt_channel_proba(clf, F)[0]
            Dm = F[:, dims][:, None, :] - mus[None, :, :]
            d2 = np.einsum("ckd,de,cke->ck", Dm, prec, Dm)
            mh[w] = np.sqrt(np.maximum(d2, 0.0)).min(axis=1).mean()
        np.savez(cache, proba=proba, maha=mh)
    d22 = json.load(open(os.path.join(HERE, "s22_results.json")))
    tau_mh = d22["calib"]["novel"]["tau_maha"]
    pred = proba.argmax(1)
    out = {"tau_maha": tau_mh,
           "pred_frac": {CLASSES[c]: float((pred == c).mean())
                         for c in range(6)},
           "unknown_maha": float((mh > tau_mh).mean()),
           "per_channel": {}}
    for ch in sorted(set(chans)):
        sel = chans == ch
        out["per_channel"][ch] = dict(
            n=int(sel.sum()),
            pred_frac={CLASSES[c]: float((pred[sel] == c).mean())
                       for c in range(6)},
            unknown_maha=float((mh[sel] > tau_mh).mean()))
    print(f"  [D:transfer] unknown_maha={out['unknown_maha']:.3f}, top class "
          f"{max(out['pred_frac'], key=out['pred_frac'].get)}", flush=True)
    res.setdefault("D", {})["transfer"] = out
    save_results(res)
    return proba, mh, out


def run_D_e2e(res, proba, mh):
    """Gated routing on P12 from the s5_real per-window stores.
    relMSE per window = mse / cal-half per-channel mean linear mse.
    Arms: fixed fills / hard_s16 (prior-evidence table) / hard_plug /
    soft_raw / soft_cal / openset_maha / oracle."""
    ctx, tgt, chans, pids = p12_cache()
    d5r = json.load(open(os.path.join(HERE, "s5_real_results.json")))
    T, _ = get_calibration()
    d22 = json.load(open(os.path.join(HERE, "s22_results.json")))
    tau_mh = d22["calib"]["novel"]["tau_maha"]
    unk = mh > tau_mh
    # patient-level cal/test split (all channel-windows of a patient stay
    # together)
    uniq = np.sort(np.unique(pids))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(uniq))
    cal_p = set(uniq[perm[:len(perm) // 2]].tolist())
    is_cal = np.array([p in cal_p for p in pids])
    out = {}
    for model in ("bolt", "timesfm"):
        recs = {r["fill"]: np.array(r["mse_per_window"])
                for r in d5r["records"]
                if r["model"] == model and r["extra_mcar"] == 0.0}
        acts = ["zero", "ffill", "linear"]
        M = np.stack([recs[a] for a in acts])          # [3, 990]
        # per-channel linear norm fit on cal
        norm = {}
        for ch in set(chans):
            sel = is_cal & (chans == ch)
            norm[ch] = recs["linear"][sel].mean()
        N = np.array([norm[c] for c in chans])
        Mr = M / N[None, :]
        cal_idx = np.flatnonzero(is_cal)
        te_idx = np.flatnonzero(~is_cal)
        pred = proba.argmax(1)
        glob = Mr[:, cal_idx].mean(axis=1)
        Lh = np.zeros((6, 3))
        for c in range(6):
            sel = cal_idx[pred[cal_idx] == c]
            Lh[c] = Mr[:, sel].mean(axis=1) if len(sel) >= 30 else glob
        plug = {c: safe_argmin_row(Lh[c], acts) for c in range(6)}
        # S5-real prior-evidence table: everything -> linear, except
        # intermittent (natural near-zero channels) -> zero
        s16_route = {MCAR: 2, BLOCK: 2, MNAR_H: 2, MNAR_E: 2, CLEAN: 2,
                     INTER: 0}
        P_cal = apply_temperature(proba, T)
        n = len(te_idx)
        hard = pred[te_idx]
        idx = {f"fixed|{a}": np.full(n, k) for k, a in enumerate(acts)}
        idx["hard_s16"] = np.array([s16_route[int(h)] for h in hard])
        idx["hard_plug"] = np.array([plug[int(h)] for h in hard])
        idx["soft_raw"] = (proba[te_idx] @ Lh).argmin(axis=1)
        idx["soft_cal"] = (P_cal[te_idx] @ Lh).argmin(axis=1)
        idx["openset_maha"] = np.where(unk[te_idx], 2, idx["hard_plug"])
        idx["oracle"] = Mr[:, te_idx].argmin(axis=0)
        rec = {"n_cal": int(is_cal.sum()), "n_test": int((~is_cal).sum()),
               "L_hat": {CLASSES[c]: {a: float(Lh[c, k])
                                      for k, a in enumerate(acts)}
                         for c in range(6)},
               "plug_table": {CLASSES[c]: acts[plug[c]] for c in range(6)},
               "unknown_frac_test": float(unk[te_idx].mean())}
        for a, ii in idx.items():
            mean, med, p95 = routing_metrics(Mr[:, te_idx], ii)
            rec[a] = dict(mean=mean, median=med, p95=p95)
        rec["regret_soft_cal"] = (rec["soft_cal"]["mean"]
                                  - rec["oracle"]["mean"])
        # per-channel detail for the interesting channels
        rec["per_channel_soft_cal"] = {}
        for ch in sorted(set(chans)):
            sel = te_idx[chans[te_idx] == ch]
            if len(sel) < 5:
                continue
            ii = (P_cal[sel] @ Lh).argmin(axis=1)
            Msel = Mr[:, sel]
            rec["per_channel_soft_cal"][ch] = dict(
                n=int(len(sel)),
                soft=float(Msel[ii, np.arange(len(sel))].mean()),
                linear=float(Msel[2].mean()),
                zero=float(Msel[0].mean()))
        out[model] = rec
        print(f"  [D:e2e:{model}] " + "  ".join(
            f"{a}={rec[a]['mean']:.3f}"
            for a in ("fixed|zero", "fixed|linear", "hard_s16", "hard_plug",
                      "soft_cal", "openset_maha", "oracle")), flush=True)
    res.setdefault("D", {})["e2e"] = out
    save_results(res)
    return out


def run_D(res):
    run_D_explore(res)
    proba, mh, _ = run_D_transfer(res)
    run_D_e2e(res, proba, mh)


# ------------------------------------------------------------------ smoke ----
def run_smoke(device):
    print("[smoke] feature/mask sanity ...", flush=True)
    _init_worker()
    X = s22._X["ETTh1"]
    s = s22._starts_eval("ETTh1")[0]
    x = X[s:s + L].T.copy()
    # Task C masks
    m6 = block_mask_blen(0.3, 0, X.shape[1], 6)
    f6 = dict(zip(s16.FEAT_NAMES, s16.channel_features(x[0], m6[0])))
    print(f"[smoke] block_b6 r=0.3: mean_run={f6['mean_run_rel']*L:.1f} "
          f"miss={f6['miss_rate']:.3f}", flush=True)
    assert 3 < f6["mean_run_rel"] * L < 12
    mq = mnar_quantile_mask(x, 0.95)
    fq = dict(zip(s16.FEAT_NAMES, s16.channel_features(x[0], mq[0])))
    print(f"[smoke] mnar_q95: miss={fq['miss_rate']:.3f} "
          f"r_bar={fq['r_bar']:.3f}", flush=True)
    assert abs(fq["miss_rate"] - 0.05) < 0.02 and fq["r_bar"] > 0.55
    m9 = s5.make_mask("mcar", 0.9, 0, 0, X.shape[1], x=x)
    assert abs(m9.mean() - 0.9) < 0.02
    # calib-stream replication (1 channel)
    v, m, lab = s22._calib_channel("novel", "ETTh1", 0, 0, x[0])
    print(f"[smoke] calS channel: lab={CLASSES[lab]} miss={m.mean():.3f}",
          flush=True)
    # neural nets forward at two lengths
    torch.manual_seed(SEED)
    mlp = build_mlp()
    p = mlp(torch.zeros(3, 14))
    assert p.shape == (3, 6)
    rn = RawNet()
    lg512, em = rn(torch.zeros(2, 2, 512))
    lg24, _ = rn(torch.zeros(2, 2, 24))
    assert lg512.shape == (2, 6) and lg24.shape == (2, 6)
    n_par = sum(p_.numel() for p_ in rn.parameters())
    print(f"[smoke] raw_net params={n_par}, len 512/24 forward OK", flush=True)
    # raw_input helper
    xi = raw_input(x[:2], m9[:2])
    assert xi.shape == (2, 2, L) and torch.isfinite(xi).all()
    # tobit fill on one mcar window
    import run_s6_mnarfix as s6
    f1, _ = s6.tail_impute(x[0], m9[0], "tail_tobit")
    assert np.isfinite(f1).all()
    # P12 tar presence (full load happens in D)
    import run_s5_real as s5r
    assert os.path.exists(s5r.TSDM_CACHE.format("A"))
    print("[smoke OK]", flush=True)


# ----------------------------------------------------------------- figure ----
def run_figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # A: synthetic soft vs hard
    ax = axes[0][0]
    As = res.get("A", {}).get("syn", {})
    if As:
        mechs = list(GRIDB_MECHS)
        arm_sel = [("fixed_linear", "#c7c7c7"), ("hard_s16", "#ff7f0e"),
                   ("hard_plug", "#9467bd"), ("soft_cal", "#1f77b4"),
                   ("oracle", "#2ca02c")]
        w = 0.16
        for ai, (a, col) in enumerate(arm_sel):
            vals = [As["mechs"][m][a]["mean"] for m in mechs]
            ax.bar(np.arange(len(mechs)) + (ai - 2) * w, vals, width=w,
                   label=a, color=col)
        ax.set_xticks(range(len(mechs)))
        ax.set_xticklabels(mechs, fontsize=8, rotation=15)
        ax.legend(fontsize=7)
    ax.set_ylabel("mean relMSE vs clean")
    ax.set_title("A. Soft routing (posterior x loss matrix) vs hard, "
                 "synthetic\neval windows (bolt)", fontsize=10)

    # B: real soft vs hard (cens/miss only population)
    ax = axes[0][1]
    Ar = res.get("A", {}).get("real", {})
    rows = []
    for k, lab_k in (("penn_bolt_auto", "Penn bolt\n(auto mask)"),
                     ("penn_timesfm_auto", "Penn timesfm\n(auto)"),
                     ("metrla_bolt", "MLA bolt"),
                     ("metrla_timesfm", "MLA timesfm")):
        if k in Ar:
            sub = Ar[k].get("cens_only") or Ar[k].get("miss_only")
            if sub:
                rows.append((lab_k, sub))
    if rows:
        arm_sel = [("hard_s16", "#ff7f0e"), ("hard_plug", "#9467bd"),
                   ("soft_cal", "#1f77b4"), ("oracle", "#2ca02c")]
        w = 0.2
        for ai, (a, col) in enumerate(arm_sel):
            vals = [r[1][a]["mean"] for r in rows]
            ax.bar(np.arange(len(rows)) + (ai - 1.5) * w, vals, width=w,
                   label=a, color=col)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r[0] for r in rows], fontsize=7)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    ax.set_ylabel("mean NMSE test half (log)")
    ax.set_title("B. Soft vs hard routing on real domains\n(cal-fit L-hat, "
                 "test-half evaluation)", fontsize=10)

    # C: model comparison, three axes
    ax = axes[0][2]
    B = res.get("B", {})
    if B.get("closed_set") and B.get("openset"):
        models = ("gbdt", "mlp_feat", "raw_net")
        closed = [np.mean([B["closed_set"]["_pooled"][m][n]
                           for m in GRIDB_MECHS]) for n in models]
        tr = []
        for n in models:
            vals = []
            if "penn_auto" in B.get("transfer", {}):
                vals.append(B["transfer"]["penn_auto"][n]["cens_mnar_recall"])
            if "metrla" in B.get("transfer", {}):
                vals.append(B["transfer"]["metrla"][n]["miss_block_recall"])
            tr.append(np.mean(vals) if vals else np.nan)
        det = [np.mean([B["openset"]["mechs"][m][n]["det_maha"]
                        for m in ("iburst", "mnar_low")]) for n in models]
        x = np.arange(3)
        w = 0.25
        for i, (vals, lab) in enumerate(((closed, "closed-set acc"),
                                         (tr, "transfer recall"),
                                         (det, "open-set det (maha)"))):
            ax.bar(x + (i - 1) * w, vals, width=w, label=lab)
            for xi, v in zip(x + (i - 1) * w, vals):
                ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)
    ax.set_title("C. Detector model comparison: gbdt vs neural\n(closed-set "
                 "/ cross-domain transfer / open-set)", fontsize=10)

    # D: Task C extrapolation curves
    ax = axes[1][0]
    C = res.get("C", {}).get("cells", {})
    if C:
        for name, col in (("gbdt", "#1f77b4"), ("mlp_feat", "#9467bd"),
                          ("raw_net", "#d62728")):
            bls = [6, 12, 48, 96]
            accs = [np.mean([C[f"block_b{b}_r{r}"][name]["acc"]
                             for r in RATES4 if f"block_b{b}_r{r}" in C
                             and name in C[f"block_b{b}_r{r}"]])
                    for b in bls]
            ax.plot(bls, accs, "o-", color=col, label=name)
        qs = [80, 85, 95, 99]
        for name, col in (("gbdt", "#1f77b4"), ("mlp_feat", "#9467bd"),
                          ("raw_net", "#d62728")):
            accs = [C[f"mnar_q{q}"][name]["acc"] for q in qs
                    if f"mnar_q{q}" in C and name in C[f"mnar_q{q}"]]
            ax.plot([100 - q for q in qs[:len(accs)]], accs, "s--", color=col)
        ax.set_xscale("log")
        ax.set_xlabel("block length (o-) / censor rate in % (s--, log x)")
        ax.legend(fontsize=7)
    ax.set_ylabel("window accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("D. In-family parameter extrapolation\n(block length sweep; "
                 "censor-quantile sweep)", fontsize=10)

    # E: rate extrapolation to 0.9
    ax = axes[1][1]
    if C:
        models = [n for n in ("gbdt", "mlp_feat", "raw_net")
                  if any(n in C.get(f"{m}_r0.9", {}) for m in
                         C_RATE09_MECHS)]
        ref = res.get("C", {}).get("reference_gridB_acc", {})
        w = 0.8 / max(2 * len(models), 1)
        x = np.arange(len(C_RATE09_MECHS))
        mcol = {"gbdt": "#1f77b4", "mlp_feat": "#9467bd", "raw_net": "#d62728"}
        for mi, name in enumerate(models):
            vin = [ref.get(m, {}).get(name, np.nan) for m in C_RATE09_MECHS]
            v09 = [C[f"{m}_r0.9"][name]["acc"] for m in C_RATE09_MECHS]
            ax.bar(x + (2 * mi - len(models) + 1) * w / 2 - w / 2, vin,
                   width=w, color=mcol[name], alpha=0.35,
                   label=f"{name} in-range" if True else None)
            ax.bar(x + (2 * mi - len(models) + 1) * w / 2 + w / 2, v09,
                   width=w, color=mcol[name], label=f"{name} p=0.9")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("mnar_", "m_") for m in C_RATE09_MECHS],
                           fontsize=8)
        ax.legend(fontsize=6, ncol=2)
    ax.set_ylabel("window accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("E. Rate extrapolation p=0.9 (training cap 0.8)\npale: "
                 "in-range mean acc (p<=0.7); solid: p=0.9", fontsize=10)

    # F: PhysioNet'12
    ax = axes[1][2]
    D = res.get("D", {})
    if D.get("transfer") and D.get("e2e"):
        models = ("bolt", "timesfm")
        arm_sel = [("fixed|zero", "#c7c7c7"), ("fixed|linear", "#999999"),
                   ("hard_plug", "#9467bd"), ("soft_cal", "#1f77b4"),
                   ("openset_maha", "#d62728"), ("oracle", "#2ca02c")]
        w = 0.13
        for ai, (a, col) in enumerate(arm_sel):
            vals = [D["e2e"][m][a]["mean"] for m in models if m in D["e2e"]]
            ax.bar(np.arange(len(vals)) + (ai - 2.5) * w, vals, width=w,
                   label=a, color=col)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, fontsize=8)
        ax.set_yscale("log")
        ax.legend(fontsize=6)
        unk = D["transfer"].get("unknown_maha")
        ax.set_xlabel(f"maha-unknown fraction on P12: {unk:.2f}", fontsize=8)
    ax.set_ylabel("mean relMSE vs linear (test, log)")
    ax.set_title("F. PhysioNet'12 third real domain: gated routing\n"
                 "(per-channel linear-normalized relMSE)", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "s24.png"), dpi=150)
    print("saved s24.png", flush=True)


# -------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="smoke",
                    help="comma list: smoke,gate,A,B,C,D,figure,all")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    modes = args.mode.split(",")
    if "all" in modes:
        modes = ["gate", "A", "B", "C", "D", "figure"]
    t0 = time.time()
    res = load_results()
    res.setdefault("meta", dict(
        track="S24: generalization hardening pack for the S16 MechGate "
              "detector/router (soft routing / deep detector comparison / "
              "in-family extrapolation / third real domain)",
        seed=SEED, n_eval=N_EVAL, cal_wi=CAL_WI,
        actions_syn=ACTIONS_SYN,
        stores=["s5_missing_results.json", "s6_mnarfix_results.json",
                "s22_ckpt/mse_block_tobit.npz", "s23_ckpt/eval_*_all_*.json",
                "s10_real_results.json", "s14_metrla_results.json",
                "s5_real_results.json"],
        note="only new s24_* files written; no existing file modified"))
    save_results(res)

    if "smoke" in modes:
        run_smoke(args.device)
        return
    if "gate" in modes:
        run_gate(res)
    if "A" in modes:
        run_A(res, args.device)
    if "B" in modes:
        run_B(res, args.device)
    if "C" in modes:
        run_C(res, args.device)
    if "D" in modes:
        run_D(res)
    if "figure" in modes:
        run_figure(res)
    print(f"[done] modes={modes} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
