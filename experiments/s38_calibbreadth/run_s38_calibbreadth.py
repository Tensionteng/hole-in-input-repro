#!/usr/bin/env python
"""S38: the mechanism-keyed calibration repair on nine datasets.

The detector and the protocol are S16's, unchanged. The only thing that changes is the set of
datasets it is deployed on: six of the nine were never seen by the detector during fitting.
See s38_notes.md for the pre-registration.
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
# One-row predict_proba calls fan out over every core; with several rounds in flight the
# machine spends its time in the thread pool. Cap it and batch the calls instead.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import torch

for d in ("s05_characterization", "s06_real_and_repairs", "s16_mechgate", "s35_breadth"):
    sys.path.insert(0, os.path.join(EXP, d))
import run_s16_mechgate as s16
import run_s35_breadth as s35

L, H = s16.L, s16.H
SEED = s16.SEED
CLASSES = s16.CLASSES
MIX = [(s16.CLEAN, 0.25), (s16.MCAR, 0.30), (s16.BLOCK, 0.20),
       (s16.MNAR_H, 0.15), (s16.MNAR_E, 0.10)]
MPROB = np.array([p for _, p in MIX])
MIN_GROUP = 30
INSAMPLE = ("ETTh1", "ETTm1", "weather")
OUT = os.path.join(HERE, "s38_results.json")


def starts_for(name, n_eval, n_cal, seed=SEED):
    """Evaluation and calibration starts, disjoint, both from the test region.

    S35's sampler is written for H=64; this round forecasts H=96, so the last valid start
    is 32 steps earlier and must be recomputed here or short targets reach np.stack."""
    X, _ = s35.load(name, 1, seed)
    if X is None:
        return None, None, None
    N = len(X)
    lo, hi = int(0.8 * N) - L, N - L - H
    if hi < lo:
        return None, None, None
    valid = np.arange(lo, hi + 1)
    rng = np.random.default_rng(seed)
    # illness has only ~100 valid starts, so cap the evaluation half or the calibration
    # pool comes out empty.
    ev = np.sort(rng.choice(valid, size=min(n_eval, len(valid) // 2), replace=False))
    pool = np.setdiff1d(valid, ev)
    rng = np.random.default_rng(seed + 999)
    ca = np.sort(rng.choice(pool, size=min(n_cal, len(pool)), replace=False))
    return X, ev, ca


def build_split(X, starts, clf, sid, keep_ch):
    """One deployment stream: sample a mechanism per series, fill by the DETECTED class.

    Features are collected for the whole split and classified in a single batched call."""
    feats, raws, masks, ys, tl = [], [], [], [], []
    for wi, s in enumerate(starts):
        xw = X[s:s + L].T
        yw = X[s + L:s + L + H].T
        for c in keep_ch:
            if not (np.isfinite(xw[c]).all() and np.isfinite(yw[c]).all()):
                continue
            if xw[c].std() < 1e-8:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 31 + sid, wi, int(c)]))
            mi = int(rng.choice(len(MIX), p=MPROB))
            tlab = MIX[mi][0]
            rate = float(rng.uniform(0.1, 0.6))
            if tlab == s16.CLEAN:
                m = np.zeros(L, bool)
            elif tlab == s16.MCAR:
                m = s16.s5.make_mask("mcar", rate, wi, 5000 + sid * 997 + int(c), 1)[0]
            elif tlab == s16.BLOCK:
                m = s16.block_mask_1d(rate, L, rng, 24)
            else:
                mech = "mnar_high" if tlab == s16.MNAR_H else "mnar_extreme"
                m = s16.s5.make_mask(mech, rate, wi, 0, 1, x=xw[c][None, :])[0]
            feats.append(s16.channel_features(xw[c], m))
            raws.append(xw[c].copy())
            masks.append(m)
            ys.append(yw[c])
            tl.append(tlab)
    pl = np.argmax(s16.proba_clf("gbdt", clf, np.stack(feats)), axis=1)
    ctxs = [s16.routed_fill_synth(r, mk, int(p)) for r, mk, p in zip(raws, masks, pl)]
    return (np.stack(ctxs).astype(np.float32), np.stack(ys).astype(np.float64),
            np.array(tl), pl.astype(int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(s35.DATASETS))
    ap.add_argument("--n-eval", type=int, default=120)
    ap.add_argument("--n-cal", type=int, default=120)
    ap.add_argument("--max-ch", type=int, default=12)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    ck = os.path.join(EXP, "s16_mechgate", "s16_ckpt", "gbdt_frozen.pkl")
    if os.path.exists(ck):
        import pickle
        clf = pickle.load(open(ck, "rb"))
        print("loaded the frozen detector from", ck, flush=True)
    else:
        print("fitting the detector on the three original datasets only ...", flush=True)
        F, y, _ = s16.build_train_features(n_win=220)
        clf = s16.fit_clf("gbdt", F, y)
    model = s16.Bolt96("cuda")

    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    res.setdefault("meta", {"L": L, "H": H, "alpha": 0.90, "mix": [[c, p] for c, p in MIX],
                            "n_eval": args.n_eval, "n_cal": args.n_cal,
                            "max_ch": args.max_ch, "insample": list(INSAMPLE),
                            "detector": "S16 gbdt, fitted on ETTh1/ETTm1/weather, frozen",
                            "classes": CLASSES, "min_group": MIN_GROUP})

    for name in args.datasets.split(","):
        t0 = time.time()
        X, ev, ca = starts_for(name, args.n_eval, args.n_cal)
        if X is None:
            print(f"SKIP {name}", flush=True)
            continue
        rng = np.random.default_rng(SEED)
        C = X.shape[1]
        keep_ch = np.sort(rng.choice(C, min(args.max_ch, C), replace=False))
        cal = build_split(X, ca, clf, 0, keep_ch)
        tst = build_split(X, ev, clf, 1, keep_ch)
        (cx, cy, ctl, cpl), (tx, ty, ttl, tpl) = cal, tst
        cq10, cq90 = model.predict_quantiles(cx)
        tq10, tq90 = model.predict_quantiles(tx)
        E = np.maximum(cq10.astype(np.float64) - cy, cy - cq90.astype(np.float64))
        w_naive = s16.conformal_w(E)
        pool_p = {k: (s16.conformal_w(E[cpl == k]) if (cpl == k).sum() >= MIN_GROUP else None)
                  for k in range(6)}
        pool_t = {k: (s16.conformal_w(E[ctl == k]) if (ctl == k).sum() >= MIN_GROUP else None)
                  for k in range(6)}
        acc = float((tpl == ttl).mean())
        tag = "in-sample" if name in INSAMPLE else "HELD OUT"
        print(f"\n== {name} ({tag}): {len(tx)} test / {len(cx)} cal series, "
              f"detector acc {acc:.3f}, {time.time()-t0:.0f}s ==", flush=True)
        print(f"  {'group':13s} {'n':>5s} {'native':>8s} {'naive':>8s} "
              f"{'mondrian':>9s} {'oracle':>8s} {'width nat':>10s} {'width mon':>10s}",
              flush=True)
        rec = {"detector_acc": acc, "n_test": len(tx), "n_cal": len(cx), "groups": {}}
        for k in range(6):
            sel = ttl == k
            if sel.sum() < 10:
                continue
            row = {"n": int(sel.sum())}
            for v in ("native", "naive", "mondrian_pred", "mondrian_oracle"):
                if v == "native":
                    w = np.zeros(H)
                elif v == "naive":
                    w = w_naive
                else:
                    pool = pool_p if v == "mondrian_pred" else pool_t
                    key = tpl[sel] if v == "mondrian_pred" else ttl[sel]
                    w = np.stack([pool[int(kk)] if pool[int(kk)] is not None else w_naive
                                  for kk in key])
                cov, wid = s16.cov_wid(tq10[sel], tq90[sel], ty[sel], w)
                row[v] = {"coverage": cov, "width": wid}
            rec["groups"][CLASSES[k]] = row
            print(f"  {CLASSES[k]:13s} {row['n']:5d} {row['native']['coverage']:8.3f} "
                  f"{row['naive']['coverage']:8.3f} {row['mondrian_pred']['coverage']:9.3f} "
                  f"{row['mondrian_oracle']['coverage']:8.3f} "
                  f"{row['native']['width']:10.3f} {row['mondrian_pred']['width']:10.3f}",
                  flush=True)
        allrow = {}
        for v in ("native", "naive", "mondrian_pred"):
            w = (np.zeros(H) if v == "native" else w_naive if v == "naive" else
                 np.stack([pool_p[int(k)] if pool_p[int(k)] is not None else w_naive
                           for k in tpl]))
            cov, wid = s16.cov_wid(tq10, tq90, ty, w)
            allrow[v] = {"coverage": cov, "width": wid}
        rec["overall"] = allrow
        print("  overall       " + " ".join(f"{allrow[v]['coverage']:8.3f}"
                                            for v in allrow), flush=True)
        res.setdefault("datasets", {})[name] = rec
        json.dump(res, open(args.out + ".tmp", "w"))
        os.replace(args.out + ".tmp", args.out)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
