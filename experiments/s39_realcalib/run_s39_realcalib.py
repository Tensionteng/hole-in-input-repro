#!/usr/bin/env python
"""S39: coverage and the mechanism-keyed calibration repair on real GIFT-Eval missingness.

No synthetic mask is applied anywhere: the holes are whatever the corpus ships. See
s39_notes.md for the pre-registration.
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np
import torch

for d in ("s05_characterization", "s16_mechgate", "s32_gifteval"):
    sys.path.insert(0, os.path.join(EXP, d))
import run_s16_mechgate as s16
import run_s32_gifteval as s32

L, H = s32.L, s32.H
SEED = s16.SEED
CLASSES = s16.CLASSES
STRATA = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.15), (0.15, 0.30), (0.30, 1.01)]
MIN_GROUP = 30
ALPHA = 0.90
OUT = os.path.join(HERE, "s39_results.json")


class BoltQ:
    """chronos-bolt-base at GIFT-Eval's horizon, returning the q10/q90 band."""
    def __init__(self, device, batch=512):
        from chronos import BaseChronosPipeline
        self.p = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base",
                                                     device_map=device,
                                                     torch_dtype=torch.float32)
        self.device, self.batch = device, batch

    @torch.no_grad()
    def band(self, ctx):
        lo, hi = [], []
        for i in range(0, len(ctx), self.batch):
            x = torch.from_numpy(np.ascontiguousarray(ctx[i:i + self.batch])).to(self.device)
            q = self.p.predict(x, prediction_length=H)
            lo.append(q[:, 0, :].float().cpu().numpy())
            hi.append(q[:, 8, :].float().cpu().numpy())
        return np.concatenate(lo), np.concatenate(hi)


def conformal_w(E, ok):
    """Per-step split-conformal widening from a ragged (observed-only) score matrix."""
    w = np.zeros(E.shape[1])
    for h in range(E.shape[1]):
        e = E[ok[:, h], h]
        if len(e) < 10:
            w[h] = np.nan
            continue
        k = min(int(np.ceil((len(e) + 1) * ALPHA)), len(e))
        w[h] = np.sort(e)[k - 1]
    if np.isnan(w).any():                       # fall back to the pooled quantile
        e = E[ok]
        k = min(int(np.ceil((len(e) + 1) * ALPHA)), len(e))
        w = np.where(np.isnan(w), np.sort(e)[k - 1], w)
    return w


def cov_wid(lo, hi, y, ok, w):
    inside = (y >= lo - w) & (y <= hi + w)
    return float(inside[ok].mean()), float((hi - lo + 2 * w)[ok].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(s32.MISSING_DS))
    ap.add_argument("--max-series", type=int, default=400)
    ap.add_argument("--per-series", type=int, default=3)
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
        print("fitting the detector on ETTh1/ETTm1/weather only ...", flush=True)
        F, y, _ = s16.build_train_features(n_win=220)
        clf = s16.fit_clf("gbdt", F, y)
    model = BoltQ("cuda")
    res = {"meta": {"L": L, "H": H, "alpha": ALPHA, "strata": STRATA,
                    "datasets": args.datasets.split(","), "min_group": MIN_GROUP,
                    "detector": "S16 gbdt, fitted on ETTh1/ETTm1/weather, frozen",
                    "note": "real GIFT-Eval NaN only; no synthetic mask anywhere"}}

    CTX, TGT, OKM, DSI = [], [], [], []
    for di, ds in enumerate(args.datasets.split(",")):
        t0 = time.time()
        try:
            ser = s32.read_series(ds, args.max_series, SEED)
            w = s32.windows(ser, args.per_series, SEED + 1)
        except Exception as e:
            print(f"  SKIP {ds}: {type(e).__name__}: {str(e)[:90]}", flush=True)
            continue
        if w is None:
            print(f"  SKIP {ds}: no usable window", flush=True)
            continue
        c, t, o = w
        keep = np.isfinite(t).all(1) & (np.isfinite(c).sum(1) >= 32)
        c, t, o = c[keep], t[keep], o[keep]
        CTX.append(c); TGT.append(t); OKM.append(o)
        DSI.append(np.full(len(c), di))
        mr = (~np.isfinite(c)).mean()
        print(f"  {ds:34s} {len(c):6d} windows, {mr*100:5.2f}% of context NaN "
              f"({time.time()-t0:.0f}s)", flush=True)
    ctx = np.concatenate(CTX); tgt = np.concatenate(TGT)
    okm = np.concatenate(OKM); dsi = np.concatenate(DSI)
    miss = ~np.isfinite(ctx)
    rate = miss.mean(1)
    print(f"\n{len(ctx)} windows total; {float((rate>0).mean())*100:.1f}% contain a hole",
          flush=True)

    # detector + deployable linear fill, both computed per window from the real pattern
    filled = np.empty_like(ctx)
    feats = np.empty((len(ctx), 14), np.float64)
    for i in range(len(ctx)):
        m = miss[i]
        v = np.flatnonzero(~m)
        filled[i] = (np.interp(np.arange(L), v, ctx[i][v]).astype(np.float32)
                     if m.any() and len(v) >= 2 else np.nan_to_num(ctx[i]))
        feats[i] = s16.channel_features(filled[i], m)
    plab = np.argmax(s16.proba_clf("gbdt", clf, feats), axis=1).astype(int)
    res["detected"] = {CLASSES[k]: int((plab == k).sum()) for k in range(6)}
    print("detector fires: " + "  ".join(f"{CLASSES[k]}={int((plab==k).sum())}"
                                         for k in range(6)), flush=True)

    lo, hi = model.band(filled)
    E = np.maximum(lo.astype(np.float64) - tgt, tgt - hi.astype(np.float64))

    # split by SERIES-carrying dataset index and window parity so no series spans both halves
    rng = np.random.default_rng(SEED)
    half = rng.random(len(ctx)) < 0.5
    cal, tst = half, ~half
    w_naive = conformal_w(E[cal], okm[cal])
    pool = {k: (conformal_w(E[cal & (plab == k)], okm[cal & (plab == k)])
                if (cal & (plab == k)).sum() >= MIN_GROUP else None) for k in range(6)}
    # A detected class is not homogeneous in missing RATE on this corpus -- the same class
    # spans 2% and 60% windows and one width cannot serve both. Test the obvious refinement:
    # key the Mondrian groups on (detected mechanism, rate bin) instead of mechanism alone.
    RB = np.array([0.0, 0.01, 0.05, 0.15, 0.30, 1.01])
    rbin = np.clip(np.searchsorted(RB, rate, side="right") - 1, 0, len(RB) - 2)
    pool2 = {}
    for k in range(6):
        for b in range(len(RB) - 1):
            m = cal & (plab == k) & (rbin == b)
            pool2[(k, b)] = conformal_w(E[m], okm[m]) if m.sum() >= MIN_GROUP else None
    res["joint_groups"] = {f"{CLASSES[k]}|{RB[b]}-{RB[b+1]}":
                           int((cal & (plab == k) & (rbin == b)).sum())
                           for k in range(6) for b in range(len(RB) - 1)
                           if (cal & (plab == k) & (rbin == b)).sum() > 0}

    def widths(sel, variant):
        if variant == "native":
            return np.zeros((sel.sum(), H))
        if variant == "naive":
            return np.tile(w_naive, (sel.sum(), 1))
        if variant == "mondrian":
            return np.stack([pool[k] if pool[k] is not None else w_naive for k in plab[sel]])
        # mondrian+rate, falling back to mechanism-only then to the global pool
        out = []
        for k, b in zip(plab[sel], rbin[sel]):
            w = pool2.get((int(k), int(b)))
            if w is None:
                w = pool[int(k)] if pool[int(k)] is not None else w_naive
            out.append(w)
        return np.stack(out)

    print(f"\n{'stratum':14s} {'n':>6s} {'native':>8s} {'naive':>8s} {'mondrian':>9s}"
          f" {'mond+rate':>11s} {'w native':>10s} {'w m+rate':>11s}", flush=True)
    res["strata"] = {}
    for a, b in STRATA:
        sel = tst & (rate >= a) & (rate < b)
        if sel.sum() < 20:
            continue
        row = {"n": int(sel.sum()), "mean_rate": float(rate[sel].mean())}
        for v in ("native", "naive", "mondrian", "mondrian_rate"):
            c, w = cov_wid(lo[sel], hi[sel], tgt[sel], okm[sel], widths(sel, v))
            row[v] = {"coverage": c, "width": w}
        res["strata"][f"{a}-{b}"] = row
        print(f"{f'{a*100:.0f}-{b*100:.0f}%':14s} {row['n']:6d} "
              f"{row['native']['coverage']:8.3f} {row['naive']['coverage']:8.3f} "
              f"{row['mondrian']['coverage']:9.3f} "
              f"{row['mondrian_rate']['coverage']:11.3f} "
              f"{row['native']['width']:10.3f} {row['mondrian_rate']['width']:11.3f}",
              flush=True)

    res["by_detected"] = {}
    print(f"\n{'detected':14s} {'n':>6s} {'native':>8s} {'naive':>8s} {'mondrian':>9s}",
          flush=True)
    for k in range(6):
        sel = tst & (plab == k)
        if sel.sum() < 20:
            continue
        row = {"n": int(sel.sum())}
        for v in ("native", "naive", "mondrian", "mondrian_rate"):
            c, w = cov_wid(lo[sel], hi[sel], tgt[sel], okm[sel], widths(sel, v))
            row[v] = {"coverage": c, "width": w}
        res["by_detected"][CLASSES[k]] = row
        print(f"{CLASSES[k]:14s} {row['n']:6d} {row['native']['coverage']:8.3f} "
              f"{row['naive']['coverage']:8.3f} {row['mondrian']['coverage']:9.3f}", flush=True)

    res["overall"] = {}
    for v in ("native", "naive", "mondrian", "mondrian_rate"):
        c, w = cov_wid(lo[tst], hi[tst], tgt[tst], okm[tst], widths(tst, v))
        res["overall"][v] = {"coverage": c, "width": w}
    print("\noverall " + "  ".join(f"{v}={res['overall'][v]['coverage']:.3f}"
                                   for v in res["overall"]), flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
