#!/usr/bin/env python
"""S26: does S25's term (3) -- the recoverable fill gap -- survive on REAL missingness?

S25 decomposed the degradation of a frozen TSFM into
    (1) information floor + (2) architectural floor + (3) fill/routing,
and found, on SYNTHETIC mechanisms, that term (3) is ~0 under mcar/block but is the
dominant term under value censoring (1.203 of a 1.875 total at mnar_high p=0.7; a learned
fill through the frozen model beat the best fixed fill on 81% of windows, paired).

The entire revised system story rests on that term being real rather than an artefact of
synthetic masks. S26 tests it on the two real mechanisms the project already owns, and the
decomposition makes a RISKY, DIRECTIONAL prediction that can fail either way:

  H1  Penmanshiel wind-farm curtailment (real VALUE CENSORING):
      a learned fill BEATS the best fixed fill (zero, NMSE 28.43 stored in S10).
  H2  METR-LA sensor outages (real BLOCK geometry):
      a learned fill does NOT beat the best fixed fill (nan, NMSE 2.00 stored in S14).

The contrast H1 and not-H2 is the test. If both win, "learning the fill always helps" and
the mechanism-specific claim is wrong (weaker result, and it would undercut the routing
story). If neither wins, term (3) is a synthetic artefact and the revised repair ladder is
wrong. Both outcomes are recorded as-is.

Windows, masks and metrics are the stored ones: `run_s6_real_censor.build_windows/extract`
(CTX=144, H=24, 602 cens + 602 ctrl, NMSE with a 100 kW^2 var floor) and
`run_s14_metrla.build_windows/extract` (CTX=512, 618 miss + 618 ctrl, 4.0 mph^2 var floor).
Anchor gates re-run the stored S10/S14 cells before anything else.

Train/eval split: the S10 per-turbine (resp. per-sensor) 50/50 cal/test split with
SEED_SPLIT=20250812. The fill net is fit on the CAL half only and every number reported is
on the held-out TEST half. Baselines are recomputed on the same TEST half so the comparison
is paired window-for-window.

Deviation, disclosed: METR-LA uses H=64 (bolt's native single-block horizon) for the
learned-fill arm, because H=96 triggers the pipeline's autoregressive quantile-mixing
rollout, which is not a clean object to differentiate through. The METR-LA anchor gate runs
at the stored H=96 to prove the harness reproduces S14; the H=64 arm carries its own
recomputed baselines and is internally consistent.

Missingness mask: the ORACLE mask (operator status codes for Penmanshiel, exact zeros for
METR-LA), matching S10/S14's main tracks. This isolates "is a better fill available" from
"can the mechanism be detected" (S16 measures the latter: 0.733 censoring recall with the
deployable status-free mask). Deployment needs both.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "tsfm_missing", "data")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(EXP, "s06_real_and_repairs"))
sys.path.insert(0, os.path.join(EXP, "s14_metrla"))
sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))

import run_s6_real_censor as s6r
import run_s14_metrla as s14
import run_s25_twofloor as s25

# the migrated layout broke the modules' HERE-relative data paths
s6r.NPZ = os.path.join(DATA, "dataset_censor", "penmanshiel2016_processed.npz")
s14.H5 = os.path.join(DATA, "dataset_metrla", "metr-la.h5")

SEED_SPLIT = 20250812
OUT = os.path.join(HERE, "s26_results.json")
CK = os.path.join(HERE, "s26_ckpt")

# stored cells the anchor gates must reproduce (bolt)
ANCHOR_PENN = {          # S10 records, 602 windows, CTX=144 H=24, mean NMSE
    "cens|keep": 40.28983, "cens|zero": 28.42657, "cens|linear": 60.16071,
    "ctrl|clean": 9.91381,
}
ANCHOR_METR = {          # S14 records, 618 windows, CTX=512 H=96, mean NMSE
    "miss|keep": 21.19334, "miss|linear": 2.74913, "miss|nan": 1.99908,
    "ctrl|clean": 3.19901,
}


# ------------------------------------------------------------------ data ----

class Penn:
    """Penmanshiel curtailment. Censored points are CLAMPED, not deleted:
    y_obs = min(y_true, cap). 'keep' feeds the clamped value, the other fills
    replace it."""
    name, CTX, H, VAR_FLOOR = "penn", 144, 24, 100.0

    def __init__(self):
        self.windows, self.series = s6r.build_windows(s6r.NPZ)
        self.ctx_obs, self.ctx_rec, self.tgt = s6r.extract(self.windows, self.series)
        self.mask = ~np.isfinite(self.ctx_obs)              # True = censored
        self.group = np.array([w["kind"] for w in self.windows])
        self.strat = np.array([w["turbine"] for w in self.windows])

    def split(self, kind):
        return cal_test_split(self.strat, self.group, kind)


class Metr:
    name, CTX, H, VAR_FLOOR = "metr", 512, 64, 4.0

    def __init__(self):
        self.windows = s14.build_windows()
        d = s14.extract(self.windows)
        self.ctx_obs, self.ctx_rec = d["ctx_obs"], d["ctx_rec"]
        self.mask = d["mask"]
        self.tgt_full = d["tgt"]                            # H=96 as stored
        self.tgt = d["tgt"][:, :self.H]                     # H=64 arm
        self.group = np.array(["miss" if w["kind"] == "miss" else "ctrl"
                               for w in self.windows])
        self.strat = np.array([w["sensor"] for w in self.windows])

    def split(self, kind):
        return cal_test_split(self.strat, self.group, kind)


def cal_test_split(strat, group, kind, seed=SEED_SPLIT):
    """Per-stratum 50/50 split, same algorithm as run_s10_real.cal_test_split."""
    by = {}
    for i in range(len(group)):
        if group[i] == kind:
            by.setdefault(strat[i], []).append(i)
    rng = np.random.default_rng(seed)
    cal, test = [], []
    for tag in sorted(by):
        lst = by[tag]
        perm = rng.permutation(len(lst))
        half = len(lst) // 2
        cal += [lst[k] for k in perm[:half]]
        test += [lst[k] for k in perm[half:]]
    return np.array(sorted(cal)), np.array(sorted(test))


def fill_ctx(ctx_obs, ctx_rec, mask, fill):
    """ctx_obs has NaN at missing. Returns [n, CTX] float32 (nan only for fill='nan')."""
    out = ctx_obs.copy()
    if fill == "keep":
        return ctx_rec.copy()
    if fill == "nan":
        return out
    if fill == "zero":
        return np.nan_to_num(out, nan=0.0)
    if fill == "linear":
        t = np.arange(out.shape[1])
        for i in range(len(out)):
            v = np.flatnonzero(np.isfinite(out[i]))
            out[i] = np.zeros(out.shape[1], np.float32) if len(v) == 0 else \
                np.interp(t, v, out[i][v]).astype(np.float32)
        return out
    raise ValueError(fill)


def nmse_per_window(pred, tgt, var_floor):
    var = np.maximum(tgt.var(axis=1), var_floor)
    return ((pred - tgt) ** 2).mean(axis=1) / var



def synth_curtail(ctx, rng, rate_lo=0.05, rate_hi=0.6):
    """Clamp the top-k values of a clean window to the k-th largest -- the geometry of a
    real curtailment episode (a plateau at the cap), not synthetic deletion."""
    out, msk = ctx.copy(), np.zeros(ctx.shape, bool)
    for i in range(len(ctx)):
        k = int(np.ceil(float(rng.uniform(rate_lo, rate_hi)) * ctx.shape[1]))
        if k < 1:
            continue
        order = np.argsort(-ctx[i], kind="stable")
        top = order[:k]
        cap = ctx[i, order[k - 1]]
        out[i, top] = cap
        msk[i, top] = True
    return out, msk


def augment_pool(ds, kind_clean, idx_clean, n_mult, seed=11):
    """Synthetically curtailed copies of the clean training windows."""
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n_mult):
        c = ds.ctx_rec[idx_clean].copy()
        o, m = synth_curtail(c, rng)
        reps.append((o, m, ds.tgt[idx_clean]))
    return (np.concatenate([r[0] for r in reps]),
            np.concatenate([r[1] for r in reps]),
            np.concatenate([r[2] for r in reps]))

# ------------------------------------------------------------------ nets ----

def robust_scale(ctx_obs):
    """Loss weight from the OBSERVED part only (no clean context exists on real data)."""
    sc = np.empty((len(ctx_obs), 1), np.float32)
    for i, r in enumerate(ctx_obs):
        v = r[np.isfinite(r)]
        s = np.std(v) if len(v) > 1 else 0.0
        a = np.abs(v).mean() if len(v) else 0.0
        sc[i, 0] = max(s, 0.01 * a, 1e-3)
    return sc


def prep(filled, mask, device):
    obs = (~mask).astype(np.float32)
    n = obs.sum(1, keepdims=True).clip(1)
    mu = (filled * obs).sum(1, keepdims=True) / n
    sd = np.sqrt(((filled - mu) ** 2 * obs).sum(1, keepdims=True) / n) + 1e-6
    return (torch.from_numpy((filled - mu) / sd).to(device),
            torch.from_numpy(obs).to(device),
            torch.from_numpy(mu).to(device), torch.from_numpy(sd).to(device))


def train_fillnet(bolt, ds, idx, conv, steps=800, bs=64, lr=1e-4, log=200, seed=7,
                  extra=None, center="linear"):
    """Fit g_theta on the CAL half, through the frozen model. Same bounded fill class as
    S25 (tanh-bounded correction within 3 observed-sigma of the linear interpolant, zero
    init = exactly linear), and the same NaN guards: bolt's InstanceNorm back-propagates
    sqrt(0)=inf on a constant filled context."""
    dev = bolt.device
    net = s25.FillNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)

    obs = ds.ctx_obs[idx].copy()
    rec = ds.ctx_rec[idx]
    msk = ds.mask[idx]
    tgt = ds.tgt[idx]
    if extra is not None:                      # synthetic-curtailment augmentation
        e_rec, e_msk, e_tgt = extra
        e_obs = e_rec.copy()
        e_obs[e_msk] = np.nan
        obs = np.concatenate([obs, e_obs])
        rec = np.concatenate([rec, e_rec])
        msk = np.concatenate([msk, e_msk])
        tgt = np.concatenate([tgt, e_tgt])
    lin = fill_ctx(obs, rec, msk, center)
    sc_all = robust_scale(obs)
    keep = np.array([np.std(lin[i]) > 1e-8 * (abs(np.mean(lin[i])) + 1e-6)
                     for i in range(len(lin))])
    lin, msk, tgt, sc_all = lin[keep], msk[keep], tgt[keep], sc_all[keep]
    print(f"  train pool: {len(lin)} windows ({int((~keep).sum())} flat dropped)", flush=True)

    hist, n_skip = [], 0
    for it in range(steps):
        b = rng.choice(len(lin), size=min(bs, len(lin)), replace=False)
        v, m, mu, sd = prep(lin[b], msk[b], dev)
        y = torch.from_numpy(tgt[b]).to(dev)
        sc = torch.from_numpy(sc_all[b]).to(dev)
        x = net(v, m) * sd + mu
        pred = bolt.median(x, m if conv == "fill_mask" else None)[:, :ds.H]
        loss = s25.huber_scaled(pred, y, sc)
        if not torch.isfinite(loss):
            n_skip += 1
            continue
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not torch.isfinite(gn):
            opt.zero_grad()
            n_skip += 1
            continue
        opt.step()
        sch.step()
        hist.append(float(loss))
        if it == 0 or (it + 1) % log == 0:
            print(f"    fillnet[{ds.name}/{conv}] {it+1}/{steps} "
                  f"loss={np.mean(hist[-log:]):.4f} skip={n_skip}", flush=True)
    assert all(torch.isfinite(q).all() for q in net.parameters()), "fill net went non-finite"
    return net, hist


@torch.no_grad()
def eval_fillnet(bolt, net, ds, idx, conv, batch=256, center="linear"):
    lin = fill_ctx(ds.ctx_obs[idx], ds.ctx_rec[idx], ds.mask[idx], center)
    msk, tgt = ds.mask[idx], ds.tgt[idx]
    outs = []
    for i in range(0, len(lin), batch):
        v, m, mu, sd = prep(lin[i:i + batch], msk[i:i + batch], bolt.device)
        x = net(v, m) * sd + mu
        assert torch.isfinite(x).all(), "fill net emitted non-finite fill values"
        outs.append(bolt.median(x, m if conv == "fill_mask" else None)[:, :ds.H]
                    .float().cpu().numpy())
    return nmse_per_window(np.concatenate(outs), tgt, ds.VAR_FLOOR)


# ------------------------------------------------------------------ gate ----

def gate(res, bolt):
    out = {}
    penn = Penn()
    for key, stored in ANCHOR_PENN.items():
        grp, fill = key.split("|")
        idx = np.flatnonzero(penn.group == grp)
        f = "keep" if fill == "clean" else fill
        ctx = fill_ctx(penn.ctx_obs[idx], penn.ctx_rec[idx], penn.mask[idx], f)
        pred = bolt.median_rollout_np(ctx, penn.H)
        v = float(nmse_per_window(pred, penn.tgt[idx], penn.VAR_FLOOR).mean())
        d = abs(v - stored) / stored
        out[f"penn|{key}"] = {"stored": stored, "rerun": v, "rel_dev": d, "pass": d <= 0.05}
        print(f"GATE penn {key:14s} stored={stored:9.5f} rerun={v:9.5f} dev={d:+.4%} "
              f"{'PASS' if d <= 0.05 else 'FAIL'}", flush=True)

    metr = Metr()
    for key, stored in ANCHOR_METR.items():
        grp, fill = key.split("|")
        idx = np.flatnonzero(metr.group == grp)
        f = "keep" if fill == "clean" else fill
        ctx = fill_ctx(metr.ctx_obs[idx], metr.ctx_rec[idx], metr.mask[idx], f)
        pred = bolt.median_rollout_np(ctx, 96)          # stored horizon
        v = float(nmse_per_window(pred, metr.tgt_full[idx], metr.VAR_FLOOR).mean())
        d = abs(v - stored) / stored
        out[f"metr|{key}"] = {"stored": stored, "rerun": v, "rel_dev": d, "pass": d <= 0.05}
        print(f"GATE metr {key:14s} stored={stored:9.5f} rerun={v:9.5f} dev={d:+.4%} "
              f"{'PASS' if d <= 0.05 else 'FAIL'}", flush=True)
    res["gate"] = out
    assert all(v["pass"] for v in out.values()), "anchor gate FAILED -- stop"
    return res, penn, metr


# ------------------------------------------------------------------- arm ----

FILLS = {"penn": ("keep", "zero", "linear", "nan"),
         "metr": ("keep", "zero", "linear", "nan")}


def arm(res, bolt, ds, kind, steps, aug_mult=0, clean_kind="ctrl", center="linear"):
    os.makedirs(CK, exist_ok=True)
    cal, test = ds.split(kind)
    print(f"\n== {ds.name}/{kind}: {len(cal)} cal (train) / {len(test)} test", flush=True)
    cell = {"n_cal": int(len(cal)), "n_test": int(len(test))}

    fixed = {}
    for f in FILLS[ds.name]:
        ctx = fill_ctx(ds.ctx_obs[test], ds.ctx_rec[test], ds.mask[test], f)
        pred = bolt.median_np(ctx)[:, :ds.H]
        fixed[f] = nmse_per_window(pred, ds.tgt[test], ds.VAR_FLOOR)
        cell[f"fixed_{f}"] = {"mean": float(fixed[f].mean()),
                              "median": float(np.median(fixed[f]))}
        print(f"  fixed {f:7s} NMSE mean={fixed[f].mean():9.4f} "
              f"median={np.median(fixed[f]):7.4f}", flush=True)
    best_mean = min(FILLS[ds.name], key=lambda f: fixed[f].mean())
    best_med = min(FILLS[ds.name], key=lambda f: np.median(fixed[f]))
    cell["best_fixed_by_mean"], cell["best_fixed_by_median"] = best_mean, best_med

    extra = None
    if aug_mult:
        cal_c, _ = ds.split(clean_kind)
        extra = augment_pool(ds, clean_kind, cal_c, aug_mult)
        print(f"  augmenting with {len(extra[0])} synthetically curtailed clean windows",
              flush=True)
    for conv in ("plain_fill", "fill_mask"):
        t0 = time.time()
        net, hist = train_fillnet(bolt, ds, cal, conv, steps=steps, extra=extra,
                                  center=center)
        torch.save(net.state_dict(),
                   os.path.join(CK,
                                f"fillnet_{ds.name}_{kind}_{conv}_aug{aug_mult}_{center}.pt"))
        e = eval_fillnet(bolt, net, ds, test, conv, center=center)
        e_tr = eval_fillnet(bolt, net, ds, cal, conv, center=center)
        ctx_tr = fill_ctx(ds.ctx_obs[cal], ds.ctx_rec[cal], ds.mask[cal], best_mean)
        fx_tr = nmse_per_window(bolt.median_np(ctx_tr)[:, :ds.H], ds.tgt[cal], ds.VAR_FLOOR)
        cell[f"learned_{conv}"] = {
            "mean": float(e.mean()), "median": float(np.median(e)),
            "train_mean": float(e_tr.mean()), "train_median": float(np.median(e_tr)),
            "train_bestfixed_mean": float(fx_tr.mean()),
            "train_bestfixed_median": float(np.median(fx_tr)),
            "train_winrate": float((e_tr < fx_tr).mean()),
            "winrate_vs_best_mean": float((e < fixed[best_mean]).mean()),
            "winrate_vs_best_median": float((e < fixed[best_med]).mean()),
            "per_window": e.tolist(), "final_loss": float(np.mean(hist[-50:])),
        }
        print(f"  learned {conv:11s} TEST mean={e.mean():9.4f} med={np.median(e):7.4f} "
              f"win%={100*(e < fixed[best_mean]).mean():.0f}%  |  TRAIN mean={e_tr.mean():9.4f} "
              f"med={np.median(e_tr):7.4f} (bestfix {fx_tr.mean():9.4f}/"
              f"{np.median(fx_tr):.4f}) win%={100*(e_tr < fx_tr).mean():.0f}% "
              f"({time.time()-t0:.0f}s)", flush=True)
    for f in FILLS[ds.name]:
        cell[f"fixed_{f}"]["per_window"] = fixed[f].tolist()
    cell["aug_mult"], cell["center"] = aug_mult, center
    res.setdefault("arms", {})[f"{ds.name}|{kind}|aug{aug_mult}|c-{center}"] = cell
    return res


def save(res):
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="gate,penn,metr")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(20250810)
    np.random.seed(20250810)

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {
        "penn": "CTX=144 H=24, S6/S10 windows, NMSE var floor 100 kW^2",
        "metr": "CTX=512 H=64 (S14 windows; stored horizon 96 used only by the gate), "
                "NMSE var floor 4.0 mph^2",
        "split": "per-turbine/-sensor 50/50 cal(train)/test, seed 20250812",
        "mask": "oracle (operator status codes / exact zeros)",
        "fill_class": "tanh-bounded, 3 observed-sigma around the linear interpolant",
        "model": "amazon/chronos-bolt-base (frozen)",
    })
    bolt = s25.Bolt("cuda")
    steps = 60 if args.smoke else args.steps

    parts = args.parts.split(",")
    penn = metr = None
    if "gate" in parts:
        res, penn, metr = gate(res, bolt)
        save(res)
    if "penn" in parts:
        penn = penn or Penn()
        res = arm(res, bolt, penn, "cens", steps, center="zero")     # baseline inside the class
        res = arm(res, bolt, penn, "cens", steps, aug_mult=6, center="zero")
        save(res)
    if "metr" in parts:
        metr = metr or Metr()
        # METR-LA's best fixed fill is `nan` (rank-0 reachable set), which no value-fill
        # class can represent by construction -- that asymmetry is the S25 P3 result, not a
        # bug. Centre on `linear`, the best REACHABLE fixed fill, and report both baselines.
        res = arm(res, bolt, metr, "miss", steps, center="linear")
        res = arm(res, bolt, metr, "miss", steps, aug_mult=6, center="linear")
        save(res)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
