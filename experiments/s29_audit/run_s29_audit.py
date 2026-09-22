#!/usr/bin/env python
"""S29: three audits that need no new data, each a standalone claim.

Arm A -- IS THE MASK ITSELF A FORECAST SIGNAL?
  In real systems missingness is often *caused by* the thing being forecast (curtailment
  happens because the wind is high; loop detectors drop out during congestion/weather). If
  so the binary mask is a leading indicator, and "missingness is noise to be repaired" is
  the wrong premise. We forecast from the mask ALONE (plus a level anchor, since a binary
  pattern carries no units) and ask whether it is competitive with a TSFM that sees every
  observed value, and whether mask features improve the TSFM's forecast on top.

Arm B -- THE IMPUTER AS AN ATTACK SURFACE
  S25 Part A showed the forecast is extraordinarily steerable through the missing positions
  (an optimised fill reaches 0.002-0.096x the clean-context error). The mirror image is a
  security claim: whoever controls the imputation pipeline -- often a third-party library or
  an upstream data vendor -- controls the forecast. We measure untargeted damage and
  targeted steering under a RANGE-CONSTRAINED threat model (fills must lie inside the
  observed min/max, so the poisoned context passes a range check), and report detectability.

Arm C -- IS REPORTED ERROR MEASURED ON A BIASED SLICE?
  Every window list in this project -- and in S6/S10/S14, and in the wider literature --
  requires the forecast TARGET to be fully observed. That is a selection on the outcome. If
  windows whose target is partly missing are harder, then every reported number in the field
  is optimistic, and nobody has quantified by how much. We rebuild the window sets without
  the complete-target filter and score on the observed part of the target only.

Windows/metrics reuse S6/S10/S14 (CTX=144,H=24 Penmanshiel; CTX=512 METR-LA), with anchor
gates against the stored numbers before anything runs.
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
sys.path.insert(0, os.path.join(EXP, "s26_realfill"))

import run_s6_real_censor as s6r
import run_s14_metrla as s14
import run_s25_twofloor as s25
import run_s26_realfill as s26
sys.path.insert(0, os.path.join(EXP, 's27_interface'))
import run_s27_interface as s27

s6r.NPZ = os.path.join(DATA, "dataset_censor", "penmanshiel2016_processed.npz")
s14.H5 = os.path.join(DATA, "dataset_metrla", "metr-la.h5")
s26.s6r.NPZ, s26.s14.H5 = s6r.NPZ, s14.H5

OUT = os.path.join(HERE, "s29_results.json")
CK = os.path.join(HERE, "s29_ckpt")
SEED = 20250810


def save(res):
    tmp = OUT + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, OUT)


# ------------------------------------------------------------------ gate ----

def gate(res, bolt):
    """Reuse S26's eight stored cells (S10 + S14)."""
    out = {}
    penn, metr = s26.Penn(), s26.Metr()
    for name, ds, anchors, hz, tgt in (("penn", penn, s26.ANCHOR_PENN, penn.H, penn.tgt),
                                       ("metr", metr, s26.ANCHOR_METR, 96, metr.tgt_full)):
        for key, stored in anchors.items():
            grp, fill = key.split("|")
            idx = np.flatnonzero(ds.group == grp)
            f = "keep" if fill == "clean" else fill
            ctx = s26.fill_ctx(ds.ctx_obs[idx], ds.ctx_rec[idx], ds.mask[idx], f)
            pred = bolt.median_rollout_np(ctx, hz)
            v = float(s26.nmse_per_window(pred, tgt[idx], ds.VAR_FLOOR).mean())
            d = abs(v - stored) / stored
            out[f"{name}|{key}"] = {"stored": stored, "rerun": v, "rel_dev": d,
                                    "pass": d <= 0.05}
            print(f"GATE {name} {key:14s} stored={stored:9.5f} rerun={v:9.5f} "
                  f"dev={d:+.4%} {'PASS' if d <= 0.05 else 'FAIL'}", flush=True)
    res["gate"] = out
    assert all(v["pass"] for v in out.values()), "gate FAILED"
    return res, penn, metr


# --------------------------------------------------------- Arm A: mask-only ----

class MaskNet(nn.Module):
    """Forecast from the BINARY MASK ONLY, plus a two-number level anchor (mu, sd of the
    observed points) so the output has units. It never sees an observed value."""

    def __init__(self, ctx, horizon, ch=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, ch, 5, padding=2), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=2, dilation=2), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=4, dilation=4), nn.GELU(),
            nn.Conv1d(ch, ch, 3, padding=8, dilation=8), nn.GELU())
        self.head = nn.Sequential(nn.Linear(ch * 3, 128), nn.GELU(), nn.Linear(128, horizon))

    def forward(self, m):                       # m: [B, CTX] 1 = missing
        h = self.body(m.unsqueeze(1))
        f = torch.cat([h.mean(-1), h.max(-1).values, h[..., -1]], -1)
        return 6.0 * torch.tanh(self.head(f) / 6.0)      # z-units, bounded


def zstats(ctx_obs):
    mu = np.empty((len(ctx_obs), 1), np.float32)
    sd = np.empty((len(ctx_obs), 1), np.float32)
    for i, r in enumerate(ctx_obs):
        v = r[np.isfinite(r)]
        mu[i, 0] = v.mean() if len(v) else 0.0
        sd[i, 0] = (v.std() if len(v) > 1 else 0.0) or 1.0
    return mu, np.maximum(sd, 1e-3)


def train_masknet(ds, idx, steps=1200, bs=64, lr=1e-3, seed=5):
    dev = "cuda"
    net = MaskNet(ds.ctx_obs.shape[1], ds.H).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    mu, sd = zstats(ds.ctx_obs[idx])
    M = ds.mask[idx].astype(np.float32)
    Y = (ds.tgt[idx] - mu) / sd
    hist = []
    for it in range(steps):
        b = rng.choice(len(M), size=min(bs, len(M)), replace=False)
        m = torch.from_numpy(M[b]).to(dev)
        y = torch.from_numpy(Y[b]).to(dev)
        loss = torch.nn.functional.huber_loss(net(m), y, delta=3.0)
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        sch.step()
        hist.append(float(loss))
    return net, hist


@torch.no_grad()
def eval_masknet(net, ds, idx):
    mu, sd = zstats(ds.ctx_obs[idx])
    m = torch.from_numpy(ds.mask[idx].astype(np.float32)).to("cuda")
    pred = net(m).float().cpu().numpy() * sd + mu
    return s26.nmse_per_window(pred, ds.tgt[idx], ds.VAR_FLOOR)


def armA(res, bolt, ds, kind, steps=1200):
    cal, test = ds.split(kind)
    print(f"\n== Arm A mask-only  {ds.name}/{kind}: {len(cal)} train / {len(test)} test",
          flush=True)
    cell = {"n_cal": int(len(cal)), "n_test": int(len(test))}
    # value-based reference: bolt with the best fixed fill
    best = None
    for f in ("keep", "zero", "linear", "nan"):
        ctx = s26.fill_ctx(ds.ctx_obs[test], ds.ctx_rec[test], ds.mask[test], f)
        e = s26.nmse_per_window(bolt.median_np(ctx)[:, :ds.H], ds.tgt[test], ds.VAR_FLOOR)
        cell[f"bolt_{f}"] = {"mean": float(e.mean()), "median": float(np.median(e))}
        if best is None or np.median(e) < np.median(best[1]):
            best = (f, e)
    # trivial baselines
    mu, sd = zstats(ds.ctx_obs[test])
    flat = s26.nmse_per_window(np.repeat(mu, ds.H, axis=1), ds.tgt[test], ds.VAR_FLOOR)
    cell["baseline_context_mean"] = {"mean": float(flat.mean()),
                                     "median": float(np.median(flat))}
    net, hist = train_masknet(ds, cal, steps=steps)
    e = eval_masknet(net, ds, test)
    cell["mask_only"] = {"mean": float(e.mean()), "median": float(np.median(e)),
                         "per_window": e.tolist(), "final_loss": float(np.mean(hist[-50:]))}
    cell["best_fixed"] = best[0]
    cell["bolt_best_per_window"] = best[1].tolist()
    print(f"  bolt({best[0]:6s}) NMSE mean={best[1].mean():9.4f} med={np.median(best[1]):7.4f}\n"
          f"  context-mean  NMSE mean={flat.mean():9.4f} med={np.median(flat):7.4f}\n"
          f"  MASK-ONLY     NMSE mean={e.mean():9.4f} med={np.median(e):7.4f}  "
          f"(beats bolt on {100*(e < best[1]).mean():.0f}% of windows)", flush=True)
    cell["mask_only_winrate_vs_bolt"] = float((e < best[1]).mean())
    res.setdefault("armA", {})[f"{ds.name}|{kind}"] = cell
    return res


# ------------------------------------------------------------ Arm B: attack ----

def attack(bolt, ds, idx, iface, mode, base_fill, steps=400, lr=0.05, target_scale=0.5):
    """Range-constrained poisoning of the FILL VALUES ONLY, per input interface.

    Threat model: the attacker controls the imputation step -- a third-party library, an
    upstream data vendor, or simply whatever `fillna` the pipeline calls -- but NOT the
    observed values. Poisoned fills must stay inside the window's observed [min, max], so
    the context still passes a range/plausibility check.

    Running this per interface turns S25's rank result into a security statement: the
    dimension of the attack surface IS rank(J_M). Under the NaN interface the fill is not an
    input at all, so the attack is structurally impossible, not merely hard.
    """
    dev = bolt.device
    base = s26.fill_ctx(ds.ctx_obs[idx], ds.ctx_rec[idx], ds.mask[idx], base_fill)
    miss = ds.mask[idx]
    obs_f = (~miss).astype(np.float32)
    y = ds.tgt[idx]
    lo = np.array([np.nanmin(r) for r in ds.ctx_obs[idx]], np.float32)[:, None]
    hi = np.array([np.nanmax(r) for r in ds.ctx_obs[idx]], np.float32)[:, None]

    if iface == "nan":                       # rank 0: no lever exists
        p = bolt.median_np(np.where(miss, np.nan, base))[:, :ds.H]
        return base, base.copy(), p, p.copy()

    x0 = torch.from_numpy(base).to(dev)
    mt = torch.from_numpy(miss.astype(np.float32)).to(dev)
    ot = torch.from_numpy(obs_f).to(dev)
    yt = torch.from_numpy(y).to(dev)
    lot, hit = torch.from_numpy(lo).to(dev), torch.from_numpy(hi).to(dev)
    fwd = lambda c: s27.median_iface(bolt, c, ot, "plain" if iface == "plain" else "native")[:, :ds.H]
    with torch.no_grad():
        p_clean = fwd(x0).cpu().numpy()
    tgt_mean = torch.from_numpy(p_clean.mean(1, keepdims=True) * target_scale).to(dev)
    v = torch.zeros_like(x0, requires_grad=True)
    opt = torch.optim.Adam([v], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        cand = torch.clamp(x0 + mt * v, lot, hit)
        p = fwd(cand)
        loss = -((p - yt) ** 2).mean(1) if mode == "untargeted" else \
            ((p.mean(1, keepdim=True) - tgt_mean) ** 2).squeeze(1)
        loss.sum().backward()
        opt.step()
    with torch.no_grad():
        cand = torch.clamp(x0 + mt * v, lot, hit)
        p_att = fwd(cand).cpu().numpy()
        cand_np = cand.cpu().numpy()
    return base, cand_np, p_clean, p_att


def armB(res, bolt, ds, kind, steps=400):
    _, test = ds.split(kind)
    # the defender's own best fixed fill is the baseline the attacker has to beat
    best, best_e = None, None
    for f in ("keep", "zero", "linear", "nan"):
        ctx = s26.fill_ctx(ds.ctx_obs[test], ds.ctx_rec[test], ds.mask[test], f)
        e = s26.nmse_per_window(bolt.median_np(ctx)[:, :ds.H], ds.tgt[test], ds.VAR_FLOOR)
        if best is None or np.median(e) < np.median(best_e):
            best, best_e = f, e
    base_fill = "linear" if best == "nan" else best
    print(f"\n== Arm B attack surface  {ds.name}/{kind}: {len(test)} windows, "
          f"defender's best fill = {best} (attack starts from {base_fill})", flush=True)
    cell = {"n": int(len(test)), "defender_best_fill": best,
            "missing_rate_mean": float(ds.mask[test].mean())}
    for iface in ("plain", "native", "nan"):
        for mode in ("untargeted", "targeted"):
            t0 = time.time()
            base, poisoned, p_clean, p_att = attack(bolt, ds, test, iface, mode,
                                                    base_fill, steps=steps)
            e_c = s26.nmse_per_window(p_clean, ds.tgt[test], ds.VAR_FLOOR)
            e_a = s26.nmse_per_window(p_att, ds.tgt[test], ds.VAR_FLOOR)
            ratio = e_a / np.maximum(e_c, 1e-12)
            shift = np.abs(p_att.mean(1) - p_clean.mean(1)) / (np.abs(p_clean.mean(1)) + 1e-6)
            cell[f"{iface}|{mode}"] = {
                "nmse_clean_median": float(np.median(e_c)),
                "nmse_attacked_median": float(np.median(e_a)),
                "damage_ratio_median": float(np.median(ratio)),
                "damage_ratio_q90": float(np.quantile(ratio, 0.9)),
                "damage_ratio_max": float(ratio.max()),
                "forecast_mean_shift_median": float(np.median(shift)),
                "frac_windows_doubled": float((ratio > 2).mean()),
            }
            c = cell[f"{iface}|{mode}"]
            print(f"  {iface:7s} {mode:11s} NMSE med {np.median(e_c):8.4f} -> "
                  f"{np.median(e_a):8.4f}  damage x{c['damage_ratio_median']:.2f} "
                  f"(q90 x{c['damage_ratio_q90']:.2f}, max x{c['damage_ratio_max']:.0f})  "
                  f"forecast moved {100*c['forecast_mean_shift_median']:.0f}%  "
                  f">2x on {100*c['frac_windows_doubled']:.0f}% ({time.time()-t0:.0f}s)",
                  flush=True)
    res.setdefault("armB", {})[f"{ds.name}|{kind}"] = cell
    return res


# ------------------------------------------- Arm C: complete-target selection ----

def penn_windows_relaxed():
    """S6's Penmanshiel windows WITHOUT the fully-observed-target requirement."""
    d = np.load(s6r.NPZ)
    tags = sorted({k[:-len("_power")] for k in d.files if k.endswith("_power")})
    CTX, H = s6r.CTX, s6r.H
    rows = []
    for tag in tags:
        p, cens, curt = d[f"{tag}_power"], d[f"{tag}_cens"], d[f"{tag}_curt"]
        n = len(p)
        for t in range(CTX, n - H, s6r.STRIDE):
            ctx_p, ctx_cens = p[t - CTX:t], cens[t - CTX:t]
            tgt_p, tgt_curt = p[t:t + H], curt[t:t + H]
            if not (np.isfinite(ctx_p) | ctx_cens).all():
                continue
            n_cens = int(ctx_cens.sum())
            if n_cens < s6r.MIN_CENS:
                continue
            tgt_ok = np.isfinite(tgt_p) & ~tgt_curt
            if tgt_ok.sum() < 4:                      # need something to score on
                continue
            rows.append(dict(turbine=tag, start=int(t),
                             complete=bool(tgt_ok.all()), obs_frac=float(tgt_ok.mean())))
    return rows, d


def armC(res, bolt):
    """Compare error on complete-target windows vs windows whose target is partly missing,
    scoring both only on the OBSERVED target points."""
    rows, d = penn_windows_relaxed()
    CTX, H = s6r.CTX, s6r.H
    ctx = np.full((len(rows), CTX), np.nan, np.float32)
    tgt = np.full((len(rows), H), np.nan, np.float32)
    ok = np.zeros((len(rows), H), bool)
    for i, w in enumerate(rows):
        tag, t = w["turbine"], w["start"]
        p, cens, curt = d[f"{tag}_power"], d[f"{tag}_cens"], d[f"{tag}_curt"]
        c = p[t - CTX:t].copy()
        c[cens[t - CTX:t]] = np.nan
        ctx[i] = c
        tgt[i] = p[t:t + H]
        ok[i] = np.isfinite(p[t:t + H]) & ~curt[t:t + H]
    filled = s26.fill_ctx(ctx, ctx, ~np.isfinite(ctx), "zero")   # S10's winning fill
    pred = bolt.median_np(filled)[:, :H]
    comp = np.array([w["complete"] for w in rows])
    with np.errstate(all="ignore"):
        se = (pred - tgt) ** 2
        num = np.where(ok, se, np.nan)
        mse_w = np.nanmean(num, axis=1)
        var_w = np.array([np.nanvar(np.where(ok[i], tgt[i], np.nan)) for i in range(len(rows))])
    nmse = mse_w / np.maximum(var_w, s6r.VAR_FLOOR)
    good = np.isfinite(nmse)
    cell = {
        "n_total": int(len(rows)), "n_complete_target": int((comp & good).sum()),
        "n_partial_target": int((~comp & good).sum()),
        "frac_windows_excluded_by_the_filter": float((~comp).mean()),
        "nmse_complete_mean": float(nmse[comp & good].mean()),
        "nmse_partial_mean": float(nmse[~comp & good].mean()),
        "nmse_complete_median": float(np.median(nmse[comp & good])),
        "nmse_partial_median": float(np.median(nmse[~comp & good])),
        "nmse_all_median": float(np.median(nmse[good])),
    }
    cell["optimism_median_pct"] = 100 * (cell["nmse_all_median"] / cell["nmse_complete_median"] - 1)
    print(f"\n== Arm C evaluation bias (Penmanshiel, censored windows, zero fill) ==\n"
          f"  windows: {cell['n_complete_target']} complete-target (what everyone scores) + "
          f"{cell['n_partial_target']} partial-target (everyone drops these) "
          f"= {100*cell['frac_windows_excluded_by_the_filter']:.0f}% excluded\n"
          f"  NMSE median: complete-target {cell['nmse_complete_median']:.4f}  "
          f"partial-target {cell['nmse_partial_median']:.4f}  all {cell['nmse_all_median']:.4f}\n"
          f"  --> reported error is optimistic by {cell['optimism_median_pct']:+.1f}% (median)",
          flush=True)
    res["armC"] = cell
    return res


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="gate,A,B,C")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--attack-steps", type=int, default=300)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(CK, exist_ok=True)

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"model": "chronos-bolt-base (frozen)",
                            "penn": "CTX=144 H=24, S6/S10 windows, NMSE floor 100 kW^2",
                            "metr": "CTX=512 H=64, S14 windows, NMSE floor 4.0 mph^2",
                            "split": "per-turbine/-sensor 50/50, seed 20250812"})
    bolt = s25.Bolt("cuda")
    steps = 100 if args.smoke else args.steps
    asteps = 30 if args.smoke else args.attack_steps

    parts = args.parts.split(",")
    penn = metr = None
    if "gate" in parts:
        res, penn, metr = gate(res, bolt)
        save(res)
    penn = penn or s26.Penn()
    metr = metr or s26.Metr()
    if "A" in parts:
        res = armA(res, bolt, penn, "cens", steps)
        res = armA(res, bolt, metr, "miss", steps)
        save(res)
    if "B" in parts:
        res = armB(res, bolt, penn, "cens", asteps)
        res = armB(res, bolt, metr, "miss", asteps)
        save(res)
    if "C" in parts:
        res = armC(res, bolt)
        save(res)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
