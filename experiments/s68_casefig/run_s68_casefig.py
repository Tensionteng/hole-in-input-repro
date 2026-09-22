#!/usr/bin/env python
"""S68 casefig: single-window case studies behind the paper's silent-failure claim.

Section 4 argues that under value censoring the model's prediction interval NARROWS
while its coverage COLLAPSES, and that under scattered dropout at the same rate the
interval widens and covers. The aggregate evidence exists (Fig. 1 middle, appE
calibration); this round dumps the per-window material for a direct visual example:
one window's forecast fan against the realized truth.

Protocol (byte-identical to the paper's harness, s25_twofloor):
  model    vanilla chronos-bolt-base, frozen, fp32, single native H=64 block
  path     DECLARED: linear fill + observation mask (1 = observed)
  windows  s25.load_windows(path, 300, SEED, H) on the last 20%; window index wi
           indexes the sorted starts AND the mask seed, as in every grid run
  masks    s25.make_mask(mech, 0.7, wi, mask_seed=0, C=1, x=window) -- identical to
           the channel-0 row of the multi-channel grid masks (mcar draws row-major,
           mnar_high is deterministic given the channel's values)
  series   channel 0 of each CSV (the harness treats channels as univariate series)

Selection (mechanical, no aesthetics): per dataset, scan window indices 0..K-1
(K=6, doubled to 12 then 24 only if no index qualifies) and take the FIRST index w
where the claimed pattern holds jointly:
  (i)   censoring: coverage of the declared 80% band falls below the nominal 0.8
        (the band "misses the truth" in the sense the paper claims)
  (ii)  censoring: declared band is narrower than the clean-context band
  (iii) scattered: declared band covers >= 90% of the horizon
  (iv)  scattered: declared band is wider than the clean-context band
The same index is used for both mechanisms of a dataset (same window, two masks).
If no index in the pool satisfies all four, the narrowing condition (ii) is dropped
and the relaxation is recorded per dataset in the JSON (this happens for ETTh1,
whose channel-0 band widens under censoring at H=64 while its coverage still
collapses; the appendix caption says so). The main-text pair uses, among datasets
whose chosen window satisfies the full pattern, the one with the lowest censoring
coverage (most complete exhibition of the failure); that dataset, K_used, and every
candidate's stats are recorded in the JSON.

Output: s68_cases.json with meta, per-window stats for the whole scanned pool, and
full arrays (context, mask, filled, truth, clean+declared quantiles) per window.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False     # strict fp32, as every eval round
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
import run_s25_twofloor as s25

L, H, SEED, RATE = s25.L, s25.H, s25.SEED, 0.7
K0, KMAX = 6, 24          # disclosed pool; expand only if no qualifying index
COV_MIN = 0.90            # scattered half of the pattern: >=90% of horizon covered
CHANNEL = 0
TSLIB = os.path.join(ROOT, "legacy_nonstationary", "tslib", "dataset")
DATASETS = {
    "ETTh1": os.path.join(TSLIB, "ETT-small", "ETTh1.csv"),
    "weather": os.path.join(TSLIB, "weather", "weather.csv"),
    "electricity": os.path.join(TSLIB, "electricity", "electricity.csv"),
}
MECHS = ("mnar_high", "mcar")      # value censoring, scattered dropout
LOCAL_BOLT = os.path.join(ROOT, "models_local", "chronos-bolt-base")
OUT = os.path.join(HERE, "s68_cases.json")


class BoltQ(s25.Bolt):
    """s25.Bolt plus all-quantile access (q=0..8 -> quantiles 0.1..0.9)."""

    @torch.no_grad()
    def quantiles_np(self, ctx_np, mask_np=None, batch=64):
        outs = []
        for i in range(0, len(ctx_np), batch):
            xb = torch.from_numpy(ctx_np[i:i + batch]).to(self.device)
            mb = None
            if mask_np is not None:
                mb = torch.from_numpy(mask_np[i:i + batch]).to(self.device).float()
            out = self.model(context=xb, mask=mb)
            outs.append(out.quantile_preds[:, :, :H].float().cpu().numpy())
        return np.concatenate(outs)                    # [B, 9, H]


def make_bolt(device):
    """Hub first (HF cache under .hf_cache); fall back to the local snapshot."""
    try:
        bolt = BoltQ(device)
        src = "amazon/chronos-bolt-base (HF cache)"
    except Exception as e:                              # noqa: BLE001
        print(f"HF load failed ({e}); falling back to {LOCAL_BOLT}", flush=True)
        from chronos import BaseChronosPipeline
        bolt = BoltQ.__new__(BoltQ)
        bolt.pipe = BaseChronosPipeline.from_pretrained(
            LOCAL_BOLT, device_map=device, torch_dtype=torch.float32)
        bolt.device = device
        bolt.model = bolt.pipe.inner_model if hasattr(bolt.pipe, "inner_model") \
            else bolt.pipe.model
        bolt.model.eval()
        for p in bolt.model.parameters():
            p.requires_grad_(False)
        src = f"local {LOCAL_BOLT}"
    return bolt, src


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    dev = "cuda"
    bolt, model_src = make_bolt(dev)
    print(f"model: {model_src}", flush=True)

    stats, cases, chosen, k_used, relaxed = {}, {}, {}, {}, {}
    for ds, path in DATASETS.items():
        X, starts = s25.load_windows(path, 300, SEED, H)
        st = starts[:KMAX]
        # channel-0 univariate series, [KMAX, L] contexts and [KMAX, H] futures
        ctx = np.stack([X[s:s + L, CHANNEL] for s in st]).astype(np.float32)
        fut = np.stack([X[s + L:s + L + H, CHANNEL] for s in st]).astype(np.float32)
        q_clean = bolt.quantiles_np(ctx, None)                     # [K, 9, H]
        w_clean = (q_clean[:, 8] - q_clean[:, 0]).mean(1)
        cov_clean = ((fut >= q_clean[:, 0]) & (fut <= q_clean[:, 8])).mean(1)
        stats[ds] = {"clean": {"coverage": cov_clean.tolist(),
                               "width": w_clean.tolist()}}
        cases[ds] = {}
        for mech in MECHS:
            masks, filled = [], []
            for wi in range(len(st)):
                x1 = ctx[wi][None]                                 # [1, L]
                m = s25.make_mask(mech, RATE, wi, 0, 1, x=x1)
                masks.append(m[0])
                filled.append(s25.fill_context(x1, m, "linear")[0])
            masks = np.stack(masks)
            filled = np.stack(filled).astype(np.float32)
            obs = (~masks).astype(np.float32)
            q_decl = bolt.quantiles_np(filled, obs)                # [K, 9, H]
            lo, hi = q_decl[:, 0], q_decl[:, 8]
            cov = ((fut >= lo) & (fut <= hi)).mean(1)
            wid = (hi - lo).mean(1)
            stats[ds][mech] = {"coverage": cov.tolist(), "width": wid.tolist(),
                               "width_rel_clean": (wid / w_clean).tolist(),
                               "n_missing": masks.sum(1).tolist()}
            cases[ds][mech] = {
                str(wi): {
                    "context": np.round(ctx[wi], 5).tolist(),
                    "mask": masks[wi].astype(int).tolist(),
                    "filled": np.round(filled[wi], 5).tolist(),
                    "truth": np.round(fut[wi], 5).tolist(),
                    "q_clean": np.round(q_clean[wi], 5).tolist(),
                    "q_decl": np.round(q_decl[wi], 5).tolist(),
                    "covered_decl": ((fut[wi] >= lo[wi]) & (fut[wi] <= hi[wi]))
                                    .astype(int).tolist(),
                    "covered_clean": ((fut[wi] >= q_clean[wi, 0])
                                      & (fut[wi] <= q_clean[wi, 8])).astype(int).tolist(),
                    "start": int(st[wi]),
                } for wi in range(len(st))}
            print(f"  {ds:11s} {mech:9s} mean cov={cov.mean():.3f} "
                  f"mean w_rel={np.mean(wid / w_clean):.3f} "
                  f"(clean cov={cov_clean.mean():.3f})", flush=True)

        # selection: first index in 0..K-1 where the joint pattern holds; if none,
        # drop the narrowing condition (disclosed), then the scattered conditions
        conds = [
            ("full", lambda w: (stats[ds]["mnar_high"]["coverage"][w] < 0.8
                                and stats[ds]["mnar_high"]["width_rel_clean"][w] < 1.0
                                and stats[ds]["mcar"]["coverage"][w] >= COV_MIN
                                and stats[ds]["mcar"]["width_rel_clean"][w] > 1.0)),
            ("no_narrow", lambda w: (stats[ds]["mnar_high"]["coverage"][w] < 0.8
                                     and stats[ds]["mcar"]["coverage"][w] >= COV_MIN
                                     and stats[ds]["mcar"]["width_rel_clean"][w] > 1.0)),
            ("miss_only", lambda w: stats[ds]["mnar_high"]["coverage"][w] < 0.8),
        ]
        pool, rule_used = None, None
        for rname, cond in conds:
            for K in (K0, 12, KMAX):
                hit = next((w for w in range(K) if cond(w)), None)
                if hit is not None:
                    pool, rule_used = (hit, K), rname
                    break
            if pool:
                break
        assert pool is not None, f"{ds}: no censoring miss in {KMAX} windows"
        if rule_used != "full":
            print(f"  WARNING {ds}: full pattern absent in pool; "
                  f"selection rule relaxed to {rule_used!r} (disclosed)", flush=True)
        chosen[ds], k_used[ds] = pool
        relaxed[ds] = rule_used
        w = pool[0]
        print(f"  CHOSEN {ds:11s} wi={w} (K={pool[1]})  "
              f"censoring cov={stats[ds]['mnar_high']['coverage'][w]:.3f} "
              f"w_rel={stats[ds]['mnar_high']['width_rel_clean'][w]:.3f} | "
              f"scattered cov={stats[ds]['mcar']['coverage'][w]:.3f} "
              f"w_rel={stats[ds]['mcar']['width_rel_clean'][w]:.3f}", flush=True)

    # main-text pair: among datasets whose chosen window shows the FULL pattern,
    # take the one whose censoring coverage is lowest (most complete exhibition)
    full_ds = [d for d in DATASETS if relaxed[d] == "full"]
    assert full_ds, "no dataset exhibits the full pattern in the scanned pool"
    main_ds = min(full_ds, key=lambda d: stats[d]["mnar_high"]["coverage"][chosen[d]])
    print(f"MAIN dataset: {main_ds} wi={chosen[main_ds]} "
          f"(full-pattern datasets: {full_ds})", flush=True)

    res = {
        "meta": {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_src,
            "protocol": {"L": L, "H": H, "rate": RATE, "seed": SEED,
                         "fill": "linear", "path": "declared (fill + obs mask)",
                         "channel": CHANNEL, "n_windows_loaded": 300,
                         "quantiles": "q0..q8 = 0.1..0.9; band = [q0, q8]"},
            "selection": {
                "rule": ("per dataset, the first window index in 0..K-1 (K=6, "
                         "doubled to 12 then 24 if no index qualifies) where: "
                         "(i) under value censoring the declared 80% band's "
                         "coverage falls below the nominal 0.8, (ii) the declared "
                         "band is narrower than the clean-context band, (iii) "
                         "under scattered dropout the declared band covers >=0.90 "
                         "of the horizon, (iv) the scattered band is wider than "
                         "the clean band. The same index is used for both "
                         "mechanisms of a dataset. If no index qualifies, (ii) "
                         "is dropped ('no_narrow'), then (iii)+(iv) "
                         "('miss_only'); relaxations are recorded per dataset."),
                "main_rule": ("main-text pair = the full-pattern dataset whose "
                              "chosen window has the lowest censoring coverage"),
                "K0": K0, "cov_min": COV_MIN, "k_used": k_used, "chosen": chosen,
                "rule_used": relaxed, "main_dataset": main_ds},
            "datasets": DATASETS,
        },
        "stats": stats,
        "cases": cases,
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh)
    os.replace(tmp, OUT)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
