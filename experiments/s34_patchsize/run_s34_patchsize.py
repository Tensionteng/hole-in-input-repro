#!/usr/bin/env python
"""S34: is patch size a lever on missingness robustness, and if so, why?

A verification run for S33 threw off an unplanned observation: Moirai under 70% block
missingness is damaged 1.807x at patch 32 but only 1.091x at patch 16 -- close to immune.
This round tests it properly, because Moirai is the one family where patch size is an
INFERENCE-TIME choice on identical weights (it was pretrained over
patch_sizes = [8, 16, 32, 64, 128]), which makes patch size a clean controlled variable in a
way it is not for any fixed-patch model.

The mechanism this round proposes and tests: a foundation model does not see missing POINTS,
it sees corrupted TOKENS. A contiguous gap of length G corrupts about G/P + 1 of the L/P
tokens, i.e. a fraction G/L + P/L. Smaller patches therefore strictly reduce the corrupted
fraction for contiguous gaps, while under scattered MCAR at high rates nearly every token is
corrupted regardless of P. So the benefit should be large for block/censoring geometry and
small for MCAR -- and, if the mechanism is right, degradation should collapse onto a single
curve when plotted against the corrupted-token fraction rather than the missing rate.

Pre-registered predictions:
  P1  Relative degradation falls monotonically with smaller patch under BLOCK missingness.
  P2  The effect is much weaker under MCAR at high rates (all tokens corrupted anyway).
  P3  Clean accuracy is NOT monotone in patch size, so any robustness gain is not simply
      "small patches are better at forecasting".
  P4  Plotted against corrupted-token fraction, the (patch, mechanism, rate) points collapse
      toward one curve. This is the mechanistic test; failure means the story is wrong even
      if P1 holds.

Metric: paired per-window relMSE against THAT PATCH SIZE's own clean forecast, median over
windows -- so each patch is its own control and P3's confound cannot leak into P1.
"""
import argparse
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

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
import run_s25_twofloor as s25

H = 64
L = s25.L
PATCHES = (8, 16, 32, 64, 128)
MECHS = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES = (0.1, 0.3, 0.5, 0.7)
OUT = os.path.join(HERE, "s34_results.json")


def build(patch):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    m = MoiraiForecast(
        module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base"),
        prediction_length=H, context_length=L, patch_size=patch,
        num_samples=20, target_dim=1, feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0).to("cuda")
    m.eval()
    return m


@torch.no_grad()
def fc(mod, ctx, miss, declared, batch=64):
    outs = []
    for i in range(0, len(ctx), batch):
        c = np.nan_to_num(ctx[i:i + batch]).astype(np.float32)
        mk = miss[i:i + batch]
        obs = (~mk) if declared else np.ones_like(mk, bool)
        pt = torch.from_numpy(c).to("cuda").unsqueeze(-1)
        om = torch.from_numpy(obs).to("cuda").unsqueeze(-1)
        pad = torch.zeros(pt.shape[:2], dtype=torch.bool, device="cuda")
        s = mod(past_target=pt, past_observed_target=om, past_is_pad=pad)
        outs.append(s.median(dim=1).values.squeeze(-1).float().cpu().numpy())
    return np.concatenate(outs)


def corrupted_token_frac(mask, patch):
    """Fraction of length-`patch` tokens containing at least one missing point."""
    n_tok = mask.shape[1] // patch
    m = mask[:, :n_tok * patch].reshape(len(mask), n_tok, patch)
    return m.any(axis=2).mean(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-win", type=int, default=120)
    ap.add_argument("--patches", default=",".join(str(p) for p in PATCHES))
    args = ap.parse_args()
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {"patches": list(PATCHES), "mechs": list(MECHS),
                            "rates": list(RATES), "L": L, "H": H,
                            "model": "moirai-1.1-R-base (same weights, patch is an "
                                     "inference-time choice)",
                            "input": "declared (observed-mask marks the holes)",
                            "metric": "paired relMSE vs THAT patch's own clean forecast"})
    for patch in [int(p) for p in args.patches.split(",")]:
        try:
            mod = build(patch)
        except Exception as e:
            print(f"patch={patch} BUILD FAIL {type(e).__name__}: {str(e)[:110]}", flush=True)
            continue
        print(f"\n== patch = {patch} ==", flush=True)
        for ds, path in s25.DATASETS.items():
            X, st = s25.load_windows(path, args.n_win, s25.SEED, H)
            _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
            try:
                base = ((fc(mod, cl, np.zeros_like(cl, bool), False) - gt_c) ** 2).mean(1)
            except Exception as e:
                print(f"  {ds}: FORWARD FAIL {type(e).__name__}: {str(e)[:90]}", flush=True)
                break
            ok = base > 1e-12
            res.setdefault("clean", {})[f"{ds}|{patch}"] = {
                "mse_median": float(np.median(base)), "mse_mean": float(base.mean())}
            for mech in MECHS:
                row = []
                for rate in RATES:
                    clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, rate, H,
                                                                fill="linear")
                    e = ((fc(mod, lin, mask, True) - gt) ** 2).mean(1)
                    r = e[ok] / base[ok]
                    ctf = corrupted_token_frac(mask, patch)
                    res.setdefault("cells", {})[f"{ds}|{mech}|{rate}|{patch}"] = {
                        "rel_median": float(np.median(r)), "rel_mean": float(r.mean()),
                        "corrupted_token_frac": float(ctf.mean()),
                        "missing_rate": float(mask.mean()), "n": int(ok.sum())}
                    row.append((float(np.median(r)), float(ctf.mean())))
                print(f"  {ds:8s} {mech:13s} " + "  ".join(
                    f"p={rt}:{v:.3f}(tok{100*c:3.0f}%)" for rt, (v, c) in zip(RATES, row)),
                    flush=True)
            print(f"  {ds:8s} clean MSE median = {np.median(base):.4f}", flush=True)
        del mod
        torch.cuda.empty_cache()
        json.dump(res, open(OUT + ".tmp", "w"))
        os.replace(OUT + ".tmp", OUT)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
