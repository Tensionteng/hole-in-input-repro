#!/usr/bin/env python
"""S60 GATE: the interface replica on NATIVE must match the stock Chronos-2 forward
bit-exactly (max|Delta| = 0, fp32, eval mode, TF32 off) on clean AND holed contexts,
in all three stock conventions:

  clean              context, no mask
  holed/explicit     filled context + explicit context_mask   (content zeroed by model)
  holed/NaN          NaN at missing, mask=None                (the declared path)

Also verified: the training loss replica (model._compute_loss through forward_iface)
matches the stock forward's loss bit-exactly, and the DUAL path differs from native
on holed input (the content actually reaches the forecast there -- sanity, not a gate).

If max|Delta| != 0 the script reports it and exits nonzero -- do NOT train on a
divergent replica.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False     # strict fp32 for the gate
torch.backends.cudnn.allow_tf32 = False

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, HERE)
import run_s25_twofloor as s25
import c2_iface

L, H = s25.L, 64
OUT = os.path.join(HERE, "s60_gate.json")


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    m = c2_iface.load_stock(dev)
    nop = H // m.chronos_config.output_patch_size

    X, st = s25.load_windows(s25.DATASETS["ETTh1"], 8, s25.SEED, H)
    _, cl, _, fut = s25.build_eval_batch(X, st, "clean", 0.0, H)
    cl_t = torch.from_numpy(cl).to(dev)
    fut_t = torch.from_numpy(fut).to(dev)

    res = {"meta": {"model": "models_local/chronos-2", "dtype": "float32",
                    "tf32": False, "H": H, "n_win": len(cl),
                    "requirement": "max|diff| == 0 on every native cell"}}
    worst = 0.0

    def cmp(tag, ctx_np, obs_np):
        """stock forward vs native replica; returns max|dQ|, |dloss|."""
        ctx = torch.from_numpy(ctx_np).to(dev)
        obs = None if obs_np is None else torch.from_numpy(obs_np).to(dev)
        with torch.no_grad():
            ref = m(context=ctx, context_mask=obs, num_output_patches=nop,
                    future_target=fut_t)
            q, loss = c2_iface.forward_iface(m, ctx, obs, "native", nop,
                                             future_target=fut_t)
        dq = float((q - ref.quantile_preds).abs().max())
        dl = float(abs(loss - ref.loss))
        res[tag] = {"max_abs_diff_quantiles": dq, "abs_diff_loss": dl,
                    "pass": dq == 0.0 and dl == 0.0}
        print(f"GATE {tag:28s} max|dQ|={dq:.3e} |dloss|={dl:.3e} "
              f"{'PASS' if res[tag]['pass'] else 'FAIL'}", flush=True)
        return max(dq, dl)

    # clean, no mask
    worst = max(worst, cmp("clean", cl, None))
    # clean, explicit all-ones mask
    worst = max(worst, cmp("clean_explicit_mask", cl, np.ones_like(cl)))

    # holed contexts, all four mechanisms x {0.3, 0.7}: explicit-mask and NaN paths
    for mech in ("mcar", "block", "mnar_high", "mnar_extreme"):
        for rate in (0.3, 0.7):
            _, lin, mask, _ = s25.build_eval_batch(X, st, mech, rate, H, fill="linear")
            obs = (~mask).astype(np.float32)
            worst = max(worst, cmp(f"{mech}|{rate}|explicit", lin, obs))
            nan_ctx = lin.copy()
            nan_ctx[mask] = np.nan
            worst = max(worst, cmp(f"{mech}|{rate}|nan", nan_ctx, None))

    # sanity (not a gate): dual keeps the content, so it must MOVE the forecast
    _, lin, mask, _ = s25.build_eval_batch(X, st, "mcar", 0.7, H, fill="linear")
    obs = (~mask).astype(np.float32)
    with torch.no_grad():
        q_nat, _ = c2_iface.forward_iface(
            m, torch.from_numpy(lin).to(dev), torch.from_numpy(obs).to(dev),
            "native", nop)
        q_dual, _ = c2_iface.forward_iface(
            m, torch.from_numpy(lin).to(dev), torch.from_numpy(obs).to(dev),
            "dual", nop)
    d = float((q_nat - q_dual).abs().max())
    res["sanity_dual_differs_from_native"] = {"max_abs_diff": d, "pass": d > 0.0}
    print(f"SANITY dual vs native (mcar 0.7): max|dQ|={d:.3e} "
          f"{'OK' if d > 0 else 'UNEXPECTED -- dual path may be dead'}", flush=True)

    res["worst_abs_diff"] = worst
    res["pass"] = worst == 0.0 and res["sanity_dual_differs_from_native"]["pass"]
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"GATE overall: worst max|diff|={worst:.3e} -> "
          f"{'PASS' if res['pass'] else 'FAIL'}; wrote {OUT}", flush=True)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
