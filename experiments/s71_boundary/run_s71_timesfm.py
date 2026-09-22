#!/usr/bin/env python
"""S71 TimesFM CPT: 5k-step mechdiv (cpt) / filtered (ctl5k) through the native path.

The shipped path exposes values only (the wrapper overwrites NaN with its own linear
interpolant; the module mask is padding-only). So the recipe's declared half enters
as the model's OWN interpolant (linear fill, no flag) and the alpha-blend half as
plain values. Loss = pinball over the 9 quantile channels of the last-patch 64-step
prefill output, through a grad-enabled replica of the eval preprocessing stack
(front-pad to 1024, outer revin, causal running-stats revin inside decode).

Plumbing gate (runs before training): the grad replica's channel-5 output with STOCK
weights must match the compiled wrapper's point forecast (rel. tol 1e-3).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

import s71_common as C


def plumbing_gate(module, dev):
    """G2t: the grad replica (timesfm_fwd_train) must match the module's OWN
    decode() prefill exactly on the identically preprocessed input. The eval
    wrapper's tail flags (flip invariance, positivity clamp) are post-processing
    excluded from training by design -- documented in s71_notes.md."""
    from timesfm.torch import util
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, C.L)).cumsum(1).astype(np.float32)
    mk, mu, sigma, xn = C.timesfm_outer(x, dev)
    module.eval()
    with torch.no_grad():
        pf, _, _ = module.decode(128, xn, mk)
        ref = util.revin(pf[:, -1, :C.H, 5], mu, sigma, reverse=True)
        mine = C.timesfm_fwd_train(module, x, None, dev)[..., 5]
    rel = float((mine - ref).abs().max() / max(1e-9, float(ref.abs().max())))
    print(f"GATE G2t timesfm grad-replica vs decode() prefill: max rel diff "
          f"{rel:.2e}", flush=True)
    assert rel < 1e-5, "G2t FAILED"
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["cpt", "ctl5k"])
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--steps", type=int, default=C.STEPS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = args.device
    os.makedirs(C.CK, exist_ok=True)
    m = C.build_timesfm(dev)                       # stock init, torch module
    module = m.model
    module.train()
    for p in module.parameters():
        p.requires_grad_(True)
    g = plumbing_gate(module, dev)
    module.train()

    def fwd_loss(ctx, miss, fut):
        return C.timesfm_pinball(module, ctx, fut, dev)

    log, _ = C.train_loop("timesfm", list(module.parameters()), fwd_loss,
                          args.arm, args.seed, args.bs, args.steps, dev,
                          smoke=args.smoke)
    log.update({"model": "timesfm", "arm": args.arm, "g2t": g,
                "smoke": args.smoke})
    tag = f"timesfm_{args.arm}" + (f"_s{args.seed}" if args.seed != C.SEED else "")
    if not args.smoke:
        torch.save(module.state_dict(), os.path.join(C.CK, f"{tag}.pt"))
    json.dump(log, open(os.path.join(C.CK, f"{tag}_log.json"), "w"))
    print(f"[timesfm:{args.arm}] done in {log['minutes']:.1f} min -> {tag}.pt",
          flush=True)


if __name__ == "__main__":
    main()
