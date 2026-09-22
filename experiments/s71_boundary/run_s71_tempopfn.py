#!/usr/bin/env python
"""S71 TempoPFN CPT: 5k-step mechdiv (cpt) / filtered (ctl5k) through the native path.

Declared half = NaN at missing positions (the learned-token path, exactly as
evaluated); filled half = alpha-blend plain values. Loss = the model's own
compute_loss (native quantile loss in scaled space) on the s62 container
conventions. bf16 autocast (the only supported mode), fp32 weights + AdamW.

Plumbing gate: training container forward with stock weights must match the eval
forward's predictions (predictions are future_values-independent), rel. tol 1e-3.
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


def plumbing_gate(model, dev):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, C.L)).cumsum(1).astype(np.float32)
    f = rng.normal(size=(4, C.H)).astype(np.float32)
    ev = C.TempoPFNEval.__new__(C.TempoPFNEval)      # reuse fc_quant without reload
    ev.m, ev.dev, ev.bf16 = model, dev, True
    ev.qidx_med = model.quantiles.index(0.5)
    model.eval()
    ref = ev.fc_quant(x, np.zeros_like(x, bool), "plain")
    cont = C.tempopfn_container(x, f, dev)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(cont)
    mine = model.scaler.inverse_scale(out["result"].float(),
                                      out["scale_statistics"])[:, :, 0, :]
    mine = mine.cpu().numpy()
    rel = float(np.abs(mine - ref).max() / max(1e-9, float(np.abs(ref).max())))
    print(f"GATE G2p tempopfn train-container vs eval forward: max rel diff "
          f"{rel:.2e}", flush=True)
    assert rel < 1e-3, "G2p FAILED"
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["cpt", "ctl5k"])
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=C.STEPS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = args.device
    os.makedirs(C.CK, exist_ok=True)
    model = C.build_tempopfn(dev)
    g = plumbing_gate(model, dev)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    def fwd_loss(ctx, miss, fut):
        cont = C.tempopfn_container(ctx, fut, dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(cont)
            return model.compute_loss(cont.future_values, out)

    log, _ = C.train_loop("tempopfn", list(model.parameters()), fwd_loss,
                          args.arm, args.seed, args.bs, args.steps, dev,
                          smoke=args.smoke)
    log.update({"model": "tempopfn", "arm": args.arm, "g2p": g,
                "smoke": args.smoke})
    tag = f"tempopfn_{args.arm}" + (f"_s{args.seed}" if args.seed != C.SEED else "")
    if not args.smoke:
        torch.save(model.state_dict(), os.path.join(C.CK, f"{tag}.pt"))
    json.dump(log, open(os.path.join(C.CK, f"{tag}_log.json"), "w"))
    print(f"[tempopfn:{args.arm}] done in {log['minutes']:.1f} min -> {tag}.pt",
          flush=True)


if __name__ == "__main__":
    main()
