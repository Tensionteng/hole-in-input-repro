#!/usr/bin/env python
"""S71 Timer-XL (timer-base-84m) CPT: 5k-step mechdiv (cpt) / filtered (ctl5k)
through the native path.

Timer has no declared missingness path (NaN -> NaN logits). CPT runs in the model's
NATIVE normalised regime: its shipped weights were pretrained on instance-normalised
windows, and the raw revin=False regime (the s57 eval convention) overflows the
backward pass on raw-scale corpus data -- measured grad-norm NaN whenever a batch
contains |x|max >~ 1e5 (see s71_notes.md). The recipe's declared half = content 0 at
missing positions after normalisation, no flag (the values-only realisation); filled
half = alpha-blend values. Loss = the model's native next-patch MSE in normalised
space over all positions via output_hidden_states + lm_heads[0], aligned with the
eval (token t predicts patch t+1; eval reads the last token's next-96). Elementwise
target masking excludes fabricated and padded target positions.

Plumbing gate: hidden-state last-token head output must equal forward().logits
(stock weights), max abs diff < 1e-4.
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
    x = rng.normal(size=(4, C.TIMER_CTX)).cumsum(1).astype(np.float32)
    t = torch.from_numpy(x).to(dev)
    model.eval()
    with torch.no_grad():
        ref = model(input_ids=t, use_cache=False).logits
        out = model(input_ids=t, use_cache=False, output_hidden_states=True)
        mine = model.lm_heads[0](out.hidden_states[-1])[:, -1, :]
    d = float((mine - ref).abs().max())
    print(f"GATE G2r timer hidden-path vs logits: max abs diff {d:.2e}", flush=True)
    assert d < 1e-4, "G2r FAILED"
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["cpt", "ctl5k"])
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=C.STEPS)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dev = args.device
    os.makedirs(C.CK, exist_ok=True)
    model = C.build_timer(dev)
    g = plumbing_gate(model, dev)
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    def fwd_loss(ctx, miss, fut):
        return C.timer_fwd_train(model, ctx, miss, fut, dev)

    log, _ = C.train_loop("timer", list(model.parameters()), fwd_loss,
                          args.arm, args.seed, args.bs, args.steps, dev,
                          smoke=args.smoke)
    log.update({"model": "timer", "arm": args.arm, "g2r": g, "smoke": args.smoke})
    tag = f"timer_{args.arm}" + (f"_s{args.seed}" if args.seed != C.SEED else "")
    if not args.smoke:
        torch.save(model.state_dict(), os.path.join(C.CK, f"{tag}.pt"))
    json.dump(log, open(os.path.join(C.CK, f"{tag}_log.json"), "w"))
    print(f"[timer:{args.arm}] done in {log['minutes']:.1f} min -> {tag}.pt",
          flush=True)


if __name__ == "__main__":
    main()
