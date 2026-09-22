#!/usr/bin/env python
"""S58: WiSE-FT weight interpolation between stock (P0) and the CPT retrofit (P5).

theta(alpha) = (1-alpha) * P0 + alpha * P5, alpha in a grid. The fine-tuning
literature's standard move for recovering pretraining accuracy while keeping the
new capability. Zero training: interpolate state dicts, dump checkpoints, and let
eval_s45.py score them like any other arm (W* arms are dual-interface).

Success bar (post-registered here, before any eval): an alpha with clean ratio
<= 1.02 vs P0 AND closure within 15 points of P5's. If the frontier trades off
cleanly, report the frontier, not a point.
"""
import argparse
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "s45_ckpt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="tiny", choices=["tiny", "base"])
    ap.add_argument("--alphas", default="0.3,0.5,0.7,0.85")
    args = ap.parse_args()

    p0 = torch.load(os.path.join(CK, f"arm_P0_{args.size}.pt"), weights_only=True,
                    map_location="cpu")
    p5 = torch.load(os.path.join(CK, f"arm_P5_{args.size}.pt"), weights_only=True,
                    map_location="cpu")
    assert set(p0) == set(p5), "state dict keys differ"
    for a in [float(x) for x in args.alphas.split(",")]:
        tag = f"arm_W{int(round(a * 100)):02d}_{args.size}.pt"
        out = {k: (1 - a) * p0[k] + a * p5[k] for k in p0}
        torch.save(out, os.path.join(CK, tag))
        print(f"alpha={a:.2f} -> {tag}", flush=True)


if __name__ == "__main__":
    main()
