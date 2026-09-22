#!/usr/bin/env python
"""S60 WiSE-FT: interpolate the CPT checkpoint 0.5/0.5 with stock Chronos-2.

theta = 0.5 * stock + 0.5 * CPT -- the same procedure as make_wiseft.py (S58)
for Chronos-Bolt. The result is the "c2-retrofit" (arm_W50.pt).
"""
import argparse
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CK = os.path.join(HERE, "s60_ckpt")
OUT = os.path.join(HERE, "s60_wiseft.json")


def main():
    # --suffix added for seed replications (2026-09-01): default '' reproduces
    # the original seed-20260901 behavior exactly.
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="",
                    help="seed-replication suffix (e.g. _s20260902): read "
                         "arm_CPT{suffix}.pt, write arm_W50{suffix}.pt and "
                         "s60_wiseft{suffix}.json; default '' = seed-20260901")
    args = ap.parse_args()
    sfx = args.suffix
    from chronos.chronos2.model import Chronos2Model
    m = Chronos2Model.from_pretrained(os.path.join(ROOT, "models_local", "chronos-2"),
                                      torch_dtype=torch.float32)
    p0 = m.state_dict()
    p5 = torch.load(os.path.join(CK, f"arm_CPT{sfx}.pt"), weights_only=True,
                    map_location="cpu")
    assert set(p0) == set(p5), "state dict keys differ"
    a = 0.5
    out = {k: (1 - a) * p0[k].float() + a * p5[k].float() for k in p0}
    torch.save(out, os.path.join(CK, f"arm_W50{sfx}.pt"))
    res = {"alpha": a, "recipe": "theta = (1-a)*stock + a*CPT",
           "n_tensors": len(out),
           "max_abs_weight_delta_vs_stock": float(
               max((out[k] - p0[k].float()).abs().max() for k in p0)),
           "out": f"s60_ckpt/arm_W50{sfx}.pt"}
    out_json = OUT.replace(".json", f"{sfx}.json")
    json.dump(res, open(out_json, "w"), indent=1)
    print(f"alpha={a:.2f} -> arm_W50{sfx}.pt ({len(out)} tensors); wrote {out_json}")


if __name__ == "__main__":
    main()
