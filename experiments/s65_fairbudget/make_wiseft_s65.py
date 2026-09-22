#!/usr/bin/env python
"""S65 WiSE-FT: interpolate each no-missingness control 0.5/0.5 with its stock.

theta = 0.5 * stock + 0.5 * control -- the same procedure as make_wiseft_s60.py
/ make_wiseft_s61.py (and make_wiseft.py S58 for Chronos-Bolt). Produces
s65_ckpt/c2_nomiss_w50.pt and s65_ckpt/m2_nomiss_w50.pt so the control arms are
scored under the identical post-processing as the recipe arms.
"""
import json
import os

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CK = os.path.join(HERE, "s65_ckpt")
OUT = os.path.join(HERE, "s65_wiseft.json")


def wiseft(name, stock_state, ctl_path, out_path, res):
    p5 = torch.load(ctl_path, weights_only=True, map_location="cpu")
    assert set(stock_state) == set(p5), f"{name}: state dict keys differ"
    a = 0.5
    out = {k: (1 - a) * stock_state[k].float() + a * p5[k].float()
           for k in stock_state}
    torch.save(out, out_path)
    res[name] = {"alpha": a, "recipe": "theta = (1-a)*stock + a*control(nomiss)",
                 "n_tensors": len(out),
                 "max_abs_weight_delta_vs_stock": float(
                     max((out[k] - stock_state[k].float()).abs().max()
                         for k in stock_state)),
                 "control": os.path.relpath(ctl_path, HERE),
                 "out": os.path.relpath(out_path, HERE)}
    print(f"{name}: alpha={a:.2f} -> {out_path} ({len(out)} tensors)")


def main():
    res = {}
    from chronos.chronos2.model import Chronos2Model
    m = Chronos2Model.from_pretrained(os.path.join(ROOT, "models_local",
                                                   "chronos-2"),
                                      torch_dtype=torch.float32)
    wiseft("c2", m.state_dict(), os.path.join(CK, "c2_nomiss.pt"),
           os.path.join(CK, "c2_nomiss_w50.pt"), res)
    del m

    from uni2ts.model.moirai2 import Moirai2Module
    m = Moirai2Module.from_pretrained(os.path.join(ROOT, "models_local",
                                                   "moirai-2.0-R-small"))
    wiseft("m2", m.state_dict(), os.path.join(CK, "m2_nomiss.pt"),
           os.path.join(CK, "m2_nomiss_w50.pt"), res)
    del m

    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
