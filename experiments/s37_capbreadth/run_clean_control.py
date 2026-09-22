#!/usr/bin/env python
"""Control: what does the two-minute input-embedding adaptation do on a CLEAN context?

Closure exceeds 100% in a few cells, which can only happen if the restored interface beats
the released model's clean forecast. The adaptation is trained on holed windows but it edits a
module the model uses on every input, so part of any gain may be generic rather than
interface-specific. This measures that part directly: adapted vs released weights, clean
context, no missingness anywhere.
"""
import json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
for d in ("s25_twofloor", "s27_interface", "s35_breadth"):
    sys.path.insert(0, os.path.join(EXP, d))
import run_s25_twofloor as s25, run_s27_interface as s27, run_s35_breadth as s35

H = s27.H
bolt = s25.Bolt("cuda")
base_sd = s27.save_proj(bolt)
CK = os.path.join(EXP, "s27_interface", "s27_ckpt")
ad = {i: torch.load(os.path.join(CK, f"proj_{i}.pt"), map_location="cuda")
      for i in ("native", "dual", "dual_obsnorm")}
out = {}
print(f"{'dataset':12s} {'released':>10s} " +
      " ".join(f"{'adapted ' + i:>20s}" for i in ad))
for name in s35.DATASETS:
    X, st = s35.load(name, 300)
    if X is None:
        continue
    _, cl, _, gt = s35.build(X, st, "clean", 0.0)
    keep = s35.finite(cl) & s35.finite(gt) & (cl.std(axis=1) > 1e-8)
    cl, gt = cl[keep], gt[keep]
    obs = np.ones_like(cl)
    s27.load_proj(bolt, base_sd)
    base = ((bolt.median_np(cl) - gt) ** 2).mean(1)
    ok = base > 1e-12
    row = {}
    for iface, sd in ad.items():
        s27.load_proj(bolt, sd)
        e = ((s27.median_iface_np(bolt, cl, obs, iface) - gt) ** 2).mean(1)
        row[iface] = float(np.median(e[ok] / base[ok]))
    s27.load_proj(bolt, base_sd)
    out[name] = {"n": int(ok.sum()), "clean_relMSE": row,
                 "insample": name in ("ETTh1", "ETTm1", "weather")}
    print(f"{name:12s} {1.0:10.4f} " +
          " ".join(f"{row[i]:20.4f}" for i in ad), flush=True)
json.dump(out, open(os.path.join(HERE, "clean_control.json"), "w"), indent=1)
print("\nwrote clean_control.json")
