#!/usr/bin/env python
"""Is the S27/S37 mismatch on datasets 2 and 3 caused by a drifting clean baseline?

S27's sweep() computes the per-series clean denominator with whatever input-patch-embedding
happens to be loaded at that moment. Inside its dataset loop the previous dataset's last
adapted interface is still loaded, so every dataset after the first is normalised by an
ADAPTED clean forecast rather than the released model's. This script recomputes the weather
cells both ways and checks which one reproduces the stored S27 values.
"""
import json, os, sys
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
for d in ("s25_twofloor", "s27_interface"):
    sys.path.insert(0, os.path.join(EXP, d))
import run_s25_twofloor as s25, run_s27_interface as s27

H = s27.H
bolt = s25.Bolt("cuda")
base_sd = s27.save_proj(bolt)
CK = os.path.join(EXP, "s27_interface", "s27_ckpt")
ad = {i: torch.load(os.path.join(CK, f"proj_{i}.pt"), map_location="cuda")
      for i in ("native", "dual", "dual_obsnorm")}
S27 = json.load(open(os.path.join(EXP, "s27_interface", "s27_results.json")))["sweep"]

X, st = s25.load_windows(s25.DATASETS["weather"], 300, s25.SEED, H)
_, cl, _, gt = s25.build_eval_batch(X, st, "clean", 0.0, H)
bases = {}
s27.load_proj(bolt, base_sd)
bases["released weights"] = ((bolt.median_np(cl) - gt) ** 2).mean(1)
s27.load_proj(bolt, ad["dual_obsnorm"])   # what is still loaded after the previous dataset
bases["adapted dual_obsnorm"] = ((bolt.median_np(cl) - gt) ** 2).mean(1)
s27.load_proj(bolt, base_sd)

clean, lin, mask, fut = s25.build_eval_batch(X, st, "block", 0.7, H, fill="linear")
obs = (~mask).astype(np.float32)
print(f"{'cell':34s} {'S27':>8s} " + " ".join(f"{k:>22s}" for k in bases))
for iface in ("native", "dual_obsnorm"):
    s27.load_proj(bolt, ad[iface])
    for a in (0.0, 1.0):
        ctx = s27.blend_fill(clean, lin, mask, a)
        e = ((s27.median_iface_np(bolt, ctx, obs, iface) - fut) ** 2).mean(1)
        stored = S27[f"adapted|weather|block|0.7|{a}|{iface}"]["median"]
        vals = []
        for k, b in bases.items():
            ok = b > 1e-12
            vals.append(float(np.median(e[ok] / b[ok])))
        print(f"weather|block|0.7|{a}|{iface:13s} {stored:8.4f} " +
              " ".join(f"{v:22.4f}" for v in vals))
s27.load_proj(bolt, base_sd)
