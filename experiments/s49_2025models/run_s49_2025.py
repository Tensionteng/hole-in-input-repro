#!/usr/bin/env python
"""S49: the 2025 generation in the census -- TiRex, FlowState, Timer-XL rho probes.

TiRex (NX-AI, NeurIPS 2025) and FlowState r1 (IBM, 2025) are the (a)+flag convention:
declaring via NaN zeroes the content and concatenates the missingness flag into the content
path, statistics from observed points only. Timer-XL (thuml, ICLR 2025) is convention (d):
no declared path at all. Probe = S30's permutation test, 3 datasets x 3 mechs x 60 windows.
"""
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
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
import run_s25_twofloor as s25
import run_s30_crossmodel as s30

L, H = s25.L, 64
LOCAL = os.path.join(ROOT, "models_local")
OUT = os.path.join(HERE, "s49_results.json")


class TiRex:
    name = "tirex-1.1"
    convs = ("plain", "nan")

    def __init__(self, dev):
        from tirex.models.tirex import TiRexZero
        self.m = TiRexZero.from_pretrained(
            os.path.join(LOCAL, "tirex", "model.ckpt"), backend="torch", device=dev)
        self.dev = dev

    @torch.no_grad()
    def fc(self, ctx, miss, conv, batch=64):
        outs = []
        for i in range(0, len(ctx), batch):
            c = torch.from_numpy(ctx[i:i + batch].copy()).to(self.dev)
            if conv == "nan":
                m_ = torch.from_numpy(miss[i:i + batch]).to(self.dev)
                c[m_] = float("nan")
            o = self.m.forecast(c, prediction_length=H)
            q = o[0] if isinstance(o, (list, tuple)) else o   # (quantiles, mean)
            outs.append(q[:, :, 4].float().cpu().numpy())
        return np.concatenate(outs)


class FlowState:
    name = "flowstate-r1"
    convs = ("plain", "nan")

    def __init__(self, dev):
        from tsfm_public.models.flowstate import FlowStateForPrediction
        self.m = FlowStateForPrediction.from_pretrained(
            os.path.join(LOCAL, "flowstate-r1")).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def fc(self, ctx, miss, conv, batch=64):
        outs = []
        for i in range(0, len(ctx), batch):
            c = torch.from_numpy(ctx[i:i + batch].copy()).to(self.dev)
            if conv == "nan":
                m_ = torch.from_numpy(miss[i:i + batch]).to(self.dev)
                c[m_] = float("nan")
            o = self.m(past_values=c.unsqueeze(-1), prediction_length=H)
            outs.append(o.quantile_outputs[:, 4, :, 0].float().cpu().numpy())
        return np.concatenate(outs)


class TimerXL:
    name = "timer-base-84m"
    convs = ("plain", "nan")

    def __init__(self, dev):
        from transformers import AutoModelForCausalLM
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "seen_tokens"):
            # Timer-XL's remote generation mixin pins the transformers-4.x cache API;
            # v5 renamed it to get_seq_length(). Code-level shim, env untouched.
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(DynamicCache, "get_usable_length"):
            DynamicCache.get_usable_length = lambda self, *a, **k: self.get_seq_length()
        self.m = AutoModelForCausalLM.from_pretrained(
            os.path.join(LOCAL, "timer-base-84m"), trust_remote_code=True,
            dtype=torch.float32).to(dev).eval()
        # transformers v5 leaves non-persistent buffers uninitialised on this load path
        # (bf16 ckpt + meta-device); rebuild the rope buffers from their formula.
        for layer in self.m.model.layers:
            rope = layer.self_attn.rotary_emb
            dim = rope.dim
            inv = 1.0 / (rope.base ** (torch.arange(0, dim, 2, dtype=torch.int64)
                                       .float().to(rope.inv_freq.device) / dim))
            rope.inv_freq.copy_(inv)
            rope._set_cos_sin_cache(seq_len=rope.max_seq_len_cached,
                                    device=rope.inv_freq.device, dtype=torch.float32)
        self.dev = dev

    @torch.no_grad()
    def fc(self, ctx, miss, conv, batch=64):
        outs = []
        for i in range(0, len(ctx), batch):
            c = ctx[i:i + batch].copy()
            if conv == "nan":
                c[miss[i:i + batch]] = float("nan")
            c = c[:, -(c.shape[1] // 96 * 96):]      # Timer-XL patches in 96s
            t = torch.from_numpy(c).to(self.dev)
            o = self.m(input_ids=t, use_cache=False)
            outs.append(o.logits[:, :H].float().cpu().numpy())
        return np.concatenate(outs)


def main():
    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for cls in (TiRex, FlowState, TimerXL):
        try:
            model = cls("cuda")
        except Exception as e:
            print(f"SKIP {cls.__name__}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            res.setdefault("models", {})[cls.__name__] = {"error": str(e)[:300]}
            continue
        print(f"\n== {model.name} ==", flush=True)
        # determinism check on a few windows
        X, st = s25.load_windows(s25.DATASETS["ETTh1"], 8, s25.SEED, H)
        _, cl, _, _ = s25.build_eval_batch(X, st, "clean", 0.0, H)
        d = float(np.abs(model.fc(cl, np.zeros_like(cl, bool), "plain")
                         - model.fc(cl, np.zeros_like(cl, bool), "plain")).max())
        print(f"  determinism max|diff| = {d:.2e}", flush=True)
        for ds, path in s25.DATASETS.items():
            X, st = s25.load_windows(path, 60, s25.SEED, H)
            for mech in ("mcar", "block", "mnar_high"):
                _, lin, mask, _ = s25.build_eval_batch(X, st, mech, 0.3, H, fill="linear")
                for conv in model.convs:
                    rng = np.random.default_rng(s25.SEED)
                    r = s30.perm_probe(model, lin, mask, conv, rng)
                    res.setdefault("probe", {})[f"{model.name}|{ds}|{mech}|{conv}"] = r
                    print(f"  {ds:8s} {mech:10s} {conv:6s} rho={r['ratio']:.4f}", flush=True)
            json.dump(res, open(OUT, "w"))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
