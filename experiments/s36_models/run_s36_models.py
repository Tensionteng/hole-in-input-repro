#!/usr/bin/env python
"""S36: the fill-reachable set across every checkpoint we can run.

S30 measured the permutation probe on five checkpoints from four families. Proposition 1
predicts a trichotomy that should hold for any model, so the interesting test is breadth:
more families, more sizes within a family, and --- new here --- families that ship NO
declared missing-data path at all, where the proposition predicts the plain-fill regime is
the only one available and the attack surface is therefore always full rank.

Added over S30:
  Moirai 1.1-R small / large       (size ladder inside a family with a declared path)
  Moirai-MoE 1.0-R small / base    (mixture-of-experts variant, declared path)
  Chronos-T5 base                  (size ladder in the tokeniser family)
  Chronos-Bolt tiny / mini / small (size ladder in the patch family)
  TimeMoE 50M / 200M               (decoder-only, no declared path)
  Sundial base 128M                (flow-matching decoder, no declared path)

Anchor gate: chronos-bolt-base must reproduce its S30 aggregates within 5%.
"""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LOCAL = os.path.join(ROOT, "models_local")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch

sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))
import run_s25_twofloor as s25
import run_s30_crossmodel as s30

L, H = s25.L, s30.H
OUT = os.path.join(HERE, "s36_results.json")
MECHS = s30.MECHS
RATE = s30.RATE


# --------------------------------------------------------------- wrappers ----

class _Bolt(s30.Bolt):
    """Chronos-Bolt at a chosen size, loaded from the local mirror."""
    def __init__(self, dev, size):
        from chronos import BaseChronosPipeline
        self.name = f"chronos-bolt-{size}"
        self.convs = ("plain", "mask", "nan")
        src = os.path.join(LOCAL, f"chronos-bolt-{size}")
        src = src if os.path.isdir(src) else f"amazon/chronos-bolt-{size}"
        self.p = BaseChronosPipeline.from_pretrained(src, device_map=dev,
                                                     torch_dtype=torch.float32)
        self.m = self.p.inner_model if hasattr(self.p, "inner_model") else self.p.model
        self.dev = dev


class _T5(s30.T5):
    def __init__(self, dev, size):
        from chronos import BaseChronosPipeline
        self.name = f"chronos-t5-{size}"
        self.convs = ("plain", "nan")
        src = os.path.join(LOCAL, f"chronos-t5-{size}")
        src = src if os.path.isdir(src) else f"amazon/chronos-t5-{size}"
        self.p = BaseChronosPipeline.from_pretrained(src, device_map=dev,
                                                     torch_dtype=torch.float32)
        self.dev = dev


class _Moirai(s30.Moirai):
    def __init__(self, dev, size, moe=False):
        if moe:
            from uni2ts.model.moirai_moe import MoiraiMoEForecast as F, MoiraiMoEModule as M
            self.name, ps = f"moirai-moe-1.0-R-{size}", 16
            path = os.path.join(LOCAL, f"moirai-moe-1.0-R-{size}")
        else:
            from uni2ts.model.moirai import MoiraiForecast as F, MoiraiModule as M
            self.name, ps = f"moirai-1.1-R-{size}", 32
            path = os.path.join(LOCAL, f"moirai-1.1-R-{size}")
        self.convs = ("plain", "nan")
        self.mod = F(module=M.from_pretrained(path), prediction_length=H, context_length=L,
                     patch_size=ps, num_samples=20, target_dim=1, feat_dynamic_real_dim=0,
                     past_feat_dynamic_real_dim=0).to(dev)
        self.mod.eval()
        self.dev = dev


class TimeMoE:
    """Decoder-only MoE. Ships no missing-data path: no mask argument, no NaN handling.

    We call the multi-horizon head directly rather than generate(), which is equivalent for
    a 64-step forecast and avoids the vendored generation mixin (see tf5_compat)."""
    convs = ("plain", "nan")

    def __init__(self, dev, size):
        import tf5_compat  # noqa: F401
        from transformers import AutoModelForCausalLM
        self.name = f"TimeMoE-{size}"
        self.m = AutoModelForCausalLM.from_pretrained(os.path.join(LOCAL, f"TimeMoE-{size}"),
                                                      trust_remote_code=True,
                                                      dtype=torch.float32).to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def fc(self, ctx, miss, conv, batch=64):
        outs = []
        for i in range(0, len(ctx), batch):
            c = ctx[i:i + batch].copy()
            if conv == "nan":
                c[miss[i:i + batch]] = np.nan
            t = torch.from_numpy(c).to(self.dev)
            o = self.m(input_ids=t, use_cache=False, max_horizon_length=H)
            outs.append(o.logits[:, -1, :H].float().cpu().numpy())
        return np.concatenate(outs)


class Sundial:
    """Flow-matching decoder. Also ships no missing-data path.

    revin=True inside the checkpoint has a broadcast bug for batch != num_samples, so we
    apply the identical instance normalisation outside the model."""
    convs = ("plain", "nan")
    name = "sundial-base-128m"

    def __init__(self, dev, n_samples=20):
        import tf5_compat  # noqa: F401
        from transformers import AutoModelForCausalLM
        self.m = AutoModelForCausalLM.from_pretrained(os.path.join(LOCAL, "sundial-base-128m"),
                                                      trust_remote_code=True,
                                                      dtype=torch.float32).to(dev).eval()
        self.dev, self.ns = dev, n_samples

    @torch.no_grad()
    def fc(self, ctx, miss, conv, batch=64):
        outs = []
        for i in range(0, len(ctx), batch):
            c = ctx[i:i + batch].copy()
            if conv == "nan":
                c[miss[i:i + batch]] = np.nan
            t = torch.from_numpy(c).to(self.dev)
            mu = t.mean(1, keepdim=True)
            sd = t.std(1, keepdim=True, unbiased=False).clamp_min(1e-2)
            torch.manual_seed(s25.SEED)
            o = self.m(input_ids=(t - mu) / sd, max_output_length=H, num_samples=self.ns,
                       revin=False, use_cache=False)
            p = o.logits * sd.unsqueeze(-1) + mu.unsqueeze(-1)
            outs.append(p.median(dim=1).values[:, :H].float().cpu().numpy())
        return np.concatenate(outs)


REGISTRY = {
    "bolt_tiny":   lambda d: _Bolt(d, "tiny"),
    "bolt_mini":   lambda d: _Bolt(d, "mini"),
    "bolt_small":  lambda d: _Bolt(d, "small"),
    "bolt_base":   lambda d: _Bolt(d, "base"),
    "t5_base":     lambda d: _T5(d, "base"),
    "moirai_s":    lambda d: _Moirai(d, "small"),
    "moirai_l":    lambda d: _Moirai(d, "large"),
    "moe_s":       lambda d: _Moirai(d, "small", moe=True),
    "moe_b":       lambda d: _Moirai(d, "base", moe=True),
    "timemoe_50":  lambda d: TimeMoE(d, "50M"),
    "timemoe_200": lambda d: TimeMoE(d, "200M"),
    "sundial":     Sundial,
}
ALL = ",".join(REGISTRY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=ALL)
    ap.add_argument("--n-win", type=int, default=60)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    out = args.out
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    res = json.load(open(out)) if os.path.exists(out) else {}
    res.setdefault("meta", {"rate": RATE, "mechs": list(MECHS), "n_win": args.n_win,
                            "n_perm": s30.N_PERM, "H": H,
                            "probe": "permutation vs redraw of the fill content",
                            "note": "same protocol as S30; extends the model axis"})
    for key in args.models.split(","):
        try:
            model = REGISTRY[key]("cuda")
        except Exception as e:
            print(f"SKIP {key}: {type(e).__name__}: {str(e)[:160]}", flush=True)
            res.setdefault("models", {})[key] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        print(f"\n== {model.name} ==", flush=True)
        cells, nanfrac = {}, {}
        for ds, path in s25.DATASETS.items():
            X, st = s25.load_windows(path, args.n_win, s25.SEED, H)
            for mech in MECHS:
                _, lin, mask, _ = s25.build_eval_batch(X, st, mech, RATE, H, fill="linear")
                for conv in model.convs:
                    rng = np.random.default_rng(s25.SEED)
                    t0 = time.time()
                    try:
                        y = model.fc(lin, mask, conv)
                        nf = float(np.mean(~np.isfinite(y)))
                        nanfrac.setdefault(conv, []).append(nf)
                        if nf > 0.5:                      # no usable declared path
                            print(f"  {ds:8s} {mech:10s} {conv:6s} non-finite output "
                                  f"({nf:.1%}) -- convention unsupported", flush=True)
                            cells[f"{ds}|{mech}|{conv}"] = {"unsupported": True,
                                                            "nonfinite_frac": nf}
                            continue
                        r = s30.perm_probe(model, lin, mask, conv, rng)
                    except Exception as e:
                        print(f"  {ds:8s} {mech:10s} {conv:6s} FAIL {type(e).__name__}: "
                              f"{str(e)[:90]}", flush=True)
                        cells[f"{ds}|{mech}|{conv}"] = {"error": f"{type(e).__name__}"}
                        continue
                    r["nonfinite_frac"] = nf
                    cells[f"{ds}|{mech}|{conv}"] = r
                    print(f"  {ds:8s} {mech:10s} {conv:6s} perm={r['perm_rms_rel']:.3e} "
                          f"redraw={r['redraw_rms_rel']:.3e} ratio={r['ratio']:.4f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        agg = {}
        for conv in model.convs:
            v = [c["ratio"] for k, c in cells.items()
                 if k.endswith("|" + conv) and "ratio" in c]
            u = [1 for k, c in cells.items()
                 if k.endswith("|" + conv) and c.get("unsupported")]
            agg[conv] = {"ratio_median": float(np.median(v)) if v else None,
                         "ratio_max": float(np.max(v)) if v else None,
                         "n_cells": len(v), "n_unsupported": len(u)}
        res.setdefault("models", {})[key] = {"name": model.name, "cells": cells, "agg": agg}
        print("  AGG " + model.name + ": " + "  ".join(
            f"{c}=" + ("n/a" if agg[c]["ratio_median"] is None
                       else f"{agg[c]['ratio_median']:.4f}") for c in agg), flush=True)
        del model
        torch.cuda.empty_cache()
        json.dump(res, open(out + ".tmp", "w"))
        os.replace(out + ".tmp", out)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
