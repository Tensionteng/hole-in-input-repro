#!/usr/bin/env python
"""S33 verification: is the causal result an artefact of how we call Moirai?

S33's causal claim rests on Moirai's declared path converting a perfect imputation into full
recovery while Chronos-Bolt's does not. Moirai is installed off-label (uni2ts `--no-deps`)
and driven through our own wrapper with `patch_size=32`, so two things need checking before
the claim can carry a paper:

  V1  Does our direct `MoiraiForecast.forward` call agree with the library's documented
      `create_predictor` / GluonTS path on identical windows?
  V2  Does the conclusion survive the patch-size choice, including the library's `"auto"`
      setting that selects patch size per series by validation loss?

V2 is the one that matters: if the alpha-sweep slope flips sign or flattens under `auto`,
the causal evidence is a configuration artefact.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))

import numpy as np
import torch

sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
import run_s25_twofloor as s25

H = 64
L = s25.L
ALPHAS = (0.0, 0.5, 1.0)
OUT = os.path.join(HERE, "s33_verify.json")


def build(patch_size, ctx_len=L):
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    m = MoiraiForecast(
        module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base"),
        prediction_length=H, context_length=ctx_len, patch_size=patch_size,
        num_samples=20, target_dim=1, feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0).to("cuda")
    m.eval()
    return m


@torch.no_grad()
def fc(mod, ctx, miss, declared, batch=64):
    outs = []
    for i in range(0, len(ctx), batch):
        c = ctx[i:i + batch].astype(np.float32)
        mk = miss[i:i + batch]
        obs = (~mk) if declared else np.ones_like(mk, bool)
        pt = torch.from_numpy(c).to("cuda").unsqueeze(-1)
        om = torch.from_numpy(obs).to("cuda").unsqueeze(-1)
        pad = torch.zeros(pt.shape[:2], dtype=torch.bool, device="cuda")
        s = mod(past_target=pt, past_observed_target=om, past_is_pad=pad)
        outs.append(s.median(dim=1).values.squeeze(-1).float().cpu().numpy())
    return np.concatenate(outs)


def v1_predictor_agreement(res, n=32):
    """Our direct forward vs the library's own create_predictor path, clean context."""
    try:
        from gluonts.dataset.common import ListDataset
        mod = build(32)
        pred = mod.create_predictor(batch_size=16)
        X, st = s25.load_windows(s25.DATASETS["ETTh1"], 12, s25.SEED, H)
        _, cl, mk, _ = s25.build_eval_batch(X, st, "clean", 0.0, H)
        cl, mk = cl[:n], mk[:n]
        torch.manual_seed(1); ours = fc(mod, cl, mk, declared=False)
        torch.manual_seed(2); ours2 = fc(mod, cl, mk, declared=False)
        self_rel = float(np.abs(ours - ours2).max() / (np.abs(ours).mean() + 1e-9))
        self_corr = float(np.corrcoef(ours.ravel(), ours2.ravel())[0, 1])
        ds = ListDataset([{"start": "2020-01-01 00:00", "target": r} for r in cl], freq="H")
        theirs = []
        for f in pred.predict(ds):
            theirs.append(np.median(f.samples, axis=0)[:H])
        theirs = np.stack(theirs)
        rel = float(np.abs(ours - theirs).max() / (np.abs(theirs).mean() + 1e-9))
        corr = float(np.corrcoef(ours.ravel(), theirs.ravel())[0, 1])
        res["v1"] = {"n": int(len(cl)), "max_rel_diff": rel, "corr": corr,
                     "self_max_rel_diff_two_seeds": self_rel, "self_corr_two_seeds": self_corr,
                     "note": "sampling is stochastic, so exact equality is not expected; "
                             "high correlation + small relative gap validates the wrapper"}
        print(f"V1 sampling noise floor (ours, 2 seeds):  max rel diff={self_rel:.4f} "
              f"corr={self_corr:.5f}", flush=True)
        print(f"V1 ours vs create_predictor:               max rel diff={rel:.4f} "
              f"corr={corr:.5f}", flush=True)
        print(f"   -> {'WITHIN the sampling noise floor' if rel <= self_rel * 1.5 else 'EXCEEDS the noise floor'}",
              flush=True)
    except Exception as e:
        res["v1"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"V1 SKIP: {type(e).__name__}: {str(e)[:160]}", flush=True)
    return res


def v2_patch_size(res, n_win=80):
    """Does the alpha-sweep slope on the DECLARED path survive the patch-size choice?"""
    print("\nV2 alpha-sweep on the declared path, by patch size "
          "(median relMSE at alpha=0 / 0.5 / 1)", flush=True)
    for ps in (16, 32, 64, "auto"):
        try:
            mod = build(ps)
        except Exception as e:
            print(f"  patch={ps}: BUILD FAIL {type(e).__name__}: {str(e)[:100]}", flush=True)
            continue
        for ds in ("ETTh1", "ETTm1"):
            X, st = s25.load_windows(s25.DATASETS[ds], n_win, s25.SEED, H)
            _, cl, _, gt_c = s25.build_eval_batch(X, st, "clean", 0.0, H)
            try:
                base = ((fc(mod, cl, np.zeros_like(cl, bool), False) - gt_c) ** 2).mean(1)
                ok = base > 1e-12
            except Exception as e:
                print(f"  patch={ps}: {ds} FORWARD FAIL {type(e).__name__}: {str(e)[:90]}",
                      flush=True)
                base = None
            for mech in ("mcar", "block"):
                if base is None:
                    continue
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, 0.7, H, fill="linear")
                vals = []
                for a in ALPHAS:
                    c = lin.copy()
                    c[mask] = (1 - a) * lin[mask] + a * clean[mask]
                    e = ((fc(mod, c, mask, True) - gt) ** 2).mean(1)
                    vals.append(float(np.median(e[ok] / base[ok])))
                res.setdefault("v2", {})[f"{ds}|{mech}|patch{ps}"] = {
                    "alphas": list(ALPHAS), "rel_median": vals,
                    "slope": vals[-1] - vals[0]}
                print(f"  patch={str(ps):5s} {ds:6s} {mech:6s} p=0.7  " +
                      " -> ".join(f"{v:.3f}" for v in vals) +
                      f"   slope {vals[-1]-vals[0]:+.3f}", flush=True)
        del mod
        torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    np.random.seed(s25.SEED)
    torch.manual_seed(s25.SEED)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res = v1_predictor_agreement(res)
    res = v2_patch_size(res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", OUT)
