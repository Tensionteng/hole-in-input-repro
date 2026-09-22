#!/usr/bin/env python
"""S62: TempoPFN in the missingness-interface census.

TempoPFN (AutoML-org/TempoPFN; Moroshan, Siems, Zela, Carstensen, Hutter 2025,
arXiv:2510.25502) is the closest prior work to the paper's prescription: it pretrains
WITH mechanism-diverse missingness (configs/example.yaml: data_augmentation.nan_augmentation
= true; src/data/augmentations.py NanAugmenter samples NaN ratios and run-lengths from
GIFT-Eval NaN statistics in data/nan_stats.json -- scattered AND block patterns), but its
interface is a declaration channel without a content path:

  src/models/model.py::_compute_embeddings
      nan_mask = torch.isnan(scaled_history)
      channel_embeddings[nan_mask] = self.nan_embedding   # learned token REPLACES the value
  src/data/scalers.py::RobustScaler.compute_statistics
      valid_data = valid_data[torch.isfinite(valid_data)]  # stats from observed points only

So at NaN positions the fill's content is hard-discarded -- it cannot reach the forecast
through the embedding (replaced) or through the scaling statistics (observed-only). The
container's history_mask is a per-series PADDING mask that zeroes whole embeddings
("Suppress padded time steps completely so padding is a pure batching artifact"), not a
per-point missingness channel; and the official GIFT-Eval predictor passes raw
NaN-containing targets straight into history_values. Hence the two probe paths:

  plain     = linear fill fed as ordinary values, nothing declared
  declared  = NaN at the missing positions ("nan"), the model's own missingness path.
              This path CANNOT accept a user fill at all -- there is no input slot that
              both declares a point missing and carries its fill value. (TempoPFN does
              not impute internally either; it substitutes a learned token. Compare
              TimesFM, which overwrites the fill with its own interpolation.)

Parts:
  gate  -- determinism (two identical calls), clean-context MSE vs chronos-bolt-base on
           the same 60 windows/dataset, redraw-denominator nonzero on the plain path.
  probe -- S30/S49 permutation census: 3 datasets x 3 mechs x {plain, nan}, 60 windows,
           rate 0.3, linear fill, L=512, H=64. Flat JSON "name|ds|mech|conv".
  sweep -- fill-quality sweep: alphas {0, 0.5, 1} (fill = (1-a)*linear + a*truth),
           3 datasets x 4 mechs (adds mnar_extreme) x rates {0.3, 0.7}, 60 windows,
           relMSE vs own clean, both paths. The declared path is computed once per
           (ds, mech, rate) -- its input is provably alpha-invariant -- and the identical
           value is recorded for every alpha (documented in the JSON).
  calib -- coverage80 / pi_width80 (central 80% PI = model quantiles 0.1..0.9) at rate
           0.7, mcar (scattered) vs mnar_high (censoring), both paths, plus bolt on the
           same windows as the collapse reference (bolt mnar_high coverage fell to
           0.217 while its interval narrowed; S5-extra/appE).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/tempopfn_triton_cache")

TEMPO = os.path.join(ROOT, "models_local", "tempopfn")
sys.path.insert(0, TEMPO)
sys.path.insert(0, os.path.join(EXP, "s25_twofloor"))
sys.path.insert(0, os.path.join(EXP, "s30_crossmodel"))

import numpy as np
import torch

import run_s25_twofloor as s25
import run_s30_crossmodel as s30

L, H = s25.L, 64
N_WIN = 60
RATE_PROBE = 0.3
MECHS_PROBE = ("mcar", "block", "mnar_high")
MECHS_SWEEP = ("mcar", "block", "mnar_high", "mnar_extreme")
RATES_SWEEP = (0.3, 0.7)
ALPHAS = (0.0, 0.5, 1.0)
OUT = os.path.join(HERE, "s62_results.json")


class TempoPFN:
    name = "tempopfn-38m"
    convs = ("plain", "nan")

    def __init__(self, dev, bf16=False):
        import yaml
        from src.models.model import TimeSeriesModel
        with open(os.path.join(TEMPO, "configs", "example.yaml")) as f:
            cfg = yaml.safe_load(f)
        self.m = TimeSeriesModel(**cfg["TimeSeriesModel"]).to(dev)
        ck = torch.load(os.path.join(TEMPO, "models", "checkpoint_38M.pth"),
                        map_location=dev, weights_only=False)
        self.m.load_state_dict(ck["model_state_dict"])
        self.m.eval()
        self.dev = dev
        self.bf16 = bf16
        self.qidx_med = self.m.quantiles.index(0.5)   # 4
        self.qidx_lo = self.m.quantiles.index(0.1)    # 0
        self.qidx_hi = self.m.quantiles.index(0.9)    # 8

    @torch.no_grad()
    def fc_quant(self, ctx, miss, conv, batch=256):
        """ctx: [B, L] float32 (filled). Returns inverse-scaled [B, H, Q]."""
        from src.data.containers import BatchTimeSeriesContainer
        from src.data.frequency import Frequency
        outs = []
        for i in range(0, len(ctx), batch):
            c = torch.from_numpy(ctx[i:i + batch].copy()).to(self.dev)
            if conv == "nan":
                m_ = torch.from_numpy(miss[i:i + batch]).to(self.dev)
                c = c.clone()
                c[m_] = float("nan")
            b = c.shape[0]
            cont = BatchTimeSeriesContainer(
                history_values=c.unsqueeze(-1),
                future_values=torch.zeros(b, H, 1, device=self.dev),
                start=[np.datetime64("2017-01-01")] * b,
                frequency=[Frequency.H] * b,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=self.bf16):
                out = self.m(cont)
            pred = self.m.scaler.inverse_scale(out["result"].float(),
                                               out["scale_statistics"])
            outs.append(pred[:, :, 0, :].cpu().numpy())
        return np.concatenate(outs)

    def fc(self, ctx, miss, conv, batch=256):
        return self.fc_quant(ctx, miss, conv, batch)[..., self.qidx_med]


class BoltQuant:
    """chronos-bolt-base full quantiles, for the clean-error gate and calibration ref."""
    name = "chronos-bolt-base"

    def __init__(self, dev):
        from chronos import BaseChronosPipeline
        self.p = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map=dev, torch_dtype=torch.float32)
        self.m = self.p.inner_model if hasattr(self.p, "inner_model") else self.p.model
        self.dev = dev

    @torch.no_grad()
    def fc_quant(self, ctx, miss, conv, batch=512):
        """conv 'plain' only (the fill-then-feed reference). Returns [B, H, 9]."""
        outs = []
        for i in range(0, len(ctx), batch):
            t = torch.from_numpy(ctx[i:i + batch].copy()).to(self.dev)
            q = self.m(context=t).quantile_preds          # [B, 9, prediction_length]
            outs.append(q[:, :, :H].permute(0, 2, 1).float().cpu().numpy())
        return np.concatenate(outs)

    def fc(self, ctx, miss, conv, batch=512):
        return self.fc_quant(ctx, miss, conv, batch)[..., 4]


def alpha_blend(lin, clean, mask, a):
    c = lin.copy()
    c[mask] = (1 - a) * lin[mask] + a * clean[mask]
    return c


def relmse(pred, gt, mse_clean):
    """per-series MSE -> ratio of means (primary) and median of ratios."""
    e = ((pred.astype(np.float64) - gt.astype(np.float64)) ** 2).mean(1)
    ok = mse_clean > 1e-12
    return {"relMSE_mean": float(e[ok].mean() / mse_clean[ok].mean()),
            "relMSE_median": float(np.median(e[ok] / mse_clean[ok])),
            "n": int(ok.sum()),
            "finite": bool(np.isfinite(e).all())}


def save(res):
    json.dump(res, open(OUT + ".tmp", "w"))
    os.replace(OUT + ".tmp", OUT)


# ------------------------------------------------------------------ gate ----

def part_gate(res, tempo, bolt):
    g = res.setdefault("gate", {})
    # 1. determinism: two identical calls, both paths
    X, st = s25.load_windows(s25.DATASETS["ETTh1"], 8, s25.SEED, H)
    _, lin, mask, _ = s25.build_eval_batch(X, st, "mcar", 0.3, H, fill="linear")
    det = {}
    for conv in tempo.convs:
        y1, y2 = tempo.fc(lin, mask, conv), tempo.fc(lin, mask, conv)
        det[conv] = float(np.abs(y1 - y2).max())
    g["determinism_maxdiff"] = det
    print(f"GATE determinism: {det}", flush=True)

    # 2. clean-context error vs bolt on the same 60 windows per dataset
    clean = {}
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, N_WIN, s25.SEED, H)
        _, cl, mk0, gt = s25.build_eval_batch(X, st, "clean", 0.0, H)
        e_t = ((tempo.fc(cl, mk0, "plain").astype(np.float64) - gt) ** 2).mean()
        e_b = ((bolt.fc(cl, mk0, "plain").astype(np.float64) - gt) ** 2).mean()
        clean[ds] = {"tempopfn": float(e_t), "bolt": float(e_b),
                     "ratio": float(e_t / e_b)}
        print(f"GATE clean {ds:8s} tempopfn={e_t:10.4f} bolt={e_b:10.4f} "
              f"ratio={e_t/e_b:.3f}", flush=True)
    g["clean_mse"] = clean

    # 3. redraw denominator clearly nonzero on the plain path (one cell)
    rng = np.random.default_rng(s25.SEED)
    r = s30.perm_probe(tempo, lin, mask, "plain", rng)
    g["redraw_plain_ETTh1_mcar"] = r
    print(f"GATE redraw plain: redraw={r['redraw_rms_rel']:.3e} ratio={r['ratio']:.4f}",
          flush=True)
    g["pass"] = (max(det.values()) == 0.0
                 and r["redraw_rms_rel"] > 1e-3
                 and all(0.2 < c["ratio"] < 5.0 for c in clean.values()))
    print(f"GATE pass = {g['pass']}", flush=True)
    assert g["pass"], "sanity gate failed -- do not trust the setup"
    return res


# ----------------------------------------------------------------- probe ----

def part_probe(res, tempo):
    probe = res.setdefault("probe", {})
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, N_WIN, s25.SEED, H)
        for mech in MECHS_PROBE:
            _, lin, mask, _ = s25.build_eval_batch(X, st, mech, RATE_PROBE, H,
                                                   fill="linear")
            for conv in tempo.convs:
                key = f"{tempo.name}|{ds}|{mech}|{conv}"
                if key in probe:
                    continue
                rng = np.random.default_rng(s25.SEED)
                t0 = time.time()
                r = s30.perm_probe(tempo, lin, mask, conv, rng)
                r["seconds"] = round(time.time() - t0, 1)
                probe[key] = r
                print(f"  {ds:8s} {mech:10s} {conv:6s} perm={r['perm_rms_rel']:.3e} "
                      f"redraw={r['redraw_rms_rel']:.3e} rho={r['ratio']:.4f} "
                      f"({r['seconds']}s)", flush=True)
            save(res)
    return res


# ----------------------------------------------------------------- sweep ----

def part_sweep(res, tempo):
    sw = res.setdefault("sweep", {})
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, N_WIN, s25.SEED, H)
        _, cl, mk0, gt = s25.build_eval_batch(X, st, "clean", 0.0, H)
        pc = tempo.fc(cl, mk0, "plain")
        mse_clean = ((pc.astype(np.float64) - gt.astype(np.float64)) ** 2).mean(1)
        sw[f"{tempo.name}|{ds}|clean"] = {"mse": float(mse_clean.mean())}
        for mech in MECHS_SWEEP:
            for rate in RATES_SWEEP:
                clean, lin, mask, gt = s25.build_eval_batch(X, st, mech, rate, H,
                                                            fill="linear")
                nan_cache = None
                for conv in tempo.convs:
                    for a in ALPHAS:
                        key = f"{tempo.name}|{ds}|{mech}|{rate}|{conv}|a={a}"
                        if key in sw:
                            continue
                        if conv == "nan":
                            # declared path: input is provably alpha-invariant (fill at
                            # missing positions is overwritten by NaN either way), so the
                            # forecast -- and hence relMSE -- is identical across alpha.
                            if nan_cache is None:
                                t0 = time.time()
                                pred = tempo.fc(lin, mask, conv)
                                nan_cache = relmse(pred, gt, mse_clean)
                                nan_cache["seconds"] = round(time.time() - t0, 1)
                            sw[key] = dict(nan_cache)
                            sw[key]["note"] = "alpha-invariant input; value copied " \
                                              "from the single declared-path evaluation"
                            continue
                        t0 = time.time()
                        pred = tempo.fc(alpha_blend(lin, clean, mask, a), mask, conv)
                        r = relmse(pred, gt, mse_clean)
                        r["seconds"] = round(time.time() - t0, 1)
                        sw[key] = r
                    vals = [sw[f"{tempo.name}|{ds}|{mech}|{rate}|{conv}|a={a}"]["relMSE_mean"]
                            for a in ALPHAS]
                    print(f"  {ds:8s} {mech:13s} p={rate} {conv:6s} " +
                          " -> ".join(f"{v:7.3f}" for v in vals), flush=True)
                save(res)
    return res


# ----------------------------------------------------------------- calib ----

def part_calib(res, tempo, bolt):
    ca = res.setdefault("calib", {})
    for ds, path in s25.DATASETS.items():
        X, st = s25.load_windows(path, N_WIN, s25.SEED, H)
        _, cl, mk0, gt = s25.build_eval_batch(X, st, "clean", 0.0, H)
        cells = [("clean", 0.0, cl, mk0)]
        for mech in ("mcar", "mnar_high"):
            _, lin, mask, _ = s25.build_eval_batch(X, st, mech, 0.7, H, fill="linear")
            cells.append((mech, 0.7, lin, mask))
        for mech, rate, ctx, mask in cells:
            for mdl, tag in ((tempo, tempo.name), (bolt, bolt.name)):
                convs = mdl.convs if hasattr(mdl, "convs") else ("plain",)
                for conv in convs:
                    key = f"{tag}|{ds}|{mech}|0.7|{conv}"
                    if key in ca:
                        continue
                    q = mdl.fc_quant(ctx, mask, conv).astype(np.float64)
                    lo, hi = (q[..., mdl.qidx_lo] if mdl is tempo else q[..., 0],
                              q[..., mdl.qidx_hi] if mdl is tempo else q[..., 8])
                    med = q[..., mdl.qidx_med] if mdl is tempo else q[..., 4]
                    hit = (gt >= lo) & (gt <= hi)
                    ca[key] = {
                        "coverage80": float(hit.mean()),
                        "pi_width80": float((hi - lo).mean()),
                        "mse_median_point": float(((med - gt) ** 2).mean()),
                    }
                    print(f"  calib {ds:8s} {mech:10s} {tag:18s} {conv:6s} "
                          f"cov80={ca[key]['coverage80']:.3f} "
                          f"w80={ca[key]['pi_width80']:.3f}", flush=True)
            save(res)
    return res


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="gate,probe,sweep,calib")
    # bf16 autocast is the ONLY supported inference mode: fla's
    # chunk_gated_delta_product kernel asserts q.dtype != float32 (see
    # fla/ops/gated_delta_product/chunk.py). The official quick start also
    # runs bf16 autocast with fp32 weights. fp32 is therefore not an option.
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    dev = "cuda:0"

    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault("meta", {
        "model": "AutoML-org/TempoPFN checkpoint_38M (fp32 weights, bf16 autocast "
                 "activations -- the only supported mode: fla's chunk_gated_delta_product "
                 "kernel rejects fp32 inputs; official quick start runs the same)",
        "interface": "plain = fill fed as values; nan = NaN at missing positions "
                     "(learned nan_embedding replaces the value; RobustScaler stats "
                     "are computed over finite/observed values only). The declared "
                     "path has no slot for a user fill.",
        "L": L, "H": H, "n_win": N_WIN, "rate_probe": RATE_PROBE,
        "mechs_probe": list(MECHS_PROBE), "mechs_sweep": list(MECHS_SWEEP),
        "rates_sweep": list(RATES_SWEEP), "alphas": list(ALPHAS),
        "seed": s25.SEED, "probe": "S30 permutation vs redraw of the fill content",
        "time_features": "fixed start 2017-01-01, Frequency.H for all series "
                         "(no calendar info given to any census model)",
        "checkpoint_sha": "f88ff0ab3231415e48e55c9dc9d4f636e7200ea5",
    })

    parts = args.parts.split(",")
    tempo = TempoPFN(dev, bf16=args.bf16)
    bolt = BoltQuant(dev) if ("gate" in parts or "calib" in parts) else None

    for p in parts:
        t0 = time.time()
        if p == "gate":
            res = part_gate(res, tempo, bolt)
        elif p == "probe":
            res = part_probe(res, tempo)
        elif p == "sweep":
            res = part_sweep(res, tempo)
        elif p == "calib":
            res = part_calib(res, tempo, bolt)
        else:
            raise ValueError(p)
        save(res)
        print(f"== part {p} done in {time.time()-t0:.0f}s", flush=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
