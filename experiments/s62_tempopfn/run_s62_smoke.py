#!/usr/bin/env python
"""S62 smoke: load TempoPFN, check determinism (fp32 and bf16), tiny forecast sanity.

TempoPFN (AutoML-org/TempoPFN, Moroshan et al. 2025) is a univariate linear-RNN TSFM.
Its missingness interface, read off src/models/model.py:

  _compute_embeddings: nan_mask = torch.isnan(scaled_history)
      channel_embeddings[nan_mask] = self.nan_embedding   # learned vector, fill discarded
  RobustScaler.compute_statistics: valid_data = valid_data[isfinite(valid_data)]
      -> scaling statistics from OBSERVED points only

so the declared path (NaN in history_values) replaces each missing point's embedding by
a single learned token and computes statistics over observed points only: the fill's
content cannot reach the forecast AT ALL (not even through the scaling statistics).
history_mask is a per-series PADDING mask (zeroes whole embeddings), not a per-point
missingness channel. The GIFT-Eval predictor passes raw NaN-containing targets straight
into history_values, i.e. NaN-in-input IS the model's official missingness interface.
"""
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

import numpy as np
import torch

import run_s25_twofloor as s25

L, H = s25.L, 64
DEV = "cuda:0"


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
        self.qidx_med = self.m.quantiles.index(0.5)

    @torch.no_grad()
    def fc_quant(self, ctx, miss, conv, batch=256):
        """ctx: [B, L] float32 numpy (filled). Returns [B, H, Q] inverse-scaled."""
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
                history_values=c.unsqueeze(-1),                    # [B, L, 1]
                future_values=torch.zeros(b, H, 1, device=self.dev),
                start=[np.datetime64("2017-01-01")] * b,
                frequency=[Frequency.H] * b,
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=self.bf16):
                out = self.m(cont)
            pred = self.m.scaler.inverse_scale(out["result"].float(),
                                               out["scale_statistics"])
            outs.append(pred[:, :, 0, :].cpu().numpy())            # [B, H, Q]
        return np.concatenate(outs)

    def fc(self, ctx, miss, conv, batch=256):
        return self.fc_quant(ctx, miss, conv, batch)[..., self.qidx_med]


def main():
    torch.manual_seed(s25.SEED)
    np.random.seed(s25.SEED)
    bf16 = "--bf16" in sys.argv
    t0 = time.time()
    model = TempoPFN(DEV, bf16=bf16)
    print(f"loaded in {time.time()-t0:.0f}s (bf16={bf16})", flush=True)

    # determinism on 8 ETTh1 windows (56 series), both conventions
    X, st = s25.load_windows(s25.DATASETS["ETTh1"], 8, s25.SEED, H)
    _, lin, mask, gt = s25.build_eval_batch(X, st, "mcar", 0.3, H, fill="linear")
    for conv in ("plain", "nan"):
        t0 = time.time()
        y1 = model.fc(lin, mask, conv)
        y2 = model.fc(lin, mask, conv)
        d = float(np.abs(y1 - y2).max())
        finite = bool(np.isfinite(y1).all())
        print(f"  conv={conv:6s} determinism max|diff|={d:.3e} finite={finite} "
              f"({time.time()-t0:.0f}s for 2x{len(lin)} series)", flush=True)

    # declared path must be EXACTLY invariant to the fill content
    rng = np.random.default_rng(s25.SEED)
    cp = lin.copy()
    for i in range(len(lin)):
        idx = np.flatnonzero(mask[i])
        cp[i, idx] = lin[i, idx][rng.permutation(len(idx))]
    d_nan = float(np.abs(model.fc(cp, mask, "nan") - model.fc(lin, mask, "nan")).max())
    d_plain = float(np.abs(model.fc(cp, mask, "plain") - model.fc(lin, mask, "plain")).max())
    print(f"  fill-permutation effect: nan={d_nan:.3e} (must be ~0) "
          f"plain={d_plain:.3e} (must be > 0)", flush=True)

    # clean-context sanity MSE on those 8 windows
    _, cl, _, gtc = s25.build_eval_batch(X, st, "clean", 0.0, H)
    pred = model.fc(cl, np.zeros_like(cl, bool), "plain")
    mse = float(((pred - gtc) ** 2).mean())
    print(f"  clean ETTh1 MSE (8 win x 7 ch, H=64): {mse:.4f}", flush=True)


if __name__ == "__main__":
    main()
