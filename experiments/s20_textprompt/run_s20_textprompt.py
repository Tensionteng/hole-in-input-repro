#!/usr/bin/env python
"""S20: mechanism-level TEXT prompts vs missing-data degradation in
text-conditioned time-series models (IMM-TSF / Time-IMM repos).

Hypothesis (pre-registered in s20_notes.md BEFORE any eval run):
  H1 (bias exists): text-conditioned forecasters fed fill-then-feed contexts
      show the same MNAR underestimation bias we found on bolt in S19 -- under
      mnar_high the signed level bias mu_pred - mu_gt of the forecast mean is
      systematically negative (low-value islands -> low level anchor).
  H2 (mechanism text calibrates): a prompt that precisely describes the
      missingness mechanism ("high readings above ~the 90th percentile were
      censored") moves the level bias toward zero and reduces relMSE vs the
      no-extra-info default prompt. Expected strongest on mnar_high p=0.7.
  H3 (generic text weak): a generic "some data is missing" prompt has no or
      strictly weaker effect than the matched mechanism prompt.
  H4 (necessity / mismatch control): a mechanism prompt that does NOT match
      the true mechanism does not help (and may hurt, e.g. "highs censored"
      on mcar data could induce over-prediction).
  H5 (clean anchor): on clean windows, all prompt tiers give ~identical
      forecasts to the default prompt (text must not hurt clean data):
      relMSE within [0.97, 1.03].

Verdict rules (pre-registered):
  "effective"   = on mnar_high p=0.7, matched mechanism text cuts |signed
                  level bias| by >=20% AND relMSE by >=2% vs default prompt,
                  paired t>2, on both datasets of a model, AND matched beats
                  mismatched (paired t>2 on per-window mse).
  "useless"     = no prompt tier moves relMSE by >=2% or bias significantly
                  anywhere.
  "conditional" = anything in between (e.g. bias corrected but MSE flat, or
                  only one model/dataset responds).

Repos (read-only, imported via sys.path -- no repo file is modified):
  repos/IMM-TSF  = benchmark library (models + text fusion). tPatchGNN +
                   FusionModel (TTF_RecAvg + MMF_GR_Add, frozen GPT-2 text
                   encoder) is the paper's flagship multimodal path: text
                   enters as per-window notes -> LLM embeddings -> recency
                   averaging -> GRU-gated residual add onto the forecast.
                   TimeLLM is the literal prompt path: the domain_des string
                   is embedded into "<|start_prompt|>Dataset: {domain_des}.
                   Forecast next ... Min/Max/Median/Trend/Top lags ..." and
                   prepended to the time-series patch tokens of a frozen GPT-2.
  repos/Time-IMM = dataset collection only (no model code); used for the
                   official-pipeline reproduction gate (ILINet).

Feasibility gate (--repro): reproduce IMM-TSF paper (arXiv:2506.10412v4,
App. L Table 9) ILINet test MSE within +-10%:
  tPatchGNN unimodal 1.6163, tPatchGNN multimodal 1.4877, TimeLLM unimodal
  1.1243. Uses THEIR parse_datasets pipeline + their training hyperparams
  (Adam lr 1e-3, w_decay 0.01, batch 8, early stop patience 3 delta 1e-4,
  seed 1; per-record z-normalization over the full record as in their code).
  Deviations from their defaults (speed only, content-preserving):
  note token cap 256 instead of 1024 (notes are ~100-token 5-sentence
  summaries) and tf32 matmul allowed on A800.

Main experiment (--core): ETTh1 + weather, windows/seeds EXACTLY as S5
(run_s5_missing.py: L=512, H=96, 300 test origins in the last 20% of the
timeline, seed 20250810; per-channel masks). Models are trained from scratch
on CLEAN windows (origins in the first 60% of the timeline, val 60-80%,
per-channel z-normalization from the train region only) -- the deployment
scenario: trained clean, deployed under censoring, the operator adds a text
note. Eval cells: clean + mech {mcar, block, mnar_high} x p {0.3, 0.7}, fill
linear (fill-then-feed, observed_mask=ones: the ONLY missingness signal the
model gets is the text). Mask seeds: 2 for mcar/block (S5 scheme), mnar_high
is rank-deterministic (1 seed). Prompt tiers per cell:
  notext  (no text channel: fusion off for tPatchGNN, empty domain_des for
           TimeLLM)
  default (the dataset description text used during training)
  generic (default + "some observations are missing, linearly interpolated")
  mech_X  (default + precise mechanism description, X in {mcar, block,
           mnar_high}) -- every cell is evaluated with ALL THREE mechanism
           texts, so matched = diagonal, mismatched = off-diagonal mean.
Metrics per cell: per-(window,channel) mse, level^2 = (mu_pred-mu_gt)^2,
shape_mse (centered; mse == lvl2 + shape_mse exactly), signed level bias
mu_pred - mu_gt; per-window mse for paired tests; relMSE vs clean/default.

Subcommands:
  --smoke    import + forward + text-swap sanity (GPU)
  --repro    official-pipeline repro (ILINet; --repro-model, --repro-dataset)
  --core     one shard: --dataset {ETTh1,weather} --model {timellm,tpatchgnn}
  --merge    scan s20_*.log -> s20_results.json (tables + paired stats)
  --summary  print tables
  --figure   s20.png (English)

Artifacts: this file, s20_results.json, s20.png, s20_notes.md, s20_*.log.
Nothing pre-existing is written; no git.
"""
import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
# hf-mirror.com currently 308-redirects resolve/HEAD to huggingface.co and the
# new huggingface_hub metadata check rejects that ("Distant resource does not
# seem to be on huggingface.co"); direct endpoint works through the proxy.
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
os.environ.setdefault("HF_HOME", os.path.join(HERE, ".hf_cache"))
sys.path.insert(0, os.path.join(HERE, "repos", "IMM-TSF"))

import numpy as np
import pandas as pd
import torch

import run_s5_missing as s5  # reuses load_windows / make_mask / fill_context

L, H = s5.L, s5.H                     # 512, 96
RATES = [0.3, 0.7]
MECHS = ["mcar", "block", "mnar_high"]
SEED_CORE = 1                         # IMM-TSF default seed
TRAIN_STRIDE = 24
RESULTS = os.path.join(HERE, "s20_results.json")

PAPER_ILINET = {  # arXiv:2506.10412v4 App. L Table 9 (test MSE)
    ("tPatchGNN", False): 1.6163,
    ("tPatchGNN", True): 1.4877,
    ("TimeLLM", False): 1.1243,
}

DATASETS = {"ETTh1": s5.DATASETS["ETTh1"], "weather": s5.DATASETS["weather"]}

DES = {
    "ETTh1": ("Hourly measurements from an electricity transformer station "
              "(oil temperature and load features)."),
    "weather": ("Ten-minute measurements from a weather station (temperature, "
                "humidity, pressure, wind, rain and related features)."),
}

GENERIC_NOTE = (" Data quality note: some observations in the past window are "
                "missing and have been filled in by linear interpolation.")
MECH_NOTE = {
    "mcar": (" Data quality note: {pct}% of the time points in the past "
             "window were lost completely at random (scattered isolated "
             "dropouts, each point missing independently) and have been "
             "filled in by linear interpolation."),
    "block": (" Data quality note: the sensors were down for several "
              "continuous outage stretches of about 24 consecutive time "
              "steps each, together covering about {pct}% of the past "
              "window; the missing stretches have been filled in by linear "
              "interpolation."),
    "mnar_high": (" Data quality note: the monitoring system censored and "
                  "deleted high readings (values above roughly the 90th "
                  "percentile of a typical window); about {pct}% of the "
                  "past window's points were removed and filled in by "
                  "linear interpolation, so the observed values "
                  "under-represent the high end."),
}


def tier_text(ds, tier, pct=30, stats_sentence=""):
    """Full prompt/note text for a tier. tier in
    {notext, default, generic, mech_mcar, mech_block, mech_mnar_high}."""
    if tier == "notext":
        return ""
    base = DES[ds] + stats_sentence
    if tier == "default":
        return base
    if tier == "generic":
        return base + GENERIC_NOTE
    mech = tier[len("mech_"):]
    return base + MECH_NOTE[mech].format(pct=pct)


def stats_sentence_from_context(ctx_filled_norm):
    """Per-window stats sentence (keeps the fusion text channel content-
    sensitive during training; computed on the FILLED context at eval --
    deployment-realistic). ctx: [C, L] normalized."""
    return (" Past-window channel values have overall mean "
            f"{ctx_filled_norm.mean():.2f}, std {ctx_filled_norm.std():.2f}, "
            f"min {ctx_filled_norm.min():.2f}, max {ctx_filled_norm.max():.2f} "
            "(z-normalized units).")


# --------------------------------------------------------------------------
# shared small utils
# --------------------------------------------------------------------------

def emit(rec, log):
    line = "RESULT " + json.dumps(rec)
    print(line, flush=True)
    with open(log, "a") as fh:
        fh.write(line + "\n")


def scan_log(path):
    recs = {}
    if not os.path.exists(path):
        return recs
    with open(path) as fh:
        for line in fh:
            if not line.startswith("RESULT "):
                continue
            try:
                r = json.loads(line[7:])
            except Exception:
                continue
            recs[r["key"]] = r
    return recs


def pgain(a_mse, b_mse):
    """Paired gain of a over b on identical windows/masks (positive = a
    better). Same statistic as S19."""
    a = np.asarray(a_mse, np.float64)
    b = np.asarray(b_mse, np.float64)
    d = b - a
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1))
    t = m / (sd / np.sqrt(n)) if sd > 0 else 0.0
    return {"n": n, "gain": m, "mean_ref": float(b.mean()),
            "rel_gain": m / float(b.mean()) if b.mean() else 0.0, "t": t,
            "ci95": 1.96 * sd / np.sqrt(n), "win_rate": float((d > 0).mean())}


def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# IMM-TSF args namespace (fields needed by collate fns and model ctors)
# --------------------------------------------------------------------------

def make_ns(**kw):
    ns = argparse.Namespace(
        device=torch.device(kw.pop("device", "cuda")),
        history=L, pred_window=H, stride=TRAIN_STRIDE,
        patch_size=24, npatch=None, patch_stride=None,
        batch_size=8, dropout=0.1, enable_text=False,
        use_text_embeddings=False, llm_model_fusion="GPT2",
        llm_layers_fusion=6, max_length=256, d_txt=768,
        recency_sigma=1.0, n_heads_fusion=1, kappa=0.5,
        TTF_module="TTF_RecAvg", MMF_module="MMF_GR_Add",
        # model fields
        C=7, hid_dim=32, te_dim=10, node_dim=10, hop=1, tf_layer=1,
        nlayer=1, n_heads=1, outlayer="Linear",
        input_len=L, pred_len=H, use_norm=1, d_ff=128, d_model=32,
        ts_vocab_size=1000, input_token_len=16, top_k=5,
        domain_des="", llm_model_timellm="GPT2", llm_layers_timellm=6,
        time_unit="hours", unit_scale=None,
        data_root="", dataset="", split_method="sample", model="tPatchGNN",
        rec_ids=None,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def build_collate(ns, model_name):
    """Their exact collate fns + multimodal wrapper, on chunk tuples
    (chunk_id, tt, vals, mask, texts)."""
    from lib.parse_datasets import (patch_variable_time_collate_fn,
                                    variable_time_collate_fn)
    base = (patch_variable_time_collate_fn if model_name == "tpatchgnn"
            else variable_time_collate_fn)
    enable_text = ns.enable_text
    use_emb = ns.use_text_embeddings
    device = ns.device
    time_max = torch.tensor(ns.history + ns.pred_window, dtype=torch.float32,
                            device=device)

    from torch.nn.utils.rnn import pad_sequence

    def collate(batch):
        numeric = [(cid, tt.to(device), vals.to(device), msk.to(device))
                   for (cid, tt, vals, msk, *_) in batch]
        out = base(numeric, ns, time_max)
        raws = [item[4] for item in batch]
        time_seqs = [torch.tensor([t for (t, _) in seq], dtype=torch.float32,
                                  device=device) for seq in raws]
        out["tau"] = pad_sequence(time_seqs, batch_first=True,
                                  padding_value=0.0)
        if enable_text and not use_emb:
            out["notes_text"] = [[txt for (_, txt) in seq] for seq in raws]
        return out

    return collate


def make_chunk(vals, note=None):
    """vals: [L+H, C] float32 (context filled, future = ground truth).
    mask = ones (fill-then-feed). note attached at the last context step."""
    tt = torch.arange(L + H, dtype=torch.float32)
    m = torch.ones_like(vals)
    texts = [] if note is None else [(float(L - 1), note)]
    return ("w", tt, vals, m, texts)


def batches(chunks, collate, bs):
    for i in range(0, len(chunks), bs):
        yield collate(chunks[i:i + bs])


# --------------------------------------------------------------------------
# core data: S5 windows + clean train/val split
# --------------------------------------------------------------------------

def core_data(ds):
    """Returns Xn [N, C] (z-normalized by TRAIN region), train/val chunk
    arrays (clean), and S5 test starts."""
    X_raw, starts = s5.load_windows(DATASETS[ds], 300, s5.SEED)
    X_raw = X_raw.astype(np.float32)
    N = len(X_raw)
    n_tr = int(0.6 * N)
    mu = X_raw[:n_tr].mean(axis=0)
    sd = X_raw[:n_tr].std(axis=0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    Xn = ((X_raw - mu) / sd).astype(np.float32)

    def windows(lo, hi, stride):
        out = []
        for s in range(lo, hi - L - H + 1, stride):
            out.append(Xn[s:s + L + H])
        return out

    train = windows(0, n_tr, TRAIN_STRIDE)
    val = windows(n_tr, int(0.8 * N), TRAIN_STRIDE)
    return Xn, train, val, starts


# --------------------------------------------------------------------------
# model construction / training / eval
# --------------------------------------------------------------------------

def build_model(model_name, ns, C):
    if model_name == "timellm":
        from models.TimeLLM import TimeLLM as _TL

        class TimeLLMPerWindow(_TL):
            """TimeLLM with PER-WINDOW domain descriptions (runtime subclass;
            repo files untouched). Set self.domain_des_list = [str] * B before
            forward; falls back to the global self.domain_des otherwise."""

            def _get_prompt(self, x_enc):
                B, Lx, N = x_enc.shape
                mins = x_enc.min(dim=1)[0]
                maxs = x_enc.max(dim=1)[0]
                meds = x_enc.median(dim=1).values
                trend = x_enc.diff(dim=1).sum(dim=1).mean(dim=1)
                FFT = torch.fft.rfft(x_enc.permute(0, 2, 1), dim=-1)
                corr = torch.fft.irfft(FFT * FFT.conj(), n=Lx, dim=-1).mean(dim=1)
                _, lags = corr.topk(min(self.top_k, Lx), dim=-1)
                if lags.size(1) < self.top_k:
                    pad = lags[:, -1, None].repeat(1, self.top_k - lags.size(1))
                    lags = torch.cat([lags, pad], dim=-1)
                des = getattr(self, "domain_des_list", None) or \
                    [self.domain_des] * B
                prompts = []
                for b in range(B):
                    tr = "upward" if trend[b].item() > 0 else "downward"
                    prompts.append(
                        f"<|start_prompt|>"
                        f"Dataset: {des[b]}. "
                        f"Forecast next {self.pred_len} from past "
                        f"{self.input_len}. "
                        f"Min {mins[b].tolist()}, "
                        f"Max {maxs[b].tolist()}, "
                        f"Median {meds[b].tolist()}, "
                        f"Trend {tr}, "
                        f"Top lags {lags[b].tolist()}."
                        f"<|end_prompt|>")
                return prompts

        return TimeLLMPerWindow(ns).to(ns.device), None
    from models.tPatchGNN import tPatchGNN
    from fusions.FusionModel import FusionModel
    model = tPatchGNN(ns).to(ns.device)
    fusion = FusionModel(ns).to(ns.device) if ns.enable_text else None
    return model, fusion


def forward_batch(model, fusion, bd, use_fusion=True):
    # TimeLLM: per-window prompt text rides along as notes_text (one note
    # per window); empty note -> empty domain description (notext tier).
    if hasattr(model, "domain_des") and "notes_text" in bd:
        model.domain_des_list = [lst[0] if lst else ""
                                 for lst in bd["notes_text"]]
    pred = model.forecasting(bd["tp_to_predict"], bd["observed_data"],
                             bd["observed_tp"], bd["observed_mask"])
    if use_fusion and fusion is not None:
        pred = fusion(bd["notes_text"], bd["tau"], bd["tp_to_predict"], pred)
    return pred


def core_note(model_name, ds, tier, pct, filled_ctx):
    """Text attached to one eval/training window, per model's text channel."""
    if model_name == "tpatchgnn":
        return tier_text(ds, tier, pct=pct,
                         stats_sentence=stats_sentence_from_context(filled_ctx))
    # timellm: dataset description + tier sentence (its prompt already carries
    # auto-computed per-window stats)
    return tier_text(ds, tier, pct=pct)


def augment_stage2(train_w, C, model_name, ds, rng):
    """Prompt-conditioned training mix (Stage 2): 40% clean+default note,
    40% missing+truthful mechanism note (mech x rate sampled), 10%
    missing+generic note, 10% clean+FALSE mechanism claim (teaches the model
    not to overreact on clean data -- keeps the H5 anchor meaningful)."""
    chunks = []
    for wi, w in enumerate(train_w):
        ctx = w[:L].T.copy()                     # [C, L] clean context
        fut = w[L:]                              # [H, C] ground truth
        u = rng.random()
        if u < 0.4:
            note = core_note(model_name, ds, "default", 30, ctx)
            chunks.append(make_chunk(torch.from_numpy(w), note))
            continue
        mech = MECHS[int(rng.integers(3))]
        rate = float(rng.choice([0.1, 0.3, 0.5, 0.7]))
        pct = int(round(rate * 100))
        if u < 0.9:  # missing (0.8 = truthful, 0.1 = generic)
            ms = int(rng.integers(100000))
            mask = s5.make_mask(mech, rate, wi, ms, C, x=ctx)
            filled = s5.fill_context(ctx, mask, "linear")
            ww = np.concatenate([filled.T, fut], axis=0).astype(np.float32)
            tier = f"mech_{mech}" if u < 0.8 else "generic"
            note = core_note(model_name, ds, tier, pct, filled)
            chunks.append(make_chunk(torch.from_numpy(ww), note))
        else:  # clean + false claim
            note = core_note(model_name, ds, f"mech_{mech}", pct, ctx)
            chunks.append(make_chunk(torch.from_numpy(w), note))
    return chunks


def train_core(model_name, ds, device, max_epoch=100, patience=3,
               lr=1e-3, wd=0.01, batch_size=8, stage2=False, log=print):
    """Stage 1 (stage2=False): train on CLEAN windows only. Text during
    training: dataset description (+ per-window stats sentence for
    tPatchGNN). Stage 2: prompt-conditioned mix (see augment_stage2).
    TimeLLM patch stride: 8 where it fits, larger for wide-C datasets
    (GPT-2's 1024 positions; repo truncates the prompt at 512 tokens, so
    patch_nums*C + 512 <= 1000; ETTh1 -> 8, weather -> 24)."""
    from utils.tools import set_seed
    set_seed(SEED_CORE)
    torch.cuda.manual_seed(SEED_CORE)
    Xn, train_w, val_w, starts = core_data(ds)
    C = Xn.shape[1]
    ns = make_ns(device=device, C=C, enable_text=True,
                 domain_des=DES[ds], dataset=ds, model=model_name,
                 batch_size=64)
    if model_name == "tpatchgnn":
        ns.patch_stride = ns.patch_size
        ns.npatch = int(np.ceil((ns.history - ns.patch_size) /
                                ns.patch_stride)) + 1
    else:
        # TimeLLM patch stride: prompt is truncated at 512 tokens by the repo
        # code, GPT-2 positions cap at 1024 -> patch_nums*C + 512 must fit.
        for st in (8, 16, 24):
            ns.stride = st
            if ((ns.input_len - 16) // st + 2) * C + 512 <= 1000:
                break
    collate = build_collate(ns, model_name)
    model, fusion = build_model(model_name, ns, C)
    params = list(model.parameters()) + (list(fusion.parameters())
                                         if fusion is not None else [])
    opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    if stage2:
        rng_aug = np.random.default_rng(SEED_CORE + 77)
        train_chunks = augment_stage2(train_w, C, model_name, ds, rng_aug)
        # val: half clean-default, half missing+truthful (deterministic)
        val_chunks = []
        for i, w in enumerate(val_w):
            ctx = w[:L].T.copy()
            if i % 2 == 0:
                note = core_note(model_name, ds, "default", 30, ctx)
                val_chunks.append(make_chunk(torch.from_numpy(w), note))
            else:
                mech = MECHS[i % 3]
                rate = [0.3, 0.5, 0.7][i % 3]
                mask = s5.make_mask(mech, rate, 9000 + i, 0, C, x=ctx)
                filled = s5.fill_context(ctx, mask, "linear")
                ww = np.concatenate([filled.T, w[L:]], axis=0).astype(np.float32)
                note = core_note(model_name, ds, f"mech_{mech}",
                                 int(round(rate * 100)), filled)
                val_chunks.append(make_chunk(torch.from_numpy(ww), note))
    else:
        train_chunks = [make_chunk(torch.from_numpy(w),
                                   core_note(model_name, ds, "default", 30,
                                             w[:L].T))
                        for w in train_w]
        val_chunks = [make_chunk(torch.from_numpy(w),
                                 core_note(model_name, ds, "default", 30,
                                           w[:L].T))
                      for w in val_w]

    use_fusion = fusion is not None

    def loss_on(bd):
        pred = forward_batch(model, fusion, bd, use_fusion)
        err = (pred - bd["data_to_predict"]) ** 2
        m = bd["mask_predicted_data"]
        return (err * m).sum() / m.sum().clamp(min=1.0) / C

    def eval_chunks(chunks, bs=24):
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for bd in batches(chunks, collate, bs):
                pred = forward_batch(model, fusion, bd, use_fusion)
                err = (pred - bd["data_to_predict"]) ** 2
                msk = bd["mask_predicted_data"]
                tot += float((err * msk).sum())
                cnt += int(msk.sum())
        return tot / max(cnt, 1)

    best_val, best_state, no_imp, best_ep = np.inf, None, 0, -1
    rng = np.random.default_rng(SEED_CORE)
    for ep in range(max_epoch):
        model.train()
        if fusion is not None:
            fusion.train()
            if hasattr(fusion.ttf, "llm_model"):
                fusion.ttf.llm_model.eval()  # frozen LLM: no dropout
        order = rng.permutation(len(train_chunks))
        tr_loss, nb = 0.0, 0
        for i in range(0, len(order), batch_size):
            bd = collate([train_chunks[j] for j in order[i:i + batch_size]])
            opt.zero_grad()
            loss = loss_on(bd)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            tr_loss += float(loss)
            nb += 1
        model.eval()
        if fusion is not None:
            fusion.eval()
        v = eval_chunks(val_chunks)
        log(f"  epoch {ep:03d} train_mse={tr_loss / max(nb, 1):.4f} "
            f"val_mse={v:.4f}", )
        if best_val - v > 1e-4:
            best_val, best_ep, no_imp = v, ep, 0
            best_state = {k: t.detach().clone() for k, t in
                          model.state_dict().items()}
            if fusion is not None:
                best_state.update({"fusion." + k: t.detach().clone() for
                                   k, t in fusion.state_dict().items()})
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state is not None:
        ms = {k: v for k, v in best_state.items()
              if not k.startswith("fusion.")}
        model.load_state_dict(ms)
        if fusion is not None:
            fs = {k[len("fusion."):]: v for k, v in best_state.items()
                  if k.startswith("fusion.")}
            fusion.load_state_dict(fs)
    model.eval()
    if fusion is not None:
        fusion.eval()
    log(f"  trained: best epoch {best_ep}, best val mse {best_val:.4f}")
    meta = {"C": C, "n_train": len(train_chunks), "n_val": len(val_chunks),
            "best_val_mse": best_val, "best_epoch": best_ep,
            "starts": starts.tolist()}
    return model, fusion, ns, collate, Xn, meta


@torch.no_grad()
def eval_core_cell(model, fusion, ns, collate, Xn, starts, ds, mech, rate,
                   tier, n_seeds=2, bs=24):
    """One eval cell. Returns per-(w,c) arrays and per-window mse.
    tier == "notext": fusion bypassed for tPatchGNN (true no-text ablation),
    empty domain_des for TimeLLM. On clean cells the mechanism texts use
    pct=30 wording (fixed choice: false-claim anchor stress test)."""
    C = Xn.shape[1]
    det = mech in ("clean", "mnar_high")
    pct = int(round(rate * 100)) if rate > 0 else 30
    use_fusion = fusion is not None and tier != "notext"
    gts, chunks = [], []
    for wi, s in enumerate(starts):
        x_clean = Xn[s:s + L].T.copy()          # [C, L]
        y = Xn[s + L:s + L + H].T.copy()        # [C, H]
        for mseed in ((0,) if det else range(n_seeds)):
            mask = (np.zeros((C, L), bool) if mech == "clean"
                    else s5.make_mask(mech, rate, wi, mseed, C, x=x_clean))
            filled = (x_clean.copy() if mech == "clean"
                      else s5.fill_context(x_clean, mask, "linear"))
            vals = np.concatenate([filled.T, y.T], axis=0)  # [L+H, C]
            note = core_note(ns.model, ds, tier, pct, filled)
            chunks.append(make_chunk(torch.from_numpy(vals), note))
            gts.append(y)
    # TimeLLM global fallback (per-window notes take precedence)
    if hasattr(model, "domain_des"):
        model.domain_des = tier_text(ds, tier, pct=pct)
    out = []
    for bd in batches(chunks, collate, bs):
        pred = forward_batch(model, fusion, bd, use_fusion)
        out.append(pred.float().cpu().numpy())
    p = np.concatenate(out).astype(np.float64)   # [S, H, C]
    gt = np.stack(gts).astype(np.float64)        # [S, C, H]
    p = p.transpose(0, 2, 1)                     # [S, C, H]
    e2 = (p - gt) ** 2
    mu_p, mu_g = p.mean(axis=2), gt.mean(axis=2)
    pc, gc = p - mu_p[:, :, None], gt - mu_g[:, :, None]
    S = p.shape[0]
    nw = len(starts)
    wc = {"mse": e2.mean(axis=2).reshape(-1),
          "lvl2": ((mu_p - mu_g) ** 2).reshape(-1),
          "shape_mse": ((pc - gc) ** 2).mean(axis=2).reshape(-1),
          "bias": (mu_p - mu_g).reshape(-1),
          "mu_gt": mu_g.reshape(-1)}
    mse_w = e2.mean(axis=(1, 2))                  # [S]
    mse_pw = mse_w.reshape(nw, -1).mean(axis=1)
    return {"wc": {k: v.tolist() for k, v in wc.items()},
            "mse": float(e2.mean()),
            "mse_per_window": mse_pw.tolist(),
            "S": S, "n_windows": nw}


# --------------------------------------------------------------------------
# smoke
# --------------------------------------------------------------------------

def run_smoke(args):
    device = args.device
    from fusions.load_llm import load_llm, embed_notes
    tok, llm = load_llm("GPT2", 6, device)
    emb, msk = embed_notes([["sensor note", "another note"]], tok, llm,
                           max_length=64)
    assert emb.shape[0] == 1 and emb.shape[1] == 2
    print("embed_notes OK", tuple(emb.shape), flush=True)

    ns = make_ns(device=device, C=2, input_len=64, pred_len=24,
                 input_token_len=16, stride=8, llm_layers_timellm=2,
                 ts_vocab_size=100, top_k=3, batch_size=4)
    from models.TimeLLM import TimeLLM
    m = TimeLLM(ns).to(device)
    B, L0, H0 = 4, 64, 24
    od = torch.randn(B, L0, 2, device=device)
    om = torch.ones(B, L0, 2, device=device)
    otp = torch.linspace(0, 0.7, L0, device=device).unsqueeze(0).repeat(B, 1)
    ptp = torch.linspace(0.7, 1.0, H0, device=device).unsqueeze(0).repeat(B, 1)
    o1 = m.forecasting(ptp, od, otp, om)
    m.domain_des = "a dataset where high values were censored"
    o2 = m.forecasting(ptp, od, otp, om)
    d = (o1 - o2).abs().max().item()
    print(f"TimeLLM forward OK {tuple(o1.shape)}, prompt-swap max|diff|={d:.3f}",
          flush=True)
    assert d > 0, "prompt has no effect on TimeLLM output"
    del m
    torch.cuda.empty_cache()

    ns2 = make_ns(device=device, C=2, enable_text=True, max_length=256)
    from fusions.FusionModel import FusionModel
    f = FusionModel(ns2).to(device)
    notes = [["clean operation"], ["high values were censored"]]
    tau = torch.tensor([[511.0], [511.0]], device=device)
    th = torch.linspace(0.84, 1.0, 24, device=device).unsqueeze(0).repeat(2, 1)
    Y = torch.randn(2, 24, 2, device=device)
    y1 = f(notes, tau, th, Y)
    y2 = f([["clean operation"]] * 2, tau, th, Y)
    d2 = (y1 - y2).abs().max().item()
    print(f"Fusion forward OK {tuple(y1.shape)}, note-swap max|diff|={d2:.3f}",
          flush=True)
    assert d2 > 0, "note has no effect on fusion output"
    print("SMOKE OK", flush=True)


# --------------------------------------------------------------------------
# repro: official IMM-TSF pipeline on ILINet
# --------------------------------------------------------------------------

def repro_ns(dataset, model_name, enable_text, device):
    """Mirror update_args_for_dataset/model + paper training config."""
    ns = make_ns(device=device, enable_text=enable_text,
                 data_root=os.path.join(HERE, "repos", "Time-IMM", "data"),
                 dataset=dataset, model=model_name)
    # dataset updates (main.py:update_args_for_dataset)
    if dataset == "ILINet":
        ns.history, ns.pred_window, ns.stride, ns.time_unit = 36, 36, 4, "weeks"
    elif dataset == "GDELT":
        ns.history, ns.pred_window, ns.stride, ns.time_unit = 14, 14, 14, "days"
    # model updates (main.py:update_args_for_model)
    if model_name == "tPatchGNN":
        ns.patch_size, ns.n_heads, ns.tf_layer, ns.nlayer = 24, 1, 1, 1
        ns.te_dim, ns.node_dim, ns.hid_dim, ns.outlayer = 10, 10, 32, "Linear"
        ns.npatch = int(np.ceil((ns.history - ns.patch_size) / ns.stride)) + 1
        ns.patch_stride = None  # parse_datasets defaults it to patch_size
    elif model_name == "TimeLLM":
        ns.input_token_len, ns.output_token_len = 16, 96
        ns.d_model, ns.d_ff = 32, 128
        ns.llm_model_timellm, ns.llm_layers_timellm = "GPT2", 6
        ns.n_heads = 2
        ns.domain_des = ("The Electricity Transformer Temperature (ETT) is a "
                         "crucial indicator in the electric power long-term "
                         "deployment.")  # main.py's shipped default
    if enable_text:
        ns.use_text_embeddings = False  # raw text through the frozen LLM
        ns.llm_model_fusion = "GPT2"
        ns.llm_layers_fusion = None     # paper: full frozen stack
        ns.max_length = 256             # deviation: notes are ~100 tokens
    ns.lr, ns.w_decay, ns.batch_size = 1e-3, 0.01, 8
    ns.patience, ns.early_stop_delta, ns.epoch = 3, 1e-4, 300
    return ns


def run_repro(args):
    from utils.tools import set_seed
    from lib.parse_datasets import parse_datasets, get_input_and_pred_len
    from lib.evaluation import compute_all_losses, evaluation
    torch.backends.cuda.matmul.allow_tf32 = True  # deviation: speed only
    torch.backends.cudnn.allow_tf32 = True
    for model_name, enable_text in args.repro_jobs:
        key = f"repro:{args.repro_dataset}:{model_name}:text={int(enable_text)}"
        if key in scan_log(args.log):
            print(f"skip {key} (in log)", flush=True)
            continue
        set_seed(SEED_CORE)
        ns = repro_ns(args.repro_dataset, model_name, enable_text,
                      args.device)
        t0 = time.time()
        data_obj = parse_datasets(ns)
        ns.C = data_obj["input_dim"]
        ns.input_len, ns.pred_len = get_input_and_pred_len(data_obj)
        if model_name == "tPatchGNN":
            from models.tPatchGNN import tPatchGNN as MC
        else:
            from models.TimeLLM import TimeLLM as MC
        model = MC(ns).to(ns.device)
        fusion = None
        if enable_text:
            from fusions.FusionModel import FusionModel
            fusion = FusionModel(ns).to(ns.device)
        params = (list(model.parameters()) +
                  (list(fusion.parameters()) if fusion is not None else []))
        opt = torch.optim.Adam(params, lr=ns.lr, weight_decay=ns.w_decay)
        best_val, best_res, no_imp = np.inf, None, 0
        for ep in range(ns.epoch):
            model.train()
            if fusion is not None:
                fusion.train()
                if hasattr(fusion.ttf, "llm_model"):
                    fusion.ttf.llm_model.eval()
            for bd in data_obj["train_dataloader"]:
                opt.zero_grad()
                res = compute_all_losses(model, fusion, bd, enable_text,
                                         ns.use_text_embeddings)
                res["loss"].backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
            model.eval()
            if fusion is not None:
                fusion.eval()
            with torch.no_grad():
                v = evaluation(model, fusion, data_obj["val_dataloader"],
                               enable_text, ns.use_text_embeddings)
                if best_val - v["mse"] > ns.early_stop_delta:
                    best_val, no_imp = v["mse"], 0
                    best_res = evaluation(model, fusion,
                                          data_obj["test_dataloader"],
                                          enable_text,
                                          ns.use_text_embeddings)
                    be = ep
                else:
                    no_imp += 1
            print(f"  [{key}] epoch {ep:03d} val_mse={v['mse']:.4f} "
                  f"best={best_val:.4f}", flush=True)
            if no_imp >= ns.patience:
                break
        rec = {"key": key, "dataset": args.repro_dataset, "model": model_name,
               "enable_text": enable_text,
               "test_mse": float(best_res["mse"]),
               "test_mae": float(best_res["mae"]),
               "best_epoch": int(be), "val_mse": float(best_val),
               "paper_mse": PAPER_ILINET.get((model_name, enable_text)),
               "seconds": round(time.time() - t0, 1)}
        rec["within_10pct"] = (rec["paper_mse"] is not None and
                               abs(rec["test_mse"] / rec["paper_mse"] - 1)
                               <= 0.10)
        emit(rec, args.log)
        print(f"REPRO {key}: test_mse={rec['test_mse']:.4f} "
              f"paper={rec['paper_mse']} within10%={rec['within_10pct']} "
              f"({rec['seconds']}s)", flush=True)
        del model, fusion, data_obj
        torch.cuda.empty_cache()
    print("REPRO DONE", flush=True)


# --------------------------------------------------------------------------
# core experiment
# --------------------------------------------------------------------------

def cell_key(ds, mech, rate, tier, stage=1):
    return f"core{'' if stage == 1 else str(stage)}:{ds}:{mech}:{rate}:{tier}"


def run_core(args):
    ds, model_name = args.dataset, args.model
    device = args.device
    log = args.log
    stage = args.stage2
    print(f"CORE{2 if stage else ''} {ds}/{model_name}: training ...",
          flush=True)
    t0 = time.time()
    model, fusion, ns, collate, Xn, meta = train_core(
        model_name, ds, device, max_epoch=args.max_epoch, stage2=stage,
        log=lambda *a: print(*a, flush=True))
    starts = np.array(meta["starts"])
    meta = {k: v for k, v in meta.items() if k != "starts"}
    meta.update({"kind": "meta",
                 "key": f"meta{2 if stage else ''}:{ds}:{model_name}",
                 "dataset": ds, "model": model_name, "stage": 2 if stage else 1,
                 "train_seconds": round(time.time() - t0, 1)})
    emit(meta, log)

    done = scan_log(log)
    # text tiers per cell
    clean_tiers = (["notext", "default", "generic"] +
                   [f"mech_{m}" for m in MECHS])
    cell_tiers = clean_tiers  # same list; mech_* texts carry the cell's pct
    jobs = [("clean", 0.0, t) for t in clean_tiers]
    for mech in MECHS:
        for rate in RATES:
            for t in cell_tiers:
                jobs.append((mech, rate, t))
    stg = 2 if stage else 1
    for mech, rate, tier in jobs:
        key = cell_key(ds, mech, rate, tier, stg)
        if key in done:
            continue
        t1 = time.time()
        rec = eval_core_cell(model, fusion, ns, collate, Xn, starts, ds,
                             mech, rate, tier, n_seeds=2)
        rec.update({"kind": "cell", "key": key, "dataset": ds,
                    "model": model_name, "mech": mech, "rate": rate,
                    "tier": tier, "stage": stg,
                    "seconds": round(time.time() - t1, 1)})
        emit(rec, log)
        print(f"  {mech:10s} p={rate:.1f} {tier:14s} mse={rec['mse']:9.4f} "
              f"({rec['seconds']}s)", flush=True)
    print(f"CORE{stg} {ds}/{model_name} DONE", flush=True)


# --------------------------------------------------------------------------
# robustness probes (text-channel sensitivity under stronger training)
# --------------------------------------------------------------------------

@torch.no_grad()
def _probe_sens(model, fusion, ns, collate, Xn, starts, ds, n=100):
    """Same mnar_high p=0.7 cell under default / matched / mismatched text:
    per-(w,c) mse arrays -> paired content sensitivity + bias."""
    out = {}
    for tier in ("default", "mech_mnar_high", "mech_block"):
        r = eval_core_cell(model, fusion, ns, collate, Xn, starts[:n], ds,
                           "mnar_high", 0.7, tier, n_seeds=1)
        out[tier] = r
    a = np.array(out["default"]["wc"]["mse"])
    b = np.array(out["mech_mnar_high"]["wc"]["mse"])
    c = np.array(out["mech_block"]["wc"]["mse"])
    return {"mse_default": out["default"]["mse"],
            "mse_mech_mnar": out["mech_mnar_high"]["mse"],
            "mse_mech_block": out["mech_block"]["mse"],
            "bias_default": float(np.mean(out["default"]["wc"]["bias"])),
            "bias_mech_mnar": float(np.mean(
                out["mech_mnar_high"]["wc"]["bias"])),
            "paired_absdiff_default_vs_mnar": float(np.abs(a - b).mean()),
            "paired_absdiff_block_vs_mnar": float(np.abs(c - b).mean())}


def run_probe(args):
    device = args.device
    log = args.log or os.path.join(HERE, "s20_probe.log")
    done = scan_log(log)
    # probe 1: tpatchgnn fusion-only fine-tune (base frozen), lr 3e-3, 30 ep
    key = "probe:tpatchgnn:ETTh1:fusion_only"
    if key not in done:
        t0 = time.time()
        model, fusion, ns, collate, Xn, meta = train_core(
            "tpatchgnn", "ETTh1", device, max_epoch=60, patience=3,
            stage2=True, log=lambda *a: print(*a, flush=True))
        for p in model.parameters():
            p.requires_grad = False
        for p in fusion.parameters():
            p.requires_grad = True
        opt = torch.optim.Adam(fusion.parameters(), lr=3e-3,
                               weight_decay=0.0)
        rng = np.random.default_rng(SEED_CORE + 5)
        train_w = core_data("ETTh1")[1]
        chunks = augment_stage2(train_w, 7, "tpatchgnn", "ETTh1", rng)
        fusion.train()
        if hasattr(fusion.ttf, "llm_model"):
            fusion.ttf.llm_model.eval()
        model.eval()
        for ep in range(30):
            order = rng.permutation(len(chunks))
            for i in range(0, len(order), 8):
                bd = collate([chunks[j] for j in order[i:i + 8]])
                opt.zero_grad()
                pred = forward_batch(model, fusion, bd, True)
                err = (pred - bd["data_to_predict"]) ** 2
                m = bd["mask_predicted_data"]
                loss = (err * m).sum() / m.sum().clamp(min=1.0) / 7
                loss.backward()
                opt.step()
        fusion.eval()
        rec = {"kind": "probe", "key": key,
               "desc": "stage2 base + fusion-only 30ep lr3e-3; text "
                       "sensitivity on mnar_high p=0.7 (100 windows)",
               **_probe_sens(model, fusion, ns, collate, Xn,
                             np.array(meta["starts"]), "ETTh1"),
               "seconds": round(time.time() - t0, 1)}
        emit(rec, log)
        print(f"PROBE {key}: {rec['paired_absdiff_default_vs_mnar']:.2e}",
              flush=True)
        del model, fusion
        torch.cuda.empty_cache()
    # probe 2: tpatchgnn stage2 with patience=10 (longer joint training)
    key = "probe:tpatchgnn:ETTh1:patience10"
    if key not in done:
        t0 = time.time()
        model, fusion, ns, collate, Xn, meta = train_core(
            "tpatchgnn", "ETTh1", device, max_epoch=80, patience=10,
            stage2=True, log=lambda *a: print(*a, flush=True))
        rec = {"kind": "probe", "key": key,
               "desc": "stage2 joint training, patience=10 (best_ep in meta)",
               "best_epoch": meta["best_epoch"],
               **_probe_sens(model, fusion, ns, collate, Xn,
                             np.array(meta["starts"]), "ETTh1"),
               "seconds": round(time.time() - t0, 1)}
        emit(rec, log)
        print(f"PROBE {key}: {rec['paired_absdiff_default_vs_mnar']:.2e}",
              flush=True)
        del model, fusion
        torch.cuda.empty_cache()
    # probe 3: timellm stage2 with patience=10
    key = "probe:timellm:ETTh1:patience10"
    if key not in done:
        t0 = time.time()
        model, fusion, ns, collate, Xn, meta = train_core(
            "timellm", "ETTh1", device, max_epoch=80, patience=10,
            stage2=True, log=lambda *a: print(*a, flush=True))
        rec = {"kind": "probe", "key": key,
               "desc": "stage2 joint training, patience=10",
               "best_epoch": meta["best_epoch"],
               **_probe_sens(model, fusion, ns, collate, Xn,
                             np.array(meta["starts"]), "ETTh1"),
               "seconds": round(time.time() - t0, 1)}
        emit(rec, log)
        print(f"PROBE {key}: {rec['paired_absdiff_default_vs_mnar']:.2e}",
              flush=True)
        del model
        torch.cuda.empty_cache()
    print("PROBE DONE", flush=True)


# --------------------------------------------------------------------------
# merge / summary / figure
# --------------------------------------------------------------------------

CORE_LOGS = lambda: [os.path.join(HERE, f) for f in os.listdir(HERE)
                     if f.startswith("s20_core") and f.endswith(".log")]


def run_merge(args):
    cells, metas = {}, {}
    for lp in CORE_LOGS():
        for k, r in scan_log(lp).items():
            if r.get("kind") == "cell":
                # key by record FIELDS (the string key is model-less in
                # stage-1 logs and would collide across models)
                cells[(r["model"], r["dataset"], int(r.get("stage", 1)),
                       r["mech"], float(r["rate"]), r["tier"])] = r
            elif r.get("kind") == "meta":
                metas[k] = r
    out = {}
    if os.path.exists(RESULTS):
        out = json.load(open(RESULTS))
    out["meta"] = {
        "track": "S20 mechanism-level text prompts vs missing-data "
                 "degradation (text-conditioned TS models, IMM-TSF repos)",
        "models": {"timellm": "IMM-TSF TimeLLM (frozen GPT2-6L prompt-as-"
                              "prefix; domain_des carries the text)",
                   "tpatchgnn": "IMM-TSF tPatchGNN + FusionModel TTF_RecAvg "
                                "+ MMF_GR_Add (frozen GPT2-6L note encoder)"},
        "data": "ETTh1 + weather; L=512 H=96; S5 windows/seed (300 test "
                "origins, last 20%, seed 20250810); train on first 60% "
                "(clean), val 60-80%; z-norm by train region",
        "cells": "clean + {mcar,block,mnar_high} x p{0.3,0.7}, linear fill, "
                 "observed_mask=ones (fill-then-feed; text is the only "
                 "missingness signal); mask seeds: 2 (mcar/block), "
                 "1 deterministic (mnar_high)",
        "tiers": ["notext", "default", "generic", "mech_mcar", "mech_block",
                  "mech_mnar_high"],
        "training_meta": metas,
    }
    out["cells"] = {"|".join(map(str, k)): v for k, v in cells.items()}
    plog = os.path.join(HERE, "s20_probe.log")
    probes = scan_log(plog) if os.path.exists(plog) else {}
    out["probes"] = {k: v for k, v in probes.items()
                     if v.get("kind") == "probe"}

    analysis = {}
    combos = sorted({(model, ds, stg) for (model, ds, stg, *_)
                     in cells.keys()})
    for (model, ds, stg) in combos:
        cl = cells.get((model, ds, stg, "clean", 0.0, "default"))
        if cl is None:
            continue
        clean_mse = cl["mse"]
        a = {"clean_default_mse": clean_mse, "table": {}, "anchors": {},
             "paired": {}, "bias": {}, "damage_decomp": {},
             "prompt_effect_decomp": {}}
        # main table + anchor
        for (md, d2, s2, mech, rate, tier), r in cells.items():
            if (md, d2, s2) != (model, ds, stg):
                continue
            name = f"{mech}:{rate}:{tier}"
            wc = r["wc"]
            a["table"][name] = {
                "mse": r["mse"], "relMSE": r["mse"] / clean_mse,
                "mean_bias": float(np.mean(wc["bias"])),
                "mean_lvl2": float(np.mean(wc["lvl2"])),
                "mean_shape": float(np.mean(wc["shape_mse"])),
                "S": r["S"]}
            if r["mech"] == "clean":
                a["anchors"][r["tier"]] = {
                    "relMSE": r["mse"] / clean_mse,
                    "paired_vs_default": pgain(r["mse_per_window"],
                                               cl["mse_per_window"])}
        # paired comparisons + decompositions per missing cell
        for mech in MECHS:
            for rate in RATES:
                base = cells.get((model, ds, stg, mech, rate, "default"))
                if base is None:
                    continue
                ckey = f"{mech}:{rate}"
                bw, bb = base["mse_per_window"], base["wc"]
                # damage decomposition vs clean-default (paired on windows
                # only for mnar (same seeds); mcar/block have 2 seeds vs
                # clean 1 -> compare unpooled means there)
                a["damage_decomp"][ckey] = {
                    "d_mse": np.mean(bb["mse"]) - clean_mse,
                    "d_lvl2": np.mean(bb["lvl2"]) - np.mean(cl["wc"]["lvl2"]),
                    "d_shape": np.mean(bb["shape_mse"]) -
                               np.mean(cl["wc"]["shape_mse"]),
                    "bias": float(np.mean(bb["bias"]))}
                for tier in ("generic", "mech_mcar", "mech_block",
                             "mech_mnar_high"):
                    r = cells.get((model, ds, stg, mech, rate, tier))
                    if r is None:
                        continue
                    nm = f"{ckey}:{tier}"
                    a["paired"][nm] = pgain(r["mse_per_window"], bw)
                    a["bias"][nm] = {
                        "mean_bias": float(np.mean(r["wc"]["bias"])),
                        "mean_abs_bias": float(np.mean(np.abs(
                            r["wc"]["bias"]))),
                        "absbias_paired_vs_default": pgain(
                            np.abs(r["wc"]["bias"]), np.abs(bb["bias"]))}
                    # prompt-effect level/shape split (exact identity)
                    dm = np.mean(bb["mse"]) - np.mean(r["wc"]["mse"])
                    dl = np.mean(bb["lvl2"]) - np.mean(r["wc"]["lvl2"])
                    ds_ = np.mean(bb["shape_mse"]) - np.mean(
                        r["wc"]["shape_mse"])
                    a["prompt_effect_decomp"][nm] = {
                        "d_mse": float(dm), "d_lvl2": float(dl),
                        "d_shape": float(ds_),
                        "level_share": float(dl / dm) if abs(dm) > 1e-12
                        else None}
                # matched vs mismatched
                matched = cells.get((model, ds, stg, mech, rate,
                                     f"mech_{mech}"))
                others = [cells.get((model, ds, stg, mech, rate, f"mech_{m}"))
                          for m in MECHS if m != mech]
                if matched is not None and all(o is not None for o in others):
                    mm_w = np.array(matched["mse_per_window"])
                    mmis_w = np.mean([o["mse_per_window"] for o in others],
                                     axis=0)
                    a["paired"][f"{ckey}:matched_vs_mismatched"] = pgain(
                        mm_w, mmis_w)
                    a["bias"][f"{ckey}:matched_vs_mismatched"] = {
                        "absbias_paired": pgain(
                            np.abs(matched["wc"]["bias"]),
                            np.abs(np.mean([o["wc"]["bias"] for o in others],
                                           axis=0)))}
        analysis[f"{ds}:{model}:s{stg}"] = a
    out["analysis"] = analysis
    save_json(out, RESULTS)
    print(f"merged {len(cells)} cells -> {RESULTS}", flush=True)


def run_summary(args):
    out = json.load(open(RESULTS))
    for dk, a in sorted(out.get("analysis", {}).items()):
        print(f"\n===== {dk} =====  clean-default mse="
              f"{a['clean_default_mse']:.4f}")
        print("  anchor (clean windows, relMSE vs default):")
        for t, v in sorted(a["anchors"].items()):
            print(f"    {t:14s} relMSE={v['relMSE']:.4f} "
                  f"(paired t={v['paired_vs_default']['t']:+.2f})")
        print("  main table (relMSE vs clean-default):")
        for name, v in sorted(a["table"].items()):
            print(f"    {name:34s} relMSE={v['relMSE']:7.4f} "
                  f"bias={v['mean_bias']:+8.4f}")
        print("  paired tier effects (vs default prompt, same windows):")
        for name, v in sorted(a["paired"].items()):
            print(f"    {name:44s} rel_gain={v['rel_gain'] * 100:+7.3f}% "
                  f"t={v['t']:+6.2f} win={v['win_rate']:.3f}")
        print("  signed level bias / matched-vs-mismatch:")
        for name, v in sorted(a["bias"].items()):
            if "absbias_paired_vs_default" in v:
                p = v["absbias_paired_vs_default"]
                print(f"    {name:40s} bias={v['mean_bias']:+8.4f} "
                      f"|bias| gain vs default {p['rel_gain'] * 100:+.2f}% "
                      f"(t={p['t']:+.2f})")
            else:
                p = v["absbias_paired"]
                print(f"    {name:40s} |bias| matched-vs-mismatched gain "
                      f"{p['rel_gain'] * 100:+.2f}% (t={p['t']:+.2f})")


def run_figure(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = json.load(open(RESULTS))
    an = out["analysis"]
    combos = sorted(an.keys())
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))

    # (a) main grid heatmap-style bars: relMSE for mnar_high p=0.7 per tier
    ax = axes[0][0]
    tiers = ["notext", "default", "generic", "mech_mcar", "mech_block",
             "mech_mnar_high"]
    x = np.arange(len(tiers))
    width = 0.8 / max(len(combos), 1)
    for i, dk in enumerate(combos):
        tab = an[dk]["table"]
        ys = [tab.get(f"mnar_high:0.7:{t}", {}).get("relMSE", np.nan)
              for t in tiers]
        ax.bar(x + (i - (len(combos) - 1) / 2) * width, ys, width, label=dk)
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("mech_", "m:") for t in tiers], fontsize=8)
    ax.set_ylabel("relMSE (vs clean default prompt)")
    ax.set_title("mnar_high p=0.7: relMSE by prompt tier")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    # (b) signed level bias on mnar_high p=0.7
    ax = axes[0][1]
    for i, dk in enumerate(combos):
        tab = an[dk]["table"]
        ys = [tab.get(f"mnar_high:0.7:{t}", {}).get("mean_bias", np.nan)
              for t in tiers]
        ax.bar(x + (i - (len(combos) - 1) / 2) * width, ys, width, label=dk)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("mech_", "m:") for t in tiers], fontsize=8)
    ax.set_ylabel("signed level bias (mu_pred - mu_gt)")
    ax.set_title("mnar_high p=0.7: does mechanism text lift the low bias?")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=7)

    # (c) all mechanisms x rates, default vs matched mechanism text
    ax = axes[1][0]
    cells_x = [(m, r) for m in MECHS for r in RATES]
    xx = np.arange(len(cells_x))
    for i, dk in enumerate(combos):
        tab = an[dk]["table"]
        y0 = [tab.get(f"{m}:{r}:default", {}).get("relMSE", np.nan)
              for m, r in cells_x]
        y1 = [tab.get(f"{m}:{r}:mech_{m}", {}).get("relMSE", np.nan)
              for m, r in cells_x]
        ax.plot(xx, y0, "o--", color=f"C{i}", alpha=0.55,
                label=f"{dk} default")
        ax.plot(xx, y1, "s-", color=f"C{i}", label=f"{dk} matched")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xticks(xx)
    ax.set_xticklabels([f"{m}\np={r}" for m, r in cells_x], fontsize=8)
    ax.set_ylabel("relMSE (vs clean default prompt)")
    ax.set_title("default vs matched mechanism text across cells")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6.5, ncol=2)

    # (d) matched vs mismatched paired rel gain, mnar_high cells only
    ax = axes[1][1]
    labels, vals, errs, colors = [], [], [], []
    for i, dk in enumerate(combos):
        for r in RATES:
            v = an[dk]["paired"].get(f"mnar_high:{r}:matched_vs_mismatched")
            if v:
                ddp, mmp, ssp = dk.split(":")
                labels.append(f"{ddp}:{mmp[:2]}:{ssp}\np={r}")
                vals.append(v["rel_gain"] * 100)
                errs.append(v["ci95"] / max(v["mean_ref"], 1e-12) * 100)
                colors.append(f"C{i}")
    ax.bar(np.arange(len(vals)), vals, yerr=errs, color=colors, capsize=2)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("paired rel gain of matched vs mismatched (%)")
    ax.set_title("necessity control on mnar_high: matched vs mismatched text")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("S20: mechanism-level text prompts under missing data "
                 "(IMM-TSF models, ETTh1 + weather)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.fig_out, dpi=140)
    print(f"figure -> {args.fig_out}", flush=True)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--repro", action="store_true")
    ap.add_argument("--repro-dataset", default="ILINet")
    ap.add_argument("--repro-jobs", default="tpatchgnn_uni,tpatchgnn_multi,timellm_uni")
    ap.add_argument("--core", action="store_true")
    ap.add_argument("--stage2", action="store_true",
                    help="prompt-conditioned training arm (text-annotated "
                         "synthetic missingness in training)")
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--model", default="timellm",
                    choices=["timellm", "tpatchgnn"])
    ap.add_argument("--max-epoch", type=int, default=100)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="robustness probes (fusion-only / patience-10)")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log", default=None)
    ap.add_argument("--fig-out", default=os.path.join(HERE, "s20.png"))
    args = ap.parse_args()

    torch.manual_seed(SEED_CORE)
    np.random.seed(SEED_CORE)

    if args.smoke:
        run_smoke(args)
        return
    if args.repro:
        jobs = []
        for j in args.repro_jobs.split(","):
            jobs.append({"tpatchgnn_uni": ("tPatchGNN", False),
                         "tpatchgnn_multi": ("tPatchGNN", True),
                         "timellm_uni": ("TimeLLM", False)}[j])
        args.repro_jobs = jobs
        if args.log is None:
            args.log = os.path.join(HERE, "s20_repro.log")
        run_repro(args)
        return
    if args.core:
        if args.log is None:
            tag = "s20_core2" if args.stage2 else "s20_core"
            args.log = os.path.join(
                HERE, f"{tag}_{args.model}_{args.dataset}.log")
        run_core(args)
        return
    if args.merge:
        run_merge(args)
        return
    if args.probe:
        run_probe(args)
        return
    if args.summary:
        run_summary(args)
        return
    if args.figure:
        run_figure(args)
        return
    raise SystemExit("nothing to do")


if __name__ == "__main__":
    main()
