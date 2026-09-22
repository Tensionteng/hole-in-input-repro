# S71 — the boundary, measured: CPT attempts on the non-restorable conventions

Pre-registered design: `DESIGN.md` (same directory). This file is the living record:
recon, conventions, provenance, gates, deviations, and the per-model verdict table.

## 1. Checkpoint provenance (recon, 2026-09-03)

- **TimesFM** — the paper's "TimesFM 2.5" row is `timesfm.TimesFM_2p5_200M_torch`
  loaded as `from_pretrained("google/timesfm-2.5-200m-pytorch")` (s30_crossmodel's
  `TimesFM` wrapper, reused by `eval_s57_leaderboard.py`). NOT in `models_local`;
  s57 fetched it through the HF mirror into the project cache:
  `.hf_cache/hub/models--google--timesfm-2.5-200m-pytorch` (snapshot with
  `model.safetensors`, present and reused here, same env:
  `HF_ENDPOINT=https://hf-mirror.com`, `HF_HOME=<root>/.hf_cache`).
  Eval config (s30/s57): `ForecastConfig(max_context=1024, max_horizon=128,
  per_core_batch_size=256, normalize_inputs=True, use_continuous_quantile_head=True)`
  — plus the dataclass DEFAULTS the s30 call inherits: `force_flip_invariance=True`
  (decode is run twice, (f(x)-f(-x))/2) and `infer_is_positive=True` (output clamped
  >= 0 when all inputs >= 0). These are eval-time post-processing around the raw
  decode; S71 evals keep them exactly (s57 comparability), and S71 CPT trains the
  raw decode without them (they are wrapper tail behaviour, not model weights; the
  G2t plumbing gate verifies the training replica against module.decode() exactly,
  max rel diff 0.0). Documented here as a convention note, not a redesign.
  Scored quantity: the wrapper's POINT forecast (`full_forecast[..., 5]`).
- **Timer-XL** — the paper's "Timer-XL" row is **`models_local/timer-base-84m`**, i.e.
  the HF `thuml/timer-base-84m` snapshot (Timer base, 84M params: 8 layers, hidden
  1024, input/output patch 96). Evidence chain: `eval_s57_leaderboard.py` REGISTRY
  maps `timerxl` / `timerxl-plain` to `run_s49_2025.TimerXL`, whose
  `name = "timer-base-84m"` and which loads that local directory; there is no other
  timer checkpoint anywhere in the project. So the paper sentence's "Timer-XL" refers
  to thuml's Timer base-84M checkpoint as evaluated in s49/s57. S71 keeps the label
  `timerxl` for continuity and trains/evals exactly this checkpoint.
  transformers-5.x shims needed (from s49): `DynamicCache.seen_tokens` /
  `get_usable_length` properties, plus rebuilding the non-persistent rope
  `inv_freq`/cos/sin buffers after load (bf16 ckpt + meta-device load leaves them
  uninitialised).
- **TempoPFN** — `models_local/tempopfn` (full HF snapshot of `AutoML-org/TempoPFN`,
  sha f88ff0ab), checkpoint `models/checkpoint_38M.pth`, model built from
  `configs/example.yaml` `TimeSeriesModel` kwargs (loss_type='quantile',
  quantiles 0.1..0.9). bf16 autocast is the ONLY supported mode (fla's
  `chunk_gated_delta_product` rejects fp32 activations; official quick start runs
  bf16 autocast with fp32 weights). Requires `PYTHONPATH=models_local/tempopfn`
  and `TRITON_CACHE_DIR` set. TempoPFN has NO s57 leaderboard row (it entered the
  census later, in s62); its stock reference is `s62_results.json`.

## 2. Per-model feeding conventions (as evaluated in s30/s57/s62; kept identical in S71)

- **TimesFM (overwriting).** convs: `plain` = filled values fed as-is; `nan`
  (declared) = NaN at missing positions. The wrapper overwrites EVERY NaN with its own
  linear interpolation (`timesfm_2p5_base.forecast`: `linear_interpolation(
  strip_leading_nans(...))`) before the torch module ever runs; the module's mask
  channel is used for left-padding ONLY in the shipped path (with L=512=16x32 there is
  no padding either; eval pads front to max_context=1024 with mask=True on the pad).
  Consequence: declared-path forecast is provably fill-invariant (s57: zero=linear=
  oracle in all 108 cells; s30: rho(nan)=0.0, rho(plain)=0.863).
- **TempoPFN (learned token).** convs: `plain` / `nan` (declared). Declared: NaN at
  missing -> `nan_embedding` REPLACES the value embedding (`model.py::
  _compute_embeddings`); `RobustScaler` stats from finite points only. No input slot
  both declares a point missing and carries a fill value; s62 verified perm=redraw=0
  exactly on the declared path.
- **Timer-XL (no declared path).** convs: `plain` only — the `nan` convention emits
  NaN logits (s57_timerxl.json is NaN in all 108 cells). Eval convention (s49/s57):
  context cropped to the last 480 (multiple of patch 96), `forward(input_ids)`,
  `logits` = last-token next-96 forecast, first 64 taken. revin NOT applied (s49
  convention). The paper's leaderboard row is `timerxl-plain`.

## 3. CPT recipe mapping (the s53/P5 mechdiv recipe through each native path)

Shared recipe (identical to the bolt retrofit, train_s53.py stage 2 / train_s45.py
`augment`): corpus = `s45_pretrain/corpus_cache` via `t45.Sampler` (85:15
real:KernelSynth, per-subject caps, held-out val tails); windows L=512 ctx + H=64 fut,
fully-observed horizons, clean contexts. Per window: 50% clean; else mech in
{mcar, block, mnar_high, mnar_extreme}, rate ~ U(0.05, 0.7); then 50% "declared",
50% alpha-blend fill (fill = (1-a)*linear + a*truth, a ~ U(0,1)). 5,000 steps,
AdamW lr 1e-4 wd 0.01, 500-step warmup, cosine to 0, grad clip 1.0, non-finite
skip. Batch sizes recorded per run below.

Native-path mapping of the two missing halves (DESIGN: "fills presented through the
model's NATIVE path ... NO new input channels"):

- **TimesFM**: the shipped missingness path has no slot for fill content at all —
  the wrapper replaces declared points with its OWN linear interpolant. So the
  recipe's "declared" half is presented as **linear fill, no flag** (the overwrite
  convention made literal: at declared positions the content IS the model's own
  interpolant), and the alpha-blend half as plain values. All windows enter the
  module with `mask=padding-only`, exactly replicating the eval preprocessing stack
  (front-pad to 1024 + `normalize_inputs` outer revin + decode's causal running-stats
  revin). Loss: pinball over channels 1..9 (quantiles 0.1..0.9) of the last-patch
  64-step prefill output vs the true future (channel 5 = the median = the evaluated
  point forecast is trained as one of the nine). Deviation note: the mechdiv recipe's
  "declared = zeroed content + flag" half has no TimesFM-native realisation (the
  module's mask=True path is never exercised by the shipped forecast API); the
  linear-fill mapping above is the native exposed behaviour, documented here.
- **TempoPFN**: declared half = NaN at missing positions (the native declared path,
  exactly as evaluated); filled half = alpha-blend plain values. Loss: the model's own
  `compute_loss` (native quantile loss, scaled space), container conventions identical
  to s62 (fixed start 2017-01-01, Frequency.H, history [B,512,1], future [B,64,1]).
- **Timer-XL**: no declared path exists; the recipe's "declared" half is presented
  as **content 0 at missing positions after normalisation, no flag** (the
  values-only realisation; TimesFM-module convention: norm-then-zero), filled half =
  alpha-blend values. CPT runs in the model's NATIVE normalised regime: inputs
  normalised by observed-context stats. Measured blocking evidence for the raw
  (revin=False, s57-eval) regime at TRAINING time: stock outputs saturate at ~+-10
  for any raw input scale, and the raw backward overflows (grad-norm NaN on every
  batch containing a window with |x|max >~ 1e5 -- most of the corpus; forward stays
  finite). The raw forward is an inference-harness convention, not a trainable
  regime; eval keeps the s57 convention unchanged. Loss = native next-patch MSE in
  normalised space, Huberised at delta=20 (raw MSE's gradient is dominated by corpus
  windows whose future leaves the context range by >1000 sd -- measured batch losses
  up to 4e6 with ~50% of batches tail-dominated; delta=20 keeps the objective
  quadratic in the bulk and bounds any single window's pull), all positions, token t
  predicts patch t+1 (the eval's alignment), elementwise target masking for
  fabricated/pad target positions. Context cropped to last 480 as in eval.

Eval (eval_s71.py) per model per arm: probe (s30 permutation rho, 3 ds x 3 mechs,
60 win, rate 0.3, linear fill, each model's convs), sweep (alpha in {0,.25,.5,.75,1},
block+mcar at rate 0.7, ETTh1/ETTm1/weather, 150 win, relMSE vs the arm's own clean;
declared views that are provably alpha-invariant — timesfm-nan, tempopfn-nan — are
computed once and copied, s62 convention), clean (same 3 ds), leaderboard row (the s57
grid: 9 ds x 4 mechs x rate 0.7 x {zero, linear, oracle}, declared path — timerxl on
plain, as in the paper).

## 4. Gates

G0 definition (DESIGN): reproduce each model's stored stock cells within 5% before
any training. References: TimesFM -> `s57_timesfm.json` (108 grid cells + 9 clean
MSEs); Timer-XL -> `s57_timerxl_plain.json`; TempoPFN -> `s62_results.json` probe +
sweep cells at s62's own protocol (60 win, alphas {0,.5,1}, rates {0.3,0.7} --
deviation from DESIGN's "s57 leaderboard" wording: tempopfn has no s57 row; s62 is
its stock reference of record, and the 150-win arm sweep is not a substitute for
60-win reproduction).

- **G0 timer PASS (2026-09-03)**: stock arm vs `s57_timerxl_plain.json`, 117 cells
  (108 grid + 9 clean), worst rel dev 1.5e-07 -- bit-level reproduction of the s57
  row. (`s71_timer.json:g0`)
- **G0 timesfm PASS (2026-09-03)**: stock arm vs `s57_timesfm.json`, 117 cells,
  worst rel dev 0.0 (exact). (`s71_timesfm.json:g0`)
- **G0 tempopfn PASS (2026-09-03)**: stock arm vs `s62_results.json` at s62's own
  protocol, 162 cells (probe rho + sweep median-of-ratios), worst 0.0 (exact).
  (`s71_tempopfn.json:g0`)

All three models passed G0 -> all three get CPT attempts (no drops).
- **G1**: CPT train loss must decrease; a flat/diverged run is a plumbing bug.

Smoke tests (2026-09-03, 40-step runs, card-local): plumbing gates exact for all
three models (G2t 0.0, G2p 0.0, G2r 0.0). Steady-state step times: timesfm ~0.1 s
@bs32 (run bs 256), tempopfn 1.17 s @bs128 (run bs 128), timer ~0.03 s @bs64 (run
bs 512). TempoPFN's first-call triton compile is ~70 s, amortised. Timer smoke
history: raw-regime overflow (grad NaN) -> native normalised regime -> Huber-20, all
documented in sec. 3. n_skip: 0 (tempopfn, timer), 0 (timesfm) in final smokes.

## 5. Run log / verdict table

Run map (2026-09-03 evening -> 2026-09-04 early morning; scheduler =
`s71_scheduler.py`, queue files `s71_queue_*.txt`, state `s71_scheduler_state.json`):
- timer: cpt/ctl5k/cpt@20260904 trained (5000 steps, bs 512; 5.5-7.7 min each) and
  fully evaluated. `s71_timer.json`, `s71_ckpt/timer_*.pt`.
- timesfm: cpt/ctl5k/cpt@20260904 trained (bs 256, ~27 min each) and fully
  evaluated. `s71_timesfm.json`, `s71_ckpt/timesfm_*.pt`.
- tempopfn: cpt/ctl5k/cpt@20260904 trained (bs 128, 105-113 min each) and fully
  evaluated. `s71_tempopfn.json`, `s71_ckpt/tempopfn_*.pt`.
All runs: n_skip 0-1% (timer cpt 51/5000, all others 0), G1 loss trends and the
fixed-panel substance check in `s71_g1_check.json`.

Structural confirmations:
- timer nan path emits 100% NaN logits in ALL arms (stock/cpt/ctl5k) -- measured on
  the S71 checkpoints directly; no declared path exists before or after CPT.
- timesfm declared (nan) path is provably fill-invariant (wrapper interpolates
  before the module runs; s57 stock row has zero=linear=oracle in all 108 cells and
  S71's G0 reproduced that exactly). Post-CPT too: the cpt/cpt@20260904 leaderboard
  rows have zero=linear=oracle in all 108 cells (`s71_timesfm.json`).
- tempopfn declared (nan) path has perm=redraw=0 exactly (s62; reproduced at G0).

## 5a. Verdicts (Q1/Q2/Q3 per DESIGN) -- all three models final

Reference column (no new bolt training): bolt-base P5 CPT rho_declared 0.832
(s56_eval_base.json), bolt-tiny P5 closure 76.7% (s53_notes.md); P4 budget control
rho 0.000, closure -19.2%.

**TimesFM (overwriting)** -- Q1 CONFIRMED, the cleanest case:
- probe rho on the declared (nan) path: 0.000 in every arm (stock, cpt, ctl5k,
  cpt@20260904), all 9 cells each -- exactly 0.0 as the input path is overwritten
  before the weights are consulted.
- alpha-sweep closure on the declared path: 0.000 (input provably alpha-invariant;
  computed once per cell, copied, noted in the JSON).
- plain path keeps reading content (rho 0.862 stock -> 0.991/0.928 cpt seeds): CPT
  did not damage the content channel; it simply cannot declare missingness.
- leaderboard: fill-invariant in every arm (zero=linear=oracle, all 108 cells).
  Level: median grid relMSE 2.02 (stock) -> 1.36 (cpt) / 1.66 (ctl5k) / 1.50 (s2);
  clean cost on the 3 probe datasets: +17% (cpt), +16% (ctl5k), +4% (s2) -- CPT
  shifts the level a little, the invariance never moves.
- Q2: ctl5k rho 0.000 = stock 0.000. Holds.
- Q3: not triggered. The boundary sentence stands for TimesFM.

**Timer-XL (no declared path)** -- Q1 confirmed for the interface question, with
the no-interface caveat spelled out:
- there is no declared path to probe: nan input -> 100% NaN logits in every arm
  (measured on the S71 checkpoints). CPT cannot create the channel, and did not.
- plain path rho (the only path; the trivial no-interface baseline of s57): 0.843
  stock -> 0.784 (cpt) / 0.822 (ctl5k) / 0.791 (cpt@20260904). Q2's "within 0.05"
  holds approximately (|drift| <= 0.06 on the median; per-cell in the JSONs).
- sweep on the plain path: closure trivially 100% on all arms (a=1 == clean input).
- leaderboard: oracle/linear ~1.00 everywhere (trivial no-interface); zero column
  median 1.39 stock -> 1.09 cpt / 1.07 ctl5k -- the same shift in both trained arms,
  i.e. general drift, not missingness-specific learning (see traffic: cpt clean MSE
  is 53x stock's; own-clean-normalised ratios move accordingly).
- clean cost at the s57 convention: +217% (cpt) / +345% (ctl5k) / +159% (s2) on the
  3-ds probe clean median -- the CPT runs in the model's native normalised regime
  (sec. 3) and the raw-regime eval function drifts. Both arms drift the same way,
  which is why the arm-to-arm comparison stays meaningful.
- G1 substance (fixed-panel native loss, median): cpt improves clean 0.459->0.275
  and missing-window loss 0.542->0.441 (declared) / 0.536->0.420 (filled); ctl5k
  matches most of that gain (0.447 / 0.463) -- the missingness-specific component is
  small. Training worked; no interface emerged.
- Q3: not triggered (no declared path; plain-path rho unchanged within noise).
  The boundary sentence stands for Timer-XL.

**TempoPFN (learned token)** -- Q1 CONFIRMED:
- probe rho on the declared (nan) path: exactly 0.0000 in all 9 cells of every arm
  (stock, cpt, ctl5k, cpt@20260904) -- the nan_embedding replacement leaves no
  channel for fill content, and 5k steps of the recipe do not create one.
- alpha-sweep closure on the declared path: 0.000 (input provably alpha-invariant;
  computed once per cell, copied, noted in the JSON).
- plain path: ETT cells unchanged (ETTh1 mcar 0.970->0.975 cpt; ETTm1 block
  0.901->0.887; ETTh1 mnar_high 0.370->0.346); weather cells move within their
  known tail-dominated noise (s62 caveat: weather/mcar stock 0.087 -> 0.076,
  weather/mnar_high 1.956 -> 0.526; these means are p99-tail artefacts, medians
  unchanged). No gain anywhere.
- leaderboard: fill-invariant in every arm (zero=linear=oracle, all 108 cells);
  median grid relMSE 1.48 (stock) -> 1.32 (cpt) / 1.48 (ctl5k) / 1.30 (s2).
- clean cost on the 3 probe datasets: +5.1% (cpt), +3.2% (ctl5k), +9.2% (s2).
- G1 substance (fixed panel, per-window median native quantile loss): clean
  0.140 -> 0.127 (cpt) / 0.119 (ctl5k); declared 0.228 -> 0.177 (cpt) vs 0.207
  (ctl5k) -- the cpt arm DID learn to forecast better given the declaration (the
  augmentation half of the recipe works on it, as s62 already showed for stock);
  what it did not learn is to read the fill's content (declared rho stays 0.0000).
- Q2: ctl5k declared rho 0.000 = stock. Holds.
- Q3: not triggered. The boundary sentence stands for TempoPFN.

## 5b. Summary verdict table

| model | path probed | rho stock -> cpt (s2) | closure stock -> cpt | Q1 (no fill-reading gained) | Q2 (ctl5k ~ stock) | Q3 falsified? |
|---|---|---|---|---|---|---|
| TimesFM | declared (nan) | 0.000 -> 0.000 (0.000) | 0% -> 0% | CONFIRMED | holds (0.000 vs 0.000) | no |
| TempoPFN | declared (nan) | 0.000 -> 0.000 (0.000) | 0% -> 0% | CONFIRMED | holds | no |
| Timer-XL | none exists (nan -> 100% NaN logits in all arms) | -- | -- | CONFIRMED (no interface post-CPT) | plain rho 0.822 vs 0.843 | no |

Reference: bolt's retrofit under the same recipe reads fills (P5-base rho_declared
0.832; P5-tiny closure 76.7%; P4 budget control rho 0.000 / closure -19.2%) -- the
contrast the paper's Section 8 claim rests on, now measured on the boundary models.

Plain-path (non-native, information only): timesfm 0.862 -> 0.991 (s2 0.928);
tempopfn ETT cells unchanged; timer 0.843 -> 0.784 (s2 0.791). No arm gains
content-reading where it did not exist; TimesFM's plain path even strengthens
slightly while its declared path stays exactly 0 -- the overwrite is in the input
pipeline, unreachable by weight updates.

## 5c. Files

- `DESIGN.md` (pre-registered spec), `s71_notes.md` (this file)
- `s71_common.py` (wrappers + training forwards + recipe mapping),
  `run_s71_{timesfm,tempopfn,timer}.py` (CPT), `eval_s71.py` (G0 + probe + sweep +
  clean + leaderboard), `s71_g1_check.py` (fixed-panel G1), `analyze_s71.py`
  (verdict extraction), `s71_scheduler.py` + `s71_queue_*.txt` (GPU idle-poller)
- `s71_{timesfm,tempopfn,timer}.json` (all cells), `s71_verdicts.json`,
  `s71_g1_check.json`, `s71_ckpt/*.pt` + `*_log.json`, `s71_*.log`

## 5d. Incidents during the run (all recovered; noted for provenance)

- First timer training pair was SIGPIPE-killed by a logging-redirect bug in the
  scheduler (`&&`-chain redirect covered only the last stage); relaunched cleanly.
- First timesfm cpt/ctl5k trainings were killed when the launching background task
  hit its timeout (process-group kill); relaunched with a no-timeout scheduler.
- Two timesfm arm evals OOM'd on a busy card because the env pin applied only to
  the first stage of the `&&` chain (evals then shared card 0 with tempopfn
  training); checkpoints were intact, evals rerun. One ctl5k eval additionally hit
  a save race on the shared JSON (both writers used the same .tmp path); fixed with
  PID-unique tmp paths + a per-model flock in eval_s71.py, cells verified complete
  by arm afterwards (18/60/3/108 per arm for timesfm/tempopfn, 9/30/3/108 timer).
- tempopfn s2 training shares card 6 with a small (4.4 GiB) job from another
  experiment for part of its run -- the scheduler took the card in a moment the
  other job dipped under 2 GiB; both fit comfortably (30.6 + 4.4 GiB of 80 GiB),
  no interference.

