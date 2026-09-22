# S72 — Root cause of the S70B full-outage explosion (Chronos-2 CPT retrofit)

Date: 2026-09-10. Inference only; no training. Work confined to
`tsfm_missing/experiments/s72_rootcause/`. Strict fp32 (TF32 off), same
conventions as s60/s65/s70/s70b. GPU discipline: ran entirely on card 3
(0 MiB at `nvidia-smi` poll; cards 0/1/2/4 busy with other tenants).

Question (from s70b_notes.md): at (block, 1.0) — ALL L=512 context positions
declared missing through the restored (dual) endpoint — the CPT checkpoint
`arm_CPT_s20260903.pt` explodes on oracle fills (median relMSE ~3.4e9,
|y| ~ 1e4–1e6 on O(10) data) and degenerates to a near-zero constant on
zero/linear fills (relMSE 8.35), while vanilla and ctl5k stay finite on
identical inputs. Candidates: H1 normalization degeneracy, H2 attention
collapse, H3 CPT weights in a bad region for this never-trained corner.

## Setup

`run_s72_rootcause.py` reuses the S70/S70B pipeline (`run_s70_rcprobe`,
`run_s70b_rcprobe`, `c2_iface`) and adds: (a) a corner ablation ladder,
(b) an instrumented forward — forward hooks on every encoder sublayer plus a
monkeypatched `MHA._eager_attention` recording pre-softmax logits and softmax
entropy — and (c) an encoder/head cross-swap between checkpoints. The traced
(eager) forward matches the production (sdpa) forward to ≤ 3e-6 relative on
every config (`trace.fidelity_rel_dev`); the c2_iface native replica is
gate-verified bit-exact vs stock, and the replica clean base matches the
pipeline base to 2.7e-6 (`reproduce.replica_base_vs_pipeline_base_maxdev`).

## 1. Reproduction (tie-out vs s70b_rcprobe.json) — PASS, bit-stable

ETTh1 ch0, 150 windows, the exact grouped S70B code path
(`fc_dual`, target + 4 declared neighbours). 9/9 cells match stored medians
at +0.000%:

| cell (block 1.0, clean ctx) | vanilla | ctl5k | cpt |
|---|---|---|---|
| zero (=linear) declared | 4.351 | 4.804 | 4.332 |
| oracle declared | 0.9725 | 2.811 | **4.277e9** (max\|y\| 1.08e6, finite) |

## 2. The key structural fact: at the corner the network output is a constant

Recovering the normalized (pre-unscaling) output v = asinh((y − loc)/scale)
from the median-quantile forecasts:

- **window-independent**: dispersion of v across the 150 ETTh1-ch0 windows is
  1.7e-8 (relative) for the oracle fill, exactly 0 for the zeros fill
  (`reproduce.v_collapse`).
- **content-independent**: median v(oracle) vs v(zeros) differ by 4.1e-9
  relative — one and the same vector.
- **endpoint-independent**: restored (dual) vs native endpoints give identical
  v (reldiff 0.0) and identical relMSE (2.24e9 both, ladder ch0).
- **covariate-independent**: grouped corner (5 declared rows) vs univariate v
  agree to 2.3e-7 (`trace.grouped_corner_test`) — mechanistically explains
  S70B's clean≡corrupt coincidence.

Why: with all-missing flags, `attention_mask` removes all 32 context patches
from the key set; the [REG] token and the 4 future patches attend only to each
other, and their inputs carry no content (time encodings + zeros). Group
attention at fully-masked context positions goes exactly uniform (measured
entropy = log 8 = 2.0794 with batch 8) but only recirculates content among
context tokens, which are never read out; the FFN is position-wise. So the
forecast reduces to **y = sinh(v)·σ + μ**, where v is a constant vector of the
(weights × flag pattern) and the data enter only through the instance-norm
statistics (μ, σ) of the filled context.

## 3. Ablation ladder at the corner (univariate, ETTh1, 32 windows)

median relMSE (ch0; ch1/ch2 in `ladder.cells`, same pattern):

| content | flag | endpoint | vanilla | ctl5k | cpt | w50 |
|---|---|---|---|---|---|---|
| oracle | all-missing | dual | 1.442 | 1.544 | **2.24e9** | 2.63e7 |
| oracle | all-missing | native | 1.442 | 1.544 | **2.24e9** | 2.63e7 |
| oracle | all-observed | dual/native | 1.0 | 1.0 | 1.0 | 1.0 |
| zeros | all-missing | dual/native | 2.524 | 2.749 | 2.519 | 2.500 |
| zeros | all-observed | dual/native | 2.524 | 2.749 | 2.619 | 2.511 |

The trigger is the **flag pattern** (all-missing), not the content and not the
restored endpoint; content decides only whether the constant v is un-scaled by
a real σ (oracle: explosion) or by the 1e-5 scale floor (zeros: near-zero
degeneracy — the same v = 12.238 multiplied by 1e-5 gives |y| ≈ 1.03).
Generality check: cpt corner explodes on ETTm1 ch0 (3.1e9) and weather ch0
(1.1e10) with the *identical* v = 12.238; vanilla stays finite.

## 4. Layer-by-layer trace (ETTh1 ch0, 8 windows, exploding config vs finite
references clean-cpt / corner-vanilla / corner-ctl5k)

- **Instance norm**: loc median 11.27, scale median 5.65, scale-at-floor
  fraction 0.0 — ordinary statistics; no division by ~zero anywhere in the
  exploding path. Input embeddings O(0.5).
- **Encoder**: no sublayer ever leaves the finite-reference envelope — max
  corner/envelope ratio over all 37 stages is **0.96** (context-position
  residual growth 0.5→73 across blocks is generic: vanilla corner 62.5,
  cpt *clean* 101.6). Future-position hidden states stay ≤ 3.8 through block
  11 for cpt corner (vanilla corner 5.4, clean 5.5); after the final layer
  norm: 15.1 (cpt corner) vs 19.0 (vanilla corner) vs 8.3 (clean).
- **Attention**: valid logits ≤ 8.3, no inf/nan. Future/[REG] queries see only
  the 5 content-free keys; their time-attention entropy (0.32–1.14 nats) is
  concentrated but **statistically identical between exploding cpt and sane
  vanilla** (0.44–1.09) — no differential collapse.
- **The head output (normalized quantiles v)**: median-quantile |v| at the
  corner = **12.238 (cpt)** vs 0.409 (vanilla), 0.466 (ctl5k), 2.217 (cpt
  clean). sinh(12.238) = 1.03e5 → y ≈ 1.03e5·5.65 + 11.3 ≈ 5.8e5. The
  extreme quantiles reach |v| = 20.4 → sinh 3.6e8 (fan max |y| 2.6e9).
  Side finding: vanilla's corner fan is tail-heavy too (q0.01 |v| = 18.7,
  |y| up to 4.6e8) — S70B's median metric was blind to it; cpt's novelty is
  that the **entire** fan sits at |v| ≥ 11.75 (clean cpt: 1.5–3.3).
- The sane models' corner constant sinh(0.41)·σ + μ ≈ 0.42σ + μ is a
  predict-the-instance-mean fallback — the reasonable degenerate behaviour.

## 5. Encoder-vs-head attribution (cross-swap at the corner, v_medq)

| encoder \ head | cpt | vanilla | ctl5k |
|---|---|---|---|
| cpt | **12.238** | 5.777 | 4.630 |
| vanilla | 6.404 | 0.409 | 0.709 |
| ctl5k | 2.486 | 0.466 | 0.466 |

Both sides contribute, roughly additively in v (0.41 + 5.37 + 5.99 ≈ 11.8 ≈
12.24); sinh then exponentiates the sum. Consistent with the static weight
diff: cpt's two most-moved leaves vs vanilla are
`output_patch_embedding.output_layer.weight` (max|Δ| 0.117) and
`encoder.final_layer_norm.weight` (max|Δ| 0.098, largest rms 0.017 of all
leaves). ctl5k's corner readout stays aligned with vanilla's.

## 6. WiSE-FT (arm_W50_s20260903, 0.5·vanilla + 0.5·cpt) at the corner

v_medq 12.238 → 9.845 (v-excess over vanilla: 11.83 → 9.44, **−20%**, not
−50%); because the un-scaling is exponential, that 20% v-cut becomes
**77–85x lower relMSE** (2.24–2.47e9 → 2.63–3.21e7 across ch0–2) and ~11x
lower max|y| (9.0e5 → 8.2e4 on ch0). Interpolation does not halve the
log-space blow-up (log10 relMSE 9.35 → 7.42 of vanilla's 0.16); it shrinks v
modestly and sinh does the rest. Weight-delta check: w50 deltas = 0.5 × cpt
deltas everywhere (ratio 0.5000 ± 2e-5), as constructed.

## 7. Verdict

**H3 — a CPT-weights property at the never-trained all-missing-flag corner,
localized to the final-layer-norm/output-head readout direction; the numerical
explosion is manufactured by the arcsinh inverse of the instance norm.**

- **H1 (normalization degeneracy): rejected as the cause.** loc/scale are
  ordinary (11.27/5.65; floor fraction 0); nothing divides by ~zero in the
  exploding config. The norm stack matters only as the *amplifier*: the
  intended `use_arcsinh` inverse applies sinh to the head output, and the
  1e-5 scale floor explains why the zeros fill (same v) collapses to ~0
  instead of exploding.
- **H2 (attention collapse): rejected.** The mask geometry (context tokens
  unreadable; group attention exactly uniform at masked positions, entropy
  log B) is shared bit-for-bit by vanilla and ctl5k on identical inputs, and
  per-layer entropy/logit profiles of cpt and vanilla at the corner are
  statistically indistinguishable; no inf, no logit blow-up (≤ 8.3).
- **H3: supported, with discriminating measurements.** (i) v is a single
  input-independent constant per model (dispersion ≤ 1.7e-8; oracle ≡ zeros ≡
  native ≡ grouped to ≤ 2.3e-7), so the phenomenon is a property of
  weights × flag-pattern alone; (ii) no encoder stage leaves the finite
  envelope (max ratio 0.96) — the first anomalous quantity is the
  median-quantile head output, 12.238 vs 0.409/0.466 for vanilla/ctl5k on the
  same constant input (30x; 5.5x the finite-reference envelope); (iii)
  cross-swapping encoder and head between cpt and vanilla splits the v-excess
  ~additively (5.78 / 6.40), matching the weight-diff locus
  (`output_patch_embedding.output_layer`, `encoder.final_layer_norm`).

## Suggested paper sentences (appendix D.8)

> At the never-trained full-outage corner (every context position declared
> missing with content present), the declaration mask removes all context
> tokens from the attention key set, so each model's forecast reduces to a
> per-window constant y = sinh(v)·σ + μ: the normalized output v is a single
> weight-dependent vector (per-window dispersion < 2e-8; identical for oracle
> and zero fills, for the restored and native endpoints, and with or without
> covariates), and the data enter only through the instance-norm statistics
> (μ, σ). Vanilla and the fair-budget control emit |v₀.₅| ≈ 0.4 — a benign
> predict-the-mean constant after un-scaling — whereas the CPT retrofit emits
> |v₀.₅| ≈ 12.2, which the arcsinh inverse (sinh) of the instance norm
> exponentiates into |ŷ| ≈ 1e5·σ (median relMSE ≈ 4.3e9 on ETTh1); a zero fill
> hits the 1e-5 scale floor instead, which turns the *same* v into the
> complementary near-zero degenerate forecast. The instability is therefore a
> CPT-weights property of the final-layer-norm/output-head readout direction
> at this flag corner — encoder activations never leave the vanilla envelope
> and attention is no more collapsed than in vanilla — not a normalization
> degeneracy or an attention collapse; WiSE-FT halves the weight displacement
> but cuts the v-excess only ~20% (12.24 → 9.85), which the exponential
> un-scaling still converts into a ~80x relMSE reduction.

## Reproduction

```
cd tsfm_missing/experiments/s72_rootcause
CUDA_VISIBLE_DEVICES=<idle> HF_HOME=../../../.hf_cache \
  ../../../.venv/bin/python run_s72_rootcause.py --phase all
# ~1 min; writes s72_rootcause.json (all measurements; verdict block at
# .verdict, per-stage trace at .trace.configs, ladder at .ladder.cells)
```

Log of the 2026-09-10 run: `s72_run.log` (GPU 3, idle-polled).
