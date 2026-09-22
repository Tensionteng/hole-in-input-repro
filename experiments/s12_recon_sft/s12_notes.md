# S12 — reconstruction-aware deep fine-tuning (RECON-SFT) vs the input-side ladder

**Question.** S11 found the true values behind fully-missing patches are linearly
decodable from mid-encoder hidden states (l4–l8, R² = 0.45–0.81 for mcar/block)
even though the input embedding carries exactly 0 — and that the forecast head
does not fully exploit that signal. S12 trains that signal explicitly: S6's
masked-augmentation SFT **+ λ·auxiliary loss** that regresses the 16 true values
of every fully-missing NaN-form patch from the encoder layer-6 hidden state
(learnable Linear(768→16) readout, clean-window instance-norm z-space, pointwise
MSE clipped at 100; head dropped at eval). If the S11 headroom is real forecast
headroom, RECON-SFT should beat SFT-mb.

**Pre-registered outcomes (written before any run; no tuning toward A):**
- **Outcome A**: RECON-SFT > SFT-mb on the mcar/block grid → the deep objective
  converts S11's headroom into gains; new method chapter.
- **Outcome B**: RECON-SFT ≈ SFT-mb → plain masked-SFT already uses the
  reconstruction headroom; S11's value is explaining why SFT works.

**Variants** (same LoRA r=16/α=32/do=0.05 on q/k/v/o, 3000 steps × 256, AdamW
1e-4 + 200 warmup, grad-clip 1.0, clipped pinball clip=100, S6-mb augmentation
mcar+block24 p~U(0.05,0.8), 20% clean, 50% NaN/50% linear input, first-70%
train split; all variants of a dataset share S6-mb's exact seed → identical
augmentation stream): `mb` = S6 rerun (anchor), `recon` λ=1.0, `recon03` λ=0.3.
~280 fully-missing NaN-form patches per batch carry the aux loss.

## Anchor gate — PASS (bit-identical, stronger than the pre-registered ±5%/±2%)

All 78 gate cells (clean + {mcar,block}×4 rates×3 ds × {zs:nan, zs:linear,
mb:nan}) match s6_sft_results.json at **ratio 1.0000 exactly**. The mb rerun's
final training losses equal S6's to the printed digit (25.3121 / 27.7125 /
26.5109). The fixed-seed S6 pipeline is fully deterministic on this hardware;
the S12 harness is an exact replica.

## Main grid — relMSE vs paired ZS clean (150 windows × 1 mask seed, S6 protocol)

mcar (nan fill unless noted):

| p   | zs:nan | zs:linear | tok (S8) | mb    | mb:lin | recon λ=1 | rec:lin | recon03 |
|-----|--------|-----------|----------|-------|--------|-----------|---------|---------|
| 0.1 | 0.9942 | 1.0013    | 0.9942   | 0.7567 | 0.7422 | **0.7508** | 0.7403 | 0.7529 |
| 0.3 | 0.9920 | 0.9976    | 0.9920   | 0.7714 | 0.7384 | **0.7656** | 0.7379 | 0.7667 |
| 0.5 | 1.0369 | 1.0045    | 1.0369   | 0.7785 | 0.7460 | **0.7748** | 0.7429 | 0.7755 |
| 0.7 | 1.1031 | 1.1058    | 1.0995   | 0.8186 | 0.7833 | **0.8087** | 0.7822 | 0.8127 |
| avg | 1.0316 | 1.0273    | 1.0307   | 0.7813 | 0.7525 | **0.7750** | 0.7508 | 0.7770 |

block:

| p   | zs:nan | zs:linear | tok (S8) | mb    | mb:lin | recon λ=1 | rec:lin | recon03 |
|-----|--------|-----------|----------|-------|--------|-----------|---------|---------|
| 0.1 | 1.0156 | 1.0100    | **0.9576** | 0.7425 | 0.7434 | 0.7409 | 0.7418 | 0.7416 |
| 0.3 | 1.0214 | 1.0587    | **0.9355** | 0.7680 | 0.7651 | 0.7678 | 0.7638 | 0.7665 |
| 0.5 | 1.0498 | 1.1853    | **0.9469** | 0.7701 | 0.7753 | 0.7711 | 0.7746 | 0.7698 |
| 0.7 | 1.1007 | 1.3247    | **1.0151** | 0.7998 | 0.8173 | **0.7941** | 0.8153 | 0.7968 |
| avg | 1.0469 | 1.1447    | **0.9638** | 0.7701 | 0.7753 | **0.7685** | 0.7739 | 0.7686 |

Per-dataset rate-avg over both mechanisms (nan): ETTh1 — zs 0.9967, tok 0.9748,
mb 0.9043, recon 0.9018, recon03 0.9046; ETTm1 — 1.0414, 1.0176, 0.9354,
**0.9264**, 0.9269; weather — 1.0796, 0.9994, 0.4874, 0.4870, 0.4870.

Paired per-window test (recon − mb, nan, n=1800 window-means per mechanism):
mcar **−0.0063 relMSE (t = −3.87, but win-rate 49.3%)**; block −0.0016
(t = −1.29, win 49.4%). The mcar edge is statistically detectable only because
the pairing is exact; its size is ~0.8% relative with a sub-50% win rate (a few
windows carry it), and λ=0.3 sits between. Own-clean-relative at p=0.7 (repair
isolated from in-domain adaptation, S6's confound guard): mcar 1.101 (mb) →
1.093 (recon); block 1.083 → 1.079. Clean cost: none (clean MSE 10.16→10.12
ETTh1, 8.64→8.63 ETTm1, 1798→1793 weather — mb vs recon).

**Reading.** The input-side ladder (zs-nan 1.03–1.05, zs-linear 1.03–1.14, S8
token 0.96–1.03) is crushed by both SFT variants (~0.77) as S6 established;
between the two deep variants the auxiliary reconstruction loss buys **at most a
sub-1% relative gain on mcar and nothing on block**. The S8 token's block win
(within zero-shot methods) reproduces in this harness.

## Mechanism diagnostic — l6 probe after training (title-figure candidate)

full_miss r2_mean at layer 6, S11 windows/PCA(64)+Ridge(1.0) pipeline, configs
p=0.7 (S11 ZS reference in brackets):

| dataset × cfg | zs | mb | recon λ=1 | recon03 |
|---|---|---|---|---|
| ETTh1 mcar    | 0.535 [0.539] | 0.512 | 0.532 | 0.524 |
| ETTh1 block   | 0.475 [0.474] | 0.478 | 0.487 | 0.481 |
| ETTm1 mcar    | 0.703 | 0.693 | 0.720 | 0.699 |
| ETTm1 block   | 0.542 | 0.547 | **0.591** | 0.562 |
| weather mcar  | 0.803 [0.802] | 0.806 | 0.815 | 0.810 |
| weather block | 0.615 [0.616] | 0.628 | **0.664** | 0.639 |

The ZS column reproduces S11 to ≤0.004 (validates the single-layer probe
variant). SFT-mb leaves the signal essentially unchanged (≈zs everywhere).
RECON-SFT raises block decodability by +0.01…+0.05 R² (best where n is large:
ETTm1/weather block) — **the representation does become more reconstructable —
yet forecast error does not move**. mcar cells have small full-miss samples
(n=514/1410) and are flat within noise. Training-curve cross-check: aux MSE
falls 1.12→0.55 (ETTh1), 1.11→0.51 (ETTm1), 0.95→0.34 (weather) while the main
pinball trajectory is indistinguishable from mb's (final 25.25 vs 25.31 on
ETTh1) — the aux objective was learned, not ignored.

## Negative controls — PASS (no gain, no leakage signal)

p=0.7, relMSE vs zs clean (own-clean-relative for SFT):

| mech | zs:nan | zs:linear | mb | recon λ=1 | recon03 |
|---|---|---|---|---|---|
| mnar_high   | 3.365 | 2.194 | 3.438 (own 4.32) | 3.422 (own 4.32) | 3.440 |
| mnar_extreme| 1.717 | 1.600 | 1.592 (own 2.07) | 1.589 (own 2.07) | 1.591 |

recon ≈ mb to the third decimal on both MNAR mechanisms, and both sit at S6's
documented SFT-mb level (mnar_high not repaired at all — slightly worse than
zs-nan on ETT; mnar_extreme's small vs-zs gain is S6's in-domain adaptation,
identical own-clean 2.07). No incremental aux effect ⇒ nothing to audit.

## Verdict — **Outcome B**

RECON-SFT ≈ SFT-mb (mcar −0.8% relative at best, block −0.2%, win-rates ≈ 50%),
while the l6 representation measurably sharpens (block +0.01…+0.05 R²).
**One-sentence mechanism conclusion:** making the mid-encoder reconstruction
more linearly decodable does not improve forecasts — plain masked-augmentation
SFT already extracts everything the context makes available, so the gap between
S11's 0.5–0.8 probe R² and perfect repair is intrinsic contextual information
loss, not head under-use; S11's headroom is real as representation but not as
forecast headroom.

## Caveats / honest notes

- The mcar edge (t=−3.87) is real only under exact pairing and is carried by a
  minority of windows (win-rate 49.3%); we do not claim it as a method win.
  λ=0.3→1.0 shows no dose response beyond noise.
- Aux loss is applied to NaN-form fully-missing patches only (the S11 decoding
  setting); linear-form series' gaps leak endpoint information through the fill
  and were excluded by design.
- Aux loss magnitude (~0.5 learned) sits ~50× below the pinball term (~25–30,
  sum-over-horizon reduction) at λ=1 — the pre-registered weight, not tuned up
  (that would be tuning toward outcome A).
- relMSE-vs-zs-clean mixes repair with in-domain adaptation (S6 Q3): weather's
  ~0.49 is mostly adaptation. Own-clean-relative isolates repair (table above);
  no conclusion differs.
- Probes: mcar p=0.7 full-miss n=514 (ETT) / 1410 (weather) — wide error bars;
  block cells (n=39k–116k) carry the R²-rise claim.
- mb's clean MSE improves over zs exactly as in S6 (10.16 vs 11.65 ETTh1 etc.);
  recon adds no extra clean cost/gain.

## Artifacts

`run_s12_recon.py` (--tiny/--train/--eval/--anchor/--probe/--merge/--summary/
--figure), `s12_results.json` (anchor cells, eval/neg grids with per-window
MSE, probe table, summary tables), `s12.png`, `s12_ckpt/` (s12_{mb,recon,
recon03}_{ds}.pt LoRA+aux-head checkpoints, eval_shard*.json, probe_*.json,
anchor.json), logs `s12_tiny.log`, `s12_train_{variant}_{ds}.log` ×9,
`s12_eval_shard{0-7}.log`, `s12_probe_{ds}_{model}.log` ×12, `s12_anchor.log`,
`s12_merge.log`.
