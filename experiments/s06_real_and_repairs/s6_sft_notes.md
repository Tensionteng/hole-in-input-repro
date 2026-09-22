# S6-F — masked-augmentation SFT of chronos-bolt-base (the "expected baseline" repair)

**Setup.** Per-dataset SFT on each dataset's train split (first 70%), LoRA r=16/α=32/
dropout 0.05 on q/k/v/o of the bolt T5 stacks (peft 0.20.0; 3.54M trainable params),
3000 steps × batch 256, AdamW lr 1e-4 (+200-step linear warmup), grad-clip 1.0. Loss =
bolt's native pinball, replicated element-wise and **clipped at 100** (normalized units) —
this fixes the s5_fix loss explosion: raw batch losses spiked to 1e4–1.3e7 on ETTm1/weather
(tiny-std channels) while the clipped loss stayed 25–90. Augmentation per series: 20%
clean; else uniform over the variant's mechanisms — SFT-mb: mcar+block(24), p~U(0.05,0.8);
SFT-all: + mnar_high top-value censoring, p~U(0.05,0.7); corrupted input shown 50% as raw
NaN / 50% linear-filled. Eval: S5 grid {mcar, block, mnar_high, mnar_extreme} × p × fills
{nan, linear}, 150 windows × 1 seed, matched dataset; relMSE vs the paired zero-shot clean
(bolt_zs re-run on the same 150 windows). 6 ckpts in `s6_sft_ckpt/` (~14 MB each).

**Q1 — how much does SFT recover?** (dataset-avg relMSE at p=0.7, linear / nan)
mcar 1.11/1.10 → **0.78–0.79**; block 1.32/1.10 → **0.80–0.84**; mnar_high 2.19/3.37 →
**1.36/1.31** (SFT-all only); mnar_extreme 1.60/1.72 → 1.47–1.62. mcar/block damage is
fully absorbed (own-clean-relative ≈ 1.0–1.1 at p=0.7 on ETT); fill choice becomes nearly
irrelevant after SFT (nan ≈ linear everywhere). Part of the headline gain is in-domain
adaptation, not repair — see Q3.

**Q2 — does MNAR augmentation generalize to MNAR test?** Yes but **mechanism-specific**.
On mnar_high (the augmented form), SFT-all ≫ SFT-mb on ETT at p≥0.3 (linear p=0.7,
vs-zs-clean: ETTh1 2.43→1.28, ETTm1 2.53→1.51; own-clean-relative: 2.78→1.48, 2.83→1.88),
while SFT-mb does not repair mnar_high at all (≈ zs; slightly worse than zs-nan on ETTh1:
4.58 vs 3.53 own-clean). On **mnar_extreme (never augmented)** SFT-all ≈ SFT-mb, even
slightly worse (ETTh1 nan p=0.7 own-clean: 2.46 vs 2.17) — no cross-MNAR transfer. On
weather mnar_high both variants are ~equal; and per the honest-negative below, neither
repairs anything there in absolute terms.

**Q3 — clean cost?** None — clean *improves* everywhere (SFT clean / zs clean): SFT-mb
0.87/0.89/0.46, SFT-all 0.86/0.80/0.53 (ETTh1/ETTm1/weather). Per-dataset SFT on 70% of
the timeline transfers strongly to the test 20% (weather clean MSE halves).

## Honest negatives / caveats
- **In-domain adaptation is a confound**: relMSE-vs-zs-clean mixes repair with adaptation.
  Own-clean-relative isolates repair — and on **weather mnar_high it exposes none**:
  absolute MSE at p=0.7 is unchanged by SFT (zs 4893 → sft_all 5085, sft_mb 4891), so the
  good-looking ratios (1.26–1.31) are pure denominator shift; relative to its own halved
  clean, SFT weather is *more* mnar-sensitive than zero-shot (2.3–2.7 vs 1.26). On ETT the
  absolute repair is real (ETTm1 mnar_high nan p=0.7: 51.4 → 15.2).
- SFT-all ≈ SFT-mb on mcar/block grids (all slightly better on ETT mcar) — MNAR
  augmentation doesn't hurt the seen mechanisms.
- 150 windows × 1 mask seed; window sample = run_s5_extra's (not S5's 300-window set), so
  bolt_zs here is a re-run, not the main-study numbers (clean MSEs match s5_extra exactly).
- Matched-dataset eval only (model trained on D tested on D); cross-dataset SFT not run.
- LoRA needed one shim: peft 0.20 requires `get_input_embeddings`, which transformers 5.14
  doesn't auto-provide for ChronosBoltModelForForecasting — patched to return the [REG]
  `shared` embedding (nothing is tied to it; peft's tied-weight check becomes a no-op).
- Loss clipping discards magnitude information on spiked batches (gradient direction kept,
  magnitude capped) — a milder alternative (per-channel loss standardization) not tried.

Artifacts: `run_s6_sft.py`, `analyze_s6_sft.py`, `s6_sft_results.json` (+ shards
`s6_sft_results_{mb,all}.json`), `s6_sft.png`, `s6_sft_ckpt/`, logs `s6_train_{mb,all}.log`,
`s6_eval_{zs,mb,all}.log`.
