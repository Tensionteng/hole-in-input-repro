# S15 — reconstruction-routed decoding: open bolt's decoder cross-attention mask on fully-missing patches

**Status: pre-registration frozen 2026-08-13, before any grid run. Smoke
(`s15_smoke.log`) passed: native decoder cross-attention puts exactly 0 mass
on fully-missing keys at all 12 layers; unmasked decode puts real mass
(max 1.000/0.838/0.607 at layers 3/5/4); unmasked predictions differ on
masked input (max 1.83), are NaN-free; clean input unmasked == native
BITWISE (0.0); restore bitwise. Results sections below are filled in after
the runs; nothing above this line is edited post-hoc.**

## Question

S11 found the true values behind fully-missing patches are linearly decodable
from mid-encoder hidden states (R² 0.5–0.8). S13 proved (a) the rebuilding is
attention-causal (full knockout → R²→0) and (b) it has **exactly zero**
predictive function, because bolt's `decode()` cross-attends with
`encoder_attention_mask` = the patch mask (chronos_bolt.py l.429-435), which
excludes fully-missing positions — the rebuilt content is architecturally
unreadable by the forecast head. S12 showed explicitly sharpening the
representation (aux loss) buys nothing. The remaining untested link: **route
the rebuilt signal to the head** by opening that mask. If the S11 headroom is
real forecast headroom, this should pay; if the head gains nothing, the
redundancy conclusion is sealed by direct evidence.

## Pre-registered outcomes (no tuning toward A)

- **Outcome A**: `sft:unmask` significantly beats `sft:mb` (S12's masked SFT)
  on the mcar/block grid — paired per-window test, mean relative gain > 2%
  with the 95% CI of the paired diff excluding 0 → new method chapter
  "reconstruction-routed decoding".
- **Outcome B**: `sft:unmask` ≈ `sft:mb` → the rebuilt content is redundant
  for the head (observed positions already carry equivalent information);
  the attribution chapter closes with direct evidence.

## Design

- **Unmask operation** (runtime monkey-patch, no library edits): replace the
  decoder cross-attention `encoder_attention_mask` by all-ones (every one of
  the 33 positions — 32 patches + [REG] — is a real position). Encoder
  untouched (reconstruction machinery stays native). Active during training
  (sft:unmask) and eval.
- **Conditions**: `zs:masked` (native bolt-nan, control/anchor), `zs:unmask`
  (ZS + unmask; OOD — the decoder never read gap positions in pretraining,
  degradation is informative), `sft:unmask` (unmask + SFT, **S12/S6 SFT-mb
  config and budget exactly**: LoRA r=16/α=32/do=0.05 q/k/v/o, 3000 steps ×
  256, AdamW 1e-4 + 200 warmup, grad-clip 1.0, clipped pinball clip=100, aug
  mcar+block24 p~U(0.05,0.8), 20% clean, 50% NaN/50% linear input, first-70%
  train split, S12-mb seed → identical augmentation/dropout stream; the ONLY
  difference vs S12's mb run is the cross-attention mask). References lifted
  from `s12_results.json`: `mb:nan` (masked SFT) and `tok:nan` (S8 token).
  `mb_anchor`: SFT-mb retrained through THIS harness (anchor gate: eval cells
  vs S12 mb:nan ±5%; checkpoint param diff vs `s12_ckpt/s12_mb_*.pt`,
  expected exactly 0).
- **Grid**: {mcar, block} × p {0.1, 0.3, 0.5, 0.7} × {ETTh1, ETTm1, weather};
  S12 protocol exactly (150 test windows s5.SEED, 1 mask seed, median
  quantile, H=96, relMSE vs paired ZS clean; per-window MSE stored).
- **Negative controls**: {mnar_high, mnar_extreme} × p=0.7 × all 4 conditions
  — no gain expected (censored information is not reconstructable).
- **Clean sanity**: clean input, unmask == masked bitwise (asserted in smoke
  AND in merge from the eval cells).
- **Anchor gates**: (1) zs:masked cells vs `s12_results.json` zs:nan ±2%
  (expect bit-exact; representative subset mcar/block × {0.3,0.7} × 3 ds +
  clean run before training); (2) mb_anchor eval vs S12 mb:nan ±5% + ckpt
  param diff report.

## Gates — ALL PASSED (bit-exact, stronger than the pre-registered tolerances)

- **smoke** (`s15_smoke.log`): native decoder cross-attention puts exactly 0
  mass on fully-missing keys at all 12 layers; unmasked decode puts real mass
  there (per-query max 1.000 / 0.838 / 0.607 / 0.502 at layers 3/5/4/6) with
  rows still summing to 1, no NaN; unmasked predictions differ from native on
  masked input (max 1.83) and are NaN-free; clean input unmasked == native
  BITWISE; post-restore bitwise.
- **zs anchor** (`s15_anchor.log`): all 27 cells (clean + mcar/block × 4 rates
  × 3 ds) reproduce `s12_results.json` zs:nan at **ratio 1.000000 exactly**
  (gate ran twice: subset of 15 before training, full 27 after).
- **mb anchor**: SFT-mb retrained through this harness reproduces S12's mb run
  to the bit — final train losses 25.3121 / 27.7125 / 26.5109 == S12's printed
  values, checkpoint parameter max|diff| vs `s12_ckpt/s12_mb_*.pt` = **0.0** on
  all three datasets, and all 27 eval cells (clean + mcar/block grid) at ratio
  1.000000 vs s12 mb:nan. The sft:unmask run therefore differs from S12's mb
  by exactly one controlled bit-level variable: the cross-attention mask.
- **clean sanity** (merge, from eval cells): `clean:zs_unmask` ==
  `clean:zs_masked` per-window bitwise on all three datasets.

## Main grid — relMSE vs paired ZS clean (150 windows × 1 mask seed, S12 protocol)

mcar (nan fill; mb:nan and tok:nan lifted from s12_results.json):

| p   | zs:masked | zs:unmask | mb:nan (S12) | sft:unmask | tok:nan (S8) |
|-----|-----------|-----------|--------------|------------|--------------|
| 0.1 | 0.9942 | 0.9942 | 0.7567 | 0.7582 | 0.9942 |
| 0.3 | 0.9920 | 0.9920 | 0.7714 | 0.7700 | 0.9920 |
| 0.5 | 1.0369 | 1.0369 | 0.7785 | 0.7771 | 1.0369 |
| 0.7 | 1.1031 | 1.1032 | 0.8186 | 0.8162 | 1.0995 |
| avg | 1.0316 | 1.0316 | 0.7813 | 0.7804 | 1.0307 |

block:

| p   | zs:masked | zs:unmask | mb:nan (S12) | sft:unmask | tok:nan (S8) |
|-----|-----------|-----------|--------------|------------|--------------|
| 0.1 | 1.0156 | 1.0143 | 0.7425 | 0.7421 | 0.9576 |
| 0.3 | 1.0214 | 1.0193 | 0.7680 | 0.7721 | 0.9355 |
| 0.5 | 1.0498 | 1.0499 | 0.7701 | 0.7709 | 0.9469 |
| 0.7 | 1.1007 | 1.0932 | 0.7998 | 0.8042 | 1.0151 |
| avg | 1.0469 | 1.0442 | 0.7701 | 0.7723 | 0.9638 |

Per-dataset rate-avg over both mechanisms (zs:masked / zs:unmask / mb /
sft:unmask / tok): ETTh1 — 0.9967, 0.9950, 0.9043, 0.9001, 0.9748; ETTm1 —
1.0414, 1.0420, 0.9354, 0.9395, 1.0176; weather — 1.0796, 1.0767, 0.4874,
0.4895, 0.9994.

Clean MSE (zs / mb / sft:unmask): ETTh1 11.6456 / 10.1632 / 10.1158; ETTm1
9.6868 / 8.6446 / 8.6509; weather 3882.00 / 1798.21 / 1808.85 — unmask SFT
pays no systematic clean cost (ETTh1 slightly better, ETTm1/weather ≤0.6%
worse).

## Paired per-window tests — sft:unmask vs sft:mb (the pre-registered endpoint)

diff = unmask − mb on identical windows/masks; rel_gain = −mean(diff)/mean(mb).

| mechanism | n | rel_gain | 95% CI | t | win-rate |
|-----------|---|----------|--------|---|----------|
| mcar      | 1800 | −0.23% | ±0.24% (incl. 0) | +1.87 | 0.504 |
| block     | 1800 | −0.61% | ±0.47% (excl. 0, wrong direction) | +2.53 | 0.485 |
| pooled    | 3600 | **−0.42%** | ±0.27% (excl. 0, wrong direction) | +3.10 | 0.495 |

Per-dataset (mcar / block): ETTh1 +0.31% (t=−1.70) / +0.63% (t=−3.01);
ETTm1 +0.11% (t=−0.34) / −0.97% (t=+1.90); weather −0.23% (t=+1.89) /
−0.62% (t=+2.54). Signs flip across datasets; the pooled effect is a small
penalty, not a gain.

Secondary — zs:unmask vs zs:masked: mcar +0.00% (p≤0.5 cells bit-identical —
fully-missing patches essentially absent; only p=0.7 windows move, mixed
sign); block **+0.55% (t=−2.81, favorable)** — the OOD zero-shot read of gap
positions helps slightly on block, but the advantage does not survive SFT
(sft:unmask block 0.7723 ≥ mb 0.7701).

## Negative controls — mostly PASS, one surprise (mnar_high ETT)

p=0.7, relMSE vs zs clean (own-clean-relative in parentheses for SFT):

| mech | zs:masked | zs:unmask | mb (S12) | sft:unmask |
|------|-----------|-----------|----------|------------|
| mnar_high    | 3.3653 | 3.3788 | 3.4384 (own 4.58/5.66/2.73) | 3.2453 (own 4.34/5.27/2.70) |
| mnar_extreme | 1.7166 | 1.7222 | 1.5917 (own 2.17/2.36/1.68) | 1.5868 (own 2.17/2.34/1.67) |

mnar_extreme: flat as pre-registered (paired +0.14% pooled, t=−0.48, win
0.509; per-ds +0.05/+0.60/+0.14%, |t|≤1.7). mnar_high: an unexpected but real
gain on ETT — paired per-window vs S12 mb: ETTh1 **+5.67% (t=−15.5, win
0.98)**, ETTm1 **+6.91% (t=−7.6, win 0.82)**, weather +0.24% (t=−3.8);
survives own-clean normalization (4.34<4.58, 5.27<5.66, 2.70<2.73). No
leakage channel exists by construction (the unmask only exposes encoder
content that S13 proved is rebuilt from observed keys + REG; the SFT data
stream is bit-identical to mb's). The cells remain catastrophically degraded
(own-rel 4.3–5.7 ≫ 1), so this is not "repair"; the plausible mechanism is
that the rebuilt gap content — a smooth, position-aware scaffold interpolating
across the censored peak regions — gives the head a better phase/continuity
prior than isolated low-value islands, even though it cannot contain the
censored peaks themselves. Flagged as a follow-up, not claimed as a method
win (it does not touch the pre-registered mcar/block endpoint).

## Verdict — **Outcome B** (with a small penalty, if anything)

The pre-registered A criterion (paired mean relative gain > 2% with 95% CI
excluding 0) is met nowhere; the pooled mcar+block endpoint shows −0.42%
(t=+3.10, CI excludes 0 in the unfavorable direction), driven by block
(−0.61%). Per-dataset signs flip (ETTh1 mildly positive, ETTm1/weather mildly
negative). Routing the rebuilt gap content to the forecast head — with 3000
steps of identical-budget training to exploit it — does not improve forecasts.

**One-sentence mechanism conclusion:** the mid-encoder reconstruction of
fully-missing patches, once made readable by the forecast head, adds nothing
beyond what the head already computes from observed positions — S11's
0.5–0.8-R² signal is representation without forecast headroom; S12 closed the
"sharpen the representation" direction, S15 closes the "route it to the head"
direction, and together they make the redundancy conclusion architecture-proof.

## Caveats / honest notes

- The block penalty (−0.61%, CI excl. 0) is real under exact pairing; at 0.6%
  relative it is practically negligible but statistically present. ETTh1 alone
  shows a small block GAIN (+0.63%, t=−3.01) — dataset heterogeneity, no
  consistent directional story.
- zs:unmask is an OOD operation; its small block gain (+0.55%) shows the
  rebuilt content is not harmful to read zero-shot, but ZS relMSE (~1.04) sits
  far above any SFT variant (~0.77), so it has no practical weight.
- mcar zs:unmask ≡ zs:masked bitwise at p≤0.5 (fully-missing patches are
  measure-zero under mcar; p^16 per patch) — the two conditions only separate
  at p=0.7, as expected architecturally.
- relMSE-vs-zs-clean mixes repair with in-domain adaptation (S6 Q3); the
  own-clean-relative and paired numbers above are adaptation-free and support
  the same verdict.
- Paired tests use S12's mb per-window MSE; my mb_anchor rerun's per-window
  data is bit-identical to it (anchor gate), so the pairing is exact.

## Artifacts

`run_s15_unmask.py` (--smoke/--train/--eval/--anchor/--merge/--summary/
--figure), `s15_results.json` (anchor incl. ckpt param diffs, clean sanity,
eval/neg grids with per-window MSE, paired tables), `s15.png`,
`s15_ckpt/` (s15_{mb_anchor,unmask}_{ds}.pt, eval_shard*.json, anchor.json),
logs `s15_smoke.log`, `s15_anchor.log`, `s15_train_{mb_anchor,unmask}_{ds}.log`
×6, `s15_eval_shard{0-7}.log`, `s15_merge.log`. Runtime: smoke ~1 min, zs
anchor subset ~1 min, training 6 × ~915 s in parallel (GPU 0-5), full eval
~4 min on 8 GPUs.
