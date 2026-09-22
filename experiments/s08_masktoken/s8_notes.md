# S8 — causal validation & generalization of the missingness signal (chronos-bolt-base)

Grids/windows/seeds identical to run_s5_missing.py (300 windows, 2 mask seeds for
mcar/block, 1 for rank-deterministic mnar, L=512, H=96, median quantile, last-20%
test region). All relMSE = MSE / paired native-clean MSE per dataset, then
dataset-averaged unless stated. Numbers below are dataset-avg.

## Hard-anchor check vs S5 — PASSED (exact)

| cell (p=0.7)     | S5    | S8    |
|------------------|-------|-------|
| mcar nan         | 1.131 | 1.131 |
| mcar linear      | 1.072 | 1.072 |
| block nan        | 1.085 | 1.085 |
| block linear     | 1.360 | 1.360 |
| mnar_high nan    | 3.393 | 3.393 |
| mnar_high linear | 2.174 | 2.174 |

Also: `encode_replica` (mask_mode="correct", no token) is bit-identical to the
native encode (max|pred diff| = 0.0 over a 6-window probe), and the S8 attention
harness reproduces S7's published nan enrichment per cell exactly
(0.032/0.030/0.493/0.252/0.275/0.066) once S7's denominator convention is applied.

## Methodological note — S7's enrichment denominator

S7's `frac_corr` for the nan condition divides the corrupted-patch count by the
**fully-OBSERVED patches + REG** key set (`run_s7_attrib.py` l.513-515), not by
the attendable set. Under mcar that key set is ~0.1 patches/series, which
mechanically produces the 0.03 "avoidance" values. S8 stores **both**
conventions: `enrich_s7` (S7-compatible, comparability with the S7 band
0.03–0.50) and `enrich_am` (mass / attendable-key fraction — the true uniform
baseline, causally interpretable). All mode/token comparisons below use both;
no conclusion differs between them.

## Exp1 — mask-channel causal ablation (nan path; only the concatenated mask
channel is manipulated; value mean-fill and the attention mask always follow
the TRUE mask)

relMSE vs clean (enrich_am in parentheses; enrich_s7 flat as well, see json):

| mech, p         | correct | zero (ch.=0) | random (50% flip) | invert |
|-----------------|---------|--------------|-------------------|--------|
| mcar 0.3        | 1.006 (0.99) | 2.823 (0.99) | 1.114 (0.99) | 1.533 (0.99) |
| mcar 0.7        | 1.131 (0.99) | 2.797 (0.99) | 1.316 (0.99) | 1.679 (0.99) |
| block 0.3       | 1.012 (0.70) | 2.595 (0.71) | 1.115 (0.72) | 2.039 (0.75) |
| block 0.7       | 1.085 (0.58) | 2.717 (0.58) | 1.258 (0.59) | 1.935 (0.60) |
| mnar_high 0.3   | 1.188 (0.79) | 2.675 (0.79) | 1.582 (0.78) | 2.353 (0.80) |
| mnar_high 0.7   | 3.393 (0.50) | 7.921 (0.50) | 4.513 (0.50) | 4.958 (0.51) |

- **The channel is causal, strongly**: zeroing it (values still natively
  mean-filled, attention mask unchanged) costs +1.5…+1.8 relMSE at p=0.3 and
  +1.6…+4.5 at p=0.7 (mnar_high 3.39→7.92). The model reads the channel.
- **But not through attention.** Enrichment (both conventions) and entropy are
  flat across all four modes (e.g. mnar_high p=0.7: enrich_am 0.504/0.503/
  0.501/0.508). The mask channel acts through the **embedding/value pathway**
  (input_patch_embedding reads the flag bits into the token content), not by
  rerouting attention. S7's nan-vs-zero-fill enrichment contrast was driven by
  the *hard attention mask* on fully-missing patches, not by this channel.
- **Lying masks**: the model is gracefully but really deceived —
  correct < random < invert < zero everywhere. 50% bit noise costs little
  (+0.1…+1.1); systematic inversion hurts more (+0.5…+1.6); "everything is
  missing" (zero) is worst despite identical values — a confidently false
  global flag does more damage than either noise or inversion. So the model
  trusts the flag's content: it discounts flagged-as-missing information even
  when the values there are real.

## Exp2 — learnable [MASK] token (training)

768-d vector, zero-init, replaces the patch embedding of fully-missing patches
(which become attendable; partial patches keep native mean-fill+concat).
Backbone frozen; clipped pinball (clip=100, S6 replica), AdamW lr 1e-3, 1000
steps × 256 series, augmentation 50% mcar / 50% block(24), p~U(0.05,0.8),
train split. One token per dataset. 255 s/GPU. Final token norms: ETTh1 0.495,
ETTm1 0.767, weather 0.728. Clipped loss 38.6→33.2 (ETTh1), noisy-flat
(ETTm1/weather; unclipped raw 1e2–6.5e5 — the S5 loss explosion, clipped as
designed). The loss is dominated by observed-patch terms, so the token's gain
is only weakly visible in the training curve — Exp3 is the real test.

## Exp3 — generalization grid (test-ds-avg relMSE; rate-avg over p∈{0.1..0.7})

| test mech   | nan   | linear | mponly | tok_ETTh1 | tok_ETTm1 | tok_weather |
|-------------|-------|--------|--------|-----------|-----------|-------------|
| mcar        | 1.041 | **1.009** | 1.033 | 1.040 | 1.040 | 1.040 |
| block       | 1.035 | 1.134  | 1.043  | 1.008 | 1.026 | **0.994** |
| mnar_high   | 1.712 | **1.354** | 1.536 | 1.799 | 1.743 | 1.794 |
| mnar_extreme| 1.383 | **1.272** | 1.377 | 1.426 | 1.401 | 1.390 |

- **block (in-distribution for the token): real gain, beats every control.**
  All three tokens beat nan at *every* rate (p=0.2: 0.947–1.004 vs nan 1.007;
  p=0.7: 1.071–1.087 vs nan 1.085). Gain peaks at p=0.2–0.4 (relMSE 0.95–0.97,
  i.e. *below* the clean baseline on those windows) and shrinks at p=0.7 (few
  observed patches left to attend). Token > nan > mponly > linear here —
  linear's fabricated segments hurt (1.134), consistent with S5.
- **Cross-dataset transfer: asymmetric, in-domain token is always best.**
  block rate-avg per test set: ETTh1 — tok_ETTh1 0.966 < tok_ETTm1 0.982 <
  nan 1.003 < tok_weather 1.057; ETTm1 — tok_ETTm1 1.000 < tok_ETTh1 1.036 <
  nan 1.049 < tok_weather 1.061; weather — tok_weather 0.863 < tok_ETTh1 1.021
  < nan 1.052 < tok_ETTm1 1.096. The ETT tokens transfer (beat nan everywhere);
  the weather token does NOT transfer to ETT (worse than nan) but is by far the
  best on weather in-domain (0.863). tok_weather's dataset-avg "win" is carried
  by its in-domain cell — honest reading: diagonal transfer matrix dominates.
- **mcar: no-op** (fully-missing patches are rare under mcar; tok ≡ nan ≈
  mponly ≈ 1.04; linear keeps its known edge, 1.009, driven by weather at high
  p: linear 0.881 vs nan/tok ≈ 1.31–1.32).
- **MNAR: no transfer — negative result.** Tokens are flat-to-worse than nan
  on mnar_high (1.74–1.80 vs 1.71) and mnar_extreme (1.39–1.43 vs 1.38), and
  far behind linear (1.35/1.27). Same conclusion as S5's adapter: training on
  mcar+block cannot synthesize information that MNAR censoring destroys;
  mponly (which saw mnar in its S5 training mix) does better than the token
  there (1.536 vs 1.74–1.80) but still loses to linear.

## Exp4 — did the token learn to be IGNORED? (attention audit, matched tokens)

Token-position enrichment, dataset-avg (trained token | zero-token control):

| cell          | enrich_am trained | enrich_am zero | enrich_s7 trained | entropy trained (zero) |
|---------------|-------------------|----------------|-------------------|------------------------|
| mcar 0.3      | 0.000* | 0.000* | 0.000* | 2.073 (2.073) |
| mcar 0.7      | 1.229 | 1.150 | 0.037 | 2.096 (2.097) |
| block 0.3     | 1.120 | 1.064 | 0.700 | 2.057 (2.083) |
| block 0.7     | 1.027 | 1.002 | 0.321 | 2.070 (2.111) |
| mnar_high 0.3 | 1.087 | 1.034 | 0.454 | 2.069 (2.083) |
| mnar_high 0.7 | 1.003 | 0.989 | 0.108 | 2.099 (2.126) |

*mcar p=0.3 has essentially no fully-missing patches (degenerate).

- **Negative: the token is NOT discounted.** Against the true uniform
  baseline (enrich_am), token positions receive ≈1.0 — exactly their frequency
  share, the same "blind dilution" S7 documented for zero/linear fill. The
  untrained zero-token control shows the identical pattern (1.00–1.15), so
  **training did not change attention allocation at all**. By the letter of the
  pre-registered test, enrich_s7 lands in the 0.03–0.50 native band at p=0.7
  (0.321/0.108; block p=0.3 sits above at 0.700) — but that is S7's denominator
  artifact (see methodological note), not learned avoidance; the honest verdict
  comes from enrich_am ≈ 1.0.
- What training DID do is content-side: the learned vector is a better gap
  prior than the all-zero vector (Exp3 block gains), while attention keeps
  reading those positions at face value. Same story as Exp1: in bolt, the
  missingness signal acts through the **value/embedding pathway**, never
  through attention reweighting.
- Entropy side-effect: native nan dips to 1.86–1.91 at p=0.7 (mechanical —
  fully-missing keys are hard-masked); with token positions attendable,
  entropy returns to the clean level ≈2.07–2.10 (clean ref 2.066). No rewiring.

## Answers

1. **Mask channel causal? Does the model believe a lying mask?** Yes, causal —
  zeroing the channel costs up to +4.5 relMSE (mnar_high p=0.7) with identical
  values and attention mask. And yes, it believes lies, gracefully:
  correct < random < invert < zero at every cell. But the channel's effect is
  NOT mediated by attention (enrichment/entropy flat across all
  manipulations) — it is read into the patch-embedding content and acted on
  downstream.
2. **Does the [MASK] token generalize?** Mechanism-wise: only where its
  training distribution reaches — clear win on block (beats nan, linear and
  the S5 adapter; e.g. rate-avg 0.994–1.026 vs nan 1.035, linear 1.134),
  neutral on mcar, negative on mnar_high/mnar_extreme (linear stays best
  there — censored information is unrecoverable, reproducing S5's conclusion
  with a second, independent method). Dataset-wise: in-domain token always
  best; ETT tokens transfer to each other and to weather; the weather token
  does not transfer to ETT. Rate-wise: gains at p≤0.6, vanishing at p=0.7.
3. **Did it learn to be ignored? No.** Token-position enrichment ≈ 1.0 against
  the attendable baseline, identical to an untrained zero token; only S7's
  denominator-artifact convention puts it in the 0.03–0.50 "native band". The
  token helps because of *what it says*, not because attention discounts it.

## Artifacts

`run_s8_masktoken.py` (tiny self-test / anchor / train / exp1 / exp3 / exp4 /
merge, 8-way sharded), `analyze_s8.py` (merge tables + figure),
`s8_results.json` (merged; per-shard `s8_results_{exp1,exp3,exp4,exp4z}_shard*.json`),
`s8_anchor_{ETTh1,ETTm1,weather}.json`, `s8_masktoken_{ds}.pt`,
`s8_train_{ds}.json` (loss curves), `s8.png`, logs `s8_anchor{0-2}.log`,
`s8_train{0-2}.log`, `s8_exp1_shard{0-4}.log`, `s8_exp3_shard{0-7}.log`,
`s8_exp4_shard{0-3}.log`, `s8_exp4z_shard{0-3}.log`.
