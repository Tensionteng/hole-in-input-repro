# S33 notes — the causal test: is the imputation cap caused by the INTERFACE?

## The design

Everything in S27/S29/S31 is consistent with "the input interface caps what can be repaired",
but all of it was measured on Chronos-Bolt. The sceptic's answer: maybe missingness is simply
irreducibly hard and the interface is incidental.

S30 supplies the discriminating case. Four of five families discard or overwrite the fill
content once a hole is declared (permutation ratio 0.0000); **Moirai does not** (0.9739). So
the interface hypothesis makes a prediction the "missingness is just hard" hypothesis does
not: **the same imputation-quality sweep that is FLAT for Bolt's declared path must be
DECREASING for Moirai's** -- same data, same masks, same task, different interface.

## Result — CONFIRMED

Median relMSE, dataset-averaged, alpha = 0 (linear fill) -> 1 (perfect imputer):

| mechanism | p | bolt.declared | moirai.declared |
|---|---|---|---|
| mcar | 0.7 | 1.215 -> **1.219** (slope **+0.004**) | 1.092 -> **0.999** (slope −0.093) |
| block | 0.7 | 1.094 -> **1.086** (slope **−0.008**) | 1.348 -> **1.002** (slope −0.346) |
| mnar_high | 0.7 | 3.067 -> 2.213 (−0.854) | 2.132 -> 1.642 (−0.490) |
| mcar | 0.3 | 1.029 -> 1.029 (**+0.000**) | 0.999 -> 1.001 |
| block | 0.3 | 1.020 -> 1.024 (+0.004) | 1.069 -> 0.996 (−0.072) |

At p=0.7 under mcar and block -- the two mechanisms where the information is genuinely still
present -- **Moirai's declared path converts a perfect imputation into complete recovery
(1.002 and 0.999), while Bolt's declared path gains nothing (+0.004, −0.008).**

This is the causal evidence. The damage is not irreducible; handed to a model whose
declaration does not erase the content, the identical missingness is fully repairable. Under
mnar_high both move -- Bolt via the (loc, scale) channel that survives its rank-2 interface,
which is exactly where a rank-2 lever should still bite (heavy censoring biases the
statistics), and neither reaches 1.0 because censoring destroys information.

## The trade-off, demonstrated inside one experiment

Gradient-free range-constrained attack through the same declared paths (median damage,
>2x fraction):

| domain | moirai.declared | bolt.declared |
|---|---|---|
| ETTh1 block | x1.20, >2x on 11% | x1.04, >2x on 5% |
| ETTm1 block | x1.47, >2x on 31% | x1.05, >2x on 4% |
| weather block | x1.51, >2x on 36% | x1.03, >2x on 14% |
| ETTh1 mnar_high | x1.68, >2x on 36% | x1.26, >2x on 5% |
| ETTm1 mnar_high | x1.61, >2x on 38% | x1.15, >2x on 8% |

**The interface that lets a good imputer help you is the same interface that lets an
adversary move you.** Moirai gains full repair from a perfect imputation and is damaged
x1.2-1.7 by a poisoned one; Bolt gains nothing and is damaged x1.0-1.3. Same data, same
attack budget. This is no longer an inference across rounds -- it is one table.

## Caveats
- Moirai runs off-label (uni2ts `--no-deps`), through our wrapper rather than the library's
  inference path; its clean forecasts are sane and its plain-path sweep behaves normally.
- The attack is random search inside the observed range, so it lower-bounds what a
  gradient-based attacker could do to Moirai (we can differentiate through Bolt but not
  through Moirai's sampler, so the comparison is deliberately handicapped in Bolt's favour
  and the gap is still large).
- alpha=1 is an oracle imputer; the claim is about the ceiling and who may approach it.

## Artifacts
`run_s33_moirai.py`, `s33_results.json`, `s33_full.log`.

---

# Verification (V1/V2) — and a correction to how the flat slope must be read

## V1 — the wrapper matches the library's own inference path

Our direct `MoiraiForecast.forward` call vs the library's documented `create_predictor`
(GluonTS) path on identical clean windows:

| comparison | max relative diff | correlation |
|---|---|---|
| ours vs ours, two sampling seeds (the noise floor) | 1.775 | 0.9732 |
| ours vs `create_predictor` | 2.142 | 0.9746 |

Moirai's forecast is sampled, so exact equality is not expected; the ours-vs-theirs gap sits
**within the model's own sampling noise floor**, with a marginally higher correlation. The
off-label install does not fabricate the S33 result.

## V2 — the causal result survives patch size 32 and 64, and patch 16 needs a different reading

alpha-sweep on Moirai's declared path, p=0.7 (median relMSE at alpha = 0 / 0.5 / 1):

| patch | ETTh1 mcar | ETTh1 block | ETTm1 mcar | ETTm1 block |
|---|---|---|---|---|
| 16 | 0.996 → 1.000 | 1.091 → 1.006 | 0.948 → 1.013 | 0.973 → 1.003 |
| 32 | 1.185 → 1.007 | 1.807 → 1.013 | 1.143 → 1.004 | 1.255 → 1.007 |
| 64 | 1.148 → 0.992 | 1.676 → 1.004 | 1.138 → 1.001 | — |

**Correction to the write-up.** A flat slope means two different things and the paper must
distinguish them:
- **Bolt's declared path is flat from 1.215 / 1.094** -- there is damage, and a perfect
  imputation cannot remove it. This is the cap.
- **Moirai at patch 16 is flat from ~1.0** -- there is no damage to remove. The slope is
  trivially flat and says nothing about the interface.

At patch 32 and 64, where Moirai *is* damaged (1.15-1.81), its declared path converts a
perfect imputation into essentially complete recovery (→1.00). That is the comparison that
carries the causal claim, and it holds at both patch sizes.

`patch_size="auto"` could not be run: it raises a tensor-shape mismatch (64 vs 72) under our
context length in this off-label install. Reported as not-run rather than worked around.

## An unplanned finding: patch size is a second, independent robustness lever

Moirai's damage at alpha=0 under 70% block missingness falls sharply with smaller patches:
ETTh1 1.807 (patch 32) / 1.676 (64) → **1.091 (patch 16)**; ETTm1 1.255 / 1.524 → **0.973**.
At patch 16 the model is essentially *immune* to 70% block missingness.

The mechanism is the one S5 originally hypothesised for Chronos-Bolt's 16-point patches: a
missing point corrupts one token, and a smaller token carries less of the context with it.
This is a second architectural lever alongside the content channel, it is cheap to act on,
and it was not on the pre-registered list -- it fell out of a verification run and should be
labelled exploratory until tested directly.
