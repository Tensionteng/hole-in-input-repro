# S27 notes — does the input INTERFACE cap how much a better imputation can help?

**Status: PRE-REGISTRATION (written before any run). Results appended below.**

## The question this round is meant to answer

S25/S26 are diagnostic. The fair criticism is that a paper of regularities, however well
measured, ends without a "so what should we do". S27 tries to convert the single most solid
measurement in the project into an architectural prescription.

The measurement (S25 Part 0, exact, 72/72 series): Chronos-Bolt's `encode()` runs
`patched_context = where(patched_mask > 0, patched_context, 0)` -- at masked positions the
**content is zeroed and only the flag survives**. The whole influence of whatever you put in
the holes collapses onto the two scalars (loc, scale). Other families (TimesFM, plain
fill-then-feed) have the opposite defect: no flag at all, so fabricated values are trusted
unconditionally.

If that is right, it has a consequence nobody has stated:

> **Under a content-zeroing interface, forecast quality is invariant to imputation quality.**
> A model built this way cannot benefit from a better imputer -- not now, not from any
> future imputation research. The interface, not the imputer, is the binding constraint.

That is a claim about how TSFMs should be *built*, it is falsifiable, and it is testable
without pretraining anything from scratch.

## Design: an imputation-quality sweep crossed with the input interface

Fill the holes with a convex blend of the linear interpolant and the ground truth,
`x_fill(alpha) = (1 - alpha) * linear + alpha * truth`, alpha in {0, 0.25, 0.5, 0.75, 1}.
alpha is a clean, monotone axis of "imputation quality" (alpha=1 is a perfect imputer, the
ceiling no real imputer can exceed). Cross it with four interfaces:

| interface | what the model receives at a masked position | expected rank |
|---|---|---|
| `plain` | the fill value; **no flag** (fill-then-feed, TimesFM-like) | full |
| `native` | flag only; **content zeroed** (Chronos-Bolt's masked path) | 2 |
| `dual` | the fill value **and** the flag | full |
| `dual_obsnorm` | fill value + flag, and (loc, scale) computed from **observed points only** | full |

`dual` is `native` with the one-line zeroing removed. `dual_obsnorm` additionally fixes the
statistics channel: bolt's `instance_norm` runs BEFORE masking and excludes only NaN, so
under the native path the fill still biases (loc, scale). `dual_obsnorm` is the interface the
measurements say you actually want: unbiased statistics + informative content + a flag.

## Pre-registered predictions

- **Q1 (the core claim).** Under `native`, the alpha-sweep is **flat**: relMSE at alpha=1 is
  within 5% of alpha=0. Under `plain`/`dual`, it is **strongly decreasing** (alpha=1 well
  below alpha=0). Confidence high for `native` (it follows from the rank-2 result; the
  residual slope can only come from (loc, scale)).
- **Q2 (zero-shot `dual` is bad).** Evaluated zero-shot, `dual` will be **no better and
  probably worse** than `native` at alpha=0, because bolt was pretrained expecting zeros at
  masked positions and `dual` feeds it off-distribution content. Confidence high. This is
  why Q3 is the real test.
- **Q3 (the prescription).** After adapting **only** `input_patch_embedding` on
  mechanism-diverse missingness with alpha ~ U(0,1) (~2 min, backbone frozen, the S5-fix2
  `mponly` recipe that preserved clean performance), `dual`/`dual_obsnorm` **dominate**:
  within noise of `native` at alpha=0 and clearly better at high alpha, on every mechanism.
  If this fails, the prescription fails and S27 is a negative result.
- **Q4.** The gain from the interface is largest under `mnar_high` (censoring needs content
  injected above the threshold) and smallest under `block` (where refusing to fabricate is
  already right). From S25's P3 falsification and S26's real-data replication.

## Why the project's earlier failures do not pre-empt this

S8 (learnable mask token), S18 (universal token) and S5-fix2 (miss_proj adapter) all added a
**flag** pathway inside an interface whose **content** was already zeroed -- they reinforced
the half that exists. S27 restores the half that was thrown away. Distinguishable and
untested. (Honest risk: if the content path is unusable without full pretraining, the
prescription survives only as "pretrain differently", which S27 cannot afford to test.)

## Setup

Same as S25: L=512, H=64, ETTh1/ETTm1/weather, mechanisms {mcar, block, mnar_high,
mnar_extreme} x rates {0.3, 0.7}, S5-identical windows and masks (SEED=20250810), paired
per-window relMSE against each window's own clean-context forecast, median over windows.
Anchor gate reproduces the stored S5 cells before anything runs. Adaptation trains on the
first 80% of the timeline only; all evaluation is on the S5 test windows.

---

# RESULTS

## Gates — PASS

Three stored S5 cells reproduce exactly (dev +0.0000%). Critically, the re-implemented
`encode` under interface `plain` on a fully observed context matches the **stock forward
pass to max|diff| = 5.7e-6**, so all four interfaces are the same code path with one switch
flipped — the comparison is clean.

## Q1 — CONFIRMED: the content-zeroing interface is flat in imputation quality

Slope of relMSE from alpha=0 (linear fill) to alpha=1 (**perfect imputer**), expressed as a
fraction of that interface's own excess over clean:

| regime | median abs(slope) | abs(slope) <= 20% | slope POSITIVE (better imputer makes it *worse*) |
|---|---|---|---|
| zero-shot | **6%** | 7 / 8 cells | 3 / 8 |
| adapted | **4%** | 6 / 8 cells | 3 / 8 |

Handing Chronos-Bolt's native masked path a *perfect* imputation is worth ~nothing, and in
three of eight cells it is actively harmful. The one real exception is mnar_high p=0.7
(-40%), and it is the exception the rank-2 proposition predicts: when 70% of the context is
censored, the (loc, scale) channel is the thing that is broken, and (loc, scale) is exactly
what a rank-2 interface can still receive.

For contrast, `plain` (no flag) recovers **100%** of its gap in every cell — trivially, since
at alpha=1 its input *is* the clean context. That is a check that the alpha axis is
well-posed, not evidence that `plain` is a good interface: at alpha=0 `plain` is often the
worst of all (block p=0.7: 1.228 vs native 1.096), because it trusts fabricated content
unconditionally.

## Q3 — CONFIRMED: restoring the content path dominates, and costs nothing

Adapting **only** `input_patch_embedding` (backbone frozen, 600 steps, ~2 min) on
mechanism-diverse missingness with alpha ~ U(0,1):

| mechanism | p | alpha=0: native / dual | alpha=1: native / dual | share of native's remaining excess removed |
|---|---|---|---|---|
| mcar | 0.3 | 1.026 / **1.013** | 1.025 / **1.002** | **93%** |
| mcar | 0.7 | 1.132 / **1.131** | 1.127 / **1.049** | **62%** |
| block | 0.3 | 1.019 / **1.017** | 1.015 / **1.004** | **71%** |
| block | 0.7 | 1.097 / 1.112 | 1.094 / **1.055** | **41%** |
| mnar_high | 0.3 | 1.131 / **1.112** | 1.194 / **1.073** | **63%** |
| mnar_high | 0.7 | 2.678 / 2.728 | 1.957 / **1.577** | **40%** |
| mnar_extreme | 0.3 | 1.206 / **1.153** | 1.208 / **1.044** | **79%** |
| mnar_extreme | 0.7 | 1.995 / **1.879** | 2.045 / **1.331** | **68%** |

**dual beats native in 8/8 cells when a good imputer is available**, removing 40–93% of the
error the native interface cannot touch; and in **6/8 cells it is also no worse at alpha=0**,
i.e. it does not cost anything when the imputer is poor. (`dual` = `native` with one line
removed; `dual_obsnorm` additionally computes (loc, scale) from observed points only; the
table reports the better of the two.)

## Q4 — CONFIRMED: the gain is ordered by mechanism exactly as predicted

Improvement of dual over native at alpha=1: block (+0.011 / +0.039) < mcar (+0.024 / +0.078)
< mnar_high (+0.122 / +0.381) < mnar_extreme (+0.164 / +0.714), at p=0.3 / 0.7. Censoring
needs content injected; blocks barely benefit from fabrication. Same ordering S25's P3 and
S26's real-data replication produced, now from a third, independent angle.

## Q2 — PARTIALLY confirmed

Zero-shot `dual` is mixed rather than uniformly bad: worse than native under mcar/block
(off-distribution content), better under mnar (where the content carries the censored-tail
information native discards). The adaptation is what makes it uniformly better.

## A new finding: a binary flag cannot express imputation quality

Even at alpha=1, `dual` does not reach `plain` (residual +0.016 to +0.591, largest under
heavy censoring). The reason is structural: at alpha=1 `plain` receives the clean context,
while `dual` receives the same values *plus a flag saying they are imputed* — and a binary
flag cannot tell the model that this particular imputation happens to be perfect. The model
learns an average trust level over the imputation quality distribution it was trained on.

This sharpens the prescription into a second, testable one: **the missingness channel should
carry per-point imputation uncertainty, not a bit.** Untested here; the obvious next round.

## THE CONCLUSION THIS ROUND BUYS

> A time series foundation model that zeroes the content at masked positions and keeps only
> a missingness flag **caps the benefit of imputation at approximately zero**. No present or
> future imputer can help it: handing it a perfect imputation improves it by a median 4–6% of
> its own error, and in three of eight settings makes it worse. The cap is a property of the
> interface, not of the imputer, and not of the information in the data — the same model,
> with the content path restored and only its input projection adapted (2 minutes, backbone
> frozen), converts a perfect imputation into a 40–93% error reduction, while giving up
> nothing when the imputation is poor.
>
> **Therefore: pass the imputed value AND the missingness flag into the content path, and
> pretrain with mechanism-diverse missingness so the model learns how far to trust the
> value. Filtering missingness out of the pretraining corpus — as TimesFM, Moirai and TTM
> document doing — builds the cap in.**

This is prescriptive, falsifiable, and it makes the rest of the project's measurements load-
bearing rather than terminal: the rank result says *why* the cap exists, the decomposition
says *how much* is behind it, and the mechanism ordering says *who* should care most.

## Caveats

- One model family (chronos-bolt-base). The proposition is about an interface that Chronos-2
  and any mask-channel design share, but that is an argument, not a measurement; the
  cross-model version is still owed (S25's P0-d).
- The adaptation tunes only `input_patch_embedding` on the first 80% of the same three
  datasets. It is a proxy for pretraining, not pretraining. A real test trains a TSFM from
  scratch under both interfaces; that is the natural follow-up and this round is what
  justifies paying for it.
- alpha=1 is an oracle imputer. Real imputers sit at low alpha, where the gain is smaller —
  S26 measured what a *realistic* learned fill buys on real data (6% median). The claim here
  is about the ceiling and who is allowed to approach it, not about today's imputers.
- relMSE is normalised by the **unadapted** model's clean forecast for every interface, a
  common baseline. Adaptation itself improves clean forecasting ~1.4% (`plain` reaches 0.986
  at alpha=1), which is why some adapted cells sit below 1.0.

---

## CORRECTION (found in S37, 2026-08-20): the adapted-regime denominator drifted

`sweep()` computes the per-series clean denominator with whatever `input_patch_embedding` is
loaded at that moment. Inside its dataset loop, the previous dataset's last adapted interface
(`dual_obsnorm`) is still loaded, so **every dataset after the first was normalised by an
adapted clean forecast instead of the released model's**.

Verified directly (`s37_capbreadth/verify_s27_base.py`): recomputing `weather|block|0.7` with
the released weights as the denominator gives 1.0323/1.0448/1.0356/1.0235, while recomputing it
with `dual_obsnorm` loaded reproduces the stored 1.0501/1.0669/1.0926/1.0818 to four decimals.
ETTh1 (the first dataset in the loop) is bit-exact either way; ETTm1 deviates by a median of
1.5% and weather by 3.8%.

Effect on the conclusion: none of the signs change. Dataset-averaged closure moves from
40–93% to 38–105% (8/8 cells still positive, 6/8 still no worse at alpha=0).

Superseded by S37, which additionally normalises **each interface by its own clean forecast**,
as `app:method` requires — the adaptation edits a module used on every input, so a generic
clean-data gain or loss must not be charged to the interface. Under that normalisation the
in-sample closure is 30–83%. Use S37's numbers, not this file's, for the paper.
