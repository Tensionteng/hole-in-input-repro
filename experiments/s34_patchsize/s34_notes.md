# S34 notes — is patch size a lever on missingness robustness, and why?

Moirai is the one family where patch size is an **inference-time** choice on identical
weights (pretrained over `patch_sizes = [8, 16, 32, 64, 128]`), which makes it a clean
controlled variable. 3 datasets x 4 mechanisms x 4 rates x 5 patch sizes, 120 windows,
declared (observed-mask) input, paired relMSE against **that patch size's own clean forecast**.

## Scorecard

| # | prediction | outcome |
|---|---|---|
| P1 | relative degradation falls monotonically with smaller patch under block | **PARTIAL** — a step, not a trend: 1.029/1.040 at patch 16/8 vs 1.281-1.371 at 32/64/128, but not monotone within either group |
| P2 | the effect is much weaker under MCAR | **CONFIRMED** — mcar spans 0.998-1.113 (1.11x) vs block 1.029-1.371 (1.33x) and mnar_high 1.415-2.585 (1.83x) |
| P3 | clean accuracy is NOT monotone in patch | **CONFIRMED, and it is the crux** — clean MSE 2.648 / 2.540 / 1.903 / **1.727** / 2.080 for patch 8/16/32/64/128. Large patches are *better* forecasters on clean data |
| P4 | degradation collapses onto the corrupted-token fraction | **FALSIFIED** |

## P4 falsified: no token-level statistic explains the effect

R^2 of relMSE against each candidate (quadratic fit, 240 cells):

| predictor | overall | mcar | block | mnar_high | mnar_extreme |
|---|---|---|---|---|---|
| corrupted-token fraction | 0.080 | 0.045 | 0.336 | 0.477 | 0.264 |
| partially-corrupted tokens | 0.045 | 0.033 | 0.328 | 0.309 | 0.387 |
| fully-missing tokens | 0.002 | 0.016 | 0.053 | 0.006 | 0.131 |
| **absolute** count of intact tokens | 0.055 | 0.041 | 0.228 | 0.190 | 0.164 |
| raw missing rate | 0.153 | 0.368 | 0.191 | 0.426 | 0.133 |

The proposed mechanism -- "a gap corrupts G/P+1 of L/P tokens, so smaller patches corrupt a
smaller fraction" -- does not survive. Four different token-level quantities were tested and
none predicts the degradation; the raw missing rate does better than all of them. **The patch
effect is real but unexplained**, and it is recorded here as unexplained rather than
attached to a story that the data rejects.

## The result that actually matters: it is a trade-off, and it usually does not pay

Relative degradation made small patches look dramatically better. Converting to **absolute**
median MSE (that patch's clean error x its relMSE) reverses most of it:

| mechanism | p | best patch (absolute) | patch 16 | patch 64 |
|---|---|---|---|---|
| mcar | 0.7 | **64** | 2.536 | **1.901** |
| block | 0.7 | **64** | 2.614 | **2.368** |
| mnar_high | 0.5 | 32 | 2.777 | 2.831 |
| **mnar_high** | **0.7** | **16** | **3.594** | 4.465 |
| mnar_extreme | 0.7 | 32 | 2.360 | 2.392 |

**Patch 64 is the best absolute choice almost everywhere.** The robust setting only pays off
under heavy value censoring: at mnar_high p=0.7, switching from patch 64 to patch 16 buys
**19%** (4.465 -> 3.594); at p=0.5 and for mnar_extreme the gain is marginal. Under mcar and
block, the ~47% clean-accuracy penalty of a small patch swamps its robustness advantage.

**Deployable statement:** patch size is a genuine, retraining-free robustness knob, but it is
worth turning only when you are in a heavy value-censoring regime -- which is exactly the
regime the mechanism detector is for.

## A methodological warning this round earned

Own-clean normalisation is the right statistic when comparing *one configuration* under
different corruptions -- it is what the rest of this project does. It is **misleading when
comparing different configurations**, because each is normalised by a different denominator.
Patch 16 looked 25-45% more robust than patch 64 and is nonetheless worse in absolute terms
in 14 of 16 cells. Any cross-configuration claim in the paper must be stated in absolute
terms or with both.

## Caveats
- Moirai only (the one family exposing patch size at inference); off-label `--no-deps`
  install, wrapper validated in S33's V1 against the library's own `create_predictor` path.
- `patch_size="auto"` could not be run (tensor-shape mismatch under our context length).
- The mechanism remains open; a fixed-patch ablation inside pretraining would settle it and
  is beyond this round.

## Artifacts
`run_s34_patchsize.py`, `s34_results.json`, `s34_full.log`.
