# S5 Track A — repair notes

Grid: mechanisms {mcar, block, mnar_high, mnar_extreme} × p ∈ {0.1,0.3,0.5,0.7} ×
{ETTh1, ETTm1, weather}; same windows/seeds as the main S5 runs. Numbers below are
dataset-avg relMSE (MSE / clean MSE) at p = 0.1/0.3/0.5/0.7 unless stated.

## Fix 0 — what "doing nothing" means (feed raw NaN)

- **bolt**: native NaN support is real and sophisticated. `chronos_bolt.py`: mask from
  `~isnan(context)` (l.280); InstanceNorm uses **nanmean / nan-std over observed
  positions only** (l.111-112) — mask-aware scaling is native; masked values become 0 in
  normalized space (= observed-mean fill, l.298) and the binary mask is concatenated into
  the patch-embedding input (l.300); a patch attends if ≥1 point is observed (l.303).
  Empirically (bolt): mcar 1.01/1.01/1.04/1.13 (≈ linear fill); **block 1.01/1.01/1.05/1.09
  — beats linear (0.99/1.05/1.18/1.36)** on long gaps, where interpolation fabricates
  misleading straight segments but mean-fill + mask flags let the model discount them;
  mnar_high 1.00/1.19/1.69/3.39 — worse than linear (1.00/1.12/1.36/2.17): mean-filling
  censored highs biases forecasts down; mnar_extreme 1.06/1.33/1.42/1.70 ≈ linear.
- **timesfm**: silently repairs for you — every input goes through `strip_leading_nans` +
  `np.interp` linear interpolation (`timesfm_2p5_base.py` l.176). Empirically nan ≡ our
  explicit linear fill: max relMSE difference 0.017 across the whole 48-config grid.
  "Doing nothing" for timesfm *is* linear interpolation.

## Fix 1 — black-box observed-stats rescale: fill × mean(|x_observed|)/mean(|x_filled|)

- **Under mcar it matches the oracle.** bolt zero+rescale 1.14/1.94/1.88/10.43 vs
  oracle-rescale 1.14/1.94/1.87/10.43 — identical to 2 decimals, no oracle knowledge.
  Recovery of zero-fill damage at p=0.5: 82.3/78.7/82.3% (ETTh1/ETTm1/weather); p=0.7:
  24–38% (value error dominates); p≤0.3 weather negative (artifact: zero-fill ≈ clean
  there, denominator ≈ 0). Same for timesfm (mcar p=0.5: 6.07 → 3.01).
- **Under block/mnar it backfires** (observed stats are biased by informative
  missingness): bolt mnar_high zero+rescale 3.80/7.42/13.82/30.71 vs plain zero
  3.73/6.31/8.91/11.32 — worse everywhere, like the oracle probe before it.
- linear+rescale is the expected no-op under mcar (≤1–2% change) and mildly harmful under
  mnar_high (3.06 vs 2.17 at p=0.7, bolt) — the biased observed mean-|x| shrinks the
  context below its true scale.

## Fix 2 — white-box bolt surgery (SHIPPED)

Zero-init `miss_proj: Linear(1, 768)` added to patch embeddings via a forward hook on
`input_patch_embedding` (mask flags already in its input); backbone frozen; miss_proj +
input_patch_embedding trained 480 steps × 256 series (native pinball loss, lr 1e-3,
random mcar/block/mnar masks at rate ~U(0.05,0.8), train split only; ~2 min on one A800).

- vs its zero-init (fix0-nan): improves the worst cells — mnar_high p=0.7: 3.39 → 2.85
  (ETTm1 5.18 → 3.73); block/weather p=0.7: 1.10 → 0.87. Flat-to-worse elsewhere.
- vs the best black-box per mechanism: **never wins on ETT** (linear wins both mnar
  mechanisms, nan wins block). The weather "wins" are an in-domain-adaptation artifact:
  fine-tuning input_patch_embedding on train-split weather improved *clean* weather MSE
  by 22% (3624 → 2833); normalized by its own clean, weather fix2 ≈ nan. ETT clean
  degrades +3%/+8%.
- **miss_proj-only ablation (input-proj frozen) — the better variant.** Clean preserved
  (ratios 0.997/1.001/1.047 on ETTh1/ETTm1/weather — all the clean drift came from
  input-proj fine-tuning) and it dominates the full adapter under missingness
  (dataset-avg relMSE): block p=0.7: 1.09 vs full-adapter 1.31; mnar_high p=0.5: 1.50 vs
  1.81, p=0.7: 2.83 vs 2.85; mcar p=0.7: 1.13 vs 1.20. Versus fix0-nan it is a genuine
  missingness-specific gain: mnar_high 1.69/3.39 → 1.50/2.83 at p=0.5/0.7 with no clean
  cost. It still does not beat plain linear under mnar (2.17 at p=0.7).
- Training caveat: the native pinball loss explodes on tiny-std channels (batch losses
  spike to 1e7–4e7 on weather); updates are effectively AdamW-normalized noise on those
  batches. A clipped loss might do better; the qualitative conclusion — a small adapter
  cannot recover censored information — is unlikely to change.

## Bottom line (bolt, p=0.7, dataset-avg relMSE)

| mechanism    | zero  | linear | nan (fix0) | zero+resc (fix1) | adapter (fix2) | adapter mp-only |
|--------------|-------|--------|------------|------------------|----------------|-----------------|
| mcar         | 13.74 | 1.07   | 1.13       | 10.43            | 1.20           | 1.13            |
| block        | 5.93  | 1.36   | **1.09**   | 10.36            | 1.31           | 1.09            |
| mnar_high    | 11.32 | **2.17** | 3.39     | 30.71            | 2.85           | 2.83            |
| mnar_extreme | 12.13 | **1.58** | 1.70     | 19.49            | 1.68           | 1.68            |

Practical guidance: never zero-fill (fix1 rescues it only under mcar); linear
interpolation is the safest default; for bolt under contiguous gaps, feeding NaN natively
is better than interpolating; no repair recovers information censored by MNAR — the
model must be trusted-region-aware or the missingness mechanism itself addressed.

Artifacts: `run_s5_fix.py`, `analyze_s5_fix.py`, `s5_fix_results.json`,
`s5_fix_bolt_adapter.pt` (+ `_mponly`), `s5_fix.png`, logs `s5_fix_*.log`.
