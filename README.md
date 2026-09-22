# hole-in-input-repro

Reproduction package for **"Do Time Series Foundation Models Use Imputed Values?"**
(ICLR 2027 submission, under double-blind review).

The paper studies the *missingness interface* of time series foundation models
(TSFMs): what a model actually reads at a declared missing position, why that
choice bounds what any imputation can achieve, and how a short
continued-pretraining (CPT) recipe restores the use of repaired values.

> **Anonymity.** This repository is private until the camera-ready deadline.
> Please do not link it from the submission or any public channel while the
> paper is under review.

## What you can do with this package

1. **Verify every reported number** against the stored experiment results.
   The verifier scripts recompute each printed quantity from the raw result
   JSONs and assert the paper's value:
   ```bash
   cd experiments
   python verify_paper_numbers_v2.py   # 95 checks, writes claims_late_rounds.json
   python verify_main_text.py
   python verify_appendix.py
   python audit_prose_numbers.py        # lists prose numerals not covered by a claim
   ```
2. **Regenerate every paper figure and table** from the stored results:
   `paper/iclr2026/figures/make_*.py` read the result files under
   `experiments/` and emit the LaTeX/PDF assets used in the manuscript.
3. **Rerun the evaluations.** Each round directory (`experiments/s*`) keeps its
   run/eval scripts and DESIGN/notes. Corpora are external (GIFT-Eval,
   Monash/UTSD classics, Penmanshiel, METR-LA); fetch/prep scripts live in
   `data/` and in the rounds' notes. Stock checkpoints come from the official
   HF repos (amazon/chronos-bolt-*, amazon/chronos-2, Salesforce/moirai-2.0-R-small,
   and the other families surveyed in the paper).
4. **Rerun the training.** The CPT retrofit, the +5k no-missingness control,
   the within-family factorial arms and the trust-arena arms are trained by
   scripts in `s45_pretrain`, `s60_c2retrofit`, `s61_m2retrofit`,
   `s65_fairbudget`, `s69_arena`. Trained checkpoints are published on
   Hugging Face (see `ckpts/README.md`); the headline WiSE-FT arms can also be
   synthesized locally as `0.5 * CPT + 0.5 * stock`.

## Layout

- `experiments/` — experiment rounds: run/eval scripts, DESIGN/notes,
  and stored result JSONs. Round names are import paths: scripts use
  `sys.path.insert` with these exact relative names, so keep the layout and
  run scripts from the round or `experiments/` root. Exploratory rounds that
  no paper claim, verifier or figure references (s08, s17, s18, s21, s23,
  s24, s43, s44) are omitted.
  - Audit spine: `verify_*.py`, `audit_prose_numbers.py`, `find_number.py`,
    `claims_*.json` (values the verifiers assert).
  - `manifest.json` (repo root) maps each paper claim to its source
    checkpoint, result JSON and generator/verifier.
- `data/` — dataset fetch/prep scripts only (no corpora in git).
- `paper/iclr2026/figures/` — figure/table generators plus
  `dataset_counts.json` and `counts.tex`, so the audit scripts run unmodified.
- `ckpts/README.md` — trained-checkpoint inventory on Hugging Face, stock
  sources, and the WiSE-FT reconstruction recipe.

## Results policy

- Per-window arrays (`"w"` and similar) are stripped from the largest result
  JSONs (the s48 family) to keep the repository light; every aggregate the
  paper relies on (median/mean/closure inputs) is preserved. Window-paired
  bootstrap recomputation therefore needs the full arrays, which will be
  published alongside the checkpoints.
- `experiments/s20_textprompt/s20_results.json` (236 MB) is excluded from git
  via `.gitignore`; it is not referenced by any paper claim.
- `s7_attrib_results.json` and `s8_results.json` keep their full per-window
  arrays.

## Environment

`requirements.txt` (torch 2.6.0+cu124, transformers 5.14.1,
chronos-forecasting 2.3.1, gluonts 0.17.0, ...). Training used 8xA800;
most evaluation scripts run on a single GPU.

## License

Apache License 2.0 (see `LICENSE`). Trained checkpoints are derived from
publicly released weights (Chronos: Apache-2.0; Moirai: Apache-2.0); their
notices travel with the checkpoint repos.
