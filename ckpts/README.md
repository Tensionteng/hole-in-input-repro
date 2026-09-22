# Trained checkpoints

Trained checkpoints are published on Hugging Face (the namespace is withheld
during double-blind review and restored at camera-ready). Stock (vanilla)
checkpoints are **not** mirrored; use the official releases:

- `amazon/chronos-bolt-{tiny,mini,small,base}`, `amazon/chronos-t5-*`,
  `amazon/chronos-2`
- `Salesforce/moirai-1.1-R-{small,large}`, `Salesforce/moirai-2.0-R-small`
- `google/timesfm-2.5-200m-pytorch`, `autonlab/TiRex`,
  `ibm-granite/granite-timeseries-flowstate-r1` (and the remaining census
  families listed in the paper's Table 2)

## Inventory (arm → paper claim)

| checkpoint | source path (training round) | serves |
| --- | --- | --- |
| bolt recipe CPT (`arm_P8_base`) | `s45_pretrain/s45_ckpt` | Table "Relative degradation after CPT" (bolt CPT row is Q50, below) |
| bolt Q50 (headline) | `0.5*arm_P8_base + 0.5*chronos-bolt-base` | bolt CPT row; absolute win rates (74.0/62.1/61.1%) |
| bolt +5k control (`arm_P4_base`) | `s45_pretrain/s45_ckpt` | bolt Control row (+4.3% clean, 9.80 zero fill) |
| c2 CPT + W50, seeds 20260901–03 | `s60_c2retrofit/s60_ckpt` | c2 CPT row; boundary probes; full-outage analysis |
| m2 CPT + W50, seeds 20260901–03 | `s61_m2retrofit/s61_ckpt` | Moirai 2.0 CPT row; 54.3% closure |
| c2/m2 +5k controls (+W50 variants) | `s65_fairbudget/s65_ckpt` | fair-budget table; c2/m2 Control rows (+5.2%/+2.5%) |
| bolt-tiny factorial arms A–D, 3 seeds | `s45_pretrain/s45_ckpt` | the 2x2 factorial table (88.7 / -47.6 / 72.7% closure) |
| bolt-tiny recipe ladder arms (9ds, 3 seeds) | `s45_pretrain/s45_ckpt` | declare-only 2.7% / fill-diverse 94.4% / full 97.6% |
| 205M capacity arms | `s45_pretrain/s45_ckpt` | "capacity alone does not open the content path" |
| input adapters (plain/native/dual) | `s27_interface/s27_ckpt` | quality-sweep adapter results (53% / 73% closure) |
| trust-arena arms A–D (final steps) | `s69_arena/s69_ckpt` | the four-quadrant arena figure |

WiSE-FT synthesis: any `Q50`/`W50` arm is `0.5 * CPT + 0.5 * stock` applied
tensor-wise (equation 3 in the paper). SAITS/BRITS imputer weights were not
retained after their runs; retrain with `s42_end2end` scripts (a few GPU
hours).

Derived from public Apache-2.0 weights (Chronos, Moirai); notices are carried
in each checkpoint repo.
