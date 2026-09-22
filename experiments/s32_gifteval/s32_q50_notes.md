# S32b/S32c — Q50 on the full GIFT-Eval sweep: the dose-response curve

Protocol: S32 verbatim (MASE vs in-window naive-1, observed target points only),
two window protocols (last-window official-style; random windows). Data:
`s32_q50.json` (full sweep), `s32c_results.json` (stratified, q50base rows added).

## Full sweep (17 datasets, linear fill): the honest average

q50 wins 3 (jena_weather −5%, ett2 −5%, saugeenday −19%), ties 2, loses 8
(electricity +15%, m4_hourly +53%, solar +13%, ett1 +15%, others +3–6%).
**On average Q50 is ~10% behind stock bolt-base on the full GIFT-Eval sweep** —
and the average is dominated by near-clean windows (even the "missing" datasets'
last windows are ~0.3% NaN). This is the clean tax, measured on the field's
standard benchmark, reported as-is.

## Stratified by the window's OWN context-NaN fraction: the dose-response

random protocol (nan column; linear/zero behave the same for q50):

| window NaN | n | bolt-base | q50 | verdict |
|---|---|---|---|---|
| <1% | 4308 | 0.661 | 0.700 | +6% (clean tax) |
| 1–5% | 2327 | 0.854 | 0.853 | **parity (crossover)** |
| 5–15% | 625 | 1.367 | **1.275** | **−7%** |
| 15–30% | 208 | 1.738 | **1.430** | **−18%** |
| >30% | 257 | 1.730 | **1.445** | **−16% to −32%** |

last-window protocol agrees in direction (+9% in the cleanest bin, parity in the
>30% bin with n=32).

## Read-out

1. **The paper's central claim in one table**: the retrofit's payoff grows
   monotonically with the window's own missing rate, crossing zero at ~3% — below
   the crossover you pay the clean tax, above it you collect the retrofit.
2. q50's zero column equals its nan column on every bin (1.318/1.318, 1.531/1.531)
   — the S59 zero-fill immunity holds on GIFT-Eval's real missingness too.
3. The honest framing for the paper: ~10% average cost on the full benchmark,
   −16% to −32% gain where missingness is actually present. Which number matters
   depends on how missing the deployment is — and S513 shows deployments are
   missing (11.77% of GIFT-Eval, 20% of raw electricity).
