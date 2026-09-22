# S30 notes — the fill-reachable set across model families (the P0-d owed since S25)

Probe: the gradient-free permutation test from S25 Part 0. Permuting the fill values AMONG
the missing positions preserves their multiset -- hence the context's mean and variance --
while destroying which value sits where. `ratio = perm-change / redraw-change` is ~1 if the
model uses the fill's CONTENT and ~0 if only its summary statistics reach the forecast.
3 datasets x 3 mechanisms x 60 windows, rate 0.3.

| family | plain fill (no declaration) | declared-missing path | ratio |
|---|---|---|---|
| chronos-bolt-base | 0.907 | explicit mask / NaN | **0.0000 / 0.0000** |
| chronos-2 | 0.898 | NaN | **0.0000** |
| chronos-t5-small | 0.950 | NaN | **0.0000** |
| timesfm-2.5-200m | 0.863 | NaN | **0.0000** |
| moirai-1.1-R-base | 0.965 | observed-mask | **0.9739** |

## Reading: four distinct behaviours behind "handles missing data"

1. **Discard the content** -- Chronos-Bolt (mask and NaN), Chronos-2 (NaN), Chronos-T5 (NaN).
   Whatever you put in the holes becomes invisible; for bolt only (loc, scale) leak, which is
   the rank-2 result of S25 Part 0, now confirmed to generalise across the Chronos family.
2. **Overwrite with the model's own imputer** -- TimesFM. Passing NaN triggers its internal
   `strip_leading_nans` + linear interpolation, so your fill is *replaced*, not discarded.
   The ratio is 0 for a different reason, and the consequence is different: through the NaN
   path you can never supply a better imputation, and through the plain path you cannot
   declare at all. TimesFM users have no way to hand the model a good fill *and* tell it the
   values are imputed.
3. **Accept the declaration but keep using the values** -- Moirai. The observed-mask is
   honoured for its own purposes, but permuting the content still moves the forecast almost
   as much as redrawing it (0.9739). Moirai is the one family where declaring a hole does not
   hide what you wrote in it.
4. **No declaration** -- every family, under plain fill-then-feed: content fully used
   (0.86-0.97).

## Why this matters

S25's rank result, S27's imputation cap, S29's attack-surface ordering and S31's
cross-channel gating were all measured on Chronos-Bolt. This round shows the *structure* is a
property of the family's input convention, not of bolt, and that the four conventions in the
wild differ in ways that have direct consequences:

- **Chronos family**: you can refuse to fabricate (good under blocks, per S31's 95% repair),
  but you can never benefit from a better imputer through the declared path (S27's cap).
- **TimesFM**: no lever in either direction -- its imputer is the only one you get.
- **Moirai**: a better imputer *can* reach it even when you declare, so S27's cap should not
  apply -- and by the same token its attack surface stays full-rank even when the hole is
  declared. **Predicted, not yet measured**: Moirai should show a downward alpha-sweep where
  bolt is flat, and should be attackable through a declared hole where bolt is immune. Both
  are direct follow-ups and would turn this table into a causal claim.

## Caveats

- Moirai is run off-label (uni2ts installed `--no-deps`); its forward is unpatched and its
  clean forecasts are sane, but the wrapper is ours, not the library's inference path.
- One checkpoint per family, patch size and context fixed per family defaults.
- The ratio conflates "content ignored" with "content overwritten"; the two are separated
  above by reading the source, not by the probe.

## Artifacts
`run_s30_crossmodel.py`, `s30_results.json`, `s30_full.log`.
