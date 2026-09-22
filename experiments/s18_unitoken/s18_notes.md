# S18 — universal [MASK] token (cross-dataset corpus)

Question: can ONE gap-prior [MASK] token, trained on a cross-dataset corpus,
escape the in-domain limitation found in S8 (transfer matrix diagonal-dominant;
the weather token does not transfer to ETT — worse than nan there)?

Eval pipeline is S8 Exp3 reused verbatim (`run_s8_masktoken.py` primitives:
same 300 test windows, same 2 mask seeds, same controls, same batch sizes,
L=512, H=96, median quantile, last-20% test region). All relMSE = MSE /
paired native-clean MSE per dataset.

## Design

- **uni**: one 768-d zero-init token trained on an equal-parts mixture of the
  ETTh1 + ETTm1 + weather train splits (first 70% each). Per batch of 256,
  element b draws its series from DS[b % 3] — exactly equal thirds
  (86/85/85). Everything else identical to S8 Exp2: backbone frozen, clipped
  pinball clip=100, AdamW lr 1e-3, 1000 steps, augmentation 50% mcar /
  50% block(24), p~U(0.05,0.8). 3 training seeds, S8b convention: base
  101/202/303 -> actual = base + 31 (132/233/334; DS_IDX=0 for the single
  mixed token).
- **uniaug**: same, plus per-window scale jitter f~exp(U(-ln2,+ln2)) and a
  linear-trend perturbation (total drift d~U(-2,+2) x std(context) across the
  L+PRED window). Perturbations come from a SEPARATE rng stream
  (seed+777000), so uni and uniaug see identical windows/masks per step —
  the comparison isolates the augmentation.
  - **Design caveat, recorded up front** (implementation-risk note required
    by the pre-registration): bolt instance-normalises each series and the
    clipped pinball loss is computed in normalised units, so pure scale
    jitter cancels almost exactly (verified numerically in the smoke test:
    max|diff| of normalised values under f=3.7 is 4.8e-07). The trend
    perturbation is the only operative ingredient; scale jitter is inert and
    is kept only for completeness. This is a known limitation of the variant,
    not a bug.
- **Eval grid**: block p in {0.1..0.7} x {ETTh1, ETTm1, weather} (full Exp3
  grid); no-gain check cells mcar@0.3 and mnar_high@0.7 (one rate each, as
  pre-registered).
- **Methods**: anchor controls nan / linear / mponly / tok_ETTh1 / tok_ETTm1
  / tok_weather (loaded from `s8_masktoken_{ds}.pt`); new uni_s{101,202,303},
  uniaug_s{101,202,303}.

## Pre-registered judgement (fixed BEFORE any results)

Block rate-avg relMSE per test ds; uni = mean over the 3 seeds;
tok_own = matched S8 in-domain token (original single-seed).

- **SUCCESS**: for every test ds, uni <= nan AND uni - tok_own <= 0.02
  ("nowhere bad, within noise of the in-domain token").
- **PARTIAL SUCCESS**: for every test ds, uni <= nan (no out-of-domain
  collapse), but some ds has uni - tok_own > 0.02 (gap to in-domain remains).
- **FAILURE**: some test ds has uni > nan (still collapses out-of-domain).

Same criteria applied to uniaug. Check cells (sanity, not success criteria):
mcar@0.3 expects |uni - nan| <= 0.02 (S8: token is a no-op under mcar);
mnar_high@0.7 expects uni <= nan + 0.10 (S8: no transfer to MNAR; slight
degradation tolerated, collapse would fail the check).

Anchor gate (before training): the 6 control methods re-evaluated on all 27
cells/ds; every cell must be within +-5% of `s8_results.json` (S8b showed the
pipeline is bit-deterministic, so ~0.0000% expected).

## Anchor gate — result

**PASSED, bit-identical.** The 6 control methods x 27 cells/ds (162 cells) +
3 clean baselines re-evaluated through this harness vs `s8_results.json`:
max |dev| = 0.0000% for every method and every clean baseline (worst cell
0.0000% << 5%). Full Exp3 replication, as expected from S8b's determinism
check.

## Training

6 tokens trained (255-262 s each on one A800). Actual seeds = base + 31
(S8b convention, DS_IDX=0). Final clipped losses ~34-36, token norms
0.57-0.87 — both in S8's range (33-37 / 0.5-1.6). Raw unclipped loss spikes
to 1e4-2e5 (the known S5 explosion, clipped as designed).

| method | actual seed | final \|tok\| | final clipped loss |
|--------|-------------|---------------|--------------------|
| uni_s101 | 132 | 0.655 | 34.20 |
| uni_s202 | 233 | 0.649 | 34.78 |
| uni_s303 | 334 | 0.865 | 35.52 |
| uniaug_s101 | 132 | 0.572 | 34.71 |
| uniaug_s202 | 233 | 0.697 | 35.57 |
| uniaug_s303 | 334 | 0.628 | 36.03 |

## Main table — block transfer matrix

Rate-avg relMSE over p in {0.1..0.7}; uni/uniaug = mean +- sd over the 3
seeds; per-seed values in `s18_results.json` (report section).

| test ds | nan | linear | mponly | tok_ETTh1 | tok_ETTm1 | tok_weather | uni | uniaug |
|---------|-----|--------|--------|-----------|-----------|-------------|-----|--------|
| ETTh1 | 1.0030 | 1.2333 | 1.0147 | **0.9662** | 0.9822 | 1.0575 | 1.0072+-0.0123 | 1.0308+-0.0078 |
| ETTm1 | 1.0489 | 1.1569 | 1.0656 | 1.0362 | **0.9999** | 1.0613 | 1.0270+-0.0107 | 1.0370+-0.0038 |
| weather | 1.0520 | 1.0129 | 1.0482 | 1.0210 | 1.0964 | **0.8625** | 0.9506+-0.0055 | 0.9630+-0.0093 |

Deviation of uni from nan per ds (95% t-CI over seeds, t=4.303):
ETTh1 +0.0042 [-0.0334, +0.0417]; ETTm1 -0.0219 [-0.0543, +0.0105];
weather -0.1014 [-0.1181, -0.0847] (significant).
Gap uni - tok_own: ETTh1 +0.0410 [+0.0035, +0.0786];
ETTm1 +0.0271 [-0.0053, +0.0595]; weather +0.0881 [+0.0713, +0.1048].

Per-rate: on ETTh1 uni tracks nan almost exactly (within +0.002 at
p<=0.4, +0.01..+0.02 at p>=0.5), while tok_own is 0.95-0.98 throughout;
on weather uni sits strictly between nan and tok_own at every rate
(e.g. p=0.3: own 0.8184 < uni 0.9147 < nan 1.0311); on ETTm1 uni is
marginally below nan at all rates.

Reading, per dataset:
- **ETTh1**: uni ~= nan (2/3 seeds slightly below, s303 +0.021 above; mean
  +0.004, CI straddles 0). The ETTh1-token's in-domain gain (-0.037) is
  entirely diluted away.
- **ETTm1**: uni keeps ~45% of the in-domain gain (-0.022 of -0.049).
- **weather**: uni keeps ~53% of the in-domain gain (-0.101 of -0.190) and
  is FAR better than any out-of-domain S8 token on weather (best was
  tok_ETTh1 1.0210; uni 0.9506).
- Versus S8's out-of-domain collapse: tok_weather on ETTh1 was +0.0545
  over nan; uni's worst cell anywhere is +0.0211 (one seed on ETTh1),
  mean +0.0042. The collapse is gone; what remains is dilution.

## Check cells (mcar@0.3 / mnar_high@0.7)

| cell | ds | nan | tok_own | uni | uniaug |
|------|----|-----|---------|-----|--------|
| mcar:0.3 | ETTh1 | 0.9781 | 0.9781 | 0.9781+-0.0000 | 0.9781+-0.0000 |
| mcar:0.3 | ETTm1 | 0.9435 | 0.9435 | 0.9435+-0.0000 | 0.9435+-0.0000 |
| mcar:0.3 | weather | 1.0975 | 1.0975 | 1.0975+-0.0000 | 1.0975+-0.0000 |
| mnar_high:0.7 | ETTh1 | 3.6671 | 4.1132 | 3.8668+-0.0704 | 3.9353+-0.0483 |
| mnar_high:0.7 | ETTm1 | 5.1838 | 5.3884 | 5.3268+-0.2914 | 5.4260+-0.1198 |
| mnar_high:0.7 | weather | 1.3281 | 1.3260 | 1.3270+-0.0009 | 1.3215+-0.0013 |

- mcar@0.3: uni == nan to the printed precision on all ds (fully-missing
  patches essentially never occur under mcar; the token path is never
  activated). Pre-registered |uni-nan| <= 0.02: PASS.
- mnar_high@0.7: uni is worse than nan on ETTh1 (+0.200) and ETTm1 (+0.143),
  exceeding the pre-registered +0.10 tolerance — same direction as S8's
  tokens (tok_own is even worse: +0.446 / +0.205), so this is the known
  "no MNAR transfer, slight degradation", not a new failure mode; uni
  roughly HALVES the in-domain token's MNAR penalty. weather is a wash
  (-0.001). Verdict on the check: fails the letter of the tolerance on
  2/3 ds; qualitatively unchanged from S8.

## Judgement

By the letter of the pre-registered criteria:

- **uni: FAILURE.** uni > nan on ETTh1 (1.0072 vs 1.0030), so "uni <= nan
  everywhere" is false; additionally all three gaps to tok_own exceed 0.02
  (+0.041 / +0.027 / +0.088).
- **uniaug: FAILURE**, and strictly worse than uni on all three ds
  (+0.024 / +0.010 / +0.012 vs uni) — the augmentation hurt: as flagged in
  the design caveat, scale jitter is inert under bolt's instance norm
  (verified 4.8e-07), so the only active ingredient was the trend
  perturbation, which shifted the training distribution away from the eval
  windows. Recorded as a degraded variant, no further tuning.

Honest reading behind the FAILURE label: the failing ETTh1 cell is an
IN-corpus domain, and the exceedance (+0.0042, CI [-0.033, +0.042]) is
seed-noise level (2/3 seeds are <= nan). What actually happened is not
out-of-domain collapse (S8's failure mode, +0.055) but DILUTION: the mixed
token is nowhere catastrophic (worst mean deviation from nan +0.004) yet
nowhere best — it is strictly dominated by the in-domain token on 3/3 test
sets (gap CIs exclude 0 on ETTh1 and weather), keeping only ~0% / 45% /
53% of the in-domain gains on ETTh1 / ETTm1 / weather.

## Conclusion

A single gap-prior token trained on a cross-dataset corpus can be made
"nowhere bad" (worst deviation from nan +0.004 mean, vs +0.055 for S8's
out-of-domain weather token) but NOT "as good as local": the in-domain
token remains better on every test set, by up to +0.088 relMSE on weather.
This STRENGTHENS S8's "in-domain prior" attribution: the token's benefit is
mostly domain-specific content, and one 768-d vector cannot hold three
domain priors at once — mixing the corpus averages them toward a compromise
that spends most of its gain budget on the most distinctive domain
(weather) and gives up the smaller ETT gains entirely. Practical
implication: route per-dataset (in-domain) tokens when the domain is known;
the universal token is only the better default when the domain is unknown
and the candidate pool includes hostile domains.

## Anomalies / process notes

1. First anchor launch crashed immediately (KeyError 'nan' — token-loading
   loop did not skip non-token methods). Fixed in `shard_main`; no data
   produced by the crashed run.
2. Log-clobbering: shard logs were written both by shell `>` redirect
   (stdout, fixed offset) and by the script's append-mode handle; stdout's
   stale offset overwrote the earliest appended bytes, destroying the CLEAN
   ETTh1 line in each anchor log (RESULT lines provably intact: the gate
   strict-parses every line, 162/162 valid). Repaired by recomputing clean
   ETTh1 (bit-identical to S8: 11.199030024288856) and appending it;
   eval phase switched to `>>` append redirect — no further issues.
3. First `s18.png` render had a broken bottom row (sharex between bar and
   curve panels); fixed and re-rendered. No numeric impact.


## Artifacts

`run_s18_unitoken.py` (tiny / anchor / gate / train / eval / merge / report),
`s18_results.json`, `s18.png`, `s18_unitoken_{uni,uniaug}_s{101,202,303}.pt`,
logs `s18_smoke.log`, `s18_anchor_shard{0-2}.log`, `s18_gate.log`,
`s18_train_gpu{0-2}.log`, `s18_eval_shard{0-2}.log`, `s18_merge.log`,
`s18_report.log`.
