# S45 — pretraining under the restored interface: a within-family controlled experiment

**Status: DESIGN (pre-registered before any training).**

## Question

The paper's causal claim is now stated as a gate: the interface is a *necessary condition*,
and the Moirai-vs-Bolt natural experiment confounds interface with pretraining (Moirai was
pretrained with a masked-modelling objective; the Chronos families on filtered corpora).
The 2-minute adapter of S27 shows the ceiling is removable but leaves two things open:

1. Does **pretraining** under the restored interface (content + flag) beat the adapter
   patch — in particular at the low-quality end, where the adapter still loses in 26/67
   cells?
2. Is the benefit carried by the **interface** or by **mechanism-diverse missingness in
   pretraining**? The adapter evidence is consistent with "you need both".

Only a from-scratch, within-family experiment answers both: same architecture, same corpus,
same budget, varying only (interface × corpus regime).

## Arms (2 x 2 factorial)

| arm | input interface | corpus regime | reads as |
|---|---|---|---|
| A | stock (content zeroed at masked positions) | filtered (NaN-containing series dropped) | the status quo, replicated |
| B | restored (content + flag) | filtered | interface alone |
| C | restored (content + flag) | mechanism-diverse missingness | the paper's prescription |
| D | stock (content zeroed) | mechanism-diverse missingness | augmentation alone (control) |

All four: chronos-**bolt-tiny** (8.7M), random init, identical seed schedule, identical
training budget, identical corpus (only the *regime* differs), identical eval harness.

Arm A is the within-experiment reference: our corpus is not Google's, so every arm is
compared against OUR arm A, and the arm-A-vs-stock-bolt-tiny ratio is reported openly as
the corpus-difference measure.

## Data

- **Source**: `autogluon/chronos_datasets` (the public portion of the Chronos training
  corpus; verified reachable via hf-mirror, 1243 files) + local SCADA/traffic.
- **Exclusions (anti-leakage)**: every source the paper evaluates on is held out of
  pretraining --- the 9 benchmark datasets (ETT h/m 1/2, weather, electricity, traffic,
  exchange, illness), Penmanshiel/Kelmarsh, METR-LA, and the GIFT-Eval corpus. Evaluation
  is therefore zero-shot for every arm, exactly as the paper evaluates the stock models.
- **Filtered regime**: drop any series containing NaN (mirrors the model papers' own
  documented filtering), then sample windows.
- **Mechanism-diverse regime**: same series; on-the-fly per-window augmentation with
  mechanism ~ {clean 50%, scattered, block, value censoring, extreme} and rate ~
  U(0.05, 0.7). At holed positions the context carries a **deployment-realistic mixture**:
  with prob 1/2 declared-as-NaN (content zeroed, flag), with prob 1/2 an alpha-blend fill
  (alpha ~ U(0,1) between linear and truth) plus flag. The model thus sees both
  "declare" and "fill-and-declare" during training, which is what deployment looks like.
- **Scale**: ~2-4M windows (context 512 + horizon 64) per arm, i.e. ~1.5-2.5B points.
  Sampled uniformly across datasets with per-dataset caps so no source dominates.

## Model and training

- Architecture: bolt-tiny, loaded from the local checkpoint's config, **random init**
  (patch 16, context 512, 4 forecast patches = native H=64, 9 quantiles).
- Objective: the bolt-native next-block quantile loss (pinball over the 9 quantiles on the
  H=64 continuation), computed only where the target is fully observed.
- Interface switch: the S27 `encode_iface` replica (gate-verified bit-exact against the
  stock forward at plain), with `native` = stock and `dual` = restored. Arm D trains with
  `native` on augmented data; arms B/C with `dual`; arm A with `native` on filtered data.
- Optimiser: AdamW, lr 3e-4, cosine to 0, 1k warmup, batch 256, fp32 (project convention;
  bolt-tiny makes this cheap), grad-clip 1.0, ~15k steps per arm. 8xA800 data-parallel or
  single-GPU sequential arms --- decided at implementation time, both fit the budget.
- The NaN-gradient guard from S27's notes (skip non-finite grad-norm steps) is active in
  every arm.

## Evaluation (the paper's own harness, unchanged)

For each arm, on the 9 held-out datasets (last-20% windows, own-clean normalisation):

1. **rho probe** (S36 protocol, 3 datasets x 3 mechs x 60 windows, plain/mask/nan).
2. **Ceiling sweep**: alpha in {0, .25, .5, .75, 1} on the 4-mech x 2-rate grid.
3. **Damage grid**: relMSE by mechanism and rate.
4. **Attack surface**: the S29 gradient protocol on METR-LA (plain vs declared).
5. **Clean accuracy**: relMSE on unholed contexts, all 9 datasets.

## Pre-registered predictions

- **P1 (replica)**: arm A behaves like stock bolt-tiny: rho_declared ~ 0, alpha sweep flat
  (the ceiling reproduces in a model we trained ourselves).
- **P2 (prescription works)**: arm C's declared path retains content (rho_declared >= 0.5)
  and its alpha sweep slopes down; at alpha=1 it closes >= 50% of arm A's untouchable
  excess at the median.
- **P3 (floor ~ declaration)**: at alpha=0, arm C is no worse than arm A's declared path in
  >= 60% of cells --- pretraining, unlike the 2-minute adapter, should lift the low end to
  near-declaration.
- **P4 (factor separation)**: arm B (interface, no augmentation) beats arm A at alpha=1
  but loses to arm C at alpha=0; arm D (augmentation, no interface) stays flat in alpha.
  If P4 fails, the story is not "interface x pretraining" but one factor alone.
- **P5 (no clean cost)**: arm C's clean accuracy within 5% of arm A.
- **P6 (trade-off returns)**: arm C's restored path reopens the attack surface relative to
  arm A's declared path (damage ratio > 1 where A is ~1).

Prediction falsification rules are the project's: every prediction is reported, confirmed
or not, in s45_notes.md and the paper's prereg scorecard.

## Gates

- G1: the eval harness reproduces the stored bolt-tiny cells of S36 before any arm is
  scored (anchor tolerance 5%).
- G2: the S27 encode replica matches the stock forward at plain to max|diff| < 1e-3 with
  the randomly-initialised tiny weights as well (catches config drift).
- G3: arm A's clean relMSE vs stock bolt-tiny is reported (not gated --- it documents the
  corpus difference) and all arms share the identical data pipeline run.

## Risks and how they are handled

- **Our bolt-tiny is worse than Google's.** Expected and harmless: conclusions are
  within-experiment contrasts, and the arm-A/stock ratio is reported openly.
- **Corpus regime confound**: the filtered regime drops series, the augmented does not, so
  dataset counts differ. Handled by sampling both regimes from the same underlying pool
  and reporting effective series counts per arm.
- **Budget blowout**: 4 arms x 15k steps at batch 256 fp32 on bolt-tiny is hours per arm
  on one A800; if slower, halve steps and note it in the prereg deviation log.
- **Augmentation at train time ≠ augmentation at eval time**: eval masks use the paper's
  fixed mechanism functions; train-time augmentation uses the same functions with sampled
  rates, so the distributions match by construction.

## Deliverables

`s45_pretrain/`: this design, `prep_corpus.py` (download/filter/sample, cached),
`train_s45.py --arm {A,B,C,D}`, `eval_s45.py` (reuses s25/s27/s36 harness), results JSON,
`s45_notes.md` with the prereg scorecard. Then one section in the paper replacing the
"natural next step" paragraph of the Limitations with the measured answer.
