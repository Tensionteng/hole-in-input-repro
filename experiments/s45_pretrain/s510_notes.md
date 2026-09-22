# S510 — learned fill x (stock vs retrofit) on real missingness: NEGATIVE result

Protocol: S26 verbatim (cal/test 50/50, fillnet tanh-bounded around the center fill,
800 steps, centers zero/linear for penn/metr). `s510_fillnet.json`, `eval_s510_fillnet.py`.
Baselines (fixed fills) from s55_realfill.json.

## Results (TEST half, NMSE mean/median)

**Penmanshiel (cens)**
- P0 best fixed fill (zero): 19.94 / 1.86
- P0 + fillnet (plain_fill): **16.16 / 1.61**  ← reproduces S26's H1: a learned fill
  through the frozen STOCK model beats the best fixed fill
- P5 + fillnet (fill_mask): 21.43 / 1.63  ← the retrofit combination does NOT beat
  the stock combination

**METR-LA (miss)**
- P0 best fixed fill (linear): 2.14 / 1.26
- P0 + fillnet: 2.21–2.43 / 1.28–1.30  ← reproduces S26's H2 (no learned-fill win here)
- P5 + fillnet: 2.69–3.05 / 1.36–1.40  ← worse than P0's combination

## Why the combination punch did not land (recorded as-is)

A fillnet can only learn fills that make the FROZEN model's output better, and real
missingness offers no ground truth to imitate — the attainable ceiling for any
learned fill here is "smooth the hole". The blind stock model already reaches that
ceiling. P5's content-reading pays only when the fill CARRIES real information
(its closure axis is gated by fill quality), which a fillnet cannot fabricate on
real data. This is the same "the fill is the bottleneck" message, now from the
learned-fill side.

## Consequence for the claims

- The real-data promise must be "never worse under any fixed fill" (S59's zero-fill
  blindness broke exactly this; P6 is the fix, re-verified on real data after
  training), NOT "big wins on real data". Penmanshiel's best fixed fill IS zero —
  so P6's real-data check is the decisive one.
- The synthetic-grid win + fill-quality gating remains the paper's headline; the
  learned-fill synergy is reported as a null result.
