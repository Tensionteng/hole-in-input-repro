# ROUNDUP v2: exploration round (2026-08-27) — recipe map, leaderboard, real data

Extends S53_S56_ROUNDUP.md. Everything pre-registered or post-registered in the
stage notes BEFORE the corresponding claims were computed; paper untouched.

## The arc of the day

S59 found the zero-fill blindness → recipe map (P6/P7/P8/P9) → WiSE-FT dial →
Q50 (P8×0.5) → real-data parity → S57 leaderboard. Q50-base is the candidate
final model.

## Results by stage

| stage | question | answer | notes/data |
|---|---|---|---|
| S58 | can the clean cost be dialled? | yes — WiSE-FT frontier; W85@base Pareto-beats raw CPT | s58_notes.md |
| S59 | any-fill robustness? | P5 is ZERO-BLIND (3–19×, 0/9); cause = H recipe never trains content=0; C arm (declare mixture) immune | s59_notes.md |
| S510 | fillnet × retrofit on real data? | negative — blind stock + fillnet already reaches the attainable ceiling; no synergy | s510_notes.md |
| S511 | benchmarks already missing? | published files NaN-free (silently cleaned); long stuck-sensor plateaus documented | s511_notes.md |
| S512 | mechanism visible? | yes — P0 attention flat over fill quality, P5 attends masked patches more at α=1 | s512_notes.md |
| recipe map | which knob costs what? | declare endpoint necessary (zero), mechanism diversity decides fix quality (P8 best, clean 1.254); ffill endpoint & clean_p are dead knobs; declare training SHRINKS attack surface (1.14–1.20 vs 1.43) | s59_recipe_map.md |
| S55 v2 | real data, any fill? | Q50 at parity with stock on penn/metr (zero 20.2 vs 19.9; metr 2.39/2.11 vs 2.38/2.14, better clean) | s55_v2_notes.md |
| S57 | vs other TSFMs? | Q50 is the only model non-trivial on ALL three fills; oracle split with Moirai 2.0 (4-4-1), linear 7-8/9, zero 9/9 vs bolt-stock & Moirai | s57_notes.md |

## Final candidate recipes (as of now)

- **Q50 (= P8 mechdiv-CPT × WiSE-FT 0.5)** — default: 62.3% closure, clean +5.8%,
  any-fill robust (incl. zero), attack 1.20×, real-data parity. @base numbers in
  s57_notes.md.
- **P5 (H blend-only CPT)** — max closure (77–78%) but zero-blind; only with a
  controlled fill.
- **Q70/P8** — max any-fill robustness, clean cost 13–25%.

## Checkpoints added (s45_ckpt/)

arm_P{6,7,8,9}_tiny.pt, arm_P{6,8}_base.pt, arm_Q{30,50,70}_tiny.pt,
arm_Q50_base.pt (plus the earlier P0–P5 family and W30–W85 WiSE-FT grids).

## Open items for the paper pass (unchanged + new)

- M_ladder H@small denominator fix (from v1).
- S55 parity table must accompany "best recipe" sentences (from v1).
- NEW: the main table should become the S57 leaderboard (Q50 row) + the any-fill
  matrix (S59, P6/Q50 rows); the zero-blindness discovery (S59) and the
  attention-map figure (S512) are new paper material.
- NEW: P9's clean_p anomaly (75% clean → worse clean) needs seeds before any
  claim; flagged in s59_recipe_map.md.
- Q50's own seeds (3x) still pending — the candidate final model needs its
  replicate runs before main-text numbers.
