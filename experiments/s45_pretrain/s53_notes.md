# S53 (CPT-from-stock) — R1–R6 scorecard

Pre-registered design: DESIGN3.md. Eval: `s53_eval_a.json` + `s53_eval_b.json`
(eval_s45.py, same grid as S45/S47/S48). Reference = P0 (stock bolt-tiny) throughout.
Command:

    analyze_s48.py --files s53_eval_a.json s53_eval_b.json --ref P0 --arms P1,P2,P3,P4,P5

## Scorecard (single seed 20260826)

| arm | rho_declared | closure α=1 | 95% CI | floor α=0 | clean vs P0 | attack |
|---|---|---|---|---|---|---|
| P0 (stock) | 0.001 | — | — | — | 1.000 | 1.01× |
| P1 (break, untrained) | 0.807 | 8.4% | [−8, +20]% | 12/24 | 1.000 | 1.55× |
| P2 (adapter only) | 0.747 | 33.9% | [+9, +42]% | 15/24 | 1.026 | 1.25× |
| P3 (two-stage CPT) | 0.789 | 77.6% | [+72, +83]% | 18/24 | 1.062 | 1.40× |
| P4 (budget control, native+filtered) | 0.000 | −19.2% | [−40, −8]% | 9/24 | 1.038 | 1.02× |
| P5 (one-stage CPT) | 0.808 | 76.7% | [+66, +82]% | 18/24 | 1.048 | 1.43× |

## Verdicts against the pre-registered predictions (DESIGN3.md)

Updated with the two replicate seeds (20260901/20260902), n=3 per arm:

| arm | closure α=1 (3 seeds) | clean vs P0 | ρ_declared |
|---|---|---|---|
| P2 (adapter only) | 37.4% ± 8.7 (30.9–47.2) | 1.013–1.026 | 0.72–0.75 |
| P3 (two-stage CPT) | 82.0% ± 4.0 (77.6–85.4) | 1.062–1.074 | 0.76–0.79 |
| P4 (budget control) | −18.9% ± 9.4 (−28.1–−9.4) | 1.038–1.129 | 0.000 |
| P5 (one-stage CPT) | 75.7% ± 2.1 (73.3–77.2) | 1.048–1.079 | 0.78–0.81 |

- **R1 (the retrofit works) — CONFIRMED, stable across seeds.** P5: 73.3–77.2%
  closure in every seed, far above the 50% bar; P3: 77.6–85.4%.
- **R2 (the cause is the regime, not the extra FLOPs) — CONFIRMED.** P4 stays flat
  in all three seeds (ρ = 0.000, closure −9.4/−19.2/−28.1%).
- **R3 (the adapter alone is not enough) — CONFIRMED.** P2's best seed (47.2%) is
  below P3's worst (77.6%); the seed intervals do not overlap.
- **R4 (stage 1 matters, or it doesn't) — REVISED after seeds: the adapter stage
  carries a small but consistent gain.** P3 beats P5 on every seed
  (+0.9/+5.9/+12.1 points), 82.0% ± 4.0 vs 75.7% ± 2.1. The single-seed tie does
  not replicate. The prescription therefore keeps BOTH stages but describes the
  trade explicitly: stage 1 buys ~6 points of closure at a wider seed spread and a
  slightly larger clean cost; a builder who wants the simpler run loses those
  ~6 points, not the effect.
- **R5 (no clean cost) — narrowly FAILED for both CPT arms.** P5 1.048/1.079/1.053
  (median 1.053), P3 1.062/1.065/1.074 (median 1.065): both sit just outside the
  pre-registered ±5% gate. Reported as-is: the retrofit costs 5–7% clean accuracy,
  not ≤5%.
- **R6 (the trade-off returns, honestly) — CONFIRMED.** Attack ratios 1.33–1.43×
  across CPT arms and seeds vs P0's 1.01× (P1's 1.55× remains the worst).

## Observations outside the scorecard

- P1's high rho (0.807) with ~zero closure is the cleanest illustration in the paper
  of why the probe alone cannot certify usability: the flag perturbs the output, but
  the untrained model cannot convert declaration into accuracy. Its attack ratio
  (1.55×) is correspondingly the worst.
- The final builder recipe is therefore **two-stage (P3)** when the extra ~6 closure
  points are worth the wider seed spread, and **one-stage (P5)** as the simpler
  default: restore the interface, then 5k steps of full-model CPT on the H regime
  (block outages, alpha-blend fill), optionally preceded by the 600-step
  embedding-only adapter. Either way it recovers 73–85% of the stock ceiling's
  untouchable excess, lifts the floor from untestable to 17–18/24, costs 5–7% clean
  accuracy (just outside the ±5% gate, disclosed), and reopens a bounded attack
  surface (1.33–1.43×) that the paper already documents.

## Caveats

- Three seeds per arm (20260826/20260901/20260902) at bolt-tiny; the bolt-base
  retrofit (P5@base) is training now.
- P2's stage-1 loss rose late in its 600 steps (40→89); consistent with the adapter
  saturating, but noted for the appendix.
