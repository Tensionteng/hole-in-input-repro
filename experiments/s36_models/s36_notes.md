# S36 — the fill-reachable set across every checkpoint we can run

## Why
S30 measured the permutation probe on five checkpoints from four families. Proposition 1
predicts a trichotomy that should hold for any model, so the interesting test is breadth: more
families, more sizes within a family, and families that ship no declared missing-data path at
all, where the proposition predicts full rank is the only regime available.

## Anchor gate — PASS, bit-exact
chronos-bolt-base reproduced all 27 stored S30 cells to 0.00% deviation.

## What ran
| checkpoint | family | declared path | result |
|---|---|---|---|
| bolt tiny / mini / small / base | Chronos-Bolt | mask / NaN | rho plain 0.87-0.97, declared **0.0000** at every size |
| chronos-t5 small / base | Chronos-T5 | NaN | declared **0.0000** |
| chronos-2 | Chronos-2 | NaN | declared **0.0000** |
| timesfm 2.5 | TimesFM | NaN | declared **0.0000** (overwrites with its own interpolation) |
| moirai small / base / large | Moirai 1.1-R | observation mask | declared **0.94-0.97** = retains content |
| moirai-moe small / base | Moirai-MoE | observation mask | declared **0.971 / 0.973** = retains content |

The headline additions: the trichotomy is constant across a 24x parameter span inside
Chronos-Bolt and a 22x span inside Moirai, so it is a property of the convention rather than of
capacity; and Moirai-MoE is a **second** family that retains fill content after a declaration,
which strengthens the natural experiment of S33 from one family to two.

## What did not run, and why it is not in the paper

Three decoder-only checkpoints could not be made numerically reliable under the pinned
transformers 5.14:

* **Sundial-base-128M** — the encoder emits NaN on clean random input, before any missingness
  is involved. Its vendored attention code targets transformers 4.3x; `attn_implementation`
  variants do not help. Excluded.
* **TimeMoE-200M** — hidden states are non-finite from layer 1 onward in both float32 and
  bfloat16, with finite weights and a complete checkpoint. Excluded.
* **TimeMoE-50M** — **intermittent**. One run produced finite forecasts on all nine plain
  cells and an aggregate rho of 0.6501; a re-run with the same seed and code produced
  non-finite output on every cell, including plain, and a standalone reproduction also failed.
  A number we cannot reproduce does not go in the paper, so the 0.6501 is discarded.

We keep TimeMoE in Table `tab:families` as the **no declared path** case with its rho columns
left blank, because that claim does not depend on running the model. Reading
`modeling_time_moe.py`: `forward` accepts `attention_mask` (a padding mask over positions) and
`loss_masks` (training only), and nothing else. There is no observation-mask argument, and a
NaN in `input_ids` passes through the input linear into every token. A user of this family has
no way to declare a hole; full rank is not something they can decline. That is the point the
row makes, and it is verifiable by reading rather than by running.

## Final table (after chronos-t5-base completed)

13 checkpoints, 6 families. Chronos-T5 base: rho plain 0.911, declared **0.0000** — the
tokenising family behaves exactly like the patching one, which is the point of including it.

Declared-path summary across the whole table:
* discards (rho at float noise, <= 2.2e-6): Chronos-Bolt x4, Chronos-2, Chronos-T5 x2
* overwrites (rho 0, but by substituting its own interpolation): TimesFM 2.5
* retains (rho 0.938-0.973): Moirai x3, Moirai-MoE x2
* no declared path at all: TimeMoE, Sundial (not measurable here; see above)
