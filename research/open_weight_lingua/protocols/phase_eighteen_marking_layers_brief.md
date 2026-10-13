# Phase Eighteen: at which layers do the program tokens read the question? (exploratory)

**Status: FROZEN 2026-10-13 (UTC), before any Phase Eighteen model forward.
Branch `phase-eighteen-marking-layers`, based on `main` at `036373d`.
Exploratory, on reused calibration and pilot prompts, question-first order
only.**

## The question

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

Phases Sixteen and Seventeen found that, with the question first, the
program tokens mark the asked variable by attending to the question. Block
that attention at every layer, and the late read takes the other variable's
value. This phase asks *when* in the network that marking happens. It
should come before the late read: in the question-first order, Gemma's read
runs from about layer 24 to 36 of 48, and Qwen's from about 21 to 24 of 28
(Phases Eight and Fourteen).

## Design

- **Models.** Gemma-3-12B and Qwen2.5-7B, locks hash-verified.
- **Prompts.** Phase Eight's calibration and pilot prompts in the
  question-first order, kept only where the mark decides the answer:
  programs ending on the other variable, with an answer that does not
  depend on it, whose final value differs from the answer. That is 718
  per family.
- **Conditions.** Every program token is blocked from every question token,
  in a range of layers:
  - `program_blind`: all layers (Phase Sixteen's condition);
  - `from_k`: layers k … L−1, for k on Phase Eight's grid (k > 0);
  - `until_k`: layers 0 … k−1, for the same k.
- **Scoring.** Teacher-forced, for the answer and for the other variable's
  final value. The bitwise no-op control runs on every prompt.
- **The marking window**, chosen on calibration and checked on pilot. With
  tolerance +0.05 on the rate of saying the other value (Phase Eight's
  release tolerance):
  - **end** = the smallest grid k at which `from_k`, and every later
    `from`, stays within tolerance of baseline;
  - **onset** = the largest grid k at which `until_k`, and every earlier
    `until`, stays within tolerance.

## Predictions (recorded before any forward; all one-sided)

1. The no-op control is exact.
2. `program_blind` reproduces Phase Sixteen on these prompts. Other-value
   answers rise by at least 30 points (Gemma) and 8 points (Qwen).
3. **Marking precedes the read.** The window ends at or before layer 24 in
   Gemma and layer 17 in Qwen, on calibration and pilot.
4. **The first layers do not mark.** Blocking only the first grid
   interval (`until_6` for Gemma, `until_3` for Qwen) raises other-value
   answers by less than 5 points.

All cells are reported whatever happens.

## Budget, honesty, limits

- **Budget.** At most 2 GPU-hours: about 23,700 forwards per family. Same
  gate and rules as before.
- **Limits.**
  - Exploratory.
  - A layer range that is harmless to block can still carry a redundant
    route.
  - Off-distribution blocks; one wording.

**Code:** `src/open_weight_lingua/marking_layers_knockout.py`, launched with
`scripts/run_marking_layers_knockout.sh`.
