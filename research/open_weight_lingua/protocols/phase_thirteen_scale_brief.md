# Phase Thirteen: does the account hold at 27B? (Gemma-3-27B, exploratory)

**Status: FROZEN 2026-10-12 (UTC), before any Phase Thirteen model forward.
Branch `phase-thirteen-scale`, based on `main` at `4c61339`. Exploratory, on
programs opened before. A confirmatory test would use a fourth freshly
generated split under its own frozen protocol.**

## The question

Phases Eight to Twelve built and confirmed, on Gemma-3-12B and Qwen2.5-7B,
an account of how these models answer "What is x?" about a short program:
- after reading the question, they look back at the program late;
- they read the asked variable's own statements;
- they take the most recent intact assignment.

Gemma-3-12B routes that look-back through its question and template
tokens, around layers 24–29 of 48. Its answer positions never need the
program directly.

Phase Thirteen asks whether the same account holds in **Gemma-3-27B**: the
same family and tokenizer (every tokenizer file is identical to the 12B's),
with 62 layers instead of 48.

## Design

The frozen Phase Eight and Phase Twelve modules run unchanged on Gemma-3-27B
(lock `configs/model-lock-gemma3-27b.json`, target revision `7a5a3053…`).
Each records this brief through `--protocol`.

- **Part A: routing (`answer_knockout.py`, Phase Eight's design).**
  - Prompts: the 27B calibration and pilot runs'
    (`calibration-20260926T183645Z-c3374c27`,
    `pilot-20260926T185550Z-d085d53c`): 1,536 prompts.
  - Teacher-forced digit accuracy.
  - From grid layer k onward, block the answer positions (variant A) or
    every post-program position (variant B) from the program tokens.
  - Grid: floor(j·62/8) = 0, 7, 15, 23, 31, 38, 46, 54.
  - Release layer: chosen on calibration, checked on pilot.
- **Part B: selection (`selection_knockout.py`, Phase Twelve's design).**
  - The same pool construction: the 27B calibration and pilot prompts plus
    the two fresh splits `fresh_2026_10` and `fresh_2026_10b`, regenerated
    and verified against their inventories.
  - Degrade the final, the final and the previous, or only the previous
    assignment, and score v0, v1 and v2.
- **Controls.** The bitwise no-op control on every prompt, in both parts.

## Predictions (recorded before any forward)

These are based on Gemma-3-12B, the sibling model.

1. The no-op control is exact.
2. **Routing as in Gemma-3-12B.**
   - Variant A (answer positions only) has release layer 0: the answer
     positions never need the program directly.
   - Variant B (everything after the program) has a release layer in the
     second half of the network (31 or later) but not the last grid layer.
3. **Selection as in both smaller models.**
   - Degrading the final assignment costs more than 25 points, and v1 is
     the most common wrong answer.
   - Degrading both makes v2 more common than v1.
   - Degrading only the previous assignment costs under 10 points.

## Budget, honesty, limits

- **Budget:** at most 4 GPU-hours. Part A has about 27,600 forwards of a
  27B model (roughly 2.5 hours); Part B is about 30 minutes. Each part
  starts only on an idle GPU with enough free memory (the 27B model needs
  about 55 GB). Foreign processes are logged, never acted on.
- **Limits.**
  - One larger model from the same family is a scale check, not a scaling
    law.
  - The prompts and pools reuse opened programs.
  - Knockout limits apply as before: an off-distribution intervention and
    teacher-forced scoring.

**Code:** the unchanged `answer_knockout.py` and `selection_knockout.py`,
launched with `scripts/run_answer_knockout.sh` and
`scripts/run_selection_knockout.sh` plus
`--protocol research/open_weight_lingua/protocols/phase_thirteen_scale_brief.md`.
