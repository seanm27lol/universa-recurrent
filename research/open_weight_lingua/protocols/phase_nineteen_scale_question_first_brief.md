# Phase Nineteen: does the question-first two-stage account hold at Gemma-3-27B? (exploratory)

**Status: FROZEN 2026-10-13 (UTC), before any Phase Nineteen model forward.
Branch `phase-nineteen-scale-question-first`, based on `main` at `a267f03`.
Exploratory, on reused calibration and pilot prompts, question-first order
only. A confirmatory test would use a fresh split under its own frozen
protocol.**

## The question

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

**What is known for Gemma-3-12B and Qwen2.5-7B (Phases Sixteen to
Eighteen).** With the question first, the model works in two stages:
- **Middle layers.** The program tokens read the question and mark both
  variables' statements. In Gemma-3-12B this has begun below layer 12 and
  is finished by layer 24 of 48.
- **Late layers.** The post-program positions read the marked program.
- **Without the marks,** Gemma-3-12B takes the most recent assignment of
  either variable.

**What is known for Gemma-3-27B (Phase Thirteen).** With the question
after, it follows the 12B's account, with the read ending deeper: release
layer 46 of 62 against 30 of 48.

Phase Nineteen asks whether the question-first account also holds at 27B,
and where its marking window lies.

## Design

The Phase Eighteen module runs unchanged, apart from a `--protocol` option
that records this brief, on **Gemma-3-27B**:
- lock `configs/model-lock-gemma3-27b.json`, target revision `7a5a3053…`;
- 62 layers, grid floor(j·62/8) = 0, 7, 15, 23, 31, 38, 46, 54.

| | |
|---|---|
| Prompts | the 27B calibration and pilot runs' prompts (`calibration-20260926T183645Z-c3374c27`, `pilot-20260926T185550Z-d085d53c`), question first; kept only where the mark decides the answer (programs ending on the other variable, an independent answer, a distinct other value) |
| Conditions | every program token blocked from every question token in all layers (`program_blind`), from layer k on (`from_k`), or below layer k (`until_k`), for k on the grid |
| Scoring | teacher-forced, for the answer and for the other variable's final value; the bitwise no-op control on every prompt |
| Window | end and onset by Phase Eighteen's rule (tolerance +0.05 on the rate of saying the other value), chosen on calibration and checked on pilot |

## Predictions (recorded before any forward; all one-sided)

1. The no-op control is exact.
2. **The program carries the variable at 27B.** `program_blind` raises
   other-value answers by at least 30 points.
3. **Marking precedes the read.** The window ends at or before layer 38 of
   62, on calibration and pilot.
4. **Deeper in the larger model,** as the read was in Phase Thirteen. The
   window ends at or after layer 31: at least half the depth, as in the
   12B (24 of 48).
5. **The first layers do not mark.** Blocking layers 0–6 (`until_7`) raises
   other-value answers by less than 5 points.

All cells are reported whatever happens.

## Budget, honesty, limits

- **Budget.** At most 3 GPU-hours: about 718 prompts × 33 forwards of a 27B
  model, at about 0.34 s each (about 2.2 hours).
  - The run starts only on an idle GPU with at least 75 GiB available.
  - Foreign processes are logged, never acted on.
  - An operational failure with zero results may be relaunched once and
    recorded.
- **Limits.**
  - Exploratory, on opened programs.
  - One larger model in the same family.
  - Off-distribution blocks; one wording.

**Code:** the unchanged `src/open_weight_lingua/marking_layers_knockout.py`,
launched with `scripts/run_marking_layers_knockout.sh` plus
`--protocol research/open_weight_lingua/protocols/phase_nineteen_scale_question_first_brief.md`.
