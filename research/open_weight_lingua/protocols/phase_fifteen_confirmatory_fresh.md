# Phases Fourteen and Fifteen confirmatory: the late read with the question first, on a fifth fresh split

**Status: FROZEN 2026-10-12 (UTC), before the confirmatory programs are
generated and before any model forward on them. Branch
`phase-fourteen-question-order`, at the commit that adds this file, after
the Phase Fifteen exploratory report (`c3d5dd4`). Every hypothesis and
analysis choice is fixed here.**

## The question

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

Two exploratory phases on opened programs found, with the question before
the program:
- **Phase Fourteen** ([report](../reports/phase_fourteen_question_order.md)).
  - The post-program positions cannot answer from the program's last line
    alone when it assigns the other variable: there is no end-of-program
    summary.
  - Gemma's read ends later than with the question after (blocking from
    layer 30 of 48 still costs about 10 points). Qwen's ends at the same
    layer (24 of 28).
- **Phase Fifteen** ([report](../reports/phase_fifteen_variable_choice.md)).
  On programs that end on the other variable, hiding that variable's lines
  from the post-program positions removes the question-first loss.

This study tests those claims once, in both models, on new programs.

## Data: a fifth fresh split

- **Exclusions.** The code rebuilds the frozen Phase Two plan and the four
  earlier fresh splits (`fresh_2026_10`, `10b`, `10c`, `10d`; seeds
  206001–209001). It verifies their inventories (`8e0d30ab…f963`,
  `67dc17bb…2a39`, `c56dfba4…a254`, `e2c9ab25…e7`) and excludes every
  canonical program from all of them.
- **Generation.** **512 new groups** under split `fresh_2026_10e`, seed
  **210001**. The inventory hash is recorded before any forward. Both
  families receive the same programs.
- **Prompts.** All 2,048 prompts, each in both orders (`after`, and `first`
  = Phase Seven's `original_before`).
- **Main stratum.** Programs whose last line assigns the other variable,
  excluding prompts whose answer passes through a copy of the other
  variable.

## Settings

As in Phases Fourteen and Fifteen: Gemma-3-12B and Qwen2.5-7B under their
locks, teacher-forced digit accuracy, and the bitwise no-op control on both
orders.

Every condition blocks every post-program position from a set of program
tokens.

| Condition | Order | Blocked from | From layer |
|---|---|---|---|
| `baseline` | after, first | nothing | — |
| `block_other` | first | every token of the other variable's lines | 0 |
| `only_last_line` | first | every program token before the last line | 0 |
| `floor` | first | the whole program | 0 |
| `late` | first | the whole program | Gemma 30, Qwen 24 (each family's question-after release layer, Phase Eight) |

The other variable's final value is also scored under `baseline` and
`block_other` (descriptive).

## Confirmatory hypotheses

Per-prompt measures, all on correctness (1 or 0):
- `residual_gap` = after `baseline` − first `block_other`;
- `recovery` = first `block_other` − first `baseline`;
- `summary` = first `only_last_line` − first `floor`;
- `late_loss` = first `baseline` − first `late`.

**Eight tests, Bonferroni-controlled.** Each uses a two-sided percentile
interval at coverage 1 − 0.05/8, from 10,000 whole-group bootstrap resamples
(seed 807100 Gemma, 807101 Qwen). Groups are resampled within each claim's
prompts.

| ID | Family | Measure | Prompts | Supported iff | Claim |
|---|---|---|---|---|---|
| G1 | Gemma-3-12B | `residual_gap` | main | upper < 0.05 | hiding the other variable's lines restores question-after accuracy |
| G2 | Gemma-3-12B | `recovery` | main | lower > 0.05 | ... and that is a real gain over the question-first baseline |
| G3 | Gemma-3-12B | `summary` | main | upper < 0.10 | no end-of-program summary with the question first |
| G4 | Gemma-3-12B | `late_loss` | all | lower > 0.05 | with the question first, the read is not finished by layer 30 |
| Q1 | Qwen2.5-7B | `residual_gap` | main | upper < 0.05 | as G1 |
| Q2 | Qwen2.5-7B | `recovery` | main | lower > 0.05 | as G2 |
| Q3 | Qwen2.5-7B | `summary` | main | upper < 0.10 | as G3 |
| Q4 | Qwen2.5-7B | `late_loss` | all | upper < 0.05 | with the question first, the read is finished by layer 24, as with the question after |

## Predictions (recorded before generation)

All eight are predicted to be supported. Below are the exploratory
estimates, with the bootstrap intervals they would give at 512 groups:

| ID | Exploratory | Expected interval | ID | Exploratory | Expected interval |
|---|---:|---|---|---:|---|
| G1 | +0.003 | about [−0.013, +0.020] | Q1 | −0.001 | about [−0.032, +0.032] (the closest call) |
| G2 | +0.158 | about [+0.118, +0.199] | Q2 | +0.485 | about [+0.425, +0.542] |
| G3 | +0.031 | about [−0.009, +0.069] | Q3 | +0.040 | about [+0.018, +0.065] |
| G4 | +0.101 | about [+0.077, +0.125] | Q4 | −0.001 | about [−0.012, +0.010] |

## Execution and limits

- **Freeze first.** This protocol, the seed addition, the module, its tests
  and the launcher are committed and pushed before generation. Tests use
  test-only seeds.
- **Runs.** One run per family, each only on an idle GPU with enough free
  memory.
  - Foreign processes are logged, never acted on.
  - An operational failure with zero results may be relaunched once and
    recorded.
  - A partial run never enters the evaluation.
  - One CPU evaluation over both runs.
- **Budget.** At most 2.5 GPU-hours: about 20,000 forwards per family.
- **Limits.**
  - Off-distribution blocks.
  - One wording.
  - `late` uses each family's question-after release layer, so G4 and Q4
    compare the orders at a single layer, not across the whole sweep.

**Code:** `src/open_weight_lingua/order_confirm.py`, launched with
`scripts/run_order_confirm.sh`.
