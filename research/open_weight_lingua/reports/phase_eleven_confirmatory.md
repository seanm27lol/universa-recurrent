# Phase Eleven confirmatory: a disrupted final assignment falls back to the previous one, on fresh programs — 2026-10-12

```text
x = 4
y = 3
x = 7              ← the asked variable's final statement (a literal)
y = 5
What is x? Reply with only the integer.      → 7   (superseded value: 4)
```

Exploratory Phases Ten and Eleven
([Ten](phase_ten_dependency_knockout.md), [Eleven](phase_eleven_supersession.md))
found two things on reused prompts.
- Blocking the asked variable's final statement from earlier lines, of
  either variable, damages the answer even when that statement is a
  literal.
- The model then mostly answers the **superseded value**, the asked
  variable's previous assignment.

Six claims were frozen
([protocol](../protocols/phase_eleven_confirmatory_fresh.md)) before the
test programs existed. They were tested on **512 newly generated groups**:
seed 207001, excluding every Phase Two program and Phase Nine's fresh split.

**All six were supported, in both models, close to the exploratory
estimates.**

## What ran

- **Freeze.**
  - The protocol, the new seed, the module, its tests and the launcher
    were committed and pushed at `1f65b9f`. The tests used test-only seeds.
  - Both manifests record that commit, the protocol hash `0df41a8bad02…`
    and source hashes identical to the frozen code.
- **Fresh programs.**
  - 512 groups under `fresh_2026_10b`: 973 attempts, with 44 candidates
    rejected because they repeated an earlier program.
  - Phase Nine's fresh split regenerated to its recorded inventory before
    it was excluded.
  - Both families received the same programs (inventory `67dc17bb65b9…`).
  - **424 prompts in 204 groups** qualified: the asked variable's final
    statement is a literal with earlier assignments, and the superseded
    value differs from the answer.
- **Settings.** As in exploratory Phase Eleven: both blocks from layer 0,
  teacher-forced scoring of the answer and the superseded value, and the
  bitwise no-op control, which held on every prompt.

| Family | Run | Manifest / results sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p11-confirm-gemma3-12b-20261012T055510Z-d723ce72` | `36625f9a…` / `cb12e5b6…` | 547 s / 697 s |
| Qwen2.5-7B | `p11-confirm-qwen2.5-7b-20261012T061602Z-5e56b0d1` | `0d129998…` / `d3b156f7…` | 312 s / 398 s |

Both runs were admitted to an idle GPU, with no foreign processes. One CPU
evaluation wrote `p11-confirm-evaluation-20261012.json` (`5a785cc7…`).

## Results

| | Baseline | Same-variable block | Other-variable block |
|---|---|---|---|
| **Gemma-3-12B** accuracy | 0.993 | 0.507 | 0.637 |
| superseded value given | 0.007 | 0.446 | 0.375 |
| share of wrong answers that are the superseded value | 3 of 3 | 0.84 | 0.92 |
| **Qwen2.5-7B** accuracy | 0.828 | 0.559 | 0.462 |
| superseded value given | 0.118 | 0.401 | 0.465 |
| share of wrong answers superseded | 0.64 | 0.80 | 0.76 |

**The six confirmatory hypotheses.** These are 99.17% whole-group bootstrap
intervals (10,000 resamples; seed 804100 Gemma, 804101 Qwen), Bonferroni
across the six tests.

| ID | Measure | Mean | Interval | Threshold | Verdict |
|---|---|---:|---|---|---|
| G1 | Gemma: loss under the same-variable block | +0.486 | [+0.405, +0.567] | > 0.05 | **Supported** |
| G2 | Gemma: loss under the other-variable block | +0.356 | [+0.273, +0.443] | > 0.05 | **Supported** |
| G3 | Gemma: superseded-value gain under the same-variable block | +0.439 | [+0.359, +0.520] | > 0.10 | **Supported** |
| Q1 | Qwen: loss under the same-variable block | +0.269 | [+0.192, +0.345] | > 0.05 | **Supported** |
| Q2 | Qwen: loss under the other-variable block | +0.366 | [+0.277, +0.457] | > 0.05 | **Supported** |
| Q3 | Qwen: superseded-value gain under the same-variable block | +0.283 | [+0.202, +0.366] | > 0.10 | **Supported** |

**Predictions.** All six were predicted to be supported. The estimates sit
within or near the exploratory values: G1 +0.49 against +0.52; Q1 +0.27
against +0.34; Q3 +0.28 against +0.33.

## What this establishes

- **On unseen programs, degrading the asked variable's final assignment
  makes both models fall back to its previous assignment.**
  - Blocking the final statement from earlier lines costs 49 (Gemma) and
    27 (Qwen) points, even though a literal's value needs none of them.
  - The model then answers the superseded value in 40–45% of prompts,
    against 1–12% at baseline. That is 76–92% of its wrong answers.
- **The degradation is not specific to the asked variable's own earlier
  lines.** Blocking the other variable's earlier lines, a set of the same
  size, costs 36 and 37 points and produces the same fallback.
- **With Phases Eight and Nine:** the late read retrieves the asked
  variable's statements, and which statement it takes depends on the
  integrity of the latest one. When that is degraded, the previous
  assignment wins.

## What this does not establish

- **Why any block degrades the final statement.** It may lose position or
  structure cues. Only layer-0 blocks were tested.
- **"Most recent intact" as a selection rule.** That the late read takes the
  most recent intact assignment fits the fallback, but has not been tested
  directly, for example by degrading the last two assignments and checking
  for a fallback to the third-latest.
- **Knockout limits.** An off-distribution intervention, teacher-forced
  scoring, literal final statements only, one task family.

Code: [supersession_confirm.py](../src/open_weight_lingua/supersession_confirm.py),
launched with `scripts/run_supersession_confirm.sh`.
