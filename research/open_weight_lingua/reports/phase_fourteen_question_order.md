# Phase Fourteen: knowing the question first does not change what the late read uses — exploratory — 2026-10-12

```text
question after:  x = 2 / y = 3 / x = x + 9 / y = 5 / What is x? Reply with only the integer.   → 11
question first:  What is x? Reply with only the integer. / x = 2 / y = 3 / x = x + 9 / y = 5   → 11
```

Phases Eight to Thirteen established an account for prompts where the
question follows the program.
- After the question, the model looks back at the program late.
- It reads the asked variable's own statements.
- It takes the most recent intact assignment.

With the question first, the model knows which variable to follow while
reading the program. Phase Fourteen
([brief](../protocols/phase_fourteen_question_order_brief.md)) asked whether
that changes the read. Two possibilities were tested:
- **(i)** the asked variable's final statement holds its computed value;
- **(ii)** the program's last line holds a summary of the asked variable.

The test lets every post-program position see only the program's last line
(`only_last_line`).

**Neither happens, in either model.**
- With only the last line visible, the answer survives only when that line
  is the asked variable's literal assignment.
- A final arithmetic statement does not hold its result, and a final line
  about the other variable carries nothing about the asked one. Accuracy
  then sits at the floor, in both question orders.
- **What does change:** in Gemma the read ends later with the question
  first (release layer 36 against 30 of 48). In Qwen it ends at the same
  layer, 24 of 28.

Five of the six predictions held. One failed as worded: Qwen's accuracy with
only a literal last line visible was not "within 10 points of baseline",
because it rose 38 points.

These are exploratory results on opened programs.

## What ran

- **Freeze.** The brief, module, tests and launcher were committed and
  pushed at `a29872f` before any forward. Both manifests record that commit
  and the brief's hash `a801f276ae93…`. All 109 source hashes equal the
  files at `a29872f`.
- **Prompts.** Phase Eight's 1,536 calibration and pilot prompts per
  family, each in both orders: 3,072 per family.
  - The `after` prompts' recorded input ids were checked against the chat
    template.
  - The `first` prompts are Phase Seven's `original_before` format.
- **Conditions.** The bitwise no-op control held on every prompt.

| Family | Run | Manifest / results / summary sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p14-order-gemma3-12b-20261012T142321Z-025ef819` | `8aa5d6ce…` / `e704cb58…` / `5b56a8eb…` | 5,508 s / 5,677 s |
| Qwen2.5-7B | `p14-order-qwen2.5-7b-20261012T160739Z-cf8e3037` | `1602d716…` / `f0c914f2…` / `9f8446fc…` | 3,139 s / 3,243 s |

Both runs were admitted to an idle GPU. The only other GPU process was the
rf-moe dashboard (451 MiB), which the gate allows.

## Results

**Reproduction.** In the `after` order, every `baseline` and
`after_program@k` cell reproduces Phase Eight's recorded run exactly. That
covers 1,536 prompts × 9 cells per family, with no correctness difference
and identical log-probabilities.

**The layer sweep.** Teacher-forced digit accuracy with every post-program
position blocked from the program from layer k (calibration split):

| Order | baseline | k = 0 | … | k = 24 | k = 30 | k = 36 | k = 42 | Release (cal / pilot) |
|---|---:|---:|---|---:|---:|---:|---:|---|
| Gemma, question after | 0.940 | 0.081 | | 0.197 | **0.923** | 0.940 | 0.941 | 30 / 30 |
| Gemma, question first | 0.854 | 0.069 | | 0.179 | 0.755 | **0.823** | 0.838 | **36 / 36** |

| Order | baseline | k = 0 | … | k = 17 | k = 21 | k = 24 | Release (cal / pilot) |
|---|---:|---:|---|---:|---:|---:|---|
| Qwen, question after | 0.819 | 0.026 | | 0.165 | 0.201 | **0.819** | 24 / 24 |
| Qwen, question first | 0.520 | 0.026 | | 0.081 | 0.116 | **0.518** | 24 / 24 |

**`only_last_line`, by what the last line does.** All 1,536 prompts per
order. "Floor" is `after_program@0`.

| Last line | n | Gemma after: base / only last / floor | Gemma first | Qwen after | Qwen first |
|---|---:|---|---|---|---|
| Asked variable, literal (`x = 7`) | 208 | 0.976 / **1.000** / 0.106 | 0.952 / **1.000** / 0.077 | 0.899 / **0.986** / 0.024 | 0.587 / **0.971** / 0.024 |
| Asked variable, arithmetic (`x = x + 9`) | 466 | 0.886 / **0.071** / 0.060 | 0.869 / **0.073** / 0.049 | 0.710 / **0.073** / 0.034 | 0.652 / **0.086** / 0.034 |
| Asked variable, copy (`x = x`) | 94 | 0.989 / 0.043 / 0.128 | 0.936 / 0.011 / 0.043 | 0.872 / 0.043 / 0.011 | 0.883 / 0.043 / 0.011 |
| Other variable | 768 | 0.965 / **0.010** / 0.082 | 0.802 / **0.098** / 0.066 | 0.866 / **0.025** / 0.017 | 0.388 / **0.057** / 0.017 |

**Predictions.**
1. The no-op control is exact. *Held.*
2. The `after` cells reproduce Phase Eight's per-prompt correctness. *Held,
   exactly, in both families.*
3. With a literal last line for the asked variable, `only_last_line` stays
   within 10 points of baseline in both orders. *Failed as worded, for Qwen
   in the `first` order:* accuracy rose 38 points, from 0.587 to 0.971. It
   held in the other three cells (+2.4, +4.8 and +8.7 points). The
   prediction was meant to rule out a *loss*. No cell lost.
4. With an arithmetic last line, `only_last_line` costs at least 30 points
   in both orders. *Held:* 57 to 82 points, ending within 6 points of the
   floor.
5. With the other variable's last line, `only_last_line` is within 10
   points of the floor in both orders. *Held:* Gemma 0.010 and 0.098
   against floors of 0.082 and 0.066; Qwen 0.025 and 0.057 against 0.017.
6. In the `first` order, the release layer is in the second half of the
   network. *Held:* Gemma 36 of 48, Qwen 24 of 28.

## What this establishes (exploratory)

- **The late read uses the same statements whichever order the question
  comes in.**
  - The model does not use advance knowledge of the question to hold the
    asked value anywhere the post-program positions can read from the
    program's last line.
  - A final arithmetic statement does not hold its result, so the read
    still combines earlier lines.
  - The other variable's final line does not hold the asked value, so there
    is no end-of-program summary.
- **In Gemma, the read ends later when the question comes first.** From
  layer 30 the read is finished with the question after, but blocking from
  layer 30 still costs 10 points with the question first. In Qwen, the read
  ends at the same layer.
- **A literal last line of the asked variable is enough on its own, and
  better than the whole program.**
  - With the question first, Qwen answers 0.587 of these prompts from the
    full program but 0.971 when the post-program positions see only that
    line.
  - So these question-first errors come from reading other statements. This
    matches Phases Six and Seven, which found that question-first errors
    concentrate where the program ends on the other variable, often taking
    that variable's value
    ([Phase Seven, breakdowns](phase_seven_prompt_factorial.md#breakdowns-descriptive)).
  - It is consistent with a late read whose choice of variable is less
    reliable when the question is far away. Phase Fifteen tests that
    directly.

## What this does not establish

- **Confirmation.** Opened programs only.
- **Where the question-first errors arise.** `only_last_line` removes
  every earlier line at once. It does not separate the other variable's
  statements from the asked variable's earlier ones.
- **Representation.** "Holds the value" means the post-program positions
  can read the answer from those tokens. It says nothing about how the
  value is represented.
- **Scope.** One wording; knockout limits as before (an off-distribution
  intervention, teacher-forced scoring).

Code:
[question_order_knockout.py](../src/open_weight_lingua/question_order_knockout.py),
launched with `scripts/run_question_order_knockout.sh`.
