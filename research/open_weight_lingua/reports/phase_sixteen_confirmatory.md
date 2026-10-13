# Phase Sixteen confirmatory: with the question first, the program carries the asked variable — 2026-10-13

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

The exploratory Phase Sixteen ([report](phase_sixteen_question_route.md))
found an unpredicted result about prompts with the question first. When the
program tokens cannot attend to the question (`program_blind`), the late
read falls back to the most recent assignment of either variable:
- programs ending on the other variable are answered with that variable's
  value;
- programs ending on the asked variable stay mostly right.

Six claims were frozen
([protocol](../protocols/phase_sixteen_confirmatory_fresh.md)) before the
test programs existed. They were tested once on **768 newly generated
groups**: seed 211001, excluding every Phase Two program and the five
earlier fresh splits.

**Five of six were supported.**
- **Gemma: all three.** Blind the program to the question, and on programs
  ending on the other variable 71% of answers become that variable's value.
  Accuracy there falls by 62 points, against 7 on programs ending on the
  asked variable.
- **Qwen: two of three.**
  - Other-value answers rise by 17 points, and accuracy on programs ending
    on the other variable falls by 21 (Q1, Q2).
  - Q3 failed. Qwen also loses 18 points on programs ending on the asked
    variable, so its contrast between the strata is not resolved:
    +0.026 [−0.029, +0.082].
  - In Qwen, cutting the program from the question confuses the variable
    *and* disrupts the read more broadly. The fallback is not specifically
    to the most recent assignment.

## What ran

- **Freeze.**
  - The protocol, the new seed, the module, its tests and the launcher
    were committed and pushed at `34e3306`. The tests used test-only seeds.
  - Both manifests record that commit and the protocol hash
    `b1600247dabe…`. All 121 source hashes equal the files at `34e3306`.
- **Fresh programs.**
  - 768 groups under `fresh_2026_10f`: 1,738 attempts. 199 candidates were
    rejected for repeating an earlier program, and 771 were invalid pairs.
  - The five earlier fresh splits regenerated to their recorded
    inventories before being excluded.
  - Both families received the same programs (inventory `4758be932923…`).
  - 3,072 question-first prompts. 1,518 are in the main stratum and 1,536
    end on the asked variable. 1,518 programs have a prompt in both.
- **Settings.** As frozen. The bitwise no-op control held on every prompt.

| Family | Run | Manifest / results sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p16-confirm-gemma3-12b-20261013T004624Z-e44d86a5` | `7ce3ff3a…` / `74ae08b4…` | 3,944 s / 4,082 s |
| Qwen2.5-7B | `p16-confirm-qwen2.5-7b-20261013T020421Z-9ef6ac5d` | `1fe67baf…` / `44f9f471…` | 2,217 s / 2,322 s |

- **GPU.** The Gemma run waited 34 minutes at the gate while rf-moe's
  nightly retraining job (`cls/run_combined_v3.py`, 630 MiB) held the GPU.
  That job was logged and left alone. It ended before the launch. During
  both runs the only other GPU process was the rf-moe dashboard (451 MiB).
- **Evaluation.** One CPU evaluation wrote
  `p16-confirm-evaluation-20261013.json` (`05d071eb…`).

## Results

Accuracy, with the rate of saying the other variable's final value in
parentheses:

| Condition | Gemma, main (1,518) | Gemma, asked last (1,536) | Qwen, main | Qwen, asked last |
|---|---|---|---|---|
| Baseline | 0.822 (12.8%) | 0.891 (2.3%) | 0.383 (19.7%) | 0.620 (5.5%) |
| Program → question blocked | **0.202 (70.6%)** | **0.822** (8.2%) | **0.173 (36.8%)** | **0.439** (9.8%) |
| Reader → `What is x?` blocked (descriptive) | 0.750 (19.3%) | 0.889 (2.6%) | 0.355 (24.1%) | 0.641 (5.6%) |

**The six confirmatory hypotheses.** These are 99.17% whole-group bootstrap
intervals (10,000 resamples; seed 808100 Gemma, 808101 Qwen), Bonferroni
across the six tests.

| ID | Measure | Units | Mean | Interval | Threshold | Verdict |
|---|---|---:|---:|---|---|---|
| G1 | Gemma: rise in other-value answers, main | 1,430 | +0.578 | [+0.533, +0.622] | lower > 0.05 | **Supported** |
| G2 | Gemma: accuracy loss, main | 1,518 | +0.620 | [+0.576, +0.662] | lower > 0.10 | **Supported** |
| G3 | Gemma: main loss − asked-last loss, per program | 1,518 | +0.553 | [+0.489, +0.613] | lower > 0 | **Supported** |
| Q1 | Qwen: rise in other-value answers, main | 1,430 | +0.171 | [+0.134, +0.210] | lower > 0.05 | **Supported** |
| Q2 | Qwen: accuracy loss, main | 1,518 | +0.209 | [+0.170, +0.251] | lower > 0.10 | **Supported** |
| Q3 | Qwen: main loss − asked-last loss, per program | 1,518 | +0.026 | [−0.029, +0.082] | lower > 0 | **Not supported** |

**Predictions.**
- All six were predicted to be supported. Q3 was flagged as the closest
  call, and it failed. Its estimate (+0.026) is a third of the exploratory
  +0.080.
- The other five lie within 3 points of their exploratory values. For
  example, G1 is +0.578 against +0.585, and Q2 +0.209 against +0.235.

**The descriptive direct-route block.** Blocking the reader from only
`What is x?` leaves the answer format visible. It is not a large effect:
- Gemma's other-value answers on the main stratum rise by 6.5 points
  (Qwen 4.4), and accuracy falls by 7 points (Qwen 3).
- Programs ending on the asked variable are untouched.

So a direct route from the question's clause to the read exists, but it is
small next to the route through the program.

## What this establishes

- **In Gemma, with the question first, the program tokens carry the asked
  variable to the late read.**
  - Without that route, the read takes the most recent assignment of
    either variable. Programs ending on the other variable are answered
    with its value; programs ending on the asked variable stay right.
  - The direct route from the question to the reading positions is
    comparatively small.
- **In Qwen, cutting the program from the question also confuses the
  variable**, raising other-value answers by 17 points. But it hurts
  programs ending on the asked variable about as much. Qwen's fallback is
  not shown to be the most recent assignment; the disruption is broader.
- **How this fits the earlier phases, in Gemma.**
  - With the question after the program, the question tokens read the
    program themselves (Phase Eight).
  - With it first, the program is read with the question in view, and the
    template positions read the result. When that marking is imperfect,
    the most recent assignment wins: the question-first errors of Phase
    Fifteen.

## What this does not establish

- **What the marking is.** Tags on the asked statements, suppression of
  the other's, or a variable carried to the program's end. Phase Seventeen
  tests that.
- **Why Qwen differs.**
- **Scope.** Off-distribution blocks; one wording; two models.

Code: [route_confirm.py](../src/open_weight_lingua/route_confirm.py),
launched with `scripts/run_route_confirm.sh`.
