# Phase Seven confirmatory: question position × wording replicates on unseen data — 2026-10-06

```text
x = 7
y = 16
y = y + 1
y = y - 3          x is 7; the program ends on y (14)

original wording:  What is x? Reply with only the integer.
expanded wording:  What is x at the end of this program? Reply with only the integer.
```

The exploratory Phase Seven factorial
([report](phase_seven_prompt_factorial.md)) attributed Phase Six's accuracy
loss differently in the two models.
- **Qwen2.5-7B:** the loss came from placing the question before the
  program.
- **Gemma-3-12B:** the loss came from position plus a wording × position
  interaction. "At the end of this program" hurt only when it came first.

Those claims were then frozen as six hypotheses
([protocol](../protocols/phase_seven_confirmatory_validation_a.md)) and
tested once on the reserved `validation_a` split, which no analysis had
seen.

**All six hypotheses were supported.**
- Qwen's loss is position only. Wording and the interaction are each
  shown to be smaller than 5 points.
- Gemma's loss is position plus a wording × position interaction. The
  wording and interaction effects are about 8 points, smaller than the
  exploratory 11–12, and their intervals do not clear the 5-point
  practical reference.

## What ran

- **Freeze.**
  - The protocol, module, tests and launcher were committed and pushed at
    `7b5210f` before `validation_a` was built or any model saw it.
  - Both run manifests record that commit and the protocol hash
    `a028382fd2889d35…`.
- **Data.**
  - `validation_a`: 256 groups, bound to the frozen Phase Two split plan by
    its hash `ff060012d35d…`.
  - That is 1,024 base prompts per family, each in four formats: 4,096
    answers per family.
  - `validation_b` stays reserved.
- **Settings.** As in exploratory Phase Seven:
  - greedy decoding, at most 8 new tokens, the pinned 128-token bucket;
  - stopping at any end-of-sequence token the model declares
    (Gemma `{1, 106}`, Qwen `{151643, 151645}`);
  - scoring under each family's frozen convention (`rstrip` Gemma, `raw`
    Qwen).

| Family | Run | Manifest sha256 | Generations sha256 | Generation / wall |
|---|---|---|---|---:|
| Gemma-3-12B | `p7-confirm-gemma3-12b-20261006T015135Z-22d1a992` | `8f6a44af5b1075b6…` | `89f183806543c59b…` | 1,734 s / 1,897 s |
| Qwen2.5-7B | `p7-confirm-qwen2.5-7b-20261006T023616Z-84ec0111` | `f5e3fa2ad23e873d…` | `45f9999772cc9b56…` | 818 s / 923 s |

- **Evaluation.** One CPU evaluation over both complete runs wrote
  `p7-confirm-evaluation-20261006.json` (`1847ce74ab2deda4…`). The evaluator
  rescored all 8,192 answers and found no disagreement with the stored
  correctness.
- **Operational record.**
  - **Qwen's first launch was killed by an external SIGKILL** during the
    launcher's CPU test suite, before the run started.
    - No run directory was created, `validation_a` was not built for Qwen,
      and no model was loaded.
    - No OOM daemon was active, and the source of the signal could not be
      identified.
    - The protocol allows one relaunch after an operational failure with
      zero answers; this was it.
  - **Other GPU activity.** Small processes from another project
    (319 MiB) appeared on the GPU during Gemma's run and Qwen's setup.
    They were logged, and nothing was stopped. Greedy answers do not depend
    on them.
  - The gate logs, GPU-process log and the killed attempt's log are kept
    locally in `runs/p7-confirm-operations-20261006/`.

## Results

Exact-answer accuracy on 1,024 base prompts (correct count in parentheses):

| Format | Gemma-3-12B | Qwen2.5-7B |
|---|---:|---:|
| Original question, after | 0.973 (996) | 0.838 (858) |
| Expanded question, after | 0.969 (992) | 0.842 (862) |
| Original question, before | 0.858 (879) | 0.536 (549) |
| Expanded question, before | 0.778 (797) | 0.535 (548) |

Every Gemma answer terminated. Qwen's terminated in 1,010–1,023 of 1,024
answers per format.

**The six confirmatory hypotheses.** These are 99.17% whole-group bootstrap
intervals (10,000 resamples; seed 801100 Gemma, 801101 Qwen), Bonferroni
across the six tests.

| ID | Claim | Estimate | Interval | Verdict |
|---|---|---:|---|---|
| G1 | Gemma: question first lowers accuracy (original wording) | −0.114 | [−0.151, −0.078] | **Supported**, beyond −0.05 |
| G2 | Gemma: with the question first, the expanded wording lowers accuracy | −0.080 | [−0.125, −0.035] | **Supported**; not beyond −0.05 |
| G3 | Gemma: the wording's harm is larger when the question comes first | −0.076 | [−0.123, −0.031] | **Supported**; not beyond −0.05 |
| Q1 | Qwen: question first lowers accuracy (original wording) | −0.302 | [−0.348, −0.255] | **Supported**, beyond −0.05 |
| Q2 | Qwen: with the question first, wording changes accuracy by less than 5 points | −0.001 | [−0.046, +0.042] | **Supported** (inside ±0.05) |
| Q3 | Qwen: the position effect differs between wordings by less than 5 points | −0.005 | [−0.049, +0.038] | **Supported** (inside ±0.05) |

**Predictions.**
- G1–G3 and Q1 were predicted to be supported, and they were.
- G1 and Q1 were expected to clear the 5-point reference, and did.
- G2 and G3 were expected to clear it as well. They did not: the effects
  replicated in sign but came out smaller than the exploratory estimates
  (−0.080 against −0.118, and −0.076 against −0.105).
- Q2 and Q3 were predicted to be borderline, roughly even odds at this
  sample size. Both passed, Q3 narrowly (lower bound −0.049).

## Secondary analyses (pre-specified, 95%)

| Contrast | Gemma | Qwen |
|---|---|---|
| Position, expanded wording | −0.190 [−0.221, −0.160] | −0.307 [−0.342, −0.271] |
| Wording, question after | −0.004 [−0.013, +0.004] | +0.004 [−0.006, +0.015] |

**Last-line concentration.** This is the position effect on prompts whose
last line assigns the *other* variable, minus the effect on prompts whose
last line assigns the asked variable. The exploratory results predicted a
negative value in every cell, and it is negative in every cell.

| Wording | Gemma | Qwen |
|---|---|---|
| Original | −0.111 [−0.170, −0.053] | −0.334 [−0.418, −0.248] |
| Expanded | −0.412 [−0.475, −0.348] | −0.430 [−0.510, −0.352] |

**Other-value answers.** These are answers that equal the other variable's
final value. With the question first and the program ending on the other
variable:
- Gemma gives that value in 64 (original wording) and 174 (expanded) of
  512 such prompts;
- Qwen gives it in 104 and 134.

With the question after the program, it happens in 3–12 of those 512
prompts for either model.

## Breakdowns (descriptive)

Accuracy, question after (original) / question before (original) / question
before (expanded):

| Stratum | n | Gemma | Qwen |
|---|---:|---|---|
| Asks about x | 512 | 0.988 / 0.863 / 0.684 | 0.848 / 0.463 / 0.404 |
| Asks about y | 512 | 0.957 / 0.854 / 0.873 | 0.828 / 0.609 / 0.666 |
| Last line assigns the asked variable | 512 | 0.969 / 0.910 / 0.973 | 0.830 / 0.695 / 0.736 |
| Last line assigns the other variable | 512 | 0.977 / 0.807 / 0.584 | 0.846 / 0.377 / 0.334 |
| Asked variable last set by a literal | 506 | 0.992 / 0.812 / 0.630 | 0.893 / 0.383 / 0.447 |
| … by arithmetic | 388 | 0.951 / 0.897 / 0.933 | 0.773 / 0.662 / 0.590 |
| … by a copy | 130 | 0.962 / 0.923 / 0.892 | 0.815 / 0.754 / 0.715 |
| Asks x, program ends on y | 278 | 0.986 / 0.799 / 0.439 | 0.856 / 0.288 / 0.180 |

The evaluation file holds every cell's counts. That includes the expanded
wording after the program, the add/subtract split, the joint
variable × last-writer table, termination, and other-value counts.

## What this establishes

- **The attribution from exploratory Phase Seven replicates on unseen
  data**, with the planned Bonferroni control.
  - Qwen2.5-7B loses about 30 points when the question precedes the
    program. The wording "at the end of this program" makes no difference
    larger than 5 points, in either position.
  - Gemma-3-12B loses about 11 points from position alone. With the
    question first, the expanded wording costs about 8 more. That wording
    × position interaction is reliably below zero, but smaller than the
    exploratory estimate, and not shown to exceed 5 points.
- **Both models' question-first errors concentrate on programs that end on
  the other variable.** That is where answers often equal the other
  variable's value. For Gemma, the expanded wording sharply amplifies this.

## What this does not establish

- **Behaviour, not mechanism.** These are answer-accuracy contrasts under
  prompt formats. They do not show where, whether or how the models store
  variable values. Other-value answers do not prove the asked value was
  unavailable internally. The probe surveys (Phases Three to Six and the
  readout audit) bound readouts; they do not settle storage.
- **One pair of wordings, two models, one task family** (integers 0–19,
  short programs). The effect sizes may not transfer to other phrasings or
  tasks.
- **One confirmatory split.** `validation_a` is now used. Any further
  confirmatory question needs `validation_b` and its own frozen protocol.

Code: [prompt_factorial_confirm.py](../src/open_weight_lingua/prompt_factorial_confirm.py),
launched with `scripts/run_prompt_factorial_confirm.sh`; evaluated with
`python -m open_weight_lingua.prompt_factorial_confirm evaluate`.
