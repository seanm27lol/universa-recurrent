# Phase Eight confirmatory: answer routing replicates on validation_b — 2026-10-06

```text
x = 7
y = 16
y = y + 1
y = y - 3
What is x? Reply with only the integer.      → 7
```

The exploratory knockout ([report](phase_eight_knockout.md)) found that
both models consult the program late, but by different routes.
- **Qwen2.5-7B:** the answer position reads the program itself, in layers
  21–23 of 28.
- **Gemma-3-12B:** the answer positions never need the program directly.
  The question and template tokens read it in layers 24–29 of 48.

Six claims from those results were frozen
([protocol](../protocols/phase_eight_confirmatory_validation_b.md)) and
tested once on `validation_b`, the last reserved split, which no analysis
had seen.

**All six were supported.** The routing difference between the two models
replicates on unseen programs with nearly the exploratory effect sizes.

## What ran

- **Freeze.**
  - The protocol, module, tests and launcher were committed and pushed at
    `8d80475` before `validation_b` was built or any model saw it.
  - Both manifests record that commit, the protocol hash
    `9888aa423aa1…`, and source hashes identical to the frozen code.
- **Data.**
  - `validation_b`: 256 groups, bound to the frozen Phase Two split plan by
    its hash.
  - 1,024 prompts per family in the original format (question after the
    program).
- **Settings.** Exactly as in the exploratory run:
  - the program span is located by decoding;
  - teacher-forced digit accuracy;
  - the per-layer additive-mask knockout in all heads;
  - the bitwise no-op control, which held on every prompt.

| Family | Run | Manifest / results sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p8-confirm-gemma3-12b-20261006T211531Z-e69aeafa` | `a5add47b…` / `d820eab5…` | 773 s / 927 s |
| Qwen2.5-7B | `p8-confirm-qwen2.5-7b-20261006T213942Z-cd61e14a` | `12e6c7ca…` / `b6bb3562…` | 436 s / 540 s |

- **Admission.** Both runs were admitted to an idle GPU at first try. One
  229 MiB process from another project appeared briefly during Gemma's run
  and was logged.
- **Evaluation.** One CPU pass over both runs wrote
  `p8-confirm-evaluation-20261006.json` (`f6b67d02…`).

## Results

Teacher-forced digit accuracy (1,024 prompts per family):

| Gemma-3-12B | Accuracy |
|---|---:|
| Baseline | 0.960 |
| Every position after the program blocked from layer 24 | 0.207 |
| Every position after the program blocked from layer 30 | 0.946 |
| Answer positions blocked at every layer | 0.941 |

| Qwen2.5-7B | Accuracy |
|---|---:|
| Baseline | 0.810 |
| Answer positions blocked from layer 21 | 0.244 |
| Answer positions blocked from layer 24 | 0.812 |
| Every position after the program blocked from layer 24 | 0.813 |

**The six confirmatory hypotheses.** Loss is the per-prompt drop in
accuracy. Intervals are 99.17% whole-group bootstrap intervals (10,000
resamples; seed 802100 Gemma, 802101 Qwen), Bonferroni across the six tests.
"Survives" needs the upper bound below 0.05; "damaged" needs the lower bound
above 0.05.

| ID | Knockout | Claim | Mean loss | Interval | Verdict |
|---|---|---|---:|---|---|
| G1 | Gemma, after the program, from 30 | survives | +0.014 | [−0.003, +0.032] | **Supported** |
| G2 | Gemma, after the program, from 24 | damaged | +0.753 | [+0.703, +0.799] | **Supported** |
| G3 | Gemma, answer positions, every layer | survives | +0.019 | [0.000, +0.038] | **Supported** |
| Q1 | Qwen, answer positions, from 24 | survives | −0.002 | [−0.009, +0.004] | **Supported** |
| Q2 | Qwen, answer positions, from 21 | damaged | +0.565 | [+0.508, +0.621] | **Supported** |
| Q3 | Qwen, after the program, from 24 | survives | −0.004 | [−0.012, +0.003] | **Supported** |

**Predictions.** All six were predicted to be supported, and each estimate
lies inside the interval expected from the exploratory spread (for example,
G2 expected about [+0.69, +0.79], observed [+0.70, +0.80]).

**Breakdowns (descriptive)** by the asked variable's last update:

| Knockout | Literal (504) | Arithmetic (412) | Copy (108) |
|---|---|---|---|
| Gemma baseline / blocked after the program from 24 | 0.998 / 0.351 | 0.908 / 0.073 | 0.981 / 0.046 |
| Qwen baseline / answer positions blocked from 21 | 0.903 / 0.419 | 0.689 / 0.063 | 0.833 / 0.120 |

When the asked value was last written as a literal, part of it survives the
damaging knockouts. When it was computed or copied, almost nothing does. The
exploratory run showed the same.

## What this establishes

- **On unseen programs, both models need a late look back at the program
  tokens after reading the question, through different routes.**
  - In Qwen, the answer position itself must attend to the program in
    layers 21–23. Blocking it there costs 57 points. Blocking it from layer
    24 on costs nothing.
  - In Gemma, the answer positions do not need the program at any layer
    (2 points). The positions after the program (the question and template
    tokens) must read it in layers 24–29: blocking them from 24 costs 75
    points, from 30 about 1.
- **This is a confirmed intervention result.** It concerns what the models'
  answers use, under this intervention, not just what probes can decode.

## What this does not establish

- **Which program tokens are read.** The blocked span is the whole program,
  including its last token. Reading the raw statements and reading
  something stored at the program tokens during processing are not
  separated.
- **Knockout is an off-distribution intervention.** The claims concern its
  effect on teacher-forced digit accuracy.
- **Grid resolution.** The critical windows are bracketed (Qwen 21–23,
  Gemma 24–29), not mapped layer by layer.
- **One prompt format** (question after the program), **one task family,
  two models.**

## Reserved data

`validation_b` is now used, so both reserved validation blocks are spent.
Further confirmatory work needs newly generated programs under a new, frozen
split plan. The natural next question is which program tokens are read: the
asked variable's own lines, or the last program token where a summary could
be stored. It needs exploratory work on reused data first.

Code: [answer_knockout_confirm.py](../src/open_weight_lingua/answer_knockout_confirm.py),
launched with `scripts/run_answer_knockout_confirm.sh`.
