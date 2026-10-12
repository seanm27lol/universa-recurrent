# Phase Nine confirmatory: the late read targets the asked statements, on fresh programs — 2026-10-12

```text
x = 7              ← the asked variable's line
y = 16             ← the other variable's lines
y = y + 1
y = y - 3          ← ends with the program's last token "3"
What is x? Reply with only the integer.      → 7
```

The exploratory span knockout ([report](phase_nine_span_knockout.md)),
on reused calibration and pilot prompts, found the same pattern in both
models. Inside each model's confirmed late route:
- blocking the asked variable's own statements is as damaging as blocking
  the whole program;
- blocking the other variable's statements costs nothing;
- when the program ends on the other variable, the last token is
  irrelevant.

Six claims were frozen
([protocol](../protocols/phase_nine_confirmatory_fresh.md)) before the test
programs existed. Both reserved validation blocks were spent, so the code
generated **256 new groups** (seed 206001), excluding every program in the
five Phase Two splits.

**All six were supported, with nearly the exploratory effect sizes.**

**Correction, 2026-10-12.** The "copy" category and the breakdown labelled
"copies the other variable" were mislabelled. The generator lets a variable
copy itself (`x = x`), and 148 of the 156 final copies here are such
self-copies. The "copies the other variable" breakdown was computed as "has
*any* copy statement" without checking the source. Only 42 of its 268
prompts (32 of 218 in the confirmatory set) contain a genuine cross-variable
copy (`x = y`). On those few prompts the results are noisy and inconsistent
across runs and models. So the earlier statement that copied values "sit at
the asked variable's copy statement by the late layers" is **not
established**. The pre-registered confirmatory claims do not depend on this
breakdown and are unaffected. Numbers below are unchanged; the labels and
that inference are corrected.

## What ran

- **Freeze.**
  - The protocol, the new seed, the module, its tests and the launcher
    were committed and pushed at `1bf1d52`.
  - The tests used a test-only seed, so the confirmatory programs were
    generated for the first time inside the frozen runs.
  - Both manifests record that commit, the protocol hash `5f63a037f0e9…`
    and source hashes identical to the frozen code.
- **Fresh programs.**
  - 256 groups under split `fresh_2026_10`: 469 generation attempts, 10
    candidates rejected because they repeated a Phase Two program.
  - Both families received the same programs (inventory hash
    `8e0d30ab8915…`, checked by the evaluator).
  - 1,024 prompts per family, in the original format.
- **Settings.** As in exploratory Phase Nine: Qwen's answer positions from
  layer 21, Gemma's post-program positions from layer 24, teacher-forced
  digit accuracy, and the bitwise no-op control, which held on every
  prompt.

| Family | Run | Manifest / results sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p9-confirm-gemma3-12b-20261012T005339Z-14893e44` | `b3b81b28…` / `f7243c6b…` | 831 s / 984 s |
| Qwen2.5-7B | `p9-confirm-qwen2.5-7b-20261012T011909Z-d5cb3c8f` | `9cdb2f3c…` / `ff070b06…` | 469 s / 554 s |

- **Admission.** The GPU gate waited about 9 minutes for another project's
  612 MiB process to finish. No foreign process appeared during either
  run.
- **Evaluation.** One CPU pass over both runs wrote
  `p9-confirm-evaluation-20261012.json` (`6e00b8f8…`).

## Results

Digit accuracy on the fresh programs (1,024 prompts per family):

| Condition | Gemma-3-12B | Qwen2.5-7B |
|---|---:|---:|
| Baseline | 0.964 | 0.816 |
| The asked variable's lines blocked | 0.265 | 0.237 |
| The other variable's lines blocked | 0.963 | 0.837 |
| The last token blocked (all prompts) | 0.784 | 0.624 |
| The last token blocked, program ends on the other variable (512) | 0.977 against 0.975 | 0.844 against 0.838 |

**The six confirmatory hypotheses.** These are 99.17% whole-group bootstrap
intervals on the per-prompt loss (10,000 resamples; seed 803100 Gemma,
803101 Qwen), Bonferroni across the six tests.

| ID | Blocked span | Prompts | Claim | Mean loss | Interval | Verdict |
|---|---|---|---|---:|---|---|
| G1 | Gemma: the asked variable's lines | 1,024 | damaged | +0.699 | [+0.649, +0.749] | **Supported** |
| G2 | Gemma: the other variable's lines | 1,024 | survives | +0.001 | [−0.010, +0.013] | **Supported** |
| G3 | Gemma: the last token | 512 (ends on the other variable) | survives | −0.002 | [−0.021, +0.016] | **Supported** |
| Q1 | Qwen: the asked variable's lines | 1,024 | damaged | +0.579 | [+0.524, +0.631] | **Supported** |
| Q2 | Qwen: the other variable's lines | 1,024 | survives | −0.021 | [−0.036, −0.008] | **Supported** |
| Q3 | Qwen: the last token | 512 (ends on the other variable) | survives | −0.006 | [−0.021, +0.006] | **Supported** |

**Predictions.** All six were predicted to be supported, and every
estimate lies within its predicted interval. One detail: Q2's loss is
negative, −0.021 [−0.036, −0.008], an interval that excludes 0. Blocking
the other variable's lines slightly *improves* Qwen's accuracy, so for Qwen
those statements are mild interference at the late read. The exploratory
run showed the same sign (−0.012).

**Breakdowns (descriptive)**, accuracy with the asked variable's lines
blocked, by the asked variable's last update:

| Last update | Gemma (baseline → blocked) | Qwen (baseline → blocked) |
|---|---|---|
| A literal (486) | 0.992 → 0.414 | 0.866 → 0.383 |
| Arithmetic (386) | 0.935 → 0.140 | 0.756 → 0.093 |
| A copy, 152 (mostly self-copies `x = x`) | 0.947 → 0.105 | 0.809 → 0.138 |

*Corrected:* the 218 prompts labelled "copies the other variable" are
mostly self-copies. On the 32 genuine cross-variable copies, blocking the
other variable's lines gives 0.625 against 0.688 (Gemma) and 0.562 against
0.531 (Qwen). Blocking the asked lines gives 0.375 and 0.469. These are too
few prompts to locate a copied value.

## What this establishes

- **On unseen programs, the late read that both models need retrieves the
  asked variable's own statements, and only those.**
  - Blocking them inside the confirmed route (Gemma's question tokens,
    layers 24+; Qwen's answer positions, layers 21+) costs 70 and 58
    points.
  - Blocking the other variable's statements, which are as long on
    average, costs nothing.
- **No end-of-program summary is needed.** When the program ends on the
  other variable, its last token is irrelevant to the answer, in both
  models.
- **With Phases Eight and Nine together:** after reading the question, both
  models look back at the asked variable's statements in a late window.

## What this does not establish

- **Where copied values sit.** Genuine cross-variable copies are too rare
  (32 prompts) to say.
- **Single-token roles.** The variable name, the operator and the operand
  within a statement were not separated.
- **Knockout limits.** An off-distribution intervention; teacher-forced
  digit accuracy; one route per model; one prompt format.
- **Same distribution.** The fresh programs come from the same generator as
  the earlier splits. This tests new programs, not new tasks.

Code: [span_knockout_confirm.py](../src/open_weight_lingua/span_knockout_confirm.py),
launched with `scripts/run_span_knockout_confirm.sh`.
