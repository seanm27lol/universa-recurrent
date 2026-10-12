# Phases Fourteen and Fifteen confirmatory: the late read with the question first, on fresh programs — 2026-10-12

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

Two exploratory phases studied prompts with the question placed before the
program.
- **Phase Fourteen** ([report](phase_fourteen_question_order.md)) found no
  end-of-program summary, and that Gemma's read ends later than with the
  question after.
- **Phase Fifteen** ([report](phase_fifteen_variable_choice.md)) found that
  hiding the other variable's lines from the post-program positions removes
  the question-first loss.

Eight claims were frozen
([protocol](../protocols/phase_fifteen_confirmatory_fresh.md)) before the
test programs existed. They were tested once on **512 newly generated
groups**: seed 210001, excluding every Phase Two program and the four
earlier fresh splits.

**Seven of eight were supported.**
- **Supported.** In both models, hiding the other variable's lines restores
  question-after accuracy on programs that end on that variable. It even
  slightly exceeds it, and it is a large gain over the question-first
  baseline. Gemma's read is not finished by layer 30 with the question
  first; Qwen's is finished by layer 24, as with the question after.
  Qwen's last line carries less than 10 points of the answer.
- **Not supported: G3.** The claim was that Gemma's last line alone carries
  less than 10 points of the answer with the question first. The upper
  bound is 0.114. Post hoc, about 40% of the excess comes from prompts
  where the answer coincides with a number on the last line or with the
  other variable's value.

## What ran

- **Freeze.**
  - The protocol, the new seed, the module, its tests and the launcher
    were committed and pushed at `bca0b10`. The tests used test-only seeds.
  - Both manifests record that commit and the protocol hash
    `d01b058057f1…`. All 115 source hashes equal the files at `bca0b10`.
- **Fresh programs.**
  - 512 groups under `fresh_2026_10e`: 1,190 attempts. 150 candidates were
    rejected for repeating an earlier program, and 528 were invalid pairs.
  - The four earlier fresh splits regenerated to their recorded
    inventories before being excluded.
  - Both families received the same programs (inventory `4f744c7c0f1b…`).
  - 2,048 prompts in both orders. **1,010** are in the main stratum:
    programs ending on the other variable, with an answer that does not
    depend on it.
- **Settings.** As frozen. The bitwise no-op control held on both orders of
  every prompt.

| Family | Run | Manifest / results sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p15-confirm-gemma3-12b-20261012T194705Z-ceaafb30` | `83a075b8…` / `1c51cb9f…` | 3,304 s / 3,457 s |
| Qwen2.5-7B | `p15-confirm-qwen2.5-7b-20261012T205421Z-f84c5e0c` | `99a8ac57…` / `919a1ad6…` | 1,882 s / 1,974 s |

Both runs were admitted to an idle GPU. The only other GPU process was the
rf-moe dashboard (451 MiB). One CPU evaluation wrote
`p15-confirm-evaluation-20261012.json` (`7dfea39d…`).

## Results

**Accuracy** (teacher-forced). The rate of saying the other variable's
final value is in parentheses.

| | Gemma, main (1,010) | Gemma, all (2,048) | Qwen, main | Qwen, all |
|---|---:|---:|---:|---:|
| Question after, baseline | 0.964 | 0.937 | 0.848 | 0.800 |
| Question first, baseline | 0.804 (16.2%) | 0.847 (9.5%) | 0.398 (20.5%) | 0.511 (13.3%) |
| Question first, `block_other` | **0.990** (5.2%) | 0.946 (3.8%) | **0.880** (5.6%) | 0.770 (5.3%) |
| Question first, `only_last_line` | 0.132 | 0.258 | 0.064 | 0.220 |
| Question first, `floor` (whole program blocked) | 0.059 | 0.062 | 0.014 | 0.030 |
| Question first, `late` (blocked from layer 30 / 24) | 0.645 | 0.765 | 0.393 | 0.513 |

**The eight confirmatory hypotheses.** These are 99.375% whole-group
bootstrap intervals (10,000 resamples; seed 807100 Gemma, 807101 Qwen),
Bonferroni across the eight tests.

| ID | Measure | Prompts | Mean | Interval | Threshold | Verdict |
|---|---|---|---:|---|---|---|
| G1 | Gemma `residual_gap` (after baseline − first `block_other`) | main | −0.026 | [−0.048, −0.006] | upper < 0.05 | **Supported** |
| G2 | Gemma `recovery` (first `block_other` − first baseline) | main | +0.186 | [+0.144, +0.231] | lower > 0.05 | **Supported** |
| G3 | Gemma `summary` (first `only_last_line` − `floor`) | main | +0.072 | [+0.031, +0.114] | upper < 0.10 | **Not supported** |
| G4 | Gemma `late_loss` (first baseline − blocked from layer 30) | all | +0.082 | [+0.060, +0.105] | lower > 0.05 | **Supported** |
| Q1 | Qwen `residual_gap` | main | −0.033 | [−0.066, −0.002] | upper < 0.05 | **Supported** |
| Q2 | Qwen `recovery` | main | +0.482 | [+0.429, +0.539] | lower > 0.05 | **Supported** |
| Q3 | Qwen `summary` | main | +0.050 | [+0.028, +0.076] | upper < 0.10 | **Supported** |
| Q4 | Qwen `late_loss` (blocked from layer 24) | all | −0.002 | [−0.017, +0.013] | upper < 0.05 | **Supported** |

**Predictions.** All eight were predicted to be supported, and G3 was not.
- Its estimate, +0.072, is more than twice the exploratory +0.031. Its
  interval excludes zero. So does Q3's: [+0.028, +0.076].
- Q1 was predicted to be the closest call. Instead, both residual gaps came
  out *negative*: hiding the other variable's lines with the question first
  beats the unmodified question-after prompt.

**Post-hoc breakdown of G3** (descriptive; it cannot change a verdict).
Some main-stratum prompts can be answered from the last line by
coincidence:
- in 54, the other variable ends on the same value as the asked one;
- in 35, the answer is the literal on the last line.

| Main-stratum prompts | n | Gemma: `only_last_line` / `floor` | Qwen: `only_last_line` / `floor` |
|---|---:|---|---|
| Answer equals the other variable's value | 54 | 0.463 / 0.037 | 0.333 / 0.019 |
| Answer is the number on the last line | 35 | 0.286 / 0.086 | 0.114 / 0.000 |
| No such coincidence | 921 | **0.106 / 0.060** | **0.047 / 0.014** |

Without coincidences, the last line alone carries 4.6 points (Gemma) and
3.3 points (Qwen) of the answer.

## What this establishes

- **On unseen programs, the question-first loss is a selection failure in
  the late read.**
  - Hide the other variable's lines from the post-program positions, and
    the question-first accuracy on programs ending on that variable
    returns to, and slightly exceeds, the question-after level: Gemma
    0.990 against 0.964, Qwen 0.880 against 0.848.
  - Wrong answers equal to the other variable's value drop to about 5%.
  - The asked variable's statements carry what is needed. The read's
    choice among statements is what fails.
- **Gemma's read ends later with the question first.** Blocking from layer
  30 still costs 8.2 points. Qwen's read is finished by layer 24 in both
  orders.
- **The end-of-program carry is small, not absent.** With the question
  first, the program's last line alone gives 5–7 points above the floor.
  About 40% of that is numeric coincidence. The rest (3–5 points) is a
  small trace of the asked value at the end of the program, seen only with
  the question first. The confirmatory test did not exclude it for Gemma at
  the 10-point margin. Phase Fourteen's "no end-of-program summary" should
  therefore read: **no summary large enough to answer from.**

## What this does not establish

- **Why the read chooses worse with the question first.** The distance
  between the question and the reading positions is a hypothesis, not a
  result.
- **Normal operation.** The blocks are off-distribution. They show what the
  read *can* do without the distractors.
- **Scope.** One wording; two models.

Code: [order_confirm.py](../src/open_weight_lingua/order_confirm.py),
launched with `scripts/run_order_confirm.sh`.
