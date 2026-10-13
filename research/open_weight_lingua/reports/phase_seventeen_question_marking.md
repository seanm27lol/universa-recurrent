# Phase Seventeen: with the question first, the program both tags the asked statements and suppresses the other's — exploratory — 2026-10-13

```text
What is x? Reply with only the integer.
x = 7              ← the asked variable's line
y = 16             ← the other variable's lines
y = y + 1
y = y - 3          ← the last line   → 7  (y ends at 14)
```

Phase Sixteen confirmed that, with the question first, the asked variable
reaches the late read through the program tokens. Gemma's read falls back
to the most recent assignment of either variable when the whole program is
cut off from the question. Phase Seventeen
([brief](../protocols/phase_seventeen_question_marking_brief.md)) cut off
only parts of the program, to see where the mark sits.

**Both kinds of marking act. In Gemma, neither one sits at the program's
end.**
- **Gemma.** Measured on programs ending on the other variable, as shares
  of the full block's rise in other-value answers:
  - blocking only the asked variable's lines from the question gives 61%;
  - only the other variable's lines, 50%;
  - only the last line, 14%.

  So Gemma tags the asked statements *and* marks the other's, spread
  across its lines.
- **Qwen.** The same shares are 92%, 78% and 57%. Every part matters, and
  the other variable's final line alone carries more than half.

The brief predicted tagging alone. Three of five predictions held for
Gemma, and two for Qwen. These are exploratory results on opened programs.

## What ran

- **Freeze.** The brief, module, tests and launcher were committed and
  pushed at `213c466` before any forward. Both manifests record that commit
  and the brief's hash `52bb3d3c21f1…`. All 124 source hashes equal the
  files at `213c466`.
- **Prompts.** Phase Eight's 1,536 calibration and pilot prompts per
  family, question first.
- **Conditions.** From layer 0, a subset of program tokens is blocked from
  every question token. Teacher-forced scoring of the answer and the other
  variable's final value. The bitwise no-op control held on every prompt.

| Family | Run | Manifest / results / summary sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p17-marking-gemma3-12b-20261013T030410Z-144d32e4` | `5c7f0894…` / `f0689da8…` / `41fde71c…` | 2,931 s / 3,079 s |
| Qwen2.5-7B | `p17-marking-qwen2.5-7b-20261013T040509Z-9ee12631` | `6c778699…` / `e1185ddc…` / `d0a54b5c…` | 1,660 s / 1,748 s |

- **GPU.** Both runs were admitted to an idle GPU. During the Gemma run,
  two small foreign processes (375 and 362 MiB) appeared at 03:21 and 03:30
  UTC. They were logged and left alone. Teacher-forced scores do not depend
  on co-tenancy: earlier phases reproduced recorded scores exactly.

## Results

Accuracy, with the rate of saying the other variable's final value in
parentheses. The main stratum is programs ending on the other variable
whose answer does not depend on it (758). "Asked last" is programs ending
on the asked variable (768).

| Program tokens blocked from the question | Gemma, main | Gemma, asked last | Qwen, main | Qwen, asked last |
|---|---|---|---|---|
| None (baseline) | 0.813 (14.6%) | 0.900 (3.0%) | 0.393 (24.1%) | 0.663 (5.4%) |
| All (`program_blind`) | 0.194 (73.1%) | 0.849 (6.7%) | 0.158 (40.9%) | 0.512 (10.0%) |
| The asked variable's lines | 0.375 (50.3%) | 0.924 (3.0%) | 0.218 (39.6%) | 0.513 (7.1%) |
| The other variable's lines | 0.515 (44.2%) | 0.905 (2.9%) | 0.282 (37.2%) | 0.638 (5.6%) |
| The last line | 0.719 (22.8%) | 0.902 (3.0%) | 0.361 (33.7%) | 0.656 (5.4%) |

`program_blind` reproduces Phase Sixteen's exploratory cells exactly. These
are the same prompts and the same condition.

**Rise in other-value answers on the main stratum.** Each cell gives the
rise, then its share of the `program_blind` rise. Intervals are 99.17%
whole-group bootstrap intervals on these 384 groups, for reference only.

| Block | Gemma | Qwen |
|---|---|---|
| All | +0.585 | +0.169 |
| The asked variable's lines | +0.357, **61%** | +0.155, **92%** |
| The other variable's lines | +0.295, **50%** | +0.131, **78%** |
| The last line | +0.082, **14%** | +0.096, **57%** |

**Predictions.**
1. The no-op control is exact. *Held.*
2. `program_blind` reproduces Phase Sixteen's rise: at least 30 points for
   Gemma and 8 for Qwen. *Held:* +58.5 and +16.9.
3. **Tagging.** `asked_blind` produces at least half of that rise. *Held:*
   61% and 92%.
4. `other_blind` produces less than half. *Failed:* Gemma reached 50.4%,
   just over the line, and Qwen 78%.
5. `last_line_blind` produces less than a quarter. *Held for Gemma* (14%).
   *Failed for Qwen* (57%).

The brief's reading rule: "If 4 fails and 3 holds, both kinds of marking
act." For Gemma it fails only marginally. "If 5 fails, the variable is
carried to the end" applies to Qwen only, and it is ambiguous there. On the
main stratum the last line *is* the other variable's most recent
statement, so a large `last_line_blind` effect also fits marking the other
variable's latest assignment.

## What this establishes (exploratory)

- **With the question first, the program marks both variables' statements
  as it reads them.**
  - Hide the question from the asked variable's lines, and the read more
    often takes the other variable's value. Hide it from the other
    variable's lines, and the same happens.
  - The read needs the asked statements tagged as relevant *and* the
    other's marked as not.
  - In Gemma the two partial blocks together exceed the full block's
    effect (61% + 50%), so the two marks overlap rather than add.
- **In Gemma the mark is not concentrated at the program's end.** The last
  line carries 14% of the effect.
- **In Qwen, the other variable's last statement matters most.** Its line
  alone carries 57% of the effect. That fits Qwen's strong pull toward the
  most recent assignment (Phase Fifteen). Marking the competing latest
  statement is where the question helps most.
- **Programs ending on the asked variable are unaffected** by the partial
  blocks in Gemma: without the mark, the recency default gives the right
  answer there. In Qwen, blocking the asked variable's lines costs 15
  points there too. This matches the broader disruption Phase Sixteen's
  confirmatory test found for Qwen.

## What this does not establish

- **Confirmation.** Opened programs only.
- **The form of the mark.** "Tagging" and "marking" name what the blocks
  reveal, not a representation.
- **Separating Qwen's last-line effect** from marking the other's latest
  statement.
- **Scope.** Off-distribution blocks; one wording.

Code:
[question_marking_knockout.py](../src/open_weight_lingua/question_marking_knockout.py),
launched with `scripts/run_question_marking_knockout.sh`.
