# Phase Seventeen confirmatory: where the program marks the asked variable, on fresh programs — 2026-10-13

```text
What is x? Reply with only the integer.
x = 7              ← the asked variable's line
y = 16             ← the other variable's lines
y = y + 1
y = y - 3          ← the last line   → 7  (y ends at 14)
```

The exploratory Phase Seventeen
([report](phase_seventeen_question_marking.md)) studied prompts with the
question first. It found that the program marks the asked variable *both*
on its own statements and on the other variable's. Blocking either set of
lines from the question raises other-value answers on programs ending on
the other variable.

Six claims were frozen
([protocol](../protocols/phase_seventeen_confirmatory_fresh.md)) before the
test programs existed. They were tested once on **768 newly generated
groups**: seed 212001, excluding every Phase Two program and the six earlier
fresh splits.

**Five of six were supported.**
- **Supported, in both models.** Blocking the asked variable's lines from
  the question raises other-value answers (G1, Q1). So does blocking the
  other variable's lines (G2, Q2).
- **Supported, in Gemma.** The mark on the other variable is not confined
  to its last line (G3).
- **Not supported: Q3.** The claim was that Qwen's last line alone carries
  more than 5 points of the effect. It came out at +0.054
  [+0.033, +0.077]: resolved above zero, but not above the 5-point
  threshold.

The Gemma effects were smaller than in the exploratory run but cleared
their thresholds easily.

## What ran

- **Freeze.**
  - The protocol, the new seed, the module, its tests and the launcher
    were committed and pushed at `bccef75`. The tests used test-only seeds.
  - Both manifests record that commit and the protocol hash
    `f64c92236559…`. All 127 source hashes equal the files at `bccef75`.
- **Fresh programs.**
  - 768 groups under `fresh_2026_10g`: 1,661 attempts. 205 candidates were
    rejected for repeating an earlier program, and 688 were invalid pairs.
  - The six earlier fresh splits regenerated to their recorded inventories
    before being excluded.
  - Both families received the same programs (inventory `6ed1d8b3ffbd…`).
  - 3,072 question-first prompts. 1,500 are in the main stratum; 1,419 of
    them have a distinct other value and are the claims' units.
- **Settings.** As frozen. The bitwise no-op control held on every prompt.

| Family | Run | Manifest / results sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p17-confirm-gemma3-12b-20261013T045634Z-11c94a0a` | `dc775e45…` / `bc978655…` | 4,870 s / 5,019 s |
| Qwen2.5-7B | `p17-confirm-qwen2.5-7b-20261013T063010Z-f36bf6d9` | `d1eed433…` / `e7a936e2…` | 2,785 s / 2,877 s |

Both runs were admitted to an idle GPU. The only other GPU process was the
rf-moe dashboard (451 MiB). One CPU evaluation wrote
`p17-confirm-evaluation-20261013.json` (`b7096daf…`).

## Results

Accuracy, with the rate of saying the other variable's final value in
parentheses:

| Program tokens blocked from the question | Gemma, main (1,500) | Gemma, asked last (1,536) | Qwen, main | Qwen, asked last |
|---|---|---|---|---|
| None (baseline) | 0.832 (12.3%) | 0.877 (3.0%) | 0.413 (19.3%) | 0.620 (5.4%) |
| The asked variable's lines | 0.451 (37.1%) | 0.915 (3.2%) | 0.211 (33.5%) | 0.473 (7.8%) |
| The other variable's lines | 0.605 (33.1%) | 0.877 (3.3%) | 0.298 (32.1%) | 0.605 (5.9%) |
| The last line | 0.773 (17.3%) | 0.894 (3.0%) | 0.411 (24.7%) | 0.596 (5.4%) |

**The six confirmatory hypotheses.** These are 99.17% whole-group bootstrap
intervals (10,000 resamples; seed 809100 Gemma, 809101 Qwen), Bonferroni
across the six tests. The units are 1,419 main-stratum prompts in 750
groups.

| ID | Measure (rise in other-value answers) | Mean | Interval | Threshold | Verdict |
|---|---|---:|---|---|---|
| G1 | Gemma: asked variable's lines blocked | +0.247 | [+0.208, +0.287] | lower > 0.05 | **Supported** |
| G2 | Gemma: other variable's lines blocked | +0.208 | [+0.172, +0.247] | lower > 0.05 | **Supported** |
| G3 | Gemma: other's lines blocked − last line blocked | +0.158 | [+0.126, +0.191] | lower > 0.05 | **Supported** |
| Q1 | Qwen: asked variable's lines blocked | +0.142 | [+0.107, +0.178] | lower > 0.05 | **Supported** |
| Q2 | Qwen: other variable's lines blocked | +0.128 | [+0.097, +0.162] | lower > 0.05 | **Supported** |
| Q3 | Qwen: last line blocked | +0.054 | [+0.033, +0.077] | lower > 0.05 | **Not supported** |

**Predictions.**
- All six were predicted to be supported. Q3 was flagged as the closest
  call, and it failed.
- **Gemma's effects shrank on fresh programs.** G1 came out at +0.247
  against +0.357 exploratory, and G2 at +0.208 against +0.295.
- **Qwen's held near their exploratory values.** Q1 came out at +0.142
  against +0.155, and Q2 at +0.128 against +0.131.

## What this establishes

- **With the question first, the program marks both variables'
  statements.** In both models:
  - hide the question from the asked variable's lines, and the read takes
    the other variable's value more often;
  - hide it from the other variable's lines, and the same happens.

  The late read relies on both marks.
- **In Gemma, the mark on the other variable is spread across its lines.**
  Its last line carries only a small part (+5 points, against +21 for all
  its lines).
- **In Qwen, the last line's share is small but present** (+5.4 points,
  above zero but below the 5-point threshold). The exploratory 57% share
  did not hold up as a large effect.
- **How this extends the account, with the question first:**
  - the program tokens attend to the question and mark which statements
    belong to the asked variable and which do not;
  - the late read uses those marks to pick the asked variable's most
    recent statement;
  - if the marks are removed, the read falls back on recency.

## What this does not establish

- **When in the network the marking happens.** Phase Eighteen tests that.
- **The form of the marks.**
- **Scope.** Off-distribution blocks; one wording; two models.

Code: [marking_confirm.py](../src/open_weight_lingua/marking_confirm.py),
launched with `scripts/run_marking_confirm.sh`.
