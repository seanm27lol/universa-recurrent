# Phase Nineteen confirmatory: the question-first two-stage account at Gemma-3-27B, on fresh programs — 2026-10-13

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

The exploratory Phase Nineteen
([report](phase_nineteen_scale_question_first.md)) found that, with the
question first, Gemma-3-27B's program tokens read the question in layers
15–30 of 62 and carry the asked variable to the late read. Five claims were
frozen ([protocol](../protocols/phase_nineteen_confirmatory_fresh.md))
before the test programs existed. They were tested once on **768 newly
generated groups**: seed 214001, excluding every Phase Two program and the
eight earlier fresh splits.

**Four of five were supported.**
- **The program carries the asked variable at 27B.** Cut it off from the
  question, and other-value answers rise by 64 points (S1).
- **The marking is finished by layer 31 of 62**, half the depth (S2).
- **It has begun below layer 23** (S4).
- **The first seven layers do not mark** (S5).
- **Not supported: S3.** The claim was that more than 5 points of the
  marking happen after layer 23. It came out at +0.061 [+0.041, +0.083]:
  above zero, below the threshold. This is the same pattern as Phase
  Eighteen's G2 and Q2.

## What ran

- **Freeze.**
  - The protocol, the new seed, the module, its tests and the launcher
    were committed and pushed at `6b8ad4f`. The tests used test-only seeds.
  - The manifest records that commit and the protocol hash
    `77fbe5bc3ab9…`. All 137 source hashes equal the files at `6b8ad4f`.
- **Fresh programs.**
  - 768 groups under `fresh_2026_10i`: 1,822 attempts. 267 candidates were
    rejected for repeating an earlier program, and 787 were invalid pairs.
  - The eight earlier fresh splits regenerated to their recorded
    inventories before being excluded.
  - Inventory `86971c2dc98b…`.
  - **1,424 prompts in 755 groups** where the mark decides the answer.
- **Settings.** As frozen; Gemma-3-27B under its lock, peak reserved GPU
  memory 55.4 GB. The bitwise no-op control held on every prompt.

| Run | Manifest / results sha256 | Knockouts / wall |
|---|---|---:|
| `p19-confirm-gemma3-27b-20261013T142219Z-23f56073` | `fb0324f6…` / `a78d6254…` | 7,148 s / 7,476 s |

The run was admitted to an idle GPU. The only other GPU process was the
rf-moe dashboard (451 MiB). One CPU evaluation wrote
`p19-confirm-evaluation-20261013.json` (`7b8f8075…`).

## Results

| Program blocked from the question in layers | none | all (0–61) | 31–61 | 23–61 | 0–22 | 0–6 |
|---|---:|---:|---:|---:|---:|---:|
| Accuracy | 0.952 | 0.206 | 0.954 | 0.842 | 0.279 | 0.953 |
| Says the other value | 0.072 | 0.708 | 0.072 | 0.133 | 0.661 | 0.073 |

**The five confirmatory hypotheses** (rise in other-value answers). These
are 99% whole-group bootstrap intervals (10,000 resamples, seed 811100),
Bonferroni across the five tests.

| ID | Blocked layers | Mean | Interval | Threshold | Verdict |
|---|---|---:|---|---|---|
| S1 | all | +0.636 | [+0.593, +0.679] | lower > 0.30 | **Supported** |
| S2 | 31–61 | −0.001 | [−0.004, +0.002] | upper < 0.05 | **Supported** |
| S3 | 23–61 | +0.061 | [+0.041, +0.083] | lower > 0.05 | **Not supported** |
| S4 | 0–22 | +0.588 | [+0.545, +0.634] | lower > 0.05 | **Supported** |
| S5 | 0–6 | +0.001 | [−0.004, +0.005] | upper < 0.05 | **Supported** |

**Predictions.**
- All five were predicted to be supported. S3 was flagged as the closest
  call, and it failed. Its estimate is about three quarters of the
  exploratory value (+0.061 against +0.082).
- S1 and S4 also shrank (+0.636 against +0.751, and +0.588 against
  +0.674), but stayed far above their thresholds.

## What this establishes

- **On unseen programs, the question-first two-stage account holds at
  Gemma-3-27B.**
  - In layers below 23, the program tokens read the question and mark the
    variables' statements.
  - The marking is finished by layer 31 of 62, with only a small remainder
    after layer 23.
  - The late read follows, and without the marks it takes the other
    variable's value.
- **Across three models, the marking ends early and mostly by about 37% of
  depth.** Phases Eighteen and Nineteen agree:
  - Gemma-3-12B: mostly by 18 of 48, finished by 24.
  - Gemma-3-27B: mostly by 23 of 62, finished by 31.
  - Qwen2.5-7B: mostly by 14 of 28, finished by 17.
  - In each, the confirmatory test found the remainder after the "mostly"
    layer to be above zero but under 5 points.
- **In both Gemma sizes the marking finishes at half the depth.** The late
  read sits deeper in the larger model (Phase Thirteen), so the two stages
  separate further with scale.

## What this does not establish

- **A scaling law.** Two sizes in one family.
- **Single-layer edges.**
- **The form of the marks.**
- **Scope.** Off-distribution blocks; one wording.

Code:
[scale_marking_confirm.py](../src/open_weight_lingua/scale_marking_confirm.py),
launched with `scripts/run_scale_marking_confirm.sh`.
