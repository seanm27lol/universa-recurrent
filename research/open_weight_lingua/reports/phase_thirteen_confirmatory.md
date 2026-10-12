# Phase Thirteen confirmatory: the account at Gemma-3-27B, on fresh programs — 2026-10-12

```text
x = 2              ← v2
y = 3
x = x + 9          ← v1 = 11
y = 5
x = 7              ← v0 = 7, the answer
What is x? Reply with only the integer.
```

The exploratory Phase Thirteen ([report](phase_thirteen_scale.md)) found
that Gemma-3-27B follows the account confirmed on the smaller models.
- **Routing.** Its answer positions never need the program. Its
  post-program positions read the program until layer 46 of 62.
- **Selection.** It takes the most recent intact assignment.

Six claims were frozen
([protocol](../protocols/phase_thirteen_confirmatory_fresh.md)) before the
test programs existed. They were tested once on **1,536 newly generated
groups**: seed 209001, excluding every Phase Two program and the three
earlier fresh splits.

**All six were supported.** R2 was predicted to be the closest call, and it
was: the blocking from layer 46 costs 2.9 points, with an upper bound of
4.4 against the 5-point threshold.

## What ran

- **Freeze.**
  - The protocol, the new seed, the module, its tests and the launcher
    were committed and pushed at `fac33db`. The tests used test-only seeds.
  - The manifest records that commit and the protocol hash
    `d3bdcf2beb69…`. All 106 source hashes equal the files at `fac33db`.
- **Fresh programs.**
  - 1,536 groups under `fresh_2026_10d`: 3,074 attempts. 250 candidates
    were rejected for repeating an earlier program, and 1,288 were invalid
    pairs.
  - The three earlier fresh splits regenerated to their recorded
    inventories before being excluded.
  - Inventory `e2c9ab251db8…`.
- **Prompts.**
  - Routing: the 2,048 prompts of the first 512 groups.
  - Selection: **430 eligible prompts in 223 groups.** That is more than
    Phase Twelve's 385 on its 1,536 groups.
- **Settings.** As in exploratory Phase Thirteen: Gemma-3-27B under its lock
  (`ca078e52`). The bitwise no-op control held on every prompt.

| Run | Manifest / results sha256 | Routing / selection / wall |
|---|---|---:|
| `p13-confirm-gemma3-27b-20261012T123053Z-84de33d2` | `b2e20cd0…` / `48a54933…` | 3,640 s / 2,148 s / 6,089 s |

- **GPU.** The run was admitted to an idle GPU at 12:22 UTC. Its peak
  reserved memory was 55.4 GB. The only other GPU process throughout was
  the rf-moe dashboard (451 MiB), which the gate allows.
- **Evaluation.** One CPU evaluation wrote `p13-confirm-evaluation-20261012.json`
  (`2200f721…`).

## Results

**Routing.** Teacher-forced digit accuracy, 2,048 prompts:

| Condition | Accuracy |
|---|---:|
| Baseline | 0.969 |
| Answer positions blocked from the program, from layer 0 | 0.960 |
| Every post-program position blocked, from layer 46 | 0.939 |
| Every post-program position blocked, from layer 38 | 0.780 |

**Selection.** Rate at which each value is said, 430 prompts:

| Degraded | v0 / v1 / v2 |
|---|---|
| Nothing | 0.993 / 0.002 / 0.007 |
| The final assignment | 0.249 / **0.753** / 0.009 |
| The final and the previous | 0.102 / 0.063 / **0.816** |
| Only the previous | **0.984** / 0.000 / 0.019 |

Among wrong answers, v1 makes up 92% under the final-only block, and v2
makes up 82% under the double block.

**The six confirmatory hypotheses.** These are 99.17% whole-group bootstrap
intervals (10,000 resamples, seed 806100), Bonferroni across the six tests.

| ID | Measure | Mean | Interval | Threshold | Verdict |
|---|---|---:|---|---|---|
| R1 | loss, answer positions blocked from layer 0 | +0.008 | [−0.001, +0.019] | upper < 0.05 | **Supported** |
| R2 | loss, post-program positions blocked from layer 46 | +0.029 | [+0.016, +0.044] | upper < 0.05 | **Supported** |
| R3 | loss, post-program positions blocked from layer 38 | +0.189 | [+0.162, +0.216] | lower > 0.05 | **Supported** |
| S1 | v1 gain when the final assignment is degraded | +0.751 | [+0.679, +0.818] | lower > 0.10 | **Supported** |
| S2 | v2 gain when the previous is degraded too | +0.807 | [+0.735, +0.870] | lower > 0.10 | **Supported** |
| S3 | loss when only the previous is degraded | +0.009 | [−0.014, +0.035] | upper < 0.05 | **Supported** |

**Predictions.** All six were predicted to be supported.
- Every estimate lies within 3 points of its exploratory value. For
  example, R2 was +0.029 against +0.029, R3 +0.189 against +0.195, and S2
  +0.807 against +0.769.
- The protocol's expected intervals were close: R2 [+0.016, +0.044]
  against [+0.014, +0.043].

## What this establishes

- **On unseen programs, the account holds in Gemma-3-27B.**
  - The answer positions never need the program.
  - The post-program positions read it late. Blocking them from layer 46
    on costs under 5 points; blocking from 38 on costs about 19, so part of
    the read still happens in layers 38–45. The exploratory sweep places
    most of it between layers 31 and 45.
  - The read takes the most recent intact assignment of the asked variable.
- **The confirmed account now covers three models.** Gemma-3-12B and
  Qwen2.5-7B (Phases Eight to Twelve) and Gemma-3-27B (this phase).
- **In the Gemma family, the read ends deeper in the larger model.** The
  release layer is 46 of 62 (74% of depth) in the 27B, against 30 of 48
  (62%) in the 12B. Here that layer is confirmed only on the "survives"
  side: blocking from 46 is harmless and blocking from 38 is not.

## What this does not establish

- **A scaling law.** Two sizes in one family.
- **The 27B's read of individual statements.** Phases Nine to Eleven
  (which statements are read, and the fallback when the final one is
  degraded) were not repeated at 27B. The selection result implies the
  fallback, but the statement-level spans were not tested.
- **Scope.** One task family. The knockout limits apply as before: the
  intervention is off-distribution, and scoring is teacher-forced.

Code: [scale_confirm.py](../src/open_weight_lingua/scale_confirm.py),
launched with `scripts/run_scale_confirm.sh`.
