# Phase Thirteen: the account holds at Gemma-3-27B, with a later read — exploratory — 2026-10-12

```text
x = 2
y = 3
x = x + 9
y = 5
x = 7
What is x? Reply with only the integer.      → 7
```

Phases Eight to Twelve built and confirmed an account of how Gemma-3-12B and
Qwen2.5-7B answer such questions.
- After the question, the model looks back at the program late.
- It reads the asked variable's own statements.
- It takes the most recent intact assignment.
- In Gemma-3-12B, the question and template tokens do the reading, around
  layers 24–29 of 48; the answer positions never need the program
  directly.

Phase Thirteen ([brief](../protocols/phase_thirteen_scale_brief.md)) ran the
unchanged Phase Eight and Phase Twelve modules on **Gemma-3-27B**. It has the
same family and identical tokenizer files, and 62 layers.

**The account holds.**
- The routing is the same as in the 12B: the answer positions never need the
  program, and the post-program tokens read it.
- The read happens later in relative depth: release layer 46 of 62 (74%),
  against 30 of 48 (62%).
- The selection rule holds even more sharply.
- All three predictions held.

These are exploratory results on opened programs.

## What ran

- **Brief.** Frozen and pushed at `ed4b4b1` before any Phase Thirteen
  forward. Both manifests record that commit, this brief
  (`phase_thirteen_scale_brief.md`, `2f6b95dbf169…`, through the new
  `--protocol` option) and source hashes identical to the frozen code.
- **Model.** Gemma-3-27B under its lock (`ca078e52`, target revision
  `7a5a3053…`). Peak GPU memory reserved: 55.4 GB.
- **Parts.**
  - Part A, routing: the 27B calibration and pilot prompts (1,536).
  - Part B, selection: the Phase Twelve pool (329 eligible prompts: 101
    reused and 228 from the two opened fresh splits, regenerated and
    verified).
- **Control.** The bitwise no-op control held on every prompt.

| Part | Run | Manifest / results / summary sha256 | Knockouts / wall |
|---|---|---|---:|
| A: routing | `p8-knockout-gemma3-27b-20261012T084251Z-ca324996` | `1d3ab8d6…` / `a83eaed0…` / `1744cf58…` | 9,364 s / 9,660 s |
| B: selection | `p12-selection-gemma3-27b-20261012T113335Z-317688f7` | `e21dbeb4…` / `338a88ea…` / `2b67a036…` | 1,615 s / 1,925 s |

Both parts were admitted to an idle GPU. One 362 MiB process from another
project appeared once and was logged.

## Results

**Part A: routing.** Teacher-forced digit accuracy. Baseline is 0.960
(calibration) and 0.965 (pilot).

| Knockout from layer | 0 | 7 | 15 | 23 | 31 | 38 | 46 | 54 |
|---|---|---|---|---|---|---|---|---|
| A: answer positions only (calibration) | 0.946 | 0.945 | 0.945 | 0.948 | 0.957 | 0.954 | 0.954 | 0.959 |
| B: every post-program position (calibration) | 0.096 | 0.098 | 0.172 | 0.160 | 0.366 | 0.769 | **0.932** | 0.960 |
| B (pilot) | 0.094 | 0.111 | 0.184 | 0.158 | 0.340 | 0.764 | **0.936** | 0.957 |

Release layers, calibration / pilot: variant A **0 / 0**, variant B
**46 / 46**.

| Release layer of variant B (every post-program position) | Layer | Share of depth |
|---|---|---:|
| Gemma-3-12B (Phase Eight) | 30 of 48 | 62% |
| **Gemma-3-27B** | **46 of 62** | **74%** |
| Qwen2.5-7B, variant A (Phase Eight) | 24 of 28 | 86% |

**Part B: selection.** Rate at which each value is said (329 prompts):

| Degraded | v0 / v1 / v2 |
|---|---|
| Nothing | 0.991 / 0.006 / 0.003 |
| The final assignment | 0.198 / **0.781** / 0.024 |
| The final and the previous | 0.097 / 0.064 / **0.793** |
| Only the previous | **1.000** / 0.000 / 0.000 |

Under the final-only block, v1 makes up 90% of wrong answers. Under the
double block, v2 makes up 80%. The reused and fresh subsets agree.

**Predictions.**
1. The no-op control is exact. *Held.*
2. Routing as in the 12B: variant A release layer 0, and variant B's in the
   second half but not the last grid layer. *Held:* 0 and 46.
3. The selection rule holds: final costs more than 25 points with v1 most
   common; both make v2 exceed v1; previous-only costs under 10 points.
   *Held:* −79 points with v1 dominant; v2 0.793 against v1 0.064; no
   cost.

## What this establishes (exploratory)

- **The account transfers to the larger sibling model.**
  - Gemma-3-27B routes the late read through its post-program tokens, as
    the 12B does, never through the answer positions directly.
  - It reads the asked variable's most recent intact assignment.
- **The read sits deeper in the 27B.** It ends around layers 38–45 of 62
  (74% of depth), against 24–29 of 48 (62%) in the 12B. The transition is
  also more gradual: 0.37 at layer 31, 0.77 at 38, 0.93 at 46.

## What this does not establish

- **Confirmation.** Opened programs only. A fourth fresh split would be
  needed.
- **A scaling law.** Two sizes in one family, one task.
- **Knockout limits** apply as before.

Code: the unchanged
[answer_knockout.py](../src/open_weight_lingua/answer_knockout.py) and
[selection_knockout.py](../src/open_weight_lingua/selection_knockout.py),
run with `--protocol` set to this phase's brief.
