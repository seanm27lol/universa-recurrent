# Phase Nineteen: the question-first two-stage account holds at Gemma-3-27B, at the same relative depth — exploratory — 2026-10-13

```text
What is x? Reply with only the integer.
x = 7
y = 16
y = y + 1
y = y - 3          → 7  (y ends at 14)
```

**What Phases Sixteen to Eighteen found,** in Gemma-3-12B and Qwen2.5-7B.
With the question first, the model works in two stages:
- in the middle layers, the program tokens read the question and mark both
  variables' statements;
- in the late layers, the post-program positions read the marked program.

Without the marks, Gemma-3-12B takes the most recent assignment of either
variable.

Phase Nineteen
([brief](../protocols/phase_nineteen_scale_question_first_brief.md)) ran the
Phase Eighteen module, unchanged apart from a `--protocol` option, on
**Gemma-3-27B** (62 layers).

**All five predictions held.**
- **The 27B relies on the marks even more.**
  - On programs ending on the other variable, it answers 0.958 correctly
    with the question first. The 12B answered 0.802 of the same prompts.
  - Cut the program off from the question, and accuracy falls to 0.134,
    with 79.9% of answers becoming the other variable's value. That is a
    rise of 75 points, against the 12B's 58 (0.802 → 0.156).
- **The marking sits at the same relative depth as in the 12B.**
  - It is finished by layer 31 of 62 (50%), against 24 of 48 (50%).
  - Almost all of it happens in layers 15–22 (24–37%).

  The late read, in contrast, sits deeper in the 27B (Phase Thirteen: 74%
  against 62%).

These are exploratory results on opened programs.

## What ran

- **Freeze.** The brief, the `--protocol` option and its test were
  committed and pushed at `3a37624` before any forward. The manifest
  records that commit and this brief (`fa75a8b92e4a…`). All 134 source
  hashes equal the files at `3a37624`.
- **Model.** Gemma-3-27B under its lock (`ca078e52`). Peak reserved GPU
  memory: 55.4 GB.
- **Prompts.** The 27B calibration and pilot runs' prompts in the
  question-first order, where the mark decides the answer: 718 (482
  calibration, 236 pilot).
- **Conditions.** As Phase Eighteen, on the 62-layer grid 0, 7, 15, 23, 31,
  38, 46, 54. The bitwise no-op control held on every prompt.

| Run | Manifest / results / summary sha256 | Knockouts / wall |
|---|---|---:|
| `p18-layers-gemma3-27b-20261013T113706Z-c1198519` | `f4275140…` / `b0fa9468…` / `6899dda0…` | 8,220 s / 8,514 s |

The run was admitted to an idle GPU. The only other GPU process was the
rf-moe dashboard (451 MiB). (The run directory keeps the module's `p18-`
prefix; its manifest records this phase's brief.)

## Results

**Rise in other-value answers over baseline**, all 718 prompts. Baseline:
accuracy 0.958, other value 0.049.

| k | 7 | 15 | 23 | 31 | 38 | 46 | 54 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Blocked from layer k on | +0.741 | +0.682 | +0.082 | **−0.001** | +0.001 | 0.000 | 0.000 |
| Blocked below layer k | **+0.004** | +0.046 | +0.674 | +0.747 | +0.751 | +0.749 | +0.751 |

Blocked in every layer: +0.751 (accuracy 0.134; 0.143 on calibration and
0.114 on pilot).

**The marking window** (tolerance +0.05; calibration / pilot):
- **End: 31 / 31.**
- **Onset: 15 / 7.** On pilot, blocking below 15 reaches +0.055, just over
  the tolerance.

**Three models compared.** The 12B and Qwen values are from Phases
Sixteen to Eighteen, on these same prompts. The window edges are
confirmed except "most done by", whose remainder was under 5 points.

| | Gemma-3-12B (48 layers) | **Gemma-3-27B (62)** | Qwen2.5-7B (28) |
|---|---|---|---|
| Rise in other-value answers with the program blind to the question | +0.585 | **+0.751** | +0.169 |
| Marking finished by layer | 24 (50%) | **31 (50%)** | 17 (61%) |
| Most of the marking done by | 18 (38%) | **23 (37%)** | 14 (50%) |
| Late read, question after (Phase Eight / Thirteen) | 24–29 (50–60%) | 38–45 (61–73%) | 21–23 (75–82%) |

**Predictions.**
1. The no-op control is exact. *Held.*
2. **The program carries the variable at 27B.** `program_blind` raises
   other-value answers by at least 30 points. *Held:* +75.1.
3. **Marking precedes the read.** The window ends at or before layer 38.
   *Held:* 31, on both splits.
4. **Deeper in the larger model:** the window ends at or after layer 31.
   *Held,* exactly at the bound. But the relative depth is *not* deeper:
   50%, as in the 12B. The prediction was framed in layers and was met; the
   relative-depth reading behind it was not.
5. **The first layers do not mark.** Blocking layers 0–6 raises
   other-value answers by less than 5 points. *Held:* +0.004.

## What this establishes (exploratory)

- **The question-first two-stage account holds at Gemma-3-27B.**
  - Middle-layer marking: the program reads the question in about layers
    15–30, most of it in 15–22.
  - The late read follows. Without the marks, the read takes the other
    variable's value: a recency fallback, stronger than in the 12B.
- **The marking scales with depth and ends at half of it**, in both Gemma
  sizes. The late read sits relatively deeper in the larger model. So the
  gap between the two stages widens with scale: in the 27B, marking ends at
  layer 31 and the read runs to about 45.
- **The larger model is much better with the question first** (0.958
  against 0.802 on these prompts). It relies on the marks more completely:
  blocking them leaves 13% accuracy, against 16% for the 12B.

## What this does not establish

- **Confirmation.** Opened programs only.
- **A scaling law.** Two sizes in one family.
- **Single-layer resolution.** The 27B grid steps are 7–8 layers.
- **Scope.** Off-distribution blocks; one wording.

Code: the unchanged
[marking_layers_knockout.py](../src/open_weight_lingua/marking_layers_knockout.py),
run with `--protocol` set to this phase's brief.
