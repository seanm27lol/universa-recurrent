# Phase Eight: where the answer reads the program — attention knockout, exploratory — 2026-10-06

```text
x = 7
y = 16
y = y + 1
y = y - 3
What is x? Reply with only the integer.      → 7
```

Phases Three to Seven measured what probes can read and how prompt formats
change answers. None tested where the answer comes from when it is
produced. Phase Eight ([brief](../protocols/phase_eight_knockout_brief.md))
blocks attention to the program tokens from layer k onward (attention
knockout; Geva et al. 2023) and asks when the answer stops needing the
program.

**The two models route the answer differently.**
- **Qwen2.5-7B:** the answer position reads the program itself, in layers
  21–23 of 28. Blocking just the answer position's view of the program from
  layer 21 onward drops accuracy from 0.82 to 0.22. From layer 24 onward it
  changes nothing.
- **Gemma-3-12B:** the answer positions never need the program directly.
  Blocking them from it at *every* layer costs about 1 point. Instead, the
  question and template tokens read the program between layers 24 and 30 of
  48, and the answer reads it from them. Blocking every position after the
  program from layer 24 onward drops accuracy to 0.20; from layer 30 onward
  it costs about 2 points.

These are exploratory results on reused calibration and pilot prompts.
Every release layer chosen on calibration replicated exactly on pilot.

## What ran

- **Brief.** Frozen and pushed at `9eacb79` before any Phase Eight forward.
  Both completed runs record that commit, the brief's hash `4957ba767ca1…`,
  and source hashes identical to the frozen code.
- **Data.** 1,536 calibration and pilot prompts per family, question after
  the program, from their saved token IDs.
- **Scoring.** By teacher forcing: the correct answer's digits are appended,
  and an answer is correct when every digit is the argmax.
- **Knockout.** In every layer from k onward, the additive attention mask is
  set to the dtype minimum from the query positions to the program span, in
  all heads. There are two variants and an eight-step layer grid; see the
  brief.
- **No-op control.** Hooks with an empty key set reproduced the baseline
  logits bitwise for all 3,072 prompts. The run would have stopped
  otherwise.

| Family | Run | Manifest / results / summary sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p8-knockout-gemma3-12b-20261006T194114Z-3edc8e84` | `acee41ec…` / `b565157c…` / `30d8caba…` | 4,222 s / 4,371 s |
| Qwen2.5-7B | `p8-knockout-qwen2.5-7b-20261006T184800Z-3c19e824` | `333bf744…` / `44d45835…` / `a327bc1d…` | 2,522 s / 2,625 s |

**Operational record.**
- **The GPU gate waited about 3.5 hours.** Another project's evaluation
  jobs (about 2.3 GB, 95% utilization) held the GPU from 14:42 to 18:39 UTC.
- **Gemma's first launch failed with 0 rows.** One of those jobs started
  just after the gate admitted Gemma at 18:10. Gemma's first run
  (`…20261006T182351Z-d895e16a`) then failed with a CUDA out-of-memory error
  in its start-up environment probe, before writing a manifest or loading
  the model.
- **It was relaunched once** after the GPU was idle. The relaunch had the
  GPU to itself apart from a 228 MiB blip before it started.

## Results (digit accuracy)

Baseline: Gemma 0.940 (calibration) and 0.951 (pilot); Qwen 0.819 and 0.832.

**Gemma-3-12B** (L = 48), calibration (pilot in parentheses where it
differs by more than 0.01):

| Knockout from layer | 0 | 6 | 12 | 18 | 24 | 30 | 36 | 42 |
|---|---|---|---|---|---|---|---|---|
| A: answer positions only | 0.929 | 0.930 | 0.929 | 0.926 | 0.923 | 0.933 (0.951) | 0.938 | 0.940 |
| B: every position after the program | 0.081 | 0.082 | 0.178 | 0.169 | 0.197 | **0.923** (0.943) | 0.940 | 0.941 |

**Qwen2.5-7B** (L = 28), calibration:

| Knockout from layer | 0 | 3 | 7 | 10 | 14 | 17 | 21 | 24 |
|---|---|---|---|---|---|---|---|---|
| A: answer positions only | 0.194 | 0.195 | 0.195 | 0.200 | 0.200 | 0.193 | 0.222 | **0.817** |
| B: every position after the program | 0.026 | 0.026 | 0.120 | 0.124 | 0.154 | 0.165 | 0.201 | **0.819** |

The pilot curves track calibration within about 0.05 everywhere (Qwen A
from layer 21: 0.271).

**Release layer.** This is the smallest grid layer from which knockout, and
knockout from every later grid layer, keeps accuracy within 5 points of
baseline. It was chosen on calibration and checked on pilot.

| | Variant A (answer only) | Variant B (after the program) |
|---|---|---|
| Gemma-3-12B | **0** on both: the program is never needed directly | **30** on both |
| Qwen2.5-7B | **24** on both | **24** on both |

**Predictions.**
1. The no-op control is exact. *Held.*
2. Variant B from layer 0 drops accuracy to at most 0.30. *Held:* Gemma
   0.081, Qwen 0.026.
3. Every release layer lies in the second half of the network. *Held for
   Qwen (24 of 28 in both variants) and for Gemma's variant B (30 of 48).
   Wrong for Gemma's variant A:* there is no dependence at any layer.
4. B's release layer is at or after A's. *Held* (Gemma 30 vs 0; Qwen 24 vs
   24).

**By what the program does to the asked variable** (all prompts):
- **Gemma, variant B:** blocked from layer 24, accuracy is 0.08 when the
  asked variable was last updated by arithmetic and 0.35 when by a literal.
  Blocked from layer 30, both recover (0.87 and 0.98).
- **Qwen, variant A:** with the block from layers 0–21, literal answers keep
  0.36–0.39 and arithmetic answers 0.06–0.11. Part of a literal answer can
  reach the answer position by another route; a computed value cannot.

## What this establishes

- **The answer depends on a late read of the program tokens, by a route
  that differs between the networks, and that read is necessary.**
  - In Qwen, the answer position itself attends to the program in layers
    21–23. Without that window the answer mostly fails.
  - In Gemma, the read happens earlier in relative depth (layers 24–29 of
    48), at the question and template tokens. The answer positions then use
    those positions, not the program.
- **This is an intervention, so it concerns use, not just readability.**
  - Whatever positions after the program carry into the late layers is not
    enough on its own. Both models must still look back at the program
    tokens after the question has been read: Qwen in layers 21–23, Gemma in
    layers 24–29.
  - For Gemma, the positions that look back are the question and template
    tokens, around the middle of the network.
- **It fits the earlier readouts.** Phase Three found the asked value
  readable only after the question, late in depth. The knockout shows that
  the program is *consulted* there, and where.

## What this does not establish

- **Knockout is an off-distribution intervention.** Blocking attention can
  overstate dependence (Wang et al. 2023 discuss ablation artefacts). The
  release layers show where blocking stops hurting, under this specific
  intervention.
- **A grid, not a layer-by-layer map.** Qwen's critical window is somewhere
  within layers 21–23 and Gemma's within 24–29. Single layers were not
  tested.
- **Which program tokens are read is not separated.** The blocked span
  covers every program token, including the last one. So "reads the raw
  statements" cannot be told apart from "reads something stored at the
  program tokens while they were processed". A knockout of part of the span
  could separate these: only the asked variable's lines, or only the last
  program token.
- **Necessity, not location of storage.** Surviving knockout from layer k
  means the answer no longer needs direct access to the program after k. It
  does not show where the value is held in between.
- **Exploratory.** Reused calibration and pilot prompts; one format
  (question after); one task family; digit accuracy under teacher forcing.

## Next

The brief's condition for a confirmatory test is met: each release layer is
clear on calibration and replicates exactly on pilot. A confirmatory
protocol for the reserved `validation_b` split is therefore justified.
Freeze it before opening the split. Its candidate claims:
- Gemma: blocking every position after the program from layer 30 costs less
  than 5 points; from layer 24 it costs more than 5 points; blocking the
  answer positions at every layer costs less than 5 points.
- Qwen: blocking the answer positions from layer 24 costs less than 5
  points; from layer 21 it costs more than 5 points.

Code: [answer_knockout.py](../src/open_weight_lingua/answer_knockout.py),
launched with `scripts/run_answer_knockout.sh`.
