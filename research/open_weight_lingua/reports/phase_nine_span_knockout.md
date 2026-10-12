# Phase Nine: the late read targets the asked variable's own statements — exploratory — 2026-10-12

```text
x = 7              ← the asked variable's line
y = 16             ← the other variable's lines
y = y + 1
y = y - 3          ← ends with the program's last token "3"
What is x? Reply with only the integer.      → 7
```

Phase Eight confirmed that both models must look back at the program tokens
after reading the question
([report](phase_eight_confirmatory.md)).
- **Qwen2.5-7B:** the answer positions, in layers 21–23.
- **Gemma-3-12B:** the positions after the program, in layers 24–29.

Phase Eight blocked the whole program, so it could not say what is read:
the raw statements, or a summary held at the program's last token.

Phase Nine ([brief](../protocols/phase_nine_span_knockout_brief.md))
blocked only parts of the program inside each model's confirmed route.
**In both models, the late read targets the asked variable's own
statements.**
- Blocking those lines does as much damage as blocking the whole program.
- Blocking the other variable's lines, which are the same size on average,
  does nothing.
- No end-of-program summary is needed. The last token matters only when it
  is part of the asked variable's statement.

These are exploratory results on reused calibration and pilot prompts, and
pilot replicates calibration closely.

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

- **Brief.** Frozen and pushed at `6c01ef9` before any Phase Nine forward.
  Both manifests record that commit, the brief's hash `82bfbef5d1a6…`, and
  source hashes identical to the frozen code.
- **Data and scoring.** 1,536 calibration and pilot prompts per family;
  teacher-forced digit accuracy; the Phase Eight knockout hook; the bitwise
  no-op control, which held on every prompt.
- **Routes.**
  - Qwen: the answer positions, from layer 21.
  - Gemma: every post-program position, from layer 24.
- **Line assignment.** Each program token belongs to the line holding its
  first character. Checked on all 3,072 prompts: no token spans two lines,
  and no span is empty. The asked and other variables' lines average 11.5
  tokens each.

| Family | Run | Manifest / results / summary sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p9-span-gemma3-12b-20261011T230122Z-0a139ff9` | `46205318…` / `b17e8575…` / `579296d7…` | 2,008 s / 2,173 s |
| Qwen2.5-7B | `p9-span-qwen2.5-7b-20261011T234645Z-332d2d62` | `2466c13c…` / `0e47bc76…` / `0f7afa91…` | 1,904 s / 2,004 s |

Both runs were admitted to an idle GPU. Small processes from another
project (382–612 MiB) appeared during the Qwen run and were logged.

## Results

**Accuracy loss** against baseline (digit accuracy), calibration / pilot.
Baseline accuracy is 0.940 / 0.951 for Gemma and 0.819 / 0.832 for Qwen.

| Blocked span | Gemma-3-12B | Qwen2.5-7B |
|---|---|---|
| The whole program | 0.743 / 0.742 | 0.598 / 0.561 |
| Everything except the last token | 0.740 / 0.732 | 0.601 / 0.576 |
| **The asked variable's lines** (minus the last token) | **0.711 / 0.682** | **0.597 / 0.604** |
| **The other variable's lines** (minus the last token) | **−0.007 / 0.000** | **−0.012 / −0.014** |
| The asked variable's final line (minus the last token) | 0.413 / 0.373 | 0.349 / 0.322 |
| The last token only | 0.221 / 0.199 | 0.214 / 0.199 |

**The last token, by whether the program ends on the asked variable.**
Accuracy with the last token blocked, against baseline, all prompts:

| Last line assigns | Gemma | Qwen |
|---|---|---|
| The asked variable (768) | 0.50 against 0.923 | 0.35 against 0.781 |
| The other variable (768) | **0.96 against 0.965** | **0.88 against 0.866** |

Blocking the last token costs nothing when it belongs to the other
variable's statement. Its whole cost comes from being the final operand of
the asked variable's statement.

**By the asked variable's last update** (accuracy with that span blocked,
all prompts):

| Last update of the asked variable | Final line blocked: Gemma / Qwen | All asked lines blocked: Gemma / Qwen |
|---|---|---|
| A literal (`x = 7`), 706 prompts | 0.37 / 0.35 (full program blocked: 0.35 / 0.39) | 0.38 / 0.36 |
| Arithmetic (`x = x + 2`), 674 | 0.63 / 0.54 | 0.12 / 0.11 |
| A copy, 156 (148 are self-copies `x = x`) | 0.94 / 0.85 | 0.16 / 0.10 |

- **Literal values** are read from the final assignment line alone:
  blocking it is as damaging as blocking the whole program.
- **Computed values** need the earlier lines too.
- **Copies** (mostly self-copies `x = x`) lose their value when the asked
  variable's lines are blocked, as computed values do. *Corrected:* the
  268-prompt "copies the other variable" breakdown was mostly self-copies.
  On the 42 genuine cross-variable copies, blocking the other variable's
  lines gives 0.71 against 0.83 (Gemma) and 0.67 against 0.64 (Qwen). Too
  few and too noisy to say where a copied value sits.

**Predictions.**
1. The no-op control is exact. *Held.*
2. The whole-program block reproduces Phase Eight (about 0.2). *Held:* 0.197
   and 0.222.
3. Everything except the last token is within 0.10 of the whole program.
   *Held:* within 0.015 on both splits.
4. The last token alone costs less than 0.10. *Wrong:* it costs about 0.21.
   The stratified table locates the whole cost in the prompts where the last
   token is the asked statement's own operand. This is the confound the
   brief recorded; it reflects the statement, not a summary.
5. The asked variable's lines cost more than the other variable's. *Held,*
   strongly: 0.71 against −0.01 (Gemma), 0.60 against −0.01 (Qwen).

## What this establishes (exploratory)

- **At the critical late layers, each model reads the asked variable's own
  statements, and only those.** The other variable's statements, about the
  same size, are not needed: their blocking costs nothing.
- **No summary at the end of the program is required.** When the program
  ends on the other variable, the last token is irrelevant to the answer.
- **Computed values need the asked variable's earlier lines as well as its
  last one.** *Corrected:* an earlier version inferred that values copied
  from the other variable sit at the copy statement by the late layers. That
  rested on a mislabelled breakdown (see the correction note above) and is
  not established.

## What this does not establish

- **Where copied values sit.** Genuine cross-variable copies are too rare
  here (42 prompts) to say.
- **Line-level spans.** Single tokens within a line (the variable name, the
  operand) were not separated.
- **Knockout limits.** An off-distribution intervention; teacher-forced
  digit accuracy; one route per model; one prompt format; one task family.
- **Exploratory, reused data.**

## Next

The pattern is clear on calibration and replicated on pilot. Both reserved
validation blocks are spent, so a confirmatory test needs **newly generated
programs**: a new split from a new seed, excluding every program in the five
Phase Two splits. It must be frozen before the programs are generated. Its
candidate claims, for each model:
- blocking the asked variable's lines damages the answer;
- blocking the other variable's lines does not;
- when the program ends on the other variable, blocking the last token does
  not.

Code: [span_knockout.py](../src/open_weight_lingua/span_knockout.py),
launched with `scripts/run_span_knockout.sh`.
