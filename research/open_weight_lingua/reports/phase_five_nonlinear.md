# Phase Five: a nonlinear probe finds no more state than a linear one — 2026-10-04

```text
x = 9          after this line: x = 9
y = 8          after this line: x = 9,  y = 8
y = 3          after this line: x = 9,  y = 3
y = y + 3      after this line: x = 9,  y = 6     ← does the model hold 6 here?
x = x + 1      after this line: x = 10, y = 6
What is x?                                        ← or x = 10 and y = 6 here?
```

Phases Three and Four used linear probes and found no readable program
state, neither at the end of the program nor at any statement boundary
([Phase Three](phase_three_stage0.md), [Phase Four](phase_four_trace.md)).
Both reports named the gap: a nonlinear code was untested. That gap has
mattered before. On Othello-GPT, linear probes missed the board that
nonlinear probes read (Li et al. 2023, *Emergent World Representations*), until
a linear code was found in a "mine/theirs" basis (Nanda, Lee & Wattenberg
2023).

Phase Five ([brief](../protocols/phase_five_brief.md)) re-asked both
questions with a small neural probe. The selection rules were unchanged,
and every selected site got a permuted-label control.

**Gate G5 failed on Gemma-3-12B, as predicted, and Phase Five closes.**
Qwen2.5-7B, reported only, also fails. The MLP reads about what the linear
probes read: the asked answer after the question, the literal on the current
line, and little else. Every earlier outcome stands unchanged.

## The probe

For one site (a token position and a block) take the block outputs
`h ∈ ℝ^d` (d = 3840 for Gemma, 3584 for Qwen). Standardize each dimension on
the training split. Then

```text
p(value | h) = softmax( W₂ · GELU(W₁ h + b₁) + b₂ ),   W₁: 256 × d,   W₂: 20 × 256
```

with dropout 0.1 on the hidden layer. One probe is trained per site and
variable:

- full-batch AdamW at learning rate 1e-3, for up to 300 epochs;
- weight decay 0.01 or 0.1, with the decay and stopping epoch chosen on the
  selection split;
- a fixed seed per fit.

A linear probe can only read a value whose class means are separated.
This MLP can also read a value coded as +c_v or −c_v, whose class mean is
zero. On that synthetic code, in `tests/test_state_nonlinear.py`, the linear
probe reads under 0.2 and the MLP over 0.95.

**Control.** At each selected site the same MLP, with the same decay and
epochs, is trained on shuffled labels and scored against the true held-out
labels. It catches memorization and leaks between splits. A reading counts
only if the control stays at or below 0.20.

Code: [state_nonlinear.py](../src/open_weight_lingua/state_nonlinear.py),
launched with `scripts/run_state_nonlinear.sh`.

## What ran

- **Brief.** Frozen and pushed at `1841db5`, before any Phase Five fit or
  forward. Both manifests record that commit, the brief's hash
  `5cf7c314f002…`, and source hashes identical to the frozen code.
- **Start.** Both runs started only after the GPU was idle.
- **Part A data.** Phase Three's saved activations, read-only (Gemma
  features `a94aba64d2af…`, Qwen `d8f54671d7af…`): 1,536 prompts at five
  positions and every block. That is 768 train, 256 select and 512 held out.
- **Part B data.** Phase Four's 2,548 recorded statement boundaries (1,246
  train, 432 select, 870 held out), captured again from the model. Every
  recorded position matched the tokenizer.
- **Validation.** The validation splits stay unopened.

| Family | Run | Manifest sha256 | Wall clock (capture / Part A / Part B) |
|---|---|---|---:|
| Gemma-3-12B (gating) | `p5-nonlinear-gemma3-12b-20261004T211741Z-421ba7af` | `7eb68dbf029f0775…` | 731 s (268 / 363 / 77) |
| Qwen2.5-7B (reported only) | `p5-nonlinear-qwen2.5-7b-20261004T214157Z-354536e6` | `5fb500ffd1913…` | 424 s (148 / 208 / 46) |

Peak GPU memory reserved: 25.0 GB (Gemma) and 15.6 GB (Qwen).

## Result (held out)

**Part A: the final state.** Phase Three's site rule picks the answer
position on both models, with the MLP as with the linear probe.

| Gemma-3-12B, answer position | MLP (block 46) x / y | Linear, Phase Three (block 45) x / y | Needed |
|---|---|---|---|
| Accuracy | 0.479 / 0.443 | 0.500 / 0.463 | min ≥ 0.80 |
| Computed values only | 0.371 / 0.387 | 0.457 / 0.415 | min ≥ 0.50 |
| Permuted-label control | 0.053 / 0.055 | 0.121 / 0.063 | ≤ 0.20 |
| The variable that was asked | 0.871 / 0.824 | 0.930 / 0.875 | descriptive |
| The variable that was not asked | 0.086 / 0.062 | 0.070 / 0.051 | descriptive |

**G5a: not met** (min 0.443, computed 0.371). Qwen2.5-7B, at block 24,
reads 0.445 / 0.322, computed 0.307 / 0.181, control at most 0.070.
Its linear probe read 0.436 / 0.381 at block 27. Split by the variable
asked, Qwen reads 0.770 / 0.562 asked and 0.121 / 0.082 not asked.

At the end of the program, before any question, the best MLP layer reads
0.281 / 0.234 on Gemma and 0.230 / 0.254 on Qwen. Computed values read
0.057 / 0.129 and 0.043 / 0.153.

**Part B: the running state.** Phase Four's layer rule picks the same layer
for R1 on both models as it did for the linear probe.

| Reading, at its selected layer | Gemma MLP x / y | Gemma linear x / y | Qwen MLP x / y | Qwen linear x / y | Needed |
|---|---|---|---|---|---|
| R1: carried values (L7 / L7; L3 / L3) | 0.527 / 0.319 | 0.672 / 0.383 | 0.576 / 0.347 | 0.707 / 0.375 | 0.80 |
| R1: carried *computed* values | **0.019 / 0.077** | 0.000 / 0.077 | **0.019 / 0.092** | 0.000 / 0.108 | 0.50 |
| R2: arithmetic results (Gemma L32 / L18; Qwen L17 / L26) | **0.184 / 0.219** | 0.176 / 0.206 | **0.221 / 0.237** | 0.184 / 0.184 | 0.80 |
| Literal updates, at the R2 layer | 0.853 / 0.953 | 0.794 / 0.906 | 0.838 / 0.888 | 0.706 / 0.909 | — |
| Permuted-label control (R1 and R2 layers) | 0.052–0.063 | — | 0.049–0.064 | — | ≤ 0.20 |

**G5b = R1 and R2: not met on either model. G5 = G5a or G5b: not met on
Gemma-3-12B. Phase Five closes.**

**Across depth** (descriptive). The best `min(x, y)` at any layer, held out:

| Category | Gemma MLP | Gemma linear | Qwen MLP | Qwen linear |
|---|---:|---:|---:|---:|
| Carried | 0.335 | 0.383 | 0.347 | 0.375 |
| Carried computed | 0.092 | 0.056 | 0.056 | 0.037 |
| Arithmetic | 0.250 | 0.221 | 0.241 | 0.206 |

The copy-the-literal baseline reads carried values at 0.904 / 0.661 and
arithmetic results at 0.044 / 0.018.

**Predictions.** The brief recorded four:

- *G5 fails.* Right.
- *At the answer position the MLP reads the asked variable well and the
  other near chance.* Right: 0.87 / 0.82 against 0.09 / 0.06 on Gemma.
- *At the end of the program, computed values stay low.* Right: at most
  0.13.
- *Arithmetic results improve on the linear 0.2 but stay well below 0.80.*
  Right in direction, but the size is negligible. All four numbers sit at
  or above the linear ones, by 1–5 points (0.18–0.24 against 0.18–0.21),
  next to the 0.80 needed. No test was frozen for whether a gain that small
  differs from noise, so none is claimed.

## What this establishes

- **The Othello-GPT reversal did not happen here.** A probe that can read a
  sign-flipped code reads the same things the linear probes read:
  - the literal written on the current line;
  - after the question, the asked variable's value;
  - nothing much else.

  Part A picks the same position as before, at a neighbouring block. R1
  picks the same layer. Only R2's layer moves, and its accuracy barely does.
- **The nonlinear probe is no better, and sometimes worse.**
  - On carried values the MLP sits 3–15 points *below* the linear probe.
  - On Gemma's asked answer it sits 5–6 points below.
  - This fits a more flexible probe trained on 768–1,246 rows: it gains
    nothing from capacity and loses some to variance.
- **The conclusion of Phases Three and Four now covers small nonlinear
  readouts.** On this task, at every site surveyed, neither Gemma-3-12B nor
  Qwen2.5-7B holds a readable running state of the program. The answer
  appears to be computed when the question arrives.

## What this does not establish

- **One small probe family.** One hidden layer of 256 units, trained on
  hundreds to about a thousand examples. A much larger probe, or far more
  data, could read a code this one misses. "Not readable by this probe" is
  weaker than "not represented".
- **Pooling across lines.** Part B's MLP sees all boundaries together.
  It could in principle separate line-specific codes, but it was given no
  line index.
- **Readability is not use, in either direction.** The model may use
  information no probe of this size decodes. A probe could also decode
  information the model ignores. Only causal tests settle use.
- **One task family:** integers 0–19, short programs, two models.

## Closure

Per the brief, a failed G5 closes Phase Five. The validation splits remain
unopened. GPU time was about 20 minutes in all, against a 4-hour budget.
Each run keeps its re-captured boundary activations locally (`raw/`,
1.8 GB for Gemma and 1.0 GB for Qwen); they are not committed.

Phases Three, Four and Five now give one answer from three angles: linear
probes at the end, linear probes per line, and nonlinear probes at both.
Probing harder on this task is unlikely to change it. A further phase
should change the task instead, so that the model must keep state, for
example several questions about one program.
