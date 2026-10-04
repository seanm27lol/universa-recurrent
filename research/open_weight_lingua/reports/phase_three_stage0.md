# Phase Three Stage 0: the model does not keep a readable state — 2026-10-04

```text
x = 9
y = 8
y = 3
y = y + 3
x = x + 1
What is x? Reply with only the integer.
```

The answer is 10, and when the question comes y is 6. Phase Three
([brief](../protocols/phase_three_brief.md)) asked whether some position and
layer of the model holds *both* values in a form a linear probe can read. If
it did, a typed record such as `x is currently 10; y is currently 6` could be
read out and edited causally. **Gate G0 failed on the primary model,
Gemma-3-12B, so under the frozen brief Phase Three stops after Stage 0.
Stages 1 and 2 were not run.** Every Phase Two outcome stands unchanged.

## What ran

- **Brief.** Frozen and pushed at `6d51d0e`, before any Phase Three forward
  (sha256 `ebd0caf40b96…`, the same in both run manifests).
- **A fix before the first run.** At `6d51d0e` the survey stored features as
  float16. Gemma's residual stream reaches about 63,000 at block 32 alone,
  against float16's 65,504 ceiling. The first queued launch was stopped
  during its idle-GPU check, before any forward or run directory existed.
  Storage was switched to float32, with a regression test, at `f31ece3`.
  Both runs used `f31ece3`.
- **Data.** 1,536 prompts per family:
  - probes trained on 768 prompts (the first 192 calibration groups);
  - the best site chosen on a separate 256 prompts (the last 64 groups);
  - held-out accuracy reported on the 512 pilot prompts.

  The two validation splits stay unopened.
- **Probe.** At five token positions and every block, a ridge one-hot probe
  reads each variable's value (20 classes). The controls are a
  label-permutation probe, a copy-the-last-literal baseline, and a subset of
  *computed* values that appear nowhere in the prompt.

| Family | Run | Manifest sha256 | Wall clock |
|---|---|---|---:|
| Gemma-3-12B (gating) | `p3-stage0-gemma3-12b-20261004T062811Z-745ea17c` | `f76834727e7f61bb…` | 3,786 s |
| Qwen2.5-7B (reported only) | `p3-stage0-qwen2.5-7b-20261004T074740Z-840871a1` | `f364ebaff06e5413…` | 667 s |

Gemma's probe fitting took 3,329 s on a CPU shared with other sessions' work;
the GPU capture took 414 s.

## Result

Best layer per position (chosen on the selection split), held-out pilot
accuracy:

| Position | Gemma-3-12B layer: x / y (computed x / y) | Qwen2.5-7B layer: x / y (computed x / y) |
|---|---|---|
| end of program | L42: 0.238 / 0.152 (0.100 / 0.081) | L7: 0.352 / 0.211 (0.114 / 0.105) |
| " x" in "What is x" | L11: 0.453 / 0.270 (0.107 / 0.089) | L7: 0.383 / 0.250 (0.143 / 0.121) |
| "?" | L13: 0.436 / 0.254 (0.114 / 0.109) | L5: 0.359 / 0.217 (0.086 / 0.109) |
| end of user message | L42: 0.434 / 0.398 (0.307 / 0.319) | L9: 0.396 / 0.180 (0.093 / 0.085) |
| answer position | **L45: 0.500 / 0.463 (0.457 / 0.415)** | **L27: 0.436 / 0.381 (0.314 / 0.290)** |

**Baselines on the same held-out prompts:**

- copying the last literal gets x right 0.664 and y right 0.445 of the time;
- the majority value gets 0.125 and 0.109;
- the label-permutation probe at the selected sites gets 0.05–0.12.

**Gate G0.** It required both values at 0.80 or more overall and 0.50 or more
on computed values, at one site. On Gemma-3-12B the selected site (answer
position, block 45) reaches 0.463 and 0.415: **not met**. On Qwen2.5-7B it
reaches 0.381 and 0.290, also below the gate, though Qwen never gated. Qwen's
best site is its last layer, whose hidden state includes the final norm.

**Predictions: mostly wrong.** I predicted both values would be readable at
the end of the program and at the end of the user message in middle layers,
and that G0 would pass. Neither held. The prediction that the answer
position holds the asked value better than the other variable was right, as
the breakdown below shows.

## Why: the model reads the asked variable, and only it (descriptive, post hoc)

The aggregate accuracy hides a split. `scripts/state_survey_breakdown.py`
refits the same probes and divides held-out accuracy by whether the variable
was the one asked. This analysis can never change G0.

| Site | Asked variable: x / y | The other variable: x / y |
|---|---|---|
| Gemma-3-12B, answer position, L45 | **0.930 / 0.875** | 0.070 / 0.051 |
| Gemma-3-12B, end of user message, L42 | 0.801 / 0.734 | 0.066 / 0.062 |
| Gemma-3-12B, end of program, L42 | 0.238 / 0.152 | 0.238 / 0.152 (the same activation) |
| Qwen2.5-7B, answer position, L27 | **0.746 / 0.707** | 0.125 / 0.055 |
| Qwen2.5-7B, answer position, L20 (the released NLA's site) | 0.316 / 0.199 | 0.176 / 0.125 |

Across depth, the minimum of x and y at the answer position stays near
0.15–0.25 for the first half of the network. It then climbs only late: on
Gemma, 0.17 at block 24 to 0.46 at block 45; on Qwen, 0.16 at block 20 to
0.38 at block 27. At the end of the program it stays flat, near 0.15–0.25, at
every depth.

## What this establishes

- **No surveyed position holds both values in a linearly readable form,** on
  either family. Before the question, neither value is readable beyond
  about the copy level. After it, the asked variable's value appears, late in
  depth, and the other variable's does not.
- **These models appear to compute the answer on demand** rather than keep a
  running record of the program's state. That explains Phase Two from the
  inside: the released NLA describers wrote answer predictions because, at
  their capture sites, the answer is what is there.
- On Qwen, at the very block its released NLA reads (block 20), even the
  asked value is only weakly readable (0.32 / 0.20). That matches its
  descriptions rarely containing the answer (12/128 in Phase Two).

## What this does not establish

- **Nonlinear codes are untested.** A nonlinear probe might read more, though
  the clean asked/not-asked split argues against a hidden full state at
  these positions.
- **Only five positions were surveyed.** The tokens of each program line
  could carry intermediate values. Hunting through them after this result
  would be the site-search the brief forbids, so it would need a new brief.
- **One toy task,** integers 0–19, 768 training prompts per probe.
- **The breakdown and depth profiles are post hoc.** They explain the gate's
  outcome; they are not part of it.

## Closure

Per the frozen brief, a failed G0 closes Phase Three after Stage 0. Stage 1
(training-free verbalization) and Stage 2 (typed edits with a locked
validation) were not run. `validation_a` and `validation_b` remain unopened.

Stage 1 and 2 code was drafted but never frozen, committed or run. The GPU
time used was about 11 minutes of capture across both families, far inside
the 24-hour budget.

For a future phase, the natural question changes. It is no longer "where is
the state?" but "does the model keep one at all, or only compute what it is
asked?" Answering it would need a task that forces the model to maintain
state (for example several questions about the same program), or per-line
and nonlinear readouts, under a new brief.
