# Phase Twelve confirmatory: the most recent intact assignment, on fresh programs — 2026-10-12

```text
x = 2              ← v2
y = 3
x = x + 9          ← v1 = 11
y = 5
x = 7              ← v0 = 7, the answer
What is x? Reply with only the integer.
```

The exploratory Phase Twelve ([report](phase_twelve_selection.md)) found, in
both models, that the late read takes the **most recent intact** assignment
of the asked variable:
- degrading the final assignment pushes the answer back to v1;
- degrading the final and the previous assignment pushes it to v2;
- degrading only the previous assignment does not hurt.

Six claims were frozen
([protocol](../protocols/phase_twelve_confirmatory_fresh.md)) before the
test programs existed. They were tested on **1,536 newly generated groups**:
seed 208001, excluding every Phase Two program and both earlier fresh
splits.

**All six were supported.**

## What ran

- **Freeze.**
  - The protocol, the new seed, the module, its tests and the launcher
    were committed and pushed at `4f606db`. The tests used test-only seeds.
  - Both manifests record that commit, the protocol hash `bb50c7471848…`
    and source hashes identical to the frozen code.
- **Fresh programs.**
  - 1,536 groups under `fresh_2026_10c`: 3,055 attempts, with 197
    candidates rejected because they repeated an earlier program.
  - Both earlier fresh splits regenerated to their recorded inventories
    before being excluded.
  - Both families received the same programs (inventory `c56dfba4c83d…`).
  - **385 eligible prompts in 196 groups.** That is fewer than the
    predicted 455; the eligibility rate was a little lower than in the
    exploratory pool.
- **Settings.** As in exploratory Phase Twelve. The bitwise no-op control
  held on every prompt.

| Family | Run | Manifest / results sha256 | Knockouts / wall |
|---|---|---|---:|
| Gemma-3-12B | `p12-confirm-gemma3-12b-20261012T074136Z-fd9fdc5f` | `4e56d5fb…` / `c7f4d553…` | 876 s / 1,023 s |
| Qwen2.5-7B | `p12-confirm-qwen2.5-7b-20261012T080812Z-dab1fe0f` | `6043a91e…` / `244a0733…` | 501 s / 582 s |

Both runs were admitted to an idle GPU, with no foreign processes. One CPU
evaluation wrote `p12-confirm-evaluation-20261012.json` (`97c10f73…`).

## Results

Rate at which each value is said (385 prompts):

| Degraded | Gemma: v0 / v1 / v2 | Qwen: v0 / v1 / v2 |
|---|---|---|
| Nothing (baseline) | 0.977 / 0.000 / 0.000 | 0.704 / 0.119 / 0.049 |
| The final assignment | 0.236 / **0.535** / 0.034 | 0.112 / **0.719** / 0.104 |
| The final and the previous assignment | 0.374 / 0.109 / **0.439** | 0.117 / 0.021 / **0.875** |
| Only the previous assignment | **0.987** / 0.000 / 0.013 | **0.870** / 0.005 / 0.094 |

**The six confirmatory hypotheses.** These are 99.17% whole-group bootstrap
intervals (10,000 resamples; seed 805100 Gemma, 805101 Qwen), Bonferroni
across the six tests.

| ID | Measure | Mean | Interval | Threshold | Verdict |
|---|---|---:|---|---|---|
| G1 | Gemma: v1 gain when the final assignment is degraded | +0.535 | [+0.449, +0.622] | lower > 0.10 | **Supported** |
| G2 | Gemma: v2 gain when the previous is degraded too | +0.405 | [+0.318, +0.491] | lower > 0.10 | **Supported** |
| G3 | Gemma: loss when only the previous is degraded | −0.010 | [−0.047, +0.023] | upper < 0.05 | **Supported** |
| Q1 | Qwen: v1 gain | +0.600 | [+0.503, +0.689] | lower > 0.10 | **Supported** |
| Q2 | Qwen: v2 gain | +0.771 | [+0.689, +0.846] | lower > 0.10 | **Supported** |
| Q3 | Qwen: loss when only the previous is degraded | −0.166 | [−0.249, −0.089] | upper < 0.05 | **Supported** |

**Predictions.** All six were predicted to be supported. The estimates lie
within the exploratory ranges (G1 +0.54 against +0.47; Q2 +0.77 against
+0.75). Qwen again *improves* when its previous assignment is degraded, by
17 points.

## What this establishes

- **On unseen programs, the late read takes the most recent intact
  assignment of the asked variable.**
  - Degrade the final assignment and the answer becomes the previous value.
  - Degrade the final and the previous assignments and it becomes the value
    before that.
  - Degrading an earlier assignment alone leaves the answer intact; for
    Qwen it removes interference.
- **The confirmed account across Phases Eight to Twelve:**
  - After reading the question, each model looks back at the program late:
    Qwen's answer position in layers 21–23, Gemma's question tokens in
    layers 24–29.
  - It reads the asked variable's own statements, not the other variable's.
  - Among those statements it retrieves the most recent intact assignment.

  That is a recency-ordered lookup at answer time. It is consistent with
  the earlier finding that no running state was recovered by the probes.

## What this does not establish

- **How statements are marked in normal operation.** The degrading block is
  off-distribution. The rule describes the read when statements have lost
  their context, not how "current" is encoded in intact statements.
- **Scope.** Literal final statements; one task family; two models.

Code: [selection_confirm.py](../src/open_weight_lingua/selection_confirm.py),
launched with `scripts/run_selection_confirm.sh`.
