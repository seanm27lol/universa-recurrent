# Change a named mathematical field, then see what computation changes

**Status: runnable research pilot. No DGX intervention result is claimed.**
No weights, claim thresholds, quantization ranges, runtime APIs or package version
are changed. Version remains 0.5.2.

## A concrete example

A flow estimate has a coefficient for each known circulation cycle. Changing one
coefficient changes the candidate flow along that cycle. A short command names
which candidate, which coefficient and how much:

```text
cycle balanced_flow[0] += 0.125
```

This does not assert that an arbitrary hidden neuron means "circulation." These
variables and their basis were defined in the model. The aim is to establish a
small, editable mathematical interface with measurable downstream effects.

```text
                      SAME paused solver
                              |
              +---------------+----------------+
              |                                |
    direct float32 numerical edit      frozen 16-bit description
              |                                |
              |                          decode + typed command
              |                                |
              +---------------+----------------+
                              |
                 SAME remaining recurrent updates
                              |
         compare final outputs AND changes from the no-edit baseline
```

## What is genuinely being tested?

The parser's mapping from a name to a coefficient is known by construction. Its
correctness is a software gate, not discovery of a semantic mechanism. The useful
empirical questions are whether an effect survives the compressed description,
whether the remaining network cancels or amplifies it, and whether deliberately
addressing a different candidate gives a different effect. The latter is a
specificity control, not guaranteed to fail on every example.

The test is related to intervention-based interpretability, as developed in
[Geiger et al., Causal Abstraction (JMLR 2025)](https://www.jmlr.org/papers/v26/23-0058.html).
It does not supply a high-level causal algorithm or verify the conditions of that
theory. It is not a learned natural-language autoencoder or a model of English.

## Immediate mathematics versus downstream behavior

For candidate k, z_k = Q_k a_k: Q_k is its fixed cycle basis and a_k its coefficient
vector. Adding delta to coordinate j predicts the immediate vector change
`delta * Q_k[:, j]`. The diagnostic reports float32 addition error in that identity.

Current route weights are softmax(logits). Adding log(2) to one current logit
multiplies its immediate odds relative to an unchanged candidate by two, apart
from arithmetic rounding. Adding the same value to ALL current logits is neutral
in exact arithmetic. It is measured as a negative control, not assumed bitwise
neutral. Prior logits and the observation context are NOT edited.

Neither rule predicts the direction of the FINAL estimate after nonlinear
updates. Those effects are measured. Arbitrary edits are not expected to improve
accuracy against the original unedited data labels.

## Fixed pilot design

Use the existing five checkpoint/claim-calibration pairs and original continuation
reports. Freeze the overflow-safe named 16-bit manifest at each midpoint and
penultimate cut. Use one new matched-distribution cohort, seed 83000, n=512.
The same inputs are reused across checkpoints, cuts and commands.

Commands are fixed without looking at the new outputs:

- For every candidate and cycle axis, add plus/minus one eighth of that field's
  **old calibration range**.
- For each candidate's current evidence, add plus/minus log(2).
- Identity and a common logit shift are controls.

The 14 commands are evaluated in direct numerical and decoded-name paths. The
12 candidate-specific commands also get a wrong-candidate control with the SAME
numerical delta. Because ranges differ by checkpoint and cut, do not interpret
cross-model effect magnitudes as interventions of identical absolute size.

The direct numerical path addresses flat indices. The parser independently
addresses named fields; on raw states the two edits must agree before testing
compression. The decoder receives only state bytes, a pinned manifest and the
command. No input context, future state, labels or final answer can select the
edit or repair a lossy state. The resumed solver retains its original context.

The edited descriptor is **base payload plus command**. There is no post-edit
recompression. Command UTF-8 bytes, base record/manifest bytes and retained context
are reported, so this is not an uncounted extra reconstruction channel. It is not
a storage-performance experiment.

## Effects and failures

Both final disagreement and difference-of-effects are saved. The latter is:

```text
(decoded edited answer - decoded no-edit answer)
  - (raw edited answer - raw no-edit answer)
```

This prevents base compression error from being mistaken for the intervention
itself. Per-example states, estimates, probabilities and decisions permit an
independent report audit. Raw pause/resume must pass. A nonfinite experimental
continuation is retained as failure; it is not silently removed from evaluation.
There are no neural-transition certificates or remote-execution attestations.
The 16-bit method is approximate: the preceding generalization run included
rare claim/abstention flips. This pilot does not repair or hide them.

## Run

```bash
bash scripts/run_mathematical_edit_study.sh REPLICATION_RESULTS ORIGINAL_CONTINUATION_RESULTS cuda
```

The runner writes a fresh folder and a reports-only ZIP in Downloads. It never
modifies saved weights. Models and claim thresholds are not refitted. See
`scripts/mathematical_edits.py` for the grammar and
`tests/test_mathematical_edits.py` for the differential tests.
