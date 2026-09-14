# Will the same ruler work on tomorrow's readings?

**Status: runnable frozen-codec validation experiment; no new DGX result yet.**
This follows [overflow-safe state continuation](overflow_continuation.md).
The model, bit packing, calibration ranges, claims and runtime version 0.5.2
are unchanged. Only this opt-in experiment and its tests are new.

## Why this is the next question

A measurement saved inside a fixed range can use a compact representation.
Outside that range, the existing codec stores the entire original float32 state.
That fixes clipping, but finer in-range rounding can still alter a decision.
An experiment on the same examples used to notice clipping cannot establish
what happens on new inputs. We therefore freeze the implementation before
opening two new input-seed blocks.

The supplied overflow run passed the independently recomputed JSON/NPZ audit:
360 case rows, 72 summaries, and 180 codec/cut/checkpoint cases. All 1,050
fallback comparisons restored original dynamic values, saved estimates,
probabilities and claim indices exactly; in-range decodes and outputs matched
the previous clipped run. These counts reuse 1,024 problems across conditions.
They are not 1,050 independent held-out problems or independent neural replay.

Across named-codec cuts, sixteen bits changed zero claims in 30,720
checkpoint/cut/input comparisons; twelve bits changed 8 and eight bits 228.
The worst final component difference was still nonzero: approximately 0.000302
at sixteen bits. The checkpoints were not supplied with the reports; this audit
checks numeric artifacts, not checkpoint-file binding or remote execution.

## The experiment

```text
Frozen checkpoints + frozen ranges + frozen claim policies
                         |
       two new input-seed blocks, 1,024 problems each
                         |
           +-------------+--------------+
           |             |              |
       matched        3x noise      fewer observations
           |             |              |
           +-------------+--------------+
                         |
       raw / zero / other-input / safe 8,12,16-bit state
                         |
          SAME remaining recurrent updates
                         |
       final differences, decisions, failures and bytes
```

The primary precision is sixteen bits, selected from the earlier exploratory
run. Twelve and eight bits remain fixed lower-budget comparisons, not options
chosen after looking at the new test set. No pass/fail error threshold or claim
of uniform decision preservation is predeclared. All changes are reported.

Default data seeds: **71000 and 72000**. They must not match documented earlier
or checkpoint-reserved seeds. Inputs are also checked against the previous
continuation test inputs. Conditions within a seed reuse the same latent truth
and labels, so there are **2,048 underlying problems**, not 6,144 independent
problems. The three observation conditions and five trained checkpoints are
paired views of these problems. Repeated observations between paired conditions
are counted rather than secretly removed.

| Condition | Explicit setting |
|---|---|
| matched | Checkpoint training noise and observation probability |
| noise_x3 | Three times the training noise standard deviation |
| sparse_040 | Observation probability 0.40, original noise level |

The existing generator repairs masks to expose at least two coordinates. Actual
mean observed counts are reported. Changing mask repairs can advance its noise
random-number stream, so the sparse condition does not isolate missingness with
every noise sample held identical. These are observation stress tests within
the same two candidate structures, not arbitrary out-of-distribution tasks.

## What stays fixed mathematically

At cut `t`, the dynamic vector is all candidate coordinates and route logits.
The encoded context, observations, masks and prior logits remain with the
solver. The decoder receives only numerical bytes and the original manifest.
Each representation replaces the dynamic vector once, then executes exactly
`T-t` original updates, where `T` is the model's trained depth.

The raw state must reproduce original rollout outputs within the existing
numerical tolerances and with identical discrete decisions. Each raw fallback
must reproduce its corresponding full-precision output fields. The experiment
never uses test labels or future states to choose whether to fall back.

Preservation error is squared difference from the ORIGINAL MODEL output.
Task error is squared difference from the SYNTHETIC TRUTH. A codec can preserve
a wrong model prediction; those metrics and the wrong-claim counts stay separate.
Zero/other-input controls test how much recovery comes from retained context.

## Recorded evidence and limits

Worker JSON and pickle-free NPZ files retain input/label arrays, original dynamic
vectors, codec manifests, restored states, final predictions and decisions.
Reports include mean and maximum output changes, per-input 99th percentiles,
claim changes, fallback counts, actual frame/metadata bytes, and context bytes.
The two seed blocks remain separate in the summary. Any nonfinite continuation
is an explicit failed case, not a removed observation or a good aggregate score.

This does not time latency, measure GPU memory, train language descriptions,
prove causal abstraction, or establish an error guarantee for all inputs.
These are named numerical coordinates with known definitions. Interpretable
prose and faithful semantic interventions remain separate future experiments.

This uses ordinary held-out evaluation, fixed-point quantization and empirical
state replacement. The implementation reuses the tested pause/resume and codec
functions rather than replacing the solver. For the distinction between a state
intervention and a semantic causal abstraction, see the primary
[causal-abstraction framework](https://www.jmlr.org/papers/v26/23-0058.html).

## Run

```bash
bash scripts/run_codec_generalization_study.sh \
  /path/to/replication/results /path/to/original-state-continuation/results cuda
```

The second folder is the ORIGINAL continuation study, not the overflow results.
It supplies the immutable codec manifests and the earlier input arrays. The
runner does not overwrite either folder or any checkpoint. A new reports-only
ZIP is produced in Downloads; no weights are packaged.

Tests cover exact raw restoration, in-range/fallback gates, frozen fitting,
manifest identity, split exclusion, paired truth, distinct inputs, explicit
failure handling, unchanged budgets, overwrite refusal and a real CPU
parent/child CLI roundtrip. CPU smoke tests are not this DGX experiment's result.
