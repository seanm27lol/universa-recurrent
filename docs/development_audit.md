# Development audit: what the first neural run exposed

The first learned implementation ran successfully, but code review found several
places where an attractive metric could have been misread. This document records
the corrections.

## Findings and fixes

| Finding | Why it mattered | Current response |
|---|---|---|
| Halted samples were still evaluated in a dense batch | “Mean steps” was not actual skipped GPU work | Added `dense` and active-sample `compact` execution; report both logical and executed update counts |
| Event logging synchronized the GPU on every step | Instrumentation could dominate a tiny model | Bulk evaluation disables event construction; one-example Lingua runs enable it explicitly |
| Device-value validation and host telemetry can synchronize CUDA | Safety checks or counters could distort benchmarks | Generated data are validated outside the hot loop; deep value checks are explicit; benchmark telemetry is materialized after synchronization |
| The transparent baseline repeated device-value checks inside each timed batch | Instrumentation could make it look slower | Timed generated-data calls retain shape/device checks but skip redundant value scans |
| Residual RMS included masked zeros | Fewer observations could look artificially easier | New checkpoints divide by observed coordinates; v0.2 checkpoints retain ambient-width semantics |
| Values in masked coordinates still entered the encoder | A caller could leak supposedly unobserved information | The model multiplies observations by the mask again at its boundary; a regression test changes masked values adversarially |
| Halting was supervised only on the true structure | Misrouted examples entered an untrained halt path | Halting targets cover every candidate trajectory |
| Absolute-quality halting did not target the compute tradeoff directly | Wrong-route trajectories could continue after their best point | Neural v1 uses a synthetic future-regret teacher; the strategy is stored in checkpoint metadata |
| Evaluation compared adaptive thresholds only with depth 8 | A shallower fixed model could be stronger | Added fixed depths 1, 2, 4, and the configured maximum |
| No transparent or generator-aware reference | Router performance lacked context | Added fit-each-subspace and Gaussian Bayes references |
| Overall MSE hid routing mistakes | Good correct-route estimates could mask failure concentration | Added route-correct and route-wrong MSE |
| Neural Lingua had no independent checker | A printed residual could be altered without detection | Added a standalone structural/event checker and `neural-verify` |
| A self-contained boundary fingerprint was not an external trust anchor | Replacing both a boundary and fingerprint creates another self-consistent claim | Optional verification binds the record to exact local checkpoint bytes, names, structures, and configuration |
| Checkpoint-bound verification could receive CUDA boundaries | NumPy conversion could fail despite successful inference | All independent checking uses device-safe CPU conversion |
| Checkpoint loading used unrestricted pickle | Serialized model files are a code-execution boundary | Uses the restricted weights-only loader, PyTorch 2.10+, and only a narrow legacy `TorchVersion` allowlist |
| Checkpoints could be overwritten or contain nonfinite tensors | Provenance and failure behavior were weak | Save refuses overwrite, records a SHA-256 sidecar, and validates all floating tensors plus the embedded basis |
| The benchmark's `n` option timed only the first batch | Reported workload could differ from requested workload | Every repetition now processes all `n` examples in declared batches |
| Neural tests could silently disappear from ordinary validation | Core tests could pass while the public neural API was absent | Dedicated neural CI names the required test files and performs train/demo/verify smoke commands |
| The first v0.3 Git tree omitted the neural runtime and tests | Documentation and package version overstated what was published | v0.3.1 restores the files and records the failed publication state rather than hiding it |

## Resulting interpretation

```text
fewer logical steps
        │
        ├─ may reduce active update examples
        │
        └─ may still be slower in wall-clock time
```

The project reports three separate quantities:

1. **logical updates:** how many state updates each sample receives;
2. **update examples:** how many examples actually enter the update network;
3. **wall-clock latency:** what the hardware measured.

No one quantity substitutes for the others.

## Compatibility

The loader recognizes v0.2 checkpoints that lack the newer configuration fields.
It preserves both their ambient-dimension residual normalization and their
pre-update halt timing. A v0.2 checkpoint can therefore be re-evaluated, but a new
v1 checkpoint is required to test the corrected training recipe.
