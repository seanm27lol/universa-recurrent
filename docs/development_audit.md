# Development audit: what the first neural run exposed

The first learned implementation ran successfully, but code review found several
places where an attractive metric could have been misread. This document records
the corrections.

## Findings and fixes

| Finding | Why it mattered | v0.3 response |
|---|---|---|
| Halted samples were still evaluated in a dense batch | “Mean steps” was not actual skipped GPU work | Added `dense` and active-sample `compact` execution; report both logical and executed update counts |
| Event logging synchronized the GPU on every step | Instrumentation could dominate a tiny model | Bulk evaluation disables event construction; one-example Lingua runs enable it explicitly |
| Device-value validation and host telemetry can also synchronize CUDA | Safety checks or counters could distort the benchmark being reported | Generated data are validated before the hot loop; optional deep value checks are explicit, device scalars are materialized after timing, and dense execution avoids per-round all-halted synchronization |
| The transparent baseline repeated deep device-value checks inside every timed batch | Synchronizations could make the baseline look slower for instrumentation reasons | Generated benchmark data are checked at construction; timed baseline calls keep shape/device checks but skip redundant value scans |
| Residual RMS included masked zeros | Fewer observations could look artificially easier | New checkpoints divide by the number of observed coordinates; old checkpoints retain legacy semantics |
| Halting was supervised only on the true structure | Misrouted examples entered an untrained halt path | Halting targets now cover every candidate trajectory |
| Absolute-quality halting did not target the compute tradeoff directly | Wrong-route trajectories could continue after their best point | Neural v1 uses a synthetic future-regret teacher; the strategy is stored in checkpoint metadata |
| Evaluation compared adaptive thresholds only with depth 8 | A shallower fixed model could be stronger | Added fixed depths 1, 2, 4, and 8 |
| No transparent or generator-aware reference | Router performance lacked context | Added fit-each-subspace and Gaussian Bayes references |
| Overall MSE hid the cost of routing mistakes | Good correct-route estimates could mask failure concentration | Added route-correct and route-wrong MSE |
| Neural Lingua had no independent checker | A printed residual could be altered without detection | Added a standalone structural/summarization checker and `neural-verify` |
| A self-contained boundary fingerprint was not an external trust anchor | Replacing both a boundary and its fingerprint could create a different self-consistent claim | Added optional verification against the exact local checkpoint, names, structure library, and configuration |
| Checkpoint-bound verification received CUDA-resident boundaries in the GPU demo path | The NumPy checker could fail even though inference succeeded | Boundary libraries are moved through a device-safe conversion before independent checking; a standalone checker remains NumPy-based |
| Checkpoint loading used unrestricted pickle | Serialized model files are a code-execution boundary | The loader uses the restricted weights-only path, requires PyTorch 2.10 or newer, keeps a narrow legacy allowlist, and still instructs readers to load only trusted checkpoints |
| Checkpoints lacked full run metadata and could be silently overwritten | Reproduction and provenance were weak | Added data/training metadata, SHA-256, deterministic setting, and overwrite refusal |
| Checkpoint tensors were structurally loadable even if a non-basis weight contained NaN or infinity | Corrupted weights could survive until inference | All floating checkpoint tensors are now checked for finiteness before save and after restricted loading |
| Per-batch metric conversion forced several host synchronizations during GPU training | The reported training loop paid avoidable telemetry overhead | Training accumulates detached metric tensors on device and materializes one vector per epoch; one fail-closed finite check remains per batch |
| Neural tests could silently skip in ordinary CI | Core CI did not prove the optional path worked | Added a dedicated CPU neural job |

## Resulting interpretation

```text
fewer logical steps
        │
        ├─ may reduce active update examples
        │
        └─ may still be slower in wall-clock time
```

The project therefore reports three separate quantities:

1. **logical updates:** how many state updates each sample receives;
2. **update examples:** how many examples actually enter the update network;
3. **wall-clock latency:** what the hardware measured.

No one quantity substitutes for the others.

## Compatibility

The loader recognizes v0.2 checkpoints that lack the new residual-normalization
field and preserves their original ambient-dimension normalization. They can be
re-evaluated, but a new v1 checkpoint is required to test the corrected training
recipe.
