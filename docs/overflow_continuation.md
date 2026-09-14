# Keep the full reading when the ruler is too short

**Implemented as an opt-in experiment. GPU behavior and byte tradeoffs still need
measurement. No neural weights, runtime API, calibration rules or package version change.**

A fixed-range sensor may record an extreme reading as its maximum. Giving it more
decimal places does not restore the part it discarded. The previous state codec
has this problem when a state exceeds a range chosen on calibration data.

```text
                    CURRENT state + FROZEN ranges
                                 |
                      Does every field fit?
                       /                 \
                     yes                  no
                      |                    |
                packed values       original float32 state
                       \                 /
                         reconstruct once
                                 |
                      same remaining updates
```

## Exact rule and accounting

For a state vector `x`, the named codec uses `y = x`; the rotated control uses
`y = x Q` with its saved orthogonal matrix `Q`. Range `r_j` belongs to field `j`.
The fallback is chosen iff any `abs(y_j) > r_j`. Neither the answer, task label,
future state nor a refit range enters that decision.

In-range values use the unchanged historical encoder. Otherwise **all fields of
x**, including current route logits, are stored as little-endian float32. This
avoids mixing a restored field with a clipped field or applying a lossy rotation
to a supposedly exact fallback. A raw fallback roundtrips the original state bytes.
An in-range record still has quantization error, so exact decisions are not promised.

Every record contains a 32-byte codec identity and a one-byte tag, then either the
quantized body or all raw state values. For the six-field example the bodies are
6, 9, or 12 bytes at 8, 12, or 16 bits; a raw body is 24 bytes. Report the measured
mean, fallback rate, common manifest bytes and separately retained input/context.
The identifier picks metadata; it is not a signature or payload-integrity proof.
This is the established exception/fallback pattern applied to fixed-point numerical
storage, not a new theorem. Float32 storage and explicit floating-point error
handling use [NumPy dtypes](https://numpy.org/doc/stable/reference/arrays.dtypes.html)
and [errstate](https://numpy.org/doc/stable/reference/generated/numpy.errstate.html).

## Run

```bash
bash scripts/run_overflow_continuation_study.sh \
  /path/to/replication/results /path/to/previous-state-continuation/results cuda
```

The existing previous JSON/NPZ outputs are required. The new runner **imports the
previous manifests verbatim**, checks the numeric archive hashes, regenerates the
current paused state with the same checkpoint, and requires raw restoration to agree
with the original rollout. Historical clipped states must decode exactly as saved.
It compares clipping and fallback at 8/12/16 bits in named and rotated coordinates.
There is no training, range fitting, new test cohort, threshold selection or speed
claim. Five checkpoints share the previous test cohort; these are paired diagnostics,
not independent new test examples. Results are written to a new directory.

The GPU receives exactly the original remaining update budget. Original input,
mask, encoder context and prior logits stay with the solver. The decoder receives
none of these, but their retained memory still counts. Codecs are numerical records;
no natural-language model is trained here.

## Checks and evidence

`scripts/overflow_state_codec.py` provides the framed encoder/decoder;
`scripts/benchmark_overflow_continuation.py` performs actual continuation.
Full-precision overflow rows must match all reference output fields within the
existing fixed tolerances, with identical discrete decisions. In-range rows must
match the clipped baseline's outputs. The report includes per-example arrays,
clipped/fallback and in-range subgroup errors, maximum component changes, 99th
percentile per-example maximum changes, changed claims and actual serialized bytes.
Nonfinite continuations are recorded as failures and excluded from aggregate
rankings, not silently discarded from the sample.

Tests cover positive/negative overflow, exact range boundaries, float32 bit
preservation, all-field fallbacks, malformed frames/manifests, immutable metadata,
wrong sites, source/reference identity and a real CPU checkpoint -> prior study ->
new parent/worker study. CPU fixtures are software tests, not results for the DGX.
The next scientific question is whether this removes the clipping error floor at
an acceptable byte cost. It does not establish semantic interpretability or total
GPU-memory compression.
