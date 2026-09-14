# Smaller stacks of temporary objects, unchanged numerical receipts

**Status: opt-in experiment. No new DGX speedup has been measured.**
This adds a generator and a diagnostic runner; the model, checker, calibration,
individual JSON schema, and package version 0.5.2 are unchanged.

## Why this is the next experiment

Imagine translating a large stack of laboratory worksheets. Working on a small
stack at a time may avoid having every translated worksheet open simultaneously.
It may also add handling overhead. We measure both possibilities rather than
assuming a smaller stack is faster.

In the supplied September 14 batched-pipeline run, at 256 inputs:

| Model / receipt | Prepared scalar (ms) | Batch check only (ms) | Bulk generation + batch check (ms) |
|---|---:|---:|---:|
| One-pass / endpoint | 25.131 | 15.181 | 12.250 |
| Four-step / endpoint | 26.171 | 16.192 | 13.272 |
| Shared eight-step / endpoint | 27.131 | 17.271 | 14.149 |
| Four-step / trajectory | 75.684 | 39.210 | 32.786 |
| Shared eight-step / trajectory | 91.224 | 52.516 | 74.400 |

These are medians of five checkpoint-level medians, each based on ten repeats.
All modes are warmed; initial preparation and model loading are excluded.
The shared full-trajectory case is a reversal: bulk generation is slower than
batch checking alone. Several requests show large intermittent pauses during
host generation or parsing/checking. The report did not record garbage
collection, so its cause is NOT established.

Independent report arithmetic audit: 100 comparison rows, 3,000 timing samples,
6,400 saved receipts; numerical receipt properties passed, and all 6,400 saved
receipts matched the prior complete-pipeline report. Checkpoint files were not
provided for independent neural replay or byte-binding. Only 256 distinct input
problems underlie these checkpoint/model/receipt combinations.
Source archive SHA-256:
`35b71b5165309b55b063541664d0436c12b488a40378e75ebc68c3d7b7789824`.

## The proposed implementation

```text
ONE unchanged GPU inference batch
                 |
      output/trajectory tensors copied to CPU
                 |
   +-------------+-----------------+------------------+
   |             |                 |                  |
 original      whole batch      16 records          64 records
 per-record    Python lists     at a time           at a time
   |             |                 |                  |
   +-------------+-----------------+------------------+
                 |
       IDENTICAL individual JSON bytes
                 |
       SAME prepared batch checker
```

The chunk size bounds temporary conversion to Python lists. It does NOT split
neural inference, shorten a trajectory, reduce precision, or discard records.
For batch size N and conversion chunk C, up to min(C,N) records are materialized
as Python values per call. Full CPU tensors and N serialized byte payloads still
exist; this is not a constant-total-memory claim.

`scripts/benchmark_bounded_receipts.py::generate_bounded` copies each required
tensor once and calls the existing bulk generator on CPU slices. Exact byte
comparison with both existing generators precedes every reported result. All
receipts still receive the unchanged batch check and scalar fallback behavior.

## Measure rather than suppress pauses

The primary pass has no GC observer and keeps the interpreter's garbage
collection enabled. It reports raw times, median and p90, retaining slow samples.
A separate pass uses Python's official
[gc.callbacks interface](https://docs.python.org/3.11/library/gc.html#gc.callbacks)
to record collection durations and which request stage was active. It does not
disable collection, freeze objects, tune thresholds, force collection, or subtract
pauses. Diagnostic timings never enter primary speed summaries. Observer
allocations may themselves perturb collection: associations are diagnostics, not
proof of a production speed improvement.

Keep only the current case's canonical byte receipts for differential checks;
do not accumulate parsed trajectory notebooks from earlier cases in the worker.
This differs from the earlier benchmark's retained audit heap. Compare modes
WITHIN the new run, not by treating old and new absolute timings as identical
execution conditions. Script/source/checkpoint hashes and GC settings are saved.

## Run on existing checkpoints

```bash
bash scripts/run_bounded_receipt_study.sh /absolute/path/to/replication/results cuda
```

Defaults: five available checkpoint files; shared-eight-step and dedicated-four-
step models; endpoint and trajectory receipts; 64 and 256 inputs; 30 ordinary
repeats plus 10 separately instrumented repeats; randomized cases and mode order;
fresh process per checkpoint. The same seed-46000 cohort is intentionally reused
for performance comparisons, not as fresh confirmatory accuracy evidence.

Input transfer, inference, host conversion, JSON serialization, parsing, per-record
checks and synchronization are timed. Loading, preparation, report writing,
network transport, queueing and post-timing differential comparisons are excluded.
All original reference files are checked for changes before/after the worker.
Existing corruption, missing-member, last-member and request-order rejection
trials run outside timing. No training or threshold changes occur.

## Decision rule and limits

Retain a bounded generator only if its within-run median AND slower-request
behavior improve enough to justify added complexity while all outputs/bytes/
verdicts match. If collection does not explain the pauses, investigate the
remaining measured host stages rather than asserting that it did. Do not globally
disable garbage collection to make a benchmark look faster. Once this regression
is resolved, return to the broader temporal-Lingua/faithfulness questions rather
than treating serialization optimization as interpretability research.

A valid numerical receipt is still not proof of the correct generating label,
a faithful description of hidden neurons, or authenticated learned transitions.
