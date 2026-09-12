# From an input batch to checked receipts: September 11 results

For 256 new observations, shared eight-step inference through completed endpoint
checking took **3,179.01 ms** when the reference was loaded for every record and
**39.28 ms** when it was prepared once inside the request. With an already
prepared reference, the request took **26.75 ms**.

This is a substantial improvement to the verified request pipeline. The saving
comes from removing repeated reference setup. Neural weights, updates, depths,
claim policies and receipt property checks were unchanged.

## Directly measured request costs

All rows below cover **256 records**, including transfers, batched inference,
record construction, JSON serialization, parsing and every bound record check.

| Model / receipt | Files per record | Prepare once, included | Already prepared | Paired speedup, preparation included |
|---|---:|---:|---:|---:|
| Shared eight-step / endpoint | 3,179.01 ms | 39.28 ms | 26.75 ms | 80.61x |
| Dedicated four-step / endpoint | 3,184.13 ms | 38.22 ms | 25.85 ms | 82.98x |
| One-pass / endpoint | 3,223.88 ms | 37.34 ms | 24.90 ms | 85.91x |
| Shared eight-step / trajectory | 3,283.98 ms | 99.60 ms | 86.49 ms | 32.84x |
| Dedicated four-step / trajectory | 3,225.49 ms | 87.03 ms | 74.74 ms | 37.15x |

Durations are medians of five checkpoint-level medians, each from five timing
repeats. Speedups are medians of within-checkpoint time ratios, not ratios of
rounded table entries. The total was timed directly; stage medians were not
added together to construct it. Endpoint and trajectory receipts check different
properties, so compare verification modes within each row.

For shared endpoint requests, preparation-included paired speedups at batch
counts 1, 16, 64 and 256 were 0.975x, 11.52x, 37.87x and 80.61x. Preparing a
fresh reference for one receipt was about 2.59% slower. Already-prepared single
requests took 2.426 ms for shared, 1.522 ms for four-step and 0.671 ms for one-pass.
Dividing a batch time by 256 gives amortized throughput cost, not single-request
latency.

Initial model loading was separately reported: median **453.78 ms**, range
448.26–524.13 ms. Initial reference preparation was **12.818 ms**, range
12.630–13.286 ms. Already-prepared requests exclude that initial setup. Network
transport, queueing, data synthesis, process startup and report writing are also
excluded. These are warmed local pipeline measurements, not deployed-service
or cold-start latency.

## The remaining cost is mainly record processing at larger batches

For shared endpoint requests at count 256 with the reference already prepared,
median-of-checkpoint median stage times were approximately:

| Stage | Time |
|---|---:|
| Inference and output construction | 2.510 ms |
| Host records and JSON serialization | 7.375 ms |
| Parsing and bound property checks | 16.730 ms |

These separate stage medians need not sum to the median total. Per-sample stage
shares, aggregated by checkpoint medians, attribute about 63% to parsing/checks,
28% to records/JSON and 9% to inference. At count one, inference is the main
steady-state cost instead. The relevant bottleneck depends on request size.

## What was independently checked

The archive contains five worker reports, 100 model/retention/batch conditions,
1,500 timed requests and **6,400 canonical receipts** from maximum-batch runs.
There are **256 distinct observation/mask inputs** shared across model fits and
timing repeats. More receipts and repetitions do not create more independent
input problems.

The unchanged NumPy property checker was locally replayed over all 6,400 saved
receipts, with **zero failures**. This checks endpoint arithmetic, candidate
geometry, policy arithmetic, and retained trajectory geometry/diagnostics as
applicable. The checker was loaded without PyTorch, checkpoint files or model
execution. This is an independent local replay of the existing checker, not a
separately invented proof system.

All 25 saved batch hashes match their receipt bytes. All 100 conditions' reported
canonical and timed receipt hashes agree within condition. The reported
rollout/final-output comparisons have zero differences and identical discrete
decisions. Their repeated endpoint/trajectory copies are not additional
independent equivalence experiments.

All 2,560 saved pairs of standalone endpoint records and endpoints embedded in
trajectory records also match exactly, with independently computed maximum
difference zero. This saved-record comparison is replayable locally, unlike
the unsupplied full GPU tensors.

The timing audit recomputed all 900 stored medians and percentiles from 1,500
samples and checked the worker/summary arithmetic, stage accounting, recorded
orders and denominators without discrepancies. Twelve timing samples exceed
twice their own condition median; the maximum is 5.13x. These outliers remain
in the reports. The [complete recomputed timing table](../experiments/results/verified_pipeline_20260911.json)
preserves raw-comparison summaries and limits.

Only maximum-batch receipts are saved. The 75 smaller-batch receipt hashes do
not equal prefixes of the maximum-batch receipts. Consequently those smaller
records cannot be replayed from the saved maximum batch. This is compatible with
batch-shape-dependent floating-point outputs and does not contradict the
within-condition equality reports. The ZIP does not establish cross-batch
bitwise equality.

The five reported checkpoint/calibration hash pairs match the earlier setup
reports. The package-source and benchmark-script hashes match the supplied local
source. Checkpoint and calibration files were not uploaded, so artifact binding
and GPU execution were not independently replayed or authenticated. A valid
receipt still does not establish correct routing or a learned transition.

## A separate, post-hoc quality comparison

The local generator reproduced the complete reported dataset fingerprint,
including truth and labels, exactly. Every observed/mask pair in all saved
receipts also matched. This permits comparison of the saved endpoint estimates
against the regenerated truth without running the neural models again.

This was a **post-hoc exploratory analysis of 256 timing inputs**, not a
preregistered quality experiment. Each entry averages five checkpoint-level
metrics; conditional wrong-route rates are averaged per checkpoint.
The [complete quality calculations](../experiments/results/verified_pipeline_quality_20260911.json)
include per-seed results, counts, input-paired comparisons and the exact dataset
reconstruction checks.

| Model | Estimate MSE | Top-route accuracy | Claim coverage | Wrong route among claims |
|---|---:|---:|---:|---:|
| Shared eight-step | 0.052423 | 76.72% | 73.36% | 13.42% |
| Dedicated four-step | 0.052814 | 77.66% | 72.42% | 11.79% |
| One-pass | 0.057171 | 77.81% | 73.98% | 12.23% |

Shared mean MSE is 8.31% lower than one-pass and is lower in all five fits.
Four-step mean MSE is 7.62% lower than one-pass. Shared versus four-step is a
small, mixed tradeoff: 0.74% lower mean MSE, four of five fits lower, but only
121 of 256 inputs have lower shared loss after averaging fits. A few larger
improvements can lower mean loss while more individual inputs favor the other
model.

The analytic Gaussian reference, which knows the synthetic prior and noise,
has MSE **0.051764** and route accuracy **78.125%** on this cohort. It was
evaluated once in CPU float64 arithmetic. It is a privileged estimation
baseline, not a receipt checker or a newly trained model.

About 13.4% of the shared model's issued structural claims select a label other
than the generating structure, even though every saved receipt passes its
declared property checks. Claim correctness and candidate feasibility remain
distinct. The policy targets the fraction of cases with a claim; it is not a
guarantee that claims have a specified error rate.

## Research decision

Retain the prepared reference for workloads that intentionally reuse a frozen
checkpoint/calibration version. Keep endpoint and trajectory modes explicit.
The file-per-record path remains a useful setup-cost control, but its repeated
large overhead is now well characterized.

The next scientific priority is claim reliability on a larger untouched cohort:
compare wrong-route rates at matched coverage for shared, four-step and one-pass,
including the privileged analytic baseline. Choose any new thresholds on a
separate calibration split and leave the final test split untouched. Keep the
four-step model as a practical baseline; this small post-hoc result does not
justify declaring either recurrent depth a universal winner.

For engineering work on large batches, profile and reduce record construction,
JSON and per-record checking while preserving every declared property. Faster
GPU inference alone has limited room to improve the current large-batch verified
pipeline.

## Provenance

Recompute report/timing accounting with the standard-library auditor (this
command does not run receipt geometry checks or neural inference):

```bash
python scripts/audit_verified_pipeline.py /path/to/verified-pipeline-reports.zip \
  --source-root . --output /tmp/verified-pipeline-audit.json
```

The new auditor has 33 focused regression tests covering altered timing,
summary, identity, ordering, coverage and archive contents. The integrated local
suite passed **395 tests**, with three CUDA tests and one optional upstream
integration test skipped. The release check passed. No neural model or producer
benchmark code was changed during this results audit.

- Original archive: `verified-pipeline-20260911-213701-LSBRa7-reports.zip`.
- Archive SHA-256: `6100c7f2f60e89427cfef9e7525437e87bac26d622d9f226f85ec049400b725b`.
- Portable source commit: `c42323df6ffb2f1533fa260760e110bd5e6ed92d`, package 0.5.2.
- Dataset seed 36000, 256 examples, fingerprint
  `176530c5c64f58dad897afe32a28e31d2721b8ea04ffc24a5484b38b9000ec07`.
- NVIDIA GB10, CUDA 13.0, PyTorch 2.14.0+cu130, Python 3.11.14, NumPy 2.4.6.
  Deterministic algorithms enabled, TF32 disabled, PyTorch CPU thread setting 1.
- Original [experiment protocol and boundaries](verified_pipeline.md).
