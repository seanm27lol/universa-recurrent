# Checking a batch without reloading the reference for every record

Suppose a service checks 64 numerical receipts against the same trusted model
and calibration. Reloading that reference for every receipt took about 771 ms in
this run. Preparing it once, then checking all 64 receipts, took about 16 ms.
The latter measurement includes preparation.

This supports using the prepared verifier when a caller deliberately pins one
checkpoint/calibration version. It is a verification setup improvement; the run
did not measure inference or train a model.

## Results

Source: `verifier-setup-20260910-211521-QMTLW9-reports.zip`, SHA-256
`8d904dfd76a4e0409d406e669ce3c3e98ab7dea36abb6cab228057cc670aaf37`.
Five frozen checkpoints, PyTorch CPU thread setting 1, five measured repeats per condition.
Every duration below covers **64 receipt checks**.

| Model / receipt | Reload files per receipt | Prepare once, included | Already prepared | Paired time reduction, preparation included |
|---|---:|---:|---:|---:|
| Shared eight-step / endpoint | 770.77 ms | 15.90 ms | 3.88 ms | 97.95% |
| Shared eight-step / trajectory | 783.60 ms | 26.41 ms | 14.19 ms | 96.65% |
| Dedicated four-step / endpoint | 771.12 ms | 15.75 ms | 3.80 ms | 97.96% |
| Dedicated four-step / trajectory | 782.89 ms | 25.12 ms | 13.00 ms | 96.79% |

Durations are medians of five checkpoint-level medians. Reductions are medians
of the five within-checkpoint reductions, `100 * (1 - prepared_ms / files_ms)`.
Ratios of rounded table entries need not equal those paired reductions exactly.
The [complete recomputed table](../experiments/results/verifier_setup_20260910.json)
also records ranges and the audit counts.

For shared endpoint receipts, preparation-included speedups at counts 1, 4, 16,
and 64 are 0.978x, 3.836x, 14.452x, and 48.856x. Preparing a new reference for
one receipt is about 2.2% slower. Savings appear by the next measured count,
four; counts two and three were not measured. The already-prepared result
excludes initial preparation, whose median was 12.017 ms (range 11.818–18.330).

Preparation-included beats files-per-record in all 300 matched timing repeats
at counts 4, 16, and 64 across both models and retention types. These are
descriptive technical timing repeats, not 300 independent datasets.

## What was independently checked

- All 720 stored medians and percentiles were recomputed from 1,200 positive,
  finite timing samples; no discrepancies were found.
- All five standalone worker reports exactly match their summary copies.
- Recorded mode orders and repeat counts agree with the plan.
- The source checkout at `a18d8cc6881f0aa3ccfa78d6f832f1111b6b18ae`
  matches the report's benchmark-script and package-source SHA-256 values.
  This consistency does not authenticate the remote execution.
- Source inspection found that every call to the prepared verifier runs the
  existing arithmetic/geometry checker, then matches the pinned artifact hashes,
  library, model, depth and policy. It does not cache an earlier verdict.

The ZIP contains **no receipts, checkpoints or calibration files**. Its five
agreement flags are reported correctness evidence; they could not be replayed
from this ZIP. The benchmark's comparison of `checks` dictionaries compares the
outer retention-check map. The endpoint's detailed map is not exposed there.
Source inspection and the regression tests provide additional evidence for the
individual binding checks.

Two reporting defects are corrected for future runs: the warning now prints the
actual repeat count, and new fields distinguish the source pool from the distinct
records in a batch. The original archive is preserved unchanged.

## Boundaries of the result

Each model/retention condition cycles four saved receipts. A count of 64 does
not mean 64 different examples. No new task accuracy, generalization, causal
interpretability, neural transition verification or execution authenticity was
established. Endpoint and trajectory receipts retain different guarantees.

The file-based path was warmed; this is not a cold-storage experiment. Seven of
1,200 timing samples exceed twice their row median, with a maximum of 2.93x.
The large batch savings are consistent across all five checkpoints, but these
timings are specific to the reported machine and workload.

A prepared reference continues to identify its original snapshot after files
change. New files require a new reference. The two modes are comparable when
the reference files remain unchanged, as checked by this benchmark.

## Next measurement

[The verified pipeline study](verified_pipeline.md) measures batched inference,
transfers, receipt construction, JSON and bound checking on a fresh cohort,
checking every example. It saves all canonical receipts so their arithmetic can
be audited after the DGX run. No expected performance result is assumed.

Recompute the timing audit with:

```bash
python scripts/audit_verifier_setup.py /path/to/verifier-setup-reports.zip --output /tmp/verifier-audit.json
```

For the implementation, pinned-reference semantics and serialization references,
see [the setup experiment](verifier_setup.md).
