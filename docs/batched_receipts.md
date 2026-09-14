# Check a stack of worksheets without averaging away mistakes

**Status: implemented as opt-in research scripts. Run the benchmark to find out
whether grouping helps on your hardware. No new speedup is claimed.**
Neural weights, training, thresholds, record formats and package version 0.5.2
are unchanged. The existing scalar prepared verifier is the baseline.

## Familiar example

A teacher can compare a whole column of calculations against one reference
sheet. That saves repeating the same arithmetic setup. But one incorrect answer
still belongs to one student: a good class average must not turn it into a pass.

```text
CPU inputs -> SAME inference -> individual JSON receipts
                                      |
                         +------------+-------------+
                         |                          |
                 prepared scalar check       grouped array checks
                 one receipt at a time       schemas/identities per receipt
                         |                          |
                         +------------+-------------+
                                      |
                  one verdict per expected input, in request order
```

This is ordinary array vectorization, not a new mathematical verification
principle. NumPy's [broadcasting guide](https://numpy.org/doc/stable/user/basics.broadcasting.html)
explains how array operations move loops out of Python, and why large temporary
arrays can erase a performance gain. Here groups are limited to 64 records;
very large or unusual records use the unchanged scalar checker.

## What is checked

Let `b` identify a record, `k` a candidate and `n` a signal coordinate. Candidate
`z[b,k,n]` has weight `p[b,k]` and a boundary matrix `B[k]`. The mixture check is

```text
mu[b,n] = sum_k p[b,k] * z[b,k,n]
```

Candidate feasibility checks `B[k] z[b,k]` against the SAME fixed tolerance as
before. Reductions over coordinates/candidates produce a boolean for EACH `b`,
never an average over records. Trajectory checks additionally cover coordinate
decoding, basis orthonormality, logit/probability readout, observed residuals,
progress arithmetic and agreement with the endpoint.

Headers, scope flags, shapes, masks, finiteness, model/depth, calibrated policy,
checkpoint/calibration identifiers and library match are checked for each record.
Claim selection still uses the historical first-index tie rule. A candidate's
certificate never transfers to the mixture. The checker does not run a neural
update, infer whether the chosen structure is true, or authenticate execution.

The fast path accepts only inside a conservative interior of the original
numerical tolerances. A boundary case, unusual shape or failed grouped check
falls back to the ORIGINAL prepared checker for that record. It is not rejected
just for missing the fast path, and neighboring records are not skipped. Fallback
counts are reported: a corrupt or pathological workload can eliminate savings.
This is differential regression validation, not a formal proof of equivalence
for all possible floating-point values or all future NumPy implementations.

`expected_count` is caller-supplied, not inferred from the received list. The
benchmark also supplies the original observations, masks, model and retention
kind as external expectations. This catches missing, extra, duplicated or
reordered inputs. Without external request expectations, internally valid
receipts alone cannot prove that the right inputs were processed. Duplicate
requests may be legitimate; the system does not globally ban equal inputs.

## What the benchmark changes

| Mode | Receipt creation | Checking |
|---|---|---|
| `prepared_scalar` | Existing implementation | Existing prepared checker |
| `batch_check` | Existing implementation | Grouped checker |
| `batch_generate_and_check` | Batched host conversions | Grouped checker |

The third mode converts complete host tensors once and shares constant metadata
during construction. It still emits the EXACT same individual JSON bytes. No
new packed protocol, omitted fields, precision reduction or binary codec is
introduced. JSON serialization and strict parsing still happen for each record.
All modes must match numerical outputs, discrete decisions, and canonical
receipt bytes. Valid-record verdicts and retained-step counts must agree.

The timed request starts with CPU inputs and ends after every receipt has a bound
verdict. Input/device copies, inference, output/receipt creation, JSON, parsing,
mathematical checks and synchronization are included. Model loading and verifier
preparation are measured separately as startup, not hidden inside the speed
ratio. Dataset generation, report writes, network/queueing and differential
safety tests are excluded. Stage times are recorded; total is measured directly.

Existing five checkpoints, shared eight-step, dedicated four-step and one-pass
controls are used. Counts 1,16,64,256 are nested prefixes of the same 256 inputs
(test seed 46000, intentionally reused for a performance ablation). Ten repetitions
regenerate receipts, not new independent inputs. Each checkpoint runs in a fresh
process; checkpoint, case and mode order are randomized and recorded. Backend
settings match the explicit prior pipeline configuration. Do not run concurrent
GPU jobs. This is a warmed local request experiment, not service latency.

Each worker also tests corruption of the final record, false scope, wrong artifact
version, malformed JSON, missing/extra members and wrong input order. Trajectory
corruption is tested when applicable. Reference files are checked for changes
before and after each worker. Preparation refers to a pinned artifact snapshot,
not automatically changing file paths. Reprepare for a different version.

## Run on the Spark

```bash
git pull --ff-only origin main
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/check_release.py
.venv/bin/python -m pytest -q tests/test_batch_receipts.py
bash scripts/run_batched_pipeline_study.sh /absolute/path/to/replication/results cuda
```

The script makes a fresh `runs/` folder and a reports-only ZIP in Downloads.
No training, recalibration, overwrite of earlier reports, upload of weights, or
Git push occurs. Reports contain every canonical maximum-batch receipt, raw
measurements, stage costs, fallback counts, per-worker comparisons and failures.
For a tiny CPU smoke test use the Python CLI with smaller counts and repeats.

## Logical review and limits

Regression tests compare both implementations on valid records, heterogeneous
batches, thresholds near tolerances, tied probabilities, bad final records,
finite values whose arithmetic overflows, large groups, missing inputs and
reference-version mismatches. No previous PASS is cached. Existing neural tests
remain unchanged. CPU tests and local fixtures are not a DGX speed result.

For malformed/numerically marginal records the scalar fallback remains the
reference behavior. The added request-count/input-binding requirements are
applied equally to both benchmark paths. Resource limits (4096 records, 64 MB
per batch, existing 10 MB single-record bound) are transport limits, not a new
claim about the mathematics. They are not a complete hostile-process sandbox.

Do not compare this result against the earlier reload-per-record baseline: the
research question is whether it improves on already-amortized verification.
