# Local validation

Updated September 9, 2026. These are development checks, not preregistered
scientific results.

## Publication repair

The first public v0.3 commit (`56163c7`) accidentally omitted the neural model,
training, neural Lingua, neural checker, and neural test files while retaining
documentation that referred to them. A DGX checkout of that commit reported
`85 passed, 1 skipped`; that was a valid check of the classical layer, but it did
**not** exercise learned recurrence.

Version 0.3.1 restores the omitted source and makes the dedicated CPU-neural CI
job fail if those files or commands disappear again.

## Local environment

- Python: 3.13.5
- NumPy: 2.3.5
- PyTorch: 2.10.0+cpu
- Pytest: 9.0.2

The local validation environment had no GPU. GPU timing and compact-execution
behavior must therefore be measured on the target machine rather than inferred.
The earlier DGX v0.2 exploratory run does not validate the repaired v0.3.1 code.

## Checks completed

- Editable installation and source-tree execution: passed.
- Classical full and compact demonstrations: passed.
- Classical independent witness checks and tamper tests: passed.
- Neural CPU train → checkpoint → demo → checkpoint-bound Lingua verification:
  passed.
- Neural dense and compact adaptive outputs: matched in routes, logical step
  counts, and final states within declared float32 tolerances.
- Dense and compact execution report logical updates separately from actual
  sample-update evaluations.
- Every nonempty five-coordinate mask pattern: transparent and Gaussian-reference
  solvers returned finite outputs.
- Unobserved-value leakage guard: changing values where the mask is zero did not
  change routes or neural outputs.
- Neural record tampering checks: altered state, boundary identity, event state,
  halt metadata, execution counts, and checkpoint identity were rejected.
- Checkpoint checks: v0.2 residual and halt semantics are preserved; unknown
  formats, nonfinite tensors, changed bases, and unsupported loader versions are
  rejected.
- Restricted loading: `weights_only=True` plus a narrow `TorchVersion` allowlist
  for legacy metadata.
- Full requested benchmark size is processed, rather than silently timing only
  the first batch.
- Python compilation: passed.
- Optional commit-pinned Universa adapter: skipped locally because this runtime
  could not resolve `github.com`; its dedicated CI job remains the integration
  check.

## Test output

```text
........................................................................ [ 69%]
........s.......................                                         [100%]
SKIPPED [1] tests/test_upstream_adapter.py:9: Optional commit-pinned Universa dependency not installed
103 passed, 1 skipped
```

## Exploratory boundaries

The repository distinguishes:

```text
logical updates ≠ examples executed by the update network ≠ wall-clock time
```

A tiny CPU smoke run validates plumbing only. It is not evidence of model quality
or acceleration. The public neural protocol remains a draft until its code,
seeds, thresholds, hardware regime, and decision rules are frozen before a
confirmatory run.

## Source identity

SHA-256 over sorted Python paths and bytes in the repaired installed package:

```text
7d4d44d8b84f8a25779d9db60f7d45f6c418976375df51664178cc54ef41c04b
```

This identifies tested source content. It does not authenticate a remote
execution.
