# Local validation

Prepared September 8, 2026. These are development checks, not preregistered
scientific results.

## Environment

- Python: 3.13.5
- NumPy: 2.3.5
- PyTorch: 2.10.0+cpu
- Pytest: 9.0.2

The local environment had no GPU. GPU speed and compact-execution behavior must
therefore be measured on the target machine rather than inferred here.

## Checks completed

- Editable installation and source-tree execution: passed.
- Classical full and compact demonstrations: passed.
- Classical independent witness checks and tamper tests: passed.
- Neural CPU train → checkpoint → demo → checkpoint-bound Lingua verification:
  passed.
- Neural dense and compact adaptive outputs: matched in routes and step counts,
  and agreed within `atol=1e-7`, `rtol=1e-6`; different batch shapes can change
  the final float32 state by a few ulps.
- Every nonempty five-coordinate mask pattern: transparent and Gaussian-reference
  solvers returned finite outputs.
- Unobserved-value leakage guard: changing masked coordinates did not change the
  neural output.
- Neural record tampering checks: altered state, diagnostics, event continuity,
  halt metadata, update counts, and checkpoint identity were rejected.
- Checkpoint checks: legacy v0.2 semantics load explicitly; unknown formats,
  nonfinite tensors, and unsupported PyTorch loader versions are rejected.
- Tiny deterministic CPU reruns: model tensors, history, and evaluation matched;
  checkpoint file hashes differed because elapsed-time metadata is intentionally
  recorded.
- Python compilation: passed.
- Local Markdown links: passed.
- GitHub Actions: configured, not yet executed for this source revision.
- Optional commit-pinned Universa adapter: skipped locally because this runtime
  could not resolve `github.com`. Its dedicated CI job remains the integration
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

SHA-256 over sorted Python paths and bytes in the installed package:

```text
a8c18dbd1165e6ad18d9ff60dd09cb693845688765fd2949fd01767ad871d1be
```

This identifies tested source content. It does not authenticate a remote
execution.
