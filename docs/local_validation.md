# Local validation

Prepared September 7, 2026. These are development checks, not preregistered results.

- Python: 3.13.5
- NumPy: 2.3.5
- Pytest: 9.0.2
- Editable installation: passed using installed dependencies (no network).
- Full-trace CLI demo and independent checker: passed.
- Compact-trace CLI demo and final-only checker: passed.
- Retained full and compact example records: checked by the test suite.
- Python compilation and publishing-script shell syntax: passed.
- Local Markdown file links: checked with no missing files.
- Publishing guard tests: offline stubs only; no repository was created.
- GitHub Actions: configured, **not executed**.
- Optional upstream Universa adapter: **skipped**, not installed. A public
  clone attempt failed because this runtime could not resolve github.com.
  The adapter is an unexecuted integration until its optional CI/test is run.

## Test output

```text
................................................................s....... [ 81%]
................                                                         [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_upstream_adapter.py:9: Optional commit-pinned Universa dependency not installed
87 passed, 1 skipped in 0.26s
```

## Exploratory timings

Raw development rows are in
[development_microbenchmark.json](../experiments/results/development_microbenchmark.json).
They compare direct, compact-recursive, and full-recursive paths on one tiny graph
family. The direct solve was faster in this development run. No general speedup,
GPU performance, or trained-model conclusion follows.

## Source identity

SHA-256 over sorted source paths and bytes: `f1210713fe47b58fff467ea61d6dd2d886593e33598af8472acf50104c9238dd`.
This identifies tested source content, not a remote execution attestation.
