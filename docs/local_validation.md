# Local validation for the GitHub-native 0.4.1 release

This is a development audit, not a preregistered scientific result.

| Item | Observed outcome |
|---|---|
| Interpreter | Python 3.13.5 |
| Numeric libraries | NumPy 2.3.5; PyTorch 2.10.0+cpu |
| Test runner | pytest 9.0.2 |
| Published v0.3.1 source/test reproduction | 101 passed; one optional Universa skip |
| Audited 0.4.1 tests | **129 passed; one optional Universa skip** |
| Package/version/module/CLI check | Passed |
| Build wheel, install to separate directory, import outside src | Passed |
| Python compilation | Passed |
| Neural v1 compatibility smoke tests | Passed in the test suite |
| Neural v2 train/eval/benchmark/demo/checkpoint-bound record check | Passed, CPU only |
| Additional eight-step v2 smoke with controls | Passed: 96 train, 32 calibration, one epoch; NOT a quality benchmark |
| Zero-residual gradient and exact-tie record regressions | Passed |
| Altered basis tensors and mislabeled commitment reasons | Rejected |
| Train/calibration seeds reused as test/benchmark seeds | Rejected |
| GPU execution/performance of 0.4.1 | **Not tested locally** |

The optional integration dependency was not installed in this container. Its CI
job is separate; a local skip does not establish an integration result.

Earlier statements of 103, 112, or 113 tests referred to other assembled file
sets, not the exact published v0.3.1 base plus this release. See
[the release audit](release_audit_0.4.1.md). Do not infer feature coverage from a
hard-coded test total. Check which modules and tests were included and run.

```text
129 passed, 1 skipped
```

The GitHub publication process runs the tests again on the committed source.
Only report that remote outcome after inspecting the actual workflow result.
