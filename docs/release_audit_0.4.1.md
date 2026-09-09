# Release 0.4.1: use the checked Git tree, not an overlay installer

## What went wrong

The earlier releases were assembled against different inherited file sets. The
v0.4 ZIP also had a stale importable version while project metadata said 0.4.0.
Changing an expected whole-source hash did not reconcile those different bases.
Some earlier reported test totals belonged to those different source snapshots.

This release was assembled against the exact source and tests of GitHub commit
`45e47cb156d1ebbb63d2751058c109dcc9e5a96b`. Their Git subtree hashes matched
`1e1577b39af98c1ad8e7a49653616a887b3f96f3` (src) and
`a4f4e118e16925b321ad3b6b27a8ec5bf3205a21` (tests). That baseline produced
101 passed and one optional integration skip in this audit.

## Corrections and evidence

| Issue reproduced or identified | Correction / regression test |
|---|---|
| Two separately edited version strings drifted | `__version__` is now the single source read by setuptools; installed metadata and CLI version must match |
| An editable install could point to a different checkout | `scripts/check_release.py` checks location, required modules, and both command families |
| Source tests alone could miss omitted wheel contents | `scripts/check_wheel.py` builds a wheel and imports it outside the source tree |
| Exactly zero residuals generated nonfinite backpropagation gradients in v2 | Use vector norms with a defined zero subgradient; zero-input tests cover shared, direct, and ambient models |
| Exact probability ties used different winners in Torch and NumPy | Declare first-index tie breaking; dense and compact tied records now verify |
| Valid library metadata could coexist with altered basis tensors in the weights | Check main and structured-control state dictionaries against the library before loading |
| A caller could mutate a model's input basis tensor after construction | Each model owns a cloned basis buffer |
| Evaluation could reuse a training/calibration seed and only label the overlap | v2 evaluation and benchmarking now refuse those seeds |
| A commit-reason string could disagree with decision timing | The checker verifies the exact reason, not only an allowed vocabulary |
| Counter materialization introduced unnecessary synchronization | Retain logical-hypothesis counts on device until reporting; compact indexing still has overhead |
| v1 timing silently assumed default noise/masking | Read the checkpoint's recorded distribution settings |
| v2 accepted zero noise although its Gaussian references require positive variance | Reject nonpositive v2 noise before training; the underlying data generator still supports noiseless examples |

## Scope

These are engineering corrections, not new scientific results. The neural v1
checkpoint format remains readable. Neural v2 retains a fixed small structure
library, parallel hypotheses, and calibrated commitment or abstention. It does
not implement arbitrary structure discovery, typed inter-space transport, neural
update replay, or semantic decoding of hidden features.

A probability-weighted provisional mixture need not satisfy either candidate
constraint. Lingua must not label it a verified single-structure answer.
Calibration thresholds are selected on the calibration split; this does not make
the probabilities calibrated or imply optimality on new data. Parameters,
examples, runtime, and training budgets remain different comparison axes.

## How to check your checkout

```bash
python -m pip install -e ".[test,neural]"
python scripts/check_release.py
python -m pytest -q
python scripts/check_wheel.py
```

Do not use a hard-coded test count as a health check. A missing test file must be
caught by release-content checks and dedicated neural CI; correctness comes from
the assertions actually run, not a number printed by an installer.

Local audit: CPU only, Python 3.13.5, PyTorch 2.10.0+cpu. The actual interpreter
and test log are recorded in `docs/local_validation.md`. GPU performance is not
claimed without another measured DGX run. CPU tests cannot certify GPU behavior.

## Established mechanisms, not novel packaging theory

- [Setuptools dynamic metadata](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html#dynamic-metadata): one version source.
- [Git stash](https://git-scm.com/docs/git-stash): preserve unfinished local work before an update.
- [PyTorch vector norm](https://docs.pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html): the numerical norm used by the residual features.
