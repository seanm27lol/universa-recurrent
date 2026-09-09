# 0.5.0 audit: two outputs, matched comparisons

This release adds a matched-output study around existing v2 models. It does not
change their neural update rules or rewrite the earlier experimental results.

## Logical issues addressed

| Issue | Change and regression check |
|---|---|
| A structural commitment replaced the weighted numerical estimate | New contract returns mixture plus a separate candidate claim; threshold invariance is tested |
| A mixture could be mistaken for a constraint-certified candidate | Independent checker distinguishes the two tensors and rejects certificate transfer |
| Only the main model received a selective policy | All probabilistic structured controls get their own cutoff under the same coverage rule and calibration observations |
| Untimed mixture quality was combined with hard-output timings | Same callable, output mode, cohort, and batch shape supply evaluation and timing; every timed output tensor is compared with its evaluated counterpart |
| Mixture-only output was counted as universal abstention | No requested claims means coverage is not applicable, not zero |
| Selective error denominators were easy to confuse | Report both wrong claims divided by all examples and by claimed examples, using integer counts |
| One timing average hid variability and execution order | Interleave cases with recorded RNG seed; report raw repeats, median and quantiles |
| A small statistical result could be mistaken for replication | Replication utility rejects duplicate training seeds/checkpoints and reports training-level spread, not pooled-example significance |
| Empty or tied decisions caused edge-case ambiguity | Explicit no-claim policy, inclusive cutoff, first-index ties, and zero-denominator tests |
| Nonfinite derived values could evade a numerical bound | Validate derived geometry and squared errors, including finite inputs that overflow |
| Previous releases omitted modules or disagreed on version | Extend release and installed-wheel checks; retain a single authoritative version literal |

## Scope of validation

New tests cover success and rejection paths, threshold invariance, corrupted
records, checkpoint/calibration binding, seed collisions, output overwrite
refusal, hidden-value masking, an end-to-end comparison, and the prohibition on
neural replay by the property checker. CPU execution is tested; DGX/CUDA timing
must be measured on the user's hardware. Exact test output is in CI, rather than
a hardcoded expected total inside an installer.

The source-transfer step verifies every changed file before creating a clean
Git commit; installation does not use a whole-checkout hash dependent on local
extras. No new user-side installer is required.

## Remaining logical boundaries

- A checked structural proposal is not proof that the chosen structure is true.
- A trusted-file hash binds bytes, not a remote execution or faithful semantics.
- Empirical coverage calibration does not guarantee test coverage or risk.
- Fixed-depth inference is deliberate in this ablation. No early-exit speedup is
  claimed; rollout allocation and output construction are included in timing.
- Existing models were trained with different objective details. Matching their
  output policies and parameter counts does not fully isolate architecture.
- The Gaussian reference knows the synthetic generator. It is privileged.
- Optional v1 comparison uses identical current test examples but historically
  different training. It is diagnostic, not a paired architecture-training trial.
- Previously viewed results motivated this design; it remains exploratory.
