# Independent acquisition implementation and artifact review

## Verdict

The artifacts support a **bounded, document-local complete-program memorization result**. Dense, untyped MoE and
typed MoE each completed 4,096 updates and 5,201,920 loss-bearing tokens, then generated 32/32 exact training
answers with canonical syntax and EOS. The same arms remain 0/32 on the separate development set. No generalization,
architectural superiority, usable semantic state interface, or readiness to scale is established.

The reviewer examined the implementation and saved evidence read-only and did not design the data correction or a
weight edit. This is independent implementation/artifact review, not an external replication or a human sign-off.
The detailed measurements and reproduction commands are in `ACQUISITION.md`.

## Principal findings

1. **The first failure is explained by positions, not missing answer labels or missing experts.** The original
   intact packer blocks cross-document attention but continues absolute row positions across examples. Only the
   11 row-first examples generate correctly in each arm. Replaying all examples at their trained positions recovers
   32/32. Answer labels and state/route targets are present. Zero-position answer CE and every saved completion
   reproduce exactly.
2. **The corrective comparison changes one data convention.** Only `train/positions.npy` and `dev/positions.npy`
   differ. The other 18 arrays, examples, loss masks, auxiliary targets and consumed-document indexes are identical.
   Initial model, optimizer and scheduler hashes match per arm. The model, loss, generation and scoring code are
   unchanged. The training commits differ, but the effective model and optimization configuration is matched.
3. **All endpoints and local checkpoint identities hold.** Both runs have 4,096 actual training records per arm,
   final step-4096 checkpoints and the declared token count. All six checkpoint hashes match their files and
   manifests. No control was truncated to let a candidate finish.
4. **Independent execution/rescoring reproduces the reports.** The scorer imports no producer modules, executes
   all 64 source programs per run, rejects no valid stored targets and reports zero score changes. It enforces
   canonical registers and complete text equality. This validates saved completions and targets, not independent
   model regeneration.
5. **Numerical history remains bounded and explicit.** All six FP32 full/cached reconstruction reports pass the
   declared relative-RMS tolerance, with exact integer routes and fixed-replay tokens. Maximum relative RMS error
   was about 5.89e-7 parent and 4.45e-7 corrective. Strict `allclose` is false. These are not bitwise results or
   BF16 serving equivalence; the earlier BF16 route failure and strict FP32 failures remain preserved.
6. **Routing is measured, not credited with superiority.** The parent untyped final dev pass-four expert zero has
   0/2,539 token-level decisions and is marked starved. Corrective untyped and typed final dev summaries have no
   zero-count expert. Full per-pass route probabilities and counts remain in the artifacts.
7. **Resource accounting is within the authorization.** Parent reservation `20261110T075533-31440af9` charged
   0.23620130574806097 h; corrective reservation `20261110T091534-4b1c96d2` charged 0.24603162081084307 h. Their sum is
   0.48223292655890404 h. Native cumulative usage is 3.785954902/12 h, leaving 8.214045098 h. CPU-only analyses
   opened no GPU reservation. No unrelated process was terminated.

## Learned-state result and limits

The scoped intervention identifies the final instruction's last overlapping prompt token and suppresses all four
slot-writer outputs at that token during pass two, the first supervised after-state. The incoming slot state is
retained; later passes and suffix forwards receive no direct suppression. All 96 baseline completions reproduce
the saved evaluation. The intervention changes **0/96 completions and 0/96 generated suffixes**; exact answers stay
32/32 per arm. Checkpoints and weights remain unchanged.

This is a negative generation-level intervention, not evidence that the state is absent. It covers one final
instruction, one pass and memorized prompts only; other paths may be redundant. It also does not demonstrate a
usable semantic state interface. Perfect training readout and a construction-time causal path do not fill that gap.

The original gradient artifact summed per-unit squared norms rather than summing parameter-gradient vectors first.
It is preserved but not used for aggregate-gradient claims. The v2 artifact computes aggregate vectors correctly,
reproduces saved CE and has a cancellation regression test. No diagnostic performs an optimizer or weight update.

## Reviewed corrections and remaining limitations

- Parent protocol/status/diagnostic hashes and live ledger charges are enforced before the corrective launch.
- Rescoring now uses exclusive output creation, with a preservation test; reproduction commands use a fresh
  temporary directory and can be repeated.
- Post-run diagnostic source and artifact hashes are recorded separately from training commits at
  `diagnostics/diagnostic_identities.json`.
- The original `evidence/` exports remain untouched. **Use `evidence-final`, not the earlier exports**, which retain
  their earlier delivery limitations.
- Each final bundle has 54 files, with 52 checksummed entries. Coverage is every file except `SHA256SUMS` and
  `bundle_manifest.json`; this convention is explicit. No weight files are included.
- Bundle reproduction was executed successfully using only the bundle and source-checkout scorer. Reproduced
  diagnostics match the preserved diagnostics except the `study` path.
- One duplicate repository-wide local test invocation was stopped before the first timed run to avoid CPU
  contention; the required remote full CI for each training commit passed instead of treating that local run as a
  result. The final native suite reports 260 passing tests. Final PR CI is reported separately with the PR commit.

## Final evidence locations

Parent, source commit `60fec73c7c925e6ddf969b97ff18f8d0f7318d35`:

```text
/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_acquisition/acquisition/20261110T075533-a8827fef/evidence-final/
```

Corrective, training source commit `a708fc5`:

```text
/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_acquisition_local_positions/acquisition/20261110T091534-c25a2b1b/evidence-final/
```

The final sealed bundles contain 47,629,190 and 48,038,186 hashed payload bytes respectively. Post-run diagnostic
code is commit `84cfa7b99fc9a52059ba99c87f4fb5d02efada32`, distinct from the training commits. Hashes identify
local code and artifacts; they do not authenticate a remote execution.

## Recommended boundary for the next experiment

Use new disjoint programs and predeclared composition/length holdouts to test generalization, with equal budgets,
lookup/executor baselines and explicit state interventions. More repetitions of the same 32 prompts would only
repeat the memorization result. Any such run needs a separately frozen protocol and allocation. No 124M campaign,
WSP edit or new GPU run was launched during this review.
