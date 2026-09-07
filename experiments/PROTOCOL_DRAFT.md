# Protocol draft: structured recurrence and property-specific witnesses

**Status: UNSEALED DESIGN DRAFT.** Toy examples and tests have already been run.
Nothing in this file is a preregistration. Freeze a protocol and fresh evaluation
seeds before making a future confirmatory claim.

## Proposed question

At matched reconstruction quality, which update strategy and evidence mode offer
the best measured tradeoff in total execution time, record size, and retained checks?

## Necessary comparisons

Cached direct constrained solve; classical reduced-coordinate iteration; future
fixed-depth neural baseline; future shared-weight recurrence; future adaptive
variant. Compare no record, compact record, and full record. Include a
same-dimensional incorrect/random subspace and ambiguous-structure controls.
Do not give only one arm privileged structure or extra measurements.

## Report separately

Constraint compilation; feature/routing costs including all evaluated candidates;
update time; record generation/serialization; checking; peak memory; record bytes;
reconstruction error; stationarity; constraint residual; refusal; budget failures;
latency distribution at batch one and useful batch sizes.

## Falsification and safeguards

If the direct baseline wins, report it. If compact mode cannot answer a history
question, record the loss rather than call it a faithful trace. Test wrong
constraints, deficient measurements, tiny/large numerical scales, changed
operators, malformed witnesses, and input-dependent halting failures.

Use independent capture/task-instance splits as relevant; never treat validation
observations consumed by the router as untouched test evidence. Retain all raw
rows, configuration, code commit, platform, seeds, and timing definitions.
