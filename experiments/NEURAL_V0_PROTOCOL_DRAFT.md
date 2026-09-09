# Neural v0 experiment protocol — SUPERSEDED DRAFT / NEVER SEALED

This was the planning record for the first exploratory implementation. It is
retained because the later code audit was informed by its run. It is not a
preregistration and supports no scientific claim.

## Original question

Can a small structure-aware recurrent network recover held-out synthetic flows
while using a learned structure route and fewer logical recurrent updates?

## Original design

- two known graph-flow constraint spaces;
- 70% observation probability and Gaussian noise std 0.05;
- one MLP router;
- a two-coordinate state per candidate space;
- one shared update reused for eight steps;
- a halt head trained against one absolute truth-MSE threshold;
- held-out dataset seeds.

## Why it was superseded

Review after the first run found that:

1. a lower logical step count did not imply skipped batched GPU computation;
2. the residual summary divided by ambient rather than observed coordinates;
3. halt supervision covered only the true candidate path;
4. evaluation omitted shallow fixed-depth, transparent structural, and
   generator-aware references;
5. the neural Lingua record had no independent checker.

Those findings are documented in `docs/development_audit.md` and addressed by the
unsealed neural v1 implementation.
