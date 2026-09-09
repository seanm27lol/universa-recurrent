# Temporal Lingua protocol — DRAFT, not implemented or preregistered

## Question

Does one bounded description of a short recurrent trajectory preserve more
behaviorally relevant information per byte/token and unit of total audit cost
than describing each state independently?

This question concerns the interpreter, not a new solver architecture. Existing
five-seed outcomes motivated it, so it is exploratory. Freeze a completed
protocol, code commit and new test data before any confirmatory evaluation.

## Stage 0: establish an observable, resumable state

Use the existing shared recurrent network, with dedicated depth 4 and a one-pass
control. Keep weights frozen and existing estimate/claim behavior unchanged.
Identify all state needed to resume at a cut: coordinates, candidate scores,
retained context and control variables. Test original-state resume against the
uninterrupted forward pass before compressing anything. No semantic claims yet.

A record must label its axes: model/checkpoint, example, recurrent step,
candidate, layer (when applicable), and token position (when applicable).

## Stage 1: mathematical trace bottleneck

Cover the same sites with: independent snapshots, a shared short-window record,
keyframe plus changes, and an equal-budget numerical compression baseline.
Known trace fields have defined semantics; this is not a learned NLA.
Record total bits/bytes and all costs, including dictionaries, coordinates,
precision metadata, adapters, reconstruction and serialization.

Require round-trip arithmetic checks on each retained field and a continuation
test at a single cut. Compare numerical error, probabilities, claimed-candidate
changes and failures. Test absent and wrong fields, ties, zero states and scales.

## Stage 2: learned language bottleneck (later work)

Fit a shared verbalizer across declared steps/sites, with a site-conditioned
reconstructor producing multiple vectors from a fixed language budget. Compare
against single-state descriptions at equal total tokens, parameter and compute
budgets, or explicitly report unmatched costs. Do not equate better MSE with
readability or semantic validity.

Controls: no-content/site-only reconstructor, shuffled descriptions across
matched examples, shuffled step IDs, input-only predictor, numerical codec,
paraphrases that preserve meaning, and controlled edits with predeclared expected
effects. Site-only success reveals a template shortcut, not recovered content.

## Leakage and causal boundaries

Split by complete independent problems/trajectories. No adjacent state from a
test trajectory can enter training. Train normalization only on training data.
An online summary uses a prefix, with no future state, hidden truth or final
answer. Retrospective reconstruction is labeled retrospective. Reconstruct a
single cut state, then run the native suffix without injecting future targets.
For recurrent models the same block at different steps is not multiple layers.

## Decision rule to specify before final testing

Set reconstruction tolerance, downstream deviation tolerance, semantic-edit
success criterion, and hardware-specific total-cost target on calibration only.
Hold out problem families and recurrence depths when feasible. Report coverage
of explained sites and an explicit unknown/unexplained option. Record generation
and checking are not part of the current v0.5.0 inference benchmark; measure them.

A useful result could be cheaper auditing without cheaper inference. A readable
but behaviorally irrelevant record fails the sufficiency goal. A good numerical
codec with unreadable codes fails the semantic goal. Retain each negative result.

See [concepts, distinctions and primary references](../docs/temporal_lingua.md).
