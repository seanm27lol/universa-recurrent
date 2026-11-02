# Performance study v2: relational grounding, matched over three seeds (protocol)

**Status.** Frozen when its sha256 is committed in `pins/perf_v2_protocol.json`; every study command refuses to run
if this file, `pins/perf_v2_train.json` or `pins/perf_v2_eval.json` differs from the pin. Everything below is fixed
before any training or evaluation of this study. The pilot (`PROTOCOL_pilot.md`, PR #68) is not repeated, and its
protocol, records, checkpoints and ledger entries are kept as they are.

## Authorization (user, 2026-11-02)

- GPU authorization v2: **32 cumulative GPU-reservation hours**, the 6.878 h already used included (`ledger.py`,
  `AUTHORIZATIONS`). Every reservation, failed, reloaded or interrupted, is charged. One process holds the GPU. No
  paid cloud resource. The pilot's commands keep authorization v1's 8-hour cap.
- Planning allocations (not guaranteed runtimes): B4/Q27 completion 4 h; profiling and paired smoke 1.5 h; matched
  training 9 h; development evaluation and further Qwen comparisons 6 h; recovery and contingency 4.6 h.
- The 500-question reserved BIRD Mini-Dev benchmark is not read. No merge.

## The intervention, chosen by the CPU diagnosis

The diagnosis (`diagnose.py`; records in `runs/.../perf_v2/diagnosis/`) read only the pilot's saved records and
checkpoints, the 64 historical questions and a fixed training-pool sample, on the CPU. It chose the intervention;
the new population was never read.

| finding (4 pilot fits, 64 historical questions) | value |
|---|---|
| teacher-forced error, first FROM table (training sample) | 49% (17%) |
| teacher-forced error, joined table | 52% (16%) |
| teacher-forced error, SELECT / WHERE columns | 42% / 33% (10% / 10%) |
| teacher-forced error, join key: first / second column | 43% / 21% (8% / 3%) |
| predicted table set equals gold | 16-21 of 64 |
| join keys right when the table set is right | 15-19 of 16-21 |
| outputs with a same-reference ON equality | 42 (45 equalities); all 6 scorer timeouts among them |

The dominant bottleneck is grounding tables and columns in unseen databases, not join keys given the right tables.
The intervention is **relational grounding** (`grounding.py`), one mechanism with four parts, all in the trunk that
T4 and C4_params share: a relation-aware schema encoder (tables see their columns and key-linked tables), item
features (keys, name mentions), plan-relative pointer relations (including the second join operand scored against
the first), and the constraint that an ON comparison may not compare one column reference with itself. The
comparison tests the mechanism as a whole, not its parts. In particular the v2 trunk has 4,218,064 more parameters
(the two encoder layers, 15,735,765 against 11,517,701), and there is no capacity-matched v1 control, so an
improvement cannot be split between the relations and the added capacity.

**Independent review before the freeze** (2026-11-02, of 1f950ec): no defect invalidating training or the
comparison. Fixed before the freeze: the "already referenced" pointer bit read enclosing and nested SELECTs (1.5% of
training column-pointer steps), and bits 5 and 6 now follow the chosen occurrence of a self-joined table; the
projection now applies the update rule's contingency; a second evaluation of a seed is refused; the decodes are
written before scoring; the decoding relation bits are built on the CPU and copied once.

## Fixed in advance

| Item | Value |
|---|---|
| Arms | T4-v1, C4-v1, T4-v2, C4-v2: {T4 specialists, C4_params shared, parameter-matched} x {grounding v1 (the pilot's), v2}. Within a grounding the arms differ only in the semantic module; within an architecture only in grounding |
| Seeds | 100, 101, 102. Under one seed every arm shares the trunk initialization (v2's added parts are drawn from a forked generator) and the example order |
| Training set | `pins/perf_v2_train.json`: the pilot's 4,741 supervisable questions minus the one not representable under v2 (`bird23tf-00003`, whose gold ON holds `T2.list_id = T2.list_id` beside its real key): 4,740, identical for all arms |
| States | the pilot's anchored cache entries, verified at preparation; nothing re-encoded |
| Recipe | the pilot's: AdamW lr 3e-4, weight decay 0.01, clip 1.0, effective batch 32 in micro-batches of 8 grouped by length, loss weights 1/1/1, TF32 matmuls in the controller (the backbone always at full precision) |
| Updates | **U = 3,000 per fit** (twice the pilot's), unless the feasibility rule below reduces it for every fit alike |
| Checkpoints | every 500 updates for recovery and diagnostics. **Only the final checkpoint (update U-1) is evaluated**; `perf-eval` refuses anything else. No checkpoint is chosen on any development question |
| Evaluation populations | `pins/perf_v2_eval.json`: **historical**, inventory positions 0-63 (the pilot's 64, which the diagnosis read); **new**, positions 64-127, never read by any arm or analysis before this freeze |
| Evaluation | greedy legal decoding, at most 256 steps, batch size one, a fresh timed backbone forward per request, then a reconstruction on the anchored cache; the pilot's integrity gates (fresh states equal the anchors and are identical across the four arms of a seed; the reconstruction equals the timed decode). Exactly one passing evaluation per seed counts |
| Scoring | the pinned EX scorer; every request in the denominator |
| Baselines | L (CPU) on both populations. B4 (Qwen3.5-4B, thinking) and Q27 (Qwen3.8-27B, thinking, effort low) in their pinned development configurations, unchanged: on positions 16-63 to complete the historical 64, and on the new 64 if the budget rule below allows |

## Feasibility and the paired smoke (before any fit)

`perf-feasibility` trains each of the four arms for 40 updates on a 160-question training-pool sample (measurement
seed 999), saves and reloads checkpoints, and decodes and scores 8 training-pool requests per arm (fresh forward and
reconstruction). It checks finite gradients, a falling loss, equal reloaded digests and equal reconstructions. Its
models are discarded and stand for no arm.

**The update rule.** With the measured per-arm costs, U = 3,000 if training and evaluating all twelve fits fits in
the hours left minus 4.6 h of contingency. Otherwise U is the largest multiple of 500 that fits, the same for every
fit, recorded in an amendment before training. Below 1,500 the study stops and reports.

## Order

1. `perf-encode` (the new 64, after recomputing 8 historical contexts against their pilot anchors).
2. `perf-feasibility`.
3. `perf-train --seed 100`, then 101, then 102 (four reservations per seed, arms in the order above).
4. `perf-eval --seed 100`, 101, 102.
5. `baseline --arm L --positions 64:128`.
6. B4, then Q27, on positions 64:128 if the remaining hours minus 1 h cover their projection from the measured
   16-63 runs.
7. `perf-analyze`, the independent verification, the report.

**If the budget stops training,** only seeds with all four arms complete enter the comparisons (a symmetric
reduction), and the report says so before interpreting anything. An arm is never trained longer than its
counterparts, and no arm is shortened alone. With fewer than three seeds the threshold below keeps its share
(10 x seeds / 3, rounded up).

**The code is frozen with this protocol.** Every fit and evaluation runs from the run worktree at the freeze commit,
so a resumed fit matches its recorded source hashes.

## Outcomes and how they are read

Write c(arm, s) for an arm's EX-correct count on a population of 64 for seed s. Differences are summed over the
three seeds: 192 question-seed pairs. **The new population is primary** for every rule; the historical one is
reported beside it and is biased toward the intervention, which the diagnosis chose on it.

**Q1, the intervention.** For each architecture a, D_a = sum_s [c(a-v2, s) - c(a-v1, s)].
- **I1, material improvement:** D_a >= +10 (5.2 points), positive in each of the three seeds, and still positive when
  any one database's questions are removed.
- **I2, material harm:** the mirror image.
- **I3, no material change:** otherwise.

**Q2, the architectures.** D = sum_s [c(T4-v2, s) - c(C4-v2, s)], with the same three conditions.
- **A1:** T4-v2 ahead.
- **A2:** C4-v2 ahead.
- **A3:** no consistent difference.

The same quantities are reported for v1 and for the historical population, descriptively.

**Reading:**
- A1 motivates a separately frozen confirmation on fresh data. It does not establish one.
- I1 for both architectures with A3: an engineering gain, not attributed to specialization.
- I3 for both: diagnose why, and recommend a distinct next experiment, not more of the same.

**Exploratory** (never a gate): a sign-flip test over database blocks for each D, enumerating every assignment of
signs to the nonzero blocks, two-sided, with the smallest attainable p. It conditions on the fitted models.

**Also reported:**
- every question, seed and arm, and every abstention, timeout and executed SQL;
- per database and per semantic stratum of the gold program (tables joined, aggregation, grouping, nesting, ordering,
  literals);
- time to SQL submission, submitted-query execution, and both together, per request and per EX-correct request;
- training GPU-hours per fit, peak memory, parameters, active MACs per request, and cache cost;
- the pilot's 1,500-update T4 and C4_params results, separately.

## Limits

This is a development comparison with three seeds. The new population is fresh to this study but comes from the
same development databases that informed the grammar's design, so it is not a confirmation. Nothing here is a
held-out or reserved-benchmark result.
