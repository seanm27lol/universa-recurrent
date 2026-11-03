# Performance study v2: relational grounding, three seeds (results; development)

This study ran the frozen protocol [`PROTOCOL_perf_v2.md`](PROTOCOL_perf_v2.md): sha256 `ba696dfd…`, frozen at
01e3e28, and run from c05e08f. It trained twelve controllers on the BIRD training release:
- {T4 specialists, C4_params shared controller} × {grounding v1 (the pilot's), v2 (relational grounding)};
- seeds 100, 101 and 102;
- 3,000 updates each.

It evaluated each fit's final checkpoint on two development populations of 64 questions each:
- **historical:** inventory positions 0-63, the pilot's questions, which the diagnosis read;
- **new:** positions 64-127, fixed before training and primary for every rule.

The intervention was chosen by the CPU-only [diagnosis](REPORT_perf_v2_diagnosis.md) before any training.

**An example.** One question from the new population (bird23tf-03980, database `talkingdata`, from the BIRD
benchmark of Li et al., NeurIPS 2023):
- **The question:** "Identify by their id all the apps that belong to the game-stress reliever category."
- **The gold program** joins the category table to the app-label table on the shared key:
  `SELECT T2.app_id FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T2.label_id
  WHERE T1.category = 'game-stress reliever'`.
- **T4 with the pilot's grounding (v1, seed 101)** picked an unrelated first table and joined the category table to
  it with `ON T2.label_id = T2.label_id`: a column compared with itself. That predicate relates nothing, so every
  app is paired with every matching category row: 113,211 rows against the gold program's 233.
- **T4 with relational grounding (v2, same seed)** produced the gold program exactly.

**What v2 changes.** Each table and column pointer now sees the schema's relations:
- which tables are linked by keys, and which columns belong to a table;
- which items the question names;
- what the partial program has already chosen.

The second join operand is scored against the first, and comparing one column reference with itself in an ON
condition is not allowed. The encoder is the relation-aware attention of RAT-SQL (Wang et al., ACL 2020). That
builds on Shaw et al. (NAACL 2018), here with a scalar bias per relation and head. The full design is in
`grounding.py` and the protocol.

This is a development comparison. The new population is fresh to this study, but its databases informed the
grammar's design, so it is not a confirmation. The 500-question reserved BIRD Mini-Dev benchmark was not read.
Nothing was merged.

The report keeps three kinds of result apart:
- **software correctness:** does the code do what it says?
- **development observations:** what happened on these 128 questions?
- **architectural claims:** what the frozen rules allow us to say.

## The four answers

**1. Did the intervention materially improve real semantic program construction? Yes, by the frozen rule, for both
architectures, though from a low base.** On the new population (192 question-seed pairs per arm):
- T4 rose from 21 to 36 EX-correct (+15; per seed +4, +9, +2);
- C4_params rose from 17 to 35 (+18; per seed +8, +4, +6).

Both meet all three conditions of I1:
- at least +10;
- positive in every seed;
- still positive when any one database is removed (the smallest remainders are +11 and +14).

The historical population shows the same: +16 and +18.

The gain is in relational construction:
- the same-reference joins and the scorer timeouts they caused fell to zero. This one is by construction: v2
  includes a hard legality restriction against them (below);
- the second join operand's teacher-forced error fell from 0.25 to 0.06-0.10;
- the joined-table error fell from 0.58 to 0.43-0.46;
- the predicted table set is right in 82 and 81 of 192 requests, against 57 and 54.

Three qualifications:
- **Capacity:** the v2 trunk has 4.2 M more parameters, so the gain cannot be split between the relations and the
  added capacity.
- **Suspicious programs:** about a quarter of v2's EX-correct answers have programs that differ from gold in ways EX
  does not see: 9 of 36 and 9 of 35, against 5 of 21 and 4 of 17 for v1. Without them, the gains are +11 and +13.
- **The hard restriction:** v2 forbids an ON comparison of a column reference with itself, beside its learned
  relational parts. The comparison tests them together, so the restriction's own contribution to the gain is not
  isolated here.
- **Absolute level:** it stays low, at 18-19%.

**2. Did T4 outperform the parameter-matched shared architecture consistently? No.**
- **Under v2:** T4-v2 − C4-v2 = +1 of 192 (per seed −2, +5, −2), which is A3: no consistent difference. The
  historical population gives +1 as well.
- **Under v1:** the difference is +4, read descriptively.
- **The pilot's asymmetry did not recur.** In the pilot, T4 cleared L in both seeds and C4_params did not. At 3,000
  updates both v1 arms are above L (21 and 17 against 12 of 192).

The protocol's reading is therefore an **engineering gain of the intervention, not attributable to
specialization**. Specialization showed no advantage under either grounding.

**3. How far is T4 from B4 and Qwen3.8-27B? Far behind on quality, and much cheaper per answer.** The Qwen
baselines ran in their pinned development configurations, with the same schemas, retrieval, tools, limits and
scorer:

| EX-correct | historical 64 | new 64 |
|---|---|---|
| T4-v2 (3 seeds) | 34 of 192 (17.7%) | 36 of 192 (18.8%) |
| C4-v2 (3 seeds) | 33 of 192 (17.2%) | 35 of 192 (18.2%) |
| B4, Qwen3.5-4B thinking | 40 of 64 (62.5%) | 41 of 64 (64.1%) |
| Q27, Qwen3.8-27B thinking, effort low | 49 of 64 (76.6%) | 50 of 64 (78.1%) |
| L, no-fit baseline | 4 of 64 | 4 of 64 |

T4-v2 answers about a quarter as many questions as Q27, and under a third as many as B4. B4's abstentions were 13
and 11, almost all from exhausting its tool-call budget. Q27 never abstained.

Cost per request runs the other way. Three kinds of cost are kept apart below and never added together.

**Marginal serving** is per request, measured in the records. For the controllers, time to submission is the fresh
4B forward plus decoding.

| serving | time to SQL submission per request (historical, new) | generation + submitted-query execution per EX-correct answer | peak memory |
|---|---|---|---|
| T4-v2 | 1.32 s, 1.20 s | 7.7 s, 6.6 s | 12.3 GB (4B backbone included) |
| B4 | 99.8 s, 89.1 s | 159.7 s, 139.2 s | 12.6 GB |
| Q27 | 142.0 s, 139.9 s | 185.6 s, 179.2 s | 58.8 GB |

So per correct answer, T4-v2 is about 20-27 times cheaper in marginal serving time, and it answers far fewer
questions. Submitted-query execution is small for every arm except v1, whose same-reference joins timed out.

**Standalone setup** is one-time, in GPU-reservation hours. Before a controller serves anything it needs:
- the training pool's backbone states, 1.936 h (the pilot's completed encode reservations, reused by every fit);
- one fit: 0.48-0.56 h, and 0.558 h for T4-v2.

The Qwen baselines need no setup beyond loading the model, which is inside their serving reservations. The
pretraining of the backbone and of the Qwen models is counted for neither side.

A setup of S hours spread over n requests adds S / n to each. For T4-v2, at n = 64 that is about 140 s per request:
- about Q27's marginal time per request;
- more than B4's.

So the controllers are cheaper only when one setup serves many requests.

**Research expenditure** is the cost of finding out, not a cost of serving. This study reserved 14.538 GPU-hours
under authorization v2: twelve fits, three evaluations, the smokes and four Qwen runs. The whole ledger, with the
pilot and earlier work, is 21.416 h.

**Timing conditions.** The Qwen times are as recorded, but some of them overlapped CPU work by the producer and
reviewer sessions on the other cores:
- B4 on positions 16-63: about 31 of 81 minutes, mostly the pilot's diagnosis. The load average exceeded 6 in 44% of
  samples.
- B4 on the new 64: about 27 of 96 minutes.
- Q27 on 16-63: briefly.
- Q27 on the new 64: none.

Core pinning does not separate memory bandwidth on this unified-memory machine. No correction is applied; the
disclosure is in `evidence/perf_v2/corrections/cpu_overlap.json` and `perf_report --corrections`.

**4. What capability remains the limiting factor? Grounding the question in an unseen schema: which tables the
question needs, and the literal values and aggregates it names.**

*Corrected on 2026-11-03.* The first version of this answer named the anchor (first FROM) table as the chief
failure, citing a teacher-forced error of 0.45-0.54. That figure mixed two different things:
- **another table of the gold program, chosen first:** 25-32% of these decisions. This is mostly a
  reference-order disagreement that an inner join may not care about, but the count also includes gold tables used
  only in a nested SELECT;
- **a table the gold program does not use:** 16-22%.

On the new population, under teacher forcing (the gold prefix given):

| decision | v1 error | v2 error |
|---|---|---|
| FROM table: a table the gold program does not use (pointer) | 0.21-0.22 | 0.16-0.22 |
| FROM table: another table of the gold program first (pointer) | 0.25-0.27 | 0.29-0.32 |
| joined table | 0.58 | 0.43-0.46 |
| WHERE literal | 0.45-0.50 | 0.44-0.49 (unchanged) |
| SELECT aggregate | 0.43-0.46 | 0.41-0.44 |
| SELECT column | 0.39-0.42 | 0.33-0.34 |
| WHERE column | 0.36-0.39 | 0.29-0.30 |

In free decoding:
- the predicted table set is still wrong in 57% of v2 requests: tables missing or extra, not merely reordered;
- even when the table set is right, only 30 of 81-82 answers are EX-correct;
- the first divergence from the gold traversal is at the first FROM table in 77-95 of 192 requests for every arm,
  a count that includes choosing another gold table first.

Relational grounding fixed what it could see: relations among schema items. It did not fix reading the question
against a new schema: which tables the question needs, and which values it names. The pilot's fits erred on the
first table in 17% of training questions, so the gap is generalization, not fitting.

**Next.** A distinct experiment, not more of this one, follows under "Recommended next experiment". A T4
confirmation is not proposed, because there was no T4 advantage to confirm.

## What was run

| step | what | GPU-hours |
|---|---|---|
| B4/Q27 completion | positions 16-63 in the pinned configurations: B4 29/48, Q27 36/48 | 1.345 + 1.983 |
| perf-encode | the new 64 contexts encoded once and anchored; 8 historical contexts recomputed against their pilot anchors, all equal | 0.044 |
| perf-feasibility | the measured projection (U = 3,000 holds: 5.73 h needed of 17.09 available after the 4.6 h contingency) and the paired smoke (four arms, 40 updates; finite gradients, falling loss, equal reloaded digests, equal reconstructions; models discarded) | 0.059 |
| perf-train | 12 fits, one attempt each, no failure or resume | 6.250 |
| perf-eval | 3 evaluations (4 arms × 128 questions each), all integrity gates passed | 0.645 |
| L on the new 64 | CPU | 0 |
| B4, Q27 on the new 64 | budget rule passed before each (remaining − 1 h ≥ projection from the 16-63 runs); B4 41/64, Q27 50/64 | 1.608 + 2.604 |

## Software correctness

- **Tests:** 340 passed and 8 skipped (they need the downloaded data) in the scaling suite at ad20692, CPU. Added for
  this study:
  - ledger v2: 6 tests. Two worktrees share one ledger and one 32 h cap, and one GPU owner; a private runs
    directory is refused; the history is preserved.
  - relational grounding: 34 tests. They cover:
    - the pilot's problematic joins;
    - legitimate self-joins and non-key joins kept;
    - 6,187 of 6,188 historical gold programs encoding to v1's actions;
    - selected-only specialist execution;
    - identical supervision and order across arms;
    - parameter and MAC accounting.
  - the performance run end to end on synthetic data: 4 tests.
  - the decision rules: 5 tests.
  - the post-evaluation diagnosis tables: 1 test.
  - the perf bundle: 2 tests. The report rebuilds from the bundle alone, and a changed score is refused even
    when the sums are rewritten.
- **Determinism:** under one seed, every arm shares the trunk initialization and the example order. v2's added
  modules are drawn from a forked generator. The verifier recomputes the sampling stream independently.
- **Evaluation integrity, all three seeds:**
  - the fresh states equal their anchors (the pilot's for the historical 64, perf-encode's for the new 64);
  - they are identical across the four arms;
  - the reconstruction equals the timed decode;
  - only the final checkpoint (update 2,999) was evaluated, once per seed.
- **Before the freeze:** an independent review found no defect invalidating training or the comparison. Its findings
  were fixed before the freeze (listed in the protocol).
- **Execution errors:** no request abstained, and no plan was invalid. Seven submitted queries failed in SQLite on
  the new population, under both groundings:
  - six "misuse of aggregate": an outer column inside a subquery's aggregate, which the grammar allows;
  - one integer overflow.

## Development observations

`python -m scaling.perf_report` prints these from the bundle alone. It first recomputes every count, rule,
leave-one-database-out value and serving total from the records, and stops on any disagreement with
`analysis.json`. Every question, with its timeouts, abstentions and tool calls, is in the
[appendix](REPORT_perf_v2_appendix.md).

Two patterns, descriptive only:
- **Where the gain is spread.** The v2 gain is spread over databases (the smallest leave-one-database-out remainder
  is +11). Most of it on the new population is in questions without aggregation (v1 2 and 1, v2 13 and 12) and with
  two tables (v1 10 and 6, v2 21 and 23).
- **Where nothing improved.** No arm solved any question that joins three or more tables: 6 on the new population
  and 4 on the historical. On the new population, no arm solved any of the 5 questions with nesting either.

Recomputed from the bundled records and equal to analysis.json (seeds [100, 101, 102]).

### Historical population (64 questions)

| arm | seed 100 | seed 101 | seed 102 | pooled |
|---|---|---|---|---|
| T4-v1 | 8 | 3 | 7 | 18 of 192 |
| C4-v1 | 4 | 4 | 7 | 15 of 192 |
| T4-v2 | 9 | 11 | 14 | 34 of 192 |
| C4-v2 | 11 | 12 | 10 | 33 of 192 |
| L | — | — | — | 4 of 64 (one run) |
| B4 | — | — | — | 40 of 64 (one run) |
| Q27 | — | — | — | 49 of 64 (one run) |

| comparison | seed 100 | seed 101 | seed 102 | pooled | leaving one database out (min .. max) | rule | sign-flip p (exploratory) |
|---|---|---|---|---|---|---|---|
| T4: v2 - v1 | +1 | +8 | +7 | +16 | +12 .. +17 | I1: material improvement | 0.012 |
| C4: v2 - v1 | +7 | +8 | +3 | +18 | +13 .. +19 | I1: material improvement | 0.020 |
| v2: T4 - C4 | -2 | -1 | +4 | +1 | -1 .. +4 | A3: no consistent difference | 1.000 |
| v1: T4 - C4 (descriptive) | +4 | -1 | +0 | +3 | +2 .. +3 | A3: no consistent difference | 0.250 |

Reading: both architectures improved materially and they do not differ consistently: an engineering gain of the intervention, not attributable to specialization

| arm (pooled over seeds) | requests | EX-correct | time to submission, mean s | submitted execution, s (timeouts) | generation per EX-correct, s | generation + execution per EX-correct, s | controller MACs per request, as recorded |
|---|---|---|---|---|---|---|---|
| T4-v1 | 192 | 18 | 1.33 | 35.4 (1) | 14.2 | 16.2 | 7.26 G |
| C4-v1 | 192 | 15 | 1.31 | 112.6 (3) | 16.8 | 24.3 | 7.36 G |
| T4-v2 | 192 | 34 | 1.32 | 9.4 (0) | 7.5 | 7.7 | 7.67 G |
| C4-v2 | 192 | 33 | 1.33 | 11.7 (0) | 7.7 | 8.1 | 7.76 G |
| L | 64 | 4 | 3.00 | 4.5 (0) | 48.1 | 49.2 | — |
| B4 | 64 | 40 | 99.81 | 1.6 (0) | 159.7 | 159.7 | — |
| Q27 | 64 | 49 | 142.03 | 2.4 (0) | 185.5 | 185.6 | — |

| database | questions | T4-v1 | C4-v1 | T4-v2 | C4-v2 | T4 v2-v1 | C4 v2-v1 | v2 T4-C4 |
|---|---|---|---|---|---|---|---|---|
| computer_student | 4 | 3 | 2 | 2 | 1 | -1 | -1 | +1 |
| disney | 4 | 1 | 1 | 2 | 2 | +1 | +1 | +0 |
| food_inspection | 4 | 3 | 3 | 5 | 8 | +2 | +5 | -3 |
| hockey | 5 | 0 | 0 | 0 | 0 | +0 | +0 | +0 |
| ice_hockey_draft | 5 | 0 | 0 | 1 | 0 | +1 | +0 | +1 |
| image_and_language | 5 | 1 | 1 | 2 | 3 | +1 | +2 | -1 |
| music_platform_2 | 4 | 3 | 2 | 3 | 2 | +0 | +0 | +1 |
| professional_basketball | 5 | 0 | 0 | 2 | 3 | +2 | +3 | -1 |
| public_review_platform | 5 | 0 | 0 | 0 | 0 | +0 | +0 | +0 |
| simpson_episodes | 5 | 1 | 1 | 2 | 2 | +1 | +1 | +0 |
| soccer_2016 | 5 | 0 | 0 | 4 | 2 | +4 | +2 | +2 |
| software_company | 4 | 0 | 0 | 0 | 0 | +0 | +0 | +0 |
| talkingdata | 4 | 1 | 1 | 4 | 2 | +3 | +1 | +2 |
| world | 5 | 5 | 4 | 7 | 8 | +2 | +4 | -1 |

| stratum | value | questions | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|---|---|
| aggregation | no | 41 | 11 | 10 | 21 | 20 |
| aggregation | yes | 23 | 7 | 5 | 13 | 13 |
| grouping | no | 58 | 16 | 15 | 28 | 28 |
| grouping | yes | 6 | 2 | 0 | 6 | 5 |
| literals | 0 | 3 | 1 | 1 | 5 | 4 |
| literals | 1 | 25 | 14 | 10 | 23 | 18 |
| literals | 2+ | 36 | 3 | 4 | 6 | 11 |
| nesting | no | 60 | 13 | 11 | 31 | 29 |
| nesting | yes | 4 | 5 | 4 | 3 | 4 |
| ordering | no | 57 | 16 | 14 | 29 | 29 |
| ordering | yes | 7 | 2 | 1 | 5 | 4 |
| tables | 1 | 21 | 9 | 9 | 12 | 13 |
| tables | 2 | 39 | 9 | 6 | 22 | 20 |
| tables | 3+ | 4 | 0 | 0 | 0 | 0 |

### New population (primary) (64 questions)

| arm | seed 100 | seed 101 | seed 102 | pooled |
|---|---|---|---|---|
| T4-v1 | 6 | 7 | 8 | 21 of 192 |
| C4-v1 | 4 | 7 | 6 | 17 of 192 |
| T4-v2 | 10 | 16 | 10 | 36 of 192 |
| C4-v2 | 12 | 11 | 12 | 35 of 192 |
| L | — | — | — | 4 of 64 (one run) |
| B4 | — | — | — | 41 of 64 (one run) |
| Q27 | — | — | — | 50 of 64 (one run) |

| comparison | seed 100 | seed 101 | seed 102 | pooled | leaving one database out (min .. max) | rule | sign-flip p (exploratory) |
|---|---|---|---|---|---|---|---|
| T4: v2 - v1 | +4 | +9 | +2 | +15 | +11 .. +15 | I1: material improvement | 0.008 |
| C4: v2 - v1 | +8 | +4 | +6 | +18 | +14 .. +19 | I1: material improvement | 0.010 |
| v2: T4 - C4 | -2 | +5 | -2 | +1 | +0 .. +2 | A3: no consistent difference | 1.000 |
| v1: T4 - C4 (descriptive) | +2 | +0 | +2 | +4 | +1 .. +5 | A3: no consistent difference | 0.500 |

Reading: both architectures improved materially and they do not differ consistently: an engineering gain of the intervention, not attributable to specialization

| arm (pooled over seeds) | requests | EX-correct | time to submission, mean s | submitted execution, s (timeouts) | generation per EX-correct, s | generation + execution per EX-correct, s | controller MACs per request, as recorded |
|---|---|---|---|---|---|---|---|
| T4-v1 | 192 | 21 | 1.20 | 113.5 (3) | 11.0 | 16.4 | 6.64 G |
| C4-v1 | 192 | 17 | 1.19 | 143.7 (3) | 13.4 | 21.9 | 6.74 G |
| T4-v2 | 192 | 36 | 1.20 | 8.0 (0) | 6.4 | 6.6 | 7.01 G |
| C4-v2 | 192 | 35 | 1.22 | 10.0 (0) | 6.7 | 7.0 | 7.10 G |
| L | 64 | 4 | 0.09 | 3.2 (0) | 1.4 | 2.2 | — |
| B4 | 64 | 41 | 89.13 | 2.6 (0) | 139.1 | 139.2 | — |
| Q27 | 64 | 50 | 139.95 | 2.6 (0) | 179.1 | 179.2 | — |

| database | questions | T4-v1 | C4-v1 | T4-v2 | C4-v2 | T4 v2-v1 | C4 v2-v1 | v2 T4-C4 |
|---|---|---|---|---|---|---|---|---|
| computer_student | 5 | 3 | 3 | 3 | 3 | +0 | +0 | +0 |
| disney | 5 | 0 | 0 | 1 | 1 | +1 | +1 | +0 |
| food_inspection | 5 | 4 | 3 | 4 | 5 | +0 | +2 | -1 |
| hockey | 4 | 0 | 0 | 1 | 0 | +1 | +0 | +1 |
| ice_hockey_draft | 4 | 0 | 0 | 1 | 1 | +1 | +1 | +0 |
| image_and_language | 4 | 0 | 0 | 2 | 3 | +2 | +3 | -1 |
| music_platform_2 | 5 | 5 | 4 | 5 | 5 | +0 | +1 | +0 |
| professional_basketball | 4 | 0 | 0 | 2 | 2 | +2 | +2 | +0 |
| public_review_platform | 5 | 0 | 0 | 0 | 0 | +0 | +0 | +0 |
| simpson_episodes | 4 | 3 | 4 | 3 | 3 | +0 | -1 | +0 |
| soccer_2016 | 5 | 3 | 0 | 4 | 3 | +1 | +3 | +1 |
| software_company | 5 | 0 | 0 | 0 | 0 | +0 | +0 | +0 |
| talkingdata | 5 | 3 | 3 | 6 | 5 | +3 | +2 | +1 |
| world | 4 | 0 | 0 | 4 | 4 | +4 | +4 | +0 |

| stratum | value | questions | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|---|---|
| aggregation | no | 36 | 2 | 1 | 13 | 12 |
| aggregation | yes | 27 | 19 | 16 | 23 | 23 |
| grouping | no | 58 | 21 | 17 | 34 | 34 |
| grouping | yes | 5 | 0 | 0 | 2 | 1 |
| literals | 0 | 7 | 0 | 0 | 2 | 5 |
| literals | 1 | 27 | 16 | 14 | 23 | 17 |
| literals | 2+ | 29 | 5 | 3 | 11 | 13 |
| nesting | no | 58 | 21 | 17 | 36 | 35 |
| nesting | yes | 5 | 0 | 0 | 0 | 0 |
| ordering | no | 53 | 21 | 17 | 33 | 29 |
| ordering | yes | 10 | 0 | 0 | 3 | 6 |
| tables | 1 | 23 | 11 | 11 | 15 | 12 |
| tables | 2 | 34 | 10 | 6 | 21 | 23 |
| tables | 3+ | 6 | 0 | 0 | 0 | 0 |
| tables | unconverted | 1 | 0 | 0 | 0 | 0 |

### Training and cost

| fit | updates | loss first -> last | s per update | reservation h | attempts | peak memory GB | trainable parameters |
|---|---|---|---|---|---|---|---|
| C4-v1/100 | 3000 | 4.484 -> 0.034 | 0.563 | 0.479 | 1 | 8.2 | 17,016,548 |
| C4-v1/101 | 3000 | 4.591 -> 0.056 | 0.566 | 0.481 | 1 | 9.9 | 17,016,548 |
| C4-v1/102 | 3000 | 4.736 -> 0.038 | 0.578 | 0.492 | 1 | 9.8 | 17,016,548 |
| C4-v2/100 | 3000 | 4.482 -> 0.019 | 0.639 | 0.543 | 1 | 8.7 | 21,234,612 |
| C4-v2/101 | 3000 | 4.585 -> 0.067 | 0.634 | 0.540 | 1 | 10.4 | 21,234,612 |
| C4-v2/102 | 3000 | 4.736 -> 0.072 | 0.637 | 0.541 | 1 | 10.3 | 21,234,612 |
| T4-v1/100 | 3000 | 4.422 -> 0.046 | 0.575 | 0.489 | 1 | 8.1 | 17,016,548 |
| T4-v1/101 | 3000 | 4.542 -> 0.067 | 0.580 | 0.493 | 1 | 9.9 | 17,016,548 |
| T4-v1/102 | 3000 | 4.669 -> 0.029 | 0.610 | 0.517 | 1 | 9.8 | 17,016,548 |
| T4-v2/100 | 3000 | 4.419 -> 0.068 | 0.652 | 0.555 | 1 | 8.7 | 21,234,612 |
| T4-v2/101 | 3000 | 4.535 -> 0.034 | 0.657 | 0.559 | 1 | 10.4 | 21,234,612 |
| T4-v2/102 | 3000 | 4.674 -> 0.047 | 0.659 | 0.561 | 1 | 10.3 | 21,234,612 |

The pilot's fits (1,500 updates, seeds 100-101) on the historical 64: C4_params/100 4, C4_params/101 10, T4/100 9, T4/101 9.

Ledger: 21.416 of 32.0 h used.

### The pilot's 1,500-update result, beside the v1 arms at 3,000 (descriptive)

The v1 arms at 3,000 updates are this study's unchanged controls, at a budget matched to the v2 arms; the answers
above use them.

Separately, the pilot's fits had the same grounding at 1,500 updates (seeds 100 and 101, historical 64):
- T4 9 and 9;
- C4_params 4 and 10.

The same seeds at 3,000 updates here scored:
- T4-v1 8 and 3;
- C4-v1 4 and 4.

**This does not isolate the number of updates.** The two sets of runs have different training streams from the
first update: their training sets differ by one question (bird23tf-00003, dropped here), so the sampled order
differs throughout. Two seeds are also few.

The longer runs did fit their training data more closely. The mean training loss over their last 100 updates is
0.05-0.06, against the pilot's 0.19-0.21 at its end; over updates 1,400-1,499 they were at 0.19-0.20, like the pilot.
Yet they did not score higher on the historical 64. That is all the comparison shows; it does not show that longer
training lowers accuracy.

## Diagnosis after evaluation (CPU; every fit under its own grounding)

`python -m scaling.diagnose perf-tables --out <bundle>/diagnosis/<stamp>_post_<commit>` prints these from
`perf_diagnosis.json`. Each arm is pooled over its three seeds.

The historical population's tables are in the [appendix](REPORT_perf_v2_appendix.md); they show the same pattern.

### New population (192 requests per arm: 3 seeds)

| free decoding | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|
| correct | 21 | 17 | 36 | 35 |
| table set equal | 57 | 54 | 82 | 81 |
| every gold table predicted | 90 | 87 | 119 | 117 |
| every predicted table in gold | 103 | 110 | 110 | 108 |
| join keys equal given tables equal | 52 | 47 | 75 | 74 |
| correct given table set equal | 19 | 15 | 30 | 30 |
| correct given table set differs | 2 | 2 | 6 | 5 |
| exact plan | 4 | 4 | 9 | 2 |
| requests with same reference join | 33 | 23 | 0 | 0 |
| timeouts | 3 | 3 | 0 | 0 |
| EX-correct with a suspicious program | 5 | 4 | 9 | 9 |

| error category (requests; a request can have several) | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|
| projection | 146 | 148 | 136 | 129 |
| table grounding | 132 | 135 | 107 | 108 |
| predicate column | 118 | 113 | 102 | 102 |
| join key (different tables) | 98 | 106 | 94 | 88 |
| aggregation | 43 | 53 | 49 | 48 |
| ordering or LIMIT | 35 | 36 | 35 | 32 |
| DISTINCT | 28 | 29 | 30 | 31 |
| join: undeclared, non-gold key | 29 | 32 | 26 | 25 |
| grouping or HAVING | 25 | 32 | 27 | 25 |
| predicate literal | 21 | 24 | 29 | 25 |
| nested query involved | 22 | 23 | 24 | 24 |
| join: same-reference equality | 33 | 23 | 0 | 0 |
| join key (same tables) | 5 | 7 | 7 | 7 |
| no comparable plan | 3 | 3 | 3 | 3 |
| execution error | 2 | 0 | 4 | 1 |
| execution timeout | 3 | 3 | 0 | 0 |
| predicate operator or shape | 0 | 2 | 1 | 1 |
| predicate: same-reference equality | 2 | 0 | 0 | 0 |

| first divergence from the gold traversal | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|
| FROM table | 81 | 79 | 77 | 95 |
| join count or kind | 36 | 42 | 37 | 29 |
| joined table | 22 | 19 | 11 | 8 |
| ON column (operand 0) | 14 | 13 | 12 | 17 |
| gold not representable | 12 | 12 | 12 | 12 |
| WHERE predicate structure | 4 | 2 | 14 | 6 |
| WHERE column | 3 | 6 | 6 | 7 |
| exact | 4 | 4 | 9 | 2 |
| WHERE literal | 6 | 3 | 5 | 3 |
| SELECT column | 3 | 4 | 3 | 3 |
| FROM structure | 3 | 2 | 1 | 2 |
| SELECT aggregate | 1 | 2 | 1 | 4 |
| GROUP BY | 1 | 0 | 1 | 1 |
| LIMIT | 2 | 1 | 0 | 0 |
| ON column (operand 1) | 0 | 1 | 2 | 0 |
| DISTINCT | 0 | 0 | 1 | 1 |
| SELECT structure | 0 | 1 | 0 | 1 |
| query structure | 0 | 1 | 0 | 1 |

| teacher-forced decision (error rate) | steps per arm | T4-v1 | C4-v1 | T4-v2 | C4-v2 |
|---|---|---|---|---|---|
| SELECT other | 609 | 0.12 | 0.12 | 0.12 | 0.11 |
| WHERE operator | 432 | 0.26 | 0.31 | 0.32 | 0.30 |
| SELECT column | 249 | 0.42 | 0.39 | 0.33 | 0.34 |
| WHERE column | 246 | 0.36 | 0.39 | 0.29 | 0.30 |
| LIMIT | 240 | 0.03 | 0.03 | 0.03 | 0.03 |
| WHERE literal | 240 | 0.45 | 0.50 | 0.44 | 0.49 |
| HAVING operator | 204 | 0.00 | 0.00 | 0.00 | 0.00 |
| DISTINCT | 201 | 0.13 | 0.12 | 0.12 | 0.12 |
| GROUP BY other | 201 | 0.04 | 0.08 | 0.03 | 0.03 |
| QUERY | 201 | 0.00 | 0.00 | 0.00 | 0.00 |
| join: no more tables | 198 | 0.22 | 0.20 | 0.18 | 0.18 |
| FROM table | 192 | 0.48 | 0.47 | 0.45 | 0.54 |
| STOP | 180 | 0.00 | 0.00 | 0.00 | 0.00 |
| ON column (operand 0) | 147 | 0.44 | 0.44 | 0.39 | 0.37 |
| ON column (operand 1) | 147 | 0.25 | 0.25 | 0.06 | 0.10 |
| ON operator | 147 | 0.00 | 0.00 | 0.00 | 0.00 |
| join: add table | 147 | 0.58 | 0.58 | 0.43 | 0.46 |
| SELECT aggregate | 87 | 0.46 | 0.43 | 0.41 | 0.44 |
| ORDER BY column | 39 | 0.31 | 0.23 | 0.31 | 0.36 |
| GROUP BY column | 24 | 0.62 | 0.62 | 0.75 | 0.71 |
| ORDER BY other | 21 | 0.33 | 0.29 | 0.24 | 0.29 |
| FROM structure | 9 | 1.00 | 1.00 | 0.89 | 0.89 |
| HAVING column | 3 | 0.67 | 0.67 | 0.67 | 1.00 |
| HAVING literal | 3 | 0.33 | 1.00 | 0.00 | 0.67 |
| mean step accuracy (representable requests) | 180 | 0.786 | 0.782 | 0.809 | 0.801 |

## Architectural claims (the frozen rules, new population primary)

| comparison | per seed (100, 101, 102) | pooled | smallest and largest leaving one database out | rule |
|---|---|---|---|---|
| T4: v2 − v1 | +4, +9, +2 | +15 | +11 .. +15 | **I1** material improvement |
| C4_params: v2 − v1 | +8, +4, +6 | +18 | +14 .. +19 | **I1** material improvement |
| v2: T4 − C4_params | −2, +5, −2 | +1 | +0 .. +2 | **A3** no consistent difference |

Reading: both architectures improved materially, and they do not differ consistently. This is an engineering gain
of the intervention, not attributable to specialization.

The exploratory database-block sign-flip p-values are 0.008 (T4), 0.010 (C4_params) and 1.000 (T4 − C4_params).
They condition on the fitted models and are never a gate.

## Resource account

Every GPU reservation is in the bundled `gpu_ledger.jsonl`, charged as reserved: load, failure and scoring
inside a reservation included.

| item | allocation (h) | used (h) | reservations |
|---|---|---|---|
| before this authorization (the pilot and earlier; cap 8 h, authorization v1) | — | 6.878 | 31 |
| B4/Q27 completion, positions 16-63 | 4.0 | 3.329 | 2 |
| profiling and the paired smoke (perf-encode, perf-feasibility) | 1.5 | 0.103 | 2 |
| matched three-seed training | 9.0 | 6.250 | 12 |
| development evaluation and the further Qwen comparisons | 6.0 | 0.645 + 4.212 | 3 + 2 |
| crash recovery and contingency | 4.6 | 0 | 0 |
| **total** | | **21.416 of 32** | 52 |

- Every reservation under authorization v2 ended `ok`: none failed, was terminated or was resumed.
- 10.584 h remain unspent. Per the authorization, they were not spent on extra seeds, larger models or
  speculative changes.
- Training ran 21% over its projection: 6.250 h against 5.163 h, or 0.48-0.56 h per fit against 0.39-0.47 h. The
  states (107 GiB) do not fit in the memory left for the page cache, so updates waited on disk reads.
- Q27 on the new 64 used 2.604 h of its 2.65 h projection (the watchdog limit was 1.5 times that).
- **The state cache:** 4,869 backbone encodings (107 GiB), built during the pilot and reused without
  re-encoding; perf-encode added the new 64 for 0.044 h.

## Corrections of 2026-11-03 (reporting only)

A review of the first version asked for seven corrections. None of them reruns a fit or changes a frozen result. No
existing evidence byte changed: three files were added to the bundle and `SHA256SUMS` rewritten.

1. **Controller MACs.** The recorded "active MACs per request" counted each operator step's chosen pointer kind only,
   and no STOP step. The decoder executes more:
   - every STOP step runs the trunk;
   - every operator step scores all pointer heads;
   - a span operator makes a second, end-pointer call.

   Re-derived from the records, executed work is 0.6% above the recorded values for every arm. Both counts are the
   controller's alone, and the frozen 4B forward over the context is in neither. The recorded values are kept; the
   column is relabelled "as recorded".
2. **CPU work during Qwen timing.** The Qwen runs were pinned to other cores, but producer and reviewer CPU jobs ran
   during parts of three of the four Qwen reservations. The scope is disclosed below and in
   `corrections/cpu_overlap.json`; no correction is applied to any time.
3. **Costs.** Marginal serving, standalone setup and research expenditure are now kept apart (answer 3 and the table
   below). The earlier per-answer figures that mixed a fit's training into 64 questions are withdrawn. The GPU ledger
   is unchanged.
4. **The first FROM table.** A CPU pass over the final checkpoints (`diagnose first-table`, new file
   `diagnosis/*_first_table_*/first_table.json`) separates choosing another table of the gold program first from
   choosing a table the gold program does not use. Answer 4 is rewritten accordingly.
5. **The hard restriction.** The same-reference ON restriction is part of the v2 intervention. Its own contribution is
   not isolated: the elimination of same-reference joins is by construction, and its share of the EX gain is not
   measured.
6. **The pilot comparison.** The pilot's 1,500-update fits and this study's 3,000-update v1 fits differ in their
   training streams from the first update. Their comparison is narrowed to what it shows.
7. **The verifier after relocation.** The reviewer made `verify_perf.py` independent of this machine's paths and
   added a public, bundle-only mode: 445 checks from a fresh clone, 0 failures (see "Independent verification").

`python -m scaling.perf_report --corrections` prints the following from the bundle alone:

#### Controller MACs per request: as recorded and as executed

Controller only; the frozen 4B backbone's forward over the context is not included in either column. As recorded: the chosen operator's pointer kind only, and no STOP step. As executed: every STOP step's trunk, and every pointer head `decode_batch` runs (`perf_corrections.executed_macs`).

| population | arm | requests | as recorded, mean | as executed, mean | difference |
|---|---|---|---|---|---|
| historical | T4-v1 | 192 | 7.259 G | 7.306 G | +0.64% |
| historical | C4-v1 | 192 | 7.360 G | 7.408 G | +0.66% |
| historical | T4-v2 | 192 | 7.672 G | 7.720 G | +0.62% |
| historical | C4-v2 | 192 | 7.758 G | 7.806 G | +0.61% |
| new | T4-v1 | 192 | 6.640 G | 6.681 G | +0.63% |
| new | C4-v1 | 192 | 6.737 G | 6.781 G | +0.65% |
| new | T4-v2 | 192 | 7.015 G | 7.057 G | +0.60% |
| new | C4-v2 | 192 | 7.102 G | 7.144 G | +0.59% |

#### Machine load while Qwen was timed (disclosure; no correction applied)

| reservation | minutes | load samples | 1-min load mean | median | max | share above 6 | overlapping producer or reviewer CPU work |
|---|---|---|---|---|---|---|---|
| B4 thinking, positions 16-63 | 81 | 161 | 6.06 | 5.13 | 28.42 | 0.44 | about 31 of 81 minutes had a producer or reviewer job running (mostly the 18-minute diagnosis); the load average exceeded 6 in 44% of samples (max 28.4) |
| Q27 thinking (low), positions 16-63 | 119 | 238 | 1.74 | 1.47 | 6.66 | 0.00 | about 8 of 119 minutes; load average above 6 in no sample (max 6.7) |
| B4 thinking, positions 64-127 | 96 | 193 | 2.04 | 1.12 | 8.20 | 0.06 | about 27 of 96 minutes (09:28:01-09:54:37); load average above 6 in 6% of samples (max 8.2) |
| Q27 thinking (low), positions 64-127 | 156 | 312 | 1.12 | 1.08 | 1.89 | 0.00 | none; load average at most 1.9 |

#### Teacher-forced FROM-table decisions: wrong order or irrelevant table

The pointer only (a wrong family or operator at the step is not counted here). In gold set: a table the gold program uses, chosen in another order; irrelevant: a table the gold program does not use.

| population | arm | FROM-table steps | right | wrong, in gold set | wrong, irrelevant | first FROM of the program: right / in gold set / irrelevant |
|---|---|---|---|---|---|---|
| historical | T4-v1 | 186 | 0.53 | 0.28 | 0.19 | 0.52 / 0.28 / 0.20 |
| historical | C4-v1 | 186 | 0.56 | 0.25 | 0.18 | 0.57 / 0.24 / 0.20 |
| historical | T4-v2 | 186 | 0.51 | 0.31 | 0.19 | 0.49 / 0.32 / 0.19 |
| historical | C4-v2 | 186 | 0.54 | 0.27 | 0.19 | 0.54 / 0.27 / 0.19 |
| new | T4-v1 | 192 | 0.52 | 0.27 | 0.21 | 0.53 / 0.26 / 0.21 |
| new | C4-v1 | 192 | 0.53 | 0.25 | 0.22 | 0.54 / 0.23 / 0.22 |
| new | T4-v2 | 192 | 0.56 | 0.29 | 0.16 | 0.56 / 0.28 / 0.16 |
| new | C4-v2 | 192 | 0.46 | 0.32 | 0.22 | 0.46 / 0.31 / 0.23 |

#### Costs, kept apart (GPU-reservation hours; never added together)

| kind | what | hours |
|---|---|---|
| standalone setup | the training pool's backbone states, completed encode reservations (4) | 1.936 |
| standalone setup | one C4-v1 fit, mean of 3 | 0.484 |
| standalone setup | one C4-v2 fit, mean of 3 | 0.541 |
| standalone setup | one T4-v1 fit, mean of 3 | 0.500 |
| standalone setup | one T4-v2 fit, mean of 3 | 0.558 |
| research | encode attempts that did not complete (the pilot's: one cut off by the host crash, one failed) | 0.590 |
| research | everything under authorization v2 (21 reservations) | 14.538 |
| research | the whole ledger, the pilot and earlier included (52 reservations) | 21.416 |

Marginal serving is per request and per EX-correct answer, measured in the records (the serving tables above); it is not a ledger quantity.

## Independent verification

A separate reviewer session wrote `independent/verify_perf.py` and ran it. It uses its own digests, sampling-stream
code, SQL parsing, EX rule and decision rules. It imports no producer module, with one exception: the gold-program
strata classification, run in a separate process from the frozen worktree.

**After the correction pass:** sha256 `f7998d7f…`, at 4c73eec. The verifier no longer depends on this machine's
paths, and it has two modes.

| mode | what it needs | checks | failures |
|---|---|---|---|
| public (`--public`) | a checkout alone: the pins, git history and the bundles `perf_v2`, `dev_v1`, `pilot_v1`; run from a fresh `git clone --no-local` with no runs root | 445 | 0 |
| local | the runs root: every SQL re-executed, checkpoints and databases re-hashed | 835 | 0 |

The public mode re-derives:
- the bundles against their sums (every file listed at 0f2f9b1 unchanged), and the ledger arithmetic;
- the training data and the sampling stream, with the checkpoint records;
- eligibility, anchors and the records' consistency;
- 2,998 leaves of `analysis.json` (0 differ; the strata only in their marginal sums);
- the executed controller MACs, from its own reading of the decoder.

Their outputs are `evidence/perf_v2/independent/verify_perf_public_relocated.json` and `verify_perf_local.json`,
summarized in `README_2026-11-03.txt` there.

**The reviewer passed all seven corrected claims** against the evidence (4c73eec against 0f2f9b1):
- the load figures reproduce from the bundled logs and ledger windows;
- its own sessions' file times corroborate the disclosed windows;
- the cost figures re-derive from the ledger;
- a spot check of the first-table split agrees within 0-3 points;
- the training streams differ from update 0;
- no frozen result or existing evidence byte changed.

It found one pre-existing rounding slip, Q27's new-population time to submission of 139.9 s not 140.0, now
corrected. It also made the nested-SELECT caveat to the first-table split, now in answer 4.

**The earlier runs, kept as recorded:**

| run | when | checks | failures | output |
|---|---|---|---|---|
| final, against `analysis.json` | before the correction pass; sha256 `d0292a58…`, at 4ff35e1 | 813 | 0 | `verify_perf.json` |
| phase 1 | before training | 221 | 0 | `verify_perf_phase1.json` |

The 813-check run's sections:

| section | checks | what it re-derives |
|---|---|---|
| pins | 37 | the protocol and both pins hash to the frozen values, at the freeze commit |
| history | 36 | the pilot's protocol, records, bundle and ledger entries are unchanged |
| ledger | 16 | the complete GPU ledger: every reservation since the pilot ended ok, none overlap, 21.416 of 32 h |
| baselines | 266 | L, B4 and Q27 on both populations: inventories, every SQL re-executed, abstention reasons, budgets; the new-64 configurations equal the 16-63 ones except positions; the Qwen-path sources are byte-identical across f81944c and c05e08f |
| train | 349 | each fit's training set equals the pin; its own sampling stream equals every update's examples, identical across the arms of a seed; the recipes; final checkpoints re-hash, and their model and optimizer digests match; parameters counted from the state dicts (v2 adds exactly 4,218,064) |
| eval | 48 | one passing evaluation per seed, of the final checkpoints; the reconstruction inventories and anchors; all 1,536 records re-executed and re-scored |
| compare | 61 | counts, Q1, Q2, leave-one-database-out, sign flips, per-question and per-database values, serving, baselines: every one of `analysis.json`'s 3,143 leaves equals the re-derivation |

**Warnings, with no score changed:**
- **Timeouts that depend on load.** Two submitted queries were recorded as wrong answers in 13.9 s and 15.9 s during
  evaluation (s100 C4-v1 bird23tf-00472, s101 C4-v1 bird23tf-03149). On the reviewer's machine load they time out at
  30 s. Both score 0 either way.
- **Spelling.** Records spell the shared arm `C4_params`; configs use `c4_params`.

**Not checked independently:**
- the strata classification;
- the teacher-forced first-table predictions themselves (no re-decode);
- that the arms of a seed share the trunk initialization at update 0 (no update-0 checkpoint is kept; the code and a
  CPU test pin it);
- the loss values (nothing was retrained);
- a CPU re-decode;
- the U = 3,000 feasibility decision.

## What this does not establish

- **Not a confirmation.** The new population is fresh to this study, but it is development data from databases that
  informed the design.
- **Not separable from capacity.** Relations and capacity are confounded: there is no capacity-matched v1 control.
- **Not separable from the hard restriction.** The same-reference ON restriction is part of v2, so its own
  contribution is not isolated.
- **Not a statement about specialization in general.** One semantic organization (four typed heads), one size, one
  backbone.
- **Not a strict accuracy.** EX scores result sets, so the counts include answers whose programs are suspicious
  (listed above).
- **Not a total cost.** Marginal serving times exclude the setup (the state cache and a fit). Qwen's times come from
  an HF eager harness with no cross-turn prefix cache, so they are upper bounds for a production server. Some of
  them overlapped other CPU work; no correction is applied.

## Recommended next experiment

Specialization has now had two chances, under v1 and under v2, and shows no consistent effect. The limiting
capability is grounding the question in an unseen schema. The next experiment should target that directly, as its
own frozen comparison. It should not scale these controllers.

**What:** a **question-schema linking stage**, trained to select the gold table set and to ground literals before
decoding:
- a bidirectional encoder reads the question with each table's columns and sampled cell values;
- it scores tables and columns jointly, as in schema-item ranking (RESDSQL, Li et al., AAAI 2023);
- literals are matched against database values (BRIDGE, Lin et al., Findings of EMNLP 2020).

**Gate:** its teacher-forced first-table and literal accuracy on held-out development databases, before any
end-to-end comparison.

## Evidence and reproduction

- **The bundle:** `evidence/perf_v2/` (weight-free; `SHA256SUMS`), built with `python -m scaling.evidence
  bundle-perf`. It holds:
  - the diagnosis runs (before and after evaluation);
  - perf-encode and perf-feasibility;
  - every fit's config, training log, checkpoint records and attempt summaries;
  - the three evaluations with their decodes and records;
  - the analysis;
  - the run logs (chains, co-tenancy);
  - the baselines run in this study;
  - the full ledger.
- **Checkpoints:** not copied. Their sha256 and digests are in `checkpoints.jsonl`; the weights stay in the runs
  root.
- **Tables:** `python -m scaling.perf_report` recomputes the decision and prints the tables. `--questions` prints
  every question, and `--corrections` the 2026-11-03 corrections. All three read only the bundles `perf_v2`, `dev_v1`
  and `pilot_v1`.
- **Correction inputs:** `corrections/cpu_overlap.json` (the CPU-overlap disclosure, from the session transcripts)
  and `diagnosis/*_first_table_*/first_table.json` (`python -m scaling.diagnose first-table`, CPU, final
  checkpoints).
- **Diagnosis tables:** `python -m scaling.diagnose perf-tables --out <bundle>/diagnosis/<stamp>_post_<commit>`.
- **The independent verifier:** `independent/verify_perf.py`, as written by the reviewer:
  `python research/ti_tnn/scaling/independent/verify_perf.py --public` from any checkout, or with
  `--runs-root <runs root>` for the local checks. Its outputs are in `evidence/perf_v2/independent/`.
- **Code:**
  - the frozen code is c05e08f, and every fit, evaluation, baseline on the new 64 and the analysis ran from it;
  - the B4/Q27 runs on 16-63 ran from f81944c, whose Qwen path is byte-identical;
  - the reporting code (bundler, report tables, diagnosis tables) was added after the freeze. It does not touch
    `perf.py` or `perf_analysis.py`.
