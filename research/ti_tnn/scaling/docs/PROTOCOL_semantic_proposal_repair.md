# Semantic proposal and typed repair: does structured feedback make one revision more useful? (development protocol)

**Status.** Frozen when its sha256 is committed in `pins/spr_protocol.json`. `spr_run run` refuses to start if this file
or `pins/spr_new_eval.json` differs from the pin. Everything below is fixed before any generation on the new
population.

**Dependency.** This study's branch is built on the grounding-interface branch (PR #70), for its direct-SQL prompt.
It is a separate PR. The historical producers and their evidence are read, never written.

## The question, with an example

The pretrained Qwen3.5-4B already writes good SQL: one short direct call answered 32 of 64 questions in the
grounding-interface study, against 9-13 for the TI-TNN controllers. So this study keeps the model's semantics. The
model proposes SQL, and the typed path parses that SQL into the existing intermediate representation, checks it
against the exact database, and reports what it observes.

Take the proposal `SELECT T2.app_id FROM label_categories AS T1 JOIN app_labels AS T2 ON T1.label_id = T1.label_id`:
- The raw execution observation, which both repair policies receive, is its status, row count and first rows.
- The typed analysis, which only TI1 receives, adds the plan and the observation "the ON predicate of the join with
  app_labels references only label_categories". It does not say which join was intended.

**The question:** does one revision with the typed analysis (TI1) correct more answers than one ordinary revision
with the same raw observation (D1)? D1 is the primary comparison, because any gain over the single proposal (D0)
could come from the extra call alone.

## The policies

| Policy | Behavior |
|---|---|
| D0 | the initial proposal, as it is |
| D1 | the same initial proposal, then one ordinary self-revision with the raw execution observation |
| TI1 | the same initial proposal, then one revision with the raw observation and the typed analysis |

## Fixed in advance

| Item | Value |
|---|---|
| Model | the pinned Qwen3.5-4B (`contracts.MODELS["B4"]`), non-thinking, the card's non-thinking sampling, at most 512 generated tokens per call, no tools |
| Initial proposal | `gi_llm.SYSTEM_DIRECT` and the B4 harness's user message (question, evidence, schema text at the 24,000-character budget, the retrieved training examples), seed `int(sha256("spr-initial:<seed>:<task_id>")[:8], 16)`. Generated once per question and sampling seed, shared by all three policies |
| Revision | the conversation so far (the initial reply included), then one user message: the raw observation (D1), or the raw observation and the typed analysis (TI1), followed by `spr_run.REVISE`. The same seed for both (`"spr-revise:<seed>:<task_id>"`). Every D1 and TI1 request is revised. The revision order alternates by a hash of seed and question |
| Raw observation | `tools.execute_select` of the initial SQL: read-only, the pinned 30 s timeout, at most 20 preview rows, cells cut at 60 characters, at most 4,000 characters |
| Typed analysis | `spr_feedback.typed_analysis` (version `spr-feedback-v1`): the plan; definite errors (identifiers the snapshot's schema lacks, validation errors, SQLite execution errors, timeouts); observational warnings (an ON predicate that does not reference the source it joins, a join with no ON, a literal not verbatim in the question or evidence, an empty result, a construct outside the IR). Never the gold program, the reference answer or evaluator metadata |
| Deadline | 180 s per request. Both revisions get the same remaining time after the initial proposal |
| Selection | D1 and TI1 alike, never by gold: the revised SQL if extracted, one read-only statement, and it executes without error or timeout; else the original if it does; else abstain. D0 submits the initial SQL as it is. Every candidate is recorded |
| Submitted query | the model's own SQL text in every policy. The compiled typed query is used for diagnostics and the preservation test only |
| Sampling seeds | 200 and 201 |
| New population | `pins/spr_new_eval.json`: inventory positions 192-255 (64 questions), checked unused by any model, controller, baseline or diagnosis run (`spr_run pin-new` scans the runs root, every evidence bundle and the pins, the grounding-interface runs included). Only three files name them: the inventory pin itself, the data-preparation gold check and the grammar's gold-program coverage analysis |
| Scoring | the pinned EX scorer (`scorer-v1`); every requested question in the denominator, abstentions, unsupported programs and timeouts included |
| Budget | at most 6 GPU-reservation hours within authorization v2's 32 cumulative hours: no reservation may let the cumulative total plus 1.5 times its projection pass 28.823 h |

## The preservation test (done before the freeze)

`spr_run preserve` took the 64 saved direct-Qwen outputs of the grounding-interface study, with no model call:
- **Supported:** 47 parse into the IR, and all 47 keep their EX outcome when compiled and executed (32 correct
  either way). 46 give identical result sets and ordered rows; one times out both ways.
- **Unexplained changes:** zero.
- **Unsupported:** 17 are reported apart (16 identifier errors in the model's SQL, 1 `YEAR` function).

This shows that the typed path preserves answers. It is not a capability gain.

## Development (done before the freeze; earlier dev questions only)

Two debugging runs used 16 of the 128 earlier development questions (the smallest sha256 of
`spr-dev-sample-v1:<task id>`) at debugging seed 900.
- **The first run** (prompt `spr-prompts-v1`, 0.117 GPU-h): 4 of its 32 revisions argued at length and hit the
  512-token ceiling before any SQL.
- **The revision instruction** was therefore changed, identically for D1 and TI1, to ask for the final query alone
  (`spr-prompts-v2`).
- **The second run** (0.070 GPU-h for 16 requests under all three policies): no truncation; revisions of about 50
  generated tokens and 4.1-4.3 s each; initial proposals 4.1 s; peak memory 11.1 GB.

The instruction was chosen on truncation, not on EX. EX on these 16 (8, 8, 9 for D0, D1, TI1) is not read.

**The projection for the new population:** 64 questions x 2 seeds, about 13 s per request for the three calls and
the executions, plus loading, about 0.5 h. Reserved at 0.6 h, the watchdog's worst case is 0.9 h. The ledger held
23.010 h at the freeze, against this work order's limit of 28.823 h, so both seeds and both repair policies fit. The
run is all or nothing: both seeds, all three policies.

## Outcomes

**Primary: EX-correct requests out of every requested question** (64 questions x 2 seeds = 128 per policy). For each
pair, the difference is summed over seeds, with per-seed values and leave-one-database-out values:
- TI1 - D1 (primary);
- D1 - D0;
- TI1 - D0.

**The research-priority rule** (not a statistical confirmation): a useful development lead is at least +8 of 128,
positive in both seeds, and still positive when any one database is left out.

**The decision:**
- **TI1 beats D1 by the rule:** structured feedback has earned further investigation.
- **TI1 harms correctness:** TI1 - D1 or TI1 - D0 is the rule's mirror image (-8 or worse, negative in both seeds and
  leaving any database out). Inspect the rejected or revised correct proposals, and do not hide regressions behind
  fewer invalid queries.
- **TI1 ties D1 but both beat D0** by the rule: ordinary revision explains the improvement.
- **Otherwise, neither revision helps:** retain direct Qwen and stop this repair direction.

If TI1 only reduces invalid queries or timeouts without improving EX, that narrower engineering benefit is reported
as such.

**Also reported:**
- paired recoveries and regressions;
- results by seed and database;
- the initial proposals' typed and untyped rates and their reasons;
- revision rejections and original-candidate fallbacks;
- final execution statuses;
- generation, validation, tool and submitted-query execution time, separately and together;
- prompt and generated tokens;
- peak GPU memory;
- every reservation.

**Kept apart:**
- syntax and type validity (typed analysis);
- execution success (read-only execution);
- agreement with the benchmark answer (EX);
- proven semantic equivalence, which is never claimed;
- faithfulness of the execution record (the recorded runs).

No claim against Qwen3.8 is made: it is not run on this population.

## Limits

A development comparison in the same database collection as the earlier studies, with two sampling seeds. Not a
confirmation. The reserved BIRD Mini-Dev 500 is not read.
