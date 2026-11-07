# Grounding interface: explicit grounding for the saved controllers (development results)

This study ran the frozen protocol [`PROTOCOL_grounding_interface.md`](PROTOCOL_grounding_interface.md): sha256
`d3215e69…`, frozen at fb4e342, run from 9c139ac. It asked:

> Does explicit, task-conditioned grounding improve execution accuracy beyond what the existing controller extracts
> from its frozen-backbone features?

The six saved v2 controllers of the performance study were tested, without retraining, under three policies:

| policy | what grounds the request |
|---|---|
| G0 | nothing: the saved controller, unchanged |
| Glex | a deterministic lexical ranking of tables and columns |
| G4 | one non-thinking Qwen3.5-4B call of at most 512 tokens, which returns a grounding record, not SQL |

The six controllers are T4-v2 and C4-v2, seeds 100, 101 and 102.

A record is consumed only through a fixed prior: ln 2 / r is added to the table- and column-pointer logits, where r is
the item's rank in the record.

The study ran in four steps:
1. a post-hoc constraint-only diagnostic on the six v1 controllers;
2. a development phase on the performance study's 128 evaluated questions, with an oracle gate;
3. the protocol freeze;
4. the evaluation on 64 new questions (inventory positions 128-191), with a short-budget direct Qwen3.5-4B SQL
   control.

This is a development comparison in the same database collection as the earlier studies. It is not a confirmation.
The reserved BIRD Mini-Dev 500 was not read, and nothing was merged. The branch is built on the corrected PR #69 (its
merge is pending), and this PR is kept separate.

The interpretability work motivated the hypothesis but does not establish it. Nothing here claims that toy-program
findings transfer to Qwen3.5-4B on SQL, that failed probes show absent state, or that a grounding record explains
the pretrained model's internal computation.

## The answers, kept separate

**1. Nothing improves consistently (the primary outcome, new population).** EX-correct out of every requested
question, pooled over the three seeds (of 192 question-seed pairs):

| | G0 | Glex | G4 |
|---|---|---|---|
| T4-v2 | 29 | 29 | 31 |
| C4-v2 | 32 | 31 | 32 |

- **The largest difference** is T4's G4 − G0, at +2 (per seed +1, 0, +1). The protocol's bar is at least +10, the
  same sign in every seed and leaving any one database out.
- **G4 − Glex** is +2 (T4) and +1 (C4); **Glex − G0** is 0 and −1.
- **None is a consistent difference.**

The frozen reading is: **nothing improves consistently. Do not spend the remainder on more seeds or larger
controllers.**

**2. No evidence for specialization.** The architecture interaction, (T4: G4 − G0) − (C4: G4 − G0), is +2 (+1, 0,
+1), not consistent.

**3. The direct short-budget Qwen3.5-4B is better than every controller policy.** It used the same added-call
configuration, asked for SQL directly. It answered 32 of 64 (50%), against the best controller policy's 12, 13 and 9
of 64 in the three seeds.

The frozen reading is: **retain that baseline and reconsider whether the controller adds value.** This is a
short-budget control, not the performance study's thinking-mode B4 or Q27, and no claim against Qwen3.8 is made
here.

**4. The interface itself moved few decisions, even with correct grounding.**
- **Ordinary policies, dev questions:** the gate passed narrowly; Glex and G4 changed the final SQL on 10.4% and
  10.2% of the 768 decodes.
- **The oracle:** the gold program's tables and columns, unordered, changed 8.9% of decodes and added 5 EX-correct
  answers pooled (+6 for C4-v2, −1 for T4-v2).
- **New questions:** G4 changed the SQL of 14-15 of 192 decodes per architecture.

This interface, a fixed ln 2 / r nudge on logits the controller is already confident about, has not shown that it
can carry grounding into better programs. That is a finding about this interface, not evidence that grounding is
irrelevant.

**5. The grounding records were mostly right about tables, and much less complete on columns.** On the new
population:

| record | valid | table recall | table precision | first listed table in the gold set | column coverage | literal values right |
|---|---|---|---|---|---|---|
| G4 | 58 of 64 | 0.96 | 0.86 | 0.93 | 0.53 | 0.73 |
| Glex | 64 of 64 | 1.00 | 0.45 | 0.85 | 0.81 | — |

- **The G4 failures:** all six invalid G4 records named a column that does not exist, and fell back to G0.
- **Reference sets** are annotations, not proof that every alternative is wrong.

**6. Costs: marginal serving, per request and per EX-correct answer** (generation plus submitted-query execution):

| | per request | per EX-correct |
|---|---|---|
| G0 | 1.33-1.36 s | 8.0-9.0 s |
| Glex | 1.34-1.37 s | 8.3-9.0 s |
| G4 | 10.3 s (the 9.0 s call included) | 62-64 s |
| direct Qwen3.5-4B, short budget | 5.1 s | 10.3 s |

The work order allotted 8 GPU-reservation hours; this study used 1.407. Every reservation ended ok.

## The constraint-only diagnostic (post hoc; CPU; not a replacement for the frozen v2 results)

The six v1 checkpoints were decoded on the 128 evaluated questions twice:
- with the original v1 policy;
- with only the same-reference ON restriction added (`gi_constraint.RestrictedV1Machine`, tested to exclude exactly
  what v2 excludes and nothing else).

The weights, states, other legality rules and decoding limits were identical.

| v1 fits, pooled over the six | original v1 policy | v1 + the restriction |
|---|---|---|
| EX-correct of 768 | 71 (original) | 70 (restricted) |
| changed decisions | — | 117 (exactly the same-reference outputs) |
| wrong to correct, correct to wrong | — | 1, 2 |
| same-reference ON outputs | 117 | 0 |
| scorer timeouts | 10 | 2 |
| self-join outputs | 117 | 117 |

The per-architecture split:

| architecture, population | EX: v1, restricted | same-reference outputs | timeouts: v1, restricted |
|---|---|---|---|
| T4-v1, historical | 18, 18 | 31 | 1, 1 |
| T4-v1, new | 21, 21 | 33 | 3, 0 |
| C4-v1, historical | 15, 14 | 30 | 3, 0 |
| C4-v1, new | 17, 17 | 23 | 3, 1 |

- **Accuracy:** the restriction alone, applied to v1, removes the tautological joins and most timeouts and keeps
  every self-join. Accuracy goes from 71 of 768 (original) to 70 of 768 (restricted): one fewer correct answer.
- **What it suggests about v2:** the performance study's v2 gain (+15 and +18 of 192 on its new population) is not
  explained by the restriction alone. This is suggestive only: post hoc, on v1 controllers, it does not isolate the
  restriction inside v2.
- **Reproduction:** the CPU v1 decode equals the saved GPU decode for 765 of 768 requests.

## The development phase (128 questions; used to build and select, as the work order allows)

**G4 prompt and parser.** The first G4 run (prompt v1) gave 111 valid records of 128:
- 7 replies left out "status";
- 3 quoted names as the schema text does;
- 1 had an invalid JSON escape;
- 6 named unknown columns.

Two mechanical parser repairs were selected: one layer of SQL quoting around a name is removed, and invalid
backslash escapes are dropped. Prompt v2 asks for all four keys and unquoted names. The second run gave 124 valid of
128, the remaining four naming unknown columns. These were chosen on validity, not on EX.

**The gate (dev decode with the v2 records).**

| EX-correct of 384 | G0 | Glex | G4 | oracle |
|---|---|---|---|---|
| C4-v2 | 68 | 69 | 71 | 74 |
| T4-v2 | 70 | 71 | 69 | 69 |

| share of the 768 decodes whose final SQL changes | Glex | G4 | oracle |
|---|---|---|---|
| | 10.4% | 10.2% | 8.9% |

The rule, committed before these runs (e759e58), was "some policy changes the final SQL on at least 10% of the dev
decodes". It passed on the ordinary policies, so the study continued to the new population. The swap check (another
question's G4 record from the same database) changed 36 of 768 decodes.

## The new population (frozen)

`python -m scaling.gi_report` prints these from the bundle alone, after checking every count, comparison, the
interaction, the direct control and the costs against `analysis.json`:

Recomputed from the bundled records and equal to analysis.json (seeds [100, 101, 102]).

| arm | policy | seed 100 | seed 101 | seed 102 | pooled of 192 |
|---|---|---|---|---|---|
| T4-v2 | G0 | 9 | 13 | 7 | 29 |
| T4-v2 | Glex | 9 | 13 | 7 | 29 |
| T4-v2 | G4 | 10 | 13 | 8 | 31 |
| C4-v2 | G0 | 12 | 11 | 9 | 32 |
| C4-v2 | Glex | 12 | 10 | 9 | 31 |
| C4-v2 | G4 | 12 | 11 | 9 | 32 |
| direct Qwen3.5-4B (short budget) | one run | — | — | — | 32 of 64 |

| comparison | seed 100 | seed 101 | seed 102 | pooled | leaving one database out (min .. max) | consistent | sign-flip p (exploratory) |
|---|---|---|---|---|---|---|---|
| T4: G4-G0 | +1 | +0 | +1 | +2 | +1 .. +2 | no | 0.500 |
| T4: Glex-G0 | +0 | +0 | +0 | +0 | +0 .. +0 | no | 1.000 |
| T4: G4-Glex | +1 | +0 | +1 | +2 | +1 .. +2 | no | 0.500 |
| C4: G4-G0 | +0 | +0 | +0 | +0 | -1 .. +1 | no | 1.000 |
| C4: Glex-G0 | +0 | -1 | +0 | -1 | -1 .. +0 | no | 1.000 |
| C4: G4-Glex | +0 | +1 | +0 | +1 | +0 .. +1 | no | 1.000 |
| (T4: G4-G0) - (C4: G4-G0) | +1 | +0 | +1 | +2 | +0 .. +3 | no | 0.750 |

Readings:
- the direct short-budget B4 answers more questions than the best controller policy in every seed: retain that baseline and reconsider whether the controller adds value
- nothing improves consistently: do not spend the remainder on more seeds or larger controllers

| arm-policy | requests | EX-correct | per request, s | per EX-correct, s | controller s | G4 call s | execution s | timeouts | abstentions |
|---|---|---|---|---|---|---|---|---|---|
| C4-G0 | 192 | 32 | 1.33 | 8.0 | 245 | 0 | 10 | 0 | 0 |
| C4-G4 | 192 | 32 | 10.30 | 61.8 | 248 | 1721 | 9 | 0 | 0 |
| C4-Glex | 192 | 31 | 1.34 | 8.3 | 246 | 0 | 10 | 0 | 0 |
| T4-G0 | 192 | 29 | 1.36 | 9.0 | 246 | 0 | 15 | 0 | 0 |
| T4-G4 | 192 | 31 | 10.33 | 64.0 | 248 | 1721 | 14 | 0 | 0 |
| T4-Glex | 192 | 29 | 1.37 | 9.0 | 248 | 0 | 15 | 0 | 0 |
| direct-B4-short | 64 | 32 | 5.14 | 10.3 | 0 | 0 | 0 | — | — |

| decodes against G0 | changed decisions | changed SQL | wrong to correct | correct to wrong | fallback to G0 | EX-correct with another table set |
|---|---|---|---|---|---|---|
| C4-G4 | 14 | 14 | 1 | 1 | 18 | 5 |
| C4-Glex | 19 | 19 | 0 | 1 | 0 | 5 |
| T4-G4 | 15 | 15 | 2 | 0 | 18 | 6 |
| T4-Glex | 14 | 14 | 0 | 0 | 0 | 5 |

| record (new population) | valid of 64 | scored (gold convertible) | table recall | table precision | table set exact | first listed in gold set | first = gold first FROM | column coverage | literal value accuracy | literal pair accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| G4 | 58 | 56 | 0.96 | 0.86 | 0.68 | 0.93 | 0.64 | 0.53 | 0.73 | 0.63 |
| Glex | 64 | 62 | 1.00 | 0.45 | 0.06 | 0.85 | 0.56 | 0.81 | — | — |

Swap check: another question's G4 record (same database) changes the final SQL of 34 of 384 decodes; EX-correct 63 with the own record, 58 with the swapped one.

| GPU reservation (this work order) | hours | status |
|---|---|---|
| gi ground dev (20261103T185105-14cc8a54) | 0.302 | ok |
| gi ground dev (20261103T191242-e3686432) | 0.323 | ok |
| gi encode-new (20261103T194759-50c68071) | 0.042 | ok |
| gi ground new (20261103T195032-c2178282) | 0.179 | ok |
| gi eval-new (20261103T200120-d90a321a) | 0.462 | ok |
| gi direct-new (20261103T202939-534bbad7) | 0.098 | ok |
| total | 1.407 | |

Ledger at the analysis: 22.823 of 32.0 h (analysis.json's `gpu_hours` keeps one entry per command name, so it shows only the second dev grounding run; the table above lists every reservation).

## Software correctness

- **Tests:** the scaling suite at 00b4837: 384 passed, 8 skipped (they need the downloaded BIRD data).
- **The study's tests** (`tests/test_ti_tnn_scaling_gi.py`, 42) cover:
  - the restricted machine against v2's restriction;
  - record parsing, validation and fallback;
  - the prior added to the pointer scores and restored;
  - a decoding controller following the prior and nothing else;
  - the oracle record;
  - Glex;
  - swap partners;
  - the work order's budget refusal;
  - the record diagnostics.
- **Integrity on the new population:**
  - 8 of 8 earlier contexts re-encoded equal to their anchors;
  - all 64 new contexts anchored;
  - every one of the 1,152 timed decodes (a fresh backbone forward each) equals its anchor and its reconstruction
    from the cache.
- **Timing conditions:**
  - The new population's reservations were quiet: one-minute load means of 1.1-1.4, with maxima of 1.5-1.8.
  - The development grounding runs were not quiet, and their times are development only:
    - the second (load mean 7.2) overlapped the producer's first dev decode;
    - the first (load mean 4.9, max 8.0) overlapped no run of this study, so its load came from other activity on
      the host.

## What this does not establish

- **Not a confirmation.** Positions 128-191 were unused by any model run before the freeze, but they come from the
  same development databases that informed the grammar's design.
- **Not a statement about grounding in general.** One integration rule (a fixed additive prior), one record format,
  one model call, and no retraining.
- **Not a comparison with the thinking-mode baselines.** The direct control is a short-budget call; the performance
  study's B4 and Q27 ran in thinking mode on another population.
- **Not a measure of faithfulness.** A record that changes an answer shows that the interface influences the system,
  not that the record is right or describes the backbone's reasoning.

## Diagnostic corrections (2026-11-05; no historical result is changed)

Five statements of this study, and of the performance study it builds on, need narrowing:

1. **The prior can only change close decisions.** A record changes a pointer decision only when, for some legal
   candidate, the prior's advantage over the selected item exceeds that item's logit lead. The largest possible
   advantage is ln 2 (about 0.69: a rank-1 item against an unlisted one), so any decision whose legal margin exceeds
   ln 2 cannot be changed by any record. The interface's small effect therefore partly reflects how often the
   controllers' margins exceed ln 2. The bounded margin diagnostic below measures this.
2. **The oracle was a small bias, not enforced grounding.** It added ln 2 to every gold table and column. Its +5 of
   768 measures what this prior does with correct information, not what correct grounding would achieve.
3. **Literals and status were not consumed.** Literal bindings and the unresolved status were recorded and scored but
   never reached the decoder. The interface could not have repaired a literal error.
4. **Exposure bias is not ruled out.** In the performance study's diagnosis, teacher-forced all-correct programs and
   exact greedy decodes coincided. That shows only that greedy decoding reproduces the gold when every step is right;
   it does not measure how a first error compounds. The forced-prefix diagnostic below tests that on a small sample.
5. **Specialization is not established.** T4 and C4_params did not differ consistently in the performance study (A3)
   or here (the interaction is +2, not consistent). Nothing here shows that specialization helps.

The next experiment must not be a larger controller or a larger ranking bias chosen after these results.

### The bounded margin diagnostic (CPU; post hoc; evaluator-only interventions)

**Setup.** `python -m scaling.gi_margin` (a new module; the frozen producer is untouched) took a deterministic
sample: the 32 dev questions with the smallest sha256 of `gi-margin-sample-v1:<task id>`. It used the saved seed-100
T4-v2 and C4-v2 checkpoints and fitted nothing. The run, `margin/20261106T024526Z` from a clean f0b6def, is identical
to an earlier uncommitted trial, which is kept and labelled.

At every table and column pointer decision along the unbiased (G0) greedy trajectory it recorded:
- the legal candidates' logits and the selected item's margin over the runner-up;
- for the G4 record and the oracle record, the applied prior difference, and whether that prior could change the
  selection with the prefix held fixed.

**The prior could almost never change a decision.**

| seed 100 | table decisions with margin > ln 2 | column decisions with margin > ln 2 | decisions G4's prior could change | decisions the oracle's prior could change |
|---|---|---|---|---|
| T4-v2 | 47 of 56 | 120 of 135 | 3 of 191 | 3 of 191 |
| C4-v2 | 54 of 61 | 137 of 146 | 6 of 207 | 6 of 207 |

The interface's small effect is explained mechanically. The controllers lead by more than ln 2 on 84-94% of their
pointer decisions, which no record can overturn.

**Two evaluator-only interventions** (they read the gold program; never deployable, not accuracy claims). EX-correct
of 32:

| seed 100 | G0 | reference-restricted tables and columns | forced reference prefix through the first divergence |
|---|---|---|---|
| T4-v2 | 6 | 7 (+2, -1) | 7, on 29 defined (G0 6 on the same) |
| C4-v2 | 6 | 10 (+4, -0) | 8, on 28 defined (G0 5 on the same) |

- **Reference restriction** removes every irrelevant first table:
  - T4: 6 irrelevant of 32 under G0, 0 restricted (11 remain another gold table first);
  - C4: 8 irrelevant, then 0 (12 remain).

  No request had a step without a reference-supported legal candidate. Yet accuracy rises only from 6 to 7 and 10.
  So even with the tables and columns right, most programs stay wrong: operators, literals, aggregates and structure
  remain.
- **Forced prefix:**
  - Undefined for 3 requests per fit: the gold program is not representable. C4 had one request with no divergence.
    No forced action was illegal.
  - After the gold prefix through the first departure, the free suffix reproduces the gold program in only 2 of 29
    (T4) and 5 of 28 (C4) cases.
  - Most first departures are table decisions (18 and 20). Forcing them right recovers only 2 and 4 correct answers.

  This is direct evidence that errors compound or recur after the first one: exposure to the model's own later
  decisions is not ruled out as a cause (correction 4).

This is a 32-question, one-seed diagnostic, and it is not a prerequisite for any further training study.

## Recommendation

Do not spend the remaining budget on more seeds, larger controllers or a stronger prior chosen after these results.

Two findings together question the controllers' value as currently built:
- even correct grounding barely moves them;
- one short Qwen3.5-4B call answers more than half the questions at about 10 s per correct answer.

A future controller experiment should be justified against that direct baseline first. One example is training the
controller to read a grounding record as an input, which is a new experiment with its own protocol, not a change to
this one.

## Independent verification

A separate reviewer session wrote `independent/verify_gi.py`.

**The final run** (sha256 `8ceff14c…`, at 1cae124): **331 checks, 0 failures.**
- **Every submitted SQL record:** all 12,416 SQL-carrying records of all seven runs (the margin diagnostic
  included) map to the executed snapshot and SQL. The mapping is `independent/sql_record_map.jsonl`.
- **Prior use:** checked before the freeze. The later reuse by the semantic-repair study's preservation test is
  listed separately.

**The earlier run**, kept as recorded (sha256 `b18c3882…`, at 429694a, from an empty output directory): **295
checks, 0 failures.** It ran on the CPU only and wrote nothing under the runs root. Its
main process imports no producer module. Producer code ran only in separate processes, flagged: the frozen code for
the schema items, the prompt's schema text, the cached states and the decoding framework, and `gi_report`.

The output is `evidence/grounding_interface_v1/independent/verify_gi.json`, summarized in `README.txt` there.

**All ten items of the work order's list pass.**

1. **Inventories.** Rebuilt from the raw release with the reviewer's own rule. Every run covers its population
   exactly once. The prior-use scan of 1,203 files finds only the three expected files.
2. **Input permissions.** The reviewer rebuilt 256 prompts with exactly the recorded token counts. Its own BM25
   reproduces every retrieved list, all from other training databases. No evaluated question's gold SQL or task id
   appears in any prompt. The oracle appears in no new-population run.
3. **Consumption.** Its own bias vectors equal the recorded records' (732 of 732). A CPU re-decode of all 1,152 timed
   decodes gives 1,149 equal. The 3 differences are G0 near ties, with logit margins of 0.0008-0.0072, likely TF32 on
   the GPU; the GPU's own reconstruction was equal. Removing literals and flipping the status changes nothing (348 of
   348).
4. **Fallback.** Its own parser agrees on every record's validity. Every one of the 36 new-population fallback
   decodes equals G0.
5. **Scores.** All 2,058 distinct question-SQL pairs were re-executed with its own EX rule, and every recorded score
   and status matches.
6. **Timing.**
   - `controller_seconds` spans the fresh forward, the Glex record and the decode.
   - The G4 call is charged to every G4 request, fallbacks included.
   - The six reservations ended ok, with no overlap, each within 1.5 times its projection. The worst case stayed at
     or under 23.175 h against the 29.416 h limit.
7. **Integrity.** All 64 new cache entries hash to their anchors, and all 1,152 fresh states equal them.
8. **The constraint diagnostic.** Its own restricted machine reproduces the restricted decodes (768 of 768), and every
   count matches.
9. **The analysis.** Every count, comparison, condition, interaction, cost and reading equals `analysis.json`.
10. **The dev phase.** The gate recomputes exactly. The freeze (fb4e342) postdates the last dev write, and the gate
    rule was committed (e759e58) before both dev decodes.

**Notes, none changing a result:**
- The first dev grounding run (prompt v1) started before e759e58; the gate applies to the decodes, which both came
  later.
- Two constraint-diagnostic queries time out at 30 s on the efficiency cores and finish, incorrect as recorded, with
  more time.
- `analysis.json`'s `gpu_hours` keeps one entry per command name (the table above lists every reservation).
- The reviewer's own reading of the record diagnostics agrees within tolerance, not exactly, because its gold parser
  differs.
- The protocol says "at least +10"; the reviewer applied |pooled| >= 10, as the performance study did. Here it changes
  nothing.

**Not checked independently:**
- the GPU timings and the model replies, beyond their consistency;
- the prompts byte for byte (by exact token count instead);
- a re-implementation of Glex;
- the sql_to_ir-based record diagnostics, exactly.

## Evidence and reproduction

- **The bundle:** `evidence/grounding_interface_v1/` (weight-free, `SHA256SUMS`), built by
  `python -m scaling.gi_report bundle`.
- **The tables:** `python -m scaling.gi_report` checks the bundle and prints the tables.
- **The commands:** `python -m scaling.gi_run <command>`, listed in `gi_run.py`'s docstring.
- **The runs:** the new population ran from the frozen worktree at 9c139ac; the development runs from 9d2475a and
  earlier commits, recorded in each manifest.
- **The ledger:** the whole ledger is in the bundle. This study's six reservations total 1.407 GPU-hours, and the
  cumulative total is 22.823 of 32.
