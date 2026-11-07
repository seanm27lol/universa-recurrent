# Diagnosis of the pilot's controllers (CPU only, before any intervention)

This is the diagnosis that chose the performance study's intervention. It read only:
- the pilot's saved evaluation records and its four final checkpoints (T4 and C4_params, seeds 100 and 101,
  1,500 updates);
- the 64 historical development questions;
- a fixed 256-question training-pool sample (the smallest sha256 of `perf-v2-diagnosis-train-sample-v1:<id>`).

It ran on the CPU, with no GPU and no training. The new evaluation population (inventory positions 64-127) was not
read. Code: `diagnose.py`. Outputs: `evidence/perf_v2/diagnosis/`. The tables print with
`python -m scaling.diagnose tables --out <dir>`.

**The CPU decoding reproduces the pilot.** Decoding in float32 on the CPU gave the saved actions for 64 of 64
requests in each fit (the pilot decoded in TF32 on the GPU).

## The answer in one paragraph

The controllers fail at **grounding tables and columns in databases they never trained on**. Under teacher
forcing, given the gold prefix, the error rates on the historical questions, with the training sample's in
parentheses, are:
- the first FROM table: 49% (17%);
- the joined table: 52% (16%);
- SELECT columns: 42% (10%);
- WHERE columns: 33% (10%).

In free decoding the predicted table set is right on only 16-21 of the 64. When it is right, the join keys are
right in 15-19 of those cases. So the join keys are not the bottleneck.

The 42 same-reference join outputs (`T1.x = T1.x`) are a symptom: in 40 of them the table set is already wrong.
They are also costly: all six scorer timeouts are among them.

T4 and C4_params fail the same way. Their category counts differ by about as much as their own seeds do.

## Training fit against free decoding

| fit | sample | teacher-forced loss | TF step accuracy | TF all steps right | free decoding exact | free EX-correct | table set right | historical EX-correct |
|---|---|---|---|---|---|---|---|---|
| T4 s100 | 256 | 0.144 | 0.954 | 101 | 101 | 149 | 196 | 9 |
| T4 s101 | 256 | 0.161 | 0.945 | 89 | 89 | 140 | 195 | 9 |
| C4_params s100 | 256 | 0.163 | 0.943 | 81 | 81 | 139 | 190 | 4 |
| C4_params s101 | 256 | 0.162 | 0.946 | 95 | 95 | 135 | 178 | 10 |

- **The fits did learn their training data,** but not perfectly. A teacher-forced loss of about 0.15 leaves 5% of
  steps wrong, and one wrong step in a 23-step program is enough to spoil it. Only 32-39% of training programs are
  decoded exactly, and 53-58% execute correctly.
- **Exposure bias is not the gap.** On every fit, "all steps right under teacher forcing" equals "exact under free
  decoding". Greedy decoding loses nothing beyond the first wrong step.

  *Correction (2026-11-05; the numbers above are unchanged).* This equality does not rule out exposure bias. It shows
  only that when every teacher-forced step is right, greedy decoding reproduces the gold program. It says nothing
  about how a first error compounds when the model then conditions on its own prefix. The forced-prefix diagnostic of
  the grounding-interface study (`REPORT_grounding_interface.md`, "Diagnostic corrections") tests that directly on a
  small sample.
- **Generalization is the gap.** On the historical questions the teacher-forced loss is 0.98-1.14 and the step
  accuracy 0.77-0.79, against 0.94-0.95 on training questions.

## Where each decision fails (teacher forced, all four fits)

| decision | historical steps | wrong | error | training-sample steps | wrong | error |
|---|---|---|---|---|---|---|
| WHERE literal | 372 | 178 | 0.48 | 1,508 | 191 | 0.127 |
| WHERE operator | 640 | 160 | 0.25 | 2,552 | 158 | 0.062 |
| SELECT column | 348 | 145 | 0.42 | 1,532 | 150 | 0.098 |
| WHERE column | 380 | 126 | 0.33 | 1,488 | 142 | 0.095 |
| FROM table | 248 | 121 | 0.49 | 1,108 | 193 | 0.174 |
| join: add table | 168 | 87 | 0.52 | 1,108 | 172 | 0.155 |
| ON column (first operand) | 168 | 72 | 0.43 | 1,104 | 88 | 0.080 |
| join: no more tables | 248 | 65 | 0.26 | 1,136 | 65 | 0.057 |
| SELECT aggregate | 104 | 60 | 0.58 | 504 | 45 | 0.089 |
| ON column (second operand) | 168 | 35 | 0.21 | 1,108 | 36 | 0.032 |
| DISTINCT | 248 | 28 | 0.11 | 1,136 | 36 | 0.032 |
| GROUP BY column | 36 | 23 | 0.64 | 100 | 31 | 0.310 |

**Grounding dominates.** The table and column pointers account for 564 of the 1,201 wrong historical
steps:
- FROM table 118 and joined table 72;
- WHERE columns 120 and SELECT columns 120;
- join keys 107;
- GROUP BY, ORDER BY and HAVING columns 27.

Literal spans and operators (178) are the next largest group.

**Tables come first in the traversal,** so table errors are the first divergence from gold in most requests. Of
the 58 representable programs:
- the first FROM table is the first divergence in 27-30;
- the join count or kind in 8-12;
- the joined table in 7-10;
- a join column in 1-4.

**Table choice and name mentions.** In the schema text a table's name comes before its columns, and the frozen 4B
reads causally. So a table's state cannot see its own columns. Under teacher forcing, the first-table pointer is
right:
- 52 of 68 times (76%) when the gold table and one of its gold-query columns are both named in the question or
  evidence;
- 58 of 112 (52%) when only the column is named;
- 17 of 48 (35%) when the table is named only by one of its words.

This fits that structural blindness, though it does not prove it: questions that name the table may also be easier.

## The 42 same-reference join outputs and the six timeouts

The regular expression on the final SQL finds them, and the plan analysis finds the same 42. Most are in the top
SELECT, and one is inside a derived table. A 43rd output has the same predicate in WHERE, not ON.

| fit | outputs with a same-reference ON equality | timeouts |
|---|---|---|
| T4 s100 | 10 | 1 |
| T4 s101 | 14 | 2 |
| C4_params s100 | 4 | 0 |
| C4_params s101 | 14 | 3 |

**What each predicate means.** Its semantics come from the resolved columns and the query, not from the text. The
42 outputs hold 45 equalities. `X = X` is `X IS NOT NULL`: it relates no two sources, so the joined source is
crossed with everything before it. What that does to the result set:

| effect | equalities |
|---|---|
| the crossed source is read elsewhere in the query: rows are paired arbitrarily | 21 |
| result-set equivalent to omitting the join, if the joined table is non-empty (EX compares sets) | 10 |
| a cross product under LIMIT: duplicates can displace distinct rows | 9 |
| a cross product under COUNT, SUM or AVG: the aggregates are multiplied | 5 |

None of them is in a gold program:
- the gold program joins the same column to a partner in 24 of the 45;
- the column has a declared foreign-key partner, but the gold program does not join on this column, in 18;
- neither exists in 3.

**Every one of the six scorer timeouts** (30.0-33.2 s) is a same-reference join output.

## EX-correct answers whose programs are suspicious

EX compares result sets, so a wrong program can score. Of the fits' EX-correct answers:
- T4 s100: 3 of 9;
- T4 s101: 4 of 9;
- C4_params s100: 2 of 4;
- C4_params s101: 5 of 10.

The programs differ from the gold program in ways EX does not see:
- **a same-reference join, harmless here:** 01627, 04507, 06228;
- **an extra joined table:** 06518, 05912, 03127;
- **a missing table:** 05697;
- **a missing WHERE conjunct:** 03526, 01625.

## Error categories by fit (64 historical questions; a request can have several)

| category | T4 s100 | T4 s101 | C4_params s100 | C4_params s101 |
|---|---|---|---|---|
| table grounding (table set differs) | 44 | 43 | 48 | 45 |
| projection (SELECT items differ) | 43 | 40 | 47 | 40 |
| predicate column | 42 | 41 | 41 | 37 |
| join key, different tables | 36 | 36 | 42 | 34 |
| aggregation | 18 | 15 | 20 | 16 |
| join on an undeclared, non-gold key | 12 | 10 | 14 | 7 |
| same-reference join | 10 | 14 | 4 | 14 |
| predicate literal | 8 | 6 | 13 | 12 |
| DISTINCT | 11 | 5 | 10 | 10 |
| ordering or LIMIT | 9 | 8 | 8 | 8 |
| grouping or HAVING | 8 | 8 | 8 | 8 |
| nested query involved | 7 | 5 | 6 | 5 |
| join key, same tables | 2 | 2 | 1 | 2 |
| execution timeout | 1 | 2 | 0 | 3 |

No request abstained, and no plan was invalid.

**T4 and C4_params.** The two architectures fail in the same places, in the same proportions. The seed-to-seed
spread within an architecture is as large as the spread between them. For example, same-reference joins are 10
and 14 for T4, and 4 and 14 for C4_params.

## The choice

The user's preferred candidate was joint, relation-aware join grounding. The diagnosis supports its relational
part, but not its scope. Join keys fail mostly after a wrong table choice, and the larger failure is table and
column grounding in general. The chosen intervention, **relational grounding** (`grounding.py`), gives every table
and column pointer the schema's relations:
- a relation-aware encoder over the items;
- key and name-mention features;
- plan-relative relations, including the second join operand scored against the first;
- the constraint that removes same-reference join predicates.

It is one mechanism, identical in T4 and C4_params. Literal grounding (48% teacher-forced error) is the other large
failure. It was not addressed in this comparison, because it is a different mechanism.

**What it costs in representability.** Of the 6,188 historical gold programs (4,741 training, 1,447 development),
6,187 encode under v2 to exactly v1's actions. The other one is training question `bird23tf-00003`, whose gold ON
holds `T2.list_id = T2.list_id` beside a real key. It is no longer representable and leaves every arm's training
set.
