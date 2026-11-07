# Would database evidence have reached the grounding errors? A CPU coverage check of a reference-blind lookup

**Status:** a CPU check on saved records. No model call, no training, no GPU hours, and no new evaluation population.
Nothing here is a model result: it measures what evidence a lookup retrieves, not whether a model would use it.

## A concrete example

The direct Qwen3.5-4B proposal for "How many businesses are actively running in Gilbert City?" was:

    SELECT COUNT(*) FROM Business WHERE city = 'Gilbert City' AND active = 'true'

It returns 0, because the database stores the city as 'Gilbert'. The lookup sees only the question, the evidence,
this SQL, the schema and the snapshot. It produced this text, which is what a revision prompt would show the model:

    Value 'Gilbert City' in Business.city: 0 rows store it verbatim (distinct values scanned: 68)
      1. stored 'Gilbert' (prefix)
    Value 'true' in Business.active: 13733 rows store it verbatim (distinct values scanned: 2)
    Value 'Gilbert City' elsewhere: not found in 35 columns searched (not proof of absence)
    ...

The lookup does not change the SQL. Whether a model then writes `city = 'Gilbert'` is the separate experiment
proposed at the end; it is not launched.

## The idea

The residual-error audit (PR #72) found that the largest correctable group of the direct proposal's errors is
grounding: 17 of the 36 executable-but-wrong cases, covering 11 questions. These are values the database stores
differently, and near-miss columns and tables. Each of these facts sits in the database itself. Before asking a model to
use such evidence, it is worth knowing whether a bounded, deterministic lookup retrieves it at all, without ever
seeing the reference answer.

## The lookup (`db_evidence.py`, version db-evidence-v1)

`retrieve(question, evidence, sql, schema, snapshot_path, limits)` returns a package of six kinds of evidence. Each is
a ranked list of candidates with their provenance (table, column, method):

| kind | what it retrieves |
|---|---|
| value | for each string literal the proposal compares with a column: the verbatim row count, and the nearest stored values of that column (case, whitespace, punctuation, prefix, containment, shared words, character similarity) |
| value_elsewhere | for a literal its own column does not store, or that the evidence or question quotes: the text columns of the proposal's tables, their declared neighbours and the three best-matching other tables that store or contain it |
| column | for each column the proposal filters on, returns or uses numerically: columns of the same and neighbouring tables close in name, or matching the question's and evidence's terms in name and description |
| table | unused tables matching the question's and evidence's terms, or close in name to the proposal's tables |
| relationship | the declared foreign keys around the proposal's tables; each proposal join checked against them; composite keys joined on part |
| representation | for each column used numerically, ordered by or compared with a literal: declared type, stored types and text formats (currency, thousands separators, percent, dates, numbers as text) |

**What it never sees.** No task id, reference SQL, expected answer, score or audit label is a parameter. The module
opens no file but the snapshot, read-only. A test checks the signature and that none of these words occurs in the
code.

**Bounds.** All are deterministic. Instruction budgets are counted by an SQLite progress handler, so a stopped query
stops at the same point on every run.

| limit | value |
|---|---|
| distinct values scanned per column | 20,000 |
| text columns searched per literal | 40 |
| literals per proposal | 8 |
| candidates per item | 5 |
| SQLite VM instructions per query / per proposal | 20 million / 400 million |
| rendered payload | 6,000 characters, whole items in a fixed order |
| wall-clock safety stop (recorded if it fires; it never did) | 120 s |

**What it does not do.** It never substitutes a literal or rewrites the SQL. An empty result does not show that a
value is absent: every scan limit hit is recorded, and the text says "not proof of absence".

## Procedure, in the order it happened

1. **The lookup, its tests and the requirements were committed together (dd45591).**
   - The tests use a small synthetic database and none of the audited examples.
   - `coverage/requirements_v1.json` names, for each of the 17 grounding-labelled cases, the database facts its repair
     needs.
   - It also gives each case a benchmark status: whether matching the reference needs something the question does not
     ask for.
2. **The retrieval ran on all 128 saved initial proposals (02d3768).**
   - These are the semantic-proposal-repair study's frozen run: 64 questions, seeds 200 and 201.
   - The run read only the bundled requests, the tasks' question and evidence, the schemas and the snapshots. It
     opened no scores.
   - The outputs were committed under `evidence/evidence_coverage_v1/retrieval/` (SHA256SUMS) before any comparison.
3. **The stored outputs were compared with the requirements and the audit (0cd8c18).** A fact counts only if its item
   is in the rendered payload a model would read.

**Disclosures:**
- **The audit came first.** The lookup was designed by the same session that wrote the audit, after reading its cases
  and their reference SQL. The lookup's kinds follow the audit's categories. The reviewer was asked to check that no
  lookup choice depends on the reference answers or on a specific audited example.
- **A smoke run came before the requirements.** Before the requirements were written, the lookup was smoke-run on the
  first three saved proposals: bird23tf-05527, a targeted case, and bird23tf-00439 and bird23tf-00412, all seed 200.
- **Two changes followed the smoke run:**
  - a LIKE predicate is counted with its own pattern;
  - other columns are searched only for a literal its own column does not store.
- **The results are in-sample.** The cases are the ones the designer had read, so the coverage below is an
  optimistic estimate for a fresh population.

## Results

**The 17 grounding-labelled cases.** Classes: none, partial, complete (every required fact shown), or unresolved
benchmark disagreement (matching the reference needs something the question does not ask for; facts still reported).

| case | seed | class | facts shown | each fact: retrieved? rank (item) | payload chars | latency s |
|---|---|---|---|---|---|---|
| bird23tf-01633 | 200 | complete | 1/1 | 'Best International TV Series' -> Award.award 'Best International TV Series': yes, rank 1 (value_elsewhere) | 1194 | 0.005 |
| bird23tf-01666 | 200 | complete | 1/1 | column Vote.votes: yes, rank 2 (column) | 920 | 0.002 |
| bird23tf-03089 | 200 | complete | 1/1 | table TeamsSC: yes, rank 1 (table) | 1265 | 0.005 |
| bird23tf-03140 | 200 | partial | 1/2 | 'GR' -> CountryLanguage.CountryCode 'GRC': yes, rank 1 (value); 'Greece' -> Country.Name 'Greece': no | 1227 | 0.005 |
| bird23tf-04354 | 200 | complete | 1/1 | 'Gilbert City' -> Business.city 'Gilbert': yes, rank 1 (value) | 969 | 0.477 |
| bird23tf-04816 | 200 | complete | 1/1 | 'Czech Republic' -> PlayerInfo.nation 'Czech Rep.': yes, rank 1 (value) | 2030 | 0.030 |
| bird23tf-04930 | 200 | unresolved benchmark disagreement | 1/3 | join Teams.tmID = Master.playerID flagged: yes, rank 1 (relationship); table Goalies: no; 'G' -> Master.pos 'G': no | 1776 | 0.023 |
| bird23tf-05527 | 200 | partial | 1/2 | movies_total_gross.total_gross text format flagged: yes, rank 1 (representation); (no lookup kind) that the inner join to characters drops the top-grossing action movie: no | 1148 | 0.004 |
| bird23tf-05954 | 200 | unresolved benchmark disagreement | 1/1 | table Ball_by_Ball: yes, rank 2 (table) | 766 | 0.005 |
| bird23tf-06494 | 200 | none | 0/1 | column Master.nameGiven: no | 1963 | 0.018 |
| bird23tf-01633 | 201 | complete | 1/1 | 'Best International TV Series' -> Award.award 'Best International TV Series': yes, rank 1 (value_elsewhere) | 1194 | 0.005 |
| bird23tf-01840 | 201 | unresolved benchmark disagreement | 0/1 | column director.director: no | 658 | 0.002 |
| bird23tf-03089 | 201 | complete | 1/1 | table TeamsSC: yes, rank 1 (table) | 1265 | 0.005 |
| bird23tf-03140 | 201 | partial | 1/2 | 'GR' -> CountryLanguage.CountryCode 'GRC': yes, rank 1 (value); 'Greece' -> Country.Name 'Greece': no | 1227 | 0.005 |
| bird23tf-04354 | 201 | complete | 1/1 | 'Gilbert City' -> Business.city 'Gilbert': yes, rank 1 (value) | 969 | 0.478 |
| bird23tf-04816 | 201 | complete | 1/1 | 'Czech Republic' -> PlayerInfo.nation 'Czech Rep.': yes, rank 1 (value) | 2030 | 0.030 |
| bird23tf-05527 | 201 | complete | 1/1 | movies_total_gross.total_gross text format flagged: yes, rank 1 (representation) | 859 | 0.002 |

| class | cases (of 17) | questions (of 11) |
|---|---|---|
| complete | 10 | 5 (bird23tf-01633, 01666, 03089, 04354, 04816) |
| complete in one seed, partial in the other | - | 1 (bird23tf-05527) |
| partial | 3 | 1 (bird23tf-03140) |
| none | 1 | 1 (bird23tf-06494) |
| unresolved benchmark disagreement | 3 | 3 (bird23tf-01840, 04930, 05954) |

Of the 14 cases whose repair is not tied up in a benchmark disagreement, the lookup showed every required fact in 10,
some in 3 and none in 1. Most were found at rank 1; `Vote.votes` was at rank 2.

**What it missed, and why:**
- **bird23tf-06494 (`nameGiven` for "the given name").** The column item scores a column by half its name closeness to
  the proposal's column and half its share of the question's and evidence's terms. "name" is a stopword, so
  `nameGiven` matches only "given", 1 of 14 terms. It scored 0.258 (closeness 0.444, terms 0.071), under the 0.3
  threshold.
- **bird23tf-01840 (`director.director`).** The proposal returned `director.name`, which holds the movie. The
  `director` column scored 0.183 (closeness 0.167, terms 0.2), under the 0.3 threshold.
- **bird23tf-03140 ('Greece').** The value item listed five codes, 'GRC', 'GRD', 'GRL', 'BGR' and 'GBR'. 'GRC' is
  first only by the alphabetical tie-break among equal prefix matches, and nothing links it to Greece. The evidence
  names Greece without quoting it, so the lookup never searched `Country.Name`. The requirement counts 'GRC' as
  retrieved, which overstates how useful that fact is alone.
- **bird23tf-04930 (`Master.pos` = 'G').** The column search for 'G' stopped at its limit, 40 of 182 columns, before
  reaching `Master`. The limit hit is recorded. This is the case the "not proof of absence" wording exists for.
  `Goalies` was not among the five tables shown.
- **bird23tf-05527 seed 200.** The currency text format was flagged. The proposal's needless inner join to
  `characters` drops the answer, and no lookup kind detects that.

**Cost and bounds, every group.**

| group | proposals | payload chars (median / p90 / max) | latency s (median / p90 / max) | queries (median / max) | VM instructions (max) | scan limits hit | timeouts | any signal |
|---|---|---|---|---|---|---|---|---|
| all | 128 | 863 / 1510 / 2187 | 0.006 / 0.105 / 5.608 | 4 / 244 | 149,559,000 | 9 | 0 | 52 |
| originally correct | 74 | 777 / 1371 / 1926 | 0.009 / 0.325 / 5.608 | 4 / 244 | 149,559,000 | 7 | 0 | 23 |
| other wrong | 37 | 797 / 1740 / 2187 | 0.004 / 0.024 / 0.403 | 3 / 7 | 15,074,000 | 1 | 0 | 16 |
| 17 grounding-labelled | 17 | 1194 / 2030 / 2030 | 0.005 / 0.477 / 0.478 | 19 / 138 | 25,845,000 | 1 | 0 | 13 |

No query reached an instruction budget, and the wall-clock safety stop never fired. Nine proposals hit a scan limit,
which is recorded in each package. No package was truncated: the largest is 2,187 characters, against the 6,000 limit.

**Signals that could prompt a change**, counted per proposal on the items a model would read.

| group | value not stored, candidates shown | literal found elsewhere | join flagged | representation flagged |
|---|---|---|---|---|
| all | 6 | 13 | 15 | 27 |
| originally correct | 0 | 7 | 2 | 16 |
| other wrong | 0 | 0 | 11 | 5 |
| 17 grounding-labelled | 6 | 6 | 2 | 6 |

- **"Value not stored, candidates shown" is specific.** It occurs in 6 proposals, all in the targeted group, and in
  none of the 74 originally correct ones.
- **The other signals also fire on correct proposals:**
  - representation flags in 16 correct proposals;
  - join flags in 2;
  - a literal found elsewhere in 7.
- **Some of those are noise:**
  - the evidence's single-letter literals ('M', 'F', 'L', 'R') matched other columns, such as `Master.firstName`
    'R';
  - an aggregate in ORDER BY (`ORDER BY COUNT(text column)`) makes the lookup flag the counted text column as "used
    numerically".
- **The risk.** Such distractors are the main way evidence could harm already-correct answers, and the proposed
  experiment measures that directly.

**A correction to the audit.** In bird23tf-04354 the database stores `active` as 'true', in lowercase. The proposal's
`active = 'true'` is right, and the reference's `LIKE 'True'` matches it case-insensitively. Only the city, 'Gilbert
City' against 'Gilbert', is a grounding error. The audit's note says the stored value is 'True'; PR #72 is corrected.

**A note on bird23tf-04816.** The audit labels it both grounding and benchmark ambiguity, because the reference uses
a shorter country list than the evidence. Among Toronto's draftees only Belarus, Czech Rep., Slovakia and Ukraine
occur. So once 'Czech Rep.' is used, the evidence's full list gives the reference's result, and the case is not an
unresolved disagreement.

## Proposed frozen comparison against ordinary revision (not launched)

**Policies.** Every policy shares the same initial direct proposal per request.

| policy | revision message |
|---|---|
| D0 | none: the initial proposal |
| D1 | ordinary revision: the raw execution observation (PR #71's D1, unchanged) |
| V1 | D1's message plus the rendered db-evidence-v1 payload, and nothing else |

- **Held equal across D1 and V1:**
  - the revision seed, conversation prefix, token limits, deadline and alternating revision order;
  - the selection rule from PR #71: revised if extracted, read-only and executes; else original; else abstain.
  - The lookup's CPU time is charged to V1.
- **The lookup is frozen as db-evidence-v1, unchanged.**
  - The misses above suggest changes: keep "name" as a term in column matching, search names the evidence mentions
    but does not quote, and widen or reorder the column search.
  - Making them now would fit the lookup to these 17 cases.
  - If wanted, they become a v2, frozen and committed before the fresh population is pinned, and they are not
    reported as part of this coverage result.
- **Population.** A fresh, never-used population, for example inventory positions 256-319, pinned only after the same
  prior-use check PR #71 used. Seeds 200 and 201, so 128 requests per policy. Its reference queries are not inspected
  for any diagnosis until the analysis is written.
- **The rule.** The registered rule from `spr_rule`, applied consistently: V1 - D1 of at least +8 of 128, positive
  in both seeds and positive leaving any one database out. V1 - D0 and D1 - D0 are secondary.
- **Declared before the run:**
  - **recoveries and regressions,** with regressions on originally correct proposals as the harm measure for
    distractor evidence;
  - **the evidence signals,** reported per request without any reference;
  - **the costs,** in total: generation, lookup, tool and submitted-query execution, tokens and peak memory.
- **The decision:**
  - if V1 meets the rule against D1, database evidence earns further work;
  - if V1 harms by the rule, inspect its regressions;
  - otherwise, stop this direction.
- **Budget.** PR #71's frozen run of three policies took 0.48 GPU hours. V1 adds one revision call per request, the
  same as TI1, so about 0.5 h is the estimate. The ledger stands at 23.49 of 32 h. This needs a new authorization,
  and nothing is requested by this report.

## Limits

- **In-sample, with a designer who had read the cases.** The coverage on a fresh population could be lower.
- **The requirements were written by the producer.** They name the facts a repair needs, judged from the audit, the
  references and read-only checks. A different reader could ask for more, for example the link from 'Greece' to
  'GRC' in more cases.
- **Coverage is not use.** A shown fact does not mean a model would act on it, or act correctly. Reasoning changes
  such as SUM against COUNT and sort direction are listed but not tested.
- **One population, one model, one lookup version.** The signal counts on correct proposals measure potential
  distraction, not harm.

## Files

- `db_evidence.py`, `evidence_coverage.py` and `tests/test_ti_tnn_scaling_db_evidence.py`.
- `coverage/requirements_v1.json`: the frozen requirements.
- `evidence/evidence_coverage_v1/`, under SHA256SUMS:
  - `retrieval/20261107T075211Z/` holds every package, with its rendered text, ranks, provenance, payload size,
    latency, query and instruction counts, scan limits and timeouts;
  - `compare/20261107T075242Z/coverage.json` holds the comparison.

To regenerate the tables:

    python -m scaling.evidence_coverage tables
