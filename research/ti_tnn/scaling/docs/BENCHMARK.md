# The benchmark of the scaling development study

**Status: implemented and tested locally (development protocol `dev-1`, nothing frozen).** This page describes the
benchmark side: data, split, inventory, schema text, tools, scorer, retrieval and the no-fit control L. The arms and
the protocol are in [PROTOCOL_dev.md](PROTOCOL_dev.md), and the shared shapes in [`contracts.py`](../contracts.py).

## One request, end to end

Development request `bird23tf-04392` is a question about the `public_review_platform` database. Every arm receives
the same inputs:
- the `Task`: the question, BIRD's evidence sentence, the db_id and `split="dev"`;
- the schema text of that database (format below);
- 3 retrieved training examples, each with its question, evidence and gold SQL, all from *other* databases;
- three read-only tools on the pinned snapshot: `describe_schema`, `lookup_values` and `execute_select`.

The arm returns SQL or abstains. The scorer then runs that SQL and the gold SQL on the snapshot, and the answer is
correct when the two result sets are equal as Python sets (BIRD's execution accuracy, EX). No arm ever sees the gold
SQL of a development question.

## Sources and pins

| Source | Identity (verified before use) |
|---|---|
| Questions: `birdsql/bird23-train-filtered` at `4068469` | `data/train-00000-of-00001.jsonl` sha256 `2ee64aa5...`; 6,601 rows, 69 databases |
| Column summaries: same repo | `train_column_meaning.json` sha256 `7928ddb4...` |
| Databases: official BIRD `train.zip` | 8,919,543,554 bytes, sha256 `66e9e311...`; it nests `train/train_databases.zip` (sha256 `902fed2d...`) |
| Reserved public benchmark: `birdsql/bird_mini_dev` at `f65faf4` | file sha256 `88ceb071...`, 500 questions; `minidev.zip` sha256 `cc48ba16...` |
| Official scorer: `bird-bench/mini_dev` at `abd11b6` | `evaluation_ex.py` sha256 `da1bbcd4...`, `evaluation_utils.py` `f6943d24...` |

Pins live in `research/ti_tnn/scaling/pins/`. They hold ids and hashes only. Each is rebuilt by
`PYTHONPATH=research/ti_tnn python -m scaling.data <step>` (or `all`), and a rebuild must reproduce the pin byte for
byte, or the build stops:

| Pin | Holds |
|---|---|
| `snapshots_v1.json` | sha256 and size of the 69 database files (32,406,294,528 bytes in total), a digest of each database's description CSVs, the archive hashes |
| `split_v1.json` | the 14 development and 55 training-pool databases, the rule, the salt, questions per database |
| `inventory_v1.json` | the 512 ordered development ids with their db_ids, the exclusions with reasons, the per-database composition |
| `smoke_train_candidates_v1.json` | the first 1,024 training-pool ids in the smoke-train order (order only; the IR decides which compile) |
| `reserved_public_pin_v1.json` | the 500 Mini-Dev question ids and db_ids, and the sha256 of its 11 SQLite files |

Task ids are `bird23tf-{i:05d}`, where `i` is the 0-based line of the JSONL. Each row also has a `content_sha256`,
the sha256 of `db_id, question, evidence, SQL` joined by `\x1f`.

**The reserved benchmark is pinned, never used.** `data.reserved_identities` keeps each record's `question_id` and
`db_id` and discards the rest of the parsed file at once. Only the 11 `.sqlite` files are extracted from
`minidev.zip`, so they can be hashed. They are read-only (mode 0444) and were never queried, and `tools.py` refuses to
open anything under `reserved_databases/`. The 11 databases (`california_schools`, `card_games`,
`codebase_community`, `debit_card_specializing`, `european_football_2`, `financial`, `formula_1`, `student_club`,
`superhero`, `thrombosis_prediction`, `toxicology`) are disjoint from the 69 training databases by name and by file
hash.

## The split and the inventory

**Split** (`contracts.SPLIT`): the 69 databases are ordered by `sha256("ti-tnn-scaling-dev-split-v1:" + db_id)`. The
first `round(0.2 * 69) = 14` are development and the other 55 are the training pool. The split is by database, so no
development database ever appears in training or retrieval.

| Development database | Questions | In the first 16 / 64 / 512 |
|---|---:|---|
| public_review_platform | 256 | 2 / 5 / 37 |
| soccer_2016 | 190 | 2 / 5 / 37 |
| hockey | 156 | 1 / 5 / 37 |
| image_and_language | 95 | 1 / 5 / 37 |
| simpson_episodes | 164 | 1 / 5 / 37 |
| world | 73 | 1 / 5 / 37 |
| ice_hockey_draft | 69 | 1 / 5 / 37 |
| professional_basketball | 113 | 1 / 5 / 37 |
| music_platform_2 | 40 | 1 / 4 / 36 |
| software_company | 56 | 1 / 4 / 36 |
| disney | 82 | 1 / 4 / 36 |
| talkingdata | 118 | 1 / 4 / 36 |
| computer_student | 53 | 1 / 4 / 36 |
| food_inspection | 65 | 1 / 4 / 36 |
| **total** | **1,530** | **16 / 64 / 512** |

The training pool has 5,071 questions over 55 databases.

**Inventory** (`contracts.INVENTORY`): within each development database, questions are ordered by
`sha256("ti-tnn-scaling-dev-inventory-v1:" + task_id)`. Rounds then take one question from each database, in split
order, skipping exhausted databases. The first 512 are the inventory, and its first 64 and first 16 are the nested
subsets. Before ordering, each of the 1,530 development gold queries was run once with the scorer's executor and its
30 s timeout. Timeouts and slow items from the parallel pass were re-run alone, and that run is final. 1,529 ran, and
one was excluded:

| Excluded | Database | Reason |
|---|---|---|
| `bird23tf-03541` | food_inspection | timeout: exceeded 30 s (both in the parallel pass and alone) |

Every remaining development gold finishes in under 5 s (the slowest takes 4.3 s, on talkingdata), and none returns an
empty result. Per-question details are in `runs/ti_tnn_scaling/data/gold_check_dev_v1.json` (git-ignored). The run
used SQLite 3.45.1 and Python 3.12.3.

## The schema text

`schema.load_schema(db_id)` reads the snapshot: `sqlite_master` order, `PRAGMA table_info` and `PRAGMA
foreign_key_list`. It merges in BIRD's `database_description/*.csv` and `train_column_meaning.json` (key
`db|table|column`, matched case-insensitively). `schema.serialize` then writes one line per item:

```
DATABASE world
TABLE City
  ID INTEGER PK -- the unique id for the city | meaning: The unique integer identifier for each city.
  Name TEXT -- the name of the city | meaning: stores text entries representing city names, such as 'Yuyao', ...
```

- A column line is the name, the declared type, `PK`, `FK -> table.column`, and after ` -- ` the description, then
  `values: ...` (BIRD's value description), then `meaning: ...` (the column-meaning summary), separated by ` | `.
- A description that only repeats the column's name is dropped. The meaning loses phrases that only restate its table
  and database (`compact_meaning`).
- Names that are not plain identifiers, or that are SQLite keywords, are quoted with backticks.
- Every table and column has a character span `[start, end)`. It covers the item's line without the indentation,
  `TABLE ` or the newline, and starts with the (quoted) name.

**Size** (Qwen3.5-4B tokenizer, about 4.2 characters per token). Without a budget, the median schema is 1,570
tokens and 65 of 69 are under 6,000. The largest are `works_cycles` (26,049), `hockey` (11,480, development),
`legislator` (8,641) and `professional_basketball` (7,495, development). Under a budget, `serialize` keeps every
column: it shortens the meaning summaries first, then the remaining descriptions. The suggested prompt budget is
`schema.PROMPT_BUDGET_CHARS = 24,000` characters. Under it every schema is at most 5,940 tokens, and only 5 change
(those four and `mondial_geo`). `describe_schema` uses that budget by default.

**Description quirks handled.** Of the 522 description CSVs, 511 are UTF-8 (64 with a byte-order mark) and 11 are
cp1252 (smart quotes, bullets, `é`). Four file names contain spaces (`Sales Orders.csv`), one header starts with an
empty field, one ends with one, and one has extra `Unnamed` columns. Three files do not match a table by name. They are matched by a unique leftover table: by suffix,
`googleplaystore*.csv` to `playstore` and `user_reviews` (app_store), and by 90% similarity, `filed_for_bankruptcy.csv`
to the misspelled table `filed_for_bankrupcy` (student_loan). Rows that describe columns absent from the database
(35 in `retail_world`, 1 in `legislator`, 1 empty row in `donor`) are left out, and `schema.description_report`
lists them.

## The tools

All three are in `tools.py`, take the request's `Snapshot` and return a JSON-serializable dict with `tool`, `args`,
`db_id`, `snapshot_sha256`, `status` (`ok` | `error` | `timeout` | `rejected`), `error` and `elapsed_seconds`.

| Tool | Also returns | Limits |
|---|---|---|
| `describe_schema(snap, tables=None)` | `text`, `tables`, `unknown_tables` | budget `PROMPT_BUDGET_CHARS` unless given |
| `lookup_values(snap, table, column, pattern=None, limit=20)` | `values` (distinct, non-NULL, most frequent first, ties by value), `truncated` | `limit` <= 20; `pattern` is a plain substring bound as a parameter, matched with LIKE (ASCII case-insensitive) |
| `execute_select(snap, sql, preview_rows=20)` | `columns`, `rows` (the preview), `row_count` (all rows), `truncated`, `timeout_seconds` | one SELECT / WITH ... SELECT / set operation |

- **Read-only, three ways.** The connection is `file:<path>?mode=ro&immutable=1` with `PRAGMA query_only=ON` and
  ATTACH disabled. `execute_select` also parses with sqlglot 30.24.0 (`read="sqlite"`): it accepts exactly one
  statement whose root is a SELECT or a set operation, and rejects DDL, DML, PRAGMA, ATTACH, VALUES and multiple
  statements. An SQLite authorizer then allows only reads and function calls. All 6,601 training gold queries pass the
  parse check.
- **Binding.** A call first checks the file's byte count and sha256 against the pin, and rejects a mismatch. The full
  hash runs once per process and file. After that, a change in size, mtime, inode or device forces a re-hash.
- **Timeout.** 30 s of wall clock (`Limits.sql_timeout_seconds`), enforced by a progress handler every 10,000
  SQLite instructions. A single long operation, such as one large sort, can overrun it by its own length.
- **Previews are for reading only.** Cells longer than 300 characters are cut, and blobs show as `<blob N bytes>`.
  The scorer never sees previews.

## The scorer

`scorer.score(snapshot, predicted_sql, gold_sql)` replicates `calculate_ex` at the pinned commit:
`correct = int(set(fetchall(predicted)) == set(fetchall(gold)))`. Errors score 0, and so does exceeding 30 s. The
official harness wraps the pair (connect, predicted query, gold query, comparison) in one `func_timeout`, so this
scorer uses one deadline for both, in the same order. The result records `correct`, a `status`, each side's status,
row count, distinct row count and seconds, the timeout, `gold_empty`, the SQLite version and the snapshot hash. A test
compiles the official `calculate_ex` and `execute_sql` from the pinned, hash-checked source and checks agreement on 10
synthetic cases. The source is cached under `runs/ti_tnn_scaling/data/raw/official_scorer/`; without the cache or a
network, the test skips.

Where the official rule surprises people (each one is tested on a synthetic database):

| The rule does | So |
|---|---|
| ignore row order | `ORDER BY x DESC` and `ASC` score the same, even when the question asks for a top 3 |
| collapse duplicates | `SELECT name` equals `SELECT DISTINCT name` |
| compare Python values | `1 == 1.0`, so INTEGER 1 matches REAL 1.0; text `'1'` does not match 1 |
| treat NULL as `None` | `(None,)` equals `(None,)` |
| compare whole rows by position | column order matters and column names do not |
| compare floats exactly | `0.1 + 0.2` does not match `0.3` |
| accept any empty result for an empty gold | even a blank or comment-only string scores 1 (no development gold is empty) |

The scorer deviates from the official harness in four deliberate ways:
1. It connects read-only, so writes fail and score 0. In the official harness, a predicted `DELETE` runs inside an
   uncommitted transaction that the gold query then reads, and DDL autocommits.
2. An abstention (`None`) scores 0.
3. A timeout interrupts the statement instead of abandoning the thread.
4. It uses the host's SQLite (3.45.1), whose functions and rounding can differ from the official runs.

## Retrieval

`retrieval.retrieve(task, k=3)` ranks training-pool questions (5,071) by Okapi BM25 (k1 = 1.2, b = 0.75; Robertson
and Zaragoza, 2009). Each document is a question plus its evidence, and the query is the request's question plus its
evidence. Tokens are lowercase alphanumeric runs minus a short stopword list. Ties are broken by task_id. The
request itself and every question from the request's own database are excluded. For development requests this
changes nothing, since their databases are not in the pool. For training requests it keeps the examples
cross-database, as they are at development time. `Index.identity` hashes the parameters and the ordered
`(task_id, content_sha256)` of the indexed questions. The index builds in about 0.05 s, and retrieval for all 512
inventory requests takes about 1 s. Every one of them gets 3 positive-score examples.

## The no-fit control L

`baseline_l.run(task, snapshot)` re-aims each of the 3 retrieved training queries at the request's database, in rank
order:
1. Parse the template SQL with sqlglot. Resolve each column in its own SELECT scope, using the source database's
   schema.
2. Choose up to 2 table assignments. These are exhaustive over each table's 6 best candidates when the template has 3
   tables or fewer, and greedy otherwise. Table scores combine lexical overlap with the request's question and
   evidence, the fit of the source columns, and a bonus when a source join lands on a target foreign key.
3. Map columns greedily, with each target column used once where possible. Join columns follow foreign keys, and a
   column named by the evidence in `<column> = <value>` wins its comparison.
4. Refill each compared literal: first the evidence's value for that column, then quoted strings, then numbers from
   the request, then a capitalized phrase confirmed by `lookup_values` (at most 2 lookups per template). Literals that
   cannot be grounded keep their template value, and the candidate is marked `ungrounded`.
5. Rename derived-table outputs and `USING` columns consistently, and qualify columns that the mapping made
   ambiguous.
6. Run the candidate with `execute_select`.

L submits the first candidate that returns at least one row. Failing that, it submits the first that ran, and if none
ran it abstains with the reasons. Tool calls are capped at `Limits.max_tool_calls` (8). Every attempt is recorded: the
template, the mappings, the fills, the SQL and each tool call. L receives no `Gold`. A test replaces `data.gold` and
`data.row` with functions that fail, and L still runs.

**Development check (not a frozen result).** Its mechanics were developed on 200 training-pool requests (the first
200 of the smoke order; templates from other databases). There, every request produced SQL that ran, and 15 of 200
were correct. One mechanics bug, columns made ambiguous by the mapping, was first seen on the development run, and its
fix is general. After the fix, L was run once on the 512-question inventory: 1 of 16, 4 of 64 and 26 of 512 were
correct (5.1%). There were no exceptions, no candidate errors and no abstentions, and 28 candidate executions timed
out. 125 submitted queries kept at least one ungrounded literal. Request time is a median of 0.02 s and at most 122 s
(several 30 s timeouts). L is a floor and is not tuned; these numbers describe its mechanics, not a comparison.

## Tests

`tests/test_ti_tnn_scaling_{data,tools,scorer,baseline_l}.py` build tiny SQLite databases inside the tests and run on
CPU in seconds. The checks that need the downloaded release or databases skip when those are absent, as in CI. The
checks of the committed pins run wherever the pins exist.

## What this does not establish

- Passing tests show that the mechanics behave as specified on the cases tested. They are not evidence about any arm's
  accuracy.
- EX is BIRD's outcome measure, with the surprises above. It does not check that a query is the right one for the
  question, only that its result set matches the gold's. BIRD's gold SQL and evidence also contain known errors.
- The development databases come from BIRD's training release. Results on them say nothing about the reserved
  Mini-Dev questions, which no arm has seen.
