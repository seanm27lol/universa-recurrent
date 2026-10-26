# C4 and T4: a typed recurrent controller that builds SQL programs (development, not frozen)

**Status:** implemented and tested on CPU only, with a tiny randomly initialised backbone. No controller has been
trained on the real backbone, and nothing here is a result about C4 against T4. The numbers below are about the
representation (coverage, round trips, parameter and compute counts), not about learning.

## A concrete example

The training question `bird23tf-00000` asks, of the `movie_platform` database:

> *Who is the director of the movie Sex, Drink and Bloodshed?*
> *Evidence:* Sex, Drink and Bloodshed refers to movie title = 'Sex, Drink and Bloodshed';

A controller does not write SQL text. It makes thirteen typed decisions, each a choice from a short menu that the
grammar has already filtered, and then STOP:

| Step | Action | Family | Pointer (pinned tokenizer, real context) |
|---|---|---|---|
| 1 | `QUERY_SELECT` | projection | |
| 2 | `FROM_TABLE` | schema | schema item 15: `movies` |
| 3 | `END` (no joins) | schema | |
| 4 | `EQ` (WHERE ... = ...) | predicate | |
| 5 | `COL0` | predicate | schema item 17: `movies.movie_title` |
| 6 | `TEXT_AS_NONE` | predicate | evidence tokens 438..443: `Sex, Drink and Bloodshed` |
| 7 | `END` (no GROUP BY) | aggregate | |
| 8 | `SELECT_ALL` (no DISTINCT) | projection | |
| 9 | `COL0` | projection | schema item 24: `movies.director_name` |
| 10 | `END` (one result column) | projection | |
| 11 | `NONE` (no HAVING) | predicate | |
| 12 | `END` (no ORDER BY) | projection | |
| 13 | `NONE` (no LIMIT) | projection | |
| 14 | STOP | | |

The actions rebuild a plan in the relational IR `tirel-1`. The compiler turns it into
`SELECT "T1"."director_name" FROM "movies" AS "T1" WHERE "T1"."movie_title" = 'Sex, Drink and Bloodshed'`, which is
executed and scored like any other arm's SQL.

**C4 and T4 differ in one place only.** At each step, a family classifier first picks one of the four families (or
STOP).
- **T4** then runs only that family's specialist, a small two-layer network.
- **C4** runs one shared network for every family.

Everything else is the same code: the backbone, the recurrent state, attention, masks, pointers, losses and decoding.
The question the study asks is whether T4's division into typed specialists helps it build correct programs.

## 1. The relational IR, tirel-1 (`ir.py`)

**Intuition.** A program is a tree of typed nodes:
- row sets: `Select`, `Compound`;
- the sources a SELECT reads: `TableSource`, `DerivedSource`;
- scalar expressions.

A column is never a free-text name. It is a reference `Column(level, source, name)`:
- `level` counts SELECT boundaries outward (0 is the SELECT that owns the clause, 1 the enclosing SELECT of a
  correlated subquery);
- `source` indexes that SELECT's FROM list.

So aliases are not part of the program. The compiler invents unique ones (T1, T2, ... in the grammar's traversal
order), and a correlated reference can never be shadowed.

**Worked example (`bird23tf-00001`).** The gold query

    SELECT user_id FROM lists_users WHERE user_subscriber = 1 GROUP BY user_id
    HAVING MAX(SUBSTR(list_creation_date_utc, 1, 4)) - MIN(SUBSTR(list_creation_date_utc, 1, 4)) >= 10

becomes `Select(from_=(FromItem(TableSource("lists_users")),), where=Compare("=", Column(0,0,"user_subscriber"),
Literal("integer", 1)), group_by=(Column(0,0,"user_id"),), items=(Column(0,0,"user_id"),), having=Compare(">=",
Binary("-", Agg("MAX", Func("SUBSTR", (...))), Agg("MIN", Func("SUBSTR", (...)))), Literal("integer", 10)))`. It
compiles to the same query with quoted identifiers and alias T1, and it returns the same rows.

**What the IR contains.** The node types are:

| Kind | Node types |
|---|---|
| sources and joins | INNER, LEFT and CROSS joins with arbitrary ON expressions; derived tables |
| projection and shape | `*` and `T.*`, DISTINCT, ORDER BY ASC/DESC, LIMIT/OFFSET |
| aggregation | COUNT/SUM/AVG/MIN/MAX (each optionally DISTINCT), COUNT(*), GROUP BY, HAVING |
| predicates | comparisons, AND/OR/NOT, LIKE, BETWEEN, IN (list and subquery), IS [NOT] NULL, [NOT] EXISTS |
| scalars | arithmetic and `\|\|`, CASE (searched and simple; IIF is CASE), CAST, scalar subqueries, set operations |
| functions | STRFTIME, SUBSTR, INSTR, LENGTH, REPLACE, TRIM/LTRIM/RTRIM, LOWER/UPPER, ROUND, ABS, JULIANDAY, DATE, DATETIME, TIME, CURRENT_TIMESTAMP, COALESCE, IFNULL, NULLIF, scalar MAX/MIN |

The function list is what the training release's gold SQL uses, plus the null-handling functions the work order
names.

**Typing follows SQLite** (<https://www.sqlite.org/datatype3.html>). Values are dynamically typed, so
`'abc' + 1`, `SUM(x > 0)` and `WHERE 5` are all legal, and the IR forbids none of them. There are two sorts, scalars
and row sets with an arity. CAST stores the affinity SQLite derives from the original type name (`CAST(x AS STRING)`
is NUMERIC). The validator rejects only what SQLite itself refuses to prepare:
- unresolved or invisible columns (a derived table cannot see its siblings; a LEFT JOIN's ON sees sources 0..k);
- aggregates in WHERE, ON or GROUP BY, or nested in another aggregate;
- HAVING, or an aggregate in ORDER BY, without an aggregate query;
- multi-column scalar or IN subqueries, and compound arity mismatches;
- `*` without FROM, and ORDER BY or LIMIT on a compound member.

Each rule is tested against SQLite 3.45.1 errors, and the edge cases SQLite accepts (HAVING with an aggregate SELECT
list and no GROUP BY, an INNER JOIN's ON naming a later table, bare columns beside aggregates) are tested as accepted.

**Coverage and round trip by outcome** (`pins/ir_coverage_v1.json`). `sql_to_ir.convert` parses each gold query with
sqlglot (`read="sqlite"`), resolves names against the pinned schema, and returns a validated plan or
`Unsupported(reason)`. The compiled plan is then scored against the gold query with the benchmark's scorer: EX set
equality on the pinned snapshot, one shared 30 s deadline.

| Split | Questions | Converted | Unsupported (reason: count) | Round trip equivalent | Gold over the 30 s limit |
|---|---|---|---|---|---|
| training pool (55 DBs) | 5,071 | 5,064 | RIGHT JOIN 3, window function 1, GROUP_CONCAT 1, IS with a non-NULL operand 1, unknown column 1 | 5,062 | 2 |
| development (14 DBs) | 1,530 | 1,525 | reference into a derived table with `*` 2, RIGHT JOIN 1, window function 1, GROUP_CONCAT 1 | 1,524 | 1 |

- **No converted query changed its outcome.** Every pair whose gold executes within the limit is equivalent.
- **The three slow gold queries.** Two time out even at 120 s. The third (`bird23tf-03541`) takes 38 s, and its
  compiled plan matches it under a 300 s deadline.
- **The plans round-trip exactly.** Compile then convert returns every converted plan unchanged.
- **Exact SQL text never matches** (identifiers are quoted and aliases renamed). That is a diagnostic only.

The operator inventory (operator → count, by split) is in the pin: 75 operators occur in the training pool and 58 in
development.

## 2. The action grammar (`actions.py`)

**Traversal.** Plans are built in pre-order over typed slots, in the style of ASDL and TRANX (Yin and Neubig, EMNLP
2018). The machine keeps the partial plan and an explicit stack of unfinished subproblems: a query slot, a SELECT in
one of its phases, an operator waiting for operands, a scalar slot. A SELECT is filled in this order:
1. FROM (all sources, then each join's ON);
2. WHERE;
3. GROUP BY;
4. the SELECT list;
5. HAVING;
6. ORDER BY;
7. LIMIT, then OFFSET.

So every column pointer comes after the tables it can refer to, and the HAVING and ORDER BY masks already know
whether the query aggregates. `decode(encode(plan)) == plan` for every encodable gold plan of both splits (6,188
plans).

**Families.**

| Family | Owns | Always in this family |
|---|---|---|
| schema | FROM tables, joins, join keys (the whole ON clause), derived tables | `FROM_*`, `JOIN_*` |
| predicate | WHERE/HAVING predicates, comparison operators, literals | comparisons, LIKE, IN, BETWEEN, IS NULL, EXISTS, AND/OR/NOT, `TEXT_*`, `NUM`, `CONST_*`, `NULL` |
| aggregate | aggregate functions, GROUP BY | `AGG_*`, `COUNT_STAR` |
| projection | SELECT items, DISTINCT, ORDER BY, LIMIT, set operations | `QUERY_*`, `SELECT_*`, `STAR*`, `ORDER_*`, `LIMIT_*`, `OFFSET_*` |

Context-dependent operators take the family of their clause: column pointers, arithmetic, CAST, functions, CASE,
scalar subqueries, END and NONE.

| Clause | Family |
|---|---|
| ON | schema |
| WHERE, HAVING | predicate |
| GROUP BY | aggregate |
| SELECT list, ORDER BY | projection |

Two overrides apply:
- inside a predicate's operands the context becomes predicate (schema inside ON);
- inside an aggregate's argument it becomes aggregate.

In `bird23tf-00001`, `MAX(SUBSTR(...))` is aggregate, its column is aggregate, and the constants `1`, `4` and `10` are
predicate. Over the encodable training-pool traversals the steps split as follows: schema 29,606; predicate 31,862;
aggregate 11,247; projection 34,021; STOP 4,741.

**Arguments.** Pointers address:
- schema items (`SerializedSchema.items` indices, tables or columns);
- token spans of the question and evidence (absolute token indices in the encoded context).

Closed vocabularies are folded into operators:
- operators;
- 30 (function, arity) pairs;
- 5 CAST affinities;
- LIMIT 1..20, OFFSET 0..20;
- the k-th visible occurrence of a self-joined table (`COL0`..`COL3`);
- derived-table positions (`DCOL_s_p`, s < 2, p < 8);
- 32 literal constants.

The union vocabulary has 479 operators: schema 107, predicate 152, aggregate 88, projection 132.

**Literal grounding v1.**
- **Constants.** A literal whose value is in the constant vocabulary is always `CONST_i`: integers -4, 0..10, 30, 60,
  100, 365 and 100000; reals 1.0, 1.8 and 100.0; text `''`, `,`, `$` and nine STRFTIME formats. These are the values
  whose training-pool occurrences were absent from the question and evidence at least three times (symbols and format
  codes only for text), plus 0..10.
- **Numbers.** Any other number is `NUM(span)`.
- **Text.** Any other text is `TEXT_<case>_<wildcard>(span)`.

A span decodes as follows:
- strip whitespace, one leading quote, and a trailing quote with any `;,.:!?)]` after it;
- for NUM, remove thousands separators and parse an integer or a canonical real;
- for TEXT, apply the case form (AS, CAP, UP, LO, TTL), then the wildcard (none, `%x%`, `x%`, `%x`).

The gold span is the first match: wildcard none before the others, case AS first, evidence before question, then the
shortest span and the earliest. With the pinned tokenizer, 95.9% of training-pool literals (8,929 of 9,314) and 96.9% of
development literals (2,859 of 2,950) ground. Queries with any ungroundable literal (323 training, 78 development) are unsupported for supervised
training and stay in every evaluation denominator.

| Split | Converted | Encodable (exact round trip) | Not encodable |
|---|---|---|---|
| training pool | 5,064 | 4,741 (93.6%) | 323, all "literal not groundable" |
| development | 1,525 | 1,447 (94.9%) | 78, all "literal not groundable" |

A traversal has 23.5 steps on average, STOP included (maximum 125).

**Masks.** Masks follow structure and scope only, and they are never stricter than SQLite. They enforce:
- columns of visible sources only (LEFT JOIN ON: sources 0..k; a derived table sees no siblings);
- no aggregate in WHERE, ON or GROUP BY, or inside an aggregate;
- HAVING and aggregates in ORDER BY only in an aggregate query;
- exactly one column in scalar and IN subqueries, and equal arity for a compound's right member;
- `*` only with a FROM clause;
- span ends that decode to a literal.

Three canonical-form rules mirror the converter: no AND directly inside AND (or OR in OR), no NOT directly over a
negatable predicate, and no unary minus directly over a number. Each removes a second spelling, not a query.

## 3. The frozen backbone (`encode.py`)

**Context template `context-v1`.** A chat-formatted user turn holds, in order:
1. three retrieved training examples;
2. `### Question`;
3. `### Evidence`;
4. `### Database schema` (the benchmark's `schema-text-v1`).

The backbone is causal, so this order lets every schema item's state read the question first; schema linking
happens inside the backbone.

**States and pooling.** The pinned Qwen3.5-4B reads the context once, without gradient, in BF16. The default layer is
the final one after the last RMSNorm (`layer=-1`); `layer=l` selects `hidden_states[l]`. A schema item's vector is the
mean of its tokens' states, located through the benchmark's character spans. The question and evidence tokens are the
literal pointer targets, and the last token is the request summary.

**Cache.** Only backbone states are cached, under `CACHE/encodings/<key[:2]>/<key>.safetensors`. The key is
sha256(repo, revision, layer, dtype, exact token ids). Files are written once by temporary file and atomic link,
never overwritten. Each request returns a cost record:
- tokenize seconds;
- forward or read seconds;
- tokens;
- bytes written and read;
- cache hit;
- context identity.

The key does not cover software or device, which are recorded as file metadata.

**Limits.**
- **Context length.** Over 32,768 tokens, retrieved examples are dropped first, then schema descriptions are
  shortened; failing that, the request raises `ContextTooLong`.
- **CPU forwards.** The linear-attention kernels from flash-linear-attention are Triton kernels, so a CPU forward (the
  tests) temporarily uses transformers' PyTorch reference functions.

## 4. The controllers (`controller.py`)

**The shared trunk** (d = 512, backbone width D = 2560):

| Part | Computation |
|---|---|
| memory | tokens `LN(W_t x + E_region)`, items `LN(W_i x_item + E_kind)` |
| initial state | `h0 = tanh(W_0 x_last)` |
| step input | the previous operator, its family and its pointer argument; the current slot; the parent operator; 40 partial-plan features (`Machine.features`) |
| recurrence | two GRU layers of width 512 (Cho et al., 2014) |
| cross-attention | 8 heads over tokens and items; keys and values projected once per request |
| state | `s_t = LN(W_s [h_t; c_t])` |
| family classifier | masked, over {4 families, STOP} |
| pointers | `q . K / sqrt(d)`, as in pointer networks (Vinyals et al., 2015); the span end query adds `W_e K_tok[start]` |

**Semantic modules.** Each module maps `(s_t, family)` to operator logits over the union vocabulary (the other
families' entries are -inf) and four 512-dimensional pointer queries (table, column, span start, span end).

| Arm | Module | Hidden | Trainable params: trunk + semantic = total | Semantic MACs per step |
|---|---|---|---|---|
| T4 | 4 specialists `512 -> 512 -> |V_f| + 2048`, rows grouped by family, only the chosen head runs | 512 | 11,517,701 + 5,498,847 = **17,016,548** (heads: schema 1,368,171; predicate 1,391,256; aggregate 1,358,424; projection 1,380,996) | 1.356 to 1.389 M (one head) |
| c4_active | one MLP `512 -> 512 -> 479 + 2048`, masked to the family | 512 | 11,517,701 + 1,559,007 = 13,076,708 | 1.556 M |
| c4_params | the same MLP, width matched to T4's total | 1,808 | 11,517,701 + 5,498,847 = **17,016,548** (0.00% from T4) | 5.495 M |

**Per-step compute shared by all arms**, at 3,000 context tokens and 60 schema items:
- trunk: 7.93 M MACs;
- pointer scoring: 3.33 M MACs for a span, 30.7 K for a table or column pointer;
- once per request (memory projections, keys and values, initial state): 6.42 GMACs.

All counts are weight multiply-accumulates (`controller.step_macs`, `request_macs`). The backbone forward is
separate.

**Identical between the arms:**
- trunk code, configuration and initialization (built first under the same seed);
- targets and loss terms;
- masks, pointers and decoding.

The only difference is `Controller.semantic`. The tests check this: replacing T4's semantic module with C4's
reproduces C4's losses exactly.

The arrangement resembles a sparsely-gated mixture of experts (Shazeer et al., 2017), but it is deliberately not
one:
- the gate (the family classifier) is supervised by the grammar's gold family;
- there is no load balancing, capacity factor or token-level routing;
- one head runs per step, never all heads followed by a selection.

## 5. Training (`train.py`)

**Teacher forcing.** The trunk reads the gold action prefix and the grammar state before each gold action. The loss
terms are:
- family cross-entropy (masked to the legal families);
- operator cross-entropy from the gold family's head (T4) or from the shared MLP masked to the gold family (C4);
- pointer cross-entropy: table or column, or span start plus span end given the gold start.

The weights are 1, 1 and 1. The terms are summed over the effective batch's steps and divided by its step count.

**Recipe.**
- AdamW: lr 3e-4, weight decay 0.01 on every parameter, betas (0.9, 0.999).
- Gradient-norm clipping at 1.0.
- Effective batch 32: micro-batches of 8 with gradient accumulation.
- Seed 100.

**Example order.** The stream is the concatenation of seeded per-epoch permutations, and update u takes positions
[32u, 32u + 32). Both arms see the same examples in the same order.

**Run configuration.** `config.json` is written before fitting and must match on resume. It holds the recipe, the
loss weights, the sampling rule, the controller configuration, the parameter report, the example ids and the
grammar's vocabulary summary.

**Checkpoints.** Each checkpoint holds the model, the optimizer, the update counter and the Python, NumPy and torch
RNG states. It is named `ckpt_u<update>_<sha16>.pt` by content hash, written once, and re-hashed when loaded.
Resuming reproduces the uninterrupted run exactly (tested).

**Logging.** Every update appends to `train_log.jsonl`: the loss terms, step counts, examples, gradient norm, wall
time and T4 head calls.

**Smoke selection** (`contracts.SMOKE_TRAIN`). The 1,024 candidates are taken in pinned smoke order, and those whose
gold SQL encodes under the grammar with the pinned tokenizer are kept: the first 128. Eleven were skipped, all for
ungroundable literals. The 128 ids are in `pins/ir_coverage_v1.json` under `smoke_train`.

**CPU time per update** (8 threads; real-width random states; mean 2,586 tokens and 21.9 steps per example):
- one micro-batch of 8: about 0.55 s for both T4 and c4_params;
- a full effective batch of 32 (4 x 8): about 2.2 to 2.4 s.

## 6. Decoding (`infer_controller.py`)

Decoding is greedy and legal at every step:
1. pick the best legal family;
2. run that family's semantic module (T4 groups the batch's rows by family);
3. pick the best legal operator;
4. pick the best legal pointer (for a span, the start first, then the end).

There is no top-k, no beam and no repair. The plan is validated and compiled. Abstentions carry one of these reasons:
`budget_exhausted` (256 steps by default), `invalid_plan`, `unsupported_operation`, or `context_too_long` (from
encoding).

Work accounting records, per request: steps, the family and the head executed at each step, and MACs split into
trunk, semantic and pointer, plus the per-request memory MACs. Batch-of-one decoding equals batched decoding (tested).

## 7. Deliberately absent

| Absent | Why |
|---|---|
| repair and execution feedback | `Limits.max_repairs_controller = 0` |
| top-k, beam, sampling | greedy legal decoding only |
| a mixture-of-experts rewrite of the backbone | the backbone is frozen; specialization lives only in the controller's semantic module |
| the sheaf bridge | one family per step, no bridging between heads |
| database value lookups for literals | the tools exist, but C4/T4 do not call them in this work order |
| row values, window functions, GROUP_CONCAT, RIGHT/FULL/NATURAL joins, CTEs, COLLATE, ESCAPE | outside tirel-1 (11 of 6,601 gold queries) |

## Evidence and limits

**Evidence.** These CPU tests, all synthetic and data-free, run in CI:
- `tests/test_ti_tnn_scaling_ir.py` (47);
- `tests/test_ti_tnn_scaling_actions.py` (32);
- `tests/test_ti_tnn_scaling_controller.py` (10).

They establish:
- SQL NULL, duplicate and ordering semantics through the compiler;
- IR and action round trips;
- that gold actions are legal and illegal ones masked;
- C4/T4 supervision equivalence and T4's selected-only head execution;
- batch equivalence;
- exact checkpoint reload and resume;
- c4_params parameter matching.

The coverage pin establishes conversion and outcome equivalence on the training release.

**What none of this establishes:**
- that either arm learns anything;
- that T4's specialists help;
- anything about the reserved benchmark, which was never read.

The literal grounding rule fixes what can be supervised. About 6% of questions cannot be supervised with it, and they
still count against both arms.

**Prior work.**
- Yin and Neubig, "TRANX: A Transition-based Neural Abstract Syntax Parser" (EMNLP 2018): typed pre-order action
  grammars.
- Wang et al., "The Zephyr Abstract Syntax Description Language" (1997): ASDL.
- Vinyals et al., "Pointer Networks" (NeurIPS 2015).
- Guo et al., "IRNet" (ACL 2019), and Wang et al., "RAT-SQL" (ACL 2020): intermediate representations and schema
  linking for text-to-SQL.
- Cho et al. (EMNLP 2014): the GRU.
- Shazeer et al., "Outrageously Large Neural Networks" (ICLR 2017): sparsely-gated experts.
- Li et al., "Can LLM Already Serve as A Database Interface? (BIRD)" (NeurIPS 2023): the task and the EX rule.
- SQLite's documentation of type affinity and operator precedence: <https://www.sqlite.org/lang_expr.html>.
