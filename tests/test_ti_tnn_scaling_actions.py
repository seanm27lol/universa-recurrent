"""The action grammar (scaling/actions.py): traversal, inverse, families, masks and literal grounding.

What these tests establish, on a synthetic schema (no downloaded data):
- **Exact inverse.** For plans covering joins, LEFT JOIN, derived tables, compounds, correlated and scalar subqueries,
  self-joins, CASE, functions, aggregates, HAVING and ORDER/LIMIT/OFFSET, `decode(encode(plan)) == plan`.
- **Gold actions are legal.** At every gold step, the masks contain the gold family, the operator and the pointer.
- **Illegal choices are masked.** Each rule is checked in the state where it applies: columns of invisible sources,
  aggregates in WHERE, HAVING without aggregation, `*` and extra columns in a scalar subquery, and the canonical-form
  rules.
- **The documented worked example** has the documented actions and families.
- **Literal grounding** follows the documented normalization and priority.
- **No dead ends, no plans SQLite refuses** (the review of 641ff3c). Seeded random legal walks over a schema with a
  20-column table always have a legal option until the plan is finished, and SQLite prepares every finished plan
  except the one documented fold (`x AND 0` as a GROUP BY term). Compound members stay within the item budget;
  GROUP BY and ORDER BY terms are never bare literals and never see enclosing queries.

A sample of gold queries runs through the grammar only when the training release is present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
pytest.importorskip("sqlglot")
from scaling import actions as A  # noqa: E402
from scaling import contracts as C  # noqa: E402
from scaling import ir  # noqa: E402
from scaling import sql_to_ir as X  # noqa: E402

TABLES = {"movies": ["movie_id", "title", "director", "year", "popularity", "release"],
          "ratings": ["rating_id", "movie_id", "user_id", "score", "critic"],
          "directors": ["name", "birth"]}
QUESTION = "Which movies did Jo or Al direct between 1999 and 2001, and which critic said good about Alpha?"
EVIDENCE = "Alpha refers to title = 'Alpha'; the first release is '2001-01-01';"


@pytest.fixture(scope="module")
def items():
    return A.SchemaItems.coerce(TABLES)


@pytest.fixture(scope="module")
def spans():
    return A.simple_regions(QUESTION, EVIDENCE)


def item(items, table, column=""):
    return items.table_item[table] if not column else items.column_item[(table, column)]


def test_worked_example_actions_and_families(items, spans):
    plan = X.convert("SELECT director FROM movies WHERE title = 'Alpha'", items.index)
    steps = A.trace(plan, items, spans)
    ops = [s.op for s in steps]
    assert ops == ["QUERY_SELECT", "FROM_TABLE", "END", "EQ", "COL0", "TEXT_AS_NONE", "END", "SELECT_ALL", "COL0",
                   "END", "NONE", "END", "NONE", None]
    families = [A.FAMILIES[s.family] if s.family < 4 else "STOP" for s in steps]
    assert families == ["projection", "schema", "schema", "predicate", "predicate", "predicate", "aggregate",
                        "projection", "projection", "projection", "predicate", "projection", "projection", "STOP"]
    literal = steps[5].target
    assert spans.text(*literal) == "Alpha" and literal[0] >= spans.regions[1].tokens[0][0]  # the evidence span


PLANS = [
    "SELECT T1.title, COUNT(*) FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
    "GROUP BY T1.title HAVING COUNT(*) > 1 ORDER BY COUNT(*) DESC LIMIT 1",
    "SELECT CAST(SUM(CASE WHEN T2.score > 3 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM movies T1 "
    "JOIN ratings T2 ON T1.movie_id = T2.movie_id",
    "SELECT title FROM movies WHERE year BETWEEN 1999 AND 2001 AND director NOT IN ('Jo', 'Al') AND title LIKE '%Alpha%'",
    "SELECT title FROM movies WHERE movie_id IN (SELECT movie_id FROM ratings WHERE critic = 'good')",
    "SELECT T.n FROM (SELECT movie_id, COUNT(*) AS n FROM ratings GROUP BY movie_id) AS T WHERE T.n > 1",
    "SELECT title FROM movies AS T1 WHERE popularity > (SELECT AVG(popularity) FROM movies AS T2 "
    "WHERE T2.director = T1.director)",
    "SELECT DISTINCT director FROM movies UNION ALL SELECT critic FROM ratings",
    "SELECT title FROM movies AS T1 WHERE NOT EXISTS (SELECT 1 FROM ratings AS T2 WHERE T2.movie_id = T1.movie_id "
    "AND T2.score < 2)",
    "SELECT a.title FROM movies a JOIN movies b ON a.director = b.director AND a.movie_id < b.movie_id",
    "SELECT title, LENGTH(title), SUBSTR(title, 1, 3), STRFTIME('%Y', release) FROM movies "
    "WHERE release = '2001-01-01' ORDER BY year DESC, title LIMIT 3 OFFSET 1",
    "SELECT COUNT(T2.rating_id) FROM movies AS T1 LEFT JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
    "WHERE T2.critic IS NULL",
    "SELECT * FROM movies WHERE title LIKE 'Alpha%' OR director = 'Jo' OR year = 1999",
    "SELECT (SELECT COUNT(*) FROM ratings) - (SELECT COUNT(*) FROM movies)",
    "SELECT IIF(popularity IS NULL, 0, 1), -popularity, COALESCE(director, title) FROM movies",
    "SELECT T1.title FROM movies AS T1 CROSS JOIN directors AS T2 WHERE T1.director = T2.name AND NOT T1.year = 2001",
]


@pytest.mark.parametrize("sql", PLANS)
def test_gold_actions_are_legal_and_decode_exactly(items, spans, sql):
    plan = X.convert(sql, items.index)
    steps = A.trace(plan, items, spans)
    for s in steps:
        assert s.legal_families[s.family]
        if s.family < 4:
            assert s.union_id in s.legal_union_ids and A.UNION[s.union_id] == (s.family, s.op)
        if s.pointer in ("table", "column"):
            assert s.target in s.pointer_legal
        elif s.pointer == "span":
            assert s.target[0] in s.pointer_legal and s.target[1] in s.end_legal
    assert steps[-1].legal_families == [False, False, False, False, True]
    actions = [s.action for s in steps[:-1]]
    assert actions == A.encode(plan, items, spans)
    assert A.decode(actions, items, spans) == plan


def drive(items, spans, ops):
    m = A.Machine(items, spans)
    for op in ops:
        m.apply(op if isinstance(op, A.Action) else A.Action(op))
    return m


def column_targets(m, k=0):
    return set(m.options()[f"COL{k}"][1])


def test_start_and_from_masks(items, spans):
    m = A.Machine(items, spans)
    assert set(m.options()) == set(A.QUERY_OPS)
    assert m.legal_families() == [False, False, False, True, False]
    with pytest.raises(A.IllegalAction):
        m.apply(A.Action("COL0", item(items, "movies", "title")))
    m.apply(A.Action("QUERY_SELECT"))
    assert set(m.options()) == {"FROM_TABLE", "FROM_DERIVED", "FROM_NONE"}
    assert not m.is_legal(A.Action("FROM_TABLE", item(items, "movies", "title")))  # a column is not a table
    assert m.is_legal(A.Action("FROM_TABLE", item(items, "movies")))


def test_where_and_having_masks(items, spans):
    m = drive(items, spans, ["QUERY_SELECT", A.Action("FROM_TABLE", item(items, "movies")), "END"])
    opts = m.options()
    assert m.top.kind == "WHERE" and "NONE" in opts
    assert not any(op.startswith("AGG_") or op == "COUNT_STAR" for op in opts)  # misuse of aggregate in WHERE
    assert column_targets(m) == {item(items, "movies", c) for c in TABLES["movies"]}  # only tables in scope
    for op in ("NONE", "END", "SELECT_ALL", A.Action("COL0", item(items, "movies", "title")), "END"):
        m.apply(op if isinstance(op, A.Action) else A.Action(op))
    assert m.top.kind == "HAVING" and set(m.options()) == {"NONE"}  # HAVING on a non-aggregate query


def test_left_join_on_sees_only_sources_to_its_left(items, spans):
    m = drive(items, spans, ["QUERY_SELECT", A.Action("FROM_TABLE", item(items, "movies")),
                             A.Action("JOIN_LEFT_TABLE", item(items, "ratings")),
                             A.Action("JOIN_INNER_TABLE", item(items, "directors")), "END"])
    assert m.top.kind == "ON" and "NONE" in m.options()  # a LEFT JOIN's ON is optional
    left = column_targets(m)
    assert item(items, "directors", "name") not in left and item(items, "ratings", "score") in left
    m.apply(A.Action("NONE"))
    assert m.top.kind == "ON" and "NONE" not in m.options()  # an INNER JOIN's ON is required
    assert item(items, "directors", "name") in column_targets(m)


def test_derived_table_cannot_see_its_siblings(items, spans):
    m = drive(items, spans, ["QUERY_SELECT", A.Action("FROM_TABLE", item(items, "movies")), "JOIN_CROSS_DERIVED",
                             "QUERY_SELECT", A.Action("FROM_TABLE", item(items, "ratings")), "END"])
    assert column_targets(m) == {item(items, "ratings", c) for c in TABLES["ratings"]}


def test_scalar_subquery_has_exactly_one_column(items, spans):
    m = drive(items, spans, ["QUERY_SELECT", A.Action("FROM_TABLE", item(items, "movies")), "END", "EQ",
                             A.Action("COL0", item(items, "movies", "year")), "SUBQ", "QUERY_SELECT",
                             A.Action("FROM_TABLE", item(items, "ratings")), "END", "NONE", "END", "SELECT_ALL"])
    assert m.top.kind == "ITEM" and "STAR" not in m.options() and "END" not in m.options()
    m.apply(A.Action("AGG_MAX", None))
    m.apply(A.Action("COL0", item(items, "ratings", "score")))
    assert m.top.kind == "ITEM" and set(m.options()) == {"END"}


def test_canonical_form_masks(items, spans):
    base = ["QUERY_SELECT", A.Action("FROM_TABLE", item(items, "movies")), "END"]
    m = drive(items, spans, base + ["NOT"])
    assert not set(m.options()) & set(A.NEGATABLE) | set(A.NEGATABLE.values()) - set(m.options()) == set()
    assert "LIKE" not in m.options() and "IS_NULL" not in m.options() and "EQ" in m.options()
    m = drive(items, spans, base + ["AND"])
    assert "AND" not in m.options() and "OR" in m.options()
    m = drive(items, spans, base + ["EQ", A.Action("COL0", item(items, "movies", "year")), "NEG"])
    opts = m.options()
    assert "NUM" not in opts and f"CONST_{A.CONSTANT_INDEX[('integer', 1)]}" not in opts and "COL0" in opts


def test_finished_machine_allows_only_stop(items, spans):
    plan = X.convert("SELECT COUNT(*) FROM movies", items.index)
    m = drive(items, spans, A.encode(plan, items, spans))
    assert m.finished and m.legal_families() == [False, False, False, False, True] and m.plan == plan
    with pytest.raises(A.IllegalAction):
        m.apply(A.Action("END"))


def test_literal_normalization_and_numbers():
    assert A.normalize_span_text("  'Alpha' ") == "Alpha"
    assert A.normalize_span_text("'Alpha';") == "Alpha"
    assert A.normalize_span_text("Inc.") == "Inc."
    assert A.normalize_span_text("\"St George's Park\"") == "St George's Park"
    assert A.number_literal("1,000") == ir.integer(1000)
    assert A.number_literal("'1998';") == ir.integer(1998)
    assert A.number_literal("3.50") == ir.real("3.5")
    assert A.number_literal("-4") == ir.integer(-4)
    assert A.number_literal("50%") is None


@pytest.mark.parametrize("value, op", [
    ("Female", "TEXT_CAP_NONE"), ("FEMALE", "TEXT_UP_NONE"), ("%female%", "TEXT_AS_ANY"), ("female%", "TEXT_AS_PRE"),
    ("Alpha", "TEXT_AS_NONE"),
])
def test_literal_grounding_forms(value, op):
    spans = A.simple_regions("How many female patients saw Alpha?", "")
    plan = ir.Select(from_=(ir.FromItem(ir.TableSource("t")),), items=(ir.Column(0, 0, "c"),),
                     where=ir.Compare("=", ir.Column(0, 0, "c"), ir.text(value)))
    acts = A.encode(plan, {"t": ["c"]}, spans)
    assert [a.op for a in acts if a.op.startswith("TEXT_")] == [op]
    assert A.decode(acts, {"t": ["c"]}, spans) == plan


def test_constants_priority_and_ungroundable_literals():
    spans = A.simple_regions("Show 1 film from 2001 in 'Alpha'", "Alpha refers to title = 'Alpha'")
    plan = ir.Select(items=(ir.integer(1), ir.text("%Y"), ir.integer(2001), ir.text("Alpha")))
    acts = A.encode(plan, {"t": ["c"]}, spans)
    lits = [a for a in acts if a.op.startswith(("CONST", "NUM", "TEXT"))]
    assert lits[0].op.startswith("CONST") and lits[1].op.startswith("CONST") and lits[2].op == "NUM"
    assert lits[3].arg[0] >= spans.regions[1].tokens[0][0]  # the evidence wins over the question
    with pytest.raises(A.NotEncodable) as info:
        A.encode(ir.Select(items=(ir.text("Omega"),)), {"t": ["c"]}, spans)
    assert info.value.reason == "literal not groundable"


def test_vocabulary_is_a_disjoint_union():
    assert len(A.UNION) == sum(len(A.FAMILY_OPS[f]) for f in range(4)) == len(set(A.UNION))
    for f in range(4):
        lo, hi = A.FAMILY_SLICES[f]
        assert [op for _, op in A.UNION[lo:hi]] == list(A.FAMILY_OPS[f])
    assert len(set(A.CONSTANTS)) == len(A.CONSTANTS)
    assert all(op in A.POINTER_KIND for _, op in A.UNION)


def test_gold_sample_through_the_grammar():
    if not (C.DATABASES.is_dir() and (C.PINS / "snapshots_v1.json").exists()):
        pytest.skip("training databases not extracted")
    from scaling import data
    from scaling import schema as schema_mod
    reasons = {}
    for r in data.rows()[::50]:
        serialized = schema_mod.serialize(schema_mod.load_schema(r.db_id))
        schema_items = A.SchemaItems(serialized.items)
        plan, error = X.try_convert(r.sql, schema_items.index)
        if error is not None:
            continue
        spans = A.simple_regions(r.question, r.evidence)
        try:
            acts = A.encode(plan, schema_items, spans)
        except A.NotEncodable as e:
            reasons[e.reason] = reasons.get(e.reason, 0) + 1
            continue
        assert A.decode(acts, schema_items, spans) == plan
    assert set(reasons) <= {"literal not groundable"}


# --- the review's mask walk (641ff3c): dead ends and plans SQLite rejects -------------------------------------------------
WIDE = {**TABLES, "wide": [f"c{i}" for i in range(20)]}  # wider than MAX_ITEMS, so `SELECT *` can outgrow a member


def _walk(m, rng):
    leaves = ("END", "NONE", "NULL", "NUM", "STAR", "COUNT_STAR", "SELECT_ALL", "SELECT_DISTINCT", "QUERY_SELECT",
              "FROM_TABLE", "FROM_NONE")

    def weight(op):
        if op in ("END", "NONE"):
            return 30.0
        if op in leaves or op.startswith(("COL", "DCOL", "CONST_", "TEXT_", "STAR_", "LIMIT_", "OFFSET_")):
            return 6.0
        return 0.15 if op not in A.SET_OP else 2.0  # compounds often, to reach the right member

    while not m.finished and m.steps < 80:
        opts = m.options()
        assert opts, f"dead end after {m.steps} steps: {[type(f).__name__ for f in m.stack]}"
        ops = list(opts)
        op = rng.choices(ops, weights=[weight(o) for o in ops])[0]
        arg = None
        if opts[op] is not None:
            kind, targets = opts[op]
            assert targets, f"{op} has no legal {kind} target"
            if kind == "span":
                start = rng.choice(list(targets))
                arg = (start, rng.choice(list(m.span_ends(op, start))))
            else:
                arg = rng.choice(list(targets))
        m.apply(A.Action(op, arg))
    return m


def test_random_legal_walks_never_dead_end_and_sqlite_accepts_the_plans(spans):
    import random
    import sqlite3
    items = A.SchemaItems.coerce(WIDE)
    con = sqlite3.connect(":memory:")
    for table, cols in WIDE.items():
        con.execute(f'CREATE TABLE "{table}" ({", ".join(chr(34) + c + chr(34) for c in cols)})')
    rng = random.Random(11)
    finished, rejected = 0, []
    for _ in range(400):
        m = _walk(A.Machine(items, spans, max_steps=80), rng)
        if not m.finished:
            continue
        finished += 1
        assert ir.validate(m.plan, items.index) == []
        sql = ir.compile(m.plan)
        try:
            con.execute("EXPLAIN " + sql).fetchall()
        except sqlite3.Error as error:
            rejected.append((str(error), sql))
    assert finished > 300
    # the one documented residual: SQLite folds a GROUP BY/ORDER BY term like `x AND 0` to the integer 0
    assert all("term out of range" in e and " AND 0" in s for e, s in rejected), rejected[:2]


def test_compound_members_stay_within_the_item_budget(spans):
    items = A.SchemaItems.coerce(WIDE)
    m = drive(items, spans, ["QUERY_UNION", "QUERY_SELECT", A.Action("FROM_TABLE", item(items, "wide")), "END",
                             "NONE", "END", "SELECT_ALL"])
    assert "STAR" not in m.options() and "COL0" in m.options()  # 20 columns would leave the right member stuck
    m = drive(items, spans, ["QUERY_UNION", "QUERY_SELECT", A.Action("FROM_TABLE", item(items, "movies")), "END",
                             "NONE", "END", "SELECT_ALL", "STAR", "END", "NONE", "QUERY_SELECT",
                             A.Action("FROM_TABLE", item(items, "wide")), "END", "NONE", "END", "SELECT_ALL"])
    assert "STAR" not in m.options()  # the right member needs exactly 6 columns
    for _ in range(5):
        m.apply(A.Action("COL0", item(items, "wide", "c0")))
        assert "END" not in m.options()
    m.apply(A.Action("COL0", item(items, "wide", "c1")))
    assert set(m.options()) == {"END"}


def test_group_and_order_terms_are_never_literals(items, spans):
    m = drive(items, spans, ["QUERY_SELECT", A.Action("FROM_TABLE", item(items, "movies")), "END", "NONE"])
    assert m.top.kind == "GROUP"
    opts = set(m.options())
    assert not opts & set(A.LITERAL_OPS) and "COL0" in opts and "END" in opts
    m.apply(A.Action("ADD"))  # inside an expression a literal is fine: GROUP BY year + 1
    assert "NUM" in m.options() or f"CONST_{A.CONSTANT_INDEX[('integer', 1)]}" in m.options()
    m = drive(items, spans, ["QUERY_SELECT", A.Action("FROM_TABLE", item(items, "movies")), "END", "NONE", "END",
                             "SELECT_ALL", A.Action("COL0", item(items, "movies", "title")), "END", "NONE",
                             "ORDER_ASC"])
    assert m.top.kind == "ORDER_EXPR" and not set(m.options()) & set(A.LITERAL_OPS)
    for sql in ("SELECT title FROM movies GROUP BY 1", "SELECT title FROM movies ORDER BY 1"):
        plan = X.convert(sql.replace(" 1", " title"), items.index)
        import dataclasses
        field = "group_by" if "GROUP" in sql else "order_by"
        literal = ir.Literal("integer", 1)
        bad = dataclasses.replace(plan, **{field: (literal,) if field == "group_by" else
                                           (ir.OrderItem(literal, False),)})
        assert any("literal" in e for e in ir.validate(bad, items.index))


def test_group_and_order_terms_cannot_see_enclosing_queries(items, spans):
    base = ["QUERY_SELECT", A.Action("FROM_TABLE", item(items, "movies")), "END", "EQ",
            A.Action("COL0", item(items, "movies", "year")), "SUBQ", "QUERY_SELECT",
            A.Action("FROM_TABLE", item(items, "ratings")), "END", "NONE"]
    m = drive(items, spans, base)
    assert m.top.kind == "GROUP" and column_targets(m) == {item(items, "ratings", c) for c in TABLES["ratings"]}
    m = drive(items, spans, base[:-1] + [A.Action("EQ"), A.Action("COL0", item(items, "ratings", "movie_id")),
                                         A.Action("COL0", item(items, "movies", "movie_id"))])
    assert m.top.kind == "GROUP"  # the subquery's WHERE did see the outer movies; its GROUP BY does not
    assert item(items, "movies", "title") not in column_targets(m)
    plan = X.convert("SELECT (SELECT COUNT(*) FROM ratings GROUP BY ratings.critic) FROM movies", items.index)
    sub = plan.items[0].query
    import dataclasses
    outer = ir.Column(1, 0, "title")
    bad = dataclasses.replace(plan, items=(ir.ScalarQuery(dataclasses.replace(sub, group_by=(outer,))),))
    assert any("enclosing query" in e for e in ir.validate(bad, items.index))
