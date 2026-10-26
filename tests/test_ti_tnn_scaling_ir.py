"""tirel-1 (scaling/ir.py) and the gold-SQL converter (scaling/sql_to_ir.py).

What these tests establish, on a synthetic SQLite database (no downloaded data):
- **Semantics through the compiler.** The compiled SQL keeps SQL's NULL logic, duplicate handling and row order:
  `= NULL` matches nothing, `NOT IN (.., NULL)` matches nothing, COUNT(col) skips NULLs, DISTINCT and UNION remove
  duplicates while UNION ALL keeps them, NULLs sort first ascending and last descending, and LIMIT/OFFSET pick the
  same rows.
- **Round trips.** For each query, plan -> JSON -> plan and plan -> SQL -> plan are exact, and the compiled SQL returns
  the same rows as the original.
- **The validator agrees with SQLite.** Every rule it enforces is an error SQLite raises, and the edge cases SQLite
  accepts are accepted.
- **Unsupported constructs** are reported with stable reasons.

The coverage pin is checked for internal consistency. A short live round trip against the pinned databases runs only
when they are present.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
pytest.importorskip("sqlglot")
from scaling import contracts as C  # noqa: E402
from scaling import ir  # noqa: E402
from scaling import sql_to_ir as X  # noqa: E402

DDL = """
CREATE TABLE movies (movie_id INTEGER PRIMARY KEY, title TEXT, director TEXT, year INT, popularity REAL);
CREATE TABLE ratings (rating_id INTEGER PRIMARY KEY, movie_id INT, user_id INT, score INT, critic TEXT);
CREATE TABLE "voice-actors" ("voice-actor" TEXT, movie TEXT);
INSERT INTO movies VALUES (1, 'Alpha', 'Jo', 1999, 1.5), (2, 'Beta', 'Al', 2001, NULL), (3, 'Gamma', 'Al', 2001, 3.0),
                          (4, 'Delta', NULL, 2001, 3.0);
INSERT INTO ratings VALUES (1, 1, 10, 5, 'good'), (2, 1, 11, 3, NULL), (3, 2, 10, 4, 'meh'), (4, 3, 12, 1, 'bad'),
                           (5, 3, 12, 1, 'bad');
INSERT INTO "voice-actors" VALUES ('X', 'Beta'), ('Y', 'Gamma');
"""


@pytest.fixture(scope="module")
def db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(DDL)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def schema(db):
    return ir.SchemaIndex.from_sqlite(db)


def rows(db, sql):
    return db.execute(sql).fetchall()


def compiled_rows(db, schema, sql):
    plan = X.convert(sql, schema)
    return rows(db, ir.compile(plan)), plan


M = ir.FromItem(ir.TableSource("movies"))


def test_worked_example_compiles_to_the_documented_sql(schema):
    plan = ir.Select(from_=(M,), where=ir.Compare("=", ir.Column(0, 0, "title"), ir.text("Alpha")),
                     items=(ir.Column(0, 0, "director"),))
    assert ir.validate(plan, schema) == []
    assert ir.compile(plan) == 'SELECT "T1"."director" FROM "movies" AS "T1" WHERE "T1"."title" = \'Alpha\''
    assert X.convert("SELECT director FROM movies WHERE title = 'Alpha'", schema) == plan


# --- semantics through the compiler -----------------------------------------------------------------------------------
def test_null_semantics_survive_compilation(db, schema):
    director = ir.Column(0, 0, "director")
    eq_null = ir.Select(from_=(M,), where=ir.Compare("=", director, ir.NULL), items=(ir.Column(0, 0, "title"),))
    is_null = ir.Select(from_=(M,), where=ir.IsNull(director), items=(ir.Column(0, 0, "title"),))
    assert rows(db, ir.compile(eq_null)) == []                      # NULL = NULL is unknown, not true
    assert rows(db, ir.compile(is_null)) == [("Delta",)]
    counts = ir.Select(from_=(M,), items=(ir.Agg("COUNT", director), ir.CountStar()))
    assert rows(db, ir.compile(counts)) == [(3, 4)]                 # COUNT(col) skips NULLs
    not_in = ir.Select(from_=(M,), where=ir.InList(ir.Column(0, 0, "year"), (ir.integer(1999), ir.NULL), True),
                       items=(ir.Column(0, 0, "title"),))
    assert rows(db, ir.compile(not_in)) == []                       # NOT IN with a NULL item is never true
    for sql in ("SELECT title FROM movies WHERE director IS NOT NULL", "SELECT COALESCE(director, 'none') FROM movies",
                "SELECT NULLIF(year, 2001), IIF(popularity IS NULL, 0, 1) FROM movies"):
        got, _ = compiled_rows(db, schema, sql)
        assert got == rows(db, sql)


def test_duplicate_semantics_survive_compilation(db, schema):
    expected_rows = {"SELECT director FROM movies": 4, "SELECT DISTINCT director FROM movies": 3,
                     "SELECT critic FROM ratings UNION SELECT director FROM movies": 6,
                     "SELECT critic FROM ratings UNION ALL SELECT director FROM movies": 9}
    for sql, n in expected_rows.items():
        got, _ = compiled_rows(db, schema, sql)
        assert len(got) == n and sorted(map(repr, got)) == sorted(map(repr, rows(db, sql)))
    got, _ = compiled_rows(db, schema, "SELECT COUNT(DISTINCT director), COUNT(director) FROM movies")
    assert got == [(2, 3)]


def test_ordering_semantics_survive_compilation(db, schema):
    asc, _ = compiled_rows(db, schema, "SELECT title FROM movies ORDER BY popularity, title")
    assert asc == [("Beta",), ("Alpha",), ("Delta",), ("Gamma",)]   # NULL first ascending
    desc, _ = compiled_rows(db, schema, "SELECT title FROM movies ORDER BY popularity DESC, title DESC")
    assert desc == [("Gamma",), ("Delta",), ("Alpha",), ("Beta",)]  # NULL last descending
    for sql in ("SELECT title FROM movies ORDER BY year DESC, title LIMIT 2 OFFSET 1",
                "SELECT T1.title, COUNT(*) AS n FROM movies AS T1 JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
                "GROUP BY T1.title ORDER BY n DESC, T1.title LIMIT 2",
                "SELECT title FROM movies ORDER BY 1 DESC"):
        got, _ = compiled_rows(db, schema, sql)
        assert got == rows(db, sql)                                 # same rows in the same order


def test_arithmetic_and_cast_affinity(db, schema):
    for sql in ("SELECT 7 / 2, 7 / 2.0, 7 % 3, -7 / 2", "SELECT CAST(SUM(score) AS REAL) * 100 / COUNT(*) FROM ratings",
                "SELECT CAST('12abc' AS STRING), CAST('12abc' AS TEXT), CAST(' 3.5' AS INT), CAST('1e3' AS DATE)",
                "SELECT SUBSTR(title, 2, 3), INSTR(title, 'a'), LENGTH(title), REPLACE(title, 'a', 'o') FROM movies"):
        got, _ = compiled_rows(db, schema, sql)
        assert got == rows(db, sql)
    assert ir.affinity_of("FLOATING POINT") == "INTEGER"            # SQLite's rule order: INT is checked first
    assert ir.affinity_of("STRING") == "NUMERIC" and ir.affinity_of("VARCHAR(10)") == "TEXT"
    assert "AS NUMERIC" in X.canonical_cast_types("SELECT CAST(x AS STRING), 'CAST(y AS STRING)'")
    assert "'CAST(y AS STRING)'" in X.canonical_cast_types("SELECT CAST(x AS STRING), 'CAST(y AS STRING)'")


ROUND_TRIP = [
    "SELECT T1.title, COUNT(*) AS n FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
    "GROUP BY T1.title HAVING n > 1 ORDER BY n DESC LIMIT 1",
    "SELECT CAST(SUM(CASE WHEN T2.score > 3 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM movies T1 "
    "JOIN ratings T2 ON T1.movie_id = T2.movie_id",
    "SELECT title FROM movies WHERE year BETWEEN 2000 AND 2002 AND director NOT IN ('Jo') AND title LIKE '%a%'",
    "SELECT title FROM movies WHERE movie_id IN (SELECT movie_id FROM ratings WHERE score >= 4)",
    "SELECT T.n FROM (SELECT movie_id, COUNT(*) AS n FROM ratings GROUP BY movie_id) AS T WHERE T.n > 1",
    "SELECT title FROM movies AS T1 WHERE popularity > (SELECT AVG(popularity) FROM movies AS T2 "
    "WHERE T2.director = T1.director)",
    "SELECT DISTINCT director FROM movies UNION ALL SELECT critic FROM ratings",
    "SELECT `voice-actor` FROM `voice-actors` WHERE movie = \"Beta\"",
    "SELECT COUNT(DISTINCT T2.user_id) FROM movies AS T1 LEFT JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
    "WHERE T2.critic IS NOT NULL",
    "SELECT title FROM movies WHERE NOT (movie_id IN (1, 2)) AND NOT EXISTS (SELECT 1 FROM ratings "
    "WHERE ratings.movie_id = movies.movie_id AND score < 0)",
    "SELECT -popularity, -5, popularity - -2, STRFTIME('%Y', '2020-01-02') - year FROM movies",
    "SELECT movies.title FROM movies, ratings WHERE movies.movie_id = ratings.movie_id",
    "SELECT a.title FROM movies a JOIN movies b ON a.director = b.director AND a.movie_id < b.movie_id",
    "SELECT title FROM movies WHERE year = 2001 INTERSECT SELECT title FROM movies WHERE popularity > 2",
    "SELECT title FROM movies JOIN ratings USING (movie_id) WHERE score = 1",
]


@pytest.mark.parametrize("sql", ROUND_TRIP)
def test_round_trips_are_exact_and_outcome_equivalent(db, schema, sql):
    plan = X.convert(sql, schema)
    assert ir.from_json(ir.to_json(plan)) == plan and ir.loads(ir.dumps(plan)) == plan
    compiled = ir.compile(plan)
    assert X.convert(compiled, schema) == plan
    assert sorted(map(repr, rows(db, compiled))) == sorted(map(repr, rows(db, sql)))


# --- the validator against SQLite ----------------------------------------------------------------------------------------
REJECTED_BY_SQLITE = [
    "SELECT title FROM movies WHERE COUNT(*) > 1",                              # misuse of aggregate (WHERE)
    "SELECT MAX(COUNT(year)) FROM movies",                                      # nested aggregate
    "SELECT COUNT(*) FROM movies GROUP BY COUNT(*)",                            # aggregate in GROUP BY
    "SELECT title FROM movies HAVING year > 1",                                 # HAVING on a non-aggregate query
    "SELECT title FROM movies ORDER BY COUNT(*)",                               # aggregate ORDER BY, non-aggregate
    "SELECT (SELECT title, year FROM movies)",                                  # scalar subquery with 2 columns
    "SELECT title FROM movies UNION SELECT title, year FROM movies",            # compound arity
    "SELECT *",                                                                 # * without FROM
    "SELECT T1.title FROM movies T1 LEFT JOIN ratings T2 ON T2.user_id = T3.user_id JOIN ratings T3 ON 1",
]
ACCEPTED_BY_SQLITE = [
    "SELECT COUNT(*) FROM movies HAVING COUNT(*) > 1",                          # HAVING, aggregate SELECT, no GROUP BY
    "SELECT * FROM movies T1 JOIN ratings T2 ON T2.user_id = T3.user_id JOIN ratings T3 ON T3.movie_id = T1.movie_id",
    "SELECT director, MAX(popularity) FROM movies",                             # bare column beside an aggregate
    "SELECT (SELECT COUNT(T1.title) FROM ratings LIMIT 1) FROM movies AS T1",   # an outer query's aggregate
    "SELECT year FROM movies GROUP BY year ORDER BY COUNT(*) DESC",
    "SELECT title FROM movies WHERE 5",                                         # dynamic typing
    "SELECT title + 1, SUM(title > 'B') FROM movies",
]


@pytest.mark.parametrize("sql", REJECTED_BY_SQLITE)
def test_validator_rejects_only_what_sqlite_rejects(db, schema, sql):
    with pytest.raises(sqlite3.OperationalError):
        db.execute(sql).fetchall()
    with pytest.raises(X.Unsupported):
        X.convert(sql, schema)


@pytest.mark.parametrize("sql", ACCEPTED_BY_SQLITE)
def test_validator_accepts_what_sqlite_runs(db, schema, sql):
    expected = rows(db, sql)
    plan = X.convert(sql, schema)
    assert ir.validate(plan, schema) == []
    assert sorted(map(repr, rows(db, ir.compile(plan)))) == sorted(map(repr, expected))


def test_validator_checks_scope_and_structure(schema):
    bad = {
        "unknown column": ir.Select(from_=(M,), items=(ir.Column(0, 0, "nope"),)),
        "level": ir.Select(from_=(M,), items=(ir.Column(1, 0, "title"),)),
        "derived sibling": ir.Select(from_=(M, ir.FromItem(ir.DerivedSource(ir.Select(items=(ir.Column(1, 0, "title"),
                                                                                              ))), "CROSS")),
                                     items=(ir.CountStar(),)),
        "inner without on": ir.Select(from_=(M, ir.FromItem(ir.TableSource("ratings"), "INNER")), items=(ir.CountStar(),)),
        "offset without limit": ir.Select(from_=(M,), items=(ir.CountStar(),), offset=1),
        "function arity": ir.Select(items=(ir.Func("SUBSTR", (ir.text("a"),)),)),
        "and arity": ir.Select(items=(ir.And((ir.integer(1),)),)),
        "real literal": ir.Select(items=(ir.Literal("real", "1.50"),)),
    }
    for name, plan in bad.items():
        assert ir.validate(plan, schema), name


@pytest.mark.parametrize("sql, reason", [
    ("SELECT ROW_NUMBER() OVER (ORDER BY year) FROM movies", "window function"),
    ("SELECT title FROM movies RIGHT JOIN ratings ON movies.movie_id = ratings.movie_id", "RIGHT JOIN"),
    ("SELECT GROUP_CONCAT(title) FROM movies", "aggregate GROUP_CONCAT"),
    ("SELECT title FROM movies WHERE director IS 'Al'", "IS with a non-NULL operand"),
    ("SELECT title FROM movies WHERE title LIKE 'a!%' ESCAPE '!'", "LIKE ESCAPE"),
    ("WITH x AS (SELECT 1) SELECT * FROM x", "WITH (CTE)"),
    ("SELECT title FROM movies NATURAL JOIN ratings", "NATURAL JOIN"),
    ("SELECT nope FROM movies", "unresolved column"),
])
def test_unsupported_reasons_are_stable(schema, sql, reason):
    with pytest.raises(X.Unsupported) as info:
        X.convert(sql, schema)
    assert info.value.reason == reason


def test_coverage_pin_is_consistent():
    path = C.PINS / X.COVERAGE_PIN
    if not path.exists():
        pytest.skip("coverage pin not written yet")
    pin = json.loads(path.read_text())
    assert pin["ir_version"] == ir.IR_VERSION
    for split in ("train", "dev"):
        s = pin["splits"][split]
        assert s["converted"] + sum(s["unsupported"].values()) == s["questions"]
        assert s["plan_round_trip_exact"] == s["converted"]
        assert set(s["round_trip"]) <= {"equivalent", "gold_not_executable"}  # no converted query changed outcome
        assert sum(s["round_trip"].values()) == s["converted"]


def test_live_round_trip_on_pinned_databases():
    if not (C.DATABASES.is_dir() and (C.PINS / "snapshots_v1.json").exists()):
        pytest.skip("training databases not extracted")
    from scaling import data, scorer
    from scaling import schema as schema_mod
    for r in data.rows()[:12]:
        plan = X.convert(r.sql, schema_mod.load_schema(r.db_id))
        assert scorer.score(data.snapshot(r.db_id), ir.compile(plan), r.sql)["correct"] == 1
