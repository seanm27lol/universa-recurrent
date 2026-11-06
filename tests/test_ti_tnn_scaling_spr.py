"""The semantic-proposal-repair study: the typed analysis, the raw observation and the selection rule.

- **Selection** is the same for D1 and TI1 and never reads gold. The revised proposal is kept if it is valid (one
  read-only statement that executes), else the original if valid, else the policy abstains.
- **The revision seed** is shared by D1 and TI1.
- **The typed analysis** separates errors (an identifier the schema does not have, a SQLite error) from observational
  warnings (an ON predicate that references only one source, a literal not in the question, an empty result). A
  legitimate self-join on a real key gets no warning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
sys.path.insert(0, str(REPO / "tests"))
pytest.importorskip("sqlglot")
from scaling import ir  # noqa: E402
from scaling import spr_feedback as F  # noqa: E402
from scaling import spr_run as S  # noqa: E402
from scaling import spr_typed as T  # noqa: E402

COLUMNS = {"movies": ["movie_id", "title", "director", "year"], "ratings": ["rating_id", "movie_id", "score", "critic"],
           "people": ["person_id", "name", "mentor_id"]}


class _Task:
    task_id, db_id = "t1", "cinema"
    question = "Which movies directed by Ada have a score above 3?"
    evidence = "Ada refers to director = 'Ada'"


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.setattr(T, "schema_index", lambda task, snap: ir.SchemaIndex(COLUMNS))
    ok = {"status": "ok", "row_count": 2, "rows": [[1], [2]], "columns": ["title"], "truncated": False}
    return lambda sql, ex=ok: F.typed_analysis(_Task(), None, sql, ex)


def test_selection_keeps_a_valid_revision_else_a_valid_original_else_abstains():
    ok, bad = {"status": "ok"}, {"status": "error"}
    o = {"sql": "SELECT 1", "execution": ok}
    assert S.select(o, {"sql": "SELECT 2", "execution": ok}) == {"source": "revised", "sql": "SELECT 2"}
    assert S.select(o, {"sql": "SELECT 2", "execution": bad})["source"] == "original"
    assert S.select(o, {"sql": None, "execution": {"status": "no_sql"}})["source"] == "original"
    assert S.select(o, {"sql": "DELETE FROM movies", "execution": ok})["source"] == "original"   # not read-only
    assert S.select({"sql": "SELECT 1", "execution": bad}, {"sql": "SELECT 2", "execution": bad}) == \
        {"source": "abstain", "sql": None}


def test_d1_and_ti1_share_the_revision_seed_and_seeds_differ_by_kind():
    assert S.seed_of("revise", 200, "t1") == S.seed_of("revise", 200, "t1")
    assert S.seed_of("revise", 200, "t1") != S.seed_of("initial", 200, "t1") != S.seed_of("revise", 201, "t1")


def test_the_raw_observation_is_bounded():
    rows = [["x" * 500, i] for i in range(20)]
    text = F.raw_observation({"status": "ok", "row_count": 1000, "rows": rows, "columns": ["a", "b"],
                              "truncated": True})
    assert "Rows returned: 1000 (showing the first 20)" in text and len(text) <= F.FEEDBACK_CHARS + 40
    assert F.raw_observation({"status": "error", "error": "no such column: q"}).startswith("Execution status: error")


def test_a_one_source_on_predicate_is_a_warning_and_a_real_self_join_is_not(analysis):
    a = analysis("SELECT T2.title FROM ratings AS T1 INNER JOIN movies AS T2 ON T1.movie_id = T1.movie_id")
    assert a["typed"] and not a["errors"]
    assert any("references only ratings (source 0)" in w for w in a["warnings"])
    b = analysis("SELECT T2.name FROM people AS T1 INNER JOIN people AS T2 ON T1.mentor_id = T2.person_id")
    assert b["typed"] and not any("references only" in w for w in b["warnings"])


def test_unknown_identifiers_are_errors_and_observations_are_warnings(analysis):
    a = analysis("SELECT budget FROM movies")
    assert not a["typed"] and any("unknown column" in e or "unresolved column" in e for e in a["errors"])
    b = analysis("SELECT title FROM movies WHERE director = 'Bob'",
                 {"status": "ok", "row_count": 0, "rows": [], "columns": ["title"], "truncated": False})
    assert not b["errors"]
    assert any("'Bob' does not occur verbatim" in w for w in b["warnings"])
    assert any("returned no rows" in w for w in b["warnings"])
    c = analysis("SELECT title FROM movies WHERE director = 'Ada'")
    assert not any("'Ada'" in w for w in c["warnings"])                     # the evidence names it
    d = analysis("SELECT title FROM movies", {"status": "error", "error": "misuse of aggregate: MAX()"})
    assert any("misuse of aggregate" in e for e in d["errors"])
    rendered = F.render_typed(a)
    assert "Errors:" in rendered and "Warnings:" in rendered


def test_a_construct_outside_the_ir_is_an_untyped_warning_not_a_rejection(analysis):
    a = analysis("WITH x AS (SELECT title FROM movies) SELECT title FROM x")
    assert not a["typed"] and not a["errors"] and any("outside the typed representation" in w for w in a["warnings"])
