"""The development scorer (research/ti_tnn/scaling/scorer.py): the official BIRD EX rule and exactly the places where
it surprises people, on synthetic SQLite databases. The agreement test loads the official `calculate_ex` and
`execute_sql` from bird-bench/mini_dev at the pinned commit (sha256 checked) and skips when that source is not
available offline or online."""
from __future__ import annotations

import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
pytest.importorskip("sqlglot")
from scaling import contracts as C  # noqa: E402
from scaling import scorer, tools  # noqa: E402


@pytest.fixture()
def snap(tmp_path):
    path = tmp_path / "people" / "people.sqlite"
    path.parent.mkdir()
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE people (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, score REAL, city TEXT);
        INSERT INTO people VALUES (1, 'Ann', 31, 1.0, 'Paris'), (2, 'Bo', 45, 2.5, NULL), (3, 'Ann', 52, 1.0, 'Oslo'),
                                  (4, 'Cy', 20, NULL, NULL);
    """)
    conn.commit()
    conn.close()
    return tools.snapshot_for_file("people", path)


def ex(snap, predicted, gold):
    return scorer.score(snap, predicted, gold)["correct"]


def test_row_order_is_ignored(snap):
    gold = "SELECT name FROM people ORDER BY age ASC LIMIT 3"
    assert ex(snap, "SELECT name FROM people ORDER BY age DESC LIMIT 3", gold) == 0  # different rows (Bo vs Cy)
    assert ex(snap, "SELECT name FROM people WHERE age < 50 ORDER BY age DESC", gold) == 1  # same rows, other order


def test_duplicates_collapse(snap):
    assert ex(snap, "SELECT DISTINCT name FROM people WHERE age > 30", "SELECT name FROM people WHERE age > 30") == 1
    assert ex(snap, "SELECT 'Ann' FROM people", "SELECT 'Ann'") == 1  # four copies equal one


def test_integer_equals_real(snap):
    assert ex(snap, "SELECT 1", "SELECT 1.0") == 1
    assert ex(snap, "SELECT COUNT(*) FROM people", "SELECT CAST(COUNT(*) AS REAL) FROM people") == 1
    assert ex(snap, "SELECT '1'", "SELECT 1") == 0  # text is not a number


def test_null_equals_null_as_none(snap):
    assert ex(snap, "SELECT city FROM people WHERE id = 4", "SELECT NULL") == 1
    assert ex(snap, "SELECT score FROM people WHERE id IN (2, 4)", "SELECT score FROM people WHERE id = 2") == 0


def test_column_order_counts_and_names_do_not(snap):
    gold = "SELECT name, age FROM people WHERE id = 1"
    assert ex(snap, "SELECT age, name FROM people WHERE id = 1", gold) == 0
    assert ex(snap, "SELECT name AS who, age AS years FROM people WHERE id = 1", gold) == 1


def test_floats_compare_exactly(snap):
    assert ex(snap, "SELECT 0.1 + 0.2", "SELECT 0.3") == 0  # 0.30000000000000004 != 0.3


def test_empty_gold_matches_any_empty_prediction(snap):
    gold = "SELECT name FROM people WHERE age > 100"
    assert ex(snap, "SELECT id FROM people WHERE id < 0", gold) == 1
    assert ex(snap, "", gold) == 1  # Python's execute("") returns no rows, as in the official harness
    assert ex(snap, "-- nothing", gold) == 1
    assert ex(snap, "SELECT id FROM people", gold) == 0
    result = scorer.score(snap, "", gold)
    assert result["gold_empty"] is True


def test_errors_timeouts_and_abstentions_score_zero(snap):
    gold = "SELECT name FROM people"
    error = scorer.score(snap, "SELECT nope FROM people", gold)
    assert error["correct"] == 0 and error["status"] == "predicted_error" and error["gold"]["status"] == "ok"
    multi = scorer.score(snap, "SELECT name FROM people; SELECT 1", gold)
    assert multi["correct"] == 0 and "one statement" in multi["predicted"]["error"]
    endless = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n"
    slow = scorer.score(snap, endless, gold, timeout_seconds=0.3)
    assert slow["correct"] == 0 and slow["status"] == "predicted_timeout" and slow["gold"]["status"] == "not_run"
    assert slow["timeout_seconds"] == 0.3 and slow["elapsed_seconds"] < 5
    gold_slow = scorer.score(snap, gold, endless, timeout_seconds=0.3)
    assert gold_slow["correct"] == 0 and gold_slow["status"] == "gold_timeout"
    abstain = scorer.score(snap, None, gold)
    assert abstain["correct"] == 0 and abstain["status"] == "abstain"


def test_writes_fail_and_leave_the_file_unchanged(snap):
    before = tools.sha256_file(snap.path)
    result = scorer.score(snap, "DELETE FROM people", "SELECT name FROM people WHERE age > 100")
    assert result["correct"] == 0 and "readonly" in result["predicted"]["error"]
    assert tools.sha256_file(snap.path) == before


def test_result_fields(snap):
    result = scorer.score(snap, "SELECT name FROM people", "SELECT name FROM people")
    assert result["correct"] == 1 and result["status"] == "correct"
    assert result["predicted"]["row_count"] == 4 and result["predicted"]["distinct_rows"] == 3
    assert result["timeout_seconds"] == C.Limits().sql_timeout_seconds
    assert result["snapshot_sha256"] == snap.sha256 and result["rule"] == C.OFFICIAL_SCORER["rule"]
    gold_alone = scorer.execute_gold(snap, "SELECT name FROM people")
    assert gold_alone["status"] == "ok" and gold_alone["row_count"] == 4


def test_scorer_rejects_a_changed_snapshot(snap):
    wrong = C.Snapshot(db_id=snap.db_id, path=snap.path, sha256="f" * 64, bytes=snap.bytes)
    with pytest.raises(tools.SnapshotMismatch):
        scorer.score(wrong, "SELECT 1", "SELECT 1")


CASES = [
    ("SELECT name FROM people ORDER BY age DESC", "SELECT name FROM people"),
    ("SELECT DISTINCT name FROM people", "SELECT name FROM people"),
    ("SELECT 1", "SELECT 1.0"),
    ("SELECT city FROM people WHERE id = 4", "SELECT NULL"),
    ("SELECT age, name FROM people", "SELECT name, age FROM people"),
    ("SELECT 0.1 + 0.2", "SELECT 0.3"),
    ("SELECT id FROM people WHERE id < 0", "SELECT name FROM people WHERE age > 100"),
    ("", "SELECT name FROM people WHERE age > 100"),
    ("SELECT name FROM people WHERE age > 40", "SELECT name FROM people WHERE age >= 45"),
    ("SELECT '1'", "SELECT 1"),
]


def test_agreement_with_the_official_source(snap, tmp_path):
    try:
        ex_source = scorer.official_source("evaluation/evaluation_ex.py")
        utils_source = scorer.official_source("evaluation/evaluation_utils.py")
    except OSError as error:  # offline and no cache; a hash mismatch (ValueError) fails instead
        pytest.skip(f"official scorer source unavailable: {error}")
    official = scorer.official_functions(ex_source, ("calculate_ex",))
    utils = scorer.official_functions(utils_source, ("connect_db", "execute_sql"), {"sqlite3": sqlite3})
    copy = tmp_path / "official_copy.sqlite"  # the official harness opens a writable connection: give it a copy
    shutil.copyfile(snap.path, copy)
    for predicted, gold in CASES:
        theirs = utils["execute_sql"](predicted, gold, str(copy), "SQLite", official["calculate_ex"])
        assert ex(snap, predicted, gold) == theirs, (predicted, gold)
    # and on raw fetchall results, without any database
    for a, b in [([(1,), (2,)], [(2,), (1,), (1,)]), ([(1,)], [(1.0,)]), ([(None,)], [(None,)]), ([], []),
                 ([(1, 2)], [(2, 1)])]:
        assert official["calculate_ex"](a, b) == int(set(a) == set(b))
