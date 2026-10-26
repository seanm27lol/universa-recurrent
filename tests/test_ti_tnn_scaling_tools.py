"""The shared read-only tools and the schema text (research/ti_tnn/scaling/tools.py, schema.py), on tiny synthetic
SQLite databases built here. The checks against the real downloaded databases skip when they are absent (CI)."""
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
from scaling import schema, tools  # noqa: E402


def make_db(root: Path, db_id: str = "shop") -> Path:
    """customers(id PK, name, city) <- orders(id PK, customer_id FK, total REAL, `Date placed` TEXT)."""
    folder = root / db_id
    (folder / "database_description").mkdir(parents=True)
    path = folder / f"{db_id}.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), total REAL,
                             "Date placed" TEXT);
    """)
    conn.executemany("INSERT INTO customers VALUES (?, ?, ?)",
                     [(1, "Ann", "Paris"), (2, "Bo", "Paris"), (3, "Cy", "Oslo"), (4, "O'Neil", "Rome"),
                      (5, "50%_off", "Rome")])
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)",
                     [(i, 1 + i % 3, 10.0 * i, f"2020-01-{i:02d}") for i in range(1, 31)])
    conn.commit()
    conn.close()
    # BIRD-style descriptions: a cp1252 file with a byte-order-free header and a file whose name differs in case
    (folder / "database_description" / "Customers.csv").write_bytes(
        "original_column_name,column_name,column_description,data_format,value_description\n"
        "id,customer id,the customer's id,integer,\n"
        "name ,,full name of the customer,text,\n"
        "city,city,city where the customer lives (café),text,\"e.g. Paris\nor Oslo\"\n".encode("cp1252"))
    (folder / "database_description" / "orders.csv").write_text(
        "﻿original_column_name,column_name,column_description,data_format,value_description\n"
        "Date placed,date placed,,text,YYYY-MM-DD\n"
        "nonexistent,,,,\n", encoding="utf-8")
    return path


@pytest.fixture()
def shop(tmp_path):
    path = make_db(tmp_path)
    meaning = tmp_path / "meaning.json"
    meaning.write_text(json.dumps({"shop|orders|total": "The 'total' column in the 'orders' table of the 'shop' "
                                                        "database stores the order total in euros."}))
    snap = tools.snapshot_for_file("shop", path)
    loaded = schema.load_schema("shop", snapshot=snap, descriptions_root=tmp_path, meaning_path=meaning)
    return snap, loaded, tmp_path, meaning


# --- schema --------------------------------------------------------------------------------------------------------
def test_schema_structure_descriptions_and_quirks(shop):
    snap, loaded, root, meaning = shop
    assert [t.name for t in loaded.tables] == ["customers", "orders"]
    customers, orders = loaded.tables
    assert [c.name for c in customers.columns] == ["id", "name", "city"]
    assert customers.columns[0].primary_key and not customers.columns[1].primary_key
    assert orders.foreign_keys == (("customer_id", "customers", "id"),)
    # cp1252 decoded, newline kept in the structure but cleaned to one space; trailing-space name matched
    assert customers.columns[2].description == "city where the customer lives (café)"
    assert customers.columns[2].value_description == "e.g. Paris or Oslo"
    assert customers.columns[1].description == "full name of the customer"
    assert orders.columns[3].value_description == "YYYY-MM-DD"
    assert orders.columns[2].meaning.startswith("The 'total' column")
    report = schema.description_report("shop", snapshot=snap, descriptions_root=root, meaning_path=meaning)
    assert report["files"] == {"Customers.csv": "cp1252", "orders.csv": "utf-8-sig"}
    assert report["unmatched_rows"] == ["orders.csv:nonexistent"]


def test_serialization_spans_and_format(shop):
    _snap, loaded, _root, _meaning = shop
    text = schema.serialize(loaded)
    assert text.text.splitlines()[0] == "DATABASE shop"
    assert "  customer_id INTEGER FK -> customers.id\n" in text.text
    assert "  `Date placed` TEXT -- values: YYYY-MM-DD\n" in text.text
    assert "-- meaning: stores the order total in euros." in text.text  # boilerplate opening removed
    assert "  id INTEGER PK -- the customer's id\n" in text.text
    kinds = [(i.kind, i.table, i.column) for i in text.items]
    assert kinds[0] == ("table", "customers", "") and len(kinds) == 2 + 3 + 4
    for item in text.items:
        span = text.text[item.start:item.end]
        assert span.startswith(schema.display_name(item.column or item.table))
        assert "\n" not in span and text.text[item.end] == "\n"
    assert schema.serialize(loaded) == text  # deterministic


def test_serialization_budget_truncates_descriptions_never_columns(shop):
    _snap, loaded, _root, _meaning = shop
    full = schema.serialize(loaded)
    bare = schema.serialize(loaded, budget_chars=1)  # nothing fits: bare structure, every column kept
    assert len(bare.items) == len(full.items) and " -- " not in bare.text
    budget = len(full.text) - 20
    cut = schema.serialize(loaded, budget_chars=budget)
    assert len(cut.text) <= budget and len(cut.items) == len(full.items) and schema.ELLIPSIS in cut.text
    # the generated meaning is shortened first; BIRD's own descriptions stay whole
    assert "full name of the customer" in cut.text and "values: e.g. Paris or Oslo" in cut.text
    assert "stores the order total in euros." not in cut.text
    tighter = schema.serialize(loaded, budget_chars=len(schema._render(loaded, meaning_cap=0).text) - 10)
    assert "meaning:" not in tighter.text and schema.ELLIPSIS in tighter.text
    for item in cut.items:
        assert cut.text[item.start:item.end].startswith(schema.display_name(item.column or item.table))


def test_description_files_fall_back_to_a_unique_leftover_table(tmp_path):
    names = ["Orders.csv", "googleplaystore.csv", "googleplaystore_user_reviews.csv", "filed_for_bankruptcy.csv",
             "unrelated.csv"]
    paths = [tmp_path / n for n in names]
    tables = ["orders", "playstore", "user_reviews", "filed_for_bankrupcy", "people"]
    found = {p.name: v for p, v in schema._match_files(paths, tables).items()}
    assert found == {"Orders.csv": ("orders", "name"), "googleplaystore.csv": ("playstore", "suffix"),
                     "googleplaystore_user_reviews.csv": ("user_reviews", "suffix"),
                     "filed_for_bankruptcy.csv": ("filed_for_bankrupcy", "similar")}
    # two equally good leftover tables: no guess
    assert schema._match_files([tmp_path / "x_items.csv"], ["items", "Items"]) == {}


def test_compact_meaning_removes_only_location_phrases():
    cases = {
        "The 'division' column in the 'divisions' table of the 'european_football_1' database stores text ids.":
            "stores text ids.",
        "In the 'AwardsMisc' table of the 'hockey' database, the 'year' column, an integer, denotes the year.":
            "denotes the year.",
        "In the 'soccer_2016' database, within the 'Batting_Style' table, the 'batting hand' column specifies it.":
            "specifies it.",
        "The 'Gross' column, an integer, records the movie's total earnings.": "records the movie's total earnings.",
        "Unique identifier for each category in the 'Categories' table of the 'public_review_platform' database.":
            "Unique identifier for each category.",
        "Values include 'NHL'.": "Values include 'NHL'.",
    }
    for raw, expected in cases.items():
        assert schema.compact_meaning(raw) == expected


def test_display_name_quotes_keywords_and_spaces():
    assert schema.display_name("movie_title") == "movie_title"
    assert schema.display_name("Date received") == "`Date received`"
    assert schema.display_name("order") == "`order`"
    assert schema.display_name("a`b") == "`a``b`"


# --- binding --------------------------------------------------------------------------------------------------------
def test_binding_rejects_a_changed_file(shop, tmp_path):
    snap, *_ = shop
    tools.bind(snap)  # verifies
    wrong = C.Snapshot(db_id=snap.db_id, path=snap.path, sha256="0" * 64, bytes=snap.bytes)
    with pytest.raises(tools.SnapshotMismatch):
        tools.bind(wrong)
    with pytest.raises(tools.SnapshotMismatch):
        tools.verify_snapshot(C.Snapshot(db_id="x", path=str(tmp_path / "missing.sqlite"), sha256="0" * 64, bytes=1))
    out = tools.execute_select(wrong, "SELECT 1")
    assert out["status"] == "error" and "SnapshotMismatch" in out["error"]


def test_reserved_databases_are_refused(tmp_path, monkeypatch):
    reserved = tmp_path / "reserved"
    path = make_db(reserved, "secret")
    monkeypatch.setattr(C, "RESERVED_DATABASES", reserved)
    with pytest.raises(tools.SnapshotMismatch):
        tools.verify_snapshot(tools.snapshot_for_file("secret", path))
    with pytest.raises(tools.SnapshotMismatch):
        tools.connect_readonly(path)


# --- execute_select -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sql", [
    "SELECT name FROM customers", "select name from customers;", "WITH c AS (SELECT city FROM customers) SELECT * FROM c",
    "SELECT city FROM customers UNION SELECT name FROM customers", "SELECT * FROM (SELECT 1 AS x)",
    "SELECT `Date placed` FROM orders LIMIT 2",
])
def test_execute_select_accepts_read_only_queries(shop, sql):
    snap, *_ = shop
    out = tools.execute_select(snap, sql)
    assert out["status"] == "ok", out["error"]


@pytest.mark.parametrize("sql", [
    "INSERT INTO customers VALUES (9, 'x', 'y')", "DELETE FROM customers", "UPDATE customers SET name = 'z'",
    "CREATE TABLE z (a)", "DROP TABLE customers", "PRAGMA table_info(customers)", "ATTACH 'other.db' AS other",
    "SELECT 1; SELECT 2", "SELECT 1; DELETE FROM customers", "BEGIN", "VACUUM", "REINDEX customers", "VALUES (1)",
    "", "   ", "-- just a comment", "SELEC name FROM", "REPLACE INTO customers VALUES (1, 'a', 'b')",
])
def test_execute_select_rejects_everything_else(shop, sql):
    snap, *_ = shop
    out = tools.execute_select(snap, sql)
    assert out["status"] == "rejected" and out["error"]
    conn = sqlite3.connect(snap.path)
    assert conn.execute("SELECT COUNT(*) FROM customers").fetchone() == (5,)
    conn.close()


def test_read_only_connection_blocks_writes_even_without_the_parse_check(shop):
    snap, *_ = shop
    conn = tools.connect_readonly(snap.path)
    for statement in ("DELETE FROM customers", "CREATE TABLE z (a)", "ATTACH ':memory:' AS m"):
        with pytest.raises(sqlite3.Error):
            conn.execute(statement)
    conn.close()


def test_preview_cap_does_not_change_the_row_count(shop):
    snap, *_ = shop
    out = tools.execute_select(snap, "SELECT id, total FROM orders ORDER BY id", preview_rows=5)
    assert out["status"] == "ok" and out["row_count"] == 30 and len(out["rows"]) == 5 and out["truncated"]
    assert out["columns"] == ["id", "total"] and out["rows"][0] == [1, 10.0]
    whole = tools.execute_select(snap, "SELECT id FROM orders", preview_rows=100)
    assert whole["row_count"] == 30 and not whole["truncated"]
    json.dumps(out)  # every tool result is JSON-serializable
    assert out["elapsed_seconds"] >= 0


def test_errors_and_timeouts_are_statuses(shop):
    snap, *_ = shop
    err = tools.execute_select(snap, "SELECT no_such_column FROM customers")
    assert err["status"] == "error" and "no_such_column" in err["error"]
    endless = ("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) "
               "SELECT COUNT(*) FROM n")
    out = tools.execute_select(snap, endless, timeout_seconds=0.3)
    assert out["status"] == "timeout" and out["row_count"] is None
    assert 0.25 < out["elapsed_seconds"] < 5


def test_display_values_are_json_safe():
    assert tools.display_value(b"\x00\x01") == "<blob 2 bytes>"
    assert tools.display_value(float("inf")) == "inf"
    long = tools.display_value("x" * 400)
    assert long.startswith("x" * 300) and long.endswith("[+100 chars]")


# --- lookup_values --------------------------------------------------------------------------------------------------
def test_lookup_values_frequency_order_patterns_and_identifiers(shop):
    snap, *_ = shop
    out = tools.lookup_values(snap, "customers", "city")
    assert out["status"] == "ok" and out["values"] == ["Paris", "Rome", "Oslo"] and not out["truncated"]
    assert tools.lookup_values(snap, "CUSTOMERS", "City", pattern="par")["values"] == ["Paris"]
    assert tools.lookup_values(snap, "customers", "name", pattern="O'Ne")["values"] == ["O'Neil"]
    # % and _ are literal characters, not wildcards
    assert tools.lookup_values(snap, "customers", "name", pattern="50%_")["values"] == ["50%_off"]
    assert tools.lookup_values(snap, "customers", "name", pattern="%")["values"] == ["50%_off"]
    assert tools.lookup_values(snap, "customers", "name", pattern="'; DROP TABLE customers; --")["values"] == []
    spaced = tools.lookup_values(snap, "orders", "Date placed", pattern="2020-01-0", limit=3)
    assert spaced["values"] == ["2020-01-01", "2020-01-02", "2020-01-03"] and spaced["truncated"]
    assert tools.lookup_values(snap, "orders", "nope")["status"] == "error"
    assert tools.lookup_values(snap, "customers", "city", limit=21)["status"] == "rejected"
    json.dumps(out)


def test_describe_schema_all_and_subset(shop, monkeypatch):
    snap, loaded, root, meaning = shop
    real_load = schema.load_schema
    monkeypatch.setattr(schema, "load_schema", lambda db_id, snapshot=None, **kw: real_load(
        db_id, snapshot=snapshot, descriptions_root=root, meaning_path=meaning))
    out = tools.describe_schema(snap)
    assert out["status"] == "ok" and out["text"] == schema.serialize(loaded).text
    sub = tools.describe_schema(snap, tables=["ORDERS", "ghost"])
    assert sub["tables"] == ["orders"] and sub["unknown_tables"] == ["ghost"] and "TABLE customers" not in sub["text"]
    assert tools.describe_schema(snap, tables=["ghost"])["status"] == "error"
    json.dumps(sub)


# --- the real snapshots (skipped without the downloaded databases) ---------------------------------------------------
def _real_pins():
    from scaling import data

    if not data.PIN_SNAPSHOTS.exists() or not (C.DATABASES / "movie_platform" / "movie_platform.sqlite").exists():
        pytest.skip("the training databases are not downloaded")
    return data


def test_real_schema_text_for_every_training_database():
    data = _real_pins()
    for db_id in data.database_ids():
        snap = data.snapshot(db_id)
        loaded = schema.load_schema(db_id, snapshot=snap)
        text = schema.serialize(loaded)
        columns = sum(len(t.columns) for t in loaded.tables)
        assert len(text.items) == len(loaded.tables) + columns
        for item in text.items:
            assert text.text[item.start:item.end].startswith(schema.display_name(item.column or item.table))
