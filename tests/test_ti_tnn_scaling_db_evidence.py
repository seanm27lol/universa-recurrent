"""The database-evidence lookup on a small synthetic shop database (no BIRD data, no audited example).

- **Value:** a literal the database stores in another case, with punctuation, or abbreviated, is found in its own
  column with the method named; a verbatim literal is counted.
- **Elsewhere:** a literal its own column lacks is found in another text column of the proposal's tables.
- **Columns, tables and relationships:** a near-miss column and table are ranked; a join off the declared keys and a
  composite key joined on part are reported.
- **Representation:** currency stored as text and numbers stored as text are reported.
- **Bounds:** scan limits and instruction budgets stop deterministically and are recorded; the SQL is never changed.
- **Inputs:** `retrieve` takes the question, evidence, SQL, schema and snapshot, and nothing else.
"""
from __future__ import annotations

import inspect
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
pytest.importorskip("sqlglot")
from scaling import contracts as C  # noqa: E402
from scaling import db_evidence as E  # noqa: E402


def _schema(path) -> C.Schema:
    conn = sqlite3.connect(path)
    tables = []
    for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY rowid"):
        cols = tuple(C.SchemaColumn(table=name, name=r[1], type=r[2], primary_key=bool(r[5]),
                                    description=DESCRIPTIONS.get((name, r[1]), ""))
                     for r in conn.execute(f'PRAGMA table_info("{name}")'))
        fks = tuple((r[3], r[2], r[4]) for r in conn.execute(f'PRAGMA foreign_key_list("{name}")'))
        tables.append(C.SchemaTable(name=name, columns=cols, foreign_keys=fks))
    conn.close()
    return C.Schema(db_id="shop", snapshot_sha256="x", tables=tuple(tables))


DESCRIPTIONS = {("store", "town"): "the town where the store is", ("store", "region_code"): "region code",
                ("sale", "amount"): "the sale amount in dollars", ("prize", "prize"): "the prize name",
                ("prize", "prize_category"): "the category of the prize"}


@pytest.fixture
def shop(tmp_path):
    path = tmp_path / "shop.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE store (store_id INTEGER PRIMARY KEY, town TEXT, region_code TEXT, active TEXT);
        CREATE TABLE sale (sale_id INTEGER PRIMARY KEY, store_id INTEGER REFERENCES store(store_id),
                           amount TEXT, units TEXT);
        CREATE TABLE stores_archive (store_id INTEGER, town TEXT);
        CREATE TABLE shelf (store_id INTEGER, shelf_no INTEGER, label TEXT, PRIMARY KEY (store_id, shelf_no));
        CREATE TABLE item (item_id INTEGER PRIMARY KEY, store_id INTEGER, shelf_no INTEGER, name TEXT);
        CREATE TABLE prize (prize_id INTEGER PRIMARY KEY, store_id INTEGER REFERENCES store(store_id),
                            prize TEXT, prize_category TEXT);
        CREATE TABLE visit (visit_id INTEGER PRIMARY KEY, note TEXT);
    """)
    conn.executemany("INSERT INTO visit VALUES (?,?)", [(i, f"note {i}") for i in range(20_000)])
    conn.executemany("INSERT INTO store VALUES (?,?,?,?)",
                     [(1, "Springfield", "NTH", "Yes"), (2, "Shelbyville", "STH", "No"),
                      (3, "St. Paul", "MID", "Yes")])
    conn.executemany("INSERT INTO sale VALUES (?,?,?,?)",
                     [(i, 1 + i % 3, f"${1000 + i:,}", str(i)) for i in range(1, 40)])
    conn.executemany("INSERT INTO shelf VALUES (?,?,?)", [(s, n, f"s{s}n{n}") for s in (1, 2) for n in (1, 2)])
    conn.executemany("INSERT INTO item VALUES (?,?,?,?)", [(1, 1, 2, "pen"), (2, 2, 2, "ink")])
    conn.executemany("INSERT INTO prize VALUES (?,?,?,?)",
                     [(1, 1, "Best Window Display", "Display"), (2, 2, "Cleanest Floor", "Hygiene")])
    conn.commit()
    conn.close()
    return path, _schema(path)


def _go(shop, sql, question="Which stores?", evidence="", limits=E.Limits()):
    path, schema = shop
    return E.retrieve(question, evidence, sql, schema, str(path), limits)


def _items(pkg, kind):
    return [i for i in pkg["items"] if i["kind"] == kind]


def test_retrieve_takes_only_the_permitted_inputs():
    params = list(inspect.signature(E.retrieve).parameters)
    assert params == ["question", "evidence", "sql", "schema", "snapshot_path", "limits"]
    source = inspect.getsource(E)
    for forbidden in ("gold", "task_id", "reference", "scores", "audit"):
        assert forbidden not in source.split('"""', 2)[2].lower(), forbidden   # below the module docstring


@pytest.mark.parametrize("literal, stored, method", [
    ("springfield", "Springfield", "case/whitespace"), ("yes", "Yes", "case/whitespace"),
    ("St Paul", "St. Paul", "punctuation"), ("Springfield Town", "Springfield", "prefix")])
def test_a_differently_stored_value_is_found_in_its_own_column(shop, literal, stored, method):
    pkg = _go(shop, f"SELECT store_id FROM store WHERE town = '{literal}' OR active = '{literal}'")
    (item,) = [i for i in _items(pkg, "value") if i["candidates"] and i["candidates"][0]["value"] == stored]
    assert item["verbatim_rows"] == 0 and item["candidates"][0]["provenance"]["method"] == method
    assert item["candidates"][0]["rank"] == 1


def test_a_verbatim_value_is_counted_and_not_searched_elsewhere(shop):
    pkg = _go(shop, "SELECT store_id FROM store WHERE town = 'Springfield'")
    (item,) = _items(pkg, "value")
    assert item["verbatim_rows"] == 1 and not _items(pkg, "value_elsewhere")


def test_a_literal_in_the_wrong_column_is_found_in_the_right_one(shop):
    pkg = _go(shop, "SELECT store_id FROM prize WHERE prize_category = 'Best Window Display'",
              question="Which store won the best window display prize?")
    (item,) = _items(pkg, "value_elsewhere")
    top = item["candidates"][0]
    assert (top["provenance"]["table"], top["provenance"]["column"], top["value"]) == \
        ("prize", "prize", "Best Window Display")
    assert "prize.prize stores 'Best Window Display'" in pkg["rendered"]


def test_an_evidence_literal_with_no_column_is_searched(shop):
    pkg = _go(shop, "SELECT COUNT(*) FROM sale", evidence="the region refers to region_code = 'MID'")
    (item,) = [i for i in _items(pkg, "value_elsewhere") if i["subject"]["literal"] == "MID"]
    assert item["subject"]["source"] == "evidence/question"
    assert any(c["provenance"]["column"] == "region_code" for c in item["candidates"])


def test_nothing_found_is_reported_as_not_proof_of_absence(shop):
    pkg = _go(shop, "SELECT store_id FROM store WHERE town = 'Atlantis'")
    assert "not proof of absence" in pkg["rendered"]


def test_currency_and_numbers_stored_as_text_are_reported(shop):
    pkg = _go(shop, "SELECT SUM(amount), MAX(units) FROM sale")
    rep = {i["subject"]["column"]: i for i in _items(pkg, "representation")}
    assert rep["amount"]["text_formats"].get("thousands separators", 0) == 0
    assert rep["amount"]["text_formats"]["currency"] == 39
    assert any("not a plain number" in c["note"] for c in rep["amount"]["candidates"])
    assert any("numbers as text" in c["note"] for c in rep["units"]["candidates"])


def test_a_near_miss_table_and_column_are_ranked(shop):
    pkg = _go(shop, "SELECT town FROM store WHERE region_code = 'NTH'", question="Which towns are in the region?")
    (tables,) = _items(pkg, "table")
    assert "stores_archive" in [c["table"] for c in tables["candidates"]]
    pkg = _go(shop, "SELECT prize_id FROM prize WHERE prize_category = 'x'", question="What is the prize name?")
    near = [c["column"] for i in _items(pkg, "column") for c in i["candidates"]]
    assert ["prize", "prize"] in near


def test_joins_are_checked_against_declared_and_composite_keys(shop):
    pkg = _go(shop, "SELECT T2.amount FROM store AS T1 JOIN sale AS T2 ON T1.store_id = T2.store_id")
    (rel,) = _items(pkg, "relationship")
    assert [j["status"] for j in rel["joins"]] == ["declared"] and not rel["candidates"]
    pkg = _go(shop, "SELECT T2.name FROM shelf AS T1 JOIN item AS T2 ON T1.shelf_no = T2.shelf_no")
    (rel,) = _items(pkg, "relationship")
    (j,) = rel["joins"]
    assert j["status"] == "not declared" and "composite key (store_id, shelf_no)" in j["note"]
    assert "store_id" in j["note"].split(";")[1]


def test_the_sql_is_never_changed_and_the_result_is_deterministic(shop):
    sql = "SELECT store_id FROM store WHERE town = 'springfield'"
    a, b = _go(shop, sql), _go(shop, sql)
    strip = lambda p: {k: v for k, v in p.items() if k != "latency_seconds"}   # noqa: E731
    assert strip(a) == strip(b)
    assert sql not in a["rendered"] and "springfield" in a["rendered"]


def test_scan_limits_and_instruction_budgets_stop_and_are_recorded(shop):
    pkg = _go(shop, "SELECT 1 FROM sale WHERE units = '7x'", limits=E.Limits(distinct_scan=5))
    assert any(s["what"] == "value scan sale.units" for s in pkg["scan_limits"])
    (item,) = _items(pkg, "value")
    assert item["scan_limit_hit"] and item["scanned_distinct"] == 5
    big = "SELECT 1 FROM visit WHERE note = 'note x'"      # 20,000 rows: past one progress-handler step
    tight = _go(shop, big, limits=E.Limits(query_instructions=5_000))
    assert tight["timeouts"] and all(t["reason"] == "query instruction budget" for t in tight["timeouts"])
    again = _go(shop, big, limits=E.Limits(query_instructions=5_000))
    assert tight["timeouts"] == again["timeouts"] and tight["instructions"] == again["instructions"]
    spent = _go(shop, big, limits=E.Limits(proposal_instructions=5_000))
    assert any(t["reason"].startswith("proposal") for t in spent["timeouts"])


def test_the_payload_is_bounded_and_whole_items_are_dropped(shop):
    pkg = _go(shop, "SELECT SUM(amount) FROM sale JOIN store ON sale.store_id = store.store_id "
                    "WHERE town = 'springfield'", limits=E.Limits(payload_chars=120))
    assert pkg["payload_chars"] <= 120 and pkg["truncated"]
    assert pkg["payload_chars_untruncated"] > 120


def test_unparseable_or_missing_sql_still_returns_a_package(shop):
    for sql in (None, "SELEC broken ((("):
        pkg = _go(shop, sql, evidence="refers to town = 'Shelbyville'")
        assert pkg["proposal"]["parsed"] is False
        assert any(i["subject"]["literal"] == "Shelbyville" for i in _items(pkg, "value_elsewhere"))


def test_value_match_methods():
    assert E.value_match("Saint Petersburg", "Saint Peter.") == (0.8, "prefix")
    assert E.value_match("NO", "NOR") == (0.8, "prefix")
    assert E.value_match("G", "Gilbert") is None          # too short to be a prefix
    assert E.value_match("apple", "orange") is None


# --- the coverage comparison ------------------------------------------------------------------------------------------
from scaling import evidence_coverage as EC  # noqa: E402


def test_a_fact_counts_only_when_its_item_is_shown(shop):
    pkg = _go(shop, "SELECT store_id FROM prize WHERE prize_category = 'Best Window Display'")
    fact = {"kind": "elsewhere_candidate", "literal": "Best Window Display", "table": "prize", "column": "prize",
            "value": "Best Window Display"}
    m = EC.match_fact(fact, pkg)
    assert m["found"] and m["shown"] and m["rank"] == 1
    hidden = _go(shop, "SELECT store_id FROM prize WHERE prize_category = 'Best Window Display'",
                 limits=E.Limits(payload_chars=10))
    m = EC.match_fact(fact, hidden)
    assert m["found"] and not m["shown"]
    assert EC.classify([fact], [m], False) == "none"
    assert EC.match_fact({"kind": "unavailable", "what": "x"}, pkg)["found"] is False


def test_the_classes():
    yes, no = {"shown": True}, {"shown": False}
    assert EC.classify([1, 2], [yes, yes], False) == "complete"
    assert EC.classify([1, 2], [yes, no], False) == "partial"
    assert EC.classify([1], [no], False) == "none"
    assert EC.classify([1], [yes], True) == "unresolved benchmark disagreement"


# --- SQL scopes (db-evidence-v1.3) -------------------------------------------------------------------------------------
@pytest.fixture
def trade(tmp_path):
    path = tmp_path / "trade.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, city TEXT, credit REAL);
        CREATE TABLE suppliers (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE orders (oid INTEGER PRIMARY KEY, cid INTEGER REFERENCES customers(id), total REAL);
        INSERT INTO customers VALUES (1, 'Alice', 'Paris', 100.0);
        INSERT INTO suppliers VALUES (1, 'Vendor', 'Rome');
        INSERT INTO orders VALUES (1, 1, 50.0);
    """)
    conn.commit()
    conn.close()
    return path, _schema(path)


def _preds(pkg):
    return {(p["literal"], p["table"], p["column"]) for p in pkg["proposal"]["predicates"]}


def test_a_reused_alias_in_a_nested_query_binds_to_its_own_table(trade):
    """The review's regression: one global alias map sent 'Paris' to suppliers.city, where it is absent."""
    sql = ("SELECT x.name FROM customers AS x WHERE x.city = 'Paris' AND EXISTS "
           "(SELECT 1 FROM suppliers AS x WHERE x.city = 'Rome')")
    path, schema = trade
    assert sqlite3.connect(path).execute(sql).fetchall() == [("Alice",)]
    pkg = _go(trade, sql)
    assert _preds(pkg) == {("Paris", "customers", "city"), ("Rome", "suppliers", "city")}
    values = {(i["subject"]["literal"], i["subject"]["table"]): i["verbatim_rows"] for i in _items(pkg, "value")}
    assert values == {("Paris", "customers"): 1, ("Rome", "suppliers"): 1}
    assert not _items(pkg, "value_elsewhere") and not pkg["proposal"]["unresolved"]


def test_a_correlated_reference_binds_to_the_enclosing_scope(trade):
    pkg = _go(trade, "SELECT c.name FROM customers AS c WHERE EXISTS "
                     "(SELECT 1 FROM orders AS o WHERE o.cid = c.id AND o.total > c.credit - 60)")
    assert ["customers", "credit"] in pkg["proposal"]["numeric"]
    joins = [(tuple(j["left"]), tuple(j["right"])) for j in pkg["proposal"]["joins"]]
    assert (("orders", "cid"), ("customers", "id")) in joins
    (rel,) = _items(pkg, "relationship")
    assert [j["status"] for j in rel["joins"]] == ["declared"]
    # an unqualified column absent from the subquery's own table is looked up outward, as SQLite does
    pkg = _go(trade, "SELECT name FROM customers WHERE EXISTS (SELECT 1 FROM orders WHERE credit > 10)")
    assert ["customers", "credit"] in pkg["proposal"]["numeric"] and not pkg["proposal"]["unresolved"]


def test_an_inner_column_shadows_an_outer_one(trade):
    pkg = _go(trade, "SELECT name FROM customers WHERE id IN (SELECT id FROM suppliers WHERE city = 'Rome')")
    assert _preds(pkg) == {("Rome", "suppliers", "city")}


def test_an_ambiguous_unqualified_column_is_unresolved_and_gets_no_evidence(trade):
    pkg = _go(trade, "SELECT customers.name FROM customers JOIN suppliers ON customers.id = suppliers.id "
                     "WHERE city = 'Paris'")
    assert not _preds(pkg) and not _items(pkg, "value")
    (u,) = [u for u in pkg["proposal"]["unresolved"] if u["column"] == "city"]
    assert u["reason"].startswith("ambiguous")
    assert not any(i["subject"].get("column") == "city" for i in _items(pkg, "representation"))


def test_derived_tables_and_ctes_are_not_bound_to_stored_columns(trade):
    for sql in ("SELECT d.city FROM (SELECT city FROM customers) AS d WHERE d.city = 'Paris'",
                "WITH d AS (SELECT city FROM customers) SELECT city FROM d WHERE city = 'Paris'"):
        pkg = _go(trade, sql)
        assert ("Paris", "customers", "city") not in _preds(pkg)
        assert any("derived table" in u["reason"] for u in pkg["proposal"]["unresolved"])


def test_ordinary_joins_and_self_joins_still_bind(trade):
    pkg = _go(trade, "SELECT T1.name FROM customers AS T1 JOIN orders AS T2 ON T1.id = T2.cid WHERE T2.total > 10")
    assert ["orders", "total"] in pkg["proposal"]["numeric"] and not pkg["proposal"]["unresolved"]
    assert [(tuple(j["left"]), tuple(j["right"])) for j in pkg["proposal"]["joins"]] == \
        [(("customers", "id"), ("orders", "cid"))]
    pkg = _go(trade, "SELECT a.name FROM customers AS a JOIN customers AS b ON a.id = b.id")
    assert len(pkg["proposal"]["joins"]) == 1          # two sources of one table: a join, not a tautology
    pkg = _go(trade, "SELECT a.name FROM customers AS a WHERE a.id = a.id")
    assert not pkg["proposal"]["joins"]


def test_the_impact_report_names_changed_bindings_and_packages(trade):
    sql = ("SELECT x.name FROM customers AS x WHERE x.city = 'Paris' AND EXISTS "
           "(SELECT 1 FROM suppliers AS x WHERE x.city = 'Rome')")
    new = _go(trade, sql)
    old = {**new, "proposal": {**new["proposal"], "predicates": [
        {"table": "suppliers", "column": "city", "op": "eq", "literal": "Paris", "string": True},
        {"table": "suppliers", "column": "city", "op": "eq", "literal": "Rome", "string": True}]},
        "rendered": new["rendered"] + "\nValue 'Paris' elsewhere: ..."}
    row = lambda p: {"task_id": "t", "seed": 1, "sql_sha256": "s", "package": p}   # noqa: E731
    cov = {"cases": [{"task_id": "t", "seed": 1, "class": "none"}]}
    out = EC.impact([row(old)], [row(new)], cov, cov)
    assert out["bindings_changed"] == 1 and out["rendered_changed"] == 1 and not out["class_changes"]
    (r,) = out["rows"]
    assert list(r["bindings_changed"]) == ["predicates"] and "Paris" in r["rendered_diff"]


def test_tables_keep_the_order_the_sql_names_them(trade):
    pkg = _go(trade, "SELECT c.name FROM orders AS o JOIN customers AS c ON o.cid = c.id WHERE EXISTS "
                     "(SELECT 1 FROM suppliers AS s WHERE s.city = c.city)")
    assert pkg["proposal"]["tables"] == ["orders", "customers", "suppliers"]
    cte = _go(trade, "WITH customers AS (SELECT * FROM suppliers) SELECT name FROM customers")
    assert cte["proposal"]["tables"] == ["suppliers"]      # a CTE named like a table is not that table


@pytest.mark.parametrize("sql", [
    "SELECT customers.name FROM customers, (SELECT city FROM suppliers) AS d WHERE city = 'Paris'",
    "WITH d AS (SELECT city FROM suppliers) SELECT customers.name FROM customers, d WHERE city = 'Paris'",
    "SELECT customers.name FROM customers, (SELECT * FROM suppliers) AS d WHERE city = 'Paris'"])
def test_a_column_a_derived_table_or_cte_may_also_supply_is_ambiguous(trade, sql):
    """The second review's finding: SQLite raises 'ambiguous column name: city'; v1.2 bound it to customers."""
    path, _schema_ = trade
    with pytest.raises(sqlite3.OperationalError, match="ambiguous"):
        sqlite3.connect(path).execute(sql).fetchall()
    pkg = _go(trade, sql)
    assert not _preds(pkg) and not _items(pkg, "value")
    assert any(u["column"] == "city" and u["reason"].startswith("ambiguous") for u in pkg["proposal"]["unresolved"])


def test_a_derived_table_that_lacks_the_column_does_not_block_it(trade):
    sql = "SELECT customers.name FROM customers, (SELECT name AS n FROM suppliers) AS d WHERE city = 'Paris'"
    path, _schema_ = trade
    assert sqlite3.connect(path).execute(sql).fetchall() == [("Alice",)]
    assert _preds(_go(trade, sql)) == {("Paris", "customers", "city")}


def test_a_correlated_column_in_a_set_operation_branch_binds_outward(trade):
    sql = ("SELECT c.name FROM customers AS c WHERE c.id IN (SELECT cid FROM orders WHERE total < c.credit "
           "UNION SELECT id FROM suppliers WHERE city = c.city)")
    path, _schema_ = trade
    assert sqlite3.connect(path).execute(sql).fetchall() == [("Alice",)]
    pkg = _go(trade, sql)
    assert ["customers", "credit"] in pkg["proposal"]["columns"]
    assert ("customers", "city") in {(tuple(j["right"]) if j["left"] == ["suppliers", "city"] else tuple(j["left"]))
                                     for j in pkg["proposal"]["joins"]}
    assert not pkg["proposal"]["unresolved"]
