"""The no-fit control L (research/ti_tnn/scaling/baseline_l.py): its mechanics on synthetic databases.

A `social` database plays the training pool (its gold SQL is the template) and a `shop` database plays the request's
database. The checks: tables, columns, joins and literals are re-aimed correctly; evidence bindings, quoted strings
and looked-up phrases fill literals; the tool budget holds; every attempt is recorded; L abstains with a reason; L is
deterministic; and L never reads gold."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
pytest.importorskip("sqlglot")
from scaling import baseline_l, data, schema, tools  # noqa: E402
from scaling import contracts as C  # noqa: E402


def build(path: Path, script: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(script)
    conn.commit()
    conn.close()
    return path


@pytest.fixture()
def world(tmp_path):
    social = build(tmp_path / "social" / "social.sqlite", """
        CREATE TABLE users (user_id INTEGER PRIMARY KEY, user_name TEXT, city TEXT);
        CREATE TABLE purchases (purchase_id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(user_id),
                                price REAL);
        INSERT INTO users VALUES (1, 'Bob', 'Lyon');
    """)
    shop = build(tmp_path / "shop" / "shop.sqlite", """
        CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, customer_name TEXT, city TEXT);
        CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id),
                             price REAL);
        INSERT INTO customers VALUES (1, 'Ann', 'Paris'), (2, 'Bo', 'Oslo'), (3, 'Ann Lee', 'Rome');
        INSERT INTO orders VALUES (1, 1, 10.0), (2, 1, 25.0), (3, 2, 5.0);
    """)
    social_snap = tools.snapshot_for_file("social", social)
    shop_snap = tools.snapshot_for_file("shop", shop)
    schemas = {"social": schema.load_schema("social", snapshot=social_snap, descriptions_root=tmp_path),
               "shop": schema.load_schema("shop", snapshot=shop_snap, descriptions_root=tmp_path)}
    return shop_snap, schemas


def example(sql, question="q", task_id="bird23tf-09999", db_id="social"):
    return C.RetrievedExample(task_id=task_id, db_id=db_id, question=question, evidence="", sql=sql, score=1.0)


def run(world, question, sql_list, evidence="", **kw):
    snap, schemas = world
    task = C.Task(task_id="dev-1", db_id="shop", question=question, evidence=evidence, split="dev")
    examples = [example(sql, task_id=f"bird23tf-0000{i}") for i, sql in enumerate(sql_list)]
    return baseline_l.run(task, snap, examples=examples, target_schema=schemas["shop"],
                          source_schema=schemas.__getitem__, **kw)


def strip_elapsed(value):
    if isinstance(value, dict):
        return {k: strip_elapsed(v) for k, v in value.items() if k != "elapsed_seconds"}
    if isinstance(value, list):
        return [strip_elapsed(v) for v in value]
    return value


def test_single_table_template_with_a_quoted_value(world):
    out = run(world, "What is the city of the customer named 'Ann'?",
              ["SELECT city FROM users WHERE user_name = 'Bob'"])
    assert out["final"]["kind"] == "sql"
    assert out["final"]["sql"] == "SELECT city FROM customers WHERE customer_name = 'Ann'"
    candidate = out["attempts"][0]["candidates"][0]
    assert candidate["status"] == "ok" and candidate["row_count"] == 1
    assert candidate["fills"][0]["source"] == "quoted" and not candidate["ungrounded"]
    assert out["tool_call_count"] == 1 and out["tool_calls"][0]["tool"] == "execute_select"
    json.dumps(out)


def test_join_follows_the_target_foreign_key(world):
    sql = ("SELECT COUNT(T2.purchase_id) FROM users AS T1 INNER JOIN purchases AS T2 ON T1.user_id = T2.user_id "
           "WHERE T1.user_name = 'Bob'")
    out = run(world, "How many orders did the customer 'Ann' place?", [sql])
    assert out["final"]["sql"] == ("SELECT COUNT(T2.order_id) FROM customers AS T1 INNER JOIN orders AS T2 "
                                   "ON T1.customer_id = T2.customer_id WHERE T1.customer_name = 'Ann'")
    assert out["tool_calls"][-1]["rows"] == [[2]]


def test_evidence_binding_beats_other_candidates(world):
    out = run(world, "Which city does the shopper live in?", ["SELECT city FROM users WHERE user_name = 'Bob'"],
              evidence="the shopper refers to customer name = 'Bo'")
    assert out["final"]["sql"] == "SELECT city FROM customers WHERE customer_name = 'Bo'"
    assert out["attempts"][0]["candidates"][0]["fills"][0]["source"] == "evidence"


def test_capitalized_phrase_is_confirmed_by_lookup(world):
    out = run(world, "What is the city of the customer Ann Lee?", ["SELECT city FROM users WHERE user_name = 'Bob'"])
    assert out["final"]["sql"] == "SELECT city FROM customers WHERE customer_name = 'Ann Lee'"
    assert [c["tool"] for c in out["tool_calls"]] == ["lookup_values", "execute_select"]
    assert out["attempts"][0]["candidates"][0]["fills"][0]["source"] == "lookup"


def test_numbers_fill_numeric_literals(world):
    out = run(world, "How many orders cost more than 8?", ["SELECT COUNT(*) FROM purchases WHERE price > 100"])
    assert out["final"]["sql"] == "SELECT COUNT(*) FROM orders WHERE price > 8"
    assert out["tool_calls"][-1]["rows"] == [[2]]


def test_columns_resolve_in_their_own_subquery_scope(world):
    sql = "SELECT CAST((SELECT COUNT(user_id) FROM users) AS REAL) / (SELECT COUNT(user_id) FROM purchases)"
    out = run(world, "What is the ratio of customers to orders?", [sql])
    assert out["final"]["sql"] == ("SELECT CAST((SELECT COUNT(customer_id) FROM customers) AS REAL) / "
                                   "(SELECT COUNT(customer_id) FROM orders)")
    assert out["tool_calls"][-1]["rows"] == [[1.0]]


def test_derived_table_outputs_follow_their_renamed_select_list(world):
    sql = ("SELECT T.user_name FROM (SELECT T1.user_name, COUNT(T2.purchase_id) AS num FROM users AS T1 "
           "INNER JOIN purchases AS T2 ON T1.user_id = T2.user_id GROUP BY T1.user_name) AS T "
           "ORDER BY T.num DESC LIMIT 1")
    out = run(world, "Which customer placed the most orders?", [sql])
    assert out["final"]["sql"].startswith("SELECT T.customer_name FROM (SELECT T1.customer_name, COUNT(T2.order_id)")
    assert out["tool_calls"][-1]["rows"] == [["Ann"]]


def test_join_using_is_renamed_or_rewritten_as_on(world, tmp_path):
    sql = "SELECT COUNT(*) FROM users JOIN purchases USING (user_id) WHERE user_name = 'Bob'"
    same = run(world, "How many orders did the customer 'Ann' place?", [sql])
    assert same["final"]["sql"] == ("SELECT COUNT(*) FROM customers JOIN orders USING (customer_id) "
                                    "WHERE customers.customer_name = 'Ann'")
    assert same["tool_calls"][-1]["rows"] == [[2]]
    path = build(tmp_path / "billing" / "billing.sqlite", """
        CREATE TABLE clients (client_id INTEGER PRIMARY KEY, client_name TEXT, city TEXT);
        CREATE TABLE invoices (invoice_id INTEGER PRIMARY KEY, buyer_id INTEGER REFERENCES clients(client_id),
                               price REAL);
        INSERT INTO clients VALUES (1, 'Ann', 'Paris');
        INSERT INTO invoices VALUES (1, 1, 3.0), (2, 1, 4.0);
    """)
    snap = tools.snapshot_for_file("billing", path)
    _shop, schemas = world
    task = C.Task(task_id="dev-2", db_id="billing", question="How many invoices did the client 'Ann' get?",
                  evidence="", split="dev")
    out = baseline_l.run(task, snap, examples=[example(sql)],
                         target_schema=schema.load_schema("billing", snapshot=snap, descriptions_root=tmp_path),
                         source_schema=schemas.__getitem__)
    assert out["final"]["sql"] == ("SELECT COUNT(*) FROM clients JOIN invoices ON clients.client_id = "
                                   "invoices.buyer_id WHERE clients.client_name = 'Ann'")
    assert out["tool_calls"][-1]["rows"] == [[2]]


def test_unqualified_columns_stay_unambiguous_after_mapping(world, tmp_path):
    path = build(tmp_path / "club" / "club.sqlite", """
        CREATE TABLE members (member_id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        CREATE TABLE payments (payment_id INTEGER PRIMARY KEY, member_id INTEGER REFERENCES members(member_id),
                               price REAL, name TEXT);
        INSERT INTO members VALUES (1, 'Ann', 'Paris'), (2, 'Bo', 'Oslo');
        INSERT INTO payments VALUES (1, 1, 25.0, 'card'), (2, 2, 5.0, 'cash');
    """)
    snap = tools.snapshot_for_file("club", path)
    _shop, schemas = world
    sql = ("SELECT user_name FROM users AS T1 INNER JOIN purchases AS T2 ON T1.user_id = T2.user_id "
           "WHERE T2.price > 100")
    task = C.Task(task_id="dev-3", db_id="club", question="Which member paid more than 20?", evidence="",
                  split="dev")
    out = baseline_l.run(task, snap, examples=[example(sql)],
                         target_schema=schema.load_schema("club", snapshot=snap, descriptions_root=tmp_path),
                         source_schema=schemas.__getitem__)
    assert out["final"]["sql"] == ("SELECT T1.name FROM members AS T1 INNER JOIN payments AS T2 "
                                   "ON T1.member_id = T2.member_id WHERE T2.price > 20")
    assert out["tool_calls"][-1]["rows"] == [["Ann"]]


def test_empty_results_fall_through_to_later_templates(world):
    out = run(world, "What is the city of the customer named 'Zed'?",
              ["SELECT city FROM users WHERE user_name = 'Bob'", "SELECT city FROM users"])
    assert out["final"]["kind"] == "sql" and out["final"]["sql"] == "SELECT city FROM customers"
    assert out["attempts"][0]["candidates"][0]["row_count"] == 0


def test_abstains_with_reasons(world):
    out = run(world, "What is the city of 'Ann'?", ["SELEC nonsense FROM", "SELECT x FROM no_such_table"])
    assert out["final"]["kind"] == "abstain" and out["final"]["sql"] is None
    assert out["attempts"][0]["status"].startswith("template_unusable")
    assert out["attempts"][1]["status"] == "no_table_assignment"
    nothing = run(world, "What is the city of 'Ann'?", [])
    assert nothing["final"]["kind"] == "abstain" and "no retrieved examples" in nothing["final"]["reason"]


def test_tool_budget_is_respected(world):
    templates = ["SELECT city FROM users WHERE user_name = 'Bob'"] * 3
    out = run(world, "What is the city of the customer Nobody Here?", templates, max_tool_calls=2)
    assert out["tool_call_count"] <= 2 and out["tool_call_budget"] == 2
    default = run(world, "What is the city of the customer Nobody Here?", templates)
    assert default["tool_call_count"] <= C.Limits().max_tool_calls


def test_deterministic_and_never_reads_gold(world, monkeypatch):
    def forbidden(*_a, **_k):
        raise AssertionError("L must never read gold")
    monkeypatch.setattr(data, "gold", forbidden)
    monkeypatch.setattr(data, "row", forbidden)
    templates = ["SELECT city FROM users WHERE user_name = 'Bob'",
                 "SELECT COUNT(T2.purchase_id) FROM users AS T1 JOIN purchases AS T2 ON T1.user_id = T2.user_id"]
    first = run(world, "How many orders were placed by 'Ann'?", templates)
    second = run(world, "How many orders were placed by 'Ann'?", templates)
    assert strip_elapsed(first) == strip_elapsed(second)
    assert {c["db_id"] for c in first["tool_calls"]} == {"shop"}


def test_real_requests_use_only_training_pool_templates():
    if not data.PIN_INVENTORY.exists() or not data.TRAIN_JSONL.exists() or not data.PIN_SNAPSHOTS.exists():
        pytest.skip("the downloaded release and pins are not present")
    pool = set(data.train_pool_databases())
    for task_id in data.inventory_ids(16)[:3]:
        task = data.task(task_id)
        out = baseline_l.run(task, data.snapshot(task.db_id))
        assert all(a["template_db_id"] in pool for a in out["attempts"])
        assert {c["db_id"] for c in out["tool_calls"]} <= {task.db_id}
        assert out["tool_call_count"] <= C.Limits().max_tool_calls
