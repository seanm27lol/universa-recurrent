"""Identities, the split, the inventory, the pins and retrieval (research/ti_tnn/scaling/data.py, retrieval.py).

The ordering rules are re-derived here with hashlib, independently of data.py. Checks that need the downloaded
training release skip when it is absent (CI); checks of the committed pins run whenever the pins exist."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from scaling import contracts as C  # noqa: E402
from scaling import data, retrieval  # noqa: E402


def key(salt, item):
    return hashlib.sha256(f"{salt}:{item}".encode()).hexdigest()


# --- identities ------------------------------------------------------------------------------------------------------
def test_ids_and_content_hash():
    assert data.task_id_for(0) == "bird23tf-00000" and data.task_id_for(1234) == "bird23tf-01234"
    expected = hashlib.sha256("db\x1fq?\x1fev\x1fSELECT 1".encode()).hexdigest()
    assert data.content_sha256("db", "q?", "ev", "SELECT 1") == expected


def test_parse_rows_uses_line_index_and_refuses_gaps():
    line = json.dumps({"db_id": "d", "question": "q", "evidence": "", "SQL": "SELECT 1"})
    parsed = data.parse_rows([line, line, ""])
    assert [r.task_id for r in parsed] == ["bird23tf-00000", "bird23tf-00001"]
    assert parsed[0].task("dev") == C.Task(task_id="bird23tf-00000", db_id="d", question="q", evidence="",
                                           split="dev")
    assert parsed[0].gold() == C.Gold(task_id="bird23tf-00000", sql="SELECT 1")
    assert not hasattr(parsed[0].task("dev"), "sql")
    with pytest.raises(ValueError):
        data.parse_rows([line, "", line])


# --- ordering rules ----------------------------------------------------------------------------------------------------
def test_split_rule_matches_an_independent_derivation():
    dbs = [f"db{i}" for i in range(69)]
    dev, pool = data.split_databases(dbs)
    ordered = sorted(dbs, key=lambda d: key(C.SPLIT["salt"], d))
    assert dev == ordered[:14] and pool == ordered[14:] and not set(dev) & set(pool)
    assert data.split_databases(list(reversed(dbs))) == (dev, pool)


def test_round_robin_and_inventory_order():
    per = {"a": ["a1", "a2", "a3"], "b": ["b1"], "c": ["c1", "c2"]}
    assert data.round_robin(per, ["a", "b", "c"], 10) == ["a1", "b1", "c1", "a2", "c2", "a3"]
    assert data.round_robin(per, ["c", "a", "b"], 4) == ["c1", "a1", "b1", "c2"]
    questions = {"x": [f"x{i}" for i in range(5)], "y": [f"y{i}" for i in range(3)]}
    order = data.inventory_order(questions, ["y", "x"], excluded={"x2"}, salt="s", size=6)
    x_sorted = [t for t in sorted(questions["x"], key=lambda t: key("s", t)) if t != "x2"]
    y_sorted = sorted(questions["y"], key=lambda t: key("s", t))
    assert order == [y_sorted[0], x_sorted[0], y_sorted[1], x_sorted[1], y_sorted[2], x_sorted[2]]
    assert "x2" not in data.inventory_order(questions, ["y", "x"], excluded={"x2"}, salt="s", size=100)


def test_smoke_order():
    ids = [f"bird23tf-{i:05d}" for i in range(50)]
    assert data.smoke_order(ids) == sorted(ids, key=lambda t: key(C.SMOKE_TRAIN["salt"], t))


def test_write_pin_refuses_silent_changes(tmp_path):
    pin = tmp_path / "p.json"
    data.write_pin(pin, {"a": [1, 2]})
    data.write_pin(pin, {"a": [1, 2]})  # identical: fine
    with pytest.raises(data.PinMismatch):
        data.write_pin(pin, {"a": [2, 1]})
    data.write_pin(pin, {"a": [2, 1]}, rewrite=True)
    assert json.loads(pin.read_text()) == {"a": [2, 1]}


def test_reserved_identities_keep_only_ids_and_db_ids(tmp_path):
    records = [{"question_id": 7, "db_id": "secret_db", "question": "q", "evidence": "e", "SQL": "SELECT 1",
                "difficulty": "simple"}]
    path = tmp_path / "reserved.json"
    path.write_text(json.dumps(records))
    out = data.reserved_identities(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert out == [(7, "secret_db")]
    with pytest.raises(data.PinMismatch):
        data.reserved_identities(path, "0" * 64)


# --- retrieval ---------------------------------------------------------------------------------------------------------
def doc(task_id, db_id, question, evidence="", sql="SELECT 1"):
    return retrieval.Document(task_id=task_id, db_id=db_id, question=question, evidence=evidence, sql=sql,
                              content_sha256=data.content_sha256(db_id, question, evidence, sql))


DOCS = [doc("t1", "films", "Who directed the movie Heat?"),
        doc("t2", "films", "How many movies were released in 1995?"),
        doc("t3", "shop", "How many customers live in Paris?", "Paris refers to city = 'Paris'"),
        doc("t4", "shop", "List the names of customers from Oslo."),
        doc("t5", "zoo", "How many animals live in the zoo?")]


def test_bm25_ranking_exclusions_and_ties():
    index = retrieval.Index(DOCS)
    task = C.Task(task_id="q", db_id="other", question="How many customers live in Rome?", evidence="", split="dev")
    hits = retrieval.retrieve(task, k=3, index=index)
    assert hits[0].task_id == "t3" and len(hits) == 3 and hits[0].sql == "SELECT 1"
    assert hits[0].score > hits[1].score >= hits[2].score
    own_db = C.Task(task_id="t3", db_id="shop", question="How many customers live in Paris?", evidence="",
                    split="train")
    assert {h.db_id for h in retrieval.retrieve(own_db, k=3, index=index)} <= {"films", "zoo"}
    no_exclusion = retrieval.retrieve(own_db, k=5, index=index, exclude_same_database=False)
    assert "t3" not in [h.task_id for h in no_exclusion] and "t4" in [h.task_id for h in no_exclusion]
    nothing = C.Task(task_id="q", db_id="other", question="zzz qqq", evidence="", split="dev")
    assert [h.task_id for h in retrieval.retrieve(nothing, k=2, index=index)] == ["t1", "t2"]  # ties by id


def test_bm25_score_matches_the_formula():
    index = retrieval.Index(DOCS)
    lengths = [len(retrieval.tokens(f"{d.question} {d.evidence}")) for d in DOCS]
    avgdl = sum(lengths) / len(lengths)
    import math

    def manual(query, position):
        total = 0.0
        doc_tokens = retrieval.tokens(f"{DOCS[position].question} {DOCS[position].evidence}")
        for term in sorted(set(retrieval.tokens(query))):
            df = sum(1 for d in DOCS if term in retrieval.tokens(f"{d.question} {d.evidence}"))
            tf = doc_tokens.count(term)
            if df == 0 or tf == 0:
                continue
            idf = math.log(1 + (len(DOCS) - df + 0.5) / (df + 0.5))
            total += idf * tf * 2.2 / (tf + 1.2 * (1 - 0.75 + 0.75 * lengths[position] / avgdl))
        return total
    query = "customers live in Paris city"
    scores = index.scores(query)
    for position in range(len(DOCS)):
        assert scores.get(position, 0.0) == pytest.approx(manual(query, position), rel=1e-12)


def test_index_identity_tracks_content():
    a = retrieval.Index(DOCS)
    assert a.identity == retrieval.Index(list(reversed(DOCS))).identity
    changed = DOCS[:-1] + [doc("t5", "zoo", "How many animals live in the zoo?", sql="SELECT 2")]
    assert retrieval.Index(changed).identity != a.identity


# --- the real release and the committed pins (skipped when absent) -------------------------------------------------
def _release():
    if not data.TRAIN_JSONL.exists():
        pytest.skip("the training release is not downloaded")
    return data.rows()


def test_release_rows_and_split_pin():
    rows = _release()
    assert len(rows) == 6601 and rows[0].task_id == "bird23tf-00000" and rows[-1].task_id == "bird23tf-06600"
    assert len({r.task_id for r in rows}) == 6601
    pin = data.read_pin(data.PIN_SPLIT)
    assert pin == data.split_payload()
    assert len(pin["dev"]) == 14 and len(pin["train_pool"]) == 55 and not set(pin["dev"]) & set(pin["train_pool"])
    assert pin["dev_questions"] + pin["train_pool_questions"] == 6601


def test_inventory_pin_follows_the_rule():
    rows = _release()
    if not data.PIN_INVENTORY.exists():
        pytest.skip("no inventory pin yet")
    pin = data.read_pin(data.PIN_INVENTORY)
    dev = data.read_pin(data.PIN_SPLIT)["dev"]
    excluded = {e["task_id"] for e in pin["excluded"]}
    questions = {d: [r.task_id for r in rows if r.db_id == d] for d in dev}
    assert pin["task_ids"] == data.inventory_order(questions, dev, excluded)
    assert len(pin["task_ids"]) == pin["size"] <= 512 and not excluded & set(pin["task_ids"])
    by_id = {r.task_id: r for r in rows}
    assert pin["db_ids"] == [by_id[t].db_id for t in pin["task_ids"]] and set(pin["db_ids"]) <= set(dev)
    for n in (16, 64):
        assert data.inventory_ids(n) == pin["task_ids"][:n]


def test_smoke_pin_is_training_pool_only():
    rows = _release()
    if not data.PIN_SMOKE.exists():
        pytest.skip("no smoke pin yet")
    pin = data.read_pin(data.PIN_SMOKE)
    pool = set(data.read_pin(data.PIN_SPLIT)["train_pool"])
    by_id = {r.task_id: r for r in rows}
    assert all(by_id[t].db_id in pool for t in pin["task_ids"]) and len(pin["task_ids"]) == 1024
    assert pin["task_ids"] == data.smoke_order([r.task_id for r in rows if r.db_id in pool])[:1024]


def test_reserved_pin_is_disjoint_and_complete():
    if not data.PIN_RESERVED.exists():
        pytest.skip("no reserved pin yet")
    pin = data.read_pin(data.PIN_RESERVED)
    assert pin["file_sha256"] == C.RESERVED_PUBLIC["file_sha256"] and pin["questions"] == 500
    assert len(pin["question_ids"]) == len(set(pin["question_ids"])) == 500 and len(pin["db_ids"]) == 500
    assert set(pin["databases"]) == set(pin["db_ids"]) and len(pin["databases"]) == 11
    split = data.read_pin(data.PIN_SPLIT)
    assert not set(pin["databases"]) & set(split["dev"] + split["train_pool"])
    if data.PIN_SNAPSHOTS.exists():
        training = {v["sha256"] for v in data.read_pin(data.PIN_SNAPSHOTS)["databases"].values()}
        assert not training & {v["sha256"] for v in pin["databases"].values()}


def test_real_index_holds_training_pool_only():
    _release()
    index = retrieval.training_pool_index()
    pool = set(data.train_pool_databases())
    assert {d.db_id for d in index.documents} == pool and len(index.documents) == 5071
    task = data.task(data.tasks("dev")[0].task_id)
    hits = retrieval.retrieve(task)
    assert len(hits) == C.Limits().retrieval_k and all(h.db_id in pool for h in hits)
    assert retrieval.retrieve(task) == hits  # deterministic
