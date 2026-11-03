"""The grounding-interface study (development): the constraint-only machine, the grounding record and its consumption.

- **Constraint only.** `gi_constraint.RestrictedV1Machine` makes exactly the second ON operand that grounding v2
  excludes illegal, and nothing else. Every other decision's options equal v1's, the same comparison in WHERE stays
  legal, a self-join on the other occurrence stays legal, and legitimate gold programs trace to v1's actions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
sys.path.insert(0, str(REPO / "tests"))
pytest.importorskip("torch")
pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("sqlglot")
import test_ti_tnn_scaling_grounding_v2 as V2  # noqa: E402
from scaling import actions as A  # noqa: E402
from scaling import gi_constraint as GC  # noqa: E402
from scaling import sql_to_ir as X  # noqa: E402

backbone, encodings, prepared = V2.backbone, V2.encodings, V2.prepared  # the fixtures


def _replay(items, machine, ops, op, table_a, table_b):
    m = machine(items, None, grounding="v1") if machine is GC.RestrictedV1Machine else machine(items, None,
                                                                                               grounding=ops)
    seq = (A.Action("QUERY_SELECT"), A.Action("FROM_TABLE", items.table_item[table_a]),
           A.Action("JOIN_INNER_TABLE", items.table_item[table_b]), A.Action("END"), A.Action(op),
           A.Action("COL0", V2._col(items, table_a, "movie_id")))
    for a in seq:
        m.apply(a)
    return m


def _legal_second_operands(m, items):
    return {(k, i) for k in ("COL0", "COL1") for i in range(len(items.items))
            if items.items[i].kind == "column" and m.is_legal(A.Action(k, i))}


@pytest.mark.parametrize("op", sorted(A.COMPARE_ACTIONS))
@pytest.mark.parametrize("tables", [("ratings", "movies"), ("ratings", "ratings")])
def test_the_restricted_v1_machine_excludes_exactly_what_v2_excludes(op, tables):
    items = V2._items()
    restricted = _replay(items, GC.RestrictedV1Machine, None, op, *tables)
    v2 = _replay(items, A.Machine, "v2", op, *tables)
    v1 = _replay(items, A.Machine, "v1", op, *tables)
    got = _legal_second_operands(restricted, items)
    assert got == _legal_second_operands(v2, items)
    assert _legal_second_operands(v1, items) - got == {("COL0", V2._col(items, tables[0], "movie_id"))}
    if tables[0] == tables[1]:  # the self-join on the other occurrence stays legal
        assert ("COL1", V2._col(items, "ratings", "movie_id")) in got


def test_the_same_comparison_in_where_stays_legal():
    items = V2._items()
    m = GC.RestrictedV1Machine(items, None, grounding="v1")
    for a in (A.Action("QUERY_SELECT"), A.Action("FROM_TABLE", items.table_item["movies"]), A.Action("END"),
              A.Action("EQ"), A.Action("COL0", V2._col(items, "movies", "title"))):
        m.apply(a)
    assert m.is_legal(A.Action("COL0", V2._col(items, "movies", "title")))


@pytest.mark.parametrize("sql", V2.LEGITIMATE)
def test_legitimate_programs_trace_to_v1_actions_under_the_restriction(sql, encodings):
    enc = encodings[0]
    plan = X.convert(sql, enc.schema.index)
    v1 = A.trace(plan, enc.schema, enc.spans, "v1")
    with GC.machine_class(GC.RestrictedV1Machine):
        restricted = A.trace(plan, enc.schema, enc.spans, "v1")
    assert A.Machine is not GC.RestrictedV1Machine  # restored
    assert [s.action for s in restricted] == [s.action for s in v1]


def test_self_joins_are_counted_per_select():
    from scaling import ir
    items = V2._items()
    plan = X.convert("SELECT T2.name FROM people AS T1 INNER JOIN people AS T2 ON T1.mentor_id = T2.person_id",
                     items.index)
    assert GC.self_joins(plan) == 1
    assert GC.self_joins(X.convert("SELECT title FROM movies", items.index)) == 0
    assert isinstance(plan, ir.Select)


# --- the grounding record ----------------------------------------------------------------------------------------------
from scaling import gi_record as GR  # noqa: E402

QUESTION, EVIDENCE = "Which movie has the highest average score?", "highest average score refers to MAX(AVG(score))"


def _record(**over):
    obj = {"tables": ["movies", "ratings"], "columns": [{"table": "ratings", "column": "score"},
                                                       {"table": "movies", "column": "title"}],
           "literals": [], "status": "ok"}
    obj.update(over)
    return obj


def test_a_valid_record_resolves_to_ranked_items_and_the_fixed_prior():
    items = V2._items()
    g = GR.resolve(_record(tables=["MOVIES", "ratings", "movies"]), items, QUESTION, EVIDENCE)
    assert g.valid and g.tables == [items.table_item["movies"], items.table_item["ratings"]]  # first occurrence
    b = g.biases(len(items.items))
    import math
    assert b[items.table_item["movies"]] == math.log(2) and b[items.table_item["ratings"]] == math.log(2) / 2
    assert b[V2._col(items, "ratings", "score")] == math.log(2) and b[V2._col(items, "movies", "title")] == \
        math.log(2) / 2
    assert sum(x != 0 for x in b) == 4  # unlisted items: no prior


@pytest.mark.parametrize("bad, reason", [
    ({"tables": ["film"]}, "unknown table"), ({"columns": [{"table": "movies", "column": "rating"}]}, "unknown column"),
    ({"status": "fine"}, "status"), ({"sql": "SELECT 1"}, "keys"), ({"columns": ["movies.title"]}, "columns")])
def test_an_invalid_record_falls_back_to_the_original_policy(bad, reason):
    obj = {**_record(), **bad}
    g = GR.resolve(obj, V2._items(), QUESTION, EVIDENCE)
    assert not g.valid and reason in g.reason and g.biases(10) is None


def test_truncated_or_unparsable_output_is_invalid():
    assert not GR.resolve(_record(), V2._items(), QUESTION, EVIDENCE, truncated=True).valid
    obj, reason = GR.parse("I think the tables are movies and ratings.")
    assert obj is None and "not JSON" in reason
    obj, _ = GR.parse('Here it is:\n```json\n{"tables": [], "columns": [], "literals": [], "status": "unresolved"}\n```')
    assert obj["status"] == "unresolved"


def test_literals_are_validated_one_by_one_and_never_invalidate_the_record():
    items = V2._items()
    lits = [{"span": "highest average score", "value": "score", "table": "ratings", "column": "score"},
            {"span": "Gamma", "value": "Gamma", "table": "movies", "column": "title"}]  # not in the question
    g = GR.resolve(_record(literals=lits), items, QUESTION, EVIDENCE)
    assert g.valid and len(g.literals) == 1 and len(g.invalid_literals) == 1


def test_the_prior_moves_pointer_scores_and_is_restored(prepared):
    import torch
    from scaling import train as T
    model = T.build_model("T4", 16, 100, "v2")
    items = prepared["v2"][1].items
    mem = model.trunk.memory(V2.K.collate([prepared["v2"][1]]))
    q = torch.randn(1, 1, V2.K.N_POINTERS, model.cfg.d_model)
    base, _, _ = model.trunk.pointer_scores(mem, q)
    b = [0.0] * items.shape[0]
    b[2] = 5.0
    with GR.pointer_prior(model, b):
        moved, _, _ = model.trunk.pointer_scores(mem, q)
    assert torch.allclose(moved - base, torch.tensor(b).expand_as(base))
    again, _, _ = model.trunk.pointer_scores(mem, q)
    assert torch.equal(again, base) and "pointer_scores" not in vars(model.trunk)
    with GR.pointer_prior(model, None):  # an invalid record: the original policy
        assert torch.equal(model.trunk.pointer_scores(mem, q)[0], base)


def test_the_oracle_record_holds_the_gold_tables_and_columns_unordered():
    items = V2._items()
    plan = X.convert("SELECT T1.title FROM movies AS T1 INNER JOIN ratings AS T2 ON T1.movie_id = T2.movie_id "
                     "WHERE T2.score > 3", items.index)
    o = GR.oracle_record(plan, items)
    assert o["tables"] == sorted([items.table_item["movies"], items.table_item["ratings"]])
    assert set(o["columns"]) == {V2._col(items, "movies", "title"), V2._col(items, "movies", "movie_id"),
                                 V2._col(items, "ratings", "movie_id"), V2._col(items, "ratings", "score")}
    b = GR.oracle_biases(o, len(items.items))
    assert len({b[i] for i in o["tables"] + o["columns"]}) == 1  # one rank for all: no first table singled out


def test_lexical_grounding_reads_only_names_descriptions_and_the_question():
    from scaling import gi_lexical as GL
    items = V2._items()
    obj = GL.ground("Who directed the movie Alpha?", "Alpha refers to title = 'Alpha';", V2.SCHEMA, items)
    g = GR.resolve(obj, items, "Who directed the movie Alpha?", "Alpha refers to title = 'Alpha';")
    assert g.valid and g.tables[0] == items.table_item["movies"]
    assert V2._col(items, "movies", "title") in g.columns
    assert GL.ground("Who directed the movie Alpha?", "", V2.SCHEMA, items) == \
        GL.ground("Who directed the movie Alpha?", "", V2.SCHEMA, items)  # deterministic


# --- the runner -------------------------------------------------------------------------------------------------------
from scaling import gi_run as GRun  # noqa: E402


def test_swap_partners_stay_in_one_database_and_skip_singletons():
    ids = ["a1", "b1", "a2", "c1", "a3", "b2"]
    db = {"a1": "A", "a2": "A", "a3": "A", "b1": "B", "b2": "B", "c1": "C"}
    p = GRun.swap_partner(ids, db)
    assert p == {"a1": "a2", "a2": "a3", "a3": "a1", "b1": "b2", "b2": "b1"}


def test_the_budget_check_refuses_a_reservation_past_the_work_order_limit(monkeypatch):
    from scaling import ledger
    monkeypatch.setattr(ledger, "usage", lambda: {"used_hours": GRun.WORK_ORDER_LIMIT - 1.0})
    assert GRun.check_budget(0.6)["fits"]                     # 1.5 x 0.6 = 0.9 < 1.0
    with pytest.raises(GRun.GiStop, match="budget"):
        GRun.check_budget(0.7)                                # 1.05 > 1.0
    assert GRun.WORK_ORDER_LIMIT == pytest.approx(21.415907 + 8.0)


def test_record_diagnostics_against_the_gold_annotation():
    gold = {"convertible": True, "tables": [1, 2], "columns": [5, 6, 7], "first_table": 1,
            "literals": [(6, "alpha")]}
    g = {"valid": True, "tables": [2, 3], "columns": [5, 6], "status": "ok", "invalid_literals": [],
         "literals": [{"item": 6, "value": "'Alpha'"}, {"item": 7, "value": "alpha"}]}
    d = GRun.record_diagnostics(g, gold)
    assert d["table_recall"] == 0.5 and d["table_precision"] == 0.5 and not d["table_set_exact"]
    assert d["first_in_gold_set"] and not d["first_equals_gold_first"]   # in the set, not the gold first table
    assert d["column_coverage"] == 2 / 3 and d["literal_value_correct"] == 2 and d["literal_pair_correct"] == 1
    assert GRun.record_diagnostics({"valid": False}, gold) == {"scored": False}


def test_the_decoder_follows_the_prior_and_nothing_else(prepared):
    """A random controller that reaches a FROM-table pointer: a large prior on another legal table moves that choice,
    the decisions before it are untouched, and decoding without the prior is unchanged afterwards."""
    from scaling import infer_controller as I
    found = None
    for seed in range(8):
        for model in V2.models(seed=seed).values():
            for ex in prepared["v2"]:
                d = I.decode_one(model, ex, max_steps=60)
                k = next((n for n, a in enumerate(d.actions) if a.op == "FROM_TABLE"), None)
                if k is not None:
                    found = (model.eval(), ex, d, k)
                    break
            if found:
                break
        if found:
            break
    assert found, "no random controller reached a FROM-table pointer"
    model, ex, base, k = found
    items = ex.schema
    target = next(i for i in items.tables if i != base.actions[k].arg)
    b = [0.0] * len(items.items)
    b[target] = 50.0                                          # far larger than ln 2: forces the choice
    with GR.pointer_prior(model, b):
        pushed = I.decode_one(model, ex, max_steps=60)
    assert pushed.actions[k].op == "FROM_TABLE" and pushed.actions[k].arg == target
    assert [a.as_json() for a in pushed.actions[:k]] == [a.as_json() for a in base.actions[:k]]
    again = I.decode_one(model, ex, max_steps=60)
    assert [a.as_json() for a in again.actions] == [a.as_json() for a in base.actions]  # restored


def test_the_parser_repairs_only_mechanical_slips():
    """Selected on the development questions: one layer of SQL quoting around a name is removed, and invalid backslash
    escapes (such as \\' inside a JSON string) are dropped before the JSON is read again. Nothing semantic changes."""
    items = V2._items()
    obj, reason = GR.parse('{"tables": ["`movies`"], "columns": [{"table": "[ratings]", "column": "\\"score\\""}],'
                           ' "literals": [{"span": "6\\\'2", "value": "x", "table": "movies", "column": "title"}],'
                           ' "status": "ok"}')
    assert reason is None and obj["literals"][0]["span"] == "6'2"
    g = GR.resolve(obj, items, QUESTION, EVIDENCE)
    assert g.valid and g.tables == [items.table_item["movies"]] and g.columns == [V2._col(items, "ratings", "score")]
    assert GR.clean_name("  `voice-actors` ") == "voice-actors" and GR.clean_name("plain") == "plain"
    assert not GR.resolve({**obj, "tables": ["`film`"]}, items, QUESTION, EVIDENCE).valid  # still an unknown table
