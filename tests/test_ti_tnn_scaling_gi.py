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
