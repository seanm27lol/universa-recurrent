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

backbone, encodings = V2.backbone, V2.encodings  # the fixtures


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
