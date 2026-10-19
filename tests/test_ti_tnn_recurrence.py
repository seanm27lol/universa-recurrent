"""Bounded recurrence, the continuation state and cost accounting (integration plan PR 7).

Software checks on small route worlds; the frozen experiment is protocols/typed_recurrence_v1.md.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from integrated import contracts as C  # noqa: E402
from integrated import costs as K  # noqa: E402
from integrated import recurrence as Q  # noqa: E402
from integrated import reference as R  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402


def small():
    return W.route_world(1, n_countries=10, cities_per_country=(2, 4), n_operators=3, terminal_share=0.1)


def every_query(world):
    return [Q.UntilQuery(world.types["Airport"].value(a), world.types["Operator"].value(o))
            for a in world.types["Airport"].values for o in world.types["Operator"].values]


def test_the_route_world_leaves_earlier_worlds_unchanged():
    w = W.route_world(200, n_countries=24)
    base = W.generate_world(200)
    assert base.identity == "74ec965ea39df892e358fb808c342a902b82c86517cefbfb5a21a1d2386e3fe0"  # as recorded in PR 4
    assert all(w.facts[r] == base.facts[r] for r in base.relations) and set(w.relations) - set(base.relations) == {"connects_to"}
    c = w.facts["connects_to"]
    assert set(c) == set(w.types["Airport"].values) and any(a == b for a, b in c.items())


def test_recurrence_matches_the_reference_for_every_query_and_every_way_of_stopping():
    world = small()
    table = G.table_registry(world)
    reasons = Counter()
    for q in every_query(world):
        for budget in (1, 3, 16):
            state = Q.run(Q.begin(q, table, world.types, budget), table, world.types)
            ans, reason = R.reference_until(world, q.start.value, q.target.value, budget)
            got = (state["answer"]["value"], "found") if state["status"] == Q.ANSWERED else (None, state["reason"])
            assert got == (ans, reason)
            reasons[reason] += 1
            if reason == "budget exhausted":
                assert state["steps_used"] == budget and state["applications"]  # the partial trace is kept
    assert {"found", "dead end", "cycle", "budget exhausted"} <= set(reasons)


def test_ill_typed_queries_are_rejected_before_any_module_runs():
    world = small()
    table = G.table_registry(world)
    city = world.types["City"].value(world.types["City"].values[0])
    q = Q.UntilQuery(city, world.types["Operator"].value(world.types["Operator"].values[0]))
    state = Q.run(Q.begin(q, table, world.types, 8), table, world.types)
    assert state["status"] == Q.ABSTAINED and state["reason"].startswith("rejected") and state["applications"] == []
    q = Q.UntilQuery(world.types["Airport"].value(world.types["Airport"].values[0]), city, "connects_to", "operator_of")
    assert Q.query_problems(q, table, world.types)


def test_pausing_after_any_iteration_and_resuming_from_json_is_exact():
    world = small()
    table = G.table_registry(world)
    for q in every_query(world)[:40]:
        full = Q.run(Q.begin(q, table, world.types, 16), table, world.types)
        for p in range(1, full["steps_used"] + 2):
            paused = Q.run(Q.begin(q, table, world.types, 16), table, world.types, pause_after=p)
            fresh = G.table_registry(world)  # a different object with the same identity
            assert Q.resume(json.loads(json.dumps(paused)), fresh, world.types) == full
    paused = Q.run(Q.begin(every_query(world)[0], table, world.types, 16), table, world.types, pause_after=1)
    other = G.Registry(world.relations)
    for rid in sorted(world.relations):
        other = other.register(G.table_module(world, rid, f"{rid}_v2"))
    with pytest.raises(ValueError):
        Q.resume(paused, other, world.types)


def test_a_module_with_hidden_state_breaks_pause_and_resume_and_the_check_sees_it():
    world = small()

    @dataclass(frozen=True)
    class Counting:  # answers from its table, except that every third call it repeats its input
        version: C.ModuleVersion
        inner: object
        calls: list

        def apply(self, value):
            self.calls.append(1)
            return value if len(self.calls) % 3 == 0 else self.inner.apply(value)

    def registry():
        inner = G.table_module(world, "connects_to")
        reg = G.Registry(world.relations)
        for rid in sorted(world.relations):
            reg = reg.register(Counting(inner.version, inner, []) if rid == "connects_to" else G.table_module(world, rid))
        return reg

    mismatches = 0
    for q in every_query(world):
        reg = registry()
        full = Q.run(Q.begin(q, reg, world.types, 16), reg, world.types)
        reg2 = registry()
        paused = Q.run(Q.begin(q, reg2, world.types, 16), reg2, world.types, pause_after=2)
        mismatches += Q.resume(json.loads(json.dumps(paused)), registry(), world.types) != full
    assert mismatches > 0


def test_fixed_depth_gives_the_same_answers_at_full_cost():
    world = small()
    table = G.table_registry(world)
    for q in every_query(world):
        rec = Q.run(Q.begin(q, table, world.types, 8), table, world.types)
        fix = Q.fixed_depth(q, table, world.types, 8)
        assert (fix["status"], fix["reason"], fix["answer"]) == (rec["status"], rec["reason"], rec["answer"])
        assert len(fix["applications"]) == 16 >= len(rec["applications"])


def test_costs_count_macs_and_timing_does_not_change_the_result():
    torch = pytest.importorskip("torch")
    from integrated import modules as M
    world = small()
    table = G.table_registry(world)
    assert K.module_macs(table.module("connects_to")) == 0
    trained = M.train_family(world, "specialists", 0, d=8, hidden=16, steps=50)
    n_airports = len(world.types["Airport"].values)
    assert K.module_macs(trained["modules"]["connects_to"]) == 8 * 16 + 16 * n_airports
    q = every_query(world)[5]
    scored = Q.run(Q.begin(q, table, world.types, 16), table, world.types)
    timed, t = K.timed(lambda: Q.run(Q.begin(q, table, world.types, 16), table, world.types), table)
    assert timed == scored and 0 <= t["module_ns"] <= t["total_ns"] and t["calls"] == len(scored["applications"])
    c = K.application_costs(scored, table)
    assert c["module_applications"] == len(scored["applications"]) and c["macs"] == 0 and c["state_bytes"] > 0


spec = importlib.util.spec_from_file_location("ti_tnn_typed_recurrence", ROOT / "bench" / "typed_recurrence.py")


def fake(system, world, init, **over):
    r = {"system": system, "world": world, "init": init, "atomic": {"connects_to": 1.0, "operator_of": 1.0},
         "queries": 1000, "correct": 1000, "same_timed": 1000,
         "module_applications_mean": 32.0 if system == "fixed_depth" else 7.0, "macs_mean": 1.0,
         "wall_us_median": 1600.0 if system == "fixed_depth" else 350.0, "module_us_median": 1.0,
         "bookkeeping_share_median": 0.2, "state_bytes_mean": 1.0}
    if system == "recurrent":
        r.update(pause_resume={"pause_points": 2000, "identical": 2000}, agrees_with_fixed_depth=1000)
    r.update(over)
    return r


def test_the_frozen_gate():
    pytest.importorskip("torch")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    runs = [fake(s, w, i) for s in S.SYSTEMS for w in (500, 501, 502) for i in range(5)]
    v = S.evaluate({"runs": runs})
    assert v["complete"] and all(v["predictions"].values()), v["predictions"]
    broken = [r if (r["system"], r["world"], r["init"]) != ("recurrent", 501, 2)
              else fake("recurrent", 501, 2, pause_resume={"pause_points": 2000, "identical": 1999}) for r in runs]
    assert S.evaluate({"runs": broken})["predictions"]["4_pause_resume_identical"] is False
    assert S.evaluate({"runs": runs[1:]})["complete"] is False
