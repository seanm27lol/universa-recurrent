"""Typed contracts, the runtime and the independent reference (integration plan PR 3).

Software checks on a finite world; no learning is involved.
"""
from __future__ import annotations

import ast
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from integrated import contracts as C  # noqa: E402
from integrated import reference as R  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import runtime as X  # noqa: E402

ARDEL_CHAIN = ("largest_city_of", "airport_of", "operator_of")


def run(world, steps, start, registry=None, max_steps=8, expected=None):
    program = C.Program(tuple(steps), world.types[world.relations[steps[0]].source].value(start), expected)
    return X.execute(program, registry or G.table_registry(world), world.types, C.Budget(max_steps))


def test_types_reject_values_outside_their_domain():
    world = W.ardel_world()
    assert world.types["Country"].value("Ardel") == C.TypedValue("Country", "Ardel")
    with pytest.raises(ValueError):
        world.types["Country"].value("Port Vey")
    with pytest.raises(ValueError):
        C.TypeSpec("City", ("a", "a"))


def test_records_reject_malformed_inputs():
    with pytest.raises(ValueError):
        C.ModuleVersion("m", "capital_of", "table", "not-a-hash")
    with pytest.raises(ValueError):
        C.RelationSpec("capital of", "Country", "City")
    with pytest.raises(ValueError):
        C.Program((), C.TypedValue("Country", "Ardel"))
    with pytest.raises(ValueError):
        C.Budget(0)
    with pytest.raises(ValueError):
        C.Decision("maybe")
    with pytest.raises(ValueError):
        C.Execution("answered", C.Decision("commit"), None, (), None)


def test_same_endpoint_relations_keep_distinct_identities():
    world = W.ardel_world()
    cap, big = world.relations["capital_of"], world.relations["largest_city_of"]
    assert (cap.source, cap.target) == (big.source, big.target) and cap.relation_id != big.relation_id
    reg = G.table_registry(world)
    assert reg.module("capital_of").version.identity != reg.module("largest_city_of").version.identity
    assert run(world, ["capital_of"], "Ardel").answer.value == "Ardel Crossing"
    assert run(world, ["largest_city_of"], "Ardel").answer.value == "Port Vey"
    assert run(world, ["largest_city_of"], "Brenmark").answer.value == run(world, ["capital_of"], "Brenmark").answer.value


def test_worlds_are_validated():
    world = W.ardel_world()
    facts = {r: dict(t) for r, t in world.facts.items()}
    partial = dict(facts, capital_of={"Ardel": "Ardel Crossing"})
    with pytest.raises(ValueError, match="cover every"):
        W.World("bad", world.types, world.relations, partial)
    airports = world.types["Airport"].values
    many_ops = dict(world.types, Operator=C.TypeSpec("Operator", tuple(f"op_{i}" for i in range(len(airports)))))
    injective = dict(facts, operator_of={a: f"op_{i}" for i, a in enumerate(airports)})
    with pytest.raises(ValueError, match="many_to_one"):
        W.World("bad", many_ops, world.relations, injective)  # declared many_to_one, but injective
    shared = dict(facts, airport_of={c: "Esk Strip" for c in world.types["City"].values})
    with pytest.raises(ValueError, match="one_to_one"):
        W.World("bad", world.types, world.relations, shared)


def test_registry_is_immutable_and_refuses_rebinding():
    world = W.ardel_world()
    empty = G.Registry(world.relations)
    one = empty.register(G.table_module(world, "capital_of"))
    assert "capital_of" not in empty.modules and "capital_of" in one.modules
    assert empty.identity != one.identity
    with pytest.raises(ValueError, match="already"):
        one.register(G.table_module(world, "capital_of", "capital_of_again"))
    wrong = G.table_module(world, "capital_of")
    with pytest.raises(ValueError):
        G.Registry(world.relations, {"largest_city_of": wrong})  # a capital_of module bound to largest_city_of
    with pytest.raises(TypeError):
        one.modules["largest_city_of"] = wrong  # the binding cannot be mutated in place


class Spy:
    """A module that records every value it is given, then answers from a table."""

    def __init__(self, inner):
        self.inner, self.version, self.seen = inner, inner.version, []

    def apply(self, value):
        self.seen.append(value)
        return self.inner.apply(value)


def spied_registry(world):
    reg, spies = G.Registry(world.relations), {}
    for rid in sorted(world.relations):
        spies[rid] = Spy(G.table_module(world, rid))
        reg = reg.register(spies[rid])
    return reg, spies


def test_ill_typed_programs_are_rejected_before_any_module_runs():
    world = W.ardel_world()
    reg, spies = spied_registry(world)
    bad = run(world, ["capital_of", "operator_of"], "Ardel", reg)  # a City is not an Airport
    assert bad.status == "rejected" and bad.trace == () and "expects type Airport, got City" in bad.problems[0]
    wrong_end = run(world, ["capital_of"], "Ardel", reg, expected="Operator")
    assert wrong_end.status == "rejected"
    unknown = X.execute(C.Program(("teleport_to",), world.types["Country"].value("Ardel")), reg, world.types, C.Budget(4))
    assert unknown.status == "rejected"
    assert all(not s.seen for s in spies.values())


def test_the_ardel_example_runs_and_its_trace_records_what_was_passed():
    world = W.ardel_world()
    reg, spies = spied_registry(world)
    out = run(world, ARDEL_CHAIN, "Ardel", reg, expected="Operator")
    assert out.status == "answered" and out.answer.value == "Tessaline Air"
    assert [s.produced.value for s in out.trace] == ["Port Vey", "Vey International", "Tessaline Air"]
    for prev, step in zip(out.trace, out.trace[1:]):
        assert step.consumed is prev.produced  # the next module received exactly the recorded object
        assert spies[step.relation_id].seen[-1] is prev.produced
    assert out.final_state.registry_identity == reg.identity
    assert run(world, ("capital_of",) + ARDEL_CHAIN[1:], "Ardel").answer.value == "Northline"


def test_budget_exhaustion_abstains_with_the_partial_trace():
    world = W.ardel_world()
    out = run(world, ARDEL_CHAIN, "Ardel", max_steps=2)
    assert out.status == "abstained" and out.decision.reason == "budget exhausted"
    assert len(out.trace) == 2 and out.final_state.current.value == "Vey International"


def test_a_module_answering_outside_its_type_is_rejected():
    world = W.ardel_world()
    good = G.table_module(world, "capital_of")

    class Liar:
        version = good.version

        def apply(self, value):
            return C.TypedValue("Airport", "Esk Strip")

    reg = G.Registry(world.relations).register(Liar())
    out = run(world, ["capital_of"], "Ardel", reg)
    assert out.status == "rejected" and "outside its declared type" in out.decision.reason


@pytest.mark.parametrize("world", [W.ardel_world()] + [W.generate_world(s) for s in (0, 1, 2)], ids=lambda w: w.name)
def test_runtime_agrees_with_the_independent_reference_on_every_path(world):
    reg = G.table_registry(world)
    paths = R.well_typed_paths(world, 3)
    assert ARDEL_CHAIN in paths and ("capital_of", "operator_of") not in paths
    for path in paths:
        source = world.relations[path[0]].source
        for start in world.types[source].values:
            assert run(world, path, start, reg).answer.value == R.reference_answer(world, path, start)


def test_the_reference_imports_nothing_from_the_runtime():
    tree = ast.parse((ROOT / "integrated" / "reference.py").read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
               {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not imported & {"runtime", "registry", ".runtime", ".registry", "integrated.runtime", "integrated.registry"}
    assert imported <= {"__future__", "itertools", "relations"}


def test_a_wrong_module_is_caught_by_comparing_with_the_reference():
    world = W.generate_world(3)
    swapped = G.TableModule(replace(G.table_module(world, "capital_of").version, module_id="swapped",
                                    relation_id="largest_city_of"),
                            world.facts["capital_of"], world.types["City"])
    reg = G.Registry(world.relations)
    for rid in sorted(world.relations):
        reg = reg.register(swapped if rid == "largest_city_of" else G.table_module(world, rid))
    differ = [c for c in world.types["Country"].values
              if world.facts["capital_of"][c] != world.facts["largest_city_of"][c]]
    assert differ  # the generator makes the two relations differ in most countries
    for c in world.types["Country"].values:
        same = run(world, ["largest_city_of"], c, reg).answer.value == R.reference_answer(world, ("largest_city_of",), c)
        assert same == (c not in differ)


def test_restricted_round_trips_hold_and_the_reverse_does_not():
    for world in (W.ardel_world(), W.generate_world(5)):
        assert R.law_violations(world) == []
        cities = world.types["City"].values
        not_capitals = [x for x in cities if x not in world.facts["capital_of"].values()]
        assert not_capitals and all(R.reference_answer(world, ("country_of", "capital_of"), x) != x for x in not_capitals)


def test_generated_worlds_keep_the_relations_apart():
    for seed in range(5):
        world = W.generate_world(seed)
        countries = world.types["Country"].values
        differ = sum(world.facts["capital_of"][c] != world.facts["largest_city_of"][c] for c in countries)
        assert differ >= 0.6 * len(countries)
        assert W.generate_world(seed).identity == world.identity  # deterministic in the seed
    assert W.generate_world(0).identity != W.generate_world(1).identity
