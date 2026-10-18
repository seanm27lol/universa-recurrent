"""Execution records, the independent checker, replay and interventions (integration plan PR 5).

Software checks on the Ardel fixture and small worlds. Every deliberately faulty module set must be caught by
the check built for it, and by no check that should not see it. The frozen experiment is
protocols/semantic_interventions_v1.md.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from integrated import contracts as C  # noqa: E402
from integrated import faults as FX  # noqa: E402
from integrated import interventions as I  # noqa: E402
from integrated import records as RC  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import runtime as X  # noqa: E402
from integrated import verification as V  # noqa: E402

ARDEL_CHAIN = ("largest_city_of", "airport_of", "operator_of")


def program(world, steps, start):
    return C.Program(tuple(steps), world.types[world.relations[steps[0]].source].value(start))


def record_of(world, registry, steps, start, interventions=None, before=None):
    p, b = program(world, steps, start), C.Budget(len(steps))
    if before is not None:
        before(p)
    ex = X.execute(p, registry, world.types, b, interventions)
    return RC.to_record(ex, p, registry, b, world.identity)


def test_a_record_is_plain_json_and_checks_clean():
    world = W.ardel_world()
    rec = record_of(world, G.table_registry(world), ARDEL_CHAIN, "Ardel")
    assert json.loads(json.dumps(rec)) == rec and rec["schema"] == RC.SCHEMA
    assert RC.record_problems(rec) == []
    assert [s["consumed"]["value"] for s in rec["trace"]] == ["Ardel", "Port Vey", "Vey International"]
    assert rec["answer"]["value"] == "Tessaline Air" and rec["route"]["source"] == RC.ROUTE_SUPPLIED
    c = V.check(rec, world)
    assert c["schema"]["ok"] and c["types"]["ok"] and c["integrity"]["digest_matches"]
    assert c["semantics"] == {"reference_answer": "Tessaline Air", "answer_correct": True,
                              "step_correct": [True, True, True], "intervened_steps": []}
    assert c["provenance"].startswith("not authenticated")


def test_tampered_records_are_caught():
    world = W.ardel_world()
    rec = record_of(world, G.table_registry(world), ARDEL_CHAIN, "Ardel")
    edited = json.loads(json.dumps(rec))
    edited["trace"][1]["consumed"]["value"] = "Lunholm"
    problems = RC.record_problems(edited)
    assert "digest does not match the record" in problems
    assert "step 1: consumed a value the previous step did not produce" in RC.record_problems(RC.seal(edited))
    edited = json.loads(json.dumps(rec))
    edited["answer"]["value"] = "Northline"
    assert "the answer is not the last value produced" in RC.record_problems(RC.seal(edited))
    edited = json.loads(json.dumps(rec))
    edited["modules"][0]["identity"] = "0" * 64
    assert any("registry identity" in p for p in RC.record_problems(RC.seal(edited)))
    assert RC.record_problems({**rec, "explanation": "added later"}) == ["unknown fields ['explanation']"]
    assert not V.check({**rec, "explanation": "x"}, world)["schema"]["ok"]


def test_the_checker_and_the_schema_import_nothing_that_computes():
    for name, allowed in (("verification.py", {"__future__", "records", "relations"}),
                          ("records.py", {"__future__", "typing", "contracts"})):
        tree = ast.parse((ROOT / "integrated" / name).read_text())
        imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
                   {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert imported <= allowed, (name, imported)


def test_a_lawful_but_wrong_module_passes_every_structural_check_and_fails_semantics():
    world = W.ardel_world()
    swapped = FX.swapped_registry(world)
    assert all(v == [] for v in I.law_defects(swapped, world).values())  # both round-trip laws still hold
    rec = record_of(world, swapped, ("capital_of", "airport_of"), "Ardel")
    c = V.check(rec, world)
    assert c["schema"]["ok"] and c["types"]["ok"] and c["integrity"]["digest_matches"]
    assert I.replay(rec, swapped, world.types)["matches"] is True  # a faithful record of a wrong computation
    assert c["semantics"]["answer_correct"] is False and c["semantics"]["step_correct"] == [False, True]
    assert c["semantics"]["reference_answer"] == "Crossing Airfield" and rec["answer"]["value"] == "Vey International"


def test_replay_refuses_another_registry_and_catches_hidden_state():
    world = W.ardel_world()
    table = G.table_registry(world)
    rec = record_of(world, table, ARDEL_CHAIN, "Ardel")
    assert I.replay(rec, table, world.types) == {"refused": None, "matches": True}
    assert I.replay(rec, FX.swapped_registry(world), world.types)["refused"]

    @dataclass(frozen=True)
    class Drifting:  # same identity on every call, different answer after the first
        version: C.ModuleVersion
        inner: object
        calls: list

        def apply(self, value):
            self.calls.append(value)
            out = self.inner.apply(value)
            return out if len(self.calls) == 1 else world.types["Operator"].value("Northline")

    inner = G.table_module(world, "operator_of")
    reg = G.Registry(world.relations)
    for rid in sorted(world.relations):
        reg = reg.register(Drifting(inner.version, inner, []) if rid == "operator_of" else G.table_module(world, rid))
    rec = record_of(world, reg, ARDEL_CHAIN, "Ardel")
    assert I.replay(rec, reg, world.types) == {"refused": None, "matches": False}


def test_interventions_are_type_checked_recorded_and_judged_against_the_counterfactual_reference():
    world = W.ardel_world()
    table = G.table_registry(world)
    p, b = program(world, ARDEL_CHAIN, "Ardel"), C.Budget(3)
    with I.probe(table) as log:
        bad = X.execute(p, table, world.types, b, {1: world.types["Country"].value("Brenmark")})
        late = X.execute(p, table, world.types, b, {3: world.types["City"].value("Esk")})
    assert bad.status == late.status == "rejected" and log == []  # nothing ran
    rec = record_of(world, table, ARDEL_CHAIN, "Ardel", {1: world.types["City"].value("Esk")})
    assert RC.record_problems(rec) == [] and rec["trace"][1]["intervened"] and rec["answer"]["value"] == "Fieldway"
    c = V.check(rec, world)
    assert c["semantics"]["reference_answer"] == "Fieldway" and c["semantics"]["intervened_steps"] == [1]
    assert I.replay(rec, table, world.types)["matches"] is True
    r = I.intervene(world, table, p, b, 1, world.types["City"].value("Esk"))
    assert r == {"eligible": True, "recorded": True, "history_independent": True, "counterfactual_correct": True}


def test_a_module_that_reads_the_question_is_caught_only_by_interventions():
    world = W.ardel_world()
    reg, question = FX.question_reading_registry(world)
    for path in __import__("integrated.reference", fromlist=["x"]).well_typed_paths(world, 3):
        for start in world.types[world.relations[path[0]].source].values:
            rec = record_of(world, reg, path, start, before=question)
            assert V.check(rec, world)["semantics"]["answer_correct"], (path, start)  # every ordinary answer is right
            assert I.replay(rec, reg, world.types, question)["matches"]
    p, b = program(world, ("capital_of", "country_of"), "Ardel"), C.Budget(2)
    assert I.consumed_as_recorded(p, reg, world.types, b, question) == {"exact": True, "undeclared_calls": 0}
    assert I.unexecuted_disabled_unchanged(p, reg, world.types, b, world.identity, question)
    r = I.intervene(world, reg, p, b, 1, world.types["City"].value("Esk"), question)
    assert r["recorded"] and not r["history_independent"] and not r["counterfactual_correct"]


def test_a_hidden_dependency_is_caught_by_the_probe_and_by_disabling():
    world = W.ardel_world()
    reg = FX.hidden_dependency_registry(world)
    p, b = program(world, ("capital_of", "airport_of"), "Brenmark"), C.Budget(2)
    assert X.execute(p, reg, world.types, b).answer.value == "Brenhaven Airport"
    assert I.consumed_as_recorded(p, reg, world.types, b) == {"exact": True, "undeclared_calls": 1}
    assert not I.unexecuted_disabled_unchanged(p, reg, world.types, b, world.identity)
    assert X.execute(p, reg, world.types, b).answer.value == "Brenhaven Airport"  # restored afterwards
    table = G.table_registry(world)
    assert I.consumed_as_recorded(p, table, world.types, b)["undeclared_calls"] == 0
    assert I.unexecuted_disabled_unchanged(p, table, world.types, b, world.identity)


def test_the_runtime_drops_data_smuggled_alongside_a_value():
    world = W.ardel_world()
    reg = FX.payload_registry(world)
    p, b = program(world, ("capital_of", "country_of"), "Ardel"), C.Budget(2)
    with I.probe(reg) as log:
        ex = X.execute(p, reg, world.types, b, {1: world.types["City"].value("Esk")})
        X.execute(p, reg, world.types, b)
    assert ex.answer.value == "Brenmark"
    assert all(type(v) is C.TypedValue and not hasattr(v, "smuggled_country") for _, v, _ in log)
    smuggled = reg.module("capital_of").apply(world.types["Country"].value("Ardel"))
    assert smuggled.smuggled_country == "Ardel"  # the fault is real; only the runtime's rebuild removes it


spec = importlib.util.spec_from_file_location("ti_tnn_semantic_interventions", ROOT / "bench" / "semantic_interventions.py")


def bench():
    pytest.importorskip("torch")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_learned_modules_are_faithful_and_the_endpoint_control_is_faithful_but_wrong():
    S = bench()
    world = W.generate_world(7, n_countries=6, cities_per_country=(2, 3), n_operators=3)
    args = S.parse_args(["--steps", "300", "--d", "8", "--hidden", "16"])
    out = {}
    for family in ("specialists", "endpoint"):
        reg, before = S.build(world, family, 0, args)
        out[family] = S.measure(world, reg, before, 2)
    for r in out.values():
        n, iv = r["executions"], r["interventions"]
        assert r["replay_matches"] == r["consumed_exact"] == r["disabled_unchanged"] == n and r["undeclared_calls"] == 0
        assert iv["history_independent"] == iv["recorded"] == iv["n"] > 0
    assert out["specialists"]["answer_correct"] == out["specialists"]["executions"]
    assert out["endpoint"]["answer_correct"] < out["endpoint"]["executions"]
    assert out["endpoint"]["interventions"]["same_endpoint_correct"] < out["endpoint"]["interventions"]["same_endpoint_eligible"]


def fake_run(family, world, init, **over):
    n, k = 100, 400
    r = {"family": family, "world": world, "init": init, "executions": n, "schema_ok": n, "types_ok": n,
         "integrity_ok": n, "answer_correct": n, "replay_matches": n, "consumed_exact": n, "undeclared_calls": 0,
         "disabled_unchanged": n, "atomic": {"capital_of": 1.0, "largest_city_of": 1.0},
         "law_defects": {"a": 0, "b": 0}, "countries_where_capital_differs": 20, "countries": 24,
         "interventions": {"n": k, "recorded": k, "history_independent": k, "eligible": 300,
                           "counterfactual_correct_eligible": 300, "same_endpoint_eligible": 100, "same_endpoint_correct": 100}}
    if family in ("endpoint", "swapped"):
        r["atomic"] = {"capital_of": 1 - 20 / 24, "largest_city_of": 1.0}
        r["interventions"].update(counterfactual_correct_eligible=200, same_endpoint_correct=50)
    if family == "question_reading":
        r["interventions"]["history_independent"] = k - 40
    if family == "hidden_dependency":
        r.update(undeclared_calls=30, disabled_unchanged=n - 30)
    for key, value in over.items():
        r[key] = value
    return r


def test_the_frozen_gate_passes_designed_outcomes_and_fails_otherwise():
    S = bench()
    runs = [fake_run(f, w, i) for w in (300, 301, 302) for i in range(5) for f in S.LEARNED] + \
           [fake_run(f, w, None) for w in (300, 301, 302) for f in S.CONSTRUCTED]
    verdict = S.evaluate({"runs": runs})
    assert verdict["complete"] and all(verdict["predictions"].values()), verdict["predictions"]
    caught = [fake_run("question_reading", 300, None, interventions={**runs[-3]["interventions"], "history_independent": 400})]
    v = S.evaluate({"runs": [r for r in runs if (r["family"], r["world"]) != ("question_reading", 300)] + caught})
    assert v["predictions"]["4_question_reading_caught_only_by_interventions"] is False
    two_fail = [fake_run("moe", 301, i, atomic={"capital_of": 0.9}) if i < 2 else fake_run("moe", 301, i) for i in range(5)]
    v = S.evaluate({"runs": [r for r in runs if (r["family"], r["world"]) != ("moe", 301)] + two_fail})
    assert v["predictions"]["3a_atomic_competence_replicates"] is False
    assert v["predictions"]["3b_counterfactual_accuracy_given_competence"] is True  # only competent runs count
    assert S.evaluate({"runs": runs[1:]})["complete"] is False
