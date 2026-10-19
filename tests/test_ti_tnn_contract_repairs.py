"""Regression tests for the contract gaps found in the review of plan PRs 4-8 (2026-10-18).

Each test reproduces one reported failure against the real registry, checker, deployment, runtime or
evaluator, and asserts the repaired behavior:
1. Gate results for candidate A cannot activate candidate B, and an altered or foreign result is refused.
2. A record naming the wrong world, checker or calibration context does not pass.
3. Mutating state shared by a base and a candidate after checking cannot change what is activated or
   rolled back to; mutating a published snapshot is detected.
4. An evaluator does not call an incomplete, duplicated or padded inventory complete.
5. A `str` subclass carrying a hidden attribute cannot pass unrecorded data to the next module.
6. Fixed-depth execution abstains where recurrence does on a missing module answer.
"""
from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
BUNDLE = Path(__file__).resolve().parents[1] / "experiments" / "results" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from integrated import contracts as C  # noqa: E402
from integrated import recurrence as Q  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import runtime as X  # noqa: E402
from integrated import snapshots as S  # noqa: E402
from integrated import update_checks as U  # noqa: E402
from integrated import updates as D  # noqa: E402
from integrated.reference import well_typed_paths  # noqa: E402

PROPOSER = "offline-update"


def ardel_move():
    world = W.ardel_world()
    facts = {r: dict(t) for r, t in world.facts.items()}
    facts["capital_of"]["Ardel"] = "Lunholm"
    return world, W.World("ardel+move", world.types, world.relations, facts), (("capital_of", "Ardel", "Lunholm"),)


def table_candidate(base, world, relation, table):
    version = C.ModuleVersion(f"{relation}_candidate", relation, "table",
                              C.content_identity({"kind": "table", "relation": relation, "table": C.facts_mapping(table)}))
    module = G.TableModule(version, table, world.types[world.relations[relation].target])
    reg = G.Registry(world.relations)
    for rid in sorted(world.relations):
        reg = reg.register(module if rid == relation else base.module(rid))
    return reg


def record(base, cand, new_world, intended, scope=("capital_of",), checker=None, **fields):
    return D.UpdateRecord(S.digest_of(base), S.digest_of(cand), scope, intended, new_world.identity,
                          fields.pop("checker_version", (checker or U.Checker()).version),
                          fields.pop("calibration_version", None), None, PROPOSER)


def answers(registry, world):
    out = []
    for path in well_typed_paths(world, 3):
        for s in world.types[world.relations[path[0]].source].values:
            ex = X.execute(C.Program(path, world.types[world.relations[path[0]].source].value(s)), registry,
                           world.types, C.Budget(len(path)))
            out.append(ex.answer.value if ex.answer else None)
    return out


def test_results_for_one_candidate_cannot_activate_another():
    world, new, intended = ardel_move()
    base = G.table_registry(world)
    checker = U.Checker()
    good = table_candidate(base, world, "capital_of", dict(new.facts["capital_of"]))
    bad = table_candidate(base, world, "capital_of", dict(world.facts["largest_city_of"]))  # honestly hashed, wrong
    good_rec, bad_rec = record(base, good, new, intended, checker=checker), record(base, bad, new, intended, checker=checker)
    good_res, bad_res = checker.check(base, good, good_rec, world, new), checker.check(base, bad, bad_rec, world, new)
    assert good_res.passed() and not bad_res.passed()
    dep = D.Deployment(base, D.Authority({PROPOSER}), checker)
    swapped = dep.activate(bad_rec, good_res)  # the reported failure: B's record with A's passing results
    assert not swapped["activated"] and any("different record" in r for r in swapped["reasons"])
    forged = dataclasses.replace(bad_res, gates={**bad_res.gates, **{g: {"outcome": U.PASS} for g in U.MANDATORY}})
    assert not dep.activate(bad_rec, forged)["activated"]  # altered outcomes break the seal
    foreign = U.Checker().check(base, good, good_rec, world, new)
    assert not dep.activate(good_rec, foreign)["activated"]  # sealed by a checker this deployment does not trust
    assert dep.activate(good_rec, good_res)["activated"] and answers(dep.registry, new) == answers(good, new)


def test_a_record_naming_the_wrong_context_does_not_pass():
    from integrated import calibration as K
    world, new, intended = ardel_move()
    base = G.table_registry(world)
    good = table_candidate(base, world, "capital_of", dict(new.facts["capital_of"]))
    checker = U.Checker()
    cases = {"wrong world": dataclasses.replace(record(base, good, new, intended, checker=checker),
                                                target_world_identity=world.identity),
             "wrong checker": record(base, good, new, intended, checker_version="some_other_checker_v9"),
             "missing calibration": record(base, good, new, intended, checker=checker, calibration_version="a" * 64)}
    dep = D.Deployment(base, D.Authority({PROPOSER}), checker)
    for name, rec in cases.items():
        res = checker.check(base, good, rec, world, new)
        assert res.outcome("record_context") != U.PASS, name
        assert not dep.activate(rec, res)["activated"], name
    assert checker.check(base, good, cases["missing calibration"], world, new).outcome("record_context") == U.INSUFFICIENT
    cal = K.fit([0.9] * 400, [True] * 400, [str(i) for i in range(400)], "interp", base.identity, "policy")
    undeclared = record(base, good, new, intended, checker=checker)
    assert checker.check(base, good, undeclared, world, new, calibration=cal).outcome("record_context") == U.FAIL


@dataclass(frozen=True)
class ArrayOperator:
    """An `operator_of` module backed by a NumPy array (the supported module interface, mutable state)."""

    version: C.ModuleVersion
    airports: tuple
    operators: tuple
    choice: np.ndarray
    target: C.TypeSpec

    def apply(self, value):
        return self.target.value(self.operators[int(self.choice[self.airports.index(value.value)])])


def array_registry(world):
    airports, operators = world.types["Airport"].values, world.types["Operator"].values
    choice = np.array([operators.index(world.facts["operator_of"][a]) for a in airports])
    module = ArrayOperator(C.ModuleVersion("operator_of_array", "operator_of", "array", "0" * 64), airports, operators,
                           choice, world.types["Operator"])
    reg = G.Registry(world.relations)
    for rid in sorted(world.relations):
        reg = reg.register(module if rid == "operator_of" else G.table_module(world, rid))
    return reg, module


def test_state_shared_with_a_candidate_cannot_change_what_is_activated_or_restored():
    world, new, intended = ardel_move()
    base, shared = array_registry(world)
    original = answers(base, world)  # captured before any candidate work
    checker = U.Checker()
    dep = D.Deployment(base, D.Authority({PROPOSER}), checker)
    cand = table_candidate(base, world, "capital_of", dict(new.facts["capital_of"]))  # shares the array module
    rec = record(base, cand, new, intended, checker=checker)
    res = checker.check(base, cand, rec, world, new)
    assert res.passed()
    checked = answers(res.snapshot.registry, new)
    before = S.digest_of(base)
    shared.choice[:] = (shared.choice + 1) % len(shared.operators)  # mutate the live, shared numeric state
    assert answers(base, world) != original and S.digest_of(base) != before  # the live registry changed, and it shows
    assert dep.activate(rec, res)["activated"]
    assert answers(dep.registry, new) == checked  # the checked copy is what runs, not the mutated live objects
    assert dep.rollback()["rolled_back"] and answers(dep.registry, world) == original


def test_tampering_with_a_published_snapshot_is_detected():
    world, new, intended = ardel_move()
    base, _ = array_registry(world)
    checker = U.Checker()
    cand = table_candidate(base, world, "capital_of", dict(new.facts["capital_of"]))
    rec = record(base, cand, new, intended, checker=checker)
    res = checker.check(base, cand, rec, world, new)
    dep = D.Deployment(base, D.Authority({PROPOSER}), checker)
    res.snapshot.registry.module("operator_of").choice[0] += 1  # reach into the checked copy
    decision = dep.activate(rec, res)
    assert not decision["activated"] and any("changed after it was checked" in r for r in decision["reasons"])
    res2 = checker.check(base, cand, rec, world, new)
    assert dep.activate(rec, res2)["activated"]
    dep.history[-1].registry.module("operator_of").choice[0] += 1  # reach into the stored base
    assert dep.rollback() == {"rolled_back": False, "reasons": ["the stored snapshot no longer matches its digest"]}


def test_learned_state_is_copied_and_closures_are_refused():
    torch = pytest.importorskip("torch")
    from integrated import modules as M
    world = W.generate_world(7, n_countries=6, cities_per_country=(2, 3), n_operators=3)
    for family in ("specialists", "shared_dense"):
        trained = M.train_family(world, family, 0, 8, 16, 50)
        reg = G.Registry(world.relations)
        for rid in sorted(world.relations):
            reg = reg.register(trained["modules"][rid])
        snap = S.publish(reg)
        net = trained["net"] if family == "shared_dense" else trained["modules"]["capital_of"].forward
        with torch.no_grad():
            next(net.parameters()).add_(1.0)  # the training object changes after publication
        assert snap.verify() and S.digest_of(reg) != snap.digest
    lam = dataclasses.replace(trained["modules"]["capital_of"], forward=lambda idx, n=net: n(idx, idx))
    closure_reg = G.Registry(world.relations, {**dict(reg.modules), "capital_of": lam})
    captured = dataclasses.replace(lam, forward=(lambda: (lambda idx: net(idx, idx)))())
    with pytest.raises(ValueError):
        S.publish(G.Registry(world.relations, {**dict(reg.modules), "capital_of": captured}))
    assert closure_reg  # a default argument is not a closure; it is digested with the function


def evaluator(name):
    spec = importlib.util.spec_from_file_location(f"repair_{name}", ROOT / "bench" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_incomplete_duplicated_or_padded_inventories_are_not_complete():
    pytest.importorskip("torch")
    L = evaluator("local_updates")
    result = json.loads((BUNDLE / "local_updates/20261018T164944/local_updates.json").read_text())
    assert L.evaluate(result)["complete"]
    runs = [r for r in result["runs"] if r["update"] in (0, 5)]  # one update of each kind per unit, as in the review
    suite = list({c["world"]: c for c in result["faulty_suite"]}.values())
    assert (len(runs), len(suite)) == (90, 3)
    reduced = {**result, "runs": runs, "faulty_suite": suite}
    assert L.evaluate(reduced)["complete"] is False  # the reported loophole: this was called complete
    assert L.evaluate({**result, "runs": result["runs"] + result["runs"][:1]})["complete"] is False
    extra = dict(result["runs"][0], update=99)
    assert L.evaluate({**result, "runs": result["runs"] + [extra]})["complete"] is False
    assert L.evaluate({**result, "faulty_suite": result["faulty_suite"][1:]})["complete"] is False
    T = evaluator("typed_composition")
    typed = json.loads((BUNDLE / "typed_composition/20261018T145735/typed_composition.json").read_text())
    assert T.evaluate(typed)["complete"]
    assert T.evaluate({**typed, "runs": typed["runs"] + typed["runs"][:1]})["complete"] is False


def test_a_str_subclass_cannot_carry_hidden_data_between_modules():
    world = W.ardel_world()

    class Tagged(str):
        pass

    @dataclass(frozen=True)
    class Smuggler:
        version: C.ModuleVersion
        target: C.TypeSpec

        def apply(self, value):
            out = Tagged(world.facts["capital_of"][value.value])
            out.secret = "Corvala"
            return C.TypedValue("City", out, 1)

    @dataclass(frozen=True)
    class Reader:
        version: C.ModuleVersion
        target: C.TypeSpec

        def apply(self, value):
            hidden = getattr(value.value, "secret", None)
            return self.target.value(hidden or world.facts["country_of"][value.value])

    reg = G.Registry(world.relations)
    for rid in sorted(world.relations):
        if rid == "capital_of":
            m = Smuggler(C.ModuleVersion("capital_of_smuggler", rid, "test", "1" * 64), world.types["City"])
        elif rid == "country_of":
            m = Reader(C.ModuleVersion("country_of_reader", rid, "test", "2" * 64), world.types["Country"])
        else:
            m = G.table_module(world, rid)
        reg = reg.register(m)
    p = C.Program(("capital_of", "country_of"), world.types["Country"].value("Ardel"))
    ex = X.execute(p, reg, world.types, C.Budget(2))
    assert ex.answer.value == "Ardel"  # the reported failure answered Corvala
    city = ex.trace[0].produced
    assert type(city.value) is str and not hasattr(city.value, "secret")
    from integrated.records import to_record
    rec = to_record(ex, p, reg, C.Budget(2), world.identity)
    restored = C.TypedValue(**rec["trace"][1]["consumed"])
    alone = X.execute(C.Program(("country_of",), restored), reg, world.types, C.Budget(1))
    assert alone.answer == ex.answer  # the whole run and the suffix from the record agree


def test_fixed_depth_abstains_where_recurrence_does():
    world = W.route_world(1, n_countries=10, cities_per_country=(2, 4), n_operators=3, terminal_share=0.1)
    table = G.table_registry(world)
    c, ops = world.facts["connects_to"], world.facts["operator_of"]
    for a in world.types["Airport"].values:  # find A -> B -> C with C's operator differing from B's
        b = c[a]
        if b != a and c[b] not in (a, b) and ops[c[b]] != ops[b]:
            break
    target = ops[c[b]]

    @dataclass(frozen=True)
    class Gap:
        version: C.ModuleVersion
        inner: object

        def apply(self, value):
            return None if value.value == b else self.inner.apply(value)

    inner = table.module("operator_of")
    reg = G.Registry(world.relations)
    for rid in sorted(world.relations):
        reg = reg.register(Gap(inner.version, inner) if rid == "operator_of" else table.module(rid))
    q = Q.UntilQuery(world.types["Airport"].value(a), world.types["Operator"].value(target))
    rec = Q.run(Q.begin(q, reg, world.types, 8), reg, world.types)
    fix = Q.fixed_depth(q, reg, world.types, 8)
    assert rec["status"] == Q.ABSTAINED and rec["reason"] == "module gave no valid answer"
    assert (fix["status"], fix["reason"], fix["answer"]) == (rec["status"], rec["reason"], rec["answer"])
