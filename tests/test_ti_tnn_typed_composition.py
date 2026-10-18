"""Learned operators on the typed world and the PR 4 gate evaluation (integration plan PR 4).

Software checks on tiny worlds and short training; the frozen experiment is protocols/typed_composition_v1.md.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from integrated import contracts as C  # noqa: E402
from integrated import modules as M  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import runtime as X  # noqa: E402

spec = importlib.util.spec_from_file_location("ti_tnn_typed_composition", ROOT / "bench" / "typed_composition.py")
TC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TC)


def small_world():
    return W.generate_world(7, n_countries=6, cities_per_country=(2, 3), n_operators=3)


@pytest.mark.parametrize("family", M.MODULE_FAMILIES)
def test_learned_modules_plug_into_the_typed_runtime(family):
    world = small_world()
    trained = M.train_family(world, family, 0, d=8, hidden=16, steps=300)
    reg = __import__("integrated.registry", fromlist=["Registry"]).Registry(world.relations)
    for rid in sorted(world.relations):
        mod = trained["modules"][rid]
        assert mod.version.relation_id == rid and len(mod.version.identity) == 64
        reg = reg.register(mod)
    start = world.types["Country"].values[0]
    out = X.execute(C.Program(("largest_city_of", "airport_of", "operator_of"), world.types["Country"].value(start)),
                    reg, world.types, C.Budget(3))
    assert out.status == "answered" and out.answer.type_id == "Operator"
    assert all(s.produced.type_id == world.relations[s.relation_id].target for s in out.trace)


def test_endpoint_family_shares_one_operator_between_same_endpoint_relations():
    world = small_world()
    mods = M.train_family(world, "endpoint", 0, d=8, hidden=16, steps=50)["modules"]
    assert mods["capital_of"].forward is mods["largest_city_of"].forward
    countries = world.types["Country"].values
    for c in countries:
        v = world.types["Country"].value(c)
        assert mods["capital_of"].apply(v) == mods["largest_city_of"].apply(v)  # it cannot tell them apart


def test_score_and_paths_cover_the_right_families():
    world = small_world()
    res = TC.score(world, "specialists", M.train_family(world, "specialists", 0, d=8, hidden=16, steps=300))
    assert set(res["atomic"]) == set(world.relations)
    assert set(res["paths"]) == {"length2", "length3", "round_trip"}
    assert TC.path_family(("capital_of", "country_of")) == "round_trip"
    assert TC.path_family(("country_of", "capital_of", "airport_of")) == "length3"
    ans = TC.score(world, "answer_only", M.train_family(world, "answer_only", 0, d=8, hidden=16, steps=50))
    assert ans["intermediate"] is None


def fake_run(family, world, init, atomic=1.0, same=1.0, chain=1.0):
    t = lambda a: {"correct": round(a * 100), "total": 100}
    return {"family": family, "world": world, "init": init, "atomic": {"capital_of": atomic},
            "same_endpoint": {"capital_of": same, "largest_city_of": same, "countries": 10},
            "paths": {"length2": t(chain), "length3": t(chain), "round_trip": t(1.0)},
            "intermediate": None if family == "answer_only" else t(chain)}


def test_gate_evaluation_applies_the_frozen_rules():
    runs = []
    for w in (200, 201, 202):
        for i in range(5):
            runs += [fake_run(f, w, i) for f in ("specialists", "shared_dense", "moe")]
            runs.append(fake_run("endpoint", w, i, atomic=0.5, same=0.4, chain=0.6))
            runs.append(fake_run("answer_only", w, i, chain=0.1))
    v = TC.evaluate({"runs": runs})
    assert v["complete"] and all(v["predictions"].values())
    one_bad = list(runs)
    one_bad[0] = fake_run("specialists", 200, 0, atomic=0.9)  # 4 of 5 competent inits in world 200: still holds
    assert TC.evaluate({"runs": one_bad})["predictions"]["1_atomic_competence"] is True
    two_bad = list(one_bad)
    two_bad[5] = fake_run("specialists", 200, 1, atomic=0.9)  # runs are ordered world, init, family: 3 of 5
    v2 = TC.evaluate({"runs": two_bad})
    assert v2["predictions"]["1_atomic_competence"] is False
    assert v2["predictions"]["4_composition_given_competence"] is True  # judged only on competent runs
    collapsed = list(runs)
    collapsed[1] = fake_run("shared_dense", 200, 0, same=0.5)
    collapsed[6] = fake_run("shared_dense", 200, 1, same=0.5)
    assert TC.evaluate({"runs": collapsed})["predictions"]["2_same_endpoint_discrimination"] is False
    assert not TC.evaluate({"runs": runs[:-1]})["complete"]
