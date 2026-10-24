"""The gates can read a shared network (prerequisite of protocols/repeated_updates_v1.md).

Every module of a shared network is a `SharedForward`: one relation's view of the one network. Before this
repair the gates could read neither its parameters nor its cost:
- `identity_and_scope` reported every changed shared module as unverifiable (insufficient evidence);
- `calibration_and_resources` reported its cost as unreadable (insufficient evidence).

So no shared-network candidate could pass every mandatory gate, however exactly it behaved and however
honestly its scope was declared. PR 8's shared candidates failed for another reason (their declared scope was
one relation), which still holds.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
torch = pytest.importorskip("torch")
from integrated import costs as C  # noqa: E402
from integrated import modules as M  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import update_checks as U  # noqa: E402

spec = importlib.util.spec_from_file_location("ti_tnn_local_updates", ROOT / "bench" / "local_updates.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


@pytest.fixture(scope="module")
def shared_update():
    """A shared base on a small world, and a rehearsed candidate for one capital move."""
    world = W.generate_world(7, n_countries=6, cities_per_country=(2, 3), n_operators=3)
    new_world, intended, rel = S.updates(world, 1, 0)[0]
    base = M.train_family(world, "shared_dense", 0, 16, 32, 600)
    tuned = M.finetune(new_world, "shared_dense", base, 0, (rel,), 200, 3e-3, True)
    reg, cand = S.registry_from(world, base["modules"]), S.registry_from(new_world, tuned["modules"])
    return world, new_world, intended, rel, reg, cand, tuned


def test_a_shared_candidate_with_its_whole_network_in_scope_has_verifiable_identity_and_cost(shared_update):
    world, new_world, intended, rel, reg, cand, tuned = shared_update
    scope = tuple(sorted(world.relations))
    gates = S.CHECKER.check(reg, cand, S.record_for(reg, cand, new_world, intended, scope), world, new_world).gates
    assert gates["identity_and_scope"]["outcome"] == U.PASS, gates["identity_and_scope"]
    assert gates["identity_and_scope"]["changed"] == list(scope) and not gates["identity_and_scope"]["unverifiable"]
    assert gates["calibration_and_resources"]["outcome"] == U.PASS, gates["calibration_and_resources"]
    macs = C.module_macs(cand.module(rel))
    assert macs == C.linear_macs(tuned["net"]) > 0  # one application: the shared body, over every entity


def test_a_shared_candidate_still_fails_when_its_scope_is_one_relation_or_its_identity_is_stale(shared_update):
    world, new_world, intended, rel, reg, cand, tuned = shared_update
    one = S.CHECKER.check(reg, cand, S.record_for(reg, cand, new_world, intended, (rel,)), world, new_world)
    problems = one.gates["identity_and_scope"]["problems"]
    assert one.outcome("identity_and_scope") == U.FAIL  # PR 8's result is unchanged
    assert any("outside the declared scope" in p for p in problems) and any("shares parameter storage" in p for p in problems)
    stale = dict(cand.modules)
    m = stale[rel]
    stale[rel] = dataclasses.replace(m, version=dataclasses.replace(m.version, identity="0" * 64))
    bad = S.registry_from(new_world, stale)
    scope = tuple(sorted(world.relations))
    gate = U.identity_and_scope(reg, bad, S.record_for(reg, bad, new_world, intended, scope))
    assert gate["outcome"] == U.FAIL and any("does not match its parameters" in p for p in gate["problems"])


def test_a_mixture_of_experts_cost_is_still_unreadable():
    world = W.generate_world(7, n_countries=6, cities_per_country=(2, 3), n_operators=3)
    moe = M.train_family(world, "moe", 0, 16, 32, 20)
    assert C.module_macs(moe["modules"]["capital_of"]) is None  # which expert runs depends on the input
