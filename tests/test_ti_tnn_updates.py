"""Offline module replacement: update records, gates, activation and rollback (integration plan PR 8).

Software checks on the Ardel fixture and small worlds; the frozen experiment is protocols/local_updates_v1.md.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import update_checks as U  # noqa: E402
from integrated import updates as D  # noqa: E402

spec = importlib.util.spec_from_file_location("ti_tnn_local_updates", ROOT / "bench" / "local_updates.py")


def bench():
    pytest.importorskip("torch")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ardel_move():
    world = W.ardel_world()
    facts = {r: dict(t) for r, t in world.facts.items()}
    facts["capital_of"]["Ardel"] = "Lunholm"
    return world, W.World("ardel+move", world.types, world.relations, facts), (("capital_of", "Ardel", "Lunholm"),)


def test_a_valid_edit_passes_every_gate_activates_and_rolls_back_exactly():
    S = bench()
    world, new, intended = ardel_move()
    base = G.table_registry(world)
    cand = S.table_candidate(base, world, new, "capital_of")
    rec = S.record_for(base, cand, new, intended, ("capital_of",))
    result = S.CHECKER.check(base, cand, rec, world, new)
    assert result.passed(), {g: result.gates[g] for g in U.MANDATORY}
    assert result.counts["intended"] > 0 and result.counts["unaffected_changed_from_base"] == 0
    dep = D.Deployment(base, D.Authority({S.PROPOSER}), S.CHECKER)
    original = dep.active
    assert dep.activate(rec, result)["activated"] and dep.active is result.snapshot
    assert dep.rollback()["rolled_back"] and dep.active is original and original.verify()
    assert dep.rollback() == {"rolled_back": False, "reasons": ["no earlier snapshot"]}


def test_each_fault_is_caught_by_its_own_gate():
    S = bench()
    args = S.parse_args(["--steps", "300", "--d", "8", "--hidden", "16", "--update-steps", "100"])
    world = W.generate_world(7, n_countries=6, cities_per_country=(2, 3), n_operators=3)
    cases = {c["case"]: c for c in S.faulty_suite(world, 0, args)}
    assert set(cases) == {"valid_table_edit", "lawful_answer_flip", "broken_inverse", "ill_typed", "out_of_scope_state_edit",
                          "stale_revision", "failed_rollback", "unauthorized", "stale_identity", "shared_storage"}
    bad = {k: c for k, c in cases.items() if not c["as_expected"]}
    assert not bad, bad
    assert cases["lawful_answer_flip"]["gates"]["types_and_laws"] == U.PASS  # lawful, and still rejected
    assert cases["unauthorized"]["claimed_confidence"] > 0.99 and not cases["unauthorized"]["activated"]
    assert any("stale" in r for r in cases["stale_revision"]["reasons"])


def test_a_mismatched_or_tampered_record_is_refused():
    S = bench()
    world, new, intended = ardel_move()
    base = G.table_registry(world)
    cand = S.table_candidate(base, world, new, "capital_of")
    rec = S.record_for(base, cand, new, intended, ("capital_of",))
    result = S.CHECKER.check(base, cand, rec, world, new)
    wrong_intent = S.record_for(base, cand, new, (("capital_of", "Ardel", "Port Vey"),), ("capital_of",))
    assert S.CHECKER.check(base, cand, wrong_intent, world, new).outcome("intended_behavior") == U.FAIL
    other = S.table_candidate(base, world, new, "capital_of", table={**new.facts["capital_of"], "Corvala": "Talmont"})
    other_rec = S.record_for(base, other, new, intended, ("capital_of",))
    dep = D.Deployment(base, D.Authority({S.PROPOSER}), S.CHECKER)
    original = dep.active
    decision = dep.activate(other_rec, result)  # the result was made for a different record and candidate
    assert not decision["activated"] and dep.active is original


def test_learned_specialist_updates_are_local_and_shared_ones_are_not():
    S = bench()
    from integrated import modules as M
    world = W.generate_world(7, n_countries=6, cities_per_country=(2, 3), n_operators=3)
    new_world, intended, rel = S.updates(world, 1, 0)[0]
    out = {}
    for name, (family, rehearse) in S.FAMILIES.items():
        base = M.train_family(world, family, 0, 16, 32, 600)
        reg = S.registry_from(world, base["modules"])
        tuned = M.finetune(new_world, family, base, 0, (rel,), 200, 3e-3, rehearse)
        cand = S.registry_from(new_world, tuned["modules"])
        out[name] = S.CHECKER.check(reg, cand, S.record_for(reg, cand, new_world, intended, (rel,)), world, new_world).gates
    assert out["specialists"]["identity_and_scope"]["outcome"] == U.PASS
    assert out["specialists"]["identity_and_scope"]["changed"] == [rel]
    for name in ("shared_same_data", "shared_rehearsal"):
        assert out[name]["identity_and_scope"]["outcome"] == U.FAIL
        assert len(out[name]["identity_and_scope"]["changed"]) == len(world.relations)


def test_the_frozen_gate():
    S = bench()
    def row(fam, w, i, k, rel, unaffected="pass", activated=None):
        act = (fam == "specialists") if activated is None else activated
        return {"family": fam, "world": w, "init": i, "update": k, "relation": rel, "base_competent": True,
                "activated": act, "rolled_back": act, "rollback_exact": act,
                "gates": {g: ("pass" if fam == "specialists" or g != "identity_and_scope" else "fail") for g in D.MANDATORY}
                | {"unaffected_behavior": unaffected},
                "counts": {"intended": 5, "intended_correct": 5, "unaffected_changed_from_base": 0, "atomic_changed_by_relation": {}}}
    runs = []
    for w in (600, 601, 602):
        for i in range(5):
            for k, rel in enumerate(["capital_of"] * 5 + ["operator_of"] * 5):
                runs.append(row("specialists", w, i, k, rel))
                runs.append(row("shared_same_data", w, i, k, rel, "fail" if rel == "capital_of" else "pass"))
                runs.append(row("shared_rehearsal", w, i, k, rel))
    suite = [{"case": c, "world": w, "as_expected": True} for c in S.FAULT_CASES for w in (600, 601, 602)]
    v = S.evaluate({"runs": runs, "faulty_suite": suite})
    assert v["complete"] and all(v["predictions"].values()), v["predictions"]
    runs[0] = row("specialists", 600, 0, 0, "capital_of", activated=False)
    runs[3] = row("specialists", 600, 0, 1, "capital_of", activated=False)
    runs[6] = row("specialists", 600, 0, 2, "capital_of", activated=False)
    assert S.evaluate({"runs": runs, "faulty_suite": suite})["predictions"]["1_specialist_updates_pass_and_activate"] is False
    assert S.evaluate({"runs": runs, "faulty_suite": suite[:2]})["complete"] is False


def test_a_calibration_bound_to_the_old_registry_and_an_over_budget_module_are_not_passed():
    S = bench()
    from integrated import calibration as K
    world, new, intended = ardel_move()
    base = G.table_registry(world)
    cand = S.table_candidate(base, world, new, "capital_of")
    rec = S.record_for(base, cand, new, intended, ("capital_of",))
    cal = K.fit([0.9] * 400, [True] * 400, [str(i) for i in range(400)], "interp", base.identity, "policy")
    gate = U.calibration_and_resources(cand, base, rec, cal, "policy")
    assert gate["outcome"] == U.INSUFFICIENT and gate["insufficient"]
    assert U.calibration_and_resources(base, base, rec, cal, "policy")["outcome"] == U.PASS
    import dataclasses
    tight = dataclasses.replace(rec, max_macs_per_application=-1)  # a table costs 0 MACs, still over a limit of -1
    assert U.calibration_and_resources(cand, base, tight)["outcome"] == U.FAIL
