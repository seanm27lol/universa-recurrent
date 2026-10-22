"""Deterministic table-module fixtures for the integration test's construction checks and injected faults
(protocols/integration_test_v1.md, revision 2: "Predictions", C1-C5 and F).

The fixture world is a small route world with one route set by hand: S -> a -> b -> c -> d, where d is a
terminal. Operators are set by hand too: T is the target, and U is any other operator. Three updates are
exercised against S0, where a = U, b = U, c = T and d = U:
- **witness:** a = T and b = T. Naive continuation after testing a gives b, which neither version gives
  (S0 gives c, S1 gives a);
- **later change:** c = U and d = T. A run paused after testing a can migrate validly;
- **correction (the review's counterexample):** the reference world already had a = T, so S0 was wrong at a.
  The update declares only b -> T, and a correct candidate passes every gate. The actual difference contains
  both a and b, so migration after testing a must restart.

No learned model is involved.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from integrated import certificates as Q  # noqa: E402
from integrated import recurrence as R  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import serving as V  # noqa: E402
from integrated import session_verification as SV  # noqa: E402
from integrated import sessions as SS  # noqa: E402
from integrated import snapshots as S  # noqa: E402
from integrated import update_checks as U  # noqa: E402
from integrated import updates as D  # noqa: E402

PROPOSER = "offline-update"
GRID = (0.999, 0.99, 0.9, 0.0)
EXPECT = Q.Expectation("query-interpreter-fixture", "query-policy-v1", "event-v1", "dist-fixture", GRID)
RECORDS = [(f"q{i}", 0.9995, True) for i in range(400)]


# --- the fixture world -------------------------------------------------------------------------------------------
BASE = W.route_world(3, n_countries=6)
AIRPORTS = BASE.types["Airport"].values
S_, A, B, C_, D_ = AIRPORTS[:5]
T, U_ = "operator_3", "operator_1"


def world_with(label, **operators) -> W.World:
    facts = {r: dict(t) for r, t in BASE.facts.items()}
    facts["connects_to"].update({S_: A, A: B, B: C_, C_: D_, D_: D_})
    ops = {S_: U_, A: U_, B: U_, C_: T, D_: U_}
    ops.update({{"a": A, "b": B, "c": C_, "d": D_}[k]: v for k, v in operators.items()})
    facts["operator_of"].update(ops)
    return W.World(f"fixture+{label}", BASE.types, BASE.relations, facts, BASE.seed)


W0 = world_with("s0")
QUERY = R.UntilQuery(BASE.types["Airport"].value(S_), BASE.types["Operator"].value(T))


def context(world, registry=None):
    snap = S.publish(registry or G.table_registry(world))
    return V.ServingContext(snap, Q.certify(RECORDS, EXPECT, snap.digest), EXPECT, world.identity)


def candidate(base_registry, new_world, relation="operator_of"):
    reg = G.Registry(new_world.relations)
    for rid in sorted(new_world.relations):
        reg = reg.register(G.table_module(new_world, rid) if rid == relation else base_registry.module(rid))
    return reg


def deploy(c0, old_world, new_world, intended, scope=("operator_of",)):
    """Check and activate the candidate for new_world from c0. Returns (deployment, C1, check result, record)."""
    checker = U.Checker()
    dep = V.ServingDeployment(c0, D.Authority({PROPOSER}), checker)
    cand = candidate(c0.snapshot.registry, new_world)
    cert = Q.certify(RECORDS, EXPECT, S.digest_of(cand))
    rec = D.UpdateRecord(c0.snapshot.digest, S.digest_of(cand), scope, intended, new_world.identity, checker.version,
                         U.calibration_identity(cert), None, PROPOSER)
    res = checker.check(c0.snapshot, cand, rec, old_world, new_world, calibration=cert, expected=EXPECT)
    c1 = V.context_for(res, cert, EXPECT, new_world.identity)
    decision = dep.activate(rec, res, c1)
    assert decision["activated"], (decision, res.outcomes())
    return dep, c1, res, rec


def uninterrupted(ctx, budget=16, query=QUERY):
    return SS.advance(SS.open_session(ctx, query, budget, BASE.types), ctx, BASE.types)


def paused(ctx, k, budget=16, query=QUERY):
    return SS.advance(SS.open_session(ctx, query, budget, BASE.types), ctx, BASE.types, calls=k)


def tables_of(*contexts):
    return {c.snapshot.digest: SS.atomic_table(c.snapshot.registry, BASE.types) for c in contexts}


def same_run(a, b):
    return json.dumps(a["state"], sort_keys=True) == json.dumps(b["state"], sort_keys=True)


# --- the bounded-query contract and pause indices -------------------------------------------------------------------
def test_budgets_outside_the_contract_are_refused_before_anything_runs():
    c0 = context(W0)
    for budget in (0, 17, -1, True, 3.0, "3", None):
        s = uninterrupted(c0, budget)
        assert SS.outcome(s)["kind"] == SS.REFUSED and not s["state"]["applications"], budget
        assert SV.check_session(s, tables_of(c0))["ok"], budget
    for budget in (1, 16):
        assert SS.outcome(uninterrupted(c0, budget))["kind"] == "result"


def test_pause_indices_count_calls_to_step_including_the_terminating_one():
    c0 = context(W0)
    full = uninterrupted(c0)
    assert SS.outcome(full) == {"kind": "result", "answer": C_, "reason": "found"} and full["calls"] == 3
    dead = uninterrupted(c0, query=R.UntilQuery(BASE.types["Airport"].value(S_), BASE.types["Operator"].value("operator_5")))
    assert SS.outcome(dead)["reason"] == "dead end"
    assert dead["calls"] == dead["state"]["steps_used"] + 1  # the terminating call does not advance
    short = uninterrupted(c0, budget=2)
    assert SS.outcome(short)["reason"] == "budget exhausted" and short["calls"] == 3 and short["state"]["steps_used"] == 2


def test_pin_reproduces_the_uninterrupted_run_at_every_pause_index_for_every_query_and_budget():
    c0 = context(W0)
    tables = tables_of(c0)
    airports, operators = BASE.types["Airport"].values, BASE.types["Operator"].values
    checked = 0
    for start in airports:
        for target in operators:
            query = R.UntilQuery(BASE.types["Airport"].value(start), BASE.types["Operator"].value(target))
            for budget in (1, 3, 16):
                full = uninterrupted(c0, budget, query)
                check = SV.check_session(full, tables, W0)
                assert check["ok"] and check["equivalent_to"] == c0.snapshot.digest and check["correct"]
                for k in range(full["calls"]):  # every pause index of a paused (unfinished) run
                    resumed = SS.resume_pin(json.loads(json.dumps(paused(c0, k, budget, query))), c0, BASE.types)
                    assert same_run(resumed, full) and resumed["calls"] == full["calls"]
                    checked += 1
    assert checked > 500


# --- the policies through an update ------------------------------------------------------------------------------
@pytest.mark.parametrize("label,ops,intended", [
    ("witness", {"a": T, "b": T}, ((("operator_of", A, T)), ("operator_of", B, T))),
    ("later", {"c": U_, "d": T}, (("operator_of", C_, U_), ("operator_of", D_, T))),
])
def test_every_policy_at_every_pause_index_matches_its_declared_contract(label, ops, intended):
    c0, w1 = context(W0), world_with(label, **ops)
    dep, c1, _, _ = deploy(c0, W0, w1, intended)
    tables = tables_of(c0, c1)
    diff, _, _ = SS.actual_difference(c0.snapshot, c1.snapshot, BASE.types)
    assert diff.keys == {(r, x) for r, x, _ in intended}
    full0, full1 = uninterrupted(c0), uninterrupted(c1)
    eligible = 0
    for k in range(full0["calls"]):
        p = paused(c0, k)
        pin = SS.resume_pin(p, c0, BASE.types)
        restart = SS.resume_restart(p, dep.active, BASE.types)
        migrate = SS.resume_migrate(p, dep.active, diff, BASE.types)
        assert same_run(pin, full0) and same_run(restart, full1)
        assert SS.outcome(migrate) == SS.outcome(full1)
        assert migrate["state"]["visited"] == full1["state"]["visited"]
        for rec, expected in ((pin, c0), (restart, c1), (migrate, c1)):
            check = SV.check_session(rec, tables, w1 if expected is c1 else W0)
            assert check["ok"] and check["equivalent_to"] == expected.snapshot.digest, (k, rec["policy"], check)
            assert check["correct"]
        eligible += migrate["migration"]["migrated"]
        if migrate["migration"]["migrated"] and k > 0:  # mixed provenance, certified S1 equivalence
            assert [s["snapshot"] for s in migrate["segments"]] == [c0.snapshot.digest, c1.snapshot.digest]
            assert migrate["state"]["applications"][0]["module_identity"] == \
                c0.snapshot.registry.module("connects_to").version.identity  # the prefix keeps its identity
    assert eligible >= 1


def test_the_two_fact_witness_gives_an_outcome_no_version_produces():
    c0, w1 = context(W0), world_with("witness", a=T, b=T)
    dep, c1, _, _ = deploy(c0, W0, w1, (("operator_of", A, T), ("operator_of", B, T)))
    assert SS.outcome(uninterrupted(c0))["answer"] == C_ and SS.outcome(uninterrupted(c1))["answer"] == A
    naive = SS.continue_unchecked(paused(c0, 1), dep.active, BASE.types)  # a was tested under S0
    assert SS.outcome(naive)["answer"] == B
    check = SV.check_session(naive, tables_of(c0, c1), w1)
    assert not check["ok"] and check["unjustified_mixing"]
    assert check["contract_violation"] and check["no_version_outcome"]


def test_one_actual_change_gives_naive_continuation_one_versions_outcome():
    c0, w1 = context(W0), world_with("one", a=T)
    dep, c1, _, _ = deploy(c0, W0, w1, (("operator_of", A, T),))
    tables = tables_of(c0, c1)
    for k in range(1, uninterrupted(c0)["calls"]):
        check = SV.check_session(SS.continue_unchecked(paused(c0, k), dep.active, BASE.types), tables, w1)
        assert check["unjustified_mixing"] and not check["no_version_outcome"]  # some version's outcome


def test_contract_violations_are_judged_on_outcomes_not_on_which_version_produced_each_application():
    c0, w1 = context(W0), world_with("harmless", a="operator_5")  # a changes, but to another non-target operator
    dep, c1, _, _ = deploy(c0, W0, w1, (("operator_of", A, "operator_5"),))
    diff, _, _ = SS.actual_difference(c0.snapshot, c1.snapshot, BASE.types)
    p = paused(c0, 1)  # a tested under S0
    assert not SS.resume_migrate(p, dep.active, diff, BASE.types)["migration"]["migrated"]  # ineligible...
    check = SV.check_session(SS.continue_unchecked(p, dep.active, BASE.types), tables_of(c0, c1), w1)
    assert check["unjustified_mixing"] and not check["ok"]
    assert not check["contract_violation"] and not check["no_version_outcome"]  # ...yet S1's outcome


def test_the_reviews_correction_counterexample_blocks_migration():
    s0_actual = world_with("s0-actual")  # S0's modules: a = U, wrong
    old_ref = world_with("old-reference", a=T)  # the old reference already had a = T
    new_ref = world_with("new-reference", a=T, b=T)  # the update declares only b -> T
    c0 = context(s0_actual)
    dep, c1, res, rec = deploy(c0, old_ref, new_ref, (("operator_of", B, T),))
    assert res.passed()  # every gate passes: the candidate is correct everywhere
    assert res.counts["unaffected_changed_from_base"] > 0  # the gate counts the change at a, but does not fail on it
    diff, _, _ = SS.actual_difference(c0.snapshot, c1.snapshot, BASE.types)
    assert diff.keys == {("operator_of", A), ("operator_of", B)}
    classes = SS.classify(diff, old_ref, new_ref, rec.intended)
    assert [x[1] for x in classes["intended"]] == [B] and [x[1] for x in classes["correction"]] == [A]
    assert not classes["regression"]
    p = paused(c0, 1)  # a tested under S0
    declared_only = {(r, x) for r, x, _ in rec.intended}
    assert not ({("operator_of", A)} & declared_only)  # a declared-change rule would have migrated...
    naive = SS.continue_unchecked(p, dep.active, BASE.types)
    assert SS.outcome(naive)["answer"] == B  # ...to this, which is not S1's answer
    migrate = SS.resume_migrate(p, dep.active, diff, BASE.types)
    assert migrate["migration"] == {"migrated": False, "difference": diff.identity, "blocked_by": [["operator_of", A]]}
    assert SS.outcome(migrate)["answer"] == A == SS.outcome(uninterrupted(c1))["answer"]
    assert SV.check_session(migrate, tables_of(c0, c1), new_ref)["ok"]


# --- rollback and retained contexts ------------------------------------------------------------------------------
def test_a_rollback_while_paused_keeps_pin_on_c0_and_restarts_on_the_restored_context():
    c0, w1 = context(W0), world_with("witness", a=T, b=T)
    dep, c1, _, _ = deploy(c0, W0, w1, (("operator_of", A, T), ("operator_of", B, T)))
    p = paused(c0, 1)
    assert dep.rollback()["rolled_back"] and dep.active is c0
    assert same_run(SS.resume_pin(p, c0, BASE.types), uninterrupted(c0))
    assert same_run(SS.resume_restart(p, dep.active, BASE.types), uninterrupted(c0))


def test_a_pinned_session_keeps_its_context_after_a_rollback_removes_it():
    c0, w1 = context(W0), world_with("witness", a=T, b=T)
    dep, c1, _, _ = deploy(c0, W0, w1, (("operator_of", A, T), ("operator_of", B, T)))
    p = paused(dep.active, 0)  # started on C1 (which finds a at its first call, so it pauses before that call)
    assert dep.rollback()["rolled_back"] and dep.active is c0 and not dep.history
    resumed = SS.resume_pin(p, c1, BASE.types)  # C1 is no longer in the deployment, but still verifies
    assert same_run(resumed, uninterrupted(c1))


def test_an_in_place_change_while_paused_passes_recurrence_resume_but_not_pin():
    c0 = context(W0)
    p = paused(c0, 1)
    module = c0.snapshot.registry.module("operator_of")
    object.__setattr__(module, "table", {**module.table, B: T})  # declared identity unchanged
    assert c0.snapshot.registry.identity == p["state"]["registry_identity"]
    accepted = R.resume(copy.deepcopy(p["state"]), c0.snapshot.registry, BASE.types)  # recurrence alone accepts it
    assert accepted["answer"]["value"] == B
    with pytest.raises(SS.SessionRefused, match="no longer serve"):
        SS.resume_pin(p, c0, BASE.types)


def test_recurrence_refuses_to_resume_a_c0_state_on_c1():
    c0, w1 = context(W0), world_with("witness", a=T, b=T)
    _, c1, _, _ = deploy(c0, W0, w1, (("operator_of", A, T), ("operator_of", B, T)))
    with pytest.raises(ValueError, match="different registry"):
        R.resume(paused(c0, 1)["state"], c1.snapshot.registry, BASE.types)
    with pytest.raises(SS.SessionRefused):
        SS.resume_pin(paused(c0, 1), c1, BASE.types)  # pin to the wrong context


# --- stale certificates, stale records, wrong justifications -------------------------------------------------------
def test_a_certificate_for_the_old_snapshot_cannot_activate():
    c0, w1 = context(W0), world_with("witness", a=T, b=T)
    checker = U.Checker()
    dep = V.ServingDeployment(c0, D.Authority({PROPOSER}), checker)
    cand = candidate(c0.snapshot.registry, w1)
    stale = c0.certificate  # computed on S0's outputs
    rec = D.UpdateRecord(c0.snapshot.digest, S.digest_of(cand), ("operator_of",),
                         (("operator_of", A, T), ("operator_of", B, T)), w1.identity, checker.version,
                         U.calibration_identity(stale), None, PROPOSER)
    res = checker.check(c0.snapshot, cand, rec, W0, w1, calibration=stale, expected=EXPECT)
    assert res.outcome("calibration_and_resources") == U.FAIL
    assert not dep.activate(rec, res, V.context_for(res, stale, EXPECT, w1.identity))["activated"]
    assert dep.active is c0


def test_a_stale_update_record_is_refused_and_paused_sessions_are_unaffected():
    c0, w1 = context(W0), world_with("witness", a=T, b=T)
    dep, c1, _, _ = deploy(c0, W0, w1, (("operator_of", A, T), ("operator_of", B, T)))
    p = paused(c0, 1)
    w2 = world_with("later", c=U_, d=T)
    checker = dep.checker
    cand = candidate(c0.snapshot.registry, w2)
    cert = Q.certify(RECORDS, EXPECT, S.digest_of(cand))
    rec = D.UpdateRecord(c0.snapshot.digest, S.digest_of(cand), ("operator_of",),
                         (("operator_of", C_, U_), ("operator_of", D_, T)), w2.identity, checker.version,
                         U.calibration_identity(cert), None, PROPOSER)  # its base is S0, no longer active
    res = checker.check(c0.snapshot, cand, rec, W0, w2, calibration=cert, expected=EXPECT)
    decision = dep.activate(rec, res, V.context_for(res, cert, EXPECT, w2.identity))
    assert not decision["activated"] and any("stale" in r for r in decision["reasons"])
    assert dep.active is c1 and same_run(SS.resume_pin(p, c0, BASE.types), uninterrupted(c0))


def test_a_migration_justification_for_another_pair_is_refused():
    c0, w1 = context(W0), world_with("later", c=U_, d=T)
    dep, c1, _, _ = deploy(c0, W0, w1, (("operator_of", C_, U_), ("operator_of", D_, T)))
    other = context(world_with("other", d=T))
    wrong, _, _ = SS.actual_difference(c0.snapshot, other.snapshot, BASE.types)
    with pytest.raises(SS.SessionRefused, match="difference"):
        SS.resume_migrate(paused(c0, 1), dep.active, wrong, BASE.types)
    diff, _, _ = SS.actual_difference(c0.snapshot, c1.snapshot, BASE.types)
    good = SS.resume_migrate(paused(c0, 1), dep.active, diff, BASE.types)
    assert good["migration"]["migrated"]
    forged = copy.deepcopy(good)
    forged["migration"]["from"] = other.snapshot.digest
    check = SV.check_session(forged, tables_of(c0, c1, other), w1)
    assert not check["ok"] and check["unjustified_switch"]


def test_a_finished_session_cannot_be_resumed_and_a_tampered_record_is_caught():
    c0 = context(W0)
    done = uninterrupted(c0)
    for resume in (lambda s: SS.resume_pin(s, c0, BASE.types), lambda s: SS.resume_restart(s, c0, BASE.types),
                   lambda s: SS.continue_unchecked(s, c0, BASE.types)):
        with pytest.raises(SS.SessionRefused, match="not paused"):
            resume(done)
    tampered = copy.deepcopy(done)
    tampered["state"]["applications"][1]["produced"]["value"] = "operator_0"
    check = SV.check_session(tampered, tables_of(c0), W0)
    assert not check["ok"] and any("application 1" in p for p in check["problems"])


def test_the_reconstruction_is_independent_of_the_producer():
    import ast
    source = (ROOT / "integrated" / "session_verification.py").read_text()
    imported = {n.module.split(".")[-1] if isinstance(n, ast.ImportFrom) and n.module else a.name
                for n in ast.walk(ast.parse(source)) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert not imported & {"recurrence", "sessions", "update_checks", "runtime", "serving"}, imported
