"""The certificate/checker interface repair and complete serving contexts
(protocols/integration_test_v1.md, revision 2: "Implementation order", item 1).

The review of the integration test's revision 1 found four gaps, each reproduced and repaired here:
1. A fixed-sequence result was a plain dictionary with no bindings. The update checker could only judge PR 6's
   `Calibration`, and against that calibration's own interpreter identity.
2. `Checker.check` called the calibration gate without a policy version, so an honestly bound non-empty
   version failed.
3. Nothing required the certificate to have certified anything, or to match its own records.
4. Activation and rollback swapped only the registry snapshot, never the certificate it was checked with.

The legacy `Calibration` path is unchanged; the existing update and contract tests cover it.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from integrated import calibration as K  # noqa: E402
from integrated import certificates as Q  # noqa: E402
from integrated import contracts as C  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import serving as V  # noqa: E402
from integrated import snapshots as S  # noqa: E402
from integrated import update_checks as U  # noqa: E402
from integrated import updates as D  # noqa: E402

GRID = (0.999, 0.99, 0.9, 0.0)
PROPOSER = "offline-update"


def expectation(**changes) -> Q.Expectation:
    base = dict(interpreter_identity="interp-1", policy_version="query-policy-v1", event_version="event-v1",
                distribution_identity="dist-1", grid=GRID)
    return Q.Expectation(**{**base, **changes})


def records(n=400, wrong=0, low=0) -> list:
    """n calibration records; `wrong` errors at score 0.95, `low` correct records at score 0.5."""
    out = [(f"q{i}", 0.9995, True) for i in range(n)]
    out += [(f"w{i}", 0.95, False) for i in range(wrong)]
    out += [(f"l{i}", 0.5, True) for i in range(low)]
    return out


@pytest.fixture(scope="module")
def update():
    """A world, its table registry (S0), an updated world and a candidate implementing it (S1)."""
    world = W.ardel_world()
    facts = {r: dict(t) for r, t in world.facts.items()}
    facts["capital_of"]["Ardel"] = "Lunholm"
    new_world = W.World("ardel+move", world.types, world.relations, facts)
    base = G.table_registry(world)
    cand = G.Registry(world.relations)
    for rid in sorted(world.relations):
        cand = cand.register(G.table_module(new_world, rid) if rid == "capital_of" else base.module(rid))
    return world, new_world, base, cand, (("capital_of", "Ardel", "Lunholm"),)


def record_for(base, cand, new_world, intended, certificate, checker):
    base_digest = base.digest if isinstance(base, S.Snapshot) else S.digest_of(base)
    return D.UpdateRecord(base_digest, S.digest_of(cand), ("capital_of",), intended, new_world.identity,
                          checker.version, U.calibration_identity(certificate), None, PROPOSER)


# --- the certificate ---------------------------------------------------------------------------------------------
def test_a_certificate_keeps_the_whole_fixed_sequence_result_and_rederives_from_its_records():
    cert = Q.certify(records(400, wrong=1, low=50), expectation(), "digest-a")
    plain = K.fixed_sequence_certify([r[1] for r in cert.records], [r[2] for r in cert.records], GRID)
    assert cert.threshold == plain["threshold"] == 0.99 and cert.certified
    assert [list(s) for s in cert.steps] == [[s["threshold"], s["accepted"], s["errors"], s["p_value"], s["rejected"]]
                                            for s in plain["steps"]]
    assert cert.rederivation_problems() == []
    forged = dataclasses.replace(cert, threshold=0.0)  # claims a looser rule than its records support
    assert forged.rederivation_problems()
    first = cert.records[0]  # accepted at every tested rule: relabelling it changes the tested counts
    relabelled = dataclasses.replace(cert, records=((first[0], first[1], False),) + cert.records[1:])
    assert relabelled.rederivation_problems()
    last = cert.records[-1]  # score 0.5, below every tested rule: the result still re-derives...
    untested = dataclasses.replace(cert, records=cert.records[:-1] + ((last[0], last[1], not last[2]),))
    assert untested.rederivation_problems() == []
    assert len({cert.identity, forged.identity, relabelled.identity, untested.identity}) == 4  # ...but not its identity


def test_a_certificate_is_judged_against_the_serving_side_never_against_itself():
    cert = Q.certify(records(), expectation(), "digest-a")
    assert cert.binding_problems(expectation(), "digest-a") == []
    assert cert.binding_problems(expectation(), "digest-b")  # another snapshot's outputs labelled it
    for field, value in (("interpreter_identity", "interp-2"), ("policy_version", "query-policy-v2"),
                         ("event_version", "event-v2"), ("distribution_identity", "dist-2"),
                         ("grid", (0.999, 0.0)), ("alpha", 0.02)):
        assert cert.binding_problems(expectation(**{field: value}), "digest-a"), field


def test_the_effective_release_rule_is_the_stricter_threshold_supported_by_the_releasing_certificate():
    strict = Q.certify(records(400, wrong=1, low=50), expectation(), "s0")  # selects 0.99
    loose = Q.certify(records(400, low=0), expectation(), "s1")  # every rule rejected: selects 0.0
    rule = Q.effective_release(strict, loose)
    assert rule["threshold"] == 0.99 and rule["supporting_step"][0] == 0.99 and rule["supporting_step"][4]
    assert Q.effective_release(loose, strict)["threshold"] == 0.99
    other = Q.certify(records(), expectation(interpreter_identity="interp-2"), "s1")
    assert Q.effective_release(strict, other)["threshold"] is None  # another interpreter: no shared score
    never = Q.certify(records(100), expectation(), "s1")  # too few records to certify anything
    assert not never.certified and Q.effective_release(strict, never)["threshold"] is None


# --- the checker interface ---------------------------------------------------------------------------------------
def test_the_calibration_gate_judges_a_certificate_against_the_expected_context(update):
    world, new_world, base, cand, intended = update
    checker = U.Checker()
    good = Q.certify(records(), expectation(), S.digest_of(cand))
    rec = record_for(base, cand, new_world, intended, good, checker)
    res = checker.check(base, cand, rec, world, new_world, calibration=good, expected=expectation())
    assert res.passed(), res.outcomes()  # an honest, non-empty policy version passes
    assert res.expectation_identity == expectation().identity
    assert checker.check(base, cand, rec, world, new_world, calibration=good).outcome(
        "calibration_and_resources") == U.INSUFFICIENT  # no expected context: insufficient evidence

    def gate(cert, expected=expectation()):
        r = record_for(base, cand, new_world, intended, cert, checker)
        return checker.check(base, cand, r, world, new_world, calibration=cert, expected=expected).gates[
            "calibration_and_resources"]

    stale = Q.certify(records(), expectation(), S.digest_of(base))  # labelled by S0's modules
    assert gate(stale)["outcome"] == U.FAIL and any("snapshot" in p for p in gate(stale)["problems"])
    assert gate(good, expectation(interpreter_identity="interp-2"))["outcome"] == U.FAIL  # not the one serving
    assert gate(good, expectation(policy_version="query-policy-v2"))["outcome"] == U.FAIL
    assert good.threshold == 0.0 and gate(dataclasses.replace(good, threshold=0.99))["outcome"] == U.FAIL  # no re-derivation
    assert gate(Q.certify(records(100), expectation(), S.digest_of(cand)))["outcome"] == U.FAIL  # certified nothing


def test_the_record_must_name_the_certificate_it_was_checked_with(update):
    world, new_world, base, cand, intended = update
    checker = U.Checker()
    good = Q.certify(records(), expectation(), S.digest_of(cand))
    other = Q.certify(records(450), expectation(), S.digest_of(cand))
    rec = record_for(base, cand, new_world, intended, other, checker)
    res = checker.check(base, cand, rec, world, new_world, calibration=good, expected=expectation())
    assert res.outcome("record_context") == U.FAIL


# --- complete serving contexts -----------------------------------------------------------------------------------
@pytest.fixture()
def deployed(update):
    world, new_world, base, cand, intended = update
    checker = U.Checker()
    s0 = S.publish(base)
    c0 = V.ServingContext(s0, Q.certify(records(), expectation(), s0.digest), expectation(), world.identity)
    dep = V.ServingDeployment(c0, D.Authority({PROPOSER}), checker)
    cert1 = Q.certify(records(410), expectation(), S.digest_of(cand))
    rec = record_for(s0, cand, new_world, intended, cert1, checker)
    res = checker.check(s0, cand, rec, world, new_world, calibration=cert1, expected=expectation())
    return dep, c0, rec, res, cert1, new_world


def test_activation_installs_and_rollback_restores_the_complete_context(deployed):
    dep, c0, rec, res, cert1, new_world = deployed
    c1 = V.context_for(res, cert1, expectation(), new_world.identity)
    decision = dep.activate(rec, res, c1)
    assert decision["activated"] and decision["context"] == c1.identity
    assert dep.active is c1 and dep.active.certificate is cert1 and dep.registry is res.snapshot.registry
    rolled = dep.rollback()
    assert rolled["rolled_back"] and dep.active is c0 and dep.active.certificate is c0.certificate
    assert dep.rollback()["rolled_back"] is False and dep.active is c0  # nothing earlier


def test_activation_refuses_a_context_that_is_not_the_one_checked(deployed):
    dep, c0, rec, res, cert1, new_world = deployed
    other_cert = Q.certify(records(420), expectation(), res.snapshot.digest)
    cases = {
        "another certificate": V.context_for(res, other_cert, expectation(), new_world.identity),
        "a copy of the snapshot": V.ServingContext(S.publish(res.snapshot.registry), cert1, expectation(),
                                                   new_world.identity),
        "another world": V.context_for(res, cert1, expectation(), "another-world"),
        "another interpreter": V.context_for(res, Q.certify(records(410), expectation(interpreter_identity="i2"),
                                                            res.snapshot.digest),
                                             expectation(interpreter_identity="i2"), new_world.identity),
    }
    for name, context in cases.items():
        decision = dep.activate(rec, res, context)
        assert not decision["activated"], name
        assert dep.active is c0, name


def test_a_result_checked_without_the_expectation_cannot_activate(update):
    world, new_world, base, cand, intended = update
    checker = U.Checker()
    s0 = S.publish(base)
    c0 = V.ServingContext(s0, Q.certify(records(), expectation(), s0.digest), expectation(), world.identity)
    dep = V.ServingDeployment(c0, D.Authority({PROPOSER}), checker)
    cert1 = Q.certify(records(410), expectation(), S.digest_of(cand))
    rec = record_for(s0, cand, new_world, intended, cert1, checker)
    res = checker.check(s0, cand, rec, world, new_world, calibration=cert1)  # no expected context
    assert not dep.activate(rec, res, V.context_for(res, cert1, expectation(), new_world.identity))["activated"]


def test_a_retained_context_outlives_rollback_and_is_reverified_before_use(deployed):
    dep, c0, rec, res, cert1, new_world = deployed
    c1 = V.context_for(res, cert1, expectation(), new_world.identity)
    assert dep.activate(rec, res, c1)["activated"]
    pinned = dep.active  # a paused session holding C1
    assert dep.rollback()["rolled_back"] and dep.active is c0
    assert c1 not in dep.history and pinned.verify()  # gone from the deployment, still usable
    table = dict(pinned.snapshot.registry.module("capital_of").table)
    table["Ardel"] = "Ardel_Crossing"  # an in-place change behind the snapshot's back
    object.__setattr__(pinned.snapshot.registry.module("capital_of"), "table", table)
    assert not pinned.verify() and any("digest" in p for p in pinned.problems())


def test_rollback_refuses_a_stored_context_that_can_no_longer_serve(deployed):
    dep, c0, rec, res, cert1, new_world = deployed
    c1 = V.context_for(res, cert1, expectation(), new_world.identity)
    assert dep.activate(rec, res, c1)["activated"]
    module = c0.snapshot.registry.module("capital_of")
    object.__setattr__(module, "table", {**module.table, "Ardel": "Lunholm"})
    decision = dep.rollback()
    assert not decision["rolled_back"] and dep.active is c1


def test_the_initial_context_must_be_able_to_serve(update):
    world, _, base, cand, _ = update
    s0 = S.publish(base)
    with pytest.raises(ValueError):
        V.ServingDeployment(V.ServingContext(s0, Q.certify(records(), expectation(), S.digest_of(cand)),
                                             expectation(), world.identity), D.Authority({PROPOSER}), U.Checker())
