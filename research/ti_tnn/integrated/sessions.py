"""Sessions: bounded until-queries that pause, survive an update, and resume under a declared version policy
(protocols/integration_test_v1.md, revision 2).

**Example.** "Which is the first airport within 6 connections of S that operator_3 operates?" starts on
serving context C0. It pauses after two calls to `step`. A checked update to the operator of the second
airport is then activated as C1. On resume:
- **`pin`** finishes on C0, the context the session started on, after re-verifying it;
- **`restart`** discards the paused run and starts again on the active context;
- **`migrate`** continues on C1 only if no application in the paused prefix consumed a (relation, input)
  whose output differs between S0 and S1 (the *actual* difference, from the snapshots' atomic-output tables).
  Otherwise it restarts.

**What a session records.**
- **The query and its budget contract:** an integer from 1 to 16, checked before anything runs.
- **The recurrence state:** `recurrence.py`, unchanged, does every step.
- **Calls:** the number of calls to `step` made while the run was running. That is the pause index; it
  counts the terminating call, which does not increase `steps_used`.
- **Segments:** which snapshot supplied each application, by digest. A prefix is never relabelled.
- **The migration justification,** when there is one.
- **Costs:** module applications, snapshot verifications, difference lookups and revalidations.

**Provenance and semantic equivalence are separate.** A valid migration has applications from two snapshots.
Its justification shows that the prefix is what S1 would have produced:
- every consumed input lies outside the actual difference;
- S1 gives every recorded output again.

`session_verification.py` checks this independently. `continue_unchecked` is the experiment's control,
naive continuation. It is never a policy, and its records carry no justification.

**Limits.** Modules must be pure functions of their input, as in `recurrence.py`. The atomic table covers
every valid input of every relation, which is only possible in a finite world.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from . import recurrence as Q
from .contracts import content_identity
from .runtime import validated_output

SCHEMA = "ti_tnn_session_v1"
MIN_BUDGET, MAX_BUDGET = 1, 16
REFUSED = "refused"


class SessionRefused(ValueError):
    """A resume the declared policy does not allow."""


# --- the budget contract -----------------------------------------------------------------------------------------
def contract_problems(query: Q.UntilQuery, budget) -> list[str]:
    """The bounded-query contract, checked before execution (`recurrence.query_problems` does not check budgets)."""
    if type(budget) is not int:  # exactly int: True, 3.0 and "3" are not budgets
        return [f"budget {budget!r} is not an integer"]
    if not MIN_BUDGET <= budget <= MAX_BUDGET:
        return [f"budget {budget} is outside {MIN_BUDGET}-{MAX_BUDGET}"]
    return []


# --- atomic tables and the actual difference ------------------------------------------------------------------
def atomic_table(registry, types) -> dict:
    """Every relation's output on every valid input: {relation: {input: output or None}}. No weights."""
    out = {}
    for rid in sorted(registry.modules):
        spec, module = registry.relations[rid], registry.module(rid)
        row = {}
        for x in types[spec.source].values:
            produced, problem = validated_output(module.apply(types[spec.source].value(x)), spec, types)
            row[x] = None if produced is None or problem else produced.value
        out[rid] = row
    return out


def table_digest(table: dict) -> str:
    return content_identity(table)


@dataclass(frozen=True)
class Difference:
    """The actual atomic-output difference between two snapshots, bound to both digests and both tables."""

    base_digest: str
    candidate_digest: str
    base_table_digest: str
    candidate_table_digest: str
    changes: tuple[tuple[str, str, str | None, str | None], ...]  # (relation, input, old output, new output)

    @property
    def identity(self) -> str:
        return content_identity({"base": self.base_digest, "candidate": self.candidate_digest,
                                 "base_table": self.base_table_digest, "candidate_table": self.candidate_table_digest,
                                 "changes": [list(c) for c in self.changes]})

    @property
    def keys(self) -> frozenset:
        return frozenset((r, x) for r, x, _, _ in self.changes)


def actual_difference(base_snapshot, candidate_snapshot, types) -> tuple[Difference, dict, dict]:
    """(difference, base table, candidate table), from the snapshots' modules on every valid input."""
    t0, t1 = atomic_table(base_snapshot.registry, types), atomic_table(candidate_snapshot.registry, types)
    changes = tuple((r, x, t0[r][x], t1[r][x]) for r in sorted(t0) for x in sorted(t0[r]) if t0[r][x] != t1[r][x])
    return (Difference(base_snapshot.digest, candidate_snapshot.digest, table_digest(t0), table_digest(t1), changes),
            t0, t1)


def classify(difference: Difference, old_world, new_world, declared) -> dict:
    """Intended (declared world changes), corrections (outside them: old output wrong, new output right) and
    regressions (anything else; a candidate with one fails `unaffected_behavior`)."""
    declared = {(r, x) for r, x, _ in declared}
    out = {"intended": [], "correction": [], "regression": []}
    for r, x, old, new in difference.changes:
        if (r, x) in declared:
            out["intended"].append([r, x, old, new])
        elif new == new_world.facts[r][x] and old != old_world.facts[r][x]:
            out["correction"].append([r, x, old, new])
        else:
            out["regression"].append([r, x, old, new])
    return out


# --- sessions ------------------------------------------------------------------------------------------------------
def open_session(context, query: Q.UntilQuery, budget, types) -> dict:
    """A session on `context` (a serving context), refused before execution if the contract or the types fail."""
    problems = contract_problems(query, budget)
    state = Q.begin(query, context.snapshot.registry, types, budget if not problems else MIN_BUDGET)
    if problems:
        state.update(status=Q.ABSTAINED, reason="rejected: budget outside the contract", problems=problems)
    return {"schema": SCHEMA, "budget": budget, "state": state, "calls": 0,
            "segments": [{"snapshot": context.snapshot.digest, "context": context.identity, "from_application": 0,
                          "from_call": 0, "how": "start"}],
            "migration": None, "restarted_from": None, "policy": None,
            "costs": {"applications": 0, "snapshot_verifications": 0, "difference_lookups": 0, "revalidations": 0}}


def running(session) -> bool:
    return session["state"]["status"] == Q.RUNNING


def _current_digest(session) -> str:
    return session["segments"][-1]["snapshot"]


def advance(session, context, types, calls: int | None = None) -> dict:
    """Up to `calls` more calls to `step` on `context` (all remaining if None). Returns a new session. The context
    must be the one the session's current segment names."""
    if context.snapshot.digest != _current_digest(session):
        raise SessionRefused("this session is not running on that snapshot")
    s = copy.deepcopy(session)
    done = 0
    while running(s) and (calls is None or done < calls):
        before = len(s["state"]["applications"])
        s["state"] = Q.step(s["state"], context.snapshot.registry, types)
        s["costs"]["applications"] += len(s["state"]["applications"]) - before
        s["calls"] += 1
        done += 1
    return s


def outcome(session) -> dict:
    """The released form: an answer, a negative terminal outcome, or a refusal (with its reason)."""
    st = session["state"]
    if st["status"] == Q.ANSWERED:
        return {"kind": "result", "answer": st["answer"]["value"], "reason": "found"}
    if st["reason"] in ("dead end", "cycle", "budget exhausted"):
        return {"kind": "result", "answer": None, "reason": st["reason"]}
    if st["status"] == Q.RUNNING:
        return {"kind": "running", "answer": None, "reason": ""}
    return {"kind": REFUSED, "answer": None, "reason": st["reason"]}


# --- version policies ------------------------------------------------------------------------------------------
def _paused(session, policy):
    """A resume needs a paused run: pause index k < n, where n is the uninterrupted run's number of calls."""
    if not running(session):
        raise SessionRefused(f"{policy}: the session is not paused; it has finished")


def resume_pin(session, pinned, types) -> dict:
    """Finish on the context the session is running on, after verifying it (digest and certificate)."""
    _paused(session, "pin")
    s = copy.deepcopy(session)
    s["costs"]["snapshot_verifications"] += 1
    if pinned.snapshot.digest != _current_digest(s):
        raise SessionRefused("pin: that is not the context this session runs on")
    problems = pinned.problems()
    if problems:
        raise SessionRefused(f"pin: the pinned context can no longer serve: {problems}")
    s["policy"] = "pin"
    return advance(s, pinned, types)


def resume_restart(session, active, types) -> dict:
    """Discard the paused run, and run the same query from the start on the active context."""
    _paused(session, "restart")
    problems = active.problems()
    if problems:
        raise SessionRefused(f"restart: the active context cannot serve: {problems}")
    q = session["state"]["query"]
    query = Q.UntilQuery(types[q["start"]["type_id"]].value(q["start"]["value"]),
                         types[q["target"]["type_id"]].value(q["target"]["value"]), q["step_relation"],
                         q["test_relation"])
    fresh = open_session(active, query, session["budget"], types)
    fresh["costs"]["snapshot_verifications"] += 1
    fresh["restarted_from"] = {"calls": session["calls"], "applications": len(session["state"]["applications"]),
                               "snapshot": _current_digest(session)}
    fresh["costs"]["discarded_applications"] = session["costs"]["applications"]
    fresh["policy"] = "restart"
    return advance(fresh, active, types)


def resume_migrate(session, active, difference: Difference, types) -> dict:
    """Continue on the active context if the prefix consumed nothing in the actual difference, after
    revalidating every prefix output on it; otherwise restart. The prefix keeps its original provenance."""
    _paused(session, "migrate")
    s = copy.deepcopy(session)
    if (difference.base_digest, difference.candidate_digest) != (_current_digest(s), active.snapshot.digest):
        raise SessionRefused("migrate: the difference is not between this session's snapshot and the active one")
    problems = active.problems()
    s["costs"]["snapshot_verifications"] += 1
    if problems:
        raise SessionRefused(f"migrate: the active context cannot serve: {problems}")
    apps = s["state"]["applications"]
    consumed = sorted({(a["relation_id"], a["consumed"]["value"]) for a in apps})
    s["costs"]["difference_lookups"] += len(consumed)
    blocked = [list(k) for k in consumed if k in difference.keys]

    def fall_back(**why):  # restart, keeping the cost of the checks that refused the migration
        out = resume_restart(s, active, types)
        for key in ("snapshot_verifications", "difference_lookups", "revalidations"):
            out["costs"][key] += s["costs"][key]
        out["migration"] = {"migrated": False, "difference": difference.identity, **why}
        out["policy"] = "migrate"
        return out

    if blocked:
        return fall_back(blocked_by=blocked)
    registry = active.snapshot.registry
    mismatched = []
    for a in apps:  # revalidation: S1 must give every recorded output again
        spec = registry.relations[a["relation_id"]]
        value = types[a["consumed"]["type_id"]].value(a["consumed"]["value"])
        produced, problem = validated_output(registry.module(a["relation_id"]).apply(value), spec, types)
        again = None if produced is None or problem else produced.value
        recorded = None if a["produced"] is None else a["produced"]["value"]
        s["costs"]["revalidations"] += 1
        if again != recorded:
            mismatched.append(a["index"])
    if mismatched:
        return fall_back(revalidation_failed=mismatched)
    s["segments"].append({"snapshot": active.snapshot.digest, "context": active.identity,
                          "from_application": len(apps), "from_call": s["calls"], "how": "migrate"})
    s["migration"] = {"migrated": True, "difference": difference.identity, "from": difference.base_digest,
                      "to": difference.candidate_digest, "prefix_inputs": [list(k) for k in consumed],
                      "blocked_by": [], "revalidated": len(apps)}
    s["policy"] = "migrate"
    return advance(s, active, types)


def continue_unchecked(session, active, types) -> dict:
    """THE CONTROL (naive continuation), never a policy: keep the paused state and continue on the active context
    with no checks. Its record has two segments and no justification."""
    _paused(session, "naive")
    s = copy.deepcopy(session)
    s["segments"].append({"snapshot": active.snapshot.digest, "context": active.identity,
                          "from_application": len(s["state"]["applications"]), "from_call": s["calls"],
                          "how": "unchecked"})
    s["policy"] = "naive"
    return advance(s, active, types)
