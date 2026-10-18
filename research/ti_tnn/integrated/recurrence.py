"""Bounded recurrence with an explicit continuation state (integration plan section 5.5, PR 7).

Example: "which is the first airport reached from Vey International that Northline operates?" Each
iteration:
- applies `connects_to` to the current airport;
- stops as a dead end if the airport connects to itself;
- stops as a cycle if the route returns to an airport already visited;
- otherwise applies `operator_of` to the new airport, and answers if the operator is the target.

A budget bounds the iterations; running out abstains. The number of iterations is not known in advance;
that is what makes this recurrence rather than a fixed program.

**The continuation state is the whole run, as plain JSON:**
- the query, and the registry identity with the module versions it is bound to;
- the current value, the visited values, the steps used and the budget;
- every module application so far (consumed and produced values), the status and the reason.

`step` maps a state to the next state and touches nothing else. A run can therefore stop after any
iteration, be written out, and be resumed in another process, with a freshly built registry of the same
identity. It reaches exactly the same final state. `resume` refuses a state bound to a different
registry.

The modules must be pure functions of their input for this to hold; the runtime cannot enforce that. A
module keeping its own state across calls would make resumed and uninterrupted runs differ, and the tests
show that check catching one.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from .contracts import TypedValue, content_identity
from .records import value_record
from .runtime import validated_output

SCHEMA = "ti_tnn_continuation_v1"
RUNNING, ANSWERED, ABSTAINED = "running", "answered", "abstained"


@dataclass(frozen=True)
class UntilQuery:
    """Follow `step_relation` from `start` until `test_relation` of the value reached equals `target`."""

    start: TypedValue
    target: TypedValue
    step_relation: str = "connects_to"
    test_relation: str = "operator_of"


def query_problems(query: UntilQuery, registry, types) -> list[str]:
    """Type-check a query before anything runs (the recurrent analogue of `runtime.program_problems`)."""
    out = []
    step, test = registry.relations.get(query.step_relation), registry.relations.get(query.test_relation)
    if step is None or test is None:
        return ["unknown relation"]
    if step.source != step.target:
        out.append(f"{query.step_relation} must map a type to itself to be repeated")
    if query.start.type_id != step.source or not types[step.source].contains(query.start.value):
        out.append(f"start {query.start.value!r} is not a {step.source}")
    if test.source != step.target:
        out.append(f"{query.test_relation} does not apply to a {step.target}")
    if query.target.type_id != test.target or not types[test.target].contains(query.target.value):
        out.append(f"target {query.target.value!r} is not a {test.target}")
    for rid in (query.step_relation, query.test_relation):
        if rid not in registry.modules:
            out.append(f"no module registered for {rid}")
    return out


def begin(query: UntilQuery, registry, types, max_steps: int) -> dict:
    """The initial state, or a rejected one if the query is ill-typed."""
    problems = query_problems(query, registry, types)
    versions = {r: registry.module(r).version.identity for r in (query.step_relation, query.test_relation)
                if r in registry.modules}
    state = {"schema": SCHEMA,
             "query": {"start": value_record(query.start), "target": value_record(query.target),
                       "step_relation": query.step_relation, "test_relation": query.test_relation},
             "registry_identity": registry.identity, "module_versions": versions,
             "current": value_record(query.start), "visited": [query.start.value], "steps_used": 0,
             "max_steps": max_steps, "applications": [], "status": RUNNING, "reason": "", "answer": None,
             "problems": problems}
    if problems:
        state.update(status=ABSTAINED, reason="rejected: ill-typed query")
    return state


def _apply(state, registry, types, rid, value: TypedValue):
    module = registry.module(rid)
    produced, problem = validated_output(module.apply(value), registry.relations[rid], types)
    state["applications"].append({"index": len(state["applications"]), "relation_id": rid,
                                  "module_identity": module.version.identity, "consumed": value_record(value),
                                  "produced": value_record(produced)})
    return None if problem else produced


def step(state: dict, registry, types) -> dict:
    """One iteration. Returns a new state; the input state is not modified."""
    if state["status"] != RUNNING:
        return state
    s = copy.deepcopy(state)
    q = s["query"]
    if s["steps_used"] >= s["max_steps"]:
        s.update(status=ABSTAINED, reason="budget exhausted")
        return s
    current = TypedValue(**s["current"])
    nxt = _apply(s, registry, types, q["step_relation"], current)
    if nxt is None:
        s.update(status=ABSTAINED, reason="module gave no valid answer")
        return s
    if nxt.value == current.value:
        s.update(status=ABSTAINED, reason="dead end")
        return s
    if nxt.value in s["visited"]:
        s.update(status=ABSTAINED, reason="cycle")
        return s
    s["visited"].append(nxt.value)
    s["current"] = value_record(nxt)
    s["steps_used"] += 1
    tested = _apply(s, registry, types, q["test_relation"], nxt)
    if tested is None:
        s.update(status=ABSTAINED, reason="module gave no valid answer")
    elif tested.value == q["target"]["value"]:
        s.update(status=ANSWERED, answer=value_record(nxt))
    return s


def run(state: dict, registry, types, pause_after: int | None = None) -> dict:
    """Iterate until the run stops, or until `pause_after` iterations of this call (to test pausing)."""
    done = 0
    while state["status"] == RUNNING and (pause_after is None or done < pause_after):
        state = step(state, registry, types)
        done += 1
    return state


def resume(state: dict, registry, types) -> dict:
    """Continue a paused run, refusing a registry other than the one the state is bound to."""
    if registry.identity != state["registry_identity"]:
        raise ValueError("the state is bound to a different registry; it cannot be resumed with this one")
    return run(state, registry, types)


def state_identity(state: dict) -> str:
    return content_identity(state)


def fixed_depth(query: UntilQuery, registry, types, max_steps: int) -> dict:
    """The control: always run the full budget of `step_relation`, then `test_relation` on every value
    reached, then pick the first match before any dead end or cycle. Same answers, no early stopping."""
    state = begin(query, registry, types, max_steps)
    if state["status"] != RUNNING:
        return state
    current, path, stop = query.start, [], None
    for _ in range(max_steps):  # the whole budget, whatever happens
        nxt = _apply(state, registry, types, query.step_relation, current)
        if nxt is None:
            stop = stop or ("module gave no valid answer", len(path))
            nxt = current
        path.append(nxt)
        current = nxt
    tests = [_apply(state, registry, types, query.test_relation, v) for v in path]
    seen = [query.start.value]
    for i, (v, t) in enumerate(zip(path, tests)):
        if stop and i >= stop[1]:
            state.update(status=ABSTAINED, reason=stop[0])
            break
        if v.value == seen[-1]:
            state.update(status=ABSTAINED, reason="dead end")
            break
        if v.value in seen:
            state.update(status=ABSTAINED, reason="cycle")
            break
        seen.append(v.value)
        if t is None:  # recurrence abstains here too; scanning on would answer where it cannot
            state.update(status=ABSTAINED, reason="module gave no valid answer")
            break
        if t.value == query.target.value:
            state.update(status=ANSWERED, answer=value_record(v))
            break
    else:
        state.update(status=ABSTAINED, reason="budget exhausted")
    state.update(current=value_record(path[-1]) if path else state["current"], visited=seen, steps_used=len(path))
    return state

