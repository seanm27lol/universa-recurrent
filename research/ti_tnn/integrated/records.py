"""Execution records: schema `ti_tnn_execution_v1` (integration plan section 5.6, PR 5).

Example: running `largest_city_of, airport_of, operator_of` on Ardel produces one record. It holds:
- the program, its input, the budget and the world's identity;
- every registered module version, and the registry identity they hash to;
- each step's module and the exact typed values it consumed and produced, marked when an intervention
  supplied the consumed value;
- the outcome: status, decision, answer, operation counts and problems;
- a digest over everything else.

The record is plain JSON. It is built from the `Execution` the runtime returned, never written afterwards
by a second model, so its values are the ones the computation passed.

`record_problems` checks the schema and the record's internal consistency:
- each step consumes what the previous step produced, unless marked intervened;
- the answer is the last value produced;
- the counts and the registry identity add up;
- the digest matches.

It does not say whether the answer is right; that is `verification.py`, against the world's facts. A
digest identifies a record; it authenticates nothing.
"""
from __future__ import annotations

from typing import Any

from .contracts import Budget, Execution, Program, TypedValue, content_identity

SCHEMA = "ti_tnn_execution_v1"
ROUTE_SUPPLIED = "supplied_program"  # no router yet (plan PR 6): the program is given, not chosen

_TOP = ("schema", "input_identity", "world_identity", "program", "budget", "registry_identity", "modules",
        "route", "trace", "status", "decision", "answer", "operation_counts", "problems", "digest")
_STEP = ("index", "relation_id", "module_id", "module_identity", "consumed", "produced", "intervened", "cost")


def value_record(v: TypedValue | None) -> dict | None:
    return None if v is None else {"type_id": v.type_id, "value": v.value, "type_version": v.type_version}


def registry_identity(modules: list[dict]) -> str:
    """The registry identity recomputed from recorded module versions (matches `Registry.identity`)."""
    return content_identity({m["relation_id"]: [m["module_id"], m["identity"]] for m in modules})


def to_record(execution: Execution, program: Program, registry, budget: Budget, world_identity: str) -> dict:
    """The record of one execution, sealed with its digest."""
    modules = [{"relation_id": rid, "module_id": m.version.module_id, "identity": m.version.identity,
                "kind": m.version.kind, "precision": m.version.precision, "calibration": m.version.calibration}
               for rid, m in sorted(registry.modules.items())]
    program_rec = {"steps": list(program.steps), "input": value_record(program.input),
                   "expected_output_type": program.expected_output_type}
    trace = [{"index": s.index, "relation_id": s.relation_id, "module_id": s.module_id,
              "module_identity": s.module_identity, "consumed": value_record(s.consumed),
              "produced": value_record(s.produced), "intervened": s.intervened, "cost": s.cost}
             for s in execution.trace]
    d = execution.decision
    record = {
        "schema": SCHEMA,
        "input_identity": content_identity(program_rec),
        "world_identity": world_identity,
        "program": program_rec,
        "budget": {"max_steps": budget.max_steps, "policy": budget.policy},
        "registry_identity": registry.identity,
        "modules": modules,
        "route": {"source": ROUTE_SUPPLIED, "distribution": None},
        "trace": trace,
        "status": execution.status,
        "decision": {"kind": d.kind, "relation_id": d.relation_id, "module_id": d.module_id, "reason": d.reason},
        "answer": value_record(execution.answer),
        "operation_counts": {"module_applications": len(trace), "cost": sum(s["cost"] for s in trace)},
        "problems": list(execution.problems),
    }
    return seal(record)


def seal(record: dict) -> dict:
    """The record with its digest (over every other field) set."""
    body = {k: v for k, v in record.items() if k != "digest"}
    return {**body, "digest": content_identity(body)}


def _is_value(v: Any) -> bool:
    return (isinstance(v, dict) and set(v) == {"type_id", "value", "type_version"}
            and isinstance(v["type_id"], str) and isinstance(v["value"], str) and isinstance(v["type_version"], int))


def record_problems(record: Any) -> list[str]:
    """Schema and internal-consistency problems (empty if none). Says nothing about correctness."""
    if not isinstance(record, dict):
        return ["a record must be a JSON object"]
    missing = [k for k in _TOP if k not in record]
    extra = [k for k in record if k not in _TOP]
    if missing or extra:
        return ([f"missing fields {missing}"] if missing else []) + ([f"unknown fields {extra}"] if extra else [])
    out = []
    if record["schema"] != SCHEMA:
        out.append(f"schema is {record['schema']!r}, not {SCHEMA}")
    if record["digest"] != seal(record)["digest"]:
        out.append("digest does not match the record")
    program = record["program"]
    if not (isinstance(program, dict) and isinstance(program.get("steps"), list) and _is_value(program.get("input"))):
        return out + ["malformed program"]
    if record["input_identity"] != content_identity(program):
        out.append("input identity does not match the program")
    modules = record["modules"]
    if not isinstance(modules, list) or record["registry_identity"] != registry_identity(modules):
        out.append("registry identity does not match the recorded module versions")
    by_relation = {m["relation_id"]: m for m in modules} if isinstance(modules, list) else {}
    trace = record["trace"]
    if not isinstance(trace, list):
        return out + ["trace must be a list"]
    previous = program["input"]
    for i, step in enumerate(trace):
        if not isinstance(step, dict) or set(step) != set(_STEP):
            out.append(f"step {i}: malformed")
            break
        if step["index"] != i or i >= len(program["steps"]) or step["relation_id"] != program["steps"][i]:
            out.append(f"step {i}: does not match the program")
        m = by_relation.get(step["relation_id"])
        if m is None or (m["module_id"], m["identity"]) != (step["module_id"], step["module_identity"]):
            out.append(f"step {i}: module version is not the registered one")
        if not _is_value(step["consumed"]) or (step["produced"] is not None and not _is_value(step["produced"])):
            out.append(f"step {i}: malformed value")
            break
        if not step["intervened"] and step["consumed"] != previous:
            out.append(f"step {i}: consumed a value the previous step did not produce")
        previous = step["produced"]
    answered = record["status"] == "answered"
    if answered != (record["answer"] is not None):
        out.append("an answer is present exactly when the status is answered")
    if answered and (len(trace) != len(program["steps"]) or record["answer"] != previous):
        out.append("the answer is not the last value produced")
    counts = record["operation_counts"]
    if counts != {"module_applications": len(trace), "cost": sum(s.get("cost", 0) for s in trace if isinstance(s, dict))}:
        out.append("operation counts do not match the trace")
    if len(trace) > record["budget"]["max_steps"]:
        out.append("the trace exceeds the budget")
    return out
