"""An independent checker for execution records (integration plan section 5.6, PR 5).

It reads two things: a record (`records.py` schema) and the world's facts. It never imports the runtime,
the registry, any module or PyTorch (a test checks this), so it cannot share a bug with the computation it
checks.

Example: a record of `capital_of, airport_of` on Ardel, made with a `capital_of` module that secretly
returns the largest city, passes every schema and type check. Its semantics field shows step 0 wrong:
Port Vey was produced, the facts say Ardel Crossing.

It reports separate fields, because none implies another:
- `schema`: the record is well formed and internally consistent;
- `types`: every value lies in its declared type, and the relations exist in this world;
- `semantics`, compared with the facts:
  - each step's own correctness, given the value it actually consumed;
  - the answer's correctness against the reference, which follows the facts from the input and applies
    any recorded interventions at their steps;
- `integrity`: the digest matches;
- `provenance`: always "not authenticated". A record says what was computed; it does not prove where or
  by whom.

Replay and causal faithfulness need the computation itself, so they are in `interventions.py`.
"""
from __future__ import annotations

from .records import record_problems, seal
from .relations import World

CHECKER_VERSION = "ti_tnn_checker_v1"
PROVENANCE = "not authenticated: no signed execution provenance was supplied"


def _type_problems(record: dict, world: World) -> list[str]:
    out = []
    if record["world_identity"] != world.identity:
        out.append("the record was made in a different world")
    program = record["program"]

    def in_domain(v, need, where):
        if v["type_id"] != need or need not in world.types or not world.types[need].contains(v["value"]):
            out.append(f"{where}: {v['value']!r} ({v['type_id']}) is not a {need}")

    steps = program["steps"]
    if any(r not in world.relations for r in steps):
        return out + ["the program uses a relation this world does not have"]
    in_domain(program["input"], world.relations[steps[0]].source, "input")
    for a, b in zip(steps, steps[1:]):
        if world.relations[a].target != world.relations[b].source:
            out.append(f"{a} then {b} is ill-typed")
    for step in record["trace"]:
        spec = world.relations[step["relation_id"]]
        in_domain(step["consumed"], spec.source, f"step {step['index']} consumed")
        if step["produced"] is not None:
            in_domain(step["produced"], spec.target, f"step {step['index']} produced")
    return out


def _semantics(record: dict, world: World) -> dict:
    steps = record["program"]["steps"]
    intervened = {s["index"]: s["consumed"]["value"] for s in record["trace"] if s["intervened"]}
    value = record["program"]["input"]["value"]
    for i, rid in enumerate(steps):  # the reference: facts from the input, with recorded interventions applied
        value = intervened.get(i, value)
        value = world.facts[rid].get(value)
        if value is None:
            break
    step_correct = [s["produced"] is not None and world.facts[s["relation_id"]].get(s["consumed"]["value"]) == s["produced"]["value"]
                    for s in record["trace"]]
    answer = record["answer"]
    return {"reference_answer": value, "answer_correct": None if answer is None else answer["value"] == value,
            "step_correct": step_correct, "intervened_steps": sorted(intervened)}


def check(record: dict, world: World) -> dict:
    """Every check this checker can make from the record and the facts alone; see the module docstring."""
    schema = record_problems(record)
    out = {"checker": CHECKER_VERSION, "provenance": PROVENANCE,
           "schema": {"ok": not schema, "problems": schema}, "types": None, "semantics": None,
           "integrity": {"digest_matches": isinstance(record, dict) and "digest" in record
                         and record["digest"] == seal(record)["digest"]}}
    if schema:
        return out
    types = _type_problems(record, world)
    out["types"] = {"ok": not types, "problems": types}
    if not types:
        out["semantics"] = _semantics(record, world)
    return out
