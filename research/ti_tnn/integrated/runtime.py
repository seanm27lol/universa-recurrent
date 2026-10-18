"""Type-checked execution of a supplied program, recording the values actually consumed.

Example: `largest_city_of, airport_of, operator_of` on Ardel. Before any module runs, the runtime checks
that each step's source type is the previous step's target type, that the input is a Country, and that
every relation has a registered module. An ill-typed program (say `capital_of, operator_of`: a City is not
an Airport) is rejected without executing anything. A valid one runs step by step:
- each module receives only the current typed value, not the question, the trace or anything else;
- its output must have the relation's declared target type and lie in that type's domain;
- the value recorded in the trace is the exact object passed to the next module;
- the budget bounds the number of module applications; running out abstains, with the partial trace.

What this establishes, by construction: type validity and a trace that records the values passed. It does
not establish that the answer is right; that is the reference evaluator's job (`reference.py`), and the
two are compared, never merged.
"""
from __future__ import annotations

from typing import Mapping

from .contracts import (Budget, Decision, Execution, Program, RunState, StepRecord, TypedValue, TypeSpec)
from .registry import Registry


def program_problems(program: Program, registry: Registry, types: Mapping[str, TypeSpec]) -> list[str]:
    """Why a program cannot run (empty if it can). Checked in full before any module is called."""
    out = []
    if program.input.type_id not in types:
        return [f"input has unknown type {program.input.type_id}"]
    if not types[program.input.type_id].contains(program.input.value):
        out.append(f"input {program.input.value!r} is not a {program.input.type_id}")
    current = program.input.type_id
    for i, rid in enumerate(program.steps):
        spec = registry.relations.get(rid)
        if spec is None:
            out.append(f"step {i}: unknown relation {rid}")
            return out
        if spec.source != current:
            out.append(f"step {i}: {rid} expects type {spec.source}, got {current}")
            return out
        if rid not in registry.modules:
            out.append(f"step {i}: no module registered for {rid}")
        current = spec.target
    if program.expected_output_type is not None and current != program.expected_output_type:
        out.append(f"program ends in a {current}, not the expected {program.expected_output_type}")
    return out


def execute(program: Program, registry: Registry, types: Mapping[str, TypeSpec], budget: Budget) -> Execution:
    """Run a supplied program under a budget; see the module docstring for the contract."""
    problems = program_problems(program, registry, types)
    if problems:
        return Execution("rejected", Decision("abstain", reason="ill-typed program"), None, (), None, tuple(problems))
    value: TypedValue = program.input
    trace: list[StepRecord] = []
    identity = registry.identity
    for i, rid in enumerate(program.steps):
        state = RunState(value, i, len(trace), identity)
        if len(trace) >= budget.max_steps:
            return Execution("abstained", Decision("abstain", rid, reason="budget exhausted"), None, tuple(trace), state)
        module = registry.module(rid)
        spec = registry.relations[rid]
        produced = module.apply(value)
        trace.append(StepRecord(i, rid, module.version.module_id, module.version.identity, value, produced))
        if produced is None:
            return Execution("abstained", Decision("abstain", rid, module.version.module_id, "module gave no answer"),
                             None, tuple(trace), state)
        if produced.type_id != spec.target or not types[spec.target].contains(produced.value):
            return Execution("rejected", Decision("abstain", rid, module.version.module_id,
                                                  "module produced a value outside its declared type"),
                             None, tuple(trace), state, (f"step {i}: {produced!r} is not a {spec.target}",))
        value = produced  # exactly the recorded value is what the next module receives
    final = RunState(value, len(program.steps), len(trace), identity)
    return Execution("answered", Decision("commit"), value, tuple(trace), final)
