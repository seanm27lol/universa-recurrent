"""Type-checked execution of a supplied program, recording the values actually consumed.

Example: `largest_city_of, airport_of, operator_of` on Ardel. Before any module runs, the runtime checks
that each step's source type is the previous step's target type, that the input is a Country, and that
every relation has a registered module. An ill-typed program (say `capital_of, operator_of`: a City is not
an Airport) is rejected without executing anything. A valid one runs step by step:
- each module receives only the current typed value, not the question, the trace or anything else;
- its output must have the relation's declared target type and lie in that type's domain;
- the output is rebuilt from the declared domain's own plain values (`TypeSpec.value`), so a module cannot pass
  hidden data along with it, not in a TypedValue subclass and not in a `str` subclass inside one;
- the value recorded in the trace is the exact object passed to the next module;
- the budget bounds the number of module applications; running out abstains, with the partial trace.

An intervention (plan PR 5) replaces the value a given step consumes with another value of the same type,
checked before anything runs, and the trace marks that step as intervened. The runtime cannot stop a module
from reading state it keeps itself (a shared context, a cached input); `interventions.py` tests for that.

What this establishes, by construction: type validity and a trace that records the values passed. It does
not establish that the answer is right; that is the reference evaluator's job (`reference.py`), and the
two are compared, never merged.
"""
from __future__ import annotations

from typing import Mapping

from .contracts import (Budget, Decision, Execution, Program, RunState, StepRecord, TypedValue, TypeSpec, canonical)
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


def validated_output(raw, spec, types) -> tuple[TypedValue | None, str | None]:
    """(the value to pass on, problem). A valid output is rebuilt from the target type's declared domain; an
    invalid one is returned only as a plain record of what the module gave, never passed on."""
    if raw is None:
        return None, None
    try:
        ok = raw.type_id == spec.target and types[spec.target].contains(raw.value)
    except Exception:
        ok = False
    if ok:
        return types[spec.target].value(raw.value), None
    try:
        shown = canonical(raw)
    except Exception:
        shown = None
    return shown, f"{shown!r} is not a {spec.target}"


def intervention_problems(program: Program, registry: Registry, types: Mapping[str, TypeSpec],
                          interventions: Mapping[int, TypedValue]) -> list[str]:
    """Why a set of interventions cannot apply to a (well-typed) program; empty if they can."""
    out = []
    for i, v in sorted(interventions.items()):
        if not 0 <= i < len(program.steps):
            out.append(f"intervention at step {i}: the program has {len(program.steps)} steps")
            continue
        need = registry.relations[program.steps[i]].source
        if v.type_id != need or v.type_id not in types or not types[need].contains(v.value):
            out.append(f"intervention at step {i}: {v!r} is not a {need}")
    return out


def execute(program: Program, registry: Registry, types: Mapping[str, TypeSpec], budget: Budget,
            interventions: Mapping[int, TypedValue] | None = None) -> Execution:
    """Run a supplied program under a budget; see the module docstring for the contract. `interventions`
    maps a step index to the value that step consumes instead of the previous step's output."""
    interventions = dict(interventions or {})
    problems = program_problems(program, registry, types)
    if not problems:
        problems = intervention_problems(program, registry, types, interventions)
    if problems:
        return Execution("rejected", Decision("abstain", reason="ill-typed program"), None, (), None, tuple(problems))
    value: TypedValue = types[registry.relations[program.steps[0]].source].value(program.input.value)
    trace: list[StepRecord] = []
    identity = registry.identity
    for i, rid in enumerate(program.steps):
        intervened = i in interventions
        if intervened:
            value = types[registry.relations[rid].source].value(interventions[i].value)
        state = RunState(value, i, len(trace), identity)
        if len(trace) >= budget.max_steps:
            return Execution("abstained", Decision("abstain", rid, reason="budget exhausted"), None, tuple(trace), state)
        module = registry.module(rid)
        spec = registry.relations[rid]
        raw = module.apply(value)
        produced, problem = validated_output(raw, spec, types)  # nothing a module attaches travels further
        trace.append(StepRecord(i, rid, module.version.module_id, module.version.identity, value, produced,
                                intervened=intervened))
        if raw is None:
            return Execution("abstained", Decision("abstain", rid, module.version.module_id, "module gave no answer"),
                             None, tuple(trace), state)
        if problem:
            return Execution("rejected", Decision("abstain", rid, module.version.module_id,
                                                  "module produced a value outside its declared type"),
                             None, tuple(trace), state, (f"step {i}: {problem}",))
        value = produced  # exactly the recorded value is what the next module receives
    final = RunState(value, len(program.steps), len(trace), identity)
    return Execution("answered", Decision("commit"), value, tuple(trace), final)
