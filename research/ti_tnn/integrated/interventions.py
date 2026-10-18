"""Replay, interventions and disabled computation: does the trace describe the computation? (plan PR 5)

A record (`records.py`) says which values each module consumed. These tests check that the computation
really depended on those values and nothing else. Each needs the computation itself, which is why they
are not in the independent checker.

- **Replay.** Re-running the recorded program, with the recorded registry, reproduces the record exactly.
  A registry whose identity differs from the recorded one is refused, not replayed.
- **Consumed as recorded.** A probe on every module object logs each call. Every runtime call must
  receive exactly the recorded object. A call one module makes to another, which the trace cannot show,
  is logged as undeclared.
- **Disabled unexecuted computation.** With every module the program does not use made to raise
  (registry identities unchanged), the execution must be unchanged.
- **Interventions.** Replace the value step k consumes with another valid value v, then compare the
  answer with:
  - the same suffix run on its own from v (history independence: the answer depends on v alone, not on
    how the run got there, or on the question);
  - the reference's counterfactual answer from v (counterfactual semantic accuracy).

  An inaccurate module can depend on v alone and still be wrong; a module that reads the question can
  be right on every ordinary input and still fail history independence. The two are separate results.

Example: run `capital_of, country_of` on Ardel and intervene at step 1 with a city in another country.
The reference's answer becomes that city's country.
- A `country_of` module that computes from its input gives that answer, as does the suffix run on its own.
- One that reads the question ("which country was asked about?") still says Ardel. It is right on every
  ordinary input, but here it is wrong, and it disagrees with the suffix run on its own.

`before`: what the host does when a question arrives (for example, sharing it with modules that should
not see it). Every execution here goes through it.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

from .contracts import Budget, Program, TypedValue
from .records import to_record
from .reference import reference_answer
from .relations import LAWS, World
from .runtime import execute


class DisabledComputation(RuntimeError):
    """Raised by a module whose computation has been disabled."""


def _run(program, registry, types, budget, interventions=None, before: Callable | None = None):
    if before is not None:
        before(program)
    return execute(program, registry, types, budget, interventions)


def _modules(registry):
    seen, out = set(), []
    for rid, m in sorted(registry.modules.items()):
        if id(m) not in seen:
            seen.add(id(m))
            out.append((rid, m))
    return out


@contextmanager
def probe(registry) -> Iterator[list]:
    """Log every apply() call on every module object as (relation_id, argument, depth). Depth 0 is a call the
    runtime made; depth 1 or more is a call made from inside another module. The override is on the module
    objects themselves, so a module holding a hidden reference to another is logged too."""
    log: list = []
    depth = [0]
    originals = []
    for rid, m in _modules(registry):
        original = type(m).apply.__get__(m)

        def logged(value, rid=rid, original=original):
            log.append((rid, value, depth[0]))
            depth[0] += 1
            try:
                return original(value)
            finally:
                depth[0] -= 1
        object.__setattr__(m, "apply", logged)
        originals.append(m)
    try:
        yield log
    finally:
        for m in originals:
            object.__delattr__(m, "apply")


@contextmanager
def disabled(registry, relation_ids) -> Iterator[None]:
    """Make the modules bound to `relation_ids` raise DisabledComputation, wherever they are referenced.
    Module versions, and so the registry identity, are unchanged."""
    targets = [registry.module(r) for r in sorted(set(relation_ids))]

    def refuse(value):
        raise DisabledComputation("this module's computation is disabled")
    for m in targets:
        object.__setattr__(m, "apply", refuse)
    try:
        yield
    finally:
        for m in targets:
            if "apply" in vars(m):
                object.__delattr__(m, "apply")


def replay(record: dict, registry, types, before: Callable | None = None) -> dict:
    """Re-run a recorded execution with `registry`; refuse if it is not the recorded registry."""
    if registry.identity != record["registry_identity"]:
        return {"refused": "the registry differs from the recorded one", "matches": None}
    p = record["program"]
    program = Program(tuple(p["steps"]), TypedValue(**p["input"]), p["expected_output_type"])
    budget = Budget(record["budget"]["max_steps"], record["budget"]["policy"])
    interventions = {s["index"]: TypedValue(**s["consumed"]) for s in record["trace"] if s["intervened"]}
    ex = _run(program, registry, types, budget, interventions, before)
    return {"refused": None, "matches": to_record(ex, program, registry, budget, record["world_identity"]) == record}


def consumed_as_recorded(program: Program, registry, types, budget: Budget, before: Callable | None = None) -> dict:
    """Run once under a probe: did every runtime call receive the recorded object, and were there calls the
    trace cannot show?"""
    with probe(registry) as log:
        ex = _run(program, registry, types, budget, None, before)
    direct = [(rid, v) for rid, v, depth in log if depth == 0]
    exact = len(direct) == len(ex.trace) and all(
        rid == s.relation_id and v is s.consumed for (rid, v), s in zip(direct, ex.trace))
    return {"exact": exact, "undeclared_calls": sum(depth > 0 for _, _, depth in log)}


def unexecuted_disabled_unchanged(program: Program, registry, types, budget: Budget, world_identity: str,
                                  before: Callable | None = None) -> bool:
    """With every module the program does not use disabled, is the execution record unchanged?"""
    original = to_record(_run(program, registry, types, budget, None, before), program, registry, budget, world_identity)
    unused = set(registry.modules) - set(program.steps)
    with disabled(registry, unused):
        try:
            ex = _run(program, registry, types, budget, None, before)
        except DisabledComputation:
            return False
    return to_record(ex, program, registry, budget, world_identity) == original


def _answer(ex):
    return ex.answer.value if ex.answer is not None else None


def intervene(world: World, registry, program: Program, budget: Budget, step: int, replacement: TypedValue,
              before: Callable | None = None) -> dict:
    """Replace the value step `step` consumes with `replacement`; compare with the suffix alone and the
    reference. `eligible`: the reference's answer changes, so the intervention can show an effect."""
    ex = _run(program, registry, world.types, budget, {step: replacement}, before)
    suffix = Program(program.steps[step:], replacement)
    alone = _run(suffix, registry, world.types, Budget(budget.max_steps), None, before)
    counterfactual = reference_answer(world, program.steps[step:], replacement.value)
    original = reference_answer(world, program.steps, program.input.value)
    recorded = (ex.status != "rejected" and len(ex.trace) > step and ex.trace[step].intervened
                and ex.trace[step].consumed == replacement)
    return {"eligible": counterfactual != original, "recorded": recorded,
            "history_independent": _answer(ex) == _answer(alone),
            "counterfactual_correct": _answer(ex) == counterfactual}


def law_defects(registry, world: World, before: Callable | None = None) -> dict:
    """For each declared restricted-domain law, the inputs on which the registry's modules break it."""
    out = {}
    for law in LAWS:
        bad = []
        for x in world.types[law["on"]].values:
            program = Program(tuple(law["path"]), world.types[law["on"]].value(x))
            if _answer(_run(program, registry, world.types, Budget(len(law["path"])), None, before)) != x:
                bad.append(x)
        out[" then ".join(law["path"])] = bad
    return out
