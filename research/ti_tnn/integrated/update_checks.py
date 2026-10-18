"""Independent gates for a candidate module update (integration plan section 7, "Independent gates"; PR 8).

Each gate returns pass, fail or insufficient evidence, with its counts. A failed mandatory gate cannot be
offset by better accuracy elsewhere. The gates judge the candidate against two worlds: the world before the
update and its updated copy. The checker is fixed code: it runs candidates, but no candidate can change it.

The plan distinguishes three kinds of locality, and so do the gates:
- **Write locality** (`identity_and_scope`):
  - only modules inside the declared scope changed;
  - every module's declared identity matches its actual parameters;
  - no changed network shares parameter storage with a module outside the scope.
- **Dependency locality** (`intended_behavior`): every answer the update should change, meaning every
  path whose reference answer differs between the two worlds, now matches the updated world.
- **Behavioral noninterference** (`unaffected_behavior`): every other well-typed path of length up to 3,
  from every start, still matches the reference. This is exhaustive in the finite world.

Also:
- `types_and_laws`: every module answers inside its declared type for every input, and the registered
  round-trip laws hold;
- `trace_and_continuation`: the candidate's records replay, its modules use only their input, and records
  made under the old registry are refused;
- `calibration_and_resources`: a calibration bound to the old registry needs refitting, and no changed
  module costs more than the registered limit.
"""
from __future__ import annotations

from .contracts import Budget, Program, content_identity, facts_mapping
from .interventions import consumed_as_recorded, intervene, law_defects, replay
from .records import to_record
from .registry import TableModule
from .reference import reference_answer, well_typed_paths
from .runtime import execute

CHECKER_VERSION = "ti_tnn_update_checks_v1"
PASS, FAIL, INSUFFICIENT = "pass", "fail", "insufficient evidence"


def _gate(outcome: str, **details) -> dict:
    return {"outcome": outcome, **details}


def _network(module):
    forward = getattr(module, "forward", None)
    return forward if forward is not None and hasattr(forward, "parameters") else None


def identity_and_scope(base, candidate, record) -> dict:
    """Write locality: which modules changed, whether that is allowed, and whether identities are honest."""
    from .modules import module_identity
    problems = []
    if set(base.modules) != set(candidate.modules):
        problems.append("the candidate does not implement the same relations")
    changed = sorted(r for r in candidate.modules if r in base.modules
                     and candidate.module(r).version.identity != base.module(r).version.identity)
    outside = [r for r in changed if r not in record.scope]
    if outside:
        problems.append(f"modules changed outside the declared scope: {outside}")
    unverifiable = []
    storage = {}
    for r, m in sorted(candidate.modules.items()):
        net = _network(m)
        if net is None:
            if isinstance(m, TableModule) and m.version.kind == "table":
                if content_identity({"kind": "table", "relation": r, "table": facts_mapping(m.table)}) != m.version.identity:
                    problems.append(f"{r}: declared identity does not match its table")
            elif r in changed:
                unverifiable.append(r)
            continue
        if module_identity(m.version.kind, r, net) != m.version.identity:
            problems.append(f"{r}: declared identity does not match its parameters")
        for p in net.parameters():
            if not bool(p.isfinite().all()):
                problems.append(f"{r}: non-finite parameters")
                break
        if r in base.modules and _network(base.module(r)) is not None:
            old = [p.shape for p in _network(base.module(r)).parameters()]
            if [p.shape for p in net.parameters()] != old:
                problems.append(f"{r}: parameter shapes changed")
        storage[r] = {p.data_ptr() for p in net.parameters()}
    for r in changed:
        for other, ptrs in storage.items():
            if other != r and other not in record.scope and r in storage and storage[r] & ptrs:
                problems.append(f"{r} shares parameter storage with {other}, outside the scope")
    outcome = FAIL if problems else INSUFFICIENT if unverifiable else PASS
    return _gate(outcome, changed=changed, problems=problems, unverifiable=unverifiable)


def types_and_laws(candidate, world) -> dict:
    """Every module answers inside its declared type for every input; registered laws hold."""
    bad = {}
    for r, spec in sorted(world.relations.items()):
        m = candidate.module(r)
        n = 0
        for v in world.types[spec.source].values:
            out = m.apply(world.types[spec.source].value(v))
            n += out is None or out.type_id != spec.target or not world.types[spec.target].contains(out.value)
        if n:
            bad[r] = n
    defects = {k: len(v) for k, v in law_defects(candidate, world).items()}
    outcome = FAIL if bad or any(defects.values()) else PASS
    return _gate(outcome, out_of_type=bad, law_defects=defects)


def _answer(registry, world, path, start):
    p = Program(path, world.types[world.relations[path[0]].source].value(start))
    ex = execute(p, registry, world.types, Budget(len(path)))
    return ex.answer.value if ex.answer is not None else None


def behavior(candidate, base, old_world, new_world, record, max_length: int = 3) -> tuple[dict, dict, dict]:
    """(intended_behavior, unaffected_behavior, counts): exhaustive over well-typed paths up to max_length."""
    declared = {tuple(x) for x in record.intended}
    actual = {(r, x, new_world.facts[r][x]) for r in new_world.relations for x in new_world.facts[r]
              if old_world.facts[r].get(x) != new_world.facts[r][x]}
    counts = {"intended": 0, "intended_correct": 0, "unaffected": 0, "unaffected_correct": 0,
              "unaffected_changed_from_base": 0, "atomic_changed_by_relation": {}}
    for path in well_typed_paths(new_world, max_length):
        for start in new_world.types[new_world.relations[path[0]].source].values:
            old, new = reference_answer(old_world, path, start), reference_answer(new_world, path, start)
            got = _answer(candidate, new_world, path, start)
            if old != new:
                counts["intended"] += 1
                counts["intended_correct"] += got == new
            else:
                counts["unaffected"] += 1
                counts["unaffected_correct"] += got == new
                moved = got != _answer(base, old_world, path, start)
                counts["unaffected_changed_from_base"] += moved
                if len(path) == 1 and moved:  # which relation's own answers moved
                    by = counts["atomic_changed_by_relation"]
                    by[path[0]] = by.get(path[0], 0) + 1
    intended_problems = [] if declared == actual else ["the declared changes are not the target world's changes"]
    if counts["intended_correct"] < counts["intended"]:
        intended_problems.append(f"{counts['intended'] - counts['intended_correct']} intended answers wrong")
    intended = _gate(FAIL if intended_problems else PASS, problems=intended_problems,
                     declared=len(declared), actual=len(actual))
    wrong = counts["unaffected"] - counts["unaffected_correct"]
    unaffected = _gate(FAIL if wrong else PASS, wrong=wrong, changed_from_base=counts["unaffected_changed_from_base"])
    return intended, unaffected, counts


def trace_and_continuation(candidate, base, world, sample: int = 60) -> dict:
    """The candidate's records replay and its modules use only their input; old records are refused."""
    problems = []
    paths = [p for p in well_typed_paths(world, 3)]
    picks = [(paths[i % len(paths)], i) for i in range(sample)]
    for path, i in picks:
        values = world.types[world.relations[path[0]].source].values
        start = world.types[world.relations[path[0]].source].value(values[i % len(values)])
        p, b = Program(path, start), Budget(len(path))
        rec = to_record(execute(p, candidate, world.types, b), p, candidate, b, world.identity)
        if replay(rec, candidate, world.types)["matches"] is not True:
            problems.append(f"replay differs on {path}")
        probe = consumed_as_recorded(p, candidate, world.types, b)
        if not probe["exact"] or probe["undeclared_calls"]:
            problems.append(f"undeclared computation on {path}")
        if len(path) > 1:
            other = [v for v in world.types[world.relations[path[1]].source].values][i % 2]
            r = intervene(world, candidate, p, b, 1, world.types[world.relations[path[1]].source].value(other))
            if not r["history_independent"]:
                problems.append(f"history-dependent answer on {path}")
        old = to_record(execute(p, base, world.types, b), p, base, b, world.identity)
        if candidate.identity != base.identity and replay(old, candidate, world.types)["refused"] is None:
            problems.append("a record made under the old registry was replayed on the candidate")
    return _gate(FAIL if problems else PASS, problems=problems[:10], checked=len(picks))


def calibration_and_resources(candidate, base, record, calibration=None, policy_version: str = "") -> dict:
    """A calibration bound to the old registry must be refit; no changed module may exceed the cost limit."""
    from .costs import module_macs
    problems, insufficient = [], []
    if calibration is not None:
        issues = calibration.binding_problems(calibration.interpreter_identity, candidate.identity, policy_version)
        if issues:
            insufficient.append("the calibration was fit for the old registry; refit before activation")
    for r in sorted(candidate.modules):
        if r in base.modules and candidate.module(r).version.identity == base.module(r).version.identity:
            continue
        macs, limit = module_macs(candidate.module(r)), record.max_macs_per_application
        if macs is None:
            insufficient.append(f"{r}: cost cannot be read off")
        elif limit is not None and macs > limit:
            problems.append(f"{r}: {macs} MACs per application, over the limit of {limit}")
    outcome = FAIL if problems else INSUFFICIENT if insufficient else PASS
    return _gate(outcome, problems=problems, insufficient=insufficient)


def check_all(base, candidate, record, old_world, new_world, calibration=None) -> dict:
    """Every mandatory gate, in order."""
    intended, unaffected, counts = behavior(candidate, base, old_world, new_world, record)
    return {"identity_and_scope": identity_and_scope(base, candidate, record),
            "types_and_laws": types_and_laws(candidate, new_world),
            "intended_behavior": intended, "unaffected_behavior": unaffected,
            "trace_and_continuation": trace_and_continuation(candidate, base, new_world),
            "calibration_and_resources": calibration_and_resources(candidate, base, record, calibration),
            "counts": counts, "checker": CHECKER_VERSION}
