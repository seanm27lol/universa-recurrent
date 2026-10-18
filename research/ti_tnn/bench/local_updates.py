"""Offline module replacement under independent gates (integration plan PR 8; protocols/local_updates_v1.md).

**Learned updates.** For each world, ten single-fact updates are drawn: five capitals moving to another
city of the same country, and five airports changing operator. For each one, at each init, two systems
are updated:
- **`specialists` (proposed):** only the affected relation's network is fine-tuned on the updated facts
  (`modules.finetune`); every other module is reused as it is.
- **`shared_same_data`:** the one shared network is fine-tuned on the same facts the specialist sees
  (the updated relation's), so every module changes;
- **`shared_rehearsal`:** the shared network is fine-tuned on every relation's updated facts (more data
  than the specialist).

Each candidate goes through every gate in `integrated/update_checks.py`. Activation is then attempted
(`integrated/updates.py`), and if it succeeds, rolled back.

**The faulty suite.** Once per world, table modules (plus two learned cases) give one candidate per
failure the plan lists, each with the gate that must catch it:
- a valid edit, which must pass, activate and roll back exactly;
- a lawful answer flip, a broken inverse and an ill-typed module;
- an out-of-scope state edit, a stale identity and shared parameter storage;
- a stale revision, a failed rollback, and an unauthorized but valid candidate.

    .venv-phase2/bin/python research/ti_tnn/bench/local_updates.py --worlds 600 601 602 --inits 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import platform
import random
import sys
import time
from types import MappingProxyType
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated import modules as M  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import inventory  # noqa: E402
from integrated import update_checks as U  # noqa: E402
from integrated import snapshots as S  # noqa: E402
from integrated import updates as D  # noqa: E402
from integrated.contracts import ModuleVersion, content_identity, facts_mapping  # noqa: E402
from integrated.reference import well_typed_paths  # noqa: E402

PROPOSER = "offline-update"
CHECKER = U.Checker()  # one checker per process; deployments accept only results it sealed
GATES_V1 = ("identity_and_scope", "types_and_laws", "intended_behavior", "unaffected_behavior",
            "trace_and_continuation", "calibration_and_resources")  # the gates the frozen summary counts
PER_KIND = 5  # the frozen inventory: five updates of each kind per world
FAULT_CASES = ("valid_table_edit", "lawful_answer_flip", "broken_inverse", "ill_typed", "out_of_scope_state_edit",
               "stale_revision", "failed_rollback", "unauthorized", "stale_identity", "shared_storage")
FAMILIES = {"specialists": ("specialists", True), "shared_same_data": ("shared_dense", False),
            "shared_rehearsal": ("shared_dense", True)}  # name -> (trained family, rehearse)
KINDS = ("capital_moves", "operator_changes")


def edited(world, facts, label):
    return W.World(f"{world.name}+{label}", world.types, world.relations, facts, world.seed)


def updates(world, n_each: int, seed: int) -> list[tuple[W.World, tuple, str]]:
    """Single-fact updates: (updated world, intended changes, relation)."""
    rng = random.Random(f"updates|{world.identity}|{seed}")
    cities = {}
    for city, country in world.facts["country_of"].items():
        cities.setdefault(country, []).append(city)
    out = []
    for kind in KINDS:
        rel = "capital_of" if kind == "capital_moves" else "operator_of"
        sources = sorted(c for c in cities if len(cities[c]) > 1) if rel == "capital_of" else list(world.types["Airport"].values)
        for x in rng.sample(sources, n_each):
            old = world.facts[rel][x]
            pool = [c for c in sorted(cities[x]) if c != old] if rel == "capital_of" else \
                   [o for o in world.types["Operator"].values if o != old]
            new = rng.choice(pool)
            facts = {r: dict(t) for r, t in world.facts.items()}
            facts[rel][x] = new
            out.append((edited(world, facts, f"{rel}:{x}->{new}"), ((rel, x, new),), rel))
    return out


def registry_from(world, modules: dict):
    reg = G.Registry(world.relations)
    for rid in sorted(world.relations):
        reg = reg.register(modules[rid])
    return reg


def record_for(base, candidate, new_world, intended, scope, proposer=PROPOSER, base_identity=None, **fields):
    """An update record naming the published digests of `base` and `candidate`."""
    return D.UpdateRecord(base_identity or S.digest_of(base), S.digest_of(candidate), tuple(scope), tuple(intended),
                          new_world.identity, fields.pop("checker_version", CHECKER.version),
                          fields.pop("calibration_version", None), fields.pop("max_macs_per_application", None),
                          proposer)


def all_answers(registry, world):
    from integrated.contracts import Budget, Program
    from integrated.runtime import execute
    out = []
    for path in well_typed_paths(world, 3):
        for s in world.types[world.relations[path[0]].source].values:
            ex = execute(Program(path, world.types[world.relations[path[0]].source].value(s)), registry, world.types,
                         Budget(len(path)))
            out.append(ex.answer.value if ex.answer else None)
    return out


def attempt(base, record, result, authority, old_world, base_answers):
    """Activate on a fresh deployment of `base`. If activated, roll back, and check that the restored snapshot
    gives the base answers captured before any candidate work."""
    dep = D.Deployment(base, authority, CHECKER)
    published = dep.active.digest
    act = dep.activate(record, result)
    out = {"activated": act["activated"], "reasons": act["reasons"]}
    if act["activated"]:
        rb = dep.rollback()
        out["rolled_back"] = rb["rolled_back"]
        out["rollback_exact"] = (dep.active.digest == published and dep.active.verify()
                                 and all_answers(dep.registry, old_world) == base_answers)
    return out


def outcomes(result):
    return result.outcomes()


def table_candidate(base, world, new_world, relation, table=None, target=None, kind="table"):
    """`base` with `relation` replaced by a table module (by default the updated world's facts)."""
    table = dict(new_world.facts[relation]) if table is None else table
    version = ModuleVersion(f"{relation}_{kind}_candidate", relation, kind,
                            content_identity({"kind": "table", "relation": relation, "table": facts_mapping(table)}))
    module = G.TableModule(version, table, target or new_world.types[new_world.relations[relation].target])
    return registry_from(new_world, {**dict(base.modules), relation: module})


def faulty_suite(world, init, args) -> list[dict]:
    rng_updates = updates(world, 1, 1000 + init)
    authority = D.Authority({PROPOSER})
    base = G.table_registry(world)
    base_answers = all_answers(base, world)  # captured before any candidate work
    new_world, intended, rel = rng_updates[0]  # a capital move
    country = intended[0][1]
    cases = []

    def judge(name, candidate, record, nw=new_world, expect=None, **extra):
        result = CHECKER.check(base, candidate, record, world, nw)
        res = {"case": name, "gates": outcomes(result), **attempt(base, record, result, authority, world, base_answers),
               **extra}
        res["expected"] = expect
        cases.append(res)
        return result

    valid = table_candidate(base, world, new_world, "capital_of")
    a_rec = record_for(base, valid, new_world, intended, ("capital_of",))
    a_res = judge("valid_table_edit", valid, a_rec, expect={"all_pass": True, "activated": True})
    flip = table_candidate(base, world, new_world, "capital_of", table=dict(world.facts["largest_city_of"]))
    judge("lawful_answer_flip", flip, record_for(base, flip, new_world, intended, ("capital_of",)),
          expect={"types_and_laws": U.PASS, "unaffected_behavior": U.FAIL, "activated": False})
    # broken inverse: the capital city of one country is reassigned to another country
    capital = world.facts["capital_of"][country]
    other = next(c for c in world.types["Country"].values if c != country)
    facts = {r: dict(t) for r, t in world.facts.items()}
    facts["country_of"][capital] = other
    broken_world = edited(world, facts, f"country_of:{capital}->{other}")
    broken = table_candidate(base, world, broken_world, "country_of")
    judge("broken_inverse", broken, record_for(base, broken, broken_world, (("country_of", capital, other),), ("country_of",)),
          nw=broken_world, expect={"types_and_laws": U.FAIL, "activated": False})
    airport_table = {c: world.facts["airport_of"][city] for c, city in new_world.facts["capital_of"].items()}
    ill = table_candidate(base, world, new_world, "capital_of", table=airport_table, target=world.types["Airport"])
    judge("ill_typed", ill, record_for(base, ill, new_world, intended, ("capital_of",)),
          expect={"types_and_laws": U.FAIL, "activated": False})
    ghost = table_candidate(base, world, new_world, "capital_of")
    ghost = table_candidate(ghost, world, new_world, "airport_of", table={**world.facts["airport_of"], "ghost_city": "ghost_airport"})
    judge("out_of_scope_state_edit", ghost, record_for(base, ghost, new_world, intended, ("capital_of",)),
          expect={"identity_and_scope": U.FAIL, "intended_behavior": U.PASS, "unaffected_behavior": U.PASS, "activated": False})
    # stale revision: two valid candidates on the same base; the second is proposed after the first is active
    dep = D.Deployment(base, authority, CHECKER)
    second_world, second_intended, _ = rng_updates[1]  # an operator change
    dep.activate(a_rec, a_res)
    b = table_candidate(base, world, second_world, "operator_of")
    b_rec = record_for(base, b, second_world, second_intended, ("operator_of",))
    b_res = CHECKER.check(base, b, b_rec, world, second_world)
    decision = dep.activate(b_rec, b_res)
    cases.append({"case": "stale_revision", "gates": outcomes(b_res), "activated": decision["activated"],
                  "reasons": decision["reasons"], "expected": {"all_pass": True, "activated": False}})
    # failed rollback: the stored snapshot is tampered with; rollback must refuse and leave the active snapshot
    dep = D.Deployment(base, authority, CHECKER)
    dep.activate(a_rec, a_res)
    active = dep.active
    stored = dep.history[-1].registry.module("operator_of")
    object.__setattr__(stored, "table", MappingProxyType(dict(second_world.facts["operator_of"])))
    rb = dep.rollback()
    cases.append({"case": "failed_rollback", "rolled_back": rb["rolled_back"], "active_unchanged": dep.active is active,
                  "reasons": rb["reasons"], "expected": {"rolled_back": False, "active_unchanged": True}})
    unauth = record_for(base, valid, new_world, intended, ("capital_of",), proposer="confident-agent")
    judge("unauthorized", valid, unauth, expect={"all_pass": True, "activated": False}, claimed_confidence=0.999)
    # learned cases: a stale declared identity, and parameters shared with a module outside the scope
    trained = M.train_family(world, "specialists", init, args.d, args.hidden, args.steps, args.lr)
    lbase = registry_from(world, trained["modules"])
    tuned = M.finetune(new_world, "specialists", trained, init, ("capital_of",), args.update_steps, args.update_lr)
    cap = tuned["modules"]["capital_of"]
    with torch.no_grad():
        drifted = copy.deepcopy(cap.forward)
        next(drifted.parameters()).add_(0.01)
    stale = registry_from(new_world, {**tuned["modules"], "capital_of": dataclasses.replace(cap, forward=drifted)})
    result = CHECKER.check(lbase, stale, record_for(lbase, stale, new_world, intended, ("capital_of",)), world, new_world)
    cases.append({"case": "stale_identity", "gates": outcomes(result), "expected": {"identity_and_scope": U.FAIL}})
    largest = trained["modules"]["largest_city_of"]
    aliased = registry_from(new_world, {**tuned["modules"], "largest_city_of": dataclasses.replace(largest, forward=cap.forward)})
    result = CHECKER.check(lbase, aliased, record_for(lbase, aliased, new_world, intended, ("capital_of",)), world, new_world)
    cases.append({"case": "shared_storage", "gates": outcomes(result), "expected": {"identity_and_scope": U.FAIL}})
    for c in cases:
        c["as_expected"] = expected_met(c)
    return cases


def expected_met(case) -> bool:
    exp = case["expected"]
    ok = True
    for k, v in exp.items():
        if k == "all_pass":
            ok &= all(o == U.PASS for o in case["gates"].values()) == v
        elif k in D.MANDATORY:
            ok &= case["gates"][k] == v
        else:
            ok &= case.get(k) == v
    if case.get("activated"):
        ok &= case.get("rollback_exact", False)
    return bool(ok)


def learned_updates(world, init, args) -> list[dict]:
    authority = D.Authority({PROPOSER})
    out = []
    bases = {f: M.train_family(world, f, init, args.d, args.hidden, args.steps, args.lr) for f in ("specialists", "shared_dense")}
    regs = {f: registry_from(world, b["modules"]) for f, b in bases.items()}
    truth = all_answers(G.table_registry(world), world)
    answers = {f: all_answers(regs[f], world) for f in regs}  # captured before any candidate work
    competent = {f: answers[f] == truth for f in regs}
    for k, (new_world, intended, rel) in enumerate(updates(world, args.per_kind, 0)):
        for family, (trained, rehearse) in FAMILIES.items():
            t0 = time.time()
            tuned = M.finetune(new_world, trained, bases[trained], init, (rel,), args.update_steps, args.update_lr, rehearse)
            update_seconds = time.time() - t0
            cand = registry_from(new_world, tuned["modules"])
            record = record_for(regs[trained], cand, new_world, intended, (rel,))
            result = CHECKER.check(regs[trained], cand, record, world, new_world)
            res = {"family": family, "world": world.seed, "init": init, "update": k, "relation": rel,
                   "intended": [list(x) for x in intended], "base_competent": competent[trained],
                   "update_seconds": round(update_seconds, 3), "gates": outcomes(result), "counts": dict(result.counts),
                   "changed_modules": result.gates["identity_and_scope"]["changed"],
                   **attempt(regs[trained], record, result, authority, world, answers[trained])}
            out.append(res)
    return out


def run(args) -> dict:
    runs, suites = [], []
    for wseed in args.worlds:
        world = W.generate_world(wseed)
        for init in args.inits:
            t0 = time.time()
            rows = learned_updates(world, init, args)
            runs += rows
            suite = faulty_suite(world, init, args) if init == args.inits[0] else []
            for c in suite:
                c.update(world=wseed, init=init)
            suites += suite
            for fam in FAMILIES:
                rs = [r for r in rows if r["family"] == fam]
                print(f"  world {wseed} init {init} {fam:12s} activated {sum(r['activated'] for r in rs)}/{len(rs)} "
                      f"scope-pass {sum(r['gates']['identity_and_scope'] == U.PASS for r in rs)} "
                      f"unaffected-pass {sum(r['gates']['unaffected_behavior'] == U.PASS for r in rs)} "
                      f"intended-pass {sum(r['gates']['intended_behavior'] == U.PASS for r in rs)} "
                      f"unaffected changed from base {sum(r['counts']['unaffected_changed_from_base'] for r in rs)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if suite:
                print(f"  world {wseed} faulty suite: " + ", ".join(f"{c['case']} {'ok' if c['as_expected'] else 'UNEXPECTED'}"
                                                                    for c in suite), flush=True)
    return {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "platform": platform.platform()}, "runs": runs, "faulty_suite": suites}


CONFIRMATION = {"worlds": (600, 601, 602), "inits": (0, 1, 2, 3, 4)}


def evaluate(result: dict, worlds=CONFIRMATION["worlds"], inits=CONFIRMATION["inits"]) -> dict:
    """The frozen predictions of protocols/local_updates_v1.md, from the recorded runs."""
    rows = result["runs"]
    by_world = lambda fam, w: [r for r in rows if r["family"] == fam and r["world"] == w and r["init"] in inits]
    expected = [(f, w, i, k, "capital_of" if k < PER_KIND else "operator_of")
                for f in FAMILIES for w in worlds for i in inits for k in range(2 * PER_KIND)]
    bad = inventory.problems([(r["family"], r["world"], r["init"], r["update"], r["relation"]) for r in rows], expected)
    bad_suite = inventory.problems([(c["case"], c["world"]) for c in result["faulty_suite"]],
                                   [(c, w) for c in FAULT_CASES for w in worlds])
    if bad or bad_suite:
        return {"complete": False, "updates": bad, "faulty_suite": bad_suite}
    spec_ok = lambda r: r["activated"] and r.get("rollback_exact", False)

    def share(w, fam, rel, outcome):
        rs = [r for r in by_world(fam, w) if r["relation"] == rel]
        return sum(r["gates"]["unaffected_behavior"] == outcome for r in rs) / len(rs)
    p = {
        "1_specialist_updates_pass_and_activate": all(
            sum(spec_ok(r) for r in by_world("specialists", w)) >= 0.95 * len(by_world("specialists", w)) for w in worlds),
        "2_every_activation_rolls_back_exactly": all(
            r.get("rolled_back") and r.get("rollback_exact") for r in rows if r["activated"]),
        "3_shared_updates_fail_write_locality": all(
            r["gates"]["identity_and_scope"] != U.PASS and not r["activated"]
            for f in ("shared_same_data", "shared_rehearsal") for w in worlds for r in by_world(f, w)),
        "4_faulty_candidates_caught_as_designed": all(c["as_expected"] for c in result["faulty_suite"] if c["world"] in worlds),
        "5_same_data_shared_interferes_through_capital_of": all(
            share(w, "shared_same_data", "capital_of", U.FAIL) >= 0.8 and share(w, "shared_same_data", "operator_of", U.PASS) >= 0.8
            for w in worlds),
        "6_rehearsal_keeps_behavior": all(
            sum(r["gates"]["unaffected_behavior"] == U.PASS for r in by_world("shared_rehearsal", w))
            >= 0.9 * len(by_world("shared_rehearsal", w)) for w in worlds),
    }
    summary = {}
    for fam in FAMILIES:
        rs = [r for w in worlds for r in by_world(fam, w)]
        summary[fam] = {"cases": len(rs), "activated": sum(r["activated"] for r in rs),
                        "gate_pass": {g: sum(r["gates"][g] == U.PASS for r in rs) for g in GATES_V1},
                        "unaffected_changed_from_base_total": sum(r["counts"]["unaffected_changed_from_base"] for r in rs),
                        "cases_with_unaffected_changes": sum(r["counts"]["unaffected_changed_from_base"] > 0 for r in rs),
                        "intended_wrong_total": sum(r["counts"]["intended"] - r["counts"]["intended_correct"] for r in rs),
                        "atomic_changed_by_relation": {rel: sum(r["counts"]["atomic_changed_by_relation"].get(rel, 0) for r in rs)
                                                       for rel in sorted({k for r in rs for k in r["counts"]["atomic_changed_by_relation"]})},
                        "base_competent": sum(r["base_competent"] for r in rs if r["update"] == 0)}
    return {"complete": True, "families": summary, "predictions": p,
            "faulty_suite": {c["case"]: sum(x["as_expected"] for x in result["faulty_suite"] if x["case"] == c["case"])
                             for c in result["faulty_suite"]}}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--worlds", nargs="+", type=int, default=list(CONFIRMATION["worlds"]))
    p.add_argument("--inits", nargs="+", type=int, default=list(CONFIRMATION["inits"]))
    p.add_argument("--per-kind", type=int, default=5, help="updates of each kind per world")
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--update-steps", type=int, default=300)
    p.add_argument("--update-lr", type=float, default=3e-3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/titnn_updates"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    result = run(args)
    out = args.out / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    (out / "local_updates.json").write_text(json.dumps(result, indent=2))
    if tuple(sorted(args.worlds)) == CONFIRMATION["worlds"] and tuple(sorted(args.inits)) == CONFIRMATION["inits"]:
        verdict = evaluate(result)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        print(json.dumps(verdict["predictions"], indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
