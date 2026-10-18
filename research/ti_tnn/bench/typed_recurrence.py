"""Bounded recurrence, pause/resume and complete cost accounting on a route network (integration plan PR 7;
protocols/typed_recurrence_v1.md).

**World:** `route_world` (48 countries; about 170 airports, each with one onward connection).

**Queries:** every (airport, operator) pair asks "which is the first airport reached from here that this
operator runs?", with a budget of 16 iterations. Depending on the route, the answer is found, or the run
must abstain at a dead end, at a cycle, or when the budget runs out.

**Systems:**
- `recurrent` (proposed): iterate with early stopping (`integrated/recurrence.py`), using learned
  specialist modules trained on atomic facts (plan PR 4), `connects_to` included;
- `fixed_depth`: the same modules, always unrolled to the full budget, with the first match picked
  afterwards;
- `recurrent_table`: recurrence with table modules (a diagnostic).

**Measurements:**
- answers and stopping reasons against the independent reference;
- module applications, MACs and continuation-state size;
- wall time per query, split into module time and bookkeeping;
- whether the timed path reproduces the scored path;
- pause/resume at every possible pause point, through JSON, with a fresh copy of the registry.

    .venv-phase2/bin/python research/ti_tnn/bench/typed_recurrence.py --worlds 500 501 502 --inits 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated import costs as C  # noqa: E402
from integrated import modules as M  # noqa: E402
from integrated import recurrence as Q  # noqa: E402
from integrated import reference as R  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import inventory  # noqa: E402

SYSTEMS = ("recurrent", "fixed_depth", "recurrent_table")


def outcome(state) -> tuple[str | None, str]:
    """(answer, reason) in the reference's vocabulary."""
    if state["status"] == Q.ANSWERED:
        return state["answer"]["value"], "found"
    return None, state["reason"]


def learned_registry(world, init, args):
    trained = M.train_family(world, "specialists", init, args.d, args.hidden, args.steps, args.lr)
    reg = G.Registry(world.relations)
    for rid in sorted(world.relations):
        reg = reg.register(trained["modules"][rid])
    return reg


def fresh_copy(world, registry):
    """A new registry whose modules hold deep copies of the networks (the only state they have): same identity,
    no shared parameter objects. The copied parameters must hash to the recorded identity."""
    reg = G.Registry(world.relations)
    for rid, m in sorted(registry.modules.items()):
        net = copy.deepcopy(m.forward)
        assert M.module_identity("specialists", rid, net) == m.version.identity
        reg = reg.register(dataclasses.replace(m, forward=net))
    assert reg.identity == registry.identity
    return reg


def atomic(world, registry):
    out = {}
    for rid, spec in sorted(world.relations.items()):
        m = registry.module(rid)
        vals = world.types[spec.source].values
        out[rid] = sum((p := m.apply(world.types[spec.source].value(v))) is not None and p.value == world.facts[rid][v]
                       for v in vals) / len(vals)
    return out


def queries(world):
    return [Q.UntilQuery(world.types["Airport"].value(a), world.types["Operator"].value(o))
            for a in world.types["Airport"].values for o in world.types["Operator"].values]


def run_system(system, world, registry, qs, budget, repeats):
    fn = (lambda q: Q.fixed_depth(q, registry, world.types, budget)) if system == "fixed_depth" else \
         (lambda q: Q.run(Q.begin(q, registry, world.types, budget), registry, world.types))
    states, rows = [], []
    for q in qs:
        state = fn(q)  # the scored run
        states.append(state)
        times, same = [], True
        for _ in range(repeats):  # the timed runs, each compared with the scored one
            timed_state, t = C.timed(lambda: fn(q), registry)
            times.append(t)
            same = same and timed_state == state
        rows.append({"costs": C.application_costs(state, registry), "same_timed": same,
                     "total_ns": statistics.median(t["total_ns"] for t in times),
                     "module_ns": statistics.median(t["module_ns"] for t in times)})
    return states, rows


def pause_resume(world, registry, qs, budget):
    """Pause each recurrent run after every possible iteration count, round-trip the state through JSON, resume it
    with a fresh copy of the registry, and compare with the uninterrupted run."""
    fresh = fresh_copy(world, registry)
    checked = identical = 0
    for q in qs:
        full = Q.run(Q.begin(q, registry, world.types, budget), registry, world.types)
        calls = full["steps_used"] + (full["status"] == Q.ABSTAINED)  # step() calls in the uninterrupted run
        for p in range(1, calls):
            paused = Q.run(Q.begin(q, registry, world.types, budget), registry, world.types, pause_after=p)
            resumed = Q.resume(json.loads(json.dumps(paused)), fresh, world.types)
            checked += 1
            identical += resumed == full
    return {"pause_points": checked, "identical": identical}


def measure(world, init, args) -> list[dict]:
    t0 = time.time()
    learned = learned_registry(world, init, args)
    train_seconds = round(time.time() - t0, 2)
    competence = atomic(world, learned)
    qs = queries(world)
    refs = [R.reference_until(world, q.start.value, q.target.value, args.budget) for q in qs]
    out = []
    for system in SYSTEMS:
        reg = G.table_registry(world) if system == "recurrent_table" else learned
        states, rows = run_system(system, world, reg, qs, args.budget, args.repeats)
        got = [outcome(s) for s in states]
        by_reason = {}
        for (ans, reason), g in zip(refs, got):
            b = by_reason.setdefault(reason, {"n": 0, "correct": 0})
            b["n"] += 1
            b["correct"] += g == (ans, reason)
        res = {"system": system, "world": world.seed, "world_identity": world.identity, "init": init,
               "registry_identity": reg.identity, "train_seconds": train_seconds if system != "recurrent_table" else 0,
               "atomic": competence if system != "recurrent_table" else atomic(world, reg),
               "queries": len(qs), "correct": sum(g == r for g, r in zip(got, refs)), "by_reason": by_reason,
               "same_timed": sum(r["same_timed"] for r in rows),
               "module_applications_mean": statistics.fmean(r["costs"]["module_applications"] for r in rows),
               "macs_mean": statistics.fmean(r["costs"]["macs"] for r in rows),
               "state_bytes_mean": statistics.fmean(r["costs"]["state_bytes"] for r in rows),
               "wall_us_median": statistics.median(r["total_ns"] for r in rows) / 1e3,
               "module_us_median": statistics.median(r["module_ns"] for r in rows) / 1e3,
               "bookkeeping_share_median": statistics.median(1 - r["module_ns"] / r["total_ns"] for r in rows)}
        if system == "recurrent":
            res["pause_resume"] = pause_resume(world, reg, qs, args.budget)
            res["agrees_with_fixed_depth"] = None  # filled below
        out.append(res)
        if system == "recurrent":
            recurrent_outcomes = got
        if system == "fixed_depth":
            out[0]["agrees_with_fixed_depth"] = sum(a == b for a, b in zip(recurrent_outcomes, got))
        print(f"  world {world.seed} init {init} {system:15s} correct {res['correct']}/{res['queries']} "
              f"apps {res['module_applications_mean']:.2f} MACs {res['macs_mean']:.0f} wall {res['wall_us_median']:.0f}us "
              f"(bookkeeping {res['bookkeeping_share_median']:.2f}) timed=scored {res['same_timed']}"
              + (f" pause/resume {res['pause_resume']['identical']}/{res['pause_resume']['pause_points']}"
                 if "pause_resume" in res else ""), flush=True)
    return out


def run(args) -> dict:
    runs = []
    for wseed in args.worlds:
        world = W.route_world(wseed, n_countries=args.countries)
        for init in args.inits:
            runs += measure(world, init, args)
    return {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "platform": platform.platform(), "threads": torch.get_num_threads()}, "runs": runs}


CONFIRMATION = {"worlds": (500, 501, 502), "inits": (0, 1, 2, 3, 4)}


def evaluate(result: dict, worlds=CONFIRMATION["worlds"], inits=CONFIRMATION["inits"]) -> dict:
    """The frozen predictions of protocols/typed_recurrence_v1.md, from the recorded runs."""
    bad = inventory.problems([(r["system"], r["world"], r["init"]) for r in result["runs"]],
                             [(s, w, i) for s in SYSTEMS for w in worlds for i in inits])
    if bad:
        return {"complete": False, **bad}
    by = {(r["system"], r["world"], r["init"]): r for r in result["runs"]}
    rec = lambda w, i: by[("recurrent", w, i)]
    fix = lambda w, i: by[("fixed_depth", w, i)]
    competent = lambda w, i: min(rec(w, i)["atomic"].values()) >= 0.99
    units = [(w, i) for w in worlds for i in inits]
    p = {
        "1_atomic_competence": all(sum(competent(w, i) for i in inits) >= 4 for w in worlds),
        "2_recurrent_matches_reference": all(rec(w, i)["correct"] >= 0.99 * rec(w, i)["queries"]
                                             for w, i in units if competent(w, i)),
        "3_fixed_depth_agrees": all(rec(w, i)["agrees_with_fixed_depth"] == rec(w, i)["queries"] for w, i in units),
        "4_pause_resume_identical": all(rec(w, i)["pause_resume"]["identical"] == rec(w, i)["pause_resume"]["pause_points"] > 0
                                        for w, i in units),
        "5_timed_equals_scored": all(by[(s, w, i)]["same_timed"] == by[(s, w, i)]["queries"] for s in SYSTEMS for w, i in units),
        "6_fewer_applications": all(rec(w, i)["module_applications_mean"] <= 0.5 * fix(w, i)["module_applications_mean"]
                                    for w, i in units),
        "7_lower_latency": all(rec(w, i)["wall_us_median"] < fix(w, i)["wall_us_median"] for w, i in units),
    }
    summary = {s: {k: [min(by[(s, w, i)][k] for w, i in units), max(by[(s, w, i)][k] for w, i in units)]
                   for k in ("module_applications_mean", "macs_mean", "wall_us_median", "module_us_median",
                             "bookkeeping_share_median", "state_bytes_mean")}
               for s in SYSTEMS}
    for s in SYSTEMS:
        summary[s]["accuracy"] = [min(by[(s, w, i)]["correct"] / by[(s, w, i)]["queries"] for w, i in units),
                                  max(by[(s, w, i)]["correct"] / by[(s, w, i)]["queries"] for w, i in units)]
    return {"complete": True, "systems": summary, "predictions": p}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--worlds", nargs="+", type=int, default=list(CONFIRMATION["worlds"]))
    p.add_argument("--inits", nargs="+", type=int, default=list(CONFIRMATION["inits"]))
    p.add_argument("--countries", type=int, default=48)
    p.add_argument("--budget", type=int, default=16)
    p.add_argument("--repeats", type=int, default=3, help="timed repetitions per query (median taken)")
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--threads", type=int, default=1, help="1 keeps per-query timing free of thread scheduling")
    p.add_argument("--out", type=Path, default=Path("runs/titnn_recurrence"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    result = run(args)
    out = args.out / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    (out / "typed_recurrence.json").write_text(json.dumps(result, indent=2))
    if tuple(sorted(args.worlds)) == CONFIRMATION["worlds"] and tuple(sorted(args.inits)) == CONFIRMATION["inits"]:
        verdict = evaluate(result)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        print(json.dumps(verdict["predictions"], indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
