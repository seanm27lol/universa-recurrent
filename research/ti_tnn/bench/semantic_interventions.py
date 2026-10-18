"""Execution records, replay and interventions on learned and deliberately faulty modules (integration plan
PR 5; protocols/semantic_interventions_v1.md).

For each world and module set, every well-typed path of length 1-3 is run from every start value. Each
execution is:
- recorded (`records.py`) and checked independently against the facts (`verification.py`);
- replayed from its record;
- run under a probe that logs every module call;
- run again with every module it does not use disabled.

Every path of length 2 or 3 is also intervened on: at each later step, `--replacements` values of the
right type are sampled (deterministically), and each replaces what that step consumes.

The module sets:
- the four learned families from plan PR 4 (`specialists`, `shared_dense`, `moe`, `endpoint`), trained
  on atomic facts;
- the table modules;
- three deliberately faulty sets from `integrated/faults.py` (`question_reading`, `hidden_dependency`,
  `swapped`).

    .venv-phase2/bin/python research/ti_tnn/bench/semantic_interventions.py --worlds 300 301 302 --inits 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated import faults as FX  # noqa: E402
from integrated import interventions as I  # noqa: E402
from integrated import modules as M  # noqa: E402
from integrated import reference as R  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import inventory  # noqa: E402
from integrated.contracts import Budget, Program  # noqa: E402
from integrated.records import to_record  # noqa: E402
from integrated.runtime import execute  # noqa: E402
from integrated.verification import check  # noqa: E402

LEARNED = ("specialists", "shared_dense", "moe", "endpoint")
CONSTRUCTED = ("table", "question_reading", "hidden_dependency", "swapped")
COMPETENT_FAMILIES = ("specialists", "shared_dense", "moe")
PURE = LEARNED + ("table", "swapped")  # modules that compute from their declared input alone
SAME_ENDPOINT = {"capital_of", "largest_city_of"}


def build(world: W.World, family: str, init: int | None, args) -> tuple:
    """(registry, before hook) for one module set."""
    if family in LEARNED:
        trained = M.train_family(world, family, init, args.d, args.hidden, args.steps, args.lr)
        registry = G.Registry(world.relations)
        for rid in sorted(world.relations):
            registry = registry.register(trained["modules"][rid])
        return registry, None
    if family == "table":
        return G.table_registry(world), None
    if family == "question_reading":
        return FX.question_reading_registry(world)
    if family == "hidden_dependency":
        return FX.hidden_dependency_registry(world), None
    if family == "swapped":
        return FX.swapped_registry(world), None
    raise ValueError(family)


def measure(world: W.World, registry, before, replacements: int) -> dict:
    """Every measurement for one module set on one world."""
    t = {k: 0 for k in ("executions", "schema_ok", "types_ok", "integrity_ok", "answer_correct", "replay_matches",
                        "consumed_exact", "undeclared_calls", "disabled_unchanged")}
    iv = {k: 0 for k in ("n", "recorded", "history_independent", "eligible", "counterfactual_correct_eligible",
                         "same_endpoint_eligible", "same_endpoint_correct")}
    atomic = {}
    for path in R.well_typed_paths(world, 3):
        source = world.relations[path[0]].source
        correct = 0
        for start in world.types[source].values:
            program, budget = Program(path, world.types[source].value(start)), Budget(len(path))
            if before is not None:
                before(program)
            ex = execute(program, registry, world.types, budget)
            record = to_record(ex, program, registry, budget, world.identity)
            c = check(record, world)
            t["executions"] += 1
            t["schema_ok"] += c["schema"]["ok"]
            t["types_ok"] += bool(c["types"] and c["types"]["ok"])
            t["integrity_ok"] += c["integrity"]["digest_matches"]
            ok = bool(c["semantics"] and c["semantics"]["answer_correct"])
            t["answer_correct"] += ok
            correct += ok
            t["replay_matches"] += I.replay(record, registry, world.types, before)["matches"] is True
            probe = I.consumed_as_recorded(program, registry, world.types, budget, before)
            t["consumed_exact"] += probe["exact"]
            t["undeclared_calls"] += probe["undeclared_calls"]
            t["disabled_unchanged"] += I.unexecuted_disabled_unchanged(program, registry, world.types, budget,
                                                                        world.identity, before)
            for k in range(1, len(path)):
                if len(ex.trace) <= k:
                    continue
                consumed = ex.trace[k].consumed.value
                domain = world.types[world.relations[path[k]].source]
                rng = random.Random(f"{world.identity}|{'/'.join(path)}|{start}|{k}")
                pool = [v for v in domain.values if v != consumed]
                for v in rng.sample(pool, min(replacements, len(pool))):
                    r = I.intervene(world, registry, program, budget, k, domain.value(v), before)
                    iv["n"] += 1
                    iv["recorded"] += r["recorded"]
                    iv["history_independent"] += r["history_independent"]
                    iv["eligible"] += r["eligible"]
                    iv["counterfactual_correct_eligible"] += r["eligible"] and r["counterfactual_correct"]
                    if r["eligible"] and SAME_ENDPOINT & set(path[k:]):  # the suffix applies capital_of or largest_city_of
                        iv["same_endpoint_eligible"] += 1
                        iv["same_endpoint_correct"] += r["counterfactual_correct"]
        if len(path) == 1:
            atomic[path[0]] = correct / len(world.types[source].values)
    differ = [c for c in world.types["Country"].values
              if world.facts["capital_of"][c] != world.facts["largest_city_of"][c]]
    return {**t, "interventions": iv, "atomic": atomic,
            "law_defects": {k: len(v) for k, v in I.law_defects(registry, world, before).items()},
            "countries_where_capital_differs": len(differ), "countries": len(world.types["Country"].values)}


def run(args) -> dict:
    runs = []
    for wseed in args.worlds:
        world = W.generate_world(wseed)
        jobs = [(f, i) for i in args.inits for f in args.families if f in LEARNED] + \
               [(f, None) for f in args.families if f in CONSTRUCTED]
        for family, init in jobs:
            t0 = time.time()
            registry, before = build(world, family, init, args)
            res = measure(world, registry, before, args.replacements)
            seconds = round(time.time() - t0, 2)
            runs.append({"family": family, "world": wseed, "world_identity": world.identity, "init": init,
                         "registry_identity": registry.identity, "seconds": seconds, **res})
            iv = res["interventions"]
            print(f"  world {wseed} {family:17s} init {init} answers {res['answer_correct']}/{res['executions']} "
                  f"replay {res['replay_matches']} probe {res['consumed_exact']} undeclared {res['undeclared_calls']} "
                  f"disabled-unchanged {res['disabled_unchanged']} | interventions {iv['n']} history-indep "
                  f"{iv['history_independent']} cf {iv['counterfactual_correct_eligible']}/{iv['eligible']} "
                  f"laws {res['law_defects']} ({seconds}s)", flush=True)
    return {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "platform": platform.platform()}, "runs": runs}


CONFIRMATION = {"worlds": (300, 301, 302), "inits": (0, 1, 2, 3, 4)}


def evaluate(result: dict, worlds=CONFIRMATION["worlds"], inits=CONFIRMATION["inits"]) -> dict:
    """The frozen predictions of protocols/semantic_interventions_v1.md, from the recorded runs."""
    want = [(f, w, i) for w in worlds for i in inits for f in LEARNED] + [(f, w, None) for w in worlds for f in CONSTRUCTED]
    bad = inventory.problems([(r["family"], r["world"], r["init"]) for r in result["runs"]], want)
    if bad:
        return {"complete": False, **bad}
    by = {(r["family"], r["world"], r["init"]): r for r in result["runs"]}
    runs = [by[k] for k in want]
    of = lambda fam: [r for r in runs if r["family"] == fam]
    competent = lambda r: min(r["atomic"].values()) >= 0.99
    cf = lambda r: r["interventions"]["counterfactual_correct_eligible"] / r["interventions"]["eligible"]
    n_iv = lambda r: r["interventions"]["n"]
    p = {}
    p["1_records_complete_and_replay"] = all(
        r[k] == r["executions"] for r in runs for k in ("schema_ok", "types_ok", "integrity_ok", "replay_matches")
    ) and all(r["interventions"]["recorded"] == n_iv(r) for r in runs)
    p["2_pure_modules_use_only_their_input"] = all(
        r["consumed_exact"] == r["executions"] and r["undeclared_calls"] == 0 and r["disabled_unchanged"] == r["executions"]
        and r["interventions"]["history_independent"] == n_iv(r) for f in PURE for r in of(f))
    p["3a_atomic_competence_replicates"] = all(
        sum(competent(by[(f, w, i)]) for i in inits) >= 4 for f in COMPETENT_FAMILIES for w in worlds)
    p["3b_counterfactual_accuracy_given_competence"] = all(
        cf(r) >= 0.95 for f in COMPETENT_FAMILIES + ("table",) for r in of(f) if competent(r))
    p["4_question_reading_caught_only_by_interventions"] = all(
        r["answer_correct"] == r["executions"] and r["replay_matches"] == r["executions"]
        and r["consumed_exact"] == r["executions"] and r["undeclared_calls"] == 0
        and r["disabled_unchanged"] == r["executions"] and r["interventions"]["history_independent"] < n_iv(r)
        for r in of("question_reading"))
    p["5_hidden_dependency_caught_by_probe_and_disabling"] = all(
        r["answer_correct"] == r["executions"] and r["undeclared_calls"] > 0 and r["disabled_unchanged"] < r["executions"]
        for r in of("hidden_dependency"))
    p["6_lawful_change_classified"] = all(
        all(v == 0 for v in r["law_defects"].values()) and r["replay_matches"] == r["executions"]
        and r["interventions"]["history_independent"] == n_iv(r)
        and abs(r["atomic"]["capital_of"] - (1 - r["countries_where_capital_differs"] / r["countries"])) < 1e-9
        for r in of("swapped"))
    se = lambda r: r["interventions"]["same_endpoint_correct"] / r["interventions"]["same_endpoint_eligible"]
    p["7_endpoint_faithful_but_wrong"] = all(
        r["interventions"]["history_independent"] == n_iv(r) and r["replay_matches"] == r["executions"] and se(r) < 0.95
        for r in of("endpoint"))
    summary = {f: {"runs": len(of(f)), "competent": sum(competent(r) for r in of(f)),
                   "answer_accuracy_min": min(r["answer_correct"] / r["executions"] for r in of(f)),
                   "history_independence_min": min(r["interventions"]["history_independent"] / n_iv(r) for r in of(f)),
                   "counterfactual_accuracy_min": min(cf(r) for r in of(f)),
                   "same_endpoint_counterfactual_accuracy_min": min(se(r) for r in of(f)),
                   "undeclared_calls_max": max(r["undeclared_calls"] for r in of(f)),
                   "disabled_unchanged_min": min(r["disabled_unchanged"] / r["executions"] for r in of(f)),
                   "law_defects_max": max(max(r["law_defects"].values()) for r in of(f))}
               for f in LEARNED + CONSTRUCTED}
    return {"complete": True, "families": summary, "predictions": p}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--worlds", nargs="+", type=int, default=list(CONFIRMATION["worlds"]))
    p.add_argument("--inits", nargs="+", type=int, default=list(CONFIRMATION["inits"]))
    p.add_argument("--families", nargs="+", default=list(LEARNED + CONSTRUCTED), choices=LEARNED + CONSTRUCTED)
    p.add_argument("--replacements", type=int, default=4, help="sampled replacement values per intervened step")
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/titnn_semantic"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    result = run(args)
    out = args.out / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    (out / "semantic_interventions.json").write_text(json.dumps(result, indent=2))
    if (tuple(sorted(args.worlds)) == CONFIRMATION["worlds"] and tuple(sorted(args.inits)) == CONFIRMATION["inits"]
            and set(args.families) == set(LEARNED + CONSTRUCTED)):
        verdict = evaluate(result)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        print(json.dumps(verdict["predictions"], indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
