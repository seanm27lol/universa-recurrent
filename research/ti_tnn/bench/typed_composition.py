"""Learned relation operators on the finite typed world (integration plan PR 4; protocols/typed_composition_v1.md).

Each family is trained on one world's atomic facts only (one relation applied to one entity), then scored:
- atomic accuracy per relation, on every fact;
- same-endpoint discrimination: `capital_of` and `largest_city_of` on the countries where they differ;
- chains run through the typed runtime with the correct program supplied: non-cancelling 2- and 3-step
  paths are gated, round trips (`capital_of` then `country_of`) are reported only;
- intermediate-value accuracy: every value a chain passes, against the independent reference.

`answer_only` has no typed interface; it is scored on the same paths by direct prediction.

    .venv-phase2/bin/python research/ti_tnn/bench/typed_composition.py --worlds 200 201 202 --inits 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated import contracts as C  # noqa: E402
from integrated import modules as M  # noqa: E402
from integrated import reference as R  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import runtime as X  # noqa: E402

CANCELLING = {("capital_of", "country_of"), ("largest_city_of", "country_of")}


def path_family(path: tuple[str, ...]) -> str:
    if any(pair in CANCELLING for pair in zip(path, path[1:])):
        return "round_trip"
    return f"length{len(path)}"


def score(world: W.World, family: str, trained: dict) -> dict:
    """Every measurement for one trained family on one world."""
    differ = [c for c in world.types["Country"].values if world.facts["capital_of"][c] != world.facts["largest_city_of"][c]]
    paths = [p for p in R.well_typed_paths(world, 3) if len(p) >= 2]
    out = {"atomic": {}, "same_endpoint": {}, "paths": {}, "intermediate": {"correct": 0, "total": 0}}
    if family == "answer_only":
        net, vocab = trained["answer_only"]
        predict = lambda path, start: M.answer_only_predict(net, vocab, world, path, start)
    else:
        registry = G.Registry(world.relations)
        for rid in sorted(world.relations):
            registry = registry.register(trained["modules"][rid])

        def predict(path, start, record=None):
            program = C.Program(path, world.types[world.relations[path[0]].source].value(start))
            ex = X.execute(program, registry, world.types, C.Budget(len(path)))
            if record is not None:
                for i, step in enumerate(ex.trace):
                    record["total"] += 1
                    record["correct"] += step.produced is not None and \
                        step.produced.value == R.reference_answer(world, path[:i + 1], start)
            return ex.answer.value if ex.answer else None
    for rid in sorted(world.relations):
        src = world.relations[rid].source
        values = world.types[src].values
        out["atomic"][rid] = sum(predict((rid,), v) == world.facts[rid][v] for v in values) / len(values)
    for rid in ("capital_of", "largest_city_of"):
        out["same_endpoint"][rid] = sum(predict((rid,), c) == world.facts[rid][c] for c in differ) / len(differ)
    out["same_endpoint"]["countries"] = len(differ)
    for path in paths:
        fam = path_family(path)
        tally = out["paths"].setdefault(fam, {"correct": 0, "total": 0})
        for start in world.types[world.relations[path[0]].source].values:
            got = predict(path, start) if family == "answer_only" else predict(path, start, out["intermediate"])
            tally["correct"] += got == R.reference_answer(world, path, start)
            tally["total"] += 1
    if family == "answer_only":
        out["intermediate"] = None  # no typed intermediate values exist
    return out


def run(args) -> dict:
    runs = []
    for wseed in args.worlds:
        world = W.generate_world(wseed)
        for init in args.inits:
            for family in args.families:
                t0 = time.time()
                trained = M.train_family(world, family, init, args.d, args.hidden, args.steps, args.lr)
                seconds = time.time() - t0
                res = score(world, family, trained)
                runs.append({"family": family, "world": wseed, "world_identity": world.identity, "init": init,
                             "train_loss": trained["loss"], "train_seconds": round(seconds, 2), **res})
                print(f"  world {wseed} init {init} {family:13s} atomic min {min(res['atomic'].values()):.3f} "
                      f"same-endpoint {min(v for k, v in res['same_endpoint'].items() if k != 'countries'):.3f} "
                      + " ".join(f"{k} {v['correct'] / v['total']:.3f}" for k, v in sorted(res['paths'].items()))
                      + f" ({seconds:.1f}s)", flush=True)
    return {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "platform": platform.platform()}, "runs": runs}


GATED = ("specialists", "shared_dense", "moe")  # the families the competence predictions cover


def evaluate(result: dict, worlds=(200, 201, 202), inits=(0, 1, 2, 3, 4)) -> dict:
    """The frozen predictions of protocols/typed_composition_v1.md, from the recorded runs."""
    runs = result["runs"]
    by = {(r["family"], r["world"], r["init"]): r for r in runs}
    missing = [(f, w, i) for f in M.FAMILIES for w in worlds for i in inits if (f, w, i) not in by]
    if missing:
        return {"complete": False, "missing": missing[:10]}
    acc = lambda t: t["correct"] / t["total"]
    atomic_ok = lambda r: min(r["atomic"].values()) >= 0.99
    same_ok = lambda r: min(r["same_endpoint"]["capital_of"], r["same_endpoint"]["largest_city_of"]) >= 0.99
    reliable = lambda fam, ok: all(sum(ok(by[(fam, w, i)]) for i in inits) >= 4 for w in worlds)
    out = {"complete": True, "families": {}}
    for fam in M.FAMILIES:
        rs = [by[(fam, w, i)] for w in worlds for i in inits]
        out["families"][fam] = {
            "atomic_min": min(min(r["atomic"].values()) for r in rs),
            "same_endpoint_min": min(min(r["same_endpoint"]["capital_of"], r["same_endpoint"]["largest_city_of"]) for r in rs),
            "paths_min": {k: min(acc(r["paths"][k]) for r in rs) for k in rs[0]["paths"]},
            "intermediate_min": None if fam == "answer_only" else min(acc(r["intermediate"]) for r in rs),
            "competent_runs": sum(atomic_ok(r) for r in rs)}
    competent = lambda r: atomic_ok(r)
    out["predictions"] = {
        "1_atomic_competence": all(reliable(f, atomic_ok) for f in GATED),
        "2_same_endpoint_discrimination": all(reliable(f, same_ok) for f in ("shared_dense", "moe")),
        "3_negative_control_collapses": all(min(by[("endpoint", w, i)]["same_endpoint"]["capital_of"],
                                                by[("endpoint", w, i)]["same_endpoint"]["largest_city_of"]) <= 0.6
                                            for w in worlds for i in inits),
        "4_composition_given_competence": all(acc(by[(f, w, i)]["paths"][k]) >= 0.95
                                              for f in GATED for w in worlds for i in inits if competent(by[(f, w, i)])
                                              for k in ("length2", "length3")),
        "5_intermediate_given_competence": all(acc(by[(f, w, i)]["intermediate"]) >= 0.95
                                               for f in GATED for w in worlds for i in inits if competent(by[(f, w, i)])),
        "6_answer_only_fails_chains": all(acc(by[("answer_only", w, i)]["paths"][k]) < 0.5
                                          for w in worlds for i in inits for k in ("length2", "length3")),
    }
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--worlds", nargs="+", type=int, default=[200, 201, 202])
    p.add_argument("--inits", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--families", nargs="+", default=list(M.FAMILIES), choices=M.FAMILIES)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/titnn_typed"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    result = run(args)
    out = args.out / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    (out / "typed_composition.json").write_text(json.dumps(result, indent=2))
    if sorted(args.worlds) == [200, 201, 202] and sorted(args.inits) == [0, 1, 2, 3, 4] and set(args.families) == set(M.FAMILIES):
        verdict = evaluate(result)
        (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
        print(json.dumps(verdict["predictions"], indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
