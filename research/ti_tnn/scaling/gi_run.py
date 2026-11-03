"""Commands of the grounding-interface study (development; `docs/PROTOCOL_grounding_interface.md`).

    python -m scaling.gi_run constraint   # post hoc: the six v1 checkpoints, v1 and v1 + the ON restriction (CPU)

Runs go to `<runs root>/grounding_interface/`. The historical producers and their evidence are read, never written.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import contracts as C

ROOT = C.RUNS / "grounding_interface"
V1_ARMS = (("T4", "v1"), ("c4_params", "v1"))


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")


def _saved_records(label: str, seed: int) -> dict:
    """The counted performance-study evaluation's records of one arm and seed, by task id."""
    from . import perf
    chosen, _all = perf.select_evaluations()
    path = chosen[seed] / label / "records.jsonl"
    return {r["task_id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def _encodings(task_ids) -> dict:
    """The anchored cached states of the evaluated questions, verified against their anchors."""
    from . import diagnose as D
    from . import encode, perf, pilot_run
    anchors, ident = perf.anchors()
    reader = encode.Backbone.cache_reader(ident, pilot_run._tokenizer())
    return {tid: D._encoding(tid, reader, encode.PILOT_CACHE, anchors) for tid in task_ids}


def _score(cache: dict, task, sql: "str | None") -> dict:
    """The pinned scorer, once per distinct (database, SQL)."""
    from . import data, scorer
    key = (task.db_id, sql)
    if key not in cache:
        cache[key] = scorer.score(data.snapshot(task.db_id), sql, data.gold(task.task_id).sql)
    return cache[key]


def _same_reference(plan) -> int:
    """ON comparisons of one column reference with itself."""
    from . import diagnose as D
    if plan is None:
        return 0
    return len([p for p in D.same_reference_predicates(plan) if p["clause"].startswith("ON")])


def constraint_cmd(args) -> None:
    """The six v1 checkpoints of the performance study on its 128 evaluated questions, decoded on the CPU with the
    original v1 policy and with only the same-reference ON restriction added (`gi_constraint`)."""
    import torch

    from . import controller as K
    from . import data, identity, infer_controller, perf, pilot_run, train
    from . import gi_constraint as GC
    torch.set_num_threads(args.threads)
    ids = perf.eval_ids()
    pop_of = {t: p for p in perf.POPULATIONS for t in perf.eval_ids(p)}
    run_dir = pilot_run._new_dir(ROOT / "constraint", "")
    fits = {f"{perf.label(a, g)}/{s}": perf.final_checkpoint(perf.run_dir_of(a, g, s)) for a, g in V1_ARMS
            for s in perf.SEEDS}
    identity.write_manifest(run_dir, config={
        "command": "gi constraint", "policies": ["v1", "v1 + same-reference ON restriction"],
        "checkpoints": {k: {"path": identity.run_ref(p),
                            **{f: (rec or {}).get(f) for f in ("sha256", "model_digest")}}
                        for k, (p, rec) in fits.items()},
        "eval_pin_sha256": identity.sha256_file(perf.EVAL_PIN), "device": "cpu", "matmul_precision": "highest",
        "threads": args.threads, "max_steps": infer_controller.DEFAULT_MAX_STEPS}, require_clean=not args.allow_dirty)
    encs = _encodings(ids)
    scores: dict = {}
    out = []
    t0 = time.time()
    for key, (path, _sha) in fits.items():
        label, seed = key.split("/")
        model = train.model_from_checkpoint(train.load_checkpoint(path), "cpu").eval()
        saved = _saved_records(label, int(seed))
        for tid in ids:
            task = data.task(tid)
            prep = K.prepare(encs[tid], grounding="v1")
            row = {"fit": key, "task_id": tid, "db_id": task.db_id, "population": pop_of[tid]}
            for name, cls in (("v1", None), ("restricted", GC.RestrictedV1Machine)):
                if cls is None:
                    dec = infer_controller.decode_one(model, prep)
                else:
                    with GC.machine_class(cls):
                        dec = infer_controller.decode_one(model, prep)
                rec = dec.as_record()
                sql = rec["final"]["sql"]
                score = _score(scores, task, sql)
                row[name] = {"actions": rec.get("actions"), "sql": sql, "kind": rec["final"]["kind"],
                             "reason": rec["final"].get("reason"), "correct": score["correct"],
                             "status": score["predicted"]["status"],
                             "exec_seconds": score["predicted"]["elapsed_seconds"],
                             "same_reference_on": _same_reference(dec.plan), "self_joins": GC.self_joins(dec.plan)}
            v1, re_ = row["v1"], row["restricted"]
            s = saved[tid]
            row["v1_equals_saved_decode"] = v1["actions"] == s["attempts"][0]["actions"]
            row["changed"] = v1["actions"] != re_["actions"]
            row["first_changed_step"] = next((i for i, (a, b) in enumerate(zip(v1["actions"] or [],
                                                                                 re_["actions"] or [])) if a != b),
                                             None) if row["changed"] else None
            out.append(row)
        print(json.dumps({"fit": key, "seconds": round(time.time() - t0)}), flush=True)
    with open(run_dir / "records.jsonl", "w") as f:
        for row in out:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    _write(run_dir / "summary.json", constraint_summary(out))
    print(json.dumps(constraint_summary(out)["by_fit_population"], indent=1))


def constraint_summary(rows: list) -> dict:
    """Per fit and population, and per architecture pooled over seeds: EX under each policy, the changes and their
    direction, timeouts, same-reference outputs, self-joins, and whether the CPU v1 decode reproduced the saved one."""
    def tally(rs):
        def n(f):
            return sum(1 for r in rs if f(r))
        return {"requests": len(rs), "v1_correct": n(lambda r: r["v1"]["correct"]),
                "restricted_correct": n(lambda r: r["restricted"]["correct"]),
                "changed_decisions": n(lambda r: r["changed"]),
                "changed_sql": n(lambda r: r["v1"]["sql"] != r["restricted"]["sql"]),
                "wrong_to_correct": n(lambda r: not r["v1"]["correct"] and r["restricted"]["correct"]),
                "correct_to_wrong": n(lambda r: r["v1"]["correct"] and not r["restricted"]["correct"]),
                "v1_timeouts": n(lambda r: r["v1"]["status"] == "timeout"),
                "restricted_timeouts": n(lambda r: r["restricted"]["status"] == "timeout"),
                "v1_errors": n(lambda r: r["v1"]["status"] == "error"),
                "restricted_errors": n(lambda r: r["restricted"]["status"] == "error"),
                "v1_same_reference_outputs": n(lambda r: r["v1"]["same_reference_on"] > 0),
                "restricted_same_reference_outputs": n(lambda r: r["restricted"]["same_reference_on"] > 0),
                "v1_self_join_outputs": n(lambda r: r["v1"]["self_joins"] > 0),
                "restricted_self_join_outputs": n(lambda r: r["restricted"]["self_joins"] > 0),
                "v1_abstentions": n(lambda r: r["v1"]["kind"] == "abstain"),
                "restricted_abstentions": n(lambda r: r["restricted"]["kind"] == "abstain"),
                "v1_reproduces_saved_decode": n(lambda r: r["v1_equals_saved_decode"])}
    out = {"by_fit_population": {}, "by_architecture_population": {}}
    for r in rows:
        out["by_fit_population"].setdefault(f"{r['fit']}/{r['population']}", []).append(r)
        out["by_architecture_population"].setdefault(f"{r['fit'].split('/')[0]}/{r['population']}", []).append(r)
    for k in out:
        out[k] = {key: tally(rs) for key, rs in sorted(out[k].items())}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("constraint")
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=constraint_cmd)
    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
