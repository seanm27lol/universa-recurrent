"""Commands of the grounding-interface study (development; `docs/PROTOCOL_grounding_interface.md`).

    python -m scaling.gi_run constraint   # post hoc: the six v1 checkpoints, v1 and v1 + the ON restriction (CPU)
    python -m scaling.gi_run ground --population dev --projected-hours 0.8   # G4 grounding calls (GPU reservation)
    python -m scaling.gi_run lexical --population dev                        # Glex records (CPU)
    python -m scaling.gi_run decode-dev                                      # G0, Glex, G4, oracle, swaps (CPU)

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
V2_ARMS = (("T4", "v2"), ("c4_params", "v2"))
POLICIES = ("G0", "Glex", "G4")
# The work order of 2026-11-03 allots at most 8 GPU-reservation hours, within authorization v2's 32 cumulative hours.
# The ledger read when it began held 21.415907 h, so no reservation may let the cumulative total pass this.
WORK_ORDER_USED_AT_START = 21.415907
WORK_ORDER_LIMIT = min(32.0, WORK_ORDER_USED_AT_START + 8.0)


class GiStop(SystemExit):
    pass


def check_budget(projected_hours: float) -> dict:
    """Before a reservation: the hours used plus 1.5 x the projection (the watchdog's worst case) must stay within
    the work order's limit."""
    from . import ledger
    u = ledger.usage()
    worst = u["used_hours"] + (1 + ledger.PROJECTION_MARGIN) * projected_hours
    out = {"used_hours": u["used_hours"], "projected_hours": projected_hours, "worst_case_hours": worst,
           "work_order_limit_hours": WORK_ORDER_LIMIT, "fits": worst <= WORK_ORDER_LIMIT}
    if not out["fits"]:
        raise GiStop(f"budget: {out}")
    return out


def population_ids(population: str) -> list:
    """dev: the performance study's 128 evaluated questions (positions 0-127). new: the pinned block (positions
    128-191), only after the protocol freeze."""
    from . import perf
    if population == "dev":
        return perf.eval_ids()
    if population == "new":
        return json.loads(NEW_PIN.read_text())["task_ids"]
    raise ValueError(population)


NEW_PIN = C.PINS / "gi_new_eval.json"


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


def _prompt_inputs(task, limits, index) -> tuple:
    """(schema text, retrieved examples) exactly as the B4 harness builds them (`cli.baseline`)."""
    from . import data, retrieval
    from . import schema as schema_mod
    snap = data.snapshot(task.db_id)
    text = schema_mod.serialize(schema_mod.load_schema(task.db_id, snap),
                                budget_chars=schema_mod.PROMPT_BUDGET_CHARS).text
    return text, retrieval.retrieve(task, limits.retrieval_k, index=index)


def _items(task):
    """The request's schema items as the controller's encoding serializes them (no budget), for a preview of a
    record's resolution; consumption resolves again against the encoding's own items."""
    from . import actions as A
    from . import schema as schema_mod
    sch = schema_mod.load_schema(task.db_id)
    return A.SchemaItems(schema_mod.serialize(sch).items, sch)


def ground_cmd(args) -> None:
    """G4: one bounded Qwen3.5-4B grounding call per question of a population (GPU reservation)."""
    from . import data, gi_llm, identity, ledger, pilot_run, retrieval
    from .runtime import Runtime
    ids = population_ids(args.population)
    limits = C.Limits()
    budget = check_budget(args.projected_hours)
    cfg = gi_llm.config(limits)
    run_dir = pilot_run._new_dir(ROOT / "ground", args.population)
    identity.write_manifest(run_dir, config={"command": f"gi ground {args.population}", "requested": ids,
                                             "prompt": gi_llm.PROMPT_VERSION, "system": gi_llm.SYSTEM_GROUND,
                                             "generation": cfg.as_dict(), "record": "gi-record-v1",
                                             "budget": budget, "seed_rule": "int(sha256('G4:' + task_id)[:8], 16)"},
                            require_clean=not args.allow_dirty, extra={"model_identity": identity.model_identity("B4")})
    index = retrieval.training_pool_index()
    out = {"requested": len(ids)}
    with ledger.Reservation(f"gi ground {args.population}", projected_hours=args.projected_hours,
                            extra={"run_dir": identity.run_ref(run_dir), "work_order_limit": WORK_ORDER_LIMIT}) as res:
        t0 = time.perf_counter()
        rt = Runtime("B4")
        out["load_seconds"] = time.perf_counter() - t0
        for tid in ids:
            res.check()
            task = data.task(tid)
            text, examples = _prompt_inputs(task, limits, index)
            rec = gi_llm.ground_call(rt, cfg, task, text, examples, _items(task))
            rec["retrieved"] = [e.task_id for e in examples]
            with open(run_dir / "records.jsonl", "a") as f:
                f.write(json.dumps(rec, sort_keys=True, default=str) + "\n")
            print(json.dumps({"task": tid, "valid": rec["grounding"]["valid"], "reason": rec["grounding"]["reason"],
                              "finish": rec["call"]["finish_reason"], "seconds": round(rec["seconds"], 2)}), flush=True)
        out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}
    _write(run_dir / "summary.json", out)


def lexical_cmd(args) -> None:
    """Glex records for a population (CPU, timed per request)."""
    from . import data, gi_lexical, identity, pilot_run
    from . import gi_record as GR
    from . import schema as schema_mod
    ids = population_ids(args.population)
    run_dir = pilot_run._new_dir(ROOT / "lexical", args.population)
    identity.write_manifest(run_dir, config={"command": f"gi lexical {args.population}", "requested": ids},
                            require_clean=not args.allow_dirty)
    with open(run_dir / "records.jsonl", "w") as f:
        for tid in ids:
            task = data.task(tid)
            t0 = time.perf_counter()
            items = _items(task)
            obj = gi_lexical.ground(task.question, task.evidence, schema_mod.load_schema(task.db_id), items)
            g = GR.resolve(obj, items, task.question, task.evidence)
            f.write(json.dumps({"task_id": tid, "record": obj, "grounding": g.as_dict(),
                                "seconds": time.perf_counter() - t0}, sort_keys=True) + "\n")


def _latest(kind: str, population: str) -> Path:
    runs = sorted(d for d in (ROOT / kind).glob(f"*_{population}") if (d / "records.jsonl").exists())
    if not runs:
        raise GiStop(f"no {kind} run for {population}")
    return runs[-1]


def _load_records(run_dir: Path) -> dict:
    return {r["task_id"]: r for r in map(json.loads, (run_dir / "records.jsonl").read_text().splitlines()) if r}


def swap_partner(ids: list, db_of: dict) -> dict:
    """Each question's swap partner: the next question of the same database in population order (cyclic); none for a
    database with one question."""
    by_db: dict = {}
    for t in ids:
        by_db.setdefault(db_of[t], []).append(t)
    out = {}
    for qs in by_db.values():
        if len(qs) > 1:
            for k, t in enumerate(qs):
                out[t] = qs[(k + 1) % len(qs)]
    return out


def decode_rows(model, enc, policies: dict, score_fn, task) -> dict:
    """Decode one request under each named policy ({name: biases or None}); score each."""
    from . import controller as K
    from . import gi_record as GR
    from . import infer_controller
    prep = K.prepare(enc, grounding=model.cfg.grounding)
    out = {}
    for name, biases in policies.items():
        with GR.pointer_prior(model, biases):
            dec = infer_controller.decode_one(model, prep)
        rec = dec.as_record()
        sc = score_fn(task, rec["final"]["sql"])
        out[name] = {"actions": rec["actions"], "sql": rec["final"]["sql"], "kind": rec["final"]["kind"],
                     "reason": rec["final"].get("reason"), "correct": sc["correct"],
                     "status": sc["predicted"]["status"], "exec_seconds": sc["predicted"]["elapsed_seconds"],
                     "work": rec["controller_work"]}
    return out


def decode_dev_cmd(args) -> None:
    """Development decoding on the CPU (cached anchored states), for the six saved v2 controllers on the 128 dev
    questions: G0, Glex, G4, the marked oracle (gold tables and columns, unordered) and the G4 record swapped with
    another question's of the same database."""
    import torch

    from . import data, identity, perf, pilot_run, sql_to_ir, train
    from . import gi_record as GR
    torch.set_num_threads(args.threads)
    ids = population_ids("dev")
    ground_dir, lex_dir = _latest("ground", "dev"), _latest("lexical", "dev")
    g4, glex = _load_records(ground_dir), _load_records(lex_dir)
    db_of = {t: data.task(t).db_id for t in ids}
    partner = swap_partner(ids, db_of)
    run_dir = pilot_run._new_dir(ROOT / "decode_dev", "")
    fits = {f"{perf.label(a, g)}/{s}": perf.final_checkpoint(perf.run_dir_of(a, g, s)) for a, g in V2_ARMS
            for s in perf.SEEDS}
    identity.write_manifest(run_dir, config={
        "command": "gi decode-dev", "ground_run": identity.run_ref(ground_dir),
        "lexical_run": identity.run_ref(lex_dir),
        "checkpoints": {k: {"path": identity.run_ref(p), **{f: (r or {}).get(f) for f in ("sha256", "model_digest")}}
                        for k, (p, r) in fits.items()},
        "policies": ["G0", "Glex", "G4", "oracle (evaluator information; diagnostic only)", "G4-swap"],
        "device": "cpu", "threads": args.threads}, require_clean=not args.allow_dirty)
    encs = _encodings(ids)
    scores: dict = {}
    t0 = time.time()
    with open(run_dir / "records.jsonl", "w") as f:
        for key, (path, _r) in fits.items():
            label, seed = key.split("/")
            model = train.model_from_checkpoint(train.load_checkpoint(path), "cpu").eval()
            saved = _saved_records(label, int(seed))
            for tid in ids:
                task, enc = data.task(tid), encs[tid]
                items, n = enc.schema, len(enc.schema.items)
                res = {name: GR.resolve(r["record"], items, task.question, task.evidence,
                                        truncated=r.get("call", {}).get("finish_reason", "eos") != "eos")
                       if r.get("record") is not None else GR.Grounding(False, r["grounding"]["reason"])
                       for name, r in (("Glex", glex[tid]), ("G4", g4[tid]))}
                plan, err = sql_to_ir.try_convert(data.gold(tid).sql, items.index)
                oracle = GR.oracle_record(plan, items) if err is None else None
                policies = {"G0": None, "Glex": res["Glex"].biases(n), "G4": res["G4"].biases(n),
                            "oracle": GR.oracle_biases(oracle, n) if oracle else None}
                if tid in partner:
                    other = g4[partner[tid]]
                    sw = GR.resolve(other["record"], items, task.question, task.evidence,
                                    truncated=other["call"]["finish_reason"] != "eos") \
                        if other.get("record") is not None else GR.Grounding(False, "partner record invalid")
                    policies["G4-swap"] = sw.biases(n)
                rows = decode_rows(model, enc, policies, lambda t, q: _score(scores, t, q), task)
                row = {"fit": key, "task_id": tid, "db_id": task.db_id, "policies": rows,
                       "grounding": {k: v.as_dict() for k, v in res.items()}, "oracle": oracle,
                       "swap_partner": partner.get(tid),
                       "g0_equals_saved_decode": rows["G0"]["actions"] == saved[tid]["attempts"][0]["actions"]}
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            print(json.dumps({"fit": key, "seconds": round(time.time() - t0)}), flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("constraint")
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=constraint_cmd)
    s = sub.add_parser("ground")
    s.add_argument("--population", choices=("dev", "new"), required=True)
    s.add_argument("--projected-hours", type=float, required=True)
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=ground_cmd)
    s = sub.add_parser("lexical")
    s.add_argument("--population", choices=("dev", "new"), required=True)
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=lexical_cmd)
    s = sub.add_parser("decode-dev")
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=decode_dev_cmd)
    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
