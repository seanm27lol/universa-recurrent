"""Commands of the grounding-interface study (development; `docs/PROTOCOL_grounding_interface.md`).

    python -m scaling.gi_run constraint   # post hoc: the six v1 checkpoints, v1 and v1 + the ON restriction (CPU)
    python -m scaling.gi_run ground --population dev --projected-hours 0.8   # G4 grounding calls (GPU reservation)
    python -m scaling.gi_run lexical --population dev                        # Glex records (CPU)
    python -m scaling.gi_run decode-dev                                      # G0, Glex, G4, oracle, swaps (CPU)
    # after the freeze (docs/PROTOCOL_grounding_interface.md and pins/gi_protocol.json committed):
    python -m scaling.gi_run encode-new --projected-hours 0.15    # the new contexts, anchored (GPU reservation)
    python -m scaling.gi_run ground --population new --projected-hours 0.35
    python -m scaling.gi_run lexical --population new
    python -m scaling.gi_run eval-new --projected-hours 0.6       # timed G0 / Glex / G4 decoding (GPU reservation)
    python -m scaling.gi_run direct-new --projected-hours 0.35    # the short-budget direct SQL control (GPU)
    python -m scaling.gi_run swap --population new                # the record-swap interface check (CPU)
    python -m scaling.gi_run analyze                              # outcomes, diagnostics and costs (CPU)

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
        protocol_identity()
        return json.loads(NEW_PIN.read_text())["task_ids"]
    raise ValueError(population)


NEW_PIN = C.PINS / "gi_new_eval.json"
PROTOCOL = C.SCALING / "docs" / "PROTOCOL_grounding_interface.md"
PROTOCOL_PIN = C.PINS / "gi_protocol.json"
MATMUL_PRECISION = "high"   # the controller's decoding precision in the performance study's timed evaluation
RECHECK = 8                 # contexts re-encoded against their anchors before new ones are written


def protocol_identity() -> dict:
    """The frozen protocol: its sha256 and the new-population pin's must equal the protocol pin's."""
    from . import identity
    if not PROTOCOL_PIN.exists():
        raise GiStop("the grounding-interface protocol is not frozen (no pins/gi_protocol.json)")
    pin = json.loads(PROTOCOL_PIN.read_text())
    if identity.sha256_file(PROTOCOL) != pin["sha256"]:
        raise GiStop("docs/PROTOCOL_grounding_interface.md differs from its frozen hash")
    if identity.sha256_file(NEW_PIN) != pin["new_eval_pin_sha256"]:
        raise GiStop("pins/gi_new_eval.json differs from the pin the protocol froze")
    return {"file": str(PROTOCOL.relative_to(C.REPO)), "sha256": pin["sha256"], "frozen_at": pin.get("frozen_at")}


def gi_anchors() -> tuple:
    """The performance study's anchors, and this study's (accepted only from runs under the frozen protocol)."""
    from . import perf
    from . import pilot as P
    found, ident = perf.anchors()
    if PROTOCOL_PIN.exists():
        mine, ident2, _ignored = P.load_anchors(root=ROOT / "encode", protocol_sha256=protocol_identity()["sha256"])
        if ident is not None and ident2 is not None and ident != ident2:
            raise GiStop("this study's anchors come from another backbone identity")
        for tid, a in mine.items():
            old = found.get(tid)
            if old is not None and (old["cache_key"], old["states_sha256"]) != (a["cache_key"], a["states_sha256"]):
                raise GiStop(f"{tid}: this study anchored another entry")
            found.setdefault(tid, a)
        ident = ident or ident2
    return found, ident


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
    from . import encode, pilot_run
    anchors, ident = gi_anchors()
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


# --- the new population (after the freeze) -------------------------------------------------------------------------
def prior_use(task_ids: list) -> list:
    """Every file under the runs root, the evidence bundles and the pins that names any of `task_ids` (the pin of
    the inventory itself, data checks and the gold-program coverage analysis are expected; no model run may)."""
    want = set(task_ids)
    hits = []
    for root in (C.RUNS, C.SCALING / "evidence", C.PINS):
        for f in sorted(Path(root).rglob("*")):
            if not f.is_file() or f.suffix not in (".json", ".jsonl", ".txt", ".log") or "cache" in f.parts:
                continue
            if f == NEW_PIN or ROOT in f.parents:
                continue
            text = f.read_text(errors="ignore")
            n = sum(1 for t in want if t in text)
            if n:
                hits.append({"file": str(f.relative_to(root)), "root": root.name, "ids": n})
    return hits


EXPECTED_PRIOR_USE = ({"root": "pins", "file": "inventory_v1.json"}, {"root": "ti_tnn_scaling", "file":
                      "data/gold_check_dev_v1.json"}, {"root": "ti_tnn_scaling", "file": "ir/coverage_details_v1.json"})


def pin_new_cmd(args) -> None:
    """Pin inventory positions 128-191 after checking that no model run, diagnosis or analysis named them."""
    from . import data
    ids = data.inventory_ids(C.INVENTORY["size"])[128:192]
    hits = prior_use(ids)
    expected = {(e["root"], e["file"]) for e in EXPECTED_PRIOR_USE}
    unexpected = [h for h in hits if (h["root"], h["file"]) not in expected]
    if unexpected:
        raise GiStop(f"positions 128-191 were already used: {unexpected[:5]}; choose the next unused block")
    _write(NEW_PIN, {"population": "new", "positions": [128, 192], "task_ids": ids,
                     "db_ids": [data.task(t).db_id for t in ids], "inventory": C.INVENTORY["salt"],
                     "prior_use": hits, "prior_use_note": "the inventory pin itself, the data-preparation gold check "
                     "(gold SQL executes on the snapshot) and the grammar's gold-program coverage analysis; no model, "
                     "controller, baseline or diagnosis run read these questions"})
    print(json.dumps({"pinned": len(ids), "prior_use": hits}, indent=1))


def encode_new_cmd(args) -> None:
    """Encode the new population's contexts into the cache and anchor them, after re-encoding RECHECK earlier
    contexts against their anchors (cross-process determinism)."""
    from . import data, encode, identity, ledger, pilot_run, retrieval
    from . import schema as schema_mod
    proto = protocol_identity()
    ids = population_ids("new")
    recheck = population_ids("dev")[:RECHECK]
    budget = check_budget(args.projected_hours)
    run_dir = pilot_run._new_dir(ROOT / "encode", "new")
    identity.write_manifest(run_dir, config={"command": "gi encode-new", "ids": ids, "recheck": recheck,
                                             "cache_root": "PILOT_CACHE", "protocol": proto, "budget": budget},
                            require_clean=not args.allow_dirty, extra={"model_identity": identity.model_identity("B4")})
    known, ident = gi_anchors()
    out = {"requested": len(ids), "rechecked": [], "encoded": 0, "already_present": 0}
    with ledger.Reservation("gi encode-new", projected_hours=args.projected_hours,
                            extra={"run_dir": identity.run_ref(run_dir), "work_order_limit": WORK_ORDER_LIMIT}) as res:
        _rt, bb = pilot_run._load_backbone()
        if ident is not None and bb.identity() != ident:
            raise GiStop("this backbone's identity differs from the anchors'")
        pilot_run._write_backbone(run_dir, bb)
        for tid in recheck:
            res.check()
            task = data.task(tid)
            fresh = encode.encode_request(bb, task, schema_mod.load_schema(task.db_id), retrieval.retrieve(task),
                                          use_cache=False)
            equal = encode.states_sha256(fresh.states) == known[tid]["states_sha256"]
            out["rechecked"].append({"task_id": tid, "equal": equal})
            if not equal:
                _write(run_dir / "summary.json", out)
                raise GiStop(f"{tid}: a fresh forward differs from its anchor")
        for tid in ids:
            res.check()
            task = data.task(tid)
            t0 = time.perf_counter()
            enc = encode.encode_request(bb, task, schema_mod.load_schema(task.db_id), retrieval.retrieve(task),
                                        cache_root=encode.PILOT_CACHE, require_verified=True)
            digest, how = enc.cost["states_sha256"], "written"
            if enc.cost["cache_hit"]:
                fresh = encode.encode_request(bb, task, schema_mod.load_schema(task.db_id), retrieval.retrieve(task),
                                              use_cache=False)
                if encode.states_sha256(fresh.states) != digest:
                    raise GiStop(f"{tid}: a fresh forward differs from the cached entry")
                how, out["already_present"] = "recomputed and equal", out["already_present"] + 1
            else:
                out["encoded"] += 1
            if tid in known and known[tid]["states_sha256"] != digest:
                raise GiStop(f"{tid}: the entry differs from its existing anchor")
            with open(run_dir / "entries.jsonl", "a") as f:
                f.write(json.dumps({"task_id": tid, "cache_key": enc.cost["cache_key"], "states_sha256": digest,
                                    "tokens": enc.cost["tokens"], "cache_hit": enc.cost["cache_hit"], "how": how,
                                    "forward_seconds": enc.cost["forward_seconds"],
                                    "seconds": time.perf_counter() - t0}, sort_keys=True) + "\n")
        out["reservation"] = {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}
    _write(run_dir / "summary.json", out)
    print(json.dumps(out, indent=1, default=str))


def _resolve_policy(name: str, rec: "dict | None", items, task) -> "object":
    """A policy's record for one request, resolved against the request's own schema items."""
    from . import gi_record as GR
    if name == "G0" or rec is None:
        return None
    if rec.get("record") is None:
        return GR.Grounding(False, (rec.get("grounding") or {}).get("reason") or "no record")
    truncated = (rec.get("call") or {}).get("finish_reason", "eos") != "eos"
    return GR.resolve(rec["record"], items, task.question, task.evidence, truncated=truncated)


def eval_new_cmd(args) -> None:
    """The timed evaluation on the new population: the six saved v2 controllers under G0, Glex and G4. Every request
    runs its own fresh backbone forward and its policy's biased decode (Glex computes its record inside the timed
    span; G4's call is timed in its own run and added in the analysis). Each decode is then reconstructed from the
    anchored cache with the same prior and must be equal. Scoring follows the reservation."""
    import hashlib

    import torch

    from . import controller as K
    from . import data, encode, gi_lexical, identity, infer_controller, ledger, perf, pilot_run, retrieval, train
    from . import gi_record as GR
    from . import schema as schema_mod
    proto = protocol_identity()
    ids = population_ids("new")
    ground_dir, lex_dir = _latest("ground", "new"), _latest("lexical", "new")
    g4 = _load_records(ground_dir)
    if sorted(g4) != sorted(ids):
        raise GiStop("the G4 run does not cover the new population")
    budget = check_budget(args.projected_hours)
    fits = {f"{perf.label(a, g)}/{s}": perf.final_checkpoint(perf.run_dir_of(a, g, s)) for a, g in V2_ARMS
            for s in perf.SEEDS}
    run_dir = pilot_run._new_dir(ROOT / "eval", "new")
    identity.write_manifest(run_dir, config={
        "command": "gi eval-new", "protocol": proto, "requested": ids, "policies": list(POLICIES),
        "ground_run": identity.run_ref(ground_dir), "lexical_run_reference": identity.run_ref(lex_dir),
        "checkpoints": {k: {"path": identity.run_ref(p), **{f: (r or {}).get(f) for f in ("sha256", "model_digest")}}
                        for k, (p, r) in fits.items()},
        "matmul_precision": MATMUL_PRECISION, "budget": budget}, require_clean=not args.allow_dirty,
        extra={"model_identity": identity.model_identity("B4")})
    anchors, ident = gi_anchors()
    reader = encode.Backbone.cache_reader(ident, pilot_run._tokenizer())
    decodes = []
    with ledger.Reservation("gi eval-new", projected_hours=args.projected_hours,
                            extra={"run_dir": identity.run_ref(run_dir), "work_order_limit": WORK_ORDER_LIMIT}) as res:
        _rt, bb = pilot_run._load_backbone()
        if bb.identity() != ident:
            raise GiStop("the backbone's identity differs from the anchors'")
        for key, (path, _r) in fits.items():
            model = train.model_from_checkpoint(train.load_checkpoint(path), "cuda").eval()
            for policy in POLICIES:
                for tid in ids:
                    res.check()
                    task = data.task(tid)
                    schema, examples = schema_mod.load_schema(task.db_id), retrieval.retrieve(task)
                    t0 = time.perf_counter()
                    enc = encode.encode_request(bb, task, schema, examples, use_cache=False)
                    if policy == "Glex":
                        lex = gi_lexical.ground(task.question, task.evidence, schema, enc.schema)
                        g = GR.resolve(lex, enc.schema, task.question, task.evidence)
                    else:
                        lex = None
                        g = _resolve_policy(policy, g4[tid], enc.schema, task)
                    biases = g.biases(len(enc.schema.items)) if g is not None else None
                    with GR.pointer_prior(model, biases):
                        dec = infer_controller._decode_at(MATMUL_PRECISION, model,
                                                          K.prepare(enc, grounding=model.cfg.grounding),
                                                          infer_controller.DEFAULT_MAX_STEPS)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - t0
                    digest = hashlib.sha256(enc.states.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    cached = encode.encode_request(reader, task, schema, examples, cache_root=encode.PILOT_CACHE,
                                                   require_hit=True, require_verified=True)
                    with GR.pointer_prior(model, biases):
                        again = infer_controller._decode_at(MATMUL_PRECISION, model,
                                                            K.prepare(cached, grounding=model.cfg.grounding),
                                                            infer_controller.DEFAULT_MAX_STEPS)
                    rec = dec.as_record()
                    decodes.append({"fit": key, "policy": policy, "task_id": tid, "db_id": task.db_id,
                                    "final": rec["final"], "actions": rec["actions"], "plan": rec["plan"],
                                    "controller_work": rec["controller_work"], "controller_seconds": elapsed,
                                    "grounding": None if g is None else g.as_dict(), "lexical_record": lex,
                                    "states_sha256": digest, "anchor_equal": digest == anchors[tid]["states_sha256"],
                                    "reconstruction_equal": again.as_record()["actions"] == rec["actions"],
                                    "encode_cost": enc.cost})
            print(json.dumps({"fit": key, "done": len(decodes)}), flush=True)
        out = {"reservation": {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}}
    _write(run_dir / "decodes.json", decodes)  # paid for: on disk before any scoring
    bad = [d for d in decodes if not (d["anchor_equal"] and d["reconstruction_equal"])]
    out["integrity"] = {"requests": len(decodes), "anchor_or_reconstruction_failures": len(bad)}
    scores: dict = {}
    with open(run_dir / "records.jsonl", "w") as f:
        for d in decodes:
            sc = _score(scores, data.task(d["task_id"]), d["final"]["sql"])
            f.write(json.dumps({**{k: v for k, v in d.items() if k != "plan"}, "score": sc}, sort_keys=True,
                               default=str) + "\n")
    _write(run_dir / "summary.json", out)
    if bad:
        raise GiStop(f"integrity: {len(bad)} decodes differ from their anchors or reconstructions")
    print(json.dumps(out, indent=1))


def direct_new_cmd(args) -> None:
    """The short-budget direct control on the new population: one non-thinking Qwen3.5-4B call for SQL per question
    (the G4 configuration), scored after the reservation."""
    from . import data, gi_llm, identity, ledger, pilot_run, retrieval
    from .runtime import Runtime
    proto = protocol_identity()
    ids = population_ids("new")
    limits = C.Limits()
    budget = check_budget(args.projected_hours)
    cfg = gi_llm.config(limits)
    run_dir = pilot_run._new_dir(ROOT / "direct", "new")
    identity.write_manifest(run_dir, config={"command": "gi direct-new", "protocol": proto, "requested": ids,
                                             "prompt": gi_llm.DIRECT_VERSION, "system": gi_llm.SYSTEM_DIRECT,
                                             "generation": cfg.as_dict(), "budget": budget,
                                             "seed_rule": "int(sha256('B4-direct:' + task_id)[:8], 16)"},
                            require_clean=not args.allow_dirty, extra={"model_identity": identity.model_identity("B4")})
    index = retrieval.training_pool_index()
    calls = []
    with ledger.Reservation("gi direct-new", projected_hours=args.projected_hours,
                            extra={"run_dir": identity.run_ref(run_dir), "work_order_limit": WORK_ORDER_LIMIT}) as res:
        t0 = time.perf_counter()
        rt = Runtime("B4")
        load = time.perf_counter() - t0
        for tid in ids:
            res.check()
            task = data.task(tid)
            text, examples = _prompt_inputs(task, limits, index)
            calls.append(gi_llm.direct_call(rt, cfg, task, text, examples))
        out = {"load_seconds": load,
               "reservation": {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}}
    scores: dict = {}
    with open(run_dir / "records.jsonl", "w") as f:
        for c in calls:
            f.write(json.dumps({**c, "score": _score(scores, data.task(c["task_id"]), c["sql"])}, sort_keys=True,
                               default=str) + "\n")
    _write(run_dir / "summary.json", out)


def swap_cmd(args) -> None:
    """The interface check (CPU, cached states): every question decoded with its own G4 record and with its swap
    partner's (another question of the same database), by the six saved v2 controllers. Kept apart from the
    ordinary evaluation; a change shows that the record influences the system, not that it is right."""
    import torch

    from . import data, identity, perf, pilot_run, train
    torch.set_num_threads(args.threads)
    ids = population_ids(args.population)
    g4_dir = _latest("ground", args.population)
    g4 = _load_records(g4_dir)
    db_of = {t: data.task(t).db_id for t in ids}
    partner = swap_partner(ids, db_of)
    fits = {f"{perf.label(a, g)}/{s}": perf.final_checkpoint(perf.run_dir_of(a, g, s)) for a, g in V2_ARMS
            for s in perf.SEEDS}
    run_dir = pilot_run._new_dir(ROOT / "swap", args.population)
    identity.write_manifest(run_dir, config={"command": f"gi swap {args.population}", "ground_run":
                                             identity.run_ref(g4_dir), "partners": partner, "device": "cpu"},
                            require_clean=not args.allow_dirty)
    encs = _encodings(ids)
    scores: dict = {}
    with open(run_dir / "records.jsonl", "w") as f:
        for key, (path, _r) in fits.items():
            model = train.model_from_checkpoint(train.load_checkpoint(path), "cpu").eval()
            for tid in ids:
                if tid not in partner:
                    continue
                task, enc = data.task(tid), encs[tid]
                own = _resolve_policy("G4", g4[tid], enc.schema, task)
                other = _resolve_policy("G4", g4[partner[tid]], enc.schema, task)
                n = len(enc.schema.items)
                rows = decode_rows(model, enc, {"G4": own.biases(n), "G4-swap": other.biases(n)},
                                   lambda t, q: _score(scores, t, q), task)
                f.write(json.dumps({"fit": key, "task_id": tid, "partner": partner[tid], "policies": rows,
                                    "own_valid": own.valid, "partner_valid": other.valid}, sort_keys=True,
                                   default=str) + "\n")


# --- analysis -------------------------------------------------------------------------------------------------------
GATE_SHARE = 0.10   # the interface gate: a policy must change the final SQL on at least this share of dev decodes


def gold_reference(tid: str, items) -> dict:
    """The gold program's tables, first FROM table, columns and (column, literal) comparisons, as item indexes and
    normalized values: annotations for the diagnostics, not proof that alternatives are wrong."""
    from . import data, ir, sql_to_ir
    from . import diagnose as D
    from . import gi_record as GR
    plan, err = sql_to_ir.try_convert(data.gold(tid).sql, items.index)
    if err is not None:
        return {"convertible": False}
    o = GR.oracle_record(plan, items)
    top = D.top_select(plan)
    first = None
    if top is not None and top.from_ and isinstance(top.from_[0].source, ir.TableSource):
        first = items.table_item.get(top.from_[0].source.table)
    lits = set()
    for sel, outer, _p in D.all_selects(plan):
        scope = outer.push(sel)
        for part in [i.on for i in sel.from_] + [sel.where, sel.having]:
            if part is None:
                continue
            for n in ir.walk(part):
                if isinstance(n, ir.Compare):
                    for col, lit in ((n.left, n.right), (n.right, n.left)):
                        if isinstance(col, ir.Column) and col.level == 0 and isinstance(lit, ir.Literal) and \
                                lit.value is not None:
                            table = scope.frames[-1][col.source][0]
                            ci = items.column_item.get((table, col.name)) or next(
                                (i for (t, c), i in items.column_item.items()
                                 if t.lower() == str(table).lower() and c.lower() == col.name.lower()), None)
                            lits.add((ci, normalize_value(lit.value)))
    return {"convertible": True, "tables": o["tables"], "columns": o["columns"], "first_table": first,
            "literals": sorted(lits, key=str)}


def normalize_value(v) -> str:
    return str(v).strip().strip("'\"").strip().lower()


def record_diagnostics(g: dict, gold: dict) -> dict:
    """One resolved record against the gold annotations."""
    if not gold.get("convertible") or not g or not g.get("valid"):
        return {"scored": False}
    tset, gt = set(g["tables"]), set(gold["tables"])
    cset, gc = set(g["columns"][:16]), set(gold["columns"])
    first = g["tables"][0] if g["tables"] else None
    lits = [(x["item"], normalize_value(x["value"])) for x in g["literals"]]
    gold_values = {v for _c, v in gold["literals"]}
    return {"scored": True, "table_recall": len(tset & gt) / len(gt) if gt else None,
            "table_precision": len(tset & gt) / len(tset) if tset else None,
            "table_set_exact": tset == gt, "first_in_gold_set": first in gt if first is not None else None,
            "first_equals_gold_first": first == gold["first_table"] if first is not None else None,
            "column_coverage": len(cset & gc) / len(gc) if gc else None,
            "literals": len(lits), "literal_value_correct": sum(v in gold_values for _c, v in lits),
            "literal_pair_correct": sum((c, v) in set(map(tuple, gold["literals"])) for c, v in lits),
            "gold_literals": len(gold["literals"]), "unresolved": g.get("status") == "unresolved",
            "invalid_literals": len(g.get("invalid_literals") or [])}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize_records(rows: list) -> dict:
    """Record diagnostics pooled over requests: rates and means."""
    sc = [r for r in rows if r.get("scored")]
    lit_total = sum(r["literals"] for r in sc)
    return {"requests": len(rows), "scored": len(sc),
            "table_recall": _mean(r["table_recall"] for r in sc),
            "table_precision": _mean(r["table_precision"] for r in sc),
            "table_set_exact": _mean(r["table_set_exact"] for r in sc),
            "first_in_gold_set": _mean(r["first_in_gold_set"] for r in sc),
            "first_equals_gold_first": _mean(r["first_equals_gold_first"] for r in sc),
            "column_coverage": _mean(r["column_coverage"] for r in sc),
            "literal_bindings": lit_total,
            "literal_value_accuracy": sum(r["literal_value_correct"] for r in sc) / lit_total if lit_total else None,
            "literal_pair_accuracy": sum(r["literal_pair_correct"] for r in sc) / lit_total if lit_total else None,
            "unresolved_rate": _mean(r["unresolved"] for r in sc),
            "invalid_literal_bindings": sum(r["invalid_literals"] for r in sc)}


def dev_summary_cmd(args) -> None:
    """The development decodes: EX per policy, influence (changed SQL against G0), the interface gate, the CPU G0
    reproduction of the saved GPU decodes, the record diagnostics and the swap check."""
    from . import data
    run = sorted((ROOT / "decode_dev").glob("*/records.jsonl"))[-1]
    rows = [json.loads(x) for x in run.read_text().splitlines() if x.strip()]
    pols = ["G0", "Glex", "G4", "oracle", "G4-swap"]
    out = {"run": run.parent.name, "decodes": len(rows), "by_arch": {}, "influence": {}, "records": {}}
    for arch in ("T4-v2", "C4-v2"):
        rs = [r for r in rows if r["fit"].startswith(arch)]
        out["by_arch"][arch] = {p: sum(r["policies"][p]["correct"] for r in rs if p in r["policies"]) for p in pols}
        out["by_arch"][arch]["requests"] = len(rs)
        for p in pols[1:]:
            have = [r for r in rs if p in r["policies"]]
            out["by_arch"][arch][f"{p}_vs_G0"] = {
                "requests": len(have),
                "changed_sql": sum(r["policies"][p]["sql"] != r["policies"]["G0"]["sql"] for r in have),
                "wrong_to_correct": sum(r["policies"][p]["correct"] > r["policies"]["G0"]["correct"] for r in have),
                "correct_to_wrong": sum(r["policies"][p]["correct"] < r["policies"]["G0"]["correct"] for r in have)}
    for p in ("Glex", "G4", "oracle"):
        n = sum(r["policies"][p]["sql"] != r["policies"]["G0"]["sql"] for r in rows)
        out["influence"][p] = {"changed_sql": n, "share": n / len(rows)}
    out["gate"] = {"rule": f"some policy changes the final SQL on >= {GATE_SHARE:.0%} of the dev decodes",
                   "passes": any(v["share"] >= GATE_SHARE for v in out["influence"].values())}
    out["g0_reproduces_saved_gpu_decode"] = sum(r["g0_equals_saved_decode"] for r in rows)
    encs = None
    per = {"Glex": [], "G4": []}
    seen = set()
    for r in rows:
        if r["task_id"] in seen:
            continue
        seen.add(r["task_id"])
        if encs is None:
            encs = _encodings([x["task_id"] for x in rows if x["fit"] == rows[0]["fit"]])
        gold = gold_reference(r["task_id"], encs[r["task_id"]].schema)
        for p in per:
            per[p].append(record_diagnostics(r["grounding"][p], gold))
    out["records"] = {p: {**summarize_records(v), "valid": sum(1 for r in rows if r["fit"] == rows[0]["fit"]
                                                                and r["grounding"][p]["valid"])}
                      for p, v in per.items()}
    _write(run.parent / "dev_summary.json", out)
    print(json.dumps(out, indent=1))


def _reading(cmp: dict, interaction: dict, direct_beats: "bool | None") -> list:
    """The protocol's readings, in order, from the per-architecture comparisons."""
    def consistent_up(d):
        return d["all_conditions"] and d["sign"] > 0
    out = []
    g4_g0 = {a: consistent_up(cmp[a]["G4-G0"]) for a in cmp}
    g4_glex = {a: consistent_up(cmp[a]["G4-Glex"]) for a in cmp}
    glex_g0 = {a: consistent_up(cmp[a]["Glex-G0"]) for a in cmp}
    if all(g4_g0.values()) and all(g4_glex.values()):
        out.append("G4 beats both G0 and Glex consistently for both architectures: the task-conditioned grounding "
                   "interface merits a larger paired study")
    elif any(g4_g0.values()) and not any(g4_glex.values()):
        out.append("G4 beats G0 but not Glex: explicit grounding helps; neural grounding has not earned its cost")
    if any(g4_g0.values()) or any(glex_g0.values()):
        if interaction["all_conditions"] and interaction["sign"] > 0:
            out.append("T4 improves more than C4_params: a development interaction worth investigating, not "
                       "confirmation")
        else:
            out.append("the architectures do not differ consistently in their improvement: an interface "
                       "improvement, not evidence for specialization")
    if direct_beats:
        out.append("the direct short-budget B4 answers more questions than the best controller policy in every "
                   "seed: retain that baseline and reconsider whether the controller adds value")
    if not any(g4_g0.values()) and not any(glex_g0.values()):
        out.append("nothing improves consistently: do not spend the remainder on more seeds or larger controllers")
    return out


def analyze_cmd(args) -> None:
    """Outcomes, diagnostics and costs on the new population (CPU), from the frozen runs."""
    from . import data, identity, ir, ledger, pilot_run
    from . import gi_record as GR
    from . import perf_analysis as PA
    proto = protocol_identity()
    ids = population_ids("new")
    db_of = {t: data.task(t).db_id for t in ids}
    ev = _latest("eval", "new")
    recs = [json.loads(x) for x in (ev / "records.jsonl").read_text().splitlines() if x.strip()]
    plans = {(d["fit"], d["policy"], d["task_id"]): d["plan"] for d in json.loads((ev / "decodes.json").read_text())}
    seeds = sorted({int(r["fit"].split("/")[1]) for r in recs})
    correct, by = {}, {}
    for r in recs:
        arch, seed = r["fit"].split("-")[0], int(r["fit"].split("/")[1])
        correct.setdefault((f"{arch}-{r['policy']}", seed), {})[r["task_id"]] = r["score"]["correct"]
        by[(arch, r["policy"], seed, r["task_id"])] = r
    for k, v in correct.items():
        if sorted(v) != sorted(ids):
            raise GiStop(f"{k}: the evaluation covers {len(v)} of {len(ids)}")
    out = {"protocol": proto, "eval_run": identity.run_ref(ev), "seeds": seeds,
           "counts": {f"{lab}/{s}": sum(v.values()) for (lab, s), v in sorted(correct.items())}, "comparisons": {}}
    for arch in ("T4", "C4"):
        out["comparisons"][arch] = {}
        for a, b in (("G4", "G0"), ("Glex", "G0"), ("G4", "Glex")):
            d = PA.paired(correct, f"{arch}-{a}", f"{arch}-{b}", seeds, ids, db_of)
            out["comparisons"][arch][f"{a}-{b}"] = d
    inter = {("I", s): {t: correct[("T4-G4", s)][t] - correct[("T4-G0", s)][t] for t in ids} for s in seeds}
    inter.update({("Z", s): {t: correct[("C4-G4", s)][t] - correct[("C4-G0", s)][t] for t in ids} for s in seeds})
    out["interaction_T4_minus_C4_of_G4_minus_G0"] = PA.paired(inter, "I", "Z", seeds, ids, db_of)
    dr = _latest("direct", "new")
    direct = [json.loads(x) for x in (dr / "records.jsonl").read_text().splitlines() if x.strip()]
    d_ok = sum(r["score"]["correct"] for r in direct)
    best = {s: max(out["counts"][f"{a}-{p}/{s}"] for a in ("T4", "C4") for p in POLICIES) for s in seeds}
    out["direct"] = {"run": identity.run_ref(dr), "correct": d_ok, "requests": len(direct),
                     "abstentions": sum(r["final"]["kind"] == "abstain" for r in direct),
                     "truncated": sum(r["call"]["finish_reason"] != "eos" for r in direct),
                     "timeouts": sum(r["score"]["predicted"]["status"] == "timeout" for r in direct),
                     "call_seconds_mean": _mean(r["seconds"] for r in direct),
                     "best_controller_policy_per_seed": best,
                     "beats_best_controller_every_seed": all(d_ok > v for v in best.values())}
    gd = _latest("ground", "new")
    g4 = _load_records(gd)
    encs = _encodings(ids)
    gold = {t: gold_reference(t, encs[t].schema) for t in ids}
    diag = {"G4": [], "Glex": []}
    for t in ids:
        task = data.task(t)
        g = _resolve_policy("G4", g4[t], encs[t].schema, task)
        diag["G4"].append(record_diagnostics(g.as_dict(), gold[t]))
        lex = by[("T4", "Glex", seeds[0], t)]["grounding"]
        diag["Glex"].append(record_diagnostics(lex, gold[t]))
    from collections import Counter
    out["records"] = {p: summarize_records(v) for p, v in diag.items()}
    out["records"]["G4"].update({
        "valid": sum(r["grounding"]["valid"] for r in g4.values()),
        "fallback_reasons": dict(Counter(r["grounding"]["reason"] for r in g4.values() if not r["grounding"]["valid"])),
        "truncated": sum(r["call"]["finish_reason"] != "eos" for r in g4.values()),
        "call_seconds_mean": _mean(r["seconds"] for r in g4.values()), "ground_run": identity.run_ref(gd)})
    out["decodes"] = {}
    for arch in ("T4", "C4"):
        for p in ("Glex", "G4"):
            rows = [(by[(arch, p, s, t)], by[(arch, "G0", s, t)]) for s in seeds for t in ids]
            alt = 0
            for (x, _g0) in rows:
                if x["score"]["correct"]:
                    plan = plans[(x["fit"], p, x["task_id"])]
                    pt = GR.oracle_record(ir.from_json(plan), encs[x["task_id"]].schema)["tables"] if plan else []
                    alt += gold[x["task_id"]].get("convertible") and set(pt) != set(gold[x["task_id"]]["tables"])
            out["decodes"][f"{arch}-{p}"] = {
                "changed_decisions_vs_G0": sum(x["actions"] != g0["actions"] for x, g0 in rows),
                "changed_sql_vs_G0": sum(x["final"]["sql"] != g0["final"]["sql"] for x, g0 in rows),
                "wrong_to_correct": sum(x["score"]["correct"] > g0["score"]["correct"] for x, g0 in rows),
                "correct_to_wrong": sum(x["score"]["correct"] < g0["score"]["correct"] for x, g0 in rows),
                "fallback_to_G0": sum(not (x["grounding"] or {}).get("valid", False) for x, _g in rows),
                "ex_correct_with_another_table_set": int(alt)}
    cost = {}
    g4_sec = {t: g4[t]["seconds"] for t in ids}
    for arch in ("T4", "C4"):
        for p in POLICIES:
            rows = [by[(arch, p, s, t)] for s in seeds for t in ids]
            ctrl = sum(r["controller_seconds"] for r in rows)
            ground = sum(g4_sec[r["task_id"]] for r in rows) if p == "G4" else 0.0
            exe = sum(r["score"]["predicted"]["elapsed_seconds"] for r in rows)
            ok = sum(r["score"]["correct"] for r in rows)
            cost[f"{arch}-{p}"] = {"requests": len(rows), "ex_correct": ok, "controller_seconds": ctrl,
                                   "grounding_call_seconds": ground, "submitted_execution_seconds": exe,
                                   "per_request_seconds": (ctrl + ground + exe) / len(rows),
                                   "per_ex_correct_seconds": (ctrl + ground + exe) / ok if ok else None,
                                   "timeouts": sum(r["score"]["predicted"]["status"] == "timeout" for r in rows),
                                   "abstentions": sum(r["final"]["kind"] == "abstain" for r in rows)}
    dsec = sum(r["seconds"] for r in direct) + sum(r["score"]["predicted"]["elapsed_seconds"] for r in direct)
    cost["direct-B4-short"] = {"requests": len(direct), "ex_correct": d_ok, "per_request_seconds": dsec / len(direct),
                               "per_ex_correct_seconds": dsec / d_ok if d_ok else None}
    out["cost"] = cost
    out["gpu_hours"] = {r["command"]: r["hours"] for r in ledger.usage()["reservations"]
                        if r["command"].startswith("gi ")}
    out["ledger"] = {k: v for k, v in ledger.usage().items() if k != "reservations"}
    sw = sorted((ROOT / "swap").glob("*_new/records.jsonl"))
    if sw:
        rows = [json.loads(x) for x in sw[-1].read_text().splitlines() if x.strip()]
        out["swap"] = {"run": identity.run_ref(sw[-1].parent), "decodes": len(rows),
                       "changed_sql": sum(r["policies"]["G4"]["sql"] != r["policies"]["G4-swap"]["sql"] for r in rows),
                       "own_correct": sum(r["policies"]["G4"]["correct"] for r in rows),
                       "swapped_correct": sum(r["policies"]["G4-swap"]["correct"] for r in rows)}
    out["reading"] = _reading(out["comparisons"], out["interaction_T4_minus_C4_of_G4_minus_G0"],
                              out["direct"]["beats_best_controller_every_seed"])
    run_dir = pilot_run._new_dir(ROOT / "analysis", "")
    identity.write_manifest(run_dir, config={"command": "gi analyze", "protocol": proto,
                                             "eval_run": identity.run_ref(ev)}, require_clean=not args.allow_dirty)
    _write(run_dir / "analysis.json", out)
    print(json.dumps({"counts": out["counts"], "rules": {a: {k: (v["pooled"], v["per_seed"], v["all_conditions"])
                                                              for k, v in c.items()} for a, c in
                                                          out["comparisons"].items()},
                      "direct": out["direct"]["correct"], "reading": out["reading"]}, indent=1, default=str))


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
    s = sub.add_parser("dev-summary")
    s.set_defaults(fn=dev_summary_cmd)
    s = sub.add_parser("analyze")
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=analyze_cmd)
    s = sub.add_parser("pin-new")
    s.set_defaults(fn=pin_new_cmd)
    for name, fn in (("encode-new", encode_new_cmd), ("eval-new", eval_new_cmd), ("direct-new", direct_new_cmd)):
        s = sub.add_parser(name)
        s.add_argument("--projected-hours", type=float, required=True)
        s.add_argument("--allow-dirty", action="store_true")
        s.set_defaults(fn=fn)
    s = sub.add_parser("swap")
    s.add_argument("--population", choices=("dev", "new"), required=True)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=swap_cmd)
    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
