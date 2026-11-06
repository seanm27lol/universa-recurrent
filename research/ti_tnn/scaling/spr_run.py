"""Commands of the semantic-proposal-repair study (development; `docs/PROTOCOL_semantic_proposal_repair.md`).

    python -m scaling.spr_run preserve                         # the typed path on #70's saved direct outputs (CPU)
    python -m scaling.spr_run dev --projected-hours 0.2        # prompt debugging on 16 earlier dev questions (GPU)
    python -m scaling.spr_run pin-new                          # positions 192-255, after a prior-use scan
    # after the freeze (the protocol and pins/spr_protocol.json committed):
    python -m scaling.spr_run run --projected-hours 0.9        # D0, D1, TI1 on the new population, seeds 200, 201
    python -m scaling.spr_run analyze                          # outcomes, lineage, costs (CPU)

**The policies.** Each question and sampling seed gets one initial proposal from the pinned Qwen3.5-4B with the
working direct-SQL prompt (`gi_llm.SYSTEM_DIRECT`). It is shared by all three policies:
- **D0:** the initial proposal, as it is.
- **D1:** the same proposal, then one ordinary self-revision with the raw execution observation.
- **TI1:** the same proposal, then one revision with the same raw observation plus the typed analysis
  (`spr_feedback`).

D1 and TI1 share everything else:
- the model, question, evidence, schema and retrieved examples;
- the initial reply in the conversation;
- the revision seed;
- 512 generated tokens per call;
- the request deadline;
- the selection rule.

Every D1 and TI1 request is revised.

**Selection (D1 and TI1 alike; never by gold).** The revised proposal is kept if its SQL was extracted, is one
read-only statement and executes without error or timeout. Otherwise the original is kept if it satisfies the same.
Otherwise the policy abstains. The original candidate stays recorded either way. D0 submits the initial proposal as
it is.

Runs go to `<runs root>/spr/`. The historical producers and their evidence are read, never written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from . import contracts as C

ROOT = C.RUNS / "spr"
SEEDS = (200, 201)
DEV_SEED = 900
DEV_SAMPLE = 16
DEV_SALT = "spr-dev-sample-v1"
MAX_NEW_TOKENS = 512
REQUEST_DEADLINE_SECONDS = 180.0   # the whole request: the initial proposal and its one revision
POLICIES = ("D0", "D1", "TI1")
LEAD = 8                           # the research-priority rule: TI1 - D1 >= 8 of 128, positive in both seeds and LOO
# This work order allows at most 6 GPU-reservation hours within authorization v2's 32 cumulative hours. The live
# ledger held 22.822751 h when it began.
WORK_ORDER_USED_AT_START = 22.822751
WORK_ORDER_LIMIT = min(32.0, WORK_ORDER_USED_AT_START + 6.0)
PROTOCOL = C.SCALING / "docs" / "PROTOCOL_semantic_proposal_repair.md"
PROTOCOL_PIN = C.PINS / "spr_protocol.json"
NEW_PIN = C.PINS / "spr_new_eval.json"
REVISE = ("Review your query against the question and the evidence, and correct it if needed. Reply with your final "
          "query in a ```sql``` code block; it may be the same query.")
PROMPT_VERSION = "spr-prompts-v1"


class SprStop(SystemExit):
    pass


def seed_of(kind: str, seed: int, task_id: str) -> int:
    return int(hashlib.sha256(f"spr-{kind}:{seed}:{task_id}".encode()).hexdigest()[:8], 16)


def check_budget(projected_hours: float) -> dict:
    from . import ledger
    u = ledger.usage()
    worst = u["used_hours"] + (1 + ledger.PROJECTION_MARGIN) * projected_hours
    out = {"used_hours": u["used_hours"], "projected_hours": projected_hours, "worst_case_hours": worst,
           "work_order_limit_hours": WORK_ORDER_LIMIT, "fits": worst <= WORK_ORDER_LIMIT}
    if not out["fits"]:
        raise SprStop(f"budget: {out}")
    return out


def protocol_identity() -> dict:
    from . import identity
    if not PROTOCOL_PIN.exists():
        raise SprStop("the protocol is not frozen (no pins/spr_protocol.json)")
    pin = json.loads(PROTOCOL_PIN.read_text())
    if identity.sha256_file(PROTOCOL) != pin["sha256"] or identity.sha256_file(NEW_PIN) != pin["new_eval_pin_sha256"]:
        raise SprStop("the protocol or the new-population pin differs from the frozen pin")
    return {"sha256": pin["sha256"], "frozen_at": pin.get("frozen_at")}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")


def dev_ids() -> list:
    from . import perf
    return sorted(perf.eval_ids(), key=lambda t: hashlib.sha256(f"{DEV_SALT}:{t}".encode()).hexdigest())[:DEV_SAMPLE]


# --- the preservation test ---------------------------------------------------------------------------------------
def preserve_cmd(args) -> None:
    """The typed path on the 64 saved direct-Qwen outputs of the grounding-interface study, with no model call: every
    supported query is parsed, compiled and executed, and its result compared with the original's."""
    from . import data, identity, pilot_run
    from . import spr_typed as T
    src = C.RUNS / "grounding_interface" / "direct" / "20261103T202939Z_new" / "records.jsonl"
    recs = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    run_dir = pilot_run._new_dir(ROOT / "preserve", "")
    identity.write_manifest(run_dir, config={"command": "spr preserve", "source": identity.run_ref(src.parent),
                                             "source_sha256": identity.sha256_file(src), "typed": T.TYPED_VERSION},
                            require_clean=not args.allow_dirty)
    rows = []
    for r in recs:
        task = data.task(r["task_id"])
        rows.append(T.preservation(task, data.snapshot(task.db_id), r["sql"]))
    for row in rows:
        if row["supported"] and row["compiled"]:
            both_failed = row["original_status"] == row["compiled_status"] and row["original_rows"] is None and \
                row["compiled_rows"] is None
            row["outcome"] = "preserved" if row["same_ex"] and (row["same_result_set"] or both_failed) else "CHANGED"
            row["note"] = "both executions ended with the same status and no rows" if both_failed else None
        else:
            row["outcome"] = "unsupported"
    with open(run_dir / "records.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    from collections import Counter
    sup = [r for r in rows if r["outcome"] != "unsupported"]
    out = {"records": len(rows), "supported": len(sup), "preserved": sum(r["outcome"] == "preserved" for r in sup),
           "changed": [r["task_id"] for r in sup if r["outcome"] == "CHANGED"],
           "same_result_set": sum(r["same_result_set"] for r in sup),
           "same_row_list": sum(r["same_row_list"] for r in sup),
           "both_failed_alike": sum(1 for r in sup if r.get("note")),
           "ex_original_supported": sum(r["original_correct"] for r in sup),
           "ex_compiled_supported": sum(r["compiled_correct"] for r in sup),
           "unsupported": dict(Counter(r["reason"] for r in rows if r["outcome"] == "unsupported"))}
    _write(run_dir / "summary.json", out)
    print(json.dumps(out, indent=1))
    if out["changed"]:
        raise SprStop(f"unexplained changes: {out['changed']}")


# --- one request -------------------------------------------------------------------------------------------------
def _call(rt, messages, cfg, seed: int, deadline: float) -> tuple:
    import torch
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    turn = rt.generate_turn(messages, None, cfg, seed, deadline=deadline)
    seconds = time.perf_counter() - t0
    return turn, {"seconds": seconds, "prompt_tokens": turn.prompt_tokens, "generated_tokens": turn.generated_tokens,
                  "finish_reason": turn.finish_reason, "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                  "content": turn.content}


def _execute(snap, sql):
    from . import tools
    t0 = time.perf_counter()
    if not sql:
        return {"status": "no_sql", "error": "no SQL in the reply", "rows": [], "columns": [], "row_count": None,
                "truncated": False, "seconds": 0.0}
    r = tools.execute_select(snap, sql)
    return {**{k: r.get(k) for k in ("status", "error", "rows", "columns", "row_count", "truncated")},
            "seconds": time.perf_counter() - t0}


def _valid(sql, execution) -> bool:
    from . import tools
    return bool(sql) and tools.check_read_only(sql) is None and execution["status"] == "ok"


def select(original: dict, revised: dict) -> dict:
    """The declared rule: revised if valid, else original if valid, else abstain."""
    if _valid(revised["sql"], revised["execution"]):
        return {"source": "revised", "sql": revised["sql"]}
    if _valid(original["sql"], original["execution"]):
        return {"source": "original", "sql": original["sql"]}
    return {"source": "abstain", "sql": None}


def serve(rt, cfg, task, snap, schema_text: str, examples: list, seed: int) -> dict:
    """One question at one sampling seed under all three policies (the initial proposal shared)."""
    from . import gi_llm, llm_agent
    from . import spr_feedback as F
    t_start = time.monotonic()
    deadline = t_start + REQUEST_DEADLINE_SECONDS
    base = gi_llm.messages(gi_llm.SYSTEM_DIRECT, task, schema_text, examples)
    turn0, call0 = _call(rt, base, cfg, seed_of("initial", seed, task.task_id), deadline)
    sql0, _raw = llm_agent.extract_final(turn0.content)
    exec0 = _execute(snap, sql0)
    analysis0 = F.typed_analysis(task, snap, sql0, exec0)
    original = {"sql": sql0, "call": call0, "execution": exec0, "typed": analysis0}
    raw = F.raw_observation(exec0)
    remaining = max(0.0, REQUEST_DEADLINE_SECONDS - (time.monotonic() - t_start))  # the same for both revisions
    out = {"task_id": task.task_id, "db_id": task.db_id, "seed": seed, "retrieved": [e.task_id for e in examples],
           "original": original, "D0": {"source": "original", "sql": sql0}, "revision_seconds_allowed": remaining}
    messages = {"D1": f"{raw}\n\n{REVISE}", "TI1": f"{raw}\n\n{F.render_typed(analysis0)}\n\n{REVISE}"}
    order = ("D1", "TI1") if seed_of("order", seed, task.task_id) % 2 == 0 else ("TI1", "D1")
    out["revision_order"] = list(order)
    for policy in order:
        message = messages[policy]
        msgs = base + [{"role": "assistant", "content": turn0.content}, {"role": "user", "content": message}]
        turn, call = _call(rt, msgs, cfg, seed_of("revise", seed, task.task_id), time.monotonic() + remaining)
        sql, _raw = llm_agent.extract_final(turn.content)
        ex = _execute(snap, sql)
        revised = {"sql": sql, "call": call, "execution": ex, "feedback": message,
                   "typed": F.typed_analysis(task, snap, sql, ex)}
        out[policy] = {"revised": revised, **select(original, revised)}
    return out


def _generation(limits):
    from . import gi_llm
    cfg = gi_llm.config(limits)
    assert cfg.max_new_tokens == MAX_NEW_TOKENS and not cfg.thinking
    return cfg


def _serve_population(args, ids: list, seeds, kind: str, proto: "dict | None") -> None:
    from . import data, gi_run, identity, ledger, pilot_run, retrieval, scorer
    from . import gi_llm, spr_feedback, spr_typed
    from .runtime import Runtime
    limits = C.Limits()
    budget = check_budget(args.projected_hours)
    cfg = _generation(limits)
    run_dir = pilot_run._new_dir(ROOT / kind, "")
    identity.write_manifest(run_dir, config={
        "command": f"spr {kind}", "protocol": proto, "requested": ids, "seeds": list(seeds),
        "system": gi_llm.SYSTEM_DIRECT, "revise": REVISE, "prompts": PROMPT_VERSION, "generation": cfg.as_dict(),
        "request_deadline_seconds": REQUEST_DEADLINE_SECONDS, "typed": spr_typed.TYPED_VERSION,
        "feedback": spr_feedback.FEEDBACK_VERSION, "preview_rows": limits.preview_rows,
        "sql_timeout_seconds": limits.sql_timeout_seconds, "budget": budget,
        "seed_rule": "int(sha256('spr-<initial|revise>:<seed>:<task_id>')[:8], 16)",
        "selection": "revised if extracted, read-only and executes; else original if so; else abstain"},
        require_clean=not args.allow_dirty, extra={"model_identity": identity.model_identity("B4")})
    index = retrieval.training_pool_index()
    rows = []
    with ledger.Reservation(f"spr {kind}", projected_hours=args.projected_hours,
                            extra={"run_dir": identity.run_ref(run_dir), "work_order_limit": WORK_ORDER_LIMIT}) as res:
        t0 = time.perf_counter()
        rt = Runtime("B4")
        load = time.perf_counter() - t0
        for seed in seeds:
            for tid in ids:
                res.check()
                task = data.task(tid)
                snap = data.snapshot(task.db_id)
                text, examples = gi_run._prompt_inputs(task, limits, index)
                row = serve(rt, cfg, task, snap, text, examples, seed)
                rows.append(row)
                with open(run_dir / "requests.jsonl", "a") as f:
                    f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                print(json.dumps({"seed": seed, "task": tid, "D1": row["D1"]["source"], "TI1": row["TI1"]["source"],
                                  "seconds": round(sum(row[p]["revised"]["call"]["seconds"] for p in ("D1", "TI1"))
                                                   + row["original"]["call"]["seconds"], 1)}), flush=True)
        out = {"load_seconds": load,
               "reservation": {"id": res.id, "hours": res.elapsed_hours(), "projected_hours": args.projected_hours}}
    with open(run_dir / "scores.jsonl", "w") as f:   # after the reservation: the scorer is not GPU work
        for row in rows:
            task = data.task(row["task_id"])
            snap, gold = data.snapshot(task.db_id), data.gold(task.task_id).sql
            for p in POLICIES:
                sc = scorer.score(snap, row[p]["sql"], gold)
                f.write(json.dumps({"seed": row["seed"], "task_id": row["task_id"], "db_id": row["db_id"], "policy": p,
                                    "source": row[p]["source"], "sql": row[p]["sql"], "score": sc}, sort_keys=True,
                                   default=str) + "\n")
    _write(run_dir / "summary.json", out)


def dev_cmd(args) -> None:
    """Prompt debugging on 16 of the 128 earlier dev questions at one debugging seed (never the new population)."""
    _serve_population(args, dev_ids(), (DEV_SEED,), "dev", None)


def run_cmd(args) -> None:
    proto = protocol_identity()
    ids = json.loads(NEW_PIN.read_text())["task_ids"]
    _serve_population(args, ids, SEEDS, "run", proto)


def pin_new_cmd(args) -> None:
    """Pin inventory positions 192-255 after checking that no model run, diagnosis or analysis named them."""
    from . import data, gi_run
    ids = data.inventory_ids(C.INVENTORY["size"])[192:256]
    want, hits = set(ids), []
    for root in (C.RUNS, C.SCALING / "evidence", C.PINS):   # everything, the grounding-interface runs included
        for f in sorted(Path(root).rglob("*")):
            if not f.is_file() or f.suffix not in (".json", ".jsonl", ".txt", ".log") or "cache" in f.parts \
                    or f == NEW_PIN or ROOT in f.parents:
                continue
            n = sum(1 for t in want if t in f.read_text(errors="ignore"))
            if n:
                hits.append({"file": str(f.relative_to(root)), "root": root.name, "ids": n})
    expected = {(e["root"], e["file"]) for e in gi_run.EXPECTED_PRIOR_USE}
    unexpected = [h for h in hits if (h["root"], h["file"]) not in expected]
    if unexpected:
        raise SprStop(f"positions 192-255 were already used: {unexpected[:5]}; choose the next unused block")
    _write(NEW_PIN, {"population": "new", "positions": [192, 256], "task_ids": ids,
                     "db_ids": [data.task(t).db_id for t in ids], "inventory": C.INVENTORY["salt"], "prior_use": hits,
                     "prior_use_note": "the inventory pin itself, the data-preparation gold check and the grammar's "
                     "gold-program coverage analysis; no model, controller, baseline or diagnosis run read them"})
    print(json.dumps({"pinned": len(ids), "prior_use": hits}, indent=1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("preserve")
    s.add_argument("--allow-dirty", action="store_true")
    s.set_defaults(fn=preserve_cmd)
    for name, fn in (("dev", dev_cmd), ("run", run_cmd)):
        s = sub.add_parser(name)
        s.add_argument("--projected-hours", type=float, required=True)
        s.add_argument("--allow-dirty", action="store_true")
        s.set_defaults(fn=fn)
    s = sub.add_parser("pin-new")
    s.set_defaults(fn=pin_new_cmd)
    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
