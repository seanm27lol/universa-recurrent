"""The typed path of the semantic-proposal-repair study: a model's SQL proposal, parsed into the existing typed
intermediate representation and checked against the exact database snapshot.

**Example.** For the proposal
`SELECT T2.app_id FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T1.label_id`,
the typed path finds that the ON predicate references only source T1 (an advisory observation, not a verdict),
compiles the plan back to SQL, and executes it read-only.

**What it does, in order.**
1. **Parse.** `sql_to_ir.try_convert` against the snapshot's own schema (`schema.load_schema(db, snapshot)`):
   identifiers resolve to that schema's tables and columns, with aliases and scope. A query outside the IR is an
   explicit *untyped fallback*. It keeps the ordinary read-only and resource checks and stays in the denominator.
2. **Validate.** `ir.validate` gives definite errors.
3. **Compile.** `ir.compile` turns the plan back into SQL. The compiled query is what the typed path executes. The
   preservation test (`preserve`) checks, query by query, that this changes no result.
4. **Observe.** Deterministic diagnostics are split into:
   - **errors:** unknown identifiers or invalid scope (from the parser), validation errors, SQLite execution errors
     and timeouts;
   - **warnings:** advisory and observational. An ON predicate that does not reference the source it joins; a
     literal that occurs nowhere in the question or evidence; an empty result.
5. **Execute.** `tools.execute_select`: read-only, the pinned timeout, a bounded preview.

**What it never claims.** A query that parses, validates and executes is not called semantically correct, and no
query is rejected for looking unusual. Non-key joins, self-joins, empty results and constructs outside the IR stay
legal.
"""
from __future__ import annotations

import time

from . import contracts as C

TYPED_VERSION = "spr-typed-v1"


def schema_index(task, snapshot):
    from . import ir
    from . import schema as schema_mod
    return ir.SchemaIndex.from_schema(schema_mod.load_schema(task.db_id, snapshot))


def parse(sql: "str | None", index) -> dict:
    """{supported, plan (JSON), compiled, errors, reason} for one proposal."""
    from . import ir, sql_to_ir
    if not sql:
        return {"supported": False, "reason": "no SQL", "detail": "", "plan": None, "compiled": None, "errors": []}
    t0 = time.perf_counter()
    try:
        plan, err = sql_to_ir.try_convert(sql, index)
    except Exception as error:  # the parser itself failed on this text: untyped, never silently dropped
        plan, err = None, type("E", (), {"reason": f"parser error: {type(error).__name__}", "detail": str(error)})()
    if err is not None:
        return {"supported": False, "reason": err.reason, "detail": (err.detail or "")[:300], "plan": None,
                "compiled": None, "errors": [], "seconds": time.perf_counter() - t0}
    errors = ir.validate(plan, index)
    compiled = ir.compile(plan) if not errors else None
    return {"supported": True, "reason": None, "detail": "", "plan": ir.to_json(plan), "compiled": compiled,
            "errors": [f"validate: {e}" for e in errors], "seconds": time.perf_counter() - t0}


def run_rows(snapshot, sql: str, timeout: "float | None" = None) -> dict:
    """Execute read-only on the snapshot with the scorer's own machinery: {status, rows (list of tuples), error}."""
    from . import scorer, tools
    timeout = float(timeout or C.Limits().sql_timeout_seconds)
    tools.verify_snapshot(snapshot)
    conn = tools.connect_readonly(snapshot.path)
    try:
        start = time.perf_counter()
        info, rows = scorer._run_side(conn, str(sql), tools.Deadline(conn, timeout, start), shared=False)
    finally:
        conn.close()
    return {"status": info["status"], "rows": rows, "error": info.get("error"),
            "seconds": time.perf_counter() - start}


def preservation(task, snapshot, sql: "str | None") -> dict:
    """The preservation check of one saved proposal: parse, compile, and compare the compiled query's execution
    with the original's, as result sets (what EX compares) and as ordered row lists, and both scores."""
    from . import data, scorer
    index = schema_index(task, snapshot)
    p = parse(sql, index)
    out = {"task_id": task.task_id, "db_id": task.db_id, "snapshot_sha256": snapshot.sha256, "sql": sql,
           "supported": p["supported"], "reason": p["reason"], "detail": p["detail"], "compiled": p["compiled"],
           "validate_errors": p["errors"]}
    if not p["supported"] or p["compiled"] is None:
        return out
    gold = data.gold(task.task_id).sql
    so, sc = scorer.score(snapshot, sql, gold), scorer.score(snapshot, p["compiled"], gold)
    ro, rc = run_rows(snapshot, sql), run_rows(snapshot, p["compiled"])
    out.update({"original_correct": so["correct"], "compiled_correct": sc["correct"],
                "original_status": so["predicted"]["status"], "compiled_status": sc["predicted"]["status"],
                "same_ex": so["correct"] == sc["correct"],
                "same_result_set": ro["rows"] is not None and rc["rows"] is not None and
                set(ro["rows"]) == set(rc["rows"]),
                "same_row_list": ro["rows"] is not None and ro["rows"] == rc["rows"],
                "same_multiset": ro["rows"] is not None and rc["rows"] is not None and
                sorted(map(repr, ro["rows"])) == sorted(map(repr, rc["rows"])),
                "original_rows": None if ro["rows"] is None else len(ro["rows"]),
                "compiled_rows": None if rc["rows"] is None else len(rc["rows"])})
    return out
