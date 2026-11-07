"""The feedback of the semantic-proposal-repair study: the raw execution observation both repair policies receive, and
the typed analysis only TI1 receives. Deterministic; no gold program, reference answer or evaluator metadata is read.

**Example.** For the proposal
`SELECT T2.app_id FROM label_categories AS T1 INNER JOIN app_labels AS T2 ON T1.label_id = T1.label_id`, the raw
observation is the execution's status, row count and first rows. The typed analysis adds:
- the plan: sources, joins, filters, projection;
- an observation: "the ON predicate of the join with app_labels references only label_categories".

The observation does not say which join was intended.

**What the analysis may say.**
- **Errors** (definite, from the snapshot and SQLite):
  - an identifier the snapshot's schema does not have, or an ambiguous or unresolved one (the parser's reasons);
  - a plan that fails validation;
  - a SQLite execution error, which covers type, operator or aggregate-placement errors SQLite actually enforces;
  - a timeout.
- **Warnings** (observations only, never verdicts):
  - a construct outside the typed representation (checked by execution only);
  - an ON predicate that does not reference the source it joins;
  - a join with no ON predicate;
  - a literal that does not occur verbatim in the question or evidence;
  - an empty result.

Nothing is rejected for looking unusual. Non-key joins, self-joins and empty results are legal.
"""
from __future__ import annotations

import time

FEEDBACK_VERSION = "spr-feedback-v1"
IDENTIFIER_REASONS = frozenset({"unknown table", "unknown column", "unresolved column", "ambiguous column",
                                "unresolved T.*", "unresolved derived column", "USING column not found"})
CELL_CHARS = 60
FEEDBACK_CHARS = 4000


def _cell(v) -> str:
    s = "NULL" if v is None else str(v)
    return s if len(s) <= CELL_CHARS else s[:CELL_CHARS] + "..."


def raw_observation(execution: dict) -> str:
    """The execution as both repair policies see it (`tools.execute_select`: at most the pinned preview rows)."""
    st = execution.get("status")
    if st != "ok":
        return f"Execution status: {st}. Error: {execution.get('error')}"
    rows = execution.get("rows") or []
    head = [f"Execution status: ok. Rows returned: {execution.get('row_count')}"
            + (f" (showing the first {len(rows)})" if execution.get("truncated") else "") + ".",
            "Columns: " + ", ".join(map(str, execution.get("columns") or []))]
    head += ["  " + " | ".join(_cell(v) for v in r) for r in rows]
    text = "\n".join(head)
    return text if len(text) <= FEEDBACK_CHARS else text[:FEEDBACK_CHARS] + "\n  ... (preview truncated)"


def _occurs(value: str, text: str) -> bool:
    v = value.strip().strip("'\"").strip()
    return bool(v) and v.lower() in text.lower()


def typed_analysis(task, snapshot, sql: "str | None", execution: dict) -> dict:
    """{typed, reason, plan_lines, errors, warnings, seconds} for one proposal."""
    from . import diagnose as D
    from . import ir
    from . import spr_typed as T
    t0 = time.perf_counter()
    errors, warnings, lines = [], [], []
    index = T.schema_index(task, snapshot)
    p = T.parse(sql, index)
    if not sql:
        errors.append("no SQL query was found in the reply")
    elif not p["supported"]:
        if p["reason"] in IDENTIFIER_REASONS:
            errors.append(f"{p['reason']} against this database's schema: {p['detail']}")
        else:
            warnings.append(f"the query uses a construct outside the typed representation ({p['reason']}); it was "
                            "checked by execution only")
    else:
        errors += p["errors"]
        plan = ir.from_json(p["plan"])
        text = f"{task.question}\n{task.evidence or ''}"
        for sel, outer, path in D.all_selects(plan):
            f = D.select_features(sel, outer)
            frames = f["scope"].frames[-1]
            name = "the main query" if path == "top" else f"the subquery at {path}"
            lines.append(f"In {name}:")
            lines.append("  sources: " + ", ".join(f"{t or 'a derived table'} (source {k})"
                                                  for k, (t, _a) in enumerate(frames)))
            for j in f["joins"]:
                lines.append(f"  join with {j['table']} (source {j['position']}): {j['join']}, ON {j['on']}")
                if j["on"] in ("-", None, "None") or j["join"] == "CROSS":
                    warnings.append(f"in {name}, the join with {j['table']} (source {j['position']}) has no ON "
                                    "predicate: every row of it pairs with every row before it")
                elif not j["constrains_new_source"]:
                    refs = ", ".join(f"{frames[k][0] or 'a derived table'} (source {k})"
                                     for k in j["on_references"]) or "no source"
                    warnings.append(f"in {name}, the ON predicate of the join with {j['table']} (source "
                                    f"{j['position']}) references only {refs}")
            for clause, values in (("filters", f["where"]), ("group by", f["group"]), ("order by", f["order"])):
                if values:
                    lines.append(f"  {clause}: " + "; ".join(map(str, values)))
            if f["having"] not in ("-", None):
                lines.append(f"  having: {f['having']}")
            lines.append("  returns: " + "; ".join(map(str, f["items"])) + (" (DISTINCT)" if f["distinct"] else "")
                         + (f"; limit {f['limit']}" if f["limit"] is not None else ""))
            for lit in f["literals"]:
                found = _occurs(lit, text)
                lines.append(f"  literal {lit}: {'occurs' if found else 'does not occur'} verbatim in the question "
                             "or evidence")
                if not found:
                    warnings.append(f"in {name}, the literal {lit} does not occur verbatim in the question or "
                                    "evidence (it may still be right)")
    st = execution.get("status")
    if st == "timeout":
        errors.append(f"execution timed out: {execution.get('error')}")
    elif st not in ("ok", None):
        errors.append(f"execution {st}: {execution.get('error')}")
    elif st == "ok" and execution.get("row_count") == 0:
        warnings.append("the query returned no rows (this can be the right answer)")
    return {"typed": bool(p["supported"]), "reason": p["reason"], "plan_lines": lines, "errors": errors,
            "warnings": warnings, "seconds": time.perf_counter() - t0}


def render_typed(analysis: dict) -> str:
    out = ["Typed analysis of your query (deterministic checks against this database's schema; the warnings are "
           "observations, not verdicts):"]
    if analysis["typed"]:
        out += ["Typed plan:"] + analysis["plan_lines"]
    else:
        out.append("Typed plan: not available.")
    out += ["Errors:"] + ([f"  - {e}" for e in analysis["errors"]] or ["  (none)"])
    out += ["Warnings:"] + ([f"  - {w}" for w in analysis["warnings"]] or ["  (none)"])
    text = "\n".join(out)
    return text if len(text) <= FEEDBACK_CHARS else text[:FEEDBACK_CHARS] + "\n  ... (analysis truncated)"
