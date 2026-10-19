"""Independent reconstruction of session records (protocols/integration_test_v1.md, revision 2: "Analysis and
independent verification").

**Example.** A record says a query paused after one call on S0, migrated to S1, and found
`country_40_city_2_airport`. This checker never runs the producer. It does not import `recurrence.py`,
`sessions.py` or the update checker. It reads only:
- the record;
- each snapshot's recorded atomic-output table (relation -> input -> output);
- the reference world's facts.

**What it re-derives:**
- **Every application.** Each recorded application must be exactly what the table of the snapshot named by
  its segment gives: the relation, the consumed value, the produced value.
- **Traversal.** Its own traversal of the bounded query over a table gives the visited airports, the
  application sequence, the number of `step` calls, and the answer or terminal reason.
- **Each policy's claim:**
  - a one-segment record (start, pin or restart) must equal the traversal over its snapshot's table;
  - a migrated record must carry a justification for exactly this snapshot pair, whose difference it
    re-derives from the two tables. Its prefix must consume nothing in that difference, S1's table must give
    every prefix output again, and the whole record must equal the traversal over S1's table: mixed
    provenance, certified S1 equivalence.
- **Unjustified mixing.** Any other record that changes snapshot is an *unjustified switch*, and never
  valid. If applications actually came from two snapshots, it is also *unjustified mixing*.
- **Naive continuation's failures:**
  - **contract violation:** the outcome differs from the traversal over the snapshot it finished on;
  - **no-version outcome:** the outcome differs from the traversal over every snapshot that supplied an
    application.
- **Correctness.** The outcome is compared with the reference world, kept separate from faithfulness.
  A faithful record can be wrong.

**What it does not establish.** That the recorded tables are the snapshots' real behaviour. The producer
computes them from the modules, and a separate check must compare them with the snapshot itself.
"""
from __future__ import annotations

from .contracts import content_identity

TERMINAL = ("dead end", "cycle", "budget exhausted")
SCHEMA = "ti_tnn_session_v1"


def table_digest(table: dict) -> str:
    return content_identity(table)


def facts_table(world) -> dict:
    """The reference world as an atomic table, in the same form."""
    return {r: dict(t) for r, t in world.facts.items()}


def traverse(table: dict, start: str, target: str, budget: int, step_relation: str = "connects_to",
             test_relation: str = "operator_of") -> dict:
    """The bounded until-query over an atomic table, written independently of `recurrence.py`. Each loop is one
    call to `step`: apply the step relation (recorded even when it ends the run), stop at a dead end, a cycle, a
    missing output or the budget, and otherwise apply the test relation to the new airport."""
    current, visited, applications, calls, steps = start, [start], [], 0, 0
    while True:
        calls += 1
        if steps >= budget:
            return _out("budget exhausted", None, visited, applications, calls)
        nxt = table[step_relation].get(current)
        applications.append([step_relation, current, nxt])
        if nxt is None:
            return _out("module gave no valid answer", None, visited, applications, calls)
        if nxt == current:
            return _out("dead end", None, visited, applications, calls)
        if nxt in visited:
            return _out("cycle", None, visited, applications, calls)
        visited.append(nxt)
        current = nxt
        steps += 1
        tested = table[test_relation].get(nxt)
        applications.append([test_relation, nxt, tested])
        if tested is None:
            return _out("module gave no valid answer", None, visited, applications, calls)
        if tested == target:
            return _out("found", nxt, visited, applications, calls)


def _out(reason, answer, visited, applications, calls):
    return {"reason": reason, "answer": answer, "visited": list(visited), "applications": applications, "calls": calls}


def _record_outcome(record) -> dict:
    st = record["state"]
    answer = st["answer"]["value"] if st.get("answer") else None
    reason = "found" if st["status"] == "answered" else st["reason"]
    apps = [[a["relation_id"], a["consumed"]["value"], None if a["produced"] is None else a["produced"]["value"]]
            for a in st["applications"]]
    return {"reason": reason, "answer": answer, "visited": list(st["visited"]), "applications": apps,
            "calls": record["calls"]}


def _same_run(a: dict, b: dict) -> bool:
    return all(a[k] == b[k] for k in ("reason", "answer", "visited", "applications", "calls"))


def difference_keys(t0: dict, t1: dict) -> set:
    return {(r, x) for r in t0 for x in t0[r] if t0[r][x] != t1[r].get(x)}


def check_session(record: dict, tables: dict, world=None) -> dict:
    """Re-derive one completed session record. `tables` maps snapshot digest -> atomic table."""
    problems, out = [], {"unjustified_mixing": False, "unjustified_switch": False, "equivalent_to": None,
                         "contract_violation": None, "no_version_outcome": None, "correct": None}
    if record.get("schema") != SCHEMA:
        return {**out, "ok": False, "problems": ["not a session record"]}
    st, q = record["state"], record["state"]["query"]
    if st["status"] == "running":
        return {**out, "ok": False, "problems": ["the session has not finished"]}
    budget = record["budget"]
    if type(budget) is not int or not 1 <= budget <= 16:
        ok = st["status"] == "abstained" and st["reason"].startswith("rejected")
        return {**out, "ok": ok, "problems": [] if ok else ["a budget outside 1-16 was executed"]}
    if st["reason"].startswith("rejected"):
        return {**out, "ok": not st["applications"], "problems": [] if not st["applications"] else
                ["a rejected query applied modules"]}
    segments = record["segments"]
    missing = [s["snapshot"] for s in segments if s["snapshot"] not in tables]
    if missing:
        return {**out, "ok": False, "problems": [f"no atomic table for snapshot {d[:12]}" for d in missing]}
    got = _record_outcome(record)
    bounds = [s["from_application"] for s in segments] + [len(got["applications"])]
    for seg, lo, hi in zip(segments, bounds, bounds[1:]):  # provenance: each application from its segment's table
        table = tables[seg["snapshot"]]
        for i in range(lo, hi):
            rel, consumed, produced = got["applications"][i]
            if table[rel].get(consumed) != produced:
                problems.append(f"application {i}: {rel}({consumed}) = {produced!r}, but that snapshot gives "
                                f"{table[rel].get(consumed)!r}")
    start, target = q["start"]["value"], q["target"]["value"]
    run = lambda d: traverse(tables[d], start, target, budget, q["step_relation"], q["test_relation"])
    suppliers = list(dict.fromkeys(seg["snapshot"] for seg, lo, hi in zip(segments, bounds, bounds[1:]) if hi > lo))
    final = segments[-1]["snapshot"]
    if len(segments) == 1:
        if not _same_run(got, run(final)):
            problems.append("the record is not the traversal over its snapshot")
        else:
            out["equivalent_to"] = final
    else:
        m = record.get("migration") or {}
        justified = (len(segments) == 2 and segments[1]["how"] == "migrate" and m.get("migrated") is True
                     and (m.get("from"), m.get("to")) == (segments[0]["snapshot"], final))
        if justified:
            t0, t1 = tables[segments[0]["snapshot"]], tables[final]
            changed = difference_keys(t0, t1)
            prefix = got["applications"][:segments[1]["from_application"]]
            if any((rel, consumed) in changed for rel, consumed, _ in prefix):
                problems.append("migration: the prefix consumed an input whose output differs between the snapshots")
                justified = False
            if any(t1[rel].get(consumed) != produced for rel, consumed, produced in prefix):
                problems.append("migration: the new snapshot does not reproduce the prefix")
                justified = False
            if justified and not _same_run(got, run(final)):
                problems.append("migration: the record is not equivalent to the traversal over the new snapshot")
                justified = False
            if justified:
                out["equivalent_to"] = final
        if not justified:
            out["unjustified_switch"] = True  # a change of snapshot without a valid justification
            out["unjustified_mixing"] = len(suppliers) > 1  # and applications actually came from two snapshots
    out["contract_violation"] = not _same_run(got, run(final))
    out["no_version_outcome"] = all(not _same_run(got, run(d)) for d in suppliers or [final])
    if world is not None:
        ref = traverse(facts_table(world), start, target, budget, q["step_relation"], q["test_relation"])
        out["correct"] = (got["reason"], got["answer"]) == (ref["reason"], ref["answer"])
    ok = not problems and not out["unjustified_switch"]
    return {**out, "ok": ok, "problems": problems}
