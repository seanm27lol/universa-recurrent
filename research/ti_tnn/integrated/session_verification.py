"""Independent reconstruction of session records (protocols/integration_test_v1.md: "Analysis and independent
verification").

**Example.** A record says a query paused after one call on S0, migrated to S1, and found
`country_40_city_2_airport`. This checker never runs the producer: it does not import `recurrence.py`,
`sessions.py`, `serving.py` or the update checker. It reads only:
- the record;
- each snapshot's exported atomic-output table (relation -> input -> output);
- the exported serving contexts (identity -> snapshot digest, ...);
- optionally, the reference world's facts.

It returns three separate judgments.

**1. Execution validity: is the record a faithful execution?** It replays the bounded query with its own
traversal, switching tables exactly where the record's segments say, at the recorded call indices. The replay
must reproduce:
- every application (relation, consumed, produced);
- the visited airports;
- the terminal result and the number of `step` calls;
- the application index at which each segment begins.

Every segment's context must resolve to an exported context on that segment's snapshot. A corrupted record,
naive or not, is an invalid execution.

**2. Policy validity: was what it did permitted?**
- **No resume and pin:** one segment, unchanged.
- **Restart:** one fresh segment. Its `restarted_from` is checked by replaying the paused prefix it discarded.
- **Migrate:** either two segments with a proof, or a restart with a proof of why migration was refused. Every
  proof field is re-derived from the tables:
  - the difference identity;
  - the snapshot pair;
  - the prefix's inputs;
  - the revalidation count;
  - the blocking inputs.
- **Naive continuation** (an unchecked switch) is never permitted.

**3. Semantic failures, judged only on valid executions,** on the outcome (the answer or terminal reason, and
the visited path):
- **contract violation:** the outcome differs from the traversal over the snapshot it finished on;
- **no-version outcome:** the outcome differs from the traversal over every snapshot that supplied an
  application.

`unjustified_switch` marks a change of snapshot without a valid justification. `unjustified_mixing` also needs
applications from two snapshots. At pause index 0 there is a switch but no mixing.

**Correctness against the reference world** is reported separately from all three. A faithful record can be
wrong.

**What it does not establish:** that an exported table is the snapshot's real behaviour. The producer computes
the tables from the modules; the export must also carry each snapshot's digest, which a separate check binds
to the table.
"""
from __future__ import annotations

from .contracts import content_identity

TERMINAL = ("dead end", "cycle", "budget exhausted")
SCHEMA = "ti_tnn_session_v1"
OUTCOME_KEYS = ("reason", "answer", "visited")


def table_digest(table: dict) -> str:
    return content_identity(table)


def facts_table(world) -> dict:
    """The reference world as an atomic table, in the same form."""
    return {r: dict(t) for r, t in world.facts.items()}


def difference(base_digest: str, candidate_digest: str, t0: dict, t1: dict) -> tuple[str, set]:
    """(identity, changed (relation, input) keys) of the actual difference, by the session schema's definition:
    every (relation, input) whose outputs differ, sorted, hashed with both snapshot digests and table digests."""
    changes = [[r, x, t0[r][x], t1[r].get(x)] for r in sorted(t0) for x in sorted(t0[r]) if t0[r][x] != t1[r].get(x)]
    identity = content_identity({"base": base_digest, "candidate": candidate_digest, "base_table": table_digest(t0),
                                 "candidate_table": table_digest(t1), "changes": changes})
    return identity, {(r, x) for r, x, _, _ in changes}


def replay(tables: list, switch_calls: list, start: str, target: str, budget: int, step_relation: str = "connects_to",
           test_relation: str = "operator_of", max_calls: int | None = None) -> dict:
    """The bounded until-query, written independently of `recurrence.py`, over a sequence of tables:
    `tables[i]` serves from call `switch_calls[i]` on (`switch_calls[0]` is 0). Each loop is one call to `step`:
    apply the step relation (recorded even when it ends the run), stop at a dead end, a cycle, a missing output
    or the budget, and otherwise apply the test relation to the new airport. `max_calls` stops early: a paused
    prefix, with reason "". Returns the outcome, the applications, the calls made, and the application index at
    which each segment that was reached began."""
    current, visited, applications, calls, steps, seg, starts = start, [start], [], 0, 0, 0, [0]

    def out(reason, answer):
        return {"reason": reason, "answer": answer, "visited": list(visited), "applications": applications,
                "calls": calls, "segment_starts": list(starts)}

    while True:
        while seg + 1 < len(tables) and calls >= switch_calls[seg + 1]:
            seg += 1
            starts.append(len(applications))
        if max_calls is not None and calls >= max_calls:
            return out("", None)
        table = tables[seg]
        calls += 1
        if steps >= budget:
            return out("budget exhausted", None)
        nxt = table[step_relation].get(current)
        applications.append([step_relation, current, nxt])
        if nxt is None:
            return out("module gave no valid answer", None)
        if nxt == current:
            return out("dead end", None)
        if nxt in visited:
            return out("cycle", None)
        visited.append(nxt)
        current = nxt
        steps += 1
        tested = table[test_relation].get(nxt)
        applications.append([test_relation, nxt, tested])
        if tested is None:
            return out("module gave no valid answer", None)
        if tested == target:
            return out("found", nxt)


def traverse(table: dict, start: str, target: str, budget: int, step_relation: str = "connects_to",
             test_relation: str = "operator_of") -> dict:
    """The bounded query over one table."""
    return replay([table], [0], start, target, budget, step_relation, test_relation)


def _record_outcome(record) -> dict:
    st = record["state"]
    answer = st["answer"]["value"] if st.get("answer") else None
    reason = "found" if st["status"] == "answered" else st["reason"]
    apps = [[a["relation_id"], a["consumed"]["value"], None if a["produced"] is None else a["produced"]["value"]]
            for a in st["applications"]]
    return {"reason": reason, "answer": answer, "visited": list(st["visited"]), "applications": apps,
            "calls": record["calls"]}


def _same_run(a: dict, b: dict) -> bool:
    """The same execution: outcome, visited path, every application and the number of calls."""
    return all(a[k] == b[k] for k in ("reason", "answer", "visited", "applications", "calls"))


def _same_outcome(a: dict, b: dict) -> bool:
    """The same outcome, as the protocol defines it for naive continuation's failures."""
    return all(a[k] == b[k] for k in OUTCOME_KEYS)


def _result(**fields) -> dict:
    base = {"execution_valid": False, "execution_problems": [], "policy_valid": False, "policy_problems": [],
            "policy": None, "unjustified_switch": False, "unjustified_mixing": False, "equivalent_to": None,
            "contract_violation": None, "no_version_outcome": None, "correct": None}
    base.update(fields)
    base["ok"] = base["execution_valid"] and base["policy_valid"]
    base["problems"] = base["execution_problems"] + base["policy_problems"]
    return base


def check_session(record: dict, tables: dict, world=None, contexts: dict | None = None) -> dict:
    """Judge one finished session record (module docstring). `tables`: snapshot digest -> atomic table;
    `contexts`: context identity -> {"snapshot": digest, ...}; every segment's context must resolve in it."""
    if record.get("schema") != SCHEMA:
        return _result(execution_problems=["not a session record"])
    st, q, policy = record["state"], record["state"]["query"], record.get("policy")
    if st["status"] == "running":
        return _result(execution_problems=["the session has not finished"], policy=policy)
    budget, segments = record["budget"], record["segments"]
    if type(budget) is not int or not 1 <= budget <= 16 or st["reason"].startswith("rejected"):
        valid = (st["status"] == "abstained" and st["reason"].startswith("rejected") and not st["applications"]
                 and record["calls"] == 0)
        return _result(execution_valid=valid, policy_valid=valid and policy is None and len(segments) == 1,
                       execution_problems=[] if valid else ["an out-of-contract or rejected query was executed"],
                       policy=policy)
    exec_problems, policy_problems = [], []
    for i, seg in enumerate(segments):
        if seg["snapshot"] not in tables:
            exec_problems.append(f"segment {i}: no table for snapshot {seg['snapshot'][:12]}")
        if contexts is None or contexts.get(seg["context"], {}).get("snapshot") != seg["snapshot"]:
            exec_problems.append(f"segment {i}: its context does not resolve to an exported context on its snapshot")
    if segments[0]["from_call"] != 0 or segments[0]["from_application"] != 0:
        exec_problems.append("the first segment does not start at call 0")
    calls_at = [s["from_call"] for s in segments]
    if calls_at != sorted(calls_at):
        exec_problems.append("segment call indices are not in order")
    if any(seg["snapshot"] not in tables for seg in segments):
        return _result(execution_problems=exec_problems, policy=policy)
    got = _record_outcome(record)
    start, target = q["start"]["value"], q["target"]["value"]
    rel = (q["step_relation"], q["test_relation"])
    again = replay([tables[s["snapshot"]] for s in segments], calls_at, start, target, budget, *rel)
    if not _same_run(got, again):
        exec_problems.append("the record is not what replaying its segments gives")
    if again["segment_starts"] != [s["from_application"] for s in segments]:
        exec_problems.append("the segments' application boundaries are not where the replay switches")
    execution_valid = not exec_problems
    hows = [s["how"] for s in segments]
    m, rf = record.get("migration"), record.get("restarted_from")

    def check_restarted_from():
        """The paused prefix a restart discarded, replayed on the snapshot it ran on."""
        if not isinstance(rf, dict) or rf.get("snapshot") not in tables:
            policy_problems.append("a restart without a verifiable paused prefix")
            return None
        if contexts is None or contexts.get(rf.get("context"), {}).get("snapshot") != rf["snapshot"]:
            policy_problems.append("the restarted prefix's context does not resolve to its snapshot")
        prefix = replay([tables[rf["snapshot"]]], [0], start, target, budget, *rel, max_calls=rf.get("calls"))
        if prefix["reason"] != "" or prefix["calls"] != rf.get("calls") or \
                len(prefix["applications"]) != rf.get("applications"):
            policy_problems.append("restarted_from does not describe a paused prefix (calls or applications differ)")
        return prefix

    equivalent_to = None
    if policy in (None, "pin"):
        if hows != ["start"] or m is not None or rf is not None:
            policy_problems.append(f"a {policy or 'plain'} record must be one unchanged segment")
        equivalent_to = segments[0]["snapshot"]
    elif policy == "restart" or (policy == "migrate" and hows == ["start"]):
        if hows != ["start"]:
            policy_problems.append("a restart must be one fresh segment")
        prefix = check_restarted_from()
        if policy == "migrate":
            if not isinstance(m, dict) or m.get("migrated") is not False:
                policy_problems.append("a migration that restarted must say it did not migrate")
            elif prefix is not None:
                ident, changed = difference(rf["snapshot"], segments[0]["snapshot"], tables[rf["snapshot"]],
                                            tables[segments[0]["snapshot"]])
                if m.get("difference") != ident:
                    policy_problems.append("the migration's difference identity is not the re-derived one")
                blocking = sorted({(a, x) for a, x, _ in prefix["applications"]} & changed)
                if not blocking or m.get("blocked_by") != [list(k) for k in blocking]:
                    policy_problems.append("the migration was refused for the wrong reason (blocked_by differs)")
        elif m is not None:
            policy_problems.append("a restart carries a migration record")
        equivalent_to = segments[0]["snapshot"]
    elif policy == "migrate" and hows == ["start", "migrate"]:
        s0, s1 = segments[0]["snapshot"], segments[1]["snapshot"]
        t0, t1 = tables[s0], tables[s1]
        ident, changed = difference(s0, s1, t0, t1)
        prefix = got["applications"][:segments[1]["from_application"]]
        inputs = sorted({(a, x) for a, x, _ in prefix})
        expected = {"migrated": True, "difference": ident, "from": s0, "to": s1,
                    "prefix_inputs": [list(k) for k in inputs], "blocked_by": [], "revalidated": len(prefix)}
        bad = sorted(k for k, v in expected.items() if not isinstance(m, dict) or m.get(k) != v)
        if bad:
            policy_problems.append(f"the migration proof does not re-derive: {bad}")
        if set(inputs) & changed:
            policy_problems.append("migration: the prefix consumed an input whose output differs between the snapshots")
        if any(t1[a].get(x) != y for a, x, y in prefix):
            policy_problems.append("migration: the new snapshot does not reproduce the prefix")
        if rf is not None:
            policy_problems.append("a migration carries a restart record")
        if execution_valid and not _same_run(got, traverse(t1, start, target, budget, *rel)):
            policy_problems.append("migration: the record is not equivalent to the traversal over the new snapshot")
        equivalent_to = s1
    else:
        policy_problems.append(f"an unjustified switch (policy {policy!r}, segments {hows})")
    policy_valid = not policy_problems
    bounds = [s["from_application"] for s in segments] + [len(got["applications"])]
    suppliers = list(dict.fromkeys(seg["snapshot"] for seg, lo, hi in zip(segments, bounds, bounds[1:]) if hi > lo))
    switched = len(segments) > 1 and not policy_valid
    fields = dict(execution_valid=execution_valid, execution_problems=exec_problems, policy_valid=policy_valid,
                  policy_problems=policy_problems, policy=policy, unjustified_switch=switched,
                  unjustified_mixing=switched and len(suppliers) > 1,
                  equivalent_to=equivalent_to if (execution_valid and policy_valid) else None)
    if execution_valid:
        final = segments[-1]["snapshot"]
        run = lambda d: traverse(tables[d], start, target, budget, *rel)
        fields["contract_violation"] = not _same_outcome(got, run(final))
        fields["no_version_outcome"] = all(not _same_outcome(got, run(d)) for d in suppliers or [final])
        if world is not None:
            ref = traverse(facts_table(world), start, target, budget, *rel)
            fields["correct"] = (got["reason"], got["answer"]) == (ref["reason"], ref["answer"])
    return _result(**fields)
