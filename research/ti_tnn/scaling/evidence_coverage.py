"""Coverage of the database-evidence lookup on the 128 saved direct proposals (CPU; no model call; no new population).

**The question.** The residual-error audit (PR #72) labelled 17 of the direct proposal's executable-but-wrong cases
(11 questions) as schema or value grounding. Examples are a stored value written differently, or a near-miss column or
table. Would a deterministic lookup, which sees only the question, evidence, proposal, schema and snapshot, have
retrieved the database facts each repair needs?

**Two steps, in this order:**
1. `retrieve`: run `db_evidence.retrieve` on every saved initial proposal of the semantic-proposal-repair study's
   frozen run (64 questions x seeds 200 and 201). It reads the bundled requests, the task's question and evidence, the
   schema and the snapshot. It reads no score, reference SQL or audit label. The outputs are written to a run
   directory and copied into `evidence/evidence_coverage_v1/` under SHA256SUMS before any comparison.
2. `compare`: match the stored outputs against `coverage/requirements_v1.json`. That file names the facts each
   targeted repair needs, and was committed before step 1 ran. The comparison reports:
   - each targeted case's class: none, partial, complete or unresolved benchmark disagreement;
   - each fact's rank, provenance and whether it is shown;
   - for every group, the payload, latency, queries, scan limits and timeouts. The groups are the 17 targeted cases,
     their 11 questions, the other wrong proposals and the originally correct ones.

    python -m scaling.evidence_coverage retrieve
    python -m scaling.evidence_coverage tables        # the report's tables, from the stored comparison
    python -m scaling.evidence_coverage compare --audit <PR #72 checkout>/research/ti_tnn/scaling/evidence/residual_audit_v1/cases.json

An empty retrieval is not proof that a value is absent, and the lookup never substitutes a literal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

from . import contracts as C

SPR_BUNDLE = C.SCALING / "evidence" / "spr_v1"
BUNDLE = C.SCALING / "evidence" / "evidence_coverage_v1"
REQUIREMENTS = C.SCALING / "coverage" / "requirements_v1.json"
ROOT = C.RUNS / "evidence_coverage"
AUDIT_CASES_SHA256 = "134a98163d548330b95d54667ca890cd76c966037420d5cb780aaa0fc2a6bc09"   # PR #72 at 10d054a
GROUNDING = "schema/value grounding"


def _load(path: Path) -> list:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def _spr_run() -> Path:
    from . import evidence
    evidence._checked(SPR_BUNDLE)
    (run,) = sorted(SPR_BUNDLE.glob("run/*"))
    return run


# --- step 1 ----------------------------------------------------------------------------------------------------------
def retrieve_cmd(args) -> None:
    from . import data, db_evidence, evidence, identity, pilot_run
    from . import schema as schema_mod
    run = _spr_run()
    reqs = _load(run / "requests.jsonl")                 # proposals only: scores.jsonl is not opened here
    out_dir = pilot_run._new_dir(ROOT / "retrieval", "")
    identity.write_manifest(out_dir, config={
        "command": "evidence_coverage retrieve", "version": db_evidence.EVIDENCE_VERSION,
        "limits": asdict(db_evidence.Limits()),
        "source": "evidence/spr_v1/" + run.relative_to(SPR_BUNDLE).as_posix() + "/requests.jsonl",
        "source_sha256": identity.sha256_file(run / "requests.jsonl"),
        "inputs": "question, evidence, proposed SQL, schema, snapshot"}, require_clean=not args.allow_dirty)
    schemas, n = {}, 0
    with open(out_dir / "retrieval.jsonl", "w") as f:
        for r in reqs:
            task = data.task(r["task_id"])
            snap = data.snapshot(task.db_id)
            if task.db_id not in schemas:
                schemas[task.db_id] = schema_mod.load_schema(task.db_id, snap)
            sql = r["original"]["sql"]
            pkg = db_evidence.retrieve(task.question, task.evidence, sql, schemas[task.db_id], snap.path)
            f.write(json.dumps({"seed": r["seed"], "task_id": r["task_id"], "db_id": task.db_id,
                                "snapshot_sha256": snap.sha256,
                                "sql_sha256": hashlib.sha256((sql or "").encode()).hexdigest(),
                                "package": pkg}, sort_keys=True, default=str) + "\n")
            n += 1
    dest = BUNDLE / "retrieval" / out_dir.name
    dest.mkdir(parents=True)
    for name in ("retrieval.jsonl", "manifest.json"):
        shutil.copy2(out_dir / name, dest / name)
    evidence.write_sums(BUNDLE)
    print(json.dumps({"proposals": n, "run": identity.run_ref(out_dir), "bundle": str(dest.relative_to(C.REPO))}))


# --- step 2 ----------------------------------------------------------------------------------------------------------
def _shown(pkg: dict) -> list:
    return [it for i, it in enumerate(pkg["items"]) if i in set(pkg["rendered_items"])]


def match_fact(fact: dict, pkg: dict) -> dict:
    """{found, shown, rank, item}: whether the package retrieves the fact, and whether a model would read it."""
    best = None
    for i, it in enumerate(pkg["items"]):
        hit = None
        k = fact["kind"]
        if k == "value_candidate" and it["kind"] == "value" and it["subject"]["literal"] == fact["literal"] and \
                (it["subject"]["table"], it["subject"]["column"]) == (fact["table"], fact["column"]):
            hit = next((c["rank"] for c in it["candidates"] if c["value"] == fact["value"]), None)
        elif k == "elsewhere_candidate" and it["kind"] == "value_elsewhere" and \
                it["subject"]["literal"] == fact["literal"]:
            hit = next((c["rank"] for c in it["candidates"] if c["value"] == fact["value"] and
                        (c["provenance"]["table"], c["provenance"]["column"]) == (fact["table"], fact["column"])), None)
        elif k == "column_candidate" and it["kind"] == "column":
            hit = next((c["rank"] for c in it["candidates"] if c["column"] == [fact["table"], fact["column"]]), None)
        elif k == "table_candidate" and it["kind"] == "table":
            hit = next((c["rank"] for c in it["candidates"] if c["table"] == fact["table"]), None)
        elif k == "join_flag" and it["kind"] == "relationship":
            want = {tuple(fact["left"]), tuple(fact["right"])}
            hit = next((1 for j in it["joins"] if {tuple(j["left"]), tuple(j["right"])} == want and
                        (j["status"] == "not declared" or j["note"])), None)
        elif k == "representation_flag" and it["kind"] == "representation" and \
                (it["subject"]["table"], it["subject"]["column"]) == (fact["table"], fact["column"]):
            hit = next((c["rank"] for c in it["candidates"] if "not a plain number" in c["note"]), None)
        if hit is not None:
            cand = {"found": True, "shown": i in pkg["rendered_items"], "rank": hit, "item": i,
                    "item_kind": it["kind"], "subject": it["subject"]}
            if best is None or (cand["shown"], -cand["rank"]) > (best["shown"], -best["rank"]):
                best = cand
    return best or {"found": False, "shown": False, "rank": None, "item": None}


def classify(required: list, matches: list, unresolved: bool) -> str:
    if unresolved:
        return "unresolved benchmark disagreement"
    shown = sum(1 for m in matches if m["shown"])
    return "complete" if shown == len(required) else "partial" if shown else "none"


def _stats(values: list) -> dict:
    values = sorted(values)
    if not values:
        return {}
    return {"median": statistics.median(values), "p90": values[min(len(values) - 1, int(0.9 * len(values)))],
            "max": values[-1], "min": values[0]}


def signals(pkg: dict) -> dict:
    """What a revision prompt would be told that could prompt a change, counted on the shown items."""
    shown = _shown(pkg)
    return {"value_not_stored_with_candidates": sum(1 for it in shown if it["kind"] == "value" and
                                                    it["verbatim_rows"] == 0 and it["candidates"]),
            "value_found_elsewhere": sum(1 for it in shown if it["kind"] == "value_elsewhere" and it["candidates"]),
            "join_flags": sum(len(it["candidates"]) for it in shown if it["kind"] == "relationship"),
            "representation_flags": sum(len(it["candidates"]) for it in shown if it["kind"] == "representation")}


def group_report(rows: list) -> dict:
    pk = [r["package"] for r in rows]
    sig = [signals(p) for p in pk]
    return {"proposals": len(rows),
            "payload_chars": _stats([p["payload_chars"] for p in pk]),
            "payload_chars_untruncated": _stats([p["payload_chars_untruncated"] for p in pk]),
            "truncated": sum(p["truncated"] for p in pk),
            "latency_seconds": _stats([p["latency_seconds"] for p in pk]),
            "queries": _stats([p["queries"] for p in pk]),
            "instructions": _stats([p["instructions"] for p in pk]),
            "with_timeouts": sum(1 for p in pk if p["timeouts"]),
            "timeouts": sum(len(p["timeouts"]) for p in pk),
            "with_scan_limits": sum(1 for p in pk if p["scan_limits"]),
            "with_errors": sum(1 for p in pk if p["errors"]),
            "wall_safety_stops": sum(p["wall_safety_stop"] for p in pk),
            "unparsed": sum(1 for p in pk if not p["proposal"]["parsed"]),
            "with_any_signal": sum(1 for s in sig if any(s.values())),
            "signals": {k: sum(1 for s in sig if s[k]) for k in sig[0]} if sig else {}}


def compare(retrieval: list, scores: list, audit: list, spec: dict) -> dict:
    key = lambda x: (x["task_id"], x["seed"])   # noqa: E731
    pos = {key(r): r for r in retrieval}
    d0 = {key(x): x["score"]["correct"] for x in scores if x["policy"] == "D0"}
    if set(pos) != set(d0) or len(pos) != 128:
        raise SystemExit("the retrieval and the D0 scores do not cover the same 128 proposals")
    grounding = {key(c) for c in audit if GROUNDING in [lab.rstrip("?") for lab in c["labels"]]}
    targeted = {key(c): c for c in spec["cases"]}
    if set(targeted) != grounding:
        raise SystemExit(f"the requirements cover {sorted(set(targeted) ^ grounding)} differently from the audit")
    cases = []
    for k in sorted(targeted, key=lambda k: (k[1], k[0])):
        c, pkg = targeted[k], pos[k]["package"]
        matches = [match_fact(f, pkg) for f in c["required"]]
        unresolved = bool(c["benchmark"] and c["benchmark"]["unresolved"])
        cases.append({"task_id": k[0], "seed": k[1], "db_id": pos[k]["db_id"],
                      "class": classify(c["required"], matches, unresolved),
                      "facts_shown": f"{sum(m['shown'] for m in matches)}/{len(matches)}",
                      "facts": [{"fact": f, **m} for f, m in zip(c["required"], matches)],
                      "other_changes": c["other_changes"], "benchmark": c["benchmark"],
                      "note": c.get("note"), "audit_correction": c.get("audit_correction"),
                      "payload_chars": pkg["payload_chars"], "latency_seconds": pkg["latency_seconds"],
                      "timeouts": pkg["timeouts"], "scan_limits": pkg["scan_limits"]})
    by_q = {}
    for c in cases:
        by_q.setdefault(c["task_id"], {})[str(c["seed"])] = c["class"]
    rest_wrong = [pos[k] for k in sorted(pos) if not d0[k] and k not in targeted]
    correct = [pos[k] for k in sorted(pos) if d0[k]]
    from collections import Counter
    return {"cases": cases, "class_counts": dict(Counter(c["class"] for c in cases)),
            "questions": by_q,
            "question_class_counts": dict(Counter("/".join(sorted(set(v.values()))) for v in by_q.values())),
            "groups": {"targeted": group_report([pos[k] for k in targeted]),
                       "other_wrong": group_report(rest_wrong), "correct": group_report(correct),
                       "all": group_report(list(pos.values()))}}


def compare_cmd(args) -> None:
    from . import evidence, identity, pilot_run
    evidence._checked(BUNDLE)
    (ret_dir,) = sorted(BUNDLE.glob("retrieval/*"))
    retrieval = _load(ret_dir / "retrieval.jsonl")
    audit_path = Path(args.audit)
    if identity.sha256_file(audit_path) != AUDIT_CASES_SHA256:
        raise SystemExit("the audit cases are not the pinned version")
    audit = json.loads(audit_path.read_text())
    spec = json.loads(REQUIREMENTS.read_text())
    scores = _load(_spr_run() / "scores.jsonl")
    out = compare(retrieval, scores, audit, spec)
    out["inputs"] = {"retrieval": "evidence/evidence_coverage_v1/" + ret_dir.relative_to(BUNDLE).as_posix(),
                     "retrieval_sha256": identity.sha256_file(ret_dir / "retrieval.jsonl"),
                     "requirements_sha256": identity.sha256_file(REQUIREMENTS), "audit_sha256": AUDIT_CASES_SHA256}
    out_dir = pilot_run._new_dir(ROOT / "compare", "")
    identity.write_manifest(out_dir, config={"command": "evidence_coverage compare", **out["inputs"]},
                            require_clean=not args.allow_dirty)
    (out_dir / "coverage.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    dest = BUNDLE / "compare" / out_dir.name
    dest.mkdir(parents=True)
    for name in ("coverage.json", "manifest.json"):
        shutil.copy2(out_dir / name, dest / name)
    evidence.write_sums(BUNDLE)
    print(json.dumps({"class_counts": out["class_counts"], "question_class_counts": out["question_class_counts"],
                      "bundle": str(dest.relative_to(C.REPO))}, indent=1))


def _fact_label(f: dict) -> str:
    k = f["kind"]
    if k in ("value_candidate", "elsewhere_candidate"):
        return f"{f['literal']!r} -> {f['table']}.{f['column']} {f['value']!r}"
    if k == "column_candidate":
        return f"column {f['table']}.{f['column']}"
    if k == "table_candidate":
        return f"table {f['table']}"
    if k == "join_flag":
        return f"join {'.'.join(f['left'])} = {'.'.join(f['right'])} flagged"
    if k == "representation_flag":
        return f"{f['table']}.{f['column']} text format flagged"
    return f"(no lookup kind) {f['what']}"


def tables(cov: dict) -> str:
    """The report's tables, from a stored coverage.json."""
    out = ["| case | seed | class | facts shown | each fact: retrieved? rank (item) | payload chars | latency s |",
           "|---|---|---|---|---|---|---|"]
    for c in cov["cases"]:
        facts = "; ".join(f"{_fact_label(f['fact'])}: " + (f"yes, rank {f['rank']} ({f['item_kind']})"
                                                             if f["shown"] else "no") for f in c["facts"])
        out.append(f"| {c['task_id']} | {c['seed']} | {c['class']} | {c['facts_shown']} | {facts} | "
                   f"{c['payload_chars']} | {c['latency_seconds']:.3f} |")
    out += ["", "| group | proposals | payload chars (median / p90 / max) | latency s (median / p90 / max) | "
            "queries (median / max) | VM instructions (max) | scan limits hit | timeouts | any signal |",
            "|---|---|---|---|---|---|---|---|---|"]
    names = {"targeted": "17 grounding-labelled", "other_wrong": "other wrong", "correct": "originally correct",
             "all": "all"}
    for g, r in cov["groups"].items():
        pc, la, q = r["payload_chars"], r["latency_seconds"], r["queries"]
        out.append(f"| {names[g]} | {r['proposals']} | {pc['median']:.0f} / {pc['p90']} / {pc['max']} | "
                   f"{la['median']:.3f} / {la['p90']:.3f} / {la['max']:.3f} | {q['median']:.0f} / {q['max']} | "
                   f"{r['instructions']['max']:,} | {r['with_scan_limits']} | {r['timeouts']} | "
                   f"{r['with_any_signal']} |")
    out += ["", "| group | value not stored, candidates shown | literal found elsewhere | join flagged | "
            "representation flagged |", "|---|---|---|---|---|"]
    for g, r in cov["groups"].items():
        s = r["signals"]
        out.append(f"| {names[g]} | {s['value_not_stored_with_candidates']} | {s['value_found_elsewhere']} | "
                   f"{s['join_flags']} | {s['representation_flags']} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieve")
    r.add_argument("--allow-dirty", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--audit", required=True)
    c.add_argument("--allow-dirty", action="store_true")
    sub.add_parser("tables")
    args = ap.parse_args(argv)
    if args.cmd == "tables":
        from . import evidence
        evidence._checked(BUNDLE)
        (path,) = sorted(BUNDLE.glob("compare/*/coverage.json"))
        print(tables(json.loads(path.read_text())))
        return 0
    {"retrieve": retrieve_cmd, "compare": compare_cmd}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
