"""Independent verifier for an integration-test run (protocols/integration_test_v1.md: "Analysis and independent
verification").

    .venv-phase2/bin/python research/ti_tnn/bench/integration_test_verify.py <run dir> [--development]

**It trusts no stored summary.** From the run's export it rebuilds:
- every reported number;
- every construction check;
- every learned verdict.

**What it never calls.** It does not call the producer's execution or decision code: `recurrence.py`,
`sessions.py`, `serving.py`, `release.py`, `update_checks.py` or `query_interpreter.py`. It uses only:
- `integration_test`'s **study generator and branch draw**, which regenerate the questions and updates from
  the world seeds. As in the robustness verifier, the generator defines what was asked;
- `session_verification`, the **independent reconstruction** of session records (its own traversal);
- the robustness verifier's **independent fixed-sequence certificate**.

**What it re-derives:**

| From | It re-derives |
|---|---|
| **Interpreter records** | every score, from its recorded components (the float32 product of the decoder-step probabilities, times both copy probabilities); the policy's pre-threshold decision (binding, types, budget contract, outcome); every per-context result, by its own traversal over that context's exported table; every correctness label, against the reference world; every certificate, recomputed from those labels; every admission and release decision, under the admitting threshold or max(t0, t1); every metric |
| **Exhaustive sweep** | the producer's outcome digest, and its compact rows, by its own traversal over each context's table (construction); correctness against the reference world (learned) |
| **Episodes** | the set of queries and pause indices, from the tables and the declared samplers; every session record, through `session_verification` (execution, policy and semantic judgments, only valid executions counted); every interpreted release decision |
| **Differences and contexts** | the actual difference of every branch, and its classes; context and certificate bindings |
| **Witness** | rebuilt from the world's facts |

**What it requires before and around the re-derivation** (added after the review of #58):
- **The experiment.** In frozen mode, the run must be the frozen design: worlds 800–802 × inits 0–4, budgets
  1–16, the recipe and the study constants. This is checked before anything is verified. No mode accepts an
  empty, repeated or incomplete inventory.
- **Session binding.** Every embedded session must belong to its declared episode: question, budget,
  relations, policy, snapshots and contexts, and the pause index wherever the record carries one. An
  interpreted session must also belong to its item's proposed query and release decision.
- **Certificates and identities.** Every certificate carries the expected alpha and delta (in frozen mode,
  0.01 and 0.05). Expectation and context identities are recomputed from their parts, and draw metadata
  matches the regenerated draws.

**What it cannot re-derive** from weight-free records, and reports as the producer's own record:
- the update gates;
- the activation and rollback mechanics (C4); their recorded identities are checked for consistency;
- the injected faults;
- that each exported table is its snapshot's behaviour. That is checked at run time, and by the deterministic
  fixtures in `tests/test_ti_tnn_sessions.py`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import integration_test as IT  # noqa: E402  (the generator and the declared samplers only)
import interpreter_robustness_verify as RV  # noqa: E402  (the independent fixed-sequence certificate)
from integrated import calibration as K  # noqa: E402  (the binomial bounds only)
from integrated import relations as W  # noqa: E402
from integrated import session_verification as SV  # noqa: E402
from integrated.contracts import content_identity  # noqa: E402

RESULT_REASONS = ("found", "dead end", "cycle", "budget exhausted")
COVERAGE, ERROR_BOUND, REFUSAL = 0.90, 0.01, 0.95
# The producer files as hashed into every run (integration_test.SOURCES), fixed in the freezing commit of
# protocols/integration_test_v1.md revision 4. The run records its commit; a commit cannot name itself.
FROZEN_SOURCES = {
    "integration_test.py": "b9482de81ff982db89bc3a074cbd5bfb87b11d898c1c380a09be6d4d36cdc138",
    "../integrated/sessions.py": "c355f59abd5a250d362e45c0540af5d0c097c06cae7c0d0855ba5e74d9c413b8",
    "../integrated/session_verification.py": "9cbb118601a17231e2504d7926da1da86db7eae1298ad00fd88bd68fcfb3fc96",
    "../integrated/serving.py": "c31ef561c55d7d31c163a522ba3bd456054f804bfc420cacac374a736e2248a5",
    "../integrated/certificates.py": "a5f2abe037f4c9808633764d3e4df5c8a492a8d437bd01ef22c2980a39e32513",
    "../integrated/update_checks.py": "3f26b74e352b9c9fd0005893d610fbfa961825c1a863c7bfdd27c11204361592",
    "../integrated/query_interpreter.py": "18f759b207bad575fc2d924c5861553ccbf1d01a562c84157db56d36ae360194",
    "../integrated/recurrence.py": "a9f4809e822b5b3a79aebdd0c5dfb8e85189571c67b07fbb707de99fb8cf2fdc",
    "../integrated/modules.py": "fe89f275281579b666f12c5d452d5911c6448b68ae4bf6cc74c19e38647622c7",
    "../integrated/release.py": "18ae8669793a7569a303805e688022fd84eba62661cd598544ef619837fd9a11",
}


class VerificationError(AssertionError):
    pass


# The frozen design (protocols/integration_test_v1.md, revision 4). Frozen mode requires the run to be exactly
# this experiment: the declared worlds, inits and configuration, and the generator's study constants.
FROZEN_DESIGN = {"worlds": [800, 801, 802], "inits": [0, 1, 2, 3, 4], "countries": 48, "interpreter_updates": 1500,
                 "d": 32, "hidden": 64, "steps": 1500, "lr": 1e-2, "update_steps": 300, "update_lr": 3e-3,
                 "exhaustive_budgets": list(range(1, 17))}
FROZEN_STUDY = {"SIZES": {"calibration": 1000, "test": 1000, "default_diagnostic": 300},
                "BRANCH_KINDS": {"operator": 4, "route": 4, "two_fact": 4}, "EPISODE_UNAFFECTED": 200,
                "INTERPRETED_UNAFFECTED": 50,
                "GRID": (0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0),
                "POLICY_VERSION": "ti_tnn_query_policy_v1", "EVENT_VERSION": "ti_tnn_integration_event_v1"}
FROZEN_RISK = {"alpha": 0.01, "delta": 0.05}


def inventory_problems(run: dict, development: bool) -> list[str]:
    """The run must be the declared experiment, checked before any unit is verified. Frozen mode requires the
    frozen design exactly; development mode requires a non-empty, complete, duplicate-free worlds x inits
    inventory as the run declares it."""
    problems, cfg = [], run["config"]
    pairs = [(u["world"], u["init"]) for u in run["units"]]
    if not run["worlds"] or not pairs:
        problems.append("the run has no worlds or no units")
    if len(set(pairs)) != len(pairs):
        problems.append("the run repeats a unit")
    if sorted(pairs) != sorted((int(w), i) for w in run["worlds"] for i in cfg["inits"]):
        problems.append("the unit inventory is not the declared worlds x inits")
    if sorted(int(w) for w in run["worlds"]) != sorted(cfg["worlds"]):
        problems.append("the worlds run are not the declared worlds")
    if not development:
        for key, value in FROZEN_DESIGN.items():
            if cfg.get(key) != value:
                problems.append(f"config {key} is {cfg.get(key)!r}, frozen {value!r}")
        if sorted(int(w) for w in run["worlds"]) != FROZEN_DESIGN["worlds"] or sorted(pairs) != sorted(
                (w, i) for w in FROZEN_DESIGN["worlds"] for i in FROZEN_DESIGN["inits"]):
            problems.append("the run is not the frozen inventory: worlds 800-802 x inits 0-4, 15 units")
        for name, value in FROZEN_STUDY.items():
            if getattr(IT, name) != value:
                problems.append(f"the generator's {name} differs from the frozen design")
    return problems


def _load(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _lines(path: Path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


# --- the policy and the release rule, re-derived ------------------------------------------------------------------
def decision(parse: dict | None, types) -> dict:
    """The pre-threshold decision from a parse's fields: the policy's binding, type and contract checks."""
    if parse is None:
        return {"execute": False, "reason": "unsupported characters", "query": None}
    if parse["outcome"] == "unsupported":
        return {"execute": False, "reason": "unsupported", "query": None}
    if parse["outcome"] == "over_budget":
        return {"execute": False, "reason": "invalid query: budget outside 1-16", "query": None}
    if parse["outcome"] != "query" or parse["relations"] != ["connects_to", "operator_of"]:
        return {"execute": False, "reason": "malformed", "query": None}
    kinds = []
    for token in (parse["start"], parse["target"]):
        hit = [t for t in sorted(types) if token in types[t].values]
        if not hit:
            return {"execute": False, "reason": "unknown entity", "query": None}
        kinds.append(hit[0])
    budget = parse["budget"]
    if kinds != ["Airport", "Operator"] or type(budget) is not int or not 1 <= budget <= 16:
        return {"execute": False, "reason": "invalid query", "query": None}
    return {"execute": True, "reason": "", "query": [parse["start"], parse["target"], budget]}


def score_of(row) -> float | None:
    """The score from its recorded components, computed as the interpreter computes it: the float32 product of the
    chosen decoder-step probabilities, times both copy probabilities."""
    import torch
    if row["parse"] is None:
        return 0.0
    probs = torch.tensor(row["step_probabilities"])
    return float(probs.prod()) * row["start_probability"] * row["target_probability"]


def result_of(table, query) -> list:
    out = SV.traverse(table, *query)
    kind = "result" if out["reason"] in RESULT_REASONS else "refused"
    return [kind, out["answer"], out["reason"]]


def release_rule(admitting: dict, releasing: dict, same: bool) -> tuple:
    """(threshold, supporting step) for release; (None, None) when there is no rule."""
    if same:
        t = admitting["threshold"]
        step = next((s for s in admitting["steps"] if s[0] == t and s[4]), None)
        return t, step
    keys = ("interpreter_identity", "policy_version", "event_version", "distribution_identity", "grid", "alpha", "delta")
    if any(admitting[k] != releasing[k] for k in keys) or not (admitting["certified"] and releasing["certified"]):
        return None, None
    t = max(admitting["threshold"], releasing["threshold"])
    step = next((s for s in releasing["steps"] if s[0] == t and s[4]), None)
    return (t, step) if step is not None else (None, None)


def session_binding_problems(line: dict, rec: dict, c0: str, s0: str, branch: dict, row: dict | None,
                             certs: dict) -> list[str]:
    """The embedded session must be the declared episode's own: its question, budget, relations and policy; its
    snapshots and contexts (C0, then this branch's C1); and its pause index, wherever the record carries one (a
    pin record finishes the uninterrupted C0 run whatever the pause index, so its content does not depend on it).
    For an interpreted episode, the question must be that item's proposed query, and the release decision must
    name the admitting and releasing contexts and certificates it was made under."""
    out = []
    s, t, budget = line["query"]
    q = rec.get("state", {}).get("query", {})
    if (q.get("start", {}).get("value"), q.get("target", {}).get("value"), q.get("step_relation"),
            q.get("test_relation"), rec.get("budget")) != (s, t, "connects_to", "operator_of", budget):
        out.append("the embedded session asks another question")
    if rec.get("policy") != line["policy"]:
        out.append("the embedded session ran another policy")
    c1, s1, k = branch["context"], branch["snapshot"], line["k"]
    seg = [(x.get("snapshot"), x.get("context"), x.get("how")) for x in rec.get("segments", [])]
    rf = rec.get("restarted_from")
    fresh_on_c1 = seg == [(s1, c1, "start")] and isinstance(rf, dict) and (rf.get("snapshot"), rf.get("context"),
                                                                          rf.get("calls")) == (s0, c0, k)
    switched = (len(seg) == 2 and seg[0] == (s0, c0, "start") and seg[1][:2] == (s1, c1)
                and rec["segments"][1].get("from_call") == k)
    m = rec.get("migration") or {}
    expected = {"pin": seg == [(s0, c0, "start")] and rf is None,
                "restart": fresh_on_c1,
                "migrate": (switched and seg[1][2] == "migrate") if m.get("migrated") else fresh_on_c1,
                "naive": switched and seg[1][2] == "unchecked"}.get(line["policy"], False)
    if not expected:
        out.append("the embedded session's snapshots, contexts or pause index are not this episode's")
    if row is not None:
        if row["decision"].get("query") != [s, t, budget]:
            out.append("the session is not this item's proposed query")
        rel = line.get("release")
        if line["policy"] == "naive":
            if rel is not None:
                out.append("naive continuation carries a release decision")
        elif rel is None or (rel.get("admitting_context"), rel.get("admitting_certificate"), rel.get("releasing_context"),
                             rel.get("releasing_certificate"), rel.get("score")) != (
                c0, content_identity(certs[c0]), seg[-1][1] if seg else None,
                content_identity(certs[seg[-1][1]]) if seg and seg[-1][1] in certs else None, row["score"]):
            out.append("the release decision does not name this item's score, contexts and certificates")
    elif line.get("item") is not None or line.get("release") is not None:
        out.append("a supplied episode carries an item or a release decision")
    return out


def certificate_problems(cert: dict, records: list, expected: dict, snapshot: str) -> list[str]:
    """A certificate re-derived independently from its re-labelled records, and bound to its context."""
    problems = []
    if [list(r) for r in cert["records"]] != [list(r) for r in records]:
        problems.append("its records are not the re-derived eligible calibration records")
    again = RV.certificate([r[1] for r in records], [r[2] for r in records], tuple(cert["grid"]), cert["alpha"],
                           cert["delta"])
    if [list(s) for s in again["steps"]] != [list(s) for s in cert["steps"]] or \
            (again["threshold"], again["certified"], again["stop_reason"]) != (cert["threshold"], cert["certified"],
                                                                               cert["stop_reason"]):
        problems.append("its fixed-sequence result does not re-derive")
    for k in ("interpreter_identity", "policy_version", "event_version", "distribution_identity", "alpha", "delta"):
        if cert[k] != expected[k]:
            problems.append(f"bound to another {k}")
    if list(cert["grid"]) != list(expected["grid"]) or cert["snapshot_digest"] != snapshot:
        problems.append("bound to another grid or snapshot")
    return problems


# --- one unit ------------------------------------------------------------------------------------------------------
def verify_unit(run_dir: Path, info: dict, world, study, branches, problems: list, budgets,
                development: bool = False) -> dict:
    d = run_dir / info["path"]
    unit = json.loads((d / "unit.json").read_text())
    tag = f"w{info['world']} i{info['init']}"
    rows, tables = _load(d / "interpreter.json.gz"), _load(d / "tables.json.gz")
    certs, contexts = _load(d / "certificates.json.gz"), _load(d / "contexts.json.gz")
    exh = _load(d / "exhaustive.json.gz")
    types = world.types
    items = {x["id"]: x for x in study["items"]}
    draws = [x for x in study["items"] if x["split"] in ("calibration", "test", "diagnostic")]
    expected = unit["expectation"]

    def fail(msg):
        problems.append(f"{tag}: {msg}")

    if [r["id"] for r in rows] != [x["id"] for x in draws]:
        fail("the interpreter records are not exactly the study's draws")
    elif any((r["split"], r["category"]) != (items[r["id"]]["split"], items[r["id"]]["category"]) for r in rows):
        fail("an interpreter record's split or category is not its regenerated draw's")
    if (unit["world"], unit["init"]) != (info["world"], info["init"]):
        fail("unit.json is not the unit the run lists")
    if len(unit["branches"]) != sum(IT.BRANCH_KINDS.values()):
        fail("the unit does not hold every declared branch")
    exp_fields = ("interpreter_identity", "policy_version", "event_version", "distribution_identity", "alpha", "delta")
    if content_identity({**{k: expected[k] for k in exp_fields}, "grid": list(expected["grid"])}) != expected["identity"]:
        fail("the expectation's identity does not re-derive")
    if expected["distribution_identity"] != study["distribution_identity"] or \
            (expected["policy_version"], expected["event_version"], tuple(expected["grid"])) != (
            IT.POLICY_VERSION, IT.EVENT_VERSION, tuple(IT.GRID)):
        fail("the expectation is not the study's distribution, policy, event and grid")
    if not development and (expected["alpha"], expected["delta"]) != (FROZEN_RISK["alpha"], FROZEN_RISK["delta"]):
        fail("the expectation's risk settings are not the frozen alpha and delta")
    for ctx_id, ctx in contexts.items():
        cert = certs.get(ctx_id) and certs[ctx_id]
        if cert is None or content_identity(cert) != ctx["certificate"] or ctx["snapshot"] not in tables:
            fail(f"context {ctx_id[:12]}: certificate or table missing or not its own")
    c0 = unit["c0"]["context"]
    worlds = {c0: world}
    for b in unit["branches"]:
        worlds[b["context"]] = IT.edited(world, f"branch{b['index']}", b["changes"])
    for ctx_id, ctx in contexts.items():
        if content_identity({"snapshot": ctx["snapshot"], "certificate": ctx["certificate"],
                             "expectation": ctx["expectation"], "world": ctx["world"]}) != ctx_id:
            fail(f"context {ctx_id[:12]}: its identity does not re-derive from its parts")
        if ctx["expectation"] != expected["identity"] or ctx_id not in worlds or \
                ctx["world"] != worlds[ctx_id].identity:
            fail(f"context {ctx_id[:12]}: not bound to this unit's expectation and its world")
    if contexts.get(c0, {}).get("snapshot") != unit["c0"]["s0_digest"] or any(
            contexts.get(b["context"], {}).get("snapshot") != b["snapshot"] for b in unit["branches"]):
        fail("a context does not hold the snapshot its unit record names")
    exported = {k: {"snapshot": v["snapshot"]} for k, v in contexts.items()}
    # interpreter records: decisions, results, labels, certificates, releases
    parse_exact = Counter()
    for r in rows:
        x = items[r["id"]]
        dec = decision(r["parse"], types)
        if score_of(r) != r["score"]:
            fail(f"{r['id']}: its score is not the product of its recorded components")
        if dec != r["decision"]:
            fail(f"{r['id']}: the recorded pre-threshold decision does not re-derive")
        if x["expected"] == "result":
            p = r["parse"]
            parse_exact[(x["split"], p is not None and p["outcome"] == "query"
                         and [p["start"], p["target"], p["budget"], p["default"]] == [x["start"], x["target"], x["budget"],
                                                                                    x["default"]])] += 1
        for ctx_id, res in r["results"].items():
            if not dec["execute"] or ctx_id not in contexts:
                fail(f"{r['id']}: a result for a draw that is not executed, or for an unknown context")
                continue
            if res != result_of(tables[contexts[ctx_id]["snapshot"]], dec["query"]):
                fail(f"{r['id']}: its result under {ctx_id[:12]} does not re-derive from that context's table")
            ref = SV.traverse(SV.facts_table(worlds[ctx_id]), x["start"], x["target"], x["budget"]) \
                if x["expected"] == "result" else None
            label = res[0] == "result" and ref is not None and (res[1], res[2]) == (ref["answer"], ref["reason"])
            if r["correct"].get(ctx_id) != label:
                fail(f"{r['id']}: its correctness label under {ctx_id[:12]} does not re-derive")
    for ctx_id, cert in certs.items():
        eligible = [[r["id"], r["score"], r["correct"][ctx_id]] for r in rows
                    if r["split"] == "calibration" and r["results"].get(ctx_id, [None])[0] == "result"]
        for msg in certificate_problems(cert, eligible, expected, contexts[ctx_id]["snapshot"]):
            fail(f"certificate {ctx_id[:12]}: {msg}")
    for r in rows:
        if r["split"] == "calibration":
            continue
        for fin, fields in r["release"].items():
            for field, rel in fields.items():
                admitting = c0 if field == "resumed" else fin
                admitted = certs[admitting]["certified"] and r["score"] >= certs[admitting]["threshold"]
                t, step = release_rule(certs[admitting], certs[fin], admitting == fin)
                res = r["results"].get(fin)
                want = admitted and res is not None and res[0] == "result" and t is not None and r["score"] >= t
                if (rel["admitted"], rel["released"]) != (admitted, want) or (want and rel["threshold"] != t):
                    fail(f"{r['id']}: its {field} release under {fin[:12]} does not re-derive")
    # metrics
    def metrics(key, field):
        test = [r for r in rows if r["split"] == "test"]
        rel = lambda r: r["release"].get(key, {}).get(field, {}).get("released", False)
        sup = [r for r in test if items[r["id"]]["expected"] == "result"]
        released = [r for r in test if rel(r)]
        errors = sum(not r["correct"].get(key, False) for r in released)
        refusal = {k: (sum(not rel(r) for r in test if r["category"] == k), sum(r["category"] == k for r in test))
                   for k in IT.REFUSAL_KINDS}
        return {"coverage": sum(rel(r) for r in sup) / max(1, len(sup)), "released": len(released), "errors": errors,
                "error_upper_95": K.clopper_pearson_upper(errors, len(released)),
                "refusal": {k: {"abstained": a, "n": n, "rate": a / n if n else None} for k, (a, n) in refusal.items()}}
    # exhaustive sweeps (construction: the producer's outcomes equal the traversal over its own table)
    exhaustive = {}
    for ctx_id, e in exh.items():
        table, wld = tables[contexts[ctx_id]["snapshot"]], worlds[ctx_id]
        facts, full, rows_ = SV.facts_table(wld), [], []
        wrong = 0
        for s in sorted(types["Airport"].values):
            for t in sorted(types["Operator"].values):
                for b in budgets:
                    o = SV.traverse(table, s, t, b)
                    full.append([o["reason"], o["answer"], o["visited"], o["applications"], o["calls"]])
                    rows_.append([o["reason"], o["answer"], o["calls"]])
                    ref = SV.traverse(facts, s, t, b)
                    wrong += (o["reason"], o["answer"]) != (ref["reason"], ref["answer"])
        exhaustive[ctx_id] = {"construction_ok": content_identity(full) == e["digest"] and rows_ == e["rows"],
                              "reference_mismatch": wrong}
    # branches, differences, and the declared episode sets
    t0 = tables[contexts[c0]["snapshot"]]
    rows_by_id = {r["id"]: r for r in rows}
    branch_verdicts, expected_keys = [], {}
    for b, declared in zip(unit["branches"], branches):
        if (b["index"], b["kind"], b["changes"]) != (declared["index"], declared["kind"], declared["changes"]):
            fail(f"branch {b['index']}: not the declared draw")
        w1 = worlds[b["context"]]
        t1 = tables.get(b["snapshot"]) if b["context"] in contexts else None
        if t1 is None:
            fail(f"branch {b['index']}: no exported table")
            continue
        ident, changed = SV.difference(contexts[c0]["snapshot"], b["snapshot"], t0, t1)
        if ident != b["difference"]["identity"]:
            fail(f"branch {b['index']}: the actual difference does not re-derive")
        declared_keys = {(r_, x_) for r_, x_, _ in declared["changes"]}
        corrections = sorted(k for k in changed - declared_keys if t1[k[0]][k[1]] == w1.facts[k[0]][k[1]]
                             and t0[k[0]][k[1]] != world.facts[k[0]][k[1]])
        regressions = sorted(changed - declared_keys - set(corrections))
        classes = {k: len(v) for k, v in b["difference"]["classes"].items()}
        if classes != {"intended": len(changed & declared_keys), "correction": len(corrections),
                       "regression": len(regressions)}:
            fail(f"branch {b['index']}: the difference classes do not re-derive")
        bv = {"index": b["index"], "activated": b["activated"], "gates_pass": all(v == "pass" for v in b["gates"].values()),
              "certified": certs[b["context"]]["certified"], "corrections": len(corrections),
              "regressions": len(regressions),
              "s1_atomic": all(t1[r_][x_] == w1.facts[r_][x_] for r_ in t1 for x_ in t1[r_])}
        branch_verdicts.append(bv)
        if not b["activated"]:
            continue
        c4 = b["c4"]
        bv["c4"] = (c4["active_after_activation"] == b["context"] and c4["installed_complete"] and c4["rolled_back"]
                    and c4["active_after_rollback"] == c0 and c4["restored_complete"])
        bv["exhaustive_construction"] = exhaustive[b["context"]]["construction_ok"]
        bv["exhaustive_reference_mismatch"] = exhaustive[b["context"]]["reference_mismatch"]
        bv["test_new"], bv["test_resumed"] = metrics(b["context"], "new"), metrics(b["context"], "resumed")
        affected, unaffected = [], []
        for s_ in sorted(types["Airport"].values):
            for t_ in sorted(types["Operator"].values):
                same = SV._same_run(SV.traverse(t0, s_, t_, 16), SV.traverse(t1, s_, t_, 16))
                (unaffected if same else affected).append((s_, t_))
        sample = sorted(random.Random(f"episodes|{world.identity}|{b['index']}").sample(
            unaffected, min(IT.EPISODE_UNAFFECTED, len(unaffected))))
        supplied = [(None, s_, t_, 16) for s_, t_ in affected + sample]
        admitted = [r for r in rows if r["split"] == "test" and r["release"][c0]["new"]["admitted"]
                    and r["decision"]["execute"]]
        moved, rest = [], []
        for r in admitted:
            q = r["decision"]["query"]
            (rest if SV._same_run(SV.traverse(t0, *q), SV.traverse(t1, *q)) else moved).append(r)
        rest = random.Random(f"interp|{world.identity}|{b['index']}").sample(rest, min(IT.INTERPRETED_UNAFFECTED,
                                                                                     len(rest)))
        interpreted = [(r["id"], *r["decision"]["query"]) for r in sorted(moved + rest, key=lambda r: r["id"])]
        for condition, qs in (("supplied", supplied), ("interpreted", interpreted)):
            for item_id, s_, t_, budget in qs:
                for k in range(SV.traverse(t0, s_, t_, budget)["calls"]):
                    for policy in ("pin", "restart", "migrate", "naive"):
                        expected_keys[(b["index"], condition, item_id, (s_, t_, budget), k, policy)] = (b, condition)
    # stream the session records once, each against its declared slot
    totals = {"supplied": Counter(), "interpreted": Counter()}
    extra = 0
    for line in _lines(d / "sessions.jsonl.gz"):
        key = (line["branch"], line["condition"], line["item"], tuple(line["query"]), line["k"], line["policy"])
        slot = expected_keys.pop(key, None)
        if slot is None:
            extra += 1
            continue
        b, condition = slot
        policy, rec, c = line["policy"], line["record"], totals[condition]
        binding = session_binding_problems(line, rec, c0, contexts[c0]["snapshot"], b,
                                           rows_by_id.get(line["item"]) if condition == "interpreted" else None, certs)
        if binding:
            fail(f"branch {b['index']} {condition} {key[3]} k={key[4]} {policy}: {binding[0]}")
            continue
        check = SV.check_session(rec, tables, worlds[c0] if policy == "pin" else worlds[b["context"]],
                                 contexts=exported)
        c["records"] += 1
        if policy == "naive":
            c["naive"] += 1
            c["c2_naive_execution_invalid"] += not check["execution_valid"]
            c["c2_naive_switch_not_flagged"] += not check["unjustified_switch"] or check["ok"]
            if check["execution_valid"]:
                c["naive_mixing"] += check["unjustified_mixing"]
                c["naive_contract_violation"] += check["contract_violation"]
                c["naive_no_version"] += check["no_version_outcome"]
            continue
        want = contexts[c0]["snapshot"] if policy == "pin" else b["snapshot"]
        c["c1_c2_policy_failures"] += not (check["ok"] and check["equivalent_to"] == want)
        if policy == "migrate":
            c["migrated"] += len(rec["segments"]) == 2
        if condition == "interpreted":
            r = rows_by_id[line["item"]]
            admitted_ = certs[c0]["certified"] and r["score"] >= certs[c0]["threshold"]
            fin = rec["segments"][-1]["context"]
            t_, _ = release_rule(certs[c0], certs[fin], fin == c0)
            o = SV._record_outcome(rec)
            want_rel = admitted_ and o["reason"] in RESULT_REASONS and t_ is not None and r["score"] >= t_
            c["release_mismatch"] += (line["release"] or {}).get("released") != want_rel
            c[f"released_{policy}"] += want_rel
    if expected_keys:
        fail(f"{len(expected_keys)} declared episode records are missing")
    if extra:
        fail(f"{extra} session records outside the declared episode sets")
    # verdicts
    c0_metrics = metrics(c0, "new")
    a0 = all(t0[r_][x_] == world.facts[r_][x_] for r_ in t0 for x_ in t0[r_])
    activated = [bv for bv in branch_verdicts if bv["activated"]]
    meets = lambda m: m["coverage"] >= COVERAGE and m["error_upper_95"] <= ERROR_BOUND
    l1 = certs[c0]["certified"] and meets(c0_metrics) and all(
        v["rate"] is not None and v["rate"] >= REFUSAL for v in c0_metrics["refusal"].values())
    l2 = a0 and exhaustive[c0]["reference_mismatch"] == 0
    l3 = (len(activated) == len(branch_verdicts) == sum(IT.BRANCH_KINDS.values()) and all(
        bv["gates_pass"] and bv["certified"] and bv["s1_atomic"] and bv["exhaustive_reference_mismatch"] == 0
        and meets(bv["test_new"]) for bv in activated))
    l4 = l1 and l2 and l3 and all(meets(bv["test_resumed"]) for bv in activated)
    construction = {
        "C1_C2": all(totals[c]["c1_c2_policy_failures"] == 0 and totals[c]["c2_naive_execution_invalid"] == 0
                     and totals[c]["c2_naive_switch_not_flagged"] == 0 for c in totals) and not expected_keys and not extra,
        "C4": all(bv.get("c4", True) for bv in branch_verdicts),
        "C5": all(e["construction_ok"] for e in exhaustive.values()),
        "release": all(totals[c]["release_mismatch"] == 0 for c in totals),
        "faults": (lambda f: None if f is None else all(f.values()))(
            next((b["faults"] for b in unit["branches"] if "faults" in b), None)),  # None: no activated branch
    }
    return {"unit": tag, "L1": l1, "L2": l2, "L3": l3, "L4": l4, "construction": construction,
            "c0_test": c0_metrics, "parse_exact": {f"{k[0]}/{k[1]}": v for k, v in parse_exact.items()},
            "branches": branch_verdicts, "episodes": {k: dict(v) for k, v in totals.items()}}


def verify(run_dir: Path, development: bool = False) -> dict:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    problems = []
    if not development:
        prov = run["provenance"]
        if prov["sha256"] != FROZEN_SOURCES or prov["research_tree_dirty"] is not False:
            problems.append("the run is not bound to the frozen source (producer-file hashes or a modified tree)")
    problems += inventory_problems(run, development)
    if problems:  # nothing is verified for a run that is not the declared experiment
        raise VerificationError(f"{len(problems)} problems; first: {problems[:5]}")
    cfg = run["config"]
    units, worlds_out = [], {}
    for wseed, winfo in run["worlds"].items():
        world = W.route_world(int(wseed), n_countries=cfg["countries"])
        study = IT.make_study(world, int(wseed))
        branches = IT.draw_branches(world)
        if branches != winfo["branches"] or study["distribution_identity"] != winfo["distribution_identity"]:
            problems.append(f"world {wseed}: the study or the branch draw does not regenerate")
        route, target = IT.witness_route(world)
        exported_witness = winfo["witness"]
        no_version = False
        if route is not None:
            s, a, b, c = route
            w1 = IT.edited(world, "witness", [["operator_of", a, target], ["operator_of", b, target]])
            f0, f1 = SV.facts_table(world), SV.facts_table(w1)
            naive = SV.replay([f0, f1], [0, 1], s, target, 16)
            no_version = all(not SV._same_outcome(naive, SV.traverse(f, s, target, 16)) for f in (f0, f1))
        if (exported_witness.get("found") is True) != (route is not None) or (route is not None and (
                exported_witness.get("route") != route or exported_witness.get("target") != target
                or exported_witness.get("naive_no_version") != no_version)):
            problems.append(f"world {wseed}: the exported witness does not re-derive")
        worlds_out[wseed] = {"witness_ok": route is not None and no_version
                             and exported_witness.get("naive_execution_valid") is True
                             and exported_witness.get("policies_ok") is True}
        for info in [u for u in run["units"] if str(u["world"]) == wseed]:
            units.append(verify_unit(run_dir, info, world, study, branches, problems, cfg["exhaustive_budgets"],
                                     development))
    inits = sorted({u["init"] for u in run["units"]})
    by_world = {w: [u for u in units if u["unit"].startswith(f"w{w} ")] for w in run["worlds"]}
    learned = {p: {w: sum(u[p] for u in us) for w, us in by_world.items()} for p in ("L1", "L2", "L3", "L4")}
    need = 4 if len(inits) == 5 else len(inits)
    verdict = {
        "construction_every_unit": all(all(v is not False for v in u["construction"].values()) for u in units)
        and all(w["witness_ok"] for w in worlds_out.values()),
        "construction_not_exercised": sorted(f"{u['unit']}: {k}" for u in units for k, v in u["construction"].items()
                                             if v is None),
        "learned_successes_by_world": learned,
        "learned_holds": {p: all(n >= need for n in v.values()) for p, v in learned.items()},
    }
    if problems:
        raise VerificationError(f"{len(problems)} problems; first: {problems[:5]}")
    return {"verified": True, "commit": run["provenance"]["commit"], "verdict": verdict, "worlds": worlds_out,
            "units": units}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--development", action="store_true", help="a development run: skip the frozen-source binding")
    args = p.parse_args(argv)
    out = verify(args.run_dir, args.development)
    print(json.dumps(out["verdict"], indent=1))
    for u in out["units"]:
        print(u["unit"], {k: u[k] for k in ("L1", "L2", "L3", "L4")}, u["construction"], u["episodes"])
    print("verified: every number, construction check and learned verdict was rebuilt from the export "
          "(gates, activation mechanics and injected faults are the producer's records)")


if __name__ == "__main__":
    main()
