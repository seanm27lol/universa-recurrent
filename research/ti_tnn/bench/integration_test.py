"""The integration test harness (protocols/integration_test_v1.md; development worlds only until the protocol is frozen).

    CUDA_VISIBLE_DEVICES= nice -n 10 .venv-phase2/bin/python research/ti_tnn/bench/integration_test.py \
        --worlds 100 --inits 100

**One unit** is one (route world, init). For each unit it:
1. **Generates the study** from the world seed: until-questions with explicit budgets 1–16, the default-budget
   diagnostic, the four refusal categories, and independent calibration and test draws. It also draws the twelve
   update branches. Nothing here depends on a fitted model.
2. **Trains** the query interpreter and the learned specialists, the PR 7 recipe on atomic facts.
3. **Builds S0's complete context:** the snapshot, its atomic table, a query certificate on the calibration draws,
   the expectation and the world.
4. **Serves C0:** every executable test and diagnostic draw is run, admitted and released through the release
   adapter. The exhaustive sweep then runs every (start, target) at every budget.
5. **Runs each branch.** Fine-tune (PR 8), then compute the atomic table, the actual difference and a certificate.
   Check with the repaired `Checker` and the expected context, and activate the complete context. Then:
   - run the exhaustive sweep;
   - serve the draws on C1, as new requests at t1 and as resumed requests at max(t0, t1);
   - run the episodes: supplied queries at budget 16, and admitted interpreted questions at their parsed
     budgets. Every policy and naive continuation runs at every pause index, is independently checked, and,
     for interpreted questions, passes through the release adapter.

   Finally, roll back and verify.
6. **Injects the protocol's faults** with the unit's own learned contexts.

Once per world, it runs **the deterministic two-fact witness** (with table modules).

**The export,** under `<out>/units/w<world>_i<init>/`, is what the independent verifier
(`integration_test_verify.py`) rebuilds every number from:
- `unit.json`: the summary, branch records, gates, differences, C4 and faults;
- `interpreter.json.gz`: every calibration, test and diagnostic draw. Each holds its occurrence id, parse,
  full-precision score and component probabilities, pre-threshold decision, per-context result and
  correctness label, and admission and release decisions;
- `tables.json.gz`, `certificates.json.gz`, `contexts.json.gz`, `exhaustive.json.gz`;
- `episodes.jsonl.gz` (compact rows) and `sessions.jsonl.gz` (every full session record, with its release
  decision).

**The confirmation** (protocols/integration_test_v1.md, frozen revision 4) runs worlds 800–802 × inits 0–4 from
the freezing commit:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 .venv-phase2/bin/python research/ti_tnn/bench/integration_test.py \
        --worlds 800 801 802 --inits 0 1 2 3 4 --out runs/titnn_integration_confirmation
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated import calibration as K  # noqa: E402
from integrated import certificates as Q  # noqa: E402
from integrated import interpreters as I  # noqa: E402
from integrated import modules as M  # noqa: E402
from integrated import query_interpreter as QI  # noqa: E402
from integrated import recurrence as R  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import release as RL  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import serving as V  # noqa: E402
from integrated import session_verification as SV  # noqa: E402
from integrated import sessions as SS  # noqa: E402
from integrated import snapshots as SN  # noqa: E402
from integrated import update_checks as U  # noqa: E402
from integrated import updates as D  # noqa: E402
from integrated.contracts import content_identity  # noqa: E402

GENERATOR_VERSION = "ti_tnn_integration_study_v1"
POLICY_VERSION = "ti_tnn_query_policy_v1"
EVENT_VERSION = "ti_tnn_integration_event_v1"
GRID = (0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0)  # the robustness study's grid
SIZES = {"calibration": 1000, "test": 1000, "default_diagnostic": 300}
TRAIN_SHARE, P_REFUSAL, REFUSAL_TRAIN_SHARE = 0.3, 0.2, 0.05
REFUSAL_KINDS = ("unsupported", "over_budget", "ill_typed", "unknown_entity")
BRANCH_KINDS = {"operator": 4, "route": 4, "two_fact": 4}
EPISODE_UNAFFECTED, INTERPRETED_UNAFFECTED = 200, 50
PROPOSER = "offline-update"
COVERAGE, ERROR_BOUND, REFUSAL = 0.90, 0.01, 0.95
SOURCES = ("integration_test.py", "../integrated/sessions.py", "../integrated/session_verification.py",
           "../integrated/serving.py", "../integrated/certificates.py", "../integrated/update_checks.py",
           "../integrated/query_interpreter.py", "../integrated/recurrence.py", "../integrated/modules.py",
           "../integrated/release.py")

BUDGET_TEMPLATES = ("which is the first airport within {n} {conn} of {s} that {t} operates ?",
                    "starting from {s} , name the first airport run by {t} within {n} {conn} .")
DEFAULT_TEMPLATES = ("which is the first airport reached from {s} that {t} operates ?",
                     "starting from {s} , name the first airport run by {t} .")
UNSUPPORTED_TEMPLATES = ("which is the first airport within {n} {conn} of {s} whose runway is longest ?",
                         "starting from {s} , name the first airport with the busiest terminal within {n} {conn} .")


# --- the study -----------------------------------------------------------------------------------------------------
def render(template, s, t=None, n=None):
    return template.format(s=s, t=t, n=n, conn="connection" if n == 1 else "connections")


def position(text, token):
    toks = text.split()
    return toks.index(token) if token is not None and toks.count(token) == 1 else None


def make_study(world, seed: int, sizes: dict | None = None) -> dict:
    """Every question for one world (module docstring, step 1)."""
    sizes = {**SIZES, **(sizes or {})}
    rng = random.Random(f"{GENERATOR_VERSION}|{world.identity}|{seed}")
    airports, operators = world.types["Airport"].values, world.types["Operator"].values
    others = world.types["City"].values + world.types["Country"].values
    n_countries, n_ops = len(world.types["Country"].values), len(operators)
    instances = [(s, t, n) for s in airports for t in operators for n in list(range(1, 17)) + [None]]
    rng.shuffle(instances)
    n_train = round(TRAIN_SHARE * len(instances))
    train_inst, held = instances[:n_train], instances[n_train:]
    held_budget = [x for x in held if x[2] is not None]
    held_default = [x for x in held if x[2] is None]
    items, texts = [], set()

    def add(**kw):
        kw["id"] = f"{world.seed}-{len(items)}"
        kw.setdefault("budget", None)
        kw.setdefault("target", None)
        kw["start_position"] = position(kw["text"], kw["start"])
        kw["target_position"] = position(kw["text"], kw["target"])
        items.append(kw)
        return kw

    def supported(split, category, s, t, n):
        tpl = rng.choice(DEFAULT_TEMPLATES if n is None else BUDGET_TEMPLATES)
        return add(split=split, category=category, kind="query", text=render(tpl, s, t, n), start=s, target=t,
                   budget=16 if n is None else n, default=n is None, expected="result",
                   outputs=["connects_to", "operator_of", QI.DEFAULT if n is None else f"b{n}", QI.END])

    def refusal(kind, split, category):
        n = rng.randint(1, 16)
        if kind == "unsupported":
            s = rng.choice(airports)
            return dict(split=split, category=category, kind=kind, text=render(rng.choice(UNSUPPORTED_TEMPLATES), s, n=n),
                        start=s, expected="refuse", outputs=[QI.UNSUPPORTED], budget=n, default=False)
        s, t = rng.choice(airports), rng.choice(operators)
        if kind == "over_budget":
            n = rng.randint(17, 40)
            outputs = ["connects_to", "operator_of", QI.OVER, QI.END]
        else:
            outputs = ["connects_to", "operator_of", f"b{n}", QI.END]
            if kind == "ill_typed":
                s = rng.choice(others)
            elif rng.random() < 0.5:
                s = f"country_{n_countries + rng.randint(0, 40)}_city_0_airport"
            else:
                t = f"operator_{n_ops + rng.randint(0, 20)}"
        return dict(split=split, category=category, kind=kind, text=render(rng.choice(BUDGET_TEMPLATES), s, t, n),
                    start=s, target=t, expected="refuse", outputs=outputs, budget=n, default=False)

    for s, t, n in train_inst:
        texts.add(supported("train", "train", s, t, n)["text"])
    quota = max(1, round(REFUSAL_TRAIN_SHARE * len(train_inst)))
    for kind in REFUSAL_KINDS:
        made = 0
        while made < quota:
            it = refusal(kind, "train", "train_refusal")
            if it["text"] not in texts:
                texts.add(it["text"])
                add(**it)
                made += 1

    def id_draw(split):
        if rng.random() < P_REFUSAL:
            kind = rng.choice(REFUSAL_KINDS)
            while True:
                it = refusal(kind, split, kind)
                if it["text"] not in texts:
                    return add(**it)
        s, t, n = rng.choice(held_budget)
        return supported(split, "query", s, t, n)

    for _ in range(sizes["calibration"]):
        id_draw("calibration")
    for _ in range(sizes["test"]):
        id_draw("test")
    for _ in range(sizes["default_diagnostic"]):
        s, t, n = rng.choice(held_default)
        supported("diagnostic", "default_budget", s, t, n)
    return {"items": items, "sizes": sizes, "generator": GENERATOR_VERSION,
            "distribution_identity": content_identity({"generator": GENERATOR_VERSION, "world": world.identity,
                                                       "seed": seed, "sizes": sizes, "p_refusal": P_REFUSAL,
                                                       "templates": [BUDGET_TEMPLATES, DEFAULT_TEMPLATES,
                                                                     UNSUPPORTED_TEMPLATES]})}


# --- update branches, drawn from the world alone --------------------------------------------------------------------
def all_outcomes(world, budget=16) -> dict:
    facts = SV.facts_table(world)
    return {(s, t): SV.traverse(facts, s, t, budget) for s in world.types["Airport"].values
            for t in world.types["Operator"].values}


def edited(world, label, changes):
    facts = {r: dict(v) for r, v in world.facts.items()}
    for r, x, y in changes:
        facts[r][x] = y
    return W.World(f"{world.name}+{label}", world.types, world.relations, facts, world.seed)


def draw_branches(world) -> list[dict]:
    """Twelve consequential updates, each an independent branch from the same world, fixed from the seed."""
    rng = random.Random(f"integration-branches|{world.identity}")
    airports, operators = list(world.types["Airport"].values), list(world.types["Operator"].values)
    base = all_outcomes(world)
    conn, ops = world.facts["connects_to"], world.facts["operator_of"]
    out = []

    def consequential(changes):
        new = edited(world, "probe", changes)
        return any(base[k] != v for k, v in all_outcomes(new).items())

    def take(kind, propose):
        made, tries = 0, 0
        while made < BRANCH_KINDS[kind] and tries < 10_000:
            tries += 1
            changes = propose()
            if changes and changes not in [b["changes"] for b in out] and consequential(changes):
                out.append({"kind": kind, "changes": changes})
                made += 1
        if made < BRANCH_KINDS[kind]:
            raise RuntimeError(f"could not draw {BRANCH_KINDS[kind]} consequential {kind} updates")

    def operator_change():
        x = rng.choice(airports)
        return [["operator_of", x, rng.choice([o for o in operators if o != ops[x]])]]

    def route_change():
        x = rng.choice(airports)
        y = rng.choice([a for a in airports if a not in (x, conn[x])])
        return [["connects_to", x, y]]

    reached = {conn[p] for p in airports if conn[p] != p}  # airports some route arrives at (a start is never tested)

    def two_fact():
        x = rng.choice(sorted(reached))
        y = conn[x]
        if y == x:
            return None
        o1 = rng.choice([o for o in operators if o != ops[x]])
        o2 = rng.choice([o for o in operators if o != ops[y]])
        return [["operator_of", x, o1], ["operator_of", y, o2]]  # two airports a route visits in turn

    take("operator", operator_change)
    take("route", route_change)
    take("two_fact", two_fact)
    for i, b in enumerate(out):
        b["index"] = i
        b["relation"] = b["changes"][0][0]
    return out


# --- the policy ----------------------------------------------------------------------------------------------------
def decide(parse, types) -> dict:
    """Bind, type-check and contract-check a parse before anything runs."""
    if parse is None:
        return {"execute": False, "reason": "unsupported characters"}
    if parse.outcome == "unsupported":
        return {"execute": False, "reason": "unsupported"}
    if parse.outcome == "over_budget":
        return {"execute": False, "reason": "invalid query: budget outside 1-16"}
    if parse.outcome != "query" or parse.relations != ("connects_to", "operator_of"):
        return {"execute": False, "reason": "malformed"}
    bound = []
    for token in (parse.start, parse.target):
        hit = [t for t in sorted(types) if types[t].contains(token)]
        if not hit:
            return {"execute": False, "reason": "unknown entity"}
        bound.append(types[hit[0]].value(token))
    query = R.UntilQuery(bound[0], bound[1])
    return {"execute": True, "reason": "", "query": query, "budget": parse.budget}


def precheck(decision, registry, types) -> dict:
    """The runtime's own checks (types; budget contract) turn an executable decision into a refusal if they fail."""
    if not decision["execute"]:
        return decision
    problems = R.query_problems(decision["query"], registry, types) + SS.contract_problems(decision["query"],
                                                                                         decision["budget"])
    return {**decision, "execute": False, "reason": "invalid query"} if problems else decision


def reference(world, item) -> dict:
    return SV.traverse(SV.facts_table(world), item["start"], item["target"], item["budget"])


def exact_parse(parse, item) -> bool:
    return (parse is not None and parse.outcome == "query" and parse.relations == ("connects_to", "operator_of")
            and (parse.start, parse.target, parse.budget, parse.default_budget)
            == (item["start"], item["target"], item["budget"], item["default"]))


def parse_fields(parse) -> dict | None:
    if parse is None:
        return None
    return {"outcome": parse.outcome, "relations": list(parse.relations), "start": parse.start, "target": parse.target,
            "budget": parse.budget, "default": parse.default_budget}


def until(types, s, t):
    return R.UntilQuery(types["Airport"].value(s), types["Operator"].value(t))


class Lines:
    """One gzip-compressed JSON object per line, written as the run goes."""

    def __init__(self, path: Path):
        self.path, self.n = path, 0
        self._f = gzip.open(path, "wt")

    def write(self, obj):
        self._f.write(json.dumps(obj, separators=(",", ":")) + "\n")
        self.n += 1

    def close(self):
        self._f.close()


# --- interpreter records: every draw, its parse, score, pre-threshold decision, results and release decisions ---
def interpreter_records(items, parses, registry, types) -> list[dict]:
    rows = []
    for item, parse in zip(items, parses):
        d = precheck(decide(parse, types), registry, types)
        rows.append({"id": item["id"], "split": item["split"], "category": item["category"],
                     "parse": parse_fields(parse), "score": parse.score if parse else 0.0,
                     "step_probabilities": list(parse.step_probabilities) if parse else None,
                     "start_probability": parse.start_probability if parse else None,
                     "target_probability": parse.target_probability if parse else None,
                     "decision": {"execute": d["execute"], "reason": d["reason"],
                                  "query": [d["query"].start.value, d["query"].target.value, d["budget"]]
                                  if d["execute"] else None},
                     "results": {}, "correct": {}, "release": {}})
    return rows


def execute_under(context, rows, items_by_id, world, types, key=None) -> dict:
    """Run every executable draw on `context`. Records each result and its correctness label (against `world`)
    under `key` (default: the context's identity). Returns the sessions, by draw."""
    key = key or context.identity
    sessions = {}
    for row in rows:
        if not row["decision"]["execute"]:
            continue
        s, t, b = row["decision"]["query"]
        rec = SS.advance(SS.open_session(context, until(types, s, t), b, types), context, types)
        o = SS.outcome(rec)
        item = items_by_id[row["id"]]
        ref = reference(world, item) if item["expected"] == "result" else None
        row["results"][key] = [o["kind"], o["answer"], o["reason"]]
        row["correct"][key] = (o["kind"] == "result" and ref is not None
                               and (o["answer"], o["reason"]) == (ref["answer"], ref["reason"]))
        sessions[row["id"]] = rec
    return sessions


def certificate_records(rows, key) -> list:
    """The certified event's records: calibration draws the policy would release before thresholding."""
    return [(r["id"], r["score"], r["correct"][key]) for r in rows
            if r["split"] == "calibration" and r["results"].get(key, [None])[0] == "result"]


def record_release(rows, sessions, admitting, finishing, contexts, field):
    """Admission under `admitting`, release of the run that finished on `finishing`, for every draw."""
    for row in rows:
        decision = RL.admit(admitting, row["score"])
        if row["id"] in sessions:
            rel = RL.release(decision, sessions[row["id"]], contexts)
        else:
            rel = {"released": False, "reason": row["decision"]["reason"], "effective_threshold": None,
                   "supporting_step": None}
        row["release"].setdefault(finishing.identity, {})[field] = {
            "admitted": decision["admitted"], "released": rel["released"], "reason": rel["reason"],
            "threshold": rel["effective_threshold"], "supporting_step": rel["supporting_step"]}


def metrics(rows, items_by_id, key, field) -> dict:
    """Coverage, accepted error and observed refusal rates for one context and release field."""
    test = [r for r in rows if r["split"] == "test"]
    rel = lambda r: r["release"].get(key, {}).get(field, {}).get("released", False)
    supported = [r for r in test if items_by_id[r["id"]]["expected"] == "result"]
    released = [r for r in test if rel(r)]
    errors = sum(not r["correct"].get(key, False) for r in released)
    refusal = {}
    for kind in REFUSAL_KINDS:
        rs = [r for r in test if r["category"] == kind]
        refusal[kind] = {"n": len(rs), "abstained": sum(not rel(r) for r in rs),
                         "rate": sum(not rel(r) for r in rs) / len(rs) if rs else None}
    return {"supported": len(supported), "released": len(released), "errors": errors,
            "coverage": sum(rel(r) for r in supported) / max(1, len(supported)),
            "error_upper_95": K.clopper_pearson_upper(errors, len(released)), "refusal": refusal}


# --- the exhaustive sweep ------------------------------------------------------------------------------------------
def exhaustive(context, table, world, types, budgets) -> tuple[dict, list, str]:
    """Every (start, target) at every budget through the sessions layer. Construction compares each with the
    independent traversal over the context's own table; correctness compares with the reference world. Returns
    (counts, compact rows [reason, answer, calls] in canonical order, digest of the full outcomes)."""
    facts = SV.facts_table(world)
    counts, rows, full = Counter(), [], []
    for s in sorted(world.types["Airport"].values):
        for t in sorted(world.types["Operator"].values):
            for b in budgets:
                rec = SS.advance(SS.open_session(context, until(types, s, t), b, types), context, types)
                got = SV._record_outcome(rec)
                counts["queries"] += 1
                counts["construction_mismatch"] += not SV._same_run(got, SV.traverse(table, s, t, b))
                ref = SV.traverse(facts, s, t, b)
                counts["reference_mismatch"] += (got["reason"], got["answer"]) != (ref["reason"], ref["answer"])
                rows.append([got["reason"], got["answer"], got["calls"]])
                full.append([got["reason"], got["answer"], got["visited"], got["applications"], got["calls"]])
    return dict(counts), rows, content_identity(full)


def atomic_correctness(table, world) -> dict:
    wrong = {r: sum(table[r][x] != world.facts[r][x] for x in table[r]) for r in table}
    return {"wrong_by_relation": wrong, "correct": not any(wrong.values())}


# --- episodes --------------------------------------------------------------------------------------------------------
COST_KEYS = ("applications", "snapshot_verifications", "difference_lookups", "revalidations")


def resume_cost(record, paused) -> int:
    """Work after the pause. A record that continued the paused session subtracts the sunk work before it; a
    restarted record is a fresh session and counts its whole run."""
    total = sum(record["costs"].get(k, 0) for k in COST_KEYS)
    if record.get("restarted_from") is None:
        total -= sum(paused["costs"].get(k, 0) for k in COST_KEYS)
    return total


def episodes(c0, c1, tables, exported, objects, world0, world1, queries, types, sessions_out, rows_out, branch,
             condition) -> dict:
    """Every policy and naive continuation at every pause index of every query, independently checked; for
    interpreted questions, every policy record also passes through the release adapter. `queries`: dicts with
    query, budget, and (interpreted) item id and admission under C0."""
    diff, _, _ = SS.actual_difference(c0.snapshot, c1.snapshot, types)
    stats = Counter()
    for q in queries:
        query, budget, admission = q["query"], q["budget"], q.get("admission")
        full0 = SS.advance(SS.open_session(c0, query, budget, types), c0, types)
        full1 = SS.advance(SS.open_session(c1, query, budget, types), c1, types)
        for k in range(full0["calls"]):
            p = json.loads(json.dumps(SS.advance(SS.open_session(c0, query, budget, types), c0, types, calls=k)))
            recs = {"pin": SS.resume_pin(p, c0, types), "restart": SS.resume_restart(p, c1, types),
                    "migrate": SS.resume_migrate(p, c1, diff, types), "naive": SS.continue_unchecked(p, c1, types)}
            stats["episodes"] += 1
            stats["c1_pin_mismatch"] += recs["pin"]["state"] != full0["state"]
            stats["c1_restart_mismatch"] += recs["restart"]["state"] != full1["state"]
            stats["c1_migrate_mismatch"] += (SS.outcome(recs["migrate"]) != SS.outcome(full1)
                                             or recs["migrate"]["state"]["visited"] != full1["state"]["visited"])
            stats["migrated"] += recs["migrate"]["migration"]["migrated"]
            for name, rec in recs.items():
                world = world0 if name == "pin" else world1
                check = SV.check_session(rec, tables, world, contexts=exported)
                release = None
                if name != "naive":
                    expected = c0 if name == "pin" else c1
                    stats["c2_policy_record_rejected"] += not (check["ok"] and
                                                              check["equivalent_to"] == expected.snapshot.digest)
                    if admission is not None:
                        release = RL.release(admission, rec, objects)
                        o = SS.outcome(rec)
                        t0, t1 = c0.certificate.threshold, c1.certificate.threshold
                        rule = t0 if name == "pin" else (max(t0, t1) if None not in (t0, t1) else None)
                        want = (admission["admitted"] and o["kind"] == "result" and rule is not None
                                and admission["score"] >= rule)
                        stats["release_rule_mismatch"] += release["released"] != want
                        stats[f"released_{name}"] += release["released"]
                else:
                    stats["c2_naive_execution_invalid"] += not check["execution_valid"]
                    stats["c2_naive_switch_not_flagged"] += not check["unjustified_switch"] or check["ok"]
                    if check["execution_valid"]:
                        stats["naive_mixing"] += check["unjustified_mixing"]
                        stats["naive_contract_violation"] += check["contract_violation"]
                        stats["naive_no_version"] += check["no_version_outcome"]
                sessions_out.write({"branch": branch, "condition": condition, "item": q.get("item"),
                                    "query": [query.start.value, query.target.value, budget], "k": k, "policy": name,
                                    "record": rec, "release": release})
                o = SS.outcome(rec)
                rows_out.write([branch, condition, q.get("item"), query.start.value, query.target.value, budget, k,
                                name, o["kind"], o["answer"], o["reason"],
                                rec["migration"]["migrated"] if rec.get("migration") else None,
                                None if release is None else release["released"]])
            stats["resume_cost_migrate"] += resume_cost(recs["migrate"], p)
            stats["resume_cost_restart"] += resume_cost(recs["restart"], p)
            if recs["migrate"]["migration"]["migrated"]:
                m = recs["migrate"]
                stats["migrated_suffix_applications"] += m["costs"]["applications"] - p["costs"]["applications"]
                stats["migrated_revalidations"] += m["costs"]["revalidations"]
                stats["migrated_lookups"] += m["costs"]["difference_lookups"]
                stats["restart_applications_where_migrated"] += recs["restart"]["costs"]["applications"]
    return dict(stats)


def affected_pairs(table0, table1, world, budget=16) -> tuple[list, list]:
    affected, unaffected = [], []
    for s in sorted(world.types["Airport"].values):
        for t in sorted(world.types["Operator"].values):
            (affected if not SV._same_run(SV.traverse(table0, s, t, budget), SV.traverse(table1, s, t, budget))
             else unaffected).append((s, t))
    return affected, unaffected


# --- the witness and the faults ------------------------------------------------------------------------------------
def table_context(world, expectation):
    snap = SN.publish(G.table_registry(world))
    return V.ServingContext(snap, Q.certify([(f"q{i}", 0.9995, True) for i in range(400)], expectation, snap.digest),
                            expectation, world.identity)


def witness_route(world):
    """The first route S -> a -> b -> c (sorted by S) with four distinct airports, a and b not run by c's operator."""
    conn, ops = world.facts["connects_to"], world.facts["operator_of"]
    for s in sorted(world.types["Airport"].values):
        a = conn[s]; b = conn[a]; c = conn[b]
        if len({s, a, b, c}) == 4 and ops[a] != ops[c] and ops[b] != ops[c]:
            return [s, a, b, c], ops[c]
    return None, None


def witness(world, types) -> dict:
    """The deterministic two-fact witness with table modules."""
    route, target = witness_route(world)
    if route is None:
        return {"found": False}
    s, a, b, c = route
    w1 = edited(world, "witness", [["operator_of", a, target], ["operator_of", b, target]])
    expectation = Q.Expectation("witness-table-interpreter", POLICY_VERSION, EVENT_VERSION, "witness", GRID)
    c0, c1 = table_context(world, expectation), table_context(w1, expectation)
    exported = {c.identity: {"snapshot": c.snapshot.digest} for c in (c0, c1)}
    tables = {c.snapshot.digest: SS.atomic_table(c.snapshot.registry, types) for c in (c0, c1)}
    diff, _, _ = SS.actual_difference(c0.snapshot, c1.snapshot, types)
    p = SS.advance(SS.open_session(c0, until(types, s, target), 16, types), c0, types, calls=1)
    naive = SV.check_session(SS.continue_unchecked(p, c1, types), tables, w1, contexts=exported)
    policies = {name: SV.check_session(rec, tables, w1 if name != "pin" else world, contexts=exported) for name, rec in (
        ("pin", SS.resume_pin(p, c0, types)), ("restart", SS.resume_restart(p, c1, types)),
        ("migrate", SS.resume_migrate(p, c1, diff, types)))}
    return {"found": True, "route": [s, a, b, c], "target": target,
            "naive_execution_valid": naive["execution_valid"], "naive_no_version": naive["no_version_outcome"],
            "naive_mixing": naive["unjustified_mixing"],
            "policies_ok": all(v["ok"] and v["correct"] for v in policies.values())}


def faults(c0, branch_ctx, deployment, checker, record, result, world, new_world, types, expectation) -> dict:
    """The protocol's injected faults, with this unit's learned contexts. True means handled as required."""
    out = {}
    first = sorted(world.types["Airport"].values)[0]
    q = until(types, first, sorted(world.types["Operator"].values)[0])
    p = SS.advance(SS.open_session(c0, q, 16, types), c0, types, calls=0)
    try:
        R.resume(copy.deepcopy(p["state"]), branch_ctx.snapshot.registry, types)
        out["recurrence_resume_on_c1_refused"] = False
    except ValueError:
        out["recurrence_resume_on_c1_refused"] = True
    copy0 = V.ServingContext(SN.publish(c0.snapshot.registry), c0.certificate, expectation, c0.world_identity)
    paused_copy = SS.advance(SS.open_session(copy0, q, 16, types), copy0, types, calls=0)
    net = copy0.snapshot.registry.module("operator_of").forward
    with torch.no_grad():
        next(net.parameters()).add_(1e-3)  # in place; the declared identity is unchanged
    try:
        SS.resume_pin(paused_copy, copy0, types)
        out["in_place_change_refused_by_pin"] = False
    except SS.SessionRefused:
        out["in_place_change_refused_by_pin"] = True
    alternative = V.ServingContext(c0.snapshot, Q.certify(list(c0.certificate.records[:-1]), expectation,
                                                          c0.snapshot.digest), expectation, c0.world_identity)
    try:
        SS.resume_pin(p, alternative, types)
        out["pin_on_another_complete_context_refused"] = alternative.identity == c0.identity
    except SS.SessionRefused:
        out["pin_on_another_complete_context_refused"] = True
    gate = U.calibration_and_resources(branch_ctx.snapshot.registry, c0.snapshot.registry, record, c0.certificate,
                                       expected=expectation, candidate_digest=branch_ctx.snapshot.digest)
    out["certificate_for_s0_fails_the_gate"] = gate["outcome"] == U.FAIL
    other = Q.Expectation("another-interpreter", POLICY_VERSION, EVENT_VERSION, expectation.distribution_identity, GRID)
    gate = U.calibration_and_resources(branch_ctx.snapshot.registry, c0.snapshot.registry, record,
                                       branch_ctx.certificate, expected=other,
                                       candidate_digest=branch_ctx.snapshot.digest)
    out["certificate_for_other_interpreter_fails_the_gate"] = gate["outcome"] == U.FAIL
    forged = Q.QueryCertificate(**{**branch_ctx.certificate.__dict__, "threshold": 0.999 if
                                   branch_ctx.certificate.threshold != 0.999 else 0.0})
    out["forged_certificate_fails_rederivation"] = bool(forged.rederivation_problems())
    swapped = V.ServingContext(result.snapshot, c0.certificate, expectation, new_world.identity)
    decision = deployment.activate(record, result, swapped)
    out["context_with_another_certificate_refused"] = (not decision["activated"] and any(
        "sealed a different certificate" in r for r in decision["reasons"]))
    for budget in (0, 17, True, 3.0):
        s = SS.open_session(c0, q, budget, types)
        out[f"budget_{budget!r}_refused"] = SS.outcome(s)["kind"] == SS.REFUSED and not s["state"]["applications"]
    return out


# --- one unit --------------------------------------------------------------------------------------------------------
def registry_from(world, modules: dict):
    reg = G.Registry(world.relations)
    for rid in sorted(world.relations):
        reg = reg.register(modules[rid])
    return reg


def export_context(ctx) -> dict:
    return {"snapshot": ctx.snapshot.digest, "certificate": ctx.certificate.identity,
            "expectation": ctx.expectation.identity, "world": ctx.world_identity}


def run_unit(world, study, branches, init, args, out: Path) -> dict:
    """One unit, with its complete export under out/units/w<world>_i<init>/."""
    types = world.types
    unit_dir = out / "units" / f"w{world.seed}_i{init}"
    unit_dir.mkdir(parents=True, exist_ok=False)
    t_start, timings = time.time(), {}
    items = study["items"]
    by_id = {x["id"]: x for x in items}
    train = [x for x in items if x["split"] == "train"]
    draws = [x for x in items if x["split"] in ("calibration", "test", "diagnostic")]
    examples = [QI.QueryExample(x["text"], tuple(x["outputs"]), x["start_position"], x["target_position"])
                for x in train]
    t0 = time.time()
    net, vocab, loss = QI.train(examples, [x["text"] for x in train], tuple(world.relations), init,
                                args.interpreter_updates)
    parses = QI.interpret(net, vocab, [x["text"] for x in draws])
    timings["interpreter"] = time.time() - t0
    expectation = Q.Expectation(QI.identity(net, vocab), POLICY_VERSION, EVENT_VERSION, study["distribution_identity"],
                                GRID)
    t0 = time.time()
    trained = M.train_family(world, "specialists", init, args.d, args.hidden, args.steps, args.lr)
    s0 = SN.publish(registry_from(world, trained["modules"]))
    timings["modules"] = time.time() - t0
    rows = interpreter_records(draws, parses, s0.registry, types)
    cal_rows = [r for r in rows if r["split"] == "calibration"]
    serve_rows = [r for r in rows if r["split"] != "calibration"]
    table0 = SS.atomic_table(s0.registry, types)
    tables = {s0.digest: table0}

    def build_context(snapshot, world_, key):
        """Execute the calibration draws on the snapshot (results stored under `key`), certify, and return the
        complete context; the calibration results are then filed under the context's identity."""
        provisional = V.ServingContext(snapshot, Q.certify([], expectation, snapshot.digest), expectation,
                                       world_.identity)
        execute_under(provisional, cal_rows, by_id, world_, types, key=key)
        cert = Q.certify(certificate_records(cal_rows, key), expectation, snapshot.digest)
        return cert

    cert0 = build_context(s0, world, "S0")
    c0 = V.ServingContext(s0, cert0, expectation, world.identity)
    for r in cal_rows:
        for d in (r["results"], r["correct"]):
            if "S0" in d:
                d[c0.identity] = d.pop("S0")
    objects, exported = {c0.identity: c0}, {c0.identity: export_context(c0)}
    certificates = {c0.identity: cert0.as_dict()}
    checker = U.Checker()
    deployment = V.ServingDeployment(c0, D.Authority({PROPOSER}), checker)
    t0 = time.time()
    sessions0 = execute_under(c0, serve_rows, by_id, world, types)
    record_release(serve_rows, sessions0, c0, c0, objects, "new")
    del sessions0
    timings["serve_c0"] = time.time() - t0
    t0 = time.time()
    exh_counts, exh_rows, exh_digest = exhaustive(c0, table0, world, types, args.exhaustive_budgets)
    exhaustive_export = {c0.identity: {"rows": exh_rows, "digest": exh_digest}}
    timings["exhaustive_c0"] = time.time() - t0
    unit = {"world": world.seed, "init": init, "interpreter_loss": loss, "expectation": {
        **{k: v for k, v in expectation.__dict__.items() if k != "grid"}, "grid": list(GRID),
        "identity": expectation.identity},
        "c0": {"context": c0.identity, "s0_digest": s0.digest, "table_digest": SV.table_digest(table0),
               "atomic": atomic_correctness(table0, world), "certificate": {"threshold": cert0.threshold,
                                                                             "certified": cert0.certified},
               "exhaustive": exh_counts, "test_new": metrics(rows, by_id, c0.identity, "new"),
               "diagnostic_released": sum(r["release"][c0.identity]["new"]["released"] for r in rows
                                          if r["split"] == "diagnostic")},
        "parse_exact": {split: {"exact": sum(exact_parse(p, x) for p, x in zip(parses, draws)
                                             if x["split"] == split and x["expected"] == "result"),
                                "supported": sum(x["split"] == split and x["expected"] == "result" for x in draws)}
                        for split in ("calibration", "test", "diagnostic")},
        "branches": []}
    sessions_out = Lines(unit_dir / "sessions.jsonl.gz")
    rows_out = Lines(unit_dir / "episodes.jsonl.gz")
    first_fault_done = False
    for branch in branches:
        tb = time.time()
        b = {"index": branch["index"], "kind": branch["kind"], "changes": branch["changes"]}
        new_world = edited(world, f"branch{branch['index']}", branch["changes"])
        tuned = M.finetune(new_world, "specialists", trained, init, (branch["relation"],), args.update_steps,
                           args.update_lr, True)
        cand = registry_from(new_world, tuned["modules"])
        cand_snap = SN.publish(cand)
        key = f"S1_{branch['index']}"
        cert1 = build_context(cand_snap, new_world, key)
        record = D.UpdateRecord(s0.digest, cand_snap.digest, (branch["relation"],),
                                tuple(tuple(c) for c in branch["changes"]), new_world.identity, checker.version,
                                U.calibration_identity(cert1), None, PROPOSER)
        result = checker.check(s0, cand, record, world, new_world, calibration=cert1, expected=expectation)
        c1 = V.context_for(result, cert1, expectation, new_world.identity)
        for r in cal_rows:
            for d in (r["results"], r["correct"]):
                if key in d:
                    d[c1.identity] = d.pop(key)
        decision = deployment.activate(record, result, c1)
        b.update(record=record.identity, gates=result.outcomes(), activated=decision["activated"],
                 activation_reasons=decision["reasons"], context=c1.identity, snapshot=cand_snap.digest,
                 certificate={"threshold": cert1.threshold, "certified": cert1.certified})
        certificates[c1.identity] = cert1.as_dict()
        objects[c1.identity], exported[c1.identity] = c1, export_context(c1)
        table1 = SS.atomic_table(c1.snapshot.registry, types)
        tables[c1.snapshot.digest] = table1
        diff, _, _ = SS.actual_difference(s0, c1.snapshot, types)
        b["difference"] = {"identity": diff.identity, "changes": [list(c) for c in diff.changes],
                           "classes": SS.classify(diff, world, new_world, record.intended)}
        b["s1_atomic"] = atomic_correctness(table1, new_world)
        b["table_digest"] = SV.table_digest(table1)
        if not decision["activated"]:  # stays in the inventory as a rejected candidate
            unit["branches"].append(b)
            continue
        b["c4"] = {"active_after_activation": deployment.active.identity,
                   "installed_complete": deployment.active is c1 and deployment.active.certificate is cert1}
        b["exhaustive"], ex_rows, ex_digest = exhaustive(c1, table1, new_world, types, args.exhaustive_budgets)
        exhaustive_export[c1.identity] = {"rows": ex_rows, "digest": ex_digest}
        sessions1 = execute_under(c1, serve_rows, by_id, new_world, types)
        record_release(serve_rows, sessions1, c1, c1, objects, "new")  # C1 serving new requests at t1
        record_release(serve_rows, sessions1, c0, c1, objects, "resumed")  # admitted under C0, released on C1
        del sessions1
        b["test_new"] = metrics(rows, by_id, c1.identity, "new")
        b["test_resumed"] = metrics(rows, by_id, c1.identity, "resumed")
        tb_tables = {s0.digest: table0, c1.snapshot.digest: table1}
        affected, unaffected = affected_pairs(table0, table1, world)
        sample = sorted(random.Random(f"episodes|{world.identity}|{branch['index']}").sample(
            unaffected, min(EPISODE_UNAFFECTED, len(unaffected))))
        supplied = [{"query": until(types, s, t), "budget": 16} for s, t in affected + sample]
        b["supplied_episodes"] = {"affected": len(affected), "unaffected_sampled": len(sample), **episodes(
            c0, c1, tb_tables, exported, objects, world, new_world, supplied, types, sessions_out, rows_out,
            branch["index"], "supplied")}
        admitted = [r for r in rows if r["split"] == "test" and r["release"][c0.identity]["new"]["admitted"]
                    and r["decision"]["execute"]]
        moved, rest = [], []
        for r in admitted:
            s, t, budget = r["decision"]["query"]
            same = SV._same_run(SV.traverse(table0, s, t, budget), SV.traverse(table1, s, t, budget))
            (rest if same else moved).append(r)
        rest = random.Random(f"interp|{world.identity}|{branch['index']}").sample(rest, min(INTERPRETED_UNAFFECTED,
                                                                                           len(rest)))
        interp = [{"query": until(types, r["decision"]["query"][0], r["decision"]["query"][1]),
                   "budget": r["decision"]["query"][2], "item": r["id"], "admission": RL.admit(c0, r["score"])}
                  for r in sorted(moved + rest, key=lambda r: r["id"])]
        b["interpreted_episodes"] = {"affected": len(moved), "unaffected_sampled": len(rest), **episodes(
            c0, c1, tb_tables, exported, objects, world, new_world, interp, types, sessions_out, rows_out,
            branch["index"], "interpreted")}
        if not first_fault_done:
            b["faults"] = faults(c0, c1, deployment, checker, record, result, world, new_world, types, expectation)
            first_fault_done = True
        rb = deployment.rollback()
        b["c4"].update(rolled_back=rb["rolled_back"], active_after_rollback=deployment.active.identity,
                       restored_complete=deployment.active is c0 and deployment.active.certificate is cert0)
        b["seconds"] = time.time() - tb
        unit["branches"].append(b)
        print(f"    branch {branch['index']:2d} {branch['kind']:8s} activated {b['activated']} cert {cert1.threshold} "
              f"| exhaustive {b['exhaustive']} | episodes {b['supplied_episodes'].get('episodes', 0)}"
              f"+{b['interpreted_episodes'].get('episodes', 0)} | {b['seconds']:.0f}s", flush=True)
    sessions_out.close()
    rows_out.close()
    unit["timings"] = {**timings, "total": time.time() - t_start}
    files = {"interpreter.json.gz": rows, "tables.json.gz": tables, "certificates.json.gz": certificates,
             "contexts.json.gz": exported, "exhaustive.json.gz": exhaustive_export}
    for name, payload in files.items():
        with gzip.open(unit_dir / name, "wt") as f:
            json.dump(payload, f)
    (unit_dir / "unit.json").write_text(json.dumps(unit))
    unit["files"] = sorted(p.name for p in unit_dir.iterdir())
    unit["sessions_written"] = sessions_out.n
    return unit


def provenance() -> dict:
    here = Path(__file__).resolve().parent
    files = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in SOURCES}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here.parent, capture_output=True,
                           text=True).stdout.strip()
    return {"commit": commit, "research_tree_dirty": bool(dirty), "sha256": files}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--worlds", nargs="+", type=int, required=True)
    p.add_argument("--inits", nargs="+", type=int, required=True)
    p.add_argument("--countries", type=int, default=48)
    p.add_argument("--interpreter-updates", type=int, default=1500)
    p.add_argument("--d", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--update-steps", type=int, default=300)
    p.add_argument("--update-lr", type=float, default=3e-3)
    p.add_argument("--exhaustive-budgets", nargs="+", type=int, default=list(range(1, 17)))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/titnn_integration_pilot"))
    return p.parse_args(argv)


def run(args, out: Path) -> dict:
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "provenance": provenance(), "environment": {"torch": torch.__version__, "python": platform.python_version(),
                                                          "platform": platform.platform(), "threads": torch.get_num_threads()},
              "worlds": {}, "units": []}
    for wseed in args.worlds:
        world = W.route_world(wseed, n_countries=args.countries)
        study = make_study(world, wseed)
        branches = draw_branches(world)
        result["worlds"][str(wseed)] = {
            "identity": world.identity, "distribution_identity": study["distribution_identity"],
            "branches": branches, "witness": witness(world, world.types),
            "counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(Counter((x["split"], x["category"])
                                                                     for x in study["items"]).items())}}
        print(f"world {wseed}: {len(study['items'])} items; witness {result['worlds'][str(wseed)]['witness']}",
              flush=True)
        for init in args.inits:
            unit = run_unit(world, study, branches, init, args, out)
            result["units"].append({"world": wseed, "init": init, "path": f"units/w{wseed}_i{init}",
                                    "seconds": unit["timings"]["total"], "sessions": unit["sessions_written"]})
            (out / "run.json").write_text(json.dumps(result))
            print(f"  unit w{wseed} i{init}: C0 {unit['c0']['test_new']['coverage']:.3f}/"
                  f"{unit['c0']['test_new']['errors']} ({unit['timings']['total']:.0f}s)", flush=True)
    (out / "run.json").write_text(json.dumps(result))
    return result


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    out = args.out / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    run(args, out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
