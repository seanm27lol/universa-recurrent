"""The integration test harness (protocols/integration_test_v1.md, revision 2: DRAFT, development pilot only).

    CUDA_VISIBLE_DEVICES= nice -n 10 .venv-phase2/bin/python research/ti_tnn/bench/integration_test.py \
        --worlds 100 --inits 100

**One unit** is one (route world, init). For each unit it:
1. **Generates the study** from the world seed: until-questions with explicit budgets 1–16, the default-budget
   diagnostic, the four refusal categories, and independent calibration and test draws. It also draws the
   twelve update branches. Nothing here depends on a fitted model.
2. **Trains** the query interpreter and the learned specialists, the PR 7 recipe on atomic facts.
3. **Builds S0's context:** the snapshot, its atomic table, a query certificate on the calibration draws, the
   expectation and the world.
4. **Checks S0:**
   - exhaustively, with every (start, target) pair at every budget through the sessions layer, compared with
     the independent traversal over S0's table (construction) and with the reference world (competence);
   - on the IID test draws.
5. **Runs each branch.** Fine-tune (PR 8), then compute the atomic table, the actual difference and a fresh
   certificate. Check with the repaired `Checker` and the expected context, and activate the complete context.
   Then run the exhaustive check, the IID test under the restart rule, and the episodes. Every policy runs at
   every pause index and is independently verified. Finally, roll back and verify.
6. **Injects the protocol's faults** with the unit's own learned contexts.

Once per world, it runs **the deterministic two-fact witness** (with table modules).

**Worlds 800–802 are refused** until the protocol is frozen and this check is removed in the freezing commit.
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
from integrated import relations as W  # noqa: E402
from integrated import serving as V  # noqa: E402
from integrated import session_verification as SV  # noqa: E402
from integrated import sessions as SS  # noqa: E402
from integrated import snapshots as SN  # noqa: E402
from integrated import update_checks as U  # noqa: E402
from integrated import updates as D  # noqa: E402
from integrated.contracts import content_identity  # noqa: E402

RESERVED_WORLDS = (800, 801, 802)
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
           "../integrated/query_interpreter.py", "../integrated/recurrence.py", "../integrated/modules.py")

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


def serve(context, decision, types) -> dict:
    session = SS.open_session(context, decision["query"], decision["budget"], types)
    return SS.advance(session, context, types)


def judged(items, parses, context, world, types) -> list[dict]:
    """For every item: what the policy would release before thresholding, and whether it is correct."""
    out = []
    for item, parse in zip(items, parses):
        d = precheck(decide(parse, types), context.snapshot.registry, types)
        row = {"id": item["id"], "score": parse.score if parse else 0.0, "result": False, "correct": False,
               "reason": d["reason"]}
        if d["execute"]:
            o = SS.outcome(serve(context, d, types))
            if o["kind"] == "result":
                ref = reference(world, item) if item["expected"] == "result" else None
                row.update(result=True, reason=o["reason"],
                           correct=ref is not None and (o["answer"], o["reason"]) == (ref["answer"], ref["reason"]))
            else:
                row["reason"] = o["reason"]
        out.append(row)
    return out


def certify_context(snapshot, cal_rows, expectation) -> Q.QueryCertificate:
    return Q.certify([(r["id"], r["score"], r["correct"]) for r in cal_rows if r["result"]], expectation,
                     snapshot.digest)


def test_metrics(items, rows, threshold) -> dict:
    supported = [(x, r) for x, r in zip(items, rows) if x["expected"] == "result"]
    released = [(x, r) for x, r in zip(items, rows) if r["result"] and threshold is not None and r["score"] >= threshold]
    errors = sum(not r["correct"] for _, r in released)
    refusal = {}
    for kind in REFUSAL_KINDS:
        rs = [r for x, r in zip(items, rows) if x["category"] == kind]
        refusal[kind] = {"n": len(rs), "abstention": (sum(not (r["result"] and threshold is not None and
                                                                 r["score"] >= threshold) for r in rs) / len(rs))
                                                     if rs else None}
    return {"threshold": threshold, "supported": len(supported), "released": len(released), "errors": errors,
            "coverage": sum(1 for x, r in released if x["expected"] == "result") / max(1, len(supported)),
            "error_upper_95": K.clopper_pearson_upper(errors, len(released)), "refusal": refusal}


# --- checks --------------------------------------------------------------------------------------------------------
def exhaustive(context, table, world, types, budgets=range(1, 17)) -> dict:
    """Every (start, target) at every budget through the sessions layer: construction (against the independent
    traversal over the context's own table) and competence (against the reference world)."""
    facts = SV.facts_table(world)
    counts = Counter()
    for s in world.types["Airport"].values:
        for t in world.types["Operator"].values:
            query = R.UntilQuery(types["Airport"].value(s), types["Operator"].value(t))
            for b in budgets:
                rec = SS.advance(SS.open_session(context, query, b, types), context, types)
                got = SV._record_outcome(rec)
                counts["queries"] += 1
                counts["construction_mismatch"] += not SV._same_run(got, SV.traverse(table, s, t, b))
                ref = SV.traverse(facts, s, t, b)
                counts["reference_mismatch"] += (got["reason"], got["answer"]) != (ref["reason"], ref["answer"])
    return dict(counts)


def atomic_correctness(table, world) -> dict:
    wrong = {r: sum(table[r][x] != world.facts[r][x] for x in table[r]) for r in table}
    return {"wrong_by_relation": wrong, "correct": not any(wrong.values())}


def episodes(c0, c1, tables, world0, world1, queries, types) -> dict:
    """Every policy and naive continuation at every pause index of every query, independently verified."""
    diff, _, _ = SS.actual_difference(c0.snapshot, c1.snapshot, types)
    stats = Counter()
    for query, budget in queries:
        full0 = SS.advance(SS.open_session(c0, query, budget, types), c0, types)
        full1 = SS.advance(SS.open_session(c1, query, budget, types), c1, types)
        for k in range(full0["calls"]):
            p = SS.advance(SS.open_session(c0, query, budget, types), c0, types, calls=k)
            p = json.loads(json.dumps(p))
            pin = SS.resume_pin(p, c0, types)
            restart = SS.resume_restart(p, c1, types)
            migrate = SS.resume_migrate(p, c1, diff, types)
            naive = SS.continue_unchecked(p, c1, types)
            stats["episodes"] += 1
            stats["c1_pin_mismatch"] += pin["state"] != full0["state"]
            stats["c1_restart_mismatch"] += restart["state"] != full1["state"]
            stats["c1_migrate_mismatch"] += (SS.outcome(migrate) != SS.outcome(full1)
                                             or migrate["state"]["visited"] != full1["state"]["visited"])
            stats["migrated"] += migrate["migration"]["migrated"]
            stats["migration_blocked"] += not migrate["migration"]["migrated"]
            for rec, expected, w in ((pin, c0, world0), (restart, c1, world1), (migrate, c1, world1)):
                check = SV.check_session(rec, tables, w)
                stats["c2_policy_record_rejected"] += not (check["ok"] and check["equivalent_to"] == expected.snapshot.digest)
            check = SV.check_session(naive, tables, world1)
            stats["naive_switch"] += check["unjustified_switch"]
            stats["naive_mixing"] += check["unjustified_mixing"]
            stats["naive_contract_violation"] += check["contract_violation"]
            stats["naive_no_version"] += check["no_version_outcome"]
            stats["c2_naive_mixing_not_flagged"] += check["ok"]
            stats["cost_migrate"] += sum(v for k_, v in migrate["costs"].items() if k_ != "discarded_applications")
            stats["cost_restart"] += sum(v for k_, v in restart["costs"].items() if k_ != "discarded_applications")
    return dict(stats)


def affected_queries(table0, table1, world, types, budget=16) -> tuple[list, list]:
    affected, unaffected = [], []
    for s in world.types["Airport"].values:
        for t in world.types["Operator"].values:
            pair = (s, t)
            (affected if not SV._same_run({**SV.traverse(table0, s, t, budget)}, SV.traverse(table1, s, t, budget))
             else unaffected).append(pair)
    return affected, unaffected


def as_queries(pairs, types, budget=16):
    return [(R.UntilQuery(types["Airport"].value(s), types["Operator"].value(t)), budget) for s, t in pairs]


# --- the witness and the faults ------------------------------------------------------------------------------------
def table_context(world, expectation):
    snap = SN.publish(G.table_registry(world))
    return V.ServingContext(snap, Q.certify([(f"q{i}", 0.9995, True) for i in range(400)], expectation, snap.digest),
                            expectation, world.identity)


def witness(world, types) -> dict:
    """The deterministic two-fact witness with table modules (module docstring)."""
    conn, ops = world.facts["connects_to"], world.facts["operator_of"]
    for s in sorted(world.types["Airport"].values):
        a = conn[s]; b = conn[a]; c = conn[b]
        if len({s, a, b, c}) == 4 and ops[a] != ops[c] and ops[b] != ops[c]:
            break
    else:
        return {"found": False}
    target = ops[c]
    w1 = edited(world, "witness", [["operator_of", a, target], ["operator_of", b, target]])
    expectation = Q.Expectation("witness-table-interpreter", POLICY_VERSION, EVENT_VERSION, "witness", GRID)
    c0, c1 = table_context(world, expectation), table_context(w1, expectation)
    query = R.UntilQuery(types["Airport"].value(s), types["Operator"].value(target))
    tables = {c.snapshot.digest: SS.atomic_table(c.snapshot.registry, types) for c in (c0, c1)}
    diff, _, _ = SS.actual_difference(c0.snapshot, c1.snapshot, types)
    p = SS.advance(SS.open_session(c0, query, 16, types), c0, types, calls=1)
    naive = SV.check_session(SS.continue_unchecked(p, c1, types), tables, w1)
    policies = {name: SV.check_session(rec, tables, w1 if name != "pin" else world) for name, rec in (
        ("pin", SS.resume_pin(p, c0, types)), ("restart", SS.resume_restart(p, c1, types)),
        ("migrate", SS.resume_migrate(p, c1, diff, types)))}
    return {"found": True, "route": [s, a, b, c], "naive_no_version": naive["no_version_outcome"],
            "naive_mixing": naive["unjustified_mixing"],
            "policies_ok": all(v["ok"] and v["correct"] for v in policies.values())}


def faults(c0, branch_ctx, deployment, checker, record, result, world, new_world, types, expectation) -> dict:
    """The protocol's injected faults, with this unit's learned contexts. True means handled as required."""
    out = {}
    q = R.UntilQuery(types["Airport"].value(world.types["Airport"].values[0]),
                     types["Operator"].value(world.types["Operator"].values[0]))
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


def run_unit(world, study, branches, init, args) -> dict:
    types = world.types
    t_start = time.time()
    timings = {}
    items = study["items"]
    train = [x for x in items if x["split"] == "train"]
    cal = [x for x in items if x["split"] == "calibration"]
    test = [x for x in items if x["split"] == "test"]
    diag = [x for x in items if x["split"] == "diagnostic"]
    examples = [QI.QueryExample(x["text"], tuple(x["outputs"]), x["start_position"], x["target_position"])
                for x in train]
    vocab_texts = [x["text"] for x in train]  # every training question, refusals included
    t0 = time.time()
    net, vocab, loss = QI.train(examples, vocab_texts, tuple(world.relations), init, args.interpreter_updates)
    timings["interpreter_train"] = time.time() - t0
    t0 = time.time()
    parses = {split: QI.interpret(net, vocab, [x["text"] for x in xs])
              for split, xs in (("calibration", cal), ("test", test), ("diagnostic", diag))}
    timings["interpret"] = time.time() - t0
    expectation = Q.Expectation(QI.identity(net, vocab), POLICY_VERSION, EVENT_VERSION, study["distribution_identity"],
                                GRID)
    t0 = time.time()
    trained = M.train_family(world, "specialists", init, args.d, args.hidden, args.steps, args.lr)
    s0_registry = registry_from(world, trained["modules"])
    timings["modules_train"] = time.time() - t0
    s0 = SN.publish(s0_registry)
    table0 = SS.atomic_table(s0.registry, types)
    t0 = time.time()
    tmp = V.ServingContext(s0, Q.certify([], expectation, s0.digest), expectation, world.identity)
    cal_rows0 = judged(cal, parses["calibration"], tmp, world, types)
    cert0 = certify_context(s0, cal_rows0, expectation)
    c0 = V.ServingContext(s0, cert0, expectation, world.identity)
    timings["certify_c0"] = time.time() - t0
    checker = U.Checker()
    deployment = V.ServingDeployment(c0, D.Authority({PROPOSER}), checker)
    unit = {"world": world.seed, "init": init, "interpreter_loss": loss, "interpreter_identity": expectation.interpreter_identity,
            "s0_digest": s0.digest, "s0_atomic": atomic_correctness(table0, world),
            "c0_certificate": {"threshold": cert0.threshold, "certified": cert0.certified, "steps": [list(s) for s in cert0.steps],
                               "records": len(cert0.records)},
            "parse": {split: {"exact": sum(exact_parse(p, x) for p, x in zip(parses[split], xs) if x["expected"] == "result"),
                              "supported": sum(x["expected"] == "result" for x in xs)}
                      for split, xs in (("calibration", cal), ("test", test), ("diagnostic", diag))},
            "branches": []}
    t0 = time.time()
    test_rows0 = judged(test, parses["test"], c0, world, types)
    unit["c0_test"] = test_metrics(test, test_rows0, cert0.threshold if cert0.certified else None)
    diag_rows0 = judged(diag, parses["diagnostic"], c0, world, types)
    unit["c0_default_diagnostic"] = test_metrics(diag, diag_rows0, cert0.threshold if cert0.certified else None)
    timings["test_c0"] = time.time() - t0
    t0 = time.time()
    unit["c0_exhaustive"] = exhaustive(c0, table0, world, types, budgets=args.exhaustive_budgets)
    timings["exhaustive_c0"] = time.time() - t0
    first_fault_done = False
    for branch in branches:
        b = {"index": branch["index"], "kind": branch["kind"], "changes": branch["changes"], "timings": {}}
        new_world = edited(world, f"branch{branch['index']}", branch["changes"])
        tb = time.time()
        tuned = M.finetune(new_world, "specialists", trained, init, (branch["relation"],), args.update_steps,
                           args.update_lr, True)
        cand = registry_from(new_world, tuned["modules"])
        b["timings"]["finetune"] = time.time() - tb
        t0 = time.time()
        cand_snap = SN.publish(cand)
        tmp1 = V.ServingContext(cand_snap, Q.certify([], expectation, cand_snap.digest), expectation, new_world.identity)
        cal_rows1 = judged(cal, parses["calibration"], tmp1, new_world, types)
        cert1 = certify_context(cand_snap, cal_rows1, expectation)
        b["timings"]["certify"] = time.time() - t0
        t0 = time.time()
        record = D.UpdateRecord(s0.digest, cand_snap.digest, (branch["relation"],), tuple(tuple(c) for c in branch["changes"]),
                                new_world.identity, checker.version, U.calibration_identity(cert1), None, PROPOSER)
        result = checker.check(s0, cand, record, world, new_world, calibration=cert1, expected=expectation)
        b["gates"] = result.outcomes()
        b["timings"]["check"] = time.time() - t0
        c1 = V.context_for(result, cert1, expectation, new_world.identity)
        decision = deployment.activate(record, result, c1)
        b["activated"] = decision["activated"]
        b["activation_reasons"] = decision["reasons"]
        b["certificate"] = {"threshold": cert1.threshold, "certified": cert1.certified, "records": len(cert1.records)}
        if not decision["activated"]:
            unit["branches"].append(b)
            continue
        b["c4_installed_complete"] = deployment.active is c1 and deployment.active.certificate is cert1
        table1 = SS.atomic_table(c1.snapshot.registry, types)
        diff, _, _ = SS.actual_difference(s0, c1.snapshot, types)
        b["difference"] = {k: len(v) for k, v in SS.classify(diff, world, new_world, record.intended).items()}
        b["s1_atomic"] = atomic_correctness(table1, new_world)
        t0 = time.time()
        b["exhaustive"] = exhaustive(c1, table1, new_world, types, budgets=args.exhaustive_budgets)
        b["timings"]["exhaustive"] = time.time() - t0
        t0 = time.time()
        rule = Q.effective_release(cert0, cert1)
        test_rows1 = judged(test, parses["test"], c1, new_world, types)
        b["restart_test"] = test_metrics(test, test_rows1, rule["threshold"])
        b["restart_test"]["supporting_step"] = rule["supporting_step"]
        b["timings"]["test"] = time.time() - t0
        tables = {s0.digest: table0, c1.snapshot.digest: table1}
        affected, unaffected = affected_queries(table0, table1, world, types)
        rng = random.Random(f"episodes|{world.identity}|{branch['index']}")
        sample = sorted(rng.sample(unaffected, min(EPISODE_UNAFFECTED, len(unaffected))))
        t0 = time.time()
        b["supplied_episodes"] = {"affected": len(affected), "unaffected_sampled": len(sample),
                                  **episodes(c0, c1, tables, world, new_world, as_queries(affected + sample, types), types)}
        b["timings"]["episodes_supplied"] = time.time() - t0
        accepted = [(x, p) for x, p, r in zip(test, parses["test"], test_rows0)
                    if r["result"] and cert0.certified and r["score"] >= cert0.threshold]
        interp = []
        for x, p in accepted:
            d = decide(p, types)
            s, t_, budget = d["query"].start.value, d["query"].target.value, d["budget"]
            moved = not SV._same_run(SV.traverse(table0, s, t_, budget), SV.traverse(table1, s, t_, budget))
            interp.append((moved, (d["query"], budget)))
        moved = [q for m, q in interp if m]
        rest = [q for m, q in interp if not m]
        rest = random.Random(f"interp|{world.identity}|{branch['index']}").sample(rest, min(INTERPRETED_UNAFFECTED, len(rest)))
        t0 = time.time()
        b["interpreted_episodes"] = {"affected": len(moved), "unaffected_sampled": len(rest),
                                     **episodes(c0, c1, tables, world, new_world, moved + rest, types)}
        b["timings"]["episodes_interpreted"] = time.time() - t0
        if not first_fault_done:
            b["faults"] = faults(c0, c1, deployment, checker, record, result, world, new_world, types, expectation)
            first_fault_done = True
        rb = deployment.rollback()
        b["rolled_back"] = rb["rolled_back"] and deployment.active is c0 and deployment.active.certificate is cert0
        b["timings"]["total"] = time.time() - tb
        unit["branches"].append(b)
        print(f"    branch {branch['index']:2d} {branch['kind']:8s} gates {'pass' if result.passed() else 'FAIL'} "
              f"cert {cert1.threshold} | exhaustive {b['exhaustive']} | episodes {b['supplied_episodes']['episodes']} "
              f"+{b['interpreted_episodes']['episodes']} | ({b['timings']['total']:.0f}s)", flush=True)
    unit["timings"] = {**timings, "total": time.time() - t_start}
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


def main(argv=None):
    args = parse_args(argv)
    if set(args.worlds) & set(RESERVED_WORLDS):
        raise SystemExit("worlds 800-802 are reserved until the protocol is frozen")
    torch.set_num_threads(args.threads)
    out = args.out / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    result = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "provenance": provenance(), "environment": {"torch": torch.__version__, "python": platform.python_version(),
                                                          "platform": platform.platform(), "threads": torch.get_num_threads()},
              "worlds": {}, "units": []}
    for wseed in args.worlds:
        world = W.route_world(wseed, n_countries=args.countries)
        study = make_study(world, wseed)
        t0 = time.time()
        branches = draw_branches(world)
        result["worlds"][str(wseed)] = {"identity": world.identity, "branches": branches,
                                        "draw_seconds": time.time() - t0, "witness": witness(world, world.types),
                                        "counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(Counter(
                                            (x["split"], x["category"]) for x in study["items"]).items())}}
        print(f"world {wseed}: {len(study['items'])} items, 12 branches drawn, witness "
              f"{result['worlds'][str(wseed)]['witness']}", flush=True)
        for init in args.inits:
            unit = run_unit(world, study, branches, init, args)
            result["units"].append(unit)
            (out / "integration_pilot.json").write_text(json.dumps(result))
            print(f"  unit w{wseed} i{init}: {json.dumps({k: unit[k] for k in ('c0_test', 's0_atomic')})[:400]} "
                  f"({unit['timings']['total']:.0f}s)", flush=True)
    (out / "integration_pilot.json").write_text(json.dumps(result))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
