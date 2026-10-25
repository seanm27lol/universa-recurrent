"""Independent verifier for the factored interpreter's engineering pilot (protocols/factored_interpreter_v1.md).

    .venv-phase2/bin/python research/ti_tnn/bench/factored_pilot_verify.py <run dir>

**Example.** Unit (world 100, init 100, full) records that F+B answered `operator_of` for one question, with
candidates `[["operator_of", "capital_of"]]` and the capital program pruned as ill-typed. From F's exported
seven-label distribution, the verifier re-runs its own implementation of every F+B step:
- refusal before search;
- the candidates above the 0.05 floor;
- the product scores;
- type pruning and execution by following the facts;
- selection, the disagreement ratio and the refusal check.

It requires the same record, field by field. It then rebuilds every certificate, metric and criterion from the
re-derived rows.

**It trusts no stored summary.** It does not import the producer's pilot, grammar, grounding, decoding, decision,
selection or metric code. It uses:
- **the generators:** the robustness study (`make_study`, `augment`) and the world (`relations.generate_world`);
- **numerics:** the binomial helpers (`calibration`: `binomial_p_value`, `clopper_pearson_upper`, `exact_interval`)
  and the canonical hash;
- **the fixed-sequence certificate** written independently in `interpreter_robustness_verify.certificate`;
- **the no-fit verifier's own implementations** (`factored_interpreter_verify`) of the grammar, training-derived
  labels, dictionaries, L, G and W.

**What it re-derives:**

| From | It re-derives |
|---|---|
| A's records (parses) | each decision, by binding and executing the proposed program, and its score: the float32 product of the step probabilities times the entity probability |
| A+K's records (step traces) | the grammar admission and slot count; at every step the allowed set from the types along the chosen prefix, and the choice (the highest-probability allowed output); the stopping rule; the score, as the product of the recorded **original** probabilities times the entity probability. A distribution renormalized over the allowed set, with every disallowed output at exactly 0, is rejected |
| F's and F+B's records (distributions) | F's greedy decision and score; F+B's complete record (candidates, programs, scores, pruning, answers, selection, decision, reason) |
| L, G, W | every row, from its own re-implementation |
| All rows | primary (exact-program) correctness and A's historical answer correctness; every certificate (exact-program for every system, answer-based for A), each deterministic policy's single risk test, every category metric under both policies, every criterion, P1's sanity-gated replication and P5's differences |
| Costs | A's realized training draws (the same generator calls) and F's drawn presentations against the aligned set; tokens and forward MACs from the layer shapes; the evaluation counts (passes, steps, candidates, module applications) |
| The run | the requested inventory (worlds × inits × regimes, once each, development worlds only), the configuration and the producer-source hashes |

**What remains the producer's record:**
- that A's parses and step traces, and F's distributions, are what the trained networks output (no weights are
  exported);
- the wall times.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402  (A's float32 score product, and A's sampling generator, re-run)

import factored_interpreter_verify as FV  # noqa: E402  (this repository's independent no-fit implementations)
import interpreter_robustness as IR  # noqa: E402  (the generator only)
import interpreter_robustness_verify as RV  # noqa: E402  (the independent fixed-sequence certificate)
from integrated import calibration as K  # noqa: E402  (numerics only)
from integrated import relations as W  # noqa: E402

PILOT_VERSION = "ti_tnn_factored_pilot_v1"
END, UNS, AMB = "<end>", "<unsupported>", "<ambiguous>"
NEURAL, CONTROLS = ("A", "AK", "F", "FB"), ("L_raw", "L_vocab", "G", "W")
GRID = (0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.6, 0.4, 0.2, 0.0)
ALPHA, DELTA = 0.01, 0.05
COVERAGE, ID_COVERAGE, ERROR_BOUND, REFUSAL, T1_ACCURACY = 0.80, 0.90, 0.01, 0.95, 0.95
FB_K, FB_FLOOR, FB_RATIO = 2, 0.05, 0.5
MAX_STEPS = 3
DESIGN = {"pilot": PILOT_VERSION, **FV.DESIGN, "updates": 1500, "batch": 64, "grid": list(GRID),
          "fb": {"k": FB_K, "floor": FB_FLOOR, "ratio": FB_RATIO}}
DEV_WORLDS, DEV_INITS = (100, 101), (100, 101)
SOURCES = ("factored_pilot.py", "factored_interpreter.py", "lexical_condition.py", "interpreter_robustness.py",
           "typed_questions.py", "../integrated/segment_grammar.py", "../integrated/segment_grounding.py",
           "../integrated/constrained_decoding.py", "../integrated/interpreters.py", "../integrated/routing.py",
           "../integrated/runtime.py", "../integrated/registry.py", "../integrated/relations.py",
           "../integrated/contracts.py", "../integrated/reference.py", "../integrated/calibration.py")


class VerificationError(AssertionError):
    pass


def _load(path: Path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def norm(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


# --- execution, by following the facts --------------------------------------------------------------------------
def entity_type(world, e):
    return next((t for t in sorted(world.types) if e in world.types[t].values), None)


def follow(world, program, e) -> tuple[str, str | None, str]:
    """(decision, answer, reason): bind, type-check, follow the facts."""
    t = entity_type(world, e)
    if t is None:
        return "abstain", None, "unknown entity"
    value = e
    for r in program:
        if world.relations[r].source != t:
            return "abstain", None, "ill-typed program"
        value, t = world.facts[r][value], world.relations[r].target
    return "answer", value, ""


# --- A, A+K, F and F+B from their records ------------------------------------------------------------------------
def a_row(rec, world) -> dict:
    score = float(torch.tensor(rec["step_probabilities"]).prod()) * rec["entity_probability"]
    if rec["outcome"] != "program":
        dec = ("abstain", None, rec["outcome"])
    else:
        dec = follow(world, rec["steps"], rec["entity"])
    return {"decision": dec[0], "answer": dec[1], "reason": dec[2], "score": score, "entity": rec["entity"],
            "program": list(rec["steps"]) if rec["outcome"] == "program" else None}


def ak_row(rec, item, world, entities, outputs, fail) -> dict:
    g = FV.read(item["text"], entities)
    if g["status"] != "ok":
        if rec.get("grammar") != g["status"] or "steps" in rec:
            fail("A+K: a grammar refusal does not re-derive")
        return {"decision": "abstain", "answer": None, "reason": f"grammar: {g['status']}", "score": None,
                "entity": None, "program": None}
    n, e = len(g["segments"]), rec["entity"]
    t = entity_type(world, e)
    program, outcome, cur = [], None, t
    if t is None:
        outcome = "unknown entity"
        if rec["steps"]:
            fail("A+K: steps were decoded for an unknown entity")
    for k, st in enumerate(rec["steps"] if t is not None else []):
        dist = st["distribution"]
        if len(dist) != len(outputs) or abs(sum(dist) - 1.0) > 1e-4 or min(dist) < 0:
            fail("A+K: a recorded step is not a full output distribution")
        if k < n:
            allowed = sorted([i for i, o in enumerate(outputs) if o in world.relations and world.relations[o].source == cur]
                             + [outputs.index(UNS), outputs.index(AMB)])
        else:
            allowed = sorted([outputs.index(END), outputs.index(UNS), outputs.index(AMB)])
        if st["allowed"] != allowed:
            fail("A+K: an allowed set does not re-derive from the types along the chosen prefix")
        disallowed = [i for i in range(len(outputs)) if i not in allowed]
        if disallowed and all(dist[i] == 0.0 for i in disallowed):
            fail("A+K: a step distribution is renormalized over its allowed set")
        chosen = max(allowed, key=lambda i: (dist[i], -i))
        if st["chosen"] != chosen:
            fail("A+K: a choice is not the highest-probability allowed output")
        o = outputs[chosen]
        if o in (UNS, AMB):
            outcome = "unsupported" if o == UNS else "ambiguous"
            if k != len(rec["steps"]) - 1:
                fail("A+K: decoding continued after a refusal")
            break
        if o == END:
            outcome = "program" if program else "malformed"
            if k != len(rec["steps"]) - 1:
                fail("A+K: decoding continued after <end>")
            break
        program.append(o)
        cur = world.relations[o].target
    outcome = outcome or "malformed"
    score = 1.0
    for st in rec["steps"]:
        score *= st["distribution"][st["chosen"]]
    score *= rec["entity_probability"]
    if (rec["outcome"], rec["program"]) != (outcome, program) or rec["score"] != score:
        fail("A+K: the outcome, program or score (original probabilities) does not re-derive")
    if outcome != "program":
        return {"decision": "abstain", "answer": None, "reason": outcome, "score": score, "entity": e, "program": None}
    dec = follow(world, program, e)
    return {"decision": dec[0], "answer": dec[1], "reason": dec[2], "score": score, "entity": e, "program": program}


def top_label(dist, labels):
    m = max(dist)
    for refusal in (UNS, AMB):
        if dist[labels.index(refusal)] == m:
            return refusal
    return labels[dist.index(m)]


def f_row(dists, labels, e, world):
    program, score = [], 1.0
    for d in dists:
        lab = top_label(d, labels)
        if lab in (UNS, AMB):
            return {"decision": "abstain", "reason": lab, "program": None, "answer": None, "score": None}
        program.append(lab)
        score *= d[labels.index(lab)]
    dec = follow(world, program, e)
    return {"decision": dec[0], "reason": dec[2], "program": program, "answer": dec[1], "score": score}


def fb_row(dists, labels, e, world):
    out = {"decision": "abstain", "reason": "", "program": None, "answer": None, "score": None, "candidates": [],
           "programs": [], "selected": None}
    for d in dists:
        lab = top_label(d, labels)
        if lab in (UNS, AMB):
            return {**out, "reason": f"refusal argmax: {lab}"}
    rels = [lab for lab in labels if lab not in (UNS, AMB)]
    cands = []
    for d in dists:
        ok = sorted((r for r in rels if d[labels.index(r)] >= FB_FLOOR), key=lambda r: (-d[labels.index(r)], r))
        if not ok:
            return {**out, "reason": "no candidate relation"}
        cands.append(ok[:FB_K])
    out["candidates"] = cands
    t = entity_type(world, e)
    progs = []
    for combo in product(*cands):
        score = 1.0
        for d, r in zip(dists, combo):
            score *= d[labels.index(r)]
        typed = t is not None and all(world.relations[r].source == s for r, s in zip(
            combo, [t] + [world.relations[x].target for x in combo[:-1]]))
        progs.append({"program": list(combo), "score": score, "pruned": not typed,
                      "answer": follow(world, combo, e)[1] if typed else None})
    out["programs"] = progs
    alive = sorted((i for i, p in enumerate(progs) if not p["pruned"]), key=lambda i: (-progs[i]["score"], progs[i]["program"]))
    if not alive:
        return {**out, "reason": "no well-typed candidate"}
    out["selected"] = alive[0]
    sel = progs[alive[0]]
    if any(progs[i]["score"] >= FB_RATIO * sel["score"] and progs[i]["answer"] != sel["answer"] for i in alive[1:]):
        return {**out, "reason": "a close candidate gives another answer"}
    for d, r in zip(dists, sel["program"]):
        p = d[labels.index(r)]
        if not (p > d[labels.index(UNS)] and p > d[labels.index(AMB)]):
            return {**out, "reason": "a refusal label outweighs the selected relation"}
    return {**out, "decision": "answer", "program": sel["program"], "answer": sel["answer"], "score": sel["score"]}


# --- correctness, certificates, metrics, criteria ----------------------------------------------------------------
def exact(x, r):
    return (r["decision"] == "answer" and x["expected"] == "answer" and r.get("program") == x["program"]
            and r.get("entity") == x["entity"] and r["answer"] == x["answer"])


def answer_correct(x, r):
    return r["decision"] == "answer" and x["expected"] == "answer" and r["answer"] == x["answer"]


def cert(cal, rows, correct_fn):
    keep = [(x, r) for x, r in zip(cal, rows) if r["decision"] == "answer"]
    c = RV.certificate([r["score"] for _, r in keep], [correct_fn(x, r) for x, r in keep], GRID, ALPHA, DELTA)
    return c, [[x["id"], r["score"], correct_fn(x, r)] for x, r in keep]


def same_certificate(produced: dict, mine: dict) -> bool:
    steps = [[s["threshold"], s["accepted"], s["errors"], s["p_value"], s["rejected"]] for s in produced["steps"]]
    return (steps == [list(s) for s in mine["steps"]] and produced["threshold"] == mine["threshold"]
            and produced["certified"] == mine["certified"] and produced["stop_reason"] == mine["stop_reason"])


def groups_of(items, rows):
    g = defaultdict(list)
    for x, r in zip(items, rows):
        g[x["category"]].append((x, r))
        if x["category"] == "T1":
            g[f"T1:{x['program'][0]}"].append((x, r))
        if x["category"] == "S1":
            g[f"S1:{x['family']}"].append((x, r))
        if "stratum" in x:
            g[f"{x['category']}:{x['stratum']}"].append((x, r))
    return g


def metrics(items, rows, released):
    out = {}
    for cat, pairs in sorted(groups_of(items, rows).items()):
        flags = [released(r) for _, r in pairs]
        acc = [pr for pr, f in zip(pairs, flags) if f]
        errors = sum(not exact(x, r) for x, r in acc)
        n = len(pairs)
        out[cat] = {"n": n, "answered": len(acc), "errors": errors, "coverage": len(acc) / n,
                    "abstention": 1 - len(acc) / n, "error_upper_95": K.clopper_pearson_upper(errors, len(acc)),
                    "abstention_interval_95": K.exact_interval(n - len(acc), n),
                    "exact_accuracy": sum(exact(x, r) for x, r in acc) / n,
                    "accidental_agreement": sum((not exact(x, r)) and x["expected"] == "answer" and r["answer"] == x["answer"]
                                                for x, r in acc),
                    "reasons": dict(Counter(r["reason"] if r["decision"] != "answer" else ("answered" if f else "threshold")
                                            for (_, r), f in zip(pairs, flags)))}
    return out


def crit(certified, cc, nc):
    ans = lambda k, cov: certified and cc[k]["coverage"] >= cov and cc[k]["error_upper_95"] <= ERROR_BOUND  # noqa: E731
    return {"certified": certified,
            "sanity": ans("ID", ID_COVERAGE) and cc["R"]["abstention"] >= REFUSAL and nc["T1"]["exact_accuracy"] >= T1_ACCURACY,
            "s2": ans("S2", COVERAGE), "s1_length3": ans("S1:length3", COVERAGE),
            "joint_novelty": ans("S3", COVERAGE) and cc["S4a_primary"]["abstention"] >= REFUSAL
            and cc["S4b_primary"]["abstention"] >= REFUSAL, "s2_exact_unthresholded": nc["S2"]["exact_accuracy"]}


def historical(items, rows, released):
    """The robustness study's answer-based category metrics (`measure`), re-implemented."""
    out = {}
    for cat, pairs in groups_of(items, rows).items():
        acc = [(x, r) for x, r in pairs if released(r)]
        errors = sum(not answer_correct(x, r) for x, r in acc)
        out[cat] = {"n": len(pairs), "answered": len(acc), "errors": errors, "coverage": len(acc) / len(pairs),
                    "abstention": 1 - len(acc) / len(pairs), "error_upper_95": K.clopper_pearson_upper(errors, len(acc))}
    return out


def historical_criteria(certified, cc, nc):
    ans = lambda k, cov: certified and cc[k]["coverage"] >= cov and cc[k]["error_upper_95"] <= ERROR_BOUND  # noqa: E731
    return {"certified": certified, "sanity": ans("ID", ID_COVERAGE) and cc["R"]["abstention"] >= REFUSAL,
            "taught_wording_learned": (nc["T1"]["answered"] - nc["T1"]["errors"]) / nc["T1"]["n"] >= T1_ACCURACY,
            "joint_novelty": ans("S3", COVERAGE) and cc["S4a_primary"]["abstention"] >= REFUSAL
            and cc["S4b_primary"]["abstention"] >= REFUSAL,
            "s2_transfer": ans("S2", COVERAGE), "s1_structure": ans("S1:length3", COVERAGE)}


# --- realized training cost -----------------------------------------------------------------------------------------
def a_macs(n, d=48, h=64, n_out=8, steps=MAX_STEPS + 1):
    h2 = 2 * h
    return n * 2 * 3 * (d * h + h * h) + n * h2 + h2 * h2 + steps * (
        h2 * h2 + 2 * n * h2 + 3 * ((d + h2) * h2 + h2 * h2) + 2 * h2 * n_out)


def f_macs(n, d=48, h=64, style_d=8, labels=7):
    return n * 2 * 3 * (d * h + h * h) + (2 * h + style_d) * labels


def a_draws(n_examples, seed, updates, batch):
    gen = torch.Generator().manual_seed(seed)
    order, k, draws = [], 0, [0] * n_examples
    for _ in range(updates):
        if k + batch > len(order):
            order, k = torch.randperm(n_examples, generator=gen).tolist(), 0
        idx = order[k:k + batch]
        k += batch
        for _ in idx:
            torch.rand(1, generator=gen)  # interpreters.train's entity-dropout draw, one per example drawn
        for i in idx:
            draws[i] += 1
    return draws


# --- one unit -------------------------------------------------------------------------------------------------------
def verify_unit(run_dir: Path, info: dict, problems: list, design: dict = DESIGN) -> dict:
    tag = f"w{info['world']} i{info['init']} {info['regime']}"
    fail = lambda m: problems.append(f"{tag}: {m}")  # noqa: E731
    d = run_dir / info["path"]
    unit = json.loads((d / "unit.json").read_text())
    world = W.generate_world(info["world"], n_countries=design["countries"])
    entities = {v for t in world.types.values() for v in t.values}
    study = IR.make_study(world, info["world"])
    train, aug, _ = FV.regime_items(study, info["regime"], world, info["world"])
    items = [x for x in study["items"] if x["split"] != "train"]
    cal = [x for x in items if x["split"] == "calibration"]
    test = [x for x in items if x["split"] == "test"]
    n_cal = len(cal)
    vocab = FV.vocabulary(train)
    contributions, _ = FV.labels(train + aug, entities)
    rels = sorted(world.relations)
    outputs = rels + [END, UNS, AMB]
    labels = rels + [UNS, AMB]
    recs = {name: _load(d / f"records_{name}.json.gz") for name in NEURAL}
    for name, rr in recs.items():
        if [r["id"] for r in rr] != [x["id"] for x in items]:
            fail(f"{name}: the records are not exactly the calibration and test items")
            return {}
    rows = {"A": [], "AK": [], "F": [], "FB": []}
    for x, rec in zip(items, recs["A"]):
        r = a_row(rec, world)
        if (r["decision"], r["answer"], r["reason"], r["score"]) != (rec["decision"], rec["answer"], rec["reason"], rec["score"]):
            fail(f"A: item {x['id']} does not re-derive from its parse")
        rows["A"].append(r)
    for x, rec in zip(items, recs["AK"]):
        r = ak_row(rec, x, world, entities, outputs, fail)
        if (r["decision"], r["answer"], r["reason"]) != (rec["decision"], rec["answer"], rec["reason"]):
            fail(f"A+K: item {x['id']}'s decision does not re-derive")
        rows["AK"].append(r)
    for x, rf, rb in zip(items, recs["F"], recs["FB"]):
        g = FV.read(x["text"], entities)
        if g["status"] != "ok":
            for name, rec in (("F", rf), ("FB", rb)):
                if rec["grammar"] != g["status"] or rec["decision"] != "abstain" or rec["reason"] != g["status"]:
                    fail(f"{name}: a grammar refusal does not re-derive")
                rows[name].append({"decision": "abstain", "reason": g["status"], "program": None, "answer": None,
                                   "score": None, "entity": None})
            continue
        dists = rf["distributions"]
        if len(dists) != len(g["segments"]) or rb["distributions"] != dists or any(
                len(dd) != len(labels) or abs(sum(dd) - 1.0) > 1e-4 for dd in dists):
            fail("F: the recorded distributions are not one full label distribution per segment")
            continue
        f = f_row(dists, labels, g["entity"], world)
        if {k: rf[k] for k in f} != f:
            fail(f"F: item {x['id']} does not re-derive from its distributions")
        b = fb_row(dists, labels, g["entity"], world)
        if {k: rb.get(k) for k in b} != norm(b):
            fail(f"F+B: item {x['id']}'s record (candidates, programs, selection or decision) does not re-derive")
        rows["F"].append({**f, "entity": g["entity"]})
        rows["FB"].append({k: b[k] for k in ("decision", "reason", "program", "answer", "score")} | {"entity": g["entity"]})
    dicts = {rep: FV.dictionary(contributions, rep, vocab) for rep in ("raw", "vocab")}
    lookups = {"L_raw": (dicts["raw"], "raw"), "L_vocab": (dicts["vocab"], "vocab"), "G": (FV.reference(), "raw"),
               "W": (FV.word_sets(contributions), "words")}
    for name, (lookup, rep) in lookups.items():
        rows[name] = [{**FV.decide(x["text"], world, entities, lookup, rep, vocab), "score": None} for x in items]
    systems = unit.get("systems", {})
    for name in NEURAL + CONTROLS:
        rr, mine = rows[name], {}
        cal_rows, test_rows = rr[:n_cal], rr[n_cal:]
        if name in NEURAL:
            c, records = cert(cal, cal_rows, exact)
            produced = systems.get(name, {}).get("certificate", {})
            if not same_certificate(produced, c) or produced.get("records") != records:
                fail(f"{name}: its exact-program certificate (or its eligible calibration labels) does not re-derive")
            th = c["threshold"] if c["certified"] else None
            certified = bool(c["certified"])
            cc = metrics(test, test_rows, lambda r, th=th: r["decision"] == "answer" and th is not None and r["score"] >= th)
        else:
            acc = [(x, r) for x, r in zip(cal, cal_rows) if r["decision"] == "answer"]
            errors = sum(not exact(x, r) for x, r in acc)
            p = K.binomial_p_value(errors, len(acc), ALPHA) if acc else 1.0
            mine_risk = {"calibration_items": n_cal, "accepted": len(acc), "errors": errors, "p_value": p,
                         "certified": bool(acc) and p <= DELTA, "error_upper_95": K.clopper_pearson_upper(errors, len(acc))}
            if norm(systems.get(name, {}).get("certificate")) != norm(mine_risk):
                fail(f"{name}: its calibration risk test does not re-derive")
            certified = mine_risk["certified"]
            cc = metrics(test, test_rows, lambda r, c_=certified: r["decision"] == "answer" and c_)
        nc = metrics(test, test_rows, lambda r: r["decision"] == "answer")
        if norm(systems.get(name, {}).get("certified")) != norm(cc) or norm(systems.get(name, {}).get("none")) != norm(nc):
            fail(f"{name}: its category metrics do not re-derive")
        if norm(systems.get(name, {}).get("criteria")) != norm(crit(certified, cc, nc)):
            fail(f"{name}: its criteria do not re-derive")
        mine[name] = nc
    # A's historical answer-based certificate and criteria, for P1
    hc, _ = cert(cal, rows["A"][:n_cal], answer_correct)
    hp = unit.get("historical_A", {})
    if not same_certificate(hp.get("certificate", {"steps": []}), hc):
        fail("A: the historical answer-based certificate does not re-derive")
    hth = hc["threshold"] if hc["certified"] else None
    hcc = historical(test, rows["A"][n_cal:], lambda r: r["decision"] == "answer" and hth is not None and r["score"] >= hth)
    hnc = historical(test, rows["A"][n_cal:], lambda r: r["decision"] == "answer")
    hcrit = historical_criteria(bool(hc["certified"]), hcc, hnc)
    if norm(hp.get("criteria")) != norm(hcrit):
        fail("A: the historical criteria do not re-derive")
    gate = hcrit["sanity"] and hcrit["taught_wording_learned"]
    p1 = {"sanity_and_t1": gate, "s2_failure_replicated": gate and not hcrit["s2_transfer"],
          "s1_failure_replicated": gate and not hcrit["s1_structure"]}
    if hp.get("p1") != p1:
        fail("A: P1's sanity-gated replication does not re-derive")
    g_s2 = metrics(test, rows["G"][n_cal:], lambda r: r["decision"] == "answer")["S2"]["exact_accuracy"]
    p5 = {n: metrics(test, rows[n][n_cal:], lambda r: r["decision"] == "answer")["S2"]["exact_accuracy"] - g_s2
          for n in ("F", "FB", "L_raw", "L_vocab", "W")}
    if norm(unit.get("p5_s2_exact_minus_G")) != norm(p5):
        fail("P5's differences do not re-derive")
    # realized training cost
    tr = unit.get("training", {})
    examples = train + aug
    draws = a_draws(len(examples), info["init"], design["updates"], design["batch"])
    lengths = [len(x["text"].split()) for x in examples]
    want_a = {"presentations": sum(draws), "examples": len(examples), "tokens": sum(a * b for a, b in zip(draws, lengths)),
              "forward_macs": sum(a * a_macs(b) for a, b in zip(draws, lengths)),
              "draws_by_category": dict(Counter(x["category"] for x, dd in zip(examples, draws) for _ in range(dd)))}
    if {k: tr.get("A", {}).get(k) for k in want_a} != norm(want_a):
        fail("A: its realized training draws, tokens or MACs do not re-derive")
    f = tr.get("F", {})
    aligned = Counter(f"{c['style']}: {' '.join(c['tokens'])} -> {c['label']}" for c in contributions)
    by_tuple = f.get("draws_by_tuple", {})
    if f.get("aligned_presentations") != len(contributions) or not set(by_tuple) <= set(aligned) or \
            sum(by_tuple.values()) != f.get("presentations") or f.get("presentations") != design["updates"] * design["batch"] or \
            f.get("tokens") != sum(v * len(k.split(" -> ")[0].split(": ", 1)[1].split()) for k, v in by_tuple.items()) or \
            f.get("forward_macs") != sum(v * f_macs(len(k.split(" -> ")[0].split(": ", 1)[1].split())) for k, v in by_tuple.items()):
        fail("F: its realized presentations, tokens or MACs are inconsistent with the aligned presentations")
    # evaluation cost counts
    ev = unit.get("evaluation_costs", {})
    admitted = [FV.read(x["text"], entities) for x in items]
    segments = sum(len(g["segments"]) for g in admitted if g["status"] == "ok")
    apps = lambda name: sum(len(r["program"]) for r in rows[name] if r["decision"] == "answer")  # noqa: E731
    want = {"A": {"encoder_passes": len(items), "decoder_steps": len(items) * (MAX_STEPS + 1)},
            "AK": {"encoder_passes": sum(g["status"] == "ok" for g in admitted),
                   "decoder_steps": sum(len(r.get("steps", [])) for r in recs["AK"]),
                   "grammar_refusals": sum(g["status"] != "ok" for g in admitted)},
            "F": {"segment_passes": segments, "module_applications": apps("F")},
            "FB": {"segment_passes": segments,
                   "candidates_proposed": sum(len(r.get("programs", [])) for r in recs["FB"]),
                   "candidates_pruned": sum(p["pruned"] for r in recs["FB"] for p in r.get("programs", [])),
                   "candidates_executed": sum(not p["pruned"] for r in recs["FB"] for p in r.get("programs", [])),
                   "module_applications": sum(len(p["program"]) for r in recs["FB"] for p in r.get("programs", [])
                                              if not p["pruned"])},
            **{n: {"segment_lookups": segments, "module_applications": apps(n)} for n in CONTROLS}}
    for name, counts in want.items():
        if {k: ev.get(name, {}).get(k) for k in counts} != counts:
            fail(f"{name}: its evaluation cost counts do not re-derive")
    return {"unit": tag, "criteria": {n: crit_summary(systems.get(n, {})) for n in NEURAL + CONTROLS}, "p1": p1, "p5": p5}


def crit_summary(s: dict) -> dict:
    return s.get("criteria", {})


def source_hashes() -> dict:
    here = Path(__file__).resolve().parent
    return {s: hashlib.sha256((here / s).read_bytes()).hexdigest() for s in SOURCES}


def verify(run_dir: Path, design: dict = DESIGN) -> dict:
    """`design` is the declared pilot design; tests pass a reduced one (fewer countries and updates), and the run's
    recorded configuration must equal whichever design is given."""
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    problems = []
    if run.get("config") != design or run.get("pilot") != PILOT_VERSION:
        problems.append("the run's configuration is not the declared pilot design")
    if run.get("provenance", {}).get("sha256") != source_hashes():
        problems.append("the run's recorded producer sources are not the sources being verified")
    req = run.get("requested", {})
    want = sorted((w, i, r) for w in req.get("worlds", []) for i in req.get("inits", []) for r in req.get("regimes", []))
    have = [(u["world"], u["init"], u["regime"]) for u in run.get("units", [])]
    if not want or req.get("regimes") != design["regimes"] or len(set(have)) != len(have) or sorted(have) != want:
        problems.append("the unit inventory is not exactly the requested worlds x inits x regimes, once each")
    if any(w not in DEV_WORLDS for w in req.get("worlds", [])) or any(i not in DEV_INITS for i in req.get("inits", [])):
        problems.append("a world or init outside development was requested before the protocol was frozen")
    if problems:
        raise VerificationError(f"{len(problems)} problems; first: {problems[:5]}")
    units = [verify_unit(run_dir, info, problems, design) for info in run["units"]]
    if problems:
        raise VerificationError(f"{len(problems)} problems; first: {problems[:5]}")
    return {"verified": True, "commit": run["provenance"]["commit"], "units": units}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir", type=Path)
    print(json.dumps(verify(p.parse_args(argv).run_dir), indent=1, default=str))


if __name__ == "__main__":
    main()
