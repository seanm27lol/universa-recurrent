"""The bounded engineering pilot of the factored-interpreter study (protocols/factored_interpreter_v1.md).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 .venv-phase2/bin/python research/ti_tnn/bench/factored_pilot.py \\
        --worlds 100 101 --inits 100 101 --out runs/titnn_factored_pilot

**Example.** In unit (world 100, init 100, full exposure), the pilot:
- **fits A** exactly as the robustness study's `word_aug` arm;
- **fits F** on the aligned segment presentations;
- **evaluates on all 17,900 calibration and test items:**
  - A, greedy;
  - A+K, A's fit decoded under the grammar's constraints;
  - F, greedy over segment labels;
  - F+B, F's fit with bounded alternatives;
  - the no-fit controls L (raw and vocab keys), G and W.

**What it measures:**
- every system's exact-program certificate on the same calibration items;
- its coverage, error and abstention per category, under primary correctness;
- A's historical answer-based certificate and criteria, computed by the robustness study's own code, for P1;
- training cost as realized: presentations drawn, tokens, forward MACs from the layer shapes and wall time. Equal
  update counts are not equal compute;
- evaluation cost per system.

**What it never does.** The pilot fits only A and F: 2 worlds × 2 inits × 2 regimes × 2 = 16 fits. It never
touches worlds 1000–1002 and never freezes anything. A+K, F+B, L, G and W are not fitted.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import factored_interpreter as FI  # noqa: E402  (grammar use, training-derived labels, L, G, W, the regimes)
import interpreter_robustness as IR  # noqa: E402  (A's historical certificate, metrics and criteria, unchanged)
from integrated import calibration as K  # noqa: E402
from integrated import constrained_decoding as CD  # noqa: E402
from integrated import interpreters as I  # noqa: E402
from integrated import reference as R  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import routing as J  # noqa: E402
from integrated import segment_grammar as SG  # noqa: E402
from integrated import segment_grounding as SGR  # noqa: E402
from integrated.contracts import Budget, Program  # noqa: E402
from integrated.runtime import execute  # noqa: E402

PILOT_VERSION = "ti_tnn_factored_pilot_v1"
GRID = IR.GRID
ALPHA, DELTA = FI.RISK_ALPHA, FI.RISK_DELTA
UPDATES, BATCH = 1500, 64
FB_K, FB_FLOOR, FB_RATIO = 2, 0.05, 0.5
NEURAL = ("A", "AK", "F", "FB")
CONTROLS = ("L_raw", "L_vocab", "G", "W")
DEV_INITS = (100, 101)
COVERAGE, ID_COVERAGE, ERROR_BOUND, REFUSAL, T1_ACCURACY = 0.80, 0.90, 0.01, 0.95, 0.95
SOURCES = ("factored_pilot.py", "factored_interpreter.py", "lexical_condition.py", "interpreter_robustness.py",
           "typed_questions.py", "../integrated/segment_grammar.py", "../integrated/segment_grounding.py",
           "../integrated/constrained_decoding.py", "../integrated/interpreters.py", "../integrated/routing.py",
           "../integrated/runtime.py", "../integrated/registry.py", "../integrated/relations.py",
           "../integrated/contracts.py", "../integrated/reference.py", "../integrated/calibration.py")


# --- execution shared by every system -------------------------------------------------------------------------
def run_program(program, entity, world, registry) -> dict:
    """Bind the entity, type-check and execute through the runtime. {status, answer, reason, applications}."""
    bound = [t for t in sorted(world.types) if world.types[t].contains(entity)]
    if not bound:
        return {"status": "abstain", "answer": None, "reason": "unknown entity", "applications": 0}
    ex = execute(Program(tuple(program), world.types[bound[0]].value(entity)), registry, world.types, Budget(len(program)))
    if ex.status != "answered":
        return {"status": "abstain", "answer": None, "reason": ex.decision.reason, "applications": 0}
    return {"status": "answer", "answer": ex.answer.value, "reason": "", "applications": len(program)}


def well_typed(program, entity_type, world) -> bool:
    types = R.path_types(world, tuple(program))
    return types is not None and types[0] == entity_type


# --- F and F+B, from the original seven-label distributions -----------------------------------------------------
def argmax_label(dist: list[float], labels) -> str:
    """The highest-probability label. A refusal label tied with the maximum wins (<unsupported> before
    <ambiguous>); among relations, ties go to the earlier label."""
    m = max(dist)
    for refusal in (J.UNSUPPORTED, J.AMBIGUOUS):
        if dist[labels.index(refusal)] == m:
            return refusal
    return next(lab for lab, p in zip(labels, dist) if p == m)


def f_greedy(dists, labels, entity, world, registry) -> dict:
    """F: each segment's argmax label, in execution order; a refusal argmax decides the abstention."""
    program, score = [], 1.0
    for dist in dists:
        lab = argmax_label(dist, labels)
        if lab in (J.UNSUPPORTED, J.AMBIGUOUS):
            return {"decision": "abstain", "reason": lab, "program": None, "answer": None, "score": None,
                    "applications": 0}
        program.append(lab)
        score *= dist[labels.index(lab)]
    ex = run_program(program, entity, world, registry)
    return {"decision": ex["status"], "reason": ex["reason"], "program": program, "answer": ex["answer"],
            "score": score, "applications": ex["applications"]}


def fb_policy(dists, labels, entity, world, registry) -> dict:
    """F+B, the complete policy of revision 2 (steps 1-7), on the original probabilities, never renormalized."""
    out = {"decision": "abstain", "reason": "", "program": None, "answer": None, "score": None, "candidates": [],
           "programs": [], "selected": None, "applications": 0}
    relations = [lab for lab in labels if lab not in (J.UNSUPPORTED, J.AMBIGUOUS)]
    for dist in dists:  # 1. refusal before search
        lab = argmax_label(dist, labels)
        if lab in (J.UNSUPPORTED, J.AMBIGUOUS):
            return {**out, "reason": f"refusal argmax: {lab}"}
    cands = []
    for dist in dists:  # 2. candidates per segment
        ranked = sorted((r for r in relations if dist[labels.index(r)] >= FB_FLOOR),
                        key=lambda r: (-dist[labels.index(r)], r))
        if not ranked:
            return {**out, "reason": "no candidate relation"}
        cands.append(ranked[:FB_K])
    out["candidates"] = cands
    etype = next((t for t in sorted(world.types) if world.types[t].contains(entity)), None)
    programs = []
    for combo in product(*cands):  # 3. programs and their scores
        score = 1.0
        for dist, r in zip(dists, combo):
            score *= dist[labels.index(r)]
        rec = {"program": list(combo), "score": score, "pruned": not (etype and well_typed(combo, etype, world)),
               "answer": None}
        if not rec["pruned"]:  # 4. type pruning, then execution of every survivor
            ex = run_program(combo, entity, world, registry)
            rec["answer"] = ex["answer"]
            out["applications"] += ex["applications"]
        programs.append(rec)
    out["programs"] = programs
    survivors = sorted((i for i, p in enumerate(programs) if not p["pruned"]),
                       key=lambda i: (-programs[i]["score"], programs[i]["program"]))
    if not survivors:  # 5. selection
        return {**out, "reason": "no well-typed candidate"}
    best = survivors[0]
    out["selected"] = best
    sel = programs[best]
    if any(programs[i]["score"] >= FB_RATIO * sel["score"] and programs[i]["answer"] != sel["answer"]
           for i in survivors[1:]):  # 6. a close survivor that disagrees
        return {**out, "reason": "a close candidate gives another answer"}
    for dist, r in zip(dists, sel["program"]):  # 7. the refusal check
        p = dist[labels.index(r)]
        if not (p > dist[labels.index(J.UNSUPPORTED)] and p > dist[labels.index(J.AMBIGUOUS)]):
            return {**out, "reason": "a refusal label outweighs the selected relation"}
    return {**out, "decision": "answer", "program": sel["program"], "answer": sel["answer"], "score": sel["score"]}


# --- correctness, certificates, measurements, criteria ----------------------------------------------------------
def primary_correct(x, row) -> bool:
    return (row["decision"] == "answer" and x["expected"] == "answer" and row.get("program") == x["program"]
            and row.get("entity") == x["entity"] and row["answer"] == x["answer"])


def certificate(cal_items, cal_rows) -> dict:
    keep = [(x, r) for x, r in zip(cal_items, cal_rows) if r["decision"] == "answer"]
    cert = K.fixed_sequence_certify([r["score"] for _, r in keep], [primary_correct(x, r) for x, r in keep], GRID,
                                    ALPHA, DELTA)
    return {**cert, "records": [[x["id"], r["score"], primary_correct(x, r)] for x, r in keep]}


def released(row, policy: str, threshold, deterministic_certified: bool | None = None) -> bool:
    if row["decision"] != "answer":
        return False
    if policy == "none":
        return True
    if deterministic_certified is not None:
        return deterministic_certified
    return threshold is not None and row["score"] >= threshold


def measure(items, rows, policy, threshold, deterministic_certified=None) -> dict:
    groups = defaultdict(list)
    for x, r in zip(items, rows):
        groups[x["category"]].append((x, r))
        if x["category"] == "T1":
            groups[f"T1:{x['program'][0]}"].append((x, r))
        if x["category"] == "S1":
            groups[f"S1:{x['family']}"].append((x, r))
        if "stratum" in x:
            groups[f"{x['category']}:{x['stratum']}"].append((x, r))
    out = {}
    for cat, pairs in sorted(groups.items()):
        flags = [released(r, policy, threshold, deterministic_certified) for _, r in pairs]
        acc = [pr for pr, f in zip(pairs, flags) if f]
        errors = sum(not primary_correct(x, r) for x, r in acc)
        n = len(pairs)
        out[cat] = {"n": n, "answered": len(acc), "errors": errors, "coverage": len(acc) / n,
                    "abstention": 1 - len(acc) / n, "error_upper_95": K.clopper_pearson_upper(errors, len(acc)),
                    "abstention_interval_95": K.exact_interval(n - len(acc), n),
                    "exact_accuracy": sum(primary_correct(x, r) for x, r in acc) / n,
                    "accidental_agreement": sum((not primary_correct(x, r)) and x["expected"] == "answer"
                                                and r["answer"] == x["answer"] for x, r in acc),
                    "reasons": dict(sorted(Counter(r["reason"] if r["decision"] != "answer" else
                                                   ("answered" if f else "threshold")
                                                   for (x, r), f in zip(pairs, flags)).items()))}
    return out


def criteria(certified: bool, cert_cats: dict, none_cats: dict) -> dict:
    answers = lambda k, cov: certified and cert_cats[k]["coverage"] >= cov and cert_cats[k]["error_upper_95"] <= ERROR_BOUND  # noqa: E731
    return {"certified": certified,
            "sanity": answers("ID", ID_COVERAGE) and cert_cats["R"]["abstention"] >= REFUSAL
            and none_cats["T1"]["exact_accuracy"] >= T1_ACCURACY,
            "s2": answers("S2", COVERAGE), "s1_length3": answers("S1:length3", COVERAGE),
            "joint_novelty": answers("S3", COVERAGE) and cert_cats["S4a_primary"]["abstention"] >= REFUSAL
            and cert_cats["S4b_primary"]["abstention"] >= REFUSAL,
            "s2_exact_unthresholded": none_cats["S2"]["exact_accuracy"]}


# --- training cost, as realized ---------------------------------------------------------------------------------
def a_forward_macs(n_tokens: int, d: int = 48, hidden: int = 64, n_out: int = 8, steps: int = J.MAX_STEPS + 1) -> int:
    """MACs of one forward pass of A's interpreter over a question of `n_tokens` words, from its layer shapes:
    the bidirectional encoder GRU, the entity head, the start layer, and per decoder step the attention, the
    GRU cell and the output layer."""
    h2 = 2 * hidden
    encoder = n_tokens * 2 * 3 * (d * hidden + hidden * hidden) + n_tokens * h2
    per_step = h2 * h2 + 2 * n_tokens * h2 + 3 * ((d + h2) * h2 + h2 * h2) + 2 * h2 * n_out
    return encoder + h2 * h2 + steps * per_step


def a_realized_batches(n_examples: int, seed: int, updates: int, batch: int, entity_dropout: float = 0.1) -> list[list[int]]:
    """The batches `interpreters.train` draws: the same generator calls in the same order (an epoch permutation when
    the current one runs out, one dropout draw per example drawn)."""
    gen = torch.Generator().manual_seed(seed)
    order, k, out = [], 0, []
    for _ in range(updates):
        if k + batch > len(order):
            order, k = torch.randperm(n_examples, generator=gen).tolist(), 0
        idx = order[k:k + batch]
        k += batch
        if entity_dropout:
            for _ in idx:
                torch.rand(1, generator=gen)
        out.append(idx)
    return out


def a_realized_draws(n_examples: int, seed: int, updates: int, batch: int, entity_dropout: float = 0.1) -> list[int]:
    """How often `interpreters.train` draws each example."""
    draws = [0] * n_examples
    for idx in a_realized_batches(n_examples, seed, updates, batch, entity_dropout):
        for i in idx:
            draws[i] += 1
    return draws


def a_padded_macs(lengths: list[int], d: int = 48, hidden: int = 64, n_out: int = 8, steps: int = J.MAX_STEPS + 1) -> int:
    """One training batch's forward MACs at its padded width W (its longest question). The encoder is packed, so it
    sees each question's own tokens. The entity head and the attention scores and context run over all W positions
    of every row."""
    h2, B, W_ = 2 * hidden, len(lengths), max(lengths)
    encoder = sum(n * 2 * 3 * (d * hidden + hidden * hidden) for n in lengths)
    per_step = B * (h2 * h2 + 3 * ((d + h2) * h2 + h2 * h2) + 2 * h2 * n_out) + 2 * B * W_ * h2
    return encoder + B * W_ * h2 + B * h2 * h2 + steps * per_step


# --- one unit ---------------------------------------------------------------------------------------------------
def run_unit(world, init: int, regime: str, out_dir: Path) -> dict:
    torch.set_num_threads(4)
    study = FI.study_for(world, world.seed)
    train, aug, _ = FI.regime_training(study, regime, world, world.seed)
    items = [x for x in study["items"] if x["split"] != "train"]
    cal = [x for x in items if x["split"] == "calibration"]
    test = [x for x in items if x["split"] == "test"]
    texts = [x["text"] for x in items]
    registry, entities = G.table_registry(world), FI.entity_names(world)
    vocab_texts = [x["text"] for x in train]
    vocab_words = {w for t in vocab_texts for w in t.split()}
    segs = [SG.segment(t, entities) for t in texts]
    rows, records, costs, training = {}, {}, {}, {}

    # A: the robustness word_aug arm, fitted and decoded exactly as there
    examples = [J.Example(x["text"], tuple(x["outputs"]), x["entity_position"]) for x in train + aug]
    t0 = time.time()
    net, vocab, loss = I.train("word", examples, vocab_texts, tuple(world.relations), init, UPDATES)
    a_seconds = time.time() - t0
    batches = a_realized_batches(len(examples), init, UPDATES, BATCH)
    draws = [0] * len(examples)
    for idx in batches:
        for i in idx:
            draws[i] += 1
    lengths = [len(I.normalize(e.text)) for e in examples]
    training["A"] = {"updates": UPDATES, "batch": BATCH, "presentations": sum(draws), "examples": len(examples),
                     "tokens": sum(d * n for d, n in zip(draws, lengths)),
                     "forward_macs_unpadded": sum(d * a_forward_macs(n) for d, n in zip(draws, lengths)),
                     "forward_macs_padded": sum(a_padded_macs([lengths[i] for i in idx]) for idx in batches),
                     "parameters": sum(p.numel() for p in net.parameters()), "seconds": round(a_seconds, 2),
                     "loss": loss, "draws_by_category": dict(sorted(Counter(
                         x["category"] for x, d in zip(train + aug, draws) for _ in range(d)).items()))}
    t0 = time.time()
    parses = I.interpret(net, vocab, texts)
    a_rows = IR.decide(parses, items, world, registry)
    for row, parse in zip(a_rows, parses):
        row["program"] = list(parse.steps) if parse.outcome == "program" else None
    costs["A"] = {"seconds": round(time.time() - t0, 2), "encoder_passes": len(items),
                  "evaluated_row_steps": len(items) * (J.MAX_STEPS + 1),  # interpret runs every step for every row
                  "emitted_decisions": sum(len(p.step_probabilities) for p in parses)}
    rows["A"] = a_rows
    records["A"] = [{"id": x["id"], "outcome": p.outcome, "steps": list(p.steps), "entity": p.entity,
                     "step_probabilities": list(p.step_probabilities), "entity_probability": p.entity_probability,
                     "score": p.score, "decision": r["decision"], "reason": r["reason"], "answer": r["answer"]}
                    for x, p, r in zip(items, parses, a_rows)]

    # A+K: A's fit, decoded under the grammar's constraints; the grammar's refusals abstain
    t0 = time.time()
    slots = [len(s.segments) if s.ok else None for s in segs]
    ak_stats = {}
    ak = CD.decode(net, vocab, texts, slots, world, stats=ak_stats)
    ak_rows, ak_records, steps_used = [], [], 0
    for x, seg, rec in zip(items, segs, ak):
        if rec is None:
            row = {"decision": "abstain", "reason": f"grammar: {seg.status}", "program": None, "entity": None,
                   "answer": None, "score": None}
            ak_records.append({"id": x["id"], "grammar": seg.status, **{k: row[k] for k in ("decision", "reason", "answer")}})
        else:
            steps_used += len(rec["steps"])
            if rec["outcome"] != "program":
                row = {"decision": "abstain", "reason": rec["outcome"], "program": None, "entity": rec["entity"],
                       "answer": None, "score": rec["score"]}
            else:
                ex = run_program(rec["program"], rec["entity"], world, registry)
                row = {"decision": ex["status"], "reason": ex["reason"], "program": rec["program"],
                       "entity": rec["entity"], "answer": ex["answer"], "score": rec["score"]}
            ak_records.append({"id": x["id"], "grammar": "ok", **rec,
                               **{k: row[k] for k in ("decision", "reason", "answer")}})
        ak_rows.append(row)
    costs["AK"] = {"seconds": round(time.time() - t0, 2), "encoder_passes": sum(s is not None for s in slots),
                   "emitted_decisions": steps_used, "evaluated_row_steps": ak_stats.get("evaluated_row_steps", 0),
                   "grammar_refusals": sum(s is None for s in slots)}
    rows["AK"], records["AK"] = ak_rows, ak_records

    # F: the segment grounder on the aligned presentations (per presentation, multiplicities preserved)
    labels_ = FI.extract_labels(train + aug, world)
    pres = [(c["style"], tuple(c["tokens"]), c["label"]) for c in labels_["contributions"]]
    fit = SGR.train_grounder(pres, vocab, tuple(world.relations), init, UPDATES, BATCH)
    by_tuple = Counter()
    for (style, toks, lab), d in zip(pres, fit["draws"]):
        by_tuple[f"{style}: {' '.join(toks)} -> {lab}"] += d
    training["F"] = {"updates": fit["updates"], "batch": fit["batch"], "presentations": sum(fit["draws"]),
                     "aligned_presentations": len(pres), "tokens": fit["tokens"],
                     "forward_macs": sum(d * SGR.forward_macs(len(t)) for (_, t, _), d in zip(pres, fit["draws"])),
                     "parameters": fit["parameters"], "seconds": fit["seconds"], "loss": fit["loss"],
                     "draws_by_tuple": dict(sorted(by_tuple.items()))}
    t0 = time.time()
    flat = [(s.style, seg) for s in segs if s.ok for seg in s.segments]
    dists_flat = SGR.ground(fit["net"], vocab, flat) if flat else []
    labels = list(fit["labels"])
    f_rows, fb_rows, f_records, fb_records, pos = [], [], [], [], 0
    fb_cost = Counter()
    for x, seg in zip(items, segs):
        if not seg.ok:
            row = {"decision": "abstain", "reason": seg.status, "program": None, "entity": None, "answer": None,
                   "score": None}
            f_rows.append(row)
            fb_rows.append(dict(row))
            f_records.append({"id": x["id"], "grammar": seg.status, "distributions": [], "decision": "abstain",
                              "reason": seg.status, "program": None, "answer": None, "score": None})
            fb_records.append({**f_records[-1], "candidates": [], "programs": [], "selected": None})
            continue
        dists = dists_flat[pos:pos + len(seg.segments)]
        pos += len(seg.segments)
        f = f_greedy(dists, labels, seg.entity, world, registry)
        b = fb_policy(dists, labels, seg.entity, world, registry)
        fb_cost["candidates_proposed"] += len(b["programs"])
        fb_cost["candidates_pruned"] += sum(p["pruned"] for p in b["programs"])
        fb_cost["candidates_executed"] += sum(not p["pruned"] for p in b["programs"])
        fb_cost["module_applications"] += b["applications"]
        f_rows.append({**{k: f[k] for k in ("decision", "reason", "program", "answer", "score")}, "entity": seg.entity})
        fb_rows.append({**{k: b[k] for k in ("decision", "reason", "program", "answer", "score")}, "entity": seg.entity})
        f_records.append({"id": x["id"], "grammar": "ok", "distributions": dists,
                          **{k: f[k] for k in ("decision", "reason", "program", "answer", "score")}})
        fb_records.append({"id": x["id"], "grammar": "ok", "distributions": dists,
                           **{k: b[k] for k in ("decision", "reason", "program", "answer", "score", "candidates",
                                                "programs", "selected")}})
    seconds = round(time.time() - t0, 2)
    n_segments = len(flat)
    costs["F"] = {"segment_passes": n_segments, "module_applications": sum(
        len(r["program"]) for r in f_rows if r["decision"] == "answer")}
    costs["FB"] = {"segment_passes": n_segments, **dict(fb_cost)}
    costs["F_and_FB_combined"] = {
        "seconds": seconds,
        "boundary": "from after F's fit to the end of both policies: the grounding pass over every admitted segment, "
                    "F's and F+B's decisions and their records. The grammar segmentation of every question ran once "
                    "before A's fit and is outside every system's timing; L, G and W re-segment inside their own"}
    rows["F"], rows["FB"], records["F"], records["FB"] = f_rows, fb_rows, f_records, fb_records

    # the no-fit controls, on the same items (deterministic; identical across inits)
    vocab_set = FI.training_vocabulary(train)
    dicts = {rep: FI.induce_dictionary(labels_["contributions"], rep, vocab_set) for rep in FI.REPRESENTATIONS}
    lookups = {"L_raw": (dicts["raw"], "raw"), "L_vocab": (dicts["vocab"], "vocab"), "G": (FI.reference_lexicon(), "raw"),
               "W": (FI.word_label_sets(labels_["contributions"]), "words")}
    for name, (lookup, rep) in lookups.items():
        t0 = time.time()
        rows[name] = [{**FI.decide(x["text"], world, registry, lookup, rep, vocab_set, entities), "score": None}
                      for x in items]
        costs[name] = {"seconds": round(time.time() - t0, 2), "segment_lookups": sum(
            len(s.segments) for s in segs if s.ok), "module_applications": sum(
            len(r["program"]) for r in rows[name] if r["decision"] == "answer")}

    # certificates, measurements and criteria on the same calibration items and endpoint
    n_cal = len(cal)
    systems = {}
    for name in NEURAL + CONTROLS:
        rr = rows[name]
        cal_rows, test_rows = rr[:n_cal], rr[n_cal:]
        if name in NEURAL:
            cert = certificate(cal, cal_rows)
            threshold = cert["threshold"] if cert["certified"] else None
            cert_cats = measure(test, test_rows, "certified", threshold)
            certified = bool(cert["certified"])
        else:
            risk = FI.calibration_risk(items, rr)
            cert = {k: v for k, v in risk.items()}
            certified = bool(risk["certified"])
            cert_cats = measure(test, test_rows, "certified", None, certified)
        none_cats = measure(test, test_rows, "none", None)
        systems[name] = {"certificate": cert, "certified": cert_cats, "none": none_cats,
                         "criteria": criteria(certified, cert_cats, none_cats)}

    # A's historical answer-based certificate and criteria (the robustness study's code, unchanged), for P1
    a_cal, a_test = rows["A"][:n_cal], rows["A"][n_cal:]
    hist_cert = IR.certify(cal, a_cal, "certified", vocab_words)
    h_threshold = hist_cert["threshold"]
    hist = {"certificate": {k: v for k, v in hist_cert.items()},
            "certified": IR.measure(test, a_test, "certified", h_threshold, vocab_words),
            "none": IR.measure(test, a_test, "none", None, vocab_words)}
    hist_unit = IR.unit_outcomes({"policies": {"certified": {"certificate": hist["certificate"], "categories": hist["certified"]},
                                               "none": {"categories": hist["none"]}}})
    hist["criteria"] = hist_unit
    hist["p1"] = {"sanity_and_t1": hist_unit["sanity"] and hist_unit["taught_wording_learned"],
                  "s2_failure_replicated": (hist_unit["sanity"] and hist_unit["taught_wording_learned"]
                                            and not hist_unit["s2_transfer"]),
                  "s1_failure_replicated": (hist_unit["sanity"] and hist_unit["taught_wording_learned"]
                                            and not hist_unit["s1_structure"])}
    p5 = {name: systems[name]["none"]["S2"]["exact_accuracy"] - systems["G"]["none"]["S2"]["exact_accuracy"]
          for name in ("F", "FB", "L_raw", "L_vocab", "W")}
    unit = {"world": world.seed, "init": init, "regime": regime, "n_calibration": n_cal, "n_test": len(test),
            "systems": systems, "historical_A": hist, "p5_s2_exact_minus_G": p5, "training": training,
            "evaluation_costs": costs}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "unit.json").write_text(json.dumps(unit, indent=1, sort_keys=True, default=str))
    for name in NEURAL:
        with gzip.open(out_dir / f"records_{name}.json.gz", "wt") as fh:
            json.dump(records[name], fh)
    return unit


def provenance() -> dict:
    here = Path(__file__).resolve().parent
    sha = {s: hashlib.sha256((here / s).read_bytes()).hexdigest() for s in SOURCES}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=here, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", str(here.parent)], cwd=here,
                                             text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"commit": commit, "research_tree_dirty": dirty, "sha256": sha, "torch": torch.__version__,
            "python": sys.version.split()[0], "threads": torch.get_num_threads()}


def config(countries: int = FI.COUNTRIES) -> dict:
    return {"pilot": PILOT_VERSION, **FI.config(countries), "updates": UPDATES, "batch": BATCH, "grid": list(GRID),
            "fb": {"k": FB_K, "floor": FB_FLOOR, "ratio": FB_RATIO},
            "ak_selection": "allowed_logits", "accounting": "v2"}


def run(worlds, inits, out: Path, countries: int = FI.COUNTRIES) -> Path:
    if any(w in FI.CONFIRMATION_WORLDS for w in worlds):
        raise SystemExit("worlds 1000-1002 are reserved for the confirmation; this protocol is not frozen")
    run_dir = out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    meta = {"pilot": PILOT_VERSION, "provenance": provenance(), "config": config(countries),
            "requested": {"worlds": list(worlds), "inits": list(inits), "regimes": list(FI.REGIMES)}, "units": []}
    for w in worlds:
        world = W.generate_world(w, n_countries=countries)
        for init in inits:
            for regime in FI.REGIMES:
                path = f"units/w{w}_i{init}_{regime}"
                t0 = time.time()
                unit = run_unit(world, init, regime, run_dir / path)
                s = unit["systems"]
                meta["units"].append({"world": w, "init": init, "regime": regime, "path": path,
                                      "seconds": round(time.time() - t0, 1)})
                print(f"w{w} i{init} {regime:8s} ({time.time() - t0:.0f}s) " + " | ".join(
                    f"{n}: cert {s[n]['certificate'].get('threshold', '-') if n in NEURAL else s[n]['certificate']['certified']} "
                    f"S2 {s[n]['none']['S2']['exact_accuracy']:.3f} S1 {s[n]['none']['S1:length3']['exact_accuracy']:.3f} "
                    f"S4a {s[n]['certified']['S4a_primary']['abstention']:.3f}" for n in NEURAL), flush=True)
    meta["finished"] = datetime.now(timezone.utc).isoformat()
    (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
    return run_dir


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--worlds", type=int, nargs="+", default=list(FI.DEVELOPMENT_WORLDS))
    p.add_argument("--inits", type=int, nargs="+", default=list(DEV_INITS))
    p.add_argument("--countries", type=int, default=FI.COUNTRIES)
    p.add_argument("--out", type=Path, default=Path("runs/titnn_factored_pilot"))
    a = p.parse_args(argv)
    print(run(a.worlds, a.inits, a.out, a.countries))


if __name__ == "__main__":
    main()
