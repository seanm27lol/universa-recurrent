"""Jev as a planning policy, calibration and the question benchmark (integration plan PR 6).

Software checks on small worlds; the frozen experiment is protocols/typed_routing_v1.md.
"""
from __future__ import annotations

import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
from integrated import calibration as K  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402


def test_clopper_pearson_matches_known_values():
    assert K.clopper_pearson_upper(0, 299) == pytest.approx(1 - 0.05 ** (1 / 299), abs=1e-6)  # just under 1%
    assert K.clopper_pearson_upper(0, 298) > 0.01
    assert K.clopper_pearson_upper(1, 10) == pytest.approx(0.3942, abs=1e-3)
    assert K.clopper_pearson_upper(0, 0) == 1.0 and K.clopper_pearson_upper(5, 5) == 1.0
    assert K.clopper_pearson_upper(2, 100) < K.clopper_pearson_upper(3, 100)


def test_platt_fit_recovers_a_transform_and_stays_finite_when_separable():
    rng = random.Random(0)
    scores, labels = [], []
    for _ in range(4000):
        s = rng.uniform(0.02, 0.98)
        scores.append(s)
        labels.append(rng.random() < 1 / (1 + math.exp(-(2 * math.log(s / (1 - s)) - 1))))
    a, b = K.platt_fit(scores, labels)
    assert a == pytest.approx(2, abs=0.25) and b == pytest.approx(-1, abs=0.25)
    a, b = K.platt_fit([0.1] * 50 + [0.9] * 50, [False] * 50 + [True] * 50)  # perfectly separable
    assert math.isfinite(a) and math.isfinite(b) and abs(a) < 50 and abs(b) < 50
    assert K.platt_fit([0.9] * 20, [True] * 20) != (math.inf, math.inf)


def test_threshold_choice_bounds_the_calibration_risk():
    probs = [0.99] * 400 + [0.6] * 100
    correct = [True] * 400 + [i % 2 == 0 for i in range(100)]
    assert K.choose_threshold(probs, correct, 0.01, 0.05) == 0.99  # the uncertain tier would break the bound
    assert K.choose_threshold([0.9] * 100, [True] * 100, 0.01, 0.05) == math.inf  # too few to certify 1%
    assert K.choose_threshold([], [], 0.01, 0.05) == math.inf


def test_reliability_and_binding():
    r = K.reliability([0.25] * 4 + [0.75] * 4, [True, False, False, False, True, True, True, False])
    assert r["ece"] == pytest.approx(0.0) and r["n"] == 8
    cal = K.fit([0.9] * 400, [True] * 400, [str(i) for i in range(400)], "interp", "reg", "policy")
    assert cal.threshold < 1 and cal.binding_problems("interp", "reg", "policy") == []
    assert len(cal.binding_problems("other", "reg2", "policy")) == 2


spec = importlib.util.spec_from_file_location("ti_tnn_typed_questions", ROOT / "bench" / "typed_questions.py")


def bench():
    pytest.importorskip("torch")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_world():
    return W.generate_world(7, n_countries=8, cities_per_country=(2, 3), n_operators=3)


def test_questions_are_unambiguous_split_by_instance_and_answered_by_the_reference():
    Q = bench()
    world = small_world()
    items = Q.make_items(world, 7)
    from integrated import reference as R
    seen = {}
    for x in items:
        toks = x["text"].split()
        assert toks.count(x["entity"]) == 1 and toks[x["entity_position"]] == x["entity"]
        if x["kind"] == "supported":
            assert x["answer"] == R.reference_answer(world, tuple(x["program"]), x["entity"])
            assert toks.count("'s") in (0, x["length"])  # one style per question: all possessive or none
            if x["split"] != "shift":
                seen.setdefault((tuple(x["program"]), x["entity"]), set()).add(x["split"])
        else:
            assert x["expected"] == "abstain"
    assert all(len(s) == 1 for s in seen.values())  # no instance in two splits
    shift = [x for x in items if x["split"] == "shift"]
    assert shift and all("'s" not in x["text"].split() for x in shift)
    assert {x["kind"] for x in items} == {"supported"} | set(Q.INVALID)


def test_the_policy_abstains_with_reasons_and_answers_through_the_runtime():
    Q = bench()
    from integrated import routing as J
    world = small_world()
    table = G.table_registry(world)
    country = world.types["Country"].values[0]
    city = world.types["City"].values[0]
    ok = J.propose(J.Parse("program", ("capital_of", "airport_of"), country, 0.9, (0.9, 1.0, 1.0), 1.0), world.types, table)
    assert ok["decision"] == "answer" and ok["answer"] == world.facts["airport_of"][world.facts["capital_of"][country]]
    ill = J.propose(J.Parse("program", ("capital_of",), city, 0.9, (0.9, 1.0), 1.0), world.types, table)
    assert ill["decision"] == "abstain" and ill["reason"] == "ill-typed program"
    assert J.propose(J.Parse("program", ("capital_of",), "country_99", 0.9, (), 1.0), world.types, table)["reason"] == "unknown entity"
    assert J.propose(J.Parse("unsupported", (), city, 0.9, (), 1.0), world.types, table)["reason"] == "unsupported"


def test_the_interpreter_learns_controlled_language_on_a_small_world():
    Q = bench()
    from integrated import routing as J
    world = small_world()
    items = Q.make_items(world, 7)
    train = [x for x in items if x["split"] == "train"]
    net, vocab, _ = J.train_interpreter([J.Example(x["text"], Q.outputs_of(x), x["entity_position"]) for x in train],
                                        tuple(world.relations), 0, epochs=40)
    test = [x for x in items if x["split"] == "test"]
    parses = J.interpret(net, vocab, [x["text"] for x in test])
    sup = [(x, p) for x, p in zip(test, parses) if x["kind"] == "supported"]
    exact = sum(p.outcome == "program" and list(p.steps) == x["program"] and p.entity == x["entity"] for x, p in sup)
    assert exact / len(sup) > 0.9
    assert all(0 < p.score <= 1 for p in parses)
    assert len(J.interpreter_identity(net, vocab)) == 64


def fake(system, world, init, **over):
    fam = lambda n, c: {"n": n, "answered": n, "errors": 0, "correct_no_threshold": c}
    split = {"interpreter_exact": 1.0, "accepted_error_upper_95": 0.004, "coverage_supported": 0.99,
             "accuracy_supported_no_threshold": 1.0, "supplied_program_accuracy": 1.0, "answered": 800,
             "accepted_errors": 0, "accepted_error": 0.0,
             "abstention_by_kind": {k: {"n": 50, "abstained": 50} for k in ("ill_typed", "unsupported", "unknown_entity", "ambiguous")},
             "by_family": {"length1": fam(200, 200), "length2": fam(200, 200 if system != "answer_only" else 30),
                           "length3": fam(200, 200 if system != "answer_only" else 40), "round_trip": fam(200, 180)}}
    r = {"system": system, "world": world, "init": init, "test": dict(split), "shift": dict(split, answered=10, accepted_error=0.0)}
    for k, v in over.items():
        r[k] = v
    return r


def test_the_frozen_gate():
    Q = bench()
    runs = [fake(s, w, i) for s in Q.SYSTEMS for w in (400, 401, 402) for i in range(5)]
    v = Q.evaluate({"runs": runs})
    assert v["complete"] and all(v["predictions"].values()), v["predictions"]
    bad_shift = [fake("typed_specialists", 401, i, shift={**runs[0]["shift"], "accepted_error": 0.3}) if i < 2
                 else fake("typed_specialists", 401, i) for i in range(5)]
    v = Q.evaluate({"runs": [r for r in runs if (r["system"], r["world"]) != ("typed_specialists", 401)] + bad_shift})
    assert v["predictions"]["6_shift_accepted_error_bounded"] is False
    assert v["predictions"]["2_selective_answering"] is True
    assert Q.evaluate({"runs": runs[1:]})["complete"] is False
