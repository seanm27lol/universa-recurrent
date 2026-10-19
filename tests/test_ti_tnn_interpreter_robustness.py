"""The interpreter robustness study's machinery (protocols/interpreter_robustness_v1.md, DRAFT): generator audits,
the character input contract, the grammar control and fixed-sequence certification. No training results."""
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
from integrated import relations as W  # noqa: E402

torch = pytest.importorskip("torch")
sys.path.insert(0, str(ROOT / "bench"))
spec = importlib.util.spec_from_file_location("ti_tnn_interpreter_robustness", ROOT / "bench" / "interpreter_robustness.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
from integrated import interpreters as I  # noqa: E402
from integrated import registry as G  # noqa: E402


@pytest.fixture(scope="module")
def study():
    world = W.generate_world(7, n_countries=10, cities_per_country=(2, 3), n_operators=3)
    st = S.make_study(world, 7, n_noncancel=60, n_round=20, n_diag=30)
    return world, st, S.augment(st, 7)


def test_the_audit_passes_and_catches_a_three_step_training_target(study):
    world, st, aug = study
    checks = S.audit(st, aug)
    assert all(checks.values()), checks
    leaked = dict(next(x for x in st["items"] if x["category"] == "train_base"))
    leaked["outputs"] = ["capital_of", "airport_of", "operator_of", "<end>"]
    assert not S.audit({**st, "items": st["items"] + [leaked]}, aug)["training_targets_at_most_two_relations"]
    taught = dict(next(x for x in st["items"] if x["category"] == "S2"), split="train", outputs=["x", "<end>"])
    assert not S.audit({**st, "items": st["items"] + [taught]}, aug)["taught_wording_only_in_one_step_training"]


def test_matched_variants_differ_only_where_declared(study):
    world, st, _ = study
    groups = {}
    for x in st["items"]:
        if x["group"]:
            groups.setdefault(x["group"], {})[x["category"]] = x
    quads = [g for g in groups.values() if "S4a" in g]
    assert quads
    for g in quads:
        assert len({(tuple(g[c]["program"]), g[c]["entity"]) for c in ("Q_base", "S3", "S4a", "S4b")}) == 1
        assert g["Q_base"]["expected"] == g["S3"]["expected"] == "answer" and g["S4a"]["expected"] == g["S4b"]["expected"] == "abstain"
        base, s4a = g["Q_base"]["text"].split(), g["S4a"]["text"].split()
        assert base[:2] == s4a[:2] or base[0] == s4a[0]  # same familiar frame
    for g in groups.values():
        if "S2" in g:
            assert g["S2"]["program"] == g["S2_base"]["program"] and len(g["S2"]["program"]) == 2
            assert g["S2"]["text"].split()[0] == g["S2_base"]["text"].split()[0]
    assert all(len(x["program"]) == 1 for x in st["items"] if x["category"] in ("T1", "train_taught"))
    assert all(len(x["program"]) == 3 for x in st["items"] if x["category"] in ("S1", "S2_L3", "S3_L3", "S4_L3", "L3_base"))


def test_the_character_contract_never_maps_characters_to_a_shared_symbol():
    assert I.normalize("What is the capital of country_3 ?") == ["what", "is", "the", "capital", "of", "country_3", "?"]
    for bad in ("what is the capital of countrÿ_3 ?", "x" * 49, " ".join(["a"] * 41)):
        with pytest.raises(I.UnsupportedInput):
            I.normalize(bad)
    b = I.make_batch("char", [["abc", "ab"], ["abd"]], None)
    assert b.chars.shape == (2, 2, 3) and bool(b.mask[1, 1]) is False
    assert not torch.equal(b.chars[0, 0], b.chars[1, 0])  # distinct spellings, distinct inputs


def test_the_grammar_control_refuses_unparsed_relation_wording_but_not_unfamiliar_wrappers(study):
    world, st, _ = study
    g = S.grammar_control()
    country = world.types["Country"].values[0]
    assert g.parse(f"could you tell me the capital of {country} ?").steps == ("capital_of",)
    assert g.parse(f"what is exactly the airport of the capital city of {country} ?").steps == ("capital_of", "airport_of")
    assert g.parse(f"name {country} 's capital 's airport .").steps == ("capital_of", "airport_of")
    for refused in (f"what is the metropolis of {country} ?", f"what is the airport of the metropolis of {country} ?",
                    f"what is the metropolis of the capital of {country} ?", f"name {country} 's metropolis ."):
        assert g.parse(refused).outcome == "unsupported", refused
    assert g.parse(f"what is the administrative centre of {country} ?").steps == ("capital_of",)  # taught wording


def test_fixed_sequence_certification_stops_at_the_first_unrejected_rule():
    scores = [0.999] * 400 + [0.9] * 400 + [0.5] * 100
    correct = [True] * 400 + [True] * 399 + [False] + [i % 2 == 0 for i in range(100)]
    cert = K.fixed_sequence_certify(scores, correct, (0.999, 0.9, 0.5, 0.0))
    assert cert["certified"] and cert["threshold"] == 0.9
    assert [s["rejected"] for s in cert["steps"]] == [True, True, False]
    assert K.fixed_sequence_certify([0.9] * 100, [True] * 100, (0.9, 0.0))["certified"] is False  # too few to certify
    with pytest.raises(ValueError):
        K.fixed_sequence_certify(scores, correct, (0.0, 0.9))
    assert [next(n for n in range(1, 900) if K.binomial_p_value(e, n, 0.01) <= 0.05) for e in range(4)] == [299, 473, 628, 773]
    assert K.certification_power(0.005, 374) == pytest.approx(0.153, abs=0.001)


def test_platt_for_reporting_never_reverses_the_order():
    a, b = K.platt_fit_increasing([0.9, 0.8, 0.7, 0.6] * 25, [False, False, True, True] * 25)
    assert a > 0
    assert K.platt_probability(0.9, a, b) > K.platt_probability(0.6, a, b)


def test_a_tiny_run_produces_every_category_policy_and_record(study):
    world, st, aug = study
    args = S.parse_args(["--updates", "20"])
    unit = S.run_unit(world, st, aug, 0, args)
    assert set(unit["arms"]) == set(S.ARMS)
    for arm in unit["arms"].values():
        assert set(arm["policies"]) == set(S.POLICIES)
        cats = arm["policies"]["none"]["categories"]
        assert set(S.ANSWER_CATEGORIES + S.ABSTAIN_CATEGORIES) <= set(cats)
        assert len(arm["records"]) == sum(x["split"] == "test" for x in st["items"])
        assert arm["policies"]["certified"]["certificate"]["steps"]
    assert unit["grammar"]["categories"]["S4a"]["abstention"] == 1.0


def test_s4_strata_follow_the_complete_typed_program():
    world = W.generate_world(100, n_countries=96)
    country = world.types["Country"].values[0]
    city = world.facts["capital_of"][country]
    # "the operator of the aerodrome near <city>": a city allows two relations, but only airport_of fits operator_of
    assert S.s4_stratum(world, ("airport_of", "operator_of"), city, 0)[0] == "type_determined"
    # capital or largest city, then country_of: two programs, one answer (the country itself)
    assert S.s4_stratum(world, ("capital_of", "country_of"), country, 0)[0] == "program_ambiguous_one_answer"
    differ = next(c for c in world.types["Country"].values
                  if world.facts["capital_of"][c] != world.facts["largest_city_of"][c])
    assert S.s4_stratum(world, ("capital_of", "airport_of"), differ, 0)[0] == "answer_ambiguous"


def test_s1_has_its_declared_non_cancelling_count_and_matched_diagnostics(study):
    world, st, _ = study
    s1 = [x for x in st["items"] if x["category"] == "S1"]
    noncancel = {(tuple(x["program"]), x["entity"]) for x in s1 if x["family"] == "length3"}
    diag = {(tuple(x["program"]), x["entity"]) for x in st["items"] if x["category"] == "L3_base"}
    assert diag and diag <= noncancel  # diagnostics are variants of S1 non-cancelling instances
    assert all("stratum" in x for x in st["items"] if x["category"] in ("S4a", "S4b", "S4_L3"))
