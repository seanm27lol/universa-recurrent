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


SMALL = {"calibration": 80, "id_test": 80, "T1": 20, "S2": 20, "S3": 20, "S4_population": 20, "S4_primary": 20,
         "S1_length3": 20, "S1_round_trip": 10, "L3_diagnostic": 10}


@pytest.fixture(scope="module")
def study():
    world = W.generate_world(7, n_countries=10, cities_per_country=(2, 3), n_operators=3)
    st = S.make_study(world, 7, SMALL)
    return world, st, S.augment(st, 7)


def test_the_audit_passes_and_catches_a_three_step_training_target(study):
    world, st, aug = study
    checks = S.audit(world, st, aug)
    assert all(checks.values()), checks
    leaked = dict(next(x for x in st["items"] if x["category"] == "train_base"))
    leaked["outputs"] = ["capital_of", "airport_of", "operator_of", "<end>"]
    assert not S.audit(world, {**st, "items": st["items"] + [leaked]}, aug)["training_targets_at_most_two_relations"]
    taught = dict(next(x for x in st["items"] if x["category"] == "S2"), split="train", outputs=["x", "<end>"])
    assert not S.audit(world, {**st, "items": st["items"] + [taught]}, aug)["taught_wording_only_in_one_step_training"]


def test_the_audit_enforces_the_declared_inventory(study):
    world, st, aug = study
    drop = lambda pred: {**st, "items": [x for x in st["items"] if not pred(x)]}
    assert not S.audit(world, drop(lambda x: x["category"] == "S1"), aug)["declared_inventory"]  # the reported gap
    one = next(x for x in st["items"] if x["category"] == "S4b_primary")
    broken = drop(lambda x: x is one)
    checks = S.audit(world, broken, aug)
    assert not checks["declared_inventory"] and not checks["variant_groups_complete_and_matched"]
    relabeled = {**st, "items": [dict(x, stratum="type_determined") if x is one else x for x in st["items"]]}
    checks = S.audit(world, relabeled, aug)
    assert not checks["s4_strata_recomputed"] and not checks["s4_primary_is_answer_ambiguous"]
    dup = {**st, "items": st["items"] + [dict(st["items"][-1])]}
    assert not S.audit(world, dup, aug)["unique_draw_ids"]
    p, ent = st["train_instances"][0]
    leak = dict(next(x for x in st["items"] if x["category"] == "ID"), program=list(p), entity=ent)
    assert not S.audit(world, {**st, "items": st["items"] + [leak]}, aug)["evaluation_instances_not_in_training"]


def test_independent_draws_may_repeat_strings_but_never_ids(study):
    world, st, _ = study
    held = [x for x in st["items"] if x["split"] != "train"]
    assert len({x["id"] for x in held}) == len(held)
    big = S.make_study(world, 7, {**SMALL, "calibration": 2000})  # with replacement, repeats are expected
    cal = [x["text"] for x in big["items"] if x["split"] == "calibration"]
    assert len(set(cal)) < len(cal)


def test_matched_variants_differ_only_where_declared(study):
    world, st, aug = study
    assert S.audit(world, st, aug)["variant_groups_complete_and_matched"]
    assert all(len(x["program"]) == 1 for x in st["items"] if x["category"] in ("T1", "train_taught"))
    assert all(len(x["program"]) == 2 for x in st["items"] if x["category"] in ("S2", "S2_base"))
    assert all(len(x["program"]) == 3 for x in st["items"] if x["category"] in ("S1", "S2_L3", "S3_L3", "S4_L3", "L3_base"))
    assert all(x["expected"] == "abstain" for x in st["items"] if x["category"] in ("S4a", "S4b", "S4a_primary", "S4b_primary"))


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


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory, monkeypatch_module):
    monkeypatch_module.setattr(S, "SIZES", {**S.SIZES, **SMALL})
    out = tmp_path_factory.mktemp("robustness")
    args = S.parse_args(["--worlds", "7", "--inits", "0", "--countries", "8", "--updates", "20"])
    result = S.run(args, out)
    (out / "interpreter_robustness.json").write_text(__import__("json").dumps(result))
    verdict = S.evaluate(result, worlds=(7,), inits=(0,))
    return out, result


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def verifier():
    spec_v = importlib.util.spec_from_file_location("ti_tnn_robustness_verify", ROOT / "bench" / "interpreter_robustness_verify.py")
    V = importlib.util.module_from_spec(spec_v)
    spec_v.loader.exec_module(V)
    return V


def test_the_verifier_rebuilds_a_run_and_rejects_tampering(tiny_run, tmp_path):
    import gzip
    import json
    import shutil
    out, result = tiny_run
    V = verifier()
    V.S.SIZES = S.SIZES  # the run used the reduced sizes
    assert V.verify(out)["verified"]

    def tampered(edit_result=None, edit_records=None):
        d = tmp_path / f"t{len(list(tmp_path.iterdir()))}"
        shutil.copytree(out, d)
        if edit_result:
            r = json.loads((d / "interpreter_robustness.json").read_text())
            edit_result(r)
            (d / "interpreter_robustness.json").write_text(json.dumps(r))
        if edit_records:
            p = d / "records" / "w7_i0.json.gz"
            with gzip.open(p, "rt") as f:
                rec = json.load(f)
            edit_records(rec)
            with gzip.open(p, "wt") as f:
                json.dump(rec, f)
        return d

    def flip_answer(rec):  # a produced answer changes: correctness and every summary built on it must disagree
        arm = rec["arms"]["word"]
        i = next(k for k, a in enumerate(arm["decision"]) if a)
        arm["answer"][i] = "not_a_real_answer"
    with pytest.raises(V.VerificationError):
        V.verify(tampered(edit_records=flip_answer))

    def summary(r):
        r["units"][0]["arms"]["word"]["policies"]["none"]["categories"]["S3"]["errors"] += 1
    with pytest.raises(V.VerificationError):
        V.verify(tampered(edit_result=summary))

    def certificate_steps(r):
        r["units"][0]["arms"]["char"]["policies"]["certified"]["certificate"]["steps"][0]["accepted"] += 1
    with pytest.raises(V.VerificationError):
        V.verify(tampered(edit_result=certificate_steps))
    gone = tampered()
    (gone / "records" / "w7_i0.json.gz").unlink()
    with pytest.raises(V.VerificationError):
        V.verify(gone)


def fake_unit(world, init, ok):
    """Per-unit summaries in which `ok` names the criteria that pass (S3, S4a, S4b, cert)."""
    good, bad = {"coverage": 1.0, "error_upper_95": 0.003, "abstention": 1.0}, {"coverage": 0.5, "error_upper_95": 0.2, "abstention": 0.5}
    cats = lambda: {"ID": dict(good), "R": dict(good), "S2": dict(bad), "S1:length3": dict(bad),
                    "S3": dict(good if "S3" in ok else bad), "S4a_primary": dict(good if "S4a" in ok else bad),
                    "S4b_primary": dict(good if "S4b" in ok else bad),
                    "T1": {"n": 10, "answered": 10, "errors": 0}}
    arm = {"policies": {"certified": {"certificate": {"certified": "cert" in ok}, "categories": cats()},
                        "none": {"categories": cats()}}}
    return {"world": world, "init": init, "arms": {a: arm for a in S.ARMS}}


def test_the_joint_rule_needs_one_unit_to_pass_everything():
    full = {"S3", "S4a", "S4b", "cert"}
    units = [fake_unit(w, i, full) for w in (700, 701, 702) for i in range(5)]
    assert S.evaluate({"units": units})["predictions"]["3_word_aug_joint_novelty"]
    split = [fake_unit(700, 0, {"S3", "S4b", "cert"}), fake_unit(700, 1, {"S4a", "S4b", "cert"}),
             fake_unit(700, 2, full), fake_unit(700, 3, full), fake_unit(700, 4, {"S3", "S4a", "cert"})]
    v = S.evaluate({"units": split + units[5:]})  # each criterion passes on 4 inits, jointly on only 2
    assert v["predictions"]["3_word_aug_joint_novelty"] is False
    uncertified = [fake_unit(701, i, {"S3", "S4a", "S4b"} | ({"cert"} if i > 1 else set())) for i in range(5)]
    v = S.evaluate({"units": units[:5] + uncertified + units[10:]})
    assert v["predictions"]["3_word_aug_joint_novelty"] is False and v["successes_by_world"]["certified"]["word_aug"][701] == 3
    assert S.evaluate({"units": units[1:]})["complete"] is False
