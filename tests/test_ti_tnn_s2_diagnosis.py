"""The S2 diagnosis (protocols/s2_diagnosis_brief.md): sequence classification, execution order, S2/S2_base
matching, the separation of the training interventions, teacher-forced failure kinds and the decision rule.
Small synthetic worlds and fits of a few updates only; no claim rests on these fits."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
torch = pytest.importorskip("torch")
import interpreter_robustness as S  # noqa: E402
import s2_development as DV  # noqa: E402
import s2_development_analysis as DA  # noqa: E402
import s2_diagnosis as D2  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import routing as J  # noqa: E402

SMALL = {"calibration": 80, "id_test": 80, "T1": 20, "S2": 20, "S3": 20, "S4_population": 20, "S4_primary": 20,
         "S1_length3": 20, "S1_round_trip": 10, "L3_diagnostic": 10}
BUNDLE = Path(__file__).resolve().parents[1] / "experiments" / "results" / "ti_tnn"


@pytest.fixture(scope="module")
def small():
    world = W.generate_world(7, n_countries=16)
    study = S.make_study(world, 7, SMALL)
    return world, study, S.augment(study, 7)


# --- sequence classification -----------------------------------------------------------------------------------
def test_program_differences_are_classified_in_execution_order():
    e = ["country_of", "capital_of"]
    assert D2.compare_programs(e, e) == ("exact",)
    assert D2.compare_programs(e, ["country_of"]) == ("dropped", 1, "capital_of")
    assert D2.compare_programs(e, ["capital_of"]) == ("dropped", 0, "country_of")
    assert D2.compare_programs(e, ["country_of", "largest_city_of"]) == ("substituted", 1, "capital_of",
                                                                          "largest_city_of")
    assert D2.compare_programs(e, ["capital_of", "country_of"]) == ("reordered",)
    assert D2.compare_programs(e, ["country_of", "capital_of", "airport_of"]) == ("extra", 2, "airport_of")
    assert D2.compare_programs(e, ["airport_of", "operator_of"]) == ("other",)
    kind, found = D2.compare_programs(["a", "a"], ["a", "a", "a"])  # three ways to add one "a"
    assert kind == "ambiguous" and len(found) == 3


def test_classes_separate_refusals_entity_errors_and_taught_positions():
    item = {"program": ["country_of", "capital_of"], "entity": "c1"}
    base = {"proposed": None, "proposed_entity": "c1", "reason": "answered", "entity_right": True}
    assert D2.classify(item, {**base, "reason": "unsupported", "proposed": []}, 0)["label"] == "refusal:unsupported"
    assert D2.classify(item, {**base, "reason": "unsupported characters"}, 0)["label"] == "refusal:unsupported characters"
    right = {**base, "proposed": ["country_of", "capital_of"]}
    assert D2.classify(item, right, 0)["label"] == "exact"
    assert D2.classify(item, {**right, "entity_right": False}, 0)["label"] == "entity_only"
    dropped_second = {**base, "proposed": ["country_of"]}
    assert D2.classify(item, dropped_second, 0)["label"] == "dropped:other"  # taught first, the other dropped
    assert D2.classify(item, dropped_second, 1)["label"] == "dropped:taught"  # taught second and dropped
    # a complete program the runtime rejects keeps its program class
    assert D2.classify(item, {**base, "reason": "ill-typed program", "proposed": ["capital_of"]}, 0)["label"] == \
        "dropped:taught"


# --- execution order and matching ------------------------------------------------------------------------------
def test_the_taught_position_is_recovered_by_re_rendering(small):
    world, study, _ = small
    city = world.types["City"].values[0]
    base = {"id": "b", "group": "g", "program": ["country_of", "capital_of"], "entity": city, "frame": "what is {q} ?",
            "text": f"what is the capital of the country of {city} ?"}
    first = {**base, "id": "s", "text": f"what is the capital of the nation of {city} ?"}  # country_of applies first
    second = {**base, "id": "s", "text": f"what is the administrative centre of the country of {city} ?"}
    assert D2.taught_position(first, base) == 0 and D2.taught_position(second, base) == 1
    with pytest.raises(ValueError):
        D2.taught_position({**first, "frame": "name {q} ."}, base)  # not a matched pair
    with pytest.raises(ValueError):
        D2.taught_position({**first, "text": f"what is the capital of the country of {city} ?"}, base)  # no taught phrase
    pairs = D2.pairs_of([x for x in study["items"] if x["split"] != "train"])
    assert len(pairs) == SMALL["S2"]
    for s2, b, j in pairs:
        assert (s2["program"], s2["entity"], s2["frame"], s2["group"]) == (b["program"], b["entity"], b["frame"],
                                                                         b["group"])
        # the phrase applied first is the one next to the entity in a prefix question
        before_entity = s2["text"].split(s2["entity"])[0]
        assert before_entity.endswith(S.TAUGHT[s2["program"][0]].replace("{x}", "")) == (j == 0)


def test_matching_rejects_missing_or_duplicated_pairs(small):
    _, study, _ = small
    held = [x for x in study["items"] if x["split"] != "train"]
    one = next(x for x in held if x["category"] == "S2_base")
    with pytest.raises(ValueError):
        D2.pairs_of([x for x in held if x is not one])
    with pytest.raises(ValueError):
        D2.pairs_of(held + [dict(one, id="dup")])


# --- the interventions -------------------------------------------------------------------------------------------
def test_the_interventions_differ_only_as_declared(small):
    world, study, aug = small
    adds = {a: DV.additions(world, study, a) for a in DV.ARMS}
    checks = DV.audit_additions(world, study, aug, adds)
    assert all(checks.values()), checks
    taught = [x for x in study["items"] if x["category"] == "train_taught"]
    assert not adds["A"] and all(len(adds[a]) == len(taught) for a in "BCD")
    assert [x["text"] for x in adds["B"]] == [x["text"] for x in taught]
    for b, c in zip(adds["B"], adds["C"]):  # the same questions, longer, with the same one-relation target
        assert (b["program"], b["entity"], b["outputs"]) == (c["program"], c["entity"], c["outputs"])
        extra = c["text"].split()
        for w in b["text"].split():
            extra.remove(w)
        assert len(extra) == c["filler_words"] and set(extra) <= set(DV.WRAPPER_WORDS)
        assert S.TAUGHT[b["program"][0]].replace("{x}", c["entity"]) in c["text"]  # the noun phrase is untouched
    train_two = {(tuple(p), s) for p, s in study["train_instances"] if len(p) == 2}
    assert all((tuple(x["program"]), x["entity"]) in train_two and len(x["outputs"]) == 3 for x in adds["D"])
    assert DV.additions(world, study, "C") == adds["C"]  # deterministic


def test_the_addition_audit_catches_leaks(small):
    world, study, aug = small
    adds = {a: DV.additions(world, study, a) for a in DV.ARMS}
    held_instance = next(x for x in study["items"] if x["category"] == "S2")
    leaks = {
        "from_training_instances": dict(adds["D"][0], program=held_instance["program"], entity=held_instance["entity"]),
        "targets_at_most_two_relations": dict(adds["D"][0], outputs=["country_of", "capital_of", "airport_of", J.END]),
        "no_held_or_novel_words": dict(adds["C"][0], text="metropolis " + adds["C"][0]["text"]),
        "no_added_question_is_an_evaluation_question": dict(adds["D"][0], text=held_instance["text"]),
    }
    for check, bad in leaks.items():
        broken = {**adds, "D" if bad.get("category") == "add_D" else "C": [bad] + adds[
            "D" if bad.get("category") == "add_D" else "C"][1:]}
        assert not DV.audit_additions(world, study, aug, broken)[check], check
    assert not DV.audit_additions(world, study, aug, {**adds, "B": adds["B"][1:]})["matched_counts"]


# --- teacher-forced failure kinds ------------------------------------------------------------------------------
def test_failure_kinds_follow_the_first_departure_and_the_teacher_forced_errors():
    E, U = J.END, J.UNSUPPORTED
    ref = ["a", "b", E]
    assert DA.failure_kind(ref, ref, ref) is None
    assert DA.failure_kind(ref, ["a", E], ["a", E, E]) == "premature_stop"
    assert DA.failure_kind(ref, ["c", "b", E], ["c", "b", E]) == "relation_confusion"
    assert DA.failure_kind(ref, ["b", E], ["b", "b", E]) == "error_propagation"  # would have continued correctly
    assert DA.failure_kind(ref, ["a", "b", "c", E], ["a", "b", "c"]) == "overrun"
    assert DA.failure_kind(ref, [U], [U, "b", E]) == "refusal_token"
    assert DA.failure_kind(ref, ["a", E], ["a", E, "c"]) == "broader"
    with pytest.raises(ValueError):  # greedy and teacher-forced decoding share the prefix up to the departure
        DA.failure_kind(ref, ["a", E], ["a", "b", E])


def test_the_decision_rule_needs_c_above_b_in_every_world_and_novelty_in_every_unit():
    unit = lambda s2, joint=True: {"S2_exact": s2, "outcome_joint_novelty": joint}
    units = {f"{a}/{w}/{i}": unit(v) for w in (100, 101) for i in (1, 2, 3)
             for a, v in (("A", 0.8), ("B", 0.8), ("C", 0.9), ("D", 0.99))}
    assert DA.decision_rule(units, (100, 101), (1, 2, 3))["proceed_to_confirmation"]
    tie = {**units, **{f"C/101/{i}": unit(0.8) for i in (1, 2, 3)}}
    assert not DA.decision_rule(tie, (100, 101), (1, 2, 3))["proceed_to_confirmation"]  # equal is not higher
    lost = {**units, "C/100/2": unit(0.9, joint=False)}
    assert not DA.decision_rule(lost, (100, 101), (1, 2, 3))["proceed_to_confirmation"]
    only_d = {**units, **{f"C/{w}/{i}": unit(0.8) for w in (100, 101) for i in (1, 2, 3)}}
    assert not DA.decision_rule(only_d, (100, 101), (1, 2, 3))["proceed_to_confirmation"]  # D cannot trigger it


def test_the_confirmation_needs_s2_and_novelty_in_the_same_unit():
    C = DA.CONFIRMATION
    run = [{"run": "r", "config": {"countries": 96, "updates": 1500},
            "provenance": {"commit": "abc", "research_tree_dirty": False}}]
    unit = lambda s2, joint, exact=0.9: {"outcome_s2_transfer": s2, "outcome_joint_novelty": joint, "S2_exact": exact}
    units = {f"{a}/{w}/{i}": unit(a == "C", True) for a in C["arms"] for w in C["worlds"] for i in C["inits"]}
    v = DA.confirmation_verdict(units, run, C["worlds"], C["inits"])
    assert v["complete"] and v["prediction_C_holds"] and v["holds"] == {"A": False, "B": False, "C": True}
    assert not v["secondary_C_above_B_every_world"]  # equal S2 exact means: not above
    better = {**units, **{f"C/{w}/{i}": unit(True, True, 0.95) for w in C["worlds"] for i in C["inits"]}}
    assert DA.confirmation_verdict(better, run, C["worlds"], C["inits"])["secondary_C_above_B_every_world"]
    # S2 in one unit and novelty in another do not add up to a passing unit
    split = {**units, **{f"C/710/{i}": unit(i % 2 == 0, i % 2 == 1) for i in C["inits"]}}
    v = DA.confirmation_verdict(split, run, C["worlds"], C["inits"])
    assert v["successes_by_world"]["C"]["710"] == 0 and not v["prediction_C_holds"]
    four = {**units, "C/711/203": unit(True, False)}  # 4 of 5 is enough in a world
    assert DA.confirmation_verdict(four, run, C["worlds"], C["inits"])["prediction_C_holds"]
    three = {**four, "C/711/204": unit(False, True)}
    assert not DA.confirmation_verdict(three, run, C["worlds"], C["inits"])["prediction_C_holds"]
    missing = {k: v for k, v in units.items() if k != "B/712/200"}
    assert not DA.confirmation_verdict(missing, run, C["worlds"], C["inits"])["complete"]
    dirty = [{**run[0], "provenance": {"commit": "abc", "research_tree_dirty": True}}]
    assert not DA.confirmation_verdict(units, dirty, C["worlds"], C["inits"])["complete"]
    wrong = [{**run[0], "config": {"countries": 48, "updates": 1500}}]
    assert not DA.confirmation_verdict(units, wrong, C["worlds"], C["inits"])["complete"]


# --- end to end on a tiny run ------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_development(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(S, "SIZES", {**S.SIZES, **SMALL})
    out = tmp_path_factory.mktemp("s2dev")
    args = DV.parse_args(["--arms", "A", "B", "C", "D", "--worlds", "7", "--inits", "0", "--countries", "16",
                          "--updates", "20", "--out", str(out)])
    result = DV.run(args, out)
    (out / "s2_development.json").write_text(json.dumps(result))
    yield out, result
    mp.undo()


def test_greedy_traces_reproduce_the_decisions_and_the_analysis_rebuilds_every_fit(tiny_development):
    out, result = tiny_development
    assert all(not u["greedy_trace_mismatches"] for u in result["units"])
    a = DA.analyse([out])
    assert sorted(a["units"]) == [f"{arm}/7/0" for arm in "ABCD"]
    assert "decision_rule" not in a and "confirmation" not in a  # world 7 is neither a development nor a confirmation world
    assert set(DA.decision_rule(a["units"], [7], [0])) >= {"proceed_to_confirmation", "C_above_B_by_world"}
    for arm in "ABCD":  # the record classes and the trace kinds describe the same greedy outputs
        for label, kinds in a["failure_kind_by_class"][arm].items():
            assert (label == "exact") == (set(kinds) == {"right"}) or label == "entity_only"
    tampered = json.loads((out / "s2_development.json").read_text())
    tampered["units"][1]["policies"]["none"]["categories"]["S2"]["exact_parse"] += 0.01
    bad = out.parent / "tampered"
    bad.mkdir()
    for sub in ("records", "traces"):
        (bad / sub).symlink_to(out / sub)
    (bad / "s2_development.json").write_text(json.dumps(tampered))
    with pytest.raises(AssertionError):
        DA.analyse([bad])


# --- the bundled historical analysis ---------------------------------------------------------------------------
def test_the_bundled_counts_rebuild_from_one_confirmation_unit():
    """Quick check of the committed historical analysis: unit 700/0 rebuilt from its bundled records."""
    summary = json.loads((BUNDLE / "s2_diagnosis/historical/s2_diagnosis.json").read_text())
    run = BUNDLE / "interpreter_robustness/20261018T232057"
    result = json.loads((run / "interpreter_robustness.json").read_text())
    world = W.generate_world(700, n_countries=96)
    held = [x for x in S.make_study(world, 700, result["sizes"])["items"] if x["split"] != "train"]
    with gzip.open(run / "records/w700_i0.json.gz", "rt") as f:
        rec = json.load(f)
    unit = next(u for u in result["units"] if (u["world"], u["init"]) == (700, 0))
    for arm in S.ARMS:
        threshold = unit["arms"][arm]["policies"]["certified"]["certificate"]["threshold"]
        rows = D2.question_rows(held, rec["arms"][arm], rec["ids"], threshold, 700, 0, arm)
        assert D2.unit_metrics(rows) == summary["per_unit"][arm]["700/0"], arm
