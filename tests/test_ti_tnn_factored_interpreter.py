"""The factored interpreter's supplied grammar and no-fit controls (protocols/factored_interpreter_v1.md, revision 2).

- **The grammar** (`integrated/segment_grammar.py`): segment boundaries and execution order on worked examples.
  It agrees with the verifier's own implementation on every generated question of development world 100.
- **Training-derived labels:** ill-typed refusals keep their relation labels. The declared exclusions are applied
  and counted. The reduced-exposure regime keeps 3 questions per taught phrase and the same distinct segments.
- **L and G on development world 100:** zero wrong decisions and zero errors over all 17,900 evaluated items, in
  both regimes and both representations. This independently reproduces the review's no-fit check.
- **The independent verifier** re-derives that run, and rejects each corruption: reordered segments, a wrong
  entity binding, a wrong label, an altered refusal decision, an incomplete inventory and changed correctness
  counts.
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
pytest.importorskip("torch")  # the robustness generator imports torch
import factored_interpreter as FI  # noqa: E402
import factored_interpreter_verify as FV  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import segment_grammar as SG  # noqa: E402

ENTITIES = {"country_29_city_0", "country_5", "country_5_city_1_airport"}


# --- the grammar ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, style, segments", [
    ("name the company operating the airfield of country_29_city_0 .", "prefix",
     (("the", "airfield", "of"), ("the", "company", "operating"))),  # innermost first
    ("could you tell me the capital of country_5 ?", "prefix", (("the", "capital", "of"),)),  # novel frame discarded
    ("name exactly the country of country_29_city_0 .", "prefix", (("the", "country", "of"),)),  # filler discarded
    ("what is country_5 's capital 's airport ?", "possessive", (("'s", "capital"), ("'s", "airport"))),  # left to right
    ("what is country_29_city_0 's largest city ?", "possessive", (("'s", "largest", "city"),)),
])
def test_the_grammar_segments_and_orders_without_a_lexicon(text, style, segments):
    seg = SG.segment(text, ENTITIES)
    assert seg.ok and seg.style == style and seg.segments == segments
    assert FV.read(text, ENTITIES)["segments"] == [list(s) for s in segments]


@pytest.mark.parametrize("text, status", [
    ("what is the capital of country_999 ?", "unknown entity"),
    ("the capital of country_5 's airport .", "malformed: mixed styles"),
    ("what is country_5 ?", "malformed: no segment"),
    ("the a the b the c the d of country_5 .", "malformed: more than 3 segments"),
    ("country_5 the capital of country_29_city_0 .", "malformed: several entities"),
])
def test_the_grammar_refuses_what_it_cannot_read(text, status):
    assert SG.segment(text, ENTITIES).status == status == FV.read(text, ENTITIES)["status"]


@pytest.fixture(scope="module")
def world100():
    world = W.generate_world(100, n_countries=FI.COUNTRIES)
    return world, FI.study_for(world, 100)


def test_the_producer_and_verifier_grammars_agree_on_every_generated_question(world100):
    world, study = world100
    entities = FI.entity_names(world)
    for x in study["items"] + FI.IR.augment(study, 100):
        seg, mine = SG.segment(x["text"], entities), FV.read(x["text"], entities)
        assert (seg.status, seg.style, [list(s) for s in seg.segments]) == (mine["status"], mine["style"], mine["segments"])


# --- training-derived labels and the regimes -----------------------------------------------------------------
def test_ill_typed_refusals_keep_their_relation_labels_and_exclusions_are_declared(world100):
    world, study = world100
    train, aug, _ = FI.regime_training(study, "full", world, 100)
    lab = FI.extract_labels(train + aug, world)
    ill = [c for c in lab["contributions"] if c["source"] == "train_refusal:ill_typed"]
    assert ill and all(c["label"] in world.relations for c in ill)  # meaningful phrases, refused only as a whole
    reasons = {e["reason"] for e in lab["excluded"]}
    assert reasons <= {"grammar: unknown entity", "refusal target over several segments: which one carries it is not given"}
    assert all(e["source"] == "train_refusal:unknown_entity" for e in lab["excluded"] if e["reason"].startswith("grammar"))


def test_the_reduced_regime_keeps_three_per_taught_phrase_and_the_same_distinct_segments(world100):
    world, study = world100
    full, aug, _ = FI.regime_training(study, "full", world, 100)
    reduced, _, dropped = FI.regime_training(study, "reduced", world, 100)
    taught = lambda items: [x for x in items if x["category"] == "train_taught"]  # noqa: E731
    per_relation = {r: sum(x["program"][0] == r for x in taught(reduced)) for r in world.relations}
    assert set(per_relation.values()) == {FI.REDUCED_PER_PHRASE} and len(dropped) == len(taught(full)) - 15
    distinct = lambda items: {(c["style"], tuple(c["tokens"]), c["label"])  # noqa: E731
                              for c in FI.extract_labels(items, world)["contributions"]}
    assert distinct(reduced + aug) == distinct(full + aug)  # F's inputs discard entity and frame: same support


def test_l_refuses_unseen_conflicting_and_refusal_labelled_segments_and_runtime_checks_types(world100):
    world, _ = world100
    from integrated import registry as G
    reg, ents = G.table_registry(world), FI.entity_names(world)
    country = world.types["Country"].values[0]
    q = f"what is the capital of {country} ?"
    key = ("prefix", ("the", "capital", "of"))
    assert FI.decide(q, world, reg, {}, entities=ents)["reason"] == "unseen segment"
    assert FI.decide(q, world, reg, {key: FI.CONFLICT}, entities=ents)["reason"] == "conflicting labels"
    assert FI.decide(q, world, reg, {key: "<unsupported>"}, entities=ents)["reason"] == "<unsupported>"
    assert FI.decide(q, world, reg, {key: "operator_of"}, entities=ents)["reason"] == "ill-typed program"
    ok = FI.decide(q, world, reg, {key: "capital_of"}, entities=ents)
    assert ok["decision"] == "answer" and ok["answer"] == world.facts["capital_of"][country]


# --- the development result and its verification -------------------------------------------------------------
@pytest.fixture(scope="module")
def dev_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("factored")
    return FI.run([100, 101], out)


@pytest.mark.parametrize("world", [100, 101])
def test_l_and_g_make_no_wrong_decision_on_the_development_worlds(dev_run, world):
    for regime in FI.REGIMES:
        unit = json.loads((dev_run / "units" / f"w{world}_{regime}" / "unit.json").read_text())
        assert unit["n_evaluated"] == 17900
        assert unit["totals"] == {s: {"wrong_decisions": 0, "errors": 0, "accidental_agreement": 0}
                                  for s in ("L_raw", "L_vocab", "G")}
        assert unit["alignment_audit"].get("misaligned", 0) == 0 and unit["support"]["conflicts"] == {"raw": [], "vocab": []}
        for cat in ("ID", "T1", "S2", "S1", "S3"):  # every supported evaluation segment was seen in training
            cov = unit["support"]["evaluation_coverage"]["raw"][cat]
            assert cov["in_dictionary"] == cov["segments"]
        for name, risk in unit["calibration_risk"].items():  # one risk test, same items and endpoint as the neural ones
            assert risk["calibration_items"] == 1000 and risk["errors"] == 0 and risk["accepted"] >= 299, name
            assert risk["certified"] and risk["p_value"] <= FI.RISK_DELTA, name


def test_the_independent_verifier_re_derives_the_development_run(dev_run):
    out = FV.verify(dev_run)
    assert out["verified"] and len(out["units"]) == 4


def _gz(path, edit):
    data = json.loads(gzip.open(path, "rt").read())
    edit(data)
    with gzip.open(path, "wt") as f:
        json.dump(data, f)


def _first(rows, pred):
    return next(r for r in rows if pred(r))


CORRUPTIONS = {
    "reordered segments": ("records.json.gz", lambda r: (lambda x: x.__setitem__("program", x["program"][::-1]))(
        _first(r["L_raw"], lambda x: x["decision"] == "answer" and len(x["program"]) == 2
               and x["program"][0] != x["program"][1]))),
    "wrong entity binding": ("records.json.gz", lambda r: _first(r["G"], lambda x: x["decision"] == "answer")
                             .__setitem__("entity", "country_0")),
    "wrong label": ("labels.json.gz", lambda d: _first(d["contributions"], lambda c: c["label"] == "capital_of")
                    .__setitem__("label", "largest_city_of")),
    "altered refusal decision": ("records.json.gz", lambda r: _first(r["L_vocab"], lambda x: x["reason"] == "<unsupported>")
                                 .__setitem__("reason", "unseen segment")),
    "missing record": ("records.json.gz", lambda r: r["L_raw"].pop()),
    "changed correctness counts": ("unit.json", lambda u: u["systems"]["L_raw"]["S2"].__setitem__("errors", 1)),
    "altered phrase exposure": ("unit.json", lambda u: u["support"]["label_multiplicities"]["airport_of"].__setitem__(
        "prefix: the airfield of", 3)),
    "altered dictionary coverage": ("unit.json", lambda u: u["support"]["evaluation_coverage"]["vocab"]["S4a_primary"]
                                    .__setitem__("refusal_labelled", 0)),
    "altered calibration risk": ("unit.json", lambda u: u["calibration_risk"]["G"].__setitem__("errors", 3)),
    "altered study identity": ("unit.json", lambda u: u.__setitem__("study_identity", "0" * 64)),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_every_corruption_of_the_no_fit_export_is_rejected(dev_run, tmp_path, name):
    bad = tmp_path / "run"
    shutil.copytree(dev_run, bad)
    file, edit = CORRUPTIONS[name]
    path = bad / "units" / "w100_full" / file
    if file.endswith(".gz"):
        _gz(path, edit)
    else:
        data = json.loads(path.read_text())
        edit(data)
        path.write_text(json.dumps(data))
    with pytest.raises(FV.VerificationError):
        FV.verify(bad)


RUN_CORRUPTIONS = {
    "one regime missing": (lambda m: m.__setitem__("units", m["units"][1:]), "inventory"),
    "both regimes of a requested world removed": (
        lambda m: m.__setitem__("units", [u for u in m["units"] if u["world"] != 101]), "inventory"),
    "a source mismatch": (lambda m: m["provenance"]["sha256"].__setitem__("factored_interpreter.py", "0" * 64), "sources"),
    "a configuration mismatch": (lambda m: m["config"].__setitem__("reduced_per_phrase", 2), "configuration"),
    "altered run-level totals": (lambda m: m["totals"]["L_raw"].__setitem__("errors", 1), "run-level totals"),
    "altered unit totals in run.json": (lambda m: m["units"][0]["totals"]["G"].__setitem__("wrong_decisions", 1),
                                        "unit totals"),
}


@pytest.mark.parametrize("name", sorted(RUN_CORRUPTIONS))
def test_every_run_level_corruption_is_rejected(dev_run, tmp_path, name):
    bad = tmp_path / "run"
    shutil.copytree(dev_run, bad)
    edit, match = RUN_CORRUPTIONS[name]
    meta = json.loads((bad / "run.json").read_text())
    edit(meta)
    (bad / "run.json").write_text(json.dumps(meta))
    with pytest.raises(FV.VerificationError, match=match):
        FV.verify(bad)


def test_the_harness_refuses_a_confirmation_world(tmp_path):
    with pytest.raises(SystemExit, match="reserved"):
        FI.run([1000], tmp_path)
