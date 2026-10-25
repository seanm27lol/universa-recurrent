"""The factored interpreter's engineering pilot: its policies, its cost accounting and its independent verification.

- **A+K's decoder.** With no constraint it reproduces A's own greedy decoding exactly (`interpreters.interpret`):
  the same outputs, probabilities and entity. Its recurrence is A's.
- **F+B, the review's fixture.** One segment over an Airport entity: `capital_of` 0.31, `<unsupported>` 0.30,
  `operator_of` 0.10. F+B must abstain. Without the refusal check (revision 1's rule) it would answer `operator_of`.
- **A's realized training draws.** Replaying the generator gives exactly the batches `interpreters.train` draws.
- **A small pilot run** (world 100 at 24 countries, 100 updates, one init, both regimes) re-derives in full. Each
  corruption is rejected:
  - a renormalized A+K score or distribution;
  - an altered A+K allowed set;
  - a missing F+B alternative;
  - an altered F+B refusal decision;
  - changed certificate correctness labels;
  - an altered historical certificate;
  - altered training presentations;
  - an incomplete inventory;
  - a configuration mismatch.
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "ti_tnn"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
torch = pytest.importorskip("torch")
import factored_interpreter as FI  # noqa: E402
import factored_pilot as P  # noqa: E402
import factored_pilot_verify as PV  # noqa: E402
from integrated import constrained_decoding as CD  # noqa: E402
from integrated import interpreters as I  # noqa: E402
from integrated import registry as G  # noqa: E402
from integrated import relations as W  # noqa: E402
from integrated import routing as J  # noqa: E402

SMALL = {"countries": 24, "updates": 100}


@pytest.fixture(scope="module")
def small_world():
    world = W.generate_world(100, n_countries=SMALL["countries"])
    study = FI.study_for(world, 100)
    train, aug, _ = FI.regime_training(study, "full", world, 100)
    return world, study, train, aug


def test_the_unconstrained_decoder_reproduces_a_greedy_decoding(small_world):
    world, study, train, aug = small_world
    torch.set_num_threads(4)
    examples = [J.Example(x["text"], tuple(x["outputs"]), x["entity_position"]) for x in train + aug]
    net, vocab, _ = I.train("word", examples, [x["text"] for x in train], tuple(world.relations), 100, 60)
    texts = [x["text"] for x in study["items"] if x["split"] != "train"][:600]
    parses = I.interpret(net, vocab, texts)
    free = CD.decode(net, vocab, texts, [J.MAX_STEPS] * len(texts), world, constrained=False)
    for p, d in zip(parses, free):
        chosen = [net.outputs[s["chosen"]] for s in d["steps"]]
        probs = [s["distribution"][s["chosen"]] for s in d["steps"]]
        assert d["entity"] == p.entity and d["entity_probability"] == p.entity_probability
        assert probs == list(p.step_probabilities)
        assert [o for o in chosen if o in world.relations] == list(p.steps)
        assert d["outcome"] == p.outcome


def test_constrained_selection_uses_allowed_logits_when_probabilities_underflow():
    """Two allowed outputs whose float32 probabilities both underflow to exactly 0, with different logits. Selecting
    by probability ties and takes the earlier index (the rule of the pilot recorded at 0e1a3c9); selecting by logit
    takes the higher logit. The score keeps the original probability (0.0), never a renormalized one."""
    logits = [60.0, -70.0, -60.0, 10.0, 0.0, -5.0, -5.0, -5.0]
    allowed = [1, 2]
    probs = torch.softmax(torch.tensor(logits), -1).tolist()
    assert probs[1] == probs[2] == 0.0
    by_probability = max(allowed, key=lambda k: (probs[k], -k))
    assert by_probability == 1 and CD.select(logits, allowed) == 2
    assert probs[CD.select(logits, allowed)] == 0.0  # the score's factor stays the unmasked probability


def test_f_plus_b_abstains_on_the_reviews_fixture(small_world):
    world = small_world[0]
    labels = list(J_labels := tuple(sorted(world.relations)) + (J.UNSUPPORTED, J.AMBIGUOUS))
    probs = {"capital_of": 0.31, J.UNSUPPORTED: 0.30, "operator_of": 0.10, "largest_city_of": 0.09, "country_of": 0.08,
             "airport_of": 0.07, J.AMBIGUOUS: 0.05}
    dist = [probs[lab] for lab in J_labels]
    airport = world.types["Airport"].values[0]
    reg = G.table_registry(world)
    out = P.fb_policy([dist], labels, airport, world, reg)
    assert out["decision"] == "abstain" and out["reason"] == "a refusal label outweighs the selected relation"
    assert out["candidates"] == [["capital_of", "operator_of"]]
    assert [(p["program"], p["pruned"]) for p in out["programs"]] == [(["capital_of"], True), (["operator_of"], False)]
    sel = out["programs"][out["selected"]]
    assert sel["program"] == ["operator_of"]  # revision 1's rule, which stopped before the refusal check, answers this
    assert PV.fb_row([dist], labels, airport, world)["reason"] == out["reason"]  # the verifier's own implementation


def test_the_realized_draw_replay_matches_what_training_draws(small_world, monkeypatch):
    world, _, train, aug = small_world
    examples = [J.Example(x["text"], tuple(x["outputs"]), x["entity_position"]) for x in train + aug]
    seen = Counter()
    real = I.make_batch

    def recording(mode, token_lists, vocab, drops=None):
        if drops is not None:  # training batches carry the dropout positions; evaluation batches do not
            seen.update(" ".join(t) for t in token_lists)
        return real(mode, token_lists, vocab, drops)
    monkeypatch.setattr(I, "make_batch", recording)
    I.train("word", examples, [x["text"] for x in train], tuple(world.relations), 7, 40)
    draws = P.a_realized_draws(len(examples), 7, 40, 64)
    replay = Counter()
    for e, n in zip(examples, draws):
        replay[" ".join(I.normalize(e.text))] += n
    assert +replay == +seen and sum(draws) == 40 * 64


# --- a small run and its verification ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_run(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(P, "UPDATES", SMALL["updates"])
    try:
        return P.run([100], [100], tmp_path_factory.mktemp("pilot"), countries=SMALL["countries"])
    finally:
        mp.undo()


DESIGN = {**PV.DESIGN_V2, **SMALL}


def test_the_small_pilot_run_re_derives(small_run):
    out = PV.verify(small_run, DESIGN)
    assert out["verified"] and len(out["units"]) == 2


def _edit_records(run, name, edit, unit="w100_i100_full"):
    path = run / "units" / unit / f"records_{name}.json.gz"
    recs = json.loads(gzip.open(path, "rt").read())
    edit(recs)
    with gzip.open(path, "wt") as f:
        json.dump(recs, f)


def _edit_unit(run, edit, unit="w100_i100_full"):
    path = run / "units" / unit / "unit.json"
    u = json.loads(path.read_text())
    edit(u)
    path.write_text(json.dumps(u))


def _renormalize(recs):
    r = next(r for r in recs if r.get("steps") and r["outcome"] == "program")
    st = r["steps"][-1]  # the forced <end> step: renormalize it over its allowed set
    total = sum(st["distribution"][i] for i in st["allowed"])
    st["distribution"] = [p / total if i in st["allowed"] else 0.0 for i, p in enumerate(st["distribution"])]
    score = 1.0
    for s in r["steps"]:
        score *= s["distribution"][s["chosen"]]
    r["score"] = score * r["entity_probability"]


def _renormalized_score_only(recs):
    r = next(r for r in recs if r.get("steps") and r["outcome"] == "program")
    st = r["steps"][-1]
    r["score"] = r["score"] / st["distribution"][st["chosen"]] * (
        st["distribution"][st["chosen"]] / sum(st["distribution"][i] for i in st["allowed"]))


def _drop_alternative(recs):
    r = next(r for r in recs if r.get("programs"))  # a confident F often leaves one candidate: drop a program anyway
    r["programs"].pop()


def _flip_refusal_check(recs):
    r = next(r for r in recs if r["decision"] == "answer")
    r.update(decision="abstain", reason="a refusal label outweighs the selected relation", answer=None, program=None,
             score=None)


def _widen_allowed(recs):
    r = next(r for r in recs if r.get("steps") and len(r["steps"]) >= 2)
    r["steps"][0]["allowed"] = list(range(len(r["steps"][0]["distribution"])))


CORRUPTIONS = {
    "A+K distribution renormalized over its allowed set": ("records", "AK", _renormalize),
    "A+K score renormalized": ("records", "AK", _renormalized_score_only),
    "A+K allowed set widened": ("records", "AK", _widen_allowed),
    "F+B alternative missing": ("records", "FB", _drop_alternative),
    "F+B refusal decision altered": ("records", "FB", _flip_refusal_check),
    "certificate correctness label changed": ("unit", None, lambda u: u["systems"]["F"]["certificate"]["records"][0]
                                              .__setitem__(2, not u["systems"]["F"]["certificate"]["records"][0][2])),
    "historical certificate altered": ("unit", None, lambda u: u["historical_A"]["certificate"].__setitem__(
        "threshold", 0.42)),
    "F presentations altered": ("unit", None, lambda u: (lambda d: d.__setitem__(next(iter(d)), d[next(iter(d))] + 1))(
        u["training"]["F"]["draws_by_tuple"])),
    "A realized draws altered": ("unit", None, lambda u: u["training"]["A"].__setitem__("tokens", 1)),
    "P5 altered": ("unit", None, lambda u: u["p5_s2_exact_minus_G"].__setitem__("F", 0.5)),
    "evaluation cost altered": ("unit", None, lambda u: u["evaluation_costs"]["FB"].__setitem__("candidates_pruned", 0)),
    "certificate alpha altered": ("unit", None, lambda u: u["systems"]["F"]["certificate"].__setitem__("alpha", 0.1)),
    "historical certificate delta altered": ("unit", None, lambda u: u["historical_A"]["certificate"].__setitem__(
        "delta", 0.1)),
    "historical eligible record altered": ("unit", None, lambda u: u["historical_A"]["certificate"]["records"][0].__setitem__(
        "correct", not u["historical_A"]["certificate"]["records"][0]["correct"])),
    "historical certified table altered": ("unit", None, lambda u: u["historical_A"]["certified"]["S2"].__setitem__(
        "errors", u["historical_A"]["certified"]["S2"]["errors"] + 1)),
    "historical none table altered": ("unit", None, lambda u: u["historical_A"]["none"]["S1"].__setitem__(
        "wrong_program", u["historical_A"]["none"]["S1"]["wrong_program"] + 1)),
    "A+K logit altered": ("records", "AK", lambda recs: (lambda st: st["logits"].__setitem__(
        st["chosen"], st["logits"][st["chosen"]] - 1e6))(next(r for r in recs if r.get("steps"))["steps"][0])),
    "A padded MACs altered": ("unit", None, lambda u: u["training"]["A"].__setitem__("forward_macs_padded", 1)),
    "A+K evaluated row-steps altered": ("unit", None, lambda u: u["evaluation_costs"]["AK"].__setitem__(
        "evaluated_row_steps", u["evaluation_costs"]["AK"]["emitted_decisions"])),
}


@pytest.mark.parametrize("name", sorted(CORRUPTIONS))
def test_every_pilot_corruption_is_rejected(small_run, tmp_path, name):
    bad = tmp_path / "run"
    shutil.copytree(small_run, bad)
    kind, system, edit = CORRUPTIONS[name]
    if kind == "records":
        _edit_records(bad, system, edit)
    else:
        _edit_unit(bad, edit)
    with pytest.raises(PV.VerificationError):
        PV.verify(bad, DESIGN)


def test_a_run_claiming_the_bound_commit_needs_that_commits_sources(small_run, tmp_path):
    """The recorded pilot is bound to 0e1a3c9: a run naming that commit is checked against its git hashes, so current
    sources cannot pass as the recorded ones."""
    bad = tmp_path / "run"
    shutil.copytree(small_run, bad)
    meta = json.loads((bad / "run.json").read_text())
    meta["provenance"]["commit"] = next(iter(PV.BOUND))
    (bad / "run.json").write_text(json.dumps(meta))
    with pytest.raises(PV.VerificationError, match="bound commit"):
        PV.verify(bad, {**PV.DESIGN, **SMALL})


@pytest.mark.parametrize("edit, match", [
    (lambda m: m.__setitem__("units", m["units"][:1]), "inventory"),
    (lambda m: m["config"]["fb"].__setitem__("ratio", 0.4), "configuration"),
    (lambda m: m["provenance"]["sha256"].__setitem__("factored_pilot.py", "0" * 64), "sources"),
])
def test_run_level_corruptions_are_rejected(small_run, tmp_path, edit, match):
    bad = tmp_path / "run"
    shutil.copytree(small_run, bad)
    meta = json.loads((bad / "run.json").read_text())
    edit(meta)
    (bad / "run.json").write_text(json.dumps(meta))
    with pytest.raises(PV.VerificationError, match=match):
        PV.verify(bad, DESIGN)


# --- the A+K diagnosis (saved records only) -------------------------------------------------------------------------
def test_the_diagnosis_classifies_every_outcome():
    import factored_pilot_diagnose as FD
    outputs = ["airport_of", "capital_of", "country_of", "largest_city_of", "operator_of", "<end>", "<unsupported>",
               "<ambiguous>"]
    item = {"entity": "country_1_city_0", "program": ["airport_of", "operator_of"]}
    dist = [0.1] * 8
    step = {"distribution": dist, "allowed": [0, 2, 6, 7], "chosen": 2}
    base = {"grammar": "ok", "entity": "country_1_city_0", "outcome": "program", "steps": [step, step, step]}
    assert FD.classify(item, {"grammar": "unknown entity"}, 0.2, outputs)[0] == "grammar refusal"
    assert FD.classify(item, {**base, "entity": "country_2"}, 0.2, outputs)[0] == "entity failure"
    assert FD.classify(item, {**base, "outcome": "unsupported"}, 0.2, outputs)[0] == "decoder refusal"
    right = {**base, "program": ["airport_of", "operator_of"]}
    assert FD.classify(item, {**right, "score": 0.5}, 0.2, outputs)[0] == "correct program, released"
    assert FD.classify(item, {**right, "score": 0.1}, 0.2, outputs)[0] == "correct program, filtered"
    cls, detail = FD.classify(item, {**base, "program": ["country_of", "operator_of"], "score": 0.5}, 0.2, outputs)
    assert cls == "wrong program, released" and detail["position"] == 0
    assert (detail["reference"], detail["selected"]) == ("airport_of", "country_of")
    # what A itself preferred at a step: <end> where the grammar still required a relation, or a forbidden relation
    stops = {"distribution": [0.01, 0.0, 0.02, 0.0, 0.0, 0.97, 0.0, 0.0], "allowed": [0, 2, 6, 7], "chosen": 2}
    assert FD.preference(stops, outputs) == ("<end>", 0.97, 0.0)
    ill_typed = {"distribution": [0.01, 0.9, 0.02, 0.0, 0.07, 0.0, 0.0, 0.0], "allowed": [0, 2, 6, 7], "chosen": 2}
    kind, p_end, p_forbidden = FD.preference(ill_typed, outputs)
    assert kind == "a relation the constraint forbids" and p_end == 0.0 and abs(p_forbidden - 0.97) < 1e-12
    cls, detail = FD.classify(item, {**base, "steps": [stops, step, step], "program": ["country_of", "operator_of"],
                                     "score": 0.01}, 0.2, outputs)
    assert cls == "wrong program, filtered" and (detail["a_preferred"], detail["p_end"]) == ("<end>", 0.97)
    # A's own first divergence: a program that starts with a wrong-type relation and then ends early is a wrong-type
    # choice, not a premature stop
    from types import SimpleNamespace as NS
    world = NS(relations={"airport_of": NS(source="City", target="Airport"), "operator_of": NS(source="Airport",
                          target="Operator"), "country_of": NS(source="City", target="Country"),
                          "capital_of": NS(source="Country", target="City")})
    type_of, a = {"country_1_city_0": "City"}, {"outcome": "program", "entity": "country_1_city_0"}
    first = lambda steps, **kw: FD.first_divergence({**a, "steps": steps, **kw}, item, world, type_of)  # noqa: E731
    assert first(["airport_of"]) == "A stopped early"
    assert first(["airport_of", "operator_of", "capital_of"]) == "A ran long"
    assert first(["capital_of"]) == "A first chose a relation of the wrong source type, at position 0"
    assert first(["airport_of", "capital_of"]) == "A first chose a relation of the wrong source type, at position 1"
    assert first(["country_of", "capital_of"]) == "A first chose a relation of the right source type, at position 0"
    assert first([], outcome="malformed") == "A malformed"


def test_the_diagnosis_partitions_every_question_of_a_run(small_run):
    import factored_pilot_diagnose as FD
    run = json.loads((small_run / "run.json").read_text())
    cfg = run["config"]
    for info in run["units"]:
        d = FD.diagnose_unit(small_run, info, cfg["countries"], cfg["updates"], cfg["batch"])
        for cat in FD.CATEGORIES:
            classes, preferred = d[cat]["classes"], d[cat]["what_a_preferred"]
            assert sum(classes.values()) == d[cat]["n"] > 0
            for kind, key in (("correct", "correct programs, at the weakest step"),
                              ("wrong", "wrong programs, at the first difference")):
                assert sum(preferred.get(key, {}).values()) == (classes[f"{kind} program, released"]
                                                                + classes[f"{kind} program, filtered"])
        assert d["selection_ties"]["tied maximum among allowed"] == 0 or d["selection_ties"]["steps"] > 0
        assert d["accounting"]["AK"]["evaluated_row_steps"] >= d["accounting"]["AK"]["emitted_decisions_recorded"]
