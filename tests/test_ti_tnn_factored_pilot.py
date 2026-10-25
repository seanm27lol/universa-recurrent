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


DESIGN = {**PV.DESIGN, **SMALL}


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
