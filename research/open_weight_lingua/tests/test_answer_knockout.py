import json

import pytest
import torch

from open_weight_lingua import answer_knockout as ak

PROGRAM = "x = 9\ny = 8\ny = y + 3"
PROMPT = PROGRAM + "\nWhat is x? Reply with only the integer."


class CharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def test_program_span_covers_exactly_the_program_text():
    text = "<u>" + PROMPT + "<m>\n"
    ids = [ord(c) for c in text]
    first, last = ak.program_span(CharTokenizer(), ids, PROGRAM)
    assert text[first : last + 1] == PROGRAM


def test_layer_grid_matches_the_brief():
    assert ak.layer_grid(48) == [0, 6, 12, 18, 24, 30, 36, 42]
    assert ak.layer_grid(28) == [0, 3, 7, 10, 14, 17, 21, 24]


def _eager(model):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    else:
        model.config._attn_implementation = "eager"
    return model.eval()


def _models(model_factory):
    from test_gemma3_arch import tiny_conditional_generation

    torch.manual_seed(0)
    return {"qwen2": _eager(model_factory(layers=4)), "gemma3": _eager(tiny_conditional_generation())}


def _hidden(model, ids):
    tensor = torch.tensor([ids])
    with torch.no_grad():
        return model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False, output_hidden_states=True)


@pytest.mark.parametrize("name", ("qwen2", "gemma3"))
def test_empty_knockout_changes_nothing(name, model_factory):
    model = _models(model_factory)[name]
    ids = [5, 6, 7, 8, 9, 10, 11]
    baseline = _hidden(model, ids).logits
    with ak.knockout(model, 0, [4, 5, 6], []):
        hooked = _hidden(model, ids).logits
    assert torch.equal(baseline, hooked)


@pytest.mark.parametrize("name", ("qwen2", "gemma3"))
def test_full_knockout_makes_later_positions_blind_to_the_program(name, model_factory):
    model = _models(model_factory)[name]
    span, after = [1, 2, 3], [4, 5, 6]
    original, edited = [5, 6, 7, 8, 9, 10, 11], [5, 12, 13, 14, 9, 10, 11]
    with ak.knockout(model, 0, after, span):
        a = _hidden(model, original).logits[0, after]
        b = _hidden(model, edited).logits[0, after]
    assert torch.equal(a, b)
    assert not torch.equal(_hidden(model, original).logits[0, after], _hidden(model, edited).logits[0, after])


@pytest.mark.parametrize("name", ("qwen2", "gemma3"))
def test_knockout_from_layer_k_leaves_earlier_layers_untouched(name, model_factory):
    model = _models(model_factory)[name]
    ids = [5, 6, 7, 8, 9, 10, 11]
    baseline = _hidden(model, ids).hidden_states
    with ak.knockout(model, 2, [4, 5, 6], [1, 2, 3]):
        hooked = _hidden(model, ids).hidden_states
    assert all(torch.equal(baseline[i], hooked[i]) for i in range(3))  # inputs to layers 0..2
    assert not torch.equal(baseline[3][0, 4:], hooked[3][0, 4:])  # output of layer 2 at blocked queries


def test_knockout_refuses_a_non_additive_mask(model_factory):
    model = model_factory(layers=2)
    model.config._attn_implementation = "sdpa"
    with ak.knockout(model, 0, [2], [0]), pytest.raises(RuntimeError, match="eager"):
        _hidden(model, [5, 6, 7])


def _rows_and_results(accuracies, baseline, layers=48):
    grid = ak.layer_grid(layers)
    rows, results = [], []
    for i in range(100):
        rows.append({"split": "calibration" if i < 50 else "pilot", "asked_last_update_kind": "literal", "last_line_assigns_asked": True})
        result = {"baseline": {"correct": i % 50 < baseline * 50, "logprob": -0.1}}
        for variant in ak.VARIANTS:
            for k, accuracy in zip(grid, accuracies):
                result[f"{variant}@{k}"] = {"correct": i % 50 < accuracy * 50, "logprob": -1.0}
        results.append(result)
    return rows, results


def test_release_layer_is_the_first_grid_layer_after_which_accuracy_stays_within_tolerance():
    rows, results = _rows_and_results([0.1, 0.2, 0.5, 0.9, 0.96, 0.94, 0.96, 0.98], baseline=0.98)
    summary = ak.summarize(rows, results, 48)
    assert summary["release_layer"]["answer"] == {"calibration": 24, "pilot": 24}
    rows, results = _rows_and_results([0.1, 0.2, 0.5, 0.96, 0.9, 0.96, 0.96, 0.98], baseline=0.98)
    assert ak.summarize(rows, results, 48)["release_layer"]["answer"]["calibration"] == 30  # the dip at 24 counts
    rows, results = _rows_and_results([0.1] * 8, baseline=0.98)
    assert ak.summarize(rows, results, 48)["release_layer"]["after_program"]["calibration"] is None


def test_end_to_end_on_fixture_models(tmp_path, monkeypatch, model_factory, tokenizer):
    """Tiny random model and fixture tokens; plumbing only, never a real result."""
    from test_pilot_stages import TINY_COUNTS, _inputs
    import transformers

    from open_weight_lingua.artifacts import RunDirectory
    from open_weight_lingua.splits import build_plan

    plan = build_plan(TINY_COUNTS)
    lock = tmp_path / "model-lock-gemma3-12b.json"
    lock.write_text(json.dumps({"models": {"target": {"revision": "r", "files": {}}}}))
    for split, name in (("calibration", "cal"), ("pilot", "pil")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text(
            json.dumps({"inputs": _inputs(plan.groups[split]), "model_lock_sha256": ak.sha256_file(lock)}))
    monkeypatch.setattr(ak, "load_model", lambda path, role, device: _eager(model_factory(layers=8)))
    monkeypatch.setattr(ak, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(ak, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(ak, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(ak, "compatibility", lambda device: {})
    monkeypatch.setattr(ak, "program_span", lambda tok, ids, body: (0, 0))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: tokenizer)
    args = ak.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                          "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"])
    run = RunDirectory(tmp_path / "knockout")
    summary = ak.run(args, run, {})
    assert summary["grid"] == ak.layer_grid(8) and set(summary["release_layer"]) == set(ak.VARIANTS)
    lines = (run.path / "results.jsonl").read_text().splitlines()
    assert len(lines) == len(json.loads((run.path / "manifest.json").read_text())["rows"])
    first = json.loads(lines[0])
    assert {"baseline", "answer@0", "after_program@7"} <= set(first)
