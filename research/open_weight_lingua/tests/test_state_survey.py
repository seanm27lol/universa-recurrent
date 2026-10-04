import json

import numpy as np

from open_weight_lingua import state_survey as ss

PROMPT = "x = 9\ny = 8\ny = 3\ny = y + 3\nx = x + 1\nWhat is x? Reply with only the integer."


class CharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def test_positions_follow_the_frozen_definitions_in_order():
    text = "<u>" + PROMPT + "<m>\n"
    row = {"id": "r", "prompt": PROMPT, "variable": "x", "input_ids": [ord(c) for c in text], "position": len(text) - 1}
    found = ss.positions(CharTokenizer(), row)

    def decoded(position):
        return text[: position + 1]

    assert decoded(found["program_end"]).endswith("x = x + 1")
    assert decoded(found["query_variable"]).endswith("What is x")
    assert decoded(found["question_mark"]).endswith("What is x?")
    assert decoded(found["user_end"]).endswith("integer.")
    assert found["answer"] == len(text) - 1


def test_labels_give_final_values_computed_flags_and_the_copy_guess():
    label = ss.labels({"prompt": PROMPT})
    assert label["x"] == {"value": 10, "computed": True, "copy_guess": 9}
    assert label["y"] == {"value": 6, "computed": True, "copy_guess": 3}


def _synthetic(n, d=32, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.integers(0, ss.VALUES, size=n)
    codebook = rng.normal(size=(ss.VALUES, d))
    return codebook[values] + noise * rng.normal(size=(n, d)), values


def test_probe_reads_a_linearly_encoded_value_and_the_permuted_control_does_not():
    x, v = _synthetic(600)
    train, select, test = slice(0, 400), slice(400, 500), slice(500, 600)
    computed = np.zeros(600, dtype=bool)
    computed[::3] = True
    out = ss.probe_accuracies(x[train], x[select], x[test], v[train], v[select], v[test],
                              computed[select], computed[test], permuted=np.random.default_rng(1).permutation(v[train]))
    assert out["onehot"]["test"] > 0.95 and out["onehot"]["test_computed"] > 0.95
    assert out["permuted"]["test"] < 0.25
    assert out["onehot"]["alpha"] in ss.ALPHA_GRID


def _entry(select_x, select_y, computed=0.5, test=0.9):
    def probe(select):
        return {"onehot": {"select": select, "select_computed": computed, "test": test, "test_computed": computed}}
    return {"x": probe(select_x), "y": probe(select_y)}


def test_site_selection_uses_the_select_split_and_frozen_tie_breaks():
    grid = {
        ("program_end", 10): _entry(0.9, 0.8),
        ("user_end", 5): _entry(0.85, 0.85, computed=0.4),
        ("answer", 3): _entry(0.85, 0.85, computed=0.6),
        ("question_mark", 3): _entry(0.85, 0.85, computed=0.6),
    }
    # max of min(x, y) = 0.85; computed tie 0.6; layer tie 3; earlier position wins
    assert ss.select_site(grid) == ("question_mark", 3)


def test_gate_g0_needs_both_overall_and_computed_accuracy():
    assert ss.gate_g0(_entry(0, 0, computed=0.5, test=0.8))["passed"] is True
    assert ss.gate_g0(_entry(0, 0, computed=0.49, test=0.95))["passed"] is False
    assert ss.gate_g0(_entry(0, 0, computed=0.9, test=0.79))["passed"] is False


def test_stage0_end_to_end_on_fixture_models(tmp_path, monkeypatch, model_factory, tokenizer):
    """Tiny random model, fixture tokens; the plumbing only, never a real-model result."""
    from test_pilot_stages import TINY_COUNTS, _inputs, _stub_load_model

    from open_weight_lingua.artifacts import RunDirectory
    from open_weight_lingua.splits import build_plan

    plan = build_plan(TINY_COUNTS)
    lock = tmp_path / "model-lock-gemma3-12b.json"
    lock.write_text(json.dumps({"models": {"target": {"revision": "r", "files": {}}}}))
    for split, name in (("calibration", "cal"), ("pilot", "pil")):
        inputs = _inputs(plan.groups[split])
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text(
            json.dumps({"inputs": inputs, "model_lock_sha256": ss.sha256_file(lock)}))
    monkeypatch.setattr(ss, "TRAIN_GROUPS", 2)
    monkeypatch.setattr(ss, "load_model", _stub_load_model(model_factory))
    monkeypatch.setattr(ss, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(ss, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(ss, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(ss, "compatibility", lambda device: {})
    monkeypatch.setattr(ss, "positions", lambda tok, row: {"program_end": 0, "query_variable": 1, "question_mark": 2, "user_end": 2, "answer": 2})
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path: tokenizer)
    args = ss.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                          "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"])
    run = RunDirectory(tmp_path / "survey")
    results, summary = ss.run(args, run, {})
    assert summary["rows"] == {"train": 8, "select": 4, "test": 16}
    assert len(results) == len(ss.POSITIONS) * summary["layers"]
    assert summary["selected_site"]["position"] in ss.POSITIONS
    assert set(summary["gate_G0"]) == {"held_out_min_accuracy", "held_out_min_computed_accuracy", "passed"}
    assert (run.path / "raw" / "features.safetensors").is_file()


def test_constant_features_give_finite_chance_level_predictions():
    x = np.ones((60, 8))
    v = np.arange(60) % ss.VALUES
    with np.errstate(all="raise"):
        out = ss.probe_accuracies(x[:40], x[40:50], x[50:], v[:40], v[40:50], v[50:],
                                  np.zeros(10, bool), np.zeros(10, bool))
    assert 0.0 <= out["onehot"]["test"] <= 0.2
