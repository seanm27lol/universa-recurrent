import json

import numpy as np

from open_weight_lingua import state_trace_survey as ts

PROMPT = "x = 9\ny = 8\ny = 3\ny = y + 3\nx = x + 1\nWhat is x? Reply with only the integer."


class CharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def test_boundaries_land_on_the_last_token_of_each_statement_from_line_two():
    text = "<u>" + PROMPT + "<m>\n"
    ids = [ord(c) for c in text]
    found = ts.boundary_positions(CharTokenizer(), ids, PROMPT)
    assert [text[: p + 1].split("\n")[-1] for p in found] == ["y = 8", "y = 3", "y = y + 3", "x = x + 1"]


def test_trace_labels_give_values_categories_and_computed_flags_per_line():
    labels = ts.trace_labels(PROMPT)
    assert [e["line"] for e in labels] == [2, 3, 4, 5]
    line4 = labels[2]
    assert line4["x"] == {"value": 9, "category": "carried", "computed": False, "copy_guess": 9}
    assert line4["y"] == {"value": 6, "category": "arithmetic", "computed": True, "copy_guess": 3}
    line5 = labels[3]
    assert line5["x"]["category"] == "arithmetic" and line5["x"]["value"] == 10
    assert line5["y"]["category"] == "carried" and line5["y"]["computed"] is True  # 6 never written
    assert labels[1]["y"]["category"] == "literal"


def test_gpu_style_probe_reads_a_linear_code_on_cpu():
    rng = np.random.default_rng(0)
    values = rng.integers(0, 20, 300)
    codebook = rng.normal(size=(20, 16))
    x = codebook[values] + 0.05 * rng.normal(size=(300, 16))
    (pred,) = ts.fit_predict(x[:200], values[:200], x[200:], (0.01,), "cpu")
    assert np.mean(pred == values[200:]) > 0.95


def _grid(carried, carried_computed, arithmetic):
    def side(**k):
        return {"select": dict(k), "test": dict(k)}
    def entry():
        return {v: side(carried=carried, carried_computed=carried_computed, arithmetic=arithmetic,
                        arithmetic_computed=arithmetic, literal=1.0, overall=0.5) for v in ("x", "y")}

    return {0: entry(), 1: entry()}


def test_gate_needs_carried_computed_state_and_arithmetic_results():
    assert ts.readings(_grid(0.9, 0.6, 0.85))["gate_G4"]["passed"] is True
    assert ts.readings(_grid(0.95, 0.3, 0.85))["gate_G4"]["passed"] is False  # last-literal memory is not state
    assert ts.readings(_grid(0.9, 0.6, 0.5))["gate_G4"]["passed"] is False
    assert ts.readings(_grid(0.9, 0.6, 0.85))["R1"]["layer"] == 0  # ties go to the earlier layer


def test_phase_four_end_to_end_on_fixture_models(tmp_path, monkeypatch, model_factory, tokenizer):
    """Tiny random model, fixture tokens; the plumbing only, never a real-model result."""
    from test_pilot_stages import TINY_COUNTS, _inputs, _stub_load_model

    from open_weight_lingua.artifacts import RunDirectory
    from open_weight_lingua.splits import build_plan

    plan = build_plan(TINY_COUNTS)
    lock = tmp_path / "model-lock-gemma3-12b.json"
    lock.write_text(json.dumps({"models": {"target": {"revision": "r", "files": {}}}}))
    for split, name in (("calibration", "cal"), ("pilot", "pil")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text(
            json.dumps({"inputs": _inputs(plan.groups[split]), "model_lock_sha256": ts.sha256_file(lock)}))
    monkeypatch.setattr(ts, "TRAIN_GROUPS", 2)
    monkeypatch.setattr(ts, "load_model", _stub_load_model(model_factory))
    monkeypatch.setattr(ts, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(ts, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(ts, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(ts, "compatibility", lambda device: {})
    monkeypatch.setattr(ts, "boundary_positions", lambda tok, ids, prompt: [min(i, len(ids) - 1) for i in range(len(ts.trace_labels(prompt)))])
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path: tokenizer)
    args = ts.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                          "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"])
    results, summary = ts.run(args, RunDirectory(tmp_path / "trace"), {})
    assert summary["rows"]["train"] > 0 and summary["rows"]["test"] > 0
    assert len(results) == summary["layers"]
    assert set(summary) >= {"R1", "R2", "gate_G4", "baselines"}
