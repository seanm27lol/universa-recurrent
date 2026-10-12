import json

from open_weight_lingua import dependency_knockout as dk
from open_weight_lingua.description_census import program
from open_weight_lingua.span_knockout import token_lines

PROGRAM = "y = 16\nx = 7\nx = y\ny = y - 3"
PROMPT = PROGRAM + "\nWhat is x? Reply with only the integer."


class CharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def _setup(prompt=PROMPT, body=PROGRAM):
    text = "<u>" + prompt + "<m>\n"
    ids = [ord(c) for c in text]
    first = text.index(body)
    span = (first, first + len(body) - 1)
    return text, span, token_lines(CharTokenizer(), ids, span, body)


def test_copy_depends_on_the_copied_variables_earlier_lines():
    text, span, lines = _setup()
    dep = dk.dependency(program(PROMPT), "x", span, lines)
    assert dep["kind"] == "cross_copy" and dep["final_line"] == 2 and dep["dependency_lines"] == [0]
    assert "".join(text[t] for t in dep["queries"]) == "x = y\n"
    assert "".join(text[t] for t in dep["keys"]) == "y = 16\n"  # the later y line is not a dependency


def test_arithmetic_and_literal_depend_on_the_asked_variables_earlier_lines():
    body = "x = 7\ny = 3\nx = x + 2\ny = 5"
    prompt = body + "\nWhat is x? Reply with only the integer."
    text, span, lines = _setup(prompt, body)
    dep = dk.dependency(program(prompt), "x", span, lines)
    assert dep["kind"] == "arithmetic" and "".join(text[t] for t in dep["keys"]) == "x = 7\n"
    literal = dk.dependency(program(prompt.replace("What is x", "What is y")), "y", span, lines)
    assert literal["kind"] == "literal" and "".join(text[t] for t in literal["keys"]) == "y = 3\n"  # the placebo


def test_a_self_copy_depends_on_the_asked_variables_own_earlier_lines():
    body = "x = 7\ny = 3\nx = x\ny = 5"
    prompt = body + "\nWhat is x? Reply with only the integer."
    text, span, lines = _setup(prompt, body)
    dep = dk.dependency(program(prompt), "x", span, lines)
    assert dep["kind"] == "self_copy" and "".join(text[t] for t in dep["keys"]) == "x = 7\n"


def test_a_first_literal_has_no_dependency_and_is_skipped(model_factory):
    body = "x = 7\ny = 3\ny = y + 1"
    prompt = body + "\nWhat is x? Reply with only the integer."
    _, span, lines = _setup(prompt, body)
    dep = dk.dependency(program(prompt), "x", span, lines)
    assert dep["kind"] == "literal" and dep["keys"] == []
    assert dk.evaluate_row(None, {"keys": []}, 4, "cpu") is None


def test_evaluate_row_runs_every_grid_layer(model_factory):
    from test_answer_knockout import _eager

    model = _eager(model_factory(layers=8))
    row = {"id": "r", "keys": [1, 2], "queries": [4, 5],
           "sequence": {"ids": [5, 6, 7, 8, 9, 10, 11, 12], "positions": [6, 7], "targets": [12, 13]}}
    out = dk.evaluate_row(model, row, 8, "cpu")
    assert set(out) == {"baseline", *(f"dependency@{k}" for k in dk.layer_grid(8))}


def test_release_layer_per_kind():
    grid = dk.layer_grid(48)
    rows, results = [], []
    curves = {"self_copy": [0.2, 0.2, 0.3, 0.9, 0.95, 0.96, 0.96, 0.96], "literal": [0.96] * 8, "arithmetic": [0.5] * 8}
    for kind, curve in curves.items():
        for i in range(100):
            rows.append({"kind": kind, "split": "calibration" if i < 50 else "pilot"})
            results.append({"baseline": {"correct": i % 50 < 48}, **{f"dependency@{k}": {"correct": i % 50 < a * 50} for k, a in zip(grid, curve)}})
    rows.append({"kind": "literal", "split": "calibration"})
    results.append(None)
    summary = dk.summarize(rows, results, 48)
    assert summary["release_layer"]["self_copy"] == {"calibration": 24, "pilot": 24}
    assert summary["release_layer"]["literal"] == {"calibration": 0, "pilot": 0}
    assert summary["release_layer"]["arithmetic"] == {"calibration": None, "pilot": None}
    assert summary["skipped_no_dependency"] == 1 and summary["release_layer"]["cross_copy"] == {}


def test_end_to_end_on_fixture_models(tmp_path, monkeypatch, model_factory, tokenizer):
    """Tiny random model and fixture tokens; plumbing only, never a real result."""
    from test_answer_knockout import _eager
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
            json.dumps({"inputs": _inputs(plan.groups[split]), "model_lock_sha256": dk.sha256_file(lock)}))
    monkeypatch.setattr(dk, "program_span", lambda tok, ids, body: (0, 1))
    monkeypatch.setattr(dk, "token_lines", lambda tok, ids, span, body: [0, 1])
    monkeypatch.setattr(dk, "load_model", lambda path, role, device: _eager(model_factory(layers=8)))
    monkeypatch.setattr(dk, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(dk, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(dk, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(dk, "compatibility", lambda device: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: tokenizer)
    args = dk.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                          "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"])
    run = RunDirectory(tmp_path / "dependency")
    summary = dk.run(args, run, {})
    assert summary["grid"] == dk.layer_grid(8) and set(summary["release_layer"]) == set(dk.KINDS)
    lines = (run.path / "results.jsonl").read_text().splitlines()
    assert len(lines) == json.loads((run.path / "manifest.json").read_text())["prompts"]
