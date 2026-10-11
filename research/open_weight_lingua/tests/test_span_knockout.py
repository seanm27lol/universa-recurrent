import json

from open_weight_lingua import span_knockout as sk
from open_weight_lingua.description_census import program

PROGRAM = "x = 7\ny = 16\ny = y + 1\ny = y - 3"
PROMPT = PROGRAM + "\nWhat is x? Reply with only the integer."


class CharTokenizer:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def _ids():
    text = "<u>" + PROMPT + "<m>\n"
    return text, [ord(c) for c in text]


def test_token_lines_assign_each_character_and_newline_to_its_line():
    text, ids = _ids()
    first = text.index(PROGRAM)
    span = (first, first + len(PROGRAM) - 1)
    lines = sk.token_lines(CharTokenizer(), ids, span, PROGRAM)
    assert lines[:6] == [0, 0, 0, 0, 0, 0]  # "x = 7\n": the newline ends line 0
    assert lines[6] == 1 and lines[-1] == 3
    assert len(lines) == len(PROGRAM)


def test_spans_separate_the_last_token_and_the_asked_lines():
    text, ids = _ids()
    first = text.index(PROGRAM)
    span = (first, first + len(PROGRAM) - 1)
    lines = sk.token_lines(CharTokenizer(), ids, span, PROGRAM)
    spans = sk.spans_for(program(PROMPT), "x", span, lines)
    decode = lambda keys: "".join(text[k] for k in keys)  # noqa: E731
    assert decode(spans["last_token"]) == "3"
    assert decode(spans["asked_lines_minus_last"]) == "x = 7\n"
    assert decode(spans["asked_final_line_minus_last"]) == "x = 7\n"
    assert decode(spans["other_lines_minus_last"]) == "y = 16\ny = y + 1\ny = y - "
    assert sorted(spans["all_minus_last"] + spans["last_token"]) == spans["full"]
    asked_y = sk.spans_for(program(PROMPT.replace("What is x", "What is y")), "y", span, lines)
    assert decode(asked_y["asked_final_line_minus_last"]) == "y = y - "  # the last token is excluded


def test_evaluate_row_blocks_only_the_chosen_span(model_factory):
    from test_answer_knockout import _eager

    model = _eager(model_factory(layers=4))
    row = {"id": "r", "span": [1, 4], "sequence": {"ids": [5, 6, 7, 8, 9, 10, 11, 12], "positions": [6, 7], "targets": [12, 13]},
           "spans": {"full": [1, 2, 3, 4], "last_token": [4], "all_minus_last": [1, 2, 3], "asked_lines_minus_last": [1, 2],
                     "other_lines_minus_last": [3], "asked_final_line_minus_last": []}}
    out = sk.evaluate_row(model, row, ("answer", 1), "cpu")
    assert set(out) == {"baseline", *sk.SPANS}
    assert out["asked_final_line_minus_last"] is None
    assert all(set(out[name]) == {"correct", "logprob"} for name in sk.SPANS if out[name] is not None)


def test_summary_reports_loss_per_span_and_skips_empty_spans():
    rows = [{"split": "calibration" if i < 6 else "pilot", "asked_last_update_kind": "literal", "last_line_assigns_asked": i % 2 == 0,
             "asked_copies_other": False, "spans": {name: [1] for name in sk.SPANS}} for i in range(8)]
    results = [{"baseline": {"correct": True, "logprob": 0.0},
                **{name: {"correct": name != "full", "logprob": 0.0} for name in sk.SPANS}} for _ in rows]
    results[0]["asked_final_line_minus_last"] = None
    summary = sk.summarize(rows, results)
    calibration = summary["by_split"]["calibration"]
    assert calibration["full"]["loss"] == 1.0 and calibration["last_token"]["loss"] == 0.0
    assert calibration["asked_final_line_minus_last"]["n"] == 5


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
            json.dumps({"inputs": _inputs(plan.groups[split]), "model_lock_sha256": sk.sha256_file(lock)}))
    monkeypatch.setattr(sk, "ROUTES", {"gemma3-12b": ("after_program", 1)})
    monkeypatch.setattr(sk, "program_span", lambda tok, ids, body: (0, 1))
    monkeypatch.setattr(sk, "token_lines", lambda tok, ids, span, body: [0, 1])
    monkeypatch.setattr(sk, "load_model", lambda path, role, device: _eager(model_factory(layers=3)))
    monkeypatch.setattr(sk, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(sk, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(sk, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(sk, "compatibility", lambda device: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: tokenizer)
    args = sk.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                          "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"])
    run = RunDirectory(tmp_path / "span")
    summary = sk.run(args, run, {})
    assert set(summary["by_split"]) == {"calibration", "pilot"} and set(summary["span_sizes"]) == set(sk.SPANS)
    lines = (run.path / "results.jsonl").read_text().splitlines()
    assert len(lines) == json.loads((run.path / "manifest.json").read_text())["prompts"]
