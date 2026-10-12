import json

from open_weight_lingua import supersession_knockout as sk

BODY = "x = 4\ny = 3\nx = 7\ny = 5"
PROMPT = BODY + "\nWhat is x? Reply with only the integer."


class CharChatTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def _item(prompt=PROMPT, variable="x", answer="7"):
    ids = [ord(c) for c in "<" + prompt + ">"]
    return {"id": "calibration-0001-A-" + variable, "group_id": "calibration-0001", "prompt": prompt,
            "variable": variable, "answer": answer, "input_ids": ids}


def test_literal_row_has_same_and_other_variable_keys_and_the_superseded_value():
    tok = CharChatTokenizer()
    row = sk.supersession_row(_item(), "calibration", tok)
    text = tok.decode(row["input_ids"])
    assert row["kind"] == "literal" and row["superseded"] == "4"
    assert "".join(text[t] for t in row["queries"]) == "x = 7\n"
    assert "".join(text[t] for t in row["keys"]["same_variable"]) == "x = 4\n"
    assert "".join(text[t] for t in row["keys"]["other_variable"]) == "y = 3\n"  # only lines before the final statement
    assert tok.decode(row["superseded_sequence"]["targets"]) == "4"


def test_value_before_follows_the_program_up_to_the_final_line():
    from open_weight_lingua.description_census import program

    statements = program("x = 4\ny = 3\nx = x + 2\ny = x\nx = 7\nWhat is x? Reply with only the integer.")
    assert sk.value_before(statements, 4, "x") == 6 and sk.value_before(statements, 4, "y") == 6
    assert sk.value_before(statements, 0, "x") is None


def test_out_of_scope_and_equal_superseded_values():
    tok = CharChatTokenizer()
    first_literal = "y = 3\nx = 7\ny = y + 1"
    assert sk.supersession_row(_item(first_literal + "\nWhat is x? Reply with only the integer."), "calibration", tok) is None
    same = "x = 7\ny = 3\nx = 7\ny = 5"
    row = sk.supersession_row(_item(same + "\nWhat is x? Reply with only the integer."), "calibration", tok)
    assert row["superseded"] == "7" and "superseded_sequence" not in row  # no preference when old == new


def test_evaluate_row_scores_the_answer_and_the_superseded_value(model_factory):
    from test_answer_knockout import _eager

    model = _eager(model_factory(layers=3))
    row = {"id": "r", "queries": [4, 5], "keys": {"same_variable": [1, 2], "other_variable": []},
           "sequence": {"ids": [5, 6, 7, 8, 9, 10, 11, 12], "positions": [6, 7], "targets": [12, 13]},
           "superseded_sequence": {"ids": [5, 6, 7, 8, 9, 10, 11, 14], "positions": [6, 7], "targets": [14, 15]}}
    out = sk.evaluate_row(model, row, "cpu")
    assert out["other_variable"] is None
    assert set(out["same_variable"]) == {"correct", "logprob", "superseded_correct", "preference"}
    assert out["baseline"]["preference"] is not None


def test_summary_reports_accuracy_superseded_rate_and_preference():
    rows = [{"kind": "literal", "split": "calibration", "keys": {"same_variable": [1, 2], "other_variable": [3]}} for _ in range(4)]
    def cond(correct, old):
        return {"correct": correct, "logprob": -1.0, "superseded_correct": old, "preference": 1.0 if old else -2.0}
    results = [{"baseline": cond(True, False), "same_variable": cond(False, True), "other_variable": cond(True, False)} for _ in rows]
    summary = sk.summarize(rows, results)["literal/calibration"]
    assert summary["same_variable"]["accuracy"] == 0.0 and summary["same_variable"]["superseded_rate"] == 1.0
    assert summary["other_variable"]["accuracy"] == 1.0 and summary["mean_keys"] == {"same_variable": 2.0, "other_variable": 1.0}


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
    monkeypatch.setattr(sk, "program_span", lambda tok, ids, body: (0, 1))
    monkeypatch.setattr(sk, "token_lines", lambda tok, ids, span, body: [0, 1])
    monkeypatch.setattr(sk, "dependency", lambda st, v, span, lines: {"kind": "literal", "final_line": 1, "queries": [1], "keys": [0]})
    monkeypatch.setattr(sk, "load_model", lambda path, role, device: _eager(model_factory(layers=3)))
    monkeypatch.setattr(sk, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(sk, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(sk, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(sk, "compatibility", lambda device: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: tokenizer)
    args = sk.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                          "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"])
    run = RunDirectory(tmp_path / "supersession")
    summary = sk.run(args, run, {})
    assert "literal/all" in summary
    lines = (run.path / "results.jsonl").read_text().splitlines()
    assert len(lines) == json.loads((run.path / "manifest.json").read_text())["prompts"] > 0
