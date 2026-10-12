import json

import pytest

from open_weight_lingua import question_route_knockout as qrk

BODY = "x = 7\ny = 16\ny = y + 1\ny = y - 3"
AFTER = BODY + "\nWhat is x? Reply with only the integer."


class CharChatTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [ord(c) for c in "<" + messages[0]["content"] + ">"]


def _item(prompt=AFTER, answer="7", variable="x"):
    return {"id": "calibration-0001-A-" + variable, "group_id": "calibration-0001", "prompt": prompt,
            "variable": variable, "answer": answer, "input_ids": [ord(c) for c in "<" + prompt + ">"]}


def test_row_locates_the_question_its_name_and_the_program():
    tok = CharChatTokenizer()
    row = qrk.route_row(_item(), "calibration", tok)
    text = tok.decode(row["input_ids"])
    assert text.startswith("<What is x?")
    assert text[row["question"][0] : row["question"][1] + 1] == "What is x? Reply with only the integer."
    assert text[row["name"]] == "x" and row["question"][0] < row["name"] < row["question"][1]
    assert text[row["span"][0] : row["span"][1] + 1] == BODY
    assert row["values"] == {"answer": 7, "other": 14} and not row["last_line_assigns_asked"]
    assert tok.decode(row["sequences"]["other"]["targets"]) == "14"


def test_name_token_must_be_unique():
    tok = CharChatTokenizer()
    ids = [ord(c) for c in "x is x"]
    with pytest.raises(ValueError, match="exactly one"):
        qrk.name_token(tok, ids, (0, 5), "x")


def test_blocks_name_the_right_queries_and_keys():
    row = {"question": [1, 10], "name": 4, "span": [12, 20]}
    reader, program, question = list(range(21, 25)), list(range(12, 21)), list(range(1, 11))
    assert qrk.blocks(row, 25, "baseline") == []
    assert qrk.blocks(row, 25, "reader_blind") == [(reader, question)]
    assert qrk.blocks(row, 25, "program_blind_name") == [(program, [4])]
    assert qrk.blocks(row, 25, "both_blind") == [(reader, question), (program, question)]


def test_conditions_differ_and_nest(model_factory):
    from test_answer_knockout import _eager

    model = _eager(model_factory(layers=3))
    seq = {"ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "positions": [9], "targets": [15]}
    row = {"id": "r", "question": [1, 3], "name": 2, "span": [4, 7], "values": {"answer": 1},
           "sequences": {"answer": seq}}
    out = qrk.evaluate_row(model, row, "cpu")
    assert set(out) == set(qrk.CONDITIONS)
    logprobs = {c: out[c]["answer"]["logprob"] for c in qrk.CONDITIONS}
    assert len(set(logprobs.values())) == len(qrk.CONDITIONS)


def test_summary_strata():
    def said(answer, other):
        return {"answer": {"correct": answer, "logprob": -1.0}, "other": {"correct": other, "logprob": -1.0}}

    rows = [{"split": "calibration", "last_line_assigns_asked": asked, "depends_on_other": False, "values": {"answer": 1, "other": 2}}
            for asked in (True, False)]
    results = [{c: said(asked or c == "baseline", not asked and c != "baseline") for c in qrk.CONDITIONS} for asked in (True, False)]
    summary = qrk.summarize(rows, results)
    main = summary["last_line_other_no_dependency"]
    assert main["baseline"]["accuracy"] == 1.0 and main["reader_blind"]["accuracy"] == 0.0 and main["reader_blind"]["says_other"] == 1.0
    assert summary["last_line_asked"]["reader_blind"]["accuracy"] == 1.0 and summary["all"]["n"] == 2


def test_end_to_end_on_fixture_models(tmp_path, monkeypatch, model_factory, tokenizer):
    """Tiny random model and fixture tokens; plumbing only, never a real result."""
    import transformers

    from test_answer_knockout import _eager
    from test_pilot_stages import TINY_COUNTS, _inputs

    from open_weight_lingua.artifacts import RunDirectory
    from open_weight_lingua.splits import build_plan

    plan = build_plan(TINY_COUNTS)
    lock = tmp_path / "model-lock-gemma3-12b.json"
    lock.write_text(json.dumps({"models": {"target": {"revision": "r", "files": {}}}}))
    for split, name in (("calibration", "cal"), ("pilot", "pil")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text(
            json.dumps({"inputs": _inputs(plan.groups[split]), "model_lock_sha256": qrk.sha256_file(lock)}))
    monkeypatch.setattr(qrk, "route_row", lambda item, split, tok: {
        "id": item["id"], "group_id": item["group_id"], "split": split, "question": [0, 1], "name": 1, "span": [2, 3],
        "values": {"answer": 1}, "last_line_assigns_asked": True, "depends_on_other": False,
        "sequences": {"answer": {"ids": list(item["input_ids"]) + [10], "positions": [len(item["input_ids"]) - 1], "targets": [10]}}})
    monkeypatch.setattr(qrk, "load_model", lambda path, role, device: _eager(model_factory(layers=3)))
    monkeypatch.setattr(qrk, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(qrk, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(qrk, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(qrk, "compatibility", lambda device: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: tokenizer)
    args = qrk.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                           "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"])
    run = RunDirectory(tmp_path / "route")
    summary = qrk.run(args, run, {})
    manifest = json.loads((run.path / "manifest.json").read_text())
    inputs = sum(len(json.loads((tmp_path / n / "manifest.json").read_text())["inputs"]) for n in ("cal", "pil"))
    assert manifest["prompts"] == inputs and summary["all"]["n"] == inputs
    assert manifest["protocol_sha256"] == qrk.sha256_file(qrk.PROTOCOL)
    first = json.loads((run.path / "results.jsonl").read_text().splitlines()[0])
    assert set(first) == {"id", *qrk.CONDITIONS}
