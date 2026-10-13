import json

from open_weight_lingua import question_marking_knockout as qmk

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


def test_query_subsets_split_the_program_by_variable_and_last_line():
    tok = CharChatTokenizer()
    row = qmk.marking_row(_item(), "calibration", tok)
    text = tok.decode(row["input_ids"])
    pick = {name: "".join(text[t] for t in tokens) for name, tokens in row["queries"].items()}
    assert pick == {"program_blind": BODY, "asked_blind": "x = 7\n", "other_blind": "y = 16\ny = y + 1\ny = y - 3",
                    "last_line_blind": "y = y - 3"}
    assert text[row["question"][0] : row["question"][1] + 1] == "What is x? Reply with only the integer."


def test_each_condition_blocks_only_its_queries(model_factory):
    from test_answer_knockout import _eager

    model = _eager(model_factory(layers=3))
    seq = {"ids": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "positions": [9], "targets": [15]}
    row = {"id": "r", "question": [1, 3], "values": {"answer": 1}, "sequences": {"answer": seq},
           "queries": {"program_blind": [4, 5, 6, 7], "asked_blind": [4, 5], "other_blind": [6, 7], "last_line_blind": [7]}}
    out = qmk.evaluate_row(model, row, "cpu")
    assert set(out) == set(qmk.CONDITIONS)
    assert len({out[c]["answer"]["logprob"] for c in qmk.CONDITIONS}) == len(qmk.CONDITIONS)


def test_summary_strata():
    def said(answer, other):
        return {"answer": {"correct": answer, "logprob": -1.0}, "other": {"correct": other, "logprob": -1.0}}

    rows = [{"split": "pilot", "last_line_assigns_asked": asked, "depends_on_other": False, "values": {"answer": 1, "other": 2}}
            for asked in (True, False)]
    results = [{c: said(asked or c in ("baseline", "asked_blind"), not asked and c not in ("baseline", "asked_blind"))
                for c in qmk.CONDITIONS} for asked in (True, False)]
    summary = qmk.summarize(rows, results)
    main = summary["last_line_other_no_dependency"]
    assert main["asked_blind"]["accuracy"] == 1.0 and main["other_blind"]["says_other"] == 1.0
    assert summary["by_split"]["calibration"] == {} and summary["all"]["n"] == 2


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
            json.dumps({"inputs": _inputs(plan.groups[split]), "model_lock_sha256": qmk.sha256_file(lock)}))
    monkeypatch.setattr(qmk, "marking_row", lambda item, split, tok: {
        "id": item["id"], "group_id": item["group_id"], "split": split, "question": [0, 1], "values": {"answer": 1},
        "last_line_assigns_asked": True, "depends_on_other": False,
        "queries": {c: [2] for c in qmk.CONDITIONS[1:]},
        "sequences": {"answer": {"ids": list(item["input_ids"]) + [10], "positions": [len(item["input_ids"]) - 1], "targets": [10]}}})
    monkeypatch.setattr(qmk, "load_model", lambda path, role, device: _eager(model_factory(layers=3)))
    monkeypatch.setattr(qmk, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(qmk, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(qmk, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(qmk, "compatibility", lambda device: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: tokenizer)
    args = qmk.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                           "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"])
    run = RunDirectory(tmp_path / "marking")
    summary = qmk.run(args, run, {})
    manifest = json.loads((run.path / "manifest.json").read_text())
    inputs = sum(len(json.loads((tmp_path / n / "manifest.json").read_text())["inputs"]) for n in ("cal", "pil"))
    assert manifest["prompts"] == inputs and summary["all"]["n"] == inputs
    assert manifest["protocol_sha256"] == qmk.sha256_file(qmk.PROTOCOL)
    first = json.loads((run.path / "results.jsonl").read_text().splitlines()[0])
    assert set(first) == {"id", *qmk.CONDITIONS}
