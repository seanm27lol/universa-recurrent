import json

from open_weight_lingua import variable_choice_knockout as vck
from open_weight_lingua.tasks import Statement

BODY = "x = 13\ny = 5\ny = 1\nx = x + 3"
AFTER = BODY + "\nWhat is y? Reply with only the integer."


class CharChatTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [ord(c) for c in "<" + messages[0]["content"] + ">"]


def _item(prompt=AFTER, answer="1", variable="y"):
    return {"id": "calibration-0001-A-" + variable, "group_id": "calibration-0001", "prompt": prompt,
            "variable": variable, "answer": answer, "input_ids": [ord(c) for c in "<" + prompt + ">"]}


def test_rows_name_the_other_and_superseded_lines_in_both_orders():
    tok = CharChatTokenizer()
    for order in ("after", "first"):
        row = vck.choice_row(_item(), "calibration", order, tok)
        text = tok.decode(row["input_ids"])
        assert "".join(text[t] for t in row["keys"]["block_other"]) == "x = 13\nx = x + 3"
        assert "".join(text[t] for t in row["keys"]["block_superseded"]) == "y = 5\n"
        assert sorted(row["keys"]["block_both"]) == sorted(row["keys"]["block_other"] + row["keys"]["block_superseded"])
        assert row["values"] == {"answer": 1, "other": 16, "previous": 5}
        assert tok.decode(row["sequences"]["other"]["targets"]) == "16"
        assert not row["last_line_assigns_asked"] and not row["depends_on_other"]
    assert vck.choice_row(_item(), "calibration", "first", tok)["prompt"].startswith("What is y?")


def test_superseded_conditions_need_a_literal_final_with_an_earlier_assignment():
    tok = CharChatTokenizer()
    single = "x = 13\ny = 5\nx = x + 3"
    row = vck.choice_row(_item(single + "\nWhat is y? Reply with only the integer.", "5"), "calibration", "after", tok)
    assert set(row["keys"]) == {"block_other"} and "previous" not in row["values"]
    arithmetic = "x = 13\ny = 5\ny = y + 1\nx = 4"
    row = vck.choice_row(_item(arithmetic + "\nWhat is y? Reply with only the integer.", "6"), "calibration", "after", tok)
    assert set(row["keys"]) == {"block_other"} and row["values"]["previous"] == 5


def test_dependency_on_the_other_variable_follows_the_chain_back_to_a_literal():
    s = Statement
    assert vck.depends_on_other([s("x", "assign", 1), s("y", "assign", 2), s("x", "copy", "y"), s("x", "add", 3)], "x")
    assert not vck.depends_on_other([s("x", "assign", 1), s("y", "assign", 2), s("x", "copy", "y"), s("x", "assign", 3)], "x")
    assert not vck.depends_on_other([s("x", "assign", 1), s("y", "copy", "x"), s("x", "copy", "x")], "x")


def test_conditions_change_only_what_they_block(model_factory):
    from test_answer_knockout import _eager

    model = _eager(model_factory(layers=3))
    seq = {"ids": [5, 6, 7, 8, 9, 10, 11, 12], "positions": [7], "targets": [13]}
    row = {"id": "r", "span": [1, 4], "values": {"answer": 1, "other": 2},
           "keys": {"block_other": [1, 2], "block_superseded": [3], "block_both": [1, 2, 3]},
           "sequences": {"answer": seq, "other": {**seq, "targets": [14]}}}
    out = vck.evaluate_row(model, row, "cpu")
    assert set(out) == set(vck.CONDITIONS) and set(out["baseline"]) == {"answer", "other"}
    assert len({out[c]["answer"]["logprob"] for c in vck.CONDITIONS}) == 4
    del row["keys"]["block_superseded"], row["keys"]["block_both"]
    out = vck.evaluate_row(model, row, "cpu")
    assert out["block_superseded"] is None and out["block_both"] is None


def test_summary_reports_rates_per_order_and_stratum():
    def said(answer, other):
        return {"answer": {"correct": answer, "logprob": -1.0}, "other": {"correct": other, "logprob": -1.0}}

    rows, results = [], []
    for order in ("after", "first"):
        for split in ("calibration", "pilot"):
            rows.append({"order": order, "split": split, "last_line_assigns_asked": False, "depends_on_other": False,
                         "asked_last_update_kind": "literal", "values": {"answer": 1, "other": 2}, "keys": {"block_other": [0]}})
            confused = order == "first"
            results.append({"baseline": said(not confused, confused), "block_other": said(True, False),
                            "block_superseded": None, "block_both": None})
    summary = vck.summarize(rows, results)
    first = summary["first"]["last_line_other_no_dependency"]
    assert first["baseline"]["accuracy"] == 0.0 and first["baseline"]["says_other"] == 1.0
    assert first["block_other"]["accuracy"] == 1.0 and "block_superseded" not in first
    assert summary["after"]["all"]["baseline"]["accuracy"] == 1.0 and summary["first"]["superseded_literal"] == {}


def test_end_to_end_on_fixture_models(tmp_path, monkeypatch, model_factory, tokenizer):
    """Tiny random model and fixture tokens; plumbing only, never a real result."""
    import transformers

    from test_answer_knockout import _eager
    from test_pilot_stages import TINY_COUNTS, _inputs

    from open_weight_lingua.artifacts import RunDirectory
    from open_weight_lingua.splits import build_plan

    plan = build_plan(TINY_COUNTS)
    lock = tmp_path / "model-lock-qwen.json"
    lock.write_text(json.dumps({"models": {"target": {"revision": "r", "files": {}}}}))
    for split, name in (("calibration", "cal"), ("pilot", "pil")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.json").write_text(
            json.dumps({"inputs": _inputs(plan.groups[split]), "model_lock_sha256": vck.sha256_file(lock)}))
    monkeypatch.setattr(vck, "choice_row", lambda item, split, order, tok: {
        "id": item["id"], "group_id": item["group_id"], "split": split, "order": order, "span": [0, 1],
        "values": {"answer": 1}, "keys": {"block_other": [0]}, "last_line_assigns_asked": True,
        "depends_on_other": False, "asked_last_update_kind": "literal",
        "sequences": {"answer": {"ids": list(item["input_ids"]) + [10], "positions": [len(item["input_ids"]) - 1], "targets": [10]}}})
    monkeypatch.setattr(vck, "load_model", lambda path, role, device: _eager(model_factory(layers=3)))
    monkeypatch.setattr(vck, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(vck, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(vck, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(vck, "compatibility", lambda device: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: tokenizer)
    args = vck.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                           "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"])
    run = RunDirectory(tmp_path / "choice")
    summary = vck.run(args, run, {})
    manifest = json.loads((run.path / "manifest.json").read_text())
    inputs = sum(len(json.loads((tmp_path / n / "manifest.json").read_text())["inputs"]) for n in ("cal", "pil"))
    assert manifest["prompts"] == 2 * inputs and set(summary) == {"after", "first"}
    assert manifest["protocol_sha256"] == vck.sha256_file(vck.PROTOCOL)
    first = json.loads((run.path / "results.jsonl").read_text().splitlines()[0])
    assert set(first) == {"id", "order", *vck.CONDITIONS} and first["block_both"] is None
