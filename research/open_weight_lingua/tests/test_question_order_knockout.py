import json

import pytest

from open_weight_lingua import question_order_knockout as qok

BODY = "x = 2\ny = 3\nx = x + 9\ny = 5"
AFTER = BODY + "\nWhat is x? Reply with only the integer."


class CharChatTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [ord(c) for c in "<" + messages[0]["content"] + ">"]


def _item(prompt=AFTER, answer="11", variable="x"):
    return {"id": "calibration-0001-A-" + variable, "group_id": "calibration-0001", "prompt": prompt,
            "variable": variable, "answer": answer, "input_ids": [ord(c) for c in "<" + prompt + ">"]}


def test_both_orders_locate_the_program_and_its_last_line():
    tok = CharChatTokenizer()
    for order in qok.ORDERS:
        row = qok.order_row(_item(), "calibration", order, tok)
        text = tok.decode(row["input_ids"])
        first, last = row["span"]
        assert text[first : last + 1] == BODY
        assert "".join(text[t] for t in row["before_last_line"]) == "x = 2\ny = 3\nx = x + 9\n"
        assert tok.decode(row["sequence"]["targets"]) == "11"
        assert not row["last_line_assigns_asked"] and row["asked_last_update_kind"] == "arithmetic"
    first_order = qok.order_row(_item(), "calibration", "first", tok)
    assert first_order["prompt"] == "What is x? Reply with only the integer.\n" + BODY
    assert first_order["sequence"]["ids"][: len(first_order["input_ids"])] == first_order["input_ids"]


def test_recorded_ids_must_match_the_chat_template():
    item = _item()
    item["input_ids"] = item["input_ids"][1:]
    with pytest.raises(ValueError, match="chat template"):
        qok.order_row(item, "calibration", "after", CharChatTokenizer())


def test_only_last_line_blocks_post_program_positions_from_earlier_lines(model_factory):
    from test_answer_knockout import _eager

    model = _eager(model_factory(layers=8))  # a grid without repeated layers
    row = {"id": "r", "span": [1, 4], "before_last_line": [1, 2, 3],
           "sequence": {"ids": [5, 6, 7, 8, 9, 10, 11, 12], "positions": [7], "targets": [13]}}
    out = qok.evaluate_row(model, row, 8, "cpu")
    assert list(out) == qok.conditions(8)
    assert out["only_last_line"]["logprob"] != out["baseline"]["logprob"]
    assert out["after_program@0"]["logprob"] != out["only_last_line"]["logprob"]  # the last line stays visible


def test_summary_reports_release_layers_and_strata_per_order():
    layers = 8
    rows, results = [], []
    for order in qok.ORDERS:
        for split in ("calibration", "pilot"):
            for asked, kind in ((False, "literal"), (True, "arithmetic")):
                rows.append({"order": order, "split": split, "last_line_assigns_asked": asked, "asked_last_update_kind": kind})
                # `after` recovers from layer 4; `first` only at the last grid layer.
                release = 4 if order == "after" else 7
                result = {c: {"correct": True, "logprob": -1.0} for c in qok.conditions(layers)}
                for k in range(release):
                    result[f"after_program@{k}"]["correct"] = False
                result["only_last_line"]["correct"] = asked
                results.append(result)
    summary = qok.summarize(rows, results, layers)
    assert summary["orders"]["after"]["release_layer"] == {"calibration": 4, "pilot": 4}
    assert summary["orders"]["first"]["release_layer"] == {"calibration": 7, "pilot": 7}
    strata = summary["orders"]["first"]["strata"]
    assert strata["last_line_assigns_other"]["only_last_line"] == 0.0 and strata["last_line_assigns_asked_arithmetic"]["only_last_line"] == 1.0
    assert strata["last_line_assigns_asked_copy"] == {}


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
            json.dumps({"inputs": _inputs(plan.groups[split]), "model_lock_sha256": qok.sha256_file(lock)}))
    monkeypatch.setattr(qok, "order_row", lambda item, split, order, tok: {
        "id": item["id"], "group_id": item["group_id"], "split": split, "order": order, "span": [0, 1], "before_last_line": [0],
        "last_line_assigns_asked": True, "asked_last_update_kind": "literal",
        "sequence": {"ids": list(item["input_ids"]) + [10], "positions": [len(item["input_ids"]) - 1], "targets": [10]}})
    monkeypatch.setattr(qok, "load_model", lambda path, role, device: _eager(model_factory(layers=3)))
    monkeypatch.setattr(qok, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(qok, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(qok, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(qok, "compatibility", lambda device: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: tokenizer)
    args = qok.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                           "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"])
    run = RunDirectory(tmp_path / "order")
    summary = qok.run(args, run, {})
    manifest = json.loads((run.path / "manifest.json").read_text())
    inputs = sum(len(json.loads((tmp_path / n / "manifest.json").read_text())["inputs"]) for n in ("cal", "pil"))
    assert manifest["prompts"] == 2 * inputs and set(summary["orders"]) == {"after", "first"}
    assert manifest["protocol_sha256"] == qok.sha256_file(qok.PROTOCOL)
    first = json.loads((run.path / "results.jsonl").read_text().splitlines()[0])
    assert first["order"] == "after" and set(first) == {"id", "order", *qok.conditions(3)}
