import json

import torch

from open_weight_lingua import selection_knockout as sel

BODY = "x = 2\ny = 3\nx = x + 9\ny = 5\nx = 7"
PROMPT = BODY + "\nWhat is x? Reply with only the integer."


class CharChatTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def _item(prompt=PROMPT, answer="7", variable="x"):
    return {"id": "calibration-0001-A-" + variable, "group_id": "calibration-0001", "prompt": prompt, "variable": variable,
            "answer": answer, "input_ids": [ord(c) for c in "<" + prompt + ">"]}


def test_eligible_row_names_the_three_values_and_the_two_statements():
    tok = CharChatTokenizer()
    row = sel.selection_row(_item(), "reused", tok)
    text = tok.decode(row["input_ids"])
    assert row["values"] == {"v0": 7, "v1": 11, "v2": 2}
    assert "".join(text[t] for t in row["final"]["queries"]) == "x = 7"
    assert "".join(text[t] for t in row["final"]["keys"]) == "x = 2\ny = 3\nx = x + 9\ny = 5\n"
    assert "".join(text[t] for t in row["previous"]["queries"]) == "x = x + 9\n"
    assert "".join(text[t] for t in row["previous"]["keys"]) == "x = 2\ny = 3\n"
    assert tok.decode(row["sequences"]["v1"]["targets"]) == "11"


def test_ineligible_prompts_are_skipped():
    tok = CharChatTokenizer()
    two = "x = 2\ny = 3\nx = 7\ny = 5"
    assert sel.selection_row(_item(two + "\nWhat is x? Reply with only the integer."), "reused", tok) is None
    arithmetic_final = "x = 2\ny = 3\nx = 5\ny = 5\nx = x + 1"
    assert sel.selection_row(_item(arithmetic_final + "\nWhat is x? Reply with only the integer.", "6"), "reused", tok) is None
    repeated = "x = 2\ny = 3\nx = 7\ny = 5\nx = 2"
    assert sel.selection_row(_item(repeated + "\nWhat is x? Reply with only the integer.", "2"), "reused", tok) is None


def test_two_nested_degradations_both_apply(model_factory):
    from test_answer_knockout import _eager

    model = _eager(model_factory(layers=3))
    ids = [5, 6, 7, 8, 9, 10, 11, 12]
    row = {"final": {"queries": [5, 6], "keys": [1, 2, 3, 4]}, "previous": {"queries": [3, 4], "keys": [1, 2]}}
    tensor = torch.tensor([ids])

    def logits(condition):
        with torch.no_grad(), sel.degraded(model, row, condition):
            return model(input_ids=tensor, attention_mask=torch.ones_like(tensor), use_cache=False).logits

    base, final, both, previous = (logits(c) for c in sel.CONDITIONS)
    assert torch.equal(base[0, :3], final[0, :3]) and not torch.equal(base[0, 5:], final[0, 5:])
    assert not torch.equal(base[0, 3:5], previous[0, 3:5]) and torch.equal(base[0, :3], previous[0, :3])
    assert not torch.equal(both[0, 3:], final[0, 3:])  # the second degradation adds to the first


def test_summary_counts_which_value_is_said():
    rows = [{"source": "reused"}, {"source": "fresh"}]
    def said(v0, v1, v2):
        return {"v0": {"correct": v0, "logprob": -1.0}, "v1": {"correct": v1, "logprob": -2.0}, "v2": {"correct": v2, "logprob": -3.0}}
    results = [{"baseline": said(True, False, False), "final": said(False, True, False),
                "final_and_previous": said(False, False, True), "previous": said(True, False, False)} for _ in rows]
    summary = sel.summarize(rows, results)["all"]
    assert summary["final"]["among_wrong"] == {"v1": 1.0, "v2": 0.0}
    assert summary["final_and_previous"]["says"]["v2"] == 1.0 and summary["previous"]["says"]["v0"] == 1.0


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
            json.dumps({"inputs": _inputs(plan.groups[split]), "model_lock_sha256": sel.sha256_file(lock)}))
    fresh = plan.groups["pilot"][:1]
    monkeypatch.setattr(sel, "build_plan", lambda: plan)
    monkeypatch.setattr(sel, "first_fresh_groups", lambda p: (fresh, {"inventory_sha256": "a"}))
    monkeypatch.setattr(sel, "second_fresh_groups", lambda p: (fresh, {"inventory_sha256": "b"}))
    monkeypatch.setattr(sel, "FRESH_INVENTORIES", {"fresh_2026_10": "a", "fresh_2026_10b": "b"})
    monkeypatch.setattr(sel, "selection_row", lambda item, source, tok: {
        "id": item["id"], "group_id": item["group_id"], "source": source, "values": {"v0": 7, "v1": 11, "v2": 2},
        "final": {"queries": [1], "keys": [0]}, "previous": {"queries": [0], "keys": []},
        "sequences": {v: {"ids": list(item["input_ids"]) + [10 + i], "positions": [len(item["input_ids"]) - 1], "targets": [10 + i]}
                      for i, v in enumerate(sel.VALUES)}})
    monkeypatch.setattr(sel, "load_model", lambda path, role, device: _eager(model_factory(layers=3)))
    monkeypatch.setattr(sel, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(sel, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(sel, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(sel, "compatibility", lambda device: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: tokenizer)
    args = sel.parse_args(["--lock", str(lock), "--calibration-run", str(tmp_path / "cal"),
                           "--pilot-run", str(tmp_path / "pil"), "--device", "cpu"])
    run = RunDirectory(tmp_path / "selection")
    summary = sel.run(args, run, {})
    manifest = json.loads((run.path / "manifest.json").read_text())
    assert manifest["by_source"]["fresh"] == 8 and set(summary) == {"all", "reused", "fresh"}
