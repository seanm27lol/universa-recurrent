import json

import pytest

from open_weight_lingua import answer_knockout_confirm as akc
from open_weight_lingua.splits import build_plan


class CharChatTokenizer:
    """Character-level stand-in with a minimal chat template; eos is chr(1)."""

    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return self.encode("<" + messages[0]["content"] + ">")


def _tiny_plan():
    from test_pilot_stages import TINY_COUNTS

    return build_plan(TINY_COUNTS)


def test_validation_rows_locate_spans_and_teacher_forced_targets(monkeypatch):
    plan = _tiny_plan()
    monkeypatch.setattr(akc, "PLAN_HASH", plan.plan_hash)
    tokenizer = CharChatTokenizer()
    rows = akc.validation_rows(plan, tokenizer)
    assert len(rows) == 4 * len(plan.groups["validation_b"])
    row = rows[0]
    first, last = row["span"]
    assert tokenizer.decode(row["input_ids"][first : last + 1]) == row["prompt"].rsplit("\nWhat is", 1)[0]
    assert tokenizer.decode(row["sequence"]["targets"]) == row["answer"]
    assert row["sequence"]["positions"][0] == len(row["input_ids"]) - 1


def test_a_different_plan_is_refused():
    with pytest.raises(ValueError, match="frozen Phase Two plan"):
        akc.validation_rows(_tiny_plan(), CharChatTokenizer())


def _fake_run(tmp_path, family, correct_by_condition, groups=40):
    run = tmp_path / family
    run.mkdir()
    stimuli, results = [], []
    for g in range(groups):
        for k in range(4):
            sid = f"validation_b-{g:04d}-{'AB'[k // 2]}-{'xy'[k % 2]}"
            stimuli.append({"id": sid, "group_id": f"validation_b-{g:04d}", "asked_last_update_kind": "literal", "last_line_assigns_asked": k % 2 == 0})
            results.append({"id": sid, **{c: {"correct": f(g, k), "logprob": 0.0} for c, f in correct_by_condition.items()}})
    (run / "stimuli.json").write_text(json.dumps(stimuli))
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    (run / "manifest.json").write_text(json.dumps({"family": family, "prompts": len(results), "plan_hash": akc.PLAN_HASH,
                                                   "protocol_sha256": akc.sha256_file(akc.PROTOCOL),
                                                   "stimuli_sha256": akc.sha256_file(run / "stimuli.json")}))
    (run / "completion.json").write_text(json.dumps({"status": "COMPLETE"}))
    return run


def test_verdicts_follow_the_frozen_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(akc, "RESAMPLES", 2000)
    always, never = (lambda g, k: True), (lambda g, k: False)
    gemma = _fake_run(tmp_path, "gemma3-12b", {"baseline": always, "after_program@24": never, "after_program@30": always,
                                               "answer@0": lambda g, k: (g * 4 + k) % 100 != 0}, groups=100)
    qwen = _fake_run(tmp_path, "qwen2.5-7b", {"baseline": always, "answer@21": never, "answer@24": always,
                                              "after_program@24": lambda g, k: k != 0}, groups=100)
    result = akc.evaluate([gemma, qwen])
    status = {h["id"]: h["status"] for h in result["hypotheses"]}
    assert status == {"G1": "SUPPORTED", "G2": "SUPPORTED", "G3": "SUPPORTED",
                      "Q1": "SUPPORTED", "Q2": "SUPPORTED", "Q3": "NOT_SUPPORTED"}  # Q3 loses 25 points
    by_id = {h["id"]: h for h in result["hypotheses"]}
    assert by_id["G2"]["mean_loss"] == pytest.approx(1.0) and by_id["Q3"]["mean_loss"] == pytest.approx(0.25)
    assert set(result["families"]["gemma3-12b"]["strata"]) == {"asked_last_update_kind", "last_line_assigns_asked"}


def test_incomplete_runs_are_refused(tmp_path):
    run = _fake_run(tmp_path, "gemma3-12b", {"baseline": lambda g, k: True, "after_program@24": lambda g, k: True,
                                              "after_program@30": lambda g, k: True, "answer@0": lambda g, k: True})
    (run / "completion.json").write_text(json.dumps({"status": "FAILED"}))
    with pytest.raises(ValueError, match="incomplete"):
        akc.evaluate([run])


def test_run_end_to_end_on_a_fixture_model(tmp_path, monkeypatch, model_factory, tokenizer):
    """Tiny random model and fixture tokens; plumbing only, never a real result."""
    import transformers

    from test_answer_knockout import _eager

    plan = _tiny_plan()
    lock = tmp_path / "model-lock-gemma3-12b.json"
    lock.write_text(json.dumps({"models": {"target": {"revision": "r", "files": {}}}}))
    monkeypatch.setattr(akc, "PLAN_HASH", plan.plan_hash)
    monkeypatch.setattr(akc, "build_plan", lambda: plan)
    monkeypatch.setattr(akc, "KNOCKOUTS", {"gemma3-12b": (("after_program", 1), ("after_program", 2), ("answer", 0))})
    monkeypatch.setattr(akc, "program_span", lambda tok, ids, body: (0, 1))
    monkeypatch.setattr(akc, "load_model", lambda path, role, device: _eager(model_factory(layers=4)))
    monkeypatch.setattr(akc, "read_lock", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(akc, "model_paths", lambda lock, cache: {"target": tmp_path})
    monkeypatch.setattr(akc, "verify_models", lambda lock, paths: {})
    monkeypatch.setattr(akc, "compatibility", lambda device: {})
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda path, **kwargs: tokenizer)
    run_dir = tmp_path / "confirm"
    akc.main(["run", "--lock", str(lock), "--device", "cpu", "--run-dir", str(run_dir)])
    completion = json.loads((run_dir / "completion.json").read_text())
    assert completion["status"] == "COMPLETE" and completion["rows_written"] == 4 * len(plan.groups["validation_b"])
    first = json.loads((run_dir / "results.jsonl").read_text().splitlines()[0])
    assert set(first) == {"id", "baseline", "after_program@1", "after_program@2", "answer@0"}
