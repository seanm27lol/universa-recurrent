"""Evaluation adapter with fake neural calls: no checkpoint/model execution."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ti_tnn"))
from native import data as D, evaluate_initial_state_study as E, structured as S
from native.initial_state_study import POPULATIONS


def test_prompt_boundary_complete_denominators_and_strict_replay_retention(tmp_path, monkeypatch):
    data = tmp_path / "prepared"
    examples = [S.make_example("dev", (S.Instruction("copy", 0, src=1),), (i % 32, i // 32, 7, 8))
                for i in range(128)]
    for name in POPULATIONS:
        D._atomic_jsonl(data / "prompts" / f"{name}.jsonl", [e.inference_record() for e in examples])
        D._atomic_jsonl(data / "evaluation" / f"{name}.jsonl", [e.to_record() for e in examples])
    checkpoint = tmp_path / "checkpoint-fixture"
    checkpoint.write_text("not a model")
    calls = []
    monkeypatch.setattr(E.E, "load_native_model", lambda *a, **kw: (object(), {}))
    monkeypatch.setattr(E.D, "load_gpt2_tokenizer", lambda *a: object())

    def generate(model, tokens, prompt, **kwargs):
        assert isinstance(prompt, str) and prompt.endswith("Answer:\n")
        assert kwargs == {"max_new_tokens": 48, "eos_token_id": D.EOS_TOKEN_ID,
                          "use_cache": True, "prompt_policy": "error"}
        calls.append(prompt)
        return {"completion_text": "malformed", "stopped_on_eos": True,
                "truncated": False, "prompt_truncated": False}

    monkeypatch.setattr(E.E, "generate_prompt", generate)
    monkeypatch.setattr(E, "AnswerMaskedDataset", lambda *a, **kw: object())
    monkeypatch.setattr(E, "teacher_forced_answer_cross_entropy", lambda *a, **kw: {"complete": True})
    monkeypatch.setattr(E, "evaluate_state_path", lambda *a, **kw: {"complete": True})
    monkeypatch.setattr(E.reconstruct, "load_replay_spec", lambda *a: {})
    monkeypatch.setattr(E.reconstruct, "reconstruct_checkpoint", lambda *a, **kw:
                        {"agreement": True, "teacher_forced": {"strict_allclose_agrees": False}})
    report = E.evaluate({"path": str(checkpoint), "sha256": D.sha256_file(checkpoint)}, data, "tokenizer", tmp_path / "output")
    assert report["complete"] and len(calls) == 1152
    assert report["reconstruction"]["teacher_forced"]["strict_allclose_agrees"] is False
    for value in report["populations"].values():
        score = value["score"]
        assert score["complete_answers"]["total"] == score["complete_answers"]["malformed"] == 128
        assert score["registers"]["total"] == score["registers"]["malformed"] == 512
        assert sum(v["total"] for v in score["by_value_change"].values()) == 512
    with pytest.raises(FileExistsError):
        E.evaluate({}, data, "tokenizer", tmp_path / "output")
