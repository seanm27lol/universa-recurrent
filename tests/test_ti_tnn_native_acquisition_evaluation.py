"""Offline checks for acquisition evaluation plumbing, not learning evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import evaluate as E  # noqa: E402
from native import evaluate_acquisition as A  # noqa: E402
from native import evaluate_state_path as S  # noqa: E402


def test_acquisition_evaluate_returns_results_and_isolates_state_outputs(tmp_path, monkeypatch):
    root = tmp_path / "data"
    structured = root / "structured"
    structured.mkdir(parents=True)
    for split in ("train", "dev"):
        (structured / f"{split}.jsonl").write_text(json.dumps({
            "example_id": f"{split}-0", "split": split, "prompt": split,
            "target_text": "r0=0 r1=0 r2=0 r3=0",
        }) + "\n")
    model = object()
    tokenizer = object()
    descriptor = {"sha256": "checkpoint-sha", "step": 3, "config": {"mode": "typed"}}
    monkeypatch.setattr(A.E, "load_native_model", lambda *args, **kwargs: (model, descriptor))
    monkeypatch.setattr(A.E, "load_gpt2_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(A, "source_records", lambda path: {"source": "records"})
    generation_calls = []
    state_calls = []

    def generation(model_arg, tokenizer_arg, records, **kwargs):
        generation_calls.append((model_arg, tokenizer_arg, records, kwargs))
        split = records[0]["split"]
        return {
            "correct": int(split == "train"),
            "records": [{"example_id": records[0]["example_id"], "correct": split == "train"}],
        }

    def state_path(checkpoint, packed, output, **kwargs):
        state_calls.append((checkpoint, packed, output, kwargs))
        return {"split": kwargs["split"], "complete": True}

    monkeypatch.setattr(A.E, "_generation_metrics", generation)
    monkeypatch.setattr(A, "evaluate_state_path", state_path)
    monkeypatch.setattr(A, "summarize", lambda records, sources: {"exact_count": int(records[0]["correct"])})
    output = tmp_path / "arms" / "typed" / "evaluation.json"
    result = A.evaluate(
        "checkpoint.pt", root, "tokenizer.json", output,
        expected_sha256="checkpoint-sha", max_new_tokens=7,
    )

    assert json.loads(output.read_text()) == result
    assert result["splits"]["train"]["scope"] == "memorization fit"
    assert result["splits"]["dev"]["scope"] == "separate held-out diagnostic"
    assert result["splits"]["train"]["generation"]["correct"] == 1
    assert result["splits"]["dev"]["generation"]["correct"] == 0
    assert [call[2] for call in state_calls] == [
        output.parent / "train_state_path.json", output.parent / "dev_state_path.json",
    ]
    assert [call[1] for call in state_calls] == [root / "train", root / "dev"]
    assert [call[3]["split"] for call in state_calls] == ["train", "dev"]
    assert all(call[3]["tokenizer_path"] == "tokenizer.json" for call in state_calls)
    assert all(call[0] is model and call[1] is tokenizer for call in generation_calls)
    assert all(call[3]["strict_registers"] and call[3]["limit"] is None for call in generation_calls)
    assert all(call[3]["max_new_tokens"] == 7 for call in generation_calls)


def test_acquisition_main_delegates_to_callable(tmp_path, monkeypatch, capsys):
    result = {
        "checkpoint": {"sha256": "checkpoint-sha"},
        "splits": {
            "train": {"generation": {"correct": 1}},
            "dev": {"generation": {"correct": 0}},
        },
    }
    calls = []
    monkeypatch.setattr(A, "evaluate", lambda *args: calls.append(args) or result)
    monkeypatch.setattr(sys, "argv", [
        "evaluate_acquisition", "--checkpoint", "checkpoint.pt", "--expected-sha256", "checkpoint-sha",
        "--data", "data", "--tokenizer", "tokenizer.json", "--output", "arm/evaluation.json",
        "--max-new-tokens", "9",
    ])
    A.main()
    assert calls == [(
        "checkpoint.pt", "data", "tokenizer.json", "arm/evaluation.json", "checkpoint-sha", 9,
    )]
    assert json.loads(capsys.readouterr().out) == {
        "checkpoint_sha256": "checkpoint-sha", "train_exact": 1, "dev_exact": 0,
        "output": "arm/evaluation.json",
    }


class _ParameterModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(0, device=device))
        self.config = SimpleNamespace(context_length=8)

    def forward(self, input_ids):
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 1, device=input_ids.device))


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[1] if text == "prompt" else [2], offsets=[(0, len(text))])

    def decode(self, ids, skip_special_tokens=False):
        return ""


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_generate_prompt_allocates_inputs_on_parameter_device(device, monkeypatch):
    model = _ParameterModel(device)
    seen = {}

    def generate(model_arg, input_ids, *, positions, **kwargs):
        seen["input_ids"] = input_ids.device
        seen["positions"] = positions.device
        return torch.tensor([[1, 3]])

    monkeypatch.setattr(E, "greedy_generate", generate)
    result = E.generate_prompt(model, _Tokenizer(), "prompt", max_new_tokens=2, eos_token_id=3)
    assert seen == {"input_ids": torch.device(device), "positions": torch.device(device)}
    assert result["generated_ids"] == [3]
    assert result["stopped_on_eos"]


def test_answer_ce_selects_split_uses_model_device_and_reports_cap(tmp_path, monkeypatch):
    records = [{"example_id": str(index), "prompt": "prompt", "target_text": "target"} for index in range(129)]
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    (tmp_path / "dev.jsonl").write_text("not read for a train-only evaluation\n")
    model = _ParameterModel("meta")
    devices = []

    def cross_entropy(logits, targets, reduction):
        devices.append((logits.device, targets.device))
        assert reduction == "sum"
        return torch.tensor(float(targets.numel()))

    monkeypatch.setattr(S.torch.nn.functional, "cross_entropy", cross_entropy)
    result = S._answer_cross_entropy(model, _Tokenizer(), tmp_path, split="train")
    assert set(result) == {"train"}
    assert result["train"] == {
        "sum": 256.0, "count": 256, "mean": 1.0,
        "examples": 129, "evaluated": 128, "complete": False, "status": "partial",
    }
    assert len(devices) == 128
    assert all(device == torch.device("meta") for pair in devices for device in pair)


class _StateDataset:
    def __len__(self):
        return 2

    def batch(self, indices):
        rows = len(indices)
        return {
            "input_ids": torch.zeros(rows, 1, dtype=torch.long),
            "positions": torch.zeros(rows, 1, dtype=torch.long),
            "doc_ids": torch.zeros(rows, 1, dtype=torch.long),
            "key_padding_mask": torch.zeros(rows, 1, dtype=torch.bool),
            "state_targets": torch.zeros(rows, 1, 4, 4, dtype=torch.long),
            "state_mask": torch.ones(rows, 1, 4, 4, dtype=torch.bool),
        }


class _StateModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, *, input_ids, return_aux, **kwargs):
        self.calls += 1
        return SimpleNamespace(state_logits=torch.zeros(*input_ids.shape, 4, 4, 2))


def test_state_path_reports_row_coverage_and_rejects_a_partial_cap(tmp_path, monkeypatch):
    model = _StateModel()
    monkeypatch.setattr(S, "load_native_model", lambda *args, **kwargs: (model, {"sha256": "checkpoint-sha"}))
    monkeypatch.setattr(S, "PackedDataset", lambda path: _StateDataset())
    output = tmp_path / "state.json"
    result = S.evaluate_state_path("checkpoint.pt", "packed", output, batch_size=1, max_batches=2)
    assert result["packed_rows"] == result["evaluated_rows"] == 2
    assert result["complete"] and result["batches"] == 2
    assert result["metrics"]["pass_0"] == {"correct": 8, "count": 8, "accuracy": 1.0}
    assert json.loads(output.read_text()) == result

    capped_output = tmp_path / "capped.json"
    with pytest.raises(ValueError, match="1 of 2 packed rows"):
        S.evaluate_state_path("checkpoint.pt", "packed", capped_output, batch_size=1, max_batches=1)
    assert model.calls == 2
    assert not capped_output.exists()
