"""Offline checkpoint reconstruction checks on tiny native models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import reconstruct as R  # noqa: E402
from native.checkpoint import CheckpointValidationError, save_checkpoint  # noqa: E402
from native.config import NativeConfig  # noqa: E402
from native.evaluate import load_native_model  # noqa: E402
from native.model import NativeModel  # noqa: E402
from native.sampler import identity_hash  # noqa: E402


def tiny_checkpoint(tmp_path, dtype=torch.float32):
    torch.manual_seed(123)
    config = NativeConfig(
        vocab_size=32,
        context_length=32,
        d_model=16,
        n_heads=4,
        d_slot=8,
        d_ff=32,
        mode="typed",
        n_state_classes=5,
    )
    model = NativeModel(config).to(device="cpu", dtype=dtype).eval()
    descriptor = save_checkpoint(
        tmp_path / f"tiny-{dtype}.pt",
        model,
        step=7,
        config=config.to_dict(),
        code_identity="native-reconstruction-test-v1",
    )
    return model, descriptor


def replay_spec(tmp_path):
    values = {
        "format": R.REPLAY_FORMAT,
        "teacher_forced": {
            "input_ids": [[1, 2, 3, 4]],
            "positions": [[0, 1, 0, 1]],
            "doc_ids": [[5, 5, 9, 9]],
            "key_padding_mask": [[False, False, False, False]],
            "labels": [[-100, 2, -100, 4]],
            "source_ids": [[0, 0, 2, 2]],
        },
        "generation_prompts": [
            {"example_id": "fixed-a", "input_ids": [1, 2], "positions": [0, 1]},
            {"example_id": "fixed-b", "input_ids": [3, 4, 5], "positions": [0, 1, 2]},
        ],
        "max_new_tokens": 3,
        "eos_token_id": 7,
    }
    path = tmp_path / "replay.json"
    path.write_text(json.dumps(values, sort_keys=True) + "\n")
    return path


def shape_replay(path, shape):
    teacher = {}
    for name in (*R.MODEL_INPUT_ARRAYS, "labels", "source_ids"):
        dtype = torch.bool if name == "key_padding_mask" else torch.long
        teacher[name] = torch.zeros(shape, dtype=dtype).tolist()
    payload = {
        "format": R.REPLAY_FORMAT,
        "teacher_forced": teacher,
        "generation_prompts": [],
        "max_new_tokens": 0,
        "eos_token_id": 7,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def test_bare_256_teacher_replay_is_rejected_before_model_execution(tmp_path, monkeypatch):
    path = shape_replay(tmp_path / "bare-256.json", (256,))
    model_called = False

    def forbidden_model_load(*args, **kwargs):
        nonlocal model_called
        model_called = True
        raise AssertionError("model execution must not start for a malformed replay")

    monkeypatch.setattr(R, "load_native_model", forbidden_model_load)
    with pytest.raises(R.ReconstructionError) as caught:
        spec = R.load_replay_spec(path)
        R.reconstruct_checkpoint(tmp_path / "unused-checkpoint.pt", spec)
    assert not model_called
    message = str(caught.value)
    assert "teacher_forced.input_ids" in message
    assert "rank-2 shape [B,T]" in message
    assert "got shape [256]" in message
    assert "use [1, 256]" in message
    assert "automatic reshaping is disabled" in message


def test_valid_batched_256_teacher_replay_loads_without_reshaping(tmp_path):
    spec = R.load_replay_spec(shape_replay(tmp_path / "batched-256.json", (1, 256)))
    for name in (*R.MODEL_INPUT_ARRAYS, "labels", "source_ids"):
        assert spec["teacher_forced"][name].shape == (1, 256)
    assert spec["teacher_forced"]["key_padding_mask"].dtype == torch.bool
    assert spec["teacher_forced"]["input_ids"].dtype == torch.long


@pytest.mark.parametrize("name", [*R.MODEL_INPUT_ARRAYS, "labels", "source_ids"])
def test_every_teacher_array_rejects_a_bare_256_vector(tmp_path, name):
    path = shape_replay(tmp_path / f"bad-{name}.json", (1, 256))
    payload = json.loads(path.read_text())
    payload["teacher_forced"][name] = payload["teacher_forced"][name][0]
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(R.ReconstructionError, match=rf"teacher_forced\.{name}.*rank-2 shape \[B,T\]"):
        R.load_replay_spec(path)


def test_teacher_arrays_must_share_batch_time_dimensions_and_have_two_tokens(tmp_path):
    path = shape_replay(tmp_path / "mismatched-labels.json", (1, 256))
    payload = json.loads(path.read_text())
    payload["teacher_forced"]["labels"] = torch.zeros(2, 256, dtype=torch.long).tolist()
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(R.ReconstructionError, match=r"labels.*expected \[1, 256\].*identical batch and time"):
        R.load_replay_spec(path)

    path = shape_replay(tmp_path / "single-token.json", (1, 1))
    with pytest.raises(R.ReconstructionError, match=r"input_ids.*T >= 2.*shape \[1, 1\]"):
        R.load_replay_spec(path)


def test_empty_teacher_batch_and_missing_source_ids_are_rejected(tmp_path):
    values = {
        name: torch.zeros((0, 256), dtype=torch.bool if name == "key_padding_mask" else torch.long)
        for name in (*R.MODEL_INPUT_ARRAYS, "labels", "source_ids")
    }
    with pytest.raises(R.ReconstructionError, match=r"input_ids.*nonempty batch B.*shape \[0, 256\]"):
        R._normalize_teacher_sample(values)

    path = shape_replay(tmp_path / "missing-source-ids.json", (1, 256))
    payload = json.loads(path.read_text())
    del payload["teacher_forced"]["source_ids"]
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(R.ReconstructionError, match=r"missing required \[B,T\] arrays \['source_ids'\]"):
        R.load_replay_spec(path)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_reconstruction_restores_actual_dtype_and_replays_full_cached_and_reference(
    tmp_path, dtype
):
    original, descriptor = tiny_checkpoint(tmp_path, dtype)
    spec = R.load_replay_spec(replay_spec(tmp_path))
    report = R.reconstruct_checkpoint(
        descriptor["path"], spec, expected_sha256=descriptor["sha256"]
    )
    assert report["checkpoint"]["sha256_verified"]
    assert report["checkpoint"]["sha256"] == descriptor["sha256"]
    assert report["configuration"]["saved"] == original.config.to_dict()
    assert report["configuration"]["restored_model"] == original.config.to_dict()
    assert report["configuration"]["exact"]
    assert report["weights"]["dtype"] == str(dtype)
    assert report["weights"]["device"] == "cpu"
    assert report["weights"]["sha256"] == identity_hash(original.state_dict())
    assert report["teacher_forced"]["full_outputs"]["logits"]["dtype"] == str(dtype)
    assert report["teacher_forced"]["full_outputs"]["logits"]["device"] == "cpu"
    assert report["teacher_forced"]["loss_tokens"] == 2
    assert report["teacher_forced"]["agrees"]
    assert report["generation"]["agrees"]
    assert report["agreement"]
    for name, comparison in report["teacher_forced"]["comparisons"].items():
        if name == "route_logits" or comparison is not None:
            assert comparison["agrees"], name
            assert "first_divergence" in comparison
            assert "first_tolerance_violation" in comparison
    assert len(report["generation"]["records"]) == 2
    assert all(record["first_divergence"] is None for record in report["generation"]["records"])

    restored, _ = load_native_model(descriptor["path"])
    for name, value in restored.state_dict().items():
        assert value.dtype == dtype
        assert torch.equal(value, original.state_dict()[name])

    output = tmp_path / "reconstruction.json"
    digest = R.write_reconstruction_report(output, report)
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    second = R.reconstruct_checkpoint(
        descriptor["path"],
        spec,
        expected_sha256=descriptor["sha256"],
        expected_report=json.loads(output.read_text()),
    )
    assert second["reference_replay"]["agrees"]
    assert second["reference_replay"]["first_divergence"] is None
    assert second["agreement"]


def test_reconstruction_refuses_wrong_checkpoint_sha_before_replay(tmp_path):
    _, descriptor = tiny_checkpoint(tmp_path)
    spec = R.load_replay_spec(replay_spec(tmp_path))
    with pytest.raises(CheckpointValidationError, match="checkpoint hash differs"):
        R.reconstruct_checkpoint(descriptor["path"], spec, expected_sha256="0" * 64)


def test_tensor_comparison_reports_first_exact_and_tolerance_divergence():
    actual = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert R.compare_tensors(actual, expected, atol=0, rtol=0)["exact"]
    actual[1, 0] = 3.01
    actual[0, 1] = 2.001
    comparison = R.compare_tensors(actual, expected, atol=0.005, rtol=0)
    assert not comparison["exact"] and not comparison["agrees"]
    assert comparison["first_divergence"]["index"] == [0, 1]
    assert comparison["first_divergence"]["absolute_difference"] == pytest.approx(0.001, abs=1e-6)
    assert comparison["first_tolerance_violation"]["index"] == [1, 0]
    assert comparison["maximum_absolute_difference"] == pytest.approx(0.01, abs=1e-6)


def test_reconstruction_cli_writes_hashes_and_replay_summary(tmp_path, capsys):
    _, descriptor = tiny_checkpoint(tmp_path)
    spec_path = replay_spec(tmp_path)
    output = tmp_path / "cli-report.json"
    status = R.main([
        "--checkpoint", descriptor["path"],
        "--expected-sha256", descriptor["sha256"],
        "--spec", str(spec_path),
        "--output", str(output),
    ])
    assert status == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["teacher_forced_agrees"] and summary["generation_agrees"]
    assert summary["checkpoint_sha256"] == descriptor["sha256"]
    assert summary["report_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    report = json.loads(output.read_text())
    assert report["teacher_forced"]["full_outputs"]["logits"]["sha256"]
    assert report["generation"]["records"][0]["full_sequence_sha256"]
