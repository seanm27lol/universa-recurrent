from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from universa_recurrent.cli import main


def test_neural_cli_end_to_end(tmp_path: Path, capsys):
    checkpoint = tmp_path / "tiny.pt"
    record = tmp_path / "trace.json"
    evaluation = tmp_path / "eval.json"
    benchmark = tmp_path / "bench.json"

    assert main([
        "neural-train",
        "--device", "cpu",
        "--seed", "111",
        "--train-size", "64",
        "--val-size", "32",
        "--epochs", "1",
        "--batch-size", "32",
        "--hidden-dim", "16",
        "--steps", "2",
        "--output", str(checkpoint),
    ]) == 0
    assert checkpoint.exists()

    assert main([
        "neural-demo",
        "--device", "cpu",
        "--checkpoint", str(checkpoint),
        "--max-steps", "2",
        "--fixed",
        "--output", str(record),
    ]) == 0
    assert record.exists()

    assert main([
        "neural-verify",
        str(record),
        "--checkpoint", str(checkpoint),
    ]) == 0

    assert main([
        "neural-eval",
        "--device", "cpu",
        "--checkpoint", str(checkpoint),
        "--n", "32",
        "--batch-size", "16",
        "--output", str(evaluation),
    ]) == 0
    assert evaluation.exists()

    assert main([
        "neural-benchmark",
        "--device", "cpu",
        "--checkpoint", str(checkpoint),
        "--n", "32",
        "--batch-size", "32",
        "--warmup", "0",
        "--repeats", "1",
        "--output", str(benchmark),
    ]) == 0
    assert benchmark.exists()
    output = capsys.readouterr().out
    assert "Independent record check: PASS" in output


def test_neural_cli_refuses_checkpoint_overwrite(tmp_path: Path):
    checkpoint = tmp_path / "exists.pt"
    checkpoint.write_bytes(b"already here")
    assert main([
        "neural-train",
        "--device", "cpu",
        "--train-size", "8",
        "--val-size", "8",
        "--epochs", "1",
        "--batch-size", "8",
        "--hidden-dim", "8",
        "--steps", "1",
        "--output", str(checkpoint),
    ]) == 2
