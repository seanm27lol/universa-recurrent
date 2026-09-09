from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")

from universa_recurrent.cli import main


def test_neural_v2_cli_end_to_end(tmp_path: Path, capsys):
    checkpoint = tmp_path / "v2.pt"
    record = tmp_path / "v2.json"
    evaluation = tmp_path / "v2-eval.json"
    benchmark = tmp_path / "v2-bench.json"

    assert main([
        "neural-v2-train",
        "--device", "cpu",
        "--seed", "801",
        "--train-size", "48",
        "--calibration-size", "24",
        "--epochs", "1",
        "--batch-size", "16",
        "--hidden-dim", "12",
        "--candidate-embedding-dim", "4",
        "--steps", "2",
        "--output", str(checkpoint),
    ]) == 0

    assert main([
        "neural-v2-demo",
        "--device", "cpu",
        "--checkpoint", str(checkpoint),
        "--output", str(record),
    ]) == 0
    assert record.exists()

    assert main([
        "neural-v2-verify",
        str(record),
        "--checkpoint", str(checkpoint),
    ]) == 0

    assert main([
        "neural-v2-eval",
        "--device", "cpu",
        "--checkpoint", str(checkpoint),
        "--seed", "8801",
        "--n", "24",
        "--batch-size", "12",
        "--output", str(evaluation),
    ]) == 0
    assert evaluation.exists()

    assert main([
        "neural-v2-benchmark",
        "--device", "cpu",
        "--checkpoint", str(checkpoint),
        "--seed", "8802",
        "--n", "24",
        "--batch-size", "12",
        "--warmup", "0",
        "--repeats", "1",
        "--output", str(benchmark),
    ]) == 0
    assert benchmark.exists()
    benchmark_payload = __import__("json").loads(benchmark.read_text())
    methods = {row["method"] for row in benchmark_payload["rows"]}
    assert "ambient_recurrent_parameter_matched" in methods
    assert "dedicated_fixed_depth_1" in methods
    output = capsys.readouterr().out
    assert "multiple live hypotheses" in output
    assert "Independent record check: PASS" in output
