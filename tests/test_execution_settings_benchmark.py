"""Regression tests for the fresh-process execution-settings benchmark."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rebenchmark_execution_modes.py"

spec = importlib.util.spec_from_file_location("execution_modes_script", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_deterministic_child_environment_is_explicit(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "stale")
    assert "CUBLAS_WORKSPACE_CONFIG" not in module.child_environment(False)
    assert module.child_environment(True)["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_summary_keeps_training_runs_as_replication_unit():
    records = []
    for seed, normal, deterministic in ((1, 10.0, 12.0), (2, 11.0, 13.0)):
        for enabled, median in ((False, normal), (True, deterministic)):
            records.append({
                "model": "shared",
                "batch_size": 1024,
                "deterministic_algorithms": enabled,
                "timing": {"median_ms": median},
                "estimate_mse": 0.05,
                "training_seed": seed,
                "estimate_sha256": f"{seed}-{enabled}",
            })
    result = module.summarize(records)
    rows = {row["deterministic_algorithms"]: row for row in result["rows"]}
    assert rows[False]["training_runs"] == 2
    assert rows[False]["median_of_run_medians_ms"] == 10.5
    assert rows[True]["median_of_run_medians_ms"] == 12.5
    ratio = result["deterministic_ratios"][0]
    assert ratio["deterministic_on_divided_by_off_median_time"] == 12.5 / 10.5


def test_parser_defaults_compare_both_batch_sizes():
    args = module.parser().parse_args([
        "--replication-dir", "/tmp/r",
        "--output-dir", "/tmp/o",
    ])
    assert args.batch_sizes == [1024, 4096]
    assert args.test_seed == 32000
    assert args.repeats == 20


def test_parent_refuses_missing_checkpoints(tmp_path):
    args = module.parser().parse_args([
        "--replication-dir", str(tmp_path / "empty"),
        "--output-dir", str(tmp_path / "out"),
    ])
    args.replication_dir.mkdir()
    try:
        module.run_parent(args)
    except ValueError as error:
        assert "no weights-*.pt" in str(error)
    else:
        raise AssertionError("missing checkpoints must be rejected")
