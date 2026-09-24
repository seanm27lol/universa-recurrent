import json
import zipfile
import pytest
import torch
from safetensors.torch import save_file
from open_weight_lingua import runner
from open_weight_lingua.artifacts import RunDirectory, write_json, load_numeric
from open_weight_lingua.audit import audit_run, summarize
from open_weight_lingua.nla_adapter import DescriptionRecord
from open_weight_lingua.tasks import generate_groups


def test_all_eight_groups_fixture_orchestration_and_independent_replay(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    """Random tiny Qwen models + stub descriptions, explicitly not Qwen results."""
    groups, _ = generate_groups("smoke", 8)
    inputs = [row for group in groups for row in group.variants()]
    for index, row in enumerate(inputs):
        row.update(
            input_ids=[3 + index // 16, 3 + index % 16, 4],
            attention_mask=[1, 1, 1],
            position=2,
        )

    # Repeated loads are the same frozen random checkpoint, not new random weights.
    def load_model(path, role, device):
        torch.manual_seed(73)
        return model_factory(layers=2 if role == "ar" else 3)

    monkeypatch.setattr(runner, "load_model", load_model)

    class StubVerbalizer:
        def __init__(self, model, tokenizer, metadata):
            self.model = model

        def verbalize(self, vector):
            assert vector.shape == (16,)
            return DescriptionRecord(
                "<explanation>Fixture only</explanation>",
                "Fixture only",
                [3, 2],
                "ok",
                2,
            )

    monkeypatch.setattr(runner, "Verbalizer", StubVerbalizer)
    save_file({"weight": torch.eye(16)}, tmp_path / "value_head.safetensors")
    run = RunDirectory(tmp_path / "run")
    manifest = {"inputs": inputs}
    write_json(run.path / "manifest.json", manifest)
    rows = runner.execute_smoke(
        run,
        {role: tmp_path for role in ("target", "av", "ar")},
        {role: tokenizer for role in ("target", "av", "ar")},
        metadata_factory("av"),
        metadata_factory("ar"),
        inputs,
        "cpu",
        {},
        {},
    )
    summary = summarize(manifest, rows)
    assert len(summary["successful_groups"]) == 8
    assert not summary["failed_groups"]
    assert summary["metrics"]["P2"]["attempted_prompt_variants"] == 32
    original = load_numeric(run.path / "raw" / f"{rows[0]['id']}-original.safetensors")
    assert original["retained_norm_float32"].dtype == torch.float32
    assert original["retained_norm_float32"].numel() == 1
    assert original["site_int64"].tolist() == [1, 0, 2]
    write_json(run.path / "summary.json", summary)
    run.reports_zip()
    assert audit_run(run.path)["status"] == "PASS"
    with zipfile.ZipFile(run.path / "reports.zip") as archive:
        assert not any(
            name.endswith(".safetensors") or name.startswith("raw/")
            for name in archive.namelist()
        )
    raw = run.path / "raw" / f"{rows[0]['id']}-behavior.safetensors"
    raw.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        audit_run(run.path)


def test_failed_and_missing_variants_stay_in_denominator():
    groups, _ = generate_groups("smoke", 8)
    inputs = [row for group in groups for row in group.variants()]
    rows = [{"id": inputs[0]["id"], "conditions": {"P2": {"status": "failed"}}}]
    summary = summarize({"inputs": inputs}, rows)
    assert len(summary["failed_groups"]) == 1 and len(summary["skipped_groups"]) == 7
    assert summary["metrics"]["P2"]["accuracy_all_variants"] == 0
    assert summary["metrics"]["P2"]["failed_or_missing"] == 32
    assert summary["metrics"]["P2"]["mean_valid_next_token_kl"] is None


def test_missing_models_produce_actionable_report(tmp_path):
    run = tmp_path / "run"
    status = runner.main(
        ["--cache", str(tmp_path / "absent"), "--run-dir", str(run), "--device", "cpu"]
    )
    assert status == 1
    completion = json.loads((run / "completion.json").read_text())
    assert completion["real_model_checks"] == "NOT RUN"
    assert "--fetch-models" in completion["error"]
    assert (run / "reports.zip").is_file()
    with pytest.raises(FileExistsError):
        runner.main(["--run-dir", str(run)])


def test_scientific_stages_unavailable():
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage", "pilot"])
