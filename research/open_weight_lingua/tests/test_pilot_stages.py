"""Milestone 2 stage orchestration with tiny random fixtures.

Fixture models, tokenizer and substituted descriptions establish runner
orchestration and accounting only; nothing here is a released-Qwen or
released-NLA measurement.
"""

import json
import statistics
import zipfile

import pytest
import torch
from safetensors.torch import save_file

from open_weight_lingua import runner
from open_weight_lingua.artifacts import RunDirectory, write_json
from open_weight_lingua.audit import FROZEN_THRESHOLDS, audit_run, summarize
from open_weight_lingua.controls import fit_pca, load_fit, save_fit
from open_weight_lingua.metrics import answer_tokens
from open_weight_lingua.nla_adapter import DescriptionRecord
from open_weight_lingua.splits import build_plan
from open_weight_lingua.text_edits import RULE_SHA256, RULE_VERSION

TINY_COUNTS = {
    "smoke": 2,
    "calibration": 3,
    "pilot": 4,
    "validation_a": 2,
    "validation_b": 2,
}
ELIGIBLE_TEXT = "Fixture text: x is now 5; y is currently 8."
ABSENT_TEXT = "Fixture text without any current-value statement."


def _inputs(groups):
    """Hand-built fixture tokenization, following tests/test_runner.py."""
    rows = []
    for index, row in enumerate(
        row for group in groups for row in group.variants()
    ):
        row.update(
            input_ids=[3 + index // 16, 3 + index % 16, 4],
            attention_mask=[1, 1, 1],
            position=2,
        )
        rows.append(row)
    return rows


def _stub_load_model(model_factory):
    def load_model(path, role, device):
        torch.manual_seed(73)
        return model_factory(layers=2 if role == "ar" else 3)

    return load_model


def _stub_verbalizer(texts):
    """Fixture descriptions per call order; None encodes an AV failure."""
    calls = {"count": 0}

    class StubVerbalizer:
        def __init__(self, model, tokenizer, metadata):
            self.model = model

        def verbalize(self, vector):
            assert vector.shape == (16,)
            text = texts[calls["count"] % len(texts)]
            calls["count"] += 1
            if text is None:
                return DescriptionRecord("fixture failure", None, [3, 2], "failed", 2)
            return DescriptionRecord(
                f"<explanation>{text}</explanation>", text, [3, 2], "ok", 2
            )

    return StubVerbalizer


def _lock_file(tmp_path):
    path = tmp_path / "model-lock.json"
    path.write_text("{}")
    return path


def test_stage_cli_defaults_and_validation_remains_absent():
    assert runner.parse_args([]).stage == "smoke"
    assert runner.parse_args(["--stage", "smoke"]).stage == "smoke"
    assert runner.parse_args(["--stage", "calibration"]).stage == "calibration"
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage", "validation"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage", "validation_a"])


def test_pilot_refuses_without_calibration_fit():
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage", "pilot"])
    args = runner.parse_args(["--stage", "pilot", "--calibration-fit", "fit"])
    assert args.calibration_fit.name == "fit"
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage", "calibration", "--calibration-fit", "fit"])


def test_group_counts_override_is_validated():
    assert runner._group_counts(json.dumps(TINY_COUNTS)) == TINY_COUNTS
    assert runner._group_counts(None) is None
    with pytest.raises(ValueError, match="group-counts-json"):
        runner._group_counts("{not json")
    with pytest.raises(ValueError, match="positive"):
        runner._group_counts(json.dumps({"smoke": 0}))


def test_load_calibration_inputs_validation(tmp_path):
    with pytest.raises((OSError, ValueError)):
        runner.load_calibration_inputs(tmp_path / "absent")
    fit = fit_pca(torch.randn(8, 16, generator=torch.Generator().manual_seed(5)))
    save_fit(fit, tmp_path / "fit")
    with pytest.raises(ValueError, match="median"):
        runner.load_calibration_inputs(tmp_path / "fit")
    (tmp_path / "fit" / "calibration_median_norm.json").write_text(
        json.dumps({"format": "open_weight_lingua.calibration_median_norm.v1",
                    "median_norm": -1.0, "plan_hash": "0" * 64})
    )
    with pytest.raises(ValueError, match="median"):
        runner.load_calibration_inputs(tmp_path / "fit")
    (tmp_path / "fit" / "calibration_median_norm.json").write_text(
        json.dumps({"format": "open_weight_lingua.calibration_median_norm.v1",
                    "median_norm": 3.25, "plan_hash": "0" * 64})
    )
    loaded, record = runner.load_calibration_inputs(tmp_path / "fit")
    assert loaded.identity == fit.identity and record["median_norm"] == 3.25
    # A direct path to the safetensors payload resolves to its directory.
    loaded_again, _ = runner.load_calibration_inputs(
        tmp_path / "fit" / "baseline_fit.safetensors"
    )
    assert loaded_again.identity == fit.identity


def test_calibration_stage_writes_loadable_fit_and_frozen_median(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    """Tiny random fixture extraction; not a real-model calibration."""
    plan = build_plan(TINY_COUNTS)
    inputs = _inputs(plan.groups["calibration"])
    monkeypatch.setattr(runner, "load_model", _stub_load_model(model_factory))
    run = RunDirectory(tmp_path / "calibration")
    rows, fit, median_record = runner.execute_calibration(
        run,
        {role: tmp_path for role in ("target", "av", "ar")},
        {"target": tokenizer},
        metadata_factory("av"),
        plan,
        inputs,
        "cpu",
        {},
        {},
    )
    assert fit.width == 16 and fit.fitted_rank > 0
    assert load_fit(run.path).identity == fit.identity
    assert median_record["median_norm"] == pytest.approx(
        statistics.median(row["original_norm"] for row in rows)
    )
    assert median_record["extractions"] == len(rows) == 12
    assert median_record["plan_hash"] == plan.plan_hash
    assert json.loads((run.path / "calibration_median_norm.json").read_text())[
        "median_norm"
    ] == pytest.approx(median_record["median_norm"])
    manifest = runner.build_manifest(
        {},
        _lock_file(tmp_path),
        inputs,
        None,
        {"layer": 1},
        {},
        stage="calibration",
        plan=plan,
        tokenization={
            "accepted": 3,
            "rejected": 0,
            "rejections": [],
            "prior_excluded_prompts": 8,
        },
    )
    assert manifest["stage"] == "numerical_calibration"
    assert manifest["plan_hash"] == plan.plan_hash
    assert manifest["split_counts"] == TINY_COUNTS
    assert manifest["groups"] == 3
    assert manifest["frozen_thresholds"] == FROZEN_THRESHOLDS
    write_json(run.path / "manifest.json", manifest)
    summary = summarize(manifest, rows)
    assert len(summary["successful_groups"]) == 3
    assert not summary["failed_groups"] and not summary["skipped_groups"]
    assert summary["calibration"]["median_norm"] == pytest.approx(
        median_record["median_norm"]
    )
    write_json(run.path / "summary.json", summary)
    write_json(
        run.path / "completion.json",
        {
            "status": "COMPLETE",
            "stage": "numerical_calibration",
            "baseline_fit_identity": fit.identity,
            "calibration_median_norm": median_record["median_norm"],
        },
    )
    run.reports_zip()
    inventory = json.loads((run.path / "inventory.json").read_text())
    assert "baseline_fit.safetensors" in {
        entry["file"] for entry in inventory["retained_locally_not_in_zip"]
    }
    assert "baseline_fit.json" in {
        entry["file"] for entry in inventory["included"]
    }
    result = audit_run(run.path)
    assert result["status"] == "PASS"
    assert result["baseline_fit_identity"] == fit.identity
    with zipfile.ZipFile(run.path / "reports.zip") as archive:
        assert not any(
            name.endswith(".safetensors") or name.startswith("raw/")
            for name in archive.namelist()
        )


def _run_pilot_fixture(tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory):
    plan = build_plan(TINY_COUNTS)
    monkeypatch.setattr(runner, "load_model", _stub_load_model(model_factory))
    calibration_run = RunDirectory(tmp_path / "calibration")
    _, fit, median_record = runner.execute_calibration(
        calibration_run,
        {role: tmp_path for role in ("target", "av", "ar")},
        {"target": tokenizer},
        metadata_factory("av"),
        plan,
        _inputs(plan.groups["calibration"]),
        "cpu",
        {},
        {},
    )
    groups = {group.group_id: group for group in plan.groups["pilot"]}
    pilot_inputs = _inputs(plan.groups["pilot"])
    for row in pilot_inputs:
        row["answer_token_ids_including_eos"] = answer_tokens(
            tokenizer, row["answer"]
        )
    # Row order per group: A-affected (receiver), A-other, B-affected, B-other.
    texts = []
    for index in range(len(pilot_inputs)):
        ordinal, within = divmod(index, 4)
        if within != 0:
            texts.append(ABSENT_TEXT)
        else:
            texts.append([ELIGIBLE_TEXT, ABSENT_TEXT, ELIGIBLE_TEXT, None][ordinal])
    monkeypatch.setattr(runner, "Verbalizer", _stub_verbalizer(texts))
    save_file({"weight": torch.eye(16)}, tmp_path / "value_head.safetensors")
    run = RunDirectory(tmp_path / "pilot")
    loaded_fit, sidecar = runner.load_calibration_inputs(calibration_run.path)
    rows = runner.execute_pilot(
        run,
        {role: tmp_path for role in ("target", "av", "ar")},
        {role: tokenizer for role in ("target", "av", "ar")},
        metadata_factory("av"),
        metadata_factory("ar"),
        pilot_inputs,
        groups,
        loaded_fit,
        sidecar["median_norm"],
        "cpu",
        {},
        {},
    )
    return plan, groups, pilot_inputs, run, loaded_fit, median_record, rows


def test_pilot_stage_conditions_edits_and_decision(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    """Full pilot orchestration on fixtures; never labeled real-model evidence."""
    plan, groups, pilot_inputs, run, fit, median_record, rows = _run_pilot_fixture(
        tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
    )
    by_id = {row["id"]: row for row in rows}
    base = ("P0", "P1", "P2", "P3", "P4", "P5", "donor", "calibration_median_norm")
    for row in rows:
        assert set(row["conditions"]) >= set(base)
    # P4 carries the no-more-than-budget accounting of its own description.
    receiver_ids = {
        group_id: f"{group_id}-A-{group.affected}"
        for group_id, group in groups.items()
    }
    own_texts = {0: ELIGIBLE_TEXT, 1: ABSENT_TEXT, 2: ELIGIBLE_TEXT}
    for ordinal, group_id in enumerate(sorted(groups)[:3]):
        for row in (r for r in rows if r["group_id"] == group_id):
            accounting = row["conditions"]["P4"].get("accounting")
            assert accounting is not None
            assert accounting["baseline_fit_identity"] == fit.identity
            text = ABSENT_TEXT if row["id"] != receiver_ids[group_id] else own_texts[ordinal]
            assert accounting["text_budget_bytes"] == len(text.encode("utf-8"))
            assert accounting["coefficient_bytes"] <= accounting["text_budget_bytes"]
    # Group 3's receiver failed AV: failed conditions, and no fabricated KL.
    failed_receiver = by_id[receiver_ids[sorted(groups)[3]]]
    assert failed_receiver["description"]["status"] == "failed"
    assert failed_receiver["conditions"]["P2"]["status"] == "failed"
    assert "next_token_kl" not in failed_receiver["conditions"]["P2"]
    assert failed_receiver["edit"]["status"] == "description_unavailable"
    # Eligible groups get edited reconstructions; the absent group gets a status.
    for ordinal in (0, 2):
        group_id = sorted(groups)[ordinal]
        for row in (r for r in rows if r["group_id"] == group_id and r["id"].split("-")[2] == "A"):
            assert row["conditions"]["edited"]["status"] == "ok"
        edit = by_id[receiver_ids[group_id]]["edit"]
        assert edit["status"] == "eligible"
        assert ("wrong_variable_edit" in by_id[receiver_ids[group_id]]["conditions"]) == (
            edit["wrong_variable"].get("edited_description") is not None
        )
    absent_receiver = by_id[receiver_ids[sorted(groups)[1]]]
    assert absent_receiver["edit"]["status"] == "absent"
    assert "edited" not in absent_receiver["conditions"]
    assert absent_receiver["edit"].get("edited_description") is None
    manifest = runner.build_manifest(
        {},
        _lock_file(tmp_path),
        pilot_inputs,
        None,
        {"layer": 1},
        {},
        stage="pilot",
        plan=plan,
        group_counts_source="dev override --group-counts-json",
        tokenization={
            "accepted": 4,
            "rejected": 0,
            "rejections": [],
            "prior_excluded_prompts": 20,
        },
        fit=fit,
        fit_sidecar_sha256="0" * 64,
        calibration_median=median_record["median_norm"],
    )
    assert manifest["stage"] == "pilot"
    assert manifest["plan_hash"] == plan.plan_hash
    assert manifest["baseline_fit_identity"] == fit.identity
    assert manifest["edit_rule"]["version"] == RULE_VERSION
    assert manifest["edit_rule"]["sha256"] == RULE_SHA256
    assert manifest["frozen_thresholds"]["edit_eligible_group_floor"] == 32
    assert manifest["frozen_thresholds"]["bootstrap_resamples"] == 3000
    assert manifest["group_counts_source"] == "dev override --group-counts-json"
    assert manifest["calibration_median_norm"] == median_record["median_norm"]
    write_json(run.path / "manifest.json", manifest)
    summary = summarize(manifest, rows)
    # Group 2's receiver loses P3 (its deranged source is group 3's failed
    # description); group 3's receiver loses P2. Failures propagate, never hidden.
    assert len(summary["successful_groups"]) == 2
    assert len(summary["failed_groups"]) == 2
    assert summary["metrics"]["P2"]["failed_or_missing"] == 1
    assert summary["metrics"]["P3"]["failed_or_missing"] == 1
    coverage = summary["edit_eligibility"]
    assert coverage["eligible_groups"] == 2
    assert coverage["exclusion_reasons"] == {
        "absent": 1,
        "ambiguous": 0,
        "description_unavailable": 1,
    }
    assert coverage["rule_sha256"] == RULE_SHA256
    assert summary["uneditable_groups"] == 2
    assert summary["metrics"]["P2"]["attempted_prompt_variants"] == 16
    assert summary["metrics"]["edited"]["attempted_prompt_variants"] == 4
    stats_record = summary["statistics"]
    assert stats_record["p2_accuracy_loss_points"]["groups"] == 4
    assert stats_record["p4_accuracy_loss_points"]["groups"] == 4
    assert stats_record["p2_minus_p3_logprob"]["groups"] == 2
    assert stats_record["p2_minus_p3_logprob"]["groups_excluded"] == 2
    assert stats_record["editing_effect"]["records_used"] == 2
    assert stats_record["pilot_block_estimates"]["block_sizes"] == {"pilot": 4}
    decision = summary["decision"]
    assert set(decision["criteria"]) == {
        "unmodified_accuracy",
        "intervention_sensitivity",
        "p2_accuracy_loss",
        "p2_minus_p3_logprob",
        "edit_eligible_groups",
    }
    assert (decision["recommendation"] == "keep") == all(
        criterion["met"] for criterion in decision["criteria"].values()
    )
    assert set(decision["unmet_criteria"]) == {
        name
        for name, criterion in decision["criteria"].items()
        if not criterion["met"]
    }
    write_json(run.path / "summary.json", summary)
    write_json(
        run.path / "completion.json",
        {
            "status": "COMPLETE_WITH_FAILURES",
            "stage": "pilot",
            "decision": decision,
            "statistics": summary["statistics"],
            "edit_eligibility": coverage,
        },
    )
    run.reports_zip()
    assert audit_run(run.path)["status"] == "PASS"


def test_pilot_cost_projection_scales_and_never_starts_validation():
    timings = {
        "target_load_and_identity": 40.0,
        "av_load_and_generate": 80.0,
        "ar_load_and_reconstruct": 60.0,
        "target_reload_and_behavior": 320.0,
        "compatibility": 1.0,
    }
    calls = {"target_identity": 40, "av": 80, "ar": 60, "target_behavior": 320}
    rows = [{"identity_seconds": 0.5} for _ in range(8)]
    projection = runner.pilot_cost_projection(timings, calls, rows, 4)
    assert projection["per_group_seconds"] == 125.0
    assert projection["projected_validation_seconds_512_groups"] == 125.0 * 512
    assert projection["projected_calibration_seconds_256_groups"] == 0.5 * 4 * 256
    assert projection["projected_validation_forward_calls"] == 500 / 4 * 512
    assert (
        projection["projected_total_seconds"]
        == projection["projected_validation_seconds_512_groups"]
        + projection["projected_calibration_seconds_256_groups"]
    )
    assert projection["within_budget"] is False  # honesty: exceeds 8 hours
    assert "estimate" in projection["basis"]  # reported, never auto-started
    empty = runner.pilot_cost_projection({}, {}, [], 0)
    assert empty["projected_total_seconds"] is None
    assert empty["within_budget"] is None


def test_pilot_summary_counts_failed_rows_and_never_fabricates_kl():
    """ITT: missing generations stay in every denominator; no zero-KL fill."""
    plan = build_plan(TINY_COUNTS)
    inputs = _inputs(plan.groups["pilot"])
    manifest = {
        "inputs": inputs,
        "stage": "pilot",
        "frozen_thresholds": FROZEN_THRESHOLDS,
    }
    rows = [
        {
            "id": inputs[0]["id"],
            "group_id": inputs[0]["group_id"],
            "conditions": {
                "P2": {"status": "failed", "reason": "required reconstruction unavailable"}
            },
        }
    ]
    summary = summarize(manifest, rows)
    assert len(summary["failed_groups"]) == 1
    assert len(summary["skipped_groups"]) == 3
    assert summary["metrics"]["P2"]["failed_or_missing"] == 16
    assert summary["metrics"]["P2"]["accuracy_all_variants"] == 0
    assert summary["metrics"]["P2"]["mean_valid_next_token_kl"] is None
    assert summary["metrics"]["edited"]["attempted_prompt_variants"] == 0
    assert summary["metrics"]["edited"]["accuracy_all_variants"] is None
    assert summary["edit_eligibility"]["eligible_groups"] == 0
    assert summary["edit_eligibility"]["exclusion_reasons"] == {
        "absent": 0,
        "ambiguous": 0,
        "description_unavailable": 4,
    }
    assert summary["statistics"]["editing_effect"]["status"] == "untested"
    assert summary["statistics"]["p2_minus_p3_logprob"]["status"] == "not_computable"
    decision = summary["decision"]
    assert decision["recommendation"] == "stop"
    assert "unmodified_accuracy" in decision["unmet_criteria"]
    assert "edit_eligible_groups" in decision["unmet_criteria"]


def test_check_target_bucket_fit_is_loud_and_pre_inference():
    fitting = [
        {
            "id": "row-fit",
            "input_ids": [0] * 120,
            "answer_token_ids_including_eos": [16, 16, 151645],
        }
    ]
    # 120 + max(8, 3) = 128: exactly at the boundary, accepted.
    runner.check_target_bucket_fit(fitting)
    # Calibration-style rows carry no answer suffix; the ceiling still applies.
    runner.check_target_bucket_fit([{"id": "row-cal", "input_ids": [0] * 120}])
    over = [
        {
            "id": "row-over",
            "input_ids": [0] * 121,
            "answer_token_ids_including_eos": [16, 151645],
        }
    ]
    with pytest.raises(ValueError, match="row-over.*exceeds the pinned target bucket 128"):
        runner.check_target_bucket_fit(over)


def test_manifest_records_target_bucket_policy(tmp_path):
    inputs = [{"id": "row", "input_ids": [3, 4], "group_id": "g"}]
    manifest = runner.build_manifest(
        {}, _lock_file(tmp_path), inputs, None, {"layer": 1}, {}
    )
    assert manifest["target_bucket"] == 128
    assert manifest["target_padding_policy"] == (
        "all target forwards right-padded to fixed length 128 for kernel-shape "
        "pinning on sm_121; masked pads contribute exactly zero (bitwise-proven); "
        "pinned regime established 2026-09-21 pre-pilot after the first pilot "
        "attempt's identity-gate failure; supersedes the unpadded "
        "smoke/calibration runs"
    )
    assert manifest["greedy_identity_gate"] == (
        "exact greedy equality required; divergence accepted only with measured "
        "within-bound prefix drift at the divergence step (same frozen "
        "SUFFIX_DRIFT_BOUND and 2026-09-21 justification); repaired pre-pilot "
        "after the first pilot attempt failed this gate on variant "
        "pilot-0000-A-x with no conditions executed; no pilot/validation "
        "outcomes inspected"
    )


def test_smoke_rows_record_greedy_identity_backstop(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    """Fixture orchestration only: pinned gates record exact, summaries count."""
    plan = build_plan(TINY_COUNTS)
    inputs = _inputs(plan.groups["smoke"])
    monkeypatch.setattr(runner, "load_model", _stub_load_model(model_factory))
    monkeypatch.setattr(
        runner, "Verbalizer", _stub_verbalizer(["Fixture only"] * len(inputs))
    )
    save_file({"weight": torch.eye(16)}, tmp_path / "value_head.safetensors")
    run = RunDirectory(tmp_path / "smoke")
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
    for row in rows:
        assert row["identity"]["greedy"]["status"] == "exact"
        assert row["p0_p1_greedy_gate"]["status"] == "exact"
        assert "target_bucket_padding" in row["payload_bytes"]
    summary = summarize({"inputs": inputs}, rows)
    assert summary["greedy_identity"] == {
        "identity_stage": {"exact": 8, "drift_diverged": 0},
        "behavior_stage": {"exact": 8, "drift_diverged": 0},
    }


def test_pilot_rows_record_greedy_backstop_counts(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    """Pilot-stage rows carry the same pinned backstop evidence as smoke."""
    plan, groups, pilot_inputs, run, fit, median_record, rows = _run_pilot_fixture(
        tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
    )
    for row in rows:
        assert row["identity"]["greedy"]["status"] == "exact"
        assert row["p0_p1_greedy_gate"]["status"] == "exact"
    manifest = {
        "inputs": pilot_inputs,
        "stage": "pilot",
        "frozen_thresholds": FROZEN_THRESHOLDS,
    }
    summary = summarize(manifest, rows)
    assert summary["greedy_identity"] == {
        "identity_stage": {"exact": 16, "drift_diverged": 0},
        "behavior_stage": {"exact": 16, "drift_diverged": 0},
    }
