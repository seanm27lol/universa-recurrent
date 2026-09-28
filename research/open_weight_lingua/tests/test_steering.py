"""Steering-assay tests with tiny random fixtures.

Fixture models, tokenizers, hand-built pilot-run directories and oracle texts
establish steering math, accounting, ITT encoding, manifest locking and audit
replay only; nothing here is a released-Qwen or released-NLA measurement.
"""

import hashlib
import json
import zipfile

import pytest
import torch
from safetensors.torch import save_file

from open_weight_lingua import runner, steering
from open_weight_lingua.artifacts import (
    RunDirectory,
    load_numeric,
    save_numeric,
    write_json,
)
from open_weight_lingua.audit import audit_run, summarize
from open_weight_lingua.splits import build_plan
from open_weight_lingua.steering import (
    ALPHA_GRID,
    EXPECTED_ROW_CONDITIONS,
    STEERING_ARMS,
    STEERING_CONDITIONS,
    STEERING_FROZEN,
    TEMPLATES,
)
from open_weight_lingua.target import Site, Target

TINY_COUNTS = {
    "smoke": 2,
    "calibration": 3,
    "pilot": 4,
    "validation_a": 2,
    "validation_b": 2,
}


class SteeringTokenizer:
    """TinyTokenizer superset: text-distinct AR ids so deltas are non-degenerate.

    The shared fixture tokenizer maps every AR prompt to the same ids, which
    would make every AR-based delta exactly zero; assigning each distinct AR
    prompt a distinct in-vocabulary id pair keeps the real AR computation path
    while letting description differences produce direction differences.
    """

    eos_token_id = 2
    unk_token_id = None
    bos_token_id = None

    def __init__(self):
        self._ar_ids = {}

    def encode(self, text, add_special_tokens=False):
        if text == "㈎":
            return [7]
        if text.isdigit():
            return [10 + int(digit) for digit in text]
        if text.endswith("</text> <summary>"):
            if text not in self._ar_ids:
                index = len(self._ar_ids)
                self._ar_ids[text] = [23 + index % 9, 23 + (index // 9) % 9, 8, 9]
            return self._ar_ids[text]
        return [20, 21, 22]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(str(i - 10) if 10 <= i < 20 else f"<{i}>" for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return [3, 6, 7, 8, 4]


@pytest.fixture
def tokenizer():
    """File-local override: the steering fixture tokenizer (shadows conftest)."""
    return SteeringTokenizer()


def _tokenized_rows(plan):
    """Hand-built fixture tokenization, following tests/test_pilot_stages.py."""
    rows = []
    for index, row in enumerate(
        row for group in plan.groups["pilot"] for row in group.variants()
    ):
        row.update(
            input_ids=[3 + index // 16, 3 + index % 16, 4],
            attention_mask=[1, 1, 1],
            position=2,
        )
        rows.append(row)
    return rows


def _group_variables(group):
    affected = group.affected
    other = next(variable for variable in ("x", "y") if variable != affected)
    return affected, other


def _fake_pilot_run(tmp_path, plan, rows, executed, tokenizer, model_factory, *,
                    broken_directions=()):
    """A minimal pinned-pilot-run fixture: directions, originals, P0 records.

    Originals and P0 generations are computed with the same seeded fixture
    target the steering stage reloads, so the cross-run checks exercise the
    bitwise/agreement happy path.
    """
    pilot_dir = tmp_path / "pilot-run"
    (pilot_dir / "raw").mkdir(parents=True)
    torch.manual_seed(73)
    target = Target(model_factory(layers=3), tokenizer)
    records = {}
    direction_rng = torch.Generator().manual_seed(99)
    for row in rows:
        status = "failed" if row["id"] in broken_directions else "ok"
        records[row["id"]] = {
            "id": row["id"],
            "group_id": row["group_id"],
            "reconstruction": {"status": status},
        }
        if status == "ok":
            direction = torch.randn(16, generator=direction_rng) + 1.0
            save_numeric(
                pilot_dir / "raw" / f"{row['id']}-direction.safetensors",
                {"ar_direction": direction},
            )
    for row in rows:
        if row["side"] != "A" and not row["affected"]:
            continue
        ids = torch.tensor([row["input_ids"]])
        mask = torch.tensor([row["attention_mask"]])
        site = Site(1, row["position"])
        captured = target.capture(ids, mask, site)
        save_numeric(
            pilot_dir / "raw" / f"{row['id']}-original.safetensors",
            {"original": captured.vector},
        )
    for row in executed:
        ids = torch.tensor([row["input_ids"]])
        mask = torch.tensor([row["attention_mask"]])
        site = Site(1, row["position"])
        records[row["id"]]["conditions"] = {
            "P0": {"generation": target.greedy(ids, mask, site)}
        }
    write_json(pilot_dir / "results.json", list(records.values()))
    write_json(
        pilot_dir / "manifest.json",
        {"stage": "pilot", "plan_hash": plan.plan_hash},
    )
    write_json(pilot_dir / "completion.json", {"status": "COMPLETE"})
    return pilot_dir


def _pilot_row_lists(plan, executed):
    direction_rows, original_rows = [], []
    for group in plan.groups["pilot"]:
        affected, other = _group_variables(group)
        direction_rows += [
            f"{group.group_id}-{side}-{variable}"
            for side in ("A", "B")
            for variable in (affected, other)
        ]
        original_rows += [
            f"{group.group_id}-A-{affected}",
            f"{group.group_id}-A-{other}",
            f"{group.group_id}-B-{affected}",
        ]
    return direction_rows, original_rows, [row["id"] for row in executed]


def _load_fixture_pilot(pilot_dir, plan, executed):
    direction_rows, original_rows, p0_rows = _pilot_row_lists(plan, executed)
    return steering.load_pilot_run(
        pilot_dir,
        plan_hash=plan.plan_hash,
        width=16,
        direction_rows=direction_rows,
        original_rows=original_rows,
        p0_rows=p0_rows,
    )


def _stub_load_model(model_factory):
    def load_model(path, role, device):
        torch.manual_seed(73)
        return model_factory(layers=2 if role == "ar" else 3)

    return load_model


def _lock_file(tmp_path):
    path = tmp_path / "model-lock.json"
    path.write_text("{}")
    return path


def test_render_template_determinism_and_hashes():
    for style, text in TEMPLATES.items():
        assert steering.template_sha256(text) == hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()
        assert steering.render_template(style, "x", 5) == steering.render_template(
            style, "x", 5
        )
    assert TEMPLATES["terse"] != TEMPLATES["structured"]
    terse = steering.render_template("terse", "y", 8)
    assert terse == "The current value of y is 8."
    structured = steering.render_template("structured", "x", 5)
    assert structured.startswith("Structured")
    assert structured.count("\n\n") == 2  # the observed AV three-paragraph style
    assert "the value of x is 5" in structured
    assert structured.count('"5"') == 2
    assert structured != steering.render_template("structured", "x", 6)
    assert structured != steering.render_template("structured", "y", 5)
    with pytest.raises(ValueError, match="style"):
        steering.render_template("verbose", "x", 5)
    with pytest.raises(ValueError, match="variable"):
        steering.render_template("terse", "z", 5)
    with pytest.raises(ValueError, match="0..19"):
        steering.render_template("terse", "x", 20)
    with pytest.raises(ValueError, match="0..19"):
        steering.render_template("terse", "x", -1)


def test_description_delta_math():
    steered = torch.tensor([1.0, 2.0, 3.0])
    original = torch.tensor([0.5, 2.0, 1.0])
    delta = steering.description_delta(steered, original)
    assert torch.allclose(delta, torch.tensor([0.5, 0.0, 2.0]))
    with pytest.raises(ValueError, match="same-width"):
        steering.description_delta(steered, torch.ones(2))
    with pytest.raises(ValueError, match="same-width"):
        steering.description_delta(steered, torch.ones(1, 3))


def test_compose_patch_is_norm_scaled_additive_steering():
    vector = torch.tensor([3.0, 4.0])  # norm 5
    delta = torch.tensor([2.0, 0.0])  # unit [1, 0]
    patched = steering.compose_patch(vector, 5.0, delta, 2.0, dtype=torch.float32)
    assert torch.allclose(patched, torch.tensor([13.0, 4.0]))
    flipped = steering.compose_patch(vector, 5.0, delta, -1.0, dtype=torch.float32)
    assert torch.allclose(flipped, torch.tensor([-2.0, 4.0]))
    noop = steering.compose_patch(vector, 5.0, delta, 0.0, dtype=torch.float32)
    assert torch.equal(noop, vector)
    native = steering.compose_patch(vector, 5.0, delta, 1.0, dtype=torch.bfloat16)
    assert native.dtype == torch.bfloat16
    with pytest.raises(ValueError, match="near-zero"):
        steering.compose_patch(vector, 5.0, torch.zeros(2), 1.0, dtype=torch.float32)
    with pytest.raises(ValueError, match="norm"):
        steering.compose_patch(vector, 0.0, delta, 1.0, dtype=torch.float32)
    with pytest.raises(ValueError, match="width"):
        steering.compose_patch(vector, 5.0, torch.ones(3), 1.0, dtype=torch.float32)
    with pytest.raises(ValueError, match="alpha"):
        steering.compose_patch(
            vector, 5.0, delta, float("nan"), dtype=torch.float32
        )


def test_wrong_variable_candidate_rule():
    assert steering.wrong_variable_candidate(8, 1) == 9
    assert steering.wrong_variable_candidate(8, -1) == 7
    assert steering.wrong_variable_candidate(19, 1) == 18
    assert steering.wrong_variable_candidate(0, -1) == 1
    assert steering.wrong_variable_candidate(5, 0) is None


def test_frozen_grid_and_control_wiring():
    assert ALPHA_GRID == (-1.0, 0.5, 1.0, 2.0)
    assert 0.0 not in ALPHA_GRID  # alpha=0 is the P0 reference row
    assert STEERING_FROZEN["evaluation_alpha"] == 1.0
    assert STEERING_FROZEN["bootstrap_resamples"] == 3000
    assert STEERING_FROZEN["bootstrap_seed"] != 203100  # distinct from the pilot
    assert steering.condition_key("oracle_terse", 1.0) == "oracle_terse@1"
    assert steering.condition_key("av_diff", -1.0) == "av_diff@-1"
    assert steering.condition_key("av_diff_control", 0.5) == "av_diff_control@0.5"
    steered_keys = [key for key in EXPECTED_ROW_CONDITIONS if "@" in key]
    assert len(steered_keys) == len(STEERING_CONDITIONS) * len(ALPHA_GRID) == 24
    assert EXPECTED_ROW_CONDITIONS[:2] == ("P0", "P1")
    for arm, spec in STEERING_ARMS.items():
        assert spec["condition"] in STEERING_CONDITIONS
        assert spec["control"] in STEERING_CONDITIONS
        assert spec["condition"] != spec["control"]
    # Each arm's control is the matched one: same-value AV control for arm 1,
    # same-style wrong-variable oracle controls for arm 2.
    assert STEERING_ARMS["av_difference"]["control"] == "av_diff_control"
    assert STEERING_ARMS["oracle_terse"]["control"] == "wrongvar_terse"
    assert STEERING_ARMS["oracle_structured"]["control"] == "wrongvar_structured"


def test_steering_inputs_receiver_rows_and_controls(tokenizer):
    plan = build_plan(TINY_COUNTS)
    rows = _tokenized_rows(plan)
    executed = steering.steering_inputs(rows, tokenizer)
    assert len(executed) == 2 * len(plan.groups["pilot"])
    assert all(row["side"] == "A" for row in executed)
    by_id = {row["id"]: row for row in rows}
    for group in plan.groups["pilot"]:
        affected, other = _group_variables(group)
        x_row = next(
            row for row in executed
            if row["group_id"] == group.group_id and row["affected"]
        )
        y_row = next(
            row for row in executed
            if row["group_id"] == group.group_id and not row["affected"]
        )
        assert x_row["id"] == f"{group.group_id}-A-{affected}"
        assert x_row["counterfactual_answer"] == by_id[
            f"{group.group_id}-B-{affected}"
        ]["answer"]
        assert x_row["counterfactual_answer"] != x_row["answer"]
        assert x_row["role"] == "affected_query"
        assert x_row["scored_answers"] == sorted(
            {x_row["answer"], x_row["counterfactual_answer"]}
        )
        assert y_row["counterfactual_answer"] == y_row["answer"]
        assert y_row["role"] == "integrity_query"
        delta_value = int(x_row["counterfactual_answer"]) - int(x_row["answer"])
        candidate = steering.wrong_variable_candidate(
            int(y_row["answer"]), delta_value
        )
        assert y_row["wrong_variable_candidate"] == candidate
        if candidate is not None:
            assert str(candidate) in y_row["scored_answers"]
        steering.check_steering_bucket_fit([x_row, y_row])
    with pytest.raises(ValueError, match="exceeds the pinned target bucket"):
        steering.check_steering_bucket_fit(
            [
                {
                    "id": "row-over",
                    "input_ids": [0] * 121,
                    "scored_answer_token_ids": {"5": [15, 2]},
                }
            ]
        )


def test_load_pilot_run_validation(tmp_path, tokenizer, model_factory):
    plan = build_plan(TINY_COUNTS)
    rows = _tokenized_rows(plan)
    executed = steering.steering_inputs(rows, tokenizer)
    pilot_dir = _fake_pilot_run(tmp_path, plan, rows, executed, tokenizer, model_factory)
    direction_rows, original_rows, p0_rows = _pilot_row_lists(plan, executed)
    with pytest.raises(ValueError, match="not found"):
        steering.load_pilot_run(
            tmp_path / "absent",
            plan_hash=plan.plan_hash,
            width=16,
            direction_rows=direction_rows,
            original_rows=original_rows,
            p0_rows=p0_rows,
        )
    pilot = steering.load_pilot_run(
        pilot_dir,
        plan_hash=plan.plan_hash,
        width=16,
        direction_rows=direction_rows,
        original_rows=original_rows,
        p0_rows=p0_rows,
    )
    assert set(pilot.directions) == set(direction_rows)
    assert all(status == "ok" for status in pilot.direction_row_status.values())
    assert all(pilot.originals.get(name) is not None for name in original_rows)
    assert all(pilot.p0_generations.get(name) is not None for name in p0_rows)
    assert pilot.path_sha256 == hashlib.sha256(
        str(pilot_dir.resolve()).encode("utf-8")
    ).hexdigest()
    # Wrong stage, wrong plan hash, and an incomplete pilot are all refused.
    manifest = json.loads((pilot_dir / "manifest.json").read_text())
    manifest["stage"] = "engineering_smoke"
    (pilot_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not a completed pilot"):
        _load_fixture_pilot(pilot_dir, plan, executed)
    manifest["stage"] = "pilot"
    manifest["plan_hash"] = "0" * 64
    (pilot_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="plan hash mismatch"):
        _load_fixture_pilot(pilot_dir, plan, executed)
    manifest["plan_hash"] = plan.plan_hash
    (pilot_dir / "manifest.json").write_text(json.dumps(manifest))
    completion = json.loads((pilot_dir / "completion.json").read_text())
    completion["status"] = "FAILED"
    (pilot_dir / "completion.json").write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="COMPLETE"):
        _load_fixture_pilot(pilot_dir, plan, executed)
    completion["status"] = "COMPLETE"
    (pilot_dir / "completion.json").write_text(json.dumps(completion))
    # A direction the pilot recorded as ok but whose file is absent is corrupt.
    removed = pilot_dir / "raw" / f"{direction_rows[0]}-direction.safetensors"
    removed.rename(pilot_dir / "raw" / "moved.safetensors")
    with pytest.raises(ValueError, match="corrupt"):
        _load_fixture_pilot(pilot_dir, plan, executed)


def test_steering_metrics_hand_computed():
    """Hand-computed flip/integrity/moved/L on two synthetic groups."""
    inputs = [
        {"id": "g1-A-x", "group_id": "g1", "side": "A", "variable": "x",
         "affected": True, "answer": "5", "counterfactual_answer": "6"},
        {"id": "g1-A-y", "group_id": "g1", "side": "A", "variable": "y",
         "affected": False, "answer": "8", "counterfactual_answer": "8"},
        {"id": "g2-A-x", "group_id": "g2", "side": "A", "variable": "x",
         "affected": True, "answer": "3", "counterfactual_answer": "4"},
        {"id": "g2-A-y", "group_id": "g2", "side": "A", "variable": "y",
         "affected": False, "answer": "7", "counterfactual_answer": "7"},
    ]
    key = steering.condition_key("av_diff", 1.0)
    rows = [
        {"id": "g1-A-x", "group_id": "g1", "conditions": {
            "P0": {"status": "ok", "generation": {"text": "5", "terminated": True}},
            key: {"status": "ok",
                  "generation": {"text": "6", "terminated": True},
                  "next_token_kl": 0.5,
                  "answer_scores": {"5": {"log_probability": -2.0},
                                    "6": {"log_probability": -0.5}}},
        }},
        {"id": "g1-A-y", "group_id": "g1", "conditions": {
            "P0": {"status": "ok", "generation": {"text": "8", "terminated": True}},
            key: {"status": "ok",
                  "generation": {"text": "8", "terminated": True},
                  "next_token_kl": 0.1,
                  "answer_scores": {"8": {"log_probability": -0.1}}},
        }},
        {"id": "g2-A-x", "group_id": "g2", "conditions": {
            "P0": {"status": "ok", "generation": {"text": "3", "terminated": True}},
            key: {"status": "failed", "reason": "steering delta unavailable"},
        }},
        {"id": "g2-A-y", "group_id": "g2", "conditions": {
            "P0": {"status": "ok", "generation": {"text": "7", "terminated": True}},
            key: {"status": "ok",
                  "generation": {"text": "9", "terminated": True},
                  "next_token_kl": 0.3,
                  "answer_scores": {"7": {"log_probability": -1.0}}},
        }},
    ]
    manifest = {
        "stage": "steering",
        "inputs": inputs,
        "steering_frozen": STEERING_FROZEN,
    }
    summary = summarize(manifest, rows)
    measured = summary["per_condition"][key]
    assert measured["groups"] == 2
    assert measured["flip_to_B"]["count"] == 1
    assert measured["flip_to_B"]["rate"] == 0.5
    assert measured["retained_A"]["rate"] == 0.0
    assert measured["moved_x"]["rate"] == 1.0  # g2 failed: counts as moved
    assert measured["y_intact"]["rate"] == 0.5  # g2's y moved off P0
    assert measured["L"]["valid"] == 1
    assert measured["L"]["excluded"] == 1
    assert measured["L"]["mean"] == 1.5
    assert measured["mean_valid_next_token_kl_x"] == 0.5
    assert measured["mean_valid_next_token_kl_y"] == pytest.approx(0.2)
    # A condition no row attempted is ITT-zero, not an error.
    absent = summary["per_condition"][steering.condition_key("oracle_terse", 2.0)]
    assert absent["flip_to_B"]["rate"] == 0.0
    assert absent["moved_x"]["rate"] == 1.0
    assert absent["L"]["status"] == "not_computable"
    decision = summary["decision"]
    assert set(decision["arms"]) == set(STEERING_ARMS)
    arm = decision["arms"]["av_difference"]
    assert arm["criteria"]["c1_flip_to_B"]["value"] == 0.5
    assert arm["criteria"]["c1_flip_to_B"]["met"] is True  # 0.5 >= 0.30
    assert arm["criteria"]["c2_y_integrity"]["met"] is False  # 0.5 < 0.90
    assert arm["success"] is False
    assert decision["frozen_criteria"] == STEERING_FROZEN


def _run_steering_fixture(tmp_path, monkeypatch, model_factory, tokenizer,
                          metadata_factory, broken_directions=()):
    plan = build_plan(TINY_COUNTS)
    rows = _tokenized_rows(plan)
    executed = steering.steering_inputs(rows, tokenizer)
    pilot_dir = _fake_pilot_run(
        tmp_path, plan, rows, executed, tokenizer, model_factory,
        broken_directions=broken_directions,
    )
    pilot = _load_fixture_pilot(pilot_dir, plan, executed)
    monkeypatch.setattr(steering, "load_model", _stub_load_model(model_factory))
    save_file({"weight": torch.eye(16)}, tmp_path / "value_head.safetensors")
    run = RunDirectory(tmp_path / "steering-run")
    result_rows = steering.execute_steering(
        run,
        {role: tmp_path for role in ("target", "av", "ar")},
        {role: tokenizer for role in ("target", "av", "ar")},
        metadata_factory("av"),
        metadata_factory("ar"),
        executed,
        pilot,
        "cpu",
        {},
        {},
    )
    return plan, executed, pilot, run, result_rows


def test_steering_stage_end_to_end_and_audit(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    """Full steering orchestration on fixtures; never real-model evidence."""
    plan, executed, pilot, run, rows = _run_steering_fixture(
        tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
    )
    assert len(rows) == 2 * len(plan.groups["pilot"])
    expected_keys = set(EXPECTED_ROW_CONDITIONS)
    for row in rows:
        assert set(row["conditions"]) == expected_keys
        assert all(
            record["status"] == "ok" for record in row["conditions"].values()
        )
        assert row["identity"]["greedy"]["status"] == "exact"
        assert row["p0_p1_greedy_gate"]["status"] == "exact"
        assert row["cross_run"]["original_vector_bitwise_equal_pilot"] is True
        assert row["cross_run"]["p0_generation_matches_pilot"] is True
    for group in plan.groups["pilot"]:
        for condition in STEERING_CONDITIONS:
            saved = load_numeric(
                run.path / "raw" / f"{group.group_id}-delta-{condition}.safetensors"
            )
            assert saved["delta"].shape == (16,)
        affected, other = _group_variables(group)
        x_row = next(
            row for row in rows
            if row["id"] == f"{group.group_id}-A-{affected}"
        )
        evidence = x_row["steering"]
        assert evidence["affected"] == affected
        assert set(evidence["deltas"]) == set(STEERING_CONDITIONS)
        oracle = evidence["deltas"]["oracle_terse"]
        assert oracle["source"] == "live_ar_oracle_template"
        assert oracle["text_a"] == steering.render_template(
            "terse", affected, int(evidence["affected_answers"]["A"])
        )
        assert oracle["text_a_sha256"] == steering.template_sha256(oracle["text_a"])
        wrongvar = evidence["deltas"]["wrongvar_structured"]
        assert wrongvar["source"] == "live_ar_oracle_template"
        candidate = evidence["wrong_variable_candidate"]["value"]
        assert wrongvar["text_b"] == steering.render_template(
            "structured", other, candidate
        )
        av = evidence["deltas"]["av_diff"]
        assert av["source"] == "pilot_saved_ar_directions"
        assert av["source_rows"] == [
            f"{group.group_id}-A-{affected}",
            f"{group.group_id}-B-{affected}",
        ]
        y_row = next(
            row for row in rows if row["id"] == f"{group.group_id}-A-{other}"
        )
        assert y_row["steering"]["role"] == "integrity_query"
        assert y_row["steering"]["group_evidence_row"] == x_row["id"]
        # The integrity row's wrong-variable condition also scores the candidate.
        wrongvar_condition = y_row["conditions"][
            steering.condition_key("wrongvar_terse", 1.0)
        ]
        assert str(candidate) in wrongvar_condition["answer_scores"]
    manifest = steering.build_steering_manifest(
        {},
        _lock_file(tmp_path),
        executed,
        {"layer": 1},
        {},
        plan=plan,
        pilot=pilot,
        tokenization={"accepted": 4, "rejected": 0, "rejections": [],
                      "prior_excluded_prompts": 0},
        group_counts_source="dev override --group-counts-json",
    )
    assert manifest["stage"] == "steering"
    assert manifest["alpha_grid"] == list(ALPHA_GRID)
    assert manifest["evaluation_alpha"] == 1.0
    assert manifest["steering_frozen"] == STEERING_FROZEN
    assert manifest["plan_hash"] == plan.plan_hash
    assert manifest["reused_split"]["split"] == "pilot"
    assert manifest["target_bucket"] == 128
    assert manifest["pilot_run"]["path_sha256"] == pilot.path_sha256
    assert manifest["pilot_run"]["manifest_sha256"] == pilot.manifest_sha256
    assert manifest["pilot_run"]["plan_hash"] == plan.plan_hash
    assert manifest["pilot_run"]["completion_status"] == "COMPLETE"
    for style, entry in manifest["oracle_templates"].items():
        assert entry["text"] == TEMPLATES[style]
        assert entry["sha256"] == steering.template_sha256(TEMPLATES[style])
    assert set(manifest["arms"]) == set(STEERING_ARMS)
    write_json(run.path / "manifest.json", manifest)
    summary = summarize(manifest, rows)
    assert summary["stage"] == "steering"
    assert len(summary["successful_groups"]) == len(plan.groups["pilot"])
    assert not summary["failed_groups"] and not summary["skipped_groups"]
    assert summary["executed_rows"] == len(executed)
    assert set(summary["per_condition"]) == {
        steering.condition_key(condition, alpha)
        for condition in STEERING_CONDITIONS
        for alpha in ALPHA_GRID
    }
    for measured in summary["per_condition"].values():
        assert measured["groups"] == len(plan.groups["pilot"])
        assert 0.0 <= measured["flip_to_B"]["rate"] <= 1.0
        assert measured["flip_to_B"]["interval"]["seed"] == 205100
        assert measured["flip_to_B"]["interval"]["resamples"] == 3000
    assert summary["greedy_identity"] == {
        "identity_stage": {"exact": len(executed), "drift_diverged": 0},
        "behavior_stage": {"exact": len(executed), "drift_diverged": 0},
    }
    assert summary["cross_run"]["pilot_p0_generation"] == {
        "matches": len(executed),
        "differs": 0,
        "missing": 0,
    }
    assert summary["cross_run"]["original_vector_bitwise"] == {
        "equal": len(executed),
        "different": 0,
        "missing": 0,
    }
    decision = summary["decision"]
    assert decision["frozen_criteria"] == STEERING_FROZEN
    assert set(decision["arms"]) == set(STEERING_ARMS)
    write_json(run.path / "summary.json", summary)
    write_json(
        run.path / "completion.json",
        {
            "status": "COMPLETE",
            "stage": "steering",
            "decision": decision,
            "plan_hash": plan.plan_hash,
            "pilot_run": manifest["pilot_run"],
        },
    )
    run.reports_zip()
    result = audit_run(run.path)
    assert result["status"] == "PASS"
    assert result["stage"] == "steering"
    assert result["groups"] == len(plan.groups["pilot"])
    with zipfile.ZipFile(run.path / "reports.zip") as archive:
        assert not any(
            name.endswith(".safetensors") or name.startswith("raw/")
            for name in archive.namelist()
        )


def _refresh_inventory_entry(run, name):
    """Re-hash one packaged file after a deliberate in-test tamper.

    The auditor checks packaged-file hashes first; refreshing the tampered
    file's inventory entry lets the steering-specific checks below be the
    ones under test (the inventory check itself is covered in test_runner.py).
    """
    inventory = json.loads((run.path / "inventory.json").read_text())
    for entry in inventory["included"]:
        if entry["file"] == name:
            entry["sha256"] = hashlib.sha256((run.path / name).read_bytes()).hexdigest()
            entry["bytes"] = (run.path / name).stat().st_size
    (run.path / "inventory.json").write_text(json.dumps(inventory))


def test_steering_audit_detects_tampering(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    plan, executed, pilot, run, rows = _run_steering_fixture(
        tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
    )
    manifest = steering.build_steering_manifest(
        {},
        _lock_file(tmp_path),
        executed,
        {"layer": 1},
        {},
        plan=plan,
        pilot=pilot,
        tokenization={"accepted": 4, "rejected": 0, "rejections": [],
                      "prior_excluded_prompts": 0},
        group_counts_source="dev override --group-counts-json",
    )
    write_json(run.path / "manifest.json", manifest)
    summary = summarize(manifest, rows)
    write_json(run.path / "summary.json", summary)
    write_json(
        run.path / "completion.json",
        {"status": "COMPLETE", "stage": "steering", "decision": summary["decision"]},
    )
    run.reports_zip()
    assert audit_run(run.path)["status"] == "PASS"
    # A tampered frozen-criteria block changes the computed decision, so a
    # consistent tamper must also regenerate the summary; the steering check
    # then pins the criteria back to the audited code constants.
    tampered = json.loads((run.path / "manifest.json").read_text())
    tampered["steering_frozen"]["flip_to_B_threshold"] = 0.01
    (run.path / "manifest.json").write_text(json.dumps(tampered))
    _refresh_inventory_entry(run, "manifest.json")
    (run.path / "summary.json").write_text(
        json.dumps(summarize(tampered, rows), sort_keys=True)
    )
    _refresh_inventory_entry(run, "summary.json")
    with pytest.raises(ValueError, match="frozen steering criteria"):
        audit_run(run.path)
    tampered["steering_frozen"] = STEERING_FROZEN
    tampered["oracle_templates"]["terse"]["text"] = "x is 5."
    (run.path / "manifest.json").write_text(json.dumps(tampered))
    _refresh_inventory_entry(run, "manifest.json")
    # Templates do not enter the summary; restore it so the template check fires.
    (run.path / "summary.json").write_text(json.dumps(summary, sort_keys=True))
    _refresh_inventory_entry(run, "summary.json")
    with pytest.raises(ValueError, match="oracle template text"):
        audit_run(run.path)
    (run.path / "manifest.json").write_text(json.dumps(manifest))
    _refresh_inventory_entry(run, "manifest.json")
    tampered = json.loads((run.path / "completion.json").read_text())
    tampered["decision"]["successful_arms"] = ["av_difference"]
    (run.path / "completion.json").write_text(json.dumps(tampered))
    _refresh_inventory_entry(run, "completion.json")
    with pytest.raises(ValueError, match="decision does not reproduce"):
        audit_run(run.path)


def test_steering_itt_when_a_pilot_direction_is_missing(
    tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory
):
    """A failed arm-1 direction fails that arm's conditions, never hidden."""
    plan = build_plan(TINY_COUNTS)
    broken_group = plan.groups["pilot"][0]
    broken_row = f"{broken_group.group_id}-A-{broken_group.affected}"
    plan, executed, pilot, run, rows = _run_steering_fixture(
        tmp_path, monkeypatch, model_factory, tokenizer, metadata_factory,
        broken_directions={broken_row},
    )
    by_id = {row["id"]: row for row in rows}
    receiver = by_id[broken_row]
    for alpha in ALPHA_GRID:
        record = receiver["conditions"][steering.condition_key("av_diff", alpha)]
        assert record["status"] == "failed"
        assert "pilot direction unavailable" in record["reason"]
        assert "next_token_kl" not in record  # no fabricated KL
        # Other arms still ran for the same row.
        assert receiver["conditions"][
            steering.condition_key("oracle_terse", alpha)
        ]["status"] == "ok"
    manifest = {
        "stage": "steering",
        "inputs": executed,
        "steering_frozen": STEERING_FROZEN,
    }
    summary = summarize(manifest, rows)
    assert summary["failed_groups"] == [broken_group.group_id]
    assert len(summary["successful_groups"]) == len(plan.groups["pilot"]) - 1
    measured = summary["per_condition"][steering.condition_key("av_diff", 1.0)]
    assert measured["x_row_failed_or_missing"] == 1
    assert measured["groups"] == len(plan.groups["pilot"])  # ITT denominator
    assert measured["flip_to_B"]["count"] <= len(plan.groups["pilot"]) - 1
    decision = summary["decision"]
    assert decision["arms"]["av_difference"]["criteria"]["c1_flip_to_B"][
        "value"
    ] == measured["flip_to_B"]["rate"]


def test_steering_main_requires_pilot_run_and_reports_missing_models(tmp_path):
    with pytest.raises(SystemExit):
        steering.parse_args([])
    args = steering.parse_args(["--pilot-run", "pilot-dir"])
    assert args.pilot_run.name == "pilot-dir"
    run = tmp_path / "run"
    status = steering.main(
        [
            "--cache", str(tmp_path / "absent"),
            "--run-dir", str(run),
            "--device", "cpu",
            "--pilot-run", str(tmp_path / "pilot-run"),
        ]
    )
    assert status == 1
    completion = json.loads((run / "completion.json").read_text())
    assert completion["stage"] == "steering"
    assert completion["real_model_checks"] == "NOT RUN"
    assert "--fetch-models" in completion["error"]
    with pytest.raises(FileExistsError):
        steering.main(
            [
                "--run-dir", str(run),
                "--pilot-run", str(tmp_path / "pilot-run"),
            ]
        )


def test_runner_stages_are_untouched_by_the_steering_assay():
    """The smoke/calibration/pilot runner paths never grew a steering stage."""
    with pytest.raises(SystemExit):
        runner.parse_args(["--stage", "steering"])
    assert runner.parse_args([]).stage == "smoke"
