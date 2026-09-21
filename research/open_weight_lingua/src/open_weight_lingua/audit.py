"""Replay engineering outcomes from local evidence, without loading any model."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

from .artifacts import load_numeric, sha256_file
from .controls import load_fit
from .metrics import exact_integer, next_token_kl
from .stats import (
    DEFAULT_RESAMPLES,
    EditingRecord,
    LogProbPair,
    block_estimates,
    editing_effect,
    paired_accuracy_loss,
    paired_logprob_difference,
)
from .tasks import VARIABLES
from .text_edits import RULE_SHA256, RULE_VERSION, parse

CONDITIONS = ("P0", "P1", "P2", "P3", "P5", "donor", "smoke_median_norm")
PILOT_BASE_CONDITIONS = (
    "P0",
    "P1",
    "P2",
    "P3",
    "P4",
    "P5",
    "donor",
    "calibration_median_norm",
)
EDIT_CONDITIONS = ("edited", "wrong_variable_edit")
PILOT_CONDITIONS = (*PILOT_BASE_CONDITIONS, *EDIT_CONDITIONS)

MEDIAN_RECORD_NAME = "calibration_median_norm.json"
MEDIAN_RECORD_FORMAT = "open_weight_lingua.calibration_median_norm.v1"

FROZEN_THRESHOLDS = {
    "unmodified_accuracy_floor": 0.80,
    "p2_accuracy_loss_upper_points": 5.0,
    "p2_minus_p3_logprob_lower_exclusive": 0.0,
    "edit_eligible_group_floor": 32,
    "bootstrap_resamples": DEFAULT_RESAMPLES,
    "bootstrap_seed": 203100,
    "stage_budget_seconds": 8 * 3600,
    "quoted_from": "phase_two_brief.md Sections 7-9 proposed design choices, not paper results",
}


def _expected_actual(manifest: dict, rows: list[dict]) -> tuple[dict, dict]:
    expected = {row["id"]: row for row in manifest["inputs"]}
    actual = {row["id"]: row for row in rows}
    if len(actual) != len(rows) or actual.keys() - expected.keys():
        raise ValueError("duplicate or unexpected result identity")
    return expected, actual


def _group_members(expected: dict) -> tuple[list, dict]:
    group_ids = sorted({row["group_id"] for row in expected.values()})
    members = {
        group: [name for name, row in expected.items() if row["group_id"] == group]
        for group in group_ids
    }
    return group_ids, members


def _attempted(actual: dict, name: str) -> bool:
    row = actual.get(name, {})
    return bool(
        row.get("attempted") or row.get("conditions") or row.get("extraction")
    )


def _greedy_identity_counts(actual: dict) -> dict:
    """Per-stage exact/drift_diverged tallies; never hidden, never decision input."""
    counts = {
        "identity_stage": {"exact": 0, "drift_diverged": 0},
        "behavior_stage": {"exact": 0, "drift_diverged": 0},
    }
    for row in actual.values():
        gate = row.get("identity", {}).get("greedy", {})
        if gate.get("status") in counts["identity_stage"]:
            counts["identity_stage"][gate["status"]] += 1
        gate = row.get("p0_p1_greedy_gate", {})
        if gate.get("status") in counts["behavior_stage"]:
            counts["behavior_stage"][gate["status"]] += 1
    return counts


def _summarize_smoke(manifest: dict, rows: list[dict]) -> dict:
    expected, actual = _expected_actual(manifest, rows)
    group_ids = {row["group_id"] for row in expected.values()}
    completed, failed, skipped = [], [], []
    for group in sorted(group_ids):
        ids = [name for name, row in expected.items() if row["group_id"] == group]
        if not any(_attempted(actual, name) for name in ids):
            skipped.append(group)
        elif all(
            name in actual
            and all(
                actual[name].get("conditions", {}).get(c, {}).get("status") == "ok"
                for c in CONDITIONS
            )
            for name in ids
        ):
            completed.append(group)
        else:
            failed.append(group)
    metrics = {}
    for condition in CONDITIONS:
        valid, correct, agreement, divergence, log_probabilities = 0, 0, 0, [], []
        for name, expected_row in expected.items():
            measured = actual.get(name, {}).get("conditions", {})
            record = measured.get(condition, {})
            if record.get("status") != "ok":
                continue
            valid += 1
            generation = record["generation"]
            correct += generation["terminated"] and exact_integer(
                generation["text"], expected_row["answer"]
            )
            baseline = measured.get("P0", {}).get("generation", {})
            agreement += generation == baseline
            divergence.append(record["next_token_kl"])
            log_probabilities.append(
                record["answer_scores"][expected_row["answer"]]["log_probability"]
            )
        metrics[condition] = {
            "attempted_prompt_variants": len(expected),
            "valid": valid,
            "failed_or_missing": len(expected) - valid,
            "exact_answers": int(correct),
            "accuracy_all_variants": correct / len(expected),
            "agreement_with_P0_all_variants": agreement / len(expected),
            "mean_valid_next_token_kl": sum(divergence) / valid if valid else None,
            "mean_valid_correct_answer_log_probability": sum(log_probabilities) / valid
            if valid
            else None,
        }
    return {
        "groups": len(group_ids),
        "prompt_variants": len(expected),
        "successful_groups": completed,
        "failed_groups": failed,
        "skipped_groups": skipped,
        "greedy_identity": _greedy_identity_counts(actual),
        "edit_eligibility": "NOT ASSESSED: Milestone 1 has no scientific edit-coverage measurement",
        "uneditable_groups": None,
        "metrics": metrics,
        "scope": "Engineering smoke only. No scientific pilot, locked validation, PCA, or text-edit claim.",
    }


def _condition_metric(expected, actual, condition, denominator_ids):
    """Per-condition counts; failed or missing rows stay in the denominator."""
    valid, correct, agreement, divergence, log_probabilities = 0, 0, 0, [], []
    for name in denominator_ids:
        expected_row = expected[name]
        measured = actual.get(name, {}).get("conditions", {})
        record = measured.get(condition, {})
        if record.get("status") != "ok":
            continue
        valid += 1
        generation = record["generation"]
        correct += generation["terminated"] and exact_integer(
            generation["text"], expected_row["answer"]
        )
        baseline = measured.get("P0", {}).get("generation", {})
        agreement += generation == baseline
        divergence.append(record["next_token_kl"])
        log_probabilities.append(
            record["answer_scores"][expected_row["answer"]]["log_probability"]
        )
    total = len(denominator_ids)
    return {
        "attempted_prompt_variants": total,
        "valid": valid,
        "failed_or_missing": total - valid,
        "exact_answers": int(correct),
        "accuracy_all_variants": correct / total if total else None,
        "agreement_with_P0_all_variants": agreement / total if total else None,
        "mean_valid_next_token_kl": sum(divergence) / valid if valid else None,
        "mean_valid_correct_answer_log_probability": sum(log_probabilities) / valid
        if valid
        else None,
    }


def _summarize_calibration(manifest: dict, rows: list[dict]) -> dict:
    expected, actual = _expected_actual(manifest, rows)
    group_ids, members = _group_members(expected)
    completed, failed, skipped = [], [], []
    for group in group_ids:
        ids = members[group]
        if not any(_attempted(actual, name) for name in ids):
            skipped.append(group)
        elif all(
            actual.get(name, {}).get("extraction", {}).get("status") == "ok"
            for name in ids
        ):
            completed.append(group)
        else:
            failed.append(group)
    norms = [
        actual[name]["original_norm"]
        for name in expected
        if actual.get(name, {}).get("extraction", {}).get("status") == "ok"
    ]
    failed_rows = [
        name
        for name in expected
        if actual.get(name, {}).get("extraction", {}).get("status") == "failed"
    ]
    return {
        "groups": len(group_ids),
        "prompt_variants": len(expected),
        "successful_groups": completed,
        "failed_groups": failed,
        "skipped_groups": skipped,
        "calibration": {
            "planned_variants": len(expected),
            "extractions": len(norms),
            "failed_extractions": len(failed_rows),
            "unattempted_variants": len(expected) - len(norms) - len(failed_rows),
            "median_norm": statistics.median(norms) if norms else None,
            "policy": "median of all successful calibration extraction norms, frozen before pilot use; the PCA fit lives in baseline_fit.safetensors at the run root",
        },
        "edit_eligibility": "NOT ASSESSED: calibration extracts activations only",
        "uneditable_groups": None,
        "metrics": {},
        "scope": "Numerical calibration only. No behavioral condition, reconstruction, or edit is measured in this stage.",
    }


def _receiver(expected, group):
    """The source-side row querying the affected variable carries the edit."""
    affected = next(
        row["variable"]
        for row in expected.values()
        if row["group_id"] == group and row["side"] == "A" and row["affected"]
    )
    return affected, f"{group}-A-{affected}"


def _edit_coverage(expected, actual, group_ids):
    """Re-derive eligibility from saved descriptions with the frozen parser.

    Cross-checks the recorded per-group status so an audit fails loudly when
    the saved evidence disagrees with the frozen rule.
    """
    per_variable = {v: {"eligible": 0, "absent": 0, "ambiguous": 0} for v in VARIABLES}
    exclusions = {"absent": 0, "ambiguous": 0, "description_unavailable": 0}
    eligible, statuses = 0, {}
    for group in group_ids:
        affected, receiver_id = _receiver(expected, group)
        record = actual.get(receiver_id, {})
        description_record = record.get("description", {})
        description = (
            description_record.get("description")
            if description_record.get("status") == "ok"
            else None
        )
        if description is None:
            statuses[group] = "description_unavailable"
            exclusions["description_unavailable"] += 1
            continue
        parsed = parse(description)
        for variable in VARIABLES:
            per_variable[variable][parsed[variable].status] += 1
        status = parsed[affected].status
        recorded = record.get("edit", {})
        if (
            recorded.get("affected_status") is not None
            and recorded["affected_status"] != status
        ):
            raise ValueError(
                f"recorded edit status disagrees with the frozen parser: {receiver_id}"
            )
        statuses[group] = status
        if status == "eligible":
            eligible += 1
        else:
            exclusions[status] += 1
    total = len(group_ids)
    coverage = {
        "groups": total,
        "eligible_groups": eligible,
        "eligible_fraction": eligible / total if total else 0.0,
        "exclusion_reasons": exclusions,
        "per_variable_status": per_variable,
        "group_statuses": statuses,
        "rule_version": RULE_VERSION,
        "rule_sha256": RULE_SHA256,
    }
    return coverage, statuses


def _group_correct(expected, actual, ids, condition) -> bool:
    """A group is correct only when every variant answers exactly (ITT)."""
    for name in ids:
        record = actual.get(name, {}).get("conditions", {}).get(condition, {})
        generation = record.get("generation") if record.get("status") == "ok" else None
        if not generation or not (
            generation["terminated"]
            and exact_integer(generation["text"], expected[name]["answer"])
        ):
            return False
    return True


def _correct_logprob(expected, actual, ids, condition):
    """Per-group mean correct-answer log probability, or None if incomplete."""
    values = []
    for name in ids:
        record = actual.get(name, {}).get("conditions", {}).get(condition, {})
        if record.get("status") != "ok":
            return None
        score = record.get("answer_scores", {}).get(expected[name]["answer"])
        if score is None:
            return None
        values.append(score["log_probability"])
    return statistics.fmean(values)


def _editing_records(expected, actual, group_ids, members, statuses):
    records, excluded = [], 0
    for group in group_ids:
        if statuses[group] != "eligible":
            continue
        _, receiver_id = _receiver(expected, group)
        record = actual.get(receiver_id, {})
        conditions = record.get("conditions", {})
        donor_id = record.get("controls", {}).get("donor_id")
        original_answer = expected[receiver_id]["answer"]
        counterfactual_answer = expected.get(donor_id, {}).get("answer")

        def pair(condition):
            measured = conditions.get(condition, {})
            if measured.get("status") != "ok" or counterfactual_answer is None:
                return None
            scores = measured.get("answer_scores", {})
            if original_answer not in scores or counterfactual_answer not in scores:
                return None
            return LogProbPair(
                counterfactual=scores[counterfactual_answer]["log_probability"],
                original=scores[original_answer]["log_probability"],
            )

        unedited, edited = pair("P2"), pair("edited")
        if unedited is None or edited is None:
            excluded += 1
            continue
        records.append(
            EditingRecord(
                unedited=unedited,
                edited=edited,
                wrong_variable=pair("wrong_variable_edit"),
                donor=pair("donor"),
            )
        )
    return records, excluded


def _pilot_statistics(expected, actual, group_ids, members, statuses, thresholds):
    resamples = thresholds["bootstrap_resamples"]
    seed = thresholds["bootstrap_seed"]
    encoding = (
        "failed or missing generations count as incorrect (intention-to-test); "
        "a group is correct only when all four variants answer exactly; "
        "P2-P3 uses per-group mean correct-answer log probability and counts "
        "groups excluded for incomplete measurements"
    )
    if not group_ids:
        empty = {"status": "not_computable", "reason": "no pilot groups", "groups": 0}
        return {
            "p2_accuracy_loss_points": dict(empty),
            "p4_accuracy_loss_points": dict(empty),
            "p2_minus_p3_logprob": dict(empty),
            "editing_effect": {
                "status": "untested",
                "reason": "no pilot groups",
                "eligible_groups": 0,
                "records_used": 0,
                "eligible_excluded_incomplete": 0,
            },
            "pilot_block_estimates": {"per_block": {}, "pooled": None, "block_sizes": {}},
            "encoding": encoding,
        }
    p0_correct = [
        _group_correct(expected, actual, members[group], "P0") for group in group_ids
    ]
    p2_correct = [
        _group_correct(expected, actual, members[group], "P2") for group in group_ids
    ]
    p4_correct = [
        _group_correct(expected, actual, members[group], "P4") for group in group_ids
    ]
    p2_loss = paired_accuracy_loss(p0_correct, p2_correct, resamples=resamples, seed=seed)
    p4_loss = paired_accuracy_loss(p0_correct, p4_correct, resamples=resamples, seed=seed)
    p2_logprob, p3_logprob, logprob_excluded = [], [], 0
    for group in group_ids:
        p2_value = _correct_logprob(expected, actual, members[group], "P2")
        p3_value = _correct_logprob(expected, actual, members[group], "P3")
        if p2_value is None or p3_value is None:
            logprob_excluded += 1
        else:
            p2_logprob.append(p2_value)
            p3_logprob.append(p3_value)
    if p2_logprob:
        contrast = asdict(
            paired_logprob_difference(
                p2_logprob, p3_logprob, resamples=resamples, seed=seed
            )
        )
        contrast["groups_excluded"] = logprob_excluded
    else:
        contrast = {
            "status": "not_computable",
            "reason": "no group has complete P2 and P3 correct-answer log probabilities",
            "groups": 0,
            "groups_excluded": logprob_excluded,
        }
    records, editing_excluded = _editing_records(
        expected, actual, group_ids, members, statuses
    )
    if records:
        effect = asdict(editing_effect(records, resamples=resamples, seed=seed))
        effect["eligible_groups"] = sum(
            status == "eligible" for status in statuses.values()
        )
        effect["records_used"] = len(records)
        effect["eligible_excluded_incomplete"] = editing_excluded
    else:
        effect = {
            "status": "untested",
            "reason": "no edit-eligible group has complete unedited and edited measurements",
            "eligible_groups": sum(
                status == "eligible" for status in statuses.values()
            ),
            "records_used": 0,
            "eligible_excluded_incomplete": editing_excluded,
        }
    blocks = block_estimates(
        [100.0 * (int(a) - int(b)) for a, b in zip(p0_correct, p2_correct)],
        ["pilot"] * len(group_ids),
    )
    return {
        "p2_accuracy_loss_points": asdict(p2_loss),
        "p4_accuracy_loss_points": asdict(p4_loss),
        "p2_minus_p3_logprob": contrast,
        "editing_effect": effect,
        "pilot_block_estimates": asdict(blocks),
        "encoding": encoding,
    }


def _sensitivity(metrics):
    detail = {}
    for name in ("donor", "P5"):
        measured = metrics[name]
        agreeing = round(
            (measured["agreement_with_P0_all_variants"] or 0.0)
            * measured["attempted_prompt_variants"]
        )
        detail[name] = {
            "generations_differing_from_P0": max(0, measured["valid"] - agreeing),
            "mean_valid_next_token_kl": measured["mean_valid_next_token_kl"],
        }
    present = any(
        control["generations_differing_from_P0"] > 0
        or (control["mean_valid_next_token_kl"] or 0.0) > 1e-6
        for control in detail.values()
    )
    return {
        "present": present,
        "controls": detail,
        "rule": "present iff donor or P5 changes at least one greedy generation or shows "
        "mean valid next-token KL above 1e-6; an operational pilot design choice",
    }


def _pilot_decision(metrics, coverage, statistics_record, thresholds):
    sensitivity = _sensitivity(metrics)
    loss = statistics_record["p2_accuracy_loss_points"]
    contrast = statistics_record["p2_minus_p3_logprob"]
    lower = contrast.get("lower")
    upper = loss.get("upper")
    accuracy = metrics["P0"]["accuracy_all_variants"]
    criteria = {
        "unmodified_accuracy": {
            "value": accuracy,
            "floor": thresholds["unmodified_accuracy_floor"],
            "met": accuracy is not None
            and accuracy >= thresholds["unmodified_accuracy_floor"],
        },
        "intervention_sensitivity": {
            "present": sensitivity["present"],
            "met": sensitivity["present"],
        },
        "p2_accuracy_loss": {
            "upper_points": upper,
            "limit_points": thresholds["p2_accuracy_loss_upper_points"],
            "met": upper is not None
            and upper <= thresholds["p2_accuracy_loss_upper_points"],
        },
        "p2_minus_p3_logprob": {
            "lower": lower,
            "status": contrast.get("status", "computed"),
            "met": lower is not None
            and lower > thresholds["p2_minus_p3_logprob_lower_exclusive"],
        },
        "edit_eligible_groups": {
            "value": coverage["eligible_groups"],
            "floor": thresholds["edit_eligible_group_floor"],
            "met": coverage["eligible_groups"]
            >= thresholds["edit_eligible_group_floor"],
        },
    }
    unmet = [name for name, criterion in criteria.items() if not criterion["met"]]
    return {
        "criteria": criteria,
        "sensitivity": sensitivity,
        "recommendation": "keep" if not unmet else "stop",
        "unmet_criteria": unmet,
        "basis": "derived only from the frozen thresholds quoted in the manifest; "
        "the pilot never starts validation",
    }


def _summarize_pilot(manifest: dict, rows: list[dict]) -> dict:
    expected, actual = _expected_actual(manifest, rows)
    group_ids, members = _group_members(expected)
    completed, failed, skipped = [], [], []
    for group in group_ids:
        ids = members[group]
        if not any(_attempted(actual, name) for name in ids):
            skipped.append(group)
        elif all(
            name in actual
            and all(
                actual[name].get("conditions", {}).get(c, {}).get("status") == "ok"
                for c in PILOT_BASE_CONDITIONS
            )
            and all(
                record.get("status") == "ok"
                for record in actual[name].get("conditions", {}).values()
            )
            for name in ids
        ):
            completed.append(group)
        else:
            failed.append(group)
    metrics = {
        condition: _condition_metric(expected, actual, condition, list(expected))
        for condition in PILOT_BASE_CONDITIONS
    }
    for condition in EDIT_CONDITIONS:
        attempted_ids = [
            name
            for name in expected
            if condition in actual.get(name, {}).get("conditions", {})
        ]
        metrics[condition] = _condition_metric(
            expected, actual, condition, attempted_ids
        )
    coverage, statuses = _edit_coverage(expected, actual, group_ids)
    thresholds = manifest.get("frozen_thresholds", FROZEN_THRESHOLDS)
    statistics_record = _pilot_statistics(
        expected, actual, group_ids, members, statuses, thresholds
    )
    return {
        "groups": len(group_ids),
        "prompt_variants": len(expected),
        "successful_groups": completed,
        "failed_groups": failed,
        "skipped_groups": skipped,
        "greedy_identity": _greedy_identity_counts(actual),
        "edit_eligibility": coverage,
        "uneditable_groups": len(group_ids) - coverage["eligible_groups"],
        "metrics": metrics,
        "statistics": statistics_record,
        "decision": _pilot_decision(
            metrics, coverage, statistics_record, thresholds
        ),
        "scope": "One bounded pilot. Pooled pilot estimates only; no validation block was "
        "opened and no semantic success is claimed from this summary.",
    }


def summarize(manifest: dict, rows: list[dict]) -> dict:
    stage = manifest.get("stage", "engineering_smoke")
    if stage == "engineering_smoke":
        return _summarize_smoke(manifest, rows)
    if stage == "numerical_calibration":
        return _summarize_calibration(manifest, rows)
    if stage == "pilot":
        return _summarize_pilot(manifest, rows)
    raise ValueError(f"unknown manifest stage: {stage}")


def audit_run(path):
    path = Path(path)
    inventory = json.loads((path / "inventory.json").read_text())
    for entry in inventory["included"] + inventory["retained_locally_not_in_zip"]:
        file = path / entry["file"]
        if file.resolve().parent not in (path.resolve(), (path / "raw").resolve()):
            raise ValueError("unsafe evidence inventory path")
        if (
            file.stat().st_size != entry["bytes"]
            or sha256_file(file) != entry["sha256"]
        ):
            raise ValueError(f"evidence hash mismatch: {entry['file']}")
    manifest = json.loads((path / "manifest.json").read_text())
    rows = json.loads((path / "results.json").read_text())
    for row in rows:
        if not row.get("conditions"):
            continue
        evidence = load_numeric(path / "raw" / f"{row['id']}-behavior.safetensors")
        for condition, record in row["conditions"].items():
            if record["status"] != "ok":
                continue
            measured = next_token_kl(evidence["P0"], evidence[condition])
            if abs(measured - record["next_token_kl"]) > 1e-6:
                raise ValueError(f"KL replay mismatch: {row['id']}/{condition}")
    calculated = summarize(manifest, rows)
    if calculated != json.loads((path / "summary.json").read_text()):
        raise ValueError("summary does not reproduce from saved results")
    stage = manifest.get("stage", "engineering_smoke")
    if stage == "engineering_smoke":
        return {
            "status": "PASS",
            "groups": calculated["groups"],
            "limits": "Replays saved counts and next-token KL, not execution provenance or all teacher-forced logits.",
        }
    if stage == "numerical_calibration":
        fit = load_fit(path)
        completion = json.loads((path / "completion.json").read_text())
        if completion.get("baseline_fit_identity") != fit.identity:
            raise ValueError("completion fit identity does not match the pinned fit")
        sidecar = json.loads((path / MEDIAN_RECORD_NAME).read_text())
        if sidecar.get("median_norm") != calculated["calibration"]["median_norm"]:
            raise ValueError("recorded median norm does not reproduce from results")
        return {
            "status": "PASS",
            "groups": calculated["groups"],
            "stage": stage,
            "baseline_fit_identity": fit.identity,
            "limits": "Replays extraction counts, the median norm, and the pinned fit "
            "identity; cannot replay target forwards or the fitting environment.",
        }
    completion = json.loads((path / "completion.json").read_text())
    if completion.get("decision") != calculated["decision"]:
        raise ValueError("completion decision does not reproduce from saved results")
    return {
        "status": "PASS",
        "groups": calculated["groups"],
        "stage": stage,
        "limits": "Replays saved counts, next-token KL, the frozen statistics and the "
        "decision; cannot replay AV/AR generation, target forwards, or wall-clock "
        "cost projections.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run), indent=2))


if __name__ == "__main__":
    main()
