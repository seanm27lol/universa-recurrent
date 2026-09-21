"""Replay engineering outcomes from local evidence, without loading any model."""

import argparse
import json
from pathlib import Path
from .artifacts import load_numeric, sha256_file
from .metrics import exact_integer, next_token_kl

CONDITIONS = ("P0", "P1", "P2", "P3", "P5", "donor", "smoke_median_norm")


def summarize(manifest: dict, rows: list[dict]) -> dict:
    expected = {row["id"]: row for row in manifest["inputs"]}
    actual = {row["id"]: row for row in rows}
    if len(actual) != len(rows) or actual.keys() - expected.keys():
        raise ValueError("duplicate or unexpected result identity")
    group_ids = {row["group_id"] for row in expected.values()}
    completed, failed, skipped = [], [], []
    for group in sorted(group_ids):
        ids = [name for name, row in expected.items() if row["group_id"] == group]
        if not any(
            actual.get(name, {}).get("attempted")
            or actual.get(name, {}).get("conditions")
            for name in ids
        ):
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
        "edit_eligibility": "NOT ASSESSED: Milestone 1 has no scientific edit-coverage measurement",
        "uneditable_groups": None,
        "metrics": metrics,
        "scope": "Engineering smoke only. No scientific pilot, locked validation, PCA, or text-edit claim.",
    }


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
    return {
        "status": "PASS",
        "groups": calculated["groups"],
        "limits": "Replays saved counts and next-token KL, not execution provenance or all teacher-forced logits.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run), indent=2))


if __name__ == "__main__":
    main()
