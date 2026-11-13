"""Score a complete prepared population from saved text, without model/producer imports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .dependency_diagnostic import score_population
from .rescore_generations import ScoringError, execute_saved_program


def score(source_path, predictions_path, output):
    """Refuse missing/extra/duplicate predictions; preserve malformed completions as failures."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    raw_source = Path(source_path).read_bytes()
    source_rows = [json.loads(line) for line in raw_source.decode().splitlines()]
    source = {r["example_id"]: r for r in source_rows}
    if not source or len(source) != len(source_rows):
        raise ScoringError("empty or duplicated source population")
    for record in source_rows:
        execute_saved_program(record)
    raw_predictions = Path(predictions_path).read_bytes()
    predictions = [json.loads(line) for line in raw_predictions.decode().splitlines()]
    ids = [r["example_id"] for r in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(source):
        raise ScoringError("missing, extra or duplicate predictions; population is incomplete")
    for record in predictions:
        # The model-facing prompt export has no target. Join gold only after generation.
        expected = source[record["example_id"]]["target_text"]
        if record.get("target_text", expected) != expected:
            raise ScoringError("prediction carries a mismatched gold target")
        record["target_text"] = expected
        for field in ("stopped_on_eos", "truncated", "prompt_truncated"):
            if not isinstance(record.get(field), bool):
                raise ScoringError(f"prediction needs explicit boolean {field}")
    report, _ = score_population(predictions, source, expected_count=len(source))
    noop = [{**r, "completion_text": " ".join(f"r{i}={v}" for i, v in enumerate(source[r["example_id"]]["initial"]))}
            for r in predictions]
    noop_report, _ = score_population(noop, source)
    report["no_op_baseline"] = {key: noop_report[key] for key in
                                ("complete_answers", "registers", "per_register")}
    report["source_sha256"] = hashlib.sha256(raw_source).hexdigest()
    report["predictions_sha256"] = hashlib.sha256(raw_predictions).hexdigest()
    report["scope"] = "saved text only; no training, generation, teacher-forced CE or mechanism claim"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = score(args.source, args.predictions, args.output)
    print(json.dumps(report["complete_answers"]))


if __name__ == "__main__":
    main()
