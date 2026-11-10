"""Export and independently reproduce a weight-free native varied-program study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

try:
    from . import rescore_generations
except ImportError:
    import rescore_generations


SPLITS = ("train", "dev", "initial_values", "unseen_composition", "longer_length")
EVALUATION_FORMAT = "native-program-study-evaluation-v1"
SUMMARY_FORMAT = "native-program-study-weight-free-summary-v1"
BUNDLE_FORMAT = "native-program-study-weight-free-evidence-v1"
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz"}


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def copy(source, destination):
    source = Path(source)
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def learning_endpoints(path, event, arm):
    records = [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip() and json.loads(line).get("event") == event
    ]
    return {
        "first": records[0] if records else None,
        "last": records[-1] if records else None,
        "records": len(records),
        "full_curve": f"arms/{arm}/learning_curve.jsonl",
    }


def without_keys(payload, excluded):
    return {key: value for key, value in payload.items() if key not in excluded}


def compact_generation(values):
    records = values.get("records", [])
    valid = values.get("format_valid_count", 0)
    return {
        "examples": values.get("examples", 0),
        "exact_answers": values.get("exact_count", 0),
        "exact_match": values.get("exact_match"),
        "format": {
            "valid_count": valid,
            "validity": values.get("format_validity"),
        },
        "eos": {
            "stopped_count": sum(bool(record.get("stopped_on_eos")) for record in records),
            "rate": values.get("eos_rate"),
        },
        "truncation": {
            "completion_count": sum(bool(record.get("completion_truncated")) for record in records),
            "completion_rate": values.get("completion_truncation_rate"),
        },
        "per_register": {
            "format_valid_denominator": valid,
            "accuracy": values.get("per_register_accuracy"),
            "mean_absolute_error": values.get("per_register_mean_absolute_error"),
        },
        "answer_value_loss": {
            "value": values.get("answer_value_loss"),
            "format_valid_denominator": valid,
            "definition": values.get("answer_value_loss_definition"),
        },
        "by_final_operation": values.get("by_final_operation", {}),
        "prerequisites": values.get("prerequisites"),
        "first_errors": values.get("first_errors", []),
        "score_changes": values.get("score_changes", []),
    }


def compact_state(values):
    keys = (
        "complete", "packed_rows", "evaluated_rows", "batches", "counts", "metrics",
        "state_schedule", "pass_2_contract", "document_position_validation",
        "answer_loss_distinction", "evidence_boundary",
    )
    return {key: values[key] for key in keys if key in values}


def compact_answer_loss(values):
    return without_keys(values, {"records"})


def checkpoint_manifest_summaries(output, mode):
    directory = output / "arms" / mode / "checkpoints"
    result = []
    for path in sorted(directory.glob("*.manifest.json")):
        manifest = load(path)
        result.append({
            "step": manifest.get("step"),
            "checkpoint_name": manifest.get("checkpoint_name"),
            "sha256": manifest.get("sha256"),
            "bytes": manifest.get("bytes"),
            "consumed_loss_tokens": manifest.get("consumed_loss_tokens"),
            "state_hashes": manifest.get("state_hashes"),
            "manifest_path": str(path.relative_to(output)),
            "manifest_sha256": digest(path),
        })
    result.sort(key=lambda item: (item["step"] is None, item["step"], item["manifest_path"]))
    return result


def selected_baseline(split_evaluation, sources):
    interventions = split_evaluation["interventions"]
    ordinary = {
        record["example_id"]: record
        for record in [
            *split_evaluation["generation"]["records"],
            *interventions["supplemental_baseline_generation"]["records"],
        ]
    }
    selected = [ordinary[record["example_id"]] for record in interventions["records"]]
    return rescore_generations.summarize_program_study(selected, sources)


def reproduction_document(summary):
    return f"""# Weight-free varied-program evidence

Run `{summary['run_id']}` at training source commit `{summary['source_commit']}`.
This bundle preserves the completed study's structured synthetic programs, generations, metrics,
checkpoint identities and numerical reports. It contains no model weights, packed NPY arrays, or raw
external corpora. Checkpoint manifests identify local weights but do not contain them.

## Interpretation boundaries

- The initial-values split assigns scalar initial values in `[24,32)`. Those numbers can still occur
  elsewhere in training, so this is not global novelty of every integer.
- Unseen composition reserves adjacent multiply-then-add instructions. It does not guarantee a direct
  data dependency between those instructions in every example.
- Final-instruction writer suppression measures output sensitivity, not semantic faithfulness. No
  output effect does not establish state absence; redundancy and unaffected paths remain possible.
- State readout, generation exact answers, answer-token cross-entropy, intervention sensitivity, and
  full/cached reconstruction agreement are distinct evidence and retain separate denominators.

## Reproduce the independent whole-study rescore

Set `BUNDLE` to this directory. Every invocation uses a fresh `mktemp` directory and the independent
scorer preserved in this bundle, so the commands can be repeated without overwriting bundle evidence:

```bash
export BUNDLE=/path/to/evidence-final
WORK="$(mktemp -d)"
PYTHONDONTWRITEBYTECODE=1 python "$BUNDLE/code/rescore_generations.py" \\
  --study "$BUNDLE" --data "$BUNDLE/data" --output "$WORK/generation_diagnostics.json"
python -c 'import json, sys; a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2])); a.pop("study", None); b.pop("study", None); assert a == b, "independent rescore differs"; print("independent rescore matches")' \\
  "$BUNDLE/generation_diagnostics.json" "$WORK/generation_diagnostics.json"
```

`generation_diagnostics.json` is the whole-study source rescore.
`generation_diagnostics.reproduced.json` is the preserved bundle-only reproduction made during export;
only the top-level `study` path is permitted to differ. It includes every primary output, every
supplemental ordinary train baseline, and every intervened output for each available arm.
`summary.json` is compact; full generations, state paths, interventions, losses, journals, and
reconstruction reports remain under `arms/`. Primary, supplemental-baseline, selected-baseline, and
intervened denominators are not pooled.

To verify every payload hash:

```bash
(cd "$BUNDLE" && sha256sum -c SHA256SUMS)
```

`SHA256SUMS` covers every file except itself and `bundle_manifest.json`. The manifest records exact
hashed-file count, total-file count, and hashed payload bytes.
"""


def export(run, data, output, *, ledger=None, reservation=None):
    run = Path(run).resolve()
    data = Path(data).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite an existing evidence bundle: {output}")
    output.mkdir(parents=True)

    launch = load(run / "launch.json")
    status = load(run / "run_status.json")
    protocol = load(run / "protocol.json")
    if digest(run / "protocol.json") != (run / "protocol.sha256").read_text().strip():
        raise RuntimeError("run protocol SHA256 does not match its sidecar")
    if launch.get("data_manifest_sha256") not in (None, digest(data / "manifest.json")):
        raise RuntimeError("source data manifest differs from the launched data identity")
    reservation = reservation or status.get("reservation_id")
    if not reservation:
        raise RuntimeError("the study has no reservation identity")
    ledger_path = Path(ledger or status.get("ledger_after", {}).get("ledger", ""))
    if not ledger_path.is_file():
        raise FileNotFoundError(f"reservation ledger not found: {ledger_path}")

    for name in ("launch.json", "protocol.json", "protocol.sha256", "run_status.json"):
        copy(run / name, output / name)
    for name in ("manifest.json", "audit.json", "selections.json"):
        copy(data / name, output / "data" / name)
    for name in ("study_protocol.json", "study_protocol.sha256"):
        if (data / name).is_file():
            copy(data / name, output / "data" / name)
    for split in SPLITS:
        copy(data / "structured" / f"{split}.jsonl", output / "data" / "structured" / f"{split}.jsonl")

    scorer = Path(__file__).with_name("rescore_generations.py").resolve()
    copy(scorer, output / "code" / "rescore_generations.py")
    engineering_config = launch.get("engineering_config")
    if engineering_config:
        copy(engineering_config, output / "code" / "engineering.json")

    ledger_lines = []
    ledger_events = []
    for line in ledger_path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("reservation") == reservation:
            ledger_lines.append(line)
            ledger_events.append(event)
    if not ledger_events:
        raise RuntimeError(f"ledger contains no events for reservation {reservation}")
    (output / "reservation_ledger.jsonl").write_text("".join(line + "\n" for line in ledger_lines))

    identities = {
        "source_commit": launch.get("source_commit"),
        "code_identity": launch.get("code_identity"),
        "protocol_sha256": launch.get("protocol_sha256"),
        "data_manifest_sha256": launch.get("data_manifest_sha256"),
        "data_audit_sha256": digest(data / "audit.json"),
        "data_selections_sha256": digest(data / "selections.json"),
        "tokenizer_sha256": launch.get("tokenizer_sha256"),
        "replay_sha256": launch.get("replay_sha256"),
        "engineering_config_sha256": launch.get("engineering_config_sha256"),
        "source_files_sha256": protocol.get("source_files_sha256", {}),
        "independent_rescorer_sha256": digest(scorer),
        "exporter_sha256": digest(Path(__file__).resolve()),
    }
    write_json(output / "identities.json", identities)

    summary = {
        "format": SUMMARY_FORMAT,
        "run_id": launch.get("run_id", status.get("run_id")),
        "source_commit": launch.get("source_commit", status.get("source_commit")),
        "complete": status.get("complete"),
        "failure": status.get("failure"),
        "reservation_id": reservation,
        "charged_hours": status.get("charged_hours"),
        "authorized_hours": status.get("authorized_hours"),
        "allocation_limit_respected": status.get("allocation_limit_respected"),
        "native_usage": {
            "used_hours": status.get("ledger_after", {}).get("native_used_hours"),
            "authorized_additional_hours": next((
                event.get("native_authorization", {}).get("additional_hours")
                for event in ledger_events
                if event.get("event") == "start"
                and event.get("native_authorization", {}).get("additional_hours") is not None
            ), None),
            "remaining_hours": status.get("ledger_after", {}).get("native_remaining_hours"),
            "used_before_hours": status.get("ledger_before", {}).get("native_used_hours"),
        },
        "study_wall_seconds": status.get("wall_seconds"),
        "matched_endpoint": {
            "updates": protocol.get("max_updates"),
            "loss_tokens": protocol.get("max_loss_tokens"),
            "batch_size": protocol.get("batch_size"),
        },
        "split_roles": protocol.get("split_roles", {}),
        "identities": identities,
        "interpretation_boundaries": {
            "initial_values": "split role is scalar initial values in [24,32); those integers are not absent everywhere in training",
            "unseen_composition": "reserved adjacent multiply then add; direct dependency is not guaranteed in every example",
            "writer_suppression": "sensitivity is not semantic faithfulness; no effect is not state absence",
            "denominators": "primary, supplemental, selected-baseline, intervened, answer-token, state-readout, and per-register counts remain separate",
        },
        "arms": {},
    }

    arm_payloads = {}
    for mode in protocol.get("arms", []):
        source_arm = run / "arms" / mode
        destination_arm = output / "arms" / mode
        arm_status_path = source_arm / "status.json"
        if arm_status_path.is_file():
            arm_status = load(arm_status_path)
            copy(arm_status_path, destination_arm / "status.json")
        else:
            arm_status = status.get("arms", {}).get(mode, {
                "status": "not_run", "reason": "no per-arm status was written",
            })
            write_json(destination_arm / "status.json", arm_status)

        for name in (
            "train-result.json", "evaluation.json", "generation_diagnostics.json", "reconstruction.json",
            *(f"{split}_state_path.json" for split in SPLITS),
        ):
            if (source_arm / name).is_file():
                copy(source_arm / name, destination_arm / name)
        for name in ("run_config.json", "learning_curve.jsonl", "checkpoint_journal.jsonl", "recovery.jsonl"):
            if (source_arm / "train" / name).is_file():
                copy(source_arm / "train" / name, destination_arm / name)
        checkpoint_directory = source_arm / "train" / "checkpoints"
        if checkpoint_directory.is_dir():
            for path in sorted(checkpoint_directory.glob("*.manifest.json")):
                copy(path, destination_arm / "checkpoints" / path.name)

        arm_summary = {
            "status": arm_status,
            "endpoint": {
                "updates": arm_status.get("updates"),
                "loss_tokens": arm_status.get("consumed_loss_tokens"),
                "updates_match": arm_status.get("updates") == protocol.get("max_updates"),
                "loss_tokens_match": arm_status.get("consumed_loss_tokens") == protocol.get("max_loss_tokens"),
            },
            "checkpoint_manifests": checkpoint_manifest_summaries(output, mode),
        }
        result_path = source_arm / "train-result.json"
        evaluation_path = source_arm / "evaluation.json"
        result = load(result_path) if result_path.is_file() else None
        evaluation = load(evaluation_path) if evaluation_path.is_file() else None
        if result is not None:
            final_checkpoint = without_keys(result.get("final_checkpoint", {}), {"path", "manifest_path"})
            final_manifest = next((
                item for item in arm_summary["checkpoint_manifests"]
                if item["sha256"] == final_checkpoint.get("sha256")
                and item["step"] == final_checkpoint.get("step")
            ), None)
            arm_summary.update({
                "configuration": (load(source_arm / "train/run_config.json").get("model")
                                  if (source_arm / "train/run_config.json").is_file() else None),
                "final_checkpoint": {
                    **final_checkpoint,
                    "bundle_manifest": final_manifest,
                    "train_result_sha256": digest(result_path),
                },
                "loss": {
                    "train": learning_endpoints(source_arm / "train/learning_curve.jsonl", "train", mode),
                    "dev": learning_endpoints(source_arm / "train/learning_curve.jsonl", "dev", mode),
                    "final_teacher_forced_dev": without_keys(result.get("dev", {}), {"checkpoint"}),
                },
                "routing": {
                    key: result.get("dev", {}).get(key)
                    for key in ("route_usage", "route_targets", "route_loss", "balance_decisions", "balance_loss")
                },
                "time": {
                    **{
                        key: result[key]
                        for key in (
                            "training_seconds", "data_loading_seconds", "forward_seconds", "backward_seconds",
                            "optimizer_seconds", "checkpoint_seconds", "actual_loss_tokens_per_second",
                        )
                        if key in result
                    },
                    "train_and_evaluation_wall_seconds": arm_status.get("wall_seconds"),
                },
                "memory": {
                    "training": {key: value for key, value in result.items() if key.startswith("cuda_")},
                    "final_dev": {
                        key: value for key, value in result.get("dev", {}).items() if key.startswith("cuda_")
                    },
                },
            })
        reconstruction_path = source_arm / "reconstruction.json"
        if reconstruction_path.is_file():
            reconstruction = load(reconstruction_path)
            arm_summary["reconstruction"] = {
                "agreement": reconstruction.get("agreement"),
                "execution": reconstruction.get("execution"),
                "checkpoint": reconstruction.get("checkpoint"),
                "teacher_forced_agrees": reconstruction.get("teacher_forced", {}).get("agrees"),
                "teacher_forced_strict_allclose_agrees": reconstruction.get(
                    "teacher_forced", {}).get("strict_allclose_agrees"),
                "generation_agrees": reconstruction.get("generation", {}).get("agrees"),
                "report_sha256": digest(reconstruction_path),
            }
        if evaluation is not None and evaluation.get("format") == EVALUATION_FORMAT:
            arm_summary["evaluation_sha256"] = digest(evaluation_path)
            arm_summary["baselines"] = evaluation.get("baselines")
        arm_payloads[mode] = {
            "summary": arm_summary,
            "evaluation": evaluation,
            "source_arm": source_arm,
        }
        summary["arms"][mode] = arm_summary

    summary["matched_endpoints_verified"] = all(
        arm["endpoint"]["updates_match"] and arm["endpoint"]["loss_tokens_match"]
        for arm in summary["arms"].values()
    )
    summary["all_arms_completed"] = all(
        arm["status"].get("status") == "completed" for arm in summary["arms"].values()
    )

    diagnostics_path = output / "generation_diagnostics.json"
    diagnostics = rescore_generations.rescore(run, data, diagnostics_path)
    with tempfile.TemporaryDirectory(prefix="native-program-study-rescore-") as temporary:
        reproduced = Path(temporary) / "generation_diagnostics.json"
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([
            sys.executable, str(output / "code" / "rescore_generations.py"),
            "--study", str(output), "--data", str(output / "data"), "--output", str(reproduced),
        ], check=True, env=environment)
        reproduced_values = load(reproduced)
        if (without_keys(diagnostics, {"study"}) != without_keys(reproduced_values, {"study"})):
            raise RuntimeError("weight-free bundle does not reproduce its whole-study generation diagnostics")
        copy(reproduced, output / "generation_diagnostics.reproduced.json")

    sources = rescore_generations.source_records(data, SPLITS)
    for mode, diagnostic_arm in diagnostics.get("arms", {}).items():
        if mode not in arm_payloads:
            continue
        evaluation = arm_payloads[mode]["evaluation"]
        arm_summary = summary["arms"][mode]
        arm_summary["splits"] = {}
        arm_summary["independent_output_counts"] = {}
        for split in SPLITS:
            split_evaluation = evaluation["splits"][split]
            interventions = diagnostic_arm["interventions"][split]
            selection = split_evaluation.get("selection", {}).get("interventions", {})
            reported_interventions = split_evaluation["interventions"]
            answer_loss = split_evaluation["state_path"]["answer_token_cross_entropy"][split]
            arm_summary["splits"][split] = {
                "scope": split_evaluation.get("scope"),
                "primary_generation": compact_generation(diagnostic_arm["structured"][split]),
                "answer_token_cross_entropy": compact_answer_loss(answer_loss),
                "state_readout": compact_state(split_evaluation["state_path"]),
                "interventions": {
                    "selected_baseline": compact_generation(selected_baseline(split_evaluation, sources)),
                    "supplemental_baseline_generation": compact_generation(
                        diagnostic_arm["supplemental_baseline_generation"][split]),
                    "intervened": compact_generation(interventions["intervened"]),
                    "selection": {
                        "examples": selection.get("examples", interventions["examples"]),
                        "reused_primary_baselines": selection.get("reused_evaluation_baselines"),
                        "additional_supplemental_baselines": selection.get("additional_ordinary_baselines"),
                        "per_final_operation": selection.get("per_final_operation"),
                    },
                    "changed_counts": {
                        "intervened_examples": interventions["examples"],
                        "changed_completions": interventions["changed_completions"],
                        "changed_completion_rate": reported_interventions.get("changed_completion_rate"),
                        "changed_generated_suffixes": interventions["changed_generated_suffixes"],
                        "changed_generated_suffix_rate": reported_interventions.get("changed_generated_suffix_rate"),
                        "by_final_operation": interventions["by_final_operation"],
                        "score_changes": interventions["score_changes"],
                    },
                    "targeted_exactly_once": reported_interventions.get("targeted_exactly_once"),
                    "four_writer_updates_suppressed_per_example": reported_interventions.get(
                        "four_writer_updates_suppressed_per_example"),
                    "evidence_boundary": reported_interventions.get("evidence_boundary"),
                },
            }
            arm_summary["independent_output_counts"][split] = {
                "primary": diagnostic_arm["structured"][split]["examples"],
                "supplemental_baseline": diagnostic_arm["supplemental_baseline_generation"][split]["examples"],
                "selected_baseline": interventions["examples"],
                "intervened": interventions["examples"],
            }
        arm_summary["independent_diagnostics_sha256"] = digest(
            output / "arms" / mode / "generation_diagnostics.json"
        ) if (output / "arms" / mode / "generation_diagnostics.json").is_file() else None

    summary["independent_reproduction"] = {
        "whole_study": "generation_diagnostics.json",
        "bundle_only": "generation_diagnostics.reproduced.json",
        "comparison": "exact JSON equality after removing only the top-level study path",
        "verified": True,
        "target_validation": diagnostics.get("target_validation"),
        "whole_study_sha256": digest(diagnostics_path),
        "bundle_only_sha256": digest(output / "generation_diagnostics.reproduced.json"),
        "reservation_ledger_sha256": digest(output / "reservation_ledger.jsonl"),
        "reservation_ledger_events": len(ledger_events),
    }
    write_json(output / "summary.json", summary)
    (output / "REPRODUCE.md").write_text(reproduction_document(summary))

    forbidden = [path for path in output.rglob("*") if path.is_file() and path.suffix in FORBIDDEN_SUFFIXES]
    if forbidden:
        raise RuntimeError(f"forbidden weight or packed-array files in evidence bundle: {forbidden}")
    files = sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name not in ("SHA256SUMS", "bundle_manifest.json")
    )
    (output / "SHA256SUMS").write_text("".join(
        f"{digest(path)}  {path.relative_to(output)}\n" for path in files
    ))
    manifest = {
        "format": BUNDLE_FORMAT,
        "run_id": summary["run_id"],
        "source_commit": summary["source_commit"],
        "reservation_id": reservation,
        "complete": status.get("complete"),
        "files_hashed": len(files),
        "files_total": len(files) + 2,
        "payload_bytes": sum(path.stat().st_size for path in files),
        "contains_weights": False,
        "contains_packed_npy_arrays": False,
        "contains_raw_external_corpora": False,
        "hash_coverage": "every file except SHA256SUMS and bundle_manifest.json",
        "independent_bundle_rescore_verified": True,
        "summary_sha256": digest(output / "summary.json"),
        "generation_diagnostics_sha256": digest(diagnostics_path),
        "generation_diagnostics_reproduced_sha256": digest(output / "generation_diagnostics.reproduced.json"),
    }
    write_json(output / "bundle_manifest.json", manifest)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ledger")
    parser.add_argument("--reservation")
    args = parser.parse_args(argv)
    summary = export(
        args.run, args.data, args.output, ledger=args.ledger, reservation=args.reservation,
    )
    output = Path(args.output).resolve()
    print(json.dumps({
        "output": str(output),
        "summary": str(output / "summary.json"),
        "summary_sha256": digest(output / "summary.json"),
        "bundle_manifest": str(output / "bundle_manifest.json"),
        "bundle_manifest_sha256": digest(output / "bundle_manifest.json"),
        "sha256sums": str(output / "SHA256SUMS"),
        "sha256sums_sha256": digest(output / "SHA256SUMS"),
        "generation_diagnostics": str(output / "generation_diagnostics.json"),
        "generation_diagnostics_sha256": digest(output / "generation_diagnostics.json"),
        "generation_diagnostics_reproduced": str(output / "generation_diagnostics.reproduced.json"),
        "generation_diagnostics_reproduced_sha256": digest(output / "generation_diagnostics.reproduced.json"),
        "complete": summary["complete"],
        "charged_hours": summary["charged_hours"],
        "matched_endpoints_verified": summary["matched_endpoints_verified"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
