"""Export and independently reproduce the six-child continuation without weights."""
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
SUMMARY_FORMAT = "native-six-child-continuation-weight-free-summary-v1"
BUNDLE_FORMAT = "native-six-child-continuation-weight-free-evidence-v1"
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".safetensors", ".onnx"}
CHILD_FILES = (
    "run_config.json",
    "train-result.json",
    "evaluation.json",
    "generation_diagnostics.json",
    "reconstruction.json",
    "answer_diagnostic_manifest.json",
    "answer_diagnostics.json",
    "learning_curve.jsonl",
    "checkpoint_journal.jsonl",
    "status.json",
    *(f"{split}_state_path.json" for split in SPLITS),
)
TIME_KEYS = (
    "training_seconds",
    "data_loading_seconds",
    "forward_seconds",
    "backward_seconds",
    "optimizer_seconds",
    "checkpoint_seconds",
    "actual_exposure_tokens_per_second",
    "actual_loss_tokens_per_second",
    "actual_main_loss_tokens_per_second",
)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, payload, *, new=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if new else "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


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


def generation_with_operations(values, records, sources):
    result = compact_generation(values)
    by_operation = {}
    for operation in rescore_generations.OPERATIONS:
        selected = [
            record for record in records
            if sources[record["example_id"]]["program"][-1].split(" ", 1)[0] == operation
        ]
        operation_values = compact_generation(rescore_generations.summarize(selected, sources))
        operation_values.pop("by_final_operation")
        operation_values.pop("first_errors")
        operation_values.pop("score_changes")
        by_operation[operation] = operation_values
    result["by_final_operation"] = by_operation
    return result


def compact_state(values):
    keys = (
        "complete", "packed_rows", "evaluated_rows", "batches", "counts", "metrics",
        "state_schedule", "pass_2_contract", "document_position_validation",
        "answer_loss_distinction", "evidence_boundary",
    )
    return {key: values[key] for key in keys if key in values}


def compact_answer_loss(values):
    return without_keys(values, {"records"})


def compact_changed_registers(values, artifact):
    result = without_keys(values, {"failure_records"})
    failures = values.get("failure_records", [])
    result["failure_records"] = {
        "count": len(failures),
        "preserved_in": artifact,
    }
    return result


def selected_baseline_records(split_evaluation):
    interventions = split_evaluation["interventions"]
    ordinary = {
        record["example_id"]: record
        for record in [
            *split_evaluation["generation"]["records"],
            *interventions["supplemental_baseline_generation"]["records"],
        ]
    }
    selected = [ordinary[record["example_id"]] for record in interventions["records"]]
    return selected


def learning_summaries(path, run_id):
    events = {
        event: {"first": None, "last": None, "records": 0}
        for event in ("train", "dev", "checkpoint")
    }
    if Path(path).is_file():
        with Path(path).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                event = record.get("event")
                if event not in events:
                    continue
                values = events[event]
                if values["first"] is None:
                    values["first"] = record
                values["last"] = record
                values["records"] += 1
    return {
        event: {**values, "full_curve": f"children/{run_id}/learning_curve.jsonl"}
        for event, values in events.items()
    }


def checkpoint_manifest_summaries(child_output):
    result = []
    directory = child_output / "checkpoints"
    for path in sorted(directory.glob("*.manifest.json")):
        manifest = load(path)
        result.append({
            "step": manifest.get("step"),
            "checkpoint_name": manifest.get("checkpoint_name"),
            "sha256": manifest.get("sha256"),
            "bytes": manifest.get("bytes"),
            "consumed_loss_tokens": manifest.get("consumed_loss_tokens"),
            "state_hashes": manifest.get("state_hashes"),
            "manifest_path": str(path.relative_to(child_output.parent.parent)),
            "manifest_sha256": digest(path),
        })
    result.sort(key=lambda item: (item["step"] is None, item["step"], item["manifest_path"]))
    return result


def compact_reconstruction(values):
    teacher = values.get("teacher_forced", {})
    comparisons = {
        name: comparison for name, comparison in teacher.get("comparisons", {}).items()
        if comparison is not None
    }
    failures = [
        {"tensor": name, **comparison}
        for name, comparison in comparisons.items()
        if not comparison.get("agrees", False)
    ]
    return {
        "agreement": values.get("agreement"),
        "execution": values.get("execution"),
        "checkpoint": without_keys(values.get("checkpoint", {}), {"path"}),
        "teacher_forced_scale_policy": {
            "agrees": teacher.get("agrees"),
            "agreement_policy": teacher.get("agreement_policy"),
        },
        "teacher_forced_strict_allclose_agrees": teacher.get("strict_allclose_agrees"),
        "strict_allclose_failures": failures,
        "tensor_comparisons": comparisons,
        "generation_agrees": values.get("generation", {}).get("agrees"),
        "bitwise_replay_claimed": False,
        "interpretation": "scale-policy and generation agreement are not bitwise or strict-allclose replay",
    }


def rescore_children(study, data, output, children, evaluation_root):
    study = Path(study).resolve()
    data = Path(data).resolve()
    evaluation_root = Path(evaluation_root).resolve()
    output = Path(output)
    combined = {
        "format": "native-independent-generation-rescore-v1",
        "study": str(study),
        "evaluation_format": EVALUATION_FORMAT,
        "evaluation_path_semantics": "bundle-relative canonical evaluation artifacts",
        "evaluation_paths": {},
        "child_modes": {child["run_id"]: child.get("mode") for child in children},
        "arms": {},
        "skipped_children": [],
    }
    target_validation = None
    with tempfile.TemporaryDirectory(prefix="native-continuation-source-rescore-") as temporary:
        for child in children:
            run_id = child["run_id"]
            mode = child.get("mode")
            evaluation_path = evaluation_root / run_id / "evaluation.json"
            if not evaluation_path.is_file():
                combined["skipped_children"].append(run_id)
                continue
            child_output = Path(temporary) / f"{run_id}.json"
            values = rescore_generations.rescore(
                study,
                data,
                child_output,
                evaluation_paths={mode: evaluation_path},
            )
            if values.get("evaluation_format") != EVALUATION_FORMAT:
                raise RuntimeError(f"{run_id}: unexpected evaluation format")
            if target_validation is None:
                target_validation = values["target_validation"]
            elif values["target_validation"] != target_validation:
                raise RuntimeError(f"{run_id}: target validation differs between children")
            combined["arms"][run_id] = values["arms"][mode]
            combined["evaluation_paths"][run_id] = f"children/{run_id}/evaluation.json"
    if target_validation is None:
        target_validation = {"executed_programs": 0, "rejected_inconsistent_targets": []}
    combined["target_validation"] = target_validation
    write_json(output, combined, new=True)
    return combined


def rescore_bundle(bundle, output):
    bundle = Path(bundle).resolve()
    protocol = load(bundle / "continuation_protocol.json")
    children = protocol.get("children", [])
    return rescore_children(
        bundle,
        bundle / "data",
        output,
        children,
        bundle / "children",
    )


def comparison_values(left, right, *, lower_is_better=False):
    if left is None or right is None:
        return {
            "A": left,
            "B": right,
            "B_minus_A": None,
            "strictly_better": None,
            "no_regression": None,
        }
    improvement = left - right if lower_is_better else right - left
    return {
        "A": left,
        "B": right,
        "B_minus_A": right - left,
        "strictly_better": improvement > 0,
        "no_regression": improvement >= 0,
    }


def teacher_metric(child_summary, split, name):
    if not child_summary or split not in child_summary.get("splits", {}):
        return None
    return child_summary["splits"][split].get("teacher_forced_answer_ce", {}).get(name, {}).get(
        "cross_entropy_mean"
    )


def build_ab_comparisons(summary, children):
    by_pair = {}
    for child in children:
        by_pair.setdefault(child.get("pair_id", child.get("mode")), {})[child.get("arm")] = child["run_id"]
    comparisons = {}
    for pair_id, arms in by_pair.items():
        a_id = arms.get("A")
        b_id = arms.get("B")
        a = summary["children"].get(a_id)
        b = summary["children"].get(b_id)
        available = bool(a and b and all(split in a.get("splits", {}) and split in b.get("splits", {})
                                        for split in SPLITS))
        pair = {
            "A_run_id": a_id,
            "B_run_id": b_id,
            "rule_metric": "primary free-generation exact-answer count; higher is better",
            "available": available,
            "splits": {},
        }
        if available:
            for split in SPLITS:
                a_split = a["splits"][split]
                b_split = b["splits"][split]
                pair["splits"][split] = {
                    "exact_answers": comparison_values(
                        a_split["primary_generation"]["exact_answers"],
                        b_split["primary_generation"]["exact_answers"],
                    ),
                    "answer_eos_ce": comparison_values(
                        teacher_metric(a, split, "answer_eos"),
                        teacher_metric(b, split, "answer_eos"),
                        lower_is_better=True,
                    ),
                    "numeric_answer_value_ce": comparison_values(
                        teacher_metric(a, split, "numeric_answer_value"),
                        teacher_metric(b, split, "numeric_answer_value"),
                        lower_is_better=True,
                    ),
                    "formatting_eos_ce": comparison_values(
                        teacher_metric(a, split, "formatting_eos"),
                        teacher_metric(b, split, "formatting_eos"),
                        lower_is_better=True,
                    ),
                    "changed_register_micro_accuracy": comparison_values(
                        a_split["saved_generation_changed_registers"]["changed_registers"]["micro"]["accuracy"],
                        b_split["saved_generation_changed_registers"]["changed_registers"]["micro"]["accuracy"],
                    ),
                }
        pair["answer_eos_strictly_better_on_every_split"] = available and all(
            values["exact_answers"]["strictly_better"] for values in pair["splits"].values()
        )
        pair["no_exact_answer_regression"] = available and all(
            values["exact_answers"]["no_regression"] for values in pair["splits"].values()
        )
        pair["rule_outcome"] = (
            "answer_eos_passes_predeclared_rule"
            if pair["answer_eos_strictly_better_on_every_split"] and pair["no_exact_answer_regression"]
            else "retain_full_sequence"
        )
        comparisons[pair_id] = pair
    return comparisons


def reproduction_document(summary):
    return f"""# Weight-free six-child continuation evidence

Run `{summary['run_id']}` at training source commit `{summary['source_commit']}` with reservation
`{summary['reservation_id']}`. This is a bounded reused-development comparison, not confirmation and
not evidence of architecture superiority. Ordered updates, epochs, and full-label exposure match;
main-loss scored-token counts and compute do not. The predeclared exact-answer rule outcome is
`{summary['predeclared_rule']['outcome']}`; it does not select or alter a checkpoint.

The bundle preserves original structured programs, all primary/supplemental/intervened generations,
answer diagnostics and failure records, numerical reports, checkpoint identities and journals. It
contains no model weights, packed original NPY arrays, answer-mask NPY arrays, or raw external
corpora. The answer-mask companion manifest is retained without its arrays. Scale-policy replay
agreement and strict-allclose failures are both reported; bitwise replay is not claimed.

## Reproduce the independent whole-continuation rescore

Set `BUNDLE` to this directory. The command uses only the bundle, writes into a fresh `mktemp`
directory, and compares every field except the top-level `study` path:

```bash
export BUNDLE=/path/to/evidence-final
WORK="$(mktemp -d)"
PYTHONDONTWRITEBYTECODE=1 python "$BUNDLE/code/export_continuation.py" \\
  --rescore-bundle "$BUNDLE" --output "$WORK/generation_diagnostics.json"
python -c 'import json, sys; a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2])); a.pop("study", None); b.pop("study", None); assert a == b, "independent rescore differs"; print("independent rescore matches")' \\
  "$BUNDLE/generation_diagnostics.json" "$WORK/generation_diagnostics.json"
```

The exporter invokes the independent scorer with an explicit `evaluation_paths` entry for every
available custom child run ID. Canonical paths in the combined diagnostics are bundle-relative so
that only `study` differs on reproduction. Original per-child diagnostics retain their original
absolute paths byte-for-byte under `children/`. Both `generation_diagnostics.json` and the preserved
bundle-only `generation_diagnostics.reproduced.json` cover primary, supplemental ordinary, and
intervened outputs. Primary, supplemental, selected-baseline, intervened, answer-token, state-readout,
and per-register denominators are never pooled.

To verify every payload hash:

```bash
(cd "$BUNDLE" && sha256sum -c SHA256SUMS)
```

`SHA256SUMS` covers every file except itself and `bundle_manifest.json`. The bundle manifest records
exact hashed-file count, total-file count, hashed payload bytes, and summary/diagnostic hashes.
"""


def export(run, data, output, *, ledger=None, reservation=None, mask_manifest=None):
    run = Path(run).resolve()
    data = Path(data).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite an existing evidence bundle: {output}")
    output.mkdir(parents=True)

    launch = load(run / "launch.json")
    status = load(run / "run_status.json")
    protocol = load(run / "continuation_protocol.json")
    fork = load(run / "fork_manifest.json")
    protocol_path = run / "continuation_protocol.json"
    if digest(protocol_path) != (run / "continuation_protocol.sha256").read_text().split()[0]:
        raise RuntimeError("continuation protocol SHA256 does not match its sidecar")
    if launch.get("protocol_sha256") not in (None, digest(protocol_path)):
        raise RuntimeError("launch protocol identity differs from continuation_protocol.json")
    if launch.get("fork_manifest_sha256") not in (None, digest(run / "fork_manifest.json")):
        raise RuntimeError("launch fork identity differs from fork_manifest.json")
    data_manifest_sha256 = digest(data / "manifest.json")
    launched_data_hash = launch.get("data", {}).get("manifest_sha256")
    if launched_data_hash not in (None, data_manifest_sha256):
        raise RuntimeError("source data manifest differs from the launched data identity")
    if protocol.get("data", {}).get("manifest_sha256") not in (None, data_manifest_sha256):
        raise RuntimeError("source data manifest differs from the protocol data identity")
    if launch.get("source_commit") != status.get("source_commit"):
        raise RuntimeError("launch and run-status source commits differ")
    reservation = reservation or status.get("reservation_id")
    if not reservation:
        raise RuntimeError("the continuation has no reservation identity")
    ledger_path = Path(ledger or status.get("ledger_after", {}).get("ledger", ""))
    if not ledger_path.is_file():
        raise FileNotFoundError(f"reservation ledger not found: {ledger_path}")

    originals = []

    def preserve(source, relative_destination):
        source = Path(source)
        destination = output / relative_destination
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.suffix in FORBIDDEN_SUFFIXES:
            raise RuntimeError(f"refusing to export weight or packed-array artifact: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        source_hash = digest(source)
        if digest(destination) != source_hash:
            raise RuntimeError(f"copied artifact differs from source: {source}")
        originals.append({
            "source": str(source),
            "bundle_path": str(relative_destination),
            "bytes": source.stat().st_size,
            "sha256": source_hash,
        })

    for name in (
        "continuation_protocol.json", "continuation_protocol.sha256", "fork_manifest.json",
        "launch.json", "run_status.json",
    ):
        preserve(run / name, name)
    for name in ("manifest.json", "audit.json", "selections.json"):
        preserve(data / name, Path("data") / name)
    for name in ("study_protocol.json", "study_protocol.sha256"):
        if (data / name).is_file():
            preserve(data / name, Path("data") / name)
    for split in SPLITS:
        preserve(data / "structured" / f"{split}.jsonl", Path("data/structured") / f"{split}.jsonl")
    if (data / "structured" / "baselines.json").is_file():
        preserve(data / "structured" / "baselines.json", "data/structured/baselines.json")

    mask_path = Path(mask_manifest or protocol.get("masks", {}).get("manifest_path", ""))
    if not mask_path.is_file():
        raise FileNotFoundError(f"answer-mask companion manifest not found: {mask_path}")
    mask_hash = digest(mask_path)
    if protocol.get("masks", {}).get("manifest_sha256") not in (None, mask_hash):
        raise RuntimeError("answer-mask companion manifest differs from the protocol identity")
    preserve(mask_path, "answer_masks/manifest.json")

    scorer = Path(__file__).with_name("rescore_generations.py").resolve()
    exporter = Path(__file__).resolve()
    preserve(scorer, "code/rescore_generations.py")
    preserve(exporter, "code/export_continuation.py")

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
        "protocol_sha256": digest(protocol_path),
        "fork_manifest_sha256": digest(run / "fork_manifest.json"),
        "data_manifest_sha256": data_manifest_sha256,
        "data_audit_sha256": digest(data / "audit.json"),
        "data_selections_sha256": digest(data / "selections.json"),
        "answer_mask_manifest_sha256": mask_hash,
        "answer_mask_identity": protocol.get("masks", {}).get("identity", status.get("mask_identity")),
        "tokenizer_sha256": launch.get("data", {}).get("tokenizer_sha256"),
        "replay_sha256": launch.get("data", {}).get("replay_sha256"),
        "source_files_sha256": protocol.get("source_files_sha256", {}),
        "independent_rescorer_sha256": digest(scorer),
        "exporter_sha256": digest(exporter),
    }
    write_json(output / "identities.json", identities)

    children = [dict(child) for child in launch.get("children", protocol.get("children", []))]
    known = {child["run_id"] for child in children}
    for run_id, child_status in status.get("children", {}).items():
        if run_id not in known:
            children.append({
                "run_id": run_id,
                "child_id": run_id,
                "pair_id": child_status.get("mode"),
                "mode": child_status.get("mode"),
                "arm": "B" if child_status.get("loss_mode") == "answer_eos" else "A",
                "loss_mode": child_status.get("loss_mode"),
            })
    if not children:
        raise RuntimeError("continuation launch contains no children")

    summary = {
        "format": SUMMARY_FORMAT,
        "run_id": launch.get("continuation_study_id", status.get("continuation_study_id")),
        "source_commit": launch.get("source_commit"),
        "reservation_id": reservation,
        "complete": status.get("complete"),
        "failure": status.get("failure"),
        "failure_history": status.get("failure_history", []),
        "previous_status_history": status.get("previous_status_history", []),
        "scope": "bounded reused-development comparison, not confirmation",
        "architecture_superiority_inferred": False,
        "comparison_boundaries": {
            "development": "existing populations and frozen selections are reused; no new population or checkpoint selection",
            "matched": "ordered full-label exposure, complete epochs, and updates at fixed endpoints",
            "not_matched": "main-loss scored-token counts and compute",
            "denominators": "ordinary primary, supplemental ordinary, selected-baseline, and intervened outputs remain separate",
            "replay": "scale-policy agreement is retained alongside strict-allclose failures; bitwise replay is not claimed",
        },
        "identities": identities,
        "answer_masks": {
            "manifest": "answer_masks/manifest.json",
            "manifest_sha256": mask_hash,
            "identity": identities["answer_mask_identity"],
            "arrays_exported": False,
            "mask_use": protocol.get("masks", {}).get("mask_use"),
        },
        "endpoint_contract": protocol.get("endpoint"),
        "exposure_accounting": status.get("exposure_accounting", protocol.get("exposure_accounting")),
        "paired_compute_match": False,
        "children": {},
        "child_failures": {},
    }

    payloads = {}
    for child in children:
        run_id = child["run_id"]
        source_child = run / run_id
        destination_child = output / "children" / run_id
        child_status_path = source_child / "status.json"
        if child_status_path.is_file():
            child_status = load(child_status_path)
        else:
            child_status = status.get("children", {}).get(run_id, {
                "status": "not_run", "reason": "no child status was written",
            })
            write_json(destination_child / "status.json", child_status)
        for name in CHILD_FILES:
            path = source_child / name
            if path.is_file():
                preserve(path, Path("children") / run_id / name)
        if (source_child / "recovery.jsonl").is_file():
            preserve(source_child / "recovery.jsonl", Path("children") / run_id / "recovery.jsonl")
        if child_status.get("status") == "completed":
            missing = [name for name in CHILD_FILES if not (source_child / name).is_file()]
            if missing:
                raise RuntimeError(f"{run_id}: completed child is missing required evidence: {missing}")
        checkpoint_directory = source_child / "checkpoints"
        if checkpoint_directory.is_dir():
            for path in sorted(checkpoint_directory.glob("*.manifest.json")):
                preserve(path, Path("children") / run_id / "checkpoints" / path.name)

        config_path = source_child / "run_config.json"
        result_path = source_child / "train-result.json"
        evaluation_path = source_child / "evaluation.json"
        answer_path = source_child / "answer_diagnostics.json"
        config = load(config_path) if config_path.is_file() else {}
        result = load(result_path) if result_path.is_file() else None
        evaluation = load(evaluation_path) if evaluation_path.is_file() else None
        answer = load(answer_path) if answer_path.is_file() else None
        if evaluation is not None and evaluation.get("format") != EVALUATION_FORMAT:
            raise RuntimeError(f"{run_id}: unsupported official evaluation format")
        for status_key, path in (
            ("train_result", result_path),
            ("evaluation", evaluation_path),
            ("generation_diagnostics", source_child / "generation_diagnostics.json"),
            ("reconstruction", source_child / "reconstruction.json"),
            ("answer_diagnostics", answer_path),
            ("answer_diagnostic_manifest", source_child / "answer_diagnostic_manifest.json"),
        ):
            expected = child_status.get(f"{status_key}_sha256")
            if expected is not None and path.is_file() and digest(path) != expected:
                raise RuntimeError(f"{run_id}: child status {status_key} hash differs from the artifact")
        journal_hash = child_status.get("fork_to_final_journal_sha256")
        if journal_hash is not None and digest(source_child / "checkpoint_journal.jsonl") != journal_hash:
            raise RuntimeError(f"{run_id}: child status checkpoint-journal hash differs")

        target = child.get("target", protocol.get("endpoint", {}).get("target", {}))
        updates_per_epoch = protocol.get("endpoint", {}).get("updates_per_epoch")
        observed_updates = child_status.get("updates")
        endpoint = {
            "target_epochs": target.get("epochs"),
            "target_updates": target.get("updates"),
            "target_full_label_exposure_tokens": target.get("exposure_tokens"),
            "observed_updates": observed_updates,
            "observed_full_label_exposure_tokens": child_status.get("consumed_exposure_tokens"),
            "derived_completed_epochs": (
                observed_updates / updates_per_epoch
                if observed_updates is not None and updates_per_epoch else None
            ),
            "updates_match": observed_updates == target.get("updates"),
            "full_label_exposure_match": (
                child_status.get("consumed_exposure_tokens") == target.get("exposure_tokens")
            ),
            "additional_updates": child_status.get("new_updates"),
            "additional_full_label_exposure_tokens": child_status.get("additional_exposure_tokens"),
        }
        manifests = checkpoint_manifest_summaries(destination_child)
        final_checkpoint = without_keys((result or {}).get("final_checkpoint", {}), {"path", "manifest_path"})
        final_manifest = next((
            item for item in manifests
            if item["sha256"] == final_checkpoint.get("sha256") and item["step"] == final_checkpoint.get("step")
        ), None)
        fork_child = fork.get("children", {}).get(run_id, {})
        initial_checkpoint = without_keys(fork_child.get("initial_checkpoint", {}), {"path", "manifest_path"})
        parent_checkpoint = without_keys(config.get("fork", {}).get("parent", {}), {"checkpoint_path"})
        additional_main = child.get("additional_main_loss_tokens")
        if additional_main is None:
            additional_main = protocol.get("exposure_accounting", {}).get(
                "additional_main_loss_tokens", {}
            ).get(child.get("loss_mode"))
        if child.get("loss_mode") == "answer_eos":
            cumulative_main = protocol.get("exposure_accounting", {}).get(
                "answer_eos_whole_history_main_loss_tokens"
            )
        else:
            cumulative_main = protocol.get("exposure_accounting", {}).get(
                "full_sequence_target_main_loss_tokens"
            )
        if cumulative_main is None and additional_main is not None:
            cumulative_main = config.get("fork", {}).get("parent_exposure_tokens", 0) + additional_main
        if child_status.get("status") != "completed":
            additional_main = None
            cumulative_main = None

        child_summary = {
            "child_id": child.get("child_id", run_id),
            "pair_id": child.get("pair_id"),
            "arm": child.get("arm"),
            "mode": child.get("mode"),
            "loss_mode": child.get("loss_mode"),
            "status": child_status,
            "configuration": config.get("model"),
            "endpoint": endpoint,
            "reported_exact_answers": child_status.get("exact"),
            "resource_accounting": {
                "cumulative_full_label_exposure_tokens": child_status.get("consumed_exposure_tokens"),
                "additional_full_label_exposure_tokens": child_status.get("additional_exposure_tokens"),
                "additional_main_loss_scored_tokens": additional_main,
                "cumulative_main_loss_scored_tokens": cumulative_main,
                "reported_consumed_loss_tokens_stopping_counter": (result or {}).get("consumed_loss_tokens"),
                "new_updates": (result or {}).get("new_updates"),
                "processed_updates_this_invocation": (result or {}).get("processed_updates_this_invocation"),
            },
            "checkpoint_hashes": {
                "parent_sha256": parent_checkpoint.get("checkpoint_sha256"),
                "fork_sha256": initial_checkpoint.get("sha256"),
                "final_sha256": final_checkpoint.get("sha256"),
            },
            "parent_checkpoint": parent_checkpoint,
            "fork_checkpoint": initial_checkpoint,
            "final_checkpoint": {
                **final_checkpoint,
                "bundle_manifest": final_manifest,
                "train_result_sha256": digest(result_path) if result_path.is_file() else None,
            },
            "checkpoint_manifests": manifests,
            "splits": {},
        }
        if result is not None:
            child_summary.update({
                "losses": {
                    **learning_summaries(source_child / "learning_curve.jsonl", run_id),
                    "final_teacher_forced_dev": without_keys(result.get("dev", {}), {"checkpoint"}),
                },
                "routing": {
                    key: result.get("dev", {}).get(key)
                    for key in ("route_usage", "route_targets", "route_loss", "balance_decisions", "balance_loss")
                },
                "time": {
                    **{key: result[key] for key in TIME_KEYS if key in result},
                    "child_wall_seconds": child_status.get("wall_seconds"),
                    "measured_stage_seconds": child_status.get("measured_stage_seconds"),
                },
                "memory": {
                    "training": {key: value for key, value in result.items() if key.startswith("cuda_")},
                    "final_dev": {
                        key: value for key, value in result.get("dev", {}).items() if key.startswith("cuda_")
                    },
                },
            })
        reconstruction_path = source_child / "reconstruction.json"
        if reconstruction_path.is_file():
            child_summary["reconstruction"] = compact_reconstruction(load(reconstruction_path))
            child_summary["reconstruction"]["report_sha256"] = digest(reconstruction_path)
        if evaluation is not None:
            child_summary["evaluation_sha256"] = digest(evaluation_path)
        if answer is not None:
            child_summary["answer_diagnostics_sha256"] = digest(answer_path)
            child_summary["teacher_forced_complete"] = answer.get("teacher_forced", {}).get("complete")
            child_summary["answer_position_coverage"] = answer.get("answer_position_coverage")
        if child_status.get("status") != "completed" or child_status.get("error"):
            summary["child_failures"][run_id] = child_status
        summary["children"][run_id] = child_summary
        payloads[run_id] = {
            "child": child,
            "status": child_status,
            "config": config,
            "result": result,
            "evaluation": evaluation,
            "answer": answer,
        }

    diagnostics_path = output / "generation_diagnostics.json"
    diagnostics = rescore_children(run, data, diagnostics_path, children, run)
    with tempfile.TemporaryDirectory(prefix="native-continuation-bundle-rescore-") as temporary:
        reproduced = Path(temporary) / "generation_diagnostics.json"
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        subprocess.run([
            sys.executable, str(output / "code" / "export_continuation.py"),
            "--rescore-bundle", str(output), "--output", str(reproduced),
        ], check=True, env=environment)
        reproduced_values = load(reproduced)
        if without_keys(diagnostics, {"study"}) != without_keys(reproduced_values, {"study"}):
            raise RuntimeError("weight-free bundle does not reproduce continuation generation diagnostics")
        shutil.copy2(reproduced, output / "generation_diagnostics.reproduced.json")

    sources = rescore_generations.source_records(data, SPLITS)
    for run_id, payload in payloads.items():
        evaluation = payload["evaluation"]
        answer = payload["answer"]
        if evaluation is None or run_id not in diagnostics["arms"]:
            continue
        diagnostic_arm = diagnostics["arms"][run_id]
        child_summary = summary["children"][run_id]
        child_summary["independent_output_counts"] = {}
        for split in SPLITS:
            split_evaluation = evaluation["splits"][split]
            reported_interventions = split_evaluation["interventions"]
            interventions = diagnostic_arm["interventions"][split]
            selection = split_evaluation.get("selection", {}).get("interventions", {})
            primary_records = split_evaluation["generation"]["records"]
            supplemental_records = reported_interventions["supplemental_baseline_generation"]["records"]
            baseline_records = selected_baseline_records(split_evaluation)
            intervened_records = [record["intervened"] for record in reported_interventions["records"]]
            answer_loss = split_evaluation["state_path"]["answer_token_cross_entropy"][split]
            teacher_split = answer["teacher_forced"]["splits"][split] if answer is not None else {}
            teacher_metrics = teacher_split.get("metrics", {})
            saved_changed = answer["saved_generation_changed_registers"]["splits"][split]
            no_op = answer["no_op_baseline"]["splits"][split]
            artifact = f"children/{run_id}/answer_diagnostics.json"
            child_summary["splits"][split] = {
                "scope": split_evaluation.get("scope"),
                "reported_primary_exact_answers": payload["status"].get("exact", {}).get(split),
                "reported_exact_matches_independent": (
                    payload["status"].get("exact", {}).get(split)
                    == diagnostic_arm["structured"][split]["exact_count"]
                ),
                "primary_generation": generation_with_operations(
                    diagnostic_arm["structured"][split], primary_records, sources
                ),
                "selected_evaluation_answer_ce": compact_answer_loss(answer_loss),
                "teacher_forced_answer_ce": {
                    "complete": teacher_split.get("complete"),
                    "rows_evaluated": teacher_split.get("rows_evaluated"),
                    "rows_total": teacher_split.get("rows_total"),
                    "batches": teacher_split.get("batches"),
                    "full_lm_label_tokens_evaluated": teacher_split.get("full_lm_label_tokens_evaluated"),
                    "answer_eos": teacher_metrics.get("answer_mask"),
                    "numeric_answer_value": teacher_metrics.get("numeric_answer_value"),
                    "formatting_eos": teacher_metrics.get("formatting_eos_mask"),
                    "definition": "teacher-forced cross-entropy sums/means/counts over separate answer+EOS, numeric-value, and formatting/EOS masks",
                },
                "state_readout": compact_state(split_evaluation["state_path"]),
                "saved_generation_changed_registers": compact_changed_registers(saved_changed, artifact),
                "no_op_baseline": {
                    "operation": answer["no_op_baseline"].get("operation"),
                    "population": answer["no_op_baseline"].get("population"),
                    **compact_changed_registers(no_op, artifact),
                },
                "construction_baselines": {
                    name: evaluation.get("baselines", {}).get(name, {}).get(split)
                    for name in ("exact_lookup", "supplied_program_executor")
                },
                "interventions": {
                    "selected_baseline": generation_with_operations(
                        rescore_generations.summarize_program_study(baseline_records, sources),
                        baseline_records,
                        sources,
                    ),
                    "supplemental_baseline_generation": generation_with_operations(
                        diagnostic_arm["supplemental_baseline_generation"][split], supplemental_records, sources
                    ),
                    "intervened": generation_with_operations(
                        interventions["intervened"], intervened_records, sources
                    ),
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
                    "full_outputs": f"children/{run_id}/evaluation.json",
                },
            }
            child_summary["independent_output_counts"][split] = {
                "primary": diagnostic_arm["structured"][split]["examples"],
                "supplemental_baseline": diagnostic_arm["supplemental_baseline_generation"][split]["examples"],
                "selected_baseline": interventions["examples"],
                "intervened": interventions["examples"],
            }
        child_summary["original_independent_diagnostics_sha256"] = digest(
            output / "children" / run_id / "generation_diagnostics.json"
        )

    summary["matched_endpoints_verified"] = all(
        child["endpoint"]["updates_match"] and child["endpoint"]["full_label_exposure_match"]
        for child in summary["children"].values()
    )
    summary["all_children_completed"] = all(
        child["status"].get("status") == "completed" for child in summary["children"].values()
    )
    summary["all_initial_values_and_longer_length_exact_zero"] = all(
        split in child.get("splits", {})
        and child["splits"][split]["primary_generation"]["exact_answers"] == 0
        for child in summary["children"].values()
        for split in ("initial_values", "longer_length")
    )
    summary["ab_comparisons"] = build_ab_comparisons(summary, children)
    rule_complete = bool(summary["ab_comparisons"]) and all(
        pair["available"] for pair in summary["ab_comparisons"].values()
    )
    answer_eos_passes = rule_complete and all(
        pair["answer_eos_strictly_better_on_every_split"] and pair["no_exact_answer_regression"]
        for pair in summary["ab_comparisons"].values()
    )
    summary["predeclared_rule"] = {
        "rule": "retain full_sequence if answer_eos is not strictly better on every split with no regression",
        "metric": "primary free-generation exact-answer count on each fixed split",
        "evaluation_complete": rule_complete,
        "answer_eos_passes": answer_eos_passes,
        "outcome": "answer_eos_passes_predeclared_rule" if answer_eos_passes else "retain_full_sequence",
        "checkpoint_selection_performed": False,
        "architecture_superiority_inferred": False,
    }

    child_resources = [child["resource_accounting"] for child in summary["children"].values()]
    child_times = [child.get("time", {}) for child in summary["children"].values()]
    summary["cumulative_resource_use"] = {
        "children": len(children),
        "charged_hours": status.get("charged_hours"),
        "cumulative_continuation_charged_hours": status.get("cumulative_continuation_charged_hours"),
        "authorized_hours": status.get("authorized_hours"),
        "allocation_limit_respected": status.get("allocation_limit_respected"),
        "study_wall_seconds": status.get("wall_seconds"),
        "sum_child_wall_seconds": sum(value.get("child_wall_seconds", 0) for value in child_times),
        "sum_training_seconds": sum(value.get("training_seconds", 0) for value in child_times),
        "cumulative_full_label_exposure_tokens_across_children": sum(
            value.get("cumulative_full_label_exposure_tokens") or 0 for value in child_resources
        ),
        "additional_full_label_exposure_tokens_across_children": sum(
            value.get("additional_full_label_exposure_tokens") or 0 for value in child_resources
        ),
        "additional_main_loss_scored_tokens_across_children": sum(
            value.get("additional_main_loss_scored_tokens") or 0 for value in child_resources
        ),
        "cumulative_main_loss_scored_tokens_across_children": sum(
            value.get("cumulative_main_loss_scored_tokens") or 0 for value in child_resources
        ),
        "native_usage": {
            "used_hours": status.get("ledger_after", {}).get("native_used_hours"),
            "authorized_additional_hours": next((
                event.get("native_authorization", {}).get("additional_hours")
                for event in ledger_events
                if event.get("event") == "start"
                and event.get("native_authorization", {}).get("additional_hours") is not None
            ), None),
            "remaining_hours": status.get("ledger_after", {}).get("native_remaining_hours"),
            "used_before_hours": status.get("ledger_before", {}).get(
                "native_used_hours", launch.get("ledger_before", {}).get("native_used_hours")
            ),
        },
        "reservation_ledger_events": len(ledger_events),
        "reservation_ledger_sha256": digest(output / "reservation_ledger.jsonl"),
    }
    summary["independent_reproduction"] = {
        "whole_continuation": "generation_diagnostics.json",
        "bundle_only": "generation_diagnostics.reproduced.json",
        "comparison": "exact JSON equality after removing only the top-level study path",
        "verified": True,
        "explicit_evaluation_paths_per_custom_child": True,
        "covers_primary_supplemental_and_intervened": True,
        "target_validation": diagnostics.get("target_validation"),
        "whole_continuation_sha256": digest(diagnostics_path),
        "bundle_only_sha256": digest(output / "generation_diagnostics.reproduced.json"),
        "rescored_children": list(diagnostics["arms"]),
        "skipped_children": diagnostics["skipped_children"],
    }
    write_json(output / "original_artifacts.json", originals)
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
        "original_files_preserved": len(originals),
        "contains_weights": False,
        "contains_packed_npy_arrays": False,
        "contains_answer_mask_npy_arrays": False,
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
    parser.add_argument("--run")
    parser.add_argument("--data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ledger")
    parser.add_argument("--reservation")
    parser.add_argument("--mask-manifest")
    parser.add_argument("--rescore-bundle")
    args = parser.parse_args(argv)
    if args.rescore_bundle:
        result = rescore_bundle(args.rescore_bundle, args.output)
        print(json.dumps({
            "status": "complete",
            "children": list(result["arms"]),
            "output": args.output,
            "sha256": digest(args.output),
        }, sort_keys=True))
        return
    if not args.run or not args.data:
        parser.error("--run and --data are required unless --rescore-bundle is used")
    summary = export(
        args.run,
        args.data,
        args.output,
        ledger=args.ledger,
        reservation=args.reservation,
        mask_manifest=args.mask_manifest,
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
        "all_children_completed": summary["all_children_completed"],
        "matched_endpoints_verified": summary["matched_endpoints_verified"],
        "predeclared_rule_outcome": summary["predeclared_rule"]["outcome"],
        "charged_hours": summary["cumulative_resource_use"]["charged_hours"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
