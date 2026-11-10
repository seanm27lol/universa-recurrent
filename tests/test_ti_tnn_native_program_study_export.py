"""Self-contained checks for varied-program evidence export and reproduction."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from native.independent import rescore_generations
from native.independent.export_program_study import SPLITS, export
from native.structured import Instruction, execute, format_prompt


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_record(split, supplemental=False):
    programs = {
        "train": [Instruction("add", 0, src=1)] if supplemental else [Instruction("set", 0, value=1)],
        "dev": [Instruction("copy", 0, src=1)],
        "initial_values": [Instruction("set", 0, value=31)],
        "unseen_composition": [Instruction("multiply", 0, src=1), Instruction("add", 2, src=3)],
        "longer_length": [Instruction("set", 1, value=2), Instruction("add", 0, src=1)],
    }
    initial = (24, 25, 26, 27) if split == "initial_values" else (2, 3, 4, 5)
    if supplemental:
        initial = (6, 7, 8, 9)
    instructions = programs[split]
    program = [str(instruction) for instruction in instructions]
    target = execute(instructions, initial).target_text
    prompt = format_prompt(instructions, initial)
    identity = hashlib.sha256((prompt + target).encode()).hexdigest()[:20]
    return {
        "example_id": f"structured-{split}-{identity}",
        "split": split,
        "prompt": prompt,
        "target_text": target,
        "program": program,
        "initial": list(initial),
        "modulus": 32,
    }


def saved(record, token, *, intervened=False):
    return {
        "example_id": record["example_id"],
        "completion_text": record["target_text"],
        "target_text": record["target_text"],
        "completion_ids": [token],
        "generated_ids": [token],
        "correct": True,
        "valid": True,
        "stopped_on_eos": not intervened,
        "truncated": intervened,
        "prompt_truncated": False,
    }


def make_study(tmp_path, *, failed=False):
    run = tmp_path / "study-run"
    data = tmp_path / "study-data"
    records = {split: source_record(split) for split in SPLITS}
    supplemental = source_record("train", supplemental=True)
    for split in SPLITS:
        split_records = [records[split]] + ([supplemental] if split == "train" else [])
        (data / "structured").mkdir(parents=True, exist_ok=True)
        (data / "structured" / f"{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in split_records)
        )
    write(data / "manifest.json", {
        "format": "native-varied-program-study-data-v1",
        "counts": {split: 2 if split == "train" else 1 for split in SPLITS},
        "source_identity": {"external_corpora_used": False},
    })
    write(data / "audit.json", {"format": "fixture-audit", "targets_validated": True})
    write(data / "selections.json", {
        "evaluation_example_ids": {split: [records[split]["example_id"]] for split in SPLITS},
        "intervention_example_ids": {
            split: [(supplemental if split == "train" else records[split])["example_id"]]
            for split in SPLITS
        },
    })

    protocol = {
        "format": "native-program-study-protocol-v1",
        "arms": ["dense", "untyped", "typed"] if failed else ["dense"],
        "max_updates": 1,
        "max_loss_tokens": 10,
        "batch_size": 1,
        "split_roles": {
            "initial_values": "scalar initial values in [24,32)",
            "unseen_composition": "adjacent multiply then add without a dependency guarantee",
        },
        "source_files_sha256": {"independent/rescore_generations.py": "fixture-source-identity"},
    }
    write(run / "protocol.json", protocol)
    protocol_sha = sha256(run / "protocol.json")
    (run / "protocol.sha256").write_text(protocol_sha + "\n")
    write(data / "study_protocol.json", protocol)
    (data / "study_protocol.sha256").write_text(sha256(data / "study_protocol.json") + "\n")
    engineering = tmp_path / "engineering.json"
    write(engineering, {"model": {"context_length": 256}})

    ledger = tmp_path / "ledger.jsonl"
    reservation = "reservation-fixture"
    ledger.write_text("".join(json.dumps(event) + "\n" for event in [
        {
            "event": "start", "reservation": reservation,
            "native_authorization": {"additional_hours": 12.0},
            "extra": {"run_id": "fixture-run"},
        },
        {"event": "heartbeat", "reservation": reservation, "elapsed": 1.0},
        {"event": "stop", "reservation": reservation, "hours": 0.25},
        {"event": "start", "reservation": "unrelated-reservation"},
    ]))
    write(run / "launch.json", {
        "format": "native-program-study-launch-v1",
        "run_id": "fixture-run",
        "source_commit": "a" * 40,
        "code_identity": "code-fixture",
        "protocol_sha256": protocol_sha,
        "data_manifest_sha256": sha256(data / "manifest.json"),
        "tokenizer_sha256": "tokenizer-fixture",
        "replay_sha256": "replay-fixture",
        "engineering_config": str(engineering),
        "engineering_config_sha256": sha256(engineering),
        "ledger_before": {"ledger": str(ledger)},
    })
    arm_status = {
        "status": "completed", "updates": 1, "consumed_loss_tokens": 10,
        "wall_seconds": 2.0, "train_seconds": 1.0,
    }
    statuses = {"dense": arm_status}
    if failed:
        statuses.update({
            "untyped": {"status": "failed", "error": "synthetic failure"},
            "typed": {"status": "not_run"},
        })
    write(run / "run_status.json", {
        "format": "native-program-study-run-status-v1",
        "run_id": "fixture-run",
        "source_commit": "a" * 40,
        "complete": not failed,
        "failure": {"error": "synthetic failure"} if failed else None,
        "reservation_id": reservation,
        "charged_hours": 0.25,
        "authorized_hours": 1.0,
        "allocation_limit_respected": True,
        "wall_seconds": 2.0,
        "arms": statuses,
        "ledger_before": {"native_used_hours": 0.5},
        "ledger_after": {
            "ledger": str(ledger), "native_used_hours": 0.75, "native_remaining_hours": 11.25,
        },
    })

    arm = run / "arms" / "dense"
    write(arm / "status.json", arm_status)
    checkpoint_sha = "b" * 64
    initial_sha = "c" * 64
    checkpoint_dir = arm / "train" / "checkpoints"
    final_manifest = checkpoint_dir / f"final-update-00000001.pt.{checkpoint_sha}.manifest.json"
    initial_manifest = checkpoint_dir / f"update-00000000.pt.{initial_sha}.manifest.json"
    write(initial_manifest, {
        "format": "native-checkpoint-v1", "checkpoint_name": "update-00000000.pt",
        "step": 0, "sha256": initial_sha, "bytes": 100, "consumed_loss_tokens": 0,
    })
    write(final_manifest, {
        "format": "native-checkpoint-v1", "checkpoint_name": "final-update-00000001.pt",
        "step": 1, "sha256": checkpoint_sha, "bytes": 200, "consumed_loss_tokens": 10,
        "state_hashes": {"model": "model-hash"},
    })
    final_checkpoint = {
        "path": str(checkpoint_dir / "final-update-00000001.pt"),
        "manifest_path": str(final_manifest), "step": 1, "sha256": checkpoint_sha,
        "bytes": 200, "consumed_loss_tokens": 10,
    }
    write(arm / "train-result.json", {
        "format": "native-train-v1", "step": 1, "consumed_loss_tokens": 10,
        "final_checkpoint": final_checkpoint,
        "training_seconds": 1.0, "data_loading_seconds": 0.1, "forward_seconds": 0.2,
        "backward_seconds": 0.3, "optimizer_seconds": 0.2, "checkpoint_seconds": 0.1,
        "actual_loss_tokens_per_second": 10.0,
        "cuda_peak_allocated_bytes": 1000, "cuda_peak_reserved_bytes": 2000,
        "dev": {
            "loss": 0.5, "objective_loss": 0.7, "route_usage": [0, 1],
            "route_targets": 2, "route_loss": 0.1, "balance_decisions": 2, "balance_loss": 0.1,
            "cuda_peak_allocated_bytes": 1000, "checkpoint": final_checkpoint,
        },
    })
    write(arm / "train" / "run_config.json", {
        "model": {"mode": "dense", "n_passes": 4}, "training": {"max_loss_tokens": 10},
    })
    (arm / "train" / "learning_curve.jsonl").write_text("".join(json.dumps(event) + "\n" for event in [
        {"event": "train", "step": 0, "loss": 2.0},
        {"event": "train", "step": 1, "loss": 1.0},
        {"event": "dev", "step": 1, "loss": 0.5},
    ]))
    (arm / "train" / "checkpoint_journal.jsonl").write_text("".join(json.dumps(event) + "\n" for event in [
        {"event": "checkpoint", "step": 0, "path": "checkpoints/update-00000000.pt", "sha256": initial_sha},
        {"event": "batch", "step": 1, "loss_tokens": 10},
        {"event": "checkpoint", "step": 1, "path": "checkpoints/final-update-00000001.pt", "sha256": checkpoint_sha},
    ]))

    splits = {}
    for index, split in enumerate(SPLITS, 1):
        primary = saved(records[split], index)
        supplemental_records = [saved(supplemental, 10)] if split == "train" else []
        target = supplemental if split == "train" else records[split]
        baseline = supplemental_records[0] if split == "train" else primary
        intervened = saved(target, 11 if split == "train" else index, intervened=split == "train")
        changed = split == "train"
        intervention = {
            "example_id": target["example_id"],
            "baseline": {
                "correct": True, "valid": True, "stopped_on_eos": True, "truncated": False,
            },
            "intervened": intervened,
            "changed_completion": changed,
            "changed_generated_suffix": changed,
        }
        state = {
            "format": "native-state-path-v1", "complete": True,
            "packed_rows": 1, "evaluated_rows": 1, "batches": 1,
            "counts": {"state_targets": 4},
            "metrics": {"pass_0": {"correct": 2, "count": 4, "accuracy": 0.5}},
            "state_schedule": "before,after,after,after",
            "pass_2_contract": "fixture contract",
            "document_position_validation": {"valid": True},
            "answer_loss_distinction": "CE is not register error",
            "evidence_boundary": "readout is not causal semantic use",
            "answer_token_cross_entropy": {
                split: {
                    "examples": 1, "evaluated": 1, "source_examples": 2 if split == "train" else 1,
                    "complete": True, "sum": 0.25, "count": 17, "mean": 0.25 / 17,
                    "records": [{"example_id": records[split]["example_id"], "mean": 0.25 / 17}],
                },
            },
        }
        splits[split] = {
            "scope": "fixture split",
            "selection": {
                "examples": 1,
                "interventions": {
                    "examples": 1,
                    "reused_evaluation_baselines": 0 if split == "train" else 1,
                    "additional_ordinary_baselines": 1 if split == "train" else 0,
                    "per_final_operation": {target["program"][-1].split(" ", 1)[0]: 1},
                },
            },
            "generation": {
                "complete": True, "examples": 1, "evaluated": 1, "correct": 1,
                "records": [primary],
            },
            "interventions": {
                "selection_examples": 1,
                "supplemental_baseline_generation": {
                    "complete": bool(supplemental_records), "examples": len(supplemental_records),
                    "evaluated": len(supplemental_records), "records": supplemental_records,
                },
                "records": [intervention],
                "changed_completions": int(changed), "changed_completion_rate": float(changed),
                "changed_generated_suffixes": int(changed), "changed_generated_suffix_rate": float(changed),
                "targeted_exactly_once": 1, "four_writer_updates_suppressed_per_example": 1,
                "evidence_boundary": "sensitivity is not semantic faithfulness; no effect is not state absence",
            },
            "state_path": state,
        }
        write(arm / f"{split}_state_path.json", state)
    write(arm / "evaluation.json", {
        "format": "native-program-study-evaluation-v1",
        "checkpoint": {"sha256": checkpoint_sha, "step": 1, "config": {"model": {"mode": "dense"}}},
        "splits": splits,
        "baselines": {
            "exact_lookup": {split: {"examples": 1, "correct": int(split == "train")} for split in SPLITS},
            "supplied_program_executor": {split: {"examples": 1, "correct": 1} for split in SPLITS},
        },
    })
    write(arm / "reconstruction.json", {
        "format": "native-reconstruction-report-v1", "agreement": True,
        "execution": {"device": "cpu", "parameter_dtype": "torch.float32", "autocast_dtype": None},
        "checkpoint": {"sha256": checkpoint_sha, "step": 1},
        "teacher_forced": {"agrees": True, "strict_allclose_agrees": True},
        "generation": {"agrees": True},
    })
    rescore_generations.rescore(run, data, arm / "generation_diagnostics.json", modes=("dense",))

    if failed:
        write(run / "arms" / "untyped" / "status.json", {
            "status": "failed", "stage": "reconstructing",
            "error": "synthetic failure", "traceback": "fixture traceback",
        })
        write(run / "arms" / "untyped" / "train" / "run_config.json", {
            "model": {"mode": "untyped"}, "partial": True,
        })
    return run, data, output_path(tmp_path), ledger, engineering


def output_path(tmp_path):
    return tmp_path / "study-run" / "evidence-final"


def source_hashes(run, data, ledger, engineering):
    paths = [ledger, engineering]
    paths.extend(path for root in (run, data) for path in root.rglob("*") if path.is_file())
    return {path: sha256(path) for path in paths}


def without_study(path):
    value = json.loads(Path(path).read_text())
    value.pop("study")
    return value


def test_program_study_export_is_weight_free_and_bundle_reproducible(tmp_path):
    run, data, output, ledger, engineering = make_study(tmp_path)
    original = source_hashes(run, data, ledger, engineering)
    summary = export(run, data, output)
    assert summary["complete"]
    assert summary["matched_endpoints_verified"]
    assert summary["all_arms_completed"]
    assert summary["charged_hours"] == 0.25
    assert summary["native_usage"] == {
        "used_hours": 0.75, "authorized_additional_hours": 12.0,
        "remaining_hours": 11.25, "used_before_hours": 0.5,
    }
    assert all(sha256(path) == expected for path, expected in original.items())

    diagnostics = json.loads((output / "generation_diagnostics.json").read_text())
    assert diagnostics["evaluation_format"] == "native-program-study-evaluation-v1"
    assert diagnostics["target_validation"]["executed_programs"] == 6
    assert diagnostics["arms"]["dense"]["supplemental_baseline_generation"]["train"]["examples"] == 1
    assert diagnostics["arms"]["dense"]["interventions"]["train"]["changed_completions"] == 1
    assert diagnostics["arms"]["dense"]["interventions"]["train"]["intervened"]["exact_count"] == 1
    assert without_study(output / "generation_diagnostics.reproduced.json") == without_study(
        output / "generation_diagnostics.json"
    )

    arm = summary["arms"]["dense"]
    assert len(arm["checkpoint_manifests"]) == 2
    assert arm["final_checkpoint"]["sha256"] == "b" * 64
    assert arm["reconstruction"]["agreement"]
    assert arm["loss"]["train"]["last"]["loss"] == 1.0
    assert arm["routing"]["route_usage"] == [0, 1]
    assert arm["memory"]["training"]["cuda_peak_allocated_bytes"] == 1000
    for split in SPLITS:
        values = arm["splits"][split]
        assert values["primary_generation"]["examples"] == 1
        assert values["primary_generation"]["exact_answers"] == 1
        assert values["answer_token_cross_entropy"]["count"] == 17
        assert "records" not in values["answer_token_cross_entropy"]
        assert values["state_readout"]["metrics"]["pass_0"]["count"] == 4
        assert values["primary_generation"]["per_register"]["format_valid_denominator"] == 1
        assert set(values["primary_generation"]["by_final_operation"]) == {"set", "add", "multiply", "copy"}
        interventions = values["interventions"]
        assert interventions["selected_baseline"]["examples"] == 1
        assert interventions["intervened"]["examples"] == 1
        assert interventions["changed_counts"]["intervened_examples"] == 1
    train_interventions = arm["splits"]["train"]["interventions"]
    assert train_interventions["supplemental_baseline_generation"]["examples"] == 1
    assert train_interventions["selection"]["reused_primary_baselines"] == 0
    assert train_interventions["selection"]["additional_supplemental_baselines"] == 1
    assert train_interventions["changed_counts"]["changed_completions"] == 1
    assert arm["splits"]["dev"]["interventions"]["changed_counts"]["changed_completions"] == 0

    for split in SPLITS:
        assert (output / "data" / "structured" / f"{split}.jsonl").read_bytes() == (
            data / "structured" / f"{split}.jsonl"
        ).read_bytes()
        assert (output / "arms" / "dense" / f"{split}_state_path.json").is_file()
    assert (output / "arms" / "dense" / "generation_diagnostics.json").is_file()
    assert not [path for path in output.rglob("*") if path.suffix in {".pt", ".npy", ".npz"}]
    ledger_events = [json.loads(line) for line in (output / "reservation_ledger.jsonl").read_text().splitlines()]
    assert len(ledger_events) == 3
    assert {event["reservation"] for event in ledger_events} == {"reservation-fixture"}
    assert "mktemp -d" in (output / "REPRODUCE.md").read_text()
    assert 'a.pop("study", None)' in (output / "REPRODUCE.md").read_text()

    repeated = tmp_path / "fresh-reproduction" / "generation_diagnostics.json"
    subprocess.run([
        sys.executable, str(output / "code" / "rescore_generations.py"),
        "--study", str(output), "--data", str(output / "data"), "--output", str(repeated),
    ], check=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    assert without_study(repeated) == without_study(output / "generation_diagnostics.json")

    hashed = {}
    for line in (output / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        hashed[relative] = expected
        assert sha256(output / relative) == expected
    actual = {
        str(path.relative_to(output)) for path in output.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "bundle_manifest.json"}
    }
    assert set(hashed) == actual
    manifest = json.loads((output / "bundle_manifest.json").read_text())
    assert manifest["files_hashed"] == len(actual)
    assert manifest["files_total"] == len(actual) + 2
    assert manifest["payload_bytes"] == sum((output / name).stat().st_size for name in actual)
    assert manifest["summary_sha256"] == sha256(output / "summary.json")
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        export(run, data, output)


def test_failed_and_unstarted_arms_remain_in_export_and_reproduction(tmp_path):
    run, data, output, _, _ = make_study(tmp_path, failed=True)
    summary = export(run, data, output)
    assert not summary["complete"]
    assert summary["failure"] == {"error": "synthetic failure"}
    assert not summary["matched_endpoints_verified"]
    assert not summary["all_arms_completed"]
    assert summary["arms"]["untyped"]["status"]["error"] == "synthetic failure"
    assert summary["arms"]["untyped"]["status"]["traceback"] == "fixture traceback"
    assert summary["arms"]["typed"]["status"]["status"] == "not_run"
    assert (output / "arms" / "untyped" / "run_config.json").is_file()
    assert (output / "arms" / "typed" / "status.json").is_file()
    assert summary["independent_reproduction"]["verified"]
    diagnostics = json.loads((output / "generation_diagnostics.json").read_text())
    assert set(diagnostics["arms"]) == {"dense"}
    assert without_study(output / "generation_diagnostics.reproduced.json") == without_study(
        output / "generation_diagnostics.json"
    )
