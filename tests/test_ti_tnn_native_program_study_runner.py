"""CPU checks for future program-study orchestration; no training or real reservation is opened."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "research" / "ti_tnn"))
from native import run_program_study as R  # noqa: E402
from native.structured import Instruction, execute, format_prompt  # noqa: E402


def protocol():
    return {
        "arms": ["dense", "untyped", "typed"],
        "max_updates": 3392,
        "max_loss_tokens": 5570640,
        "batch_size": 8,
        "seed": 20261112,
        "model_seed": 20261112,
        "data_seed": 20261113,
        "learning_rate": 0.0003,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "grad_clip": 1.0,
        "lambda_balance": 0.01,
        "lambda_route": 0.1,
        "lambda_state": 0.1,
        "warmup_fraction": 0.02,
        "final_lr_fraction": 0.1,
        "checkpoint_every": 500,
        "dev_every": 500,
        "dev_batches": 0,
        "max_new_tokens": 48,
        "position_policy": "document",
        "projected_total_hours": 0.8,
        "allocation_request_hours": 1.25,
    }


def write_root(tmp_path):
    root = tmp_path / "program-data"
    root.mkdir()
    counts = {"train": 4096, "dev": 128, "initial_values": 128,
              "unseen_composition": 128, "longer_length": 128}
    manifest = {
        "position_policy": "document",
        "counts": counts,
        "splits": {split: {"examples": count} for split, count in counts.items()},
        "train_epoch": {
            "batch_size": 8,
            "loss_tokens_per_epoch": 348165,
            "updates_per_epoch": 212,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (root / "audit.json").write_text('{"status": "synthetic"}\n')
    (root / "selections.json").write_text('{"status": "synthetic"}\n')
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("synthetic tokenizer\n")
    source = tmp_path / "repository/research/ti_tnn/native/run_program_study.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic frozen source\n")
    tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    frozen = {
        **protocol(),
        "source_files_sha256": {
            "run_program_study.py": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "data": {
            "root": str(root),
            "manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            "audit_sha256": hashlib.sha256((root / "audit.json").read_bytes()).hexdigest(),
            "selections_sha256": hashlib.sha256((root / "selections.json").read_bytes()).hexdigest(),
            "tokenizer_path": str(tokenizer),
            "tokenizer_file_sha256": tokenizer_sha256,
            "tokenizer": {"path": str(tokenizer), "sha256": tokenizer_sha256},
        },
    }
    protocol_bytes = (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode()
    (root / "study_protocol.json").write_bytes(protocol_bytes)
    (root / "study_protocol.sha256").write_text(hashlib.sha256(protocol_bytes).hexdigest() + "\n")
    return root


def mock_plan(monkeypatch, endpoint=(3392, 5570640)):
    monkeypatch.setattr(R, "PackedDataset", lambda root, split: object())
    monkeypatch.setattr(R.train, "make_sampler", lambda dataset, batch_size, seed: object())
    monkeypatch.setattr(R.train, "_planned_budget", lambda sampler, maximum: endpoint)


def training_result(arguments):
    checkpoint = Path(arguments.output) / "checkpoints" / "final-update-00003392.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"synthetic {arguments.mode} checkpoint".encode())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "format": "native-train-v1",
        "mode": arguments.mode,
        "step": 3392,
        "planned_updates": 3392,
        "consumed_loss_tokens": 5570640,
        "training_seconds": 10.0,
        "final_checkpoint": {"path": str(checkpoint), "step": 3392, "sha256": digest},
    }


def source_record(split, operation, index):
    initial = (index % 32, (index + 1) % 32, (index + 2) % 32, (index + 3) % 32)
    final = {
        "set": f"set r0 {index % 32}",
        "add": "add r0 r1",
        "multiply": "multiply r0 r1",
        "copy": "copy r0 r1",
    }[operation]
    program = [
        f"set r3 {R.SPLITS.index(split)}",
        f"set r2 {index % 32}",
        f"set r1 {index // 32}",
        final,
    ]
    instructions = [Instruction.parse(text) for text in program]
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


def write_evaluation_sources(root):
    root = Path(root)
    if all((root / "structured" / f"{split}.jsonl").is_file() for split in R.SPLITS):
        return
    structured = root / "structured"
    structured.mkdir(parents=True, exist_ok=True)
    operations = R.evaluate_program_study.OPERATIONS
    for split in R.SPLITS:
        records = [
            source_record(split, operation, index)
            for operation in operations
            for index in range(32)
        ]
        if split == "train":
            records.extend(
                source_record(split, operation, index)
                for operation in operations
                for index in range(32, 32 + (8 if operation in ("set", "add") else 7))
            )
        (structured / f"{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )


def source_records(root, split):
    path = Path(root) / "structured" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def saved_generation(record):
    token_id = len(record["example_id"]) % 251 + 1
    return {
        "example_id": record["example_id"],
        "completion_text": record["target_text"],
        "target_text": record["target_text"],
        "correct": True,
        "valid": True,
        "stopped_on_eos": True,
        "truncated": False,
        "prompt_truncated": False,
        "prompt_ids": [1],
        "completion_ids": [token_id],
        "generated_ids": [token_id],
    }


def intervention_records(records):
    ordinary = records[:R.EVALUATION_COUNT]
    supplemental = records[R.EVALUATION_COUNT:]
    selected = []
    for operation in R.evaluate_program_study.OPERATIONS:
        additional = [record for record in supplemental if record["program"][-1].startswith(operation + " ")]
        selected.extend(additional)
        reused = [record for record in ordinary if record["program"][-1].startswith(operation + " ")]
        selected.extend(reused[:R.evaluate_program_study.INTERVENTIONS_PER_OPERATION - len(additional)])
    return selected, supplemental


def saved_interventions(records):
    details = []
    for index, record in enumerate(records):
        baseline = saved_generation(record)
        intervened = saved_generation(record)
        if index == 0:
            intervened["completion_ids"] = [baseline["completion_ids"][0] + 1]
            intervened["generated_ids"] = [baseline["generated_ids"][0] + 1]
        details.append({
            "example_id": record["example_id"],
            "final_operation": record["program"][-1].split(" ", 1)[0],
            "intervened": intervened,
            "changed_completion": baseline["completion_ids"] != intervened["completion_ids"],
            "changed_generated_suffix": baseline["generated_ids"] != intervened["generated_ids"],
        })
    return details


def complete_evaluation(**kwargs):
    splits = {}
    for split in R.SPLITS:
        records = source_records(kwargs["data"], split)
        ordinary = records[:R.EVALUATION_COUNT]
        interventions, supplemental = intervention_records(records)
        generation_records = [saved_generation(record) for record in ordinary]
        supplemental_records = [saved_generation(record) for record in supplemental]
        intervention_details = saved_interventions(interventions)
        by_operation = {}
        for operation in R.evaluate_program_study.OPERATIONS:
            selected = [record for record in intervention_details if record["final_operation"] == operation]
            by_operation[operation] = {
                "examples": len(selected),
                "changed_completions": sum(record["changed_completion"] for record in selected),
                "changed_generated_suffixes": sum(record["changed_generated_suffix"] for record in selected),
            }
        splits[split] = {
            "selection": {
                "examples": R.EVALUATION_COUNT,
                "interventions": {
                    "examples": R.INTERVENTION_COUNT,
                    "additional_ordinary_baselines": len(supplemental),
                    "reused_evaluation_baselines": R.INTERVENTION_COUNT - len(supplemental),
                },
            },
            "generation": {
                "complete": True,
                "evaluated": R.EVALUATION_COUNT,
                "correct": R.EVALUATION_COUNT,
                "valid": R.EVALUATION_COUNT,
                "records": generation_records,
            },
            "interventions": {
                "selection_examples": R.INTERVENTION_COUNT,
                "records": intervention_details,
                "changed_completions": sum(record["changed_completion"] for record in intervention_details),
                "changed_generated_suffixes": sum(
                    record["changed_generated_suffix"] for record in intervention_details
                ),
                "by_final_operation": by_operation,
                "supplemental_baseline_generation": {
                    "complete": True,
                    "examples": len(supplemental),
                    "evaluated": len(supplemental),
                    "records": supplemental_records,
                },
            },
            "state_path": {
                "complete": True,
                "answer_token_cross_entropy": {
                    split: {"complete": True, "evaluated": R.EVALUATION_COUNT},
                },
            },
        }
    evaluation = {
        "format": R.evaluate_program_study.FORMAT,
        "checkpoint": {"sha256": kwargs["expected_sha256"], "step": 3392},
        "splits": splits,
    }
    kwargs["output"].write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
    return evaluation


def mock_successful_arms(monkeypatch, training=None, evaluation=None):
    training_calls = []
    evaluation_calls = []
    reconstruction_calls = []

    def run_training(arguments):
        training_calls.append(arguments)
        if training is not None:
            return training(arguments)
        return training_result(arguments)

    def evaluate(**kwargs):
        evaluation_calls.append(kwargs)
        write_evaluation_sources(kwargs["data"])
        if evaluation is not None:
            return evaluation(**kwargs)
        return complete_evaluation(**kwargs)

    def reconstruct_checkpoint(*args, **kwargs):
        reconstruction_calls.append((args, kwargs))
        return {
            "agreement": True,
            "execution": {
                "device": "cpu",
                "parameter_dtype": "torch.float32",
                "autocast_dtype": None,
            },
        }

    monkeypatch.setattr(R.train, "run_training", run_training)
    monkeypatch.setattr(R.evaluate_program_study, "evaluate", evaluate)
    monkeypatch.setattr(R.reconstruct, "load_replay_spec", lambda path: {"replay": str(path)})
    monkeypatch.setattr(R.reconstruct, "reconstruct_checkpoint", reconstruct_checkpoint)
    monkeypatch.setattr(R.reconstruct, "write_reconstruction_report", lambda path, report: "b" * 64)
    return training_calls, evaluation_calls, reconstruction_calls


def test_default_cli_dry_run_reads_live_ledger_without_artifacts_or_reservation(tmp_path, monkeypatch, capsys):
    root = write_root(tmp_path)
    before = {"used_hours": 24.0, "native_remaining_hours": 2.0, "reservations": []}
    usage_calls = []
    monkeypatch.setattr(R.ledger, "validate_shared_reservation_paths", lambda: (
        tmp_path / "ledger.jsonl", tmp_path / "gpu.lock"
    ))
    monkeypatch.setattr(R.ledger, "usage", lambda path: usage_calls.append(path) or before)
    mock_plan(monkeypatch)
    monkeypatch.setattr(R.ledger, "Reservation", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("dry-run attempted to construct a reservation")
    ))
    arguments = [
        "--data", str(root), "--repository", str(tmp_path / "repository"),
        "--python", sys.executable, "--tokenizer", str(tmp_path / "tokenizer.json"),
    ]
    assert R.main(arguments) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "dry_run"
    assert result["projection"]["sampler_planned_updates"] == 3392
    assert result["projection"]["sampler_planned_loss_tokens"] == 5570640
    assert result["projection"]["required_native_hours"] == pytest.approx(1.25)
    assert result["artifacts_created"] is False and result["reservation_created"] is False
    assert len(usage_calls) == 1
    assert not (root / "runs").exists()
    assert not (tmp_path / "ledger.jsonl").exists()
    assert not (tmp_path / "gpu.lock").exists()


@pytest.mark.parametrize("execute", [False, True], ids=["dry_run", "execute"])
@pytest.mark.parametrize(("failure", "match"), [
    ("protocol_sha", "study_protocol.sha256"),
    ("source", "frozen source file identity mismatch"),
    ("manifest", "frozen data identity mismatch: manifest.json"),
    ("audit", "frozen data identity mismatch: audit.json"),
    ("selections", "frozen data identity mismatch: selections.json"),
    ("tokenizer_path", "tokenizer path differs"),
    ("tokenizer_hash", "tokenizer hash differs"),
])
def test_frozen_identity_failures_precede_ledger_and_artifacts(
        tmp_path, monkeypatch, failure, match, execute):
    root = write_root(tmp_path)
    tokenizer = tmp_path / "tokenizer.json"
    supplied_tokenizer = tokenizer
    if failure == "protocol_sha":
        (root / "study_protocol.sha256").write_text("0" * 64 + "\n")
    elif failure == "source":
        source = tmp_path / "repository/research/ti_tnn/native/run_program_study.py"
        source.write_text("# changed after protocol freezing\n")
    elif failure in ("manifest", "audit", "selections"):
        with (root / f"{failure}.json").open("a") as handle:
            handle.write("\n")
    elif failure == "tokenizer_path":
        supplied_tokenizer = tmp_path / "other-tokenizer.json"
        supplied_tokenizer.write_bytes(tokenizer.read_bytes())
    elif failure == "tokenizer_hash":
        tokenizer.write_text("changed tokenizer\n")
    monkeypatch.setattr(R.ledger, "validate_shared_reservation_paths", lambda: (_ for _ in ()).throw(
        AssertionError("identity failure reached the ledger")
    ))
    monkeypatch.setattr(R, "planned_endpoint", lambda *args: (_ for _ in ()).throw(
        AssertionError("identity failure reached sampler planning")
    ))
    with pytest.raises(RuntimeError, match=match):
        R.run(
            root, tmp_path / "repository", supplied_tokenizer,
            execute=execute, authorize_hours=1.25 if execute else None,
        )
    assert not (root / "runs").exists()


@pytest.mark.parametrize(("execute", "hours"), [
    (True, None),
    (True, 1.24),
    (False, 1.25),
])
def test_authorization_requires_both_flags_and_an_exact_future_allocation(
        tmp_path, monkeypatch, execute, hours):
    root = write_root(tmp_path)
    monkeypatch.setattr(R.ledger, "validate_shared_reservation_paths", lambda: (_ for _ in ()).throw(
        AssertionError("authorization refusal should precede ledger access")
    ))
    monkeypatch.setattr(R.ledger, "Reservation", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("authorization refusal attempted a reservation")
    ))
    with pytest.raises(RuntimeError, match="authorize-hours"):
        R.run(root, tmp_path, tmp_path / "tokenizer.json", execute=execute, authorize_hours=hours)
    assert not (root / "runs").exists()


def test_sampler_must_derive_the_frozen_equal_update_and_token_endpoint(tmp_path, monkeypatch):
    mock_plan(monkeypatch, endpoint=(3392, 5570639))
    with pytest.raises(RuntimeError, match="not the required update/token endpoint"):
        R.planned_endpoint(tmp_path, protocol())


def test_projection_bounds_watchdog_shutdown_by_request_and_live_remaining():
    frozen = protocol()
    plan = R.projection(frozen, {"native_remaining_hours": 1.25})
    assert plan["maximum_watchdog_hours"] == pytest.approx(1.2)
    assert plan["required_native_hours"] == pytest.approx(1.25)
    assert plan["total_loss_tokens"] == 3 * 16 * 348165
    assert plan["total_updates"] == 3 * 3392
    assert plan["evaluation_examples_per_split"] == 128
    assert plan["intervention_examples_per_split"] == 32
    assert plan["fits_budget"]
    R.require_budget(plan)

    short_request = R.projection(
        {**frozen, "allocation_request_hours": 1.24}, {"native_remaining_hours": 2.0}
    )
    assert not short_request["fits_allocation_request"]
    with pytest.raises(RuntimeError, match="allocation_request_hours"):
        R.require_budget(short_request)
    short_live = R.projection(frozen, {"native_remaining_hours": 1.24})
    assert not short_live["fits_live_native_remaining"]
    with pytest.raises(RuntimeError, match="live native remaining"):
        R.require_budget(short_live)


def test_mocked_runner_validates_equal_endpoints_hashes_and_full_evaluation(tmp_path, monkeypatch):
    training_calls, evaluation_calls, reconstruction_calls = mock_successful_arms(monkeypatch)
    root = tmp_path / "data"
    run_dir = tmp_path / "attempt"
    statuses = R._execute(protocol(), root, run_dir, "tokenizer.json", "replay.json")
    assert [status["updates"] for status in statuses.values()] == [3392, 3392, 3392]
    assert [status["consumed_loss_tokens"] for status in statuses.values()] == [5570640] * 3
    assert all(status["status"] == "completed" for status in statuses.values())
    assert all(status["exact"] == {split: 128 for split in R.SPLITS} for status in statuses.values())
    assert len({call.output for call in training_calls}) == 3
    assert {call.mode for call in training_calls} == set(R.ARMS)
    for arguments in training_calls:
        assert arguments.device == "cuda"
        assert arguments.dev_batches == 0
        assert arguments.data_seed == 20261113
        assert arguments.model_seed == 20261112
        assert arguments.lambda_balance == 0.01
        assert arguments.lambda_route == 0.1
        assert arguments.lambda_state == 0.1
    for mode, call in zip(R.ARMS, evaluation_calls):
        assert call["data"] == root
        assert call["output"] == run_dir / "arms" / mode / "evaluation.json"
        assert call["max_new_tokens"] == 48
        assert call["expected_sha256"] == statuses[mode]["checkpoint"]["sha256"]
        assert call["checkpoint"] == statuses[mode]["checkpoint"]["path"]
    for args, kwargs in reconstruction_calls:
        assert kwargs["device"] == "cpu" and kwargs["autocast_dtype"] is None
        assert kwargs["expected_sha256"] == hashlib.sha256(Path(args[0]).read_bytes()).hexdigest()
    for mode in R.ARMS:
        arm = run_dir / "arms" / mode
        status_path = arm / "status.json"
        assert json.loads(status_path.read_text())["status"] == "completed"
        assert Path(statuses[mode]["checkpoint"]["path"]).is_file()
        evaluation_path = arm / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text())
        for split in R.SPLITS:
            selection = evaluation["splits"][split]["selection"]["interventions"]
            assert selection["additional_ordinary_baselines"] == (30 if split == "train" else 0)
            assert selection["reused_evaluation_baselines"] == (2 if split == "train" else 32)
        diagnostics_path = arm / "generation_diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text())
        assert diagnostics["evaluation_format"] == R.evaluate_program_study.FORMAT
        assert set(diagnostics["arms"]) == {mode}
        arm_diagnostics = diagnostics["arms"][mode]
        for split in R.SPLITS:
            metrics = arm_diagnostics["structured"][split]
            assert metrics["examples"] == metrics["exact_count"] == R.EVALUATION_COUNT
            assert len(metrics["records"]) == R.EVALUATION_COUNT
            assert metrics["score_changes"] == []
            assert metrics["prerequisites"]["complete"] == R.EVALUATION_COUNT
            assert {operation: values["examples"] for operation, values in metrics["by_final_operation"].items()} == {
                operation: R.evaluate_program_study.PER_OPERATION_COUNT
                for operation in R.evaluate_program_study.OPERATIONS
            }
            supplemental = arm_diagnostics["supplemental_baseline_generation"][split]
            assert supplemental["examples"] == (30 if split == "train" else 0)
            assert supplemental["prerequisites"]["complete"] == supplemental["examples"]
            intervention = arm_diagnostics["interventions"][split]
            assert intervention["examples"] == R.INTERVENTION_COUNT
            assert intervention["changed_completions"] == intervention["changed_generated_suffixes"] == 1
            assert intervention["score_changes"] == []
            assert intervention["intervened"]["prerequisites"]["complete"] == R.INTERVENTION_COUNT
            assert {operation: values["examples"] for operation, values in intervention["by_final_operation"].items()} == {
                operation: R.evaluate_program_study.INTERVENTIONS_PER_OPERATION
                for operation in R.evaluate_program_study.OPERATIONS
            }
        assert statuses[mode]["evaluation_sha256"] == hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
        assert statuses[mode]["generation_diagnostics_sha256"] == hashlib.sha256(
            diagnostics_path.read_bytes()
        ).hexdigest()
        assert statuses[mode]["reconstruction_sha256"] == "b" * 64


def test_ordinary_failure_is_persisted_and_other_arms_continue(tmp_path, monkeypatch):
    def fail_dense(arguments):
        if arguments.mode == "dense":
            raise RuntimeError("synthetic failure")
        return training_result(arguments)

    mock_successful_arms(monkeypatch, fail_dense)
    run_dir = tmp_path / "attempt"
    statuses = R._execute(protocol(), tmp_path / "data", run_dir, "tokenizer.json", "replay.json")
    assert statuses["dense"]["status"] == "failed"
    assert statuses["dense"]["stage"] == "training"
    assert "synthetic failure" in statuses["dense"]["error"]
    assert all(statuses[mode]["status"] == "completed" for mode in ("untyped", "typed"))
    assert json.loads((run_dir / "arms" / "dense" / "status.json").read_text()) == statuses["dense"]


def test_base_exception_preserves_statuses_and_aborts_remaining_arms(tmp_path, monkeypatch):
    class WatchdogStop(BaseException):
        pass

    def stop_untyped(arguments):
        if arguments.mode == "untyped":
            raise WatchdogStop("synthetic watchdog stop")
        return training_result(arguments)

    mock_successful_arms(monkeypatch, stop_untyped)
    run_dir = tmp_path / "attempt"
    with pytest.raises(WatchdogStop):
        R._execute(protocol(), tmp_path / "data", run_dir, "tokenizer.json", "replay.json")
    recovered = R._recover_statuses(run_dir, {
        mode: {"status": "not_run"} for mode in R.ARMS
    })
    assert recovered["dense"]["status"] == "completed"
    assert recovered["untyped"]["status"] == "interrupted"
    assert recovered["untyped"]["stage"] == "training"
    assert recovered["typed"]["status"] == "not_run"
    assert not (run_dir / "arms" / "typed").exists()


def test_checkpoint_hash_mismatch_fails_before_evaluation(tmp_path, monkeypatch):
    def wrong_hash(arguments):
        result = training_result(arguments)
        result["final_checkpoint"]["sha256"] = "0" * 64
        return result

    _, evaluation_calls, _ = mock_successful_arms(monkeypatch, wrong_hash)
    status = R._execute(protocol(), tmp_path / "data", tmp_path / "attempt", "tokenizer.json", "replay.json")
    assert all(item["status"] == "failed" for item in status.values())
    assert all("SHA256" in item["error"] for item in status.values())
    assert not evaluation_calls


@pytest.mark.parametrize("missing_file", [False, True], ids=["evaluation_raises", "evaluation_file_missing"])
def test_failed_or_missing_evaluation_never_invokes_independent_rescore(
        tmp_path, monkeypatch, missing_file):
    def evaluate(**kwargs):
        if not missing_file:
            raise RuntimeError("synthetic evaluation failure")
        result = complete_evaluation(**kwargs)
        kwargs["output"].unlink()
        return result

    _, _, reconstruction_calls = mock_successful_arms(monkeypatch, evaluation=evaluate)
    rescore_calls = []

    def forbidden_rescore(**kwargs):
        rescore_calls.append(kwargs)
        raise AssertionError("evaluation failure must not invoke independent rescoring")

    monkeypatch.setattr(R.rescore_generations, "rescore", forbidden_rescore)
    run_dir = tmp_path / "attempt"
    statuses = R._execute(protocol(), tmp_path / "data", run_dir, "tokenizer.json", "replay.json")
    assert not rescore_calls
    assert not reconstruction_calls
    for mode in R.ARMS:
        arm = run_dir / "arms" / mode
        assert statuses[mode]["status"] == "failed"
        assert statuses[mode]["stage"] == "evaluating"
        assert Path(statuses[mode]["checkpoint"]["path"]).is_file()
        assert not (arm / "evaluation.json").exists()
        assert not (arm / "generation_diagnostics.json").exists()
        assert json.loads((arm / "status.json").read_text()) == statuses[mode]


def test_independent_score_change_fails_arm_and_preserves_diagnostic(tmp_path, monkeypatch):
    def changed_score(**kwargs):
        evaluation = complete_evaluation(**kwargs)
        if kwargs["output"].parent.name == "dense":
            evaluation["splits"]["dev"]["generation"]["records"][0]["correct"] = False
            kwargs["output"].write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
        return evaluation

    _, _, reconstruction_calls = mock_successful_arms(monkeypatch, evaluation=changed_score)
    run_dir = tmp_path / "attempt"
    statuses = R._execute(protocol(), tmp_path / "data", run_dir, "tokenizer.json", "replay.json")
    dense = run_dir / "arms" / "dense"
    assert statuses["dense"]["status"] == "failed"
    assert statuses["dense"]["stage"] == "rescoring"
    assert "independent rescore changed scores" in statuses["dense"]["error"]
    assert Path(statuses["dense"]["checkpoint"]["path"]).is_file()
    assert (dense / "evaluation.json").is_file()
    diagnostics_path = dense / "generation_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    changes = diagnostics["arms"]["dense"]["structured"]["dev"]["score_changes"]
    assert len(changes) == 1
    assert changes[0]["metric"] == "exact_match"
    assert changes[0]["original"] is False and changes[0]["independent"] is True
    assert json.loads((dense / "status.json").read_text()) == statuses["dense"]
    assert all(statuses[mode]["status"] == "completed" for mode in ("untyped", "typed"))
    assert len(reconstruction_calls) == 2


def test_independent_record_count_mismatch_fails_arm(tmp_path, monkeypatch):
    def dropped_record(**kwargs):
        evaluation = complete_evaluation(**kwargs)
        if kwargs["output"].parent.name == "dense":
            evaluation["splits"]["train"]["generation"]["records"].pop()
            kwargs["output"].write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
        return evaluation

    mock_successful_arms(monkeypatch, evaluation=dropped_record)
    run_dir = tmp_path / "attempt"
    statuses = R._execute(protocol(), tmp_path / "data", run_dir, "tokenizer.json", "replay.json")
    assert statuses["dense"]["status"] == "failed"
    assert statuses["dense"]["stage"] == "rescoring"
    assert "train: incomplete 128-example independent generation rescore" in statuses["dense"]["error"]
    diagnostics = json.loads((run_dir / "arms" / "dense" / "generation_diagnostics.json").read_text())
    assert diagnostics["arms"]["dense"]["structured"]["train"]["examples"] == 127
    assert diagnostics["arms"]["dense"]["structured"]["train"]["score_changes"] == []
    assert all(statuses[mode]["status"] == "completed" for mode in ("untyped", "typed"))


def test_independent_changed_count_mismatch_fails_arm(tmp_path, monkeypatch):
    def changed_count(**kwargs):
        evaluation = complete_evaluation(**kwargs)
        if kwargs["output"].parent.name == "dense":
            evaluation["splits"]["dev"]["interventions"]["changed_completions"] += 1
            kwargs["output"].write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
        return evaluation

    mock_successful_arms(monkeypatch, evaluation=changed_count)
    run_dir = tmp_path / "attempt"
    statuses = R._execute(protocol(), tmp_path / "data", run_dir, "tokenizer.json", "replay.json")
    assert statuses["dense"]["status"] == "failed"
    assert statuses["dense"]["stage"] == "rescoring"
    assert "independent changed completion count differs" in statuses["dense"]["error"]
    diagnostics = json.loads((run_dir / "arms" / "dense" / "generation_diagnostics.json").read_text())
    intervention = diagnostics["arms"]["dense"]["interventions"]["dev"]
    assert intervention["changed_completions"] == 1
    assert intervention["score_changes"] == []
    assert all(statuses[mode]["status"] == "completed" for mode in ("untyped", "typed"))


@pytest.mark.parametrize("kind", ["primary", "supplemental", "intervened"])
def test_incomplete_prerequisites_fail_each_rescored_population(tmp_path, monkeypatch, kind):
    def truncated_prompt(**kwargs):
        evaluation = complete_evaluation(**kwargs)
        if kwargs["output"].parent.name == "dense":
            split = "train" if kind == "supplemental" else "dev"
            split_evaluation = evaluation["splits"][split]
            if kind == "primary":
                record = split_evaluation["generation"]["records"][0]
            elif kind == "supplemental":
                record = split_evaluation["interventions"]["supplemental_baseline_generation"]["records"][0]
            else:
                record = split_evaluation["interventions"]["records"][0]["intervened"]
            record["prompt_truncated"] = True
            kwargs["output"].write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
        return evaluation

    mock_successful_arms(monkeypatch, evaluation=truncated_prompt)
    run_dir = tmp_path / "attempt"
    statuses = R._execute(protocol(), tmp_path / "data", run_dir, "tokenizer.json", "replay.json")
    assert statuses["dense"]["status"] == "failed"
    assert statuses["dense"]["stage"] == "rescoring"
    assert "incomplete prerequisites" in statuses["dense"]["error"]
    diagnostics = json.loads((run_dir / "arms" / "dense" / "generation_diagnostics.json").read_text())
    arm = diagnostics["arms"]["dense"]
    if kind == "primary":
        summary = arm["structured"]["dev"]
    elif kind == "supplemental":
        summary = arm["supplemental_baseline_generation"]["train"]
    else:
        summary = arm["interventions"]["dev"]["intervened"]
    assert summary["prerequisites"]["complete"] == summary["examples"] - 1
    assert summary["prerequisites"]["truncated_or_incomplete"] == 1
    assert all(statuses[mode]["status"] == "completed" for mode in ("untyped", "typed"))


def test_execute_uses_one_inherited_owner_and_records_identities_and_charges(tmp_path, monkeypatch, capsys):
    root = write_root(tmp_path)
    protocol_bytes = (root / "study_protocol.json").read_bytes()
    manifest_bytes = (root / "manifest.json").read_bytes()
    replay = tmp_path / "ti_tnn_native" / "replay.json"
    replay.parent.mkdir()
    replay.write_text("synthetic replay\n")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("synthetic tokenizer\n")
    repository = tmp_path / "repository"
    config = repository / "research/ti_tnn/native/configs/engineering.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("{}\n")
    prior = root / "runs" / "prior" / "sentinel.txt"
    prior.parent.mkdir(parents=True)
    prior.write_text("preserve me\n")
    before = {"used_hours": 24.0, "native_remaining_hours": 2.0, "reservations": []}
    after = {"used_hours": 24.5, "native_remaining_hours": 1.5, "reservations": []}
    usage = iter((before, after))
    monkeypatch.setattr(R.ledger, "validate_shared_reservation_paths", lambda: (
        tmp_path / "ledger.jsonl", tmp_path / "gpu.lock"
    ))
    monkeypatch.setattr(R.ledger, "usage", lambda path: next(usage))
    monkeypatch.setattr(R, "_validate_runtime", lambda repository, python: None)
    monkeypatch.setattr(R, "_source_state", lambda repository: ("c" * 40, "d" * 64))
    monkeypatch.setattr(R.reconstruct, "load_replay_spec", lambda path: {"replay": str(path)})
    mock_plan(monkeypatch)
    owner = {"active": None}
    monkeypatch.setattr(R.ledger, "active", lambda: owner["active"])
    reservations = []

    class FakeReservation:
        def __init__(self, command, **kwargs):
            self.command = command
            self.kwargs = kwargs
            self.id = "reservation-1"
            self.checks = 0
            reservations.append(self)

        def __enter__(self):
            assert owner["active"] is None
            owner["active"] = self
            return self

        def __exit__(self, *args):
            owner["active"] = None
            return False

        def check(self):
            self.checks += 1

    monkeypatch.setattr(R.ledger, "Reservation", FakeReservation)

    def inherited_training(arguments):
        active = R.ledger.active()
        assert active is not None
        assert active.kwargs["extra"]["native_acquisition_run_id"] == "study-test"
        assert os.environ["TI_TNN_NATIVE_ACQUISITION_RUN_ID"] == "study-test"
        return training_result(arguments)

    mock_successful_arms(monkeypatch, inherited_training)
    assert R.run(
        root, repository, tokenizer, "study-test", execute=True, authorize_hours=1.25,
        python=sys.executable,
    ) == 0
    assert len(reservations) == 1
    reservation = reservations[0]
    assert reservation.kwargs["projected_hours"] == 0.8
    assert reservation.kwargs["extra"]["source_commit"] == "c" * 40
    assert reservation.kwargs["extra"]["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    assert reservation.kwargs["extra"]["data_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert reservation.checks == 3
    assert owner["active"] is None
    assert "TI_TNN_NATIVE_ACQUISITION_RUN_ID" not in os.environ
    run_dir = root / "runs" / "study-test"
    launch = json.loads((run_dir / "launch.json").read_text())
    status = json.loads((run_dir / "run_status.json").read_text())
    assert launch["source_commit"] == status["source_commit"] == "c" * 40
    assert launch["code_identity"] == status["code_identity"] == "d" * 64
    assert launch["protocol_sha256"] == status["protocol_sha256"] == hashlib.sha256(protocol_bytes).hexdigest()
    assert launch["data_manifest_sha256"] == status["data_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert (run_dir / "protocol.json").read_bytes() == protocol_bytes
    assert status["complete"] and status["charged_hours"] == pytest.approx(0.5)
    assert status["reservation_id"] == "reservation-1"
    assert set(status["checkpoints"]) == set(R.ARMS)
    assert all(values["updates"] == 3392 for values in status["arms"].values())
    assert prior.read_text() == "preserve me\n"
    assert (root / "study_protocol.json").read_bytes() == protocol_bytes
    assert (root / "manifest.json").read_bytes() == manifest_bytes
    assert json.loads(capsys.readouterr().out)["complete"] is True
