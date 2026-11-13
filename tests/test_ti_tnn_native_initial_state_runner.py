"""CPU fixtures for orchestration contracts; no model training or inference."""
import copy
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/ti_tnn"))
from native import checkpoint as C, data as D, ledger, run_initial_state_study as R, train
from native import evaluate_initial_state_study as evaluation
from native.config import NativeConfig


def fixture_protocol(tmp_path):
    return {"format": "native-initial-state-preparation-v2", "conditions": list(R.CONDITIONS),
            "modes": list(R.MODES), "objective": "full_sequence", "model_seed": 2026111202,
            "sampler_seed": 2026111203, "lambda_balance": .01, "lambda_route": .1, "lambda_state": .1,
            "optimizer": {"learning_rate": .0003, "weight_decay": .01, "betas": [.9, .999], "eps": 1e-8, "grad_clip": 1.},
            "scheduler": {"warmup_fraction": .02, "final_lr_fraction": .1},
            "checkpoint_every": 500, "dev_every": 500, "dev_batches": 0,
            "endpoint_per_run": {"batch_size": 8, "updates": 1424, "full_label_tokens": 2590592, "cursor": 11376},
            "execution": {"study_id": R.STUDY_ID, "output": str(tmp_path / "comparison"),
                          "allocation_hours": 4., "native_cap_hours": 12., "source_commit": "source",
                          "environment_prefix": sys.prefix, "preparation": str(tmp_path),
                          "source_files": {}, "installed_versions": {}, "training_configs": {}},
            "artifact_sha256": {}, "resource_projection": {
                "per_model_per_condition_seconds": {mode: {"total": 300.} for mode in R.MODES},
                "common_seconds": {"loading": 60.}, "shutdown_allowance_hours": .05,
                "checkpoint_storage_bytes_all_runs": 32, "additional_data_logs_export_storage_allowance_bytes": 100}}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    p = fixture_protocol(tmp_path)
    token = tmp_path / "tokenizer"
    token.write_text("fixture-tokenizer")
    p["tokenizer_sha256"] = D.sha256_file(token)
    for arm in R.ARMS:
        condition, mode = arm.rsplit("-", 1)
        args = R.arguments(p, tmp_path / condition, tmp_path / "unused", mode)
        sampler = SimpleNamespace(data_identity=f"data-{condition}", order_identity="fixed-order")
        p["execution"]["training_configs"][arm] = train.fresh_run_config(
            args, NativeConfig.from_json(args.config), Path(args.data), sampler, sampler, 1424, 2590592, "code")
        p["execution"]["training_configs"][arm]["model"]["mode"] = mode
    path, lock = tmp_path / "ledger.jsonl", tmp_path / "gpu.lock"
    path.write_text(json.dumps({"event": "start", "reservation": "history", "t": 1., "cap_hours": ledger.CAP_HOURS}) + "\n"
                    + json.dumps({"event": "end", "reservation": "history", "t": 2., "elapsed": ledger.HISTORICAL_HOURS * 3600,
                                  "status": "complete"}) + "\n")
    monkeypatch.setattr(ledger, "validate_shared_reservation_paths", lambda: (path, lock))
    monkeypatch.setattr(ledger._base, "foreign_gpu_processes", lambda: [])
    monkeypatch.setattr(R, "preflight", lambda *args, **kwargs: p)
    calls = []

    def fake_train(args):
        binding = args.initial_state_study_binding
        config = train.bind_initial_state_owner(binding["expected_config"], binding, ledger.active())
        arm = Path(args.output).parent.name
        calls.append((arm, args))
        directory = Path(args.output)
        checkpoint = directory / "checkpoints/final-update-00001424.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("CPU fixture, not model weights")
        R.write_json(directory / "run_config.json", config)
        return {"format": train.TRAIN_FORMAT, "mode": args.mode, "loss_mode": args.loss_mode,
                "step": 1424, "planned_updates": 1424, "cursor": 11376,
                "consumed_loss_tokens": args.max_loss_tokens, "max_loss_tokens": args.max_loss_tokens,
                "final_checkpoint": {"path": str(checkpoint), "sha256": D.sha256_file(checkpoint)}}

    def fake_load(path, **kwargs):
        assert kwargs["expected_sha256"] == D.sha256_file(path)
        assert R.read_json(Path(path).parents[1] / "run_config.json") == kwargs["expected_config"]
        config = kwargs["expected_config"]
        assert config["code_identity"] == kwargs["expected_code_identity"]
        assert config["train_data_identity"] == kwargs["expected_data_identity"]
        assert config["order_identity"] == kwargs["expected_order_identity"]
        return {"step": 1424, "cursor": 11376, "consumed_loss_tokens": 2590592}

    def fake_evaluate(checkpoint, data, tokenizer, output):
        output.mkdir(parents=True, exist_ok=False)
        report = {"complete": True, "checkpoint_sha256": checkpoint["sha256"],
                  "populations": {name: {"score": {"complete_answers": {"correct": 0, "total": 128},
                                                   "registers": {"total": 512}},
                                         "answer_ce": {"complete": True}, "state_readout": {"complete": True}}
                                  for name in R.POPULATIONS}, "reconstruction": {"agreement": True}}
        R.write_json(output / "evaluation.json", report)
        return report

    monkeypatch.setattr(train, "run_training", fake_train)
    monkeypatch.setattr(C, "load_checkpoint", fake_load)
    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    return SimpleNamespace(p=p, root=tmp_path, tokenizer=token, ledger=path, lock=lock, calls=calls,
                           run=lambda **kw: R.run(tmp_path, token, "protocol", **kw))


def test_dry_run_creates_no_reservation_training_or_comparison(harness, monkeypatch):
    before = harness.ledger.read_bytes()
    monkeypatch.setattr(ledger, "Reservation", lambda **kw: pytest.fail("dry run reserved"))
    report = harness.run()
    assert report["status"] == "dry_run" and report["incomplete_arms"] == list(R.ARMS)
    assert not harness.calls and harness.ledger.read_bytes() == before
    assert not harness.lock.exists() and not (harness.root / "comparison").exists()


@pytest.mark.parametrize("hours", [None, 0, 3, 5, True, float("nan")])
def test_execution_requires_exact_explicit_authorization(harness, hours):
    with pytest.raises(PermissionError):
        harness.run(execute=True, authorize_hours=hours)
    assert not harness.calls and not (harness.root / "comparison").exists()


def test_all_eight_fixed_endpoints_and_no_duplicate_launch(harness):
    report = harness.run(execute=True, authorize_hours=4.)
    assert report["complete"] and set(report["arms"]) == set(R.ARMS)
    assert [arm for arm, _ in harness.calls] == list(R.ARMS)
    for arm, args in harness.calls:
        assert args.max_loss_tokens == 2590592 and args.batch_size == 8
        assert args.model_seed == 2026111202 and args.data_seed == 2026111203
        assert args.loss_mode == "full_sequence" and args.checkpoint_every == 500
        assert args.mode == arm.rsplit("-", 1)[1]
    with pytest.raises(FileExistsError):
        harness.run(execute=True, authorize_hours=4.)
    assert harness.run(execute=True, authorize_hours=4., resume=True)["status"] == "already_complete"
    assert len(harness.calls) == 8
    assert len([e for e in ledger.read(harness.ledger) if e["event"] == "start"]) == 2


def test_failed_arm_missing_arms_and_explicit_recovery_keep_prior_cost(harness, monkeypatch):
    original = evaluation.evaluate
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fixture failure")))
    with pytest.raises(RuntimeError, match="fixture failure"):
        harness.run(execute=True, authorize_hours=4.)
    first = R.read_json(harness.root / "comparison/attempt-0001.json")
    assert not first["complete"] and first["arms"][R.ARMS[0]]["status"] == "failed"
    assert all(not a["complete"] for a in first["arms"].values())
    assert all(first["arms"][a]["status"] == "pending" for a in R.ARMS[1:])
    assert first["screen"]["status"] == "incomplete"
    first_bytes = (harness.root / "comparison/attempt-0001.json").read_bytes()
    monkeypatch.setattr(evaluation, "evaluate", original)
    prior = ledger.usage(harness.ledger)["native_used_hours"]
    report = harness.run(execute=True, authorize_hours=4., resume=True)
    assert report["complete"] and report["plan"]["prior_charged_hours"] == pytest.approx(prior)
    assert len(harness.calls) == 8  # first completed training reused after evaluation failure
    assert (harness.root / "comparison/attempt-0001.json").read_bytes() == first_bytes
    assert ledger.usage(harness.ledger)["native_used_hours"] >= prior


def test_cumulative_allocation_not_refreshed_and_native_cap_is_separate(harness):
    with pytest.raises(ledger.BudgetExceeded):
        R.projection(harness.p, {}, 3.5, 5.0)
    with pytest.raises(ledger.BudgetExceeded):
        R.projection(harness.p, {}, 0., .1)
    plan = R.projection(harness.p, dict.fromkeys(R.ARMS[:6]), 2., 3.)
    assert plan["prior_charged_hours"] == 2. and plan["cumulative_upper_plan_hours"] > 2.


def test_open_own_reservation_and_different_comparison_identity_fail_closed(harness):
    identity = R.owner(harness.p, "protocol")
    with harness.ledger.open("a") as handle:
        handle.write(json.dumps({"event": "start", "reservation": "interrupted", "t": 3.,
                                 "extra": {"initial_state_study": identity}}) + "\n")
    with pytest.raises(RuntimeError, match="open reservation"):
        harness.run()
    different = {**identity, "output": str(harness.root / "duplicate")}
    with pytest.raises(RuntimeError, match="different immutable"):
        R.prior_costs(harness.ledger, different)


def test_owner_binding_is_immutable_across_interrupted_reservation_rollovers(harness):
    config = harness.p["execution"]["training_configs"][R.ARMS[0]]
    identity = R.owner(harness.p, "protocol")
    binding = {"owner": identity, "expected_config": config}
    encoded = []
    for attempt in ("first", "interrupted-recovery", "next-recovery"):
        reservation = SimpleNamespace(id=attempt, extra={"initial_state_study": identity})
        bound = train.bind_initial_state_owner(config, binding, reservation)
        encoded.append(train._json_bytes(bound))
        assert "reservation_id" not in bound and "acquisition_run_id" not in bound
    assert encoded[0] == encoded[1] == encoded[2]
    with pytest.raises(RuntimeError, match="active reservation"):
        train.bind_initial_state_owner(config, binding, None)
    with pytest.raises(RuntimeError, match="authorized initial-state owner"):
        train.bind_initial_state_owner(config, binding, SimpleNamespace(extra={"native_acquisition_run_id": "old"}))


def test_final_checkpoint_result_gap_never_reenters_trainer(harness, monkeypatch):
    original = train.run_training
    final_results = {}

    def interrupted(args):
        result = original(args)
        final_results.update(result)
        raise RuntimeError("interrupted after final journal before result")

    monkeypatch.setattr(train, "run_training", interrupted)
    with pytest.raises(RuntimeError, match="after final journal"):
        harness.run(execute=True, authorize_hours=4.)
    arm = R.ARMS[0]
    directory = harness.root / "comparison/arms" / arm
    final = Path(final_results["final_checkpoint"]["path"])
    before = final.read_bytes()
    data = Path(harness.p["execution"]["training_configs"][arm]["data"])
    R.write_json(data / "manifest.json", {"fixture": True})
    manifest = {**final_results["final_checkpoint"], "bytes": len(before), "step": 1424,
                "cursor": 11376, "consumed_loss_tokens": 2590592}
    monkeypatch.setattr(C, "recover", lambda *a, **kw: SimpleNamespace(
        checkpoint_path=final, manifest=manifest, requires_replay=False, invalid_checkpoints=[]))
    monkeypatch.setattr(train, "run_training", lambda *a, **kw: pytest.fail("final descriptor recovery trained"))
    monkeypatch.setattr(evaluation, "evaluate", lambda *a, **kw: pytest.fail("descriptor recovery evaluated"))
    recovered = R.recover_final_result(harness.p, R.owner(harness.p, "protocol"), arm, directory)
    assert recovered["recovered_final_descriptor_only"] and final.read_bytes() == before
    assert not (directory / "training_result.json").exists()
    monkeypatch.setattr(C, "recover", lambda *a, **kw: SimpleNamespace(
        checkpoint_path=directory / "earlier.pt", requires_replay=False, invalid_checkpoints=[]))
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        R.recover_final_result(harness.p, R.owner(harness.p, "protocol"), arm, directory)


def test_concurrent_resume_and_changed_budget_at_gpu_lock_fail_closed(harness, monkeypatch):
    output = harness.root / "comparison"
    output.mkdir()
    R.write_json(output / "claim.json", R.owner(harness.p, "protocol"))
    with (output / "comparison.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            harness.run(execute=True, authorize_hours=4., resume=True)
    original = ledger.Reservation

    class BudgetRace(original):
        def __enter__(self):
            result = super().__enter__()
            self.used_before = ledger.CAP_HOURS - .01
            return result

    monkeypatch.setattr(ledger, "Reservation", BudgetRace)
    with pytest.raises(ledger.BudgetExceeded):
        harness.run(execute=True, authorize_hours=4., resume=True)
    assert not harness.calls
    assert R.read_json(output / "attempt-0001.json")["complete"] is False


@pytest.mark.parametrize("field", ["code_identity", "train_data_identity", "dev_data_identity", "order_identity", "model", "training"])
def test_training_rejects_source_data_model_sampler_or_endpoint_mutation(harness, field):
    config = harness.p["execution"]["training_configs"][R.ARMS[0]]
    binding = {"owner": R.owner(harness.p, "protocol"), "expected_config": config}
    changed = {**config, field: "mutated"}
    with pytest.raises(C.CheckpointIdentityMismatch):
        train.bind_initial_state_owner(changed, binding, SimpleNamespace(extra={"initial_state_study": binding["owner"]}))


def test_tampered_completed_artifacts_are_not_skipped(harness):
    harness.run(execute=True, authorize_hours=4.)
    path = harness.root / "comparison/arms" / R.ARMS[0] / "training_result.json"
    path.write_text("{}")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        harness.run(execute=True, authorize_hours=4., resume=True)


def test_real_preflight_hashes_and_training_identity(tmp_path, monkeypatch):
    p = fixture_protocol(tmp_path)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.write_text("tokenizer fixture")
    artifact = tmp_path / "data.npy"
    artifact.write_text("array fixture")
    p["tokenizer_sha256"] = D.sha256_file(tokenizer)
    p["artifact_sha256"] = {"data.npy": D.sha256_file(artifact)}
    p["execution"]["training_configs"] = dict.fromkeys(R.ARMS, {"model": "fixed", "order_identity": "fixed"})
    monkeypatch.setattr(R, "source_commit", lambda *a, **kw: "source")
    monkeypatch.setattr(R, "locked_environment", lambda *a: {})
    monkeypatch.setattr(R, "training_config", lambda *a: {"model": "fixed", "order_identity": "fixed"})
    R.write_json(tmp_path / "protocol.json", p)
    digest = D.sha256_file(tmp_path / "protocol.json")
    R.preflight(tmp_path, tokenizer, digest)
    with pytest.raises(RuntimeError, match="protocol SHA"):
        R.preflight(tmp_path, tokenizer, "wrong")
    monkeypatch.setattr(R, "training_config", lambda *a: {"model": "wrong", "order_identity": "fixed"})
    with pytest.raises(RuntimeError, match="source/data/model/sampler"):
        R.preflight(tmp_path, tokenizer, digest)
    artifact.write_text("changed")
    with pytest.raises(RuntimeError, match="artifact/source"):
        R.preflight(tmp_path, tokenizer, digest)


def test_original_development_screen_and_missing_control():
    evaluations = {arm: {"populations": {name: {"score": {"complete_answers": {"correct": 10}}}
                                          for name in R.POPULATIONS}} for arm in R.ARMS}
    assert not R.development_screen(evaluations)["recommend_value_reuse"]
    for mode in ("dense", "shared"):
        for population in ("transition_values_id", "program_values_id", "composition"):
            evaluations[f"multi_state-{mode}"]["populations"][population]["score"]["complete_answers"]["correct"] += 8
    assert R.development_screen(evaluations)["recommend_composition_transfer"]
    evaluations["multi_state-shared"]["populations"]["dev"]["score"]["complete_answers"]["correct"] -= 1
    assert not R.development_screen(evaluations)["recommend_value_reuse"]
    del evaluations["repeat_one-shared"]
    assert R.development_screen(evaluations)["status"] == "incomplete"
