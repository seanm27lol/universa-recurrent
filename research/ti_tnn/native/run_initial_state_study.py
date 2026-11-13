"""Thin fixed eight-arm adapter to the existing native reservation and trainer.

Default is a read-only dry run. Execution requires the exact protocol hash and
explicit four-hour authorization from the existing cumulative native allowance.
There is no polling scheduler, new watchdog, or alternate ledger.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from . import checkpoint as C, data as D, ledger, train
from .config import NativeConfig
from .initial_state_study import CONDITIONS, POPULATIONS

MODES = ("dense", "shared", "untyped", "typed")
ARMS = tuple(f"{condition}-{mode}" for condition in CONDITIONS for mode in MODES)
STUDY_ID = "native-initial-state-paired-v2"
ALLOCATION = 4.0


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    C.atomic_write(path, train._json_bytes(value))


def source_commit(repository, *, clean=False):
    if clean and subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True):
        raise RuntimeError("execution requires a clean committed source worktree")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()


def arguments(protocol, data, output, mode):
    """Every arm gets the same registered endpoint; only mode/data differ."""
    root = Path(__file__).parent
    endpoint, optimizer, scheduler = (protocol[k] for k in ("endpoint_per_run", "optimizer", "scheduler"))
    values = {
        "config": root / "configs/engineering.json", "mode": mode, "data": data, "output": output,
        "device": "cuda", "loss-mode": "full_sequence", "batch-size": endpoint["batch_size"],
        "max-loss-tokens": endpoint["full_label_tokens"], "seed": protocol["model_seed"],
        "model-seed": protocol["model_seed"], "data-seed": protocol["sampler_seed"],
        "learning-rate": optimizer["learning_rate"], "weight-decay": optimizer["weight_decay"],
        "eps": optimizer["eps"], "grad-clip": optimizer["grad_clip"],
        "warmup-fraction": scheduler["warmup_fraction"], "final-lr-fraction": scheduler["final_lr_fraction"],
        "projected-hours": protocol["resource_projection"]["per_model_per_condition_seconds"][mode]["total"] / 3600,
    }
    for name in ("lambda_balance", "lambda_route", "lambda_state", "checkpoint_every", "dev_every", "dev_batches"):
        values[name.replace("_", "-")] = protocol[name]
    argv = [item for key, value in values.items() for item in (f"--{key}", str(value))]
    return train.parser().parse_args([*argv, "--betas", *map(str, optimizer["betas"])])


def training_config(protocol, data, mode):
    args = arguments(protocol, data, Path(data) / "unused-preflight-output", mode)
    model = replace(NativeConfig.from_json(args.config), mode=mode)
    sampler = train.make_sampler(D.PackedDataset(data, "train"), args.batch_size, args.data_seed)
    dev_sampler = train.make_sampler(D.PackedDataset(data, "dev"), args.batch_size, args.data_seed)
    updates, tokens = train._planned_budget(sampler, args.max_loss_tokens)
    endpoint = protocol["endpoint_per_run"]
    if (updates, tokens) != (endpoint["updates"], endpoint["full_label_tokens"]):
        raise RuntimeError("sampler endpoint differs from fixed protocol")
    for _ in range(updates):
        sampler.next_batch()
    if sampler.cursor != endpoint["cursor"]:
        raise RuntimeError("sampler cursor differs from fixed endpoint")
    return train.fresh_run_config(args, model, Path(data), sampler, dev_sampler,
                                  updates, tokens, train._code_identity())


def freeze_execution(protocol, preparation, output):
    """CPU-only binding; called by preparation after data are written."""
    root = Path(__file__).parent
    repository = root.parents[2]
    files = [*root.glob("*.py"), *root.joinpath("independent").glob("*.py"),
             root / "configs/engineering.json", repository / "requirements-native-lock.txt",
             root.parent / "scaling/ledger.py"]
    return {"study_id": STUDY_ID, "source_commit": source_commit(repository),
            "source_files": {str(p.relative_to(repository)): D.sha256_file(p) for p in sorted(files)},
            "installed_versions": locked_environment(repository),
            "python": sys.executable, "environment_prefix": sys.prefix,
            "preparation": str(Path(preparation).resolve()), "output": str(Path(output).resolve()),
            "allocation_hours": ALLOCATION, "native_cap_hours": ledger.ADDITIONAL_HOURS,
            "training_configs": {f"{condition}-{mode}": training_config(protocol, Path(preparation) / condition, mode)
                                 for condition in CONDITIONS for mode in MODES}}


def locked_environment(repository):
    versions = {}
    for line in (Path(repository) / "requirements-native-lock.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.split("==")
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"pinned environment mismatch for {name}: {actual} != {expected}")
        versions[name] = actual
    return versions


def verify_hashes(root, hashes):
    root = Path(root).resolve()
    for name, expected in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or D.sha256_file(path) != expected:
            raise RuntimeError(f"artifact/source identity mismatch: {name}")


def preflight(preparation, tokenizer, expected_hash, *, execute=False):
    preparation, tokenizer = Path(preparation).resolve(), Path(tokenizer).resolve()
    protocol_path = preparation / "protocol.json"
    if D.sha256_file(protocol_path) != expected_hash:
        raise RuntimeError("protocol SHA256 mismatch")
    p = read_json(protocol_path)
    e = p["execution"]
    if (p["format"] != "native-initial-state-preparation-v2" or e["study_id"] != STUDY_ID
            or p["conditions"] != list(CONDITIONS) or p["modes"] != list(MODES)
            or p["objective"] != "full_sequence" or e["allocation_hours"] != ALLOCATION
            or e["native_cap_hours"] != ledger.ADDITIONAL_HOURS):
        raise RuntimeError("unknown fixed comparison contract")
    if e["preparation"] != str(preparation) or D.sha256_file(tokenizer) != p["tokenizer_sha256"]:
        raise RuntimeError("preparation/tokenizer identity mismatch")
    root = Path(__file__).resolve().parents[3]
    if source_commit(root, clean=execute) != e["source_commit"]:
        raise RuntimeError("source commit mismatch")
    if str(Path(sys.prefix).resolve()) != str(Path(e["environment_prefix"]).resolve()):
        raise RuntimeError("use the pinned preparation environment")
    verify_hashes(root, e["source_files"])
    if locked_environment(root) != e["installed_versions"]:
        raise RuntimeError("installed environment identity mismatch")
    verify_hashes(preparation, p["artifact_sha256"])
    for condition in CONDITIONS:
        for mode in MODES:
            if training_config(p, preparation / condition, mode) != e["training_configs"][f"{condition}-{mode}"]:
                raise RuntimeError("source/data/model/sampler configuration mismatch")
    return p


def owner(protocol, protocol_hash):
    e = protocol["execution"]
    return {"study_id": e["study_id"], "protocol_sha256": protocol_hash,
            "output": e["output"], "allocation_hours": ALLOCATION}


def prior_costs(path, identity, *, current_reservation=None):
    """Read preserved base-ledger charges; never repair or close reservations here."""
    usage = ledger.usage(path)
    charges = {r["reservation"]: r for r in usage["reservations"]}
    total, ids = 0.0, []
    for event in ledger.read(path):
        if event.get("event") != "start":
            continue
        recorded = event.get("extra", {}).get("initial_state_study")
        if not recorded or recorded.get("study_id") != identity["study_id"]:
            continue
        if recorded != identity:
            raise RuntimeError("comparison already has a different immutable launch identity")
        reservation = event["reservation"]
        if reservation == current_reservation:
            continue
        if reservation in ids:
            raise RuntimeError("duplicate comparison reservation start")
        charge = charges[reservation]
        if charge["open"]:
            raise RuntimeError("comparison has an open reservation; owner reconciliation required")
        total += charge["hours"]
        ids.append(reservation)
    return total, ids, usage


def projection(protocol, completed, prior, native_remaining):
    resource = protocol["resource_projection"]
    seconds = sum(resource["common_seconds"].values())
    seconds += sum(resource["per_model_per_condition_seconds"][arm.rsplit("-", 1)[1]]["total"]
                   for arm in ARMS if arm not in completed)
    hours = seconds / 3600
    required = hours * 1.5 + resource["shutdown_allowance_hours"]
    if not all(math.isfinite(v) and v >= 0 for v in (hours, prior, native_remaining)):
        raise ValueError("nonfinite/negative comparison cost")
    if prior + required > ALLOCATION or required > native_remaining:
        raise ledger.BudgetExceeded("remaining complete comparison plus margin does not fit cumulative allocation/native cap")
    return {"prior_charged_hours": prior, "projected_hours": hours,
            "watchdog_hours": hours * 1.5, "shutdown_allowance_hours": resource["shutdown_allowance_hours"],
            "cumulative_upper_plan_hours": prior + required, "allocation_hours": ALLOCATION,
            "native_remaining_hours": native_remaining, "incomplete_arms": [a for a in ARMS if a not in completed]}


def validate_training(protocol, identity, arm, directory, result):
    """Validate actual checkpoint content and all saved identities, never status prose."""
    endpoint = protocol["endpoint_per_run"]
    checkpoint = result.get("final_checkpoint", {})
    path = directory / "train/checkpoints" / f"final-update-{endpoint['updates']:08d}.pt"
    if (Path(checkpoint.get("path", "")).resolve() != path.resolve()
            or result.get("mode") != arm.rsplit("-", 1)[1]
            or result.get("loss_mode") != "full_sequence"
            or any(result.get(k) != v for k, v in {
                "step": endpoint["updates"], "planned_updates": endpoint["updates"], "cursor": endpoint["cursor"],
                "consumed_loss_tokens": endpoint["full_label_tokens"], "max_loss_tokens": endpoint["full_label_tokens"]}.items())):
        raise RuntimeError(f"{arm}: incomplete or changed endpoint")
    config = {**protocol["execution"]["training_configs"][arm], "initial_state_study": identity}
    if read_json(directory / "train/run_config.json") != config:
        raise RuntimeError("persisted training configuration mismatch")
    payload = C.load_checkpoint(path, expected_sha256=checkpoint["sha256"], expected_config=config,
                                expected_code_identity=config["code_identity"],
                                expected_data_identity=config["train_data_identity"],
                                expected_order_identity=config["order_identity"])
    if (payload["step"], payload["cursor"], payload["consumed_loss_tokens"]) != (
            endpoint["updates"], endpoint["cursor"], endpoint["full_label_tokens"]):
        raise RuntimeError("checkpoint endpoint mismatch")
    return checkpoint


def recover_final_result(protocol, identity, arm, directory):
    """Bridge final-checkpoint/result-write interruption without a model call.

    A durable journaled final is adopted after existing checkpoint validation.
    An unjournaled or corrupt final fails closed and is never overwritten by a
    second trainer invocation. Intermediate recovery stays with native.train.
    """
    endpoint = protocol["endpoint_per_run"]
    final = directory / "train/checkpoints" / f"final-update-{endpoint['updates']:08d}.pt"
    if not final.exists():
        return None
    config = {**protocol["execution"]["training_configs"][arm], "initial_state_study": identity}
    plan = C.recover(directory / "train" / train.JOURNAL_NAME,
                     expected_config=config, expected_code_identity=config["code_identity"],
                     expected_data_identity=config["train_data_identity"], expected_order_identity=config["order_identity"],
                     expected_identities={"dev_data": config["dev_data_identity"],
                                          "data_manifest": D.sha256_file(Path(config["data"]) / "manifest.json")})
    if plan.checkpoint_path.resolve() != final.resolve() or plan.requires_replay or plan.invalid_checkpoints:
        raise RuntimeError("existing final checkpoint is not a valid journaled endpoint; refusing to overwrite it")
    descriptor = {**plan.manifest, "path": str(final),
                  "manifest_path": str(C.manifest_path(final, plan.manifest["sha256"]))}
    result = {"format": train.TRAIN_FORMAT, "mode": arm.rsplit("-", 1)[1], "loss_mode": "full_sequence",
              "step": endpoint["updates"], "planned_updates": endpoint["updates"], "cursor": endpoint["cursor"],
              "consumed_loss_tokens": endpoint["full_label_tokens"], "max_loss_tokens": endpoint["full_label_tokens"],
              "final_checkpoint": descriptor, "recovered_final_descriptor_only": True,
              "timing_scope": "original learning logs and cumulative ledger retained; invocation timers not reconstructed"}
    validate_training(protocol, identity, arm, directory, result)
    return result


def validate_evaluation(evaluation, checkpoint):
    if evaluation.get("checkpoint_sha256") != checkpoint["sha256"] or evaluation.get("complete") is not True:
        raise RuntimeError("evaluation checkpoint mismatch or incomplete evaluation")
    if set(evaluation.get("populations", {})) != set(POPULATIONS):
        raise RuntimeError("evaluation population missing")
    for name, report in evaluation["populations"].items():
        if (report["score"]["complete_answers"]["total"] != 128
                or report["score"]["registers"]["total"] != 512
                or report["answer_ce"]["complete"] is not True
                or report["state_readout"]["complete"] is not True):
            raise RuntimeError(f"{name}: incomplete fixed evaluation")
    if evaluation["reconstruction"]["agreement"] is not True:
        raise RuntimeError("bounded CPU FP32 replay failed; preserve failure")


def completed_arms(protocol, identity, output):
    completed = {}
    for arm in ARMS:
        directory = output / "arms" / arm
        marker = directory / "completion.json"
        if not marker.is_file():
            continue
        value = read_json(marker)
        if value["owner"] != identity:
            raise RuntimeError("completed arm owner mismatch")
        verify_hashes(directory, value["artifacts"])
        checkpoint = validate_training(protocol, identity, arm, directory, read_json(directory / "training_result.json"))
        evaluation = read_json(directory / value["evaluation"])
        validate_evaluation(evaluation, checkpoint)
        completed[arm] = value
    return completed


def development_screen(evaluations):
    """The original complete-answer screen; new register strata are descriptive."""
    if set(evaluations) != set(ARMS):
        return {"status": "incomplete", "recommend_value_reuse": None, "recommend_composition_transfer": None}
    deltas = {mode: {name: evaluations[f"multi_state-{mode}"]["populations"][name]["score"]["complete_answers"]["correct"]
                         - evaluations[f"repeat_one-{mode}"]["populations"][name]["score"]["complete_answers"]["correct"]
                     for name in POPULATIONS} for mode in MODES}
    value = all(deltas[m]["transition_values_id"] >= 8 and deltas[m]["program_values_id"] >= 8
                and deltas[m]["dev"] >= 0 for m in ("dense", "shared"))
    return {"status": "complete_development_screen", "complete_answer_deltas": deltas,
            "recommend_value_reuse": value,
            "recommend_composition_transfer": value and all(deltas[m]["composition"] >= 8 for m in ("dense", "shared")),
            "scope": "single-seed development screen, not a confirmation test"}


def execute_arm(protocol, identity, preparation, tokenizer, output, arm, attempt, reservation):
    from .evaluate_initial_state_study import evaluate

    condition, mode = arm.rsplit("-", 1)
    directory = output / "arms" / arm
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "training_result.json"
    if result_path.is_file():
        result = read_json(result_path)
    else:
        result = recover_final_result(protocol, identity, arm, directory)
        if result is None:
            args = arguments(protocol, preparation / condition, directory / "train", mode)
            args.initial_state_study_binding = {"owner": identity, "expected_config": protocol["execution"]["training_configs"][arm]}
            reservation.check()
            result = train.run_training(args)
        write_json(result_path, result)
    checkpoint = validate_training(protocol, identity, arm, directory, result)
    reservation.check()
    # A failed evaluation gets a fresh directory on explicit recovery; prior
    # saved outputs and errors remain untouched. Training is not repeated.
    evaluation_path = directory / f"evaluation-attempt-{attempt:04d}" / "evaluation.json"
    evaluation = evaluate(checkpoint, preparation / condition, tokenizer, evaluation_path.parent)
    validate_evaluation(evaluation, checkpoint)
    artifacts = {str(p.relative_to(directory)): D.sha256_file(p)
                 for p in sorted(evaluation_path.parent.rglob("*")) if p.is_file()}
    artifacts["training_result.json"] = D.sha256_file(result_path)
    completion = {"owner": identity, "evaluation": str(evaluation_path.relative_to(directory)), "artifacts": artifacts}
    write_json(directory / "completion.json", completion)
    return completion


def run(preparation, tokenizer, expected_protocol_sha256, *, execute=False, authorize_hours=None, resume=False):
    if execute and (type(authorize_hours) not in (int, float) or authorize_hours != ALLOCATION):
        raise PermissionError("explicit owner authorization requires --execute --authorize-hours 4")
    if not execute and authorize_hours is not None:
        raise ValueError("authorization is accepted only with explicit execution")
    started = time.perf_counter()
    p = preflight(preparation, tokenizer, expected_protocol_sha256, execute=execute)
    preparation, tokenizer = Path(preparation).resolve(), Path(tokenizer).resolve()
    identity = owner(p, expected_protocol_sha256)
    output = Path(identity["output"])
    ledger_path, ledger_lock = ledger.validate_shared_reservation_paths()
    prior, ids, usage = prior_costs(ledger_path, identity)
    if output.exists():
        if not resume:
            raise FileExistsError("comparison already claimed; explicit --resume is required, never a duplicate launch")
        if read_json(output / "claim.json") != identity:
            raise RuntimeError("comparison claim identity mismatch")
    elif resume:
        raise FileNotFoundError("cannot resume an unclaimed comparison")
    elif ids:
        raise RuntimeError("ledger already records this comparison; missing claim is not a fresh launch")
    completed = completed_arms(p, identity, output)
    if len(completed) == len(ARMS):
        return {"status": "already_complete", "training_launched": False, "reservation_created": False}
    plan = projection(p, completed, prior, usage["native_remaining_hours"])
    plan.update(owner=identity, prior_reservations=ids, preflight_seconds=time.perf_counter() - started,
                training_launched=False, reservation_created=False, status="dry_run")
    if not execute:
        return plan
    required_storage = (p["resource_projection"]["checkpoint_storage_bytes_all_runs"]
                        + p["resource_projection"]["additional_data_logs_export_storage_allowance_bytes"])
    if shutil.disk_usage(preparation).free < required_storage:
        raise RuntimeError("insufficient storage for complete comparison")
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "claim.json", identity)
    # One exclusive canonical comparison claim plus the existing shared GPU lock.
    with (output / "comparison.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("comparison is already running") from error
        completed = completed_arms(p, identity, output)
        if len(completed) == len(ARMS):
            return {"status": "already_complete", "training_launched": False, "reservation_created": False}
        prior, ids, usage = prior_costs(ledger_path, identity)
        plan = projection(p, completed, prior, usage["native_remaining_hours"])
        attempt = len(list(output.glob("attempt-*.json"))) + 1
        status = {"owner": identity, "attempt": attempt, "complete": False, "failure": None, "plan": plan,
                  "arms": {arm: {"status": "complete" if arm in completed else "pending", "complete": arm in completed}
                           for arm in ARMS}, "screen": development_screen({})}
        status_path = output / f"attempt-{attempt:04d}.json"
        write_json(status_path, status)
        try:
            with ledger.Reservation(command=f"native.initial_state_study {identity['study_id']} attempt={attempt}",
                                    projected_hours=plan["projected_hours"], path=ledger_path, lock=ledger_lock,
                                    extra={"initial_state_study": identity}) as reservation:
                # Admission is rechecked under the shared GPU lock, including
                # costs accrued between preflight and lock acquisition.
                charged, _, _ = prior_costs(ledger_path, identity, current_reservation=reservation.id)
                status["plan"] = projection(p, completed, charged, ledger.CAP_HOURS - reservation.used_before)
                status["reservation_id"] = reservation.id
                for arm in ARMS:
                    if arm in completed:
                        continue
                    status["arms"][arm] = {"status": "running", "complete": False}
                    write_json(status_path, status)
                    try:
                        completed[arm] = execute_arm(p, identity, preparation, tokenizer, output, arm, attempt, reservation)
                    except BaseException as error:
                        status["arms"][arm] = {"status": "failed", "complete": False, "failure": repr(error)}
                        raise
                    status["arms"][arm] = {"status": "complete", "complete": True}
                    write_json(status_path, status)
                    reservation.check()
                evaluations = {arm: read_json(output / "arms" / arm / completed[arm]["evaluation"]) for arm in ARMS}
                status["screen"] = development_screen(evaluations)
                status["complete"] = all(v["complete"] for v in status["arms"].values())
        except BaseException as error:
            status["failure"] = repr(error)
            status["complete"] = False
            raise
        finally:
            status["ledger_after"] = ledger.usage(ledger_path)
            write_json(status_path, status)
        return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-hours", type=float)
    parser.add_argument("--resume", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args(argv))), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
