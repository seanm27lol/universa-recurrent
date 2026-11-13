"""Fixed four-arm binding to native.train and the existing shared reservation.

Default is read-only, including a structured budget-blocked result. This module
adds no scheduler, watchdog, alternate ledger or implicit recovery operation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

from . import checkpoint as C, data as D, ledger, train
from . import run_initial_state_study as I
from .config import NativeConfig
from .evaluate_mixed_learning import evaluate_endpoint
from .initial_state_study import POPULATIONS

MODES = ("dense", "shared", "untyped", "typed")
STUDY_ID = "native-mixed-learning-v1"
FORMAT = "native-mixed-learning-execution-v1"
ALLOCATION = 4.75
ENDPOINT = {"requested_loss_tokens": 40_000_000, "full_label_tokens": 40_001_786,
            "updates": 19_862, "cursor": 158_896, "batch_size": 8}
read_json, write_json, verify_hashes = I.read_json, I.write_json, I.verify_hashes


def arguments(protocol, data, output, mode):
    args = I.arguments(protocol, data, output, mode)
    args.max_loss_tokens = protocol["endpoint_per_run"]["requested_loss_tokens"]
    return args


def training_configs(protocol, data):
    """Recompute the common ordered exposure once, without constructing a model."""
    endpoint = protocol["endpoint_per_run"]
    sampler = train.make_sampler(D.PackedDataset(data, "train"), endpoint["batch_size"], protocol["sampler_seed"])
    dev = train.make_sampler(D.PackedDataset(data, "dev"), endpoint["batch_size"], protocol["sampler_seed"])
    digest = hashlib.sha256()
    while sampler.consumed_loss_tokens < endpoint["requested_loss_tokens"]:
        digest.update(sampler.next_batch().batch_identity.encode())
    manifest = read_json(Path(data) / "manifest.json")
    observed = (sampler.batches_consumed, sampler.cursor, sampler.consumed_loss_tokens)
    if observed != (endpoint["updates"], endpoint["cursor"], endpoint["full_label_tokens"]):
        raise RuntimeError("sampler endpoint mismatch")
    if any(manifest["endpoint"][key] != value for key, value in {
        "data_identity": sampler.data_identity, "order_identity": sampler.order_identity,
        "order_prefix_sha256": digest.hexdigest()}.items()):
        raise RuntimeError("prepared sampler/order identity mismatch")
    configs = {}
    for mode in MODES:
        args = arguments(protocol, data, Path(data) / "unused-read-only-output", mode)
        model = replace(NativeConfig.from_json(args.config), mode=mode)
        configs[mode] = train.fresh_run_config(args, model, Path(data), sampler, dev,
            endpoint["updates"], endpoint["full_label_tokens"], train._code_identity())
    return configs


def preflight(protocol, tokenizer, expected_protocol_sha256):
    protocol, tokenizer = Path(protocol).resolve(), Path(tokenizer).resolve()
    if D.sha256_file(protocol) != expected_protocol_sha256:
        raise RuntimeError("protocol SHA256 mismatch")
    p = read_json(protocol)
    e = p["execution"]
    if (p["format"] != FORMAT or e["study_id"] != STUDY_ID or p["modes"] != list(MODES)
            or p["objective"] != "full_sequence" or p["endpoint_per_run"] != ENDPOINT
            or p["initialization"] != {"kind": "fresh_random", "parent_checkpoint": None, "seed": 20261108}
            or p["model_seed"] != 20261108 or p["sampler_seed"] != 20261108
            or e["allocation_hours"] != ALLOCATION or e["native_cap_hours"] != ledger.ADDITIONAL_HOURS):
        raise RuntimeError("fixed mixed-learning contract changed")
    if {key: p[key] for key in ("lambda_balance", "lambda_route", "lambda_state")} != {
            "lambda_balance": .01, "lambda_route": .1, "lambda_state": .1}:
        raise RuntimeError("auxiliary objective changed")
    root = Path(__file__).resolve().parents[3]
    if I.source_commit(root, clean=True) != e["source_commit"]:
        raise RuntimeError("source commit mismatch")
    if (str(Path(sys.prefix).resolve()) != str(Path(e["environment_prefix"]).resolve())
            or sys.executable != e["python"]):
        raise RuntimeError("use the pinned native interpreter")
    verify_hashes(root, e["source_files"])
    if I.locked_environment(root) != e["installed_versions"]:
        raise RuntimeError("installed environment mismatch")
    D.verify_file(tokenizer, p["tokenizer_sha256"])
    data = Path(e["preparation"])
    D.verify_file(data / "manifest.json", p["preparation_sha256"])
    verify_hashes(data, p["artifact_sha256"])
    if training_configs(p, data) != e["training_configs"]:
        raise RuntimeError("source/data/model/sampler/training configuration mismatch")
    if Path(e["output"]).resolve() != ledger.shared_runs().resolve().parent / "ti_tnn_native_mixed_learning_v1":
        raise RuntimeError("comparison output is not canonical")
    return p


def owner(protocol, digest):
    return {"study_id": STUDY_ID, "protocol_sha256": digest,
            "output": protocol["execution"]["output"], "allocation_hours": ALLOCATION}


def prior_costs(path, identity, *, current_reservation=None):
    usage = ledger.usage(path)
    charges = {r["reservation"]: r for r in usage["reservations"]}
    total, ids = 0., []
    for event in ledger.read(path):
        if event.get("event") != "start":
            continue
        recorded = event.get("extra", {}).get("mixed_learning_study")
        if not recorded or recorded.get("study_id") != STUDY_ID:
            continue
        if recorded != identity:
            raise RuntimeError("comparison already has a different immutable identity")
        rid = event["reservation"]
        if rid == current_reservation:
            continue
        if rid in ids or charges[rid]["open"]:
            raise RuntimeError("duplicate or open comparison reservation; owner reconciliation required")
        ids.append(rid)
        total += charges[rid]["hours"]
    return total, ids, usage


def projection(protocol, completed, prior, native_remaining):
    r = protocol["resource_projection"]
    hours = (sum(r["common_seconds"].values()) + sum(r["per_model_per_condition_seconds"][m]["total"]
             for m in MODES if m not in completed)) / 3600
    # The shared ledger owns this margin. Do not discount its projected_hours.
    watchdog = hours * (1 + ledger._base.PROJECTION_MARGIN)
    required = watchdog + r["shutdown_allowance_hours"]
    if not all(math.isfinite(v) and v >= 0 for v in (hours, prior, native_remaining, required)):
        raise ValueError("nonfinite or negative resource projection")
    return {"projected_hours": hours, "watchdog_hours": watchdog,
        "margin": ledger._base.PROJECTION_MARGIN, "shutdown_allowance_hours": r["shutdown_allowance_hours"],
        "prior_charged_hours": prior, "required_remaining_hours": required,
        "cumulative_upper_plan_hours": prior + required, "allocation_hours": ALLOCATION,
        "native_remaining_hours": native_remaining, "allocation_shortfall_hours": max(0., prior + required - ALLOCATION),
        "native_shortfall_hours": max(0., required - native_remaining),
        "fits": prior + required <= ALLOCATION and required <= native_remaining,
        "incomplete_arms": [m for m in MODES if m not in completed]}


def bound_config(protocol, identity, mode):
    return {**protocol["execution"]["training_configs"][mode], "mixed_learning_study": identity}


def validate_training(protocol, identity, mode, directory, result):
    endpoint = protocol["endpoint_per_run"]
    checkpoint = result.get("final_checkpoint", {})
    path = directory / "train/checkpoints" / f"final-update-{endpoint['updates']:08d}.pt"
    if (Path(checkpoint.get("path", "")).resolve() != path.resolve() or result.get("mode") != mode
            or result.get("loss_mode") != "full_sequence" or any(result.get(key) != value for key, value in {
                "step": endpoint["updates"], "planned_updates": endpoint["updates"], "cursor": endpoint["cursor"],
                "consumed_loss_tokens": endpoint["full_label_tokens"], "max_loss_tokens": endpoint["requested_loss_tokens"]}.items())):
        raise RuntimeError("incomplete or changed training endpoint")
    config = bound_config(protocol, identity, mode)
    if read_json(directory / "train/run_config.json") != config:
        raise RuntimeError("persisted training configuration mismatch")
    payload = C.load_checkpoint(path, expected_sha256=checkpoint["sha256"], expected_config=config,
        expected_code_identity=config["code_identity"], expected_data_identity=config["train_data_identity"],
        expected_order_identity=config["order_identity"], expected_identities={"dev_data": config["dev_data_identity"],
        "data_manifest": D.sha256_file(Path(config["data"]) / "manifest.json")})
    if (payload["step"], payload["cursor"], payload["consumed_loss_tokens"]) != (
            endpoint["updates"], endpoint["cursor"], endpoint["full_label_tokens"]):
        raise RuntimeError("checkpoint endpoint mismatch")
    return checkpoint


def recover_final_result(protocol, identity, mode, directory):
    endpoint = protocol["endpoint_per_run"]
    final = directory / "train/checkpoints" / f"final-update-{endpoint['updates']:08d}.pt"
    if not final.exists():
        return None
    config = bound_config(protocol, identity, mode)
    plan = C.recover(directory / "train" / train.JOURNAL_NAME, expected_config=config,
        expected_code_identity=config["code_identity"], expected_data_identity=config["train_data_identity"],
        expected_order_identity=config["order_identity"], expected_identities={"dev_data": config["dev_data_identity"],
        "data_manifest": D.sha256_file(Path(config["data"]) / "manifest.json")})
    if plan.checkpoint_path.resolve() != final.resolve() or plan.requires_replay or plan.invalid_checkpoints:
        raise RuntimeError("final checkpoint is not a valid journaled endpoint; refusing overwrite")
    result = {"mode": mode, "loss_mode": "full_sequence", "step": endpoint["updates"],
        "planned_updates": endpoint["updates"], "cursor": endpoint["cursor"],
        "consumed_loss_tokens": endpoint["full_label_tokens"], "max_loss_tokens": endpoint["requested_loss_tokens"],
        "recovered_final_descriptor_only": True, "timing_scope": "prior logs/costs retained, no invented timers",
        "final_checkpoint": {**plan.manifest, "path": str(final),
                              "manifest_path": str(C.manifest_path(final, plan.manifest["sha256"]))}}
    validate_training(protocol, identity, mode, directory, result)
    return result


def validate_evaluation(protocol, report, checkpoint):
    if (report.get("checkpoint_sha256") != checkpoint["sha256"] or report.get("complete") is not True
            or report.get("preparation_sha256") != protocol["preparation_sha256"]
            or set(report.get("capability", {})) != {"wikipedia", "code"}):
        raise RuntimeError("incomplete or changed capability evaluation")
    for source, population in report["capability"].items():
        ce, gen = population["teacher_forced"], population["continuation"]
        if (ce["complete"] is not True or ce["sequences"] != 128
                or ce["sources"][source]["loss_tokens"] != protocol["capability_loss_tokens"][source]
                or gen["complete"] is not True or gen["examples"] != 64 or gen["token_decisions"] != 2048
                or gen["missing"] != 0):
            raise RuntimeError("capability denominator or completeness mismatch")
    I.validate_evaluation(report["synthetic_diagnostics"], checkpoint)


def completed_arms(protocol, identity, output):
    result = {}
    for mode in MODES:
        directory = output / "arms" / mode
        if not (directory / "completion.json").is_file():
            continue
        value = read_json(directory / "completion.json")
        if value["owner"] != identity:
            raise RuntimeError("completed arm owner mismatch")
        if (value["evaluation"] not in value["artifacts"]
                or not (directory / value["evaluation"]).resolve().is_relative_to(directory.resolve())):
            raise RuntimeError("evaluation is not a bound arm artifact")
        verify_hashes(directory, value["artifacts"])
        checkpoint = validate_training(protocol, identity, mode, directory, read_json(directory / "training_result.json"))
        validate_evaluation(protocol, read_json(directory / value["evaluation"]), checkpoint)
        result[mode] = value
    return result


def execute_arm(protocol, identity, tokenizer, output, mode, attempt, reservation):
    data = Path(protocol["execution"]["preparation"])
    directory = output / "arms" / mode
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "training_result.json"
    if result_path.exists():
        result = read_json(result_path)
    else:
        result = recover_final_result(protocol, identity, mode, directory)
        if result is None:
            args = arguments(protocol, data, directory / "train", mode)
            args.mixed_learning_binding = {"owner": identity, "expected_config": protocol["execution"]["training_configs"][mode]}
            reservation.check()
            result = train.run_training(args)
        write_json(result_path, result)
    checkpoint = validate_training(protocol, identity, mode, directory, result)
    reservation.check()
    evaluation_path = directory / f"evaluation-attempt-{attempt:04d}" / "evaluation.json"
    report = evaluate_endpoint(checkpoint, data, tokenizer, evaluation_path.parent)
    validate_evaluation(protocol, report, checkpoint)
    artifacts = {str(p.relative_to(directory)): D.sha256_file(p) for p in evaluation_path.parent.rglob("*") if p.is_file()}
    artifacts["training_result.json"] = D.sha256_file(result_path)
    value = {"owner": identity, "evaluation": str(evaluation_path.relative_to(directory)), "artifacts": artifacts}
    write_json(directory / "completion.json", value)
    return value


def check_storage(output):
    existing = output
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    if free < 64 * 1024**3:
        raise RuntimeError("insufficient output-filesystem storage for complete comparison")
    return free


def check_idle(ledger_path, gpu_lock, output):
    # Read-only lock probes do not create, truncate or annotate any lock file.
    paths = [gpu_lock]
    if (output / "comparison.lock").exists():
        paths.append(output / "comparison.lock")
    for path in paths:
        with path.open("r") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("conflicting reservation or comparison is active") from error
            fcntl.flock(handle, fcntl.LOCK_UN)
    charges = {v["reservation"]: v for v in ledger.usage(ledger_path)["reservations"]}
    for event in ledger.read(ledger_path):
        if event.get("event") == "start" and event.get("authorization") == 3 and charges[event["reservation"]]["open"]:
            raise RuntimeError("native reservation is open; owner reconciliation required")


def run(protocol, tokenizer, expected_protocol_sha256, *, execute=False, authorize_hours=None, resume=False):
    if execute and (type(authorize_hours) not in (int, float) or authorize_hours != ALLOCATION):
        raise PermissionError("explicit owner authorization requires --execute --authorize-hours 4.75")
    if not execute and authorize_hours is not None:
        raise ValueError("authorization only accompanies explicit execution")
    started = time.perf_counter()
    p = preflight(protocol, tokenizer, expected_protocol_sha256)
    identity = owner(p, expected_protocol_sha256)
    output = Path(identity["output"])
    path, lock = ledger.validate_shared_reservation_paths()
    prior, ids, usage = prior_costs(path, identity)
    check_idle(path, lock, output)
    free = check_storage(output)
    if output.exists():
        if not resume:
            raise FileExistsError("comparison already claimed; explicit --resume required")
        if read_json(output / "claim.json") != identity:
            raise RuntimeError("comparison claim identity mismatch")
    elif resume or ids:
        raise RuntimeError("missing comparison claim is not a fresh launch or recovery")
    completed = completed_arms(p, identity, output)
    if len(completed) == len(MODES):
        return {"status": "already_complete", "training_launched": False, "reservation_created": False}
    plan = projection(p, completed, prior, usage["native_remaining_hours"])
    result = {**plan, "owner": identity, "prior_reservations": ids, "preflight_seconds": time.perf_counter() - started,
        "identity_endpoint_environment_checks": "passed", "output_free_bytes": free,
        "training_launched": False, "reservation_created": False,
        "status": "dry_run" if plan["fits"] else "blocked_budget"}
    if not execute:
        return result
    if not plan["fits"]:
        raise ledger.BudgetExceeded(json.dumps(result, sort_keys=True))
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "claim.json", identity)
    with (output / "comparison.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = completed_arms(p, identity, output)
        if len(completed) == len(MODES):
            return {"status": "already_complete", "training_launched": False, "reservation_created": False}
        prior, ids, usage = prior_costs(path, identity)
        plan = projection(p, completed, prior, usage["native_remaining_hours"])
        if not plan["fits"]:
            raise ledger.BudgetExceeded("updated cumulative allocation/native balance cannot fit all remaining arms")
        attempt = len(list(output.glob("attempt-*.json"))) + 1
        status = {"status": "running", "owner": identity, "attempt": attempt, "plan": plan, "complete": False, "failure": None,
            "arms": {m: {"status": "complete" if m in completed else "pending", "complete": m in completed} for m in MODES}}
        destination = output / f"attempt-{attempt:04d}.json"
        write_json(destination, status)
        try:
            with ledger.Reservation(command=f"native.mixed_learning {STUDY_ID} attempt={attempt}",
                    projected_hours=plan["projected_hours"], path=path, lock=lock,
                    extra={"mixed_learning_study": identity}) as reservation:
                status["reservation_id"] = reservation.id
                prior, _, _ = prior_costs(path, identity, current_reservation=reservation.id)
                status["plan"] = projection(p, completed, prior, ledger.CAP_HOURS - reservation.used_before)
                if not status["plan"]["fits"]:
                    raise ledger.BudgetExceeded("budget changed at shared-lock admission")
                for mode in MODES:
                    if mode in completed:
                        continue
                    status["arms"][mode] = {"status": "running", "complete": False}
                    write_json(destination, status)
                    try:
                        completed[mode] = execute_arm(p, identity, tokenizer, output, mode, attempt, reservation)
                    except BaseException as error:
                        status["arms"][mode] = {"status": "failed", "complete": False, "failure": repr(error)}
                        raise
                    status["arms"][mode] = {"status": "complete", "complete": True}
                    write_json(destination, status)
                    reservation.check()
                status["complete"] = len(completed) == len(MODES)
                status["status"] = "complete" if status["complete"] else "incomplete"
        except BaseException as error:
            status["failure"] = repr(error)
            status["complete"] = False
            status["status"] = "failed"
            raise
        finally:
            status["ledger_after"] = ledger.usage(path)
            write_json(destination, status)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "tokenizer", "expected-protocol-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorize-hours", type=float)
    parser.add_argument("--resume", action="store_true")
    report = run(**vars(parser.parse_args()))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 2 if report["status"] == "blocked_budget" else 0


if __name__ == "__main__":
    raise SystemExit(main())
