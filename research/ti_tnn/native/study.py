"""Freeze, execute, and summarize the matched four-arm native engineering study.

For example, all four arms may start with a provisional 20 million shifted-label
budget.  Feasibility is decided from measured engineering profiles before any arm's
comparative outcome is read.  Atomic batches can cross a requested token count, so
the protocol freezes the exact common endpoint produced by the shared sampler.

The parent orchestrator never opens a GPU reservation.  ``native.train`` owns its
CUDA reservation directly.  Evaluation and reconstruction run on the declared FP32 CPU serving path,
so separate child wrappers hold one reservation each while running their exact
``python -m native.evaluate`` and ``python -m native.reconstruct`` commands.  State
interventions use the evaluator's Python API inside the evaluation reservation.

A passing orchestration test establishes protocol and accounting behavior only.  It
is not learning evidence, and the serious 125M projection is not an authorization.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Mapping, Sequence

from .config import MODES, NativeConfig


STUDY_FORMAT = "native-four-arm-engineering-study-v1"
RUN_STATUS_FORMAT = "native-four-arm-run-status-v1"
SUMMARY_FORMAT = "native-four-arm-summary-v1"
DEVELOPMENT_SEED = 20261108
PROVISIONAL_LOSS_TOKENS = 20_000_000
SERIOUS_LOSS_TOKENS = 125_000_000
RESERVE_FRACTION = 0.20
STRUCTURED_SPLITS = ("train", "dev", "unseen_composition", "longer_length")
DEFAULT_TRAIN = {
    "batch_size": 8,
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "grad_clip": 1.0,
    "lambda_balance": 0.01,
    "lambda_route": 0.1,
    "lambda_state": 0.1,
    "warmup_fraction": 0.02,
    "final_lr_fraction": 0.1,
    "checkpoint_every": 100,
    "dev_every": 100,
    "dev_batches": 16,
}
DEFAULT_EVALUATION = {
    "batch_size": 8,
    "dev_batches": 16,
    "limit_per_split": 128,
    "code_limit": 64,
    "max_new_tokens": 48,
    "code_max_new_tokens": 128,
    "eos_token_id": 50256,
    "use_cache": True,
}
DEFAULT_INTERVENTIONS = (
    {"pass_index": 0, "slot_index": 0, "kind": "zero", "source_slot": None},
    {"pass_index": 0, "slot_index": 0, "kind": "shuffle", "source_slot": None},
    {"pass_index": 0, "slot_index": 0, "kind": "wrong_slot", "source_slot": 1},
)


class StudyError(RuntimeError):
    pass


class StudyIncomplete(StudyError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(payload) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload) -> str:
    data = _json_bytes(payload)
    _atomic_write(path, data)
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, (dict, list)):
        raise StudyError(f"{path}: expected a JSON object or array")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(value) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _descriptor(path: Path, *, name: str | None = None) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    result = {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
    if name is not None:
        result["name"] = name
    return result


def _verify_descriptors(descriptors: Sequence[Mapping]) -> None:
    changed = []
    for descriptor in descriptors:
        path = Path(descriptor["path"])
        if not path.is_file() or _sha256(path) != descriptor["sha256"]:
            changed.append(str(path))
    if changed:
        raise StudyError("active producer or frozen input files changed: " + ", ".join(changed))


def _run_git(repository: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
    )
    return completed.stdout


def validate_shared_reservation_paths(ledger_path, ledger_lock):
    """Fail closed unless native.ledger confirms the anchored ledger and lock."""
    from . import ledger as native_ledger

    validator = getattr(native_ledger, "validate_shared_reservation_paths", None)
    if validator is None:
        raise StudyError("native.ledger.validate_shared_reservation_paths is required before GPU work")
    return validator(Path(ledger_path), Path(ledger_lock))


def git_snapshot(repository) -> dict:
    """Require a clean commit and return the tracked native producer set."""
    repository = Path(repository).resolve()
    commit = _run_git(repository, ["rev-parse", "HEAD"]).strip()
    status = _run_git(repository, ["status", "--porcelain=v1", "--untracked-files=all"])
    if status.strip():
        raise StudyError("the study requires a clean committed repository; git status is nonempty")
    tracked = _run_git(
        repository,
        ["ls-files", "-z", "--", "research/ti_tnn/native", "research/ti_tnn/scaling/ledger.py"],
    ).split("\0")
    return {
        "root": str(repository),
        "commit": commit,
        "clean": True,
        "tracked_files": sorted(path for path in tracked if path),
    }


def _native_directory() -> Path:
    return Path(__file__).resolve().parent


def _repository_root() -> Path:
    return _native_directory().parents[2]


def _code_files(repository: Path) -> list[Path]:
    native = repository / "research" / "ti_tnn" / "native"
    files = sorted(native.glob("*.py"))
    files.append(repository / "research" / "ti_tnn" / "scaling" / "ledger.py")
    if not all(path.is_file() for path in files):
        raise StudyError("native producer code is incomplete")
    return files


def _native_code_identity(repository: Path) -> str:
    root = repository / "research" / "ti_tnn" / "native"
    files = sorted(root.glob("*.py"))
    scaling_ledger = root.parent / "scaling" / "ledger.py"
    if scaling_ledger.is_file():
        files.append(scaling_ledger)
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root.parent)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _repository_binding(repository: Path, snapshot_fn) -> tuple[dict, list[dict]]:
    snapshot = snapshot_fn(repository)
    if not snapshot.get("clean") or not snapshot.get("commit"):
        raise StudyError("repository snapshot did not establish a clean commit")
    tracked = set(snapshot.get("tracked_files", ()))
    descriptors = []
    for path in _code_files(repository):
        relative = str(path.relative_to(repository))
        if tracked and relative not in tracked:
            raise StudyError(f"active producer {relative} is not committed")
        descriptors.append(_descriptor(path, name=relative))
    binding = {
        "root": str(repository.resolve()),
        "commit": snapshot["commit"],
        "clean": True,
        "native_code_identity": _native_code_identity(repository),
        "code_files": descriptors,
    }
    return binding, descriptors


def _validate_train_hyperparameters(overrides: Mapping | None) -> dict:
    unknown = set(overrides or ()) - set(DEFAULT_TRAIN)
    if unknown:
        raise StudyError(f"unknown fixed training hyperparameters: {sorted(unknown)}")
    result = {**DEFAULT_TRAIN, **dict(overrides or {})}
    for name in ("batch_size", "checkpoint_every", "dev_every", "dev_batches"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StudyError(f"{name} must be a positive integer")
    if len(result["betas"]) != 2:
        raise StudyError("betas must contain exactly two values")
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise StudyError(f"{key} must be finite")
    if result["warmup_fraction"] != DEFAULT_TRAIN["warmup_fraction"] \
            or result["final_lr_fraction"] != DEFAULT_TRAIN["final_lr_fraction"]:
        raise StudyError("profiled scheduler fractions are fixed at warmup 0.02 and final LR fraction 0.1")
    return result


class PreparedData:
    """Memory-mapped prepared inputs and exact sampler-prefix planning on CPU."""

    def __init__(self, root: Path, *, batch_size: int, data_seed: int, dev_sequence_limit: int) -> None:
        from .data import ARRAY_NAMES, PackedDataset
        from .sampler import identity_hash
        from .train import make_sampler

        if isinstance(dev_sequence_limit, bool) or not isinstance(dev_sequence_limit, int) or dev_sequence_limit <= 0:
            raise StudyError("dev_sequence_limit must be a positive integer")
        self.root = Path(root).resolve()
        self.batch_size = batch_size
        self.data_seed = data_seed
        self.train = PackedDataset(self.root, "train")
        self.dev = PackedDataset(self.root, "dev")
        self._make_sampler = make_sampler
        self._identity_hash = identity_hash
        template = make_sampler(self.train, batch_size, data_seed)
        dev_template = make_sampler(self.dev, batch_size, data_seed)
        self.train_data_identity = template.data_identity
        self.dev_data_identity = dev_template.data_identity
        self.order_identity = template.order_identity
        self.dev_source_sequences = len(self.dev)
        selected = min(self.dev_source_sequences, dev_sequence_limit)
        self.dev_sequences = []
        for index in range(selected):
            row = {name: self.dev.arrays[name][index] for name in ARRAY_NAMES}
            self.dev_sequences.append({"index": index, "sha256": identity_hash(row)})
        self.dev_loss_tokens = sum(
            int((self.dev.arrays["labels"][index][1:] != -100).sum()) for index in range(selected)
        )
        if not self.dev_sequences or self.dev_loss_tokens <= 0:
            raise StudyError("the bounded frozen dev sample has no shifted loss targets")

    def _record(self, sampler, digest: hashlib.sha256, requested: int) -> dict:
        return {
            "requested_loss_tokens": requested,
            "actual_loss_tokens": sampler.consumed_loss_tokens,
            "planned_updates": sampler.batches_consumed,
            "terminal_cursor": sampler.cursor,
            "order_identity": sampler.order_identity,
            "train_data_identity": sampler.data_identity,
            "order_prefix_sha256": digest.hexdigest(),
            "token_rule": "labels[:, 1:] != -100; whole batches stop at the first count >= requested budget",
        }

    def plan(self, requested: int) -> dict:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
            raise StudyError("requested loss-token budget must be a positive integer")
        sampler = self._make_sampler(self.train, self.batch_size, self.data_seed)
        digest = hashlib.sha256()
        while sampler.consumed_loss_tokens < requested:
            batch = sampler.next_batch()
            digest.update(batch.batch_identity.encode())
        return self._record(sampler, digest, requested)

    def plan_updates(self, updates: int) -> dict:
        sampler = self._make_sampler(self.train, self.batch_size, self.data_seed)
        digest = hashlib.sha256()
        for _ in range(updates):
            batch = sampler.next_batch()
            digest.update(batch.batch_identity.encode())
        return self._record(sampler, digest, sampler.consumed_loss_tokens)


def _jsonl(path: Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise StudyError(f"{path}: blank JSONL line {line_number}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise StudyError(f"{path}: line {line_number} is not an object")
            records.append(record)
    return records


def _sample_identities(path: Path, *, limit: int) -> tuple[list[dict], int]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise StudyError("frozen generation limits must be positive integers")
    result = []
    seen = set()
    source_examples = 0
    for index, record in enumerate(_jsonl(path)):
        source_examples += 1
        for field in ("example_id", "prompt", "target_text"):
            if not isinstance(record.get(field), str):
                raise StudyError(f"{path}: every sample needs string {field}")
        if record["example_id"] in seen:
            raise StudyError(f"{path}: duplicate example_id {record['example_id']!r}")
        seen.add(record["example_id"])
        if index < limit:
            result.append({
                "index": index,
                "example_id": record["example_id"],
                "record_sha256": _identity(record),
                "prompt_sha256": hashlib.sha256(record["prompt"].encode()).hexdigest(),
                "target_text_sha256": hashlib.sha256(record["target_text"].encode()).hexdigest(),
            })
    return result, source_examples


def _freeze_generation(data: Path, *, structured_limit: int, code_limit: int) -> dict:
    from .evaluate import load_structured_records

    load_structured_records(data / "structured")
    structured = {}
    for split in STRUCTURED_SPLITS:
        path = data / "structured" / f"{split}.jsonl"
        samples, source_examples = _sample_identities(path, limit=structured_limit)
        if not samples:
            raise StudyError(f"{path}: the bounded frozen generation sample is empty")
        structured[split] = {
            **_descriptor(path),
            "population": "bounded_frozen_sample",
            "selection": f"first {structured_limit} records in saved JSONL order",
            "source_examples": source_examples,
            "samples": samples,
        }
    code_path = data / "code_generation.jsonl"
    code_samples, code_source_examples = _sample_identities(code_path, limit=code_limit)
    if not code_samples:
        raise StudyError(f"{code_path}: the bounded frozen code sample is empty")
    return {
        "population": "bounded_frozen_sample",
        "structured": structured,
        "code": {
            **_descriptor(code_path),
            "population": "bounded_frozen_sample",
            "selection": f"first {code_limit} records in saved JSONL order",
            "source_examples": code_source_examples,
            "samples": code_samples,
        },
    }


def _freeze_replay(path: Path) -> dict:
    from .reconstruct import load_replay_spec
    from .sampler import identity_hash

    spec = load_replay_spec(path)
    teacher = identity_hash(spec["teacher_forced"])
    prompts = [
        {
            "example_id": prompt.get("example_id"),
            "input_ids": prompt["input_ids"].cpu().tolist(),
            "positions": prompt["positions"].cpu().tolist(),
            "sha256": identity_hash({"input_ids": prompt["input_ids"], "positions": prompt["positions"]}),
        }
        for prompt in spec["generation_prompts"]
    ]
    return {
        **_descriptor(path),
        "teacher_forced_sha256": teacher,
        "generation_prompts": prompts,
        "max_new_tokens": spec["max_new_tokens"],
        "eos_token_id": spec.get("eos_token_id"),
    }


def _number(values: Mapping, key: str, *, positive: bool = False) -> float:
    if key not in values or isinstance(values[key], bool) or not isinstance(values[key], (int, float)):
        raise StudyError(f"engineering profile needs numeric {key}")
    result = float(values[key])
    if not math.isfinite(result) or result < 0 or (positive and result <= 0):
        raise StudyError(f"engineering profile {key} has an invalid value {values[key]!r}")
    return result


def _profile_projection(raw: Mapping) -> dict:
    if raw.get("format") == "native-profile-v1":
        if raw.get("status") != "ok" or raw.get("dry_run") or not raw.get("timing_valid") \
                or not raw.get("valid_for_projection") or not raw.get("correctness", {}).get("valid"):
            raise StudyError("only successful, cache-validated CUDA engineering profiles can project the study")
        if raw.get("config", {}).get("name") != "engineering":
            raise StudyError("native-profile-v1 feasibility requires engineering.json")
        training = raw["training"]
        teacher = raw["evaluation"]
        checkpoint = raw["checkpoint"]
        measured_tokens = _number(training, "actual_shifted_loss_tokens", positive=True)
        measured_seconds = _number(training, "wall_seconds", positive=True)
        measured_dev_tokens = _number(teacher, "actual_shifted_loss_tokens", positive=True)
        measured_dev_seconds = _number(teacher, "wall_seconds", positive=True)
        complete = raw.get("complete_evaluation")
        if not isinstance(complete, Mapping):
            raise StudyError(
                "native-profile-v1 needs complete_evaluation projections for structured/code generation and state interventions"
            )
        reconstruction = raw.get("reconstruction", {})
        if not complete.get("valid") or not reconstruction.get("valid"):
            raise StudyError("complete evaluation and reconstruction must be valid for projection")
        code = complete.get("components", {}).get("code_generation", {})
        return {
            "train_fixed_seconds": _number(raw["setup"], "fixed_seconds"),
            "loss_tokens_per_second": measured_tokens / measured_seconds,
            "seconds_per_update": 0.0,
            "seconds_per_checkpoint": _number(checkpoint, "wall_seconds", positive=True),
            "checkpoint_bytes_each": _number(checkpoint, "bytes", positive=True),
            "seconds_per_dev_token": measured_dev_seconds / measured_dev_tokens,
            "measured_dev_tokens": measured_dev_tokens,
            "evaluation": {
                "teacher_forced": _number(complete, "teacher_forced_seconds"),
                "structured_generation": _number(complete, "structured_generation_seconds"),
                "code_generation": _number(code, "measured_seconds"),
                "state_interventions": _number(complete, "state_interventions_seconds"),
            },
            "reconstruction_seconds": _number(reconstruction, "seconds"),
            "code_projection": "bounded measured_seconds only; full-source extrapolation is never used",
            "source_format": "native-profile-v1",
        }
    measured = raw.get("measured", raw)
    if not isinstance(measured, Mapping):
        raise StudyError("engineering profile measured section must be an object")
    training = measured.get("training")
    evaluation = measured.get("complete_evaluation")
    if not isinstance(training, Mapping) or not isinstance(evaluation, Mapping):
        raise StudyError("engineering profile needs measured training and complete_evaluation sections")
    reconstruction = measured.get("reconstruction", {})
    if not isinstance(reconstruction, Mapping):
        raise StudyError("measured.reconstruction must be an object")
    return {
        "train_fixed_seconds": _number(training, "fixed_seconds"),
        "loss_tokens_per_second": _number(training, "loss_tokens_per_second", positive=True),
        "seconds_per_update": _number(training, "seconds_per_update"),
        "seconds_per_checkpoint": _number(training, "seconds_per_checkpoint"),
        "checkpoint_bytes_each": _number(training, "checkpoint_bytes_each"),
        "seconds_per_complete_dev_pass": _number(training, "seconds_per_complete_dev_pass"),
        "evaluation": {
            name: _number(evaluation, f"{name}_seconds")
            for name in ("teacher_forced", "structured_generation", "code_generation", "state_interventions")
        },
        "reconstruction_seconds": _number(reconstruction, "seconds"),
        "source_format": "study-measured-v1",
    }


def _profile_provenance(raw: Mapping) -> dict:
    """Retain runtime/identity evidence, never per-arm quality outcomes, in the freeze."""
    if raw.get("format") == "native-profile-v1":
        return {
            key: raw[key]
            for key in (
                "format", "mode", "status", "dry_run", "timing_valid", "valid_for_projection",
                "config", "data", "code", "execution", "setup", "correctness", "complete_evaluation",
                "reconstruction", "reservation",
            )
            if key in raw
        } | {
            "training": {
                key: raw["training"][key]
                for key in ("measured_steps", "actual_shifted_loss_tokens", "wall_seconds", "warmup_steps_excluded")
                if key in raw["training"]
            },
            "evaluation": {
                key: raw["evaluation"][key]
                for key in (
                    "teacher_forced", "batches", "batch_size", "actual_shifted_loss_tokens",
                    "data_loading_seconds", "forward_seconds", "wall_seconds",
                )
                if key in raw["evaluation"]
            },
            "checkpoint": {
                key: raw["checkpoint"][key]
                for key in ("wall_seconds", "bytes", "real_write")
                if key in raw["checkpoint"]
            },
        }
    measured = raw.get("measured", {})
    training = measured.get("training", {})
    complete = measured.get("complete_evaluation", {})
    reconstruction = measured.get("reconstruction", {})
    return {
        "mode": raw.get("mode"),
        "execution": dict(raw.get("execution", {})),
        "measured": {
            "training": {
                key: training[key]
                for key in (
                    "fixed_seconds", "loss_tokens_per_second", "seconds_per_update",
                    "seconds_per_checkpoint", "checkpoint_bytes_each", "seconds_per_complete_dev_pass",
                )
                if key in training
            },
            "complete_evaluation": {
                key: complete[key]
                for key in (
                    "teacher_forced_seconds", "structured_generation_seconds",
                    "code_generation_seconds", "state_interventions_seconds",
                )
                if key in complete
            },
            "reconstruction": {"seconds": reconstruction.get("seconds")},
        },
    }


def _load_profiles(paths: Sequence[Path]) -> tuple[dict, list[dict]]:
    expanded = []
    descriptors = []
    for supplied in paths:
        path = Path(supplied).resolve()
        candidates = sorted(path.glob("*.json")) if path.is_dir() else [path]
        if not candidates:
            raise StudyError(f"no engineering profile JSONs found in {path}")
        for candidate in candidates:
            raw = _read_json(candidate)
            if not isinstance(raw, Mapping):
                raise StudyError(f"{candidate}: profile must be an object")
            descriptor = _descriptor(candidate)
            descriptors.append(descriptor)
            nested = raw.get("profiles")
            if isinstance(nested, Mapping):
                for mode, values in nested.items():
                    if not isinstance(values, Mapping):
                        raise StudyError(f"{candidate}: profile {mode} must be an object")
                    expanded.append((mode, values, descriptor))
                continue
            mode = raw.get("mode")
            if mode is not None:
                expanded.append((mode, raw, descriptor))
                continue
            modes = raw.get("modes")
            if isinstance(modes, Mapping) and all(
                isinstance(values, Mapping) and ("measured" in values or "training" in values)
                for values in modes.values()
            ):
                for name, values in modes.items():
                    expanded.append((name, values, descriptor))
                continue
            raise StudyError(f"{candidate}: profile needs a mode or a profiles/modes mapping")
    profiles = {}
    for mode, raw, descriptor in expanded:
        if mode not in MODES:
            raise StudyError(f"unknown engineering profile mode {mode!r}")
        if mode in profiles:
            raise StudyError(f"duplicate engineering profile for {mode}")
        profiles[mode] = {
            "mode": mode,
            "source": descriptor,
            "measured": _profile_provenance(raw),
            "projection": _profile_projection(raw),
        }
    missing = set(MODES) - set(profiles)
    if missing:
        raise StudyError(f"missing measured engineering profiles: {sorted(missing)}")
    return profiles, descriptors


def _orchestration_only_profile_commit(repository: Mapping, profile_commit: str, reported_identity: str) -> dict | None:
    if "root" not in repository:
        return None
    root = Path(repository["root"])
    current = repository["commit"]
    if profile_commit == current:
        return None
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", profile_commit, current],
        cwd=root, capture_output=True, check=False,
    )
    if ancestor.returncode:
        return None
    changed = subprocess.run(
        ["git", "diff", "--name-only", profile_commit, current, "--", "research/ti_tnn/native"],
        cwd=root, capture_output=True, text=True, check=False,
    )
    allowed = [
        "research/ti_tnn/native/evaluate.py",
        "research/ti_tnn/native/reconstruct.py",
        "research/ti_tnn/native/study.py",
    ]
    if changed.returncode or changed.stdout.splitlines() != allowed:
        return None
    native = root / "research" / "ti_tnn" / "native"
    files = sorted(native.glob("*.py"))
    scaling_ledger = native.parent / "scaling" / "ledger.py"
    if scaling_ledger.is_file():
        files.append(scaling_ledger)
    digest = hashlib.sha256()
    for path in files:
        relative_repository = path.relative_to(root)
        if relative_repository.as_posix() in allowed:
            content = subprocess.run(
                ["git", "show", f"{profile_commit}:{relative_repository.as_posix()}"],
                cwd=root, capture_output=True, check=False,
            )
            if content.returncode:
                return None
            payload = content.stdout
        else:
            payload = path.read_bytes()
        relative = path.relative_to(native.parent)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    if digest.hexdigest() != reported_identity:
        return None
    return {
        "profile_commit": profile_commit,
        "frozen_commit": current,
        "changed_native_files": allowed,
        "reason": "instrumentation-only change; model, trainer, profiler and data producer bytes are identical; reconstruction projection uses the original FP32 serving measurement",
    }


def _validate_native_profile_bindings(
    profiles: Mapping,
    engineering_config: Path,
    data: Path,
    train: Mapping,
    evaluation: Mapping,
    repository: Mapping,
    identities: Mapping | None = None,
) -> None:
    from .data import ARRAY_NAMES as ARRAY_NAMES_FOR_STUDY

    base = NativeConfig.from_json(engineering_config)
    expected_config_sha256 = _sha256(engineering_config)
    expected_manifest_sha256 = _sha256(data / "manifest.json")
    recipe_names = (
        "batch_size", "learning_rate", "weight_decay", "betas", "eps", "grad_clip",
        "lambda_balance", "lambda_route", "lambda_state",
    )
    for mode, profile in profiles.items():
        raw = profile["measured"]
        if raw.get("format") != "native-profile-v1":
            execution = raw.get("execution", {})
            expected_limits = {
                "batch_size": train["batch_size"],
                "evaluation_batch_size": evaluation["batch_size"],
                "dev_batches": evaluation["dev_batches"],
                "limit_per_split": evaluation["limit_per_split"],
                "code_limit": evaluation["code_limit"],
                "max_new_tokens": evaluation["max_new_tokens"],
                "code_max_new_tokens": evaluation["code_max_new_tokens"],
                "eos_token_id": evaluation["eos_token_id"],
                "use_cache": evaluation["use_cache"],
            }
            for name, wanted in expected_limits.items():
                if execution.get(name) != wanted:
                    raise StudyError(f"{mode} profile execution limit {name} does not match the frozen protocol")
            continue
        profile_code = raw.get("code", {})
        profile_commit = profile_code.get("git_commit_at_attempt_start")
        if profile_commit == repository["commit"]:
            if profile_code.get("identity") != repository["native_code_identity"]:
                raise StudyError(f"{mode} profile code identity or attempt commit differs from the frozen repository")
        else:
            compatibility = _orchestration_only_profile_commit(
                repository, profile_commit, profile_code.get("identity", "")
            )
            if compatibility is None:
                raise StudyError(f"{mode} profile code identity or attempt commit differs from the frozen repository")
            profile["commit_equivalence"] = compatibility
        config = raw.get("config", {})
        if config.get("sha256") != expected_config_sha256:
            raise StudyError(f"{mode} engineering profile used a different source config")
        if config.get("model") != replace(base, mode=mode).to_dict():
            raise StudyError(f"{mode} engineering profile used different model dimensions")
        recipe = config.get("profile", {})
        for name in recipe_names:
            wanted = list(train[name]) if name == "betas" else train[name]
            if recipe.get(name) != wanted:
                raise StudyError(f"{mode} engineering profile changed training recipe field {name}")
        if recipe.get("evaluation_batches") != train["dev_batches"] \
                or recipe.get("evaluation_batch_size") != train["batch_size"]:
            raise StudyError(f"{mode} profile training-dev execution limits do not match the frozen protocol")
        for name in ("max_new_tokens", "code_max_new_tokens", "eos_token_id"):
            if recipe.get(name) != evaluation[name]:
                raise StudyError(f"{mode} profile decoding field {name} does not match the frozen protocol")
        if recipe.get("requested_dtype") != "bfloat16" or raw.get("execution", {}).get("mode") != "eager":
            raise StudyError(f"{mode} profile must match eager bfloat16 native.train autocast")
        components = raw.get("complete_evaluation", {}).get("components", {})
        teacher = components.get("teacher_forced", {})
        structured = components.get("structured_generation", {})
        code = components.get("code_generation", {})
        if teacher.get("batch_size") != evaluation["batch_size"] \
                or teacher.get("complete_evaluation_max_batches") != evaluation["dev_batches"] \
                or teacher.get("max_batches") != evaluation["dev_batches"]:
            raise StudyError(f"{mode} profile complete teacher-forced limits do not match the frozen protocol")
        if structured.get("limit_per_split") != evaluation["limit_per_split"] \
                or structured.get("effective_limit_per_split") != evaluation["limit_per_split"]:
            raise StudyError(f"{mode} profile structured limit does not match the frozen protocol")
        if code.get("limit") != evaluation["code_limit"]:
            raise StudyError(f"{mode} profile code limit does not match the frozen protocol")
        for name, component, component_bound, evaluation_bound in (
            ("structured", structured, "max_new_tokens", "max_new_tokens"),
            ("code", code, "max_new_tokens", "code_max_new_tokens"),
        ):
            if component.get(component_bound) != evaluation[evaluation_bound] \
                    or component.get("eos_token_id") != evaluation["eos_token_id"] \
                    or component.get("use_cache") is not evaluation["use_cache"]:
                raise StudyError(f"{mode} profile {name} decoding limits do not match the frozen protocol")
        if identities is not None:
            sampler = raw.get("execution", {}).get("sampler", {})
            expected_sampler = {
                "factory": "native.train.make_sampler",
                "data_seed": identities["development_seed"],
                "data_identity": identities["train_data_identity"],
                "dev_data_identity": identities["dev_data_identity"],
                "order_identity": identities["order_identity"],
            }
            for name, wanted in expected_sampler.items():
                if sampler.get(name) != wanted:
                    raise StudyError(f"{mode} profile sampler {name} does not match the frozen data order")
            if recipe.get("seed") != identities["development_seed"] \
                    or recipe.get("data_seed") != identities["development_seed"]:
                raise StudyError(f"{mode} profile seed differs from the frozen development seed")
            measured_data = raw.get("data", {})
            for name in ("train_data_identity", "dev_data_identity", "order_identity"):
                if measured_data.get(name) != identities[name]:
                    raise StudyError(f"{mode} profile {name} does not match the frozen data")
            for name, wanted in (
                ("tokenizer_sha256", identities["tokenizer_sha256"]),
                ("code_generation_sha256", identities["code_generation_sha256"]),
                ("reconstruction_replay_sha256", identities["replay_sha256"]),
            ):
                if recipe.get(name) != wanted:
                    raise StudyError(f"{mode} profile {name} does not match the frozen evaluation inputs")
            if recipe.get("structured_sha256") != identities["structured_sha256"]:
                raise StudyError(f"{mode} profile structured inputs differ from the frozen generation splits")
            if raw.get("reconstruction", {}).get("replay_spec_sha256") != identities["replay_sha256"] \
                    or components.get("state_interventions", {}).get("replay_spec_sha256") != identities["replay_sha256"]:
                raise StudyError(f"{mode} profile replay identity differs between evaluation and reconstruction")
        if measured_data.get("manifest_sha256") != expected_manifest_sha256:
            raise StudyError(f"{mode} engineering profile used a different prepared manifest")
        for split in ("train", "dev"):
            for name in ARRAY_NAMES_FOR_STUDY:
                expected = _sha256(data / split / f"{name}.npy")
                actual = measured_data.get("splits", {}).get(split, {}).get("arrays", {}).get(name, {}).get("sha256")
                if actual != expected:
                    raise StudyError(f"{mode} engineering profile changed {split}/{name}.npy")


def project_arm(
    profile: Mapping,
    plan: Mapping,
    train: Mapping,
    *,
    dev_loss_tokens: int | None = None,
    reserve_fraction: float = 0.0,
) -> dict:
    """Project all training, checkpoints, complete evaluation, and replay hours."""
    projection = profile["projection"] if "projection" in profile else _profile_projection(profile)
    updates = plan["planned_updates"]
    tokens = plan["actual_loss_tokens"]
    periodic_checkpoints = (updates - 1) // train["checkpoint_every"]
    periodic_dev = (updates - 1) // train["dev_every"]
    checkpoints = 2 + periodic_checkpoints
    dev_passes = 1 + periodic_dev
    if "seconds_per_dev_token" in projection:
        dev_tokens = dev_loss_tokens or projection["measured_dev_tokens"]
        seconds_per_dev_pass = dev_tokens * projection["seconds_per_dev_token"]
    else:
        seconds_per_dev_pass = projection["seconds_per_complete_dev_pass"]
    dev_seconds = dev_passes * seconds_per_dev_pass
    checkpoint_seconds = checkpoints * projection["seconds_per_checkpoint"]
    train_seconds = (
        projection["train_fixed_seconds"]
        + tokens / projection["loss_tokens_per_second"]
        + updates * projection["seconds_per_update"]
        + checkpoint_seconds
        + dev_seconds
    )
    evaluation_components = dict(projection["evaluation"])
    evaluation_seconds = sum(evaluation_components.values())
    reconstruction_seconds = projection["reconstruction_seconds"]
    total_seconds = train_seconds + evaluation_seconds + reconstruction_seconds
    margin = 1.0 + reserve_fraction
    return {
        "mode": profile["mode"],
        "actual_loss_tokens": tokens,
        "planned_updates": updates,
        "checkpoints": {
            "initial": 1,
            "periodic": periodic_checkpoints,
            "final": 1,
            "total": checkpoints,
            "seconds": checkpoint_seconds,
            "projected_bytes": checkpoints * projection["checkpoint_bytes_each"],
        },
        "complete_train_dev_passes": dev_passes,
        "seconds_per_complete_dev_pass": seconds_per_dev_pass,
        "train_seconds": train_seconds,
        "complete_evaluation_seconds": evaluation_seconds,
        "complete_evaluation_components_seconds": evaluation_components,
        "reconstruction_seconds": reconstruction_seconds,
        "total_seconds": total_seconds,
        "total_gpu_hours": total_seconds / 3600.0,
        "planning_margin": reserve_fraction,
        "projected_reservation_gpu_hours": total_seconds * margin / 3600.0,
        "stage_measured_hours": {
            "train": train_seconds / 3600.0,
            "evaluate": evaluation_seconds / 3600.0,
            "reconstruct": reconstruction_seconds / 3600.0,
        },
        "stage_projected_hours": {
            "train": train_seconds * margin / 3600.0,
            "evaluate": evaluation_seconds * margin / 3600.0,
            "reconstruct": reconstruction_seconds * margin / 3600.0,
        },
        "accounting": "disjoint measured fixed/token/update costs plus every checkpoint/dev pass, complete evaluation/interventions, and reconstruction; reservation projections include the planning margin",
    }


def _project_study(
    profiles: Mapping,
    plan: Mapping,
    train: Mapping,
    *,
    dev_loss_tokens: int,
    reserve_fraction: float,
) -> dict:
    arms = {
        mode: project_arm(
            profiles[mode], plan, train,
            dev_loss_tokens=dev_loss_tokens, reserve_fraction=reserve_fraction,
        )
        for mode in MODES
    }
    raw_hours = sum(arm["total_gpu_hours"] for arm in arms.values())
    return {
        "arms": arms,
        "total_gpu_hours": raw_hours,
        "total_checkpoint_bytes": sum(
            arm["checkpoints"]["projected_bytes"] for arm in arms.values()
        ),
        "planning_margin": reserve_fraction,
        "projected_reservation_hours": raw_hours * (1.0 + reserve_fraction),
    }


def _data_files(data: Path) -> list[dict]:
    files = sorted(path for path in data.rglob("*") if path.is_file())
    if not files:
        raise StudyError(f"prepared data directory {data} is empty")
    return [_descriptor(path, name=str(path.relative_to(data))) for path in files]


def _write_infeasible(output: Path, report: Mapping) -> dict:
    result = {"format": STUDY_FORMAT, "status": "infeasible", **report}
    _write_json(output / "infeasible.json", result)
    return result


def freeze_study(
    output,
    *,
    data,
    tokenizer,
    replay_spec,
    engineering_profiles: Sequence,
    engineering_config=None,
    serious_config=None,
    repository=None,
    development_seed: int = DEVELOPMENT_SEED,
    provisional_loss_tokens: int = PROVISIONAL_LOSS_TOKENS,
    reserve_fraction: float = RESERVE_FRACTION,
    available_gpu_hours: float | None = None,
    available_storage_bytes: int | None = None,
    ledger_path=None,
    ledger_lock=None,
    train_hyperparameters: Mapping | None = None,
    evaluation: Mapping | None = None,
    reservation_path_validator=None,
    git_snapshot_fn=git_snapshot,
) -> dict:
    """Freeze design, samples, exact token endpoint, and measured feasibility before outcomes."""
    output = Path(output).resolve()
    data = Path(data).resolve()
    repository = Path(repository).resolve() if repository else _repository_root()
    engineering_config = Path(engineering_config).resolve() if engineering_config else _native_directory() / "configs" / "engineering.json"
    serious_config = Path(serious_config).resolve() if serious_config else _native_directory() / "configs" / "serious.json"
    if output.exists() and any(output.iterdir()):
        raise StudyError("freeze requires a new or empty output directory; existing outcomes are never inspected")
    output.mkdir(parents=True, exist_ok=True)
    if isinstance(development_seed, bool) or not isinstance(development_seed, int):
        raise StudyError("development_seed must be an integer")
    if isinstance(provisional_loss_tokens, bool) or not isinstance(provisional_loss_tokens, int) or provisional_loss_tokens <= 0:
        raise StudyError("provisional_loss_tokens must be a positive integer")
    if not isinstance(reserve_fraction, (int, float)) or isinstance(reserve_fraction, bool) \
            or not math.isfinite(reserve_fraction) or not RESERVE_FRACTION <= reserve_fraction < 1:
        raise StudyError("reserve_fraction must be finite and at least 0.20")
    train = _validate_train_hyperparameters(train_hyperparameters)
    evaluation_rules = {**DEFAULT_EVALUATION, **dict(evaluation or {})}
    if set(evaluation_rules) != set(DEFAULT_EVALUATION):
        raise StudyError("unknown evaluation scoring rule")
    for name in ("batch_size", "dev_batches", "limit_per_split", "code_limit"):
        value = evaluation_rules[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StudyError(f"evaluation {name} must be a positive integer")
    if train["dev_batches"] != evaluation_rules["dev_batches"] \
            or train["batch_size"] != evaluation_rules["batch_size"]:
        raise StudyError("train and external evaluation must share the frozen dev batch size and batch limit")
    binding, code_descriptors = _repository_binding(repository, git_snapshot_fn)
    data_descriptors = _data_files(data)
    tokenizer_descriptor = _descriptor(tokenizer, name="tokenizer.json")
    replay = _freeze_replay(Path(replay_spec).resolve())
    generation = _freeze_generation(
        data,
        structured_limit=evaluation_rules["limit_per_split"],
        code_limit=evaluation_rules["code_limit"],
    )
    profiles, profile_descriptors = _load_profiles([Path(path) for path in engineering_profiles])
    engineering_raw = _read_json(engineering_config)
    serious_raw = _read_json(serious_config)
    if engineering_raw.get("name") != "engineering" or serious_raw.get("name") != "serious":
        raise StudyError("engineering/serious configs have unexpected profile names")
    for mode in MODES:
        if mode not in engineering_raw.get("modes", {}) or mode not in serious_raw.get("modes", {}):
            raise StudyError("engineering and serious configs must contain all four modes")
    source_descriptors = [
        _descriptor(engineering_config, name="engineering.json"),
        _descriptor(serious_config, name="serious.json"),
    ]
    prepared = PreparedData(
        data,
        batch_size=train["batch_size"],
        data_seed=development_seed,
        dev_sequence_limit=evaluation_rules["dev_batches"] * evaluation_rules["batch_size"],
    )
    profile_identities = {
        "development_seed": development_seed,
        "train_data_identity": prepared.train_data_identity,
        "dev_data_identity": prepared.dev_data_identity,
        "order_identity": prepared.order_identity,
        "tokenizer_sha256": tokenizer_descriptor["sha256"],
        "replay_sha256": replay["sha256"],
        "structured_sha256": {
            split: generation["structured"][split]["sha256"] for split in STRUCTURED_SPLITS
        },
        "code_generation_sha256": generation["code"]["sha256"],
    }
    _validate_native_profile_bindings(
        profiles, engineering_config, data, train, evaluation_rules, binding, profile_identities
    )
    provisional_plan = prepared.plan(provisional_loss_tokens)
    minimum_updates = max(3, train["dev_every"] + 1)
    minimum_plan = prepared.plan_updates(minimum_updates)
    if ledger_path is None:
        from . import ledger as native_ledger

        ledger_path = native_ledger.shared_runs() / "gpu_ledger.jsonl"
    ledger_path = Path(ledger_path).resolve()
    ledger_lock = Path(ledger_lock).resolve() if ledger_lock else ledger_path.with_name("gpu.lock")
    ledger_usage = None
    if available_gpu_hours is None:
        from . import ledger as native_ledger

        (reservation_path_validator or validate_shared_reservation_paths)(ledger_path, ledger_lock)
        ledger_usage = native_ledger.usage(ledger_path)
        available_gpu_hours = ledger_usage["native_remaining_hours"]
    if not isinstance(available_gpu_hours, (int, float)) or isinstance(available_gpu_hours, bool) \
            or not math.isfinite(available_gpu_hours) or available_gpu_hours < 0:
        raise StudyError("available_gpu_hours must be a finite nonnegative measured allowance")
    if available_storage_bytes is None:
        available_storage_bytes = shutil.disk_usage(output).free
    if isinstance(available_storage_bytes, bool) or not isinstance(available_storage_bytes, int) \
            or available_storage_bytes < 0:
        raise StudyError("available_storage_bytes must be a nonnegative integer")
    available_hours = min(float(available_gpu_hours), 12.0)
    raw_hour_cap = available_hours / (1.0 + reserve_fraction)
    projection_inputs = {
        "dev_loss_tokens": prepared.dev_loss_tokens,
        "reserve_fraction": reserve_fraction,
    }
    provisional_projection = _project_study(profiles, provisional_plan, train, **projection_inputs)
    minimum_projection = _project_study(profiles, minimum_plan, train, **projection_inputs)

    def fits(projection: Mapping) -> bool:
        return (
            projection["projected_reservation_hours"] <= available_hours
            and projection["total_checkpoint_bytes"] <= available_storage_bytes
        )

    feasibility = {
        "available_native_gpu_hours": float(available_gpu_hours),
        "available_hours_within_native_cap": available_hours,
        "reserve_fraction": reserve_fraction,
        "planning_multiplier": 1.0 + reserve_fraction,
        "maximum_raw_gpu_hours_after_reserve": raw_hour_cap,
        "minimum_updates_for_useful_curve": minimum_updates,
        "minimum_plan": minimum_plan,
        "minimum_projection": minimum_projection,
        "provisional_plan": provisional_plan,
        "provisional_projection": provisional_projection,
        "storage": {
            "available_bytes": available_storage_bytes,
            "policy": "the complete frozen all-checkpoints schedule must fit before any arm starts",
            "provisional_checkpoint_bytes": provisional_projection["total_checkpoint_bytes"],
            "minimum_checkpoint_bytes": minimum_projection["total_checkpoint_bytes"],
        },
        "ledger_usage_at_freeze": ledger_usage,
        "ledger_path": str(ledger_path),
        "ledger_sha256_at_freeze": _sha256(ledger_path) if ledger_path.is_file() else None,
        "basis": "measured engineering profiles only; raw complete cost is multiplied by 1 + planning margin; no comparative arm outcome is an input",
    }
    chosen_plan = provisional_plan
    chosen_projection = provisional_projection
    reduction = None
    if minimum_plan["actual_loss_tokens"] > provisional_plan["actual_loss_tokens"]:
        return _write_infeasible(output, {
            "reason": "the requested budget is smaller than the minimum useful complete curve",
            "feasibility": feasibility,
            "profiles": profiles,
        })
    if not fits(provisional_projection):
        if not fits(minimum_projection):
            return _write_infeasible(output, {
                "reason": "measured GPU/storage projections cannot fund even the minimum complete curve with the declared reserve",
                "feasibility": feasibility,
                "profiles": profiles,
            })
        low = minimum_plan["requested_loss_tokens"]
        high = provisional_loss_tokens
        best_plan = minimum_plan
        best_projection = minimum_projection
        while low <= high:
            middle = (low + high) // 2
            candidate_plan = prepared.plan(middle)
            candidate_projection = _project_study(profiles, candidate_plan, train, **projection_inputs)
            if fits(candidate_projection):
                best_plan, best_projection = candidate_plan, candidate_projection
                low = middle + 1
            else:
                high = middle - 1
        chosen_plan, chosen_projection = best_plan, best_projection
        reduction = {
            "reason": "the provisional 20M/arm projection exceeds measured GPU-hour or checkpoint-storage allowances",
            "selected_before_comparative_outcomes": True,
            "equal_actual_loss_tokens_per_arm": chosen_plan["actual_loss_tokens"],
            "provisional_projected_reservation_hours": provisional_projection["projected_reservation_hours"],
            "selected_projected_reservation_hours": chosen_projection["projected_reservation_hours"],
            "provisional_checkpoint_bytes": provisional_projection["total_checkpoint_bytes"],
            "selected_checkpoint_bytes": chosen_projection["total_checkpoint_bytes"],
        }
    if not fits(chosen_projection):
        raise AssertionError("budget search selected an infeasible GPU/storage projection")
    configs = {}
    config_descriptors = []
    base_model = NativeConfig.from_json(engineering_config)
    for mode in MODES:
        mode_config = replace(base_model, mode=mode)
        raw = dict(engineering_raw)
        raw["model"] = mode_config.to_dict()
        path = output / "configs" / f"{mode}.json"
        digest = _write_json(path, raw)
        configs[mode] = {"path": str(path), "sha256": digest, "model": mode_config.to_dict()}
        config_descriptors.append(_descriptor(path, name=f"frozen-{mode}.json"))
    scoring_rules = {
        "population": "bounded_frozen_sample",
        "full_corpus_evaluation": False,
        "teacher_forced": {
            "population": "bounded_frozen_sample",
            "selection": "first dev rows in packed order, bounded by dev_batches * batch_size",
            "shift": "logits[:, :-1] -> labels[:, 1:]",
            "sources": "source_ids[:, 1:] with micro-averaged cross entropy and next-token accuracy",
            "selected_dev_sequences": len(prepared.dev_sequences),
            "selected_dev_loss_tokens": prepared.dev_loss_tokens,
            "source_dev_sequences": prepared.dev_source_sequences,
            "batch_size": evaluation_rules["batch_size"],
            "max_batches": evaluation_rules["dev_batches"],
            "full_dev_or_full_source_claim": False,
        },
        "structured": {
            "population": "bounded_frozen_sample",
            "metric": "complete canonical final-register target text and independent strict parser",
            "all_splits": list(STRUCTURED_SPLITS),
            "selection": "first N records of each saved split",
            "limit_per_split": evaluation_rules["limit_per_split"],
            "source_examples": {
                split: generation["structured"][split]["source_examples"] for split in STRUCTURED_SPLITS
            },
            "prompt_policy": "error",
            "gold_controls_decoding": False,
            "full_source_generation_claim": False,
        },
        "code": {
            "population": "bounded_frozen_sample",
            "metric": "bounded free-generated code-suffix exact text match",
            "selection": "first N records in saved JSONL order",
            "limit": evaluation_rules["code_limit"],
            "source_examples": generation["code"]["source_examples"],
            "prompt_policy": "keep_last",
            "gold_controls_decoding": False,
            "full_source_generation_claim": False,
        },
        "reconstruction": "exact checkpoint parameters on the FP32 CPU serving path; floating-tensor relative RMS error <= 0.05 of reference RMS, exact integer routes and exact cached/full generated tokens; strict allclose and BF16 diagnostics retained separately",
    }
    active_files = (
        code_descriptors + data_descriptors + source_descriptors + profile_descriptors
        + [tokenizer_descriptor, _descriptor(replay_spec, name="replay.json")] + config_descriptors
    )
    _verify_descriptors(active_files)
    protocol = {
        "format": STUDY_FORMAT,
        "status": "frozen",
        "frozen_at": _now(),
        "development": {
            "seed": development_seed,
            "model_seed": development_seed,
            "data_seed": development_seed,
            "role": "single declared development seed; not a multi-seed scientific result",
            "arms": list(MODES),
            "baseline_order": ["dense", "shared"],
            "candidate_order": ["untyped", "typed"],
        },
        "repository": binding,
        "data": {
            "root": str(data),
            "files": data_descriptors,
            "train_data_identity": prepared.train_data_identity,
            "dev_data_identity": prepared.dev_data_identity,
            "order_identity": prepared.order_identity,
            "dev_population": "bounded_frozen_sample",
            "dev_selection": "first rows up to dev_batches * batch_size",
            "dev_sequence_limit": evaluation_rules["dev_batches"] * evaluation_rules["batch_size"],
            "dev_source_sequences": prepared.dev_source_sequences,
            "dev_sequences": prepared.dev_sequences,
            "dev_sequence_order": list(range(len(prepared.dev_sequences))),
            "dev_loss_tokens": prepared.dev_loss_tokens,
            "manifest_sha256": _sha256(data / "manifest.json"),
        },
        "configs": configs,
        "engineering_config": engineering_raw,
        "serious_config": serious_raw,
        "profiles": profiles,
        "train": train,
        "budget": {
            **chosen_plan,
            "provisional_requested_loss_tokens": provisional_loss_tokens,
            "equal_across_arms": True,
            "reduction": reduction,
            "projection": chosen_projection,
            "storage": {
                "available_bytes": available_storage_bytes,
                "required_all_checkpoint_bytes": chosen_projection["total_checkpoint_bytes"],
                "fits": chosen_projection["total_checkpoint_bytes"] <= available_storage_bytes,
                "retention": "all checkpoints in the frozen schedule",
            },
        },
        "feasibility": feasibility,
        "evaluation": {
            **evaluation_rules,
            "population": "bounded_frozen_sample",
            "full_corpus_evaluation": False,
            "scoring_rules": scoring_rules,
            "samples": generation,
            "tokenizer": tokenizer_descriptor,
            "replay": replay,
        },
        "interventions": {
            "specifications": [dict(spec) for spec in DEFAULT_INTERVENTIONS],
            "samples": [
                {"split": split, **generation["structured"][split]["samples"][0]}
                for split in ("dev", "unseen_composition")
            ],
            "api": "native.evaluate.evaluate_state_interventions",
            "limits": "sensitivity is not proof of faithful or complete slot use",
        },
        "execution": {
            "python": sys.executable,
            "working_directory": str(repository),
            "order": list(MODES),
            "stages": ["train", "evaluate", "reconstruct"],
            "train_device": "cuda",
            "evaluation_device": "cpu",
            "reconstruction_device": "cpu",
            "reconstruction_autocast_dtype": None,
            "parent_opens_reservation": False,
            "one_child_reservation_per_stage": True,
            "ledger_path": str(ledger_path),
            "ledger_lock": str(ledger_lock),
            "stop_on_failed_or_incomplete_arm": True,
            "budget_reallocation_after_freeze": False,
        },
        "active_files": active_files,
        "evidence_boundary": "bounded_frozen_sample metrics are not full-corpus estimates; single-seed engineering measurements are not multi-seed learning or composition evidence",
    }
    protocol_path = output / "protocol.json"
    protocol_sha256 = _write_json(protocol_path, protocol)
    _atomic_write(output / "protocol.sha256", f"{protocol_sha256}\n".encode())
    return {"format": STUDY_FORMAT, "status": "frozen", "protocol": protocol, "protocol_sha256": protocol_sha256}


def _load_protocol(study: Path, snapshot_fn=git_snapshot, *, verify_repository: bool = True) -> tuple[dict, str]:
    study = Path(study).resolve()
    protocol_path = study / "protocol.json"
    receipt_path = study / "protocol.sha256"
    if not protocol_path.is_file() or not receipt_path.is_file():
        if (study / "infeasible.json").is_file():
            raise StudyIncomplete("the measured study was infeasible; no runnable protocol was frozen")
        raise StudyIncomplete("no frozen protocol and hash receipt exist")
    actual = _sha256(protocol_path)
    expected = receipt_path.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise StudyError("protocol.json differs from its frozen SHA256 receipt")
    protocol = _read_json(protocol_path)
    if protocol.get("format") != STUDY_FORMAT or protocol.get("status") != "frozen":
        raise StudyError("unsupported or unfrozen study protocol")
    _verify_descriptors(protocol["active_files"])
    if verify_repository:
        snapshot = snapshot_fn(Path(protocol["repository"]["root"]))
        if not snapshot.get("clean") or snapshot.get("commit") != protocol["repository"]["commit"]:
            raise StudyError("repository commit or cleanliness differs from the frozen protocol")
        if _native_code_identity(Path(protocol["repository"]["root"])) != protocol["repository"]["native_code_identity"]:
            raise StudyError("native code identity differs from the frozen protocol")
    return protocol, actual


def reservation_accounting(ledger_path, *, reservation_id=None, mode, stage, started, output=None) -> dict:
    """Read, never open, a ledger and retain every event for one stage reservation."""
    from . import ledger as native_ledger

    path = Path(ledger_path)
    events = list(native_ledger.read(path))
    starts = [event for event in events if event.get("event") == "start"]
    selected = None
    if reservation_id:
        selected = next((event for event in reversed(starts) if event.get("reservation") == reservation_id), None)
    if selected is None:
        for event in reversed(starts):
            extra = event.get("extra", {})
            if extra.get("native_study_arm") == mode and extra.get("native_study_stage") == stage:
                selected = event
                break
            command = event.get("command", "")
            if stage == "train" and f"native.train mode={mode}" in command and output and str(output) in command:
                if event.get("t", 0) >= started - 2:
                    selected = event
                    break
    selected_id = selected.get("reservation") if selected else reservation_id
    usage = native_ledger.usage(path)
    reservation = next(
        (item for item in usage["reservations"] if item["reservation"] == selected_id), None
    ) if selected_id else None
    return {
        "ledger_path": str(path),
        "ledger_sha256": _sha256(path) if path.is_file() else None,
        "reservation_id": selected_id,
        "reservation": reservation,
        "events": [event for event in events if event.get("reservation") == selected_id] if selected_id else [],
        "aggregate": {key: value for key, value in usage.items() if key != "reservations"},
        "status": "accounted" if reservation is not None else "not_found",
    }


def _environment(protocol: Mapping) -> dict:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(protocol["repository"]["root"]) / "research" / "ti_tnn")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _completed_result(result) -> tuple[int, str, str]:
    return int(result.returncode), result.stdout or "", result.stderr or ""


def _last_json(text: str, label: str) -> dict:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise StudyError(f"{label} produced no JSON result")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise StudyError(f"{label} result is not a JSON object")
    return value


def _train_command(protocol: Mapping, mode: str, output: Path) -> list[str]:
    train = protocol["train"]
    arguments = [
        "--config", protocol["configs"][mode]["path"],
        "--mode", mode,
        "--data", protocol["data"]["root"],
        "--output", str(output),
        "--device", protocol["execution"]["train_device"],
        "--max-loss-tokens", str(protocol["budget"]["requested_loss_tokens"]),
        "--batch-size", str(train["batch_size"]),
        "--seed", str(protocol["development"]["seed"]),
        "--model-seed", str(protocol["development"]["model_seed"]),
        "--data-seed", str(protocol["development"]["data_seed"]),
        "--learning-rate", str(train["learning_rate"]),
        "--weight-decay", str(train["weight_decay"]),
        "--betas", *[str(value) for value in train["betas"]],
        "--eps", str(train["eps"]),
        "--grad-clip", str(train["grad_clip"]),
        "--lambda-balance", str(train["lambda_balance"]),
        "--lambda-route", str(train["lambda_route"]),
        "--lambda-state", str(train["lambda_state"]),
        "--warmup-fraction", str(train["warmup_fraction"]),
        "--final-lr-fraction", str(train["final_lr_fraction"]),
        "--checkpoint-every", str(train["checkpoint_every"]),
        "--dev-every", str(train["dev_every"]),
        "--dev-batches", str(train["dev_batches"]),
        "--projected-hours", str(protocol["budget"]["projection"]["arms"][mode]["stage_projected_hours"]["train"]),
        "--ledger-path", protocol["execution"]["ledger_path"],
        "--ledger-lock", protocol["execution"]["ledger_lock"],
    ]
    return [protocol["execution"]["python"], "-m", "native.train", *arguments]


def _evaluate_command(protocol: Mapping, checkpoint: Mapping, output: Path) -> list[str]:
    evaluation = protocol["evaluation"]
    return [
        protocol["execution"]["python"], "-m", "native.evaluate",
        "--checkpoint", checkpoint["path"],
        "--expected-sha256", checkpoint["sha256"],
        "--packed-dev", str(Path(protocol["data"]["root"]) / "dev"),
        "--structured", str(Path(protocol["data"]["root"]) / "structured"),
        "--tokenizer", evaluation["tokenizer"]["path"],
        "--code-generation", evaluation["samples"]["code"]["path"],
        "--output", str(output),
        "--batch-size", str(evaluation["batch_size"]),
        "--max-batches", str(evaluation["dev_batches"]),
        "--limit-per-split", str(evaluation["limit_per_split"]),
        "--code-limit", str(evaluation["code_limit"]),
        "--max-new-tokens", str(evaluation["max_new_tokens"]),
        "--code-max-new-tokens", str(evaluation["code_max_new_tokens"]),
        "--eos-token-id", str(evaluation["eos_token_id"]),
    ] + ([] if evaluation["use_cache"] else ["--no-cache"])


def _reconstruct_command(protocol: Mapping, checkpoint: Mapping, output: Path) -> list[str]:
    command = [
        protocol["execution"]["python"], "-m", "native.reconstruct",
        "--checkpoint", checkpoint["path"],
        "--expected-sha256", checkpoint["sha256"],
        "--spec", protocol["evaluation"]["replay"]["path"],
        "--output", str(output),
        "--device", protocol["execution"]["reconstruction_device"],
    ]
    if protocol["execution"].get("reconstruction_autocast_dtype") is not None:
        command.extend(["--autocast-dtype", protocol["execution"]["reconstruction_autocast_dtype"]])
    return command


def _reserved_command(
    protocol: Mapping,
    mode: str,
    stage: str,
    exact_command: Sequence[str],
    arm_directory: Path,
    protocol_sha256: str,
    *,
    checkpoint: Mapping,
) -> list[str]:
    if exact_command[2] not in ("native.evaluate", "native.reconstruct"):
        raise StudyError("only evaluate/reconstruct child wrappers are supported")
    projected = protocol["budget"]["projection"]["arms"][mode]["stage_projected_hours"][stage]
    arguments = [
        protocol["execution"]["python"], "-m", "native.study", "_reserved",
        "--module", exact_command[2],
        "--projected-hours", str(projected),
        "--ledger-path", protocol["execution"]["ledger_path"],
        "--ledger-lock", protocol["execution"]["ledger_lock"],
        "--mode", mode,
        "--stage", stage,
        "--protocol-sha256", protocol_sha256,
        "--protocol", str(arm_directory.parent.parent / "protocol.json"),
        "--child-stdout", str(arm_directory / f"{stage}.native.stdout.log"),
        "--child-stderr", str(arm_directory / f"{stage}.native.stderr.log"),
    ]
    if stage == "evaluate":
        arguments.extend([
            "--interventions", str(arm_directory / "state_interventions.json"),
            "--checkpoint-for-interventions", checkpoint["path"],
            "--expected-checkpoint-sha256", checkpoint["sha256"],
        ])
    return [*arguments, "--", *exact_command[3:]]


def _write_status(path: Path, payload: Mapping) -> None:
    _write_json(path, payload)


def _validate_train(protocol: Mapping, mode: str, arm: Path) -> dict:
    result = _read_json(arm / "train-result.json")
    budget = protocol["budget"]
    expected = {
        "format": "native-train-v1",
        "mode": mode,
        "device": protocol["execution"]["train_device"],
        "max_loss_tokens": budget["requested_loss_tokens"],
        "consumed_loss_tokens": budget["actual_loss_tokens"],
        "step": budget["planned_updates"],
        "planned_updates": budget["planned_updates"],
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise StudyIncomplete(f"{mode} training {key} is {result.get(key)!r}, expected {value!r}")
    if not result.get("reservation_id"):
        raise StudyIncomplete(f"{mode} training did not report its reservation")
    checkpoint = result.get("final_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise StudyIncomplete(f"{mode} has no final checkpoint descriptor")
    checkpoint_path = Path(checkpoint["path"])
    if not checkpoint_path.is_file() or _sha256(checkpoint_path) != checkpoint["sha256"]:
        raise StudyIncomplete(f"{mode} final checkpoint bytes or hash are incomplete")
    manifest = _read_json(Path(checkpoint["manifest_path"]))
    if manifest.get("sha256") != checkpoint["sha256"] or manifest.get("consumed_loss_tokens") != budget["actual_loss_tokens"] \
            or manifest.get("step") != budget["planned_updates"] or checkpoint.get("step") != budget["planned_updates"]:
        raise StudyIncomplete(f"{mode} final checkpoint manifest does not match the frozen endpoint")
    run_config = _read_json(Path(result["final_checkpoint"]["path"]).parents[1] / "run_config.json")
    if run_config.get("source_config_sha256") != protocol["configs"][mode]["sha256"]:
        raise StudyIncomplete(f"{mode} used a different engineering config")
    if run_config.get("code_identity") != protocol["repository"]["native_code_identity"]:
        stage_train = _read_json(arm / "stage-train.json")
        compatibility = _orchestration_only_profile_commit(
            protocol["repository"],
            stage_train.get("training_commit", ""),
            run_config.get("code_identity", ""),
        )
        if compatibility is None:
            raise StudyIncomplete(f"{mode} used different native code")
    for key, value in (
        ("train_data_identity", protocol["data"]["train_data_identity"]),
        ("dev_data_identity", protocol["data"]["dev_data_identity"]),
        ("order_identity", protocol["data"]["order_identity"]),
    ):
        if run_config.get(key) != value:
            raise StudyIncomplete(f"{mode} used different {key}")
    training = run_config.get("training", {})
    for key, value in protocol["train"].items():
        if training.get(key) != value:
            raise StudyIncomplete(f"{mode} changed fixed training hyperparameter {key}")
    for key, value in (
        ("seed", protocol["development"]["seed"]),
        ("model_seed", protocol["development"]["model_seed"]),
        ("data_seed", protocol["development"]["data_seed"]),
        ("device", protocol["execution"]["train_device"]),
        ("max_loss_tokens", budget["requested_loss_tokens"]),
        ("planned_updates", budget["planned_updates"]),
        ("planned_consumed_loss_tokens", budget["actual_loss_tokens"]),
    ):
        if training.get(key) != value:
            raise StudyIncomplete(f"{mode} changed frozen training field {key}")
    dev = result.get("dev", {})
    expected_dev_batches = min(
        protocol["train"]["dev_batches"],
        math.ceil(protocol["data"]["dev_source_sequences"] / protocol["train"]["batch_size"]),
    )
    if dev.get("loss_tokens") != protocol["data"]["dev_loss_tokens"] \
            or dev.get("consumed_loss_tokens") != budget["actual_loss_tokens"] \
            or dev.get("batches") != expected_dev_batches:
        raise StudyIncomplete(f"{mode} final training dev pass differs from the bounded frozen sample or endpoint")
    learning_path = Path(result["learning_log"])
    learning = _jsonl(learning_path)
    train_steps = {record.get("step") for record in learning if record.get("event") in ("train", "replay")}
    if train_steps != set(range(1, budget["planned_updates"] + 1)):
        raise StudyIncomplete(f"{mode} learning curve is missing training updates")
    dev_records = [record for record in learning if record.get("event") == "dev"]
    expected_dev_steps = [
        step for step in range(1, budget["planned_updates"] + 1)
        if step == budget["planned_updates"] or step % protocol["train"]["dev_every"] == 0
    ]
    if [record.get("step") for record in dev_records] != expected_dev_steps:
        raise StudyIncomplete(f"{mode} periodic/final dev records do not match the frozen schedule")
    if any(
        record.get("loss_tokens") != protocol["data"]["dev_loss_tokens"]
        or record.get("batches") != expected_dev_batches
        for record in dev_records
    ):
        raise StudyIncomplete(f"{mode} periodic/final dev record differs from the bounded frozen sample")
    journal_path = Path(result["journal"])
    journal = _jsonl(journal_path)
    checkpoint_records = []
    for record in journal:
        if record.get("event") != "checkpoint":
            continue
        path = (journal_path.parent / record["path"]).resolve()
        if journal_path.parent.resolve() not in path.parents or not path.is_file() or _sha256(path) != record["sha256"]:
            raise StudyIncomplete(f"{mode} journal checkpoint is missing or has a changed hash")
        checkpoint_records.append({**record, "path": str(path), "actual_sha256": _sha256(path)})
    required_checkpoints = budget["projection"]["arms"][mode]["checkpoints"]["total"]
    if len(checkpoint_records) < required_checkpoints:
        raise StudyIncomplete(f"{mode} has only {len(checkpoint_records)} of {required_checkpoints} projected checkpoints")
    return {
        "result": result,
        "run_config": run_config,
        "learning": learning,
        "checkpoints": checkpoint_records,
        "learning_path": str(learning_path),
        "journal_path": str(journal_path),
    }


def _assert_sample_results(frozen: Mapping, report: Mapping, label: str) -> None:
    selected = len(frozen["samples"])
    source = frozen["source_examples"]
    source_complete = selected == source
    if report.get("examples") != source or report.get("evaluated") != selected:
        raise StudyIncomplete(f"{label} did not evaluate every selected example from the full source population")
    if report.get("complete") is not source_complete \
            or report.get("status") != ("complete" if source_complete else "partial"):
        raise StudyIncomplete(f"{label} has inconsistent bounded-sample completion status")
    records = report.get("records", [])
    actual = [record.get("example_id") for record in records]
    expected = [sample["example_id"] for sample in frozen["samples"]]
    if actual != expected:
        raise StudyIncomplete(f"{label} generation sample identities or order differ from the freeze")


def _validate_evaluation(protocol: Mapping, mode: str, arm: Path, checkpoint: Mapping) -> dict:
    report = _read_json(arm / "evaluation.json")
    if report.get("format") != "native-evaluation-v1" or report.get("checkpoint", {}).get("sha256") != checkpoint["sha256"]:
        raise StudyIncomplete(f"{mode} evaluation is missing or belongs to another checkpoint")
    teacher = report.get("teacher_forced", {})
    source_sequences = protocol["data"]["dev_source_sequences"]
    expected_batches = min(
        protocol["evaluation"]["dev_batches"],
        math.ceil(source_sequences / protocol["evaluation"]["batch_size"]),
    )
    source_complete = expected_batches * protocol["evaluation"]["batch_size"] >= source_sequences
    if teacher.get("complete") is not source_complete \
            or teacher.get("sequences") != len(protocol["data"]["dev_sequences"]) \
            or teacher.get("batches") != expected_batches:
        raise StudyIncomplete(f"{mode} teacher-forced evaluation differs from the bounded frozen dev sample")
    if teacher.get("overall", {}).get("loss_tokens") != protocol["data"]["dev_loss_tokens"]:
        raise StudyIncomplete(f"{mode} teacher-forced evaluation did not score every selected dev target")
    structured = report.get("structured_generation", {})
    if set(structured.get("splits", {})) != set(STRUCTURED_SPLITS):
        raise StudyIncomplete(f"{mode} structured generation is missing a frozen split")
    for split in STRUCTURED_SPLITS:
        frozen = protocol["evaluation"]["samples"]["structured"][split]
        values = structured["splits"][split]
        _assert_sample_results(frozen, values, f"{mode}/{split}")
        file = structured.get("files", {}).get(split, {})
        if file.get("sha256") != frozen["sha256"] or file.get("examples") != frozen["source_examples"]:
            raise StudyIncomplete(f"{mode}/{split} structured source hash or population changed")
        if values.get("max_new_tokens") != protocol["evaluation"]["max_new_tokens"]:
            raise StudyIncomplete(f"{mode}/{split} changed its frozen decoding bound")
    code = report.get("code_generation", {})
    frozen_code = protocol["evaluation"]["samples"]["code"]
    _assert_sample_results(frozen_code, code, f"{mode}/code")
    if code.get("sha256") != frozen_code["sha256"] or code.get("max_new_tokens") != protocol["evaluation"]["code_max_new_tokens"]:
        raise StudyIncomplete(f"{mode} code generation changed its source or decoding bound")
    routing = teacher.get("routing", {})
    expected_routing = "ok" if mode in ("untyped", "typed") else "not_applicable"
    if routing.get("status") != expected_routing:
        raise StudyIncomplete(f"{mode} routing diagnostics are absent or inconsistent")
    interventions = _read_json(arm / "state_interventions.json")
    if interventions.get("format") != "native-state-interventions-v1" \
            or interventions.get("checkpoint", {}).get("sha256") != checkpoint["sha256"] \
            or interventions.get("replay_spec_sha256") != protocol["evaluation"]["replay"]["sha256"]:
        raise StudyIncomplete(f"{mode} state interventions are absent or belong to another checkpoint/replay")
    if interventions.get("specifications") != protocol["interventions"]["specifications"] \
            or interventions.get("samples") != protocol["interventions"]["samples"]:
        raise StudyIncomplete(f"{mode} state-intervention identities or controls differ from the freeze")
    if len(interventions.get("interventions", [])) != len(protocol["interventions"]["specifications"]):
        raise StudyIncomplete(f"{mode} state interventions are incomplete")
    expected_prompts = len(protocol["interventions"]["samples"])
    expected_ids = [sample["example_id"] for sample in protocol["interventions"]["samples"]]
    for intervention in interventions["interventions"]:
        records = intervention.get("generation_effect", {}).get("records", [])
        if intervention.get("generation_effect", {}).get("examples") != expected_prompts \
                or [record.get("example_id") for record in records] != expected_ids:
            raise StudyIncomplete(f"{mode} state intervention skipped or reordered a frozen prompt")
    return {"report": report, "interventions": interventions}


def _validate_reconstruction(protocol: Mapping, mode: str, arm: Path, checkpoint: Mapping) -> dict:
    report = _read_json(arm / "reconstruction.json")
    if report.get("format") != "native-reconstruction-report-v1" or not report.get("agreement"):
        raise StudyIncomplete(f"{mode} reconstruction is absent or does not agree")
    if report.get("checkpoint", {}).get("sha256") != checkpoint["sha256"] \
            or not report["checkpoint"].get("sha256_verified"):
        raise StudyIncomplete(f"{mode} reconstruction belongs to another checkpoint")
    if report.get("replay_spec_sha256") != protocol["evaluation"]["replay"]["sha256"]:
        raise StudyIncomplete(f"{mode} reconstruction used a different replay specification")
    if not report.get("teacher_forced", {}).get("agrees") or not report.get("generation", {}).get("agrees"):
        raise StudyIncomplete(f"{mode} full and cached replay disagree")
    return report


def _arm_bundle(protocol: Mapping, mode: str, study: Path) -> dict:
    arm = study / "arms" / mode
    status_path = arm / "status.json"
    if not status_path.is_file():
        raise StudyIncomplete(f"{mode} has not started or has no preserved status")
    status = _read_json(status_path)
    if status.get("status") != "completed":
        raise StudyIncomplete(f"{mode} status is {status.get('status')!r}, not completed")
    if status.get("protocol_sha256") != _sha256(study / "protocol.json") \
            or status.get("actual_loss_tokens") != protocol["budget"]["actual_loss_tokens"]:
        raise StudyIncomplete(f"{mode} completed a different protocol or token budget")
    trained = _validate_train(protocol, mode, arm)
    checkpoint = trained["result"]["final_checkpoint"]
    evaluated = _validate_evaluation(protocol, mode, arm, checkpoint)
    reconstructed = _validate_reconstruction(protocol, mode, arm, checkpoint)
    stages = []
    for stage in ("train", "evaluate", "reconstruct"):
        record = _read_json(arm / f"stage-{stage}.json")
        if record.get("status") != "completed" or record.get("returncode") != 0:
            raise StudyIncomplete(f"{mode}/{stage} did not complete successfully")
        accounting = record.get("gpu_accounting", {})
        reservation = accounting.get("reservation")
        if not reservation or reservation.get("open") or reservation.get("status") != "ok" \
                or not isinstance(reservation.get("hours"), (int, float)) or reservation["hours"] <= 0:
            raise StudyIncomplete(f"{mode}/{stage} lacks complete per-reservation GPU accounting")
        stages.append(record)
    return {
        "mode": mode,
        "status": status,
        "train": trained,
        "evaluation": evaluated["report"],
        "interventions": evaluated["interventions"],
        "reconstruction": reconstructed,
        "stages": stages,
    }


def _stage_accounting(accounting_reader, protocol: Mapping, result: Mapping | None, mode: str, stage: str, started: float, output: Path) -> dict:
    reservation_id = result.get("reservation_id") if result else None
    return accounting_reader(
        protocol["execution"]["ledger_path"],
        reservation_id=reservation_id,
        mode=mode,
        stage=stage,
        started=started,
        output=output,
    )


def run_study(
    study,
    *,
    runner=subprocess.run,
    accounting_reader=reservation_accounting,
    reservation_path_validator=None,
    git_snapshot_fn=git_snapshot,
) -> int:
    """Run arms in baseline-then-candidate order and stop on the first incomplete arm."""
    study = Path(study).resolve()
    try:
        protocol, protocol_sha256 = _load_protocol(study, git_snapshot_fn)
        (reservation_path_validator or validate_shared_reservation_paths)(
            protocol["execution"]["ledger_path"], protocol["execution"]["ledger_lock"]
        )
    except Exception as error:
        _write_status(study / "run-status.json", {
            "format": RUN_STATUS_FORMAT, "status": "incomplete", "reason": str(error), "updated_at": _now(),
        })
        return 3
    run_status = {
        "format": RUN_STATUS_FORMAT,
        "status": "running",
        "protocol_sha256": protocol_sha256,
        "equal_actual_loss_tokens_per_arm": protocol["budget"]["actual_loss_tokens"],
        "arms": {},
        "updated_at": _now(),
    }
    _write_status(study / "run-status.json", run_status)
    for mode in protocol["execution"]["order"]:
        arm = study / "arms" / mode
        status_path = arm / "status.json"
        if status_path.exists():
            previous = _read_json(status_path)
            if previous.get("status") == "completed":
                try:
                    _arm_bundle(protocol, mode, study)
                    run_status["arms"][mode] = {"status": "completed", "resumed": False}
                    _write_status(study / "run-status.json", run_status)
                    continue
                except Exception as error:
                    run_status.update(status="incomplete", reason=str(error), failed_arm=mode, updated_at=_now())
                    _write_status(study / "run-status.json", run_status)
                    return 3
            run_status.update(
                status="incomplete",
                reason=f"{mode} has preserved {previous.get('status')!r} status; refusing implicit retry or budget reallocation",
                failed_arm=mode,
                updated_at=_now(),
            )
            _write_status(study / "run-status.json", run_status)
            return 3
        arm.mkdir(parents=True)
        arm_status = {
            "status": "incomplete",
            "mode": mode,
            "protocol_sha256": protocol_sha256,
            "actual_loss_tokens": protocol["budget"]["actual_loss_tokens"],
            "phase": "train",
            "updated_at": _now(),
        }
        _write_status(status_path, arm_status)
        train_directory = arm / "train"
        commands = {
            "train": _train_command(protocol, mode, train_directory),
        }
        stage = "train"
        stage_record = None
        try:
            for stage in ("train", "evaluate", "reconstruct"):
                stage_record = None
                _verify_descriptors(protocol["active_files"])
                arm_status.update(status="incomplete", phase=stage, updated_at=_now())
                _write_status(status_path, arm_status)
                if stage == "train":
                    exact = commands["train"]
                    invocation = exact
                    output = train_directory
                else:
                    train_result = _read_json(arm / "train-result.json")
                    checkpoint = train_result["final_checkpoint"]
                    if stage == "evaluate":
                        exact = _evaluate_command(protocol, checkpoint, arm / "evaluation.json")
                    else:
                        exact = _reconstruct_command(protocol, checkpoint, arm / "reconstruction.json")
                    invocation = _reserved_command(
                        protocol, mode, stage, exact, arm, protocol_sha256, checkpoint=checkpoint
                    )
                    output = arm
                started = time.time()
                monotonic = time.monotonic()
                completed = runner(
                    invocation,
                    cwd=protocol["execution"]["working_directory"],
                    env=_environment(protocol),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                returncode, stdout, stderr = _completed_result(completed)
                elapsed = time.monotonic() - monotonic
                _atomic_write(arm / f"{stage}.stdout.log", stdout.encode())
                _atomic_write(arm / f"{stage}.stderr.log", stderr.encode())
                parsed = _last_json(stdout, f"{mode}/{stage}") if stdout.strip() else None
                if stage == "train" and parsed is not None:
                    _write_json(arm / "train-result.json", parsed)
                accounting = _stage_accounting(
                    accounting_reader, protocol, parsed, mode, stage, started, output
                )
                stage_record = {
                    "mode": mode,
                    "stage": stage,
                    "status": "command_completed" if returncode == 0 else "failed",
                    "returncode": returncode,
                    "started_unix": started,
                    "wall_seconds": elapsed,
                    "exact_native_command": list(exact),
                    "reservation_command": list(invocation),
                    "stdout": str(arm / f"{stage}.stdout.log"),
                    "stderr": str(arm / f"{stage}.stderr.log"),
                    "result": parsed,
                    "gpu_accounting": accounting,
                }
                if returncode != 0:
                    raise StudyError(f"{mode}/{stage} exited with status {returncode}")
                if stage == "train":
                    _validate_train(protocol, mode, arm)
                elif stage == "evaluate":
                    checkpoint = _read_json(arm / "train-result.json")["final_checkpoint"]
                    _validate_evaluation(protocol, mode, arm, checkpoint)
                else:
                    checkpoint = _read_json(arm / "train-result.json")["final_checkpoint"]
                    _validate_reconstruction(protocol, mode, arm, checkpoint)
                reservation = accounting.get("reservation")
                if not reservation or reservation.get("open") or reservation.get("status") != "ok" \
                        or not isinstance(reservation.get("hours"), (int, float)) or reservation["hours"] <= 0:
                    raise StudyError(f"{mode}/{stage} has incomplete or failed GPU accounting")
                stage_record["status"] = "completed"
                _write_json(arm / f"stage-{stage}.json", stage_record)
                _verify_descriptors(protocol["active_files"])
            arm_status.update(status="completed", phase="completed", updated_at=_now())
            _write_status(status_path, arm_status)
            run_status["arms"][mode] = {"status": "completed"}
            _write_status(study / "run-status.json", run_status)
        except Exception as error:
            failure = {
                "mode": mode,
                "stage": stage,
                "status": "failed",
                "error": str(error),
                "updated_at": _now(),
            }
            arm_status.update(status="failed", phase=stage, error=str(error), updated_at=_now())
            _write_status(status_path, arm_status)
            if stage_record is not None:
                stage_record.update(status="failed", error=str(error))
                _write_json(arm / f"stage-{stage}.json", stage_record)
            run_status.update(status="incomplete", failed_arm=mode, failed_stage=stage, reason=str(error), updated_at=_now())
            run_status["arms"][mode] = failure
            _write_status(study / "run-status.json", run_status)
            return 3
    summary = summarize_study(study, git_snapshot_fn=git_snapshot_fn)
    if summary["status"] != "complete":
        run_status.update(status="incomplete", reason="final artifact validation refused a complete summary", updated_at=_now())
        _write_status(study / "run-status.json", run_status)
        return 3
    run_status.update(status="completed", summary=str(study / "summary.json"), updated_at=_now())
    _write_status(study / "run-status.json", run_status)
    return 0


def _curve(records: Sequence[Mapping]) -> dict:
    def point(record: Mapping) -> dict:
        keys = (
            "event", "step", "update", "epoch", "cursor", "consumed_loss_tokens", "loss_tokens",
            "language_loss", "route_loss", "state_loss", "balance_loss", "weighted_auxiliary_loss",
            "objective_loss", "learning_rate", "grad_norm", "wall_seconds",
        )
        return {key: record[key] for key in keys if key in record}
    return {
        "train": [point(record) for record in records if record.get("event") in ("train", "replay")],
        "dev": [point(record) for record in records if record.get("event") == "dev"],
        "complete_learning_log_included_by_artifact_hash": True,
    }


def propose_serious_125m(
    arms: Mapping,
    engineering_config: Mapping,
    serious_config: Mapping,
    *,
    target_loss_tokens: int = SERIOUS_LOSS_TOKENS,
    reserve_fraction: float = RESERVE_FRACTION,
    checkpoint_every: int = DEFAULT_TRAIN["checkpoint_every"],
    retention_limit: int = 2,
    available_storage_bytes: int | None = None,
) -> dict:
    """Scale measured stage costs independently; never grant a serious reservation."""
    if target_loss_tokens != SERIOUS_LOSS_TOKENS:
        raise StudyError("this proposal function is explicitly fixed to the serious 125M allocation")
    for name, value in (("checkpoint_every", checkpoint_every), ("retention_limit", retention_limit)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StudyError(f"{name} must be a positive integer")
    if available_storage_bytes is not None and (
        isinstance(available_storage_bytes, bool)
        or not isinstance(available_storage_bytes, int)
        or available_storage_bytes < 0
    ):
        raise StudyError("available_storage_bytes must be a nonnegative integer or None")
    projections = {}
    total_hours = 0.0
    retain_all_hours = 0.0
    retain_all_bytes = 0
    bounded_bytes = 0
    for mode in MODES:
        arm = arms[mode]
        tokens = arm["actual_loss_tokens"]
        updates = arm["planned_updates"]
        stages = {stage["stage"]: stage for stage in arm["reservations"]}
        hours = sum(stage["gpu_accounting"]["reservation"]["hours"] for stage in arm["reservations"])
        train_hours = stages["train"]["gpu_accounting"]["reservation"]["hours"]
        evaluation_hours = stages["evaluate"]["gpu_accounting"]["reservation"]["hours"]
        reconstruction_hours = stages["reconstruct"]["gpu_accounting"]["reservation"]["hours"]
        checkpoint_seconds = arm["time"]["training"].get("checkpoint_seconds")
        checkpoint_count = len(arm["checkpoints"]["all_journal_checkpoints"])
        checkpoint_bytes = arm["checkpoints"]["final"]["bytes"]
        if tokens <= 0 or updates <= 0 or hours <= 0 or train_hours <= 0 \
                or evaluation_hours <= 0 or reconstruction_hours <= 0 \
                or not isinstance(checkpoint_seconds, (int, float)) or checkpoint_seconds <= 0 \
                or checkpoint_count <= 0 or checkpoint_bytes <= 0:
            raise StudyIncomplete(f"{mode} lacks measured stage, checkpoint, or token/GPU-hour evidence")
        checkpoint_hours = checkpoint_seconds / 3600.0
        training_without_checkpoints = train_hours - checkpoint_hours
        if training_without_checkpoints <= 0:
            raise StudyIncomplete(f"{mode} checkpoint time exceeds its training reservation")
        engineering_macs = engineering_config["modes"][mode]["decoded_token_macs"]
        serious_macs = serious_config["modes"][mode]["decoded_token_macs"]
        engineering_eval_macs = engineering_config["modes"][mode]["decoded_token_macs_with_probes"]
        serious_eval_macs = serious_config["modes"][mode]["decoded_token_macs_with_probes"]
        engineering_parameters = engineering_config["modes"][mode]["total"]
        serious_parameters = serious_config["modes"][mode]["total"]
        token_growth = target_loss_tokens / tokens
        training_mac_ratio = engineering_macs / serious_macs
        evaluation_mac_ratio = engineering_eval_macs / serious_eval_macs
        checkpoint_size_ratio = serious_parameters / engineering_parameters
        projected_training_hours = training_without_checkpoints * token_growth * training_mac_ratio
        projected_evaluation_hours = evaluation_hours * evaluation_mac_ratio
        projected_reconstruction_hours = reconstruction_hours * evaluation_mac_ratio
        hours_per_checkpoint = checkpoint_hours / checkpoint_count
        bounded_checkpoint_hours = hours_per_checkpoint * checkpoint_size_ratio * retention_limit
        projected_updates = math.ceil(updates * token_growth)
        all_checkpoint_count = 2 + (projected_updates - 1) // checkpoint_every
        retain_all_checkpoint_hours = hours_per_checkpoint * checkpoint_size_ratio * all_checkpoint_count
        bytes_per_checkpoint = math.ceil(checkpoint_bytes * checkpoint_size_ratio)
        arm_retain_all_bytes = all_checkpoint_count * bytes_per_checkpoint
        arm_bounded_bytes = retention_limit * bytes_per_checkpoint
        projected_hours = (
            projected_training_hours + projected_evaluation_hours
            + projected_reconstruction_hours + bounded_checkpoint_hours
        )
        total_hours += projected_hours
        retain_all_hours += (
            projected_training_hours + projected_evaluation_hours
            + projected_reconstruction_hours + retain_all_checkpoint_hours
        )
        retain_all_bytes += arm_retain_all_bytes
        bounded_bytes += arm_bounded_bytes
        projections[mode] = {
            "measured_actual_loss_tokens": tokens,
            "measured_planned_updates": updates,
            "measured_total_reservation_gpu_hours": hours,
            "measured_train_reservation_gpu_hours": train_hours,
            "measured_checkpoint_seconds": checkpoint_seconds,
            "measured_checkpoint_count": checkpoint_count,
            "token_growth": token_growth,
            "training_mac_ratio": training_mac_ratio,
            "evaluation_reconstruction_mac_ratio": evaluation_mac_ratio,
            "checkpoint_size_ratio": checkpoint_size_ratio,
            "engineering_decoded_token_macs": engineering_macs,
            "serious_decoded_token_macs": serious_macs,
            "proposed_serious_loss_tokens": target_loss_tokens,
            "projected_updates": projected_updates,
            "component_gpu_hours": {
                "training_without_checkpoints": projected_training_hours,
                "bounded_checkpoints": bounded_checkpoint_hours,
                "fixed_evaluation": projected_evaluation_hours,
                "fixed_reconstruction": projected_reconstruction_hours,
            },
            "retain_all_alternative": {
                "checkpoint_count": all_checkpoint_count,
                "checkpoint_gpu_hours": retain_all_checkpoint_hours,
                "checkpoint_bytes": arm_retain_all_bytes,
            },
            "bounded_retention_requirement": {
                "checkpoint_count": retention_limit,
                "checkpoint_gpu_hours": bounded_checkpoint_hours,
                "checkpoint_bytes": arm_bounded_bytes,
                "bytes_per_checkpoint": bytes_per_checkpoint,
            },
            "proposed_serious_gpu_hours": projected_hours,
        }
    warnings = []
    retain_all_fits = None
    bounded_fits = None
    if available_storage_bytes is None:
        warnings.append("Storage capacity was not supplied; neither retain-all nor bounded retention is affirmed affordable.")
    else:
        retain_all_fits = retain_all_bytes <= available_storage_bytes
        bounded_fits = bounded_bytes <= available_storage_bytes
        if not retain_all_fits:
            warnings.append("The estimated retain-all schedule exceeds available storage and is not the proposed retention policy.")
        if not bounded_fits:
            warnings.append("Even bounded retention exceeds available storage; a future allocation must add storage or reduce the schedule.")
    return {
        "status": "measured_proposal_only_not_authorization",
        "authorization": False,
        "opens_reservation": False,
        "target_loss_tokens_per_arm": target_loss_tokens,
        "arms": projections,
        "typed_profile": serious_config["modes"]["typed"],
        "total_four_arm_gpu_hours_before_reserve": total_hours,
        "retain_all_alternative_gpu_hours_before_reserve": retain_all_hours,
        "reserve_fraction": reserve_fraction,
        "required_available_gpu_hours_to_preserve_reserve": total_hours * (1.0 + reserve_fraction),
        "storage": {
            "available_bytes": available_storage_bytes,
            "retain_all_checkpoint_bytes": retain_all_bytes,
            "retain_all_fits": retain_all_fits,
            "bounded_retention_checkpoint_bytes": bounded_bytes,
            "bounded_retention_fits": bounded_fits,
            "required_retention": {
                "policy": "retain only the final checkpoint and latest recovery checkpoint per arm",
                "max_checkpoints_per_arm": retention_limit,
                "required_before_any_future_allocation": True,
            },
            "warnings": warnings,
        },
        "method": "training scales by token growth and decoded-token MAC ratio; fixed evaluation/reconstruction scales only by probe-inclusive MAC ratio; checkpoint I/O and storage scale by parameter/storage ratio with bounded retention",
        "limits": [
            "Evaluation and reconstruction are fixed bounded-population costs and are not multiplied by training-token growth.",
            "The retain-all schedule is an affordability warning, not the proposed policy; bounded retention is required for any future serious freeze.",
            "Update counts and atomic endpoints are estimates until a future freeze replans them.",
            "MAC and parameter scaling are engineering estimates, not measured serious throughput or learning projections.",
            "This JSON does not authorize GPU work or alter the native 12-hour authorization.",
        ],
    }


def _incomplete_summary(study: Path, protocol: Mapping | None, reasons: Sequence[str]) -> dict:
    previous = study / "summary.json"
    previous_hash = _sha256(previous) if previous.is_file() else None
    if previous.exists():
        previous.unlink()
    preserved = {}
    for mode in MODES:
        arm = study / "arms" / mode
        state = {}
        status_path = arm / "status.json"
        if status_path.is_file():
            try:
                state["status"] = _read_json(status_path)
            except Exception as error:
                state["status"] = {"status": "unreadable", "error": str(error)}
        stages = []
        for stage in ("train", "evaluate", "reconstruct"):
            path = arm / f"stage-{stage}.json"
            if path.is_file():
                try:
                    stages.append(_read_json(path))
                except Exception as error:
                    stages.append({"stage": stage, "status": "unreadable", "error": str(error)})
        if stages:
            state["stages"] = stages
            state["gpu_accounting"] = {
                stage["stage"]: stage.get("gpu_accounting") for stage in stages
            }
        if state:
            preserved[mode] = state
    report = {
        "format": SUMMARY_FORMAT,
        "status": "incomplete",
        "reasons": list(reasons),
        "protocol_sha256": _sha256(study / "protocol.json") if protocol is not None else None,
        "previous_complete_summary_sha256": previous_hash,
        "preserved_arm_status_and_accounting": preserved,
        "comparative_outcomes_emitted": False,
        "updated_at": _now(),
    }
    _write_json(study / "incomplete.json", report)
    return report


def summarize_study(study, *, git_snapshot_fn=git_snapshot) -> dict:
    """Emit comparative outcomes only after identical budgets and every artifact validate."""
    study = Path(study).resolve()
    try:
        protocol, _ = _load_protocol(study, git_snapshot_fn)
    except Exception as error:
        return _incomplete_summary(study, None, [str(error)])
    bundles = {}
    reasons = []
    for mode in MODES:
        try:
            bundles[mode] = _arm_bundle(protocol, mode, study)
        except Exception as error:
            reasons.append(str(error))
    if reasons:
        return _incomplete_summary(study, protocol, reasons)
    endpoints = {bundle["train"]["result"]["consumed_loss_tokens"] for bundle in bundles.values()}
    if endpoints != {protocol["budget"]["actual_loss_tokens"]}:
        return _incomplete_summary(study, protocol, [
            f"arm endpoints {sorted(endpoints)} differ from frozen {protocol['budget']['actual_loss_tokens']}"
        ])
    arms = {}
    for mode, bundle in bundles.items():
        result = bundle["train"]["result"]
        artifacts = {}
        for name, path in (
            ("train_result", study / "arms" / mode / "train-result.json"),
            ("learning_curve", Path(bundle["train"]["learning_path"])),
            ("checkpoint_journal", Path(bundle["train"]["journal_path"])),
            ("evaluation", study / "arms" / mode / "evaluation.json"),
            ("state_interventions", study / "arms" / mode / "state_interventions.json"),
            ("reconstruction", study / "arms" / mode / "reconstruction.json"),
        ):
            artifacts[name] = _descriptor(path)
        arms[mode] = {
            "mode": mode,
            "requested_loss_tokens": result["max_loss_tokens"],
            "actual_loss_tokens": result["consumed_loss_tokens"],
            "planned_updates": result["planned_updates"],
            "loss_vs_token": _curve(bundle["train"]["learning"]),
            "time": {
                "training": {
                    key: result.get(key) for key in (
                        "data_loading_seconds", "forward_seconds", "backward_seconds", "optimizer_seconds",
                        "checkpoint_seconds", "actual_loss_tokens_per_second", "wall_seconds",
                    ) if key in result
                },
                "stage_wall_seconds": {stage["stage"]: stage["wall_seconds"] for stage in bundle["stages"]},
                "charged_gpu_hours": {
                    stage["stage"]: stage["gpu_accounting"]["reservation"]["hours"] for stage in bundle["stages"]
                },
            },
            "evaluation_population": "bounded_frozen_sample",
            "full_corpus_evaluation": False,
            "per_source_metrics": bundle["evaluation"]["teacher_forced"]["sources"],
            "per_source_metrics_scope": "selected frozen dev rows only; not full-source estimates",
            "teacher_forced": bundle["evaluation"]["teacher_forced"],
            "generation": {
                "structured": bundle["evaluation"]["structured_generation"],
                "code": bundle["evaluation"]["code_generation"],
            },
            "routing": {
                "checkpointed_evaluation": bundle["evaluation"]["teacher_forced"]["routing"],
                "final_training_dev": result.get("dev", {}).get("route_usage", []),
            },
            "state_interventions": bundle["interventions"],
            "checkpoints": {
                "final": result["final_checkpoint"],
                "all_journal_checkpoints": bundle["train"]["checkpoints"],
            },
            "reconstruction": bundle["reconstruction"],
            "artifacts": artifacts,
            "reservations": bundle["stages"],
            "status": "completed",
        }
    summary = {
        "format": SUMMARY_FORMAT,
        "status": "complete",
        "created_at": _now(),
        "protocol_sha256": _sha256(study / "protocol.json"),
        "repository_commit": protocol["repository"]["commit"],
        "native_code_identity": protocol["repository"]["native_code_identity"],
        "development_seed": protocol["development"]["seed"],
        "evaluation_population": "bounded_frozen_sample",
        "full_corpus_evaluation": False,
        "evaluation_limits": {
            "dev_batches": protocol["evaluation"]["dev_batches"],
            "evaluation_batch_size": protocol["evaluation"]["batch_size"],
            "structured_limit_per_split": protocol["evaluation"]["limit_per_split"],
            "code_limit": protocol["evaluation"]["code_limit"],
        },
        "equal_actual_loss_tokens_per_arm": protocol["budget"]["actual_loss_tokens"],
        "requested_loss_tokens_per_arm": protocol["budget"]["requested_loss_tokens"],
        "budget_reduction": protocol["budget"]["reduction"],
        "arms": arms,
        "future_serious_125m_proposal": propose_serious_125m(
            arms,
            protocol["engineering_config"],
            protocol["serious_config"],
            reserve_fraction=protocol["feasibility"]["reserve_fraction"],
            checkpoint_every=protocol["train"]["checkpoint_every"],
            available_storage_bytes=protocol["budget"]["storage"]["available_bytes"],
        ),
        "gpu_accounting": {
            "scope": "every start, heartbeat, end, charge, status, command and foreign-process observation retained per stage reservation",
            "reservations": {
                mode: {stage["stage"]: stage["gpu_accounting"] for stage in values["reservations"]}
                for mode, values in arms.items()
            },
        },
        "evidence_boundary": protocol["evidence_boundary"],
    }
    _write_json(study / "summary.json", summary)
    (study / "incomplete.json").unlink(missing_ok=True)
    return summary


def run_state_interventions(protocol: Mapping, checkpoint_path: str, expected_sha256: str, output: Path) -> dict:
    """Run the committed intervention API on the frozen replay input and prompts."""
    from . import evaluate as E
    from .reconstruct import load_replay_spec

    model, checkpoint = E.load_native_model(checkpoint_path, expected_sha256=expected_sha256, device="cpu")
    replay = load_replay_spec(protocol["evaluation"]["replay"]["path"])
    if replay.get("spec_sha256") != protocol["evaluation"]["replay"]["sha256"]:
        raise StudyError("loaded replay specification hash differs from the frozen replay hash")
    prompts = []
    for frozen in protocol["interventions"]["samples"]:
        path = Path(protocol["evaluation"]["samples"]["structured"][frozen["split"]]["path"])
        record = next(
            (record for record in _jsonl(path) if record["example_id"] == frozen["example_id"]), None
        )
        if record is None or _identity(record) != frozen["record_sha256"]:
            raise StudyError(f"intervention sample {frozen['example_id']} changed after freeze")
        prompts.append(record)
    tokenizer = E.load_gpt2_tokenizer(protocol["evaluation"]["tokenizer"]["path"])
    specifications = [E.InterventionSpec(**spec) for spec in protocol["interventions"]["specifications"]]
    report = E.evaluate_state_interventions(
        model,
        replay["teacher_forced"],
        prompts,
        specifications,
        tokenizer,
        max_new_tokens=protocol["evaluation"]["max_new_tokens"],
        eos_token_id=protocol["evaluation"]["eos_token_id"],
        use_cache=protocol["evaluation"]["use_cache"],
    )
    result = {
        "format": "native-state-interventions-v1",
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint["sha256"]},
        "replay_spec_sha256": protocol["evaluation"]["replay"]["sha256"],
        "specifications": protocol["interventions"]["specifications"],
        "samples": protocol["interventions"]["samples"],
        **report,
    }
    _write_json(Path(output), result)
    return result


def _verified_child_protocol(path, expected_sha256: str) -> dict:
    protocol_path = Path(path)
    receipt = protocol_path.with_name("protocol.sha256")
    if not protocol_path.is_file() or not receipt.is_file():
        raise StudyError("child protocol or SHA256 receipt is missing")
    actual = _sha256(protocol_path)
    recorded = receipt.read_text(encoding="utf-8").strip()
    if actual != expected_sha256 or recorded != expected_sha256:
        raise StudyError("child protocol SHA256 receipt does not match the parent-frozen protocol")
    protocol = _read_json(protocol_path)
    if not isinstance(protocol, dict):
        raise StudyError("child protocol is not a JSON object")
    return protocol


def _reserved_main(args: argparse.Namespace) -> int:
    """Child-only reservation wrapper; the parent orchestrator never calls this function."""
    from . import ledger as native_ledger

    protocol = _verified_child_protocol(args.protocol, args.protocol_sha256)
    validate_shared_reservation_paths(args.ledger_path, args.ledger_lock)
    exact = [sys.executable, "-m", args.module, *args.native_arguments]
    reservation = native_ledger.Reservation(
        command=" ".join(shlex.quote(part) for part in exact),
        projected_hours=args.projected_hours,
        path=Path(args.ledger_path),
        lock=Path(args.ledger_lock),
        extra={
            "native_study_arm": args.mode,
            "native_study_stage": args.stage,
            "native_study_protocol_sha256": args.protocol_sha256,
        },
    )
    child = None
    error = None
    try:
        with reservation:
            child = subprocess.run(exact, capture_output=True, text=True, check=False)
            _atomic_write(Path(args.child_stdout), (child.stdout or "").encode())
            _atomic_write(Path(args.child_stderr), (child.stderr or "").encode())
            if child.returncode:
                raise StudyError(f"{args.module} exited with status {child.returncode}")
            if args.interventions:
                run_state_interventions(
                    protocol,
                    args.checkpoint_for_interventions,
                    args.expected_checkpoint_sha256,
                    Path(args.interventions),
                )
    except BaseException as caught:
        error = repr(caught)
    metadata = {
        "reservation_id": reservation.id,
        "returncode": child.returncode if child is not None and child.returncode else (1 if error else 0),
        "exact_native_command": exact,
        "child_stdout": args.child_stdout,
        "child_stderr": args.child_stderr,
        "state_interventions": args.interventions,
        "error": error,
    }
    print(json.dumps(metadata, sort_keys=True))
    return int(metadata["returncode"])


def parser() -> argparse.ArgumentParser:
    root = _repository_root()
    result = argparse.ArgumentParser(prog="python -m native.study", description=__doc__.split("\n\n")[0])
    subcommands = result.add_subparsers(dest="command", required=True)
    freeze = subcommands.add_parser("freeze")
    freeze.add_argument("--output", required=True)
    freeze.add_argument("--data", required=True)
    freeze.add_argument("--tokenizer", required=True)
    freeze.add_argument("--replay-spec", required=True)
    freeze.add_argument("--engineering-profile", action="append", required=True)
    freeze.add_argument("--engineering-config", default=str(_native_directory() / "configs" / "engineering.json"))
    freeze.add_argument("--serious-config", default=str(_native_directory() / "configs" / "serious.json"))
    freeze.add_argument("--repository", default=str(root))
    freeze.add_argument("--development-seed", type=int, default=DEVELOPMENT_SEED)
    freeze.add_argument("--provisional-loss-tokens", type=int, default=PROVISIONAL_LOSS_TOKENS)
    freeze.add_argument("--reserve-fraction", type=float, default=RESERVE_FRACTION)
    freeze.add_argument("--available-gpu-hours", type=float)
    freeze.add_argument("--available-storage-bytes", type=int)
    freeze.add_argument("--ledger-path")
    freeze.add_argument("--ledger-lock")
    freeze.add_argument("--train-hyperparameters", help="JSON overrides for the fixed shared training recipe")
    for name in ("run", "summarize"):
        command = subcommands.add_parser(name)
        command.add_argument("--study", required=True)
    reserved = subcommands.add_parser("_reserved")
    reserved.add_argument("--module", required=True, choices=("native.evaluate", "native.reconstruct"))
    reserved.add_argument("--projected-hours", required=True, type=float)
    reserved.add_argument("--ledger-path", required=True)
    reserved.add_argument("--ledger-lock", required=True)
    reserved.add_argument("--mode", required=True, choices=MODES)
    reserved.add_argument("--stage", required=True, choices=("evaluate", "reconstruct"))
    reserved.add_argument("--protocol-sha256", required=True)
    reserved.add_argument("--child-stdout", required=True)
    reserved.add_argument("--child-stderr", required=True)
    reserved.add_argument("--interventions")
    reserved.add_argument("--protocol", required=True)
    reserved.add_argument("--checkpoint-for-interventions")
    reserved.add_argument("--expected-checkpoint-sha256")
    reserved.add_argument("native_arguments", nargs=argparse.REMAINDER)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "freeze":
            overrides = _read_json(Path(args.train_hyperparameters)) if args.train_hyperparameters else None
            frozen = freeze_study(
                args.output,
                data=args.data,
                tokenizer=args.tokenizer,
                replay_spec=args.replay_spec,
                engineering_profiles=args.engineering_profile,
                engineering_config=args.engineering_config,
                serious_config=args.serious_config,
                repository=args.repository,
                development_seed=args.development_seed,
                provisional_loss_tokens=args.provisional_loss_tokens,
                reserve_fraction=args.reserve_fraction,
                available_gpu_hours=args.available_gpu_hours,
                available_storage_bytes=args.available_storage_bytes,
                ledger_path=args.ledger_path,
                ledger_lock=args.ledger_lock,
                train_hyperparameters=overrides,
            )
            print(json.dumps({
                "status": frozen["status"],
                "protocol_sha256": frozen.get("protocol_sha256"),
                "actual_loss_tokens_per_arm": frozen.get("protocol", {}).get("budget", {}).get("actual_loss_tokens"),
                "reason": frozen.get("reason"),
            }, sort_keys=True))
            return 0 if frozen["status"] == "frozen" else 4
        if args.command == "run":
            return run_study(args.study)
        if args.command == "summarize":
            summary = summarize_study(args.study)
            print(json.dumps(summary, sort_keys=True, allow_nan=False))
            return 0 if summary["status"] == "complete" else 3
        if args.command == "_reserved":
            if args.native_arguments and args.native_arguments[0] == "--":
                args.native_arguments = args.native_arguments[1:]
            if args.interventions and not (args.checkpoint_for_interventions and args.expected_checkpoint_sha256):
                raise StudyError("interventions require a checkpoint path and expected SHA256")
            return _reserved_main(args)
        raise StudyError(f"unknown command {args.command}")
    except (StudyError, ValueError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2


freeze = freeze_study
run = run_study
summarize = summarize_study
propose_serious_allocation = propose_serious_125m


if __name__ == "__main__":
    raise SystemExit(main())
