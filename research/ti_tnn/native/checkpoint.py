"""Durable native checkpoints and explicit journal recovery.

A checkpoint is fsynced before an atomic replace and has a JSON SHA-256 manifest. Recovery never advances a sampler or
trains implicitly: it selects the newest valid logged checkpoint and returns the complete subsequent batch records that
a trainer must replay. Only an unterminated journal tail may be truncated; complete acknowledged records are preserved.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Mapping

import numpy as np
import torch

from .sampler import SamplerBatch, identity_hash


CHECKPOINT_FORMAT = "native-checkpoint-v1"
JOURNAL_FORMAT = "native-recovery-journal-v1"


class CheckpointError(RuntimeError):
    """Base class for checkpoint failures."""


class CheckpointValidationError(CheckpointError):
    """A checkpoint or its manifest is missing, corrupt, or internally inconsistent."""


class CheckpointIdentityMismatch(CheckpointError, ValueError):
    """A valid checkpoint belongs to a different configuration, code, data, or order."""


class JournalError(RuntimeError):
    """Base class for recovery-journal failures."""


class JournalCorrupt(JournalError):
    """A complete journal record is malformed or breaks the acknowledged sequence."""


class JournalIdentityMismatch(JournalError, ValueError):
    """A journal belongs to different data or batch order."""


class RecoveryError(JournalError):
    """No safe recovery plan can be constructed."""


class ReplayMismatch(RecoveryError):
    """The deterministic sampler does not reproduce a logged batch."""


def fsync_dir(directory) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write(path, data: bytes) -> None:
    """Write bytes in the destination directory, fsync, atomically replace, then fsync the directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def capture_rng_state() -> dict:
    """Capture Python, NumPy and Torch CPU RNGs, plus CUDA states only if CUDA is already initialized."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = [tensor.clone() for tensor in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping, *, strict_cuda: bool = False) -> bool:
    """Restore captured RNGs and return whether CUDA states were restored.

    CUDA is never initialized merely to restore a checkpoint. If a checkpoint contains CUDA states but CUDA is not
    already initialized, ``strict_cuda=True`` refuses a partial restoration; the default leaves those states untouched.
    """
    cuda_ready = torch.cuda.is_available() and torch.cuda.is_initialized()
    if strict_cuda and "cuda" in state and not cuda_ready:
        raise CheckpointValidationError("checkpoint contains CUDA RNG states but CUDA is not initialized")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].detach().to(device="cpu", dtype=torch.uint8))
    if "cuda" not in state:
        return False
    if cuda_ready:
        torch.cuda.set_rng_state_all(state["cuda"])
        return True
    return False


def manifest_path(path, sha256: str | None = None) -> Path:
    """Return the immutable manifest name for specific checkpoint bytes, hashing an existing file if needed."""
    path = Path(path)
    digest = sha256_file(path) if sha256 is None else sha256
    return path.with_name(f"{path.name}.{digest}.manifest.json")


def _as_identity(value):
    return value if value is None or isinstance(value, str) else identity_hash(value)


def _state_hashes(model_state, optimizer_state, scheduler_state, sampler_state) -> dict:
    return {
        "model": identity_hash(model_state),
        "optimizer": identity_hash(optimizer_state),
        "scheduler": identity_hash(scheduler_state),
        "sampler": identity_hash(sampler_state),
    }


def save_checkpoint(path, model, optimizer=None, scheduler=None, *, sampler=None,
                    cursor: int | None = None, consumed_loss_tokens: int | None = None,
                    step: int | None = None, config=None, code_identity=None,
                    data_identity=None, order_identity=None, identities: Mapping | None = None,
                    extra: Mapping | None = None) -> dict:
    """Save complete training state and return the durable manifest descriptor.

    Cursor and token counters are derived from ``sampler`` when omitted. Explicit values must agree with its state.
    ``extra`` is reserved for run metadata; trainer loops and loss computation remain outside this module.
    """
    path = Path(path)
    sampler_state = None
    if sampler is not None:
        sampler_state = dict(sampler.state_dict() if hasattr(sampler, "state_dict") else sampler)
        if cursor is None:
            cursor = sampler_state.get("cursor")
        if consumed_loss_tokens is None:
            consumed_loss_tokens = sampler_state.get("consumed_loss_tokens")
        if step is None:
            step = sampler_state.get("batches_consumed")
        if data_identity is None:
            data_identity = sampler_state.get("data_identity")
        if order_identity is None:
            order_identity = sampler_state.get("order_identity")
        if cursor != sampler_state.get("cursor"):
            raise ValueError("cursor differs from sampler state")
        if consumed_loss_tokens != sampler_state.get("consumed_loss_tokens"):
            raise ValueError("consumed_loss_tokens differs from sampler state")
    for name, value in (("cursor", cursor), ("consumed_loss_tokens", consumed_loss_tokens), ("step", step)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be a non-negative integer or None")
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict() if optimizer is not None else None
    scheduler_state = scheduler.state_dict() if scheduler is not None else None
    identity_values = {
        "config": identity_hash(config),
        "code": _as_identity(code_identity),
        "data": _as_identity(data_identity),
        "order": _as_identity(order_identity),
    }
    for key, value in (identities or {}).items():
        if key in identity_values:
            raise ValueError(f"identity {key!r} must be supplied through its named argument")
        identity_values[str(key)] = _as_identity(value)
    state_hashes = _state_hashes(model_state, optimizer_state, scheduler_state, sampler_state)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": step,
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "sampler": sampler_state,
        "cursor": cursor,
        "consumed_loss_tokens": consumed_loss_tokens,
        "rng": capture_rng_state(),
        "config": config,
        "identities": identity_values,
        "state_hashes": state_hashes,
        "extra": dict(extra or {}),
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    manifest = {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_name": path.name,
        "sha256": digest,
        "bytes": len(data),
        "step": step,
        "cursor": cursor,
        "consumed_loss_tokens": consumed_loss_tokens,
        "identities": identity_values,
        "state_hashes": state_hashes,
    }
    sidecar = manifest_path(path, digest)
    atomic_write(sidecar, (json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False) + "\n").encode())
    atomic_write(path, data)
    return {
        "path": str(path),
        "manifest_path": str(sidecar),
        "sha256": digest,
        "bytes": len(data),
        "step": step,
        "cursor": cursor,
        "consumed_loss_tokens": consumed_loss_tokens,
        "identities": identity_values,
        "state_hashes": state_hashes,
    }


def _check_expected_identities(payload: Mapping, *, expected_config=None, expected_code_identity=None,
                               expected_data_identity=None, expected_order_identity=None,
                               expected_identities: Mapping | None = None) -> None:
    recorded = payload["identities"]
    expected = dict(expected_identities or {})
    for key, value in (("config", expected_config), ("code", expected_code_identity),
                       ("data", expected_data_identity), ("order", expected_order_identity)):
        if value is not None:
            if key in expected:
                raise ValueError(f"expected identity {key!r} was supplied twice")
            expected[key] = value
    for key, value in expected.items():
        wanted = identity_hash(value) if key == "config" else _as_identity(value)
        if recorded.get(key) != wanted:
            raise CheckpointIdentityMismatch(
                f"cannot resume: {key} identity mismatch (checkpoint={recorded.get(key)!r}, expected={wanted!r})")


def _validate_payload(payload: Mapping) -> None:
    if not isinstance(payload, Mapping) or payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointValidationError("unsupported or malformed native checkpoint payload")
    required = {"step", "model", "optimizer", "scheduler", "sampler", "cursor", "consumed_loss_tokens",
                "rng", "config", "identities", "state_hashes", "extra"}
    missing = sorted(required - set(payload))
    if missing:
        raise CheckpointValidationError(f"checkpoint payload is missing {missing}")
    identities = payload["identities"]
    if not isinstance(identities, Mapping) or not {"config", "code", "data", "order"} <= set(identities):
        raise CheckpointValidationError("checkpoint identities are incomplete")
    if identities["config"] != identity_hash(payload["config"]):
        raise CheckpointValidationError("configuration does not match its recorded identity")
    sampler = payload["sampler"]
    if sampler is not None:
        if not isinstance(sampler, Mapping):
            raise CheckpointValidationError("sampler payload must be a mapping or None")
        if payload["cursor"] != sampler.get("cursor"):
            raise CheckpointValidationError("checkpoint cursor differs from its sampler state")
        if payload["consumed_loss_tokens"] != sampler.get("consumed_loss_tokens"):
            raise CheckpointValidationError("checkpoint token count differs from its sampler state")
    expected_hashes = _state_hashes(payload["model"], payload["optimizer"], payload["scheduler"], sampler)
    if payload["state_hashes"] != expected_hashes:
        raise CheckpointValidationError("checkpoint state hashes do not match its payload")


def load_checkpoint(path, *, expected_sha256: str | None = None, expected_config=None,
                    expected_code_identity=None, expected_data_identity=None, expected_order_identity=None,
                    expected_identities: Mapping | None = None) -> dict:
    """Read and validate a checkpoint before returning its payload; no live training state is changed."""
    path = Path(path)
    try:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise CheckpointValidationError(f"{path}: checkpoint hash differs from the journal")
        sidecar = manifest_path(path, digest)
        if not sidecar.is_file():
            raise CheckpointValidationError(f"{path}: checkpoint hash manifest is missing for content {digest}")
        manifest = json.loads(sidecar.read_text())
        if manifest.get("format") != CHECKPOINT_FORMAT or manifest.get("checkpoint_name") != path.name:
            raise CheckpointValidationError(f"{path}: invalid checkpoint manifest")
        if manifest.get("sha256") != digest or manifest.get("bytes") != len(data):
            raise CheckpointValidationError(f"{path}: checkpoint content does not match its hash manifest")
        try:
            payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        except Exception as error:
            raise CheckpointValidationError(f"{path}: cannot deserialize checkpoint: {error}") from error
        _validate_payload(payload)
        if manifest.get("state_hashes") != payload["state_hashes"]:
            raise CheckpointValidationError(f"{path}: manifest state hashes differ from the payload")
        for key in ("step", "cursor", "consumed_loss_tokens", "identities"):
            if manifest.get(key) != payload[key]:
                raise CheckpointValidationError(f"{path}: manifest {key} differs from the payload")
        _check_expected_identities(
            payload,
            expected_config=expected_config,
            expected_code_identity=expected_code_identity,
            expected_data_identity=expected_data_identity,
            expected_order_identity=expected_order_identity,
            expected_identities=expected_identities,
        )
    except CheckpointIdentityMismatch:
        raise
    except CheckpointValidationError:
        raise
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        raise CheckpointValidationError(f"{path}: invalid checkpoint or manifest: {error}") from error
    payload["path"] = str(path)
    payload["sha256"] = digest
    payload["manifest"] = manifest
    return payload


validate_checkpoint = load_checkpoint


@dataclass(frozen=True)
class RestoreResult:
    step: int | None
    cursor: int | None
    consumed_loss_tokens: int | None
    checkpoint: dict


def restore_checkpoint(source, model, optimizer=None, scheduler=None, sampler=None, *, restore_rng: bool = True,
                       strict_cuda: bool = False, expected_config=None, expected_code_identity=None,
                       expected_data_identity=None, expected_order_identity=None,
                       expected_identities: Mapping | None = None) -> RestoreResult:
    """Explicitly restore model/training/sampler state, then RNGs, returning the exact restored counters."""
    if isinstance(source, (str, os.PathLike)):
        payload = load_checkpoint(
            source,
            expected_config=expected_config,
            expected_code_identity=expected_code_identity,
            expected_data_identity=expected_data_identity,
            expected_order_identity=expected_order_identity,
            expected_identities=expected_identities,
        )
    else:
        payload = source
        _validate_payload(payload)
        _check_expected_identities(
            payload,
            expected_config=expected_config,
            expected_code_identity=expected_code_identity,
            expected_data_identity=expected_data_identity,
            expected_order_identity=expected_order_identity,
            expected_identities=expected_identities,
        )
    model.load_state_dict(payload["model"])
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise CheckpointValidationError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise CheckpointValidationError("checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if sampler is not None:
        if payload["sampler"] is None:
            raise CheckpointValidationError("checkpoint has no sampler state")
        sampler.load_state_dict(payload["sampler"])
    if restore_rng:
        restore_rng_state(payload["rng"], strict_cuda=strict_cuda)
    return RestoreResult(payload["step"], payload["cursor"], payload["consumed_loss_tokens"], payload)


@dataclass(frozen=True)
class JournalEntry:
    record: dict
    offset: int
    end: int


class CheckpointJournal:
    """Single-writer, fsynced JSONL log of acknowledged batches and checkpoint descriptors."""

    def __init__(self, path):
        self.path = Path(path)

    def _append(self, record: dict) -> dict:
        record = {"journal_format": JOURNAL_FORMAT, **record}
        encoded = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            with open(self.path, "rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise JournalCorrupt("refusing to append after an incomplete journal tail; truncate it explicitly")
        with open(self.path, "ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_dir(self.path.parent)
        return record

    def append_batch(self, batch: SamplerBatch, sampler, *, step: int | None = None) -> dict:
        state = sampler.state_dict()
        if state["cursor"] != batch.cursor or state["consumed_loss_tokens"] != batch.consumed_loss_tokens:
            raise JournalError("sampler state is not the state immediately after this batch")
        if state["batches_consumed"] != batch.batches_consumed:
            raise JournalError("sampler batch counter is not the state immediately after this batch")
        if state["order_identity"] != batch.order_identity:
            raise JournalIdentityMismatch("batch and sampler order identities differ")
        return self._append({
            "event": "batch",
            "step": batch.batches_consumed if step is None else step,
            "batch_identity": batch.batch_identity,
            "indices": list(batch.indices),
            "dropped_indices": list(batch.dropped_indices),
            "epoch": batch.epoch,
            "loss_tokens": batch.loss_tokens,
            "cursor": batch.cursor,
            "consumed_loss_tokens": batch.consumed_loss_tokens,
            "batches_consumed": batch.batches_consumed,
            "data_identity": state["data_identity"],
            "order_identity": state["order_identity"],
            "sampler": state,
        })

    log_batch = append_batch

    def append_checkpoint(self, checkpoint, *, step: int | None = None) -> dict:
        if isinstance(checkpoint, (str, os.PathLike)):
            source = Path(checkpoint)
            checkpoint = json.loads(manifest_path(source).read_text())
            checkpoint["path"] = str(source)
        path = Path(checkpoint["path"])
        identities = checkpoint.get("identities", {})
        return self._append({
            "event": "checkpoint",
            "step": checkpoint.get("step") if step is None else step,
            "path": os.path.relpath(path, self.path.parent),
            "sha256": checkpoint["sha256"],
            "bytes": checkpoint["bytes"],
            "cursor": checkpoint.get("cursor"),
            "consumed_loss_tokens": checkpoint.get("consumed_loss_tokens"),
            "data_identity": identities.get("data"),
            "order_identity": identities.get("order"),
        })

    log_checkpoint = append_checkpoint

    def entries(self) -> tuple[list[JournalEntry], int, bool, int]:
        return _read_journal(self.path)

    def truncate_incomplete_tail(self, *, expected_valid_offset: int | None = None) -> int:
        """Remove only an unterminated final record, never a complete acknowledged batch or checkpoint."""
        entries, valid_offset, incomplete, size = _read_journal(self.path)
        del entries
        if expected_valid_offset is not None and valid_offset != expected_valid_offset:
            raise JournalCorrupt("journal changed after recovery; refusing to truncate at a stale offset")
        if not incomplete:
            return 0
        with open(self.path, "r+b") as handle:
            handle.truncate(valid_offset)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_dir(self.path.parent)
        return size - valid_offset


Journal = CheckpointJournal


def _read_journal(path) -> tuple[list[JournalEntry], int, bool, int]:
    path = Path(path)
    if not path.exists():
        return [], 0, False, 0
    data = path.read_bytes()
    entries = []
    offset = 0
    incomplete = False
    while offset < len(data):
        newline = data.find(b"\n", offset)
        if newline < 0:
            incomplete = True
            break
        end = newline + 1
        raw = data[offset:newline]
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        try:
            record = json.loads(raw)
            if not isinstance(record, dict) or record.get("journal_format") != JOURNAL_FORMAT:
                raise ValueError("not a native recovery-journal record")
            if record.get("event") not in ("batch", "checkpoint"):
                raise ValueError("unknown journal event")
        except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise JournalCorrupt(f"{path}: corrupt complete record at byte {offset}: {error}") from error
        entries.append(JournalEntry(record, offset, end))
        offset = end
    return entries, offset, incomplete, len(data)


@dataclass(frozen=True)
class LoggedBatch:
    step: int
    batch: SamplerBatch
    sampler_state: dict
    data_identity: str
    order_identity: str
    offset: int

    @property
    def cursor(self) -> int:
        return self.batch.cursor

    @property
    def consumed_loss_tokens(self) -> int:
        return self.batch.consumed_loss_tokens


def _nonnegative(record: Mapping, key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JournalCorrupt(f"journal {key} must be a non-negative integer")
    return value


def _logged_batch(entry: JournalEntry) -> LoggedBatch:
    record = entry.record
    step = _nonnegative(record, "step")
    for key in ("cursor", "consumed_loss_tokens", "batches_consumed", "epoch", "loss_tokens"):
        _nonnegative(record, key)
    state = record.get("sampler")
    if not isinstance(state, dict):
        raise JournalCorrupt("batch record has no sampler state")
    try:
        batch = SamplerBatch(
            indices=tuple(record["indices"]),
            dropped_indices=tuple(record.get("dropped_indices", ())),
            epoch=record["epoch"],
            cursor=record["cursor"],
            loss_tokens=record["loss_tokens"],
            consumed_loss_tokens=record["consumed_loss_tokens"],
            batches_consumed=record["batches_consumed"],
            order_identity=record["order_identity"],
        )
    except (KeyError, TypeError) as error:
        raise JournalCorrupt(f"malformed batch record: {error}") from error
    if batch.batch_identity != record.get("batch_identity"):
        raise JournalCorrupt("batch identity does not match its recorded contents")
    for key in ("cursor", "consumed_loss_tokens", "batches_consumed"):
        if state.get(key) != record[key]:
            raise JournalCorrupt(f"batch {key} differs from its sampler state")
    if state.get("order_identity") != record.get("order_identity"):
        raise JournalCorrupt("batch order identity differs from its sampler state")
    if state.get("data_identity") != record.get("data_identity"):
        raise JournalCorrupt("batch data identity differs from its sampler state")
    return LoggedBatch(step, batch, state, record["data_identity"], record["order_identity"], entry.offset)


@dataclass(frozen=True)
class RecoveryPlan:
    journal: CheckpointJournal
    checkpoint_path: Path
    checkpoint: dict
    replay_batches: tuple[LoggedBatch, ...]
    invalid_checkpoints: tuple[dict, ...]
    valid_journal_bytes: int
    journal_bytes: int
    incomplete_tail: bool

    @property
    def requires_replay(self) -> bool:
        return bool(self.replay_batches)

    @property
    def manifest(self) -> dict:
        return self.checkpoint["manifest"]

    @property
    def acknowledged_cursor(self) -> int | None:
        return self.replay_batches[-1].cursor if self.replay_batches else self.checkpoint["cursor"]

    @property
    def acknowledged_consumed_loss_tokens(self) -> int | None:
        return (self.replay_batches[-1].consumed_loss_tokens if self.replay_batches
                else self.checkpoint["consumed_loss_tokens"])

    def restore(self, model, optimizer=None, scheduler=None, sampler=None, **kwargs) -> RestoreResult:
        """Restore the selected checkpoint only; replay remains a separate explicit trainer operation."""
        return restore_checkpoint(self.checkpoint, model, optimizer, scheduler, sampler, **kwargs)

    def validate_replay(self, sampler) -> tuple[SamplerBatch, ...]:
        """Reproduce every required batch on a clone, leaving the supplied sampler's cursor untouched.

        This validates order and accounting, not optimizer replay. A trainer must execute these batches in order and
        restore the selected RNG state before applying their updates.
        """
        if self.checkpoint["cursor"] is not None and sampler.cursor != self.checkpoint["cursor"]:
            raise ReplayMismatch("sampler is not at the selected checkpoint cursor")
        clone = sampler.clone()
        batches = []
        for logged in self.replay_batches:
            actual = clone.next_batch()
            if actual != logged.batch or clone.state_dict() != logged.sampler_state:
                raise ReplayMismatch(f"deterministic replay differs at logged step {logged.step}")
            batches.append(actual)
        return tuple(batches)

    def truncate_incomplete_tail(self) -> int:
        return self.journal.truncate_incomplete_tail(expected_valid_offset=self.valid_journal_bytes)


def _resolve_journal_path(journal: Path, reference: str) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else journal.parent / path


def recover(journal_path, *, expected_config=None, expected_code_identity=None,
            expected_data_identity=None, expected_order_identity=None,
            expected_identities: Mapping | None = None) -> RecoveryPlan:
    """Select the newest valid checkpoint and plan replay of every later complete batch record.

    Missing/corrupt newer checkpoints are reported in ``invalid_checkpoints``. A valid checkpoint with an identity
    mismatch is never bypassed in favor of an older run. No model, RNG, sampler, or journal bytes are modified here.
    """
    expected_data_identity = _as_identity(expected_data_identity)
    expected_order_identity = _as_identity(expected_order_identity)
    journal = CheckpointJournal(journal_path)
    entries, valid_bytes, incomplete, journal_bytes = _read_journal(journal.path)
    batches = [_logged_batch(entry) for entry in entries if entry.record["event"] == "batch"]
    for previous, current in zip(batches, batches[1:]):
        if current.step != previous.step + 1:
            raise JournalCorrupt(f"acknowledged batch steps jump from {previous.step} to {current.step}")
    data_identities = {record.get("data_identity") for record in (entry.record for entry in entries)
                       if record.get("data_identity") is not None}
    order_identities = {record.get("order_identity") for record in (entry.record for entry in entries)
                        if record.get("order_identity") is not None}
    if len(data_identities) > 1 or len(order_identities) > 1:
        raise JournalIdentityMismatch("journal mixes different data or order identities")
    if expected_data_identity is not None and data_identities and data_identities != {expected_data_identity}:
        raise JournalIdentityMismatch("cannot resume: journal data identity mismatch")
    if expected_order_identity is not None and order_identities and order_identities != {expected_order_identity}:
        raise JournalIdentityMismatch("cannot resume: journal order identity mismatch")
    selected_entry = None
    selected_payload = None
    invalid = []
    for entry in reversed(entries):
        record = entry.record
        if record["event"] != "checkpoint":
            continue
        path = _resolve_journal_path(journal.path, record.get("path", ""))
        try:
            payload = load_checkpoint(
                path,
                expected_sha256=record.get("sha256"),
                expected_config=expected_config,
                expected_code_identity=expected_code_identity,
                expected_data_identity=expected_data_identity,
                expected_order_identity=expected_order_identity,
                expected_identities=expected_identities,
            )
        except CheckpointIdentityMismatch:
            raise
        except (CheckpointValidationError, OSError) as error:
            invalid.append({"path": str(path), "offset": entry.offset, "error": str(error)})
            continue
        for key in ("step", "cursor", "consumed_loss_tokens"):
            if record.get(key) != payload[key]:
                raise JournalCorrupt(f"checkpoint journal {key} differs from the named valid checkpoint")
        if record.get("bytes") != payload["manifest"]["bytes"]:
            raise JournalCorrupt("checkpoint journal byte count differs from its manifest")
        selected_entry = entry
        selected_payload = payload
        break
    if selected_entry is None or selected_payload is None:
        raise RecoveryError(f"no valid logged checkpoint remains; invalid checkpoints: {invalid}")
    selected_identities = selected_payload["identities"]
    if data_identities and selected_identities["data"] not in data_identities:
        raise JournalIdentityMismatch("selected checkpoint data identity differs from the journal")
    if order_identities and selected_identities["order"] not in order_identities:
        raise JournalIdentityMismatch("selected checkpoint order identity differs from the journal")
    replay = tuple(batch for batch in batches if batch.offset > selected_entry.offset)
    if replay and selected_payload["step"] is not None and replay[0].step != selected_payload["step"] + 1:
        raise JournalCorrupt("first replay batch does not immediately follow the selected checkpoint step")
    for batch in replay:
        if batch.data_identity != selected_identities["data"] or batch.order_identity != selected_identities["order"]:
            raise JournalIdentityMismatch("replay batch identities differ from the selected checkpoint")
    for damaged in invalid:
        if damaged["offset"] <= selected_entry.offset:
            continue
        record = next(entry.record for entry in entries if entry.offset == damaged["offset"])
        if any(batch.cursor == record.get("cursor")
               and batch.consumed_loss_tokens == record.get("consumed_loss_tokens")
               and batch.step == record.get("step") for batch in replay):
            continue
        if (record.get("cursor") == selected_payload["cursor"]
                and record.get("consumed_loss_tokens") == selected_payload["consumed_loss_tokens"]
                and record.get("step") == selected_payload["step"]):
            continue
        raise RecoveryError("a missing/corrupt checkpoint acknowledges progress not covered by complete batch records")
    return RecoveryPlan(
        journal=journal,
        checkpoint_path=Path(selected_payload["path"]),
        checkpoint=selected_payload,
        replay_batches=replay,
        invalid_checkpoints=tuple(invalid),
        valid_journal_bytes=valid_bytes,
        journal_bytes=journal_bytes,
        incomplete_tail=incomplete,
    )


recover_checkpoint = recover
