"""Deterministic batch order and loss-token accounting for packed sequences.

The sampler owns only batch order. Its epoch generators are local CPU generators, so constructing a model or drawing a
model initializer cannot change the training order and advancing the sampler cannot change a model initializer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
from pathlib import PurePath
from typing import Iterator, Mapping, Sequence

import numpy as np
import torch


SAMPLER_FORMAT = "native-packed-sampler-v1"


class SamplerIdentityError(ValueError):
    """A saved sampler belongs to different data or a different order specification."""


class SamplerStateError(ValueError):
    """A sampler state is malformed or inconsistent with this sampler."""


def _frame(digest: "hashlib._Hash", tag: bytes, payload: bytes = b"") -> None:
    digest.update(len(tag).to_bytes(2, "big"))
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _feed_identity(digest: "hashlib._Hash", value) -> None:
    if value is None:
        _frame(digest, b"none")
    elif isinstance(value, bool):
        _frame(digest, b"bool", b"1" if value else b"0")
    elif isinstance(value, int):
        _frame(digest, b"int", str(value).encode())
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("non-finite floats are not allowed in a data or order identity")
        _frame(digest, b"float", value.hex().encode())
    elif isinstance(value, str):
        _frame(digest, b"str", value.encode())
    elif isinstance(value, bytes):
        _frame(digest, b"bytes", value)
    elif isinstance(value, bytearray):
        _frame(digest, b"bytes", bytes(value))
    elif isinstance(value, PurePath):
        _frame(digest, b"path", str(value).encode())
    elif torch.is_tensor(value):
        tensor = value.detach().to(device="cpu").contiguous()
        header = json.dumps([str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        _frame(digest, b"tensor", header)
        raw = tensor.view(torch.uint8).numpy().tobytes() if tensor.numel() else b""
        _frame(digest, b"tensor-data", raw)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("object arrays are not allowed in a data or order identity")
        array = np.ascontiguousarray(value)
        header = json.dumps([array.dtype.str, list(array.shape)], separators=(",", ":")).encode()
        _frame(digest, b"ndarray", header)
        _frame(digest, b"ndarray-data", array.tobytes(order="C"))
    elif isinstance(value, np.generic):
        _feed_identity(digest, value.item())
    elif is_dataclass(value) and not isinstance(value, type):
        _frame(digest, b"dataclass", f"{type(value).__module__}.{type(value).__qualname__}".encode())
        _feed_identity(digest, asdict(value))
    elif isinstance(value, Mapping):
        items = list(value.items())
        items.sort(key=lambda item: (type(item[0]).__qualname__, identity_hash(item[0])))
        _frame(digest, b"mapping", len(items).to_bytes(8, "big"))
        for key, item in items:
            _feed_identity(digest, key)
            _feed_identity(digest, item)
    elif isinstance(value, (list, tuple)):
        _frame(digest, type(value).__name__.encode(), len(value).to_bytes(8, "big"))
        for item in value:
            _feed_identity(digest, item)
    else:
        raise TypeError(f"{type(value).__name__} is not allowed in a data or order identity")


def identity_hash(value) -> str:
    """Hash nested configuration/data content without depending on pickle layout or global RNG state."""
    digest = hashlib.sha256()
    _feed_identity(digest, value)
    return digest.hexdigest()


@dataclass(frozen=True)
class SamplerBatch(Sequence[int]):
    """One acknowledged-order candidate; counters are the values immediately after this batch."""

    indices: tuple[int, ...]
    epoch: int
    cursor: int
    loss_tokens: int
    consumed_loss_tokens: int
    batches_consumed: int
    order_identity: str
    dropped_indices: tuple[int, ...] = ()

    @property
    def step(self) -> int:
        return self.batches_consumed

    @property
    def batch_identity(self) -> str:
        return identity_hash({
            "format": SAMPLER_FORMAT,
            "order_identity": self.order_identity,
            "epoch": self.epoch,
            "indices": list(self.indices),
            "dropped_indices": list(self.dropped_indices),
            "cursor": self.cursor,
            "loss_tokens": self.loss_tokens,
            "consumed_loss_tokens": self.consumed_loss_tokens,
            "batches_consumed": self.batches_consumed,
        })

    def __len__(self) -> int:
        return len(self.indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __getitem__(self, item):
        return self.indices[item]


class PackedSequenceSampler:
    """Cycle through packed sequences in seeded epoch permutations.

    ``cursor`` counts packed-sequence positions, including an explicitly reported dropped suffix when ``drop_last`` is
    true. ``consumed_loss_tokens`` counts only emitted examples. A sequence must provide ``loss_mask``, ``labels``
    (with ``ignore_index``), or an explicit non-negative ``loss_tokens``/``n_loss_tokens`` count.
    """

    def __init__(self, sequences, batch_size: int, seed: int, *, shuffle: bool = True,
                 drop_last: bool = False, ignore_index: int = -100):
        self._sequences = tuple(sequences)
        if not self._sequences:
            raise ValueError("packed sequence data is empty")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if drop_last and batch_size > len(self._sequences):
            raise ValueError("drop_last with batch_size larger than the data would produce empty epochs")
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.ignore_index = ignore_index
        self._loss_token_counts = tuple(self._count_loss_tokens(sequence) for sequence in self._sequences)
        self.data_identity = identity_hash({"format": SAMPLER_FORMAT, "sequences": list(self._sequences)})
        self.order_identity = identity_hash({
            "format": SAMPLER_FORMAT,
            "data_identity": self.data_identity,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "ignore_index": self.ignore_index,
        })
        self.cursor = 0
        self.consumed_loss_tokens = 0
        self.batches_consumed = 0

    @property
    def data_hash(self) -> str:
        return self.data_identity

    @property
    def order_hash(self) -> str:
        return self.order_identity

    @property
    def epoch(self) -> int:
        return self.cursor // len(self._sequences)

    @property
    def offset(self) -> int:
        return self.cursor % len(self._sequences)

    @property
    def batches_per_epoch(self) -> int:
        n = len(self._sequences)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __len__(self) -> int:
        return self.batches_per_epoch

    @staticmethod
    def _field(sequence, name: str):
        if isinstance(sequence, Mapping):
            return sequence.get(name)
        return getattr(sequence, name, None)

    def _count_loss_tokens(self, sequence) -> int:
        for name in ("loss_tokens", "n_loss_tokens"):
            count = self._field(sequence, name)
            if count is not None:
                if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or int(count) < 0:
                    raise ValueError(f"{name} must be a non-negative integer")
                return int(count)
        mask = self._field(sequence, "loss_mask")
        if mask is not None:
            tensor = torch.as_tensor(mask).detach().to(device="cpu")
            if not torch.all((tensor == 0) | (tensor == 1)):
                raise ValueError("loss_mask must contain only zero and one")
            return int(torch.count_nonzero(tensor).item())
        labels = self._field(sequence, "labels")
        if labels is not None:
            tensor = torch.as_tensor(labels).detach().to(device="cpu")
            return int(torch.count_nonzero(tensor != self.ignore_index).item())
        raise ValueError("each packed sequence needs loss_mask, labels, or an explicit loss-token count")

    def _epoch_seed(self, epoch: int) -> int:
        material = f"{SAMPLER_FORMAT}\0{self.seed}\0{epoch}".encode()
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)

    def epoch_order(self, epoch: int) -> tuple[int, ...]:
        """Return an epoch permutation without changing this sampler or any global RNG."""
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if not self.shuffle:
            return tuple(range(len(self._sequences)))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._epoch_seed(epoch))
        return tuple(int(index) for index in torch.randperm(
            len(self._sequences), generator=generator, device="cpu").tolist())

    def next_batch(self) -> SamplerBatch:
        n = len(self._sequences)
        usable = n - (n % self.batch_size) if self.drop_last else n
        epoch, offset = divmod(self.cursor, n)
        order = self.epoch_order(epoch)
        end = min(offset + self.batch_size, usable)
        if end == offset:
            raise SamplerStateError("cursor points at a dropped suffix rather than an emittable batch")
        indices = tuple(order[offset:end])
        dropped = tuple(order[usable:]) if self.drop_last and end == usable else ()
        next_cursor = (epoch + 1) * n if self.drop_last and end == usable else epoch * n + end
        loss_tokens = sum(self._loss_token_counts[index] for index in indices)
        batch = SamplerBatch(
            indices=indices,
            epoch=epoch,
            cursor=next_cursor,
            loss_tokens=loss_tokens,
            consumed_loss_tokens=self.consumed_loss_tokens + loss_tokens,
            batches_consumed=self.batches_consumed + 1,
            order_identity=self.order_identity,
            dropped_indices=dropped,
        )
        self.cursor = batch.cursor
        self.consumed_loss_tokens = batch.consumed_loss_tokens
        self.batches_consumed = batch.batches_consumed
        return batch

    def next_indices(self) -> list[int]:
        return list(self.next_batch().indices)

    def __iter__(self) -> Iterator[SamplerBatch]:
        """Yield the remainder of the current epoch; ``next_batch`` itself cycles indefinitely."""
        epoch = self.epoch
        while self.epoch == epoch:
            yield self.next_batch()

    def state_dict(self) -> dict:
        return {
            "format": SAMPLER_FORMAT,
            "data_identity": self.data_identity,
            "order_identity": self.order_identity,
            "data_hash": self.data_identity,
            "order_hash": self.order_identity,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "ignore_index": self.ignore_index,
            "cursor": self.cursor,
            "consumed_loss_tokens": self.consumed_loss_tokens,
            "batches_consumed": self.batches_consumed,
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Validate the complete state before changing any counter, then restore it exactly."""
        if not isinstance(state, Mapping):
            raise SamplerStateError("sampler state must be a mapping")
        if state.get("format") != SAMPLER_FORMAT:
            raise SamplerStateError(f"unsupported sampler format {state.get('format')!r}")
        for key, actual in (("data_identity", self.data_identity), ("data_hash", self.data_identity),
                            ("order_identity", self.order_identity), ("order_hash", self.order_identity)):
            if key in state and state[key] != actual:
                kind = "data" if key.startswith("data") else "order"
                raise SamplerIdentityError(f"cannot resume: {kind} identity mismatch")
        for key in ("data_identity", "order_identity"):
            if key not in state:
                raise SamplerStateError(f"sampler state is missing {key}")
        expected = {
            "seed": self.seed,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "ignore_index": self.ignore_index,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise SamplerIdentityError(f"cannot resume: order specification differs at {key}")
        counters = {}
        for key in ("cursor", "consumed_loss_tokens", "batches_consumed"):
            value = state.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SamplerStateError(f"{key} must be a non-negative integer")
            counters[key] = value
        n = len(self._sequences)
        usable = n - (n % self.batch_size) if self.drop_last else n
        if counters["cursor"] % n >= usable:
            raise SamplerStateError("cursor points into a dropped suffix")
        if counters["batches_consumed"] > counters["cursor"]:
            raise SamplerStateError("batches_consumed cannot exceed cursor")
        self.cursor = counters["cursor"]
        self.consumed_loss_tokens = counters["consumed_loss_tokens"]
        self.batches_consumed = counters["batches_consumed"]

    def clone(self) -> "PackedSequenceSampler":
        return PackedSequenceSampler.from_state_dict(self._sequences, self.state_dict())

    @classmethod
    def from_state_dict(cls, sequences, state: Mapping) -> "PackedSequenceSampler":
        sampler = cls(
            sequences,
            batch_size=state["batch_size"],
            seed=state["seed"],
            shuffle=state.get("shuffle", True),
            drop_last=state.get("drop_last", False),
            ignore_index=state.get("ignore_index", -100),
        )
        sampler.load_state_dict(state)
        return sampler

    def to_json(self) -> str:
        return json.dumps(self.state_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, sequences, state: str) -> "PackedSequenceSampler":
        return cls.from_state_dict(sequences, json.loads(state))
