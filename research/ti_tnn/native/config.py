"""Configuration for the four-pass native causal language model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


MODES = ("dense", "shared", "untyped", "typed")


@dataclass(frozen=True)
class NativeConfig:
    """Dimensions shared by the dense, shared, untyped, and typed arms."""

    vocab_size: int = 50257
    context_length: int = 256
    d_model: int = 192
    n_heads: int = 4
    d_slot: int = 48
    d_ff: int = 768
    mode: str = "dense"
    n_state_classes: int = 32
    n_passes: int = 4
    n_experts: int = 4
    n_slots: int = 4

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        for name in (
            "vocab_size",
            "context_length",
            "d_model",
            "n_heads",
            "d_slot",
            "d_ff",
            "n_state_classes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.n_passes, self.n_experts, self.n_slots) != (4, 4, 4):
            raise ValueError("the native architecture has exactly four passes, experts, and slots")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, int | str]) -> "NativeConfig":
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "NativeConfig":
        values = json.loads(Path(path).read_text())
        return cls.from_dict(values.get("model", values))
