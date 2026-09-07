"""Small, explicit input checks shared by the producer, not its checker."""
from __future__ import annotations
import numpy as np


def matrix(value, name: str, *, allow_empty_rows: bool = False) -> np.ndarray:
    out = np.array(value, dtype=np.float64, copy=True)
    if out.ndim != 2 or out.shape[1] == 0 or (out.shape[0] == 0 and not allow_empty_rows):
        raise ValueError(f"{name} must be a nonempty 2-D matrix")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain finite numbers")
    out.setflags(write=False)
    return out


def vector(value, size: int, name: str) -> np.ndarray:
    out = np.array(value, dtype=np.float64, copy=True)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return out


def positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)
