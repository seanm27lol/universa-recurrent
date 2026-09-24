"""Direction diagnostics and a separate, explicit retained-norm channel."""

import math
import torch


def unit_direction(vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 1 or not vector.is_floating_point():
        raise ValueError("expected one floating-point vector")
    value = vector.float()
    norm = torch.linalg.vector_norm(value)
    if not torch.isfinite(value).all() or not torch.isfinite(norm) or norm <= 1e-12:
        raise ValueError("nonfinite or near-zero vector norm")
    return value / norm


def restore_norm(
    direction: torch.Tensor, retained_norm: float, *, dtype: torch.dtype
) -> torch.Tensor:
    """Normalize in float32 and cast once. The four-byte norm is not AR output."""
    if not math.isfinite(retained_norm) or retained_norm <= 1e-12:
        raise ValueError("retained norm must be finite and positive")
    result = (unit_direction(direction) * retained_norm).to(dtype)
    if not torch.isfinite(result).all():
        raise ValueError("replacement overflows target dtype")
    return result


def direction_metrics(original, reconstruction) -> dict:
    a, b = unit_direction(original), unit_direction(reconstruction)
    if a.shape != b.shape:
        raise ValueError("vector widths differ")
    return {
        "cosine": float(a @ b),
        "unit_direction_squared_l2": float((a - b).square().sum()),
    }
