"""P4 calibration-fitted PCA baseline on unit directions.

Establishes a numerical reconstruction control fitted only on pooled,
unlabeled calibration activations, with rank chosen by a declared
byte-budget rule rather than per-example reconstruction quality. It does
not establish that generic reconstruction matches or beats language, and
it performs no norm restoration: geometry.restore_norm applies the
retained receiver norm at patch time. Per-example coefficient bytes are
no-more-than-budget; the shared mean/basis storage is reported separately,
never folded into the per-example language budget.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from open_weight_lingua.artifacts import (
    load_numeric,
    save_numeric,
    sha256_file,
    write_json,
)
from open_weight_lingua.geometry import unit_direction

FIT_FORMAT_VERSION = "open_weight_lingua.pca_fit.v1"
FIT_TENSORS_NAME = "baseline_fit.safetensors"
FIT_SIDECAR_NAME = "baseline_fit.json"
BYTES_PER_COEFFICIENT = 2  # coefficients are quantized to float16
FIT_DTYPE = torch.float32


@dataclass(frozen=True)
class PCAFit:
    """Mean direction and orthonormal basis fitted on unit directions.

    mean is the arithmetic mean of unit directions and is not itself a
    unit vector. basis rows are orthonormal, shape (fitted_rank, width).
    identity is a sha256 content hash suitable for the manifest field
    baseline_fit_identity; it identifies this artifact, nothing more.
    Construct via fit_pca or load_fit.
    """

    mean: torch.Tensor
    basis: torch.Tensor
    fitted_rank: int
    width: int
    dtype: torch.dtype
    identity: str


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _content_hash(mean: torch.Tensor, basis: torch.Tensor) -> str:
    header = json.dumps(
        {
            "format": FIT_FORMAT_VERSION,
            "dtype": _dtype_name(mean.dtype),
            "mean_shape": list(mean.shape),
            "basis_shape": list(basis.shape),
        },
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(header)
    for tensor in (mean, basis):
        value = tensor.detach().cpu().contiguous()
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _direction_matrix(activations) -> torch.Tensor:
    if isinstance(activations, torch.Tensor):
        if activations.ndim != 2:
            raise ValueError("expected a 2D tensor of calibration activations")
        rows = list(activations.detach().cpu())
    else:
        rows = list(activations)
    if not rows:
        raise ValueError("at least one calibration activation is required")
    directions = [unit_direction(row) for row in rows]
    width = directions[0].shape[0]
    if any(direction.shape[0] != width for direction in directions):
        raise ValueError("calibration activations have inconsistent widths")
    return torch.stack(directions)


def fit_pca(activations) -> PCAFit:
    """Fit on pooled calibration activations; no labels enter here.

    Each activation is converted to a unit direction (nonfinite, non-float
    and near-zero inputs are rejected). The basis spans the numerical rank
    of the centered scatter actually available, which can be less than both
    the sample count and the width.
    """
    directions = _direction_matrix(activations)
    width = directions.shape[1]
    mean = directions.mean(0)
    centered = directions - mean
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    if singular.numel() == 0 or float(singular[0]) <= 0.0:
        fitted_rank = 0
    else:
        tolerance = (
            torch.finfo(torch.float32).eps
            * max(centered.shape)
            * float(singular[0])
        )
        fitted_rank = int((singular > tolerance).sum())
    basis = vh[:fitted_rank].contiguous()
    return PCAFit(
        mean=mean.contiguous(),
        basis=basis,
        fitted_rank=fitted_rank,
        width=width,
        dtype=FIT_DTYPE,
        identity=_content_hash(mean, basis),
    )


def reconstruct(fit: PCAFit, vector, *, text_bytes: int) -> tuple[torch.Tensor, dict]:
    """Project onto the top-k basis with k = min(fitted_rank, text_bytes // 2).

    Coefficients are quantized to float16, the estimate is renormalized to a
    unit direction in float32, and the norm is left to the patch-time
    restore_norm step. A collapsed (near-zero) estimate raises instead of
    manufacturing a direction. The budget is interpreted as
    no-more-than-budget: coefficient bytes never exceed text_bytes, and any
    remainder is reported as unused.
    """
    if isinstance(text_bytes, bool) or not isinstance(text_bytes, int) or text_bytes < 0:
        raise ValueError("text_bytes must be a nonnegative integer")
    direction = unit_direction(vector)
    if direction.shape[0] != fit.width:
        raise ValueError("vector width does not match the fitted width")
    rank = min(fit.fitted_rank, text_bytes // BYTES_PER_COEFFICIENT)
    if rank:
        basis = fit.basis[:rank]
        coefficients = (basis @ (direction - fit.mean)).to(torch.float16)
        estimate = fit.mean + basis.T @ coefficients.float()
    else:
        estimate = fit.mean
    reconstructed = unit_direction(estimate)
    coefficient_bytes = BYTES_PER_COEFFICIENT * rank
    accounting = {
        "budget_rule": "rank = min(fitted_rank, floor(text_bytes / 2))",
        "budget_interpretation": "no-more-than-budget",
        "baseline_fit_identity": fit.identity,
        "text_budget_bytes": text_bytes,
        "fitted_rank": fit.fitted_rank,
        "rank_used": rank,
        "coefficient_dtype": "float16",
        "bytes_per_coefficient": BYTES_PER_COEFFICIENT,
        "coefficient_bytes": coefficient_bytes,
        "unused_budget_bytes": text_bytes - coefficient_bytes,
        "shared_dtype": _dtype_name(fit.dtype),
        "shared_mean_basis_bytes": fit.width
        * (1 + fit.fitted_rank)
        * fit.mean.element_size(),
    }
    return reconstructed, accounting


def save_fit(fit: PCAFit, directory) -> dict:
    """Write the safetensors payload and JSON sidecar; refuses overwrites."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors_path = directory / FIT_TENSORS_NAME
    save_numeric(tensors_path, {"mean": fit.mean, "basis": fit.basis})
    sidecar = {
        "format": FIT_FORMAT_VERSION,
        "width": fit.width,
        "fitted_rank": fit.fitted_rank,
        "dtype": _dtype_name(fit.dtype),
        "mean_shape": [fit.width],
        "basis_shape": [fit.fitted_rank, fit.width],
        "baseline_fit_identity": fit.identity,
        "tensors_file": FIT_TENSORS_NAME,
        "tensors_sha256": sha256_file(tensors_path),
        "budget_interpretation": "no-more-than-budget",
    }
    write_json(directory / FIT_SIDECAR_NAME, sidecar)
    return sidecar


def load_fit(directory) -> PCAFit:
    """Reload a fit, validating shapes, finiteness and content identity.

    The identity pins the exact bytes fitted; a self-consistent but altered
    file is rejected. Geometric properties (orthonormality) are inherited
    from the pinned content, not re-derived here.
    """
    directory = Path(directory)
    with (directory / FIT_SIDECAR_NAME).open() as stream:
        sidecar = json.load(stream)
    if sidecar.get("format") != FIT_FORMAT_VERSION:
        raise ValueError("unsupported baseline fit format")
    width, fitted_rank = sidecar["width"], sidecar["fitted_rank"]
    if (
        isinstance(width, bool)
        or isinstance(fitted_rank, bool)
        or not isinstance(width, int)
        or not isinstance(fitted_rank, int)
        or width < 1
        or fitted_rank < 0
    ):
        raise ValueError("invalid baseline fit dimensions")
    if sidecar["mean_shape"] != [width] or sidecar["basis_shape"] != [
        fitted_rank,
        width,
    ]:
        raise ValueError("baseline fit shapes do not match sidecar")
    dtype = getattr(torch, str(sidecar["dtype"]), None)
    if dtype != FIT_DTYPE:
        raise ValueError("unsupported baseline fit dtype")
    tensors_path = directory / FIT_TENSORS_NAME
    if sha256_file(tensors_path) != sidecar["tensors_sha256"]:
        raise ValueError("baseline fit payload does not match sidecar")
    values = load_numeric(tensors_path)
    if set(values) != {"mean", "basis"}:
        raise ValueError("baseline fit payload must contain mean and basis")
    mean, basis = values["mean"], values["basis"]
    if mean.shape != (width,) or basis.shape != (fitted_rank, width):
        raise ValueError("baseline fit tensor shapes do not match sidecar")
    if mean.dtype != dtype or basis.dtype != dtype:
        raise ValueError("baseline fit tensor dtype does not match sidecar")
    identity = _content_hash(mean, basis)
    if identity != sidecar["baseline_fit_identity"]:
        raise ValueError("baseline fit identity mismatch")
    return PCAFit(
        mean=mean,
        basis=basis,
        fitted_rank=fitted_rank,
        width=width,
        dtype=dtype,
        identity=identity,
    )
