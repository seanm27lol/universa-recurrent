"""Synthetic structured-flow data for the first learned recurrence experiment.

Each target is sampled inside one of two known constraint subspaces. The model
sees a noisy, partially observed ambient vector and must infer both the
structure and the clean vector. This is deliberately tiny and inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..examples import flow_example


@dataclass(frozen=True)
class NeuralBatchSpec:
    ambient_dim: int
    latent_dim: int
    num_structures: int
    noise_std: float
    observe_probability: float


def candidate_bases() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return names, orthonormal bases, and boundaries for the toy task."""
    example = flow_example(7)
    dimensions = {candidate.latent_dimension for candidate in example.candidates}
    if len(dimensions) != 1:
        raise ValueError("the current neural task requires equal latent dimensions")
    names = [candidate.name for candidate in example.candidates]
    bases = np.stack([candidate.basis for candidate in example.candidates]).astype(np.float32)
    boundaries = np.stack([candidate.boundary for candidate in example.candidates]).astype(np.float32)
    return names, bases, boundaries


class StructuredFlowDataset(Dataset):
    """Deterministic partial noisy observations of structured flows.

    Coordinates are standard normal inside one candidate subspace. Observation
    masks are independent of the structure label. A mask is repaired only when
    it exposes fewer than two ambient coordinates.
    """

    def __init__(
        self,
        n: int,
        *,
        seed: int,
        noise_std: float = 0.05,
        observe_probability: float = 0.7,
    ) -> None:
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not np.isfinite(observe_probability) or not (0.0 < observe_probability <= 1.0):
            raise ValueError("observe_probability must be finite and in (0,1]")
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")

        names, bases, _ = candidate_bases()
        rng = np.random.default_rng(seed)
        num_structures, ambient_dim, latent_dim = bases.shape

        labels = rng.integers(0, num_structures, size=n, endpoint=False)
        coordinates = rng.normal(0, 1, size=(n, latent_dim)).astype(np.float32)
        truth = np.einsum("bnd,bd->bn", bases[labels], coordinates, optimize=True).astype(np.float32)

        masks = (rng.random((n, ambient_dim)) < observe_probability).astype(np.float32)
        for index in range(n):
            if masks[index].sum() < 2:
                masks[index, rng.choice(ambient_dim, size=2, replace=False)] = 1.0

        noise = rng.normal(0, noise_std, size=(n, ambient_dim)).astype(np.float32)
        observed = masks * (truth + noise)

        self.observed = torch.from_numpy(observed)
        self.mask = torch.from_numpy(masks)
        self.truth = torch.from_numpy(truth)
        self.label = torch.from_numpy(labels.astype(np.int64))
        self.names = names
        self.seed = seed
        self.spec = NeuralBatchSpec(
            ambient_dim=ambient_dim,
            latent_dim=latent_dim,
            num_structures=num_structures,
            noise_std=float(noise_std),
            observe_probability=float(observe_probability),
        )

    def __len__(self) -> int:
        return int(self.observed.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observed": self.observed[index],
            "mask": self.mask[index],
            "truth": self.truth[index],
            "label": self.label[index],
        }
