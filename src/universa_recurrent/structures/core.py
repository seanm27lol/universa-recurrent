"""Cache the linear algebra defining B z = 0.

Example: for a circulation, B adds incoming flow and subtracts outgoing
flow at each junction. Q is a coordinate system for all balanced flows.
This is an original reference implementation, not vendored Universa code.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from .._validation import matrix, vector, positive


def incidence_matrix(nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """One column per directed edge: -1 at its tail, +1 at its head."""
    if isinstance(nodes, bool) or not isinstance(nodes, int) or nodes < 1 or not edges:
        raise ValueError("Provide a positive node count and at least one edge")
    b = np.zeros((nodes, len(edges)), dtype=np.float64)
    for col, (tail, head) in enumerate(edges):
        if (not isinstance(tail, int) or not isinstance(head, int)
                or isinstance(tail, bool) or isinstance(head, bool)
                or not 0 <= tail < nodes or not 0 <= head < nodes or tail == head):
            raise ValueError("Edges need distinct, valid integer endpoints")
        b[tail, col] = -1.0
        b[head, col] = 1.0
    return b


@dataclass(frozen=True)
class CompiledConstraint:
    """Instance-local, copied, read-only arrays: no mutable global cache.

    Change B or its rank convention by compiling a new instance. SVD and
    the multiplier map are built ONCE, not at every recurrent step.
    """
    name: str
    boundary: np.ndarray
    basis: np.ndarray
    multiplier_map: np.ndarray
    rank: int
    rank_rtol: float
    fingerprint: str

    @classmethod
    def compile(cls, name: str, boundary, *, rank_rtol: float = 1e-12):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("A constraint needs a nonempty name")
        b = matrix(boundary, "boundary", allow_empty_rows=True)
        rtol = positive(rank_rtol, "rank_rtol")
        if rtol >= 1:
            raise ValueError("rank_rtol must be below 1")
        u, singular, vh = np.linalg.svd(b, full_matrices=True)
        cutoff = rtol * singular[0] if singular.size else 0.0
        rank = int(np.count_nonzero(singular > cutoff))
        q = vh[rank:].T.copy()
        # pinv(B.T) v supplies lambda for v - P v = B.T lambda.
        multiplier = ((u[:, :rank] / singular[:rank]) @ vh[:rank]
                      if rank else np.zeros((b.shape[0], b.shape[1])))
        scale = max(1.0, float(np.linalg.norm(b)))
        if (np.linalg.norm(b @ q) > 10 * rtol * scale
                or not np.allclose(q.T @ q, np.eye(q.shape[1]), atol=1e-10, rtol=0)):
            raise ValueError("Compiled basis failed its numerical checks")
        q.setflags(write=False)
        multiplier.setflags(write=False)
        payload = str(b.shape).encode() + b.astype('<f8').tobytes() + repr(rtol).encode()
        key = hashlib.sha256(payload).hexdigest()
        return cls(name, b, q, multiplier, rank, rtol, key)

    @property
    def ambient_dimension(self) -> int:
        return self.boundary.shape[1]

    @property
    def latent_dimension(self) -> int:
        return self.basis.shape[1]

    def project(self, values) -> np.ndarray:
        """Apply Q(Q.T v), without building an ambient n-by-n projector."""
        v = vector(values, self.ambient_dimension, "values")
        return self.basis @ (self.basis.T @ v)

    def projection_witness(self, values) -> tuple[np.ndarray, np.ndarray]:
        """Return projected z and lambda; checking needs only B, v, z, lambda."""
        v = vector(values, self.ambient_dimension, "values")
        return self.project(v), self.multiplier_map @ v
