"""Optional adapter to the separately installed, commit-pinned Universa.

No upstream source is copied. The base demo does not require Universa.
Install with: python -m pip install -e ".[universa]"
"""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from .core import CompiledConstraint


def compile_from_universa(name: str, boundary) -> CompiledConstraint:
    try:
        from universa.operators import nullspace_basis
    except ImportError as exc:
        raise ImportError('Install the pinned adapter extra: pip install -e ".[universa]"') from exc
    local = CompiledConstraint.compile(name, boundary)
    cert = nullspace_basis(np.array(local.boundary, copy=True))
    q = np.array(cert.basis, dtype=np.float64, copy=True)
    if (q.shape != local.basis.shape or not np.isfinite(q).all()
            or not np.allclose(q @ q.T, local.basis @ local.basis.T, atol=1e-10, rtol=0)):
        raise ValueError("Upstream and local numerical nullspaces disagree; refusing")
    q.setflags(write=False)
    return replace(local, basis=q)
