"""An honest, non-learned selection baseline: score ALL candidates.

Validation measurements are consumed by routing, not treated as untouched
final evaluation. This is not a cheap learned router or a discovery head.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ._validation import matrix, vector, positive
from .recurrence import ReconstructionProblem, direct_solve
from .structures import CompiledConstraint


@dataclass(frozen=True)
class RoutingDecision:
    chosen: CompiledConstraint | None
    scores: dict[str, float]
    reason: str


def select_constraint(candidates: list[CompiledConstraint], training: ReconstructionProblem,
                      validation_measurement, validation_observed, *, margin: float = 1e-6) -> RoutingDecision:
    if not candidates or len({c.name for c in candidates}) != len(candidates):
        raise ValueError("Need candidates with unique names")
    a = matrix(validation_measurement, "validation_measurement")
    y = vector(validation_observed, a.shape[0], "validation_observed")
    threshold = positive(margin, "margin")
    if a.shape[1] != training.measurement.shape[1]:
        raise ValueError("Validation dimension does not match training")
    scores = {}
    for candidate in candidates:
        state = direct_solve(training, candidate)
        scores[candidate.name] = float(np.mean((a @ state - y)**2))
    ranked = sorted(candidates, key=lambda c: scores[c.name])
    if len(ranked) > 1 and scores[ranked[1].name] - scores[ranked[0].name] <= threshold:
        return RoutingDecision(None, scores, "refused: validation evidence does not separate candidates")
    return RoutingDecision(ranked[0], scores, "lowest validation MSE; all candidates were solved")
