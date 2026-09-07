"""One small synthetic circulation, with missing and noisy measurements."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .structures import CompiledConstraint, incidence_matrix
from .recurrence import ReconstructionProblem


@dataclass(frozen=True)
class FlowExample:
    problem: ReconstructionProblem
    candidates: list[CompiledConstraint]
    validation_measurement: np.ndarray
    validation_observed: np.ndarray
    truth_for_evaluation_only: np.ndarray


def flow_example(seed: int = 7) -> FlowExample:
    rng = np.random.default_rng(seed)
    # e0:0->1, e1:1->2, e2:2->3, e3:3->0, e4:0->2.
    boundary = incidence_matrix(4, [(0,1), (1,2), (2,3), (3,0), (0,2)])
    truth = np.array([1., 1., 1.4, 1.4, 0.4])
    measurement = np.eye(5)[[0, 2, 4]] * np.array([1., 0.5, 0.8])[:, None]
    observed = measurement @ truth + rng.normal(0, 0.01, 3)
    validation = np.eye(5)[[1, 3]]
    validation_y = validation @ truth + rng.normal(0, 0.01, 2)
    incorrect = boundary.copy()
    incorrect[:, 4] *= -1
    candidates = [CompiledConstraint.compile("balanced_flow", boundary),
                  CompiledConstraint.compile("wrong_diagonal_constraint", incorrect)]
    # Truth is deliberately NOT supplied to the router or recurrent solver.
    return FlowExample(ReconstructionProblem(measurement, observed, ridge=0.01),
                       candidates, validation, validation_y, truth)
