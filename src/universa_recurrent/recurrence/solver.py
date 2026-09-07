"""A transparent baseline: projected gradient descent in reduced coordinates.

Find z minimizing 0.5*||A z-y||^2 + ridge/2*||z||^2, subject to B z=0.
Write z=Q a. Reuse the smaller state a and the same update on every step.
This is classical optimization, NOT a trained neural model or a novel solver.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .._validation import matrix, vector, positive
from ..structures import CompiledConstraint
from ..lingua.trace import make_record


@dataclass(frozen=True)
class ReconstructionProblem:
    measurement: np.ndarray
    observed: np.ndarray
    ridge: float = 0.01

    def __post_init__(self):
        a = matrix(self.measurement, "measurement")
        y = vector(self.observed, a.shape[0], "observed")
        y.setflags(write=False)
        object.__setattr__(self, "measurement", a)
        object.__setattr__(self, "observed", y)
        object.__setattr__(self, "ridge", positive(self.ridge, "ridge"))

    def loss(self, state) -> float:
        z = vector(state, self.measurement.shape[1], "state")
        residual = self.measurement @ z - self.observed
        return float(0.5 * residual @ residual + 0.5 * self.ridge * (z @ z))

    def gradient(self, state) -> np.ndarray:
        z = vector(state, self.measurement.shape[1], "state")
        return self.measurement.T @ (self.measurement @ z - self.observed) + self.ridge * z


def _reduced(problem, constraint):
    if problem.measurement.shape[1] != constraint.ambient_dimension:
        raise ValueError("Measurement and constraint dimensions do not match")
    q = constraint.basis
    aq = problem.measurement @ q
    hessian = aq.T @ aq + problem.ridge * np.eye(q.shape[1])
    rhs = aq.T @ problem.observed
    return hessian, rhs


def direct_solve(problem: ReconstructionProblem, constraint: CompiledConstraint) -> np.ndarray:
    """Strong baseline: solve the small constrained quadratic directly."""
    hessian, rhs = _reduced(problem, constraint)
    coords = np.linalg.solve(hessian, rhs) if rhs.size else rhs.copy()
    return constraint.basis @ coords


@dataclass(frozen=True)
class SolveResult:
    state: np.ndarray
    iterations: int
    stop_reason: str
    record: dict


def solve(problem: ReconstructionProblem, constraint: CompiledConstraint, *,
          max_steps: int = 256, tolerance: float = 1e-9,
          adaptive: bool = True, trace_mode: str = "compact") -> SolveResult:
    """Refine a, stop at a declared stationarity tolerance or a hard budget.

    Full records retain each numerical projection witness. Compact records
    retain only step summaries and a final optimization witness. They CANNOT
    independently certify the omitted intermediate history.
    """
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if type(adaptive) is not bool:
        raise ValueError("adaptive must be a boolean")
    tol = positive(tolerance, "tolerance")
    if trace_mode not in ("compact", "full"):
        raise ValueError("trace_mode must be 'compact' or 'full'")
    hessian, rhs = _reduced(problem, constraint)
    q = constraint.basis
    largest = float(np.linalg.eigvalsh(hessian)[-1]) if rhs.size else 1.0
    step_size = 1.0 / largest
    coords = np.zeros_like(rhs)
    state = q @ coords
    events = []
    # Positive ridge makes the reduced objective strictly convex when dim>0.
    while len(events) < max_steps:
        reduced_gradient = hessian @ coords - rhs
        if adaptive and np.linalg.norm(reduced_gradient) <= tol:
            break
        old_state = state
        coords = coords - step_size * reduced_gradient
        state = q @ coords
        event = {
            "step": len(events) + 1,
            "operation": "projected_gradient_step",
            "structure": constraint.name,
            "objective": problem.loss(state),
            "stationarity": float(np.linalg.norm(hessian @ coords - rhs)),
            "evidence": "runtime_summary_only",
        }
        if trace_mode == "full":
            proposal = old_state - step_size * problem.gradient(old_state)
            event["projection_witness"] = {
                "proposal": proposal.tolist(), "output": state.tolist(),
                "multiplier": (constraint.multiplier_map @ proposal).tolist(),
            }
            event["evidence"] = "numerical_witness_retained"
        events.append(event)
    remaining = float(np.linalg.norm(hessian @ coords - rhs))
    stop_reason = ("stationarity_tolerance" if remaining <= tol else "iteration_budget")
    dual = -constraint.multiplier_map @ problem.gradient(state)
    record = make_record(problem, constraint, state, dual, events, trace_mode,
                         step_size, tol, max_steps, adaptive, stop_reason)
    return SolveResult(state.copy(), len(events), stop_reason, record)
