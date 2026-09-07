"""Check properties with matrix-vector products, not a rerun of the solve.

Projection: B z=0 and v-z=B.T lambda.
Optimality: B z=0 and grad(loss)(z)+B.T lambda=0.
Exact equations certify the mathematical properties. Here tolerance-based
checks are numerical evidence; no unconditional distance-to-truth bound is claimed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class CheckResult:
    accepted: bool
    reason: str
    feasibility: float = float("inf")
    stationarity: float = float("inf")
    intermediate_checks: int = 0


def _array(value, ndim, name):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != ndim or not np.isfinite(a).all():
        raise ValueError(f"{name} has an invalid shape or nonfinite values")
    return a


def _inputs(boundary, state, multiplier, atol, rtol):
    b, z, lam = _array(boundary, 2, "boundary"), _array(state, 1, "state"), _array(multiplier, 1, "multiplier")
    if b.shape != (lam.size, z.size) or z.size == 0:
        raise ValueError("Witness dimensions do not match")
    if (not np.isfinite([atol, rtol]).all() or atol <= 0 or rtol < 0
            or isinstance(atol, bool) or isinstance(rtol, bool)):
        raise ValueError("Tolerances must be finite with atol>0 and rtol>=0")
    feasibility = float(np.linalg.norm(b @ z))
    limit = atol + rtol * max(1.0, float(np.linalg.norm(b) * np.linalg.norm(z)))
    return b, z, lam, feasibility, limit


def check_projection(boundary, proposal, state, multiplier, *, atol=1e-8, rtol=1e-8) -> CheckResult:
    try:
        b, z, lam, feas, limit = _inputs(boundary, state, multiplier, atol, rtol)
        v = _array(proposal, 1, "proposal")
        if v.shape != z.shape:
            raise ValueError("Proposal and output dimensions do not match")
        residual = float(np.linalg.norm(v - z - b.T @ lam))
        stat_limit = atol + rtol * max(1.0, float(np.linalg.norm(v)))
        ok = (np.isfinite([feas, limit, residual, stat_limit]).all()
              and feas <= limit and residual <= stat_limit)
        return CheckResult(bool(ok), "projection checked within tolerance" if ok else "projection check failed", feas, residual)
    except (TypeError, ValueError, OverflowError) as exc:
        return CheckResult(False, str(exc))


def check_optimality(measurement, observed, boundary, state, multiplier, *, ridge,
                     atol=1e-8, rtol=1e-8) -> CheckResult:
    try:
        b, z, lam, feas, limit = _inputs(boundary, state, multiplier, atol, rtol)
        a, y = _array(measurement, 2, "measurement"), _array(observed, 1, "observed")
        if a.shape != (y.size, z.size) or y.size == 0:
            raise ValueError("Measurement dimensions do not match")
        if isinstance(ridge, bool) or not np.isfinite(ridge) or ridge <= 0:
            raise ValueError("ridge must be positive and finite")
        gradient = a.T @ (a @ z - y) + ridge * z
        residual = float(np.linalg.norm(gradient + b.T @ lam))
        stat_limit = atol + rtol * max(1.0, float(np.linalg.norm(a.T @ y)))
        ok = (np.isfinite([feas, limit, residual, stat_limit]).all()
              and feas <= limit and residual <= stat_limit)
        return CheckResult(bool(ok), "final optimality checked within tolerance" if ok else "final optimality check failed", feas, residual)
    except (TypeError, ValueError, OverflowError) as exc:
        return CheckResult(False, str(exc))


def verify_record(record, *, atol=1e-8, rtol=1e-8) -> CheckResult:
    """Full mode checks the retained update chain; compact checks the final claim.

    JSON is untrusted. Schema/shape/finite checks fail closed. These checks
    do not authenticate where a record came from, or establish which of
    several valid histories physically ran on some other machine.
    """
    try:
        if not isinstance(record, dict) or record.get("schema") != "universa-recurrent.trace.v1":
            raise ValueError("Unknown trace schema")
        mode = record["mode"]
        expected_scope = ("full_numeric_witnesses" if mode == "full" else
                          "not_retained; summaries_are_not_independently_verified")
        if record.get("intermediate_evidence") != expected_scope:
            raise ValueError("Declared evidence scope does not match the trace mode")
        if mode not in ("compact", "full"):
            raise ValueError("Unknown trace mode")
        if record.get("claim") != "approximate_optimality_for_the_supplied_constrained_quadratic":
            raise ValueError("Unsupported claim")
        p, f, s = record["problem"], record["final"], record["solver"]
        a, y = _array(p["measurement"], 2, "measurement"), _array(p["observed"], 1, "observed")
        shape = p["boundary_shape"]
        if not isinstance(shape, list) or len(shape) != 2 or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError("Invalid boundary shape")
        b = np.asarray(p["boundary"], dtype=np.float64)
        if shape[0] == 0 and b.shape == (0,):
            b = b.reshape(0, shape[1])
        if b.shape != tuple(shape):
            raise ValueError("Boundary shape does not match its values")
        result = check_optimality(a, y, b, f["state"], f["multiplier"], ridge=p["ridge"], atol=atol, rtol=rtol)
        if not result.accepted:
            return result
        events = record["events"]
        if not isinstance(events, list) or type(s["iterations"]) is not int or s["iterations"] != len(events):
            raise ValueError("Iteration count does not match retained events")
        if (type(s["max_steps"]) is not int or s["max_steps"] < len(events)
                or type(s["adaptive"]) is not bool or s["stop_reason"] not in ("stationarity_tolerance", "iteration_budget")):
            raise ValueError("Invalid solver metadata")
        if not np.isfinite(s["tolerance"]) or s["tolerance"] <= 0:
            raise ValueError("Invalid solver tolerance")
        eta = s["step_size"]
        if isinstance(eta, bool) or not np.isfinite(eta) or eta <= 0:
            raise ValueError("Invalid step size")
        previous = np.zeros(a.shape[1])
        checked = 0
        for index, event in enumerate(events, 1):
            if (type(event["step"]) is not int or event["step"] != index
                    or event["operation"] != "projected_gradient_step"
                    or event["structure"] != p["structure_name"]):
                raise ValueError("Invalid or reordered event")
            if not np.isfinite([event["objective"], event["stationarity"]]).all() or min(event["objective"], event["stationarity"]) < 0:
                raise ValueError("Invalid event summary")
            if mode == "compact":
                if "projection_witness" in event or event["evidence"] != "runtime_summary_only":
                    raise ValueError("Compact evidence metadata does not match its scope")
                continue
            if event["evidence"] != "numerical_witness_retained":
                raise ValueError("Missing full numerical evidence")
            witness = event["projection_witness"]
            expected = previous - eta * (a.T @ (a @ previous - y) + p["ridge"] * previous)
            proposal = _array(witness["proposal"], 1, "proposal")
            if proposal.shape != expected.shape or not np.allclose(proposal, expected, atol=atol, rtol=rtol):
                raise ValueError("Recorded proposal is not the claimed update from the prior state")
            test = check_projection(b, proposal, witness["output"], witness["multiplier"], atol=atol, rtol=rtol)
            if not test.accepted:
                raise ValueError(f"Projection witness failed at step {index}")
            previous = _array(witness["output"], 1, "output")
            residual = a @ previous - y
            objective = float(0.5 * residual @ residual + 0.5 * p["ridge"] * (previous @ previous))
            if not np.isclose(event["objective"], objective, atol=atol, rtol=rtol):
                raise ValueError("Recorded objective does not match the step output")
            checked += 1
        if mode == "full" and not np.allclose(previous, f["state"], atol=atol, rtol=rtol):
            raise ValueError("Final state does not match the retained update chain")
        detail = "full numerical update chain" if mode == "full" else "final claim only; intermediate summaries not verified"
        return CheckResult(True, detail, result.feasibility, result.stationarity, checked)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        return CheckResult(False, f"Invalid record: {exc}")
