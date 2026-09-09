"""Independent property checks for neural Lingua records.

The checker does not call the recurrent update network. It checks the final
mathematical claims and record consistency. Optional checkpoint binding verifies
that the named structures/configuration came from exact local checkpoint bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any

import numpy as np

from .lingua import array_fingerprint


@dataclass(frozen=True)
class NeuralCheckResult:
    accepted: bool
    reason: str
    checks: dict[str, bool]


def _finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _close(recorded: float, recomputed: float, *, atol: float, rtol: float) -> bool:
    return math.isfinite(recorded) and math.isclose(recorded, recomputed, abs_tol=atol, rel_tol=rtol)


def verify_neural_record(
    record: dict[str, Any],
    *,
    checkpoint: Path | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> NeuralCheckResult:
    """Verify one v1 record; return a failure result instead of trusting text."""
    checks: dict[str, bool] = {}
    try:
        if not isinstance(record, dict) or record.get("schema") != "universa-recurrent.neural-trace.v1":
            raise ValueError("unsupported neural trace schema")
        route = record.get("route")
        selected = record.get("selected_boundary")
        inputs = record.get("input")
        execution = record.get("execution")
        final = record.get("final")
        library = record.get("structure_library")
        checkpoint_record = record.get("checkpoint")
        if not all(isinstance(value, dict) for value in (route, selected, inputs, execution, final, library, checkpoint_record)):
            raise ValueError("record sections are missing")

        boundary = _finite_array(selected.get("matrix"), name="selected boundary", ndim=2)
        state = _finite_array(final.get("state"), name="final state", ndim=1)
        observed = _finite_array(inputs.get("observed"), name="observed", ndim=1)
        mask = _finite_array(inputs.get("mask"), name="mask", ndim=1)
        if state.shape != observed.shape or state.shape != mask.shape:
            raise ValueError("state, observation, and mask shapes differ")
        if boundary.shape[1] != state.shape[0]:
            raise ValueError("boundary width does not match state")
        if not np.all((mask == 0.0) | (mask == 1.0)) or mask.sum() < 1:
            raise ValueError("mask must be binary with at least one observation")

        fingerprint = selected.get("fingerprint")
        checks["selected_boundary_fingerprint"] = (
            isinstance(fingerprint, str) and fingerprint == array_fingerprint(boundary)
        )
        if not checks["selected_boundary_fingerprint"]:
            raise ValueError("selected boundary fingerprint mismatch")

        names = library.get("names")
        boundary_fingerprints = library.get("boundary_fingerprints")
        route_index = route.get("index")
        if isinstance(route_index, bool) or not isinstance(route_index, int):
            raise ValueError("route index must be an integer")
        if not isinstance(names, list) or not isinstance(boundary_fingerprints, list):
            raise ValueError("structure library metadata malformed")
        if not (0 <= route_index < len(names)) or len(boundary_fingerprints) != len(names):
            raise ValueError("route/library dimensions are invalid")
        checks["route_name"] = route.get("name") == names[route_index]
        checks["route_boundary_fingerprint"] = boundary_fingerprints[route_index] == fingerprint
        if not checks["route_name"] or not checks["route_boundary_fingerprint"]:
            raise ValueError("selected route is inconsistent with the recorded library")

        probabilities = _finite_array(route.get("probabilities"), name="route probabilities", ndim=1)
        if probabilities.shape[0] != len(names) or np.any(probabilities < 0):
            raise ValueError("route probabilities have invalid shape or sign")
        checks["route_probabilities"] = math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-5, rel_tol=1e-5)
        if not checks["route_probabilities"]:
            raise ValueError("route probabilities do not sum to one")
        checks["route_argmax"] = int(np.argmax(probabilities)) == route_index
        if not checks["route_argmax"]:
            raise ValueError("recorded route is not the probability argmax")

        feasibility = float(np.linalg.norm(boundary @ state))
        recorded_feasibility = float(final.get("structural_feasibility_residual"))
        tolerance = float(final.get("feasibility_tolerance"))
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("invalid feasibility tolerance")
        checks["feasibility_value"] = _close(recorded_feasibility, feasibility, atol=atol, rtol=rtol)
        checks["feasibility_pass"] = feasibility <= tolerance
        if not checks["feasibility_value"] or not checks["feasibility_pass"]:
            raise ValueError("final structural feasibility check failed")

        observed_rms = float(np.sqrt(np.sum((mask * (state - observed)) ** 2) / mask.sum()))
        checks["observed_residual"] = _close(
            float(final.get("observed_coordinate_rms")), observed_rms, atol=atol, rtol=rtol
        )
        if not checks["observed_residual"]:
            raise ValueError("final observed-coordinate residual mismatch")

        events = record.get("events")
        if not isinstance(events, list):
            raise ValueError("events must be a list")
        fixed_depth = execution.get("fixed_depth")
        mode = execution.get("mode")
        max_steps = execution.get("max_steps")
        min_steps = execution.get("min_steps")
        logical_steps = execution.get("logical_steps")
        update_examples = execution.get("update_examples")
        recurrent_rounds = execution.get("recurrent_rounds")
        halt_threshold = execution.get("halt_threshold")
        for name, value in (
            ("max_steps", max_steps),
            ("min_steps", min_steps),
            ("logical_steps", logical_steps),
            ("update_examples", update_examples),
            ("recurrent_rounds", recurrent_rounds),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if not isinstance(fixed_depth, bool):
            raise ValueError("fixed_depth must be bool")
        if mode not in ("dense", "compact"):
            raise ValueError("execution mode must be dense or compact")
        # A Lingua record currently contains one example and enables event
        # logging, so both implementations stop once that example is done.
        if update_examples != logical_steps or recurrent_rounds != len(events):
            raise ValueError("execution counters do not match the retained one-example trace")
        checks["execution_counters"] = True
        halt_threshold = float(halt_threshold)
        if not math.isfinite(halt_threshold) or not (1 <= min_steps <= max_steps):
            raise ValueError("invalid execution controls")
        if logical_steps != len(events) or not (1 <= logical_steps <= max_steps):
            raise ValueError("event count does not match logical steps")

        halted = False
        last_state = None
        for expected_step, event in enumerate(events, start=1):
            if not isinstance(event, dict) or event.get("step") != expected_step:
                raise ValueError("event steps are not contiguous")
            if event.get("active_before") != 1:
                raise ValueError("one-example trace must have one active sample per recorded step")
            if event.get("route_index") != route_index:
                raise ValueError("event route differs from final route")
            event_state = _finite_array(event.get("state_after"), name="event state", ndim=1)
            if event_state.shape != state.shape:
                raise ValueError("event state has wrong shape")
            event_rms = float(np.sqrt(np.sum((mask * (event_state - observed)) ** 2) / mask.sum()))
            if not _close(float(event.get("observed_residual")), event_rms, atol=atol, rtol=rtol):
                raise ValueError("event residual does not match event state")
            halt_probability = float(event.get("halt_probability"))
            if not math.isfinite(halt_probability) or not (0.0 <= halt_probability <= 1.0):
                raise ValueError("invalid event halt probability")
            event_halted = bool(event.get("halted_after"))
            expected_halt = (not fixed_depth) and expected_step >= min_steps and halt_probability >= halt_threshold
            if event_halted != expected_halt:
                raise ValueError("event halt decision is inconsistent with threshold")
            if event.get("newly_halted") != int(event_halted):
                raise ValueError("event halt count is inconsistent")
            if halted:
                raise ValueError("events continue after a halt")
            halted = event_halted
            last_state = event_state
        if last_state is None or not np.allclose(last_state, state, atol=atol, rtol=rtol):
            raise ValueError("final event state differs from final record state")
        if fixed_depth:
            if logical_steps != max_steps or halted:
                raise ValueError("fixed-depth trace ended inconsistently")
        elif logical_steps < max_steps and not halted:
            raise ValueError("adaptive trace stopped without a threshold crossing")
        checks["event_sequence"] = True

        if checkpoint is not None:
            from .train import load_checkpoint

            _, metadata, _ = load_checkpoint(checkpoint, device_name="cpu")
            checks["checkpoint_sha256"] = checkpoint_record.get("sha256") == metadata.get("checkpoint_sha256")
            checks["checkpoint_format"] = checkpoint_record.get("format") == metadata.get("format")
            checks["checkpoint_config"] = record.get("model_config") == metadata.get("config")
            checks["checkpoint_names"] = names == metadata.get("names")
            boundaries = metadata.get("boundaries")
            if hasattr(boundaries, "detach"):
                boundaries = boundaries.detach().cpu().numpy()
            boundaries = _finite_array(boundaries, name="checkpoint boundaries", ndim=3)
            checks["checkpoint_boundary"] = np.array_equal(
                np.asarray(boundaries[route_index], dtype=np.float64), boundary
            )
            if not all(
                checks[key]
                for key in (
                    "checkpoint_sha256",
                    "checkpoint_format",
                    "checkpoint_config",
                    "checkpoint_names",
                    "checkpoint_boundary",
                )
            ):
                raise ValueError("record is not bound to the supplied checkpoint")

        return NeuralCheckResult(True, "neural final claims and record consistency verified", checks)
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        return NeuralCheckResult(False, str(error), checks)


__all__ = ["NeuralCheckResult", "verify_neural_record"]
