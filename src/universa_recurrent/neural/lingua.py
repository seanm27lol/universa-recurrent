"""Conservative Lingua records for learned recurrence.

The record names explicit structures and measured dynamics. It does not assign
human concepts to arbitrary hidden features or claim a causal explanation of
the learned router/update network.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Any

import numpy as np
import torch


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def array_fingerprint(value: Any) -> str:
    """Stable SHA-256 over a canonical float64 array representation."""
    array = np.ascontiguousarray(_numpy(value), dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(b"\0float64\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def neural_record(
    model: Any,
    inference: dict,
    observed: torch.Tensor,
    mask: torch.Tensor,
    names: list[str],
    boundaries: torch.Tensor | np.ndarray,
    *,
    checkpoint_sha256: str,
    checkpoint_format: str,
) -> dict[str, Any]:
    """Create a one-example neural trace whose final claims are checkable."""
    if observed.ndim != 2 or observed.shape[0] != 1 or mask.shape != observed.shape:
        raise ValueError("neural Lingua records currently require one [1,N] example")
    routes = inference.get("routes")
    states = inference.get("state")
    probabilities = inference.get("route_probabilities")
    steps_taken = inference.get("steps_taken")
    if not all(torch.is_tensor(value) for value in (routes, states, probabilities, steps_taken)):
        raise ValueError("inference output is missing required tensors")
    route = int(routes[0].item())
    if not (0 <= route < len(names)):
        raise ValueError("route index out of range")

    boundary_library = _numpy(boundaries)
    if boundary_library.ndim != 3 or boundary_library.shape[0] != len(names):
        raise ValueError("boundaries must have shape [K,M,N]")
    boundary = np.asarray(boundary_library[route], dtype=np.float64)
    state = np.asarray(_numpy(states[0]), dtype=np.float64)
    observed_np = np.asarray(_numpy(observed[0]), dtype=np.float64)
    mask_np = np.asarray(_numpy(mask[0]), dtype=np.float64)
    if state.shape != observed_np.shape or boundary.shape[1] != state.shape[0]:
        raise ValueError("record dimensions do not agree")

    feasibility = float(np.linalg.norm(boundary @ state))
    observed_count = max(1.0, float(mask_np.sum()))
    observed_rms = float(np.sqrt(np.sum((mask_np * (state - observed_np)) ** 2) / observed_count))
    route_probabilities = np.asarray(_numpy(probabilities[0]), dtype=np.float64)
    events = inference.get("events")
    if not isinstance(events, list):
        raise ValueError("inference events must be a list")

    config = asdict(model.config)
    return {
        "schema": "universa-recurrent.neural-trace.v1",
        "scope": {
            "checked_or_checkable": [
                "selected_boundary_membership",
                "final_observed_coordinate_residual",
                "event_sequence_consistency",
                "optional_local_checkpoint_binding",
            ],
            "not_claimed": [
                "semantic_meaning_of_hidden_features",
                "causal_explanation_of_router",
                "optimality_of_neural_output",
                "correctness_of_selected_structure",
                "network_update_replay",
                "remote_execution_attestation",
            ],
        },
        "checkpoint": {
            "sha256": checkpoint_sha256,
            "format": checkpoint_format,
        },
        "model_config": config,
        "structure_library": {
            "names": list(names),
            "boundary_fingerprints": [
                array_fingerprint(boundary_library[index]) for index in range(len(names))
            ],
        },
        "route": {
            "index": route,
            "name": names[route],
            "probabilities": route_probabilities.tolist(),
        },
        "selected_boundary": {
            "matrix": boundary.tolist(),
            "fingerprint": array_fingerprint(boundary),
        },
        "input": {
            "observed": observed_np.tolist(),
            "mask": mask_np.tolist(),
        },
        "execution": {
            "mode": str(inference.get("execution_mode")),
            "fixed_depth": bool(inference.get("fixed_depth")),
            "max_steps": int(inference.get("max_steps")),
            "min_steps": int(inference.get("min_steps")),
            "halt_threshold": float(inference.get("halt_threshold")),
            "logical_steps": int(steps_taken[0].item()),
            "update_examples": int(inference.get("update_examples")),
            "recurrent_rounds": int(inference.get("recurrent_rounds")),
        },
        "events": events,
        "final": {
            "state": state.tolist(),
            "structural_feasibility_residual": feasibility,
            "observed_coordinate_rms": observed_rms,
            "feasibility_tolerance": 1e-5,
        },
    }


def describe_neural(record: dict[str, Any]) -> str:
    route = record["route"]
    execution = record["execution"]
    final = record["final"]
    lines = [
        "Neural structured recurrence (v1)",
        f"Structure selected: {route['name']} (p={max(route['probabilities']):.3f})",
        f"Logical recurrent updates: {execution['logical_steps']}",
        f"Execution mode: {execution['mode']}; update examples: {execution['update_examples']}",
        f"Final structural feasibility residual: {final['structural_feasibility_residual']:.3e}",
        "Lingua scope: explicit structures and measured dynamics; hidden-feature semantics are not claimed.",
    ]
    events = record.get("events", [])
    if events:
        first, last = events[0], events[-1]
        lines.append(
            "Observed-residual summary: "
            f"{first.get('observed_residual', first.get('mean_observed_residual', 0.0)):.4f} -> "
            f"{last.get('observed_residual', last.get('mean_observed_residual', 0.0)):.4f}"
        )
        lines.append(
            "Halting signal summary: "
            f"{first.get('halt_probability', first.get('mean_halt_probability', 0.0)):.3f} -> "
            f"{last.get('halt_probability', last.get('mean_halt_probability', 0.0)):.3f}"
        )
    return "\n".join(lines)


__all__ = ["array_fingerprint", "describe_neural", "neural_record"]
