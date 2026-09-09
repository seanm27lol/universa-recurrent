"""Lingua records for uncertainty-aware multi-hypothesis recurrence.

The record exposes candidate-specific states, residuals, probabilities, route
revisions, and the exact commit/continue/abstain rule.  It does not claim that
probabilities are perfectly calibrated, that the selected structure is true, or
that arbitrary hidden features have human meanings.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from .lingua import array_fingerprint
from .v2 import DecisionPolicy, MultiHypothesisRecurrentNet

_REASON_NAMES = {
    0: "early_threshold",
    1: "final_threshold",
    2: "forced_commit",
    3: "abstain_low_confidence",
}


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def v2_record(
    model: MultiHypothesisRecurrentNet,
    inference: dict[str, object],
    observed: torch.Tensor,
    mask: torch.Tensor,
    names: list[str],
    boundaries: torch.Tensor | np.ndarray,
    *,
    checkpoint_sha256: str,
    checkpoint_format: str,
    policy: DecisionPolicy,
) -> dict[str, Any]:
    """Create a one-example record with candidate-level mathematical evidence."""
    if observed.ndim != 2 or observed.shape[0] != 1 or mask.shape != observed.shape:
        raise ValueError("v2 Lingua records currently require one [1,N] example")
    if len(names) != model.config.num_structures:
        raise ValueError("candidate names do not match model")
    boundary_library = np.asarray(_numpy(boundaries), dtype=np.float64)
    if boundary_library.ndim != 3 or boundary_library.shape[0] != len(names):
        raise ValueError("boundaries must have shape [K,M,N]")

    required = (
        "state",
        "candidate_states",
        "route_probabilities",
        "candidate_residuals",
        "top_routes",
        "routes",
        "committed",
        "decision_steps",
        "decision_reason_codes",
        "route_revisions",
    )
    if not all(torch.is_tensor(inference.get(key)) for key in required):
        raise ValueError("inference output is missing required tensors")
    state = np.asarray(_numpy(inference["state"])[0], dtype=np.float64)
    candidate_states = np.asarray(
        _numpy(inference["candidate_states"])[0], dtype=np.float64
    )
    probabilities = np.asarray(
        _numpy(inference["route_probabilities"])[0], dtype=np.float64
    )
    candidate_residuals = np.asarray(
        _numpy(inference["candidate_residuals"])[0], dtype=np.float64
    )
    top_route = int(_numpy(inference["top_routes"])[0])
    selected_route = int(_numpy(inference["routes"])[0])
    committed = bool(_numpy(inference["committed"])[0])
    decision_step = int(_numpy(inference["decision_steps"])[0])
    reason_code = int(_numpy(inference["decision_reason_codes"])[0])
    route_revisions = int(_numpy(inference["route_revisions"])[0])
    if reason_code not in _REASON_NAMES:
        raise ValueError("unknown decision reason code")
    if candidate_states.shape != (
        len(names), model.config.ambient_dim
    ) or probabilities.shape != (len(names),):
        raise ValueError("candidate output dimensions are invalid")
    if candidate_residuals.shape != (len(names),):
        raise ValueError("candidate residual dimensions are invalid")

    observed_np = np.asarray(_numpy(observed)[0], dtype=np.float64)
    mask_np = np.asarray(_numpy(mask)[0], dtype=np.float64)
    observed_count = max(1.0, float(mask_np.sum()))
    candidate_feasibility = [
        float(np.linalg.norm(boundary_library[index] @ candidate_states[index]))
        for index in range(len(names))
    ]
    final_observed_rms = float(
        np.sqrt(np.sum((mask_np * (state - observed_np)) ** 2) / observed_count)
    )

    events_raw = inference.get("events")
    if not isinstance(events_raw, list):
        raise ValueError("inference events must be a list")
    events: list[dict[str, Any]] = []
    for event in events_raw:
        if not isinstance(event, dict):
            raise ValueError("events must be dictionaries")
        event_copy = dict(event)
        event_copy["hypotheses"] = [
            {
                "index": index,
                "name": names[index],
                "probability": float(event["route_probabilities"][index]),
                "observed_residual": float(event["candidate_residuals"][index]),
                "state": list(event["candidate_states"][index]),
                "structural_feasibility_residual": float(
                    np.linalg.norm(
                        boundary_library[index]
                        @ np.asarray(event["candidate_states"][index], dtype=np.float64)
                    )
                ),
            }
            for index in range(len(names))
        ]
        events.append(event_copy)

    rejected = [
        {
            "index": index,
            "name": names[index],
            "probability": float(probabilities[index]),
            "observed_residual": float(candidate_residuals[index]),
        }
        for index in range(len(names))
        if not committed or index != selected_route
    ]
    final: dict[str, Any] = {
        "decision": "commit" if committed else "abstain",
        "decision_reason": _REASON_NAMES[reason_code],
        "decision_step": decision_step,
        "top_candidate": {
            "index": top_route,
            "name": names[top_route],
            "probability": float(probabilities[top_route]),
        },
        "route_probabilities": probabilities.tolist(),
        "candidate_residuals": candidate_residuals.tolist(),
        "candidate_states": candidate_states.tolist(),
        "candidate_feasibility_residuals": candidate_feasibility,
        "state": state.tolist(),
        "state_kind": "selected_candidate" if committed else "posterior_mixture",
        "observed_coordinate_rms": final_observed_rms,
        "rejected_alternatives": rejected,
    }
    if committed:
        if selected_route != top_route:
            raise ValueError("committed route must equal top candidate")
        selected_boundary = boundary_library[selected_route]
        final["selected_structure"] = {
            "index": selected_route,
            "name": names[selected_route],
            "boundary": selected_boundary.tolist(),
            "boundary_fingerprint": array_fingerprint(selected_boundary),
            "structural_feasibility_residual": candidate_feasibility[selected_route],
            "feasibility_tolerance": 1e-5,
        }
    else:
        if selected_route != -1:
            raise ValueError("abstained output must not claim a selected route")
        final["selected_structure"] = None

    return {
        "schema": "universa-recurrent.neural-v2-trace.v1",
        "scope": {
            "checked_or_checkable": [
                "candidate_boundary_membership",
                "candidate_observed_residuals",
                "probability_and_margin_arithmetic",
                "commit_continue_abstain_policy",
                "final_selected_or_mixture_state",
                "route_revision_and_execution_counters",
                "optional_local_checkpoint_binding",
            ],
            "not_claimed": [
                "perfect_probability_calibration",
                "correctness_of_the_selected_structure",
                "semantic_meaning_of_hidden_features",
                "causal_explanation_of_evidence_updates",
                "network_update_replay",
                "optimality_of_neural_output",
                "remote_execution_attestation",
            ],
        },
        "checkpoint": {
            "sha256": checkpoint_sha256,
            "format": checkpoint_format,
        },
        "model_config": asdict(model.config),
        "policy": policy.as_dict(),
        "structure_library": {
            "names": list(names),
            "boundaries": boundary_library.tolist(),
            "boundary_fingerprints": [
                array_fingerprint(boundary_library[index])
                for index in range(len(names))
            ],
        },
        "input": {
            "observed": observed_np.tolist(),
            "mask": mask_np.tolist(),
        },
        "execution": {
            "mode": str(inference.get("execution_mode")),
            "max_steps": int(inference.get("max_steps")),
            "logical_steps": decision_step,
            "logical_hypothesis_updates": int(
                inference.get("logical_hypothesis_updates")
            ),
            "candidate_update_examples": int(
                inference.get("candidate_update_examples")
            ),
            "recurrent_rounds": int(inference.get("recurrent_rounds")),
            "route_revisions": route_revisions,
            "fixed_depth": bool(inference.get("fixed_depth")),
            "force_commit": bool(inference.get("force_commit")),
        },
        "events": events,
        "final": final,
    }


def describe_v2(record: dict[str, Any]) -> str:
    final = record["final"]
    execution = record["execution"]
    top = final["top_candidate"]
    if final["decision"] == "commit":
        headline = (
            f"Decision: COMMIT to {final['selected_structure']['name']} "
            f"(p={top['probability']:.3f})"
        )
    else:
        headline = (
            f"Decision: ABSTAIN from a single structure; provisional mixture "
            f"(top={top['name']}, p={top['probability']:.3f})"
        )
    lines = [
        "Neural structured recurrence (v2: multiple live hypotheses)",
        headline,
        f"Reason: {final['decision_reason']} at step {final['decision_step']}",
        f"Route revisions before decision: {execution['route_revisions']}",
        (
            "Candidate-update work: "
            f"{execution['candidate_update_examples']} evaluations; "
            f"mode={execution['mode']}"
        ),
    ]
    events = record.get("events", [])
    if events:
        first = events[0]["route_probabilities"]
        last = events[-1]["route_probabilities"]
        names = record["structure_library"]["names"]
        lines.append(
            "Probability path: "
            + "; ".join(
                f"{name} {first[index]:.3f}→{last[index]:.3f}"
                for index, name in enumerate(names)
            )
        )
    if final["decision"] == "commit":
        lines.append(
            "Final selected-structure residual: "
            f"{final['selected_structure']['structural_feasibility_residual']:.3e}"
        )
    else:
        lines.append(
            "No single-subspace feasibility claim is made for the mixture output."
        )
    lines.append(
        "Lingua scope: explicit hypothesis dynamics and mathematical checks only; "
        "hidden-feature semantics are not claimed."
    )
    return "\n".join(lines)


__all__ = ["describe_v2", "v2_record"]
