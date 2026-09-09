"""Independent checks for neural v2 Lingua records.

The checker validates candidate-specific mathematical claims and the explicit
commit/continue/abstain rule. It does not run the neural network or infer the
meaning of hidden features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any

import numpy as np

from .lingua import array_fingerprint
from .v2 import DecisionPolicy


@dataclass(frozen=True)
class V2CheckResult:
    accepted: bool
    reason: str
    checks: dict[str, bool]


def _array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _close(first: float, second: float, *, atol: float, rtol: float) -> bool:
    return math.isfinite(first) and math.isclose(first, second, abs_tol=atol, rel_tol=rtol)


def verify_v2_record(
    record: dict[str, Any],
    *,
    checkpoint: Path | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> V2CheckResult:
    checks: dict[str, bool] = {}
    try:
        if not isinstance(record, dict) or record.get("schema") != "universa-recurrent.neural-v2-trace.v1":
            raise ValueError("unsupported neural v2 trace schema")
        library = record.get("structure_library")
        inputs = record.get("input")
        execution = record.get("execution")
        final = record.get("final")
        checkpoint_record = record.get("checkpoint")
        policy_data = record.get("policy")
        if not all(
            isinstance(value, dict)
            for value in (library, inputs, execution, final, checkpoint_record, policy_data)
        ):
            raise ValueError("record sections are missing")
        policy = DecisionPolicy(**policy_data)

        names = library.get("names")
        fingerprints = library.get("boundary_fingerprints")
        if not isinstance(names, list) or len(names) < 2 or not all(isinstance(name, str) and name for name in names):
            raise ValueError("structure names are malformed")
        boundaries = _array(library.get("boundaries"), name="boundaries", ndim=3)
        if boundaries.shape[0] != len(names):
            raise ValueError("boundary library size does not match names")
        if not isinstance(fingerprints, list) or len(fingerprints) != len(names):
            raise ValueError("boundary fingerprints are malformed")
        checks["library_fingerprints"] = all(
            isinstance(fingerprints[index], str)
            and fingerprints[index] == array_fingerprint(boundaries[index])
            for index in range(len(names))
        )
        if not checks["library_fingerprints"]:
            raise ValueError("boundary library fingerprint mismatch")

        observed = _array(inputs.get("observed"), name="observed", ndim=1)
        mask = _array(inputs.get("mask"), name="mask", ndim=1)
        if observed.shape != mask.shape or boundaries.shape[2] != observed.shape[0]:
            raise ValueError("input and boundary dimensions do not agree")
        if not np.all((mask == 0.0) | (mask == 1.0)) or mask.sum() < 1:
            raise ValueError("mask must be binary with at least one observation")
        observed_count = float(mask.sum())

        mode = execution.get("mode")
        if mode not in ("dense", "compact"):
            raise ValueError("execution mode must be dense or compact")
        fixed_depth = execution.get("fixed_depth")
        force_commit = execution.get("force_commit")
        if not isinstance(fixed_depth, bool) or not isinstance(force_commit, bool):
            raise ValueError("fixed_depth and force_commit must be bool")
        max_steps = _integer(execution.get("max_steps"), name="max_steps", minimum=1)
        logical_steps = _integer(execution.get("logical_steps"), name="logical_steps", minimum=1)
        logical_hypothesis_updates = _integer(
            execution.get("logical_hypothesis_updates"),
            name="logical_hypothesis_updates",
            minimum=1,
        )
        candidate_update_examples = _integer(
            execution.get("candidate_update_examples"),
            name="candidate_update_examples",
            minimum=1,
        )
        recurrent_rounds = _integer(
            execution.get("recurrent_rounds"), name="recurrent_rounds", minimum=1
        )
        route_revisions_recorded = _integer(
            execution.get("route_revisions"), name="route_revisions", minimum=0
        )
        if not (1 <= logical_steps <= max_steps) or policy.min_steps > max_steps:
            raise ValueError("logical/policy steps are inconsistent")
        expected_logical = logical_steps * len(names)
        checks["logical_hypothesis_updates"] = logical_hypothesis_updates == expected_logical
        if mode == "compact":
            checks["candidate_update_examples"] = candidate_update_examples == expected_logical
            checks["recurrent_rounds"] = recurrent_rounds == logical_steps
        else:
            checks["candidate_update_examples"] = candidate_update_examples == max_steps * len(names)
            checks["recurrent_rounds"] = recurrent_rounds == max_steps
        if not all(
            checks[key]
            for key in (
                "logical_hypothesis_updates",
                "candidate_update_examples",
                "recurrent_rounds",
            )
        ):
            raise ValueError("execution counters are inconsistent")

        events = record.get("events")
        if not isinstance(events, list) or len(events) != logical_steps:
            raise ValueError("events must match logical decision steps")
        previous_top: int | None = None
        revisions = 0
        last_states: np.ndarray | None = None
        last_probabilities: np.ndarray | None = None
        last_residuals: np.ndarray | None = None
        last_decision: str | None = None

        for expected_step, event in enumerate(events, start=1):
            if not isinstance(event, dict) or event.get("step") != expected_step:
                raise ValueError("event steps are not contiguous")
            decision = event.get("decision")
            if decision not in ("continue", "commit", "abstain"):
                raise ValueError("event decision is invalid")
            states = _array(event.get("candidate_states"), name="candidate states", ndim=2)
            probabilities = _array(event.get("route_probabilities"), name="route probabilities", ndim=1)
            residuals = _array(event.get("candidate_residuals"), name="candidate residuals", ndim=1)
            if states.shape != (len(names), observed.shape[0]):
                raise ValueError("candidate states have wrong shape")
            if probabilities.shape != (len(names),) or residuals.shape != (len(names),):
                raise ValueError("candidate probability/residual shape mismatch")
            if np.any(probabilities < 0.0) or not math.isclose(
                float(probabilities.sum()), 1.0, abs_tol=1e-5, rel_tol=1e-5
            ):
                raise ValueError("event probabilities are invalid")
            recomputed_residuals = np.sqrt(
                np.sum((mask[None, :] * (states - observed[None, :])) ** 2, axis=-1)
                / observed_count
            )
            if not np.allclose(residuals, recomputed_residuals, atol=atol, rtol=rtol):
                raise ValueError("candidate residuals do not match candidate states")
            feasibility = np.asarray(
                [np.linalg.norm(boundaries[index] @ states[index]) for index in range(len(names))]
            )
            if np.any(feasibility > 1e-5):
                raise ValueError("candidate state violates its own structure")

            top_order = np.argsort(-probabilities, kind="stable")
            top = int(top_order[0])
            pmax = float(probabilities[top])
            margin = pmax - float(probabilities[top_order[1]])
            if event.get("top_route") != top:
                raise ValueError("event top route is incorrect")
            if not _close(float(event.get("top_probability")), pmax, atol=atol, rtol=rtol):
                raise ValueError("event top probability is incorrect")
            if not _close(float(event.get("margin")), margin, atol=atol, rtol=rtol):
                raise ValueError("event margin is incorrect")

            hypotheses = event.get("hypotheses")
            if not isinstance(hypotheses, list) or len(hypotheses) != len(names):
                raise ValueError("event hypothesis descriptions are malformed")
            for index, hypothesis in enumerate(hypotheses):
                if not isinstance(hypothesis, dict):
                    raise ValueError("hypothesis entry must be a dictionary")
                if hypothesis.get("index") != index or hypothesis.get("name") != names[index]:
                    raise ValueError("hypothesis identity is inconsistent")
                if not _close(float(hypothesis.get("probability")), float(probabilities[index]), atol=atol, rtol=rtol):
                    raise ValueError("hypothesis probability mismatch")
                if not _close(float(hypothesis.get("observed_residual")), float(residuals[index]), atol=atol, rtol=rtol):
                    raise ValueError("hypothesis residual mismatch")
                hypothesis_state = _array(hypothesis.get("state"), name="hypothesis state", ndim=1)
                if not np.allclose(hypothesis_state, states[index], atol=atol, rtol=rtol):
                    raise ValueError("hypothesis state mismatch")
                if not _close(
                    float(hypothesis.get("structural_feasibility_residual")),
                    float(feasibility[index]),
                    atol=atol,
                    rtol=rtol,
                ):
                    raise ValueError("hypothesis feasibility mismatch")

            early_eligible = (
                expected_step >= policy.min_steps
                and pmax >= policy.commit_probability
                and margin >= policy.commit_margin
            )
            is_last_event = expected_step == logical_steps
            if not is_last_event:
                if decision != "continue":
                    raise ValueError("nonfinal event cannot decide")
                if early_eligible and not fixed_depth:
                    raise ValueError("trace continued after early commitment threshold")
            else:
                if decision == "continue":
                    raise ValueError("final event must commit or abstain")
                if expected_step < max_steps:
                    if fixed_depth or not early_eligible or decision != "commit":
                        raise ValueError("early final decision is inconsistent")
                else:
                    final_eligible = (
                        pmax >= policy.final_probability
                        and margin >= policy.final_margin
                    )
                    expected_commit = force_commit or ((not fixed_depth and early_eligible) or final_eligible)
                    if (decision == "commit") != expected_commit:
                        raise ValueError("final commit/abstain decision violates policy")

            if previous_top is not None and previous_top != top:
                revisions += 1
            previous_top = top
            last_states = states
            last_probabilities = probabilities
            last_residuals = residuals
            last_decision = decision

        if last_states is None or last_probabilities is None or last_residuals is None:
            raise ValueError("trace has no final event")
        checks["event_sequence"] = True
        checks["route_revisions"] = revisions == route_revisions_recorded
        if not checks["route_revisions"]:
            raise ValueError("route revision count mismatch")

        final_decision = final.get("decision")
        if final_decision != last_decision or final_decision not in ("commit", "abstain"):
            raise ValueError("final decision differs from event trace")
        decision_step = _integer(final.get("decision_step"), name="final decision_step", minimum=1)
        if decision_step != logical_steps:
            raise ValueError("final decision step mismatch")
        final_probabilities = _array(final.get("route_probabilities"), name="final probabilities", ndim=1)
        final_residuals = _array(final.get("candidate_residuals"), name="final residuals", ndim=1)
        final_states = _array(final.get("candidate_states"), name="final candidate states", ndim=2)
        if not np.allclose(final_probabilities, last_probabilities, atol=atol, rtol=rtol):
            raise ValueError("final probabilities differ from final event")
        if not np.allclose(final_residuals, last_residuals, atol=atol, rtol=rtol):
            raise ValueError("final residuals differ from final event")
        if not np.allclose(final_states, last_states, atol=atol, rtol=rtol):
            raise ValueError("final candidate states differ from final event")

        top = int(np.argmax(final_probabilities))
        top_candidate = final.get("top_candidate")
        if not isinstance(top_candidate, dict):
            raise ValueError("top candidate metadata missing")
        if top_candidate.get("index") != top or top_candidate.get("name") != names[top]:
            raise ValueError("final top candidate identity mismatch")
        if not _close(float(top_candidate.get("probability")), float(final_probabilities[top]), atol=atol, rtol=rtol):
            raise ValueError("final top probability mismatch")

        final_state = _array(final.get("state"), name="final state", ndim=1)
        if final_state.shape != observed.shape:
            raise ValueError("final state has wrong shape")
        if final_decision == "commit":
            selected = final.get("selected_structure")
            if not isinstance(selected, dict):
                raise ValueError("committed record lacks selected structure")
            selected_index = _integer(selected.get("index"), name="selected index", minimum=0)
            if selected_index != top or selected.get("name") != names[top]:
                raise ValueError("selected structure is not final argmax")
            if final.get("state_kind") != "selected_candidate":
                raise ValueError("committed state kind is invalid")
            if not np.allclose(final_state, final_states[top], atol=atol, rtol=rtol):
                raise ValueError("committed state differs from selected candidate")
            selected_boundary = _array(selected.get("boundary"), name="selected boundary", ndim=2)
            if not np.array_equal(selected_boundary, boundaries[top]):
                raise ValueError("selected boundary differs from library")
            if selected.get("boundary_fingerprint") != fingerprints[top]:
                raise ValueError("selected boundary fingerprint mismatch")
            feasibility = float(np.linalg.norm(selected_boundary @ final_state))
            tolerance = float(selected.get("feasibility_tolerance"))
            if not math.isfinite(tolerance) or tolerance <= 0.0 or feasibility > tolerance:
                raise ValueError("selected state feasibility failed")
            if not _close(
                float(selected.get("structural_feasibility_residual")),
                feasibility,
                atol=atol,
                rtol=rtol,
            ):
                raise ValueError("selected feasibility value mismatch")
            checks["selected_structure"] = True
        else:
            if final.get("selected_structure") is not None:
                raise ValueError("abstained record must not select a structure")
            if final.get("state_kind") != "posterior_mixture":
                raise ValueError("abstained state kind is invalid")
            mixture = np.einsum("k,kn->n", final_probabilities, final_states)
            if not np.allclose(final_state, mixture, atol=atol, rtol=rtol):
                raise ValueError("abstained state is not the recorded posterior mixture")
            checks["posterior_mixture"] = True

        final_rms = float(
            np.sqrt(np.sum((mask * (final_state - observed)) ** 2) / observed_count)
        )
        checks["final_observed_residual"] = _close(
            float(final.get("observed_coordinate_rms")), final_rms, atol=atol, rtol=rtol
        )
        if not checks["final_observed_residual"]:
            raise ValueError("final observed-coordinate residual mismatch")

        rejected = final.get("rejected_alternatives")
        if not isinstance(rejected, list):
            raise ValueError("rejected alternatives must be a list")
        expected_rejected = set(range(len(names)))
        if final_decision == "commit":
            expected_rejected.remove(top)
        found_rejected: set[int] = set()
        for item in rejected:
            if not isinstance(item, dict):
                raise ValueError("rejected alternative is malformed")
            index = _integer(item.get("index"), name="rejected index", minimum=0)
            if not (0 <= index < len(names)) or index in found_rejected:
                raise ValueError("rejected alternative index invalid or duplicated")
            found_rejected.add(index)
            if item.get("name") != names[index]:
                raise ValueError("rejected alternative name mismatch")
            if not _close(float(item.get("probability")), float(final_probabilities[index]), atol=atol, rtol=rtol):
                raise ValueError("rejected alternative probability mismatch")
            if not _close(float(item.get("observed_residual")), float(final_residuals[index]), atol=atol, rtol=rtol):
                raise ValueError("rejected alternative residual mismatch")
        checks["rejected_alternatives"] = found_rejected == expected_rejected
        if not checks["rejected_alternatives"]:
            raise ValueError("rejected alternatives do not match decision")

        reason = final.get("decision_reason")
        if final_decision == "abstain" and reason != "abstain_low_confidence":
            raise ValueError("abstention reason is inconsistent")
        if final_decision == "commit" and reason not in (
            "early_threshold",
            "final_threshold",
            "forced_commit",
        ):
            raise ValueError("commit reason is inconsistent")
        expected_reason = (
            "abstain_low_confidence" if final_decision == "abstain"
            else "early_threshold" if logical_steps < max_steps
            else "forced_commit" if force_commit
            else "final_threshold"
        )
        if reason != expected_reason:
            raise ValueError("decision reason does not match timing and policy")
        checks["decision_policy"] = True

        if checkpoint is not None:
            from .v2_train import load_v2_checkpoint

            model, _, metadata, _ = load_v2_checkpoint(checkpoint, device_name="cpu")
            checks["checkpoint_sha256"] = checkpoint_record.get("sha256") == metadata.get("checkpoint_sha256")
            checks["checkpoint_format"] = checkpoint_record.get("format") == metadata.get("format")
            checks["checkpoint_config"] = record.get("model_config") == model.config.__dict__
            checks["checkpoint_names"] = names == metadata.get("names")
            checkpoint_boundaries = _array(metadata.get("boundaries"), name="checkpoint boundaries", ndim=3)
            checks["checkpoint_boundaries"] = np.array_equal(checkpoint_boundaries, boundaries)
            calibration = metadata.get("calibration")
            checks["checkpoint_policy"] = (
                isinstance(calibration, dict)
                and calibration.get("policy") == policy.as_dict()
            )
            required = (
                "checkpoint_sha256",
                "checkpoint_format",
                "checkpoint_config",
                "checkpoint_names",
                "checkpoint_boundaries",
                "checkpoint_policy",
            )
            if not all(checks[name] for name in required):
                raise ValueError("record is not bound to the supplied v2 checkpoint")

        return V2CheckResult(
            True,
            "v2 hypothesis dynamics, decision rule, and final mathematical claims verified",
            checks,
        )
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as error:
        return V2CheckResult(False, str(error), checks)


__all__ = ["V2CheckResult", "verify_v2_record"]
