"""Readable JSON records with an explicit boundary on retained evidence."""
from __future__ import annotations
import json
from pathlib import Path

SCHEMA = "universa-recurrent.trace.v1"


def make_record(problem, constraint, state, dual, events, mode,
                step_size, tolerance, max_steps, adaptive, stop_reason) -> dict:
    return {
        "schema": SCHEMA,
        "mode": mode,
        "claim": "approximate_optimality_for_the_supplied_constrained_quadratic",
        "not_claimed": ["correct_structure_for_the_world", "neural_reasoning_faithfulness",
                        "cryptographic_execution_attestation", "inference_speedup"],
        "problem": {"measurement": problem.measurement.tolist(),
                    "observed": problem.observed.tolist(), "ridge": problem.ridge,
                    "boundary": constraint.boundary.tolist(),
                    "boundary_shape": list(constraint.boundary.shape),
                    "structure_name": constraint.name},
        "solver": {"step_size": step_size, "tolerance": tolerance,
                   "max_steps": max_steps, "adaptive": adaptive,
                   "iterations": len(events), "stop_reason": stop_reason},
        "events": events,
        "final": {"state": state.tolist(), "multiplier": dual.tolist()},
        "intermediate_evidence": ("full_numeric_witnesses" if mode == "full"
                                  else "not_retained; summaries_are_not_independently_verified"),
    }


def save_record(record: dict, path, *, overwrite: bool = False) -> None:
    """No accidental overwrite of an earlier run; reject NaN/Infinity JSON."""
    target = Path(path)
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(payload)


def load_record(path) -> dict:
    def reject(value):
        raise ValueError(f"Nonfinite JSON value is forbidden: {value}")
    with Path(path).open(encoding="utf-8") as stream:
        out = json.load(stream, parse_constant=reject)
    if not isinstance(out, dict):
        raise ValueError("A trace must be a JSON object")
    return out


def describe(record: dict) -> str:
    """A deterministic rendering of recorded fields, not an invented rationale."""
    solver = record["solver"]
    return (f"Structure: {record['problem']['structure_name']}\n"
            f"Operation: repeated measurement fitting constrained to balanced states.\n"
            f"Iterations: {solver['iterations']}; stop: {solver['stop_reason']}.\n"
            f"Trace: {record['mode']}; final numerical witness retained.\n"
            "This explains explicit operations, not why a learned model would choose them.")
