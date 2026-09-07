"""Run the example, read its trace, or independently check a stored record."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
from .examples import flow_example
from .routing import select_constraint
from .recurrence import solve, direct_solve
from .lingua import describe, save_record, load_record
from .verification import verify_record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Reconstruct a small balanced flow")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--max-steps", type=int, default=256)
    demo.add_argument("--trace", choices=["compact", "full"], default="compact")
    demo.add_argument("--fixed", action="store_true", help="Use the full iteration budget")
    demo.add_argument("--output", type=Path, help="Save JSON without overwriting existing files")
    verify = commands.add_parser("verify", help="Check a retained JSON witness")
    verify.add_argument("record", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            check = verify_record(load_record(args.record))
            print(f"{'PASS' if check.accepted else 'FAIL'}: {check.reason}")
            print(f"Intermediate steps independently checked: {check.intermediate_checks}")
            return 0 if check.accepted else 1
        example = flow_example(args.seed)
        decision = select_constraint(example.candidates, example.problem,
                                     example.validation_measurement, example.validation_observed)
        print("A small circulation: estimate flow without creating or destroying it at junctions.")
        print("Synthetic measurements; no trained neural model; no speedup claim.\n")
        for name, score in decision.scores.items():
            print(f"Candidate {name:28s} validation MSE {score:.6f}")
        if decision.chosen is None:
            print(decision.reason)
            return 2
        chosen = decision.chosen
        result = solve(example.problem, chosen, max_steps=args.max_steps,
                       adaptive=not args.fixed, trace_mode=args.trace)
        reference = direct_solve(example.problem, chosen)
        print("\n" + describe(result.record))
        print(f"Coordinates: {chosen.ambient_dimension} edge values -> {chosen.latent_dimension} cycle coordinates")
        print("Reconstruction:", np.array2string(result.state, precision=5))
        print(f"Distance from direct-solve baseline: {np.linalg.norm(result.state-reference):.3e}")
        print(f"Distance from hidden synthetic truth (evaluation only): {np.linalg.norm(result.state-example.truth_for_evaluation_only):.3e}")
        check = verify_record(result.record)
        print(f"Witness: {'PASS' if check.accepted else 'FAIL'} ({check.reason})")
        if args.output:
            save_record(result.record, args.output)
            print(f"Saved {args.output}")
        return 0 if check.accepted else 1
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
