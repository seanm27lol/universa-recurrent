"""Run, evaluate, benchmark, explain, and verify structured recurrence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .examples import flow_example
from .lingua import describe, load_record, save_record
from .recurrence import direct_solve, solve
from .routing import select_constraint
from .verification import verify_record

_NEURAL_INSTALL = 'Neural extras are missing. Install with: python -m pip install -e ".[neural]"'


def _write_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _neural_train(args: argparse.Namespace) -> int:
    try:
        from .neural.train import train_checkpoint
    except ImportError as error:
        raise RuntimeError(_NEURAL_INSTALL) from error
    train_checkpoint(
        args.output,
        device_name=args.device,
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        steps=args.steps,
        lr=args.lr,
        noise_std=args.noise_std,
        observe_probability=args.observe_probability,
        future_regret_tolerance=args.future_regret_tolerance,
    )
    return 0


def _neural_eval(args: argparse.Namespace) -> int:
    try:
        from .neural.train import evaluate_checkpoint
    except ImportError as error:
        raise RuntimeError(_NEURAL_INSTALL) from error
    result = evaluate_checkpoint(
        args.checkpoint,
        device_name=args.device,
        seed=args.seed,
        n=args.n,
        batch_size=args.batch_size,
    )
    if args.output:
        _write_json(args.output, result)
        print(f"Saved {args.output}")
    return 0


def _neural_benchmark(args: argparse.Namespace) -> int:
    try:
        from .neural.train import benchmark_checkpoint
    except ImportError as error:
        raise RuntimeError(_NEURAL_INSTALL) from error
    result = benchmark_checkpoint(
        args.checkpoint,
        device_name=args.device,
        seed=args.seed,
        n=args.n,
        batch_size=args.batch_size,
        halt_threshold=args.halt_threshold,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    if args.output:
        _write_json(args.output, result)
        print(f"Saved {args.output}")
    return 0


def _neural_demo(args: argparse.Namespace) -> int:
    try:
        import torch
        from .neural.data import StructuredFlowDataset
        from .neural.lingua import describe_neural, neural_record
        from .neural.train import load_checkpoint
        from .neural.verification import verify_neural_record
    except ImportError as error:
        raise RuntimeError(_NEURAL_INSTALL) from error

    model, metadata, device = load_checkpoint(args.checkpoint, device_name=args.device)
    raw_metadata = metadata.get("raw_metadata", {})
    noise_std = 0.05
    observe_probability = 0.7
    if isinstance(raw_metadata, dict):
        training = raw_metadata.get("training")
        if isinstance(training, dict):
            noise_std = float(training.get("noise_std", noise_std))
            observe_probability = float(training.get("observe_probability", observe_probability))
    dataset = StructuredFlowDataset(
        1,
        seed=args.seed,
        noise_std=noise_std,
        observe_probability=observe_probability,
    )
    sample = dataset[0]
    observed = sample["observed"][None].to(device)
    mask = sample["mask"][None].to(device)
    truth = sample["truth"][None].to(device)
    label = int(sample["label"].item())
    output = model.infer(
        observed,
        mask,
        max_steps=args.max_steps,
        halt_threshold=args.halt_threshold,
        min_steps=args.min_steps,
        execution_mode=args.execution_mode,
        fixed_depth=args.fixed,
        record_events=True,
        validate_values=True,
    )
    names = metadata["names"]
    boundaries = metadata["boundaries"]
    assert isinstance(names, list)
    record = neural_record(
        model,
        output,
        observed,
        mask,
        names,
        boundaries,
        checkpoint_sha256=str(metadata["checkpoint_sha256"]),
        checkpoint_format=str(metadata["format"]),
    )
    predicted = int(output["routes"][0].item())  # type: ignore[index,union-attr]
    state = output["state"]
    assert torch.is_tensor(state)
    mse = float((state - truth).square().mean().item())
    check = verify_neural_record(record, checkpoint=args.checkpoint)

    print(describe_neural(record))
    print(f"True synthetic structure (evaluation only): {names[label]}")
    print(f"Route correct: {predicted == label}")
    print(f"Reconstruction MSE against hidden synthetic truth: {mse:.6f}")
    print(f"Independent record check: {'PASS' if check.accepted else 'FAIL'} ({check.reason})")
    print(
        "Important: Lingua checks explicit structures and dynamics; it does not "
        "decode hidden-feature semantics or replay the neural update."
    )
    if args.output:
        _write_json(args.output, record)
        print(f"Saved {args.output}")
    return 0 if check.accepted else 1


def _neural_verify(args: argparse.Namespace) -> int:
    try:
        from .neural.verification import verify_neural_record
    except ImportError as error:
        raise RuntimeError(_NEURAL_INSTALL) from error
    record = json.loads(args.record.read_text(encoding="utf-8"))
    check = verify_neural_record(record, checkpoint=args.checkpoint)
    print(f"{'PASS' if check.accepted else 'FAIL'}: {check.reason}")
    for name, passed in sorted(check.checks.items()):
        print(f"  {'PASS' if passed else 'FAIL'} {name}")
    return 0 if check.accepted else 1


def _classical_demo(args: argparse.Namespace) -> int:
    example = flow_example(args.seed)
    decision = select_constraint(
        example.candidates,
        example.problem,
        example.validation_measurement,
        example.validation_observed,
    )
    print("A small circulation: estimate flow without creating or destroying it at junctions.")
    print("Synthetic measurements; classical baseline; no speedup claim.\n")
    for name, score in decision.scores.items():
        print(f"Candidate {name:28s} validation MSE {score:.6f}")
    if decision.chosen is None:
        print(decision.reason)
        return 2
    chosen = decision.chosen
    result = solve(
        example.problem,
        chosen,
        max_steps=args.max_steps,
        adaptive=not args.fixed,
        trace_mode=args.trace,
    )
    reference = direct_solve(example.problem, chosen)
    print("\n" + describe(result.record))
    print(
        f"Coordinates: {chosen.ambient_dimension} edge values -> "
        f"{chosen.latent_dimension} cycle coordinates"
    )
    print("Reconstruction:", np.array2string(result.state, precision=5))
    print(f"Distance from direct-solve baseline: {np.linalg.norm(result.state-reference):.3e}")
    print(
        "Distance from hidden synthetic truth (evaluation only): "
        f"{np.linalg.norm(result.state-example.truth_for_evaluation_only):.3e}"
    )
    check = verify_record(result.record)
    print(f"Witness: {'PASS' if check.accepted else 'FAIL'} ({check.reason})")
    if args.output:
        save_record(result.record, args.output)
        print(f"Saved {args.output}")
    return 0 if check.accepted else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="Run the transparent classical flow reconstruction")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--max-steps", type=int, default=256)
    demo.add_argument("--trace", choices=["compact", "full"], default="compact")
    demo.add_argument("--fixed", action="store_true", help="Use the full iteration budget")
    demo.add_argument("--output", type=Path, help="Save JSON without overwriting existing files")

    verify = commands.add_parser("verify", help="Check a retained classical JSON witness")
    verify.add_argument("record", type=Path)

    neural_train = commands.add_parser("neural-train", help="Train exploratory learned structured recurrence")
    neural_train.add_argument("--output", type=Path, default=Path("checkpoints/neural_v1.pt"))
    neural_train.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    neural_train.add_argument("--seed", type=int, default=4242)
    neural_train.add_argument("--train-size", type=int, default=20_000)
    neural_train.add_argument("--val-size", type=int, default=4_000)
    neural_train.add_argument("--epochs", type=int, default=20)
    neural_train.add_argument("--batch-size", type=int, default=512)
    neural_train.add_argument("--hidden-dim", type=int, default=64)
    neural_train.add_argument("--steps", type=int, default=8)
    neural_train.add_argument("--lr", type=float, default=2e-3)
    neural_train.add_argument("--noise-std", type=float, default=0.05)
    neural_train.add_argument("--observe-probability", type=float, default=0.7)
    neural_train.add_argument("--future-regret-tolerance", type=float, default=1e-4)

    neural_eval = commands.add_parser("neural-eval", help="Compare adaptive, fixed-depth, and reference methods")
    neural_eval.add_argument("--checkpoint", type=Path, default=Path("checkpoints/neural_v1.pt"))
    neural_eval.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    neural_eval.add_argument("--seed", type=int, default=9000)
    neural_eval.add_argument("--n", type=int, default=4_000)
    neural_eval.add_argument("--batch-size", type=int, default=1024)
    neural_eval.add_argument("--output", type=Path)

    neural_benchmark = commands.add_parser("neural-benchmark", help="Measure dense, compact, fixed, and transparent baselines")
    neural_benchmark.add_argument("--checkpoint", type=Path, default=Path("checkpoints/neural_v1.pt"))
    neural_benchmark.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    neural_benchmark.add_argument("--seed", type=int, default=9100)
    neural_benchmark.add_argument("--n", type=int, default=65_536)
    neural_benchmark.add_argument("--batch-size", type=int, default=4096)
    neural_benchmark.add_argument("--halt-threshold", type=float, default=0.50)
    neural_benchmark.add_argument("--warmup", type=int, default=5)
    neural_benchmark.add_argument("--repeats", type=int, default=20)
    neural_benchmark.add_argument("--output", type=Path)

    neural_demo = commands.add_parser("neural-demo", help="Emit one conservative neural Lingua record")
    neural_demo.add_argument("--checkpoint", type=Path, default=Path("checkpoints/neural_v1.pt"))
    neural_demo.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    neural_demo.add_argument("--seed", type=int, default=9001)
    neural_demo.add_argument("--max-steps", type=int, default=8)
    neural_demo.add_argument("--min-steps", type=int, default=1)
    neural_demo.add_argument("--halt-threshold", type=float, default=0.80)
    neural_demo.add_argument("--execution-mode", choices=["dense", "compact"], default="compact")
    neural_demo.add_argument("--fixed", action="store_true")
    neural_demo.add_argument("--output", type=Path)

    neural_verify = commands.add_parser("neural-verify", help="Independently check a retained neural Lingua record")
    neural_verify.add_argument("record", type=Path)
    neural_verify.add_argument("--checkpoint", type=Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "neural-train":
            return _neural_train(args)
        if args.command == "neural-eval":
            return _neural_eval(args)
        if args.command == "neural-benchmark":
            return _neural_benchmark(args)
        if args.command == "neural-demo":
            return _neural_demo(args)
        if args.command == "neural-verify":
            return _neural_verify(args)
        if args.command == "verify":
            check = verify_record(load_record(args.record))
            print(f"{'PASS' if check.accepted else 'FAIL'}: {check.reason}")
            print(f"Intermediate steps independently checked: {check.intermediate_checks}")
            return 0 if check.accepted else 1
        return _classical_demo(args)
    except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
