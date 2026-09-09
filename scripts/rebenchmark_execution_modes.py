#!/usr/bin/env python3
"""Re-benchmark existing dual-output checkpoints under explicit execution settings.

The parent process launches a fresh child interpreter for every checkpoint,
batch-size, and deterministic-algorithms setting. This prevents PyTorch's global
execution flags from leaking from training into timing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def _positive_int(value: str | int) -> int:
    integer = int(value)
    if integer < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return integer


def _nonnegative_int(value: str | int) -> int:
    integer = int(value)
    if integer < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return integer


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def child_environment(deterministic: bool) -> dict[str, str]:
    """Return a fresh child environment with deterministic CUDA config explicit."""
    env = dict(os.environ)
    if deterministic:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    return env


def _quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def linear(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        location = q * (len(ordered) - 1)
        low = int(math.floor(location))
        high = int(math.ceil(location))
        if low == high:
            return ordered[low]
        fraction = location - low
        return ordered[low] * (1 - fraction) + ordered[high] * fraction

    return {
        "median_ms": statistics.median(ordered),
        "p10_ms": linear(0.10),
        "p90_ms": linear(0.90),
    }


def summarize(records: list[dict]) -> dict:
    """Aggregate per-checkpoint medians without pooling timing repeats as fits."""
    groups: dict[tuple[str, int, bool], list[dict]] = defaultdict(list)
    for record in records:
        groups[(str(record["model"]), int(record["batch_size"]),
                bool(record["deterministic_algorithms"]))].append(record)

    rows = []
    for (model, batch_size, deterministic), items in sorted(groups.items()):
        medians = [float(item["timing"]["median_ms"]) for item in items]
        mses = [float(item["estimate_mse"]) for item in items]
        rows.append({
            "model": model,
            "batch_size": batch_size,
            "deterministic_algorithms": deterministic,
            "training_runs": len(items),
            "median_of_run_medians_ms": statistics.median(medians),
            "mean_of_run_medians_ms": statistics.mean(medians),
            "sample_sd_of_run_medians_ms": statistics.stdev(medians) if len(medians) > 1 else None,
            "mean_estimate_mse": statistics.mean(mses),
            "estimate_mse_sample_sd": statistics.stdev(mses) if len(mses) > 1 else None,
            "per_training_seed": [{
                "training_seed": item["training_seed"],
                "median_ms": item["timing"]["median_ms"],
                "estimate_mse": item["estimate_mse"],
                "estimate_sha256": item["estimate_sha256"],
            } for item in sorted(items, key=lambda x: int(x["training_seed"]))],
        })

    lookup = {(row["model"], row["batch_size"], row["deterministic_algorithms"]): row
              for row in rows}
    ratios = []
    for model, batch_size in sorted({(row["model"], row["batch_size"]) for row in rows}):
        off = lookup.get((model, batch_size, False))
        on = lookup.get((model, batch_size, True))
        if off is None or on is None:
            continue
        ratios.append({
            "model": model,
            "batch_size": batch_size,
            "deterministic_on_divided_by_off_median_time": (
                on["median_of_run_medians_ms"] / off["median_of_run_medians_ms"]),
            "mean_mse_difference_on_minus_off": on["mean_estimate_mse"] - off["mean_estimate_mse"],
        })
    return {
        "format": "universa-recurrent.execution-settings-study.v1.summary",
        "rows": rows,
        "deterministic_ratios": ratios,
        "note": ("Timing rows summarize independent trained checkpoints. Raw timing repetitions "
                 "are retained per child result and are not treated as independent training replications."),
    }


def _sync(device) -> None:
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def run_child(args: argparse.Namespace) -> dict:
    # Import torch only after the parent has set the child environment.
    import torch
    from universa_recurrent.neural.data import StructuredFlowDataset
    from universa_recurrent.neural.dual_output import ClaimPolicy
    from universa_recurrent.neural.dual_study import collect, load_estimators

    torch.use_deterministic_algorithms(bool(args.deterministic))
    # Pin this rather than inherit a process-global value from an earlier job.
    torch.backends.cudnn.benchmark = False

    engines, metadata, device, reserved = load_estimators(args.checkpoint, args.device, None)
    settings = metadata.get("training")
    if not isinstance(settings, dict):
        raise ValueError("checkpoint is missing training metadata")
    training_seed = settings.get("seed")
    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise ValueError("checkpoint is missing an integer training seed")
    if args.test_seed in reserved:
        raise ValueError("benchmark test seed overlaps checkpoint training/calibration")

    dataset = StructuredFlowDataset(args.n, seed=args.test_seed,
        noise_std=float(settings["noise_std"]),
        observe_probability=float(settings["observe_probability"]))
    observed = dataset.observed.to(device)
    mask = dataset.mask.to(device)
    truth = dataset.truth.to(device)

    wanted = {
        "shared": "mixture",
        "fixed_depth_4": "mixture",
        "untied": "mixture",
        "ambient": "estimate_only",
        "direct": "mixture",
    }
    available = {engine.name: engine for engine in engines}
    missing = sorted(set(wanted) - set(available))
    if missing:
        raise ValueError(f"checkpoint is missing required controls: {missing}")

    records = []
    for model_name, mode in wanted.items():
        engine = available[model_name]
        policy = None if model_name == "ambient" else ClaimPolicy(None)
        baseline = collect(engine, observed, mask, policy, mode, args.batch_size)
        estimate = baseline["estimate"]
        estimate_cpu = estimate.detach().cpu().contiguous()
        estimate_hash = hashlib.sha256(estimate_cpu.numpy().tobytes()).hexdigest()
        mse = float((estimate.double() - truth.double()).square().mean().item())
        del baseline, estimate

        for _ in range(args.warmup):
            warm = collect(engine, observed, mask, policy, mode, args.batch_size)
            del warm
        _sync(device)

        samples = []
        for _ in range(args.repeats):
            _sync(device)
            started = time.perf_counter_ns()
            timed = collect(engine, observed, mask, policy, mode, args.batch_size)
            _sync(device)
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
            timed_estimate = timed["estimate"].detach().cpu()
            if not torch.allclose(timed_estimate, estimate_cpu, atol=1e-7, rtol=1e-6):
                raise RuntimeError(f"{model_name}: timed output changed within one child process")
            del timed, timed_estimate

        records.append({
            "model": model_name,
            "output_mode": mode,
            "training_seed": training_seed,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": str(metadata["checkpoint_sha256"]),
            "test_seed": args.test_seed,
            "n": args.n,
            "batch_size": args.batch_size,
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda,
            "estimate_mse": mse,
            "estimate_sha256": estimate_hash,
            "timing": {
                **_quantiles(samples),
                "samples_ms": samples,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "scope": ("Model inference, fixed-depth rollout construction, output construction and Python "
                          "batching. Excludes checkpoint loading, host-to-device transfer, metrics, Lingua, and calibration."),
            },
        })

    return {
        "format": "universa-recurrent.execution-settings-study.v1.child",
        "checkpoint_sha256": str(metadata["checkpoint_sha256"]),
        "training_seed": training_seed,
        "test_seed": args.test_seed,
        "batch_size": args.batch_size,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "rows": records,
    }


def run_parent(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite {args.output_dir}")
    checkpoints = sorted(args.replication_dir.glob("weights-*.pt"))
    if not checkpoints:
        raise ValueError(f"no weights-*.pt checkpoints found in {args.replication_dir}")
    args.output_dir.mkdir(parents=True)

    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir()
    all_rows = []
    child_results = []

    for checkpoint in checkpoints:
        for batch_size in args.batch_sizes:
            for deterministic in (False, True):
                output = raw_dir / (f"{checkpoint.stem}-b{batch_size}-"
                                    f"{'det' if deterministic else 'normal'}.json")
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--child",
                    "--checkpoint", str(checkpoint.resolve()),
                    "--device", args.device,
                    "--test-seed", str(args.test_seed),
                    "--n", str(args.n),
                    "--batch-size", str(batch_size),
                    "--warmup", str(args.warmup),
                    "--repeats", str(args.repeats),
                    "--output", str(output),
                ]
                if deterministic:
                    command.append("--deterministic")
                print(f"[{checkpoint.name}] batch={batch_size} deterministic={deterministic}", flush=True)
                subprocess.run(command, check=True, env=child_environment(deterministic))
                child = json.loads(output.read_text(encoding="utf-8"))
                child_results.append(str(output.relative_to(args.output_dir)))
                all_rows.extend(child["rows"])

    result = {
        "format": "universa-recurrent.execution-settings-study.v1.complete",
        "replication_dir": str(args.replication_dir.resolve()),
        "checkpoints": [str(path.resolve()) for path in checkpoints],
        "test_seed": args.test_seed,
        "n": args.n,
        "batch_sizes": args.batch_sizes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "fresh_process_per_checkpoint_batch_and_setting": True,
        "child_results": child_results,
        "summary": summarize(all_rows),
        "limits": [
            "This is a timing-control study, not a new training replication.",
            "Deterministic mode includes CUBLAS_WORKSPACE_CONFIG=:4096:8.",
            "The same held-out cohort is intentionally reused across settings.",
            "Results are hardware- and software-stack-specific.",
        ],
    }
    _write_json(args.output_dir / "summary.json", result)

    print("\nmodel                    batch  det  median-of-runs ms  mean MSE")
    for row in result["summary"]["rows"]:
        print(f"{row['model']:<24} {row['batch_size']:>5}  "
              f"{str(row['deterministic_algorithms']):>5}  "
              f"{row['median_of_run_medians_ms']:>17.4f}  "
              f"{row['mean_estimate_mse']:.6f}")
    print(f"\nSaved: {args.output_dir / 'summary.json'}")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--replication-dir", type=Path)
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    result.add_argument("--test-seed", type=_nonnegative_int, default=32000)
    result.add_argument("--n", type=_positive_int, default=4000)
    result.add_argument("--batch-sizes", nargs="+", type=_positive_int, default=[1024, 4096])
    result.add_argument("--batch-size", type=_positive_int, help=argparse.SUPPRESS)
    result.add_argument("--warmup", type=_nonnegative_int, default=5)
    result.add_argument("--repeats", type=_positive_int, default=20)
    result.add_argument("--deterministic", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.child:
            if args.checkpoint is None or args.batch_size is None or args.output is None:
                raise ValueError("child mode requires checkpoint, batch-size, and output")
            payload = run_child(args)
            _write_json(args.output, payload)
            return 0
        if args.replication_dir is None or args.output_dir is None:
            raise ValueError("parent mode requires --replication-dir and --output-dir")
        run_parent(args)
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
