"""Training, evaluation, checkpoint loading, and timing for neural recurrence."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import os
import platform
import re
import time
from typing import Callable

import torch
from torch.utils.data import DataLoader

from .baselines import fit_each_structure, gaussian_generator_reference
from .data import StructuredFlowDataset, candidate_bases
from .model import NeuralConfig, StructuredRecurrentNet, training_loss

_CHECKPOINT_FORMAT = "universa-recurrent.neural-checkpoint.v1"
_MIN_TORCH = (2, 10)


def _torch_version_tuple() -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None:
        raise RuntimeError(f"cannot parse PyTorch version {torch.__version__!r}")
    return int(match.group(1)), int(match.group(2))


def _require_supported_torch() -> None:
    if _torch_version_tuple() < _MIN_TORCH:
        raise RuntimeError(
            f"PyTorch >= {_MIN_TORCH[0]}.{_MIN_TORCH[1]} is required; found {torch.__version__}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pick_device(requested: str = "auto") -> torch.device:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def build_model(
    device: torch.device,
    hidden_dim: int = 64,
    steps: int = 8,
    *,
    residual_normalization: str = "observed",
) -> tuple[StructuredRecurrentNet, list[str], torch.Tensor, torch.Tensor]:
    names, bases_np, boundaries_np = candidate_bases()
    bases = torch.from_numpy(bases_np).to(device)
    config = NeuralConfig(
        ambient_dim=bases.shape[1],
        latent_dim=bases.shape[2],
        num_structures=bases.shape[0],
        hidden_dim=hidden_dim,
        steps=steps,
        residual_normalization=residual_normalization,  # type: ignore[arg-type]
    )
    model = StructuredRecurrentNet(bases, config).to(device)
    return model, names, torch.from_numpy(bases_np), torch.from_numpy(boundaries_np)


def _all_floating_tensors_finite(tree: object, *, path: str = "root") -> None:
    if torch.is_tensor(tree):
        if tree.is_floating_point() and not torch.isfinite(tree).all():
            raise ValueError(f"non-finite tensor in checkpoint at {path}")
        return
    if isinstance(tree, dict):
        for key, value in tree.items():
            _all_floating_tensors_finite(value, path=f"{path}.{key}")
    elif isinstance(tree, (list, tuple)):
        for index, value in enumerate(tree):
            _all_floating_tensors_finite(value, path=f"{path}[{index}]")


def _environment(device: torch.device) -> dict[str, object]:
    result: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type == "cuda":
        result["cuda_runtime"] = torch.version.cuda
        result["device_name"] = torch.cuda.get_device_name(device)
    return result


def _write_json_no_overwrite(path: Path, payload: dict) -> None:
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_checkpoint(
    output: Path,
    *,
    device_name: str = "auto",
    seed: int = 4242,
    train_size: int = 20_000,
    val_size: int = 4_000,
    epochs: int = 20,
    batch_size: int = 512,
    hidden_dim: int = 64,
    steps: int = 8,
    lr: float = 2e-3,
    noise_std: float = 0.05,
    observe_probability: float = 0.7,
    future_regret_tolerance: float = 1e-4,
    deterministic: bool = True,
) -> dict[str, object]:
    """Train one exploratory checkpoint and refuse to overwrite artifacts."""
    _require_supported_torch()
    sidecar = output.with_name(output.name + ".sha256")
    if output.exists() or sidecar.exists():
        existing = output if output.exists() else sidecar
        raise ValueError(f"Refusing to overwrite {existing}")
    for name, value in (
        ("train_size", train_size),
        ("val_size", val_size),
        ("epochs", epochs),
        ("batch_size", batch_size),
        ("hidden_dim", hidden_dim),
        ("steps", steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (
        ("lr", lr),
        ("noise_std", noise_std),
        ("observe_probability", observe_probability),
        ("future_regret_tolerance", future_regret_tolerance),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if lr <= 0 or noise_std < 0 or not (0 < observe_probability <= 1) or future_regret_tolerance < 0:
        raise ValueError("invalid training hyperparameter range")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)

    device = pick_device(device_name)
    model, names, bases_cpu, boundaries_cpu = build_model(device, hidden_dim, steps)
    training_data = StructuredFlowDataset(
        train_size,
        seed=seed + 1,
        noise_std=noise_std,
        observe_probability=observe_probability,
    )
    validation_data = StructuredFlowDataset(
        val_size,
        seed=seed + 2,
        noise_std=noise_std,
        observe_probability=observe_probability,
    )
    generator = torch.Generator().manual_seed(seed + 3)
    loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    metric_names = (
        "loss",
        "true_route_final_mse",
        "mixed_final_mse",
        "router_ce",
        "halt_bce",
    )
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        sums = torch.zeros(len(metric_names), dtype=torch.float64, device=device)
        count = 0
        for batch in loader:
            observed = batch["observed"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            truth = batch["truth"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            outputs = model.forward_train(observed, mask)
            loss, metrics = training_loss(
                outputs,
                truth,
                labels,
                future_regret_tolerance=future_regret_tolerance,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            values = torch.stack([metrics[name].to(torch.float64) for name in metric_names])
            sums += values * observed.shape[0]
            count += observed.shape[0]
        epoch_values = (sums / count).detach().cpu().tolist()
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({name: float(value) for name, value in zip(metric_names, epoch_values)})
        history.append(row)
        print(
            f"epoch {epoch:02d} loss={row['loss']:.5f} "
            f"true_route_mse={row['true_route_final_mse']:.5f} "
            f"router_ce={row['router_ce']:.4f}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    validation = evaluate_model(
        model,
        validation_data,
        device=device,
        batch_size=min(1024, max(1, val_size)),
        max_steps=steps,
        halt_threshold=0.8,
        execution_mode="compact",
    )

    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    payload: dict[str, object] = {
        "format": _CHECKPOINT_FORMAT,
        "model_state": model_state,
        "config": asdict(model.config),
        "library": {
            "names": names,
            "bases": bases_cpu,
            "boundaries": boundaries_cpu,
        },
        "training": {
            "seed": seed,
            "train_seed": seed + 1,
            "validation_seed": seed + 2,
            "train_size": train_size,
            "validation_size": val_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "noise_std": noise_std,
            "observe_probability": observe_probability,
            "future_regret_tolerance": future_regret_tolerance,
            "halt_teacher": "future_regret",
            "deterministic_algorithms": deterministic,
        },
        "history": history,
        "validation": validation,
        "training_seconds": elapsed,
        "environment": _environment(device),
    }
    _all_floating_tensors_finite(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    torch.save(payload, temporary)
    os.replace(temporary, output)
    digest = sha256_file(output)
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")

    result: dict[str, object] = {
        "saved": str(output),
        "sha256": digest,
        "device": str(device),
        "seconds": elapsed,
        **validation,
    }
    print(json.dumps(result, indent=2))
    return result


def _normalize_loaded_payload(raw: dict, path: Path) -> dict[str, object]:
    """Normalize v0.2 and v1 checkpoint layouts into one validated view."""
    if not isinstance(raw, dict):
        raise ValueError("checkpoint root must be a dictionary")
    names_expected, bases_np, boundaries_np = candidate_bases()
    bases_expected = torch.from_numpy(bases_np)
    boundaries_expected = torch.from_numpy(boundaries_np)

    if raw.get("format") == _CHECKPOINT_FORMAT:
        state = raw.get("model_state")
        library = raw.get("library")
        if not isinstance(library, dict):
            raise ValueError("checkpoint library metadata missing")
        names = library.get("names")
        bases = library.get("bases")
        boundaries = library.get("boundaries")
        config_data = raw.get("config")
        format_name = _CHECKPOINT_FORMAT
    elif "model" in raw and "config" in raw:
        # v0.2 compatibility. It normalized residuals by ambient width.
        state = raw.get("model")
        names = raw.get("names")
        bases = bases_expected
        boundaries = raw.get("boundaries")
        config_data = dict(raw.get("config", {}))
        config_data.setdefault("residual_normalization", "ambient")
        config_data.setdefault("halt_timing", "before_update")
        format_name = "universa-recurrent.neural-checkpoint.v0.2-legacy"
    else:
        raise ValueError("unrecognized checkpoint format")

    if not isinstance(state, dict) or not isinstance(config_data, dict):
        raise ValueError("checkpoint model/config metadata missing")
    if names != names_expected:
        raise ValueError("checkpoint candidate names do not match this task")
    if not torch.is_tensor(bases):
        bases = torch.as_tensor(bases)
    if not torch.is_tensor(boundaries):
        boundaries = torch.as_tensor(boundaries)
    bases = bases.detach().cpu().float()
    boundaries = boundaries.detach().cpu().float()
    if bases.shape != bases_expected.shape or not torch.allclose(bases, bases_expected, atol=0, rtol=0):
        raise ValueError("checkpoint candidate bases do not match this release")
    if boundaries.shape != boundaries_expected.shape or not torch.allclose(
        boundaries, boundaries_expected, atol=0, rtol=0
    ):
        raise ValueError("checkpoint candidate boundaries do not match this release")

    config = NeuralConfig(**config_data)
    _all_floating_tensors_finite(state, path="model_state")
    state_bases = state.get("bases")
    if not torch.is_tensor(state_bases) or state_bases.shape != bases_expected.shape:
        raise ValueError("checkpoint model state is missing the candidate bases")
    if not torch.equal(state_bases.detach().cpu().float(), bases_expected):
        raise ValueError("checkpoint model-state bases do not match this release")
    return {
        "format": format_name,
        "model_state": state,
        "config": asdict(config),
        "names": list(names_expected),
        "bases": bases,
        "boundaries": boundaries,
        "raw_metadata": {
            key: value
            for key, value in raw.items()
            if key not in ("model", "model_state", "library")
        },
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
    }


def load_checkpoint(
    path: Path,
    *,
    device_name: str = "auto",
) -> tuple[StructuredRecurrentNet, dict[str, object], torch.device]:
    """Load a trusted checkpoint through PyTorch's restricted weights-only path."""
    _require_supported_torch()
    if not path.is_file():
        raise FileNotFoundError(path)
    device = pick_device(device_name)
    # Some v0.2 files stored ``torch.__version__`` as TorchVersion rather
    # than a plain string. Allowlist only that narrow legacy metadata type;
    # model code/classes remain disallowed by the weights-only loader.
    from torch.torch_version import TorchVersion
    with torch.serialization.safe_globals([TorchVersion]):
        raw = torch.load(path, map_location="cpu", weights_only=True)
    metadata = _normalize_loaded_payload(raw, path)
    config = NeuralConfig(**metadata["config"])  # type: ignore[arg-type]
    bases = metadata["bases"]
    assert torch.is_tensor(bases)
    model = StructuredRecurrentNet(bases.to(device), config).to(device)
    state = metadata["model_state"]
    assert isinstance(state, dict)
    model.load_state_dict(state, strict=True)
    _all_floating_tensors_finite(model.state_dict(), path="loaded_model")
    model.eval()
    return model, metadata, device


def _empty_accumulator() -> dict[str, float | int]:
    return {
        "examples": 0,
        "correct_routes": 0,
        "squared_error": 0.0,
        "correct_squared_error": 0.0,
        "wrong_squared_error": 0.0,
        "correct_elements": 0,
        "wrong_elements": 0,
        "logical_updates": 0,
        "update_examples": 0,
        "recurrent_rounds": 0,
        "batches": 0,
    }


def _accumulate_prediction(
    accumulator: dict[str, float | int],
    *,
    state: torch.Tensor,
    routes: torch.Tensor,
    truth: torch.Tensor,
    labels: torch.Tensor,
    steps_taken: torch.Tensor | None = None,
    update_examples: int = 0,
    recurrent_rounds: int = 0,
) -> None:
    errors = (state - truth).square()
    correct = routes == labels
    wrong = ~correct
    count = int(state.shape[0])
    ambient = int(state.shape[1])
    accumulator["examples"] += count
    accumulator["correct_routes"] += int(correct.sum().item())
    accumulator["squared_error"] += float(errors.sum().item())
    if correct.any():
        accumulator["correct_squared_error"] += float(errors[correct].sum().item())
        accumulator["correct_elements"] += int(correct.sum().item()) * ambient
    if wrong.any():
        accumulator["wrong_squared_error"] += float(errors[wrong].sum().item())
        accumulator["wrong_elements"] += int(wrong.sum().item()) * ambient
    if steps_taken is not None:
        accumulator["logical_updates"] += int(steps_taken.sum().item())
    accumulator["update_examples"] += int(update_examples)
    accumulator["recurrent_rounds"] += int(recurrent_rounds)
    accumulator["batches"] += 1


def _finalize_metrics(
    accumulator: dict[str, float | int],
    *,
    ambient_dim: int,
) -> dict[str, object]:
    examples = int(accumulator["examples"])
    if examples < 1:
        raise ValueError("cannot finalize empty metrics")
    correct_elements = int(accumulator["correct_elements"])
    wrong_elements = int(accumulator["wrong_elements"])
    return {
        "n": examples,
        "router_accuracy": int(accumulator["correct_routes"]) / examples,
        "reconstruction_mse": float(accumulator["squared_error"]) / (examples * ambient_dim),
        "route_correct_mse": (
            float(accumulator["correct_squared_error"]) / correct_elements
            if correct_elements
            else None
        ),
        "route_wrong_mse": (
            float(accumulator["wrong_squared_error"]) / wrong_elements
            if wrong_elements
            else None
        ),
        "route_correct_n": correct_elements // ambient_dim,
        "route_wrong_n": wrong_elements // ambient_dim,
        "mean_logical_steps": int(accumulator["logical_updates"]) / examples,
        "update_examples_per_sample": int(accumulator["update_examples"]) / examples,
        "mean_recurrent_rounds_per_batch": int(accumulator["recurrent_rounds"]) / max(1, int(accumulator["batches"])),
    }


@torch.no_grad()
def evaluate_model(
    model: StructuredRecurrentNet,
    dataset: StructuredFlowDataset,
    *,
    device: torch.device,
    batch_size: int = 1024,
    max_steps: int | None = None,
    halt_threshold: float = 0.80,
    execution_mode: str = "compact",
    fixed_depth: bool = False,
    force_true_routes: bool = False,
) -> dict[str, object]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    accumulator = _empty_accumulator()
    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        truth = batch["truth"].to(device)
        labels = batch["label"].to(device)
        output = model.infer(
            observed,
            mask,
            max_steps=max_steps,
            halt_threshold=halt_threshold,
            execution_mode=execution_mode,  # type: ignore[arg-type]
            fixed_depth=fixed_depth,
            forced_routes=labels if force_true_routes else None,
        )
        state = output["state"]
        routes = output["routes"]
        steps_taken = output["steps_taken"]
        assert torch.is_tensor(state) and torch.is_tensor(routes) and torch.is_tensor(steps_taken)
        _accumulate_prediction(
            accumulator,
            state=state,
            routes=routes,
            truth=truth,
            labels=labels,
            steps_taken=steps_taken,
            update_examples=int(output["update_examples"]),
            recurrent_rounds=int(output["recurrent_rounds"]),
        )
    metrics = _finalize_metrics(accumulator, ambient_dim=dataset.spec.ambient_dim)
    metrics.update(
        {
            "halt_threshold": float(halt_threshold),
            "max_steps": int(max_steps or model.config.steps),
            "execution_mode": execution_mode,
            "fixed_depth": bool(fixed_depth),
            "forced_true_routes": bool(force_true_routes),
        }
    )
    return metrics


@torch.no_grad()
def _evaluate_reference(
    dataset: StructuredFlowDataset,
    *,
    model: StructuredRecurrentNet,
    device: torch.device,
    batch_size: int,
    kind: str,
) -> dict[str, object]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    accumulator = _empty_accumulator()
    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        truth = batch["truth"].to(device)
        labels = batch["label"].to(device)
        if kind == "fit_each_structure":
            output = fit_each_structure(observed, mask, model.bases, validate_values=False)
            state = output["state"]
            routes = output["routes"]
        elif kind in ("gaussian_hard", "gaussian_mixture"):
            output = gaussian_generator_reference(
                observed,
                mask,
                model.bases,
                noise_std=dataset.spec.noise_std,
                validate_values=False,
            )
            routes = output["routes"]
            state = output["hard_state"] if kind == "gaussian_hard" else output["mixture_state"]
        else:
            raise ValueError(f"unknown reference {kind}")
        _accumulate_prediction(
            accumulator,
            state=state,
            routes=routes,
            truth=truth,
            labels=labels,
        )
    result = _finalize_metrics(accumulator, ambient_dim=dataset.spec.ambient_dim)
    result.update({"method": kind, "privileged": kind.startswith("gaussian")})
    return result


@torch.no_grad()
def evaluate_checkpoint(
    path: Path,
    *,
    device_name: str = "auto",
    seed: int = 9000,
    n: int = 4000,
    batch_size: int = 1024,
) -> dict[str, object]:
    """Compare adaptive recurrence, fixed depths, and explicit references."""
    model, metadata, device = load_checkpoint(path, device_name=device_name)
    raw_metadata = metadata.get("raw_metadata", {})
    noise_std = 0.05
    observe_probability = 0.7
    if isinstance(raw_metadata, dict):
        training = raw_metadata.get("training")
        if isinstance(training, dict):
            noise_std = float(training.get("noise_std", noise_std))
            observe_probability = float(training.get("observe_probability", observe_probability))
    dataset = StructuredFlowDataset(
        n,
        seed=seed,
        noise_std=noise_std,
        observe_probability=observe_probability,
    )

    rows: list[dict[str, object]] = []
    for threshold in (0.50, 0.65, 0.80, 0.90):
        row = evaluate_model(
            model,
            dataset,
            device=device,
            batch_size=batch_size,
            max_steps=model.config.steps,
            halt_threshold=threshold,
            execution_mode="compact",
        )
        row.update({"mode": "adaptive", "label": f"adaptive_{threshold:.2f}"})
        rows.append(row)
    for depth in (1, 2, 4, model.config.steps):
        if depth > model.config.steps:
            continue
        row = evaluate_model(
            model,
            dataset,
            device=device,
            batch_size=batch_size,
            max_steps=depth,
            halt_threshold=1.0,
            execution_mode="dense",
            fixed_depth=True,
        )
        row.update({"mode": "fixed_depth", "label": f"fixed_{depth}"})
        rows.append(row)

    true_route = evaluate_model(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        max_steps=model.config.steps,
        halt_threshold=1.0,
        execution_mode="dense",
        fixed_depth=True,
        force_true_routes=True,
    )
    references = [
        _evaluate_reference(
            dataset,
            model=model,
            device=device,
            batch_size=batch_size,
            kind=kind,
        )
        for kind in ("fit_each_structure", "gaussian_hard", "gaussian_mixture")
    ]
    result: dict[str, object] = {
        "checkpoint": str(path),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "checkpoint_format": metadata["format"],
        "device": str(device),
        "seed": seed,
        "dataset": {
            "n": n,
            "noise_std": dataset.spec.noise_std,
            "observe_probability": dataset.spec.observe_probability,
        },
        "rows": rows,
        "true_route_fixed_depth": true_route,
        "references": references,
        "notes": [
            "Adaptive rows use active-sample compaction; logical steps are not wall-clock speed.",
            "The Gaussian references know the exact toy generator and are privileged.",
            "The soft Gaussian mixture is the squared-error reference; hard route accuracy is diagnostic.",
        ],
    }
    print(json.dumps(result, indent=2))
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(
    function: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, object]:
    result: object = None
    for _ in range(warmup):
        result = function()
    _synchronize(device)
    started = time.perf_counter()
    for _ in range(repeats):
        result = function()
    _synchronize(device)
    milliseconds = (time.perf_counter() - started) * 1000.0 / repeats
    return milliseconds, result


@torch.no_grad()
def benchmark_checkpoint(
    path: Path,
    *,
    device_name: str = "auto",
    seed: int = 9100,
    n: int = 65_536,
    batch_size: int = 4096,
    halt_threshold: float = 0.50,
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, object]:
    """Measure wall time over ``n`` examples with tensors resident on device.

    Each repeated call processes the same ``n`` examples in batches of at most
    ``batch_size``. Python batching, dynamic active-index construction, and the
    model/reference operation are included. Dataset creation and host-to-device
    transfer are excluded and reported as such.
    """
    if n < 1 or batch_size < 1 or warmup < 0 or repeats < 1:
        raise ValueError("invalid benchmark sizes")
    if not math.isfinite(halt_threshold) or not (0 <= halt_threshold <= 1):
        raise ValueError("halt_threshold must lie in [0,1]")
    model, metadata, device = load_checkpoint(path, device_name=device_name)
    dataset = StructuredFlowDataset(n, seed=seed)
    observed_all = dataset.observed.to(device)
    mask_all = dataset.mask.to(device)
    truth_all = dataset.truth.to(device)
    labels_all = dataset.label.to(device)
    model._validate_inputs(observed_all, mask_all, validate_values=True)

    def model_runner(**kwargs: object) -> dict[str, object]:
        states: list[torch.Tensor] = []
        routes: list[torch.Tensor] = []
        steps: list[torch.Tensor] = []
        update_examples = 0
        recurrent_rounds = 0
        for start_index in range(0, n, batch_size):
            stop_index = min(n, start_index + batch_size)
            output = model.infer(
                observed_all[start_index:stop_index],
                mask_all[start_index:stop_index],
                record_events=False,
                **kwargs,
            )
            state = output["state"]
            route = output["routes"]
            step_count = output["steps_taken"]
            assert torch.is_tensor(state) and torch.is_tensor(route) and torch.is_tensor(step_count)
            states.append(state)
            routes.append(route)
            steps.append(step_count)
            update_examples += int(output["update_examples"])
            recurrent_rounds += int(output["recurrent_rounds"])
        return {
            "state": torch.cat(states, dim=0),
            "routes": torch.cat(routes, dim=0),
            "steps_taken": torch.cat(steps, dim=0),
            "update_examples": update_examples,
            "recurrent_rounds": recurrent_rounds,
            "num_batches": len(states),
        }

    def baseline_runner() -> dict[str, object]:
        states: list[torch.Tensor] = []
        routes: list[torch.Tensor] = []
        for start_index in range(0, n, batch_size):
            stop_index = min(n, start_index + batch_size)
            output = fit_each_structure(
                observed_all[start_index:stop_index],
                mask_all[start_index:stop_index],
                model.bases,
                validate_values=False,
            )
            states.append(output["state"])
            routes.append(output["routes"])
        return {
            "state": torch.cat(states, dim=0),
            "routes": torch.cat(routes, dim=0),
            "num_batches": len(states),
        }

    cases: list[tuple[str, Callable[[], object]]] = []
    for depth in dict.fromkeys((1, 2, 4, model.config.steps)):
        if depth <= model.config.steps:
            cases.append(
                (
                    f"fixed_depth_{depth}",
                    lambda depth=depth: model_runner(
                        max_steps=depth,
                        halt_threshold=1.0,
                        fixed_depth=True,
                        execution_mode="dense",
                    ),
                )
            )
    cases.extend(
        [
            (
                "adaptive_dense",
                lambda: model_runner(
                    max_steps=model.config.steps,
                    halt_threshold=halt_threshold,
                    execution_mode="dense",
                ),
            ),
            (
                "adaptive_compact",
                lambda: model_runner(
                    max_steps=model.config.steps,
                    halt_threshold=halt_threshold,
                    execution_mode="compact",
                ),
            ),
            ("fit_each_structure", baseline_runner),
        ]
    )

    rows: list[dict[str, object]] = []
    for name, function in cases:
        milliseconds, raw = _time_call(
            function, device=device, warmup=warmup, repeats=repeats
        )
        if not isinstance(raw, dict):
            raise RuntimeError("benchmark function returned an unexpected value")
        state = raw["state"]
        routes = raw["routes"]
        assert torch.is_tensor(state) and torch.is_tensor(routes)
        mse = float((state - truth_all).square().mean().item())
        route_accuracy = float((routes == labels_all).float().mean().item())
        row: dict[str, object] = {
            "method": name,
            "milliseconds_for_n_examples": milliseconds,
            "microseconds_per_example": milliseconds * 1000.0 / n,
            "examples_per_second": n / (milliseconds / 1000.0),
            "reconstruction_mse": mse,
            "router_accuracy": route_accuracy,
        }
        if "steps_taken" in raw:
            steps_taken = raw["steps_taken"]
            assert torch.is_tensor(steps_taken)
            row.update(
                {
                    "mean_logical_steps": float(steps_taken.float().mean().item()),
                    "update_examples_per_sample": int(raw["update_examples"]) / n,
                    "mean_recurrent_rounds_per_batch": int(raw["recurrent_rounds"]) / int(raw["num_batches"]),
                }
            )
        rows.append(row)

    result: dict[str, object] = {
        "checkpoint": str(path),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch": str(torch.__version__),
        "seed": seed,
        "n": n,
        "batch_size": batch_size,
        "num_batches_per_repeat": math.ceil(n / batch_size),
        "warmup": warmup,
        "repeats": repeats,
        "halt_threshold": halt_threshold,
        "timed_scope": "all n model/reference calls with inputs already on device; Python batching and compact indexing included; data generation and transfer excluded",
        "rows": rows,
        "warning": "Lower logical/update counts are not a speedup unless wall-clock time is also lower at comparable quality.",
    }
    print(json.dumps(result, indent=2))
    return result


__all__ = [
    "benchmark_checkpoint",
    "build_model",
    "evaluate_checkpoint",
    "evaluate_model",
    "load_checkpoint",
    "pick_device",
    "sha256_file",
    "train_checkpoint",
    "_write_json_no_overwrite",
]
