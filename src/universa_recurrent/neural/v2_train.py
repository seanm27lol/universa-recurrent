"""Training, calibration, evaluation, and timing for uncertainty-aware v2."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math
import os
import time
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .. import __version__
from .baselines import fit_each_structure, gaussian_generator_reference
from .data import StructuredFlowDataset, candidate_bases
from .train import (
    _all_floating_tensors_finite,
    _environment,
    _require_supported_torch,
    pick_device,
    sha256_file,
)
from .v2 import (
    AmbientRecurrentControl,
    CalibrationConfig,
    DecisionPolicy,
    DirectMultiHypothesisNet,
    MultiHypothesisRecurrentNet,
    UntiedMultiHypothesisNet,
    V2Config,
    ambient_training_loss,
    apply_policy_to_trajectory,
    direct_training_loss,
    parameter_count,
    v2_training_loss,
)

_CHECKPOINT_FORMAT = "universa-recurrent.neural-checkpoint.v2"


def _write_json_no_overwrite(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_positive_ints(**values: int) -> None:
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def _make_config(
    bases: torch.Tensor,
    *,
    hidden_dim: int,
    steps: int,
    candidate_embedding_dim: int = 8,
) -> V2Config:
    return V2Config(
        ambient_dim=int(bases.shape[1]),
        latent_dim=int(bases.shape[2]),
        num_structures=int(bases.shape[0]),
        hidden_dim=hidden_dim,
        candidate_embedding_dim=candidate_embedding_dim,
        steps=steps,
    )


def _closest_control_width(
    bases: torch.Tensor,
    main_count: int,
    *,
    steps: int,
    candidate_embedding_dim: int,
    kind: str,
    max_width: int,
) -> tuple[int, int]:
    """Return the width whose parameter count is closest to the main model."""
    best: tuple[int, int, int] | None = None
    for width in range(4, max_width + 1):
        config = _make_config(
            bases,
            hidden_dim=width,
            steps=steps,
            candidate_embedding_dim=candidate_embedding_dim,
        )
        if kind == "direct":
            model: nn.Module = DirectMultiHypothesisNet(bases, config)
        elif kind == "untied":
            model = UntiedMultiHypothesisNet(bases, config)
        else:
            raise ValueError(f"unknown control kind {kind}")
        count = parameter_count(model)
        distance = abs(count - main_count)
        candidate = (distance, width, count)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1], best[2]



def _closest_ambient_width(
    ambient_dim: int,
    steps: int,
    main_count: int,
    *,
    max_width: int,
) -> tuple[int, int]:
    best: tuple[int, int, int] | None = None
    for width in range(4, max_width + 1):
        model = AmbientRecurrentControl(ambient_dim, width, steps)
        count = parameter_count(model)
        candidate = (abs(count - main_count), width, count)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1], best[2]

def build_v2_models(
    device: torch.device,
    *,
    hidden_dim: int = 64,
    steps: int = 8,
    candidate_embedding_dim: int = 8,
    include_controls: bool = True,
) -> tuple[
    MultiHypothesisRecurrentNet,
    DirectMultiHypothesisNet | None,
    UntiedMultiHypothesisNet | None,
    dict[str, object],
]:
    names, bases_np, boundaries_np = candidate_bases()
    bases = torch.from_numpy(bases_np).to(device)
    main_config = _make_config(
        bases,
        hidden_dim=hidden_dim,
        steps=steps,
        candidate_embedding_dim=candidate_embedding_dim,
    )
    main = MultiHypothesisRecurrentNet(bases, main_config).to(device)
    main_count = parameter_count(main)
    direct: DirectMultiHypothesisNet | None = None
    untied: UntiedMultiHypothesisNet | None = None
    controls: dict[str, object] = {}
    if include_controls:
        direct_width, direct_count = _closest_control_width(
            bases.detach().cpu(),
            main_count,
            steps=steps,
            candidate_embedding_dim=candidate_embedding_dim,
            kind="direct",
            max_width=max(16, hidden_dim * 3),
        )
        direct_config = _make_config(
            bases,
            hidden_dim=direct_width,
            steps=steps,
            candidate_embedding_dim=candidate_embedding_dim,
        )
        direct = DirectMultiHypothesisNet(bases, direct_config).to(device)

        untied_width, untied_count = _closest_control_width(
            bases.detach().cpu(),
            main_count,
            steps=steps,
            candidate_embedding_dim=candidate_embedding_dim,
            kind="untied",
            max_width=max(8, hidden_dim),
        )
        untied_config = _make_config(
            bases,
            hidden_dim=untied_width,
            steps=steps,
            candidate_embedding_dim=candidate_embedding_dim,
        )
        untied = UntiedMultiHypothesisNet(bases, untied_config).to(device)
        controls = {
            "direct": {
                "config": asdict(direct_config),
                "parameters": direct_count,
                "parameter_ratio_to_main": direct_count / main_count,
            },
            "untied": {
                "config": asdict(untied_config),
                "parameters": untied_count,
                "parameter_ratio_to_main": untied_count / main_count,
            },
        }
    metadata: dict[str, object] = {
        "names": names,
        "bases": torch.from_numpy(bases_np),
        "boundaries": torch.from_numpy(boundaries_np),
        "main_config": asdict(main_config),
        "main_parameters": main_count,
        "controls": controls,
    }
    return main, direct, untied, metadata


def _loader(
    dataset: StructuredFlowDataset,
    *,
    batch_size: int,
    shuffle_seed: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(shuffle_seed),
        num_workers=0,
        pin_memory=pin_memory,
    )


def _train_model(
    model: nn.Module,
    dataset: StructuredFlowDataset,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    shuffle_seed: int,
    label: str,
    loss_kind: str,
) -> list[dict[str, float | int]]:
    loader = _loader(
        dataset,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    metric_names = (
        "loss",
        "true_route_final_mse",
        "mixture_final_mse",
        "route_ce",
    )
    history: list[dict[str, float | int]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        sums = torch.zeros(len(metric_names), dtype=torch.float64, device=device)
        count = 0
        for batch in loader:
            observed = batch["observed"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            truth = batch["truth"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if loss_kind == "recurrent":
                assert isinstance(model, (MultiHypothesisRecurrentNet, UntiedMultiHypothesisNet))
                outputs = model.forward_train(observed, mask)
                loss, metrics = v2_training_loss(outputs, truth, labels)
            elif loss_kind == "direct":
                assert isinstance(model, DirectMultiHypothesisNet)
                outputs = model(observed, mask)
                loss, metrics = direct_training_loss(outputs, truth, labels)
            elif loss_kind == "ambient":
                assert isinstance(model, AmbientRecurrentControl)
                states = model.rollout(observed, mask)
                loss, metrics = ambient_training_loss(states, truth)
            else:
                raise ValueError(f"unknown loss kind {loss_kind}")
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {label} loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            values = torch.stack(
                [metrics[name].to(torch.float64) for name in metric_names]
            )
            sums += values * observed.shape[0]
            count += observed.shape[0]
        values = (sums / count).detach().cpu().tolist()
        row: dict[str, float | int] = {"epoch": epoch}
        row.update({name: float(value) for name, value in zip(metric_names, values)})
        history.append(row)
        if label == "main":
            print(
                f"epoch {epoch:02d} loss={row['loss']:.5f} "
                f"true_route_mse={row['true_route_final_mse']:.5f} "
                f"route_ce={row['route_ce']:.4f}"
            )
    model.eval()
    return history


@torch.no_grad()
def collect_trajectory(
    model: MultiHypothesisRecurrentNet | UntiedMultiHypothesisNet,
    dataset: StructuredFlowDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    collected: dict[str, list[torch.Tensor]] = {
        "states": [],
        "route_logits": [],
        "route_probabilities": [],
        "residuals": [],
        "progress": [],
    }
    truths: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        observed = batch["observed"].to(device)
        mask = batch["mask"].to(device)
        output = model.rollout(observed, mask)
        for key in collected:
            collected[key].append(output[key].detach().cpu())
        truths.append(batch["truth"].cpu())
        labels.append(batch["label"].cpu())
    trajectory = {
        key: torch.cat(parts, dim=1)
        for key, parts in collected.items()
    }
    return trajectory, torch.cat(truths), torch.cat(labels)


def policy_metrics(
    trajectory: dict[str, torch.Tensor],
    truth: torch.Tensor,
    labels: torch.Tensor,
    policy: DecisionPolicy,
    *,
    calibration: CalibrationConfig | None = None,
    force_commit: bool = False,
    fixed_depth: bool = False,
) -> dict[str, object]:
    output = apply_policy_to_trajectory(
        trajectory,
        policy,
        force_commit=force_commit,
        fixed_depth=fixed_depth,
    )
    state = output["state"]
    committed = output["committed"]
    top_routes = output["top_routes"]
    decision_steps = output["decision_steps"]
    route_revisions = output["route_revisions"]
    assert all(torch.is_tensor(value) for value in (state, committed, top_routes, decision_steps, route_revisions))
    state = state  # type: ignore[assignment]
    committed = committed  # type: ignore[assignment]
    top_routes = top_routes  # type: ignore[assignment]
    decision_steps = decision_steps  # type: ignore[assignment]
    route_revisions = route_revisions  # type: ignore[assignment]
    errors = (state - truth).square().mean(dim=-1)
    coverage = float(committed.float().mean().item())
    committed_n = int(committed.sum().item())
    abstained = ~committed
    result: dict[str, object] = {
        "n": int(truth.shape[0]),
        "coverage": coverage,
        "committed_n": committed_n,
        "abstained_n": int(abstained.sum().item()),
        "top_route_accuracy": float((top_routes == labels).float().mean().item()),
        "wrong_commit_rate": float((committed & (top_routes != labels)).float().mean().item()),
        "selective_route_accuracy": (
            float((top_routes[committed] == labels[committed]).float().mean().item())
            if committed_n
            else None
        ),
        "output_mse": float(errors.mean().item()),
        "committed_mse": float(errors[committed].mean().item()) if committed_n else None,
        "abstained_mse": float(errors[abstained].mean().item()) if abstained.any() else None,
        "mean_steps": float(decision_steps.float().mean().item()),
        "mean_route_revisions": float(route_revisions.float().mean().item()),
        "policy": policy.as_dict(),
        "force_commit": force_commit,
        "fixed_depth": fixed_depth,
    }
    if calibration is not None:
        objective = (
            float(result["output_mse"])
            + calibration.step_cost * float(result["mean_steps"]) / trajectory["states"].shape[0]
            + calibration.abstain_cost * (1.0 - coverage)
            + calibration.wrong_commit_cost * float(result["wrong_commit_rate"])
        )
        result["objective"] = objective
        result["eligible"] = coverage >= calibration.min_coverage
    return result


def calibrate_policy(
    trajectory: dict[str, torch.Tensor],
    truth: torch.Tensor,
    labels: torch.Tensor,
    *,
    calibration: CalibrationConfig,
) -> tuple[DecisionPolicy, dict[str, object], list[dict[str, object]]]:
    """Choose a policy on calibration data only using a declared objective."""
    rows: list[dict[str, object]] = []
    for commit_probability in (0.65, 0.75, 0.85, 0.90, 0.95):
        for commit_margin in (0.05, 0.10, 0.20, 0.30, 0.40):
            for final_probability in (0.50, 0.55, 0.60, 0.65, 0.70):
                if final_probability > commit_probability:
                    continue
                for final_margin in (0.00, 0.05, 0.10, 0.20):
                    if final_margin > commit_margin:
                        continue
                    for min_steps in (1, 2, 3):
                        if min_steps > trajectory["states"].shape[0]:
                            continue
                        policy = DecisionPolicy(
                            commit_probability=commit_probability,
                            commit_margin=commit_margin,
                            final_probability=final_probability,
                            final_margin=final_margin,
                            min_steps=min_steps,
                        )
                        row = policy_metrics(
                            trajectory,
                            truth,
                            labels,
                            policy,
                            calibration=calibration,
                        )
                        rows.append(row)
    eligible = [row for row in rows if row["eligible"]]
    pool = eligible or rows
    best = min(
        pool,
        key=lambda row: (
            float(row["objective"]),
            -float(row["coverage"]),
            float(row["mean_steps"]),
        ),
    )
    policy = DecisionPolicy(**best["policy"])  # type: ignore[arg-type]
    summary = {
        **best,
        "calibration": calibration.as_dict(),
        "candidate_policies": len(rows),
        "eligible_policies": len(eligible),
        "fallback_no_eligible_policy": not bool(eligible),
    }
    return policy, summary, rows


def _state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def train_v2_checkpoint(
    output: Path,
    *,
    device_name: str = "auto",
    seed: int = 5242,
    train_size: int = 20_000,
    calibration_size: int = 4_000,
    epochs: int = 20,
    batch_size: int = 512,
    hidden_dim: int = 64,
    candidate_embedding_dim: int = 8,
    steps: int = 8,
    learning_rate: float = 2e-3,
    noise_std: float = 0.05,
    observe_probability: float = 0.7,
    calibration_step_cost: float = 0.002,
    calibration_abstain_cost: float = 0.02,
    calibration_wrong_commit_cost: float = 0.05,
    min_coverage: float = 0.75,
    include_controls: bool = True,
    deterministic: bool = True,
) -> dict[str, object]:
    """Train v2 and calibrate commit/abstain policy on a disjoint split."""
    _require_supported_torch()
    sidecar = output.with_name(output.name + ".sha256")
    if output.exists() or sidecar.exists():
        existing = output if output.exists() else sidecar
        raise ValueError(f"Refusing to overwrite {existing}")
    _validate_positive_ints(
        train_size=train_size,
        calibration_size=calibration_size,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        candidate_embedding_dim=candidate_embedding_dim,
        steps=steps,
    )
    for name, value in (
        ("learning_rate", learning_rate),
        ("noise_std", noise_std),
        ("observe_probability", observe_probability),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if learning_rate <= 0 or noise_std <= 0 or not (0 < observe_probability <= 1):
        raise ValueError("learning rate and noise_std must be positive; observe_probability must be in (0,1]")
    if not isinstance(include_controls, bool):
        raise ValueError("include_controls must be bool")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    device = pick_device(device_name)
    main, direct, untied, model_metadata = build_v2_models(
        device,
        hidden_dim=hidden_dim,
        steps=steps,
        candidate_embedding_dim=candidate_embedding_dim,
        include_controls=include_controls,
    )
    ambient: AmbientRecurrentControl | None = None
    fixed_depth_models: dict[str, MultiHypothesisRecurrentNet] = {}
    if include_controls:
        ambient_width, ambient_count = _closest_ambient_width(
            main.config.ambient_dim,
            steps,
            parameter_count(main),
            max_width=max(16, hidden_dim * 3),
        )
        ambient = AmbientRecurrentControl(
            main.config.ambient_dim, ambient_width, steps
        ).to(device)
        control_metadata = model_metadata["controls"]
        assert isinstance(control_metadata, dict)
        control_metadata["ambient"] = {
            "config": {
                "ambient_dim": main.config.ambient_dim,
                "hidden_dim": ambient_width,
                "steps": steps,
            },
            "parameters": ambient_count,
            "parameter_ratio_to_main": ambient_count / parameter_count(main),
        }
        for depth in (1, 2, 4):
            if depth >= steps:
                continue
            name = f"fixed_depth_{depth}"
            config = _make_config(
                main.bases,
                hidden_dim=hidden_dim,
                steps=depth,
                candidate_embedding_dim=candidate_embedding_dim,
            )
            fixed = MultiHypothesisRecurrentNet(main.bases, config).to(device)
            fixed_depth_models[name] = fixed
            control_metadata[name] = {
                "config": asdict(config),
                "parameters": parameter_count(fixed),
                "parameter_ratio_to_main": parameter_count(fixed) / parameter_count(main),
                "dedicated_depth_training": True,
            }
    training_data = StructuredFlowDataset(
        train_size,
        seed=seed + 1,
        noise_std=noise_std,
        observe_probability=observe_probability,
    )
    calibration_data = StructuredFlowDataset(
        calibration_size,
        seed=seed + 2,
        noise_std=noise_std,
        observe_probability=observe_probability,
    )
    started = time.perf_counter()
    histories: dict[str, list[dict[str, float | int]]] = {}
    histories["main"] = _train_model(
        main,
        training_data,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        shuffle_seed=seed + 3,
        label="main",
        loss_kind="recurrent",
    )
    if direct is not None:
        histories["direct"] = _train_model(
            direct,
            training_data,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            shuffle_seed=seed + 3,
            label="direct",
            loss_kind="direct",
        )
    if untied is not None:
        histories["untied"] = _train_model(
            untied,
            training_data,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            shuffle_seed=seed + 3,
            label="untied",
            loss_kind="recurrent",
        )
    if ambient is not None:
        histories["ambient"] = _train_model(
            ambient,
            training_data,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            shuffle_seed=seed + 3,
            label="ambient",
            loss_kind="ambient",
        )
    for name, fixed in fixed_depth_models.items():
        histories[name] = _train_model(
            fixed,
            training_data,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            shuffle_seed=seed + 3,
            label=name,
            loss_kind="recurrent",
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    trajectory, calibration_truth, calibration_labels = collect_trajectory(
        main,
        calibration_data,
        device=device,
        batch_size=min(1024, calibration_size),
    )
    calibration_config = CalibrationConfig(
        step_cost=calibration_step_cost,
        abstain_cost=calibration_abstain_cost,
        wrong_commit_cost=calibration_wrong_commit_cost,
        min_coverage=min_coverage,
    )
    policy, calibration_summary, _ = calibrate_policy(
        trajectory,
        calibration_truth,
        calibration_labels,
        calibration=calibration_config,
    )

    controls_payload: dict[str, object] = {}
    if direct is not None:
        controls_payload["direct"] = {
            **model_metadata["controls"]["direct"],  # type: ignore[index]
            "model_state": _state_dict_cpu(direct),
            "history": histories["direct"],
        }
    if untied is not None:
        controls_payload["untied"] = {
            **model_metadata["controls"]["untied"],  # type: ignore[index]
            "model_state": _state_dict_cpu(untied),
            "history": histories["untied"],
        }
    if ambient is not None:
        controls_payload["ambient"] = {
            **model_metadata["controls"]["ambient"],  # type: ignore[index]
            "model_state": _state_dict_cpu(ambient),
            "history": histories["ambient"],
        }
    for name, fixed in fixed_depth_models.items():
        controls_payload[name] = {
            **model_metadata["controls"][name],  # type: ignore[index]
            "model_state": _state_dict_cpu(fixed),
            "history": histories[name],
        }

    payload: dict[str, object] = {
        "format": _CHECKPOINT_FORMAT,
        "software_version": __version__,
        "model_state": _state_dict_cpu(main),
        "config": model_metadata["main_config"],
        "library": {
            "names": model_metadata["names"],
            "bases": model_metadata["bases"],
            "boundaries": model_metadata["boundaries"],
        },
        "training": {
            "seed": seed,
            "train_seed": seed + 1,
            "calibration_seed": seed + 2,
            "shuffle_seed": seed + 3,
            "train_size": train_size,
            "calibration_size": calibration_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "noise_std": noise_std,
            "observe_probability": observe_probability,
            "deterministic_algorithms": deterministic,
            "test_seed_not_consumed": True,
        },
        "history": histories["main"],
        "calibration": {
            "policy": policy.as_dict(),
            "summary": calibration_summary,
            "objective": calibration_config.as_dict(),
        },
        "controls": controls_payload,
        "parameter_counts": {
            "main": model_metadata["main_parameters"],
            **{
                name: details["parameters"]
                for name, details in model_metadata["controls"].items()  # type: ignore[union-attr]
            },
        },
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
        "calibrated_policy": policy.as_dict(),
        "calibration_summary": calibration_summary,
        "parameter_counts": payload["parameter_counts"],
        "note": "Calibration metrics are not untouched test results.",
    }
    print(json.dumps(result, indent=2))
    return result


def _check_state_bases(state: dict, expected: torch.Tensor, name: str) -> None:
    """A library description cannot vouch for a different tensor in the weights."""
    actual = state.get("bases")
    if not torch.is_tensor(actual) or not torch.equal(actual.detach().cpu(), expected):
        raise ValueError(f"{name} model-state bases do not match the candidate library")


def _require_held_out_seed(seed: int, metadata: dict[str, object]) -> None:
    """Refuse accidentally evaluating on the training/calibration generator seed."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("evaluation seed must be a nonnegative integer")
    training = metadata.get("training")
    if not isinstance(training, dict):
        raise ValueError("training split metadata missing")
    if seed in (training.get("train_seed"), training.get("calibration_seed")):
        raise ValueError("evaluation seed overlaps the training or calibration split")


def load_v2_checkpoint(
    path: Path,
    *,
    device_name: str = "auto",
) -> tuple[
    MultiHypothesisRecurrentNet,
    dict[str, nn.Module],
    dict[str, object],
    torch.device,
]:
    """Load a trusted v2 checkpoint through the restricted weights-only path."""
    _require_supported_torch()
    if not path.is_file():
        raise FileNotFoundError(path)
    device = pick_device(device_name)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or raw.get("format") != _CHECKPOINT_FORMAT:
        raise ValueError("not a neural v2 checkpoint")
    config_data = raw.get("config")
    state = raw.get("model_state")
    library = raw.get("library")
    calibration = raw.get("calibration")
    if not isinstance(config_data, dict) or not isinstance(state, dict):
        raise ValueError("checkpoint model metadata missing")
    if not isinstance(library, dict) or not isinstance(calibration, dict):
        raise ValueError("checkpoint library/calibration metadata missing")
    names_expected, bases_np, boundaries_np = candidate_bases()
    if library.get("names") != names_expected:
        raise ValueError("checkpoint candidate names do not match this release")
    bases = torch.as_tensor(library.get("bases")).float()
    boundaries = torch.as_tensor(library.get("boundaries")).float()
    bases_expected = torch.from_numpy(bases_np)
    boundaries_expected = torch.from_numpy(boundaries_np)
    if not torch.equal(bases, bases_expected):
        raise ValueError("checkpoint candidate bases do not match this release")
    if not torch.equal(boundaries, boundaries_expected):
        raise ValueError("checkpoint boundaries do not match this release")
    _all_floating_tensors_finite(raw)
    config = V2Config(**config_data)
    main = MultiHypothesisRecurrentNet(bases.to(device), config).to(device)
    _check_state_bases(state, bases_expected, "main")
    main.load_state_dict(state, strict=True)
    main.eval()

    controls: dict[str, nn.Module] = {}
    raw_controls = raw.get("controls", {})
    if not isinstance(raw_controls, dict):
        raise ValueError("checkpoint controls metadata is malformed")
    if "direct" in raw_controls:
        entry = raw_controls["direct"]
        if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict) or not isinstance(entry.get("model_state"), dict):
            raise ValueError("direct control metadata is malformed")
        direct = DirectMultiHypothesisNet(
            bases.to(device), V2Config(**entry["config"])
        ).to(device)
        _check_state_bases(entry["model_state"], bases_expected, "direct")
        direct.load_state_dict(entry["model_state"], strict=True)
        direct.eval()
        controls["direct"] = direct
    if "untied" in raw_controls:
        entry = raw_controls["untied"]
        if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict) or not isinstance(entry.get("model_state"), dict):
            raise ValueError("untied control metadata is malformed")
        untied = UntiedMultiHypothesisNet(
            bases.to(device), V2Config(**entry["config"])
        ).to(device)
        _check_state_bases(entry["model_state"], bases_expected, "untied")
        untied.load_state_dict(entry["model_state"], strict=True)
        untied.eval()
        controls["untied"] = untied
    if "ambient" in raw_controls:
        entry = raw_controls["ambient"]
        if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict) or not isinstance(entry.get("model_state"), dict):
            raise ValueError("ambient control metadata is malformed")
        ambient_config = entry["config"]
        ambient = AmbientRecurrentControl(
            int(ambient_config["ambient_dim"]),
            int(ambient_config["hidden_dim"]),
            int(ambient_config["steps"]),
        ).to(device)
        ambient.load_state_dict(entry["model_state"], strict=True)
        ambient.eval()
        controls["ambient"] = ambient
    for name, entry in raw_controls.items():
        if not name.startswith("fixed_depth_"):
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("config"), dict) or not isinstance(entry.get("model_state"), dict):
            raise ValueError(f"{name} control metadata is malformed")
        fixed = MultiHypothesisRecurrentNet(
            bases.to(device), V2Config(**entry["config"])
        ).to(device)
        _check_state_bases(entry["model_state"], bases_expected, "fixed")
        fixed.load_state_dict(entry["model_state"], strict=True)
        fixed.eval()
        controls[name] = fixed

    policy_data = calibration.get("policy")
    if not isinstance(policy_data, dict):
        raise ValueError("calibrated policy missing")
    loaded_policy = DecisionPolicy(**policy_data)
    if loaded_policy.min_steps > config.steps:
        raise ValueError("checkpoint policy min_steps exceeds model depth")
    metadata: dict[str, object] = {
        key: value
        for key, value in raw.items()
        if key not in ("model_state", "controls")
    }
    metadata.update(
        {
            "names": names_expected,
            "bases": bases,
            "boundaries": boundaries,
            "checkpoint_path": str(path),
            "checkpoint_sha256": sha256_file(path),
            "control_metadata": {
                key: {
                    subkey: subvalue
                    for subkey, subvalue in value.items()
                    if subkey != "model_state"
                }
                for key, value in raw_controls.items()
                if isinstance(value, dict)
            },
        }
    )
    return main, controls, metadata, device


def _dataset_settings(metadata: dict[str, object]) -> tuple[float, float]:
    training = metadata.get("training", {})
    if not isinstance(training, dict):
        return 0.05, 0.7
    noise = float(training.get("noise_std", 0.05))
    observed = float(training.get("observe_probability", 0.7))
    if not math.isfinite(noise) or noise <= 0 or not (0 < observed <= 1):
        raise ValueError("v2 evaluation requires positive noise_std and observation probability in (0,1]")
    return noise, observed


def _prediction_metrics(
    *,
    state: torch.Tensor,
    top_routes: torch.Tensor,
    committed: torch.Tensor,
    truth: torch.Tensor,
    labels: torch.Tensor,
    decision_steps: torch.Tensor | None = None,
    route_revisions: torch.Tensor | None = None,
    route_probabilities: torch.Tensor | None = None,
) -> dict[str, object]:
    errors = (state - truth).square().mean(dim=-1)
    coverage = float(committed.float().mean().item())
    committed_n = int(committed.sum().item())
    abstained = ~committed
    result: dict[str, object] = {
        "n": int(truth.shape[0]),
        "coverage": coverage,
        "committed_n": committed_n,
        "abstained_n": int(abstained.sum().item()),
        "top_route_accuracy": float((top_routes == labels).float().mean().item()),
        "wrong_commit_rate": float((committed & (top_routes != labels)).float().mean().item()),
        "selective_route_accuracy": (
            float((top_routes[committed] == labels[committed]).float().mean().item())
            if committed_n
            else None
        ),
        "output_mse": float(errors.mean().item()),
        "committed_mse": float(errors[committed].mean().item()) if committed_n else None,
        "abstained_mse": float(errors[abstained].mean().item()) if abstained.any() else None,
    }
    if decision_steps is not None:
        result["mean_steps"] = float(decision_steps.float().mean().item())
    if route_revisions is not None:
        result["mean_route_revisions"] = float(route_revisions.float().mean().item())
        result["revision_fraction"] = float((route_revisions > 0).float().mean().item())
    if route_probabilities is not None:
        if (
            route_probabilities.ndim != 2
            or route_probabilities.shape[0] != labels.shape[0]
            or route_probabilities.shape[1] < 2
        ):
            raise ValueError("route_probabilities must have shape [B,K]")
        if not torch.isfinite(route_probabilities).all():
            raise ValueError("route_probabilities must be finite")
        one_hot = torch.nn.functional.one_hot(
            labels, num_classes=route_probabilities.shape[1]
        ).to(route_probabilities.dtype)
        clipped = route_probabilities.clamp_min(1e-12)
        confidence, predicted = route_probabilities.max(dim=-1)
        correct = (predicted == labels).to(route_probabilities.dtype)
        ece = route_probabilities.new_zeros(())
        for bin_index in range(10):
            lower = bin_index / 10.0
            upper = (bin_index + 1) / 10.0
            in_bin = (confidence >= lower) & (
                confidence <= upper if bin_index == 9 else confidence < upper
            )
            if in_bin.any():
                ece = ece + in_bin.float().mean() * (
                    confidence[in_bin].mean() - correct[in_bin].mean()
                ).abs()
        result.update(
            {
                "route_brier": float(
                    (route_probabilities - one_hot).square().sum(dim=-1).mean().item()
                ),
                "route_nll": float(
                    -clipped[torch.arange(labels.shape[0], device=labels.device), labels]
                    .log()
                    .mean()
                    .item()
                ),
                "route_ece_10bin": float(ece.item()),
                "mean_route_confidence": float(confidence.mean().item()),
            }
        )
    return result


@torch.no_grad()
def evaluate_v2_checkpoint(
    path: Path,
    *,
    device_name: str = "auto",
    seed: int = 12_000,
    n: int = 4_000,
    batch_size: int = 1024,
) -> dict[str, object]:
    """Evaluate the frozen calibrated policy once on a separately seeded set."""
    _validate_positive_ints(n=n, batch_size=batch_size)
    main, controls, metadata, device = load_v2_checkpoint(
        path, device_name=device_name
    )
    _require_held_out_seed(seed, metadata)
    noise_std, observe_probability = _dataset_settings(metadata)
    dataset = StructuredFlowDataset(
        n,
        seed=seed,
        noise_std=noise_std,
        observe_probability=observe_probability,
    )
    calibration = metadata["calibration"]
    assert isinstance(calibration, dict) and isinstance(calibration["policy"], dict)
    policy = DecisionPolicy(**calibration["policy"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def collect_main(
        *,
        selected_policy: DecisionPolicy,
        max_steps: int,
        fixed_depth: bool,
        force_commit: bool,
    ) -> dict[str, object]:
        states: list[torch.Tensor] = []
        tops: list[torch.Tensor] = []
        committed_values: list[torch.Tensor] = []
        truths: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        steps_values: list[torch.Tensor] = []
        revisions: list[torch.Tensor] = []
        probabilities_values: list[torch.Tensor] = []
        candidate_updates = 0
        for batch in loader:
            observed = batch["observed"].to(device)
            mask = batch["mask"].to(device)
            output = main.infer(
                observed,
                mask,
                policy=selected_policy,
                max_steps=max_steps,
                execution_mode="compact",
                fixed_depth=fixed_depth,
                force_commit=force_commit,
            )
            states.append(output["state"])  # type: ignore[arg-type]
            tops.append(output["top_routes"])  # type: ignore[arg-type]
            committed_values.append(output["committed"])  # type: ignore[arg-type]
            steps_values.append(output["decision_steps"])  # type: ignore[arg-type]
            revisions.append(output["route_revisions"])  # type: ignore[arg-type]
            probabilities_values.append(output["route_probabilities"])  # type: ignore[arg-type]
            candidate_updates += int(output["candidate_update_examples"])
            truths.append(batch["truth"].to(device))
            labels.append(batch["label"].to(device))
        state = torch.cat(states)
        top = torch.cat(tops)
        committed_tensor = torch.cat(committed_values)
        truth = torch.cat(truths)
        label = torch.cat(labels)
        decision_steps = torch.cat(steps_values)
        revision_tensor = torch.cat(revisions)
        probability_tensor = torch.cat(probabilities_values)
        result = _prediction_metrics(
            state=state,
            top_routes=top,
            committed=committed_tensor,
            truth=truth,
            labels=label,
            decision_steps=decision_steps,
            route_revisions=revision_tensor,
            route_probabilities=probability_tensor,
        )
        result["candidate_update_examples_per_sample"] = candidate_updates / n
        return result

    rows: list[dict[str, object]] = []
    calibrated = collect_main(
        selected_policy=policy,
        max_steps=main.config.steps,
        fixed_depth=False,
        force_commit=False,
    )
    calibrated.update({"method": "v2_calibrated_policy", "policy": policy.as_dict()})
    rows.append(calibrated)

    force_policy = DecisionPolicy(
        commit_probability=1.0,
        commit_margin=1.0,
        final_probability=0.0,
        final_margin=0.0,
        min_steps=1,
    )
    for depth in dict.fromkeys((1, 2, 4, main.config.steps)):
        if depth <= main.config.steps:
            row = collect_main(
                selected_policy=force_policy,
                max_steps=depth,
                fixed_depth=True,
                force_commit=True,
            )
            row.update({"method": f"v2_shared_truncated_depth_{depth}", "trained_for_depth": main.config.steps})
            rows.append(row)

    all_truth = dataset.truth.to(device)
    all_labels = dataset.label.to(device)
    all_observed = dataset.observed.to(device)
    all_mask = dataset.mask.to(device)
    trajectory = main.rollout(all_observed, all_mask)
    final_states = trajectory["states"][-1]
    final_probabilities = trajectory["route_probabilities"][-1]
    mixture = torch.einsum("bk,bkn->bn", final_probabilities, final_states)
    top = final_probabilities.argmax(dim=-1)
    rows.append(
        {
            "method": "v2_final_posterior_mixture",
            **_prediction_metrics(
                state=mixture,
                top_routes=top,
                committed=torch.zeros(n, dtype=torch.bool, device=device),
                truth=all_truth,
                labels=all_labels,
                route_probabilities=final_probabilities,
            ),
        }
    )
    index = torch.arange(n, device=device)
    true_state = final_states[index, all_labels]
    rows.append(
        {
            "method": "v2_true_structure_evaluation_only",
            "n": n,
            "output_mse": float((true_state - all_truth).square().mean().item()),
            "privileged": True,
        }
    )

    if "direct" in controls:
        direct = controls["direct"]
        assert isinstance(direct, DirectMultiHypothesisNet)
        output = direct(all_observed, all_mask)
        probabilities = output["route_probabilities"]
        states = output["states"]
        top = probabilities.argmax(dim=-1)
        selected = states[index, top]
        rows.append(
            {
                "method": "direct_parameter_matched_hard",
                **_prediction_metrics(
                    state=selected,
                    top_routes=top,
                    committed=torch.ones(n, dtype=torch.bool, device=device),
                    truth=all_truth,
                    labels=all_labels,
                    route_probabilities=probabilities,
                ),
                "parameters": parameter_count(direct),
            }
        )
        rows.append(
            {
                "method": "direct_parameter_matched_mixture",
                **_prediction_metrics(
                    state=output["mixture_state"],
                    top_routes=top,
                    committed=torch.zeros(n, dtype=torch.bool, device=device),
                    truth=all_truth,
                    labels=all_labels,
                    route_probabilities=probabilities,
                ),
                "parameters": parameter_count(direct),
            }
        )

    if "untied" in controls:
        untied = controls["untied"]
        assert isinstance(untied, UntiedMultiHypothesisNet)
        output = untied.rollout(all_observed, all_mask)
        states = output["states"][-1]
        probabilities = output["route_probabilities"][-1]
        top = probabilities.argmax(dim=-1)
        selected = states[index, top]
        rows.append(
            {
                "method": f"untied_parameter_matched_hard_depth_{untied.config.steps}",
                **_prediction_metrics(
                    state=selected,
                    top_routes=top,
                    committed=torch.ones(n, dtype=torch.bool, device=device),
                    truth=all_truth,
                    labels=all_labels,
                    route_probabilities=probabilities,
                ),
                "parameters": parameter_count(untied),
            }
        )
        rows.append(
            {
                "method": f"untied_parameter_matched_mixture_depth_{untied.config.steps}",
                **_prediction_metrics(
                    state=torch.einsum("bk,bkn->bn", probabilities, states),
                    top_routes=top,
                    committed=torch.zeros(n, dtype=torch.bool, device=device),
                    truth=all_truth,
                    labels=all_labels,
                    route_probabilities=probabilities,
                ),
                "parameters": parameter_count(untied),
            }
        )

    if "ambient" in controls:
        ambient = controls["ambient"]
        assert isinstance(ambient, AmbientRecurrentControl)
        ambient_state = ambient.rollout(all_observed, all_mask)[-1]
        rows.append(
            {
                "method": f"ambient_recurrent_parameter_matched_depth_{ambient.steps}",
                "n": n,
                "output_mse": float((ambient_state - all_truth).square().mean().item()),
                "parameters": parameter_count(ambient),
                "structural_route": False,
            }
        )

    for name, control in sorted(controls.items()):
        if not name.startswith("fixed_depth_"):
            continue
        assert isinstance(control, MultiHypothesisRecurrentNet)
        output = control.rollout(all_observed, all_mask)
        states = output["states"][-1]
        probabilities = output["route_probabilities"][-1]
        top = probabilities.argmax(dim=-1)
        selected = states[index, top]
        rows.append(
            {
                "method": f"dedicated_{name}",
                **_prediction_metrics(
                    state=selected,
                    top_routes=top,
                    committed=torch.ones(n, dtype=torch.bool, device=device),
                    truth=all_truth,
                    labels=all_labels,
                    route_probabilities=probabilities,
                ),
                "parameters": parameter_count(control),
                "dedicated_depth_training": True,
            }
        )

    references: list[dict[str, object]] = []
    transparent = fit_each_structure(all_observed, all_mask, main.bases, validate_values=False)
    references.append(
        {
            "method": "fit_each_structure",
            **_prediction_metrics(
                state=transparent["state"],
                top_routes=transparent["routes"],
                committed=torch.ones(n, dtype=torch.bool, device=device),
                truth=all_truth,
                labels=all_labels,
            ),
            "privileged": False,
        }
    )
    gaussian = gaussian_generator_reference(
        all_observed,
        all_mask,
        main.bases,
        noise_std=noise_std,
        validate_values=False,
    )
    for method, state in (
        ("gaussian_hard", gaussian["hard_state"]),
        ("gaussian_mixture", gaussian["mixture_state"]),
    ):
        references.append(
            {
                "method": method,
                **_prediction_metrics(
                    state=state,
                    top_routes=gaussian["routes"],
                    committed=(
                        torch.ones(n, dtype=torch.bool, device=device)
                        if method == "gaussian_hard"
                        else torch.zeros(n, dtype=torch.bool, device=device)
                    ),
                    truth=all_truth,
                    labels=all_labels,
                    route_probabilities=gaussian["route_probabilities"],
                ),
                "privileged": True,
            }
        )

    result: dict[str, object] = {
        "checkpoint": str(path),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "checkpoint_format": _CHECKPOINT_FORMAT,
        "device": str(device),
        "test_seed": seed,
        "test_seed_distinct_from_training": seed not in {
            int(metadata["training"]["train_seed"]),  # type: ignore[index]
            int(metadata["training"]["calibration_seed"]),  # type: ignore[index]
        },
        "dataset": {
            "n": n,
            "noise_std": noise_std,
            "observe_probability": observe_probability,
        },
        "calibrated_policy": policy.as_dict(),
        "calibration_summary": calibration.get("summary"),
        "rows": rows,
        "references": references,
        "notes": [
            "The policy was chosen on the checkpoint's calibration split, not this test seed.",
            "Abstained examples return a probability-weighted provisional mixture, not a single-structure certificate.",
            "Truncated depths share one max-depth-trained model; separately trained depth controls are reported when present.",
            "Direct and untied controls are approximately parameter matched; exact counts are reported.",
            "The Gaussian references know the exact synthetic generator and are privileged.",
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
    return (time.perf_counter() - started) * 1000.0 / repeats, result


@torch.no_grad()
def benchmark_v2_checkpoint(
    path: Path,
    *,
    device_name: str = "auto",
    seed: int = 12_100,
    n: int = 65_536,
    batch_size: int = 4096,
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, object]:
    _validate_positive_ints(n=n, batch_size=batch_size, repeats=repeats)
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    main, controls, metadata, device = load_v2_checkpoint(path, device_name=device_name)
    _require_held_out_seed(seed, metadata)
    noise_std, observe_probability = _dataset_settings(metadata)
    dataset = StructuredFlowDataset(
        n,
        seed=seed,
        noise_std=noise_std,
        observe_probability=observe_probability,
    )
    observed = dataset.observed.to(device)
    mask = dataset.mask.to(device)
    truth = dataset.truth.to(device)
    labels = dataset.label.to(device)
    main._validate_inputs(observed, mask, validate_values=True)
    calibration = metadata["calibration"]
    assert isinstance(calibration, dict) and isinstance(calibration["policy"], dict)
    policy = DecisionPolicy(**calibration["policy"])

    def batched_main(mode: str, *, fixed: bool = False, depth: int | None = None) -> dict[str, object]:
        outputs: list[dict[str, object]] = []
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            outputs.append(
                main.infer(
                    observed[start:stop],
                    mask[start:stop],
                    policy=policy,
                    max_steps=depth or main.config.steps,
                    execution_mode=mode,  # type: ignore[arg-type]
                    fixed_depth=fixed,
                    force_commit=fixed,
                )
            )
        return {
            "state": torch.cat([output["state"] for output in outputs]),  # type: ignore[list-item]
            "top_routes": torch.cat([output["top_routes"] for output in outputs]),  # type: ignore[list-item]
            "committed": torch.cat([output["committed"] for output in outputs]),  # type: ignore[list-item]
            "decision_steps": torch.cat([output["decision_steps"] for output in outputs]),  # type: ignore[list-item]
            "candidate_update_examples": sum(int(output["candidate_update_examples"]) for output in outputs),
        }

    def direct_runner() -> dict[str, object]:
        direct = controls["direct"]
        assert isinstance(direct, DirectMultiHypothesisNet)
        parts = []
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            parts.append(direct(observed[start:stop], mask[start:stop]))
        probabilities = torch.cat([part["route_probabilities"] for part in parts])
        states = torch.cat([part["states"] for part in parts])
        top = probabilities.argmax(dim=-1)
        index = torch.arange(n, device=device)
        return {
            "state": states[index, top],
            "top_routes": top,
            "committed": torch.ones(n, dtype=torch.bool, device=device),
        }

    def untied_runner() -> dict[str, object]:
        untied = controls["untied"]
        assert isinstance(untied, UntiedMultiHypothesisNet)
        parts = []
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            parts.append(untied.rollout(observed[start:stop], mask[start:stop]))
        probabilities = torch.cat([part["route_probabilities"][-1] for part in parts])
        states = torch.cat([part["states"][-1] for part in parts])
        top = probabilities.argmax(dim=-1)
        index = torch.arange(n, device=device)
        return {
            "state": states[index, top],
            "top_routes": top,
            "committed": torch.ones(n, dtype=torch.bool, device=device),
        }

    def ambient_runner() -> dict[str, object]:
        ambient = controls["ambient"]
        assert isinstance(ambient, AmbientRecurrentControl)
        states = []
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            states.append(ambient.rollout(observed[start:stop], mask[start:stop])[-1])
        return {
            "state": torch.cat(states),
            "route_metrics_applicable": False,
        }

    def fixed_control_runner(control_name: str) -> dict[str, object]:
        control = controls[control_name]
        assert isinstance(control, MultiHypothesisRecurrentNet)
        parts = []
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            parts.append(control.rollout(observed[start:stop], mask[start:stop]))
        probabilities = torch.cat([part["route_probabilities"][-1] for part in parts])
        states = torch.cat([part["states"][-1] for part in parts])
        top = probabilities.argmax(dim=-1)
        index = torch.arange(n, device=device)
        return {
            "state": states[index, top],
            "top_routes": top,
            "committed": torch.ones(n, dtype=torch.bool, device=device),
        }

    def transparent_runner() -> dict[str, object]:
        states = []
        routes = []
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            output = fit_each_structure(
                observed[start:stop], mask[start:stop], main.bases, validate_values=False
            )
            states.append(output["state"])
            routes.append(output["routes"])
        return {
            "state": torch.cat(states),
            "top_routes": torch.cat(routes),
            "committed": torch.ones(n, dtype=torch.bool, device=device),
        }

    cases: list[tuple[str, Callable[[], dict[str, object]]]] = [
        (
            f"v2_fixed_depth_{main.config.steps}",
            lambda: batched_main("dense", fixed=True, depth=main.config.steps),
        ),
        ("v2_calibrated_dense", lambda: batched_main("dense")),
        ("v2_calibrated_compact", lambda: batched_main("compact")),
    ]
    if "direct" in controls:
        cases.append(("direct_parameter_matched", direct_runner))
    if "untied" in controls:
        cases.append(("untied_parameter_matched", untied_runner))
    if "ambient" in controls:
        cases.append(("ambient_recurrent_parameter_matched", ambient_runner))
    for control_name in sorted(controls):
        if control_name.startswith("fixed_depth_"):
            cases.append(
                (
                    f"dedicated_{control_name}",
                    lambda control_name=control_name: fixed_control_runner(control_name),
                )
            )
    cases.append(("fit_each_structure", transparent_runner))

    rows: list[dict[str, object]] = []
    for name, function in cases:
        milliseconds, raw = _time_call(
            function, device=device, warmup=warmup, repeats=repeats
        )
        state = raw["state"]
        assert torch.is_tensor(state)
        base_row: dict[str, object] = {
            "method": name,
            "milliseconds_for_n_examples": milliseconds,
            "microseconds_per_example": milliseconds * 1000.0 / n,
            "examples_per_second": n / (milliseconds / 1000.0),
        }
        if raw.get("route_metrics_applicable", True):
            top = raw["top_routes"]
            committed = raw["committed"]
            assert torch.is_tensor(top) and torch.is_tensor(committed)
            base_row.update(
                _prediction_metrics(
                    state=state,
                    top_routes=top,
                    committed=committed,
                    truth=truth,
                    labels=labels,
                    decision_steps=(
                        raw.get("decision_steps")
                        if torch.is_tensor(raw.get("decision_steps"))
                        else None
                    ),
                )
            )
        else:
            base_row.update(
                {
                    "n": n,
                    "output_mse": float((state - truth).square().mean().item()),
                    "route_metrics_applicable": False,
                }
            )
        if "candidate_update_examples" in raw:
            base_row["candidate_update_examples_per_sample"] = (
                int(raw["candidate_update_examples"]) / n
            )
        rows.append(base_row)

    result: dict[str, object] = {
        "checkpoint": str(path),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "seed": seed,
        "n": n,
        "batch_size": batch_size,
        "warmup": warmup,
        "repeats": repeats,
        "policy": policy.as_dict(),
        "timed_scope": "inputs resident on device; Python batching and dynamic indexing included; data generation and transfer excluded",
        "rows": rows,
        "warning": "Coverage, quality, and latency must be read together. Skipped candidate updates do not imply lower wall-clock time.",
    }
    print(json.dumps(result, indent=2))
    return result
