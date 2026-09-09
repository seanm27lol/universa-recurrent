"""A small structure-aware recurrent network.

The learned system routes once into one explicit candidate subspace and then
reuses one update network over latent coordinates.  The representation
``state = basis @ coordinates`` keeps every decoded state inside the selected
subspace up to floating-point error.

The implementation intentionally separates:

* logical updates assigned to samples;
* examples actually sent through the update network; and
* recurrent rounds reached by the batch.

Those quantities are not interchangeable with wall-clock speed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

ResidualNormalization = Literal["observed", "ambient"]
HaltTiming = Literal["after_update", "before_update"]
ExecutionMode = Literal["dense", "compact"]


@dataclass(frozen=True)
class NeuralConfig:
    ambient_dim: int = 5
    latent_dim: int = 2
    num_structures: int = 2
    hidden_dim: int = 64
    steps: int = 8
    residual_normalization: ResidualNormalization = "observed"
    halt_timing: HaltTiming = "after_update"

    def __post_init__(self) -> None:
        for name in ("ambient_dim", "latent_dim", "num_structures", "hidden_dim", "steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.residual_normalization not in ("observed", "ambient"):
            raise ValueError("residual_normalization must be 'observed' or 'ambient'")
        if self.halt_timing not in ("after_update", "before_update"):
            raise ValueError("halt_timing must be 'after_update' or 'before_update'")


class StructuredRecurrentNet(nn.Module):
    """Route once, then recurrently update coordinates in the selected space."""

    def __init__(self, bases: torch.Tensor, config: NeuralConfig | None = None) -> None:
        super().__init__()
        if bases.ndim != 3:
            raise ValueError("bases must have shape [K,N,D]")
        if not bases.is_floating_point():
            raise ValueError("bases must be floating point")
        if not torch.isfinite(bases).all():
            raise ValueError("bases must be finite")
        num_structures, ambient_dim, latent_dim = bases.shape
        cfg = config or NeuralConfig(
            ambient_dim=ambient_dim,
            latent_dim=latent_dim,
            num_structures=num_structures,
        )
        if (cfg.num_structures, cfg.ambient_dim, cfg.latent_dim) != (
            num_structures,
            ambient_dim,
            latent_dim,
        ):
            raise ValueError("config dimensions do not match bases")

        gram = torch.einsum("knd,kne->kde", bases.double(), bases.double())
        identity = torch.eye(latent_dim, dtype=torch.float64, device=bases.device)
        if not torch.allclose(gram, identity.expand_as(gram), atol=1e-6, rtol=1e-6):
            raise ValueError("candidate bases must have orthonormal columns")

        self.config = cfg
        self.register_buffer("bases", bases.float().contiguous())
        self.encoder = nn.Sequential(
            nn.Linear(2 * ambient_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        self.router = nn.Linear(cfg.hidden_dim, num_structures)
        update_input = cfg.hidden_dim + 2 * latent_dim + 1
        self.update = nn.Sequential(
            nn.Linear(update_input, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, latent_dim),
        )
        self.halt = nn.Sequential(
            nn.Linear(update_input, max(1, cfg.hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, cfg.hidden_dim // 2), 1),
        )

    def _validate_inputs(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        validate_values: bool,
    ) -> None:
        if observed.ndim != 2 or mask.shape != observed.shape:
            raise ValueError("observed and mask must have matching [B,N] shapes")
        if observed.shape[1] != self.config.ambient_dim:
            raise ValueError("observation width does not match model ambient_dim")
        if observed.device != mask.device or observed.device != self.bases.device:
            raise ValueError("observed, mask, and model bases must share a device")
        if observed.dtype != mask.dtype or observed.dtype != self.bases.dtype:
            raise ValueError("observed, mask, and bases must share a dtype")
        if not observed.is_floating_point():
            raise ValueError("observed and mask must be floating point")
        if not isinstance(validate_values, bool):
            raise ValueError("validate_values must be bool")
        if validate_values:
            # These checks read device values and can synchronize CUDA.  Use once
            # for untrusted inputs, not inside a timed generated-data hot loop.
            if not torch.isfinite(observed).all() or not torch.isfinite(mask).all():
                raise ValueError("observed and mask must be finite")
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("mask must contain only 0 and 1")
            if not torch.all(mask.sum(dim=-1) >= 1):
                raise ValueError("every example must expose at least one coordinate")

    def _context(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Mask again at the model boundary so callers cannot leak values through
        # coordinates declared unobserved merely by supplying nonzero padding.
        return self.encoder(torch.cat((observed * mask, mask), dim=-1))

    def _residual_rms(self, residual: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        squared = residual.square().sum(dim=-1, keepdim=True)
        if self.config.residual_normalization == "observed":
            denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            while denominator.ndim < squared.ndim:
                denominator = denominator.unsqueeze(1)
        else:
            denominator = float(self.config.ambient_dim)
        return torch.sqrt(squared / denominator)

    def _candidate_features(
        self,
        context: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        bases: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return update features, decoded states, and observed-coordinate RMS."""
        if bases.ndim == 3 and coords.ndim == 3:
            states = torch.einsum("knd,bkd->bkn", bases, coords)
            residual = mask[:, None, :] * (states - observed[:, None, :])
            reduced = torch.einsum("knd,bkn->bkd", bases, residual)
            rms = self._residual_rms(residual, mask)
            expanded_context = context[:, None, :].expand(-1, coords.shape[1], -1)
        elif bases.ndim == 3 and coords.ndim == 2:
            states = torch.einsum("bnd,bd->bn", bases, coords)
            residual = mask * (states - observed)
            reduced = torch.einsum("bnd,bn->bd", bases, residual)
            rms = self._residual_rms(residual, mask)
            expanded_context = context
        else:
            raise ValueError("unexpected basis/coordinate ranks")
        features = torch.cat((expanded_context, coords, reduced, rms), dim=-1)
        return features, states, rms.squeeze(-1)

    def _candidate_step_all(
        self,
        context: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, _, _ = self._candidate_features(
            context, observed, mask, coords, self.bases
        )
        next_coords = coords + self.update(features)
        next_features, next_states, next_rms = self._candidate_features(
            context, observed, mask, next_coords, self.bases
        )
        halt_features = next_features if self.config.halt_timing == "after_update" else features
        halt_logits = self.halt(halt_features).squeeze(-1)
        return next_coords, next_states, halt_logits, next_rms

    def _candidate_step_selected(
        self,
        context: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        bases: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features, _, _ = self._candidate_features(
            context, observed, mask, coords, bases
        )
        next_coords = coords + self.update(features)
        next_features, next_states, next_rms = self._candidate_features(
            context, observed, mask, next_coords, bases
        )
        halt_features = next_features if self.config.halt_timing == "after_update" else features
        halt_logits = self.halt(halt_features).squeeze(-1)
        return next_coords, next_states, halt_logits, next_rms

    def forward_train(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        steps: int | None = None,
        validate_values: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Unroll every candidate trajectory for supervised training."""
        self._validate_inputs(observed, mask, validate_values=validate_values)
        limit = self.config.steps if steps is None else steps
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("steps must be a positive integer")

        context = self._context(observed, mask)
        router_logits = self.router(context)
        batch_size = observed.shape[0]
        coords = observed.new_zeros(
            (batch_size, self.config.num_structures, self.config.latent_dim)
        )
        states: list[torch.Tensor] = []
        halt_logits: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        for _ in range(limit):
            coords, state, halt_logit, residual = self._candidate_step_all(
                context, observed, mask, coords
            )
            states.append(state)
            halt_logits.append(halt_logit)
            residuals.append(residual)
        return {
            "router_logits": router_logits,
            "states": states,
            "halt_logits": halt_logits,
            "residuals": residuals,
        }

    @torch.no_grad()
    def infer(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        max_steps: int | None = None,
        halt_threshold: float = 0.80,
        min_steps: int = 1,
        execution_mode: ExecutionMode = "compact",
        fixed_depth: bool = False,
        forced_routes: torch.Tensor | None = None,
        record_events: bool = False,
        validate_values: bool = False,
    ) -> dict[str, torch.Tensor | list[dict] | int | str | bool]:
        """Hard-route inference with dense or active-sample execution.

        ``dense`` freezes logically halted samples but still sends the complete
        batch through the update network on every round.  ``compact`` gathers
        only active samples, so ``update_examples`` reflects real skipped sample
        evaluations. Dynamic indexing can still make compact execution slower.
        """
        self._validate_inputs(observed, mask, validate_values=validate_values)
        limit = self.config.steps if max_steps is None else max_steps
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("max_steps must be a positive integer")
        if isinstance(min_steps, bool) or not isinstance(min_steps, int) or not (1 <= min_steps <= limit):
            raise ValueError("min_steps must be an integer in [1,max_steps]")
        if execution_mode not in ("dense", "compact"):
            raise ValueError("execution_mode must be 'dense' or 'compact'")
        if not isinstance(fixed_depth, bool) or not isinstance(record_events, bool):
            raise ValueError("fixed_depth and record_events must be bool")
        if not math.isfinite(halt_threshold):
            raise ValueError("halt_threshold must be finite")
        if not fixed_depth and not (0.0 <= halt_threshold <= 1.0):
            raise ValueError("adaptive halt_threshold must lie in [0,1]")

        context = self._context(observed, mask)
        router_logits = self.router(context)
        route_probabilities = torch.softmax(router_logits, dim=-1)
        routes = router_logits.argmax(dim=-1)
        if forced_routes is not None:
            if forced_routes.shape != routes.shape or forced_routes.device != routes.device:
                raise ValueError("forced_routes must match route shape and device")
            if forced_routes.dtype != torch.long:
                raise ValueError("forced_routes must use torch.long")
            if validate_values and not torch.all(
                (forced_routes >= 0) & (forced_routes < self.config.num_structures)
            ):
                raise ValueError("forced route index out of range")
            routes = forced_routes

        selected_bases = self.bases[routes]
        batch_size = observed.shape[0]
        coords = observed.new_zeros((batch_size, self.config.latent_dim))
        steps_taken = torch.zeros(batch_size, dtype=torch.long, device=observed.device)
        events: list[dict] = []
        update_examples = 0
        rounds = 0

        if execution_mode == "dense":
            active = torch.ones(batch_size, dtype=torch.bool, device=observed.device)
            for step in range(1, limit + 1):
                next_coords_all, next_states_all, halt_logits_all, residuals_all = (
                    self._candidate_step_selected(
                        context, observed, mask, coords, selected_bases
                    )
                )
                active_before = active
                coords = torch.where(active_before[:, None], next_coords_all, coords)
                steps_taken = steps_taken + active_before.long()
                halt_probabilities = torch.sigmoid(halt_logits_all)
                newly_halted = (
                    torch.zeros_like(active_before)
                    if fixed_depth or step < min_steps
                    else active_before & (halt_probabilities >= halt_threshold)
                )
                update_examples += batch_size
                rounds = step
                if record_events:
                    events.append(
                        self._event(
                            step,
                            active_before,
                            newly_halted,
                            residuals_all,
                            halt_probabilities,
                            next_states_all,
                            routes,
                        )
                    )
                active = active_before & ~newly_halted
                # Human-facing event mode already synchronizes CUDA. In that
                # mode it is safe to stop once the whole batch has halted;
                # the timed dense path keeps a fixed loop and records no events.
                if record_events and not bool(active.any().item()):
                    break
            final_state = torch.einsum("bnd,bd->bn", selected_bases, coords)
        else:
            active_indices = torch.arange(batch_size, device=observed.device)
            for step in range(1, limit + 1):
                if active_indices.numel() == 0:
                    break
                c = context.index_select(0, active_indices)
                o = observed.index_select(0, active_indices)
                m = mask.index_select(0, active_indices)
                q = selected_bases.index_select(0, active_indices)
                a = coords.index_select(0, active_indices)
                next_coords, next_states, halt_logits, residuals = (
                    self._candidate_step_selected(c, o, m, a, q)
                )
                coords.index_copy_(0, active_indices, next_coords)
                steps_taken.index_add_(
                    0,
                    active_indices,
                    torch.ones_like(active_indices, dtype=torch.long),
                )
                halt_probabilities = torch.sigmoid(halt_logits)
                can_halt = (
                    torch.zeros_like(halt_probabilities, dtype=torch.bool)
                    if fixed_depth or step < min_steps
                    else halt_probabilities >= halt_threshold
                )
                newly_halted_full = torch.zeros(
                    batch_size, dtype=torch.bool, device=observed.device
                )
                newly_halted_full.index_copy_(0, active_indices, can_halt)
                active_before_full = torch.zeros_like(newly_halted_full)
                active_before_full[active_indices] = True
                update_examples += int(active_indices.numel())
                rounds = step
                if record_events:
                    residuals_full = observed.new_zeros(batch_size)
                    hprob_full = observed.new_zeros(batch_size)
                    states_full = observed.new_zeros((batch_size, self.config.ambient_dim))
                    residuals_full.index_copy_(0, active_indices, residuals)
                    hprob_full.index_copy_(0, active_indices, halt_probabilities)
                    states_full.index_copy_(0, active_indices, next_states)
                    events.append(
                        self._event(
                            step,
                            active_before_full,
                            newly_halted_full,
                            residuals_full,
                            hprob_full,
                            states_full,
                            routes,
                        )
                    )
                active_indices = active_indices[~can_halt]
            final_state = torch.einsum("bnd,bd->bn", selected_bases, coords)

        final_residual = mask * (final_state - observed)
        final_rms = self._residual_rms(final_residual, mask).squeeze(-1)
        return {
            "state": final_state,
            "coordinates": coords,
            "routes": routes,
            "router_logits": router_logits,
            "route_probabilities": route_probabilities,
            "steps_taken": steps_taken,
            "logical_updates": steps_taken.sum(),
            "update_examples": int(update_examples),
            "recurrent_rounds": int(rounds),
            "execution_mode": execution_mode,
            "fixed_depth": fixed_depth,
            "max_steps": int(limit),
            "min_steps": int(min_steps),
            "halt_threshold": float(halt_threshold),
            "events": events,
            "final_observed_residual_rms": final_rms,
        }

    @staticmethod
    def _event(
        step: int,
        active_before: torch.Tensor,
        newly_halted: torch.Tensor,
        residuals: torch.Tensor,
        halt_probabilities: torch.Tensor,
        states: torch.Tensor,
        routes: torch.Tensor,
    ) -> dict:
        """Materialize human-facing telemetry. This intentionally synchronizes."""
        active_count = int(active_before.sum().item())
        if active_count:
            mean_residual = float(residuals[active_before].mean().item())
            mean_halt = float(halt_probabilities[active_before].mean().item())
        else:
            mean_residual = 0.0
            mean_halt = 1.0
        event: dict[str, object] = {
            "step": int(step),
            "active_before": active_count,
            "newly_halted": int(newly_halted.sum().item()),
            "mean_observed_residual": mean_residual,
            "mean_halt_probability": mean_halt,
        }
        if states.shape[0] == 1:
            event["state_after"] = states[0].detach().cpu().tolist()
            event["route_index"] = int(routes[0].item())
            event["observed_residual"] = float(residuals[0].item())
            event["halt_probability"] = float(halt_probabilities[0].item())
            event["halted_after"] = bool(newly_halted[0].item())
        return event


def future_regret_targets(
    states: list[torch.Tensor],
    truth: torch.Tensor,
    *,
    tolerance: float,
) -> torch.Tensor:
    """Return [T,B,K] ready-to-stop targets from future synthetic regret.

    A step is ready when no later available step improves truth MSE by more than
    ``tolerance``. The teacher is training-only and uses hidden synthetic truth.
    """
    if not states:
        raise ValueError("states must be nonempty")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    stacked = torch.stack(states, dim=0)
    if stacked.ndim != 4 or truth.ndim != 2:
        raise ValueError("states must be [T,B,K,N] and truth [B,N]")
    mse = (stacked - truth[None, :, None, :]).square().mean(dim=-1)
    future_best = torch.empty_like(mse)
    running = mse[-1]
    future_best[-1] = running
    for index in range(mse.shape[0] - 2, -1, -1):
        running = torch.minimum(running, mse[index + 1])
        future_best[index] = running
    improvement = mse - future_best
    targets = (improvement <= tolerance).to(mse.dtype)
    targets[-1] = 1.0
    return targets


def training_loss(
    outputs: dict,
    truth: torch.Tensor,
    labels: torch.Tensor,
    *,
    router_weight: float = 0.25,
    halt_weight: float = 0.10,
    future_regret_tolerance: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Training objective with all-candidate future-regret halt supervision."""
    states = outputs.get("states")
    router_logits = outputs.get("router_logits")
    halt_logits = outputs.get("halt_logits")
    if not isinstance(states, list) or not states:
        raise ValueError("outputs must contain nonempty state list")
    if not isinstance(halt_logits, list) or len(halt_logits) != len(states):
        raise ValueError("halt logits must align with state list")
    if not torch.is_tensor(router_logits):
        raise ValueError("router_logits missing")
    if truth.ndim != 2 or labels.ndim != 1 or labels.shape[0] != truth.shape[0]:
        raise ValueError("truth/label shapes are invalid")
    if not math.isfinite(router_weight) or router_weight < 0:
        raise ValueError("router_weight must be finite and nonnegative")
    if not math.isfinite(halt_weight) or halt_weight < 0:
        raise ValueError("halt_weight must be finite and nonnegative")

    batch_size = truth.shape[0]
    index = torch.arange(batch_size, device=truth.device)
    final_all = states[-1]
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long")
    true_state = final_all[index, labels]
    route_probabilities = F.softmax(router_logits, dim=-1)
    mixed_state = torch.einsum("bk,bkn->bn", route_probabilities, final_all)

    recurrent = F.mse_loss(true_state, truth)
    mixed = F.mse_loss(mixed_state, truth)
    router = F.cross_entropy(router_logits, labels)
    targets = future_regret_targets(
        states, truth, tolerance=future_regret_tolerance
    )
    halt_tensor = torch.stack(halt_logits, dim=0)
    halt = F.binary_cross_entropy_with_logits(halt_tensor, targets)
    total = recurrent + 0.5 * mixed + router_weight * router + halt_weight * halt
    metrics = {
        "loss": total.detach(),
        "true_route_final_mse": recurrent.detach(),
        "mixed_final_mse": mixed.detach(),
        "router_ce": router.detach(),
        "halt_bce": halt.detach(),
    }
    return total, metrics
