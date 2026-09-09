"""Uncertainty-aware structured recurrence.

Neural v2 keeps one latent state per candidate structure until the evidence is
strong enough to commit.  A policy can therefore make three explicit decisions:

``continue``
    update every still-plausible structural hypothesis again;
``commit``
    choose one candidate and return its constrained state; or
``abstain``
    decline a single-structure claim and return a probability-weighted
    provisional mixture.

The model updates route evidence after every recurrent step.  This is a direct
response to neural v1's dominant failure mode: most aggregate error came from
continuing inside a structure chosen incorrectly at the first step.

The module deliberately separates a mathematical guarantee from a learned
judgment.  Every candidate state is parameterized as ``z_k = Q_k a_k`` and is
therefore in candidate subspace ``im(Q_k)`` up to floating-point error.  The
posterior probabilities, commitment decision, and semantic interpretation of
hidden features are learned or calibrated and are not certified by that fact.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

ExecutionMode = Literal["dense", "compact"]
DecisionName = Literal["continue", "commit", "abstain"]


@dataclass(frozen=True)
class V2Config:
    """Architecture dimensions for multi-hypothesis recurrence."""

    ambient_dim: int = 5
    latent_dim: int = 2
    num_structures: int = 2
    hidden_dim: int = 64
    candidate_embedding_dim: int = 8
    steps: int = 8

    def __post_init__(self) -> None:
        for name in (
            "ambient_dim",
            "latent_dim",
            "num_structures",
            "hidden_dim",
            "candidate_embedding_dim",
            "steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_structures < 2:
            raise ValueError("v2 requires at least two candidate structures")


@dataclass(frozen=True)
class DecisionPolicy:
    """Readable commitment/abstention rule used only at inference.

    Early commitment requires both ``commit_probability`` and
    ``commit_margin`` after ``min_steps``.  At the final available step, an
    unresolved example commits only if the weaker final thresholds pass;
    otherwise it abstains.
    """

    commit_probability: float = 0.85
    commit_margin: float = 0.20
    final_probability: float = 0.60
    final_margin: float = 0.05
    min_steps: int = 2

    def __post_init__(self) -> None:
        for name in ("commit_probability", "commit_margin", "final_probability", "final_margin"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be finite and in [0,1]")
            object.__setattr__(self, name, value)
        if isinstance(self.min_steps, bool) or not isinstance(self.min_steps, int) or self.min_steps < 1:
            raise ValueError("min_steps must be a positive integer")
        if self.final_probability > self.commit_probability:
            raise ValueError("final_probability cannot exceed commit_probability")
        if self.final_margin > self.commit_margin:
            raise ValueError("final_margin cannot exceed commit_margin")

    def as_dict(self) -> dict[str, float | int]:
        return {
            "commit_probability": self.commit_probability,
            "commit_margin": self.commit_margin,
            "final_probability": self.final_probability,
            "final_margin": self.final_margin,
            "min_steps": self.min_steps,
        }


@dataclass(frozen=True)
class CalibrationConfig:
    """Cost-aware rule for choosing a policy on a calibration split."""

    step_cost: float = 0.002
    abstain_cost: float = 0.02
    wrong_commit_cost: float = 0.05
    min_coverage: float = 0.75

    def __post_init__(self) -> None:
        for name in ("step_cost", "abstain_cost", "wrong_commit_cost"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        coverage = float(self.min_coverage)
        if not math.isfinite(coverage) or not (0.0 <= coverage <= 1.0):
            raise ValueError("min_coverage must be finite and in [0,1]")
        object.__setattr__(self, "min_coverage", coverage)

    def as_dict(self) -> dict[str, float]:
        return {
            "step_cost": self.step_cost,
            "abstain_cost": self.abstain_cost,
            "wrong_commit_cost": self.wrong_commit_cost,
            "min_coverage": self.min_coverage,
        }


def _validate_bases(bases: torch.Tensor, config: V2Config) -> None:
    if bases.ndim != 3:
        raise ValueError("bases must have shape [K,N,D]")
    if not bases.is_floating_point() or not torch.isfinite(bases).all():
        raise ValueError("bases must be finite floating-point tensors")
    if tuple(bases.shape) != (
        config.num_structures,
        config.ambient_dim,
        config.latent_dim,
    ):
        raise ValueError("basis dimensions do not match config")
    gram = torch.einsum("knd,kne->kde", bases.double(), bases.double())
    identity = torch.eye(config.latent_dim, dtype=torch.float64, device=bases.device)
    if not torch.allclose(gram, identity.expand_as(gram), atol=1e-6, rtol=1e-6):
        raise ValueError("candidate bases must have orthonormal columns")


def _top_statistics(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape [B,K] with K >= 2")
    top_values, top_indices = torch.topk(probabilities, k=2, dim=-1)
    return top_values[:, 0], top_values[:, 0] - top_values[:, 1], probabilities.argmax(dim=-1)


class MultiHypothesisRecurrentNet(nn.Module):
    """Update every candidate state and revise route evidence at each step."""

    def __init__(self, bases: torch.Tensor, config: V2Config | None = None) -> None:
        super().__init__()
        if bases.ndim != 3:
            raise ValueError("bases must have shape [K,N,D]")
        inferred = V2Config(
            ambient_dim=int(bases.shape[1]),
            latent_dim=int(bases.shape[2]),
            num_structures=int(bases.shape[0]),
        )
        cfg = config or inferred
        _validate_bases(bases, cfg)
        self.config = cfg
        self.register_buffer("bases", bases.float().contiguous().clone())

        self.encoder = nn.Sequential(
            nn.Linear(2 * cfg.ambient_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        self.candidate_embedding = nn.Embedding(
            cfg.num_structures, cfg.candidate_embedding_dim
        )
        self.prior_router = nn.Linear(cfg.hidden_dim, cfg.num_structures)

        update_width = (
            cfg.hidden_dim
            + cfg.candidate_embedding_dim
            + 2 * cfg.latent_dim
            + 2
        )
        self.update = nn.Sequential(
            nn.Linear(update_width, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
        )

        evidence_width = (
            cfg.hidden_dim
            + cfg.candidate_embedding_dim
            + 2 * cfg.latent_dim
            + 4
        )
        self.evidence = nn.Sequential(
            nn.Linear(evidence_width, max(8, cfg.hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(8, cfg.hidden_dim // 2), 1),
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
            if not torch.isfinite(observed).all() or not torch.isfinite(mask).all():
                raise ValueError("observed and mask must be finite")
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("mask must contain only 0 and 1")
            if not torch.all(mask.sum(dim=-1) >= 1):
                raise ValueError("every example must expose at least one coordinate")

    def _context(self, observed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat((observed * mask, mask), dim=-1))

    def _candidate_geometry(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        bases: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode candidates and return reduced and RMS observed residuals."""
        if coords.ndim != 3 or bases.ndim not in (3, 4):
            raise ValueError(
                "coords must be [B,K,D]; bases must be [K,N,D] or [B,K,N,D]"
            )
        if bases.ndim == 3 and bases.shape[0] == self.config.num_structures and bases.shape[1:] == (
            self.config.ambient_dim,
            self.config.latent_dim,
        ):
            states = torch.einsum("knd,bkd->bkn", bases, coords)
            reduced = torch.einsum(
                "knd,bkn->bkd",
                bases,
                mask[:, None, :] * (states - observed[:, None, :]),
            )
        elif bases.ndim == 4 and bases.shape == (
            coords.shape[0],
            self.config.num_structures,
            self.config.ambient_dim,
            self.config.latent_dim,
        ):
            states = torch.einsum("bknd,bkd->bkn", bases, coords)
            reduced = torch.einsum(
                "bknd,bkn->bkd",
                bases,
                mask[:, None, :] * (states - observed[:, None, :]),
            )
        else:
            raise ValueError("unexpected per-candidate basis shape")
        residual = mask[:, None, :] * (states - observed[:, None, :])
        denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        rms = torch.linalg.vector_norm(residual, dim=-1) / denominator.sqrt()
        return states, reduced, rms

    def _expanded_candidate_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        indices = torch.arange(self.config.num_structures, device=device)
        return self.candidate_embedding(indices)[None].expand(batch_size, -1, -1)

    def _step(
        self,
        context: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        prior_logits: torch.Tensor,
        current_logits: torch.Tensor,
        bases: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = observed.shape[0]
        active_bases = self.bases if bases is None else bases
        states, reduced, rms = self._candidate_geometry(
            observed, mask, coords, active_bases
        )
        probabilities = torch.softmax(current_logits, dim=-1)
        candidate_embedding = self._expanded_candidate_embedding(batch_size, observed.device)
        expanded_context = context[:, None, :].expand(-1, self.config.num_structures, -1)
        update_features = torch.cat(
            (
                expanded_context,
                candidate_embedding,
                coords,
                reduced,
                rms[..., None],
                probabilities[..., None],
            ),
            dim=-1,
        )
        next_coords = coords + self.update(update_features)
        next_states, next_reduced, next_rms = self._candidate_geometry(
            observed, mask, next_coords, active_bases
        )
        progress = rms - next_rms
        prior_per_candidate = prior_logits[..., None]
        evidence_features = torch.cat(
            (
                expanded_context,
                candidate_embedding,
                next_coords,
                next_reduced,
                next_rms[..., None],
                progress[..., None],
                probabilities[..., None],
                prior_per_candidate,
            ),
            dim=-1,
        )
        evidence_delta = self.evidence(evidence_features).squeeze(-1)
        next_logits = prior_logits + evidence_delta
        return next_coords, next_states, next_logits, next_rms, progress

    def rollout(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        steps: int | None = None,
        validate_values: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Return every candidate trajectory for training and calibration."""
        self._validate_inputs(observed, mask, validate_values=validate_values)
        limit = self.config.steps if steps is None else steps
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("steps must be a positive integer")
        if limit > self.config.steps:
            raise ValueError("steps cannot exceed the configured recurrent depth")

        context = self._context(observed, mask)
        prior_logits = self.prior_router(context)
        current_logits = prior_logits
        coords = observed.new_zeros(
            observed.shape[0], self.config.num_structures, self.config.latent_dim
        )
        states: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        progresses: list[torch.Tensor] = []
        coordinates: list[torch.Tensor] = []
        for _ in range(limit):
            coords, state, current_logits, rms, progress = self._step(
                context,
                observed,
                mask,
                coords,
                prior_logits,
                current_logits,
            )
            coordinates.append(coords)
            states.append(state)
            logits.append(current_logits)
            residuals.append(rms)
            progresses.append(progress)
        stacked_logits = torch.stack(logits, dim=0)
        return {
            "prior_logits": prior_logits,
            "coordinates": torch.stack(coordinates, dim=0),
            "states": torch.stack(states, dim=0),
            "route_logits": stacked_logits,
            "route_probabilities": torch.softmax(stacked_logits, dim=-1),
            "residuals": torch.stack(residuals, dim=0),
            "progress": torch.stack(progresses, dim=0),
        }

    def forward_train(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        steps: int | None = None,
        validate_values: bool = False,
    ) -> dict[str, torch.Tensor]:
        return self.rollout(
            observed, mask, steps=steps, validate_values=validate_values
        )

    @torch.no_grad()
    def infer(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        policy: DecisionPolicy,
        max_steps: int | None = None,
        execution_mode: ExecutionMode = "compact",
        fixed_depth: bool = False,
        force_commit: bool = False,
        record_events: bool = False,
        validate_values: bool = False,
    ) -> dict[str, object]:
        """Run commit/continue/abstain inference.

        ``dense`` computes every candidate for every example through every round,
        then applies the policy to the retained trajectory. ``compact`` removes
        examples after commitment or abstention.  The latter skips candidate
        updates but can still be slower because of dynamic indexing.
        """
        self._validate_inputs(observed, mask, validate_values=validate_values)
        if not isinstance(policy, DecisionPolicy):
            raise TypeError("policy must be DecisionPolicy")
        limit = self.config.steps if max_steps is None else max_steps
        if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= self.config.steps):
            raise ValueError("max_steps must be in [1, configured steps]")
        if policy.min_steps > limit:
            raise ValueError("policy min_steps cannot exceed max_steps")
        if execution_mode not in ("dense", "compact"):
            raise ValueError("execution_mode must be 'dense' or 'compact'")
        if not isinstance(fixed_depth, bool) or not isinstance(force_commit, bool):
            raise ValueError("fixed_depth and force_commit must be bool")
        if record_events and observed.shape[0] != 1:
            raise ValueError("event recording currently requires batch size 1")

        if execution_mode == "dense":
            trajectory = self.rollout(observed, mask, steps=limit)
            result = apply_policy_to_trajectory(
                trajectory,
                policy,
                force_commit=force_commit,
                fixed_depth=fixed_depth,
            )
            result.update(
                {
                    "execution_mode": "dense",
                    "candidate_update_examples": int(
                        observed.shape[0] * self.config.num_structures * limit
                    ),
                    "recurrent_rounds": int(limit),
                    "max_steps": int(limit),
                }
            )
            if record_events:
                result["events"] = _events_from_trajectory(
                    trajectory,
                    result,
                    policy,
                )
            else:
                result["events"] = []
            return result

        return self._infer_compact(
            observed,
            mask,
            policy=policy,
            limit=limit,
            fixed_depth=fixed_depth,
            force_commit=force_commit,
            record_events=record_events,
        )

    @torch.no_grad()
    def _infer_compact(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        policy: DecisionPolicy,
        limit: int,
        fixed_depth: bool,
        force_commit: bool,
        record_events: bool,
    ) -> dict[str, object]:
        batch_size = observed.shape[0]
        k = self.config.num_structures
        context_all = self._context(observed, mask)
        prior_all = self.prior_router(context_all)
        logits_all = prior_all.clone()
        coords_all = observed.new_zeros(batch_size, k, self.config.latent_dim)
        active = torch.arange(batch_size, device=observed.device)

        final_candidate_states = observed.new_zeros(batch_size, k, self.config.ambient_dim)
        final_probabilities = observed.new_zeros(batch_size, k)
        final_residuals = observed.new_zeros(batch_size, k)
        final_top_routes = torch.full(
            (batch_size,), -1, dtype=torch.long, device=observed.device
        )
        committed = torch.zeros(batch_size, dtype=torch.bool, device=observed.device)
        decision_steps = torch.zeros(batch_size, dtype=torch.long, device=observed.device)
        reason_codes = torch.full(
            (batch_size,), -1, dtype=torch.long, device=observed.device
        )
        revision_counts = torch.zeros(batch_size, dtype=torch.long, device=observed.device)
        previous_top = torch.full_like(final_top_routes, -1)
        candidate_update_examples = 0
        rounds = 0
        events: list[dict[str, object]] = []

        for step in range(1, limit + 1):
            if active.numel() == 0:
                break
            c = context_all.index_select(0, active)
            o = observed.index_select(0, active)
            m = mask.index_select(0, active)
            a = coords_all.index_select(0, active)
            prior = prior_all.index_select(0, active)
            current = logits_all.index_select(0, active)
            next_coords, states, next_logits, residuals, _ = self._step(
                c, o, m, a, prior, current
            )
            coords_all.index_copy_(0, active, next_coords)
            logits_all.index_copy_(0, active, next_logits)
            probabilities = torch.softmax(next_logits, dim=-1)
            top_probability, margin, top_route = _top_statistics(probabilities)

            old_top = previous_top.index_select(0, active)
            revisions = (old_top >= 0) & (old_top != top_route)
            revision_counts.index_add_(0, active, revisions.long())
            previous_top.index_copy_(0, active, top_route)

            if fixed_depth and step < limit:
                decide_commit = torch.zeros_like(top_probability, dtype=torch.bool)
                decide_abstain = torch.zeros_like(decide_commit)
            else:
                early = (
                    (step >= policy.min_steps)
                    & (top_probability >= policy.commit_probability)
                    & (margin >= policy.commit_margin)
                )
                if step < limit:
                    decide_commit = early
                    decide_abstain = torch.zeros_like(decide_commit)
                else:
                    final_ok = (
                        (top_probability >= policy.final_probability)
                        & (margin >= policy.final_margin)
                    )
                    decide_commit = (
                        torch.ones_like(final_ok)
                        if force_commit
                        else (early | final_ok)
                    )
                    decide_abstain = ~decide_commit

            decided = decide_commit | decide_abstain
            if decided.any():
                decided_indices = active[decided]
                final_candidate_states.index_copy_(0, decided_indices, states[decided])
                final_probabilities.index_copy_(0, decided_indices, probabilities[decided])
                final_residuals.index_copy_(0, decided_indices, residuals[decided])
                final_top_routes.index_copy_(0, decided_indices, top_route[decided])
                committed.index_copy_(0, decided_indices, decide_commit[decided])
                decision_steps.index_fill_(0, decided_indices, step)

                reason = torch.full_like(top_route[decided], 3)
                if force_commit and step == limit:
                    reason.fill_(2)  # forced final commit
                else:
                    early_decided = (
                        (step < limit)
                        & (top_probability[decided] >= policy.commit_probability)
                        & (margin[decided] >= policy.commit_margin)
                        & (step >= policy.min_steps)
                    )
                    reason = torch.where(
                        decide_abstain[decided],
                        torch.full_like(reason, 3),
                        torch.where(
                            early_decided,
                            torch.zeros_like(reason),
                            torch.ones_like(reason),
                        ),
                    )
                reason_codes.index_copy_(0, decided_indices, reason)

            candidate_update_examples += int(active.numel()) * k
            rounds = step
            if record_events:
                decision = "continue"
                if bool(decide_commit[0].item()):
                    decision = "commit"
                elif bool(decide_abstain[0].item()):
                    decision = "abstain"
                events.append(
                    _single_event(
                        step,
                        states[0],
                        probabilities[0],
                        residuals[0],
                        top_route[0],
                        top_probability[0],
                        margin[0],
                        decision,
                    )
                )
            active = active[~decided]

        if (decision_steps == 0).any():
            raise RuntimeError("compact inference ended with undecided examples")

        index = torch.arange(batch_size, device=observed.device)
        selected_states = final_candidate_states[index, final_top_routes]
        mixture_states = torch.einsum(
            "bk,bkn->bn", final_probabilities, final_candidate_states
        )
        final_state = torch.where(committed[:, None], selected_states, mixture_states)
        routes = torch.where(
            committed,
            final_top_routes,
            torch.full_like(final_top_routes, -1),
        )
        return {
            "state": final_state,
            "candidate_states": final_candidate_states,
            "route_probabilities": final_probabilities,
            "candidate_residuals": final_residuals,
            "top_routes": final_top_routes,
            "routes": routes,
            "committed": committed,
            "abstained": ~committed,
            "decision_steps": decision_steps,
            "decision_reason_codes": reason_codes,
            "route_revisions": revision_counts,
            "mean_logical_steps": decision_steps.float().mean(),
            "logical_hypothesis_updates": decision_steps.sum() * k,
            "candidate_update_examples": candidate_update_examples,
            "recurrent_rounds": rounds,
            "execution_mode": "compact",
            "max_steps": limit,
            "policy": policy.as_dict(),
            "fixed_depth": fixed_depth,
            "force_commit": force_commit,
            "events": events,
        }


class DirectMultiHypothesisNet(nn.Module):
    """One-pass structured control using the same candidate library.

    The training harness can choose its hidden width to approximately match the
    recurrent model's parameter count. Exact counts are always reported because
    parameter matching alone does not equalize sequential compute or inductive bias.
    """

    def __init__(self, bases: torch.Tensor, config: V2Config) -> None:
        super().__init__()
        _validate_bases(bases, config)
        self.config = config
        self.register_buffer("bases", bases.float().contiguous().clone())
        self.encoder = nn.Sequential(
            nn.Linear(2 * config.ambient_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.candidate_embedding = nn.Embedding(
            config.num_structures, config.candidate_embedding_dim
        )
        candidate_width = config.hidden_dim + config.candidate_embedding_dim
        self.coordinate_head = nn.Sequential(
            nn.Linear(candidate_width, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        evidence_width = candidate_width + config.latent_dim + 1
        self.evidence = nn.Sequential(
            nn.Linear(evidence_width, max(8, config.hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(8, config.hidden_dim // 2), 1),
        )

    def forward(self, observed: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if observed.ndim != 2 or mask.shape != observed.shape:
            raise ValueError("observed and mask must have matching [B,N] shapes")
        context = self.encoder(torch.cat((observed * mask, mask), dim=-1))
        embeddings = self.candidate_embedding(
            torch.arange(self.config.num_structures, device=observed.device)
        )[None].expand(observed.shape[0], -1, -1)
        expanded_context = context[:, None, :].expand(-1, self.config.num_structures, -1)
        features = torch.cat((expanded_context, embeddings), dim=-1)
        coordinates = self.coordinate_head(features)
        states = torch.einsum("knd,bkd->bkn", self.bases, coordinates)
        residual = mask[:, None, :] * (states - observed[:, None, :])
        denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        rms = torch.linalg.vector_norm(residual, dim=-1) / denominator.sqrt()
        logits = self.evidence(
            torch.cat((features, coordinates, rms[..., None]), dim=-1)
        ).squeeze(-1)
        probabilities = torch.softmax(logits, dim=-1)
        return {
            "states": states,
            "coordinates": coordinates,
            "route_logits": logits,
            "route_probabilities": probabilities,
            "residuals": rms,
            "mixture_state": torch.einsum("bk,bkn->bn", probabilities, states),
        }


class UntiedMultiHypothesisNet(MultiHypothesisRecurrentNet):
    """Depth-specific update/evidence control with otherwise matching semantics."""

    def __init__(self, bases: torch.Tensor, config: V2Config) -> None:
        super().__init__(bases, config)
        update_template = self.update
        evidence_template = self.evidence
        import copy

        self.updates = nn.ModuleList(
            copy.deepcopy(update_template) for _ in range(config.steps)
        )
        self.evidences = nn.ModuleList(
            copy.deepcopy(evidence_template) for _ in range(config.steps)
        )
        del self.update
        del self.evidence

    def _step_at_depth(
        self,
        depth: int,
        context: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
        prior_logits: torch.Tensor,
        current_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = observed.shape[0]
        states, reduced, rms = self._candidate_geometry(observed, mask, coords, self.bases)
        probabilities = torch.softmax(current_logits, dim=-1)
        candidate_embedding = self._expanded_candidate_embedding(batch_size, observed.device)
        expanded_context = context[:, None, :].expand(-1, self.config.num_structures, -1)
        update_features = torch.cat(
            (
                expanded_context,
                candidate_embedding,
                coords,
                reduced,
                rms[..., None],
                probabilities[..., None],
            ),
            dim=-1,
        )
        next_coords = coords + self.updates[depth](update_features)
        next_states, next_reduced, next_rms = self._candidate_geometry(
            observed, mask, next_coords, self.bases
        )
        progress = rms - next_rms
        evidence_features = torch.cat(
            (
                expanded_context,
                candidate_embedding,
                next_coords,
                next_reduced,
                next_rms[..., None],
                progress[..., None],
                probabilities[..., None],
                prior_logits[..., None],
            ),
            dim=-1,
        )
        next_logits = prior_logits + self.evidences[depth](evidence_features).squeeze(-1)
        return next_coords, next_states, next_logits, next_rms, progress

    def rollout(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        steps: int | None = None,
        validate_values: bool = False,
    ) -> dict[str, torch.Tensor]:
        self._validate_inputs(observed, mask, validate_values=validate_values)
        limit = self.config.steps if steps is None else steps
        if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= self.config.steps):
            raise ValueError("steps must be in [1, configured steps]")
        context = self._context(observed, mask)
        prior_logits = self.prior_router(context)
        current_logits = prior_logits
        coords = observed.new_zeros(
            observed.shape[0], self.config.num_structures, self.config.latent_dim
        )
        states: list[torch.Tensor] = []
        logits: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        progresses: list[torch.Tensor] = []
        coordinates: list[torch.Tensor] = []
        for depth in range(limit):
            coords, state, current_logits, rms, progress = self._step_at_depth(
                depth, context, observed, mask, coords, prior_logits, current_logits
            )
            coordinates.append(coords)
            states.append(state)
            logits.append(current_logits)
            residuals.append(rms)
            progresses.append(progress)
        stacked_logits = torch.stack(logits, dim=0)
        return {
            "prior_logits": prior_logits,
            "coordinates": torch.stack(coordinates, dim=0),
            "states": torch.stack(states, dim=0),
            "route_logits": stacked_logits,
            "route_probabilities": torch.softmax(stacked_logits, dim=-1),
            "residuals": torch.stack(residuals, dim=0),
            "progress": torch.stack(progresses, dim=0),
        }

    def infer(self, *args: object, **kwargs: object) -> dict[str, object]:  # type: ignore[override]
        # Untied control currently uses dense rollout.  That is sufficient for
        # quality comparison; it is not advertised as adaptive execution.
        execution_mode = kwargs.get("execution_mode", "dense")
        if execution_mode != "dense":
            raise ValueError("untied control supports dense inference only")
        return super().infer(*args, **kwargs)  # type: ignore[arg-type]

    def _step(self, *args: object, **kwargs: object):  # type: ignore[override]
        raise RuntimeError("untied control must use rollout, not shared _step")


class AmbientRecurrentControl(nn.Module):
    """Unstructured recurrent control over the full ambient vector.

    It receives the same masked observation and reuses one update network, but it
    has no candidate basis, route, or structural certificate.  Its hidden width
    is chosen separately so total trainable parameters can be reported near the
    v2 model's count.
    """

    def __init__(self, ambient_dim: int, hidden_dim: int, steps: int) -> None:
        super().__init__()
        for name, value in (
            ("ambient_dim", ambient_dim),
            ("hidden_dim", hidden_dim),
            ("steps", steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.ambient_dim = ambient_dim
        self.hidden_dim = hidden_dim
        self.steps = steps
        self.encoder = nn.Sequential(
            nn.Linear(2 * ambient_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        update_width = hidden_dim + 2 * ambient_dim + 1
        self.update = nn.Sequential(
            nn.Linear(update_width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ambient_dim),
        )

    def rollout(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
        *,
        steps: int | None = None,
    ) -> torch.Tensor:
        if observed.ndim != 2 or mask.shape != observed.shape:
            raise ValueError("observed and mask must have matching [B,N] shapes")
        if observed.shape[1] != self.ambient_dim:
            raise ValueError("observation width does not match ambient control")
        limit = self.steps if steps is None else steps
        if isinstance(limit, bool) or not isinstance(limit, int) or not (1 <= limit <= self.steps):
            raise ValueError("steps must be in [1, configured steps]")
        context = self.encoder(torch.cat((observed * mask, mask), dim=-1))
        state = observed.new_zeros(observed.shape)
        states: list[torch.Tensor] = []
        denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        for _ in range(limit):
            residual = mask * (state - observed)
            rms = torch.linalg.vector_norm(residual, dim=-1, keepdim=True) / denominator.sqrt()
            state = state + self.update(torch.cat((context, state, residual, rms), dim=-1))
            states.append(state)
        return torch.stack(states, dim=0)


def ambient_training_loss(
    states: torch.Tensor,
    truth: torch.Tensor,
    *,
    trajectory_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if states.ndim != 3 or truth.ndim != 2 or states.shape[1:] != truth.shape:
        raise ValueError("ambient states must be [T,B,N] and truth [B,N]")
    if not math.isfinite(trajectory_weight) or trajectory_weight < 0.0:
        raise ValueError("trajectory_weight must be finite and nonnegative")
    mse_by_step = (states - truth[None]).square().mean(dim=(1, 2))
    final = mse_by_step[-1]
    trajectory = mse_by_step.mean()
    total = final + trajectory_weight * trajectory
    return total, {
        "loss": total.detach(),
        "true_route_final_mse": final.detach(),
        "mixture_final_mse": final.detach(),
        "route_ce": final.detach().new_zeros(()),
    }

def v2_training_loss(
    outputs: dict[str, torch.Tensor],
    truth: torch.Tensor,
    labels: torch.Tensor,
    *,
    router_weight: float = 0.25,
    mixture_weight: float = 0.50,
    trajectory_weight: float = 0.10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Deeply supervise candidate states and route evidence."""
    states = outputs.get("states")
    logits = outputs.get("route_logits")
    probabilities = outputs.get("route_probabilities")
    if not torch.is_tensor(states) or states.ndim != 4:
        raise ValueError("states must have shape [T,B,K,N]")
    if not torch.is_tensor(logits) or logits.shape[:3] != states.shape[:3]:
        raise ValueError("route logits must align with states")
    if not torch.is_tensor(probabilities) or probabilities.shape != logits.shape:
        raise ValueError("route probabilities must align with logits")
    if truth.ndim != 2 or labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError("truth and labels have invalid shapes or dtypes")
    if truth.shape[0] != states.shape[1] or labels.shape[0] != truth.shape[0]:
        raise ValueError("batch dimensions do not align")
    for name, value in (
        ("router_weight", router_weight),
        ("mixture_weight", mixture_weight),
        ("trajectory_weight", trajectory_weight),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")

    batch = truth.shape[0]
    index = torch.arange(batch, device=truth.device)
    true_states = states[:, index, labels]
    true_mse_by_step = (true_states - truth[None]).square().mean(dim=(1, 2))
    final_true = true_mse_by_step[-1]
    mixture_states = torch.einsum("tbk,tbkn->tbn", probabilities, states)
    mixture_mse_by_step = (mixture_states - truth[None]).square().mean(dim=(1, 2))
    final_mixture = mixture_mse_by_step[-1]
    route_ce = torch.stack(
        [F.cross_entropy(logits[step], labels) for step in range(logits.shape[0])]
    ).mean()
    trajectory = true_mse_by_step.mean()
    total = (
        final_true
        + mixture_weight * final_mixture
        + router_weight * route_ce
        + trajectory_weight * trajectory
    )
    metrics = {
        "loss": total.detach(),
        "true_route_final_mse": final_true.detach(),
        "mixture_final_mse": final_mixture.detach(),
        "route_ce": route_ce.detach(),
        "trajectory_true_mse": trajectory.detach(),
    }
    return total, metrics


def direct_training_loss(
    outputs: dict[str, torch.Tensor],
    truth: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    states = outputs["states"]
    logits = outputs["route_logits"]
    probabilities = outputs["route_probabilities"]
    index = torch.arange(truth.shape[0], device=truth.device)
    true_state = states[index, labels]
    true_mse = F.mse_loss(true_state, truth)
    mixture = torch.einsum("bk,bkn->bn", probabilities, states)
    mixture_mse = F.mse_loss(mixture, truth)
    route_ce = F.cross_entropy(logits, labels)
    total = true_mse + 0.5 * mixture_mse + 0.25 * route_ce
    return total, {
        "loss": total.detach(),
        "true_route_final_mse": true_mse.detach(),
        "mixture_final_mse": mixture_mse.detach(),
        "route_ce": route_ce.detach(),
    }


def apply_policy_to_trajectory(
    trajectory: dict[str, torch.Tensor],
    policy: DecisionPolicy,
    *,
    force_commit: bool = False,
    fixed_depth: bool = False,
) -> dict[str, object]:
    """Apply one policy to a retained dense trajectory without rerunning a model."""
    states = trajectory.get("states")
    probabilities = trajectory.get("route_probabilities")
    residuals = trajectory.get("residuals")
    if not torch.is_tensor(states) or states.ndim != 4:
        raise ValueError("trajectory states must be [T,B,K,N]")
    if not torch.is_tensor(probabilities) or probabilities.shape != states.shape[:3]:
        raise ValueError("trajectory probabilities do not align")
    if not torch.is_tensor(residuals) or residuals.shape != states.shape[:3]:
        raise ValueError("trajectory residuals do not align")
    if policy.min_steps > states.shape[0]:
        raise ValueError("policy min_steps exceeds trajectory length")

    steps, batch, structures, _ = states.shape
    top_values, top_indices = torch.topk(probabilities, k=2, dim=-1)
    top_probability = top_values[..., 0]
    margin = top_values[..., 0] - top_values[..., 1]
    top_route = probabilities.argmax(dim=-1)

    early = (
        (top_probability >= policy.commit_probability)
        & (margin >= policy.commit_margin)
    )
    step_numbers = torch.arange(1, steps + 1, device=states.device)[:, None]
    early &= step_numbers >= policy.min_steps
    if fixed_depth:
        early[:] = False
    before_final = early.clone()
    before_final[-1] = False
    any_early = before_final.any(dim=0)
    first_early = before_final.to(torch.int64).argmax(dim=0)
    decision_index = torch.where(
        any_early,
        first_early,
        torch.full_like(first_early, steps - 1),
    )

    final_ok = (
        (top_probability[-1] >= policy.final_probability)
        & (margin[-1] >= policy.final_margin)
    )
    committed = any_early | final_ok
    if force_commit:
        committed = torch.ones_like(committed)
    batch_index = torch.arange(batch, device=states.device)
    candidate_states = states[decision_index, batch_index]
    final_probabilities = probabilities[decision_index, batch_index]
    final_residuals = residuals[decision_index, batch_index]
    final_top_routes = top_route[decision_index, batch_index]
    selected = candidate_states[batch_index, final_top_routes]
    mixture = torch.einsum("bk,bkn->bn", final_probabilities, candidate_states)
    final_state = torch.where(committed[:, None], selected, mixture)
    routes = torch.where(
        committed, final_top_routes, torch.full_like(final_top_routes, -1)
    )

    revisions = torch.zeros(batch, dtype=torch.long, device=states.device)
    for step in range(1, steps):
        relevant = step <= decision_index
        revisions += relevant & (top_route[step] != top_route[step - 1])

    reason_codes = torch.where(
        any_early,
        torch.zeros_like(decision_index),
        torch.where(
            committed,
            torch.full_like(decision_index, 2 if force_commit else 1),
            torch.full_like(decision_index, 3),
        ),
    )
    decision_steps = decision_index + 1
    return {
        "state": final_state,
        "candidate_states": candidate_states,
        "route_probabilities": final_probabilities,
        "candidate_residuals": final_residuals,
        "top_routes": final_top_routes,
        "routes": routes,
        "committed": committed,
        "abstained": ~committed,
        "decision_steps": decision_steps,
        "decision_reason_codes": reason_codes,
        "route_revisions": revisions,
        "mean_logical_steps": decision_steps.float().mean(),
        "logical_hypothesis_updates": decision_steps.sum() * structures,
        "policy": policy.as_dict(),
        "fixed_depth": fixed_depth,
        "force_commit": force_commit,
    }


def _single_event(
    step: int,
    candidate_states: torch.Tensor,
    probabilities: torch.Tensor,
    residuals: torch.Tensor,
    top_route: torch.Tensor,
    top_probability: torch.Tensor,
    margin: torch.Tensor,
    decision: DecisionName,
) -> dict[str, object]:
    return {
        "step": int(step),
        "decision": decision,
        "top_route": int(top_route.item()),
        "top_probability": float(top_probability.item()),
        "margin": float(margin.item()),
        "route_probabilities": probabilities.detach().cpu().tolist(),
        "candidate_residuals": residuals.detach().cpu().tolist(),
        "candidate_states": candidate_states.detach().cpu().tolist(),
    }


def _events_from_trajectory(
    trajectory: dict[str, torch.Tensor],
    result: dict[str, object],
    policy: DecisionPolicy,
) -> list[dict[str, object]]:
    states = trajectory["states"]
    probabilities = trajectory["route_probabilities"]
    residuals = trajectory["residuals"]
    assert states.shape[1] == 1
    decision_step = int(result["decision_steps"][0].item())  # type: ignore[index,union-attr]
    committed = bool(result["committed"][0].item())  # type: ignore[index,union-attr]
    events: list[dict[str, object]] = []
    for index in range(decision_step):
        pmax, margin, top = _top_statistics(probabilities[index])
        is_last = index + 1 == decision_step
        decision: DecisionName = "continue"
        if is_last:
            decision = "commit" if committed else "abstain"
        events.append(
            _single_event(
                index + 1,
                states[index, 0],
                probabilities[index, 0],
                residuals[index, 0],
                top[0],
                pmax[0],
                margin[0],
                decision,
            )
        )
    return events


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
