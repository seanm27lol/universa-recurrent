"""Two answers to two questions: estimate a flow; optionally propose its structure.

Like reporting a weighted weather forecast separately from an alert decision,
changing the alert threshold must not change the forecast. No model is retrained
here: this module isolates the output interface on the existing v2 checkpoints.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

import torch
from torch import nn

from .v2 import (AmbientRecurrentControl, DirectMultiHypothesisNet,
                 MultiHypothesisRecurrentNet, parameter_count)
from .baselines import fit_each_structure, gaussian_generator_reference

OutputMode = Literal['mixture', 'mixture_with_claim', 'hard']
MODES = ('mixture', 'mixture_with_claim', 'hard')


def positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


@dataclass(frozen=True)
class ClaimPolicy:
    """An inclusive confidence cutoff, not a claim of calibrated correctness.

    None explicitly means reject all. A cutoff of zero permits every example.
    Tied candidates use the first index, matching torch.argmax and numpy.argmax.
    """
    threshold: float | None
    target_coverage: float = 0.75

    def __post_init__(self) -> None:
        for name in ('threshold', 'target_coverage'):
            value = getattr(self, name)
            if value is None and name == 'threshold':
                continue
            if (isinstance(value, bool) or not isinstance(value, (float, int))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f'{name} must be finite and in [0,1]')

    def as_dict(self) -> dict:
        return asdict(self)


def validate_candidates(states: torch.Tensor, probabilities: torch.Tensor) -> None:
    if states.ndim != 3 or states.shape[0] < 1 or states.shape[1] < 2 or states.shape[2] < 1:
        raise ValueError('candidate states must be nonempty [batch,candidates,ambient]')
    if probabilities.shape != states.shape[:2]:
        raise ValueError('probabilities must match [batch,candidates]')
    if states.dtype != probabilities.dtype or states.device != probabilities.device:
        raise ValueError('candidate states and probabilities must share dtype/device')
    if not states.is_floating_point() or not torch.isfinite(states).all():
        raise ValueError('candidate states must be finite floating point')
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError('probabilities must be finite and in [0,1]')
    if not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities[:, 0]),
                          atol=1e-6, rtol=1e-6):
        raise ValueError('probabilities must sum to one')


def fit_coverage_policy(probabilities: torch.Tensor, target_coverage: float = 0.75) -> ClaimPolicy:
    """Largest cutoff retaining at least the requested calibration coverage.

    Uses scores only, never test labels or reconstruction errors. Include all
    ties at the cutoff; achieved coverage can exceed the target. This is an
    empirical coverage rule, NOT a statistical guarantee of test coverage/risk.
    """
    ClaimPolicy(None, target_coverage)  # validate even when target is zero
    if probabilities.ndim != 2 or probabilities.shape[0] < 1 or probabilities.shape[1] < 2:
        raise ValueError('probabilities must be nonempty [batch,candidates]')
    validate_candidates(torch.zeros((*probabilities.shape, 1),
                                    dtype=probabilities.dtype, device=probabilities.device), probabilities)
    if target_coverage == 0:
        return ClaimPolicy(None, target_coverage)
    count = max(1, math.ceil(target_coverage * probabilities.shape[0]))
    ranked = probabilities.max(-1).values.sort(descending=True).values
    return ClaimPolicy(float(ranked[count - 1].item()), target_coverage)


def construct_output(states: torch.Tensor, probabilities: torch.Tensor,
                     policy: ClaimPolicy, mode: OutputMode = 'mixture_with_claim',
                     *, validate: bool = True) -> dict[str, torch.Tensor]:
    """No labels, hidden truth, or future information are accepted here.

    The claim-state is a separate tensor. It is zero-filled when no claim is
    issued; callers MUST consult claim_mask. Human-facing records use None.
    """
    if mode not in MODES or not isinstance(policy, ClaimPolicy):
        raise ValueError('unsupported output mode or claim policy')
    if validate:
        validate_candidates(states, probabilities)
    batch = states.shape[0]
    top = probabilities.argmax(-1)
    confidence = probabilities.gather(1, top[:, None]).squeeze(1)
    index = torch.arange(batch, device=states.device)
    hard = states[index, top]
    # Kept independent of policy: no threshold may replace this estimate.
    estimate = (probabilities[..., None] * states).sum(1) if mode != 'hard' else hard
    if mode == 'mixture':
        accepted = torch.zeros(batch, dtype=torch.bool, device=states.device)
    elif mode == 'hard':
        accepted = torch.ones(batch, dtype=torch.bool, device=states.device)
    elif policy.threshold is None:
        accepted = torch.zeros(batch, dtype=torch.bool, device=states.device)
    else:
        accepted = confidence >= policy.threshold
    return {'estimate': estimate, 'candidate_states': states,
            'probabilities': probabilities, 'top_route': top,
            'claim_mask': accepted,
            'claim_index': torch.where(accepted, top, torch.full_like(top, -1)),
            'claim_state': torch.where(accepted[:, None], hard, torch.zeros_like(hard))}


@dataclass
class FixedEstimator:
    """An adapter, not a new architecture. All neural paths run their trained depth.

    No structural claim changes the number of iterations. This deliberately
    isolates output semantics before introducing a separately calibrated stop.
    Rollout history allocation is included in timing and disclosed in reports.
    """
    name: str
    model: nn.Module | None
    bases: torch.Tensor
    family: str
    noise_std: float = 0.05
    checkpoint_sha256: str = ''

    @property
    def structured(self) -> bool:
        return self.family not in ('ambient', 'fit_each_structure')

    @property
    def depth(self) -> int:
        if self.family in ('direct', 'gaussian', 'fit_each_structure'):
            return 1
        if self.family == 'ambient':
            return int(self.model.steps)
        return int(self.model.config.steps)

    def description(self) -> dict:
        return {'model': self.name, 'family': self.family, 'trained_depth': self.depth,
                'parameters': parameter_count(self.model) if self.model is not None else 0,
                'privileged': self.family == 'gaussian',
                'checkpoint_sha256': self.checkpoint_sha256,
                'candidate_states_per_sample': (self.bases.shape[0] * self.depth
                    if self.structured else None)}

    @torch.inference_mode()
    def candidates(self, observed: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model is not None:
            self.model.eval()
        if self.family == 'direct':
            result = self.model(observed, mask)
            return result['states'], result['route_probabilities']
        if self.family == 'gaussian':
            result = gaussian_generator_reference(observed, mask, self.bases,
                noise_std=self.noise_std, validate_values=False)
            return result['candidate_states'], result['route_probabilities']
        if self.family == 'v1':
            result = self.model.forward_train(observed, mask)
            return result['states'][-1], result['router_logits'].softmax(-1)
        if self.family not in ('shared', 'untied', 'dedicated'):
            raise ValueError(f'{self.name} does not produce comparable candidate probabilities')
        result = self.model.rollout(observed, mask)
        return result['states'][-1], result['route_probabilities'][-1]

    @torch.inference_mode()
    def predict(self, observed: torch.Tensor, mask: torch.Tensor,
                policy: ClaimPolicy | None, mode: str) -> dict[str, torch.Tensor]:
        if self.family == 'ambient':
            if mode != 'estimate_only':
                raise ValueError('ambient model does not make structural claims')
            self.model.eval()
            result = self.model.rollout(observed, mask)
            return {'estimate': result[-1]}
        if self.family == 'fit_each_structure':
            if mode != 'hard_reference':
                raise ValueError('residual scores are not invented route probabilities')
            result = fit_each_structure(observed, mask, self.bases, validate_values=False)
            return {'estimate': result['state']}
        if policy is None:
            raise ValueError('structured estimator requires a declared policy')
        states, probabilities = self.candidates(observed, mask)
        return construct_output(states, probabilities, policy, mode, validate=False)


def metrics(output: dict[str, torch.Tensor], truth: torch.Tensor,
            labels: torch.Tensor, mode: str) -> dict:
    """Integer counts and float64 reductions, including both error denominators."""
    estimate = output['estimate']
    if estimate.shape != truth.shape or truth.ndim != 2 or truth.shape[0] < 1:
        raise ValueError('estimate/truth shapes must match nonempty [batch,ambient]')
    if not torch.isfinite(estimate).all() or not torch.isfinite(truth).all():
        raise ValueError('nonfinite estimate/truth cannot produce accepted metrics')
    n = truth.shape[0]
    errors = (estimate.double() - truth.double()).square().mean(-1)
    if not torch.isfinite(errors).all():
        raise ValueError('nonfinite derived squared error')
    row = {'n': n, 'estimate_mse': float(errors.mean().item()),
           'output_mode': mode, 'claims_requested': mode in ('mixture_with_claim', 'hard')}
    if 'probabilities' not in output:
        return {**row, 'route_metrics_applicable': False}
    states, p = output['candidate_states'], output['probabilities']
    validate_candidates(states, p)
    if labels.shape != (n,) or labels.dtype != torch.long or labels.device != p.device:
        raise ValueError('labels must be a matching long vector')
    if ((labels < 0) | (labels >= p.shape[1])).any():
        raise ValueError('label out of range')
    top, accepted = output['top_route'], output['claim_mask']
    correct = top == labels
    probability = p.double()
    onehot = torch.nn.functional.one_hot(labels, p.shape[1]).double()
    brier = (probability - onehot).square().sum(-1).mean()
    logp = probability[torch.arange(n, device=p.device), labels].clamp_min(1e-15).log()
    row.update({'route_metrics_applicable': True,
                'top_route_accuracy': int(correct.sum().item()) / n,
                'route_brier': float(brier.item()), 'route_nll': float(-logp.mean().item())})
    if not row['claims_requested']:
        # An estimator not asked for a claim did not "abstain on every example".
        return {**row, 'coverage': None, 'claimed_n': None, 'abstained_n': None,
                'wrong_claim_rate_all': None, 'wrong_claim_rate_claimed': None}
    count = int(accepted.sum().item())
    wrong = int((accepted & ~correct).sum().item())
    candidate_errors = (output['claim_state'].double() - truth.double()).square().mean(-1)
    row.update({'claimed_n': count, 'abstained_n': n-count,
                'coverage': count/n, 'wrong_claim_n': wrong,
                'wrong_claim_rate_all': wrong/n,
                'wrong_claim_rate_claimed': wrong/count if count else None,
                'selective_route_accuracy': (count-wrong)/count if count else None,
                'estimate_mse_on_claimed': float(errors[accepted].mean().item()) if count else None,
                'estimate_mse_on_abstained': float(errors[~accepted].mean().item()) if count < n else None,
                'claim_candidate_mse_on_claimed': float(candidate_errors[accepted].mean().item()) if count else None})
    return row
