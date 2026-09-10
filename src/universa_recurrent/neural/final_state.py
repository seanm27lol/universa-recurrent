"""Keep the current worksheet, not a copy of every previous worksheet.

These inference-only adapters execute the SAME trained updates at the SAME depth
as v2.rollout. They do not build lists/stacks of intermediate states. This is a
retention experiment, not early stopping, a new model, or a promised speedup.
The original rollout and training paths remain untouched as reference controls.
"""
from __future__ import annotations

from dataclasses import fields
import torch

from .dual_output import FixedEstimator
from .v2 import MultiHypothesisRecurrentNet, UntiedMultiHypothesisNet, AmbientRecurrentControl


def _limit(steps: int | None, configured: int) -> int:
    value = configured if steps is None else steps
    if type(value) is not int or not 1 <= value <= configured:
        raise ValueError('steps must be an integer in [1, configured depth]')
    return value


@torch.inference_mode()
def final_candidates(model: MultiHypothesisRecurrentNet, observed: torch.Tensor,
                     mask: torch.Tensor, *, steps: int | None = None,
                     validate_values: bool = True) -> dict[str, torch.Tensor]:
    """Return final candidate coordinates, states and diagnostics, without a time axis.

    Reuse the producer's actual update functions. Untied depth MUST dispatch to
    its depth-specific update, not the shared block inherited from the base class.
    Only the last probability readout is materialized; internal step probabilities
    used to compute subsequent states are still calculated normally.
    """
    if not isinstance(model, MultiHypothesisRecurrentNet) or model.training:
        raise ValueError('final_candidates requires an evaluated v2 recurrent model')
    model._validate_inputs(observed, mask, validate_values=validate_values)
    if observed.shape[0] < 1:
        raise ValueError('empty inference batch')
    limit = _limit(steps, model.config.steps)
    context = model._context(observed, mask)
    prior = model.prior_router(context)
    logits = prior
    coords = observed.new_zeros(observed.shape[0], model.config.num_structures, model.config.latent_dim)
    for depth in range(limit):
        if isinstance(model, UntiedMultiHypothesisNet):
            coords, state, logits, rms, progress = model._step_at_depth(
                depth, context, observed, mask, coords, prior, logits)
        else:
            coords, state, logits, rms, progress = model._step(
                context, observed, mask, coords, prior, logits)
    return {'prior_logits': prior, 'coordinates': coords, 'states': state,
            'route_logits': logits, 'route_probabilities': logits.softmax(-1),
            'residuals': rms, 'progress': progress}


@torch.inference_mode()
def final_ambient(model: AmbientRecurrentControl, observed: torch.Tensor,
                  mask: torch.Tensor, *, steps: int | None = None,
                  validate_values: bool = True) -> torch.Tensor:
    """Ambient control with the original update algebra but no history stack.

    The short update expression intentionally matches AmbientRecurrentControl's
    rollout. Regression tests and per-checkpoint runtime checks guard divergence.
    No parameter, buffer or checkpoint format is changed.
    """
    if not isinstance(model, AmbientRecurrentControl) or model.training:
        raise ValueError('final_ambient requires an evaluated ambient model')
    if observed.ndim != 2 or mask.shape != observed.shape or observed.shape[0] < 1:
        raise ValueError('observed/mask must be matching nonempty [batch,ambient]')
    parameter = next(model.parameters())
    if (observed.shape[1] != model.ambient_dim or observed.device != mask.device
            or observed.device != parameter.device or observed.dtype != mask.dtype
            or observed.dtype != parameter.dtype):
        raise ValueError('input width, device and dtype must match model')
    if type(validate_values) is not bool:
        raise ValueError('validate_values must be bool')
    if validate_values and (not torch.isfinite(observed).all() or not torch.isfinite(mask).all()
                            or not ((mask == 0) | (mask == 1)).all() or not (mask.sum(-1) >= 1).all()):
        raise ValueError('finite observations and nonempty binary masks required')
    limit = _limit(steps, model.steps)
    context = model.encoder(torch.cat((observed * mask, mask), dim=-1))
    state = observed.new_zeros(observed.shape)
    denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    for _ in range(limit):
        residual = mask * (state - observed)
        rms = torch.linalg.vector_norm(residual, dim=-1, keepdim=True) / denominator.sqrt()
        state = state + model.update(torch.cat((context, state, residual, rms), dim=-1))
    return state


class FinalStateEstimator(FixedEstimator):
    """Opt-in drop-in adapter for collect(); historical benchmarks stay unchanged."""

    @classmethod
    def from_reference(cls, reference: FixedEstimator) -> 'FinalStateEstimator':
        # Do not use asdict(): it would deep-copy model weights.
        return cls(**{field.name: getattr(reference, field.name) for field in fields(FixedEstimator)})

    @torch.inference_mode()
    def candidates(self, observed, mask):
        if self.family in ('shared', 'untied', 'dedicated'):
            self.model.eval()
            result = final_candidates(self.model, observed, mask, validate_values=False)
            return result['states'], result['route_probabilities']
        # Direct and analytic controls have no recurrent history to eliminate.
        return super().candidates(observed, mask)

    @torch.inference_mode()
    def predict(self, observed, mask, policy, mode):
        if self.family == 'ambient':
            if mode != 'estimate_only':
                raise ValueError('ambient model does not make structural claims')
            self.model.eval()
            return {'estimate': final_ambient(self.model, observed, mask, validate_values=False)}
        return super().predict(observed, mask, policy, mode)
