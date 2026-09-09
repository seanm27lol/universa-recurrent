"""Small, explicit references for the synthetic neural experiment.

These are not competitors for every future task. They answer two concrete
questions for the present generator:

* What does a simple fit-each-subspace rule achieve?
* What is achievable when the evaluator is given the generator's Gaussian
  prior and noise level?
"""
from __future__ import annotations

import math

import torch


def _validate(
    observed: torch.Tensor,
    mask: torch.Tensor,
    bases: torch.Tensor,
    *,
    validate_values: bool,
) -> tuple[int, int, int, int]:
    if observed.ndim != 2 or mask.shape != observed.shape:
        raise ValueError("observed and mask must have matching [B,N] shapes")
    if bases.ndim != 3:
        raise ValueError("bases must have shape [K,N,D]")
    batch_size, ambient_dim = observed.shape
    num_structures, basis_ambient, latent_dim = bases.shape
    if ambient_dim != basis_ambient:
        raise ValueError("basis and observation dimensions do not match")
    if observed.device != mask.device or observed.device != bases.device:
        raise ValueError("observed, mask, and bases must share a device")
    if not observed.is_floating_point() or not mask.is_floating_point():
        raise ValueError("observed and mask must be floating-point tensors")
    if observed.dtype != mask.dtype or observed.dtype != bases.dtype:
        raise ValueError("observed, mask, and bases must share a dtype")
    if not isinstance(validate_values, bool):
        raise ValueError("validate_values must be bool")
    if validate_values:
        # These checks inspect device values and therefore synchronize CUDA.
        # Keep them on for untrusted direct calls; trusted generated benchmark
        # data can validate once outside the timed hot loop.
        if not torch.isfinite(observed).all() or not torch.isfinite(mask).all():
            raise ValueError("observed and mask must be finite")
        if not torch.isfinite(bases).all():
            raise ValueError("bases must be finite")
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must contain only 0 and 1")
        if not torch.all(mask.sum(dim=-1) >= 1):
            raise ValueError("every example must expose at least one coordinate")
    return batch_size, ambient_dim, num_structures, latent_dim


def fit_each_structure(
    observed: torch.Tensor,
    mask: torch.Tensor,
    bases: torch.Tensor,
    *,
    ridge: float = 1e-6,
    validate_values: bool = True,
) -> dict[str, torch.Tensor]:
    """Fit every candidate subspace and route by masked residual.

    This is a transparent non-learned baseline. A tiny ridge makes rank-deficient
    mask patterns well-defined; it is not tuned to the held-out set.
    """
    _, _, _, latent_dim = _validate(
        observed, mask, bases, validate_values=validate_values
    )
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be positive and finite")

    dtype = observed.dtype
    gram = torch.einsum("bn,knd,kne->bkde", mask, bases, bases)
    identity = torch.eye(latent_dim, dtype=dtype, device=observed.device)
    gram = gram + ridge * identity
    right = torch.einsum("bn,knd,bn->bkd", mask, bases, observed)
    coordinates = torch.linalg.solve(gram, right.unsqueeze(-1)).squeeze(-1)
    states = torch.einsum("knd,bkd->bkn", bases, coordinates)
    residual = mask[:, None, :] * (states - observed[:, None, :])
    denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    scores = residual.square().sum(dim=-1) / denominator
    routes = scores.argmin(dim=-1)
    index = torch.arange(observed.shape[0], device=observed.device)
    state = states[index, routes]
    return {
        "routes": routes,
        "state": state,
        "candidate_states": states,
        "scores": scores,
    }


def gaussian_generator_reference(
    observed: torch.Tensor,
    mask: torch.Tensor,
    bases: torch.Tensor,
    *,
    noise_std: float,
    validate_values: bool = True,
) -> dict[str, torch.Tensor]:
    """Bayes reference under the exact synthetic data-generating assumptions.

    Coordinates have an isotropic standard-normal prior and observed entries
    receive independent Gaussian noise. This is privileged knowledge of the toy
    generator, so it is a privileged reference rather than a deployable method.
    Its posterior mixture is Bayes-optimal for squared error under those exact
    assumptions; the hard MAP route is a separate diagnostic and need not be
    the best possible hard-route decision for MSE.
    """
    _, _, _, latent_dim = _validate(
        observed, mask, bases, validate_values=validate_values
    )
    if not math.isfinite(noise_std) or noise_std <= 0:
        raise ValueError("noise_std must be positive and finite")

    variance = float(noise_std) ** 2
    gram = torch.einsum("bn,knd,kne->bkde", mask, bases, bases) / variance
    identity = torch.eye(
        latent_dim, dtype=observed.dtype, device=observed.device
    )
    precision = gram + identity
    right = torch.einsum("bn,knd,bn->bkd", mask, bases, observed) / variance
    coordinates = torch.linalg.solve(
        precision, right.unsqueeze(-1)
    ).squeeze(-1)
    candidate_states = torch.einsum("knd,bkd->bkn", bases, coordinates)

    # Matrix determinant lemma and Woodbury identity avoid variable-size
    # covariance matrices for the different masks.
    observed_count = mask.sum(dim=-1, keepdim=True)
    logdet_precision = torch.linalg.slogdet(precision).logabsdet
    logdet_covariance = observed_count * math.log(variance) + logdet_precision
    y_quadratic = (mask * observed.square()).sum(dim=-1, keepdim=True) / variance
    correction = torch.einsum("bkd,bkd->bk", right, coordinates)
    quadratic = y_quadratic - correction
    log_likelihood = -0.5 * (
        quadratic
        + logdet_covariance
        + observed_count * math.log(2.0 * math.pi)
    )

    route_probabilities = torch.softmax(log_likelihood, dim=-1)
    routes = route_probabilities.argmax(dim=-1)
    index = torch.arange(observed.shape[0], device=observed.device)
    hard_state = candidate_states[index, routes]
    mixture_state = torch.einsum(
        "bk,bkn->bn", route_probabilities, candidate_states
    )
    return {
        "routes": routes,
        "hard_state": hard_state,
        "mixture_state": mixture_state,
        "candidate_states": candidate_states,
        "route_probabilities": route_probabilities,
        "log_likelihood": log_likelihood,
    }
