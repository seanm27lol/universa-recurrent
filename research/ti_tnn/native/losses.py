"""External, float32 training losses for the native causal language model.

Data labels contain same-position token IDs. Language loss scores ``logits[:, :-1]`` against ``labels[:, 1:]``;
ignored targets at sequence/document starts and padding contribute neither loss nor supervision counts. Auxiliary
supervision remains at its original token positions, accepts batched or batch-broadcast forms, and is never modified.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100
N_PASSES = 4
N_EXPERTS = 4
N_SLOTS = 4


def _cross_entropy_sum(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=logits.device.type, enabled=False):
        return F.cross_entropy(logits.float(), targets, reduction="sum")


def _masked_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[:-1] != targets.shape or targets.shape != mask.shape:
        raise ValueError(
            "logits, targets, and mask must agree in every non-class dimension; got "
            f"{tuple(logits.shape)}, {tuple(targets.shape)}, and {tuple(mask.shape)}"
        )
    targets = targets.to(device=logits.device, dtype=torch.long)
    valid = mask.to(device=logits.device, dtype=torch.bool) & targets.ne(IGNORE_INDEX)
    count = valid.sum()
    selected_logits = logits[valid]
    selected_targets = targets[valid]
    loss = _cross_entropy_sum(selected_logits, selected_targets) / count.clamp_min(1)
    return loss, count


def _batch_supervision(
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    batch: int,
    unbatched_shape: tuple[int, ...],
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batched_shape = (batch, *unbatched_shape)
    if targets.shape != mask.shape:
        raise ValueError(
            f"{name} targets and mask must have identical shapes, got "
            f"{tuple(targets.shape)} and {tuple(mask.shape)}"
        )
    if targets.shape == batched_shape:
        return targets, mask
    if targets.shape == unbatched_shape:
        return (
            targets.unsqueeze(0).expand(*batched_shape),
            mask.unsqueeze(0).expand(*batched_shape),
        )
    raise ValueError(
        f"{name} targets and mask must both have shape {batched_shape} or {unbatched_shape}, "
        f"got {tuple(targets.shape)}"
    )


def next_token_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causally shifted next-token cross entropy and its supervised token count.

    ``logits`` has shape ``[B,T,V]`` and same-position ``labels`` has shape ``[B,T]``, with ``T >= 2``. Exactly
    ``logits[:, :-1]`` is scored against ``labels[:, 1:]``. The scalar loss is float32, including under autocast,
    and the count is an int64 tensor on the logits' device. If every shifted label is -100, loss is differentiable
    zero.
    """
    if logits.ndim != 3:
        raise ValueError(f"language logits must have shape [B,T,V], got {tuple(logits.shape)}")
    if logits.shape[1] < 2:
        raise ValueError(f"language logits must have T >= 2 for next-token shifting, got {logits.shape[1]}")
    if labels.shape != logits.shape[:-1]:
        raise ValueError(
            f"labels must have shape {tuple(logits.shape[:-1])}, got {tuple(labels.shape)}"
        )
    shifted_labels = labels[:, 1:]
    return _masked_cross_entropy(logits[:, :-1], shifted_labels, shifted_labels.ne(IGNORE_INDEX))


def _switch_load_balance(
    route_logits: torch.Tensor,
    route_indices: torch.Tensor,
    key_padding_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if route_logits.ndim != 4 or route_logits.shape[2:] != (N_PASSES, N_EXPERTS):
        raise ValueError(
            f"route logits must have shape [B,T,{N_PASSES},{N_EXPERTS}], got {tuple(route_logits.shape)}"
        )
    if route_indices.shape != route_logits.shape[:-1]:
        raise ValueError(
            f"route indices must have shape {tuple(route_logits.shape[:-1])}, got {tuple(route_indices.shape)}"
        )
    if key_padding_mask is not None and key_padding_mask.shape != route_logits.shape[:2]:
        raise ValueError(
            f"key padding mask must have shape {tuple(route_logits.shape[:2])}, "
            f"got {tuple(key_padding_mask.shape)}"
        )
    indices = route_indices.to(device=route_logits.device, dtype=torch.long).detach()
    selected_logits = route_logits
    if key_padding_mask is not None:
        live = ~key_padding_mask.to(device=route_logits.device, dtype=torch.bool)
        selected_logits = route_logits[live]
        indices = indices[live]
    selected_logits = selected_logits.reshape(-1, N_EXPERTS)
    indices = indices.reshape(-1)
    count = torch.tensor(indices.numel(), dtype=torch.int64, device=route_logits.device)
    with torch.autocast(device_type=route_logits.device.type, enabled=False):
        probabilities = F.softmax(selected_logits.float(), dim=-1)
        denominator = max(indices.numel(), 1)
        fractions = F.one_hot(indices, num_classes=N_EXPERTS).float().sum(dim=0) / denominator
        mean_probabilities = probabilities.sum(dim=0) / denominator
        balance = N_EXPERTS * (fractions * mean_probabilities).sum()
    return balance, count


def switch_load_balance(
    route_logits: torch.Tensor,
    route_indices: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``4 * sum_e f_e P_e`` for live batch, token, and pass routing decisions.

    ``f_e`` is the detached hard-selection fraction and ``P_e`` is the mean float32 router probability. A true
    ``key_padding_mask`` entry excludes all four passes at that token from both quantities. Omitting the mask retains
    the original all-position behavior. An empty live set produces differentiable zero rather than NaN.
    """
    balance, _ = _switch_load_balance(route_logits, route_indices, key_padding_mask)
    return balance


def masked_route_cross_entropy(
    route_logits: torch.Tensor, route_targets: torch.Tensor, route_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return route cross entropy and count for ``[B,T,4]`` or broadcast ``[T,4]`` supervision."""
    if route_logits.ndim != 4 or route_logits.shape[2:] != (N_PASSES, N_EXPERTS):
        raise ValueError(
            f"route logits must have shape [B,T,{N_PASSES},{N_EXPERTS}], got {tuple(route_logits.shape)}"
        )
    targets, mask = _batch_supervision(
        route_targets,
        route_mask,
        batch=route_logits.shape[0],
        unbatched_shape=(route_logits.shape[1], N_PASSES),
        name="route",
    )
    return _masked_cross_entropy(route_logits, targets, mask)


def masked_state_cross_entropy(
    state_logits: torch.Tensor, state_targets: torch.Tensor, state_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return state loss and count for ``[B,T,4,4]`` or broadcast ``[T,4,4]`` supervision."""
    if state_logits.ndim != 5 or state_logits.shape[2:4] != (N_PASSES, N_SLOTS):
        raise ValueError(
            f"state logits must have shape [B,T,{N_PASSES},{N_SLOTS},C], got {tuple(state_logits.shape)}"
        )
    targets, mask = _batch_supervision(
        state_targets,
        state_mask,
        batch=state_logits.shape[0],
        unbatched_shape=(state_logits.shape[1], N_PASSES, N_SLOTS),
        name="state",
    )
    return _masked_cross_entropy(state_logits, targets, mask)


def compute_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    key_padding_mask: torch.Tensor | None = None,
    route_logits: torch.Tensor | None = None,
    route_indices: torch.Tensor | None = None,
    route_targets: torch.Tensor | None = None,
    route_mask: torch.Tensor | None = None,
    state_logits: torch.Tensor | None = None,
    state_targets: torch.Tensor | None = None,
    state_mask: torch.Tensor | None = None,
    lambda_balance: float = 1.0,
    lambda_route: float = 1.0,
    lambda_state: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute native losses without passing auxiliary tensors into the model.

    Language labels use the same causal shift as :func:`next_token_cross_entropy`; auxiliary targets retain their
    original token positions. ``key_padding_mask`` excludes padded routes from load balancing only. The dictionary
    contains separate ``language``, ``route``, ``state``, and ``balance`` terms plus ``total``. ``token_count`` counts
    shifted non-ignored labels, ``route_count`` and ``state_count`` count supervised auxiliary entries, and
    ``balance_count`` counts included routing decisions (four per live token). Omitting the padding mask retains the
    original all-route behavior. Targets and masks must be supplied in pairs, in either matching batched or matching
    unbatched form; fully masked or absent auxiliary terms are zero.
    """
    if key_padding_mask is not None and key_padding_mask.shape != logits.shape[:-1]:
        raise ValueError(
            f"key padding mask must have shape {tuple(logits.shape[:-1])}, got {tuple(key_padding_mask.shape)}"
        )
    language, token_count = next_token_cross_entropy(logits, labels)
    zero = language.new_zeros(())
    zero_count = token_count.new_zeros(())
    route = zero
    state = zero
    balance = zero
    route_count = zero_count
    state_count = zero_count
    balance_count = zero_count

    if (route_logits is None) != (route_indices is None):
        raise ValueError("route_logits and route_indices must be supplied together")
    if (route_targets is None) != (route_mask is None):
        raise ValueError("route_targets and route_mask must be supplied together")
    if (state_targets is None) != (state_mask is None):
        raise ValueError("state_targets and state_mask must be supplied together")

    if route_logits is not None:
        balance, balance_count = _switch_load_balance(route_logits, route_indices, key_padding_mask)
        if route_targets is not None:
            route, route_count = masked_route_cross_entropy(route_logits, route_targets, route_mask)
    elif route_targets is not None:
        raise ValueError("route supervision requires route_logits and route_indices")

    if state_targets is not None:
        if state_logits is None:
            raise ValueError("state supervision requires state_logits")
        state, state_count = masked_state_cross_entropy(state_logits, state_targets, state_mask)

    total = language + lambda_balance * balance + lambda_route * route + lambda_state * state
    return {
        "total": total,
        "language": language,
        "route": route,
        "state": state,
        "balance": balance,
        "token_count": token_count,
        "route_count": route_count,
        "state_count": state_count,
        "balance_count": balance_count,
    }


__all__ = [
    "IGNORE_INDEX",
    "compute_losses",
    "masked_route_cross_entropy",
    "masked_state_cross_entropy",
    "next_token_cross_entropy",
    "switch_load_balance",
]
