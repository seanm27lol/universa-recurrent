"""Greedy complete-sequence decoding using the model's per-pass attention cache."""

from __future__ import annotations

import torch
from torch import Tensor


def _metadata(
    input_ids: Tensor,
    positions: Tensor | None,
    doc_ids: Tensor | None,
    key_padding_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    batch, length = input_ids.shape
    if key_padding_mask is None:
        key_padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    else:
        key_padding_mask = key_padding_mask.to(input_ids.device).bool()
        if key_padding_mask.ndim == 1:
            key_padding_mask = key_padding_mask.unsqueeze(0).expand(batch, length)
    if key_padding_mask.shape != input_ids.shape:
        raise ValueError("key_padding_mask must have shape [B,T] or [T]")
    if positions is None:
        positions = ((~key_padding_mask).long().cumsum(dim=1) - 1).clamp_min(0)
    else:
        positions = positions.to(input_ids.device).long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(batch, length)
    if positions.shape != input_ids.shape:
        raise ValueError("positions must have shape [B,T] or [T]")
    if doc_ids is None:
        doc_ids = torch.zeros_like(input_ids, dtype=torch.long)
    else:
        doc_ids = doc_ids.to(input_ids.device).long()
        if doc_ids.ndim == 1:
            doc_ids = doc_ids.unsqueeze(0).expand(batch, length)
    if doc_ids.shape != input_ids.shape:
        raise ValueError("doc_ids must have shape [B,T] or [T]")
    return positions, doc_ids, key_padding_mask


def _last_valid(mask: Tensor) -> Tensor:
    indices = torch.arange(mask.shape[1], device=mask.device).expand(mask.shape)
    last = indices.masked_fill(mask, -1).max(dim=1).values
    return torch.where(
        last >= 0,
        last,
        torch.full_like(last, mask.shape[1] - 1),
    )


@torch.no_grad()
def greedy_generate(
    model,
    input_ids: Tensor,
    max_new_tokens: int,
    *,
    positions: Tensor | None = None,
    doc_ids: Tensor | None = None,
    key_padding_mask: Tensor | None = None,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    use_cache: bool = True,
) -> Tensor:
    """Append up to ``max_new_tokens`` greedy tokens, without sampling.

    Finished rows receive padding while other rows continue. Generation stops
    early only after every row has emitted EOS. No auxiliary tensor is accepted.
    """

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative")
    if input_ids.ndim != 2 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have nonempty shape [B,T]")
    if max_new_tokens == 0:
        return input_ids.clone()
    positions, doc_ids, key_padding_mask = _metadata(
        input_ids, positions, doc_ids, key_padding_mask
    )
    sequences = input_ids
    all_positions = positions
    all_doc_ids = doc_ids
    all_key_padding_mask = key_padding_mask
    current_ids = input_ids
    current_positions = positions
    current_doc_ids = doc_ids
    current_key_padding_mask = key_padding_mask
    cache = None
    last_valid = _last_valid(key_padding_mask)
    batch_index = torch.arange(input_ids.shape[0], device=input_ids.device)
    next_positions = positions[batch_index, last_valid].unsqueeze(1) + 1
    next_doc_ids = doc_ids[batch_index, last_valid].unsqueeze(1)
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for step in range(max_new_tokens):
        output = model(
            current_ids,
            positions=current_positions,
            doc_ids=current_doc_ids,
            key_padding_mask=current_key_padding_mask,
            cache=cache,
            use_cache=use_cache,
        )
        if step == 0:
            next_logits = output.logits[batch_index, last_valid]
        else:
            next_logits = output.logits[:, -1]
        next_tokens = torch.argmax(next_logits, dim=-1)
        if eos_token_id is not None:
            padding_value = pad_token_id if pad_token_id is not None else eos_token_id
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, padding_value),
                next_tokens,
            )
        generated_padding = finished.unsqueeze(1)
        sequences = torch.cat((sequences, next_tokens.unsqueeze(1)), dim=1)
        all_positions = torch.cat((all_positions, next_positions), dim=1)
        all_doc_ids = torch.cat((all_doc_ids, next_doc_ids), dim=1)
        all_key_padding_mask = torch.cat((all_key_padding_mask, generated_padding), dim=1)
        if eos_token_id is not None:
            finished = finished | (next_tokens == eos_token_id)
        cache = output.cache
        if use_cache:
            current_ids = next_tokens.unsqueeze(1)
            current_positions = next_positions
            current_doc_ids = next_doc_ids
            current_key_padding_mask = generated_padding
        else:
            current_ids = sequences
            current_positions = all_positions
            current_doc_ids = all_doc_ids
            current_key_padding_mask = all_key_padding_mask
        next_positions = next_positions + (~generated_padding).long()
        if eos_token_id is not None and finished.all():
            break
    return sequences


generate_greedy = greedy_generate
