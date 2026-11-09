"""Four-pass native Transformer with explicit slots and optional top-1 experts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .config import NativeConfig


@dataclass
class NativeCache:
    """Per-pass attention K/V plus the positions and masks needed to extend it."""

    keys: list[Tensor]
    values: list[Tensor]
    positions: Tensor
    doc_ids: Tensor
    key_padding_mask: Tensor

    @property
    def length(self) -> int:
        return self.keys[0].shape[2]

    @property
    def batch_size(self) -> int:
        return self.doc_ids.shape[0]


@dataclass
class NativeOutput:
    logits: Tensor
    route_logits: Tensor | None = None
    state_logits: Tensor | None = None
    cache: NativeCache | None = None
    route_indices: Tensor | None = None
    route_probs: Tensor | None = None
    route_weights: Tensor | None = None
    load_stats: dict[str, Tensor] | None = None
    h: Tensor | None = None
    slots: Tensor | None = None

    @property
    def hidden_trace(self) -> Tensor | None:
        return self.h

    @property
    def state_trace(self) -> Tensor | None:
        return self.slots


class DenseTransform(nn.Module):
    def __init__(self, config: NativeConfig) -> None:
        super().__init__()
        self.up = nn.Linear(2 * config.d_model, config.d_ff)
        self.down = nn.Linear(config.d_ff, config.d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.down(torch.nn.functional.gelu(self.up(inputs)))


class CausalAttention(nn.Module):
    def __init__(self, config: NativeConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.d_model)
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        hidden: Tensor,
        past_key: Tensor | None,
        past_value: Tensor | None,
        doc_ids: Tensor,
        key_padding_mask: Tensor,
        past_length: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, length, d_model = hidden.shape
        n_heads = self.config.n_heads
        head_dim = self.config.head_dim
        qkv = self.qkv(self.norm(hidden))
        qkv = qkv.reshape(batch, length, 3, n_heads, head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        if past_key is not None:
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)
        key_length = key.shape[2]
        scores = query @ key.transpose(-2, -1) / math.sqrt(head_dim)
        query_positions = past_length + torch.arange(length, device=hidden.device)
        key_positions = torch.arange(key_length, device=hidden.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        same_document = doc_ids[:, past_length:, None] == doc_ids[:, None, :]
        visible = (~key_padding_mask)[:, None, :]
        live_query = (~key_padding_mask)[:, past_length:, None]
        allowed = causal[None, :, :] & same_document & visible & live_query
        scores = scores.masked_fill(~allowed[:, None, :, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        attended = (weights @ value).transpose(1, 2).reshape(batch, length, d_model)
        return self.output(attended), key, value


class NativeModel(nn.Module):
    """Randomly initialized A/B/C/D model; auxiliary targets are never inputs."""

    def __init__(self, config: NativeConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        d_s = config.d_slot
        self.token_embedding = nn.Embedding(config.vocab_size, d)
        self.position_embedding = nn.Embedding(config.context_length, d)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        self.slot_initializers = nn.Parameter(torch.empty(4, d, d_s))
        nn.init.normal_(self.slot_initializers, std=0.02)
        self.output_norm = nn.LayerNorm(d)
        self.output_head = nn.Linear(d, config.vocab_size, bias=False)
        nn.init.normal_(self.output_head.weight, std=0.02)
        self.state_probes = nn.ModuleList(
            [nn.Linear(d_s, config.n_state_classes, bias=False) for _ in range(4)]
        )

        # This common construction order aligns corresponding initial draws across modes.
        first_attention = CausalAttention(config)
        first_transform = DenseTransform(config)
        self.h_reader = nn.Linear(d, d, bias=False)
        self.h_writer = nn.Linear(d, d, bias=False)
        generic_slot_reader = nn.Linear(d_s, d, bias=False)
        generic_slot_writer = nn.Linear(2 * d, d_s, bias=False)

        if config.mode == "dense":
            self.attentions = nn.ModuleList(
                [first_attention] + [CausalAttention(config) for _ in range(3)]
            )
            self.transforms = nn.ModuleList(
                [first_transform] + [DenseTransform(config) for _ in range(3)]
            )
        else:
            self.attention = first_attention
            if config.mode == "shared":
                self.transform = first_transform
            else:
                self.experts = nn.ModuleList(
                    [first_transform] + [DenseTransform(config) for _ in range(3)]
                )

        if config.mode == "typed":
            self.slot_readers = nn.ModuleList([nn.Linear(d_s, d, bias=False) for _ in range(4)])
            self.slot_writers = nn.ModuleList([nn.Linear(2 * d, d_s, bias=False) for _ in range(4)])
            self.slot_codes = nn.Parameter(torch.empty(4, d))
            nn.init.normal_(self.slot_codes, std=0.02)
            self.router = nn.Linear(2 * d, 4)
        else:
            self.slot_reader = generic_slot_reader
            self.slot_writer = generic_slot_writer
            if config.mode == "untyped":
                self.router = nn.Linear(d, 4)

    def _attention(self, pass_index: int) -> CausalAttention:
        if self.config.mode == "dense":
            return self.attentions[pass_index]
        return self.attention

    def _read_slots(self, slots: Tensor) -> tuple[Tensor, list[Tensor]]:
        if self.config.mode == "typed":
            projected = [
                self.slot_readers[index](slots[:, :, index, :]) + self.slot_codes[index]
                for index in range(4)
            ]
        else:
            projected = [self.slot_reader(slots[:, :, index, :]) for index in range(4)]
        return torch.stack(projected, dim=2).mean(dim=2), projected

    def _write_slots(
        self,
        slots: Tensor,
        transformed: Tensor,
        projected: list[Tensor],
    ) -> Tensor:
        updates = []
        for index in range(4):
            writer_inputs = torch.cat((transformed, projected[index]), dim=-1)
            if self.config.mode == "typed":
                update = self.slot_writers[index](writer_inputs)
            else:
                update = self.slot_writer(writer_inputs)
            updates.append(slots[:, :, index, :] + update)
        return torch.stack(updates, dim=2)

    def _run_experts(self, inputs: Tensor, router_inputs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch, length, _ = inputs.shape
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        logits = self.router(router_inputs.reshape(-1, router_inputs.shape[-1]))
        probabilities = torch.softmax(logits.float(), dim=-1)
        indices = torch.argmax(probabilities, dim=-1)
        one_hot = torch.nn.functional.one_hot(indices, num_classes=4).to(probabilities.dtype)
        straight_through = probabilities + (one_hot - probabilities).detach()
        outputs = None
        for expert_index, expert in enumerate(self.experts):
            selected = torch.nonzero(indices == expert_index, as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            expert_output = expert(flat_inputs.index_select(0, selected))
            coefficient = straight_through[selected, expert_index].to(expert_output.dtype).unsqueeze(-1)
            if outputs is None:
                outputs = expert_output.new_zeros((flat_inputs.shape[0], self.config.d_model))
            outputs.index_copy_(0, selected, expert_output * coefficient)
        assert outputs is not None
        return (
            outputs.reshape(batch, length, self.config.d_model),
            logits.reshape(batch, length, 4),
            probabilities.reshape(batch, length, 4),
            straight_through.reshape(batch, length, 4),
        )

    def _normalize_positions(
        self,
        positions: Tensor | None,
        input_ids: Tensor,
        cache: NativeCache | None,
    ) -> Tensor:
        batch, length = input_ids.shape
        if positions is None:
            if cache is None:
                positions = torch.arange(length, device=input_ids.device)
            else:
                return (cache.positions[:, -1:] + 1 + torch.arange(length, device=input_ids.device)).expand(batch, length)
        positions = positions.to(device=input_ids.device, dtype=torch.long)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(batch, length)
        if positions.shape != input_ids.shape:
            raise ValueError("positions must have shape [B,T] or [T]")
        if positions.numel() and (positions.min() < 0 or positions.max() >= self.config.context_length):
            raise ValueError("positions exceed the configured context_length")
        return positions

    def _normalize_tokens_metadata(
        self,
        values: Tensor | None,
        input_ids: Tensor,
        name: str,
        cache: NativeCache | None,
    ) -> Tensor:
        batch, length = input_ids.shape
        if values is None:
            if name == "key_padding_mask":
                return torch.zeros_like(input_ids, dtype=torch.bool)
            if cache is not None:
                return cache.doc_ids[:, -1:].expand(batch, length)
            return torch.zeros_like(input_ids, dtype=torch.long)
        values = values.to(input_ids.device)
        if values.ndim == 1:
            values = values.unsqueeze(0).expand(batch, length)
        if values.shape != input_ids.shape:
            raise ValueError(f"{name} must have shape [B,T] or [T]")
        if name == "key_padding_mask":
            return values.bool()
        return values.long()

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor | None = None,
        doc_ids: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        cache: NativeCache | None = None,
        use_cache: bool = False,
        return_aux: bool = False,
        return_trace: bool = False,
    ) -> NativeOutput:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have nonempty shape [B,T]")
        batch, length = input_ids.shape
        if cache is not None:
            if len(cache.keys) != 4 or len(cache.values) != 4:
                raise ValueError("cache must contain four per-pass K/V pairs")
            if cache.batch_size != batch:
                raise ValueError("cache batch size does not match input_ids")
        positions = self._normalize_positions(positions, input_ids, cache)
        doc_ids = self._normalize_tokens_metadata(doc_ids, input_ids, "doc_ids", cache)
        key_padding_mask = self._normalize_tokens_metadata(
            key_padding_mask, input_ids, "key_padding_mask", cache
        )
        past_length = cache.length if cache is not None else 0
        if cache is not None:
            all_positions = torch.cat((cache.positions, positions), dim=1)
            all_doc_ids = torch.cat((cache.doc_ids, doc_ids), dim=1)
            all_key_padding_mask = torch.cat((cache.key_padding_mask, key_padding_mask), dim=1)
        else:
            all_positions = positions
            all_doc_ids = doc_ids
            all_key_padding_mask = key_padding_mask

        embedded = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = embedded
        slots = torch.stack(
            [embedded @ self.slot_initializers[index] for index in range(4)], dim=2
        )
        present_keys: list[Tensor] = []
        present_values: list[Tensor] = []
        hidden_trace = []
        slot_trace = []
        state_trace = []
        route_logit_trace = []
        route_probability_trace = []
        route_weight_trace = []
        route_index_trace = []

        for pass_index in range(4):
            past_key = cache.keys[pass_index] if cache is not None else None
            past_value = cache.values[pass_index] if cache is not None else None
            attended, key, value = self._attention(pass_index)(
                hidden,
                past_key,
                past_value,
                all_doc_ids,
                all_key_padding_mask,
                past_length,
            )
            present_keys.append(key)
            present_values.append(value)
            query = hidden + attended
            h_read = self.h_reader(query)
            slot_read, projected_slots = self._read_slots(slots)
            transform_input = torch.cat((h_read, slot_read), dim=-1)
            if self.config.mode in ("untyped", "typed"):
                router_input = transform_input if self.config.mode == "typed" else h_read
                transformed, route_logits, route_probs, route_weights = self._run_experts(
                    transform_input, router_input
                )
                route_logit_trace.append(route_logits)
                route_probability_trace.append(route_probs)
                route_weight_trace.append(route_weights)
                route_index_trace.append(torch.argmax(route_probs, dim=-1))
            elif self.config.mode == "dense":
                transformed = self.transforms[pass_index](transform_input)
            else:
                transformed = self.transform(transform_input)
            hidden = query + self.h_writer(transformed)
            slots = self._write_slots(slots, transformed, projected_slots)
            if return_trace:
                hidden_trace.append(hidden)
                slot_trace.append(slots)
            if return_aux:
                state_trace.append(
                    torch.stack(
                        [self.state_probes[index](slots[:, :, index, :]) for index in range(4)],
                        dim=2,
                    )
                )

        logits = self.output_head(self.output_norm(hidden))
        new_cache = None
        if use_cache:
            new_cache = NativeCache(
                present_keys,
                present_values,
                all_positions,
                all_doc_ids,
                all_key_padding_mask,
            )
        route_logits = route_probs = route_weights = route_indices = None
        load_stats = None
        if self.config.mode in ("untyped", "typed"):
            stacked_logits = torch.stack(route_logit_trace, dim=2)
            stacked_probs = torch.stack(route_probability_trace, dim=2)
            stacked_weights = torch.stack(route_weight_trace, dim=2)
            route_indices = torch.stack(route_index_trace, dim=2)
            if return_aux or return_trace:
                route_logits = stacked_logits
            if return_aux:
                route_probs = stacked_probs
                route_weights = stacked_weights
                flat_probs = stacked_probs.reshape(-1, 4)
                flat_indices = route_indices.reshape(-1)
                selected_fraction = torch.nn.functional.one_hot(
                    flat_indices, num_classes=4
                ).to(flat_probs.dtype).mean(dim=0).detach()
                load_stats = {
                    "selected_fraction": selected_fraction,
                    "mean_probability": flat_probs.mean(dim=0),
                }
        return NativeOutput(
            logits=logits,
            route_logits=route_logits,
            state_logits=torch.stack(state_trace, dim=2) if return_aux else None,
            cache=new_cache,
            route_indices=route_indices,
            route_probs=route_probs,
            route_weights=route_weights,
            load_stats=load_stats,
            h=torch.stack(hidden_trace, dim=2) if return_trace else None,
            slots=torch.stack(slot_trace, dim=2) if return_trace else None,
        )

    def generate(self, input_ids: Tensor, max_new_tokens: int, **kwargs: Any) -> Tensor:
        from .generation import greedy_generate

        return greedy_generate(self, input_ids, max_new_tokens=max_new_tokens, **kwargs)
