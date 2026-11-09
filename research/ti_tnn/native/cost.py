"""Exact stored-parameter and matrix-multiply accounting for native models."""

from __future__ import annotations

from .config import MODES, NativeConfig


def dense_transform_parameters(config: NativeConfig) -> int:
    d = config.d_model
    return 3 * d * config.d_ff + config.d_ff + d


def dense_transform_macs(config: NativeConfig) -> int:
    return 3 * config.d_model * config.d_ff


def attention_parameters(config: NativeConfig) -> int:
    d = config.d_model
    return 4 * d * d + 6 * d


def parameter_breakdown(config: NativeConfig) -> dict[str, int]:
    """Count each unique trainable tensor once, including all stored experts."""

    d = config.d_model
    d_s = config.d_slot
    attention_cells = 4 if config.mode == "dense" else 1
    transforms = 4 if config.mode in ("dense", "untyped", "typed") else 1
    typed = config.mode == "typed"
    routed = config.mode in ("untyped", "typed")
    return {
        "token_embedding": config.vocab_size * d,
        "position_embedding": config.context_length * d,
        "slot_initializers": 4 * d * d_s,
        "attention_cells": attention_cells * attention_parameters(config),
        "dense_transforms": transforms * dense_transform_parameters(config),
        "h_reader_writer": 2 * d * d,
        "slot_readers_writers": (12 if typed else 3) * d * d_s,
        "slot_codes": 4 * d if typed else 0,
        "router": 4 * ((2 if typed else 1) * d + 1) if routed else 0,
        "state_probes": 4 * d_s * config.n_state_classes,
        "output_norm": 2 * d,
        "output_head": config.vocab_size * d,
    }


def parameter_counts(config: NativeConfig) -> dict[str, int]:
    """Return total storage and the nominal one-expert active parameter set.

    Active parameters count the embedding tables and output head by their stored
    shapes. For routed arms, one dense expert replaces the four-expert bank;
    repeated use of shared weights does not multiply this parameter count.
    """

    breakdown = parameter_breakdown(config)
    total = sum(breakdown.values())
    active = total
    if config.mode in ("untyped", "typed"):
        active -= 3 * dense_transform_parameters(config)
    return {"total": total, "active": active}


def total_parameters(config: NativeConfig) -> int:
    return parameter_counts(config)["total"]


def active_parameters(config: NativeConfig) -> int:
    return parameter_counts(config)["active"]


def count_model_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def mac_breakdown(
    config: NativeConfig,
    sequence_length: int = 1,
    cache_length: int = 0,
    *,
    include_probes: bool = False,
    incremental: bool = False,
) -> dict[str, int]:
    """Count MACs performed by the implementation for new tokens.

    Full execution evaluates a padded T-by-(cache+T) attention matrix. With
    ``incremental=True``, tokens are decoded one at a time and the actual key
    lengths are cache+1 through cache+T. Masked-away attention entries in the
    full matrix are therefore intentionally included in full execution work.
    """

    if sequence_length <= 0 or cache_length < 0:
        raise ValueError("sequence_length must be positive and cache_length nonnegative")
    d = config.d_model
    d_s = config.d_slot
    tokens = sequence_length
    if incremental:
        attention_positions = tokens * cache_length + tokens * (tokens + 1) // 2
    else:
        attention_positions = tokens * (cache_length + tokens)
    routed = config.mode in ("untyped", "typed")
    typed = config.mode == "typed"
    return {
        "slot_initializers": tokens * 4 * d * d_s,
        "attention_projections": 4 * tokens * 4 * d * d,
        "attention_scores_and_values": 4 * 2 * d * attention_positions,
        "transforms": 4 * tokens * dense_transform_macs(config),
        "routers": 4 * tokens * 4 * ((2 if typed else 1) * d) if routed else 0,
        "h_readers_writers": 4 * tokens * 2 * d * d,
        "slot_readers_writers": 4 * tokens * (12 if typed else 3) * d * d_s,
        "state_probes": 4 * tokens * 4 * d_s * config.n_state_classes if include_probes else 0,
        "output_head": tokens * d * config.vocab_size,
    }


def executed_macs(
    config: NativeConfig,
    sequence_length: int = 1,
    cache_length: int = 0,
    *,
    include_probes: bool = False,
    incremental: bool = False,
) -> int:
    return sum(
        mac_breakdown(
            config,
            sequence_length,
            cache_length,
            include_probes=include_probes,
            incremental=incremental,
        ).values()
    )


def token_macs(config: NativeConfig, context_length: int, *, include_probes: bool = False) -> int:
    """MACs for one decoded token attending to ``context_length`` keys."""

    if context_length <= 0:
        raise ValueError("context_length must be positive")
    return executed_macs(
        config,
        sequence_length=1,
        cache_length=context_length - 1,
        include_probes=include_probes,
        incremental=True,
    )


def counts_for_all_modes(config: NativeConfig, context_length: int | None = None) -> dict[str, dict[str, int]]:
    from dataclasses import replace

    context_length = context_length or config.context_length
    return {
        mode: {
            **parameter_counts(replace(config, mode=mode)),
            "decoded_token_macs": token_macs(replace(config, mode=mode), context_length),
            "decoded_token_macs_with_probes": token_macs(
                replace(config, mode=mode), context_length, include_probes=True
            ),
        }
        for mode in MODES
    }
