"""Behavioral scores; these measure outcomes, not semantic truth."""

import re
import torch


def exact_integer(text: str, expected: str) -> bool:
    # No stripping: extra whitespace, commentary and leading zeros are failures.
    return bool(re.fullmatch(r"(?:0|[1-9][0-9]*)", text)) and text == expected


def answer_tokens(tokenizer, answer: str) -> list[int]:
    """Continuation is separately tokenized answer bytes, followed by EOS.

    We append these IDs to the frozen chat prefix, without retokenizing it.
    This convention includes answer termination and supports multi-token numbers.
    """
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)", answer):
        raise ValueError("answer must be a canonical nonnegative integer")
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ids or tokenizer.decode(ids, skip_special_tokens=False) != answer:
        raise ValueError("answer tokenization does not round trip")
    if tokenizer.eos_token_id is None:
        raise ValueError("EOS is required for full-answer scoring")
    return [*ids, tokenizer.eos_token_id]


def sequence_log_probability(
    logits, prefix_length: int, answer_ids: list[int]
) -> float:
    if prefix_length < 1 or not answer_ids or logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("invalid teacher-forcing inputs")
    positions = logits[
        0, prefix_length - 1 : prefix_length + len(answer_ids) - 1
    ].float()
    if len(positions) != len(answer_ids) or not torch.isfinite(positions).all():
        raise ValueError("missing or nonfinite answer logits")
    target = torch.tensor(answer_ids, device=logits.device)
    return float(positions.log_softmax(-1).gather(1, target[:, None]).sum())


def next_token_kl(reference_logits, intervention_logits) -> float:
    """KL(P0 || intervention), full vocabulary, float32 log-softmax."""
    if reference_logits.shape != intervention_logits.shape:
        raise ValueError("vocabulary shapes differ")
    if (
        not torch.isfinite(reference_logits).all()
        or not torch.isfinite(intervention_logits).all()
    ):
        raise ValueError("KL requires finite logits")
    p, q = (
        reference_logits.float().log_softmax(-1),
        intervention_logits.float().log_softmax(-1),
    )
    return float((p.exp() * (p - q)).sum())
