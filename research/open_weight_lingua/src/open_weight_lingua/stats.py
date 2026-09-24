"""Whole-group bootstrap estimates for the Section 8 decision rules.

Resampling whole problem groups keeps paired prompt variants and conditions
together; variants within one group are not independent samples. These
percentile intervals are uncertainty statements conditional on one checkpoint,
task, and site, not preregistered results. Ratio-style "fraction recovered"
metrics are never computed here: near-zero donor effects invalidate them, so
editing contrasts are absolute log-probability differences only.
"""

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
import math
import random
import statistics
from typing import TypeVar

T = TypeVar("T")

DEFAULT_RESAMPLES = 3000
_TWO_SIDED_TAIL = 0.025
_ONE_SIDED_TAIL = 0.05


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile interval; its width reflects the group count, not variants."""

    point: float
    lower: float
    upper: float
    resamples: int
    seed: int


@dataclass(frozen=True)
class UpperEstimate:
    """Point estimate with a one-sided 95% upper bound, used for losses."""

    point: float
    upper: float
    groups: int
    resamples: int
    seed: int


@dataclass(frozen=True)
class LowerEstimate:
    """Point estimate with a one-sided 95% lower bound, used for gains."""

    point: float
    lower: float
    groups: int
    resamples: int
    seed: int


@dataclass(frozen=True)
class LogProbPair:
    """Full-sequence log probabilities of the two candidate answers under a patch."""

    counterfactual: float
    original: float

    def preference(self) -> float:
        """L = log P(counterfactual answer) - log P(original answer)."""
        return self.counterfactual - self.original


@dataclass(frozen=True)
class EditingRecord:
    """One group's answer log probabilities under each reconstruction condition."""

    unedited: LogProbPair
    edited: LogProbPair
    wrong_variable: LogProbPair | None = None
    donor: LogProbPair | None = None


@dataclass(frozen=True)
class EditingContrast:
    """Absolute paired effect of one contrast condition versus unedited."""

    eligible_groups: int
    point: float
    lower: float
    upper: float


@dataclass(frozen=True)
class EditingEffectEstimate:
    groups: int
    mean_l_unedited: float
    mean_l_edited: float
    effect: BootstrapInterval
    wrong_variable: EditingContrast | None
    donor: EditingContrast | None


@dataclass(frozen=True)
class BlockEstimates:
    per_block: dict[Hashable, float]
    pooled: float
    block_sizes: dict[Hashable, int]


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(sorted_values[low])
    fraction = position - low
    return float(
        sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction
    )


def _binary(values: Sequence[bool]) -> list[int]:
    result = []
    for value in values:
        if value not in (0, 1):
            raise ValueError("correctness must be encoded per group as 0/1 or bool")
        result.append(int(value))
    if not result:
        raise ValueError("at least one group is required")
    return result


def _finite(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError("at least one group is required")
    if not all(math.isfinite(value) for value in result):
        raise ValueError("values must be finite; failures are encoded by the caller")
    return result


def _interval(
    values: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    resamples: int,
    seed: int,
    tail: float,
) -> tuple[float, float, float]:
    groups = list(values)
    if not groups:
        raise ValueError("at least one group is required")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    point = float(statistic(groups))
    rng = random.Random(seed)
    size = len(groups)
    estimates = []
    for _ in range(resamples):
        # random.Random(seed).random() is reproducible across CPython versions.
        resampled = [groups[int(rng.random() * size)] for _ in range(size)]
        estimates.append(float(statistic(resampled)))
    if not math.isfinite(point) or not all(map(math.isfinite, estimates)):
        raise ValueError("statistic must return finite values")
    estimates.sort()
    return point, _quantile(estimates, tail), _quantile(estimates, 1.0 - tail)


def group_bootstrap(
    group_values: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int,
) -> BootstrapInterval:
    """Two-sided 95% percentile interval, resampling whole groups with replacement.

    `statistic` receives each resampled group multiset: groups may repeat, and
    a group's paired values always move together. The point estimate is the
    statistic on the original groups. Deterministic under the fixed seed.
    """
    point, lower, upper = _interval(
        group_values, statistic, resamples=resamples, seed=seed, tail=_TWO_SIDED_TAIL
    )
    return BootstrapInterval(point, lower, upper, resamples, seed)


def paired_accuracy_loss(
    p0_correct: Sequence[bool],
    condition_correct: Sequence[bool],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int,
) -> UpperEstimate:
    """Paired exact-answer accuracy loss of a condition versus P0, in points.

    Intention-to-test contract: the caller has already encoded every missing or
    failed generation as False and supplies each group exactly once in both
    sequences. This function never drops or reweights groups, so the
    denominator is always all groups. The bound is the one-sided 95% upper
    percentile estimate the preservation rule compares against 5 points.
    """
    reference, condition = _binary(p0_correct), _binary(condition_correct)
    if len(reference) != len(condition):
        raise ValueError("paired sequences must cover the same groups")
    losses = [a - b for a, b in zip(reference, condition)]

    def mean_loss_points(sample: Sequence[int]) -> float:
        return 100.0 * statistics.fmean(sample)

    point, _, upper = _interval(
        losses,
        mean_loss_points,
        resamples=resamples,
        seed=seed,
        tail=_ONE_SIDED_TAIL,
    )
    return UpperEstimate(point, upper, len(losses), resamples, seed)


def paired_logprob_difference(
    p2_logprob: Sequence[float],
    p3_logprob: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int,
) -> LowerEstimate:
    """Paired correct-answer log-probability difference (P2 minus P3).

    Positive values favor P2; the P2-beats-P3 criterion requires this one-sided
    95% lower percentile estimate to be positive. Non-finite inputs are
    rejected rather than silently excluded: how a failed reconstruction or
    generation is encoded remains the caller's explicit, reported decision.
    """
    p2, p3 = _finite(p2_logprob), _finite(p3_logprob)
    if len(p2) != len(p3):
        raise ValueError("paired sequences must cover the same groups")
    differences = [a - b for a, b in zip(p2, p3)]
    point, lower, _ = _interval(
        differences,
        statistics.fmean,
        resamples=resamples,
        seed=seed,
        tail=_ONE_SIDED_TAIL,
    )
    return LowerEstimate(point, lower, len(differences), resamples, seed)


def editing_effect(
    records: Sequence[EditingRecord],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int,
) -> EditingEffectEstimate:
    """Paired text-edit effect on counterfactual answer preference.

    Per group and condition, L = log P(counterfactual) - log P(original); the
    edit effect is L(edited) - L(unedited), bootstrapped over whole groups with
    a two-sided 95% interval. Wrong-variable-edit and raw-donor contrasts are
    computed identically over the groups eligible for each. All effects are
    absolute; no ratio or "fraction recovered" value is ever formed. The shared
    seed makes resample indices identical across equal-sized contrasts.
    """
    records = list(records)
    if not records:
        raise ValueError("at least one group is required")
    for record in records:
        for pair in (
            record.unedited,
            record.edited,
            record.wrong_variable,
            record.donor,
        ):
            if pair is not None and not (
                math.isfinite(pair.counterfactual) and math.isfinite(pair.original)
            ):
                raise ValueError("answer log probabilities must be finite")
    l_unedited = [record.unedited.preference() for record in records]
    l_edited = [record.edited.preference() for record in records]
    effect = group_bootstrap(
        [edited - unedited for edited, unedited in zip(l_edited, l_unedited)],
        statistics.fmean,
        resamples=resamples,
        seed=seed,
    )

    def eligible_pairs(
        select: Callable[[EditingRecord], LogProbPair | None],
    ) -> list[tuple[LogProbPair, LogProbPair]]:
        return [
            (pair, record.unedited)
            for record in records
            if (pair := select(record)) is not None
        ]

    def contrast(pairs: list[tuple[LogProbPair, LogProbPair]]) -> EditingContrast | None:
        if not pairs:
            return None
        result = group_bootstrap(
            [choice.preference() - base.preference() for choice, base in pairs],
            statistics.fmean,
            resamples=resamples,
            seed=seed,
        )
        return EditingContrast(len(pairs), result.point, result.lower, result.upper)

    return EditingEffectEstimate(
        groups=len(records),
        mean_l_unedited=statistics.fmean(l_unedited),
        mean_l_edited=statistics.fmean(l_edited),
        effect=effect,
        wrong_variable=contrast(eligible_pairs(lambda record: record.wrong_variable)),
        donor=contrast(eligible_pairs(lambda record: record.donor)),
    )


def block_estimates(
    values: Sequence[float],
    block_labels: Sequence[Hashable],
    *,
    statistic: Callable[[Sequence[float]], float] = statistics.fmean,
) -> BlockEstimates:
    """Per-block point estimates plus the pooled estimate over all groups.

    Locked validation reports two 256-group blocks and requires each block's
    point estimate in the intended direction alongside the pooled criterion;
    the pilot reports pooled numbers. The pooled value applies the statistic to
    every group once, so blocks never reweight groups; with equal block sizes
    it equals the average of the per-block values.
    """
    values = _finite(values)
    if len(values) != len(block_labels):
        raise ValueError("each group needs exactly one block label")
    blocks: dict[Hashable, list[float]] = {}
    for value, label in zip(values, block_labels):
        blocks.setdefault(label, []).append(value)
    return BlockEstimates(
        per_block={label: float(statistic(group)) for label, group in blocks.items()},
        pooled=float(statistic(values)),
        block_sizes={label: len(group) for label, group in blocks.items()},
    )
