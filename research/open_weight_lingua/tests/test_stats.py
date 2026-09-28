import math
import statistics

import pytest

from open_weight_lingua.stats import (
    DEFAULT_RESAMPLES,
    EditingRecord,
    LogProbPair,
    block_estimates,
    editing_effect,
    group_bootstrap,
    paired_accuracy_loss,
    paired_logprob_difference,
)


def _record(l_unedited, l_edited, l_wrong=None, l_donor=None):
    def pair(preference):
        if preference is None:
            return None
        return LogProbPair(counterfactual=preference - 1.0, original=-1.0)

    return EditingRecord(
        unedited=pair(l_unedited),
        edited=pair(l_edited),
        wrong_variable=pair(l_wrong),
        donor=pair(l_donor),
    )


def test_default_resamples_matches_brief():
    assert DEFAULT_RESAMPLES == 3000


def test_group_bootstrap_constant_effect_collapses_interval():
    result = group_bootstrap([2.5] * 20, statistics.fmean, resamples=500, seed=11)
    assert result.point == result.lower == result.upper == 2.5
    assert result.resamples == 500 and result.seed == 11


def test_group_bootstrap_zero_effect_straddles_zero():
    result = group_bootstrap(
        [1.0, -1.0] * 12, statistics.fmean, resamples=1000, seed=5
    )
    assert result.point == 0.0
    assert result.lower < 0.0 < result.upper


def test_group_bootstrap_fixed_seed_reproduces_interval():
    values = [math.sin(i * 1.7) * 3.0 for i in range(30)]
    first = group_bootstrap(values, statistics.fmean, resamples=1000, seed=99)
    second = group_bootstrap(values, statistics.fmean, resamples=1000, seed=99)
    assert first == second
    different = group_bootstrap(values, statistics.fmean, resamples=1000, seed=100)
    assert different.point == first.point
    assert (different.lower, different.upper) != (first.lower, first.upper)


def test_group_bootstrap_units_are_groups_not_variants():
    means = [3.0] * 4 + [-3.0] * 4
    grouped = [(m, m, m, m) for m in means]

    def flattened_mean(sample):
        return statistics.fmean(value for group in sample for value in group)

    group_level = group_bootstrap(grouped, flattened_mean, resamples=2000, seed=17)
    scalar_level = group_bootstrap(means, statistics.fmean, resamples=2000, seed=17)
    # Identical within-group variants add no information to the interval.
    assert group_level.point == pytest.approx(scalar_level.point)
    assert group_level.lower == pytest.approx(scalar_level.lower)
    assert group_level.upper == pytest.approx(scalar_level.upper)
    # Resampling the 32 variants as independent halves the width: wrong here.
    variants = [value for m in means for value in (m, m, m, m)]
    variant_level = group_bootstrap(variants, statistics.fmean, resamples=2000, seed=17)
    group_width = group_level.upper - group_level.lower
    variant_width = variant_level.upper - variant_level.lower
    assert group_width > 1.5 * variant_width


def test_group_bootstrap_rejects_bad_inputs():
    with pytest.raises(ValueError, match="at least one group"):
        group_bootstrap([], statistics.fmean, resamples=10, seed=1)
    with pytest.raises(ValueError, match="resamples"):
        group_bootstrap([1.0], statistics.fmean, resamples=0, seed=1)
    with pytest.raises(ValueError, match="finite"):
        group_bootstrap(
            [1.0, 2.0], lambda sample: float("nan"), resamples=10, seed=1
        )


def test_paired_accuracy_loss_upper_bound_and_intention_to_test():
    p0 = [True] * 16
    # Failed generations are already encoded as incorrect by the caller; the
    # denominator stays all 16 groups either way.
    condition = [True] * 12 + [False] * 4
    result = paired_accuracy_loss(p0, condition, resamples=2000, seed=23)
    assert result.point == pytest.approx(25.0)
    assert result.groups == 16
    perfect = paired_accuracy_loss(p0, p0, resamples=2000, seed=23)
    assert perfect.point == perfect.upper == 0.0
    half = paired_accuracy_loss(p0, [True, False] * 8, resamples=2000, seed=23)
    assert half.point == pytest.approx(50.0)
    assert half.upper > half.point


def test_paired_accuracy_loss_input_validation():
    with pytest.raises(ValueError, match="same groups"):
        paired_accuracy_loss([True, False], [True], resamples=10, seed=1)
    with pytest.raises(ValueError, match="0/1"):
        paired_accuracy_loss([True, 0.5], [True, False], resamples=10, seed=1)
    with pytest.raises(ValueError, match="at least one group"):
        paired_accuracy_loss([], [], resamples=10, seed=1)


def test_paired_logprob_difference_lower_bound_orientation():
    p2 = [-0.2, -0.4, -0.1, -0.3, -0.5, -0.2] * 4
    p3 = [-2.0, -1.5, -2.5, -1.0, -1.8, -2.2] * 4
    result = paired_logprob_difference(p2, p3, resamples=2000, seed=31)
    assert result.point > 0.0
    assert 0.0 < result.lower <= result.point
    symmetric = paired_logprob_difference(
        [1.0, -1.0] * 12, [0.0] * 24, resamples=2000, seed=31
    )
    assert symmetric.point == 0.0
    assert symmetric.lower < 0.0


def test_paired_logprob_difference_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        paired_logprob_difference(
            [0.0, float("-inf")], [0.0, 0.0], resamples=10, seed=1
        )
    with pytest.raises(ValueError, match="same groups"):
        paired_logprob_difference([0.0], [0.0, 0.0], resamples=10, seed=1)


def test_editing_effect_constant_effects_collapse_intervals():
    records = [_record(-4.0, -0.5, l_wrong=-3.5, l_donor=-3.5) for _ in range(12)]
    result = editing_effect(records, resamples=500, seed=41)
    assert result.groups == 12
    assert result.mean_l_unedited == pytest.approx(-4.0)
    assert result.mean_l_edited == pytest.approx(-0.5)
    assert result.effect.point == result.effect.lower == result.effect.upper == 3.5
    assert result.wrong_variable is not None
    assert result.wrong_variable.eligible_groups == 12
    assert (
        result.wrong_variable.point
        == result.wrong_variable.lower
        == result.wrong_variable.upper
        == 0.5
    )
    assert result.donor is not None
    assert result.donor.point == result.donor.lower == result.donor.upper == 0.5


def test_editing_effect_zero_effect_straddles_zero():
    records = []
    for i in range(20):
        base = -4.0 + (i % 5) * 0.3
        records.append(_record(base, base + (0.4 if i % 2 == 0 else -0.4)))
    result = editing_effect(records, resamples=2000, seed=43)
    assert result.effect.point == pytest.approx(0.0, abs=1e-9)
    assert result.effect.lower < 0.0 < result.effect.upper
    assert result.wrong_variable is None and result.donor is None


def test_editing_effect_eligibility_subsets_and_absolute_donor():
    records = []
    for i in range(12):
        wrong = None if i < 2 else -3.5
        records.append(_record(-4.0, -0.5, l_wrong=wrong, l_donor=-4.0))
    result = editing_effect(records, resamples=500, seed=47)
    assert result.wrong_variable is not None
    assert result.wrong_variable.eligible_groups == 10
    assert result.donor is not None and result.donor.eligible_groups == 12
    # A donor effect of exactly zero stays zero and finite; nothing is divided.
    assert result.donor.point == result.donor.lower == result.donor.upper == 0.0
    assert math.isfinite(result.donor.point)


def test_editing_effect_rejects_non_finite_and_empty():
    with pytest.raises(ValueError, match="at least one group"):
        editing_effect([], resamples=10, seed=1)
    bad = EditingRecord(
        unedited=LogProbPair(counterfactual=float("nan"), original=-1.0),
        edited=LogProbPair(counterfactual=-2.0, original=-1.0),
    )
    with pytest.raises(ValueError, match="finite"):
        editing_effect([bad], resamples=10, seed=1)


def test_block_estimates_partition_and_group_weighted_pooled():
    values = [0.0, 10.0, 20.0, 30.0]
    labels = ["validation_a", "validation_a", "validation_b", "validation_b"]
    result = block_estimates(values, labels)
    assert result.per_block == {"validation_a": 5.0, "validation_b": 25.0}
    assert result.pooled == 15.0
    assert result.block_sizes == {"validation_a": 2, "validation_b": 2}
    uneven = block_estimates([0.0, 10.0, 20.0], ["a", "a", "b"])
    assert uneven.per_block == {"a": 5.0, "b": 20.0}
    # Pooled weighs every group equally; it is not the average of block values.
    assert uneven.pooled == pytest.approx(10.0)


def test_block_estimates_validation():
    with pytest.raises(ValueError, match="exactly one block label"):
        block_estimates([1.0, 2.0], ["a"])
    with pytest.raises(ValueError, match="at least one group"):
        block_estimates([], [])
    with pytest.raises(ValueError, match="finite"):
        block_estimates([float("nan")], ["a"])


def test_pooled_criterion_with_per_block_point_estimates():
    p0 = [True] * 512
    condition = [i % 32 != 0 for i in range(512)]
    labels = ["validation_a"] * 256 + ["validation_b"] * 256
    pooled = paired_accuracy_loss(p0, condition, resamples=1000, seed=53)
    assert pooled.point == pytest.approx(100.0 * 16 / 512)
    assert pooled.upper >= pooled.point
    losses = [float(a) - float(c) for a, c in zip(p0, condition)]
    blocks = block_estimates(losses, labels)
    assert blocks.per_block["validation_a"] == pytest.approx(8 / 256)
    assert blocks.per_block["validation_b"] == pytest.approx(8 / 256)
    assert blocks.pooled == pytest.approx(16 / 512)
