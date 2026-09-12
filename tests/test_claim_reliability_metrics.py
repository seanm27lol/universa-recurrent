"""Calibration ties, risk denominators, and input-paired statistical checks."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
metrics = importlib.import_module('claim_reliability_metrics')


def probabilities(confidences):
    p = np.asarray(confidences, dtype=np.float64)
    return np.stack((p, 1 - p), axis=-1)


def fixture_arrays():
    return {
        'calibration_probabilities': probabilities([.9, .8, .7, .6]),
        'test_probabilities': np.array([[.9, .1], [.7, .3], [.4, .6], [.2, .8]]),
        'test_estimates': np.array([[1., 0.], [0., 2.], [3., 0.], [0., 0.]]),
        'truth': np.zeros((4, 2)),
        'labels': np.array([0, 1, 1, 0]),
    }


def test_threshold_is_largest_inclusive_cutoff_and_keeps_all_ties():
    result = metrics.fit_policies(probabilities([.95, .8, .8, .6, .5]), [0, .4, .5, 1])
    assert result['0']['threshold'] is None
    assert result['0']['calibration_claimed_count'] == 0
    # ceil(.4*5)=2, but the second-highest score ties the third.
    assert result['0.4']['threshold'] == .8
    assert result['0.4']['calibration_claimed_count'] == 3
    assert result['0.4']['calibration_coverage'] == .6
    assert result['0.5']['calibration_claimed_count'] == 3
    assert result['1']['calibration_claimed_count'] == 5
    assert result['1']['threshold'] == .5


def test_frozen_threshold_survives_changed_test_scores_and_labels():
    arrays = fixture_arrays()
    policies = metrics.fit_policies(arrays['calibration_probabilities'], [.75])
    frozen = copy.deepcopy(policies)
    first = metrics.analyze_model(**arrays, coverages=[.75], policies=policies)
    arrays['labels'] = 1 - arrays['labels']
    arrays['test_probabilities'] = probabilities([.99, .99, .99, .99])
    second = metrics.analyze_model(**arrays, coverages=[.75], policies=policies)
    assert policies == frozen
    assert first['coverage_results']['0.75']['threshold'] == second['coverage_results']['0.75']['threshold'] == .7
    assert second['coverage_results']['0.75']['coverage'] == 1


def test_altered_frozen_policy_is_rejected():
    arrays = fixture_arrays()
    policies = metrics.fit_policies(arrays['calibration_probabilities'], [.75])
    policies['0.75']['threshold'] = .6
    with pytest.raises(ValueError, match='frozen policies'):
        metrics.analyze_model(**arrays, coverages=[.75], policies=policies)


def test_metrics_match_hand_calculation():
    arrays = fixture_arrays()
    row = metrics.analyze_model(**arrays, coverages=[.75])
    claims = row['coverage_results']['0.75']
    assert row['top_accuracy'] == .5
    assert row['mse'] == 14 / 8
    assert row['brier_score'] == pytest.approx((.02 + .98 + .32 + 1.28) / 4)
    assert row['nll'] == pytest.approx(-np.log([.9, .3, .6, .2]).mean())
    assert claims['claimed_count'] == 3
    assert claims['wrong_claim_count'] == 2
    assert claims['coverage'] == .75
    assert claims['wrong_claim_rate'] == 2 / 3
    assert claims['wrong_claim_rate_all_inputs'] == .5
    assert claims['wrong_claim_rate_wilson95'][0] < 2 / 3 < claims['wrong_claim_rate_wilson95'][1]
    json.dumps(row, allow_nan=False)


def test_no_claims_is_undefined_risk_not_zero():
    arrays = fixture_arrays()
    arrays['calibration_probabilities'] = probabilities([.99] * 4)
    row = metrics.analyze_model(**arrays, coverages=[0, .75])
    for claim_row in row['coverage_results'].values():
        assert claim_row['claimed_count'] == claim_row['wrong_claim_count'] == 0
        assert claim_row['wrong_claim_rate'] is None
        assert claim_row['wrong_claim_rate_wilson95'] is None
        assert claim_row['wrong_claim_rate_all_inputs'] == 0


def test_nll_clips_zero_true_class_probability_and_ties_choose_first_class():
    arrays = fixture_arrays()
    arrays['test_probabilities'] = np.array([[1., 0.], [.5, .5], [.5, .5], [0., 1.]])
    arrays['labels'] = np.array([1, 0, 1, 1])
    row = metrics.analyze_model(**arrays)
    assert row['top_accuracy'] == .5
    assert row['nll'] == pytest.approx(-(np.log(1e-15) + 2 * np.log(.5)) / 4)


@pytest.mark.parametrize('bad', [np.zeros((0, 2)), [0.5, 0.5], [[1.]],
                                [[.6, .6]], [[-.1, 1.1]], [[np.nan, .5]],
                                [[np.inf, 0]], [[True, False]], [['.5', '.5']],
                                [[.5 + 0j, .5]]])
def test_rejects_invalid_probabilities(bad):
    with pytest.raises(ValueError):
        metrics.fit_policies(bad)


def test_probability_normalization_has_absolute_tolerance_without_renormalizing():
    accepted = np.array([[.8, .2000009]])
    assert metrics.fit_policies(accepted, [1])['1']['threshold'] == .8
    with pytest.raises(ValueError, match='sum to one'):
        metrics.fit_policies([[.8, .2000011]])


@pytest.mark.parametrize('bad', [[], [.5, .5], [-.1], [1.1], [True], [float('nan')], ['.5']])
def test_rejects_invalid_coverages(bad):
    with pytest.raises(ValueError):
        metrics.fit_policies([[.8, .2]], bad)


@pytest.mark.parametrize(('field', 'value'), [
    ('truth', np.zeros(8)), ('truth', np.zeros((4, 0))),
    ('test_estimates', np.zeros((3, 2))), ('test_estimates', np.full((4, 2), np.inf)),
    ('labels', np.array([0., 1., 0., 1.])), ('labels', np.array([[0], [1], [0], [1]])),
    ('labels', np.array([0, 1, 0, 2])), ('labels', np.array([0, 1, 0, -1])),
    ('labels', np.array([False] * 4)),
    ('calibration_probabilities', np.ones((4, 3)) / 3),
])
def test_rejects_invalid_prediction_shapes_labels_and_values(field, value):
    arrays = fixture_arrays()
    arrays[field] = value
    with pytest.raises(ValueError):
        metrics.analyze_model(**arrays)


def test_rejects_overflowing_derived_squared_error():
    arrays = fixture_arrays()
    arrays['test_estimates'] = np.full((4, 2), 1e308)
    with pytest.raises(ValueError, match='derived squared error'):
        metrics.analyze_model(**arrays)


def test_wilson_known_interval_and_extremes():
    assert metrics.wilson95(5, 10) == pytest.approx([.236593090512564, .763406909487436])
    assert metrics.wilson95(0, 10)[0] == 0
    assert metrics.wilson95(10, 10)[1] == 1
    assert metrics.wilson95(0, 0) is None
    with pytest.raises(ValueError):
        metrics.wilson95(11, 10)
    with pytest.raises(ValueError):
        metrics.wilson95(True, 10)


def bootstrap_payload(claimed, wrong, error):
    return {'claimed': np.asarray(claimed, dtype=bool),
            'wrong': np.asarray(wrong, dtype=bool),
            'squared_error': np.asarray(error, dtype=float)}


def test_identical_predictions_have_exact_zero_paired_intervals():
    same = bootstrap_payload([[1] * 6] * 2, [[0, 1, 1, 0, 1, 0]] * 2,
                             [[1, 4, 2, 8, 3, 6]] * 2)
    result = metrics.paired_bootstrap(same, same, n_resamples=100, seed=2)
    for key in ['wrong_claim_rate_difference', 'mse_difference', 'coverage_difference']:
        assert result[key]['estimate'] == 0
        assert result[key]['percentile95'] == [0, 0]
        assert result[key]['undefined_resamples'] == 0


def test_input_pairing_with_different_claim_subsets_matches_manual_multinomial():
    left = bootstrap_payload([[1, 1, 0, 1], [1, 0, 1, 1]],
                             [[1, 0, 1, 1], [0, 1, 1, 0]],
                             [[1, 2, 3, 4], [4, 3, 2, 1]])
    right = bootstrap_payload([[0, 1, 1, 1], [1, 1, 0, 1]],
                              [[1, 1, 0, 0], [0, 1, 0, 1]],
                              [[4, 2, 1, 0], [1, 1, 2, 4]])
    count = 500
    weights = np.random.default_rng(12).multinomial(4, [.25] * 4, size=count)
    values = {'wrong_claim_rate_difference': [], 'mse_difference': [], 'coverage_difference': []}
    for w in weights:
        left_denominator = (left['claimed'] * w).sum(axis=1)
        right_denominator = (right['claimed'] * w).sum(axis=1)
        if (left_denominator > 0).all() and (right_denominator > 0).all():
            lr = ((left['claimed'] & left['wrong']) * w).sum(axis=1) / left_denominator
            rr = ((right['claimed'] & right['wrong']) * w).sum(axis=1) / right_denominator
            values['wrong_claim_rate_difference'].append(lr.mean() - rr.mean())
        values['mse_difference'].append(((left['squared_error'] - right['squared_error']) * w).mean())
        values['coverage_difference'].append(((left['claimed'].astype(float) - right['claimed']) * w).mean())
    result = metrics.paired_bootstrap(left, right, n_resamples=count, seed=12, chunk_size=7)
    for key, draws in values.items():
        assert result[key]['defined_resamples'] == len(draws)
        expected = np.quantile(draws, [.025, .975]).tolist() if len(draws) == count else None
        if expected is None:
            assert result[key]['percentile95'] is None
        else:
            assert result[key]['percentile95'] == pytest.approx(expected)


def test_wrong_claim_rate_averages_fit_ratios_instead_of_pooling_claims():
    left = bootstrap_payload([[1, 0, 0, 0], [1, 1, 1, 1]],
                             [[1, 0, 0, 0], [0, 0, 0, 0]], np.zeros((2, 4)))
    right = bootstrap_payload(np.ones((2, 4)), np.zeros((2, 4)), np.zeros((2, 4)))
    result = metrics.paired_bootstrap(left, right, n_resamples=100, seed=3)
    assert result['wrong_claim_rate_difference']['estimate'] == .5  # pooled would be .2
    assert result['wrong_claim_rate_difference']['undefined_resamples'] > 0
    assert result['wrong_claim_rate_difference']['percentile95'] is None
    assert result['coverage_difference']['estimate'] == -.375
    assert result['coverage_difference']['percentile95'] is not None


def test_bootstrap_reproducibility_chunk_independence_and_direction():
    left = bootstrap_payload(np.ones((2, 8)), np.ones((2, 8)), np.ones((2, 8)) * 2)
    right = bootstrap_payload(np.ones((2, 8)), np.zeros((2, 8)), np.ones((2, 8)))
    first = metrics.paired_bootstrap(left, right, n_resamples=113, seed=44, chunk_size=1)
    second = metrics.paired_bootstrap(left, right, n_resamples=113, seed=44, chunk_size=13)
    assert first == second
    assert first['wrong_claim_rate_difference']['percentile95'] == [1, 1]
    assert first['mse_difference']['percentile95'] == [1, 1]
    reverse = metrics.paired_bootstrap(right, left, n_resamples=113, seed=44)
    assert reverse['wrong_claim_rate_difference']['percentile95'] == [-1, -1]
    json.dumps(first, allow_nan=False)


def test_zero_original_claims_keeps_primary_point_and_interval_undefined():
    left = bootstrap_payload([[0] * 4], [[1] * 4], [[0] * 4])
    right = bootstrap_payload([[1] * 4], [[0] * 4], [[0] * 4])
    result = metrics.paired_bootstrap(left, right, n_resamples=10)
    assert result['wrong_claim_rate_difference'] == {
        'estimate': None, 'percentile95': None, 'defined_resamples': 0, 'undefined_resamples': 10}


@pytest.mark.parametrize(('keyword', 'value'), [('seed', -1), ('seed', True),
                                              ('n_resamples', 0), ('n_resamples', 1.5),
                                              ('chunk_size', False), ('chunk_size', 0)])
def test_invalid_bootstrap_controls(keyword, value):
    same = bootstrap_payload([[1] * 2], [[0] * 2], [[0] * 2])
    with pytest.raises(ValueError):
        metrics.paired_bootstrap(same, same, **{keyword: value})


def test_invalid_bootstrap_arrays_and_unpaired_shapes():
    same = bootstrap_payload([[1] * 2], [[0] * 2], [[0] * 2])
    changed = copy.deepcopy(same)
    changed['claimed'] = changed['claimed'].astype(int)
    with pytest.raises(ValueError, match='Boolean'):
        metrics.paired_bootstrap(changed, same)
    changed = copy.deepcopy(same)
    changed['squared_error'][0, 0] = -1
    with pytest.raises(ValueError, match='negative'):
        metrics.paired_bootstrap(changed, same)
    changed = bootstrap_payload([[1] * 2] * 2, [[0] * 2] * 2, [[0] * 2] * 2)
    with pytest.raises(ValueError, match='identical'):
        metrics.paired_bootstrap(changed, same)


def test_aggregate_preserves_single_reference_and_fixed_fit_counts():
    arrays = fixture_arrays()
    payload = {key: arrays[key] for key in ('calibration_probabilities', 'test_probabilities', 'test_estimates')}
    payload['policies'] = metrics.fit_policies(arrays['calibration_probabilities'])
    models = {'shared': [payload, copy.deepcopy(payload)],
              'fixed_depth_4': [copy.deepcopy(payload), copy.deepcopy(payload)],
              'gaussian': [copy.deepcopy(payload)]}
    result = metrics.aggregate_models(models, arrays['truth'], arrays['labels'], n_resamples=30)
    assert result['models']['gaussian']['n_fits'] == 1
    assert result['models']['gaussian']['descriptive_fit_variation']['mse']['sample_sd'] is None
    assert result['models']['shared']['n_fits'] == 2
    assert result['primary_comparison']['n_fits'] == 2
    assert result['primary_comparison']['n_unique_test_inputs'] == 4
    assert result['primary_comparison']['mse_difference']['percentile95'] == [0, 0]
    assert result['models']['shared']['descriptive_fit_variation']['coverage_results']['0.75']['coverage']['mean'] == .75
    json.dumps(result, allow_nan=False)


def test_aggregate_rejects_missing_primary_and_primary_not_in_sweep():
    arrays = fixture_arrays()
    with pytest.raises(ValueError, match='both named'):
        metrics.aggregate_models({}, arrays['truth'], arrays['labels'])
    with pytest.raises(ValueError, match='among coverages'):
        metrics.aggregate_models({}, arrays['truth'], arrays['labels'], coverages=[.5])


def test_aggregate_undefined_fit_risk_is_not_silently_averaged_away():
    arrays = fixture_arrays()
    payload = {key: arrays[key] for key in ('calibration_probabilities', 'test_probabilities', 'test_estimates')}
    no_claims = copy.deepcopy(payload)
    no_claims['calibration_probabilities'] = probabilities([.99] * 4)
    result = metrics.aggregate_models({'shared': [payload, no_claims], 'fixed_depth_4': [payload, payload]},
                                     arrays['truth'], arrays['labels'], n_resamples=10)
    variation = result['models']['shared']['descriptive_fit_variation']['coverage_results']['0.75']['wrong_claim_rate']
    assert variation['defined_fits'] == 1
    assert variation['total_fits'] == 2
    assert variation['mean'] is None
    assert result['primary_comparison']['wrong_claim_rate_difference']['estimate'] is None


def test_rejects_overflowing_mse_reduction():
    arrays = fixture_arrays()
    arrays['truth'] = np.zeros((4, 1))
    arrays['test_estimates'] = np.full((4, 1), 1e154)
    with pytest.raises(ValueError, match='derived mse'):
        metrics.analyze_model(**arrays)
