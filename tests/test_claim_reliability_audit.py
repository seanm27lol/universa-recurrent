"""Independent quality arithmetic, corrupted data, and bootstrap accounting."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_claim_reliability.py'
SPEC = importlib.util.spec_from_file_location('claim_quality_audit', SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture_arrays():
    return dict(cal=np.array([[.9, .1], [.8, .2], [.8, .2], [.5, .5]]),
                test=np.array([[.9, .1], [.8, .2], [.3, .7], [.1, .9]]),
                estimate=np.tile(np.arange(4.)[:, None], (1, 5)),
                truth=np.zeros((4, 5)), label=np.array([0, 1, 1, 0]))


def test_inclusive_order_statistic_preserves_ties():
    result = audit.policies(fixture_arrays()['cal'])
    assert result['0.5']['threshold'] == .8
    assert result['0.5']['calibration_claimed_count'] == 3
    assert result['0.5']['calibration_coverage'] == .75
    assert result['0.9']['threshold'] == .5


def test_known_metrics_and_distinct_error_denominators():
    result, _ = audit.row_metrics(**fixture_arrays())
    selected = result['coverage_results']['0.75']
    assert selected['claimed_count'] == 3
    assert selected['wrong_claim_count'] == 2
    assert selected['wrong_claim_rate'] == 2 / 3
    assert selected['wrong_claim_rate_all_inputs'] == .5
    assert result['top_accuracy'] == .5
    assert result['mse'] == 3.5
    assert result['brier_score'] == pytest.approx((.02 + 1.28 + .18 + 1.62) / 4)
    assert result['nll'] == pytest.approx(-np.log([.9, .2, .7, .1]).mean())


@pytest.mark.parametrize('key', ['threshold', 'wrong_claim_count', 'wrong_claim_rate'])
def test_changed_claim_metric_is_detected(key):
    result, _ = audit.row_metrics(**fixture_arrays())
    expected = result['coverage_results']['0.75']
    changed = dict(expected)
    changed[key] += 1e-9 if key == 'threshold' else 1
    state = audit.Audit()
    state.compare(expected, changed, 'model.coverage', 'metrics')
    assert len(state.mismatches) == 1
    assert state.mismatches[0]['path'].endswith('.' + key)


@pytest.mark.parametrize('labels', [np.array([0., 1., 1., 0.]), np.array([0, -1, 1, 0]),
                                  np.array([0, 2, 1, 0]), np.array([[0, 1, 1, 0]])])
def test_invalid_labels_are_rejected(labels):
    data = fixture_arrays()
    data['label'] = labels
    with pytest.raises(ValueError, match='labels'):
        audit.row_metrics(**data)


def test_wrong_truth_shape_and_overflow_are_rejected():
    data = fixture_arrays()
    data['truth'] = np.zeros((4, 1))
    with pytest.raises(ValueError, match='shape'):
        audit.row_metrics(**data)
    data = fixture_arrays()
    data['estimate'][:] = 1e308
    with pytest.raises(ValueError, match='derived squared errors'):
        audit.row_metrics(**data)


def test_no_claims_and_incomplete_fit_mean_stay_undefined():
    data = fixture_arrays()
    data['cal'][:] = [.99, .01]
    result, _ = audit.row_metrics(**data)
    assert result['coverage_results']['0.75']['claimed_count'] == 0
    assert result['coverage_results']['0.75']['wrong_claim_rate'] is None
    result = audit.variation([None, .2])
    assert result['defined_fits'] == 1
    for key in ['mean', 'minimum', 'maximum', 'sample_sd']:
        assert result[key] is None
    json.dumps(result, allow_nan=False)


def test_zero_error_wilson_upper_bound_is_not_zero():
    assert audit.wilson(0, 100)[0] == 0
    assert audit.wilson(0, 100)[1] == pytest.approx(.03699349820698568)
    assert audit.wilson(0, 0) is None


def payload(claimed, wrong, losses):
    return {'claimed': np.array(claimed, dtype=bool),
            'wrong': np.array(wrong, dtype=bool), 'loss': np.array(losses, dtype=float)}


def test_bootstrap_resamples_same_inputs_and_averages_conditional_ratios():
    left = [payload([1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 3, 2]),
            payload([1, 1, 1, 0], [0, 0, 0, 1], [2, 1, 3, 4])]
    right = [payload([1, 1, 1, 1], [0, 1, 0, 1], [1, 1, 2, 3]),
             payload([1, 1, 1, 1], [0, 0, 0, 0], [0, 2, 1, 2])]
    results, draws = audit.bootstrap({'shared': left, 'fixed_depth_4': right}, repeats=40, seed=7)
    # Mean of left fit risks is (1/1 + 0/3)/2=.5, not pooled1/4=.25.
    assert results['wrong_claim_rate_difference']['estimate'] == .25
    rng = np.random.default_rng(7)
    for i in range(40):
        indices = np.repeat(np.arange(4), rng.multinomial(4, [.25] * 4))
        differences = []
        for l, r in zip(left, right):
            ratios = []
            for fit in [l, r]:
                claim = fit['claimed'][indices]
                ratios.append(np.count_nonzero(claim & fit['wrong'][indices]) / claim.sum()
                              if claim.any() else None)
            differences.append(ratios[0] - ratios[1] if None not in ratios else None)
        expected = float(np.mean(differences)) if None not in differences else None
        assert draws['wrong_claim_rate_difference'][i] == pytest.approx(expected) if expected is not None else draws['wrong_claim_rate_difference'][i] is None
        expected_mse = np.mean([np.mean(l['loss'][indices] - r['loss'][indices]) for l, r in zip(left, right)])
        assert draws['mse_difference'][i] == pytest.approx(expected_mse)
    assert results['wrong_claim_rate_difference']['undefined_resamples'] > 0
    assert results['wrong_claim_rate_difference']['percentile95'] is None
    json.dumps(draws, allow_nan=False)


def test_identical_paired_predictions_have_zero_bootstrap_difference():
    value = [payload([1, 1], [1, 0], [.2, .8])]
    results, draws = audit.bootstrap({'shared': value, 'fixed_depth_4': copy.deepcopy(value)}, repeats=7)
    for key, result in results.items():
        assert result['estimate'] == 0
        assert result['percentile95'] == [0, 0]
        assert draws[key] == [0] * 7


def test_bootstrap_all_empty_claims_are_json_null():
    value = [payload([0, 0], [1, 0], [.2, .8])]
    result, draws = audit.bootstrap({'shared': value, 'fixed_depth_4': value}, repeats=5)
    risk = result['wrong_claim_rate_difference']
    assert risk['estimate'] is None
    assert risk['percentile95'] is None
    assert risk['undefined_resamples'] == 5
    assert draws['wrong_claim_rate_difference'] == [None] * 5
    json.dumps((result, draws), allow_nan=False)


def test_comparison_state_does_not_leak_between_runs():
    first, second = audit.Audit(), audit.Audit()
    first.compare(1, 2, 'bad', 'metric')
    second.compare(1, 1, 'good', 'metric')
    assert first.mismatches
    assert not second.mismatches
    assert dict(second.counts) == {'metric': 1}
    assert second.differences['metric'] == 0


def npz_bytes(**arrays):
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def test_numeric_npz_round_trip_requires_exact_inventory():
    raw = npz_bytes(x=np.arange(3))
    assert np.array_equal(audit.read_arrays(raw, {'x': (3,)})['x'], np.arange(3))
    with pytest.raises(ValueError, match='inventory'):
        audit.read_arrays(raw, {'y': (3,)})


@pytest.mark.parametrize('value', [np.array(['unsafe'], dtype=object), np.array([np.nan])])
def test_object_or_nonfinite_npz_is_rejected(value):
    with pytest.raises(ValueError, match='invalid'):
        audit.read_arrays(npz_bytes(x=value), {'x': (1,)})


def test_forged_npy_header_rejected_before_loading():
    npy = io.BytesIO()
    np.lib.format.write_array_header_1_0(npy, {'shape': (10**12,), 'fortran_order': False, 'descr': '<f8'})
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, 'w') as archive:
        archive.writestr('x.npy', npy.getvalue())
    with pytest.raises(ValueError, match='shape'):
        audit.read_arrays(raw.getvalue(), {'x': (1,)})


def test_duplicate_json_key_rejected():
    with pytest.raises(ValueError, match='duplicate JSON'):
        json.loads('{"threshold": 0.5, "threshold": 0.9}', object_pairs_hook=audit.object_pairs)


def test_outer_archive_rejects_extra_or_missing_members(tmp_path):
    source = tmp_path / 'report.zip'
    with zipfile.ZipFile(source, 'w') as archive:
        archive.writestr('results/summary.json', '{}')
    with pytest.raises(ValueError, match='inventory'):
        audit.audit_archive(source)


def test_cli_errors_cleanly_and_preserves_existing_outputs(tmp_path, capsys):
    missing = tmp_path / 'missing.zip'
    output = tmp_path / 'output'
    output.mkdir()
    marker = output / 'keep.txt'
    marker.write_text('preserve me')
    assert audit.main([str(missing), str(output)]) == 2
    assert 'already exists' in capsys.readouterr().err
    assert marker.read_text() == 'preserve me'
    assert audit.main([str(missing), str(tmp_path / 'new-output')]) == 2
    assert 'Audit error:' in capsys.readouterr().err
    assert not (tmp_path / 'new-output').exists()
