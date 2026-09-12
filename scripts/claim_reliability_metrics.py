"""NumPy-only analysis of confidence thresholds fixed before test evaluation.

A claim selects the class with largest predicted probability. Its confidence is
that probability, and all examples tied at a calibrated threshold are claimed.
Calibration targets empirical coverage; it promises neither test coverage nor a
bound on the probability of a wrong claim. MSE is mean squared coordinate error.
The multiclass Brier score sums classwise squared errors before averaging inputs.

The paired bootstrap resamples input indices, keeping every fitted model's
prediction on the same sampled input together. It conditions on the supplied
fits and frozen calibration policies; it does not quantify training-population
or calibration-sample uncertainty. Its primary statistic averages the conditional
wrong-claim rates of the individual fits, rather than pooling their claims.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any

import numpy as np

DEFAULT_COVERAGES = (0.25, 0.5, 0.75, 0.9)
_Z95 = 1.959963984540054


def _float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf':
        raise ValueError(f'{name} must contain real numbers')
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} must be finite')
    return array


def _probabilities(value: Any, name: str) -> np.ndarray:
    array = _float_array(value, name)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError(f'{name} must have nonempty shape [inputs, classes>=2]')
    if (array < 0).any() or (array > 1).any():
        raise ValueError(f'{name} must lie in [0, 1]')
    if not np.allclose(array.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError(f'{name} rows must sum to one within 1e-6')
    return array


def _coverage(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('coverage must be a real number in [0, 1]')
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError('coverage must be a real number in [0, 1]')
    return result


def coverage_key(value: float) -> str:
    """Canonical JSON key for a target coverage, for example ``'0.75'``."""
    return format(_coverage(value), '.12g')


def _coverages(values: Sequence[float]) -> list[float]:
    result = [_coverage(value) for value in values]
    if not result or len({coverage_key(value) for value in result}) != len(result):
        raise ValueError('coverages must be nonempty and have distinct canonical keys')
    return result


def fit_policies(calibration_probabilities: Any,
                 coverages: Sequence[float] = DEFAULT_COVERAGES) -> dict:
    """Fit only to calibration scores, with inclusive ties and no labels.

    For target c>0 and n calibration inputs, the cutoff is the ceil(c*n)-th
    largest maximum class probability. It is the largest cutoff accepting at
    least that many calibration inputs. At c=0, threshold=None claims nothing.
    Returned dictionaries contain plain Python values and can be frozen as JSON.
    """
    probability = _probabilities(calibration_probabilities, 'calibration_probabilities')
    confidence = probability.max(axis=1)
    ranked = np.sort(confidence)[::-1]
    policies = {}
    for target in _coverages(coverages):
        threshold = float(ranked[math.ceil(target * len(ranked)) - 1]) if target else None
        claimed_count = int(np.count_nonzero(confidence >= threshold)) if target else 0
        policies[coverage_key(target)] = {
            'target_coverage': target,
            'threshold': threshold,
            'calibration_count': len(confidence),
            'calibration_claimed_count': claimed_count,
            'calibration_coverage': claimed_count / len(confidence),
        }
    return policies


def wilson95(wrong_count: int, claimed_count: int) -> list[float] | None:
    """Descriptive Wilson binomial interval for one fit's accepted test inputs.

    This conditions on that fit and its fixed threshold. It is not a simultaneous
    interval over fits or coverage levels, and it excludes calibration uncertainty.
    """
    if (isinstance(wrong_count, (bool, np.bool_))
            or isinstance(claimed_count, (bool, np.bool_))
            or not isinstance(wrong_count, (int, np.integer))
            or not isinstance(claimed_count, (int, np.integer))
            or claimed_count < 0 or not 0 <= wrong_count <= claimed_count):
        raise ValueError('counts must be integers with 0 <= wrong <= claimed')
    if not claimed_count:
        return None
    proportion = wrong_count / claimed_count
    denominator = 1 + _Z95**2 / claimed_count
    center = (proportion + _Z95**2 / (2 * claimed_count)) / denominator
    radius = (_Z95 * math.sqrt(proportion * (1 - proportion) / claimed_count
                              + _Z95**2 / (4 * claimed_count**2)) / denominator)
    return [0.0 if wrong_count == 0 else max(0.0, center - radius),
            1.0 if wrong_count == claimed_count else min(1.0, center + radius)]


def _test_arrays(test_probabilities: Any, test_estimates: Any,
                 truth: Any, labels: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = _probabilities(test_probabilities, 'test_probabilities')
    estimate = _float_array(test_estimates, 'test_estimates')
    target = _float_array(truth, 'truth')
    n = len(probability)
    if (target.ndim != 2 or target.shape[0] != n or target.shape[1] < 1
            or estimate.shape != target.shape):
        raise ValueError('test_estimates and truth must have identical shape [inputs, coordinates>=1]')
    label = np.asarray(labels)
    if label.dtype.kind not in 'iu' or label.shape != (n,):
        raise ValueError('labels must be an integer vector of shape [inputs]')
    if (label < 0).any() or (label >= probability.shape[1]).any():
        raise ValueError('labels are outside the class range')
    with np.errstate(over='ignore', invalid='ignore'):
        squared_error = np.square(estimate - target).mean(axis=1)
    if not np.isfinite(squared_error).all():
        raise ValueError('derived squared error must be finite')
    return probability, label, squared_error


def _finite_mean(values: np.ndarray, name: str) -> float:
    with np.errstate(over='ignore', invalid='ignore'):
        mean = float(values.mean())
    if not math.isfinite(mean):
        raise ValueError(f'derived {name} must be finite')
    return mean


def analyze_model(calibration_probabilities: Any, test_probabilities: Any,
                  test_estimates: Any, truth: Any, labels: Any,
                  coverages: Sequence[float] = DEFAULT_COVERAGES, *,
                  policies: Mapping | None = None) -> dict:
    """Report one frozen fit on one common test cohort, without changing policy.

    If policies are supplied, they must exactly equal fitting the supplied
    calibration scores. The runner must persist these policies before generating
    or evaluating test data; this function can verify equality but not chronology.
    ``wrong_claim_rate`` is None when there are no claims, never a claimed zero.
    """
    calibration = _probabilities(calibration_probabilities, 'calibration_probabilities')
    probability, label, squared_error = _test_arrays(test_probabilities, test_estimates, truth, labels)
    if calibration.shape[1] != probability.shape[1]:
        raise ValueError('calibration and test probabilities must have the same class count')
    fitted = fit_policies(calibration, coverages)
    if policies is not None and (not isinstance(policies, Mapping) or dict(policies) != fitted):
        raise ValueError('frozen policies differ from calibration-only policies')
    n, classes = probability.shape
    wrong = probability.argmax(axis=1) != label
    one_hot = np.eye(classes, dtype=np.float64)[label]
    confidence = probability.max(axis=1)
    coverage_results = {}
    for key, policy in fitted.items():
        claimed = (confidence >= policy['threshold'] if policy['threshold'] is not None
                   else np.zeros(n, dtype=bool))
        count = int(claimed.sum())
        errors = int((claimed & wrong).sum())
        coverage_results[key] = {
            **policy,
            'test_count': n,
            'claimed_count': count,
            'abstained_count': n - count,
            'wrong_claim_count': errors,
            'coverage': count / n,
            'wrong_claim_rate': errors / count if count else None,
            'wrong_claim_rate_all_inputs': errors / n,
            'wrong_claim_rate_wilson95': wilson95(errors, count),
        }
    return {
        'n_calibration': len(calibration), 'n_test': n, 'n_classes': classes,
        'top_accuracy': float((~wrong).mean()),
        'mse': _finite_mean(squared_error, 'mse'),
        'brier_score': float(np.square(probability - one_hot).sum(axis=1).mean()),
        'nll': float(-np.log(np.maximum(probability[np.arange(n), label], 1e-15)).mean()),
        'coverage_results': coverage_results,
    }


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _bootstrap_arrays(values: Mapping, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(values, Mapping) or set(values) != {'claimed', 'wrong', 'squared_error'}:
        raise ValueError(f'{name} must contain claimed, wrong, squared_error')
    claimed = np.asarray(values['claimed'])
    wrong = np.asarray(values['wrong'])
    error = _float_array(values['squared_error'], f'{name}.squared_error')
    if (claimed.dtype.kind != 'b' or wrong.dtype.kind != 'b' or claimed.ndim != 2
            or min(claimed.shape) < 1 or claimed.shape != wrong.shape or claimed.shape != error.shape):
        raise ValueError(f'{name} arrays must share nonempty [fits, inputs] shape and Boolean masks')
    if (error < 0).any():
        raise ValueError('squared_error cannot be negative')
    return claimed, wrong, error


def paired_bootstrap(left: Mapping, right: Mapping, *, n_resamples: int = 2000,
                     seed: int = 43000, chunk_size: int = 32) -> dict:
    """Bootstrap left minus right, pairing the same input across all fixed fits.

    Input mappings contain Boolean ``claimed`` and ``wrong`` masks and per-input
    ``squared_error``, each [fits, inputs]. The wrong mask describes top-class
    errors for all inputs; its intersection with claimed supplies the numerator.
    Resamples use multinomial counts (equivalent to drawing n input indices with
    replacement); counts are generated in bounded chunks. They never resample
    fits. Percentile endpoints use NumPy's linear quantile interpolation.

    The wrong-claim statistic is mean_f(errors_f / claims_f). A resample with
    no claims for any fit makes it undefined. We report the number of such draws
    and omit that interval if any occur, rather than silently dropping draws.
    """
    repetitions = _positive_integer(n_resamples, 'n_resamples')
    chunk = min(_positive_integer(chunk_size, 'chunk_size'), repetitions, 64)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    left_claimed, left_wrong, left_error = _bootstrap_arrays(left, 'left')
    right_claimed, right_wrong, right_error = _bootstrap_arrays(right, 'right')
    if left_claimed.shape != right_claimed.shape:
        raise ValueError('paired models must have identical [fits, inputs] shape and fit order')
    fits, n = left_claimed.shape
    claimed = np.concatenate((left_claimed, right_claimed)).astype(np.float64)
    erroneous = np.concatenate((left_claimed & left_wrong, right_claimed & right_wrong)).astype(np.float64)
    # Averaging linear per-input differences before resampling preserves the
    # declared mean-over-fits statistic and avoids unnecessary bootstrap storage.
    with np.errstate(over='ignore', invalid='ignore'):
        per_input_mse = (left_error - right_error).mean(axis=0)
    if not np.isfinite(per_input_mse).all():
        raise ValueError('derived per-input MSE difference must be finite')
    per_input_coverage = (left_claimed.astype(np.float64) - right_claimed).mean(axis=0)
    original_claims = claimed.sum(axis=1)
    original_errors = erroneous.sum(axis=1)
    point_risk = None
    if (original_claims > 0).all():
        rates = original_errors / original_claims
        point_risk = float(rates[:fits].mean() - rates[fits:].mean())
    risk_draws = np.full(repetitions, np.nan)
    mse_draws = np.empty(repetitions)
    coverage_draws = np.empty(repetitions)
    rng = np.random.default_rng(int(seed))
    sampling_probabilities = np.full(n, 1 / n, dtype=np.float64)
    for start in range(0, repetitions, chunk):
        stop = min(start + chunk, repetitions)
        counts = rng.multinomial(n, sampling_probabilities, size=stop - start).astype(np.float64)
        denominator = counts @ claimed.T
        numerator = counts @ erroneous.T
        valid = (denominator > 0).all(axis=1)
        rates = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        difference = rates[:, :fits].mean(axis=1) - rates[:, fits:].mean(axis=1)
        risk_draws[start:stop] = np.where(valid, difference, np.nan)
        with np.errstate(over='ignore', invalid='ignore'):
            mse_draws[start:stop] = (counts / n) @ per_input_mse
        coverage_draws[start:stop] = (counts / n) @ per_input_coverage
    if not np.isfinite(mse_draws).all() or not np.isfinite(coverage_draws).all():
        raise ValueError('derived bootstrap MSE and coverage must be finite')

    def interval(point: float | None, draws: np.ndarray) -> dict:
        defined = int(np.isfinite(draws).sum())
        bounds = np.quantile(draws, [0.025, 0.975], method='linear').tolist() if defined == repetitions else None
        return {'estimate': point, 'percentile95': bounds,
                'defined_resamples': defined, 'undefined_resamples': repetitions - defined}

    return {
        'n_fits': fits, 'n_unique_test_inputs': n, 'n_resamples': repetitions, 'seed': int(seed),
        'resampling_unit': 'input; same multinomial counts across both models and every fixed fit',
        'conditioning': 'fixed fitted models and fixed calibration policies; no fit or calibration resampling',
        'statistic': 'left minus right; mean of per-fit conditional wrong-claim rates',
        'interval_method': 'percentile 2.5 and 97.5; linear quantiles; no multiplicity correction',
        'wrong_claim_rate_difference': interval(point_risk, risk_draws),
        'mse_difference': interval(_finite_mean(per_input_mse, 'MSE difference'), mse_draws),
        'coverage_difference': interval(float(per_input_coverage.mean()), coverage_draws),
    }


def _variation(values: Sequence[float | None]) -> dict:
    finite = [float(value) for value in values if value is not None]
    complete = len(finite) == len(values)
    return {
        'per_fit': list(values), 'defined_fits': len(finite), 'total_fits': len(values),
        'mean': float(np.mean(finite)) if complete else None,
        'minimum': min(finite) if complete else None,
        'maximum': max(finite) if complete else None,
        'sample_sd': float(np.std(finite, ddof=1)) if complete and len(finite) > 1 else None,
    }


def aggregate_models(models: Mapping, truth: Any, labels: Any, *,
                     coverages: Sequence[float] = DEFAULT_COVERAGES,
                     primary_coverage: float = 0.75, n_resamples: int = 2000,
                     seed: int = 43000, chunk_size: int = 32,
                     left_model: str = 'shared', right_model: str = 'fixed_depth_4') -> dict:
    """Analyze model -> ordered frozen-fit payloads and compare the primary pair.

    Each payload has calibration_probabilities, test_probabilities, test_estimates
    and optionally policies. The caller must keep fit order aligned across the
    primary pair and use exactly the same truth/labels/input order for all models.
    A deterministic analytic reference may appear once instead of being copied
    across five fits. All model variations are descriptive, not intervals over a
    training population. The primary uncertainty calculation compares only the
    named left and right models.
    """
    targets = _coverages(coverages)
    primary_key = coverage_key(primary_coverage)
    if _coverage(primary_coverage) not in targets:
        raise ValueError('primary_coverage must be among coverages')
    if not isinstance(models, Mapping) or not models or left_model not in models or right_model not in models:
        raise ValueError('models must include both named primary models')
    if left_model == right_model:
        raise ValueError('primary model names must differ')
    summaries, paired_inputs = {}, {}
    for model_name, payloads in models.items():
        if not isinstance(model_name, str) or not model_name:
            raise ValueError('model names must be nonempty strings')
        if not isinstance(payloads, Sequence) or isinstance(payloads, (str, bytes)) or not payloads:
            raise ValueError('each model needs a nonempty ordered fit sequence')
        rows = []
        masks, wrong_masks, errors = [], [], []
        for payload in payloads:
            if not isinstance(payload, Mapping) or not {
                'calibration_probabilities', 'test_probabilities', 'test_estimates'
            } <= set(payload):
                raise ValueError('fit payload lacks calibration probabilities or test predictions')
            row = analyze_model(payload['calibration_probabilities'], payload['test_probabilities'],
                                payload['test_estimates'], truth, labels, targets,
                                policies=payload.get('policies'))
            rows.append(row)
            if model_name in (left_model, right_model):
                probability, label, error = _test_arrays(payload['test_probabilities'],
                                                       payload['test_estimates'], truth, labels)
                threshold = row['coverage_results'][primary_key]['threshold']
                masks.append(probability.max(axis=1) >= threshold if threshold is not None
                             else np.zeros(len(probability), dtype=bool))
                wrong_masks.append(probability.argmax(axis=1) != label)
                errors.append(error)
        summaries[model_name] = {
            'n_fits': len(rows), 'per_fit': rows,
            'descriptive_fit_variation': {
                **{key: _variation([row[key] for row in rows])
                   for key in ('mse', 'top_accuracy', 'brier_score', 'nll')},
                'coverage_results': {
                    key: {metric: _variation([row['coverage_results'][key][metric] for row in rows])
                          for metric in ('coverage', 'wrong_claim_rate')}
                    for key in (coverage_key(target) for target in targets)
                },
            },
        }
        if model_name in (left_model, right_model):
            paired_inputs[model_name] = {'claimed': np.stack(masks), 'wrong': np.stack(wrong_masks),
                                         'squared_error': np.stack(errors)}
    paired = paired_bootstrap(paired_inputs[left_model], paired_inputs[right_model],
                              n_resamples=n_resamples, seed=seed, chunk_size=chunk_size)
    return {'models': summaries, 'primary_comparison': {
        'left_model': left_model, 'right_model': right_model,
        'target_coverage': float(primary_coverage), **paired,
    }}
