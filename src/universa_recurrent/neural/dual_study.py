"""Matched-output experiment: same weights, data, coverage rule, and timed output.

This is a controlled next experiment, not a claim that another architecture won.
Models are inherited from v2; all run to their trained depth. The structural
claim is a separate output and cannot truncate or replace the numerical estimate.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math
import platform
import random
import subprocess
import time

import numpy as np
import torch

from .. import __version__
from .data import StructuredFlowDataset
from .dual_output import (ClaimPolicy, FixedEstimator, MODES, construct_output,
                          fit_coverage_policy, metrics, positive_int, validate_candidates)
from .train import load_checkpoint, sha256_file
from .v2_train import load_v2_checkpoint

FORMAT = 'universa-recurrent.dual-study.v1'


def write_json(path: Path, payload: dict) -> None:
    """Exclusive creation prevents silently replacing previous evidence."""
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        handle.write(text)


def read_json(path: Path) -> dict:
    def reject(value):
        raise ValueError(f'nonfinite JSON literal: {value}')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON field: {key}')
            result[key] = value
        return result
    obj = json.loads(path.read_text('utf-8'), parse_constant=reject, object_pairs_hook=pairs)
    if not isinstance(obj, dict):
        raise ValueError('JSON record must be an object')
    return obj


def _seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('seeds must be nonnegative integers')
    return value


def reserved_seeds(metadata: dict, legacy: bool = False) -> set[int]:
    train = metadata.get('raw_metadata', {}).get('training') if legacy else metadata.get('training')
    if not isinstance(train, dict):
        raise ValueError('missing training provenance: cannot claim a held-out comparison')
    keys = ('train_seed', 'validation_seed') if legacy else ('train_seed', 'calibration_seed')
    if any(key not in train for key in keys):
        raise ValueError('checkpoint lacks explicit data-split seeds')
    return {_seed(train[key]) for key in keys}


def dataset_fingerprint(dataset: StructuredFlowDataset) -> str:
    digest = hashlib.sha256()
    for name in ('observed', 'mask', 'truth', 'label'):
        array = getattr(dataset, name).numpy()
        digest.update(name.encode() + str(array.shape).encode() + str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_identity() -> dict:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0')
        digest.update(path.read_bytes())
    try:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {'package_version': __version__, 'python_source_sha256': digest.hexdigest(),
            'git_commit': commit, 'python': platform.python_version(),
            'numpy': np.__version__, 'torch': str(torch.__version__)}


def load_estimators(checkpoint: Path, device_name: str,
                    v1_checkpoint: Path | None = None) -> tuple[list[FixedEstimator], dict, torch.device, set[int]]:
    main, controls, metadata, device = load_v2_checkpoint(checkpoint, device_name=device_name)
    if not {'direct', 'untied', 'ambient'} <= set(controls):
        raise ValueError('Matched comparison needs direct, untied, and ambient controls; train v2 without --no-controls')
    settings = metadata.get('training')
    if not isinstance(settings, dict):
        raise ValueError('missing training metadata')
    noise = float(settings['noise_std'])
    observe = float(settings['observe_probability'])
    if not math.isfinite(noise) or noise <= 0 or not math.isfinite(observe) or not 0 < observe <= 1:
        raise ValueError('invalid stored data settings')
    sha = str(metadata['checkpoint_sha256'])
    engines = [FixedEstimator('shared', main, main.bases, 'shared', noise, sha)]
    for name, model in sorted(controls.items()):
        family = 'dedicated' if name.startswith('fixed_depth_') else name
        if family not in ('dedicated', 'direct', 'untied', 'ambient'):
            raise ValueError(f'unrecognized comparison model: {name}')
        engines.append(FixedEstimator(name, model, main.bases, family, noise, sha))
    used = reserved_seeds(metadata)
    if v1_checkpoint is not None:
        old, old_meta, old_device = load_checkpoint(v1_checkpoint, device_name=device_name)
        old_settings = old_meta.get('raw_metadata', {}).get('training', {})
        if (old_settings.get('noise_std') != noise or old_settings.get('observe_probability') != observe
                or not torch.equal(old.bases, main.bases) or old_device != device):
            raise ValueError('v1 and v2 must share the same library and observation/noise settings')
        used |= reserved_seeds(old_meta, legacy=True)
        engines.append(FixedEstimator('v1_all_candidates_diagnostic', old, old.bases, 'v1', noise,
                                      str(old_meta['checkpoint_sha256'])))
    engines.extend([FixedEstimator('gaussian_generator_reference', None, main.bases, 'gaussian', noise, sha),
                    FixedEstimator('fit_each_structure', None, main.bases, 'fit_each_structure', noise, sha)])
    return engines, metadata, device, used


def _resident(dataset: StructuredFlowDataset, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(getattr(dataset, name).to(device) for name in ('observed', 'mask', 'truth', 'label'))


@torch.inference_mode()
def collect(engine: FixedEstimator, observed: torch.Tensor, mask: torch.Tensor,
            policy: ClaimPolicy | None, mode: str, batch_size: int) -> dict[str, torch.Tensor]:
    """The exact same function supplies evaluation and timed predictions.

    Includes Python batching, model inference, output construction and tensor
    concatenation. Does not include labels, metrics, transfers, logging or checks.
    """
    positive_int(batch_size, 'batch_size')
    parts: dict[str, list[torch.Tensor]] = {}
    for start in range(0, observed.shape[0], batch_size):
        prediction = engine.predict(observed[start:start+batch_size], mask[start:start+batch_size], policy, mode)
        for key, value in prediction.items():
            parts.setdefault(key, []).append(value)
    if not parts:
        raise ValueError('cannot predict an empty batch')
    return {key: torch.cat(values, 0) for key, values in parts.items()}


def calibrate(engines: list[FixedEstimator], dataset: StructuredFlowDataset,
              device: torch.device, coverage: float, batch_size: int) -> tuple[dict[str, ClaimPolicy], dict]:
    observed, mask, truth, labels = _resident(dataset, device)
    policies, reports = {}, {}
    for engine in engines:
        if not engine.structured:
            continue
        outputs = collect(engine, observed, mask, ClaimPolicy(None), 'mixture', batch_size)
        policy = fit_coverage_policy(outputs['probabilities'], coverage)
        policies[engine.name] = policy
        checked = construct_output(outputs['candidate_states'], outputs['probabilities'], policy)
        reports[engine.name] = {'policy': policy.as_dict(),
            'calibration_metrics': metrics(checked, truth, labels, 'mixture_with_claim')}
    return policies, {'format': FORMAT + '.calibration',
        'data': {'seed': dataset.seed, 'n': len(dataset), 'sha256': dataset_fingerprint(dataset),
                 **asdict(dataset.spec)},
        'rule': 'largest inclusive score cutoff achieving target calibration coverage; labels not used to select cutoff',
        'correctness_guarantee': False, 'test_coverage_guarantee': False,
        'models': reports}


def _sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def cases(engines: list[FixedEstimator]) -> list[tuple[FixedEstimator, str]]:
    result = []
    for engine in engines:
        modes = MODES if engine.structured else ('estimate_only',) if engine.family == 'ambient' else ('hard_reference',)
        result.extend((engine, mode) for mode in modes)
    return result


def _row_key(engine: FixedEstimator, mode: str) -> str:
    return engine.name + '/' + mode


def evaluate_and_time(engines: list[FixedEstimator], policies: dict[str, ClaimPolicy],
                      dataset: StructuredFlowDataset, device: torch.device, *, batch_size: int,
                      warmup: int, repeats: int, order_seed: int) -> tuple[dict, dict]:
    """Warm, interleave cases, and report median/quantiles rather than one average.

    The same test cohort, batch shape, and function are used for quality and
    latency. Zero repeats disables timing for small correctness-only runs.
    """
    positive_int(batch_size, 'batch_size')
    for value, name in ((warmup, 'warmup'), (repeats, 'repeats')):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    _seed(order_seed)
    observed, mask, truth, labels = _resident(dataset, device)
    all_cases = cases(engines)
    predictions, quality, errors, timings = {}, {}, {}, {}
    for engine, mode in all_cases:
        key = _row_key(engine, mode)
        out = collect(engine, observed, mask, policies.get(engine.name), mode, batch_size)
        quality[key] = {**engine.description(), **metrics(out, truth, labels, mode)}
        errors[key] = (out['estimate'].double() - truth.double()).square().mean(-1).cpu().numpy()
        # Avoid retaining whole CUDA output histories across all cases.
        predictions[key] = {field: value.detach().cpu() for field, value in out.items()}
        timings[key] = []
        del out
        if repeats:
            for _ in range(warmup):
                collect(engine, observed, mask, policies.get(engine.name), mode, batch_size)
    _sync(device)
    order = list(range(len(all_cases)))
    rng = random.Random(order_seed)
    repeat_orders = []
    for repeat in range(repeats):
        rng.shuffle(order)
        repeat_orders.append([_row_key(*all_cases[i]) for i in order])
        for index in order:
            engine, mode = all_cases[index]
            key = _row_key(engine, mode)
            _sync(device)
            started = time.perf_counter_ns()
            out = collect(engine, observed, mask, policies.get(engine.name), mode, batch_size)
            _sync(device)
            elapsed = (time.perf_counter_ns() - started) / 1_000_000
            timings[key].append(elapsed)
            # This comparison occurs after timing; don't pair untimed mixture
            # quality with a timed hard-output path.
            for field, value in out.items():
                actual, expected = value.detach().cpu(), predictions[key][field]
                equal = (torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
                         if actual.is_floating_point() else torch.equal(actual, expected))
                if not equal:
                    raise RuntimeError(f'{key}/{field}: timed output differs from evaluated output')
            del out
    rows = list(quality.values())
    paired = []
    if 'shared/mixture' in errors:
        for key in sorted(errors):
            if key.endswith('/mixture') and key != 'shared/mixture':
                delta = errors['shared/mixture'] - errors[key]
                paired.append({'comparison': 'shared/mixture minus ' + key,
                    'mean_paired_mse_difference': float(delta.mean()),
                    'test_example_difference_sd': float(delta.std(ddof=1)) if len(delta) > 1 else None,
                    'n_paired_examples': len(delta),
                    'note': 'One trained-model pair, not independent training replication or a significance test.'})
    timed_rows = []
    for key, values in timings.items():
        if values:
            med = float(np.median(values))
            timed_rows.append({**quality[key], 'median_ms_for_n_examples': med,
                'p10_ms': float(np.quantile(values, 0.1)), 'p90_ms': float(np.quantile(values, 0.9)),
                'microseconds_per_example': med * 1000 / len(dataset), 'samples_ms': values})
    identity = {'seed': dataset.seed, 'n': len(dataset), 'sha256': dataset_fingerprint(dataset),
                'batch_size': batch_size, **asdict(dataset.spec)}
    env = {**source_identity(), 'device': str(device),
           'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
           'cuda_runtime': torch.version.cuda if device.type == 'cuda' else None}
    return ({'format': FORMAT + '.evaluation', 'dataset': identity, 'environment': env,
             'rows': rows, 'paired_test_example_differences': paired},
            {'format': FORMAT + '.benchmark', 'dataset': identity, 'environment': env,
             'warmup_per_case': warmup if repeats else 0, 'repeats': repeats,
             'order_seed': order_seed, 'case_order_each_repeat': repeat_orders,
             'timed_scope': 'Same prediction callable/cohort/batch shape as evaluation; includes fixed-depth model, retained rollout construction, output policy and concatenation; excludes transfers, loading, calibration, metrics, Lingua and checking.',
             'timed_output_verified_against_evaluation': bool(repeats), 'rows': timed_rows,
             'warning': 'Inference timing, not end-to-end service latency. No adaptive skipping; candidate evaluations are not FLOPs.'})


def run_study(checkpoint: Path, output_dir: Path, *, device_name: str = 'auto',
              calibration_seed: int = 21000, test_seed: int = 22000,
              calibration_size: int = 4000, n: int = 4000, batch_size: int = 1024,
              target_coverage: float = 0.75, warmup: int = 5, repeats: int = 20,
              order_seed: int = 23000, v1_checkpoint: Path | None = None) -> dict:
    """Run a new study without touching the trained checkpoint or existing runs."""
    for value, name in ((calibration_size, 'calibration_size'), (n, 'n'), (batch_size, 'batch_size')):
        positive_int(value, name)
    ClaimPolicy(None, target_coverage)
    _seed(calibration_seed); _seed(test_seed); _seed(order_seed)
    if calibration_seed == test_seed:
        raise ValueError('calibration and test seeds must differ')
    for value in (warmup, repeats):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError('warmup/repeats must be nonnegative integers')
    if output_dir.exists():
        raise ValueError(f'Refusing to overwrite study folder: {output_dir}')
    engines, metadata, device, reserved = load_estimators(checkpoint, device_name, v1_checkpoint)
    if {calibration_seed, test_seed} & reserved:
        raise ValueError('new calibration/test seed overlaps a training or previously calibrated data seed')
    settings = metadata['training']
    kwargs = {name: settings[name] for name in ('noise_std', 'observe_probability')}
    calibration = StructuredFlowDataset(calibration_size, seed=calibration_seed, **kwargs)
    test = StructuredFlowDataset(n, seed=test_seed, **kwargs)
    output_dir.mkdir(parents=True, exist_ok=False)
    plan = {'format': FORMAT + '.plan', 'checkpoint': str(checkpoint.resolve()),
            'checkpoint_sha256': metadata['checkpoint_sha256'],
            'source': source_identity(), 'training_seed': settings.get('seed'), 'calibration_seed': calibration_seed,
            'test_seed': test_seed, 'reserved_data_seeds': sorted(reserved),
            'target_coverage': target_coverage, 'models': [e.description() for e in engines],
            'status': 'started; COMPLETED.json will only exist after every check passes',
            'scope': 'Same checkpoint weights, fixed trained depths and matched output policies. Existing v2 losses differ between architectures; this does not isolate recurrence alone.'}
    write_json(output_dir / 'PLAN.json', plan)
    print('Calibrating the same coverage rule separately for every structured model...')
    policies, artifact = calibrate(engines, calibration, device, target_coverage, batch_size)
    artifact['checkpoint_sha256'] = metadata['checkpoint_sha256']
    artifact['model_sources'] = {e.name: e.checkpoint_sha256 for e in engines if e.structured}
    artifact['test_seed_reserved'] = test_seed
    write_json(output_dir / 'calibration.json', artifact)
    print('Evaluating and timing identical output modes on one held-out cohort...')
    evaluation, benchmark = evaluate_and_time(engines, policies, test, device,
        batch_size=batch_size, warmup=warmup, repeats=repeats, order_seed=order_seed)
    evaluation['calibration_sha256'] = sha256_file(output_dir / 'calibration.json')
    write_json(output_dir / 'eval.json', evaluation)
    write_json(output_dir / 'benchmark.json', benchmark)
    # Illustrate exactly one existing test example without inventing a new test
    # distribution or hunting for an attractive result.
    from .dual_lingua import make_record, verify_record
    engine = next(e for e in engines if e.name == 'shared')
    observed, mask = test.observed[:1].to(device), test.mask[:1].to(device)
    prediction = engine.predict(observed, mask, policies['shared'], 'mixture_with_claim')
    record = make_record(prediction, policies['shared'], observed, mask,
        metadata['names'], metadata['boundaries'], model_name='shared',
        checkpoint_sha256=str(metadata['checkpoint_sha256']),
        calibration_sha256=sha256_file(output_dir / 'calibration.json'),
        trained_depth=engine.depth)
    checked = verify_record(record, checkpoint=checkpoint, calibration=output_dir / 'calibration.json')
    write_json(output_dir / 'lingua.json', record)
    write_json(output_dir / 'verification.json', checked)
    if not checked['accepted']:
        raise RuntimeError('Dual-output Lingua record failed: ' + checked['reason'])
    if sha256_file(checkpoint) != metadata['checkpoint_sha256']:
        raise RuntimeError('checkpoint changed during study')
    if v1_checkpoint is not None:
        old = next(e for e in engines if e.family == 'v1')
        if sha256_file(v1_checkpoint) != old.checkpoint_sha256:
            raise RuntimeError('v1 checkpoint changed during study')
    result = {'format': FORMAT + '.complete', 'output_dir': str(output_dir.resolve()),
              'checkpoint_sha256': metadata['checkpoint_sha256'], 'verified': True,
              'rows': len(evaluation['rows']), 'timed_rows': len(benchmark['rows'])}
    write_json(output_dir / 'COMPLETED.json', result)
    print('\nmodel / mode                          estimate MSE    coverage')
    for row in evaluation['rows']:
        coverage = row.get('coverage')
        print(f"{row['model']+'/'+row['output_mode']:<42} {row['estimate_mse']:.6f}      "
              + ('not requested' if coverage is None else f'{coverage:.3f}'))
    print('Lingua PASS: estimate arithmetic and separately scoped claim checked; no neural replay.')
    print(f'Results: {output_dir}')
    return result
