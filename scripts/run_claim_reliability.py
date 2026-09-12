"""Freeze every coverage policy, then measure claims on one untouched cohort.

This compares frozen estimators at separately calibrated target coverage. It
records the arrays needed to recompute quality; it does not produce or verify
property receipts, authenticate execution, train models, or measure latency.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from benchmark_verified_pipeline import source_identity
from claim_reliability_metrics import analyze_model, aggregate_models, fit_policies
from universa_recurrent.neural.data import StructuredFlowDataset
from universa_recurrent.neural.dual_output import ClaimPolicy, FixedEstimator
from universa_recurrent.neural.dual_study import collect, dataset_fingerprint, load_estimators, read_json, write_json
from universa_recurrent.neural.retention_study import configure
from universa_recurrent.neural.train import sha256_file

FORMAT = 'universa-recurrent.claim-reliability.v1'
ROOT = Path(__file__).resolve().parents[1]
MODELS = ('shared', 'fixed_depth_4', 'direct', 'gaussian_generator_reference')
NEURAL_DEPTHS = {'shared': 8, 'fixed_depth_4': 4, 'direct': 1}
FIXED = {
    'format': 'universa-recurrent.claim-reliability.protocol.v1',
    'study_id': 'claim-reliability-v1', 'calibration_size': 5000, 'test_size': 20000,
    'calibration_seed': 41000, 'test_seed': 42000,
    'coverages': [.25, .5, .75, .9], 'primary_coverage': .75,
    'primary_comparison': ['shared', 'fixed_depth_4'], 'models': list(MODELS),
    'training_seeds': [6100, 7100, 8100, 9100, 10100], 'batch_size': 1024,
    'cpu_threads': 1, 'bootstrap_repeats': 2000, 'bootstrap_seed': 43000,
    'prior_data_seeds': [21000, 22000, 31000, 32000, 33000, 36000],
    'smoke_calibration_seed': 51000, 'smoke_test_seed': 52000,
}
SCOPE = {
    'estimator': 'Frozen weights, full trained depths, mixture estimate independent of claim threshold.',
    'calibration': 'New score-only inclusive coverage cutoffs; all fits frozen before any final test generation.',
    'reference': 'One privileged Gaussian generator reference, CPU float64; reused across matching fits, not independent training replications.',
    'evidence': 'Saved inputs, truth, labels, route probabilities and mixture estimates support independent quality recomputation. No receipt/property validity or execution authentication is claimed.',
    'uncertainty': 'Common input resampling conditional on these frozen fits and calibration policies; not uncertainty over new training runs.',
    'timing': 'No benchmark latency claim; model loading, source checking, data generation, and reporting are not timed.',
}


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return result


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def script_identity():
    return {name: sha256_file(ROOT / 'scripts' / name) for name in
            ('run_claim_reliability.py', 'claim_reliability_metrics.py',
             'benchmark_verified_pipeline.py', 'benchmark_verifier_setup.py')}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def array_identity(value):
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject or not np.isfinite(array).all():
        raise ValueError('saved arrays must be finite numeric arrays without objects')
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'sha256': hashlib.sha256(array.tobytes()).hexdigest()}


def input_rows(observed, mask):
    observed, mask = np.asarray(observed), np.asarray(mask)
    if observed.ndim != 2 or mask.shape != observed.shape or len(observed) < 1:
        raise ValueError('observed/mask must match nonempty [N,D]')
    if not np.isfinite(observed).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError('nonfinite observations or invalid masks')
    joined = np.ascontiguousarray(np.concatenate([observed, mask], axis=1))
    joined[joined == 0] = 0  # Treat signed zero as the same observed value.
    return joined.view(np.dtype((np.void, joined.dtype.itemsize * joined.shape[1]))).ravel()


def input_identity(observed, mask):
    rows = input_rows(observed, mask)
    unique = np.unique(rows)
    if len(unique) != len(rows):
        raise ValueError('duplicate observed/mask inputs in cohort')
    return {'count': len(rows), 'distinct_count': len(unique),
            'ordered_sha256': hashlib.sha256(rows.tobytes()).hexdigest(),
            'unique_sha256': hashlib.sha256(unique.tobytes()).hexdigest()}


def assert_disjoint(calibration, test):
    left = input_rows(calibration['observed'], calibration['mask'])
    right = input_rows(test['observed'], test['mask'])
    if np.intersect1d(left, right).size:
        raise ValueError('calibration and test observed/mask inputs overlap')


def save_arrays(path, arrays):
    identities = {name: array_identity(array) for name, array in arrays.items()}
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return {'file': Path(path).name, 'sha256': sha256_file(path), 'arrays': identities}


def load_arrays(directory, artifact):
    filename = artifact['file']
    if Path(filename).name != filename or not filename.endswith('.npz'):
        raise ValueError('invalid array artifact filename')
    path = directory / filename
    if sha256_file(path) != artifact['sha256']:
        raise ValueError('array artifact changed')
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(artifact['arrays']):
            raise ValueError('array fields differ from report')
        arrays = {name: archive[name] for name in archive.files}
    if {name: array_identity(value) for name, value in arrays.items()} != artifact['arrays']:
        raise ValueError('array content differs from reported identities')
    return arrays


def seeds_in_provenance(value):
    """Conservatively reserve every explicit seed in checkpoint/calibration data."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == 'seed' or key.endswith('_seed') or key == 'test_seed_reserved':
                found.add(_integer(item, key))
            elif isinstance(item, (dict, list)):
                found |= seeds_in_provenance(item)
    elif isinstance(value, list):
        for item in value:
            found |= seeds_in_provenance(item)
    return found


def validate_protocol(path, smoke=False, calibration_size=32, test_size=64):
    protocol = read_json(path)
    for key, expected in FIXED.items():
        if key not in protocol or type(protocol[key]) is not type(expected) or protocol[key] != expected:
            raise ValueError(f'protocol differs from frozen v1 field {key}; create a new protocol/study version')
    effective = dict(protocol)
    if smoke:
        if not 1 <= calibration_size <= 1000 or not 1 <= test_size <= 1000:
            raise ValueError('smoke cohort sizes must be between 1 and 1000')
        effective.update(calibration_seed=protocol['smoke_calibration_seed'],
                         test_seed=protocol['smoke_test_seed'],
                         calibration_size=calibration_size, test_size=test_size,
                         bootstrap_repeats=50)
    return protocol, effective


def validate_loaded(engines, metadata, expected_seed):
    by_name = {engine.name: engine for engine in engines}
    if not set(MODELS) <= set(by_name):
        raise ValueError('checkpoint lacks required model')
    for name, depth in NEURAL_DEPTHS.items():
        if by_name[name].depth != depth:
            raise ValueError(f'{name} must use trained depth {depth}')
    settings = metadata['training']
    if _integer(settings['seed'], 'training seed') != expected_seed:
        raise ValueError('checkpoint training seed differs from filename')
    bases = by_name['shared'].bases.detach().cpu().numpy()
    for name in MODELS:
        if not np.array_equal(by_name[name].bases.detach().cpu().numpy(), bases):
            raise ValueError('model bases differ within fit')
    return {'noise_std': settings['noise_std'], 'observe_probability': settings['observe_probability'],
            'names': metadata['names'], 'bases': array_identity(bases),
            'boundaries': array_identity(metadata['boundaries'].detach().cpu().numpy()),
            'models': {name: {'family': by_name[name].family,
                             'trained_depth': by_name[name].depth} for name in MODELS}}


def inspect_references(replication_dir, protocol, smoke):
    checkpoints = sorted(replication_dir.glob('weights-*.pt'))
    if not checkpoints:
        raise ValueError('no weights-*.pt in replication directory')
    records, seen, library, reserved = [], set(), None, set(protocol['prior_data_seeds'])
    for checkpoint in checkpoints:
        try:
            seed = int(checkpoint.stem.removeprefix('weights-'))
        except ValueError as error:
            raise ValueError('invalid checkpoint filename') from error
        if seed in seen:
            raise ValueError('duplicate training seed')
        seen.add(seed)
        calibration = replication_dir / f'study-{seed}' / 'calibration.json'
        cp_sha, cal_sha = sha256_file(checkpoint), sha256_file(calibration)
        engines, metadata, _, used = load_estimators(checkpoint, 'cpu')
        description = validate_loaded(engines, metadata, seed)
        if metadata['checkpoint_sha256'] != cp_sha:
            raise ValueError('checkpoint changed during inspection')
        previous = read_json(calibration)
        if previous.get('checkpoint_sha256') != cp_sha:
            raise ValueError('previous calibration belongs to another checkpoint')
        if any(previous.get('model_sources', {}).get(name) != cp_sha for name in NEURAL_DEPTHS):
            raise ValueError('previous model calibration sources differ from checkpoint')
        library = description if library is None else library
        if description != library:
            raise ValueError('data settings, library, families or trained depths differ across fits')
        reserved |= used | seeds_in_provenance(metadata) | seeds_in_provenance(previous)
        records.append({'training_seed': seed, 'checkpoint': checkpoint.name,
                        'checkpoint_sha256': cp_sha,
                        'previous_calibration': str(calibration.relative_to(replication_dir)),
                        'previous_calibration_sha256': cal_sha})
        if sha256_file(checkpoint) != cp_sha or sha256_file(calibration) != cal_sha:
            raise ValueError('reference changed during inspection')
        del engines
    if not smoke and seen != set(protocol['training_seeds']):
        raise ValueError('confirmatory run requires exactly the five frozen training seeds')
    if not seen <= set(protocol['training_seeds']):
        raise ValueError('unexpected training seed outside frozen fit set')
    if len({record['checkpoint_sha256'] for record in records}) != len(records):
        raise ValueError('duplicate checkpoint bytes across fits')
    requested = {protocol['calibration_seed'], protocol['test_seed']}
    if len(requested) != 2 or requested & reserved:
        raise ValueError('calibration/test seed overlaps reserved training or prior data')
    return sorted(records, key=lambda value: value['training_seed']), library, sorted(reserved)


def check_immutable(args, manifest, manifest_sha):
    if sha256_file(args.output_dir / 'manifest.json') != manifest_sha:
        raise ValueError('manifest changed')
    if source_identity() != manifest['source'] or script_identity() != manifest['scripts_sha256']:
        raise ValueError('source changed during study')
    for ref in manifest['references']:
        if (sha256_file(args.replication_dir / ref['checkpoint']) != ref['checkpoint_sha256']
                or sha256_file(args.replication_dir / ref['previous_calibration']) != ref['previous_calibration_sha256']):
            raise ValueError('reference changed during study')


def check_lock(directory, manifest, manifest_sha, expected_hash):
    path = directory / 'calibration-lock.json'
    if not path.is_file() or sha256_file(path) != expected_hash:
        raise ValueError('missing or changed global calibration lock')
    lock = read_json(path)
    if (lock['format'] != FORMAT + '.calibration-lock' or lock['manifest_sha256'] != manifest_sha
            or lock['training_seeds'] != [ref['training_seed'] for ref in manifest['references']]
            or lock['test_generation_permitted'] is not True):
        raise ValueError('global calibration lock is incomplete or inconsistent')
    for item in lock['calibrations']:
        if sha256_file(directory / item['report_file']) != item['report_sha256']:
            raise ValueError('calibration report changed after policy freeze')
        report = read_json(directory / item['report_file'])
        if report['policies'] != item['policies']:
            raise ValueError('locked policies differ from calibration report')
        load_arrays(directory, report['artifact'])
    if [item['training_seed'] for item in lock['calibrations']] != lock['training_seeds']:
        raise ValueError('global calibration lock omits a fit')
    return lock


def _dataset_arrays(dataset):
    return {name: getattr(dataset, 'label' if name == 'labels' else name).numpy()
            for name in ('observed', 'mask', 'truth', 'labels')}


@torch.inference_mode()
def run_phase(args):
    manifest = read_json(args.output_dir / 'manifest.json')
    check_immutable(args, manifest, args.manifest_sha256)
    protocol = manifest['effective_protocol']
    torch.set_num_threads(protocol['cpu_threads'])
    execution = configure(True)
    reference = next((item for item in manifest['references']
                      if item['training_seed'] == args.training_seed), None)
    if reference is None:
        raise ValueError('worker seed absent from frozen manifest')
    report_path = args.output_dir / f'{args.phase}-{args.training_seed}.json'
    array_path = report_path.with_suffix('.npz')
    if report_path.exists() or array_path.exists():
        raise ValueError('refusing to overwrite phase artifacts')
    lock = None
    if args.phase == 'test':
        lock = check_lock(args.output_dir, manifest, args.manifest_sha256, args.lock_sha256)
    engines, metadata, device, _ = load_estimators(args.replication_dir / reference['checkpoint'], args.device)
    if validate_loaded(engines, metadata, args.training_seed) != manifest['library']:
        raise ValueError('loaded library differs from frozen manifest')
    if metadata['checkpoint_sha256'] != reference['checkpoint_sha256']:
        raise ValueError('loaded checkpoint differs from frozen reference')
    # This is the only dataset construction site. No test data is constructed
    # until every calibration report is sealed by the global lock above.
    dataset = StructuredFlowDataset(protocol[args.phase + '_size'], seed=protocol[args.phase + '_seed'],
        noise_std=manifest['library']['noise_std'],
        observe_probability=manifest['library']['observe_probability'])
    arrays = _dataset_arrays(dataset)
    identity = input_identity(arrays['observed'], arrays['mask'])
    fingerprint = dataset_fingerprint(dataset)
    calibration_report, calibration_arrays, policies = None, None, {}
    if args.phase == 'test':
        item = next(item for item in lock['calibrations'] if item['training_seed'] == args.training_seed)
        calibration_report = read_json(args.output_dir / item['report_file'])
        calibration_arrays = load_arrays(args.output_dir, calibration_report['artifact'])
        assert_disjoint(calibration_arrays, arrays)
        policies = item['policies']
    by_name = {engine.name: engine for engine in engines}
    descriptions, analysis = {}, {}
    first_seed = manifest['references'][0]['training_seed']
    for name in MODELS:
        engine = by_name[name]
        if name == 'gaussian_generator_reference' and args.training_seed != first_seed:
            prior_report = read_json(args.output_dir / f'{args.phase}-{first_seed}.json')
            if prior_report['dataset_sha256'] != fingerprint:
                raise ValueError('reference cohort differs from current fit')
            prior = load_arrays(args.output_dir, prior_report['artifact'])
            probability = prior[name + '__probabilities']
            estimate = prior.get(name + '__estimates')
            computed = False
        else:
            if name == 'gaussian_generator_reference':
                engine = FixedEstimator(name, None, engine.bases.detach().cpu().double(), 'gaussian',
                                        engine.noise_std, engine.checkpoint_sha256)
                observed, mask = dataset.observed.double(), dataset.mask.double()
            else:
                observed, mask = dataset.observed.to(device), dataset.mask.to(device)
            output = collect(engine, observed, mask, ClaimPolicy(None), 'mixture', protocol['batch_size'])
            probability = output['probabilities'].detach().cpu().numpy()
            estimate = output['estimate'].detach().cpu().numpy()
            del output, observed, mask
            computed = True
        arrays[name + '__probabilities'] = probability
        if args.phase == 'calibration':
            policies[name] = fit_policies(probability, protocol['coverages'])
        else:
            arrays[name + '__estimates'] = estimate
            analysis[name] = analyze_model(calibration_arrays[name + '__probabilities'], probability,
                estimate, arrays['truth'], arrays['labels'], protocol['coverages'], policies=policies[name])
        descriptions[name] = {**engine.description(), 'computed_in_this_worker': computed,
            'reference_first_training_seed': first_seed if engine.family == 'gaussian' else None,
            'prediction_dtype': str(probability.dtype),
            'prediction_device': 'cpu' if engine.family == 'gaussian' else str(device)}
        print(f'{args.phase}: fit {args.training_seed}, {name}, {len(dataset)} inputs', flush=True)
    artifact = save_arrays(array_path, arrays)
    check_immutable(args, manifest, args.manifest_sha256)
    if args.phase == 'test':
        check_lock(args.output_dir, manifest, args.manifest_sha256, args.lock_sha256)
    report = {'format': FORMAT + '.' + args.phase, 'training_seed': args.training_seed,
        'manifest_sha256': args.manifest_sha256, 'calibration_lock_sha256': args.lock_sha256,
        'checkpoint_sha256': reference['checkpoint_sha256'], 'dataset_sha256': fingerprint,
        'dataset': {'seed': dataset.seed, 'n': len(dataset), **asdict(dataset.spec)},
        'inputs': identity, 'artifact': artifact, 'policies': policies, 'analysis': analysis,
        'models': descriptions, 'source': manifest['source'], 'execution': execution,
        'environment': {'neural_device': str(device), 'torch': str(torch.__version__),
            'cuda_runtime': torch.version.cuda,
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
            'numpy': np.__version__}, 'smoke': manifest['smoke'], 'scope': SCOPE}
    write_json(report_path, report)
    return report


def validate_phase_reports(directory, manifest, phase, manifest_sha):
    reports, common, reference_arrays = [], None, None
    for ref in manifest['references']:
        report = read_json(directory / f'{phase}-{ref["training_seed"]}.json')
        if (report['format'] != FORMAT + '.' + phase or report['training_seed'] != ref['training_seed']
                or report['checkpoint_sha256'] != ref['checkpoint_sha256']
                or report['manifest_sha256'] != manifest_sha or report['source'] != manifest['source']
                or set(report['policies']) != set(MODELS)):
            raise ValueError('phase report identity mismatch')
        arrays = load_arrays(directory, report['artifact'])
        if input_identity(arrays['observed'], arrays['mask']) != report['inputs']:
            raise ValueError('phase report input identity mismatch')
        raw = {key: arrays[key] for key in ('observed', 'mask', 'truth', 'labels')}
        if common is None:
            common = raw
            reference_arrays = {key: value for key, value in arrays.items()
                                if key.startswith('gaussian_generator_reference__')}
        elif (any(not np.array_equal(raw[key], common[key]) for key in raw)
                or any(not np.array_equal(arrays[key], value) for key, value in reference_arrays.items())):
            raise ValueError('cohort or Gaussian reference differs across fits')
        for name in MODELS:
            if phase == 'calibration' and report['policies'][name] != fit_policies(
                    arrays[name + '__probabilities'], manifest['effective_protocol']['coverages']):
                raise ValueError('saved calibration policy differs from score-only cutoff')
        reports.append(report)
    return reports, common


def _worker(args, phase, seed, manifest_sha, lock_sha=None):
    command = [sys.executable, str(Path(__file__).resolve()), '--replication-dir', str(args.replication_dir.resolve()),
        '--output-dir', str(args.output_dir.resolve()), '--device', args.device,
        '--phase', phase, '--training-seed', str(seed), '--manifest-sha256', manifest_sha]
    if lock_sha:
        command.extend(['--lock-sha256', lock_sha])
    subprocess.run(command, check=True, env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'))


def run_parent(args):
    if args.output_dir.exists():
        raise ValueError('refusing to overwrite output directory')
    protocol_sha = sha256_file(args.protocol)
    original, protocol = validate_protocol(args.protocol, args.smoke,
        args.smoke_calibration_size, args.smoke_test_size)
    if not args.smoke and (args.smoke_calibration_size != 32 or args.smoke_test_size != 64):
        raise ValueError('smoke size overrides require --smoke')
    torch.set_num_threads(protocol['cpu_threads'])
    source, scripts = source_identity(), script_identity()
    references, library, reserved = inspect_references(args.replication_dir, protocol, args.smoke)
    if sha256_file(args.protocol) != protocol_sha or read_json(args.protocol) != original:
        raise ValueError('protocol changed during study setup')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {'format': FORMAT + '.manifest', 'protocol': original,
        'protocol_sha256': protocol_sha, 'effective_protocol': protocol,
        'smoke': args.smoke, 'prespecified_protocol': not args.smoke,
        'source': source, 'scripts_sha256': scripts, 'references': references,
        'library': library, 'reserved_data_seeds': reserved,
        'phase_order': ['all_calibration_workers', 'exclusive_global_lock', 'all_test_workers', 'aggregate'],
        'scope': SCOPE, 'status': 'started; summary.json exists only after all checks pass'}
    write_json(args.output_dir / 'manifest.json', manifest)
    manifest_sha = sha256_file(args.output_dir / 'manifest.json')
    for ref in references:
        _worker(args, 'calibration', ref['training_seed'], manifest_sha)
    check_immutable(args, manifest, manifest_sha)
    calibration_reports, calibration_inputs = validate_phase_reports(args.output_dir, manifest, 'calibration', manifest_sha)
    locked = []
    for report in calibration_reports:
        filename = f'calibration-{report["training_seed"]}.json'
        locked.append({'training_seed': report['training_seed'], 'report_file': filename,
                       'report_sha256': sha256_file(args.output_dir / filename),
                       'artifact_sha256': report['artifact']['sha256'], 'policies': report['policies']})
    write_json(args.output_dir / 'calibration-lock.json', {
        'format': FORMAT + '.calibration-lock', 'manifest_sha256': manifest_sha,
        'training_seeds': [ref['training_seed'] for ref in references],
        'calibrations': locked, 'test_generation_permitted': True,
        'note': 'All model/fit/target cutoffs frozen before any test dataset is generated.'})
    lock_sha = sha256_file(args.output_dir / 'calibration-lock.json')
    print('All calibration policies locked. Starting the untouched test cohort.', flush=True)
    for ref in references:
        _worker(args, 'test', ref['training_seed'], manifest_sha, lock_sha)
    test_reports, test_inputs = validate_phase_reports(args.output_dir, manifest, 'test', manifest_sha)
    assert_disjoint(calibration_inputs, test_inputs)
    models = {name: [] for name in MODELS}
    for cal, test in zip(calibration_reports, test_reports):
        cal_arrays, test_arrays = load_arrays(args.output_dir, cal['artifact']), load_arrays(args.output_dir, test['artifact'])
        if test['calibration_lock_sha256'] != lock_sha or test['policies'] != cal['policies']:
            raise ValueError('test policy differs from global calibration lock')
        for name in MODELS:
            expected = analyze_model(cal_arrays[name + '__probabilities'], test_arrays[name + '__probabilities'],
                test_arrays[name + '__estimates'], test_arrays['truth'], test_arrays['labels'],
                protocol['coverages'], policies=cal['policies'][name])
            if test['analysis'][name] != expected:
                raise ValueError('saved quality report differs from independently reread arrays')
            if name != 'gaussian_generator_reference' or not models[name]:
                models[name].append({'calibration_probabilities': cal_arrays[name + '__probabilities'],
                    'test_probabilities': test_arrays[name + '__probabilities'],
                    'test_estimates': test_arrays[name + '__estimates'], 'policies': cal['policies'][name]})
    aggregate = aggregate_models(models, test_inputs['truth'], test_inputs['labels'],
        coverages=protocol['coverages'], primary_coverage=protocol['primary_coverage'], n_resamples=protocol['bootstrap_repeats'],
        seed=protocol['bootstrap_seed'])
    check_immutable(args, manifest, manifest_sha)
    check_lock(args.output_dir, manifest, manifest_sha, lock_sha)
    summary = {'format': FORMAT + '.complete', 'smoke': args.smoke, 'prespecified_protocol': not args.smoke,
        'manifest_sha256': manifest_sha, 'calibration_lock_sha256': lock_sha,
        'source': source, 'scripts_sha256': scripts, 'training_seeds': [ref['training_seed'] for ref in references],
        'calibration_dataset_sha256': calibration_reports[0]['dataset_sha256'],
        'test_dataset_sha256': test_reports[0]['dataset_sha256'],
        'calibration_inputs': calibration_reports[0]['inputs'], 'test_inputs': test_reports[0]['inputs'],
        'all_calibration_policies_locked_before_test': True,
        'calibration_test_inputs_disjoint': True, 'all_common_cohorts_equal': True,
        'all_saved_arrays_reloaded_and_metrics_recomputed': True,
        'gaussian_reference_independent_fits': 1, 'aggregate': aggregate,
        'per_fit': [{'training_seed': report['training_seed'], 'analysis': report['analysis']} for report in test_reports],
        'artifacts': {path.name: sha256_file(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()},
        'scope': SCOPE, 'warning': 'SMOKE ONLY: not confirmatory research evidence.' if args.smoke else
            'One fixed common test cohort; fit copies do not create new independent inputs. Conditional bootstrap uncertainty does not cover new training runs.'}
    write_json(args.output_dir / 'summary.json', summary)
    print('COMPLETE:', args.output_dir / 'summary.json', flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replication-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--protocol', type=Path, default=ROOT / 'experiments/claim_reliability_v1.json')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-calibration-size', type=positive, default=32)
    parser.add_argument('--smoke-test-size', type=positive, default=64)
    parser.add_argument('--phase', choices=('calibration', 'test'), help=argparse.SUPPRESS)
    parser.add_argument('--training-seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--manifest-sha256', help=argparse.SUPPRESS)
    parser.add_argument('--lock-sha256', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if args.phase:
            if args.training_seed is None or not args.manifest_sha256 or (args.phase == 'test' and not args.lock_sha256):
                raise ValueError('worker phase requires frozen manifest, seed and test calibration lock')
            run_phase(args)
        else:
            run_parent(args)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print('Error:', error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
