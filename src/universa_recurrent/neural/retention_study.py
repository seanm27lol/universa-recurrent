"""Measure keeping only the answer versus retaining a numerical lab notebook.

Frozen weights, fixed depths, paired implementations. GPU inference timings and
sampled Lingua costs have DIFFERENT denominators; never subtract or combine them
as if both covered the full test cohort. Run with --help for the parent driver.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from .data import StructuredFlowDataset
from .dual_output import ClaimPolicy, construct_output, metrics
from .dual_study import collect, load_estimators, read_json, write_json, dataset_fingerprint, source_identity
from .dual_lingua import make_record as endpoint_record
from .final_state import FinalStateEstimator, final_candidates
from .retention_lingua import make_record, verify_record, dumps, loads
from .train import sha256_file

MODELS = ('shared', 'fixed_depth_4', 'untied', 'ambient', 'direct')
FORMAT = 'universa-recurrent.retention-study.v1'
ATOL, RTOL = 1e-7, 1e-6


def compare_outputs(actual: dict, expected: dict) -> dict:
    """Check ALL returned tensors. Numerical errors are not hidden behind mean MSE."""
    if set(actual) != set(expected):
        raise ValueError('output fields differ')
    report = {}
    for name in expected:
        a, e = actual[name].detach().cpu(), expected[name].detach().cpu()
        if a.shape != e.shape or a.dtype != e.dtype:
            raise ValueError(f'{name}: shape/dtype differs')
        if a.is_floating_point():
            if not torch.isfinite(a).all() or not torch.isfinite(e).all():
                raise ValueError(f'{name}: nonfinite output')
            if not torch.allclose(a, e, atol=ATOL, rtol=RTOL):
                raise ValueError(f'{name}: final-only and reference outputs differ')
            report[name] = {'max_absolute_difference': float((a.double()-e.double()).abs().max()),
                            'bitwise_equal': torch.equal(a, e)}
        else:
            if not torch.equal(a, e):
                raise ValueError(f'{name}: discrete claim/route changed')
            report[name] = {'exact_match': True}
    return report


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def stats(values):
    return {'median_ms': float(np.median(values)), 'p10_ms': float(np.quantile(values, .1)),
            'p90_ms': float(np.quantile(values, .9)), 'samples_ms': values}


def measure(tasks, device, warmup, repeats, order_seed, check=None):
    """Interleave paired cases. Synchronization is included; checks follow timing."""
    for call in tasks.values():
        for _ in range(warmup):
            value = call()
            del value
    _sync(device)
    rng, times, orders = random.Random(order_seed), {k: [] for k in tasks}, []
    keys = list(tasks)
    for _ in range(repeats):
        rng.shuffle(keys)
        orders.append(list(keys))
        for key in keys:
            _sync(device)
            started = time.perf_counter_ns()
            value = tasks[key]()
            _sync(device)
            times[key].append((time.perf_counter_ns()-started)/1e6)
            if check is not None:
                check(key, value)
            del value
    return {k: stats(v) for k, v in times.items()}, orders


def _peak_extra(call, device):
    """One separate, warmed allocation measurement; not total device/host memory."""
    if device.type != 'cuda':
        return None
    gc.collect()
    _sync(device)
    start = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    value = call()
    _sync(device)
    extra = torch.cuda.max_memory_allocated(device)-start
    del value
    return int(extra)


def configure(deterministic):
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    return {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'num_threads': torch.get_num_threads(), 'num_interop_threads': torch.get_num_interop_threads()}


@torch.inference_mode()
def audit_costs(engine, metadata, policy, observed, mask, calibration_sha, checkpoint,
                calibration_path, device, samples, repeats, order_seed):
    """First samples in order; serial one-example records, NOT batch-n inference."""
    count = min(samples, len(observed))
    def produce(kind):
        output = []
        for index in range(count):
            obs, m = observed[index:index+1], mask[index:index+1]
            if kind == 'trajectory':
                history = engine.model.rollout(obs, m)
                candidate = {key: val[-1] for key, val in history.items() if key != 'prior_logits'}
            else:
                history = None
                candidate = final_candidates(engine.model, obs, m, validate_values=False)
            prediction = construct_output(candidate['states'], candidate['route_probabilities'], policy,
                                          'mixture_with_claim', validate=False)
            output.append((prediction, history, obs, m))
        return output
    def generate(kind, prepared):
        blobs = []
        for prediction, history, obs, m in prepared:
            endpoint = endpoint_record(prediction, policy, obs, m, metadata['names'], metadata['boundaries'],
                model_name=engine.name, checkpoint_sha256=metadata['checkpoint_sha256'],
                calibration_sha256=calibration_sha, trained_depth=engine.depth)
            record = make_record(endpoint, history=history if kind == 'trajectory' else None, bases=engine.bases)
            blobs.append(dumps(record))
        return blobs
    def check_blobs(blobs):
        checks = [verify_record(loads(blob)) for blob in blobs]
        if not all(check['accepted'] for check in checks):
            raise ValueError('numerical Lingua check failed: ' + str(checks))
        return checks
    prepared = {kind: produce(kind) for kind in ('endpoint', 'trajectory')}
    for index in range(count):
        compare_outputs(prepared['endpoint'][index][0], prepared['trajectory'][index][0])
    retained = {kind: generate(kind, prepared[kind]) for kind in prepared}
    for blobs in retained.values():
        check_blobs(blobs)
    def pipeline(kind):
        return check_blobs(generate(kind, produce(kind)))
    tasks = {}
    for kind in prepared:
        tasks[kind+'/generation_json'] = lambda kind=kind: generate(kind, prepared[kind])
        tasks[kind+'/parse_property_check'] = lambda kind=kind: check_blobs(retained[kind])
        tasks[kind+'/inference_through_check'] = lambda kind=kind: pipeline(kind)
    timing, order = measure(tasks, device, 1, repeats, order_seed)
    bound = {}
    for kind in prepared:
        # A separate cold-ish operation: file hash/deserialization and calibration
        # binding are not disguised as the cheap arithmetic-only checker.
        _sync(device)
        started = time.perf_counter_ns()
        result = verify_record(loads(retained[kind][0]), checkpoint=checkpoint, calibration=calibration_path)
        _sync(device)
        bound[kind] = {'milliseconds_single_call': (time.perf_counter_ns()-started)/1e6, 'result': result}
        if not result['accepted']:
            raise ValueError('checkpoint/calibration binding failed')
    return {'model': engine.name, 'sample_count': count, 'sample_indices': list(range(count)),
            'serial_one_example_inference_then_generation_and_check': True, 'timing_unit': 'milliseconds for sample_count examples',
            'timings': timing, 'order_each_repeat': order,
            'serialized_bytes': {kind: [len(blob) for blob in blobs] for kind, blobs in retained.items()},
            'checkpoint_binding_single_calls': bound,
            'sample_records': {kind: [loads(blob) for blob in blobs] for kind, blobs in retained.items()},
            'scope': {'generation_json': 'Precomputed on-device outputs -> host copies, record construction, UTF-8 JSON bytes.',
                      'parse_property_check': 'JSON parse + NumPy property checks, no checkpoint loading or neural replay.',
                      'inference_through_check': 'Measured directly: resident input -> inference + host record + JSON + property check.',
                      'excluded': 'Request transfer, checkpoint loading, calibration, filesystem writing; binding separately sampled.',
                      'warning': 'Do not subtract or add these medians, or extrapolate sample cost to the batch-n inference rows.'}}


@torch.inference_mode()
def run_worker(args):
    execution = configure(args.deterministic)
    engines, metadata, device, reserved = load_estimators(args.checkpoint, args.device)
    calibration_sha = sha256_file(args.calibration)
    calibration = read_json(args.calibration)
    if calibration.get('checkpoint_sha256') != metadata['checkpoint_sha256']:
        raise ValueError('calibration belongs to a different checkpoint')
    if args.test_seed in reserved | {calibration['data']['seed']}:
        raise ValueError('test seed overlaps training or calibration')
    settings = metadata['training']
    dataset = StructuredFlowDataset(args.n, seed=args.test_seed, noise_std=settings['noise_std'],
                                    observe_probability=settings['observe_probability'])
    obs, mask, truth, labels = (getattr(dataset, k).to(device) for k in ('observed', 'mask', 'truth', 'label'))
    by_name = {e.name: e for e in engines}
    if not set(MODELS) <= set(by_name):
        raise ValueError('missing required comparison models')
    tasks, baselines, policies, rows, equivalence = {}, {}, {}, {}, {}
    for name in MODELS:
        engine = by_name[name]
        mode = 'estimate_only' if name == 'ambient' else 'mixture_with_claim'
        if name == 'ambient':
            policy = None
        else:
            if calibration['model_sources'][name] != metadata['checkpoint_sha256']:
                raise ValueError('calibration model source differs')
            policy = ClaimPolicy(**calibration['models'][name]['policy'])
        policies[name] = policy
        for path, target in (('rollout', engine), ('final_only', FinalStateEstimator.from_reference(engine))):
            key = name+'/'+path
            tasks[key] = lambda target=target, policy=policy, mode=mode: collect(target, obs, mask, policy, mode, args.batch_size)
            out = tasks[key]()
            rows[key] = {'model': name, 'path': path, 'output_mode': mode,
                         **metrics(out, truth, labels, mode)}
            baselines[key] = {field: v.detach().cpu() for field, v in out.items()}
            del out
        equivalence[name] = compare_outputs(baselines[name+'/final_only'], baselines[name+'/rollout'])
    def check(key, value):
        compare_outputs(value, baselines[key.split('/')[0]+'/rollout'])
    times, order = measure(tasks, device, args.warmup, args.repeats, args.order_seed, check)
    for key, row in rows.items():
        row['timing'] = times[key]
        row['peak_extra_torch_cuda_bytes'] = _peak_extra(tasks[key], device)
    audit = [audit_costs(by_name[name], metadata, policies[name], obs, mask, calibration_sha,
                        args.checkpoint, args.calibration, device, args.audit_samples, args.audit_repeats,
                        args.order_seed+100+index) for index, name in enumerate(('shared', 'fixed_depth_4'))]
    if sha256_file(args.checkpoint) != metadata['checkpoint_sha256'] or sha256_file(args.calibration) != calibration_sha:
        raise ValueError('input artifact changed during benchmark')
    return {'format': FORMAT+'.worker', 'training_seed': settings['seed'],
            'checkpoint_sha256': metadata['checkpoint_sha256'], 'calibration_sha256': calibration_sha,
            'dataset_sha256': dataset_fingerprint(dataset), 'test_seed': args.test_seed, 'n': args.n,
            'batch_size': args.batch_size, 'execution': execution,
            'environment': {**source_identity(), 'device': str(device),
                            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                            'cuda_runtime': torch.version.cuda},
            'equivalence': equivalence, 'all_timed_outputs_checked': True, 'atol': ATOL, 'rtol': RTOL,
            'inference_rows': list(rows.values()), 'timing_order_each_repeat': order, 'lingua_costs': audit,
            'inference_scope': 'Milliseconds for all n examples; inputs resident; model, output policy and concatenation included. No transfers, metrics, Lingua or checks. Peak extra is PyTorch allocator only, warmed, measured separately.',
            'limits': ['Same depth and weights, not early stopping or compressed hidden-state computation.',
                       'Numerical equivalence is tested within fixed tolerances; discrete claims/routes must match exactly.',
                       'Trajectory checker checks geometry and diagnostics, not learned transitions or execution authenticity.']}


def summarize(workers):
    groups = {}
    for w in workers:
        for row in w['inference_rows']:
            groups.setdefault((row['model'], w['batch_size']), {}).setdefault(w['training_seed'], {})[row['path']] = row
    out = []
    for (model, batch), seeds in sorted(groups.items()):
        pairs = []
        for seed, paths in sorted(seeds.items()):
            r, f = paths['rollout'], paths['final_only']
            pairs.append({'training_seed': seed, 'rollout_ms': r['timing']['median_ms'],
                          'final_only_ms': f['timing']['median_ms'],
                          'rollout_divided_by_final_time': r['timing']['median_ms']/f['timing']['median_ms'],
                          'mse_difference_final_minus_rollout': f['estimate_mse']-r['estimate_mse'],
                          'rollout_peak_extra_cuda_bytes': r['peak_extra_torch_cuda_bytes'],
                          'final_peak_extra_cuda_bytes': f['peak_extra_torch_cuda_bytes']})
        out.append({'model': model, 'batch_size': batch, 'checkpoint_pairs': len(pairs),
                    'median_paired_speed_ratio': statistics.median(p['rollout_divided_by_final_time'] for p in pairs),
                    'median_rollout_ms': statistics.median(p['rollout_ms'] for p in pairs),
                    'median_final_only_ms': statistics.median(p['final_only_ms'] for p in pairs), 'per_seed': pairs})
    return out


def run_parent(args):
    checkpoints = sorted(args.replication_dir.glob('weights-*.pt'))
    if not checkpoints:
        raise ValueError('no weights-*.pt in replication directory')
    jobs = []
    for checkpoint in checkpoints:
        seed = int(checkpoint.stem.removeprefix('weights-'))
        calibration = args.replication_dir/f'study-{seed}'/'calibration.json'
        if not calibration.is_file():
            raise ValueError(f'missing calibration report: {calibration}')
        for batch in args.batch_sizes:
            jobs.append((checkpoint, calibration, seed, batch))
    if args.output_dir.exists():
        raise ValueError('refusing to overwrite output folder')
    args.output_dir.mkdir(parents=True)
    rng = random.Random(args.order_seed)
    rng.shuffle(jobs)
    source = source_identity()
    write_json(args.output_dir/'PLAN.json', {'format': FORMAT+'.plan', 'source': source,
        'training_checkpoint_files': [p.name for p in checkpoints], 'test_seed': args.test_seed,
        'n': args.n, 'batch_sizes': args.batch_sizes, 'deterministic': args.deterministic,
        'warmup': args.warmup, 'repeats': args.repeats, 'audit_samples': args.audit_samples,
        'audit_repeats': args.audit_repeats, 'order_seed': args.order_seed,
        'job_order': [[seed,batch] for _,_,seed,batch in jobs], 'no_training': True})
    workers = []
    hashes, seen, dataset_sha = {}, set(), None
    for checkpoint, calibration, seed, batch in jobs:
        output = args.output_dir/f'seed-{seed}-batch-{batch}.json'
        cmd = [sys.executable, '-m', __name__ if __name__ != '__main__' else 'universa_recurrent.neural.retention_study',
               '--checkpoint', str(checkpoint.resolve()), '--calibration', str(calibration.resolve()),
               '--worker-output', str(output.resolve()), '--device', args.device, '--batch-size', str(batch)]
        for key in ('test_seed', 'n', 'warmup', 'repeats', 'audit_samples', 'audit_repeats', 'order_seed'):
            cmd.extend(['--'+key.replace('_','-'), str(getattr(args,key))])
        if args.deterministic:
            cmd.append('--deterministic')
        env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8')
        print(f'Checking and timing frozen checkpoint {seed}, batch cap {batch}', flush=True)
        subprocess.run(cmd, check=True, env=env)
        w = read_json(output)
        for field in ('python_source_sha256', 'package_version', 'git_commit'):
            if w['environment'][field] != source[field]:
                raise ValueError('source changed between parent and worker: ' + field)
        if w['training_seed'] != seed or (seed,batch) in seen:
            raise ValueError('duplicate/mismatched training seed')
        seen.add((seed,batch))
        digest = w['checkpoint_sha256']
        if (seed in hashes and hashes[seed] != digest) or any(s != seed and d == digest for s,d in hashes.items()):
            raise ValueError('duplicate or changed checkpoint bytes')
        hashes[seed] = digest
        dataset_sha = dataset_sha or w['dataset_sha256']
        if w['dataset_sha256'] != dataset_sha:
            raise ValueError('test cohorts differ across conditions')
        workers.append(w)
    summary = {'format': FORMAT+'.complete', 'rows': summarize(workers),
               'fresh_process_per_checkpoint_and_batch': True, 'dataset_sha256': dataset_sha,
               'checkpoint_hashes': hashes, 'all_equivalence_checks_passed': True,
               'worker_reports': [f'seed-{s}-batch-{b}.json' for s,b in sorted(seen)],
               'warning': 'Lingua sample timings are in each worker report and are NOT timings for the full inference cohort.'}
    write_json(args.output_dir/'summary.json', summary)
    print('\nmodel                  batch    rollout ms    final ms    paired ratio')
    for row in summary['rows']:
        print(f"{row['model']:<22} {row['batch_size']:>5} {row['median_rollout_ms']:>13.3f} {row['median_final_only_ms']:>11.3f} {row['median_paired_speed_ratio']:>13.3f}")
    print('Completed:', args.output_dir, flush=True)
    return summary


def integer(minimum):
    def parse(value):
        result = int(value)
        if result < minimum:
            raise argparse.ArgumentTypeError(f'must be >= {minimum}')
        return result
    return parse


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replication-dir', type=Path)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    p.add_argument('--test-seed', type=integer(0), default=32000)
    p.add_argument('--n', type=integer(1), default=4000)
    p.add_argument('--batch-sizes', type=integer(1), nargs='+', default=[1024,4096])
    p.add_argument('--warmup', type=integer(0), default=5)
    p.add_argument('--repeats', type=integer(1), default=20)
    p.add_argument('--audit-samples', type=integer(1), default=4)
    p.add_argument('--audit-repeats', type=integer(1), default=3)
    p.add_argument('--order-seed', type=integer(0), default=34000)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--checkpoint', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--calibration', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--batch-size', type=integer(1), help=argparse.SUPPRESS)
    p.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    try:
        if args.worker_output is not None:
            if args.checkpoint is None or args.calibration is None or args.batch_size is None:
                raise ValueError('worker requires checkpoint, calibration and batch size')
            if args.worker_output.exists():
                raise ValueError('refusing to overwrite worker report')
            write_json(args.worker_output, run_worker(args))
        else:
            if args.replication_dir is None or args.output_dir is None:
                raise ValueError('provide replication-dir and output-dir')
            if len(set(args.batch_sizes)) != len(args.batch_sizes):
                raise ValueError('duplicate batch sizes')
            run_parent(args)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print('Error:', error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
