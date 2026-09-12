"""Time a complete request: CPU inputs -> newly generated, bound-checked receipts.

Like checking every worksheet against one pinned reference manual, preparation
may be shared but every receipt still gets its full checks. This experiment uses
frozen weights and fresh held-out inputs; it neither trains nor recalibrates.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import torch

from benchmark_verifier_setup import PreparedVerifier, digest, positive, stats
from universa_recurrent.neural.data import StructuredFlowDataset
from universa_recurrent.neural.dual_lingua import make_record as endpoint_record
from universa_recurrent.neural.dual_output import ClaimPolicy, construct_output
from universa_recurrent.neural.dual_study import (
    dataset_fingerprint, load_estimators, read_json, source_identity as package_source_identity, write_json,
)
from universa_recurrent.neural.final_state import final_candidates
from universa_recurrent.neural.retention_lingua import (
    FIELDS, dumps, loads, make_record, verify_record,
)
from universa_recurrent.neural.retention_study import ATOL, RTOL, compare_outputs, configure

FORMAT = 'universa-recurrent.verified-pipeline.v1'
MODES = ('files_per_record', 'prepare_once_per_batch', 'reuse_prepared_snapshot')
MODELS = ('shared', 'fixed_depth_4', 'direct')
PRIOR_TEST_SEEDS = {22000, 32000, 33000}
STAGES = ('reference_prepare', 'cpu_to_device', 'inference',
          'device_to_host_records_json', 'parse_bound_verify')
SCOPE = {
    'timed_start': 'Requested observed and mask tensors are resident on CPU; models are loaded.',
    'timed_end': 'All N newly generated receipts strictly parsed and accepted by bound verification.',
    'included': 'CPU-to-device transfer, one batched inference, output construction, device-to-host copies, all N records and JSON serialization, strict JSON parsing, full per-record arithmetic/geometry and reference binding, CUDA synchronization.',
    'preparation': 'files_per_record hashes and loads reference files for each receipt; prepare_once_per_batch includes one preparation inside total; reuse_prepared_snapshot excludes initial preparation, reported separately.',
    'excluded': 'Dataset synthesis, model loading (reported separately), initial snapshot preparation (reported separately), outside-timing equivalence/audit/checksum checks, report writing and process startup.',
    'warm_files': 'Warmups run every mode. File-based checking does not imply cold filesystem caches.',
    'timing_unit': 'Milliseconds for the entire N-example request; total is directly measured, never a sum of stage medians.',
    'repetition': 'Each repetition regenerates receipts for the same cohort. Counts are nested prefixes; checkpoint repetitions are not new independent inputs.',
    'binding': 'Prepared modes bind an immutable snapshot of particular checkpoint/calibration bytes; they do not track changed paths. The study checks files for changes before and after each worker.',
    'guarantees': 'Endpoint checking covers final arithmetic/geometry only; trajectory checking also covers retained geometry/diagnostics. Neither proves learned transitions, correct routing, execution authenticity, or real-world accuracy.',
}


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def elapsed(start):
    return (time.perf_counter_ns() - start) / 1e6


def batch_digest(payloads):
    """Length-delimited hash covers every serialized receipt, in request order."""
    result = hashlib.sha256()
    for payload in payloads:
        result.update(len(payload).to_bytes(8, 'big'))
        result.update(payload)
    return result.hexdigest()


def input_identity(observed, mask):
    payloads = [dumps({'observed': obs.tolist(), 'mask': m.tolist()})
                for obs, m in zip(observed, mask)]
    unique = sorted(set(payloads))
    return {'count': len(payloads), 'distinct_observed_mask_count': len(unique),
            'ordered_observed_mask_sha256': batch_digest(payloads),
            'unique_observed_mask_sha256': batch_digest(unique)}


def script_identity():
    root = Path(__file__).resolve().parent
    return {name: digest(root / name) for name in
            ('benchmark_verified_pipeline.py', 'benchmark_verifier_setup.py')}


def source_identity():
    """A portable source folder must not inherit an unrelated parent Git commit."""
    source = package_source_identity()
    root = Path(__file__).resolve().parents[1]
    try:
        git_root = subprocess.check_output(
            ['git', '-C', str(root), 'rev-parse', '--show-toplevel'],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_root = None
    if git_root is None or Path(git_root).resolve() != root:
        source['git_commit'] = None
    marker = root / 'SOURCE_COMMIT.txt'
    source['portable_source_commit'] = marker.read_text().strip() if marker.is_file() else None
    return source


@torch.inference_mode()
def infer(engine, policy, observed, mask, retention):
    history = None
    if engine.family == 'direct':
        if retention != 'endpoint':
            raise ValueError('direct control has endpoint receipts only')
        candidate = engine.model(observed, mask)
    elif retention == 'trajectory':
        history = engine.model.rollout(observed, mask)
        candidate = {key: value if key == 'prior_logits' else value[-1]
                     for key, value in history.items()}
    else:
        candidate = final_candidates(engine.model, observed, mask, validate_values=False)
    output = construct_output(candidate['states'], candidate['route_probabilities'],
                              policy, 'mixture_with_claim', validate=False)
    return output, history, candidate


def generate_receipts(engine, metadata, policy, observed_cpu, mask_cpu,
                      output, history, calibration_sha):
    """Transfer batched outputs once, then use the existing one-example schema."""
    host_output = {key: value.detach().cpu() for key, value in output.items()}
    host_history = (None if history is None else
                    {key: history[key].detach().cpu() for key in FIELDS})
    bases = engine.bases.detach().cpu() if history is not None else None
    payloads = []
    for index in range(len(observed_cpu)):
        one_output = {key: value[index:index + 1] for key, value in host_output.items()}
        one_history = (None if host_history is None else
                       {key: value[:, index:index + 1] for key, value in host_history.items()})
        endpoint = endpoint_record(
            one_output, policy, observed_cpu[index:index + 1], mask_cpu[index:index + 1],
            metadata['names'], metadata['boundaries'], model_name=engine.name,
            checkpoint_sha256=metadata['checkpoint_sha256'],
            calibration_sha256=calibration_sha, trained_depth=engine.depth,
        )
        payloads.append(dumps(make_record(endpoint, history=one_history, bases=bases)))
    return payloads


def verify_batch(payloads, mode, context, checkpoint, calibration):
    verified_count = 0
    for payload in payloads:
        record = loads(payload)
        verdict = (verify_record(record, checkpoint=checkpoint, calibration=calibration)
                   if mode == 'files_per_record' else context.verify(record))
        if not verdict['accepted'] or not verdict['checkpoint_bound'] or not verdict['calibration_bound']:
            raise ValueError('bound receipt verification failed: ' + str(verdict))
        verified_count += 1
    return verified_count


@torch.inference_mode()
def timed_request(engine, metadata, policy, observed_cpu, mask_cpu, retention,
                  mode, prepared, checkpoint, calibration, calibration_sha, device):
    """Total includes setup for prepare-once; returned tensors are audited afterward."""
    if mode not in MODES:
        raise ValueError('unknown verification mode')
    sync(device)
    start = time.perf_counter_ns()
    stages = dict.fromkeys(STAGES, 0.0)
    context = prepared
    if mode == 'prepare_once_per_batch':
        stage_start = time.perf_counter_ns()
        context = PreparedVerifier.prepare(checkpoint, calibration)
        stages['reference_prepare'] = elapsed(stage_start)
    stage_start = time.perf_counter_ns()
    observed, mask = observed_cpu.to(device), mask_cpu.to(device)
    sync(device)
    stages['cpu_to_device'] = elapsed(stage_start)
    stage_start = time.perf_counter_ns()
    output, history, _ = infer(engine, policy, observed, mask, retention)
    sync(device)
    stages['inference'] = elapsed(stage_start)
    stage_start = time.perf_counter_ns()
    payloads = generate_receipts(engine, metadata, policy, observed_cpu, mask_cpu,
                                 output, history, calibration_sha)
    sync(device)
    stages['device_to_host_records_json'] = elapsed(stage_start)
    stage_start = time.perf_counter_ns()
    verified = verify_batch(payloads, mode, context, checkpoint, calibration)
    sync(device)
    stages['parse_bound_verify'] = elapsed(stage_start)
    total_ms = elapsed(start)
    if verified != len(observed_cpu) or len(payloads) != len(observed_cpu):
        raise ValueError('incomplete request verification')
    sample = {'total_ms': total_ms, 'stages_ms': stages, 'verified_count': verified}
    return sample, payloads, output


def audit_receipts(payloads, prepared, checkpoint, calibration, observed, mask):
    """Check every saved receipt with both implementations, outside timing."""
    if len(payloads) != len(observed):
        raise ValueError('saved receipt count differs from input count')
    for index, payload in enumerate(payloads):
        record = loads(payload)
        endpoint = record['endpoint']
        if (endpoint['observed'] != observed[index].tolist()
                or endpoint['mask'] != mask[index].tolist()):
            raise ValueError('saved receipt input differs from the common cohort')
        old = verify_record(record, checkpoint=checkpoint, calibration=calibration)
        new = prepared.verify(record)
        if (not old['accepted'] or not new['accepted'] or old['checks'] != new['checks']
                or old['retained_steps_checked'] != new['retained_steps_checked']
                or not all(old['checks'].values()) or not all(new['checks'].values())
                or not old['checkpoint_bound'] or not old['calibration_bound']
                or not new['checkpoint_bound'] or not new['calibration_bound']):
            raise ValueError('original/prepared acceptance or outer property checks disagree')
    return {'receipt_count': len(payloads), 'all_accepted': True,
            'outer_checks_and_retained_steps_agree': True,
            'all_records_bound_to_checkpoint_and_calibration': True}


@torch.inference_mode()
def run_worker(args):
    torch.set_num_threads(args.cpu_threads)
    execution = configure(True)
    cp_sha, cal_sha = digest(args.checkpoint), digest(args.calibration)
    load_start = time.perf_counter_ns()
    engines, metadata, device, reserved = load_estimators(args.checkpoint, args.device)
    sync(device)
    load_ms = elapsed(load_start)
    if metadata['checkpoint_sha256'] != cp_sha:
        raise ValueError('checkpoint changed during model loading')
    calibration = read_json(args.calibration)
    reserved = reserved | {calibration['data']['seed']} | PRIOR_TEST_SEEDS
    if 'test_seed_reserved' in calibration:
        reserved.add(calibration['test_seed_reserved'])
    if args.test_seed in reserved:
        raise ValueError('test seed overlaps training, calibration or previously reserved test data')
    setup_start = time.perf_counter_ns()
    prepared = PreparedVerifier.prepare(args.checkpoint, args.calibration)
    setup_ms = elapsed(setup_start)
    settings = metadata['training']
    dataset = StructuredFlowDataset(max(args.record_counts), seed=args.test_seed,
        noise_std=settings['noise_std'], observe_probability=settings['observe_probability'])
    observed_cpu, mask_cpu = dataset.observed, dataset.mask
    identity = input_identity(observed_cpu, mask_cpu)
    if identity['distinct_observed_mask_count'] != identity['count']:
        raise ValueError('cohort contains duplicate observed/mask inputs')
    by_name = {engine.name: engine for engine in engines}
    if not set(args.models) <= set(by_name):
        raise ValueError('checkpoint lacks requested models')
    rng = random.Random(args.order_seed + settings['seed'])
    rows, saved = [], []
    for name in args.models:
        engine = by_name[name]
        if calibration['model_sources'][name] != cp_sha:
            raise ValueError('calibrated model source differs from checkpoint')
        policy = ClaimPolicy(**calibration['models'][name]['policy'])
        retentions = ('endpoint',) if name == 'direct' else ('endpoint', 'trajectory')
        for count in args.record_counts:
            obs_cpu, m_cpu = observed_cpu[:count], mask_cpu[:count]
            observed, mask = obs_cpu.to(device), m_cpu.to(device)
            reference_kind = 'endpoint' if name == 'direct' else 'trajectory'
            reference, _, reference_candidates = infer(engine, policy, observed, mask, reference_kind)
            final, _, final_candidate = infer(engine, policy, observed, mask, 'endpoint')
            equivalence = {'outputs': compare_outputs(final, reference),
                           'candidates': compare_outputs(final_candidate, reference_candidates)}
            # A direct control has no discarded history; compare its normal API.
            if name == 'direct':
                equivalence['normal_predict'] = compare_outputs(
                    final, engine.predict(observed, mask, policy, 'mixture_with_claim'))
            reference_cpu = {key: value.detach().cpu() for key, value in reference.items()}
            del reference, final, reference_candidates, final_candidate, observed, mask
            for retention in retentions:
                def run(mode):
                    sample, payloads, output = timed_request(
                        engine, metadata, policy, obs_cpu, m_cpu, retention, mode, prepared,
                        args.checkpoint, args.calibration, cal_sha, device)
                    # Complete-output comparison and digest computation are outside timing.
                    compare_outputs(output, reference_cpu)
                    sample['batch_receipts_sha256'] = batch_digest(payloads)
                    sample['serialized_bytes'] = sum(map(len, payloads))
                    return sample, payloads

                canonical = None
                for mode in MODES:
                    for _ in range(args.warmup):
                        _, canonical = run(mode)
                samples = {mode: [] for mode in MODES}
                orders = []
                for _ in range(args.repeats):
                    order = list(MODES)
                    rng.shuffle(order)
                    orders.append(order)
                    for mode in order:
                        sample, payloads = run(mode)
                        samples[mode].append(sample)
                        if canonical is None:
                            canonical = payloads
                timings = {mode: stats([sample['total_ms'] for sample in values])
                           for mode, values in samples.items()}
                receipt_hashes = {sample['batch_receipts_sha256'] for values in samples.values()
                                  for sample in values}
                rows.append({'model': name, 'retention': retention, 'record_count': count,
                    'inputs': input_identity(obs_cpu, m_cpu), 'equivalence': equivalence,
                    'timings': timings, 'samples': samples, 'mode_order_each_repeat': orders,
                    'canonical_batch_receipts_sha256': batch_digest(canonical),
                    'unique_timed_receipt_hash_count': len(receipt_hashes),
                    'bitwise_receipts_equal_across_timed_runs': len(receipt_hashes) == 1,
                    'all_timed_outputs_compared': True, 'all_timed_receipts_bound_checked': True})
                if count == max(args.record_counts):
                    audit = audit_receipts(canonical, prepared, args.checkpoint, args.calibration,
                                           obs_cpu, m_cpu)
                    saved.append({'model': name, 'retention': retention,
                        'record_count': count, 'batch_receipts_sha256': batch_digest(canonical),
                        'audit': audit, 'records': [loads(payload) for payload in canonical]})
                print(f'Seed {settings["seed"]}: {name}/{retention}, all {count} receipts checked', flush=True)
    if cp_sha != digest(args.checkpoint) or cal_sha != digest(args.calibration):
        raise ValueError('reference files changed during benchmark')
    return {'format': FORMAT + '.worker', 'training_seed': settings['seed'],
        'checkpoint_sha256': cp_sha, 'calibration_sha256': cal_sha,
        'test_seed': args.test_seed, 'reserved_data_seeds': sorted(reserved),
        'dataset_sha256': dataset_fingerprint(dataset), 'inputs': identity,
        'record_counts': args.record_counts, 'repeats': args.repeats, 'warmup': args.warmup,
        'model_load_ms': load_ms, 'initial_prepare_ms': setup_ms,
        'execution': execution, 'source': source_identity(), 'scripts_sha256': script_identity(),
        'environment': {'device': str(device), 'cuda_runtime': torch.version.cuda,
                        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'},
        'rows': rows, 'saved_receipts': saved, 'scope': SCOPE, 'atol': ATOL, 'rtol': RTOL,
        'all_equivalence_checks_passed': True, 'all_saved_receipts_audited': True,
        'no_training_or_recalibration': True}


def summarize(workers):
    groups = {}
    for worker in workers:
        for row in worker['rows']:
            key = (row['model'], row['retention'], row['record_count'])
            baseline = row['timings']['files_per_record']['median_ms']
            groups.setdefault(key, []).append({'training_seed': worker['training_seed'],
                'median_ms': {mode: row['timings'][mode]['median_ms'] for mode in MODES},
                'files_divided_by_mode_time': {
                    mode: baseline / row['timings'][mode]['median_ms'] for mode in MODES}})
    return [{'model': key[0], 'retention': key[1], 'record_count': key[2],
             'checkpoint_pairs': len(pairs), 'per_seed': pairs,
             'median_checkpoint_time_ms': {mode: statistics.median(p['median_ms'][mode] for p in pairs)
                                           for mode in MODES},
             'median_paired_speed_ratio': {mode: statistics.median(
                 p['files_divided_by_mode_time'][mode] for p in pairs) for mode in MODES}}
            for key, pairs in sorted(groups.items())]


def run_parent(args):
    if args.output_dir.exists():
        raise ValueError('refusing to overwrite output directory')
    checkpoints = sorted(args.replication_dir.glob('weights-*.pt'))
    if not checkpoints:
        raise ValueError('no weights-*.pt in replication directory')
    jobs, seen = [], set()
    for checkpoint in checkpoints:
        seed = int(checkpoint.stem.removeprefix('weights-'))
        calibration = args.replication_dir / f'study-{seed}' / 'calibration.json'
        if seed in seen or not calibration.is_file():
            raise ValueError('duplicate seed or missing calibration')
        seen.add(seed)
        jobs.append((checkpoint, calibration, seed, digest(checkpoint), digest(calibration)))
    if len({job[3] for job in jobs}) != len(jobs):
        raise ValueError('duplicate checkpoint bytes across training seeds')
    random.Random(args.order_seed).shuffle(jobs)
    source, scripts = source_identity(), script_identity()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / 'PLAN.json', {
        'format': FORMAT + '.plan', 'source': source, 'scripts_sha256': scripts,
        'record_counts': args.record_counts, 'models': args.models, 'test_seed': args.test_seed,
        'repeats': args.repeats, 'warmup': args.warmup, 'order_seed': args.order_seed,
        'device': args.device, 'cpu_threads': args.cpu_threads, 'scope': SCOPE,
        'job_order': [job[2] for job in jobs], 'no_training_or_recalibration': True,
        'fresh_process_per_checkpoint': True})
    workers, files = [], []
    cohort = None
    for checkpoint, calibration, seed, cp_sha, cal_sha in jobs:
        output = args.output_dir / f'seed-{seed}.json'
        cmd = [sys.executable, str(Path(__file__).resolve()),
               '--checkpoint', str(checkpoint.resolve()), '--calibration', str(calibration.resolve()),
               '--worker-output', str(output.resolve()), '--device', args.device,
               '--record-counts', *map(str, args.record_counts), '--models', *args.models]
        for key in ('test_seed', 'repeats', 'warmup', 'order_seed', 'cpu_threads'):
            cmd.extend(['--' + key.replace('_', '-'), str(getattr(args, key))])
        subprocess.run(cmd, check=True, env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        worker = read_json(output)
        if (worker['training_seed'] != seed or worker['checkpoint_sha256'] != cp_sha
                or worker['calibration_sha256'] != cal_sha
                or digest(checkpoint) != cp_sha or digest(calibration) != cal_sha):
            raise ValueError('reference files changed or training seed mismatched')
        if worker['source'] != source or worker['scripts_sha256'] != scripts:
            raise ValueError('source changed between parent and worker')
        identity = (worker['dataset_sha256'], worker['inputs'])
        cohort = identity if cohort is None else cohort
        if identity != cohort:
            raise ValueError('input cohorts differ across checkpoints')
        workers.append(worker)
        files.append(output.name)
    summary = {'format': FORMAT + '.complete', 'source': source, 'scripts_sha256': scripts,
        'worker_reports': files, 'checkpoint_runs': len(workers),
        'dataset_sha256': cohort[0], 'inputs': cohort[1], 'rows': summarize(workers),
        'all_equivalence_checks_passed': True, 'all_saved_receipts_audited': True,
        'saved_receipt_count': sum(group['record_count'] for worker in workers
                                   for group in worker['saved_receipts']),
        'scope': SCOPE, 'warning': 'Descriptive timings on a repeated common cohort, not statistical significance. More checkpoints and timing repeats do not create new independent inputs.'}
    write_json(args.output_dir / 'summary.json', summary)
    print('COMPLETE:', args.output_dir / 'summary.json', flush=True)
    return summary


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('must be nonnegative')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replication-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--record-counts', type=positive, nargs='+', default=[1, 16, 64, 256])
    parser.add_argument('--models', choices=MODELS, nargs='+', default=list(MODELS))
    parser.add_argument('--test-seed', type=nonnegative, default=36000)
    parser.add_argument('--order-seed', type=nonnegative, default=37000)
    parser.add_argument('--repeats', type=positive, default=5)
    parser.add_argument('--warmup', type=nonnegative, default=1)
    parser.add_argument('--cpu-threads', type=positive, default=1)
    parser.add_argument('--checkpoint', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--calibration', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if len(set(args.record_counts)) != len(args.record_counts) or len(set(args.models)) != len(args.models):
            raise ValueError('record counts and models must be unique')
        if args.worker_output is not None:
            if args.checkpoint is None or args.calibration is None:
                raise ValueError('worker needs checkpoint and calibration')
            if args.worker_output.exists():
                raise ValueError('refusing to overwrite worker report')
            write_json(args.worker_output, run_worker(args))
        else:
            if args.replication_dir is None or args.output_dir is None:
                raise ValueError('provide replication-dir and output-dir')
            run_parent(args)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print('Error:', error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
