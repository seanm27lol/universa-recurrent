"""Compare batch receipt processing with the already-prepared scalar verifier.

Every mode starts at CPU-resident input and ends at a verdict for EVERY record.
The model, policy, individual JSON format, and numerical guarantees are fixed.
This is an opt-in warmed-request experiment, not a speedup claim.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

import torch

import benchmark_verified_pipeline as original
from benchmark_verifier_setup import PreparedVerifier, digest, positive, stats
from batch_receipt_verifier import (BatchPreparedVerifier, expected_inputs,
    generate_payloads, scalar_batch)
from universa_recurrent.neural.data import StructuredFlowDataset
from universa_recurrent.neural.dual_output import ClaimPolicy
from universa_recurrent.neural.dual_study import (load_estimators, read_json,
    write_json, dataset_fingerprint)
from universa_recurrent.neural.retention_lingua import dumps, loads
from universa_recurrent.neural.retention_study import configure, compare_outputs

FORMAT = 'universa-recurrent.batched-pipeline.v1'
MODES = ('prepared_scalar', 'batch_check', 'batch_generate_and_check')
SCOPE = {
    'included': 'CPU input transfer; fixed-depth inference and output construction; host copies; serialization of every individual receipt; strict JSON parsing; per-record reference, request-input and geometry checks; GPU synchronization.',
    'excluded': 'Model loading and verifier preparation (reported separately), synthesis, process startup, network/queueing, report-file writing and outside-timing differential/corruption tests.',
    'baseline': 'Already-prepared scalar verifier, NOT reloading checkpoint files per receipt.',
    'modes': 'prepared_scalar uses original generation/checks; batch_check changes only checking; batch_generate_and_check additionally batches host conversion during generation.',
    'identity': 'Same canonical per-record JSON bytes in every mode at a fixed checkpoint/model/count/retention; expected inputs are caller-owned and checked in every mode.',
    'guarantees': 'Mathematical properties and pinned snapshot identity only. No verified neural transitions, route truth, hidden-feature semantics or execution authentication.',
    'cohort': 'Same cohort across checkpoints, counts are nested prefixes. Timing repeats and model variants are not independent inputs. Prior test cohort reuse is intentional for performance comparison.',
}


def scripts_identity():
    root = Path(__file__).resolve().parent
    return {name: digest(root / name) for name in (
        'benchmark_batched_pipeline.py', 'batch_receipt_verifier.py',
        'benchmark_verified_pipeline.py', 'benchmark_verifier_setup.py')}


def verdict_signature(result):
    return [(v['accepted'], v.get('retained_steps_checked'),
             v.get('checkpoint_bound', False), v.get('calibration_bound', False))
            for v in result['verdicts']]


def check(context, batched, mode, payloads, expectations):
    if mode not in MODES:
        raise ValueError('unknown verification mode')
    if mode == 'prepared_scalar':
        return scalar_batch(context, payloads, expected_count=len(expectations), expected=expectations)
    return batched.verify_payloads(payloads, expected_count=len(expectations), expected=expectations)


def rejection_tests(payloads, expectations, context, batched):
    """Corrupt the LAST member, not just an easy first record. Never edit weights."""
    trials = {}
    def mutate(name, edit):
        records = [loads(p) for p in payloads]
        edit(records[-1]); trials[name] = [dumps(r) for r in records]
    mutate('last_estimate', lambda r: r['endpoint']['estimate'].__setitem__(0, r['endpoint']['estimate'][0] + 1))
    mutate('checkpoint_identity', lambda r: r['endpoint'].__setitem__('checkpoint_sha256', '0' * 64))
    mutate('calibration_identity', lambda r: r['endpoint'].__setitem__('calibration_sha256', '0' * 64))
    mutate('false_scope', lambda r: r['scope'].__setitem__('execution_authenticated', True))
    trials['missing_last'] = payloads[:-1]
    trials['extra_last'] = payloads + payloads[-1:]
    trials['malformed_last'] = payloads[:-1] + [b'{"a":1,"a":2}']
    if len(payloads) > 1:
        trials['duplicate_last_input'] = payloads[:-1] + payloads[:1]
        trials['reordered_inputs'] = list(reversed(payloads))
    if loads(payloads[-1])['retention'] == 'trajectory':
        mutate('last_trajectory', lambda r: r['trajectory']['states'][0][0].__setitem__(0, 1000))
    for name, bad in trials.items():
        for mode in ('prepared_scalar', 'batch_check'):
            result = check(context, batched, mode, bad, expectations)
            if result['accepted'] or result['checked_count'] != len(bad):
                raise ValueError(f'{name}/{mode}: corrupt/incomplete batch was not rejected')
    return {'tests': list(trials), 'both_rejected_every_trial': True}


@torch.inference_mode()
def request(engine, metadata, policy, obs, mask, retention, mode,
            context, batched, cal_sha, device, expectations):
    original.sync(device)
    start = time.perf_counter_ns()
    t = time.perf_counter_ns()
    x, m = obs.to(device), mask.to(device)
    original.sync(device)
    stages = {'input_transfer': original.elapsed(t)}
    t = time.perf_counter_ns()
    output, history, _ = original.infer(engine, policy, x, m, retention)
    original.sync(device)
    stages['inference'] = original.elapsed(t)
    t = time.perf_counter_ns()
    gen = generate_payloads if mode == 'batch_generate_and_check' else original.generate_receipts
    payloads = gen(engine, metadata, policy, obs, mask, output, history, cal_sha)
    original.sync(device)
    stages['receipt_generation_json'] = original.elapsed(t)
    t = time.perf_counter_ns()
    result = check(context, batched, mode, payloads, expectations)
    original.sync(device)
    stages['parse_bound_check'] = original.elapsed(t)
    total = original.elapsed(start)
    if not result['accepted'] or result['checked_count'] != len(obs):
        raise ValueError('Request did not verify every expected receipt')
    return dict(total_ms=total, stages_ms=stages,
        checked_count=result['checked_count'], accepted_count=result['accepted_count'],
        vectorized_records=result.get('vectorized_records', 0),
        scalar_fallback_records=result.get('scalar_fallback_records', 0)), payloads, output


def worker(args):
    torch.set_num_threads(args.cpu_threads)
    execution = configure(True)
    identities = scripts_identity()
    cp_sha, cal_sha = digest(args.checkpoint), digest(args.calibration)
    start = time.perf_counter_ns()
    engines, metadata, device, reserved = load_estimators(args.checkpoint, args.device)
    original.sync(device)
    model_load_ms = original.elapsed(start)
    artifact = read_json(args.calibration)
    if args.test_seed in reserved | {artifact['data']['seed']}:
        raise ValueError('test seed overlaps training/calibration')
    if artifact['checkpoint_sha256'] != cp_sha or metadata['checkpoint_sha256'] != cp_sha:
        raise ValueError('checkpoint/calibration mismatch')
    start = time.perf_counter_ns()
    context = PreparedVerifier.prepare(args.checkpoint, args.calibration)
    batched = BatchPreparedVerifier(context)
    preparation_ms = original.elapsed(start)
    data = StructuredFlowDataset(max(args.record_counts), seed=args.test_seed,
        noise_std=metadata['training']['noise_std'],
        observe_probability=metadata['training']['observe_probability'])
    identity = original.input_identity(data.observed, data.mask)
    if identity['distinct_observed_mask_count'] != len(data):
        raise ValueError('non-distinct benchmark inputs')
    available = {e.name: e for e in engines}
    if not set(args.models) <= set(available):
        raise ValueError('missing model controls')
    rng = random.Random(args.order_seed + metadata['training']['seed'])
    cases = [(model, count, kind) for model in args.models for count in args.record_counts
             for kind in (('endpoint',) if model == 'direct' else ('endpoint', 'trajectory'))]
    rng.shuffle(cases)
    rows, saved = [], []
    for model, count, kind in cases:
        engine = available[model]
        if artifact['model_sources'][model] != cp_sha:
            raise ValueError('calibrated model mismatch')
        policy = ClaimPolicy(**artifact['models'][model]['policy'])
        obs, mask = data.observed[:count], data.mask[:count]
        expected = expected_inputs(obs, mask, model, kind)
        # Canonical bytes and differential checks occur outside measured repeats.
        _, canonical, reference = request(engine, metadata, policy, obs, mask, kind,
            'prepared_scalar', context, batched, cal_sha, device, expected)
        reference = {k: v.detach().cpu() for k, v in reference.items()}
        canon_hash = original.batch_digest(canonical)
        scalar = check(context, batched, 'prepared_scalar', canonical, expected)
        vector = check(context, batched, 'batch_check', canonical, expected)
        if verdict_signature(scalar) != verdict_signature(vector):
            raise ValueError('scalar/vector per-record verdict mismatch')
        safety = rejection_tests(canonical, expected, context, batched)
        samples = {mode: [] for mode in MODES}
        def run(mode):
            sample, payloads, out = request(engine, metadata, policy, obs, mask, kind,
                mode, context, batched, cal_sha, device, expected)
            compare_outputs(out, reference)
            if payloads != canonical:
                raise ValueError('Receipt bytes differ across modes or repeats')
            sample['receipts_sha256'] = original.batch_digest(payloads)
            sample['serialized_bytes'] = sum(map(len, payloads))
            return sample
        for mode in MODES:
            for _ in range(args.warmup):
                run(mode)
        orders = []
        for _ in range(args.repeats):
            order = list(MODES); rng.shuffle(order); orders.append(order)
            for mode in order:
                samples[mode].append(run(mode))
        rows.append(dict(model=model, retention=kind, record_count=count,
            samples=samples, timings={mode: stats([s['total_ms'] for s in v]) for mode, v in samples.items()},
            mode_order_each_repeat=orders, rejection_tests=safety,
            per_record_verdicts_agree=True, identical_receipt_bytes=True,
            canonical_receipts_sha256=canon_hash))
        if count == max(args.record_counts):
            saved.append(dict(model=model, retention=kind, record_count=count,
                records=[loads(p) for p in canonical], verdict_signature=verdict_signature(vector),
                batch_receipts_sha256=canon_hash))
        print(f"Seed {metadata['training']['seed']}: {model}/{kind}, {count} receipts: all modes agree", flush=True)
    if digest(args.checkpoint) != cp_sha or digest(args.calibration) != cal_sha or scripts_identity() != identities:
        raise ValueError('source or reference changed during run')
    return dict(format=FORMAT+'.worker', training_seed=metadata['training']['seed'],
        checkpoint_sha256=cp_sha, calibration_sha256=cal_sha, source=original.source_identity(),
        scripts_sha256=identities, execution=execution, environment=dict(device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        torch=str(torch.__version__), cuda_runtime=torch.version.cuda),
        model_load_ms=model_load_ms, initial_prepare_ms=preparation_ms, dataset_sha256=dataset_fingerprint(data),
        test_seed=args.test_seed, inputs=identity, repeats=args.repeats, warmup=args.warmup,
        rows=rows, saved_receipts=saved, scope=SCOPE)


def summarize(workers):
    groups = {}
    for w in workers:
        for r in w['rows']:
            groups.setdefault((r['model'], r['retention'], r['record_count']), []).append(
                dict(training_seed=w['training_seed'], median_ms={m:r['timings'][m]['median_ms'] for m in MODES}))
    result = []
    for (model, kind, count), items in sorted(groups.items()):
        result.append(dict(model=model, retention=kind, record_count=count,
            checkpoint_pairs=len(items), per_seed=items,
            median_checkpoint_ms={m:statistics.median(x['median_ms'][m] for x in items) for m in MODES},
            median_paired_speed_ratio={m:statistics.median(x['median_ms']['prepared_scalar']/x['median_ms'][m]
                for x in items) for m in MODES}))
    return result


def parent(args):
    if args.output_dir.exists():
        raise ValueError('refusing existing output folder')
    jobs = []
    for cp in sorted(args.replication_dir.glob('weights-*.pt')):
        seed = int(cp.stem.removeprefix('weights-'))
        cal = args.replication_dir / f'study-{seed}/calibration.json'
        if not cal.is_file():
            raise ValueError(f'missing calibration for seed {seed}')
        jobs.append((seed, cp, cal, digest(cp), digest(cal)))
    if not jobs or len({j[3] for j in jobs}) != len(jobs):
        raise ValueError('distinct saved checkpoints required')
    random.Random(args.order_seed).shuffle(jobs)
    args.output_dir.mkdir(parents=True)
    source, scripts = original.source_identity(), scripts_identity()
    write_json(args.output_dir/'PLAN.json', dict(format=FORMAT+'.plan', source=source,
        scripts_sha256=scripts, test_seed=args.test_seed, record_counts=args.record_counts,
        models=args.models, repeats=args.repeats, warmup=args.warmup, order_seed=args.order_seed,
        cpu_threads=args.cpu_threads, device=args.device, job_order=[j[0] for j in jobs], scope=SCOPE))
    workers=[]
    for seed, cp, cal, ch, ah in jobs:
        out=args.output_dir/f'seed-{seed}.json'
        cmd=[sys.executable, str(Path(__file__).resolve()), '--checkpoint',str(cp.resolve()),
             '--calibration',str(cal.resolve()), '--worker-output',str(out.resolve()),
             '--device',args.device,'--record-counts',*map(str,args.record_counts),'--models',*args.models]
        for key in ('test_seed','order_seed','repeats','warmup','cpu_threads'):
            cmd += ['--'+key.replace('_','-'), str(getattr(args,key))]
        subprocess.run(cmd,check=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        w=read_json(out)
        if (w['checkpoint_sha256'] != ch or w['calibration_sha256'] != ah or w['training_seed'] != seed
            or digest(cp)!=ch or digest(cal)!=ah or w['source'] != source or w['scripts_sha256'] != scripts):
            raise ValueError('parent/worker identity mismatch')
        if workers and (w['dataset_sha256'],w['inputs']) != (workers[0]['dataset_sha256'],workers[0]['inputs']):
            raise ValueError('cohort mismatch')
        workers.append(w)
    summary=dict(format=FORMAT+'.complete', checkpoint_runs=len(workers), rows=summarize(workers),
        dataset_sha256=workers[0]['dataset_sha256'], inputs=workers[0]['inputs'],
        worker_reports=[f'seed-{w["training_seed"]}.json' for w in workers],
        all_outputs_and_receipts_agree=True, all_rejection_tests_passed=True, scope=SCOPE)
    write_json(args.output_dir/'summary.json',summary)
    print('\nmodel / receipt / count      scalar ms    batch-check ms    batch-both ms')
    for r in summary['rows']:
        print(f"{r['model']}/{r['retention']}/{r['record_count']:<4} " +
              ' '.join(f"{r['median_checkpoint_ms'][m]:12.3f}" for m in MODES))
    print('COMPLETE:',args.output_dir/'summary.json')
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('replication-dir','output-dir','checkpoint','calibration','worker-output'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--models',nargs='+',choices=['shared','fixed_depth_4','direct'],default=['shared','fixed_depth_4','direct'])
    p.add_argument('--record-counts',type=positive,nargs='+',default=[1,16,64,256])
    p.add_argument('--test-seed',type=original.nonnegative,default=46000)
    p.add_argument('--order-seed',type=original.nonnegative,default=48000)
    p.add_argument('--repeats',type=positive,default=10)
    p.add_argument('--warmup',type=original.nonnegative,default=2)
    p.add_argument('--cpu-threads',type=positive,default=1)
    a=p.parse_args(argv)
    try:
        if len(set(a.record_counts))!=len(a.record_counts) or len(set(a.models))!=len(a.models) or max(a.record_counts)>4096:
            raise ValueError('duplicate conditions or count above 4096')
        if a.worker_output:
            if not a.checkpoint or not a.calibration or a.worker_output.exists():
                raise ValueError('worker needs references and unused output path')
            write_json(a.worker_output,worker(a))
        else:
            if not a.replication_dir or not a.output_dir:
                raise ValueError('provide replication-dir and output-dir')
            parent(a)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, subprocess.CalledProcessError) as e:
        print('Error:',e,file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
