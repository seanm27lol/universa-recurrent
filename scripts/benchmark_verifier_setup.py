"""Check the fixed reference files once, but check EVERY new receipt.

Analogy: validate the laboratory's reference manual once per session, then check
all calculations against that pinned edition. This is NOT caching a PASS verdict.
Run with --help. Models, training, record formats and old checkers are unchanged.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
from zipfile import ZipFile, BadZipFile

import numpy as np


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class PreparedVerifier:
    """Trusted in-process snapshot, constructed ONLY from trusted local files.

    This deliberately binds to a specific artifact version, not mutable current
    file paths. Construct a new instance when files change. It cannot authenticate
    the producer, prove routing correctness, or verify learned transitions.
    No deserialization of a caller-supplied 'prepared context' is provided.
    """
    _snapshot: bytes

    @classmethod
    def prepare(cls, checkpoint: Path, calibration: Path) -> 'PreparedVerifier':
        from universa_recurrent.neural.v2_train import load_v2_checkpoint
        from universa_recurrent.neural.v2 import DirectMultiHypothesisNet, MultiHypothesisRecurrentNet
        from universa_recurrent.neural.dual_study import read_json, FORMAT
        from universa_recurrent.neural.dual_output import ClaimPolicy
        cp_sha, cal_sha = digest(checkpoint), digest(calibration)
        main, controls, meta, _ = load_v2_checkpoint(checkpoint, device_name='cpu')
        artifact = read_json(calibration)
        if (meta['checkpoint_sha256'] != cp_sha or artifact.get('format') != FORMAT + '.calibration'
                or artifact.get('checkpoint_sha256') != cp_sha):
            raise ValueError('checkpoint/calibration identity mismatch')
        entries = {}
        for name, model in {'shared': main, **controls}.items():
            if not isinstance(model, (DirectMultiHypothesisNet, MultiHypothesisRecurrentNet)):
                continue  # Ambient estimates and analytic references have no candidate claim here.
            entry = artifact.get('models', {}).get(name)
            if entry is None:
                continue
            policy = entry['policy']
            if (set(policy) != {'threshold', 'target_coverage'}
                    or artifact['model_sources'].get(name) != cp_sha):
                raise ValueError('invalid calibrated model identity/policy')
            ClaimPolicy(**policy)
            entries[name] = {'depth': 1 if isinstance(model, DirectMultiHypothesisNet) else model.config.steps,
                             'policy': policy}
        if not entries:
            raise ValueError('no supported calibrated models')
        b = meta['boundaries']
        boundaries = b.detach().cpu().tolist() if hasattr(b, 'detach') else np.asarray(b).tolist()
        snapshot = encoded({'checkpoint_sha256': cp_sha, 'calibration_sha256': cal_sha,
                            'library': {'names': meta['names'], 'boundaries': boundaries}, 'models': entries})
        # Detect ordinary concurrent file changes during preparation. This is not
        # protection against a malicious process controlling these local files.
        if digest(checkpoint) != cp_sha or digest(calibration) != cal_sha:
            raise ValueError('reference files changed during preparation')
        result = object.__new__(cls)
        object.__setattr__(result, '_snapshot', snapshot)
        return result

    def manifest(self) -> dict:
        """Return a fresh copy so callers cannot mutate the cached reference."""
        return json.loads(self._snapshot)

    def verify(self, record: dict) -> dict:
        from universa_recurrent.neural.retention_lingua import verify_record
        checked = verify_record(record)  # ALL existing record arithmetic/geometry still runs.
        if not checked['accepted']:
            return checked
        try:
            snapshot, endpoint = self.manifest(), record['endpoint']
            entry = snapshot['models'].get(endpoint['model'])
            for key in ('checkpoint_sha256', 'calibration_sha256', 'library'):
                if endpoint[key] != snapshot[key]:
                    raise ValueError('pinned ' + key + ' mismatch')
            if entry is None or endpoint['trained_depth'] != entry['depth'] or endpoint['policy'] != entry['policy']:
                raise ValueError('pinned model/depth/policy mismatch')
        except (ValueError, KeyError, TypeError) as error:
            return {'accepted': False, 'reason': str(error), 'checks': checked.get('checks', {})}
        return {**checked, 'checkpoint_bound': True, 'calibration_bound': True,
                'binding_scope': 'trusted immutable snapshot; current disk state is NOT rechecked',
                'snapshot_sha256': hashlib.sha256(self._snapshot).hexdigest()}


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return n


def stats(values):
    return {'median_ms': statistics.median(values), 'p10_ms': float(np.quantile(values, .1)),
            'p90_ms': float(np.quantile(values, .9)), 'samples_ms': values}


def read_workers(archive: Path) -> list[dict]:
    """Read only bounded JSON members, never extract paths or execute ZIP contents."""
    from universa_recurrent.neural.retention_lingua import loads
    with ZipFile(archive) as z:
        infos = z.infolist()
        if len(infos) > 1000 or sum(i.file_size for i in infos) > 100_000_000:
            raise ValueError('report ZIP too large')
        if len({i.filename for i in infos}) != len(infos):
            raise ValueError('duplicate ZIP member')
        workers = []
        for i in infos:
            if not i.filename.endswith('.json'):
                continue
            report = loads(z.read(i))
            if not isinstance(report, dict):
                raise ValueError('JSON report must be an object')
            if (report.get('format') == 'universa-recurrent.retention-study.v1.worker'
                    and report.get('batch_size') == 1024):
                workers.append(report)
    seeds = [w['training_seed'] for w in workers]
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('training seed must be a nonnegative integer')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('one batch-1024 report per distinct training seed required')
    if len({w['dataset_sha256'] for w in workers}) != 1:
        raise ValueError('workers used different cohorts')
    return sorted(workers, key=lambda w: w['training_seed'])


def benchmark_worker(worker, checkpoint, calibration, counts, repeats, order_seed):
    from universa_recurrent.neural.retention_lingua import loads, verify_record
    cp_sha, cal_sha = digest(checkpoint), digest(calibration)
    if cp_sha != worker['checkpoint_sha256'] or cal_sha != worker['calibration_sha256']:
        raise ValueError('local reference files do not match saved receipts')
    started = time.perf_counter_ns()
    prepared = PreparedVerifier.prepare(checkpoint, calibration)
    first_setup_ms = (time.perf_counter_ns() - started) / 1e6
    rows = []
    rng = random.Random(order_seed + worker['training_seed'])
    for group in worker['lingua_costs']:
        for kind in ('endpoint', 'trajectory'):
            records = group['sample_records'][kind]
            if not records or len(records) != group['sample_count']:
                raise ValueError('missing sampled records')
            # Explicitly re-use the saved first four records. Increasing count
            # measures amortization, NOT additional independent audit coverage.
            payloads = [encoded(r) for r in records]
            for r in records:
                old = verify_record(r, checkpoint=checkpoint, calibration=calibration)
                new = prepared.verify(r)
                if not old['accepted'] or not new['accepted'] or old['checks'] != new['checks']:
                    raise ValueError('original and prepared verification disagree')
            for count in counts:
                batch = [payloads[i % len(payloads)] for i in range(count)]
                def run(mode):
                    context = PreparedVerifier.prepare(checkpoint, calibration) if mode == 'prepare_once_per_batch' else prepared
                    for payload in batch:
                        record = loads(payload)
                        verdict = (verify_record(record, checkpoint=checkpoint, calibration=calibration)
                                   if mode == 'files_per_record' else context.verify(record))
                        if not verdict['accepted']:
                            raise ValueError('record check failed during timing: ' + verdict['reason'])
                timings = {m: [] for m in ('files_per_record', 'prepare_once_per_batch', 'reuse_prepared_snapshot')}
                for mode in timings:
                    run(mode)  # Warm imports/caches. "Files" does not mean cold filesystem I/O.
                order = list(timings)
                orders = []
                for _ in range(repeats):
                    rng.shuffle(order);orders.append(list(order))
                    for mode in order:
                        start = time.perf_counter_ns();run(mode)
                        timings[mode].append((time.perf_counter_ns() - start) / 1e6)
                rows.append({'model': group['model'], 'retention': kind, 'record_count': count,
                             'distinct_saved_records': len(payloads), 'timings': {m: stats(v) for m,v in timings.items()},
                             'mode_order_each_repeat': orders})
    if cp_sha != digest(checkpoint) or cal_sha != digest(calibration):
        raise ValueError('reference files changed during benchmark')
    return {'training_seed': worker['training_seed'], 'checkpoint_sha256': cp_sha,
            'calibration_sha256': cal_sha, 'initial_prepare_ms': first_setup_ms,
            'accepted_decisions_and_existing_checks_agree': True, 'rows': rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', required=True, type=Path)
    parser.add_argument('--replication-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--record-counts', nargs='+', type=positive, default=[1,4,16,64])
    parser.add_argument('--repeats', type=positive, default=3)
    parser.add_argument('--cpu-threads', type=positive, default=1)
    parser.add_argument('--order-seed', type=int, default=35000)
    args = parser.parse_args(argv)
    try:
        if len(set(args.record_counts)) != len(args.record_counts) or args.order_seed < 0:
            raise ValueError('counts must be unique and order seed nonnegative')
        if args.output_dir.exists():
            raise ValueError('refusing to overwrite output directory')
        workers = read_workers(args.reports)
        refs = [(args.replication_dir/f'weights-{w["training_seed"]}.pt',
                 args.replication_dir/f'study-{w["training_seed"]}/calibration.json') for w in workers]
        for cp, cal in refs:
            if not cp.is_file() or not cal.is_file():
                raise ValueError('missing local checkpoint or calibration file')
        import torch
        from universa_recurrent.neural.dual_study import write_json, source_identity
        torch.set_num_threads(args.cpu_threads)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        plan = {'format': 'universa-recurrent.verifier-setup.v1', 'source': source_identity(),
                'script_sha256': digest(Path(__file__)), 'input_zip_sha256': digest(args.reports),
                'record_counts': args.record_counts, 'repeats': args.repeats, 'order_seed': args.order_seed,
                'cpu_threads': torch.get_num_threads(), 'cpu_interop_threads': torch.get_num_interop_threads(),
                'no_training': True, 'no_gpu_inference': True, 'new_independent_record_samples': False,
                'scope': 'CPU JSON parsing, complete per-record property checks and artifact binding. No inference, generation or filesystem report-writing in timing. Prepared modes bind a pinned reference snapshot, not changing file paths.'}
        write_json(args.output_dir/'PLAN.json', plan)
        results = []
        for w, (cp, cal) in zip(workers, refs):
            print('Checking and timing seed', w['training_seed'], flush=True)
            result = benchmark_worker(w, cp, cal, args.record_counts, args.repeats, args.order_seed)
            write_json(args.output_dir/f'seed-{w["training_seed"]}.json', result)
            results.append(result)
        summary = {'format': plan['format']+'.complete', 'all_checks_agreed': True,
                   'checkpoint_runs': len(results), 'results': results, 'scope': plan['scope'],
                   'warning': 'Three repeats by default; descriptive timing, not statistical significance. Repeated saved receipts measure setup amortization, not generalization or execution authenticity.'}
        write_json(args.output_dir/'summary.json', summary)
        print('COMPLETE:', args.output_dir/'summary.json')
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, BadZipFile) as e:
        print('Error:', e, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
