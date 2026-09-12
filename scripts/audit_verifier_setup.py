"""Recompute verifier setup timings from a reports-only ZIP using the stdlib.

Like checking a laboratory notebook, this checks reported samples and accounting.
It does not replay receipts or authenticate an execution. Source hashes identify
reported artifacts; the archive contains neither those artifacts nor receipts.
ZIP members are read with size limits, never extracted or executed.

Usage: python scripts/audit_verifier_setup.py reports.zip --output recomputed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
from zipfile import BadZipFile, ZipFile


FORMAT = 'universa-recurrent.verifier-setup.v1'
MODELS = ('shared', 'fixed_depth_4')
RETENTIONS = ('endpoint', 'trajectory')
MODES = ('files_per_record', 'prepare_once_per_batch', 'reuse_prepared_snapshot')
MAX_ARCHIVE_BYTES = 50_000_000
MAX_MEMBER_BYTES = 10_000_000
MAX_TOTAL_BYTES = 50_000_000
MAX_MEMBERS = 1000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(data: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f'duplicate JSON field: {key}')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError(f'nonfinite JSON constant: {value}')

    def finite(value):
        if isinstance(value, float):
            require(math.isfinite(value), 'nonfinite JSON number')
        elif isinstance(value, dict):
            for child in value.values():
                finite(child)
        elif isinstance(value, list):
            for child in value:
                finite(child)

    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
        require(isinstance(value, dict), 'JSON report must be an object')
        finite(value)
    except (UnicodeError, RecursionError) as error:
        raise ValueError('invalid JSON encoding or nesting') from error
    return value


def positive_number(value, label):
    require(type(value) in (int, float) and math.isfinite(value) and value > 0,
            f'{label} must be a positive finite number')
    return value


def integer(value, label, minimum=1):
    require(type(value) is int and value >= minimum, f'invalid {label}')
    return value


def sha256(value, label):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
            f'invalid reported {label}')


def quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def read_archive(path: Path):
    require(path.stat().st_size <= MAX_ARCHIVE_BYTES, 'report ZIP too large')
    with ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(infos) <= MAX_MEMBERS, 'too many ZIP members')
        require(len(names) == len(set(names)), 'duplicate ZIP member')
        require(sum(info.file_size for info in infos) <= MAX_TOTAL_BYTES,
                'uncompressed report ZIP too large')
        reports = {}
        for info in infos:
            require(info.filename == 'results/' or re.fullmatch(
                r'results/(PLAN|summary|seed-[0-9]+)\.json', info.filename) is not None,
                f'unexpected ZIP member: {info.filename}')
            require(info.file_size <= MAX_MEMBER_BYTES, 'ZIP member too large')
            if info.is_dir():
                continue
            with archive.open(info) as handle:
                data = handle.read(MAX_MEMBER_BYTES + 1)
            require(len(data) <= MAX_MEMBER_BYTES and len(data) == info.file_size,
                    'ZIP member size mismatch or limit exceeded')
            reports[info.filename] = read_json(data)
    require('results/PLAN.json' in reports and 'results/summary.json' in reports,
            'missing PLAN or summary')
    return reports


def audit(path: Path) -> dict:
    reports = read_archive(path)
    plan, summary = reports['results/PLAN.json'], reports['results/summary.json']
    require(plan.get('format') == FORMAT, 'unsupported plan format')
    counts = plan.get('record_counts')
    require(isinstance(counts, list) and counts, 'missing planned record counts')
    for count in counts:
        integer(count, 'record count')
    require(len(set(counts)) == len(counts), 'duplicate planned record count')
    repeats = integer(plan.get('repeats'), 'repeats')
    order_seed = integer(plan.get('order_seed'), 'order seed', 0)
    for field in ('cpu_threads', 'cpu_interop_threads'):
        integer(plan.get(field), field)
    for field, expected in (('no_training', True), ('no_gpu_inference', True),
                            ('new_independent_record_samples', False)):
        require(plan.get(field) is expected, f'unexpected reported {field}')
    require(isinstance(plan.get('scope'), str) and plan['scope'], 'missing reported scope')
    for field in ('script_sha256', 'input_zip_sha256'):
        sha256(plan.get(field), field)
    source = plan.get('source')
    require(isinstance(source, dict), 'missing source identity')
    sha256(source.get('python_source_sha256'), 'python source hash')
    require(isinstance(source.get('git_commit'), str) and re.fullmatch(
        r'[0-9a-f]{40}', source['git_commit']) is not None, 'invalid reported source commit')
    for field in ('package_version', 'python', 'numpy', 'torch'):
        require(isinstance(source.get(field), str) and source[field],
                f'missing reported source {field}')

    grid = [(model, retention, count) for model in MODELS
            for retention in RETENTIONS for count in counts]
    workers, worker_medians, all_samples, hashes = [], {}, [], set()
    statistic_checks = 0
    for name, worker in reports.items():
        if not name.startswith('results/seed-'):
            continue
        seed = integer(worker.get('training_seed'), 'training seed', 0)
        require(name == f'results/seed-{seed}.json', 'worker filename/seed mismatch')
        require(seed not in worker_medians, 'duplicate worker seed')
        for field in ('checkpoint_sha256', 'calibration_sha256'):
            sha256(worker.get(field), field)
        require(worker['checkpoint_sha256'] not in hashes, 'duplicate checkpoint across workers')
        hashes.add(worker['checkpoint_sha256'])
        positive_number(worker.get('initial_prepare_ms'), 'initial preparation')
        require(worker.get('accepted_decisions_and_existing_checks_agree') is True,
                'worker did not report complete check agreement')
        rows = worker.get('rows')
        require(isinstance(rows, list) and len(rows) == len(grid), 'wrong worker row count')
        rng, medians, pools = random.Random(order_seed + seed), {}, {}
        for row, expected in zip(rows, grid):
            require(isinstance(row, dict), 'invalid condition row')
            integer(row.get('record_count'), 'row record count')
            key = (row.get('model'), row.get('retention'), row['record_count'])
            require(key == expected, 'worker model/retention/count grid or row order mismatch')
            pool = integer(row.get('distinct_saved_records'), 'saved-record pool size')
            require(pools.setdefault(key[:2], pool) == pool, 'source pool changes between batch sizes')
            if 'source_record_pool_size' in row:
                require(type(row['source_record_pool_size']) is int and
                        row['source_record_pool_size'] == pool, 'source-pool fields disagree')
            if 'distinct_records_in_batch' in row:
                distinct = integer(row['distinct_records_in_batch'], 'distinct records in batch')
                require(distinct <= min(pool, key[2]), 'distinct records exceed batch or pool')
            orders = row.get('mode_order_each_repeat')
            require(isinstance(orders, list) and len(orders) == repeats,
                    'mode order repeat count mismatch')
            order = list(MODES)
            for recorded in orders:
                rng.shuffle(order)
                require(recorded == order, 'mode order differs from planned random sequence')
            timings = row.get('timings')
            require(isinstance(timings, dict) and set(timings) == set(MODES),
                    'missing or extra timing modes')
            medians[key] = {}
            for mode in MODES:
                timing = timings[mode]
                require(isinstance(timing, dict), 'invalid timing object')
                values = timing.get('samples_ms')
                require(isinstance(values, list) and len(values) == repeats,
                        'timing repeat count mismatch')
                for value in values:
                    positive_number(value, 'timing sample')
                median = statistics.median(values)
                for field, q in (('median_ms', .5), ('p10_ms', .1), ('p90_ms', .9)):
                    supplied = positive_number(timing.get(field), field)
                    require(math.isclose(supplied, quantile(values, q), rel_tol=1e-12,
                                         abs_tol=1e-9), f'incorrect {field}')
                    statistic_checks += 1
                medians[key][mode] = median
                all_samples.extend((value / median, seed, *key, mode, value, median)
                                   for value in values)
        workers.append(worker)
        worker_medians[seed] = medians
    require(workers, 'missing worker reports')
    workers.sort(key=lambda worker: worker['training_seed'])
    require(summary.get('format') == FORMAT + '.complete', 'unsupported summary format')
    require(type(summary.get('checkpoint_runs')) is int and
            summary['checkpoint_runs'] == len(workers), 'summary checkpoint count mismatch')
    if 'repeats' in summary:
        require(type(summary['repeats']) is int and summary['repeats'] == repeats,
                'summary/plan repeat count mismatch')
    require(summary.get('all_checks_agreed') is True, 'summary did not report agreement')
    require(summary.get('scope') == plan['scope'], 'summary/plan scope mismatch')
    embedded = summary.get('results')
    require(isinstance(embedded, list) and len(embedded) == len(workers) and
            all(isinstance(worker, dict) and type(worker.get('training_seed')) is int
                for worker in embedded), 'invalid summary workers')
    require(sorted(embedded, key=lambda worker: worker['training_seed']) == workers,
            'summary differs from independent worker reports')

    result_rows = []
    for model, retention, count in grid:
        values = {mode: [worker_medians[w['training_seed']][model, retention, count][mode]
                         for w in workers] for mode in MODES}
        result = {'model': model, 'retention': retention, 'record_count': count}
        for mode in MODES:
            result[mode + '_ms'] = statistics.median(values[mode])
            result[mode + '_ms_min'] = min(values[mode])
            result[mode + '_ms_max'] = max(values[mode])
        for mode in MODES[1:]:
            pairs = list(zip(values[MODES[0]], values[mode]))
            reductions = [100 * (1 - new / old) for old, new in pairs]
            result[mode + '_paired_speedup'] = statistics.median(old / new for old, new in pairs)
            result[mode + '_paired_time_reduction_pct'] = statistics.median(reductions)
            result[mode + '_paired_time_reduction_pct_range'] = [min(reductions), max(reductions)]
        result_rows.append(result)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b''):
            digest.update(chunk)
    warnings = []
    if repeats != 3 and 'Three repeats by default' in str(summary.get('warning', '')):
        warnings.append(f'Summary warning mentions three repeats; PLAN and samples use {repeats}.')
    return {
        'format': 'universa-recurrent.verifier-setup-audit.v1',
        'archive_sha256': digest.hexdigest(), 'worker_count': len(workers),
        'condition_rows': len(workers) * len(grid), 'timing_samples': len(all_samples),
        'statistic_checks': statistic_checks, 'consistency_errors': [],
        'report_consistency_passed': True, 'all_reported_agreement': True,
        'summary_agreement_consistent': True,
        'training_seeds': [w['training_seed'] for w in workers],
        'record_counts': counts, 'repeats': repeats,
        'initial_prepare_ms': {str(w['training_seed']): w['initial_prepare_ms'] for w in workers},
        'initial_prepare_ms_median': statistics.median(w['initial_prepare_ms'] for w in workers),
        'rows': result_rows,
        'largest_sample_to_row_median': sorted(all_samples, reverse=True)[:10],
        'number_samples_gt_2x_row_median': sum(row[0] > 2 for row in all_samples),
        'number_samples_gt_10x_row_median': sum(row[0] > 10 for row in all_samples),
        'reported_identity': {key: plan[key] for key in ('source', 'script_sha256',
            'input_zip_sha256', 'no_training', 'no_gpu_inference', 'new_independent_record_samples')},
        'warnings': warnings,
        'scope': 'Report consistency only. Hashes and execution flags are reported identity, '
                 'not proof of executed code. No receipts, checkpoints, calibration files, '
                 'model outputs, or execution authenticity were independently verified. '
                 'Durations are medians of checkpoint medians; speedups and reductions are '
                 'medians of within-checkpoint ratios. Repeated records measure setup '
                 'amortization, not new validation coverage or statistical significance.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(args.archive)
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.output:
            with args.output.open('x', encoding='utf-8') as handle:
                handle.write(text)
        print(text, end='')
    except (ValueError, OSError, BadZipFile, RuntimeError, KeyError, TypeError) as error:
        parser.exit(1, f'audit failed: {error}\n')


if __name__ == '__main__':
    main()
