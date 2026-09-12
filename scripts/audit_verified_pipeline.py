"""Recompute a complete-request timing report without running the producer.

This is a laboratory-notebook audit: raw durations, stage accounting, summaries,
saved receipt bytes, and input identities are checked independently. Receipt
geometry and GPU execution are outside its scope. All hashes and execution flags
are reported identities, not authentication. ZIP contents are never extracted.

Usage: python scripts/audit_verified_pipeline.py reports.zip --output audit.json
Add --source-root /path/to/repository to compare reported source hashes with files.
Only the Python standard library is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics as stats
from zipfile import BadZipFile, ZipFile


FORMAT = 'universa-recurrent.verified-pipeline.v1'
MODELS = ('shared', 'fixed_depth_4', 'direct')
MODES = ('files_per_record', 'prepare_once_per_batch', 'reuse_prepared_snapshot')
STAGES = ('reference_prepare', 'cpu_to_device', 'inference',
          'device_to_host_records_json', 'parse_bound_verify')
SCRIPTS = ('benchmark_verified_pipeline.py', 'benchmark_verifier_setup.py')
OUTPUT_FLOATS = ('candidate_states', 'claim_state', 'estimate', 'probabilities')
OUTPUT_DISCRETE = ('claim_index', 'claim_mask', 'top_route')
RECURRENT_FLOATS = ('coordinates', 'prior_logits', 'progress', 'residuals',
                    'route_logits', 'route_probabilities', 'states')
DIRECT_FLOATS = ('coordinates', 'mixture_state', 'residuals', 'route_logits',
                 'route_probabilities', 'states')
MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES = 100_000_000
MAX_MEMBER_BYTES = 20_000_000
MAX_MEMBERS = 1000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=1):
    require(type(value) is int and value >= minimum, f'invalid {label}')
    return value


def number(value, label, allow_zero=False):
    require(type(value) in (int, float), f'invalid numeric {label}')
    try:
        valid = math.isfinite(value) and (value >= 0 if allow_zero else value > 0)
    except OverflowError:
        valid = False
    require(valid, f'invalid numeric {label}')
    return value


def sha(value, label):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None,
            f'invalid {label} hash')


def same(actual, expected, label):
    """Compare report structure exactly, allowing only rounding of floats."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), f'{label}: fields differ')
        for key, value in expected.items():
            same(actual[key], value, f'{label}.{key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), f'{label}: list differs')
        for index, (left, right) in enumerate(zip(actual, expected)):
            same(left, right, f'{label}[{index}]')
    elif type(expected) is float:
        require(type(actual) in (int, float) and math.isclose(actual, expected,
                rel_tol=1e-12, abs_tol=1e-9), f'{label}: numeric mismatch')
    else:
        require(type(actual) is type(expected) and actual == expected, f'{label}: value mismatch')


def read_json(data):
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


def read_archive(path):
    require(path.stat().st_size <= MAX_ARCHIVE_BYTES, 'report ZIP too large')
    with ZipFile(path) as archive:
        infos = archive.infolist()
        require(len(infos) <= MAX_MEMBERS, 'too many ZIP members')
        require(len({info.filename for info in infos}) == len(infos), 'duplicate ZIP member')
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


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1_048_576), b''):
            result.update(block)
    return result.hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def batch_digest(payloads):
    result = hashlib.sha256()
    for payload in payloads:
        result.update(len(payload).to_bytes(8, 'big'))
        result.update(payload)
    return result.hexdigest()


def input_identity(records):
    payloads = [encoded({'observed': record['endpoint']['observed'],
                         'mask': record['endpoint']['mask']}) for record in records]
    unique = sorted(set(payloads))
    return {'count': len(payloads), 'distinct_observed_mask_count': len(unique),
            'ordered_observed_mask_sha256': batch_digest(payloads),
            'unique_observed_mask_sha256': batch_digest(unique)}


def timing_stats(values):
    ordered = sorted(values)
    def quantile(q):
        index = (len(ordered) - 1) * q
        low, high = math.floor(index), math.ceil(index)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {'median_ms': stats.median(values), 'p10_ms': quantile(.1),
            'p90_ms': quantile(.9), 'samples_ms': values}


def retentions(model):
    return ('endpoint',) if model == 'direct' else ('endpoint', 'trajectory')


def check_source(plan, source_root):
    source, scripts = plan.get('source'), plan.get('scripts_sha256')
    require(isinstance(source, dict), 'missing source identity')
    sha(source.get('python_source_sha256'), 'package source')
    for key in ('python', 'numpy', 'torch', 'package_version'):
        require(isinstance(source.get(key), str) and source[key], f'missing source {key}')
    for key in ('git_commit', 'portable_source_commit'):
        value = source.get(key)
        require(value is None or (isinstance(value, str) and re.fullmatch(r'[0-9a-f]{40}', value)),
                f'invalid reported {key}')
    require(isinstance(scripts, dict) and set(scripts) == set(SCRIPTS), 'script identity fields differ')
    for key, value in scripts.items():
        sha(value, key)
    if source_root is None:
        return {'local_source_files_compared': False}
    actual_scripts = {name: digest(source_root / 'scripts' / name) for name in SCRIPTS}
    same(actual_scripts, scripts, 'local benchmark source')
    root = source_root / 'src' / 'universa_recurrent'
    paths = sorted(root.rglob('*.py'))
    require(paths, 'missing local package source')
    package = hashlib.sha256()
    for path in paths:
        package.update(path.relative_to(root).as_posix().encode() + b'\0')
        package.update(path.read_bytes())
    same(package.hexdigest(), source['python_source_sha256'], 'local package source')
    return {'local_source_files_compared': True, 'benchmark_scripts_match': True,
            'package_python_files_match': True,
            'scope': 'Local bytes match reported hashes; this does not prove those bytes executed.'}


def check_equivalence(report, model):
    groups = {'outputs': (OUTPUT_FLOATS, OUTPUT_DISCRETE),
              'candidates': (DIRECT_FLOATS if model == 'direct' else RECURRENT_FLOATS, ())}
    if model == 'direct':
        groups['normal_predict'] = (OUTPUT_FLOATS, OUTPUT_DISCRETE)
    require(isinstance(report, dict) and set(report) == set(groups), 'equivalence groups differ')
    differences, exact_count, equal_count = [], 0, 0
    for name, (floating, discrete) in groups.items():
        values = report[name]
        require(isinstance(values, dict) and set(values) == set(floating) | set(discrete),
                'equivalence tensor fields differ')
        for field in floating:
            entry = values[field]
            require(isinstance(entry, dict) and set(entry) == {'max_absolute_difference', 'bitwise_equal'},
                    'invalid floating equivalence entry')
            difference = number(entry['max_absolute_difference'], 'output difference', True)
            require(type(entry['bitwise_equal']) is bool and entry['bitwise_equal'] == (difference == 0),
                    'output equality flag and difference disagree')
            differences.append(difference)
            equal_count += entry['bitwise_equal']
        for field in discrete:
            same(values[field], {'exact_match': True}, 'discrete output comparison')
            exact_count += 1
    return differences, exact_count, equal_count


def audit(path: Path, source_root: Path | None = None):
    reports = read_archive(path)
    plan, summary = reports['results/PLAN.json'], reports['results/summary.json']
    same(plan.get('format'), FORMAT + '.plan', 'plan format')
    counts, models, jobs = plan.get('record_counts'), plan.get('models'), plan.get('job_order')
    for values, label in ((counts, 'record counts'), (models, 'models'), (jobs, 'job order')):
        require(isinstance(values, list) and values, f'missing planned {label}')
    for count in counts:
        integer(count, 'record count')
    for seed in jobs:
        integer(seed, 'training seed', 0)
    require(all(isinstance(model, str) and model in MODELS for model in models), 'unknown model')
    require(all(len(set(values)) == len(values) for values in (counts, models, jobs)),
            'duplicate planned count, model, or seed')
    repeats = integer(plan.get('repeats'), 'repeats')
    warmup = integer(plan.get('warmup'), 'warmup', 0)
    order_seed = integer(plan.get('order_seed'), 'order seed', 0)
    integer(plan.get('cpu_threads'), 'CPU threads')
    test_seed = integer(plan.get('test_seed'), 'test seed', 0)
    require(plan.get('device') in ('cpu', 'cuda'), 'unsupported device')
    for key in ('fresh_process_per_checkpoint', 'no_training_or_recalibration'):
        same(plan.get(key), True, key)
    require(isinstance(plan.get('scope'), dict) and plan['scope'], 'missing scope')
    source_checks = check_source(plan, source_root)
    # The producer sorts weights filenames lexicographically before shuffling.
    expected_jobs = sorted(jobs, key=lambda seed: f'weights-{seed}.pt')
    random.Random(order_seed).shuffle(expected_jobs)
    same(jobs, expected_jobs, 'planned checkpoint order')
    expected_names = {'results/PLAN.json', 'results/summary.json'} | {
        f'results/seed-{seed}.json' for seed in jobs}
    require(set(reports) == expected_names, 'planned worker reports differ from ZIP members')
    workers = [reports[f'results/seed-{seed}.json'] for seed in jobs]
    grid = [(model, retention, count) for model in models for count in counts
            for retention in retentions(model)]
    saved_grid = [(model, retention, max(counts)) for model in models for retention in retentions(model)]
    groups, cohorts, checkpoints, source_inputs = {}, [], set(), {}
    outliers, differences, equivalences = [], [], {}
    discrete_checks = exact_floats = sample_count = statistic_checks = saved_count = verified_count = 0
    receipt_equal_rows = canonical_equal_rows = 0
    minimum_overhead = math.inf
    for seed, worker in zip(jobs, workers):
        same(worker.get('format'), FORMAT + '.worker', 'worker format')
        same(worker.get('training_seed'), seed, 'worker seed')
        for key in ('source', 'scripts_sha256', 'scope', 'record_counts', 'repeats', 'warmup', 'test_seed'):
            same(worker.get(key), plan[key], f'worker {key}')
        for key in ('all_equivalence_checks_passed', 'all_saved_receipts_audited', 'no_training_or_recalibration'):
            same(worker.get(key), True, f'worker {key}')
        for key in ('checkpoint_sha256', 'calibration_sha256', 'dataset_sha256'):
            sha(worker.get(key), key)
        require(worker['checkpoint_sha256'] not in checkpoints, 'duplicate checkpoint bytes')
        checkpoints.add(worker['checkpoint_sha256'])
        reserved = worker.get('reserved_data_seeds')
        require(isinstance(reserved, list), 'missing reserved data seeds')
        for reserved_seed in reserved:
            integer(reserved_seed, 'reserved seed', 0)
        require(len(set(reserved)) == len(reserved) and test_seed not in reserved,
                'duplicate reserved seed or test/training/calibration overlap')
        require({22000, 32000, 33000}.issubset(reserved), 'prior test seeds absent from reserved list')
        for key in ('model_load_ms', 'initial_prepare_ms'):
            number(worker.get(key), key)
        same(worker.get('atol'), 1e-7, 'reported output atol')
        same(worker.get('rtol'), 1e-6, 'reported output rtol')
        execution, environment = worker.get('execution'), worker.get('environment')
        require(isinstance(execution, dict) and isinstance(environment, dict), 'missing execution metadata')
        same(execution.get('num_threads'), plan['cpu_threads'], 'CPU thread setting')
        integer(execution.get('num_interop_threads'), 'interop threads')
        for key, value in {'cublas_workspace_config': ':4096:8', 'cudnn_allow_tf32': False,
                'cudnn_benchmark': False, 'cudnn_deterministic': True, 'deterministic_algorithms': True,
                'deterministic_warn_only': False, 'float32_matmul_precision': 'highest',
                'matmul_allow_tf32': False}.items():
            same(execution.get(key), value, f'execution {key}')
        same(environment.get('device'), plan['device'], 'worker device')
        if workers[0] is not worker:
            same(execution, workers[0]['execution'], 'execution settings across workers')
            same(environment, workers[0]['environment'], 'environment across workers')
        saved = worker.get('saved_receipts')
        require(isinstance(saved, list) and len(saved) == len(saved_grid), 'saved receipt groups differ')
        saved_metadata = {}
        for group, key in zip(saved, saved_grid):
            require(isinstance(group, dict), 'invalid saved group')
            same([group.get('model'), group.get('retention'), group.get('record_count')], list(key),
                 'saved model/retention/count order')
            records = group.get('records')
            require(isinstance(records, list) and len(records) == key[2], 'saved receipt count mismatch')
            for record in records:
                require(isinstance(record, dict) and isinstance(record.get('endpoint'), dict),
                        'missing receipt endpoint')
                endpoint = record['endpoint']
                for field in ('observed', 'mask'):
                    require(isinstance(endpoint.get(field), list) and endpoint[field], 'missing receipt input')
                same(record.get('retention'), key[1], 'saved receipt retention')
                same(endpoint.get('model'), key[0], 'saved receipt model')
                for field in ('checkpoint_sha256', 'calibration_sha256'):
                    same(endpoint.get(field), worker[field], f'saved receipt {field}')
            identity = input_identity(records)
            same(identity['distinct_observed_mask_count'], key[2], 'unique saved inputs')
            same(worker.get('inputs'), identity, 'worker input identity')
            for count in counts:
                prefix = input_identity(records[:count])
                same(prefix, source_inputs.setdefault(count, prefix), 'common nested input cohort')
            payloads = [encoded(record) for record in records]
            receipt_hash = batch_digest(payloads)
            same(group.get('batch_receipts_sha256'), receipt_hash, 'saved receipt bytes hash')
            same(group.get('audit'), {'receipt_count': key[2], 'all_accepted': True,
                'outer_checks_and_retained_steps_agree': True,
                'all_records_bound_to_checkpoint_and_calibration': True}, 'saved receipt audit flags')
            saved_metadata[key] = (receipt_hash, sum(map(len, payloads)))
            saved_count += len(records)
        cohorts.append((worker['dataset_sha256'], worker['inputs']))
        same(cohorts[-1], cohorts[0], 'cohort across checkpoints')
        rows = worker.get('rows')
        require(isinstance(rows, list) and len(rows) == len(grid), 'condition grid size differs')
        rng = random.Random(order_seed + seed)
        for row, key in zip(rows, grid):
            require(isinstance(row, dict), 'invalid condition row')
            same([row.get('model'), row.get('retention'), row.get('record_count')], list(key),
                 'condition grid or row order differs')
            same(row.get('inputs'), source_inputs[key[2]], 'row input identity')
            for flag in ('all_timed_outputs_compared', 'all_timed_receipts_bound_checked'):
                same(row.get(flag), True, flag)
            eq = row.get('equivalence')
            floating, discrete, equal = check_equivalence(eq, key[0])
            differences.extend(floating)
            discrete_checks += discrete
            exact_floats += equal
            eq_key = (seed, key[0], key[2])
            same(eq, equivalences.setdefault(eq_key, eq), 'endpoint/trajectory equivalence reports')
            orders = []
            for _ in range(repeats):
                order = list(MODES)
                rng.shuffle(order)
                orders.append(order)
            same(row.get('mode_order_each_repeat'), orders, 'mode order and plan seed')
            samples, timings = row.get('samples'), row.get('timings')
            require(isinstance(samples, dict) and set(samples) == set(MODES) and
                    isinstance(timings, dict) and set(timings) == set(MODES), 'timing modes differ')
            receipt_hashes, byte_sizes, medians = set(), {}, {}
            for mode in MODES:
                values = samples[mode]
                require(isinstance(values, list) and len(values) == repeats, 'sample repeat count differs')
                totals = []
                for sample in values:
                    require(isinstance(sample, dict), 'invalid raw sample')
                    total = number(sample.get('total_ms'), 'total time')
                    totals.append(total)
                    same(sample.get('verified_count'), key[2], 'sample verified receipt count')
                    size = integer(sample.get('serialized_bytes'), 'serialized byte count')
                    receipt_hash = sample.get('batch_receipts_sha256')
                    sha(receipt_hash, 'timed receipt batch')
                    receipt_hashes.add(receipt_hash)
                    same(size, byte_sizes.setdefault(receipt_hash, size), 'same receipt hash/byte size')
                    stages = sample.get('stages_ms')
                    require(isinstance(stages, dict) and set(stages) == set(STAGES), 'stage schema differs')
                    for stage in STAGES:
                        number(stages[stage], stage, allow_zero=stage == 'reference_prepare')
                    if mode == 'prepare_once_per_batch':
                        number(stages['reference_prepare'], 'included reference preparation')
                    else:
                        same(float(stages['reference_prepare']), 0., 'excluded reference preparation')
                    overhead = total - sum(stages.values())
                    require(overhead >= -1e-9, 'stage durations exceed directly measured total')
                    minimum_overhead = min(minimum_overhead, overhead)
                    verified_count += key[2]
                    sample_count += 1
                supplied = timings[mode]
                require(isinstance(supplied, dict), 'invalid timing summary')
                same(supplied, timing_stats(totals), 'timing samples and statistics')
                statistic_checks += 3
                medians[mode] = stats.median(totals)
                outliers.extend({'ratio_to_condition_median': value / medians[mode],
                    'training_seed': seed, 'model': key[0], 'retention': key[1],
                    'record_count': key[2], 'mode': mode, 'sample_index': index, 'total_ms': value}
                    for index, value in enumerate(totals))
            same(row.get('unique_timed_receipt_hash_count'), len(receipt_hashes), 'unique timed hash count')
            same(row.get('bitwise_receipts_equal_across_timed_runs'), len(receipt_hashes) == 1,
                 'timed receipt equality flag')
            receipt_equal_rows += len(receipt_hashes) == 1
            canonical = row.get('canonical_batch_receipts_sha256')
            sha(canonical, 'canonical receipt batch')
            canonical_equal_rows += receipt_hashes == {canonical}
            if key in saved_metadata:
                saved_hash, saved_size = saved_metadata[key]
                same(canonical, saved_hash, 'row/saved canonical receipt hash')
                if saved_hash in byte_sizes:
                    same(byte_sizes[saved_hash], saved_size, 'saved/timed serialized byte count')
            groups.setdefault(key, []).append({'training_seed': seed, 'median_ms': medians,
                'files_divided_by_mode_time': {mode: medians[MODES[0]] / medians[mode] for mode in MODES},
                'samples': samples})

    expected_summary = []
    result_rows = []
    for key, pairs in sorted(groups.items()):
        medians = {mode: stats.median(pair['median_ms'][mode] for pair in pairs) for mode in MODES}
        ratios = {mode: stats.median(pair['files_divided_by_mode_time'][mode] for pair in pairs)
                  for mode in MODES}
        expected_summary.append({'model': key[0], 'retention': key[1], 'record_count': key[2],
            'checkpoint_pairs': len(pairs), 'per_seed': [{k: v for k, v in pair.items() if k != 'samples'}
                                                      for pair in pairs],
            'median_checkpoint_time_ms': medians, 'median_paired_speed_ratio': ratios})
        reductions = {mode: [100 * (1 - pair['median_ms'][mode] / pair['median_ms'][MODES[0]])
                            for pair in pairs] for mode in MODES}
        stages = {}
        for mode in MODES:
            mode_stages = {}
            for stage in (*STAGES, 'unattributed_overhead', 'records_and_verification'):
                def duration(sample):
                    if stage == 'unattributed_overhead':
                        return sample['total_ms'] - sum(sample['stages_ms'].values())
                    if stage == 'records_and_verification':
                        return sample['stages_ms']['device_to_host_records_json'] + sample['stages_ms']['parse_bound_verify']
                    return sample['stages_ms'][stage]
                per_seed = [{'training_seed': pair['training_seed'],
                    'median_ms': stats.median(duration(sample) for sample in pair['samples'][mode]),
                    'median_sample_share_pct': stats.median(100 * duration(sample) / sample['total_ms']
                                                            for sample in pair['samples'][mode])}
                            for pair in pairs]
                mode_stages[stage] = {'median_checkpoint_stage_ms': stats.median(p['median_ms'] for p in per_seed),
                    'median_checkpoint_sample_share_pct': stats.median(p['median_sample_share_pct'] for p in per_seed),
                    'mean_sample_share_pct': stats.mean(100 * duration(sample) / sample['total_ms']
                        for pair in pairs for sample in pair['samples'][mode]), 'per_seed': per_seed}
            stages[mode] = mode_stages
        result_rows.append({'model': key[0], 'retention': key[1], 'record_count': key[2],
            'median_checkpoint_time_ms': medians, 'median_paired_speed_ratio': ratios,
            'median_paired_time_reduction_pct': {mode: stats.median(values) for mode, values in reductions.items()},
            'paired_time_reduction_pct_range': {mode: [min(values), max(values)] for mode, values in reductions.items()},
            'matched_repeat_faster_count': {mode: sum(new['total_ms'] < old['total_ms']
                for pair in pairs for old, new in zip(pair['samples'][MODES[0]], pair['samples'][mode]))
                for mode in MODES}, 'matched_repeat_pairs': len(pairs) * repeats,
            'per_seed': expected_summary[-1]['per_seed'], 'stages': stages})
    for key, expected in {'format': FORMAT + '.complete', 'source': plan['source'],
            'scripts_sha256': plan['scripts_sha256'], 'worker_reports': [f'seed-{seed}.json' for seed in jobs],
            'checkpoint_runs': len(workers), 'dataset_sha256': cohorts[0][0], 'inputs': cohorts[0][1],
            'rows': expected_summary, 'all_equivalence_checks_passed': True,
            'all_saved_receipts_audited': True, 'saved_receipt_count': saved_count, 'scope': plan['scope']}.items():
        same(summary.get(key), expected, f'summary {key}')
    return {'format': 'universa-recurrent.verified-pipeline-report-audit.v1',
        'archive_sha256': digest(path), 'report_consistency_passed': True,
        'worker_count': len(workers), 'condition_rows': len(workers) * len(grid),
        'mode_condition_rows': len(workers) * len(grid) * len(MODES),
        'timing_samples': sample_count, 'statistic_checks': statistic_checks,
        'raw_total_to_timing_array_checks': sample_count,
        'summary_numeric_timing_checks': {
            'per_seed_medians': len(workers) * len(grid) * len(MODES),
            'per_seed_speed_ratios': len(workers) * len(grid) * len(MODES),
            'aggregate_medians': len(grid) * len(MODES),
            'aggregate_speed_ratios': len(grid) * len(MODES)},
        'stage_sum_checks': sample_count, 'minimum_total_minus_stage_sum_ms': minimum_overhead,
        'timed_receipt_verifications_reported': verified_count, 'saved_receipt_count': saved_count,
        'saved_receipt_byte_hashes_recomputed': len(workers) * len(saved_grid),
        'unique_observed_mask_inputs_recomputed': max(counts),
        'training_seeds_in_job_order': jobs, 'test_seed': test_seed, 'record_counts': counts,
        'repeats': repeats, 'warmup': warmup, 'inputs': cohorts[0][1],
        'reported_numeric_equivalence': {'floating_entries': len(differences),
            'zero_difference_equal_entries': exact_floats, 'exact_discrete_entries': discrete_checks,
            'max_absolute_difference': max(differences),
            'unique_model_count_checkpoint_comparisons': len(equivalences),
            'all_rows_report_bitwise_receipt_equality': receipt_equal_rows == len(workers) * len(grid),
            'timed_hashes_equal_canonical_rows': canonical_equal_rows,
            'scope': 'Tensor differences and equality are reported measurements; full prediction tensors are absent. '
                     'Endpoint/trajectory rows repeat the same model/count equivalence comparison.'},
        'initial_prepare_ms': {str(w['training_seed']): w['initial_prepare_ms'] for w in workers},
        'initial_prepare_ms_median': stats.median(w['initial_prepare_ms'] for w in workers),
        'model_load_ms': {str(w['training_seed']): w['model_load_ms'] for w in workers},
        'model_load_ms_median': stats.median(w['model_load_ms'] for w in workers),
        'reported_source': plan['source'], 'reported_scripts_sha256': plan['scripts_sha256'],
        'source_checks': source_checks, 'reported_environment': workers[0]['environment'],
        'rows': result_rows,
        'number_samples_gt_2x_condition_median': sum(row['ratio_to_condition_median'] > 2 for row in outliers),
        'number_samples_gt_10x_condition_median': sum(row['ratio_to_condition_median'] > 10 for row in outliers),
        'largest_sample_to_condition_median': sorted(outliers, key=lambda row: row['ratio_to_condition_median'], reverse=True)[:10],
        'scope': 'Report/timing/serialized-identity accounting only; receipt arithmetic and geometry are not checked here. '
                 'No weights or full prediction tensors are available for GPU replay or checkpoint-bound verification. '
                 'Reported hashes and flags do not authenticate execution. Total latency is measured directly, never '
                 'a sum of stage medians. Stage shares are calculated for each individual sample before aggregation; '
                 'medians of different stages need not add to 100%. The records_and_verification field combines '
                 'two stages within each sample and must not be counted again in a stage total. '
                 'Counts are nested prefixes of one shared cohort; repeated requests/checkpoints are not new inputs. '
                 'Already-prepared mode excludes initial preparation; all request times exclude model loading, '
                 'dataset generation, report writing, and process startup. Timings are descriptive, not significance tests.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-root', type=Path)
    args = parser.parse_args(argv)
    try:
        report = audit(args.archive, args.source_root)
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.output:
            with args.output.open('x', encoding='utf-8') as handle:
                handle.write(text)
        print(text, end='')
    except (ValueError, OSError, BadZipFile, RuntimeError, KeyError, TypeError, OverflowError) as error:
        parser.exit(1, f'audit failed: {error}\n')


if __name__ == '__main__':
    main()
