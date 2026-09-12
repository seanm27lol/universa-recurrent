"""Synthetic complete-request report accounting; no tensor libraries required."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
from zipfile import ZipFile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_verified_pipeline.py'
SPEC = importlib.util.spec_from_file_location('verified_pipeline_audit', SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture_reports():
    jobs = [10, 20]
    random.Random(37).shuffle(jobs)
    plan = {'format': AUDIT.FORMAT + '.plan', 'record_counts': [1, 2],
        'models': ['shared', 'direct'], 'repeats': 3, 'warmup': 1, 'order_seed': 37,
        'job_order': jobs, 'test_seed': 36000, 'device': 'cpu', 'cpu_threads': 1,
        'scope': {'included': 'complete synthetic request'}, 'fresh_process_per_checkpoint': True,
        'no_training_or_recalibration': True, 'scripts_sha256': {name: 'a' * 64 for name in AUDIT.SCRIPTS},
        'source': {'python_source_sha256': 'b' * 64, 'git_commit': None,
            'portable_source_commit': 'c' * 40, 'python': '3.11', 'numpy': '2.0',
            'torch': '2.10', 'package_version': '0.5.2'}}
    workers = []
    for seed in jobs:
        worker = {key: copy.deepcopy(plan[key]) for key in ('source', 'scripts_sha256', 'scope',
                  'record_counts', 'repeats', 'warmup', 'test_seed')}
        worker.update({'format': AUDIT.FORMAT + '.worker', 'training_seed': seed,
            'checkpoint_sha256': f'{seed:064x}', 'calibration_sha256': f'{seed + 1:064x}',
            'dataset_sha256': 'd' * 64, 'reserved_data_seeds': [22000, 32000, 33000, seed + 1],
            'model_load_ms': 100., 'initial_prepare_ms': 10., 'atol': 1e-7, 'rtol': 1e-6,
            'execution': {'num_threads': 1, 'num_interop_threads': 2,
                'cublas_workspace_config': ':4096:8', 'cudnn_allow_tf32': False,
                'cudnn_benchmark': False, 'cudnn_deterministic': True,
                'deterministic_algorithms': True, 'deterministic_warn_only': False,
                'float32_matmul_precision': 'highest', 'matmul_allow_tf32': False},
            'environment': {'device': 'cpu', 'device_name': 'cpu', 'cuda_runtime': None},
            'all_equivalence_checks_passed': True, 'all_saved_receipts_audited': True,
            'no_training_or_recalibration': True, 'rows': [], 'saved_receipts': []})
        rng = random.Random(37 + seed)
        for model in plan['models']:
            all_records = {}
            for retention in AUDIT.retentions(model):
                records = [{'retention': retention, 'endpoint': {'model': model,
                    'observed': [float(index), 1.], 'mask': [1., 0.],
                    'checkpoint_sha256': worker['checkpoint_sha256'],
                    'calibration_sha256': worker['calibration_sha256']}} for index in range(2)]
                all_records[retention] = records
                worker['inputs'] = AUDIT.input_identity(records)
                worker['saved_receipts'].append({'model': model, 'retention': retention,
                    'record_count': 2, 'records': records,
                    'batch_receipts_sha256': AUDIT.batch_digest([AUDIT.encoded(r) for r in records]),
                    'audit': {'receipt_count': 2, 'all_accepted': True,
                        'outer_checks_and_retained_steps_agree': True,
                        'all_records_bound_to_checkpoint_and_calibration': True}})
            eq = {'outputs': {key: {'max_absolute_difference': 0., 'bitwise_equal': True}
                               for key in AUDIT.OUTPUT_FLOATS},
                  'candidates': {key: {'max_absolute_difference': 0., 'bitwise_equal': True}
                     for key in (AUDIT.DIRECT_FLOATS if model == 'direct' else AUDIT.RECURRENT_FLOATS)}}
            eq['outputs'].update({key: {'exact_match': True} for key in AUDIT.OUTPUT_DISCRETE})
            if model == 'direct':
                eq['normal_predict'] = copy.deepcopy(eq['outputs'])
            for count in plan['record_counts']:
                for retention in AUDIT.retentions(model):
                    records = all_records[retention][:count]
                    payloads = [AUDIT.encoded(r) for r in records]
                    receipt_hash = AUDIT.batch_digest(payloads)
                    orders = []
                    for _ in range(3):
                        order = list(AUDIT.MODES)
                        rng.shuffle(order)
                        orders.append(order)
                    row = {'model': model, 'retention': retention, 'record_count': count,
                        'inputs': AUDIT.input_identity(records), 'equivalence': copy.deepcopy(eq),
                        'all_timed_outputs_compared': True, 'all_timed_receipts_bound_checked': True,
                        'canonical_batch_receipts_sha256': receipt_hash,
                        'unique_timed_receipt_hash_count': 1,
                        'bitwise_receipts_equal_across_timed_runs': True,
                        'mode_order_each_repeat': orders, 'samples': {}, 'timings': {}}
                    for mode in AUDIT.MODES:
                        samples = []
                        for index, inference in enumerate((10., 10., 100.)):
                            stages = {'reference_prepare': 10. if mode == 'prepare_once_per_batch' else 0.,
                                'cpu_to_device': 1., 'inference': inference,
                                'device_to_host_records_json': 2. * count,
                                'parse_bound_verify': float(count * (20 if mode == 'files_per_record' else 3) * (index + 1))}
                            samples.append({'total_ms': sum(stages.values()) + .5,
                                'stages_ms': stages, 'verified_count': count,
                                'serialized_bytes': sum(map(len, payloads)), 'batch_receipts_sha256': receipt_hash})
                        totals = [s['total_ms'] for s in samples]
                        ordered = sorted(totals)
                        row['samples'][mode] = samples
                        row['timings'][mode] = {'samples_ms': totals, 'median_ms': ordered[1],
                            'p10_ms': ordered[0] + .2 * (ordered[1] - ordered[0]),
                            'p90_ms': ordered[1] + .8 * (ordered[2] - ordered[1])}
                    worker['rows'].append(row)
        workers.append(worker)
    summary_rows = []
    keys = sorted((r['model'], r['retention'], r['record_count']) for r in workers[0]['rows'])
    for key in keys:
        pairs = []
        for worker in workers:
            row = next(r for r in worker['rows'] if (r['model'], r['retention'], r['record_count']) == key)
            medians = {mode: statistics.median(s['total_ms'] for s in row['samples'][mode]) for mode in AUDIT.MODES}
            pairs.append({'training_seed': worker['training_seed'], 'median_ms': medians,
                          'files_divided_by_mode_time': {mode: medians[AUDIT.MODES[0]] / medians[mode] for mode in AUDIT.MODES}})
        summary_rows.append({'model': key[0], 'retention': key[1], 'record_count': key[2],
            'checkpoint_pairs': 2, 'per_seed': pairs,
            'median_checkpoint_time_ms': {mode: statistics.median(p['median_ms'][mode] for p in pairs) for mode in AUDIT.MODES},
            'median_paired_speed_ratio': {mode: statistics.median(p['files_divided_by_mode_time'][mode] for p in pairs) for mode in AUDIT.MODES}})
    summary = {key: copy.deepcopy(plan[key]) for key in ('source', 'scripts_sha256', 'scope')}
    summary.update({'format': AUDIT.FORMAT + '.complete', 'checkpoint_runs': 2,
        'worker_reports': [f'seed-{seed}.json' for seed in jobs], 'dataset_sha256': 'd' * 64,
        'inputs': workers[0]['inputs'], 'rows': summary_rows, 'saved_receipt_count': 12,
        'all_equivalence_checks_passed': True, 'all_saved_receipts_audited': True})
    return {'results/PLAN.json': plan, 'results/summary.json': summary,
            **{f'results/seed-{w["training_seed"]}.json': w for w in workers}}


def save(tmp_path, reports):
    path = tmp_path / 'reports.zip'
    with ZipFile(path, 'w') as archive:
        for name, report in reports.items():
            archive.writestr(name, json.dumps(report))
    return path


def test_raw_timing_accounting_and_individual_sample_shares(tmp_path):
    reports = fixture_reports()
    result = AUDIT.audit(save(tmp_path, reports))
    assert result['report_consistency_passed']
    assert result['condition_rows'] == 12
    assert result['mode_condition_rows'] == 36
    assert result['timing_samples'] == result['statistic_checks'] == 108
    assert result['saved_receipt_count'] == 12
    assert result['unique_observed_mask_inputs_recomputed'] == 2
    row = next(r for r in result['rows'] if r['model'] == 'shared' and r['retention'] == 'endpoint' and r['record_count'] == 1)
    samples = reports['results/seed-10.json']['rows'][0]['samples']['reuse_prepared_snapshot']
    stage = row['stages']['reuse_prepared_snapshot']['inference']
    shares = [100 * s['stages_ms']['inference'] / s['total_ms'] for s in samples]
    assert stage['median_checkpoint_sample_share_pct'] == statistics.median(shares)
    ratio_of_medians = 100 * statistics.median(s['stages_ms']['inference'] for s in samples) / statistics.median(s['total_ms'] for s in samples)
    assert stage['median_checkpoint_sample_share_pct'] != ratio_of_medians
    assert sum(row['stages']['reuse_prepared_snapshot'][s]['mean_sample_share_pct']
               for s in (*AUDIT.STAGES, 'unattributed_overhead')) == pytest.approx(100)
    assert 'geometry are not checked' in result['scope']


@pytest.mark.parametrize('case', ['median', 'quantile', 'raw_total', 'stage_total', 'stage_schema',
    'setup_cost', 'verified_count', 'repeats', 'grid', 'mode_order', 'source', 'reserved',
    'equality_flag', 'discrete', 'equivalence_schema', 'receipt_flag', 'receipt_hash',
    'receipt_content', 'input_identity', 'byte_count', 'summary_ratio', 'summary_seed',
    'missing_worker', 'duplicate_checkpoint', 'boolean_time'])
def test_report_corruption_rejected(tmp_path, case):
    reports = fixture_reports()
    worker = reports['results/seed-10.json']
    row, summary = worker['rows'][0], reports['results/summary.json']
    sample = row['samples']['files_per_record'][0]
    if case == 'median': row['timings']['files_per_record']['median_ms'] += 1
    if case == 'quantile': row['timings']['files_per_record']['p90_ms'] += 1
    if case == 'raw_total': sample['total_ms'] += 1
    if case == 'stage_total': sample['stages_ms']['inference'] = sample['total_ms'] + 1
    if case == 'stage_schema': del sample['stages_ms']['cpu_to_device']
    if case == 'setup_cost': sample['stages_ms']['reference_prepare'] = .1
    if case == 'verified_count': sample['verified_count'] = 0
    if case == 'repeats': reports['results/PLAN.json']['repeats'] = 4
    if case == 'grid': worker['rows'].reverse()
    if case == 'mode_order': row['mode_order_each_repeat'][0].reverse()
    if case == 'source': worker['source']['python_source_sha256'] = 'f' * 64
    if case == 'reserved': worker['reserved_data_seeds'].append(36000)
    if case == 'equality_flag': row['equivalence']['outputs']['estimate']['bitwise_equal'] = False
    if case == 'discrete': row['equivalence']['outputs']['claim_index']['exact_match'] = False
    if case == 'equivalence_schema': del row['equivalence']['candidates']['states']
    if case == 'receipt_flag': row['bitwise_receipts_equal_across_timed_runs'] = False
    if case == 'receipt_hash': worker['saved_receipts'][0]['batch_receipts_sha256'] = 'e' * 64
    if case == 'receipt_content': worker['saved_receipts'][0]['records'][0]['endpoint']['extra'] = 1
    if case == 'input_identity': row['inputs']['ordered_observed_mask_sha256'] = 'f' * 64
    if case == 'byte_count': sample['serialized_bytes'] += 1
    if case == 'summary_ratio': summary['rows'][0]['median_paired_speed_ratio']['reuse_prepared_snapshot'] += 1
    if case == 'summary_seed': summary['rows'][0]['per_seed'].reverse()
    if case == 'missing_worker': del reports['results/seed-20.json']
    if case == 'duplicate_checkpoint': reports['results/seed-20.json']['checkpoint_sha256'] = worker['checkpoint_sha256']
    if case == 'boolean_time': sample['total_ms'] = True
    with pytest.raises(ValueError):
        AUDIT.audit(save(tmp_path, reports))


@pytest.mark.parametrize('data', [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'[]'])
def test_strict_json(data):
    with pytest.raises(ValueError):
        AUDIT.read_json(data)


def test_archive_boundaries_and_duplicate_members(tmp_path, monkeypatch):
    reports = fixture_reports()
    reports['../outside.json'] = {'untrusted': 'data'}
    with pytest.raises(ValueError, match='unexpected ZIP member'):
        AUDIT.audit(save(tmp_path, reports))
    path = save(tmp_path, fixture_reports())
    with ZipFile(path, 'a') as archive, pytest.warns(UserWarning):
        archive.writestr('results/PLAN.json', '{}')
    with pytest.raises(ValueError, match='duplicate ZIP member'):
        AUDIT.audit(path)
    path = save(tmp_path, fixture_reports())
    monkeypatch.setattr(AUDIT, 'MAX_MEMBER_BYTES', 100)
    with pytest.raises(ValueError, match='ZIP member too large'):
        AUDIT.audit(path)


def test_optional_local_source_hash_comparison(tmp_path):
    plan = fixture_reports()['results/PLAN.json']
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in AUDIT.SCRIPTS:
        (scripts / name).write_text('source bytes are read, never imported\n')
        plan['scripts_sha256'][name] = AUDIT.digest(scripts / name)
    package = tmp_path / 'src/universa_recurrent'
    package.mkdir(parents=True)
    (package / 'example.py').write_bytes(b'example source\n')
    plan['source']['python_source_sha256'] = hashlib.sha256(b'example.py\0example source\n').hexdigest()
    assert AUDIT.check_source(plan, tmp_path)['local_source_files_compared']
    (package / 'example.py').write_bytes(b'changed source\n')
    with pytest.raises(ValueError, match='local package source'):
        AUDIT.check_source(plan, tmp_path)


def test_standalone_cli_and_no_overwrite(tmp_path):
    archive, output = save(tmp_path, fixture_reports()), tmp_path / 'audit.json'
    command = [sys.executable, '-I', str(SCRIPT), str(archive), '--output', str(output)]
    completed = subprocess.run(command, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())['report_consistency_passed']
    assert subprocess.run(command, capture_output=True).returncode != 0
