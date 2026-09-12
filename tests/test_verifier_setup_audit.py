"""Independent report accounting and bounded parsing, without torch or numpy."""
import copy
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
from zipfile import ZipFile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/audit_verifier_setup.py'
SPEC = importlib.util.spec_from_file_location('verifier_setup_audit', SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def fixture_reports():
    plan = {'format': AUDIT.FORMAT, 'record_counts': [1, 4], 'repeats': 2,
            'order_seed': 35000, 'cpu_threads': 1, 'cpu_interop_threads': 2,
            'no_training': True, 'no_gpu_inference': True,
            'new_independent_record_samples': False, 'scope': 'CPU checks only',
            'script_sha256': 'a' * 64, 'input_zip_sha256': 'b' * 64,
            'source': {'python_source_sha256': 'c' * 64, 'git_commit': 'd' * 40,
                       'python': '3.11', 'numpy': '2.0', 'torch': '2.10',
                       'package_version': '0.5.2'}}
    workers = []
    for seed in (10, 20):
        rng, rows = random.Random(plan['order_seed'] + seed), []
        for model in AUDIT.MODELS:
            for retention in AUDIT.RETENTIONS:
                for count in plan['record_counts']:
                    order, orders = list(AUDIT.MODES), []
                    for _ in range(plan['repeats']):
                        rng.shuffle(order)
                        orders.append(list(order))
                    timings = {}
                    for mode, base in zip(AUDIT.MODES, (10 * count, 10 + count, count)):
                        samples = [float(base), float(base + 2)]
                        timings[mode] = {'samples_ms': samples, 'median_ms': base + 1.,
                                         'p10_ms': base + .2, 'p90_ms': base + 1.8}
                    rows.append({'model': model, 'retention': retention, 'record_count': count,
                                 'distinct_saved_records': 4, 'mode_order_each_repeat': orders,
                                 'timings': timings})
        workers.append({'training_seed': seed, 'checkpoint_sha256': f'{seed:064x}',
                        'calibration_sha256': f'{seed + 1:064x}', 'initial_prepare_ms': 12.,
                        'accepted_decisions_and_existing_checks_agree': True, 'rows': rows})
    summary = {'format': AUDIT.FORMAT + '.complete', 'checkpoint_runs': 2,
               'all_checks_agreed': True, 'scope': plan['scope'],
               'results': copy.deepcopy(workers), 'warning': 'Three repeats by default'}
    return {'results/PLAN.json': plan, 'results/summary.json': summary,
            **{f'results/seed-{w["training_seed"]}.json': w for w in workers}}


def save(tmp_path, reports):
    path = tmp_path / 'reports.zip'
    with ZipFile(path, 'w') as archive:
        for name, data in reports.items():
            archive.writestr(name, json.dumps(data))
    return path


def test_recomputed_values_and_explicit_limits(tmp_path):
    result = AUDIT.audit(save(tmp_path, fixture_reports()))
    assert result['report_consistency_passed']
    assert result['worker_count'] == 2
    assert result['condition_rows'] == 16
    assert result['timing_samples'] == 96
    assert result['statistic_checks'] == 144
    row = result['rows'][1]
    assert row['files_per_record_ms'] == 41
    assert row['prepare_once_per_batch_ms'] == 15
    assert row['prepare_once_per_batch_paired_speedup'] == pytest.approx(41 / 15)
    assert row['reuse_prepared_snapshot_paired_time_reduction_pct'] == pytest.approx(100 * (1 - 5 / 41))
    assert result['warnings']
    assert 'not proof of executed code' in result['scope']


@pytest.mark.parametrize('case', ['median', 'quantile', 'repeats', 'counts', 'model',
    'row_order', 'mode_order', 'mode', 'seed', 'hash', 'source', 'flag', 'sample',
    'missing_worker', 'summary', 'summary_repeats', 'agreement', 'initial', 'pool',
    'distinct', 'duplicate_count'])
def test_inconsistent_reports_rejected(tmp_path, case):
    reports = fixture_reports()
    plan, worker = reports['results/PLAN.json'], reports['results/seed-10.json']
    row = worker['rows'][0]
    timing = row['timings']['files_per_record']
    if case == 'median': timing['median_ms'] += 1
    if case == 'quantile': timing['p10_ms'] += 1
    if case == 'repeats': plan['repeats'] += 1
    if case == 'counts': plan['record_counts'].append(16)
    if case == 'model': row['model'] = 'unexpected'
    if case == 'row_order': worker['rows'].reverse()
    if case == 'mode_order': row['mode_order_each_repeat'][0].reverse()
    if case == 'mode': del row['timings']['reuse_prepared_snapshot']
    if case == 'seed': worker['training_seed'] = 11
    if case == 'hash': worker['checkpoint_sha256'] = 'bad'
    if case == 'source': plan['source']['git_commit'] = 'unknown'
    if case == 'flag': plan['no_training'] = 1
    if case == 'sample': timing['samples_ms'][0] = True
    if case == 'missing_worker': del reports['results/seed-20.json']
    if case == 'summary': reports['results/summary.json']['results'][0]['initial_prepare_ms'] = 50
    if case == 'summary_repeats': reports['results/summary.json']['repeats'] = 3
    if case == 'agreement': worker['accepted_decisions_and_existing_checks_agree'] = False
    if case == 'initial': worker['initial_prepare_ms'] = -1
    if case == 'pool': worker['rows'][1]['distinct_saved_records'] = 2
    if case == 'distinct': row['distinct_records_in_batch'] = 2
    if case == 'duplicate_count': plan['record_counts'] = [1, 1]
    with pytest.raises(ValueError):
        AUDIT.audit(save(tmp_path, reports))


@pytest.mark.parametrize('data', [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'[]'])
def test_strict_json(data):
    with pytest.raises(ValueError):
        AUDIT.read_json(data)


@pytest.mark.parametrize('name', ['../outside.json', '/tmp/outside.json', 'results/instructions.sh'])
def test_unexpected_archive_paths_rejected(tmp_path, name):
    reports = fixture_reports()
    reports[name] = {'text': 'untrusted content'}
    with pytest.raises(ValueError, match='unexpected ZIP member'):
        AUDIT.audit(save(tmp_path, reports))


def test_duplicate_member_and_size_limits(tmp_path, monkeypatch):
    path = save(tmp_path, fixture_reports())
    with ZipFile(path, 'a') as archive, pytest.warns(UserWarning):
        archive.writestr('results/PLAN.json', '{}')
    with pytest.raises(ValueError, match='duplicate ZIP member'):
        AUDIT.audit(path)
    path = save(tmp_path, fixture_reports())
    monkeypatch.setattr(AUDIT, 'MAX_MEMBER_BYTES', 20)
    with pytest.raises(ValueError, match='ZIP member too large'):
        AUDIT.audit(path)


def test_cli_creates_report_and_refuses_overwrite(tmp_path):
    path = save(tmp_path, fixture_reports())
    output = tmp_path / 'audit.json'
    command = [sys.executable, '-I', str(SCRIPT), str(path), '--output', str(output)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())['report_consistency_passed']
    again = subprocess.run(command, text=True, capture_output=True)
    assert again.returncode != 0
    assert 'audit failed' in again.stderr
