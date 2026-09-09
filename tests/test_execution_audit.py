"""Keep the timing experiment's identity checks stricter than its speed claims."""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/rebenchmark_execution.py'
spec = importlib.util.spec_from_file_location('execution_audit', SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def original(tmp_path):
    root = tmp_path / 'original'; root.mkdir()
    checkpoint = root / 'weights-6100.pt'; checkpoint.write_bytes(b'fixture, not a torch checkpoint')
    study = root / 'study-6100'; study.mkdir()
    digest = audit.sha(checkpoint)
    data = {'seed': 32000, 'n': 4, 'batch_size': 2, 'sha256': 'same-cohort',
            'noise_std': 0.05, 'observe_probability': 0.7}
    audit.write(root / 'PLAN.json', {'training_seeds': [6100], 'test_seed': 32000, 'calibration_seed': 31000})
    audit.write(root / 'summary.json', {'training_seeds': [6100], 'replicates': 1})
    audit.write(study / 'PLAN.json', {'training_seed': 6100, 'test_seed': 32000, 'checkpoint_sha256': digest})
    audit.write(study / 'COMPLETED.json', {'verified': True, 'checkpoint_sha256': digest})
    audit.write(study / 'calibration.json', {'checkpoint_sha256': digest, 'data': {'seed': 31000}})
    audit.write(study / 'eval.json', {'dataset': data, 'calibration_sha256': audit.sha(study / 'calibration.json')})
    audit.write(study / 'benchmark.json', {'dataset': data})
    return root


def test_inventory_and_frozen_input_hashes(tmp_path):
    root = original(tmp_path)
    entries = audit.inventory(root)
    assert len(entries) == 1 and entries[0]['seed'] == 6100
    audit.verify_inputs(entries[0])
    (root / 'study-6100/calibration.json').write_text('{}')
    with pytest.raises(ValueError, match='input changed'):
        audit.verify_inputs(entries[0])


def test_wrong_checkpoint_refused(tmp_path):
    root = original(tmp_path)
    (root / 'weights-6100.pt').write_bytes(b'replaced')
    with pytest.raises(ValueError, match='hash mismatch'):
        audit.inventory(root)


def test_incomplete_or_wrong_cohort_refused(tmp_path):
    root = original(tmp_path)
    path = root / 'study-6100/benchmark.json'
    path.write_text('{"dataset": {"n": 5}}')
    with pytest.raises(ValueError, match='different cohorts'):
        audit.inventory(root)


def test_report_paths_cannot_escape(tmp_path):
    root = original(tmp_path)
    checkpoint = root / 'weights-6100.pt'
    outside = tmp_path / 'outside.pt'; outside.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink(); checkpoint.symlink_to(outside)
    with pytest.raises(ValueError, match='within the replication'):
        audit.inventory(root)


def test_counterbalanced_fresh_process_schedule():
    tasks = audit.schedule([6100, 7100, 8100, 9100, 10100], 2, 41000)
    assert tasks == audit.schedule([6100, 7100, 8100, 9100, 10100], 2, 41000)
    assert len(tasks) == 20
    for seed in (6100, 7100, 8100, 9100, 10100):
        first = [t['profile'] for t in tasks if t['seed'] == seed and t['pass'] == 0]
        second = [t['profile'] for t in tasks if t['seed'] == seed and t['pass'] == 1]
        assert first == second[::-1]
        assert all(t['case_order_seed'] == 41000 + t['pass'] for t in tasks)


@pytest.mark.parametrize('value', [True, -1, 0, 1.5, '2'])
def test_invalid_repeat_counts(value):
    with pytest.raises(ValueError):
        audit.integer(value, 'repeats')


def test_duplicate_and_nonfinite_json_rejected(tmp_path):
    p = tmp_path / 'bad.json'
    for text in ('{"a":1,"a":2}', '{"a":NaN}', '[]'):
        p.write_text(text)
        with pytest.raises(ValueError):
            audit.read(p)


def test_json_overwrite_refused(tmp_path):
    p = tmp_path / 'result.json'; audit.write(p, {'answer': 1})
    with pytest.raises(FileExistsError):
        audit.write(p, {'answer': 2})
    assert audit.read(p)['answer'] == 1


def test_common_environment_does_not_mutate_parent(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', 'parent-value')
    child = audit.child_environment()
    assert child['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == 'parent-value'


def test_prediction_comparison_numeric_tolerance_and_discrete_decisions(tmp_path):
    a, b = tmp_path / 'a.npz', tmp_path / 'b.npz'
    arrays = {'shared/mixture/estimate': np.array([[1., 2.]], np.float32),
              'shared/mixture/claim_index': np.array([-1], np.int64)}
    np.savez(a, **arrays)
    changed = {**arrays, 'shared/mixture/estimate': arrays['shared/mixture/estimate'] + 1e-7}
    np.savez(b, **changed)
    assert audit.compare_arrays(a, b)['all_outputs_match']
    changed['shared/mixture/claim_index'] = np.array([0], np.int64)
    np.savez(b, **changed)
    assert not audit.compare_arrays(a, b)['all_outputs_match']
    changed['shared/mixture/estimate'][0, 0] = np.nan
    np.savez(b, **changed)
    with pytest.raises(ValueError, match='nonfinite'):
        audit.compare_arrays(a, b)


def test_writing_into_original_or_existing_directory_refused(tmp_path):
    root = original(tmp_path)
    with pytest.raises(ValueError, match='outside'):
        audit.run(root, root / 'new', device='cpu')
    output = tmp_path / 'existing'; output.mkdir()
    with pytest.raises(FileExistsError):
        audit.run(root, output, device='cpu')


def test_fresh_worker_profiles_and_end_to_end_existing_checkpoint(tmp_path):
    pytest.importorskip('torch')
    if importlib.util.find_spec('universa_recurrent') is None:
        pytest.skip('full repository installation required for integration test')
    # Train only to produce a disposable CPU fixture. The rebenchmark runner
    # itself must never call training or mutate these weights.
    root = tmp_path / 'original'
    setup = [sys.executable, '-m', 'universa_recurrent.neural.dual_cli', 'replicate',
             '--device', 'cpu', '--seeds', '6100', '--train-size', '16', '--epochs', '1',
             '--train-batch-size', '8', '--hidden-dim', '8', '--steps', '2',
             '--calibration-size', '8', '--n', '8', '--batch-size', '4',
             '--warmup', '0', '--repeats', '0', '--output-dir', str(root)]
    subprocess.run(setup, check=True, capture_output=True, text=True, env=audit.child_environment(), timeout=90)
    digest = audit.sha(root / 'weights-6100.pt')
    output = tmp_path / 'comparison'
    archive = audit.run(root, output, device='cpu', passes=1, warmup=0, repeats=1)
    assert audit.sha(root / 'weights-6100.pt') == digest
    summary = audit.read(output / 'summary.json')
    assert summary['status'] == 'complete' and summary['all_profile_outputs_match']
    assert any(row['method'] == 'shared/mixture' for row in summary['rows'])
    profiles = [audit.read(p) for p in output.glob('*/execution.json')]
    assert len({p['pid'] for p in profiles}) == 2
    assert {p['settings']['deterministic_algorithms'] for p in profiles} == {True, False}
    assert {p['settings']['matmul_fp32_precision'] for p in profiles} == {'ieee'}
    with audit.ZipFile(archive) as bundle:
        assert 'summary.json' in bundle.namelist()
        assert all(Path(name).suffix in ('.json', '.log') for name in bundle.namelist())
    # Mutated precision is not attributed to determinism.
    execution = next(output.glob('*/execution.json'))
    record = audit.read(execution)
    record['settings']['matmul_fp32_precision'] = 'tf32'
    execution.write_text(__import__('json').dumps(record))
    with pytest.raises(ValueError, match='non-profile execution controls'):
        audit.summarize(output, audit.read(output / 'PLAN.json')['tasks'])
