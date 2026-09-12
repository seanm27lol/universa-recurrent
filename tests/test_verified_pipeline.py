"""Every new request must regenerate and bind-check all of its receipts."""
from __future__ import annotations

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

torch = pytest.importorskip('torch')
SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
pipeline = importlib.import_module('benchmark_verified_pipeline')


@pytest.fixture(scope='module')
def references(tmp_path_factory):
    from universa_recurrent.neural.dual_study import calibrate, load_estimators, write_json
    from universa_recurrent.neural.data import StructuredFlowDataset
    from universa_recurrent.neural.v2_train import train_v2_checkpoint
    root = tmp_path_factory.mktemp('verified-pipeline')
    checkpoint = root / 'weights-7100.pt'
    torch.set_num_threads(1)
    train_v2_checkpoint(checkpoint, device_name='cpu', seed=7100, train_size=16,
        calibration_size=8, epochs=1, batch_size=8, hidden_dim=8,
        candidate_embedding_dim=2, steps=8, include_controls=True)
    engines, metadata, device, _ = load_estimators(checkpoint, 'cpu')
    _, calibration = calibrate(engines, StructuredFlowDataset(8, seed=7200), device, .75, 8)
    calibration['checkpoint_sha256'] = metadata['checkpoint_sha256']
    calibration['model_sources'] = {engine.name: engine.checkpoint_sha256
                                     for engine in engines if engine.structured}
    path = root / 'study-7100/calibration.json'
    write_json(path, calibration)
    return root, checkpoint, path


def worker_args(references, **changes):
    _, checkpoint, calibration = references
    args = dict(checkpoint=checkpoint, calibration=calibration, device='cpu',
                cpu_threads=1, test_seed=36000, order_seed=37000,
                record_counts=[1, 2], repeats=1, warmup=0, models=list(pipeline.MODELS))
    args.update(changes)
    return SimpleNamespace(**args)


@pytest.fixture(scope='module')
def result(references):
    return pipeline.run_worker(worker_args(references))


def test_complete_requests_and_saved_receipts(result):
    assert result['all_equivalence_checks_passed']
    assert result['all_saved_receipts_audited']
    assert result['inputs']['distinct_observed_mask_count'] == 2
    assert len(result['rows']) == 10
    assert len(result['saved_receipts']) == 5
    for group in result['saved_receipts']:
        assert len(group['records']) == group['record_count'] == 2
        assert group['audit']['all_accepted']
        assert group['audit']['outer_checks_and_retained_steps_agree']
        payloads = [pipeline.dumps(record) for record in group['records']]
        assert group['batch_receipts_sha256'] == pipeline.batch_digest(payloads)
    for row in result['rows']:
        assert row['all_timed_outputs_compared']
        assert row['all_timed_receipts_bound_checked']
        assert row['unique_timed_receipt_hash_count'] == 1
        assert row['bitwise_receipts_equal_across_timed_runs']
        for mode, samples in row['samples'].items():
            assert len(samples) == 1
            sample = samples[0]
            assert sample['verified_count'] == row['record_count']
            assert sample['total_ms'] >= sum(sample['stages_ms'].values())
            assert sample['total_ms'] == row['timings'][mode]['samples_ms'][0]
            assert (sample['stages_ms']['reference_prepare'] > 0) == (mode == 'prepare_once_per_batch')


@pytest.mark.parametrize('field', ['estimate', 'checkpoint', 'scope', 'trajectory'])
def test_corrupted_receipt_is_rejected(references, result, field):
    _, checkpoint, calibration = references
    record = copy.deepcopy(next(group for group in result['saved_receipts']
                                if group['retention'] == 'trajectory')['records'][1])
    if field == 'estimate':
        record['endpoint']['estimate'][0] += 1
    elif field == 'checkpoint':
        record['endpoint']['checkpoint_sha256'] = 'a' * 64
    elif field == 'scope':
        record['scope']['execution_authenticated'] = True
    else:
        record['trajectory']['states'][0][0][0] += 1
    context = pipeline.PreparedVerifier.prepare(checkpoint, calibration)
    for mode in ('files_per_record', 'reuse_prepared_snapshot'):
        with pytest.raises(ValueError, match='bound receipt'):
            pipeline.verify_batch([pipeline.dumps(record)], mode, context, checkpoint, calibration)


@pytest.mark.parametrize('payload', [b'{"a":1,"a":2}', b'{"a":NaN}'])
def test_strict_json_before_verification(payload):
    with pytest.raises(ValueError):
        pipeline.verify_batch([payload], 'reuse_prepared_snapshot', None, None, None)


def test_all_batch_members_are_checked(references, result):
    _, checkpoint, calibration = references
    records = copy.deepcopy(result['saved_receipts'][0]['records'])
    records[-1]['endpoint']['estimate'][0] += 1
    context = pipeline.PreparedVerifier.prepare(checkpoint, calibration)
    with pytest.raises(ValueError):
        pipeline.verify_batch([pipeline.dumps(r) for r in records],
                              'reuse_prepared_snapshot', context, checkpoint, calibration)


def test_input_audit_rejects_mismatched_record(references, result):
    _, checkpoint, calibration = references
    records = result['saved_receipts'][0]['records']
    observed = torch.tensor([r['endpoint']['observed'] for r in records])
    masks = torch.tensor([r['endpoint']['mask'] for r in records])
    observed[1, 0] += 1
    with pytest.raises(ValueError, match='common cohort'):
        pipeline.audit_receipts([pipeline.dumps(r) for r in records],
            pipeline.PreparedVerifier.prepare(checkpoint, calibration), checkpoint, calibration,
            observed, masks)


@pytest.mark.parametrize('seed', [7101, 7102, 7200, 22000, 32000, 33000])
def test_reserved_data_seeds_rejected(references, seed):
    with pytest.raises(ValueError, match='overlaps'):
        pipeline.run_worker(worker_args(references, test_seed=seed))


def test_changed_reference_is_rejected(references, monkeypatch):
    original = pipeline.digest
    checkpoint = references[1]
    calls = 0
    def changed(path):
        nonlocal calls
        if path == checkpoint:
            calls += 1
            if calls > 1:
                return '0' * 64
        return original(path)
    monkeypatch.setattr(pipeline, 'digest', changed)
    with pytest.raises(ValueError, match='changed'):
        pipeline.run_worker(worker_args(references, models=['direct'], record_counts=[1]))


def test_full_batch_checksum_includes_order_and_last_receipt():
    assert pipeline.batch_digest([b'a', b'bc']) != pipeline.batch_digest([b'ab', b'c'])
    assert pipeline.batch_digest([b'a', b'b']) != pipeline.batch_digest([b'b', b'a'])
    assert pipeline.batch_digest([b'a', b'b']) != pipeline.batch_digest([b'a', b'c'])


def test_duplicate_cohort_rejected(references, monkeypatch):
    original = pipeline.StructuredFlowDataset
    def duplicated(*args, **kwargs):
        dataset = original(*args, **kwargs)
        dataset.observed[1] = dataset.observed[0]
        dataset.mask[1] = dataset.mask[0]
        return dataset
    monkeypatch.setattr(pipeline, 'StructuredFlowDataset', duplicated)
    with pytest.raises(ValueError, match='duplicate observed'):
        pipeline.run_worker(worker_args(references))


def test_portable_source_does_not_inherit_parent_git_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, 'package_source_identity', lambda: {'git_commit': 'unrelated'})
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *args, **kwargs: str(tmp_path))
    assert pipeline.source_identity()['git_commit'] is None


def test_mismatched_cohort_across_checkpoints_rejected(tmp_path, monkeypatch):
    root = tmp_path / 'replication'
    for seed in (1, 2):
        directory = root / f'study-{seed}'
        directory.mkdir(parents=True)
        (root / f'weights-{seed}.pt').write_bytes(str(seed).encode())
        (directory / 'calibration.json').write_bytes(b'{}')
    source, scripts = pipeline.source_identity(), pipeline.script_identity()
    monkeypatch.setattr(pipeline, 'source_identity', lambda: source)
    def fake_worker(cmd, **kwargs):
        checkpoint = Path(cmd[cmd.index('--checkpoint') + 1])
        calibration = Path(cmd[cmd.index('--calibration') + 1])
        output = Path(cmd[cmd.index('--worker-output') + 1])
        seed = int(checkpoint.stem.removeprefix('weights-'))
        pipeline.write_json(output, {'training_seed': seed, 'source': source,
            'scripts_sha256': scripts, 'checkpoint_sha256': pipeline.digest(checkpoint),
            'calibration_sha256': pipeline.digest(calibration),
            'dataset_sha256': str(seed), 'inputs': {'count': 2}, 'rows': []})
    monkeypatch.setattr(pipeline.subprocess, 'run', fake_worker)
    args = SimpleNamespace(replication_dir=root, output_dir=tmp_path / 'out',
        order_seed=37000, record_counts=[1, 2], models=['direct'], test_seed=36000,
        repeats=1, warmup=0, device='cpu', cpu_threads=1)
    with pytest.raises(ValueError, match='input cohorts differ'):
        pipeline.run_parent(args)


def test_cpu_command_line_writes_complete_reports_only(references, tmp_path):
    root, _, _ = references
    out = tmp_path / 'out'
    args = ['--replication-dir', str(root), '--output-dir', str(out), '--device', 'cpu',
            '--record-counts', '1', '2', '--models', 'direct', '--repeats', '1', '--warmup', '0']
    assert pipeline.main(args) == 0
    summary = pipeline.read_json(out / 'summary.json')
    assert summary['saved_receipt_count'] == 2
    assert summary['all_equivalence_checks_passed']
    assert summary['checkpoint_runs'] == 1
    assert not list(out.rglob('*.pt'))
    assert pipeline.main(args) == 2


def test_cli_rejects_missing_inputs_and_duplicate_counts(tmp_path):
    common = ['--replication-dir', str(tmp_path), '--output-dir', str(tmp_path / 'new')]
    assert pipeline.main(common) == 2
    assert pipeline.main(common + ['--record-counts', '1', '1']) == 2
