"""The held-out phase cannot begin until every model policy is frozen."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
study = importlib.import_module('run_claim_reliability')


@pytest.fixture(scope='module')
def references(tmp_path_factory):
    from universa_recurrent.neural.dual_study import calibrate, load_estimators, write_json
    from universa_recurrent.neural.data import StructuredFlowDataset
    from universa_recurrent.neural.v2_train import train_v2_checkpoint
    root = tmp_path_factory.mktemp('claim-reliability-references')
    torch.set_num_threads(1)
    for seed in (7100, 8100):
        checkpoint = root / f'weights-{seed}.pt'
        train_v2_checkpoint(checkpoint, device_name='cpu', seed=seed, train_size=16,
            calibration_size=8, epochs=1, batch_size=8, hidden_dim=8,
            candidate_embedding_dim=2, steps=8, include_controls=True)
        engines, metadata, device, _ = load_estimators(checkpoint, 'cpu')
        _, calibration = calibrate(engines, StructuredFlowDataset(8, seed=seed + 100), device, .75, 8)
        calibration['checkpoint_sha256'] = metadata['checkpoint_sha256']
        calibration['model_sources'] = {engine.name: engine.checkpoint_sha256
                                       for engine in engines if engine.structured}
        calibration['test_seed_reserved'] = seed + 200
        write_json(root / f'study-{seed}/calibration.json', calibration)
    return root


def arguments(references, output, **changes):
    values = dict(replication_dir=references, output_dir=output, device='cpu',
                  protocol=study.ROOT / 'experiments/claim_reliability_v1.json',
                  smoke=True, smoke_calibration_size=8, smoke_test_size=16)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture(scope='module')
def completed(references, tmp_path_factory):
    directory = tmp_path_factory.mktemp('claim-reliability-output') / 'results'
    result = study.run_parent(arguments(references, directory))
    return directory, result


def test_cpu_cli_complete_workflow(references, tmp_path):
    output = tmp_path / 'cli'
    result = subprocess.run([sys.executable, str(SCRIPTS / 'run_claim_reliability.py'),
        '--replication-dir', str(references), '--output-dir', str(output), '--device', 'cpu',
        '--smoke', '--smoke-calibration-size', '8', '--smoke-test-size', '8'],
        text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = study.read_json(output / 'summary.json')
    assert summary['format'] == study.FORMAT + '.complete'
    assert summary['smoke'] and not summary['prespecified_protocol']
    assert summary['aggregate']['models']['gaussian_generator_reference']['n_fits'] == 1
    assert summary['aggregate']['models']['shared']['n_fits'] == 2
    assert 'All calibration policies locked' in result.stdout


def test_all_fits_frozen_before_any_test(completed):
    directory, summary = completed
    manifest = study.read_json(directory / 'manifest.json')
    lock = study.read_json(directory / 'calibration-lock.json')
    lock_time = (directory / 'calibration-lock.json').stat().st_mtime_ns
    assert lock['training_seeds'] == [7100, 8100]
    for seed in lock['training_seeds']:
        assert (directory / f'calibration-{seed}.json').stat().st_mtime_ns <= lock_time
        assert (directory / f'test-{seed}.json').stat().st_mtime_ns >= lock_time
        test = study.read_json(directory / f'test-{seed}.json')
        cal = study.read_json(directory / f'calibration-{seed}.json')
        assert test['policies'] == cal['policies']
        assert cal['dataset']['seed'] == 51000
        assert test['dataset']['seed'] == 52000
        assert test['calibration_lock_sha256'] == summary['calibration_lock_sha256']
    assert manifest['protocol']['test_seed'] == 42000
    assert manifest['effective_protocol']['test_seed'] == 52000
    assert summary['all_calibration_policies_locked_before_test']
    assert summary['calibration_test_inputs_disjoint']


def test_saved_arrays_suffice_for_quality_recomputation(completed):
    directory, summary = completed
    for seed in (7100, 8100):
        cal = study.read_json(directory / f'calibration-{seed}.json')
        test = study.read_json(directory / f'test-{seed}.json')
        ca, ta = study.load_arrays(directory, cal['artifact']), study.load_arrays(directory, test['artifact'])
        assert set(('observed', 'mask', 'truth', 'labels')) <= set(ca) & set(ta)
        study.assert_disjoint(ca, ta)
        for name in study.MODELS:
            derived = study.analyze_model(ca[name + '__probabilities'], ta[name + '__probabilities'],
                ta[name + '__estimates'], ta['truth'], ta['labels'], policies=cal['policies'][name])
            assert derived == test['analysis'][name]
            assert ta[name + '__estimates'].dtype == (np.float64 if name.startswith('gaussian') else np.float32)
    assert summary['all_saved_arrays_reloaded_and_metrics_recomputed']


def test_gaussian_computed_once_and_reused(completed):
    directory, _ = completed
    for phase in ('calibration', 'test'):
        one = study.read_json(directory / f'{phase}-7100.json')
        two = study.read_json(directory / f'{phase}-8100.json')
        assert one['models']['gaussian_generator_reference']['computed_in_this_worker']
        assert not two['models']['gaussian_generator_reference']['computed_in_this_worker']
        assert one['models']['gaussian_generator_reference']['prediction_device'] == 'cpu'
        a, b = study.load_arrays(directory, one['artifact']), study.load_arrays(directory, two['artifact'])
        for key in a:
            if key.startswith('gaussian'):
                assert np.array_equal(a[key], b[key])


def test_existing_output_refused(references, completed):
    with pytest.raises(ValueError, match='overwrite'):
        study.run_parent(arguments(references, completed[0]))


def test_default_requires_all_five_fits(references):
    with pytest.raises(ValueError, match='exactly the five'):
        study.inspect_references(references, study.FIXED, False)


def test_no_smoke_override_without_smoke(references, tmp_path):
    with pytest.raises(ValueError, match='require --smoke'):
        study.run_parent(arguments(references, tmp_path / 'unused', smoke=False))


@pytest.mark.parametrize('key,value', [('test_seed', 36000), ('test_size', 8),
    ('primary_coverage', .5), ('models', ['direct']), ('bootstrap_repeats', 1)])
def test_changed_confirmatory_protocol_refused(tmp_path, key, value):
    protocol = study.read_json(study.ROOT / 'experiments/claim_reliability_v1.json')
    protocol[key] = value
    path = tmp_path / 'protocol.json'
    study.write_json(path, protocol)
    with pytest.raises(ValueError, match='frozen v1 field'):
        study.validate_protocol(path)


def test_all_explicit_provenance_seeds_reserved():
    payload = {'training': {'seed': 7100, 'train_seed': 7101, 'calibration_seed': 7102},
               'calibration': {'data': {'seed': 31000}, 'test_seed_reserved': 32000},
               'old': [{'evaluation_seed': 36000}]}
    assert study.seeds_in_provenance(payload) == {7100, 7101, 7102, 31000, 32000, 36000}
    with pytest.raises(ValueError, match='nonnegative integer'):
        study.seeds_in_provenance({'seed': True})


@pytest.mark.parametrize('seed', [7101, 7200, 7300, 21000, 31000, 36000])
def test_reserved_seeds_refused(references, seed):
    protocol = {**study.FIXED, 'calibration_seed': seed, 'test_seed': 52000}
    with pytest.raises(ValueError, match='overlaps reserved'):
        study.inspect_references(references, protocol, True)


def test_duplicate_inputs_and_signed_zero_refused():
    with pytest.raises(ValueError, match='duplicate'):
        study.input_identity(np.array([[0., 1.], [-0., 1.]], np.float32), np.ones((2, 2), np.float32))
    first = {'observed': np.array([[1., 0.]]), 'mask': np.array([[1., 0.]])}
    second = {'observed': np.array([[1., -0.]]), 'mask': np.array([[1., 0.]])}
    with pytest.raises(ValueError, match='overlap'):
        study.assert_disjoint(first, second)


def test_array_artifact_rejects_modification(tmp_path):
    identity = study.save_arrays(tmp_path / 'data.npz', {'a': np.array([1., 2.])})
    assert study.load_arrays(tmp_path, identity)['a'].tolist() == [1., 2.]
    with (tmp_path / 'data.npz').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='changed'):
        study.load_arrays(tmp_path, identity)


def test_object_arrays_refused(tmp_path):
    with pytest.raises(ValueError, match='without objects'):
        study.save_arrays(tmp_path / 'data.npz', {'a': np.array([{}], dtype=object)})


def test_test_phase_without_lock_never_generates_dataset(references, completed, tmp_path, monkeypatch):
    directory, _ = completed
    manifest_bytes = (directory / 'manifest.json').read_bytes()
    (tmp_path / 'manifest.json').write_bytes(manifest_bytes)
    def forbidden(*args, **kwargs):
        pytest.fail('test dataset generated before complete global lock')
    monkeypatch.setattr(study, 'StructuredFlowDataset', forbidden)
    args = arguments(references, tmp_path, phase='test', training_seed=7100,
        manifest_sha256=study.sha256_file(tmp_path / 'manifest.json'), lock_sha256='0' * 64)
    with pytest.raises(ValueError, match='lock'):
        study.run_phase(args)


def test_lock_rejects_missing_fit(completed, tmp_path):
    directory, summary = completed
    lock = study.read_json(directory / 'calibration-lock.json')
    lock['calibrations'] = []
    study.write_json(tmp_path / 'calibration-lock.json', lock)
    manifest = study.read_json(directory / 'manifest.json')
    with pytest.raises(ValueError, match='omits a fit'):
        study.check_lock(tmp_path, manifest, summary['manifest_sha256'],
                         study.sha256_file(tmp_path / 'calibration-lock.json'))


def test_changed_source_or_reference_refused(references, completed, monkeypatch):
    directory, summary = completed
    manifest = study.read_json(directory / 'manifest.json')
    args = arguments(references, directory)
    monkeypatch.setattr(study, 'source_identity', lambda: {'different': True})
    with pytest.raises(ValueError, match='source changed'):
        study.check_immutable(args, manifest, summary['manifest_sha256'])


def test_depth_is_validated(references):
    engines, metadata, _, _ = study.load_estimators(references / 'weights-7100.pt', 'cpu')
    shared = next(engine for engine in engines if engine.name == 'shared')
    shared.model.config = copy.copy(shared.model.config)
    object.__setattr__(shared.model.config, 'steps', 7)
    with pytest.raises(ValueError, match='trained depth 8'):
        study.validate_loaded(engines, metadata, 7100)


def test_protocol_change_during_setup_refused(references, tmp_path, monkeypatch):
    path = tmp_path / 'protocol.json'
    path.write_bytes((study.ROOT / 'experiments/claim_reliability_v1.json').read_bytes())
    original = study.inspect_references
    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        path.write_text(path.read_text() + '\n')
        return result
    monkeypatch.setattr(study, 'inspect_references', changing)
    with pytest.raises(ValueError, match='protocol changed'):
        study.run_parent(arguments(references, tmp_path / 'output', protocol=path))
    assert not (tmp_path / 'output').exists()
