import json
from pathlib import Path

import pytest

from native import run_acquisition


def protocol():
    return {
        'arms': ['dense', 'untyped', 'typed'],
        'max_updates': 4096,
        'max_loss_tokens': 5201920,
        'batch_size': 8,
        'seed': 20261110,
        'model_seed': 20261110,
        'data_seed': 20261110,
        'learning_rate': 0.0003,
        'weight_decay': 0.01,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'grad_clip': 1.0,
        'lambda_balance': 0.01,
        'lambda_route': 0.1,
        'lambda_state': 0.1,
        'warmup_fraction': 0.02,
        'final_lr_fraction': 0.1,
        'checkpoint_every': 500,
        'dev_every': 500,
        'dev_batches': 16,
        'example_count': 32,
        'evaluation_examples_per_split': 32,
        'max_new_tokens': 48,
    }


def complete_evaluation(*args, **kwargs):
    return {
        'splits': {
            split: {
                'generation': {'complete': True, 'evaluated': 32, 'correct': 32},
                'state_path': {'complete': True},
            }
            for split in ('train', 'dev')
        }
    }


def training_result(arguments):
    arm = Path(arguments.output).parent
    checkpoint = arm / 'train' / 'checkpoints' / 'final-update-00004096.pt'
    return {
        'format': 'native-train-v1', 'mode': arguments.mode,
        'step': 4096, 'planned_updates': 4096, 'consumed_loss_tokens': 5201920,
        'training_seconds': 10.0,
        'final_checkpoint': {'path': str(checkpoint), 'step': 4096, 'sha256': 'a' * 64},
    }


def test_mocked_runner_completes_equal_endpoints_and_isolates_outputs(tmp_path, monkeypatch):
    arguments = []
    monkeypatch.setattr(run_acquisition.train, 'run_training', lambda args: arguments.append(args) or training_result(args))
    monkeypatch.setattr(run_acquisition, 'evaluate', complete_evaluation)
    monkeypatch.setattr(run_acquisition.reconstruct, 'load_replay_spec', lambda path: {})
    monkeypatch.setattr(run_acquisition.reconstruct, 'reconstruct_checkpoint', lambda *args, **kwargs: {'agreement': True})
    monkeypatch.setattr(run_acquisition.reconstruct, 'write_reconstruction_report', lambda path, report: 'b' * 64)
    statuses = run_acquisition._execute(protocol(), tmp_path, tmp_path / 'attempt', 'tokenizer.json', 'replay.json')
    assert [status['updates'] for status in statuses.values()] == [4096, 4096, 4096]
    assert all(status['train_exact'] == 32 for status in statuses.values())
    assert len({call.output for call in arguments}) == 3
    assert {call.mode for call in arguments} == {'dense', 'untyped', 'typed'}
    for mode in statuses:
        assert (tmp_path / 'attempt' / 'arms' / mode / 'status.json').is_file()


def test_runner_preserves_a_failed_arm_and_does_not_truncate_the_other_arms(tmp_path, monkeypatch):
    def train(args):
        if args.mode == 'dense':
            raise RuntimeError('synthetic failure')
        return training_result(args)

    monkeypatch.setattr(run_acquisition.train, 'run_training', train)
    monkeypatch.setattr(run_acquisition, 'evaluate', complete_evaluation)
    monkeypatch.setattr(run_acquisition.reconstruct, 'load_replay_spec', lambda path: {})
    monkeypatch.setattr(run_acquisition.reconstruct, 'reconstruct_checkpoint', lambda *args, **kwargs: {'agreement': True})
    monkeypatch.setattr(run_acquisition.reconstruct, 'write_reconstruction_report', lambda path, report: 'b' * 64)
    statuses = run_acquisition._execute(protocol(), tmp_path, tmp_path / 'attempt', 'tokenizer.json', 'replay.json')
    assert statuses['dense']['status'] == 'failed'
    assert 'synthetic failure' in statuses['dense']['error']
    assert all(statuses[mode]['status'] == 'completed' for mode in ('untyped', 'typed'))


def test_reservation_limit_and_original_projection_leave_shutdown_margin():
    assert run_acquisition.RESERVATION_PROJECTED_HOURS * 1.5 + run_acquisition.SHUTDOWN_ALLOWANCE_HOURS <= 1.0
    assert run_acquisition.RESERVATION_PROJECTED_HOURS * 1.5 == pytest.approx(0.945)


def test_interruption_preserves_completed_and_current_arm_status(tmp_path, monkeypatch):
    class WatchdogStop(BaseException):
        pass

    def train(args):
        if args.mode == 'untyped':
            raise WatchdogStop('synthetic watchdog stop')
        return training_result(args)

    monkeypatch.setattr(run_acquisition.train, 'run_training', train)
    monkeypatch.setattr(run_acquisition, 'evaluate', complete_evaluation)
    monkeypatch.setattr(run_acquisition.reconstruct, 'load_replay_spec', lambda path: {})
    monkeypatch.setattr(run_acquisition.reconstruct, 'reconstruct_checkpoint', lambda *args, **kwargs: {'agreement': True})
    monkeypatch.setattr(run_acquisition.reconstruct, 'write_reconstruction_report', lambda path, report: 'b' * 64)
    run_dir = tmp_path / 'attempt'
    with pytest.raises(WatchdogStop):
        run_acquisition._execute(protocol(), tmp_path, run_dir, 'tokenizer.json', 'replay.json')
    recovered = run_acquisition._recover_statuses(run_dir, {mode: {'status': 'not_run'} for mode in protocol()['arms']})
    assert recovered['dense']['status'] == 'completed'
    assert recovered['dense']['updates'] == 4096
    assert recovered['untyped']['status'] == 'interrupted'
    assert recovered['untyped']['stage'] == 'training'
    assert recovered['typed']['status'] == 'not_run'


def test_source_provenance_binds_original_manifest_and_tokenizer(tmp_path):
    from native.prepare_acquisition import write_source_provenance
    source = tmp_path / 'source'
    (source / 'structured').mkdir(parents=True)
    (source / 'manifest.json').write_text('{"format": "preserved"}\n')
    for split in ('train', 'dev'):
        (source / 'structured' / f'{split}.jsonl').write_text(f'{{"split": "{split}"}}\n')
    tokenizer = tmp_path / 'tokenizer.json'
    tokenizer.write_text('{}\n')
    provenance = write_source_provenance(source, tmp_path / 'output', tokenizer)
    assert provenance['source_data'] == str(source)
    assert len(provenance['source_manifest_sha256']) == 64
    assert len(provenance['structured_files']['train']) == 64
    assert len(provenance['tokenizer_sha256']) == 64
