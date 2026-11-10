import hashlib
import json

from native.independent.export_acquisition_bundle import export
from native.structured import Instruction, make_example


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


def test_acquisition_export_is_weight_free_and_independently_rescoreable(tmp_path):
    run = tmp_path / 'data/acquisition/attempt-1'
    data = tmp_path / 'data'
    arm = run / 'arms/dense'
    protocol = {
        'arms': ['dense'], 'max_updates': 1, 'max_loss_tokens': 10, 'batch_size': 1,
        'target_exact_answers': 1,
    }
    write(run / 'protocol.json', protocol)
    (run / 'protocol.sha256').write_text(hashlib.sha256((run / 'protocol.json').read_bytes()).hexdigest() + '\n')
    ledger = tmp_path / 'ledger.jsonl'
    ledger.write_text(json.dumps({
        'event': 'start', 'reservation': 'reservation-1',
        'extra': {'native_acquisition_run_id': 'attempt-1'},
    }) + '\n')
    write(run / 'launch.json', {
        'run_id': 'attempt-1', 'source_commit': 'a' * 40,
        'ledger_before': {'ledger': str(ledger)},
    })
    write(run / 'run_status.json', {
        'complete': True, 'charged_hours': 0.01, 'one_hour_limit_respected': True,
        'reservation_id': 'reservation-1', 'failure': None,
    })
    write(arm / 'status.json', {
        'status': 'completed', 'updates': 1, 'consumed_loss_tokens': 10, 'wall_seconds': 1.0,
    })
    manifest = tmp_path / 'checkpoint.manifest.json'
    write(manifest, {'weights': 'local-only.pt', 'sha256': 'b' * 64})
    write(arm / 'train-result.json', {
        'final_checkpoint': {'path': 'local-only.pt', 'manifest_path': str(manifest), 'sha256': 'b' * 64},
        'training_seconds': 1.0, 'data_loading_seconds': 0.1, 'forward_seconds': 0.2,
        'backward_seconds': 0.3, 'optimizer_seconds': 0.4, 'checkpoint_seconds': 0.1,
        'actual_loss_tokens_per_second': 10.0, 'cuda_peak_allocated_bytes': 100,
        'dev': {'route_usage': []},
    })
    write(arm / 'train/learning_curve.jsonl', {'event': 'train', 'loss': 1.0})
    with (arm / 'train/learning_curve.jsonl').open('a') as handle:
        handle.write(json.dumps({'event': 'dev', 'loss': 2.0}) + '\n')
    splits = {}
    (data / 'structured').mkdir(parents=True)
    for split in ('train', 'dev'):
        source = make_example(split, [Instruction('set', 0, value=1)], (0, 0, 0, 0)).to_record()
        (data / 'structured' / f'{split}.jsonl').write_text(json.dumps(source) + '\n')
        saved = {
            'example_id': source['example_id'], 'completion_text': source['target_text'],
            'target_text': source['target_text'], 'correct': True, 'valid': True,
            'stopped_on_eos': True, 'truncated': False, 'prompt_truncated': False,
        }
        splits[split] = {
            'scope': 'memorization fit' if split == 'train' else 'separate held-out diagnostic',
            'generation': {'complete': True, 'evaluated': 1, 'correct': 1, 'records': [saved]},
            'independent_diagnostics': {'exact_count': 1, 'records': [{'exact_match': True}]},
            'state_path': {
                'answer_token_cross_entropy': {split: {'mean': 0.1, 'count': 17}},
                'metrics': {'pass_1': {'accuracy': 1.0}}, 'complete': True,
                'state_schedule': 'before, after, after, after',
                'evidence_boundary': 'readout is not causal semantic use',
            },
        }
    write(arm / 'evaluation.json', {
        'format': 'native-acquisition-evaluation-v1',
        'checkpoint': {'config': {'model': {'mode': 'dense'}}}, 'splits': splits,
    })
    for split in ('train', 'dev'):
        write(arm / f'{split}_state_path.json', splits[split]['state_path'])
    write(arm / 'reconstruction.json', {
        'agreement': True, 'execution': {'device': 'cpu'},
        'teacher_forced': {'agrees': True}, 'generation': {'agrees': True},
    })
    for name in ('manifest.json', 'acquisition_protocol.json', 'acquisition_selection.json', 'source_provenance.json'):
        write(data / name, {'test_fixture': True})
    output = tmp_path / 'evidence'
    summary = export(run, data, output)
    assert summary['matched_endpoints_verified']
    assert summary['all_memorization_targets_met']
    assert summary['arms']['dense']['splits']['train']['answer_token_cross_entropy']['train']['mean'] == 0.1
    diagnostics = json.loads((output / 'generation_diagnostics.json').read_text())
    assert diagnostics['arms']['dense']['structured']['train']['exact_count'] == 1
    assert diagnostics['target_validation']['executed_programs'] == 2
    assert not list(output.rglob('*.pt'))
    for line in (output / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected


def test_failed_and_unstarted_arms_remain_exportable_and_rescoreable(tmp_path):
    run = tmp_path / 'data/acquisition/failed-attempt'
    data = tmp_path / 'data'
    protocol = {
        'arms': ['dense', 'untyped'], 'max_updates': 4096, 'max_loss_tokens': 5201920,
        'batch_size': 8, 'target_exact_answers': 31,
    }
    write(run / 'protocol.json', protocol)
    (run / 'protocol.sha256').write_text(hashlib.sha256((run / 'protocol.json').read_bytes()).hexdigest() + '\n')
    ledger = tmp_path / 'ledger.jsonl'
    ledger.write_text(json.dumps({
        'event': 'start', 'reservation': 'reservation-failed',
        'extra': {'native_acquisition_run_id': 'failed-attempt'},
    }) + '\n')
    write(run / 'launch.json', {
        'run_id': 'failed-attempt', 'source_commit': 'a' * 40,
        'ledger_before': {'ledger': str(ledger)},
    })
    write(run / 'run_status.json', {
        'complete': False, 'charged_hours': 0.01, 'one_hour_limit_respected': True,
        'reservation_id': 'reservation-failed', 'failure': None,
        'arms': {'dense': {'status': 'failed'}, 'untyped': {'status': 'not_run'}},
    })
    write(run / 'arms/dense/status.json', {
        'status': 'failed', 'stage': 'reconstructing', 'error': 'synthetic failure',
    })
    (data / 'structured').mkdir(parents=True)
    splits = {}
    for split in ('train', 'dev'):
        source = make_example(split, [Instruction('set', 0, value=1)], (0, 0, 0, 0)).to_record()
        (data / 'structured' / f'{split}.jsonl').write_text(json.dumps(source) + '\n')
        saved = {
            'example_id': source['example_id'], 'completion_text': source['target_text'],
            'target_text': source['target_text'], 'correct': True, 'valid': True,
            'stopped_on_eos': True, 'truncated': False, 'prompt_truncated': False,
        }
        splits[split] = {
            'generation': {'records': [saved]},
            'state_path': {'answer_token_cross_entropy': {split: {'mean': 0.1}}},
        }
    write(run / 'arms/dense/evaluation.json', {
        'format': 'native-acquisition-evaluation-v1', 'splits': splits,
    })
    for name in ('manifest.json', 'acquisition_protocol.json', 'acquisition_selection.json', 'source_provenance.json'):
        write(data / name, {'test_fixture': True})
    output = tmp_path / 'evidence'
    summary = export(run, data, output)
    assert summary['arms']['dense']['status']['error'] == 'synthetic failure'
    assert summary['arms']['untyped']['status']['status'] == 'not_run'
    assert not summary['matched_endpoints_verified']
    assert (output / 'arms/dense/evaluation.json').is_file()
    import subprocess
    import sys
    from pathlib import Path
    subprocess.run([
        sys.executable, 'research/ti_tnn/native/independent/rescore_generations.py',
        '--study', str(output), '--data', str(output / 'data'),
        '--output', str(tmp_path / 'second-reproduction.json'),
    ], cwd=Path(__file__).parents[1], check=True)
    first = json.loads((output / 'generation_diagnostics.json').read_text())
    reproduced = json.loads((output / 'generation_diagnostics.reproduced.json').read_text())
    assert {key: value for key, value in first.items() if key != 'study'} == {
        key: value for key, value in reproduced.items() if key != 'study'
    }
