"""Export one acquisition attempt as compact, independently scorable, weight-free evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoints(path, event):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()
               if line.strip() and json.loads(line).get('event') == event]
    return {'first': records[0] if records else None, 'last': records[-1] if records else None,
            'records': len(records), 'full_curve': f'arms/{path.parent.parent.name}/learning_curve.jsonl'}


def compact_split(values):
    generation = {key: value for key, value in values['generation'].items() if key != 'records'}
    independent = {key: value for key, value in values['independent_diagnostics'].items() if key != 'records'}
    return {
        'scope': values['scope'],
        'generation': generation,
        'independent_diagnostics': independent,
        'answer_token_cross_entropy': values['state_path']['answer_token_cross_entropy'],
        'state_readout': values['state_path']['metrics'],
        'state_complete': values['state_path']['complete'],
        'state_schedule': values['state_path']['state_schedule'],
        'state_evidence_boundary': values['state_path']['evidence_boundary'],
    }


def export(run, data, output):
    run = Path(run).resolve()
    data = Path(data).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise RuntimeError(f'refusing to overwrite an existing evidence bundle: {output}')
    output.mkdir(parents=True)
    launch = load(run / 'launch.json')
    status = load(run / 'run_status.json')
    protocol = load(run / 'protocol.json')
    assert digest(run / 'protocol.json') == (run / 'protocol.sha256').read_text().strip()

    def copy(source, destination):
        destination = output / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    summary = {
        'format': 'native-acquisition-weight-free-summary-v1',
        'run_id': launch['run_id'],
        'source_commit': launch['source_commit'],
        'complete': status['complete'],
        'matched_endpoint': {'updates': protocol['max_updates'], 'loss_tokens': protocol['max_loss_tokens'],
                             'batch_size': protocol['batch_size']},
        'target_exact_answers': protocol['target_exact_answers'],
        'memorization_not_generalization': True,
        'charged_hours': status['charged_hours'],
        'one_hour_limit_respected': status['one_hour_limit_respected'],
        'reservation_id': status['reservation_id'],
        'failure': status['failure'],
        'arms': {},
    }
    for mode in protocol['arms']:
        arm = run / 'arms' / mode
        arm_status = load(arm / 'status.json')
        copy(arm / 'status.json', f'arms/{mode}/status.json')
        if arm_status['status'] != 'completed':
            summary['arms'][mode] = arm_status
            continue
        result = load(arm / 'train-result.json')
        evaluation = load(arm / 'evaluation.json')
        reconstruction = load(arm / 'reconstruction.json')
        for name in ('train-result.json', 'evaluation.json', 'train_state_path.json',
                     'dev_state_path.json', 'reconstruction.json'):
            copy(arm / name, f'arms/{mode}/{name}')
        for name in ('run_config.json', 'learning_curve.jsonl', 'checkpoint_journal.jsonl', 'recovery.jsonl'):
            source = arm / 'train' / name
            if source.is_file():
                copy(source, f'arms/{mode}/{name}')
        checkpoint = result['final_checkpoint']
        copy(checkpoint['manifest_path'], f'checkpoints/{mode}.manifest.json')
        summary['arms'][mode] = {
            'status': arm_status,
            'configuration': evaluation['checkpoint']['config']['model'],
            'final_checkpoint': checkpoint,
            'loss_vs_tokens': {
                'train': endpoints(arm / 'train/learning_curve.jsonl', 'train'),
                'dev': endpoints(arm / 'train/learning_curve.jsonl', 'dev'),
            },
            'final_teacher_forced_dev': result['dev'],
            'measured_time': {
                'training_seconds': result['training_seconds'],
                'data_loading_seconds': result['data_loading_seconds'],
                'forward_seconds': result['forward_seconds'],
                'backward_seconds': result['backward_seconds'],
                'optimizer_seconds': result['optimizer_seconds'],
                'checkpoint_seconds': result['checkpoint_seconds'],
                'train_and_evaluation_wall_seconds': arm_status['wall_seconds'],
                'actual_loss_tokens_per_second': result['actual_loss_tokens_per_second'],
            },
            'memory': {key: value for key, value in result.items() if key.startswith('cuda_')},
            'splits': {split: compact_split(values) for split, values in evaluation['splits'].items()},
            'routing': result['dev']['route_usage'],
            'reconstruction': {
                'agreement': reconstruction['agreement'],
                'execution': reconstruction['execution'],
                'teacher_forced_agrees': reconstruction['teacher_forced']['agrees'],
                'generation_agrees': reconstruction['generation']['agrees'],
                'report_sha256': digest(arm / 'reconstruction.json'),
            },
            'memorization_target_met': (
                evaluation['splits']['train']['generation']['correct'] >= protocol['target_exact_answers']),
        }
    summary['matched_endpoints_verified'] = all(
        arm.get('status', {}).get('updates') == protocol['max_updates']
        and arm.get('status', {}).get('consumed_loss_tokens') == protocol['max_loss_tokens']
        for arm in summary['arms'].values())
    summary['all_memorization_targets_met'] = all(
        arm.get('memorization_target_met') is True for arm in summary['arms'].values())

    for name in ('launch.json', 'protocol.json', 'protocol.sha256', 'run_status.json'):
        copy(run / name, name)
    for name in ('manifest.json', 'acquisition_protocol.json', 'acquisition_selection.json', 'source_provenance.json'):
        copy(data / name, f'data/{name}')
    for split in ('train', 'dev'):
        copy(data / 'structured' / f'{split}.jsonl', f'data/structured/{split}.jsonl')
    source = Path(__file__).resolve().parents[1] / 'configs/engineering.json'
    copy(source, 'architecture.json')
    ledger_path = Path(launch['ledger_before']['ledger'])
    events = [json.loads(line) for line in ledger_path.read_text().splitlines() if line.strip()]
    reservation_ids = {event['reservation'] for event in events
                       if event.get('event') == 'start'
                       and event.get('extra', {}).get('native_acquisition_run_id') == launch['run_id']}
    selected = [event for event in events if event['reservation'] in reservation_ids]
    (output / 'acquisition_ledger.jsonl').write_text(
        ''.join(json.dumps(event, sort_keys=True) + '\n' for event in selected))
    subprocess.run([
        sys.executable, str(Path(__file__).with_name('rescore_generations.py')),
        '--study', str(run), '--data', str(data), '--output', str(output / 'generation_diagnostics.json'),
    ], check=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    report = Path(__file__).resolve().parents[1] / 'ACQUISITION.md'
    if report.is_file():
        copy(report, 'ACQUISITION.md')
    (output / 'REPRODUCE.md').write_text(f'''# Weight-free acquisition evidence

Run: `{launch['run_id']}` at source commit `{launch['source_commit']}`.
This bundle contains no neural weights, packed arrays or raw text/code corpus. The 32 training examples and 32
separate development examples are executable synthetic programs. Checkpoint manifests identify, but do not contain,
the preserved local weights. Training-set exact answers measure memorization only, not generalization or superiority.

To independently execute all source programs and rescore every saved generation:

```bash
python research/ti_tnn/native/independent/rescore_generations.py \\
  --study . --data ./data --output ./generation_diagnostics.reproduced.json
```

`generation_diagnostics.json` is the preserved first rescore. Apart from the `study` path, the reproduced JSON must
match it. `summary.json` contains the compact reported tables; full per-example completions and answer losses are in
`arms/`. `acquisition_ledger.jsonl` contains all start, heartbeat and end events for this run's one reservation.
`architecture.json` distinguishes unique parameters and nominal decoded-token MAC rules from measured token throughput.
''')
    files = sorted(path for path in output.rglob('*') if path.is_file() and path.name != 'SHA256SUMS')
    (output / 'SHA256SUMS').write_text(''.join(f'{digest(path)}  {path.relative_to(output)}\n' for path in files))
    manifest = {
        'format': 'native-acquisition-weight-free-evidence-v1', 'run_id': launch['run_id'],
        'source_commit': launch['source_commit'], 'files': len(files) + 1,
        'bytes': sum(path.stat().st_size for path in files), 'contains_weights': False,
        'complete': status['complete'],
    }
    (output / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summary = export(args.run, args.data, args.output)
    print(json.dumps({'output': args.output, 'complete': summary['complete'],
                      'charged_hours': summary['charged_hours'],
                      'train_exact': {mode: arm.get('splits', {}).get('train', {}).get('generation', {}).get('correct')
                                      for mode, arm in summary['arms'].items()}}, sort_keys=True))


if __name__ == '__main__':
    main()
