"""Run the frozen 32-example acquisition protocol with full per-arm accounting."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from . import ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--tokenizer', required=True)
    args = parser.parse_args()
    root = Path(args.data)
    protocol = json.loads((root / 'acquisition_protocol.json').read_text())
    remaining = ledger.usage()['native_remaining_hours']
    if remaining < 1.0 or protocol['projected_total_hours'] > min(1.0, remaining):
        raise RuntimeError(f'acquisition projection does not fit one hour: remaining={remaining}')
    python = args.python
    (root / 'acquisition').mkdir(exist_ok=True)
    statuses = {}
    for mode in protocol['arms']:
        arm = root / 'arms' / mode
        arm.mkdir(parents=True, exist_ok=True)
        train = [
            python, '-m', 'native.train', '--config', 'research/ti_tnn/native/configs/engineering.json',
            '--mode', mode, '--data', str(root), '--output', str(arm / 'train'), '--device', 'cuda',
            '--max-loss-tokens', str(protocol['max_loss_tokens']), '--batch-size', '8',
            '--seed', str(protocol['seed']), '--model-seed', str(protocol['model_seed']),
            '--data-seed', str(protocol['data_seed']), '--learning-rate', str(protocol['learning_rate']),
            '--weight-decay', str(protocol['weight_decay']), '--betas', *map(str, protocol['betas']),
            '--eps', str(protocol['eps']), '--grad-clip', str(protocol['grad_clip']),
            '--lambda-balance', str(protocol['lambda_balance']), '--lambda-route', str(protocol['lambda_route']),
            '--lambda-state', str(protocol['lambda_state']), '--warmup-fraction', str(protocol['warmup_fraction']),
            '--final-lr-fraction', str(protocol['final_lr_fraction']),
            '--checkpoint-every', str(protocol['checkpoint_every']), '--dev-every', str(protocol['dev_every']),
            '--dev-batches', str(protocol['dev_batches']), '--projected-hours', str(protocol['projected_hours_per_train']),
        ]
        started = time.time()
        result = subprocess.run(train, cwd=args.repository, capture_output=True, text=True, check=False)
        (arm / 'train.stdout.log').write_text(result.stdout)
        (arm / 'train.stderr.log').write_text(result.stderr)
        if result.returncode:
            statuses[mode] = {'status': 'failed', 'stage': 'train', 'error': result.stderr}
            (arm / 'status.json').write_text(json.dumps(statuses[mode], indent=2) + '\n')
            continue
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        (arm / 'train-result.json').write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        if payload['step'] != protocol['max_updates']:
            statuses[mode] = {'status': 'failed', 'stage': 'train', 'error': f"expected {protocol['max_updates']} updates, got {payload['step']}"}
            (arm / 'status.json').write_text(json.dumps(statuses[mode], indent=2) + '\n')
            continue
        checkpoint = payload['final_checkpoint']
        path, lock = ledger.validate_shared_reservation_paths()
        with ledger.Reservation(f'native.acquisition evaluate {mode}', projected_hours=protocol['projected_hours_per_evaluation'],
                                path=path, lock=lock, extra={'native_acquisition_arm': mode, 'native_acquisition_stage': 'evaluate'}):
            for module, output in (('native.evaluate_acquisition', arm / 'evaluation.json'),
                                   ('native.reconstruct', arm / 'reconstruction.json')):
                command = [python, '-m', module, '--checkpoint', checkpoint['path'], '--expected-sha256', checkpoint['sha256']]
                if module == 'native.evaluate_acquisition':
                    command += ['--data', str(root), '--tokenizer', args.tokenizer, '--output', str(output),
                                '--max-new-tokens', str(protocol['max_new_tokens'])]
                else:
                    command += ['--spec', str(root.parent / 'ti_tnn_native/replay.json'), '--output', str(output), '--device', 'cpu']
                result = subprocess.run(command, cwd=args.repository, capture_output=True, text=True, check=False)
                (arm / f'{module.rsplit(".", 1)[-1]}.stdout.log').write_text(result.stdout)
                (arm / f'{module.rsplit(".", 1)[-1]}.stderr.log').write_text(result.stderr)
                if result.returncode:
                    statuses[mode] = {'status': 'failed', 'stage': module, 'error': result.stderr}
                    (arm / 'status.json').write_text(json.dumps(statuses[mode], indent=2) + '\n')
                    break
            else:
                statuses[mode] = {'status': 'completed', 'checkpoint': checkpoint,
                                  'train_seconds': time.time() - started}
                (arm / 'status.json').write_text(json.dumps(statuses[mode], indent=2, sort_keys=True) + '\n')
                continue
            break
    (root / 'run_status.json').write_text(json.dumps({'format': 'native-acquisition-run-status-v1', 'arms': statuses}, indent=2) + '\n')
    print(json.dumps(statuses, sort_keys=True))
    return 0 if all(value.get('status') == 'completed' for value in statuses.values()) else 3


if __name__ == '__main__':
    main()
