"""Run the frozen 32-example acquisition protocol under one bounded reservation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import torch

from . import ledger
from . import reconstruct
from . import train
from .data import PackedDataset, sha256_file
from .evaluate_acquisition import evaluate
from .train import make_sampler

BUDGET_HOURS = 1.0
RESERVATION_PROJECTED_HOURS = 0.63
SHUTDOWN_ALLOWANCE_HOURS = 0.05


def _write(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    with temporary.open('rb') as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def _arguments(protocol, root, arm, mode, projected_hours):
    return [
        '--config', str(Path(train.__file__).resolve().parents[3] /
                        'research/ti_tnn/native/configs/engineering.json'),
        '--mode', mode, '--data', str(root), '--output', str(arm / 'train'), '--device', 'cuda',
        '--max-loss-tokens', str(protocol['max_loss_tokens']), '--batch-size', str(protocol['batch_size']),
        '--seed', str(protocol['seed']), '--model-seed', str(protocol['model_seed']),
        '--data-seed', str(protocol['data_seed']), '--learning-rate', str(protocol['learning_rate']),
        '--weight-decay', str(protocol['weight_decay']), '--betas', *map(str, protocol['betas']),
        '--eps', str(protocol['eps']), '--grad-clip', str(protocol['grad_clip']),
        '--lambda-balance', str(protocol['lambda_balance']), '--lambda-route', str(protocol['lambda_route']),
        '--lambda-state', str(protocol['lambda_state']), '--warmup-fraction', str(protocol['warmup_fraction']),
        '--final-lr-fraction', str(protocol['final_lr_fraction']),
        '--checkpoint-every', str(protocol['checkpoint_every']), '--dev-every', str(protocol['dev_every']),
        '--dev-batches', str(protocol['dev_batches']), '--projected-hours', str(projected_hours),
    ]


def _recover_statuses(run_dir, statuses):
    return {
        mode: json.loads((run_dir / 'arms' / mode / 'status.json').read_text())
        if (run_dir / 'arms' / mode / 'status.json').is_file() else values
        for mode, values in statuses.items()
    }


def _execute(protocol, root, run_dir, tokenizer, replay):
    statuses = {mode: {'status': 'not_run', 'required_updates': protocol['max_updates']}
                for mode in protocol['arms']}
    for mode in protocol['arms']:
        started = time.perf_counter()
        arm = run_dir / 'arms' / mode
        arm.mkdir(parents=True, exist_ok=False)
        statuses[mode]['status'] = 'training'
        _write(arm / 'status.json', statuses[mode])
        try:
            payload = train.run_training(train.parser().parse_args(
                _arguments(protocol, root, arm, mode, RESERVATION_PROJECTED_HOURS)))
            _write(arm / 'train-result.json', payload)
            checkpoint = payload['final_checkpoint']
            expected_path = (arm / 'train' / 'checkpoints').resolve()
            if (payload['format'] != train.TRAIN_FORMAT or payload['mode'] != mode
                    or payload['step'] != protocol['max_updates']
                    or payload['consumed_loss_tokens'] != protocol['max_loss_tokens']
                    or payload['planned_updates'] != protocol['max_updates']
                    or checkpoint['step'] != protocol['max_updates']
                    or Path(checkpoint['path']).resolve().parent != expected_path):
                raise RuntimeError(f'{mode}: final training endpoint or checkpoint identity does not match the protocol')
            statuses[mode]['status'] = 'evaluating'
            _write(arm / 'status.json', statuses[mode])
            evaluation = evaluate(checkpoint['path'], root, tokenizer, arm / 'evaluation.json',
                                  expected_sha256=checkpoint['sha256'],
                                  max_new_tokens=protocol['max_new_tokens'])
            for split, expected_examples in (('train', protocol['example_count']),
                                             ('dev', protocol['evaluation_examples_per_split'])):
                values = evaluation['splits'][split]
                if not values['generation']['complete'] or values['generation']['evaluated'] != expected_examples:
                    raise RuntimeError(f'{mode}: incomplete {split} generation')
                if not values['state_path']['complete']:
                    raise RuntimeError(f'{mode}: incomplete {split} state-path evaluation')
            statuses[mode]['status'] = 'reconstructing'
            _write(arm / 'status.json', statuses[mode])
            report = reconstruct.reconstruct_checkpoint(
                checkpoint['path'], reconstruct.load_replay_spec(replay),
                expected_sha256=checkpoint['sha256'], device='cpu')
            digest = reconstruct.write_reconstruction_report(arm / 'reconstruction.json', report)
            if not report['agreement']:
                raise RuntimeError(f'{mode}: FP32 full/cached reconstruction disagrees; see reconstruction.json')
            statuses[mode] = {
                'status': 'completed', 'updates': payload['step'],
                'consumed_loss_tokens': payload['consumed_loss_tokens'],
                'checkpoint': checkpoint, 'train_seconds': payload['training_seconds'],
                'wall_seconds': time.perf_counter() - started,
                'train_exact': evaluation['splits']['train']['generation']['correct'],
                'dev_exact': evaluation['splits']['dev']['generation']['correct'],
                'reconstruction_sha256': digest,
            }
        except Exception as error:
            statuses[mode] = {'status': 'failed', 'stage': statuses[mode]['status'],
                              'error': str(error), 'traceback': traceback.format_exc(),
                              'wall_seconds': time.perf_counter() - started}
        except BaseException as error:
            statuses[mode] = {'status': 'interrupted', 'stage': statuses[mode]['status'],
                              'error': str(error), 'kind': type(error).__name__,
                              'traceback': traceback.format_exc(),
                              'wall_seconds': time.perf_counter() - started}
            _write(arm / 'status.json', statuses[mode])
            raise
        finally:
            _write(arm / 'status.json', statuses[mode])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return statuses


def _reservation_projection(protocol, before):
    prior = protocol.get('acquisition_prior_reservation_hours', 0.0)
    if not 0 <= prior < BUDGET_HOURS:
        raise RuntimeError(f'invalid prior acquisition charge: {prior}')
    parent = protocol.get('parent_acquisition')
    if parent:
        parent_status_path = Path(parent['run_dir']) / 'run_status.json'
        if sha256_file(parent_status_path) != parent['run_status_sha256']:
            raise RuntimeError('the parent acquisition status no longer matches the corrective protocol')
        parent_protocol = Path(parent['run_dir']) / 'protocol.json'
        if sha256_file(parent_protocol) != parent['protocol_sha256']:
            raise RuntimeError('the parent training protocol no longer matches the corrective protocol')
        if (Path(parent['run_dir']) / 'protocol.sha256').read_text().strip() != parent['protocol_sha256']:
            raise RuntimeError('the parent protocol sidecar differs from its recorded identity')
        parent_status = json.loads(parent_status_path.read_text())
        if (parent_status['reservation_id'] != parent['reservation_id']
                or parent_status['charged_hours'] != prior):
            raise RuntimeError('parent acquisition identity or charge differs from the corrective protocol')
        reservations = {item['reservation']: item for item in before['reservations']}
        if (parent['reservation_id'] not in reservations
                or abs(reservations[parent['reservation_id']]['hours'] - prior) > 1e-9):
            raise RuntimeError('the live ledger does not reproduce the parent acquisition charge')
        for name, key in (('position_diagnostic', 'position_diagnostic_sha256'),
                          ('loss_gradient_diagnostic', 'loss_gradient_diagnostic_sha256')):
            if sha256_file(Path(parent['run_dir']) / f'{name}.json') != parent[key]:
                raise RuntimeError(f'parent {name} no longer matches the corrective protocol')
    if prior + protocol['projected_total_hours'] > 0.8 * BUDGET_HOURS:
        raise RuntimeError('combined acquisition projection does not preserve a 20% planning margin')
    available = BUDGET_HOURS - prior
    if before['native_remaining_hours'] < available:
        raise RuntimeError('the live native ledger cannot cover the remaining acquisition allowance')
    projected = min(RESERVATION_PROJECTED_HOURS,
                    (available - SHUTDOWN_ALLOWANCE_HOURS) / 1.5)
    if projected <= 0 or prior + projected * 1.5 + SHUTDOWN_ALLOWANCE_HOURS > BUDGET_HOURS:
        raise RuntimeError('the combined acquisition watchdog and shutdown allowance exceed one hour')
    return projected, prior


def run(data, repository, tokenizer, run_id=None):
    root = Path(data).resolve()
    repository = Path(repository).resolve()
    tokenizer = Path(tokenizer).resolve()
    replay = root.parent / 'ti_tnn_native' / 'replay.json'
    run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    run_dir = root / 'acquisition' / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    protocol_bytes = (root / 'acquisition_protocol.json').read_bytes()
    protocol = json.loads(protocol_bytes)
    provenance = root / 'source_provenance.json'
    for required in (root / 'manifest.json', root / 'acquisition_selection.json', provenance,
                     tokenizer, replay, repository / 'research/ti_tnn/native/configs/engineering.json'):
        if not required.is_file():
            raise FileNotFoundError(required)
    source_identity = json.loads(provenance.read_text())
    source_data = Path(source_identity['source_data']).resolve()
    if sha256_file(source_data / 'manifest.json') != source_identity['source_manifest_sha256']:
        raise RuntimeError('the preserved source manifest differs from acquisition provenance')
    for split, expected_hash in source_identity['structured_files'].items():
        if sha256_file(source_data / 'structured' / f'{split}.jsonl') != expected_hash:
            raise RuntimeError(f'the preserved source {split} data differs from acquisition provenance')
    if tokenizer != Path(source_identity['tokenizer_path']).resolve() or sha256_file(tokenizer) != source_identity['tokenizer_sha256']:
        raise RuntimeError('the acquisition tokenizer differs from frozen source provenance')
    if protocol['arms'] != ['dense', 'untyped', 'typed'] or protocol['projected_total_hours'] > 0.75:
        raise RuntimeError('the acquisition requires the frozen three arms and a total projection <=0.75 h')
    manifest = json.loads((root / 'manifest.json').read_text())
    for split in ('train', 'dev'):
        if manifest['splits'][split].get('position_policy', 'row') != protocol.get('position_policy', 'row'):
            raise RuntimeError(f'{split} packed positions differ from the acquisition protocol')
    sampler = make_sampler(PackedDataset(root, 'train'), protocol['batch_size'], protocol['data_seed'])
    planned_updates, planned_tokens = train._planned_budget(sampler, protocol['max_loss_tokens'])
    if (planned_updates, planned_tokens) != (protocol['max_updates'], protocol['max_loss_tokens']):
        raise RuntimeError('the frozen token budget does not yield the declared equal update endpoint')
    if Path(sys.executable).resolve() != Path(repository).parent.joinpath(
            'universa-recurrent-native/.venv-native/bin/python').resolve():
        raise RuntimeError(f'use the pinned native environment, not {sys.executable}')
    imported = Path(train.__file__).resolve()
    if imported != repository / 'research/ti_tnn/native/train.py':
        raise RuntimeError(f'PYTHONPATH selected {imported}, not the requested source worktree')
    source = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repository, capture_output=True,
                            text=True, check=True).stdout.strip()
    if subprocess.run(['git', 'status', '--porcelain'], cwd=repository, capture_output=True,
                      text=True, check=True).stdout:
        raise RuntimeError('the source-bound acquisition requires a clean, committed worktree')
    path, lock = ledger.validate_shared_reservation_paths()
    before = ledger.usage(path)
    reservation_hours, prior_hours = _reservation_projection(protocol, before)
    _write(run_dir / 'protocol.json', protocol)
    (run_dir / 'protocol.sha256').write_text(hashlib.sha256(protocol_bytes).hexdigest() + '\n')
    _write(run_dir / 'launch.json', {
        'format': 'native-acquisition-launch-v2', 'run_id': run_id, 'source_commit': source,
        'data': str(root), 'data_manifest_sha256': sha256_file(root / 'manifest.json'),
        'source_provenance_sha256': sha256_file(provenance), 'tokenizer': str(tokenizer),
        'tokenizer_sha256': sha256_file(tokenizer), 'replay': str(replay),
        'replay_sha256': sha256_file(replay), 'ledger_before': before,
        'prior_acquisition_hours': prior_hours,
        'combined_projected_hours': prior_hours + protocol['projected_total_hours'],
        'reservation_projected_hours': reservation_hours,
        'maximum_watchdog_hours': reservation_hours * 1.5,
        'shutdown_allowance_hours': SHUTDOWN_ALLOWANCE_HOURS,
    })
    statuses = {mode: {'status': 'not_run'} for mode in protocol['arms']}
    reservation_id = None
    failure = None
    started = time.perf_counter()
    old_run_id = os.environ.get('TI_TNN_NATIVE_ACQUISITION_RUN_ID')
    os.environ['TI_TNN_NATIVE_ACQUISITION_RUN_ID'] = run_id
    try:
        with ledger.Reservation(
                f'native.acquisition run={run_id}', projected_hours=reservation_hours,
                path=path, lock=lock,
                extra={'native_acquisition_run_id': run_id, 'source_commit': source,
                       'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
                       'prior_acquisition_hours': prior_hours}) as active:
            reservation_id = active.id
            statuses = _execute(protocol, root, run_dir, tokenizer, replay)
    except BaseException as error:
        failure = {'kind': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc()}
        statuses = _recover_statuses(run_dir, statuses)
    finally:
        if old_run_id is None:
            os.environ.pop('TI_TNN_NATIVE_ACQUISITION_RUN_ID', None)
        else:
            os.environ['TI_TNN_NATIVE_ACQUISITION_RUN_ID'] = old_run_id
    after = ledger.usage(path)
    complete = failure is None and set(statuses) == set(protocol['arms']) and all(
        item.get('status') == 'completed' and item.get('updates') == protocol['max_updates']
        for item in statuses.values())
    charged = after['used_hours'] - before['used_hours']
    cumulative_charged = prior_hours + charged
    result = {
        'format': 'native-acquisition-run-status-v2', 'run_id': run_id, 'source_commit': source,
        'complete': complete, 'arms': statuses, 'failure': failure, 'reservation_id': reservation_id,
        'wall_seconds': time.perf_counter() - started, 'charged_hours': charged,
        'prior_acquisition_hours': prior_hours,
        'cumulative_acquisition_charged_hours': cumulative_charged,
        'one_hour_limit_respected': cumulative_charged <= BUDGET_HOURS,
        'ledger_before': before, 'ledger_after': after,
        'evidence_boundary': '32-example acquisition is memorization only, not generalization or architectural superiority',
    }
    _write(run_dir / 'run_status.json', result)
    print(json.dumps({'run_dir': str(run_dir), 'complete': complete, 'charged_hours': charged,
                      'cumulative_acquisition_charged_hours': cumulative_charged,
                      'arms': statuses}, sort_keys=True, allow_nan=False))
    return 0 if complete else 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--python', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--run-id', help='new immutable attempt identifier; existing directories are refused')
    args = parser.parse_args()
    if Path(args.python).resolve() != Path(sys.executable).resolve():
        raise RuntimeError('--python must be the interpreter running this command')
    return run(args.data, args.repository, args.tokenizer, args.run_id)


if __name__ == '__main__':
    raise SystemExit(main())
