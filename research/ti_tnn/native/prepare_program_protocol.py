"""Freeze the varied-program protocol and its measured, unapproved allocation proposal."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from . import ledger
from .data import sha256_file
from .independent.rescore_generations import (
    PROGRAM_STUDY_SPLITS,
    execute_saved_program,
    source_records,
    summarize_program_study,
)
from .prepare_program_study import SPLIT_ROLES
from .run_program_study import (
    ARMS,
    EXPECTED_PROTOCOL,
    SHUTDOWN_ALLOWANCE_HOURS,
    SPLITS,
    WATCHDOG_MULTIPLIER,
)

EPOCHS = 16
EVALUATION_SAFETY_FACTOR = 1.5
RECONSTRUCTION_SECONDS_PER_ARM = 10.0
REFERENCE_RUN = Path('/home/seanjazm27/projects/universa-recurrent-scaling/runs/'
                     'ti_tnn_native_acquisition_local_positions/acquisition/20261110T091534-c25a2b1b')


def measure_scoring_rate(manifest, data_root):
    selections = manifest['selections']
    supplemental = set(selections['intervention_example_ids']['train']) - set(
        selections['evaluation_example_ids']['train'])
    selected = []
    for split in SPLITS:
        selected.extend(selections['evaluation_example_ids'][split])
    saved = [
        {
            'example_id': example_id,
            'completion_text': None,
            'target_text': None,
            'correct': True,
            'valid': True,
            'stopped_on_eos': True,
            'truncated': False,
            'prompt_truncated': False,
        }
        for example_id in selected + sorted(supplemental)
    ]
    sources = source_records(data_root, PROGRAM_STUDY_SPLITS)
    for record in saved:
        record['completion_text'] = record['target_text'] = sources[record['example_id']]['target_text']
    interventions = [
        {
            'example_id': example_id,
            'completion_text': sources[example_id]['target_text'],
            'target_text': sources[example_id]['target_text'],
            'correct': True,
            'valid': True,
            'stopped_on_eos': True,
            'truncated': False,
            'prompt_truncated': False,
        }
        for split in SPLITS for example_id in selections['intervention_example_ids'][split]
    ]
    started = time.perf_counter()
    for _ in ARMS:
        for source in sources.values():
            execute_saved_program(source)
        summarize_program_study(saved, sources)
        summarize_program_study(interventions, sources)
    measured = time.perf_counter() - started
    examples = len(saved) + len(interventions)
    rate = measured / (len(ARMS) * examples)
    return {
        'measured_seconds_per_example': rate,
        'budget_seconds_per_example': max(rate, 0.001),
        'scored_outputs_per_arm': examples,
        'supplemental_baseline_generations_per_arm': len(supplemental),
        'measurement': 'three arm-local independent target-execution and generation-summary passes; no neural inference',
    }


def compute_projection(manifest, reference_run, *, epochs=EPOCHS, before=None, scoring_measurement=None):
    if epochs != EPOCHS:
        raise ValueError(f'this frozen study requires exactly {EPOCHS} epochs')
    epoch = manifest['train_epoch']
    loss_tokens = epoch['loss_tokens_per_epoch'] * epochs
    updates = epoch['updates_per_epoch'] * epochs
    if (loss_tokens, updates) != (EXPECTED_PROTOCOL['max_loss_tokens'], EXPECTED_PROTOCOL['max_updates']):
        raise ValueError('the data and epoch count do not produce the frozen endpoint')
    measured = {}
    for mode in ARMS:
        arm = Path(reference_run) / 'arms' / mode
        result = json.loads((arm / 'train-result.json').read_text())
        status = json.loads((arm / 'status.json').read_text())
        evaluation = json.loads((arm / 'evaluation.json').read_text())
        examples = sum(values['generation']['evaluated'] for values in evaluation['splits'].values())
        evaluation_seconds = status['wall_seconds'] - result['training_seconds'] - result['checkpoint_seconds']
        measured[mode] = {
            'train_result_sha256': sha256_file(arm / 'train-result.json'),
            'status_sha256': sha256_file(arm / 'status.json'),
            'loss_tokens_per_second': result['actual_loss_tokens_per_second'],
            'checkpoint_seconds': result['checkpoint_seconds'],
            'evaluation_wall_seconds_including_replay': evaluation_seconds,
            'evaluation_examples': examples,
            'evaluation_seconds_per_example': evaluation_seconds / examples,
            'dev_forward_seconds_per_batch': result['dev']['forward_seconds'] / result['dev']['batches'],
        }
    minimum_throughput = min(values['loss_tokens_per_second'] for values in measured.values())
    evaluation_rate = max(values['evaluation_seconds_per_example'] for values in measured.values())
    budgeted_evaluation_rate = evaluation_rate * EVALUATION_SAFETY_FACTOR
    dev_evaluations = EXPECTED_PROTOCOL['max_updates'] // EXPECTED_PROTOCOL['dev_every'] + 1
    dev_batches = math.ceil(manifest['splits']['dev']['rows'] / EXPECTED_PROTOCOL['batch_size'])
    if scoring_measurement is None:
        scoring_measurement = measure_scoring_rate(manifest, Path(manifest['audit_file']).parent)
    per_arm = {
        'training': loss_tokens / minimum_throughput,
        'periodic_and_final_dev': (
            max(values['dev_forward_seconds_per_batch'] for values in measured.values())
            * dev_batches * dev_evaluations
        ),
        'checkpoint': max(values['checkpoint_seconds'] for values in measured.values()),
        'generation_answer_loss_and_state_evaluation': 5 * 128 * budgeted_evaluation_rate,
        'supplemental_intervention_baselines': (
            scoring_measurement['supplemental_baseline_generations_per_arm'] * budgeted_evaluation_rate),
        'targeted_interventions': 5 * 32 * budgeted_evaluation_rate,
        'independent_rescoring': (
            scoring_measurement['scored_outputs_per_arm'] * scoring_measurement['budget_seconds_per_example']),
        'fixed_checkpoint_reconstruction': RECONSTRUCTION_SECONDS_PER_ARM,
    }
    per_arm['total'] = sum(per_arm.values())
    total_seconds = per_arm['total'] * len(ARMS)
    projected_hours = total_seconds / 3600
    watchdog_hours = projected_hours * WATCHDOG_MULTIPLIER
    required_hours = watchdog_hours + SHUTDOWN_ALLOWANCE_HOURS
    allocation_hours = math.ceil(max(projected_hours / 0.8, required_hours) * 4) / 4
    return {
        'epochs_per_arm': epochs,
        'per_arm_seconds': per_arm,
        'total_seconds_all_arms': total_seconds,
        'projected_total_hours': projected_hours,
        'maximum_watchdog_hours': watchdog_hours,
        'shutdown_allowance_hours': SHUTDOWN_ALLOWANCE_HOURS,
        'required_hours_with_shutdown': required_hours,
        'allocation_request_hours': allocation_hours,
        'planning_margin_hours': allocation_hours - projected_hours,
        'planning_margin_fraction': (allocation_hours - projected_hours) / allocation_hours,
        'allocation_headroom_hours': allocation_hours - required_hours,
        'live_native_remaining_hours_at_proposal': None if before is None else before['native_remaining_hours'],
        'fits_live_native_ledger': None if before is None else allocation_hours <= before['native_remaining_hours'],
        'measurement_inputs': {
            'reference_run': str(Path(reference_run).resolve()),
            'slowest_measured_loss_tokens_per_second': minimum_throughput,
            'slowest_measured_evaluation_seconds_per_example': evaluation_rate,
            'evaluation_safety_factor': EVALUATION_SAFETY_FACTOR,
            'budgeted_evaluation_seconds_per_example': budgeted_evaluation_rate,
            'evaluation_examples_per_arm': 5 * 128,
            'intervention_examples_per_arm': 5 * 32,
            'dev_evaluations_per_arm': dev_evaluations,
            'dev_batches_per_evaluation': dev_batches,
            'fixed_reconstruction_seconds_per_arm': RECONSTRUCTION_SECONDS_PER_ARM,
            'independent_scoring': scoring_measurement,
            'arms': measured,
        },
    }


def source_identities(repository):
    root = Path(repository) / 'research/ti_tnn/native'
    names = (
        'program_study.py', 'prepare_program_study.py', 'evaluate_program_study.py',
        'evaluate_state_path.py', 'diagnose_learned_state.py', 'run_program_study.py',
        'prepare_program_protocol.py', 'train.py', 'model.py', 'data.py',
        'independent/rescore_generations.py', 'configs/engineering.json',
    )
    return {name: sha256_file(root / name) for name in names}


def prepare(data, repository, tokenizer, reference_run=REFERENCE_RUN, before=None, output=None):
    root = Path(data).resolve()
    repository = Path(repository).resolve()
    tokenizer = Path(tokenizer).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['position_policy'] != 'document':
        raise ValueError('the future study requires document-local positions')
    before = ledger.usage(ledger.validate_shared_reservation_paths()[0]) if before is None else before
    projection = compute_projection(manifest, reference_run, before=before)
    if not projection['fits_live_native_ledger']:
        raise RuntimeError('the proposed allocation does not fit the live native ledger')
    protocol = {
        'format': 'native-program-study-protocol-v1',
        'status': 'proposed_not_authorized_or_launched',
        **EXPECTED_PROTOCOL,
        'epochs': EPOCHS,
        'position_policy': 'document',
        'packing': 'intact',
        'structured_overflow': 'error',
        'objective': 'existing LM + load balance + route + state objective; no answer-focused objective or architecture change',
        'state_schedule': 'before,after,after,after',
        'splits': list(SPLITS),
        'split_roles': SPLIT_ROLES,
        'data': {
            'root': str(root), 'manifest_sha256': sha256_file(root / 'manifest.json'),
            'audit_sha256': sha256_file(root / 'audit.json'),
            'selections_sha256': sha256_file(root / 'selections.json'),
            'train_examples': 4096, 'evaluation_examples_per_split': 128,
            'intervention_examples_per_split': 32,
            'tokenizer': manifest['tokenizer'],
            'tokenizer_path': str(tokenizer),
            'tokenizer_file_sha256': sha256_file(tokenizer),
            'source_identity': manifest['source_identity'],
        },
        'evaluation': {
            'exact_answers': 'complete greedy target text after the complete prompt; no target or semantic labels at inference',
            'answer_token_cross_entropy': 'teacher-forced CE over all answer tokens and EOS, separately from register error',
            'per_operation': 'exact and canonical-format accuracy grouped by final instruction operation; 32 per family per split',
            'per_register': 'accuracy and mean absolute error over canonical format-valid completions, with coverage counts',
            'state_readout': 'all packed rows per split; generation and answer CE use the separately selected 128 per split',
            'interventions': 'pass-two final-instruction writer-update suppression on 32 selected examples per split',
            'supplemental_baselines': '30 additional ordinary train generations outside the primary 128; each arm-local scorer covers them',
            'independent_rescoring': '830 primary/supplemental/intervened outputs per arm; prerequisites, changed counts and per-operation metrics validated',
            'intervention_limits': 'an effect is not semantic faithfulness; no effect can be redundancy and is not state absence',
            'baselines': 'full-train exact lookup and supplied-program executor; the executor is not the neural model',
            'reconstruction': 'unautocast CPU FP32 full/cached replay with exact integer routes and fixed replay tokens',
        },
        'projection': projection,
        'projected_total_hours': projection['projected_total_hours'],
        'allocation_request_hours': projection['allocation_request_hours'],
        'allocation_status': 'proposal only; no new authorization, GPU reservation, training, or WSP is active',
        'execution_gate': 'require --execute and --authorize-hours exactly equal to allocation_request_hours after owner approval',
        'source_files_sha256': source_identities(repository),
        'completion_requirements': [
            'all three arms reach the same declared update and loss-token endpoint',
            'all five 128-example evaluations and 32-example intervention sets complete per arm',
            'all supplemental baseline and intervened outputs are independently rescored with complete prerequisites',
            'independent changed-completion and generated-suffix counts agree with the saved interventions',
            'checkpoint, optimizer, scheduler, RNG and sampler state remain resumable',
            'fixed evaluation replay and independent generation rescoring complete without truncating a control',
        ],
        'prior_pipeline_check': 'the completed 32-example acquisition is not this training population',
        'no_qwen_no_124M_no_WSP': True,
    }
    output = Path(output) if output is not None else root / 'study_protocol.json'
    encoded = (json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    with output.open('xb') as handle:
        handle.write(encoded)
    output.with_suffix('.sha256').write_text(hashlib.sha256(encoded).hexdigest() + '\n')
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--repository', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--reference-run', default=str(REFERENCE_RUN))
    parser.add_argument('--output')
    args = parser.parse_args()
    protocol = prepare(args.data, args.repository, args.tokenizer, args.reference_run, output=args.output)
    print(json.dumps({'status': protocol['status'], 'max_updates': protocol['max_updates'],
                      'max_loss_tokens': protocol['max_loss_tokens'],
                      'projected_total_hours': protocol['projected_total_hours'],
                      'allocation_request_hours': protocol['allocation_request_hours'],
                      'output': args.output or str(Path(args.data) / 'study_protocol.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
