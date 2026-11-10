"""Suppress one supervised pass-two after-state update during cached CPU generation."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch

from . import evaluate as E
from .data import EOS_TOKEN_ID, load_gpt2_tokenizer, sha256_file
from .model import NativeModel
from .structured import Instruction, encode_with_offsets


FORMAT = 'native-acquisition-learned-state-intervention-v1'
ARMS = ('dense', 'untyped', 'typed')
PASS_INDEX = 1
STATE_SCHEDULE = ('before', 'after', 'after', 'after')


@dataclass(frozen=True)
class InstructionAnchor:
    example_id: str
    instruction_index: int
    instruction: str
    token_index: int
    token_id: int
    token_text: str
    instruction_char_start: int
    instruction_char_end: int
    token_char_start: int
    token_char_end: int

    def metadata(self) -> dict:
        return {
            'instruction_index': self.instruction_index,
            'instruction': self.instruction,
            'token_index': self.token_index,
            'token_id': self.token_id,
            'token_text': self.token_text,
            'instruction_char_start': self.instruction_char_start,
            'instruction_char_end': self.instruction_char_end,
            'token_char_start': self.token_char_start,
            'token_char_end': self.token_char_end,
            'selection_rule': 'last prompt token overlapping the final canonical instruction line',
        }


def final_instruction_anchor(record: Mapping, tokenizer) -> InstructionAnchor:
    program = record.get('program')
    prompt = record.get('prompt')
    example_id = record.get('example_id')
    if not isinstance(program, (list, tuple)) or not program or not isinstance(prompt, str):
        raise ValueError(f'{example_id}: a nonempty saved program and prompt are required')
    instructions = [Instruction.parse(line) for line in program]
    if any(str(instruction) != line for instruction, line in zip(instructions, program)):
        raise ValueError(f'{example_id}: program instructions must use canonical text')
    prompt_ids, offsets = encode_with_offsets(tokenizer, prompt)
    cursor = prompt.find('Program:\n')
    if cursor < 0:
        raise ValueError(f'{example_id}: prompt has no Program section')
    cursor += len('Program:\n')
    final = None
    for instruction_index, line in enumerate(program):
        start = prompt.find(line + '\n', cursor)
        if start < 0:
            raise ValueError(f'{example_id}: cannot locate instruction line {line!r}')
        end = start + len(line)
        cursor = end + 1
        final = instruction_index, line, start, end
    instruction_index, line, start, end = final
    overlaps = [
        index for index, (left, right) in enumerate(offsets)
        if right > left and left < end and right > start
    ]
    if not overlaps:
        raise ValueError(f'{example_id}: final instruction has no overlapping prompt token')
    token_index = overlaps[-1]
    token_start, token_end = offsets[token_index]
    return InstructionAnchor(
        example_id=example_id,
        instruction_index=instruction_index,
        instruction=line,
        token_index=token_index,
        token_id=int(prompt_ids[token_index]),
        token_text=E._decode(tokenizer, [prompt_ids[token_index]]),
        instruction_char_start=start,
        instruction_char_end=end,
        token_char_start=token_start,
        token_char_end=token_end,
    )


@contextmanager
def suppress_after_state_update(model: NativeModel, token_index: int):
    """Zero all four slot-writer outputs at one absolute token in pass two only.

    The model forward pre-hook converts the absolute prompt token to a local token
    using cache length. Writer hooks see only the selected output vectors; generated
    suffix tokens have different absolute indices and receive no direct intervention.
    """
    if isinstance(token_index, bool) or not isinstance(token_index, int) or token_index < 0:
        raise ValueError('token_index must be a nonnegative integer')
    original_write = model._write_slots
    previous_write = model.__dict__.get('_write_slots')
    writers = list(model.slot_writers) if model.config.mode == 'typed' else [model.slot_writer]
    state = {
        'forwards': 0,
        'completed_forwards': 0,
        'targeted_forwards': 0,
        'suppressed_writer_updates': 0,
        'next_pass': 0,
        'current_pass': None,
        'writer_in_pass': 0,
        'local_index': None,
        'suppressed_this_forward': 0,
    }
    handles = []

    def forward_start(module, args, kwargs):
        input_ids = kwargs.get('input_ids', args[0] if args else None)
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError('learned-state suppression requires one input row')
        cache = kwargs.get('cache', args[4] if len(args) > 4 else None)
        past_length = cache.length if cache is not None else 0
        local_index = token_index - past_length
        if not 0 <= local_index < input_ids.shape[1]:
            local_index = None
        state.update({
            'forwards': state['forwards'] + 1,
            'next_pass': 0,
            'current_pass': None,
            'writer_in_pass': 0,
            'local_index': local_index,
            'suppressed_this_forward': 0,
        })

    def writer_output(module, inputs, output):
        pass_index = state['current_pass']
        if pass_index is None or output.ndim != 3 or output.shape[0] != 1:
            raise RuntimeError('slot writer fired outside the coordinated four-pass forward')
        local_index = state['local_index']
        if pass_index == PASS_INDEX and local_index is not None:
            if output.shape[1] <= local_index:
                raise RuntimeError('selected writer output does not contain the target token')
            output = output.clone()
            output[:, local_index, :] = 0
            state['suppressed_writer_updates'] += 1
            state['suppressed_this_forward'] += 1
        state['writer_in_pass'] += 1
        if state['writer_in_pass'] > 4:
            raise RuntimeError('more than four slot writers fired in one pass')
        return output

    def coordinated_write(slots, transformed, projected):
        pass_index = state['next_pass']
        if not 0 <= pass_index < 4:
            raise RuntimeError('slot write schedule is not before,after,after,after')
        state['current_pass'] = pass_index
        state['writer_in_pass'] = 0
        result = original_write(slots, transformed, projected)
        if state['writer_in_pass'] != 4:
            raise RuntimeError('each pass must invoke exactly four slot writers')
        state['next_pass'] += 1
        state['current_pass'] = None
        return result

    def forward_end(module, args, output):
        if state['next_pass'] != 4 or state['current_pass'] is not None:
            raise RuntimeError('model forward did not complete all four slot writes')
        selected = state['local_index'] is not None
        expected = 4 if selected else 0
        if state['suppressed_this_forward'] != expected:
            raise RuntimeError('writer suppression did not match the selected pass/token contract')
        state['targeted_forwards'] += int(selected)
        state['completed_forwards'] += 1

    handles.append(model.register_forward_pre_hook(forward_start, with_kwargs=True))
    for writer in writers:
        handles.append(writer.register_forward_hook(writer_output))
    handles.append(model.register_forward_hook(forward_end))
    model._write_slots = coordinated_write
    try:
        yield state
    finally:
        for handle in handles:
            handle.remove()
        if previous_write is None:
            model.__dict__.pop('_write_slots', None)
        else:
            model._write_slots = previous_write


def _measure_generation(generated: Mapping, target_text: str) -> dict:
    result = dict(generated)
    parsed_prediction = E.parse_final_registers(result['completion_text'])
    parsed_target = E.parse_final_registers(target_text)
    if parsed_target is None:
        raise ValueError('gold target is not canonical final-register text')
    result['valid'] = parsed_prediction is not None
    result['correct'] = (
        parsed_prediction is not None
        and parsed_prediction == parsed_target
        and result['completion_text'] == target_text
    )
    return result


def _saved_comparison(generated: Mapping, saved: Mapping) -> dict:
    fields = (
        'completion_text', 'completion_tokens', 'correct', 'valid', 'stopped_on_eos',
        'truncated', 'original_prompt_tokens', 'prompt_truncated',
    )
    actual = dict(generated)
    actual['completion_tokens'] = len(generated['completion_ids'])
    matches = {field: actual[field] == saved[field] for field in fields}
    return {'all_match': all(matches.values()), 'field_matches': matches}


def _snapshot(model: NativeModel) -> tuple[dict, dict]:
    weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    return weights, gradients


def _unchanged_snapshot(model: NativeModel, before: tuple[dict, dict]) -> dict:
    weights, gradients = before
    current_weights = model.state_dict()
    weights_unchanged = current_weights.keys() == weights.keys() and all(
        torch.equal(current_weights[name].detach().cpu(), value)
        for name, value in weights.items()
    )
    current_gradients = dict(model.named_parameters())
    gradients_unchanged = current_gradients.keys() == gradients.keys() and all(
        (current_gradients[name].grad is None and value is None)
        or (
            current_gradients[name].grad is not None
            and value is not None
            and torch.equal(current_gradients[name].grad.detach().cpu(), value)
        )
        for name, value in gradients.items()
    )
    result = {
        'state_dict_entries': len(weights),
        'state_dict_exactly_unchanged': weights_unchanged,
        'grad_fields_exactly_unchanged': gradients_unchanged,
        'grad_fields_none_before': all(value is None for value in gradients.values()),
        'grad_fields_none_after': all(parameter.grad is None for parameter in model.parameters()),
    }
    if not weights_unchanged or not gradients_unchanged:
        raise RuntimeError('diagnostic changed model weights or parameter .grad fields')
    return result


def _condition_counts(examples: list[dict], condition: str) -> dict:
    generations = [example[condition] for example in examples]
    count = len(generations)
    return {
        'examples': count,
        'exact': sum(item['correct'] for item in generations),
        'exact_match': sum(item['correct'] for item in generations) / count,
        'format_valid': sum(item['valid'] for item in generations),
        'format_validity': sum(item['valid'] for item in generations) / count,
        'stopped_on_eos': sum(item['stopped_on_eos'] for item in generations),
        'eos_rate': sum(item['stopped_on_eos'] for item in generations) / count,
        'truncated': sum(item['truncated'] for item in generations),
        'truncation_rate': sum(item['truncated'] for item in generations) / count,
    }


def _arm_counts(examples: list[dict], saved_dev_exact: int) -> dict:
    return {
        'baseline': _condition_counts(examples, 'baseline'),
        'intervened': _condition_counts(examples, 'intervened'),
        'changed_completions': sum(example['changed_completion'] for example in examples),
        'changed_generated_suffixes': sum(example['changed_generated_suffix'] for example in examples),
        'saved_baseline_completions_reproduced': sum(
            example['saved_evaluation_comparison']['all_match'] for example in examples
        ),
        'targeted_exactly_once': sum(
            example['intervention']['targeted_forwards'] == 1 for example in examples
        ),
        'four_writer_updates_suppressed_per_example': sum(
            example['intervention']['suppressed_writer_updates'] == 4 for example in examples
        ),
        'saved_dev_exact_memorization_excluded': saved_dev_exact,
    }


def _write_new_json(output: Path, result: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def diagnose(run_directory, data_root, tokenizer_path=None, output=None) -> dict:
    run_directory = Path(run_directory).resolve()
    data_root = Path(data_root).resolve()
    output = run_directory / 'learned_state_intervention.json' if output is None else Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite an existing diagnostic: {output}')
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only diagnostic requires CUDA_VISIBLE_DEVICES=''")
    launch = json.loads((run_directory / 'launch.json').read_text())
    protocol = json.loads((run_directory / 'protocol.json').read_text())
    status = json.loads((run_directory / 'run_status.json').read_text())
    if not status.get('complete') or status.get('failure') is not None:
        raise ValueError('the acquisition run is not complete')
    if protocol.get('format') != 'native-acquisition-protocol-v2-document-positions':
        raise ValueError('learned-state diagnosis requires the corrected document-position acquisition')
    if protocol['arms'] != list(ARMS) or set(status['arms']) != set(ARMS):
        raise ValueError('the diagnostic requires dense, untyped, and typed arms')
    if Path(launch['data']).resolve() != data_root:
        raise ValueError('data root differs from acquisition launch provenance')
    if sha256_file(data_root / 'manifest.json') != launch['data_manifest_sha256']:
        raise ValueError('data manifest differs from acquisition launch provenance')
    if hashlib.sha256((data_root / 'acquisition_protocol.json').read_bytes()).hexdigest() != (
        run_directory / 'protocol.sha256'
    ).read_text().strip():
        raise ValueError('acquisition protocol hash differs from the run')
    if protocol != json.loads((data_root / 'acquisition_protocol.json').read_text()):
        raise ValueError('run protocol and frozen acquisition protocol differ')
    tokenizer_path = Path(tokenizer_path or launch['tokenizer']).resolve()
    if tokenizer_path != Path(launch['tokenizer']).resolve():
        raise ValueError('tokenizer differs from acquisition launch provenance')
    if sha256_file(tokenizer_path) != launch['tokenizer_sha256']:
        raise ValueError('tokenizer bytes differ from acquisition launch provenance')
    tokenizer = load_gpt2_tokenizer(tokenizer_path)
    records = E._load_jsonl(data_root / 'structured/train.jsonl')
    if len(records) != 32 or protocol['example_count'] != len(records):
        raise ValueError('the diagnostic requires exactly the frozen 32 train examples')
    anchors = {record['example_id']: final_instruction_anchor(record, tokenizer) for record in records}
    result = {
        'format': FORMAT,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'run_directory': str(run_directory),
        'data_root': str(data_root),
        'source_commit': launch['source_commit'],
        'data_manifest_sha256': launch['data_manifest_sha256'],
        'tokenizer': {'path': str(tokenizer_path), 'sha256': launch['tokenizer_sha256']},
        'device': 'cpu',
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_version': torch.__version__,
        'generation': {
            'position_policy': 'zero-based prompt and continuing generated positions',
            'max_new_tokens': protocol['max_new_tokens'],
            'use_cache': True,
            'greedy': True,
            'gold_targets_used_for_decoding': False,
        },
        'intervention': {
            'state_schedule': ','.join(STATE_SCHEDULE),
            'target_state': STATE_SCHEDULE[PASS_INDEX],
            'pass_index_zero_based': PASS_INDEX,
            'pass_number_one_based': PASS_INDEX + 1,
            'operation': 'zero every slot-writer output vector at the final-instruction anchor token',
            'slot_writers_per_target': 4,
            'incoming_slot_state_preserved': True,
            'other_pass_writer_outputs_directly_modified': False,
            'input_tokens_labels_or_positions_modified': False,
            'suffix_policy': 'no direct suppression; suffix computation changes only through the resulting cache',
        },
        'negative_history': {
            'parent_run_directory': protocol['parent_acquisition']['run_dir'],
            'parent_result_preserved': '11/32 zero-based train exact per arm; position mismatch diagnosis remains negative history',
            'corrected_train_scope': '32-example memorization only, not generalization',
            'separate_dev_results_preserved': True,
        },
        'limits': [
            'An output effect establishes causal dependence on this suppressed update under this local counterfactual, not semantic faithfulness or answer encoding.',
            'No output effect may reflect redundancy or other paths and does not prove that learned state is absent.',
            'One token, one supervised after-state update, and 32 memorized train prompts do not exhaust other tokens, passes, examples, or state uses.',
            'This diagnostic does not rewrite the earlier negative acquisition or establish held-out generalization.',
        ],
        'arms': {},
    }
    for arm_name in ARMS:
        arm_status = status['arms'][arm_name]
        checkpoint = arm_status['checkpoint']
        if arm_status.get('status') != 'completed' or arm_status.get('updates') != protocol['max_updates']:
            raise ValueError(f'{arm_name}: arm did not complete the required updates')
        if checkpoint['step'] != protocol['max_updates']:
            raise ValueError(f'{arm_name}: checkpoint step differs from the protocol')
        evaluation_path = run_directory / 'arms' / arm_name / 'evaluation.json'
        evaluation = json.loads(evaluation_path.read_text())
        saved_generation = evaluation['splits']['train']['generation']
        if (not saved_generation['complete'] or saved_generation['evaluated'] != len(records)
                or saved_generation['max_new_tokens'] != protocol['max_new_tokens']):
            raise ValueError(f'{arm_name}: saved baseline generation is incomplete or uses another bound')
        saved_by_id = {record['example_id']: record for record in saved_generation['records']}
        if set(saved_by_id) != {record['example_id'] for record in records}:
            raise ValueError(f'{arm_name}: saved generation identities differ from train records')
        model, descriptor = E.load_native_model(
            checkpoint['path'], expected_sha256=checkpoint['sha256'], device='cpu',
        )
        if next(model.parameters()).device.type != 'cpu' or model.training or model.config.mode != arm_name:
            raise RuntimeError(f'{arm_name}: expected an eval-mode CPU checkpoint for the matching arm')
        before = _snapshot(model)
        examples = []
        for record in records:
            example_id = record['example_id']
            anchor = anchors[example_id]
            baseline = _measure_generation(E.generate_prompt(
                model,
                tokenizer,
                record['prompt'],
                max_new_tokens=protocol['max_new_tokens'],
                eos_token_id=EOS_TOKEN_ID,
                use_cache=True,
                prompt_policy='error',
            ), record['target_text'])
            comparison = _saved_comparison(baseline, saved_by_id[example_id])
            if not comparison['all_match']:
                raise RuntimeError(f'{arm_name}/{example_id}: ordinary generation differs from saved evaluation')
            with suppress_after_state_update(model, anchor.token_index) as intervention:
                intervened = _measure_generation(E.generate_prompt(
                    model,
                    tokenizer,
                    record['prompt'],
                    max_new_tokens=protocol['max_new_tokens'],
                    eos_token_id=EOS_TOKEN_ID,
                    use_cache=True,
                    prompt_policy='error',
                ), record['target_text'])
            intervention = dict(intervention)
            if intervention['completed_forwards'] != intervention['forwards']:
                raise RuntimeError(f'{arm_name}/{example_id}: incomplete instrumented forward')
            if intervention['targeted_forwards'] != 1 or intervention['suppressed_writer_updates'] != 4:
                raise RuntimeError(f'{arm_name}/{example_id}: final instruction was not targeted exactly once')
            examples.append({
                'example_id': example_id,
                'target_text': record['target_text'],
                'final_instruction': anchor.metadata(),
                'baseline': baseline,
                'saved_evaluation_comparison': comparison,
                'intervention': intervention,
                'intervened': intervened,
                'changed_completion': baseline['completion_ids'] != intervened['completion_ids'],
                'changed_generated_suffix': baseline['generated_ids'] != intervened['generated_ids'],
                'first_generated_token_divergence': E._first_token_divergence(
                    intervened['generated_ids'], baseline['generated_ids'],
                ),
                'first_completion_text_divergence': E._first_text_divergence(
                    intervened['completion_text'], baseline['completion_text'],
                ),
            })
        invariants = _unchanged_snapshot(model, before)
        checkpoint_sha256_after = sha256_file(checkpoint['path'])
        if checkpoint_sha256_after != checkpoint['sha256']:
            raise RuntimeError(f'{arm_name}: checkpoint bytes changed during diagnosis')
        result['arms'][arm_name] = {
            'checkpoint': {
                'path': checkpoint['path'],
                'sha256': checkpoint['sha256'],
                'sha256_after': checkpoint_sha256_after,
                'file_unchanged': True,
                'step': descriptor['step'],
                'config': descriptor['config'],
            },
            'saved_evaluation': {'path': str(evaluation_path), 'sha256': sha256_file(evaluation_path)},
            'counts': _arm_counts(
                examples, evaluation['splits']['dev']['generation']['correct'],
            ),
            'invariants': invariants,
            'examples': examples,
        }
        print(json.dumps({'arm': arm_name, **result['arms'][arm_name]['counts']}, sort_keys=True), flush=True)
        del model, descriptor, before
        gc.collect()
    result['summary'] = {
        'arms': len(ARMS),
        'examples_per_arm': len(records),
        'baseline_exact_by_arm': {
            arm: result['arms'][arm]['counts']['baseline']['exact'] for arm in ARMS
        },
        'intervened_exact_by_arm': {
            arm: result['arms'][arm]['counts']['intervened']['exact'] for arm in ARMS
        },
        'changed_completions_by_arm': {
            arm: result['arms'][arm]['counts']['changed_completions'] for arm in ARMS
        },
        'intervened_format_valid_by_arm': {
            arm: result['arms'][arm]['counts']['intervened']['format_valid'] for arm in ARMS
        },
        'intervened_stopped_on_eos_by_arm': {
            arm: result['arms'][arm]['counts']['intervened']['stopped_on_eos'] for arm in ARMS
        },
        'intervened_truncated_by_arm': {
            arm: result['arms'][arm]['counts']['intervened']['truncated'] for arm in ARMS
        },
        'saved_dev_exact_by_arm': {
            arm: result['arms'][arm]['counts']['saved_dev_exact_memorization_excluded'] for arm in ARMS
        },
        'baseline_saved_completions_reproduced_in_every_arm': all(
            result['arms'][arm]['counts']['saved_baseline_completions_reproduced'] == len(records)
            for arm in ARMS
        ),
        'weights_grad_fields_and_checkpoints_unchanged_in_every_arm': all(
            result['arms'][arm]['invariants']['state_dict_exactly_unchanged']
            and result['arms'][arm]['invariants']['grad_fields_exactly_unchanged']
            and result['arms'][arm]['checkpoint']['file_unchanged']
            for arm in ARMS
        ),
        'all_interventions_targeted_exactly_once': all(
            result['arms'][arm]['counts']['targeted_exactly_once'] == len(records)
            and result['arms'][arm]['counts']['four_writer_updates_suppressed_per_example'] == len(records)
            for arm in ARMS
        ),
    }
    _write_new_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='completed corrected acquisition run directory')
    parser.add_argument('--data', required=True, help='frozen corrected acquisition data root')
    parser.add_argument('--tokenizer', help='must match acquisition launch provenance')
    parser.add_argument('--output', help='new JSON path; defaults to RUN/learned_state_intervention.json')
    args = parser.parse_args()
    output = Path(args.output).resolve() if args.output else Path(args.run).resolve() / 'learned_state_intervention.json'
    result = diagnose(args.run, args.data, args.tokenizer, output)
    print(json.dumps({'output': str(output), **result['summary']}, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
