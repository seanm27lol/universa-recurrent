"""Replay the completed acquisition with zero-based or trained packed positions."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from . import evaluate as E
from .data import EOS_TOKEN_ID, IGNORE_INDEX, PackedDataset, SOURCE_IDS, load_gpt2_tokenizer, sha256_file
from .generation import greedy_generate
from .structured import encode_with_offsets


FORMAT = 'native-acquisition-position-diagnostic-v1'
ARMS = ('dense', 'untyped', 'typed')


@dataclass(frozen=True)
class PackedAlignment:
    doc_id: int
    row: int
    column: int
    start_position: int
    prompt_ids: tuple[int, ...]
    answer_ids: tuple[int, ...]

    @property
    def document_ids(self) -> tuple[int, ...]:
        return self.prompt_ids + self.answer_ids

    @property
    def other_target_ids(self) -> tuple[int, ...]:
        return self.document_ids[1:len(self.prompt_ids)]

    def metadata(self) -> dict:
        return {
            'doc_id': self.doc_id,
            'packed_row': self.row,
            'packed_column': self.column,
            'trained_start_position': self.start_position,
            'document_tokens': len(self.document_ids),
            'prompt_tokens': len(self.prompt_ids),
            'answer_tokens_including_eos': len(self.answer_ids),
            'other_tokens_excluding_document_start': len(self.other_target_ids),
            'packed_tokens_verified': True,
            'packed_absolute_positions_verified': True,
            'packed_labels_verified': True,
            'cross_document_attention_blocked_by_doc_ids': True,
        }


def locate_packed_document(
    arrays: Mapping[str, np.ndarray],
    doc_id: int,
    prompt_ids: Sequence[int],
    answer_ids: Sequence[int],
) -> PackedAlignment:
    """Locate one intact document and verify its tokens, labels, and absolute positions."""
    prompt = tuple(map(int, prompt_ids))
    answer = tuple(map(int, answer_ids))
    if not prompt or not answer:
        raise ValueError('prompt and answer must both contain tokens')
    document = prompt + answer
    locations = np.argwhere(np.asarray(arrays['doc_ids']) == doc_id)
    if len(locations) != len(document):
        raise ValueError(f'doc_id {doc_id}: found {len(locations)} tokens, expected {len(document)}')
    rows = np.unique(locations[:, 0])
    if len(rows) != 1:
        raise ValueError(f'doc_id {doc_id}: document is not intact in one packed row')
    row = int(rows[0])
    columns = locations[:, 1]
    column = int(columns[0])
    if not np.array_equal(columns, np.arange(column, column + len(document))):
        raise ValueError(f'doc_id {doc_id}: document tokens are not contiguous')
    selection = (row, columns)
    if not np.array_equal(np.asarray(arrays['input_ids'])[selection], np.asarray(document)):
        raise ValueError(f'doc_id {doc_id}: packed tokens differ from prompt + answer + EOS')
    if np.asarray(arrays['key_padding_mask'])[selection].any():
        raise ValueError(f'doc_id {doc_id}: document overlaps padding')
    if not np.all(np.asarray(arrays['source_ids'])[selection] == SOURCE_IDS['structured']):
        raise ValueError(f'doc_id {doc_id}: document is not marked as structured')
    expected_positions = np.arange(column, column + len(document))
    actual_positions = np.asarray(arrays['positions'])[selection]
    if not np.array_equal(actual_positions, expected_positions):
        raise ValueError(f'doc_id {doc_id}: packed positions do not continue from absolute column {column}')
    expected_labels = np.asarray(document, dtype=np.int32).copy()
    expected_labels[0] = IGNORE_INDEX
    if not np.array_equal(np.asarray(arrays['labels'])[selection], expected_labels):
        raise ValueError(f'doc_id {doc_id}: causal labels differ at document start or within the document')
    return PackedAlignment(doc_id, row, column, int(actual_positions[0]), prompt, answer)


def align_train_examples(records, consumed, arrays, tokenizer) -> list[PackedAlignment]:
    by_example = {}
    for item in consumed:
        example_id = item['example_id']
        if example_id in by_example:
            raise ValueError(f'duplicate consumed example {example_id}')
        by_example[example_id] = item
    if set(by_example) != {record['example_id'] for record in records}:
        raise ValueError('structured train records and consumed document identities differ')
    alignments = []
    for record in records:
        example_id = record['example_id']
        item = by_example[example_id]
        prompt_ids = tuple(map(int, encode_with_offsets(tokenizer, record['prompt'])[0]))
        answer_ids = tuple(map(int, encode_with_offsets(tokenizer, record['target_text'])[0])) + (EOS_TOKEN_ID,)
        document_ids = tuple(map(int, encode_with_offsets(
            tokenizer, record['prompt'] + record['target_text'],
        )[0])) + (EOS_TOKEN_ID,)
        if document_ids != prompt_ids + answer_ids:
            raise ValueError(f'{example_id}: independent prompt/answer tokenization does not align with the full document')
        content_sha256 = hashlib.sha256(
            (record['prompt'] + record['target_text']).encode('utf-8'),
        ).hexdigest()
        if item.get('content_sha256') != content_sha256:
            raise ValueError(f'{example_id}: consumed content hash differs from the structured record')
        if item.get('tokens') != len(document_ids):
            raise ValueError(f'{example_id}: consumed token count differs from the aligned document')
        alignment = locate_packed_document(arrays, item['doc_id'], prompt_ids, answer_ids)
        alignments.append(alignment)
    return alignments


def _decode(tokenizer, ids) -> str:
    try:
        return tokenizer.decode(list(ids), skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(list(ids))


@torch.no_grad()
def generate_at_position(model, tokenizer, alignment, target_text, start_position, max_new_tokens) -> dict:
    prompt_ids = alignment.prompt_ids
    context_length = model.config.context_length
    first_position = int(start_position)
    next_position = first_position + len(prompt_ids)
    available = context_length - next_position
    if first_position < 0 or available <= 0:
        raise ValueError('prompt positions leave no room for generation in the configured context')
    generation_bound = min(max_new_tokens, available)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device='cpu')
    positions = torch.arange(first_position, next_position, dtype=torch.long, device='cpu').unsqueeze(0)
    sequence = greedy_generate(
        model,
        input_ids,
        max_new_tokens=generation_bound,
        positions=positions,
        doc_ids=torch.zeros_like(input_ids),
        eos_token_id=EOS_TOKEN_ID,
        pad_token_id=EOS_TOKEN_ID,
        use_cache=True,
    )
    generated = sequence[0, len(prompt_ids):].cpu().tolist()
    completion = generated[:generated.index(EOS_TOKEN_ID)] if EOS_TOKEN_ID in generated else generated
    completion_text = _decode(tokenizer, completion)
    parsed_prediction = E.parse_final_registers(completion_text)
    parsed_target = E.parse_final_registers(target_text)
    stopped = EOS_TOKEN_ID in generated
    return {
        'start_position': first_position,
        'prompt_last_position': next_position - 1,
        'requested_max_new_tokens': max_new_tokens,
        'generation_bound': generation_bound,
        'position_bound_limited': generation_bound != max_new_tokens,
        'generated_ids_including_eos': generated,
        'completion_ids': completion,
        'completion_text': completion_text,
        'canonical_valid': parsed_prediction is not None,
        'canonical_exact': (
            parsed_prediction is not None
            and parsed_prediction == parsed_target
            and completion_text == target_text
        ),
        'stopped_on_eos': stopped,
        'truncated': not stopped and len(generated) == generation_bound,
    }


def _ce_metric(loss_sum: float, count: int) -> dict:
    return {'sum': loss_sum, 'count': count, 'mean': loss_sum / count if count else None}


@torch.no_grad()
def teacher_forced_ce(model, alignment, start_position) -> dict:
    """Score next tokens in one isolated document; answer includes EOS, other excludes its first token."""
    document = alignment.document_ids
    prompt_length = len(alignment.prompt_ids)
    input_ids = torch.tensor([document], dtype=torch.long, device='cpu')
    positions = torch.arange(
        int(start_position), int(start_position) + len(document), dtype=torch.long, device='cpu',
    ).unsqueeze(0)
    logits = model(
        input_ids,
        positions=positions,
        doc_ids=torch.zeros_like(input_ids),
    ).logits.float()
    answer_logits = logits[0, prompt_length - 1:-1]
    answer_targets = input_ids[0, prompt_length:]
    answer_sum = float(torch.nn.functional.cross_entropy(
        answer_logits, answer_targets, reduction='sum',
    ))
    other_logits = logits[0, :prompt_length - 1]
    other_targets = input_ids[0, 1:prompt_length]
    other_sum = float(torch.nn.functional.cross_entropy(
        other_logits, other_targets, reduction='sum',
    )) if other_targets.numel() else 0.0
    return {
        'answer_token': _ce_metric(answer_sum, len(alignment.answer_ids)),
        'other_token': _ce_metric(other_sum, len(alignment.other_target_ids)),
    }


def _aggregate_ce(examples, condition) -> dict:
    result = {}
    for group in ('answer_token', 'other_token'):
        loss_sum = 0.0
        count = 0
        for example in examples:
            metric = example[condition]['cross_entropy'][group]
            loss_sum += metric['sum']
            count += metric['count']
        result[group] = _ce_metric(loss_sum, count)
    return result


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _saved_comparison(generated, saved) -> dict:
    return {
        'completion_text': saved['completion_text'],
        'completion_text_match': generated['completion_text'] == saved['completion_text'],
        'completion_tokens': saved['completion_tokens'],
        'completion_token_count_match': generated['completion_ids'].__len__() == saved['completion_tokens'],
        'canonical_exact': saved['correct'],
        'canonical_exact_match': generated['canonical_exact'] == saved['correct'],
        'stopped_on_eos': saved['stopped_on_eos'],
        'stopped_on_eos_match': generated['stopped_on_eos'] == saved['stopped_on_eos'],
        'truncated': saved['truncated'],
        'truncated_match': generated['truncated'] == saved['truncated'],
    }


def _arm_counts(examples) -> dict:
    row_first = [example for example in examples if example['trained_start_position'] == 0]
    later = [example for example in examples if example['trained_start_position'] != 0]
    zero_exact_ids = {example['example_id'] for example in examples if example['zero_position']['canonical_exact']}
    row_first_ids = {example['example_id'] for example in row_first}
    return {
        'examples': len(examples),
        'row_first_examples': len(row_first),
        'non_row_first_examples': len(later),
        'zero_position_exact': sum(example['zero_position']['canonical_exact'] for example in examples),
        'trained_position_exact': sum(example['trained_position']['canonical_exact'] for example in examples),
        'row_first_zero_position_exact': sum(example['zero_position']['canonical_exact'] for example in row_first),
        'non_row_first_zero_position_exact': sum(example['zero_position']['canonical_exact'] for example in later),
        'row_first_trained_position_exact': sum(example['trained_position']['canonical_exact'] for example in row_first),
        'non_row_first_trained_position_exact': sum(
            example['trained_position']['canonical_exact'] for example in later
        ),
        'zero_position_exact_examples_are_exactly_row_first': zero_exact_ids == row_first_ids,
        'zero_position_canonical_valid': sum(example['zero_position']['canonical_valid'] for example in examples),
        'trained_position_canonical_valid': sum(example['trained_position']['canonical_valid'] for example in examples),
        'zero_position_saved_completion_matches': sum(
            example['zero_position']['saved_comparison']['completion_text_match'] for example in examples
        ),
        'zero_position_saved_canonical_exact_matches': sum(
            example['zero_position']['saved_comparison']['canonical_exact_match'] for example in examples
        ),
        'zero_position_saved_completion_token_count_matches': sum(
            example['zero_position']['saved_comparison']['completion_token_count_match'] for example in examples
        ),
        'zero_position_saved_stopped_on_eos_matches': sum(
            example['zero_position']['saved_comparison']['stopped_on_eos_match'] for example in examples
        ),
        'zero_position_saved_truncated_matches': sum(
            example['zero_position']['saved_comparison']['truncated_match'] for example in examples
        ),
        'zero_position_stopped_on_eos': sum(example['zero_position']['stopped_on_eos'] for example in examples),
        'trained_position_stopped_on_eos': sum(example['trained_position']['stopped_on_eos'] for example in examples),
        'zero_position_truncated': sum(example['zero_position']['truncated'] for example in examples),
        'trained_position_truncated': sum(example['trained_position']['truncated'] for example in examples),
    }


def diagnose(run_directory, data_root, tokenizer_path=None, output=None) -> dict:
    run_directory = Path(run_directory).resolve()
    data_root = Path(data_root).resolve()
    output = run_directory / 'position_diagnostic.json' if output is None else Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite an existing diagnostic: {output}')
    launch = json.loads((run_directory / 'launch.json').read_text())
    protocol = json.loads((run_directory / 'protocol.json').read_text())
    status = json.loads((run_directory / 'run_status.json').read_text())
    if not status.get('complete') or status.get('failure') is not None:
        raise ValueError('the acquisition run is not complete')
    if protocol['arms'] != list(ARMS) or set(status['arms']) != set(ARMS):
        raise ValueError('the diagnostic requires the dense, untyped, and typed acquisition arms')
    if Path(launch['data']).resolve() != data_root:
        raise ValueError('data root differs from acquisition launch provenance')
    if sha256_file(data_root / 'manifest.json') != launch['data_manifest_sha256']:
        raise ValueError('data manifest differs from acquisition launch provenance')
    protocol_digest = hashlib.sha256((data_root / 'acquisition_protocol.json').read_bytes()).hexdigest()
    if protocol_digest != (run_directory / 'protocol.sha256').read_text().strip():
        raise ValueError('acquisition protocol hash differs from the run')
    if protocol != json.loads((data_root / 'acquisition_protocol.json').read_text()):
        raise ValueError('run protocol and frozen acquisition protocol differ')
    tokenizer_path = Path(tokenizer_path or launch['tokenizer']).resolve()
    if tokenizer_path != Path(launch['tokenizer']).resolve():
        raise ValueError('tokenizer differs from acquisition launch provenance')
    if sha256_file(tokenizer_path) != launch['tokenizer_sha256']:
        raise ValueError('tokenizer bytes differ from acquisition launch provenance')
    manifest = json.loads((data_root / 'manifest.json').read_text())
    if manifest.get('packing') != 'intact':
        raise ValueError('position diagnosis requires intact document packing')
    dataset = PackedDataset(data_root, 'train')
    tokenizer = load_gpt2_tokenizer(tokenizer_path)
    records = E._load_jsonl(data_root / 'structured/train.jsonl')
    if len(records) != protocol['example_count']:
        raise ValueError('structured train example count differs from the protocol')
    if any(record.get('split') != 'train' or E.parse_final_registers(record['target_text']) is None
           for record in records):
        raise ValueError('train records contain a split or canonical-target mismatch')
    alignments = align_train_examples(
        records,
        _read_jsonl(data_root / 'train_consumed_documents.jsonl'),
        dataset.arrays,
        tokenizer,
    )
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
        'max_updates': protocol['max_updates'],
        'max_new_tokens': protocol['max_new_tokens'],
        'position_conditions': {
            'zero_position': 'prompt and generated tokens use positions 0,1,...',
            'trained_position': 'prompt starts at its packed row column; generated positions continue from there',
        },
        'cross_entropy_definition': (
            'Teacher-forced next-token CE on each complete isolated document with the same causal doc_ids attention boundary. '
            'answer_token includes target_text tokens and EOS; other_token includes prompt targets after the unpredictable '
            'document-start token. No gold token is used during either greedy generation.'
        ),
        'evidence_boundary': (
            'Counterfactual replay of one frozen 32-example memorization run only; no optimizer, weight, architecture, '
            'protocol, checkpoint, old evaluation, or data-array change and no generalization claim.'
        ),
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
        if (not saved_generation['complete']
                or saved_generation['evaluated'] != len(records)
                or saved_generation['max_new_tokens'] != protocol['max_new_tokens']):
            raise ValueError(f'{arm_name}: saved zero-position generation is incomplete or uses another bound')
        saved_by_id = {record['example_id']: record for record in saved_generation['records']}
        if set(saved_by_id) != {record['example_id'] for record in records}:
            raise ValueError(f'{arm_name}: saved generation identities differ from train records')
        model, descriptor = E.load_native_model(
            checkpoint['path'], expected_sha256=checkpoint['sha256'], device='cpu',
        )
        if next(model.parameters()).device.type != 'cpu' or model.training:
            raise RuntimeError(f'{arm_name}: diagnostic model is not an eval-mode CPU model')
        if model.config.context_length != manifest['context_length']:
            raise ValueError(f'{arm_name}: model and packed context lengths differ')
        examples = []
        for record, alignment in zip(records, alignments):
            zero = generate_at_position(
                model, tokenizer, alignment, record['target_text'], 0, protocol['max_new_tokens'],
            )
            zero['cross_entropy'] = teacher_forced_ce(model, alignment, 0)
            zero['saved_comparison'] = _saved_comparison(zero, saved_by_id[record['example_id']])
            trained = generate_at_position(
                model, tokenizer, alignment, record['target_text'],
                alignment.start_position, protocol['max_new_tokens'],
            )
            trained['cross_entropy'] = teacher_forced_ce(model, alignment, alignment.start_position)
            examples.append({
                'example_id': record['example_id'],
                'target_text': record['target_text'],
                **alignment.metadata(),
                'zero_position': zero,
                'trained_position': trained,
            })
        zero_ce = _aggregate_ce(examples, 'zero_position')
        trained_ce = _aggregate_ce(examples, 'trained_position')
        saved_answer_ce = evaluation['splits']['train']['state_path']['answer_token_cross_entropy']['train']
        replay_answer_ce = zero_ce['answer_token']
        answer_ce_matches = (
            replay_answer_ce['count'] == saved_answer_ce['count']
            and math.isclose(replay_answer_ce['sum'], saved_answer_ce['sum'], rel_tol=1e-10, abs_tol=1e-8)
        )
        result['arms'][arm_name] = {
            'checkpoint': {
                'path': checkpoint['path'],
                'sha256': checkpoint['sha256'],
                'step': descriptor['step'],
                'config': descriptor['config'],
            },
            'saved_evaluation': {'path': str(evaluation_path), 'sha256': sha256_file(evaluation_path)},
            'counts': _arm_counts(examples),
            'cross_entropy': {'zero_position': zero_ce, 'trained_position': trained_ce},
            'saved_zero_position_answer_token_cross_entropy': saved_answer_ce,
            'zero_position_answer_token_cross_entropy_matches_saved': answer_ce_matches,
            'examples': examples,
        }
        print(json.dumps({'arm': arm_name, **result['arms'][arm_name]['counts']}, sort_keys=True), flush=True)
        del model, descriptor
        gc.collect()
    result['summary'] = {
        'arms': len(ARMS),
        'examples_per_arm': len(records),
        'zero_position_exact_by_arm': {arm: result['arms'][arm]['counts']['zero_position_exact'] for arm in ARMS},
        'trained_position_exact_by_arm': {
            arm: result['arms'][arm]['counts']['trained_position_exact'] for arm in ARMS
        },
        'zero_position_reproduces_11_of_32_in_every_arm': all(
            result['arms'][arm]['counts']['zero_position_exact'] == 11 for arm in ARMS
        ),
        'trained_position_reaches_32_of_32_in_every_arm': all(
            result['arms'][arm]['counts']['trained_position_exact'] == 32 for arm in ARMS
        ),
        'zero_position_saved_completions_reproduced_in_every_arm': all(
            result['arms'][arm]['counts']['zero_position_saved_completion_matches'] == len(records) for arm in ARMS
        ),
        'zero_position_exact_examples_are_exactly_row_first_in_every_arm': all(
            result['arms'][arm]['counts']['zero_position_exact_examples_are_exactly_row_first'] for arm in ARMS
        ),
        'zero_position_answer_token_cross_entropy_matches_saved_in_every_arm': all(
            result['arms'][arm]['zero_position_answer_token_cross_entropy_matches_saved'] for arm in ARMS
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, help='completed acquisition run directory')
    parser.add_argument('--data', required=True, help='frozen acquisition data root')
    parser.add_argument('--tokenizer', help='must match the acquisition launch provenance')
    parser.add_argument('--output', help='new JSON path; defaults to RUN/position_diagnostic.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        raise RuntimeError("CPU-only diagnostic requires CUDA_VISIBLE_DEVICES=''")
    result = diagnose(args.run, args.data, args.tokenizer, args.output)
    print(json.dumps({
        'device': device,
        'output': str(Path(args.output).resolve()) if args.output else str(Path(args.run).resolve() / 'position_diagnostic.json'),
        **result['summary'],
    }, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
