"""Measure supervised after-state readout without claiming causal semantic use."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .data import EOS_TOKEN_ID, PackedDataset, load_gpt2_tokenizer
from .evaluate import _model_device, load_native_model
from .structured import SPLIT_NAMES

PROGRAM_STUDY_SPLITS = ('train', 'dev', 'initial_values', 'unseen_composition', 'longer_length')


@torch.no_grad()
def _answer_cross_entropy(model, tokenizer, structured_directory, split=None, evaluation_ids=None):
    if split is not None and split not in PROGRAM_STUDY_SPLITS:
        raise ValueError(f'unknown structured split {split!r}')
    structured_directory = Path(structured_directory)
    device = _model_device(model)
    results = {}
    for split_name in SPLIT_NAMES if split is None else (split,):
        path = structured_directory / f'{split_name}.jsonl'
        if not path.is_file():
            results[split_name] = {
                'sum': 0.0, 'count': 0, 'mean': None, 'examples': 0, 'evaluated': 0,
                'complete': False, 'status': 'not_run',
            }
            continue
        source_records = [json.loads(line) for line in path.read_text().splitlines()]
        if evaluation_ids is None:
            records = source_records[:128]
        else:
            if not isinstance(evaluation_ids, (list, tuple)) or any(
                    not isinstance(example_id, str) for example_id in evaluation_ids):
                raise ValueError('evaluation_ids must be a list of strings')
            if len(evaluation_ids) != len(set(evaluation_ids)):
                raise ValueError('evaluation_ids must not contain duplicates')
            by_id = {}
            for record in source_records:
                example_id = record['example_id']
                if example_id in by_id:
                    raise ValueError(f'{split_name}: duplicate source example_id {example_id}')
                by_id[example_id] = record
            missing = [example_id for example_id in evaluation_ids if example_id not in by_id]
            if missing:
                raise ValueError(f'{split_name}: unknown evaluation IDs, starting with {missing[:3]}')
            records = [by_id[example_id] for example_id in evaluation_ids]
        evaluated = len(records)
        loss_sum = 0.0
        count = 0
        per_example = []
        for record in records:
            prompt_ids = tokenizer.encode(record['prompt'], add_special_tokens=False).ids
            if not prompt_ids:
                raise ValueError(f"{record['example_id']}: answer cross-entropy requires a nonempty prompt")
            answer_ids = tokenizer.encode(record['target_text'], add_special_tokens=False).ids + [EOS_TOKEN_ID]
            ids = prompt_ids + answer_ids
            if len(ids) > model.config.context_length:
                raise ValueError(f"{record['example_id']}: complete prompt and answer exceed context length")
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            labels = torch.full((1, len(ids)), -100, dtype=torch.long, device=device)
            labels[0, len(prompt_ids):] = torch.tensor(answer_ids, dtype=torch.long, device=device)
            logits = model(input_ids).logits
            selected_logits = logits[:, len(prompt_ids) - 1:-1].float()
            selected_targets = labels[:, len(prompt_ids):]
            example_loss = float(torch.nn.functional.cross_entropy(
                selected_logits.reshape(-1, selected_logits.shape[-1]),
                selected_targets.reshape(-1), reduction='sum',
            ))
            loss_sum += example_loss
            count += len(answer_ids)
            if evaluation_ids is not None:
                per_example.append({
                    'example_id': record['example_id'],
                    'sum': example_loss,
                    'count': len(answer_ids),
                    'mean': example_loss / len(answer_ids),
                    'prompt_tokens': len(prompt_ids),
                    'answer_tokens_including_eos': len(answer_ids),
                })
        if evaluation_ids is None:
            complete = evaluated == len(source_records)
            summary = {'examples': len(source_records)}
        else:
            complete = evaluated == len(evaluation_ids)
            summary = {
                'examples': evaluated,
                'source_examples': len(source_records),
                'selection': 'manifest_or_explicit_evaluation_ids',
            }
        results[split_name] = {
            'sum': loss_sum, 'count': count, 'mean': loss_sum / count if count else None,
            **summary, 'evaluated': evaluated, 'complete': complete,
            'status': 'complete' if complete else 'partial',
            **({'records': per_example} if evaluation_ids is not None else {}),
        }
    return results


def _validate_document_positions(dataset):
    arrays = dataset.arrays
    positions = np.asarray(arrays['positions'])
    doc_ids = np.asarray(arrays['doc_ids'])
    padding = np.asarray(arrays['key_padding_mask'])
    live = ~padding
    if np.any(positions[padding] != 0):
        raise ValueError('document-position policy requires zero positions at padding')
    if np.any(doc_ids[live] < 0):
        raise ValueError('live packed tokens must have nonnegative document IDs')
    document_starts = 0
    if len(positions):
        first = live[:, 0]
        if np.any(positions[first, 0] != 0):
            raise ValueError('every live packed row must start at document position zero')
        document_starts += int(first.sum())
        for column in range(1, positions.shape[1]):
            current = live[:, column]
            continuing = current & live[:, column - 1] & (doc_ids[:, column] == doc_ids[:, column - 1])
            starting = current & ~continuing
            if np.any(positions[starting, column] != 0):
                raise ValueError(f'document position does not reset at column {column}')
            if np.any(positions[continuing, column] != positions[continuing, column - 1] + 1):
                raise ValueError(f'document position is not contiguous within a document at column {column}')
            document_starts += int(starting.sum())
    return {
        'position_policy': 'document',
        'packed_rows_checked': len(positions),
        'live_tokens_checked': int(live.sum()),
        'document_starts_checked': document_starts,
        'valid': True,
    }


@torch.no_grad()
def evaluate_state_path(checkpoint, packed_dev, output, expected_sha256=None, batch_size=8, max_batches=16,
                        structured_directory=None, tokenizer_path=None, split=None, evaluation_ids=None,
                        position_policy=None):
    if split is not None and split not in PROGRAM_STUDY_SPLITS:
        raise ValueError(f'unknown structured split {split!r}')
    if batch_size <= 0:
        raise ValueError('batch_size must be positive')
    if max_batches is not None and max_batches < 0:
        raise ValueError('max_batches must be nonnegative or None')
    if position_policy not in (None, 'document'):
        raise ValueError("position_policy must be None or 'document'")
    model, descriptor = load_native_model(checkpoint, expected_sha256=expected_sha256)
    dataset = PackedDataset(packed_dev)
    position_validation = _validate_document_positions(dataset) if position_policy == 'document' else None
    packed_rows = len(dataset)
    row_limit = packed_rows if max_batches is None else min(packed_rows, batch_size * max_batches)
    if row_limit < packed_rows:
        raise ValueError(
            f'max_batches={max_batches} caps state-path evaluation at {row_limit} of {packed_rows} packed rows'
        )
    counts = Counter()
    batches = 0
    evaluated_rows = 0
    for start in range(0, row_limit, batch_size):
        indices = list(range(start, min(start + batch_size, packed_rows)))
        batch = dataset.batch(indices)
        result = model(**{name: batch[name] for name in ('input_ids', 'positions', 'doc_ids', 'key_padding_mask')},
                       return_aux=True)
        predictions = result.state_logits.argmax(dim=-1)
        targets = batch['state_targets']
        mask = batch['state_mask']
        for pass_index in range(4):
            for slot_index in range(4):
                selected = mask[:, :, pass_index, slot_index]
                correct = (predictions[:, :, pass_index, slot_index] == targets[:, :, pass_index, slot_index]) & selected
                counts[f'pass_{pass_index}_slot_{slot_index}_correct'] += int(correct.sum())
                counts[f'pass_{pass_index}_slot_{slot_index}_count'] += int(selected.sum())
        batches += 1
        evaluated_rows += len(indices)
    metrics = {}
    for pass_index in range(4):
        correct = sum(counts[f'pass_{pass_index}_slot_{slot_index}_correct'] for slot_index in range(4))
        count = sum(counts[f'pass_{pass_index}_slot_{slot_index}_count'] for slot_index in range(4))
        metrics[f'pass_{pass_index}'] = {'correct': correct, 'count': count, 'accuracy': correct / count if count else None}
    result = {
        'format': 'native-state-path-evaluation-v1',
        'checkpoint': {'path': str(checkpoint), 'sha256': descriptor['sha256']},
        'packed_rows': packed_rows,
        'evaluated_rows': evaluated_rows,
        'complete': evaluated_rows == packed_rows,
        'batches': batches,
        'metrics': metrics,
        'counts': dict(counts),
        'state_schedule': 'before, after, after, after',
        'pass_2_contract': 'instruction after-state is supervised before the pass-3 downstream read',
        'answer_token_cross_entropy': None if structured_directory is None else _answer_cross_entropy(
            model,
            load_gpt2_tokenizer(tokenizer_path or structured_directory.parent / 'cache/tokenizer/tokenizer.json'),
            structured_directory,
            split=split,
            evaluation_ids=evaluation_ids,
        ),
        'answer_loss_distinction': 'answer-token cross-entropy is model loss, separate from mean absolute register error',
        'evidence_boundary': 'probe accuracy measures readout only; it does not prove the state is used for generation',
    }
    if position_validation is not None:
        result['document_position_validation'] = position_validation
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--packed-dev', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-batches', type=int, default=16)
    parser.add_argument('--structured')
    parser.add_argument('--tokenizer')
    parser.add_argument('--split', choices=PROGRAM_STUDY_SPLITS)
    args = parser.parse_args()
    result = evaluate_state_path(args.checkpoint, args.packed_dev, args.output, args.expected_sha256,
                                 args.batch_size, args.max_batches, args.structured, args.tokenizer, args.split)
    print(json.dumps({'checkpoint_sha256': result['checkpoint']['sha256'], 'metrics': result['metrics']}, sort_keys=True))


if __name__ == '__main__':
    main()
