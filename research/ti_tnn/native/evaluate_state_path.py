"""Measure supervised after-state readout without claiming causal semantic use."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .data import PackedDataset, load_gpt2_tokenizer
from .evaluate import load_native_model
from .structured import SPLIT_NAMES


@torch.no_grad()
def _answer_cross_entropy(model, tokenizer, structured_directory):
    structured_directory = Path(structured_directory)
    results = {}
    for split in SPLIT_NAMES:
        records = [json.loads(line) for line in (structured_directory / f'{split}.jsonl').read_text().splitlines()]
        loss_sum = 0.0
        count = 0
        for record in records[:128]:
            prompt_ids = tokenizer.encode(record['prompt'], add_special_tokens=False).ids
            answer_ids = tokenizer.encode(record['target_text'], add_special_tokens=False).ids + [50256]
            ids = prompt_ids + answer_ids
            if len(ids) > model.config.context_length:
                raise ValueError(f"{record['example_id']}: complete prompt and answer exceed context length")
            input_ids = torch.tensor([ids], dtype=torch.long)
            labels = torch.full((1, len(ids)), -100, dtype=torch.long)
            labels[0, len(prompt_ids):] = torch.tensor(answer_ids, dtype=torch.long)
            logits = model(input_ids).logits
            selected_logits = logits[:, len(prompt_ids) - 1:-1].float()
            selected_targets = labels[:, len(prompt_ids):]
            loss_sum += float(torch.nn.functional.cross_entropy(
                selected_logits.reshape(-1, selected_logits.shape[-1]),
                selected_targets.reshape(-1), reduction='sum',
            ))
            count += len(answer_ids)
        results[split] = {'sum': loss_sum, 'count': count, 'mean': loss_sum / count if count else None}
    return results


@torch.no_grad()
def evaluate_state_path(checkpoint, packed_dev, output, expected_sha256=None, batch_size=8, max_batches=16,
                        structured_directory=None, tokenizer_path=None):
    model, descriptor = load_native_model(checkpoint, expected_sha256=expected_sha256)
    dataset = PackedDataset(packed_dev)
    counts = Counter()
    batches = 0
    for start in range(0, min(len(dataset), batch_size * max_batches), batch_size):
        indices = list(range(start, min(start + batch_size, len(dataset))))
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
    metrics = {}
    for pass_index in range(4):
        correct = sum(counts[f'pass_{pass_index}_slot_{slot_index}_correct'] for slot_index in range(4))
        count = sum(counts[f'pass_{pass_index}_slot_{slot_index}_count'] for slot_index in range(4))
        metrics[f'pass_{pass_index}'] = {'correct': correct, 'count': count, 'accuracy': correct / count if count else None}
    result = {
        'format': 'native-state-path-evaluation-v1',
        'checkpoint': {'path': str(checkpoint), 'sha256': descriptor['sha256']},
        'batches': batches,
        'metrics': metrics,
        'counts': dict(counts),
        'state_schedule': 'before, after, after, after',
        'pass_2_contract': 'instruction after-state is supervised before the pass-3 downstream read',
        'answer_token_cross_entropy': None if structured_directory is None else _answer_cross_entropy(
            model,
            load_gpt2_tokenizer(tokenizer_path or structured_directory.parent / 'cache/tokenizer/tokenizer.json'),
            structured_directory,
        ),
        'answer_loss_distinction': 'answer-token cross-entropy is model loss, separate from mean absolute register error',
        'evidence_boundary': 'probe accuracy measures readout only; it does not prove the state is used for generation',
    }
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
    args = parser.parse_args()
    result = evaluate_state_path(args.checkpoint, args.packed_dev, args.output, args.expected_sha256,
                                 args.batch_size, args.max_batches, args.structured, args.tokenizer)
    print(json.dumps({'checkpoint_sha256': result['checkpoint']['sha256'], 'metrics': result['metrics']}, sort_keys=True))


if __name__ == '__main__':
    main()
