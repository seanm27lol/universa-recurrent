"""Measure supervised after-state readout without claiming causal semantic use."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .data import PackedDataset
from .evaluate import load_native_model


@torch.no_grad()
def evaluate_state_path(checkpoint, packed_dev, output, expected_sha256=None, batch_size=8, max_batches=16):
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
        'pass_2_contract': 'instruction after-state is supervised before the pass-3 downstream read',
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
    args = parser.parse_args()
    result = evaluate_state_path(args.checkpoint, args.packed_dev, args.output, args.expected_sha256,
                                 args.batch_size, args.max_batches)
    print(json.dumps({'checkpoint_sha256': result['checkpoint']['sha256'], 'metrics': result['metrics']}, sort_keys=True))


if __name__ == '__main__':
    main()
