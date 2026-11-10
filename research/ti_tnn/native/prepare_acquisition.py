"""Freeze a 32-example balanced acquisition diagnostic without changing the MoE architecture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import data as D
from . import ledger
from . import structured
from .prepare_structured_study import prepare
from .train import make_sampler


def write_source_provenance(source, output, tokenizer):
    source = Path(source).resolve()
    provenance = {
        'format': 'native-acquisition-source-provenance-v1',
        'source_data': str(source),
        'source_manifest_sha256': D.sha256_file(source / 'manifest.json'),
        'tokenizer_path': str(Path(tokenizer).resolve()),
        'tokenizer_sha256': D.sha256_file(tokenizer),
        'structured_files': {
            split: D.sha256_file(source / 'structured' / f'{split}.jsonl')
            for split in ('train', 'dev')
        },
    }
    path = Path(output) / 'source_provenance.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--max-updates', type=int, default=4096)
    args = parser.parse_args()
    source = Path(args.source)
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f'refusing to overwrite an existing acquisition freeze: {output}')
    (output / 'structured').mkdir(parents=True, exist_ok=True)
    tokenizer = args.tokenizer
    selected = {family: [] for family in structured.OP_ROUTES}
    for line in (source / 'structured/train.jsonl').read_text().splitlines():
        record = json.loads(line)
        family = record['program'][0].split()[0]
        if len(selected[family]) < 8:
            selected[family].append(record)
    assert all(len(records) == 8 for records in selected.values())
    train = [record for records in selected.values() for record in records]
    heldout = [json.loads(line) for line in (source / 'structured/dev.jsonl').read_text().splitlines()][:32]
    train_hashes = {record['prompt'] + record['target_text'] for record in train}
    assert not train_hashes & {record['prompt'] + record['target_text'] for record in heldout}
    for split, records in (('train', train), ('dev', heldout)):
        (output / 'structured' / f'{split}.jsonl').write_text(
            ''.join(json.dumps(record, sort_keys=True) + '\n' for record in records))
    write_source_provenance(source, output, tokenizer)
    (output / 'manifest.json').write_text(json.dumps({'format': 'native-acquisition-source-v1'}, indent=2) + '\n')
    prepare(output, output, 10**12, 10**12, tokenizer_path=tokenizer)
    selection = {
        'format': 'native-acquisition-selection-v1',
        'source_data': str(source.resolve()),
        'source_provenance_sha256': D.sha256_file(output / 'source_provenance.json'),
        'examples_per_operation_family': {family: len(records) for family, records in selected.items()},
        'train_example_ids': [record['example_id'] for record in train],
        'heldout_example_ids': [record['example_id'] for record in heldout],
        'train_content_sha256': hashlib.sha256(
            ''.join(record['prompt'] + record['target_text'] for record in train).encode()).hexdigest(),
        'heldout_content_sha256': hashlib.sha256(
            ''.join(record['prompt'] + record['target_text'] for record in heldout).encode()).hexdigest(),
        'selection_rule': 'first eight complete examples per operation family from preserved train; first32 preserved dev for separate diagnostic',
    }
    (output / 'acquisition_selection.json').write_text(json.dumps(selection, indent=2, sort_keys=True) + '\n')
    packed = D.PackedDataset(output, 'train')
    sampler = make_sampler(packed, 8, 20261110)
    for _ in range(args.max_updates):
        sampler.next_batch()
    max_loss_tokens = sampler.consumed_loss_tokens
    protocol = {
        'format': 'native-acquisition-protocol-v1',
        'arms': ['dense', 'untyped', 'typed'],
        'purpose': 'bounded memorization diagnostic, not generalization or architectural superiority',
        'max_updates': args.max_updates,
        'max_loss_tokens': max_loss_tokens,
        'batch_size': 8,
        'seed': 20261110,
        'model_seed': 20261110,
        'data_seed': 20261110,
        'learning_rate': 0.0003,
        'weight_decay': 0.01,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'grad_clip': 1.0,
        'lambda_balance': 0.01,
        'lambda_route': 0.1,
        'lambda_state': 0.1,
        'warmup_fraction': 0.02,
        'final_lr_fraction': 0.1,
        'checkpoint_every': 500,
        'dev_every': 500,
        'dev_batches': 16,
        'objective': 'corrected existing LM + load balance + route + state objective; no answer-focused objective',
        'target_exact_answers': 31,
        'example_count': 32,
        'evaluation_examples_per_split': 32,
        'max_new_tokens': 48,
        'reconstruction': 'FP32 relative RMS error <= 0.05, exact integer routes and exact generated tokens',
        'projected_hours_per_train': 0.15,
        'projected_hours_per_evaluation': 0.10,
        'projected_total_hours': 0.75,
        'live_native_remaining_hours_at_freeze': ledger.usage()['native_remaining_hours'],
        'authorization': 'existing native v3 only; no 124M campaign and no WSP',
    }
    (output / 'acquisition_protocol.json').write_text(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'max_updates': args.max_updates, 'max_loss_tokens': max_loss_tokens,
                      'projected_total_hours': protocol['projected_total_hours'],
                      'live_native_remaining_hours': protocol['live_native_remaining_hours_at_freeze']}, sort_keys=True))


if __name__ == '__main__':
    main()
