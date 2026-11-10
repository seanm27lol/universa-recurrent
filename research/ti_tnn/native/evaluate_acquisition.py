"""Evaluate the bounded acquisition fit and separate held-out diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import evaluate as E
from .evaluate_state_path import evaluate_state_path
from .independent.rescore_generations import source_records, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--data', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=48)
    args = parser.parse_args()
    root = Path(args.data)
    model, descriptor = E.load_native_model(args.checkpoint, expected_sha256=args.expected_sha256)
    tokenizer = E.load_gpt2_tokenizer(args.tokenizer)
    sources = source_records(root)
    result = {
        'format': 'native-acquisition-evaluation-v1',
        'checkpoint': {'path': args.checkpoint, 'sha256': descriptor['sha256'],
                       'step': descriptor['step'], 'config': descriptor['config']},
        'splits': {},
        'evidence_boundary': 'training-set fit is memorization only; held-out diagnostics remain separate',
    }
    for split in ('train', 'dev'):
        records = [json.loads(line) for line in (root / 'structured' / f'{split}.jsonl').read_text().splitlines()]
        generation = E._generation_metrics(
            model, tokenizer, records, max_new_tokens=args.max_new_tokens, eos_token_id=50256,
            strict_registers=True, use_cache=True, limit=None, prompt_policy='error',
        )
        state = evaluate_state_path(
            args.checkpoint, root / split, root / 'acquisition' / f'{split}_state_path.json',
            expected_sha256=args.expected_sha256, batch_size=8, max_batches=16,
            structured_directory=root / 'structured', tokenizer_path=args.tokenizer,
        )
        diagnostics = summarize(generation['records'], sources)
        result['splits'][split] = {
            'generation': generation,
            'independent_diagnostics': diagnostics,
            'state_path': state,
            'scope': 'memorization fit' if split == 'train' else 'separate held-out diagnostic',
        }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'checkpoint_sha256': descriptor['sha256'],
                      'train_exact': result['splits']['train']['generation']['correct'],
                      'dev_exact': result['splits']['dev']['generation']['correct'],
                      'output': args.output}, sort_keys=True))


if __name__ == '__main__':
    main()
