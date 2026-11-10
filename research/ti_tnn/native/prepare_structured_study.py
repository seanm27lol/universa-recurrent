"""Prepare a bounded intact structured-only dataset from preserved examples."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data as D
from . import structured


def prepare(source, output, train_tokens, dev_tokens, context_length=256, tokenizer_path=None,
            position_policy='row'):
    source = Path(source)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = D.load_gpt2_tokenizer(tokenizer_path or source.parent / 'cache/tokenizer/tokenizer.json')
    manifest = {
        'format': 'native-intact-structured-study-data-v1',
        'source_data': str(source),
        'source_manifest_sha256': D.sha256_file(source / 'manifest.json'),
        'tokenizer': D.TOKENIZER_SPEC,
        'context_length': context_length,
        'packing': 'intact',
        'position_policy': position_policy,
        'structured_overflow': 'exclude',
        'prerequisites': 'initial values and complete ordered program source are present in every packed example',
        'splits': {},
    }
    for split, budget in (('train', train_tokens), ('dev', dev_tokens)):
        runs = []
        excluded = []
        consumed = []
        total = 0
        for line in (source / 'structured' / f'{split}.jsonl').read_text().splitlines():
            record = json.loads(line)
            program = tuple(structured.Instruction.parse(text) for text in record['program'])
            example = structured.make_example(split, program, record['initial'])
            tokenized = D.tokenize_structured(example, tokenizer)
            if len(tokenized.input_ids) > context_length:
                excluded.append({'example_id': example.example_id, 'tokens': len(tokenized.input_ids)})
                continue
            if total + len(tokenized.input_ids) > budget:
                break
            doc_id = len(runs)
            runs.append(D.TokenRun(doc_id, tokenized, len(tokenized.input_ids)))
            consumed.append({'doc_id': doc_id, 'example_id': example.example_id,
                             'content_sha256': example.content_sha256, 'tokens': len(tokenized.input_ids)})
            total += len(tokenized.input_ids)
        if total <= 0:
            raise D.DataError(f'{split}: no fitting intact structured examples are available')
        arrays, stats = D.pack_runs(runs, context_length=context_length, packing='intact', structured_overflow='exclude',
                                    position_policy=position_policy)
        files = {name: D._write_array(output / split, name, arrays[name]) for name in D.ARRAY_NAMES}
        consumed_path = output / f'{split}_consumed_documents.jsonl'
        D._atomic_jsonl(consumed_path, consumed)
        manifest['splits'][split] = {
            **stats,
            'requested_tokens': budget,
            'consumed_tokens': total,
            'underfill_tokens': budget - total,
            'underfill_reason': 'whole-example boundary or finite fitting inventory; no document was split to meet the cap',
            'arrays': files,
            'consumed_document_file': str(consumed_path),
            'consumed_document_sha256': D.sha256_file(consumed_path),
            'long_examples_excluded': len(excluded),
            'long_excluded_tokens': sum(item['tokens'] for item in excluded),
            'long_examples': excluded,
        }
    D._atomic_json(output / 'manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--train-tokens', type=int, default=2_000_000)
    parser.add_argument('--dev-tokens', type=int, default=100_000)
    parser.add_argument('--tokenizer')
    args = parser.parse_args()
    manifest = prepare(args.source, args.output, args.train_tokens, args.dev_tokens, tokenizer_path=args.tokenizer)
    print(json.dumps({split: {'tokens': values['tokens'], 'loss_tokens': values['loss_tokens'],
                              'long_examples_excluded': values['long_examples_excluded']}
                      for split, values in manifest['splits'].items()}, sort_keys=True))


if __name__ == '__main__':
    main()
