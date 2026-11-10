"""Attach model-computed answer-token cross-entropy to independent rescore artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge(root, diagnostics, paired_diagnostics):
    root = Path(root)
    for path, prefix in ((Path(diagnostics), ''), (Path(paired_diagnostics), 'paired_')):
        document = json.loads(path.read_text())
        for mode in document['arms']:
            losses = json.loads((root / 'answer_losses' / f'{prefix}{mode}.json').read_text())
            document['arms'][mode]['answer_token_cross_entropy'] = losses['answer_token_cross_entropy']
        path.write_text(json.dumps(document, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--paired-diagnostics', required=True)
    args = parser.parse_args()
    merge(args.root, args.diagnostics, args.paired_diagnostics)
    print(json.dumps({'status': 'complete', 'diagnostics': args.diagnostics,
                      'paired_diagnostics': args.paired_diagnostics}, sort_keys=True))


if __name__ == '__main__':
    main()
