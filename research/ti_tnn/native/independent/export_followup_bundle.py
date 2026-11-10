"""Export a compact, weight-free follow-up evidence bundle without producer imports."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native')
FOLLOW = Path('/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native_followup')
OUT = FOLLOW / 'evidence'
MODES = ('dense', 'shared', 'untyped', 'typed')
SPLITS = ('train', 'dev', 'unseen_composition', 'longer_length')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def copy(relative_source, relative_destination):
    destination = OUT / relative_destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(relative_source, destination)


def main():
    if OUT.exists():
        raise RuntimeError(f'refusing to overwrite an existing evidence bundle: {OUT}')
    OUT.mkdir(parents=True)
    summary = load(ROOT / 'study_v7/summary.json')
    diagnostics = load(FOLLOW / 'generation_diagnostics_executed.json')
    reported = {'format': 'native-followup-reported-tables-v1', 'arms': {}}
    for mode in MODES:
        arm_summary = summary['arms'][mode]
        reported['arms'][mode] = {
            'loss_vs_token_endpoints': {
                'train_first': arm_summary['loss_vs_token']['train'][0],
                'train_last': arm_summary['loss_vs_token']['train'][-1],
                'dev_first': arm_summary['loss_vs_token']['dev'][0],
                'dev_last': arm_summary['loss_vs_token']['dev'][-1],
            },
            'per_source_metrics': arm_summary['per_source_metrics'],
            'generation_diagnostics': diagnostics['arms'][mode],
            'routing': arm_summary['routing'],
            'time': arm_summary['time'],
            'final_checkpoint': arm_summary['checkpoints']['final'],
            'state_interventions': arm_summary['state_interventions'],
            'reconstruction': arm_summary['reconstruction'],
        }
        arm_source = ROOT / 'study_v7/arms' / mode
        copy(arm_source / 'evaluation.json', f'arms/{mode}/evaluation.json')
        copy(arm_source / 'train-result.json', f'arms/{mode}/train-result.json')
        copy(arm_source / 'train/run_config.json', f'arms/{mode}/run_config.json')
        final = arm_summary['checkpoints']['final']
        copy(final['manifest_path'], f'checkpoints/{mode}.manifest.json')
        copy(arm_source / 'reconstruction.json', f'numerical_history/{mode}.fp32.json')
    (OUT / 'reported_tables.json').write_text(json.dumps(reported, indent=2, sort_keys=True) + '\n')
    copy(ROOT / 'study_v7/protocol.json', 'protocol.json')
    copy(ROOT / 'study_v7/protocol.sha256', 'protocol.sha256')
    copy(ROOT / 'study_v7/INDEPENDENT_REVIEW.md', 'original_113_check_review.md')
    copy(ROOT / 'study_v7/independent_review.json', 'original_113_check_review.json')
    copy(ROOT / 'study_v5/arms/untyped/reconstruction.json', 'numerical_history/untyped.bf16.route_failure.json')
    copy(ROOT / 'study_v3/arms/dense/reconstruction-omp4-failed.json', 'numerical_history/dense.fp32.strict_failure.json')
    copy(FOLLOW / 'generation_diagnostics.json', 'original_generation_diagnostics.json')
    copy(FOLLOW / 'generation_diagnostics_executed.json', 'generation_diagnostics.json')
    copy(FOLLOW / 'paired_generation_diagnostics.json', 'paired/original_generation_diagnostics.json')
    copy(FOLLOW / 'paired_generation_diagnostics_executed.json', 'paired/generation_diagnostics.json')
    for mode in MODES:
        copy(FOLLOW / f'answer_losses/{mode}.json', f'answer_losses/{mode}.json')
    for mode in ('untyped', 'typed'):
        copy(FOLLOW / f'answer_losses/paired_{mode}.json', f'answer_losses/paired_{mode}.json')
    copy(FOLLOW / 'paired_protocol.json', 'paired/protocol.json')
    copy(FOLLOW / 'study_data/manifest.json', 'paired/study_data/manifest.json')
    for mode in ('untyped', 'typed'):
        arm_source = FOLLOW / 'arms' / mode
        for name in ('train-result.json', 'evaluation.json', 'state_path.json', 'reconstruction.json'):
            copy(arm_source / name, f'paired/arms/{mode}/{name}')
    copy(ROOT / 'data/manifest.json', 'data/manifest.json')
    copy(ROOT / 'data/code_generation.jsonl', 'data/code_generation.jsonl')
    for split in SPLITS:
        records = [json.loads(line) for line in (ROOT / f'data/structured/{split}.jsonl').read_text().splitlines()]
        selected = records[:128]
        destination = OUT / f'data/structured/{split}.jsonl'
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(''.join(json.dumps(record, sort_keys=True) + '\n' for record in selected))
    boundary = load(FOLLOW / 'structured_boundary_audit.json')
    (OUT / 'structured_boundary_audit.json').write_text(json.dumps({
        'format': boundary['format'],
        'source_data': boundary['source_data'],
        'summary': {
            split: {key: value for key, value in values.items() if key != 'crossing_examples'}
            for split, values in boundary['summary'].items()
        },
        'interpretation': boundary['interpretation'],
        'full_inventory_location': str(FOLLOW / 'structured_boundary_audit.json'),
    }, indent=2, sort_keys=True) + '\n')
    copy('/home/seanjazm27/projects/universa-recurrent-native-followup/research/ti_tnn/native/fixtures/structured_boundary_counterexample.json',
         'structured_boundary_counterexample.json')
    ledger_source = Path('/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling/gpu_ledger.jsonl')
    ledger_events = [json.loads(line) for line in ledger_source.read_text().splitlines() if line.strip()]
    native_ids = {event['reservation'] for event in ledger_events if event.get('authorization') == 3}
    native_events = [line for line in ledger_source.read_text().splitlines() if line.strip()
                     and json.loads(line)['reservation'] in native_ids]
    (OUT / 'native_ledger.jsonl').write_text('\n'.join(native_events) + '\n')
    code_revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd='/home/seanjazm27/projects/universa-recurrent-native-followup',
                                   capture_output=True, text=True, check=True).stdout.strip()
    reproduction = f'''# Weight-free follow-up evidence bundle

This bundle contains prompts, completions, diagnostics, checkpoint manifests, protocol/configuration identities,
and numerical history. It contains no model weights, packed arrays, or raw corpus files. The original 113-check
review is preserved byte-for-byte; it is an artifact-consistency review, not independent semantic validation.

## Reproduce the independent generation rescore

```bash
PYTHONDONTWRITEBYTECODE=1 python research/ti_tnn/native/independent/rescore_generations.py \
  --study . --data ./data --output ./generation_diagnostics.reproduced.json
```

For the paired intact-data result:

```bash
PYTHONDONTWRITEBYTECODE=1 python research/ti_tnn/native/independent/rescore_generations.py \
  --study ./paired --data ./data --output ./paired/generation_diagnostics.reproduced.json

# Attach answer-token cross-entropy from the preserved checkpoint evaluations.
PYTHONDONTWRITEBYTECODE=1 python research/ti_tnn/native/independent/merge_answer_losses.py \
  --root . --diagnostics ./generation_diagnostics.json \
  --paired-diagnostics ./paired/generation_diagnostics.json
```

The current diagnostics file is `./generation_diagnostics.json`. `reported_tables.json` contains the compact reported
loss, source, routing, timing, intervention and reconstruction tables. The final FP32 and failed BF16/strict CPU
reports remain separate in `numerical_history/`.

Source follow-up revision at export: `{code_revision}`.
Original merged PR #76 revision: `eaa5faa9d4476296e60386bf7cd7179f9cc63b00`.
'''
    (OUT / 'REPRODUCE.md').write_text(reproduction)
    copy(FOLLOW / 'generation_diagnostics_executed.json', 'generation_diagnostics.json')
    report_path = Path('/home/seanjazm27/projects/universa-recurrent-native-followup/research/ti_tnn/native/FOLLOWUP.md')
    if report_path.is_file():
        copy(report_path, 'FOLLOWUP.md')
    files = sorted(path for path in OUT.rglob('*') if path.is_file() and path.name != 'SHA256SUMS')
    (OUT / 'SHA256SUMS').write_text(''.join(f'{sha256(path)}  {path.relative_to(OUT)}\n' for path in files))
    manifest = {
        'format': 'native-followup-weight-free-evidence-v1',
        'files': len(files) + 1,
        'bytes': sum(path.stat().st_size for path in files),
        'original_review_unchanged': True,
        'contains_weights': False,
        'source_revision': code_revision,
    }
    (OUT / 'bundle_manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(json.dumps(manifest, sort_keys=True))


if __name__ == '__main__':
    main()
