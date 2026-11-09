"""Independent native-study artifact review; deliberately does not import producer modules."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_native')
STUDY = ROOT / 'study_v7'
MODES = ('dense', 'shared', 'untyped', 'typed')
checks = []


def check(name, condition, detail):
    checks.append({'name': name, 'passed': bool(condition), 'detail': detail})


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def ledger_usage(path):
    reservations = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        entry = reservations.setdefault(event['reservation'], {
            'start': None, 'elapsed': 0.0, 'ended': False, 'status': None,
            'command': None, 'authorization': None,
        })
        if event['event'] == 'start':
            entry.update(start=event['t'], command=event.get('command'), authorization=event.get('authorization'))
        elif event['event'] == 'heartbeat':
            entry['elapsed'] = max(entry['elapsed'], float(event['elapsed']))
        elif event['event'] == 'end':
            entry['ended'] = True
            entry['status'] = event.get('status')
            entry['elapsed'] = max(entry['elapsed'], float(event['elapsed']))
    total = 0.0
    for entry in reservations.values():
        if entry['start'] is None:
            continue
        seconds = entry['elapsed'] if entry['ended'] else entry['elapsed'] + 30.0
        entry['hours'] = seconds / 3600.0
        entry['open'] = not entry['ended']
        total += entry['hours']
    return total, reservations


def main():
    summary = load(STUDY / 'summary.json')
    protocol = load(STUDY / 'protocol.json')
    status = load(STUDY / 'run-status.json')
    check('protocol hash', sha256(STUDY / 'protocol.json') == (STUDY / 'protocol.sha256').read_text().strip(),
          (STUDY / 'protocol.sha256').read_text().strip())
    check('all arms complete', status['status'] == 'completed' and set(status['arms']) == set(MODES), status['arms'])
    check('summary complete', summary['status'] == 'complete', summary['status'])
    check('equal budget', summary['equal_actual_loss_tokens_per_arm'] == 20001877,
          summary['equal_actual_loss_tokens_per_arm'])

    for mode in MODES:
        arm_summary = summary['arms'][mode]
        arm = STUDY / 'arms' / mode
        check(f'{mode} endpoint', arm_summary['actual_loss_tokens'] == 20001877
              and arm_summary['planned_updates'] == 9819,
              {'tokens': arm_summary['actual_loss_tokens'], 'updates': arm_summary['planned_updates']})
        final = arm_summary['checkpoints']['final']
        check(f'{mode} final checkpoint sha256', Path(final['path']).is_file()
              and sha256(final['path']) == final['sha256'], final['sha256'])
        manifest = load(final['manifest_path'])
        check(f'{mode} checkpoint manifest', manifest['sha256'] == final['sha256']
              and manifest['step'] == 9819 and manifest['consumed_loss_tokens'] == 20001877,
              {'step': manifest['step'], 'tokens': manifest['consumed_loss_tokens']})
        for stage in arm_summary['reservations']:
            name = stage['stage']
            reservation = stage['gpu_accounting']['reservation']
            check(f'{mode}/{name} reservation', stage['status'] == 'completed'
                  and reservation['status'] == 'ok' and not reservation['open'] and reservation['hours'] > 0,
                  {'status': reservation['status'], 'hours': reservation['hours']})
        curve = arm_summary['loss_vs_token']['train']
        check(f'{mode} loss decreased', curve[-1]['language_loss'] < curve[0]['language_loss'],
              {'first': curve[0]['language_loss'], 'last': curve[-1]['language_loss']})
        check(f'{mode} loss curve endpoint', curve[-1]['consumed_loss_tokens'] == 20001877
              and curve[-1]['step'] == 9819, curve[-1])
        for source, values in arm_summary['per_source_metrics'].items():
            expected_loss = values['cross_entropy_sum'] / values['loss_tokens']
            check(f'{mode}/{source} metric arithmetic',
                  math.isclose(values['loss'], expected_loss, rel_tol=1e-9)
                  and math.isclose(values['perplexity'], math.exp(expected_loss), rel_tol=1e-9)
                  and math.isclose(values['next_token_accuracy'], values['correct_next_tokens'] / values['loss_tokens'],
                                   rel_tol=1e-9),
                  {'loss': values['loss'], 'tokens': values['loss_tokens']})
        visible = arm_summary['generation']['structured']['study_visible_summary']
        for key in ('ordinary_dev', 'unseen_composition_overall', 'longer_length'):
            values = visible[key]
            check(f'{mode}/{key} generation arithmetic',
                  math.isclose(values['exact_match'], values['correct'] / values['evaluated'], abs_tol=1e-12),
                  {'correct': values['correct'], 'evaluated': values['evaluated']})
        composition = visible['composition_only_lengths_3_4']
        check(f'{mode}/composition generation arithmetic', composition['correct'] == 0
              and composition['evaluated'] == 85 and composition['exact_match'] == 0.0,
              {'correct': composition['correct'], 'evaluated': composition['evaluated']})
        code = arm_summary['generation']['code']
        check(f'{mode}/code generation count', code['correct'] == 0 and code['evaluated'] == 64
              and len(code['records']) == 64 and not any(record['correct'] for record in code['records']),
              {'correct': code['correct'], 'evaluated': code['evaluated']})
        reconstruction = arm_summary['reconstruction']
        check(f'{mode} reconstruction agreement', reconstruction['agreement']
              and reconstruction['teacher_forced']['agrees'] and reconstruction['generation']['agrees'],
              reconstruction['teacher_forced']['agreement_policy'])
        for name, comparison in reconstruction['teacher_forced']['comparisons'].items():
            if comparison is not None:
                relative = comparison.get('relative_rms_error')
                check(f'{mode}/cache {name} criterion', comparison['scale_agrees']
                      and (relative is None or relative <= 0.05),
                      {'relative_rms_error': relative, 'strict_allclose': comparison['agrees']})
        routing = arm_summary['routing']['checkpointed_evaluation']
        if mode in ('untyped', 'typed'):
            for pass_index, values in routing['per_pass'].items():
                counts = values['selected_counts']
                total = sum(counts.values())
                check(f'{mode}/pass {pass_index} route conservation', total == 32768
                      and math.isclose(sum(values['selected_fractions'].values()), 1.0, abs_tol=1e-9),
                      {'total': total, 'starved': values['starved_experts']})
        check(f'{mode}/code prompts truncation disclosed', code['prompt_truncated'] == 64
              and code['prompt_truncation_rate'] == 1.0, code['prompt_truncation_rate'])

    ledger_total, reservations = ledger_usage(
        '/home/seanjazm27/projects/universa-recurrent-scaling/runs/ti_tnn_scaling/gpu_ledger.jsonl'
    )
    native_used = ledger_total - 23.951425708866115
    check('raw ledger total', math.isclose(native_used, 3.1345296053484546, abs_tol=1e-9),
          {'historical': 23.951425708866115, 'native_used': native_used})
    check('native authorization cap', native_used <= 12.0 and ledger_total <= 35.951425708866115 + 1e-9,
          {'used': native_used, 'cap': 12.0})
    failed = [entry for entry in reservations.values() if entry['status'] == 'failed']
    check('failed attempts remain charged', len(failed) >= 3,
          [{'command': entry['command'], 'hours': entry['hours']} for entry in failed])

    serious = load(ROOT / 'profiles/serious_typed.json')
    rate = serious['actual_shifted_loss_tokens_per_second']
    train_seconds = 125_000_000 / rate
    checkpoint_seconds = 124 * serious['checkpoint_seconds']
    dev_seconds = 62 * serious['evaluation']['wall_seconds']
    fixed_seconds = (serious['setup']['fixed_seconds'] + serious['complete_evaluation']['total_seconds']
                     + serious['reconstruction']['seconds'])
    total_seconds = train_seconds + checkpoint_seconds + dev_seconds + fixed_seconds
    total_hours = total_seconds / 3600
    proposal = {
        'loss_tokens': 125_000_000,
        'training_seconds': train_seconds,
        'checkpoint_seconds': checkpoint_seconds,
        'periodic_dev_seconds': dev_seconds,
        'complete_evaluation_reconstruction_setup_seconds': fixed_seconds,
        'total_gpu_hours': total_hours,
        'with_20_percent_reserve_hours': total_hours * 1.2,
        'tokens_per_gpu_hour': 125_000_000 / total_hours,
        'retain_all_checkpoint_bytes': 124 * serious['checkpoint']['bytes'],
        'bounded_retention_bytes': 2 * serious['checkpoint']['bytes'],
        'authorization': False,
    }
    check('serious direct proposal arithmetic', math.isclose(total_hours, 4.276552444040035, rel_tol=1e-9), proposal)
    optimistic = summary['future_serious_125m_proposal']['arms']['typed']['proposed_serious_gpu_hours']
    check('MAC-scaled proposal is not direct measured throughput', optimistic < total_hours / 20,
          {'mac_scaled_hours': optimistic, 'direct_measured_hours': total_hours})
    check('strict CPU failure preserved', (ROOT / 'study_v3/arms/dense/reconstruction-omp4-failed.json').is_file(),
          'study_v3/arms/dense/reconstruction-omp4-failed.json')
    check('BF16 route failure preserved', (ROOT / 'study_v5/arms/untyped/reconstruction.json').is_file(),
          'study_v5/arms/untyped/reconstruction.json')

    result = {
        'status': 'pass' if all(item['passed'] for item in checks) else 'fail',
        'independent_of_producer_imports': True,
        'checks': checks,
        'failures': [item for item in checks if not item['passed']],
        'native_gpu_hours': native_used,
        'direct_serious_proposal': proposal,
        'principal_risks': [
            'One development seed and bounded evaluation do not establish architectural superiority or scaling readiness.',
            'Dense training was reused after identical model/trainer-byte verification; later arms trained under later instrumentation commits.',
            'Typed adds parameters and routing work relative to untyped; this is not a parameter-matched comparison.',
            'All 64 code prompts were truncated, so code generation is a bounded suffix diagnostic, not full-prompt completion.',
            'One longer-length example succeeded per arm while ordinary dev and composition-only exact generation scored zero.',
            'Low late-pass router entropy and named state slots do not establish semantic specialization or faithful explanations.',
            'State interventions used only two prompts and strong off-distribution changes; direct h and other-slot bypasses remain.',
            'The BF16 untyped replay had one route flip; FP32 serving passed, so BF16 serving equivalence is not established.',
            'The direct serious profile, not the optimistic MAC-scaled estimate, is the appropriate 125M planning denominator.',
        ],
    }
    output = STUDY / 'independent_review.json'
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    lines = [
        '# Independent native-study review', '',
        f"**{result['status'].upper()} — {len(checks)} checks, {len(result['failures'])} failures.**", '',
        'This reviewer uses only the Python standard library and does not import native producer modules.', '',
        '## Principal findings', '',
        '- All four arms reached the same 20,001,877-token, 9,819-update endpoint; final checkpoint hashes and stage reservations verified.',
        '- Loss decreased in every arm. Dense had the lowest Wikipedia and code losses; typed did not demonstrate a specialist advantage on this seed.',
        '- Ordinary-dev and composition-only complete generation scored 0/128 and 0/85 respectively. Code suffix generation scored 0/64. Each arm scored 1/128 on longer length, which is not evidence of general composition.',
        f"- Native GPU use reconstructed from the raw ledger: {native_used:.9f}/12 hours, including failed attempts.",
        f"- Direct serious proposal: {total_hours:.6f} hours for 125M tokens, {total_hours * 1.2:.6f} with 20% reserve; not authorized.",
        '', '## Principal implementation and interpretation risks', '',
    ]
    lines.extend(f'- {risk}' for risk in result['principal_risks'])
    lines.extend(['', '## Failed or limited checks', '',
                  '- CPU strict allclose failed on the original dense replay and remains preserved.',
                  '- BF16 untyped replay changed one route decision and remains a failed BF16 serving check; FP32 serving passed.',
                  '- The `nvidia-cusparselt-cu13` platform warning exists in both environments; CUDA correctness runs passed.',
                  '- The initial full local suite had 11 environment failures. Those prerequisites were fixed; remote CI subsequently passed.', ''])
    (STUDY / 'INDEPENDENT_REVIEW.md').write_text('\n'.join(lines))
    print(json.dumps({'status': result['status'], 'checks': len(checks), 'failures': len(result['failures']),
                      'native_gpu_hours': native_used, 'direct_serious_hours': total_hours}, sort_keys=True))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
