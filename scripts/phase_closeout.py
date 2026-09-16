"""One locked validation, then close this phase, even if prediction is poor.

Runs the unchanged local-response experiment on two fresh input blocks. This is
not another model or a new approximation. Every rule still requires model probes.
Primary selection and practical thresholds are fixed in the adjacent protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np

FORMAT = 'universa-recurrent.phase-closeout.v1'
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'experiments/phase_closeout_v1.json'
SEEDS = (91000, 92000)
FITS = (6100, 7100, 8100, 9100, 10100)
ORDERS = ('none', 'linear', 'diagonal_quadratic', 'full_quadratic')
HORIZONS = ('readout_at_cut', 'original_remaining_updates')
RADII = (0.125, 0.0625)
SCALES = (0.25, 0.5, 1.0)
FAMILIES = ('two_cycles_0', 'two_cycles_1', 'cross_candidate',
            'cycle_evidence_0', 'cycle_evidence_1', *(f'dense_{i}' for i in range(4)))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError('duplicate JSON key: ' + key)
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('invalid JSON value: ' + value)
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def protocol():
    p = read(PROTOCOL)
    if (p['format'] != FORMAT + '.protocol' or p['data_seeds'] != list(SEEDS)
        or p['training_seeds'] != list(FITS) or p['n_per_seed'] != 256
        or p['primary']['model'] != 'fixed_depth_4' or p['primary']['cut'] != 2
        or p['primary']['horizon'] != 'original_remaining_updates'
        or p['primary']['radius'] != .125 or p['primary']['order'] != 'linear'
        or p['primary']['scales'] != list(SCALES) or p['primary']['families'] != list(FAMILIES)
        or p['primary']['relative_rms_limit_for_limited_utility'] != .8
        or p['primary']['relative_rms_limit_for_close_prediction'] != .1
        or p['uncertainty']['repeats'] != 3000 or p['uncertainty']['seed'] != 93000
        or p['uncertainty']['upper_percentile'] != .95):
        raise ValueError('locked protocol changed; do not relabel a different experiment as this validation')
    return p


def check_sources(p):
    for filename, wanted in p['frozen_scripts_sha256'].items():
        if digest(ROOT / 'scripts' / filename) != wanted:
            raise ValueError('frozen helper differs: ' + filename)


def cell_key(row):
    return (row['model'], row['cut'], row['horizon'], row['radius'], row['scale'], row['order'])


def label(key):
    return '__'.join(map(str, key))


def numeric(value):
    a = np.asarray(value)
    if a.dtype.kind not in 'fiu' or not a.size or not np.isfinite(a).all():
        raise ValueError('nonfinite or nonnumeric saved prediction')
    return a.astype(np.float64)


def recompute_worker(report_path):
    """Audit every saved score, then keep input-paired sufficient loss statistics.

    Full probes/predictions remain unchanged beside this report on the user's
    machine. This summary is not an independent execution of the neural network.
    """
    w = read(report_path)
    if (w['format'] != 'universa-recurrent.local-response.v1.worker'
        or w['all_raw_gates_passed'] is not True
        or w['per_example_file'] != f'seed-{w["training_seed"]}.npz'):
        raise ValueError('invalid worker identity/gate')
    array_path = report_path.parent / w['per_example_file']
    if digest(array_path) != w['per_example_sha256']:
        raise ValueError('numeric report hash mismatch')
    if any(f['status'] != 'finite' for f in w['fits']) or any(r['status'] != 'finite' for r in w['rows']):
        return {'failed': True, 'training_seed': w['training_seed'], 'report': w}, {}, {}
    grouped = {}; seen = set(); worst = {}
    with np.load(array_path, allow_pickle=False) as archive:
        # Load each compressed array once rather than repeatedly decompressing it.
        a = {name: archive[name] for name in archive.files}
    inputs = {k: a[k] for k in ('observed', 'mask', 'truth', 'labels')}
    n = w['n']
    if inputs['observed'].shape != (n, 5) or inputs['mask'].shape != (n, 5):
        raise ValueError('input dimensions differ')
    for row in w['rows']:
        case = (row['model'], row['cut'], row['horizon'], row['radius'], row['query_index'], row['order'])
        if case in seen:
            raise ValueError('duplicate result case')
        seen.add(case)
        query = w['queries'][row['query_index']]
        if (row['family'], row['scale']) != (query['family'], query['scale']):
            raise ValueError('query metadata mismatch')
        stem = f'{row["model"]}__cut{row["cut"]}__{row["horizon"]}'
        p = numeric(a[stem + f'__radius{row["radius"]:g}__predictions__{row["order"]}'][row['query_index']])
        y = numeric(a[stem + f'__query{row["query_index"]}__actual'])
        b = numeric(a[stem + '__baseline'])
        if p.shape != y.shape or p.shape != b.shape or p.shape != (n, 5):
            raise ValueError('prediction dimensions differ')
        error = np.square(p - y).mean(-1)
        energy = np.square(y - b).mean(-1)
        maximum = float(np.abs(p-y).max())
        expected = {'prediction_mse': float(error.mean()), 'effect_mse': float(energy.mean()),
                    'relative_rms_error': float(np.sqrt(error.mean()/energy.mean())) if energy.mean() > 1e-12 else None,
                    'relative_floor_mse': 1e-12, 'worst_component_error': maximum,
                    'p99_max_component_error': float(np.quantile(np.abs(p-y).max(1), .99))}
        for field, value in expected.items():
            reported = row['metrics'][field]
            if (value is None and reported is not None) or (value is not None and
                (reported is None or not np.isclose(reported, value, rtol=1e-11, atol=1e-14))):
                raise ValueError('saved score mismatch: ' + field)
        if 'control' in row['family']:
            continue
        key = cell_key(row)
        grouped.setdefault(key, []).append((row['family'], error, energy))
        worst[key] = max(maximum, worst.get(key, 0.0))
    if len(seen) != 2 * 2 * 2 * 2 * 33 * 4:
        raise ValueError('incomplete original evaluator cases')
    compact = {}
    for key, cases in grouped.items():
        if {c[0] for c in cases} != set(FAMILIES) or len(cases) != len(FAMILIES):
            raise ValueError('missing or repeated noncontrol query')
        compact[key] = np.stack((np.mean([v[1] for v in cases], axis=0),
                                 np.mean([v[2] for v in cases], axis=0)))
    for fit in w['fits']:
        if fit['probe_evaluations'] != 73 or fit['coefficient_bytes_per_input'] != 1720:
            raise ValueError('probe/storage accounting changed')
    metadata = {k: w[k] for k in ('training_seed', 'test_seed', 'n', 'dataset_sha256',
        'checkpoint_sha256', 'claim_calibration_sha256', 'previous_report_sha256',
        'source', 'raw_gates', 'fits', 'scope')}
    metadata.update(failed=False, report_sha256=digest(report_path), arrays_sha256=digest(array_path),
                    rows_recomputed=len(seen), worst_by_cell={label(k): v for k, v in worst.items()})
    return metadata, compact, inputs


def ratio(error, energy):
    e, d = float(np.mean(error)), float(np.mean(energy))
    if not np.isfinite(e) or not np.isfinite(d) or e < 0 or d < 0:
        raise ValueError('invalid loss totals')
    return math.sqrt(e/d) if d > 1e-12 else None


def paired_bootstrap(blocks, repeats=3000, seed=93000):
    """Inputs, not models or queries, are resampled. Models already averaged."""
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(repeats):
        e = d = 0.0
        for block in blocks:
            if block.ndim != 2 or block.shape[0] != 2 or not np.isfinite(block).all() or (block < 0).any():
                raise ValueError('invalid paired losses')
            indexes = rng.integers(0, block.shape[1], size=block.shape[1])
            e += block[0, indexes].sum(); d += block[1, indexes].sum()
        if d <= 1e-12:
            return None
        ratios.append(math.sqrt(e/d))
    return {'percentile_interval_95': np.quantile(ratios, [.025, .975]).tolist(),
            'one_sided_upper_95': float(np.quantile(ratios, .95)),
            'bootstrap_repeats': repeats, 'seed': seed}


def primary_result(blocks, p):
    point = ratio(np.concatenate([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks]))
    per_block = [ratio(b[0], b[1]) for b in blocks]
    uncertainty = paired_bootstrap(blocks, p['uncertainty']['repeats'], p['uncertainty']['seed'])
    def supported(limit):
        return (point is not None and uncertainty is not None and
                uncertainty['one_sided_upper_95'] <= limit and
                all(v is not None and v <= limit for v in per_block))
    return dict(relative_rms_error=point, per_seed_block_relative_rms=per_block,
                uncertainty=uncertainty,
                limited_utility_criterion_met=supported(p['primary']['relative_rms_limit_for_limited_utility']),
                close_prediction_criterion_met=supported(p['primary']['relative_rms_limit_for_close_prediction']))


def collect(run_dir, p):
    run_dir = Path(run_dir)
    bundles = []; cells = {}; inputs_by_seed = {}; failures = []; primary_blocks = []
    compact_arrays = {}
    for data_seed in p['data_seeds']:
        summary = read(run_dir / f'data-{data_seed}' / 'summary.json')
        if summary['format'] != 'universa-recurrent.local-response.v1.complete' or summary['checkpoint_runs'] != len(FITS):
            raise ValueError('incomplete held-out block')
        primary_fits = []
        for training_seed in p['training_seeds']:
            path = run_dir / f'data-{data_seed}' / f'seed-{training_seed}.json'
            meta, scores, inputs = recompute_worker(path)
            if meta['training_seed'] != training_seed:
                raise ValueError('fit identity mismatch')
            if meta['failed']:
                failures.append({'data_seed': data_seed, 'training_seed': training_seed})
                continue
            if meta['test_seed'] != data_seed or meta['n'] != p['n_per_seed']:
                raise ValueError('test cohort differs from locked protocol')
            if data_seed not in inputs_by_seed:
                inputs_by_seed[data_seed] = inputs
            for name, array in inputs.items():
                if not np.array_equal(array, inputs_by_seed[data_seed][name]):
                    raise ValueError('input cohort differs between fits')
            bundle_id = f'data{data_seed}__fit{training_seed}'
            bundles.append(dict(meta, data_seed=data_seed, bundle_id=bundle_id))
            for key, array in scores.items():
                compact_arrays[bundle_id + '__' + label(key)] = array
                cells.setdefault(key, []).append((data_seed, training_seed, array))
            prefix = p['primary']
            primary_fits.append(np.mean([scores[(prefix['model'], prefix['cut'], prefix['horizon'],
                                     prefix['radius'], scale, prefix['order'])] for scale in prefix['scales']], axis=0))
        if len(primary_fits) == len(FITS):
            primary_blocks.append(np.mean(primary_fits, axis=0))
    for seed, group in inputs_by_seed.items():
        for name, array in group.items():
            compact_arrays[f'data{seed}__input__{name}'] = array
    rows = []
    for key, cases in sorted(cells.items()):
        joined = np.concatenate([case[2] for case in cases], axis=1)
        rows.append(dict(zip(('model', 'cut', 'horizon', 'radius', 'scale', 'order'), key),
                         prediction_mse=float(joined[0].mean()), effect_mse=float(joined[1].mean()),
                         relative_rms_error=ratio(joined[0], joined[1]), checkpoint_blocks=len(cases)))
    # Validate disjoint underlying input problems, not merely unequal seed labels.
    seen = set()
    for seed, group in inputs_by_seed.items():
        signatures = {row.tobytes() for row in np.c_[group['observed'], group['mask']]}
        if len(signatures) != p['n_per_seed'] or seen & signatures:
            raise ValueError('duplicate or overlapping held-out observations')
        seen.update(signatures)
    primary = (primary_result(primary_blocks, p) if not failures and len(primary_blocks) == len(SEEDS)
               else {'status': 'numerical_failure', 'limited_utility_criterion_met': False,
                     'close_prediction_criterion_met': False})
    return {'format': FORMAT + '.complete', 'phase_closed_after_review': True,
            'no_automatic_followup': True, 'underlying_inputs': sum(len(g['observed']) for g in inputs_by_seed.values()),
            'primary': primary, 'secondary_rows': rows, 'failures': failures,
            'worker_audits': bundles,
            'scope': 'Input-paired loss audit conditional on saved neural outputs. This does not replay weights or prove semantics. All noncontrol families are equally weighted; controls remain in full local reports.'}, compact_arrays


def references(replication, previous, p):
    files = {}
    for seed in FITS:
        for name, path in [('checkpoint', replication / f'weights-{seed}.pt'),
                           ('calibration', replication / f'study-{seed}/calibration.json'),
                           ('continuation', previous / f'seed-{seed}.json')]:
            if not path.is_file():
                raise ValueError(f'missing {name} for fit {seed}: {path}')
            files[str(path.resolve())] = digest(path)
    return files


def run(args):
    p = protocol(); check_sources(p)
    if args.output_dir.exists():
        raise ValueError('refusing to overwrite an existing validation')
    refs = references(args.replication_dir, args.previous_dir, p)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Written BEFORE target model evaluations. Frozen helper source also checked.
    locked = {'protocol': p, 'protocol_sha256': digest(PROTOCOL),
              'runner_sha256': digest(Path(__file__)), 'reference_file_hashes': refs}
    write(args.output_dir/'LOCKED_PROTOCOL.json', locked)
    for seed in p['data_seeds']:
        print(f'FINAL VALIDATION: new input block {seed}', flush=True)
        subprocess.run([sys.executable, str(ROOT/'scripts/benchmark_local_response.py'),
            '--replication-dir', str(args.replication_dir.resolve()),
            '--previous-dir', str(args.previous_dir.resolve()),
            '--output-dir', str((args.output_dir/f'data-{seed}').resolve()),
            '--device', args.device, '--n', str(p['n_per_seed']), '--test-seed', str(seed),
            '--models', *p['models']], check=True)
    if digest(PROTOCOL) != locked['protocol_sha256'] or digest(Path(__file__)) != locked['runner_sha256']:
        raise ValueError('protocol or runner changed during validation')
    check_sources(p)
    if any(digest(Path(path)) != value for path, value in refs.items()):
        raise ValueError('reference files changed during validation')
    report, arrays = collect(args.output_dir, p)
    report['protocol_sha256'] = locked['protocol_sha256']
    output = args.output_dir/'upload'
    output.mkdir()
    with (output/'per_input_losses.npz').open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    report['per_input_losses_sha256'] = digest(output/'per_input_losses.npz')
    write(output/'summary.json', report)
    # Do not publish host-local absolute reference paths in the portable bundle.
    portable = {'protocol': p, 'protocol_sha256': locked['protocol_sha256'],
                'runner_sha256': locked['runner_sha256'], 'reference_hashes': sorted(refs.values())}
    write(output/'LOCKED_PROTOCOL.json', portable)
    for seed in p['data_seeds']:
        for fit in FITS:
            write(output/f'data-{seed}-fit-{fit}.json', read(args.output_dir/f'data-{seed}/seed-{fit}.json'))
    destination = Path.home()/'Downloads'
    destination.mkdir(parents=True, exist_ok=True)
    tag = args.output_dir.parent.name if args.output_dir.name == 'results' else args.output_dir.name
    archive = destination/f'{tag}-reports.zip'
    with ZipFile(archive, 'x', ZIP_DEFLATED) as bundle:
        for path in sorted(output.iterdir()):
            bundle.write(path, 'results/'+path.name)
    print('\nFinal validation completed. No further experiment is scheduled.')
    print(json.dumps(report['primary'], indent=2))
    print('UPLOAD:', archive)
    print('Full raw probes remain at:', args.output_dir)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replication-dir', type=Path, required=True)
    parser.add_argument('--previous-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args(argv)
    try:
        run(args)
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print('Validation stopped:', error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
