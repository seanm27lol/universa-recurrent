"""Run the matched dual-output study. Example: python -m universa_recurrent.neural.dual_cli --help."""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import statistics
import sys

from .dual_study import (read_json, run_study, write_json, reserved_seeds, source_identity,
                         positive_int, _seed)


def summarize_runs(paths: list[Path]) -> dict:
    """Summarize training replicates, never pretend n*seeds examples are independent fits."""
    if not paths or len(set(map(str, paths))) != len(paths):
        raise ValueError('distinct study paths required')
    studies = [read_json(p / 'eval.json') for p in paths]
    plans = [read_json(p / 'PLAN.json') for p in paths]
    seeds, shas = [], []
    for p, plan in zip(paths, plans):
        completed = read_json(p / 'COMPLETED.json')
        if completed.get('verified') is not True:
            raise ValueError('refuse incomplete studies')
        seed = plan.get('training_seed')
        if seed is None:
            raise ValueError('replication needs explicit training_seed in each plan')
        seeds.append(seed); shas.append(plan['checkpoint_sha256'])
    if len(set(seeds)) != len(seeds) or len(set(shas)) != len(shas):
        raise ValueError('duplicate training seed/checkpoint is not independent replication')
    if len({s['dataset']['sha256'] for s in studies}) != 1:
        raise ValueError('replicate summary requires the same held-out test cohort')
    maps = [{r['model']+'/'+r['output_mode']: r for r in s['rows']} for s in studies]
    keys = set(maps[0])
    if any(set(m) != keys for m in maps):
        raise ValueError('model/output modes differ across replicates')
    def describe(values):
        return {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values) if len(values)>1 else None,
                'min': min(values), 'max': max(values), 'per_seed': values}
    rows = []
    for key in sorted(keys):
        row = {'method': key, 'estimate_mse_across_training_seeds': describe([m[key]['estimate_mse'] for m in maps])}
        if all(m[key].get('coverage') is not None for m in maps):
            row['coverage_across_training_seeds'] = describe([m[key]['coverage'] for m in maps])
        rows.append(row)
    differences = []
    for control in ('direct/mixture', 'untied/mixture', 'ambient/estimate_only'):
        if control in keys:
            values = [m['shared/mixture']['estimate_mse'] - m[control]['estimate_mse'] for m in maps]
            differences.append({'comparison': 'shared/mixture minus '+control, **describe(values)})
    return {'format': 'universa-recurrent.dual-replication.v1', 'training_seeds': seeds,
            'replicates': len(paths), 'test_sha256': studies[0]['dataset']['sha256'], 'rows': rows,
            'paired_training_seed_differences': differences,
            'note': 'Descriptive independent-training spread on one shared test cohort; no significance or universal superiority claim.'}


def replicate(args) -> dict:
    from .v2_train import train_v2_checkpoint
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('training seeds must be unique')
    for seed in args.seeds:
        _seed(seed)
    reserved = {seed + offset for seed in args.seeds for offset in (1,2)}
    if len(reserved) != 2 * len(args.seeds):
        raise ValueError('training/calibration seed schedules overlap between replicates')
    if {args.calibration_seed, args.test_seed} & reserved or args.calibration_seed == args.test_seed:
        raise ValueError('study data seeds overlap the declared training splits')
    if args.output_dir.exists():
        raise ValueError('refusing to overwrite replication folder')
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'PLAN.json', {'training_seeds': args.seeds,
        'calibration_seed': args.calibration_seed, 'test_seed': args.test_seed,
        'training_data_seeds': sorted(reserved), 'source': source_identity(),
        'scope': 'Existing v2 architectures and objectives; output policies compared symmetrically.'})
    paths = []
    for seed in args.seeds:
        checkpoint = args.output_dir/f'weights-{seed}.pt'
        train_v2_checkpoint(checkpoint, device_name=args.device, seed=seed,
            train_size=args.train_size, calibration_size=args.calibration_size,
            epochs=args.epochs, batch_size=args.train_batch_size,
            hidden_dim=args.hidden_dim, candidate_embedding_dim=8, steps=args.steps,
            include_controls=True)
        study = args.output_dir/f'study-{seed}'
        run_study(checkpoint, study, device_name=args.device,
            calibration_seed=args.calibration_seed, test_seed=args.test_seed,
            calibration_size=args.calibration_size, n=args.n, batch_size=args.batch_size,
            target_coverage=args.coverage, warmup=args.warmup, repeats=args.repeats,
            order_seed=args.order_seed)
        paths.append(study)
    summary = summarize_runs(paths)
    write_json(args.output_dir/'summary.json', summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    compare = commands.add_parser('compare', help='Reuse an existing v2 checkpoint; no training or weight changes')
    compare.add_argument('--checkpoint', required=True, type=Path)
    compare.add_argument('--v1-checkpoint', type=Path, help='Optional historical v1 diagnostic on the SAME examples; different training remains a confound')
    repeat = commands.add_parser('replicate', help='Train existing architectures on multiple seeds, then compare outputs fairly')
    repeat.add_argument('--seeds', nargs='+', type=int, default=[6100,7100,8100])
    repeat.add_argument('--train-size', type=int, default=20000)
    repeat.add_argument('--epochs', type=int, default=20)
    repeat.add_argument('--train-batch-size', type=int, default=512)
    repeat.add_argument('--hidden-dim', type=int, default=64)
    repeat.add_argument('--steps', type=int, default=8)
    for p in (compare, repeat):
        p.add_argument('--output-dir', type=Path, required=True)
        p.add_argument('--device', choices=['cpu','cuda','auto'], default='auto')
        p.add_argument('--calibration-seed', type=int, default=21000)
        p.add_argument('--test-seed', type=int, default=22000)
        p.add_argument('--calibration-size', type=int, default=4000)
        p.add_argument('--n', type=int, default=4000)
        p.add_argument('--batch-size', type=int, default=1024)
        p.add_argument('--coverage', type=float, default=0.75)
        p.add_argument('--warmup', type=int, default=5)
        p.add_argument('--repeats', type=int, default=20)
        p.add_argument('--order-seed', type=int, default=23000)
    verify = commands.add_parser('verify', help='Check mixture arithmetic and the separately scoped structural claim')
    verify.add_argument('record', type=Path)
    verify.add_argument('--checkpoint', type=Path)
    verify.add_argument('--calibration', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'verify':
            from .dual_lingua import verify_record
            result = verify_record(read_json(args.record), checkpoint=args.checkpoint, calibration=args.calibration)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result['accepted'] else 1
        if args.command == 'replicate':
            result = replicate(args)
        else:
            result = run_study(args.checkpoint, args.output_dir, device_name=args.device,
                calibration_seed=args.calibration_seed, test_seed=args.test_seed,
                calibration_size=args.calibration_size, n=args.n, batch_size=args.batch_size,
                target_coverage=args.coverage, warmup=args.warmup, repeats=args.repeats,
                order_seed=args.order_seed, v1_checkpoint=args.v1_checkpoint)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
