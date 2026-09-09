"""Recompute a replication summary from reports, without loading model weights.

Like checking a laboratory notebook, this checks agreement between the reported
counts, timing samples, and summaries. It does NOT authenticate a GPU execution
or repeat checkpoint-bound verification. ZIP members are read, never extracted.

Usage: python scripts/audit_replication.py reports.zip --output audit.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics as stats
from zipfile import ZipFile


def read_json(data: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON field: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'nonfinite JSON constant: {value}')
    obj = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(obj, dict):
        raise ValueError('expected a JSON object')
    def finite(value):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('nonfinite number')
        if isinstance(value, dict):
            for item in value.values(): finite(item)
        if isinstance(value, list):
            for item in value: finite(item)
    finite(obj)
    return obj


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-9)


def keyed(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for row in rows:
        key = row['model'] + '/' + row['output_mode']
        require(key not in result, 'duplicate model/output row')
        result[key] = row
    return result


def audit(path: Path) -> dict:
    with ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        require(len(names) == len(set(names)), 'duplicate ZIP member')
        require(len(names) <= 500 and sum(m.file_size for m in members) <= 50_000_000,
                'report archive too large')
        for name in names:
            p = PurePosixPath(name)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in name,
                    'unsafe ZIP member')
        def raw(name):
            return archive.read('results/' + name)
        def read(name):
            return read_json(raw(name))
        summary = read('summary.json')
        seeds = summary['training_seeds']
        require(isinstance(seeds, list) and len(seeds) >= 2, 'need at least two training seeds')
        require(all(type(s) is int and s >= 0 for s in seeds), 'invalid training seed')
        require(len(set(seeds)) == len(seeds), 'duplicate training seed')
        parent = read('PLAN.json')
        require(parent['training_seeds'] == seeds, 'replication plan differs from summary')
        evaluations, benchmarks, hashes, cohorts, calibration_cohorts, versions = [], [], set(), set(), set(), set()
        source_commits = set()
        for seed in seeds:
            prefix = f'study-{seed}/'
            plan, done, ev, bm, cal, checked = [read(prefix + n + '.json') for n in
                ('PLAN', 'COMPLETED', 'eval', 'benchmark', 'calibration', 'verification')]
            require(plan['training_seed'] == seed and done.get('verified') is True
                    and checked.get('accepted') is True, 'study incomplete or checker did not accept')
            sha = plan['checkpoint_sha256']
            require(sha not in hashes, 'duplicate checkpoint is not a training replicate')
            hashes.add(sha)
            require(sha == done['checkpoint_sha256'] == cal['checkpoint_sha256'], 'checkpoint identity mismatch')
            require(ev['calibration_sha256'] == hashlib.sha256(raw(prefix+'calibration.json')).hexdigest(),
                    'calibration hash mismatch')
            require(ev['dataset'] == bm['dataset'], 'evaluation and timing cohorts differ')
            require(plan['test_seed'] == parent['test_seed'] == ev['dataset']['seed'], 'test seed mismatch')
            require(plan['calibration_seed'] == parent['calibration_seed'] == cal['data']['seed'],
                    'calibration seed mismatch')
            reserved = set(parent['training_data_seeds']) | set(plan['reserved_data_seeds'])
            require(plan['test_seed'] != plan['calibration_seed'] and
                    not {plan['test_seed'],plan['calibration_seed']} & reserved, 'data seed overlap')
            cohorts.add(ev['dataset']['sha256']); calibration_cohorts.add(cal['data']['sha256'])
            versions.add(ev['environment']['package_version']); source_commits.add(ev['environment']['git_commit'])
            e, b = keyed(ev['rows']), keyed(bm['rows'])
            require(set(e) == set(b) and len(e) == done['rows'] == done['timed_rows'], 'missing/extra timed rows')
            if evaluations:
                require(set(e) == set(evaluations[0]), 'methods differ across replicates')
            for key, row in e.items():
                require(all(b[key].get(k) == v for k,v in row.items()), 'timed/evaluated output metadata disagree')
                require(row['n'] == ev['dataset']['n'] and row['estimate_mse'] >= 0, 'invalid evaluation size/error')
                samples = b[key]['samples_ms']
                require(samples and len(samples) == bm['repeats'] and all(t > 0 for t in samples), 'invalid timing samples')
                require(close(stats.median(samples), b[key]['median_ms_for_n_examples']), 'incorrect timing median')
                if row.get('coverage') is not None:
                    n, claimed, wrong = row['n'], row['claimed_n'], row['wrong_claim_n']
                    require(0 <= wrong <= claimed <= n and row['abstained_n'] == n-claimed, 'invalid claim counts')
                    require(close(row['coverage'],claimed/n) and close(row['wrong_claim_rate_all'],wrong/n), 'incorrect claim denominator')
                    require((claimed == 0 and row['wrong_claim_rate_claimed'] is None) or
                            (claimed > 0 and close(row['wrong_claim_rate_claimed'],wrong/claimed)), 'incorrect selective error')
                if key.endswith('/mixture_with_claim'):
                    require(row['estimate_mse'] == e[key.replace('/mixture_with_claim','/mixture')]['estimate_mse'],
                            'claim policy changed mixture error')
            evaluations.append(e); benchmarks.append(b)
        require(len(cohorts) == len(calibration_cohorts) == len(versions) == len(source_commits) == 1,
                'replicates do not share data/source settings')
        require(summary['test_sha256'] in cohorts and summary['replicates'] == len(seeds), 'summary identity mismatch')
        summary_rows = {r['method']:r for r in summary['rows']}
        require(len(summary_rows) == len(summary['rows']) and set(summary_rows) == set(evaluations[0]), 'summary method mismatch')
        result_rows = []
        shared_values = [e['shared/mixture']['estimate_mse'] for e in evaluations]
        for key in sorted(evaluations[0]):
            values = [e[key]['estimate_mse'] for e in evaluations]
            supplied = summary_rows[key]['estimate_mse_across_training_seeds']
            require(values == supplied['per_seed'] and close(stats.mean(values), supplied['mean'])
                    and close(stats.stdev(values), supplied['sample_sd']), 'incorrect supplied summary')
            if not key.endswith('/mixture') and key != 'ambient/estimate_only':
                continue
            latencies = [b[key]['median_ms_for_n_examples'] for b in benchmarks]
            result_rows.append({'method':key, 'mse_mean':stats.mean(values),'mse_sample_sd':stats.stdev(values),
                'per_seed_mse':values, 'mean_of_run_median_ms':stats.mean(latencies), 'per_seed_median_ms':latencies,
                'shared_lower_mse_count':sum(a < c for a,c in zip(shared_values,values))})
        return {'format':'universa-recurrent.replication-report-audit.v1',
            'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_commit':next(iter(source_commits)), 'package_version':next(iter(versions)),
            'training_seeds':seeds,'test_cohort_sha256':next(iter(cohorts)),
            'test_seed':parent['test_seed'],'calibration_seed':parent['calibration_seed'],
            'examples_per_run':ev['dataset']['n'],'batch_size':ev['dataset']['batch_size'],
            'report_consistency_passed':True,'reported_verification_passes':len(seeds),
            'rows':result_rows,
            'scope':'Report consistency only: no model weights, GPU execution, or semantic explanation independently verified. Timing is the mean of per-run medians; SD is across training/data seeds on one common test cohort, not a confidence interval.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.archive)
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(text)
    print(text)


if __name__ == '__main__':
    main()
