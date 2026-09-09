"""Compare execution settings without retraining: like testing one car in two gears.

Each checkpoint/profile/pass runs in a fresh interpreter. Original observations,
claim cutoffs and weights are frozen; only deterministic-algorithm enforcement
changes. See docs/execution_audit.md for the measured scope and its limitations.
This launcher imports neither PyTorch nor project neural modules before spawning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

PROFILES = ('normal', 'deterministic')
COMMON_ENV = {
    'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'CUDA_LAUNCH_BLOCKING': '0',
    'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '0', 'NVIDIA_TF32_OVERRIDE': '0',
    'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1', 'PYTHONHASHSEED': '0',
}
INPUT_NAMES = ('PLAN.json', 'COMPLETED.json', 'calibration.json', 'eval.json', 'benchmark.json')
FORMAT = 'universa-recurrent.execution-audit.v1'


def read(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'duplicate JSON field: {key}')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError(f'nonfinite JSON literal: {value}')
    value = json.loads(path.read_text('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError(f'{path}: expected an object')
    return value


def write(path: Path, value: dict) -> None:
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as handle:
        handle.write(text)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def inventory(root: Path) -> list[dict]:
    """Resolve only expected local checkpoint paths, not paths supplied in reports."""
    root = root.resolve(strict=True)
    plan, summary = read(root / 'PLAN.json'), read(root / 'summary.json')
    seeds = plan['training_seeds']
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('distinct nonempty training seeds required')
    if summary['training_seeds'] != seeds or summary['replicates'] != len(seeds):
        raise ValueError('replication summary and plan disagree')
    entries = []
    cohort = None
    for seed in seeds:
        integer(seed, 'training seed', 0)
        checkpoint = root / f'weights-{seed}.pt'
        study = root / f'study-{seed}'
        for path in (checkpoint, study, *(study / name for name in INPUT_NAMES)):
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
                raise ValueError(f'input must be a real file/directory within the replication: {path}')
        p, done = read(study / 'PLAN.json'), read(study / 'COMPLETED.json')
        evaluation, benchmark = read(study / 'eval.json'), read(study / 'benchmark.json')
        calibration = read(study / 'calibration.json')
        digest = sha(checkpoint)
        if p['training_seed'] != seed or done.get('verified') is not True:
            raise ValueError('incomplete or mislabeled original study')
        if any(d['checkpoint_sha256'] != digest for d in (p, done, calibration)):
            raise ValueError('checkpoint hash mismatch; do not substitute/retrain weights')
        calibration_sha = sha(study / 'calibration.json')
        if evaluation['calibration_sha256'] != calibration_sha:
            raise ValueError('calibration hash mismatch')
        if evaluation['dataset'] != benchmark['dataset']:
            raise ValueError('old evaluation and timing used different cohorts/settings')
        data = benchmark['dataset']
        if data['seed'] != plan['test_seed'] or data['seed'] != p['test_seed']:
            raise ValueError('original test seed mismatch')
        if calibration['data']['seed'] != plan['calibration_seed']:
            raise ValueError('original calibration seed mismatch')
        integer(data['n'], 'cohort size'); integer(data['batch_size'], 'batch size')
        if cohort is not None and data != cohort:
            raise ValueError('replicates must use identical test data and batch size')
        cohort = data
        files = [checkpoint, *(study / name for name in INPUT_NAMES)]
        entries.append({'seed': seed, 'checkpoint': str(checkpoint), 'study': str(study),
                        'hashes': {str(p): sha(p) for p in files}, 'dataset': data})
    return entries


def verify_inputs(entry: dict) -> None:
    for name, expected in entry['hashes'].items():
        if sha(Path(name)) != expected:
            raise ValueError(f'input changed during audit: {name}')


def schedule(seeds: list[int], passes: int, order_seed: int) -> list[dict]:
    """Counterbalance profile order within each checkpoint; do not pool as new fits."""
    integer(passes, 'passes'); integer(order_seed, 'order seed', 0)
    rng = random.Random(order_seed)
    starting = {seed: rng.randrange(2) for seed in seeds}
    tasks = []
    for block in range(passes):
        order = list(seeds)
        rng.shuffle(order)
        for seed in order:
            modes = PROFILES if (starting[seed] + block) % 2 == 0 else PROFILES[::-1]
            for profile in modes:
                tasks.append({'seed': seed, 'pass': block, 'profile': profile,
                              'case_order_seed': order_seed + block})
    return tasks


def child_environment() -> dict[str, str]:
    return dict(os.environ, **COMMON_ENV)


def configure(profile: str) -> dict:
    """Fresh worker only. No legacy TF32 setters are mixed with the new API."""
    if profile not in PROFILES:
        raise ValueError('unknown execution profile')
    if any(os.environ.get(k) != v for k, v in COMMON_ENV.items()):
        raise ValueError('worker must be launched with the declared common environment')
    import torch
    import torch.utils.deterministic
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.backends.fp32_precision = 'ieee'
    torch.backends.cuda.matmul.fp32_precision = 'ieee'
    torch.backends.cudnn.fp32_precision = 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'ieee'
    torch.backends.cudnn.rnn.fp32_precision = 'ieee'
    torch.backends.cudnn.benchmark = False
    # Keep this additional switch identical. The use_deterministic_algorithms
    # switch below remains responsible for enforcing deterministic operations.
    torch.backends.cudnn.deterministic = False
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.use_deterministic_algorithms(profile == 'deterministic', warn_only=False)
    return execution_settings()


def execution_settings() -> dict:
    import torch
    import torch.utils.deterministic
    enabled = torch.are_deterministic_algorithms_enabled()
    return {'deterministic_algorithms': enabled,
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'fill_uninitialized_memory': torch.utils.deterministic.fill_uninitialized_memory,
            'effective_uninitialized_memory_fill': bool(enabled and torch.utils.deterministic.fill_uninitialized_memory),
            'matmul_fp32_precision': torch.backends.cuda.matmul.fp32_precision,
            'cudnn_conv_fp32_precision': torch.backends.cudnn.conv.fp32_precision,
            'cudnn_rnn_fp32_precision': torch.backends.cudnn.rnn.fp32_precision,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'environment_controls': {k: os.environ.get(k) for k in COMMON_ENV},
            'autocast': False, 'compiled': False}


def telemetry() -> dict:
    """Best-effort, read-only device snapshot outside timing; never alter clocks."""
    command = ['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,power.draw,utilization.gpu,clocks.sm',
               '--format=csv,noheader,nounits']
    try:
        value = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return {'available': value.returncode == 0, 'snapshot': value.stdout.strip(),
                'error': value.stderr.strip() if value.returncode else None}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'available': False, 'error': str(error)}


def worker(task_path: Path) -> None:
    task = read(task_path)
    entry = task['entry']; verify_inputs(entry)
    settings = configure(task['profile'])
    import numpy as np
    import torch
    from universa_recurrent.neural.dual_output import ClaimPolicy, metrics
    from universa_recurrent.neural.dual_study import (
        load_estimators, dataset_fingerprint, evaluate_and_time, cases, collect, source_identity)
    from universa_recurrent.neural.data import StructuredFlowDataset
    study = Path(entry['study'])
    frozen = read(study / 'calibration.json')
    engines, metadata, device, reserved = load_estimators(Path(entry['checkpoint']), task['device'])
    if metadata['training']['seed'] != entry['seed']:
        raise ValueError('checkpoint training seed mismatch')
    if entry['dataset']['seed'] in reserved or frozen['data']['seed'] in reserved:
        raise ValueError('study seed overlaps checkpoint training/calibration')
    structured = {e.name for e in engines if e.structured}
    if structured != set(frozen['models']) or structured != set(frozen['model_sources']):
        raise ValueError('frozen policy model set differs from checkpoint models')
    cp_sha = entry['hashes'][entry['checkpoint']]
    if metadata['checkpoint_sha256'] != cp_sha or any(s != cp_sha for s in frozen['model_sources'].values()):
        raise ValueError('policy/checkpoint binding mismatch')
    policies = {name: ClaimPolicy(**item['policy']) for name, item in frozen['models'].items()}
    data = entry['dataset']
    kwargs = {k: metadata['training'][k] for k in ('noise_std', 'observe_probability')}
    if any(kwargs[k] != data[k] for k in kwargs):
        raise ValueError('checkpoint and original test settings disagree')
    dataset = StructuredFlowDataset(data['n'], seed=data['seed'], **kwargs)
    if dataset_fingerprint(dataset) != data['sha256']:
        raise ValueError('reconstructed cohort differs from original; refusing comparison')
    before = telemetry() if device.type == 'cuda' else {'available': False, 'reason': 'CPU smoke test'}
    evaluation, benchmark = evaluate_and_time(engines, policies, dataset, device,
        batch_size=data['batch_size'], warmup=task['warmup'], repeats=task['repeats'],
        order_seed=task['case_order_seed'])
    after = telemetry() if device.type == 'cuda' else before
    if execution_settings() != settings:
        raise ValueError('execution settings changed during timing')
    # Retain all output tensors after timing to compare numerical AND discrete
    # decisions between profiles. NPZ uses ordinary arrays, not pickle.
    arrays = {}
    evaluated = {r['model'] + '/' + r['output_mode']: r for r in evaluation['rows']}
    truth, labels = dataset.truth.to(device), dataset.label.to(device)
    with torch.inference_mode():
        observed, mask = dataset.observed.to(device), dataset.mask.to(device)
        for engine, mode in cases(engines):
            out = collect(engine, observed, mask, policies.get(engine.name), mode, data['batch_size'])
            checked = metrics(out, truth, labels, mode)
            expected = evaluated[f'{engine.name}/{mode}']
            if not math.isclose(checked['estimate_mse'], expected['estimate_mse'], abs_tol=1e-8, rel_tol=1e-6):
                raise ValueError('post-timing predictions differ from evaluated quality')
            if checked.get('claimed_n') != expected.get('claimed_n') or checked.get('wrong_claim_n') != expected.get('wrong_claim_n'):
                raise ValueError('post-timing decisions differ from evaluation')
            for name, value in out.items():
                arrays[f'{engine.name}/{mode}/{name}'] = value.detach().cpu().numpy()
    output = task_path.parent
    np.savez_compressed(output / 'predictions.npz', **arrays)
    verify_inputs(entry)
    write(output / 'eval.json', evaluation)
    write(output / 'benchmark.json', benchmark)
    write(output / 'execution.json', {'format': FORMAT, 'profile': task['profile'],
        'seed': entry['seed'], 'pass': task['pass'], 'pid': os.getpid(),
        'settings': settings, 'source': source_identity(), 'device': str(device),
        'telemetry_before': before, 'telemetry_after': after,
        'inputs_unchanged': True, 'input_hashes': entry['hashes'],
        'predictions_sha256': sha(output / 'predictions.npz'),
        'calibration': 'Frozen original cutoffs; not recalibrated in either profile.'})


def compare_arrays(left: Path, right: Path) -> dict:
    """Same output structure, tight numeric tolerance, exact discrete decisions."""
    import numpy as np
    rows = {}
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise ValueError('prediction field sets differ')
        for name in a.files:
            x, y = a[name], b[name]
            if x.shape != y.shape or x.dtype != y.dtype:
                raise ValueError(f'prediction shape/dtype mismatch: {name}')
            if np.issubdtype(x.dtype, np.floating):
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError('nonfinite predictions')
                delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
                detail = {'equal': bool(np.allclose(x, y, atol=1e-6, rtol=1e-5)),
                          'max_absolute_difference': float(delta.max(initial=0))}
            else:
                detail = {'equal': bool(np.array_equal(x, y)), 'different_elements': int(np.count_nonzero(x != y))}
            key, field = name.rsplit('/', 1)
            rows.setdefault(key, {'equal': True, 'fields': {}})
            rows[key]['fields'][field] = detail
            rows[key]['equal'] = rows[key]['equal'] and detail['equal']
    return {'atol': 1e-6, 'rtol': 1e-5, 'discrete_decisions': 'exact', 'rows': rows,
            'all_outputs_match': all(r['equal'] for r in rows.values())}


def summarize(output: Path, tasks: list[dict]) -> dict:
    groups = {}
    for task in tasks:
        directory = output / task['directory']
        record = read(directory / 'execution.json')
        if record['profile'] != task['profile'] or record['seed'] != task['seed'] or record['pass'] != task['pass']:
            raise ValueError('worker result is mislabeled')
        if record['predictions_sha256'] != sha(directory / 'predictions.npz'):
            raise ValueError('saved predictions changed')
        groups.setdefault((task['seed'], task['pass']), {})[task['profile']] = directory
    comparisons = []
    sources, datasets = set(), set()
    for (seed, block), group in groups.items():
        if set(group) != set(PROFILES):
            raise ValueError('missing profile; do not summarize a partial pair')
        a, b = (read(group[p] / 'execution.json') for p in PROFILES)
        if a['input_hashes'] != b['input_hashes'] or a['source'] != b['source'] or a['device'] != b['device']:
            raise ValueError('paired runs changed inputs, source, or device')
        sa, sb = dict(a['settings']), dict(b['settings'])
        if sa.pop('deterministic_algorithms') is not False or sb.pop('deterministic_algorithms') is not True:
            raise ValueError('profiles did not run their declared deterministic settings')
        sa.pop('effective_uninitialized_memory_fill'); sb.pop('effective_uninitialized_memory_fill')
        if sa != sb:
            raise ValueError('paired runs differ in non-profile execution controls')
        sources.add(json.dumps(a['source'], sort_keys=True))
        reports = [read(group[p] / 'benchmark.json') for p in PROFILES]
        if reports[0]['dataset'] != reports[1]['dataset']:
            raise ValueError('paired cohorts differ')
        datasets.add(json.dumps(reports[0]['dataset'], sort_keys=True))
        comparison = compare_arrays(group['normal'] / 'predictions.npz', group['deterministic'] / 'predictions.npz')
        maps = [{r['model'] + '/' + r['output_mode']: r for r in report['rows']} for report in reports]
        if set(maps[0]) != set(maps[1]) or set(maps[0]) != set(comparison['rows']):
            raise ValueError('timed modes differ from compared predictions')
        for key in sorted(maps[0]):
            normal, deterministic = maps[0][key], maps[1][key]
            t0, t1 = normal['median_ms_for_n_examples'], deterministic['median_ms_for_n_examples']
            if not all(math.isfinite(t) and t > 0 for t in (t0, t1)):
                raise ValueError('invalid timings')
            checks = comparison['rows'][key]
            comparisons.append({'seed': seed, 'pass': block, 'method': key,
                'normal_ms': t0, 'deterministic_ms': t1,
                'deterministic_over_normal_ms_ratio': t1 / t0,
                'normal_mse': normal['estimate_mse'], 'deterministic_mse': deterministic['estimate_mse'],
                'matched_output_ratio': t1 / t0 if checks['equal'] else None,
                'output_check': checks})
    if len(sources) != 1 or len(datasets) != 1:
        raise ValueError('source/cohort changed across checkpoint pairs')
    aggregate = []
    for key in sorted({r['method'] for r in comparisons}):
        rows = [r for r in comparisons if r['method'] == key]
        ratios = [r['matched_output_ratio'] for r in rows if r['matched_output_ratio'] is not None]
        aggregate.append({'method': key, 'normal_median_ms': statistics.median(r['normal_ms'] for r in rows),
            'deterministic_median_ms': statistics.median(r['deterministic_ms'] for r in rows),
            'matched_pairs': len(ratios), 'pairs': len(rows),
            'matched_ratio_median': statistics.median(ratios) if ratios else None,
            'matched_ratio_range': [min(ratios), max(ratios)] if ratios else None})
    return {'format': FORMAT + '.summary', 'status': 'complete',
        'all_profile_outputs_match': all(r['output_check']['equal'] for r in comparisons),
        'rows': aggregate, 'paired_measurements': comparisons, 'source': json.loads(next(iter(sources))),
        'dataset': json.loads(next(iter(datasets))),
        'scope': 'Existing fixed-depth inference only. Same weights, inputs, frozen cutoffs and precision. No training or GPU speed claim before measurements.',
        'limits': ['Normal means deterministic enforcement off, not a guarantee of nondeterminism or fastest execution.',
            'Common IEEE precision, one host thread and cuBLAS workspace are controlled, not reconstructed historical settings.',
            'Includes normal deterministic memory-fill side effects; does not isolate an individual kernel cause.',
            'Includes rollout allocation and outputs; excludes model loading, transfers, calibration, Lingua and checking.',
            'Passes and timing repeats are repeated measurements, not independent model training replicates.',
            'Telemetry snapshots cannot rule out all thermal or background-load effects.']}


def run(root: Path, output: Path, *, device: str = 'cuda', passes: int = 2,
        warmup: int = 5, repeats: int = 20, order_seed: int = 41000) -> Path:
    integer(passes, 'passes'); integer(warmup, 'warmup', 0); integer(repeats, 'repeats')
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    root = root.resolve(strict=True); output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError('write new results outside the original replication folder')
    entries = inventory(root)  # all checkpoints/hashes checked BEFORE starting
    tasks = schedule([e['seed'] for e in entries], passes, order_seed)
    output.mkdir(parents=True, exist_ok=False)
    by_seed = {e['seed']: e for e in entries}
    for task in tasks:
        task['directory'] = f"seed-{task['seed']}-pass-{task['pass']}-{task['profile']}"
        task.update({'entry': by_seed[task['seed']], 'warmup': warmup, 'repeats': repeats, 'device': device})
    write(output / 'PLAN.json', {'format': FORMAT, 'original_replication': str(root),
        'tasks': tasks, 'runner_sha256': sha(Path(__file__)), 'common_environment': COMMON_ENV,
        'no_retraining': True, 'calibration': 'reuse original files without recalibration',
        'output_equivalence': {'atol': 1e-6, 'rtol': 1e-5, 'discrete': 'exact'}})
    for index, task in enumerate(tasks, 1):
        directory = output / task['directory']; directory.mkdir()
        write(directory / 'task.json', task)
        print(f"[{index}/{len(tasks)}] seed {task['seed']} | pass {task['pass'] + 1} | {task['profile']}", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), '_worker', str(directory / 'task.json')]
        with (directory / 'worker.log').open('x') as log:
            completed = subprocess.run(command, env=child_environment(), stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Worker failed; original files unchanged. See {directory / 'worker.log'}")
    for entry in entries:
        verify_inputs(entry)
    summary = summarize(output, tasks)
    write(output / 'summary.json', summary)
    write(output / 'COMPLETED.json', {'format': FORMAT + '.complete', 'workers': len(tasks),
        'all_profile_outputs_match': summary['all_profile_outputs_match'], 'inputs_unchanged': True})
    archive = output.parent / (output.name + '-reports.zip')
    with ZipFile(archive, 'x', ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.log'):
                bundle.write(path, path.relative_to(output))
    print('\nMethod                              normal ms   deterministic ms   matched pairs')
    for row in summary['rows']:
        if row['method'].endswith(('/mixture', '/estimate_only')):
            print(f"{row['method']:<36} {row['normal_median_ms']:9.3f} {row['deterministic_median_ms']:18.3f}   {row['matched_pairs']}/{row['pairs']}")
    print(f"\nAll outputs match within declared checks: {summary['all_profile_outputs_match']}")
    print(f'Report ZIP (no weights or prediction arrays): {archive}')
    return archive


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('replication_dir', type=Path, help='Original results/ folder containing weights-*.pt and study-*')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--order-seed', type=int, default=41000)
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) == 2 and args[0] == '_worker':
            worker(Path(args[1])); return 0
        a = parser.parse_args(args)
        run(a.replication_dir, a.output_dir, device=a.device, passes=a.passes,
            warmup=a.warmup, repeats=a.repeats, order_seed=a.order_seed)
        return 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
