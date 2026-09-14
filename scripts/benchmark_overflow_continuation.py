"""Compare clipping with full-state overflow fallback using the previous frozen ranges.

No refitting, training, new observations, or claim-threshold selection. The same
saved test inputs and cut states isolate this codec change. This is numerical
preservation, not language interpretation or a speed benchmark.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

import numpy as np
import torch

import benchmark_state_continuation as prior
from continuation_codec import StateCodec
from overflow_state_codec import OverflowStateCodec
from universa_recurrent.neural.dual_output import ClaimPolicy, construct_output
from universa_recurrent.neural.dual_study import load_estimators, read_json, write_json
from universa_recurrent.neural.retention_study import configure, compare_outputs
from universa_recurrent.neural.train import sha256_file

FORMAT = 'universa-recurrent.overflow-continuation.v1'
SCOPE = {
    'change': 'One tag selects unchanged compact values or the ENTIRE original float32 state on overflow.',
    'frozen': 'Previous ranges, matrices, model weights, cuts, input cohort, remaining updates and claim policies.',
    'fallback_selection': 'Only the current state and frozen ranges; no labels, future state, or final answer.',
    'retained_context': 'Original observations, masks, encoder context and prior logits remain with the solver. Decoder gets none.',
    'limits': 'In-range rounding may still change outputs or decisions. No learned language, speedup, or total-memory claim.',
    'sampling': 'Previous common test cohort reused for a paired diagnostic, not fresh confirmatory evidence.',
    'bytes': 'Report all value bytes, one fallback tag, 32-byte codec identity, shared manifest and retained context separately.',
    'failure': 'Nonfinite continuations are explicit failures; no aggregate ranking when a checkpoint fails.',
}


def source():
    result = prior.source()
    scripts = Path(__file__).resolve().parent
    result['overflow_scripts'] = {name: sha256_file(scripts / name) for name in
        ('benchmark_overflow_continuation.py', 'overflow_state_codec.py')}
    return result


def same_float(actual, expected, label):
    if (actual.shape != expected.shape or not np.isfinite(expected).all() or
        not np.allclose(actual, expected, atol=prior.ATOL, rtol=prior.RTOL)):
        raise ValueError(label + ' changed; do not compare this run with the old state')


def safe_arrays(path, expected_sha):
    if sha256_file(path) != expected_sha:
        raise ValueError('previous numeric archive hash mismatch')
    # No pickle or Python objects. Only access the finite numeric arrays we need.
    return np.load(path, allow_pickle=False)


def subgroup(values, mask):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {'n': 0, 'final_change_mse': None, 'claim_changes': 0, 'max_absolute_change': None}
    return dict(n=int(mask.sum()),
        final_change_mse=float(values['final_estimate_mse_to_reference'][mask].mean()),
        claim_changes=int(values['claim_changed'][mask].sum()),
        max_absolute_change=float(values['max_absolute_change_per_example'][mask].max()))


@torch.inference_mode()
def worker(checkpoint, claim_path, previous_path, out, device_name):
    torch.set_num_threads(1)
    execution = configure(True)
    start_source = source()
    cp_hash, claim_hash, previous_hash = map(sha256_file, (checkpoint, claim_path, previous_path))
    old = read_json(previous_path)
    if (old.get('format') != prior.FORMAT + '.worker' or not old.get('all_restoration_gates_passed') or
        old['checkpoint_sha256'] != cp_hash or old['claim_calibration_sha256'] != claim_hash):
        raise ValueError('previous report does not match reference files or restoration gates')
    seed = old['training_seed']
    if type(seed) is not int or old['per_example_file'] != f'seed-{seed}.npz':
        raise ValueError('invalid prior array filename')
    arrays_path = previous_path.parent / old['per_example_file']
    engines, meta, device, reserved = load_estimators(checkpoint, device_name)
    claim = read_json(claim_path)
    if (meta['checkpoint_sha256'] != cp_hash or claim['checkpoint_sha256'] != cp_hash or
        meta['training']['seed'] != seed):
        raise ValueError('checkpoint/calibration mismatch')
    if {old['calibration_seed'], old['test_seed']} & (reserved | {claim['data']['seed']}):
        raise ValueError('codec splits overlap training or claim calibration')
    if old['calibration_seed'] == old['test_seed']:
        raise ValueError('codec splits overlap')
    by_name = {engine.name: engine for engine in engines}
    rows, arrays, manifests, gates = [], {}, {}, []
    with safe_arrays(arrays_path, old['per_example_sha256']) as data:
        obs = data['test_observed'].copy()
        masks = data['test_mask'].copy()
        truth_array, labels_array = data['test_truth'].copy(), data['test_labels'].copy()
        n = old['n']
        if (obs.dtype != np.float32 or masks.dtype != np.float32 or obs.ndim != 2 or
            masks.shape != obs.shape or len(obs) != n or truth_array.shape != obs.shape or
            not np.isfinite(obs).all() or not np.isfinite(masks).all() or
            not np.isfinite(truth_array).all() or labels_array.shape != (n,) or
            not np.isin(labels_array, [0, 1]).all()):
            raise ValueError('invalid previous input arrays')
        unique = len({row.tobytes() for row in np.concatenate((obs, masks), axis=1)})
        if unique != n:
            raise ValueError('duplicated previous inputs')
        x, m = torch.from_numpy(obs).to(device), torch.from_numpy(masks).to(device)
        truth, labels = torch.from_numpy(truth_array), torch.from_numpy(labels_array)
        arrays.update(test_observed=obs, test_mask=masks, test_truth=truth_array, test_labels=labels_array)
        for name in ('shared', 'fixed_depth_4'):
            engine = by_name[name]
            model = engine.model.eval()
            if claim['model_sources'][name] != cp_hash:
                raise ValueError('claim source mismatch')
            policy = ClaimPolicy(**claim['models'][name]['policy'])
            h = model.rollout(x, m)
            reference = construct_output(h['states'][-1], h['route_probabilities'][-1], policy, 'mixture_with_claim')
            reference = prior.as_cpu(reference)
            del h
            same_float(reference['estimate'].numpy(), data[f'{name}__reference_estimate'], 'reference estimate')
            if not np.array_equal(reference['claim_index'].numpy(), data[f'{name}__reference_claim_index']):
                raise ValueError('reference decisions changed')
            arrays[f'{name}__reference_estimate'] = reference['estimate'].numpy()
            for cut in prior.cuts_for(model.config.steps):
                point = prior.pause(model, x, m, cut)
                original = data[f'{name}__cut{cut}__original_dynamic'].copy()
                same_float(point.vector(), original, 'paused state')
                raw, steps = prior.continue_from(model, point, original, policy)
                gates.append(dict(model=name, cut=cut, checks=compare_outputs(raw, reference)))
                if steps != model.config.steps - cut:
                    raise ValueError('wrong remaining updates')
                arrays[f'{name}__cut{cut}__original_dynamic'] = original
                # Actual raw framing is kept as a storage reference, unchanged.
                raw_meta = old['codecs'][f'{name}__cut{cut}__raw_f32']
                for bits in (8, 12, 16):
                    for kind in ('named', 'rotated'):
                        key = f'{name}__cut{cut}__{kind}_{bits}bit'
                        frozen = old['codecs'][key]
                        codec = OverflowStateCodec.from_manifest(frozen)
                        site = frozen['site']
                        if (site['model'] != name or site['cut'] != cut or site['depth'] != model.config.steps or
                            site['checkpoint_sha256'] != cp_hash or site['calibration_sha256'] != old['calibration_sha256'] or
                            site['names'] != meta['names'] or frozen['calibration_n'] != old['calibration_n']):
                            raise ValueError('frozen codec site mismatch')
                        base = codec.base_codec()
                        clipped_payloads, clipping = base.encode(original)
                        clipped = base.decode(clipped_payloads)
                        # This gate ensures the baseline uses EXACTLY the previous ranges/rounding.
                        if not np.array_equal(clipped, data[key + '__restored_dynamic']):
                            raise ValueError('historical quantized baseline changed')
                        payloads, fallback = codec.encode(original)
                        restored = codec.decode(payloads)
                        if (int(fallback.sum()) != clipping['clipped_examples'] or
                            restored[fallback].tobytes() != original[fallback].tobytes() or
                            not np.array_equal(restored[~fallback], clipped[~fallback])):
                            raise ValueError('fallback or unchanged in-range decoding gate failed')
                        manifests[key] = codec.manifest()
                        arrays[key + '__fallback_mask'] = fallback
                        clipped_output = None
                        for mode, decoded in (('clip', clipped), ('overflow_safe', restored)):
                            method = f'{kind}_{bits}bit_{mode}'
                            array_key = f'{name}__cut{cut}__{method}'
                            storage = base.storage(n) if mode == 'clip' else codec.storage(payloads)
                            common = dict(model=name, cut=cut, depth=model.config.steps, method=method,
                                bits=bits, representation=kind, mode=mode, fallback_n=int(fallback.sum()),
                                storage=storage, retained_context_bytes_per_example=point.retained_bytes_per_example(),
                                original_unframed_state_bytes_per_example=original.shape[1] * 4,
                                raw_framed_record_bytes_per_example=32 + raw_meta['width'] * 4,
                                frozen_base_sha256=base.identity.hex())
                            try:
                                output, updates = prior.continue_from(model, point, decoded, policy)
                            except prior.NumericalContinuationError as exc:
                                rows.append({**common, 'status': 'nonfinite_continuation', 'failure': str(exc), 'metrics': None})
                                continue
                            if updates != steps:
                                raise ValueError('codec changed continuation budget')
                            if mode == 'clip':
                                clipped_output = prior.as_cpu(output)
                                same_float(clipped_output['estimate'].numpy(), data[key + '__final_estimate'], 'clipped continuation')
                            elif clipped_output is not None and (~fallback).any():
                                unchanged = {k: v.detach().cpu()[~fallback] for k, v in output.items()}
                                old_unchanged = {k: v[~fallback] for k, v in clipped_output.items()}
                                compare_outputs(unchanged, old_unchanged)
                            metrics, values = prior.effects(output, reference, truth, labels, original, decoded)
                            final = output['estimate'].detach().cpu().numpy()
                            per_max = np.max(np.abs(final.astype(float) - reference['estimate'].numpy()), axis=1)
                            values['max_absolute_change_per_example'] = per_max
                            if mode == 'overflow_safe' and fallback.any():
                                # Full-state fallbacks must preserve ALL output fields, not just the mean.
                                subset = {k: v.detach().cpu()[fallback] for k, v in output.items()}
                                target = {k: v[fallback] for k, v in reference.items()}
                                compare_outputs(subset, target)
                            metrics['p99_absolute_change_per_example'] = float(np.quantile(per_max, .99))
                            rows.append({**common, 'status': 'finite', 'metrics': metrics,
                                'clipped_or_fallback_subset': subgroup(values, fallback),
                                'in_range_subset': subgroup(values, ~fallback), 'executed_updates': updates})
                            values.update(restored_dynamic=decoded.copy(), final_estimate=final,
                                final_probabilities=output['probabilities'].detach().cpu().numpy(),
                                final_claim_index=output['claim_index'].detach().cpu().numpy())
                            arrays.update({array_key + '__' + field: value for field, value in values.items()})
                print(f'{name} cut {cut}/{model.config.steps}: raw gate and exact overflow restoration passed', flush=True)
    if (sha256_file(checkpoint) != cp_hash or sha256_file(claim_path) != claim_hash or
        sha256_file(previous_path) != previous_hash or sha256_file(arrays_path) != old['per_example_sha256'] or
        source() != start_source):
        raise ValueError('source/reference/report changed during run')
    array_file = out.with_suffix('.npz')
    with array_file.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return dict(format=FORMAT + '.worker', source=start_source, execution=execution,
        checkpoint_sha256=cp_hash, claim_calibration_sha256=claim_hash, training_seed=seed,
        previous_report_sha256=previous_hash, previous_arrays_sha256=old['per_example_sha256'],
        n=n, test_sha256=old['test_sha256'], calibration_sha256=old['calibration_sha256'],
        device=str(device), rows=rows, manifests=manifests, raw_gates=gates,
        all_exact_fallback_gates_passed=True, no_refitting=True, no_training=True,
        per_example_file=array_file.name, per_example_sha256=sha256_file(array_file), scope=SCOPE)


def summarize(workers):
    maps = [{(r['model'], r['cut'], r['method']): r for r in w['rows']} for w in workers]
    if any(set(m) != set(maps[0]) for m in maps):
        raise ValueError('worker cases differ')
    result = []
    for key in sorted(maps[0]):
        cases = [m[key] for m in maps]
        failures = sum(r['status'] != 'finite' for r in cases)
        result.append(dict(model=key[0], cut=key[1], method=key[2], failed_checkpoints=failures,
            mean_final_change=None if failures else statistics.mean(r['metrics']['final_estimate_mse_to_reference'] for r in cases),
            worst_absolute_change=None if failures else max(r['metrics']['max_absolute_estimate_change'] for r in cases),
            changed_claims=None if failures else sum(r['metrics']['claim_index_changed_n'] for r in cases),
            checkpoint_input_comparisons=sum(w['n'] for w in workers),
            per_checkpoint=cases))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('replication-dir', 'previous-dir', 'output-dir', 'checkpoint', 'claim-calibration', 'previous-report', 'worker-output'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = p.parse_args(argv)
    try:
        if args.worker_output:
            if not all((args.checkpoint, args.claim_calibration, args.previous_report)):
                raise ValueError('worker needs checkpoint, calibration and previous report')
            if args.worker_output.exists() or args.worker_output.with_suffix('.npz').exists():
                raise ValueError('refusing existing worker artifacts')
            write_json(args.worker_output, worker(args.checkpoint, args.claim_calibration,
                args.previous_report, args.worker_output, args.device))
            return 0
        if not all((args.replication_dir, args.previous_dir, args.output_dir)):
            raise ValueError('supply replication-dir, previous-dir and output-dir')
        if args.output_dir.exists():
            raise ValueError('refusing existing output folder')
        completed = read_json(args.previous_dir / 'summary.json')
        if completed.get('format') != prior.FORMAT + '.complete' or not completed.get('all_restoration_gates_passed'):
            raise ValueError('previous continuation run is incomplete')
        jobs = []
        for report in sorted(args.previous_dir.glob('seed-*.json')):
            old = read_json(report)
            seed = old['training_seed']
            if type(seed) is not int or report.name != f'seed-{seed}.json':
                raise ValueError('invalid prior report seed')
            cp = args.replication_dir / f'weights-{seed}.pt'
            cal = args.replication_dir / f'study-{seed}/calibration.json'
            if not cp.is_file() or not cal.is_file():
                raise ValueError(f'missing checkpoint or calibration for seed {seed}')
            jobs.append((seed, cp, cal, report))
        if not jobs or len(jobs) != completed['checkpoint_runs']:
            raise ValueError('missing prior worker reports')
        random.Random(64000).shuffle(jobs)
        start_source = source()
        args.output_dir.mkdir(parents=True)
        write_json(args.output_dir / 'PLAN.json', dict(format=FORMAT + '.plan', scope=SCOPE,
            source=start_source, bits=[8, 12, 16], job_order=[j[0] for j in jobs],
            previous_summary_sha256=sha256_file(args.previous_dir / 'summary.json')))
        workers = []
        for seed, cp, cal, report in jobs:
            out = args.output_dir / f'seed-{seed}.json'
            command = [sys.executable, str(Path(__file__).resolve()), '--checkpoint', str(cp.resolve()),
                '--claim-calibration', str(cal.resolve()), '--previous-report', str(report.resolve()),
                '--worker-output', str(out.resolve()), '--device', args.device]
            subprocess.run(command, check=True, env=dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8'))
            w = read_json(out)
            if (w['source'] != start_source or w['training_seed'] != seed or
                w['checkpoint_sha256'] != sha256_file(cp) or w['claim_calibration_sha256'] != sha256_file(cal) or
                w['previous_report_sha256'] != sha256_file(report)):
                raise ValueError('worker identity changed')
            if workers and (w['test_sha256'], w['calibration_sha256']) != (workers[0]['test_sha256'], workers[0]['calibration_sha256']):
                raise ValueError('worker cohorts disagree')
            workers.append(w)
        summary = dict(format=FORMAT + '.complete', checkpoint_runs=len(workers), source=start_source,
            rows=summarize(workers), all_exact_fallback_gates_passed=True, scope=SCOPE)
        write_json(args.output_dir / 'summary.json', summary)
        print('COMPLETE:', args.output_dir / 'summary.json', flush=True)
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print('Error:', error, file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
