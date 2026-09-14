"""Test a frozen state codec on new readings, not the examples that motivated it.

A ruler tested on yesterday's readings must also handle tomorrow's noisier or
incomplete readings. Nothing is refitted here: weights, codec ranges, bit widths,
claim thresholds and remaining computation are fixed. This is a numerical
preservation test, not a language interpreter or a performance benchmark.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

import numpy as np
import torch

import benchmark_state_continuation as prior
from overflow_state_codec import OverflowStateCodec
from universa_recurrent.neural.data import StructuredFlowDataset
from universa_recurrent.neural.dual_output import ClaimPolicy, construct_output
from universa_recurrent.neural.dual_study import load_estimators, read_json, write_json, dataset_fingerprint
from universa_recurrent.neural.retention_study import configure, compare_outputs
from universa_recurrent.neural.train import sha256_file

FORMAT = 'universa-recurrent.codec-generalization.v1'
MODELS = ('shared', 'fixed_depth_4')
BITS = (8, 12, 16)
CONDITIONS = ('matched', 'noise_x3', 'sparse_040')
DEFAULT_SEEDS = (71000, 72000)
# Documented evaluation cohorts are not fresh confirmation cohorts.
PREVIOUS_EVAL_SEEDS = frozenset((9000, 9001, 9100, 12000, 12001, 12100, 22000,
                                31000, 32000, 33000, 36000, 46000, 61000, 62000))
SCOPE = {
    'primary_question': 'Does frozen overflow-safe 16-bit continuation preserve outputs and decisions on previously unused input seeds?',
    'comparisons': 'Frozen named 8/12/16-bit codecs; raw restoration, zero-state and other-input-state controls. No new rotation comparison.',
    'frozen': 'Original checkpoint, codec-calibration ranges, candidate spaces, claim policies, cut positions and remaining updates.',
    'conditions': 'Training-distribution parameters; three times its noise standard deviation; observation probability 0.40 with the original generator mask repair.',
    'pairing': 'Two independent seed blocks, shared across checkpoints. Conditions within a seed have identical latent truth/labels, not independent problems. Sparse-mask repair can alter the noise RNG sequence.',
    'decoder': 'Numerical bytes and pinned manifest only. Original input, mask, encoded context and prior logits remain with the solver, not the decoder.',
    'claims': 'Changed claims measure disagreement with the original model, not truth. Actual task error and wrong generating labels are reported separately.',
    'limits': 'No universal error bound, semantic explanation, learned language, causal-abstraction proof, inference speedup or total-memory saving is established.',
    'failures': 'Nonfinite lossy/control continuations are kept as failed cases; a failed case prevents an aggregate ranking. Raw-restoration failures stop the run.',
    'bytes': 'Value bytes, fallback tag, codec identity, shared metadata and retained context count separately. No GPU memory benchmark.',
}


def source():
    result = prior.source()
    here = Path(__file__).resolve().parent
    result['generalization_scripts'] = {n: sha256_file(here / n) for n in
        ('benchmark_codec_generalization.py', 'overflow_state_codec.py')}
    return result


def settings_for(training: dict, condition: str) -> dict:
    if condition not in CONDITIONS:
        raise ValueError('unknown condition')
    noise = float(training['noise_std'])
    observe = float(training['observe_probability'])
    if condition == 'noise_x3':
        noise *= 3.0
    if condition == 'sparse_040':
        observe = .40
    if not np.isfinite(noise) or noise < 0 or not 0 < observe <= 1:
        raise ValueError('invalid dataset parameters')
    return dict(noise_std=noise, observe_probability=observe)


def validate_seeds(seeds, forbidden=()):
    if (not seeds or len(set(seeds)) != len(seeds) or
        any(type(s) is not int or s < 0 for s in seeds)):
        raise ValueError('distinct nonnegative data seeds required')
    if set(seeds) & (set(forbidden) | PREVIOUS_EVAL_SEEDS):
        raise ValueError('fresh data seed overlaps a previous or reserved split')


def input_rows(observed, mask):
    joined = np.ascontiguousarray(np.concatenate((observed.numpy(), mask.numpy()), axis=1))
    return {row.tobytes() for row in joined}


def frozen_codec(old, model, name, cut, bits, cp_hash, names):
    key = f'{name}__cut{cut}__named_{bits}bit'
    manifest = old['codecs'][key]
    expected = dict(checkpoint_sha256=cp_hash, model=name, cut=cut,
        depth=model.config.steps, num_structures=model.config.num_structures,
        latent_dim=model.config.latent_dim, names=names,
        calibration_sha256=old['calibration_sha256'])
    if (manifest['site'] != expected or manifest['kind'] != 'named' or
        manifest['bits'] != bits or manifest['calibration_n'] != old['calibration_n']):
        raise ValueError('frozen codec site/bit budget differs from original report')
    return OverflowStateCodec.from_manifest(manifest)


def subset_outputs(output, mask):
    return {k: v.detach().cpu()[mask] for k, v in output.items()}


def measure(output, reference, truth, labels, original, restored):
    metrics, _ = prior.effects(output, reference, truth, labels, original, restored)
    estimate = output['estimate'].detach().cpu().numpy()
    ref = reference['estimate'].numpy()
    change = np.abs(estimate.astype(np.float64) - ref.astype(np.float64)).max(1)
    metrics['p99_absolute_change_per_example'] = float(np.quantile(change, .99))
    metrics['reference_claimed_n'] = int(reference['claim_mask'].sum())
    metrics['reference_wrong_claim_n'] = int((reference['claim_mask'] &
                                               (reference['top_route'] != labels)).sum())
    arrays = dict(final_estimate=estimate, final_probabilities=output['probabilities'].detach().cpu().numpy(),
                  final_claim_index=output['claim_index'].detach().cpu().numpy(), restored_dynamic=restored.copy())
    return metrics, arrays


@torch.inference_mode()
def worker(args):
    torch.set_num_threads(1)
    execution = configure(True)
    identities = source()
    refs = (args.checkpoint, args.claim_calibration, args.previous_report)
    ref_hashes = tuple(map(sha256_file, refs))
    cp_hash, claim_hash, old_hash = ref_hashes
    old = read_json(args.previous_report)
    if (old.get('format') != prior.FORMAT + '.worker' or
        old.get('all_restoration_gates_passed') is not True or
        old['checkpoint_sha256'] != cp_hash or old['claim_calibration_sha256'] != claim_hash):
        raise ValueError('original continuation report does not match reference files')
    engines, meta, device, reserved = load_estimators(args.checkpoint, args.device)
    claims = read_json(args.claim_calibration)
    if (meta['checkpoint_sha256'] != cp_hash or claims['checkpoint_sha256'] != cp_hash or
        meta['training']['seed'] != old['training_seed']):
        raise ValueError('checkpoint identity mismatch')
    forbidden = set(reserved) | {claims['data']['seed'], old['calibration_seed'], old['test_seed'], old['training_seed']}
    if 'test_seed_reserved' in claims:
        forbidden.add(claims['test_seed_reserved'])
    validate_seeds(args.data_seeds, forbidden)
    old_npz = args.previous_report.parent / old['per_example_file']
    if old['per_example_file'] != f"seed-{old['training_seed']}.npz" or sha256_file(old_npz) != old['per_example_sha256']:
        raise ValueError('old input array identity mismatch')
    with np.load(old_npz, allow_pickle=False) as saved:
        old_inputs = input_rows(torch.from_numpy(saved['test_observed']), torch.from_numpy(saved['test_mask']))
    by_name = {e.name: e for e in engines}
    if not set(args.models) <= set(by_name):
        raise ValueError('missing recurrent models')
    codecs, policies = {}, {}
    for name in args.models:
        engine = by_name[name]
        if claims['model_sources'][name] != cp_hash:
            raise ValueError('claim-policy source mismatch')
        policies[name] = ClaimPolicy(**claims['models'][name]['policy'])
        for cut in prior.cuts_for(engine.model.config.steps):
            for bits in BITS:
                key = f'{name}__cut{cut}__named_{bits}bit'
                codecs[key] = frozen_codec(old, engine.model, name, cut, bits, cp_hash, meta['names'])
    rows, arrays, gates, cohorts = [], {}, [], []
    previous_seed_inputs = set(old_inputs)
    for seed in args.data_seeds:
        latent_reference = None
        this_seed_inputs = set()
        for condition in args.conditions:
            params = settings_for(meta['training'], condition)
            data = StructuredFlowDataset(args.n, seed=seed, **params)
            seen = input_rows(data.observed, data.mask)
            if len(seen) != args.n:
                raise ValueError('duplicate inputs inside a cohort')
            # Conditions are paired, so unchanged observations across conditions are
            # allowed and counted, not misrepresented as fresh independent inputs.
            if seen & previous_seed_inputs:
                raise ValueError('new observations duplicate a previous seed cohort')
            overlap = len(seen & this_seed_inputs)
            this_seed_inputs.update(seen)
            truth_identity = (data.truth.numpy().tobytes(), data.label.numpy().tobytes())
            if latent_reference is not None and truth_identity != latent_reference:
                raise ValueError('conditions did not preserve the latent problem/label pairing')
            latent_reference = truth_identity
            cohort = f'data{seed}__{condition}'
            cohorts.append(dict(key=cohort, data_seed=seed, condition=condition, n=args.n,
                parameters=params, dataset_sha256=dataset_fingerprint(data),
                distinct_observed_mask_count=len(seen),
                repeated_observations_from_earlier_conditions=overlap,
                mean_observed_coordinates=float(data.mask.sum(1).mean())))
            for key, value in dict(observed=data.observed, mask=data.mask, truth=data.truth, labels=data.label).items():
                arrays[f'{cohort}__{key}'] = value.numpy()
            x, m = data.observed.to(device), data.mask.to(device)
            for name in args.models:
                model = by_name[name].model.eval()
                policy = policies[name]
                h = model.rollout(x, m)
                reference = prior.as_cpu(construct_output(h['states'][-1], h['route_probabilities'][-1],
                                                       policy, 'mixture_with_claim'))
                del h
                model_key = f'{cohort}__{name}'
                for key in ('estimate', 'probabilities', 'claim_index', 'top_route'):
                    arrays[f'{model_key}__reference_{key}'] = reference[key].numpy()
                for cut in prior.cuts_for(model.config.steps):
                    point = prior.pause(model, x, m, cut)
                    original = point.vector()
                    site = f'{model_key}__cut{cut}'
                    arrays[site + '__original_dynamic'] = original
                    raw, steps = prior.continue_from(model, point, original, policy)
                    checks = compare_outputs(raw, reference)
                    if steps != model.config.steps - cut:
                        raise ValueError('wrong raw continuation budget')
                    gates.append(dict(cohort=cohort, model=name, cut=cut, checks=checks))
                    choices = [('raw_state', original, None, None),
                               ('zero_dynamic', np.zeros_like(original), None, None),
                               ('other_input_dynamic', np.roll(original, 1, axis=0).copy(), None, None)]
                    for bits in BITS:
                        key = f'{name}__cut{cut}__named_{bits}bit'
                        codec = codecs[key]
                        payloads, fallback = codec.encode(original)
                        restored = codec.decode(payloads)
                        clipped_payloads, _ = codec.base_codec().encode(original)
                        clipped = codec.base_codec().decode(clipped_payloads)
                        if (restored[fallback].tobytes() != original[fallback].tobytes() or
                            not np.array_equal(restored[~fallback], clipped[~fallback])):
                            raise ValueError('fallback or in-range decode was not preserved')
                        storage = codec.storage(payloads)
                        choices.append((f'named_{bits}bit_safe', restored, fallback, storage))
                    for method, restored, fallback, storage in choices:
                        row = dict(cohort=cohort, data_seed=seed, condition=condition, model=name, cut=cut,
                            depth=model.config.steps, method=method, n=args.n,
                            fallback_n=None if fallback is None else int(fallback.sum()), storage=storage,
                            retained_context_bytes_per_example=point.retained_bytes_per_example(),
                            unframed_dynamic_bytes_per_example=original.shape[1] * 4)
                        try:
                            out, used = prior.continue_from(model, point, restored, policy)
                        except prior.NumericalContinuationError as error:
                            if method == 'raw_state':
                                raise
                            rows.append(dict(**row, status='nonfinite_continuation', metrics=None, failure=str(error)))
                            continue
                        if used != steps:
                            raise ValueError('a representation changed the remaining budget')
                        fallback_check = None
                        if fallback is not None and fallback.any():
                            fallback_check = compare_outputs(subset_outputs(out, fallback), subset_outputs(reference, fallback))
                        metrics, values = measure(out, reference, data.truth, data.label, original, restored)
                        if method == 'raw_state':
                            compare_outputs(out, reference)
                        if fallback is not None:
                            values['fallback_mask'] = fallback
                        key = site + '__' + method
                        for field, value in values.items():
                            arrays[key + '__' + field] = value
                        rows.append(dict(**row, status='finite', metrics=metrics, executed_updates=used,
                                         fallback_output_checks=fallback_check))
                    if not np.array_equal(original, point.vector()):
                        raise ValueError('original cut state was mutated')
                print(f"seed {old['training_seed']}: {cohort}/{name}: raw gates passed; all codecs tested", flush=True)
        previous_seed_inputs.update(this_seed_inputs)
    if (tuple(map(sha256_file, refs)) != ref_hashes or source() != identities or
        sha256_file(old_npz) != old['per_example_sha256']):
        raise ValueError('reference/source changed during experiment')
    path = args.worker_output.with_suffix('.npz')
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)
    return dict(format=FORMAT + '.worker', training_seed=old['training_seed'], source=identities,
        checkpoint_sha256=cp_hash, claim_calibration_sha256=claim_hash, previous_report_sha256=old_hash,
        previous_input_arrays_sha256=old['per_example_sha256'], reserved_seeds=sorted(forbidden | PREVIOUS_EVAL_SEEDS),
        execution=execution, device=str(device), data_seeds=args.data_seeds, n=args.n, cohorts=cohorts,
        rows=rows, codecs={key: codec.manifest() for key, codec in codecs.items()},
        policies={key: value.as_dict() for key,value in policies.items()}, raw_gates=gates,
        all_raw_restoration_gates_passed=True, no_refitting=True, no_training=True,
        per_example_file=path.name, per_example_sha256=sha256_file(path), scope=SCOPE)


def summarize(workers):
    """Keep independent seed blocks visible; do not pool repetitions as new data."""
    maps = [{(r['data_seed'], r['condition'], r['model'], r['cut'], r['method']): r
             for r in w['rows']} for w in workers]
    if not maps or any(set(m) != set(maps[0]) for m in maps):
        raise ValueError('workers have different cases')
    result = []
    for key in sorted(maps[0]):
        cases = [m[key] for m in maps]
        failed = sum(r['status'] != 'finite' for r in cases)
        good = not failed
        result.append(dict(data_seed=key[0], condition=key[1], model=key[2], cut=key[3], method=key[4],
            checkpoint_input_comparisons=sum(r['n'] for r in cases), failed_checkpoints=failed,
            mean_final_mse=None if not good else statistics.mean(r['metrics']['final_estimate_mse_to_reference'] for r in cases),
            worst_component_change=None if not good else max(r['metrics']['max_absolute_estimate_change'] for r in cases),
            claims_changed=None if not good else sum(r['metrics']['claim_index_changed_n'] for r in cases),
            routes_changed=None if not good else sum(r['metrics']['top_route_changed_n'] for r in cases),
            fallback_n=None if cases[0]['fallback_n'] is None else sum(r['fallback_n'] for r in cases),
            mean_task_mse=None if not good else statistics.mean(r['metrics']['task_mse'] for r in cases),
            per_checkpoint=[dict(training_seed=w['training_seed'], status=r['status'], metrics=r['metrics'],
                                 fallback_n=r['fallback_n'], storage=r['storage']) for w,r in zip(workers,cases)]))
    return result


def parent(args):
    if args.output_dir.exists():
        raise ValueError('refusing to overwrite output directory')
    old_summary = read_json(args.previous_dir / 'summary.json')
    if (old_summary.get('format') != prior.FORMAT + '.complete' or
        old_summary.get('all_restoration_gates_passed') is not True):
        raise ValueError('original continuation study is incomplete')
    jobs = []
    for report in sorted(args.previous_dir.glob('seed-*.json')):
        old = read_json(report); seed = old['training_seed']
        if type(seed) is not int or report.name != f'seed-{seed}.json':
            raise ValueError('invalid worker seed')
        cp = args.replication_dir / f'weights-{seed}.pt'
        cal = args.replication_dir / f'study-{seed}/calibration.json'
        if not cp.is_file() or not cal.is_file():
            raise ValueError('missing frozen checkpoint or claim calibration')
        jobs.append((seed,cp,cal,report,tuple(map(sha256_file,(cp,cal,report)))))
    if (not jobs or len(jobs) != old_summary['checkpoint_runs'] or
        len(set(j[4][0] for j in jobs)) != len(jobs)):
        raise ValueError('missing or duplicated original checkpoints')
    random.Random(73000).shuffle(jobs)
    identities = source()
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'PLAN.json',dict(format=FORMAT+'.plan', source=identities, scope=SCOPE,
        n=args.n, data_seeds=args.data_seeds, conditions=args.conditions, models=args.models, bits=list(BITS),
        previous_summary_sha256=sha256_file(args.previous_dir/'summary.json'),
        job_order=[j[0] for j in jobs], device=args.device, no_training=True, no_refitting=True))
    workers=[]
    for seed,cp,cal,report,hashes in jobs:
        output=args.output_dir/f'seed-{seed}.json'
        cmd=[sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(cp.resolve()),
            '--claim-calibration',str(cal.resolve()),'--previous-report',str(report.resolve()),
            '--worker-output',str(output.resolve()),'--device',args.device,'--n',str(args.n),
            '--data-seeds',*map(str,args.data_seeds),'--conditions',*args.conditions,'--models',*args.models]
        subprocess.run(cmd,check=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        w=read_json(output)
        if (w['training_seed']!=seed or w['source']!=identities or tuple(map(sha256_file,(cp,cal,report)))!=hashes or
            (w['checkpoint_sha256'],w['claim_calibration_sha256'],w['previous_report_sha256'])!=hashes or
            sha256_file(output.with_suffix('.npz'))!=w['per_example_sha256']):
            raise ValueError('worker/source/reference identity mismatch')
        if workers and w['cohorts']!=workers[0]['cohorts']:
            raise ValueError('new input cohorts differ across checkpoints')
        workers.append(w)
    result=dict(format=FORMAT+'.complete',source=identities,scope=SCOPE,checkpoint_runs=len(workers),
        worker_files=[f'seed-{w["training_seed"]}.json' for w in workers],cohorts=workers[0]['cohorts'],
        all_raw_restoration_gates_passed=all(w['all_raw_restoration_gates_passed'] for w in workers),
        all_successful_fallback_subset_checks_passed=True, rows=summarize(workers),
        failed_cases=sum(r['status']!='finite' for w in workers for r in w['rows']),
        warning='An observed zero decision-change count is not a universal guarantee. Shifted accuracy and preservation are different metrics.')
    write_json(args.output_dir/'summary.json',result)
    print('COMPLETE:',args.output_dir/'summary.json',flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('replication-dir','previous-dir','output-dir','checkpoint','claim-calibration','previous-report','worker-output'):
        p.add_argument('--'+name,type=Path)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--n',type=prior.positive,default=1024)
    p.add_argument('--data-seeds',type=prior.natural,nargs='+',default=list(DEFAULT_SEEDS))
    p.add_argument('--conditions',choices=CONDITIONS,nargs='+',default=list(CONDITIONS))
    p.add_argument('--models',choices=MODELS,nargs='+',default=list(MODELS))
    args=p.parse_args(argv)
    try:
        validate_seeds(args.data_seeds)
        if not 2<=args.n<=16384 or len(set(args.conditions))!=len(args.conditions) or len(set(args.models))!=len(args.models):
            raise ValueError('n must be 2..16384; condition/model lists must be unique')
        if args.worker_output:
            if not all((args.checkpoint,args.claim_calibration,args.previous_report)):
                raise ValueError('worker needs checkpoint, claim calibration and original report')
            if args.worker_output.exists() or args.worker_output.with_suffix('.npz').exists():
                raise ValueError('refusing existing worker files')
            write_json(args.worker_output,worker(args));return 0
        if not all((args.replication_dir,args.previous_dir,args.output_dir)):
            raise ValueError('supply replication-dir, previous-dir and output-dir')
        parent(args);return 0
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,FloatingPointError,subprocess.CalledProcessError) as error:
        print('Error:',error,file=sys.stderr);return 2


if __name__=='__main__':
    raise SystemExit(main())
