"""Do two mathematical edits combine as the sum of their separate effects?

A spreadsheet can contain individually correct formulas whose combined change is
not additive. Here, known-field commands are applied at ONE paused state, without
an intervening update. Compare the actual joint response with a sum formed using
ONLY the two individual responses. This is a conditional interaction diagnostic,
not a learned interpreter, a free prediction, or a speed/compression benchmark.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import os
import random
import statistics
import subprocess
import sys

import numpy as np
import torch

import benchmark_state_continuation as prior
from continuation_codec import canonical
from mathematical_edits import Edit, render, direct_edit, apply_command
from overflow_state_codec import OverflowStateCodec
from universa_recurrent.neural.data import StructuredFlowDataset
from universa_recurrent.neural.dual_output import ClaimPolicy, construct_output
from universa_recurrent.neural.dual_study import load_estimators, read_json, write_json, dataset_fingerprint
from universa_recurrent.neural.retention_study import configure, compare_outputs
from universa_recurrent.neural.train import sha256_file

FORMAT = 'universa-recurrent.edit-composition.v1'
SCALES = (0.25, 0.5, 1.0)
HORIZONS = ('readout_at_cut', 'original_remaining_updates')
FORBIDDEN = {9000, 9001, 12000, 12001, 12100, 22000, 31000, 32000, 33000,
             36000, 46000, 61000, 62000, 71000, 72000, 83000}
SCOPE = {
    'question': 'Does y(A+B)-y(0) approximately equal [y(A)-y(0)]+[y(B)-y(0)]?',
    'oracle_cost': 'The additive prediction requires a baseline and TWO single-edit evaluations on this input. No joint outcome enters that prediction. It is not a zero-cost or learned predictor.',
    'edits': 'Two typed additions at the SAME cut, with no intervening recurrence or requantization. Old calibration scales times 0.25, 0.5 or 1; named sixteen-bit overflow-safe codec.',
    'horizons': 'An immediate readout control executes zero updates; the continuation path executes ALL original remaining updates. They are labeled separately.',
    'separation': 'Nonadditivity within each path and codec distortion of the joint effect are separate measurements. A nonlinear readout can already cause interaction without further recurrence.',
    'controls': 'Raw restoration; raw numeric/parsed pair equality; individual A/B/no-edit effects; inverse-coordinate and common-logit controls; immediate readout versus continuation.',
    'sampling': 'One new common matched-distribution cohort. Checkpoints, cuts, pairs, scales and horizons reuse the same underlying inputs. No confirmatory significance claim.',
    'frozen': 'No weights, ranges, thresholds, reference files or package version are changed. Parser sees only state, schema and commands; resumed solver retains original input/context.',
    'bytes': 'Base state framing, shared codec and retained context are reported. Both command strings count additionally. No full-memory or performance result is claimed.',
    'limits': 'Hand-specified field semantics, not discovered neuron meanings or natural language. Additivity is a hypothesis, not a prerequisite for a working interpreter. No correct-route or authentication guarantee.',
}


def source():
    result = prior.source()
    here = Path(__file__).resolve().parent
    result['composition_scripts'] = {n: sha256_file(here / n) for n in
        ('benchmark_edit_composition.py', 'mathematical_edits.py', 'overflow_state_codec.py')}
    return result


def pair_plan(base, scale):
    """Fixed pairs cover within/across candidates and two cancellation controls."""
    if (type(scale) not in (float, int) or scale not in SCALES or
        base['kind'] != 'named' or base['bits'] != 16 or
        base['site']['num_structures'] != 2 or base['site']['latent_dim'] != 2):
        raise ValueError('pilot requires named 16-bit, two candidates/two coordinates and a fixed scale')
    def c(k, j, sign=1):
        return Edit('cycle', k, j, sign * scale * .125 * base['ranges'][2*k+j])
    def e(k):
        return Edit('evidence', k, None, scale * math.log(2))
    return [
        ('within_candidate_0', c(0, 0), c(0, 1)),
        ('within_candidate_1', c(1, 0), c(1, 1)),
        ('across_candidates', c(0, 0), c(1, 0, -1)),
        ('cycle_and_evidence_0', c(0, 0), e(0)),
        ('cycle_and_evidence_1', c(1, 0), e(1)),
        ('common_evidence_control', e(0), e(1)),
        ('inverse_cycle_control', c(0, 0), c(0, 0, -1)),
    ]


def pair_states(vector, first, second, names, d, *, parsed):
    """Every branch starts at the same base. B-alone is NOT applied after A."""
    if parsed:
        a = apply_command(vector, render(first, names, d), names, d)
        b = apply_command(vector, render(second, names, d), names, d)
        both = apply_command(a, render(second, names, d), names, d)
    else:
        a = direct_edit(vector, first, names, d)
        b = direct_edit(vector, second, names, d)
        both = direct_edit(a, second, names, d)
    return {'a': a, 'b': b, 'joint': both}


def additive_prediction(base, a, b):
    """Does not accept or inspect the actual joint response."""
    y0, ya, yb = (np.asarray(v, dtype=np.float64) for v in (base, a, b))
    if (y0.ndim != 2 or not y0.size or ya.shape != y0.shape or yb.shape != y0.shape
        or not all(np.isfinite(v).all() for v in (y0, ya, yb))):
        raise ValueError('finite equal-shaped response arrays required')
    return y0 + (ya-y0) + (yb-y0)


def interaction_metrics(base, a, b, joint):
    pred = additive_prediction(base, a, b)
    y0 = np.asarray(base, dtype=np.float64)
    ya, yb, yab = (np.asarray(v, dtype=np.float64) for v in (a, b, joint))
    if yab.shape != y0.shape or not np.isfinite(yab).all():
        raise ValueError('invalid joint response')
    delta, err = yab-y0, yab-pred
    energy, error = float(np.mean(delta**2)), float(np.mean(err**2))
    # Relative errors near a cancellation are undefined, not artificially zero.
    return {
        'joint_effect_mse': energy, 'additive_error_mse': error,
        'relative_rms_interaction': math.sqrt(error/energy) if energy > 1e-12 else None,
        'relative_metric_floor_mse': 1e-12,
        'worst_absolute_interaction': float(np.abs(err).max()),
        'p99_max_component_interaction': float(np.quantile(np.abs(err).max(1), .99)),
        'no_effect_prediction_mse': energy,
        'a_only_prediction_mse': float(np.mean((yab-ya)**2)),
        'b_only_prediction_mse': float(np.mean((yab-yb)**2)),
    }


def save_output(arrays, key, output):
    for name in ('estimate', 'probabilities', 'claim_index'):
        arrays[key+'__'+name] = output[name].detach().cpu().numpy().copy()


def codec_effect(raw, raw_base, decoded, decoded_base):
    def arr(v): return v['estimate'].detach().cpu().numpy().astype(float)
    yr, y0, yd, d0 = map(arr, (raw, raw_base, decoded, decoded_base))
    error = (yd-d0)-(yr-y0)
    return {
        'joint_effect_disagreement_mse': float(np.mean(error**2)),
        'final_mse_to_raw_joint': float(np.mean((yd-yr)**2)),
        'worst_component_difference': float(np.abs(yd-yr).max()),
        'changed_claims_vs_raw_joint': int((decoded['claim_index'].cpu()!=raw['claim_index'].cpu()).sum()),
    }


def cut_points(depth):
    if type(depth) is not int or depth < 2:
        raise ValueError('at least two configured updates required')
    return sorted({depth//2, depth-1})


def continue_path(model, point, state, policy, horizon):
    if horizon not in HORIZONS:
        raise ValueError('unknown horizon')
    result, updates = prior.continue_from(model, point, state, policy,
                                         stop_at_cut=horizon=='readout_at_cut')
    expected = 0 if horizon=='readout_at_cut' else model.config.steps-point.cut
    if updates != expected:
        raise ValueError('changed update budget')
    return prior.as_cpu(result)


@torch.inference_mode()
def worker(args):
    if args.test_seed in FORBIDDEN or not 2 <= args.n <= 4096:
        raise ValueError('new data seed and size in [2,4096] required')
    torch.set_num_threads(1)
    execution, initial_source = configure(True), source()
    paths = (args.checkpoint, args.claim_calibration, args.previous_report)
    hashes = tuple(sha256_file(p) for p in paths)
    cp_sha, cal_sha, old_sha = hashes
    old, claims = read_json(args.previous_report), read_json(args.claim_calibration)
    engines, meta, device, reserved = load_estimators(args.checkpoint, args.device)
    if (old.get('format') != prior.FORMAT+'.worker' or not old.get('all_restoration_gates_passed')
        or old['checkpoint_sha256'] != cp_sha or old['claim_calibration_sha256'] != cal_sha
        or meta['checkpoint_sha256'] != cp_sha or claims['checkpoint_sha256'] != cp_sha
        or old['training_seed'] != meta['training']['seed']):
        raise ValueError('frozen reference identity mismatch')
    excluded = set(reserved) | {claims['data']['seed'], old['calibration_seed'], old['test_seed'], old['training_seed']}
    if 'test_seed_reserved' in claims: excluded.add(claims['test_seed_reserved'])
    if args.test_seed in excluded:
        raise ValueError('input seed overlaps training/calibration')
    data = StructuredFlowDataset(args.n, seed=args.test_seed, **{k: meta['training'][k]
                                 for k in ('noise_std', 'observe_probability')})
    joined = np.concatenate((data.observed.numpy(), data.mask.numpy()), axis=1)
    if len({r.tobytes() for r in joined}) != args.n:
        raise ValueError('duplicate inputs')
    arrays = {k: v.numpy() for k,v in {'observed':data.observed, 'mask':data.mask,
               'truth':data.truth, 'labels':data.label}.items()}
    x, m = data.observed.to(device), data.mask.to(device)
    engines = {e.name:e for e in engines}
    rows, sites, gates = [], [], []
    for name in args.models:
        model = engines[name].model.eval()
        if claims['model_sources'][name] != cp_sha:
            raise ValueError('claim source mismatch')
        policy = ClaimPolicy(**claims['models'][name]['policy'])
        names, d = meta['names'], model.config.latent_dim
        history = model.rollout(x, m)
        full = prior.as_cpu(construct_output(history['states'][-1], history['route_probabilities'][-1],
                                             policy, 'mixture_with_claim'))
        del history
        for cut in cut_points(model.config.steps):
            sitekey = f'{name}__cut{cut}'
            point = prior.pause(model,x,m,cut); original = point.vector()
            raw = continue_path(model,point,original,policy,'original_remaining_updates')
            gates.append({'model':name,'cut':cut,'checks':compare_outputs(raw,full)})
            frozen = old['codecs'][sitekey+'__named_16bit']
            expected_site = dict(checkpoint_sha256=cp_sha,model=name,cut=cut,depth=model.config.steps,
                num_structures=model.config.num_structures,latent_dim=d,names=names,
                calibration_sha256=old['calibration_sha256'])
            if (frozen['site'] != expected_site or frozen['bits'] != 16 or frozen['kind'] != 'named'
                or frozen['calibration_n'] != old['calibration_n']):
                raise ValueError('frozen codec site mismatch')
            codec = OverflowStateCodec.from_manifest(frozen)
            payloads, fallback = codec.encode(original); decoded = codec.decode(payloads)
            arrays.update({sitekey+'__original_dynamic':original,sitekey+'__decoded_dynamic':decoded,
                           sitekey+'__fallback_mask':fallback})
            sites.append(dict(model=name,cut=cut,codec=codec.manifest(),storage=codec.storage(payloads),
                              retained_context_bytes_per_input=point.retained_bytes_per_example()))
            bases = {}
            for horizon in HORIZONS:
                for path,state in [('raw',original),('decoded',decoded)]:
                    base = continue_path(model,point,state,policy,horizon)
                    bases[(horizon,path)] = base
                    save_output(arrays,sitekey+'__'+horizon+'__'+path+'__baseline',base)
            for scale in args.scales:
                for family,first,second in pair_plan(frozen,scale):
                    casekey = sitekey+f'__{family}__scale{scale:g}'
                    commands = [render(e,names,d) for e in (first,second)]
                    raw_states = pair_states(original,first,second,names,d,parsed=False)
                    parsed_raw = pair_states(original,first,second,names,d,parsed=True)
                    for which in raw_states:
                        if raw_states[which].tobytes()!=parsed_raw[which].tobytes():
                            raise ValueError('raw parser/numeric composition mismatch')
                    dec_states = pair_states(decoded,first,second,names,d,parsed=True)
                    for path,states in [('raw',raw_states),('decoded',dec_states)]:
                        arrays[casekey+'__'+path+'__joint_state'] = states['joint']
                    for horizon in HORIZONS:
                        outputs = {}; statuses = {}
                        for path,states in [('raw',raw_states),('decoded',dec_states)]:
                            current = {}
                            try:
                                for which,state in states.items():
                                    current[which] = continue_path(model,point,state,policy,horizon)
                                    save_output(arrays,casekey+'__'+horizon+'__'+path+'__'+which,current[which])
                            except prior.NumericalContinuationError as error:
                                statuses[path] = dict(status='nonfinite_continuation',failure=str(error),metrics=None)
                                continue
                            outputs[path] = current
                            baseline = bases[(horizon,path)]
                            response = [v['estimate'].numpy() for v in [baseline,current['a'],current['b'],current['joint']]]
                            met = interaction_metrics(*response)
                            met['changed_claims_vs_own_baseline'] = int((current['joint']['claim_index']!=baseline['claim_index']).sum())
                            statuses[path] = dict(status='finite',metrics=met)
                        comparison = None
                        if set(outputs)=={'raw','decoded'}:
                            comparison = codec_effect(outputs['raw']['joint'],bases[(horizon,'raw')],
                                                       outputs['decoded']['joint'],bases[(horizon,'decoded')])
                            if fallback.any():
                                for which in ('a','b','joint'):
                                    compare_outputs({k:v[fallback] for k,v in outputs['decoded'][which].items()},
                                                    {k:v[fallback] for k,v in outputs['raw'][which].items()})
                        rows.append(dict(model=name,cut=cut,family=family,scale=scale,horizon=horizon,n=args.n,
                            commands=commands,command_json_bytes=len(canonical(commands)),
                            executed_updates_per_evaluation=0 if horizon=='readout_at_cut' else model.config.steps-cut,
                            paths=statuses,codec_comparison=comparison,
                            prediction_uses_joint_outcome=False))
            if original.tobytes()!=point.vector().tobytes():
                raise ValueError('original state mutated')
            print(f'{name} cut {cut}: separate/joint edits and both horizons recorded',flush=True)
    if hashes != tuple(sha256_file(p) for p in paths) or source()!=initial_source:
        raise ValueError('source/reference changed')
    array_path = args.worker_output.with_suffix('.npz')
    with array_path.open('xb') as handle: np.savez_compressed(handle,**arrays)
    return dict(format=FORMAT+'.worker',source=initial_source,execution=execution,device=str(device),
        checkpoint_sha256=cp_sha,claim_calibration_sha256=cal_sha,previous_report_sha256=old_sha,
        training_seed=meta['training']['seed'],test_seed=args.test_seed,n=args.n,
        dataset_sha256=dataset_fingerprint(data),rows=rows,sites=sites,raw_gates=gates,
        all_raw_gates_passed=True,no_training=True,no_refitting=True,scope=SCOPE,
        per_example_file=array_path.name,per_example_sha256=sha256_file(array_path))


def parent(args):
    if args.output_dir.exists(): raise ValueError('refusing existing output directory')
    jobs=[]
    for cp in sorted(args.replication_dir.glob('weights-*.pt')):
        seed=int(cp.stem.removeprefix('weights-'))
        cal=args.replication_dir/f'study-{seed}/calibration.json';old=args.previous_dir/f'seed-{seed}.json'
        if not cal.is_file() or not old.is_file(): raise ValueError(f'missing references for {seed}')
        jobs.append((seed,cp,cal,old,tuple(sha256_file(p) for p in (cp,cal,old))))
    if not jobs or len({j[4][0] for j in jobs})!=len(jobs): raise ValueError('distinct checkpoints required')
    random.Random(86000).shuffle(jobs);initial=source();args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'PLAN.json',dict(format=FORMAT+'.plan',source=initial,scope=SCOPE,n=args.n,
        test_seed=args.test_seed,models=args.models,scales=args.scales,job_order=[j[0] for j in jobs]))
    workers=[]
    for seed,cp,cal,old,hashes in jobs:
        out=args.output_dir/f'seed-{seed}.json'
        cmd=[sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(cp.resolve()),
             '--claim-calibration',str(cal.resolve()),'--previous-report',str(old.resolve()),
             '--worker-output',str(out.resolve()),'--device',args.device,'--n',str(args.n),
             '--test-seed',str(args.test_seed),'--models',*args.models,'--scales',*map(str,args.scales)]
        subprocess.run(cmd,check=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        w=read_json(out)
        if (w['source']!=initial or w['training_seed']!=seed or
            tuple(w[k] for k in ('checkpoint_sha256','claim_calibration_sha256','previous_report_sha256'))!=hashes
            or tuple(sha256_file(p) for p in (cp,cal,old))!=hashes):
            raise ValueError('worker/reference identity mismatch')
        if workers and w['dataset_sha256']!=workers[0]['dataset_sha256']: raise ValueError('cohort mismatch')
        workers.append(w)
    result=dict(format=FORMAT+'.complete',source=initial,scope=SCOPE,checkpoint_runs=len(workers),
        worker_files=[f'seed-{w["training_seed"]}.json' for w in workers],n=args.n,test_seed=args.test_seed,
        dataset_sha256=workers[0]['dataset_sha256'],all_raw_gates_passed=True,
        case_rows=sum(len(w['rows']) for w in workers),
        failed_path_cases=sum(p['status']!='finite' for w in workers for r in w['rows'] for p in r['paths'].values()),
        note='Nonadditivity is a result, not a failed implementation. Do not pool repetitions as independent inputs.')
    write_json(args.output_dir/'summary.json',result)
    print('COMPLETE:',args.output_dir/'summary.json',flush=True)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('replication-dir','previous-dir','output-dir','checkpoint','claim-calibration','previous-report','worker-output'):
        p.add_argument('--'+k,type=Path)
    p.add_argument('--models',nargs='+',choices=['shared','fixed_depth_4'],default=['shared','fixed_depth_4'])
    p.add_argument('--scales',nargs='+',type=float,choices=SCALES,default=list(SCALES))
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--n',type=prior.positive,default=256)
    p.add_argument('--test-seed',type=prior.natural,default=85000)
    args=p.parse_args(argv)
    try:
        if args.test_seed in FORBIDDEN or not 2<=args.n<=4096 or len(set(args.models))!=len(args.models) or len(set(args.scales))!=len(args.scales):
            raise ValueError('invalid size, duplicate cases or previously used seed')
        if args.worker_output is not None:
            if not all((args.checkpoint,args.claim_calibration,args.previous_report)): raise ValueError('worker needs references')
            if args.worker_output.exists() or args.worker_output.with_suffix('.npz').exists(): raise ValueError('refusing overwrite')
            write_json(args.worker_output,worker(args))
        else:
            if not all((args.replication_dir,args.previous_dir,args.output_dir)): raise ValueError('supply directories')
            parent(args)
        return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,FloatingPointError,subprocess.CalledProcessError) as error:
        print('Error:',error,file=sys.stderr);return 2


if __name__=='__main__': raise SystemExit(main())
