"""Predict unqueried edit responses with local mathematical interaction terms.

Raw named state only, to isolate response prediction from previously measured
codec error. Fitting samples 73 small perturbations of EACH input's paused state.
Both prescribed radii are reported; neither is selected using target answers.
No neural training, range refitting, new claims or speedup is implied.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math, os, random, subprocess, sys, time
import numpy as np
import torch

import benchmark_state_continuation as prior
from local_response import stencil, build_rule, query_plan, score
from universa_recurrent.neural.data import StructuredFlowDataset
from universa_recurrent.neural.dual_output import ClaimPolicy, construct_output
from universa_recurrent.neural.dual_study import load_estimators, read_json, write_json, dataset_fingerprint
from universa_recurrent.neural.retention_study import configure, compare_outputs
from universa_recurrent.neural.train import sha256_file

FORMAT='universa-recurrent.local-response.v1'
HORIZONS=('readout_at_cut','original_remaining_updates')
RADII=(.125,.0625)
ORDERS=('none','linear','diagonal_quadratic','full_quadratic')
FORBIDDEN={9000,9001,12000,12001,12100,22000,31000,32000,33000,36000,46000,61000,62000,71000,72000,83000,85000}
SCOPE={
 'question':'Can a local response rule forecast unqueried joint numerical changes, including interactions?',
 'state':'Six raw named dynamic fields; original context stays with the model used to construct the rule. Compression is deliberately excluded from this prediction test.',
 'construction':'One baseline, twelve single-axis probes and sixty mixed probes: 73 batched model evaluations per cut/horizon/radius. Every new input needs its own rule. No target response enters construction.',
 'prediction':'Rule coefficients and actual normalized edit displacement only; no model callback, target labels, or target response.',
 'controls':'No effect, linear, diagonal quadratic, full quadratic; two preset probe radii; immediate readout versus all original remaining updates; common-logit and zero edits.',
 'query':'33 fixed queries: five two-field families, four dense signed families and two controls, at three sizes. Targets are outside the probe radius, except the explicit zero control.',
 'cost':'Save coefficient bytes, raw stencil outputs, setup evaluation counts and one descriptive setup time. No speedup or compression claim. Query truth requires separate model calls for evaluation.',
 'limits':'Classical local finite-difference approximation; not a global surrogate, exact derivative, learned English, discovered latent semantics, or all-input certificate. Float32 probing and nonlinearity may limit accuracy.',
 'sampling':'One new matched cohort shared across five frozen fits and all conditions. Those repetitions are not independent problems.',
}


def identity():
    result=prior.source();root=Path(__file__).resolve().parent
    result['local_response_scripts']={n:sha256_file(root/n) for n in ('local_response.py','benchmark_local_response.py')}
    return result


def validate_args(args):
    if args.test_seed in FORBIDDEN or not 2<=args.n<=1024 or len(set(args.models))!=len(args.models):
        raise ValueError('fresh seed, distinct models, and size in [2,1024] required')


def displaced(original, scales, u):
    if original.dtype!=np.float32 or original.ndim!=2 or original.shape[1]!=6 or not np.isfinite(original).all():
        raise ValueError('complete finite float32 state required')
    scales=np.asarray(scales,dtype=float);u=np.asarray(u,dtype=float)
    if scales.shape!=(6,) or u.shape!=(6,) or not np.isfinite(scales).all() or not np.isfinite(u).all() or (scales<=0).any():
        raise ValueError('six finite positive scales and six offsets required')
    with np.errstate(over='raise',invalid='raise'):
        state=(original.astype(float)+scales*u).astype(np.float32)
    return state,(state.astype(float)-original.astype(float))/scales


def response(model,point,state,policy,horizon):
    if horizon not in HORIZONS:raise ValueError('unknown horizon')
    output,steps=prior.continue_from(model,point,state,policy,stop_at_cut=horizon=='readout_at_cut')
    if steps!=(0 if horizon=='readout_at_cut' else model.config.steps-point.cut):
        raise ValueError('changed remaining budget')
    return output['estimate'].detach().cpu().numpy().astype(float)


@torch.inference_mode()
def worker(args):
    validate_args(args);torch.set_num_threads(1)
    execution=configure(True);src=identity()
    paths=(args.checkpoint,args.claim_calibration,args.previous_report);hashes=tuple(map(sha256_file,paths))
    cp,cal,old_sha=hashes;old=read_json(args.previous_report);claims=read_json(args.claim_calibration)
    engines,meta,device,reserved=load_estimators(args.checkpoint,args.device)
    if (old.get('format')!=prior.FORMAT+'.worker' or old.get('all_restoration_gates_passed') is not True or
        old['checkpoint_sha256']!=cp or old['claim_calibration_sha256']!=cal or meta['checkpoint_sha256']!=cp or claims['checkpoint_sha256']!=cp or
        old['training_seed']!=meta['training']['seed']):raise ValueError('reference identity mismatch')
    excluded=set(reserved)|{claims['data']['seed'],old['calibration_seed'],old['test_seed'],old['training_seed']}
    if 'test_seed_reserved' in claims:excluded.add(claims['test_seed_reserved'])
    if args.test_seed in excluded:raise ValueError('test seed overlaps training or calibration')
    data=StructuredFlowDataset(args.n,seed=args.test_seed,**{k:meta['training'][k] for k in ('noise_std','observe_probability')})
    joined=np.ascontiguousarray(np.c_[data.observed.numpy(),data.mask.numpy()])
    if len(set(map(bytes,joined)))!=args.n:raise ValueError('non-distinct inputs')
    arrays={k:v.numpy() for k,v in dict(observed=data.observed,mask=data.mask,truth=data.truth,labels=data.label).items()}
    x,m=data.observed.to(device),data.mask.to(device);engines={e.name:e for e in engines}
    rows=[];fits=[];sites=[];gates=[];queries=query_plan()
    arrays['query_offsets']=np.stack([q[2] for q in queries])
    for name in args.models:
        engine=engines[name];model=engine.model.eval()
        if model.config.num_structures!=2 or model.config.latent_dim!=2:raise ValueError('pilot needs six state fields')
        if claims['model_sources'][name]!=cp:raise ValueError('wrong calibrated model')
        policy=ClaimPolicy(**claims['models'][name]['policy'])
        hist=model.rollout(x,m);full=construct_output(hist['states'][-1],hist['route_probabilities'][-1],policy,'mixture_with_claim');del hist
        for cut in sorted({model.config.steps//2,model.config.steps-1}):
            key=f'{name}__cut{cut}';point=prior.pause(model,x,m,cut);original=point.vector()
            raw,_=prior.continue_from(model,point,original,policy)
            gates.append(dict(model=name,cut=cut,checks=compare_outputs(raw,full)))
            base=old['codecs'][key+'__named_16bit']
            expected=dict(checkpoint_sha256=cp,model=name,cut=cut,depth=model.config.steps,num_structures=2,latent_dim=2,names=meta['names'],calibration_sha256=old['calibration_sha256'])
            if base['site']!=expected or base['kind']!='named' or base['bits']!=16 or base['calibration_n']!=old['calibration_n']:raise ValueError('wrong frozen scale source')
            scales=np.r_[np.asarray(base['ranges'][:4],float)*.125,[math.log(2),math.log(2)]]
            arrays[key+'__state']=original;arrays[key+'__scales']=scales
            sites.append(dict(model=name,cut=cut,names=meta['names'],frozen_manifest=base,
                retained_context_bytes_per_input=point.retained_bytes_per_example(),raw_dynamic_bytes_per_input=24))
            targets=[displaced(original,scales,q[2]) for q in queries]
            for qi,(state,actual_u) in enumerate(targets):
                arrays[key+f'__query{qi}__state']=state;arrays[key+f'__query{qi}__actual_u']=actual_u
            for horizon in HORIZONS:
                skey=key+'__'+horizon
                # All response rules and predictions are fixed before any target is evaluated.
                prepared=[]
                for radius in RADII:
                    rkey=skey+f'__radius{radius:g}';offsets=stencil(6,radius)
                    prior.sync(device);start=time.perf_counter_ns();probe_results=[]
                    try:
                        for u in offsets:probe_results.append(response(model,point,displaced(original,scales,u)[0],policy,horizon))
                        responses=np.stack(probe_results);rule=build_rule(offsets,responses)
                        predictions={order:np.stack([rule.predict(u,order=order) for _,u in targets]) for order in ORDERS}
                    except (prior.NumericalContinuationError,FloatingPointError) as error:
                        fits.append(dict(model=name,cut=cut,horizon=horizon,radius=radius,status='nonfinite_probe_or_prediction',reason=str(error),completed_probe_evaluations=len(probe_results)))
                        if probe_results:arrays[rkey+'__partial_probes']=np.stack(probe_results)
                        continue
                    prior.sync(device);setup_ms=(time.perf_counter_ns()-start)/1e6
                    arrays[rkey+'__offsets']=offsets;arrays[rkey+'__probes']=responses
                    for k,v in dict(base=rule.base,jacobian=rule.jacobian,hessian=rule.hessian).items():arrays[rkey+'__'+k]=v
                    for order,v in predictions.items():arrays[rkey+'__predictions__'+order]=v
                    fits.append(dict(model=name,cut=cut,horizon=horizon,radius=radius,status='finite',probe_evaluations=len(offsets),
                        probe_example_evaluations=len(offsets)*args.n,coefficient_bytes_per_input=rule.stored_bytes_per_input(),
                        setup_and_all_predictions_ms=setup_ms,timing_scope='one unoptimized descriptive time, includes all probes and 33 predictions; not a benchmark',
                        target_outcomes_seen_during_fit=False))
                    prepared.append((radius,rule,predictions))
                # Measured targets are evaluation only. Predictor does not receive them.
                baseline=response(model,point,original,policy,horizon);arrays[skey+'__baseline']=baseline
                for qi,((family,scale,u),(state,_)) in enumerate(zip(queries,targets)):
                    try:actual=response(model,point,state,policy,horizon)
                    except prior.NumericalContinuationError as error:
                        rows.append(dict(model=name,cut=cut,horizon=horizon,query_index=qi,status='nonfinite_target',reason=str(error)));continue
                    arrays[skey+f'__query{qi}__actual']=actual
                    for radius,rule,predictions in prepared:
                        if not np.array_equal(rule.base,baseline):raise ValueError('baseline changed while probing')
                        for order in ORDERS:
                            met=score(predictions[order][qi],actual,baseline)
                            rows.append(dict(model=name,cut=cut,horizon=horizon,query_index=qi,family=family,scale=scale,radius=radius,
                                order=order,status='finite',n=args.n,metrics=met))
                print(f'{name} cut {cut} / {horizon}: rules built; all held-out edit queries evaluated',flush=True)
            if point.vector().tobytes()!=original.tobytes():raise ValueError('pause state mutated')
    if tuple(map(sha256_file,paths))!=hashes or identity()!=src:raise ValueError('source/reference changed')
    npz=args.worker_output.with_suffix('.npz')
    with npz.open('xb') as f:np.savez_compressed(f,**arrays)
    return dict(format=FORMAT+'.worker',source=src,execution=execution,device=str(device),training_seed=meta['training']['seed'],
        checkpoint_sha256=cp,claim_calibration_sha256=cal,previous_report_sha256=old_sha,test_seed=args.test_seed,n=args.n,dataset_sha256=dataset_fingerprint(data),
        queries=[dict(index=i,family=f,scale=s,nominal_offset=u.tolist()) for i,(f,s,u) in enumerate(queries)],
        fits=fits,rows=rows,sites=sites,raw_gates=gates,all_raw_gates_passed=True,per_example_file=npz.name,per_example_sha256=sha256_file(npz),scope=SCOPE)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('replication-dir','previous-dir','output-dir','checkpoint','claim-calibration','previous-report','worker-output'):p.add_argument('--'+k,type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--models',nargs='+',choices=['shared','fixed_depth_4'],default=['shared','fixed_depth_4'])
    p.add_argument('--n',type=prior.positive,default=256);p.add_argument('--test-seed',type=prior.natural,default=87000)
    args=p.parse_args(argv)
    try:
        validate_args(args)
        if args.worker_output:
            if not all((args.checkpoint,args.claim_calibration,args.previous_report)):raise ValueError('missing worker references')
            if args.worker_output.exists() or args.worker_output.with_suffix('.npz').exists():raise ValueError('refusing overwrite')
            write_json(args.worker_output,worker(args));return 0
        if not all((args.replication_dir,args.previous_dir,args.output_dir)):raise ValueError('supply replication, original continuation, output folders')
        if args.output_dir.exists():raise ValueError('refusing overwrite')
        jobs=[]
        for cp in sorted(args.replication_dir.glob('weights-*.pt')):
            seed=int(cp.stem.removeprefix('weights-'));cal=args.replication_dir/f'study-{seed}/calibration.json';old=args.previous_dir/f'seed-{seed}.json'
            if not cal.is_file() or not old.is_file():raise ValueError('missing reference')
            jobs.append((seed,cp,cal,old,tuple(map(sha256_file,(cp,cal,old)))))
        if not jobs or len({j[-1][0] for j in jobs})!=len(jobs):raise ValueError('distinct checkpoints required')
        random.Random(89000).shuffle(jobs);src=identity();args.output_dir.mkdir(parents=True)
        write_json(args.output_dir/'PLAN.json',dict(format=FORMAT+'.plan',scope=SCOPE,source=src,n=args.n,test_seed=args.test_seed,models=args.models,radii=list(RADII),primary_radius=.125,secondary_radius=.0625,job_order=[j[0] for j in jobs]))
        reports=[];cohort=None
        for seed,cp,cal,old,hashes in jobs:
            out=args.output_dir/f'seed-{seed}.json'
            cmd=[sys.executable,str(Path(__file__).resolve()),'--checkpoint',str(cp.resolve()),'--claim-calibration',str(cal.resolve()),'--previous-report',str(old.resolve()),'--worker-output',str(out.resolve()),'--device',args.device,'--models',*args.models,'--n',str(args.n),'--test-seed',str(args.test_seed)]
            subprocess.run(cmd,check=True,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
            w=read_json(out)
            if w['source']!=src or w['training_seed']!=seed or tuple(map(sha256_file,(cp,cal,old)))!=hashes or (w['checkpoint_sha256'],w['claim_calibration_sha256'],w['previous_report_sha256'])!=hashes:raise ValueError('worker identity mismatch')
            if cohort is not None and w['dataset_sha256']!=cohort:raise ValueError('cohorts differ')
            cohort=w['dataset_sha256'];reports.append(w)
        result=dict(format=FORMAT+'.complete',checkpoint_runs=len(reports),n=args.n,test_seed=args.test_seed,source=src,dataset_sha256=cohort,
            worker_files=[f'seed-{w["training_seed"]}.json' for w in reports],all_raw_gates_passed=all(w['all_raw_gates_passed'] for w in reports),
            failed_fits=sum(f['status']!='finite' for w in reports for f in w['fits']),failed_target_cases=sum(r['status']!='finite' for w in reports for r in w['rows']),
            scope=SCOPE,note='Inspect radius sensitivity, held-out query errors and cost before selecting any response model. Failures remain visible.')
        write_json(args.output_dir/'summary.json',result);print('COMPLETE:',args.output_dir/'summary.json',flush=True);return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,FloatingPointError,subprocess.CalledProcessError) as e:
        print('Error:',e,file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
